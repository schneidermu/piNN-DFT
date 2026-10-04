"""Run the one-stage variational Lap CLI, with explicit four-task PCD mode.

The CLI consumes an external hash-bound Minnesota group store and AO-factor
cache. PySCF is not imported; it is only needed to create the cache upstream.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lap_checkpoint import load_lap_checkpoint
from lap_moo_optimizers import build_optimizer
from lap_moo_panel import MinnesotaGroupStore
from lap_moo_protocol import (
    FOUR_TASK_NAMES,
    METHOD_IDS,
    OPERATOR_PRECISION_SOURCE_PATHS,
    OPTIMIZER_FAMILIES,
    PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
    PCD_QP_TOLERANCE,
    PCD_SOURCE_FILE_PATHS,
    TASK_NAMES,
    SamplingStream,
    build_sampling_manifest_from_catalog,
    file_sha256,
    make_protocol_metadata,
    read_sampling_manifest,
    validate_protocol_metadata,
    write_sampling_manifest,
)
from lap_moo_training import (
    AOFactorChunk,
    ChemistryBatchObjective,
    MRKSOperatorSystem,
    compute_isolated_task_gradients,
    load_moo_checkpoint,
    make_cosine_scheduler,
    make_mrks_objective_factories,
    make_three_objective_factories,
    materialize_task_zeros,
    named_trainable_parameters,
    save_moo_checkpoint,
    train_moo_update,
)
from lap_operator_data import (
    OPERATOR_PROTOCOL,
    load_central_operator_record,
    verify_operator_corpus,
)
from optuna_joint import DEFAULT_MRKS_DISPERSIONS, load_mrks_dispersions
from train_lap import DEFAULT_REACTION_DISPERSIONS, load_reaction_dispersions

AO_CACHE_PROTOCOL = "lap-ao-factor-cache-f32-v1"


def _external_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must stay outside the repository: {resolved}")
    return resolved


def _current_pcd_source_identity() -> tuple[str, dict[str, str]]:
    """Return run-level Git revision and exact critical PCD source hashes."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in PCD_SOURCE_FILE_PATHS
    }


def _initial_aggregator_state(method: str) -> dict[str, Any] | None:
    """Keep fresh PCD initialization explicit without changing legacy state."""
    return None if method == "pcd" else {}


def _run_scheduled_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    objective_factories,
    *,
    method: str,
    hyperparameters: dict[str, Any],
    aggregator_state: dict[str, Any] | None,
    scheduler,
    world_size: int,
    step_rule: str = "optimizer",
    vector_armijo: dict[str, Any] | None = None,
    task_order: tuple[str, ...] = TASK_NAMES,
):
    """Apply one CLI update through the declared optimizer or PCD step rule."""
    return train_moo_update(
        model,
        optimizer,
        objective_factories,
        method=method,
        hyperparameters=hyperparameters,
        aggregator_state=aggregator_state,
        scheduler=scheduler,
        world_size=world_size,
        record_update_geometry=True,
        step_rule=step_rule,
        vector_armijo=vector_armijo,
        task_order=task_order,
    )


def _vector_armijo_rejection_record(
    *, update_index: int, rank_samples: list[dict[str, Any]], result: Any
) -> dict[str, Any]:
    if result.accepted:
        raise ValueError("A vector-Armijo stop record requires a rejected update.")
    return {
        "event": "vector_armijo_search_stopped",
        "update": update_index,
        "rank_samples": rank_samples,
        "losses": result.losses,
        "gradient_diagnostics": result.diagnostics,
        "step_diagnostics": result.diagnostics.get("vector_armijo"),
        "stop_reason": result.stop_reason,
        "sampling_cursor_next_update": update_index,
    }


def _capture_nash_aggregation_failure(
    output_dir: Path,
    *,
    update_index: int,
    sample: dict[str, Any],
    exception: BaseException,
    protocol: dict[str, Any],
    sampling_manifest: dict[str, Any],
    model: torch.nn.Module,
    objective_factories,
    device: torch.device,
) -> dict[str, Any]:
    """Persist the small raw-gradient fixture for a failed Nash solve."""
    rng_state = torch.get_rng_state().cpu().numpy().tobytes()
    if device.type == "cuda":
        rng_state += torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    rng_sha256 = hashlib.sha256(rng_state).hexdigest()
    losses, sparse = compute_isolated_task_gradients(model, objective_factories)
    dense = materialize_task_zeros(model, sparse)
    parameter_names = tuple(sorted(named_trainable_parameters(model)))
    vectors = {
        task: torch.cat(
            [
                dense[task][name]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .reshape(-1)
                for name in parameter_names
            ]
        ).numpy()
        for task in ("chem", "exc", "op")
    }
    matrix = np.stack([vectors[task] for task in ("chem", "exc", "op")])
    gram = matrix @ matrix.T
    checkpoint = output_dir / "latest.pt"
    raw_path = output_dir / f"nash_failed_attempt_{update_index + 1:04d}_raw_gradients.npz"
    np.savez_compressed(
        raw_path,
        parameter_names=np.asarray(parameter_names, dtype=np.str_),
        chem=matrix[0],
        exc=matrix[1],
        op=matrix[2],
    )
    report = {
        "schema": "lap-moo-nash-failed-attempt-v1",
        "method": "nash_mtl",
        "attempted_update_index_zero_based": update_index,
        "attempted_update_number_one_based": update_index + 1,
        "sampling_cursor_of_latest_checkpoint": update_index,
        "sample": sample,
        "exception": f"{type(exception).__name__}: {exception}",
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint.exists() else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else None,
        "protocol_metadata_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sampling_manifest_sha256": sampling_manifest["manifest_sha256"],
        "rng_state_sha256_at_failure_before_diagnostic_replay": rng_sha256,
        "task_losses": losses,
        "task_gradient_norms": {
            task: float(np.linalg.norm(vectors[task]))
            for task in ("chem", "exc", "op")
        },
        "raw_gradient_gram": gram.tolist(),
        "raw_gradient_artifact_path": str(raw_path.resolve()),
        "raw_gradient_artifact_sha256": file_sha256(raw_path),
        "parameter_count": int(matrix.shape[1]),
        "protocol": protocol,
    }
    report_path = output_dir / f"nash_failed_attempt_{update_index + 1:04d}_diagnostic.json"
    _write_json(report_path, report)
    return report


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        return data.item() if data.numel() == 1 else data.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _distributed_runtime(requested_device: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device(
            f"cuda:{local_rank}"
            if world_size > 1
            else (requested_device if ":" in requested_device else "cuda:0")
        )
        torch.cuda.set_device(device.index if device.index is not None else 0)
        backend = "nccl"
    else:
        device = torch.device(requested_device)
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return device, rank, local_rank, world_size


def _seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CentralAOCache:
    """Hash-bound, one-system-at-a-time central feature/AO-factor reader."""

    def __init__(
        self,
        central_data_dir: Path,
        ao_cache_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size: int,
    ):
        import h5py

        del h5py
        if chunk_size <= 0:
            raise ValueError("AO cache chunk size must be positive.")
        self.central_data_dir = central_data_dir
        self.ao_cache_dir = ao_cache_dir
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.central_manifest_path = central_data_dir / "manifest.json"
        self.central_manifest = _read_json(self.central_manifest_path)
        system_names = self.central_manifest.get("built_systems")
        if not isinstance(system_names, list) or len(system_names) != 90:
            raise ValueError("The central operator source must prove the audited 90-system corpus.")
        self.verification = verify_operator_corpus(
            central_data_dir,
            expected_systems=system_names,
            require_all90=True,
        )
        self.central_manifest_sha256 = file_sha256(self.central_manifest_path)
        self.central_records = {
            row["system_name"]: row for row in self.central_manifest["records"]
        }
        self.cache_manifest_path = ao_cache_dir / "manifest.json"
        self.cache_manifest = _read_json(self.cache_manifest_path)
        if (
            self.cache_manifest.get("protocol") != AO_CACHE_PROTOCOL
            or self.cache_manifest.get("operator_protocol") != OPERATOR_PROTOCOL
            or self.cache_manifest.get("cache_dtype") != "float32"
            or self.cache_manifest.get("central_data_manifest_sha256")
            != self.central_manifest_sha256
        ):
            raise ValueError("AO-factor cache is incompatible or bound to different central data.")
        cache_rows = self.cache_manifest.get("records")
        if not isinstance(cache_rows, list) or not cache_rows:
            raise ValueError("AO-factor cache manifest has no system records.")
        self.cache_records = {row["system_name"]: row for row in cache_rows}
        if len(self.cache_records) != len(cache_rows):
            raise ValueError("AO-factor cache contains duplicate system identities.")
        if not set(self.cache_records) <= set(self.central_records):
            raise ValueError("AO cache includes a system outside the central corpus.")
        self.system_names = sorted(self.cache_records)
        self.cache_manifest_sha256 = file_sha256(self.cache_manifest_path)
        self._verified_cache_files: set[str] = set()
        self._current: MRKSOperatorSystem | None = None
        self.last_load_seconds = 0.0
        self.last_system_cache_bytes = 0

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest record path escapes its source directory.")
        target = (root / path).resolve()
        if root.resolve() not in target.parents:
            raise ValueError("Manifest record path escapes its source directory.")
        return target

    def load(self, system_name: str) -> MRKSOperatorSystem:
        import h5py

        if self._current is not None and self._current.name == system_name:
            self.last_load_seconds = 0.0
            return self._current
        started = time.perf_counter()
        if system_name not in self.cache_records:
            raise KeyError(f"AO cache has no factors for {system_name}.")
        central_row = self.central_records[system_name]
        cache_row = self.cache_records[system_name]
        record_path = self._safe_child(self.central_data_dir, central_row["file"])
        record_digest = file_sha256(record_path)
        if record_digest != central_row["file_sha256"]:
            raise ValueError(f"Central record hash mismatch for {system_name}.")
        record = load_central_operator_record(record_path)
        if record.metadata["record_sha256"] != central_row["record_sha256"]:
            raise ValueError(f"Central logical record hash mismatch for {system_name}.")
        if (
            cache_row.get("source_record_sha256") != record.metadata["record_sha256"]
            or cache_row.get("source_file_sha256") != central_row["file_sha256"]
        ):
            raise ValueError(f"AO factors for {system_name} do not match the central record.")
        cache_path = self._safe_child(self.ao_cache_dir, cache_row["file"])
        if system_name not in self._verified_cache_files:
            if file_sha256(cache_path) != cache_row.get("cache_file_sha256"):
                raise ValueError(f"AO cache file hash mismatch for {system_name}.")
            self._verified_cache_files.add(system_name)
        ngrid = len(record.weights)
        nao = record.dmks.shape[0]
        features = torch.as_tensor(
            np.array(record.DensityDescriptorsN10, copy=True),
            device=self.device,
            dtype=self.dtype,
        )
        weights = torch.as_tensor(
            np.array(record.weights, copy=True), device=self.device, dtype=self.dtype
        )
        ao_chunks = []
        with h5py.File(cache_path, "r") as handle:
            if (
                handle.attrs.get("protocol") != AO_CACHE_PROTOCOL
                or handle.attrs.get("operator_protocol") != OPERATOR_PROTOCOL
                or handle.attrs.get("system_name") != system_name
                or handle.attrs.get("source_record_sha256") != record.metadata["record_sha256"]
                or handle.attrs.get("dtype") != "float32"
            ):
                raise ValueError(f"AO cache metadata mismatch for {system_name}.")
            expected_shapes = {
                "phi": (ngrid, nao),
                "grad_phi": (ngrid, 3, nao),
                "lap_phi": (ngrid, nao),
            }
            for key, shape in expected_shapes.items():
                if key not in handle or handle[key].shape != shape or handle[key].dtype != np.dtype("<f4"):
                    raise ValueError(f"AO cache {key} shape or dtype mismatch for {system_name}.")
            for start in range(0, ngrid, self.chunk_size):
                stop = min(start + self.chunk_size, ngrid)
                rows = slice(start, stop)
                factors = {
                    key: torch.as_tensor(
                        np.asarray(handle[key][rows]), device=self.device, dtype=self.dtype
                    )
                    for key in expected_shapes
                }
                ao_chunks.append(
                    AOFactorChunk(
                        rows,
                        factors["phi"],
                        factors["grad_phi"],
                        factors["lap_phi"],
                    )
                )
        reference = torch.as_tensor(
            np.array(record.RefAO, copy=True), device=self.device, dtype=torch.float64
        )
        overlap = torch.as_tensor(
            np.array(record.Overlap, copy=True), device=self.device, dtype=torch.float64
        )
        target = torch.as_tensor(float(record.Exc), device=self.device, dtype=torch.float64)
        system = MRKSOperatorSystem(
            name=system_name,
            features=features,
            weights=weights,
            exc_target=target,
            reference_operator=reference,
            overlap=overlap,
            ao_chunks=tuple(ao_chunks),
        )
        self.last_load_seconds = time.perf_counter() - started
        self.last_system_cache_bytes = sum(
            factor.numel() * factor.element_size()
            for chunk in ao_chunks
            for factor in (chunk.phi, chunk.grad_phi, chunk.lap_phi)
        )
        self._current = system
        return system


def _method_configuration(method: str, hyperparameters_path: Path | None, calibration_path: Path | None):
    if method not in METHOD_IDS:
        raise ValueError(f"Unsupported MOO method {method!r}.")
    provided = {} if hyperparameters_path is None else _read_json(hyperparameters_path)
    fixed_record = None
    if method == "fixed":
        if calibration_path is None:
            raise ValueError("Fixed scalarization needs the frozen calibration report.")
        calibration = _read_json(calibration_path)
        weights = calibration.get("fixed_weights", calibration.get("weights"))
        if not isinstance(weights, list) or len(weights) != 3:
            raise ValueError("Calibration report must contain three frozen fixed_weights.")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in weights):
            raise ValueError("Fixed scalarization weights must be finite and positive.")
        hparams = {**provided, "fixed_weights": [float(value) for value in weights]}
        fixed_record = {
            "fixed_weights": hparams["fixed_weights"],
            "calibration_report_sha256": file_sha256(calibration_path),
            "derivation": calibration.get("derivation", "see hash-bound calibration report"),
        }
    elif method == "imtl_g":
        hparams = provided
    elif method == "cagrad":
        hparams = {"c": 0.4, "rescale": "paper_unscaled", **provided}
    elif method == "nash_mtl":
        hparams = {
            "solver": "newton_potential",
            "max_iter": 100,
            "tol": 1e-10,
            "update_every": 1,
            **provided,
        }
    else:
        allowed = {"tau", "beta", "eps", "qp_tolerance"}
        unknown = set(provided) - allowed
        if unknown:
            raise ValueError(f"Unsupported PCD method hyperparameters: {sorted(unknown)}.")
        hparams = {
            "tau": 0.02,
            "beta": 0.999,
            "eps": 1.0e-8,
            "qp_tolerance": PCD_QP_TOLERANCE,
            **provided,
        }
    return hparams, fixed_record


def _load_sampling_manifest(
    path: Path,
    *,
    catalog: list[dict[str, Any]],
    system_names: list[str],
    source_hashes: dict[str, str],
    updates: int,
    seed: int,
    world_size: int,
    rank: int,
    four_task: bool = False,
) -> dict[str, Any]:
    if four_task:
        # A v2 manifest is the immutable batch plan; never resample its tasks.
        found = read_sampling_manifest(path)
        if (found["schema"] != "lap-moo-sampling-manifest-v2"
                or found.get("task_order") != list(FOUR_TASK_NAMES)
                or len(found["entries"]) != updates
                or found.get("seed") != seed or found.get("world_size") != world_size
                or found.get("mrks_systems") != sorted(system_names)
                or found.get("source_hashes") != source_hashes):
            raise ValueError("Four-task manifest differs from requested catalog/source/seed/world/update settings.")
        source_catalog = {(r["database"], r["reaction_id"]): set(r["variants"])
                          for r in catalog}
        locked_catalog = {(r["database"], r["reaction_id"]): set(r["variants"])
                          for r in found["reaction_catalog"]}
        if (len(locked_catalog) != len(found["reaction_catalog"])
                or locked_catalog.keys() != source_catalog.keys()
                or any(not v or not v <= source_catalog[k] for k, v in locked_catalog.items())):
            raise ValueError("Four-task locked reaction variants differ from the source catalog.")
        return found
    expected = build_sampling_manifest_from_catalog(
        catalog,
        system_names,
        updates=updates,
        seed=seed,
        source_hashes=source_hashes,
        world_size=world_size,
    )
    if rank == 0:
        if path.exists():
            found = read_sampling_manifest(path)
            if found["schema"] != "lap-moo-sampling-manifest-v1":
                raise ValueError("Three-task mode requires a v1 sampling manifest.")
            if found["manifest_sha256"] != expected["manifest_sha256"]:
                raise ValueError("Existing sampling manifest does not match requested data/seed/update settings.")
        else:
            write_sampling_manifest(path, expected)
    if world_size > 1:
        dist.barrier()
    found = read_sampling_manifest(path)
    if found["manifest_sha256"] != expected["manifest_sha256"]:
        raise ValueError("Rank-local sampling manifest differs from the requested stream.")
    return found


def _catalog_and_systems_for_panel(
    panel_path: Path | None,
    group_store: MinnesotaGroupStore,
    systems: CentralAOCache,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    if panel_path is None:
        catalog = [
            {
                "database": row["database"],
                "reaction_id": row["reaction_id"],
                "variants": list(row["variant_suffixes"]),
            }
            for row in group_store.manifest["groups"]
        ]
        return catalog, systems.system_names, None

    panel = _read_json(panel_path)
    if panel.get("schema") != "lap-moo-survey-panel-v1":
        raise ValueError("Unsupported sampling panel definition.")
    mn = panel.get("minnesota", {})
    mrks = panel.get("mrks", {})
    if mn.get("canonical_source_sha256") != group_store.manifest.get(
        "source_pickle_sha256_verified"
    ):
        raise ValueError("Sampling panel and Minnesota group store use different sources.")
    if mrks.get("central_manifest_sha256") != systems.central_manifest_sha256:
        raise ValueError("Sampling panel and mRKS cache use different central corpora.")
    store_by_identity = {
        (row["database"], row["reaction_id"]): row
        for row in group_store.manifest["groups"]
    }
    catalog = []
    for row in mn.get("reactions", []):
        identity = (row["database"], int(row["reaction_id"]))
        stored = store_by_identity.get(identity)
        variants = sorted(row.get("variant_suffixes", []))
        if (
            stored is None
            or stored["source_group_key"] != row.get("source_group_key")
            or variants != stored["variant_suffixes"]
            or row.get("canonical_variant_suffix") not in variants
        ):
            raise ValueError(f"Sampling panel reaction does not match stored source: {identity}.")
        catalog.append(
            {"database": identity[0], "reaction_id": identity[1], "variants": variants}
        )
    mrks_rows = mrks.get("systems", [])
    selected_systems = [row["system_name"] for row in mrks_rows]
    if (
        len(catalog) != 27
        or len(selected_systems) != 15
        or len(set(selected_systems)) != 15
        or not set(selected_systems) <= set(systems.system_names)
    ):
        raise ValueError("Sampling panel must resolve to 27 reactions and 15 cached mRKS systems.")
    return catalog, selected_systems, file_sha256(panel_path)


def _four_task_objectives(model, shadow, sample, store, system,
                          reaction_dispersions, mrks_dispersions, point_chunk_size):
    """Consume listed variants/weights; use existing bounded chemistry and AO paths."""
    objectives = {}
    for task in FOUR_TASK_NAMES[:2]:
        batch = sample["task_samples"][task]
        reactions = tuple(store.load_variant(
            (row["database"], row["reaction_id"]), row["variant_suffix"],
        ) for row in batch)
        objectives[task] = ChemistryBatchObjective(
            model, shadow, reactions, tuple(row["weight"] for row in batch), reaction_dispersions,
        )
    energy, operator = make_mrks_objective_factories(
        model, system, point_chunk_size=point_chunk_size, dispersions=mrks_dispersions,
    )
    objectives.update(exc=energy, op=operator)
    return objectives


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predopt-checkpoint", type=Path, required=True)
    parser.add_argument("--minnesota-store-manifest", type=Path, required=True)
    parser.add_argument("--central-data-dir", type=Path, required=True)
    parser.add_argument("--ao-cache-dir", type=Path, required=True)
    parser.add_argument("--panel-definition", type=Path)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHOD_IDS, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--step-rule",
        choices=("optimizer", PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE),
        default="optimizer",
        help="Use the existing optimizer/schedule path or direct PCD vector-Armijo steps.",
    )
    parser.add_argument(
        "--optimizer-family",
        choices=OPTIMIZER_FAMILIES,
        default="radamw",
        help="Optimizer used after MOO aggregation; default preserves historical RAdamW behavior.",
    )
    parser.add_argument(
        "--muon-learning-rate",
        type=float,
        help="Required only for muon_adamw. --learning-rate remains the AdamW fallback rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optimizer weight decay (defaults to 1e-2 for optimizer mode; direct PCD requires 0).",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--point-chunk-size", type=int, default=256)
    parser.add_argument("--ao-cache-chunk-size", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--four-task-pcd", action="store_true", help="Consume immutable v2 relchem/ae17 batches with direct PCD Armijo.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--method-hyperparameters", type=Path)
    parser.add_argument("--fixed-calibration-report", type=Path)
    parser.add_argument("--reaction-dispersions", type=Path, default=DEFAULT_REACTION_DISPERSIONS)
    parser.add_argument("--mrks-dispersions", type=Path, default=DEFAULT_MRKS_DISPERSIONS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(args.updates, args.point_chunk_size, args.ao_cache_chunk_size, args.checkpoint_every) <= 0:
        raise ValueError("Update count, chunk sizes, and checkpoint interval must be positive.")
    direct_armijo = args.step_rule == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    if args.four_task_pcd and (args.method != "pcd" or not direct_armijo
            or args.dtype != "float32" or args.fixed_calibration_report is not None
            or args.optimizer_family != "radamw"):
        raise ValueError("Four-task mode requires F32 direct PCD Armijo with no optimizer/scalarization override.")
    if args.weight_decay is None:
        args.weight_decay = 0.0 if direct_armijo else 1.0e-2
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Optimizer learning rate and weight decay are invalid.")
    if direct_armijo and args.method != "pcd":
        raise ValueError("Direct vector-Armijo stepping requires --method pcd.")
    if direct_armijo and args.weight_decay != 0.0:
        raise ValueError("Direct vector-Armijo stepping does not accept weight decay.")
    if direct_armijo and args.muon_learning_rate is not None:
        raise ValueError("Direct vector-Armijo stepping does not use optimizer learning rates.")
    if not direct_armijo and args.optimizer_family == "muon_adamw":
        if args.muon_learning_rate is None or args.muon_learning_rate <= 0:
            raise ValueError("muon_adamw requires --muon-learning-rate > 0.")
    elif not direct_armijo and args.muon_learning_rate is not None:
        raise ValueError("--muon-learning-rate is only valid with --optimizer-family muon_adamw.")
    stop_after = args.updates if args.stop_after is None else args.stop_after
    if not 0 < stop_after <= args.updates:
        raise ValueError("--stop-after must be positive and no greater than --updates.")
    device, rank, _local_rank, world_size = _distributed_runtime(args.device)
    rank_args = {
        name: _external_path(getattr(args, name), name.replace("_", " "))
        for name in (
            "predopt_checkpoint",
            "minnesota_store_manifest",
            "central_data_dir",
            "ao_cache_dir",
            "sampling_manifest",
            "output_dir",
        )
    }
    rank_args["reaction_dispersions"] = Path(args.reaction_dispersions).resolve()
    rank_args["mrks_dispersions"] = Path(args.mrks_dispersions).resolve()
    if args.panel_definition is not None:
        rank_args["panel_definition"] = _external_path(
            args.panel_definition, "panel definition"
        )
    for name, path in rank_args.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)
    if args.method_hyperparameters is not None:
        args.method_hyperparameters = _external_path(args.method_hyperparameters, "method hyperparameters")
    if args.fixed_calibration_report is not None:
        args.fixed_calibration_report = _external_path(args.fixed_calibration_report, "fixed calibration report")
    output_dir = rank_args["output_dir"]
    if rank == 0:
        if output_dir.exists() and args.resume is None:
            raise FileExistsError(f"Refusing to overwrite MOO output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    if world_size > 1:
        dist.barrier()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    _seed_process(args.seed + rank)
    predopt_sha = file_sha256(rank_args["predopt_checkpoint"])
    model, predopt_payload = load_lap_checkpoint(
        rank_args["predopt_checkpoint"], device=device, dtype=dtype
    )
    if (
        predopt_payload.get("predopt_only") is not True
        or predopt_payload.get("predopt_epochs") != 2
        or float(predopt_payload.get("predopt_lr", -1.0)) != 1e-2
        or predopt_payload.get("predopt_seed") != 41
    ):
        raise ValueError("Initialization must be the verified two-epoch canonical PBE predopt checkpoint.")
    model.train()

    group_store = MinnesotaGroupStore(rank_args["minnesota_store_manifest"], cache_groups=1)
    if group_store.manifest.get("group_count") != 268:
        raise ValueError("The training Minnesota store must contain all 268 base groups.")
    systems = CentralAOCache(
        rank_args["central_data_dir"],
        rank_args["ao_cache_dir"],
        device=device,
        dtype=dtype,
        chunk_size=args.ao_cache_chunk_size,
    )
    catalog, sampling_system_names, panel_sha = _catalog_and_systems_for_panel(
        rank_args.get("panel_definition"), group_store, systems
    )
    if args.four_task_pcd:
        # Four-task chemistry covers all268; panel provenance still scopes cached systems.
        catalog, _, _ = _catalog_and_systems_for_panel(None, group_store, systems)
    source_hashes = {
        "minnesota_group_store_manifest": group_store.manifest_sha256,
        "central_operator_manifest": systems.central_manifest_sha256,
        "ao_factor_cache_manifest": systems.cache_manifest_sha256,
        "predopt_checkpoint": predopt_sha,
        "reaction_dispersions": file_sha256(rank_args["reaction_dispersions"]),
        "mrks_dispersions": file_sha256(rank_args["mrks_dispersions"]),
    }
    if panel_sha is not None:
        source_hashes["panel_definition"] = panel_sha
    manifest = _load_sampling_manifest(
        rank_args["sampling_manifest"],
        catalog=catalog,
        system_names=sampling_system_names,
        source_hashes=source_hashes,
        updates=args.updates,
        seed=args.seed,
        world_size=world_size,
        rank=rank,
        four_task=args.four_task_pcd,
    )
    if args.method == "fixed" and args.fixed_calibration_report is None:
        raise ValueError("Fixed scalarization requires --fixed-calibration-report.")
    method_hparams, fixed_scalarization = _method_configuration(
        args.method, args.method_hyperparameters, args.fixed_calibration_report
    )
    vector_armijo = None
    if direct_armijo:
        optimizer = None
        scheduler = None
        optimizer_metadata = {"name": "none"}
        lr_schedule = {"name": "none"}
        vector_armijo = {
            "c": 1.0e-4,
            "rho": 0.5,
            "max_backtracks": 20,
            "initial_step_size": args.learning_rate,
        }
    else:
        optimizer, optimizer_metadata = build_optimizer(
            model,
            family=args.optimizer_family,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            muon_learning_rate=args.muon_learning_rate,
        )
        scheduler = make_cosine_scheduler(optimizer, total_updates=args.updates)
        lr_schedule = {
            "name": "cosine",
            "total_updates": args.updates,
            "minimum_lr_ratio": 0.1,
            "same_shape_for_all_methods": True,
        }
    pcd_protocol_kwargs: dict[str, Any] = {}
    pcd_source_git_commit: str | None = None
    pcd_source_hashes: dict[str, str] | None = None
    if args.method == "pcd":
        pcd_source_git_commit, pcd_source_hashes = _current_pcd_source_identity()
        pcd_protocol_kwargs = {
            "world_size": world_size,
            "sampling_manifest_file_sha256": file_sha256(rank_args["sampling_manifest"]),
            "source_code_sha256": pcd_source_hashes,
        }
    if args.four_task_pcd:
        pcd_protocol_kwargs.update(task_order=FOUR_TASK_NAMES, operator_precision_source_sha256={
            p: file_sha256(REPO_ROOT / p) for p in OPERATOR_PRECISION_SOURCE_PATHS
        })
    protocol = make_protocol_metadata(
        architecture=model.architecture,
        method=args.method,
        method_hyperparameters=method_hparams,
        fixed_scalarization=fixed_scalarization,
        optimizer=optimizer_metadata,
        lr_schedule=lr_schedule,
        predopt_checkpoint_sha256=predopt_sha,
        sampling_manifest_sha256=manifest["manifest_sha256"],
        minnesota_data_sha256=group_store.manifest_sha256,
        operator_corpus_manifest_sha256=systems.central_manifest_sha256,
        ao_cache_manifest_sha256=systems.cache_manifest_sha256,
        reaction_dispersions_sha256=file_sha256(rank_args["reaction_dispersions"]),
        mrks_dispersions_sha256=file_sha256(rank_args["mrks_dispersions"]),
        random_seed=args.seed,
        dtype=args.dtype,
        grid_chunk_size=args.point_chunk_size,
        ao_cache_chunk_size=args.ao_cache_chunk_size,
        step_rule=PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE if direct_armijo else None,
        vector_armijo=vector_armijo,
        **pcd_protocol_kwargs,
    )
    validate_protocol_metadata(protocol)

    run_protocol_path = output_dir / "protocol.json"
    if rank == 0 and args.resume is None:
        _write_json(run_protocol_path, protocol)
        _write_json(
            output_dir / "sampling_manifest_identity.json",
            {
                "sampling_manifest_path": str(rank_args["sampling_manifest"]),
                "sampling_manifest_sha256": manifest["manifest_sha256"],
            },
        )
    elif args.resume is not None and (rank == 0 or args.four_task_pcd):
        # All four-task ranks fail together before the resume barrier.
        if not run_protocol_path.exists() or _read_json(run_protocol_path) != protocol:
            raise ValueError("Existing output protocol does not match the resume request.")
    if world_size > 1:
        dist.barrier()

    next_update = 0
    aggregator_state = _initial_aggregator_state(args.method)
    if args.resume is not None:
        resume_path = _external_path(args.resume, "resume checkpoint")
        next_update, aggregator_state = load_moo_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_protocol_metadata=protocol,
            sampling_manifest=manifest,
            map_location=device,
            restore_rng=True,
        )
    if next_update > stop_after:
        raise ValueError("Resume cursor exceeds --stop-after.")

    chemistry_shadow = copy.deepcopy(model).double() if args.four_task_pcd else None
    reaction_dispersions = load_reaction_dispersions(str(rank_args["reaction_dispersions"]))
    mrks_dispersions = load_mrks_dispersions(str(rank_args["mrks_dispersions"]))
    stream = SamplingStream(manifest)
    history_path = output_dir / "updates.jsonl"
    history = history_path.open("a" if args.resume is not None else "x", encoding="utf-8") if rank == 0 else None
    completed_updates = next_update
    stopped_update = None
    stop_reason = None
    try:
        for update_index in range(next_update, stop_after):
            sample = stream.entry(update_index, rank)
            reaction = None
            if args.four_task_pcd:
                system = systems.load(sample["mrks_system"])
                objective_factories = _four_task_objectives(
                    model, chemistry_shadow, sample, group_store, system,
                    reaction_dispersions, mrks_dispersions, args.point_chunk_size,
                )
            else:
                reaction_identity = sample["reaction"]
                reaction = group_store.load_variant(
                    (reaction_identity["database"], reaction_identity["reaction_id"]),
                    sample["variant_suffix"],
                )
                system = systems.load(sample["mrks_system"])
                objective_factories = make_three_objective_factories(
                    model,
                    reaction,
                    system,
                    device=device,
                    dtype=dtype,
                    reaction_dispersions=reaction_dispersions,
                    mrks_dispersions=mrks_dispersions,
                    point_chunk_size=args.point_chunk_size,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                result = _run_scheduled_update(
                    model,
                    optimizer,
                    objective_factories,
                    method=args.method,
                    hyperparameters=method_hparams,
                    aggregator_state=aggregator_state,
                    scheduler=scheduler,
                    world_size=world_size,
                    step_rule=args.step_rule,
                    vector_armijo=vector_armijo,
                    task_order=FOUR_TASK_NAMES if args.four_task_pcd else TASK_NAMES,
                )
            except Exception as exc:
                if args.method == "nash_mtl" and "Nash-MTL" in str(exc):
                    failure_report = _capture_nash_aggregation_failure(
                        output_dir,
                        update_index=update_index,
                        sample=sample,
                        exception=exc,
                        protocol=protocol,
                        sampling_manifest=manifest,
                        model=model,
                        objective_factories=objective_factories,
                        device=device,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "nash_aggregation_failure_captured",
                                "diagnostic": failure_report,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if direct_armijo and rank == 0:
                    failure_row = {
                        "event": "vector_armijo_update_error",
                        "update": update_index,
                        "rank_samples": manifest["entries"][update_index]["per_rank"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "sampling_cursor_next_update": update_index,
                    }
                    history.write(json.dumps(_json_safe(failure_row), sort_keys=True, allow_nan=False) + "\n")
                    history.flush()
                    print(json.dumps(_json_safe(failure_row), sort_keys=True, allow_nan=False), flush=True)
                raise
            if not result.accepted:
                failure_row = _vector_armijo_rejection_record(
                    update_index=update_index,
                    rank_samples=manifest["entries"][update_index]["per_rank"],
                    result=result,
                )
                if rank == 0:
                    history.write(json.dumps(_json_safe(failure_row), sort_keys=True, allow_nan=False) + "\n")
                    history.flush()
                    print(json.dumps(_json_safe(failure_row), sort_keys=True, allow_nan=False), flush=True)
                save_moo_checkpoint(
                    output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    protocol_metadata=protocol,
                    sampling_manifest=manifest,
                    next_update=update_index,
                    aggregator_state=aggregator_state,
                )
                stopped_update = update_index
                stop_reason = result.stop_reason
                del objective_factories, reaction, system, result
                break
            aggregator_state = result.aggregator_state
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "update": update_index,
                "rank_samples": manifest["entries"][update_index]["per_rank"],
                "losses": result.losses,
                "gradient_diagnostics": result.diagnostics,
                "aggregator_state": aggregator_state,
                "learning_rate_for_next_update": result.learning_rate,
                "optimizer_learning_rates_for_next_update": None if optimizer is None else (
                    optimizer.learning_rates()
                    if hasattr(optimizer, "learning_rates")
                    else {"main": float(optimizer.param_groups[0]["lr"])}
                ),
                "system_cache_load_seconds": systems.last_load_seconds,
                "system_ao_cache_bytes": systems.last_system_cache_bytes,
                "step_seconds": time.perf_counter() - started,
                "point_chunk_size": args.point_chunk_size,
                "ao_cache_chunk_size": args.ao_cache_chunk_size,
            }
            if device.type == "cuda":
                row["cuda_memory"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                }
            if not all(np.isfinite(float(value)) for value in result.losses.values()):
                raise FloatingPointError(f"Nonfinite task loss at update {update_index}.")
            if rank == 0:
                history.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
                history.flush()
                print(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False), flush=True)
            completed = update_index + 1
            completed_updates = completed
            if completed % args.checkpoint_every == 0 or completed == stop_after:
                save_moo_checkpoint(
                    output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    protocol_metadata=protocol,
                    sampling_manifest=manifest,
                    next_update=completed,
                    aggregator_state=aggregator_state,
                )
            del objective_factories, reaction, system, result
    finally:
        if history is not None:
            history.close()
    if rank == 0:
        summary = {
            "protocol_metadata_sha256": hashlib.sha256(
                json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "method": args.method,
            "updates_completed": completed_updates,
            "updates_target": args.updates,
            "run_complete": completed_updates == args.updates,
            "sampling_manifest_sha256": manifest["manifest_sha256"],
            "predopt_checkpoint_sha256": predopt_sha,
            "final_checkpoint": str((output_dir / "latest.pt").resolve()),
        }
        if args.method == "pcd":
            summary["source_code_git_commit"] = pcd_source_git_commit
            summary["source_code_sha256"] = pcd_source_hashes
        if stop_reason is not None:
            summary["stop_reason"] = stop_reason
            summary["stopped_update"] = stopped_update
        if direct_armijo:
            summary["step_rule"] = PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
        _write_json(output_dir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
