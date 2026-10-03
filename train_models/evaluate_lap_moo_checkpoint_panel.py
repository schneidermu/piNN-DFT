"""Evaluate one-stage Lap MOO checkpoints on the fixed 27-pair training panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lap_checkpoint import load_lap_checkpoint
from lap_moo_panel import MinnesotaGroupStore
from lap_moo_protocol import (
    PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
    SamplingStream,
    canonical_sha256,
    file_sha256,
    read_sampling_manifest,
    validate_protocol_metadata,
)
from lap_moo_training import make_three_objective_factories, named_trainable_parameters
from optuna_joint import DEFAULT_MRKS_DISPERSIONS, load_mrks_dispersions
from train_lap import DEFAULT_REACTION_DISPERSIONS, load_reaction_dispersions
from train_lap_moo import CentralAOCache, _external_path

DEFAULT_PREDOPT = Path(
    r"C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\predopt_fgpu_20261001T192623\lap_pbe_predopt.pt"
)
DEFAULT_PANEL = Path(r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\panel_definition.json")
DEFAULT_GROUP_STORE = Path(
    r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mn_group_store_268\manifest.json"
)
DEFAULT_CENTRAL = Path(r"C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\all90")
DEFAULT_AO_CACHE = Path(
    r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mrks_15system_ao_cache"
)
TASKS = ("chem", "exc", "op")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Panel summaries require finite, nonempty values.")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        task: {
            "loss": _quantiles([float(row["losses"][task]) for row in rows]),
            "ratio_to_predopt": _quantiles(
                [float(row["ratios_to_predopt"][task]) for row in rows]
            ),
        }
        for task in TASKS
    }


def _trainable_parameter_l2_norm(model: torch.nn.Module) -> float:
    parameters = named_trainable_parameters(model)
    return math.sqrt(
        sum(float(torch.sum(value.detach().double() ** 2).cpu()) for value in parameters.values())
    )


def _validate_candidate_checkpoint(
    payload: dict[str, Any],
    *,
    method: str,
    sampling_manifest_sha256: str,
    predopt_checkpoint_sha256: str,
    updates: int,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless checkpoint, stream, architecture, and schedule agree."""
    if payload.get("checkpoint_kind") != "lap-moo-one-stage":
        raise ValueError(f"{method} checkpoint is not a clean one-stage MOO checkpoint.")
    protocol = payload.get("protocol_metadata")
    validate_protocol_metadata(protocol, allow_v1_read_only=True)
    if canonical_sha256(protocol) != payload.get("protocol_metadata_sha256"):
        raise ValueError(f"{method} checkpoint protocol metadata hash mismatch.")
    if (
        protocol.get("method") != method
        or protocol.get("sampling_manifest_sha256") != sampling_manifest_sha256
    ):
        raise ValueError(f"{method} checkpoint does not match the requested method/stream.")
    if protocol.get("predopt_checkpoint_sha256") != predopt_checkpoint_sha256:
        raise ValueError(f"{method} checkpoint does not use the selected PBE predopt.")
    cursor = payload.get("sampling_cursor", {}).get("next_update")
    if cursor != updates:
        raise ValueError(f"{method} checkpoint cursor does not match its recorded update count.")
    scheduler_state = payload.get("scheduler_state_dict")
    direct_armijo = protocol.get("step_rule") == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    if direct_armijo:
        if scheduler_state is not None or payload.get("optimizer_state_dict") is not None:
            raise ValueError("Direct vector-Armijo checkpoint has unexpected optimizer or scheduler state.")
    elif scheduler_state is None or scheduler_state.get("last_epoch") != updates:
        raise ValueError(f"{method} checkpoint scheduler epoch does not match its cursor.")
    if dict(payload.get("model_kwargs", {})) != model_kwargs:
        raise ValueError(f"{method} checkpoint architecture differs from the PBE model.")
    return {
        "method": method,
        "updates": updates,
        "protocol_metadata_sha256": payload["protocol_metadata_sha256"],
        "state": payload["model_state_dict"],
    }


def _validate_candidate_method_ids(candidates: Sequence[Sequence[str]]) -> None:
    """Reject panel inputs that would overwrite results keyed by method ID."""
    method_ids = [candidate[0] for candidate in candidates]
    duplicates = sorted({method for method in method_ids if method_ids.count(method) > 1})
    if "predopt" in method_ids:
        raise ValueError("The predopt method ID is reserved for the zero-update baseline.")
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(
            f"Candidate method IDs must be unique within one panel; duplicate: {joined}. "
            "Use separate panel invocations for same-method optimizer candidates."
        )


def _evaluate_model(
    model: torch.nn.Module,
    *,
    candidate: dict[str, Any],
    stream: SamplingStream,
    group_store: MinnesotaGroupStore,
    systems: CentralAOCache,
    device: torch.device,
    dtype: torch.dtype,
    reaction_dispersions,
    mrks_dispersions,
    point_chunk_size: int,
    reference_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows = []
    model.train()
    for update in range(27):
        sample = stream.entry(update)
        identity = sample["reaction"]
        reaction = group_store.load_variant(
            (identity["database"], identity["reaction_id"]), sample["variant_suffix"]
        )
        system = systems.load(sample["mrks_system"])
        factories = make_three_objective_factories(
            model,
            reaction,
            system,
            device=device,
            dtype=dtype,
            reaction_dispersions=reaction_dispersions,
            mrks_dispersions=mrks_dispersions,
            point_chunk_size=point_chunk_size,
        )
        losses = {}
        for task in TASKS:
            loss = factories[task]()
            value = float(loss.detach().double().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite {task} panel loss at update {update}.")
            losses[task] = value
            del loss
        if reference_rows is None:
            ratios = {task: 1.0 for task in TASKS}
        else:
            baseline = reference_rows[update]["losses"]
            ratios = {
                task: losses[task] / float(baseline[task]) for task in TASKS
            }
        rows.append(
            {
                "identity": {
                    "update": update,
                    "database": identity["database"],
                    "reaction_id": int(identity["reaction_id"]),
                    "variant_suffix": sample["variant_suffix"],
                    "mrks_system": sample["mrks_system"],
                },
                "losses": losses,
                "ratios_to_predopt": ratios,
            }
        )
        del factories, reaction, system
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "event": "panel_sample",
                    "candidate": candidate["method"],
                    "update": update,
                    "losses": losses,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predopt-checkpoint", type=Path, default=DEFAULT_PREDOPT)
    parser.add_argument("--panel-definition", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--minnesota-store-manifest", type=Path, default=DEFAULT_GROUP_STORE)
    parser.add_argument("--central-data-dir", type=Path, default=DEFAULT_CENTRAL)
    parser.add_argument("--ao-cache-dir", type=Path, default=DEFAULT_AO_CACHE)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=4, metavar=("METHOD", "STATUS", "UPDATES", "CHECKPOINT"), default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--point-chunk-size", type=int, default=256)
    parser.add_argument("--ao-cache-chunk-size", type=int, default=4096)
    parser.add_argument("--reaction-dispersions", type=Path, default=DEFAULT_REACTION_DISPERSIONS)
    parser.add_argument("--mrks-dispersions", type=Path, default=DEFAULT_MRKS_DISPERSIONS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_candidate_method_ids(args.candidate)
    if args.point_chunk_size <= 0 or args.ao_cache_chunk_size <= 0:
        raise ValueError("Panel chunk sizes must be positive.")
    output_path = _external_path(args.output, "panel output")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite panel output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "predopt": _external_path(args.predopt_checkpoint, "PBE predopt checkpoint"),
        "panel": _external_path(args.panel_definition, "panel definition"),
        "group_store": _external_path(args.minnesota_store_manifest, "Minnesota group-store manifest"),
        "central": _external_path(args.central_data_dir, "central operator data"),
        "ao_cache": _external_path(args.ao_cache_dir, "AO-factor cache"),
        "sampling": _external_path(args.sampling_manifest, "sampling manifest"),
        "reaction_dispersions": Path(args.reaction_dispersions).resolve(),
        "mrks_dispersions": Path(args.mrks_dispersions).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dtype = torch.float32
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    manifest = read_sampling_manifest(paths["sampling"])
    if manifest.get("manifest_sha256") != "550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928":
        raise ValueError("The fixed panel must use the frozen 27-reaction sampling stream.")
    stream = SamplingStream(manifest)
    model, predopt_payload = load_lap_checkpoint(paths["predopt"], device=device, dtype=dtype)
    predopt_sha = file_sha256(paths["predopt"])
    if (
        predopt_payload.get("predopt_only") is not True
        or predopt_payload.get("predopt_epochs") != 2
        or float(predopt_payload.get("predopt_lr", -1.0)) != 1e-2
        or predopt_payload.get("predopt_seed") != 41
    ):
        raise ValueError("Panel baseline must be the verified canonical two-epoch PBE predopt.")
    predopt_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_parameter_l2_norm = _trainable_parameter_l2_norm(model)
    group_store = MinnesotaGroupStore(paths["group_store"], cache_groups=1)
    systems = CentralAOCache(
        paths["central"],
        paths["ao_cache"],
        device=device,
        dtype=dtype,
        chunk_size=args.ao_cache_chunk_size,
    )
    reactions = manifest.get("reaction_catalog", [])
    if len(reactions) != 27 or len(manifest.get("mrks_systems", [])) != 15:
        raise ValueError("Fixed panel stream must cover 27 reactions and 15 mRKS systems.")
    if len({(entry["database"], entry["reaction_id"]) for entry in reactions}) != 27:
        raise ValueError("Fixed panel stream reaction identities are not unique.")
    reaction_dispersions = load_reaction_dispersions(str(paths["reaction_dispersions"]))
    mrks_dispersions = load_mrks_dispersions(str(paths["mrks_dispersions"]))
    started = time.perf_counter()
    candidates = [
        {
            "method": "predopt",
            "status": "zero-update baseline",
            "updates": 0,
            "path": str(paths["predopt"]),
            "checkpoint_sha256": predopt_sha,
            "protocol_metadata_sha256": None,
            "state": predopt_state,
        }
    ]
    for method, status, updates_text, checkpoint_text in args.candidate:
        checkpoint_path = _external_path(checkpoint_text, f"{method} checkpoint")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validated = _validate_candidate_checkpoint(
            payload,
            method=method,
            sampling_manifest_sha256=manifest["manifest_sha256"],
            predopt_checkpoint_sha256=predopt_sha,
            updates=int(updates_text),
            model_kwargs=dict(model.model_kwargs),
        )
        candidates.append(
            {
                **validated,
                "status": status,
                "path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
            }
        )
    evaluations = {}
    baseline_rows = None
    panel_identities = None
    for candidate in candidates:
        model.load_state_dict(candidate["state"], strict=True)
        rows = _evaluate_model(
            model,
            candidate=candidate,
            stream=stream,
            group_store=group_store,
            systems=systems,
            device=device,
            dtype=dtype,
            reaction_dispersions=reaction_dispersions,
            mrks_dispersions=mrks_dispersions,
            point_chunk_size=args.point_chunk_size,
            reference_rows=baseline_rows,
        )
        if candidate["method"] == "predopt":
            baseline_rows = rows
            panel_identities = [row["identity"] for row in rows]
        by_database: dict[str, list[dict[str, Any]]] = {}
        by_system: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_database.setdefault(row["identity"]["database"], []).append(row)
            by_system.setdefault(row["identity"]["mrks_system"], []).append(row)
        evaluations[candidate["method"]] = {
            "status": candidate["status"],
            "updates": candidate["updates"],
            "checkpoint_path": candidate["path"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "protocol_metadata_sha256": candidate["protocol_metadata_sha256"],
            "summary": {
                "overall": _stats(rows),
                "by_database": {key: _stats(value) for key, value in sorted(by_database.items())},
                "by_mrks_system": {key: _stats(value) for key, value in sorted(by_system.items())},
            },
            "rows": rows,
        }
        del candidate["state"]
    output = {
        "schema": "lap-moo-checkpoint-full27-panel-v1",
        "status": "training-data diagnostics only; no optimizer updates or external test sets",
        "device": str(device),
        "dtype": "float32",
        "point_chunk_size": args.point_chunk_size,
        "ao_cache_chunk_size": args.ao_cache_chunk_size,
        "panel_definition_sha256": file_sha256(paths["panel"]),
        "sampling_manifest_path": str(paths["sampling"]),
        "sampling_manifest_sha256": manifest["manifest_sha256"],
        "predopt_checkpoint": str(paths["predopt"]),
        "predopt_sha256": predopt_sha,
        "parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "initial_parameter_l2_norm": initial_parameter_l2_norm,
        "panel_identities": panel_identities,
        "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_path, output)
    print(json.dumps({"output": str(output_path), "elapsed_seconds": output["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
