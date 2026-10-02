"""Run the frozen-panel raw-gradient survey for one-stage Lap MOO."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from lap_checkpoint import load_lap_checkpoint
from lap_moo_panel import MinnesotaGroupStore
from lap_moo_protocol import (
    METHOD_IDS,
    TASK_NAMES,
    SamplingStream,
    build_sampling_manifest_from_catalog,
    file_sha256,
    read_sampling_manifest,
    write_sampling_manifest,
)
from lap_moo_training import (
    compute_isolated_task_gradients,
    make_three_objective_factories,
    materialize_task_zeros,
    named_trainable_parameters,
)
from moo_aggregators import aggregate_task_gradients
from optuna_joint import DEFAULT_MRKS_DISPERSIONS, load_mrks_dispersions
from train_lap import DEFAULT_REACTION_DISPERSIONS, load_reaction_dispersions
from train_lap_moo import CentralAOCache, _external_path

SCHEMA = "lap-moo-raw-gradient-survey-v2"
_DEFAULT_PREDOPT = Path(
    r"C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\predopt_fgpu_20261001T192623\lap_pbe_predopt.pt"
)
_DEFAULT_PANEL = Path(r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\panel_definition.json")
_DEFAULT_GROUP_STORE = Path(
    r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mn_group_store_268\manifest.json"
)
_DEFAULT_CENTRAL_DATA = Path(r"C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\all90")
_DEFAULT_AO_CACHE = Path(
    r"C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mrks_15system_ao_cache"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _qstats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Quantile summaries require finite nonempty values.")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _flat_vectors(
    gradients: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    vectors = {}
    for task in TASK_NAMES:
        parts = [
            gradients[task][name].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            for name in sorted(gradients[task])
        ]
        vectors[task] = torch.cat(parts)
    return vectors


def _pairwise_cosines(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    norms = {task: float(torch.linalg.vector_norm(value).item()) for task, value in vectors.items()}
    output = {}
    for i, left in enumerate(TASK_NAMES):
        for right in TASK_NAMES[i + 1 :]:
            denominator = norms[left] * norms[right]
            output[f"{left}:{right}"] = (
                float(torch.dot(vectors[left], vectors[right]).item() / denominator)
                if denominator
                else 0.0
            )
    return output


def _calibration_weights(norm_rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    medians = {
        task: float(np.median([row[task] for row in norm_rows])) for task in TASK_NAMES
    }
    if any(value <= 0.0 or not math.isfinite(value) for value in medians.values()):
        raise ValueError("Fixed calibration requires positive finite median raw gradient norms.")
    geometric_mean = math.prod(medians.values()) ** (1.0 / len(TASK_NAMES))
    weights = {task: geometric_mean / medians[task] for task in TASK_NAMES}
    return medians, weights


def _panel_inputs(
    panel_path: Path,
    group_store: MinnesotaGroupStore,
    systems: CentralAOCache,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    panel = _read_json(panel_path)
    if panel.get("schema") != "lap-moo-survey-panel-v1":
        raise ValueError("Unsupported MOO survey panel definition.")
    mn = panel.get("minnesota", {})
    mrks = panel.get("mrks", {})
    if mn.get("canonical_source_sha256") != group_store.manifest.get(
        "source_pickle_sha256_verified"
    ):
        raise ValueError("Panel and Minnesota indexed store refer to different source pickles.")
    if mrks.get("central_manifest_sha256") != systems.central_manifest_sha256:
        raise ValueError("Panel and AO cache refer to different central operator corpora.")
    reactions = mn.get("reactions")
    mrks_rows = mrks.get("systems")
    if not isinstance(reactions, list) or len(reactions) != 27:
        raise ValueError("The frozen survey panel must contain 27 Minnesota reactions.")
    if not isinstance(mrks_rows, list) or len(mrks_rows) != 15:
        raise ValueError("The frozen survey panel must contain 15 mRKS systems.")
    store_by_identity = {
        (row["database"], row["reaction_id"]): row
        for row in group_store.manifest["groups"]
    }
    catalog = []
    databases = set()
    for row in reactions:
        identity = (row["database"], int(row["reaction_id"]))
        stored = store_by_identity.get(identity)
        if stored is None or stored["source_group_key"] != row["source_group_key"]:
            raise ValueError(f"Panel reaction does not match indexed Minnesota identity {identity}.")
        variants = sorted(row["variant_suffixes"])
        if variants != stored["variant_suffixes"]:
            raise ValueError(f"Panel augmentation set differs from indexed source for {identity}.")
        if row["canonical_variant_suffix"] not in variants:
            raise ValueError(f"Canonical variant is absent for {identity}.")
        catalog.append(
            {
                "database": identity[0],
                "reaction_id": identity[1],
                "variants": variants,
            }
        )
        databases.add(identity[0])
    if len(databases) != 9:
        raise ValueError("The survey panel must include all nine Minnesota databases.")
    systems_names = [row["system_name"] for row in mrks_rows]
    if len(set(systems_names)) != 15 or not set(systems_names) <= set(systems.system_names):
        raise ValueError("The panel mRKS identities are missing from the AO-factor cache.")
    return catalog, systems_names, panel


def _method_diagnostics(
    samples: list[dict[str, Any]],
    fixed_weights: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    configurations = {
        "fixed": {"fixed_weights": [fixed_weights[name] for name in TASK_NAMES]},
        "imtl_g": {},
        "cagrad": {"c": 0.4, "rescale": "paper_unscaled"},
        "nash_mtl": {
            "solver": "newton_potential",
            "max_iter": 100,
            "tol": 1e-10,
            "update_every": 1,
        },
        "pcd": {"tau": 0.02, "beta": 0.999, "eps": 1.0e-8, "qp_tolerance": 1.0e-9},
    }
    states: dict[str, dict[str, Any] | None] = {
        method: (None if method == "pcd" else {}) for method in METHOD_IDS
    }
    reports: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_IDS}
    for sample in samples:
        for method in METHOD_IDS:
            try:
                joint, diagnostics, state = aggregate_task_gradients(
                    sample["gradients"],
                    method=method,
                    hyperparameters=configurations[method],
                    state=states[method],
                )
                states[method] = state
                joint_vector = torch.cat(
                    [
                        joint[name].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                        for name in sorted(joint)
                        if joint[name] is not None
                    ]
                )
                reports[method].append(
                    {
                        "update": sample["update"],
                        "coefficients": diagnostics["coefficients"],
                        "joint_gradient_norm": float(torch.linalg.vector_norm(joint_vector).item()),
                        "task_directional_dots": diagnostics["directional_task_dots"],
                        "raw_gradient_gram": diagnostics["gram"],
                        "raw_gradient_cosine_matrix": diagnostics["task_cosines"],
                        "solver_status": diagnostics["solver_status"],
                        "solver_iterations": diagnostics["solver_iterations"],
                        "solver_residual": diagnostics["solver_residual"],
                    }
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                reports[method].append(
                    {"update": sample["update"], "error": f"{type(exc).__name__}: {exc}"}
                )
    return reports


def run_raw_gradient_survey(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _external_path(args.output_dir, "survey output directory")
    manifest_path = _external_path(args.sampling_manifest, "sampling manifest")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite survey output directory: {output_dir}")
    if not 27 <= args.samples <= 40:
        raise ValueError("The raw-gradient survey uses 27–40 paired samples.")
    if args.manifest_updates < args.samples or args.manifest_updates < 100:
        raise ValueError("The shared persisted stream must cover the survey and later short runs.")

    paths = {
        "predopt": _external_path(args.predopt_checkpoint, "predopt checkpoint"),
        "panel": _external_path(args.panel_definition, "panel definition"),
        "group_store": _external_path(args.minnesota_store_manifest, "Minnesota store manifest"),
        "central": _external_path(args.central_data_dir, "central operator corpus"),
        "ao_cache": _external_path(args.ao_cache_dir, "AO-factor cache"),
        "reaction_dispersions": Path(args.reaction_dispersions).resolve(),
        "mrks_dispersions": Path(args.mrks_dispersions).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA survey requested but unavailable.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    _seed(args.seed)
    predopt_hash = file_sha256(paths["predopt"])
    model, predopt_payload = load_lap_checkpoint(
        paths["predopt"], device=device, dtype=dtype
    )
    if (
        predopt_payload.get("predopt_only") is not True
        or predopt_payload.get("predopt_epochs") != 2
        or float(predopt_payload.get("predopt_lr", -1.0)) != 1e-2
        or predopt_payload.get("predopt_seed") != 41
    ):
        raise ValueError("Survey must use the verified two-epoch canonical PBE predopt checkpoint.")
    model.train()

    group_store = MinnesotaGroupStore(paths["group_store"], cache_groups=1)
    systems = CentralAOCache(
        paths["central"],
        paths["ao_cache"],
        device=device,
        dtype=dtype,
        chunk_size=args.ao_cache_chunk_size,
    )
    catalog, system_names, _panel = _panel_inputs(paths["panel"], group_store, systems)
    source_hashes = {
        "predopt_checkpoint": predopt_hash,
        "panel_definition": file_sha256(paths["panel"]),
        "minnesota_group_store_manifest": group_store.manifest_sha256,
        "central_operator_manifest": systems.central_manifest_sha256,
        "ao_factor_cache_manifest": systems.cache_manifest_sha256,
        "reaction_dispersions": file_sha256(paths["reaction_dispersions"]),
        "mrks_dispersions": file_sha256(paths["mrks_dispersions"]),
    }
    expected_manifest = build_sampling_manifest_from_catalog(
        catalog,
        system_names,
        updates=args.manifest_updates,
        seed=args.seed,
        source_hashes=source_hashes,
    )
    if manifest_path.exists():
        sampling_manifest = read_sampling_manifest(manifest_path)
        if sampling_manifest["manifest_sha256"] != expected_manifest["manifest_sha256"]:
            raise ValueError("Persisted survey stream does not match panel/data/seed inputs.")
    else:
        write_sampling_manifest(manifest_path, expected_manifest)
        sampling_manifest = expected_manifest
    stream = SamplingStream(sampling_manifest)
    reaction_dispersions = load_reaction_dispersions(str(paths["reaction_dispersions"]))
    mrks_dispersions = load_mrks_dispersions(str(paths["mrks_dispersions"]))

    samples: list[dict[str, Any]] = []
    norm_rows = []
    parameter_names = tuple(named_trainable_parameters(model))
    started = time.perf_counter()
    for update in range(args.samples):
        sample = stream.entry(update)
        identity = sample["reaction"]
        reaction = group_store.load_variant(
            (identity["database"], identity["reaction_id"]), sample["variant_suffix"]
        )
        system = systems.load(sample["mrks_system"])
        objectives = make_three_objective_factories(
            model,
            reaction,
            system,
            device=device,
            dtype=dtype,
            reaction_dispersions=reaction_dispersions,
            mrks_dispersions=mrks_dispersions,
            point_chunk_size=args.point_chunk_size,
        )
        sample_started = time.perf_counter()
        losses, sparse = compute_isolated_task_gradients(model, objectives)
        dense = materialize_task_zeros(model, sparse)
        dense_cpu = {
            task: {name: grad.detach().cpu() for name, grad in dense[task].items()}
            for task in TASK_NAMES
        }
        vectors = _flat_vectors(dense_cpu)
        raw_norms = {
            task: float(torch.linalg.vector_norm(vector).item())
            for task, vector in vectors.items()
        }
        norm_rows.append(raw_norms)
        sample_row = {
            "update": update,
            "database": identity["database"],
            "reaction_id": identity["reaction_id"],
            "variant_suffix": sample["variant_suffix"],
            "mrks_system": sample["mrks_system"],
            "losses": losses,
            "raw_gradient_norms": raw_norms,
            "raw_gradient_cosines": _pairwise_cosines(vectors),
            "system_cache_load_seconds": systems.last_load_seconds,
            "sample_seconds": time.perf_counter() - sample_started,
        }
        samples.append({"row": sample_row, "gradients": dense_cpu, "vectors": vectors})
        print(
            json.dumps(
                {"survey_sample": update + 1, "of": args.samples, **sample_row},
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        del objectives, reaction, system, sparse, dense, dense_cpu

    median_norms, fixed_weights = _calibration_weights(norm_rows)
    aggregation_samples = [
        {"update": item["row"]["update"], "gradients": item["gradients"]}
        for item in samples
    ]
    aggregation = _method_diagnostics(aggregation_samples, fixed_weights)

    quantiles: dict[str, Any] = {"by_database": {}, "by_system": {}}
    for group_key, destination in (
        ("database", quantiles["by_database"]),
        ("mrks_system", quantiles["by_system"]),
    ):
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in samples:
            strata[str(item["row"][group_key])].append(item["row"])
        for identity, rows in sorted(strata.items()):
            update_ids = {row["update"] for row in rows}
            aggregator_quantiles = {}
            for method, method_rows in aggregation.items():
                selected = [
                    row
                    for row in method_rows
                    if row["update"] in update_ids and "error" not in row
                ]
                if not selected:
                    aggregator_quantiles[method] = {"successful_samples": 0}
                    continue
                aggregator_quantiles[method] = {
                    "successful_samples": len(selected),
                    "joint_gradient_norm": _qstats(
                        [float(row["joint_gradient_norm"]) for row in selected]
                    ),
                    "coefficients": {
                        task: _qstats(
                            [float(row["coefficients"][task]) for row in selected]
                        )
                        for task in TASK_NAMES
                    },
                    "task_directional_dots": {
                        task: _qstats(
                            [float(row["task_directional_dots"][task]) for row in selected]
                        )
                        for task in TASK_NAMES
                    },
                    "solver_residual": _qstats(
                        [float(row["solver_residual"]) for row in selected]
                    ),
                }
            destination[identity] = {
                "sample_count": len(rows),
                "task_losses": {
                    task: _qstats([float(row["losses"][task]) for row in rows])
                    for task in TASK_NAMES
                },
                "raw_gradient_norms": {
                    task: _qstats([float(row["raw_gradient_norms"][task]) for row in rows])
                    for task in TASK_NAMES
                },
                "raw_gradient_cosines": {
                    cosine: _qstats(
                        [float(row["raw_gradient_cosines"][cosine]) for row in rows]
                    )
                    for cosine in rows[0]["raw_gradient_cosines"]
                },
                "aggregation_methods": aggregator_quantiles,
            }

    flat_path = output_dir / "raw_task_gradients.npz"
    matrices = {
        task: np.stack(
            [item["vectors"][task].numpy() for item in samples], axis=0
        )
        for task in TASK_NAMES
    }
    np.savez_compressed(
        flat_path,
        parameter_names=np.asarray(parameter_names, dtype=np.str_),
        sample_identity=np.asarray(
            [
                f"{item['row']['database']}:{item['row']['reaction_id']}:{item['row']['variant_suffix']}|{item['row']['mrks_system']}"
                for item in samples
            ],
            dtype=np.str_,
        ),
        **matrices,
    )
    report = {
        "schema": SCHEMA,
        "status": "raw gradients measured; no optimizer update or training performed",
        "checkpoint": str(paths["predopt"].resolve()),
        "checkpoint_sha256": predopt_hash,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "panel_definition": str(paths["panel"].resolve()),
        "panel_definition_sha256": file_sha256(paths["panel"]),
        "sampling_manifest": str(manifest_path.resolve()),
        "sampling_manifest_sha256": sampling_manifest["manifest_sha256"],
        "minnesota_group_store_manifest_sha256": group_store.manifest_sha256,
        "central_operator_manifest_sha256": systems.central_manifest_sha256,
        "ao_cache_manifest_sha256": systems.cache_manifest_sha256,
        "reaction_dispersions_sha256": source_hashes["reaction_dispersions"],
        "mrks_dispersions_sha256": source_hashes["mrks_dispersions"],
        "sample_count": args.samples,
        "manifest_updates": args.manifest_updates,
        "parameter_count": sum(parameter.numel() for parameter in named_trainable_parameters(model).values()),
        "parameter_order": list(parameter_names),
        "raw_task_gradient_medians": median_norms,
        "fixed_calibration": {
            "method": "lambda_i = geometric_mean(median_raw_gradient_norms) / median_raw_gradient_norm_i",
            "fixed_weights": [fixed_weights[task] for task in TASK_NAMES],
            "weight_by_task": fixed_weights,
            "geometric_mean_of_weights": math.prod(fixed_weights.values()) ** (1.0 / len(TASK_NAMES)),
            "frozen_from_initial_predopt_panel": True,
        },
        "gradient_artifact": {
            "path": str(flat_path.resolve()),
            "sha256": file_sha256(flat_path),
            "dtype": "float64",
            "shape_by_task": {task: list(matrix.shape) for task, matrix in matrices.items()},
        },
        "aggregation_geometry": aggregation,
        "stratified_quantiles": quantiles,
        "samples": [item["row"] for item in samples],
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
    }
    report_path = output_dir / "raw_gradient_survey.json"
    _write_json(report_path, report)
    print(json.dumps({"report": str(report_path.resolve()), "elapsed_seconds": report["elapsed_seconds"], "fixed_weights": report["fixed_calibration"]["fixed_weights"]}, sort_keys=True), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("survey",), default="survey")
    parser.add_argument("--predopt-checkpoint", type=Path, default=_DEFAULT_PREDOPT)
    parser.add_argument("--panel-definition", type=Path, default=_DEFAULT_PANEL)
    parser.add_argument("--minnesota-store-manifest", type=Path, default=_DEFAULT_GROUP_STORE)
    parser.add_argument("--central-data-dir", type=Path, default=_DEFAULT_CENTRAL_DATA)
    parser.add_argument("--ao-cache-dir", type=Path, default=_DEFAULT_AO_CACHE)
    parser.add_argument("--reaction-dispersions", type=Path, default=DEFAULT_REACTION_DISPERSIONS)
    parser.add_argument("--mrks-dispersions", type=Path, default=DEFAULT_MRKS_DISPERSIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-manifest", type=Path)
    parser.add_argument("--samples", type=int, default=27)
    parser.add_argument("--manifest-updates", type=int, default=150)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--point-chunk-size", type=int, default=256)
    parser.add_argument("--ao-cache-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(args.point_chunk_size, args.ao_cache_chunk_size) <= 0:
        raise ValueError("Chunk sizes must be positive.")
    if args.sampling_manifest is None:
        args.sampling_manifest = args.output_dir / "sampling_manifest.json"
    run_raw_gradient_survey(args)


if __name__ == "__main__":
    main()
