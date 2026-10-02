"""Summarize hash-bound raw-gradient survey JSON without recomputing gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

LEGACY_RAW_SCHEMA = "lap-moo-raw-gradient-survey-v1"
RAW_SCHEMA = "lap-moo-raw-gradient-survey-v2"
SUMMARY_SCHEMA = "lap-moo-gradient-geometry-summary-v3"
TASKS = ("chem", "exc", "op")
COSINES = ("chem:exc", "chem:op", "exc:op")
LEGACY_METHODS = ("fixed", "imtl_g", "cagrad", "nash_mtl")
METHODS = (*LEGACY_METHODS, "pcd")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("Cannot summarize an empty list of measurements.")
    if not np.isfinite(array).all():
        raise ValueError("Cannot summarize nonfinite survey measurements.")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _sample_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(rows)}
    for field in ("losses", "raw_gradient_norms"):
        result[field] = {
            task: _quantiles([float(row[field][task]) for row in rows]) for task in TASKS
        }
    result["raw_gradient_cosines"] = {
        pair: _quantiles([float(row["raw_gradient_cosines"][pair]) for row in rows])
        for pair in COSINES
    }
    return result


def _method_statistics(
    rows: list[dict[str, Any]], samples_by_update: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    failures = [row for row in rows if "error" in row]
    valid = [row for row in rows if "error" not in row]
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "successful_count": len(valid),
        "error_count": len(failures),
        "errors": [
            {"update": int(row["update"]), "error": str(row["error"])} for row in failures
        ],
        "solver_status_counts": dict(
            sorted(Counter(str(row.get("solver_status", "missing")) for row in valid).items())
        ),
    }
    if not valid:
        return result
    result["coefficients"] = {
        task: _quantiles([float(row["coefficients"][task]) for row in valid])
        for task in TASKS
    }
    result["task_directional_dots"] = {
        task: _quantiles([float(row["task_directional_dots"][task]) for row in valid])
        for task in TASKS
    }
    result["joint_gradient_norm"] = _quantiles(
        [float(row["joint_gradient_norm"]) for row in valid]
    )
    result["solver_iterations"] = _quantiles(
        [float(row["solver_iterations"]) for row in valid]
    )
    result["solver_residual"] = _quantiles(
        [float(row["solver_residual"]) for row in valid]
    )
    unit_projections: dict[str, Any] = {}
    task_cosines: dict[str, Any] = {}
    for task in TASKS:
        projections = []
        cosines = []
        zero_denominator_count = 0
        for row in valid:
            sample = samples_by_update[int(row["update"])]
            task_norm = float(sample["raw_gradient_norms"][task])
            joint_norm = float(row["joint_gradient_norm"])
            if task_norm <= 0.0 or joint_norm <= 0.0:
                zero_denominator_count += 1
                continue
            dot = float(row["task_directional_dots"][task])
            projections.append(dot / task_norm)
            cosines.append(dot / (task_norm * joint_norm))
        unit_projections[task] = {
            "zero_denominator_count": zero_denominator_count,
            **(_quantiles(projections) if projections else {"count": 0}),
        }
        task_cosines[task] = {
            "zero_denominator_count": zero_denominator_count,
            **(_quantiles(cosines) if cosines else {"count": 0}),
        }
    result["unit_task_projections"] = unit_projections
    result["joint_task_cosines"] = task_cosines
    return result


def summarize_raw_gradient_survey(
    report: dict[str, Any], *, source_sha256: str | None = None
) -> dict[str, Any]:
    """Group raw geometry and aggregator diagnostics by DB and mRKS system."""
    source_schema = report.get("schema")
    if source_schema not in (RAW_SCHEMA, LEGACY_RAW_SCHEMA):
        raise ValueError(f"Expected raw survey schema {RAW_SCHEMA!r} or {LEGACY_RAW_SCHEMA!r}.")
    methods = METHODS if source_schema == RAW_SCHEMA else LEGACY_METHODS
    samples = report.get("samples")
    aggregation = report.get("aggregation_geometry")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Raw gradient survey has no sample records.")
    if not isinstance(aggregation, dict) or set(aggregation) != set(methods):
        raise ValueError("Raw survey aggregation methods differ from its schema version.")

    sample_by_update: dict[int, dict[str, Any]] = {}
    for row in samples:
        update = int(row["update"])
        if update in sample_by_update:
            raise ValueError(f"Duplicate survey update {update}.")
        for field in ("database", "mrks_system", "losses", "raw_gradient_norms", "raw_gradient_cosines"):
            if field not in row:
                raise ValueError(f"Raw survey sample {update} is missing {field!r}.")
        sample_by_update[update] = row

    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        by_database[str(row["database"])].append(row)
        by_system[str(row["mrks_system"])].append(row)

    summary_methods: dict[str, Any] = {}
    for method in methods:
        method_rows = aggregation[method]
        if not isinstance(method_rows, list):
            raise TypeError(f"Aggregation rows for {method!r} must be a list.")
        seen_updates = set()
        grouped_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            update = int(row["update"])
            if update not in sample_by_update:
                raise ValueError(f"{method} output refers to unknown sample update {update}.")
            if update in seen_updates:
                raise ValueError(f"{method} contains duplicate update {update}.")
            seen_updates.add(update)
            sample = sample_by_update[update]
            grouped_db[str(sample["database"])].append(row)
            grouped_system[str(sample["mrks_system"])].append(row)
        if seen_updates != set(sample_by_update):
            raise ValueError(f"{method} diagnostics do not cover every raw-gradient sample.")
        summary_methods[method] = {
            "overall": _method_statistics(method_rows, sample_by_update),
            "by_database": {
                key: _method_statistics(value, sample_by_update)
                for key, value in sorted(grouped_db.items())
            },
            "by_system": {
                key: _method_statistics(value, sample_by_update)
                for key, value in sorted(grouped_system.items())
            },
        }

    result = {
        "schema": SUMMARY_SCHEMA,
        "source_schema": RAW_SCHEMA,
        "source_report_sha256": source_sha256,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "sampling_manifest_sha256": report.get("sampling_manifest_sha256"),
        "panel_definition_sha256": report.get("panel_definition_sha256"),
        "sample_count": len(samples),
        "sample_identities": [
            {
                "update": int(row["update"]),
                "database": row["database"],
                "reaction_id": int(row["reaction_id"]),
                "variant_suffix": row["variant_suffix"],
                "mrks_system": row["mrks_system"],
            }
            for row in samples
        ],
        "raw_geometry": {
            "overall": _sample_geometry(samples),
            "by_database": {
                key: _sample_geometry(value) for key, value in sorted(by_database.items())
            },
            "by_system": {
                key: _sample_geometry(value) for key, value in sorted(by_system.items())
            },
        },
        "fixed_calibration": report.get("fixed_calibration"),
        "methods": summary_methods,
        "optimizer_updates": int(report.get("optimizer_updates", -1)),
        "elapsed_seconds": float(report.get("elapsed_seconds", math.nan)),
        "interpretation_note": (
            "Pre-training gradient geometry only. task_directional_dots are raw inner products "
            "d·g_task; unit_task_projections divide by ||g_task|| and joint_task_cosines also "
            "divide by ||d||. Their signs indicate first-order tendencies for a common negative "
            "SGD step, not the effect of an optimizer-transformed parameter update."
        ),
    }
    if not math.isfinite(result["elapsed_seconds"]):
        result["elapsed_seconds"] = None
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.raw_report.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    summary = summarize_raw_gradient_survey(
        report, source_sha256=file_sha256(args.raw_report)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    _main()
