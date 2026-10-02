"""Tests for the survey-only MOO gradient geometry postprocessor."""

from __future__ import annotations

import copy

import pytest
from lap_moo_analysis import (
    METHODS,
    RAW_SCHEMA,
    summarize_raw_gradient_survey,
)


def _sample(update: int, database: str, system: str, scale: float) -> dict:
    return {
        "update": update,
        "database": database,
        "reaction_id": 100 + update,
        "variant_suffix": "_grid_2",
        "mrks_system": system,
        "losses": {"chem": 1.0 * scale, "exc": 2.0 * scale, "op": 3.0 * scale},
        "raw_gradient_norms": {
            "chem": 2.0 * scale,
            "exc": 4.0 * scale,
            "op": 8.0 * scale,
        },
        "raw_gradient_cosines": {
            "chem:exc": 0.1 * scale,
            "chem:op": -0.2 * scale,
            "exc:op": 0.3 * scale,
        },
    }


def _method_row(update: int, coefficient_scale: float) -> dict:
    return {
        "update": update,
        "coefficients": {
            "chem": coefficient_scale,
            "exc": 0.25,
            "op": 0.75 - coefficient_scale,
        },
        "joint_gradient_norm": 5.0 * coefficient_scale,
        "task_directional_dots": {
            "chem": 1.0 * coefficient_scale,
            "exc": -2.0 * coefficient_scale,
            "op": 3.0 * coefficient_scale,
        },
        "solver_iterations": 4,
        "solver_residual": 1e-12,
        "solver_status": "converged",
    }


def _report() -> dict:
    samples = [
        _sample(0, "DB-A", "H2", 1.0),
        _sample(1, "DB-A", "CO", 2.0),
        _sample(2, "DB-B", "H2", 3.0),
    ]
    return {
        "schema": RAW_SCHEMA,
        "checkpoint_sha256": "checkpoint-hash",
        "sampling_manifest_sha256": "sampling-hash",
        "panel_definition_sha256": "panel-hash",
        "optimizer_updates": 0,
        "elapsed_seconds": 12.5,
        "fixed_calibration": {"weight_by_task": {"chem": 1.0, "exc": 2.0, "op": 3.0}},
        "samples": samples,
        "aggregation_geometry": {
            method: [
                _method_row(0, 0.1),
                _method_row(1, 0.2),
                {"update": 2, "error": "ValueError: singular objective"},
            ]
            for method in METHODS
        },
    }


def test_summary_joins_rows_by_update_and_reports_strata_and_failures() -> None:
    result = summarize_raw_gradient_survey(_report(), source_sha256="raw-hash")

    assert result["source_report_sha256"] == "raw-hash"
    assert result["sample_count"] == 3
    assert result["optimizer_updates"] == 0
    assert result["elapsed_seconds"] == 12.5
    assert result["sample_identities"][1] == {
        "update": 1,
        "database": "DB-A",
        "reaction_id": 101,
        "variant_suffix": "_grid_2",
        "mrks_system": "CO",
    }
    assert result["raw_geometry"]["overall"]["losses"]["chem"]["median"] == 2.0
    assert result["raw_geometry"]["by_database"]["DB-A"]["sample_count"] == 2
    assert result["raw_geometry"]["by_system"]["H2"]["sample_count"] == 2

    method = result["methods"]["cagrad"]
    assert method["overall"]["successful_count"] == 2
    assert method["overall"]["error_count"] == 1
    assert method["overall"]["errors"] == [
        {"update": 2, "error": "ValueError: singular objective"}
    ]
    assert method["overall"]["coefficients"]["chem"]["median"] == pytest.approx(0.15)
    assert method["overall"]["unit_task_projections"]["chem"]["median"] == pytest.approx(
        (0.1 / 2.0 + 0.2 / 4.0) / 2
    )
    assert method["overall"]["joint_task_cosines"]["chem"]["count"] == 2
    # Update 1 is DB-A/CO; the join must not accidentally use row order.
    assert method["by_database"]["DB-A"]["task_directional_dots"]["chem"]["median"] == pytest.approx(0.15)
    assert "raw inner products" in result["interpretation_note"]
    assert "optimizer-transformed parameter update" in result["interpretation_note"]


def test_summary_rejects_unknown_schema_and_incomplete_coverage() -> None:
    report = _report()
    report["schema"] = "other"
    with pytest.raises(ValueError, match="Expected raw survey schema"):
        summarize_raw_gradient_survey(report)

    report = _report()
    report["aggregation_geometry"]["fixed"].pop()
    with pytest.raises(ValueError, match="do not cover every"):
        summarize_raw_gradient_survey(report)


def test_summary_rejects_nonfinite_and_duplicate_sample_updates() -> None:
    report = _report()
    report["samples"][0]["raw_gradient_norms"]["chem"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_raw_gradient_survey(report)

    report = _report()
    report["samples"][1]["update"] = 0
    with pytest.raises(ValueError, match="Duplicate survey update"):
        summarize_raw_gradient_survey(report)


def test_normalized_projection_excludes_zero_norm_denominators() -> None:
    report = _report()
    report["samples"][0]["raw_gradient_norms"]["chem"] = 0.0
    report["aggregation_geometry"]["fixed"][0]["task_directional_dots"]["chem"] = 0.0

    result = summarize_raw_gradient_survey(report)
    projection = result["methods"]["fixed"]["overall"]["unit_task_projections"]["chem"]
    cosine = result["methods"]["fixed"]["overall"]["joint_task_cosines"]["chem"]
    assert projection["count"] == 1
    assert projection["zero_denominator_count"] == 1
    assert cosine["count"] == 1
    assert cosine["zero_denominator_count"] == 1


def test_input_report_is_not_mutated() -> None:
    report = _report()
    original = copy.deepcopy(report)
    summarize_raw_gradient_survey(report)
    assert report == original
