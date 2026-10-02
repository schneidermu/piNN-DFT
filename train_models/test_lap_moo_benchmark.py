"""Focused tests for raw-gradient survey reporting and fixed calibration."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lap_moo_protocol import TASK_NAMES
from run_lap_moo_benchmark import (
    _calibration_weights,
    _method_diagnostics,
    _pairwise_cosines,
)


def test_fixed_calibration_uses_geomean_of_median_raw_task_norms():
    rows = [
        {"chem": 1.0, "exc": 2.0, "op": 8.0},
        {"chem": 3.0, "exc": 4.0, "op": 10.0},
        {"chem": 2.0, "exc": 6.0, "op": 12.0},
    ]
    medians, weights = _calibration_weights(rows)
    expected_medians = {"chem": 2.0, "exc": 4.0, "op": 10.0}
    geometric_mean = math.prod(expected_medians.values()) ** (1.0 / 3.0)
    assert medians == expected_medians
    for task in TASK_NAMES:
        assert weights[task] == pytest.approx(geometric_mean / expected_medians[task])
    assert math.prod(weights.values()) == pytest.approx(1.0)


def test_survey_records_raw_cosines_and_all_method_geometry():
    gradients = {
        "chem": {"p": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        "exc": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
        "op": {"p": torch.tensor([1.0, 1.0], dtype=torch.float64)},
    }
    assert _pairwise_cosines({task: gradient["p"] for task, gradient in gradients.items()}) == {
        "chem:exc": 0.0,
        "chem:op": pytest.approx(1.0 / math.sqrt(2.0)),
        "exc:op": pytest.approx(1.0 / math.sqrt(2.0)),
    }
    results = _method_diagnostics(
        [{"update": 0, "gradients": gradients}],
        {"chem": 2.0 ** (-1.0 / 3.0), "exc": 2.0 ** (-1.0 / 3.0), "op": 2.0 ** (2.0 / 3.0)},
    )
    assert set(results) == {"fixed", "imtl_g", "cagrad", "nash_mtl"}
    for method, rows in results.items():
        assert len(rows) == 1, method
        assert "error" not in rows[0], rows[0]
        assert set(rows[0]["coefficients"]) == set(TASK_NAMES)
        assert set(rows[0]["task_directional_dots"]) == set(TASK_NAMES)
        assert math.isfinite(rows[0]["joint_gradient_norm"])
