"""Focused regression tests for the four clean MOO gradient rules."""

from __future__ import annotations

import json

import pytest
import torch

from train_models.moo_aggregators import aggregate_task_gradients

TASKS = ("chem", "exc", "op")


def _gradients(rows: list[list[float]], *, dtype: torch.dtype = torch.float64):
    return {
        task: {"weight": torch.tensor(row, dtype=dtype)}
        for task, row in zip(TASKS, rows)
    }


def test_fixed_uses_frozen_weights_and_retains_none_geometry():
    gradients = _gradients([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    gradients["chem"]["unused"] = None
    gradients["exc"]["unused"] = None
    gradients["op"]["unused"] = None
    joint, diagnostics, state = aggregate_task_gradients(
        gradients,
        method="fixed",
        hyperparameters={"fixed_weights": [1.0, 2.0, 3.0]},
    )
    torch.testing.assert_close(joint["weight"], torch.tensor([1 / 3, 4 / 3, 3], dtype=torch.float64))
    assert joint["unused"] is None
    assert diagnostics["coefficients"] == {"chem": 1 / 3, "exc": 2 / 3, "op": 1.0}
    assert diagnostics["task_order"] == list(TASKS)
    assert state == {}
    json.dumps(diagnostics, allow_nan=False)


def test_imtl_g_uses_raw_gradient_sum_alpha_magnitude():
    gradients = _gradients([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    joint, diagnostics, _ = aggregate_task_gradients(gradients, method="imtl_g")
    scale = 1.0 / (1.0 + 0.5 + 1.0 / 3.0)
    torch.testing.assert_close(
        joint["weight"], torch.full((3,), scale, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    assert diagnostics["coefficients"] == pytest.approx(
        {"chem": scale, "exc": scale / 2.0, "op": scale / 3.0}
    )
    assert diagnostics["solver_status"] == "converged"


def test_cagrad_returns_paper_unscaled_direction():
    gradients = _gradients([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    joint, diagnostics, _ = aggregate_task_gradients(
        gradients, method="cagrad", hyperparameters={"c": 0.4, "rescale": "paper_unscaled"}
    )
    torch.testing.assert_close(
        joint["weight"], torch.full((3,), 1.4 / 3.0, dtype=torch.float64),
        atol=1e-10, rtol=1e-10,
    )
    assert diagnostics["coefficients"] == pytest.approx({task: 1.4 / 3.0 for task in TASKS})
    assert diagnostics["solver_status"] == "converged"


def test_nash_orthogonal_solution_and_json_warm_state():
    gradients = _gradients([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    joint, diagnostics, state = aggregate_task_gradients(gradients, method="nash_mtl")
    torch.testing.assert_close(joint["weight"], torch.ones(3, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    assert diagnostics["coefficients"] == pytest.approx({task: 1.0 for task in TASKS})
    assert diagnostics["solver_residual"] <= 1e-10
    assert state["method"] == "nash_mtl"
    json.dumps(state, allow_nan=False)
    warm_joint, warm_diagnostics, warm_state = aggregate_task_gradients(
        gradients, method="nash_mtl", state=state
    )
    torch.testing.assert_close(warm_joint["weight"], joint["weight"])
    assert warm_diagnostics["solver_residual"] <= 1e-10
    assert warm_state == state


def test_nash_handles_six_order_norm_span_without_gram_ridge():
    gradients = _gradients([[1, 0, 0], [0, 1e3, 0], [0, 0, 1e6]])
    joint, diagnostics, _ = aggregate_task_gradients(gradients, method="nash_mtl")
    torch.testing.assert_close(joint["weight"], torch.ones(3, dtype=torch.float64), atol=2e-7, rtol=2e-7)
    assert diagnostics["solver_residual"] <= 1e-10
    assert diagnostics["solver_status"] == "converged"


def test_normalized_methods_fail_closed_for_zero_task_gradient():
    gradients = _gradients([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="nonzero task gradients"):
        aggregate_task_gradients(gradients, method="imtl_g")
    with pytest.raises(ValueError, match="unbounded for zero task gradients"):
        aggregate_task_gradients(gradients, method="nash_mtl")
