"""Focused regression tests for the four clean MOO gradient rules."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from train_models.moo_aggregators import aggregate_task_gradients

TASKS = ("chem", "exc", "op")


def _gradients(rows: list[list[float]], *, dtype: torch.dtype = torch.float64):
    return {
        task: {"weight": torch.tensor(row, dtype=dtype)}
        for task, row in zip(TASKS, rows)
    }


def _released_stationary_direction(
    rows: list[list[float]], *, dtype: torch.dtype, tau: float
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Small oracle for the released K=3 PCD arithmetic at this fixture.

    Preserve the reference's dot-product, NumPy solve, and tensor accumulation
    order. This matters because exact cancellation can vary by BLAS platform.
    """
    flat = [torch.tensor(row, dtype=dtype) for row in rows]
    dots = torch.stack(
        [torch.dot(flat[i], flat[j]) for i in range(3) for j in range(i, 3)]
    ).double().cpu().numpy()
    gram = np.empty((3, 3))
    offset = 0
    for i in range(3):
        for j in range(i, 3):
            gram[i, j] = gram[j, i] = dots[offset]
            offset += 1

    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    sq_norms = np.diag(gram)
    ema_v = 0.999 * np.zeros_like(sq_norms) + (1.0 - 0.999) * sq_norms
    ema_vhat = ema_v / (1.0 - 0.999**1)
    scales = 1.0 / np.sqrt(ema_vhat + 1.0e-8)
    normalized_gram = gram * np.outer(scales, scales)
    secondary_gram = normalized_gram[1:, 1:]
    rhs = tau * np.diag(secondary_gram) - normalized_gram[1:, 0]
    weights = np.array([1.0, 0.0, 0.0])
    active: tuple[int, ...] = ()
    if np.any(rhs > 0.0):
        multipliers = np.linalg.solve(secondary_gram[:1, :1], rhs[:1])
        if np.all(multipliers >= -1.0e-9):
            multipliers = np.maximum(multipliers, 0.0)
            slack = secondary_gram[:, :1] @ multipliers - rhs
            if np.all(slack >= -1.0e-9 * max(float(np.max(np.diag(normalized_gram))), np.finfo(float).tiny)):
                weights[1] = multipliers[0]
                active = (1,)

    coefficients = weights * scales
    direction = flat[0] * float(coefficients[0])
    for index in range(1, 3):
        if coefficients[index] != 0.0:
            direction.add_(flat[index], alpha=float(coefficients[index]))
    direction_norm = direction.norm()
    factor = torch.where(
        direction_norm > 0,
        float(norms[0]) / direction_norm,
        torch.zeros_like(direction_norm),
    )
    return direction.mul_(factor), active


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


def test_pcd_k3_both_active_matches_closed_form_and_rescales_to_primary_norm():
    gradients = _gradients([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
    tau = 0.2
    joint, diagnostics, state = aggregate_task_gradients(
        gradients, method="pcd", hyperparameters={"tau": tau}
    )

    primary_tilde = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    primary_tilde /= torch.sqrt(primary_tilde.square().sum() + 1.0e-8)
    secondary_scale = 1.0 / torch.sqrt(torch.tensor(1.0 + 1.0e-8, dtype=torch.float64))
    secondary_tilde = torch.eye(3, dtype=torch.float64)[1:] * secondary_scale
    # The active constraints are orthogonal, so the KKT multipliers are the
    # deficits from tau and the projection is available in closed form.
    mu_exc = tau - float(secondary_tilde[0] @ primary_tilde) / float(
        secondary_tilde[0] @ secondary_tilde[0]
    )
    mu_op = tau
    direction = primary_tilde + mu_exc * secondary_tilde[0] + mu_op * secondary_tilde[1]
    expected = direction * (torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) / direction.norm())
    torch.testing.assert_close(joint["weight"], expected, rtol=1e-11, atol=1e-11)
    assert diagnostics["active"] == ["exc", "op"]
    assert diagnostics["active_indices"] == [1, 2]
    assert diagnostics["mu"] == pytest.approx({"exc": mu_exc, "op": mu_op})
    assert diagnostics["feasible"] is True
    assert diagnostics["constraints"]["exc"]["lhs"] == pytest.approx(tau)
    assert diagnostics["constraints"]["op"]["lhs"] == pytest.approx(tau)
    assert diagnostics["pre_final_rescale_norm"] == pytest.approx(float(direction.norm()))
    assert float(torch.linalg.vector_norm(joint["weight"])) == pytest.approx(2.0**0.5)
    assert state["method"] == "pcd" and state["task_order"] == list(TASKS)


def test_pcd_inactive_zero_secondary_and_primary_fallback_are_canonical():
    inactive, inactive_info, _ = aggregate_task_gradients(
        _gradients([[1, 1, 1], [1, 0, 0], [0, 1, 0]]),
        method="pcd",
        hyperparameters={"tau": 0.1},
    )
    torch.testing.assert_close(inactive["weight"], torch.tensor([1, 1, 1], dtype=torch.float64))
    assert inactive_info["active"] == [] and inactive_info["feasible"] is True
    assert inactive_info["raw_equivalent_coefficients"] == pytest.approx(
        {"chem": 1.0, "exc": 0.0, "op": 0.0}
    )

    zero_secondary, zero_info, _ = aggregate_task_gradients(
        _gradients([[1, 0, 0], [-1, 0, 0], [0, 0, 0]]),
        method="pcd",
        hyperparameters={"tau": 0.2},
    )
    assert zero_info["feasible"] is True
    assert zero_info["constraints"]["op"]["rhs"] == 0.0
    assert zero_info["mu"]["op"] == 0.0
    assert torch.isfinite(zero_secondary["weight"]).all()

    fallback, fallback_info, _ = aggregate_task_gradients(
        _gradients([[0, 1, 0], [1, 0, 0], [-2, 0, 0]]),
        method="pcd",
        hyperparameters={"tau": 0.2},
    )
    torch.testing.assert_close(fallback["weight"], torch.tensor([0, 1, 0], dtype=torch.float64))
    assert fallback_info["feasible"] is False
    assert fallback_info["solver_status"] == "infeasible_primary_fallback"
    assert fallback_info["active"] == []
    assert fallback_info["raw_equivalent_coefficients"] == pytest.approx(
        {"chem": 1.0, "exc": 0.0, "op": 0.0}
    )


def test_pcd_primary_zero_and_qp_stationary_match_upstream_dtype_behavior():
    primary_zero, primary_info, primary_state = aggregate_task_gradients(
        _gradients([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        method="pcd",
        hyperparameters={"tau": 0.1},
    )
    assert torch.count_nonzero(primary_zero["weight"]) == 0
    assert primary_info["solver_status"] == "primary_zero"
    assert primary_info["feasible"] is True
    assert primary_state["t"] == 1
    assert primary_state["v"] == pytest.approx([0.0, 0.001, 0.001])

    qp_zero, qp_info, qp_state = aggregate_task_gradients(
        _gradients([[1, 0, 0], [-1, 0, 0], [0, 1, 0]]),
        method="pcd",
        hyperparameters={"tau": 0.0},
    )
    rows = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]
    reference_qp_zero, reference_active = _released_stationary_direction(
        rows, dtype=torch.float64, tau=0.0
    )
    torch.testing.assert_close(qp_zero["weight"], reference_qp_zero, rtol=1e-12, atol=1e-12)
    expected_status = (
        "zero_direction"
        if torch.count_nonzero(reference_qp_zero) == 0
        else "active_set"
    )
    assert qp_info["solver_status"] == expected_status
    assert qp_info["active"] == ["exc"]
    assert qp_info["active_indices"] == list(reference_active)
    assert qp_info["feasible"] is True
    assert qp_state["t"] == 1

    # The released operation order can round the sum to exact zero at float32.
    qp_zero_float32, qp_float32_info, _ = aggregate_task_gradients(
        _gradients(rows, dtype=torch.float32),
        method="pcd",
        hyperparameters={"tau": 0.0},
    )
    reference_float32, _ = _released_stationary_direction(
        rows, dtype=torch.float32, tau=0.0
    )
    torch.testing.assert_close(qp_zero_float32["weight"], reference_float32, rtol=0.0, atol=0.0)
    expected_float32_status = (
        "zero_direction"
        if torch.count_nonzero(reference_float32) == 0
        else "active_set"
    )
    assert qp_float32_info["solver_status"] == expected_float32_status
    assert qp_float32_info["active"] == ["exc"]


def test_pcd_ema_bias_correction_and_resume_state_are_task_aligned():
    first = _gradients([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    _, first_info, first_state = aggregate_task_gradients(
        first, method="pcd", hyperparameters={"tau": 0.02}
    )
    assert first_state["t"] == 1
    assert first_info["ema_squared_norms"] == pytest.approx(
        {"chem": 0.001, "exc": 0.004, "op": 0.009}
    )
    assert first_info["bias_corrected_ema_squared_norms"] == pytest.approx(
        {"chem": 1.0, "exc": 4.0, "op": 9.0}
    )

    second = _gradients([[2, 0, 0], [0, 1, 0], [0, 0, 4]])
    json_restored_state = json.loads(json.dumps(first_state))
    _, second_info, second_state = aggregate_task_gradients(
        second, method="pcd", hyperparameters={"tau": 0.02}, state=json_restored_state
    )
    assert second_state["t"] == 2
    expected_v = [0.999 * 0.001 + 0.001 * 4, 0.999 * 0.004 + 0.001, 0.999 * 0.009 + 0.001 * 16]
    assert second_state["v"] == pytest.approx(expected_v)
    assert second_info["normalization_scales"]["chem"] == pytest.approx(
        1.0 / (second_info["bias_corrected_ema_squared_norms"]["chem"] + 1.0e-8) ** 0.5
    )
    json.dumps(second_state, allow_nan=False)
    json.dumps(second_info, allow_nan=False)

    bad_state = dict(second_state, tau=0.1)
    with pytest.raises(ValueError, match="tau.*active configuration"):
        aggregate_task_gradients(
            second, method="pcd", hyperparameters={"tau": 0.02}, state=bad_state
        )
    bad_state = dict(second_state, v=[0.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite, nonnegative"):
        aggregate_task_gradients(
            second, method="pcd", hyperparameters={"tau": 0.02}, state=bad_state
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": "nash_mtl"}, "version or method"),
        ({"task_order": ["exc", "chem", "op"]}, "task order"),
        ({"beta": 0.9}, "beta.*active configuration"),
        ({"eps": 0.0}, "eps.*active configuration"),
        ({"t": -1}, "step count"),
        ({"t": True}, "step count"),
    ],
)
def test_pcd_resume_state_rejects_mismatched_configuration_and_count(changes, message):
    gradients = _gradients([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    _, _, state = aggregate_task_gradients(gradients, method="pcd")
    bad_state = dict(state, **changes)

    with pytest.raises(ValueError, match=message):
        aggregate_task_gradients(gradients, method="pcd", state=bad_state)


def test_pcd_resume_state_requires_mapping():
    gradients = _gradients([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(TypeError, match="state must be a mapping"):
        aggregate_task_gradients(gradients, method="pcd", state=object())


def test_pcd_rejects_noncanonical_tasks_and_invalid_hyperparameters():
    gradients = _gradients([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="canonical order"):
        aggregate_task_gradients(
            {"chem": gradients["chem"], "exc": gradients["exc"]}, method="pcd"
        )
    for tau in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match=r"tau in \[0, 1\]"):
            aggregate_task_gradients(
                gradients, method="pcd", hyperparameters={"tau": tau}
            )
