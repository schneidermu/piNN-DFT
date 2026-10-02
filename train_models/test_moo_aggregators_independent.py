"""Independent small-vector checks for the MOO aggregation mathematics.

These tests intentionally construct their own reference cases instead of
reusing diagnostics or helper routines from ``moo_aggregators``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import moo_aggregators
from moo_aggregators import aggregate_task_gradients

TASKS = ("chem", "exc", "op")
DTYPE = torch.float64


def _task_gradients(
    vectors: list[list[float]], *, dtype: torch.dtype = DTYPE
) -> dict[str, dict[str, torch.Tensor]]:
    assert len(vectors) == len(TASKS)
    return {
        task: {"p": torch.tensor(vector, dtype=dtype)}
        for task, vector in zip(TASKS, vectors, strict=True)
    }


def _aggregate(vectors, method: str, hyperparameters=None, state=None):
    joint, diagnostics, new_state = aggregate_task_gradients(
        _task_gradients(vectors),
        method=method,
        hyperparameters=hyperparameters,
        state=state,
    )
    assert tuple(joint) == ("p",)
    assert joint["p"] is not None
    assert torch.isfinite(joint["p"]).all()
    return joint["p"], diagnostics, new_state


def _imtl_reference(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the IMTL-G equal-unit-projection equations directly."""
    directions = torch.nn.functional.normalize(vectors, dim=1)
    constraints = torch.stack(
        (
            vectors @ (directions[0] - directions[2]),
            vectors @ (directions[1] - directions[2]),
            torch.ones(3, dtype=vectors.dtype),
        )
    )
    alpha = torch.linalg.solve(
        constraints, torch.tensor([0.0, 0.0, 1.0], dtype=vectors.dtype)
    )
    return alpha @ vectors, alpha


def _unit_axes(scale: float = 1.0) -> list[list[float]]:
    return [[scale, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_fixed_scalarization_uses_frozen_equal_weights_and_keeps_structure():
    vectors = [[1.0, -2.0], [3.0, 4.0], [-5.0, 7.0]]
    actual, _, _ = _aggregate(
        vectors, "fixed", {"fixed_weights": [1.0, 1.0, 1.0]}
    )
    expected = torch.tensor(vectors, dtype=DTYPE).mean(dim=0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_fixed_scalarization_is_sensitive_to_a_single_task_rescaling():
    base, _, _ = _aggregate(
        _unit_axes(), "fixed", {"fixed_weights": [1.0, 1.0, 1.0]}
    )
    rescaled, _, _ = _aggregate(
        _unit_axes(1.0e6), "fixed", {"fixed_weights": [1.0, 1.0, 1.0]}
    )
    assert torch.linalg.vector_norm(rescaled - base) > 1.0e5


def test_imtl_g_equalizes_task_dot_products_for_orthogonal_unit_gradients():
    actual, _, _ = _aggregate(_unit_axes(), "imtl_g")
    dots = torch.tensor(_unit_axes(), dtype=DTYPE) @ actual
    torch.testing.assert_close(dots, torch.full((3,), 1.0 / 3.0, dtype=DTYPE))
    torch.testing.assert_close(
        actual, torch.full((3,), 1.0 / 3.0, dtype=DTYPE), rtol=1e-10, atol=1e-12
    )


def test_imtl_g_uses_raw_gradients_with_sum_alpha_one_not_a_unit_gradient_mean():
    vectors = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    actual, _, _ = _aggregate(vectors, "imtl_g")
    # With orthogonal tasks, equal projections onto unit task directions give
    # alpha_i ||g_i|| = t. The paper's sum(alpha)=1 convention yields
    # t = 1 / sum_i(1 / ||g_i||) = 6/11.
    expected = torch.full((3,), 6.0 / 11.0, dtype=DTYPE)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)


def test_imtl_g_positive_task_rescaling_preserves_signed_ray_when_scale_factor_is_positive():
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    scales = torch.tensor([1.0e6, 1.0, 1.0], dtype=DTYPE)
    base = torch.tensor(vectors, dtype=DTYPE)
    reference, diagnostics, _ = _aggregate(vectors, "imtl_g")
    actual, _, _ = _aggregate((base * scales[:, None]).tolist(), "imtl_g")
    alpha = torch.tensor(
        [diagnostics["coefficients"][task] for task in TASKS], dtype=DTYPE
    )
    signed_ray_scale = torch.sum(alpha / scales)
    assert signed_ray_scale > 0.0
    # Appendix B.1's positive rescaling law is d' = d / S, where
    # S = sum(alpha_i / k_i); the orientation is preserved only when S > 0.
    torch.testing.assert_close(
        actual, reference / signed_ray_scale, rtol=1e-8, atol=1e-10
    )
    direction_cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=0)
    assert float(direction_cosine) > 1.0 - 1e-9
    assert abs(float(torch.linalg.vector_norm(actual - reference))) > 0.1


def test_imtl_g_canonical_common_ascent_sample_can_flip_under_positive_task_rescaling():
    # A 3-D Cholesky realization of the normalized Gram matrix from survey
    # update 2: NCCE31 reaction 0, level2_mura, paired with BH. The original
    # 9,446-parameter gradients are retained in the external survey artifact
    # (SHA-256 cc7c2f0de49044dc2bed98cbd279ea871fd41aca69dc42b14cab978cdb97a2b).
    # The compressed fixture preserves their complete task Gram matrix up to
    # a common scale, including the nearly collinear exc/op pair.
    vectors = torch.tensor(
        [
            [0.03299863957113141, 0.0, 0.0],
            [-0.6862002712328036, 0.7274126667580285, 0.0],
            [-5.7631836566146256e-05, 6.962087076077482e-05, 2.206505033760477e-05],
        ],
        dtype=DTYPE,
    )
    reference, diagnostics, _ = _aggregate(vectors.tolist(), "imtl_g")
    independent_reference, independent_alpha = _imtl_reference(vectors)
    torch.testing.assert_close(
        reference, independent_reference, rtol=1e-8, atol=1e-12
    )
    alpha = torch.tensor(
        [diagnostics["coefficients"][task] for task in TASKS], dtype=DTYPE
    )
    torch.testing.assert_close(alpha, independent_alpha, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(
        alpha,
        torch.tensor([-0.0046521431, -0.0002416091, 1.0048937522], dtype=DTYPE),
        rtol=1e-6,
        atol=1e-10,
    )
    unit_dots = torch.nn.functional.normalize(vectors, dim=1) @ reference
    torch.testing.assert_close(
        unit_dots,
        torch.full((3,), -0.00004563603312, dtype=DTYPE),
        rtol=2e-7,
        atol=1e-13,
    )
    assert torch.all(unit_dots < 0.0)

    scales = torch.tensor([1.0, 1.0, 1000.0], dtype=DTYPE)
    rescaled = vectors * scales[:, None]
    actual, _, _ = _aggregate(rescaled.tolist(), "imtl_g")
    scaled_reference, _ = _imtl_reference(rescaled)
    torch.testing.assert_close(actual, scaled_reference, rtol=1e-8, atol=1e-12)
    signed_ray_scale = torch.sum(alpha / scales)
    assert float(signed_ray_scale) == pytest.approx(-0.00388885847, abs=2e-9)
    # This is the independent Appendix B.1 rescaling law. S<0, so the
    # canonical direction reverses even though every task was scaled positively.
    torch.testing.assert_close(
        actual, reference / signed_ray_scale, rtol=1e-6, atol=1e-12
    )
    cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=0)
    assert float(cosine) < -1.0 + 1e-10
    rescaled_unit_dots = torch.nn.functional.normalize(rescaled, dim=1) @ actual
    assert torch.all(rescaled_unit_dots > 0.0)


def test_imtl_g_duplicate_directions_use_a_finite_minimum_norm_solution():
    _actual, _, _ = _aggregate(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "imtl_g"
    )
    torch.testing.assert_close(
        _actual, torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE), rtol=1e-9, atol=1e-11
    )


def test_imtl_g_matches_independent_random_gram_reference():
    generator = torch.Generator(device="cpu").manual_seed(911)
    raw = torch.randn((3, 11), generator=generator, dtype=DTYPE)
    directions = torch.nn.functional.normalize(raw, dim=1)
    constraints = torch.stack(
        (
            raw @ (directions[0] - directions[2]),
            raw @ (directions[1] - directions[2]),
            torch.ones(3, dtype=DTYPE),
        )
    )
    alpha = torch.linalg.solve(
        constraints, torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    )
    expected = alpha @ raw

    actual, _, _ = _aggregate(raw.tolist(), "imtl_g")
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    task_projections = directions @ actual
    torch.testing.assert_close(
        task_projections,
        task_projections.mean().expand_as(task_projections),
        rtol=1e-8,
        atol=1e-10,
    )


def test_imtl_g_reports_exact_opposite_degeneracy_without_inventing_a_direction():
    actual, diagnostics, _ = _aggregate(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], "imtl_g"
    )
    torch.testing.assert_close(actual, torch.zeros(2, dtype=DTYPE), rtol=0.0, atol=1e-12)
    assert diagnostics["solver_status"] == "degenerate_zero_direction"


def test_imtl_g_fails_closed_for_a_zero_norm_task():
    with pytest.raises(ValueError, match="zero"):
        _aggregate([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], "imtl_g")


def test_cagrad_matches_unscaled_paper_rule_on_identical_gradients():
    actual, _, _ = _aggregate(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], "cagrad"
    )
    # Here every simplex mixture is the same vector; the stated rule is
    # g_bar + c ||g_bar|| g_w / ||g_w|| with c=0.4.
    torch.testing.assert_close(
        actual, torch.tensor([1.4, 0.0], dtype=DTYPE), rtol=1e-10, atol=1e-12
    )


def test_cagrad_simplex_solution_and_c_value_on_orthogonal_gradients():
    actual, _, _ = _aggregate(_unit_axes(), "cagrad")
    # For three orthogonal unit vectors g_bar dot g_w is constant over the
    # simplex; the norm term is minimized by its uniform point.
    torch.testing.assert_close(
        actual, torch.full((3,), 7.0 / 15.0, dtype=DTYPE), rtol=1e-9, atol=1e-11
    )


def test_cagrad_solves_a_conflicting_active_vertex_case():
    actual, diagnostics, _ = _aggregate(
        [[1.0, 0.0], [0.0, 1.0], [-0.2, -0.2]], "cagrad"
    )
    # By symmetry, write g_w=(t,t). Its 1-D objective is minimized at the
    # negative-gradient vertex (w=(0,0,1)); the paper update is then (0.16,0.16).
    torch.testing.assert_close(
        actual, torch.tensor([0.16, 0.16], dtype=DTYPE), rtol=1e-8, atol=1e-10
    )
    assert diagnostics["solver_status"] == "converged"


def test_cagrad_is_equivariant_to_common_scaling_but_depends_on_task_scaling():
    base = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    base_result, _, _ = _aggregate(base, "cagrad")
    common_scaled, _, _ = _aggregate(
        [[7.0 * x for x in vector] for vector in base], "cagrad"
    )
    torch.testing.assert_close(common_scaled, 7.0 * base_result, rtol=1e-9, atol=1e-11)
    one_task_scaled = [[1.0e6, 0.0], [0.0, 1.0], [1.0, 1.0]]
    changed, _, _ = _aggregate(one_task_scaled, "cagrad")
    assert torch.linalg.vector_norm(changed - base_result) > 1.0


def test_nash_mtl_has_closed_form_solution_for_orthogonal_unit_gradients():
    actual, diagnostics, _ = _aggregate(_unit_axes(), "nash_mtl")
    torch.testing.assert_close(
        actual, torch.ones(3, dtype=DTYPE), rtol=1e-9, atol=1e-11
    )
    # alpha_i (K alpha)_i = 1 is the Nash stationarity condition.
    assert diagnostics.get("solver_status") in {"converged", "ok", "success", None}


def test_nash_mtl_is_invariant_to_positive_task_rescaling():
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    scaled = [[1.0e6 * x for x in vectors[0]], [1.0e-6 * x for x in vectors[1]], vectors[2]]
    reference, _, _ = _aggregate(vectors, "nash_mtl")
    actual, _, _ = _aggregate(scaled, "nash_mtl")
    torch.testing.assert_close(actual, reference, rtol=1e-7, atol=1e-9)


def test_nash_mtl_handles_exact_and_near_singular_gram_matrices_without_ridge():
    duplicated, _, _ = _aggregate(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], "nash_mtl"
    )
    torch.testing.assert_close(
        duplicated, torch.tensor([3.0**0.5, 0.0], dtype=DTYPE), rtol=1e-8, atol=1e-10
    )
    near_duplicate, diagnostics, _ = _aggregate(
        [[1.0, 0.0], [1.0, 1.0e-7], [1.0, -1.0e-7]], "nash_mtl"
    )
    assert torch.isfinite(near_duplicate).all()
    residual = diagnostics.get(
        "solver_residual", diagnostics.get("stationarity_residual", diagnostics.get("residual"))
    )
    if residual is not None:
        assert float(residual) <= 1.0e-10


def test_nash_potential_line_search_resolves_captured_cuda_roundoff_fixture(monkeypatch):
    # This scaled Gram matrix was computed on CUDA for update 10 of the paired
    # F2 Nash run (IP13 reaction 6, level2_gauss_chebyshev, ClHS). The previous
    # Armijo check compared two rounded potential values near -14.78493 and
    # stalled at a residual of 1.938e-9 for the full 100-iteration budget.
    gram = [
        [3.997719848565211e-3, 5.961893389825772e-2, -3.7543648220183496e-7],
        [5.961893389825772e-2, 0.9999999999999999, -7.470547510353375e-6],
        [-3.7543648220183496e-7, -7.470547510353375e-6, 7.135745770821002e-11],
    ]
    scale = 62044.16034213791
    warm_alpha = [0.0003465894506675357, 0.0001338689526493784, 13.101441825375444]
    eigenvalues = [4.51089132e-12, 4.41731788e-4, 1.00355599]

    def captured_geometry(tasks, names, values, norms):
        del tasks, names, values, norms
        return (
            gram,
            gram,
            eigenvalues,
            float(eigenvalues[-1] / eigenvalues[0]),
            3,
            scale,
        )

    monkeypatch.setattr(moo_aggregators, "_geometry", captured_geometry)
    joint, diagnostics, _ = aggregate_task_gradients(
        _task_gradients(_unit_axes()),
        method="nash_mtl",
        hyperparameters={"solver": "newton_potential", "max_iter": 100, "tol": 1e-10},
        state={"method": "nash_mtl", "task_order": list(TASKS), "alpha": warm_alpha},
    )

    assert diagnostics["solver_iterations"] < 100
    assert diagnostics["solver_residual"] <= 1e-10
    assert all(value is not None and torch.isfinite(value).all() for value in joint.values())


def test_nash_mtl_fails_closed_for_opposite_or_zero_task_directions():
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        _aggregate([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], "nash_mtl")
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        _aggregate([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], "nash_mtl")


def test_zero_task_gradient_is_supported_by_fixed_and_cagrad_fails_closed_if_optimum_is_zero():
    vectors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    fixed, _, _ = _aggregate(vectors, "fixed", {"fixed_weights": [1.0, 1.0, 1.0]})
    torch.testing.assert_close(
        fixed, torch.tensor([1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=DTYPE)
    )
    # The CAGrad simplex optimum is the zero task gradient, so its normalized
    # direction in the paper's update is undefined and must fail closed.
    with pytest.raises(RuntimeError, match="zero weighted gradient"):
        _aggregate(vectors, "cagrad")


def test_random_reference_calls_are_deterministic_and_preserve_tensor_metadata():
    generator = torch.Generator(device="cpu").manual_seed(7129)
    vectors = torch.randn((3, 17), generator=generator, dtype=DTYPE)
    payload = _task_gradients(vectors.tolist())
    payload = {
        task: {
            "weight": gradient["p"].reshape(1, 17),
            "bias": (
                torch.tensor(1.0, dtype=DTYPE)
                if task == "chem"
                else torch.tensor(-1.0, dtype=DTYPE)
                if task == "op"
                else None
            ),
        }
        for task, gradient in payload.items()
    }
    outputs = []
    for _ in range(2):
        joint, _, _ = aggregate_task_gradients(
            payload, method="fixed", hyperparameters={"fixed_weights": [1.0, 1.0, 1.0]}
        )
        assert tuple(joint) == ("bias", "weight")
        assert joint["weight"].shape == (1, 17)
        assert joint["weight"].dtype == DTYPE
        torch.testing.assert_close(joint["bias"], torch.zeros((), dtype=DTYPE))
        outputs.append(joint["weight"].clone())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs[0].reshape(-1), vectors.mean(dim=0), rtol=0.0, atol=1e-14)


def test_pcd_many_random_updates_match_pinned_reference_across_json_resume():
    """Compare complete updates and resumed EMA state with the pinned release."""
    upstream_root = REPO_ROOT.parent / "lap_pcd_runs_20261002" / "upstream_pcd"
    if not (upstream_root / "pcd" / "optim.py").is_file():
        pytest.skip("the task's pinned external PCD clone is not available")
    upstream_text = str(upstream_root)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    try:
        from pcd.optim import PCD as ReferencePCD
    except ImportError as exc:  # pragma: no cover - clone integrity guard
        pytest.fail(f"could not import the pinned PCD reference: {exc}")

    rng = torch.Generator(device="cpu").manual_seed(80421)
    cases = 80
    split_at = cases // 2
    for dtype in (torch.float32, torch.float64):
        for tau in (0.0, 0.02, 0.1):
            reference_parameter = torch.nn.Parameter(torch.zeros(13, dtype=dtype))
            reference = ReferencePCD([reference_parameter], tau=tau)
            state = None
            for step in range(cases):
                if step % 20 == 0:
                    fixture = (
                        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                        if step % 40 == 0
                        else [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
                    )
                    rows = [row + [0.0] * 10 for row in fixture]
                else:
                    raw = torch.randn((3, 13), generator=rng, dtype=dtype)
                    task_scales = torch.pow(
                        torch.tensor(10.0, dtype=dtype),
                        torch.empty((3, 1), dtype=dtype).uniform_(-2.0, 2.0, generator=rng),
                    )
                    rows = (raw * task_scales).tolist()
                tensors = [torch.tensor(row, dtype=dtype) for row in rows]
                reference_info = reference.apply_gradients([[tensor] for tensor in tensors])
                joint, diagnostics, next_state = aggregate_task_gradients(
                    _task_gradients(rows, dtype=dtype),
                    method="pcd",
                    hyperparameters={"tau": tau},
                    state=state,
                )

                tolerance = 3e-5 if dtype == torch.float32 else 2e-10
                torch.testing.assert_close(
                    joint["p"], reference_parameter.grad,
                    rtol=tolerance, atol=tolerance * 1e-2,
                )
                assert diagnostics["active_indices"] == list(reference_info.active)
                assert diagnostics["feasible"] is reference_info.feasible
                assert diagnostics["mu"]["exc"] == pytest.approx(
                    reference_info.mu[0], rel=tolerance, abs=tolerance * 1e-2
                )
                assert diagnostics["mu"]["op"] == pytest.approx(
                    reference_info.mu[1], rel=tolerance, abs=tolerance * 1e-2
                )
                assert next_state["t"] == step + 1
                assert next_state["v"] == pytest.approx(
                    reference.normalizer.v.tolist(), rel=tolerance, abs=tolerance * 1e-2
                )
                state = next_state

                if step + 1 == split_at:
                    state = json.loads(json.dumps(state))
                    reference_state = json.loads(json.dumps(reference.state_dict()))
                    reference_parameter = torch.nn.Parameter(torch.zeros(13, dtype=dtype))
                    reference = ReferencePCD([reference_parameter], tau=tau)
                    reference.load_state_dict(reference_state)


def test_pcd_released_epsilon_is_material_for_small_first_step_gradients():
    rows = [[1.0, 0.0], [1e-6, 0.0], [0.0, 1.0]]
    _, diagnostics, state = aggregate_task_gradients(
        _task_gradients(rows), method="pcd", hyperparameters={"tau": 0.02}
    )
    # At t=1, vhat equals ||g||^2. With released eps=1e-8, the small
    # secondary stays 100x below unit normalized norm; epsilon-free scaling
    # would erase this real finite-step distinction.
    assert diagnostics["normalized_task_gradient_norms"]["exc"] == pytest.approx(
        1e-6 / (1e-12 + 1e-8) ** 0.5, rel=1e-12
    )
    assert diagnostics["normalized_task_gradient_norms"]["exc"] == pytest.approx(0.01, rel=1e-4)
    assert diagnostics["eps"] == 1e-8
    assert state["v"] == pytest.approx([1e-3, 1e-15, 1e-3])


def test_pcd_positive_secondary_rescaling_is_asymptotically_invariant():
    vectors = [[1.0, 0.0, 0.0], [-0.6, 0.8, 0.0], [0.2, -0.1, 1.0]]
    reference, base_info, _ = _aggregate(vectors, "pcd", {"tau": 0.02})
    rescaled_vectors = [vectors[0], [100.0 * x for x in vectors[1]], vectors[2]]
    actual, scaled_info, _ = _aggregate(
        rescaled_vectors, "pcd", {"tau": 0.02}
    )
    torch.testing.assert_close(actual, reference, rtol=2e-8, atol=2e-10)
    assert base_info["feasible"] is scaled_info["feasible"]
    assert base_info["active"] == scaled_info["active"]
