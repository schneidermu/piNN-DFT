"""
Constraint satisfaction and spin-symmetry tests for pcPBELMLOptimizerV2.

All constraints are structural (enforced by construction) so they pass for any
network weights.  Run with:  pytest test.py -v

Coverage
--------
1.  Output shape — always (N, 29) for all 4 ablation variants.
2.  Exchange UEG constraint      — mu = true_mu  at (sigma=0, tau=tau_TF(2rho)/2).
3.  G_x UEG constraint           — G_NN = 0      at same point (use_g_x=True only).
4.  Correlation UEG constraint   — beta = true_beta at (sigma=0, tau=tau_TF(rho)).
5.  G_c sigma_zero constraint    — G_c = 1.0     at same point (use_g_c=True only).
6.  High-density constraint      — gamma = true_gamma as rho → ∞.
7.  G_c rho_inf constraint       — G_c = 1.0     as rho → ∞ (use_g_c=True only).
8.  Pass-through constants       — indices 2–21 always equal PBE_CONSTANTS.
9.  Disabled-flag baselines      — G_NN = 0 when use_g_x=False; G_c = 1 when use_g_c=False.
10. Spin symmetry (correlation)  — beta, gamma, G_c invariant under spin swap.
11. Spin symmetry (exchange)     — mu, kappa, G_NN swap under spin swap.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals.constants import EPS_RHO, NN_OUTPUT_SCALE_PBE, PBE_CONSTANTS
from dft_functionals import PBE
from NN_models import pcPBELMLOptimizerV2

# ---------------------------------------------------------------------------
# Physical constant (must match NN_models.py)
# ---------------------------------------------------------------------------
_C_TF: float = 3.0 / 10.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)

# ---------------------------------------------------------------------------
# Output-tensor layout (indices into the (N, 29) output)
# ---------------------------------------------------------------------------
IDX_BETA      = 0
IDX_GAMMA     = 1
IDX_FILL_SLICE = slice(2, 22)   # 20 pass-through constants
IDX_KAPPA_UP  = 22
IDX_MU_UP     = 23
IDX_KAPPA_DOWN = 24
IDX_MU_DOWN   = 25
IDX_GNN_UP    = 26
IDX_GNN_DOWN  = 27
IDX_GC        = 28

# Reference values (PBE constants factored out)
_TCP = PBE_CONSTANTS[0]
TRUE_BETA  = _TCP[IDX_BETA].item()
TRUE_GAMMA = _TCP[IDX_GAMMA].item()
TRUE_MU    = _TCP[IDX_MU_UP].item()   # same for up and down

ATOL = 1e-5   # absolute tolerance for all constraint checks

# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
VARIANTS = [
    ("PBE-L",     False, False),
    ("PBE-LGx",   True,  False),
    ("PBE-LGc",   False, True),
    ("PBE-LGxGc", True,  True),
]


@pytest.fixture(
    scope="module",
    params=VARIANTS,
    ids=[v[0] for v in VARIANTS],
)
def model_info(request):
    """Yields (model, use_g_x, use_g_c).  Created once per variant per module."""
    name, use_g_x, use_g_c = request.param
    m = pcPBELMLOptimizerV2(num_layers=6, h_dim=32, use_g_x=use_g_x, use_g_c=use_g_c)
    m.eval()
    return m, use_g_x, use_g_c


# ---------------------------------------------------------------------------
# Input constructors
# ---------------------------------------------------------------------------

def _ueg_exchange_input(rho_a: torch.Tensor, rho_b: torch.Tensor) -> torch.Tensor:
    """
    UEG exchange constraint point: sigma = 0, tau = tau_TF(2*rho) / 2, lapl = 0.

    Must include EPS_RHO to match the model's internal constraint construction:
        tau_tf_2rho_a = C_TF * (2*rho_a + EPS_RHO) ** (5/3)
    """
    zeros = torch.zeros_like(rho_a)
    tau_a = _C_TF * (2.0 * rho_a + EPS_RHO) ** (5.0 / 3.0) / 2.0
    tau_b = _C_TF * (2.0 * rho_b + EPS_RHO) ** (5.0 / 3.0) / 2.0
    return torch.stack(
        [rho_a, rho_b, zeros, zeros, zeros, tau_a, tau_b, zeros, zeros], dim=1
    )


def _ueg_corr_input(rho_a: torch.Tensor, rho_b: torch.Tensor) -> torch.Tensor:
    """
    Physical spin-resolved UEG correlation point: sigma = 0, lapl = 0.

    Matches the model's internal construction:
        tau_sigma = 1/2 * C_TF * (2*rho_sigma + EPS_RHO) ** (5/3)
    """
    zeros = torch.zeros_like(rho_a)
    tau_a = 0.5 * _C_TF * (2.0 * rho_a + EPS_RHO) ** (5.0 / 3.0)
    tau_b = 0.5 * _C_TF * (2.0 * rho_b + EPS_RHO) ** (5.0 / 3.0)
    return torch.stack(
        [rho_a, rho_b, zeros, zeros, zeros, tau_a, tau_b, zeros, zeros], dim=1
    )


def _rho_inf_input(N: int = 4) -> torch.Tensor:
    """
    High-density + nonzero-gradient proxy input for the rho → ∞ constraint.

    rho = 1e4: tanh((1e4)^(1/3)) = tanh(21.5) = 1.0 in float32, so the density
    descriptor equals all_rho_inf(...) exactly.

    sigma = 1e12: gives s ≈ 0.48.  This is *required* for the G_c rho_inf test —
    without gradient, x_corr_ueg_desc (s=0, UEG) collapses to the same point as
    x_corr_rho_inf (s from real input), making the Lagrange denominator d_sq_01 = 0
    and the correction degenerate.  With nonzero sigma the two constraint points
    are distinct (d_sq_01 ≈ 1) and G_c = 1 is enforced exactly.

    tau = 2 * tau_TF: gives alpha ≈ 0.06 (nonzero), adding to d_sq_01.
    """
    rho   = torch.full((N,), 1e4)
    sigma = torch.full((N,), 1e12)
    tau   = _C_TF * (rho + EPS_RHO) ** (5.0 / 3.0) * 2.0
    lapl  = torch.zeros(N)
    return torch.stack([rho, rho, sigma, 2 * sigma, sigma, tau, tau, lapl, lapl], dim=1)


def _s_inf_input(N: int = 4) -> torch.Tensor:
    """
    Rapidly-varying limit proxy input for the G_c s -> inf constraint.

    Very small density with very large sigma drives the tanhed reduced-gradient
    descriptors to 1 while keeping the point distinct from the other G_c anchors.
    The three sigma columns are equal so log-augmented descriptor variants also
    match their all_s_inf anchor exactly.
    """
    rho = torch.full((N,), 1e-6)
    sigma = torch.full((N,), 1e12)
    tau = _C_TF * (rho + EPS_RHO) ** (5.0 / 3.0) * 2.0
    lapl = torch.zeros(N)
    return torch.stack([rho, rho, sigma, sigma, sigma, tau, tau, lapl, lapl], dim=1)


def _random_input(N: int = 8, seed: int = 42) -> torch.Tensor:
    """Physically plausible random input (N, 9)."""
    torch.manual_seed(seed)
    rho_a = torch.rand(N) * 0.5 + 0.01
    rho_b = torch.rand(N) * 0.3 + 0.01
    sigma_a   = torch.rand(N) * 0.1
    sigma_b   = torch.rand(N) * 0.1
    sigma_tot = sigma_a + sigma_b + torch.rand(N) * 0.05
    tau_a = _C_TF * rho_a ** (5.0 / 3.0) * (1.0 + torch.rand(N) * 0.3)
    tau_b = _C_TF * rho_b ** (5.0 / 3.0) * (1.0 + torch.rand(N) * 0.3)
    lapl_a = torch.rand(N) * 0.01
    lapl_b = torch.rand(N) * 0.01
    return torch.stack(
        [rho_a, rho_b, sigma_a, sigma_tot, sigma_b, tau_a, tau_b, lapl_a, lapl_b], dim=1
    )


def _spin_swap(x: torch.Tensor) -> torch.Tensor:
    """
    Swap spin-up and spin-down channels.

    Input columns: [rho_a, rho_b, sigma_aa, sigma_tot, sigma_bb, tau_a, tau_b, lapl_a, lapl_b]
    Swapped:       [rho_b, rho_a, sigma_bb, sigma_tot, sigma_aa, tau_b, tau_a, lapl_b, lapl_a]
    """
    return x[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]]


# ---------------------------------------------------------------------------
# Shared density batches used across multiple tests
# ---------------------------------------------------------------------------
_RHO_A = torch.tensor([0.05, 0.1, 0.5, 1.0, 2.0])
_RHO_B = torch.tensor([0.05, 0.1, 0.3, 0.8, 1.5])
_N     = len(_RHO_A)

_X_UEG_EXCH = _ueg_exchange_input(_RHO_A, _RHO_B)
_X_UEG_CORR = _ueg_corr_input(_RHO_A, _RHO_B)
_X_RHO_INF  = _rho_inf_input(_N)
_X_S_INF    = _s_inf_input(_N)
_X_RAND     = _random_input(8)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

def test_output_shape(model_info):
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_RAND)
    assert out.shape == (8, 29), f"Expected (8, 29), got {out.shape}"


# ---------------------------------------------------------------------------
# 2–3. Exchange UEG constraint
# ---------------------------------------------------------------------------

def test_mu_up_at_ueg_exchange(model_info):
    """mu_up factor = 1 → output[IDX_MU_UP] = true_mu at UEG exchange point."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_UEG_EXCH)
    torch.testing.assert_close(
        out[:, IDX_MU_UP], torch.full((_N,), TRUE_MU), atol=ATOL, rtol=0,
        msg="mu_up must equal true_mu at the UEG exchange constraint point",
    )


def test_mu_down_at_ueg_exchange(model_info):
    """mu_down factor = 1 → output[IDX_MU_DOWN] = true_mu at UEG exchange point."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_UEG_EXCH)
    torch.testing.assert_close(
        out[:, IDX_MU_DOWN], torch.full((_N,), TRUE_MU), atol=ATOL, rtol=0,
        msg="mu_down must equal true_mu at the UEG exchange constraint point",
    )


def test_gnn_up_zero_at_ueg_exchange(model_info):
    """G_NN_up = tanh(0) * 1 = 0 at UEG exchange point (use_g_x only)."""
    m, use_g_x, _ = model_info
    if not use_g_x:
        pytest.skip("use_g_x=False — G_NN is a fixed additive baseline (0.0)")
    with torch.no_grad():
        out = m(_X_UEG_EXCH)
    torch.testing.assert_close(
        out[:, IDX_GNN_UP], torch.zeros(_N), atol=ATOL, rtol=0,
        msg="G_NN_up must be 0 at the UEG exchange constraint point",
    )


def test_gnn_down_zero_at_ueg_exchange(model_info):
    """G_NN_down = tanh(0) * 1 = 0 at UEG exchange point (use_g_x only)."""
    m, use_g_x, _ = model_info
    if not use_g_x:
        pytest.skip("use_g_x=False — G_NN is a fixed additive baseline (0.0)")
    with torch.no_grad():
        out = m(_X_UEG_EXCH)
    torch.testing.assert_close(
        out[:, IDX_GNN_DOWN], torch.zeros(_N), atol=ATOL, rtol=0,
        msg="G_NN_down must be 0 at the UEG exchange constraint point",
    )


# ---------------------------------------------------------------------------
# 4–5. Correlation UEG constraint (sigma=0, tau=tau_TF)
# ---------------------------------------------------------------------------

def test_beta_at_ueg_corr(model_info):
    """beta factor = 1 → true beta at the physical spin-resolved UEG point."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_UEG_CORR)
    torch.testing.assert_close(
        out[:, IDX_BETA], torch.full((_N,), TRUE_BETA), atol=ATOL, rtol=0,
        msg="beta must equal true_beta at the UEG correlation constraint point",
    )


def test_gc_one_at_sigma_zero(model_info):
    """G_c = 1 at physical spin-resolved UEG via Lagrange correction."""
    m, _, use_g_c = model_info
    if not use_g_c:
        pytest.skip("use_g_c=False — G_c is a fixed baseline (1.0)")
    with torch.no_grad():
        out = m(_X_UEG_CORR)
    torch.testing.assert_close(
        out[:, IDX_GC], torch.ones(_N), atol=ATOL, rtol=0,
        msg="G_c must be 1 at the sigma=0 (UEG correlation) constraint point",
    )


# ---------------------------------------------------------------------------
# 6–7. High-density constraint (rho → ∞)
# ---------------------------------------------------------------------------

def test_gamma_at_rho_inf(model_info):
    """gamma factor = 1 → output[IDX_GAMMA] = true_gamma as rho → ∞."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_RHO_INF)
    torch.testing.assert_close(
        out[:, IDX_GAMMA], torch.full((_N,), TRUE_GAMMA), atol=ATOL, rtol=0,
        msg="gamma must equal true_gamma at the high-density limit",
    )


def test_gc_one_at_rho_inf(model_info):
    """G_c = 1.0 at rho → ∞ via Lagrange correction (use_g_c only)."""
    m, _, use_g_c = model_info
    if not use_g_c:
        pytest.skip("use_g_c=False — G_c is a fixed baseline (1.0)")
    with torch.no_grad():
        out = m(_X_RHO_INF)
    torch.testing.assert_close(
        out[:, IDX_GC], torch.ones(_N), atol=ATOL, rtol=0,
        msg="G_c must be 1 at the high-density (rho → ∞) constraint point",
    )


def test_gc_one_at_s_inf(model_info):
    """G_c = 1.0 at the rapidly-varying limit proxy where s -> ∞ (use_g_c only)."""
    m, _, use_g_c = model_info
    if not use_g_c:
        pytest.skip("use_g_c=False — G_c is a fixed baseline (1.0)")
    with torch.no_grad():
        out = m(_X_S_INF)
    torch.testing.assert_close(
        out[:, IDX_GC], torch.ones(_N), atol=ATOL, rtol=0,
        msg="G_c must equal 1 at the s -> ∞ constraint point",
    )


# ---------------------------------------------------------------------------
# 8. Pass-through constants (indices 2–21)
# ---------------------------------------------------------------------------

def test_fill_constants_unchanged(model_info):
    """Indices 2–21 must always equal PBE_CONSTANTS (factor = 1.0)."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_RAND)
    expected = _TCP[IDX_FILL_SLICE].expand(8, -1)
    torch.testing.assert_close(
        out[:, IDX_FILL_SLICE], expected, atol=ATOL, rtol=0,
        msg="Indices 2–21 are pass-through and must not change",
    )


def test_physical_spin_ueg_anchor_is_used():
    """Independently build physical tau and verify beta/Gc constraints there."""
    rho_a = torch.tensor([0.07, 0.4, 1.3], dtype=torch.float64)
    rho_b = torch.tensor([0.11, 0.8, 0.6], dtype=torch.float64)
    zeros = torch.zeros_like(rho_a)
    tau_a = 0.5 * _C_TF * (2.0 * rho_a + EPS_RHO) ** (5.0 / 3.0)
    tau_b = 0.5 * _C_TF * (2.0 * rho_b + EPS_RHO) ** (5.0 / 3.0)
    physical_ueg = torch.stack(
        [rho_a, rho_b, zeros, zeros, zeros, tau_a, tau_b, zeros, zeros], dim=1
    )

    # Independently check the spin-resolved Fermi-gas coefficient.
    ratio_a = tau_a / (_C_TF * (rho_a + EPS_RHO) ** (5.0 / 3.0))
    ratio_b = tau_b / (_C_TF * (rho_b + EPS_RHO) ** (5.0 / 3.0))
    expected_ratio_a = ((2.0 * rho_a + EPS_RHO) / (rho_a + EPS_RHO)) ** (5.0 / 3.0) / 2.0
    expected_ratio_b = ((2.0 * rho_b + EPS_RHO) / (rho_b + EPS_RHO)) ** (5.0 / 3.0) / 2.0
    torch.testing.assert_close(ratio_a, expected_ratio_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ratio_b, expected_ratio_b, rtol=1e-12, atol=1e-12)

    model = pcPBELMLOptimizerV2(
        num_layers=6, h_dim=32, use_g_x=False, use_g_c=True
    ).double().eval()
    descriptors = model.get_density_descriptors(physical_ueg)
    descriptor_ratio_a = tau_a / (
        _C_TF * (rho_a + EPS_RHO) ** (5.0 / 3.0) + EPS_RHO
    )
    descriptor_ratio_b = tau_b / (
        _C_TF * (rho_b + EPS_RHO) ** (5.0 / 3.0) + EPS_RHO
    )
    expected_alpha_a = torch.tanh(descriptor_ratio_a - 1.0)
    expected_alpha_b = torch.tanh(descriptor_ratio_b - 1.0)
    torch.testing.assert_close(descriptors[:, 5], expected_alpha_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(descriptors[:, 6], expected_alpha_b, rtol=1e-12, atol=1e-12)

    with torch.no_grad():
        output = model(physical_ueg)
    torch.testing.assert_close(
        output[:, IDX_BETA],
        torch.full_like(rho_a, float(PBE_CONSTANTS[0, IDX_BETA])),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(output[:, IDX_GC], torch.ones_like(rho_a), rtol=0, atol=1e-12)


def test_canonical_pbe_constants_and_nn_output_scales_are_separate():
    torch.testing.assert_close(PBE_CONSTANTS[:, 26:28], torch.zeros_like(PBE_CONSTANTS[:, 26:28]))
    torch.testing.assert_close(PBE_CONSTANTS[:, 28], torch.ones_like(PBE_CONSTANTS[:, 28]))
    torch.testing.assert_close(
        NN_OUTPUT_SCALE_PBE[:, 26:28], torch.ones_like(NN_OUTPUT_SCALE_PBE[:, 26:28])
    )


def test_learned_gnn_outputs_remain_live_with_canonical_pbe_constants():
    torch.manual_seed(20260929)
    model = pcPBELMLOptimizerV2(
        num_layers=6, h_dim=32, use_g_x=True, use_g_c=False
    ).eval()
    with torch.no_grad():
        output = model(_X_RAND)
    assert torch.max(torch.abs(output[:, IDX_GNN_UP:IDX_GNN_DOWN + 1])) > 1e-8


def test_canonical_pbe_exchange_has_unity_enhancement_at_zero_gradient():
    constants = PBE_CONSTANTS.expand(3, -1)
    spin_up_constants = constants[:, list(range(22)) + [22, 23, 26]]
    fx = PBE.pbe_f0(torch.zeros(3, dtype=torch.float64), spin_up_constants)
    torch.testing.assert_close(fx, torch.ones_like(fx), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 9. Disabled-flag baselines
# ---------------------------------------------------------------------------

def test_gnn_is_zero_when_disabled(model_info):
    """When use_g_x=False, G_NN_up and G_NN_down must be 0.0 everywhere."""
    m, use_g_x, _ = model_info
    if use_g_x:
        pytest.skip("use_g_x=True — G_NN is a learned quantity")
    with torch.no_grad():
        out = m(_X_RAND)
    torch.testing.assert_close(out[:, IDX_GNN_UP], torch.zeros(8), atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_GNN_DOWN], torch.zeros(8), atol=ATOL, rtol=0)


def test_gc_is_one_when_disabled(model_info):
    """When use_g_c=False, G_c must be 1.0 everywhere."""
    m, _, use_g_c = model_info
    if use_g_c:
        pytest.skip("use_g_c=True — G_c is a learned quantity")
    with torch.no_grad():
        out = m(_X_RAND)
    torch.testing.assert_close(out[:, IDX_GC], torch.ones(8), atol=ATOL, rtol=0)


def _gc_probe(x: torch.Tensor) -> torch.Tensor:
    return 0.3 * x.square().sum(dim=1, keepdim=True) + 0.2 * x.sum(dim=1, keepdim=True)


def _gc_interpolate(x_real, x1, x2, x3):
    return pcPBELMLOptimizerV2._lagrange_correct_Gc(
        _gc_probe(x_real),
        _gc_probe(x1),
        _gc_probe(x2),
        _gc_probe(x3),
        x_real,
        x1,
        x2,
        x3,
    )


def test_gc_cross_weights_match_old_interpolation_away_from_degeneracy():
    torch.manual_seed(20260929)
    dtype = torch.float64
    x_real = torch.randn(64, 7, dtype=dtype)
    x1 = torch.randn(64, 7, dtype=dtype) + 2.0
    x2 = torch.randn(64, 7, dtype=dtype) - 2.0
    x3 = torch.randn(64, 7, dtype=dtype) + torch.tensor([1., -1., 2., -2., 3., -3., 0.], dtype=dtype)
    g_real, g1, g2, g3 = (_gc_probe(x) for x in (x_real, x1, x2, x3))

    def dist(a, b):
        return torch.tanh((a - b).square().sum(dim=1, keepdim=True))

    d0, d1, d2 = (dist(x_real, x) for x in (x1, x2, x3))
    d01, d02, d12 = dist(x1, x2), dist(x1, x3), dist(x2, x3)
    c0 = d1 * d2 / (d01 * d02)
    c1 = d0 * d2 / (d01 * d12)
    c2 = d0 * d1 / (d02 * d12)
    old = (
        c0 * (g_real - g1 + 1.0)
        + c1 * (g_real - g2 + 1.0)
        + c2 * (g_real - g3 + 1.0)
    ) / (c0 + c1 + c2)

    actual = pcPBELMLOptimizerV2._lagrange_correct_Gc(
        g_real, g1, g2, g3, x_real, x1, x2, x3
    )
    torch.testing.assert_close(actual, old, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_gc_is_exactly_one_at_each_anchor(anchor):
    dtype = torch.float64
    x1 = torch.tensor([[0.0, 0.0]], dtype=dtype)
    x2 = torch.tensor([[1.0, 0.0]], dtype=dtype)
    x3 = torch.tensor([[0.0, 1.0]], dtype=dtype)
    points = [x1, x2, x3]
    result = _gc_interpolate(points[anchor], x1, x2, x3)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_gc_pairwise_constraint_intersections_are_finite_and_exact(pair):
    dtype = torch.float64
    points = [
        torch.tensor([[0.0, 0.0]], dtype=dtype),
        torch.tensor([[1.0, 0.0]], dtype=dtype),
        torch.tensor([[0.0, 1.0]], dtype=dtype),
    ]
    points[pair[1]] = points[pair[0]].clone()
    x_real = points[pair[0]].clone()
    result = _gc_interpolate(x_real, *points)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


def test_gc_triple_constraint_intersection_is_finite_and_exact():
    x_real = torch.tensor([[0.4, -0.2, 0.8]], dtype=torch.float64)
    result = _gc_interpolate(x_real, x_real.clone(), x_real.clone(), x_real.clone())
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


def test_gc_first_derivatives_remain_finite_near_degenerate_intersection():
    dtype = torch.float64
    derivative_norms = []
    for scale in (1e-2, 1e-4, 1e-6):
        x_real = torch.tensor([[3.0, 4.0]], dtype=dtype, requires_grad=True) * scale
        x1 = torch.tensor([[0.0, 0.0]], dtype=dtype, requires_grad=True) * scale
        x2 = torch.tensor([[1.0, 0.0]], dtype=dtype, requires_grad=True) * scale
        x3 = torch.tensor([[0.0, 1.0]], dtype=dtype, requires_grad=True) * scale
        result = _gc_interpolate(x_real, x1, x2, x3)
        grads = torch.autograd.grad(result.sum(), (x_real, x1, x2, x3))
        assert torch.isfinite(result).all()
        assert all(torch.isfinite(grad).all() for grad in grads)
        derivative_norms.append(max(grad.norm().item() for grad in grads))

    assert max(derivative_norms) < 1e3, derivative_norms


# ---------------------------------------------------------------------------
# 10. Spin symmetry — correlation branch
# ---------------------------------------------------------------------------

def test_beta_spin_symmetric(model_info):
    """beta must be invariant under spin-channel swap."""
    m, _, _ = model_info
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_BETA], out_swap[:, IDX_BETA], atol=ATOL, rtol=0)


def test_gamma_spin_symmetric(model_info):
    """gamma must be invariant under spin-channel swap."""
    m, _, _ = model_info
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_GAMMA], out_swap[:, IDX_GAMMA], atol=ATOL, rtol=0)


def test_gc_spin_symmetric(model_info):
    """G_c must be invariant under spin-channel swap (use_g_c only)."""
    m, _, use_g_c = model_info
    if not use_g_c:
        pytest.skip("use_g_c=False — G_c is a fixed 1.0")
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_GC], out_swap[:, IDX_GC], atol=ATOL, rtol=0)


# ---------------------------------------------------------------------------
# 11. Spin symmetry — exchange branch (weight-tied, outputs swap)
# ---------------------------------------------------------------------------

def test_mu_swaps_under_spin_exchange(model_info):
    """mu_up(x) == mu_down(x_swap) and mu_down(x) == mu_up(x_swap)."""
    m, _, _ = model_info
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_MU_UP],   out_swap[:, IDX_MU_DOWN],   atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_MU_DOWN],  out_swap[:, IDX_MU_UP],    atol=ATOL, rtol=0)


def test_kappa_swaps_under_spin_exchange(model_info):
    """kappa_up(x) == kappa_down(x_swap) and vice versa."""
    m, _, _ = model_info
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_KAPPA_UP],   out_swap[:, IDX_KAPPA_DOWN], atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_KAPPA_DOWN],  out_swap[:, IDX_KAPPA_UP],  atol=ATOL, rtol=0)


def test_gnn_swaps_under_spin_exchange(model_info):
    """G_NN_up(x) == G_NN_down(x_swap) and vice versa (use_g_x only)."""
    m, use_g_x, _ = model_info
    if not use_g_x:
        pytest.skip("use_g_x=False — G_NN is a fixed additive 0.0 (swap is trivially satisfied)")
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_GNN_UP],  out_swap[:, IDX_GNN_DOWN], atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_GNN_DOWN], out_swap[:, IDX_GNN_UP],  atol=ATOL, rtol=0)
