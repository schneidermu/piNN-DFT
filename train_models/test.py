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
8.  Pass-through constants       — indices 2–21 always equal true_constants_PBE.
9.  Disabled-flag baselines      — G_NN = 1 when use_g_x=False; G_c = 1 when use_g_c=False.
10. Spin symmetry (correlation)  — beta, gamma, G_c invariant under spin swap.
11. Spin symmetry (exchange)     — mu, kappa, G_NN swap under spin swap.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals.constants import EPS_RHO, true_constants_PBE
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

# Reference values (true_constants_PBE factored out)
_TCP = true_constants_PBE[0]
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
    UEG correlation constraint point: sigma = 0, tau = tau_TF(rho), lapl = 0.

    Matches the model's internal construction:
        tau_tf_rho_a = C_TF * (rho_a + EPS_RHO) ** (5/3)
    """
    zeros = torch.zeros_like(rho_a)
    tau_a = _C_TF * (rho_a + EPS_RHO) ** (5.0 / 3.0)
    tau_b = _C_TF * (rho_b + EPS_RHO) ** (5.0 / 3.0)
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
        pytest.skip("use_g_x=False — G_NN is a fixed baseline (1.0)")
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
        pytest.skip("use_g_x=False — G_NN is a fixed baseline (1.0)")
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
    """beta factor = 1 → output[IDX_BETA] = true_beta at UEG correlation point."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_UEG_CORR)
    torch.testing.assert_close(
        out[:, IDX_BETA], torch.full((_N,), TRUE_BETA), atol=ATOL, rtol=0,
        msg="beta must equal true_beta at the UEG correlation constraint point",
    )


def test_gc_one_at_sigma_zero(model_info):
    """G_c = 1.0 at UEG correlation point (sigma=0) via Lagrange correction (use_g_c only)."""
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


# ---------------------------------------------------------------------------
# 8. Pass-through constants (indices 2–21)
# ---------------------------------------------------------------------------

def test_fill_constants_unchanged(model_info):
    """Indices 2–21 must always equal true_constants_PBE (factor = 1.0)."""
    m, _, _ = model_info
    with torch.no_grad():
        out = m(_X_RAND)
    expected = _TCP[IDX_FILL_SLICE].expand(8, -1)
    torch.testing.assert_close(
        out[:, IDX_FILL_SLICE], expected, atol=ATOL, rtol=0,
        msg="Indices 2–21 are pass-through and must not change",
    )


# ---------------------------------------------------------------------------
# 9. Disabled-flag baselines
# ---------------------------------------------------------------------------

def test_gnn_is_one_when_disabled(model_info):
    """When use_g_x=False, G_NN_up and G_NN_down must be 1.0 everywhere."""
    m, use_g_x, _ = model_info
    if use_g_x:
        pytest.skip("use_g_x=True — G_NN is a learned quantity")
    with torch.no_grad():
        out = m(_X_RAND)
    torch.testing.assert_close(out[:, IDX_GNN_UP],   torch.ones(8), atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_GNN_DOWN],  torch.ones(8), atol=ATOL, rtol=0)


def test_gc_is_one_when_disabled(model_info):
    """When use_g_c=False, G_c must be 1.0 everywhere."""
    m, _, use_g_c = model_info
    if use_g_c:
        pytest.skip("use_g_c=True — G_c is a learned quantity")
    with torch.no_grad():
        out = m(_X_RAND)
    torch.testing.assert_close(out[:, IDX_GC], torch.ones(8), atol=ATOL, rtol=0)


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
        pytest.skip("use_g_x=False — G_NN is a fixed 1.0 (swap is trivially satisfied)")
    x_swap = _spin_swap(_X_RAND)
    with torch.no_grad():
        out      = m(_X_RAND)
        out_swap = m(x_swap)
    torch.testing.assert_close(out[:, IDX_GNN_UP],  out_swap[:, IDX_GNN_DOWN], atol=ATOL, rtol=0)
    torch.testing.assert_close(out[:, IDX_GNN_DOWN], out_swap[:, IDX_GNN_UP],  atol=ATOL, rtol=0)
