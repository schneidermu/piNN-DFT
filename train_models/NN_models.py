"""
Neural network models for physics-constrained PBE functional optimization.

The sole model exported by this module is `pcPBELMLOptimizerV2`, which learns
to output locally-modified PBE parameters as a function of LLMGGA + zeta
density descriptors, subject to exact physical constraints.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals import true_constants_PBE
from dft_functionals.constants import (
    BETA_CORR_INDEX,
    EPS_RHO,
    EPS_SIGMA,
    GAMMA_CORR_INDEX,
    KAPPA_EX_INDEX,
    LAPL_ALPHA_INDEX,
    LAPL_BETA_INDEX,
    LLMGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY,
    LLMGGA_SPIN_INVERTED_SLICE,
    LLMGGA_SPIN_SCALING_MULTIPLIER,
    LLMGGA_ZETA_DESCRIPTOR_DIMENSIONALITY,
    MU_EX_INDEX,
    RHO_ALPHA_INDEX,
    RHO_BETA_INDEX,
    S_ALPHA_INDEX,
    S_BETA_INDEX,
    S_TOTAL_INDEX,
    TAU_ALPHA_INDEX,
    TAU_BETA_INDEX,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# (3π²)^(1/3): prefactor in Fermi wavevector kF = _C_FERMI · ρ^(1/3)
_C_FERMI: float = (3.0 * np.pi**2) ** (1.0 / 3.0)

# Thomas-Fermi KE density: τ_TF = _C_TF · ρ^(5/3)
_C_TF: float = 3.0 / 10.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)

# Reduced Laplacian normalization: q = ∇²ρ / (_C_LAPL · ρ^(5/3))
_C_LAPL: float = 4.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)

# ---------------------------------------------------------------------------
# Architectural constants
# ---------------------------------------------------------------------------

# Number of pass-through PBE constants at indices 2–21 of the 26-element output
_N_FILL_CONSTANTS: int = 20

# ---------------------------------------------------------------------------
# Activation scales
# ---------------------------------------------------------------------------

_KAPPA_SCALE: float = 4.0   # kappa_activation: sigmoid(scale · (x + 0.5))
_BETA_SCALE: float = 8.0    # beta_activation:  sigmoid(scale · x)
_BETA_SHIFT: float = 1.5    # beta_activation:  (sigmoid(...) + shift) / 2  → (0.75, 1.25)

# ---------------------------------------------------------------------------
# Log-scale parameter initializations
# ---------------------------------------------------------------------------

_LOG_SCALE_RHO_INIT: float = 0.0
_LOG_SCALE_SIGMA_INIT: float = 0.0
_LOG_SCALE_TAU_INIT: float = 0.0
_LOG_SCALE_LAPL_INIT: float = 0.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with two FC layers, each followed by LayerNorm and GELU.

    The residual connection is applied before the final activation, with optional
    dropout for regularization.

    Args:
        h_dim: Hidden dimension (input and output sizes are equal).
        dropout: Dropout probability applied after the residual addition.
    """

    def __init__(self, h_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        out = self.fc(x)
        out = self.dropout(out + residue)
        return self.activation(out)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class pcPBELMLOptimizerV2(nn.Module):
    """
    Physics-constrained PBE functional optimizer using LLMGGA + zeta descriptors.

    The model learns to output 26 modified PBE constants as a function of local
    density information. Constraints are enforced via a *difference construction*:
    each constrained parameter p is computed as

        p = activation(p_real - p_at_constraint_point)

    which guarantees that p evaluates to the physical reference value at the
    constraint point, independent of network weights.

    Architecture
    ------------
    Exchange branch:
        Separate spin-up / spin-down paths sharing weights (weight-tied).
        Input: 3-dim LLMGGA exchange descriptors per spin channel (s, α, q).
        Output: (mu, kappa) per spin channel.

    Correlation branch:
        Input: 10-dim LLMGGA + zeta descriptors.
        Spin symmetrization is performed within the hidden layers by averaging
        the representation of the input and its spin-swapped counterpart.
        Output is fused with the symmetrized exchange hidden representation to
        predict (beta, gamma).

    Constraints enforced
    --------------------
    - mu:    shifted_elu(mu_real    - mu_UEG)         → 1 at UEG limit (s→0, τ→τ_TF)
    - beta:  beta_activation(beta_real - beta_UEG)    → 1 at UEG limit (s→0)
    - gamma: shifted_elu(gamma_real - gamma_rho_inf)  → 1 at high-density limit (ρ→∞)
    - kappa: kappa_activation(kappa_real)             → bounded in (0, 1)

    Learnable log-scale parameters (log_scale_rho, log_scale_sigma, log_scale_tau,
    log_scale_lapl) control descriptor normalization; they are initialized to
    physically motivated values and are frozen during the pre-optimization phase.

    Output
    ------
    Tensor of shape (N, 28) equal to predicted_factors * true_constants_PBE.
    The 20 pass-through constants at indices 2–21 are returned unchanged.

    Args:
        num_layers: Total number of ResBlock layers. Exchange gets
            (num_layers // 2 - 1) blocks; correlation gets num_symm_blocks
            symmetrization blocks plus (num_layers // 2 - 1 - num_symm_blocks)
            post-symmetrization blocks.
        h_dim: Hidden dimension for all linear and residual layers.
        nconstants_x: Exchange output constants (default 3: mu, kappa, G_NN).
        nconstants_c: Correlation output constants (default 2: beta, gamma).
        dropout: Dropout probability in ResBlocks.
        num_symm_blocks: ResBlocks dedicated to spin symmetrization in the
            correlation branch. Must satisfy num_symm_blocks < num_layers // 2 - 1.
        DFT: Reserved for future use.
    """

    def __init__(
        self,
        num_layers: int,
        h_dim: int,
        nconstants_x: int = 3,  # (mu, kappa, G_NN) for exchange
        nconstants_c: int = 2,
        dropout: float = 0.2,
        num_symm_blocks: int = 1,
        DFT: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Spin scaling buffer — follows model.to(device) automatically
        self.register_buffer("scaling_array", LLMGGA_SPIN_SCALING_MULTIPLIER.float())

        # Learnable descriptor normalization scales (log-space for positivity)
        self.log_scale_rho   = nn.Parameter(torch.full((1,), _LOG_SCALE_RHO_INIT))
        self.log_scale_sigma = nn.Parameter(torch.full((1,), _LOG_SCALE_SIGMA_INIT))
        self.log_scale_tau   = nn.Parameter(torch.full((1,), _LOG_SCALE_TAU_INIT))
        self.log_scale_lapl  = nn.Parameter(torch.full((1,), _LOG_SCALE_LAPL_INIT))

        # --- Correlation branch ---
        self.c_input_layers = nn.Sequential(
            nn.Linear(LLMGGA_ZETA_DESCRIPTOR_DIMENSIONALITY, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )
        self.c_symmetrization_blocks = nn.Sequential(
            *[ResBlock(h_dim, dropout) for _ in range(num_symm_blocks)]
        )
        num_post_symm_blocks = (num_layers // 2 - 1) - num_symm_blocks
        self.c_post_symm_blocks = nn.Sequential(
            *[ResBlock(h_dim, dropout) for _ in range(num_post_symm_blocks)]
        )
        # Output fuses correlation hidden state (h_dim) with exchange hidden state (h_dim)
        self.c_output_layer = nn.Linear(2 * h_dim, nconstants_c, bias=True)

        # --- Exchange branch (weight-tied across spin channels) ---
        x_feature_modules = [
            nn.Linear(LLMGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers // 2 - 1):
            x_feature_modules.append(ResBlock(h_dim, dropout))
        self.x_feature_extractor = nn.Sequential(*x_feature_modules)
        self.x_output_layer = nn.Linear(h_dim, nconstants_x, bias=True)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    @staticmethod
    def kappa_activation(x: torch.Tensor) -> torch.Tensor:
        """Maps ℝ → (0, 1). Bounds the exchange enhancement factor kappa."""
        return torch.sigmoid(_KAPPA_SCALE * (x + 0.5))

    @staticmethod
    def beta_activation(x: torch.Tensor) -> torch.Tensor:
        """Maps ℝ → (0.75, 1.25). Reflects the weak density-dependence of beta."""
        return (torch.sigmoid(_BETA_SCALE * x) + _BETA_SHIFT) / 2.0

    @staticmethod
    def shifted_elu(x: torch.Tensor) -> torch.Tensor:
        """ELU shifted so output ≥ 0. Used for mu and gamma constraint construction."""
        return nn.functional.elu(x) + 1.0

    @staticmethod
    def g_nn_activation(x: torch.Tensor) -> torch.Tensor:
        """Maps ℝ → (-1, 1). G_NN correction term for exchange enhancement."""
        return torch.tanh(x)

    # ------------------------------------------------------------------
    # Constraint-point constructors
    # ------------------------------------------------------------------

    @staticmethod
    def all_sigma_zero(x: torch.Tensor) -> torch.Tensor:
        """
        Returns descriptor tensor at the UEG limit (s=0, α_iso=1, q=0) while
        preserving density normalization and spin polarization (zeta).

        The UEG limit is valid for any spin polarization, so zeta (last column)
        is preserved from the input. This is the constraint point for beta.

        Descriptor indices zeroed: 2,3,4 (s), 5,6 (α), 7,8 (q).
        Index 9 (zeta) is preserved.

        Args:
            x: Normalized descriptor tensor of shape (N, 10).

        Returns:
            Tensor of shape (N, 10) with gradient/KE/Laplacian descriptors zeroed.
        """
        densities_norm = x[:, :2]
        zeros_s     = torch.zeros(x.shape[0], 3, device=x.device)
        zeros_alpha = torch.zeros(x.shape[0], 2, device=x.device)
        zeros_q     = torch.zeros(x.shape[0], 2, device=x.device)
        zeta        = x[:, -1].unsqueeze(1)
        return torch.cat([densities_norm, zeros_s, zeros_alpha, zeros_q, zeta], dim=1)

    @staticmethod
    def all_rho_inf(x: torch.Tensor) -> torch.Tensor:
        """
        Returns descriptor tensor at the high-density limit (ρ_α = ρ_β → ∞).

        Used to enforce the gamma constraint: γ → 1 as ρ → ∞. The normalized
        density descriptor approaches 1 as ρ^(1/3)/scale_rho → ∞ via tanh → 1.

        Args:
            x: Normalized descriptor tensor of shape (N, 10).

        Returns:
            Tensor of shape (N, 10) with density descriptors (indices 0,1) set to 1.
        """
        ones = torch.ones(x.shape[0], 2, device=x.device)
        return torch.cat([ones, x[:, S_ALPHA_INDEX:]], dim=1)

    # ------------------------------------------------------------------
    # Descriptor computation
    # ------------------------------------------------------------------

    def get_density_descriptors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes 10 normalized density descriptors from raw DFT grid data.

        Descriptors 0–8 are passed through tanh for bounded output in (-1, 1).
        Descriptor 9 (zeta) is already in [-1, 1] by definition.

        Descriptor layout:
            0: n_α  = tanh(ρ_α^(1/3) / scale_rho)
            1: n_β  = tanh(ρ_β^(1/3) / scale_rho)
            2: s_α  = tanh(|∇ρ_α| / (2·kF·ρ_α) / scale_sigma)
            3: s_tot = tanh(|∇ρ| / (2·kF·ρ) / scale_sigma)
            4: s_β  = tanh(|∇ρ_β| / (2·kF·ρ_β) / scale_sigma)
            5: α_α  = tanh(clamp((τ_α - τ_W_α) / τ_TF_α - 1, ≥-1) / scale_tau)
            6: α_β  = tanh(clamp((τ_β - τ_W_β) / τ_TF_β - 1, ≥-1) / scale_tau)
            7: q_α  = tanh(∇²ρ_α / (_C_LAPL · ρ_α^(5/3)) / scale_lapl)
            8: q_β  = tanh(∇²ρ_β / (_C_LAPL · ρ_β^(5/3)) / scale_lapl)
            9: ζ   = (ρ_α - ρ_β) / (ρ_α + ρ_β)

        Args:
            x: Raw DFT grid tensor of shape (N, 9) with columns:
               [ρ_α, ρ_β, σ_αα, σ_tot, σ_ββ, τ_α, τ_β, ∇²ρ_α, ∇²ρ_β]

        Returns:
            Descriptor tensor of shape (N, 10).
        """
        scale_rho   = torch.exp(self.log_scale_rho)
        scale_sigma = torch.exp(self.log_scale_sigma)
        scale_alpha = torch.exp(self.log_scale_tau)
        scale_q     = torch.exp(self.log_scale_lapl)

        rho_a     = x[:, RHO_ALPHA_INDEX]
        rho_b     = x[:, RHO_BETA_INDEX]
        sigma_a   = x[:, S_ALPHA_INDEX]
        sigma_tot = x[:, S_TOTAL_INDEX]
        sigma_b   = x[:, S_BETA_INDEX]
        tau_a     = x[:, TAU_ALPHA_INDEX]
        tau_b     = x[:, TAU_BETA_INDEX]
        lapl_a    = x[:, LAPL_ALPHA_INDEX]
        lapl_b    = x[:, LAPL_BETA_INDEX]

        # Density normalization: ρ^(1/3)
        n_alpha = (rho_a + EPS_RHO) ** (1.0 / 3.0) / scale_rho
        n_beta  = (rho_b + EPS_RHO) ** (1.0 / 3.0) / scale_rho

        # Reduced gradient: s = |∇ρ| / (2·kF·ρ), kF = _C_FERMI·ρ^(1/3)
        s_alpha = torch.sqrt(sigma_a   + EPS_SIGMA) / (rho_a          + EPS_RHO) ** (4.0 / 3.0) / _C_FERMI / 2.0 / scale_sigma
        s_norm  = torch.sqrt(sigma_tot + EPS_SIGMA) / (rho_a + rho_b  + EPS_RHO) ** (4.0 / 3.0) / _C_FERMI / 2.0 / scale_sigma
        s_beta  = torch.sqrt(sigma_b   + EPS_SIGMA) / (rho_b          + EPS_RHO) ** (4.0 / 3.0) / _C_FERMI / 2.0 / scale_sigma

        # Iso-orbital indicator: α = (τ - τ_W) / τ_TF, centered at UEG (α_UEG=1 → α-1=0)
        tau_tf_alpha = _C_TF * (rho_a + EPS_RHO) ** (5.0 / 3.0)
        tau_tf_beta  = _C_TF * (rho_b + EPS_RHO) ** (5.0 / 3.0)
        tau_w_alpha  = sigma_a / (8.0 * (rho_a + EPS_RHO))
        tau_w_beta   = sigma_b / (8.0 * (rho_b + EPS_RHO))
        alpha_alpha = torch.clamp((tau_a - tau_w_alpha) / (tau_tf_alpha + EPS_RHO) - 1.0, min=-1.0) / scale_alpha
        alpha_beta  = torch.clamp((tau_b - tau_w_beta)  / (tau_tf_beta  + EPS_RHO) - 1.0, min=-1.0) / scale_alpha

        # Reduced Laplacian: q = ∇²ρ / (_C_LAPL · ρ^(5/3))
        q_alpha = lapl_a / (_C_LAPL * (rho_a + EPS_RHO) ** (5.0 / 3.0)) / scale_q
        q_beta  = lapl_b / (_C_LAPL * (rho_b + EPS_RHO) ** (5.0 / 3.0)) / scale_q

        # Spin polarization (already in [-1, 1], not passed through tanh)
        zeta = (rho_a - rho_b) / (rho_a + rho_b + EPS_RHO)

        desc = torch.stack(
            [n_alpha, n_beta, s_alpha, s_norm, s_beta, alpha_alpha, alpha_beta, q_alpha, q_beta],
            dim=1,
        )
        return torch.cat([torch.tanh(desc), zeta.unsqueeze(1)], dim=1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the 26-element modified PBE constant tensor for a batch of
        grid points.

        The forward pass evaluates the network at three descriptor points per batch:
          1. Real descriptors       → mu, beta, gamma, kappa values
          2. UEG descriptors (s→0)  → constraint anchors for mu and beta
          3. High-density limit     → constraint anchor for gamma

        Each constrained parameter is computed as activation(real - anchor),
        satisfying the physical constraint independent of network weights.

        Args:
            x: Raw DFT grid tensor of shape (N, 9).

        Returns:
            Tensor of shape (N, 26): modified_factors * true_constants_PBE.
            Indices 0,1 are beta, gamma; indices 2–21 pass through unchanged;
            indices 22–25 are kappa_up, mu_up, kappa_down, mu_down.
        """
        # ---- Descriptors at real density ----
        x_correlation_desc     = self.get_density_descriptors(x)
        x_exchange_desc_scaled = self.get_density_descriptors(self.scaling_array * x)

        # Spin-swapped correlation descriptors (zeta negated to preserve antisymmetry)
        swapped_indices = LLMGGA_SPIN_INVERTED_SLICE + [-1]
        x_corr_desc_swapped = x_correlation_desc[:, swapped_indices].clone()
        x_corr_desc_swapped = torch.cat(
            [x_corr_desc_swapped[:, :-1], -x_corr_desc_swapped[:, -1:]], dim=1
        )

        # ---- Exchange: real values ----
        hidden_x_up_scaled   = self.x_feature_extractor(
            x_exchange_desc_scaled[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX]]
        )
        hidden_x_down_scaled = self.x_feature_extractor(
            x_exchange_desc_scaled[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX]]
        )
        hidden_x_symm = (hidden_x_up_scaled + hidden_x_down_scaled) / 2.0

        params_x_up_real   = self.x_output_layer(hidden_x_up_scaled)
        params_x_down_real = self.x_output_layer(hidden_x_down_scaled)
        mu_up_real,   kappa_up_real   = params_x_up_real[:,   MU_EX_INDEX].view(-1, 1), params_x_up_real[:,   KAPPA_EX_INDEX].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, MU_EX_INDEX].view(-1, 1), params_x_down_real[:, KAPPA_EX_INDEX].view(-1, 1)
        g_nn_up_real   = params_x_up_real[:, 2].view(-1, 1)      # NEW: G_NN for spin-up
        g_nn_down_real = params_x_down_real[:, 2].view(-1, 1)    # NEW: G_NN for spin-down

        # ---- Correlation: real values ----
        h_pre_symm         = self.c_symmetrization_blocks(self.c_input_layers(x_correlation_desc))
        h_pre_symm_swapped = self.c_symmetrization_blocks(self.c_input_layers(x_corr_desc_swapped))
        h_post_symm = (h_pre_symm + h_pre_symm_swapped) / 2.0
        hidden_c    = self.c_post_symm_blocks(h_post_symm)

        params_c_real = self.c_output_layer(torch.cat([hidden_c, hidden_x_symm], dim=1))
        beta_real  = params_c_real[:, BETA_CORR_INDEX].view(-1, 1)
        gamma_real = params_c_real[:, GAMMA_CORR_INDEX].view(-1, 1)

        # ---- Exchange UEG constraint (s→0, τ→τ_TF) ----
        rho_a = x[:, RHO_ALPHA_INDEX]
        rho_b = x[:, RHO_BETA_INDEX]
        zeros = torch.zeros_like(rho_a)

        tau_tf_2rho_a = _C_TF * (2.0 * rho_a + EPS_RHO) ** (5.0 / 3.0)
        tau_tf_2rho_b = _C_TF * (2.0 * rho_b + EPS_RHO) ** (5.0 / 3.0)
        raw_ueg_exch_input = torch.stack(
            [rho_a, rho_b, zeros, zeros, zeros, tau_tf_2rho_a / 2.0, tau_tf_2rho_b / 2.0, zeros, zeros],
            dim=1,
        )
        x_exch_ueg_desc   = self.get_density_descriptors(self.scaling_array * raw_ueg_exch_input)
        hidden_x_up_ueg   = self.x_feature_extractor(x_exch_ueg_desc[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX]])
        hidden_x_down_ueg = self.x_feature_extractor(x_exch_ueg_desc[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX]])
        mu_up_at_constraint   = self.x_output_layer(hidden_x_up_ueg)[:,   MU_EX_INDEX].view(-1, 1)
        mu_down_at_constraint = self.x_output_layer(hidden_x_down_ueg)[:, MU_EX_INDEX].view(-1, 1)
        g_nn_up_at_constraint   = self.x_output_layer(hidden_x_up_ueg)[:, 2].view(-1, 1)     # NEW: G_NN UEG constraint (spin-up)
        g_nn_down_at_constraint = self.x_output_layer(hidden_x_down_ueg)[:, 2].view(-1, 1)   # NEW: G_NN UEG constraint (spin-down)

        # ---- Correlation UEG constraint (s→0) for beta ----
        tau_tf_rho_a = _C_TF * (rho_a + EPS_RHO) ** (5.0 / 3.0)
        tau_tf_rho_b = _C_TF * (rho_b + EPS_RHO) ** (5.0 / 3.0)
        raw_ueg_corr_input = torch.stack(
            [rho_a, rho_b, zeros, zeros, zeros, tau_tf_rho_a, tau_tf_rho_b, zeros, zeros],
            dim=1,
        )
        x_corr_ueg_desc = self.get_density_descriptors(raw_ueg_corr_input)
        x_corr_ueg_desc_swapped = x_corr_ueg_desc[:, swapped_indices].clone()
        x_corr_ueg_desc_swapped = torch.cat(
            [x_corr_ueg_desc_swapped[:, :-1], -x_corr_ueg_desc_swapped[:, -1:]], dim=1
        )

        x_exch_ueg_for_corr  = self.get_density_descriptors(self.scaling_array * raw_ueg_corr_input)
        hidden_x_up_for_beta   = self.x_feature_extractor(x_exch_ueg_for_corr[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX]])
        hidden_x_down_for_beta = self.x_feature_extractor(x_exch_ueg_for_corr[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX]])
        hidden_x_symm_for_beta = (hidden_x_up_for_beta + hidden_x_down_for_beta) / 2.0

        h_pre_symm_beta         = self.c_symmetrization_blocks(self.c_input_layers(x_corr_ueg_desc))
        h_pre_symm_swapped_beta = self.c_symmetrization_blocks(self.c_input_layers(x_corr_ueg_desc_swapped))
        h_c_ueg = self.c_post_symm_blocks((h_pre_symm_beta + h_pre_symm_swapped_beta) / 2.0)
        beta_at_constraint = self.c_output_layer(
            torch.cat([h_c_ueg, hidden_x_symm_for_beta], dim=1)
        )[:, BETA_CORR_INDEX].view(-1, 1)

        # ---- High-density constraint (ρ→∞) for gamma ----
        x_corr_rho_inf         = self.all_rho_inf(x_correlation_desc)
        x_corr_rho_inf_swapped = self.all_rho_inf(x_corr_desc_swapped)
        h_pre_symm_rho_inf         = self.c_symmetrization_blocks(self.c_input_layers(x_corr_rho_inf))
        h_pre_symm_swapped_rho_inf = self.c_symmetrization_blocks(self.c_input_layers(x_corr_rho_inf_swapped))
        h_c_constr_gamma = self.c_post_symm_blocks(
            (h_pre_symm_rho_inf + h_pre_symm_swapped_rho_inf) / 2.0
        )
        gamma_at_constraint = self.c_output_layer(
            torch.cat([h_c_constr_gamma, hidden_x_symm], dim=1)
        )[:, GAMMA_CORR_INDEX].view(-1, 1)

        # ---- Apply constraint activations ----
        beta     = self.beta_activation(beta_real  - beta_at_constraint)
        gamma    = self.shifted_elu(gamma_real     - gamma_at_constraint)
        gamma = torch.clamp(gamma, min=1.0e-2)
        mu_up    = self.shifted_elu(mu_up_real     - mu_up_at_constraint)
        mu_down  = self.shifted_elu(mu_down_real   - mu_down_at_constraint)
        kappa_up   = self.kappa_activation(kappa_up_real)
        kappa_down = self.kappa_activation(kappa_down_real)
        g_nn_up   = self.g_nn_activation(g_nn_up_real - g_nn_up_at_constraint)      # NEW: G_NN with UEG constraint
        g_nn_down = self.g_nn_activation(g_nn_down_real - g_nn_down_at_constraint)  # NEW: G_NN with UEG constraint

        # ---- Assemble 28-element output tensor ----
        # Layout: [beta, gamma, <20 pass-through>, kappa_up, mu_up, kappa_down, mu_down, g_nn_up, g_nn_down]
        constants_batch = true_constants_PBE.repeat(x.shape[0], 1).to(x.device)
        fill_tensor = torch.ones([x.shape[0], _N_FILL_CONSTANTS], device=x.device)
        final_tensor = torch.hstack(
            [beta, gamma, fill_tensor, kappa_up, mu_up, kappa_down, mu_down, g_nn_up, g_nn_down]
        )

        final_constants = final_tensor * constants_batch

        gamma_phys = torch.clamp(final_constants[:, 1:2], min=1.0e-2)
        final_constants = torch.hstack(
            [final_constants[:, :1], gamma_phys, final_constants[:, 2:]]
        )

        return final_constants
