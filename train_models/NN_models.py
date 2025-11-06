import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals import true_constants_PBE
from dft_functionals.constants import (
    BETA_CORR_INDEX, EPS_LOG, EPS_RHO, EPS_SIGMA, GAMMA_CORR_INDEX,
    KAPPA_EX_INDEX, LAPL_ALPHA_INDEX, LAPL_BETA_INDEX,
    LLMGGA_DESCRIPTOR_DIMENSIONALITY,
    LLMGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY, LLMGGA_SPIN_INVERTED_SLICE,
    LLMGGA_SPIN_SCALING_MULTIPLIER, MGGA_DESCRIPTOR_DIMENSIONALITY,
    MGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY, MGGA_SPIN_INVERTED_SLICE,
    MGGA_SPIN_SCALING_MULTIPLIER, MU_EX_INDEX, RHO_ALPHA_INDEX, RHO_BETA_INDEX,
    S_ALPHA_INDEX, S_BETA_INDEX, S_TOTAL_INDEX, TAU_ALPHA_INDEX,
    TAU_BETA_INDEX)

random.seed(42)

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

sigmoid = torch.nn.Sigmoid()
elu = torch.nn.ELU()

"""
Define an nn.Module class for a simple residual block with equal dimensions
"""


class ResBlock(nn.Module):
    """
    Iniialize a residual block with two FC followed by (LayerNorm + GELU + dropout) layers
    """

    def __init__(self, h_dim, dropout):
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

    def forward(self, x):
        residue = x
        out = self.fc(x)
        out = self.dropout(out + residue)
        return self.activation(out)


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, dropout, DFT=None, constants=[]):
        super().__init__()

        self.DFT = DFT
        self.constants = constants
        if self.constants:
            nconstants = len(constants)

        modules = []
        modules.extend(
            [
                nn.Linear(MGGA_DESCRIPTOR_DIMENSIONALITY, h_dim, bias=False),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ]
        )

        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim, dropout))

        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)

    def dm21_like_sigmoid(self, x):
        """
        Custom sigmoid translates from [-inf, +inf] to [0, 2]
        """

        exp = torch.exp(-0.5 * x)
        return 2 / (1 + exp)

    def unsymm_forward(self, x):

        x = self.hidden_layers(x)
        x = 1.05 * self.dm21_like_sigmoid(x)

        return x

    @staticmethod
    def get_density_descriptors(x):
        """
        0 - alpha density
        1 - beta density
        2 - alpha gradient
        3 - total gradiet
        4 - beta gradient
        5 - tau alpha
        6 - tau beta
        """

        n_alpha = x[:, RHO_ALPHA_INDEX] ** (1 / 3)
        n_beta = x[:, RHO_BETA_INDEX] ** (1 / 3)

        s_alpha = (
            torch.sqrt(x[:, S_ALPHA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_norm = (
            torch.sqrt(x[:, S_TOTAL_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_beta = (
            torch.sqrt(x[:, S_BETA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )

        tau_tf_alpha = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_tf_beta = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_w_alpha = x[:, S_ALPHA_INDEX] / (8 * (x[:, RHO_ALPHA_INDEX] + EPS_RHO))
        tau_w_beta = x[:, S_BETA_INDEX] / (8 * (x[:, RHO_BETA_INDEX] + EPS_RHO))

        tau_alpha = (x[:, TAU_ALPHA_INDEX] - tau_w_alpha) / tau_tf_alpha
        tau_beta = (x[:, TAU_BETA_INDEX] - tau_w_beta) / tau_tf_beta

        X = torch.stack(
            [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha, tau_beta], dim=1
        )

        X[:, TAU_ALPHA_INDEX:] = X[:, TAU_ALPHA_INDEX:] - 1
        X = torch.tanh(X)

        return X

    def forward(self, x):
        """
        Returns:
            spin-symmetrized enhancement factor for LDA exhange energy
        """

        x = self.get_density_descriptors(x)

        result = (
            self.unsymm_forward(x) + self.unsymm_forward(x[:, MGGA_SPIN_INVERTED_SLICE])
        ) / 2

        return result


class pcPBEMLOptimizer(nn.Module):
    def __init__(
        self, num_layers, h_dim, nconstants_x=2, nconstants_c=2, dropout=0.2, DFT=None
    ):
        super().__init__()

        self.DFT = DFT

        modules_x = []  # NN part for exchange
        modules_c = []  # NN part for correlation

        input_layer_c = [
            nn.Linear(MGGA_DESCRIPTOR_DIMENSIONALITY, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        input_layer_x = [
            nn.Linear(MGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        modules_x.extend(input_layer_x)
        modules_c.extend(input_layer_c)

        for _ in range(num_layers // 2 - 1):
            modules_x.append(ResBlock(h_dim, dropout))
            modules_c.append(ResBlock(h_dim, dropout))

        modules_x.append(nn.Linear(h_dim, nconstants_x, bias=True))
        modules_c.append(nn.Linear(h_dim, nconstants_c, bias=True))

        self.hidden_layers_x = nn.Sequential(*modules_x)
        self.hidden_layers_c = nn.Sequential(*modules_c)

    def kappa_activation(self, x):
        """
        Translates values from [-inf, +inf] to [0, 1]
        """
        return sigmoid(4 * (x + 0.5))

    def beta_activation(self, x):
        """
        Translates values from [-inf, +inf] to [0.75, 1.25] as beta is weakly dependent on density
        """
        return (sigmoid(8 * x) + 1.5) / 2

    def get_exchange_constants(self, x):

        x_x_up = self.hidden_layers_x(x[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX]])  # Slice out density descriptors
        x_x_down = self.hidden_layers_x(x[:, [S_BETA_INDEX, TAU_BETA_INDEX]])

        return (
            x_x_up[:, 1].view(-1, 1),
            x_x_up[:, 0].view(-1, 1),
            x_x_down[:, 1].view(-1, 1),
            x_x_down[:, 0].view(-1, 1),
        )

    def get_correlation_constants(self, x):

        x_c = self.hidden_layers_c(x)

        return x_c[:, 0].view(-1, 1), x_c[:, 1].view(-1, 1)

    @staticmethod
    def all_sigma_zero(x):
        """
        Function for parameter beta constraint
        """
        return torch.hstack([x[:, :S_ALPHA_INDEX], torch.zeros([x.shape[0], x.shape[1]-2]).to(x.device)])

    @staticmethod
    def all_sigma_inf(x):
        """
        Function for PW91 correlation parameters constraint
        """
        return torch.hstack(
            [x[:, :S_ALPHA_INDEX], torch.ones([x.shape[0], 3]).to(x.device), x[:, TAU_ALPHA_INDEX:]]
        )

    @staticmethod
    def all_rho_inf(x):
        """
        Function for parameter gamma constraint
        """
        return torch.hstack([torch.ones([x.shape[0], 2]).to(x.device), x[:, S_ALPHA_INDEX:]])

    @staticmethod
    def shifted_elu(x):
        return elu(x) + 1

    @staticmethod
    def get_density_descriptors(x):
        """
        0 - alpha density
        1 - beta density
        2 - alpha gradient
        3 - total gradiet
        4 - beta gradient
        5 - tau alpha
        6 - tau beta
        """

        n_alpha = x[:, RHO_ALPHA_INDEX] ** (1 / 3)
        n_beta = x[:, RHO_BETA_INDEX] ** (1 / 3)

        s_alpha = (
            torch.sqrt(x[:, S_ALPHA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_norm = (
            torch.sqrt(x[:, S_TOTAL_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_beta = (
            torch.sqrt(x[:, S_BETA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )

        tau_tf_alpha = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_tf_beta = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_w_alpha = x[:, S_ALPHA_INDEX] / (8 * (x[:, RHO_ALPHA_INDEX] + EPS_RHO))
        tau_w_beta = x[:, S_BETA_INDEX] / (8 * (x[:, RHO_BETA_INDEX] + EPS_RHO))

        tau_alpha = (x[:, TAU_ALPHA_INDEX] - tau_w_alpha) / tau_tf_alpha
        tau_beta = (x[:, TAU_BETA_INDEX] - tau_w_beta) / tau_tf_beta

        X = torch.stack(
            [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha, tau_beta], dim=1
        )

        X[:, TAU_ALPHA_INDEX:] = X[:, TAU_ALPHA_INDEX:] - 1
        X = torch.tanh(X)

        return X

    def get_exchange_descriptors(self, x):
        scaling_array = MGGA_SPIN_SCALING_MULTIPLIER.to(x.device)

        return self.get_density_descriptors(scaling_array * x)

    def forward(self, x):

        x_exchange_desc = self.get_exchange_descriptors(x)
        x_correlation_desc = self.get_density_descriptors(x)

        x_correlation_desc_swapped = x_correlation_desc[:, MGGA_SPIN_INVERTED_SLICE]
        params_c_real = (
            self.hidden_layers_c(x_correlation_desc)
            + self.hidden_layers_c(x_correlation_desc_swapped)
        ) / 2
        beta_real, gamma_real = params_c_real[:, BETA_CORR_INDEX].view(-1, 1), params_c_real[
            :, GAMMA_CORR_INDEX
        ].view(-1, 1)

        x_corr_sigma_zero = self.all_sigma_zero(x_correlation_desc)
        x_corr_sigma_zero_swapped = self.all_sigma_zero(x_correlation_desc_swapped)
        params_c_sigma_zero = (
            self.hidden_layers_c(x_corr_sigma_zero)
            + self.hidden_layers_c(x_corr_sigma_zero_swapped)
        ) / 2
        beta_at_constraint = params_c_sigma_zero[:, BETA_CORR_INDEX].view(-1, 1)

        x_corr_rho_inf = self.all_rho_inf(x_correlation_desc)
        x_corr_rho_inf_swapped = self.all_rho_inf(x_correlation_desc_swapped)
        params_c_rho_inf = (
            self.hidden_layers_c(x_corr_rho_inf)
            + self.hidden_layers_c(x_corr_rho_inf_swapped)
        ) / 2
        gamma_at_constraint = params_c_rho_inf[:, GAMMA_CORR_INDEX].view(-1, 1)

        params_x_up_real = self.hidden_layers_x(x_exchange_desc[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX]])
        params_x_down_real = self.hidden_layers_x(x_exchange_desc[:, [S_BETA_INDEX, TAU_BETA_INDEX]])
        mu_up_real, kappa_up_real = params_x_up_real[:, MU_EX_INDEX].view(
            -1, 1
        ), params_x_up_real[:, KAPPA_EX_INDEX].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, MU_EX_INDEX].view(
            -1, 1
        ), params_x_down_real[:, KAPPA_EX_INDEX].view(-1, 1)

        x_exch_s_zero = self.all_sigma_zero(x_exchange_desc)
        params_x_s_zero_up = self.hidden_layers_x(x_exch_s_zero[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX]])
        params_x_s_zero_down = self.hidden_layers_x(x_exch_s_zero[:, [S_BETA_INDEX, TAU_BETA_INDEX]])
        mu_up_at_constraint = params_x_s_zero_up[:, MU_EX_INDEX].view(-1, 1)
        mu_down_at_constraint = params_x_s_zero_down[:, MU_EX_INDEX].view(-1, 1)

        beta = self.beta_activation(beta_real - beta_at_constraint)
        gamma = self.shifted_elu(gamma_real - gamma_at_constraint)
        mu_up = self.shifted_elu(mu_up_real - mu_up_at_constraint)
        mu_down = self.shifted_elu(mu_down_real - mu_down_at_constraint)
        kappa_up = self.kappa_activation(kappa_up_real)
        kappa_down = self.kappa_activation(kappa_down_real)

        constants_batch = true_constants_PBE.repeat(x_exchange_desc.shape[0], 1).to(
            x_exchange_desc.device
        )
        fill_tensor = torch.ones(
            [x_exchange_desc.shape[0], 20], device=x_exchange_desc.device
        )
        final_tensor = torch.hstack(
            [
                beta,
                gamma,
                fill_tensor,
                kappa_up,
                mu_up,
                kappa_down,
                mu_down,
            ]
        )
        return final_tensor * constants_batch
    

class pcPBELMLOptimizer(pcPBEMLOptimizer):

    def __init__(
        self, num_layers, h_dim, nconstants_x=2, nconstants_c=2, dropout=0.2, DFT=None
    ):
        super().__init__(num_layers, h_dim, nconstants_x, nconstants_c, dropout, DFT)

        self.DFT = DFT

        modules_x = []  # NN part for exchange
        modules_c = []  # NN part for correlation

        input_layer_c = [
            nn.Linear(LLMGGA_DESCRIPTOR_DIMENSIONALITY*2, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        input_layer_x = [
            nn.Linear(LLMGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY*2, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        modules_x.extend(input_layer_x)
        modules_c.extend(input_layer_c)

        for _ in range(num_layers // 2 - 1):
            modules_x.append(ResBlock(h_dim, dropout))
            modules_c.append(ResBlock(h_dim, dropout))

        modules_x.append(nn.Linear(h_dim, nconstants_x, bias=True))
        modules_c.append(nn.Linear(h_dim, nconstants_c, bias=True))

        self.hidden_layers_x = nn.Sequential(*modules_x)
        self.hidden_layers_c = nn.Sequential(*modules_c)
        
        
    @staticmethod
    def all_sigma_zero(x):
        """
        Function for parameter beta constraint
        """
        return torch.hstack([x[:, :S_ALPHA_INDEX], torch.zeros([x.shape[0], 7]).to(x.device), x[:, 9:]])

    @staticmethod
    def all_sigma_inf(x):
        """
        Function for PW91 correlation parameters constraint
        """
        return torch.hstack(
            [x[:, :S_ALPHA_INDEX], torch.ones([x.shape[0], 3]).to(x.device), x[:, TAU_ALPHA_INDEX:]]
        )

    @staticmethod
    def all_rho_inf(x):
        """
        Function for parameter gamma constraint
        """
        return torch.hstack([torch.ones([x.shape[0], 2]).to(x.device), x[:, S_ALPHA_INDEX:]])
        
    
    def get_exchange_constants(self, x):

        x_x_up = self.hidden_layers_x(x[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX, S_ALPHA_INDEX+9, TAU_ALPHA_INDEX+9, LAPL_ALPHA_INDEX+9]])
        x_x_down = self.hidden_layers_x(x[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX, S_BETA_INDEX+9, TAU_BETA_INDEX+9, LAPL_BETA_INDEX+9]])

        return (
            x_x_up[:, 1].view(-1, 1),
            x_x_up[:, 0].view(-1, 1),
            x_x_down[:, 1].view(-1, 1),
            x_x_down[:, 0].view(-1, 1),
        )
        
    def get_exchange_descriptors(self, x):
        scaling_array = LLMGGA_SPIN_SCALING_MULTIPLIER.to(x.device)

        return self.get_density_descriptors(scaling_array * x)
    
    
    @staticmethod
    def get_density_descriptors(x):
        """
        0 - alpha density
        1 - beta density
        2 - alpha gradient
        3 - total gradiet
        4 - beta gradient
        5 - tau alpha
        6 - tau beta
        7 - lapl alpha
        8 - lapl beta
        """

        n_alpha = x[:, RHO_ALPHA_INDEX] ** (1 / 3)
        n_beta = x[:, RHO_BETA_INDEX] ** (1 / 3)

        s_alpha = (
            torch.sqrt(x[:, S_ALPHA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_norm = (
            torch.sqrt(x[:, S_TOTAL_INDEX] + EPS_SIGMA)
            / (x[:, RHO_ALPHA_INDEX] + x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_beta = (
            torch.sqrt(x[:, S_BETA_INDEX] + EPS_SIGMA)
            / (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )

        tau_tf_alpha = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_tf_beta = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (5 / 3)
        )
        tau_w_alpha = x[:, S_ALPHA_INDEX] / (8 * (x[:, RHO_ALPHA_INDEX] + EPS_RHO))
        tau_w_beta = x[:, S_BETA_INDEX] / (8 * (x[:, RHO_BETA_INDEX] + EPS_RHO))

        tau_alpha = (x[:, TAU_ALPHA_INDEX] - tau_w_alpha) / tau_tf_alpha - 1
        tau_beta = (x[:, TAU_BETA_INDEX] - tau_w_beta) / tau_tf_beta - 1

        q_alpha = x[:, LAPL_ALPHA_INDEX] / (4 * (3 * torch.pi**2) ** (2 / 3) * (x[:, RHO_ALPHA_INDEX] + EPS_RHO) ** (5/3))
        q_beta = x[:, LAPL_BETA_INDEX] / (4 * (3 * torch.pi**2) ** (2 / 3) * (x[:, RHO_BETA_INDEX] + EPS_RHO) ** (5/3))

        X = torch.hstack(
            [
                torch.tanh(
                    torch.stack(
                        [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha, tau_beta, q_alpha, q_beta], dim=1
                    )
                ),
                torch.log(
                    EPS_LOG + torch.stack(
                        [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha+1, tau_beta+1, abs(q_alpha), abs(q_beta)], dim=1
                    )
                )
            ]
        )

        return X
    
    def forward(self, x):

        x_exchange_desc = self.get_exchange_descriptors(x)
        x_correlation_desc = self.get_density_descriptors(x)

        log_indices = [idx + 9 for idx in LLMGGA_SPIN_INVERTED_SLICE]
        x_correlation_desc_swapped = x_correlation_desc[:, LLMGGA_SPIN_INVERTED_SLICE + log_indices]
        params_c_real = (
            self.hidden_layers_c(x_correlation_desc)
            + self.hidden_layers_c(x_correlation_desc_swapped)
        ) / 2
        beta_real, gamma_real = params_c_real[:, BETA_CORR_INDEX].view(-1, 1), params_c_real[
            :, GAMMA_CORR_INDEX
        ].view(-1, 1)

        x_corr_sigma_zero = self.all_sigma_zero(x_correlation_desc)
        x_corr_sigma_zero_swapped = self.all_sigma_zero(x_correlation_desc_swapped)
        params_c_sigma_zero = (
            self.hidden_layers_c(x_corr_sigma_zero)
            + self.hidden_layers_c(x_corr_sigma_zero_swapped)
        ) / 2
        beta_at_constraint = params_c_sigma_zero[:, BETA_CORR_INDEX].view(-1, 1)

        x_corr_rho_inf = self.all_rho_inf(x_correlation_desc)
        x_corr_rho_inf_swapped = self.all_rho_inf(x_correlation_desc_swapped)
        params_c_rho_inf = (
            self.hidden_layers_c(x_corr_rho_inf)
            + self.hidden_layers_c(x_corr_rho_inf_swapped)
        ) / 2
        gamma_at_constraint = params_c_rho_inf[:, GAMMA_CORR_INDEX].view(-1, 1)

        params_x_up_real = self.hidden_layers_x(x_exchange_desc[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX, S_ALPHA_INDEX+9, TAU_ALPHA_INDEX+9, LAPL_ALPHA_INDEX+9]])
        params_x_down_real = self.hidden_layers_x(x_exchange_desc[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX, S_BETA_INDEX+9, TAU_BETA_INDEX+9, LAPL_BETA_INDEX+9]])
        mu_up_real, kappa_up_real = params_x_up_real[:, MU_EX_INDEX].view(
            -1, 1
        ), params_x_up_real[:, KAPPA_EX_INDEX].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, MU_EX_INDEX].view(
            -1, 1
        ), params_x_down_real[:, KAPPA_EX_INDEX].view(-1, 1)

        x_exch_s_zero = self.all_sigma_zero(x_exchange_desc)
        params_x_s_zero_up = self.hidden_layers_x(x_exch_s_zero[:, [S_ALPHA_INDEX, TAU_ALPHA_INDEX, LAPL_ALPHA_INDEX, S_ALPHA_INDEX+9, TAU_ALPHA_INDEX+9, LAPL_ALPHA_INDEX+9]])
        params_x_s_zero_down = self.hidden_layers_x(x_exch_s_zero[:, [S_BETA_INDEX, TAU_BETA_INDEX, LAPL_BETA_INDEX, S_BETA_INDEX+9, TAU_BETA_INDEX+9, LAPL_BETA_INDEX+9]])
        mu_up_at_constraint = params_x_s_zero_up[:, MU_EX_INDEX].view(-1, 1)
        mu_down_at_constraint = params_x_s_zero_down[:, MU_EX_INDEX].view(-1, 1)

        beta = self.beta_activation(beta_real - beta_at_constraint)
        gamma = self.shifted_elu(gamma_real - gamma_at_constraint)
        mu_up = self.shifted_elu(mu_up_real - mu_up_at_constraint)
        mu_down = self.shifted_elu(mu_down_real - mu_down_at_constraint)
        kappa_up = self.kappa_activation(kappa_up_real)
        kappa_down = self.kappa_activation(kappa_down_real)

        constants_batch = true_constants_PBE.repeat(x_exchange_desc.shape[0], 1).to(
            x_exchange_desc.device
        )
        fill_tensor = torch.ones(
            [x_exchange_desc.shape[0], 20], device=x_exchange_desc.device
        )
        final_tensor = torch.hstack(
            [
                beta,
                gamma,
                fill_tensor,
                kappa_up,
                mu_up,
                kappa_down,
                mu_down,
            ]
        )
        return final_tensor * constants_batch
    


class pcPBEstar(pcPBEMLOptimizer):

    def forward(self, x):

        x_exchange_desc = self.get_exchange_descriptors(x)
        x_correlation_desc = self.get_density_descriptors(x)

        x_correlation_desc_swapped = x_correlation_desc[:, [1, 0, 4, 3, 2, 6, 5]]
        params_c_real = (
            self.hidden_layers_c(x_correlation_desc)
            + self.hidden_layers_c(x_correlation_desc_swapped)
        ) / 2
        beta_real, gamma_real = params_c_real[:, 0].view(-1, 1), params_c_real[
            :, 1
        ].view(-1, 1)

        params_x_up_real = self.hidden_layers_x(x_exchange_desc[:, [2, 5]])
        params_x_down_real = self.hidden_layers_x(x_exchange_desc[:, [4, 6]])
        mu_up_real, kappa_up_real = params_x_up_real[:, 0].view(
            -1, 1
        ), params_x_up_real[:, 1].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, 0].view(
            -1, 1
        ), params_x_down_real[:, 1].view(-1, 1)

        beta = self.beta_activation(beta_real)
        gamma = self.shifted_elu(gamma_real)
        mu_up = self.shifted_elu(mu_up_real)
        mu_down = self.shifted_elu(mu_down_real)
        kappa_up = self.kappa_activation(kappa_up_real)
        kappa_down = self.kappa_activation(kappa_down_real)

        constants_batch = true_constants_PBE.repeat(x_exchange_desc.shape[0], 1).to(
            x_exchange_desc.device
        )
        fill_tensor = torch.ones(
            [x_exchange_desc.shape[0], 20], device=x_exchange_desc.device
        )
        final_tensor = torch.hstack(
            [
                beta,
                gamma,
                fill_tensor,
                kappa_up,
                mu_up,
                kappa_down,
                mu_down,
            ]
        )
        return final_tensor * constants_batch


class pcPBEdoublestar(pcPBEMLOptimizer):

    def __init__(
        self, num_layers, h_dim, nconstants_x=1, nconstants_c=1, dropout=0.2, DFT=None
    ):
        super().__init__(
            num_layers,
            h_dim,
            nconstants_x=nconstants_x,
            nconstants_c=nconstants_c,
            dropout=dropout,
            DFT=DFT,
        )

    @staticmethod
    def _compute_l(a, b, delta=1.0):
        """
        Computes the weighting function l(a - b) = tanh(|a-b|^2 / delta^2)
        """
        dist_sq = torch.sum((a - b) ** 2, dim=1)
        return torch.tanh(dist_sq / delta**2)

    def forward(self, x):
        f0 = 1.0

        x_exchange_desc = self.get_exchange_descriptors(x)
        x_correlation_desc = self.get_density_descriptors(x)

        x_up_input = x_exchange_desc[:, [2, 5]]
        x_down_input = x_exchange_desc[:, [4, 6]]
        x0_full_tensor = self.all_sigma_zero(x_exchange_desc)
        x0_X_sliced = x0_full_tensor[:, [2, 5]]

        f_x_up = self.hidden_layers_x(x_up_input)
        f_x_down = self.hidden_layers_x(x_down_input)
        f_x0 = self.hidden_layers_x(x0_X_sliced)

        Theta_up = f_x_up - f_x0 + f0
        Theta_down = f_x_down - f_x0 + f0
        Fx_up = self.shifted_elu(Theta_up - 1.0).squeeze(-1)
        Fx_down = self.shifted_elu(Theta_down - 1.0).squeeze(-1)

        constraint_points = [
            self.all_sigma_zero(x_correlation_desc),
            self.all_rho_inf(x_correlation_desc),
            self.all_sigma_inf(x_correlation_desc),
        ]
        num_constraints = len(constraint_points)

        f_c = self.hidden_layers_c(x_correlation_desc)
        f_c_at_constraints = [self.hidden_layers_c(p) for p in constraint_points]

        lagrange_weights = []
        for i in range(num_constraints):
            numerator = 1.0
            denominator = 1.0
            for j in range(num_constraints):
                if i != j:
                    numerator = numerator * self._compute_l(
                        x_correlation_desc, constraint_points[j]
                    )
                    denominator = denominator * self._compute_l(
                        constraint_points[i], constraint_points[j]
                    )

            weight = numerator / (denominator + 1e-9)
            lagrange_weights.append(weight)

        thetas = [f_c - f_c_at_constraints[i] + f0 for i in range(num_constraints)]

        Fc_intermediate = torch.zeros_like(f_c)
        for i in range(num_constraints):
            Fc_intermediate += thetas[i] * lagrange_weights[i].view(-1, 1)

        sum_of_weights = torch.sum(torch.stack(lagrange_weights, dim=0), dim=0)

        Fc_intermediate = Fc_intermediate / (sum_of_weights.view(-1, 1) + 1e-9)

        Fc = self.shifted_elu(Fc_intermediate - 1.0).squeeze(-1)

        return Fx_up, Fx_down, Fc


def test_model_constraints(model, model_name):
    """
    Runs a suite of tests for constraints on a given model.
    """
    print(f"--- Running Constraint Verification for {model_name} ---")

    BATCH_SIZE = 8
    model.eval()
    
    # Determine input dimensionality based on model type
    if isinstance(model, pcPBELMLOptimizer):
        input_dim = LLMGGA_DESCRIPTOR_DIMENSIONALITY  # LLMGGA includes laplacian
    else:
        input_dim = MGGA_DESCRIPTOR_DIMENSIONALITY  # Standard MGGA descriptors
    
    nn_inputs_placeholder = torch.zeros(
        BATCH_SIZE, input_dim, device=next(model.parameters()).device
    )
    all_passed = True

    rho_a = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device) * 5 + 0.1
    rho_b = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device) * 5 + 0.1
    grad_zero = torch.zeros(BATCH_SIZE, device=nn_inputs_placeholder.device)
    lapl_zero = torch.zeros(BATCH_SIZE, device=nn_inputs_placeholder.device)

    if isinstance(model, pcPBEdoublestar):
        print("\n[1/4] Checking Exchange UEG Limit (Fx -> 1.0)...")
        tau_tf_2rho_a = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * (2 * rho_a) ** (5 / 3)
        tau_tf_2rho_b = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * (2 * rho_b) ** (5 / 3)
        tau_a_test = tau_tf_2rho_a / 2
        tau_b_test = tau_tf_2rho_b / 2
        Fx_up, Fx_down, _ = model(
            torch.stack([rho_a,
            rho_b,
            grad_zero,
            grad_zero,
            grad_zero,
            tau_a_test,
            tau_b_test,], dim=1)
        )
        fx_up_passed = torch.allclose(Fx_up, torch.ones_like(Fx_up))
        fx_down_passed = torch.allclose(Fx_down, torch.ones_like(Fx_down))
        print(
            f"Fx (spin-up) -> 1.0? {'PASS' if fx_up_passed else 'FAIL'} (Mean: {Fx_up.mean().item():.6f})"
        )
        print(
            f"Fx (spin-down) -> 1.0? {'PASS' if fx_down_passed else 'FAIL'} (Mean: {Fx_down.mean().item():.6f})"
        )
        if not (fx_up_passed and fx_down_passed):
            all_passed = False

        print("\n[2/4] Checking Correlation UEG Limit (Fc -> 1.0)...")
        tau_tf_rho_a = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * rho_a ** (5 / 3)
        tau_tf_rho_b = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * rho_b ** (5 / 3)
        _, _, Fc_ueg = model(
            torch.stack([rho_a,
            rho_b,
            grad_zero,
            grad_zero,
            grad_zero,
            tau_tf_rho_a,
            tau_tf_rho_b,], dim=1)
        )
        fc_ueg_passed = torch.allclose(Fc_ueg, torch.ones_like(Fc_ueg))
        print(
            f"Fc -> 1.0? {'PASS' if fc_ueg_passed else 'FAIL'} (Mean: {Fc_ueg.mean().item():.6f})"
        )
        if not fc_ueg_passed:
            all_passed = False

        print("\n[3/4] Checking Correlation High-Density Limit (Fc -> 1.0)...")
        rho_large = torch.full((BATCH_SIZE,), 1e7, device=nn_inputs_placeholder.device)
        rand_grads = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device) * 10
        rand_taus = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device) * 100
        _, _, Fc_hd = model(
            torch.stack([rho_large,
            rho_large,
            rand_grads,
            rand_grads,
            rand_grads * 2,
            rand_taus,
            rand_taus,], dim=1)
        )
        fc_hd_passed = torch.allclose(Fc_hd, torch.ones_like(Fc_hd), atol=1e-5)
        print(
            f"Fc -> 1.0? {'PASS' if fc_hd_passed else 'FAIL'} (Mean: {Fc_hd.mean().item():.6f})"
        )
        if not fc_hd_passed:
            all_passed = False

    elif isinstance(model, pcPBEMLOptimizer):
        true_factors = true_constants_PBE.repeat(BATCH_SIZE, 1).to(
            nn_inputs_placeholder.device
        )

        print("\n[TEST 1/4] Checking `mu` constraint (Exchange UEG Limit)...")
        tau_tf_2rho_a = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * (2 * rho_a) ** (5 / 3)
        tau_tf_2rho_b = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * (2 * rho_b) ** (5 / 3)
        tau_a_test = tau_tf_2rho_a / 2
        tau_b_test = tau_tf_2rho_b / 2
        
        if isinstance(model, pcPBELMLOptimizer):
            # pcPBELMLOptimizer requires 9 dimensions (includes Laplacian)
            output_mu = (
                model(
                    torch.stack([rho_a,
                    rho_b,
                    grad_zero,
                    grad_zero,
                    grad_zero,
                    tau_a_test,
                    tau_b_test,
                    lapl_zero,
                    lapl_zero,], dim=1)
                )
                / true_factors
            )
        else:
            # pcPBEMLOptimizer uses 7 dimensions
            output_mu = (
                model(
                    torch.stack([rho_a,
                    rho_b,
                    grad_zero,
                    grad_zero,
                    grad_zero,
                    tau_a_test,
                    tau_b_test,], dim=1)
                )
                / true_factors
            )
        mu_up, mu_down = output_mu[:, 23], output_mu[:, 25]
        mu_up_passed = torch.allclose(mu_up, torch.ones_like(mu_up))
        mu_down_passed = torch.allclose(mu_down, torch.ones_like(mu_down))
        print(
            f"Mu (spin-up) -> 1.0? {'PASS' if mu_up_passed else 'FAIL'} (Mean: {mu_up.mean().item():.6f})"
        )
        print(
            f"Mu (spin-down) -> 1.0? {'PASS' if mu_down_passed else 'FAIL'} (Mean: {mu_down.mean().item():.6f})"
        )
        if not (mu_up_passed and mu_down_passed):
            all_passed = False

        print("\n[TEST 2/4] Checking `beta` constraint (Correlation UEG Limit)...")
        tau_tf_rho_a = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * rho_a ** (5 / 3)
        tau_tf_rho_b = 3 / 10 * (3 * np.pi**2) ** (2 / 3) * rho_b ** (5 / 3)
        
        if isinstance(model, pcPBELMLOptimizer):
            output_beta = (
                model(
                    torch.stack([rho_a,
                    rho_b,
                    grad_zero,
                    grad_zero,
                    grad_zero,
                    tau_tf_rho_a,
                    tau_tf_rho_b,
                    lapl_zero,
                    lapl_zero,], dim=1)
                )
                / true_factors
            )
        else:
            output_beta = (
                model(
                    torch.stack([rho_a,
                    rho_b,
                    grad_zero,
                    grad_zero,
                    grad_zero,
                    tau_tf_rho_a,
                    tau_tf_rho_b,], dim=1)
                )
                / true_factors
            )
        beta = output_beta[:, 0]
        beta_passed = torch.allclose(beta, torch.ones_like(beta))
        print(
            f"Beta -> 1.0? {'PASS' if beta_passed else 'FAIL'} (Mean: {beta.mean().item():.6f})"
        )
        if not beta_passed:
            all_passed = False

        print(
            "\n[TEST 3/4] Checking `gamma` constraint (Correlation High-Density Limit)..."
        )
        rho_large = torch.full((BATCH_SIZE,), 1e6, device=nn_inputs_placeholder.device)
        rand_grads = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device)
        rand_taus = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device)
        rand_lapl = torch.rand(BATCH_SIZE, device=nn_inputs_placeholder.device)
        
        if isinstance(model, pcPBELMLOptimizer):
            output_gamma = (
                model(
                    torch.stack([rho_large,
                    rho_large,
                    rand_grads,
                    rand_grads,
                    rand_grads,
                    rand_taus,
                    rand_taus,
                    rand_lapl,
                    rand_lapl,], dim=1)
                )
                / true_factors
            )
        else:
            output_gamma = (
                model(
                    torch.stack([rho_large,
                    rho_large,
                    rand_grads,
                    rand_grads,
                    rand_grads,
                    rand_taus,
                    rand_taus,], dim=1)
                )
                / true_factors
            )
        gamma = output_gamma[:, 1]
        gamma_passed = torch.allclose(gamma, torch.ones_like(gamma))
        print(
            f"Gamma -> 1.0? {'PASS' if gamma_passed else 'FAIL'} (Mean: {gamma.mean().item():.6f})"
        )
        if not gamma_passed:
            all_passed = False

        print("\n[TEST 4/4] Checking Spin Symmetry...")
        rho1, rho2 = torch.rand(BATCH_SIZE) + 0.1, torch.rand(BATCH_SIZE) + 0.1
        grad1, grad2 = torch.rand(BATCH_SIZE), torch.rand(BATCH_SIZE)
        tau1, tau2 = torch.rand(BATCH_SIZE) * 10, torch.rand(BATCH_SIZE) * 10
        lapl1, lapl2 = torch.rand(BATCH_SIZE), torch.rand(BATCH_SIZE)
        
        if isinstance(model, pcPBELMLOptimizer):
            out1 = model(
                torch.stack([rho1, rho2, grad1, grad1 + grad2, grad2, tau1, tau2, lapl1, lapl2], dim=1)
            )
            out2 = model(
                torch.stack([rho2, rho1, grad2, grad1 + grad2, grad1, tau2, tau1, lapl2, lapl1], dim=1)
            )
        else:
            out1 = model(
                torch.stack([rho1, rho2, grad1, grad1 + grad2, grad2, tau1, tau2], dim=1)
            )
            out2 = model(
                torch.stack([rho2, rho1, grad2, grad1 + grad2, grad1, tau2, tau1,], dim=1)
            )
        corr_symm = torch.allclose(out1[:, 0], out2[:, 0]) and torch.allclose(
            out1[:, 1], out2[:, 1]
        )
        exch_swap = torch.allclose(out1[:, 22], out2[:, 24]) and torch.allclose(
            out1[:, 24], out2[:, 22]
        )
        print(
            f"  > Correlation parameters are symmetric? {'PASS' if corr_symm else 'FAIL'}"
        )
        print(
            f"  > Exchange parameters correctly swapped? {'PASS' if exch_swap else 'FAIL'}"
        )
        if not (corr_symm and exch_swap):
            all_passed = False

    else:
        print(f"ERROR: Model type {model_name} not recognized for testing.")
        return False

    print("\n--- Summary ---")
    if all_passed:
        print(f"SUCCESS: All constraints for {model_name} were met.")
    else:
        print(f"FAILURE: One or more constraints for {model_name} were not satisfied.")

    return all_passed


if __name__ == "__main__":
    pbe_optimizer_model = pcPBEMLOptimizer(num_layers=4, h_dim=16)
    test_model_constraints(pbe_optimizer_model, "pcPBEMLOptimizer")

    print("\n" + "=" * 60 + "\n")
    pbe_pbel_model = pcPBELMLOptimizer(num_layers=4, h_dim=16)
    test_model_constraints(pbe_pbel_model, "pcPBELMLOptimizer")

    print("\n" + "=" * 60 + "\n")
    pbe_doublestar_model = pcPBEdoublestar(num_layers=4, h_dim=16)
    test_model_constraints(pbe_doublestar_model, "pcPBEdoublestar")
