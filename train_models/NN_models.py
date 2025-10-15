import random

import numpy as np
import torch
from torch import nn

random.seed(42)

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

true_constants_PBE = torch.Tensor(
    [
        [
            0.06672455,
            (1 - torch.log(torch.Tensor([2]))) / (np.pi**2),
            1.709921,
            7.5957,
            14.1189,
            10.357,
            3.5876,
            6.1977,
            3.6231,
            1.6382,
            3.3662,
            0.88026,
            0.49294,
            0.62517,
            0.49671,
            # 1,  1,  1,
            0.031091,
            0.015545,
            0.016887,
            0.21370,
            0.20548,
            0.11125,
            -3 / 8 * (3 / np.pi) ** (1 / 3) * 4 ** (2 / 3),
            0.8040,
            0.2195149727645171,
            0.8040,
            0.2195149727645171,
        ]
    ]
)  # .to(device)

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
                nn.Linear(7, h_dim, bias=False),
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

        eps_sigma = 1e-30
        eps_rho = 1e-10

        n_alpha = x[:, 0] ** (1 / 3)
        n_beta = x[:, 1] ** (1 / 3)

        s_alpha = (
            torch.sqrt(x[:, 2] + eps_sigma)
            / (x[:, 0] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_norm = (
            torch.sqrt(x[:, 3] + eps_sigma)
            / (x[:, 0] + x[:, 1] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_beta = (
            torch.sqrt(x[:, 4] + eps_sigma)
            / (x[:, 1] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )

        tau_tf_alpha = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, 0] + eps_rho) ** (5 / 3)
        )
        tau_tf_beta = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, 1] + eps_rho) ** (5 / 3)
        )
        tau_w_alpha = x[:, 2] / (8 * (x[:, 0] + eps_rho))
        tau_w_beta = x[:, 4] / (8 * (x[:, 1] + eps_rho))

        tau_alpha = (x[:, 5] - tau_w_alpha) / tau_tf_alpha
        tau_beta = (x[:, 6] - tau_w_beta) / tau_tf_beta

        X = torch.stack(
            [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha, tau_beta], dim=1
        )

        X[:, 5:] = X[:, 5:] - 1
        X = torch.tanh(X)

        return X

    def forward(self, x):
        """
        Returns:
            spin-symmetrized enhancement factor for LDA exhange energy
        """

        x = self.get_density_descriptors(x)

        result = (
            self.unsymm_forward(x) + self.unsymm_forward(x[:, [1, 0, 4, 3, 2, 6, 5]])
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
            nn.Linear(7, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        input_layer_x = [
            nn.Linear(2, h_dim, bias=False),
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

        x_x_up = self.hidden_layers_x(x[:, [2, 5]])  # Slice out density descriptors
        x_x_down = self.hidden_layers_x(x[:, [4, 6]])

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
        return torch.hstack([x[:, :2], torch.zeros([x.shape[0], 5]).to(x.device)])

    @staticmethod
    def all_sigma_inf(x):
        """
        Function for PW91 correlation parameters constraint
        """
        return torch.hstack(
            [x[:, :2], torch.ones([x.shape[0], 3]).to(x.device), x[:, 5:]]
        )

    @staticmethod
    def all_rho_inf(x):
        """
        Function for parameter gamma constraint
        """
        return torch.hstack([torch.ones([x.shape[0], 2]).to(x.device), x[:, 2:]])

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

        eps_sigma = 1e-30
        eps_rho = 1e-10

        n_alpha = x[:, 0] ** (1 / 3)
        n_beta = x[:, 1] ** (1 / 3)

        s_alpha = (
            torch.sqrt(x[:, 2] + eps_sigma)
            / (x[:, 0] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_norm = (
            torch.sqrt(x[:, 3] + eps_sigma)
            / (x[:, 0] + x[:, 1] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )
        s_beta = (
            torch.sqrt(x[:, 4] + eps_sigma)
            / (x[:, 1] + eps_rho) ** (4 / 3)
            / (3 * np.pi**2) ** (1 / 3)
            / 2
        )

        tau_tf_alpha = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, 0] + eps_rho) ** (5 / 3)
        )
        tau_tf_beta = (
            3 / 10 * (3 * np.pi**2) ** (2 / 3) * (x[:, 1] + eps_rho) ** (5 / 3)
        )
        tau_w_alpha = x[:, 2] / (8 * (x[:, 0] + eps_rho))
        tau_w_beta = x[:, 4] / (8 * (x[:, 1] + eps_rho))

        tau_alpha = (x[:, 5] - tau_w_alpha) / tau_tf_alpha
        tau_beta = (x[:, 6] - tau_w_beta) / tau_tf_beta

        X = torch.stack(
            [n_alpha, n_beta, s_alpha, s_norm, s_beta, tau_alpha, tau_beta], dim=1
        )

        X[:, 5:] = X[:, 5:] - 1
        X = torch.tanh(X)

        return X

    def get_exchange_descriptors(self, x):
        scaling_array = torch.tensor([2, 2, 4, 4, 4, 2, 2]).to(x.device)

        return self.get_density_descriptors(scaling_array * x)

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

        x_corr_sigma_zero = self.all_sigma_zero(x_correlation_desc)
        x_corr_sigma_zero_swapped = self.all_sigma_zero(x_correlation_desc_swapped)
        params_c_sigma_zero = (
            self.hidden_layers_c(x_corr_sigma_zero)
            + self.hidden_layers_c(x_corr_sigma_zero_swapped)
        ) / 2
        beta_at_constraint = params_c_sigma_zero[:, 0].view(-1, 1)

        x_corr_rho_inf = self.all_rho_inf(x_correlation_desc)
        x_corr_rho_inf_swapped = self.all_rho_inf(x_correlation_desc_swapped)
        params_c_rho_inf = (
            self.hidden_layers_c(x_corr_rho_inf)
            + self.hidden_layers_c(x_corr_rho_inf_swapped)
        ) / 2
        gamma_at_constraint = params_c_rho_inf[:, 1].view(-1, 1)

        params_x_up_real = self.hidden_layers_x(x_exchange_desc[:, [2, 5]])
        params_x_down_real = self.hidden_layers_x(x_exchange_desc[:, [4, 6]])
        mu_up_real, kappa_up_real = params_x_up_real[:, 0].view(
            -1, 1
        ), params_x_up_real[:, 1].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, 0].view(
            -1, 1
        ), params_x_down_real[:, 1].view(-1, 1)

        x_exch_s_zero = self.all_sigma_zero(x_exchange_desc)
        params_x_s_zero_up = self.hidden_layers_x(x_exch_s_zero[:, [2, 5]])
        params_x_s_zero_down = self.hidden_layers_x(x_exch_s_zero[:, [4, 6]])
        mu_up_at_constraint = params_x_s_zero_up[:, 0].view(-1, 1)
        mu_down_at_constraint = params_x_s_zero_down[:, 0].view(-1, 1)

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
