"""
Neural network models for test_models - re-exported from train_models.

This module re-exports the NN models from train_models for use in test_models.
Both packages use identical base implementations. If test_models needs different
behavior, specific classes and methods can be overridden here.
"""

import sys
from pathlib import Path

import torch

# Add train_models to path to import base models
train_models_path = Path(__file__).parent.parent.parent / "train_models"
sys.path.insert(0, str(train_models_path))

from NN_models import (
    MLOptimizer as MLOptimizer_train,
    pcPBEMLOptimizer as pcPBEMLOptimizer_train,
    pcPBELMLOptimizer as pcPBELMLOptimizer_train,
    pcPBEdoublestar as pcPBEdoublestar_train,
    true_constants_PBE
)


class MLOptimizer(MLOptimizer_train):

    @staticmethod
    def get_density_descriptors(
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):

        eps_rho = 1e-10
        eps_sigma = 1e-30

        nn_inputs[:, 0] = rho_a_inp ** (1 / 3)
        nn_inputs[:, 1] = rho_b_inp ** (1 / 3)
        nn_inputs[:, 2] = (
            torch.sqrt(grad_a_inp + eps_sigma)
            / rho_a_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 3] = (
            torch.sqrt(grad_inp + eps_sigma)
            / (rho_a_inp + rho_b_inp - eps_rho) ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 4] = (
            torch.sqrt(grad_b_inp + eps_sigma)
            / rho_b_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        tau_tf_alpha = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_a_inp) ** (5 / 3)
        tau_tf_beta = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_b_inp) ** (5 / 3)
        tau_w_alpha = grad_a_inp / (8 * rho_a_inp)
        tau_w_beta = grad_b_inp / (8 * rho_b_inp)
        nn_inputs[:, 5] = (tau_a_inp - tau_w_alpha) / tau_tf_alpha - 1
        nn_inputs[:, 6] = (tau_b_inp - tau_w_beta) / tau_tf_beta - 1

        return torch.tanh(nn_inputs)

    def forward(
        self,
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):
        """
        Returns:
            spin-symmetrized enhancement factor for LDA exhange energy
        """

        x = self.get_density_descriptors(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
        )

        result = (
            self.unsymm_forward(x) + self.unsymm_forward(x[:, [1, 0, 4, 3, 2, 6, 5]])
        ) / 2

        return result


class pcPBEMLOptimizer(pcPBEMLOptimizer_train):

    @staticmethod
    def get_density_descriptors(
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):

        eps_rho = 1e-10
        eps_sigma = 1e-30

        nn_inputs[:, 0] = rho_a_inp ** (1 / 3)
        nn_inputs[:, 1] = rho_b_inp ** (1 / 3)
        nn_inputs[:, 2] = (
            torch.sqrt(grad_a_inp + eps_sigma)
            / rho_a_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 3] = (
            torch.sqrt(grad_inp + eps_sigma)
            / (rho_a_inp + rho_b_inp - eps_rho) ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 4] = (
            torch.sqrt(grad_b_inp + eps_sigma)
            / rho_b_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        tau_tf_alpha = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_a_inp) ** (5 / 3)
        tau_tf_beta = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_b_inp) ** (5 / 3)
        tau_w_alpha = grad_a_inp / (8 * rho_a_inp)
        tau_w_beta = grad_b_inp / (8 * rho_b_inp)
        nn_inputs[:, 5] = (tau_a_inp - tau_w_alpha) / tau_tf_alpha - 1
        nn_inputs[:, 6] = (tau_b_inp - tau_w_beta) / tau_tf_beta - 1

        return torch.tanh(nn_inputs)

    def forward(
        self,
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):

        x_exchange_desc = self.get_density_descriptors(
            nn_inputs,
            2 * rho_a_inp,
            2 * rho_b_inp,
            4 * grad_a_inp,
            4 * grad_b_inp,
            4 * grad_inp,
            2 * tau_a_inp,
            2 * tau_b_inp,
        )

        x_correlation_desc = self.get_density_descriptors(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
        )

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

    def forward(
        self,
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):

        x_exchange_desc = self.get_density_descriptors(
            nn_inputs,
            2 * rho_a_inp,
            2 * rho_b_inp,
            4 * grad_a_inp,
            4 * grad_b_inp,
            4 * grad_inp,
            2 * tau_a_inp,
            2 * tau_b_inp,
        )

        x_correlation_desc = self.get_density_descriptors(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
        )

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


class pcPBEdoublestar(pcPBEMLOptimizer, pcPBEdoublestar_train):

    def forward(
        self,
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
    ):
        f0 = 1.0

        x_exchange_desc = self.get_density_descriptors(
            nn_inputs,
            2 * rho_a_inp,
            2 * rho_b_inp,
            4 * grad_a_inp,
            4 * grad_b_inp,
            4 * grad_inp,
            2 * tau_a_inp,
            2 * tau_b_inp,
        )

        x_correlation_desc = self.get_density_descriptors(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
        )

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
    
    
class pcPBELMLOptimizer(pcPBELMLOptimizer_train):

    @staticmethod
    def get_density_descriptors(
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
        lapl_a_inp,
        lapl_b_inp,
    ):

        
        eps_rho = 1e-10
        eps_sigma = 1e-30

        nn_inputs[:, 0] = rho_a_inp ** (1 / 3)
        nn_inputs[:, 1] = rho_b_inp ** (1 / 3)
        nn_inputs[:, 2] = (
            torch.sqrt(grad_a_inp + eps_sigma)
            / rho_a_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 3] = (
            torch.sqrt(grad_inp + eps_sigma)
            / (rho_a_inp + rho_b_inp - eps_rho) ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        nn_inputs[:, 4] = (
            torch.sqrt(grad_b_inp + eps_sigma)
            / rho_b_inp ** (4 / 3)
            / (3 * torch.pi**2) ** (1 / 3)
            / 2
        )
        tau_tf_alpha = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_a_inp) ** (5 / 3)
        tau_tf_beta = 3 / 10 * (3 * torch.pi**2) ** (2 / 3) * (rho_b_inp) ** (5 / 3)
        tau_w_alpha = grad_a_inp / (8 * rho_a_inp)
        tau_w_beta = grad_b_inp / (8 * rho_b_inp)
        nn_inputs[:, 5] = (tau_a_inp - tau_w_alpha) / tau_tf_alpha - 1
        nn_inputs[:, 6] = (tau_b_inp - tau_w_beta) / tau_tf_beta - 1
        nn_inputs[:, 7] = lapl_a_inp / (4 * (3 * torch.pi**2) ** (2 / 3) * (rho_a_inp) ** (5/3))
        nn_inputs[:, 8] = lapl_b_inp/ (4 * (3 * torch.pi**2) ** (2 / 3) * (rho_b_inp) ** (5/3))
        

        return torch.tanh(nn_inputs)
    

    def forward(
        self,
        nn_inputs,
        rho_a_inp,
        rho_b_inp,
        grad_a_inp,
        grad_b_inp,
        grad_inp,
        tau_a_inp,
        tau_b_inp,
        lapl_a_inp,
        lapl_b_inp,
    ):

        x_exchange_desc = self.get_density_descriptors(
            nn_inputs,
            2 * rho_a_inp,
            2 * rho_b_inp,
            4 * grad_a_inp,
            4 * grad_b_inp,
            4 * grad_inp,
            2 * tau_a_inp,
            2 * tau_b_inp,
            4 * lapl_a_inp,
            4 * lapb_b_inp,
        )

        x_correlation_desc = self.get_density_descriptors(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
            lapl_a_inp,
            lapl_b_inp,
        )

        x_correlation_desc_swapped = x_correlation_desc[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]]
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

        params_x_up_real = self.hidden_layers_x(x_exchange_desc[:, [2, 5, 7]])
        params_x_down_real = self.hidden_layers_x(x_exchange_desc[:, [4, 6, 8]])
        mu_up_real, kappa_up_real = params_x_up_real[:, 0].view(
            -1, 1
        ), params_x_up_real[:, 1].view(-1, 1)
        mu_down_real, kappa_down_real = params_x_down_real[:, 0].view(
            -1, 1
        ), params_x_down_real[:, 1].view(-1, 1)

        x_exch_s_zero = self.all_sigma_zero(x_exchange_desc)
        params_x_s_zero_up = self.hidden_layers_x(x_exch_s_zero[:, [2, 5, 7]])
        params_x_s_zero_down = self.hidden_layers_x(x_exch_s_zero[:, [4, 6, 8]])
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



def NN_XALPHA_model(num_layers=6, h_dim=128, nconstants=1, dropout=0.0, DFT="XALPHA"):
    return MLOptimizer(
        num_layers=num_layers,
        h_dim=h_dim,
        nconstants=nconstants,
        dropout=dropout,
        DFT=DFT,
    )


def NN_PBE_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEMLOptimizer(
        num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT
    )


def NN_PBE_star_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEstar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT)


def NN_PBE_star_star_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEdoublestar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT)
