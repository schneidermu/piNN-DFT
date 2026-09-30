"""Tau-free NN-PBE, with separate architecture and unchanged V2 parameterization."""
# ruff: noqa: N999 -- retain the repository's NN_models naming convention

import math

import torch
from NN_models import pcPBELMLOptimizerV2
from torch import nn

from dft_functionals.constants import EPS_RHO, EPS_SIGMA, NN_OUTPUT_SCALE_PBE

ARCHITECTURE = "pcPBELMLOptimizerV2Lap-v1"
DESCRIPTOR_PROTOCOL = "rho-sigma-total-lapl-tau-free-v1"


class pcPBELMLOptimizerV2Lap(pcPBELMLOptimizerV2):
    """Raw N×9 input; columns 5:7 (tau) are never read.

    Correlation: n_a,n_b,s_a,s_total,s_b,q_a,q_b,zeta.
    Exchange: spin-scaled (s_sigma,q_sigma), with tied weights.
    A deterministic functional requires dropout=0 (including during training).
    """

    def __init__(self, num_layers=6, h_dim=32, dropout=0.0, **kwargs):
        if dropout != 0:
            raise ValueError("Lap energy/derivative consistency requires dropout=0.")
        super().__init__(num_layers, h_dim, dropout=dropout, **kwargs)
        self.c_input_layers[0] = nn.Linear(8, h_dim, bias=False)
        self.x_feature_extractor[0] = nn.Linear(2, h_dim, bias=False)
        self.register_buffer("lap_architecture_version", torch.tensor(1))
        self.architecture = ARCHITECTURE
        self.descriptor_protocol = DESCRIPTOR_PROTOCOL
        self.model_kwargs = {
            "num_layers": num_layers,
            "h_dim": h_dim,
            "dropout": dropout,
            "use_g_x": self.use_g_x,
            "use_g_c": self.use_g_c,
        }

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if (
            "lap_architecture_version" not in state_dict
            or int(state_dict["lap_architecture_version"]) != 1
        ):
            raise ValueError(
                "Architecture mismatch: a tau checkpoint is not a Lap checkpoint."
            )
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @staticmethod
    def all_sigma_zero(x):
        return torch.cat([x[:, :2], torch.zeros_like(x[:, 2:7]), x[:, 7:8]], 1)

    @staticmethod
    def all_rho_inf(x):
        return torch.cat([torch.ones_like(x[:, :2]), x[:, 2:]], 1)

    @staticmethod
    def all_s_inf(x):
        # Preserve q and zeta; rapid gradients do not imply zero curvature.
        return torch.cat([x[:, :2], torch.ones_like(x[:, 2:5]), x[:, 5:]], 1)

    @staticmethod
    def get_correlation_descriptors(x):
        a, b = x[:, 0], x[:, 1]
        n = (x[:, :2] + EPS_RHO).pow(1 / 3)
        den = torch.stack([a, a + b, b], 1) + EPS_RHO
        # Subtract the regularizer's zero value so the exact UEG maps to s=0.
        s = (torch.sqrt(x[:, 2:5] + EPS_SIGMA) - math.sqrt(EPS_SIGMA)) / (
            2 * (3 * math.pi**2) ** (1 / 3) * den.pow(4 / 3)
        )
        q = x[:, 7:9] / (
            4 * (3 * math.pi**2) ** (2 / 3) * (x[:, :2] + EPS_RHO).pow(5 / 3)
        )
        zeta = ((a - b) / (a + b + EPS_RHO)).unsqueeze(1)
        return torch.cat([torch.tanh(torch.cat([n, s, q], 1)), zeta], 1)

    get_exchange_descriptors = get_correlation_descriptors

    @staticmethod
    def swap_descriptors(d):
        return torch.cat([d[:, [1, 0, 4, 3, 2, 6, 5]], -d[:, 7:8]], 1)

    def exchange_hidden(self, d):
        return self.x_feature_extractor(d[:, [2, 5]]), self.x_feature_extractor(
            d[:, [4, 6]]
        )

    def correlation_output(self, d, hx):
        h = self.c_symmetrization_blocks(self.c_input_layers(d))
        hs = self.c_symmetrization_blocks(self.c_input_layers(self.swap_descriptors(d)))
        h = self.c_post_symm_blocks((h + hs) / 2)
        return self.c_output_layer(torch.cat([h, hx], 1))

    def forward_descriptors(self, d, ex):
        """Also exposes exact bounded descriptor anchors for direct tests."""
        up, down = self.exchange_hidden(ex)
        hx = (up + down) / 2
        xu, xd = self.x_output_layer(up), self.x_output_layer(down)
        ue = self.all_sigma_zero(ex)
        hu, hd = self.exchange_hidden(ue)
        au, ad = self.x_output_layer(hu), self.x_output_layer(hd)
        x1, x2, x3 = self.all_sigma_zero(d), self.all_rho_inf(d), self.all_s_inf(d)
        real = self.correlation_output(d, hx)
        c1 = self.correlation_output(x1, (hu + hd) / 2)
        c2 = self.correlation_output(x2, hx)
        c3 = self.correlation_output(x3, hx)
        beta = self.beta_activation(real[:, :1] - c1[:, :1])
        gamma = self.gamma_activation(real[:, 1:2] - c2[:, 1:2])
        gx = (
            torch.cat(
                [
                    self.g_nn_activation(xu[:, 2:3] - au[:, 2:3]),
                    self.g_nn_activation(xd[:, 2:3] - ad[:, 2:3]),
                ],
                1,
            )
            if self.use_g_x
            else d.new_zeros((len(d), 2))
        )
        # Anchor network fusion must match the real fusion at each anchor.
        # At UEG ex=ue; at high density and s infinity exchange is held fixed,
        # exactly as in V2's distance-weighted construction.
        gc = (
            self.activate_g_c(
                self._lagrange_correct_Gc(
                    real[:, 2:3], c1[:, 2:3], c2[:, 2:3], c3[:, 2:3], d, x1, x2, x3
                )
            )
            if self.use_g_c
            else d.new_ones((len(d), 1))
        )
        out = torch.cat(
            [
                beta,
                gamma,
                d.new_ones((len(d), 20)),
                self.kappa_activation(xu[:, 1:2]),
                self.shifted_elu(xu[:, :1] - au[:, :1]),
                self.kappa_activation(xd[:, 1:2]),
                self.shifted_elu(xd[:, :1] - ad[:, :1]),
                gx,
                gc,
            ],
            1,
        )
        return out * NN_OUTPUT_SCALE_PBE.to(out)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 9:
            raise ValueError(
                "Lap model expects raw (N,9) rho/sigma-total/tau-placeholder/lapl."
            )
        # Do not even multiply tau placeholders by the legacy scaling buffer.
        scaled = torch.cat(
            [2 * x[:, :2], 4 * x[:, 2:5], torch.zeros_like(x[:, 5:7]), 2 * x[:, 7:9]], 1
        )
        return self.forward_descriptors(
            self.get_correlation_descriptors(x), self.get_exchange_descriptors(scaled)
        )
