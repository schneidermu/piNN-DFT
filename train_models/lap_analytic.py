"""Deterministic analytic densities/functionals for convergence diagnostics."""

import torch
from lap_vxc import stencil_coordinates
from torch import nn


class PolynomialEnergy(nn.Module):
    def __init__(self, a=0.7, b=0.0, c=0.0, dtype=torch.float64):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([a, b, c], dtype=dtype))

    def forward(self, rho, sigma, lap):
        a, b, c = self.coefficients
        return (
            a * rho.square().sum(-1)
            + b * (sigma[:, 0] + sigma[:, 2])
            + c * lap.square().sum(-1)
        )


def analytic_features(xyz, equal_spin=False):
    """rho_s=4+amplitude_s sum cos(x_i); exact grad/lap, bi-lap=-lap."""
    amplitude = xyz.new_tensor([0.25, 0.25 if equal_spin else 0.4])
    rho = 4 + xyz.cos().sum(-1, keepdim=True) * amplitude
    grad = -xyz.sin().unsqueeze(-2) * amplitude[:, None]
    lap = -xyz.cos().sum(-1, keepdim=True) * amplitude
    return torch.cat([rho, grad.flatten(-2), lap], -1)


def features(h, equal_spin=False, n=11, dtype=torch.float64):
    xyz = torch.linspace(0.15, 1.7, 3 * n, dtype=torch.float64).reshape(n, 3)
    return analytic_features(stencil_coordinates(xyz, h), equal_spin).to(dtype)
