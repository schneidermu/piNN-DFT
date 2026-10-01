"""Independent spin-resolved PBE Euler-potential reference for diagnostics.

The reference evaluates the repository's canonical PBE energy density and forms
the GGA Euler derivative

    v_s = d e / d rho_s - div(d e / d grad(rho_s)).

AO values, first derivatives, and second derivatives are evaluated directly at
each requested Cartesian point. No grid interpolation or spatial finite
differences are used. Coordinates and AO derivatives use Bohr units.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Keep this module runnable both as `python train_models/...py` and from pytest
# with train_models on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dft_functionals import PBE
from dft_functionals.constants import PBE_CONSTANTS


@dataclass(frozen=True)
class AOSpinDensityDerivatives:
    """Spin density and Cartesian derivatives at N points.

    Shapes are density (N, 2), gradient (N, 2, 3), and Hessian
    (N, 2, 3, 3), with spin order alpha, beta and Cartesian order x, y, z.
    """

    density: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray

    @property
    def laplacian(self) -> np.ndarray:
        return np.trace(self.hessian, axis1=-2, axis2=-1)


def sigma_from_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Convert spin gradients (N,2,3) to standard (aa,ab,bb) sigma."""
    if gradient.ndim != 3 or gradient.shape[1:] != (2, 3):
        raise ValueError("Spin gradients must have shape (N,2,3).")
    alpha, beta = gradient[:, 0], gradient[:, 1]
    return torch.stack(
        [
            (alpha * alpha).sum(-1),
            (alpha * beta).sum(-1),
            (beta * beta).sum(-1),
        ],
        dim=-1,
    )


def pbe_energy_density(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return canonical repository PBE energy per volume at each point."""
    if rho.ndim != 2 or rho.shape[1] != 2:
        raise ValueError("Spin densities must have shape (N,2).")
    if sigma.shape != (len(rho), 3):
        raise ValueError("Standard sigma must have shape (N,3) in aa,ab,bb order.")
    constants = PBE_CONSTANTS.to(dtype=rho.dtype, device=rho.device).expand(
        len(rho), -1
    )
    epsilon = PBE.F_PBE(rho, sigma, constants, rho.device)
    return epsilon * rho.sum(dim=-1)


def euler_potential_from_gga(
    rho: torch.Tensor,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    energy_density: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Evaluate a spin-resolved GGA Euler derivative from local AO data.

    ``energy_density`` receives ``rho[N,2]`` and standard sigma ``[N,3]`` and
    returns the energy density per volume ``[N]``. Local first and second
    derivatives are obtained by autograd. The spatial divergence is assembled
    analytically from the density gradients and Hessians:

      div A_a = 2 e_aa lap(rho_a) + e_ab lap(rho_b)
                + 2 grad(e_aa).grad(rho_a) + grad(e_ab).grad(rho_b),

    with the analogous beta-spin expression. Here A_s is the derivative of the
    energy density with respect to grad(rho_s), so this is the standard
    spin-resolved LibXC sigma convention, not a total-gradient convention.
    """
    if rho.ndim != 2 or rho.shape[1] != 2 or len(rho) == 0:
        raise ValueError("rho must be a nonempty (N,2) tensor.")
    if gradient.shape != (len(rho), 2, 3):
        raise ValueError("gradient must have shape (N,2,3).")
    if hessian.shape != (len(rho), 2, 3, 3):
        raise ValueError("hessian must have shape (N,2,3,3).")
    if not (rho.is_floating_point() and gradient.is_floating_point()):
        raise TypeError("Density and derivative tensors must be floating point.")
    if rho.device != gradient.device or rho.device != hessian.device:
        raise ValueError("Density and derivative tensors must share one device.")
    if rho.dtype != gradient.dtype or rho.dtype != hessian.dtype:
        raise ValueError("Density and derivative tensors must share one dtype.")
    if not all(torch.isfinite(x).all() for x in (rho, gradient, hessian)):
        raise ValueError("Density and derivative tensors must be finite.")

    # The AO field is fixed reference data; autograd differentiates only the
    # local energy expression, then the chain rule contracts those derivatives
    # with the independently evaluated spatial density derivatives.
    g = gradient.detach()
    h = hessian.detach()
    rho_local = rho.detach().clone().requires_grad_(True)
    sigma_local = sigma_from_gradient(g).detach().requires_grad_(True)
    with torch.enable_grad():
        e = energy_density(rho_local, sigma_local)
        if e.shape != (len(rho),):
            raise ValueError("energy_density must return a length-N tensor.")
        if not e.requires_grad:
            raise ValueError("energy_density must depend on rho and/or sigma.")
        c, es = torch.autograd.grad(
            e.sum(), (rho_local, sigma_local), create_graph=True, allow_unused=True
        )
        if c is None:
            c = torch.zeros_like(rho_local)
        if es is None:
            es = torch.zeros_like(sigma_local)

        # Local derivatives of each sigma coefficient with respect to rho and
        # sigma. Outputs for different points are independent by construction.
        des_drho = []
        des_dsigma = []
        for channel in range(3):
            component = es[:, channel]
            if component.requires_grad:
                drho, dsigma = torch.autograd.grad(
                    component.sum(),
                    (rho_local, sigma_local),
                    retain_graph=channel < 2,
                    allow_unused=True,
                )
            else:
                drho = dsigma = None
            des_drho.append(torch.zeros_like(rho_local) if drho is None else drho)
            des_dsigma.append(
                torch.zeros_like(sigma_local) if dsigma is None else dsigma
            )

    # Spatial gradients of standard spin sigmas. Hessian axes are derivative
    # direction followed by gradient component: H[i,j] = d_i d_j rho.
    grad_sigma_aa = 2 * torch.einsum("nij,nj->ni", h[:, 0], g[:, 0])
    grad_sigma_ab = torch.einsum("nij,nj->ni", h[:, 0], g[:, 1]) + torch.einsum(
        "nij,nj->ni", h[:, 1], g[:, 0]
    )
    grad_sigma_bb = 2 * torch.einsum("nij,nj->ni", h[:, 1], g[:, 1])
    grad_sigma = torch.stack([grad_sigma_aa, grad_sigma_ab, grad_sigma_bb], dim=1)

    drho = torch.stack(des_drho, dim=1)  # (N, sigma-channel, rho-spin)
    dsigma = torch.stack(des_dsigma, dim=1)  # (N, sigma-channel, sigma-channel)
    grad_es = torch.einsum("njs,nsk->njk", drho, g) + torch.einsum(
        "njl,nlk->njk", dsigma, grad_sigma
    )

    lap = torch.diagonal(h, dim1=-2, dim2=-1).sum(-1)
    div_alpha = (
        2 * es[:, 0] * lap[:, 0]
        + es[:, 1] * lap[:, 1]
        + 2 * (grad_es[:, 0] * g[:, 0]).sum(-1)
        + (grad_es[:, 1] * g[:, 1]).sum(-1)
    )
    div_beta = (
        2 * es[:, 2] * lap[:, 1]
        + es[:, 1] * lap[:, 0]
        + 2 * (grad_es[:, 2] * g[:, 1]).sum(-1)
        + (grad_es[:, 1] * g[:, 0]).sum(-1)
    )
    minus_div_a = -torch.stack([div_alpha, div_beta], dim=-1)
    vector_a = torch.stack(
        [
            2 * es[:, 0, None] * g[:, 0] + es[:, 1, None] * g[:, 1],
            2 * es[:, 2, None] * g[:, 1] + es[:, 1, None] * g[:, 0],
        ],
        dim=1,
    )
    return {
        "energy_density": e.detach(),
        "rho_derivative": c.detach(),
        "sigma_derivative": es.detach(),
        "sigma": sigma_local.detach(),
        "A": vector_a.detach(),
        "minus_div_A": minus_div_a.detach(),
        "Vxc": (c + minus_div_a).detach(),
    }


def pbe_euler_potential(
    rho: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Spin-resolved canonical PBE potential ``d e/drho - div(A)``."""
    return euler_potential_from_gga(rho, gradient, hessian, pbe_energy_density)


def ao_spin_density_value_gradient(
    mol, spin_density_matrices: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate only spin density and gradient using PySCF ``deriv=1`` AOs.

    This lower-cost path is intended for h-sweep stencil points, where only
    rho and grad(rho) are needed to form local GGA ``A`` vectors. It returns
    ``(density[N,2], gradient[N,2,3])`` in alpha/beta and x/y/z order.
    """
    from pyscf.dft import numint

    dm = np.asarray(spin_density_matrices, dtype=np.float64)
    xyz = np.asarray(coords, dtype=np.float64)
    nao = mol.nao_nr()
    if dm.shape != (2, nao, nao) or not np.isfinite(dm).all():
        raise ValueError(
            "Two finite spin AO density matrices matching mol are required."
        )
    if not np.allclose(dm, dm.transpose(0, 2, 1), rtol=0, atol=1e-12):
        raise ValueError("Spin AO density matrices must be real symmetric.")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Coordinates must be a nonempty (N,3) array in Bohr.")
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates must be finite.")

    ao = np.asarray(numint.eval_ao(mol, xyz, deriv=1), dtype=np.float64)
    if ao.shape != (4, len(xyz), nao):
        raise ValueError("PySCF deriv=1 AO array has an unexpected component layout.")
    value = ao[0]
    first = ao[1:4]
    rho = np.empty((len(xyz), 2), dtype=np.float64)
    gradient = np.empty((len(xyz), 2, 3), dtype=np.float64)
    for spin in range(2):
        d = dm[spin]
        rho[:, spin] = np.einsum("pi,ij,pj->p", value, d, value, optimize=True)
        gradient[:, spin] = 2 * np.einsum(
            "kpi,ij,pj->pk", first, d, value, optimize=True
        )
    return rho, gradient


def ao_spin_density_derivatives(
    mol, spin_density_matrices: np.ndarray, coords: np.ndarray
) -> AOSpinDensityDerivatives:
    """Evaluate spin AO densities, gradients, and Hessians with PySCF.

    The input matrix order is alpha, beta and must match ``mol``'s exact AO
    ordering. PySCF's Cartesian ``eval_ao(..., deriv=2)`` component order is
    value, x/y/z, xx/xy/xz/yy/yz/zz; the same second-derivative indexing is
    used by the repository's MGGA Laplacian path.
    """
    from pyscf.dft import numint

    dm = np.asarray(spin_density_matrices, dtype=np.float64)
    xyz = np.asarray(coords, dtype=np.float64)
    nao = mol.nao_nr()
    if dm.shape != (2, nao, nao) or not np.isfinite(dm).all():
        raise ValueError(
            "Two finite spin AO density matrices matching mol are required."
        )
    if not np.allclose(dm, dm.transpose(0, 2, 1), rtol=0, atol=1e-12):
        raise ValueError("Spin AO density matrices must be real symmetric.")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Coordinates must be a nonempty (N,3) array in Bohr.")
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates must be finite.")

    ao = np.asarray(numint.eval_ao(mol, xyz, deriv=2), dtype=np.float64)
    if ao.shape != (10, len(xyz), nao):
        raise ValueError("PySCF deriv=2 AO array has an unexpected component layout.")
    value = ao[0]
    first = ao[1:4]
    second_index = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    second = np.empty((3, 3, len(xyz), nao), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            second[i, j] = ao[second_index[i][j]]

    rho = np.empty((len(xyz), 2), dtype=np.float64)
    gradient = np.empty((len(xyz), 2, 3), dtype=np.float64)
    hessian = np.empty((len(xyz), 2, 3, 3), dtype=np.float64)
    for spin in range(2):
        d = dm[spin]
        rho[:, spin] = np.einsum("pi,ij,pj->p", value, d, value, optimize=True)
        gradient[:, spin] = 2 * np.einsum(
            "kpi,ij,pj->pk", first, d, value, optimize=True
        )
        hessian[:, spin] = 2 * np.einsum(
            "klpi,ij,pj->pkl", second, d, value, optimize=True
        ) + 2 * np.einsum("kpi,ij,lpj->pkl", first, d, first, optimize=True)
    return AOSpinDensityDerivatives(rho, gradient, hessian)


def pbe_ao_potential(
    mol, spin_density_matrices: np.ndarray, coords: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate the PBE Euler potential directly from reference spin AO matrices."""
    fields = ao_spin_density_derivatives(mol, spin_density_matrices, coords)
    tensors = [
        torch.as_tensor(x, dtype=torch.float64)
        for x in (fields.density, fields.gradient, fields.hessian)
    ]
    result = pbe_euler_potential(*tensors)
    return {key: value.cpu().numpy() for key, value in result.items()}
