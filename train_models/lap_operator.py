"""Torch-native AO XC operators for the tau-free Laplacian functional.

The module consumes central-grid AO values and their first two derivatives.
It does not use spatial stencils or PySCF/NumPy in the differentiable path.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# ``lap_vxc`` is also used as a directly executed training module and imports
# sibling files by their historical top-level names.
_TRAIN_MODELS_DIR = str(Path(__file__).resolve().parent)
if _TRAIN_MODELS_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_MODELS_DIR)

try:  # Support both ``import train_models.lap_operator`` and script imports.
    from .lap_vxc import LapEnergy, local_partials, sigma_from_gradients
except ImportError:  # pragma: no cover - exercised by direct script imports
    from lap_vxc import LapEnergy, local_partials, sigma_from_gradients


OPERATOR_PROTOCOL = "lap-weakform-ao-v1"
FEATURE_COUNT = 10
FEATURE_LAYOUT = (
    "rho_alpha",
    "rho_beta",
    "grad_alpha_x",
    "grad_alpha_y",
    "grad_alpha_z",
    "grad_beta_x",
    "grad_beta_y",
    "grad_beta_z",
    "lapl_alpha",
    "lapl_beta",
)


@dataclass(frozen=True)
class LapOperatorMetadata:
    """Immutable identifying metadata for h-free AO-operator artifacts."""

    protocol: str = OPERATOR_PROTOCOL
    descriptor_protocol: str = "rho-sigma-total-lapl-tau-free-v1"
    feature_layout: tuple[str, ...] = FEATURE_LAYOUT
    matrix_convention: str = "dE_xc/dP_total; symmetric AO matrix"
    spin_convention: str = "P_alpha=P_beta=P_total/2"
    loss_convention: str = "||S^-1/2(V_pred-V_ref)S^-1/2||_F^2/n_ao"

    def to_dict(self) -> dict[str, object]:
        """Return a fresh serialization-ready copy of the frozen metadata."""
        return asdict(self)


def operator_checkpoint_metadata() -> LapOperatorMetadata:
    """Create immutable metadata identifying a weak-form operator artifact."""
    return LapOperatorMetadata()


def validate_operator_metadata(metadata: Mapping[str, object] | LapOperatorMetadata) -> None:
    """Reject stencil/h-based or otherwise incompatible operator metadata."""
    values = metadata.to_dict() if isinstance(metadata, LapOperatorMetadata) else metadata
    if not isinstance(values, Mapping):
        raise TypeError("operator metadata must be a mapping or LapOperatorMetadata.")
    if "h" in values or "stencil" in values:
        raise ValueError("AO-operator metadata cannot contain h or stencil fields.")
    if values.get("protocol") != OPERATOR_PROTOCOL:
        raise ValueError("Incompatible variational AO-operator protocol.")
    if values.get("descriptor_protocol") != "rho-sigma-total-lapl-tau-free-v1":
        raise ValueError("Incompatible Lap density descriptor protocol.")
    if tuple(values.get("feature_layout", ())) != FEATURE_LAYOUT:
        raise ValueError("Incompatible AO-operator density feature layout.")


def _check_grid_inputs(phi, grad_phi, lap_phi, weights):
    if phi.ndim != 2:
        raise ValueError("phi must have shape (n_grid, n_ao).")
    ngrid, nao = phi.shape
    if ngrid == 0 or nao == 0:
        raise ValueError("The AO grid and basis must be nonempty.")
    if grad_phi.shape != (ngrid, 3, nao):
        raise ValueError("grad_phi must have shape (n_grid, 3, n_ao).")
    if lap_phi.shape != (ngrid, nao):
        raise ValueError("lap_phi must have shape (n_grid, n_ao).")
    if weights.shape != (ngrid,):
        raise ValueError("weights must have shape (n_grid,).")
    tensors = (phi, grad_phi, lap_phi, weights)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("AO values, derivatives, and weights must be real floats.")
    if any(tensor.device != phi.device for tensor in tensors):
        raise ValueError("All AO grid tensors must be on the same device.")
    if any(tensor.dtype != phi.dtype for tensor in tensors):
        raise ValueError("All AO grid tensors must have the same dtype.")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("AO grid tensors must be finite.")
    return ngrid, nao


def _check_features(features, ngrid):
    if features.shape != (ngrid, FEATURE_COUNT):
        raise ValueError(
            f"features must have shape (n_grid, {FEATURE_COUNT}) in layout "
            f"{FEATURE_LAYOUT}."
        )
    if not features.is_floating_point():
        raise TypeError("Density features must be real floating point tensors.")


def spin_density_features_from_ao(
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    lap_phi: torch.Tensor,
    dm_alpha: torch.Tensor,
    dm_beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate spin densities and derivatives from AO data and spin DMs.

    AO inputs use ``phi[n_grid,n_ao]``, ``grad_phi[n_grid,3,n_ao]``, and
    ``lap_phi[n_grid,n_ao]``. Density matrices are the usual full AO matrices.
    The returned columns follow :data:`FEATURE_LAYOUT`.
    """
    ngrid, nao = _check_grid_inputs(phi, grad_phi, lap_phi, phi.new_ones(phi.shape[0]))
    del ngrid
    if dm_alpha.shape != (nao, nao) or dm_beta.shape != (nao, nao):
        raise ValueError("Each spin density matrix must have shape (n_ao, n_ao).")
    if dm_alpha.device != phi.device or dm_beta.device != phi.device:
        raise ValueError("Density matrices and AO data must be on the same device.")
    if dm_alpha.dtype != phi.dtype or dm_beta.dtype != phi.dtype:
        raise ValueError("Density matrices and AO data must have the same dtype.")
    if not torch.isfinite(dm_alpha).all() or not torch.isfinite(dm_beta).all():
        raise ValueError("Density matrices must be finite.")

    rho_alpha = torch.einsum("gi,ij,gj->g", phi, dm_alpha, phi)
    rho_beta = torch.einsum("gi,ij,gj->g", phi, dm_beta, phi)
    grad_alpha = torch.einsum("gdi,ij,gj->gd", grad_phi, dm_alpha, phi)
    grad_alpha = grad_alpha + torch.einsum("gi,ij,gdj->gd", phi, dm_alpha, grad_phi)
    grad_beta = torch.einsum("gdi,ij,gj->gd", grad_phi, dm_beta, phi)
    grad_beta = grad_beta + torch.einsum("gi,ij,gdj->gd", phi, dm_beta, grad_phi)

    lapl_alpha = torch.einsum("gi,ij,gj->g", lap_phi, dm_alpha, phi)
    lapl_alpha = lapl_alpha + torch.einsum("gi,ij,gj->g", phi, dm_alpha, lap_phi)
    lapl_alpha = lapl_alpha + 2.0 * torch.einsum(
        "gdi,ij,gdj->g", grad_phi, dm_alpha, grad_phi
    )
    lapl_beta = torch.einsum("gi,ij,gj->g", lap_phi, dm_beta, phi)
    lapl_beta = lapl_beta + torch.einsum("gi,ij,gj->g", phi, dm_beta, lap_phi)
    lapl_beta = lapl_beta + 2.0 * torch.einsum(
        "gdi,ij,gdj->g", grad_phi, dm_beta, grad_phi
    )

    return torch.cat(
        (
            rho_alpha[:, None],
            rho_beta[:, None],
            grad_alpha,
            grad_beta,
            lapl_alpha[:, None],
            lapl_beta[:, None],
        ),
        dim=1,
    )


def rks_density_features_from_ao(
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    lap_phi: torch.Tensor,
    dm_total: torch.Tensor,
) -> torch.Tensor:
    """Evaluate RKS features from PySCF's total closed-shell density matrix."""
    if dm_total.ndim != 2 or dm_total.shape[0] != dm_total.shape[1]:
        raise ValueError("dm_total must have shape (n_ao, n_ao).")
    half_dm = 0.5 * dm_total
    return spin_density_features_from_ao(
        phi, grad_phi, lap_phi, half_dm, half_dm
    )


def integrated_xc_energy(
    energy,
    features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Integrate the LapEnergy local energy density on a fixed grid.

    The descriptors remain attached to their inputs, so this helper can also
    support density-matrix directional-derivative checks.
    """
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape (n_grid, {FEATURE_COUNT}).")
    if weights.shape != (features.shape[0],):
        raise ValueError("weights must have shape (n_grid,).")
    if weights.device != features.device or weights.dtype != features.dtype:
        raise ValueError("features and weights must share device and dtype.")
    rho = features[:, :2]
    grad = features[:, 2:8].reshape(-1, 2, 3)
    sigma = sigma_from_gradients(grad)
    lapl = features[:, 8:10]
    local_energy_density = energy(rho, sigma, lapl)
    if local_energy_density.shape != weights.shape:
        raise ValueError("energy must return one local energy density per grid point.")
    return torch.sum(local_energy_density * weights)


def _assemble_from_local_partials(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    lap_phi: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    ngrid, nao = _check_grid_inputs(phi, grad_phi, lap_phi, weights)
    if c.shape != (ngrid,) or a.shape != (ngrid, 3) or b.shape != (ngrid,):
        raise ValueError("Local partials must have shapes (n_grid,), (n_grid,3), (n_grid,).")
    if any(t.device != phi.device or t.dtype != phi.dtype for t in (c, a, b)):
        raise ValueError("Local partials and AO grid tensors must share device and dtype.")
    if not all(torch.isfinite(t).all() for t in (c, a, b)):
        raise ValueError("Local partials must be finite.")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    matrix = phi.new_zeros((nao, nao))
    for start in range(0, ngrid, chunk_size):
        stop = min(start + chunk_size, ngrid)
        p = phi[start:stop]
        dp = grad_phi[start:stop]
        lp = lap_phi[start:stop]
        w = weights[start:stop]

        wc = (w * c[start:stop])[:, None]
        block = p.T @ (p * wc)

        wa = w[:, None] * a[start:stop]
        for axis in range(3):
            d = dp[:, axis]
            wa_axis = wa[:, axis : axis + 1]
            block = block + d.T @ (p * wa_axis) + p.T @ (d * wa_axis)

        wb = (w * b[start:stop])[:, None]
        block = block + lp.T @ (p * wb) + p.T @ (lp * wb)
        for axis in range(3):
            d = dp[:, axis]
            block = block + 2.0 * d.T @ (d * wb)
        matrix = matrix + block

    return 0.5 * (matrix + matrix.T)


def assemble_spin_operators(
    energy,
    features: torch.Tensor,
    weights: torch.Tensor,
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    lap_phi: torch.Tensor,
    *,
    chunk_size: int = 2048,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``V_alpha,V_beta`` from the same local XC energy functional.

    ``features`` has columns described by :data:`FEATURE_LAYOUT`. The local
    partials are computed with ``create_graph=True`` by default so parameter
    gradients pass through the AO operators.
    """
    ngrid, _ = _check_grid_inputs(phi, grad_phi, lap_phi, weights)
    _check_features(features, ngrid)
    if features.device != phi.device or features.dtype != phi.dtype:
        raise ValueError("Density features and AO grid tensors must share device and dtype.")
    _, c, a, b = local_partials(energy, features, create_graph=create_graph)
    return tuple(
        _assemble_from_local_partials(
            c[:, spin], a[:, spin], b[:, spin], weights, phi, grad_phi, lap_phi, chunk_size
        )
        for spin in (0, 1)
    )


def assemble_rks_operator(
    energy,
    features: torch.Tensor,
    weights: torch.Tensor,
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    lap_phi: torch.Tensor,
    *,
    chunk_size: int = 2048,
    create_graph: bool = True,
) -> torch.Tensor:
    """Assemble the total-density RKS matrix from equal alpha/beta channels."""
    va, vb = assemble_spin_operators(
        energy,
        features,
        weights,
        phi,
        grad_phi,
        lap_phi,
        chunk_size=chunk_size,
        create_graph=create_graph,
    )
    return 0.5 * (va + vb)


def project_scalar_vxc(
    phi: torch.Tensor,
    weights: torch.Tensor,
    vxc: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Project a common scalar RKS ``v_xc(r)`` onto the AO basis.

    This is the mRKS reference convention: ``V[m,n] = sum_g w_g v_g
    phi[g,m] phi[g,n]``. No spin or closed-shell factor is applied.
    """
    if phi.ndim != 2 or weights.shape != (phi.shape[0],) or vxc.shape != weights.shape:
        raise ValueError("Expected phi[n_grid,n_ao] and weights/vxc[n_grid].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if any(t.device != phi.device or t.dtype != phi.dtype for t in (weights, vxc)):
        raise ValueError("phi, weights, and vxc must share device and dtype.")
    if not all(torch.isfinite(t).all() for t in (phi, weights, vxc)):
        raise ValueError("phi, weights, and vxc must be finite.")
    matrix = phi.new_zeros((phi.shape[1], phi.shape[1]))
    for start in range(0, phi.shape[0], chunk_size):
        stop = min(start + chunk_size, phi.shape[0])
        p = phi[start:stop]
        wv = (weights[start:stop] * vxc[start:stop])[:, None]
        matrix = matrix + p.T @ (p * wv)
    return 0.5 * (matrix + matrix.T)


def _inverse_sqrt_overlap(overlap: torch.Tensor) -> torch.Tensor:
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1] or overlap.shape[0] == 0:
        raise ValueError("overlap must be a nonempty square matrix.")
    if not overlap.is_floating_point() or not torch.isfinite(overlap).all():
        raise ValueError("overlap must be a finite real floating-point matrix.")
    if not torch.allclose(overlap, overlap.T, rtol=1e-7, atol=1e-10):
        raise ValueError("overlap must be symmetric.")
    symmetric = 0.5 * (overlap + overlap.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = eigenvalues.abs().max()
    tolerance = torch.finfo(overlap.dtype).eps * max(overlap.shape) * scale
    if eigenvalues[0] <= tolerance:
        raise ValueError("overlap must be numerically positive definite.")
    return (eigenvectors * eigenvalues.rsqrt()[None, :]) @ eigenvectors.T


def orthonormalize_operator(operator: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Return the symmetric-orthogonalized matrix ``S^-1/2 V S^-1/2``."""
    if operator.shape != overlap.shape:
        raise ValueError("operator and overlap must have the same square shape.")
    if operator.device != overlap.device or operator.dtype != overlap.dtype:
        raise ValueError("operator and overlap must share device and dtype.")
    if not torch.isfinite(operator).all():
        raise ValueError("operator must be finite.")
    sinvhalf = _inverse_sqrt_overlap(overlap)
    return sinvhalf @ operator @ sinvhalf


def operator_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    overlap: torch.Tensor,
) -> torch.Tensor:
    """Basis-invariant squared Hilbert-Schmidt error per AO function."""
    if predicted.shape != reference.shape or predicted.shape != overlap.shape:
        raise ValueError("predicted, reference, and overlap must have matching shapes.")
    if predicted.device != reference.device or predicted.dtype != reference.dtype:
        raise ValueError("predicted and reference must share device and dtype.")
    if predicted.device != overlap.device or predicted.dtype != overlap.dtype:
        raise ValueError("operators and overlap must share device and dtype.")
    sinvhalf = _inverse_sqrt_overlap(overlap)
    delta = predicted - reference
    delta_orth = sinvhalf @ delta @ sinvhalf
    return delta_orth.square().sum() / predicted.shape[0]


__all__ = [
    "FEATURE_COUNT",
    "FEATURE_LAYOUT",
    "OPERATOR_PROTOCOL",
    "LapEnergy",
    "LapOperatorMetadata",
    "assemble_rks_operator",
    "assemble_spin_operators",
    "integrated_xc_energy",
    "operator_checkpoint_metadata",
    "operator_loss",
    "orthonormalize_operator",
    "project_scalar_vxc",
    "rks_density_features_from_ao",
    "spin_density_features_from_ao",
    "validate_operator_metadata",
]
