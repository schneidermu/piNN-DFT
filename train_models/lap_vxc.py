"""Local autograd + Cartesian finite differences for full spin XC potentials.

All density quantities are independently evaluated at each stencil coordinate.
No operation interpolates a quadrature grid. Coordinates and h are in Bohr.
"""

import math
import sys
from pathlib import Path

import torch
from torch import nn

# Support the repository's documented `python file.py` execution boundary.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lap_stencil_operators import (
    STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
    divergence,
    get_stencil,
    laplacian,
)
from lap_stencil_operators import (
    stencil_coordinates as cartesian_stencil_coordinates,
)

from dft_functionals import PBE

STENCIL_VERSION = STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1
STENCIL_ORDER = ("center", "+x", "-x", "+y", "-y", "+z", "-z")
STENCIL_ORDER_13 = (
    "center",
    "+x",
    "-x",
    "+2x",
    "-2x",
    "+y",
    "-y",
    "+2y",
    "-2y",
    "+z",
    "-z",
    "+2z",
    "-2z",
)
STENCIL_DERIVATIVE_ORDER = 2
SUPPORTED_STENCIL_VERSIONS = (
    STENCIL_VERSION,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
)
# Features per point: rho_a,rho_b,grad_a_x,y,z,grad_b_x,y,z,lapl_a,lapl_b.
FEATURE_COUNT = 10


def stencil_order_for_version(version):
    """Return the derivative order bound to an explicit persisted version ID."""
    if version == STENCIL_VERSION:
        return STENCIL_DERIVATIVE_ORDER
    if version == STENCIL_VERSION_13_POINT_FOURTH_ORDER:
        return 4
    raise ValueError(f"Unsupported Lap stencil version {version!r}.")


def stencil_positions_for_version(version):
    """Return the persisted offset labels bound to one explicit version ID."""
    stencil_order_for_version(version)
    if version == STENCIL_VERSION:
        return STENCIL_ORDER
    return STENCIL_ORDER_13


def validate_stencil_selection(order, version):
    """Validate one supported persisted version/order pair without shape inference."""
    if version not in SUPPORTED_STENCIL_VERSIONS:
        raise ValueError(f"Unsupported Lap stencil version {version!r}.")
    return get_stencil(order, version)


def stencil_coordinates(
    coords,
    h,
    *,
    order=STENCIL_DERIVATIVE_ORDER,
    version=STENCIL_VERSION,
):
    if not math.isfinite(h) or h <= 0 or coords.dtype != torch.float64:
        raise ValueError(
            "Positive finite h in Bohr and float64 coordinates are required."
        )
    validate_stencil_selection(order, version)
    return cartesian_stencil_coordinates(
        coords, h, order=order, version=version
    )


def sigma_from_gradients(grad):
    """(...,2,3) -> standard LibXC/PySCF (aa,ab,bb)."""
    a, b = grad[..., 0, :], grad[..., 1, :]
    return torch.stack([(a * a).sum(-1), (a * b).sum(-1), (b * b).sum(-1)], -1)


def sigma_standard_to_total(s):
    """(aa,ab,bb) -> model (aa,total,bb). No closed-shell assumption."""
    return torch.stack(
        [s[..., 0], s[..., 0] + 2 * s[..., 1] + s[..., 2], s[..., 2]], -1
    )


def sigma_total_to_standard(s):
    return torch.stack(
        [s[..., 0], (s[..., 1] - s[..., 0] - s[..., 2]) / 2, s[..., 2]], -1
    )


def gradient_chain_rule(esigma, grad):
    a = (
        2 * esigma[..., 0, None] * grad[..., 0, :]
        + esigma[..., 1, None] * grad[..., 1, :]
    )
    b = (
        2 * esigma[..., 2, None] * grad[..., 1, :]
        + esigma[..., 1, None] * grad[..., 0, :]
    )
    return torch.stack([a, b], -2)


class LapEnergy(nn.Module):
    """The same existing modified-PBE expression, without detached constants."""

    def __init__(self, model):
        super().__init__()
        if (
            getattr(model, "descriptor_protocol", None)
            != "rho-sigma-total-lapl-tau-free-v1"
        ):
            raise ValueError(
                "LapEnergy requires an explicitly tau-free Lap architecture."
            )
        self.model = model

    def forward(self, rho, sigma, lapl):
        raw = torch.cat(
            [rho, sigma_standard_to_total(sigma), torch.zeros_like(rho), lapl], -1
        )
        constants = self.model(raw)
        return PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(-1)


def local_partials(energy, features, create_graph=True):
    """Return e,C,A,B; retain every local NN parameter dependency."""
    rho = features[..., :2].detach().requires_grad_(True)
    grad = features[..., 2:8].reshape(-1, 2, 3)
    sigma = sigma_from_gradients(grad).detach().requires_grad_(True)
    lapl = features[..., 8:10].detach().requires_grad_(True)
    with torch.enable_grad():
        e = energy(rho, sigma, lapl)
        if not e.requires_grad:
            partials = (None, None, None)
        else:
            partials = torch.autograd.grad(
                e.sum(),
                (rho, sigma, lapl),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
    c, es, b = [
        torch.zeros_like(x) if y is None else y
        for x, y in zip((rho, sigma, lapl), partials)
    ]
    return e, c, gradient_chain_rule(es, grad), b


def euler_components(
    energy,
    features,
    h,
    create_graph=True,
    *,
    order=STENCIL_DERIVATIVE_ORDER,
    version=STENCIL_VERSION,
):
    stencil = validate_stencil_selection(order, version)
    if features.ndim != 3 or features.shape[1:] != (
        stencil.point_count,
        FEATURE_COUNT,
    ):
        raise ValueError(
            f"Full Vxc requires independently evaluated (N,{stencil.point_count},10) "
            f"features for {version}."
        )
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Finite-difference h must be positive, finite, in Bohr.")
    e, c, a, b = local_partials(energy, features.reshape(-1, 10), create_graph)
    # Assembly in float64 cannot undo float32 errors in local derivatives.
    # Use a double model for convergence diagnostics; do not silently cast it.
    n = len(features)
    c, a, b = (
        c.reshape(n, stencil.point_count, 2).double(),
        a.reshape(n, stencil.point_count, 2, 3).double(),
        b.reshape(n, stencil.point_count, 2).double(),
    )
    # Operator inputs keep the offset axis immediately before the Cartesian
    # component axis. Move spin ahead of offsets for these stencil kernels.
    vector_by_spin = a.movedim(2, 1)
    scalar_by_spin = b.movedim(2, 1)
    # Keep the named terms available for diagnostics while sharing the exact
    # selected stencil weights with the main C - div(A) + lap(B) expression.
    div = divergence(vector_by_spin, h, order=order, version=version)
    lap = laplacian(scalar_by_spin, h, order=order, version=version)
    return {
        "energy": e.reshape(n, stencil.point_count)[:, 0],
        "C": c[:, 0],
        "minus_div_A": -div,
        "lap_B": lap,
        "Vxc": c[:, 0] - div + lap,
    }


def potential_loss(prediction, target, rho, weights):
    """Absolute POINTWISE density-weighted MSE; never project a constant."""
    q = rho.double() * weights.double()[:, None]
    norm = q.sum()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Nonpositive/nonfinite electron normalization.")
    return (q * (prediction.double() - target.double()).square()).sum() / norm


class _ChunkedLoss(torch.autograd.Function):
    """Recompute and release each stencil graph during backward.

    Parameters are explicit Function inputs, so ordinary autograd/DDP/manual
    objective merging sees the gradient normally. No optimizer step occurs here.
    Input reference features are fixed data, never trainable tensors.
    """

    @staticmethod
    def forward(
        ctx,
        energy,
        features,
        target,
        weights,
        h,
        chunk_size,
        order,
        version,
        *parameters,
    ):
        ctx.energy, ctx.h, ctx.chunk_size = energy, h, chunk_size
        ctx.order, ctx.version = order, version
        ctx.save_for_backward(features, target, weights, *parameters)
        loss = features.new_zeros((), dtype=torch.float64)
        norm = (features[:, 0, :2].double() * weights.double()[:, None]).sum()
        if norm <= 0 or not torch.isfinite(norm):
            raise ValueError("Invalid density normalization.")
        ctx.norm = norm
        for start in range(0, len(features), chunk_size):
            sl = slice(start, start + chunk_size)
            with torch.enable_grad():
                pred = euler_components(
                    energy, features[sl], h, False, order=order, version=version
                )["Vxc"]
            q = features[sl, 0, :2].double() * weights[sl].double()[:, None]
            loss += (q * (pred.detach() - target[sl]).square()).sum() / norm
        return loss

    @staticmethod
    def backward(ctx, upstream):
        features, target, weights, *parameters = ctx.saved_tensors
        accumulated = [torch.zeros_like(p) for p in parameters]
        for start in range(0, len(features), ctx.chunk_size):
            sl = slice(start, start + ctx.chunk_size)
            with torch.enable_grad():
                pred = euler_components(
                    ctx.energy,
                    features[sl],
                    ctx.h,
                    True,
                    order=ctx.order,
                    version=ctx.version,
                )["Vxc"]
                q = features[sl, 0, :2].double() * weights[sl].double()[:, None]
                loss = (q * (pred - target[sl]).square()).sum() / ctx.norm
                if loss.requires_grad:
                    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
                    for total, g in zip(accumulated, grads):
                        if g is not None:
                            total.add_(g)
        return (None,) * 8 + tuple(g * upstream for g in accumulated)


def full_vxc_loss(
    energy,
    features,
    target,
    weights,
    h,
    point_chunk_size=4096,
    *,
    order=STENCIL_DERIVATIVE_ORDER,
    version=STENCIL_VERSION,
):
    if point_chunk_size <= 0:
        raise ValueError("Point chunk size must be positive.")
    validate_stencil_selection(order, version)
    if target.shape != (len(features), 2) or weights.shape != (len(features),):
        raise ValueError("Vxc must preserve (N,2) spin channels, with (N,) weights.")
    if any(isinstance(m, nn.Dropout) and m.p != 0 for m in energy.modules()):
        raise ValueError(
            "Chunk recomputation requires a deterministic energy (dropout=0)."
        )
    params = tuple(p for p in energy.parameters() if p.requires_grad)
    return _ChunkedLoss.apply(
        energy,
        features,
        target,
        weights,
        h,
        point_chunk_size,
        order,
        version,
        *params,
    )


def integrated_energy(energy, features, weights, point_chunk_size=4096):
    """Center energy from the same functional; checkpoint each center chunk."""
    from torch.utils.checkpoint import checkpoint

    if point_chunk_size <= 0:
        raise ValueError("Point chunk size must be positive.")
    result = features.new_zeros((), dtype=torch.float64)
    for start in range(0, len(features), point_chunk_size):
        f = features[start : start + point_chunk_size, 0]
        rho, grad, lap = f[:, :2], f[:, 2:8].reshape(-1, 2, 3), f[:, 8:]
        e = checkpoint(
            energy, rho, sigma_from_gradients(grad), lap, use_reentrant=False
        )
        result = (
            result
            + (e.double() * weights[start : start + point_chunk_size].double()).sum()
        )
    return result
