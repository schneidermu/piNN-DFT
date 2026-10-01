"""Versioned Cartesian finite-difference operators for sampled point fields.

The operators act on values evaluated at Cartesian offsets from each center;
they do not assume a regular tensor grid or perform boundary extrapolation.
Callers must select both the derivative order and its immutable version ID.

``cartesian-7-point-second-order-v1`` uses offsets
``(0,0,0), (+/-1,0,0), (0,+/-1,0), (0,0,+/-1)`` in units of ``h``. Its
first derivative is ``(f(+h)-f(-h))/(2h)`` on each axis and its Laplacian is
``sum_axis (f(+h)-2f(0)+f(-h))/h**2``.

``cartesian-13-point-fourth-order-v1`` uses the center followed, for each
axis, by ``+h, -h, +2h, -2h``. Its first derivative on each axis is
``(-f(+2h)+8f(+h)-8f(-h)+f(-2h))/(12h)``. Its Laplacian is the sum over
axes of ``(-f(+2h)+16f(+h)-30f(0)+16f(-h)-f(-2h))/(12h**2)``.

The existing persisted 7-point corpus ID ``cartesian-7-rho-grad-lapl-v1`` is
also mapped explicitly to the same 7-point offsets and weights so that its
metadata remains unchanged. It is never accepted for order 4.

Scalar samples have shape ``(..., n_offsets)``. Vector samples used for
divergence have shape ``(..., n_offsets, 3)``, with Cartesian components in
the last dimension. Leading dimensions are preserved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

STENCIL_VERSION_7_POINT_SECOND_ORDER = "cartesian-7-point-second-order-v1"
STENCIL_VERSION_13_POINT_FOURTH_ORDER = "cartesian-13-point-fourth-order-v1"
STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1 = "cartesian-7-rho-grad-lapl-v1"


@dataclass(frozen=True, slots=True)
class CartesianStencil:
    """Immutable offset and coefficient specification for one stencil version."""

    order: int
    version: str
    offsets: tuple[tuple[int, int, int], ...]
    first_derivative_weights: tuple[tuple[float, ...], ...]
    laplacian_weights: tuple[float, ...]

    @property
    def point_count(self) -> int:
        return len(self.offsets)


_OFFSETS_7 = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
_FIRST_7 = (
    (0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5),
)
_LAPLACIAN_7 = (  # Numerator weights; divide the dot product by h**2.
    -6.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)

_OFFSETS_13 = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (2, 0, 0),
    (-2, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 2, 0),
    (0, -2, 0),
    (0, 0, 1),
    (0, 0, -1),
    (0, 0, 2),
    (0, 0, -2),
)
_FIRST_13 = (
    (
        0.0,
        2.0 / 3.0,
        -2.0 / 3.0,
        -1.0 / 12.0,
        1.0 / 12.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / 3.0,
        -2.0 / 3.0,
        -1.0 / 12.0,
        1.0 / 12.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / 3.0,
        -2.0 / 3.0,
        -1.0 / 12.0,
        1.0 / 12.0,
    ),
)
_LAPLACIAN_13 = (
    -7.5,
    4.0 / 3.0,
    4.0 / 3.0,
    -1.0 / 12.0,
    -1.0 / 12.0,
    4.0 / 3.0,
    4.0 / 3.0,
    -1.0 / 12.0,
    -1.0 / 12.0,
    4.0 / 3.0,
    4.0 / 3.0,
    -1.0 / 12.0,
    -1.0 / 12.0,
)

_STENCILS: Mapping[tuple[int, str], CartesianStencil] = MappingProxyType(
    {
        (2, STENCIL_VERSION_7_POINT_SECOND_ORDER): CartesianStencil(
            order=2,
            version=STENCIL_VERSION_7_POINT_SECOND_ORDER,
            offsets=_OFFSETS_7,
            first_derivative_weights=_FIRST_7,
            laplacian_weights=_LAPLACIAN_7,
        ),
        (2, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1): CartesianStencil(
            order=2,
            version=STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
            offsets=_OFFSETS_7,
            first_derivative_weights=_FIRST_7,
            laplacian_weights=_LAPLACIAN_7,
        ),
        (4, STENCIL_VERSION_13_POINT_FOURTH_ORDER): CartesianStencil(
            order=4,
            version=STENCIL_VERSION_13_POINT_FOURTH_ORDER,
            offsets=_OFFSETS_13,
            first_derivative_weights=_FIRST_13,
            laplacian_weights=_LAPLACIAN_13,
        ),
    }
)


def get_stencil(order: int, version: str) -> CartesianStencil:
    """Return the exact requested stencil, rejecting every unknown pairing."""
    if type(order) is not int or not isinstance(version, str):
        raise ValueError("Stencil order must be an int and version must be a string.")
    try:
        return _STENCILS[(order, version)]
    except KeyError as exc:
        if version in {
            STENCIL_VERSION_7_POINT_SECOND_ORDER,
            STENCIL_VERSION_13_POINT_FOURTH_ORDER,
            STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
        }:
            raise ValueError(
                f"Stencil order {order} does not match version {version!r}."
            ) from exc
        raise ValueError(f"Unsupported Cartesian stencil version {version!r}.") from exc


def _positive_finite_h(h: float) -> float:
    if isinstance(h, bool) or not isinstance(h, (int, float)):
        raise TypeError("Grid spacing h must be a numeric value in Bohr.")
    h = float(h)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Grid spacing h must be a positive finite number in Bohr.")
    return h


def _require_float_tensor(values: torch.Tensor, name: str) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError(f"{name} must be a floating-point torch.Tensor.")


def _require_scalar_samples(samples: torch.Tensor, stencil: CartesianStencil) -> None:
    _require_float_tensor(samples, "Scalar samples")
    if samples.ndim < 1 or samples.shape[-1] != stencil.point_count:
        raise ValueError(
            f"{stencil.version} requires scalar samples with last dimension "
            f"{stencil.point_count}; got {tuple(samples.shape)}."
        )


def stencil_coordinates(
    coords: torch.Tensor,
    h: float,
    *,
    order: int,
    version: str,
) -> torch.Tensor:
    """Expand centers ``(...,3)`` to exact requested Cartesian stencil points."""
    stencil = get_stencil(order, version)
    h = _positive_finite_h(h)
    _require_float_tensor(coords, "Coordinates")
    if coords.ndim < 1 or coords.shape[-1] != 3:
        raise ValueError(f"Coordinates must have shape (..., 3); got {coords.shape}.")
    offsets = coords.new_tensor(stencil.offsets) * h
    return coords.unsqueeze(-2) + offsets


def first_derivative(
    samples: torch.Tensor,
    h: float,
    *,
    order: int,
    version: str,
) -> torch.Tensor:
    """Return ``(df/dx, df/dy, df/dz)`` from scalar offset samples."""
    stencil = get_stencil(order, version)
    h = _positive_finite_h(h)
    _require_scalar_samples(samples, stencil)
    weights = samples.new_tensor(stencil.first_derivative_weights)
    return torch.einsum("...s,as->...a", samples, weights) / h


def laplacian(
    samples: torch.Tensor,
    h: float,
    *,
    order: int,
    version: str,
) -> torch.Tensor:
    """Return the Cartesian Laplacian of scalar offset samples."""
    stencil = get_stencil(order, version)
    h = _positive_finite_h(h)
    _require_scalar_samples(samples, stencil)
    weights = samples.new_tensor(stencil.laplacian_weights)
    return torch.einsum("...s,s->...", samples, weights) / (h * h)


def divergence(
    vector_samples: torch.Tensor,
    h: float,
    *,
    order: int,
    version: str,
) -> torch.Tensor:
    """Return ``dFx/dx + dFy/dy + dFz/dz`` from vector offset samples."""
    stencil = get_stencil(order, version)
    h = _positive_finite_h(h)
    _require_float_tensor(vector_samples, "Vector samples")
    expected = (stencil.point_count, 3)
    if vector_samples.ndim < 2 or tuple(vector_samples.shape[-2:]) != expected:
        raise ValueError(
            f"{stencil.version} requires vector samples with last dimensions "
            f"{expected}; got {tuple(vector_samples.shape[-2:])}."
        )
    weights = vector_samples.new_tensor(stencil.first_derivative_weights)
    result = vector_samples.new_zeros(vector_samples.shape[:-2])
    for axis in range(3):
        result = result + torch.einsum(
            "...s,s->...", vector_samples[..., :, axis], weights[axis]
        )
    return result / h


def minus_divergence_plus_laplacian(
    vector_samples: torch.Tensor,
    scalar_samples: torch.Tensor,
    h: float,
    *,
    order: int,
    version: str,
) -> torch.Tensor:
    """Compose ``-div(A) + lap(B)`` for sampled vector ``A`` and scalar ``B``."""
    return -divergence(vector_samples, h, order=order, version=version) + laplacian(
        scalar_samples, h, order=order, version=version
    )


__all__ = [
    "STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1",
    "STENCIL_VERSION_7_POINT_SECOND_ORDER",
    "STENCIL_VERSION_13_POINT_FOURTH_ORDER",
    "CartesianStencil",
    "divergence",
    "first_derivative",
    "get_stencil",
    "laplacian",
    "minus_divergence_plus_laplacian",
    "stencil_coordinates",
]
