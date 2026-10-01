"""Manufactured-function checks for the versioned Cartesian stencils."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lap_stencil_operators import (
    STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
    STENCIL_VERSION_7_POINT_SECOND_ORDER,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
    divergence,
    first_derivative,
    get_stencil,
    laplacian,
    minus_divergence_plus_laplacian,
    stencil_coordinates,
)

STENCILS = (
    (2, STENCIL_VERSION_7_POINT_SECOND_ORDER, 7, 4.0),
    (2, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1, 7, 4.0),
    (4, STENCIL_VERSION_13_POINT_FOURTH_ORDER, 13, 16.0),
)


def _centers() -> torch.Tensor:
    axes = [torch.linspace(-0.6, 0.6, 5, dtype=torch.float64) for _ in range(3)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def _f_and_derivatives(xyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x, y, z = xyz.unbind(-1)
    f = torch.sin(0.7 * x) * torch.cos(1.1 * y) * torch.exp(0.3 * z)
    dfdx = 0.7 * torch.cos(0.7 * x) * torch.cos(1.1 * y) * torch.exp(0.3 * z)
    dfdy = -1.1 * torch.sin(0.7 * x) * torch.sin(1.1 * y) * torch.exp(0.3 * z)
    dfdz = 0.3 * f
    lap_f = (-(0.7**2) - 1.1**2 + 0.3**2) * f
    return f, torch.stack((dfdx, dfdy, dfdz), dim=-1), lap_f


def _a_and_b(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = xyz.unbind(-1)
    ax = torch.sin(0.8 * x) * torch.cos(0.4 * y) * torch.exp(0.2 * z)
    ay = torch.cos(0.5 * x) * torch.sin(0.9 * y) * torch.exp(0.3 * z)
    az = torch.exp(-0.6 * x) * torch.cos(0.7 * y) * torch.sin(0.5 * z)
    div_a = (
        0.8 * torch.cos(0.8 * x) * torch.cos(0.4 * y) * torch.exp(0.2 * z)
        + 0.9 * torch.cos(0.5 * x) * torch.cos(0.9 * y) * torch.exp(0.3 * z)
        + 0.5 * torch.exp(-0.6 * x) * torch.cos(0.7 * y) * torch.cos(0.5 * z)
    )
    b = _b_field(xyz)
    lap_b = (-(0.6**2) - 0.8**2 + 0.4**2) * b
    return torch.stack((ax, ay, az), dim=-1), -div_a + lap_b


def _b_field(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = xyz.unbind(-1)
    return torch.cos(0.6 * x) * torch.sin(0.8 * y) * torch.exp(0.4 * z)


def _sample_scalar(
    centers: torch.Tensor,
    h: float,
    order: int,
    version: str,
    function,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = stencil_coordinates(centers, h, order=order, version=version)
    return function(coords), coords


@pytest.mark.parametrize("order,version,point_count,_expected_ratio", STENCILS)
def test_offsets_are_exact_and_ordered(order, version, point_count, _expected_ratio):
    stencil = get_stencil(order, version)
    assert stencil.point_count == point_count
    assert stencil.offsets[0] == (0, 0, 0)
    centers = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    coords = stencil_coordinates(centers, 0.25, order=order, version=version)
    assert torch.equal(coords[0], centers[0] + 0.25 * torch.tensor(stencil.offsets))


@pytest.mark.parametrize("order,version,_point_count,expected_ratio", STENCILS)
def test_first_derivative_converges_at_declared_order(
    order, version, _point_count, expected_ratio
):
    centers = _centers()
    exact = _f_and_derivatives(centers)[1]
    errors = []
    for h in (0.24, 0.12):
        values, _ = _sample_scalar(
            centers, h, order, version, lambda xyz: _f_and_derivatives(xyz)[0]
        )
        estimate = first_derivative(values, h, order=order, version=version)
        errors.append(torch.sqrt(torch.mean((estimate - exact).square())).item())
    assert 0.8 * expected_ratio < errors[0] / errors[1] < expected_ratio * 1.5


@pytest.mark.parametrize("order,version,_point_count,expected_ratio", STENCILS)
def test_laplacian_converges_at_declared_order(
    order, version, _point_count, expected_ratio
):
    centers = _centers()
    exact = _f_and_derivatives(centers)[2]
    errors = []
    for h in (0.24, 0.12):
        values, _ = _sample_scalar(
            centers, h, order, version, lambda xyz: _f_and_derivatives(xyz)[0]
        )
        estimate = laplacian(values, h, order=order, version=version)
        errors.append(torch.sqrt(torch.mean((estimate - exact).square())).item())
    assert 0.8 * expected_ratio < errors[0] / errors[1] < expected_ratio * 1.5


@pytest.mark.parametrize("order,version,_point_count,expected_ratio", STENCILS)
def test_composed_minus_divergence_plus_lapb_converges(
    order, version, _point_count, expected_ratio
):
    centers = _centers()
    exact = _a_and_b(centers)[1]
    errors = []
    for h in (0.24, 0.12):
        coords = stencil_coordinates(centers, h, order=order, version=version)
        a, _ = _a_and_b(coords)
        b = _b_field(coords)
        estimate = minus_divergence_plus_laplacian(
            a, b, h, order=order, version=version
        )
        errors.append(torch.sqrt(torch.mean((estimate - exact).square())).item())
    assert 0.8 * expected_ratio < errors[0] / errors[1] < expected_ratio * 1.5


def test_rejects_order_version_and_sample_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        get_stencil(2, STENCIL_VERSION_13_POINT_FOURTH_ORDER)
    legacy = get_stencil(2, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1)
    assert legacy.version == STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1
    assert legacy.point_count == 7
    with pytest.raises(ValueError, match="does not match"):
        get_stencil(4, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1)
    with pytest.raises(ValueError, match="requires scalar samples"):
        first_derivative(
            torch.zeros((2, 7), dtype=torch.float64),
            0.1,
            order=4,
            version=STENCIL_VERSION_13_POINT_FOURTH_ORDER,
        )
    with pytest.raises(ValueError, match="requires vector samples"):
        divergence(
            torch.zeros((2, 7, 2), dtype=torch.float64),
            0.1,
            order=2,
            version=STENCIL_VERSION_7_POINT_SECOND_ORDER,
        )
