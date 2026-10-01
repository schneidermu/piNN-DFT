"""Focused validation for the independent AO/PBE Euler-potential reference."""

import numpy as np
import pytest
import torch
from lap_stencil_operators import (
    STENCIL_VERSION_7_POINT_SECOND_ORDER,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
    divergence,
    laplacian,
    stencil_coordinates,
)
from lap_stencil_reference import (
    ao_spin_density_derivatives,
    ao_spin_density_value_gradient,
    euler_potential_from_gga,
    pbe_ao_potential,
    pbe_euler_potential,
    sigma_from_gradient,
)


def smooth_fields(coords):
    """Closed-form positive two-spin density field and Cartesian derivatives."""
    x, y, z = coords.unbind(-1)
    rho_a = 1.5 + 0.13 * torch.sin(x) + 0.08 * torch.cos(2 * y) + 0.10 * torch.sin(z)
    rho_b = 1.1 + 0.09 * torch.cos(x) + 0.07 * torch.sin(2 * y) + 0.06 * torch.cos(z)
    grad_a = torch.stack(
        [0.13 * torch.cos(x), -0.16 * torch.sin(2 * y), 0.10 * torch.cos(z)], -1
    )
    grad_b = torch.stack(
        [-0.09 * torch.sin(x), 0.14 * torch.cos(2 * y), -0.06 * torch.sin(z)], -1
    )
    zero = torch.zeros_like(x)
    hess_a = torch.stack(
        [
            torch.stack([-0.13 * torch.sin(x), zero, zero], -1),
            torch.stack([zero, -0.32 * torch.cos(2 * y), zero], -1),
            torch.stack([zero, zero, -0.10 * torch.sin(z)], -1),
        ],
        dim=-2,
    )
    hess_b = torch.stack(
        [
            torch.stack([-0.09 * torch.cos(x), zero, zero], -1),
            torch.stack([zero, -0.28 * torch.sin(2 * y), zero], -1),
            torch.stack([zero, zero, -0.06 * torch.cos(z)], -1),
        ],
        dim=-2,
    )
    return (
        torch.stack([rho_a, rho_b], -1),
        torch.stack([grad_a, grad_b], -2),
        torch.stack([hess_a, hess_b], -3),
    )


def smooth_bilaplacian(coords):
    x, y, z = coords.unbind(-1)
    lap4_a = 0.13 * torch.sin(x) + 1.28 * torch.cos(2 * y) + 0.10 * torch.sin(z)
    lap4_b = 0.09 * torch.cos(x) + 1.12 * torch.sin(2 * y) + 0.06 * torch.cos(z)
    return torch.stack([lap4_a, lap4_b], -1)


def test_manufactured_smooth_gga_matches_closed_form_spin_potential():
    coords = torch.tensor(
        [[0.2, 0.3, 0.7], [1.0, 0.8, 1.3], [1.5, 0.4, 0.9]],
        dtype=torch.float64,
    )
    rho, gradient, hessian = smooth_fields(coords)
    sigma_coefficients = torch.tensor([0.21, -0.13, 0.34], dtype=torch.float64)

    def manufactured_energy(density, sigma):
        return (
            0.7 * density[:, 0].square()
            + 0.4 * density[:, 1].square()
            + (sigma * sigma_coefficients).sum(-1)
        )

    result = euler_potential_from_gga(rho, gradient, hessian, manufactured_energy)
    lap = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
    c = torch.stack([1.4 * rho[:, 0], 0.8 * rho[:, 1]], -1)
    minus_div = -torch.stack(
        [
            2 * sigma_coefficients[0] * lap[:, 0] + sigma_coefficients[1] * lap[:, 1],
            2 * sigma_coefficients[2] * lap[:, 1] + sigma_coefficients[1] * lap[:, 0],
        ],
        -1,
    )
    torch.testing.assert_close(result["rho_derivative"], c, atol=1e-13, rtol=0)
    torch.testing.assert_close(result["minus_div_A"], minus_div, atol=1e-13, rtol=0)
    torch.testing.assert_close(result["Vxc"], c + minus_div, atol=1e-13, rtol=0)
    torch.testing.assert_close(
        result["sigma"], sigma_from_gradient(gradient), atol=1e-14, rtol=0
    )


def test_manufactured_laplacian_energy_components_and_composed_orders():
    axes = [torch.linspace(-0.6, 0.6, 5, dtype=torch.float64) for _ in range(3)]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    rho, gradient, hessian = smooth_fields(centers)
    lap = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
    bilap = smooth_bilaplacian(centers)
    a = torch.tensor([0.7, 0.9], dtype=torch.float64)
    b = torch.tensor([0.37, 0.29], dtype=torch.float64)
    c = torch.tensor([0.11, 0.14], dtype=torch.float64)

    # Manufactured energy density e = sum_s(a_s rho_s^2 +
    # b_s |grad rho_s|^2 + c_s (lap rho_s)^2). Obtain its local C, A, B
    # independently with autograd, then check each closed-form Euler term.
    rho_local = rho.detach().requires_grad_(True)
    grad_local = gradient.detach().requires_grad_(True)
    lap_local = lap.detach().requires_grad_(True)
    energy = (
        (a * rho_local.square()).sum(-1)
        + (b * grad_local.square().sum(-1)).sum(-1)
        + (c * lap_local.square()).sum(-1)
    )
    c_local, a_local, b_local = torch.autograd.grad(
        energy.sum(), (rho_local, grad_local, lap_local)
    )
    c_exact = 2 * a * rho
    a_exact = 2 * b[None, :, None] * gradient
    b_exact = 2 * c * lap
    minus_div_exact = -2 * b * lap
    lap_b_exact = 2 * c * bilap
    vxc_exact = c_exact + minus_div_exact + lap_b_exact
    torch.testing.assert_close(c_local, c_exact, atol=1e-13, rtol=0)
    torch.testing.assert_close(a_local, a_exact, atol=1e-13, rtol=0)
    torch.testing.assert_close(b_local, b_exact, atol=1e-13, rtol=0)

    # Independently differentiate the manufactured A and B fields in space
    # and verify their exact divergence/Laplacian formulas channel by channel.
    xyz = centers.detach().clone().requires_grad_(True)
    _, grad_field, hess_field = smooth_fields(xyz)
    lap_field = torch.diagonal(hess_field, dim1=-2, dim2=-1).sum(-1)
    a_field = 2 * b[None, :, None] * grad_field
    b_field = 2 * c * lap_field
    div_a_spins = []
    lap_b_spins = []
    for spin in range(2):
        div_components = []
        for axis in range(3):
            grad_a_component = torch.autograd.grad(
                a_field[:, spin, axis].sum(), xyz, retain_graph=True
            )[0]
            div_components.append(grad_a_component[:, axis])
        div_a_spins.append(torch.stack(div_components, -1).sum(-1))

        grad_b = torch.autograd.grad(
            b_field[:, spin].sum(), xyz, create_graph=True, retain_graph=True
        )[0]
        lap_b_components = []
        for axis in range(3):
            hess_b_component = torch.autograd.grad(
                grad_b[:, axis].sum(), xyz, retain_graph=True
            )[0]
            lap_b_components.append(hess_b_component[:, axis])
        lap_b_spins.append(torch.stack(lap_b_components, -1).sum(-1))
    minus_div_autograd = -torch.stack(div_a_spins, -1)
    lap_b_autograd = torch.stack(lap_b_spins, -1)
    torch.testing.assert_close(minus_div_autograd, minus_div_exact, atol=1e-13, rtol=0)
    torch.testing.assert_close(lap_b_autograd, lap_b_exact, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        c_local + minus_div_autograd + lap_b_autograd,
        vxc_exact,
        atol=1e-12,
        rtol=0,
    )

    centers_by_order = (
        (2, STENCIL_VERSION_7_POINT_SECOND_ORDER),
        (4, STENCIL_VERSION_13_POINT_FOURTH_ORDER),
    )
    observed_orders = {}
    for order, version in centers_by_order:
        total_errors = []
        for step in (0.24, 0.12, 0.06):
            sample_coords = stencil_coordinates(
                centers, step, order=order, version=version
            )
            _, grad_s, hess_s = smooth_fields(sample_coords)
            lap_s = torch.diagonal(hess_s, dim1=-2, dim2=-1).sum(-1)
            # Move spin before the stencil axis for the standalone operators.
            a_s = (2 * b[None, None, :, None] * grad_s).transpose(1, 2)
            b_s = (2 * c[None, None, :] * lap_s).transpose(1, 2)
            minus_div = -divergence(a_s, step, order=order, version=version)
            lap_b = laplacian(b_s, step, order=order, version=version)
            estimated = c_exact + minus_div + lap_b
            total_errors.append(
                torch.sqrt(torch.mean((estimated - vxc_exact).square())).item()
            )
        observed_orders[order] = [
            np.log2(total_errors[i] / total_errors[i + 1]) for i in range(2)
        ]

    assert all(1.9 < value < 2.1 for value in observed_orders[2])
    assert all(3.8 < value < 4.2 for value in observed_orders[4])


def test_pbe_spatial_divergence_agrees_with_finite_difference_of_A():
    center = torch.tensor([[0.41, 0.62, 0.77]], dtype=torch.float64)
    rho, gradient, hessian = smooth_fields(center)
    analytic = pbe_euler_potential(rho, gradient, hessian)

    def evaluate_a(points):
        density, grad, hess = smooth_fields(points)
        return pbe_euler_potential(density, grad, hess)["A"]

    errors = []
    for step in (2e-3, 1e-3):
        divergence = torch.zeros((1, 2), dtype=torch.float64)
        for axis in range(3):
            offset = torch.zeros_like(center)
            offset[:, axis] = step
            derivative = (
                evaluate_a(center + offset)[:, :, axis]
                - evaluate_a(center - offset)[:, :, axis]
            ) / (2 * step)
            divergence += derivative
        error = (analytic["minus_div_A"] + divergence).abs().max()
        errors.append(float(error))
    assert errors[1] < errors[0] / 3.7
    assert errors[1] < 1e-7


def test_local_pbe_rho_and_sigma_partials_match_pyscf_libxc():
    pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    rho = torch.tensor([[0.8, 0.5], [1.3, 0.9], [0.24, 0.41]], dtype=torch.float64)
    gradient = torch.tensor(
        [
            [[0.13, -0.07, 0.03], [0.05, 0.02, -0.04]],
            [[-0.2, 0.1, 0.08], [0.04, -0.13, 0.09]],
            [[0.04, 0.02, -0.01], [-0.02, 0.03, 0.06]],
        ],
        dtype=torch.float64,
    )
    hessian = torch.zeros((len(rho), 2, 3, 3), dtype=torch.float64)
    result = pbe_euler_potential(rho, gradient, hessian)
    libxc_density = [
        np.vstack([rho[:, spin].numpy(), gradient[:, spin].numpy().T])
        for spin in range(2)
    ]
    exc, vxc, _, _ = libxc.eval_xc("PBE", libxc_density, spin=1, deriv=1)
    vrho, vsigma = vxc

    # The repository's canonical PBE constants are rounded (not LibXC's full
    # precision), so energy and C agree to about 4e-7 here. The local
    # sigma partials agree to the precision shown by LibXC's double outputs.
    np.testing.assert_allclose(
        result["energy_density"].numpy() / rho.sum(-1).numpy(),
        exc,
        rtol=0,
        atol=4e-7,
    )
    np.testing.assert_allclose(
        result["rho_derivative"].numpy(), vrho, rtol=0, atol=5e-7
    )
    np.testing.assert_allclose(
        result["sigma_derivative"].numpy(), vsigma, rtol=0, atol=5e-8
    )


def test_ao_hessian_trace_matches_pyscf_mgga_laplacian_and_pbe_is_finite():
    pytest.importorskip("pyscf")
    from pyscf import gto
    from pyscf.dft import numint

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    dm = np.zeros((2, mol.nao_nr(), mol.nao_nr()), dtype=np.float64)
    dm[0] = np.diag(np.linspace(0.45, 0.75, mol.nao_nr()))
    dm[1] = np.diag(np.linspace(0.30, 0.55, mol.nao_nr()))
    coords = np.asarray([[0.3, 0.2, 0.4], [-0.5, 0.1, 0.8], [0.1, 0.6, 1.1]])
    fields = ao_spin_density_derivatives(mol, dm, coords)
    rho, gradient = ao_spin_density_value_gradient(mol, dm, coords)
    np.testing.assert_allclose(rho, fields.density, rtol=0, atol=2e-13)
    np.testing.assert_allclose(gradient, fields.gradient, rtol=0, atol=2e-13)
    # Check every Hessian column against an independent spatial derivative of
    # the AO-derived gradient, including mixed Cartesian components. The trace
    # comparison below alone would not catch a swapped off-diagonal index.
    step = 2e-5
    for axis in range(3):
        offset = np.zeros_like(coords)
        offset[:, axis] = step
        _, gradient_plus = ao_spin_density_value_gradient(mol, dm, coords + offset)
        _, gradient_minus = ao_spin_density_value_gradient(mol, dm, coords - offset)
        fd_column = (gradient_plus - gradient_minus) / (2 * step)
        np.testing.assert_allclose(
            fields.hessian[:, :, :, axis], fd_column, rtol=2e-8, atol=2e-9
        )
    ao = numint.eval_ao(mol, coords, deriv=2)
    for spin in range(2):
        reference = numint.eval_rho(
            mol, ao, dm[spin], xctype="MGGA", hermi=1, with_lapl=True
        )
        np.testing.assert_allclose(fields.density[:, spin], reference[0], atol=2e-13)
        np.testing.assert_allclose(
            fields.gradient[:, spin], reference[1:4].T, atol=2e-13
        )
        np.testing.assert_allclose(fields.laplacian[:, spin], reference[4], atol=2e-13)

    reference = pbe_ao_potential(mol, dm, coords)
    assert reference["Vxc"].shape == (len(coords), 2)
    assert np.isfinite(reference["Vxc"]).all()
