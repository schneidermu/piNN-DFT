"""Numerical regressions for the PySCF 2.11 XC integration path."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf

from test_models.DFT import numint as numint_module
from test_models.DFT.numint import NumIntWithLaplacian


class PolynomialXCNumInt(NumIntWithLaplacian):
    """A differentiable test XC with independently controlled ingredients."""

    def __init__(self, rho=0.0, sigma=0.0, lapl=0.0, tau=0.0):
        super().__init__()
        self.coefficients = (rho, sigma, lapl, tau)

    def _xc_type(self, xc_code):
        return "MGGA"

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, **kwargs):
        c_rho, c_sigma, c_lapl, c_tau = self.coefficients
        if spin == 0:
            density = rho[0]
            gradient = rho[1:4]
            sigma = np.einsum("xp,xp->p", gradient, gradient)
            energy_density = (
                c_rho * density**2
                + c_sigma * sigma
                + c_lapl * rho[4]
                + c_tau * rho[5]
            )
            exc = energy_density / np.maximum(density, 1e-200)
            vxc = (
                2.0 * c_rho * density,
                np.full(density.size, c_sigma),
                np.full(density.size, c_lapl),
                np.full(density.size, c_tau),
            )
            return exc, vxc

        rho_a, rho_b = rho
        density = rho_a[0] + rho_b[0]
        grad_a, grad_b = rho_a[1:4], rho_b[1:4]
        grad_total = grad_a + grad_b
        sigma_total = np.einsum("xp,xp->p", grad_total, grad_total)
        energy_density = (
            c_rho * density**2
            + c_sigma * sigma_total
            + c_lapl * (rho_a[4] + rho_b[4])
            + c_tau * (rho_a[5] + rho_b[5])
        )
        exc = energy_density / np.maximum(density, 1e-200)
        vxc = (
            np.column_stack((2.0 * c_rho * density, 2.0 * c_rho * density)),
            np.tile((c_sigma, 2.0 * c_sigma, c_sigma), (density.size, 1)),
            np.full((density.size, 2), c_lapl),
            np.full((density.size, 2), c_tau),
        )
        return exc, vxc


@pytest.fixture(scope="module")
def h2_system():
    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    grids = dft.gen_grid.Grids(mol)
    # The regression differentiates a fixed quadrature; production grid
    # accuracy is irrelevant here, while a compact grid keeps tests quick.
    grids.level = 0
    grids.build()
    reference = scf.RHF(mol)
    reference.conv_tol = 1e-12
    reference.kernel()
    return mol, grids, reference.make_rdm1()


def _symmetric_direction(shape):
    rng = np.random.default_rng(2409)
    direction = rng.normal(size=shape)
    direction = 0.5 * (direction + direction.T)
    return direction / np.linalg.norm(direction)


def _rks_energy(ni, mol, grids, density_matrix):
    return ni.nr_rks(mol, grids, "TEST-MGGA", density_matrix)[1]


def _uks_energy(ni, mol, grids, density_matrices):
    return ni.nr_uks(mol, grids, "TEST-MGGA", density_matrices)[1]


def _assert_finite_difference(fd, matrix_derivative):
    assert np.isfinite(fd)
    assert np.isfinite(matrix_derivative)
    assert np.isclose(fd, matrix_derivative, rtol=3e-6, atol=2e-9), (
        f"finite difference {fd:.12g} != matrix derivative "
        f"{matrix_derivative:.12g}"
    )


def _check_uks_direction(ni, mol, grids, dma, dmb, spin, direction):
    dms = (dma, dmb)
    _, _, vmat = ni.nr_uks(mol, grids, "TEST-MGGA", dms)
    epsilons = (1e-4, 3e-5, 1e-5)
    analytic = np.einsum("ij,ji->", vmat[spin], direction)
    errors = []
    for epsilon in epsilons:
        plus = [dma.copy(), dmb.copy()]
        minus = [dma.copy(), dmb.copy()]
        plus[spin] += epsilon * direction
        minus[spin] -= epsilon * direction
        fd = (_uks_energy(ni, mol, grids, tuple(plus)) - _uks_energy(ni, mol, grids, tuple(minus))) / (2.0 * epsilon)
        errors.append(abs(fd - analytic))
    assert min(errors) <= 2e-9 + 3e-6 * abs(analytic), (analytic, errors)
    return analytic


def test_rks_matrix_is_energy_derivative_including_tau_and_laplacian(h2_system):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(rho=0.17, sigma=0.09, lapl=0.13, tau=0.31)
    direction = _symmetric_direction(dm.shape)
    _, _, vmat = ni.nr_rks(mol, grids, "TEST-MGGA", dm)
    analytic = np.einsum("ij,ji->", vmat, direction)
    errors = []
    for epsilon in (1e-4, 3e-5, 1e-5):
        fd = (
            _rks_energy(ni, mol, grids, dm + epsilon * direction)
            - _rks_energy(ni, mol, grids, dm - epsilon * direction)
        ) / (2.0 * epsilon)
        errors.append(abs(fd - analytic))
    assert min(errors) <= 2e-9 + 3e-6 * abs(analytic), (analytic, errors)


@pytest.mark.parametrize("spin", [0, 1], ids=["alpha", "beta"])
def test_uks_spin_matrix_matches_independent_density_finite_difference(h2_system, spin):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(rho=0.17, sigma=0.09, lapl=0.13, tau=0.31)
    dma = dmb = dm / 2.0
    direction = _symmetric_direction(dm.shape)
    _check_uks_direction(ni, mol, grids, dma, dmb, spin, direction)


@pytest.mark.parametrize("spin", [0, 1], ids=["alpha", "beta"])
def test_uks_tau_matrix_has_the_full_variational_coefficient(h2_system, spin):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(tau=1.0)
    direction = _symmetric_direction(dm.shape)
    _check_uks_direction(ni, mol, grids, dm / 2.0, dm / 2.0, spin, direction)


def test_uks_tau_finite_difference_detects_legacy_half_factor(h2_system, monkeypatch):
    original_tau_dot = numint_module._tau_dot_sparse

    def legacy_tau_dot(bra, ket, wv, *args, **kwargs):
        return original_tau_dot(bra, ket, 0.5 * wv, *args, **kwargs)

    monkeypatch.setattr(numint_module, "_tau_dot_sparse", legacy_tau_dot)
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(tau=1.0)
    with pytest.raises(AssertionError):
        _check_uks_direction(
            ni, mol, grids, dm / 2.0, dm / 2.0, 0, _symmetric_direction(dm.shape)
        )


@pytest.mark.parametrize("spin", [0, 1], ids=["alpha", "beta"])
def test_uks_laplacian_matrix_matches_finite_difference(h2_system, spin):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(lapl=1.0)
    direction = _symmetric_direction(dm.shape)
    _check_uks_direction(ni, mol, grids, dm / 2.0, dm / 2.0, spin, direction)


def test_uks_small_nonzero_laplacian_derivative_is_not_dropped(h2_system):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(lapl=1e-12)
    direction = _symmetric_direction(dm.shape)
    derivative = _check_uks_direction(
        ni, mol, grids, dm / 2.0, dm / 2.0, 0, direction
    )
    assert derivative != 0.0


def test_closed_shell_rks_and_uks_agree(h2_system):
    mol, grids, dm = h2_system
    ni = PolynomialXCNumInt(rho=0.17, sigma=0.09, lapl=0.13, tau=0.31)
    nelec_rks, exc_rks, vmat_rks = ni.nr_rks(mol, grids, "TEST-MGGA", dm)
    nelec_uks, exc_uks, vmat_uks = ni.nr_uks(
        mol, grids, "TEST-MGGA", (dm / 2.0, dm / 2.0)
    )
    np.testing.assert_allclose(nelec_uks.sum(axis=0), nelec_rks, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(exc_uks, exc_rks, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(vmat_uks[0], vmat_rks, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(vmat_uks[1], vmat_rks, rtol=1e-9, atol=1e-9)


def test_canonical_pbe_energy_and_first_derivatives_match_pyscf_211():
    from dft_functionals import PBE, PBE_CONSTANTS
    import torch

    version = tuple(int(part) for part in pyscf.__version__.split(".")[:2])
    assert version >= (2, 11), f"This regression targets PySCF 2.11+, got {pyscf.__version__}"

    rho_spin = np.array(
        [
            [[0.61, 0.28, 0.83], [0.07, -0.025, 0.035], [0.02, 0.03, -0.01], [0.015, -0.01, 0.02]],
            [[0.39, 0.17, 0.52], [0.025, -0.018, 0.02], [-0.01, 0.015, 0.01], [0.02, 0.012, -0.01]],
        ],
        dtype=np.float64,
    )
    rho = torch.tensor(rho_spin[:, 0, :].T.copy(), dtype=torch.float64, requires_grad=True)
    gradients = rho_spin[:, 1:4, :]
    sigma_np = np.stack(
        [
            np.einsum("ip,ip->p", gradients[0], gradients[0]),
            np.einsum("ip,ip->p", gradients[0], gradients[1]),
            np.einsum("ip,ip->p", gradients[1], gradients[1]),
        ],
        axis=1,
    )
    sigma = torch.tensor(sigma_np, dtype=torch.float64, requires_grad=True)
    constants = PBE_CONSTANTS.to(dtype=torch.float64).expand(rho.shape[0], -1)
    exc = PBE.F_PBE(rho, sigma, constants, torch.device("cpu"))
    energy = (rho.sum(dim=1) * exc).sum()
    vrho, vsigma = torch.autograd.grad(energy, (rho, sigma))

    reference_exc, reference_vxc = __import__("pyscf.dft.libxc", fromlist=["eval_xc"]).eval_xc(
        "PBE", rho_spin, spin=1, deriv=1
    )[:2]
    np.testing.assert_allclose(exc.detach().numpy(), reference_exc, rtol=1e-6, atol=2e-8)
    np.testing.assert_allclose(vrho.detach().numpy(), reference_vxc[0], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(vsigma.detach().numpy(), reference_vxc[1], rtol=2e-5, atol=2e-7)
