"""Independent weak-form XC operator tests; all fixtures avoid spatial FD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lap_operator import (
    assemble_rks_operator,
    integrated_xc_energy,
    operator_loss,
    rks_density_features_from_ao,
)
from lap_vxc import LapEnergy
from test_lap import small_model

from dft_functionals import PBE
from dft_functionals.constants import PBE_CONSTANTS


class ScalarPolynomialEnergy(nn.Module):
    """Spin-symmetric total-density polynomial local energy density."""

    def __init__(self, a=0.7, b=0.0, c=0.0, dtype=torch.float64):
        super().__init__()
        self.register_buffer("coefficients", torch.tensor([a, b, c], dtype=dtype))

    def forward(self, rho, sigma, lapl):
        a, b, c = self.coefficients
        rho_total = rho.sum(-1)
        sigma_total = sigma[:, 0] + 2.0 * sigma[:, 1] + sigma[:, 2]
        lapl_total = lapl.sum(-1)
        return a * rho_total.square() + b * sigma_total + c * lapl_total.square()


class CanonicalPBEEnergy(nn.Module):
    def forward(self, rho, sigma, lapl):
        del lapl
        constants = PBE_CONSTANTS.to(dtype=rho.dtype, device=rho.device).expand(
            len(rho), -1
        )
        return PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(-1)


class ScalarParameterEnergy(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def forward(self, rho, sigma, lapl):
        rt = rho.sum(-1)
        st = sigma[:, 0] + 2 * sigma[:, 1] + sigma[:, 2]
        lt = lapl.sum(-1)
        return self.theta * (rt.square() + 0.23 * st + 0.11 * lt.square())


def _gaussian_fixture(order=24):
    """Three s-Gaussian AOs and exact tensor Gauss-Hermite quadrature."""
    alpha = np.full(3, 0.7, dtype=np.float64)
    centers = np.array(
        [[-0.36, 0.02, 0.05], [0.28, -0.21, 0.08], [0.09, 0.33, -0.27]],
        dtype=np.float64,
    )
    # Golub-Welsch Hermite rule from its symmetric Jacobi matrix. Use Torch's
    # eigensolver so importing this test does not mix NumPy and Torch MKL runtimes.
    offdiag = torch.sqrt(torch.arange(1, order, dtype=torch.float64) / 2.0)
    jacobi = torch.diag(offdiag, diagonal=1) + torch.diag(offdiag, diagonal=-1)
    nodes_t, vectors = torch.linalg.eigh(jacobi)
    nodes = nodes_t.numpy()
    one_weights = np.sqrt(np.pi) * vectors[0].square().numpy()
    gx, gy, gz = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    mesh = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    gwx, gwy, gwz = np.meshgrid(one_weights, one_weights, one_weights, indexing="ij")
    # Weak and strong polynomial integrands decay as exp(-2.8 |r|^2).
    beta = 4.0 * alpha[0]
    coords = mesh / np.sqrt(beta)
    weights = (
        gwx * gwy * gwz * np.exp(np.square(mesh).sum(-1).reshape(order, order, order))
        / beta**1.5
    ).reshape(-1)

    displacement = coords[:, None, :] - centers[None, :, :]
    radius2 = np.square(displacement).sum(-1)
    phi = np.exp(-alpha[None, :] * radius2)
    grad = (-2.0 * alpha[None, :, None] * displacement * phi[..., None]).transpose(
        0, 2, 1
    )
    lap = (4.0 * alpha[None, :] ** 2 * radius2 - 6.0 * alpha[None, :]) * phi
    # Positive definite total RKS density matrix.
    dm = np.array(
        [[0.82, 0.12, 0.04], [0.12, 0.63, 0.08], [0.04, 0.08, 0.47]],
        dtype=np.float64,
    )
    tensors = [torch.as_tensor(x, dtype=torch.float64) for x in (phi, grad, lap, weights)]
    return {
        "alpha": alpha,
        "centers": centers,
        "coords": coords,
        "phi_np": phi,
        "grad_np": grad,
        "lap_np": lap,
        "weights_np": weights,
        "dm_np": dm,
        "phi": tensors[0],
        "grad_phi": tensors[1],
        "lap_phi": tensors[2],
        "weights": tensors[3],
        "dm": torch.as_tensor(dm, dtype=torch.float64),
    }


def _pair_density(fields, dm):
    """Independent analytic rho, grad, lap, and bi-lap from Gaussian AO pairs."""
    alpha, centers, coords = fields["alpha"], fields["centers"], fields["coords"]
    rho = np.zeros(len(coords), dtype=np.float64)
    grad = np.zeros((len(coords), 3), dtype=np.float64)
    lap = np.zeros(len(coords), dtype=np.float64)
    bilap = np.zeros(len(coords), dtype=np.float64)
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            gamma = alpha[i] + alpha[j]
            center = (alpha[i] * centers[i] + alpha[j] * centers[j]) / gamma
            prefactor = np.exp(
                -(alpha[i] * alpha[j] / gamma)
                * np.square(centers[i] - centers[j]).sum()
            )
            x = coords - center
            r2 = np.square(x).sum(-1)
            pair = prefactor * np.exp(-gamma * r2)
            coefficient = dm[i, j]
            rho += coefficient * pair
            grad += coefficient * (-2.0 * gamma * x) * pair[:, None]
            lap += coefficient * (4.0 * gamma**2 * r2 - 6.0 * gamma) * pair
            bilap += coefficient * (
                16.0 * gamma**4 * r2**2 - 80.0 * gamma**3 * r2 + 60.0 * gamma**2
            ) * pair
    return rho, grad, lap, bilap


def _independent_rks_features(fields, dm_total):
    """Rebuild RKS descriptors directly, separate from the tested helper."""
    p = 0.5 * np.asarray(dm_total, dtype=np.float64)
    phi, grad, lap_phi = fields["phi_np"], fields["grad_np"], fields["lap_np"]
    rho = np.einsum("gi,ij,gj->g", phi, p, phi, optimize=True)
    grad_rho = np.einsum("gdi,ij,gj->gd", grad, p, phi, optimize=True)
    grad_rho += np.einsum("gi,ij,gdj->gd", phi, p, grad, optimize=True)
    lap_rho = np.einsum("gi,ij,gj->g", lap_phi, p, phi, optimize=True)
    lap_rho += np.einsum("gi,ij,gj->g", phi, p, lap_phi, optimize=True)
    lap_rho += 2.0 * np.einsum("gdi,ij,gdj->g", grad, p, grad, optimize=True)
    # ``p`` is already dm_total/2, so these are spin densities directly.
    rho_spin = np.column_stack([rho, rho])
    grad_spin = np.stack([grad_rho, grad_rho], axis=1)
    lap_spin = np.column_stack([lap_rho, lap_rho])
    ga, gb = grad_spin[:, 0], grad_spin[:, 1]
    sigma = np.column_stack(
        [(ga * ga).sum(-1), (ga * gb).sum(-1), (gb * gb).sum(-1)]
    )
    f = np.column_stack(
        [rho_spin, grad_spin[:, 0], grad_spin[:, 1], lap_spin]
    )
    return torch.as_tensor(f, dtype=torch.float64), torch.as_tensor(sigma, dtype=torch.float64)


def _weak_and_strong(fields, a, b=0.0, c=0.0):
    energy = ScalarPolynomialEnergy(a=a, b=b, c=c)
    features = rks_density_features_from_ao(
        fields["phi"], fields["grad_phi"], fields["lap_phi"], fields["dm"]
    )
    weak = assemble_rks_operator(
        energy,
        features,
        fields["weights"],
        fields["phi"],
        fields["grad_phi"],
        fields["lap_phi"],
        chunk_size=2048,
        create_graph=False,
    ).detach().numpy()
    rho, _, lap, bilap = _pair_density(fields, fields["dm_np"])
    strong_v = 2.0 * a * rho - 2.0 * b * lap + 2.0 * c * bilap
    # Independent NumPy projection of the exact pointwise Euler potential.
    strong = np.einsum(
        "g,gi,gj->ij", fields["weights_np"] * strong_v,
        fields["phi_np"], fields["phi_np"], optimize=True
    )
    return weak, strong


@pytest.mark.parametrize("a", [0.7, 1.3])
def test_manufactured_lda_weak_operator_equals_analytic_projection(a):
    fields = _gaussian_fixture(order=24)
    weak, strong = _weak_and_strong(fields, a)
    np.testing.assert_allclose(weak, strong, rtol=3e-12, atol=3e-12)


def test_manufactured_gga_weak_operator_equals_analytic_projection():
    fields = _gaussian_fixture(order=24)
    weak, strong = _weak_and_strong(fields, a=0.73, b=0.31)
    np.testing.assert_allclose(weak, strong, rtol=4e-11, atol=4e-12)


def test_manufactured_laplacian_weak_operator_equals_analytic_bilaplacian_projection():
    fields = _gaussian_fixture(order=28)
    weak, strong = _weak_and_strong(fields, a=0.73, b=0.31, c=0.17)
    np.testing.assert_allclose(weak, strong, rtol=8e-10, atol=8e-11)


def test_density_matrix_directional_derivative_converges_for_lda_pbe_and_lap_nn():
    fields = _gaussian_fixture(order=8)
    direction = torch.tensor(
        [[0.13, -0.04, 0.03], [-0.04, -0.08, 0.02], [0.03, 0.02, 0.06]],
        dtype=torch.float64,
    )
    dm = fields["dm"]
    functionals = {
        "LDA": ScalarPolynomialEnergy(a=0.37),
        "PBE": CanonicalPBEEnergy(),
        "LapNN": LapEnergy(small_model()),
    }
    errors = {}
    for name, energy in functionals.items():
        features = rks_density_features_from_ao(
            fields["phi"], fields["grad_phi"], fields["lap_phi"], dm
        )
        predicted = assemble_rks_operator(
            energy,
            features,
            fields["weights"],
            fields["phi"],
            fields["grad_phi"],
            fields["lap_phi"],
            chunk_size=128,
            create_graph=False,
        )
        analytic = torch.sum(predicted * direction).item()
        sequence = []
        for eps in (3e-4, 1e-4, 3e-5, 1e-5):
            plus, sigma_plus = _independent_rks_features(fields, (dm + eps * direction).numpy())
            minus, sigma_minus = _independent_rks_features(fields, (dm - eps * direction).numpy())
            # The canonical feature layout carries gradients, not sigmas; assert
            # the independent sigma construction agrees with its chain rule.
            torch.testing.assert_close(
                sigma_plus,
                torch.stack(
                    [
                        (plus[:, 2:5] ** 2).sum(-1),
                        (plus[:, 2:5] * plus[:, 5:8]).sum(-1),
                        (plus[:, 5:8] ** 2).sum(-1),
                    ],
                    dim=-1,
                ),
                rtol=1e-13,
                atol=1e-14,
            )
            torch.testing.assert_close(
                sigma_minus,
                torch.stack(
                    [
                        (minus[:, 2:5] ** 2).sum(-1),
                        (minus[:, 2:5] * minus[:, 5:8]).sum(-1),
                        (minus[:, 5:8] ** 2).sum(-1),
                    ],
                    dim=-1,
                ),
                rtol=1e-13,
                atol=1e-14,
            )
            ep = integrated_xc_energy(energy, plus, fields["weights"])
            em = integrated_xc_energy(energy, minus, fields["weights"])
            numeric = ((ep - em) / (2.0 * eps)).item()
            sequence.append(abs(numeric - analytic))
        errors[name] = sequence
        scale = max(1.0, abs(analytic))
        assert min(sequence) / scale < 2e-7, (name, analytic, sequence)
        # Error first decreases as the symmetric step shrinks, then reaches a
        # roundoff floor (the smallest epsilon need not be best).
        assert min(sequence[1:3]) <= sequence[0], (name, analytic, sequence)
    assert set(errors) == {"LDA", "PBE", "LapNN"}


def test_operator_loss_backpropagates_through_lap_nn_and_gradcheck():
    fields = _gaussian_fixture(order=4)
    model = small_model()
    energy = LapEnergy(model)
    features = rks_density_features_from_ao(
        fields["phi"], fields["grad_phi"], fields["lap_phi"], fields["dm"]
    )
    predicted = assemble_rks_operator(
        energy,
        features,
        fields["weights"],
        fields["phi"],
        fields["grad_phi"],
        fields["lap_phi"],
        chunk_size=32,
        create_graph=True,
    )
    overlap = torch.einsum(
        "g,gi,gj->ij", fields["weights"], fields["phi"], fields["phi"]
    )
    loss = operator_loss(predicted, torch.zeros_like(predicted), overlap)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(torch.linalg.vector_norm(g)) for g in grads) > 0.0

    # Check the outer parameter derivative of an operator loss independently
    # on a one-parameter analytic Lap-level energy.
    tiny = _gaussian_fixture(order=3)
    tiny_features = rks_density_features_from_ao(
        tiny["phi"], tiny["grad_phi"], tiny["lap_phi"], tiny["dm"]
    )
    tiny_overlap = torch.einsum(
        "g,gi,gj->ij", tiny["weights"], tiny["phi"], tiny["phi"]
    )

    def loss_of_theta(theta):
        v = assemble_rks_operator(
            ScalarParameterEnergy(theta),
            tiny_features,
            tiny["weights"],
            tiny["phi"],
            tiny["grad_phi"],
            tiny["lap_phi"],
            chunk_size=16,
            create_graph=True,
        )
        return operator_loss(v, torch.zeros_like(v), tiny_overlap)

    theta = torch.tensor(0.83, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(loss_of_theta, (theta,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_spin_resolved_operators_collapse_to_rks_without_a_half_factor():
    fields = _gaussian_fixture(order=8)
    energy = ScalarPolynomialEnergy(a=0.4, b=0.13, c=0.08)
    features = rks_density_features_from_ao(
        fields["phi"], fields["grad_phi"], fields["lap_phi"], fields["dm"]
    )
    # Symmetry of the energy and equal-spin density imply V_alpha=V_beta=V_RKS.
    from lap_operator import assemble_spin_operators

    va, vb = assemble_spin_operators(
        energy,
        features,
        fields["weights"],
        fields["phi"],
        fields["grad_phi"],
        fields["lap_phi"],
        chunk_size=128,
        create_graph=False,
    )
    vrks = assemble_rks_operator(
        energy,
        features,
        fields["weights"],
        fields["phi"],
        fields["grad_phi"],
        fields["lap_phi"],
        chunk_size=128,
        create_graph=False,
    )
    torch.testing.assert_close(va, vb, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(vrks, va, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(vrks, 0.5 * (va + vb), rtol=0, atol=0)
    # The closed-shell energy directional derivative contracts with the common
    # matrix once, not one-half of it.
    direction = torch.tensor(
        [[0.03, 0.01, 0.00], [0.01, -0.02, 0.01], [0.00, 0.01, 0.02]],
        dtype=torch.float64,
    )
    eps = 1e-5
    dm = fields["dm"]
    eplus = integrated_xc_energy(
        energy,
        rks_density_features_from_ao(
            fields["phi"], fields["grad_phi"], fields["lap_phi"], dm + eps * direction
        ),
        fields["weights"],
    )
    eminus = integrated_xc_energy(
        energy,
        rks_density_features_from_ao(
            fields["phi"], fields["grad_phi"], fields["lap_phi"], dm - eps * direction
        ),
        fields["weights"],
    )
    fd = (eplus - eminus) / (2 * eps)
    contraction = torch.sum(vrks * direction)
    torch.testing.assert_close(fd, contraction, rtol=3e-8, atol=3e-9)


def test_torch_operator_matches_actual_pyscf_lap_rks_matrix():
    pytest.importorskip("pyscf")
    from lap_operator import LapEnergy, assemble_rks_operator
    from pyscf import gto
    from pyscf.dft import numint as pyscf_numint

    from test_models.DFT.lap_functional import LapFunctional

    mol = gto.M(
        atom="H 0 0 -0.70; H 0 0 0.70",
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    functional = LapFunctional(small_model())
    mf = functional.make_rks(mol)
    mf.grids.level = 1
    mf.grids.build()
    dm = mf.get_init_guess()
    ao = pyscf_numint.eval_ao(mol, mf.grids.coords, deriv=2)
    phi = torch.as_tensor(ao[0], dtype=torch.float64)
    grad_phi = torch.as_tensor(ao[1:4].transpose(1, 0, 2), dtype=torch.float64)
    lap_phi = torch.as_tensor(ao[4] + ao[7] + ao[9], dtype=torch.float64)
    weights = torch.as_tensor(mf.grids.weights, dtype=torch.float64)
    features = rks_density_features_from_ao(phi, grad_phi, lap_phi, torch.as_tensor(dm))
    matrix_torch = assemble_rks_operator(
        LapEnergy(functional.model),
        features,
        weights,
        phi,
        grad_phi,
        lap_phi,
        chunk_size=256,
        create_graph=False,
    )
    # This is the matrix returned by the existing NumIntWithLaplacian RKS path.
    _, _, matrix_pyscf = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)
    delta = matrix_torch.numpy() - matrix_pyscf
    relative = np.linalg.norm(delta) / np.linalg.norm(matrix_pyscf)
    assert relative < 2e-9, (relative, np.max(np.abs(delta)))
    assert np.max(np.abs(delta)) < 2e-10
