"""Focused checks for the h-free Torch AO operator implementation."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from train_models.lap_operator import (
    LapEnergy,
    LapOperatorMetadata,
    assemble_rks_operator,
    assemble_spin_operators,
    integrated_xc_energy,
    operator_checkpoint_metadata,
    operator_loss,
    orthonormalize_operator,
    project_scalar_vxc,
    rks_density_features_from_ao,
    spin_density_features_from_ao,
    validate_operator_metadata,
)
from train_models.NN_models_lap import pcPBELMLOptimizerV2Lap


class _PolynomialEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(
            torch.tensor([0.7, 1.1, 0.13, -0.08, 0.19, 0.23, -0.17], dtype=torch.float64)
        )

    def forward(self, rho, sigma, lapl):
        p = self.coefficients
        return (
            p[0] * rho[:, 0]
            + p[1] * rho[:, 1]
            + p[2] * sigma[:, 0]
            + p[3] * sigma[:, 1]
            + p[4] * sigma[:, 2]
            + p[5] * lapl[:, 0]
            + p[6] * lapl[:, 1]
        )


def _random_inputs(seed=13, ngrid=5, nao=3):
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    phi = rand(ngrid, nao)
    grad_phi = rand(ngrid, 3, nao)
    lap_phi = rand(ngrid, nao)
    weights = torch.rand(ngrid, generator=generator, dtype=torch.float64) + 0.2
    features = rand(ngrid, 10)
    features[:, :2] = features[:, :2].abs() + 0.6
    features[:, 8:] *= 0.1
    return features, weights, phi, grad_phi, lap_phi


def _explicit_operator(c, a, b, weights, phi, grad_phi, lap_phi):
    nao = phi.shape[1]
    result = phi.new_zeros((nao, nao))
    for g in range(phi.shape[0]):
        p = phi[g]
        for i in range(nao):
            for j in range(nao):
                value = c[g] * p[i] * p[j]
                for axis in range(3):
                    value = value + a[g, axis] * (
                        grad_phi[g, axis, i] * p[j]
                        + p[i] * grad_phi[g, axis, j]
                    )
                value = value + b[g] * (
                    lap_phi[g, i] * p[j]
                    + p[i] * lap_phi[g, j]
                    + 2.0
                    * sum(
                        grad_phi[g, axis, i] * grad_phi[g, axis, j]
                        for axis in range(3)
                    )
                )
                result[i, j] = result[i, j] + weights[g] * value
    return result


def test_spin_assembly_matches_full_ao_product_derivative_and_chunks():
    features, weights, phi, grad_phi, lap_phi = _random_inputs()
    energy = _PolynomialEnergy()

    va, vb = assemble_spin_operators(
        energy, features, weights, phi, grad_phi, lap_phi, chunk_size=2
    )
    p = energy.coefficients.detach()
    grad = features[:, 2:8].reshape(-1, 2, 3)
    a_alpha = 2 * p[2] * grad[:, 0] + p[3] * grad[:, 1]
    a_beta = 2 * p[4] * grad[:, 1] + p[3] * grad[:, 0]
    expected_alpha = _explicit_operator(
        torch.full_like(weights, p[0]),
        a_alpha,
        torch.full_like(weights, p[5]),
        weights,
        phi,
        grad_phi,
        lap_phi,
    )
    expected_beta = _explicit_operator(
        torch.full_like(weights, p[1]),
        a_beta,
        torch.full_like(weights, p[6]),
        weights,
        phi,
        grad_phi,
        lap_phi,
    )
    torch.testing.assert_close(va, expected_alpha)
    torch.testing.assert_close(vb, expected_beta)
    torch.testing.assert_close(va, va.T)

    va_full, vb_full = assemble_spin_operators(
        energy, features, weights, phi, grad_phi, lap_phi, chunk_size=len(weights)
    )
    torch.testing.assert_close(va, va_full)
    torch.testing.assert_close(vb, vb_full)


def test_rks_spin_factors_and_dm_gradient_match_operator():
    features, weights, phi, grad_phi, lap_phi = _random_inputs(seed=17)
    energy = _PolynomialEnergy()
    va, vb = assemble_spin_operators(
        energy, features, weights, phi, grad_phi, lap_phi, chunk_size=2
    )
    vrks = assemble_rks_operator(
        energy, features, weights, phi, grad_phi, lap_phi, chunk_size=2
    )
    torch.testing.assert_close(vrks, 0.5 * (va + vb))

    # Closed-shell spin channels are half the total density. Their common
    # scalar PySCF derivatives must yield the same AO matrix without a final 1/2.
    closed = features.clone()
    closed[:, 1] = closed[:, 0]
    closed[:, 5:8] = closed[:, 2:5]
    closed[:, 9] = closed[:, 8]
    vrks_closed = assemble_rks_operator(
        energy, closed, weights, phi, grad_phi, lap_phi, chunk_size=3
    )
    p = energy.coefficients.detach()
    total_grad = closed[:, 2:5] + closed[:, 5:8]
    common_vrho = torch.full_like(weights, 0.5 * (p[0] + p[1]))
    common_vsigma = torch.full_like(weights, (p[2] + p[3] + p[4]) / 4.0)
    common_vlapl = torch.full_like(weights, 0.5 * (p[5] + p[6]))
    expected_rks = _explicit_operator(
        common_vrho,
        2.0 * common_vsigma[:, None] * total_grad,
        common_vlapl,
        weights,
        phi,
        grad_phi,
        lap_phi,
    )
    torch.testing.assert_close(vrks_closed, expected_rks)

    # The hand-assembled weak operator is the exact derivative with respect to
    # a full AO density matrix, including the off-diagonal trace convention.
    dm_a = torch.eye(phi.shape[1], dtype=phi.dtype, requires_grad=True) * 0.4
    dm_b = torch.eye(phi.shape[1], dtype=phi.dtype, requires_grad=True) * 0.3
    dm_a.retain_grad()
    dm_b.retain_grad()
    dm_features = spin_density_features_from_ao(
        phi, grad_phi, lap_phi, dm_a, dm_b
    )
    energy_value = integrated_xc_energy(energy, dm_features, weights)
    grad_a, grad_b = torch.autograd.grad(energy_value, (dm_a, dm_b))
    assert torch.isfinite(grad_a).all() and torch.isfinite(grad_b).all()
    dm_va, dm_vb = assemble_spin_operators(
        energy, dm_features, weights, phi, grad_phi, lap_phi, chunk_size=2
    )
    torch.testing.assert_close(grad_a, dm_va)
    torch.testing.assert_close(grad_b, dm_vb)

    total_dm = torch.eye(phi.shape[1], dtype=phi.dtype)
    features_from_rks = rks_density_features_from_ao(
        phi, grad_phi, lap_phi, total_dm
    )
    features_from_spin = spin_density_features_from_ao(
        phi, grad_phi, lap_phi, total_dm / 2, total_dm / 2
    )
    torch.testing.assert_close(features_from_rks, features_from_spin)


def test_scalar_projection_and_overlap_metric():
    phi = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.5], dtype=torch.float64)
    vxc = torch.tensor([2.0, -0.25], dtype=torch.float64)
    expected = sum(
        weights[g] * vxc[g] * torch.outer(phi[g], phi[g]) for g in range(2)
    )
    projected = project_scalar_vxc(phi, weights, vxc, chunk_size=1)
    torch.testing.assert_close(projected, expected)

    # A common scalar RKS reference projects once; no closed-shell half factor.
    unit_projection = project_scalar_vxc(
        torch.ones((1, 1), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(unit_projection, torch.tensor([[2.0]], dtype=torch.float64))

    overlap = torch.tensor([[1.3, 0.1], [0.1, 0.8]], dtype=torch.float64)
    pred = torch.tensor([[0.5, 0.2], [0.2, -0.3]], dtype=torch.float64)
    ref = torch.tensor([[0.2, -0.1], [-0.1, -0.2]], dtype=torch.float64)
    sinvhalf = torch.linalg.eigh(overlap)
    manual_sinvhalf = (
        sinvhalf[1] * sinvhalf[0].rsqrt()[None, :]
    ) @ sinvhalf[1].T
    expected_orth = manual_sinvhalf @ pred @ manual_sinvhalf
    torch.testing.assert_close(orthonormalize_operator(pred, overlap), expected_orth)
    torch.testing.assert_close(
        operator_loss(pred, ref, overlap),
        (manual_sinvhalf @ (pred - ref) @ manual_sinvhalf).square().sum() / 2,
    )
    with pytest.raises(ValueError, match="positive definite"):
        orthonormalize_operator(torch.eye(2, dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64))


def test_operator_loss_is_congruence_invariant_and_normalized_per_ao():
    dtype = torch.float64
    overlap = torch.tensor(
        [[1.2, 0.08, 0.02], [0.08, 0.9, 0.04], [0.02, 0.04, 1.1]],
        dtype=dtype,
    )
    pred = torch.tensor(
        [[0.4, 0.1, -0.03], [0.1, -0.2, 0.05], [-0.03, 0.05, 0.3]],
        dtype=dtype,
    )
    ref = torch.tensor(
        [[0.1, -0.04, 0.02], [-0.04, 0.25, 0.01], [0.02, 0.01, -0.1]],
        dtype=dtype,
    )
    transform = torch.tensor(
        [[1.1, 0.2, -0.05], [0.1, 0.9, 0.12], [0.03, -0.08, 1.2]],
        dtype=dtype,
    )
    baseline = operator_loss(pred, ref, overlap)
    changed_basis = operator_loss(
        transform.T @ pred @ transform,
        transform.T @ ref @ transform,
        transform.T @ overlap @ transform,
    )
    torch.testing.assert_close(changed_basis, baseline, rtol=1e-10, atol=1e-12)

    # One unit matrix error has squared Hilbert-Schmidt norm 1; divide by nAO.
    expected_normalized = operator_loss(
        torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=dtype)),
        torch.zeros((3, 3), dtype=dtype),
        torch.eye(3, dtype=dtype),
    )
    torch.testing.assert_close(expected_normalized, torch.tensor(1.0 / 3.0, dtype=dtype))


def test_operator_checkpoint_metadata_is_immutable_and_excludes_stencil_fields():
    metadata = operator_checkpoint_metadata()
    assert isinstance(metadata, LapOperatorMetadata)
    with pytest.raises(FrozenInstanceError):
        metadata.protocol = "legacy-stencil-vxc"
    values = metadata.to_dict()
    validate_operator_metadata(values)
    assert "h" not in values and "stencil" not in values
    with pytest.raises(ValueError, match="h or stencil"):
        validate_operator_metadata({**values, "stencil": {"version": "7pt"}})
    with pytest.raises(ValueError, match="Incompatible variational"):
        validate_operator_metadata({**values, "protocol": "lap-strong-form-stencil-v1"})


def test_lap_energy_operator_backpropagates_to_network_parameters():
    torch.manual_seed(41)
    model = pcPBELMLOptimizerV2Lap(
        num_layers=2, h_dim=8, use_g_x=True, use_g_c=True
    ).double()
    energy = LapEnergy(model)
    ngrid, nao = 4, 3
    features = torch.zeros((ngrid, 10), dtype=torch.float64)
    features[:, :2] = 0.45
    features[:, 2:8] = torch.randn((ngrid, 6), dtype=torch.float64) * 0.01
    features[:, 8:10] = torch.randn((ngrid, 2), dtype=torch.float64) * 0.02
    weights = torch.full((ngrid,), 0.2, dtype=torch.float64)
    phi = torch.randn((ngrid, nao), dtype=torch.float64)
    grad_phi = torch.randn((ngrid, 3, nao), dtype=torch.float64)
    lap_phi = torch.randn((ngrid, nao), dtype=torch.float64)

    operator = assemble_rks_operator(
        energy, features, weights, phi, grad_phi, lap_phi, chunk_size=2
    )
    operator.square().sum().backward()
    parameter_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)
    assert sum(grad.abs().sum() for grad in parameter_grads) > 0


def test_float32_operator_uses_double_learned_branch_without_mutating_model(monkeypatch):
    import train_models.lap_operator as implementation

    torch.manual_seed(41)
    model = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8, use_g_x=True, use_g_c=True)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    features, weights, phi, grad_phi, lap_phi = (value.float() for value in _random_inputs())
    observed = []
    original_forward = model.forward
    original_pbe = implementation.PBE.F_PBE

    def forward(raw):
        assert raw.dtype == torch.float64
        assert all(p.dtype == torch.float64 for p in model.parameters())
        result = original_forward(raw)
        assert result.dtype == torch.float64
        observed.append("learned-double")
        return result

    def pbe(rho, sigma, constants, *args, **kwargs):
        assert rho.dtype == sigma.dtype == constants.dtype == torch.float32
        observed.append("pbe-original")
        return original_pbe(rho, sigma, constants, *args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(implementation.PBE, "F_PBE", pbe)
    matrix = assemble_rks_operator(LapEnergy(model), features, weights, phi, grad_phi, lap_phi)
    assert matrix.dtype == torch.float64
    assert "learned-double" in observed and "pbe-original" in observed
    matrix.square().sum().backward()
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state.items())
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
