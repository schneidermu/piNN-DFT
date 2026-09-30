"""Scientific regression tests: analytic Euler derivatives and tau-free NN."""

import pytest
import torch
from lap_checkpoint import checkpoint_payload, load_lap_checkpoint
from lap_data import evaluate_stencil, normalize_target, validate_record
from lap_vxc import (
    LapEnergy,
    euler_components,
    full_vxc_loss,
    gradient_chain_rule,
    integrated_energy,
    local_partials,
    potential_loss,
    sigma_from_gradients,
    sigma_standard_to_total,
    sigma_total_to_standard,
    stencil_coordinates,
)
from NN_models import pcPBELMLOptimizerV2
from NN_models_lap import pcPBELMLOptimizerV2Lap
from torch import nn

from dft_functionals.constants import NN_OUTPUT_SCALE_PBE


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
    """rho_s=4+amplitude_s sum cos(x_i), with exact grad/lap/bi-lap."""
    amplitude = xyz.new_tensor([0.25, 0.25 if equal_spin else 0.4])
    rho = 4 + xyz.cos().sum(-1, keepdim=True) * amplitude
    grad = -xyz.sin().unsqueeze(-2) * amplitude[:, None]
    lap = -xyz.cos().sum(-1, keepdim=True) * amplitude
    return torch.cat([rho, grad.flatten(-2), lap], -1)


def features(h, equal_spin=False, n=11, dtype=torch.float64):
    xyz = torch.linspace(0.15, 1.7, 3 * n, dtype=torch.float64).reshape(n, 3)
    return analytic_features(stencil_coordinates(xyz, h), equal_spin).to(dtype)


@pytest.mark.parametrize("equal_spin", [False, True])
@pytest.mark.parametrize("b,c", [(0.0, 0.0), (0.3, 0.0), (0.3, 0.2)])
def test_analytic_euler_convergence(b, c, equal_spin):
    energy = PolynomialEnergy(b=b, c=c)
    errors = []
    for h in (0.2, 0.1, 0.05):
        f = features(h, equal_spin)
        parts = euler_components(energy, f, h)
        rho, lap = f[:, 0, :2], f[:, 0, 8:]
        exact = 1.4 * rho - 2 * b * lap - 2 * c * lap  # bi-lap=-lap
        errors.append(float((parts["Vxc"] - exact).abs().max().detach()))
        if b == c == 0:
            assert torch.equal(parts["minus_div_A"], torch.zeros_like(rho))
            assert torch.equal(parts["lap_B"], torch.zeros_like(rho))
            torch.testing.assert_close(parts["Vxc"], exact, atol=1e-14, rtol=0)
    if b or c:
        assert 3.9 < errors[0] / errors[1] < 4.1
        assert 3.9 < errors[1] / errors[2] < 4.1


def test_sigma_chain_rule_open_and_closed_shell():
    torch.manual_seed(7)
    for equal in (False, True):
        g = torch.randn(9, 2, 3, dtype=torch.float64, requires_grad=True)
        if equal:
            g = g[:, :1].repeat(1, 2, 1).detach().requires_grad_(True)
        s = sigma_from_gradients(g)
        torch.testing.assert_close(
            sigma_total_to_standard(sigma_standard_to_total(s)), s
        )
        es = torch.tensor([2.0, 3.0, 5.0], dtype=g.dtype).expand_as(s)
        direct = torch.autograd.grad((s * es).sum(), g)[0]
        torch.testing.assert_close(gradient_chain_rule(es, g), direct)


def test_absolute_pointwise_loss_keeps_gauge():
    rho = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    w = torch.tensor([2.0, 5.0], dtype=rho.dtype)
    target = torch.zeros_like(rho)
    pred = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=rho.dtype)
    expected = ((rho * w[:, None]) * pred.square()).sum() / (rho * w[:, None]).sum()
    assert potential_loss(pred, target, rho, w) == expected
    assert potential_loss(target + 3, target, rho, w) == 9


@pytest.mark.parametrize("chunk", [1, 4, 20])
def test_chunked_matches_loss_and_gradients(chunk):
    h = 0.1
    f = features(h)
    w = torch.linspace(0.1, 1.0, len(f), dtype=f.dtype)
    t = f[:, 0, :2] * 0.1
    energy = PolynomialEnergy(b=0.3, c=0.2)
    direct = potential_loss(euler_components(energy, f, h)["Vxc"], t, f[:, 0, :2], w)
    dg = torch.autograd.grad(direct, energy.parameters())
    bounded = full_vxc_loss(energy, f, t, w, h, chunk)
    bg = torch.autograd.grad(bounded, energy.parameters())
    torch.testing.assert_close(bounded, direct, atol=1e-12, rtol=1e-12)
    for a, b in zip(dg, bg):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    center = f[:, 0]
    reference = (
        energy(
            center[:, :2],
            sigma_from_gradients(center[:, 2:8].reshape(-1, 2, 3)),
            center[:, 8:],
        )
        * w
    ).sum()
    torch.testing.assert_close(integrated_energy(energy, f, w, chunk), reference)


def small_model():
    torch.manual_seed(41)
    return pcPBELMLOptimizerV2Lap(2, 8, use_g_x=True, use_g_c=True).double()


def raw_grid():
    f = features(0.1)[:, 0]
    return torch.cat(
        [
            f[:, :2],
            sigma_standard_to_total(sigma_from_gradients(f[:, 2:8].reshape(-1, 2, 3))),
            torch.ones_like(f[:, :2]),
            f[:, 8:],
        ],
        -1,
    )


def test_tau_exactly_absent_and_spin_symmetry():
    model, raw = small_model(), raw_grid().requires_grad_(True)
    baseline = model(raw)
    changed = raw.detach().clone()
    changed[:, 5:7] = torch.linspace(
        -1e200, 1e200, len(raw) * 2, dtype=raw.dtype
    ).reshape(-1, 2)
    assert torch.equal(model(changed), baseline)
    grad = torch.autograd.grad(baseline.sum(), raw)[0]
    assert torch.equal(grad[:, 5:7], torch.zeros_like(grad[:, 5:7]))
    swapped = raw.detach()[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]]
    torch.testing.assert_close(
        model(swapped),
        baseline[:, list(range(22)) + [24, 25, 22, 23, 27, 26, 28]],
        atol=1e-14,
        rtol=1e-14,
    )
    assert baseline.shape == (11, 29)


def test_every_exact_anchor():
    model = small_model()
    d = model.get_correlation_descriptors(raw_grid())
    ue = model.all_sigma_zero(d)
    out = model.forward_descriptors(ue, ue) / NN_OUTPUT_SCALE_PBE.to(d)
    torch.testing.assert_close(
        out[:, [0, 23, 25, 28]],
        torch.ones_like(out[:, [0, 23, 25, 28]]),
        atol=1e-14,
        rtol=0,
    )
    assert torch.equal(out[:, 26:28], torch.zeros_like(out[:, 26:28]))
    hi = model.all_rho_inf(d)
    out_hi = model.forward_descriptors(hi, d) / NN_OUTPUT_SCALE_PBE.to(d)
    torch.testing.assert_close(
        out_hi[:, [1, 28]], torch.ones_like(out_hi[:, [1, 28]]), atol=1e-14, rtol=0
    )
    si = model.all_s_inf(d)
    assert torch.equal(si[:, 5:], d[:, 5:])
    out_si = model.forward_descriptors(si, d) / NN_OUTPUT_SCALE_PBE.to(d)
    torch.testing.assert_close(
        out_si[:, 28], torch.ones_like(out_si[:, 28]), atol=1e-14, rtol=0
    )
    actual = model(raw_grid()) / NN_OUTPUT_SCALE_PBE.to(d)
    assert ((actual[:, [22, 24]] > 0) & (actual[:, [22, 24]] < 1)).all()
    assert torch.equal(
        model.get_correlation_descriptors(
            raw_grid() * torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=d.dtype)
        )[:, 2:7],
        torch.zeros_like(d[:, 2:7]),
    )


def test_nn_full_potential_has_parameter_gradients_and_chunk_equivalence():
    energy = LapEnergy(small_model())
    f = features(0.1, n=3)
    w = torch.ones(3, dtype=f.dtype)
    target = torch.zeros(3, 2, dtype=f.dtype)
    direct = potential_loss(
        euler_components(energy, f, 0.1)["Vxc"], target, f[:, 0, :2], w
    )
    params = tuple(energy.parameters())
    dg = torch.autograd.grad(direct, params, allow_unused=True)
    bounded = full_vxc_loss(energy, f, target, w, 0.1, 1)
    bg = torch.autograd.grad(bounded, params)
    torch.testing.assert_close(bounded, direct)
    assert all(torch.isfinite(g).all() for g in bg)
    assert sum(float(g.abs().sum()) for g in bg) > 0
    for a, b in zip(dg, bg):
        torch.testing.assert_close(
            torch.zeros_like(b) if a is None else a, b, atol=1e-8, rtol=1e-7
        )


def test_checkpoint_separation_and_legacy_loading(tmp_path):
    model = small_model()
    p = tmp_path / "lap.pt"
    torch.save(checkpoint_payload(model), p)
    loaded, _ = load_lap_checkpoint(p)
    assert torch.equal(loaded(raw_grid()), model(raw_grid()))
    old = pcPBELMLOptimizerV2(2, 8).double()
    old.load_state_dict(old.state_dict())
    with pytest.raises(ValueError, match="Architecture mismatch"):
        model.load_state_dict(old.state_dict())
    torch.save({"model_state_dict": old.state_dict()}, p)
    with pytest.raises(ValueError, match="mismatch"):
        load_lap_checkpoint(p)


def test_missing_source_fails_closed_and_spin_channels_preserved():
    with pytest.raises(ValueError, match="reference spin AO"):
        validate_record({"Grid": raw_grid(), "Vrho": torch.ones(11)})
    with pytest.raises(ValueError, match="reference spin AO"):
        evaluate_stencil(torch.zeros(1, 3, dtype=torch.float64), 0.1, None)
    f = features(0.1)
    t = torch.arange(22, dtype=f.dtype).reshape(11, 2)
    assert torch.equal(normalize_target(t, f, 1, "spin-resolved"), t)
    with pytest.raises(ValueError, match="closed-shell"):
        normalize_target(t[:, 0], f, 1, "common-rks")
    eq = features(0.1, equal_spin=True)
    assert torch.equal(
        normalize_target(t[:, 0], eq, 0, "common-rks"), t[:, :1].repeat(1, 2)
    )


def test_fd_float32_cancellation_is_measured():
    energy64 = PolynomialEnergy(b=0.3, c=0.2)
    energy32 = PolynomialEnergy(b=0.3, c=0.2, dtype=torch.float32)
    h = 1e-4
    f = features(h)
    exact = 1.4 * f[:, 0, :2] - f[:, 0, 8:]
    e64 = (euler_components(energy64, f, h)["Vxc"] - exact).abs().max()
    e32 = (euler_components(energy32, f.float(), h)["Vxc"] - exact).abs().max()
    assert e64 < 1e-6
    assert e32 > 100 * e64


def test_local_nn_derivative_and_parameter_gradient_against_finite_difference():
    energy = LapEnergy(small_model())
    f = features(0.1, n=3)
    center = f[:, 0]
    _, c, _, _ = local_partials(energy, center)
    rho = center[:, :2]
    sigma = sigma_from_gradients(center[:, 2:8].reshape(-1, 2, 3))
    lap = center[:, 8:]
    step = 1e-5
    changed = torch.zeros_like(rho)
    changed[:, 0] = step
    fd = (energy(rho + changed, sigma, lap) - energy(rho - changed, sigma, lap)) / (
        2 * step
    )
    torch.testing.assert_close(c[:, 0], fd, rtol=1e-7, atol=1e-8)
    w, target = torch.ones(3, dtype=f.dtype), torch.zeros(3, 2, dtype=f.dtype)
    p = next(energy.parameters())
    loss = full_vxc_loss(energy, f, target, w, 0.1, 1)
    g = torch.autograd.grad(loss, p)[0]
    index = int(g.abs().argmax())
    baseline = p.detach().clone()
    with torch.no_grad():
        p.flatten()[index] += step
    plus = full_vxc_loss(energy, f, target, w, 0.1, 2).detach()
    with torch.no_grad():
        p.copy_(baseline)
        p.flatten()[index] -= step
    minus = full_vxc_loss(energy, f, target, w, 0.1, 2).detach()
    with torch.no_grad():
        p.copy_(baseline)
    torch.testing.assert_close(
        g.flatten()[index], (plus - minus) / (2 * step), rtol=2e-4, atol=1e-7
    )
