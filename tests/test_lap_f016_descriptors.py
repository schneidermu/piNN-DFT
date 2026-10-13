import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lap_f016_descriptors import DiagnosticEnergy, DiagnosticQModel, descriptors
from NN_models_lap import pcPBELMLOptimizerV2Lap

from train_models.lap_operator import assemble_rks_operator
from train_models.lap_vxc import LapEnergy


def raw():
    g = torch.Generator().manual_seed(1601)
    x = torch.randn(13, 9, generator=g, dtype=torch.float64)
    x[:, :5] = x[:, :5].abs() + .1
    return x


@pytest.mark.parametrize('representation', ['R1', 'R2'])
def test_expanded_anchor_actions(representation):
    source = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8).double().eval()
    clone = DiagnosticQModel(source, representation)
    d = descriptors(raw(), representation)
    zero = clone.all_sigma_zero(d)
    assert torch.count_nonzero(zero[:, 2:7]) == 0
    if representation == 'R2':
        assert torch.count_nonzero(zero[:, 8:]) == 0
    torch.testing.assert_close(clone.all_rho_inf(d)[:, 2:], d[:, 2:])
    torch.testing.assert_close(clone.all_s_inf(d)[:, 5:], d[:, 5:])
    torch.testing.assert_close(clone.swap_descriptors(clone.swap_descriptors(d)), d)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_zero_weight_embedding_forward_and_input_gradient(dtype):
    source = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8).to(dtype).eval()
    clone = DiagnosticQModel(source, 'R2')
    x = raw().to(dtype).requires_grad_(True)
    original, embedded = source(x), clone(x)
    tolerance = 1e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(original, embedded, atol=tolerance, rtol=tolerance)
    a = torch.autograd.grad(original.sum(), x)[0]
    b = torch.autograd.grad(embedded.sum(), x)[0]
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(b[:, 5:7]) == 0


def test_zero_weight_operator_embedding():
    source = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8).double().eval()
    clone = DiagnosticQModel(source, 'R2')
    g = torch.Generator().manual_seed(1602)
    f = torch.randn(7, 10, generator=g, dtype=torch.float64)
    f[:, :2] = f[:, :2].abs() + 1
    w = torch.ones(7, dtype=torch.float64)
    p = torch.randn(7, 3, generator=g, dtype=torch.float64)
    dp = torch.randn(7, 3, 3, generator=g, dtype=torch.float64)
    lp = torch.randn(7, 3, generator=g, dtype=torch.float64)
    a = assemble_rks_operator(LapEnergy(source), f, w, p, dp, lp)
    b = assemble_rks_operator(DiagnosticEnergy(clone), f, w, p, dp, lp)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_mixed_physical_partials_are_symmetric():
    source = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8).double().eval()
    model = DiagnosticQModel(source, 'R2')
    x = raw()[:1].requires_grad_(True)
    value = model(x).square().sum()
    first = torch.autograd.grad(value, x, create_graph=True)[0]
    rho_lap = torch.autograd.grad(first[0, 0], x, retain_graph=True)[0][0, 7]
    lap_rho = torch.autograd.grad(first[0, 7], x)[0][0, 0]
    torch.testing.assert_close(rho_lap, lap_rho, atol=1e-10, rtol=1e-10)


def test_embedding_original_parameter_gradient_mapping():
    source = pcPBELMLOptimizerV2Lap(num_layers=2, h_dim=8).double().eval()
    clone = DiagnosticQModel(source, 'R2')
    x = raw()
    original = torch.autograd.grad(source(x).square().sum(), tuple(source.parameters()))
    expanded = torch.autograd.grad(clone(x).square().sum(), tuple(clone.parameters()))
    for ((name, _), a, b) in zip(source.named_parameters(), original, expanded, strict=True):
        if name == 'c_input_layers.0.weight':
            b = b[:, :8]
        elif name == 'x_feature_extractor.0.weight':
            b = b[:, :2]
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
