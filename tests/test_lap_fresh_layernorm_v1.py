"""Small fixtures for the startup diagnostic, not another training arm."""

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lap_fresh_layernorm_v1 as audit


def test_exact_fresh_identity_and_module_ownership():
    m = audit.fresh()
    assert (
        audit.pilot.state_hash(m)
        == "359088b9713e532be0f811b4bd9f5f4f91b34d531a936db7548f46b2b410aedd"
    )
    mapping = audit.layout(m)
    assert sum(end - start for _, start, end, _, _ in mapping) == 9446
    assert next(r for r in mapping if r[0] == "x_feature_extractor.1.bias")[3]
    assert not next(r for r in mapping if r[0] == "x_feature_extractor.0.weight")[3]
    optimizer = torch.optim.AdamW(audit.names(m).values())
    assert not optimizer.state


def test_independent_ln_first_and_mixed_derivatives():
    for magnitude in (0.0, 1e-8, 0.7):
        h = (
            torch.linspace(-1, 1, 32, dtype=torch.float64) * magnitude
        ).requires_grad_()
        w = torch.ones_like(h, requires_grad=True)
        b = torch.zeros_like(h, requires_grad=True)
        probe = torch.linspace(0.2, 0.8, 32, dtype=torch.float64)
        a = F.layer_norm(h, (32,), w, b, 1e-5)
        c = audit.independent_ln(h, w, b, 1e-5)
        torch.testing.assert_close(a, c, atol=1e-10, rtol=1e-9)
        ga = torch.autograd.grad(a @ probe, (h, w, b), create_graph=True)
        gc = torch.autograd.grad(c @ probe, (h, w, b), create_graph=True)
        for x, y in zip(ga, gc):
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)
        ma = torch.autograd.grad(ga[0] @ probe, w)[0]
        mc = torch.autograd.grad(gc[0] @ probe, w)[0]
        torch.testing.assert_close(ma, mc, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(ga[2], probe)  # own bias has no 1/sqrt(eps)


def test_semantic_replicas_preserve_primal_and_mixed_gradient():
    original = audit.fresh().double()
    replica = audit.instrument(copy.deepcopy(original), audit.Observer())
    raw = torch.tensor(
        [
            [0.7, 0.4, 0.03, 0.05, 0.02, 0.0, 0.0, 0.01, -0.02],
            [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    a, b = original(raw), replica(raw)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    da = torch.autograd.grad(a.sum(), raw, create_graph=True)[0]
    db = torch.autograd.grad(b.sum(), raw, create_graph=True)[0]
    torch.testing.assert_close(da, db, atol=1e-10, rtol=1e-9)
    old_names, new_names = (
        dict(original.named_parameters()),
        dict(replica.named_parameters()),
    )
    ga = torch.autograd.grad(
        da.square().sum(), tuple(old_names.values()), allow_unused=True
    )
    gb = torch.autograd.grad(
        db.square().sum(), tuple(new_names.values()), allow_unused=True
    )
    ga, gb = dict(zip(old_names, ga)), dict(zip(new_names, gb))
    for name, _, _, ln, _ in audit.layout(original):
        if ln:
            parent, leaf = name.rsplit(".", 1)
            pieces = [
                g
                for key, g in gb.items()
                if key.startswith(parent + ".replicas.")
                and key.endswith("__" + leaf)
                and g is not None
            ]
            actual = sum(pieces, torch.zeros_like(old_names[name]))
        else:
            actual = torch.zeros_like(old_names[name]) if gb[name] is None else gb[name]
        expected = torch.zeros_like(old_names[name]) if ga[name] is None else ga[name]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


def test_state_numbering_first_rest_and_bias_correction():
    g1 = np.array([1.0, -2.0, 4.0])
    for t in (1, 2, 16, 32, 33):
        m, v = audit.first_components(g1, t)
        np.testing.assert_allclose(m, 0.1 * 0.9 ** (t - 1) * g1)
        np.testing.assert_allclose(v, 0.001 * 0.999 ** (t - 1) * g1 * g1)
        assert np.all(v >= 0)
    theta = torch.tensor([0.2, -0.1, 0.8], dtype=torch.float32)
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    g = torch.tensor(g1, dtype=torch.float32)
    actual, _mn, vn, den = audit.proposal(theta, m, v, g, 1)
    expected = theta * (1 - 2.5e-5 * 0.01) - 2.5e-5 * g / (g.abs() + 1e-8)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(theta, torch.tensor([0.2, -0.1, 0.8]))
    assert torch.equal(m, torch.zeros_like(m))
    assert torch.equal(v, torch.zeros_like(v))
    torch.testing.assert_close(den, vn.sqrt() / (1 - 0.999) ** 0.5 + 1e-8)


def test_restoration_and_no_extra_optimizer_path():
    model = audit.fresh()
    baseline = audit.pilot.state_hash(model)
    rng = audit.pilot.rng()
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.parameters()).add_(1)
    audit.pilot.restore_rng(rng)
    assert audit.pilot.state_hash(model) == baseline
    assert torch.equal(torch.get_rng_state(), rng[2])
    source = inspect.getsource(audit)
    assert source.count("optimizer.step()") == 1
    assert "for t in range(3)" in source and "if t < 2:" in source
    assert "kernel(" not in source and "Clean28" not in source
    assert "checkpoint" not in inspect.getsource(audit.energy_first_chunk)


def test_native_proposal_matches_single_tensor_adamw_without_mutation():
    p = torch.nn.Parameter(torch.tensor([0.3, -0.7]))
    optimizer = torch.optim.AdamW(
        [p], lr=2.5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    p.grad = torch.tensor([2.0, -3.0])
    optimizer.step()  # tiny synthetic fixture, not a molecular replay/intervention
    theta = p.detach().clone()
    state = optimizer.state[p]
    m, v = state["exp_avg"].clone(), state["exp_avg_sq"].clone()
    g = torch.tensor([-0.4, 0.2])
    predicted, _, _, _ = audit.proposal(theta, m, v, g, 2)
    p.grad = g
    optimizer.step()
    torch.testing.assert_close(predicted, p.detach(), atol=1e-7, rtol=1e-7)


def test_mean_three_and_deterministic_coordinate_selection():
    m = audit.fresh()
    g = np.ones(9446)
    chosen = audit.coordinate_directions(m, g)
    assert chosen == audit.coordinate_directions(m, g)
    assert chosen[0][0] != chosen[1][0]
    assert chosen[2][0].startswith("x_")
    vectors = torch.stack([torch.ones(9446) * k for k in (1.0, 2.0, 3.0)])
    torch.testing.assert_close(vectors.mean(0), torch.ones(9446) * 2)
