"""Focused training cadence, restoration and full-matrix reduction fixtures."""

import copy

import numpy as np
import torch

from tools import lap_fresh_grid_pilot_v1 as pilot


def test_independent_copies_and_fresh_native_adamw():
    one = torch.nn.Linear(2, 2)
    a, b = copy.deepcopy(one), copy.deepcopy(one)
    assert pilot.state_hash(a) == pilot.state_hash(b)
    assert all(
        x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters())
    )
    opt = torch.optim.AdamW(a.parameters(), lr=2.5e-5, foreach=False)
    assert not opt.state
    opt.zero_grad(set_to_none=True)
    for p in a.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert all(int(s["step"]) == 1 for s in opt.state.values())


def test_full_matrix_loss_precedes_square_and_mean():
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    target = torch.eye(2, dtype=torch.float64)
    chunks = [
        parameter * torch.ones(2, 2, dtype=torch.float64),
        parameter * torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)),
    ]
    total = sum(chunks)
    loss = ((total - target) ** 2).sum() / 2
    wrong = sum(((chunk - target / 2) ** 2).sum() / 2 for chunk in chunks)
    assert not torch.isclose(loss, wrong)
    g = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    upstream = 2 * (total.detach() - target) / 2
    partials = [
        torch.autograd.grad((v * upstream).sum(), parameter, retain_graph=True)[0]
        for v in chunks
    ]
    assert torch.allclose(g, sum(partials), atol=1e-14, rtol=0)


def test_mean_of_three_gradients_and_no_stale_accumulation():
    p = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    losses = [(p - target) ** 2 for target in [1.0, 2.0, 3.0]]
    individual = [torch.autograd.grad(l, p, retain_graph=True)[0] for l in losses]
    joint = torch.autograd.grad(sum(losses) / 3, p)[0]
    assert torch.allclose(joint, torch.stack(individual).mean())
    assert p.grad is None
    anew = torch.autograd.grad((p - 4.0) ** 2, p)[0]
    assert torch.allclose(anew, 2 * (p - 4))


def test_mapped_selection_preserves_source_precision_and_multiplicity():
    source = np.array([0.123456789123, 0.9, 0.7], dtype=np.float64)
    mapping = np.array([2, 0, 2])
    unique, inverse = np.unique(mapping, return_inverse=True)
    assert np.array_equal(source[unique][inverse], source[mapping])
    assert source[mapping].dtype == np.float64
    assert len(source[mapping]) == 3


def test_safe_resume_of_new_fixture_only():
    saved = pilot.rng()
    a = torch.nn.Linear(2, 1)
    models = {"SUBSET": a, "FULL_SOURCE": copy.deepcopy(a)}
    opts = {
        k: torch.optim.AdamW(m.parameters(), lr=2.5e-5, foreach=False)
        for k, m in models.items()
    }
    counters = {k: 1 for k in models}
    for k, m in models.items():
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
    expected = {k: pilot.state_hash(m) for k, m in models.items()}
    path = pilot.OUT / "unit_fixture_resume.pt"
    receipt = pilot.checkpoint(path, models, opts, counters)
    for m in models.values():
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1)
    restored = pilot.restore_checkpoint(receipt, models, opts)
    assert restored == counters
    assert all(pilot.state_hash(models[k]) == expected[k] for k in models)
    assert all(int(s["step"]) == 1 for opt in opts.values() for s in opt.state.values())
    path.unlink()
    pilot.restore_rng(saved)


def test_only_new_hash_bound_checkpoint_loader_and_no_scf():
    source = __import__("inspect").getsource(pilot)
    assert ".kernel(" not in source
    assert "B8" not in source and "P536" not in source
    assert "path.parent.resolve() == OUT.resolve()" in source
