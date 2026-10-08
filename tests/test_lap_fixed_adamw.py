import copy

import numpy as np
import pytest
import torch

from train_models.lap_fixed_adamw import (
    TASKS,
    adamw_step,
    eligible,
    fixed_coefficients,
    select_checkpoint,
    weighted_gradient,
)


def test_fixed_median_normalization_and_floor():
    scales, weights = fixed_coefficients([[1, 2, 0, 8], [3, 4, 0, 12], [2, 3, 0, 10]], epsilon=0.1)
    assert list(scales.values()) == [2, 3, 0, 10]
    assert list(weights.values()) == [1 / 8, 1 / 12, 2.5, 1 / 40]
    with pytest.raises(ValueError):
        fixed_coefficients([[float('nan')] * 4])


def test_exact_scalarized_gradient_and_order():
    theta = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
    losses = [theta.square().sum(), theta.sum(), theta.prod(), (theta - 2).square().sum()]
    weights = dict(zip(TASKS, [0.2, 0.3, 0.4, 0.5], strict=True))
    raw = {t: {'p': torch.autograd.grad(loss, theta, retain_graph=True)[0]} for t, loss in zip(TASKS, losses, strict=True)}
    expected = torch.autograd.grad(sum(weights[t] * loss for t, loss in zip(TASKS, losses, strict=True)), theta)[0]
    torch.testing.assert_close(weighted_gradient(raw, weights)['p'], expected, rtol=1e-15, atol=1e-15)
    with pytest.raises(ValueError):
        weighted_gradient(dict(reversed(list(raw.items()))), weights)


def test_native_adamw_no_pareto_gate_and_resume_exact():
    model = torch.nn.Linear(2, 1, bias=False)
    reference = copy.deepcopy(model)
    weights = dict.fromkeys(TASKS, 0.25)
    raw = {t: {'weight': torch.tensor([[1., -2.]], dtype=torch.float64) * sign}
           for t, sign in zip(TASKS, [1, 1, 1, -1], strict=True)}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    control = torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=False)
    reference.weight.grad = weighted_gradient(raw, weights)['weight'].float()
    control.step()
    adamw_step(model, opt, raw, weights)
    assert torch.equal(model.weight, reference.weight)
    # One task opposes the joint direction; the ordinary optimizer still updates.
    assert all(p.grad is None for p in model.parameters())
    resumed = copy.deepcopy(model)
    restored = torch.optim.AdamW(resumed.parameters(), lr=1e-4, foreach=False)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    adamw_step(model, opt, raw, weights)
    adamw_step(resumed, restored, raw, weights)
    assert torch.equal(model.weight, resumed.weight)


def test_four_task_selection_and_validation_only_after_eligibility():
    baseline = dict.fromkeys(TASKS, 1.)
    bad = {'cursor': 1, 'objectives': {**baseline, 'op': 1.001}, 'validation': {'clean28': 1.}}
    good = {'cursor': 2, 'objectives': dict.fromkeys(TASKS, 0.99), 'validation': {'clean28': 6.4}}
    better = {'cursor': 3, 'objectives': dict.fromkeys(TASKS, 0.98), 'validation': {'clean28': 6.3}}
    assert not eligible(baseline, baseline)
    assert select_checkpoint([bad], baseline) is None
    assert select_checkpoint([bad, good, better], baseline) is better
    assert not eligible(dict.fromkeys(TASKS, np.nan), baseline)
