import copy

import numpy as np
import pytest
import torch

from tools import stabilize_iid_adamw as branch


def test_lr_only_restore_preserves_moments_rng_and_actual_adamw_scaling(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False, weight_decay=.01)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state['step'].fill_(70)
    before = copy.deepcopy(optimizer.state_dict())
    calibration = {'lambda': dict.fromkeys(branch.run.TASKS, .25)}
    path = tmp_path / 'checkpoint70.pt'
    branch.run.save(path, model, optimizer, 70, 'manifest', calibration, total_updates=90)
    expected_rng = branch.run.existing.capture_rng_state()
    outputs, moments = [], []
    for lr in (1e-4, 3e-5, 1e-5):
        current = copy.deepcopy(model)
        opt = torch.optim.AdamW(current.parameters(), lr=.1)
        np.random.random(3)
        assert branch.restore_lr(branch.run.restore, lr, path, current, opt, 'manifest', calibration) == 70
        expected = copy.deepcopy(before)
        for group in expected['param_groups']:
            group['lr'] = lr
        assert branch.run.existing.equal(opt.state_dict(), expected)
        assert branch.run.existing.equal(branch.run.existing.capture_rng_state(), expected_rng)
        old = torch.cat([p.detach().flatten() for p in current.parameters()])
        for parameter in current.parameters():
            parameter.grad = torch.ones_like(parameter)
        opt.step()
        outputs.append(torch.cat([p.detach().flatten() for p in current.parameters()]) - old)
        moments.append(copy.deepcopy(opt.state_dict()['state']))
    assert branch.run.existing.equal(moments[0], moments[1])
    assert branch.run.existing.equal(moments[0], moments[2])
    for index, ratio in ((1, .3), (2, .1)):
        assert abs(float(outputs[index].norm() / outputs[0].norm()) - ratio) < 5e-4
        assert float(torch.nn.functional.cosine_similarity(outputs[index], outputs[0], dim=0)) > .9999


def test_control_cannot_be_retrained():
    with pytest.raises(AssertionError):
        branch.train('A')


def test_unexpected_loaded_lr_fails_without_modifying_moments():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    with pytest.raises(AssertionError):
        branch.restore_lr(lambda *a, **k: 70, 3e-5, None, model, optimizer)
    assert optimizer.param_groups[0]['lr'] == 2e-4
