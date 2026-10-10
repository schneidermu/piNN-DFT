import ast
import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_b8_swa_continuation as experiment


def _five(clean):
    names = ('raw_64', 'raw_128', 'raw_192', 'raw_251', 'swa_251')
    return [{'name': name, 'clean28': value, 'order': index} for index, (name, value) in enumerate(zip(names, clean, strict=True))]


def test_swa_schedule_and_fixed_learning_rate():
    assert experiment.SWA_LOCAL == tuple(range(32, 249, 8)) + (251,)
    assert len(experiment.SWA_LOCAL) == 29
    assert experiment.swa_expected_count(31) == 0
    assert experiment.swa_expected_count(32) == 1
    assert experiment.swa_expected_count(64) == 5
    assert experiment.swa_expected_count(251) == 29
    assert 24 not in experiment.SWA_LOCAL and 256 not in experiment.SWA_LOCAL
    assert experiment.NEW_LR == experiment.LR_FACTOR * experiment.PARENT_LR
    assert experiment.NEW_LR == 0.25 * 1e-4


def test_swa_average_does_not_change_live_weights_or_buffers():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 3.0]))
            self.frozen = nn.Parameter(torch.tensor([9.0]), requires_grad=False)
            self.register_buffer('version', torch.tensor([1]))

    toy = Toy()
    parameters = {'weight': toy.weight}
    total = experiment.blank_swa(parameters)
    experiment.observe_swa(total, parameters)
    with torch.no_grad():
        toy.weight.copy_(torch.tensor([3.0, 7.0]))
    experiment.observe_swa(total, parameters)
    assert torch.equal(total['weight'] / 2, torch.tensor([2.0, 5.0], dtype=torch.float64))
    assert torch.equal(toy.weight.detach().cpu(), torch.tensor([3.0, 7.0]))
    state = experiment.swa_state(toy.state_dict(), total, 2, {'weight': torch.float32})
    assert state['weight'].dtype == torch.float32
    assert torch.equal(state['weight'], torch.tensor([2.0, 5.0]))
    assert torch.equal(state['frozen'], toy.frozen.detach().cpu())
    assert torch.equal(state['version'], toy.version.detach().cpu())
    assert torch.equal(toy.weight.detach().cpu(), torch.tensor([3.0, 7.0]))


def test_manifest_hash_and_eight_presentations():
    assert b8.sha256_file(experiment.PARENT_MANIFEST) == experiment.MANIFEST_SHA
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    assert b8.read_json(experiment.PARENT_MANIFEST) == batches
    assert b8.presentation_counts(batches) == {row['relchem']['identity']: 8 for row in rows}
    assert all(len({spec['identity'] for spec in batch['relchem']}) == 8 for batch in batches)
    assert experiment.B8_T128_CLEAN28 == b8.read_json(experiment.PARENT_VALIDATION)['clean28']


def test_optimizer_moments_survive_the_learning_rate_change():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    source = torch.optim.AdamW(
        [parameter], lr=experiment.PARENT_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    loss = parameter.pow(2).sum()
    loss.backward()
    source.step()
    saved = copy.deepcopy(source.state_dict())
    moment = saved['state'][0]['exp_avg'].clone()
    restarted = nn.Parameter(torch.tensor([4.0, 5.0]))
    optimizer = experiment._new_optimizer({'w': restarted})
    experiment.prepare_loaded_optimizer(optimizer, saved, 1, experiment.PARENT_LR)
    assert optimizer.param_groups[0]['lr'] == experiment.NEW_LR
    assert int(optimizer.state[restarted]['step']) == 1
    assert torch.equal(optimizer.state[restarted]['exp_avg'], moment)
    try:
        experiment.inspect_optimizer_state({'state': {}, 'param_groups': saved['param_groups']}, 128, experiment.PARENT_LR)
    except ValueError as error:
        assert 'unavailable' in str(error)
    else:
        raise AssertionError('missing optimizer state was accepted')


def test_checkpoint_round_trip_keeps_moments_and_swa(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, 'OUT', tmp_path)
    torch.manual_seed(5)
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.NEW_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    loss = model(torch.tensor([[1.0, -1.0]])).sum()
    loss.backward()
    optimizer.step()
    step = optimizer.state[model.weight]['step']
    if torch.is_tensor(step):
        step.fill_(experiment.PARENT_CURSOR + 32)
    else:
        optimizer.state[model.weight]['step'] = experiment.PARENT_CURSOR + 32
    total = experiment.blank_swa({'weight': model.weight})
    experiment.observe_swa(total, {'weight': model.weight})
    saved_weight = model.weight.detach().clone()
    saved_moment = optimizer.state[model.weight]['exp_avg'].detach().clone()
    saved_swa = total['weight'].clone()
    path = tmp_path / 'latest.pt'
    experiment.save_checkpoint(path, model, optimizer, 32, [], total, 1, 'protocol', 'parent')
    with torch.no_grad():
        model.weight.add_(3)
    torch.manual_seed(99)
    restored = nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    local, logs, swa_sum, swa_count = experiment.restore_checkpoint(path, restored, restored_optimizer)
    assert local == 32 and logs == [] and swa_count == 1
    assert torch.equal(restored.weight, saved_weight)
    assert torch.equal(restored_optimizer.state[restored.weight]['exp_avg'], saved_moment)
    assert int(restored_optimizer.state[restored.weight]['step']) == experiment.PARENT_CURSOR + 32
    assert restored_optimizer.param_groups[0]['lr'] == experiment.NEW_LR
    assert torch.equal(swa_sum['weight'], saved_swa)


def test_classify_is_go_only_when_gate_and_ratios_pass():
    above = _five((9.2, 9.1, 9.3, 9.4, 9.05))
    assert experiment.classify(above, {}) == 'NO-GO'
    below = _five((9.2, 9.1, 9.3, 9.4, 8.5))
    try:
        experiment.classify(below, {})
    except ValueError as error:
        assert 'scientific ratios' in str(error)
    else:
        raise AssertionError('a gate pass without ratios was classified')
    eligible = {task: 0.5 for task in b8.TASKS}
    assert experiment.classify(below, {'swa_251': eligible}) == 'GO'
    assert experiment.classify(below, {'swa_251': {**eligible, 'relchem': 1.0}}) == 'NO-GO'
    assert experiment.select_lowest(below)['name'] == 'swa_251'


def test_vector_delta_matches_cpu_and_cuda_references():
    reference = {'w': torch.tensor([1.0, 0.0])}
    current = {'w': torch.tensor([1.0, 2.0])}
    assert experiment._vector_delta(current, reference) == 2.0
    if torch.cuda.is_available():
        current = {'w': current['w'].cuda()}
        assert experiment._vector_delta(current, reference) == 2.0


def test_training_file_does_not_call_backward_step_or_to():
    source = Path(experiment.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'backward' not in calls and 'step' not in calls and 'to' not in calls
    assert 'adamw_step' in source
