import ast
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8

J251 = ROOT / 'relchem_joint_epoch_sampling_manifest.json'


def _rows():
    return b8.read_json(J251)


def test_json_writer_keeps_lf_line_endings(tmp_path):
    path = tmp_path / 'sample.json'
    b8.write_json(path, {'cursor': 1, 'name': 'b8'})
    raw = path.read_bytes()
    assert b'\r\n' not in raw
    assert raw.endswith(b'\n')


def test_j251_manifest_hash_and_batch_properties():
    assert b8.sha256_file(J251) == b8.J251_MANIFEST_SHA
    rows = _rows()
    batches = b8.build_batches(rows)
    properties = b8.assert_batch_properties(batches, rows)
    assert properties['presentations'] == 2008
    assert properties['presentations_each'] == 8
    assert b8.presentation_counts(batches) == {row['relchem']['identity']: 8 for row in rows}
    assert b8.dump(batches) == b8.dump(b8.build_batches(rows))
    b8.assert_split_disjoint(batches)


def test_manifest_rejects_a_changed_identity_and_a_short_schedule():
    rows = _rows()
    batches = b8.build_batches(rows)
    batches[3]['relchem'][2] = dict(batches[3]['relchem'][2], identity='reaction_not_in_manifest')
    try:
        b8.assert_batch_properties(batches, rows)
    except ValueError as error:
        assert 'cyclic' in str(error) or 'eight' in str(error)
    else:
        raise AssertionError('changed identity was accepted')
    try:
        b8.build_batches(rows[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('short manifest was accepted')


def test_mean_divides_by_eight_and_singleton_does_not():
    pairs = []
    originals = []
    for index in range(1, 9):
        grad = {
            'w': torch.tensor([float(index)], dtype=torch.float64),
            'v': torch.tensor([float(index), 0.0], dtype=torch.float64),
        }
        originals.append({name: piece.clone() for name, piece in grad.items()})
        pairs.append((float(index), grad))
    loss, mean = b8.combine_gradients(pairs)
    assert loss == sum(range(1, 9)) / 8
    assert torch.equal(mean['w'], torch.tensor([4.5], dtype=torch.float64))
    assert torch.equal(mean['v'], torch.tensor([4.5, 0.0], dtype=torch.float64))
    for original, (_, grad) in zip(originals, pairs, strict=True):
        for name in original:
            assert torch.equal(grad[name], original[name])
    single_loss, single = b8.combine_gradients(pairs[:1])
    assert single_loss == 1.0
    assert torch.equal(single['w'], torch.tensor([1.0], dtype=torch.float64))


def test_unused_parameter_is_zero_and_duplicates_are_rejected():
    template = {'v': torch.zeros(2, dtype=torch.float64)}
    pairs = [(float(index), {'w': torch.tensor([float(index)], dtype=torch.float64)}) for index in range(1, 9)]
    loss, mean = b8.combine_gradients(pairs, names=('w', 'v'), zero_templates=template)
    assert loss == 4.5
    assert torch.equal(mean['v'], torch.zeros(2, dtype=torch.float64))
    assert list(mean) == ['w', 'v']
    try:
        b8.combine_gradients(pairs, names=('w', 'w'), zero_templates=template)
    except ValueError as error:
        assert 'duplicate' in str(error)
    else:
        raise AssertionError('duplicate parameter name was accepted')


def test_f32_and_reordered_parameters_are_rejected():
    bad = [(1.0, {'w': torch.tensor([1.0], dtype=torch.float32)})]
    try:
        b8.combine_gradients(bad)
    except ValueError as error:
        assert 'F64' in str(error)
    else:
        raise AssertionError('F32 gradient was accepted')
    first = {'w': torch.zeros(1, dtype=torch.float64), 'v': torch.zeros(1, dtype=torch.float64)}
    second = {'v': torch.zeros(1, dtype=torch.float64), 'w': torch.zeros(1, dtype=torch.float64)}
    try:
        b8.combine_gradients([(0.0, first), (0.0, second)])
    except ValueError as error:
        assert 'order' in str(error)
    else:
        raise AssertionError('reordered parameters were accepted')


def test_coefficients_are_applied_once_and_do_not_mutate_other_tasks():
    from train_models.lap_fixed_adamw import TASKS
    raw = {task: {'p': torch.ones(1, dtype=torch.float64)} for task in TASKS}
    before = raw['ae17']['p'].clone()
    joint = b8.scalarize(raw, b8.LAMBDAS)
    expected = torch.zeros(1, dtype=torch.float64)
    for task in TASKS:
        expected = expected + raw[task]['p'] * b8.LAMBDAS[task]
    assert torch.equal(joint['p'], expected)
    assert torch.equal(raw['ae17']['p'], before)
    assert tuple(b8.LAMBDAS) == TASKS
    doubled = {task: value * 2 for task, value in b8.LAMBDAS.items()}
    assert not torch.equal(joint['p'], b8.scalarize(raw, doubled)['p'])


def test_checkpoint_round_trip_has_no_extra_update():
    original_out = b8.OUT
    try:
        storage = Path(tempfile.mkdtemp())
        b8.OUT = storage
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
        source = torch.tensor([[1.0, -1.0]])
        loss = model(source).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_weight = model.weight.detach().clone()
        saved_state = {key: tensor.detach().clone() for key, tensor in optimizer.state[model.weight].items() if torch.is_tensor(tensor)}
        path = storage / 'latest.pt'
        b8.save_checkpoint(path, model, optimizer, 7, 'manifest', 'protocol', [{'cursor': 7}])
        sentinel = torch.rand(3).clone()
        with torch.no_grad():
            model.weight.add_(4)
        torch.manual_seed(999)
        restored = nn.Linear(2, 1, bias=False)
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
        cursor, logs = b8.restore_checkpoint(path, restored, restored_optimizer, 'manifest', 'protocol')
        assert cursor == 7 and logs == [{'cursor': 7}]
        assert torch.equal(restored.weight, saved_weight)
        assert torch.equal(torch.rand(3), sentinel)
        for key, tensor in saved_state.items():
            assert torch.equal(restored_optimizer.state[restored.weight][key], tensor)
        steps = int(restored_optimizer.state[restored.weight]['step'])
        assert steps == 1
    finally:
        b8.OUT = original_out
        shutil.rmtree(storage, ignore_errors=True)


def test_optimizer_boundary_is_only_in_adamw_step():
    assert b8._ast_guard() == []
    source = (ROOT / 'train_models' / 'lap_fixed_adamw.py').read_text(encoding='utf-8')
    assert source.count('.to(p.dtype)') == 1
    experiment = Path(b8.__file__).read_text(encoding='utf-8')
    tree = ast.parse(experiment)
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert 'backward' not in calls and 'step' not in calls and 'to' not in calls


def test_classify_branches():
    clean = {cursor: 9.0 for cursor in b8.CLEAN_MILESTONES}
    assert b8.classify({'complete': True, 'clean28': clean, 'diagnostic_complete': True, 'science': {}}) == 'NO-GO'
    assert b8.classify({'complete': False, 'clean28': clean, 'diagnostic_complete': False, 'science': {}}) == 'PARTIAL'
    passing = dict(clean)
    passing[64] = 8.0
    science = {'64': {'status': 'complete', 'ratios': {task: 0.5 for task in b8.TASKS}}}
    assert b8.classify({'complete': True, 'clean28': passing, 'science': science}) == 'STRONG PROGRESS'
    passing[64] = 6.5
    assert b8.classify({'complete': True, 'clean28': passing, 'science': science}) == 'BREAKTHROUGH'
    failed = {'64': {'status': 'complete', 'ratios': {'relchem': 1.1}, 'stopped_at': 'relchem'}}
    passing[64] = 8.0
    assert b8.classify({'complete': True, 'clean28': passing, 'science': failed}) == 'ACCURACY-ONLY PROGRESS'
    missing = {'64': {'status': 'incomplete'}}
    assert b8.classify({'complete': True, 'clean28': passing, 'science': missing}) == 'PARTIAL'


def test_frozen_controls_match_receipts():
    metrics = b8.read_json(ROOT / 'relchem_joint_epoch_metrics.json')
    assert metrics['arms']['J']['ratios'] == b8.J251_RATIOS
    calibration = b8.read_json(ROOT.parent / 'lap_relchem_joint_epoch_20261009' / 'J' / 'calibration.json')
    assert b8.sha256_file(ROOT.parent / 'lap_relchem_joint_epoch_20261009' / 'J' / 'calibration.json') == b8.CALIBRATION_SHA
    assert calibration['lambda'] == b8.LAMBDAS
    j251_validation = b8.read_json(ROOT.parent / 'lap_relchem_joint_epoch_20261009' / 'J' / 'endpoint_251_validation.json')
    p536_validation = b8.read_json(ROOT.parent / 'lap_relchem_joint_epoch_20261009' / 'R' / 'endpoint_0_validation.json')
    assert j251_validation['metrics']['clean28'] == b8.J251_CLEAN28
    assert p536_validation['metrics']['clean28'] == b8.P536_CLEAN28
    historical = b8.read_json(ROOT / 'iid_adamw_lr_stabilization_metrics.json')
    assert historical['selected_candidate']['clean28'] == b8.HISTORICAL_BEST_CLEAN28
