import copy
import itertools

import pytest
import torch

from tools import relchem_joint_epoch as e
from train_models.lap_chemistry_sampling import (
    epoch_samples,
    fixed_evaluation,
    validate_evaluation,
)


def population():
    return {f'{task}:{i}': {'task': task, 'database': task, 'reaction_id': i,
                               'variants': {str(v): {} for v in range(8)}}
            for task, n in (('relchem', 251), ('ae17', 17)) for i in range(n)}


def test_epoch_and_independent_fixed_evaluation():
    rows = population()
    a = epoch_samples(rows, [str(i) for i in range(90)])
    assert a == epoch_samples(rows, [str(i) for i in range(90)])
    assert len(a) == len({s['relchem']['identity'] for s in a}) == 251
    assert a != epoch_samples(rows, [str(i) for i in range(90)], epoch=1)
    assert len({s['mrks_id'] for s in a[:90]}) == 90
    fixed = fixed_evaluation(rows)
    assert fixed == fixed_evaluation(dict(reversed(list(rows.items()))))
    assert len(fixed['rows']) == 268
    assert fixed['seed'] != 202610092
    assert len({r['identity'] for r in fixed['rows']}) == 268
    assert all(s['relchem']['weight'] == 1.0 for s in a)


def test_exhaustive_or_incomplete_evaluation_fails_closed():
    rows = population()
    fixed = fixed_evaluation(rows)['rows']
    with pytest.raises(ValueError):
        validate_evaluation(fixed + [dict(fixed[0], variant='7')], rows)
    with pytest.raises(ValueError):
        validate_evaluation(fixed[:-1], rows)


def test_relchem_only_native_boundary_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    names = e.run.existing.named_trainable_parameters(model)
    raw = {n: torch.ones_like(p, dtype=torch.float64) for n, p in names.items()}
    initial = copy.deepcopy(model.state_dict())
    other = copy.deepcopy(model)
    reference = torch.optim.AdamW(other.parameters(), lr=1e-4, foreach=False)
    coefficient = 0.017015480965588553
    joint = e.relchem_step(model, optimizer, raw, coefficient)
    assert all(g.dtype == torch.float64 for g in joint.values())
    for p in other.parameters():
        p.grad = (torch.ones_like(p, dtype=torch.float64) * coefficient).float()
    reference.step()
    assert e.run.existing.equal(model.state_dict(), other.state_dict())
    checkpoint = tmp_path / 'checkpoint.pt'
    e.run.save(checkpoint, model, optimizer, 1, 'manifest', {'lambda': coefficient})
    e.relchem_step(model, optimizer, raw, coefficient)
    expected, moments = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    assert e.run.restore(checkpoint, model, optimizer, 'manifest', {'lambda': coefficient}) == 1
    e.relchem_step(model, optimizer, raw, coefficient)
    assert e.run.existing.equal(model.state_dict(), expected)
    assert e.run.existing.equal(optimizer.state_dict(), moments)
    assert not e.run.existing.equal(model.state_dict(), initial)


def test_joint_exact_f64_sum():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    raw = {t: {n: torch.full_like(p, j + 1, dtype=torch.float64)
               for n, p in e.run.existing.named_trainable_parameters(model).items()}
           for j, t in enumerate(e.run.TASKS)}
    coefficients = dict(zip(e.run.TASKS, [0.017, 0.00005, 0.000015, 0.336], strict=True))
    joint = e.run.adamw_step(model, optimizer, raw, coefficients)
    assert all(torch.equal(g, sum(raw[t][n] * coefficients[t] for t in e.run.TASKS))
               for n, g in joint.items())


def test_real_relchem_branch_excludes_joint_tasks_and_resumes(tmp_path, monkeypatch):
    """Exercise the actual arm loop with cheap objectives and a safe runtime pause."""
    monkeypatch.setattr(e, 'OUT', tmp_path)
    monkeypatch.setattr(e, 'verify', lambda folder: {'initial_state': {}, 'lr': 1e-4,
        'adamw': {'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01, 'foreach': False},
        'sampling_manifest_sha256': 'manifest', 'dataset_sha256': e.run.DATA_SHA})
    monkeypatch.setattr(e.time, 'perf_counter', lambda: next(clock))
    clock = itertools.count()
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    initial = torch.nn.Linear(2, 1)
    monkeypatch.setattr(e.run, 'model_at', lambda state: (copy.deepcopy(initial), copy.deepcopy(initial).double()))
    monkeypatch.setattr(e.run.existing.lap_training, 'tensor_record', lambda r, device, dtype: r)
    def forbidden(*args, **kwargs):
        raise AssertionError('R must not compute any joint task gradient')
    monkeypatch.setattr(e.run, 'measure', forbidden)
    class Bundle:
        def __init__(self, root):
            self.manifest = {'logical_sha256': e.run.DATA_SHA}
        def chemistry_dispersions(self):
            return {}
        def chemistry(self, split):
            assert split == 'train_relchem'
            return self
        def load_variant(self, identity, variant):
            return {'Grid': torch.zeros(1, 9)}
        def close(self):
            pass
    class Objective:
        def __init__(self, model):
            self.model = model
        def value_and_grad(self):
            return 1.0, {n: torch.ones_like(p, dtype=torch.float64)
                         for n, p in e.run.existing.named_trainable_parameters(self.model).items()}
    monkeypatch.setattr(e.run, 'PublicationDataset', Bundle)
    monkeypatch.setattr(e.run, 'chemistry', lambda model, *args: Objective(model))
    folder = tmp_path / 'R'
    folder.mkdir()
    e.run.write(tmp_path / 'preflight.json', {})
    e.run.write(folder / 'protocol.json', {})
    e.run.write(folder / 'calibration.json', {'lambda': {'relchem': 0.017}})
    e.run.write(folder / 'sampling_manifest.json', [{'relchem': {'identity': str(i), 'variant': 'v'}} for i in range(251)])
    e.train('R', seconds=3)
    path = folder / 'ordinary_sgd_adamw/latest.pt'
    saved = torch.load(path, weights_only=False)
    assert saved['cursor'] == 1 and saved['epoch_cursor'] == 1
    e.train('R', seconds=3)
    resumed = torch.load(path, weights_only=False)
    assert resumed['cursor'] == 2
    assert all(set(log['losses']) == {'relchem'} for log in resumed['logs'])
    reference = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(e.run.existing.named_trainable_parameters(reference).values(),
                                  lr=1e-4, foreach=False)
    for _ in range(2):
        raw = {n: torch.ones_like(p, dtype=torch.float64)
               for n, p in e.run.existing.named_trainable_parameters(reference).items()}
        e.relchem_step(reference, optimizer, raw, 0.017)
    assert e.run.existing.equal(resumed['model'], reference.state_dict())
    assert e.run.existing.equal(resumed['optimizer'], optimizer.state_dict())
