import copy
import json
from types import SimpleNamespace

import numpy as np
import torch

import train_lap_microbatch as run
from tools.adamw_lr_sweep import (
    probe_manifest,
    ranking_key,
    screen_eligibility,
    uncertainty,
)


def test_probe_population_weights_and_independent_variant_ids():
    rows = {}
    for db, count in [('small', 4), ('big', 247), ('AE17', 17)]:
        for i in range(count):
            key = f'{db}:{i}'
            rows[key] = {'id': key, 'task': 'ae17' if db == 'AE17' else 'relchem',
                         'database': db, 'variants': {str(v): {} for v in range(8)}}
    bundle = SimpleNamespace(reactions=rows, systems={f's{i}': {'n_grid': i} for i in range(90)})
    panel = probe_manifest(bundle)
    assert panel == probe_manifest(bundle)
    assert all(abs(sum(r['weight'] for r in panel[t]) - 1) < 1e-15 for t in panel)
    assert abs(sum(r['weight'] for r in panel['relchem'] if r['stratum'] == 'small') - 4 / 251) < 1e-15
    assert len({r['identity'] for r in panel['mrks']}) == 8


def test_selection_does_not_require_per_step_pareto_descent():
    logs = [{'task_progress_actual_step': [-.1, .2, .1, .3]} for _ in range(10)]
    ratios = dict(zip(run.TASKS, [.98, 1.02, .9, .95]))
    allowed, destructive = screen_eligibility(ratios, logs)
    assert allowed and not any(destructive.values())
    bad = {**ratios, 'relchem': 1.11}
    assert not screen_eligibility(bad, logs)[0]
    a = {'lr': 1e-5, 'probe_ratios': ratios}
    b = {'lr': 1e-3, 'probe_ratios': {**ratios, 'relchem': .99}}
    assert ranking_key(a) < ranking_key(b)


def test_paired_probe_uncertainty_identical_losses_gives_unit_ratio():
    receipt = {'rows': {t + str(i): {'task': t, 'stratum': 's', 'weight': .5, 'loss': float(i + 1)}
                       for t in run.TASKS for i in range(2)}}
    assert all(x == [1., 1.] for x in uncertainty(receipt, receipt).values())


def test_constant_lr_real_native_adamw_resume_and_raw_vector_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    data = tmp_path / 'data'
    (data / 'mrks').mkdir(parents=True)
    run.write(data / 'mrks/dispersion.json', {})
    monkeypatch.setattr(run, 'DATA', data)
    class Bundle:
        def __init__(self, *args):
            self.manifest = {'logical_sha256': run.DATA_SHA}
        def chemistry_dispersions(self):
            return {}
        def close(self):
            pass
    monkeypatch.setattr(run, 'PublicationDataset', Bundle)
    def model_at(state):
        model = torch.nn.Linear(2, 1)
        return model, copy.deepcopy(model).double()
    monkeypatch.setattr(run, 'model_at', model_at)
    def measure(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk):
        parameters = run.existing.named_trainable_parameters(model)
        raw = {t: {n: (p.detach().double() + j + 1) for n, p in parameters.items()}
               for j, t in enumerate(run.TASKS)}
        return {'sample': entry, 'norms': {t: float(torch.cat([g.flatten() for g in raw[t].values()]).norm()) for t in run.TASKS},
                'losses': {t: 1. for t in run.TASKS}, 'seconds': {'tasks': .01},
                'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}, raw
    monkeypatch.setattr(run, 'measure', measure)
    def folder(name):
        p = tmp_path / name
        p.mkdir()
        run.write(p / 'sampling_manifest.json', [{'cursor': i} for i in range(3)])
        run.write(p / 'calibration.json', {'lambda': dict.fromkeys(run.TASKS, .25),
            'manifest_sha256': run.file_sha256(p / 'sampling_manifest.json')})
        run.write(p / 'protocol.json', {'dataset_sha256': run.DATA_SHA, 'initial_state': {}, 'exc_chunk_size': 4096,
            'adamw': {'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01, 'foreach': False}})
        return p
    resumed, whole = folder('resumed'), folder('whole')
    run.train(resumed, 3, stop_at=1, learning_rate=1e-4, constant_lr=True, diagnostics=True)
    run.train(resumed, 3, learning_rate=1e-4, constant_lr=True, diagnostics=True)
    run.train(whole, 3, learning_rate=1e-4, constant_lr=True, diagnostics=True)
    a = torch.load(resumed / 'ordinary_sgd_adamw/latest.pt', weights_only=False)
    b = torch.load(whole / 'ordinary_sgd_adamw/latest.pt', weights_only=False)
    assert a['scheduler'] is None and a['cursor'] == 3
    assert run.existing.equal(a['model'], b['model']) and run.existing.equal(a['optimizer'], b['optimizer'])
    assert run.existing.equal(a['rng'], b['rng'])
    assert [r['learning_rate'] for r in a['logs']] == [1e-4] * 3
    assert [r['moment_step'] for r in a['logs']] == [1, 2, 3]
    assert all(r['step_norm'] > 0 and r['adamw_moments']['exp_avg_sq']['norm'] > 0 for r in a['logs'])
    path = resumed / 'ordinary_sgd_adamw/raw_gradients/update_000.npz'
    assert run.file_sha256(path) == a['logs'][0]['raw_gradients_sha256']
    with np.load(path) as arrays:
        assert arrays.files == list(run.TASKS) and all(arrays[t].dtype == np.float64 for t in run.TASKS)
    protocol = json.loads((resumed / 'ordinary_sgd_adamw/protocol.json').read_text())
    assert protocol['scheduler'] == 'constant' and protocol['lr'] == 1e-4
