import copy
from fractions import Fraction

import pytest
import torch

from tools import lap_dbstrat_importance as experiment


def population():
    rows = []
    for database, count in experiment.EXPECTED_COUNTS.items():
        for index in range(count):
            rows.append({'id': f'{database}-{index:03d}', 'database': database, 'reaction_id': index,
                         'task': 'relchem', 'variants': [f'v{k}' for k in range(8)]})
    return rows


def control_rows():
    rows = []
    for cursor in range(experiment.UPDATES):
        rows.append({'cursor': cursor, 'relchem': {'identity': 'iid', 'database': 'DBH76', 'reaction_id': 0,
                                                    'variant': 'v0', 'weight': 1.0},
                     'ae17': {'identity': f'ae-{cursor % 17}', 'database': 'AE17', 'reaction_id': cursor % 17,
                              'variant': 'v1', 'weight': 1.0},
                     'mrks_id': f'mrks-{cursor}'})
    return rows


def reactions_for(manifest, relchem_rows):
    reactions = {row['id']: row for row in relchem_rows}
    for row in manifest:
        identity = row['ae17']['identity']
        reactions[identity] = {'id': identity, 'database': 'AE17', 'task': 'ae17', 'variants': ['v1'],
                               'reaction_id': row['ae17']['reaction_id']}
    return reactions


def test_exact_importance_identity_over_all_identities():
    gradients = []
    for database, count in experiment.EXPECTED_COUNTS.items():
        for index in range(count):
            gradients.append((database, Fraction(index + 1, 3)))
    total = experiment.expectation_identity(gradients)
    probability = Fraction(1, experiment.POPULATION)
    assert total == sum(probability * gradient for _database, gradient in gradients)
    for database, count in experiment.EXPECTED_COUNTS.items():
        weight = experiment.importance_fraction(count)
        assert Fraction(1, 8 * count) * weight == Fraction(1, 251)


def test_stratified_manifest_keeps_ae17_mrks_and_reweights_relchem():
    rows = population()
    control = control_rows()
    manifest = experiment.build_manifest(rows, control)
    systems = {row['mrks_id'] for row in control}
    experiment.validate_manifest(manifest, reactions_for(manifest, rows), systems, control)
    assert experiment.build_manifest(rows, control) == manifest
    assert len({row['relchem']['identity'] for row in manifest}) <= 90
    for row, old in zip(manifest, control, strict=True):
        assert row['ae17'] == old['ae17'] and row['mrks_id'] == old['mrks_id']
        assert row['relchem']['weight'] == 1.0
        assert row['relchem']['importance_denominator'] == 251
        database_n = experiment.EXPECTED_COUNTS[row['relchem']['database']]
        assert row['relchem']['importance_numerator'] == 8 * database_n


def test_correction_uses_ratio_once_and_ignores_frequency_weight():
    entry = {'relchem': {'weight': 1.0, 'database_n': 4, 'importance_numerator': 32,
                         'importance_denominator': 251}}
    base = torch.tensor([2.0, -4.0], dtype=torch.float64)
    raw = {task: {'p': base * (index + 1)} for index, task in enumerate(experiment.TASKS)}
    originals = {task: raw[task]['p'].clone() for task in experiment.TASKS}
    updated, info = experiment.apply_importance(raw, entry, experiment.LAMBDAS)
    weight = torch.tensor(32, dtype=torch.float64) / torch.tensor(251, dtype=torch.float64)
    torch.testing.assert_close(updated['relchem']['p'], originals['relchem'] * weight)
    for task in ('ae17', 'exc', 'op'):
        assert updated[task]['p'].data_ptr() == raw[task]['p'].data_ptr()
        torch.testing.assert_close(updated[task]['p'], originals[task])
    assert info['importance_weight'] == pytest.approx(32 / 251)
    rejected = copy.deepcopy(entry)
    rejected['relchem']['weight'] = 7.0
    with pytest.raises(ValueError, match='frequency weight'):
        experiment.apply_importance(raw, rejected, experiment.LAMBDAS)


def test_measure_wrapper_applies_correction_once():
    entry = {'relchem': {'weight': 1.0, 'database_n': 104, 'importance_numerator': 832,
                         'importance_denominator': 251}}

    def native(*_args, **_kwargs):
        raw = {task: {'p': torch.ones(2, dtype=torch.float64)} for task in experiment.TASKS}
        return {'norms': {task: 1.0 for task in experiment.TASKS}, 'sample': entry}, raw

    wrapped = experiment.make_corrected_measure(native, experiment.LAMBDAS)
    record, raw = wrapped(None, None, None, entry, None, None)
    assert record['importance_applied_once'] is True
    weight = 832 / 251
    torch.testing.assert_close(raw['relchem']['p'], torch.ones(2, dtype=torch.float64) * weight)
    torch.testing.assert_close(raw['ae17']['p'], torch.ones(2, dtype=torch.float64))

    def already(*_args, **_kwargs):
        raw_gradients = {task: {'p': torch.ones(2, dtype=torch.float64)} for task in experiment.TASKS}
        return {'importance_applied_once': True}, raw_gradients

    with pytest.raises(RuntimeError, match='already applied'):
        experiment.make_corrected_measure(already, experiment.LAMBDAS)(None, None, None, entry, None, None)


def test_reused_calibration_changes_only_manifest_binding():
    original = {'lambda': dict(experiment.LAMBDAS), 'scales': {'relchem': 1.0}, 'manifest_sha256': 'old'}
    cloned = copy.deepcopy(original)
    cloned['manifest_sha256'] = 'new'
    cloned['provenance'] = {'kind': 'reuse of a historical calibration, not a new calibration'}
    assert experiment.calibration_changes(cloned, original) == {'manifest_sha256', 'provenance'}
    path = experiment.CONTROL / 'calibration.json'
    if not path.exists():
        pytest.skip('historical calibration is not on this machine')
    historical = experiment.run.read(path)
    rebound = experiment.reused_calibration(historical, 'new-manifest', experiment.CONTROL_CALIBRATION_SHA)
    assert experiment.calibration_changes(rebound, historical) == {'manifest_sha256', 'provenance'}
    assert {task: rebound['lambda'][task] for task in experiment.TASKS} == experiment.LAMBDAS
    assert rebound['provenance']['original_calibration'] == historical
    assert rebound['provenance']['kind'].startswith('reuse of a historical calibration')


def test_real_population_matches_control_streams():
    if not experiment.CONTROL.exists() or not experiment.run.DATA.exists():
        pytest.skip('local dataset or control artifacts are absent')
    bundle = experiment.run.PublicationDataset(experiment.run.DATA)
    try:
        rows = list(bundle.reactions.values())
        assert experiment.relchem_population(rows)
        control = experiment.run.read(experiment.CONTROL / 'sampling_manifest.json')
        manifest = experiment.build_manifest(rows, control)
        experiment.validate_manifest(manifest, bundle.reactions, bundle.systems, control)
    finally:
        bundle.close()


def test_decision_labels():
    assert experiment.classify(8.4, 8.6, True, True) == 'GO'
    assert experiment.classify(8.58, 8.60, True, True) == 'PARTIAL'
    assert experiment.classify(8.4, 8.6, False, True) == 'PARTIAL'
    assert experiment.classify(9.4, 8.6, True, False) == 'NO-GO'


def test_training_loop_copies_native_checkpoints_without_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'OUTPUT', tmp_path)
    monkeypatch.setattr(experiment, 'prepare', lambda: None)
    monkeypatch.setattr(experiment, '_protected_hashes', lambda: {'kept': 'same'})
    arm = tmp_path / 'ordinary_sgd_adamw'
    arm.mkdir()
    manifest = control_rows()
    experiment.run.write(tmp_path / 'sampling_manifest.json', manifest)
    experiment.run.write(tmp_path / 'calibration.json', {'lambda': experiment.LAMBDAS})
    experiment.run.write(tmp_path / 'preflight.json', {'passed': True})
    experiment.run.write(tmp_path / 'protocol.json', {'protected_hashes': {'kept': 'same'}})
    calls = []

    def fake_train(_folder, updates, stop_at, **kwargs):
        assert updates == 90 and kwargs['learning_rate'] == 1e-4 and kwargs['constant_lr'] is True
        latest = arm / 'latest.pt'
        current = 0 if not latest.exists() else torch.load(latest, map_location='cpu', weights_only=False)['cursor']
        calls.append((current, stop_at))
        logs = []
        for index in range(stop_at):
            logs.append({'importance_applied_once': True, 'learning_rate': 1e-4, 'sample': manifest[index],
                         'joint_gradient_norm_corrected': 1.0, 'weighted_gradient_norm': 1.0})
        torch.save({'cursor': stop_at, 'logs': logs, 'scheduler': None, 'optimizer': {'state': {}}, 'model': {}}, latest)
        if not (arm / 'checkpoint_0.pt').exists():
            torch.save({'cursor': 0, 'logs': [], 'scheduler': None, 'optimizer': {'state': {}}, 'model': {}},
                       arm / 'checkpoint_0.pt')

    class _Model:
        def load_state_dict(self, _state):
            return None

    monkeypatch.setattr(experiment.run, 'train', fake_train)
    monkeypatch.setattr(experiment.run, 'model_at', lambda _state: (_Model(), None))
    monkeypatch.setattr(experiment.run.existing, 'digest', lambda _model: experiment.INITIAL_SHA)
    experiment.train()
    assert calls == [(0, 20), (20, 59), (59, 70), (70, 80), (80, 90)]
    experiment.train()
    assert calls == [(0, 20), (20, 59), (59, 70), (70, 80), (80, 90)]
    for cursor in (0, 20, 59, 70, 80, 90):
        assert torch.load(arm / f'checkpoint_{cursor}.pt', map_location='cpu', weights_only=False)['cursor'] == cursor


def test_evaluator_has_no_importance_path():
    text = (experiment.run.ROOT / 'tools' / 'evaluate_microbatch_endpoint.py').read_text(encoding='utf-8')
    assert 'importance' not in text
    assert 'def measure' not in text
