"""Controlled constant-LR screen; reuse the qualified microbatch trainer/evaluator."""
import argparse
import hashlib
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run

OUT = run.ROOT.parent / 'lap_adamw_lr_sweep_20261008'
PARENT = run.ROOT.parent / 'lap_population_adamw_90update_20261008'
LRS = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
PROBE_SEED = 931001


def label(lr):
    return format(lr, '.0e').replace('e-0', 'e-')


def probe_manifest(bundle):
    """Independent stratified diagnostic, never the training sampler."""
    rows = list(bundle.reactions.values())
    result = {'relchem': [], 'ae17': [], 'mrks': []}
    def rng(domain):
        return random.Random(int.from_bytes(hashlib.sha256(f'{PROBE_SEED}:{domain}'.encode()).digest(), 'big'))
    for task in ('relchem', 'ae17'):
        groups = {}
        for row in rows:
            if row['task'] == task:
                groups.setdefault(row['database'], []).append(row)
        total = sum(map(len, groups.values()))
        for db, group in sorted(groups.items()):
            generator = rng(task + ':' + db)
            count = 2 if task == 'relchem' else 8
            for row in generator.sample(sorted(group, key=lambda x: x['id']), count):
                result[task].append({'identity': row['id'], 'variant': generator.choice(sorted(row['variants'])),
                                     'stratum': db, 'weight': len(group) / total / count})
    ordered = sorted(bundle.systems, key=lambda x: (bundle.systems[x]['n_grid'], x))
    for q, indices in enumerate(np.array_split(np.arange(len(ordered)), 4)):
        group = [ordered[int(i)] for i in indices]
        for identity in rng(f'mrks:{q}').sample(group, 2):
            result['mrks'].append({'identity': identity, 'stratum': str(q), 'weight': len(group) / len(ordered) / 2})
    return result


def freeze():
    if OUT.exists():
        raise FileExistsError('Frozen sweep exists; use screen/select/continue, do not regenerate')
    bundle = run.PublicationDataset(run.DATA)
    try:
        assert bundle.manifest['logical_sha256'] == run.DATA_SHA
        probe = probe_manifest(bundle)
    finally:
        bundle.close()
    OUT.mkdir()
    run.write(OUT / 'probe_manifest.json', probe)
    protocol = {'source_commit': 'a1081eac10c6ceab4e8989cd0a82369ef4a02baf', 'learning_rates': list(LRS),
        'scheduler': 'constant throughout 20-screen and continued 90-update arms', 'screen_updates': 20,
        'total_updates': 90, 'probe_seed': PROBE_SEED, 'probe': '16 stratified relchem; 8 AE17; 8 grid-size-stratified mRKS, shared Exc/op',
        'probe_weighting': 'Canonical population stratum size / task population / sampled stratum count',
        'bootstrap': '5000 paired within-stratum resamples; seed931002; limited coverage is not exact qualification',
        'training_seed': 41, 'sampling_manifest_sha256': run.file_sha256(PARENT / 'sampling_manifest.json'),
        'calibration_sha256': run.file_sha256(PARENT / 'calibration.json'), 'dataset_sha256': run.DATA_SHA,
        'trainer_sha256': run.file_sha256(run.ROOT / 'train_lap_microbatch.py'), 'driver_sha256': run.file_sha256(__file__),
        'probe_manifest_sha256': run.file_sha256(OUT / 'probe_manifest.json'),
        'screening': {'eligible': 'Finite 20-update trajectory, relchem probe ratio<=1.05, every probe ratio<=1.25, at least one ratio<1, no persistent destructive flag',
            'persistent_destructive': 'A task has negative actual-step progress in >=8 of last10 updates AND its probe ratio>1.10',
            'ranking': 'Relchem improvement first; lower relchem ratio; more improved tasks; lower maximum ratio; lower LR',
            'max_selected': 3, 'clean28_used': False},
        'endpoint': 'New fixed one-variant-per-identity t0 and endpoints: 251 relchem, 17 AE17, full90; historical exhaustive baselines must not be reused',
        'per_arm_runtime_cap_seconds': 1800}
    run.write(OUT / 'frozen_protocol.json', protocol)
    for lr in LRS:
        folder = OUT / label(lr)
        folder.mkdir()
        for name in ('protocol.json', 'calibration.json', 'sampling_manifest.json'):
            shutil.copyfile(PARENT / name, folder / name)
        local = run.read(folder / 'protocol.json')
        local.update(initial_lr=lr, final_lr=lr, scheduler='constant', only_change='Constant learning rate magnitude')
        run.write(folder / 'protocol.json', local)
    print('FROZEN', protocol, flush=True)


def verify():
    protocol = run.read(OUT / 'frozen_protocol.json')
    assert protocol['trainer_sha256'] == run.file_sha256(run.ROOT / 'train_lap_microbatch.py')
    assert protocol['driver_sha256'] == run.file_sha256(__file__)
    assert protocol['probe_manifest_sha256'] == run.file_sha256(OUT / 'probe_manifest.json')
    for lr in LRS:
        folder = OUT / label(lr)
        assert run.file_sha256(folder / 'sampling_manifest.json') == protocol['sampling_manifest_sha256']
        assert run.file_sha256(folder / 'calibration.json') == protocol['calibration_sha256']
    return protocol


def probe(folder=None):
    """Read-only small probe, resumable by unit; restores the caller RNG."""
    verify()
    output = OUT / 'probe_t0.json' if folder is None else folder / 'probe_t20.json'
    checkpoint = PARENT / 'ordinary_sgd_adamw/checkpoint_0.pt' if folder is None else folder / 'ordinary_sgd_adamw/checkpoint_20.pt'
    checkpoint_sha = run.file_sha256(checkpoint)
    receipt = run.read(output) if output.exists() else {'checkpoint_sha256': checkpoint_sha, 'rows': {}}
    assert receipt['checkpoint_sha256'] == checkpoint_sha
    if receipt.get('complete'):
        return receipt
    rng = run.existing.capture_rng_state()
    model, shadow = run.model_at(run.read(PARENT / 'protocol.json')['initial_state'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    manifest = run.read(OUT / 'probe_manifest.json')
    try:
        dispersion = bundle.chemistry_dispersions()
        for task in ('relchem', 'ae17'):
            for row in manifest[task]:
                key = task + '/' + row['identity'] + '/' + row['variant']
                if key in receipt['rows']:
                    continue
                reaction = bundle.chemistry('train_' + task).load_variant(row['identity'], row['variant'])
                reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
                if len(reaction['Grid']) > 131072:
                    reaction['model_point_chunk_size'] = 16384
                with torch.no_grad():
                    value = float(run.chemistry(model, shadow, reaction, dispersion)())
                assert np.isfinite(value)
                receipt['rows'][key] = {'task': task, **row, 'loss': value}
                run.write(output, receipt)
                del reaction
        mrks_dispersion = run.read(run.DATA / 'mrks/dispersion.json')
        for row in manifest['mrks']:
            if all(task + '/' + row['identity'] in receipt['rows'] for task in ('exc', 'op')):
                continue
            system = bundle.mrks().operator_system(row['identity'], device='cuda', dtype=torch.float32, chunk_size=4096)
            factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256,
                                                        exc_chunk_size=4096, dispersions=mrks_dispersion)
            for task, objective in zip(('exc', 'op'), factories, strict=True):
                value = float(objective().detach())
                assert np.isfinite(value)
                receipt['rows'][task + '/' + row['identity']] = {'task': task, **row, 'loss': value}
            run.write(output, receipt)
            del system, factories
        receipt['objectives'] = {t: sum(r['weight'] * r['loss'] for r in receipt['rows'].values() if r['task'] == t) for t in run.TASKS}
        assert run.existing.digest(model) == before
        receipt.update(complete=True, model_unchanged=True, parameter_gradients_computed=False)
        run.write(output, receipt)
    finally:
        bundle.close()
        run.existing.restore_rng_state(rng)
    print('PROBE', 't0' if folder is None else folder.name, receipt['objectives'], flush=True)
    return receipt


def uncertainty(current, baseline):
    rng = np.random.default_rng(931002)
    result = {}
    for task in run.TASKS:
        rows = [(key, row) for key, row in current['rows'].items() if row['task'] == task]
        groups = {}
        for key, row in rows:
            groups.setdefault(row['stratum'], []).append(key)
        numerator, denominator = np.zeros(5000), np.zeros(5000)
        for keys in groups.values():
            indices = rng.integers(0, len(keys), (5000, len(keys)))
            a = np.array([current['rows'][k]['loss'] for k in keys])
            b = np.array([baseline['rows'][k]['loss'] for k in keys])
            weight = sum(current['rows'][k]['weight'] for k in keys)
            numerator += weight * a[indices].mean(axis=1)
            denominator += weight * b[indices].mean(axis=1)
        result[task] = np.percentile(numerator / denominator, [2.5, 97.5]).tolist()
    return result


def ranking_key(row):
    r = row['probe_ratios']
    return (r['relchem'] >= 1, r['relchem'], -sum(v < 1 for v in r.values()), max(r.values()), row['lr'])


def screen_eligibility(ratios, logs):
    destructive = {t: ratios[t] > 1.10 and sum(r['task_progress_actual_step'] is not None and r['task_progress_actual_step'][j] < 0
        for r in logs[-10:]) >= 8 for j, t in enumerate(run.TASKS)}
    eligible = ratios['relchem'] <= 1.05 and max(ratios.values()) <= 1.25 and min(ratios.values()) < 1 and not any(destructive.values())
    return bool(eligible), destructive


def screen_result(lr):
    folder = OUT / label(lr)
    saved = torch.load(folder / 'ordinary_sgd_adamw/checkpoint_20.pt', map_location='cpu', weights_only=False)
    baseline, endpoint = run.read(OUT / 'probe_t0.json'), run.read(folder / 'probe_t20.json')
    ratios = {t: endpoint['objectives'][t] / baseline['objectives'][t] for t in run.TASKS}
    eligible, destructive = screen_eligibility(ratios, saved['logs'])
    return {'lr': lr, 'stable': True, 'screen_complete': True, 'probe_ratios': ratios,
        'probe_ratio_bootstrap95': uncertainty(endpoint, baseline), 'persistent_destructive': destructive,
        'eligible_to_continue': bool(eligible), 'probe_is_exact_full_corpus': False,
        'mean_update_seconds': float(np.mean([r['total_seconds'] for r in saved['logs']])),
        'peak_live_gib': max(r['peak_allocated_bytes'] for r in saved['logs']) / 2**30,
        'checkpoint20_sha256': run.file_sha256(folder / 'ordinary_sgd_adamw/checkpoint_20.pt')}


def train_arm(lr, stop):
    folder = OUT / label(lr)
    try:
        run.train(folder, 90, stop_at=stop, runtime_seconds=1800, learning_rate=lr, constant_lr=True, diagnostics=True)
    except (AssertionError, FloatingPointError, RuntimeError, ValueError, OSError) as error:
        latest = folder / 'ordinary_sgd_adamw/latest.pt'
        failure = {'lr': lr, 'stable': False, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'last_safe_checkpoint_sha256': run.file_sha256(latest) if latest.exists() else None}
        run.write(folder / 'failure.json', failure)
        print('FAILED_ARM', failure, flush=True)
        return False
    latest = torch.load(folder / 'ordinary_sgd_adamw/latest.pt', map_location='cpu', weights_only=False)
    if latest['cursor'] != stop:
        raise RuntimeError(f'Safe runtime pause at cursor{latest["cursor"]}; resume unchanged, no selection on incomplete arm')
    assert latest['scheduler'] is None and all(r['learning_rate'] == lr for r in latest['logs'])
    return True


def screen():
    verify()
    probe()
    for lr in LRS:
        folder = OUT / label(lr)
        if (folder / 'failure.json').exists() or (folder / 'screen_result.json').exists():
            continue
        print('SCREEN_START', lr, flush=True)
        if train_arm(lr, 20):
            probe(folder)
            result = screen_result(lr)
            run.write(folder / 'screen_result.json', result)
            print('SCREEN_RESULT', result, flush=True)
        torch.cuda.empty_cache()


def select():
    verify()
    path = OUT / 'selection.json'
    if path.exists():
        raise FileExistsError('Continuation selection is frozen; do not reselect')
    rows = []
    for lr in LRS:
        folder = OUT / label(lr)
        result = folder / 'screen_result.json'
        rows.append(run.read(result) if result.exists() else run.read(folder / 'failure.json'))
    selected = sorted((r for r in rows if r.get('eligible_to_continue')), key=ranking_key)[:3]
    result = {'selected_learning_rates': [r['lr'] for r in selected], 'rows': rows,
              'training_only': True, 'clean28_accessed': False, 'selection_rule': run.read(OUT / 'frozen_protocol.json')['screening']}
    run.write(path, result)
    print('SELECTED', result, flush=True)


def continue_selected():
    verify()
    for lr in run.read(OUT / 'selection.json')['selected_learning_rates']:
        folder = OUT / label(lr)
        if (folder / 'failure.json').exists() or (folder / 'ordinary_sgd_adamw/checkpoint_90.pt').exists():
            continue
        print('CONTINUE_START', lr, flush=True)
        train_arm(lr, 90)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'screen', 'select', 'continue'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'freeze': freeze, 'screen': screen, 'select': select, 'continue': continue_selected}[args.stage]()
