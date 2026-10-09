"""Two matched LR-only branches from preserved t70; reuse qualified trainer."""
import argparse
import copy
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from tools.evaluate_microbatch_endpoint import evaluate

SOURCE = run.ROOT.parent / 'lap_iid_adamw_t59_t90_20261009'
OUTPUT = run.ROOT.parent / 'lap_iid_adamw_lr_branches_20261009'
RATES = {'B': 3e-5, 'C': 1e-5}


def original_integrity():
    prior = run.read(run.ROOT / 'iid_adamw_t59_t90_metrics.json')
    for name, sha in prior['hashes'].items():
        assert run.file_sha256(SOURCE / name) == sha, name
    protocol = prior['source_hashes']['source_protocol']
    assert run.file_sha256(run.__file__) == protocol['source_sha256']
    for name, sha in protocol['physics_sources_sha256'].items():
        assert run.file_sha256(run.ROOT / 'train_models' / name) == sha
    return prior


def prepare():
    prior = original_integrity()
    saved = torch.load(SOURCE / 'ordinary_sgd_adamw/checkpoint_70.pt', map_location='cpu', weights_only=False)
    assert saved['cursor'] == len(saved['logs']) == 70 and saved['scheduler'] is None
    assert saved['total_updates'] == 90 and saved['rng']
    assert all(float(s['step']) == 70 for s in saved['optimizer']['state'].values())
    assert all(g['lr'] == 1e-4 and g['betas'] == (0.9, 0.999) and g['eps'] == 1e-8
               and g['weight_decay'] == .01 and g['foreach'] is False for g in saved['optimizer']['param_groups'])
    manifest = run.read(SOURCE / 'sampling_manifest.json')
    assert len(manifest) == 90 and saved['manifest_sha256'] == run.file_sha256(SOURCE / 'sampling_manifest.json')
    assert all(r['sample'] == manifest[i] for i, r in enumerate(saved['logs']))
    assert saved['calibration'] == run.read(SOURCE / 'calibration.json')
    OUTPUT.mkdir(exist_ok=True)
    for arm in ('A', 'B', 'C'):
        folder = OUTPUT / arm
        if not folder.exists():
            (folder / 'ordinary_sgd_adamw').mkdir(parents=True)
            for name in ('protocol.json', 'calibration.json', 'sampling_manifest.json', 'evaluation_manifest.json'):
                shutil.copyfile(SOURCE / name, folder / name)
            for name in ('baseline_0_chemistry_one_variant.json', 'baseline_0_mrks.json', 'baseline_0_validation.json'):
                shutil.copyfile(SOURCE / name, folder / name)
            shutil.copyfile(SOURCE / 'ordinary_sgd_adamw/parameter_order.json', folder / 'ordinary_sgd_adamw/parameter_order.json')
            shutil.copyfile(SOURCE / 'ordinary_sgd_adamw/checkpoint_70.pt', folder / 'ordinary_sgd_adamw/checkpoint_70.pt')
            if arm == 'A':
                for cursor in (59, 80, 90):
                    shutil.copyfile(SOURCE / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt', folder / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt')
                for name in prior['hashes']:
                    if name.startswith('endpoint_'):
                        shutil.copyfile(SOURCE / name, folder / name)
            else:
                shutil.copyfile(SOURCE / 'ordinary_sgd_adamw/checkpoint_70.pt', folder / 'ordinary_sgd_adamw/latest.pt')
                shutil.copyfile(SOURCE / 'endpoint_70_validation.json', folder / 'endpoint_70_validation.json')
        assert run.file_sha256(folder / 'ordinary_sgd_adamw/checkpoint_70.pt') == prior['hashes']['ordinary_sgd_adamw/checkpoint_70.pt']
        assert run.file_sha256(folder / 'sampling_manifest.json') == saved['manifest_sha256']
    common = OUTPUT / 'common_samples_70_89.json'
    if common.exists():
        assert run.read(common) == manifest[70:90]
    else:
        run.write(common, manifest[70:90])
    protocol = {'source_checkpoint_sha256': prior['hashes']['ordinary_sgd_adamw/checkpoint_70.pt'],
                'rates': {'A': 1e-4, **RATES}, 'new_updates': 40, 'start_cursor': 70, 'end_cursor': 90,
                'common_samples_sha256': run.file_sha256(common), 'whole_manifest_sha256': saved['manifest_sha256'],
                'fixed_coefficients': saved['calibration']['lambda'], 'dataset_sha256': run.DATA_SHA,
                'evaluation_manifest_sha256': run.file_sha256(SOURCE / 'evaluation_manifest.json'),
                'trainer_sha256': run.file_sha256(run.__file__), 'evaluator_sha256': run.file_sha256(run.ROOT / 'tools/evaluate_microbatch_endpoint.py'),
                'wrapper_sha256': run.file_sha256(__file__), 'reference_protocol': prior['source_hashes']['source_protocol'],
                'only_change': 'Override param-group LR after native restore; all moments/counters/RNG retained',
                'selection': 'Best new Clean28 among B/C t80/t90; exactly one full scientific endpoint',
                'reference_artifact_hashes': prior['hashes']}
    path = OUTPUT / 'frozen_protocol.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    print('PREPARE_PASS control preserved; t70 moments70; common20 samples frozen', flush=True)


def restore_lr(native_restore, lr, *args, **kwargs):
    """Only the allowed LR field changes after qualified native state restoration."""
    cursor = native_restore(*args, **kwargs)
    optimizer = args[2]
    before = copy.deepcopy(optimizer.state_dict())
    assert 70 <= cursor < 90 and all(float(s['step']) == cursor for s in before['state'].values())
    assert all(g['lr'] == (1e-4 if cursor == 70 else lr) for g in before['param_groups'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    expected = copy.deepcopy(before)
    for group in expected['param_groups']:
        group['lr'] = lr
    assert run.existing.equal(optimizer.state_dict(), expected)
    return cursor


def train(arm):
    assert arm in RATES
    frozen = run.read(OUTPUT / 'frozen_protocol.json')
    assert run.file_sha256(__file__) == frozen['wrapper_sha256']
    original_integrity()
    for cursor in (70, 80):
        for stage in ('chemistry_one_variant', 'mrks'):
            assert run.read(OUTPUT / f'A/endpoint_{cursor}_{stage}.json')['complete'], 'Stage A incomplete'
    folder, lr = OUTPUT / arm, RATES[arm]
    order = run.read(folder / 'ordinary_sgd_adamw/parameter_order.json')
    starting = torch.load(folder / 'ordinary_sgd_adamw/checkpoint_70.pt', map_location='cpu', weights_only=False)
    base = torch.cat([starting['model'][r['name']].flatten().double() for r in order]).cuda()
    native_restore, native_diagnostics = run.restore, run.adamw_diagnostics

    def diagnostic(model, optimizer, raw, before, initial):
        result = native_diagnostics(model, optimizer, raw, before, initial)
        current = torch.cat([p.detach().flatten().double() for p in run.existing.named_trainable_parameters(model).values()])
        result['parameter_displacement_from_t70'] = float((current - base).norm())
        moment_step = result['moment_step']
        path = folder / f'actual_step_{moment_step}.npy'
        np.save(path, (current - before).cpu().numpy())
        result['actual_step_sha256'] = run.file_sha256(path)
        return result

    with patch.object(run, 'restore', lambda *a, **k: restore_lr(native_restore, lr, *a, **k)), \
         patch.object(run, 'adamw_diagnostics', diagnostic):
        for stop in (80, 90):
            latest = folder / 'ordinary_sgd_adamw/latest.pt'
            current = torch.load(latest, map_location='cpu', weights_only=False)['cursor']
            if current < stop:
                run.train(folder, 90, stop_at=stop, runtime_seconds=1800, learning_rate=lr,
                          constant_lr=True, diagnostics=True)
                current = torch.load(latest, map_location='cpu', weights_only=False)['cursor']
            if current != stop and not (folder / f'ordinary_sgd_adamw/checkpoint_{stop}.pt').exists():
                raise RuntimeError(f'Safe pause at {current}; no changed-setting retry')
            if current == stop:
                target = folder / f'ordinary_sgd_adamw/checkpoint_{stop}.pt'
                if not target.exists():
                    shutil.copyfile(latest, target)
    saved = torch.load(folder / 'ordinary_sgd_adamw/latest.pt', map_location='cpu', weights_only=False)
    assert len(saved['logs']) == saved['cursor'] == 90
    assert saved['logs'][:70] == starting['logs']
    manifest = run.read(folder / 'sampling_manifest.json')
    assert all(r['sample'] == manifest[i] and r['learning_rate'] == lr for i, r in enumerate(saved['logs'][70:], 70))
    original_integrity()
    print('BRANCH_COMPLETE', arm, 'exactly20 updates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'evaluate', 'train'))
    parser.add_argument('--arm', choices=('A', 'B', 'C'))
    parser.add_argument('--cursor', type=int, choices=(70, 80, 90))
    parser.add_argument('--objective', choices=('chemistry', 'mrks', 'validation'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'train':
        train(args.arm)
    else:
        original_integrity()
        if None in (args.arm, args.cursor, args.objective):
            parser.error('Evaluation requires arm, cursor, objective')
        evaluate(OUTPUT / args.arm, args.cursor, args.objective, 1800)
        original_integrity()
