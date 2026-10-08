"""Bounded 2x2 coefficient screen, reusing the qualified trainer and frozen probe."""
import copy
import itertools
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import adamw_lr_sweep as sweep

run = sweep.run
OUT = run.ROOT.parent / 'lap_adamw_weight_factorial_20261008'
CONTROL = sweep.OUT / '1e-4'
ARMS = {'A': (1, 1), 'B': (2, 1), 'C': (1, 2), 'D': (2, 2)}
EXPECTED59 = '8938753bee6cfcb55ccdac9c02a15cbb3e3b2093290c9122e5e00ad400a966aa'


def coefficients(base, arm):
    result = dict(base)
    result['relchem'] *= ARMS[arm][0]
    result['op'] *= ARMS[arm][1]
    return result


def freeze():
    sweep.verify()
    latest = CONTROL / 'ordinary_sgd_adamw/latest.pt'
    assert run.file_sha256(latest) == EXPECTED59
    old = run.read(CONTROL / 'ordinary_sgd_adamw/protocol.json')
    assert old['source_sha256'] == run.file_sha256(run.ROOT / 'train_lap_microbatch.py')
    assert old['lr'] == 1e-4 and old['scheduler'] == 'constant'
    for name, sha in old['physics_sources_sha256'].items():
        assert run.file_sha256(run.ROOT / 'train_models' / name) == sha
    if OUT.exists():
        return
    OUT.mkdir()
    base = run.read(CONTROL / 'calibration.json')['lambda']
    bundle = run.PublicationDataset(run.DATA)
    try:
        rows = list(bundle.reactions.values())
        rng = random.Random(931003)
        diagnostic = [{'cursor': i, **{t: run.sample_pair(rows, t, i, seed=931003) for t in ('relchem', 'ae17')},
                       'mrks_id': rng.choice(sorted(bundle.systems))} for i in range(3)]
    finally:
        bundle.close()
    run.write(OUT / 'protocol.json', {'reference_commit': '09876e8dabeaf92e1934742ec1d15276fa077409',
        'arms': {a: coefficients(base, a) for a in ARMS}, 'updates': 20, 'lr': 1e-4,
        'scheduler': 'constant', 'control_reused': True, 'control_protocol': old,
        't59_sha256': EXPECTED59, 'diagnostic_manifest': diagnostic,
        'driver_sha256': run.file_sha256(__file__), 'probe_manifest_sha256': run.file_sha256(sweep.OUT / 'probe_manifest.json'),
        'bootstrap': '5000 paired within-stratum resamples seed931002; paired effects on t20/t0 ratios; limited panel only',
        'decision': 'Report uncertainty; no automatic continuation or winner; no validation/full-corpus endpoints'})
    folder = OUT / 't59'
    (folder / 'ordinary_sgd_adamw').mkdir(parents=True)
    shutil.copyfile(latest, folder / 'ordinary_sgd_adamw/checkpoint_20.pt')
    for arm in ('B', 'C', 'D'):
        folder = OUT / arm
        folder.mkdir()
        for name in ('protocol.json', 'calibration.json', 'sampling_manifest.json'):
            shutil.copyfile(CONTROL / name, folder / name)
        calibration = run.read(folder / 'calibration.json')
        calibration['lambda'] = coefficients(base, arm)
        calibration['factorial_multipliers'] = ARMS[arm]
        run.write(folder / 'calibration.json', calibration)


def direction_audit():
    path = OUT / 'directions.json'
    if path.exists():
        return
    protocol = run.read(OUT / 'protocol.json')
    model, shadow = run.model_at(run.read(CONTROL / 'protocol.json')['initial_state'])
    initial = copy.deepcopy(model.state_dict())
    bundle = run.PublicationDataset(run.DATA)
    result = []
    try:
        for entry in protocol['diagnostic_manifest']:
            model.load_state_dict(initial)
            record, raw = run.measure(model, shadow, bundle, entry, bundle.chemistry_dispersions(),
                                      run.read(run.DATA / 'mrks/dispersion.json'), 4096)
            g = torch.stack([torch.cat([v.flatten() for v in raw[t].values()]) for t in run.TASKS])
            norms = g.norm(dim=1)
            directions, per_arm = {}, {}
            for arm in ARMS:
                model.load_state_dict(initial)
                params = run.existing.named_trainable_parameters(model)
                before = torch.cat([p.detach().flatten().double() for p in params.values()])
                opt = torch.optim.AdamW(params.values(), lr=1e-4,
                    **{**run.read(CONTROL / 'protocol.json')['adamw'], 'betas': (0.9, 0.999)})
                joint = run.adamw_step(model, opt, raw, protocol['arms'][arm])
                d = torch.cat([v.flatten() for v in joint.values()])
                displacement = torch.cat([p.detach().flatten().double() for p in params.values()]) - before
                directions[arm] = d
                per_arm[arm] = {'weighted_norm': float(d.norm()), 'task_dots_gradient_direction': (g @ d).tolist(),
                    'normalized_task_progress': ((g @ d) / norms / d.norm()).tolist(),
                    'first_step_norm': float(displacement.norm()),
                    'predicted_loss_changes_actual_adamw': (g @ displacement).tolist()}
            result.append({'sample': entry, 'norms': record['norms'],
                'task_cosines': ((g @ g.T) / norms[:, None] / norms[None, :]).tolist(), 'arms': per_arm,
                'gradient_direction_cosines': {a+b: float(torch.dot(directions[a], directions[b]) /
                    directions[a].norm() / directions[b].norm()) for a,b in itertools.combinations(ARMS, 2)}})
            del raw, g, joint, directions
    finally:
        bundle.close()
    run.write(path, result)


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze()
    assert run.read(OUT / 'protocol.json')['driver_sha256'] == run.file_sha256(__file__)
    sweep.probe(OUT / 't59')
    print('T59_PROBE_COMPLETE', flush=True)
    direction_audit()
    print('DIRECTIONS_COMPLETE', flush=True)
    for arm in ('B', 'C', 'D'):
        folder = OUT / arm
        print('ARM_START', arm, flush=True)
        run.train(folder, 90, stop_at=20, runtime_seconds=1800, learning_rate=1e-4, constant_lr=True, diagnostics=True)
        saved = torch.load(folder / 'ordinary_sgd_adamw/latest.pt', map_location='cpu', weights_only=False)
        assert saved['cursor'] == 20
        sweep.probe(folder)
        print('ARM_PROBE_COMPLETE', arm, flush=True)
        torch.cuda.empty_cache()
    assert run.file_sha256(CONTROL / 'ordinary_sgd_adamw/latest.pt') == EXPECTED59


if __name__ == '__main__':
    main()
