"""Bounded continuation of the preserved IID LR=1e-4 arm; no protocol changes."""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from tools.evaluate_microbatch_endpoint import evaluate

SOURCE = run.ROOT.parent / 'lap_adamw_lr_sweep_20261008/1e-4'
OUTPUT = run.ROOT.parent / 'lap_iid_adamw_t59_t90_20261009'
HISTORICAL = run.ROOT.parent / 'lap_relchem_joint_epoch_20261009/R'
START_SHA = '8938753bee6cfcb55ccdac9c02a15cbb3e3b2093290c9122e5e00ad400a966aa'


def prepare():
    source = SOURCE / 'ordinary_sgd_adamw/latest.pt'
    assert run.file_sha256(source) == START_SHA
    saved = torch.load(source, map_location='cpu', weights_only=False)
    assert saved['cursor'] == len(saved['logs']) == 59
    assert saved['total_updates'] == 90 and saved['scheduler'] is None
    assert saved['rng'] and saved['optimizer']['state']
    assert all(float(v['step']) == 59 for v in saved['optimizer']['state'].values())
    protocol = run.read(SOURCE / 'ordinary_sgd_adamw/protocol.json')
    assert protocol['source_sha256'] == run.file_sha256(run.__file__)
    for name, sha in protocol['physics_sources_sha256'].items():
        assert run.file_sha256(run.ROOT / 'train_models' / name) == sha
    assert saved['manifest_sha256'] == protocol['manifest_sha256'] == run.file_sha256(SOURCE / 'sampling_manifest.json')
    assert saved['calibration'] == run.read(SOURCE / 'calibration.json')
    assert run.file_sha256(SOURCE / 'calibration.json') == protocol['calibration_sha256']
    assert all(r['learning_rate'] == 1e-4 for r in saved['logs'])
    manifest = run.read(SOURCE / 'sampling_manifest.json')
    assert len(manifest) == 90
    assert all(r['sample'] == manifest[i] for i, r in enumerate(saved['logs']))
    model, _ = run.model_at(run.read(SOURCE / 'protocol.json')['initial_state'])
    order = [{'name': n, 'shape': list(p.shape), 'numel': p.numel()}
             for n, p in run.existing.named_trainable_parameters(model).items()]
    assert order == run.read(SOURCE / 'ordinary_sgd_adamw/parameter_order.json')
    assert sum(r['numel'] for r in order) == 9446
    model.load_state_dict(saved['model'])
    assert run.existing.digest(model) == saved['logs'][-1]['after_sha256']
    if not OUTPUT.exists():
        # Preserve the original arm, including moments and RNG, byte-for-byte.
        shutil.copytree(SOURCE, OUTPUT)
        shutil.copyfile(source, OUTPUT / 'ordinary_sgd_adamw/checkpoint_59.pt')
        shutil.copyfile(HISTORICAL / 'evaluation_manifest.json', OUTPUT / 'evaluation_manifest.json')
        for stage in ('chemistry_one_variant', 'mrks', 'validation'):
            shutil.copyfile(HISTORICAL / f'endpoint_0_{stage}.json', OUTPUT / f'baseline_0_{stage}.json')
    assert run.file_sha256(OUTPUT / 'ordinary_sgd_adamw/checkpoint_59.pt') == START_SHA
    run.write(OUTPUT / 'continuation_receipt.json', {
        'starting_checkpoint_sha256': START_SHA, 'starting_cursor': 59,
        'additional_updates': 31, 'final_cursor': 90, 'original_unchanged': True,
        'parameter_order_verified': True, 'optimizer_steps': 59, 'rng_present': True,
        'source_protocol': protocol, 'evaluation_manifest_sha256': run.file_sha256(OUTPUT / 'evaluation_manifest.json'),
        'evaluator_sha256': run.file_sha256(run.ROOT / 'tools/evaluate_microbatch_endpoint.py'),
        'continuation_tool_sha256': run.file_sha256(__file__),
    })
    print('PREFLIGHT_PASS cursor59; original preserved', flush=True)


def train():
    prepare()
    for cursor in (70, 80, 90):
        latest = OUTPUT / 'ordinary_sgd_adamw/latest.pt'
        current = torch.load(latest, map_location='cpu', weights_only=False)['cursor']
        if current < cursor:
            run.train(OUTPUT, 90, stop_at=cursor, runtime_seconds=1800,
                      learning_rate=1e-4, constant_lr=True, diagnostics=True)
            current = torch.load(latest, map_location='cpu', weights_only=False)['cursor']
        if current != cursor and not (OUTPUT / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt').exists():
            raise RuntimeError(f'Safe pause at {current}; no automatic changed-setting retry')
        if current == cursor:
            target = OUTPUT / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt'
            if not target.exists():
                shutil.copyfile(latest, target)
        assert run.file_sha256(SOURCE / 'ordinary_sgd_adamw/latest.pt') == START_SHA
    print('COMPLETE exactly31 additional updates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--cursor', type=int, choices=(59, 70, 80, 90))
    parser.add_argument('--objective', choices=('validation', 'chemistry', 'mrks'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'train':
        train()
    else:
        if args.cursor is None or args.objective is None:
            parser.error('Evaluation needs cursor and objective')
        evaluate(OUTPUT, args.cursor, args.objective, 1800)
