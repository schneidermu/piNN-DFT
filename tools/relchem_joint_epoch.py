"""Two matched, one-epoch AdamW arms; reuse qualified objectives and checkpoints."""
import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from train_models.lap_chemistry_sampling import epoch_samples, fixed_evaluation

OUT = run.ROOT.parent / 'lap_relchem_joint_epoch_20261009'
PARENT = run.ROOT.parent / 'lap_population_adamw_90update_20261008'
SOURCES = ('train_lap_microbatch.py', 'train_models/lap_fixed_adamw.py',
           'train_models/lap_chemistry_sampling.py', 'train_models/lap_moo_training.py',
           'train_models/lap_training.py', 'train_models/lap_vxc.py',
           'train_models/lap_operator.py', 'tools/relchem_joint_epoch.py',
           'tools/evaluate_microbatch_endpoint.py')


def freeze():
    if OUT.exists():
        raise RuntimeError('Preserve frozen experiment; do not overwrite')
    bundle = run.PublicationDataset(run.DATA)
    try:
        assert bundle.manifest['logical_sha256'] == run.DATA_SHA
        assert sum(r['task'] == 'relchem' for r in bundle.reactions.values()) == 251
        assert sum(r['task'] == 'ae17' for r in bundle.reactions.values()) == 17
        assert len(bundle.systems) == 90
        manifest = epoch_samples(bundle.reactions, bundle.systems)
        evaluation = fixed_evaluation(bundle.reactions)
        assert len(manifest) == len({e['relchem']['identity'] for e in manifest}) == 251
        assert manifest == epoch_samples(bundle.reactions, bundle.systems)
        parent = run.read(PARENT / 'protocol.json')
        calibration = run.read(PARENT / 'calibration.json')
        OUT.mkdir()
        run.write(OUT / 'sampling_manifest.json', manifest)
        run.write(OUT / 'evaluation_manifest.json', evaluation)
        protocol = {'initial_state': parent['initial_state'], 'dataset_sha256': run.DATA_SHA,
                        'adamw': parent['adamw'], 'lr': 1e-4, 'scheduler': None, 'total_updates': 251,
                        'exc_chunk_size': 4096, 'operator_chunk_size': 256, 'chemistry_chunk': 16384,
                        'sampling_seed': 202610092, 'evaluation_seed': 202610091,
                        'sampling_manifest_sha256': run.file_sha256(OUT / 'sampling_manifest.json'),
                        'evaluation_manifest_sha256': run.file_sha256(OUT / 'evaluation_manifest.json'),
                        'source_commit': '9abd2621484f38d0c291b1a9608936a93d6eb67d',
                        'source_hashes': {p: run.file_sha256(run.ROOT / p) for p in SOURCES},
                        'arms': {'R': ['relchem'], 'J': list(run.TASKS)},
                        'evaluation': '251 relchem +17 AE17; one fixed variant each; all90 mRKS',
                        'stop': 'Exactly one chemistry epoch; no automatic continuation'}
        run.write(OUT / 'protocol.json', protocol)
        for arm in ('R', 'J'):
            folder = OUT / arm
            folder.mkdir()
            for name, content in (('protocol.json', protocol), ('calibration.json', calibration),
                                  ('sampling_manifest.json', manifest), ('evaluation_manifest.json', evaluation)):
                run.write(folder / name, content)
            assert run.file_sha256(folder / 'sampling_manifest.json') == protocol['sampling_manifest_sha256']
    finally:
        bundle.close()


def verify(folder):
    protocol = run.read(folder / 'protocol.json')
    for source, digest in protocol['source_hashes'].items():
        if run.file_sha256(run.ROOT / source) != digest:
            raise ValueError('Frozen scientific source changed: ' + source)
    for name in ('sampling', 'evaluation'):
        assert run.file_sha256(folder / (name + '_manifest.json')) == protocol[name + '_manifest_sha256']
    return protocol


def relchem_step(model, optimizer, raw, coefficient):
    """One F64 coefficient application and one native-F32 optimizer-boundary cast."""
    parameters = run.existing.named_trainable_parameters(model)
    if tuple(raw) != tuple(parameters):
        raise ValueError('Single-task coordinate mismatch')
    joint = {n: g.double() * coefficient for n, g in raw.items()}
    if not all(torch.isfinite(g).all() for g in joint.values()):
        raise FloatingPointError('Nonfinite relchem derivative')
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in parameters.items():
        parameter.grad = joint[name].to(parameter.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not all(torch.isfinite(p).all() for p in parameters.values()):
        raise FloatingPointError('Nonfinite parameters')
    return joint


def qualify():
    protocol = verify(OUT / 'J')
    model, shadow = run.model_at(protocol['initial_state'])
    bundle = run.PublicationDataset(run.DATA)
    try:
        entry = run.read(OUT / 'sampling_manifest.json')[0]
        checks = run.parity(model, shadow, bundle, entry, bundle.chemistry_dispersions())
        run.write(OUT / 'preflight.json', {'checks': checks, 'initial_tensor_sha256': run.existing.digest(model),
                  'source_hashes': protocol['source_hashes'], 'manifests_reproducible': True,
                  'competing_gradients_in_R': False, 'all_variants_evaluated': False})
    finally:
        bundle.close()


def train(arm, seconds=3500):
    folder = OUT / arm
    protocol = verify(folder)
    if not (OUT / 'preflight.json').exists():
        raise ValueError('Numerical preflight required')
    torch.manual_seed(41)
    np.random.seed(41)
    random.seed(41)
    model, shadow = run.model_at(protocol['initial_state'])
    parameters = run.existing.named_trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters.values(), lr=protocol['lr'],
                                 **{**protocol['adamw'], 'betas': tuple(protocol['adamw']['betas'])})
    calibration = run.read(folder / 'calibration.json')
    manifest = run.read(folder / 'sampling_manifest.json')
    manifest_sha = protocol['sampling_manifest_sha256']
    target = folder / 'ordinary_sgd_adamw'
    target.mkdir(exist_ok=True)
    latest = target / 'latest.pt'
    logs, cursor = [], 0

    def save(path, index):
        temporary = path.with_suffix('.tmp')
        torch.save({'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                        'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': run.existing.capture_rng_state(),
                        'cursor': index, 'epoch': index // 251, 'epoch_cursor': index % 251,
                        'manifest_sha256': manifest_sha, 'calibration': calibration, 'scheduler': None,
                        'logs': list(logs), 'total_updates': 251, 'configuration': protocol, 'arm': arm,
                        'protocol_sha256': run.file_sha256(folder / 'protocol.json')}, temporary)
        temporary.replace(path)

    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['configuration'] != protocol or saved['arm'] != arm:
            raise ValueError('Checkpoint protocol/arm differs')
        cursor = run.restore(latest, model, optimizer, manifest_sha, calibration)
        logs = saved['logs']
    else:
        save(target / 'checkpoint_0.pt', 0)
        save(latest, 0)
    bundle = run.PublicationDataset(run.DATA)
    began = time.perf_counter()
    try:
        assert bundle.manifest['logical_sha256'] == protocol['dataset_sha256']
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = run.read(run.DATA / 'mrks/dispersion.json')
        for index in range(cursor, 251):
            if time.perf_counter() - began > seconds:
                break
            entry = manifest[index]
            timer = time.perf_counter()
            before = run.existing.digest(model)
            if arm == 'J':
                record, raw = run.measure(model, shadow, bundle, entry, dispersion, mrks_dispersion,
                                          exc_chunk_size=4096)
                joint = run.adamw_step(model, optimizer, raw, calibration['lambda'])
            else:
                torch.cuda.reset_peak_memory_stats()
                sample = entry['relchem']
                reaction = bundle.chemistry('train_relchem').load_variant(sample['identity'], sample['variant'])
                reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
                if len(reaction['Grid']) > 131072:
                    reaction['model_point_chunk_size'] = 16384
                value, grad = run.chemistry(model, shadow, reaction, dispersion).value_and_grad()
                record = {'sample': sample, 'losses': {'relchem': value},
                              'norms': {'relchem': float(torch.cat([g.flatten() for g in grad.values()]).norm())}}
                joint = relchem_step(model, optimizer, grad, calibration['lambda']['relchem'])
                del reaction, grad
                raw = None
            torch.cuda.synchronize()
            record.update(cursor=index + 1, epoch=0, epoch_cursor=index + 1,
                          before_sha256=before, after_sha256=run.existing.digest(model), lr=protocol['lr'],
                          weighted_gradient_norm=float(torch.cat([g.flatten() for g in joint.values()]).norm()),
                          total_seconds=time.perf_counter() - timer,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            logs.append(record)
            del raw, joint
            cursor = index + 1
            save(latest, cursor)
            if cursor % 25 == 0 or cursor == 251:
                save(target / f'checkpoint_{cursor}.pt', cursor)
            run.write(folder / 'training_status.json', {'cursor': cursor, 'epoch': cursor // 251,
                      'epoch_cursor': cursor % 251, 'complete': cursor == 251,
                      'checkpoint_sha256': run.file_sha256(latest), 'logs': logs})
            print('EPOCH_UPDATE', arm, cursor, record['total_seconds'], record['losses'], flush=True)
    except (FloatingPointError, RuntimeError) as error:
        run.write(folder / 'failure.json', {'cursor': cursor, 'error': str(error), 'preserved_checkpoint': str(latest)})
        raise
    finally:
        bundle.close()
    print('EPOCH_BOUNDARY', arm, cursor, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'qualify', 'train'))
    parser.add_argument('--arm', choices=('R', 'J'))
    parser.add_argument('--seconds', type=int, default=3500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'freeze':
        freeze()
    elif args.action == 'qualify':
        qualify()
    else:
        if args.arm is None or not 0 < args.seconds <= 3600:
            raise ValueError('Arm and at-most-one-hour bound required')
        train(args.arm, args.seconds)
