"""Single-reaction four-task AdamW. Profile first; no SVRG or acceptance search."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'train_models'))
import lap_symmetric_moo_trajectory_arena as existing
from train_models.lap_fixed_adamw import (
    TASKS,
    adamw_step,
    fixed_coefficients,
)
from train_models.lap_moo_protocol import _cycle_permutation, file_sha256
from train_models.publication_data import PublicationDataset

DATA = ROOT.parent / 'publication_dataset_v1'
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
OLD = ROOT.parent / 'lap_simple_four_task_pilot_20261008'
OUT = ROOT.parent / 'lap_single_reaction_adamw_20261008'
PROFILE_INDICES = (1, 5, 6)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sample_pair(rows, task, index, seed=41):
    """Independent seeded uniform identity/variant draws; no DB oversampling."""
    ordered = sorted(r['id'] for r in rows if r['task'] == task)
    if not ordered or index < 0:
        raise ValueError('Nonempty population and nonnegative cursor required')
    digest = hashlib.sha256(f'{seed}:{task}:{index}:identity'.encode()).digest()
    identity = random.Random(int.from_bytes(digest, 'big')).choice(ordered)
    record = next(r for r in rows if r['id'] == identity)
    variants = sorted(record['variants'])
    if len(variants) != 8:
        raise ValueError('Exactly eight augmentation variants required')
    digest = hashlib.sha256(f'{seed}:{task}:{index}:variant'.encode()).digest()
    variant = random.Random(int.from_bytes(digest, 'big')).choice(variants)
    return {'identity': identity, 'database': record['database'], 'reaction_id': record['reaction_id'],
            'variant': variant, 'weight': 1.0}


def sampling(bundle, count=90):
    rows = list(bundle.reactions.values())
    assert sum(r['task'] == 'relchem' for r in rows) == 251
    assert sum(r['task'] == 'ae17' for r in rows) == 17
    systems = sorted(bundle.systems)
    assert len(systems) == 90
    return [{'cursor': i, **{task: sample_pair(rows, task, i) for task in ('relchem', 'ae17')},
             'mrks_id': _cycle_permutation(systems, 41, 'mrks:identity', i // 90)[i % 90]}
            for i in range(count)]


def model_at(state):
    assert file_sha256(state['path']) == state['file_sha256']
    model = existing._pilot_model(torch.device('cuda'), torch.float32)
    model.load_state_dict(torch.load(state['path'], map_location='cpu', weights_only=False)['model'])
    assert existing.digest(model) == state['state_sha256']
    return model, copy.deepcopy(model).double()


def chemistry(model, shadow, reaction, dispersion):
    # The production evaluator releases each singleton graph before the next task.
    return existing.core.ChemistryBatchObjective(model, shadow, (reaction,), (1.0,), dispersion)


def measure(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk_size=None):
    times, losses, raw, grids = {}, {}, {}, {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for task in ('relchem', 'ae17'):
        timer = time.perf_counter()
        row = entry[task]
        reaction = bundle.chemistry('train_' + task).load_variant(row['identity'], row['variant'])
        # Transfers are measured separately without altering dtype/precision boundaries.
        reaction = existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
        if len(reaction['Grid']) > 131072:
            reaction['model_point_chunk_size'] = 16384
        torch.cuda.synchronize()
        times['chemistry_loading_transfers'] = times.get('chemistry_loading_transfers', 0) + time.perf_counter() - timer
        grids[task] = len(reaction['Grid'])
        timer = time.perf_counter()
        loss, gradient = chemistry(model, shadow, reaction, dispersion).value_and_grad()
        torch.cuda.synchronize()
        times[task] = time.perf_counter() - timer
        losses[task], raw[task] = loss, gradient
        del reaction
    timer = time.perf_counter()
    system = bundle.mrks().operator_system(entry['mrks_id'], device='cuda', dtype=torch.float32, chunk_size=4096)
    torch.cuda.synchronize()
    times['mrks_loading_transfers'] = time.perf_counter() - timer
    exc, op = existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256,
                                                      dispersions=mrks_dispersion, exc_chunk_size=exc_chunk_size)
    for task, objective in (('exc', exc), ('op', op)):
        timer = time.perf_counter()
        value, gradients = existing.core.compute_isolated_task_gradients(model, {task: objective}, task_order=(task,))
        materialized = existing.core.materialize_task_zeros(model, gradients, task_order=(task,))
        torch.cuda.synchronize()
        times[task] = time.perf_counter() - timer
        losses.update(value)
        raw[task] = {n: g.double() for n, g in materialized[task].items()}
    norms = {task: float(torch.cat([g.flatten() for g in raw[task].values()]).norm()) for task in TASKS}
    assert all(np.isfinite(value) and value > 0 for value in norms.values())
    record = {'sample': entry, 'system': system.name, 'losses': losses, 'norms': norms,
              'seconds': times, 'grid_points': grids, 'gradient_elapsed_seconds': time.perf_counter() - began,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    return record, raw


def parity(model, shadow, bundle, entry, dispersion):
    results = {}
    for task in ('relchem', 'ae17'):
        row = entry[task]
        reaction = bundle.chemistry('train_' + task).load_variant(row['identity'], row['variant'])
        value, gradient = chemistry(model, shadow, reaction, dispersion).value_and_grad()
        parameters = existing.named_trainable_parameters(shadow)
        direct = existing.make_reaction_objective(shadow, reaction, device='cuda', dtype=torch.float64, dispersions=dispersion)()
        derivative = torch.autograd.grad(direct, tuple(parameters.values()), allow_unused=True)
        bitwise = all(torch.equal(gradient[name], torch.zeros_like(p) if g is None else g)
                      for (name, p), g in zip(parameters.items(), derivative, strict=True))
        assert bitwise and value == float(direct.detach())
        results[task] = {'loss_exact': True, 'gradient_bitwise': bitwise}
        del direct, derivative, reaction
    return results


def save(path, model, optimizer, cursor, manifest_sha, calibration, scheduler=None, logs=(), total_updates=None):
    temporary = path.with_suffix('.tmp')
    torch.save({'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': existing.capture_rng_state(),
                'cursor': cursor, 'manifest_sha256': manifest_sha, 'calibration': calibration,
                'scheduler': None if scheduler is None else scheduler.state_dict(),
                'logs': list(logs), 'total_updates': total_updates}, temporary)
    temporary.replace(path)


def restore(path, model, optimizer, manifest_sha, calibration, scheduler=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['manifest_sha256'] == manifest_sha and saved['calibration'] == calibration
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    if saved.get('scheduler') is not None:
        if scheduler is None:
            raise ValueError('Scheduled checkpoint requires the same scheduler')
        scheduler.load_state_dict(saved['scheduler'])
    existing.restore_rng_state(saved['rng'])
    return saved['cursor']


def profile(folder):
    if folder.exists():
        raise RuntimeError('Use a fresh diagnostic folder; do not overwrite a completed run')
    folder.mkdir(parents=True)
    prior = read(OLD / 'protocol.json')
    old_checkpoint_sha = file_sha256(OLD / 'adamw/latest.pt')
    bundle = PublicationDataset(DATA)
    assert bundle.manifest['logical_sha256'] == DATA_SHA
    manifest = sampling(bundle)
    write(folder / 'sampling_manifest.json', manifest)
    manifest_sha = file_sha256(folder / 'sampling_manifest.json')
    protocol = {'task_order': list(TASKS), 'initial_state': prior['initial_state'], 'dataset_sha256': DATA_SHA,
                'sampling_manifest_sha256': manifest_sha, 'profile_indices': list(PROFILE_INDICES),
                'chemistry': 'independent uniform identity and uniform variant; no SVRG',
                'coefficients': 'new protocol: median three frozen minibatch gradient norms; epsilon=1e-12, C=4',
                'adamw': prior['adamw'], 'lr': 1e-6, 'retained_training_updates': 0,
                'source_sha256': file_sha256(__file__)}
    write(folder / 'protocol.json', protocol)
    torch.manual_seed(41)
    np.random.seed(41)
    random.seed(41)
    model, shadow = model_at(prior['initial_state'])
    initial = copy.deepcopy(model.state_dict())
    initial_rng = existing.capture_rng_state()
    dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = read(DATA / 'mrks/dispersion.json')
    checks = parity(model, shadow, bundle, manifest[PROFILE_INDICES[0]], dispersion)
    records, cache = [], []
    for index in PROFILE_INDICES:
        record, raw = measure(model, shadow, bundle, manifest[index], dispersion, mrks_dispersion)
        cache.append({task: {n: g.cpu() for n, g in values.items()} for task, values in raw.items()})
        records.append(record)
        del raw
        print('MICROBATCH_GRADIENTS', index, record['seconds'], record['peak_allocated_bytes'], flush=True)
    scales, coefficients = fixed_coefficients([[r['norms'][task] for task in TASKS] for r in records])
    calibration = {'scales': scales, 'lambda': coefficients, 'epsilon': 1e-12, 'C': 4.0,
                   'manifest_sha256': manifest_sha, 'sample_indices': list(PROFILE_INDICES),
                   'initial_state_sha256': existing.digest(model)}
    write(folder / 'calibration.json', calibration)
    optimizer = torch.optim.AdamW(existing.named_trainable_parameters(model).values(), lr=1e-6,
                                 **{**prior['adamw'], 'betas': tuple(prior['adamw']['betas'])})
    for record, cached in zip(records, cache, strict=True):
        model.load_state_dict(initial)
        optimizer.state.clear()
        raw = {task: {n: g.cuda() for n, g in values.items()} for task, values in cached.items()}
        torch.cuda.synchronize()
        timer = time.perf_counter()
        joint = adamw_step(model, optimizer, raw, coefficients)
        torch.cuda.synchronize()
        record['seconds']['weighted_aggregation_adamw'] = time.perf_counter() - timer
        record['total_seconds'] = sum(record['seconds'].values())
        displacement = torch.cat([(model.state_dict()[n] - initial[n]).double().flatten() for n in joint]).norm()
        record['displacement_norm'] = float(displacement)
        reconstructed = sum(coefficients[t] * torch.cat([v.flatten() for v in raw[t].values()]) for t in TASKS)
        assert torch.equal(torch.cat([g.flatten() for g in joint.values()]), reconstructed)
    # Exact real checkpoint/resume proof with two ordinary native AdamW steps.
    save(folder / 'diagnostic_checkpoint.pt', model, optimizer, 1, manifest_sha, calibration)
    rng = existing.capture_rng_state()
    adamw_step(model, optimizer, raw, coefficients)
    expected = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    assert restore(folder / 'diagnostic_checkpoint.pt', model, optimizer, manifest_sha, calibration) == 1
    assert existing.equal(existing.capture_rng_state(), rng)
    adamw_step(model, optimizer, raw, coefficients)
    assert existing.equal(model.state_dict(), expected) and existing.equal(optimizer.state_dict(), expected_optimizer)
    model.load_state_dict(initial)
    existing.restore_rng_state(initial_rng)
    assert existing.digest(model) == prior['initial_state']['state_sha256']
    assert file_sha256(OLD / 'adamw/latest.pt') == old_checkpoint_sha
    write(folder / 'metrics.json', {'records': records, 'calibration': calibration, 'parity': checks,
          'aggregation_exact': True, 'checkpoint_resume_exact': True, 'restoration_exact': True,
          'old_cursor7_checkpoint_sha256': old_checkpoint_sha, 'retained_updates': 0,
          'new_full251_reference_computations': 0, 'svrg_used': False,
          'physical_gpu_bytes': torch.cuda.get_device_properties(0).total_memory})
    bundle.close()
    print('MICROBATCH_PROFILE_COMPLETE', folder, flush=True)


def adamw_diagnostics(model, optimizer, raw, before, initial):
    """Actual rounded parameter movement and native moments; no update policy."""
    parameters = existing.named_trainable_parameters(model)
    current = torch.cat([p.detach().flatten().double() for p in parameters.values()])
    descent = before - current
    length = float(descent.norm())
    matrix = torch.stack([torch.cat([g.flatten().double() for g in raw[t].values()]) for t in TASKS])
    moments = {}
    for name in ('exp_avg', 'exp_avg_sq'):
        vector = torch.cat([optimizer.state[p][name].flatten().double() for p in parameters.values()])
        if not torch.isfinite(vector).all():
            raise FloatingPointError('Nonfinite native AdamW moments')
        moments[name] = {'norm': float(vector.norm()), 'max_abs': float(vector.abs().max())}
    return {'step_norm': length, 'parameter_displacement_from_t0': float((current - initial).norm()),
            'relative_parameter_displacement_from_t0': float((current - initial).norm() / initial.norm()),
            'task_predicted_loss_change_actual_step': (-matrix @ descent).tolist(),
            'task_progress_actual_step': None if length == 0 else
                ((matrix @ descent) / (matrix.norm(dim=1) * length)).tolist(),
            'adamw_moments': moments,
            'moment_step': int(next(iter(optimizer.state.values()))['step'])}


def train(folder, updates, stop_at=None, runtime_seconds=3600, *,
          learning_rate=1e-6, constant_lr=False, diagnostics=False):
    """Explicit bounded opt-in; never called by profiling or qualification."""
    torch.manual_seed(41)
    np.random.seed(41)
    random.seed(41)
    protocol, calibration = read(folder / 'protocol.json'), read(folder / 'calibration.json')
    manifest = read(folder / 'sampling_manifest.json')
    manifest_sha = file_sha256(folder / 'sampling_manifest.json')
    if not 0 < updates <= len(manifest):
        raise ValueError('Updates must fit the frozen manifest; no automatic extension')
    if manifest_sha != calibration['manifest_sha256']:
        raise ValueError('Frozen sampling/calibration mismatch')
    bundle = PublicationDataset(DATA)
    assert bundle.manifest['logical_sha256'] == protocol['dataset_sha256'] == DATA_SHA
    model, shadow = model_at(protocol['initial_state'])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive finite learning rate required')
    optimizer = torch.optim.AdamW(existing.named_trainable_parameters(model).values(), lr=learning_rate,
                                 **{**protocol['adamw'], 'betas': tuple(protocol['adamw']['betas'])})
    scheduler = None if constant_lr else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=1e-7)
    arm = folder / 'ordinary_sgd_adamw'
    arm.mkdir(exist_ok=True)
    run_protocol = {'updates': updates, 'lr': learning_rate,
                    'scheduler': 'constant' if constant_lr else 'cosine to1e-7',
                    'exc_chunk_size': protocol.get('exc_chunk_size', 256), 'operator_chunk_size': 256,
                    'chemistry_model_chunk': 16384, 'chunk_threshold': 131072,
                    'manifest_sha256': manifest_sha, 'calibration_sha256': file_sha256(folder / 'calibration.json'),
                    'source_sha256': file_sha256(__file__), 'dataset_sha256': DATA_SHA,
                    'physics_sources_sha256': {name: file_sha256(ROOT / 'train_models' / name)
                                              for name in ('lap_training.py', 'lap_moo_training.py', 'lap_operator.py',
                                                           'lap_fixed_adamw.py')}}
    if diagnostics:
        run_protocol['diagnostics'] = 'Actual native AdamW movement/moments and SHA-bound raw gradients'
    if (arm / 'protocol.json').exists():
        assert read(arm / 'protocol.json') == run_protocol
    else:
        write(arm / 'protocol.json', run_protocol)
    if diagnostics:
        (arm / 'raw_gradients').mkdir(exist_ok=True)
        write(arm / 'parameter_order.json', [{'name': n, 'shape': list(p.shape), 'numel': p.numel()}
              for n, p in existing.named_trainable_parameters(model).items()])
    initial = torch.cat([p.detach().flatten().double() for p in existing.named_trainable_parameters(model).values()])
    latest = arm / 'latest.pt'
    logs, cursor = [], 0
    if latest.exists():
        prior = torch.load(latest, map_location='cpu', weights_only=False)
        assert prior['total_updates'] == updates
        cursor = restore(latest, model, optimizer, manifest_sha, calibration, scheduler)
        logs = prior['logs']
    else:
        save(latest, model, optimizer, 0, manifest_sha, calibration, scheduler, logs, updates)
        save(arm / 'checkpoint_0.pt', model, optimizer, 0, manifest_sha, calibration, scheduler, logs, updates)
    dispersion, mrks_dispersion = bundle.chemistry_dispersions(), read(DATA / 'mrks/dispersion.json')
    began = time.perf_counter()
    try:
        for index in range(cursor, min(updates, stop_at if stop_at is not None else updates)):
            if time.perf_counter() - began >= runtime_seconds:
                print('SAFE_RUNTIME_PAUSE', index, flush=True)
                break
            before = existing.digest(model)
            before_vector = torch.cat([p.detach().flatten().double()
                for p in existing.named_trainable_parameters(model).values()]) if diagnostics else None
            record, raw = measure(model, shadow, bundle, manifest[index], dispersion, mrks_dispersion,
                                  run_protocol['exc_chunk_size'])
            torch.cuda.synchronize()
            timer = time.perf_counter()
            step_learning_rate = optimizer.param_groups[0]['lr']
            joint = adamw_step(model, optimizer, raw, calibration['lambda'])
            if scheduler is not None:
                scheduler.step()
            torch.cuda.synchronize()
            record['seconds']['weighted_aggregation_adamw'] = time.perf_counter() - timer
            record.update(total_seconds=sum(record['seconds'].values()), before_sha256=before,
                          after_sha256=existing.digest(model), learning_rate=step_learning_rate,
                          weighted_gradient_norm=float(torch.cat([g.flatten() for g in joint.values()]).norm()))
            if diagnostics:
                timer = time.perf_counter()
                record.update(adamw_diagnostics(model, optimizer, raw, before_vector, initial))
                path = arm / 'raw_gradients' / f'update_{index:03}.npz'
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez(stream, **{t: torch.cat([g.flatten() for g in raw[t].values()]).cpu().numpy()
                                       for t in TASKS})
                temporary.replace(path)
                record['raw_gradients_sha256'] = file_sha256(path)
                torch.cuda.synchronize()
                record['diagnostic_seconds'] = time.perf_counter() - timer
                record['total_seconds'] += record['diagnostic_seconds']
                record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            logs.append(record)
            save(latest, model, optimizer, index + 1, manifest_sha, calibration, scheduler, logs, updates)
            if index + 1 in (10, 20, 45, 90):
                save(arm / f'checkpoint_{index + 1}.pt', model, optimizer, index + 1,
                     manifest_sha, calibration, scheduler, logs, updates)
            print('ADAMW_UPDATE', index + 1, record['total_seconds'], flush=True)
            del raw, joint
    finally:
        bundle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--train-updates', type=int, help='Explicit opt-in after profiling; no evaluation or automatic extension')
    args = parser.parse_args()
    torch.set_num_threads(1)
    import os
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single-GPU protocol: do not silently double task batches/weights')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.train_updates is None:
        profile(args.output)
    else:
        train(args.output, args.train_updates)
