"""Fixed learning-rate continuation from B8 update 128, with a parameter-only SWA readout.

One new arm. At most 251 AdamW updates. The B8 t128 moments stay. The learning rate
is 0.25 times the B8 rate and does not change. SWA reads post-update snapshots and
does not enter the forward, backward, or optimizer path.
"""
import copy
import hashlib
import math
import subprocess
import time
from pathlib import Path

import lap_b8_adamw as b8
import torch

PROTOCOL_ID = 'b8-fixed-lr-swa-v1'
PARENT_CURSOR = 128
PARENT_LR = 1e-4
LR_FACTOR = 0.25
NEW_LR = LR_FACTOR * PARENT_LR
NEW_UPDATES = 251
RAW_CHECKPOINTS = (64, 128, 192, 251)
SWA_LOCAL = tuple(range(32, 249, 8)) + (251,)
HIGHLIGHTS = ('SIE4x4-15', 'WCPT18-15', 'HEAVY28-16', 'BSR36-31', 'BHPERI-11')
DATABASES = ('DBH76', 'pTC13', 'NCCE31', 'IP13', 'ABDE4', 'MGAE109', 'EA13', 'PA8')
B8_T128_CLEAN28 = 9.096700111658418
OUT = b8.REPO.parent / 'lap_b8_swa_continuation_20261010'
PARENT_CHECKPOINT = b8.OUT / 'ordinary_sgd_adamw' / 'checkpoint_128.pt'
PARENT_MANIFEST = b8.OUT / 'b8_manifest.json'
PARENT_VALIDATION = b8.OUT / 'validation_128.json'
MANIFEST_SHA = 'b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc'


def swa_expected_count(local_update):
    return sum(point <= local_update for point in SWA_LOCAL)


def moment_sha(optimizer_state):
    digest = hashlib.sha256()
    for key in sorted(optimizer_state['state']):
        state = optimizer_state['state'][key]
        digest.update(str(key).encode())
        digest.update(str(int(state['step'])).encode())
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def blank_swa(parameters):
    return {
        name: torch.zeros_like(parameter.detach(), dtype=torch.float64, device='cpu')
        for name, parameter in parameters.items()
    }


def observe_swa(total, parameters):
    """Add a detached float64 copy. The live parameter storage is not written."""
    for name, parameter in parameters.items():
        before = parameter.detach().cpu().clone()
        if parameter.dtype != torch.float32 or name not in total:
            raise ValueError('SWA can accumulate only the original float32 trainable parameters')
        total[name].add_(parameter.detach().double().cpu())
        if not torch.equal(parameter.detach().cpu(), before) or parameter.dtype != torch.float32:
            raise RuntimeError('SWA modified a live parameter')
    return total


def swa_state(model_state, total, count, parameter_dtypes):
    """Average trainable parameters. Buffers and frozen tensors stay as copied."""
    if count <= 0:
        raise ValueError('SWA checkpoint needs a positive snapshot count')
    state = {name: value.detach().cpu().clone() for name, value in model_state.items()}
    for name, dtype in parameter_dtypes.items():
        restored = torch.empty_like(state[name], dtype=dtype, device='cpu')
        restored.copy_(total[name] / count)
        if restored.dtype != dtype:
            raise ValueError('SWA did not restore the checkpoint dtype')
        state[name] = restored
    return state


def inspect_optimizer_state(saved_state, expected_step, source_lr):
    """Reject a missing, reset, or mismatched AdamW state before it is loaded."""
    if not saved_state.get('state'):
        raise ValueError('Exact compatible optimizer state is unavailable')
    group = saved_state['param_groups'][0]
    if group['lr'] != source_lr or tuple(group['betas']) != (0.9, 0.999):
        raise ValueError('Optimizer learning rate or betas do not match the source checkpoint')
    if group['weight_decay'] != 0.01 or group['eps'] != 1e-8 or group.get('foreach') is not False:
        raise ValueError('Optimizer weight decay, epsilon, or foreach differs from B8')
    steps = {int(item['step']) for item in saved_state['state'].values()}
    if steps != {expected_step}:
        raise ValueError('Optimizer step counter does not match the checkpoint')
    if not any(torch.count_nonzero(item['exp_avg']) for item in saved_state['state'].values()):
        raise ValueError('Optimizer moments are missing')
    return moment_sha(saved_state)


def prepare_loaded_optimizer(optimizer, saved_state, expected_step, source_lr):
    """Load moments unchanged, then set the continuation learning rate."""
    before = inspect_optimizer_state(saved_state, expected_step, source_lr)
    optimizer.load_state_dict(copy.deepcopy(saved_state))
    if moment_sha(optimizer.state_dict()) != before:
        raise ValueError('Loading the optimizer changed its moments')
    for loaded in optimizer.param_groups:
        loaded['lr'] = NEW_LR
    if {int(item['step']) for item in optimizer.state.values()} != {expected_step}:
        raise ValueError('Setting the learning rate changed the optimizer step')
    if any(loaded['lr'] != NEW_LR for loaded in optimizer.param_groups):
        raise ValueError('Continuation learning rate was not applied')
    return before


def classify(rows, ratios):
    """GO only when a finished candidate clears both the accuracy gate and all four ratios."""
    if len(rows) != 5 or any(not math.isfinite(row['clean28']) for row in rows):
        raise ValueError('Classification needs five finite Clean28 evaluations')
    passed = False
    for row in rows:
        if not b8.beats_historical(row['clean28']):
            continue
        if row['name'] not in ratios:
            raise ValueError('Accuracy-gate candidate has no scientific ratios')
        if b8.scientifically_eligible(ratios[row['name']]):
            passed = True
    return 'GO' if passed else 'NO-GO'


def select_lowest(rows):
    return min(rows, key=lambda row: (row['clean28'], row['order']))


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason})
    raise SystemExit(reason)


def _confine(path):
    path = Path(path)
    resolved = path.resolve()
    if not resolved.is_relative_to(OUT.resolve()):
        raise ValueError('Refusing to write outside the continuation scratch')
    if resolved.is_relative_to(b8.OUT.resolve()):
        raise ValueError('Refusing to write into the completed B8 scratch')
    return path


def _new_optimizer(parameters):
    return torch.optim.AdamW(
        list(parameters.values()), lr=NEW_LR, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, foreach=False)


def _check_model(model):
    run = b8._import_run()
    parameters = run.existing.named_trainable_parameters(model)
    coordinates = sum(parameter.numel() for parameter in parameters.values())
    version = int(model.lap_architecture_version)
    if (type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or coordinates != 9446 or version != 1
            or any('tau' in name.lower() for name in parameters)):
        _stop('Model architecture does not match the qualified checkpoint')
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        _stop('Trainable parameters are not float32')
    return parameters


def _load_parent_weights(model, saved):
    model.load_state_dict(saved['model'])
    parameters = _check_model(model)
    optimizer = _new_optimizer(parameters)
    try:
        digest = prepare_loaded_optimizer(optimizer, saved['optimizer'], PARENT_CURSOR, PARENT_LR)
    except (ValueError, RuntimeError) as error:
        _stop('Preflight NO-GO: ' + str(error))
    return parameters, optimizer, digest


def _finite_map(grad):
    return all(bool(torch.isfinite(piece).all()) for piece in grad.values())


def _parameter_finite(parameters):
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters.values())


def _vector_delta(current, reference):
    parts = []
    for name in reference:
        left = current[name].detach().double().cpu()
        right = reference[name].detach().double().cpu()
        parts.append((left - right).reshape(-1))
    return float(torch.linalg.vector_norm(torch.cat(parts)))


def _state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _origin_weights(parameter_names):
    saved = torch.load(OUT / 'ordinary_sgd_adamw' / 'checkpoint_0.pt', map_location='cpu', weights_only=False)
    origin = {name: saved['model'][name].detach().cpu().clone() for name in parameter_names}
    return origin, _state_digest(saved['model'])


def save_checkpoint(path, model, optimizer, local_update, logs, swa_total, swa_count, protocol_sha, parent_sha):
    path = _confine(path)
    payload = {
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'rng': b8._capture_rng(),
        'local_update': local_update,
        'cumulative_optimizer_step': PARENT_CURSOR + local_update,
        'logs': list(logs),
        'swa_sum': {name: value.detach().cpu().clone() for name, value in swa_total.items()},
        'swa_count': swa_count,
        'swa_schedule': list(SWA_LOCAL),
        'protocol_id': PROTOCOL_ID,
        'manifest_sha256': MANIFEST_SHA,
        'protocol_sha256': protocol_sha,
        'parent_checkpoint_sha256': parent_sha,
        'parent_cursor': PARENT_CURSOR,
        'lambdas': dict(b8.LAMBDAS),
        'offsets': list(b8.OFFSETS),
        'lr': NEW_LR,
        'scheduler': None,
        'kind': 'adamw',
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('scheduler') is not None or saved.get('kind') != 'adamw':
        _stop('Resume checkpoint is not this continuation')
    if saved.get('manifest_sha256') != MANIFEST_SHA or saved.get('lr') != NEW_LR:
        _stop('Resume checkpoint does not match the frozen continuation protocol')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    b8._restore_rng(saved['rng'])
    local_update = int(saved['local_update'])
    steps = {int(state['step']) for state in optimizer.state.values()}
    if steps != {PARENT_CURSOR + local_update}:
        _stop('Resume would skip or replay a saved optimizer update')
    if int(saved['swa_count']) != swa_expected_count(local_update):
        _stop('Resume SWA count does not match the completed snapshots')
    if any(group['lr'] != NEW_LR for group in optimizer.param_groups):
        _stop('Resume learning rate is not the fixed continuation rate')
    return local_update, list(saved['logs']), saved['swa_sum'], int(saved['swa_count'])


def _protocol_payload(source, parent_sha):
    return {
        'protocol_id': PROTOCOL_ID,
        'parent_protocol_id': b8.PROTOCOL_ID,
        'parent_checkpoint': str(PARENT_CHECKPOINT),
        'parent_checkpoint_sha256': parent_sha,
        'parent_cursor': PARENT_CURSOR,
        'parent_lr': PARENT_LR,
        'lr_factor': LR_FACTOR,
        'lr': NEW_LR,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.01,
        'foreach': False,
        'clipping': None,
        'scheduler': None,
        'new_updates': NEW_UPDATES,
        'swa_local_updates': list(SWA_LOCAL),
        'swa_dtype': 'float64',
        'swa_restored_dtype': 'float32',
        'swa_targets': 'trainable parameters only',
        'lambdas': dict(b8.LAMBDAS),
        'offsets': list(b8.OFFSETS),
        'manifest_sha256': MANIFEST_SHA,
        'initial_state': source['initial_state'],
        'exc_chunk_size': 4096,
        'j251_retrained': False,
        'b8_retrained': False,
    }


def _write_protocol(source, parent_sha):
    payload = _protocol_payload(source, parent_sha)
    path = OUT / 'protocol.json'
    if path.is_file():
        if b8.read_json(path) != payload:
            _stop('Continuation protocol file changed')
        return b8.sha256_file(path)
    b8.write_json(path, payload)
    return b8.sha256_file(path)


def _batches():
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    stored = b8.read_json(PARENT_MANIFEST)
    if stored != batches:
        _stop('Parsed B8 manifest does not match the frozen batch builder')
    b8.assert_batch_properties(batches, rows)
    if b8.presentation_counts(batches) != {row['relchem']['identity']: 8 for row in rows}:
        _stop('Continuation traversal does not present every reaction eight times')
    return batches


def _swa_probe(model, parameters):
    run = b8._import_run()
    before = run.existing.digest(model)
    values = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    total = blank_swa(parameters)
    observe_swa(total, parameters)
    observe_swa(total, parameters)
    for name, parameter in parameters.items():
        if not torch.equal(parameter.detach(), values[name]):
            _stop('SWA probe modified live weights')
        mean = total[name] / 2
        if not torch.equal(mean, values[name].detach().double().cpu()):
            _stop('SWA probe average is not the arithmetic mean')
    state = swa_state(
        model.state_dict(), total, 2, {name: parameter.dtype for name, parameter in parameters.items()})
    if run.existing.digest(model) != before:
        _stop('Building an SWA state modified the live model')
    for name, parameter in parameters.items():
        if not torch.equal(state[name], parameter.detach().cpu()):
            _stop('SWA probe did not restore the trainable values')
    return True


def preflight(source, batches, parent_sha):
    run = b8._import_run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    parent_before = b8.sha256_file(PARENT_CHECKPOINT)
    if parent_before != parent_sha:
        _stop('Parent checkpoint hash changed before preflight')
    saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    model, shadow = run.model_at(source['initial_state'])
    parameters, optimizer, moment_digest = _load_parent_weights(model, saved)
    loaded_digest = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    rows = []
    peak_allocated = 0
    peak_reserved = 0
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        _swa_probe(model, parameters)
        if {int(state['step']) for state in optimizer.state.values()} != {PARENT_CURSOR}:
            _stop('SWA probe changed the optimizer step')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for index in b8.PARITY_INDICES:
            batch = batches[index]
            reference_loss, reference_grad, _reference_seconds = b8.reference_mean(
                model, shadow, bundle, batch['relchem'], dispersion)
            loss, grad, _infos, _seconds = b8.qualified_mean(
                model, shadow, bundle, batch['relchem'], dispersion)
            discrepancy = b8.relative_l2(grad, reference_grad)
            scalar_gap = abs(loss - reference_loss) / max(abs(reference_loss), 1e-30)
            if discrepancy > b8.RELATIVE_L2_LIMIT or scalar_gap > b8.RELATIVE_L2_LIMIT:
                _stop(f'Chemistry gradient parity failed at manifest index {index}')
            row = {'index': index, 'relative_l2': discrepancy, 'scalar_relative': scalar_gap}
            if index == 0:
                other_losses, other_raw, _system, _other_seconds = b8.other_tasks(
                    model, shadow, bundle, batch, dispersion, mrks_dispersion)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                _measured, measured_raw = run.measure(
                    model, shadow, bundle, b8.load_j251_rows()[0], dispersion, mrks_dispersion, exc_chunk_size=4096)
                other_l2 = {task: b8.relative_l2(other_raw[task], measured_raw[task]) for task in ('ae17', 'exc', 'op')}
                if any(value > b8.OTHER_TASK_L2_LIMIT for value in other_l2.values()):
                    _stop('AE17, Exc, or operator gradient differs from the B8 measurement path')
                raw = {'relchem': grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op']}
                pointers = {task: {name: piece.data_ptr() for name, piece in raw[task].items()} for task in b8.TASKS[1:]}
                joint = b8.scalarize(raw, b8.LAMBDAS)
                if any(raw[task][name].data_ptr() != pointers[task][name] for task in b8.TASKS[1:] for name in raw[task]):
                    _stop('Scalarization mutated a non-relchem gradient')
                manual = {
                    name: sum(raw[task][name].double() * b8.LAMBDAS[task] for task in b8.TASKS) for name in joint
                }
                coefficient_l2 = b8.relative_l2(joint, manual)
                if coefficient_l2 > b8.RELATIVE_L2_LIMIT:
                    _stop('Task coefficients were not applied exactly once')
                row.update({
                    'other_task_relative_l2': other_l2,
                    'coefficient_relative_l2': coefficient_l2,
                    'other_losses': other_losses,
                    'optimizer_step': False,
                })
                del measured_raw, joint, manual
            rows.append(row)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            del grad
            torch.cuda.empty_cache()
        if {int(state['step']) for state in optimizer.state.values()} != {PARENT_CURSOR}:
            _stop('Preflight took an optimizer step')
        if run.existing.digest(model) != loaded_digest or b8.sha256_file(PARENT_CHECKPOINT) != parent_before:
            _stop('Preflight changed the model or the B8 checkpoint')
    finally:
        bundle.close()
        del model, shadow, optimizer
        torch.cuda.empty_cache()
    receipt = {
        'passed': True,
        'optimizer_step': False,
        'gpu_seconds': time.perf_counter() - started,
        'parent_checkpoint_sha256': parent_sha,
        'optimizer_moment_sha256': moment_digest,
        'loaded_model_sha256': loaded_digest,
        'trainable_coordinates': 9446,
        'parent_cursor': PARENT_CURSOR,
        'parameter_dtype': 'float32',
        'lr': NEW_LR,
        'batches': rows,
        'swa_probe': True,
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
    }
    b8.write_json(OUT / 'preflight.json', receipt)
    print('SWA_PREFLIGHT', receipt['gpu_seconds'], flush=True)
    return receipt


def _log_row(local_update, batch, infos, relchem_loss, relchem_grad, other_losses, other_raw, joint,
             update_norm, displacement, before, after, seconds, swa_count):
    norms = [row['grad_norm'] for row in infos]
    losses = [row['loss'] for row in infos]
    mean_of_norms = sum(norms) / len(norms)
    norm_of_mean = b8._grad_norm(relchem_grad)
    return {
        'local_update': local_update,
        'cumulative_optimizer_step': PARENT_CURSOR + local_update,
        'manifest_index': batch['cursor'],
        'reactions': infos,
        'loss_mean': relchem_loss,
        'loss_min': min(losses),
        'loss_max': max(losses),
        'ae17_loss': other_losses['ae17'],
        'exc_loss': other_losses['exc'],
        'op_loss': other_losses['op'],
        'ae17_norm': b8._grad_norm(other_raw['ae17']),
        'exc_norm': b8._grad_norm(other_raw['exc']),
        'op_norm': b8._grad_norm(other_raw['op']),
        'norm_of_mean': norm_of_mean,
        'mean_of_norms': mean_of_norms,
        'norm_ratio': norm_of_mean / mean_of_norms if mean_of_norms else None,
        'joint_norm': b8._grad_norm(joint),
        'update_norm': update_norm,
        'displacement_from_t128': displacement,
        'lr': NEW_LR,
        'seconds': seconds,
        'before_sha256': before,
        'after_sha256': after,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'finite': True,
        'divided_by': 8,
        'swa_count': swa_count,
        'presentations': 8,
    }


def _save_swa(model, parameters, total, count, parent_sha, protocol_sha):
    path = _confine(OUT / 'swa_251.pt')
    state = swa_state(
        model.state_dict(), total, count, {name: parameter.dtype for name, parameter in parameters.items()})
    payload = {
        'model': state,
        'kind': 'swa',
        'local_update': NEW_UPDATES,
        'swa_count': count,
        'swa_schedule': list(SWA_LOCAL),
        'accumulation_dtype': 'float64',
        'restored_dtype': 'float32',
        'optimizer_averaged': False,
        'buffers_averaged': False,
        'protocol_id': PROTOCOL_ID,
        'manifest_sha256': MANIFEST_SHA,
        'protocol_sha256': protocol_sha,
        'parent_checkpoint_sha256': parent_sha,
        'scheduler': None,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(source, batches, protocol_sha, parent_sha):
    run = b8._import_run()
    target = OUT / 'ordinary_sgd_adamw'
    target.mkdir(parents=True, exist_ok=True)
    latest = target / 'latest.pt'
    model, shadow = run.model_at(source['initial_state'])
    if latest.is_file():
        parameters = _check_model(model)
        optimizer = _new_optimizer(parameters)
        local_update, logs, swa_total, swa_count = restore_checkpoint(latest, model, optimizer)
        parameters = _check_model(model)
        if local_update != len(logs):
            _stop('Checkpoint cursor does not match its log')
        for index in range(len(logs) - 1):
            if logs[index]['after_sha256'] != logs[index + 1]['before_sha256']:
                _stop('Model hash chain broke')
    else:
        saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
        parameters, optimizer, _digest = _load_parent_weights(model, saved)
        logs = []
        local_update = 0
        swa_total = blank_swa(parameters)
        swa_count = 0
        save_checkpoint(target / 'checkpoint_0.pt', model, optimizer, 0, logs, swa_total, swa_count, protocol_sha, parent_sha)
        save_checkpoint(latest, model, optimizer, 0, logs, swa_total, swa_count, protocol_sha, parent_sha)
    origin, origin_digest = _origin_weights(parameters)
    if local_update == 0:
        if run.existing.digest(model) != origin_digest:
            _stop('Initialization digest does not match checkpoint 0')
    elif logs[0]['before_sha256'] != origin_digest:
        _stop('Training did not start from B8 t128')
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for index in range(local_update, NEW_UPDATES):
            batch = batches[index]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            before = run.existing.digest(model)
            before_weights = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            relchem_loss, relchem_grad, infos, _elapsed = b8.qualified_mean(
                model, shadow, bundle, batch['relchem'], dispersion)
            other_losses, other_raw, _system, _other = b8.other_tasks(
                model, shadow, bundle, batch, dispersion, mrks_dispersion)
            raw = {
                'relchem': relchem_grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op'],
            }
            if any(not math.isfinite(value) for value in (relchem_loss, *other_losses.values())):
                _stop('Nonfinite loss')
            if not _finite_map(relchem_grad) or any(not _finite_map(other_raw[task]) for task in ('ae17', 'exc', 'op')):
                _stop('Nonfinite gradient')
            joint = run.adamw_step(model, optimizer, raw, b8.LAMBDAS)
            done = index + 1
            steps = {int(state['step']) for state in optimizer.state.values()}
            if steps != {PARENT_CURSOR + done} or any(group['lr'] != NEW_LR for group in optimizer.param_groups):
                _stop('AdamW did not take exactly one fixed-rate step')
            if not _parameter_finite(parameters) or not _finite_map(joint):
                _stop('Nonfinite parameter or joint gradient')
            if done in SWA_LOCAL:
                observe_swa(swa_total, parameters)
                swa_count += 1
            if swa_count != swa_expected_count(done):
                _stop('SWA snapshot count drifted')
            update_norm = _vector_delta(parameters, before_weights)
            displacement = _vector_delta(parameters, origin)
            torch.cuda.synchronize()
            row = _log_row(
                done, batch, infos, relchem_loss, relchem_grad, other_losses, other_raw, joint,
                update_norm, displacement, before, run.existing.digest(model),
                time.perf_counter() - started, swa_count)
            logs.append(row)
            del raw, joint, relchem_grad, other_raw, before_weights
            torch.cuda.empty_cache()
            save_checkpoint(latest, model, optimizer, done, logs, swa_total, swa_count, protocol_sha, parent_sha)
            milestone = target / f'checkpoint_{done}.pt'
            if done in RAW_CHECKPOINTS and not milestone.exists():
                save_checkpoint(milestone, model, optimizer, done, logs, swa_total, swa_count, protocol_sha, parent_sha)
            if done == NEW_UPDATES:
                _save_swa(model, parameters, swa_total, swa_count, parent_sha, protocol_sha)
            b8.write_json(OUT / 'training_status.json', {
                'local_update': done,
                'cumulative_optimizer_step': PARENT_CURSOR + done,
                'complete': done == NEW_UPDATES,
                'swa_count': swa_count,
                'logs': logs,
            })
            print('SWA_UPDATE', done, PARENT_CURSOR + done, f'{row["seconds"]:.3f}', f'{row["loss_mean"]:.6f}', swa_count, flush=True)
    finally:
        bundle.close()
    if b8.sha256_file(PARENT_CHECKPOINT) != parent_sha:
        _stop('Training changed the B8 checkpoint')
    return local_update if not logs else logs[-1]['local_update']


def _load_eval_model(source, path):
    run = b8._import_run()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model, _shadow = run.model_at(source['initial_state'])
    model.load_state_dict(saved['model'])
    return model, run


def _evaluate_one(source, bundle, path, name, order, cursor):
    run = b8._import_run()
    model, _run = _load_eval_model(source, path)
    before = run.existing.digest(model)
    torch.cuda.synchronize()
    started = time.perf_counter()
    from tools.evaluate_microbatch_endpoint import validation
    result = validation(model, bundle)
    torch.cuda.synchronize()
    if run.existing.digest(model) != before:
        _stop('Validation changed the checkpoint weights')
    rows = [row for row in result['reaction_rows'] if row['clean']]
    if len(rows) != 28:
        _stop('Clean28 did not contain 28 reactions')
    payload = {
        'name': name, 'order': order, 'local_update': cursor, 'clean28': result['clean28'],
        'reaction_rows': rows, 'full30_discarded': True, 'full30_used_for_selection': False,
        'gpu_seconds': time.perf_counter() - started, 'optimizer_step': False,
        'checkpoint_sha256': b8.sha256_file(path), 'kind': 'swa' if name.startswith('swa') else 'adamw',
    }
    b8.write_json(OUT / f'validation_{name}.json', payload)
    print('SWA_CLEAN28', name, result['clean28'], flush=True)
    del model
    torch.cuda.empty_cache()
    return payload


def evaluate_clean28(source):
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    specs = [(f'raw_{cursor}', order, cursor, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt')
             for order, cursor in enumerate(RAW_CHECKPOINTS)]
    specs.append(('swa_251', 4, 251, OUT / 'swa_251.pt'))
    try:
        for name, order, cursor, path in specs:
            receipt_path = OUT / f'validation_{name}.json'
            if receipt_path.is_file():
                continue
            if not path.is_file():
                _stop('Preregistered checkpoint is missing: ' + name)
            _evaluate_one(source, bundle, path, name, order, cursor)
    finally:
        bundle.close()


def _chemistry_rows(source, path, task):
    run = b8._import_run()
    evaluation = b8.read_json(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    if evaluation['policy'] != 'one-variant-per-identity-v1':
        _stop('Evaluation manifest policy changed')
    model, shadow = run.model_at(source['initial_state'])
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    rows = []
    started = time.perf_counter()
    try:
        dispersion = bundle.chemistry_dispersions()
        selected = [row for row in evaluation['rows'] if row['task'] == task]
        expected = 251 if task == 'relchem' else 17
        if len(selected) != expected:
            _stop('Evaluation manifest does not contain one variant per identity')
        for spec in selected:
            reaction = b8._load_reaction(bundle, task, {
                'identity': spec['identity'], 'variant': spec['variant'],
                'database': bundle.reactions[spec['identity']]['database'],
            })
            with torch.no_grad():
                value = float(run.chemistry(model, shadow, reaction, dispersion)())
            if not math.isfinite(value):
                _stop('Nonfinite chemistry scalar')
            rows.append({
                'identity': spec['identity'], 'variant': spec['variant'], 'task': task,
                'database': bundle.reactions[spec['identity']]['database'], 'loss': value,
            })
            del reaction
        if run.existing.digest(model) != before:
            _stop('Scalar evaluation changed the checkpoint weights')
    finally:
        bundle.close()
    objective = sum(row['loss'] for row in rows) / len(rows)
    return {
        'task': task, 'objective': objective, 'ratio': objective / b8.BASELINE[task],
        'rows': rows, 'gpu_seconds': time.perf_counter() - started, 'gradients': False,
    }


def _mrks_objectives(source, path):
    run = b8._import_run()
    model, _shadow = _load_eval_model(source, path)
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    rows = {}
    started = time.perf_counter()
    try:
        dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for identity in sorted(bundle.systems):
            system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
            exc, op = run.existing.core.make_mrks_objective_factories(
                model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=dispersion)
            values = {'exc': float(exc().detach()), 'op': float(op().detach())}
            if any(not math.isfinite(value) for value in values.values()):
                _stop('Nonfinite mRKS scalar')
            rows[identity] = values
            del system, exc, op
            torch.cuda.empty_cache()
        if run.existing.digest(model) != before or len(rows) != 90:
            _stop('mRKS evaluation changed the model or did not cover 90 systems')
    finally:
        bundle.close()
    return {
        task: sum(row[task] for row in rows.values()) / 90 for task in ('exc', 'op')
    } | {'gpu_seconds': time.perf_counter() - started, 'gradients': False, 'systems': 90}


def _database_report(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['database'], []).append(row['loss'])
    report = {}
    for name, values in sorted(grouped.items()):
        report[name] = {
            'count': len(values),
            'mean': sum(values) / len(values),
            'contribution': sum(values) / 251,
        }
    return report


def diagnose_and_qualify(source, selected):
    path = _candidate_path(selected)
    diagnostic_path = OUT / 'diagnostic_relchem.json'
    if not diagnostic_path.is_file():
        scalar = _chemistry_rows(source, path, 'relchem')
        payload = {
            'name': selected['name'],
            'local_update': selected['local_update'],
            'kind': selected['kind'],
            'objective': scalar['objective'],
            'ratio': scalar['ratio'],
            'reference': b8.BASELINE['relchem'],
            'databases': _database_report(scalar['rows']),
            'gradients': False,
            'gpu_seconds': scalar['gpu_seconds'],
            'role': 'diagnostic_not_qualification',
        }
        b8.write_json(diagnostic_path, payload)
        print('SWA_DIAGNOSTIC', selected['name'], scalar['objective'], scalar['ratio'], flush=True)
    ratios = {}
    for candidate in _finished_rows():
        if not b8.beats_historical(candidate['clean28']):
            continue
        science_path = OUT / f'science_{candidate["name"]}.json'
        if science_path.is_file():
            ratios[candidate['name']] = b8.read_json(science_path)['ratios']
            continue
        relchem = _chemistry_rows(source, _candidate_path(candidate), 'relchem')
        ae17 = _chemistry_rows(source, _candidate_path(candidate), 'ae17')
        mrks = _mrks_objectives(source, _candidate_path(candidate))
        science = {
            'relchem': relchem['ratio'],
            'ae17': ae17['ratio'],
            'exc': mrks['exc'] / b8.BASELINE['exc'],
            'op': mrks['op'] / b8.BASELINE['op'],
        }
        b8.write_json(science_path, {'name': candidate['name'], 'ratios': science, 'gradients': False})
        ratios[candidate['name']] = science
    return ratios


def _candidate_path(candidate):
    if candidate['name'] == 'swa_251':
        return OUT / 'swa_251.pt'
    return OUT / 'ordinary_sgd_adamw' / f'checkpoint_{candidate["local_update"]}.pt'


def _finished_rows():
    rows = []
    for name in ('raw_64', 'raw_128', 'raw_192', 'raw_251', 'swa_251'):
        path = OUT / f'validation_{name}.json'
        if not path.is_file():
            return []
        rows.append(b8.read_json(path))
    return rows


def _git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=b8.REPO, text=True, encoding='utf-8').strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def _fmt(value):
    return f'{value:.9f}'


def _recommendation(selected, raw_rows, swa_row):
    ordered = sorted(raw_rows, key=lambda row: row['order'])
    raw_best = min(raw_rows, key=lambda row: row['clean28'])
    gain = B8_T128_CLEAN28 - selected['clean28']
    gap = selected['clean28'] - b8.HISTORICAL_BEST_CLEAN28
    descending = all(ordered[index]['clean28'] > ordered[index + 1]['clean28'] for index in range(len(ordered) - 1))
    if gain > 0.05 and descending and swa_row['clean28'] < raw_best['clean28']:
        return (
            f'The raw Clean28 path is still descending and remains {gap:.3f} above the historical best. '
            'The next experiment should continue that raw AdamW state at the same fixed rate for one pass. '
            'This average should stay a readout.'
        )
    difference = swa_row['clean28'] - raw_best['clean28']
    direction = 'higher' if difference > 0 else 'lower'
    return (
        f'The best Clean28 is {gain:.3f} below B8 t128 and remains {gap:.3f} above the historical best. '
        'The raw checkpoints do not form a descending sequence, and the SWA readout is '
        f'{abs(difference):.3f} {direction} than the best raw checkpoint. '
        'Another 251 updates at 2.5e-5, or a different window of this average, is not supported by that gap. '
        'The next experiment should start again from B8 t128, keep the same loss, and change the update direction. '
        'The Full251 diagnostic moved farther than Clean28. That chemistry-objective change is not a reason to extend this arm.'
    )


def render():
    rows = _finished_rows()
    preflight_receipt = b8.read_json(OUT / 'preflight.json') if (OUT / 'preflight.json').is_file() else None
    status = b8.read_json(OUT / 'training_status.json') if (OUT / 'training_status.json').is_file() else None
    diagnostic = b8.read_json(OUT / 'diagnostic_relchem.json') if (OUT / 'diagnostic_relchem.json').is_file() else None
    parent = b8.read_json(PARENT_VALIDATION)
    complete = bool(rows) and status is not None and status.get('complete') and diagnostic is not None
    ratios = {}
    for row in rows:
        science_path = OUT / f'science_{row["name"]}.json'
        if science_path.is_file():
            ratios[row['name']] = b8.read_json(science_path)['ratios']
    classification = classify(rows, ratios) if complete else 'INCOMPLETE'
    selected = select_lowest(rows) if rows else None
    logs = status['logs'] if status else []
    seconds = sum(row['seconds'] for row in logs)
    peak_allocated = max((row['peak_allocated_bytes'] for row in logs), default=0)
    peak_reserved = max((row['peak_reserved_bytes'] for row in logs), default=0)
    if preflight_receipt:
        peak_allocated = max(peak_allocated, preflight_receipt.get('peak_allocated_bytes', 0))
        peak_reserved = max(peak_reserved, preflight_receipt.get('peak_reserved_bytes', 0))
    lines = [
        '# B8 fixed-LR continuation with weight averaging',
        '',
        f'Classification: **{classification}**.',
        '',
        (
            'One continuation from B8 t128. The learning rate was fixed at one quarter of the B8 rate. '
            'SWA was a readout and did not enter training.'
        ),
        '',
        '## Initialization',
        '',
        f'- Parent checkpoint: `{PARENT_CHECKPOINT}`',
        f'- Parent file SHA256: `{preflight_receipt["parent_checkpoint_sha256"] if preflight_receipt else "unavailable"}`',
        f'- Optimizer moment SHA256: `{preflight_receipt["optimizer_moment_sha256"] if preflight_receipt else "unavailable"}`',
        f'- Loaded model SHA256: `{preflight_receipt["loaded_model_sha256"] if preflight_receipt else "unavailable"}`',
        f'- Manifest SHA256: `{MANIFEST_SHA}`',
        f'- Parent AdamW step: {PARENT_CURSOR}',
        '- Trainable coordinates: 9446 float32',
        f'- Continuation learning rate: `{NEW_LR}` = 0.25 * `{PARENT_LR}`',
        '- Betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, foreach false, no clipping, no scheduler',
        '',
        '## SWA',
        '',
        (
            'Post-update trainable parameters at local updates '
            + ', '.join(str(point) for point in SWA_LOCAL)
            + f'. Count {len(SWA_LOCAL)}. Accumulation is float64. Evaluation restores float32. '
            + 'Buffers, frozen tensors, and optimizer moments are not averaged.'
        ),
        '',
        '## Clean28',
        '',
        '| Candidate | Kind | Local update | Clean28 | Versus B8 t128 | Versus historical best |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f'| {row["name"]} | {row["kind"]} | {row["local_update"]} | {_fmt(row["clean28"])} | '
            f'{row["clean28"] - B8_T128_CLEAN28:+.6f} | {row["clean28"] - b8.HISTORICAL_BEST_CLEAN28:+.6f} |'
        )
    lines.extend([
        '',
        (
            f'B8 t128 reference: {_fmt(B8_T128_CLEAN28)}. Historical best: {_fmt(b8.HISTORICAL_BEST_CLEAN28)}. '
            'Target remains substantially below 7. Full30 was discarded.'
        ),
        '',
    ])
    if selected is not None:
        if any(b8.beats_historical(row['clean28']) for row in rows):
            gate_line = 'Candidates below the historical best were sent to the four scientific ratios.'
        else:
            gate_line = 'No candidate is below the historical best, so the four scientific ratios were not evaluated.'
        lines.extend([
            f'Selected after all five evaluations: `{selected["name"]}` at Clean28 {_fmt(selected["clean28"])}.',
            gate_line,
            '',
        ])
    if diagnostic is not None:
        lines.extend([
            '## Full251 diagnostic',
            '',
            (
                f'`{diagnostic["name"]}` relchem objective {_fmt(diagnostic["objective"])}, '
                f'ratio {_fmt(diagnostic["ratio"])} versus the corrected P536 reference {_fmt(diagnostic["reference"])}. '
                'No gradients. This diagnostic is not scientific qualification.'
            ),
            '',
            '| Database | Count | Mean | Contribution |',
            '| --- | ---: | ---: | ---: |',
        ])
        for name in DATABASES:
            item = diagnostic['databases'].get(name)
            if item is None:
                continue
            lines.append(f'| {name} | {item["count"]} | {_fmt(item["mean"])} | {_fmt(item["contribution"])} |')
        lines.append('')
        by_id = {row['reaction_id']: row['weighted_absolute_error'] for row in parent['reaction_rows']}
        chosen = b8.read_json(OUT / f'validation_{selected["name"]}.json')
        lines.extend([
            '## Clean28 reactions versus B8 t128',
            '',
            '| Reaction | B8 t128 | Selected | Delta |',
            '| --- | ---: | ---: | ---: |',
        ])
        improved = 0
        for row in chosen['reaction_rows']:
            base = by_id[row['reaction_id']]
            delta = row['weighted_absolute_error'] - base
            if delta < 0:
                improved += 1
            lines.append(
                f'| {row["reaction_id"]} | {_fmt(base)} | {_fmt(row["weighted_absolute_error"])} | {delta:+.6f} |'
            )
        lines.extend([
            '',
            f'{improved} of 28 reactions have a lower weighted absolute error than B8 t128. '
            + 'Named comparisons include ' + ', '.join(HIGHLIGHTS) + '.',
            '',
        ])
    if rows:
        raw_rows = [row for row in rows if row['kind'] == 'adamw']
        swa_row = next(row for row in rows if row['kind'] == 'swa')
        raw_best = min(raw_rows, key=lambda row: row['clean28'])
        lines.extend([
            '## Raw trajectory and SWA',
            '',
            f'Best raw Clean28 is {_fmt(raw_best["clean28"])} at `{raw_best["name"]}`. '
            f'SWA Clean28 is {_fmt(swa_row["clean28"])}. '
            + (
                'The average is lower than every measured raw checkpoint on this seed.'
                if swa_row['clean28'] < raw_best['clean28']
                else 'The average is not lower than the best raw checkpoint on this seed.'
            ),
            'That comparison is an observed Clean28 difference. It is not evidence about loss-landscape flatness.',
            '',
        ])
        if classification in ('GO', 'NO-GO'):
            lines.extend(['## Recommendation', '', _recommendation(selected, raw_rows, swa_row), ''])
    if logs:
        ratios_norm = [row['norm_ratio'] for row in logs]
        lines.extend([
            '## Resources',
            '',
            f'- Completed local updates: {logs[-1]["local_update"]}',
            f'- Cumulative optimizer step: {logs[-1]["cumulative_optimizer_step"]}',
            f'- Sum of recorded update seconds: {seconds:.3f}',
            f'- Peak allocated bytes: {peak_allocated}',
            f'- Peak reserved bytes: {peak_reserved}',
            f'- Final displacement from B8 t128: {logs[-1]["displacement_from_t128"]:.6e}',
            (
                f'- Norm-ratio min / median / max: {min(ratios_norm):.6f} / '
                f'{sorted(ratios_norm)[len(ratios_norm) // 2]:.6f} / {max(ratios_norm):.6f}'
            ),
            (
                '- The first process stopped while recording displacement after an unsaved local update. '
                'Resume used the saved local-0 checkpoint, whose AdamW step was still 128, and then completed 251 new updates.'
            ),
            '',
        ])
    if preflight_receipt:
        lines.extend(['## Preflight', ''])
        for row in preflight_receipt['batches']:
            lines.append(
                f'- Manifest index {row["index"]}: chemistry relative L2 {row["relative_l2"]:.3e}, '
                f'scalar relative {row["scalar_relative"]:.3e}'
            )
        lines.extend([
            f'- Optimizer step during preflight: {preflight_receipt["optimizer_step"]}',
            f'- SWA probe: {preflight_receipt["swa_probe"]}',
            '',
        ])
    lines.extend([
        '## Git',
        '',
        f'Base commit before this report: `{_git_head()}`. The experiment commit is the child of that commit.',
        '',
    ])
    report = b8.REPO / 'lap_b8_swa_continuation_report.md'
    report.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    metrics = {
        'classification': classification,
        'protocol_id': PROTOCOL_ID,
        'parent_cursor': PARENT_CURSOR,
        'lr': NEW_LR,
        'manifest_sha256': MANIFEST_SHA,
        'swa_local_updates': list(SWA_LOCAL),
        'clean28': {row['name']: row['clean28'] for row in rows},
        'selected': None if selected is None else {'name': selected['name'], 'clean28': selected['clean28']},
        'diagnostic': diagnostic,
        'historical_best_clean28': b8.HISTORICAL_BEST_CLEAN28,
        'b8_t128_clean28': B8_T128_CLEAN28,
        'preflight': preflight_receipt,
        'resources': {
            'update_seconds': seconds,
            'peak_allocated_bytes': peak_allocated,
            'peak_reserved_bytes': peak_reserved,
            'local_updates': 0 if not logs else logs[-1]['local_update'],
            'cumulative_optimizer_step': None if not logs else logs[-1]['cumulative_optimizer_step'],
        },
    }
    b8.write_json(b8.REPO / 'lap_b8_swa_continuation_metrics.json', metrics)
    return classification


def execute():
    if len(SWA_LOCAL) != 29 or SWA_LOCAL[0] != 32 or SWA_LOCAL[-1] != 251:
        _stop('SWA schedule is not the preregistered set')
    source = b8.verify_sources()
    parent_sha = b8.sha256_file(PARENT_CHECKPOINT)
    saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    if saved.get('cursor') != PARENT_CURSOR or saved.get('protocol_id') != b8.PROTOCOL_ID:
        _stop('Preflight NO-GO: parent cursor or protocol is not B8 t128')
    if saved.get('scheduler') is not None or saved.get('manifest_sha256') != MANIFEST_SHA:
        _stop('Preflight NO-GO: parent scheduler or manifest hash is not the B8 arm')
    if saved.get('lambdas') != b8.LAMBDAS or tuple(saved.get('offsets')) != b8.OFFSETS:
        _stop('Preflight NO-GO: parent loss protocol differs from B8')
    if b8.sha256_file(PARENT_MANIFEST) != MANIFEST_SHA:
        _stop('Preflight NO-GO: frozen B8 manifest hash does not match the preregistered SHA256')
    try:
        inspect_optimizer_state(saved['optimizer'], PARENT_CURSOR, PARENT_LR)
    except ValueError as error:
        _stop('Preflight NO-GO: ' + str(error))
    del saved
    protocol_sha = _write_protocol(source, parent_sha)
    batches = _batches()
    if not (OUT / 'preflight.json').is_file():
        preflight(source, batches, parent_sha)
    elif not b8.read_json(OUT / 'preflight.json').get('passed'):
        _stop('Preflight did not pass')
    train(source, batches, protocol_sha, parent_sha)
    evaluate_clean28(source)
    rows = _finished_rows()
    if len(rows) != 5:
        _stop('Clean28 evaluations are incomplete')
    diagnose_and_qualify(source, select_lowest(rows))
    classification = render()
    print('SWA_CLASS', classification, flush=True)
    return classification


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        execute()
    except SystemExit as error:
        if (OUT / 'preflight.json').is_file() or (OUT / 'training_status.json').is_file():
            render()
        print('SWA_STOP', error, flush=True)
        raise
