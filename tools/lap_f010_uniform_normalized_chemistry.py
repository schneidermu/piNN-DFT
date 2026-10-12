"""One continuation from B8 t128. Chemistry is c * sqrt(r^2+1e-20) / A_d.

No legacy database or frequency factor. c matches the full-set chemistry
gradient norm to the original relchem gradient and is not retuned. AdamW
moments stay as saved under the old objective. Clean28 is read only after
the 251 updates finish.
"""
import csv
import math
import time
from pathlib import Path

import lap_b8_adamw as b8
import lap_b8_swa_continuation as continuation
import lap_f011_normalized_chemistry as arm
import lap_f011_trajectory_gradient_audit as traj
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np
import torch

PROTOCOL_ID = 'f010-uniform-normalized-chemistry-v1'
PARENT_CURSOR = continuation.PARENT_CURSOR
NEW_LR = continuation.NEW_LR
NEW_UPDATES = 251
RAW_CHECKPOINTS = (64, 128, 192, 251)
COUNT = 251
OUT = b8.REPO.parent / 'lap_f010_uniform_normalized_chemistry_20261011'
PARENT_CHECKPOINT = continuation.PARENT_CHECKPOINT
MOMENT_PREFIX = '4659e23c'
CHECKPOINT_PREFIX = '1f38f7f8'
F011_CLEAN28 = {0: 9.096700111658418, 64: 8.782020261263, 128: 8.800214350716, 192: 8.857739596910, 251: 8.938697016838}
GEOMETRY = ('t64', 't128', 't192', 't251')


def f010_raw_coefficients(residuals, scales, databases):
    """Chain-rule weights for mean(phi / A_d). Factorial F110: sqrt penalty, no legacy factor."""
    ones = np.ones(len(residuals), dtype=np.float64)
    return fact.reaction_coefficients(residuals, scales, databases, ones, 'F110')


def f010_raw_scalar(residuals, scales, databases):
    ones = np.ones(len(residuals), dtype=np.float64)
    return fact.diagnostic_scalar(residuals, scales, databases, ones, 'F110')


def f010_singleton_scale(coefficient, reference_scale, legacy_factor):
    """Remove factor_d from a production singleton, then apply c / A_d."""
    if not math.isfinite(legacy_factor) or legacy_factor <= 0.0:
        raise ValueError('Legacy factor must be finite and strictly positive')
    return arm.chemistry_scale(coefficient, reference_scale) / legacy_factor


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _confine(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(OUT.resolve()):
        raise ValueError('Refusing to write outside the F010 scratch')
    return resolved


def _csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _pairs(vectors):
    names = tuple(vectors)
    return [
        {'left': left, 'right': right, 'cosine': prior.cosine(vectors[left], vectors[right])}
        for left in names for right in names
    ]


def _geometry_row(name, local, vectors, coefficient_jw):
    weighted = {
        'f010': b8.LAMBDAS['relchem'] * vectors['f010'],
        'ae17': b8.LAMBDAS['ae17'] * vectors['ae17'],
        'exc': b8.LAMBDAS['exc'] * vectors['exc'],
        'op': b8.LAMBDAS['op'] * vectors['op'],
    }
    assembled = sum(weighted.values())
    if float(np.linalg.norm(assembled - vectors['total'])) > 1e-8:
        _stop(f'{name} total gradient is not the weighted task sum')
    directional = {task: prior.predicted_change(coefficient_jw, piece) for task, piece in weighted.items()}
    return {
        'checkpoint': name, 'local_update': local, 'parameter_vector': 'f010',
        'cos_jw_f010': prior.cosine(vectors['jw'], vectors['f010']),
        'cos_jw_op': prior.cosine(vectors['jw'], vectors['op']),
        'cos_jw_ae17': prior.cosine(vectors['jw'], vectors['ae17']),
        'cos_jw_exc': prior.cosine(vectors['jw'], vectors['exc']),
        'cos_jw_total': prior.cosine(vectors['jw'], vectors['total']),
        'cos_f010_f011': prior.cosine(vectors['f010'], vectors['f011']),
        'cos_f010_operator': prior.cosine(vectors['f010'], vectors['op']),
        'cancellation_ratio': prior.cancellation_ratio(tuple(weighted.values())),
        'directional_f010': directional['f010'], 'directional_ae17': directional['ae17'],
        'directional_exc': directional['exc'], 'directional_op': directional['op'],
        'directional_total': sum(directional.values()),
        'weighted_f010': float(np.linalg.norm(weighted['f010'])),
        'weighted_ae17': float(np.linalg.norm(weighted['ae17'])),
        'weighted_exc': float(np.linalg.norm(weighted['exc'])),
        'weighted_op': float(np.linalg.norm(weighted['op'])),
        'raw_f010': float(np.linalg.norm(vectors['f010'])),
        'raw_jw': float(np.linalg.norm(vectors['jw'])),
    }


def cpu_gates():
    rows, jacobian, residuals, databases, scales, counts, raw, saved = arm._load_geometry()
    if len(rows) != COUNT or len(set(databases)) < 1:
        _stop('J251 geometry is not the frozen 251-reaction set')
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        _stop('A_d is not finite and positive')
    g_old = fact.production_coefficients(residuals, raw) @ jacobian
    g_raw = f010_raw_coefficients(residuals, scales, databases) @ jacobian
    g_jw = arm.jw_coefficients(residuals, scales, databases) @ jacobian
    g_f011_raw = arm.f011_coefficients(residuals, raw, scales, databases) @ jacobian
    if float(np.linalg.norm(g_raw - g_jw) / np.linalg.norm(g_jw)) > 1e-8:
        _stop('Unscaled F010 gradient does not match the JW gradient')
    if float(np.linalg.norm(g_old - saved) / np.linalg.norm(saved)) > arm.COSINE_TOLERANCE:
        _stop('Chain-rule relchem gradient does not match the saved production gradient')
    if abs(fact.production_scalar(residuals, raw) - arm.PRIOR_RELCHEM['initial']) > arm.SCALAR_TOLERANCE:
        _stop('Cached Full251 relchem scalar does not match the receipt')
    jw_scalar = j251.j251_wtmad_style(residuals, databases, scales)
    raw_scalar = f010_raw_scalar(residuals, scales, databases)
    if abs(jw_scalar - arm.PRIOR_JW['initial']) > arm.SCALAR_TOLERANCE or abs(raw_scalar - jw_scalar) > 1e-8:
        _stop('Unscaled F010 scalar does not match JW')
    legacy_scalar = fact.diagnostic_scalar(residuals, scales, databases, raw, 'F111')
    if abs(raw_scalar - legacy_scalar) < 1e-4:
        _stop('Removing the legacy factors did not change the chemistry scalar')
    uniform = f010_raw_coefficients(residuals, scales, databases)
    legacy = arm.f011_coefficients(residuals, raw, scales, databases)
    if np.allclose(uniform, legacy):
        _stop('F010 coefficients still contain the legacy factors')
    calibration = arm.calibrate_norm(g_old, g_raw)
    f011_receipt = b8.read_json(b8.REPO / 'lap_f011_normalization_receipt.json')
    if abs(calibration['old_norm'] - f011_receipt['old_norm']) > 1e-6:
        _stop('Original chemistry gradient norm does not match the F011 calibration receipt')
    if abs(calibration['c'] - f011_receipt['c']) < 1e-6 or calibration['relative_norm_error'] > 1e-12:
        _stop('c010 reused c011 or failed to match the gradient norms')
    physical = {task: prior._load_vector(prior.OUT / 'b8_t128' / f'{task}.npy.npz') for task in ('ae17', 'exc', 'op')}
    if any(vector is None for vector in physical.values()):
        _stop('Saved physical-task gradients are missing')
    g_f011 = f011_receipt['c'] * g_f011_raw
    g_f010 = calibration['scaled_gradient']
    vectors = {'jw': g_jw, 'f010': g_f010, 'f011': g_f011, 'relchem': saved, **physical}
    vectors['total'] = (
        b8.LAMBDAS['relchem'] * g_f010 + b8.LAMBDAS['ae17'] * physical['ae17']
        + b8.LAMBDAS['exc'] * physical['exc'] + b8.LAMBDAS['op'] * physical['op']
    )
    row = _geometry_row('t0', 0, vectors, g_jw)
    if abs(row['cos_jw_f010'] - 1.0) > 1e-8:
        _stop('Scaled F010 is not the JW direction')
    receipt = {
        'protocol_id': PROTOCOL_ID,
        'c': calibration['c'],
        'c011': f011_receipt['c'],
        'old_norm': calibration['old_norm'],
        'f010_raw_norm': calibration['f011_norm'],
        'scaled_norm': calibration['scaled_norm'],
        'relative_norm_error': calibration['relative_norm_error'],
        'cosine_old_f010': calibration['cosine_old'],
        'cosine_jw_f010': row['cos_jw_f010'],
        'cosine_f010_f011': row['cos_f010_f011'],
        'cosine_f010_operator': row['cos_f010_operator'],
        'cosine_jw_operator': row['cos_jw_op'],
        'cosine_jw_total': row['cos_jw_total'],
        'jw_predicted_change': row['directional_total'],
        'jw_contributions': {
            'f010': row['directional_f010'], 'ae17': row['directional_ae17'],
            'exc': row['directional_exc'], 'op': row['directional_op'],
        },
        'cancellation_ratio': row['cancellation_ratio'],
        'geometry': row,
        'pairs': _pairs(vectors),
        'scales': scales,
        'counts': counts,
        'scales_sha256': arm.scales_sha256(scales),
        'reference_scales_sha256': b8.sha256_file(j251.OUT / 'reference_scales.json'),
        'jw_scalar': jw_scalar,
        'f010_raw_scalar': raw_scalar,
        'optimizer_updates': 0,
        'adamw_moments': 'retained from the original chemistry objective; shared with the F011 continuation',
        'databases_t0': _database_pieces(databases, g_jw, arm.jw_coefficients(residuals, scales, databases), jacobian),
    }
    return receipt


def _database_pieces(databases, g_jw, coefficients, jacobian):
    rows = []
    for database in sorted(set(databases)):
        mask = np.array([item == database for item in databases], dtype=np.float64)
        piece = (coefficients * mask) @ jacobian
        rows.append({
            'database': database, 'cosine_jw': prior.cosine(piece, g_jw),
            'jw_gradient_norm': float(np.linalg.norm(piece)),
        })
    return rows


def _batches():
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    stored = b8.read_json(continuation.PARENT_MANIFEST)
    if stored != batches or b8.sha256_file(continuation.PARENT_MANIFEST) != continuation.MANIFEST_SHA:
        _stop('B8 manifest does not match the F011 schedule')
    b8.assert_batch_properties(batches, rows)
    if b8.presentation_counts(batches) != {row['relchem']['identity']: 8 for row in rows}:
        _stop('The schedule does not present every reaction eight times')
    return batches


def _check_model(model):
    parameters = b8._import_run().existing.named_trainable_parameters(model)
    coordinates = sum(parameter.numel() for parameter in parameters.values())
    if (type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or coordinates != 9446
            or int(model.lap_architecture_version) != 1):
        _stop('Model architecture does not match B8 t128')
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        _stop('Trainable parameters are not float32')
    return parameters


def f010_mean(model, shadow, bundle, specs, dispersion, scales, coefficient):
    torch.cuda.synchronize()
    started = time.perf_counter()
    pairs = []
    infos = []
    for spec in specs:
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        value, grad = b8._singleton(model, shadow, reaction, dispersion)
        factor = j251._factor(spec['database'])
        scale = f010_singleton_scale(coefficient, scales[spec['database']], factor)
        scaled_grad = {name: piece.detach() * scale for name, piece in grad.items()}
        infos.append({
            'identity': spec['identity'], 'variant': spec['variant'], 'database': spec['database'],
            'loss': float(value) * scale, 'unscaled_loss': float(value), 'legacy_factor': factor,
            'grad_norm': float(torch.linalg.vector_norm(torch.cat([piece.reshape(-1) for piece in scaled_grad.values()]))),
        })
        pairs.append((float(value) * scale, scaled_grad))
        del reaction, grad, scaled_grad
        torch.cuda.empty_cache()
    loss, grad = b8.combine_gradients(pairs)
    torch.cuda.synchronize()
    return loss, grad, infos, time.perf_counter() - started


def live_full_gradient(model, shadow, bundle, dispersion, scales, coefficient, names):
    specs = arm._selected(bundle)
    old_loss = 0.0
    new_loss = 0.0
    old = None
    new = None
    for index, spec in enumerate(specs):
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        value, grad = b8._singleton(model, shadow, reaction, dispersion)
        scale = f010_singleton_scale(coefficient, scales[spec['database']], j251._factor(spec['database']))
        if old is None:
            old = {name: torch.zeros_like(piece) for name, piece in grad.items()}
            new = {name: torch.zeros_like(piece) for name, piece in grad.items()}
        if tuple(grad) != names:
            _stop('Live chemistry parameter order changed')
        for name in names:
            old[name].add_(grad[name])
            new[name].add_(grad[name], alpha=scale)
        old_loss += float(value)
        new_loss += float(value) * scale
        del reaction, grad
        torch.cuda.empty_cache()
        if (index + 1) % 25 == 0:
            print('F010_PREFLIGHT_REACTION', index + 1, flush=True)
    for name in names:
        old[name].div_(COUNT)
        new[name].div_(COUNT)
    return old_loss / COUNT, new_loss / COUNT, old, new


def live_gates(source, receipt):
    parent_sha = b8.sha256_file(PARENT_CHECKPOINT)
    if not parent_sha.startswith(CHECKPOINT_PREFIX):
        _stop('Parent checkpoint SHA256 prefix is not the B8 t128 receipt')
    run = b8._import_run()
    saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    model, shadow = run.model_at(source['initial_state'])
    parameters = _check_model(model)
    model.load_state_dict(saved['model'])
    parameters = _check_model(model)
    optimizer = arm._new_optimizer(parameters)
    moment = continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], PARENT_CURSOR, continuation.PARENT_LR)
    if not moment.startswith(MOMENT_PREFIX):
        _stop('AdamW moment SHA256 prefix is not the B8 t128 receipt')
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    if tuple(parameters) != names:
        _stop('Parameter order differs from the gradient audit')
    digest = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        batches = _batches()
        _loss, grad, infos, _elapsed = f010_mean(
            model, shadow, bundle, batches[0]['relchem'], dispersion, receipt['scales'], receipt['c'])
        for info in infos:
            removed = info['unscaled_loss'] * arm.chemistry_scale(receipt['c'], receipt['scales'][info['database']])
            if abs(info['loss'] * info['legacy_factor'] - removed) > 1e-10:
                _stop('Minibatch F010 still multiplies by the legacy factor')
        other_losses, other_raw, _system, _seconds = b8.other_tasks(
            model, shadow, bundle, batches[0], dispersion, mrks_dispersion)
        _measured, measured_raw = run.measure(
            model, shadow, bundle, b8.load_j251_rows()[0], dispersion, mrks_dispersion, exc_chunk_size=4096)
        other_l2 = {task: b8.relative_l2(other_raw[task], measured_raw[task]) for task in ('ae17', 'exc', 'op')}
        if any(value > b8.OTHER_TASK_L2_LIMIT for value in other_l2.values()):
            _stop('AE17, Exc, or operator gradient differs from the original measurement path')
        old_scalar, new_scalar, old_grad, new_grad = live_full_gradient(
            model, shadow, bundle, dispersion, receipt['scales'], receipt['c'], names)
        if abs(old_scalar - arm.PRIOR_RELCHEM['initial']) > arm.SCALAR_TOLERANCE:
            _stop(f'Live Full251 relchem scalar {old_scalar} missed the receipt')
        _rows, jacobian, residuals, databases, scales, _counts, raw, _saved = arm._load_geometry()
        chain_old = fact.production_coefficients(residuals, raw) @ jacobian
        chain_new = receipt['c'] * f010_raw_coefficients(residuals, scales, databases) @ jacobian
        old_gap = float(np.linalg.norm(arm._flat(names, old_grad) - chain_old) / np.linalg.norm(chain_old))
        new_gap = float(np.linalg.norm(arm._flat(names, new_grad) - chain_new) / np.linalg.norm(chain_new))
        if old_gap > arm.LIVE_RELATIVE_L2 or new_gap > arm.LIVE_RELATIVE_L2:
            _stop(f'Live full gradients disagree with the chain rule: {old_gap}, {new_gap}')
        if abs(new_scalar - receipt['c'] * f010_raw_scalar(residuals, scales, databases)) > arm.SCALAR_TOLERANCE:
            _stop('Live F010 scalar disagrees with the independent formula')
        legacy_live = receipt['c'] * fact.diagnostic_scalar(residuals, scales, databases, raw, 'F111')
        if abs(new_scalar - legacy_live) < 1e-4:
            _stop('Live F010 scalar still matches the legacy-weighted formula')
        if abs(float(np.linalg.norm(chain_new)) - receipt['old_norm']) / receipt['old_norm'] > 1e-12:
            _stop('c010 does not match the full-set gradient norms')
        if {int(state['step']) for state in optimizer.state.values()} != {PARENT_CURSOR}:
            _stop('Preflight took an optimizer step')
        if continuation.moment_sha(optimizer.state_dict()) != moment or run.existing.digest(model) != digest:
            _stop('Preflight changed the model or the AdamW moments')
        if b8.sha256_file(PARENT_CHECKPOINT) != parent_sha:
            _stop('Preflight changed the parent checkpoint')
        del grad, old_grad, new_grad, measured_raw
    finally:
        bundle.close()
        del model, shadow, optimizer
        torch.cuda.empty_cache()
    receipt = dict(receipt)
    receipt['live'] = {
        'passed': True, 'old_scalar': old_scalar, 'new_scalar': new_scalar,
        'old_relative_l2': old_gap, 'new_relative_l2': new_gap,
        'other_task_relative_l2': other_l2, 'other_losses': other_losses,
        'parent_checkpoint_sha256': parent_sha, 'optimizer_moment_sha256': moment,
        'loaded_model_sha256': digest, 'optimizer_step': False,
    }
    return receipt


def _save_checkpoint(path, model, optimizer, cursor, logs, coefficient, parent_sha):
    path = _confine(path)
    payload = {
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': {key: value for key, value in optimizer.state_dict().items()},
        'rng': b8._capture_rng(), 'cursor': cursor, 'logs': list(logs),
        'protocol_id': PROTOCOL_ID, 'manifest_sha256': continuation.MANIFEST_SHA,
        'c': coefficient, 'lambdas': dict(b8.LAMBDAS), 'lr': NEW_LR, 'scheduler': None,
        'parent_checkpoint_sha256': parent_sha,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path, model, optimizer, coefficient):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('c') != coefficient or saved.get('scheduler') is not None:
        _stop('Resume checkpoint is not this F010 arm')
    if saved.get('lr') != NEW_LR or saved.get('lambdas') != b8.LAMBDAS:
        _stop('Resume checkpoint changed the learning rate or task coefficients')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    b8._restore_rng(saved['rng'])
    return int(saved['cursor']), list(saved['logs'])


def train(source, coefficient, scales):
    run = b8._import_run()
    parent_sha = b8.sha256_file(PARENT_CHECKPOINT)
    target = _confine(OUT / 'ordinary_sgd_adamw')
    target.mkdir(parents=True, exist_ok=True)
    latest = target / 'latest.pt'
    model, shadow = run.model_at(source['initial_state'])
    batches = _batches()
    if latest.is_file():
        parameters = _check_model(model)
        optimizer = arm._new_optimizer(parameters)
        local_update, logs = _load_checkpoint(latest, model, optimizer, coefficient)
        parameters = _check_model(model)
    else:
        saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
        parameters = _check_model(model)
        model.load_state_dict(saved['model'])
        parameters = _check_model(model)
        optimizer = arm._new_optimizer(parameters)
        continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], PARENT_CURSOR, continuation.PARENT_LR)
        logs = []
        local_update = 0
        _save_checkpoint(latest, model, optimizer, 0, logs, coefficient, parent_sha)
    parent_weights = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)['model']
    origin = {name: parent_weights[name].detach().clone() for name in parameters}
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
            relchem_loss, relchem_grad, infos, _elapsed = f010_mean(
                model, shadow, bundle, batch['relchem'], dispersion, scales, coefficient)
            other_losses, other_raw, _system, _other = b8.other_tasks(
                model, shadow, bundle, batch, dispersion, mrks_dispersion)
            raw = {'relchem': relchem_grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op']}
            if any(not math.isfinite(value) for value in (relchem_loss, *other_losses.values())):
                _stop('Nonfinite loss')
            joint = run.adamw_step(model, optimizer, raw, b8.LAMBDAS)
            done = index + 1
            steps = {int(state['step']) for state in optimizer.state.values()}
            if steps != {PARENT_CURSOR + done} or any(group['lr'] != NEW_LR for group in optimizer.param_groups):
                _stop('AdamW did not take exactly one fixed-rate step')
            row = {
                'local_update': done, 'cumulative_optimizer_step': PARENT_CURSOR + done,
                'manifest_index': batch['cursor'], 'chemistry_loss': relchem_loss,
                'ae17_loss': other_losses['ae17'], 'exc_loss': other_losses['exc'], 'op_loss': other_losses['op'],
                'composite_loss': sum(b8.LAMBDAS[task] * value for task, value in (
                    ('relchem', relchem_loss), ('ae17', other_losses['ae17']),
                    ('exc', other_losses['exc']), ('op', other_losses['op']),
                )),
                'lr': NEW_LR, 'gradient_norm': b8._grad_norm(joint),
                'update_norm': arm._vector_delta(parameters, before_weights),
                'displacement_from_t128': arm._vector_delta(parameters, origin),
                'nonfinite': False, 'seconds': time.perf_counter() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'before_sha256': before, 'after_sha256': run.existing.digest(model), 'reactions': infos,
            }
            logs.append(row)
            del raw, joint, relchem_grad, other_raw, before_weights
            torch.cuda.empty_cache()
            _save_checkpoint(latest, model, optimizer, done, logs, coefficient, parent_sha)
            if done in RAW_CHECKPOINTS:
                _save_checkpoint(target / f'checkpoint_{done}.pt', model, optimizer, done, logs, coefficient, parent_sha)
            b8.write_json(OUT / 'training_status.json', {
                'local_update': done, 'complete': done == NEW_UPDATES, 'logs': logs,
                'cumulative_optimizer_step': PARENT_CURSOR + done,
            })
            print('F010_UPDATE', done, f'{row["seconds"]:.3f}', f'{row["chemistry_loss"]:.6f}', flush=True)
    finally:
        bundle.close()
    if b8.sha256_file(PARENT_CHECKPOINT) != parent_sha:
        _stop('Training changed the B8 checkpoint')
    return logs[-1]['local_update']


def training_metrics(source, scales, coefficient):
    run = b8._import_run()
    targets = [('initial', 0, PARENT_CHECKPOINT)] + [
        (f't{cursor}', cursor, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt') for cursor in RAW_CHECKPOINTS
    ]
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        specs = arm._selected(bundle)
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        rows_out = []
        for name, cursor, path in targets:
            done = OUT / f'metrics_{name}.json'
            if done.is_file():
                rows_out.append(b8.read_json(done))
                continue
            if not path.is_file():
                _stop('Metric checkpoint is missing: ' + name)
            model, shadow, _run = arm._load_eval_model(source, path)
            shadow.load_state_dict(model.state_dict(), strict=True)
            before = run.existing.digest(model)
            reactions = []
            for spec in specs:
                reaction = b8._load_reaction(bundle, 'relchem', spec)
                with torch.no_grad():
                    prediction, reference = j251._prediction(shadow, reaction, dispersion)
                residual = float((prediction - reference).detach().cpu())
                factor = j251._factor(spec['database'])
                penalty = factor * math.sqrt(1e-20 + residual * residual)
                bare = math.sqrt(1e-20 + residual * residual)
                reactions.append({
                    'identity': spec['identity'], 'reaction_id': spec['reaction_id'], 'database': spec['database'],
                    'residual': residual, 'absolute_residual': abs(residual),
                    'reference': float(reference.detach().cpu()), 'A_d': scales[spec['database']],
                    'legacy_factor': factor, 'old_penalty': penalty,
                    'f010_penalty': coefficient * bare / scales[spec['database']],
                    'f011_penalty': penalty * arm.chemistry_scale(arm_c011(), scales[spec['database']]),
                })
                del reaction, prediction, reference
                torch.cuda.empty_cache()
            ae17 = arm.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
            mrks = arm.mrks_scalars(model, bundle, mrks_dispersion)
            if run.existing.digest(model) != before:
                _stop('Training-set evaluation changed the checkpoint')
            databases = [row['database'] for row in reactions]
            residuals = [row['residual'] for row in reactions]
            payload = {
                'name': name, 'local_update': cursor, 'checkpoint_sha256': b8.sha256_file(path),
                'jw': j251.j251_wtmad_style(residuals, databases, scales),
                'mae': j251.j251_mae(residuals),
                'relchem': sum(row['old_penalty'] for row in reactions) / COUNT,
                'f010_raw': f010_raw_scalar(residuals, scales, databases),
                'f010': sum(row['f010_penalty'] for row in reactions) / COUNT,
                'f011': sum(row['f011_penalty'] for row in reactions) / COUNT,
                'ae17': ae17, 'exc': mrks['exc'], 'op': mrks['op'], 'reactions': reactions,
            }
            payload['composite_f010'] = (
                b8.LAMBDAS['relchem'] * payload['f010'] + b8.LAMBDAS['ae17'] * ae17
                + b8.LAMBDAS['exc'] * mrks['exc'] + b8.LAMBDAS['op'] * mrks['op']
            )
            payload['composite_relchem'] = (
                b8.LAMBDAS['relchem'] * payload['relchem'] + b8.LAMBDAS['ae17'] * ae17
                + b8.LAMBDAS['exc'] * mrks['exc'] + b8.LAMBDAS['op'] * mrks['op']
            )
            if abs(payload['f010'] - coefficient * payload['f010_raw']) > 1e-8:
                _stop('Scaled F010 is not c010 times the raw objective')
            if name == 'initial' and (
                    abs(payload['relchem'] - arm.PRIOR_RELCHEM['initial']) > arm.SCALAR_TOLERANCE
                    or abs(payload['jw'] - arm.PRIOR_JW['initial']) > arm.SCALAR_TOLERANCE):
                _stop('Initial J251 metrics do not match the B8 t128 receipts')
            b8.write_json(done, payload)
            rows_out.append(payload)
            print('F010_METRICS', name, payload['jw'], payload['f010'], payload['relchem'], flush=True)
            del model, shadow
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    return rows_out


def arm_c011():
    return b8.read_json(b8.REPO / 'lap_f011_normalization_receipt.json')['c']


def evaluate_clean28(source):
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    try:
        from tools.evaluate_microbatch_endpoint import validation
        for cursor in RAW_CHECKPOINTS:
            path = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
            receipt = OUT / f'validation_t{cursor}.json'
            if receipt.is_file():
                continue
            model, _shadow, _run = arm._load_eval_model(source, path)
            before = run.existing.digest(model)
            result = validation(model, bundle)
            if run.existing.digest(model) != before or len([row for row in result['reaction_rows'] if row['clean']]) != 28:
                _stop('Clean28 evaluation changed the model or the subset')
            b8.write_json(receipt, {
                'name': f't{cursor}', 'local_update': cursor, 'clean28': result['clean28'],
                'checkpoint_sha256': b8.sha256_file(path), 'optimizer_step': False,
            })
            print('F010_CLEAN28', cursor, result['clean28'], flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        bundle.close()


def qualify(source, clean_rows):
    ratios = {}
    needed = [row for row in clean_rows if b8.beats_historical(row['clean28'])]
    if not needed:
        return ratios
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for row in needed:
            path = OUT / f'science_{row["name"]}.json'
            if path.is_file():
                ratios[row['name']] = b8.read_json(path)['ratios']
                continue
            model, shadow, _run = arm._load_eval_model(source, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{row["local_update"]}.pt')
            science = {
                'relchem': arm.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'relchem') / b8.BASELINE['relchem'],
                'ae17': arm.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17') / b8.BASELINE['ae17'],
                'exc': arm.mrks_scalars(model, bundle, mrks_dispersion)['exc'] / b8.BASELINE['exc'],
                'op': arm.mrks_scalars(model, bundle, mrks_dispersion)['op'] / b8.BASELINE['op'],
            }
            b8.write_json(path, {'name': row['name'], 'ratios': science, 'used_f010_in_ratio': False})
            ratios[row['name']] = science
            del model, shadow
    finally:
        bundle.close()
    return ratios


def gradient_audit(source, coefficient, scales):
    run = b8._import_run()
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    bundle = run.PublicationDataset(run.DATA)
    rows_out = []
    try:
        dispersion = bundle.chemistry_dispersions()
        specs = arm._selected(bundle)
        for name in GEOMETRY:
            folder = OUT / 'geometry' / name
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / 'geometry.json'
            if done.is_file():
                rows_out.append(b8.read_json(done))
                continue
            path = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{name[1:]}.pt'
            file_hash = b8.sha256_file(path)
            model, shadow, _run = arm._load_eval_model(source, path)
            digest = run.existing.digest(model)
            rows, jacobian = traj._reaction_jacobian(model, shadow, bundle, dispersion, specs, names, folder)
            if [row['identity'] for row in rows] != [spec['identity'] for spec in specs]:
                _stop('F010 Jacobian order is not the J251 evaluation order')
            residuals = [row['residual'] for row in rows]
            databases = [row['database'] for row in rows]
            raw = np.array([j251._factor(database) for database in databases], dtype=np.float64)
            physical = traj._physical(model, shadow, bundle, folder, names)
            f011_c = arm_c011()
            vectors = {
                'jw': arm.jw_coefficients(residuals, scales, databases) @ jacobian,
                'f010': coefficient * (f010_raw_coefficients(residuals, scales, databases) @ jacobian),
                'f011': f011_c * (arm.f011_coefficients(residuals, raw, scales, databases) @ jacobian),
                'ae17': physical['ae17'], 'exc': physical['exc'], 'op': physical['op'],
            }
            vectors['total'] = (
                b8.LAMBDAS['relchem'] * vectors['f010'] + b8.LAMBDAS['ae17'] * vectors['ae17']
                + b8.LAMBDAS['exc'] * vectors['exc'] + b8.LAMBDAS['op'] * vectors['op']
            )
            payload = _geometry_row(name, int(name[1:]), vectors, vectors['jw'])
            payload['databases'] = _database_pieces(databases, vectors['jw'], arm.jw_coefficients(residuals, scales, databases), jacobian)
            payload['checkpoint_sha256'] = file_hash
            if run.existing.digest(model) != digest or b8.sha256_file(path) != file_hash:
                _stop('Gradient measurement changed an F010 checkpoint')
            b8.write_json(done, payload)
            rows_out.append(payload)
            print('F010_GEOMETRY', name, payload['cos_jw_total'], payload['cos_f010_f011'], flush=True)
            del model, shadow, jacobian, vectors
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    return rows_out


def _f011_metrics():
    path = b8.REPO / 'lap_f011_checkpoint_metrics.csv'
    with path.open(encoding='utf-8', newline='\n') as handle:
        return list(csv.DictReader(handle))


def _f011_databases():
    path = b8.REPO / 'lap_f011_database_errors.csv'
    with path.open(encoding='utf-8', newline='\n') as handle:
        return list(csv.DictReader(handle))


def render(receipt, status, metrics, clean_rows, ratios, geometry, label):
    f011_metrics = {row['name']: row for row in _f011_metrics()}
    f011_db = _f011_databases()
    logs = [] if status is None else status['logs']
    preflight = [{'quantity': key, 'value': receipt[key]} for key in (
        'c', 'old_norm', 'f010_raw_norm', 'scaled_norm', 'relative_norm_error',
        'cosine_old_f010', 'cosine_jw_f010', 'cosine_f010_f011', 'cosine_f010_operator',
        'cosine_jw_operator', 'cosine_jw_total', 'jw_predicted_change', 'cancellation_ratio',
    )]
    for pair in receipt['pairs']:
        preflight.append({'quantity': f"cosine_{pair['left']}_{pair['right']}", 'value': pair['cosine']})
    _csv(b8.REPO / 'lap_f010_preflight_gradient_geometry.csv', preflight, ('quantity', 'value'))
    _csv(b8.REPO / 'lap_f010_training_metrics.csv', logs, (
        'local_update', 'cumulative_optimizer_step', 'chemistry_loss', 'ae17_loss', 'exc_loss', 'op_loss',
        'composite_loss', 'lr', 'gradient_norm', 'update_norm', 'nonfinite', 'seconds', 'peak_allocated_bytes',
    ))
    checkpoint_rows = []
    database_rows = []
    reaction_rows = []
    comparison = []
    for item in metrics:
        checkpoint_rows.append({key: item[key] for key in (
            'name', 'local_update', 'jw', 'mae', 'relchem', 'f010_raw', 'f010', 'f011', 'ae17', 'exc', 'op',
            'composite_f010', 'composite_relchem', 'checkpoint_sha256',
        )})
        other_name = 'initial' if item['local_update'] == 0 else item['name']
        other = f011_metrics[other_name]
        comparison.append({
            'local_update': item['local_update'],
            'f010_jw': item['jw'], 'f011_jw': float(other['jw']),
            'jw_difference': item['jw'] - float(other['jw']),
            'jw_relative_difference': (item['jw'] - float(other['jw'])) / float(other['jw']),
            'f010_clean28': '' if item['local_update'] == 0 else next(row['clean28'] for row in clean_rows if row['local_update'] == item['local_update']),
            'f011_clean28': F011_CLEAN28[item['local_update']],
            'parameter_note': 'same B8 t128 weights' if item['local_update'] == 0 else 'different parameter vectors',
        })
        grouped = {}
        for row in item['reactions']:
            grouped.setdefault(row['database'], []).append(row)
            reaction_rows.append({'checkpoint': item['name'], **row})
        for database, group in sorted(grouped.items()):
            count = len(group)
            f011_match = next((
                row for row in f011_db
                if row['checkpoint'] == other_name and row['database'] == database
            ), None)
            jw_mean = sum(row['absolute_residual'] / row['A_d'] for row in group) / count
            database_rows.append({
                'checkpoint': item['name'], 'database': database, 'count': count, 'A_d': group[0]['A_d'],
                'mae': sum(row['absolute_residual'] for row in group) / count,
                'jw_mean': jw_mean,
                'jw_contribution': sum(row['absolute_residual'] / row['A_d'] for row in group) / COUNT,
                'mae_contribution': sum(row['absolute_residual'] for row in group) / COUNT,
                'relchem_contribution': sum(row['old_penalty'] for row in group) / COUNT,
                'f010_contribution': sum(row['f010_penalty'] for row in group) / COUNT,
                'f011_jw_mean': '' if f011_match is None else float(f011_match['jw']),
                'jw_mean_minus_f011': '' if f011_match is None else jw_mean - float(f011_match['jw']),
            })
    geometry_rows = [receipt['geometry'], *geometry]
    for row in geometry_rows:
        match = next((item for item in comparison if item['local_update'] == row['local_update']), None)
        if match is not None:
            match['cos_f010_f011_at_f010_weights'] = row['cos_f010_f011']
            match['cos_jw_operator_at_f010_weights'] = row['cos_jw_op']
            match['cos_jw_total_at_f010_weights'] = row['cos_jw_total']
    for row in geometry:
        for piece in row['databases']:
            target = next((
                item for item in database_rows
                if item['checkpoint'] == row['checkpoint'] and item['database'] == piece['database']
            ), None)
            if target is not None:
                target['cosine_jw'] = piece['cosine_jw']
    for piece in receipt['databases_t0']:
        target = next((item for item in database_rows if item['checkpoint'] == 'initial' and item['database'] == piece['database']), None)
        if target is not None:
            target['cosine_jw'] = piece['cosine_jw']
    _csv(b8.REPO / 'lap_f010_checkpoint_metrics.csv', checkpoint_rows, list(checkpoint_rows[0]))
    _csv(b8.REPO / 'lap_f010_database_errors.csv', database_rows, list(database_rows[0]))
    _csv(b8.REPO / 'lap_f010_reaction_errors.csv', reaction_rows, list(reaction_rows[0]))
    _csv(b8.REPO / 'lap_f010_gradient_geometry.csv', geometry_rows, list(geometry_rows[0]))
    _csv(b8.REPO / 'lap_f010_comparison_f011.csv', comparison, list(comparison[0]))
    b8.write_json(b8.REPO / 'lap_f010_normalization_receipt.json', {
        key: value for key, value in receipt.items() if key != 'pairs'
    })
    b8.write_json(b8.REPO / 'lap_f010_evaluation_receipts.json', {
        'metrics': checkpoint_rows, 'clean28': clean_rows, 'geometry': geometry_rows,
        'optimizer_updates': 0 if status is None else status['local_update'],
    })
    _plots(metrics, clean_rows, geometry_rows)
    _report(receipt, logs, metrics, clean_rows, ratios, geometry_rows, comparison, database_rows, label)


def _plots(metrics, clean_rows, geometry_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f011 = {int(row['local_update']): float(row['jw']) for row in _f011_metrics()}
    local = [item['local_update'] for item in metrics]
    figure, axis = plt.subplots()
    axis.plot(local, [item['jw'] for item in metrics], marker='o', label='F010 JW')
    axis.plot(local, [f011[item['local_update']] for item in metrics], marker='o', label='F011 JW')
    axis.set_xlabel('Local update')
    axis.set_ylabel('JW')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f010_jw_comparison.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.plot([row['local_update'] for row in clean_rows], [row['clean28'] for row in clean_rows], marker='o', label='F010 Clean28')
    published = [key for key in F011_CLEAN28 if key != 0]
    axis.plot(published, [F011_CLEAN28[key] for key in published], marker='o', label='F011 Clean28')
    axis.axhline(b8.HISTORICAL_BEST_CLEAN28, color='black', linewidth=0.8, label='Historical best')
    axis.set_xlabel('Local update')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f010_clean28_comparison.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.plot([row['local_update'] for row in geometry_rows], [row['cos_jw_total'] for row in geometry_rows], marker='o', label='JW/total')
    axis.plot([row['local_update'] for row in geometry_rows], [row['cos_jw_op'] for row in geometry_rows], marker='o', label='JW/operator')
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.set_xlabel('Local update')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f010_gradient_alignment.png', dpi=120)
    plt.close(figure)


def _report(receipt, logs, metrics, clean_rows, ratios, geometry_rows, comparison, database_rows, label):
    by_name = {item['name']: item for item in metrics}
    initial = by_name['initial']
    final = by_name['t251']
    seconds = sum(row['seconds'] for row in logs)
    peak = max((row['peak_allocated_bytes'] for row in logs), default=0)
    lines = [
        '# F010 uniform normalized chemistry',
        '',
        f'Classification: **{label}**.',
        '',
        'One arm from B8 t128. The chemistry term is `c010 * sqrt(r^2 + 1e-20) / A_d`, with no legacy database or frequency factor, averaged as in the B8 minibatch. Operator, Exc, AE17, lambdas, and the learning rate are the F011 values.',
        '',
        'The restored AdamW moments were accumulated under the original chemistry objective. That limitation is shared with the F011 arm. This is one matched-sequence, single-seed comparison.',
        '',
        '## Calibration',
        '',
        f'- `c010` = `{receipt["c"]:.12f}`',
        f'- `c011` was `{receipt["c011"]:.12f}` and was not reused',
        f'- `||g_old||` = `{receipt["old_norm"]:.6f}`',
        f'- `||g_F010_raw||` = `{receipt["f010_raw_norm"]:.6f}`',
        f'- `||c010 g_F010_raw||` = `{receipt["scaled_norm"]:.6f}`',
        f'- Relative norm error `{receipt["relative_norm_error"]:.3e}`',
        f'- cos(F010, JW) = `{receipt["cosine_jw_f010"]:.6f}`',
        f'- cos(F010, F011 formula) = `{receipt["cosine_f010_f011"]:.6f}`',
        f'- cos(F010, operator) = `{receipt["cosine_f010_operator"]:.6f}`',
        f'- cos(JW, total) = `{receipt["cosine_jw_total"]:.6f}`',
        f'- A_d SHA256 `{receipt["scales_sha256"]}`',
        f'- Parent checkpoint `{receipt["live"]["parent_checkpoint_sha256"]}`',
        f'- Parent moments `{receipt["live"]["optimizer_moment_sha256"]}`',
        '',
        '## Training-set metrics',
        '',
        '| Checkpoint | JW | MAE | Relchem | F010 raw | F010 | F011 diagnostic | AE17 | Exc | Operator | F010 composite |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for item in metrics:
        lines.append(
            f'| {item["name"]} | {item["jw"]:.6f} | {item["mae"]:.6f} | {item["relchem"]:.6f} | {item["f010_raw"]:.6f} | '
            f'{item["f010"]:.6f} | {item["f011"]:.6f} | {item["ae17"]:.6f} | {item["exc"]:.6f} | {item["op"]:.6f} | {item["composite_f010"]:.6f} |'
        )
    lines.extend([
        '',
        f'JW moved from `{initial["jw"]:.12f}` to `{final["jw"]:.12f}` (`{final["jw"] - initial["jw"]:+.6f}`).',
        'F011 JW at the same local counts is in the comparison CSV. A lower training JW is not a claim about Clean28.',
        '',
        '## F010 versus F011',
        '',
        '| Local | F010 JW | F011 JW | Difference | F010 Clean28 | F011 Clean28 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in comparison:
        clean = '' if row['f010_clean28'] == '' else f'{float(row["f010_clean28"]):.12f}'
        lines.append(
            f'| {row["local_update"]} | {row["f010_jw"]:.6f} | {row["f011_jw"]:.6f} | {row["jw_difference"]:+.6f} | {clean} | {float(row["f011_clean28"]):.12f} |'
        )
    lines.extend(['', '## Clean28', '', '| Checkpoint | Clean28 | Versus F011 same count | Versus historical best |', '| --- | ---: | ---: | ---: |'])
    for row in clean_rows:
        lines.append(
            f'| {row["name"]} | {row["clean28"]:.12f} | {row["clean28"] - F011_CLEAN28[row["local_update"]]:+.6f} | '
            f'{row["clean28"] - b8.HISTORICAL_BEST_CLEAN28:+.6f} |'
        )
    best = min(clean_rows, key=lambda row: row['clean28'])
    lines.extend([
        '',
        f'Descriptive minimum among the four frozen checkpoints: `{best["name"]}` at `{best["clean28"]:.12f}`.',
        f'Historical best `{b8.HISTORICAL_BEST_CLEAN28:.12f}`. F011 t64 `{F011_CLEAN28[64]:.12f}`. F011 t251 `{F011_CLEAN28[251]:.12f}`.',
        'Clean28 was not used to train, to set `c010`, or to choose a checkpoint. It is a development-set observation.',
        '',
        '## Databases',
        '',
        'JW mean is the within-database mean of `|r| / A_d`, the same reduction as the F011 database table. JW contribution keeps the global `1/251` denominator.',
        '',
        '| Checkpoint | Database | Count | A_d | MAE | JW mean | JW contribution | Versus F011 JW mean | cos(piece, JW) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in database_rows:
        if row['database'] not in arm.FOCUS and row['checkpoint'] not in ('initial', 't251'):
            continue
        versus = '' if row['jw_mean_minus_f011'] == '' else f'{row["jw_mean_minus_f011"]:+.6f}'
        cosine = '' if 'cosine_jw' not in row else f'{row["cosine_jw"]:.6f}'
        lines.append(
            f'| {row["checkpoint"]} | {row["database"]} | {row["count"]} | {row["A_d"]:.6f} | {row["mae"]:.6f} | '
            f'{row["jw_mean"]:.6f} | {row["jw_contribution"]:.6f} | {versus} | {cosine} |'
        )
    lines.extend(['', '## Gradient geometry at F010 weights', '', '| Checkpoint | JW/F010 | JW/operator | JW/total | F010/F011 | Cancellation |', '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in geometry_rows:
        lines.append(
            f'| {row["checkpoint"]} | {row["cos_jw_f010"]:.6f} | {row["cos_jw_op"]:.6f} | {row["cos_jw_total"]:.6f} | '
            f'{row["cos_f010_f011"]:.6f} | {row["cancellation_ratio"]:.6f} |'
        )
    lines.extend([
        '',
        'F011-trajectory cosines were measured at the F011 weights. The cosines above are measured at the F010 weights. They are not the same parameter vectors after t0.',
        '',
    ])
    if ratios:
        lines.extend(['## Scientific ratios', '', 'Ratios use the original relchem loss, not F010.', ''])
        for name, science in ratios.items():
            lines.append(f'- `{name}`: ' + ', '.join(f'{task} {science[task]:.6f}' for task in b8.TASKS))
        lines.append('')
    lines.extend([
        '## Runtime',
        '',
        f'- Updates: {0 if not logs else logs[-1]["local_update"]}',
        f'- Final cumulative step: {0 if not logs else logs[-1]["cumulative_optimizer_step"]}',
        f'- Wall seconds inside optimizer updates: `{seconds:.3f}`',
        f'- Peak allocated bytes during updates: `{peak}`',
        '- No interruption is recorded unless a resume line appears in the training log.',
        '',
        '## Git',
        '',
        'Provenance is the commit that contains this report. The arm started from `0c6b55dab355996e2e5746df0693c92e65a27965`.',
        '',
    ])
    (b8.REPO / 'lap_f010_uniform_normalized_training_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    source = b8.verify_sources()
    receipt_path = OUT / 'normalization_receipt.json'
    if receipt_path.is_file() and b8.read_json(receipt_path).get('live', {}).get('passed'):
        receipt = b8.read_json(receipt_path)
    else:
        receipt = cpu_gates()
        receipt = live_gates(source, receipt)
        b8.write_json(receipt_path, receipt)
        print('F010_PREFLIGHT', receipt['c'], receipt['cosine_jw_f010'], receipt['cosine_f010_f011'], flush=True)
    status_path = OUT / 'training_status.json'
    if not (status_path.is_file() and b8.read_json(status_path).get('complete')):
        train(source, receipt['c'], receipt['scales'])
    status = b8.read_json(status_path)
    if status['local_update'] != NEW_UPDATES or status['cumulative_optimizer_step'] != PARENT_CURSOR + NEW_UPDATES:
        _stop('The arm did not finish at cumulative step 379')
    metrics = training_metrics(source, receipt['scales'], receipt['c'])
    evaluate_clean28(source)
    clean_rows = [b8.read_json(OUT / f'validation_t{cursor}.json') for cursor in RAW_CHECKPOINTS]
    ratios = qualify(source, clean_rows)
    geometry = gradient_audit(source, receipt['c'], receipt['scales'])
    label = arm.classification(clean_rows, ratios)
    render(receipt, status, metrics, clean_rows, ratios, geometry, label)
    print('F010_CLASS', label, flush=True)
    return label


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        execute()
    except SystemExit as error:
        print('F010_STOP', error, flush=True)
        raise
