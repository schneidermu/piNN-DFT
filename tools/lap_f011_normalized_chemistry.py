"""One continuation from B8 t128. Chemistry weights gain a frozen 1/A_d factor.

The legacy database factor stays outside the square root. c matches the full-set
chemistry gradient norm to the original relchem gradient and is not retuned.
AdamW moments stay as saved under the old objective. No SWA and no Clean28
access until the 251 updates have finished.
"""
import hashlib
import math
import time
from pathlib import Path

import lap_b8_adamw as b8
import lap_b8_swa_continuation as continuation
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np
import torch

PROTOCOL_ID = 'f011-normalized-chemistry-v1'
PARENT_CURSOR = continuation.PARENT_CURSOR
PARENT_LR = continuation.PARENT_LR
NEW_LR = continuation.NEW_LR
NEW_UPDATES = 251
RAW_CHECKPOINTS = (64, 128, 192, 251)
COUNT = 251
LIVE_RELATIVE_L2 = 1e-6
SCALAR_TOLERANCE = 1e-6
COSINE_TOLERANCE = 1e-8
OUT = b8.REPO.parent / 'lap_f011_normalized_chemistry_20261011'
PARENT_CHECKPOINT = continuation.PARENT_CHECKPOINT
PARENT_MANIFEST = continuation.PARENT_MANIFEST
MANIFEST_SHA = continuation.MANIFEST_SHA
MOMENT_PREFIX = '4659e23c'
CHECKPOINT_PREFIX = '1f38f7f8'
FOCUS = ('DBH76', 'pTC13', 'PA8', 'ABDE4', 'MGAE109', 'NCCE31')
PRIOR_JW = {
    'initial': 0.1564146381387076,
    'continuation_t251': 0.15059583661919929,
}
PRIOR_RELCHEM = {
    'initial': 1.195939284232437,
    'continuation_t251': 1.1706491841330284,
}
PRIOR_CLEAN28 = {
    'b8_t128': 9.096700111658418,
    'continuation_t251': 9.087061815657618,
}


def f011_coefficients(residuals, raw_weights, scales, databases):
    """Weights for (1/251) sum f_d/A_d * d(phi)/dr. A_d is outside phi."""
    coefficients = np.empty(len(residuals), dtype=np.float64)
    for index, residual in enumerate(residuals):
        coefficients[index] = raw_weights[index] * fact.deriv_original(residual) / (scales[databases[index]] * len(residuals))
    return coefficients


def jw_coefficients(residuals, scales, databases):
    coefficients = np.empty(len(residuals), dtype=np.float64)
    for index, residual in enumerate(residuals):
        coefficients[index] = fact.deriv_l1(residual) / (scales[databases[index]] * len(residuals))
    return coefficients


def chemistry_scale(coefficient, reference_scale):
    """Multiply one production singleton by c/A_d. Do not divide the residual."""
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError('c must be finite and strictly positive')
    if not math.isfinite(reference_scale) or reference_scale <= 0.0:
        raise ValueError('A_d must be finite and strictly positive')
    return coefficient / reference_scale


def calibrate_norm(old_gradient, f011_gradient):
    old_norm = float(np.linalg.norm(old_gradient))
    f011_norm = float(np.linalg.norm(f011_gradient))
    if old_norm <= 0.0 or f011_norm <= 0.0 or not math.isfinite(old_norm * f011_norm):
        raise ValueError('Chemistry gradient norms are not strictly positive')
    coefficient = old_norm / f011_norm
    scaled = coefficient * f011_gradient
    return {
        'old_norm': old_norm,
        'f011_norm': f011_norm,
        'c': coefficient,
        'scaled_norm': float(np.linalg.norm(scaled)),
        'relative_norm_error': abs(float(np.linalg.norm(scaled)) - old_norm) / old_norm,
        'cosine_old': prior.cosine(old_gradient, scaled),
        'scaled_gradient': scaled,
        'f011_gradient': np.asarray(f011_gradient, dtype=np.float64),
    }


def scales_sha256(scales):
    digest = hashlib.sha256()
    for database in sorted(scales):
        digest.update(database.encode())
        digest.update(np.float64(scales[database]).tobytes())
    return digest.hexdigest()


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason})
    raise SystemExit(reason)


def _confine(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(OUT.resolve()):
        raise ValueError('Refusing to write outside the F011 scratch')
    return resolved


def _flat(names, grad):
    return np.concatenate([grad[name].detach().reshape(-1).double().cpu().numpy() for name in names])


def _load_geometry():
    rows, jacobian = j251._load_reactions(j251.OUT / 'b8_t128' / 'reactions.npz')
    if rows is None or jacobian.shape != (COUNT, 9446):
        _stop('B8 t128 reaction Jacobian is missing')
    residuals = [row['residual'] for row in rows]
    references = [row['reference'] for row in rows]
    databases = [row['database'] for row in rows]
    scales, counts = j251.database_reference_scales(references, databases)
    frozen = b8.read_json(j251.OUT / 'reference_scales.json')['scales']
    if any(abs(scales[key] - frozen[key]) > 0.0 for key in scales):
        _stop('A_d does not match the frozen J251 reference scales')
    for database, expected in fact.PRIOR_SCALES.items():
        if abs(scales[database] - expected) > 5e-4:
            _stop(f'A_d for {database} disagrees with the factorial report')
    b8._import_run()
    raw = np.array([j251._factor(database) for database in databases], dtype=np.float64)
    saved = prior._load_vector(prior.OUT / 'b8_t128' / 'relchem.npy.npz')
    if saved is None or saved.shape != (9446,):
        _stop('Saved production relchem gradient is missing')
    return rows, jacobian, residuals, databases, scales, counts, raw, saved


def cpu_gates():
    rows, jacobian, residuals, databases, scales, counts, raw, saved = _load_geometry()
    g_old = fact.production_coefficients(residuals, raw) @ jacobian
    g_f011 = f011_coefficients(residuals, raw, scales, databases) @ jacobian
    g_jw = jw_coefficients(residuals, scales, databases) @ jacobian
    if float(np.linalg.norm(g_old - saved) / np.linalg.norm(saved)) > COSINE_TOLERANCE:
        _stop('Chain-rule relchem gradient does not match the saved production gradient')
    if abs(fact.production_scalar(residuals, raw) - PRIOR_RELCHEM['initial']) > SCALAR_TOLERANCE:
        _stop('Cached Full251 relchem scalar does not match the receipt')
    calibration = calibrate_norm(g_old, g_f011)
    factorial = b8.read_json(fact.OUT / 'b8_t128' / 'result.json')
    expected_cosine = factorial['cosine']['F010']['F011']
    observed = prior.cosine(g_jw, calibration['scaled_gradient'])
    if abs(observed - expected_cosine) > COSINE_TOLERANCE or observed <= 0.0:
        _stop('F011 is not the factorial JW-aligned direction')
    if calibration['relative_norm_error'] > 1e-12 or not (calibration['c'] > 0.0):
        _stop('Norm calibration failed')
    vectors = {
        task: prior._load_vector(prior.OUT / 'b8_t128' / f'{task}.npy.npz')
        for task in ('ae17', 'exc', 'op')
    }
    if any(vector is None for vector in vectors.values()):
        _stop('Saved physical-task gradients are missing')
    chemistry = b8.LAMBDAS['relchem'] * calibration['scaled_gradient']
    pieces = {
        'relchem': prior.predicted_change(g_jw, chemistry),
        'ae17': prior.predicted_change(g_jw, b8.LAMBDAS['ae17'] * vectors['ae17']),
        'exc': prior.predicted_change(g_jw, b8.LAMBDAS['exc'] * vectors['exc']),
        'op': prior.predicted_change(g_jw, b8.LAMBDAS['op'] * vectors['op']),
    }
    total = chemistry + sum(b8.LAMBDAS[task] * vectors[task] for task in ('ae17', 'exc', 'op'))
    combined = prior.predicted_change(g_jw, total)
    if abs(sum(pieces.values()) - combined) > 1e-6:
        _stop('JW directional contributions do not sum')
    old_total = sum(b8.LAMBDAS[task] * vector for task, vector in (
        ('relchem', saved), ('ae17', vectors['ae17']), ('exc', vectors['exc']), ('op', vectors['op']),
    ))
    receipt = {
        'protocol_id': PROTOCOL_ID,
        'c': calibration['c'],
        'old_norm': calibration['old_norm'],
        'f011_norm': calibration['f011_norm'],
        'scaled_norm': calibration['scaled_norm'],
        'relative_norm_error': calibration['relative_norm_error'],
        'cosine_old_f011': calibration['cosine_old'],
        'cosine_jw_f011': observed,
        'factorial_cosine_jw_f011': expected_cosine,
        'cosine_jw_old': prior.cosine(g_jw, saved),
        'cosine_jw_operator': prior.cosine(g_jw, vectors['op']),
        'cosine_jw_total_new': prior.cosine(g_jw, total),
        'cosine_total_new_total_old': prior.cosine(total, old_total),
        'jw_predicted_change': combined,
        'jw_unit_predicted_change': prior.normalized_change(g_jw, total),
        'jw_contributions': pieces,
        'scales': scales,
        'counts': counts,
        'scales_sha256': scales_sha256(scales),
        'optimizer_updates': 0,
        'adamw_moments': 'retained from the original chemistry objective; not claimed to be optimal for F011',
    }
    return receipt, rows


def _batches():
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    stored = b8.read_json(PARENT_MANIFEST)
    if stored != batches or b8.sha256_file(PARENT_MANIFEST) != MANIFEST_SHA:
        _stop('B8 manifest does not match the quarter-rate continuation schedule')
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


def _new_optimizer(parameters):
    return torch.optim.AdamW(
        list(parameters.values()), lr=NEW_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)


def f011_mean(model, shadow, bundle, specs, dispersion, scales, coefficient):
    torch.cuda.synchronize()
    started = time.perf_counter()
    pairs = []
    infos = []
    for spec in specs:
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        value, grad = b8._singleton(model, shadow, reaction, dispersion)
        scale = chemistry_scale(coefficient, scales[spec['database']])
        scaled_grad = {name: piece.detach() * scale for name, piece in grad.items()}
        flat = torch.cat([piece.reshape(-1) for piece in scaled_grad.values()])
        infos.append({
            'identity': spec['identity'], 'variant': spec['variant'], 'database': spec['database'],
            'loss': float(value) * scale, 'unscaled_loss': float(value),
            'grad_norm': float(torch.linalg.vector_norm(flat)),
        })
        pairs.append((float(value) * scale, scaled_grad))
        del reaction, grad, scaled_grad, flat
        torch.cuda.empty_cache()
    loss, grad = b8.combine_gradients(pairs)
    torch.cuda.synchronize()
    return loss, grad, infos, time.perf_counter() - started


def _selected(bundle):
    evaluation = b8.read_json(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    if b8.sha256_file(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json') != b8.EVAL_MANIFEST_SHA:
        _stop('Evaluation manifest hash changed')
    rows = [row for row in evaluation['rows'] if row['task'] == 'relchem']
    if len(rows) != COUNT or len({row['identity'] for row in rows}) != COUNT:
        _stop('J251 evaluation manifest is not 251 unique reactions')
    specs = []
    for row in rows:
        meta = bundle.reactions[row['identity']]
        specs.append({
            'identity': row['identity'], 'variant': row['variant'],
            'database': meta['database'], 'reaction_id': meta['reaction_id'],
        })
    return specs


def live_full_gradient(model, shadow, bundle, dispersion, scales, coefficient, names):
    specs = _selected(bundle)
    old_loss = 0.0
    new_loss = 0.0
    old = None
    new = None
    for index, spec in enumerate(specs):
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        value, grad = b8._singleton(model, shadow, reaction, dispersion)
        scale = chemistry_scale(coefficient, scales[spec['database']])
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
            print('F011_PREFLIGHT_REACTION', index + 1, flush=True)
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
    optimizer = _new_optimizer(parameters)
    moment = continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], PARENT_CURSOR, PARENT_LR)
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
        reference_loss, reference_grad, _seconds = b8.reference_mean(
            model, shadow, bundle, batches[0]['relchem'], dispersion)
        loss, grad, _infos, _elapsed = b8.qualified_mean(
            model, shadow, bundle, batches[0]['relchem'], dispersion)
        if (b8.relative_l2(grad, reference_grad) > b8.RELATIVE_L2_LIMIT
                or abs(loss - reference_loss) > b8.RELATIVE_L2_LIMIT):
            _stop('Unscaled chemistry mean failed its parity check')
        other_losses, other_raw, _system, _other_seconds = b8.other_tasks(
            model, shadow, bundle, batches[0], dispersion, mrks_dispersion)
        _measured, measured_raw = run.measure(
            model, shadow, bundle, b8.load_j251_rows()[0], dispersion, mrks_dispersion, exc_chunk_size=4096)
        other_l2 = {task: b8.relative_l2(other_raw[task], measured_raw[task]) for task in ('ae17', 'exc', 'op')}
        if any(value > b8.OTHER_TASK_L2_LIMIT for value in other_l2.values()):
            _stop('AE17, Exc, or operator gradient differs from the original measurement path')
        scaled_loss, scaled_grad, infos, _elapsed = f011_mean(
            model, shadow, bundle, batches[0]['relchem'], dispersion, receipt['scales'], receipt['c'])
        manual = 0.0
        for info in infos:
            manual += info['unscaled_loss'] * chemistry_scale(receipt['c'], receipt['scales'][info['database']])
        if abs(scaled_loss - manual / 8) > 1e-12:
            _stop('Minibatch F011 scalar is not the mean of c/A_d times the production singletons')
        old_scalar, new_scalar, old_grad, new_grad = live_full_gradient(
            model, shadow, bundle, dispersion, receipt['scales'], receipt['c'], names)
        if abs(old_scalar - PRIOR_RELCHEM['initial']) > SCALAR_TOLERANCE:
            _stop(f'Live Full251 relchem scalar {old_scalar} missed the receipt')
        _rows, jacobian, residuals, databases, scales, _counts, raw, _saved = _load_geometry()
        chain_old = fact.production_coefficients(residuals, raw) @ jacobian
        chain_new = receipt['c'] * (f011_coefficients(residuals, raw, scales, databases) @ jacobian)
        old_gap = float(np.linalg.norm(_flat(names, old_grad) - chain_old) / np.linalg.norm(chain_old))
        new_gap = float(np.linalg.norm(_flat(names, new_grad) - chain_new) / np.linalg.norm(chain_new))
        if old_gap > LIVE_RELATIVE_L2 or new_gap > LIVE_RELATIVE_L2:
            _stop(f'Live full gradients disagree with the chain rule: {old_gap}, {new_gap}')
        formula = receipt['c'] * fact.diagnostic_scalar(
            residuals, scales, databases, raw / raw.mean(), 'F011') * float(raw.mean())
        if abs(new_scalar - formula) > SCALAR_TOLERANCE:
            _stop('Live F011 scalar disagrees with the independent formula')
        if abs(float(np.linalg.norm(chain_new)) - receipt['old_norm']) / receipt['old_norm'] > 1e-12:
            _stop('c does not match the full-set gradient norms')
        if {int(state['step']) for state in optimizer.state.values()} != {PARENT_CURSOR}:
            _stop('Preflight took an optimizer step')
        if continuation.moment_sha(optimizer.state_dict()) != moment or run.existing.digest(model) != digest:
            _stop('Preflight changed the model or the AdamW moments')
        if b8.sha256_file(PARENT_CHECKPOINT) != parent_sha:
            _stop('Preflight changed the parent checkpoint')
        del reference_loss, loss, grad, scaled_grad, old_grad, new_grad, measured_raw
    finally:
        bundle.close()
        del model, shadow, optimizer
        torch.cuda.empty_cache()
    receipt = dict(receipt)
    receipt['live'] = {
        'passed': True,
        'old_scalar': old_scalar,
        'new_scalar': new_scalar,
        'old_relative_l2': old_gap,
        'new_relative_l2': new_gap,
        'other_task_relative_l2': other_l2,
        'other_losses': other_losses,
        'parent_checkpoint_sha256': parent_sha,
        'optimizer_moment_sha256': moment,
        'loaded_model_sha256': digest,
        'optimizer_step': False,
    }
    return receipt


def _vector_delta(current, reference):
    parts = [(current[name].detach().double().cpu() - reference[name].detach().double().cpu()).reshape(-1) for name in reference]
    return float(torch.linalg.vector_norm(torch.cat(parts)))


def _save_checkpoint(path, model, optimizer, cursor, logs, coefficient, parent_sha):
    path = _confine(path)
    payload = {
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': {key: value for key, value in optimizer.state_dict().items()},
        'rng': b8._capture_rng(),
        'cursor': cursor,
        'logs': list(logs),
        'protocol_id': PROTOCOL_ID,
        'manifest_sha256': MANIFEST_SHA,
        'c': coefficient,
        'lambdas': dict(b8.LAMBDAS),
        'lr': NEW_LR,
        'scheduler': None,
        'parent_checkpoint_sha256': parent_sha,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path, model, optimizer, coefficient):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('c') != coefficient or saved.get('scheduler') is not None:
        _stop('Resume checkpoint is not this F011 arm')
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
        optimizer = _new_optimizer(parameters)
        local_update, logs = _load_checkpoint(latest, model, optimizer, coefficient)
        parameters = _check_model(model)
    else:
        saved = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
        parameters = _check_model(model)
        model.load_state_dict(saved['model'])
        parameters = _check_model(model)
        optimizer = _new_optimizer(parameters)
        continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], PARENT_CURSOR, PARENT_LR)
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
            relchem_loss, relchem_grad, infos, _elapsed = f011_mean(
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
                'local_update': done,
                'cumulative_optimizer_step': PARENT_CURSOR + done,
                'manifest_index': batch['cursor'],
                'chemistry_loss': relchem_loss,
                'ae17_loss': other_losses['ae17'],
                'exc_loss': other_losses['exc'],
                'op_loss': other_losses['op'],
                'composite_loss': sum(b8.LAMBDAS[task] * value for task, value in (
                    ('relchem', relchem_loss), ('ae17', other_losses['ae17']),
                    ('exc', other_losses['exc']), ('op', other_losses['op']),
                )),
                'lr': NEW_LR,
                'gradient_norm': b8._grad_norm(joint),
                'chemistry_gradient_norm': b8._grad_norm(relchem_grad),
                'ae17_gradient_norm': b8._grad_norm(other_raw['ae17']),
                'exc_gradient_norm': b8._grad_norm(other_raw['exc']),
                'op_gradient_norm': b8._grad_norm(other_raw['op']),
                'update_norm': _vector_delta(parameters, before_weights),
                'displacement_from_t128': _vector_delta(parameters, origin),
                'nonfinite': False,
                'seconds': time.perf_counter() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'before_sha256': before,
                'after_sha256': run.existing.digest(model),
                'reactions': infos,
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
            print('F011_UPDATE', done, f'{row["seconds"]:.3f}', f'{row["chemistry_loss"]:.6f}', flush=True)
    finally:
        bundle.close()
    if b8.sha256_file(PARENT_CHECKPOINT) != parent_sha:
        _stop('Training changed the B8 checkpoint')
    return logs[-1]['local_update']


def _load_eval_model(source, path):
    run = b8._import_run()
    model, shadow = run.model_at(source['initial_state'])
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    _check_model(model)
    return model, shadow, run


def training_metrics(source, scales, coefficient):
    run = b8._import_run()
    targets = [('initial', 0, PARENT_CHECKPOINT)] + [
        (f't{cursor}', cursor, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt') for cursor in RAW_CHECKPOINTS
    ]
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        specs = _selected(bundle)
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        rows_out = []
        for name, cursor, path in targets:
            done = OUT / f'metrics_{name}.json'
            if done.is_file():
                rows_out.append(b8.read_json(done))
                continue
            if not path.is_file():
                _stop('Metric checkpoint is missing: ' + name)
            model, shadow, _run = _load_eval_model(source, path)
            shadow.load_state_dict(model.state_dict(), strict=True)
            before = run.existing.digest(model)
            reactions = []
            for spec in specs:
                reaction = b8._load_reaction(bundle, 'relchem', spec)
                with torch.no_grad():
                    prediction, reference = j251._prediction(shadow, reaction, dispersion)
                residual = float((prediction - reference).detach().cpu())
                reference_value = float(reference.detach().cpu())
                factor = j251._factor(spec['database'])
                penalty = factor * math.sqrt(1e-20 + residual * residual)
                reactions.append({
                    'identity': spec['identity'], 'reaction_id': spec['reaction_id'], 'database': spec['database'],
                    'residual': residual, 'absolute_residual': abs(residual), 'reference': reference_value,
                    'A_d': scales[spec['database']], 'legacy_factor': factor,
                    'old_penalty': penalty,
                    'f011_penalty': penalty * chemistry_scale(coefficient, scales[spec['database']]),
                })
                del reaction, prediction, reference
                torch.cuda.empty_cache()
            ae17 = continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
            mrks = mrks_scalars(model, bundle, mrks_dispersion)
            if run.existing.digest(model) != before:
                _stop('Training-set evaluation changed the checkpoint')
            databases = [row['database'] for row in reactions]
            residuals = [row['residual'] for row in reactions]
            payload = {
                'name': name, 'local_update': cursor, 'checkpoint_sha256': b8.sha256_file(path),
                'jw': j251.j251_wtmad_style(residuals, databases, scales),
                'mae': j251.j251_mae(residuals),
                'relchem': sum(row['old_penalty'] for row in reactions) / COUNT,
                'f011': sum(row['f011_penalty'] for row in reactions) / COUNT,
                'ae17': ae17, 'exc': mrks['exc'], 'op': mrks['op'],
                'reactions': reactions,
            }
            payload['composite'] = (
                b8.LAMBDAS['relchem'] * payload['f011'] + b8.LAMBDAS['ae17'] * payload['ae17']
                + b8.LAMBDAS['exc'] * payload['exc'] + b8.LAMBDAS['op'] * payload['op']
            )
            if name == 'initial' and (
                    abs(payload['relchem'] - PRIOR_RELCHEM['initial']) > SCALAR_TOLERANCE
                    or abs(payload['jw'] - PRIOR_JW['initial']) > SCALAR_TOLERANCE):
                _stop('Initial J251 metrics do not match the B8 t128 receipts')
            b8.write_json(done, payload)
            rows_out.append(payload)
            print('F011_METRICS', name, payload['jw'], payload['relchem'], payload['f011'], flush=True)
            del model, shadow
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    return rows_out


def continuation_chemistry_scalar(model, shadow, bundle, dispersion, task):
    evaluation = b8.read_json(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    selected = [row for row in evaluation['rows'] if row['task'] == task]
    total = 0.0
    run = b8._import_run()
    for spec in selected:
        reaction = b8._load_reaction(bundle, task, {
            'identity': spec['identity'], 'variant': spec['variant'],
            'database': bundle.reactions[spec['identity']]['database'],
        })
        with torch.no_grad():
            total += float(run.chemistry(model, shadow, reaction, dispersion)())
        del reaction
        torch.cuda.empty_cache()
    return total / len(selected)


def mrks_scalars(model, bundle, dispersion):
    run = b8._import_run()
    totals = {'exc': 0.0, 'op': 0.0}
    systems = sorted(bundle.systems)
    for identity in systems:
        system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
        exc, op = run.existing.core.make_mrks_objective_factories(
            model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=dispersion)
        with torch.no_grad():
            totals['exc'] += float(exc().detach())
            totals['op'] += float(op().detach())
        del system, exc, op
        torch.cuda.empty_cache()
    return {task: totals[task] / len(systems) for task in totals}


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
            model, _shadow, _run = _load_eval_model(source, path)
            before = run.existing.digest(model)
            result = validation(model, bundle)
            if run.existing.digest(model) != before or len([row for row in result['reaction_rows'] if row['clean']]) != 28:
                _stop('Clean28 evaluation changed the model or the subset')
            b8.write_json(receipt, {
                'name': f't{cursor}', 'local_update': cursor, 'clean28': result['clean28'],
                'checkpoint_sha256': b8.sha256_file(path), 'optimizer_step': False,
            })
            print('F011_CLEAN28', cursor, result['clean28'], flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        bundle.close()


def qualify(source, clean_rows):
    ratios = {}
    bundle_needed = [row for row in clean_rows if b8.beats_historical(row['clean28'])]
    if not bundle_needed:
        return ratios
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for row in bundle_needed:
            path = OUT / f'science_{row["name"]}.json'
            if path.is_file():
                ratios[row['name']] = b8.read_json(path)['ratios']
                continue
            model, shadow, _run = _load_eval_model(source, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{row["local_update"]}.pt')
            relchem = continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'relchem')
            ae17 = continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
            mrks = mrks_scalars(model, bundle, mrks_dispersion)
            science = {
                'relchem': relchem / b8.BASELINE['relchem'],
                'ae17': ae17 / b8.BASELINE['ae17'],
                'exc': mrks['exc'] / b8.BASELINE['exc'],
                'op': mrks['op'] / b8.BASELINE['op'],
            }
            b8.write_json(path, {'name': row['name'], 'ratios': science, 'used_f011_in_ratio': False})
            ratios[row['name']] = science
            del model, shadow
    finally:
        bundle.close()
    return ratios


def classification(clean_rows, ratios):
    if len(clean_rows) != 4 or any(not math.isfinite(row['clean28']) for row in clean_rows):
        return 'INVALID — experiment or verification failure'
    screened = [row for row in clean_rows if b8.beats_historical(row['clean28'])]
    if not screened:
        return 'NO-GO — Clean28 gate not passed'
    if any(row['name'] not in ratios or not b8.scientifically_eligible(ratios[row['name']]) for row in screened):
        return 'NO-GO — scientific qualification failed'
    return 'GO — scientific qualification passed'


def _write_tables(receipt, status, metrics, clean_rows):
    preflight_rows = [
        {'quantity': 'c', 'value': receipt['c']},
        {'quantity': 'old_norm', 'value': receipt['old_norm']},
        {'quantity': 'f011_norm', 'value': receipt['f011_norm']},
        {'quantity': 'scaled_norm', 'value': receipt['scaled_norm']},
        {'quantity': 'relative_norm_error', 'value': receipt['relative_norm_error']},
        {'quantity': 'cosine_old_f011', 'value': receipt['cosine_old_f011']},
        {'quantity': 'cosine_jw_f011', 'value': receipt['cosine_jw_f011']},
        {'quantity': 'cosine_jw_total_new', 'value': receipt['cosine_jw_total_new']},
        {'quantity': 'jw_predicted_change', 'value': receipt['jw_predicted_change']},
    ]
    for task, value in receipt['jw_contributions'].items():
        preflight_rows.append({'quantity': f'jw_contribution_{task}', 'value': value})
    _csv(b8.REPO / 'lap_f011_preflight_gradients.csv', preflight_rows, ('quantity', 'value'))
    logs = [] if status is None else status['logs']
    _csv(b8.REPO / 'lap_f011_training_metrics.csv', logs, (
        'local_update', 'cumulative_optimizer_step', 'chemistry_loss', 'ae17_loss', 'exc_loss', 'op_loss',
        'composite_loss', 'lr', 'gradient_norm', 'update_norm', 'nonfinite', 'seconds', 'peak_allocated_bytes',
    ))
    checkpoint_rows = []
    database_rows = []
    reaction_rows = []
    for item in metrics:
        checkpoint_rows.append({key: item[key] for key in (
            'name', 'local_update', 'jw', 'mae', 'relchem', 'f011', 'ae17', 'exc', 'op', 'composite', 'checkpoint_sha256',
        )})
        grouped = {}
        for row in item['reactions']:
            grouped.setdefault(row['database'], []).append(row)
            reaction_rows.append({'checkpoint': item['name'], **{key: row[key] for key in row if key != 'reference'}})
        for database, group in sorted(grouped.items()):
            database_rows.append({
                'checkpoint': item['name'], 'database': database, 'count': len(group),
                'mae': sum(row['absolute_residual'] for row in group) / len(group),
                'jw': sum(row['absolute_residual'] / row['A_d'] for row in group) / len(group),
                'relchem': sum(row['old_penalty'] for row in group) / COUNT,
                'f011': sum(row['f011_penalty'] for row in group) / COUNT,
            })
    _csv(b8.REPO / 'lap_f011_checkpoint_metrics.csv', checkpoint_rows, (
        'name', 'local_update', 'jw', 'mae', 'relchem', 'f011', 'ae17', 'exc', 'op', 'composite', 'checkpoint_sha256',
    ))
    _csv(b8.REPO / 'lap_f011_database_errors.csv', database_rows, ('checkpoint', 'database', 'count', 'mae', 'jw', 'relchem', 'f011'))
    _csv(b8.REPO / 'lap_f011_reaction_errors.csv', reaction_rows, (
        'checkpoint', 'identity', 'reaction_id', 'database', 'residual', 'absolute_residual', 'A_d',
        'legacy_factor', 'old_penalty', 'f011_penalty',
    ))
    b8.write_json(b8.REPO / 'lap_f011_normalization_receipt.json', receipt)
    b8.write_json(b8.REPO / 'lap_f011_evaluation_receipts.json', {
        'metrics': checkpoint_rows, 'clean28': clean_rows, 'optimizer_updates': 0 if status is None else status['local_update'],
    })


def _csv(path, rows, fields):
    import csv
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def render(receipt, status, metrics, clean_rows, ratios, label):
    _write_tables(receipt, status, metrics, clean_rows)
    by_name = {item['name']: item for item in metrics}
    initial = by_name.get('initial')
    final = by_name.get('t251')
    logs = [] if status is None else status['logs']
    seconds = sum(row['seconds'] for row in logs)
    peak = max((row['peak_allocated_bytes'] for row in logs), default=0)
    lines = [
        '# F011 reference-normalized chemistry training',
        '',
        f'Classification: **{label}**.',
        '',
        'One arm from B8 t128. The chemistry term is `c * factor_d / A_d * sqrt(r^2 + 1e-20)`, averaged as in the B8 minibatch. Operator, Exc, AE17, lambdas, and the learning rate were not changed.',
        '',
        'The restored AdamW moments were accumulated under the original chemistry objective. That is a matched continuation control, not a claim that those moments are optimal for F011.',
        '',
        '## Calibration',
        '',
        f'- `c` = `{receipt["c"]:.12f}`',
        f'- `||g_old||` = `{receipt["old_norm"]:.6f}`',
        f'- `||g_F011||` = `{receipt["f011_norm"]:.6f}`',
        f'- `||c g_F011||` = `{receipt["scaled_norm"]:.6f}`',
        f'- Relative norm error `{receipt["relative_norm_error"]:.3e}`',
        f'- cos(old, c g_F011) = `{receipt["cosine_old_f011"]:.6f}`',
        f'- cos(JW, c g_F011) = `{receipt["cosine_jw_f011"]:.6f}`',
        f'- cos(JW, new total) = `{receipt["cosine_jw_total_new"]:.6f}`',
        f'- First-order predicted JW change under the full new gradient: `{receipt["jw_predicted_change"]:.6f}`',
        f'- A_d SHA256 `{receipt["scales_sha256"]}`',
        '',
        '## Training-set metrics',
        '',
        '| Checkpoint | JW | MAE | Original relchem | F011 | AE17 | Exc | Operator | Composite |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for item in metrics:
        lines.append(
            f'| {item["name"]} | {item["jw"]:.6f} | {item["mae"]:.6f} | {item["relchem"]:.6f} | {item["f011"]:.6f} | '
            f'{item["ae17"]:.6f} | {item["exc"]:.6f} | {item["op"]:.6f} | {item["composite"]:.6f} |'
        )
    if initial and final:
        lines.extend([
            '',
            f'JW moved from `{initial["jw"]:.6f}` to `{final["jw"]:.6f}` (`{final["jw"] - initial["jw"]:+.6f}`).',
            f'The quarter-rate continuation ended at JW `{PRIOR_JW["continuation_t251"]:.6f}` and original relchem `{PRIOR_RELCHEM["continuation_t251"]:.6f}`.',
            'The first-order prediction is the directional derivative at t128. It is not the predicted result of 251 AdamW updates.',
            '',
            '## Focus databases',
            '',
            '| Database | Initial JW | Final JW | Initial MAE | Final MAE |',
            '| --- | ---: | ---: | ---: | ---: |',
        ])
        for database in FOCUS:
            start = _database_mean(initial, database, 'jw')
            end = _database_mean(final, database, 'jw')
            lines.append(
                f'| {database} | {start:.6f} | {end:.6f} | {_database_mean(initial, database, "mae"):.6f} | {_database_mean(final, database, "mae"):.6f} |'
            )
    lines.extend(['', '## Clean28', '', '| Checkpoint | Clean28 | Versus continuation t251 | Versus historical best |', '| --- | ---: | ---: | ---: |'])
    for row in clean_rows:
        lines.append(
            f'| {row["name"]} | {row["clean28"]:.12f} | {row["clean28"] - PRIOR_CLEAN28["continuation_t251"]:+.6f} | '
            f'{row["clean28"] - b8.HISTORICAL_BEST_CLEAN28:+.6f} |'
        )
    lines.extend([
        '',
        f'B8 t128 reference `{PRIOR_CLEAN28["b8_t128"]:.12f}`. Continuation t251 reference `{PRIOR_CLEAN28["continuation_t251"]:.12f}`.',
        'Clean28 was not used to train, to set `c`, or to choose a checkpoint. It was used in earlier gradient audits, so this score is not an untouched holdout.',
        '',
    ])
    if ratios:
        lines.extend(['## Scientific ratios', '', 'Ratios use the original relchem loss, not F011.', ''])
        for name, science in ratios.items():
            lines.append(f'- `{name}`: ' + ', '.join(f'{task} {science[task]:.6f}' for task in b8.TASKS))
        lines.append('')
    lines.extend([
        '## Runtime',
        '',
        f'- Updates: {0 if not logs else logs[-1]["local_update"]}',
        f'- Wall seconds inside optimizer updates: `{seconds:.3f}`',
        f'- Peak allocated bytes: `{peak}`',
        '',
        '## Git',
        '',
        'Provenance is the commit that contains this report.',
        '',
    ])
    (b8.REPO / 'lap_f011_normalized_chemistry_training_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def _database_mean(metric, database, kind):
    group = [row for row in metric['reactions'] if row['database'] == database]
    if kind == 'mae':
        return sum(row['absolute_residual'] for row in group) / len(group)
    return sum(row['absolute_residual'] / row['A_d'] for row in group) / len(group)


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    source = b8.verify_sources()
    receipt_path = OUT / 'normalization_receipt.json'
    if receipt_path.is_file() and b8.read_json(receipt_path).get('live', {}).get('passed'):
        receipt = b8.read_json(receipt_path)
    else:
        receipt, _rows = cpu_gates()
        receipt = live_gates(source, receipt)
        b8.write_json(receipt_path, receipt)
        print('F011_PREFLIGHT', receipt['c'], receipt['cosine_jw_f011'], receipt['cosine_jw_total_new'], flush=True)
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
    label = classification(clean_rows, ratios)
    render(receipt, status, metrics, clean_rows, ratios, label)
    print('F011_CLASS', label, flush=True)
    return label


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        execute()
    except SystemExit as error:
        print('F011_STOP', error, flush=True)
        raise
