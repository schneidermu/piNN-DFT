"""One F010 continuation with the operator coefficient doubled.

Chemistry, AE17, Exc, the network, the optimizer, and the batch manifest stay
at the F010 values. Clean28 is read only after 251 updates.
"""
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import lap_b8_adamw as b8
import lap_b8_swa_continuation as continuation
import lap_f010_uniform_normalized_chemistry as f010
import lap_f011_normalized_chemistry as arm
import lap_f011_trajectory_gradient_audit as traj
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np
import torch

PROTOCOL_ID = 'f012-operator-priority-v1'
OUT = b8.REPO.parent / 'lap_f012_operator_priority_20261012'
PARENT_CHECKPOINT_SHA = '1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b'
PARENT_MOMENT_SHA = '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
C010_PRINTED = 9.974823077811
AD_SHA = '3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d'
LAMBDA_OP_F012 = 2.0 * b8.LAMBDAS['op']
F010_CLEAN28 = {64: 8.717171432568, 128: 8.689881959056, 192: 8.750258845113, 251: 8.814699816309}
F011_CLEAN28 = f010.F011_CLEAN28
GEOMETRY = ('t64', 't128', 't192', 't251')


def lambdas_f012():
    values = dict(b8.LAMBDAS)
    values['op'] = LAMBDA_OP_F012
    if tuple(values) != b8.TASKS or values['op'] != 2.0 * b8.LAMBDAS['op']:
        raise ValueError('F012 coefficients changed a non-operator task')
    return values


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _flat(names, grad):
    return np.concatenate([
        np.asarray(grad[name].detach().cpu().reshape(-1), dtype=np.float64) for name in names
    ])


def _named(names, shapes, flat):
    offset = 0
    tensors = {}
    for name in names:
        count = int(np.prod(shapes[name]))
        tensors[name] = torch.from_numpy(np.array(flat[offset:offset + count], dtype=np.float64)).reshape(shapes[name])
        offset += count
    return tensors


def _assemble(components, coefficients):
    return sum(coefficients[task] * components[task] for task in b8.TASKS)


def _load_components():
    receipt = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')
    if abs(receipt['c'] - C010_PRINTED) > 5e-13 or receipt['scales_sha256'] != AD_SHA:
        _stop('Frozen c010 or A_d hash does not match the F010 receipt')
    rows, jacobian, residuals, databases, scales, _counts, _raw, _saved = arm._load_geometry()
    if arm.scales_sha256(scales) != AD_SHA or scales != receipt['scales']:
        _stop('A_d values do not match the frozen calibration')
    chemistry = receipt['c'] * (f010.f010_raw_coefficients(residuals, scales, databases) @ jacobian)
    physical = {task: prior._load_vector(prior.OUT / 'b8_t128' / f'{task}.npy.npz') for task in ('ae17', 'exc', 'op')}
    jw = arm.jw_coefficients(residuals, scales, databases) @ jacobian
    if any(vector is None or vector.shape != chemistry.shape for vector in physical.values()):
        _stop('Saved B8 t128 physical gradients are missing')
    if float(np.linalg.norm(chemistry) - receipt['scaled_norm']) / receipt['scaled_norm'] > 1e-12:
        _stop('Reconstructed F010 chemistry gradient missed the frozen norm')
    if abs(prior.cosine(chemistry, jw) - 1.0) > 1e-8:
        _stop('F010 chemistry is not the JW direction at B8 t128')
    return {
        'relchem': chemistry, 'ae17': physical['ae17'], 'exc': physical['exc'], 'op': physical['op'], 'jw': jw,
        'c': receipt['c'], 'scales': scales, 'residuals': residuals, 'databases': databases, 'rows': rows,
    }


def preflight(source):
    path = OUT / 'preflight.json'
    if path.is_file() and b8.read_json(path).get('passed'):
        return b8.read_json(path)
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    parent_sha = b8.sha256_file(f010.PARENT_CHECKPOINT)
    if parent_sha != PARENT_CHECKPOINT_SHA:
        failures.append('parent checkpoint hash')
    if b8.sha256_file(continuation.PARENT_MANIFEST) != continuation.MANIFEST_SHA:
        failures.append('batch manifest hash')
    batches = b8.build_batches(b8.load_j251_rows())
    if b8.read_json(continuation.PARENT_MANIFEST) != batches:
        failures.append('batch manifest entries')
    components = _load_components()
    coefficients = lambdas_f012()
    base = _assemble(components, b8.LAMBDAS)
    doubled = _assemble(components, coefficients)
    if float(np.linalg.norm(doubled - (base + b8.LAMBDAS['op'] * components['op']))) > 1e-8:
        failures.append('F012 is not F010 plus the original weighted operator gradient')
    run = b8._import_run()
    saved = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    model, shadow = run.model_at(source['initial_state'])
    parameters = f010._check_model(model)
    model.load_state_dict(saved['model'])
    parameters = f010._check_model(model)
    names = tuple(parameters)
    shapes = {name: tuple(value.shape) for name, value in parameters.items()}
    raw = {task: _named(names, shapes, components[task]) for task in b8.TASKS}
    produced = _flat(names, b8.scalarize(raw, coefficients))
    if float(np.linalg.norm(produced - doubled) / np.linalg.norm(doubled)) > 1e-12:
        failures.append('scalarize does not reproduce the assembled F012 gradient')
    norms = {task: float(np.linalg.norm(coefficients[task] * components[task])) for task in b8.TASKS}
    expected = {'relchem': 0.2565537743567935, 'ae17': 0.2877941162405871, 'exc': 0.30371084045950864, 'op': 0.41214462680519004}
    if any(abs(norms[task] - expected[task]) / expected[task] > 1e-9 for task in b8.TASKS):
        failures.append('weighted norms missed the F010 regression values')
    dots = {
        'jw': float(np.dot(components['jw'], doubled)),
        'op': float(np.dot(components['op'], doubled)),
        'ae17': float(np.dot(components['ae17'], doubled)),
        'exc': float(np.dot(components['exc'], doubled)),
    }
    if not (math.isfinite(dots['jw']) and dots['jw'] > 0.0 and math.isfinite(dots['op']) and dots['op'] > 0.0):
        failures.append('F012 total is not a descent direction for JW and the operator')
    optimizer = arm._new_optimizer(parameters)
    moment = continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], continuation.PARENT_CURSOR, continuation.PARENT_LR)
    if moment != PARENT_MOMENT_SHA:
        failures.append('AdamW moment hash')
    numels = [value.numel() for value in parameters.values()]
    moments = traj._moments(saved, numels)
    param = traj._flat_state(saved, names)
    group = moments['group']
    displacement = prior.hypothetical_adamw_displacement(
        param, doubled, moments['moment'], moments['second'], moments['step'],
        f010.NEW_LR, tuple(group['betas']), group['eps'], group['weight_decay'])
    if not np.isfinite(displacement).all() or not np.isfinite(doubled).all():
        failures.append('nonfinite AdamW displacement or gradient')
    if continuation.moment_sha(optimizer.state_dict()) != moment:
        failures.append('hypothetical AdamW step changed the moments')
    probe = torch.tensor([1.5], device=next(model.parameters()).device, dtype=torch.float64)
    if abs(float(model.activate_g_c(probe)) - 1.5) > 1e-12:
        failures.append('G_c activation is not shifted ELU')
    digest = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        mrks = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        _losses, other_raw, _system, _seconds = b8.other_tasks(model, shadow, bundle, batches[0], dispersion, mrks)
        _measured, measured_raw = run.measure(
            model, shadow, bundle, b8.load_j251_rows()[0], dispersion, mrks, exc_chunk_size=4096)
        gaps = {task: b8.relative_l2(other_raw[task], measured_raw[task]) for task in ('ae17', 'exc', 'op')}
        if any(value > b8.OTHER_TASK_L2_LIMIT for value in gaps.values()):
            failures.append('operator parity')
        if run.existing.digest(model) != digest or {int(state['step']) for state in optimizer.state.values()} != {128}:
            failures.append('preflight mutated the model or the step counter')
    finally:
        bundle.close()
        del model, shadow, optimizer
        torch.cuda.empty_cache()
    tests = subprocess.run(
        [sys.executable, '-m', 'pytest', 'train_models/test_lap_operator_unit.py', 'train_models/test_lap.py', '-q', '--tb=line'],
        cwd=b8.REPO, check=False, capture_output=True, text=True)
    if tests.returncode != 0:
        failures.append('operator or Lap regression tests')
    weighted = {task: coefficients[task] * components[task] for task in b8.TASKS}
    base_weighted = {task: b8.LAMBDAS[task] * components[task] for task in b8.TASKS}
    payload = {
        'passed': not failures, 'failures': failures, 'c': components['c'], 'scales_sha256': AD_SHA,
        'lambdas': coefficients, 'parent_checkpoint_sha256': parent_sha, 'moment_sha256': moment,
        'weighted_norms': norms, 'full_norm': float(np.linalg.norm(doubled)),
        'f010_norm': float(np.linalg.norm(base)),
        'cancellation_ratio': prior.cancellation_ratio(tuple(weighted.values())),
        'dots': dots,
        'cos_jw_op': prior.cosine(components['jw'], components['op']),
        'cos_jw_f010_total': prior.cosine(components['jw'], base),
        'cos_jw_f012_total': prior.cosine(components['jw'], doubled),
        'cos_op_f010_total': prior.cosine(components['op'], base),
        'cos_op_f012_total': prior.cosine(components['op'], doubled),
        'pairs': f010._pairs({**{task: components[task] for task in b8.TASKS}, 'jw': components['jw'], 'f012': doubled}),
        'block_norms': prior.norm_shares(doubled, prior.block_indices(
            names, numels, prior.parameter_blocks(run.model_at(source['initial_state'])[0]))),
        'adamw_dots': {
            'jw': float(np.dot(components['jw'], displacement)),
            'op': float(np.dot(components['op'], displacement)),
            'ae17': float(np.dot(components['ae17'], displacement)),
            'exc': float(np.dot(components['exc'], displacement)),
        },
        'predicted_jw': prior.predicted_change(components['jw'], doubled),
        'predicted_op': prior.predicted_change(components['op'], doubled),
        'f010_weighted_norms': {task: float(np.linalg.norm(base_weighted[task])) for task in b8.TASKS},
        'optimizer_updates': 0, 'regression_tests': tests.returncode,
    }
    b8.write_json(path, payload)
    if failures:
        _stop('NO-GO — F012 preflight failed: ' + ', '.join(failures))
    print('F012_PREFLIGHT', payload['cos_jw_f012_total'], payload['dots']['jw'], payload['dots']['op'], flush=True)
    return payload


def _save(path, model, optimizer, cursor, logs, coefficient, parent_sha):
    payload = {
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': {key: value for key, value in optimizer.state_dict().items()},
        'rng': b8._capture_rng(), 'cursor': cursor, 'logs': list(logs),
        'protocol_id': PROTOCOL_ID, 'manifest_sha256': continuation.MANIFEST_SHA,
        'c': coefficient, 'lambdas': lambdas_f012(), 'lr': f010.NEW_LR, 'scheduler': None,
        'parent_checkpoint_sha256': parent_sha,
    }
    temporary = Path(path).with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path, model, optimizer, coefficient):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('c') != coefficient or saved.get('lambdas') != lambdas_f012():
        _stop('Resume checkpoint is not this F012 arm')
    if saved.get('lr') != f010.NEW_LR or saved.get('scheduler') is not None:
        _stop('Resume checkpoint changed the learning rate')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    b8._restore_rng(saved['rng'])
    return int(saved['cursor']), list(saved['logs'])


def train(source, coefficient, scales):
    run = b8._import_run()
    parent_sha = b8.sha256_file(f010.PARENT_CHECKPOINT)
    target = OUT / 'ordinary_sgd_adamw'
    target.mkdir(parents=True, exist_ok=True)
    latest = target / 'latest.pt'
    model, shadow = run.model_at(source['initial_state'])
    batches = f010._batches()
    coefficients = lambdas_f012()
    if latest.is_file():
        parameters = f010._check_model(model)
        optimizer = arm._new_optimizer(parameters)
        local_update, logs = _load(latest, model, optimizer, coefficient)
        parameters = f010._check_model(model)
    else:
        saved = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
        parameters = f010._check_model(model)
        model.load_state_dict(saved['model'])
        parameters = f010._check_model(model)
        optimizer = arm._new_optimizer(parameters)
        continuation.prepare_loaded_optimizer(optimizer, saved['optimizer'], continuation.PARENT_CURSOR, continuation.PARENT_LR)
        logs = []
        local_update = 0
        _save(latest, model, optimizer, 0, logs, coefficient, parent_sha)
    origin = {name: torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)['model'][name].detach().clone() for name in parameters}
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        for index in range(local_update, f010.NEW_UPDATES):
            batch = batches[index]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            before = run.existing.digest(model)
            before_weights = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            chemistry, chemistry_grad, infos, _elapsed = f010.f010_mean(
                model, shadow, bundle, batch['relchem'], dispersion, scales, coefficient)
            other_losses, other_raw, _system, _other = b8.other_tasks(
                model, shadow, bundle, batch, dispersion, mrks_dispersion)
            raw = {'relchem': chemistry_grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op']}
            losses = {'relchem': chemistry, 'ae17': other_losses['ae17'], 'exc': other_losses['exc'], 'op': other_losses['op']}
            if any(not math.isfinite(value) for value in losses.values()):
                _stop('Nonfinite loss')
            joint = run.adamw_step(model, optimizer, raw, coefficients)
            done = index + 1
            steps = {int(state['step']) for state in optimizer.state.values()}
            if steps != {continuation.PARENT_CURSOR + done} or any(group['lr'] != f010.NEW_LR for group in optimizer.param_groups):
                _stop('AdamW did not take exactly one fixed-rate step')
            row = {
                'local_update': done, 'cumulative_optimizer_step': continuation.PARENT_CURSOR + done,
                'manifest_index': batch['cursor'],
                'batch_identities': ' '.join(info['identity'] for info in infos),
                'chemistry_loss': chemistry, 'ae17_loss': losses['ae17'], 'exc_loss': losses['exc'], 'op_loss': losses['op'],
                'weighted_chemistry': coefficients['relchem'] * chemistry,
                'weighted_ae17': coefficients['ae17'] * losses['ae17'],
                'weighted_exc': coefficients['exc'] * losses['exc'],
                'weighted_op': coefficients['op'] * losses['op'],
                'composite_loss': sum(coefficients[task] * losses[task] for task in b8.TASKS),
                'chemistry_grad_norm': b8._grad_norm(chemistry_grad),
                'ae17_grad_norm': b8._grad_norm(other_raw['ae17']),
                'exc_grad_norm': b8._grad_norm(other_raw['exc']),
                'op_grad_norm': b8._grad_norm(other_raw['op']),
                'lr': f010.NEW_LR, 'gradient_norm': b8._grad_norm(joint),
                'update_norm': arm._vector_delta(parameters, before_weights),
                'displacement_from_t128': arm._vector_delta(parameters, origin),
                'nonfinite': False, 'seconds': time.perf_counter() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'before_sha256': before, 'after_sha256': run.existing.digest(model),
            }
            logs.append(row)
            del raw, joint, chemistry_grad, other_raw, before_weights
            torch.cuda.empty_cache()
            _save(latest, model, optimizer, done, logs, coefficient, parent_sha)
            if done in f010.RAW_CHECKPOINTS:
                _save(target / f'checkpoint_{done}.pt', model, optimizer, done, logs, coefficient, parent_sha)
            b8.write_json(OUT / 'training_status.json', {
                'local_update': done, 'complete': done == f010.NEW_UPDATES, 'logs': logs,
                'cumulative_optimizer_step': continuation.PARENT_CURSOR + done,
            })
            print('F012_UPDATE', done, f'{row["seconds"]:.3f}', f'{row["op_loss"]:.6f}', f'{row["chemistry_loss"]:.6f}', flush=True)
    finally:
        bundle.close()
    if b8.sha256_file(f010.PARENT_CHECKPOINT) != parent_sha:
        _stop('Training changed the B8 checkpoint')
    return logs[-1]['local_update']


def training_metrics(source, scales, coefficient):
    run = b8._import_run()
    targets = [('initial', 0, f010.PARENT_CHECKPOINT)] + [
        (f't{cursor}', cursor, OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt') for cursor in f010.RAW_CHECKPOINTS
    ]
    bundle = run.PublicationDataset(run.DATA)
    coefficients = lambdas_f012()
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
                bare = math.sqrt(1e-20 + residual * residual)
                reactions.append({
                    'identity': spec['identity'], 'database': spec['database'], 'residual': residual,
                    'absolute_residual': abs(residual), 'A_d': scales[spec['database']],
                    'old_penalty': factor * bare, 'f010_penalty': coefficient * bare / scales[spec['database']],
                })
                del reaction, prediction, reference
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
                'relchem': sum(row['old_penalty'] for row in reactions) / f010.COUNT,
                'f010': sum(row['f010_penalty'] for row in reactions) / f010.COUNT,
                'ae17': ae17, 'exc': mrks['exc'], 'op': mrks['op'], 'reactions': reactions,
            }
            payload['composite_f012'] = sum(coefficients[task] * payload[key] for task, key in (
                ('relchem', 'f010'), ('ae17', 'ae17'), ('exc', 'exc'), ('op', 'op')))
            payload['composite_f010'] = sum(b8.LAMBDAS[task] * payload[key] for task, key in (
                ('relchem', 'f010'), ('ae17', 'ae17'), ('exc', 'exc'), ('op', 'op')))
            if name == 'initial' and abs(payload['jw'] - arm.PRIOR_JW['initial']) > arm.SCALAR_TOLERANCE:
                _stop('Initial JW does not match the B8 t128 receipt')
            b8.write_json(done, payload)
            rows_out.append(payload)
            print('F012_METRICS', name, payload['jw'], payload['op'], flush=True)
            del model, shadow
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    return rows_out


def evaluate_clean28(source):
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    try:
        from tools.evaluate_microbatch_endpoint import validation
        for cursor in f010.RAW_CHECKPOINTS:
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
            print('F012_CLEAN28', cursor, result['clean28'], flush=True)
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
            b8.write_json(path, {'name': row['name'], 'ratios': science, 'used_f012_in_ratio': False})
            ratios[row['name']] = science
            del model, shadow
    finally:
        bundle.close()
    return ratios


def gradient_audit(source, coefficient, scales):
    run = b8._import_run()
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    coefficients = lambdas_f012()
    bundle = run.PublicationDataset(run.DATA)
    rows_out = []
    try:
        dispersion = bundle.chemistry_dispersions()
        specs = arm._selected(bundle)
        previous = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)['model']
        for name in GEOMETRY:
            folder = OUT / 'geometry' / name
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / 'geometry.json'
            if done.is_file():
                rows_out.append(b8.read_json(done))
                previous = torch.load(OUT / 'ordinary_sgd_adamw' / f'checkpoint_{name[1:]}.pt', map_location='cpu', weights_only=False)['model']
                continue
            path = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{name[1:]}.pt'
            file_hash = b8.sha256_file(path)
            model, shadow, _run = arm._load_eval_model(source, path)
            rows, jacobian = traj._reaction_jacobian(model, shadow, bundle, dispersion, specs, names, folder)
            residuals = [row['residual'] for row in rows]
            databases = [row['database'] for row in rows]
            physical = traj._physical(model, shadow, bundle, folder, names)
            components = {
                'relchem': coefficient * (f010.f010_raw_coefficients(residuals, scales, databases) @ jacobian),
                'ae17': physical['ae17'], 'exc': physical['exc'], 'op': physical['op'],
            }
            jw = arm.jw_coefficients(residuals, scales, databases) @ jacobian
            f010_total = _assemble(components, b8.LAMBDAS)
            f012_total = _assemble(components, coefficients)
            current = torch.load(path, map_location='cpu', weights_only=False)
            displacement = traj._flat_state(current, names) - traj._flat_state({'model': previous}, names)
            weighted = {task: coefficients[task] * components[task] for task in b8.TASKS}
            payload = {
                'checkpoint': name, 'local_update': int(name[1:]), 'checkpoint_sha256': file_hash,
                'cos_jw_total': prior.cosine(jw, f012_total),
                'cos_jw_operator': prior.cosine(jw, components['op']),
                'cos_operator_total': prior.cosine(components['op'], f012_total),
                'cos_f010_total_f012_total': prior.cosine(f010_total, f012_total),
                'cancellation_ratio': prior.cancellation_ratio(tuple(weighted.values())),
                'weighted_norms': {task: float(np.linalg.norm(weighted[task])) for task in b8.TASKS},
                'cos_jw_ae17': prior.cosine(jw, components['ae17']),
                'cos_jw_exc': prior.cosine(jw, components['exc']),
                'pairs': f010._pairs({**components, 'jw': jw, 'f010_total': f010_total, 'f012_total': f012_total}),
                'dot_jw_displacement': float(np.dot(jw, displacement)),
                'dot_op_displacement': float(np.dot(components['op'], displacement)),
                'displacement_norm': float(np.linalg.norm(displacement)),
                'moment_cosine_jw': prior.cosine(traj._moments(current, [current['model'][name].numel() for name in names])['moment'], jw),
            }
            if abs(float(np.linalg.norm(f012_total - (f010_total + b8.LAMBDAS['op'] * components['op'])))) > 1e-8:
                _stop('Post-training F012 gradient is not F010 plus the operator term')
            b8.write_json(done, payload)
            rows_out.append(payload)
            previous = current['model']
            print('F012_GEOMETRY', name, payload['cos_jw_total'], payload['cos_jw_operator'], flush=True)
            del model, shadow, jacobian
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    return rows_out


def classify(clean_rows, ratios):
    primary = next(row for row in clean_rows if row['local_update'] == 251)
    if not b8.beats_historical(primary['clean28']):
        return 'NO-GO — F012 primary Clean28 gate not passed'
    science = ratios.get('t251')
    if science is None or not b8.scientifically_eligible(science):
        return 'NO-GO — F012 scientific qualification failed'
    return 'GO — F012 primary endpoint and all scientific gates passed'


def render(preflight_payload, status, metrics, clean_rows, ratios, geometry, label):
    logs = [] if status is None else status['logs']
    _csv(b8.REPO / 'lap_f012_operator_priority_training_log.csv', logs, (
        'local_update', 'cumulative_optimizer_step', 'manifest_index', 'batch_identities', 'chemistry_loss', 'ae17_loss', 'exc_loss', 'op_loss',
        'weighted_chemistry', 'weighted_ae17', 'weighted_exc', 'weighted_op', 'composite_loss',
        'chemistry_grad_norm', 'ae17_grad_norm', 'exc_grad_norm', 'op_grad_norm', 'gradient_norm',
        'lr', 'update_norm', 'nonfinite', 'seconds', 'before_sha256', 'after_sha256',
    ))
    f010_metrics = {row['name']: row for row in csv.DictReader((b8.REPO / 'lap_f010_checkpoint_metrics.csv').open(encoding='utf-8'))}
    checkpoint_rows = []
    database_rows = []
    for item in metrics:
        other = f010_metrics['initial' if item['local_update'] == 0 else item['name']]
        checkpoint_rows.append({
            'name': item['name'], 'local_update': item['local_update'], 'jw': item['jw'], 'mae': item['mae'],
            'relchem': item['relchem'], 'f010': item['f010'], 'ae17': item['ae17'], 'exc': item['exc'], 'op': item['op'],
            'composite_f012': item['composite_f012'], 'f010_jw': float(other['jw']), 'f010_op': float(other['op']),
            'f010_ae17': float(other['ae17']), 'f010_exc': float(other['exc']),
            'checkpoint_sha256': item['checkpoint_sha256'],
        })
        grouped = {}
        for row in item['reactions']:
            grouped.setdefault(row['database'], []).append(row)
        for database, group in sorted(grouped.items()):
            database_rows.append({
                'checkpoint': item['name'], 'database': database, 'count': len(group),
                'jw_mean': sum(row['absolute_residual'] / row['A_d'] for row in group) / len(group),
                'mae': sum(row['absolute_residual'] for row in group) / len(group),
            })
    _csv(b8.REPO / 'lap_f012_operator_priority_checkpoint_metrics.csv', checkpoint_rows, list(checkpoint_rows[0]))
    _csv(b8.REPO / 'lap_f012_operator_priority_database_metrics.csv', database_rows, list(database_rows[0]))
    geometry_rows = geometry
    _csv(b8.REPO / 'lap_f012_operator_priority_gradient_geometry.csv', [
        {'checkpoint': row['checkpoint'], 'cos_jw_total': row['cos_jw_total'], 'cos_jw_operator': row['cos_jw_operator'],
         'cos_operator_total': row['cos_operator_total'], 'cancellation_ratio': row['cancellation_ratio'],
         'dot_jw_displacement': row['dot_jw_displacement'], 'dot_op_displacement': row['dot_op_displacement']}
        for row in geometry_rows
    ], ('checkpoint', 'cos_jw_total', 'cos_jw_operator', 'cos_operator_total', 'cancellation_ratio', 'dot_jw_displacement', 'dot_op_displacement'))
    clean_table = []
    for row in clean_rows:
        clean_table.append({
            'local_update': row['local_update'], 'clean28': row['clean28'],
            'f010': F010_CLEAN28[row['local_update']], 'f011': F011_CLEAN28[row['local_update']],
            'minus_f010': row['clean28'] - F010_CLEAN28[row['local_update']],
            'minus_f011': row['clean28'] - F011_CLEAN28[row['local_update']],
            'minus_historical': row['clean28'] - b8.HISTORICAL_BEST_CLEAN28,
            'checkpoint_sha256': row['checkpoint_sha256'],
        })
    _csv(b8.REPO / 'lap_f012_operator_priority_clean28.csv', clean_table, list(clean_table[0]))
    preflight_rows = [{'quantity': key, 'value': preflight_payload[key]} for key in (
        'cos_jw_f012_total', 'cos_jw_f010_total', 'cos_op_f012_total', 'full_norm', 'cancellation_ratio', 'predicted_jw', 'predicted_op',
    )]
    for task, value in preflight_payload['dots'].items():
        preflight_rows.append({'quantity': f'dot_{task}', 'value': value})
    for task, value in preflight_payload['adamw_dots'].items():
        preflight_rows.append({'quantity': f'adamw_dot_{task}', 'value': value})
    b8.write_json(b8.REPO / 'lap_f012_operator_priority_preflight.json', preflight_payload)
    b8.write_json(b8.REPO / 'lap_f012_operator_priority_protocol.json', {
        'protocol_id': PROTOCOL_ID, 'parent_commit': 'd9cc0e84387af375c6f626b45245d04d6d2f2232',
        'c010': preflight_payload['c'], 'lambdas': lambdas_f012(), 'lr': f010.NEW_LR,
        'updates': f010.NEW_UPDATES, 'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8,
        'weight_decay': 0.01, 'foreach': False, 'scheduler': None,
        'parent_checkpoint_sha256': PARENT_CHECKPOINT_SHA, 'moment_sha256': PARENT_MOMENT_SHA,
        'manifest_sha256': continuation.MANIFEST_SHA, 'a_d_sha256': AD_SHA,
    })
    final = next(row for row in checkpoint_rows if row['local_update'] == 251)
    primary = next(row for row in clean_table if row['local_update'] == 251)
    hypothesis = (
        'supported for this matched F010 comparison'
        if final['op'] < final['f010_op'] and primary['clean28'] < primary['f010']
        else 'not supported for this matched F010 comparison'
    )
    lines = [
        '# F012 operator priority',
        '',
        f'**{label}**',
        '',
        f'The operator coefficient is `{LAMBDA_OP_F012}` = 2 × `{b8.LAMBDAS["op"]}`. Chemistry, AE17, Exc, AdamW, and the B8 manifest are the F010 settings. `c010` = `{preflight_payload["c"]}`.',
        '',
        'Clean28 is a development panel, not an untouched test set.',
        '',
        '## Preflight',
        '',
        f'- JW dot F012 total: `{preflight_payload["dots"]["jw"]}`',
        f'- Operator dot F012 total: `{preflight_payload["dots"]["op"]}`',
        f'- AE17 dot: `{preflight_payload["dots"]["ae17"]}`, Exc dot: `{preflight_payload["dots"]["exc"]}`',
        f'- cos(JW, F012): `{preflight_payload["cos_jw_f012_total"]}`',
        f'- Hypothetical AdamW dots: `{preflight_payload["adamw_dots"]}`',
        '',
        '## Checkpoints',
        '',
        '| Update | JW | F010 JW | Operator | F010 operator | AE17 | Exc | Clean28 | vs historical |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row, clean in zip(checkpoint_rows, [None, *clean_table], strict=False):
        if row['local_update'] == 0:
            lines.append(f"| 0 | {row['jw']:.6f} | {row['f010_jw']:.6f} | {row['op']:.6f} | {row['f010_op']:.6f} | {row['ae17']:.6f} | {row['exc']:.6f} |  |  |")
        else:
            lines.append(
                f"| {row['local_update']} | {row['jw']:.6f} | {row['f010_jw']:.6f} | {row['op']:.6f} | {row['f010_op']:.6f} | "
                f"{row['ae17']:.6f} | {row['exc']:.6f} | {clean['clean28']:.12f} | {clean['minus_historical']:+.6f} |"
            )
    lines.extend([
        '',
        f'Primary t251 Clean28 minus historical best `{b8.HISTORICAL_BEST_CLEAN28}`: `{primary["minus_historical"]:+.6f}`.',
        f'The operator-priority hypothesis is {hypothesis}.',
        'This result does not authorize another coefficient.',
        '',
        '## Qualification',
        '',
        'No scientific ratios were run.' if not ratios else 'Ratios use the original losses, not F012.',
        '',
    ])
    for name, science in ratios.items():
        lines.append(f'- `{name}`: ' + ', '.join(f'{task} {science[task]:.6f}' for task in b8.TASKS))
    seconds = sum(row['seconds'] for row in logs)
    lines.extend(['', f'Optimizer-update wall seconds: `{seconds:.3f}`.', ''])
    (b8.REPO / 'lap_f012_operator_priority_training_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    b8.write_json(b8.REPO / 'lap_f012_operator_priority_receipt.json', {
        'label': label, 'hypothesis': hypothesis, 'primary_clean28': primary['clean28'],
        'historical_best': b8.HISTORICAL_BEST_CLEAN28, 'checkpoints': checkpoint_rows,
        'clean28': clean_table, 'ratios': ratios, 'optimizer_updates': 0 if not logs else logs[-1]['local_update'],
    })


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    source = b8.verify_sources()
    payload = preflight(source)
    status_path = OUT / 'training_status.json'
    if not (status_path.is_file() and b8.read_json(status_path).get('complete')):
        train(source, payload['c'], payload['scales'] if 'scales' in payload else _load_components()['scales'])
    status = b8.read_json(status_path)
    if status['local_update'] != f010.NEW_UPDATES:
        _stop('The arm did not finish 251 updates')
    scales = _load_components()['scales']
    metrics = training_metrics(source, scales, payload['c'])
    evaluate_clean28(source)
    clean_rows = [b8.read_json(OUT / f'validation_t{cursor}.json') for cursor in f010.RAW_CHECKPOINTS]
    ratios = qualify(source, clean_rows)
    geometry = gradient_audit(source, payload['c'], scales)
    label = classify(clean_rows, ratios)
    render(payload, status, metrics, clean_rows, ratios, geometry, label)
    print('F012_CLASS', label, flush=True)
    return label


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        execute()
    except SystemExit as error:
        print('F012_STOP', error, flush=True)
        raise
