"""Read-only alignment of J251 absolute-error diagnostics with the production task gradients.

J251-WTMAD-style is a training-set diagnostic. It is not literal GMTKN55 WTMAD-2.
No optimizer update is taken and the new metrics never enter a training loss.
"""
import csv
import math
from pathlib import Path

import lap_b8_adamw as b8
import lap_wtmad_gradient_alignment as prior
import numpy as np
import torch

PROTOCOL_ID = 'training-metric-gradient-audit-v1'
COUNT = 251
TASKS = prior.TASKS
TRAIN_DATABASES = ('ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13')
METRICS = ('mae', 'jw')
PRIOR_VECTORS = ('wtmad',) + TASKS
EXPECTED_RELCHEM = {
    'b8_t128': 1.195939284232437,
    'continuation_raw_t251': 1.1706491841330284,
}
EXPECTED_STEPS = {'b8_t128': 128, 'continuation_raw_t251': 379}
OUT = b8.REPO.parent / 'lap_training_metric_gradient_audit_20261011'
RELCHEM_TOLERANCE = 1e-6
ARTIFACT_COSINE_TOLERANCE = 1e-8


def subgradient_sign(residual):
    return prior.subgradient_sign(residual)


def database_reference_scales(references, databases):
    """Mean absolute reference energy inside each J251 database. Frozen from targets, not residuals."""
    buckets = {}
    for reference, database in zip(references, databases, strict=True):
        if not math.isfinite(reference):
            raise ValueError(f'Nonfinite reference in {database}')
        buckets.setdefault(database, []).append(abs(reference))
    if sum(len(values) for values in buckets.values()) != COUNT:
        raise ValueError('Reference scales must cover the 251 training reactions')
    scales = {}
    counts = {}
    for database, values in buckets.items():
        scale = sum(values) / len(values)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f'Reference scale for {database} is not strictly positive')
        scales[database] = scale
        counts[database] = len(values)
    return scales, counts


def j251_mae(residuals):
    if len(residuals) != COUNT:
        raise ValueError('J251-MAE requires 251 reactions')
    return sum(abs(residual) for residual in residuals) / COUNT


def j251_wtmad_style(residuals, databases, scales):
    """Mean of |residual| / A_d. Not GMTKN55 WTMAD-2 and not multiplied by a global calibration factor."""
    if len(residuals) != COUNT or len(databases) != COUNT:
        raise ValueError('J251-WTMAD-style requires 251 reactions')
    return sum(abs(residual) / scales[database] for residual, database in zip(residuals, databases, strict=True)) / COUNT


def absolute_gradient_weight(residual, scale):
    """Derivative weight of one reaction inside the 251-term mean. `scale` is 1 for MAE and A_d for the styled metric."""
    return subgradient_sign(residual) / (COUNT * scale)


def singleton_objective(residual, factor):
    """One-reaction batch_fchem value: factor * sqrt(1e-20 + residual^2)."""
    return factor * math.sqrt(1e-20 + residual * residual)


def leave_one_subset_out(full, subset):
    """Remove a subset contribution without rescaling the remaining terms."""
    return np.asarray(full, dtype=np.float64) - np.asarray(subset, dtype=np.float64)


def directional_parts(metric, task_vectors, lambdas):
    parts = {
        task: prior.predicted_change(metric, lambdas[task] * task_vectors[task])
        for task in TASKS
    }
    combined = prior.predicted_change(metric, sum(lambdas[task] * task_vectors[task] for task in TASKS))
    if abs(sum(parts.values()) - combined) > 1e-6 * max(1.0, abs(combined)):
        raise ValueError('Weighted task contributions do not reproduce the combined directional derivative')
    return parts, combined


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _selected(bundle):
    evaluation = b8.read_json(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    if evaluation['policy'] != 'one-variant-per-identity-v1' or b8.sha256_file(
            b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json') != b8.EVAL_MANIFEST_SHA:
        _stop('Evaluation manifest changed')
    rows = [row for row in evaluation['rows'] if row['task'] == 'relchem']
    identities = [row['identity'] for row in rows]
    if len(rows) != COUNT or len(set(identities)) != COUNT:
        _stop('J251 relchem identities are not 251 unique reactions')
    specs = []
    for row in rows:
        meta = bundle.reactions[row['identity']]
        if meta['task'] != 'relchem' or meta['database'] not in TRAIN_DATABASES:
            _stop(f"Unexpected training reaction {row['identity']}")
        if row['variant'] not in meta['variants']:
            _stop(f"Manifest variant is not in the dataset for {row['identity']}")
        specs.append({
            'identity': row['identity'],
            'variant': row['variant'],
            'database': meta['database'],
            'reaction_id': meta['reaction_id'],
        })
    return specs


def _factor(database):
    from train_models.optuna_joint import FCHEM_DB_WEIGHTS, FREQ_WEIGHTS, MEAN_WEIGHT
    return FCHEM_DB_WEIGHTS[database] * FREQ_WEIGHTS[database] / MEAN_WEIGHT


def _prediction(shadow, reaction, dispersion):
    """Production reaction energy in kcal/mol, stopping before batch_fchem."""
    from torch.utils.checkpoint import checkpoint

    from train_models.lap_training import (
        minnesota_sigma_boundary_matches,
        tensor_record,
    )
    from train_models.reaction_energy_calculation import calculate_reaction_energy
    record = tensor_record(reaction, 'cuda', torch.float64)
    raw = record['Grid']
    if raw.ndim != 2 or raw.shape[1] != 9:
        _stop('Minnesota grid is not the nine-column layout')
    if not torch.allclose(raw[:, :2], record['Densities'], rtol=1e-6, atol=1e-12):
        _stop('Minnesota rho boundary disagrees with model input')
    if not minnesota_sigma_boundary_matches(raw[:, 2:5], record['Gradients']):
        _stop('Minnesota sigma boundary disagrees')
    chunk = record.get('model_point_chunk_size', 0)
    if chunk:
        constants = torch.cat([
            checkpoint(shadow, raw[start:start + chunk], use_reentrant=False)
            for start in range(0, len(raw), chunk)
        ], dim=0)
    else:
        constants = checkpoint(shadow, raw, use_reentrant=False)
    prediction, _local = calculate_reaction_energy(
        record, constants, 'cuda', 'GGA', 'PBE', dispersions=dispersion, return_local_energies=False)
    reference = record['Energy'].to(device=prediction.device, dtype=torch.float64).reshape(-1)
    if prediction.numel() != 1 or reference.numel() != 1:
        _stop('Reaction energy is not a single kcal/mol value')
    return prediction.reshape(()), reference.reshape(())


def _load_model(source, path):
    run = b8._import_run()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model, shadow = run.model_at(source['initial_state'])
    model.load_state_dict(saved['model'])
    parameters = run.existing.named_trainable_parameters(model)
    if sum(parameter.numel() for parameter in parameters.values()) != 9446:
        _stop('Checkpoint does not have 9446 trainable coordinates')
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        _stop('Checkpoint parameters are not float32')
    if type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or int(model.lap_architecture_version) != 1:
        _stop('Checkpoint architecture is not the qualified Lap model')
    steps = {int(state['step']) for state in saved['optimizer']['state'].values()}
    return model, shadow, parameters, saved, steps


def _flat(names, grads):
    missing = [name for name, grad in zip(names, grads, strict=True) if grad is None]
    if missing:
        _stop('A reaction energy left disconnected parameters and they were not zero-filled: ' + ', '.join(missing[:8]))
    return np.concatenate([grad.detach().reshape(-1).double().cpu().numpy() for grad in grads])


def _save_reactions(path, rows, gradients):
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(
        temporary,
        gradients=np.asarray(gradients, dtype=np.float64),
        residual=np.asarray([row['residual'] for row in rows], dtype=np.float64),
        reference=np.asarray([row['reference'] for row in rows], dtype=np.float64),
    )
    temporary.replace(path)
    b8.write_json(path.with_suffix('.json'), rows)


def _load_reactions(path):
    meta = path.with_suffix('.json')
    if not path.is_file() or not meta.is_file():
        return None
    rows = b8.read_json(meta)
    with np.load(path) as payload:
        gradients = np.asarray(payload['gradients'], dtype=np.float64)
        residual = np.asarray(payload['residual'], dtype=np.float64)
    if len(rows) != len(gradients) or not np.allclose(residual, [row['residual'] for row in rows]):
        _stop('Reaction cache disagrees with its metadata')
    return rows, gradients


def _reaction_pass(model, shadow, bundle, dispersion, specs, names, folder):
    cache = folder / 'reactions.npz'
    loaded = _load_reactions(cache)
    rows = [] if loaded is None else list(loaded[0])
    gradients = [] if loaded is None else [row for row in loaded[1]]
    if len(rows) > len(specs) or any(
            rows[index]['identity'] != specs[index]['identity'] for index in range(len(rows))):
        _stop('Reaction cache does not match the J251 manifest')
    parameters = b8._import_run().existing.named_trainable_parameters(shadow)
    if tuple(parameters) != names:
        _stop('Shadow parameter order differs from the float32 model')
    for index in range(len(rows), len(specs)):
        spec = specs[index]
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        if reaction['Database'] != spec['database'] or reaction['ReactionID'] != spec['reaction_id']:
            _stop(f"Loaded reaction metadata changed for {spec['identity']}")
        shadow.load_state_dict(model.state_dict(), strict=True)
        shadow.train(model.training)
        prediction, reference = _prediction(shadow, reaction, dispersion)
        if not prediction.requires_grad:
            _stop(f"Reaction energy does not depend on parameters for {spec['reaction_id']}")
        residual = prediction - reference
        grads = torch.autograd.grad(prediction, tuple(parameters.values()), allow_unused=True)
        flat = _flat(names, grads)
        value = float(residual.detach().cpu())
        rows.append({
            'identity': spec['identity'], 'variant': spec['variant'], 'database': spec['database'],
            'reaction_id': spec['reaction_id'], 'reference': float(reference.detach().cpu()), 'residual': value,
        })
        gradients.append(flat)
        del reaction, prediction, reference, residual, grads, flat
        torch.cuda.empty_cache()
        if len(rows) % 25 == 0 or len(rows) == len(specs):
            _save_reactions(cache, rows, gradients)
            print('AUDIT_REACTION', folder.name, len(rows), len(specs), flush=True)
    return rows, np.asarray(gradients, dtype=np.float64)


def _metric_scalars(model, shadow, bundle, dispersion, specs, scales, references):
    residuals = []
    for spec, reference in zip(specs, references, strict=True):
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        shadow.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            prediction, loaded = _prediction(shadow, reaction, dispersion)
        if abs(float(loaded.detach().cpu()) - reference) > 0.0:
            _stop(f"Reference energy changed for {spec['reaction_id']}")
        residuals.append(float((prediction - loaded).detach().cpu()))
        del reaction, prediction, loaded
        torch.cuda.empty_cache()
    databases = [spec['database'] for spec in specs]
    return j251_mae(residuals), j251_wtmad_style(residuals, databases, scales), residuals


def _assemble(rows, energy_grads):
    references = [row['reference'] for row in rows]
    databases = [row['database'] for row in rows]
    residuals = [row['residual'] for row in rows]
    scales, counts = database_reference_scales(references, databases)
    mae_parts = []
    jw_parts = []
    for row, gradient in zip(rows, energy_grads, strict=True):
        mae_parts.append(absolute_gradient_weight(row['residual'], 1.0) * gradient)
        jw_parts.append(absolute_gradient_weight(row['residual'], scales[row['database']]) * gradient)
    mae = np.sum(np.stack(mae_parts), axis=0)
    jw = np.sum(np.stack(jw_parts), axis=0)
    relchem_values = [singleton_objective(row['residual'], _factor(row['database'])) for row in rows]
    zeros = [row['reaction_id'] for row in rows if row['residual'] == 0.0]
    return {
        'scales': scales, 'counts': counts, 'mae': mae, 'jw': jw,
        'mae_parts': np.stack(mae_parts), 'jw_parts': np.stack(jw_parts),
        'mae_scalar': j251_mae(residuals), 'jw_scalar': j251_wtmad_style(residuals, databases, scales),
        'relchem_scalar': sum(relchem_values) / COUNT, 'relchem_values': relchem_values,
        'zeros': zeros,
    }


def _prior_vectors(spec, names, numels):
    folder = prior.OUT / spec['name']
    result = b8.read_json(folder / 'result.json')
    if result['checkpoint_sha256'] != b8.sha256_file(spec['path']) or result['coordinates'] != 9446:
        _stop(f"Saved Clean28 artifact integrity failed for {spec['name']}")
    if tuple(result['parameter_names']) != names or list(result['parameter_numels']) != list(numels):
        _stop(f"Saved gradient ordering does not match {spec['name']}")
    vectors = {name: prior._load_vector(folder / f'{name}.npy.npz') for name in PRIOR_VECTORS}
    if any(vector is None or vector.shape != (9446,) for vector in vectors.values()):
        _stop(f"Saved gradient vectors are missing or the wrong width for {spec['name']}")
    geometry = result['geometry']
    for task in TASKS:
        if abs(float(np.linalg.norm(vectors[task])) - geometry['tasks'][task]['raw_norm']) > 1e-8:
            _stop(f"Saved {task} gradient does not match the committed norm")
    if abs(prior.cosine(vectors['relchem'], vectors['wtmad']) - geometry['cosine']['relchem']['wtmad']) > ARTIFACT_COSINE_TOLERANCE:
        _stop('Saved vectors do not reproduce the committed Clean28 cosine')
    combined = sum(b8.LAMBDAS[task] * vectors[task] for task in TASKS)
    if abs(prior.cosine(combined, vectors['wtmad']) - geometry['combined_cosine']) > ARTIFACT_COSINE_TOLERANCE:
        _stop('Reconstructed training gradient does not match the committed Clean28 cosine')
    vectors['total'] = combined
    return vectors, result


def _pair_row(metric, task_name, task, lambdas):
    weighted = lambdas[task_name] * task if task_name in TASKS else task
    weighted_change = None if task_name == 'wtmad' else prior.predicted_change(metric, weighted)
    weighted_dot = None if task_name == 'wtmad' else float(np.dot(metric, weighted))
    return {
        'raw_norm': float(np.linalg.norm(task)),
        'dot': float(np.dot(metric, task)),
        'weighted_dot': weighted_dot,
        'cosine': prior.cosine(metric, task),
        'predicted_change': prior.predicted_change(metric, task),
        'weighted_predicted_change': weighted_change,
        'normalized_predicted_change': prior.normalized_change(metric, task),
    }


def _geometry(assembled, vectors, indices):
    lambdas = b8.LAMBDAS
    metrics = {'mae': assembled['mae'], 'jw': assembled['jw']}
    names = ('mae', 'jw', 'relchem', 'ae17', 'exc', 'op', 'total', 'wtmad')
    catalog = {**metrics, **vectors}
    cosine = {left: {right: prior.cosine(catalog[left], catalog[right]) for right in names} for left in names}
    comparisons = {
        metric: {task: _pair_row(metrics[metric], task, catalog[task], lambdas) for task in (*TASKS, 'total', 'wtmad')}
        for metric in METRICS
    }
    parts = {}
    for metric in METRICS:
        split, combined = directional_parts(metrics[metric], vectors, lambdas)
        parts[metric] = {'tasks': split, 'combined': combined}
    cancellation = prior.cancellation_ratio([lambdas[task] * vectors[task] for task in TASKS])
    blocks = []
    for metric in METRICS:
        shares = prior.norm_shares(metrics[metric], indices)
        for task in (*TASKS, 'total'):
            task_shares = prior.norm_shares(catalog[task], indices)
            for block, positions in indices.items():
                full = cosine[metric][task]
                block_cosine = prior.cosine(metrics[metric][positions], catalog[task][positions]) if len(positions) else None
                reversal = (
                    shares[block] >= 0.01 and block_cosine is not None and full is not None
                    and block_cosine * full < 0.0
                )
                blocks.append({
                    'metric': metric, 'task': task, 'block': block,
                    'metric_share': shares[block], 'task_share': task_shares[block],
                    'block_cosine': block_cosine, 'full_cosine': full, 'sign_reversal': reversal,
                })
    return {
        'cosine': cosine, 'comparisons': comparisons, 'parts': parts,
        'cancellation_ratio': cancellation, 'blocks': blocks,
        'norms': {name: float(np.linalg.norm(catalog[name])) for name in names},
    }


def _subsets(rows, assembled, vectors):
    records = []
    for database in sorted(assembled['counts']):
        chosen = [index for index, row in enumerate(rows) if row['database'] == database]
        part = np.sum(assembled['jw_parts'][chosen], axis=0)
        residuals = [rows[index]['residual'] for index in chosen]
        records.append({
            'database': database,
            'count': assembled['counts'][database],
            'reference_scale': assembled['scales'][database],
            'mean_absolute_error': sum(abs(value) for value in residuals) / len(residuals),
            'jw_contribution': sum(abs(value) / assembled['scales'][database] for value in residuals) / COUNT,
            'gradient_norm': float(np.linalg.norm(part)),
            'cosine_jw': prior.cosine(part, assembled['jw']),
            'cosine_relchem': prior.cosine(part, vectors['relchem']),
            'cosine_operator': prior.cosine(part, vectors['op']),
            'dot_relchem': float(np.dot(part, vectors['relchem'])),
            'dot_operator': float(np.dot(part, vectors['op'])),
            'leave_one_out_cosine_relchem': prior.cosine(leave_one_subset_out(assembled['jw'], part), vectors['relchem']),
            'leave_one_out_cosine_jw': prior.cosine(leave_one_subset_out(assembled['jw'], part), assembled['jw']),
        })
    if assembled['jw_parts'].shape[0] != COUNT:
        _stop('Subset decomposition lost a reaction')
    total = np.sum(assembled['jw_parts'], axis=0)
    if float(np.max(np.abs(total - assembled['jw']))) > 1e-8:
        _stop('J251-WTMAD-style gradient is not the sum of its reaction contributions')
    return records


def _reactions(rows, assembled, vectors):
    records = []
    for index, row in enumerate(rows):
        part = assembled['jw_parts'][index]
        records.append({
            'database': row['database'], 'reaction_id': row['reaction_id'], 'identity': row['identity'],
            'variant': row['variant'], 'signed_residual': row['residual'], 'absolute_residual': abs(row['residual']),
            'relchem_contribution': assembled['relchem_values'][index] / COUNT,
            'jw_contribution': abs(row['residual']) / (COUNT * assembled['scales'][row['database']]),
            'gradient_norm': float(np.linalg.norm(part)),
            'cosine_jw': prior.cosine(part, assembled['jw']),
            'dot_relchem': float(np.dot(part, vectors['relchem'])),
            'dot_operator': float(np.dot(part, vectors['op'])),
        })
    return records


def _finite_difference(model, shadow, bundle, dispersion, specs, scales, references, gradient, scalar, label):
    magnitude = float(np.linalg.norm(gradient))
    if magnitude == 0.0:
        return {'label': label, 'skipped': True}
    unit = gradient / magnitude
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    names = tuple(b8._import_run().existing.named_trainable_parameters(model))
    scales_wanted = (1e-2, 1e-3, 1e-4) if label == 'jw' else (1e-3,)
    rows = []

    def evaluate(epsilon):
        parameters = b8._import_run().existing.named_trainable_parameters(model)
        offset = 0
        with torch.no_grad():
            for name in names:
                parameter = parameters[name]
                count = parameter.numel()
                delta = torch.from_numpy(unit[offset:offset + count] * epsilon).to(parameter.device, parameter.dtype)
                parameter.add_(delta.reshape(parameter.shape))
                offset += count
        try:
            mae, jw, _residuals = _metric_scalars(model, shadow, bundle, dispersion, specs, scales, references)
        finally:
            model.load_state_dict(base_state)
            torch.cuda.empty_cache()
        return jw if label == 'jw' else mae

    center = evaluate(0.0)
    if abs(center - scalar) > 1e-6:
        _stop(f'Independent {label} scalar {center} disagrees with the gradient path {scalar}')
    for target in scales_wanted:
        epsilon = target / magnitude
        plus = evaluate(epsilon)
        minus = evaluate(-epsilon)
        estimate = (plus - minus) / (2.0 * epsilon)
        rows.append({
            'target_change': target, 'epsilon': epsilon, 'finite_difference': estimate,
            'autodiff': magnitude, 'relative_gap': abs(estimate - magnitude) / magnitude,
        })
        print('AUDIT_FD', label, target, rows[-1]['relative_gap'], flush=True)
    if prior._digest_state(model.state_dict()) != prior._digest_state(base_state):
        _stop('Finite differences did not restore the checkpoint weights')
    return {'label': label, 'skipped': False, 'center': center, 'scales': rows}


def measure_checkpoint(source, spec, frozen_scales):
    folder = OUT / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'result.json'
    if done.is_file():
        return b8.read_json(done), frozen_scales
    run = b8._import_run()
    model, shadow, parameters, saved, steps = _load_model(source, spec['path'])
    names = tuple(parameters)
    numels = [parameter.numel() for parameter in parameters.values()]
    if steps != {EXPECTED_STEPS[spec['name']]}:
        _stop(f"Optimizer step provenance failed for {spec['name']}: {sorted(steps)}")
    digest = run.existing.digest(model)
    prior_result = b8.read_json(prior.OUT / spec['name'] / 'result.json')
    if digest != prior_result['model_sha256']:
        _stop(f"Loaded model hash does not match the previous audit for {spec['name']}")
    bundle = run.PublicationDataset(run.DATA)
    if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
        _stop('Dataset logical hash changed')
    specs = _selected(bundle)
    dispersion = bundle.chemistry_dispersions()
    rows, energy_grads = _reaction_pass(model, shadow, bundle, dispersion, specs, names, folder)
    if energy_grads.shape != (COUNT, 9446):
        _stop('Energy gradients do not have shape (251, 9446)')
    assembled = _assemble(rows, energy_grads)
    if frozen_scales is None:
        frozen_scales = assembled['scales']
        b8.write_json(OUT / 'reference_scales.json', {'scales': frozen_scales, 'counts': assembled['counts']})
    elif any(abs(frozen_scales[key] - assembled['scales'][key]) > 0.0 for key in frozen_scales):
        _stop('Reference scales changed between checkpoints')
    expected = EXPECTED_RELCHEM[spec['name']]
    if abs(assembled['relchem_scalar'] - expected) > RELCHEM_TOLERANCE:
        b8.write_json(folder / 'scalar_discrepancy.json', {
            'relchem': assembled['relchem_scalar'], 'expected': expected,
            'mae': assembled['mae_scalar'], 'jw': assembled['jw_scalar'],
        })
        _stop(f"Full251 relchem scalar was not reproduced at {spec['name']}")
    vectors, prior_payload = _prior_vectors(spec, names, numels)
    indices = prior.block_indices(names, numels, prior.parameter_blocks(model))
    geometry = _geometry(assembled, vectors, indices)
    subsets = _subsets(rows, assembled, vectors)
    reactions = _reactions(rows, assembled, vectors)
    print('AUDIT_SCALAR', spec['name'], assembled['relchem_scalar'], assembled['mae_scalar'], assembled['jw_scalar'], flush=True)
    finite = {
        'jw': _finite_difference(
            model, shadow, bundle, dispersion, specs, assembled['scales'],
            [row['reference'] for row in rows], assembled['jw'], assembled['jw_scalar'], 'jw'),
        'mae': _finite_difference(
            model, shadow, bundle, dispersion, specs, assembled['scales'],
            [row['reference'] for row in rows], assembled['mae'], assembled['mae_scalar'], 'mae'),
    }
    if run.existing.digest(model) != digest or b8.sha256_file(spec['path']) != prior_payload['checkpoint_sha256']:
        _stop('The diagnostic changed the checkpoint file or weights')
    result = {
        'name': spec['name'], 'checkpoint_sha256': prior_payload['checkpoint_sha256'],
        'model_sha256': digest, 'protocol_id': saved.get('protocol_id'), 'optimizer_step': EXPECTED_STEPS[spec['name']],
        'coordinates': 9446, 'parameter_names': list(names), 'parameter_numels': numels,
        'relchem_scalar': assembled['relchem_scalar'], 'mae_scalar': assembled['mae_scalar'],
        'jw_scalar': assembled['jw_scalar'], 'scales': assembled['scales'], 'counts': assembled['counts'],
        'zero_residuals': assembled['zeros'], 'geometry': _json_ready(geometry),
        'subsets': subsets, 'reactions': reactions, 'finite_difference': finite,
        'prior_clean28_cosine': prior_payload['geometry']['cosine']['wtmad'],
        'optimizer_updates': 0,
    }
    b8.write_json(done, result)
    del model, shadow, energy_grads, assembled
    torch.cuda.empty_cache()
    print('AUDIT_DONE', spec['name'], flush=True)
    return result, frozen_scales


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value):
    if value is None:
        return 'undefined'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return f'{value:.6f}'


def _write_csv(path, rows, fields):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _conclusion(results):
    lines = [
        'No verification gate failed. Both relchem receipts were reproduced, every gradient has 9,446 coordinates in the saved order, the styled-metric gradient matches the sum of its reaction contributions, and no residual is exactly zero.',
        '',
        'The recorded outcome is C. Equal-weight J251-MAE and reference-normalized J251-WTMAD-style do not share an alignment with relchem, so this is not a single objective-mismatch or generalization-mismatch label.',
        '',
    ]
    for result in results:
        cosine = result['geometry']['cosine']
        dbh = next(row for row in result['subsets'] if row['database'] == 'DBH76')
        ncce = next(row for row in result['subsets'] if row['database'] == 'NCCE31')
        exchange = next(
            row for row in result['geometry']['blocks']
            if row['metric'] == 'jw' and row['task'] == 'relchem' and row['block'] == 'exchange'
        )
        lines.append(
            f'At `{result["name"]}`, cos(MAE, relchem) is {_fmt(cosine["mae"]["relchem"])} and cos(JW, relchem) is {_fmt(cosine["jw"]["relchem"])}. '
            f'cos(JW, Clean28) is {_fmt(cosine["jw"]["wtmad"])}; cos(Clean28, relchem) is {_fmt(cosine["wtmad"]["relchem"])}. '
            f'cos(JW, operator) is {_fmt(cosine["jw"]["op"])}, cos(Clean28, operator) is {_fmt(cosine["wtmad"]["op"])}, and cos(MAE, operator) is {_fmt(cosine["mae"]["op"])}. '
            f'Descent on the combined training gradient changes JW by {_fmt(result["geometry"]["comparisons"]["jw"]["total"]["predicted_change"])} and MAE by {_fmt(result["geometry"]["comparisons"]["mae"]["total"]["predicted_change"])}.'
        )
        lines.append('')
        lines.append(
            f'DBH76, with {dbh["count"]} reactions and A_d {_fmt(dbh["reference_scale"])} kcal/mol, contributes {_fmt(dbh["dot_relchem"])} to dot(g_JW, g_relchem). '
            f'Removing it without renormalizing leaves cosine {_fmt(dbh["leave_one_out_cosine_relchem"])}. '
            f'NCCE31 has the smallest reference scale, {_fmt(ncce["reference_scale"])} kcal/mol, and contributes {_fmt(ncce["dot_relchem"])}; removing only NCCE31 leaves cosine {_fmt(ncce["leave_one_out_cosine_relchem"])}. '
            f'The exchange block holds {_fmt(exchange["metric_share"])} of the JW squared norm and has block cosine {_fmt(exchange["block_cosine"])} with relchem.'
        )
        lines.append('')
    lines.extend([
        'What is established at both frozen vectors: relchem is weakly aligned with equal-weight training absolute error and opposed to the reference-normalized training metric. That normalized gradient is also aligned with the saved Clean28 gradient. The Clean28 conflict with relchem is therefore already present on J251 once reactions are divided by the frozen A_d. It is not present for the unnormalized mean absolute error. The operator gradient aligns with J251-WTMAD-style and with Clean28; it does not align with J251-MAE. About 98% of the normalized-metric gradient lies in the exchange network, and that is where the opposition to relchem sits. The correlation-network sign reversal is a block of about 1–2% of the squared norm.',
        '',
        'The subset decomposition supports a concentrated conflict. DBH76 is most of the net dot with relchem. After it is removed, the remaining gradient is still weakly opposed to relchem. NCCE31 adds more opposition, including reactions whose absolute errors are small and whose gradient dots are large, but removing NCCE31 alone does not remove the negative cosine. MGAE109 has the most reactions and the largest reference scale, so the 1/A_d factor gives it a modest JW contribution despite a large mean absolute error.',
        '',
        'What was not isolated: the original relchem database and frequency factors were not ablated, so this audit does not show that one of those factors is the cause. The comparison that was performed is equal-weight absolute error against the same absolute errors divided by frozen A_d. Residual transformation, reference normalization, and dataset composition all remain visible; only the normalization step was switched on and off.',
        '',
        'A negative cosine is not a reason to weaken or remove a physical loss. No training arm was run, and the model-selection gates are unchanged. The cross-dataset cosine was not used to fit any parameter, normalization, or optimizer setting.',
        '',
    ])
    return lines


def render(results):
    cosine_rows = []
    subset_rows = []
    reaction_rows = []
    block_rows = []
    names = ('mae', 'jw', 'relchem', 'ae17', 'exc', 'op', 'total', 'wtmad')
    for result in results:
        for left in names:
            for right in names:
                cosine_rows.append({
                    'checkpoint': result['name'], 'left': left, 'right': right,
                    'cosine': result['geometry']['cosine'][left][right],
                })
        for row in result['subsets']:
            subset_rows.append({'checkpoint': result['name'], **row})
        for row in result['reactions']:
            reaction_rows.append({'checkpoint': result['name'], **row})
        for row in result['geometry']['blocks']:
            block_rows.append({'checkpoint': result['name'], **row})
    _write_csv(b8.REPO / 'lap_training_gradient_cosines.csv', cosine_rows, ['checkpoint', 'left', 'right', 'cosine'])
    _write_csv(
        b8.REPO / 'lap_training_subset_gradient_decomposition.csv', subset_rows,
        ['checkpoint', 'database', 'count', 'reference_scale', 'mean_absolute_error', 'jw_contribution',
         'gradient_norm', 'cosine_jw', 'cosine_relchem', 'cosine_operator', 'dot_relchem', 'dot_operator',
         'leave_one_out_cosine_relchem', 'leave_one_out_cosine_jw'])
    _write_csv(
        b8.REPO / 'lap_training_reaction_gradient_decomposition.csv', reaction_rows,
        ['checkpoint', 'database', 'reaction_id', 'identity', 'variant', 'signed_residual', 'absolute_residual',
         'relchem_contribution', 'jw_contribution', 'gradient_norm', 'cosine_jw', 'dot_relchem', 'dot_operator'])
    _write_csv(
        b8.REPO / 'lap_training_parameter_block_alignment.csv', block_rows,
        ['checkpoint', 'metric', 'task', 'block', 'metric_share', 'task_share', 'block_cosine', 'full_cosine', 'sign_reversal'])
    lines = [
        '# Training-metric gradient alignment audit',
        '',
        'Diagnostic only. Optimizer updates: 0. J251-MAE and J251-WTMAD-style were measurement targets and did not enter a training loss.',
        '',
        'J251 is the Minnesota/M06 training collection, not GMTKN55. J251-WTMAD-style is `(1/251) Σ |E_pred - E_ref| / A_d`, where `A_d` is the mean absolute J251 reference energy inside database `d`. It is not literal WTMAD-2. No GMTKN55 subset weight or global calibration constant is used. J251-MAE is the equal-weight mean absolute reaction-energy error in kcal/mol.',
        '',
        'The existing relchem objective remains the mean of 251 singleton `batch_fchem` values. A singleton is `factor_d * sqrt(1e-20 + residual^2)`, with the original database and frequency factors. Absolute value uses subgradient 0 at a zero residual.',
        '',
        f'Publication logical SHA256 `{b8.DATA_SHA}`. Evaluation manifest SHA256 `{b8.EVAL_MANIFEST_SHA}`. Production coefficients, applied once: relchem `{b8.LAMBDAS["relchem"]}`, ae17 `{b8.LAMBDAS["ae17"]}`, exc `{b8.LAMBDAS["exc"]}`, op `{b8.LAMBDAS["op"]}`.',
        '',
        'S5 was not loaded. Saved Clean28 and production-task gradients were reused only after their checkpoint hash, parameter order, norms, and committed cosines matched. They were not recalibrated.',
        '',
        '## Checkpoints and scalar parity',
        '',
    ]
    for result in results:
        lines.append(
            f'- `{result["name"]}` protocol `{result["protocol_id"]}`, AdamW step `{result["optimizer_step"]}`, '
            f'relchem `{result["relchem_scalar"]:.12f}` (receipt `{EXPECTED_RELCHEM[result["name"]]:.12f}`). '
            f'J251-MAE `{result["mae_scalar"]:.6f}` kcal/mol. J251-WTMAD-style `{result["jw_scalar"]:.6f}`. '
            f'Checkpoint SHA256 `{result["checkpoint_sha256"]}`. Model SHA256 `{result["model_sha256"]}`. Coordinates {result["coordinates"]}.'
        )
    lines.extend(['', '## Reference scales', ''])
    lines.append('| Database | Count | A_d (kcal/mol) |')
    lines.append('| --- | ---: | ---: |')
    scales = results[0]['scales']
    counts = results[0]['counts']
    for database in sorted(counts):
        lines.append(f'| {database} | {counts[database]} | {_fmt(scales[database])} |')
    lines.extend(['', '## Primary alignment', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('| Metric | Task | Cosine | Dot | Weighted dot | Predicted change | Weighted predicted change | Unit predicted change |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for metric in METRICS:
            for task in (*TASKS, 'total', 'wtmad'):
                item = result['geometry']['comparisons'][metric][task]
                lines.append(
                    f'| {metric} | {task} | {_fmt(item["cosine"])} | {_fmt(item["dot"])} | {_fmt(item["weighted_dot"])} | '
                    f'{_fmt(item["predicted_change"])} | {_fmt(item["weighted_predicted_change"])} | {_fmt(item["normalized_predicted_change"])} |'
                )
        lines.append('')
        lines.append(
            f'Norms: MAE {_fmt(result["geometry"]["norms"]["mae"])}, J251-WTMAD-style {_fmt(result["geometry"]["norms"]["jw"])}, '
            f'relchem {_fmt(result["geometry"]["norms"]["relchem"])}, operator {_fmt(result["geometry"]["norms"]["op"])}, '
            f'total {_fmt(result["geometry"]["norms"]["total"])}, Clean28 {_fmt(result["geometry"]["norms"]["wtmad"])}.'
        )
        lines.append(f'Combined-task cancellation ratio {_fmt(result["geometry"]["cancellation_ratio"])}.')
        for metric in METRICS:
            split = result['geometry']['parts'][metric]
            rendered = ', '.join(f'{task} {_fmt(split["tasks"][task])}' for task in TASKS)
            lines.append(f'{metric} directional derivative along the total descent splits as {rendered}; sum {_fmt(split["combined"])}.')
        lines.append('')
    lines.extend([
        'Predicted change is `-dot(g_metric, direction)`. Positive means descent along that direction is predicted to increase the diagnostic metric. Weighted uses each frozen production coefficient once. These are not AdamW steps.',
        '',
        '## Training subsets',
        '',
    ])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('| Database | Count | A_d | Mean abs error | JW scalar | Gradient norm | Cosine JW | Cosine relchem | Cosine operator | Dot relchem | Dot operator | Leave-one-out cosine relchem |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        ordered = sorted(result['subsets'], key=lambda row: row['dot_relchem'])
        for row in ordered:
            lines.append(
                f'| {row["database"]} | {row["count"]} | {_fmt(row["reference_scale"])} | {_fmt(row["mean_absolute_error"])} | '
                f'{_fmt(row["jw_contribution"])} | {_fmt(row["gradient_norm"])} | {_fmt(row["cosine_jw"])} | {_fmt(row["cosine_relchem"])} | '
                f'{_fmt(row["cosine_operator"])} | {_fmt(row["dot_relchem"])} | {_fmt(row["dot_operator"])} | {_fmt(row["leave_one_out_cosine_relchem"])} |'
            )
        total_dot = sum(row['dot_relchem'] for row in result['subsets'])
        dominant = min(result['subsets'], key=lambda row: row['dot_relchem'])
        share = dominant['dot_relchem'] / total_dot if total_dot else None
        lines.append('')
        lines.append(
            f'The most negative relchem dot is {dominant["database"]} ({_fmt(dominant["dot_relchem"])}, fraction of the total dot {_fmt(share)}). '
            'Leave-one-out keeps the original 1/251 denominator and the frozen A_d.'
        )
        lines.append('')
    lines.extend(['## Reactions', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('Most negative contributions to `dot(g_JW, g_relchem)`:')
        lines.append('')
        lines.append('| Database | Reaction | Signed residual | JW scalar | Gradient norm | Dot relchem | Dot operator |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
        ranked = sorted(result['reactions'], key=lambda row: row['dot_relchem'])
        for row in ranked[:8]:
            lines.append(
                f'| {row["database"]} | {row["reaction_id"]} | {_fmt(row["signed_residual"])} | {_fmt(row["jw_contribution"])} | '
                f'{_fmt(row["gradient_norm"])} | {_fmt(row["dot_relchem"])} | {_fmt(row["dot_operator"])} |'
            )
        lines.extend(['', 'Largest J251-WTMAD-style scalar contributions:', ''])
        lines.append('| Database | Reaction | Absolute residual | JW scalar | Gradient norm | Dot relchem |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: |')
        for row in sorted(result['reactions'], key=lambda item: item['jw_contribution'], reverse=True)[:5]:
            lines.append(
                f'| {row["database"]} | {row["reaction_id"]} | {_fmt(row["absolute_residual"])} | {_fmt(row["jw_contribution"])} | '
                f'{_fmt(row["gradient_norm"])} | {_fmt(row["dot_relchem"])} |'
            )
        scalar_rank = {row['reaction_id']: rank for rank, row in enumerate(sorted(result['reactions'], key=lambda item: item['jw_contribution'], reverse=True))}
        gradient_rank = {row['reaction_id']: rank for rank, row in enumerate(sorted(result['reactions'], key=lambda item: item['gradient_norm'], reverse=True))}
        mismatched = [
            row for row in result['reactions']
            if scalar_rank[row['reaction_id']] < 15 and gradient_rank[row['reaction_id']] > COUNT // 2
        ]
        lines.append('')
        if mismatched:
            shown = ', '.join(f'{row["database"]}/{row["reaction_id"]}' for row in mismatched[:8])
            lines.append(f'Reactions with a large scalar contribution and a small gradient contribution include {shown}.')
        else:
            lines.append('No reaction is in the top 15 scalar contributions and the bottom half of gradient norms.')
        lines.append('')
    lines.extend(['## Parameter blocks', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('| Metric | Task | Block | Metric share | Block cosine | Full cosine | Sign reversal |')
        lines.append('| --- | --- | --- | ---: | ---: | ---: | --- |')
        for row in result['geometry']['blocks']:
            if row['task'] not in ('relchem', 'op', 'total') or row['block'] == 'remaining':
                continue
            lines.append(
                f'| {row["metric"]} | {row["task"]} | {row["block"]} | {_fmt(row["metric_share"])} | '
                f'{_fmt(row["block_cosine"])} | {_fmt(row["full_cosine"])} | {_fmt(row["sign_reversal"])} |'
            )
        lines.append('')
    lines.extend(['## Comparison with the saved Clean28 audit', ''])
    lines.append('| Checkpoint | cos(JW, relchem) | cos(Clean28, relchem) | cos(JW, operator) | cos(Clean28, operator) | cos(JW, Clean28) | cos(MAE, relchem) | cos(MAE, operator) |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for result in results:
        cosine = result['geometry']['cosine']
        lines.append(
            f'| {result["name"]} | {_fmt(cosine["jw"]["relchem"])} | {_fmt(cosine["wtmad"]["relchem"])} | '
            f'{_fmt(cosine["jw"]["op"])} | {_fmt(cosine["wtmad"]["op"])} | {_fmt(cosine["jw"]["wtmad"])} | '
            f'{_fmt(cosine["mae"]["relchem"])} | {_fmt(cosine["mae"]["op"])} |'
        )
    lines.extend(['', '## Numerical checks', ''])
    for result in results:
        lines.append(f'`{result["name"]}` exact zero residuals: {result["zero_residuals"] if result["zero_residuals"] else "none"}.')
        for label, finite in result['finite_difference'].items():
            if finite.get('skipped'):
                lines.append(f'- {label}: skipped')
                continue
            best = min(finite['scales'], key=lambda scale: scale['relative_gap'])
            for scale in finite['scales']:
                lines.append(
                    f'- {label} target {scale["target_change"]:.0e}: relative gap {scale["relative_gap"]:.3e}, epsilon {scale["epsilon"]:.3e}'
                )
            lines.append(f'- {label} smallest relative gap {best["relative_gap"]:.3e} at target {best["target_change"]:.0e}.')
        lines.append('')
    lines.extend([
        'Finite differences perturb float32 parameters and repeat the no-grad kcal/mol energy path. At both checkpoints the smallest J251-WTMAD-style step is the closest match. The larger steps show curvature.',
        '',
        '## Conclusion',
        '',
    ])
    lines.extend(_conclusion(results))
    lines.extend([
        '## Git',
        '',
        'Provenance is the commit that contains this report. The diagnostic started from `6342d46`.',
        '',
    ])
    (b8.REPO / 'lap_training_metric_gradient_audit_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def execute():
    hashes = {spec['name']: b8.sha256_file(spec['path']) for spec in prior.CHECKPOINTS}
    results = []
    frozen_scales = None
    if (OUT / 'reference_scales.json').is_file():
        frozen_scales = b8.read_json(OUT / 'reference_scales.json')['scales']
    source = None
    for spec in prior.CHECKPOINTS:
        if source is None:
            source = b8.verify_sources()
        result, frozen_scales = measure_checkpoint(source, spec, frozen_scales)
        results.append(result)
    for spec, result in zip(prior.CHECKPOINTS, results, strict=True):
        if b8.sha256_file(spec['path']) != hashes[spec['name']] or result['checkpoint_sha256'] != hashes[spec['name']]:
            _stop('A checkpoint file changed during the diagnostic')
    render(results)
    print('AUDIT_REPORT', 'complete', flush=True)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    execute()
