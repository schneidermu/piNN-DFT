"""Read-only alignment of the literal Clean28 gradient with the production task gradients.

No optimizer update is taken. Checkpoints are loaded and then left unchanged.
S5 is excluded: its architecture and nine-database metric are not this coordinate system.
"""
import csv
import hashlib
import math
from pathlib import Path

import lap_b8_adamw as b8
import numpy as np
import torch

PROTOCOL_ID = 'wtmad-gradient-alignment-v1'
KCAL = 627.5095
CLEAN_COUNT = 28
TASKS = ('relchem', 'ae17', 'exc', 'op')
VECTORS = ('wtmad',) + TASKS + ('combined',)
HIGHLIGHTS = ('SIE4x4-15', 'WCPT18-15', 'HEAVY28-16', 'BSR36-31', 'BHPERI-11')
EXCLUDED_REACTION = 'SIE4x4-15'
SCALAR_ABS_TOLERANCE = 1e-8
OUT = b8.REPO.parent / 'lap_wtmad_gradient_alignment_20261010'
CHECKPOINTS = (
    {
        'name': 'b8_t128',
        'path': b8.OUT / 'ordinary_sgd_adamw' / 'checkpoint_128.pt',
        'validation': b8.OUT / 'validation_128.json',
        'expected_clean28': 9.096700111658418,
    },
    {
        'name': 'continuation_raw_t251',
        'path': b8.REPO.parent / 'lap_b8_swa_continuation_20261010' / 'ordinary_sgd_adamw' / 'checkpoint_251.pt',
        'validation': b8.REPO.parent / 'lap_b8_swa_continuation_20261010' / 'validation_raw_251.json',
        'expected_clean28': 9.087061815657618,
    },
)


def subgradient_sign(residual):
    """Subgradient of abs at zero is zero. Positive and negative residuals use the ordinary sign."""
    if residual > 0.0:
        return 1.0
    if residual < 0.0:
        return -1.0
    return 0.0


def clean28_scalar(weighted_absolute_errors):
    values = tuple(weighted_absolute_errors)
    if len(values) != CLEAN_COUNT:
        raise ValueError('Clean28 requires the historical 28 reactions')
    return float(sum(values) / CLEAN_COUNT)


def reaction_gradient_scale(weight, residual):
    """Derivative of the 28-reaction mean through abs, without changing the denominator."""
    return float(weight) * subgradient_sign(residual) / CLEAN_COUNT


def cosine(left, right):
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0 or not math.isfinite(left_norm * right_norm):
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def cancellation_ratio(vectors):
    """Zero when the vectors are parallel; one when they cancel exactly."""
    stacked = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
    denominator = float(sum(np.linalg.norm(vector) for vector in stacked))
    if denominator == 0.0:
        return None
    return 1.0 - float(np.linalg.norm(np.sum(np.stack(stacked), axis=0)) / denominator)


def predicted_change(gradient, direction):
    """First-order Clean28 change for a unit step opposite `direction`."""
    return -float(np.dot(gradient, direction))


def normalized_change(gradient, direction):
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return None
    return predicted_change(gradient, direction / norm)


def parameter_blocks(model):
    """Partition trainable names. LayerNorm scales are not also counted inside a network."""
    normalization = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for name, _parameter in module.named_parameters(recurse=False):
                full = f'{module_name}.{name}' if module_name else name
                normalization.add(full)
    assignment = {}
    for name in b8._import_run().existing.named_trainable_parameters(model):
        if name in normalization:
            assignment[name] = 'normalization'
        elif name.startswith(('x_feature_extractor.', 'x_output_layer.')):
            assignment[name] = 'exchange'
        elif name.startswith('c_'):
            assignment[name] = 'correlation'
        else:
            assignment[name] = 'remaining'
    return assignment


def block_indices(names, numels, assignment):
    indices = {block: [] for block in ('exchange', 'correlation', 'normalization', 'remaining')}
    offset = 0
    for name, count in zip(names, numels, strict=True):
        indices[assignment[name]].extend(range(offset, offset + count))
        offset += count
    if offset != sum(numels):
        raise ValueError('Parameter blocks do not cover the coordinate vector')
    covered = [index for values in indices.values() for index in values]
    if sorted(covered) != list(range(offset)):
        raise ValueError('Parameter blocks overlap or leave a gap')
    return {block: np.asarray(values, dtype=np.int64) for block, values in indices.items()}


def norm_shares(vector, indices):
    total = float(np.dot(vector, vector))
    shares = {}
    for block, positions in indices.items():
        if len(positions) == 0:
            shares[block] = 0.0
            continue
        part = vector[positions]
        shares[block] = 0.0 if total == 0.0 else float(np.dot(part, part) / total)
    return shares


def hypothetical_adamw_displacement(param, grad, exp_avg, exp_avg_sq, step, lr, betas, eps, weight_decay):
    """One analytical AdamW displacement. The supplied tensors are not written back."""
    beta1, beta2 = betas
    nxt = int(step) + 1
    moment = beta1 * exp_avg + (1.0 - beta1) * grad
    second = beta2 * exp_avg_sq + (1.0 - beta2) * np.square(grad)
    corrected = moment / (1.0 - beta1**nxt)
    second_hat = second / (1.0 - beta2**nxt)
    update = corrected / (np.sqrt(second_hat) + eps) + weight_decay * param
    return -lr * update


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _digest_state(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_model(source, path):
    run = b8._import_run()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model, shadow = run.model_at(source['initial_state'])
    model.load_state_dict(saved['model'])
    parameters = run.existing.named_trainable_parameters(model)
    coordinates = sum(parameter.numel() for parameter in parameters.values())
    if coordinates != 9446 or any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        _stop('Checkpoint parameters are not the 9446 float32 coordinates')
    if type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or int(model.lap_architecture_version) != 1:
        _stop('Checkpoint architecture is not the qualified Lap model')
    return model, shadow, parameters, saved


def _official_scalar(model, bundle):
    from tools.evaluate_microbatch_endpoint import validation
    before = _digest_state(model.state_dict())
    result = validation(model, bundle)
    if _digest_state(model.state_dict()) != before:
        _stop('The historical Clean28 evaluator changed parameter values')
    rows = [row for row in result['reaction_rows'] if row['clean']]
    if len(rows) != CLEAN_COUNT:
        _stop('Historical Clean28 did not return 28 reactions')
    return result['clean28'], rows


def _species_integral(energy, features, weights):
    from train_models.lap_vxc import sigma_from_gradients
    total = None
    for start in range(0, len(weights), 4096):
        feature = torch.from_numpy(features[start:start + 4096]).to('cuda', torch.float64)
        weight = torch.from_numpy(weights[start:start + 4096]).to('cuda', torch.float64)
        value = energy(feature[:, :2], sigma_from_gradients(feature[:, 2:8].reshape(-1, 2, 3)), feature[:, 8:])
        piece = (value * weight).sum()
        total = piece if total is None else total + piece
        del feature, weight, value
    return total


def _literal_rows(double, bundle):
    """Same fixed-density sum as the historical evaluator, retaining the XC graph when enabled."""
    from train_models.lap_vxc import LapEnergy
    energy = LapEnergy(double)
    energies = {}
    for key, row in bundle.validation_species.items():
        group = bundle.handles.open(row['shard'])[row['group']]
        integral = _species_integral(energy, group['features'][...], group['weights'][...])
        energies[key] = float(group['nonxc'][()]) + float(integral.detach().cpu()) + row['primary_dispersion_hartree']
        del integral
        torch.cuda.empty_cache()
    clean = set(bundle.splits['diet30_clean_validation']['ids'])
    rows = []
    for key, row in bundle.validation_reactions.items():
        predicted = sum(component['coefficient'] * energies[component['species_id']] for component in row['components']) * KCAL
        error = predicted - row['reference_energy_kcal_mol']
        rows.append({
            'reaction_id': row['source_id'],
            'key': key,
            'signed_error_kcal_mol': error,
            'diet_weight': float(row['diet_weight']),
            'weighted_absolute_error': abs(error) * float(row['diet_weight']),
            'clean': key in clean,
            'components': row['components'],
        })
    selected = [row for row in rows if row['clean']]
    return clean28_scalar(row['weighted_absolute_error'] for row in selected), selected


def _species_gradients(double, parameters, bundle, needed):
    from train_models.lap_vxc import LapEnergy
    names = tuple(parameters)
    numels = [parameter.numel() for parameter in parameters.values()]
    energy = LapEnergy(double)
    leaves = tuple(parameters.values())
    stored = {}
    for index, (key, row) in enumerate(bundle.validation_species.items(), start=1):
        if key not in needed:
            continue
        group = bundle.handles.open(row['shard'])[row['group']]
        features = group['features'][...]
        weights = group['weights'][...]
        accumulator = [torch.zeros(parameter.numel(), dtype=torch.float64, device='cpu') for parameter in leaves]
        for start in range(0, len(weights), 4096):
            integral = _species_integral(energy, features[start:start + 4096], weights[start:start + 4096])
            try:
                grads = torch.autograd.grad(integral, leaves, allow_unused=False)
            except RuntimeError as error:
                if 'used in the graph' not in str(error) and 'does not require grad' not in str(error):
                    raise
                grads = torch.autograd.grad(integral, leaves, allow_unused=True)
            missing = [name for name, grad in zip(names, grads, strict=True) if grad is None]
            if missing:
                _stop('WTMAD left disconnected parameters and they were not zero-filled: ' + ', '.join(missing[:8]))
            for piece, grad in zip(accumulator, grads, strict=True):
                piece.add_(grad.detach().reshape(-1).double().cpu())
            del integral, grads
        stored[key] = torch.cat(accumulator).numpy()
        del accumulator
        torch.cuda.empty_cache()
        if index % 10 == 0 or index == len(bundle.validation_species):
            print('WTMAD_SPECIES', index, len(bundle.validation_species), flush=True)
    width = sum(numels)
    if any(value.shape != (width,) for value in stored.values()):
        _stop('A species gradient does not have 9446 coordinates')
    return names, numels, stored


def _reaction_vectors(species_grads, reactions):
    vectors = {}
    for row in reactions:
        scale = reaction_gradient_scale(row['diet_weight'], row['signed_error_kcal_mol'])
        gradient = np.zeros(next(iter(species_grads.values())).shape, dtype=np.float64)
        for component in row['components']:
            gradient += component['coefficient'] * KCAL * species_grads[component['species_id']]
        vectors[row['reaction_id']] = scale * gradient
    total = np.sum(np.stack(list(vectors.values())), axis=0)
    return vectors, total


def _mean_chemistry(model, shadow, bundle, task, dispersion):
    run = b8._import_run()
    evaluation = b8.read_json(b8.REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    if evaluation['policy'] != 'one-variant-per-identity-v1':
        _stop('Evaluation manifest policy changed')
    selected = [row for row in evaluation['rows'] if row['task'] == task]
    expected = 251 if task == 'relchem' else 17
    if len(selected) != expected:
        _stop('Chemistry evaluation manifest does not have one variant per identity')
    names = tuple(run.existing.named_trainable_parameters(model))
    accumulator = None
    total = 0.0
    for index, spec in enumerate(selected, start=1):
        reaction = b8._load_reaction(bundle, task, {
            'identity': spec['identity'], 'variant': spec['variant'],
            'database': bundle.reactions[spec['identity']]['database'],
        })
        value, grad = b8._singleton(model, shadow, reaction, dispersion)
        missing = [name for name in names if name not in grad or grad[name] is None]
        if missing:
            _stop(task + ' left disconnected parameters and they were not zero-filled: ' + ', '.join(missing[:8]))
        if accumulator is None:
            accumulator = {name: torch.zeros_like(grad[name]) for name in names}
        for name in names:
            accumulator[name].add_(grad[name])
        total += float(value)
        del reaction, grad
        torch.cuda.empty_cache()
        if index % 25 == 0 or index == len(selected):
            print('TASK_CHEMISTRY', task, index, len(selected), flush=True)
    count = len(selected)
    for name in names:
        accumulator[name].div_(count)
    flat = np.concatenate([accumulator[name].detach().cpu().reshape(-1).numpy() for name in names])
    return total / count, flat, []


def _mean_mrks(model, bundle, dispersion):
    run = b8._import_run()
    names = tuple(run.existing.named_trainable_parameters(model))
    systems = sorted(bundle.systems)
    if len(systems) != 90:
        _stop('mRKS did not contain 90 systems')
    accumulator = {task: {name: None for name in names} for task in ('exc', 'op')}
    totals = {'exc': 0.0, 'op': 0.0}
    for index, identity in enumerate(systems, start=1):
        system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
        exc_factory, op_factory = run.existing.core.make_mrks_objective_factories(
            model, system, point_chunk_size=256, dispersions=dispersion, exc_chunk_size=4096)
        for task, factory in (('exc', exc_factory), ('op', op_factory)):
            values, grads = run.existing.core.compute_isolated_task_gradients(
                model, {task: factory}, task_order=(task,))
            totals[task] += values[task]
            for name in names:
                grad = grads[task][name]
                if grad is None:
                    _stop(task + ' left a disconnected parameter and it was not zero-filled: ' + name)
                piece = grad.detach().double().cpu()
                if accumulator[task][name] is None:
                    accumulator[task][name] = torch.zeros_like(piece)
                accumulator[task][name].add_(piece)
            del values, grads
        del system, exc_factory, op_factory
        torch.cuda.empty_cache()
        if index % 10 == 0 or index == len(systems):
            print('TASK_MRKS', index, len(systems), flush=True)
    flats = {}
    for task in ('exc', 'op'):
        if any(accumulator[task][name] is None for name in names):
            missing = [name for name in names if accumulator[task][name] is None]
            _stop(f'{task} left parameters without a gradient and they were not zero-filled: {missing[:8]}')
        for name in names:
            accumulator[task][name].div_(90)
        flats[task] = np.concatenate([
            accumulator[task][name].reshape(-1).numpy() for name in names
        ])
    return {task: totals[task] / 90 for task in ('exc', 'op')}, flats, {task: [] for task in ('exc', 'op')}


def _finite_difference(model, bundle, names, gradient, label):
    magnitude = float(np.linalg.norm(gradient))
    if magnitude == 0.0:
        return {'label': label, 'skipped': True}
    unit = gradient / magnitude
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    results = []
    for target in (1e-2, 1e-3, 1e-4):
        epsilon = target / magnitude
        plus = _perturbed_scalar(model, bundle, names, unit, epsilon, base_state)
        minus = _perturbed_scalar(model, bundle, names, unit, -epsilon, base_state)
        estimate = (plus - minus) / (2.0 * epsilon)
        results.append({
            'target_change': target,
            'epsilon': epsilon,
            'finite_difference': estimate,
            'autodiff': magnitude,
            'relative_gap': abs(estimate - magnitude) / magnitude,
            'plus': plus,
            'minus': minus,
        })
    if _digest_state(model.state_dict()) != _digest_state(base_state):
        _stop('Finite differences did not restore the checkpoint weights')
    return {'label': label, 'skipped': False, 'scales': results}


def _perturbed_scalar(model, bundle, names, unit, epsilon, base_state):
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
        value, _rows = _official_scalar(model, bundle)
    finally:
        model.load_state_dict(base_state)
        torch.cuda.empty_cache()
    return value


def _geometry(vectors, indices):
    matrix = {left: {right: cosine(vectors[left], vectors[right]) for right in VECTORS} for left in VECTORS}
    weighted = {task: b8.LAMBDAS[task] * vectors[task] for task in TASKS}
    combined = np.sum(np.stack([weighted[task] for task in TASKS]), axis=0)
    if not np.allclose(combined, vectors['combined']):
        _stop('The combined gradient is not the single application of the four coefficients')
    rows = {}
    for task in TASKS:
        rows[task] = {
            'raw_norm': float(np.linalg.norm(vectors[task])),
            'weighted_norm': float(np.linalg.norm(weighted[task])),
            'dot': float(np.dot(vectors['wtmad'], vectors[task])),
            'weighted_dot': float(np.dot(vectors['wtmad'], weighted[task])),
            'predicted_change': predicted_change(vectors['wtmad'], vectors[task]),
            'weighted_predicted_change': predicted_change(vectors['wtmad'], weighted[task]),
            'normalized_predicted_change': normalized_change(vectors['wtmad'], vectors[task]),
            'weighted_normalized_predicted_change': normalized_change(vectors['wtmad'], weighted[task]),
        }
    blocks = {}
    for task in VECTORS:
        shares = norm_shares(vectors[task], indices)
        cosines = {}
        for block, positions in indices.items():
            if len(positions) == 0:
                cosines[block] = None
                continue
            cosines[block] = cosine(vectors['wtmad'][positions], vectors[task][positions])
        blocks[task] = {'share': shares, 'cosine_with_wtmad': cosines}
    hidden = []
    full = matrix['wtmad']
    for task in TASKS:
        for block, value in blocks[task]['cosine_with_wtmad'].items():
            if value is None or full[task] is None:
                continue
            share = blocks['wtmad']['share'][block]
            if share >= 0.01 and value * full[task] < 0.0:
                hidden.append({'task': task, 'block': block, 'block_cosine': value, 'full_cosine': full[task], 'wtmad_share': share})
    return {
        'cosine': matrix,
        'tasks': rows,
        'wtmad_norm': float(np.linalg.norm(vectors['wtmad'])),
        'combined_norm': float(np.linalg.norm(vectors['combined'])),
        'combined_cosine': matrix['wtmad']['combined'],
        'combined_predicted_change': predicted_change(vectors['wtmad'], vectors['combined']),
        'combined_normalized_predicted_change': normalized_change(vectors['wtmad'], vectors['combined']),
        'cancellation_ratio': cancellation_ratio([weighted[task] for task in TASKS]),
        'blocks': blocks,
        'hidden_opposition': hidden,
    }


def _reaction_report(reaction_vectors, total, reactions):
    rows = []
    for row in reactions:
        gradient = reaction_vectors[row['reaction_id']]
        rows.append({
            'reaction_id': row['reaction_id'],
            'signed_error_kcal_mol': row['signed_error_kcal_mol'],
            'absolute_error_kcal_mol': abs(row['signed_error_kcal_mol']),
            'scalar_contribution': row['weighted_absolute_error'] / CLEAN_COUNT,
            'gradient_norm': float(np.linalg.norm(gradient)),
            'cosine_with_wtmad': cosine(gradient, total),
            'diet_weight': row['diet_weight'],
        })
    excluded = reaction_vectors[EXCLUDED_REACTION]
    remainder = total - excluded
    return rows, remainder, excluded


def _adamw_alignment(saved, names, numels, combined, wtmad):
    group = saved['optimizer']['param_groups'][0]
    state = saved['optimizer']['state']
    if len(state) != len(names):
        return {'available': False, 'reason': 'optimizer state count does not match trainable tensors'}
    steps = {int(item['step']) for item in state.values()}
    if len(steps) != 1:
        return {'available': False, 'reason': 'optimizer steps are not uniform'}
    step = steps.pop()
    moments = []
    seconds = []
    for index, count in enumerate(numels):
        item = state[index]
        moment = np.asarray(item['exp_avg'].detach().cpu().reshape(-1), dtype=np.float64)
        second = np.asarray(item['exp_avg_sq'].detach().cpu().reshape(-1), dtype=np.float64)
        if moment.shape != (count,) or second.shape != (count,):
            return {'available': False, 'reason': 'optimizer moment shape does not match a parameter'}
        moments.append(moment)
        seconds.append(second)
    flat_param = []
    for name, count in zip(names, numels, strict=True):
        tensor = np.asarray(saved['model'][name].detach().cpu().reshape(-1), dtype=np.float64)
        if tensor.shape != (count,):
            return {'available': False, 'reason': 'saved parameter shape does not match the gradient'}
        flat_param.append(tensor)
    param = np.concatenate(flat_param)
    moment = np.concatenate(moments)
    second = np.concatenate(seconds)
    displacement = hypothetical_adamw_displacement(
        param, combined, moment, second, step, group['lr'], tuple(group['betas']), group['eps'], group['weight_decay'])
    beta1 = next(iter(group['betas']))
    corrected = moment / (1.0 - beta1**step)
    return {
        'available': True,
        'executed': False,
        'step': step,
        'lr': group['lr'],
        'momentum_cosine': cosine(corrected, wtmad),
        'hypothetical_full_objective_cosine': cosine(displacement, wtmad),
        'note': (
            'The hypothetical displacement inserts the full four-task gradient into the frozen moments. '
            'It was not applied, and it is not the next training minibatch.'
        ),
    }


def _save_species(path, species):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(species)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, keys=np.asarray(keys), gradients=np.stack([species[key] for key in keys]))
    temporary.replace(path)


def _load_species(path):
    if not Path(path).is_file():
        return None
    with np.load(path) as payload:
        keys = [str(key) for key in payload['keys']]
        gradients = np.asarray(payload['gradients'], dtype=np.float64)
    return {key: gradients[index] for index, key in enumerate(keys)}


def _save_vector(path, vector):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, vector=np.asarray(vector, dtype=np.float64))
    temporary.replace(path)


def _load_vector(path):
    if not Path(path).is_file():
        return None
    with np.load(path) as payload:
        return np.asarray(payload['vector'], dtype=np.float64)


def measure_checkpoint(source, spec):
    folder = OUT / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    file_hash = b8.sha256_file(spec['path'])
    receipt_path = spec['validation']
    expected = spec['expected_clean28']
    stored = b8.read_json(receipt_path)['clean28']
    if stored != expected:
        _stop(f"Frozen Clean28 receipt changed for {spec['name']}")
    run = b8._import_run()
    model, shadow, parameters, saved = _load_model(source, spec['path'])
    digest = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        official, official_rows = _official_scalar(model, bundle)
        if abs(official - expected) > SCALAR_ABS_TOLERANCE:
            b8.write_json(folder / 'scalar_discrepancy.json', {
                'official': official, 'expected': expected, 'stored_file': stored,
            })
            _stop(f'Historical Clean28 scalar was not reproduced at {spec["name"]}: {official} versus {expected}')
        copied = {name: value.detach().clone() for name, value in model.state_dict().items()}
        replica = run.model_at(source['initial_state'])[0]
        replica.load_state_dict(copied)
        replica = replica.double().eval()
        with torch.no_grad():
            literal, reactions = _literal_rows(replica, bundle)
        if abs(literal - official) > SCALAR_ABS_TOLERANCE:
            b8.write_json(folder / 'scalar_discrepancy.json', {'literal': literal, 'official': official})
            _stop(f'Literal Clean28 path disagrees with the historical evaluator at {spec["name"]}')
        by_official = {row['reaction_id']: row['signed_error_kcal_mol'] for row in official_rows}
        for row in reactions:
            official_error = by_official[row['reaction_id']]
            if abs(row['signed_error_kcal_mol'] - official_error) > 1e-6:
                _stop(f"Reaction residual disagrees for {row['reaction_id']}")
            if subgradient_sign(row['signed_error_kcal_mol']) != subgradient_sign(official_error):
                _stop(f"Reaction residual sign disagrees for {row['reaction_id']}")
        zeros = [row['reaction_id'] for row in reactions if row['signed_error_kcal_mol'] == 0.0]
        b8.write_json(folder / 'scalar.json', {
            'clean28': literal, 'official': official, 'expected': expected,
            'zero_residuals': zeros, 'checkpoint_sha256': file_hash, 'model_sha256': digest,
        })
        print('WTMAD_SCALAR', spec['name'], literal, flush=True)
        names = tuple(parameters)
        assignment = parameter_blocks(model)
        numels = [parameter.numel() for parameter in parameters.values()]
        indices = block_indices(names, numels, assignment)
        needed = {component['species_id'] for row in reactions for component in row['components']}
        species_path = folder / 'species_grads.npz'
        species = _load_species(species_path)
        if species is None:
            _names, _numels, species = _species_gradients(
                replica, run.existing.named_trainable_parameters(replica), bundle, needed)
            _save_species(species_path, species)
        if run.existing.digest(model) != digest:
            _stop('WTMAD differentiation changed the float32 checkpoint weights')
        reaction_vectors, wtmad = _reaction_vectors(species, reactions)
        _save_vector(folder / 'wtmad.npy.npz', wtmad)
        reaction_rows, remainder, excluded = _reaction_report(reaction_vectors, wtmad, reactions)
        b8.write_json(folder / 'reactions.json', {
            'rows': reaction_rows,
            'excluded_reaction': EXCLUDED_REACTION,
            'excluded_norm': float(np.linalg.norm(excluded)),
            'remainder_norm': float(np.linalg.norm(remainder)),
            'remainder_cosine_with_wtmad': cosine(remainder, wtmad),
        })
        np.save(folder / 'remainder.npy', remainder)
        del replica, species
        torch.cuda.empty_cache()
        task_vectors = {}
        objectives = {}
        disconnected = {}
        chemistry_dispersion = bundle.chemistry_dispersions()
        for task, count in (('relchem', 251), ('ae17', 17)):
            path = folder / f'{task}.npy.npz'
            cached = _load_vector(path)
            if cached is None:
                objective, cached, missing = _mean_chemistry(model, shadow, bundle, task, chemistry_dispersion)
                _save_vector(path, cached)
                b8.write_json(folder / f'{task}.json', {'objective': objective, 'disconnected': missing})
            else:
                objective = b8.read_json(folder / f'{task}.json')['objective']
                missing = b8.read_json(folder / f'{task}.json')['disconnected']
            if cached.shape != wtmad.shape:
                _stop(f'{task} gradient dimension differs from WTMAD')
            task_vectors[task] = cached
            objectives[task] = objective
            disconnected[task] = missing
            del cached
        mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
        mrks_ready = all((folder / f'{task}.npy.npz').is_file() for task in ('exc', 'op'))
        if not mrks_ready:
            mrks_objectives, flats, mrks_missing = _mean_mrks(model, bundle, mrks_dispersion)
            for task in ('exc', 'op'):
                _save_vector(folder / f'{task}.npy.npz', flats[task])
                b8.write_json(folder / f'{task}.json', {'objective': mrks_objectives[task], 'disconnected': mrks_missing[task]})
        for task in ('exc', 'op'):
            task_vectors[task] = _load_vector(folder / f'{task}.npy.npz')
            meta = b8.read_json(folder / f'{task}.json')
            objectives[task] = meta['objective']
            disconnected[task] = meta['disconnected']
        if run.existing.digest(model) != digest or b8.sha256_file(spec['path']) != file_hash:
            _stop('Task gradients changed the checkpoint file or weights')
        combined = np.sum(np.stack([b8.LAMBDAS[task] * task_vectors[task] for task in TASKS]), axis=0)
        vectors = {'wtmad': wtmad, 'combined': combined, **task_vectors}
        geometry = _geometry(vectors, indices)
        remainder = np.load(folder / 'remainder.npy')
        geometry['sie_exclusion'] = {
            'cosine_with_wtmad': cosine(remainder, wtmad),
            'cosine_with_tasks': {task: cosine(remainder, task_vectors[task]) for task in TASKS},
            'cosine_with_combined': cosine(remainder, combined),
            'predicted_changes': {task: predicted_change(remainder, task_vectors[task]) for task in TASKS},
        }
        fd_path = folder / 'finite_difference.json'
        if fd_path.is_file():
            finite = b8.read_json(fd_path)
        else:
            finite = _finite_difference(model, bundle, names, wtmad, 'wtmad')
            b8.write_json(fd_path, finite)
        alignment = _adamw_alignment(saved, names, numels, combined, wtmad)
        payload = {
            'name': spec['name'],
            'checkpoint_sha256': file_hash,
            'model_sha256': digest,
            'clean28': literal,
            'objectives': objectives,
            'disconnected': disconnected,
            'coordinates': int(wtmad.shape[0]),
            'parameter_names': list(names),
            'parameter_numels': numels,
            'blocks': {name: assignment[name] for name in names},
            'geometry': _jsonable(geometry),
            'reactions': reaction_rows,
            'finite_difference': finite,
            'adamw': _jsonable(alignment),
            'zero_residuals': zeros,
            'optimizer_updates': 0,
        }
        b8.write_json(folder / 'result.json', payload)
        print('WTMAD_DONE', spec['name'], flush=True)
        return payload
    finally:
        bundle.close()
        del model, shadow
        torch.cuda.empty_cache()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value):
    if value is None:
        return 'undefined'
    return f'{value:.6f}'


def _write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render(results):
    matrix_rows = []
    reaction_rows = []
    block_rows = []
    for result in results:
        geometry = result['geometry']
        for left in VECTORS:
            row = {'checkpoint': result['name'], 'gradient': left}
            for right in VECTORS:
                row[right] = geometry['cosine'][left][right]
            matrix_rows.append(row)
        for row in result['reactions']:
            reaction_rows.append({'checkpoint': result['name'], **row})
        for task, block in geometry['blocks'].items():
            for name, share in block['share'].items():
                block_rows.append({
                    'checkpoint': result['name'], 'gradient': task, 'block': name,
                    'squared_norm_share': share, 'cosine_with_wtmad': block['cosine_with_wtmad'][name],
                })
    _write_csv(
        b8.REPO / 'lap_wtmad_gradient_alignment_matrices.csv', matrix_rows,
        ['checkpoint', 'gradient', *VECTORS])
    _write_csv(
        b8.REPO / 'lap_wtmad_gradient_alignment_reactions.csv', reaction_rows,
        ['checkpoint', 'reaction_id', 'signed_error_kcal_mol', 'absolute_error_kcal_mol',
         'scalar_contribution', 'gradient_norm', 'cosine_with_wtmad', 'diet_weight'])
    _write_csv(
        b8.REPO / 'lap_wtmad_gradient_alignment_blocks.csv', block_rows,
        ['checkpoint', 'gradient', 'block', 'squared_norm_share', 'cosine_with_wtmad'])
    lines = [
        '# Literal Clean28 WTMAD-2 gradient alignment',
        '',
        'Diagnostic only. Optimizer updates: 0. No parameter was saved from a perturbed or averaged model.',
        '',
        'The differentiated scalar is the historical Clean28 evaluator: the mean, over the 28 `diet30_clean_validation` reactions, of `abs(error_kcal) * diet_weight`. `error_kcal = (stoichiometry · species_energy) * 627.5095 - reference`. Each species energy is the frozen non-XC term plus the model XC integral on the frozen PBE0 density plus the frozen D3(BJ) term. `diet_weight` is the stored GMTKN55 subset weight. Those weights were not recomputed from these 28 reactions. The subgradient of `abs` at a zero residual is 0. This is not canonical full-GMTKN55 WTMAD-2 and not a squared loss.',
        '',
        f'Publication logical SHA256 `{b8.DATA_SHA}`. Evaluation manifest SHA256 `{b8.EVAL_MANIFEST_SHA}` (`one-variant-per-identity-v1`). Chemistry is the mean of 251 singleton gradients. AE17 is the mean of 17 singletons. Exc and the weak-form operator are means of the same 90 mRKS systems. Coefficients are applied once: relchem `{b8.LAMBDAS["relchem"]}`, ae17 `{b8.LAMBDAS["ae17"]}`, exc `{b8.LAMBDAS["exc"]}`, op `{b8.LAMBDAS["op"]}`.',
        '',
        'S5 was not evaluated. Its historical run uses a different architecture and a nine-database RMSE that includes AE17, so the same coordinates, task definitions, and exact evaluator were not established. No S5 weights were loaded.',
        '',
        '## Scalar parity',
        '',
    ]
    for result in results:
        expected = next(item['expected_clean28'] for item in CHECKPOINTS if item['name'] == result['name'])
        scalar = b8.read_json(OUT / result['name'] / 'scalar.json')
        lines.append(
            f'- `{result["name"]}` historical evaluator `{scalar["official"]:.15f}` equals receipt `{expected:.15f}`. '
            f'The differentiated path returned `{scalar["clean28"]:.15f}` (absolute difference `{abs(scalar["clean28"] - scalar["official"]):.3e}`). '
            f'Checkpoint SHA256 `{result["checkpoint_sha256"]}`. Loaded-model SHA256 `{result["model_sha256"]}`. '
            f'Coordinates {result["coordinates"]}.'
        )
    lines.extend(['', '## Cosine similarity', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', '', '| Gradient | ' + ' | '.join(VECTORS) + ' |', '| --- | ' + ' | '.join(['---:'] * len(VECTORS)) + ' |'])
        for left in VECTORS:
            values = ' | '.join(_fmt(result['geometry']['cosine'][left][right]) for right in VECTORS)
            lines.append(f'| {left} | {values} |')
        lines.append('')
    lines.extend(['## Norms, dots, and first-order changes', ''])
    for result in results:
        geometry = result['geometry']
        lines.extend([
            f'### {result["name"]}',
            '',
            (
                f'WTMAD norm {_fmt(geometry["wtmad_norm"])}. Combined norm {_fmt(geometry["combined_norm"])}. '
                f'Cosine with the combined training gradient {_fmt(geometry["combined_cosine"])}. '
                f'Cancellation ratio {_fmt(geometry["cancellation_ratio"])}.'
            ),
            '',
            '| Task | Raw norm | Weighted norm | Dot | Weighted dot | Raw predicted change | Weighted predicted change | Unit predicted change |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for task in TASKS:
            item = geometry['tasks'][task]
            lines.append(
                f'| {task} | {_fmt(item["raw_norm"])} | {_fmt(item["weighted_norm"])} | {_fmt(item["dot"])} | '
                f'{_fmt(item["weighted_dot"])} | {_fmt(item["predicted_change"])} | '
                f'{_fmt(item["weighted_predicted_change"])} | {_fmt(item["normalized_predicted_change"])} |'
            )
        lines.append(
            f'| combined | {_fmt(geometry["combined_norm"])} | {_fmt(geometry["combined_norm"])} | '
            f'{_fmt(-geometry["combined_predicted_change"])} | {_fmt(-geometry["combined_predicted_change"])} | '
            f'{_fmt(geometry["combined_predicted_change"])} | {_fmt(geometry["combined_predicted_change"])} | '
            f'{_fmt(geometry["combined_normalized_predicted_change"])} |'
        )
        lines.extend([
            '',
            (
                'Predicted change is `-dot(g_wtmad, direction)`. A positive value means descent along that direction is predicted to increase Clean28 at this frozen vector. '
                'Raw uses the task gradient, weighted uses each frozen coefficient once, and unit uses the task gradient normalized to length 1. '
                'Cosine is not an AdamW update prediction. The large raw AE17 and Exc changes come from unnormalized loss scales; the weighted column is their contribution to the training objective.'
            ),
            '',
            (
                f'Production objective scalars at this vector: relchem {_fmt(result["objectives"]["relchem"])}, '
                f'ae17 {_fmt(result["objectives"]["ae17"])}, exc {_fmt(result["objectives"]["exc"])}, '
                f'op {_fmt(result["objectives"]["op"])}.'
            ),
            '',
        ])
    lines.extend(['## Parameter blocks', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', '', '| Gradient | Exchange | Correlation | Normalization | Remaining |', '| --- | ---: | ---: | ---: | ---: |'])
        for task in VECTORS:
            share = result['geometry']['blocks'][task]['share']
            lines.append(
                f'| {task} | {_fmt(share["exchange"])} | {_fmt(share["correlation"])} | '
                f'{_fmt(share["normalization"])} | {_fmt(share["remaining"])} |'
            )
        lines.extend(['', '| Gradient | Exchange cosine | Correlation cosine | Normalization cosine |', '| --- | ---: | ---: | ---: |'])
        for task in VECTORS:
            cosine = result['geometry']['blocks'][task]['cosine_with_wtmad']
            lines.append(
                f'| {task} | {_fmt(cosine["exchange"])} | {_fmt(cosine["correlation"])} | {_fmt(cosine["normalization"])} |'
            )
        hidden = result['geometry']['hidden_opposition']
        if hidden:
            lines.extend(['', 'Blocks holding at least 1% of the WTMAD squared norm whose cosine sign differs from the full-vector cosine:'])
            for item in hidden:
                weak = ''
                if abs(item['full_cosine']) < 0.05 and abs(item['block_cosine']) < 0.05:
                    weak = ' Both values are near zero, so this sign difference is not strong opposition.'
                lines.append(
                    f'- {item["task"]} / {item["block"]}: block cosine {_fmt(item["block_cosine"])}, '
                    f'full cosine {_fmt(item["full_cosine"])}.{weak}'
                )
        else:
            lines.extend(['', 'No parameter block with at least 1% of the WTMAD squared norm has a cosine sign opposite the full-vector cosine.'])
        lines.append('')
    lines.extend(['## Reactions and SIE4x4-15 removed', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', '', '| Reaction | Signed residual | Scalar contribution | Gradient norm | Cosine with full WTMAD |', '| --- | ---: | ---: | ---: | ---: |'])
        by_id = {row['reaction_id']: row for row in result['reactions']}
        for name in HIGHLIGHTS:
            row = by_id[name]
            lines.append(
                f'| {name} | {_fmt(row["signed_error_kcal_mol"])} | {_fmt(row["scalar_contribution"])} | '
                f'{_fmt(row["gradient_norm"])} | {_fmt(row["cosine_with_wtmad"])} |'
            )
        exclusion = result['geometry']['sie_exclusion']
        lines.extend([
            '',
            f'Removing {EXCLUDED_REACTION} without changing the 1/28 denominator leaves a remainder whose cosine with the full gradient is {_fmt(exclusion["cosine_with_wtmad"])}.',
            'Its cosines with the physical gradients are '
            + ', '.join(f'{task} {_fmt(exclusion["cosine_with_tasks"][task])}' for task in TASKS)
            + f', and {_fmt(exclusion["cosine_with_combined"])} with the combined gradient.',
            'Scalar contribution and gradient contribution are different quantities. The full reaction table is in the reactions CSV.',
            '',
        ])
    lines.extend(['## Local alignment', ''])
    for result in results:
        lines.append(f'At `{result["name"]}`:')
        for task in TASKS:
            item = result['geometry']['tasks'][task]
            weighted = item['weighted_predicted_change']
            if weighted is None:
                relation = 'the weighted directional derivative is undefined'
            elif weighted > 0.0:
                relation = 'its coefficient-weighted contribution locally opposes Clean28 reduction'
            elif weighted < 0.0:
                relation = 'its coefficient-weighted contribution locally agrees with Clean28 reduction'
            else:
                relation = 'its coefficient-weighted first-order change is zero'
            lines.append(
                f'- {task}: {relation} (weighted `{_fmt(weighted)}`, raw `{_fmt(item["predicted_change"])}`, '
                f'cosine `{_fmt(result["geometry"]["cosine"]["wtmad"][task])}`).'
            )
        combined = result['geometry']['combined_predicted_change']
        lines.append(
            f'- combined training gradient: weighted predicted change `{_fmt(combined)}`, '
            f'cosine `{_fmt(result["geometry"]["combined_cosine"])}`.'
        )
        sie = next(row for row in result['reactions'] if row['reaction_id'] == EXCLUDED_REACTION)
        lines.append(
            f'- {EXCLUDED_REACTION} scalar contribution `{_fmt(sie["scalar_contribution"])}` '
            f'versus gradient norm `{_fmt(sie["gradient_norm"])}` '
            f'(full WTMAD norm `{_fmt(result["geometry"]["wtmad_norm"])}`).'
        )
        lines.append('')
    lines.extend([
        'Across both frozen vectors the weak-form operator gradient is aligned with the Clean28 gradient, and coefficient-weighted descent on it is predicted to decrease Clean28. The relchem gradient is opposed at both vectors. AE17 and Exc change sides: nearly orthogonal with a very small weighted increase at B8 t128, and weakly aligned with a small weighted decrease at the continuation checkpoint. The combined training gradient stays nearly orthogonal to Clean28. Its first-order predicted change is a small residual, positive at B8 t128 and negative at the continuation checkpoint, after the larger operator and relchem contributions cancel. That is consistent with the training objective moving while Clean28 stays near 9, and it does not identify operator, Exc, or AE17 as a constraint to remove.',
        '',
        'A negative cosine or a positive predicted change is a local observation at these two frozen vectors. It does not show that a physical task is globally harmful, and it is not a reason to weaken or remove that constraint.',
        '',
        '## Numerical checks',
        '',
    ])
    for result in results:
        finite = result['finite_difference']
        lines.append(f'`{result["name"]}` symmetric finite differences along the WTMAD gradient:')
        if finite.get('skipped'):
            lines.append('- skipped because the gradient norm is zero')
        else:
            for scale in finite['scales']:
                lines.append(
                    f'- target change {scale["target_change"]:.1e}: relative gap {scale["relative_gap"]:.3e}, '
                    f'epsilon {scale["epsilon"]:.3e}, finite difference {scale["finite_difference"]:.6f}, '
                    f'autodiff norm {scale["autodiff"]:.6f}'
                )
        best = min(finite['scales'], key=lambda scale: scale['relative_gap'])
        lines.append(
            f'- Smallest relative gap is {best["relative_gap"]:.3e} at target change {best["target_change"]:.0e}.'
        )
        zeros = result['zero_residuals']
        lines.append(f'- Exact zero residuals: {zeros if zeros else "none"}.')
        lines.append(f'- Disconnected task parameters: {result["disconnected"]}.')
        lines.append('')
    lines.extend([
        'WTMAD derivatives use the float64 evaluator copy. Task derivatives use the production F64 chemistry shadow and F64 mRKS gradients on the same 9,446 float32 coordinates. Finite differences perturb that float32 storage and then call the historical evaluator. At B8 t128 the largest step is the closest match and the smallest step shows float32 rounding. At the continuation checkpoint the middle step is the closest match; the largest step shows curvature and the smallest step again shows rounding.',
        '',
        'The two checkpoints are the same loss family. Agreement between them is not independence from the Diet28 sample, the fixed PBE0 densities, or the one-variant chemistry manifest.',
        '',
    ])
    if any(result.get('adamw', {}).get('available') for result in results):
        lines.extend(['## Frozen AdamW moments', ''])
        for result in results:
            adamw = result.get('adamw', {})
            if not adamw.get('available'):
                lines.append(f'- `{result["name"]}`: moments not used ({adamw.get("reason", "unavailable")}).')
                continue
            lines.append(
                f'- `{result["name"]}` saved momentum cosine with WTMAD {_fmt(adamw["momentum_cosine"])}. '
                f'Hypothetical full-objective AdamW displacement cosine {_fmt(adamw["hypothetical_full_objective_cosine"])}. '
                'The displacement was not applied.'
            )
        lines.append('')
    lines.extend([
        '## Git',
        '',
        'Provenance is the commit that contains this report. The diagnostic started from `13c1070`.',
        '',
    ])
    report = b8.REPO / 'lap_wtmad_gradient_alignment_report.md'
    report.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    b8.write_json(b8.REPO / 'lap_wtmad_gradient_alignment_metrics.json', {
        'protocol_id': PROTOCOL_ID,
        'optimizer_updates': 0,
        's5_included': False,
        'results': results,
    })


def execute():
    source = b8.verify_sources()
    hashes = {spec['name']: b8.sha256_file(spec['path']) for spec in CHECKPOINTS}
    results = []
    for spec in CHECKPOINTS:
        result_path = OUT / spec['name'] / 'result.json'
        if result_path.is_file():
            results.append(b8.read_json(result_path))
            continue
        results.append(measure_checkpoint(source, spec))
    for spec, result in zip(CHECKPOINTS, results, strict=True):
        if b8.sha256_file(spec['path']) != hashes[spec['name']] or result['checkpoint_sha256'] != hashes[spec['name']]:
            _stop('A checkpoint file changed during the diagnostic')
    render(results)
    print('WTMAD_REPORT', 'complete', flush=True)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    execute()
