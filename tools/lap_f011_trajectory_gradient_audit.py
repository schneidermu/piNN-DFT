"""Read-only geometry of the finished F011 trajectory.

No optimizer step, no Clean28 recomputation, and no checkpoint writes.
"""
import csv
import itertools
import time
from pathlib import Path

import lap_b8_adamw as b8
import lap_f011_normalized_chemistry as arm
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np

PROTOCOL_ID = 'f011-trajectory-gradient-audit-v1'
OUT = b8.REPO.parent / 'lap_f011_trajectory_gradient_audit_20261011'
TRAIN = arm.OUT
C_F011 = 9.705596345234895
SCALAR_TOLERANCE = 1e-6
PHYSICAL_TOLERANCE = 1e-5
GRADIENT_TOLERANCE = 1e-6
COUNT = 251
CLEAN28 = {0: 9.096700111658418, 64: 8.782020261263, 128: 8.800214350716, 192: 8.857739596910, 251: 8.938697016838}
CHECKS = (
    {'name': 't0', 'local': 0, 'path': arm.PARENT_CHECKPOINT, 'step': 128},
    {'name': 't64', 'local': 64, 'path': TRAIN / 'ordinary_sgd_adamw' / 'checkpoint_64.pt', 'step': 192},
    {'name': 't128', 'local': 128, 'path': TRAIN / 'ordinary_sgd_adamw' / 'checkpoint_128.pt', 'step': 256},
    {'name': 't192', 'local': 192, 'path': TRAIN / 'ordinary_sgd_adamw' / 'checkpoint_192.pt', 'step': 320},
    {'name': 't251', 'local': 251, 'path': TRAIN / 'ordinary_sgd_adamw' / 'checkpoint_251.pt', 'step': 379},
)
EXPECTED = {
    't0': {'jw': 0.1564146381387076, 'mae': 8.53656789224606, 'relchem': 1.1959392842324374, 'f011': 0.823824025356387,
           'ae17': 3.27546495584018, 'exc': 14.679044990002591, 'op': 0.030933124717599843},
    't64': {'jw': 0.14827453583058048, 'mae': 8.273986729372957, 'relchem': 1.2658785202759986, 'f011': 0.7405388079337765,
            'ae17': 3.8491993014504695, 'exc': 8.360124310622039, 'op': 0.029225784016261303},
    't128': {'jw': 0.14549091131821804, 'mae': 8.15533186101313, 'relchem': 1.2929439384491221, 'f011': 0.7214446197159761,
             'ae17': 1.2346524316719045, 'exc': 6.980487275624526, 'op': 0.02937096989784003},
    't192': {'jw': 0.14407922413960736, 'mae': 8.140767226559175, 'relchem': 1.293709932179434, 'f011': 0.710459394177316,
             'ae17': 2.2317319183544244, 'exc': 11.120048139123403, 'op': 0.029378555268566694},
    't251': {'jw': 0.14353130944011927, 'mae': 8.271702202270674, 'relchem': 1.2711547072155926, 'f011': 0.6981418845318483,
             'ae17': 1.3788842845658227, 'exc': 7.837101212786942, 'op': 0.029234984468957573},
}
TASKS = ('f011', 'ae17', 'exc', 'op')
VECTORS = ('jw', 'mae', 'f010', 'f011', 'relchem', 'ae17', 'exc', 'op', 'total')
FOCUS = arm.FOCUS


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def chemistry_scalars(residuals, databases, scales, raw, coefficient):
    f011 = coefficient * sum(
        raw[index] / scales[databases[index]] * fact.phi_original(residual)
        for index, residual in enumerate(residuals)
    ) / len(residuals)
    return {
        'jw': j251.j251_wtmad_style(residuals, databases, scales),
        'mae': j251.j251_mae(residuals),
        'relchem': fact.production_scalar(residuals, raw),
        'f011': f011,
    }


def chemistry_gradients(residuals, databases, scales, raw, coefficient, jacobian):
    """Task gradients before the outer lambdas. F010 is defined as JW, factor 1."""
    jw = arm.jw_coefficients(residuals, scales, databases) @ jacobian
    relchem = fact.production_coefficients(residuals, raw) @ jacobian
    f011 = coefficient * (arm.f011_coefficients(residuals, raw, scales, databases) @ jacobian)
    mae_coeff = np.array([fact.deriv_l1(residual) / len(residuals) for residual in residuals])
    return {'jw': jw, 'f010': jw.copy(), 'mae': mae_coeff @ jacobian, 'relchem': relchem, 'f011': f011}


def _flat_state(saved, names):
    return np.concatenate([
        np.asarray(saved['model'][name].detach().cpu().reshape(-1), dtype=np.float64) for name in names
    ])


def _moments(saved, numels):
    state = saved['optimizer']['state']
    group = saved['optimizer']['param_groups'][0]
    steps = {int(item['step']) for item in state.values()}
    if len(steps) != 1:
        _stop('Optimizer steps are not uniform')
    moments = []
    seconds = []
    for index, count in enumerate(numels):
        item = state[index]
        moment = np.asarray(item['exp_avg'].detach().cpu().reshape(-1), dtype=np.float64)
        second = np.asarray(item['exp_avg_sq'].detach().cpu().reshape(-1), dtype=np.float64)
        if moment.shape != (count,) or second.shape != (count,):
            _stop('Optimizer moment shape does not match a parameter')
        moments.append(moment)
        seconds.append(second)
    return {
        'step': steps.pop(),
        'group': group,
        'moment': np.concatenate(moments),
        'second': np.concatenate(seconds),
    }


def _save_vectors(path, vectors):
    temporary = Path(path).with_suffix('.partial.npz')
    np.savez_compressed(temporary, **{key: np.asarray(value, dtype=np.float64) for key, value in vectors.items()})
    temporary.replace(path)


def _load_vectors(path):
    with np.load(path) as payload:
        return {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}


def _reaction_jacobian(model, shadow, bundle, dispersion, specs, names, folder):
    cache = folder / 'reactions.npz'
    loaded = j251._load_reactions(cache)
    rows = [] if loaded is None else list(loaded[0])
    gradients = [] if loaded is None else [row for row in loaded[1]]
    if len(rows) > len(specs) or any(rows[index]['identity'] != specs[index]['identity'] for index in range(len(rows))):
        _stop('Reaction cache does not match the J251 manifest')
    import torch
    parameters = b8._import_run().existing.named_trainable_parameters(shadow)
    if tuple(parameters) != names:
        _stop('Shadow parameter order differs from the float32 model')
    for index in range(len(rows), len(specs)):
        spec = specs[index]
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        shadow.load_state_dict(model.state_dict(), strict=True)
        prediction, reference = j251._prediction(shadow, reaction, dispersion)
        residual = prediction - reference
        grads = torch.autograd.grad(prediction, tuple(parameters.values()), allow_unused=True)
        flat = j251._flat(names, grads)
        rows.append({
            'identity': spec['identity'], 'variant': spec['variant'], 'database': spec['database'],
            'reaction_id': spec['reaction_id'], 'reference': float(reference.detach().cpu()),
            'residual': float(residual.detach().cpu()),
        })
        gradients.append(flat)
        del reaction, prediction, reference, residual, grads, flat
        torch.cuda.empty_cache()
        if len(rows) % 25 == 0 or len(rows) == len(specs):
            j251._save_reactions(cache, rows, gradients)
            print('TRAJ_REACTION', folder.name, len(rows), flush=True)
    return rows, np.asarray(gradients, dtype=np.float64)


def _physical(model, shadow, bundle, folder, names):
    path = folder / 'physical.npz'
    if path.is_file():
        return _load_vectors(path)
    import torch
    dispersion = bundle.chemistry_dispersions()
    ae17_scalar, ae17, _missing = prior._mean_chemistry(model, shadow, bundle, 'ae17', dispersion)
    mrks = b8.read_json(b8._import_run().DATA / 'mrks' / 'dispersion.json')
    objectives, flats, _missing_mrks = prior._mean_mrks(model, bundle, mrks)
    vectors = {'ae17': ae17, 'exc': flats['exc'], 'op': flats['op']}
    _save_vectors(path, vectors)
    b8.write_json(folder / 'physical.json', {'ae17': ae17_scalar, 'exc': objectives['exc'], 'op': objectives['op']})
    del ae17
    torch.cuda.empty_cache()
    return vectors


def _next_step(model, shadow, bundle, names, local, coefficient, scales, saved, numels):
    """Analytical AdamW step for the historical next minibatch. Nothing is written back."""
    import torch
    if local >= arm.NEW_UPDATES:
        return {'available': False, 'reason': 'no update followed this checkpoint'}
    batches = arm._batches()
    batch = batches[local]
    if batch['cursor'] != local:
        _stop('The next F011 minibatch is not the recorded schedule position')
    dispersion = bundle.chemistry_dispersions()
    mrks = b8.read_json(b8._import_run().DATA / 'mrks' / 'dispersion.json')
    _loss, chem_grad, _infos, _elapsed = arm.f011_mean(
        model, shadow, bundle, batch['relchem'], dispersion, scales, coefficient)
    _other_losses, other_raw, _system, _seconds = b8.other_tasks(model, shadow, bundle, batch, dispersion, mrks)
    raw = {'relchem': chem_grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op']}
    joint = {
        name: sum(raw[task][name].double() * b8.LAMBDAS[task] for task in b8.TASKS) for name in names
    }
    flat = np.concatenate([
        joint[name].detach().cpu().to(dtype=torch.float32).double().reshape(-1).numpy() for name in names
    ])
    moments = _moments(saved, numels)
    param = _flat_state(saved, names)
    group = moments['group']
    displacement = prior.hypothetical_adamw_displacement(
        param, flat, moments['moment'], moments['second'], moments['step'],
        group['lr'], tuple(group['betas']), group['eps'], group['weight_decay'])
    return {
        'available': True, 'executed': False, 'manifest_index': local,
        'displacement': displacement, 'minibatch_gradient': flat,
        'step': moments['step'], 'lr': group['lr'],
    }


def measure(spec, source):
    import torch
    folder = OUT / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'result.json'
    if done.is_file():
        return b8.read_json(done)
    file_hash = b8.sha256_file(spec['path'])
    run = b8._import_run()
    model, shadow, parameters, saved, steps = j251._load_model(source, spec['path'])
    if steps != {spec['step']}:
        _stop(f'Optimizer step at {spec["name"]} is {steps}, not {spec["step"]}')
    names = tuple(parameters)
    numels = [parameter.numel() for parameter in parameters.values()]
    if sum(numels) != 9446 or tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names']) != names:
        _stop('Parameter order or count differs from the B8 t128 audit')
    digest = run.existing.digest(model)
    moment = arm.continuation.moment_sha(saved['optimizer'])
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
            _stop('Dataset logical hash changed')
        dispersion = bundle.chemistry_dispersions()
        specs = arm._selected(bundle)
        if spec['name'] == 't0':
            rows, jacobian = j251._load_reactions(j251.OUT / 'b8_t128' / 'reactions.npz')
            if rows is None:
                _stop('The B8 t128 reaction Jacobian is missing')
            if [row['identity'] for row in rows] != [spec_row['identity'] for spec_row in specs]:
                _stop('The saved Jacobian is not in the J251 evaluation order')
        else:
            rows, jacobian = _reaction_jacobian(model, shadow, bundle, dispersion, specs, names, folder)
        residuals = [row['residual'] for row in rows]
        databases = [row['database'] for row in rows]
        references = [row['reference'] for row in rows]
        scales, counts = j251.database_reference_scales(references, databases)
        frozen = b8.read_json(j251.OUT / 'reference_scales.json')['scales']
        if any(abs(scales[key] - frozen[key]) > 0.0 for key in scales):
            _stop('A_d changed relative to the frozen J251 audit')
        raw = np.array([j251._factor(database) for database in databases], dtype=np.float64)
        chem = chemistry_gradients(residuals, databases, scales, raw, C_F011, jacobian)
        scalars = chemistry_scalars(residuals, databases, scales, raw, C_F011)
        if spec['name'] == 't0':
            saved_relchem = prior._load_vector(prior.OUT / 'b8_t128' / 'relchem.npy.npz')
            gap = float(np.linalg.norm(chem['relchem'] - saved_relchem) / np.linalg.norm(saved_relchem))
            if gap > GRADIENT_TOLERANCE:
                _stop(f'T0 relchem chain rule missed the saved production gradient by {gap}')
            physical = {
                'ae17': prior._load_vector(prior.OUT / 'b8_t128' / 'ae17.npy.npz'),
                'exc': prior._load_vector(prior.OUT / 'b8_t128' / 'exc.npy.npz'),
                'op': prior._load_vector(prior.OUT / 'b8_t128' / 'op.npy.npz'),
            }
            physical_scalars = {
                task: b8.read_json(prior.OUT / 'b8_t128' / f'{task}.json')['objective'] for task in ('ae17', 'exc', 'op')
            }
        else:
            physical = _physical(model, shadow, bundle, folder, names)
            physical_scalars = b8.read_json(folder / 'physical.json')
        expected = EXPECTED[spec['name']]
        for key in ('jw', 'mae', 'relchem', 'f011'):
            if abs(scalars[key] - expected[key]) > SCALAR_TOLERANCE:
                _stop(f'{spec["name"]} {key} {scalars[key]} missed the F011 receipt {expected[key]}')
        for key in ('ae17', 'exc', 'op'):
            if abs(physical_scalars[key] - expected[key]) > PHYSICAL_TOLERANCE:
                _stop(f'{spec["name"]} {key} {physical_scalars[key]} missed the F011 receipt {expected[key]}')
        total = (
            b8.LAMBDAS['relchem'] * chem['f011'] + b8.LAMBDAS['ae17'] * physical['ae17']
            + b8.LAMBDAS['exc'] * physical['exc'] + b8.LAMBDAS['op'] * physical['op']
        )
        vectors = {**chem, **physical, 'total': total}
        if any(vector.shape != (9446,) or not np.all(np.isfinite(vector)) for vector in vectors.values()):
            _stop(f'A gradient at {spec["name"]} is not a finite 9446-vector')
        if float(np.linalg.norm(vectors['jw'] - vectors['f010'])) != 0.0:
            _stop('F010 was not defined as the same vector as JW')
        _save_vectors(folder / 'gradients.npz', vectors)
        next_step = _next_step(model, shadow, bundle, names, spec['local'], C_F011, scales, saved, numels)
        if next_step.get('displacement') is not None:
            np.save(folder / 'next_displacement.npy', next_step.pop('displacement'))
            next_step.pop('minibatch_gradient')
        if run.existing.digest(model) != digest or b8.sha256_file(spec['path']) != file_hash:
            _stop('Gradient measurement changed the checkpoint')
        if arm.continuation.moment_sha(saved['optimizer']) != moment:
            _stop('Gradient measurement changed the optimizer moments')
        payload = {
            'name': spec['name'], 'local': spec['local'], 'checkpoint_sha256': file_hash,
            'model_sha256': digest, 'moment_sha256': moment, 'optimizer_step': spec['step'],
            'coordinates': 9446, 'scalars': {**scalars, **{key: physical_scalars[key] for key in ('ae17', 'exc', 'op')}},
            'scales': scales, 'counts': counts, 'rows': rows, 'c': C_F011,
            'lambdas': dict(b8.LAMBDAS), 'next_step': next_step, 'optimizer_updates': 0,
            'parameter_names': list(names), 'parameter_numels': numels,
        }
        payload['scalars']['composite'] = (
            b8.LAMBDAS['relchem'] * scalars['f011'] + b8.LAMBDAS['ae17'] * physical_scalars['ae17']
            + b8.LAMBDAS['exc'] * physical_scalars['exc'] + b8.LAMBDAS['op'] * physical_scalars['op']
        )
        b8.write_json(done, payload)
        print('TRAJ_DONE', spec['name'], payload['scalars']['jw'], flush=True)
        return payload
    finally:
        bundle.close()
        del model, shadow
        torch.cuda.empty_cache()


def _fd_f011(source, scales):
    import torch
    spec = CHECKS[-1]
    folder = OUT / spec['name']
    path = folder / 'finite_difference.json'
    if path.is_file():
        return b8.read_json(path)
    vectors = _load_vectors(folder / 'gradients.npz')
    gradient = vectors['f011']
    magnitude = float(np.linalg.norm(gradient))
    unit = gradient / magnitude
    run = b8._import_run()
    model, shadow, parameters, _saved, steps = j251._load_model(source, spec['path'])
    names = tuple(b8.read_json(folder / 'result.json')['parameter_names'])
    if steps != {spec['step']} or tuple(parameters) != names:
        _stop('Finite-difference checkpoint does not match the measured vector')
    before = b8.sha256_file(spec['path'])
    base = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    bundle = run.PublicationDataset(run.DATA)
    specs = arm._selected(bundle)
    raw = {spec_row['database']: j251._factor(spec_row['database']) for spec_row in specs}

    def evaluate(epsilon):
        named = run.existing.named_trainable_parameters(model)
        offset = 0
        with torch.no_grad():
            for name in parameters:
                parameter = named[name]
                count = parameter.numel()
                delta = torch.from_numpy(unit[offset:offset + count] * epsilon).to(parameter.device, parameter.dtype)
                parameter.add_(delta.reshape(parameter.shape))
                offset += count
        try:
            total = 0.0
            dispersion = bundle.chemistry_dispersions()
            for spec_row in specs:
                reaction = b8._load_reaction(bundle, 'relchem', spec_row)
                shadow.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    prediction, reference = j251._prediction(shadow, reaction, dispersion)
                residual = float((prediction - reference).detach().cpu())
                total += C_F011 * raw[spec_row['database']] / scales[spec_row['database']] * fact.phi_original(residual)
                del reaction, prediction, reference
                torch.cuda.empty_cache()
            return total / COUNT
        finally:
            model.load_state_dict(base)
            torch.cuda.empty_cache()

    try:
        epsilon = 1e-3 / magnitude
        estimate = (evaluate(epsilon) - evaluate(-epsilon)) / (2.0 * epsilon)
    finally:
        bundle.close()
    if b8.sha256_file(spec['path']) != before or prior._digest_state(model.state_dict()) != prior._digest_state(base):
        _stop('Finite differences changed the t251 checkpoint')
    payload = {
        'variant': 'f011', 'target_change': 1e-3, 'epsilon': epsilon,
        'finite_difference': estimate, 'autodiff': magnitude,
        'relative_gap': abs(estimate - magnitude) / magnitude, 'executed_optimizer_step': False,
    }
    b8.write_json(path, payload)
    print('TRAJ_FD', payload['relative_gap'], flush=True)
    return payload


def _geometry(vectors, indices):
    cosine = {left: {right: prior.cosine(vectors[left], vectors[right]) for right in VECTORS} for left in VECTORS}
    weighted = {
        'f011': b8.LAMBDAS['relchem'] * vectors['f011'],
        'ae17': b8.LAMBDAS['ae17'] * vectors['ae17'],
        'exc': b8.LAMBDAS['exc'] * vectors['exc'],
        'op': b8.LAMBDAS['op'] * vectors['op'],
    }
    assembled = sum(weighted.values())
    if float(np.linalg.norm(assembled - vectors['total'])) > 1e-8:
        _stop('The total gradient is not the sum of the coefficient-weighted tasks')
    directional = {task: prior.predicted_change(vectors['jw'], weighted[task]) for task in TASKS}
    combined = prior.predicted_change(vectors['jw'], vectors['total'])
    if abs(sum(directional.values()) - combined) > 1e-6:
        _stop('JW task contributions do not sum to the total directional derivative')
    blocks = {}
    for name in ('jw', 'f011', 'op', 'total'):
        shares = prior.norm_shares(vectors[name], indices)
        cosines = {}
        dots = {}
        for block, positions in indices.items():
            if len(positions) == 0:
                cosines[block] = None
                dots[block] = 0.0
                continue
            cosines[block] = prior.cosine(vectors['jw'][positions], vectors[name][positions]) if name != 'jw' else 1.0
            dots[block] = float(np.dot(vectors['jw'][positions], vectors[name][positions]))
        blocks[name] = {'share': shares, 'cosine_with_jw': cosines, 'dot_with_jw': dots}
    full_dot = float(np.dot(vectors['jw'], vectors['total']))
    for block, value in blocks['total']['dot_with_jw'].items():
        blocks['total']['dot_with_jw'][block] = value
        blocks['total'].setdefault('dot_share', {})[block] = None if full_dot == 0.0 else value / full_dot
    return {
        'cosine': cosine,
        'raw_norm': {name: float(np.linalg.norm(vectors[name])) for name in VECTORS},
        'weighted_norm': {task: float(np.linalg.norm(weighted[task])) for task in TASKS},
        'directional': directional,
        'directional_total': combined,
        'cancellation_ratio': prior.cancellation_ratio(list(weighted.values())),
        'blocks': blocks,
    }


def _databases(result, vectors, jacobian):
    rows = result['rows']
    residuals = [row['residual'] for row in rows]
    databases = [row['database'] for row in rows]
    scales = result['scales']
    raw = np.array([j251._factor(row['database']) for row in rows])
    jw_coeff = arm.jw_coefficients(residuals, scales, databases)
    out = []
    assembled = np.zeros(9446)
    scalar_jw = 0.0
    for database in sorted(set(databases)):
        chosen = [index for index, row in enumerate(rows) if row['database'] == database]
        part = jw_coeff[chosen] @ jacobian[chosen]
        assembled += part
        contribution = sum(abs(residuals[index]) / scales[database] for index in chosen) / COUNT
        scalar_jw += contribution
        signed = [residuals[index] for index in chosen]
        out.append({
            'database': database, 'count': len(chosen), 'A_d': scales[database],
            'mean_signed_residual': sum(signed) / len(chosen),
            'mean_absolute_residual': sum(abs(value) for value in signed) / len(chosen),
            'jw_contribution': contribution,
            'relchem_contribution': sum(raw[index] * fact.phi_original(residuals[index]) for index in chosen) / COUNT,
            'f011_contribution': sum(
                C_F011 * raw[index] / scales[database] * fact.phi_original(residuals[index]) for index in chosen
            ) / COUNT,
            'jw_gradient_norm': float(np.linalg.norm(part)),
            'cosine_jw': prior.cosine(part, vectors['jw']),
            'dot_total': float(np.dot(part, vectors['total'])),
            'dot_operator': float(np.dot(part, vectors['op'])),
        })
    if abs(scalar_jw - result['scalars']['jw']) > 1e-8:
        _stop('Database JW contributions do not sum')
    if float(np.linalg.norm(assembled - vectors['jw'])) > 1e-6:
        _stop('Database JW gradients do not sum')
    return out


def _reactions(result, vectors, jacobian):
    rows = result['rows']
    residuals = [row['residual'] for row in rows]
    databases = [row['database'] for row in rows]
    scales = result['scales']
    raw = np.array([j251._factor(row['database']) for row in rows])
    jw_coeff = arm.jw_coefficients(residuals, scales, databases)
    out = []
    for index, row in enumerate(rows):
        out.append({
            'database': row['database'], 'reaction_id': row['reaction_id'], 'identity': row['identity'],
            'signed_residual': row['residual'], 'absolute_residual': abs(row['residual']),
            'jw_contribution': abs(row['residual']) / scales[row['database']] / COUNT,
            'relchem_contribution': raw[index] * fact.phi_original(row['residual']) / COUNT,
            'f011_contribution': C_F011 * raw[index] / scales[row['database']] * fact.phi_original(row['residual']) / COUNT,
            'jw_gradient_norm': float(abs(jw_coeff[index]) * np.linalg.norm(jacobian[index])),
            'dot_total': float(jw_coeff[index] * np.dot(jacobian[index], vectors['total'])),
            'dot_operator': float(jw_coeff[index] * np.dot(jacobian[index], vectors['op'])),
        })
    if abs(sum(row['jw_contribution'] for row in out) - result['scalars']['jw']) > 1e-8:
        _stop('Reaction JW contributions do not sum')
    return out


def _displacement(results, vectors, thetas, moments):
    rows = []
    order = [spec['name'] for spec in CHECKS]
    for left, right in itertools.pairwise(order):
        delta = thetas[right] - thetas[left]
        g_jw = vectors[left]['jw']
        g_total = vectors[left]['total']
        observed = results[right]['scalars']['jw'] - results[left]['scalars']['jw']
        linear = float(np.dot(g_jw, delta))
        trap = 0.5 * float(np.dot(g_jw + vectors[right]['jw'], delta))
        rows.append({
            'interval': f'{left}->{right}',
            'displacement_norm': float(np.linalg.norm(delta)),
            'dot_jw': linear,
            'cosine_jw': prior.cosine(g_jw, delta),
            'dot_total': float(np.dot(g_total, delta)),
            'cosine_total': prior.cosine(g_total, delta),
            'jw_change': observed,
            'linear_estimate': linear,
            'nonlinear_remainder': observed - linear,
            'trapezoid_estimate': trap,
            'jw_cosine_at_end': prior.cosine(vectors[right]['jw'], delta),
            'momentum_cosine_jw': prior.cosine(moments[left], g_jw),
            'momentum_cosine_total': prior.cosine(moments[left], g_total),
        })
    return rows


def _next_alignment(results, vectors):
    rows = []
    for spec in CHECKS:
        result = results[spec['name']]
        path = OUT / spec['name'] / 'next_displacement.npy'
        if not path.is_file():
            rows.append({'checkpoint': spec['name'], 'available': False})
            continue
        displacement = np.load(path)
        rows.append({
            'checkpoint': spec['name'], 'available': True, 'executed': False,
            'manifest_index': result['next_step']['manifest_index'],
            'cosine_jw': prior.cosine(vectors[spec['name']]['jw'], displacement),
            'cosine_total': prior.cosine(vectors[spec['name']]['total'], displacement),
            'norm': float(np.linalg.norm(displacement)),
        })
    return rows


def _write_csv(path, rows, fields):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plots(results, displacements, geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [spec['name'] for spec in CHECKS]
    local = [CHECKS[index]['local'] for index in range(5)]
    jw = [results[name]['scalars']['jw'] for name in order]
    f011 = [results[name]['scalars']['f011'] for name in order]
    figure, axis = plt.subplots()
    axis.plot(local, jw, marker='o', label='JW')
    axis.plot(local, f011, marker='o', label='F011')
    axis.set_xlabel('Local update')
    axis.set_ylabel('Scalar')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_scalars.png', dpi=120)
    plt.close(figure)
    cosine_total = [geometry[name]['cosine']['jw']['total'] for name in order]
    figure, axis = plt.subplots()
    axis.plot(local, cosine_total, marker='o')
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.set_xlabel('Local update')
    axis.set_ylabel('cos(JW, total)')
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_jw_total_cosine.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.plot(local, [geometry[name]['cosine']['jw']['op'] for name in order], marker='o', label='JW/operator')
    axis.plot(local, [geometry[name]['cosine']['jw']['f011'] for name in order], marker='o', label='JW/F011')
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.set_xlabel('Local update')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_task_cosines.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    width = 0.35
    positions = np.arange(len(FOCUS))
    start = []
    end = []
    for database in FOCUS:
        start.append(next(row['jw_contribution'] for row in results['t64']['databases'] if row['database'] == database))
        end.append(next(row['jw_contribution'] for row in results['t251']['databases'] if row['database'] == database))
    axis.bar(positions - width / 2, start, width, label='t64')
    axis.bar(positions + width / 2, end, width, label='t251')
    axis.set_xticks(positions, FOCUS, rotation=30, ha='right')
    axis.set_ylabel('JW contribution')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_database_jw.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    for task in TASKS:
        axis.plot(local, [geometry[name]['directional'][task] for name in order], marker='o', label=task)
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.set_xlabel('Local update')
    axis.set_ylabel('Predicted JW change')
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_directional.png', dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots()
    labels = [row['interval'] for row in displacements]
    axis.plot(labels, [row['jw_change'] for row in displacements], marker='o', label='Observed JW change')
    axis.plot(labels, [row['linear_estimate'] for row in displacements], marker='o', label='First-order estimate')
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(b8.REPO / 'lap_f011_trajectory_displacements.png', dpi=120)
    plt.close(figure)


def _fmt(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return f'{float(value):.6f}'


def render(results, geometry, displacements, databases, reactions, finite, next_steps, seconds, peak):
    order = [spec['name'] for spec in CHECKS]
    metric_rows = []
    cosine_rows = []
    norm_rows = []
    directional_rows = []
    database_rows = []
    reaction_rows = []
    block_rows = []
    by_reaction = {}
    for name in order:
        item = results[name]
        metric_rows.append({'checkpoint': name, 'local': item['local'], **item['scalars'], 'clean28_receipt': CLEAN28[item['local']]})
        for left in VECTORS:
            cosine_rows.append({'checkpoint': name, 'left': left, **{right: geometry[name]['cosine'][left][right] for right in VECTORS}})
        norm_rows.append({'checkpoint': name, **{f'raw_{key}': geometry[name]['raw_norm'][key] for key in VECTORS},
                          **{f'weighted_{key}': geometry[name]['weighted_norm'][key] for key in TASKS},
                          'cancellation_ratio': geometry[name]['cancellation_ratio']})
        directional_rows.append({'checkpoint': name, **geometry[name]['directional'], 'total': geometry[name]['directional_total']})
        for row in databases[name]:
            database_rows.append({'checkpoint': name, **row})
        for row in reactions[name]:
            by_reaction.setdefault(row['identity'], {})[name] = row
    for identity, group in by_reaction.items():
        base = group['t0']
        later = group['t64']
        final = group['t251']
        reaction_rows.append({
            **{f'{name}_{key}': group[name][key] for name in order for key in (
                'signed_residual', 'absolute_residual', 'jw_contribution', 'relchem_contribution', 'f011_contribution')},
            'database': base['database'], 'reaction_id': base['reaction_id'], 'identity': identity,
            'jw_change_from_t0': final['jw_contribution'] - base['jw_contribution'],
            'jw_change_from_t64': final['jw_contribution'] - later['jw_contribution'],
            't64_dot_total': later['dot_total'], 't251_dot_total': final['dot_total'],
            't64_dot_operator': later['dot_operator'], 't251_dot_operator': final['dot_operator'],
            't64_gradient_norm': later['jw_gradient_norm'], 't251_gradient_norm': final['jw_gradient_norm'],
        })
    for name in order:
        for gradient in ('jw', 'f011', 'op', 'total'):
            block = geometry[name]['blocks'][gradient]
            for label in block['share']:
                block_rows.append({
                    'checkpoint': name, 'gradient': gradient, 'block': label,
                    'squared_norm_share': block['share'][label],
                    'cosine_with_jw': block['cosine_with_jw'][label],
                    'dot_with_jw': block['dot_with_jw'][label],
                    'dot_share': block.get('dot_share', {}).get(label),
                })
    _write_csv(b8.REPO / 'lap_f011_trajectory_metrics.csv', metric_rows,
               ['checkpoint', 'local', 'jw', 'mae', 'relchem', 'f011', 'ae17', 'exc', 'op', 'composite', 'clean28_receipt'])
    _write_csv(b8.REPO / 'lap_f011_trajectory_gradient_cosines.csv', cosine_rows, ['checkpoint', 'left', *VECTORS])
    _write_csv(b8.REPO / 'lap_f011_trajectory_gradient_norms.csv', norm_rows,
               ['checkpoint', *[f'raw_{key}' for key in VECTORS], *[f'weighted_{key}' for key in TASKS], 'cancellation_ratio'])
    _write_csv(b8.REPO / 'lap_f011_trajectory_directional_derivatives.csv', directional_rows, ['checkpoint', *TASKS, 'total'])
    _write_csv(b8.REPO / 'lap_f011_trajectory_displacements.csv', displacements, list(displacements[0]))
    _write_csv(b8.REPO / 'lap_f011_trajectory_database_errors.csv', database_rows, list(database_rows[0]))
    _write_csv(b8.REPO / 'lap_f011_trajectory_reaction_errors.csv', reaction_rows, list(reaction_rows[0]))
    _write_csv(b8.REPO / 'lap_f011_trajectory_parameter_blocks.csv', block_rows, list(block_rows[0]))
    _plots(results, displacements, geometry)
    lines = [
        '# F011 trajectory and gradient geometry audit',
        '',
        'Diagnostic only. Optimizer updates: 0. Clean28 was not recomputed. The five Clean28 numbers below are the frozen receipts.',
        '',
        f'`c = {C_F011:.12f}`. F010 is defined as the same vector as JW, so their cosine is 1 by construction. F011 is the chemistry gradient after `c` and before `lambda_chem`.',
        '',
        '## Scalars',
        '',
        '| Checkpoint | JW | MAE | Relchem | F011 | AE17 | Exc | Operator | Composite | Clean28 receipt |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['checkpoint']} | {_fmt(row['jw'])} | {_fmt(row['mae'])} | {_fmt(row['relchem'])} | {_fmt(row['f011'])} | "
            f"{_fmt(row['ae17'])} | {_fmt(row['exc'])} | {_fmt(row['op'])} | {_fmt(row['composite'])} | {row['clean28_receipt']:.12f} |"
        )
    lines.extend(['', '## Interval changes', '', '| Interval | ΔJW | ΔMAE | Δrelchem | ΔF011 | ΔAE17 | ΔExc | Δoperator | ΔClean28 receipt |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for left, right in itertools.pairwise(metric_rows):
        lines.append(
            f"| {left['checkpoint']}->{right['checkpoint']} | {right['jw'] - left['jw']:+.6f} | {right['mae'] - left['mae']:+.6f} | "
            f"{right['relchem'] - left['relchem']:+.6f} | {right['f011'] - left['f011']:+.6f} | {right['ae17'] - left['ae17']:+.6f} | "
            f"{right['exc'] - left['exc']:+.6f} | {right['op'] - left['op']:+.6f} | {right['clean28_receipt'] - left['clean28_receipt']:+.6f} |"
        )
    lines.extend(['', '## Gradient alignment', '', '| Checkpoint | JW/total | JW/operator | JW/F011 | JW/old relchem | JW/AE17 | JW/Exc | F010/F011 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for name in order:
        cosine = geometry[name]['cosine']['jw']
        lines.append(
            f"| {name} | {_fmt(cosine['total'])} | {_fmt(cosine['op'])} | {_fmt(cosine['f011'])} | {_fmt(cosine['relchem'])} | "
            f"{_fmt(cosine['ae17'])} | {_fmt(cosine['exc'])} | {_fmt(geometry[name]['cosine']['f010']['f011'])} |"
        )
    lines.extend(['', '## Directional JW derivatives', '', 'Negative means descent on that coefficient-weighted task locally lowers JW.', '', '| Checkpoint | F011 | AE17 | Exc | Operator | Total |', '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name in order:
        item = geometry[name]
        lines.append(
            f"| {name} | {_fmt(item['directional']['f011'])} | {_fmt(item['directional']['ae17'])} | {_fmt(item['directional']['exc'])} | "
            f"{_fmt(item['directional']['op'])} | {_fmt(item['directional_total'])} |"
        )
    lines.extend(['', '## Checkpoint displacements', '', 'A favorable JW descent has negative cosine between `g_JW` and the displacement. The linear estimate is `dot(g_JW, Δtheta)`, not an AdamW prediction. The trapezoid is an approximation, not a line integral. Momentum cosines in the displacement CSV are the raw first moment at the start of the interval, not the bias-corrected AdamW direction and not the next minibatch.', ''])
    lines.append('| Interval | ||Δθ|| | cos(JW, Δθ) | cos(total, Δθ) | Observed ΔJW | Linear | Remainder | Trapezoid |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in displacements:
        lines.append(
            f"| {row['interval']} | {_fmt(row['displacement_norm'])} | {_fmt(row['cosine_jw'])} | {_fmt(row['cosine_total'])} | "
            f"{row['jw_change']:+.6f} | {row['linear_estimate']:+.6f} | {row['nonlinear_remainder']:+.6f} | {row['trapezoid_estimate']:+.6f} |"
        )
    lines.extend(['', '## Next logged minibatch', '', 'The displacement below is the analytical AdamW step for the historical next eight-reaction minibatch. It uses that minibatch gradient, not the full J251 gradient. It was not applied. A positive cosine with a gradient means the step points the same way as that gradient.', ''])
    for row in next_steps:
        if not row['available']:
            lines.append(f"- `{row['checkpoint']}`: no following update.")
        else:
            lines.append(
                f"- `{row['checkpoint']}` manifest {row['manifest_index']}: cos(JW, step) `{_fmt(row['cosine_jw'])}`, "
                f"cos(total, step) `{_fmt(row['cosine_total'])}`."
            )
    lines.extend(['', '## Databases', '', 'JW contributions use the global `1/251` denominator. The full eight-database table is the database CSV.', ''])
    for name in order:
        lines.extend([f'### {name}', '', '| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |', '| --- | ---: | ---: | ---: | ---: |'])
        for row in databases[name]:
            if row['database'] in FOCUS:
                lines.append(
                    f"| {row['database']} | {_fmt(row['jw_contribution'])} | {_fmt(row['mean_absolute_residual'])} | {_fmt(row['dot_total'])} | {_fmt(row['dot_operator'])} |"
                )
        lines.append('')
    worsened = sorted(reaction_rows, key=lambda row: row['jw_change_from_t64'], reverse=True)[:8]
    improved = sorted(reaction_rows, key=lambda row: row['jw_change_from_t64'])[:8]
    lines.extend(['## Reactions after t64', '', 'Largest JW-contribution increases:', ''])
    for row in worsened:
        lines.append(f"- {row['database']} {row['reaction_id']}: `{row['jw_change_from_t64']:+.6f}`")
    lines.extend(['', 'Largest JW-contribution decreases:', ''])
    for row in improved:
        lines.append(f"- {row['database']} {row['reaction_id']}: `{row['jw_change_from_t64']:+.6f}`")
    lines.extend(['', '## Blocks', '', '| Checkpoint | JW exchange share | F011 exchange share | Operator exchange share | Exchange JW/total cosine |', '| --- | ---: | ---: | ---: | ---: |'])
    for name in order:
        lines.append(
            f"| {name} | {_fmt(geometry[name]['blocks']['jw']['share']['exchange'])} | {_fmt(geometry[name]['blocks']['f011']['share']['exchange'])} | "
            f"{_fmt(geometry[name]['blocks']['op']['share']['exchange'])} | {_fmt(geometry[name]['blocks']['total']['cosine_with_jw']['exchange'])} |"
        )
    cancel = ', '.join(f"{name} {_fmt(geometry[name]['cancellation_ratio'])}" for name in order)
    lines.extend([
        '',
        '## Finite difference',
        '',
        f"t251 F011 symmetric difference at target 1e-3: relative gap `{finite['relative_gap']:.3e}`.",
        '',
        '## Patterns',
        '',
        f"Task cancellation ratios were {cancel}. Pattern A is the t64 snapshot: cos(JW, total) falls from {_fmt(geometry['t0']['cosine']['jw']['total'])} to {_fmt(geometry['t64']['cosine']['jw']['total'])}, and the total JW directional derivative there is {_fmt(geometry['t64']['directional_total'])}. AE17 and Exc supply the positive parts ({_fmt(geometry['t64']['directional']['ae17'])} and {_fmt(geometry['t64']['directional']['exc'])}); F011 and the operator stay negative. The cosine is positive again at t128, t192, and t251, and all four task derivatives are negative at t192 and t251. That is not a sign change that persists.",
        '',
        'Pattern B is the scalar split after t64. JW decreases on every later interval while each frozen Clean28 receipt increases. MAE also rises on t192->t251 while JW falls. No Clean28 gradient was measured, so this is a discrepancy between recorded scalars, not a demonstrated reversal of a Clean28 gradient.',
        '',
        'Pattern C is not supported for the four checkpoint chords. Each chord has negative cosine with g_JW, and the observed JW change has the same sign as `dot(g_JW, Δtheta)`. The t64 instantaneous total derivative is slightly positive, while the chord that leaves t64 still lowers JW. The raw first moment at t64 has cosine '
        + _fmt(next(row['momentum_cosine_jw'] for row in displacements if row['interval'] == 't64->t128'))
        + ' with g_JW, so the retained moment still points along JW there. Individual AdamW steps were not reconstructed. The t192 next-minibatch step, which was not applied, has cosine '
        + _fmt(next(row['cosine_jw'] for row in next_steps if row['checkpoint'] == 't192'))
        + ' with g_JW.',
        '',
        'Pattern D is supported. DBH76 and NCCE31 account for the JW decrease and keep improving after t64. pTC13, PA8, and ABDE4 worsen, mostly on t0->t64, and remain above their t64 JW contributions at t251. Their own JW-gradient pieces have negative cosine with the full g_JW at every checkpoint.',
        '',
        '## Answers',
        '',
    ])
    jw_deltas = [metric_rows[index + 1]['jw'] - metric_rows[index]['jw'] for index in range(4)]
    if not all(value < 0.0 for value in jw_deltas):
        _stop('JW did not fall on every interval; the written Q1 does not apply')
    cosines = [geometry[name]['cosine']['jw']['total'] for name in order]
    lines.append(
        'Q1. JW changed by '
        + ', '.join(f'{value:+.6f}' for value in jw_deltas)
        + ' across the four intervals. It decreased on every interval, including after t64. The decrements become smaller. On t192->t251, MAE changes by '
        + f"{metric_rows[4]['mae'] - metric_rows[3]['mae']:+.6f} while JW changes by {jw_deltas[3]:+.6f}."
    )
    lines.append(
        'Q2. cos(JW, total) was '
        + ', '.join(_fmt(value) for value in cosines)
        + '. It is negative only at t64, by a small amount, and returns to about 0.50–0.55 afterward. A positive cosine is not a claim that every AdamW update decreased JW.'
    )
    lines.append(
        'Q3. cos(JW, operator) was '
        + ', '.join(_fmt(geometry[name]['cosine']['jw']['op']) for name in order)
        + '. It drops at t64 and then recovers, and it stays positive. The operator directional contribution was '
        + ', '.join(_fmt(geometry[name]['directional']['op']) for name in order)
        + ', negative at every checkpoint.'
    )
    lines.append(
        'Q4. Post-t64 JW contribution changes: '
        + ', '.join(
            f"{database} {next(row['jw_contribution'] for row in databases['t251'] if row['database'] == database) - next(row['jw_contribution'] for row in databases['t64'] if row['database'] == database):+.6f}"
            for database in FOCUS
        )
        + '. DBH76 and NCCE31 continue down. pTC13, PA8, and ABDE4 stay above their t64 contributions. The same three already have negative cosine between their JW-gradient piece and the full g_JW at t0.'
    )
    lines.append(
        'Q5. Checkpoint-chord cosines with JW were '
        + ', '.join(f"{row['interval']} {_fmt(row['cosine_jw'])}" for row in displacements)
        + '. All four are negative, so each measured chord points toward JW descent. The linear estimates match the observed JW changes in sign. These chords are not the individual AdamW steps. Raw first-moment cosines with JW at the four interval starts were '
        + ', '.join(_fmt(row['momentum_cosine_jw']) for row in displacements)
        + '.'
    )
    lines.append(
        'Q6. JW kept falling while the frozen Clean28 receipt rose after t64: '
        + ', '.join(f"{row['clean28_receipt']:.12f}" for row in metric_rows)
        + '. The full training gradient remained a separate object from that scalar. No Clean28 gradient was computed, so this audit does not identify a Clean28 cause.'
    )
    f010 = [geometry[name]['cosine']['f010']['f011'] for name in order]
    chem_d = [geometry[name]['directional']['f011'] for name in order]
    lines.append(
        'Q7. cos(F010, F011) was '
        + ', '.join(_fmt(value) for value in f010)
        + '. F010 and F011 are nearly the same direction at t0 and t64. From t128 onward the cosine is about 0.51, so replacing F011 with F010 would materially change the chemistry gradient. The F011 directional contribution stays negative at every checkpoint ('
        + ', '.join(_fmt(value) for value in chem_d)
        + '), so the measured chemistry term does not itself turn against JW.'
    )
    opposed = [
        database for database in ('pTC13', 'PA8', 'ABDE4')
        if all(
            next(row['cosine_jw'] for row in databases[name] if row['database'] == database) < 0.0
            for name in order
        )
    ]
    stake = sum(
        next(row['jw_contribution'] for row in databases['t251'] if row['database'] == database)
        for database in ('pTC13', 'PA8', 'ABDE4')
    )
    lines.append(
        'Q8. Two training-only hypotheses, neither executed and neither selected with Clean28. '
        'First, replace the legacy chemistry factors with the uniform JW gradient (F010) under the same B8 t128 controls, because cos(F010, F011) falls from '
        f'{_fmt(f010[0])} to {_fmt(f010[2])} and stays near {_fmt(f010[4])}. '
        'Second, do not spend the next arm on DBH76 alone or on another external 1/A_d factor: DBH76 remains aligned with g_JW and its JW contribution falls through t251, and 1/A_d is already inside F011. '
        f"{', '.join(opposed)} keep a JW-gradient piece opposed to the full g_JW, but their combined t251 JW contribution is {_fmt(stake)}. "
        'That is a smaller training question than the F010 substitution. The t64 cosine dip is not a reason to stop: the chords through and after it still lower JW.'
    )
    lines.extend([
        '',
        '## Runtime',
        '',
        f'- Seconds: `{seconds:.3f}`',
        f'- Peak allocated bytes: `{peak}`',
        '',
        '## Git',
        '',
        'Provenance is the commit that contains this report. The diagnostic started from `f49df9fd10571902b841b0d1c1d885ffcd8fbe50`.',
        '',
    ])
    (b8.REPO / 'lap_f011_trajectory_gradient_audit_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    b8.write_json(b8.REPO / 'lap_f011_trajectory_audit_receipt.json', {
        'protocol_id': PROTOCOL_ID,
        'c': C_F011,
        'f010_definition': 'identical to JW',
        'clean28_recomputed': False,
        'optimizer_updates': 0,
        'finite_difference_relative_gap': finite['relative_gap'],
        'checkpoints': {name: results[name]['checkpoint_sha256'] for name in order},
        'seconds': seconds,
        'peak_allocated_bytes': peak,
    })


def execute():
    import torch
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source = b8.verify_sources()
    b8._import_run()
    results = {}
    for spec in CHECKS:
        results[spec['name']] = measure(spec, source)
    finite = _fd_f011(source, results['t251']['scales'])
    vectors = {name: _load_vectors(OUT / name / 'gradients.npz') for name in results}
    saved_blocks = b8.read_json(prior.OUT / 'b8_t128' / 'result.json')
    indices = prior.block_indices(
        results['t0']['parameter_names'], results['t0']['parameter_numels'], saved_blocks['blocks'])
    geometry = {}
    databases = {}
    reactions = {}
    thetas = {}
    moments = {}
    for spec in CHECKS:
        name = spec['name']
        geometry[name] = _geometry(vectors[name], indices)
        jacobian_path = j251.OUT / 'b8_t128' / 'reactions.npz' if name == 't0' else OUT / name / 'reactions.npz'
        _rows, jacobian = j251._load_reactions(jacobian_path)
        databases[name] = _databases(results[name], vectors[name], jacobian)
        reactions[name] = _reactions(results[name], vectors[name], jacobian)
        results[name]['databases'] = databases[name]
        saved = torch.load(spec['path'], map_location='cpu', weights_only=False)
        thetas[name] = _flat_state(saved, results[name]['parameter_names'])
        moments[name] = _moments(saved, results[name]['parameter_numels'])['moment']
        del saved
    displacements = _displacement(results, vectors, thetas, moments)
    next_steps = _next_alignment(results, vectors)
    peak = torch.cuda.max_memory_allocated()
    render(results, geometry, displacements, databases, reactions, finite, next_steps, time.perf_counter() - started, peak)
    print('TRAJ_REPORT', 'complete', flush=True)


if __name__ == '__main__':
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        execute()
    except SystemExit as error:
        print('TRAJ_STOP', error, flush=True)
        raise
