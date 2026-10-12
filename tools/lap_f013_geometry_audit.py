"""Read-only four-task geometry at the frozen F010 and F012 checkpoints.

No optimizer step. Gradients come from the existing reaction Jacobians and
90-system physical gradients. Temporary perturbations are not saved.
"""
import math

import lap_b8_adamw as b8
import lap_f010_uniform_normalized_chemistry as f010
import lap_f011_normalized_chemistry as arm
import lap_f011_trajectory_gradient_audit as traj
import lap_f012_operator_priority as f012
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np
from scipy.optimize import minimize

PROTOCOL_ID = 'f013-geometry-audit-v1'
OUT = b8.REPO.parent / 'lap_f013_geometry_audit_20261012'
C010 = 9.974823077811354
TASKS = ('jw', 'ae17', 'exc', 'op')
PROBLEM_DB = ('pTC13', 'PA8', 'ABDE4', 'EA13')
MARGIN_FLOOR = 1e-8
# Predeclared before any minibatch gradient is inspected.
MINIBATCH_INDICES = (0, 63, 64, 127, 128)

STATES = (
    {'name': 'b8_t128', 'arm': 'b8', 'local': 0,
     'checkpoint': f010.PARENT_CHECKPOINT,
     'sha256': '1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b',
     'reactions': b8.REPO.parent / 'lap_training_metric_gradient_audit_20261011' / 'b8_t128' / 'reactions.npz',
     'physical': prior.OUT / 'b8_t128'},
    {'name': 'f010_t64', 'arm': 'f010', 'local': 64,
     'checkpoint': f010.OUT / 'ordinary_sgd_adamw' / 'checkpoint_64.pt',
     'sha256': 'eeb9a3d5099c8db7b961b95ca11db08c4a037339c1037c3cf6ced900edc5e4dd',
     'reactions': f010.OUT / 'geometry' / 't64' / 'reactions.npz',
     'physical': f010.OUT / 'geometry' / 't64' / 'physical.npz'},
    {'name': 'f010_t128', 'arm': 'f010', 'local': 128,
     'checkpoint': f010.OUT / 'ordinary_sgd_adamw' / 'checkpoint_128.pt',
     'sha256': 'bd5ecc45fa44793e95d82492e2666d1c40c6fa6068b4f2e266824d3774fbc822',
     'reactions': f010.OUT / 'geometry' / 't128' / 'reactions.npz',
     'physical': f010.OUT / 'geometry' / 't128' / 'physical.npz'},
    {'name': 'f010_t251', 'arm': 'f010', 'local': 251,
     'checkpoint': f010.OUT / 'ordinary_sgd_adamw' / 'checkpoint_251.pt',
     'sha256': '1dba8af1a11d5f98fdc5b40d8be6b137d9773af3a6f2bc9eb1482750b3a5feb3',
     'reactions': f010.OUT / 'geometry' / 't251' / 'reactions.npz',
     'physical': f010.OUT / 'geometry' / 't251' / 'physical.npz'},
    {'name': 'f012_t64', 'arm': 'f012', 'local': 64,
     'checkpoint': f012.OUT / 'ordinary_sgd_adamw' / 'checkpoint_64.pt',
     'sha256': '8a1efed0c4feae8dc75290800d5c255534ae23b557cdf2e35563a7a0c8158838',
     'reactions': f012.OUT / 'geometry' / 't64' / 'reactions.npz',
     'physical': f012.OUT / 'geometry' / 't64' / 'physical.npz'},
    {'name': 'f012_t128', 'arm': 'f012', 'local': 128,
     'checkpoint': f012.OUT / 'ordinary_sgd_adamw' / 'checkpoint_128.pt',
     'sha256': '100ce7f50ff6526c04e7747de2b2ba35466f6032ba3d28d789bca8c18f200996',
     'reactions': f012.OUT / 'geometry' / 't128' / 'reactions.npz',
     'physical': f012.OUT / 'geometry' / 't128' / 'physical.npz'},
    {'name': 'f012_t251', 'arm': 'f012', 'local': 251,
     'checkpoint': f012.OUT / 'ordinary_sgd_adamw' / 'checkpoint_251.pt',
     'sha256': 'a4d3a91e1f8e49cd3d10eb146ccc9a031a78fd7191a6db0eb5a9c37b63342d8a',
     'reactions': f012.OUT / 'geometry' / 't251' / 'reactions.npz',
     'physical': f012.OUT / 'geometry' / 't251' / 'physical.npz'},
)


def _stop(reason):
    OUT.mkdir(parents=True, exist_ok=True)
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not math.isfinite(norm):
        return None, norm
    return vector / norm, norm


def maxmin(vectors):
    """Minimum-norm point in the convex hull of the supplied vectors.

    vectors are already the objects whose positive combination is tested.
    For normalized feasibility they must be unit vectors. Returns gamma = ||m||,
    so the common-descent margin of d = m/||m|| is gamma when gamma > 0.
    """
    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    count, _width = stacked.shape
    starts = [np.full(count, 1.0 / count)]
    for index in range(count):
        vertex = np.zeros(count)
        vertex[index] = 1.0
        starts.append(vertex)
    for left in range(count):
        for right in range(left + 1, count):
            pair = np.zeros(count)
            pair[left] = pair[right] = 0.5
            starts.append(pair)

    def objective(alpha):
        mixed = alpha @ stacked
        return 0.5 * float(np.dot(mixed, mixed))

    constraint = {'type': 'eq', 'fun': lambda alpha: float(np.sum(alpha) - 1.0)}
    bounds = [(0.0, 1.0)] * count
    best = None
    for start in starts:
        found = minimize(objective, start, method='SLSQP', bounds=bounds, constraints=constraint,
                         tol=1e-14, options={'maxiter': 500, 'ftol': 1e-16})
        if best is None or found.fun < best.fun:
            best = found
    alpha = np.clip(np.asarray(best.x, dtype=np.float64), 0.0, None)
    if float(alpha.sum()) <= 0.0:
        _stop('Convex-hull solver returned an empty combination')
    alpha /= alpha.sum()
    mixed = alpha @ stacked
    gamma = float(np.linalg.norm(mixed))
    direction = None if gamma <= MARGIN_FLOOR else mixed / gamma
    margins = stacked @ direction if direction is not None else np.zeros(count)
    stationarity = float(np.max(np.abs(margins - gamma))) if direction is not None else 0.0
    return {
        'gamma': gamma,
        'alpha': alpha,
        'direction': direction,
        'margins': margins,
        'success': bool(best.success),
        'kkt_margin_spread': stationarity,
        'objective': float(best.fun),
        'dual_gap': abs(float(best.fun) - 0.5 * gamma * gamma),
    }


def gram(vectors):
    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    matrix = stacked @ stacked.T
    values = np.linalg.eigvalsh(matrix)
    values = np.clip(values, 0.0, None)
    rank_tol = 1e-8 * float(values[-1]) if values[-1] > 0.0 else 0.0
    rank = int(np.sum(values > rank_tol))
    condition = float(values[-1] / values[0]) if values[0] > rank_tol else math.inf
    return {'eigenvalues': values, 'rank': rank, 'condition': condition}


def _physical(spec):
    path = spec['physical']
    if path.suffix == '.npz':
        with np.load(path) as payload:
            return {key: np.asarray(payload[key], dtype=np.float64) for key in ('ae17', 'exc', 'op')}
    return {task: prior._load_vector(path / f'{task}.npy.npz') for task in ('ae17', 'exc', 'op')}


def load_state(spec, scales):
    if b8.sha256_file(spec['checkpoint']) != spec['sha256']:
        _stop(spec['name'] + ' checkpoint hash does not match the receipt')
    loaded = j251._load_reactions(spec['reactions'])
    if loaded is None:
        _stop(spec['name'] + ' reaction Jacobian is missing')
    rows, jacobian = loaded
    if jacobian.shape != (251, 9446):
        _stop(spec['name'] + ' Jacobian is not 251 by 9446')
    residuals = [row['residual'] for row in rows]
    databases = [row['database'] for row in rows]
    pieces = {}
    for database in sorted(set(databases)):
        mask = np.array([item == database for item in databases], dtype=np.float64)
        coefficients = arm.jw_coefficients(residuals, scales, databases) * mask
        pieces[database] = coefficients @ jacobian
    jw = arm.jw_coefficients(residuals, scales, databases) @ jacobian
    if float(np.linalg.norm(jw - sum(pieces.values()))) > 1e-8:
        _stop(spec['name'] + ' database pieces do not rebuild JW')
    physical = _physical(spec)
    components = {'jw': jw, 'ae17': physical['ae17'], 'exc': physical['exc'], 'op': physical['op']}
    if any(vector.shape != (9446,) or not np.isfinite(vector).all() for vector in components.values()):
        _stop(spec['name'] + ' task gradient is not a finite 9446-vector')
    f010_total = (
        b8.LAMBDAS['relchem'] * C010 * jw + b8.LAMBDAS['ae17'] * components['ae17']
        + b8.LAMBDAS['exc'] * components['exc'] + b8.LAMBDAS['op'] * components['op']
    )
    f012_total = f010_total + b8.LAMBDAS['op'] * components['op']
    return {
        'rows': rows, 'jacobian': jacobian, 'residuals': residuals, 'databases': databases,
        'pieces': pieces, 'components': components, 'f010': f010_total, 'f012': f012_total,
        'jw_scalar': j251.j251_wtmad_style(residuals, databases, scales),
    }


def _active(solved, names):
    if solved['direction'] is None:
        return [names[index] for index, value in enumerate(solved['alpha']) if value > 1e-4]
    return [names[index] for index, value in enumerate(solved['margins']) if abs(value - solved['gamma']) <= 1e-5]


def solve_named(named_vectors):
    present = [(name, vector) for name, vector in named_vectors if vector is not None]
    units = []
    kept = []
    insensitive = []
    for name, vector in present:
        direction, norm = unit(vector)
        if direction is None or norm < 1e-10:
            insensitive.append(name)
            continue
        units.append(direction)
        kept.append(name)
    if len(units) < 2:
        return {'gamma': None, 'names': kept, 'insensitive': insensitive, 'alpha': [], 'active': []}
    solved = maxmin(units)
    solved['names'] = kept
    solved['insensitive'] = insensitive
    solved['active'] = _active(solved, kept)
    solved['alpha_by_name'] = {name: float(value) for name, value in zip(kept, solved['alpha'], strict=True)}
    return solved


def adamw_parts(param, grad, moment, second, step):
    beta1, beta2 = 0.9, 0.999
    nxt = int(step) + 1
    first = beta1 * moment + (1.0 - beta1) * grad
    second_next = beta2 * second + (1.0 - beta2) * np.square(grad)
    corrected = first / (1.0 - beta1 ** nxt)
    second_hat = second_next / (1.0 - beta2 ** nxt)
    preconditioned = corrected / (np.sqrt(second_hat) + 1e-8)
    decay = 0.01 * param
    exact = -2.5e-5 * (preconditioned + decay)
    # B: no historical first moment. The current gradient is preconditioned by the updated second moment.
    fresh = grad / (np.sqrt(second_hat) + 1e-8)
    diagonal = -2.5e-5 * (fresh + decay)
    plain = -2.5e-5 * grad
    scale = float(np.linalg.norm(exact))
    if scale > 0.0:
        plain = plain * (scale / max(float(np.linalg.norm(plain)), 1e-30))
        diagonal = diagonal * (scale / max(float(np.linalg.norm(diagonal)), 1e-30))
    return {
        'exact': exact, 'diagonal': diagonal, 'plain': plain,
        'preconditioned': -2.5e-5 * preconditioned, 'decay': -2.5e-5 * decay,
        'norm': scale,
    }


def geometry(scales, indices):
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    rows = []
    certificates = {}
    database_rows = []
    block_rows = []
    adamw_rows = []
    loaded = {}
    for spec in STATES:
        state = load_state(spec, scales)
        loaded[spec['name']] = state
        components = state['components']
        units = []
        raw_norms = {}
        for task in TASKS:
            direction, norm = unit(components[task])
            units.append(direction)
            raw_norms[task] = norm
        solved = maxmin(units)
        matrix = gram([components[task] for task in TASKS])
        weighted = {
            'jw': b8.LAMBDAS['relchem'] * C010 * components['jw'],
            'ae17': b8.LAMBDAS['ae17'] * components['ae17'],
            'exc': b8.LAMBDAS['exc'] * components['exc'],
            'op': b8.LAMBDAS['op'] * components['op'],
        }
        certificates[spec['name']] = {
            'four': {
                'gamma': solved['gamma'],
                'alpha': {task: float(value) for task, value in zip(TASKS, solved['alpha'], strict=True)},
                'active': _active(solved, TASKS), 'kkt_margin_spread': solved['kkt_margin_spread'],
                'dual_gap': solved['dual_gap'], 'success': solved['success'],
                'eigenvalues': matrix['eigenvalues'].tolist(), 'rank': matrix['rank'],
                'condition': matrix['condition'],
            }
        }
        direction = solved['direction']
        cos_f010 = None if direction is None else prior.cosine(direction, state['f010'])
        cos_f012 = None if direction is None else prior.cosine(direction, state['f012'])
        rows.append({
            'checkpoint': spec['name'], 'gamma': solved['gamma'], 'active': ' '.join(_active(solved, TASKS)),
            'cos_f010': cos_f010, 'cos_f012': cos_f012,
            'cos_jw_op': prior.cosine(components['jw'], components['op']),
            'cos_jw_ae17': prior.cosine(components['jw'], components['ae17']),
            'cos_jw_exc': prior.cosine(components['jw'], components['exc']),
            'cos_ae17_exc': prior.cosine(components['ae17'], components['exc']),
            'cos_ae17_op': prior.cosine(components['ae17'], components['op']),
            'cos_exc_op': prior.cosine(components['exc'], components['op']),
            'cancellation_f010': prior.cancellation_ratio(tuple(weighted.values())),
            'cancellation_f012': prior.cancellation_ratio((
                weighted['jw'], weighted['ae17'], weighted['exc'], 2.0 * weighted['op'],
            )),
            'jw_scalar': state['jw_scalar'], 'rank': matrix['rank'], 'condition': matrix['condition'],
            **{f'norm_{task}': raw_norms[task] for task in TASKS},
        })
        four_plus = [(task, components[task]) for task in TASKS]
        four_plus += [(database, state['pieces'][database]) for database in PROBLEM_DB]
        all_db = [(database, state['pieces'][database]) for database in sorted(state['pieces'])]
        all_db += [(task, components[task]) for task in ('ae17', 'exc', 'op')]
        for label, group in (
            ('four_plus_problem_db', four_plus),
            ('databases_plus_physical', all_db),
        ):
            answer = solve_named(group)
            certificates[spec['name']][label] = {
                'gamma': answer['gamma'], 'alpha': answer.get('alpha_by_name', {}),
                'active': answer.get('active', []), 'insensitive': answer.get('insensitive', []),
                'dual_gap': answer.get('dual_gap'),
            }
        subspaces = {
            'full': np.arange(9446),
            'exchange': indices['exchange'],
            'correlation': indices['correlation'],
            'normalization': indices['normalization'],
            'exchange_correlation': np.concatenate([indices['exchange'], indices['correlation']]),
            'exchange_normalization': np.concatenate([indices['exchange'], indices['normalization']]),
            'correlation_normalization': np.concatenate([indices['correlation'], indices['normalization']]),
        }
        for label, positions in subspaces.items():
            answer = solve_named([(task, components[task][positions]) for task in TASKS])
            block_rows.append({
                'checkpoint': spec['name'], 'subspace': label, 'width': len(positions),
                'gamma': answer['gamma'], 'active': ' '.join(answer.get('active', [])),
                'insensitive': ' '.join(answer.get('insensitive', [])),
                **{f'projected_norm_{task}': float(np.linalg.norm(components[task][positions])) for task in TASKS},
            })
        for database, piece in state['pieces'].items():
            database_rows.append({
                'checkpoint': spec['name'], 'database': database,
                'norm': float(np.linalg.norm(piece)),
                'cosine_jw': prior.cosine(piece, components['jw']),
                'cosine_op': prior.cosine(piece, components['op']),
                'cosine_ae17': prior.cosine(piece, components['ae17']),
                'cosine_exc': prior.cosine(piece, components['exc']),
                'dot_f010': float(np.dot(piece, state['f010'])),
                'dot_f012': float(np.dot(piece, state['f012'])),
            })
        saved = torch_load(spec['checkpoint'])
        numels = [int(saved['model'][name].numel()) for name in names]
        moments = traj._moments(saved, numels)
        param = traj._flat_state(saved, names)
        for policy, grad in (('f010', state['f010']), ('f012', state['f012'])):
            parts = adamw_parts(param, grad, moments['moment'], moments['second'], moments['step'])
            adamw_rows.append({
                'checkpoint': spec['name'], 'policy': policy, 'norm': parts['norm'],
                'step': moments['step'],
                **{f'dot_{task}': float(np.dot(components[task], parts['exact'])) for task in TASKS},
                'cos_exact_f010': prior.cosine(parts['exact'], state['f010']),
                'cos_exact_descent': None if direction is None else prior.cosine(parts['exact'], -direction),
                'cos_plain_exact': prior.cosine(parts['plain'], parts['exact']),
                'cos_diagonal_exact': prior.cosine(parts['diagonal'], parts['exact']),
                'precondition_fraction': float(np.linalg.norm(parts['preconditioned']) / max(parts['norm'], 1e-30)),
                'decay_fraction': float(np.linalg.norm(parts['decay']) / max(parts['norm'], 1e-30)),
                'margin_exact': None if direction is None else min(
                    float(np.dot(unit(components[task])[0], -parts['exact'] / parts['norm'])) for task in TASKS
                ),
            })
        state['direction'] = direction
        state['solved'] = solved
        print('F013_STATE', spec['name'], f"{solved['gamma']:.6f}", ' '.join(_active(solved, TASKS)), flush=True)
    return loaded, rows, certificates, database_rows, block_rows, adamw_rows, names


def torch_load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    scales = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')['scales']
    if arm.scales_sha256(scales) != f012.AD_SHA:
        _stop('A_d hash does not match the frozen calibration')
    source = b8.verify_sources()
    run = b8._import_run()
    model, _shadow = run.model_at(source['initial_state'])
    parameters = run.existing.named_trainable_parameters(model)
    names = tuple(parameters)
    numels = [value.numel() for value in parameters.values()]
    indices = prior.block_indices(names, numels, prior.parameter_blocks(model))
    file_names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    if file_names != names:
        _stop('Parameter order differs from the gradient-audit receipt')
    loaded, rows, certificates, database_rows, block_rows, adamw_rows, names = geometry(scales, indices)
    b8.write_json(OUT / 'summary.json', {
        'rows': rows, 'certificates': certificates, 'databases': database_rows,
        'blocks': block_rows, 'adamw': adamw_rows, 'names': list(names), 'optimizer_updates': 0,
    })
    print('F013_CPU_DONE', len(rows), flush=True)
    return loaded


if __name__ == '__main__':
    execute()
