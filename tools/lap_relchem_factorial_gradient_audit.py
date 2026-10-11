"""Read-only factorial split of the Full251 relchem gradient.

The eight diagnostics are measurement targets. None is trained.
Clean28 is not evaluated again.
"""
import csv
import math
from pathlib import Path

import lap_b8_adamw as b8
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np

PROTOCOL_ID = 'relchem-factorial-gradient-audit-v1'
PENALTY_EPSILON = 1e-20
COUNT = 251
VARIANTS = (
    'F000', 'F001', 'F010', 'F011', 'F100', 'F101', 'F110', 'F111',
)
EDGES = (
    ('F000', 'F100', 'penalty'), ('F010', 'F110', 'penalty'),
    ('F001', 'F101', 'penalty'), ('F011', 'F111', 'penalty'),
    ('F000', 'F010', 'reference_normalization'), ('F100', 'F110', 'reference_normalization'),
    ('F001', 'F011', 'reference_normalization'), ('F101', 'F111', 'reference_normalization'),
    ('F000', 'F001', 'legacy_weight'), ('F010', 'F011', 'legacy_weight'),
    ('F100', 'F101', 'legacy_weight'), ('F110', 'F111', 'legacy_weight'),
)
OUT = b8.REPO.parent / 'lap_relchem_factorial_gradient_audit_20261011'
COSINE_TOLERANCE = 1e-8
RELCHEM_TOLERANCE = 1e-6
GRADIENT_TOLERANCE = 1e-6
PRIOR_SCALES = {'DBH76': 18.836, 'NCCE31': 3.536786, 'MGAE109': 504.447019}


def penalty_bits(variant):
    """Bit order is penalty, external 1/A_d, legacy weight."""
    if variant not in VARIANTS or len(variant) != 4 or not variant.startswith('F'):
        raise ValueError(variant)
    return tuple(bit == '1' for bit in variant[1:])


def phi_l1(residual):
    return abs(residual)


def phi_original(residual):
    return math.sqrt(PENALTY_EPSILON + residual * residual)


def deriv_l1(residual):
    return j251.subgradient_sign(residual)


def deriv_original(residual):
    return residual / math.sqrt(PENALTY_EPSILON + residual * residual)


def mean_one_weights(raw_weights):
    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape[0] == 0 or np.any(raw <= 0.0) or not np.all(np.isfinite(raw)):
        raise ValueError('Legacy weights must be finite and positive')
    return raw / raw.mean()


def reaction_coefficients(residuals, scales, databases, legacy_weights, variant):
    """Chain-rule weight for dr_i/dtheta inside one diagnostic. The 1/251 factor is included."""
    original, external, legacy = penalty_bits(variant)
    if len(residuals) != len(databases) or len(residuals) != len(legacy_weights):
        raise ValueError('Factorial inputs do not have one row per reaction')
    coefficients = np.empty(len(residuals), dtype=np.float64)
    for index, residual in enumerate(residuals):
        derivative = deriv_original(residual) if original else deriv_l1(residual)
        scale = 1.0 / scales[databases[index]] if external else 1.0
        weight = legacy_weights[index] if legacy else 1.0
        coefficients[index] = derivative * scale * weight / len(residuals)
    return coefficients


def diagnostic_scalar(residuals, scales, databases, legacy_weights, variant):
    original, external, legacy = penalty_bits(variant)
    total = 0.0
    for index, residual in enumerate(residuals):
        value = phi_original(residual) if original else phi_l1(residual)
        scale = 1.0 / scales[databases[index]] if external else 1.0
        weight = legacy_weights[index] if legacy else 1.0
        total += value * scale * weight
    return total / len(residuals)


def production_coefficients(residuals, raw_weights):
    """Full251 singleton-mean gradient weights, before multiplication by the Jacobian."""
    coefficients = np.empty(len(residuals), dtype=np.float64)
    for index, residual in enumerate(residuals):
        coefficients[index] = raw_weights[index] * deriv_original(residual) / len(residuals)
    return coefficients


def production_scalar(residuals, raw_weights):
    return sum(raw * phi_original(residual) for residual, raw in zip(residuals, raw_weights, strict=True)) / len(residuals)


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _raw_factor(database):
    return j251._factor(database)


def _load_prior(spec):
    folder = prior.OUT / spec['name']
    payload = b8.read_json(folder / 'result.json')
    if payload['checkpoint_sha256'] != b8.sha256_file(spec['path']) or payload['coordinates'] != 9446:
        _stop(f'Saved task-gradient artifact failed for {spec["name"]}')
    names = tuple(payload['parameter_names'])
    numels = list(payload['parameter_numels'])
    vectors = {task: prior._load_vector(folder / f'{task}.npy.npz') for task in ('relchem', 'ae17', 'exc', 'op')}
    if any(vector is None or vector.shape != (9446,) for vector in vectors.values()):
        _stop(f'Saved task gradients are missing for {spec["name"]}')
    vectors['total'] = sum(b8.LAMBDAS[task] * vectors[task] for task in j251.TASKS)
    training = b8.read_json(j251.OUT / spec['name'] / 'result.json')
    if training['checkpoint_sha256'] != payload['checkpoint_sha256']:
        _stop(f'J251 Jacobian checkpoint does not match the task-gradient checkpoint for {spec["name"]}')
    rows, jacobian = j251._load_reactions(j251.OUT / spec['name'] / 'reactions.npz')
    if jacobian.shape != (COUNT, 9446) or len(rows) != COUNT:
        _stop(f'J251 Jacobian is not 251 by 9446 for {spec["name"]}')
    return payload, training, names, numels, vectors, rows, jacobian


def _quoted_cosines():
    path = b8.REPO / 'lap_training_gradient_cosines.csv'
    quoted = {}
    with path.open(encoding='utf-8', newline='\n') as handle:
        for row in csv.DictReader(handle):
            quoted.setdefault(row['checkpoint'], {})[(row['left'], row['right'])] = float(row['cosine'])
    return quoted


def _analyze(spec, quoted):
    payload, training, names, numels, vectors, rows, jacobian = _load_prior(spec)
    residuals = [row['residual'] for row in rows]
    references = [row['reference'] for row in rows]
    databases = [row['database'] for row in rows]
    scales, counts = j251.database_reference_scales(references, databases)
    frozen = b8.read_json(j251.OUT / 'reference_scales.json')['scales']
    if any(abs(scales[key] - frozen[key]) > 0.0 for key in scales):
        _stop('A_d changed relative to the frozen J251 audit')
    for database, expected in PRIOR_SCALES.items():
        if abs(scales[database] - expected) > 5e-4:
            _stop(f'A_d for {database} disagrees with the previous report')
    raw = np.array([_raw_factor(database) for database in databases], dtype=np.float64)
    legacy = mean_one_weights(raw)
    produced = production_scalar(residuals, raw)
    if abs(produced - j251.EXPECTED_RELCHEM[spec['name']]) > RELCHEM_TOLERANCE:
        _stop(f'Full251 relchem scalar was not reproduced at {spec["name"]}: {produced}')
    if abs(produced - training['relchem_scalar']) > RELCHEM_TOLERANCE:
        _stop(f'Relchem scalar disagrees with the J251 audit at {spec["name"]}')
    chain = production_coefficients(residuals, raw) @ jacobian
    relative = float(np.linalg.norm(chain - vectors['relchem']) / np.linalg.norm(vectors['relchem']))
    if relative > GRADIENT_TOLERANCE:
        _stop(f'Chain-rule relchem gradient disagrees with production autodiff at {spec["name"]}: {relative}')
    built = {}
    for variant in VARIANTS:
        coefficients = reaction_coefficients(residuals, scales, databases, legacy, variant)
        gradient = coefficients @ jacobian
        if not np.all(np.isfinite(gradient)):
            _stop(f'{variant} gradient is nonfinite at {spec["name"]}')
        built[variant] = {
            'scalar': diagnostic_scalar(residuals, scales, databases, legacy, variant),
            'gradient': gradient,
            'coefficients': coefficients,
            'nonzero_reactions': int(np.count_nonzero(coefficients)),
        }
    if abs(built['F000']['scalar'] - training['mae_scalar']) > 1e-8:
        _stop('F000 is not the previous J251-MAE scalar')
    if abs(built['F010']['scalar'] - training['jw_scalar']) > 1e-8:
        _stop('F010 is not the previous J251-WTMAD-style scalar')
    scale = float(raw.mean())
    if abs(scale * built['F101']['scalar'] - produced) > 1e-8:
        _stop('F101 is not the mean-one rescaling of the production relchem scalar')
    if float(np.linalg.norm(scale * built['F101']['gradient'] - chain)) > 1e-8:
        _stop('F101 gradient is not the mean-one rescaling of the production gradient')
    checks = {
        ('mae', 'relchem'): prior.cosine(built['F000']['gradient'], vectors['relchem']),
        ('jw', 'relchem'): prior.cosine(built['F010']['gradient'], vectors['relchem']),
        ('jw', 'op'): prior.cosine(built['F010']['gradient'], vectors['op']),
        ('mae', 'op'): prior.cosine(built['F000']['gradient'], vectors['op']),
    }
    for key, value in checks.items():
        if abs(value - quoted[spec['name']][key]) > COSINE_TOLERANCE:
            _stop(f'Previous cosine {key} was not reproduced at {spec["name"]}')
    indices = prior.block_indices(names, numels, payload['blocks'])
    reference = built['F010']['gradient']
    cosine = {
        left: {right: prior.cosine(built[left]['gradient'], built[right]['gradient']) for right in VARIANTS}
        for left in VARIANTS
    }
    comparisons = {}
    for variant, item in built.items():
        gradient = item['gradient']
        comparisons[variant] = {
            'scalar': item['scalar'],
            'norm': float(np.linalg.norm(gradient)),
            'finite': True,
            'nonzero_reactions': item['nonzero_reactions'],
            'cos_f010': prior.cosine(gradient, reference),
            'cos_relchem': prior.cosine(gradient, vectors['relchem']),
            'cos_operator': prior.cosine(gradient, vectors['op']),
            'cos_exc': prior.cosine(gradient, vectors['exc']),
            'cos_ae17': prior.cosine(gradient, vectors['ae17']),
            'cos_total': prior.cosine(gradient, vectors['total']),
            'dot_f010': float(np.dot(reference, gradient)),
            'predicted_change': prior.predicted_change(reference, gradient),
            'unit_predicted_change': prior.normalized_change(reference, gradient),
        }
        shares = prior.norm_shares(gradient, indices)
        comparisons[variant]['shares'] = shares
        for block, positions in indices.items():
            block_dot = float(np.dot(gradient[positions], reference[positions])) if len(positions) else 0.0
            comparisons[variant][block] = {
                'share': shares[block],
                'cosine_f010': prior.cosine(gradient[positions], reference[positions]) if len(positions) else None,
                'dot_f010': block_dot,
                'dot_share': None if comparisons[variant]['dot_f010'] == 0.0 else block_dot / comparisons[variant]['dot_f010'],
            }
    effects = []
    for source, target, factor in EDGES:
        left = built[source]['gradient']
        right = built[target]['gradient']
        difference = right - left
        effects.append({
            'source': source, 'target': target, 'factor': factor,
            'pair_cosine': prior.cosine(left, right),
            'cos_f010_source': comparisons[source]['cos_f010'],
            'cos_f010_target': comparisons[target]['cos_f010'],
            'cos_relchem_source': comparisons[source]['cos_relchem'],
            'cos_relchem_target': comparisons[target]['cos_relchem'],
            'relchem_sign_flip': comparisons[source]['cos_relchem'] * comparisons[target]['cos_relchem'] < 0.0,
            'f010_sign_flip': comparisons[source]['cos_f010'] * comparisons[target]['cos_f010'] < 0.0,
            'norm_source': comparisons[source]['norm'],
            'norm_target': comparisons[target]['norm'],
            'difference_norm': float(np.linalg.norm(difference)),
            'difference_projection_f010': float(np.dot(difference, reference) / np.linalg.norm(reference)),
            'exchange_cosine_source': comparisons[source]['exchange']['cosine_f010'],
            'exchange_cosine_target': comparisons[target]['exchange']['cosine_f010'],
            'correlation_cosine_source': comparisons[source]['correlation']['cosine_f010'],
            'correlation_cosine_target': comparisons[target]['correlation']['cosine_f010'],
        })
    databases_out = []
    for variant, item in built.items():
        assembled = np.zeros_like(item['gradient'])
        for database in sorted(counts):
            chosen = [index for index, row in enumerate(rows) if row['database'] == database]
            part = item['coefficients'][chosen] @ jacobian[chosen]
            assembled += part
            objective = 0.0
            original, external, weighted = penalty_bits(variant)
            for index in chosen:
                value = phi_original(residuals[index]) if original else phi_l1(residuals[index])
                scale = 1.0 / scales[database] if external else 1.0
                weight = legacy[index] if weighted else 1.0
                objective += value * scale * weight / COUNT
            removed = item['gradient'] - part
            databases_out.append({
                'variant': variant, 'database': database, 'count': counts[database],
                'reference_scale': scales[database], 'objective_contribution': objective,
                'gradient_norm': float(np.linalg.norm(part)),
                'dot_f010': float(np.dot(part, reference)),
                'dot_relchem': float(np.dot(part, vectors['relchem'])),
                'dot_operator': float(np.dot(part, vectors['op'])),
                'leave_one_out_cosine_f010': prior.cosine(removed, reference),
                'leave_one_out_cosine_relchem': prior.cosine(removed, vectors['relchem']),
            })
        if float(np.max(np.abs(sum(row['objective_contribution'] for row in databases_out if row['variant'] == variant) - item['scalar']))) > 1e-8:
            _stop(f'Database objectives do not sum for {variant}')
        if float(np.linalg.norm(assembled - item['gradient'])) > 1e-8:
            _stop(f'Database gradients do not sum for {variant}')
    reactions = []
    for index, row in enumerate(rows):
        record = {
            'database': row['database'], 'reaction_id': row['reaction_id'], 'identity': row['identity'],
            'signed_residual': row['residual'], 'absolute_residual': abs(row['residual']),
            'original_penalty': phi_original(row['residual']), 'A_d': scales[row['database']],
            'legacy_weight_raw': float(raw[index]), 'legacy_weight': float(legacy[index]),
            'jacobian_norm': float(np.linalg.norm(jacobian[index])),
        }
        for variant in VARIANTS:
            coefficient = built[variant]['coefficients'][index]
            value_weight = coefficient / (deriv_original(row['residual']) if penalty_bits(variant)[0] else deriv_l1(row['residual'])) if coefficient != 0.0 else 0.0
            penalty = phi_original(row['residual']) if penalty_bits(variant)[0] else phi_l1(row['residual'])
            record[f'scalar_{variant}'] = penalty * value_weight
            record[f'dot_f010_{variant}'] = float(coefficient * np.dot(jacobian[index], reference))
        reactions.append(record)
    for variant in VARIANTS:
        reaction_dot = sum(record[f'dot_f010_{variant}'] for record in reactions)
        full_dot = float(np.dot(built[variant]['gradient'], reference))
        if abs(reaction_dot - full_dot) > 1e-6:
            _stop(f'Reaction dots do not sum for {variant}')
        reaction_gradient = built[variant]['coefficients'] @ jacobian
        if float(np.linalg.norm(reaction_gradient - built[variant]['gradient'])) > 1e-8:
            _stop(f'Reaction gradients do not sum for {variant}')
    derivative_gap = max(abs(deriv_original(residual) - deriv_l1(residual)) for residual in residuals)
    return {
        'name': spec['name'], 'checkpoint_sha256': payload['checkpoint_sha256'],
        'model_sha256': payload['model_sha256'], 'coordinates': 9446,
        'protocol_id': training.get('protocol_id'), 'optimizer_step': training.get('optimizer_step'),
        'relchem_scalar': produced, 'relchem_gradient_relative_l2': relative,
        'legacy_weight_mean': float(raw.mean()), 'scales': scales, 'counts': counts,
        'derivative_gap': derivative_gap, 'zero_residuals': [row['reaction_id'] for row in rows if row['residual'] == 0.0],
        'cosine': cosine, 'comparisons': comparisons, 'effects': effects,
        'databases': databases_out, 'reactions': reactions, 'finite_difference': None,
        'gradients': {variant: item['gradient'] for variant, item in built.items()},
        'optimizer_updates': 0,
    }


def _finite_difference(spec, gradient, scalar):
    import torch
    run = b8._import_run()
    source = b8.verify_sources()
    model, shadow, _parameters, _saved, steps = j251._load_model(source, spec['path'])
    if steps != {j251.EXPECTED_STEPS[spec['name']]}:
        _stop(f'Finite-difference checkpoint step changed for {spec["name"]}')
    bundle = run.PublicationDataset(run.DATA)
    if bundle.manifest['logical_sha256'] != b8.DATA_SHA:
        _stop('Dataset logical hash changed')
    specs = j251._selected(bundle)
    dispersion = bundle.chemistry_dispersions()
    magnitude = float(np.linalg.norm(gradient))
    unit = gradient / magnitude
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    names = tuple(run.existing.named_trainable_parameters(model))
    before = b8.sha256_file(spec['path'])

    def evaluate(epsilon):
        parameters = run.existing.named_trainable_parameters(model)
        offset = 0
        with torch.no_grad():
            for name in names:
                parameter = parameters[name]
                count = parameter.numel()
                delta = torch.from_numpy(np.asarray(unit[offset:offset + count] * epsilon)).to(parameter.device, parameter.dtype)
                parameter.add_(delta.reshape(parameter.shape))
                offset += count
        try:
            print('FACTORIAL_FD_EVAL', spec['name'], epsilon, flush=True)
            total = 0.0
            for spec_row in specs:
                reaction = b8._load_reaction(bundle, 'relchem', spec_row)
                shadow.load_state_dict(model.state_dict(), strict=True)
                prediction, reference = j251._prediction(shadow, reaction, dispersion)
                residual = float((prediction - reference).detach().cpu())
                total += phi_original(residual)
                del reaction, prediction, reference
                torch.cuda.empty_cache()
            return total / COUNT
        finally:
            model.load_state_dict(base_state)
            torch.cuda.empty_cache()

    center = evaluate(0.0)
    if abs(center - scalar) > 1e-6:
        _stop(f'Independent original-penalty scalar {center} disagrees with the chain rule {scalar}')
    epsilon = 1e-3 / magnitude
    estimate = (evaluate(epsilon) - evaluate(-epsilon)) / (2.0 * epsilon)
    if b8.sha256_file(spec['path']) != before or prior._digest_state(model.state_dict()) != prior._digest_state(base_state):
        _stop('Finite differences changed the checkpoint')
    return {
        'variant': 'F100', 'target_change': 1e-3, 'epsilon': epsilon,
        'finite_difference': estimate, 'autodiff': magnitude,
        'relative_gap': abs(estimate - magnitude) / magnitude, 'center': center,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
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
    return f'{float(value):.6f}'


def _write_csv(path, rows, fields):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _findings(results):
    lines = [
        'No Clean28 evaluation was repeated. The Full251 relchem scalar and the chain-rule gradient matched the saved production autodiff at both checkpoints. F000 and F010 reproduced the previous MAE and JW scalars and their cosines with relchem and the operator.',
        '',
        'The production Full251 objective is the mean of 251 singleton penalties. For one reaction, `batch_fchem` returns `factor_d * sqrt(1e-20 + r^2)`, and `factor_d = database_weight_d * frequency_weight_d / mean_database_factor`. The factor multiplies the penalty. It is not inside the square root, and the penalty does not divide by `A_d` or by `|E_ref|`. The eight cells are therefore a real factorization of that singleton-mean objective. `batch_fchem` also contains an unused grouped-RMSE branch that would apply if several reactions of one database were passed to one call. The Full251 gradient does not use that branch: each call receives one reaction.',
        '',
    ]
    for result in results:
        penalty = [row for row in result['effects'] if row['factor'] == 'penalty']
        normalization = [row for row in result['effects'] if row['factor'] == 'reference_normalization']
        legacy = [row for row in result['effects'] if row['factor'] == 'legacy_weight']
        lines.append(
            f'At `{result["name"]}`, every penalty edge has pair cosine {_fmt(min(row["pair_cosine"] for row in penalty))} '
            f'and the derivative gap is {_fmt(result["derivative_gap"])}. '
            'Adding external `1/A_d` changes the cosine with relchem on '
            + ', '.join(
                f'{row["source"]}->{row["target"]} from {_fmt(row["cos_relchem_source"])} to {_fmt(row["cos_relchem_target"])}'
                for row in normalization
            )
            + '. '
            'Adding legacy weights changes the cosine with F010 on '
            + ', '.join(
                f'{row["source"]}->{row["target"]} from {_fmt(row["cos_f010_source"])} to {_fmt(row["cos_f010_target"])}'
                for row in legacy
            )
            + '.'
        )
        production = [row for row in result['databases'] if row['variant'] == 'F101']
        ordered = sorted(production, key=lambda row: row['dot_f010'])
        lines.append(
            'Under the production-proportional gradient F101, database contributions to `dot(g_F010, g_F101)` run from '
            + ', '.join(f'{row["database"]} {_fmt(row["dot_f010"])}' for row in ordered)
            + '. '
            'DBH76 after `1/A_d`, inside F010, contributes '
            f'{_fmt(next(row["dot_relchem"] for row in result["databases"] if row["variant"] == "F010" and row["database"] == "DBH76"))} '
            'to `dot(g_F010, g_relchem)`.'
        )
        lines.append('')
    lines.extend([
        'Q1. The original penalty is not responsible. With normalization and legacy weights held fixed, L1 and `sqrt(1e-20+r^2)` have pair cosine 1 at both checkpoints, and their derivatives agree at every retained residual. The production penalty contains no `A_d` and no `|E_ref|` denominator, so this comparison is a real loss-shape comparison. The shape does not move the gradient.',
        '',
        'Q2. External `1/A_d` reverses the cosine with the production relchem gradient under both penalties and both weighting schemes. On equal weights the cosine with relchem goes from positive to about -0.66. On legacy weights it goes from +1 to about -0.63, and the cosine with F010 goes from about -0.66 to about +0.98. The unnormalized and normalized equal-weight gradients remain partly aligned with each other (pair cosine about 0.46 to 0.49). `1/A_d` changes the sign against relchem. It does not make the two gradients point opposite each other.',
        '',
        'Q3. Legacy weights do reverse the gradient relative to F010 when external normalization is absent. F000 to F001 changes the cosine with F010 from +0.462 to -0.662 at t128 and from +0.491 to -0.673 at t251, and F001 is exactly the production relchem direction. When `1/A_d` is already present, the same weights do not reverse that sign: F010 and F011 stay aligned at cosine 0.980. Legacy weights create the production conflict. They do not keep it once reference normalization is added. The two factors interact, and `1/A_d` dominates the joint cell.',
        '',
        'Q4. DBH76 does not generate the negative projection of the production gradient onto F010. That projection is positive for DBH76 under every factorial cell, including F101. At both checkpoints the negative F101 pieces are pTC13, PA8, and ABDE4, in that order. The equal-weight gradient is aligned with F010 mainly through MGAE109; the legacy frequency and database factors suppress that positive piece and leave the negative pieces in front. A second, different decomposition remains true: inside F010, the DBH76 slice is most of the negative dot with the full relchem gradient, and removing DBH76 from F010 without renormalizing weakens that cosine from about -0.66 to about -0.20. Removing pTC13 from F101 weakens its cosine with F010 from about -0.66 to about -0.20. The conflict is the interaction of those two weightings. It is not the original penalty, and it is not a large DBH76 legacy multiplier. The DBH76 frequency factor is `1/76`.',
        '',
        'Q5. Locally, at both frozen vectors, the production relchem gradient is the legacy-weighted absolute-error gradient and is opposed to F010. That is a local fact about these two checkpoints. It is not a statement about the rest of the path or about AdamW.',
        '',
        'Q6. Next experiments, ranked from this J251 geometry only. None is run here. Each keeps the operator, Exc, and AE17 objectives.',
        '',
        '1. Keep the original per-reaction penalty and multiply it by the frozen external `1/A_d`. The cube puts that gradient on the F010 side whether or not the legacy factors remain (cosine 1 or 0.980). Judge the arm on the training-set normalized absolute error.',
        '',
        '2. Do not spend an arm replacing `sqrt(1e-20+r^2)` with `|r|`. Those gradients are identical here.',
        '',
        '3. If an `1/A_d` arm is not the one being run, the smaller weight experiment is to change the pTC13, PA8, ABDE4, and MGAE109 factors inside the current penalty. A DBH76-only frequency change is not the correction this cube supports: DBH76\'s own production-weighted contribution points with F010.',
        '',
        'Both checkpoints agree on the penalty identity, on which edges flip, and on which databases carry the negative production projection. A negative cosine is not a reason to remove a physical loss.',
        '',
    ])
    return lines


def render(results, receipt):
    cosine_rows = []
    gradient_rows = []
    effect_rows = []
    database_rows = []
    reaction_rows = []
    block_rows = []
    for result in results:
        for left in VARIANTS:
            for right in VARIANTS:
                cosine_rows.append({'checkpoint': result['name'], 'left': left, 'right': right, 'cosine': result['cosine'][left][right]})
        for variant in VARIANTS:
            item = result['comparisons'][variant]
            gradient_rows.append({
                'checkpoint': result['name'], 'variant': variant, 'scalar': item['scalar'], 'norm': item['norm'],
                'finite': item['finite'], 'nonzero_reactions': item['nonzero_reactions'],
                'cos_f010': item['cos_f010'], 'cos_relchem': item['cos_relchem'], 'cos_operator': item['cos_operator'],
                'cos_exc': item['cos_exc'], 'cos_ae17': item['cos_ae17'], 'cos_total': item['cos_total'],
                'dot_f010': item['dot_f010'], 'predicted_change': item['predicted_change'],
                'unit_predicted_change': item['unit_predicted_change'],
                'exchange_share': item['exchange']['share'], 'correlation_share': item['correlation']['share'],
                'normalization_share': item['normalization']['share'],
            })
            for block in ('exchange', 'correlation', 'normalization', 'remaining'):
                block_rows.append({
                    'checkpoint': result['name'], 'variant': variant, 'block': block,
                    'squared_norm_share': item[block]['share'], 'cosine_with_f010': item[block]['cosine_f010'],
                    'dot_with_f010': item[block]['dot_f010'], 'dot_share_with_f010': item[block]['dot_share'],
                })
        for row in result['effects']:
            effect_rows.append({'checkpoint': result['name'], **row})
        for row in result['databases']:
            database_rows.append({'checkpoint': result['name'], **row})
        for row in result['reactions']:
            reaction_rows.append({'checkpoint': result['name'], **row})
    _write_csv(b8.REPO / 'lap_relchem_factorial_cosines.csv', cosine_rows, ['checkpoint', 'left', 'right', 'cosine'])
    _write_csv(
        b8.REPO / 'lap_relchem_factorial_gradients.csv', gradient_rows,
        ['checkpoint', 'variant', 'scalar', 'norm', 'finite', 'nonzero_reactions', 'cos_f010', 'cos_relchem',
         'cos_operator', 'cos_exc', 'cos_ae17', 'cos_total', 'dot_f010', 'predicted_change', 'unit_predicted_change',
         'exchange_share', 'correlation_share', 'normalization_share'])
    _write_csv(
        b8.REPO / 'lap_relchem_factorial_effects.csv', effect_rows,
        ['checkpoint', 'source', 'target', 'factor', 'pair_cosine', 'cos_f010_source', 'cos_f010_target',
         'cos_relchem_source', 'cos_relchem_target', 'relchem_sign_flip', 'f010_sign_flip', 'norm_source', 'norm_target',
         'difference_norm', 'difference_projection_f010', 'exchange_cosine_source', 'exchange_cosine_target',
         'correlation_cosine_source', 'correlation_cosine_target'])
    _write_csv(
        b8.REPO / 'lap_relchem_database_decomposition.csv', database_rows,
        ['checkpoint', 'variant', 'database', 'count', 'reference_scale', 'objective_contribution', 'gradient_norm',
         'dot_f010', 'dot_relchem', 'dot_operator', 'leave_one_out_cosine_f010', 'leave_one_out_cosine_relchem'])
    reaction_fields = [
        'checkpoint', 'database', 'reaction_id', 'identity', 'signed_residual', 'absolute_residual', 'original_penalty',
        'A_d', 'legacy_weight_raw', 'legacy_weight', 'jacobian_norm',
    ]
    for variant in VARIANTS:
        reaction_fields.extend([f'scalar_{variant}', f'dot_f010_{variant}'])
    _write_csv(b8.REPO / 'lap_relchem_reaction_decomposition.csv', reaction_rows, reaction_fields)
    _write_csv(
        b8.REPO / 'lap_relchem_parameter_blocks.csv', block_rows,
        ['checkpoint', 'variant', 'block', 'squared_norm_share', 'cosine_with_f010', 'dot_with_f010', 'dot_share_with_f010'])
    b8.write_json(b8.REPO / 'lap_relchem_factorization_receipt.json', receipt)
    lines = [
        '# Relchem loss factorial gradient audit',
        '',
        'Diagnostic only. Optimizer updates: 0. The eight factorial objectives were not trained.',
        '',
        '## Production formula',
        '',
        'The Full251 relchem objective differentiated in the preceding audits is the arithmetic mean of 251 singleton calls. Each call is one J251 reaction, one quadrature variant, and weight 1 inside `ChemistryBatchObjective`.',
        '',
        '`r_i = E_pred_i - E_ref_i` in kcal/mol, on the fixed PBE0-density production path.',
        '',
        '`phi_orig(r) = sqrt(1e-20 + r^2)`.',
        '',
        '`factor_d = FCHEM_DB_WEIGHTS[d] * FREQ_WEIGHTS[d] / MEAN_WEIGHT`.',
        '',
        '`MEAN_WEIGHT` is the mean of `FCHEM_DB_WEIGHTS * FREQ_WEIGHTS` over the nine coded databases, including AE17, which is not one of the 251 relchem reactions. AE17 therefore changes only the global scale.',
        '',
        '`L_relchem = (1/251) sum_i factor_d(i) * phi_orig(r_i)`.',
        '',
        'This is `C * sum_i w_i * phi_i` with `C = 1/251`, `w_i = factor_d(i)`, and `phi_i = phi_orig`. The frequency weights are the legacy inverse database sizes (`1/76` for DBH76, `1/109` for MGAE109, `1/31` for NCCE31), not the J251 subset counts. The eight B8 presentations are not factors in this scalar. `phi_orig` has no reference-energy denominator. The `1e-20` term is a smoothing epsilon, not a reference scale.',
        '',
        'F000 is equal-weight mean absolute error. F010 is the previous J251-WTMAD-style diagnostic. F101 uses the same penalty and legacy factors as production, with those factors rescaled to mean 1 over the 251 reactions. It is proportional to `L_relchem` by that mean factor. The proportionality was checked on the scalar and on the gradient.',
        '',
        f'Publication logical SHA256 `{b8.DATA_SHA}`. Evaluation manifest SHA256 `{b8.EVAL_MANIFEST_SHA}`. S5 was not loaded. Clean28 was not recomputed.',
        '',
        '## Checkpoints',
        '',
    ]
    for result in results:
        lines.append(
            f'- `{result["name"]}` relchem `{result["relchem_scalar"]:.12f}` against receipt `{j251.EXPECTED_RELCHEM[result["name"]]:.12f}`. '
            f'Chain-rule gradient relative L2 `{result["relchem_gradient_relative_l2"]:.3e}`. '
            f'Checkpoint SHA256 `{result["checkpoint_sha256"]}`. Model SHA256 `{result["model_sha256"]}`. Coordinates {result["coordinates"]}.'
        )
    lines.extend(['', '## Factorial gradients', ''])
    for result in results:
        lines.extend([
            f'### {result["name"]}',
            '',
            '| Variant | Penalty | 1/A_d | Legacy | Scalar | Norm | cos F010 | cos relchem | cos operator | cos total | Predicted F010 change | Unit change |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for variant in VARIANTS:
            item = result['comparisons'][variant]
            penalty, external, legacy = penalty_bits(variant)
            lines.append(
                f'| {variant} | {"original" if penalty else "L1"} | {"yes" if external else "no"} | {"yes" if legacy else "no"} | '
                f'{_fmt(item["scalar"])} | {_fmt(item["norm"])} | {_fmt(item["cos_f010"])} | {_fmt(item["cos_relchem"])} | '
                f'{_fmt(item["cos_operator"])} | {_fmt(item["cos_total"])} | {_fmt(item["predicted_change"])} | {_fmt(item["unit_predicted_change"])} |'
            )
        lines.extend(['', '| | ' + ' | '.join(VARIANTS) + ' |', '| --- | ' + ' | '.join(['---:'] * len(VARIANTS)) + ' |'])
        for left in VARIANTS:
            values = ' | '.join(_fmt(result['cosine'][left][right]) for right in VARIANTS)
            lines.append(f'| {left} | {values} |')
        lines.append('')
    lines.extend(['## Paired factor changes', '', 'A cosine difference is an observable paired change, not an additive share of causation.', ''])
    for result in results:
        lines.extend([
            f'### {result["name"]}',
            '',
            '| Edge | Factor | Pair cosine | cos(source, F010) | cos(target, F010) | cos(source, relchem) | cos(target, relchem) | Relchem sign flip | F010 sign flip |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |',
        ])
        for row in result['effects']:
            lines.append(
                f'| {row["source"]}->{row["target"]} | {row["factor"]} | {_fmt(row["pair_cosine"])} | {_fmt(row["cos_f010_source"])} | '
                f'{_fmt(row["cos_f010_target"])} | {_fmt(row["cos_relchem_source"])} | {_fmt(row["cos_relchem_target"])} | '
                f'{_fmt(row["relchem_sign_flip"])} | {_fmt(row["f010_sign_flip"])} |'
            )
        lines.append('')
    lines.extend(['## Databases', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('| Variant | Database | Count | A_d | Objective | Gradient norm | Dot F010 | Dot relchem | Dot operator | Leave-one-out cos F010 |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for row in result['databases']:
            if row['variant'] not in ('F000', 'F001', 'F010', 'F011'):
                continue
            lines.append(
                f'| {row["variant"]} | {row["database"]} | {row["count"]} | {_fmt(row["reference_scale"])} | {_fmt(row["objective_contribution"])} | '
                f'{_fmt(row["gradient_norm"])} | {_fmt(row["dot_f010"])} | {_fmt(row["dot_relchem"])} | {_fmt(row["dot_operator"])} | '
                f'{_fmt(row["leave_one_out_cosine_f010"])} |'
            )
        lines.append('')
    lines.extend(['## Reactions', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', '', 'Largest negative contributions of the production-proportional variant F101 to `dot(g_F010, g_F101)`:', ''])
        lines.append('| Database | Reaction | Residual | A_d | Legacy weight | Jacobian norm | Dot F010 |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
        ranked = sorted(result['reactions'], key=lambda row: row['dot_f010_F101'])
        for row in ranked[:8]:
            lines.append(
                f'| {row["database"]} | {row["reaction_id"]} | {_fmt(row["signed_residual"])} | {_fmt(row["A_d"])} | '
                f'{_fmt(row["legacy_weight"])} | {_fmt(row["jacobian_norm"])} | {_fmt(row["dot_f010_F101"])} |'
            )
        lines.append('')
    lines.extend(['## Parameter blocks', ''])
    for result in results:
        lines.extend([f'### {result["name"]}', ''])
        lines.append('| Variant | Exchange share | Exchange cos F010 | Exchange dot share | Correlation share | Correlation cos F010 | Correlation dot share | Normalization share | Remaining share |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for variant in VARIANTS:
            item = result['comparisons'][variant]
            lines.append(
                f'| {variant} | {_fmt(item["exchange"]["share"])} | {_fmt(item["exchange"]["cosine_f010"])} | {_fmt(item["exchange"]["dot_share"])} | '
                f'{_fmt(item["correlation"]["share"])} | {_fmt(item["correlation"]["cosine_f010"])} | {_fmt(item["correlation"]["dot_share"])} | '
                f'{_fmt(item["normalization"]["share"])} | {_fmt(item["remaining"]["share"])} |'
            )
        lines.append('')
    lines.extend(['## Numerical checks', ''])
    for result in results:
        finite = result.get('finite_difference')
        if not finite:
            lines.append(f'- `{result["name"]}` original-penalty finite difference: not recorded.')
        else:
            lines.append(
                f'- `{result["name"]}` F100 symmetric difference at target {finite["target_change"]:.0e}: '
                f'relative gap {finite["relative_gap"]:.3e}, epsilon {finite["epsilon"]:.3e}.'
            )
        lines.append(f'- Exact zero residuals: {result["zero_residuals"] if result["zero_residuals"] else "none"}.')
    lines.extend(['', '## Conclusion', ''])
    lines.extend(_findings(results))
    lines.extend(['## Git', '', 'Provenance is the commit that contains this report. The diagnostic started from `acf6a49`.', ''])
    (b8.REPO / 'lap_relchem_factorial_gradient_audit_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def execute(include_finite_difference=True):
    b8._import_run()
    quoted = _quoted_cosines()
    hashes = {spec['name']: b8.sha256_file(spec['path']) for spec in prior.CHECKPOINTS}
    results = []
    for spec in prior.CHECKPOINTS:
        folder = OUT / spec['name']
        folder.mkdir(parents=True, exist_ok=True)
        cached = folder / 'result.json'
        if cached.is_file():
            result = b8.read_json(cached)
        else:
            analyzed = _analyze(spec, quoted)
            gradients = analyzed.pop('gradients')
            result = _jsonable(analyzed)
            b8.write_json(cached, result)
            print('FACTORIAL', spec['name'], result['relchem_gradient_relative_l2'], flush=True)
            if include_finite_difference:
                finite = _jsonable(_finite_difference(spec, gradients['F100'], result['comparisons']['F100']['scalar']))
                result['finite_difference'] = finite
                b8.write_json(cached, result)
                print('FACTORIAL_FD', spec['name'], finite['relative_gap'], flush=True)
        if include_finite_difference and not result.get('finite_difference'):
            analyzed = _analyze(spec, quoted)
            finite = _jsonable(_finite_difference(spec, analyzed['gradients']['F100'], result['comparisons']['F100']['scalar']))
            result['finite_difference'] = finite
            b8.write_json(cached, result)
            print('FACTORIAL_FD', spec['name'], finite['relative_gap'], flush=True)
        results.append(result)
    for spec, result in zip(prior.CHECKPOINTS, results, strict=True):
        if b8.sha256_file(spec['path']) != hashes[spec['name']] or result['checkpoint_sha256'] != hashes[spec['name']]:
            _stop('A checkpoint file changed during the diagnostic')
    receipt = {
        'protocol_id': PROTOCOL_ID,
        'optimizer_updates': 0,
        'penalty_epsilon': PENALTY_EPSILON,
        'formula': 'L_relchem = mean_i factor_d(i) * sqrt(1e-20 + r_i^2)',
        'factor': 'FCHEM_DB_WEIGHTS[d] * FREQ_WEIGHTS[d] / MEAN_WEIGHT',
        'f000': 'equal-weight mean absolute error',
        'f010': 'J251-WTMAD-style, external 1/A_d, not literal GMTKN55 WTMAD-2',
        'f101': 'original penalty with legacy factors rescaled to mean 1; proportional to Full251 relchem',
        'grouped_rmse_branch': 'present in batch_fchem and inactive for the singleton Full251 objective',
        'clean28_recomputed': False,
        's5_loaded': False,
        'results': [
            {
                'name': result['name'], 'relchem_scalar': result['relchem_scalar'],
                'relchem_gradient_relative_l2': result['relchem_gradient_relative_l2'],
                'checkpoint_sha256': result['checkpoint_sha256'],
            }
            for result in results
        ],
    }
    render(results, receipt)
    print('FACTORIAL_REPORT', 'complete', flush=True)


if __name__ == '__main__':
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    execute()
