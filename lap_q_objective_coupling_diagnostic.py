"""Array-only coupling of qualified single-state q Jacobians and saved gradients."""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
DATA = ROOT.parent
OUT = DATA/'lap_q_objective_coupling_runs_20261006'
EXACT = DATA/'lap_q_exact_jacobian_runs_20261006'
OLD = DATA/'lap_q_controllability_runs_20261005'
INIT = DATA/'lap_init_landscape_runs_20261005'
SOLVER = DATA/'lap_cursor10_geometry_runs_20261005/analyze.py'
TASKS = ('full251', 'ae17', 'exc', 'op')


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b, reference_a=None, reference_b=None, tiny=1e-8):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    if reference_a is not None and na <= tiny*reference_a:
        return None
    if reference_b is not None and nb <= tiny*reference_b:
        return None
    return float(a@b/(na*nb))


def coupling(matrix, vh, singular, gradients, threshold=1e-3, tiny=1e-8):
    rank = int(np.sum(singular/singular[0] >= threshold))
    basis = vh[:rank]
    assert np.max(np.abs(basis@basis.T-np.eye(rank))) <= 1e-10
    norms = np.linalg.norm(gradients, axis=1)
    assert np.all(norms > 0) and np.isfinite(gradients).all()
    coefficients = gradients@vh.T
    projected = gradients@basis.T@basis
    remainder = gradients-projected
    induced = -(gradients/norms[:, None])@matrix.T
    records = {}
    for i, name in enumerate(TASKS):
        response = induced[i]
        truncated = -(projected[i]/norms[i])@matrix.T
        residual = -(remainder[i]/norms[i])@matrix.T
        records[name] = {'gradient_norm': float(norms[i]), 'projection': {},
                         'signed_coefficients_first8': coefficients[i, :8].tolist(),
                         'normalized_coefficients_first8': (coefficients[i, :8]/norms[i]).tolist(),
                         'induced': {'norm': float(np.linalg.norm(response)),
                                     'RMS': float(np.linalg.norm(response)/np.sqrt(128)),
                                     'median_abs': float(np.median(np.abs(response))),
                                     'alpha_norm': float(np.linalg.norm(response[:64])),
                                     'beta_norm': float(np.linalg.norm(response[64:])),
                                     'max_abs': float(np.max(np.abs(response))),
                                     'resolved_component_norm': float(np.linalg.norm(truncated)),
                                     'complement_response_norm': float(np.linalg.norm(residual))}}
        for label, requested in [('1', 1), ('2', 2), ('3', 3), ('4', 4), ('8', 8), ('full', rank)]:
            k = min(requested, rank)
            value = np.linalg.norm(coefficients[i, :k])
            records[name]['projection'][label] = {'used_k': k, 'fraction': float((value/norms[i])**2),
                                                   'absolute_norm': float(value)}
    pairs = {}
    for i, j in itertools.combinations(range(4), 2):
        denominator = norms[i]*norms[j]
        qdot, pdot = float(projected[i]@projected[j]/denominator), float(remainder[i]@remainder[j]/denominator)
        negative = max(0., -qdot)+max(0., -pdot)
        pairs[TASKS[i]+'__'+TASKS[j]] = {
            'full_cosine': cosine(gradients[i], gradients[j]),
            'q_component_cosine': cosine(projected[i], projected[j], norms[i], norms[j], tiny),
            'complement_cosine': cosine(remainder[i], remainder[j], norms[i], norms[j], tiny),
            'induced_q_cosine': cosine(induced[i], induced[j]),
            'q_dot_over_full_norms': qdot, 'complement_dot_over_full_norms': pdot,
            'q_share_of_negative_dot_contributions': max(0., -qdot)/negative if negative > 0 else None}
        assert abs(qdot+pdot-pairs[TASKS[i]+'__'+TASKS[j]]['full_cosine']) <= 1e-12
    return {'resolved_rank': rank, 'objectives': records, 'pairs': pairs}, projected, remainder


def overlap(a, b):
    values = np.linalg.svd(a@b.T, compute_uv=False)
    values = np.clip(values, 0., 1.)
    return {'canonical_correlations': values.tolist(),
            'principal_angles_degrees': np.degrees(np.arccos(values)).tolist(),
            'largest_angle_degrees': float(np.degrees(np.arccos(values)).max()),
            'mean_squared_overlap': float(np.mean(values**2))}


def margin(components, norms, tiny, expected_solver_sha):
    if any(np.linalg.norm(row) <= tiny*norm for row, norm in zip(components, norms)):
        return {'status': 'undefined/tiny component'}
    assert sha(SOLVER) == expected_solver_sha
    source = SOLVER.read_text(encoding='utf-8-sig')
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'simplex')
    namespace = {'torch': torch, 'np': np, 'itertools': itertools}
    exec(ast.get_source_segment(source, node), namespace)  # noqa: S102 -- SHA-verified existing pure solver function.
    units = torch.from_numpy(components/np.linalg.norm(components, axis=1)[:, None])
    combination, metadata = namespace['simplex'](units)
    weights = torch.tensor(metadata['coefficients'], dtype=torch.float64)
    gram = units@units.T
    values = gram@weights
    objective = float(weights@values)
    active = weights > 1e-10
    certificate = {'simplex_primal': max(abs(float(weights.sum())-1), max(0., -float(weights.min()))),
                   'active_stationarity': float((values[active]-objective).abs().max()),
                   'inactive_dual': float(torch.clamp(objective-values, min=0).max()),
                   'complementarity': float((weights*(values-objective)).abs().max())}
    assert max(certificate.values()) <= 1e-9
    gamma = float(combination.norm())
    products = (units@(combination/gamma)).tolist() if gamma > 0 else None
    return {'status': 'PASS', 'gamma': gamma, 'gradient_space_products': products,
            'descent_update_sign': 'apply -d, so predicted products are negative of these',
            'MGDA': metadata, 'certificate': certificate}


def verify(protocol):
    for path, expected in protocol['immutable_files'].items():
        assert sha(path) == expected, ('hash mismatch', path)
    assert sha(__file__) == protocol['script_sha256']
    order = protocol['parameter_order']
    assert len({row['name'] for row in order}) == len(order) == 34
    assert [row['name'] for row in order] == sorted(row['name'] for row in order)
    cursor = 0
    for row in order:
        assert row['start'] == cursor and row['stop']-row['start'] == int(np.prod(row['shape']))
        cursor = row['stop']
    assert cursor == 9446
    return len(protocol['immutable_files'])


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = load(OLD/'protocol.json')
    exact = load(ROOT/'lap_q_exact_parameter_jacobian_qualification_metrics.json')
    assert exact['independent_review']['status'] == 'PASS'
    init = load(INIT/'protocol.json')
    bindings = []
    immutable = {str(ROOT/'lap_q_exact_parameter_jacobian_qualification_metrics.json'):
                 sha(ROOT/'lap_q_exact_parameter_jacobian_qualification_metrics.json'),
                 str(OLD/'protocol.json'): sha(OLD/'protocol.json'),
                 str(EXACT/'protocol.json'): sha(EXACT/'protocol.json'),
                 str(INIT/'protocol.json'): sha(INIT/'protocol.json'), str(INIT/'matrix.py'): sha(INIT/'matrix.py')}
    for regime in ('P67', 'P536'):
        binding = next(row for row in old['gradient_bindings'] if row['seed'] == 11 and row['regime'] == regime)
        qualification = next(row for row in exact['states'] if row['state']['regime'] == regime)
        receipt = load(binding['receipt_path'])
        assert receipt['state_sha256'] == binding['state_sha256'] == qualification['state']['state_sha256']
        assert receipt['state_file_sha256'] == qualification['state']['file_sha256']
        assert receipt['array_sha256'] == binding['sha256']
        assert receipt['diagnostic_script_sha256'] == sha(INIT/'matrix.py')
        assert receipt['protocol_sha256'] == sha(INIT/'protocol.json')
        assert receipt['source_sha256'] == init['source_sha256']
        assert qualification['qualified'] and all(row['PASS'] for rows in qualification['FD'].values() for row in rows)
        bindings.append({'regime': regime, 'gradient': binding, 'jacobian': {
            'path': qualification['path'], 'sha256': qualification['sha256'], 'state': qualification['state']}})
        immutable.update({binding['path']: binding['sha256'], binding['receipt_path']: binding['receipt_sha256'],
                          qualification['path']: qualification['sha256'],
                          qualification['state']['path']: qualification['state']['file_sha256']})
    for path, digest in init['source_sha256'].items():
        immutable[str(ROOT/path)] = digest
    solver_sha = load(bindings[0]['gradient']['receipt_path'])['geometry']['MGDA']['solver_source_sha256']
    immutable[str(SOLVER)] = solver_sha
    order = load(EXACT/'protocol.json')['parameter_order']
    assert order == old['parameter_order']
    protocol = {'starting_commit': '61278273c7c167c51372ebd9dfd17028e2e38fcd', 'states': bindings,
                'immutable_files': immutable, 'parameter_order': order, 'total_coordinates': 9446,
                'gradient_ordering_evidence': 'SHA-bound matrix.py flat(): gs zipped with sorted unique named_trainable_parameters(model/shadow); no explicit embedded tensor-order manifest in legacy arrays. Exact executed source and helper are verified, not guessed.',
                'task_order': list(TASKS), 'gradient_sign': 'stored +grad L; analyzed step=-g/||g||, no parameter update',
                'scientific_precision': 'chemistry matched-F64 shadow; Exc/operator existing production precision with F32 leaves, gradients widened after autograd; not recomputed',
                'q_precision': 'Qualified F64 q Jacobian on exact widened F32 state/source values; same128 rows alpha64/beta64',
                'resolved_relative_singular_threshold': 1e-3, 'top_k': [1, 2, 3, 4, 8, 'full'],
                'tiny_component_relative_norm': 1e-8, 'optional_common_descent': True,
                'interpretation': 'Probe-resolved q subspace, not exclusive q physics; complement can contain lower-gain q modes. Primary D/E comparison is separate from within-state A/B/C mechanism.',
                'descriptive_classification_rules': {
                    'E': 'top1 abs cosine>=.95, full mean squared overlap>=.8; max fraction difference<=.10; max induced cosine difference<=.10; induced norm ratios between.5 and2',
                    'D': 'failure of E with substantial subspace rotation (top1<.95 or full overlap<.8) and coupling change (fraction>.10 or inducedcosine>.10); otherwise statewise A/B/C or mixed',
                    'A': 'chem/local full projection fractions>=.10, induced cosine<=-.8, q share of negative contributions>=.5, in eachstate',
                    'B': 'all fractions<.01 and normalized induced gain norm/sigma1<.1',
                    'C': 'chemfraction<=.01; Exc/operatorfraction>=.1; chemistrynegativeconflict qshare<.5',
                    'scope': 'Predeclared descriptive local cutoffs, not statistical significance or production thresholds.'},
                'mode_sign_comparison': 'raw within-state SVD coefficients retained; P536 corresponding-mode signs aligned to P67 using dot sign for descriptive cross-state tables only',
                'threshold_diagnostic': 'repeat array projections at fixed1e-2,1e-4 solely sensitivity; primary remains1e-3',
                'script_sha256': sha(__file__), 'no_model_evaluation': True, 'no_training': True}
    verify(protocol)
    dump(OUT/'protocol.json', protocol)
    print('FROZEN; immutable hashes', len(immutable), flush=True)


def run():
    protocol = load(OUT/'protocol.json')
    before = verify(protocol)
    results, bases, raw_arrays = {}, {}, {}
    for binding in protocol['states']:
        name = binding['regime']
        tensors = torch.load(binding['gradient']['path'], map_location='cpu', weights_only=False)['aggregates']
        assert list(tensors) == list(TASKS)
        assert all(tuple(value.shape) == (9446,) and value.dtype == torch.float64 for value in tensors.values())
        gradients = np.stack([tensors[task].numpy() for task in TASKS])
        receipt = load(binding['gradient']['receipt_path'])
        for i, task in enumerate(TASKS):
            assert abs(np.linalg.norm(gradients[i])/receipt['geometry']['norms'][task]-1) <= 1e-12
        exact = np.load(binding['jacobian']['path'])
        matrix, vh, singular = exact['J'], exact['Vh'], exact['singular']
        assert matrix.shape == (128, 9446) and vh.shape == (128, 9446)
        assert np.isfinite(matrix).all() and np.isfinite(vh).all()
        assert np.linalg.norm(matrix-(exact['U']*singular)@vh)/np.linalg.norm(matrix) <= 1e-12
        metrics, projected, remainder = coupling(matrix, vh, singular, gradients,
                                                protocol['resolved_relative_singular_threshold'],
                                                protocol['tiny_component_relative_norm'])
        metrics['identity'] = binding
        metrics['common_descent'] = {label: margin(value, np.linalg.norm(gradients, axis=1),
                         protocol['tiny_component_relative_norm'], protocol['immutable_files'][str(SOLVER)])
                                     for label, value in [('q', projected), ('complement', remainder)]}
        metrics['dominant_modes'] = [{'index': i+1, 'sigma': float(singular[i]),
             'Jacobian_energy_fraction': float(singular[i]**2/np.sum(singular**2)),
             'signed_coefficients': {task: float(gradients[j]@vh[i]) for j, task in enumerate(TASKS)}}
                                    for i in range(3)]
        metrics['threshold_sensitivity'] = {}
        for threshold in (1e-2, 1e-4):
            diagnostic, _, _ = coupling(matrix, vh, singular, gradients, threshold)
            metrics['threshold_sensitivity'][str(threshold)] = {
                'rank': diagnostic['resolved_rank'],
                'fractions': {task: value['projection']['full']['fraction'] for task, value in diagnostic['objectives'].items()},
                'q_conflict_share': {pair: value['q_share_of_negative_dot_contributions'] for pair, value in diagnostic['pairs'].items()}}
        results[name] = metrics
        bases[name] = vh
        raw_arrays[name] = gradients
    overlap_rows = {}
    ranks = [results[name]['resolved_rank'] for name in ('P67', 'P536')]
    for label, desired in [('1', 1), ('2', 2), ('3', 3), ('4', 4), ('8', 8), ('full', min(ranks))]:
        used = min(desired, min(ranks))
        overlap_rows[label] = {'used_k': used, **overlap(bases['P67'][:used], bases['P536'][:used])}
    mode_alignment = []
    for i in range(8):
        dot = float(bases['P67'][i]@bases['P536'][i])
        sign = -1 if dot < 0 else 1
        mode_alignment.append({'mode': i+1, 'raw_vector_cosine': dot, 'sign_invariant_cosine': abs(dot),
                               'P536_sign_alignment': sign,
                               'P67_unit_gradient_coefficients': (raw_arrays['P67']@bases['P67'][i]/np.linalg.norm(raw_arrays['P67'], axis=1)).tolist(),
                               'P536_aligned_unit_gradient_coefficients': (sign*raw_arrays['P536']@bases['P536'][i]/np.linalg.norm(raw_arrays['P536'], axis=1)).tolist()})
    fractions_difference = max(abs(results['P67']['objectives'][task]['projection']['full']['fraction']-
                                  results['P536']['objectives'][task]['projection']['full']['fraction']) for task in TASKS)
    cosine_difference = max(abs(results['P67']['pairs'][pair]['induced_q_cosine']-
                               results['P536']['pairs'][pair]['induced_q_cosine']) for pair in results['P67']['pairs'])
    gains = {task: results['P536']['objectives'][task]['induced']['norm']/results['P67']['objectives'][task]['induced']['norm'] for task in TASKS}
    similar = (mode_alignment[0]['sign_invariant_cosine'] >= .95
               and overlap_rows['full']['mean_squared_overlap'] >= .8
               and fractions_difference <= .10 and cosine_difference <= .10
               and all(.5 <= value <= 2 for value in gains.values()))
    rotated = mode_alignment[0]['sign_invariant_cosine'] < .95 or overlap_rows['full']['mean_squared_overlap'] < .8
    changed = fractions_difference > .10 or cosine_difference > .10
    mechanisms = {}
    for name, result in results.items():
        obj, pairs = result['objectives'], result['pairs']
        contested = [task for task in TASKS[1:] if obj['full251']['projection']['full']['fraction'] >= .1
                     and obj[task]['projection']['full']['fraction'] >= .1
                     and pairs['full251__'+task]['induced_q_cosine'] <= -.8
                     and pairs['full251__'+task]['q_share_of_negative_dot_contributions'] >= .5]
        mechanisms[name] = {'q_driven_A_supported_pairs': contested}
    primary = 'E' if similar else 'D' if rotated and changed else 'A' if all(row['q_driven_A_supported_pairs'] for row in mechanisms.values()) else 'mixed'
    after = verify(protocol)
    dump(OUT/'results.json', {'states': results, 'subspace_overlap': overlap_rows, 'mode_alignment': mode_alignment,
              'comparison': {'max_projection_fraction_difference': fractions_difference,
                             'max_induced_cosine_difference': cosine_difference, 'P536_P67_induced_gain_ratios': gains},
              'within_state_mechanism': mechanisms, 'primary_case': primary,
              'hash_before': before, 'hash_after': after, 'mismatches': 0,
              'parameter_order_compatible': True, 'state_mutation': 'NONE',
              'model_evaluations': 0, 'expensive_gradient_recomputations': 0})
    print('CASE', primary, 'overlap', overlap_rows['full']['mean_squared_overlap'],
          'fractiondiff', fractions_difference, 'responsediff', cosine_difference, flush=True)
    for name, row in results.items():
        print(name, 'fractions', {task: obj['projection']['full']['fraction'] for task, obj in row['objectives'].items()},
              'induced', {task: obj['induced']['norm'] for task, obj in row['objectives'].items()},
              'qresponsecos', {pair: value['induced_q_cosine'] for pair, value in row['pairs'].items()}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.stage]()
