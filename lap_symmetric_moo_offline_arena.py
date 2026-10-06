"""Offline symmetric direction screen; no model loading or differentiation."""
import argparse
from pathlib import Path

import numpy as np
import torch

import lap_q_objective_coupling_diagnostic as reuse
from train_models.moo_aggregators import aggregate_task_gradients

ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_symmetric_moo_offline_arena_runs_20261006'
TASKS = reuse.TASKS
METHODS = {'RAW_EQUAL_MEAN': ('fixed', {'fixed_weights': [1., 1., 1., 1.]}),
           'IMTL-G': ('imtl_g', {}),
           'CAGrad': ('cagrad', {'c': .4, 'rescale': 'paper_unscaled', 'max_iter': 500, 'ftol': 1e-12}),
           'Nash-MTL': ('nash_mtl', {'max_iter': 100, 'tol': 1e-10}),
           'PCD_CONTROL': ('pcd', {'tau': .02, 'beta': .999, 'eps': 1e-8, 'qp_tolerance': 1e-9})}


def maxmin(g, solver_sha):
    norms = np.linalg.norm(g, axis=1)
    assert np.all(norms > 0) and np.isfinite(g).all()
    units = g/norms[:, None]
    certificate = reuse.margin(units, np.ones(len(g)), 0, solver_sha)
    weights = np.array(certificate['MGDA']['coefficients'])
    direction = weights@units
    gamma = np.linalg.norm(direction)
    if gamma <= 1e-12:  # labeling only; preserve raw gamma and vector.
        return direction, {'gamma_star': float(gamma), 'stationary_label': True, **certificate}
    direction /= gamma
    assert abs(float((units@direction).min())-gamma) <= 1e-9
    return direction, {'gamma_star': float(gamma), 'stationary_label': False, **certificate}


def progress(g, direction, gamma):
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return {'status': 'degenerate', 'native_norm': float(norm)}
    unit = direction/norm
    dots = g@unit
    p = dots/np.linalg.norm(g, axis=1)
    return {'status': 'PASS', 'native_norm': float(norm), 'p': p.tolist(), 'raw_dot': dots.tolist(),
            'p_min': float(p.min()), 'p_mean': float(p.mean()), 'p_range': float(np.ptp(p)),
            'p_std': float(np.std(p)), 'common_descent': bool(np.all(p > 0)),
            'near_zero_labels': (np.abs(p) <= 1e-12).tolist(),
            'efficiency': float(p.min()/gamma) if gamma > 0 else None}


def tier(rows):
    if len(rows) != 4 or any(row['status'] != 'PASS' for row in rows):
        return 3
    if all(row['common_descent'] for row in rows):
        return 1
    return 2 if sum(row['p_min'] < 0 for row in rows) <= 1 and min(row['p_min'] for row in rows) >= -.02 else 3


def redundant(rows):
    return all(row['abs_cosine'] >= .995 and row['max_progress_difference'] <= .01 for row in rows)


def selection(summary, diversity):
    eligible = [name for name in ('IMTL-G', 'CAGrad', 'Nash-MTL') if summary[name]['tier'] <= 2]
    fallback = False
    if not eligible:
        eligible = ['UNIT_MEAN'] if summary['UNIT_MEAN']['tier'] <= 2 else []
        fallback = True
    # Exact float64 lexicographic order; no post-hoc practical-tie tolerance.
    complexity = {'IMTL-G': 0, 'Nash-MTL': 1, 'CAGrad': 2, 'UNIT_MEAN': -1}
    ranked = sorted(eligible, key=lambda n: (summary[n]['tier'], -summary[n]['worst_p_min'],
                    -summary[n]['minimum_efficiency'], summary[n]['efficiency_range'],
                    -summary[n]['median_p_min'], summary[n]['median_imbalance'], complexity[n]))
    if not ranked:
        return {'slot1': 'UNIT_MAXMIN', 'slot2': None, 'status': 'NO QUALIFIED SECOND CANDIDATE'}
    chosen = ranked[0]
    independent = [name for name in ranked if not redundant(diversity[name])]
    if redundant(diversity[chosen]) and independent:
        chosen = independent[0]
    return {'slot1': 'UNIT_MAXMIN', 'slot2': chosen, 'ranked_eligible': ranked,
            'UNIT_MEAN_fallback': fallback, 'second_redundant': redundant(diversity[chosen]), 'status': 'PASS'}


def verify(protocol):
    for path, digest in protocol['immutable_files'].items():
        assert reuse.sha(path) == digest, path
    assert reuse.sha(__file__) == protocol['script_sha256']
    return len(protocol['immutable_files'])


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = reuse.load(reuse.OLD/'protocol.json')
    prior = reuse.load(ROOT/'lap_q_objective_coupling_audit_protocol.json')
    second = reuse.load(ROOT/'lap_q_objective_coupling_seed23_replication_protocol.json')
    immutable = {**prior['immutable_files'], **second['immutable_files']}
    immutable[str(ROOT/'train_models/moo_aggregators.py')] = reuse.sha(ROOT/'train_models/moo_aggregators.py')
    bindings = []
    for seed in (11, 23):
        for regime in ('P67', 'P536'):
            state = next(row for row in old['states'] if row['seed'] == seed and row['regime'] == regime)
            gradient = next(row for row in old['gradient_bindings'] if row['seed'] == seed and row['regime'] == regime)
            receipt = reuse.load(gradient['receipt_path'])
            assert receipt['state_sha256'] == state['state_sha256'] == gradient['state_sha256']
            assert receipt['state_file_sha256'] == state['file_sha256']
            assert receipt['array_sha256'] == gradient['sha256']
            if seed == 11:
                q = next(row for row in prior['states'] if row['regime'] == regime)['jacobian']
            else:
                result = reuse.load(ROOT/'lap_q_objective_coupling_seed23_replication_metrics.json')
                q = result['qualifications'][regime]
            immutable[q['path']] = q['sha256']
            bindings.append({'key': f'{seed}_{regime}', 'state': state, 'gradient': gradient,
                             'jacobian': {'path': q['path'], 'sha256': q['sha256']}})
    protocol = {'starting_commit': '3d127128cf2168aedfe3f9419eee1c04261e0dc7', 'bindings': bindings,
                'immutable_files': immutable, 'parameter_order': old['parameter_order'],
                'task_order': list(TASKS), 'task_labels': ['chemistry', 'AE17', 'Exc', 'operator'],
                'numel': 9446, 'methods': METHODS, 'UNIT_MAXMIN': 'Existing exact all-face unit-gradient simplex solver; unit output for positive gamma; KKT<=1e-9.',
                'zero_margin_label_tolerance': 1e-12, 'near_zero_progress_label_tolerance': 1e-12,
                'selection': 'Exact lexicographic tier,worst p_min,min efficiency,efficiency range,median p_min,median imbalance; exact ties prefer IMTL-G,Nash-MTL,CAGrad. Slot1 reserved UNIT_MAXMIN; no control eligibility.',
                'tiers': 'Tier1 all raw p>0 in all4 states; Tier2 negative in at most1 state and every p_min>=-.02; otherwiseTier3. Failed/degenerate direction=Tier3.',
                'redundancy': 'All4 abs cosine>=.995 AND all4 max progress difference<=.01; skip redundant slot2 if another nonredundantTier1/2 exists.',
                'structured_reconstruction_tolerance': 1e-10, 'cold_start': 'state=None independently each method/state',
                'ordering_evidence': prior['gradient_ordering_evidence'],
                'no_model_evaluation': True, 'no_training': True, 'no_cross_seed_vector_cosines': True,
                'optional_q': 'Existing SHA-bound exact matrices only; never used for selection.',
                'script_sha256': reuse.sha(__file__)}
    verify(protocol)
    reuse.dump(OUT/'protocol.json', protocol)
    print('FROZEN', len(immutable), 'hashes', flush=True)


def run():
    protocol = reuse.load(OUT/'protocol.json')
    before = verify(protocol)
    states, vectors = {}, {}
    for binding in protocol['bindings']:
        tensors = torch.load(binding['gradient']['path'], map_location='cpu', weights_only=False)['aggregates']
        assert list(tensors) == list(TASKS)
        assert all(v.dtype == torch.float64 and tuple(v.shape) == (9446,) for v in tensors.values())
        g = np.stack([tensors[t].numpy() for t in TASKS])
        norms = np.linalg.norm(g, axis=1)
        receipt = reuse.load(binding['gradient']['receipt_path'])
        assert max(abs(norms[i]/receipt['geometry']['norms'][t]-1) for i, t in enumerate(TASKS)) <= 1e-12
        units = g/norms[:, None]
        structured = {t: {row['name']: tensors[t][row['start']:row['stop']].reshape(row['shape'])
                         for row in protocol['parameter_order']} for t in TASKS}
        direction, certificate = maxmin(g, protocol['immutable_files'][str(reuse.SOLVER)])
        gamma = certificate['gamma_star']
        assert gamma > 1e-12 and not certificate['stationary_label'], 'STOP: UNIT_MAXMIN not qualified'
        candidate_vectors = {'UNIT_MAXMIN': direction, 'UNIT_MEAN': units.mean(axis=0)}
        rows = {'UNIT_MAXMIN': {**progress(g, direction, gamma), 'diagnostics': certificate},
                'UNIT_MEAN': progress(g, units.mean(axis=0), gamma)}
        for label, (method, hparams) in METHODS.items():
            try:
                joint, diagnostics, state = aggregate_task_gradients(structured, method=method,
                       hyperparameters=hparams, state=None, task_order=TASKS)
                direction = torch.cat([joint[row['name']].flatten() for row in protocol['parameter_order']]).numpy()
                candidate_vectors[label] = direction
                row = progress(g, direction, gamma)
                coefficients = np.array([diagnostics['coefficients'][t] for t in TASKS])
                fractions = np.abs(coefficients)/np.sum(np.abs(coefficients))
                positive = fractions[fractions > 0]
                row.update({'diagnostics': diagnostics, 'cold_start_returned_state': state,
                            'negative_coefficient_tasks': [t for t, c in zip(TASKS, coefficients) if c < 0],
                            'absolute_coefficient_fractions': fractions.tolist(),
                            'coefficient_entropy': float(-np.sum(positive*np.log(positive))),
                            'coefficient_concentration': float(fractions.max()),
                            'coefficient_times_gradient_norm_fractions': (np.abs(coefficients)*norms/np.sum(np.abs(coefficients)*norms)).tolist()})
                # Independent direct reconstruction of structured diagnostics.
                dots = g@direction
                reported = np.array([diagnostics['directional_task_dots'][t] for t in TASKS])
                assert np.max(np.abs(dots-reported)/np.maximum(np.abs(dots), 1.)) <= 1e-10
                assert np.max(np.abs(np.array(diagnostics['gram'])-g@g.T)/np.maximum(np.abs(g@g.T), 1.)) <= 1e-10
                row['direct_structured_validation'] = 'PASS'
                if method == 'cagrad':
                    extra = coefficients-.25
                    row['simplex_weights'] = (extra/extra.sum()).tolist()
                rows[label] = row
            except (RuntimeError, ValueError, FloatingPointError) as error:
                rows[label] = {'status': 'FAILED', 'error': str(error)}
        q = np.load(binding['jacobian']['path'])
        chem_responses = -(g/norms[:, None])@q['J'].T
        for name, direction in candidate_vectors.items():
            if np.linalg.norm(direction) == 0:
                continue
            unit = direction/np.linalg.norm(direction)
            response = -q['J']@unit
            rows[name]['q_diagnostic'] = {'norm': float(np.linalg.norm(response)),
                'task_response_cosines': {t: reuse.cosine(response, chem_responses[i]) for i, t in enumerate(TASKS)},
                'dominant_mode_step_coefficient': float(-q['Vh'][0]@unit),
                'mode_sign': 'as stored in SHA-bound exact SVD; arbitrary sign, not compared across states'}
        key = binding['key']
        states[key] = {'identity': binding, 'norms': norms.tolist(), 'gram': (g@g.T).tolist(),
                       'gamma_star': gamma, 'candidates': rows}
        vectors[key] = candidate_vectors
        print(key, {n: row.get('p_min', row['status']) for n, row in rows.items()}, flush=True)
    summary, diversity = {}, {}
    for name in ('UNIT_MAXMIN', 'UNIT_MEAN', *METHODS):
        rows = [s['candidates'][name] for s in states.values()]
        valid = [row for row in rows if row['status'] == 'PASS']
        efficiencies = [row['efficiency'] for row in valid]
        summary[name] = {'tier': tier(rows), 'common_descent_states': sum(row.get('common_descent', False) for row in rows),
            'failures': sum(row['status'] != 'PASS' for row in rows),
            'worst_p_min': min([row['p_min'] for row in valid], default=None),
            'median_p_min': float(np.median([row['p_min'] for row in valid])) if valid else None,
            'minimum_efficiency': min(efficiencies, default=None),
            'median_efficiency': float(np.median(efficiencies)) if valid else None,
            'efficiency_range': float(np.ptp(efficiencies)) if valid else None,
            'median_imbalance': float(np.median([row['p_range'] for row in valid])) if valid else None}
        diversity[name] = []
        for key, s in states.items():
            if name in vectors[key] and s['candidates'][name]['status'] == 'PASS':
                diversity[name].append({'state': key, 'abs_cosine': abs(reuse.cosine(vectors[key][name], vectors[key]['UNIT_MAXMIN'])),
                    'max_progress_difference': float(np.max(np.abs(np.array(s['candidates'][name]['p'])-s['candidates']['UNIT_MAXMIN']['p'])))})
    assert summary['UNIT_MAXMIN']['tier'] == 1
    selected = selection(summary, diversity)
    paired = {}
    for seed in (11, 23):
        paired[str(seed)] = {}
        for name in summary:
            a, b = states[f'{seed}_P67']['candidates'][name], states[f'{seed}_P536']['candidates'][name]
            if a['status'] == b['status'] == 'PASS':
                paired[str(seed)][name] = {k: b[k]-a[k] for k in ('p_min', 'efficiency', 'p_range')}
    reuse.dump(OUT/'results.json', {'states': states, 'summary': summary, 'diversity': diversity,
               'selection': selected, 'paired_changes': paired, 'hash_before': before,
               'hash_after': verify(protocol), 'mismatches': 0, 'no_state_mutation': True,
               'protocol_sha256': reuse.sha(OUT/'protocol.json')})
    print('SELECTION', selected, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
