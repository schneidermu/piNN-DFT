"""Frozen read-only DB exactification and paired-state SVRG diagnostics."""
import argparse
import copy
import itertools
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import lap_chemistry_estimator_fidelity_arena as old

ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_chemistry_db_variance_svrg_runs_20261007'
sha, load, dump = old.sha, old.load, old.dump
START = '49af98facb2a055cea25fff3456ba6e7a9872e7f'


def stats(values):
    return dict(zip(('min', 'p05', 'median'), np.quantile(values, [0, .05, .5]).tolist()))


def cv_gradient(gold_ref, current, reference, batch):
    # Same ordered IDs and weights in BOTH terms; no separate reference draw.
    ids = np.arange(len(current))
    return gold_ref+old.estimate(current, ids, batch)-old.estimate(reference, ids, batch)


def db_parts(gradients, manifest, batch):
    result = {}
    for db in manifest['database_order']:
        chosen = [(i, w) for i, w in batch if manifest['catalog'][i]['database'] == db]
        result[db] = sum((w*gradients[i] for i, w in chosen), np.zeros(gradients.shape[1]))
    return result


def qualify(rows):
    cases = {key: [r for r in rows if r['case'] == key] for key in dict.fromkeys(r['case'] for r in rows)}
    assert len(rows) == 128 and len(cases) == 8 and all(len(v) == 16 for v in cases.values())
    rates = {key: sum(r['p_full'] is not None and r['p_full'] > 0 for r in group)/16 for key, group in cases.items()}
    progress = [r['p_full'] for r in rows if r['p_full'] is not None]
    count = sum(r['p_full'] is not None and r['p_full'] > 0 for r in rows)
    good = (count/len(rows) >= .95 and min(rates.values()) >= .9 and len(progress) == len(rows)
            and min(progress) >= old.TAIL_FLOOR and all(r['solver_qualified'] for r in rows))
    return {'observations': len(rows), 'descent_count': count, 'descent_fraction': count/len(rows),
            'per_case_success': rates, 'worst_case_success': min(rates.values()),
            'cosine': stats([r['cosine'] for r in rows]), 'p_full': stats(progress) if progress else None,
            'median_gamma': float(np.median([r['gamma'] for r in rows])),
            'catastrophic_tail_count': sum(v < old.TAIL_FLOOR for v in progress),
            'solver_qualified': all(r['solver_qualified'] for r in rows), 'qualifies': bool(good)}


def cost(k, refresh):
    # Conservative deployable budget: charge both sampled terms for all ten updates.
    refs = 2 if refresh else 1
    paired = 10*2*8*k
    return {'reference_refreshes': refs, 'reference_backwards': refs*251,
            'paired_sample_backwards_per_update': 16*k, 'paired_sample_backwards_10': paired,
            'total_backwards_10': refs*251+paired, 'ratio_to_current_K1': (refs*251+paired)/80,
            'ratio_to_K4': (refs*251+paired)/320, 'ratio_to_FULL251_each_update': (refs*251+paired)/2510,
            'startup_identity_optimized_backwards_10': refs*251+(10-refs)*16*k}


def verify(p):
    for path, expected in p['immutable_files'].items():
        assert sha(path) == expected, path
    assert sha(__file__) == p['script_sha256']
    assert sha(OUT/'replicate_manifest.json') == p['replicate_manifest_sha256']
    return len(p['immutable_files'])+2


def freeze():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == START
    assert not (OUT/'protocol.json').exists()
    OUT.mkdir(exist_ok=True)
    previous = load(old.OUT/'protocol.json')
    old.verify(previous)
    metrics = load(ROOT/'lap_chemistry_estimator_fidelity_arena_metrics.json')
    immutable = dict(previous['immutable_files'])
    for path in (old.OUT/'protocol.json', old.OUT/'replicate_manifest.json', ROOT/'lap_chemistry_estimator_fidelity_arena.py',
                 ROOT/'lap_chemistry_estimator_fidelity_arena_metrics.json'):
        immutable[str(path)] = sha(path)
    for row in metrics['states']:
        assert sha(row['cache_path']) == row['cache_sha256']
        immutable[row['cache_path']] = row['cache_sha256']
        receipt = old.OUT/(row['key']+'_result.json')
        immutable[str(receipt)] = sha(receipt)
    cases = []
    for start in ('11_P67', '11_P536', '23_P67', '23_P536'):
        for target, ref, kind in ((5, 0, 'u5'), (9, 0, 'u9_static'), (9, 5, 'u9_refresh')):
            cases.append({'case': start+'_'+kind, 'target': start+f'_u{target}', 'reference': start+f'_u{ref}', 'kind': kind})
    (OUT/'replicate_manifest.json').write_bytes((old.OUT/'replicate_manifest.json').read_bytes())
    p = {'starting_commit': START, 'script_sha256': sha(__file__), 'immutable_files': immutable,
         'bindings': metrics['states'], 'parameter_order': previous['parameter_order'], 'inputs': previous['inputs'],
         'precision': previous['precision'], 'task_order': previous['task_order'], 'cases': cases,
         'replicate_manifest_sha256': sha(OUT/'replicate_manifest.json'),
         'sampling': 'Reuse exact prior 16 permutations/seeds; IDs, canonical variants and n_d/(251 K) weights identical across target/reference.',
         'DB_analysis_K': [1, 2, 4], 'SVRG_K': [1, 2],
         'candidate_statistics': 'Static uses u5 and u9_static; refresh-5 uses u5 and u9_refresh: 8 cases/128 rows each. u5 shared. 12 distinct cases/192 unique rows per K, no theta=reference observations.',
         'gate': previous['qualification'], 'full_reconstruction_relative_L2_max': 1e-11,
         'DB_ranking': 'Lexicographic: larger pooled success-probability uplift across all 12 states/K1,K2,K4; larger catastrophic-tail count reduction; larger median paired p_full change; alphabetical tie. Best pair is top two; no pair search.',
         'variance': 'Population within-DB dispersion mean ||g_j-mu_d||^2; exact finite-population sample MSE w_d^2*(n-K)/(K*(n-1))*dispersion. Signed cross-error attribution e_d dot e_total/||e_total||^2; negative values retained.',
         'cost': 'Full reference cost251 each; static one refresh, refresh-5 two. Ten paired sample terms charged at16K per update even when update0 could reuse gold. Current K1=80, K4=320, full251 eachupdate=2510 backwards/10. No cached terms counted free.',
         'selection': 'Cheapest qualifying total10 backward count, then K, then static. Gate identical to prior; no post-hoc adjustment.',
         'scope': 'No model update, no optimizer/controller change, no training; only missing singleton chemistry gradients and cached-vector analysis.'}
    dump(OUT/'protocol.json', p)
    for name, target in [('protocol.json', 'lap_chemistry_db_variance_svrg_audit_protocol.json'),
                         ('replicate_manifest.json', 'lap_chemistry_db_variance_svrg_audit_replicate_manifest.json')]:
        (ROOT/target).write_bytes((OUT/name).read_bytes())
    print('FROZEN', verify(p), 'hashes', flush=True)


def cache():
    p, m = load(OUT/'protocol.json'), load(OUT/'replicate_manifest.json')
    verify(p)
    store = old.prior.arena.MinnesotaGroupStore(p['inputs']['store'], cache_groups=1)
    disp = old.prior.arena.load_reaction_dispersions(p['inputs']['disp'])
    for binding in p['bindings']:
        receipt = OUT/(binding['key']+'_complete_cache.json')
        if receipt.exists():
            saved = load(receipt)
            assert saved['protocol_sha256'] == sha(OUT/'protocol.json')
            assert sha(saved['path']) == saved['sha256']
            continue
        with np.load(binding['cache_path']) as source:
            existing = source['canonical_indices'].tolist()
            gradients = np.empty((251, 9446), dtype=np.float64)
            timings = np.empty(251)
            gradients[existing] = source['gradients']
            timings[existing] = source['timings']
        missing = sorted(set(range(251))-set(existing))
        model = old.prior.arena._pilot_model(torch.device('cuda'), torch.float32)
        model.load_state_dict(torch.load(binding['state_path'], map_location='cpu', weights_only=False)['model'], strict=True)
        assert old.prior.arena.digest(model) == binding['state_sha256']
        params = old.prior.arena.named_trainable_parameters(model)
        assert [(n, list(v.shape), v.numel()) for n, v in params.items()] == [(r['name'], r['shape'], r['numel']) for r in p['parameter_order']]
        shadow, rng = copy.deepcopy(model).double(), old.prior.arena.capture_rng_state()
        print('BEGIN', binding['key'], len(missing), 'missing singletons', flush=True)
        for offset, i in enumerate(missing):
            begin = time.perf_counter()
            row = m['catalog'][i]
            reaction = store.load_variant((row['database'], row['reaction_id']), row['variant_suffix'])
            objective = old.prior.ChemistryBatchObjective(model, shadow, (reaction,), (1.,), disp)
            value, gradient = objective.value_and_grad()
            gradients[i] = np.concatenate([gradient[n].detach().double().cpu().numpy().reshape(-1) for n in params])
            timings[i] = time.perf_counter()-begin
            assert np.isfinite(gradients[i]).all() and np.isfinite(value)
            assert old.prior.arena.digest(model) == binding['state_sha256']
            assert all(v.grad is None for v in model.parameters())
            del reaction, objective, gradient
            if (offset+1) % 16 == 0 or offset+1 == len(missing):
                print('CACHE', binding['key'], offset+1, '/', len(missing), flush=True)
        assert old.prior.arena.equal(rng, old.prior.arena.capture_rng_state())
        gold = np.load(binding['array_path'])['g_full251']
        error = float(np.linalg.norm(gradients.mean(axis=0)-gold)/np.linalg.norm(gold))
        assert error <= p['full_reconstruction_relative_L2_max'], error
        with np.load(binding['cache_path']) as source:
            np.testing.assert_array_equal(gradients[existing], source['gradients'])
        path = OUT/(binding['key']+'_full_singletons.npz')
        np.savez_compressed(path, gradients=gradients, canonical_indices=np.arange(251), timings=timings)
        dump(receipt, {'key': binding['key'], 'path': str(path), 'sha256': sha(path),
                      'state_sha256': binding['state_sha256'], 'source_cache_sha256': binding['cache_sha256'],
                      'protocol_sha256': sha(OUT/'protocol.json'), 'reused': len(existing), 'computed': len(missing),
                      'full_gradient_reconstruction_relative_L2': error, 'model_state_unchanged': True, 'RNG_unchanged': True})
        del model, shadow, gradients
        torch.cuda.empty_cache()
        verify(p)
        print('COMPLETE', binding['key'], 'gold reconstruction', error, flush=True)


def solve(chemistry, raw, gold, solver):
    tasks = np.vstack((chemistry, raw[1:]))
    d, meta = old.prior.arena.offline.maxmin(tasks, solver)
    p = tasks@d/np.linalg.norm(tasks, axis=1)
    truth = np.vstack((gold, raw[1:]))@d/np.linalg.norm(np.vstack((gold, raw[1:])), axis=1)
    qualified = (meta['gamma_star'] > 1e-12 and np.all(p > 0) and abs(np.linalg.norm(d)-1) <= 1e-12
                 and max(meta['certificate'].values()) <= 1e-9)
    norm, fullnorm = np.linalg.norm(chemistry), np.linalg.norm(gold)
    cosine = float(chemistry@gold/(norm*fullnorm))
    return d, {'gamma': meta['gamma_star'], 'p_full': float(truth[0]) if qualified else None,
               'estimated_progress': p.tolist(), 'true_progress': truth.tolist(),
               'strict_common_descent': bool(np.all(p > 0)), 'solver_qualified': bool(qualified),
               'KKT': meta['certificate'], 'coefficients': meta['MGDA']['coefficients'],
               'cosine': cosine, 'norm_ratio': float(norm/fullnorm),
               'angular_error': float(np.linalg.norm(chemistry/norm-gold/fullnorm))}


def impact(rows):
    delta = [r['exact_p_full']-r['base_p_full'] for r in rows]
    return {'observations': len(rows), 'base_descent_fraction': float(np.mean([r['base_p_full'] > 0 for r in rows])),
            'exact_descent_fraction': float(np.mean([r['exact_p_full'] > 0 for r in rows])),
            'descent_probability_uplift': float(np.mean([(r['exact_p_full'] > 0)-(r['base_p_full'] > 0) for r in rows])),
            'p_full_change': stats(delta), 'base_p_full': stats([r['base_p_full'] for r in rows]),
            'exact_p_full': stats([r['exact_p_full'] for r in rows]),
            'catastrophic_tail_reduction': sum(r['base_p_full'] < old.TAIL_FLOOR for r in rows)-sum(r['exact_p_full'] < old.TAIL_FLOOR for r in rows)}


def analyze():
    p, m = load(OUT/'protocol.json'), load(OUT/'replicate_manifest.json')
    hashes = verify(p)
    data, receipts = {}, []
    solver = p['immutable_files'][str(old.prior.arena.offline.reuse.SOLVER)]
    for b in p['bindings']:
        r = load(OUT/(b['key']+'_complete_cache.json'))
        assert r['state_sha256'] == b['state_sha256'] and r['protocol_sha256'] == sha(OUT/'protocol.json')
        assert sha(r['path']) == r['sha256']
        receipts.append(r)
        with np.load(r['path']) as cache_data, np.load(b['array_path']) as arrays:
            data[b['key']] = {'g': cache_data['gradients'].copy(), 'raw': arrays['raw_original'].copy(), 'gold': arrays['g_full251'].copy()}
    db_rows, variances, bases = [], [], {}
    exact = {}
    for key, state in data.items():
        g, raw, gold = state['g'], state['raw'], state['gold']
        exact[key] = {}
        for db in m['database_order']:
            indices = [i for i, row in enumerate(m['catalog']) if row['database'] == db]
            vectors, n = g[indices], len(indices)
            mean = vectors.mean(axis=0)
            exact[key][db] = n/251*mean
            dispersion = float(np.mean(np.sum((vectors-mean)**2, axis=1)))
            variances.append({'key': key, 'database': db, 'n': n, 'mean_norm': float(np.linalg.norm(mean)),
                              'weighted_mean_norm': float(n/251*np.linalg.norm(mean)), 'dispersion': dispersion,
                              'finite_population_weighted_MSE': {str(k): (n/251)**2*(n-k)/(k*(n-1))*dispersion for k in (1, 2, 4)}})
        for k, rep in itertools.product((1, 2, 4), range(16)):
            batch = old.batch_rows(m, rep, k)
            parts = db_parts(g, m, batch)
            estimate = old.estimate(g, np.arange(251), batch)
            dbase, base = solve(estimate, raw, gold, solver)
            assert base['solver_qualified']
            bases[(key, k, rep)] = (estimate, parts, dbase, base)
            total_error = estimate-gold
            for db in m['database_order']:
                error = parts[db]-exact[key][db]
                _, result = solve(estimate-error, raw, gold, solver)
                assert result['solver_qualified']
                cosine = float(parts[db]@exact[key][db]/(np.linalg.norm(parts[db])*np.linalg.norm(exact[key][db])))
                db_rows.append({'key': key, 'K': k, 'replicate': rep, 'database': db,
                                'base_p_full': base['p_full'], 'exact_p_full': result['p_full'],
                                'weighted_contribution_squared_error': float(error@error), 'cosine_to_DB_mean': cosine,
                                'signed_total_error_fraction': float(error@total_error/(total_error@total_error)),
                                'marginal_backwards': m['counts'][db]-k, 'result': result})
        print('DB_ANALYZED', key, flush=True)
    for record in variances:
        selected_rows = [r for r in db_rows if r['key'] == record['key'] and r['database'] == record['database']]
        record['empirical_by_K'] = {}
        for k in (1, 2, 4):
            selected = [r for r in selected_rows if r['K'] == k]
            record['empirical_by_K'][str(k)] = {
                'weighted_contribution_MSE': float(np.mean([r['weighted_contribution_squared_error'] for r in selected])),
                'cosine_to_DB_mean': stats([r['cosine_to_DB_mean'] for r in selected]),
                'cosine_standard_deviation': float(np.std([r['cosine_to_DB_mean'] for r in selected])),
                'signed_total_error_fraction': stats([r['signed_total_error_fraction'] for r in selected])}
    ranking = []
    for db in m['database_order']:
        rows = [r for r in db_rows if r['database'] == db]
        ranking.append({'database': db, **impact(rows), 'by_K': {str(k): impact([r for r in rows if r['K'] == k]) for k in (1, 2, 4)},
                        'by_state': {key: impact([r for r in rows if r['key'] == key]) for key in data},
                        'cosine_to_DB_mean': stats([r['cosine_to_DB_mean'] for r in rows]),
                        'mean_weighted_contribution_MSE': float(np.mean([r['weighted_contribution_squared_error'] for r in rows])),
                        'signed_error_fraction': stats([r['signed_total_error_fraction'] for r in rows]),
                        'marginal_backwards_by_K': {str(k): m['counts'][db]-k for k in (1, 2, 4)}})
    ranking.sort(key=lambda r: (-r['descent_probability_uplift'], -r['catastrophic_tail_reduction'], -r['p_full_change']['median'], r['database']))
    pair = [r['database'] for r in ranking[:2]]
    pair_rows = []
    for (key, k, rep), (estimate, parts, _, base) in bases.items():
        replaced = estimate+sum((exact[key][db]-parts[db] for db in pair), np.zeros(9446))
        _, result = solve(replaced, data[key]['raw'], data[key]['gold'], solver)
        assert result['solver_qualified']
        pair_rows.append({'key': key, 'K': k, 'replicate': rep, 'base_p_full': base['p_full'], 'exact_p_full': result['p_full'], 'result': result})
    svrg = []
    for case in p['cases']:
        target, reference = data[case['target']], data[case['reference']]
        full_direction, _ = solve(target['gold'], target['raw'], target['gold'], solver)
        for k, rep in itertools.product((1, 2), range(16)):
            batch = old.batch_rows(m, rep, k)
            gcv = cv_gradient(reference['gold'], target['g'], reference['g'], batch)
            direction, result = solve(gcv, target['raw'], target['gold'], solver)
            ordinary = bases[(case['target'], k, rep)][2]
            svrg.append({**case, 'K': k, 'replicate': rep, **result,
                         'direction_cosine_fullchem': float(direction@full_direction),
                         'direction_cosine_ordinary': float(direction@ordinary)})
    candidates = []
    for k, refresh in itertools.product((1, 2), (False, True)):
        kinds = ('u5', 'u9_refresh' if refresh else 'u9_static')
        rows = [r for r in svrg if r['K'] == k and r['kind'] in kinds]
        candidates.append({'name': f'SVRG K{k} '+('refresh-5' if refresh else 'static'), 'K': k, 'refresh': refresh,
                           **qualify(rows), 'cost': cost(k, refresh)})
    passing = sorted([c for c in candidates if c['qualifies']], key=lambda c: (c['cost']['total_backwards_10'], c['K'], c['refresh']))
    selected = passing[0]['name'] if passing else 'NO SVRG K<=2 QUALIFIES'
    classification = ('CASE D' if not passing else 'CASE A' if passing[0]['K'] == 1 and not passing[0]['refresh']
                      else 'CASE B' if passing[0]['refresh'] and not any(c['qualifies'] and not c['refresh'] for c in candidates)
                      else 'CASE C')
    dump(OUT/'analysis_rows.json', {'database_counterfactuals': db_rows, 'variances': variances,
                                 'pair_counterfactuals': pair_rows, 'SVRG': svrg})
    metrics = {'starting_commit': START, 'protocol_sha256': sha(OUT/'protocol.json'),
               'manifest_sha256': sha(OUT/'replicate_manifest.json'), 'state_cache_receipts': receipts,
               'ranking': ranking, 'DB_variances': variances,
               'best_pair': {'databases': pair, **impact(pair_rows),
                             'by_K': {str(k): impact([r for r in pair_rows if r['K'] == k]) for k in (1, 2, 4)},
                             'marginal_backwards_by_K': {str(k): sum(m['counts'][db]-k for db in pair) for k in (1, 2, 4)}},
               'SVRG_candidates': candidates, 'selected': selected, 'classification': classification,
               'unique_SVRG_rows_per_K': 192, 'candidate_rows': 128,
               'details': {'path': str(OUT/'analysis_rows.json'), 'sha256': sha(OUT/'analysis_rows.json')},
               'hashes_checked': hashes+len(receipts), 'hash_mismatches': 0, 'state_mutation': 'NONE',
               'comparison_costs_backwards_10': {'current_K1': 80, 'K4': 320, 'FULL251_each_update': 2510}}
    dump(OUT/'metrics.json', metrics)
    (ROOT/'lap_chemistry_db_variance_svrg_audit_metrics.json').write_bytes((OUT/'metrics.json').read_bytes())
    print('ANALYSIS_COMPLETE', selected, classification, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'cache', 'analyze'))
    stage = parser.parse_args().stage
    {'freeze': freeze, 'cache': cache, 'analyze': analyze}[stage]()
