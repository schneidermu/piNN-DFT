"""Frozen stratified chemistry-gradient screen; no training or model update."""
import argparse
import copy
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import lap_fullchem_surrogate_fidelity_audit as prior

ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_chemistry_estimator_fidelity_runs_20261006'
KS = (1, 2, 4, 8)
SEEDS = tuple(range(41000, 41016))
TAIL_FLOOR = -.02  # Frozen normalized-progress safety boundary from offline Arena Tier2.
sha = prior.arena.offline.reuse.sha
load = prior.arena.offline.reuse.load
dump = prior.arena.offline.reuse.dump


def draw_manifest(catalog, census=False):
    databases = sorted({r['database'] for r in catalog})
    pools = {db: [i for i, row in enumerate(catalog) if row['database'] == db] for db in databases}
    replicates = []
    for replicate, seed in enumerate(SEEDS):
        selections = {}
        for db_index, db in enumerate(databases):
            rng = np.random.default_rng(np.random.SeedSequence([seed, db_index]))
            selections[db] = rng.permutation(pools[db])[:8].tolist()
        replicates.append({'replicate': replicate, 'seed': seed, 'database_permutation_prefixes': selections})
    return {'schema': 'lap-chemistry-stratified-replicates-v1', 'catalog': catalog,
            'database_order': databases, 'counts': {db: len(pools[db]) for db in databases},
            'K': list(KS), 'replicates': replicates, 'small_database_policy': 'census' if census else 'strict',
            'sampling': 'Uniform random permutation without replacement per DB; nested prefixes across K; identical draws across all states.'}


def batch_rows(manifest, replicate, k):
    rows = []
    for db in manifest['database_order']:
        n = manifest['counts'][db]
        if n < k and manifest['small_database_policy'] == 'strict':
            return None
        size = min(k, n)
        indices = manifest['replicates'][replicate]['database_permutation_prefixes'][db][:size]
        assert len(indices) == size and len(set(indices)) == size
        rows.extend((i, n/251/size) for i in indices)
    assert abs(sum(w for _, w in rows)-1) <= 1e-14
    return rows


def estimate(singletons, indices, batch):
    offsets = {index: offset for offset, index in enumerate(indices)}
    result = np.zeros(singletons.shape[1], dtype=np.float64)
    for index, weight in batch:
        result += weight*singletons[offsets[index]]
    return result


def summarize(rows):
    table = []
    for k in KS:
        selected = [row for row in rows if row['K'] == k]
        if not selected:
            table.append({'K': k, 'status': 'INFEASIBLE', 'qualifies': False})
            continue
        assert len(selected) == 192
        states = {key: [r for r in selected if r['key'] == key] for key in dict.fromkeys(r['key'] for r in selected)}
        rates = {key: sum(r['fullchem_descent'] for r in group)/16 for key, group in states.items()}
        successes = sum(r['fullchem_descent'] for r in selected)
        p = [r['p_full'] for r in selected if r['p_full'] is not None]
        cos = [r['cosine'] for r in selected]
        qualified = (successes/192 >= .95 and min(rates.values()) >= .90
                     and len(p) == 192 and min(p) >= TAIL_FLOOR and all(r['solver_qualified'] for r in selected))
        table.append({'K': k, 'status': 'EVALUATED', 'replicates': 192, 'descent_count': successes,
            'fullchem_descent_fraction': successes/192, 'per_state_descent_fraction': rates,
            'cosine': dict(zip(('min', 'p05', 'median'), np.quantile(cos, [0, .05, .5]).tolist())),
            'p_full': dict(zip(('min', 'p05', 'median'), np.quantile(p, [0, .05, .5]).tolist())) if p else None,
            'median_gamma': float(np.median([r['gamma'] for r in selected])),
            'unit_maxmin_solver_qualified': all(r['solver_qualified'] for r in selected),
            'catastrophic_negative_tail_count': sum(r['p_full'] is not None and r['p_full'] < TAIL_FLOOR for r in selected),
            'reaction_evaluations_per_replicate': selected[0]['reaction_evaluations'],
            'evaluation_count_ratio_to_K1': selected[0]['reaction_evaluations']/8,
            'evaluation_count_ratio_to_FULL251': selected[0]['reaction_evaluations']/251,
            'qualifies': bool(qualified),
            'failures': [{'key': r['key'], 'replicate': r['replicate'], 'p_full': r['p_full'],
                          'solver_qualified': r['solver_qualified']} for r in selected if not r['fullchem_descent']]})
    return table


def verify(protocol):
    for path, expected in protocol['immutable_files'].items():
        assert sha(path) == expected, path
    assert sha(__file__) == protocol['script_sha256']
    assert sha(OUT/'replicate_manifest.json') == protocol['replicate_manifest_sha256']
    return len(protocol['immutable_files'])+2


def freeze(census):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == 'bc3aa4f5e684e0e79b4c71288ce0b751114322e0'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = load(prior.OUT/'protocol.json')
    prior.verify(old)
    metrics_path = ROOT/'lap_fullchem_surrogate_fidelity_audit_metrics.json'
    metrics = load(metrics_path)
    git_bytes = subprocess.check_output(['git', 'show', 'HEAD:'+metrics_path.name], cwd=ROOT)
    assert git_bytes == metrics_path.read_bytes()
    entries = [e['per_rank'][0] for e in prior.arena.read_sampling_manifest(prior.arena.FULL)['entries'] if e['per_rank'][0]['reaction']['database'] != 'AE17']
    assert len(entries) == 251
    catalog = [{**row['reaction'], 'variant_suffix': row['variant_suffix']} for row in entries]
    assert len({(r['database'], r['reaction_id']) for r in catalog}) == 251
    manifest = draw_manifest(catalog, census)
    dump(OUT/'replicate_manifest.json', manifest)
    immutable = dict(old['immutable_files'])
    immutable[str(prior.OUT/'protocol.json')] = sha(prior.OUT/'protocol.json')
    immutable[str(metrics_path)] = sha(metrics_path)
    bindings = []
    original_manifest = prior.arena.read_sampling_manifest(prior.arena.MANIFEST)
    for row in metrics['states']:
        assert sha(row['array_path']) == row['array_sha256']
        immutable[row['array_path']] = row['array_sha256']
        samples = original_manifest['entries'][row['update']]['per_rank'][0]['task_samples']['relchem']
        original_batch = []
        for sample in samples:
            index = next(i for i, c in enumerate(catalog) if all(c[f] == sample[f] for f in ('database', 'reaction_id', 'variant_suffix')))
            original_batch.append([index, sample['weight']])
        bindings.append({name: row[name] for name in ('key', 'start', 'update', 'state_path', 'state_sha256',
            'state_file_sha256', 'array_path', 'array_sha256', 'geometry_path', 'geometry_sha256')} | {'original_batch': original_batch})
    assert len(bindings) == 12
    protocol = {'starting_commit': 'bc3aa4f5e684e0e79b4c71288ce0b751114322e0',
        'bindings': bindings, 'parameter_order': old['parameter_order'], 'inputs': old['inputs'],
        'immutable_files': immutable, 'script_sha256': sha(__file__),
        'replicate_manifest_sha256': sha(OUT/'replicate_manifest.json'), 'K': list(KS), 'replicates': 16,
        'task_order': list(prior.arena.ORDER), 'precision': old['precision'],
        'estimator': 'Sum over DBs n_d/251 times uniform without-replacement sample mean of corrected singleton batch_fchem gradients; exact canonical variants.',
        'full_reference': 'Reuse SHA-bound full251 gradients; never used to select samples, weights or directions.',
        'qualification': {'overall_success_min': .95, 'each_state_success_min': .90,
            'catastrophic_negative_tail_floor': TAIL_FLOOR, 'tail_rule': 'No normalized true fullchem progress below -0.02; inherited offline Arena Tier2 safety boundary, frozen before evaluation.',
            'solver_KKT_max': 1e-9, 'common_descent_gamma_min': 1e-12},
        'original_gradient_reconstruction_relative_L2_max': 1e-11,
        'cost': 'Exact singleton backward counts plus summed measured singleton timings. FULL251 runtime only projected from per-DB mean singleton timings, not remeasured; no new full251 gradient.',
        'small_database_policy': manifest['small_database_policy'],
        'scope': 'No training, no model perturbation, no new objective/method, no production-source changes; all failures retained.'}
    dump(OUT/'protocol.json', protocol)
    for src, dst in [('protocol.json', 'lap_chemistry_estimator_fidelity_arena_protocol.json'),
                     ('replicate_manifest.json', 'lap_chemistry_estimator_fidelity_arena_replicate_manifest.json')]:
        (ROOT/dst).write_bytes((OUT/src).read_bytes())
    print('FROZEN', protocol['replicate_manifest_sha256'], Counter(c['database'] for c in catalog), flush=True)


def run():
    p, manifest = load(OUT/'protocol.json'), load(OUT/'replicate_manifest.json')
    verify(p)
    store = prior.arena.MinnesotaGroupStore(p['inputs']['store'], cache_groups=1)
    dispersions = prior.arena.load_reaction_dispersions(p['inputs']['disp'])
    all_rows = []
    for binding in p['bindings']:
        receipt = OUT/(binding['key']+'_result.json')
        if receipt.exists():
            saved = load(receipt)
            assert saved['protocol_sha256'] == sha(OUT/'protocol.json')
            assert sha(saved['cache_path']) == saved['cache_sha256']
            all_rows.extend(saved['rows'])
            continue
        needed = sorted({i for k in KS for rep in range(16) for i, _ in (batch_rows(manifest, rep, k) or [])} |
                        {i for i, _ in binding['original_batch']})
        model = prior.arena._pilot_model(torch.device('cuda'), torch.float32)
        model.load_state_dict(torch.load(binding['state_path'], map_location='cpu', weights_only=False)['model'], strict=True)
        assert prior.arena.digest(model) == binding['state_sha256']
        parameters = prior.arena.named_trainable_parameters(model)
        assert [(name, list(v.shape), v.numel()) for name, v in parameters.items()] == [(r['name'], r['shape'], r['numel']) for r in p['parameter_order']]
        assert sum(v.numel() for v in parameters.values()) == 9446
        shadow, rng = copy.deepcopy(model).double(), prior.arena.capture_rng_state()
        gradients, timings, losses = [], [], []
        print('BEGIN', binding['key'], len(needed), 'unique reaction gradients', flush=True)
        for offset, index in enumerate(needed):
            start = time.perf_counter()
            row = manifest['catalog'][index]
            reaction = store.load_variant((row['database'], row['reaction_id']), row['variant_suffix'])
            objective = prior.ChemistryBatchObjective(model, shadow, (reaction,), (1.,), dispersions)
            value, gradient = objective.value_and_grad()
            vector = np.concatenate([gradient[name].detach().double().cpu().numpy().reshape(-1) for name in parameters])
            assert vector.shape == (9446,) and np.isfinite(vector).all() and np.isfinite(value)
            gradients.append(vector)
            losses.append(value)
            timings.append(time.perf_counter()-start)
            assert prior.arena.digest(model) == binding['state_sha256']
            assert all(v.grad is None for v in model.parameters())
            del reaction, objective, gradient
            if (offset+1) % 16 == 0 or offset+1 == len(needed):
                print('CACHE', binding['key'], offset+1, '/', len(needed), flush=True)
        assert prior.arena.equal(prior.arena.capture_rng_state(), rng)
        singletons = np.stack(gradients)
        old_arrays = np.load(binding['array_path'])
        raw, gold = old_arrays['raw_original'], old_arrays['g_full251']
        reconstructed = estimate(singletons, needed, binding['original_batch'])
        error = np.linalg.norm(reconstructed-raw[0])/np.linalg.norm(raw[0])
        assert error <= p['original_gradient_reconstruction_relative_L2_max'], error
        cache = OUT/(binding['key']+'_singleton_gradients.npz')
        np.savez_compressed(cache, gradients=singletons, canonical_indices=needed, timings=timings, losses=losses)
        offsets = {index: offset for offset, index in enumerate(needed)}
        full_time_projection = sum(manifest['counts'][db]*np.mean([timings[offsets[i]] for i in needed if manifest['catalog'][i]['database'] == db]) for db in manifest['database_order'])
        rows = []
        for k in KS:
            for rep in range(16):
                batch = batch_rows(manifest, rep, k)
                if batch is None:
                    continue
                estimated = estimate(singletons, needed, batch)
                ne, nf = np.linalg.norm(estimated), np.linalg.norm(gold)
                assert ne > 0 and nf > 0
                cosine = float(estimated@gold/(ne*nf))
                substituted = raw.copy()
                substituted[0] = estimated
                direction, metadata = prior.arena.offline.maxmin(substituted, p['immutable_files'][str(prior.arena.offline.reuse.SOLVER)])
                valid = metadata['gamma_star'] > 1e-12 and not metadata['stationary_label']
                products = substituted@direction/np.linalg.norm(substituted, axis=1) if valid else None
                true_products = (np.vstack((gold, raw[1:]))@direction/np.linalg.norm(np.vstack((gold, raw[1:])), axis=1)) if valid else None
                rows.append({'key': binding['key'], 'K': k, 'replicate': rep, 'cosine': cosine,
                    'norm_ratio': float(ne/nf), 'unit_gradient_angular_error': float(np.linalg.norm(estimated/ne-gold/nf)),
                    'angle_degrees': float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                    'gamma': metadata['gamma_star'], 'p_est': float(products[0]) if valid else None,
                    'p_full': float(true_products[0]) if valid else None,
                    'estimated_task_progresses': products.tolist() if valid else None,
                    'true_scientific_task_progresses': true_products.tolist() if valid else None,
                    'estimated_strict_common_descent': bool(valid and np.all(products > 0)),
                    'fullchem_descent': bool(valid and true_products[0] > 0),
                    'solver_qualified': bool(valid and np.all(products > 0) and abs(np.linalg.norm(direction)-1) <= 1e-12),
                    'coefficients': metadata['MGDA']['coefficients'], 'KKT_certificate': metadata['certificate'],
                    'reaction_evaluations': len(batch),
                    'singleton_gradient_seconds_sum': float(sum(timings[offsets[i]] for i, _ in batch)),
                    'FULL251_singleton_seconds_projection': float(full_time_projection)})
        saved = {**binding, 'protocol_sha256': sha(OUT/'protocol.json'), 'cache_path': str(cache), 'cache_sha256': sha(cache),
                 'original_sample_gradient_reconstruction_relative_L2': float(error), 'unique_reactions_computed': len(needed),
                 'model_state_unchanged': True, 'RNG_unchanged': True, 'rows': rows}
        dump(receipt, saved)
        all_rows.extend(rows)
        del model, shadow, gradients, singletons, old_arrays
        torch.cuda.empty_cache()
        verify(p)
        print('COMPLETE', binding['key'], flush=True)
    dump(OUT/'arena_rows.json', all_rows)
    print(json.dumps(summarize(all_rows), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--small-database-census', action='store_true')
    args = parser.parse_args()
    freeze(args.small_database_census) if args.stage == 'freeze' else run()
