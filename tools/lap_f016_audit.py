"""Read-only F016 stages, training-only physical inputs and immutable protocol."""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import numpy as np
import torch
from lap_f016_descriptors import descriptors, raw_q

ROOT = old.ROOT
OUT = ROOT.parent / 'lap_f016_audit_20261013'


def grids(bundle):
    result = {}
    for spec in old.chemistry._selected(bundle):
        meta = bundle.reactions[spec['identity']]
        for ref, species in zip(meta['variants'][spec['variant']], meta['components'], strict=True):
            row = result.setdefault(ref['grid_id'], {'source': ref, 'species': species, 'databases': set()})
            row['databases'].add(spec['database'])
    return sorted(result.items())


def freeze():
    old.verify()
    OUT.mkdir(exist_ok=True)
    path = ROOT / 'lap_f016_descriptor_protocol.json'
    if path.exists():
        verify()
        return
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == '8e2fe20b1874fbaded101ea8cbb2a949b3013a91'
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/lap_full_vxc'], cwd=ROOT, check=True)
    names = ['NN_models_lap.py', 'NN_models.py', 'lap_operator.py', 'lap_vxc.py']
    pins = {str(ROOT / 'train_models' / n): old.sha(ROOT / 'train_models' / n) for n in names}
    for p in old.OUT.glob('descriptor_*.*'):
        pins[str(p)] = old.sha(p)
    for n in ('lap_f015_protocol.json', 'lap_f015_provenance.json', 'lap_f015_species_xc_decomposition.csv',
              'lap_f015_region_reaction_decomposition.csv', 'lap_f015_spatial_energy_differences.csv'):
        pins[str(ROOT / n)] = old.sha(ROOT / n)
    old.write(path, {'experiment': 'F016', 'reference_commit': head, 'pins': pins,
                    'updates': 0, 'training': False, 'representations': {'R0': 'tanh(q)', 'R1': 'tanh(q/4)', 'R2': ['tanh(q)', 'tanh(q/4)']},
                    'thresholds': [.95, .99, .999], 'density_boundaries': [1e-6, 1e-3],
                    'raw_q_separation': 1., 'near_collision_float32_ULPs': [1, 4],
                    'sample': 'F015 sorted495 grids stride101; per-density eligible linspace50000; original identities/order',
                    'distance': '8-coordinate R0/R1 Euclidean; R2 append q/4 scaled by 1/sqrt(2), both q channels by 1/sqrt(2); original six non-q coordinates unchanged',
                    'near_environment_distances': [.001, .01, .05],
                    'tolerances': {'forward_f64_atol': 1e-12, 'forward_f64_rtol': 1e-12,
                                   'forward_f32_atol': 2e-6, 'forward_f32_rtol': 2e-6,
                                   'input_gradient_f64_atol': 1e-10, 'input_gradient_f64_rtol': 1e-10,
                                   'operator_f64_atol': 1e-10, 'operator_f64_rtol': 1e-10,
                                   'fd_f64_atol': 1e-7, 'fd_f64_rtol': 1e-4,
                                   'fd_mixed_atol': 1e-5, 'fd_mixed_rtol': .05},
                    'r2_anchor_metric': 'Original first8 coordinates only. Added channels excluded from Lagrange distances to preserve exact R0 embedding.',
                    'no_validation_or_test_inputs': True, 'one_variant_per_identity': True})
    old.write(ROOT / 'lap_f016_feature_layouts.json', {
        'raw': ['rho_a', 'rho_b', 'sigma_a', 'sigma_total', 'sigma_b', 'tau_a_UNUSED', 'tau_b_UNUSED', 'lap_a', 'lap_b'],
        'R0': ['n_a', 'n_b', 's_a', 's_total', 's_b', 'q_a', 'q_b', 'zeta'],
        'R1': ['n_a', 'n_b', 's_a', 's_total', 's_b', 'q4_a', 'q4_b', 'zeta'],
        'R2': ['n_a', 'n_b', 's_a', 's_total', 's_b', 'q_a', 'q_b', 'zeta', 'q4_a', 'q4_b'],
        'exchange_consumed': {'R0': [[2, 5], [4, 6]], 'R1': [[2, 5], [4, 6]], 'R2': [[2, 5, 8], [4, 6, 9]]},
        'R2_extra_parameters_h32': 96})
    verify()


def verify():
    old.verify()
    p = old.read(ROOT / 'lap_f016_descriptor_protocol.json')
    for name, digest in p['pins'].items():
        assert old.sha(name) == digest, name
    return p


def collision(q, scale, dtype):
    """Unique raw-q values per output bin; count affected values, not n² pairs."""
    unique = np.unique(q)
    t = torch.tanh(torch.tensor(unique, dtype=dtype) / scale).numpy()
    _, starts, counts = np.unique(t, return_index=True, return_counts=True)
    # q is sorted and tanh monotone, so bin endpoints exactly establish spread.
    spread = unique[starts + counts - 1] - unique[starts]
    return {'unique_raw_values': len(unique), 'colliding_raw_values': int(counts[counts > 1].sum()),
            'separated_ge1_colliding_values': int(counts[spread >= 1].sum()),
            'maximum_colliding_raw_q_span': float(spread.max()), 'exact_endpoint_fraction': float(np.mean(abs(t) == 1)),
            'comparison': 'within selected grid/path/spin; distinct widened-source q values'}


def physical():
    verify()
    torch.set_num_threads(1)
    bundle = old.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    saturation, collisions, conditioning = [], [], []
    began = time.perf_counter()
    for i, (gid, info) in enumerate(grids(bundle)):
        receipt = OUT / f'grid_{i:03}.json'
        if receipt.exists():
            payload = old.read(receipt)
            assert payload['grid_id'] == gid
        else:
            group = bundle.handles.open(info['source']['shard'])[info['source']['group']]
            raw = group['Grid'][...].astype(np.float64)
            w = np.maximum(group['Weights'][...].astype(np.float64), 0)
            rho = raw[:, :2].sum(1)
            ew = w * np.maximum(rho, 0)
            sat, col, cond = [], [], []
            for pathway in ('correlation', 'exchange'):
                x = torch.from_numpy(raw.copy())
                if pathway == 'exchange':
                    x *= x.new_tensor([2, 2, 4, 4, 4, 2, 2, 2, 2])
                q = raw_q(x).numpy()
                for spin in range(2):
                    v = q[:, spin]
                    for scale in (1., 4.):
                        t = torch.tanh(torch.from_numpy(v) / scale).numpy()
                        d1 = (1 - t*t) / scale
                        d2 = -2 * t * d1 / scale
                        for region, mask in (('all', np.ones(len(v), bool)), ('rho_lt_1e-6', rho < 1e-6),
                                             ('rho_1e-6_to_1e-3', (rho >= 1e-6) & (rho < 1e-3)), ('rho_ge_1e-3', rho >= 1e-3)):
                            if not mask.any():
                                continue
                            common = {'grid_id': gid, 'species': info['species'], 'databases': ';'.join(sorted(info['databases'])),
                                      'pathway': pathway, 'spin': spin, 'scale': scale, 'density_regime': region, 'n': int(mask.sum())}
                            for threshold in (.95, .99, .999):
                                hit = mask & (abs(t) > threshold)
                                sat.append({**common, 'threshold': threshold, 'n_hit': int(hit.sum()),
                                            'positive_mass': float(w[mask].sum()), 'positive_hit': float(w[hit].sum()),
                                            'electron_mass': float(ew[mask].sum()), 'electron_hit': float(ew[hit].sum())})
                            cond.append({**common, 'median_abs_d1': float(np.median(d1[mask])),
                                         'electron_mean_abs_d1': float(ew[mask] @ d1[mask] / ew[mask].sum()) if ew[mask].sum() else 0.,
                                         'max_abs_d2': float(abs(d2[mask]).max()), 'd1_zero_fraction': float(np.mean(d1[mask] == 0))})
                        for dtype in (torch.float32, torch.float64):
                            col.append({'grid_id': gid, 'pathway': pathway, 'spin': spin, 'scale': scale,
                                        'dtype': str(dtype), **collision(v, scale, dtype)})
            payload = {'grid_id': gid, 'saturation': sat, 'collisions': col, 'conditioning': cond}
            old.write(receipt, payload)
            # Reuse exactly F015 sample ordering, stored raw values not inverted tanh.
            p = OUT / f'sample_{i:03}.npz'
            np.savez_compressed(p, raw=raw[::101], weights=w[::101], grid_index=np.full(len(raw[::101]), i))
        saturation.extend(payload['saturation'])
        collisions.extend(payload['collisions'])
        conditioning.extend(payload['conditioning'])
        if i % 80 == 0:
            print('F016_PHYSICAL', i, 'seconds', time.perf_counter() - began, flush=True)
    bundle.close()
    for name, rows in (('saturation', saturation), ('collisions', collisions), ('conditioning', conditioning)):
        old.write(OUT / f'{name}_full.json', rows)
    # Compact pooled population fractions, preserving every stratum.
    pooled = []
    for key in sorted({(r['pathway'], r['spin'], r['scale'], r['density_regime'], r['threshold']) for r in saturation}):
        chosen = [r for r in saturation if (r['pathway'], r['spin'], r['scale'], r['density_regime'], r['threshold']) == key]
        n = sum(r['n'] for r in chosen)
        pm, em = sum(r['positive_mass'] for r in chosen), sum(r['electron_mass'] for r in chosen)
        pooled.append(dict(zip(('pathway', 'spin', 'scale', 'density_regime', 'threshold'), key, strict=True)) |
                      {'n_points': n, 'point_fraction': sum(r['n_hit'] for r in chosen)/n,
                       'positive_weight_fraction': sum(r['positive_hit'] for r in chosen)/pm if pm else 0.,
                       'electron_fraction': sum(r['electron_hit'] for r in chosen)/em if em else 0.})
    old.table('lap_f016_saturation_comparison.csv', pooled)
    old.table('lap_f016_f32_f64_collisions.csv', collisions)
    old.table('lap_f016_descriptor_conditioning.csv', conditioning)


def environments():
    from scipy.spatial import cKDTree

    verify()
    sample = [np.load(p) for p in sorted(OUT.glob('sample_*.npz'))]
    raw = np.concatenate([p['raw'] for p in sample])
    ew = np.concatenate([p['weights'] * p['raw'][:, :2].sum(1) for p in sample])
    for p in sample:
        p.close()
    output, ambiguity, near = [], [], []
    for floor in (1e-6, 1e-3):
        eligible = np.flatnonzero(raw[:, :2].sum(1) >= floor)
        indices = eligible[np.linspace(0, len(eligible)-1, min(50000, len(eligible)), dtype=int)]
        x = torch.from_numpy(raw[indices])
        q, rho, w = raw_q(x).numpy(), x[:, :2].sum(1).numpy(), ew[indices]
        d = {r: descriptors(x, r).numpy() for r in ('R0', 'R1', 'R2')}
        # Maintain total q-block metric weight when adding two channels.
        d['R2'][:, [5, 6, 8, 9]] /= math.sqrt(2)
        base_distance, base_nn = cKDTree(d['R0']).query(d['R0'], k=2, workers=1)
        base_neighbor = base_nn[:, 1]
        qgap = np.linalg.norm(q - q[base_neighbor], axis=1)
        for r, vector in d.items():
            distance, nn = cKDTree(vector).query(vector, k=2, workers=1)
            n = nn[:, 1]
            tau = raw[indices, 5:7]
            tf = np.maximum(.3*(3*np.pi**2)**(2/3)*(rho+1e-10)**(5/3), 1e-12)
            td = np.linalg.norm(tau - tau[n], axis=1)/tf
            same_pair_distance = np.linalg.norm(vector-vector[base_neighbor], axis=1)
            for bound in (.001, .01, .05):
                baseline = (base_distance[:, 1] <= bound) & (qgap >= 1)
                resolved = baseline & (same_pair_distance > bound)
                output.append({'density_floor': floor, 'representation': r, 'distance_bound': bound,
                               'samples': len(indices), 'R0_near_pairs_qgap_ge1': int(baseline.sum()),
                               'resolved_same_pairs': int(resolved.sum()),
                               'resolved_fraction': float(resolved.sum()/baseline.sum()) if baseline.any() else 0.,
                               'electron_resolved_fraction': float(w[resolved].sum()/w[baseline].sum()) if w[baseline].sum() else 0.,
                               'median_baseline_qgap': float(np.median(qgap[baseline])) if baseline.any() else 0.,
                               'own_near_pairs': int((distance[:, 1] <= bound).sum())})
                mask = distance[:, 1] <= bound
                ambiguity.append({'density_floor': floor, 'representation': r, 'distance_bound': bound,
                                  'pairs': int(mask.sum()), 'median_tau_difference_TF': float(np.median(td[mask])),
                                  'p90_tau_difference_TF': float(np.quantile(td[mask], .9)),
                                  'electron_fraction_tau_gap_gt0_1': float(w[mask & (td > .1)].sum()/w[mask].sum())})
            # Floating-point pair collisions using the SAME nearest R0 pair.
            for scale in (1., 4.):
                t = torch.tanh(torch.tensor(q, dtype=torch.float32)/scale).numpy()
                other = t[base_neighbor]
                ulp = np.abs(np.spacing(np.maximum(abs(t), abs(other))))
                for tolerance in (0, 1, 4):
                    hit = (abs(t-other) <= tolerance*ulp).all(1) & (qgap >= 1)
                    near.append({'density_floor': floor, 'scale': scale, 'float32_ULPs': tolerance,
                                 'qgap_ge1_pairs': int((qgap >= 1).sum()), 'collision_pairs': int(hit.sum()),
                                 'electron_collision_mass_fraction': float(w[hit].sum()/w.sum())})
        np.savez_compressed(OUT / f'environment_{floor}.npz', indices=indices, raw=raw[indices], weight=w,
                            base_neighbor=base_neighbor, qgap=qgap)
    old.table('lap_f016_environment_distinguishability.csv', output)
    old.table('lap_f016_tau_residual_ambiguity.csv', ambiguity)
    old.table('lap_f016_near_collisions.csv', near)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'physical', 'environments'))
    args = parser.parse_args()
    globals()[args.stage]()
