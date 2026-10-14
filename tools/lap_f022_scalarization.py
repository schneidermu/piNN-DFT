"""Cost-first F022 cached geometry; no optimizer or GPU evaluation in this stage."""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter

import lap_f015_audit as audit
import lap_f020_stream as stream
import numpy as np
import torch

ROOT = audit.ROOT
OUT = ROOT.parent / 'lap_f022_scalarization_20261014'
PARENT = '6b858373a1fdffcd82a07bc44c7eeaaf5f9dc49a'
CHECKPOINT = '1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b'
MOMENT = '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
SCALES = '3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d'
b8, f010 = audit.geometry.b8, audit.geometry.f010


def write(name, value):
    audit.write(ROOT / ('lap_f022_' + name + '.json'), value)


def interval(cjl, cjo, clo):
    """Intersect three strict affine inequalities; never choose a fitted alpha."""
    lower, upper = 0., float('inf')
    for intercept, slope in ((1., cjl), (cjl, 1.), (cjo, clo)):
        if slope > 0:
            lower = max(lower, -intercept / slope)
        elif slope < 0:
            upper = min(upper, -intercept / slope)
        elif intercept <= 0:
            raise ValueError('Empty strict alpha interval')
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError('No finite nonempty alpha interval')
    return lower, upper


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite([na, nb]).all() or min(na, nb) <= 0:
        raise ValueError('Finite nonzero gradient norms required')
    return float(a @ b / na / nb)


def chemistry(jacobian, residuals, databases, scales):
    factors = np.array([audit.chemistry._factor(d) for d in databases])
    phi = np.sqrt(residuals**2 + 1e-20)
    coeff_j = f010.f010_raw_coefficients(residuals, scales, databases)
    coeff_l = f010.fact.production_coefficients(residuals, factors)
    return {'JW': coeff_j @ jacobian, 'legacy': coeff_l @ jacobian,
            'MAE': np.sign(residuals) @ jacobian / len(residuals)}, coeff_j, coeff_l, factors, phi


def calibrate(gj, gl, go):
    nj, nl = float(np.linalg.norm(gj)), float(np.linalg.norm(gl))
    cjl, cjo, clo = cosine(gj, gl), cosine(gj, go), cosine(gl, go)
    lower, upper = interval(cjl, cjo, clo)
    alpha = (lower + upper) / 2
    direction = gj / nj + alpha * gl / nl
    D = float(np.linalg.norm(direction))
    K = audit.geometry.C010 * nj / D
    a, b = K / nj, K * alpha / nl
    assert np.allclose(a * gj + b * gl, K * direction, rtol=1e-10, atol=1e-10)
    assert abs(np.linalg.norm(a * gj + b * gl) / (audit.geometry.C010 * nj) - 1) < 1e-10
    assert all(v @ direction > 0 for v in (gj, gl, go))
    return {'alpha_min': lower, 'alpha_max': upper, 'alpha': alpha,
            'cosine_JW_legacy': cjl, 'cosine_JW_operator': cjo, 'cosine_legacy_operator': clo,
            'JW_norm_S': nj, 'legacy_norm_S': nl, 'D': D, 'K': K, 'a': a, 'b': b,
            'chemistry_gradient_norm': float(np.linalg.norm(a * gj + b * gl))}


def freeze():
    subprocess.run(['git', 'merge-base', '--is-ancestor', PARENT, 'HEAD'], cwd=ROOT, check=True)
    path = ROOT / 'lap_f022_protocol.json'
    sources = {str(p.relative_to(ROOT)): audit.sha(p) for p in (
        ROOT / 'tools/lap_f022_scalarization.py', ROOT / 'tools/lap_f015_audit.py',
        ROOT / 'tools/lap_f013_geometry_audit.py', ROOT / 'tools/lap_f020_stream.py',
        ROOT / 'tools/lap_training_metric_gradient_audit.py',
        ROOT / 'tools/lap_relchem_factorial_gradient_audit.py',
        ROOT / 'tools/lap_f010_uniform_normalized_chemistry.py',
        ROOT / 'tools/lap_b8_adamw.py', ROOT / 'train_models/lap_fixed_adamw.py',
        ROOT / 'train_models/optuna_joint.py', ROOT / 'train_models/lap_operator.py')}
    protocol = {'experiment': 'F022', 'parent': PARENT, 'architecture': 'R0', 'coordinates': 9446,
        'checkpoint_sha256': CHECKPOINT, 'moment_sha256': MOMENT,
        'manifest_sha256': f010.continuation.MANIFEST_SHA, 'A_d_sha256': SCALES,
        'source_hashes': sources, 'calibration_population': 'S: exact F020 stream identity+variant',
        'scientific_population': 'E: one fixed evaluation variant per 251/17 identities; full90 physical',
        'alpha_rule': 'midpoint of finite strict JW/legacy/operator chemistry-only interval',
        'c010': audit.geometry.C010, 'lambdas': b8.LAMBDAS, 'positions': list(range(64)),
        'preflight_seconds': 900, 'active_seconds': 5400, 'wall_seconds': 10800,
        'max_updates': 64, 'max_arms': 1, 'lr': 2.5e-5, 'betas': [.9, .999],
        'eps': 1e-8, 'weight_decay': .01, 'foreach': False, 'scheduler': None,
        'clipping': None, 'geometry_gate': 'strict positive g_J dot G and g_L dot G on S',
        'numerical_rtol': 1e-10, 'numerical_atol': 1e-10,
        'Clean28': 'once on final t64 only if all four original P536 ratios strictly <1',
        'baseline': b8.BASELINE}
    OUT.mkdir(exist_ok=True)
    if path.exists():
        assert audit.read(path) == protocol, 'Frozen F022 protocol changed'
    else:
        write('protocol', protocol)
        audit.write(OUT / 'clock.json', {'origin_unix': time.time(), 'active_seconds': 0.})
    return protocol


def authenticate(protocol):
    old = audit.read(ROOT / 'lap_f015_protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        p = ROOT / relative
        if str(p) in old['sources']:
            assert audit.sha(p) == old['sources'][str(p)]
        assert audit.sha(p) == digest
    saved = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert audit.sha(f010.PARENT_CHECKPOINT) == CHECKPOINT
    assert f010.continuation.moment_sha(saved['optimizer']) == MOMENT
    assert audit.sha(f010.continuation.PARENT_MANIFEST) == protocol['manifest_sha256']
    scales = audit.read(ROOT / 'lap_f010_normalization_receipt.json')['scales']
    assert audit.geometry.arm.scales_sha256(scales) == SCALES
    spec = audit.geometry.STATES[0]
    receipt = audit.read(ROOT / 'lap_f013_gradient_receipts.json')['states'][0]
    assert receipt['checkpoint_sha256'] == CHECKPOINT
    assert audit.sha(spec['reactions']) == receipt['reactions_sha256']
    paths = [spec['reactions'], spec['reactions'].with_suffix('.json'),
             spec['physical'] / 'result.json']
    paths += [spec['physical'] / (t + '.npy.npz') for t in ('ae17', 'exc', 'op')]
    for p in paths:
        assert audit.sha(p) == old['pins'][str(p)]
    reconstructed = audit.read(ROOT / 'lap_f020_jacobian_reconstruction.json')
    assert audit.sha(stream.OUT / 'stream_jacobian.npz') == reconstructed['sha256']
    assert audit.sha(stream.OUT / 'stream_rows.json') == reconstructed['mapping_sha256']
    rowmap = audit.read(stream.OUT / 'stream_rows.json')
    physical_meta = audit.read(spec['physical'] / 'result.json')
    names = tuple(physical_meta['parameter_names'])
    assert len(names) == len(set(names)) and names == tuple(sorted(names))
    assert sum(saved['model'][n].numel() for n in names) == 9446
    order_sha = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    assert all(r['parameter_order_sha256'] == order_sha and r['checkpoint_sha256'] == CHECKPOINT for r in rowmap)
    assert physical_meta['checkpoint_sha256'] == CHECKPOINT
    bundle = audit.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    for r in rowmap:
        assert stream.source_hash(bundle.reactions[r['identity']], r['variant'], bundle) == r['source_hash']
    bundle.close()
    batches = f010._batches()
    lookup = {(r['identity'], r['variant']): i for i, r in enumerate(rowmap)}
    indices = [[lookup[(r['identity'], r['variant'])] for r in b['relchem']] for b in batches]
    assert Counter(i for ix in indices for i in ix) == dict.fromkeys(range(251), 8)
    assert len(indices) == 251 and all(len(ix) == 8 for ix in indices)
    provenance = {'authenticated': True, 'checkpoint_sha256': CHECKPOINT, 'moment_sha256': MOMENT,
        'parameter_order_sha256': order_sha, 'manifest_sha256': protocol['manifest_sha256'],
        'dataset_logical_sha256': b8.DATA_SHA, 'A_d_sha256': SCALES,
        'stream_jacobian_sha256': reconstructed['sha256'], 'stream_mapping_sha256': reconstructed['mapping_sha256'],
        'evaluation_cache_sha256': receipt['reactions_sha256'],
        'physical_vector_hashes': {t: audit.sha(spec['physical'] / (t + '.npy.npz')) for t in ('ae17', 'exc', 'op')},
        'source_hashes': protocol['source_hashes'], 'source_records_verified': 251,
        'physical_parameter_names_match': True, 'optimizer_updates': 0, 'Clean28_accessed': False}
    write('provenance', provenance)
    return scales, spec, rowmap, indices


def execute():
    protocol = freeze()
    start = time.perf_counter()
    scales, spec, rows, indices = authenticate(protocol)
    with np.load(stream.OUT / 'stream_jacobian.npz') as cache:
        J, e = cache['gradients'].copy(), cache['residual'].copy()
    assert J.shape == (251, 9446) and np.isfinite(J).all() and np.isfinite(e).all()
    db = [r['database'] for r in rows]
    s, cj, cl, factors, phi = chemistry(J, e, db, scales)
    state = audit.geometry.load_state(spec, scales)
    physical = {t: state['components'][t] for t in ('ae17', 'exc', 'op')}
    evaluated, _ej, _el, _ef, _ep = chemistry(state['jacobian'], np.asarray(state['residuals']), state['databases'], scales)
    np.testing.assert_allclose(evaluated['JW'], state['components']['jw'], rtol=1e-10, atol=1e-10)
    normalization = audit.read(ROOT / 'lap_f010_normalization_receipt.json')
    assert abs(np.linalg.norm(evaluated['legacy']) / normalization['old_norm'] - 1) < 1e-10
    assert abs(cosine(evaluated['legacy'], evaluated['JW']) - normalization['cosine_old_f010']) < 1e-10
    expected = audit.read(ROOT / 'lap_f013_gradient_receipts.json')['states'][0]['relchem_scalar']
    assert abs(float((_ef * _ep).mean()) - expected) < 1e-10
    assert abs(np.linalg.norm(s['JW']) - audit.read(ROOT / 'lap_f020_stream_calibration.json')['JW_norm_S']) < 1e-10
    cal = calibrate(s['JW'], s['legacy'], physical['op'])
    write('feasible_interval', {k: cal[k] for k in ('alpha_min', 'alpha_max', 'alpha',
        'cosine_JW_legacy', 'cosine_JW_operator', 'cosine_legacy_operator')})
    write('gradient_calibration', {**cal, 'population': 'S', 'factors': dict(zip(db, factors, strict=True)),
        'JW_scalar_S': float((phi / np.array([scales[d] for d in db])).mean()),
        'legacy_scalar_S': float((phi * factors).mean()), 'legacy_scalar_E': float((_ep * _ef).mean()),
        'JW_scalar_E': state['jw_scalar'], 'normalization_identity_passed': True,
        'legacy_norm_and_JW_cosine_receipt_parity': True})
    chemistry_mix = cal['a'] * s['JW'] + cal['b'] * s['legacy']
    weighted_physics = {t: b8.LAMBDAS[t] * physical[t] for t in physical}
    P = sum(weighted_physics.values())
    G = b8.LAMBDAS['relchem'] * chemistry_mix + P
    control = b8.LAMBDAS['relchem'] * audit.geometry.C010 * s['JW'] + P
    vectors = {'JW_contribution': b8.LAMBDAS['relchem'] * cal['a'] * s['JW'],
        'legacy_contribution': b8.LAMBDAS['relchem'] * cal['b'] * s['legacy'],
        **{t + '_contribution': v for t, v in weighted_physics.items()},
        'chemistry_only': b8.LAMBDAS['relchem'] * chemistry_mix, 'physics_only': P,
        'F010': control, 'F022': G}
    geometry = []
    for population, raw in [('S', s), ('E', evaluated)]:
        for direction, vector in vectors.items():
            for target, g in {**raw, **physical}.items():
                geometry.append({'population': population, 'direction': direction, 'target': target,
                    'gradient_norm': float(np.linalg.norm(vector)), 'cosine': cosine(g, vector),
                    'positive_progress_dot': float(g @ vector),
                    'negative_gradient_direction_derivative': -float(g @ vector),
                    'unit_negative_gradient_derivative': -float(g @ vector / np.linalg.norm(vector))})
    audit.table('lap_f022_full_gradient_geometry.csv', geometry)
    database_rows = []
    for population, jac, residual, labels in [('S', J, e, db), ('E', state['jacobian'], np.asarray(state['residuals']), state['databases'])]:
        for d in sorted(set(labels)):
            mask = np.array(labels) == d
            raw, _c1, _c2, _f, _p = chemistry(jac[mask], residual[mask], [d]*int(mask.sum()), scales)
            for direction, vector in [('F010', control), ('F022', G)]:
                database_rows.append({'population': population, 'database': d, 'direction': direction,
                    **{t + '_negative_gradient_derivative': -float(g @ vector) for t, g in raw.items()}})
    audit.table('lap_f022_database_directions.csv', database_rows)
    singleton_j, singleton_l = J * (251*cj[:, None]), J * (251*cl[:, None])
    singleton_mix = cal['a'] * singleton_j + cal['b'] * singleton_l
    mixture_full = singleton_mix.mean(axis=0)
    np.testing.assert_allclose(mixture_full, chemistry_mix, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.array([singleton_mix[ix].mean(axis=0) for ix in indices]).mean(axis=0), mixture_full, rtol=1e-10, atol=1e-10)
    fidelity = []
    for position, ix in enumerate(indices[:64]):
        old = audit.geometry.C010 * singleton_j[ix].mean(axis=0)
        new = singleton_mix[ix].mean(axis=0)
        full_old = audit.geometry.C010 * s['JW']
        for arm, vector, full in [('F010', old, full_old), ('F022', new, mixture_full)]:
            fidelity.append({'position': position, 'arm': arm, 'cosine': cosine(vector, full),
                'positive_dot': bool(vector @ full > 0), 'norm_ratio': float(np.linalg.norm(vector) / np.linalg.norm(full)),
                'relative_squared_error': float(np.linalg.norm(vector-full)**2 / np.linalg.norm(full)**2),
                'db_weighted_norm_contributions': json.dumps({d: float(np.linalg.norm(
                    (audit.geometry.C010 * singleton_j if arm == 'F010' else singleton_mix)[[i for i in ix if db[i] == d]].sum(axis=0) / 8))
                    for d in sorted({db[i] for i in ix})}, sort_keys=True)})
    audit.table('lap_f022_batch_fidelity.csv', fidelity)
    summary = {arm: {'positive_dot_count': sum(r['positive_dot'] for r in fidelity if r['arm'] == arm),
        'p10_cosine': float(np.quantile([r['cosine'] for r in fidelity if r['arm'] == arm], .1)),
        'median_cosine': float(np.median([r['cosine'] for r in fidelity if r['arm'] == arm])),
        'max_norm_ratio': max(r['norm_ratio'] for r in fidelity if r['arm'] == arm),
        'relative_gradient_MSE': float(np.mean([r['relative_squared_error'] for r in fidelity if r['arm'] == arm]))}
        for arm in ('F010', 'F022')}
    passed = bool(s['JW'] @ G > 0 and s['legacy'] @ G > 0)
    active = time.perf_counter() - start
    clock = audit.read(OUT / 'clock.json')
    wall = time.time() - clock['origin_unix']
    assert active < 900 and wall < 10800
    decision = {'classification': 'C' if not passed else 'D', 'cached_gradient_preflight': 'PASS' if passed else 'FAIL',
        'JW_dot_G': float(s['JW'] @ G), 'legacy_dot_G': float(s['legacy'] @ G),
        'operator_dot_G': float(physical['op'] @ G),
        'legacy_dot_chemistry': float(s['legacy'] @ (G-P)), 'legacy_dot_physics': float(s['legacy'] @ P),
        'composite_cancellation_ratio': float(np.linalg.norm(G) / sum(np.linalg.norm(v) for k, v in vectors.items() if k.endswith('_contribution'))),
        'fidelity': summary, 'native_step': 'NOT RUN', 'training_launched': False,
        'optimizer_updates': 0, 'Clean28': 'NOT EVALUATED', 'active_compute_seconds': active,
        'whole_workflow_wall_seconds': wall, 'budget_respected': True,
        'reason': 'Fixed physical contribution reverses legacy descent despite feasible chemistry-only mixture.' if not passed else
                  'Cached gate passes; dependent native qualification remains to be executed.',
        'next_experiment': 'Read-only feasibility audit of fixed scalarization with legacy-preserving physical supervision; first reuse this geometry to bound the unchanged-physics cone, no training or coefficient rescue.'}
    write('preflight_decision', decision)
    write('decision', decision)
    write('resource_budget', {'preflight_limit_seconds': 900, 'active_limit_seconds': 5400,
        'wall_limit_seconds': 10800, 'active_compute_seconds': active, 'whole_workflow_wall_seconds': wall,
        'training_projection': 'NOT RUN — primary cached geometry gate failed' if not passed else 'PENDING',
        'complete_endpoint_reserve': 'Not consumed; training not authorized', 'max_training_updates': 64, 'max_arms': 1})
    print(json.dumps({**cal, **decision}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'cached'))
    args = parser.parse_args()
    {'freeze': freeze, 'cached': execute}[args.stage]()
