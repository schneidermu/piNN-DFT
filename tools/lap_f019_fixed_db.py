"""F019 cached, read-only qualification. No training or development-data access."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import lap_f013_geometry_audit as geometry
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'ABDE4': 4, 'DBH76': 70, 'EA13': 11, 'IP13': 13, 'MGAE109': 104,
          'NCCE31': 28, 'PA8': 8, 'pTC13': 13}
PARENT = '47ca67599f34ea77aae69eef6d12d80fd3c2344f'


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weights(databases):
    return np.array([251 / (8 * COUNTS[d]) for d in databases], dtype=np.float64)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def execute():
    b8 = geometry.b8
    receipt = b8.read_json(ROOT / 'lap_f010_normalization_receipt.json')
    authoritative = b8.read_json(ROOT / 'lap_f013_gradient_receipts.json')
    scales = receipt['scales']
    assert geometry.arm.scales_sha256(scales) == authoritative['a_d_sha256']
    batches = geometry.f010._batches()
    manifest_sha = geometry.f010.continuation.MANIFEST_SHA
    seed = int(hashlib.sha256((manifest_sha + ':F019-32-positions').encode()).hexdigest()[:16], 16)
    positions = sorted(np.random.default_rng(seed).choice(251, 32, replace=False).tolist())
    protocol = {'experiment': 'F019', 'parent': PARENT, 'architecture': 'R0',
                'a_d_sha256': authoritative['a_d_sha256'], 'manifest_sha256': manifest_sha,
                'counts': COUNTS, 'positions_seed': seed, 'positions': positions,
                'reconstruction_atol': 1e-10, 'reconstruction_rtol': 1e-10,
                'positive_dot_min': 24, 'p10_max_deterioration': 0.10,
                'c010': geometry.C010, 'lambdas': b8.LAMBDAS,
                'lr': 2.5e-5, 'betas': [0.9, 0.999], 'eps': 1e-8,
                'weight_decay': 0.01, 'foreach': False, 'scheduler': None,
                'clipping': None, 'conditional_updates': 128,
                'chemistry_presentations_per_update': 8,
                'evaluation_policy': 'one-fixed-variant-per-identity',
                'clean28': 'only final t128 after training-only qualification',
                'stop_on_failed_preflight': True}
    target = ROOT / 'lap_f019_protocol.json'
    if target.exists():
        assert b8.read_json(target) == protocol, 'Frozen protocol changed'
    else:
        write(target.name, protocol)
    saved = torch.load(geometry.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    moment = geometry.f010.continuation.moment_sha(saved['optimizer'])
    assert moment == '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
    selected = [s for s in geometry.STATES if s['name'] in
                ('b8_t128', 'f010_t128', 'f010_t251', 'f012_t128')]
    provenance = {'optimizer_moment_sha256': moment, 'states': [],
                  'protocol_sha256': b8.sha256_file(target), 'optimizer_updates': 0}
    old_pins = b8.read_json(ROOT / 'lap_f015_protocol.json')['pins']
    results, rows = [], []
    c019 = None
    base = None
    for spec in selected:
        old = next(s for s in authoritative['states'] if s['checkpoint'] == spec['name'])
        assert b8.sha256_file(spec['reactions']) == old['reactions_sha256']
        metadata = spec['reactions'].with_suffix('.json')
        assert b8.sha256_file(metadata) == old_pins[str(metadata)]
        if spec['physical'].suffix != '.npz':
            for task in ('ae17', 'exc', 'op'):
                path = spec['physical'] / f'{task}.npy.npz'
                assert b8.sha256_file(path) == old_pins[str(path)]
        physical_sha = (b8.sha256_file(spec['physical']) if spec['physical'].suffix == '.npz'
                        else old['physical_sha256'])
        if spec['physical'].suffix == '.npz':
            assert physical_sha == old['physical_sha256']
        state = geometry.load_state(spec, scales)
        db = state['databases']
        assert Counter(db) == COUNTS
        w = weights(db)
        coefficients = geometry.arm.jw_coefficients(state['residuals'], scales, db)
        singleton = coefficients[:, None] * state['jacobian'] * 251
        gj = singleton.mean(axis=0)
        gd = (w[:, None] * singleton).mean(axis=0)
        assert np.allclose(gj, state['components']['jw'], atol=1e-10, rtol=1e-10)
        if c019 is None:
            c019 = geometry.C010 * float(np.linalg.norm(gj) / np.linalg.norm(gd))
            base = (state, singleton, gj, gd, w)
        new = state['f010'] + b8.LAMBDAS['relchem'] * (c019 * gd - geometry.C010 * gj)
        losses = np.sqrt(np.square(state['residuals']) + 1e-20) / np.array([scales[d] for d in db])
        pieces = [singleton[np.array(db) == d].mean(axis=0) / 8 for d in sorted(COUNTS)]
        assert np.allclose(sum(pieces), gd, atol=1e-10, rtol=1e-10)
        results.append({'checkpoint': spec['name'], 'jw': float(losses.mean()),
                        'db': float(np.mean(w * losses)), 'g_j_norm': float(np.linalg.norm(gj)),
                        'g_db_norm': float(np.linalg.norm(gd)), 'cosine': cosine(gj, gd),
                        'composite_cosine': cosine(state['f010'], new)})
        for d in sorted(COUNTS):
            vector = singleton[np.array(db) == d].mean(axis=0)
            rows.append({'checkpoint': spec['name'], 'database': d,
                         'weight': 251 / (8 * COUNTS[d]),
                         'f010_negative_unit_direction_derivative': -float(vector @ state['f010'] / np.linalg.norm(state['f010'])),
                         'f019_negative_unit_direction_derivative': -float(vector @ new / np.linalg.norm(new))})
        provenance['states'].append(old)
    write('lap_f019_provenance.json', provenance)
    write('lap_f019_gradient_calibration.json', {'c019': c019, 'states': results})
    table('lap_f019_database_gradient_geometry.csv', rows)
    state, singleton, gj, gd, w = base
    lookup = {(r['identity'], r['variant']): i for i, r in enumerate(state['rows'])}
    mismatches = [dict(position=batch['cursor'], **r) for batch in batches
                  for r in batch['relchem'] if (r['identity'], r['variant']) not in lookup]
    if mismatches:
        write('lap_f019_preflight_decision.json', {
            'classification': 'PARTIAL', 'preflight': 'BLOCKED',
            'reason': 'Frozen full251 Jacobian and historical stream have different grid variants; '
                      'required exact full-cycle reconstruction is not authenticated.',
            'mismatched_presentations': len(mismatches), 'total_cycle_presentations': 2008,
            'unique_mismatched_identities': len({r['identity'] for r in mismatches}),
            'mismatches': mismatches, 'stage_b': 'NOT RUN', 'stage_c': 'NOT RUN',
            'training_launched': False, 'optimizer_updates': 0, 'chemistry_presentations': 0,
            'clean28': 'NOT EVALUATED', 'c019_status': 'evaluation-manifest calibration only; not training-qualified'})
        print('PARTIAL: preflight blocked by variant mismatch', len(mismatches))
        return
    all_indices = [[lookup[(r['identity'], r['variant'])] for r in batch['relchem']] for batch in batches]
    assert Counter(i for indices in all_indices for i in indices) == dict.fromkeys(range(251), 8)
    cycle = np.mean([np.mean(singleton[ix] * w[ix, None], axis=0) for ix in all_indices], axis=0)
    assert np.allclose(cycle, gd, atol=1e-10, rtol=1e-10)
    fidelity = []
    for position in positions:
        ix = all_indices[position]
        old = singleton[ix].mean(axis=0)
        new = (singleton[ix] * w[ix, None]).mean(axis=0)
        fidelity.append({'position': position, 'f010_cosine': cosine(old, gj),
                         'f019_cosine': cosine(new, gd),
                         'f010_norm_ratio': float(np.linalg.norm(old) / np.linalg.norm(gj)),
                         'f019_norm_ratio': float(np.linalg.norm(new) / np.linalg.norm(gd)),
                         'f019_relative_error': float(np.linalg.norm(new - gd) / np.linalg.norm(gd)),
                         'maximum_weight': float(w[ix].max()), 'finite': bool(np.isfinite(new).all())})
    table('lap_f019_stochastic_fidelity.csv', fidelity)
    positive = sum(r['f019_cosine'] > 0 for r in fidelity)
    p10_old = float(np.quantile([r['f010_cosine'] for r in fidelity], .1))
    p10_new = float(np.quantile([r['f019_cosine'] for r in fidelity], .1))
    decision = {'positive_dot_count': positive, 'p10_f010': p10_old, 'p10_f019': p10_new,
                'full_cycle_reconstruction_l2': float(np.linalg.norm(cycle - gd)),
                'stage_b_pass': positive >= 24 and p10_new >= p10_old - .10,
                'training_launched': False, 'optimizer_updates': 0, 'clean28': 'NOT EVALUATED'}
    write('lap_f019_preflight_decision.json', decision)
    print(json.dumps(decision))


if __name__ == '__main__':
    execute()
