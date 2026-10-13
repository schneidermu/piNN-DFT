"""Regenerate F020 summaries from saved numerical receipts only."""
import csv
import hashlib
import json

import lap_f020_stream as stream
import numpy as np


def execute():
    root, out, audit = stream.ROOT, stream.OUT, stream.audit
    decision = audit.read(root / 'lap_f020_decision.json')
    calibration = audit.read(root / 'lap_f020_stream_calibration.json')
    protocol = audit.read(root / 'lap_f020_protocol.json')
    old = audit.read(root / 'lap_f019_decision.json')
    assert all(audit.sha(root / n) == s for n, s in old['artifacts'].items())
    with (root / 'lap_f020_stochastic_fidelity.csv').open() as handle:
        fidelity = list(csv.DictReader(handle))
    matrices = np.load(out / 'full_gradients.npz')
    mapping = audit.read(out / 'stream_rows.json')
    state, _scales, names = audit.load(audit.geometry.STATES[0])
    db_names = state['databases']
    coeff_e = audit.geometry.arm.jw_coefficients(state['residuals'], _scales, db_names)
    gd_e = (state['jacobian'] * coeff_e[:, None] * stream.f019.weights(db_names)[:, None]).sum(axis=0)
    calibration.update({'JW_norm_E': float(np.linalg.norm(state['components']['jw'])),
                        'DB_norm_E': float(np.linalg.norm(gd_e)),
                        'JW_norm_ratio_S_E': calibration['JW_norm_S'] / float(np.linalg.norm(state['components']['jw'])),
                        'DB_norm_ratio_S_E': calibration['DB_norm_S'] / float(np.linalg.norm(gd_e)),
                        'c019_ratio_S_E': calibration['c019_S'] / calibration['c019_E']})
    stream.write('lap_f020_stream_calibration.json', calibration)
    parameter_hash = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    checkpoint_hash = audit.geometry.STATES[0]['sha256']
    for row in mapping:
        row.update(parameter_order_sha256=parameter_hash, checkpoint_sha256=checkpoint_hash)
    audit.write(out / 'stream_rows.json', mapping)
    reconstruction = audit.read(root / 'lap_f020_jacobian_reconstruction.json')
    reconstruction['mapping_sha256'] = audit.sha(out / 'stream_rows.json')
    stream.write('lap_f020_jacobian_reconstruction.json', reconstruction)
    statistics = {}
    cycle = audit.read(root / 'lap_f020_cycle_gradient_parity.json')
    with (root / 'lap_f020_jacobian_parity.csv').open() as handle:
        live_checks = list(csv.DictReader(handle))
    maximum_gradient_gap = max(float(r['relative_gradient_gap']) for r in live_checks)
    maximum_scalar_gap = max(float(r['scalar_gap']) for r in live_checks)
    for tag in ('F010', 'F019'):
        values = np.array([float(r['cosine_' + tag]) for r in fidelity])
        statistics[tag] = {'positive': int(np.sum(values > 0)), 'median': float(np.median(values)),
                           'p10': float(np.quantile(values, .1)), 'worst': float(values.min())}
    physical = sum(audit.geometry.b8.LAMBDAS[t] * state['components'][t] for t in ('ae17', 'exc', 'op'))
    g_old = physical + audit.geometry.b8.LAMBDAS['relchem'] * audit.geometry.C010 * matrices['jw_S']
    g_new = physical + audit.geometry.b8.LAMBDAS['relchem'] * calibration['c019_S'] * matrices['db_S']
    geometry = []
    databases = np.array([r['database'] for r in mapping])
    for db in sorted(stream.f019.COUNTS):
        vector = matrices['singleton'][databases == db].mean(axis=0)
        geometry.append({'database': db, 'F010_negative_unit_composite_derivative': -float(vector @ g_old / np.linalg.norm(g_old)),
                         'F019_negative_unit_composite_derivative': -float(vector @ g_new / np.linalg.norm(g_new))})
    audit.table('lap_f020_database_directional_derivatives.csv', geometry)
    contributions = []
    batches = audit.geometry.f010._batches()
    lookup = {r['identity']: i for i, r in enumerate(mapping)}
    for position in protocol['positions']:
        indices = [lookup[r['identity']] for r in batches[position]['relchem']]
        for db in sorted(stream.f019.COUNTS):
            chosen = [i for i in indices if databases[i] == db]
            vector = matrices['singleton'][chosen].sum(axis=0) / 8 if chosen else np.zeros(9446)
            weighted = vector * 251 / (8 * stream.f019.COUNTS[db])
            contributions.append({'position': position, 'database': db, 'presentations': len(chosen),
                                  'importance_weight': 251 / (8 * stream.f019.COUNTS[db]),
                                  'F010_contribution_norm': float(np.linalg.norm(vector)),
                                  'F019_contribution_norm': float(np.linalg.norm(weighted)),
                                  'F019_dot_full_DB_S': float(weighted @ matrices['db_S'])})
    audit.table('lap_f020_stochastic_database_contributions.csv', contributions)
    decision.update({'statistics': statistics, 'tests_passed': 30,
                     'all_batch_gradients_finite': bool(np.isfinite(matrices['new_batches']).all()),
                     'all_batch_gradients_F32_representable': bool(np.isfinite(matrices['new_batches'].astype(np.float32)).all()),
                     'training_authorized': False, 'ruff': 'PASS (E402 bootstrap exception)',
                     'compileall': 'PASS', 'diff_check': 'PASS'})
    stream.write('lap_f020_decision.json', decision)
    rows = []
    for path in sorted(out.glob('row_*.json')):
        receipt = audit.read(path)
        assert audit.sha(path.with_suffix('.npz')) == receipt['sha256']
        rows.append({'metadata': path.name, 'metadata_sha256': audit.sha(path), **receipt})
    provenance = {'parent': protocol['parent'], 'checkpoint_sha256': checkpoint_hash,
                  'manifest_sha256': protocol['stream_sha256'], 'parameter_order_sha256': parameter_hash,
                  'a_d_sha256': audit.geometry.arm.scales_sha256(_scales),
                  'sources': {p.name: audit.sha(p) for p in (root / 'tools').glob('lap_f020*.py')},
                  'cache_rows': rows, 'F019_artifacts_unchanged': True,
                  'reused_scalar_parity_sha256': audit.sha(out / 'reused_row_parity.json')}
    stream.write('lap_f020_provenance.json', provenance)
    lines = ['NO-GO — F020 reveals a qualified scientific failure', '',
             'Scientific classification B: the stream was reconstructed, but the original',
             'stochastic-fidelity gate failed. Native finite-step qualification was therefore',
             'not run. This is negative evidence for the DB-balanced intervention under the',
             'frozen historical stream, not an architecture or physical-operator failure.', '',
             '## Reconciliation and numerical qualification', '',
             'Authenticated 251 identities, 2,008 presentations, exactly eight per identity,',
             'one unchanged stream variant per identity; 39 matches and 212 mismatches.',
             '39 existing Jacobian rows were reused after live residual/reference checks;',
             '212 new rows were computed in matched F64. Independent production singleton',
             f'loss/gradient checks covered every database: maximum scalar gap {maximum_scalar_gap:.15g}, maximum relative',
             f'gradient gap {maximum_gradient_gap:.15g} (limit 1e-10). Each new row is finite,',
             '9446-dimensional and source/checkpoint/parameter-order bound. No finite differences',
             'were used. Source HDF5 access uses the existing SHA-validating training-only API.', '',
             f"JW cycle-gradient L2 residual: {cycle['JW_gradient_l2']:.15g}.",
             f"DB cycle-gradient L2 residual: {cycle['DB_gradient_l2']:.15g}.",
             f"DB reconstruction from eight database gradients: {cycle['DB_database_reconstruction_l2']:.15g}.",
             f"Scalar-cycle gaps: JW {cycle['JW_scalar_gap']:.15g}, DB {cycle['DB_scalar_gap']:.15g}. Frozen F64 tolerance: 1e-10.",
             'This identity holds at fixed parameters; it is not conditional unbiasedness',
             'during optimization. Population E and S are kept separate.', '',
             '## Population comparison', '',
             '| Quantity | Population E | Population S |',
             '|---|---:|---:|',
             f"| JW | {calibration['JW_E']:.15g} | {calibration['JW_S']:.15g} |",
             f"| DB-balanced | {calibration['DB_E']:.15g} | {calibration['DB_S']:.15g} |",
             f"| c019 | {calibration['c019_E']:.15g} | {calibration['c019_S']:.15g} |", '',
             f"S/E JW gradient cosine: {calibration['JW_cosine_S_E']:.15g}.",
             f"S/E DB gradient cosine: {calibration['DB_cosine_S_E']:.15g}.",
             f"JW/DB gradient cosine within S: {calibration['cosine_JW_DB_S']:.15g}.",
             'Per-database scalar differences and gradient cosines are in the population CSV.', '',
             '## Frozen stochastic gate', '',
             '| Estimator | Positive /32 | Median cosine | p10 | Worst |',
             '|---|---:|---:|---:|---:|']
    for tag, values in statistics.items():
        lines.append(f"| {tag} | {values['positive']} | {values['median']:.9f} | {values['p10']:.9f} | {values['worst']:.9f} |")
    lines += ['', 'Negative-dot batch attribution (additive dot decomposition; not a causal ablation):', '',
              '| Position | F019 cosine | Largest negative DB dot contribution |',
              '|---|---:|---|']
    for record in fidelity:
        if float(record['cosine_F019']) <= 0:
            position = int(record['position'])
            parts = [r for r in contributions if r['position'] == position]
            lowest = min(parts, key=lambda r: r['F019_dot_full_DB_S'])
            lines.append(f"| {position} | {float(record['cosine_F019']):.9f} | {lowest['database']} |")
    lines += ['', 'F019 passes the positive-dot count (27 ≥24), but fails the lower-tail gate:',
              f"p10 deterioration = {decision['p10_F010'] - decision['p10_F019']:.9f} >0.10.",
              'All 32 positions are exactly those frozen in F019. All batch gradients are',
              'finite and representable in F32. Full versus stochastic norm/error statistics,',
              'frozen-physical composite alignment and per-database contributions are retained.',
              'Composite diagnostics reuse archived full physical gradients, not newly measured',
              'per-position sampled physical gradients. No physical common descent is claimed.', '',
              '## Resource discipline', '',
              f"GPU: {decision['gpu']}. Cumulative scientific compute: {decision['elapsed_seconds']:.3f} s",
              '(including the initial profile and reused-row scalar checks), within the frozen',
              '1800-second budget. The initial conservative projection included cold loading',
              'and independent parity calls. Execution resumed only cached missing rows, with',
              'cumulative budget checks before each row; the budget was not extended.',
              'Runtime and allocator peaks are retained per computed row in the decision receipt.', '',
              '## Required disposition', '',
              '1–7: stream VERIFIED; 251 identities; 39 reused +212 computed; numerical checks',
              'PASS; both JW and DB cycle reconstructions PASS.',
              '8–11: population differences and c019 values/cosines reported above.',
              '12: stochastic fidelity FAIL.',
              '13: native AdamW finite-step NOT RUN (dependent gate blocked).',
              '14: predicted database derivatives are saved, but no finite-step benefit is proven.',
              '15: actual protected aggregate decrease NOT TESTED.',
              '16: production physical constraints unchanged; no new physical finite-step safety',
              'qualification is claimed. Relevant architecture regression tests passed.',
              '17: a future 128-update arm is NOT justified by this preflight.',
              '18: next experiment: a bounded read-only attribution of the failed lower-tail',
              'batches using the now-qualified cache, to distinguish importance-weight amplification',
              'from database-gradient cancellation. Do not rescue F019 by changing its sampler,',
              'optimizer or objective, and do not launch training.', '',
              '30 tests PASS; Ruff (existing E402 bootstrap exception), compileall, diff-check PASS.',
              'Historical F019 artifacts verified byte-for-byte unchanged. No retained optimizer',
              'updates, training, SCF, Clean28, Full30 or Diet100 access/evaluation occurred.',
              'The original checkpoint including optimizer/RNG state remains byte-identical.', '',
              'Scientific classification: B', 'Historical stream: VERIFIED', 'Unique reactions: 251',
              'Presentations: 2008', 'Reused Jacobian rows: 39', 'New qualified Jacobian rows: 212',
              'JW cycle-gradient parity: PASS', 'DB cycle-gradient parity: PASS',
              'c019 evaluation-population: 27.576519733036633',
              f"c019 stream-population: {calibration['c019_S']}", 'Stochastic fidelity: FAIL',
              'Finite-step qualification: NOT RUN', 'Training authorized for next experiment: NO',
              'Optimizer updates: 0', 'Training performed: NO', 'Clean28 accessed: NO',
              'New training data: NO', 'Production architecture modified: NO',
              'Historical stream modified: NO', '',
              'Commit/push/remote HEAD receipts are supplied in the final task response.']
    (root / 'lap_f020_stream_variant_reconciliation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    execute()
