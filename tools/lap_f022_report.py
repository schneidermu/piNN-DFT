"""F022 report from saved cached results only; no model evaluation."""
import csv
import json
import time

import lap_f022_scalarization as exp
import numpy as np


def rows(name):
    with (exp.ROOT / ('lap_f022_' + name + '.csv')).open() as f:
        return list(csv.DictReader(f))


def main():
    a = exp.audit
    cal = a.read(exp.ROOT / 'lap_f022_gradient_calibration.json')
    decision = a.read(exp.ROOT / 'lap_f022_decision.json')
    assert decision['classification'] == 'C' and decision['optimizer_updates'] == 0
    decision['next_experiment'] = ('Read-only cached affine-feasibility audit for simultaneous JW/legacy/operator descent '
        'with the original physical contribution fixed; determine necessary chemistry-contribution bounds, '
        'without choosing new coefficients or running training.')
    geometry = rows('full_gradient_geometry')
    database = rows('database_directions')
    batches = rows('batch_fidelity')
    summary = []
    for db in sorted(cal['factors']):
        for arm in ('F010', 'F022'):
            contributions = [json.loads(r['db_weighted_norm_contributions']).get(db, 0.) for r in batches if r['arm'] == arm]
            summary.append({'database': db, 'arm': arm, 'present_batches': sum(v > 0 for v in contributions),
                'mean_weighted_gradient_norm_contribution': float(np.mean(contributions)),
                'max_weighted_gradient_norm_contribution': max(contributions),
                'coefficient_per_penalty': exp.audit.geometry.C010 / exp.audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['scales'][db]
                if arm == 'F010' else cal['a'] / exp.audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['scales'][db] + cal['b']*cal['factors'][db]})
    a.table('lap_f022_database_batch_contributions.csv', summary)
    clock = a.read(exp.OUT / 'clock.json')
    wall = time.time() - clock['origin_unix']
    assert wall < 900  # Conservative upper bound: all elapsed time is counted as active.
    decision['total_wall_seconds_since_protocol_freeze'] = wall
    decision['active_scientific_compute_upper_bound_seconds'] = wall
    decision['compute_accounting'] = 'cached execute body timed; total frozen-workflow wall bounds all active work, including imports, tests and reporting'
    exp.write('decision', decision)
    budget = a.read(exp.ROOT / 'lap_f022_resource_budget.json')
    budget.update({'active_scientific_compute_upper_bound_seconds': wall, 'total_wall_seconds_since_protocol_freeze': wall,
        'read_only_preflight_limit_respected': True, 'active_limit_respected': True, 'wall_limit_respected': True,
        'GPU_evaluations': 0, 'training_and_endpoint_projection': 'NOT RUN; mandatory geometry gate failed before profiling'})
    exp.write('resource_budget', budget)
    provenance = a.read(exp.ROOT / 'lap_f022_provenance.json')
    assert a.sha(exp.f010.PARENT_CHECKPOINT) == exp.CHECKPOINT
    provenance.update({'checkpoint_preserved_after_preflight': True,
        'constraint_receipt_reused': 'lap_f021_phys4_resume_parity.json',
        'constraint_receipt_sha256': a.sha(exp.ROOT / 'lap_f021_phys4_resume_parity.json'),
        'protocol_sha256': a.sha(exp.ROOT / 'lap_f022_protocol.json'),
        'native_moment_restoration': 'NOT RUN; no optimizer was constructed or mutated',
        'mixed_historical_receipt': 'Only checkpoint and parameter-order fields consumed; no development score used',
        'GPU_evaluations': 0})
    exp.write('provenance', provenance)
    text = ['C — FIXED MIXTURE NO-GO', '',
        ('The single prespecified chemistry mixture is feasible in chemistry-only geometry, but the unchanged '
        'physical contributions reverse legacy descent. The complete F022 gradient fails the primary gate. '
        'No optimizer was created or stepped; native counterfactual, finite-step evaluation, training, full90 '
        're-evaluation and Clean28 were not run. No alternative alpha or coefficient was tested.'), '',
        (f"Population-S cosines: JW/legacy {cal['cosine_JW_legacy']:.12f}; JW/operator {cal['cosine_JW_operator']:.12f}; "
        f"legacy/operator {cal['cosine_legacy_operator']:.12f}."),
        f"Strict alpha interval ({cal['alpha_min']:.12f}, {cal['alpha_max']:.12f}); midpoint {cal['alpha']:.12f}.",
        (f"Frozen a={cal['a']:.15g}, b={cal['b']:.15g}, K={cal['K']:.15g}, D={cal['D']:.15g}. "
        f"Chemistry gradient norm {cal['chemistry_gradient_norm']:.12f} equals c010 × ||gJW,S|| (relative tolerance 1e-10)."), '',
        '| Target (S chemistry/full physical) | F010 dot | F022 chemistry-only dot | F022 physics-only dot | F022 full dot |',
        '|---|---:|---:|---:|---:|']
    for t in ('JW', 'legacy', 'MAE', 'ae17', 'exc', 'op'):
        selected = {r['direction']: float(r['positive_progress_dot']) for r in geometry if r['population'] == 'S' and r['target'] == t}
        text += [f"| {t} | {selected['F010']:.10g} | {selected['chemistry_only']:.10g} | {selected['physics_only']:.10g} | {selected['F022']:.10g} |"]
    text += ['', ('Positive dot predicts decrease along the negative gradient; negative dot predicts increase. '
        f"Legacy chemistry-only margin {decision['legacy_dot_chemistry']:.10f} is overwhelmed by physical contribution "
        f"{decision['legacy_dot_physics']:.10f}. Full-gradient cancellation ratio is {decision['composite_cancellation_ratio']:.10f}. "
        'Operator has positive full-gradient progress; this is a linear diagnostic, not finite-step safety qualification.'), '',
        '| Population-S database | MAE derivative along −G010 | MAE derivative along −G022 |', '|---|---:|---:|']
    for db in sorted(cal['factors']):
        selected = {r['direction']: float(r['MAE_negative_gradient_derivative']) for r in database if r['population'] == 'S' and r['database'] == db}
        text += [f"| {db} | {selected['F010']:.8f} | {selected['F022']:.8f} |"]
    text += ['', ('Negative derivative means improvement. ABDE4, PA8 and pTC13 still have adverse derivatives, '
        'although smaller than F010; EA13 retains less favorable predicted improvement. DBH76/NCCE31 improvement '
        'also weakens. IP13/MGAE109 adverse slopes shrink. These are derivatives, not observed endpoint changes. '
        'Population-E derivatives are separately saved in database_directions.csv and full_gradient_geometry.csv.'), '',
        '| Estimator (first 64 historical batches) | Positive dots | p10 cosine | Median cosine | Max norm ratio | Relative gradient MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, r in decision['fidelity'].items():
        text += [f"| {arm} | {r['positive_dot_count']}/64 | {r['p10_cosine']:.8f} | {r['median_cosine']:.8f} | {r['max_norm_ratio']:.8f} | {r['relative_gradient_MSE']:.8f} |"]
    text += ['', ('F022 stochastic fidelity deteriorates materially. ABDE4 contributes in 8/64 batches and its maximum '
        'component norm rises from 3.7850 to 81.1977; NCCE31 reaches 88.5181. PA8/IP13/pTC13 also have much '
        'larger components. Several contributions change and cancel; this does not establish single-database causality. '
        'Every database contribution, presentation frequency '
        'and combined penalty coefficient is retained in database_batch_contributions.csv; norms do not add because '
        'database gradients cancel. No rare-database sampler or post-hoc normalization was introduced.'), '',
        ('Provenance: parent checkpoint, inherited moments, manifest, canonical A_d, F020 S Jacobian/mapping/source '
        'records, F013/F015 E Jacobian and full90 physical vectors were authenticated. Parameter ordering is sorted '
        'and unique (9446 coordinates). Legacy factors come directly from optuna_joint FCHEM_DB_WEIGHTS × '
        'FREQ_WEIGHTS / MEAN_WEIGHT. E scalar, gradient norm and JW cosine reproduce the qualified F010 receipts; '
        'tests independently verify production singleton scalar/gradient parity and single combined scalar equivalence. '
        'This does not substitute a multi-reaction database RMSE for the mean singleton objective.'), '',
        ('S and E each use one existing variant per identity and are not conflated. Calibration uses S exclusively. '
        'All physical vectors are the same full17/full90 qualified gradients as F010, not PHYS4 prefix means or '
        'historical15 gradients. No live full-corpus gradient was recalculated.'), '',
        ('B8 constraint/restoration receipts are reused because no model or moment state was perturbed. Relevant '
        'operator/AdamW/chemistry regression tests passed: 22 tests, Ruff and compileall PASS. No new GPU anchor calculation is claimed. No pilot '
        'runtime projection or endpoint reserve is fabricated after the early stopping gate.'), '',
        (f"Measured cached execution body: {decision['active_compute_seconds']:.3f}s. Frozen-workflow wall: {wall:.3f}s; "
        'counting this entire interval as active gives a conservative upper bound below 900/5400/10800s limits. '
        'Source inspection preceded the frozen numerical execution; no GPU computation was performed.'), '',
        'Answers to the frozen questions:', '',
        '1–5. Authentication PASS; exact S cosines, strict interval, midpoint and fixed a/b are above.',
        '6. NO: full G descends JW but increases legacy. Chemistry-only direction passes both.',
        '7. Operator linear progress is positive; actual native/finite-step safety NOT RUN.',
        '8. Inherited-moment native counterfactual NOT RUN because the earlier mandatory gate failed.',
        '9. YES: stochastic alignment and relative MSE deteriorate (table above).',
        '10–11. Training not authorized; 0/64 optimizer updates.',
        '12–15. No final endpoint exists; no observed JW, legacy, MAE or database improvement claimed. Directional comparisons are above.',
        '16–18. P536 eligibility NOT RUN; Clean28 NOT EVALUATED; historical best NOT TESTED.',
        '19. Cached execution and conservative active/wall bounds are above; GPU evaluations zero.',
        '20. Close this fixed mixture. One next experiment: ' + decision['next_experiment'], '',
        ('This failure applies to the frozen F022 mixture with unchanged physics coefficients. It does not establish '
        'an R0 representation ceiling or impossibility of all fixed scalarizations. No rescue experiment was launched.'), '',
        'Scientific classification: C', 'Architecture: R0', 'Scalar objective: fixed F022',
        'Curriculum: NO', 'Gradient clipping: NO', 'Dynamic task weighting: NO',
        f"Alpha: {cal['alpha']}", f"Chemistry coefficient a: {cal['a']}", f"Chemistry coefficient b: {cal['b']}",
        'Cached-gradient preflight: FAIL', 'Native-step preflight: NOT RUN', 'Training launched: NO', 'Optimizer updates: 0/64',
        'Final JW: unavailable', 'Final legacy relchem: unavailable', 'Final Minnesota MAE: unavailable',
        'P536 relchem ratio: unavailable', 'P536 AE17 ratio: unavailable', 'P536 Exc ratio: unavailable', 'P536 operator ratio: unavailable',
        'Physical eligibility: NOT RUN', 'Clean28: NOT EVALUATED', 'Historical best beaten: NOT TESTED',
        f'Active scientific compute: <= {wall:.3f} seconds (conservative bound)', f'Total frozen-workflow wall time: {wall:.3f} seconds',
        'Budget respected: YES', 'Optimizer reset: NO', 'Architecture modified: NO', 'New training data: NO', 'SCF executed: NO',
        'F022 commit: commit containing this report; returned after Git verification',
        'Git push / verified remote HEAD: returned after push, avoiding a self-referential commit hash']
    (exp.ROOT/'lap_f022_report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
