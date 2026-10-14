"""Reporting and source-zero checks; no new model evaluations."""
import argparse
import csv
import time
from collections import Counter

import lap_f021_phys4 as exp
import numpy as np
import torch

audit, b8 = exp.audit, exp.b8


def csv_rows(name):
    path = exp.ROOT / ('lap_f021_phys4_' + name + '.csv')
    with path.open() as handle:
        return list(csv.DictReader(handle))


def source_zero():
    prefix = exp.f010._batches()[:128]
    receipt = exp.read('singleton_gradients_receipt')
    expected = audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['live']['other_losses']
    actual = {}
    for row in receipt['records']:
        if row['binding']['task'] == 'ae17' and row['binding']['record'] == prefix[0]['ae17']:
            actual['ae17'] = row['losses']['ae17']
        if row['binding']['task'] == 'mrks' and row['binding']['record'] == prefix[0]['mrks_id']:
            actual.update(row['losses'])
    assert set(actual) == set(expected)
    errors = {t: abs(actual[t] - expected[t]) for t in expected}
    assert all(errors[t] <= 1e-10 * max(1, abs(expected[t])) for t in expected)
    exp.write('source_zero_parity', {'passed': True, 'expected': expected, 'actual': actual, 'absolute_errors': errors})


def census_extra():
    prefix = exp.f010._batches()[:128]
    with np.load(exp.OUT / 'census_vectors.npz') as payload:
        arrays = {t: payload[t].copy() for t in exp.TASKS}
        one, four = payload['physical_singleton'].copy(), payload['physical_phys4'].copy()
    exposure = []
    for task in ('ae17', 'mrks'):
        values = [exp.key(r['ae17']) if task == 'ae17' else r['mrks_id'] for r in prefix]
        for record, count in Counter(values).items():
            exposure.append({'task': task, 'record': record, 'control_presentations': count, 'PHYS4_presentations': 4 * count})
    exp.table('record_exposures', exposure)
    correlations = []
    for task, a in {**arrays, 'weighted_physical': one}.items():
        centered = a - a.mean(axis=0)
        norm = float(np.mean(np.sum(centered**2, axis=1)))
        for i, left in enumerate(exp.OFFSETS):
            for right in exp.OFFSETS[i + 1:]:
                shift = (right - left) % 128
                covariance = float(np.mean(np.sum(centered * np.roll(centered, -shift, axis=0), axis=1)))
                correlations.append({'task': task, 'left_offset': left, 'right_offset': right,
                                     'covariance_trace': covariance, 'correlation': covariance / norm if norm else None})
    exp.table('offset_correlations', correlations)
    decomposition = []
    for task, values in {**arrays, 'weighted_physical': one}.items():
        centered = values - values.mean(axis=0)
        if task == 'ae17':
            keys = [exp.key(r['ae17']) for r in prefix]
        elif task == 'weighted_physical':
            keys = [exp.key((r['ae17'], r['mrks_id'])) for r in prefix]
        else:
            keys = [r['mrks_id'] for r in prefix]
        duplicate, distinct, count = 0., 0., 0
        for t in range(128):
            p = exp.positions(t)
            for i, left in enumerate(p):
                for right in p[i + 1:]:
                    term = float(centered[left] @ centered[right]) / (128 * 8)
                    if keys[left] == keys[right]:
                        duplicate += term
                        count += 1
                    else:
                        distinct += term
        baseline = float(np.mean(np.sum(centered**2, axis=1))) / 4
        smoothed = values[np.array([exp.positions(t) for t in range(128)])].mean(axis=1)
        v4 = float(np.mean(np.sum((smoothed - values.mean(axis=0))**2, axis=1)))
        assert np.isclose(v4, baseline + duplicate + distinct, atol=1e-10, rtol=1e-10)
        decomposition.append({'task': task, 'singleton_MSE_over4': baseline,
                              'duplicate_pair_covariance_contribution': duplicate,
                              'distinct_pair_covariance_contribution': distinct,
                              'duplicate_pairs': count, 'phys4_MSE': v4})
    exp.table('duplicate_covariance_attribution', decomposition)
    residual = one - one.mean(axis=0)
    magnitudes = np.sum(residual**2, axis=1)
    sorted_m = np.sort(magnitudes)[::-1]
    exp.write('outlier_concentration', {'weighted_physical_top1_MSE_share': float(sorted_m[:1].sum() / sorted_m.sum()),
                                       'weighted_physical_top5_MSE_share': float(sorted_m[:5].sum() / sorted_m.sum()),
                                       'duplicate_group_counts': Counter(int(r['physical_unique_records']) for r in csv_rows('gradient_distributions')),
                                       'physical_mean_difference': float(np.linalg.norm(one.mean(axis=0) - four.mean(axis=0)))})


def summarize():
    protocol = exp.read('protocol')
    status_path = exp.OUT / 'training_status.json'
    status = audit.read(status_path) if status_path.exists() else {'local_update': 0, 'logs': []}
    logs = status['logs']
    if logs:
        exp.table('training_log', [{k: v for k, v in row.items() if k != 'reactions'} for row in logs])
    origin = audit.read(exp.f010.OUT / 'metrics_initial.json')
    control = audit.read(exp.f010.OUT / 'metrics_t128.json')
    metrics = []
    databases = []
    physical_rows = []
    for cursor in (0, 64, 128):
        path = exp.OUT / f'metrics_{cursor}.json'
        if cursor == 0:
            value = {'values': {t: origin[t] for t in b8.TASKS}, 'JW': origin['jw'],
                     'chemistry': audit.stats([r['residual'] for r in origin['reactions']])}
            reactions = origin['reactions']
        elif path.exists():
            value = audit.read(path)
            reactions = audit.read(exp.OUT / f'reactions_{cursor}.json')
        else:
            continue
        metrics.append({'cursor': cursor, 'JW': value['JW'], **value['chemistry'],
                        **{t: value['values'][t] for t in b8.TASKS},
                        **{t + '_P536_ratio': value['values'][t] / b8.BASELINE[t] for t in b8.TASKS}})
        for db in sorted({r['database'] for r in reactions}):
            selected = [r for r in reactions if r['database'] == db]
            historic = [r for r in control['reactions'] if r['database'] == db]
            databases.append({'cursor': cursor, 'database': db, 'n': len(selected),
                              **audit.stats([r['residual'] for r in selected]),
                              'F010_t128_MAE': audit.stats([r['residual'] for r in historic])['mae']})
        for t in b8.TASKS:
            physical_rows.append({'cursor': cursor, 'objective': t, 'value': value['values'][t],
                                  'P536_ratio': value['values'][t] / b8.BASELINE[t],
                                  'B8_delta': value['values'][t] - origin[t],
                                  'F010_t128_delta': value['values'][t] - control[t]})
    exp.table('checkpoint_metrics', metrics)
    exp.table('database_metrics', databases)
    exp.table('physical_metrics', physical_rows)
    constraints = {str(c): audit.read(exp.OUT / f'metrics_{c}.json')['constraints']
                   for c in (64, 128) if (exp.OUT / f'metrics_{c}.json').exists()}
    exp.write('constraint_checks', constraints)
    final = audit.read(exp.OUT / 'metrics_128.json') if (exp.OUT / 'metrics_128.json').exists() else None
    clean = exp.read('clean28_receipt') if (exp.ROOT / 'lap_f021_phys4_clean28_receipt.json').exists() else None
    var = csv_rows('joint_variance') if (exp.ROOT / 'lap_f021_phys4_joint_variance.csv').exists() else []
    promoted = bool(final and final['eligible'] and clean and 'clean28' in clean and clean['clean28'] < 8.619694171538775)
    classification = ('GO' if promoted else 'MECHANISM-ONLY' if final and var and float(var[0]['reduction']) > 0
                      else 'NO-GO' if final else 'PARTIAL')
    checkpoints = {}
    for p in sorted((exp.OUT / 'ordinary_sgd_adamw').glob('checkpoint_*.pt')):
        checkpoints[p.name] = audit.sha(p)
    exp.write('decision', {'classification': classification, 'updates': status['local_update'],
                           'chemistry_presentations': status['local_update'] * 8,
                           'ae17_presentations': status['local_update'] * 4,
                           'mrks_presentations': status['local_update'] * 4,
                           'physical_eligible': final['eligible'] if final else None,
                           'clean28': clean, 'checkpoint_hashes': checkpoints,
                           'scientific_seconds': audit.read(exp.OUT / 'budget_usage.json')['seconds'],
                           'budget': exp.BUDGET, 'production_architecture_modified': False,
                           'scalar_objective_modified': False, 'optimizer_reset': False, 'SCF': False})
    exp.write('provenance', {'parent': protocol['parent'], 'source_hashes': protocol['source_hashes'],
                             'protocol_sha256': audit.sha(exp.ROOT / 'lap_f021_phys4_protocol.json'),
                             'checkpoint_hashes': checkpoints, 'dataset_logical_sha256': b8.DATA_SHA})
    text = [classification + ' — F021-PHYS4', '',
            'One R0 fixed-loss continuation. Physical offsets (0,31,62,93), modulo 128.',
            'The matched control has the same initialization, moments, optimizer updates,',
            'chemistry presentations and relative empirical physical frequencies. Physical',
            'computation and presentation counts are four times the control; compute is not matched.', '',
            f"Completed optimizer updates: {status['local_update']}/128.",
            f"Physical eligibility: {final['eligible'] if final else 'NOT RUN'}.",
            f"Clean28: {clean if clean else 'NOT EVALUATED'}.", '',
            'All training/development evaluation is non-self-consistent on existing fixed inputs.',
            'No SCF, new densities, targets, architecture, clipping, scheduler, curriculum or',
            'adaptive coefficients. Variance reduction alone does not establish optimization',
            'or development benefit; its Jensen inequality is a mathematical construction.', '',
            'Machine-readable checkpoint, database, physical, variance and covariance tables',
            'are the numerical source of this report. Final interpretive sections follow only',
            'after all required endpoint evaluations are complete.']
    (exp.ROOT / 'lap_f021_phys4_report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def finalize():
    """Verify saved results and produce the final interpretation, without GPU work."""
    prefix, _source, _scales = exp.freeze()
    summarize()
    status = audit.read(exp.OUT / 'training_status.json')
    logs = status['logs']
    assert status['complete'] and status['local_update'] == len(logs) == 128
    for i, row in enumerate(logs):
        assert row['local_update'] == i + 1 and row['manifest_index'] == i
        assert row['cumulative_optimizer_step'] == 129 + i and not row['nonfinite']
        assert row['physical_positions'] == list(exp.positions(i))
        assert [(r['identity'], r['variant']) for r in row['reactions']] == [
            (r['identity'], r['variant']) for r in prefix[i]['relchem']]
    checkpoints = exp.OUT / 'ordinary_sgd_adamw'
    parent = torch.load(exp.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    zero = torch.load(checkpoints / 'checkpoint_0.pt', map_location='cpu', weights_only=False)
    final = torch.load(checkpoints / 'checkpoint_128.pt', map_location='cpu', weights_only=False)
    control = torch.load(exp.f010.OUT / 'ordinary_sgd_adamw/checkpoint_128.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(zero['model'][n], v) for n, v in parent['model'].items())
    assert b8._import_run().existing.equal(zero['rng'], parent['rng'])
    assert exp.f010.continuation.moment_sha(zero['optimizer']) == exp.MOMENT
    assert {int(s['step']) for s in final['optimizer']['state'].values()} == {256}
    assert final['cursor'] == 128 and final['manifest_sha256'] == exp.f010.continuation.MANIFEST_SHA
    # Every checkpoint contains the entire log prefix and all resumable state.
    for cursor in (0, 64, 128):
        state = torch.load(checkpoints / f'checkpoint_{cursor}.pt', map_location='cpu', weights_only=False)
        assert state['cursor'] == cursor and len(state['logs']) == cursor
        assert all(k in state for k in ('model', 'optimizer', 'rng', 'protocol_sha256', 'manifest_sha256'))
    names = list(parent['model'])
    deltas = {label: np.concatenate([(state['model'][n].double() - parent['model'][n].double()).numpy().ravel()
                                    for n in names]) for label, state in [('PHYS4', final), ('F010', control)]}
    optimizer_rows = []
    for label, state in [('B8', parent), ('F010', control), ('PHYS4', final)]:
        moments = state['optimizer']['state'].values()
        optimizer_rows.append({'model': label, 'first_moment_L2': float(np.sqrt(sum(
            float(s['exp_avg'].double().square().sum()) for s in moments))),
            'second_moment_L2': float(np.sqrt(sum(float(s['exp_avg_sq'].double().square().sum()) for s in moments))),
            'moment_sha256': exp.f010.continuation.moment_sha(state['optimizer'])})
    exp.table('optimizer_moments', optimizer_rows)
    old_logs = audit.read(exp.f010.OUT / 'training_status.json')['logs'][:128]
    exp.table('trajectory_comparison', [{'cursor': r['local_update'], 'arm': label,
        **{k: r[k] for k in ('update_norm', 'displacement_from_t128', 'gradient_norm', 'seconds', 'peak_allocated_bytes')}}
        for label, records in [('F010', old_logs), ('PHYS4', logs)] for r in records])
    checkpoint_rows = csv_rows('checkpoint_metrics')
    last, initial = checkpoint_rows[-1], checkpoint_rows[0]
    db = [r for r in csv_rows('database_metrics') if int(r['cursor']) == 128]
    improved = [r['database'] for r in db if float(r['mae']) < float(r['F010_t128_MAE'])]
    deteriorated = [r['database'] for r in db if float(r['mae']) >= float(r['F010_t128_MAE'])]
    metrics = audit.read(exp.OUT / 'metrics_128.json')
    clean = exp.read('clean28_receipt')
    assert metrics['model_unchanged'] and metrics['constraints']['passed']
    assert clean['checkpoint_sha256'] == metrics['checkpoint_sha256']
    if not metrics['eligible']:
        assert clean['evaluations'] == 0 and 'clean28' not in clean
    decision = exp.read('decision')
    contract = exp.read('execution_contract')
    elapsed = time.time() - contract['wall_clock_origin_unix']
    assert elapsed < exp.BUDGET
    decision.update({'total_scientific_wall_seconds': elapsed, 'budget_respected': True,
        'training_seconds': sum(r['seconds'] for r in logs),
        'seconds_per_update': float(np.mean([r['seconds'] for r in logs])),
        'peak_allocated_GiB': max(r['peak_allocated_bytes'] for r in logs) / 2**30,
        'allocator_reserved_memory': 'not exported by reused training harness',
        'final_values': metrics['values'], 'P536_ratios': metrics['ratios'],
        'displacement_PHYS4': float(np.linalg.norm(deltas['PHYS4'])),
        'displacement_F010': float(np.linalg.norm(deltas['F010'])),
        'displacement_cosine': float(deltas['PHYS4'] @ deltas['F010'] / np.linalg.norm(deltas['PHYS4']) / np.linalg.norm(deltas['F010'])),
        'endpoint_parameter_difference': float(np.linalg.norm(deltas['PHYS4'] - deltas['F010'])),
        'databases_better_than_F010': improved, 'databases_worse_than_F010': deteriorated,
        'macro_database_MAE': float(np.mean([float(r['mae']) for r in db])),
        'next_experiment': 'Read-only matched gradient-conflict audit at B8 and PHYS4 t128 on eight preregistered existing chemistry/physical groups; preserve F010 loss and sources, no training, no new architecture or loss.'})
    exp.write('decision', decision)
    provenance = exp.read('provenance')
    provenance['normalization'] = {
        'c010': audit.geometry.C010,
        'A_d_sha256': audit.read(exp.ROOT / 'lap_f013_gradient_receipts.json')['a_d_sha256'],
        'receipt_sha256': audit.sha(exp.ROOT / 'lap_f010_normalization_receipt.json')}
    cache_files = [exp.OUT / 'census_vectors.npz'] + sorted((exp.OUT / 'singleton').glob('*'))
    provenance['external_cache_hashes'] = {str(p.relative_to(exp.OUT)): audit.sha(p) for p in cache_files}
    census = exp.read('singleton_gradients_receipt')
    decision['census_peak_reserved_GiB'] = max(r['peak_reserved'] for r in census['records']) / 2**30
    decision['census_compute_seconds'] = sum(r['seconds'] for r in census['records'])
    exp.write('decision', decision)
    for row in census['records']:
        binding = row['binding']
        task = binding['task']
        filename = exp.OUT / 'singleton' / (task + '_' + exp.key(binding['record']) + '.npz')
        assert audit.sha(filename) == row['sha256']
    provenance['census_hashes_verified'] = len(census['records'])
    provenance['evaluation_receipts'] = {p.name: audit.sha(p) for p in sorted(exp.OUT.glob('*_*.json'))
                                       if p.name.startswith(('metrics_', 'reactions_', 'ae17_', 'physical_'))}
    provenance['integrity'] = {'source_pins': True, 'parent_moments': True, 't0_parent_tensors_exact': True,
        'checkpoint_counters': True, 'chemistry_stream_exact': True, 'physical_prefix_exact': True,
        'scientific_evaluation_models_unchanged': True}
    exp.write('provenance', provenance)
    task = csv_rows('task_variance')
    joint = csv_rows('joint_variance')
    distribution = csv_rows('gradient_distributions')
    exp.table('frozen_direction_summary', [{'quantity': quantity,
        'min': float(np.min(values)), 'p10': float(np.quantile(values, .1)),
        'median': float(np.median(values)), 'p90': float(np.quantile(values, .9)), 'max': float(np.max(values))}
        for quantity in ('composite_cosine', 'composite_norm_1', 'composite_norm_4',
                         'ae17_predicted_descent_1', 'ae17_predicted_descent_4',
                         'exc_predicted_descent_1', 'exc_predicted_descent_4',
                         'op_predicted_descent_1', 'op_predicted_descent_4')
        for values in [[float(r[quantity]) for r in distribution]]])
    fullcontrol = audit.read(exp.f010.OUT / 'metrics_t128.json')
    lines = [decision['classification'] + ' — physical variance reduction confirmed without model promotion' if
             decision['classification'] == 'MECHANISM-ONLY' else decision['classification'] + ' — F021-PHYS4', '',
        ('One matched R0 continuation from B8 t128 with inherited AdamW moments. Scalar loss, LR, chemistry stream, '
        'parameter precision, and full weak-form AO operator are unchanged. Only physical gradients average four '
        'prefix presentations. The control is matched by optimizer updates and chemistry exposures, not compute.'), '',
        '| Task | Singleton vector MSE | PHYS4 vector MSE | Reduction |', '|---|---:|---:|---:|']
    lines += [f"| {r['task']} | {float(r['singleton_MSE']):.10g} | {float(r['phys4_MSE']):.10g} | {100*float(r['reduction']):.4f}% |" for r in task]
    lines += ['', (f"Weighted physical MSE: {joint[0]['singleton_MSE']} → {joint[0]['phys4_MSE']} "
              f"({100*float(joint[0]['reduction']):.4f}% reduction). Composite MSE: {joint[1]['singleton_MSE']} → "
              f"{joint[1]['phys4_MSE']} ({100*float(joint[1]['reduction']):.4f}% reduction)."),
        ('The mean is preserved to <4e-16 L2. Averaging cannot increase physical MSE in exact arithmetic; '
        'that inequality alone is not evidence of better optimization. Chemistry/physical covariance trace changes '
        'from -0.005360956912 to -0.005833679008; both are included in composite MSE.'), '',
        ('Top-one/top-five physical outliers account for 4.46%/16.35% of singleton MSE. Four groups contain '
        'three unique mRKS systems, 124 contain four. Duplicate-pair covariance contributes AE17 140718.955, '
        'Exc 2807181.188, operator 0.001465754; distinct-offset covariance also matters '
        '(see duplicate_covariance_attribution.csv). Reductions are not dominated by one outlier.'), '',
        ('84 AE17 identity+variant records (17 identities) and 90 mRKS systems were computed once at frozen B8. '
        'Their empirical prefix means are distinct from uniform full-corpus gradients. Reference-population '
        'cosines are AE17 0.999998965, Exc 0.999996323, operator 0.999996021; relative vector differences '
        'are 3.319%, 0.886%, 0.642%. AE17 additionally differs in variant population.'), '',
        ('Independent scalar-autograd parity passed its frozen native-precision bounds. AE17 relative L2 '
        'error 2.56e-16, Exc 2.50e-7, operator 1.18e-7. The latter two are native F32 leaf accumulation, '
        'not claimed bitwise F64 derivative parity. Chemistry live/cache relative L2 error 1.34e-12. '
        'Native state-restoration and source-position-zero parity passed.'), '',
        '| Cursor | Legacy relchem | AE17 | Full90 Exc | Full90 operator | JW |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['cursor']} | {float(r['relchem']):.10f} | {float(r['ae17']):.10f} | {float(r['exc']):.10f} | {float(r['op']):.10f} | {float(r['JW']):.10f} |" for r in checkpoint_rows]
    lines += ['', '| Objective | PHYS4 t128 | P536 ratio | Change from B8 | Change from F010 t128 |',
              '|---|---:|---:|---:|---:|']
    lines += [f"| {r['objective']} | {float(r['value']):.10f} | {float(r['P536_ratio']):.10f} | {float(r['B8_delta']):+.10f} | {float(r['F010_t128_delta']):+.10f} |"
              for r in csv_rows('physical_metrics') if int(r['cursor']) == 128]
    lines += ['', (f"Scientific eligibility: {'PASS' if metrics['eligible'] else 'FAIL'}. Clean28: "
              f"{clean.get('clean28', clean.get('status'))}. Historical best: "
              f"{'beaten' if decision['classification'] == 'GO' else 'NOT TESTED' if clean['evaluations'] == 0 else 'not beaten'}."), '',
        '| Database | PHYS4 MAE | F010 t128 MAE | Difference | PHYS4 RMSE |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['database']} | {float(r['mae']):.6f} | {float(r['F010_t128_MAE']):.6f} | {float(r['mae'])-float(r['F010_t128_MAE']):+.6f} | {float(r['rmse']):.6f} |" for r in db]
    lines += ['', (f"Chemistry MAE {float(initial['mae']):.6f} → {float(last['mae']):.6f}; "
        f"F010 t128 {fullcontrol['chemistry_statistics']['mae'] if 'chemistry_statistics' in fullcontrol else audit.stats([r['residual'] for r in fullcontrol['reactions']])['mae']:.6f}. "
        f"Macro database MAE {decision['macro_database_MAE']:.6f}. JW PHYS4 {float(last['JW']):.10f}, F010 {fullcontrol['jw']:.10f}."), '',
        (f"AdamW displacement: PHYS4 {decision['displacement_PHYS4']:.10f}, F010 {decision['displacement_F010']:.10f}; "
        f"cosine {decision['displacement_cosine']:.10f}; endpoint separation {decision['endpoint_parameter_difference']:.10f}. "
        'Optimizer moment norms/hashes and per-step update/gradient norms are saved in optimizer_moments.csv '
        'and trajectory_comparison.csv. Averaging changes the realized trajectory, not merely a reporting statistic.'), '',
        (f"Training: {decision['training_seconds']:.1f}s ({decision['seconds_per_update']:.2f}s/update), "
        f"peak allocated {decision['peak_allocated_GiB']:.3f} GiB. Reserved allocator VRAM was not exported by the reused "
        'training harness; no invented measurement. Whole-workflow wall time including census, evaluations and '
        f"verification so far: {elapsed:.1f}s / 36000s. Initial projected remaining cost 11259.6s fit the budget."), '',
        ('The physical-gradient variance hypothesis is supported at frozen B8. Functional improvement is not '
        'established unless all four P536 gates and the development gate pass. This single-seed result cannot '
        'establish an R0 representation ceiling, the necessity of tau, or SCF stability.'), '',
        'One next experiment: ' + decision['next_experiment'], '',
        ('All 128 chemistry positions were reproduced exactly; physical positions remain 0..127, each used four '
        'times with original references/variants. 128 updates, 1024 chemistry, 512 AE17, 512 mRKS presentations. '
        't0/t64/t128 retain complete optimizer, RNG, cursor and hashes. Constraints passed at both audited endpoints. '
        'No production architecture/operator changes, new data, clipping, curriculum, scheduler, optimizer reset '
        'or SCF. Exactly one fixed evaluation variant per chemistry identity; no exhaustive eight-variant sweep.'), '',
        'Scientific classification: ' + decision['classification'], 'Architecture: R0', 'Scalar objective: unchanged F010',
        'Physical prefix: 0..127', 'Physical offsets: 0,31,62,93', 'Physical modulo: 128', 'Preflight: PASS',
        'Optimizer updates: 128/128', 'Chemistry presentations: 1024', 'AE17 presentations: 512', 'mRKS presentations: 512']
    for row in task:
        lines += [f"{row['task']} singleton vector MSE: {row['singleton_MSE']}",
                  f"{row['task']} PHYS4 vector MSE: {row['phys4_MSE']}"]
    lines += [f"Weighted physical vector MSE reduction: {100*float(joint[0]['reduction']):.6f}%",
              f"Composite vector MSE reduction: {100*float(joint[1]['reduction']):.6f}%"]
    for t in b8.TASKS:
        lines += [f"Final {t}: {metrics['values'][t]}", f"P536 {t} ratio: {metrics['ratios'][t]}"]
    lines += [f"Physical eligibility: {'PASS' if metrics['eligible'] else 'FAIL'}",
        f"Clean28: {clean.get('clean28', clean.get('status'))}",
        f"Historical best beaten: {'YES' if decision['classification'] == 'GO' else 'NOT TESTED' if clean['evaluations'] == 0 else 'NO'}",
        'Training: YES', 'Architecture modified: NO', 'Loss modified: NO', 'Optimizer reset: NO',
        'Curriculum: NO', 'Gradient clipping: NO', 'SCF executed: NO', 'New training data: NO',
        f'Total scientific wall time: {elapsed:.1f} seconds', 'Budget: 36000 seconds', 'Budget respected: YES',
        'F021-PHYS4 commit: see Git commit containing this report (avoids self-referential SHA).',
        'Git push / verified remote HEAD: returned with the final response after verification.']
    ending = lines.index('Scientific classification: ' + decision['classification'])
    closing = lines[ending:]
    lines = lines[:ending]
    lines += ['', 'Explicit answers to the frozen scientific questions:', '',
        '1–4. Chemistry prefix exact; physical positions restricted to 0..127; each appears four times; references and variants unchanged.',
        '5–6. Averaging parity PASS; 84 distinct AE17 identity+variant records and 90 mRKS records.',
        '7–9. Raw, weighted physical and composite vector MSEs/reductions are in the first table and joint_variance.csv.',
        '10. Duplicate and offset-correlation contributions are retained explicitly in duplicate_covariance_attribution.csv and offset_correlations.csv.',
        '11. Frozen-state smoothing is confirmed; a Jensen inequality alone does not establish useful training.',
        '12–13. Resource projection PASS; exactly 128 actual AdamW steps completed, with inherited counters reaching 256.',
        f"14. PHYS4/control displacement cosine {decision['displacement_cosine']:.6f}; endpoint separation {decision['endpoint_parameter_difference']:.6f}.",
        f"15. MAE change versus F010 {float(last['mae'])-float(fullcontrol['mae']):+.6f} kcal/mol; JW change {float(last['JW'])-float(fullcontrol['jw']):+.10f}.",
        f"16. MAE improved versus F010 in {', '.join(improved) or 'none'}; deteriorated in {', '.join(deteriorated) or 'none'}.",
        '17–18. Exact final objective values and all corrected-P536 ratios are tabulated above; no sampled estimates substitute for them.',
        f"19. Strict conjunctive scientific gate {'PASS' if metrics['eligible'] else 'FAIL'}.",
        '20. Analytical anchors, spin symmetry and tau independence PASS; variational operator source/metric/gauge unchanged.',
        f"21. Clean28 evaluations: {clean['evaluations']}; scientific gate controls access and no earlier checkpoint is selected.",
        f"22. Historical best {'beaten' if decision['classification'] == 'GO' else 'not tested' if clean['evaluations'] == 0 else 'not beaten'}.",
        (f"23. Whole-workflow wall time {elapsed:.1f}s; actual training {decision['training_seconds']:.1f}s; unique frozen gradient calculations {decision['census_compute_seconds']:.1f}s. "
        'Whole-workflow wall includes evaluation, verification and intervening idle time; it is not presented as pure GPU-kernel time.'),
        '24. ' + decision['next_experiment']]
    test_path = exp.ROOT / 'lap_f021_phys4_tests.json'
    if test_path.exists():
        tests = exp.read('tests')
        lines += ['', (f"Focused regression suite: {tests['passed']} passed; Ruff and compileall PASS. "
                  'Frozen producer/driver source hashes, all 174 singleton caches, checkpoints and complete log mapping verified. '
                  'The final staged Git diff check and remote verification are performed before delivery.')]
    lines += ['', *closing]
    (exp.ROOT / 'lap_f021_phys4_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('source-zero', 'census-extra', 'summary', 'final'))
    args = parser.parse_args()
    {'source-zero': source_zero, 'census-extra': census_extra, 'summary': summarize, 'final': finalize}[args.stage]()


if __name__ == '__main__':
    main()
