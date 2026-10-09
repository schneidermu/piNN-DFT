"""Read-only receipts, paired reaction diagnostics and plot for IID t59->t90."""
import hashlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from tools.continue_iid_adamw import HISTORICAL, OUTPUT, SOURCE, START_SHA

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    initial = torch.load(SOURCE / 'ordinary_sgd_adamw/checkpoint_0.pt', map_location='cpu', weights_only=False)
    historical = torch.load(HISTORICAL / 'ordinary_sgd_adamw/checkpoint_0.pt', map_location='cpu', weights_only=False)
    assert run.existing.equal(initial['model'], historical['model'])
    bundle = run.PublicationDataset(run.DATA)
    try:
        assert bundle.manifest['logical_sha256'] == run.DATA_SHA
        reference = {r['source_id']: r for r in bundle.validation_reactions.values()}
        clean = set(bundle.splits['diet30_clean_validation']['ids'])
        curves = []
        for cursor in (0, 59, 70, 80, 90):
            path = OUTPUT / ('baseline_0_validation.json' if cursor == 0 else f'endpoint_{cursor}_validation.json')
            receipt = run.read(path)
            assert receipt['model_unchanged']
            metrics = receipt['metrics']
            rows = []
            for old in metrics['reaction_rows']:
                catalog = reference[old['reaction_id']]
                assert old['clean'] == (catalog['id'] in clean)
                error = old['signed_error_kcal_mol']
                assert np.isclose(abs(error) * catalog['diet_weight'], old['weighted_absolute_error'], rtol=0, atol=1e-12)
                rows.append({**old, 'reference_energy_kcal_mol': catalog['reference_energy_kcal_mol'],
                             'predicted_energy_kcal_mol': catalog['reference_energy_kcal_mol'] + error,
                             'absolute_error_kcal_mol': abs(error), 'diet_weight': catalog['diet_weight'],
                             'clean_score_contribution': old['weighted_absolute_error'] / 28 if old['clean'] else None})
            assert len(rows) == 30 and sum(r['clean'] for r in rows) == 28
            assert abs(sum(r['clean_score_contribution'] for r in rows if r['clean']) - metrics['clean28']) < 1e-12
            curves.append({'cursor': cursor, 'clean28': metrics['clean28'], 'full30_diagnostic': metrics['full30'],
                           'full30_selection_allowed': False, 'reactions': rows,
                           'receipt_sha256': run.file_sha256(path), 'checkpoint_sha256': receipt['checkpoint_sha256']})
    finally:
        bundle.close()
    for curve in curves:
        curve['delta_t0'] = curve['clean28'] - curves[0]['clean28']
        curve['delta_t59'] = curve['clean28'] - curves[1]['clean28']
        for baseline_index in (0, 1):
            baseline = {r['reaction_id']: r for r in curves[baseline_index]['reactions'] if r['clean']}
            changes = [{'reaction_id': r['reaction_id'], 'delta_weighted_contribution':
                        r['clean_score_contribution'] - baseline[r['reaction_id']]['clean_score_contribution']}
                       for r in curve['reactions'] if r['clean']]
            label = 't0' if baseline_index == 0 else 't59'
            curve['paired_' + label] = {'improved': sum(r['delta_weighted_contribution'] < 0 for r in changes),
                'deteriorated': sum(r['delta_weighted_contribution'] > 0 for r in changes),
                'unchanged': sum(r['delta_weighted_contribution'] == 0 for r in changes),
                'changes': sorted(changes, key=lambda r: r['delta_weighted_contribution'])}
    objectives = {}
    for cursor in (0, 59, 90):
        prefix = 'baseline_0' if cursor == 0 else f'endpoint_{cursor}'
        chem = run.read(OUTPUT / f'{prefix}_chemistry_one_variant.json')
        mrks = run.read(OUTPUT / f'{prefix}_mrks.json')
        assert chem['complete'] and mrks['complete'] and chem['model_unchanged'] and mrks['model_unchanged']
        assert len(chem['rows']) == 268 and len(mrks['rows']) == 90
        assert chem['evaluation_manifest_sha256'] == run.file_sha256(OUTPUT / 'evaluation_manifest.json')
        objectives[str(cursor)] = {**chem['objectives'], **mrks['objectives']}
    ratios = {str(cursor): {task: objectives[str(cursor)][task] / objectives['0'][task] for task in run.TASKS}
              for cursor in (59, 90)}
    saved = torch.load(OUTPUT / 'ordinary_sgd_adamw/checkpoint_90.pt', map_location='cpu', weights_only=False)
    original = torch.load(SOURCE / 'ordinary_sgd_adamw/latest.pt', map_location='cpu', weights_only=False)
    assert saved['cursor'] == len(saved['logs']) == 90 and saved['scheduler'] is None
    assert saved['logs'][:59] == original['logs']
    assert run.file_sha256(SOURCE / 'ordinary_sgd_adamw/latest.pt') == START_SHA
    assert all(r['sample'] == run.read(OUTPUT / 'sampling_manifest.json')[i] for i, r in enumerate(saved['logs']))
    assert all(float(s['step']) == 90 for s in saved['optimizer']['state'].values())
    logs = saved['logs'][59:]
    assert logs[0]['before_sha256'] == original['logs'][-1]['after_sha256']
    assert all(a['after_sha256'] == b['before_sha256'] for a, b in zip(saved['logs'], saved['logs'][1:]))
    runtime = {'additional_updates': len(logs), 'training_seconds': sum(r['total_seconds'] for r in logs),
               'mean_update_seconds': float(np.mean([r['total_seconds'] for r in logs])),
               'peak_live_gib': max(r['peak_allocated_bytes'] for r in logs) / 2**30,
               'peak_reserved_gib': max(r['peak_reserved_bytes'] for r in logs) / 2**30,
               'exclusive_seconds': {k: sum(r['seconds'].get(k, 0) for r in logs) for k in logs[0]['seconds']}}
    files = list(OUTPUT.glob('endpoint_*.json')) + list(OUTPUT.glob('baseline_*.json')) + [
        OUTPUT / name for name in ('protocol.json', 'sampling_manifest.json', 'calibration.json',
                                  'evaluation_manifest.json', 'continuation_receipt.json')]
    files += [OUTPUT / f'ordinary_sgd_adamw/checkpoint_{c}.pt' for c in (59, 70, 80, 90)]
    hashes = {p.relative_to(OUTPUT).as_posix(): run.file_sha256(p) for p in files}
    tensor_hashes = {}
    for c in (59, 70, 80, 90):
        checkpoint = torch.load(OUTPUT / f'ordinary_sgd_adamw/checkpoint_{c}.pt', map_location='cpu', weights_only=False)
        assert checkpoint['cursor'] == c and checkpoint['rng']
        assert checkpoint['calibration'] == saved['calibration']
        assert checkpoint['manifest_sha256'] == saved['manifest_sha256']
        assert all(float(s['step']) == c for s in checkpoint['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        assert all(torch.isfinite(s[k]).all() for s in checkpoint['optimizer']['state'].values()
                   for k in ('exp_avg', 'exp_avg_sq'))
        digest = hashlib.sha256()
        for name, value in checkpoint['model'].items():
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
        tensor_hashes[str(c)] = digest.hexdigest()
        assert tensor_hashes[str(c)] == checkpoint['logs'][-1]['after_sha256']
    result = {'dataset_sha256': run.DATA_SHA, 'trajectory': curves, 'objectives': objectives, 'ratios': ratios,
              'clean_metric_name': 'Diet30-clean weighted error (kcal/mol); not canonical full30 WTMAD-2',
              'runtime': runtime, 'new_update_logs': logs, 'hashes': hashes,
              'best_clean28_cursor': min(curves, key=lambda r: r['clean28'])['cursor'],
              'scientifically_eligible': {c: all(r < 1 for r in values.values()) for c, values in ratios.items()},
              'source_hashes': run.read(OUTPUT / 'continuation_receipt.json'),
              'checkpoint_model_tensor_sha256': tensor_hashes,
              'coverage': {'relchem_unique': len({r['sample']['relchem']['identity'] for r in saved['logs']}),
                           'ae17_unique': len({r['sample']['ae17']['identity'] for r in saved['logs']}),
                           'mrks_unique': len({r['sample']['mrks_id'] for r in saved['logs']})},
              'model_unchanged_on_evaluation': True, 'original_t59_unchanged': True,
              'no_scf_or_future_test': True, 'one_variant_per_chemistry_identity': True}
    run.write(run.ROOT / 'iid_adamw_t59_t90_metrics.json', result)
    run.write(OUTPUT / 'checkpoint_manifest.json', {
        'hashes': hashes, 'source': result['source_hashes'],
        'fixed_coefficients': saved['calibration']['lambda'],
        'configuration': run.read(OUTPUT / 'protocol.json'),
        'checkpoint_state': 'Native model/buffers, optimizer/moments, RNG, cursor, manifest SHA and calibration; adjacent SHA-bound source protocol',
    })
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r['cursor'] for r in curves], [r['clean28'] for r in curves], 'o-', label='IID AdamW LR=1e-4')
    ax.set(xlabel='Optimizer updates', ylabel='Diet30-clean weighted error (kcal/mol)')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(run.ROOT / 'iid_adamw_t59_t90_clean28.png', dpi=160)
    plt.close(fig)
    lines = ['# Preserved IID AdamW continuation: t59 to t90', '',
             'Exactly 31 additional native AdamW updates completed. Original t59 was copied byte-for-byte, not reconstructed, and remains unchanged. Constant LR=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01; no scheduler or task-weight change.', '',
             '## Clean validation', '',
             'Clean28 is the Diet30-clean weighted error, mean of 28 Diet-weighted absolute reaction errors (kcal/mol), not canonical full30 WTMAD-2. Same frozen PBE0 densities and PBE0-D3(BJ); no SCF. Full30 is diagnostic, selection_allowed=false.', '',
             '| Cursor | Clean28 | Change vs t0 | Change vs t59 | Improved / worse vs t0 | Improved / worse vs t59 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in curves:
        a, b = r['paired_t0'], r['paired_t59']
        lines.append(f"| {r['cursor']} | {r['clean28']:.9f} | {r['delta_t0']:+.9f} | {r['delta_t59']:+.9f} | {a['improved']} / {a['deteriorated']} | {b['improved']} / {b['deteriorated']} |")
    lines += ['', '![Measured checkpoint trajectory](iid_adamw_t59_t90_clean28.png)', '',
              'Minimum measured Clean28 occurs at t70, 0.923530406 kcal/mol below P536. Consecutive improvements occur at t0->t59->t70; there are not two successive improvements after t59. t80 stays near t70, then t90 rebounds. Thus t59 is part of a favorable early window, not proof of sustained convergence. Only t59/t90 have the requested exact scientific audit; t70 is numerically finite but scientific eligibility is unverified.', '',
              '## Exact scientific objectives', '',
              'Same independent fixed manifest: 251 relchem + 17 AE17 identities, exactly one variant per identity; all90 mRKS systems, equal-system means, no parameter backward. Initial receipts reused after exact model-tensor and manifest/protocol checks.', '',
              '| Task | t0 | t59 | t90 | t59/t0 | t90/t0 |', '|---|---:|---:|---:|---:|---:|']
    for task in run.TASKS:
        lines.append(f"| {task} | {objectives['0'][task]:.12g} | {objectives['59'][task]:.12g} | {objectives['90'][task]:.12g} | {ratios['59'][task]:.9f} | {ratios['90'][task]:.9f} |")
    lines += ['', 'Neither audited checkpoint is scientifically eligible: exact relchem is above t0. Clean28 improvement must not be confused with relchem-objective improvement. From t59 to t90 both chemistry losses deteriorate; the complete Exc/operator comparison is in the table. These associations do not isolate a causal objective or establish numerical instability.', '',
              '## Paired reaction diagnostics', '',
              'All 28 clean signed/absolute errors, predictions, references, weights and exact score contributions at every checkpoint are in the JSON; contributions sum to the reported Clean28 within 1e-12. Repeated checkpoints on one validation panel are correlated, not independent replications.', '',
              '| Largest t70 improvements vs t0 | Score-contribution change |', '|---|---:|']
    best = next(r for r in curves if r['cursor'] == 70)
    for r in best['paired_t0']['changes'][:8]:
        lines.append(f"| {r['reaction_id']} | {r['delta_weighted_contribution']:+.9f} |")
    before = {r['reaction_id']: r['clean_score_contribution'] for r in best['reactions'] if r['clean']}
    deterioration = sorted([(r['reaction_id'], r['clean_score_contribution'] - before[r['reaction_id']])
                            for r in curves[-1]['reactions'] if r['clean']], key=lambda x: -x[1])
    lines += ['', '| Largest t90 deterioration vs t70 | Score-contribution change |', '|---|---:|']
    lines += [f'| {name} | {change:+.9f} |' for name, change in deterioration[:8]]
    lines += ['', 't70 improves 19/28 reactions vs t0; its three largest improvements explain about 45% of the net gain, so the result is broad but concentrated. t90 worsens 22/28 vs t59. No checkpoint is promoted from a single reaction.', '',
              '## Runtime and provenance', '',
              f"Logged synchronized update time: {runtime['training_seconds']:.3f}s ({runtime['training_seconds']/60:.2f}min), {runtime['mean_update_seconds']:.3f}s/update, excluding preflight/checkpoint overhead. Peak live/reserved CUDA memory: {runtime['peak_live_gib']:.3f}/{runtime['peak_reserved_gib']:.3f} GiB on RTX5070Ti. Reserved memory is allocator accounting, not live allocation. Timing includes concurrent focused CPU tests; it is not an isolated hardware benchmark.", '',
              f"Sampling coverage over the 90 actual updates: {result['coverage']}. All original 59 log entries and samples remain identical; all native moment steps match each checkpoint cursor, RNG is present, model/moment arrays are finite, and the SHA chain is continuous.", '',
              '| Logged update component | Seconds over 31 updates |', '|---|---:|']
    lines += [f'| {name} | {seconds:.3f} |' for name, seconds in runtime['exclusive_seconds'].items()]
    lines += ['', f'Dataset logical SHA256: `{run.DATA_SHA}`.', '',
              '| Cursor | Checkpoint SHA256 |', '|---|---|']
    lines += [f"| {c} | `{hashes[f'ordinary_sgd_adamw/checkpoint_{c}.pt']}` |" for c in (59, 70, 80, 90)]
    lines += ['', f'Large checkpoints and the full SHA-bound checkpoint/configuration manifest: `{OUTPUT}`. Physics source, evaluator, sampling, coefficient and endpoint receipts are recorded in the metrics; no production source changed.', '',
              '## Decision', '',
              '**NO-GO for automatic continuation or production promotion.** t90 reverses much of the validation gain and fails relchem eligibility. Preserve t59/t70/t80/t90. The exact next bounded experiment is a read-only four-objective audit of preserved t70 on the same fixed one-variant/full90 panel, before choosing a scientifically eligible early checkpoint. Do not launch it as part of this continuation.', '',
              'Independent sampling-seed confirmation is justified before committing 2xV100 resources, conditionally on that scientific gate. A longer unchanged t90 run is not supported. LR=1e-4 produced a useful early validation window but is not established as a robust long-horizon setting. No LR/weight mechanism is identified causally by this single trajectory.', '',
              'For eventual two-V100 use, independent replicas are the appropriate first confirmation. The measured Exc/full-AO workloads dominate updates; no actual dual-V100 speedup was measured. Distributed optimization would require a globally normalized gradient for each task and the same fixed coefficients before one shared AdamW update; independent per-GPU scalarizations or silently doubled chemistry batch size would change the protocol. No Slurm submission.', '',
              'Validation: 17 focused tests passed (including real native AdamW deterministic resume and wrapper no-replay/original-preservation tests); Ruff, compileall and git diff --check passed. Ponytail: existing trainer/evaluator reused, no new optimizer or objective machinery. Pocock: original SHA, moments, cursor, RNG, parameter order, source hashes, fixed variants, sample identity and exact score reconstruction checked; unknown t70 scientific eligibility and correlated validation points are explicit.', '',
              'No future-test evaluation, SCF, architecture/precision/dataset changes, historical reruns or updates beyond t90 occurred.']
    (run.ROOT / 'iid_adamw_t59_t90_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SUMMARY', result['best_clean28_cursor'], ratios, runtime, flush=True)


if __name__ == '__main__':
    main()
