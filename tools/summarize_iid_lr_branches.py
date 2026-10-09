"""Select once by Clean28 and summarize cached matched-branch diagnostics."""
import argparse
import hashlib
import os
import sys
from pathlib import Path

# Receipt-only CPU BLAS: avoid loading a second OpenMP runtime beside PyTorch.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from tools.stabilize_iid_adamw import OUTPUT, RATES, SOURCE, original_integrity


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(a @ b / denominator)


def select():
    candidates = []
    for arm in RATES:
        for cursor in (80, 90):
            folder = OUTPUT / arm
            receipt = run.read(folder / f'endpoint_{cursor}_validation.json')
            sha = run.file_sha256(folder / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt')
            assert receipt['model_unchanged'] and receipt['checkpoint_sha256'] == sha
            candidates.append({'arm': arm, 'cursor': cursor, 'clean28': receipt['metrics']['clean28'], 'checkpoint_sha256': sha})
    best = min(candidates, key=lambda r: (r['clean28'], r['arm'], r['cursor']))
    path = OUTPUT / 'selected_candidate.json'
    if path.exists():
        assert run.read(path) == best
    else:
        run.write(path, best)
    return best


def vector(state, order):
    return np.concatenate([state['model'][r['name']].numpy().reshape(-1).astype(np.float64) for r in order])


def summary():
    original_integrity()
    best = select()
    protocol = run.read(OUTPUT / 'frozen_protocol.json')
    assert run.file_sha256(run.ROOT / 'tools/evaluate_microbatch_endpoint.py') == protocol['evaluator_sha256']
    assert run.file_sha256(run.ROOT / 'tools/stabilize_iid_adamw.py') == protocol['wrapper_sha256']
    order = run.read(SOURCE / 'ordinary_sgd_adamw/parameter_order.json')
    start = torch.load(SOURCE / 'ordinary_sgd_adamw/checkpoint_70.pt', map_location='cpu', weights_only=False)
    base = vector(start, order)
    assert len(base) == 9446 and len({r['name'] for r in order}) == len(order)
    checkpoints, hashes, tensor_hashes = {}, {}, {}
    for arm in ('A', 'B', 'C'):
        assert run.file_sha256(OUTPUT / arm / 'sampling_manifest.json') == protocol['whole_manifest_sha256']
        checkpoints[arm] = {}
        for cursor in (70, 80, 90):
            path = OUTPUT / arm / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['cursor'] == len(state['logs']) == cursor and state['scheduler'] is None
            assert state['manifest_sha256'] == protocol['whole_manifest_sha256'] and state['calibration'] == start['calibration']
            assert state['logs'][:70] == start['logs']
            assert state['rng']
            assert all(a['after_sha256'] == b['before_sha256'] for a, b in zip(state['logs'], state['logs'][1:]))
            assert all(float(v['step']) == cursor for v in state['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert all(torch.isfinite(v[k]).all() for v in state['optimizer']['state'].values() for k in ('exp_avg', 'exp_avg_sq'))
            checkpoints[arm][cursor] = state
            hashes[path.relative_to(OUTPUT).as_posix()] = run.file_sha256(path)
            digest = hashlib.sha256()
            for name, value in state['model'].items():
                digest.update(name.encode())
                digest.update(value.contiguous().numpy().tobytes())
            tensor_hashes[f'{arm}{cursor}'] = digest.hexdigest()
            assert digest.hexdigest() == state['logs'][-1]['after_sha256']
    bundle = run.PublicationDataset(run.DATA)
    catalog = {r['source_id']: r for r in bundle.validation_reactions.values()}
    clean_ids = set(bundle.splits['diet30_clean_validation']['ids'])
    assert bundle.manifest['logical_sha256'] == run.DATA_SHA
    bundle.close()
    curves = {}
    common_reference = run.read(OUTPUT / 'A/endpoint_70_validation.json')['metrics']
    baseline_rows = {r['reaction_id']: r for r in common_reference['reaction_rows'] if r['clean']}
    for arm in checkpoints:
        curves[arm] = {}
        for cursor in (70, 80, 90):
            path = OUTPUT / arm / f'endpoint_{cursor}_validation.json'
            receipt = run.read(path)
            assert receipt['model_unchanged'] and receipt['checkpoint_sha256'] == hashes[f'{arm}/ordinary_sgd_adamw/checkpoint_{cursor}.pt']
            rows = []
            for r in receipt['metrics']['reaction_rows']:
                assert r['clean'] == (catalog[r['reaction_id']]['id'] in clean_ids)
                if not r['clean']:
                    continue
                metadata = catalog[r['reaction_id']]
                weighted = abs(r['signed_error_kcal_mol']) * metadata['diet_weight']
                assert abs(weighted - r['weighted_absolute_error']) < 1e-12
                rows.append({**r, 'reference_energy_kcal_mol': metadata['reference_energy_kcal_mol'],
                             'predicted_energy_kcal_mol': metadata['reference_energy_kcal_mol'] + r['signed_error_kcal_mol'],
                             'absolute_error_kcal_mol': abs(r['signed_error_kcal_mol']), 'diet_weight': metadata['diet_weight'],
                             'score_contribution': weighted / 28,
                             'contribution_change_from_t70': (weighted - baseline_rows[r['reaction_id']]['weighted_absolute_error']) / 28})
            score = receipt['metrics']['clean28']
            assert len(rows) == 28 and abs(sum(r['score_contribution'] for r in rows) - score) < 1e-12
            assert len({r['reaction_id'] for r in rows}) == 28
            curves[arm][str(cursor)] = {'clean28': score, 'delta_t70': score - common_reference['clean28'],
                                      'improved_reactions': sum(r['contribution_change_from_t70'] < 0 for r in rows),
                                      'deteriorated_reactions': sum(r['contribution_change_from_t70'] > 0 for r in rows),
                                      'reaction_contributions': rows, 'full30_diagnostic': receipt['metrics']['full30'],
                                      'full30_selection_allowed': False}
    baseline = {**run.read(OUTPUT / 'A/baseline_0_chemistry_one_variant.json')['objectives'],
                **run.read(OUTPUT / 'A/baseline_0_mrks.json')['objectives']}
    for arm in ('B', 'C'):
        for cursor in ('80', '90'):
            control_rows = {r['reaction_id']: r for r in curves['A'][cursor]['reaction_contributions']}
            paired_changes = []
            for row in curves[arm][cursor]['reaction_contributions']:
                row['contribution_change_from_matched_control'] = row['score_contribution'] - control_rows[row['reaction_id']]['score_contribution']
                paired_changes.append(row['contribution_change_from_matched_control'])
            curves[arm][cursor]['improved_reactions_vs_matched_control'] = sum(d < 0 for d in paired_changes)
            curves[arm][cursor]['delta_matched_control'] = curves[arm][cursor]['clean28'] - curves['A'][cursor]['clean28']
    audits = {}
    for arm, cursor in [('A', c) for c in (59, 70, 80, 90)] + [(best['arm'], best['cursor'])]:
        chem = run.read(OUTPUT / arm / f'endpoint_{cursor}_chemistry_one_variant.json')
        mrks = run.read(OUTPUT / arm / f'endpoint_{cursor}_mrks.json')
        assert chem['complete'] and mrks['complete'] and chem['model_unchanged'] and mrks['model_unchanged']
        assert len(chem['rows']) == 268 and len(mrks['rows']) == 90
        assert chem['evaluation_manifest_sha256'] == protocol['evaluation_manifest_sha256']
        assert chem['checkpoint_sha256'] == mrks['checkpoint_sha256'] == run.file_sha256(
            OUTPUT / arm / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt')
        values = {**chem['objectives'], **mrks['objectives']}
        ratios = {task: values[task] / baseline[task] for task in run.TASKS}
        audits[f'{arm}{cursor}'] = {'objectives': values, 'ratios_t0': ratios,
                                  'ratios_t70': {task: values[task] / audits['A70']['objectives'][task] for task in run.TASKS} if 'A70' in audits else None,
                                  'eligible': all(r < 1 for r in ratios.values())}
    manifest = run.read(SOURCE / 'sampling_manifest.json')
    lambdas = protocol['fixed_coefficients']
    logs = {arm: checkpoints[arm][90]['logs'][70:90] for arm in checkpoints}
    net, comparisons = {}, []
    for arm in checkpoints:
        assert all(r['sample'] == manifest[i] and r['learning_rate'] == protocol['rates'][arm]
                   for i, r in enumerate(logs[arm], 70))
        net[arm] = {}
        for cursor in (80, 90):
            displacement = vector(checkpoints[arm][cursor], order) - base
            control = vector(checkpoints['A'][cursor], order) - base
            net[arm][str(cursor)] = {'displacement_norm_from_t70': float(np.linalg.norm(displacement)),
                                    'cosine_to_control_net_displacement': cosine(displacement, control)}
            if arm != 'A':
                assert np.isclose(np.linalg.norm(displacement), checkpoints[arm][cursor]['logs'][-1]['parameter_displacement_from_t70'],
                                  rtol=1e-12, atol=1e-12)
    first_exact = True
    for index in range(70, 90):
        gradients, steps = {}, {}
        for arm in checkpoints:
            folder = SOURCE if arm == 'A' else OUTPUT / arm
            path = folder / f'ordinary_sgd_adamw/raw_gradients/update_{index:03}.npz'
            log = logs[arm][index - 70]
            assert run.file_sha256(path) == log['raw_gradients_sha256']
            with np.load(path) as archive:
                gradients[arm] = {task: archive[task] for task in run.TASKS}
            for task in run.TASKS:
                assert np.isclose(np.linalg.norm(gradients[arm][task]), log['norms'][task], rtol=1e-12, atol=1e-12)
            if arm != 'A':
                step_path = folder / f'actual_step_{index + 1}.npy'
                assert run.file_sha256(step_path) == log['actual_step_sha256']
                steps[arm] = np.load(step_path)
                assert np.isclose(np.linalg.norm(steps[arm]), log['step_norm'], rtol=1e-12, atol=1e-12)
                dots = [float(gradients[arm][task] @ steps[arm]) for task in run.TASKS]
                assert np.allclose(dots, log['task_predicted_loss_change_actual_step'], rtol=1e-10, atol=1e-12)
        if index == 70:
            first_exact = all(np.array_equal(gradients['A'][task], gradients[arm][task])
                              for arm in ('B', 'C') for task in run.TASKS)
            assert all(np.allclose(gradients['A'][task], gradients[arm][task], rtol=1e-12, atol=1e-12)
                       for arm in ('B', 'C') for task in run.TASKS)
        joint = {arm: sum(lambdas[task] * g[task] for task in run.TASKS) for arm, g in gradients.items()}
        comparisons.append({'update': index + 1,
                            'weighted_gradient_cosine_to_control': {arm: cosine(joint['A'], joint[arm]) for arm in ('B', 'C')},
                            'actual_step_B_C_cosine': cosine(steps['B'], steps['C'])})
    runtime = {arm: {'updates': 20, 'seconds': sum(r['total_seconds'] for r in rows),
                     'mean_update_seconds': float(np.mean([r['total_seconds'] for r in rows])),
                     'peak_live_gib': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
                     'peak_reserved_gib': max(r['peak_reserved_bytes'] for r in rows) / 2**30,
                     'sum_step_norms': sum(r['step_norm'] for r in rows)} for arm, rows in logs.items()}
    for path in OUTPUT.rglob('*.json'):
        if path.name not in ('prequalification_protocol.json', 'checkpoint_manifest.json'):
            hashes[path.relative_to(OUTPUT).as_posix()] = run.file_sha256(path)
    result = {'protocol': protocol, 'baseline_objectives': baseline, 'audits': audits, 'curves': curves,
              'selected_candidate': best, 'matched_update_logs': logs, 'net_displacements': net,
              'direction_comparisons': comparisons, 'first_raw_gradients_bitwise_equal': first_exact,
              'runtime': runtime, 'hashes': hashes, 'checkpoint_tensor_sha256': tensor_hashes,
              'control_instantaneous_step_vectors': 'Not archived; no control replay. Exact control step norms and task progresses reused; net t80/t90 vectors reconstructed from preserved checkpoints.',
              'no_historical_mutation': True, 'no_scf_or_future_test': True, 'new_optimizer_updates': 40}
    run.write(run.ROOT / 'iid_adamw_lr_stabilization_metrics.json', result)
    run.write(OUTPUT / 'checkpoint_manifest.json', {'hashes': hashes, 'protocol': protocol, 'best_candidate': best})
    print('SUMMARY', best, audits, runtime, net, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'summary'))
    args = parser.parse_args()
    if args.stage == 'select':
        print(select(), flush=True)
    else:
        summary()
