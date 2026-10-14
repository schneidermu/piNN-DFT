"""Census algebra and one F010 continuation with PHYS4-only adaptation."""
import argparse
import copy
import json
import time
from contextlib import contextmanager

import lap_f021_phys4 as exp
import numpy as np
import torch

audit, b8, f010 = exp.audit, exp.b8, exp.f010


def cache(task, record):
    path = exp.OUT / 'singleton' / (task + '_' + exp.key(record) + '.npz')
    receipt = audit.read(path.with_suffix('.json'))
    assert receipt['sha256'] == audit.sha(path)
    return np.load(path), receipt


def variance(values):
    centered = values - values.mean(axis=0)
    return float(np.mean(np.sum(centered * centered, axis=1)))


def covariance(a, b):
    return float(np.mean(np.sum((a - a.mean(axis=0)) * (b - b.mean(axis=0)), axis=1)))


def chemistry_batches(prefix):
    folder = exp.ROOT.parent / 'lap_f020_stream_20261013'
    receipt = audit.read(exp.ROOT / 'lap_f020_jacobian_reconstruction.json')
    assert audit.sha(folder / 'stream_jacobian.npz') == receipt['sha256']
    assert audit.sha(folder / 'stream_rows.json') == receipt['mapping_sha256']
    rows = audit.read(folder / 'stream_rows.json')
    with np.load(folder / 'stream_jacobian.npz') as data:
        jacobian, residuals = data['gradients'].copy(), data['residual'].copy()
    scales = audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['scales']
    coefficients = f010.f010_raw_coefficients(residuals, scales, [r['database'] for r in rows]) * 251
    singleton = jacobian * coefficients[:, None]
    lookup = {(r['identity'], r['variant']): i for i, r in enumerate(rows)}
    return np.array([singleton[[lookup[(r['identity'], r['variant'])] for r in b['relchem']]].mean(axis=0) for b in prefix])


def analysis(prefix):
    protocol = exp.read('protocol')
    receipt = exp.read('singleton_gradients_receipt')
    assert receipt['complete']
    arrays = {task: [] for task in exp.TASKS}
    for row in prefix:
        with cache('ae17', row['ae17'])[0] as payload:
            arrays['ae17'].append(payload['ae17'].copy())
        with cache('mrks', row['mrks_id'])[0] as payload:
            for task in ('exc', 'op'):
                arrays[task].append(payload[task].copy())
    arrays = {t: np.array(v, dtype=np.float64) for t, v in arrays.items()}
    ix = np.array([exp.positions(t) for t in range(exp.TOTAL)])
    mean4 = {t: a[ix].mean(axis=1) for t, a in arrays.items()}
    rows = []
    for task in exp.TASKS:
        a, z = arrays[task], mean4[task]
        assert np.allclose(a.mean(axis=0), z.mean(axis=0), rtol=1e-10, atol=1e-10)
        v1, v4 = variance(a), variance(z)
        assert v4 <= v1 + 1e-10 * max(v1, 1)
        rows.append({'task': task, 'singleton_MSE': v1, 'phys4_MSE': v4,
                     'ratio': v4 / v1 if v1 else None, 'reduction': 1 - v4 / v1 if v1 else None,
                     'weighted_singleton_MSE': v1 * b8.LAMBDAS[task] ** 2,
                     'weighted_phys4_MSE': v4 * b8.LAMBDAS[task] ** 2,
                     'RMS_per_coordinate_1': np.sqrt(v1 / 9446), 'RMS_per_coordinate_4': np.sqrt(v4 / 9446)})
    exp.table('task_variance', rows)
    full = audit.geometry._physical(audit.geometry.STATES[0])
    population_rows = []
    for task in exp.TASKS:
        mu = arrays[task].mean(axis=0)
        population_rows.append({'task': task, 'prefix_mean_norm': float(np.linalg.norm(mu)),
                                'qualified_uniform_mean_norm': float(np.linalg.norm(full[task])),
                                'prefix_uniform_cosine': float(mu @ full[task] / (np.linalg.norm(mu) * np.linalg.norm(full[task]))),
                                'relative_vector_difference': float(np.linalg.norm(mu - full[task]) / np.linalg.norm(full[task]))})
    exp.table('reference_population_comparison', population_rows)
    weighted = {t: arrays[t] * b8.LAMBDAS[t] for t in exp.TASKS}
    f1 = sum(weighted.values())
    f4 = sum(mean4[t] * b8.LAMBDAS[t] for t in exp.TASKS)
    chem = chemistry_batches(prefix) * protocol['c010'] * b8.LAMBDAS['relchem']
    g1, g4 = chem + f1, chem + f4
    assert np.allclose(g1.mean(axis=0), g4.mean(axis=0), atol=1e-10, rtol=1e-10)
    combined = []
    for name, a, z in (('physical', f1, f4), ('composite', g1, g4)):
        v1, v4 = variance(a), variance(z)
        combined.append({'object': name, 'singleton_MSE': v1, 'phys4_MSE': v4,
                         'ratio': v4 / v1 if v1 else None, 'reduction': 1 - v4 / v1 if v1 else None,
                         'mean_difference_L2': float(np.linalg.norm(a.mean(axis=0) - z.mean(axis=0))),
                         'RMS_per_coordinate_1': np.sqrt(v1 / 9446), 'RMS_per_coordinate_4': np.sqrt(v4 / 9446)})
    exp.table('joint_variance', combined)
    covrows = []
    for i, task in enumerate(exp.TASKS):
        for other in exp.TASKS[i:]:
            a = covariance(weighted[task], weighted[other])
            z = covariance(mean4[task] * b8.LAMBDAS[task], mean4[other] * b8.LAMBDAS[other])
            covrows.append({'left': task, 'right': other, 'singleton_covariance_trace': a,
                            'phys4_covariance_trace': z, 'multiplicity_in_sum': 1 if task == other else 2})
    for label, a, z in (('chemistry_physical', f1, f4),):
        covrows.append({'left': 'chemistry', 'right': label, 'singleton_covariance_trace': covariance(chem, a),
                        'phys4_covariance_trace': covariance(chem, z), 'multiplicity_in_sum': 2})
    assert np.isclose(variance(g1), variance(chem) + variance(f1) + 2 * covariance(chem, f1), rtol=1e-10, atol=1e-10)
    assert np.isclose(variance(g4), variance(chem) + variance(f4) + 2 * covariance(chem, f4), rtol=1e-10, atol=1e-10)
    exp.table('covariance_decomposition', covrows)
    np.savez_compressed(exp.OUT / 'census_vectors.npz', **arrays, **{t + '_phys4': v for t, v in mean4.items()},
                        physical_singleton=f1, physical_phys4=f4, composite_singleton=g1, composite_phys4=g4)
    per_position = []
    for t in range(128):
        record = {'position': t, 'composite_cosine': float(g1[t] @ g4[t] / (np.linalg.norm(g1[t]) * np.linalg.norm(g4[t]))),
                  'composite_norm_1': float(np.linalg.norm(g1[t])), 'composite_norm_4': float(np.linalg.norm(g4[t])),
                  'physical_unique_records': len({prefix[p]['mrks_id'] for p in exp.positions(t)})}
        for task in exp.TASKS:
            mu = arrays[task].mean(axis=0)
            record[task + '_cosine_1'] = float(arrays[task][t] @ mu / (np.linalg.norm(arrays[task][t]) * np.linalg.norm(mu)))
            record[task + '_cosine_4'] = float(mean4[task][t] @ mu / (np.linalg.norm(mean4[task][t]) * np.linalg.norm(mu)))
            record[task + '_predicted_descent_1'] = float(mu @ g1[t] / np.linalg.norm(g1[t]))
            record[task + '_predicted_descent_4'] = float(mu @ g4[t] / np.linalg.norm(g4[t]))
        per_position.append(record)
    exp.table('gradient_distributions', per_position)


def parity(prefix, source, budget):
    model, shadow, run, names = exp.loaded(source)
    bundle = audit.TrainingBundle(exp.ROOT.parent / 'publication_dataset_v1')
    dispersion = bundle.chemistry_dispersions()
    md = audit.read(bundle.root / 'mrks/dispersion.json')
    snapshots = [copy.deepcopy(m.state_dict()) for m in (model, shadow)]
    rng = b8._capture_rng()
    records = []
    try:
        # Actual four-entry scalar-autograd reference on small, predefined records.
        group = tuple(exp.read('protocol')['scalar_reference_positions'])
        small = group[:2]
        for task in exp.TASKS:
            budget.check(120)
            params = run.existing.named_trainable_parameters(shadow if task == 'ae17' else model)
            scalars, expected = [], []
            kept = []
            for p in group:
                if task == 'ae17':
                    reaction = b8._load_reaction(bundle, 'ae17', prefix[p]['ae17'])
                    kept.append(reaction)
                    scalars.append(run.existing.make_reaction_objective(shadow, reaction, device='cuda', dtype=torch.float64, dispersions=dispersion)())
                    with cache('ae17', prefix[p]['ae17'])[0] as data:
                        expected.append(data[task].copy())
                else:
                    system = bundle.mrks().operator_system(prefix[p]['mrks_id'], device='cuda', dtype=torch.float32, chunk_size=4096)
                    kept.append(system)
                    factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=md)
                    scalars.append(factories[0 if task == 'exc' else 1]())
                    with cache('mrks', prefix[p]['mrks_id'])[0] as data:
                        expected.append(data[task].copy())
            average = sum(scalars) / 4
            grads = torch.autograd.grad(average, tuple(params.values()), allow_unused=True)
            actual = np.concatenate([(torch.zeros_like(p) if g is None else g).detach().double().cpu().numpy().reshape(-1)
                                     for p, g in zip(params.values(), grads, strict=True)])
            reference = np.mean(expected, axis=0)
            gap = float(np.linalg.norm(actual - reference))
            bound = 1e-10 + (16 * np.finfo(np.float32).eps * np.mean([np.linalg.norm(v) for v in expected]) if task != 'ae17' else 1e-10 * np.linalg.norm(reference))
            records.append({'task': task, 'reference_positions': json.dumps(group), 'absolute_gradient_gap': gap,
                             'relative_gradient_gap': gap / max(np.linalg.norm(reference), 1e-30), 'bound': bound, 'passed': gap <= bound})
            assert gap <= bound, 'Independent scalar-gradient averaging parity failed'
            del scalars, average, grads, kept
            torch.cuda.empty_cache()
        _chemistry_loss, chemistry_grad, _info, seconds = f010.f010_mean(model, shadow, bundle, prefix[0]['relchem'], dispersion,
                                                                      audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['scales'], audit.geometry.C010)
        reference = chemistry_batches(prefix)[0] * audit.geometry.C010
        gap = np.linalg.norm(exp.flat(chemistry_grad, names) - reference) / np.linalg.norm(reference)
        assert gap <= 1e-10
        records.append({'task': 'chemistry_F010', 'reference_positions': '[0]', 'absolute_gradient_gap': float(np.linalg.norm(exp.flat(chemistry_grad, names) - reference)),
                         'relative_gradient_gap': float(gap), 'bound': 1e-10, 'passed': True})
        exp.table('gradient_parity', records)
        # Independently profile fixed-density full-system scalar cost; no parameter backward.
        scalar_seconds = []
        for p in (small[0], 0, max(range(128), key=lambda p: bundle.systems[prefix[p]['mrks_id']]['n_grid'] * bundle.systems[prefix[p]['mrks_id']]['n_ao'])):
            budget.check(120)
            start = time.perf_counter()
            system = bundle.mrks().operator_system(prefix[p]['mrks_id'], device='cuda', dtype=torch.float32, chunk_size=4096)
            factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=md)
            with torch.no_grad():
                values = [float(factory()) for factory in factories]
            assert np.isfinite(values).all()
            torch.cuda.synchronize()
            scalar_seconds.append({'position': p, 'seconds': time.perf_counter() - start})
            del system, factories
            torch.cuda.empty_cache()
    finally:
        for m, saved in zip((model, shadow), snapshots, strict=True):
            m.load_state_dict(saved)
            assert all(torch.equal(v, m.state_dict()[n]) for n, v in saved.items())
        b8._restore_rng(rng)
        assert run.existing.equal(rng, b8._capture_rng())
        bundle.close()
    census_receipt = exp.read('singleton_gradients_receipt')
    ae = [r['seconds'] for r in census_receipt['records'] if r['binding']['task'] == 'ae17']
    mr = [r['seconds'] for r in census_receipt['records'] if r['binding']['task'] == 'mrks']
    update = seconds + 4 * (np.quantile(ae, .9) + np.quantile(mr, .9))
    evaluation_reserve = 2 * (90 * max(r['seconds'] for r in scalar_seconds) + 268 * seconds / 8) + 1200
    projected = update * 128 + evaluation_reserve
    passed = projected + budget.used() < exp.BUDGET
    exp.write('resource_budget', {'limit': exp.BUDGET, 'used_preflight': budget.used(), 'chemistry_batch_seconds': seconds,
                                 'physical_scalar_profiles': scalar_seconds, 'projected_update_seconds_p90': float(update),
                                 'mandatory_evaluation_and_clean28_reserve': float(evaluation_reserve),
                                 'projected_remaining': float(projected), 'fits': bool(passed)})
    exp.write('preflight_decision', {'passed': bool(passed), 'gradient_parity': True, 'restoration': True,
                                    'exposure_parity': True, 'training_authorized': bool(passed)})


@contextmanager
def training_adapter(prefix, budget):
    """Temporarily adapt the existing experimental F010 harness, never production physics."""
    saved_module = {n: getattr(f010, n) for n in ('OUT', 'PROTOCOL_ID', 'NEW_UPDATES', 'RAW_CHECKPOINTS', '_save_checkpoint', '_load_checkpoint', 'f010_mean')}
    original_other = b8.other_tasks
    original_mean = f010.f010_mean
    target = exp.OUT / 'ordinary_sgd_adamw'
    diagnostic = {}
    protocol_sha = audit.sha(exp.ROOT / 'lap_f021_phys4_protocol.json')

    def save(path, model, optimizer, cursor, logs, coefficient, parent_sha):
        assert coefficient == audit.geometry.C010
        if cursor == 0:
            parent = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
            b8._restore_rng(parent['rng'])
        if logs:
            logs[-1]['physical_positions'] = list(exp.positions(cursor - 1))
            logs[-1].update(diagnostic)
        payload = {'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                   'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': b8._capture_rng(), 'cursor': cursor,
                   'logs': copy.deepcopy(logs), 'protocol_id': 'F021-PHYS4', 'protocol_sha256': protocol_sha,
                   'manifest_sha256': f010.continuation.MANIFEST_SHA, 'c': coefficient, 'lambdas': b8.LAMBDAS,
                   'lr': 2.5e-5, 'scheduler': None, 'parent_checkpoint_sha256': parent_sha}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        if cursor == 0:
            torch.save(payload, target / 'checkpoint_0.pt')

    def restore(path, model, optimizer, coefficient):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['protocol_sha256'] == protocol_sha and payload['protocol_id'] == 'F021-PHYS4'
        assert payload['manifest_sha256'] == f010.continuation.MANIFEST_SHA and payload['c'] == coefficient
        assert payload['lambdas'] == b8.LAMBDAS and payload['lr'] == 2.5e-5 and payload['scheduler'] is None
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        b8._restore_rng(payload['rng'])
        assert {int(s['step']) for s in optimizer.state.values()} == {128 + payload['cursor']}
        return payload['cursor'], payload['logs']

    def mean(*args, **kwargs):
        budget.check(exp.read('resource_budget')['mandatory_evaluation_and_clean28_reserve'])
        result = original_mean(*args, **kwargs)
        diagnostic['chemistry_gradient_norm'] = b8._grad_norm(result[1])
        return result

    def other(model, shadow, bundle, entry, dispersion, md):
        start = time.perf_counter()
        accum, losses = None, dict.fromkeys(exp.TASKS, 0.0)
        for p in exp.positions(entry['cursor']):
            budget.check(exp.read('resource_budget')['mandatory_evaluation_and_clean28_reserve'])
            values, gradients, _name, _elapsed = original_other(model, shadow, bundle, prefix[p], dispersion, md)
            if accum is None:
                accum = {t: {n: torch.zeros_like(g, dtype=torch.float64) for n, g in gradients[t].items()} for t in exp.TASKS}
            for t in exp.TASKS:
                losses[t] += values[t] / 4
                for n, g in gradients[t].items():
                    accum[t][n].add_(g, alpha=.25)
        diagnostic['physical_gradient_norms'] = {t: b8._grad_norm(accum[t]) for t in exp.TASKS}
        return losses, accum, 'PHYS4-prefix', time.perf_counter() - start

    try:
        f010.OUT, f010.PROTOCOL_ID = exp.OUT, 'F021-PHYS4'
        f010.NEW_UPDATES, f010.RAW_CHECKPOINTS = 128, (64, 128)
        f010._save_checkpoint, f010._load_checkpoint, f010.f010_mean = save, restore, mean
        b8.other_tasks = other
        yield
    finally:
        for n, value in saved_module.items():
            setattr(f010, n, value)
        b8.other_tasks = original_other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('analysis', 'parity', 'train'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = exp.Budget()
    try:
        prefix, source, scales = exp.freeze()
        if args.stage == 'analysis':
            analysis(prefix)
        elif args.stage == 'parity':
            parity(prefix, source, budget)
        else:
            assert exp.read('preflight_decision')['passed'], 'Training preflight failed'
            with training_adapter(prefix, budget):
                f010.train(source, audit.geometry.C010, scales)
    finally:
        budget.check()


if __name__ == '__main__':
    main()
