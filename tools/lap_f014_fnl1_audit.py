"""Read-only audit of one fixed nonlinear scalarization. No optimizer step."""
import math

import lap_b8_adamw as b8
import lap_b8_swa_continuation as continuation
import lap_f011_normalized_chemistry as arm
import lap_f011_trajectory_gradient_audit as traj
import lap_f013_geometry_audit as f013
import lap_f013_local_probes as probe
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np

PROTOCOL = 'f014-fnl1-v1'
OUT = b8.REPO.parent / 'lap_f014_fnl1_audit_20261012'
B = {
    'jw': 0.1564146381387076,
    'ae17': 3.27546495584018,
    'exc': 14.679044990002591,
    'op': 0.030933124717599843,
}
S = {
    'jw': 1.5115724830787063,
    'ae17': 5597.740971893248,
    'exc': 20120.43676933683,
    'op': 0.6133549684729634,
}
MOMENT_SHA = '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
REFERENCE = probe.STEP_NORM
TASKS = f013.TASKS
DATABASES = ('ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13')
PROBLEM = f013.PROBLEM_DB


def _stop(reason):
    OUT.mkdir(parents=True, exist_ok=True)
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def weights(losses):
    return {task: losses[task] / (B[task] * S[task]) for task in TASKS}


def scalar(losses):
    return 0.5 * sum(losses[task] ** 2 / (B[task] * S[task]) for task in TASKS)


def fnl1_gradient(components, losses):
    coefficient = weights(losses)
    total = sum(coefficient[task] * components[task] for task in TASKS)
    parts = {task: coefficient[task] * components[task] for task in TASKS}
    return total, coefficient, parts


def adamw_reset(param, grad):
    """First AdamW step from zero moments and step counter 0.

    Bias correction cancels the (1-beta) factors, leaving g / (|g| + eps).
    """
    preconditioned = grad / (np.abs(grad) + 1e-8)
    return -2.5e-5 * (preconditioned + 0.01 * param)


def _physical(spec):
    if spec['name'] == 'b8_t128':
        return dict(B)
    return b8.read_json(spec['physical'].with_name('physical.json'))


def _losses(state, physical):
    return {'jw': state['jw_scalar'], 'ae17': physical['ae17'], 'exc': physical['exc'], 'op': physical['op']}


def _database_losses(residuals, databases, scales):
    totals = {name: 0.0 for name in DATABASES}
    for residual, database in zip(residuals, databases, strict=True):
        totals[database] += math.sqrt(1e-20 + residual * residual) / scales[database]
    return {name: value / 251.0 for name, value in totals.items()}


def geometry(scales):
    rows = []
    database_rows = []
    adamw_rows = []
    loaded = {}
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    for spec in f013.STATES:
        state = f013.load_state(spec, scales)
        physical = _physical(spec)
        losses = _losses(state, physical)
        if spec['name'] == 'b8_t128':
            for task in TASKS:
                norm = float(np.linalg.norm(state['components'][task]))
                if abs(norm - S[task]) > 1e-12 or abs(losses[task] - B[task]) > 1e-12:
                    _stop('B8 reference scale does not match F013')
        gradient, coefficient, parts = fnl1_gradient(state['components'], losses)
        units = {task: f013.unit(state['components'][task])[0] for task in TASKS}
        solved = f013.maxmin([units[task] for task in TASKS])
        extended = f013.solve_named(
            [(task, state['components'][task]) for task in TASKS]
            + [(name, state['pieces'][name]) for name in PROBLEM]
        )
        direction, grad_norm = f013.unit(gradient)
        margins = {task: float(np.dot(units[task], direction)) for task in TASKS}
        worst = min(margins.values())
        saved = f013.torch_load(spec['checkpoint'])
        numels = [int(saved['model'][name].numel()) for name in names]
        moments = traj._moments(saved, numels)
        param = traj._flat_state(saved, names)
        if spec['name'] == 'b8_t128':
            if moments['step'] != 128 or continuation.moment_sha(saved['optimizer']) != MOMENT_SHA:
                _stop('B8 AdamW receipt does not match')
            equal = [float(np.linalg.norm(parts[task])) for task in TASKS]
            if max(abs(value - 1.0) for value in equal) > 1e-8:
                _stop('B8 FNL1 contributions are not unit vectors')
        historical = f013.adamw_parts(param, gradient, moments['moment'], moments['second'], moments['step'])
        reset = adamw_reset(param, gradient)
        for label, delta in (('historical', historical['exact']), ('reset', reset)):
            adamw_rows.append({
                'checkpoint': spec['name'], 'moments': label,
                'displacement_norm': float(np.linalg.norm(delta)),
                'cosine_with_fnl1': prior.cosine(delta, -direction),
                'worst_task_margin': min(float(np.dot(units[task], -delta / np.linalg.norm(delta))) for task in TASKS),
                **{f'dot_{task}': float(np.dot(state['components'][task], delta)) for task in TASKS},
                **{f'dot_{name}': float(np.dot(state['pieces'][name], delta)) for name in DATABASES},
            })
        f010_margin = min(float(np.dot(units[task], f013.unit(state['f010'])[0])) for task in TASKS)
        f012_margin = min(float(np.dot(units[task], f013.unit(state['f012'])[0])) for task in TASKS)
        rows.append({
            'checkpoint': spec['name'],
            'fnl1_scalar': scalar(losses),
            'fnl1_gradient_norm': grad_norm,
            'cancellation': prior.cancellation_ratio(tuple(parts.values())),
            **{f'coefficient_{task}': coefficient[task] for task in TASKS},
            **{f'contribution_norm_{task}': float(np.linalg.norm(parts[task])) for task in TASKS},
            **{f'descent_{task}': float(np.dot(state['components'][task], direction)) for task in TASKS},
            'worst_task_margin': worst,
            'f013_margin': solved['gamma'],
            'f010_margin': f010_margin,
            'f012_margin': f012_margin,
            'cosine_f013': prior.cosine(direction, solved['direction']),
            'inside_cone': worst > 1e-8,
        })
        candidates = {
            'neg_fnl1': -direction,
            'neg_f010': -f013.unit(state['f010'])[0],
            'neg_f012': -f013.unit(state['f012'])[0],
            'f013_common': -solved['direction'],
            'f013_extended': None if extended.get('direction') is None else -extended['direction'],
        }
        for label, step in candidates.items():
            if step is None:
                continue
            dots = {name: float(np.dot(state['pieces'][name], step)) for name in DATABASES}
            for name, value in dots.items():
                piece = state['pieces'][name]
                database_rows.append({
                    'checkpoint': spec['name'], 'direction': label, 'database': name,
                    'directional_derivative': value,
                    'normalized': prior.cosine(piece, step),
                    'improves': value < 0.0,
                })
        loaded[spec['name']] = {
            'state': state, 'losses': losses, 'gradient': gradient, 'direction': direction,
            'solved': solved, 'extended': extended, 'param': param, 'moments': moments,
            'historical': historical['exact'],
        }
        print('F014_STATE', spec['name'], f'{worst:.6f}', rows[-1]['inside_cone'], flush=True)
    return rows, database_rows, adamw_rows, loaded, names


def chemistry_expectation(scales, loaded):
    """Exact sampler average of the naive JW term. Physical tasks are not included."""
    rows = []
    manifest = b8.load_j251_rows()
    batches = b8.build_batches(manifest)
    if b8.sha256_file(continuation.PARENT_MANIFEST) != continuation.MANIFEST_SHA:
        _stop('Training manifest hash changed')
    for name in ('b8_t128', 'f010_t128', 'f012_t128'):
        state = loaded[name]['state']
        index_of = {row['identity']: index for index, row in enumerate(state['rows'])}
        coefficient = weights(loaded[name]['losses'])['jw']
        exact = coefficient * state['components']['jw']
        accumulator = np.zeros(9446)
        for batch in batches:
            chosen = []
            for item in batch['relchem']:
                index = index_of[item['identity']]
                chosen.append((state['residuals'][index], state['databases'][index], state['jacobian'][index]))
            loss = sum(math.sqrt(1e-20 + residual * residual) / scales[database] for residual, database, _row in chosen) / 8.0
            grad = sum(fact.deriv_l1(residual) / scales[database] * row for residual, database, row in chosen) / 8.0
            accumulator += (loss / (B['jw'] * S['jw'])) * grad
        mean = accumulator / len(batches)
        rows.append({
            'checkpoint': name, 'term': 'jw_population',
            'cosine': prior.cosine(mean, exact),
            'relative_l2': float(np.linalg.norm(mean - exact) / np.linalg.norm(exact)),
        })
    return rows


def execute_geometry():
    OUT.mkdir(parents=True, exist_ok=True)
    scales = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')['scales']
    if arm.scales_sha256(scales) != '3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d':
        _stop('A_d hash changed')
    rows, database_rows, adamw_rows, loaded, names = geometry(scales)
    population = chemistry_expectation(scales, loaded)
    b8.write_json(OUT / 'geometry.json', {
        'rows': rows, 'databases': database_rows, 'adamw': adamw_rows,
        'population': population, 'names': list(names), 'optimizer_updates': 0,
    })
    print('F014_GEOMETRY_DONE', flush=True)
    return loaded


def _write_tables():
    payload = b8.read_json(OUT / 'geometry.json')
    root = b8.REPO

    def csv(path, rows):
        import csv as csv_module
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open('w', newline='\n', encoding='utf-8') as handle:
            writer = csv_module.DictWriter(handle, fieldnames=fields, restval='', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    csv(root / 'lap_f014_fnl1_checkpoint_geometry.csv', payload['rows'])
    csv(root / 'lap_f014_fnl1_database_directions.csv', payload['databases'])
    csv(root / 'lap_f014_fnl1_adamw_counterfactuals.csv', payload['adamw'])
    stochastic = list(payload['population'])
    stochastic.extend(b8.read_json(path) for path in sorted((OUT / 'probes').glob('stochastic_*.json')))
    if stochastic:
        csv(root / 'lap_f014_fnl1_stochastic_estimator.csv', stochastic)
    finite = [b8.read_json(path) for path in sorted((OUT / 'probes').glob('finite_*.json'))]
    if finite:
        flat = []
        for row in finite:
            for task in list(TASKS) + list(DATABASES):
                if task not in row['predicted']:
                    continue
                observed = row['observed'][task] - row['baseline'][task]
                predicted = row['predicted'][task]
                flat.append({
                    'checkpoint': row['checkpoint'], 'direction': row['direction'], 'scale': row['scale'],
                    'task': task, 'predicted': predicted, 'observed': observed,
                    'sign_agrees': predicted * observed > 0.0,
                    'restored': row['restored'], 'spin_max_abs': row['constraints']['spin_max_abs'],
                })
        csv(root / 'lap_f014_fnl1_finite_step.csv', flat)


def _observed(model, shadow, bundle, dispersion, mrks_dispersion, scales):
    import torch
    specs = arm._selected(bundle)
    residuals = []
    databases = []
    for item in specs:
        reaction = b8._load_reaction(bundle, 'relchem', item)
        with torch.no_grad():
            prediction, reference = j251._prediction(shadow, reaction, dispersion)
        residuals.append(float((prediction - reference).detach().cpu()))
        databases.append(item['database'])
        del reaction, prediction, reference
    ae17 = arm.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
    mrks = arm.mrks_scalars(model, bundle, mrks_dispersion)
    observed = {
        'jw': j251.j251_wtmad_style(residuals, databases, scales),
        'ae17': ae17, 'exc': mrks['exc'], 'op': mrks['op'],
        'finite': all(math.isfinite(value) for value in (ae17, mrks['exc'], mrks['op'])),
    }
    observed.update(_database_losses(residuals, databases, scales))
    return observed


def _naive_gradient(losses, grads):
    coefficient = weights(losses)
    return sum(coefficient[task] * grads[task] for task in TASKS), coefficient


def stochastic(scales, names):
    import torch
    source = b8.verify_sources()
    run = b8._import_run()
    batches = b8.build_batches(b8.load_j251_rows())
    bundle = run.PublicationDataset(run.DATA)
    dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
    probes = OUT / 'probes'
    probes.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in f013.STATES:
        if spec['name'] not in REFERENCE:
            continue
        state = f013.load_state(spec, scales)
        full, _coefficient, _parts = fnl1_gradient(state['components'], _losses(state, _physical(spec)))
        index_of = {row['identity']: index for index, row in enumerate(state['rows'])}
        model, shadow, _run = arm._load_eval_model(source, spec['checkpoint'])
        for cursor in f013.MINIBATCH_INDICES:
            receipt = probes / f'stochastic_{spec["name"]}_{cursor}.json'
            if receipt.is_file():
                rows.append(b8.read_json(receipt))
                continue
            chosen = batches[cursor]
            chemistry_grad = np.zeros(9446)
            chemistry_loss = 0.0
            for item in chosen['relchem']:
                index = index_of[item['identity']]
                residual = state['residuals'][index]
                database = state['databases'][index]
                chemistry_loss += math.sqrt(1e-20 + residual * residual) / scales[database]
                chemistry_grad += fact.deriv_l1(residual) / scales[database] * state['jacobian'][index]
            chemistry_loss /= 8.0
            chemistry_grad /= 8.0
            task_losses, raw, system_name, _seconds = b8.other_tasks(
                model, shadow, bundle, chosen, dispersion, mrks_dispersion)
            estimates = {
                'jw': chemistry_loss,
                'ae17': task_losses['ae17'],
                'exc': task_losses['exc'],
                'op': task_losses['op'],
            }
            grads = {
                'jw': chemistry_grad,
                'ae17': probe._flat(raw['ae17'], names),
                'exc': probe._flat(raw['exc'], names),
                'op': probe._flat(raw['op'], names),
            }
            naive, coefficient = _naive_gradient(estimates, grads)
            record = {
                'checkpoint': spec['name'], 'cursor': cursor, 'mrks_id': system_name,
                'cosine_full': prior.cosine(naive, full),
                'relative_l2': float(np.linalg.norm(naive - full) / np.linalg.norm(full)),
                **{f'coefficient_{task}': coefficient[task] for task in TASKS},
                **{f'full_descent_{task}': float(np.dot(state['components'][task], f013.unit(full)[0])) for task in TASKS},
                **{f'naive_descent_{task}': float(np.dot(state['components'][task], f013.unit(naive)[0])) for task in TASKS},
            }
            b8.write_json(receipt, record)
            rows.append(record)
            print('F014_BATCH', spec['name'], cursor, f"{record['cosine_full']:.4f}", flush=True)
        del model, shadow
        torch.cuda.empty_cache()
    return rows


def finite_steps(scales, names):
    import torch
    source = b8.verify_sources()
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
    probes = OUT / 'probes'
    probes.mkdir(parents=True, exist_ok=True)
    rows = []
    cached = {}
    for name in REFERENCE:
        for direction_name in ('neg_fnl1', 'adamw_fnl1'):
            for scale in (0.25, 1.0):
                receipt = probes / f'finite_{name}_{direction_name}_{scale}.json'
                if receipt.is_file():
                    rows.append(b8.read_json(receipt))
                    continue
                spec = next(item for item in f013.STATES if item['name'] == name)
                if name not in cached:
                    state = f013.load_state(spec, scales)
                    losses = _losses(state, _physical(spec))
                    gradient, _coefficient, _parts = fnl1_gradient(state['components'], losses)
                    saved = f013.torch_load(spec['checkpoint'])
                    moments = traj._moments(saved, [int(saved['model'][item].numel()) for item in names])
                    param = traj._flat_state(saved, names)
                    historical = f013.adamw_parts(
                        param, gradient, moments['moment'], moments['second'], moments['step'])['exact']
                    cached[name] = (state, losses, {
                        'neg_fnl1': probe._scaled(-gradient, REFERENCE[name]),
                        'adamw_fnl1': probe._scaled(historical, REFERENCE[name]),
                    })
                state, losses, directions = cached[name]
                delta = directions[direction_name]
                model, shadow, _run = arm._load_eval_model(source, spec['checkpoint'])
                named = dict(model.named_parameters())
                parameters = [named[item] for item in names]
                before = probe._hash_tensors(parameters)
                file_hash = b8.sha256_file(spec['checkpoint'])
                clones = [parameter.detach().clone() for parameter in parameters]
                rng = b8._capture_rng()
                probe._apply(parameters, scale * delta)
                shadow.load_state_dict(model.state_dict(), strict=True)
                observed = _observed(model, shadow, bundle, dispersion, mrks_dispersion, scales)
                limits = probe._constraints(model)
                with torch.no_grad():
                    for parameter, clone in zip(parameters, clones, strict=True):
                        parameter.copy_(clone)
                after = probe._hash_tensors(parameters)
                b8._restore_rng(rng)
                if after != before or b8.sha256_file(spec['checkpoint']) != file_hash:
                    _stop('Finite-step restore changed parameter bytes')
                baseline = dict(losses)
                baseline.update(_database_losses(state['residuals'], state['databases'], scales))
                predicted = {
                    task: float(np.dot(state['components'][task], scale * delta)) for task in TASKS
                }
                predicted.update({
                    db: float(np.dot(state['pieces'][db], scale * delta)) for db in DATABASES
                })
                record = {
                    'checkpoint': name, 'direction': direction_name, 'scale': scale,
                    'predicted': predicted, 'baseline': baseline, 'observed': observed,
                    'constraints': limits, 'restored': True, 'checkpoint_unchanged': True,
                    'output_finite': observed['finite'] and limits['output_finite'],
                }
                b8.write_json(receipt, record)
                rows.append(record)
                print('F014_STEP', name, direction_name, scale, f"{observed['jw']:.6f}", f"{observed['op']:.6f}", flush=True)
                del model, shadow
                torch.cuda.empty_cache()
    return rows


def execute_gpu():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    scales = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')['scales']
    names = tuple(b8.read_json(prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    stochastic(scales, names)
    finite_steps(scales, names)
    _write_tables()
    print('F014_GPU_DONE', flush=True)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    payload = b8.read_json(OUT / 'geometry.json')
    rows = payload['rows']
    names = [row['checkpoint'] for row in rows]
    root = b8.REPO
    figure, axis = plt.subplots(figsize=(8, 4))
    for task in TASKS:
        axis.plot(range(len(rows)), [row[f'contribution_norm_{task}'] for row in rows], marker='o', label=task)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=8)
    axis.set_ylabel('contribution norm')
    figure.savefig(root / 'lap_f014_fnl1_contributions.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    for key, label in (('worst_task_margin', 'FNL1'), ('f010_margin', 'F010'), ('f012_margin', 'F012')):
        axis.plot(range(len(rows)), [row[key] for row in rows], marker='o', label=label)
    axis.axhline(0, color='black', lw=0.6)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=8)
    figure.savefig(root / 'lap_f014_fnl1_margins.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(len(rows)), [row['worst_task_margin'] for row in rows], marker='o', label='FNL1')
    axis.plot(range(len(rows)), [row['f013_margin'] for row in rows], marker='o', label='F013 max-min')
    axis.axhline(0, color='black', lw=0.6)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=8)
    figure.savefig(root / 'lap_f014_fnl1_vs_mgda.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    chosen = [row for row in payload['databases'] if row['direction'] == 'neg_fnl1']
    for database in PROBLEM:
        series = [row['directional_derivative'] for row in chosen if row['database'] == database]
        axis.plot(range(len(names)), series, marker='o', label=database)
    axis.axhline(0, color='black', lw=0.6)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=7)
    axis.set_ylabel('g · step  (negative improves)')
    figure.savefig(root / 'lap_f014_fnl1_databases.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    historical = [row for row in payload['adamw'] if row['moments'] == 'historical']
    axis.plot(range(len(historical)), [row['dot_op'] for row in historical], marker='o', label='operator g·Δθ')
    axis.plot(range(len(historical)), [row['worst_task_margin'] for row in historical], marker='o', label='AdamW cone margin')
    axis.axhline(0, color='black', lw=0.6)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=8)
    figure.savefig(root / 'lap_f014_fnl1_adamw.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    stochastic = [b8.read_json(path) for path in sorted((OUT / 'probes').glob('stochastic_*.json'))]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.scatter([row['cursor'] for row in stochastic], [row['cosine_full'] for row in stochastic], c='#3d5a80')
    axis.axhline(0, color='black', lw=0.6)
    axis.set_ylabel('cosine of naive minibatch with full FNL1')
    axis.set_xlabel('predeclared cursor')
    figure.savefig(root / 'lap_f014_fnl1_minibatch.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    finite = []
    for path in sorted((OUT / 'probes').glob('finite_*.json')):
        row = b8.read_json(path)
        for task in TASKS:
            finite.append((row['predicted'][task], row['observed'][task] - row['baseline'][task]))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter([item[0] for item in finite], [item[1] for item in finite], s=18, c='#3d5a80')
    limit = max(abs(value) for pair in finite for value in pair)
    axis.plot([-limit, limit], [-limit, limit], color='#9b2226', lw=0.8)
    axis.set_xlabel('first-order prediction')
    axis.set_ylabel('observed change')
    figure.savefig(root / 'lap_f014_fnl1_finite_step.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    print('F014_RENDER_DONE', flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        execute_gpu()
    elif len(sys.argv) > 1 and sys.argv[1] == 'render':
        render()
    else:
        execute_geometry()
        _write_tables()
