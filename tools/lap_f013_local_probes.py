"""Minibatch gradients and in-memory finite steps. No checkpoint is written."""
import hashlib
import math

import lap_b8_adamw as b8
import lap_b8_swa_continuation as continuation
import lap_f011_normalized_chemistry as arm
import lap_f013_geometry_audit as audit
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import numpy as np

PROBE = audit.OUT / 'probes'
# Executed update norm of the step that left the checkpoint. t251 has no next step.
STEP_NORM = {
    'b8_t128': 0.0006498670299681904,
    'f010_t128': 0.0009123532029559539,
    'f012_t128': 0.0009340652217141161,
}
FINITE = (
    ('b8_t128', 'common'),
    ('b8_t128', 'neg_f010'),
    ('b8_t128', 'adamw_f010'),
    ('b8_t128', 'extended'),
    ('f010_t128', 'common'),
    ('f010_t128', 'neg_f010'),
    ('f010_t128', 'adamw_f010'),
    ('f012_t128', 'common'),
    ('f012_t128', 'neg_f012'),
    ('f012_t128', 'adamw_f012'),
)
SCALES = (0.25, 1.0)
PRIORITY = ('b8_t128', 'f010_t128', 'f012_t128', 'f010_t251', 'f012_t251')


def _flat(raw, names):
    return np.concatenate([
        np.asarray(raw[name].detach().cpu(), dtype=np.float64).reshape(-1) for name in names
    ])


def _hash_tensors(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _apply(parameters, delta):
    import torch
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            piece = np.ascontiguousarray(delta[offset:offset + count])
            parameter.add_(torch.from_numpy(piece).to(parameter).reshape_as(parameter))
            offset += count
    if offset != len(delta):
        raise RuntimeError('diagnostic displacement does not match the parameter vector')


def _scaled(vector, norm):
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise RuntimeError('diagnostic direction is zero')
    return vector * (norm / length)


def displacements(state, param, moments, norm):
    components = [state['components'][task] for task in audit.TASKS]
    solved = audit.maxmin([audit.unit(vector)[0] for vector in components])
    extended = [(task, state['components'][task]) for task in audit.TASKS]
    extended += [(name, state['pieces'][name]) for name in audit.PROBLEM_DB]
    extra = audit.solve_named(extended)
    f010 = audit.adamw_parts(param, state['f010'], moments['moment'], moments['second'], moments['step'])
    f012 = audit.adamw_parts(param, state['f012'], moments['moment'], moments['second'], moments['step'])
    directions = {
        'common': None if solved['direction'] is None else _scaled(-solved['direction'], norm),
        'neg_f010': _scaled(-state['f010'], norm),
        'neg_f012': _scaled(-state['f012'], norm),
        'adamw_f010': _scaled(f010['exact'], norm),
        'adamw_f012': _scaled(f012['exact'], norm),
        'extended': None if extra.get('direction') is None else _scaled(-extra['direction'], norm),
    }
    return directions, solved, extra


def _scalars(model, shadow, bundle, dispersion, mrks_dispersion, scales):
    import torch
    specs = arm._selected(bundle)
    residuals = []
    databases = []
    for spec in specs:
        reaction = b8._load_reaction(bundle, 'relchem', spec)
        with torch.no_grad():
            prediction, reference = j251._prediction(shadow, reaction, dispersion)
        residuals.append(float((prediction - reference).detach().cpu()))
        databases.append(spec['database'])
        del reaction, prediction, reference
    ae17 = arm.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
    mrks = arm.mrks_scalars(model, bundle, mrks_dispersion)
    return {
        'jw': j251.j251_wtmad_style(residuals, databases, scales),
        'ae17': ae17,
        'exc': mrks['exc'],
        'op': mrks['op'],
        'finite': all(math.isfinite(value) for value in (ae17, mrks['exc'], mrks['op'])),
    }


def _constraints(model):
    import torch
    from test_lap import raw_grid

    from dft_functionals.constants import PBE_CONSTANTS
    grid = raw_grid().to(device='cuda', dtype=torch.float32)
    first = model(grid)
    swapped = model(grid[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]])
    expected = first[:, list(range(22)) + [24, 25, 22, 23, 27, 26, 28]]
    return {
        'spin_max_abs': float((swapped - expected).abs().max().detach().cpu()),
        'passthrough_max_abs': float((first[:, 2:22] - PBE_CONSTANTS.to(first)[0, 2:22]).abs().max().detach().cpu()),
        'output_finite': bool(torch.isfinite(first).all()),
    }


def minibatch(scales, names):
    import torch
    source = b8.verify_sources()
    run = b8._import_run()
    manifest = b8.load_j251_rows()
    if b8.sha256_file(continuation.PARENT_MANIFEST) != continuation.MANIFEST_SHA:
        audit._stop('Training manifest hash changed')
    batches = b8.build_batches(manifest)
    if b8.read_json(continuation.PARENT_MANIFEST) != batches:
        audit._stop('Rebuilt batches do not match the frozen manifest')
    bundle = run.PublicationDataset(run.DATA)
    dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = b8.read_json(run.DATA / 'mrks' / 'dispersion.json')
    rows = []
    for spec in audit.STATES:
        if spec['name'] not in PRIORITY:
            continue
        state = audit.load_state(spec, scales)
        index_of = {row['identity']: index for index, row in enumerate(state['rows'])}
        model, shadow, _run = arm._load_eval_model(source, spec['checkpoint'])
        for cursor in audit.MINIBATCH_INDICES:
            receipt = PROBE / f'minibatch_{spec["name"]}_{cursor}.json'
            if receipt.is_file():
                rows.append(b8.read_json(receipt))
                continue
            chosen = batches[cursor]
            chemistry = np.zeros(9446)
            for item in chosen['relchem']:
                index = index_of[item['identity']]
                chemistry += fact.deriv_l1(state['residuals'][index]) / scales[state['databases'][index]] * state['jacobian'][index]
            chemistry /= 8.0
            _losses, raw, system_name, _seconds = b8.other_tasks(
                model, shadow, bundle, chosen, dispersion, mrks_dispersion)
            physical = {task: _flat(raw[task], names) for task in ('ae17', 'exc', 'op')}
            record = {
                'checkpoint': spec['name'], 'cursor': cursor, 'mrks_id': system_name,
                'identities': [item['identity'] for item in chosen['relchem']],
                'jw_cosine': float(np.dot(chemistry, state['components']['jw']) / (
                    np.linalg.norm(chemistry) * np.linalg.norm(state['components']['jw']))),
                'jw_norm_ratio': float(np.linalg.norm(chemistry) / np.linalg.norm(state['components']['jw'])),
            }
            for policy, op_weight in (('f010', b8.LAMBDAS['op']), ('f012', 2.0 * b8.LAMBDAS['op'])):
                composite = (
                    b8.LAMBDAS['relchem'] * audit.C010 * chemistry
                    + b8.LAMBDAS['ae17'] * physical['ae17']
                    + b8.LAMBDAS['exc'] * physical['exc']
                    + op_weight * physical['op']
                )
                full = state[policy]
                record[f'{policy}_cosine'] = float(np.dot(composite, full) / (np.linalg.norm(composite) * np.linalg.norm(full)))
                for task in audit.TASKS:
                    record[f'{policy}_dot_{task}'] = float(np.dot(state['components'][task], -composite))
                    record[f'full_{policy}_dot_{task}'] = float(np.dot(state['components'][task], -full))
            b8.write_json(receipt, record)
            rows.append(record)
            print('F013_BATCH', spec['name'], cursor, f"{record['jw_cosine']:.4f}", f"{record['f012_cosine']:.4f}", flush=True)
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
    rows = []
    cached = {}
    for name, direction_name in FINITE:
        for scale in SCALES:
            if direction_name == 'extended' and scale != 1.0:
                continue
            receipt = PROBE / f'finite_{name}_{direction_name}_{scale}.json'
            if receipt.is_file():
                rows.append(b8.read_json(receipt))
                continue
            spec = next(item for item in audit.STATES if item['name'] == name)
            if name not in cached:
                state = audit.load_state(spec, scales)
                saved = audit.torch_load(spec['checkpoint'])
                moments = audit.traj._moments(saved, [int(saved['model'][item].numel()) for item in names])
                param = audit.traj._flat_state(saved, names)
                cached[name] = (state, displacements(state, param, moments, STEP_NORM[name])[0])
            state, directions = cached[name]
            delta = directions[direction_name]
            if delta is None:
                audit._stop(name + ' ' + direction_name + ' has no certified direction')
            model, shadow, _run = arm._load_eval_model(source, spec['checkpoint'])
            named = dict(model.named_parameters())
            parameters = [named[item] for item in names]
            before = _hash_tensors(parameters)
            clones = [parameter.detach().clone() for parameter in parameters]
            rng = b8._capture_rng()
            _apply(parameters, scale * delta)
            shadow.load_state_dict(model.state_dict(), strict=True)
            observed = _scalars(model, shadow, bundle, dispersion, mrks_dispersion, scales)
            limits = _constraints(model)
            with torch.no_grad():
                for parameter, clone in zip(parameters, clones, strict=True):
                    parameter.copy_(clone)
            after = _hash_tensors(parameters)
            b8._restore_rng(rng)
            if after != before:
                audit._stop('Finite-step restore changed parameter bytes')
            physical = b8.read_json(spec['physical'].with_name('physical.json')) if spec['physical'].suffix == '.npz' else {
                'ae17': 3.27546495584018, 'exc': 14.679044990002591, 'op': 0.030933124717599843,
            }
            predicted = {task: float(np.dot(state['components'][task], scale * delta)) for task in audit.TASKS}
            record = {
                'checkpoint': name, 'direction': direction_name, 'scale': scale,
                'step_norm': scale * STEP_NORM[name],
                'predicted': predicted,
                'baseline': {'jw': state['jw_scalar'], **physical},
                'observed': observed,
                'constraints': limits,
                'restored': True,
                'output_finite': observed['finite'] and limits['output_finite'],
            }
            b8.write_json(receipt, record)
            rows.append(record)
            print('F013_STEP', name, direction_name, scale, f"{observed['jw']:.6f}", f"{observed['op']:.6f}", flush=True)
            del model, shadow
            torch.cuda.empty_cache()
    return rows


def execute():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    PROBE.mkdir(parents=True, exist_ok=True)
    scales = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')['scales']
    names = tuple(b8.read_json(audit.prior.OUT / 'b8_t128' / 'result.json')['parameter_names'])
    batches = minibatch(scales, names)
    steps = finite_steps(scales, names)
    b8.write_json(audit.OUT / 'probe_summary.json', {'batches': batches, 'steps': steps, 'optimizer_updates': 0})
    print('F013_PROBE_DONE', len(batches), len(steps), flush=True)


if __name__ == '__main__':
    execute()
