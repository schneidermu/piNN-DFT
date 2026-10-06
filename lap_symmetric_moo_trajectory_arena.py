"""Bounded eight-arm experiment using qualified objectives and direct Armijo hook."""
import argparse
import copy
import hashlib
import random
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'train_models'))

import lap_moo_training as core
import lap_training
from lap_moo_panel import MRKS_PANEL_SYSTEMS, MinnesotaGroupStore
from lap_moo_protocol import FOUR_TASK_NAMES, read_sampling_manifest
from lap_moo_training import (
    capture_rng_state,
    compute_isolated_task_gradients,
    make_mrks_objective_factories,
    make_reaction_objective,
    named_trainable_parameters,
    restore_rng_state,
    train_moo_update,
)
from optuna_joint import batch_fchem, compute_fchem_from_errors, load_mrks_dispersions
from train_lap import load_reaction_dispersions
from train_lap_moo import CentralAOCache, _four_task_objectives
from train_lap_s5 import _pilot_model

import lap_symmetric_moo_offline_arena as offline

OUT = ROOT.parent/'lap_symmetric_moo_10update_runs_20261006'
MANIFEST = ROOT.parent/'lap_25_runs_20261005/sampling_manifest_25.json'
FULL = ROOT.parent/'lap_chem_full268_eval_runs_20261004/sampling_manifest_268.json'
ORDER = FOUR_TASK_NAMES
ALPHA_PCD = 6.632573669086685e-7


def digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def aggregate(raw, selected, state, order, solver_sha):
    if selected not in ('UNIT_MAXMIN', 'Nash-MTL'):
        raise ValueError('Unknown experimental method')
    names = tuple(sorted(raw[order[0]]))
    matrix = np.stack([torch.cat([raw[t][n].detach().double().cpu().flatten() for n in names]).numpy() for t in order])
    if selected == 'UNIT_MAXMIN':
        native, meta = offline.maxmin(matrix, solver_sha)
        if meta['gamma_star'] <= 1e-12:
            raise RuntimeError('UNIT_MAXMIN gamma<=frozen feasibility tolerance1e-12')
        state_after = {}
        meta['active_tasks'] = [order[i] for i, v in enumerate(meta['gradient_space_products'])
                                if abs(v-meta['gamma_star']) <= 1e-9]
    else:
        joint, meta, state_after = offline.aggregate_task_gradients(raw, method='nash_mtl',
               hyperparameters={'max_iter': 100, 'tol': 1e-10}, state=copy.deepcopy(state), task_order=order)
        native = torch.cat([joint[n].detach().double().cpu().flatten() for n in names]).numpy()
    norm = np.linalg.norm(native)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError('zero/nonfinite native direction')
    unit = native/norm
    progress = offline.progress(matrix, unit, meta.get('gamma_star', 1))
    norms = np.linalg.norm(matrix, axis=1)
    gram = matrix@matrix.T
    eigen = np.linalg.eigvalsh(gram)
    rank = int(np.sum(eigen > eigen[-1]*4*np.finfo(float).eps))
    meta.update({'arena_method': selected, 'controller_api_compatibility_tag': 'pcd; custom aggregator hook, no PCD calculation',
                 'native_norm': float(norm), 'normalized_progress': progress,
                 'task_gradient_norms': dict(zip(order, norms.tolist())), 'raw_gram': gram.tolist(),
                 'raw_gram_rank': rank, 'raw_gram_condition': float(eigen[-1]/eigen[0]) if rank == 4 else None,
                 'task_cosine_matrix': (gram/np.outer(norms, norms)).tolist(), 'feasible': progress['common_descent']})
    result, cursor = {}, 0
    for name in names:
        template = raw[order[0]][name]
        count = template.numel()
        result[name] = torch.from_numpy(unit[cursor:cursor+count].copy()).reshape(template.shape).to(template.device)
        cursor += count
    assert cursor == len(unit)
    return result, meta, state_after, {'raw': matrix, 'native': native, 'unit': unit}


def endpoint_ratios(current, baseline):
    values = np.array([current[t]/baseline[t] for t in ORDER])
    assert np.all(values > 0) and np.isfinite(values).all()
    return {'task_ratios': dict(zip(ORDER, values.tolist())), 'R_max': float(values.max()),
            'R_mean': float(values.mean()), 'geometric_mean': float(np.exp(np.mean(np.log(values)))),
            'max_abs_log': float(np.max(np.abs(np.log(values)))), 'improved_objectives': int(np.sum(values < 1))}


def winner(summary):
    def finite_or_infinite(value):
        return float('inf') if value is None else value
    def key(name):
        s = summary[name]
        return (-s['trajectory_passes'], -s['scientific_passes'],
                s['worst_Rmax10'] if s['trajectory_passes'] == 4 else float('inf'),
                -s['paired_wins'], finite_or_infinite(s['median_Rmax10']), s['severe_events'], finite_or_infinite(s['median_Rmax5']))
    ranked = sorted(summary, key=key)
    preferred, other = ranked
    a, b = summary[preferred], summary[other]
    clear = (a['trajectory_passes'] == 4 and a['scientific_passes'] >= b['scientific_passes']
             and a['paired_wins'] >= 3 and a['worst_Rmax10'] < finite_or_infinite(b['worst_Rmax10'])
             and not a['systematic_pathology'])
    return {'lexicographic_preference': preferred, 'classification':
            ('CLEAR UNIT_MAXMIN WIN' if preferred == 'UNIT_MAXMIN' else 'CLEAR NASH WIN') if clear else 'INCONCLUSIVE'}


def advance_cursor(cursor, accepted):
    return cursor+1 if accepted else cursor


def checkpoint(path, model, state, cursor, protocol):
    torch.save({'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                'aggregator': copy.deepcopy(state), 'cursor': cursor, 'rng': capture_rng_state(),
                'model_sha256': digest(model), 'protocol_sha256': offline.reuse.sha(OUT/'protocol.json'),
                'manifest_sha256': protocol['manifest_file_sha256']}, path)


def resume(path, model, protocol):
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert data['protocol_sha256'] == offline.reuse.sha(OUT/'protocol.json')
    assert data['manifest_sha256'] == protocol['manifest_file_sha256'] == offline.reuse.sha(MANIFEST)
    verify(protocol)  # validation before model/RNG mutation.
    model.load_state_dict(data['model'], strict=True)
    assert digest(model) == data['model_sha256']
    restore_rng_state(data['rng'])
    return data['cursor'], data['aggregator']


def verify(protocol):
    for path, value in protocol['immutable_files'].items():
        assert offline.reuse.sha(path) == value, path
    assert offline.reuse.sha(__file__) == protocol['script_sha256']
    return len(protocol['immutable_files'])


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    base = offline.reuse.load(offline.OUT/'protocol.json')
    results = offline.reuse.load(ROOT/'lap_symmetric_moo_offline_arena_metrics.json')
    immutable = dict(base['immutable_files'])
    immutable[str(MANIFEST)] = 'eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc'
    immutable[str(FULL)] = offline.reuse.sha(FULL)
    manifest = read_sampling_manifest(MANIFEST)
    assert manifest['manifest_sha256'] == '2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3'
    assert len(manifest['entries']) == 25 and manifest['task_order'] == list(ORDER)
    assert manifest['seed'] == 41 and manifest['world_size'] == 1
    references = {}
    for binding in base['bindings']:
        tensors = torch.load(binding['gradient']['path'], map_location='cpu', weights_only=False)['aggregates']
        raw = {t: {row['name']: tensors[t][row['start']:row['stop']].reshape(row['shape'])
                  for row in base['parameter_order']} for t in offline.TASKS}
        direction, _, _ = offline.aggregate_task_gradients(raw, method='pcd',
                             hyperparameters=offline.METHODS['PCD_CONTROL'][1], state=None, task_order=offline.TASKS)
        norm = float(torch.cat([direction[row['name']].flatten() for row in base['parameter_order']]).norm())
        expected = results['states'][binding['key']]['candidates']['PCD_CONTROL']['native_norm']
        assert abs(norm/expected-1) <= 1e-12
        references[binding['key']] = {'PCD_native_norm': norm, 'eta0': ALPHA_PCD*norm}
    inputs = {'store': ROOT.parent/'lap_moo_runs_20261001/mn_group_store_268/manifest.json',
              'central': ROOT.parent/'lap_operator_runs_20261001/all90/manifest.json',
              'ao': ROOT.parent/'lap_moo_runs_20261001/mrks_15system_ao_cache/manifest.json',
              'disp': ROOT/'train_models/dispersions/dispersions.pickle',
              'mrksdisp': ROOT/'train_models/dispersions/dispersions_mrks.pickle'}
    immutable.update({str(path): offline.reuse.sha(path) for path in inputs.values()})
    source_keys = {'store': 'minnesota_group_store_manifest', 'central': 'central_operator_manifest',
                   'ao': 'ao_factor_cache_manifest', 'disp': 'reaction_dispersions', 'mrksdisp': 'mrks_dispersions'}
    assert all(immutable[str(inputs[key])] == manifest['source_hashes'][source]
               for key, source in source_keys.items())
    protocol = {'starting_commit': 'e0faea894a29e169ef29fc319e39b43a5ee6cc57', 'bindings': base['bindings'],
                'arms': [{'key': b['key']+'_'+m, 'start': b['key'], 'method': m}
                         for b in base['bindings'] for m in ('UNIT_MAXMIN', 'Nash-MTL')],
                'references': references, 'task_order': list(ORDER), 'parameter_order': base['parameter_order'],
                'manifest_path': str(MANIFEST), 'manifest_file_sha256': immutable[str(MANIFEST)],
                'manifest_canonical_sha256': manifest['manifest_sha256'], 'manifest_entries': list(range(11)),
                'user_supplied_canonical_sha256': '2e87f7e79a66b425628fdf59a821f785e05059f193a492d4a4f3130d7411e9f3',
                'canonical_label_discrepancy': 'Prompt fdf differs one character from computed bdf and prior committed25protocol; exact requested file-byte SHA agrees. Reuse byte-identical prior manifest, no regeneration.',
                'immutable_files': immutable, 'script_sha256': offline.reuse.sha(__file__),
                'inputs': {k: str(v) for k, v in inputs.items()}, 'systems': list(MRKS_PANEL_SYSTEMS),
                'runtime': 'main F32 CUDA; chemistry matched-F64 CUDA; repaired operator production path; chunk256/AO4096; seed41/world1',
                'armijo': {'c': 1e-4, 'rho': .5, 'max_backtracks': 20}, 'max_updates': 10,
                'monitoring': [0, 5, 10], 'stress_probe_entry': 10,
                'Nash': {'max_iter': 100, 'tol': 1e-10, 'warm_state': 'arm-local; commit only upon accepted update'},
                'maxmin_feasibility_tolerance': 1e-12, 'near_boundary_label': 1e-6,
                'shared_controller': 'Existing train_moo_update direct Armijo aggregator hook; method=pcd is API guard tag only, returned algorithm is UNIT_MAXMIN/Nash. Production sources unchanged.',
                'promotion': 'Clear winner requires all4 trajectory passes, scientific passes>=other, >=3 completed paired Rmax wins, better finite worst Rmax, no systematic pathology; otherwise INCONCLUSIVE.',
                'winner_lexicographic': ['trajectory passes', 'scientific passes', 'lower worst Rmax10', 'paired wins', 'lower median Rmax10', 'fewer severe controller events', 'lower median Rmax5'],
                'missing_endpoints': 'Unavailable, never fabricate ratios; ranking worst for any incomplete arm is infinite (serialized None with flag); paired Rmax wins only both completed.',
                'pathology_definition': 'Severe event: accepted eta/eta0<1e-3 or>=10backtracks. Systematic: median accepted eta/eta0<=1/16 or majority accepted steps>=4backtracks.',
                'resume_validation': 'At cursor5 reload same checkpoint after hash/protocol validation; require bitwise model, aggregator and RNG identity.',
                'baseline_cache': 'Paired t0 may reuse identical state-hash/source/protocol-bound full scalar receipt; never old different state baseline.'}
    verify(protocol)
    offline.reuse.dump(OUT/'protocol.json', protocol)
    print('FROZEN references', references, flush=True)


def _monitor_once(model, shadow, context, target):
    before, rng = digest(model), capture_rng_state()
    shadow.load_state_dict(model.state_dict(), strict=True)
    rows = []
    for entry in context['full']['entries']:
        sample = entry['per_rank'][0]
        ident = sample['reaction']
        reaction = context['store'].load_variant((ident['database'], ident['reaction_id']), sample['variant_suffix'])
        captured = {}
        def reducer(databases, prediction, reference, ident=ident, captured=captured):
            assert databases == [ident['database']] and prediction.dtype == torch.float64
            captured['error'] = float(prediction.detach())-float(reference.detach())
            return batch_fchem(databases, prediction, reference)
        with torch.no_grad(), patch.object(lap_training, 'batch_fchem', reducer):
            value = float(make_reaction_objective(shadow, reaction, device='cuda', dtype=torch.float64,
                                                  dispersions=context['disp'])())
        rows.append({**ident, 'loss': value, **captured})
        del reaction
    errors = {db: [r['error'] for r in rows if r['database'] == db] for db in sorted({r['database'] for r in rows})}
    nonae, _ = compute_fchem_from_errors({db: values for db, values in errors.items() if db != 'AE17'})
    values = {'relchem': sum(r['loss'] for r in rows if r['database'] != 'AE17')/251,
              'ae17': sum(r['loss'] for r in rows if r['database'] == 'AE17')/17}
    systems = []
    for name in MRKS_PANEL_SYSTEMS:
        system = context['systems'].load(name)
        factories = make_mrks_objective_factories(model, system, point_chunk_size=256, dispersions=context['mrksdisp'])
        local = {task: float(factory().detach()) for task, factory in zip(('exc', 'op'), factories)}
        systems.append({'system': name, **local})
        del system, factories
    for task in ('exc', 'op'):
        values[task] = float(np.mean([r[task] for r in systems]))
    assert all(np.isfinite(list(values.values()))) and all(value > 0 for value in values.values())
    assert digest(model) == before and all(p.grad is None for p in model.parameters())
    restore_rng_state(rng)
    assert equal(capture_rng_state(), rng)
    record = {'model_sha256': before, 'objectives': values, 'chemistry_rows': rows, 'systems': systems,
              'human': {'J_rel': values['relchem'], 'nonAE_RMSE': nonae,
                        'AE_MAE': float(np.mean(np.abs(errors['AE17']))),
                        'AE_RMSE': float(np.sqrt(np.mean(np.square(errors['AE17'])))),
                        'per_database': {db: {'MAE': float(np.mean(np.abs(v))), 'RMSE': float(np.sqrt(np.mean(np.square(v))))}
                                         for db, v in errors.items()},
                        **{task+'_median': float(np.median([r[task] for r in systems])) for task in ('exc', 'op')}},
              'read_only': True, 'protocol_sha256': offline.reuse.sha(OUT/'protocol.json')}
    offline.reuse.dump(target, record)
    print('MONITOR', target.name, values, flush=True)
    return record


def monitor(model, shadow, context, target):
    before, rng = digest(model), capture_rng_state()
    try:
        return _monitor_once(model, shadow, context, target)
    finally:
        assert digest(model) == before
        restore_rng_state(rng)
        assert equal(capture_rng_state(), rng)


def objectives(model, shadow, entry, context):
    sample = entry['per_rank'][0]
    system = context['systems'].load(sample['mrks_system'])
    return _four_task_objectives(model, shadow, sample, context['store'], system,
                                 context['disp'], context['mrksdisp'], 256)


def run():
    protocol = offline.reuse.load(OUT/'protocol.json')
    verify(protocol)
    manifest = read_sampling_manifest(MANIFEST)
    context = {'full': read_sampling_manifest(FULL),
               'store': MinnesotaGroupStore(protocol['inputs']['store'], cache_groups=1),
               'disp': load_reaction_dispersions(protocol['inputs']['disp']),
               'mrksdisp': load_mrks_dispersions(protocol['inputs']['mrksdisp']),
               'systems': CentralAOCache(Path(protocol['inputs']['central']).parent,
                            Path(protocol['inputs']['ao']).parent, device=torch.device('cuda'),
                            dtype=torch.float32, chunk_size=4096)}
    for arm in protocol['arms']:
        folder = OUT/arm['key']
        folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            print('existing finished arm, no restart', arm['key'], flush=True)
            continue
        binding = next(b for b in protocol['bindings'] if b['key'] == arm['start'])
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        torch.cuda.manual_seed_all(41)
        model = _pilot_model(torch.device('cuda'), torch.float32)
        model.load_state_dict(torch.load(binding['state']['path'], map_location='cpu', weights_only=False)['model'], strict=True)
        model.train()
        assert digest(model) == binding['state']['state_sha256']
        assert sum(p.numel() for p in named_trainable_parameters(model).values()) == 9446
        assert list(named_trainable_parameters(model)) == [row['name'] for row in protocol['parameter_order']]
        shadow = copy.deepcopy(model).double()
        eta0 = protocol['references'][arm['start']]['eta0']
        result = {'arm': arm, 'updates': [], 'checkpoints': {}, 'cursor': 0, 'status': 'RUNNING'}
        baseline_path = OUT/(arm['start']+'_baseline.json')
        baseline = offline.reuse.load(baseline_path) if baseline_path.exists() else monitor(model, shadow, context, baseline_path)
        assert baseline['model_sha256'] == digest(model) and baseline['protocol_sha256'] == offline.reuse.sha(OUT/'protocol.json')
        result['checkpoints']['0'] = baseline
        result['baseline_artifact'] = {'path': str(baseline_path), 'sha256': offline.reuse.sha(baseline_path)}
        state, previous = None, None
        latest = folder/'latest.pt'
        if latest.exists():
            result = offline.reuse.load(folder/'progress.json')
            cursor, state = resume(latest, model, protocol)
            assert cursor == result['cursor']
        else:
            checkpoint(latest, model, state, 0, protocol)
            shutil.copyfile(latest, folder/'checkpoint_0.pt')
            offline.reuse.dump(folder/'progress.json', result)
        started = time.perf_counter()
        phase = 'resume'
        try:
            while result['cursor'] < 10:
                cursor = result['cursor']
                phase = 'update'
                before, state_before = digest(model), copy.deepcopy(state)
                data = {}
                def hook(raw, selected=arm['method'], data=data, **kwargs):
                    names = tuple(sorted(raw[ORDER[0]]))
                    data['raw'] = np.stack([torch.cat([raw[t][n].detach().double().cpu().flatten()
                                                     for n in names]).numpy() for t in ORDER])
                    joint, meta, next_state, arrays = aggregate(raw, selected, kwargs['state'], ORDER,
                                               protocol['immutable_files'][str(offline.reuse.SOLVER)])
                    data.update(arrays)
                    return joint, meta, next_state
                factories = objectives(model, shadow, manifest['entries'][cursor], context)
                restorations = []
                original_restore = core._restore_model_state
                def checked_restore(target_model, target_state, original_restore=original_restore, restorations=restorations):
                    original_restore(target_model, target_state)
                    assert all(torch.equal(target_model.state_dict()[n], v) for n, v in target_state.items())
                    restorations.append({'exact': True, 'max_abs_difference': 0})
                with patch.object(core, '_restore_model_state', checked_restore):
                    update = train_moo_update(model, None, factories, method='pcd', aggregator_state=state,
                              aggregator=hook, world_size=1, task_order=ORDER, step_rule='pcd_direct_vector_armijo',
                              vector_armijo={'initial_step_size': eta0, **protocol['armijo']})
                record = {'cursor': cursor, 'sample': manifest['entries'][cursor], 'before_sha256': before,
                          'after_sha256': digest(model), 'accepted': update.accepted, 'stop_reason': update.stop_reason,
                          'losses': update.losses, 'diagnostics': update.diagnostics,
                          'restoration_checks': restorations, 'aggregator_state_before': state_before,
                          'aggregator_state_returned': update.aggregator_state}
                if data:
                    path = folder/f'update_{cursor}_geometry.npz'
                    np.savez_compressed(path, **data)
                    record['geometry_path'], record['geometry_sha256'] = str(path), offline.reuse.sha(path)
                    if previous is not None:
                        record['consecutive_direction_cosine'] = offline.reuse.cosine(previous, data['unit'])
                    previous = data['unit']
                result['updates'].append(record)
                del factories
                if not update.accepted:
                    assert digest(model) == before and equal(state, state_before)
                    result['status'] = 'FAILED_UPDATE'
                    break
                state = update.aggregator_state
                result['cursor'] = advance_cursor(result['cursor'], update.accepted)
                phase = 'checkpoint'
                checkpoint(latest, model, state, result['cursor'], protocol)
                offline.reuse.dump(folder/'progress.json', result)
                print('ARM', arm['key'], 'accepted', result['cursor'], 't', update.diagnostics['vector_armijo']['accepted_t'], flush=True)
                if result['cursor'] in (5, 10):
                    phase = 'monitor'
                    shutil.copyfile(latest, folder/f'checkpoint_{result["cursor"]}.pt')
                    target = folder/f'monitor_{result["cursor"]}.json'
                    panel = monitor(model, shadow, context, target)
                    panel['relative'] = endpoint_ratios(panel['objectives'], baseline['objectives'])
                    panel['system_ratios'] = {task: [r[task]/b[task] for r, b in zip(panel['systems'], baseline['systems'])]
                                             for task in ('exc', 'op')}
                    panel['system_wins'] = {task: sum(v < 1 for v in ratios) for task, ratios in panel['system_ratios'].items()}
                    panel['chemistry_identity_wins'] = {role: sum(abs(a['error']) < abs(b['error'])
                         for a, b in zip(panel['chemistry_rows'], baseline['chemistry_rows'])
                         if (a['database'] == 'AE17') == (role == 'AE17')) for role in ('nonAE', 'AE17')}
                    panel['monitor_artifact'] = {'path': str(target), 'sha256': offline.reuse.sha(target)}
                    result['checkpoints'][str(result['cursor'])] = panel
                    if result['cursor'] == 5:
                        rng, snapshot, saved_state = capture_rng_state(), digest(model), copy.deepcopy(state)
                        resumed_cursor, resumed_state = resume(latest, model, protocol)
                        assert resumed_cursor == 5 and digest(model) == snapshot
                        assert equal(state, resumed_state) and equal(capture_rng_state(), rng)
                        state = saved_state
                        result['resume_exact'] = True
                    offline.reuse.dump(folder/'progress.json', result)
            if result['cursor'] == 10:
                result['status'] = 'TRAJECTORY-PASS'
                result['scientific_pass'] = result['checkpoints']['10']['relative']['R_max'] < 1
                before, rng, held = digest(model), capture_rng_state(), copy.deepcopy(state)
                try:
                    loss, raw = compute_isolated_task_gradients(model, objectives(model, shadow, manifest['entries'][10], context), task_order=ORDER)
                    raw = {task: {name: torch.zeros_like(p, dtype=torch.float64) if raw[task][name] is None else raw[task][name].double()
                                  for name, p in named_trainable_parameters(model).items()} for task in ORDER}
                    _, meta, _, _ = aggregate(raw, arm['method'], copy.deepcopy(state), ORDER,
                                              protocol['immutable_files'][str(offline.reuse.SOLVER)])
                    result['stress_probe'] = {'losses': loss, 'diagnostics': meta, 'status': 'PASS'}
                except (RuntimeError, FloatingPointError) as error:
                    result['stress_probe'] = {'status': 'FAILED', 'error': str(error)}
                restore_rng_state(rng)
                assert digest(model) == before and equal(state, held) and equal(capture_rng_state(), rng)
                result['stress_probe']['read_only'] = True
        except (RuntimeError, FloatingPointError) as error:
            result['status'] = 'FAILED_SOLVER_OR_NUMERICAL'
            result['error'] = str(error)
            if phase == 'update':
                assert digest(model) == before, 'Failed attempt must restore exact committed model'
                failure = {'cursor': result['cursor'], 'phase': phase, 'before_sha256': before, 'after_sha256': digest(model),
                           'accepted': False, 'error': str(error), 'committed_state_advanced': False}
                if data.get('raw') is not None:
                    path = folder/f'failed_{result["cursor"]}_geometry.npz'
                    np.savez_compressed(path, **data)
                    norms = np.linalg.norm(data['raw'], axis=1)
                    failure.update(geometry_path=str(path), geometry_sha256=offline.reuse.sha(path),
                                   raw_norms=norms.tolist(), raw_gram=(data['raw']@data['raw'].T).tolist())
                result['failure_receipt'] = failure
        result.update(elapsed_seconds=time.perf_counter()-started, final_model_sha256=digest(model),
                      checkpoint_path=str(latest), checkpoint_sha256=offline.reuse.sha(latest))
        result['checkpoint_artifacts'] = {str(i): {'path': str(folder/f'checkpoint_{i}.pt'),
                  'sha256': offline.reuse.sha(folder/f'checkpoint_{i}.pt')} for i in (0, 5, 10)
                  if (folder/f'checkpoint_{i}.pt').exists()}
        offline.reuse.dump(folder/'result.json', result)
        print('FINISHED', arm['key'], result['status'], result['cursor'], flush=True)
        del model, shadow
        torch.cuda.empty_cache()
        verify(protocol)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
