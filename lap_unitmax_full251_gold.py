"""Four bounded gold-control trajectories; existing solver and Armijo unchanged."""
import argparse
import copy
import random
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import lap_fullchem_surrogate_fidelity_audit as chem

h = chem.arena
ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_unitmax_full251_gold_runs_20261007'
START = '789d04d0638aa9ded44f0e9012cb5b4fc584a235'
sha, load, dump = h.offline.reuse.sha, h.offline.reuse.load, h.offline.reuse.dump


class Full251Chemistry(h.core.ChemistryBatchObjective):
    """Adapter for the qualified streamed fullchem evaluator, not a new loss."""
    def __init__(self, model, shadow, context):
        self.model, self.shadow, self.context = model, shadow, context
        assert len(context['entries']) == 251
        assert tuple(h.named_trainable_parameters(model)) == tuple(h.named_trainable_parameters(shadow))
        assert all(p.dtype == torch.float64 for p in shadow.parameters())

    def evaluate(self, gradient):
        count = 0
        original = self.context['store'].load_variant
        def progress(*args, **kwargs):
            nonlocal count
            result = original(*args, **kwargs)
            count += 1
            if count % 16 == 0 or count == 251:
                print('FULL251', 'gradient' if gradient else 'scalar', count, '/251', flush=True)
            return result
        with patch.object(self.context['store'], 'load_variant', progress):
            return chem.fullchem(self.model, self.shadow, self.context, gradient)

    def value_and_grad(self):
        value, vector = self.evaluate(True)
        parameters = h.named_trainable_parameters(self.model)
        gradients, offset = {}, 0
        for name, p in parameters.items():
            count = p.numel()
            gradients[name] = torch.from_numpy(vector[offset:offset+count].copy()).reshape(p.shape).to(p.device)
            offset += count
        assert offset == len(vector) and all(g.dtype == torch.float64 for g in gradients.values())
        return value, gradients

    def __call__(self):
        value, _ = self.evaluate(False)
        return next(self.shadow.parameters()).new_tensor(value)


def objectives(model, shadow, entry, context):
    factories = h.objectives(model, shadow, entry, context)
    factories['relchem'] = Full251Chemistry(model, shadow, context)
    assert tuple(factories) == h.ORDER
    return factories


def verify(p):
    for path, expected in p['immutable_files'].items():
        assert sha(path) == expected, path
    assert sha(__file__) == p['script_sha256']
    return len(p['immutable_files'])+1


def freeze():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == START
    assert not (OUT/'protocol.json').exists()
    base = load(h.OUT/'protocol.json')
    h.verify(base)
    OUT.mkdir(exist_ok=True)
    immutable = dict(base['immutable_files'])
    for path in (h.OUT/'protocol.json', ROOT/'lap_symmetric_moo_trajectory_arena.py', ROOT/'lap_fullchem_surrogate_fidelity_audit.py'):
        immutable[str(path)] = sha(path)
    gold = load(ROOT/'lap_fullchem_surrogate_fidelity_audit_metrics.json')
    gold_initial = {r['start']: r for r in gold['states'] if r['update'] == 0}
    for r in gold_initial.values():
        immutable[r['array_path']] = r['array_sha256']
    p = {**base, 'starting_commit': START, 'script_sha256': sha(__file__), 'immutable_files': immutable,
         'arms': [{'key': b['key'], 'start': b['key'], 'method': 'UNIT_MAXMIN'} for b in base['bindings']],
         'scientific_change': 'Only chemistry gradient and Armijo scalar become exact arithmetic-mean full251 corrected batch_fchem; matched-F64 and canonical variants unchanged.',
         'gold_initial': gold_initial, 'initial_gradient_replay_relative_L2_max': 1e-11,
         'gate': 'All4 reach10; all4 Rmax10<1; no solver failure or pathological step collapse. No epsilon.',
         'collapse_reporting': 'Reuse prior Arena labels: severe if backtracks>=10 or accepted_t<1e-3; systematic if median accepted_t<=1/16 or more than half accepted steps have>=4 backtracks. Qualification requires neither. These labels do not alter acceptance.',
         'scope': 'Exactly4 starts, maximum10 accepted opportunities each; no stress probe, SVRG, new optimizer, new initialization or extension.'}
    dump(OUT/'protocol.json', p)
    (ROOT/'lap_unitmax_full251_10update_gold_protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
    print('FROZEN', verify(p), 'hashes', p['references'], flush=True)


def run():
    p = load(OUT/'protocol.json')
    verify(p)
    manifest = h.read_sampling_manifest(h.MANIFEST)
    context = {'full': h.read_sampling_manifest(h.FULL),
               'store': h.MinnesotaGroupStore(p['inputs']['store'], cache_groups=1),
               'disp': h.load_reaction_dispersions(p['inputs']['disp']),
               'mrksdisp': h.load_mrks_dispersions(p['inputs']['mrksdisp']),
               'systems': h.CentralAOCache(Path(p['inputs']['central']).parent, Path(p['inputs']['ao']).parent,
                                         device=torch.device('cuda'), dtype=torch.float32, chunk_size=4096)}
    context['entries'] = [e for e in context['full']['entries'] if e['per_rank'][0]['reaction']['database'] != 'AE17']
    assert len(context['entries']) == 251
    # Existing checkpoint/monitor helpers bind their OUT constant to this experiment.
    with patch.object(h, 'OUT', OUT), patch.object(h, 'verify', verify):
        for arm in p['arms']:
            folder = OUT/arm['key']
            folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():
                assert load(folder/'result.json')['protocol_sha256'] == sha(OUT/'protocol.json')
                continue
            b = next(b for b in p['bindings'] if b['key'] == arm['start'])
            random.seed(41)
            np.random.seed(41)
            torch.manual_seed(41)
            torch.cuda.manual_seed_all(41)
            model = h._pilot_model(torch.device('cuda'), torch.float32)
            model.load_state_dict(torch.load(b['state']['path'], map_location='cpu', weights_only=False)['model'], strict=True)
            model.train()
            assert h.digest(model) == b['state']['state_sha256']
            assert list(h.named_trainable_parameters(model)) == [r['name'] for r in p['parameter_order']]
            shadow = copy.deepcopy(model).double()
            latest = folder/'latest.pt'
            if latest.exists():
                result = load(folder/'progress.json')
                cursor, state = h.resume(latest, model, p)
                assert cursor == result['cursor']
            else:
                baseline = h.monitor(model, shadow, context, folder/'monitor_0.json')
                result = {'arm': arm, 'protocol_sha256': sha(OUT/'protocol.json'), 'cursor': 0,
                          'updates': [], 'checkpoints': {'0': baseline}, 'status': 'RUNNING'}
                state = None
                h.checkpoint(latest, model, state, 0, p)
                shutil.copyfile(latest, folder/'checkpoint_0.pt')
                dump(folder/'progress.json', result)
            while result['cursor'] < 10:
                cursor = result['cursor']
                before, rng, held = h.digest(model), h.capture_rng_state(), copy.deepcopy(state)
                arrays = {}
                phase = 'update'
                def hook(raw, arrays=arrays, cursor=cursor, arm=arm, result=result, **kwargs):
                    direction, meta, next_state, geometry = h.aggregate(raw, 'UNIT_MAXMIN', kwargs['state'], h.ORDER,
                                                                    p['immutable_files'][str(h.offline.reuse.SOLVER)])
                    arrays.update(geometry)
                    if cursor == 0:
                        gold = np.load(p['gold_initial'][arm['start']]['array_path'])['g_full251']
                        error = float(np.linalg.norm(geometry['raw'][0]-gold)/np.linalg.norm(gold))
                        assert error <= p['initial_gradient_replay_relative_L2_max']
                        result['initial_gradient_replay'] = {'relative_L2': error, 'bitwise_equal': bool(np.array_equal(geometry['raw'][0], gold))}
                    return direction, meta, next_state
                restorations = []
                restore = h.core._restore_model_state
                def exact_restore(target, snapshot, restore=restore, restorations=restorations):
                    restore(target, snapshot)
                    assert all(torch.equal(target.state_dict()[n], v) for n, v in snapshot.items())
                    restorations.append({'exact': True, 'max_abs_difference': 0})
                print('UPDATE_BEGIN', arm['key'], cursor, flush=True)
                try:
                    factories = objectives(model, shadow, manifest['entries'][cursor], context)
                    with patch.object(h.core, '_restore_model_state', exact_restore):
                        update = h.train_moo_update(model, None, factories, method='pcd', aggregator_state=state,
                              aggregator=hook, world_size=1, task_order=h.ORDER, step_rule='pcd_direct_vector_armijo',
                              vector_armijo={'initial_step_size': p['references'][arm['start']]['eta0'], **p['armijo']})
                    del factories
                    record = {'cursor': cursor, 'sample': manifest['entries'][cursor], 'before_sha256': before,
                              'after_sha256': h.digest(model), 'accepted': update.accepted, 'losses': update.losses,
                              'stop_reason': update.stop_reason, 'diagnostics': update.diagnostics,
                              'restoration_checks': restorations}
                    if arrays:
                        path = folder/f'update_{cursor}_geometry.npz'
                        np.savez_compressed(path, **arrays)
                        record.update(geometry_path=str(path), geometry_sha256=sha(path))
                    result['updates'].append(record)
                    if not update.accepted:
                        assert h.digest(model) == before and h.equal(state, held)
                        assert h.equal(h.capture_rng_state(), rng)
                        result['status'] = 'FAILED_UPDATE'
                        break
                    state = update.aggregator_state
                    result['cursor'] += 1
                    phase = 'checkpoint'
                    h.checkpoint(latest, model, state, result['cursor'], p)
                    dump(folder/'progress.json', result)
                    print('ACCEPTED', arm['key'], result['cursor'], update.diagnostics['vector_armijo']['accepted_t'], flush=True)
                    if result['cursor'] in (5, 10):
                        phase = 'monitor'
                        shutil.copyfile(latest, folder/f'checkpoint_{result["cursor"]}.pt')
                        panel = h.monitor(model, shadow, context, folder/f'monitor_{result["cursor"]}.json')
                        panel['relative'] = h.endpoint_ratios(panel['objectives'], result['checkpoints']['0']['objectives'])
                        result['checkpoints'][str(result['cursor'])] = panel
                        snapshot, saved_rng = h.digest(model), h.capture_rng_state()
                        resumed_cursor, resumed_state = h.resume(latest, model, p)
                        assert resumed_cursor == result['cursor'] and h.digest(model) == snapshot
                        assert h.equal(state, resumed_state) and h.equal(saved_rng, h.capture_rng_state())
                        result['resume_exact'] = True
                        dump(folder/'progress.json', result)
                except (RuntimeError, FloatingPointError) as error:
                    if phase == 'update':
                        assert h.digest(model) == before and h.equal(state, held)
                    result['status'] = 'FAILED_SOLVER_OR_NUMERICAL'
                    result['failure'] = {'cursor': cursor, 'phase': phase, 'error': str(error), 'before_sha256': before,
                                         'after_sha256': h.digest(model), 'committed_state_advanced': phase != 'update'}
                    break
            if result['cursor'] == 10:
                result['status'] = 'TRAJECTORY-PASS'
                result['scientific_pass'] = result['checkpoints']['10']['relative']['R_max'] < 1
            result.update(final_model_sha256=h.digest(model), checkpoint_path=str(latest), checkpoint_sha256=sha(latest))
            result['checkpoint_artifacts'] = {str(i): {'path': str(folder/f'checkpoint_{i}.pt'), 'sha256': sha(folder/f'checkpoint_{i}.pt')}
                                              for i in (0, 5, 10) if (folder/f'checkpoint_{i}.pt').exists()}
            dump(folder/'result.json', result)
            print('FINISHED', arm['key'], result['status'], result.get('scientific_pass'), flush=True)
            del model, shadow
            torch.cuda.empty_cache()
            verify(p)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
