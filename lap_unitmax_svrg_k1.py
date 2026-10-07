"""Static SVRG K1 chemistry adapter around the existing gold trajectory loop."""
import argparse
import copy
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import lap_unitmax_full251_gold as gold

h = gold.h
ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_unitmax_svrg_k1_runs_20261007'
GOLD_OUT = gold.OUT
OLD_ARENA_OUT = h.OUT
load, dump, sha = gold.load, gold.dump, gold.sha
MODELS = weakref.WeakKeyDictionary()
PROTOCOL = None


def combine(full_reference, current, sampled_reference):
    return full_reference + (current - sampled_reference)


def flat(gradients):
    return torch.cat([v.detach().double().cpu().flatten() for v in gradients.values()]).numpy()


def mapped(vector, model):
    gradients, offset = {}, 0
    for name, parameter in h.named_trainable_parameters(model).items():
        count = parameter.numel()
        gradients[name] = torch.from_numpy(vector[offset:offset+count].copy()).reshape(parameter.shape).to(parameter.device)
        offset += count
    assert offset == len(vector)
    return gradients


def batch_identity(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def counted(batch, state, kind):
    original = torch.autograd.grad
    def backward(*args, **kwargs):
        state['cost'][kind] += 1
        return original(*args, **kwargs)
    try:
        with patch.object(torch.autograd, 'grad', backward):
            return batch.value_and_grad()
    finally:
        dump(state['folder']/'chemistry_cost.json', state['cost'])


def reference_state(model):
    if model in MODELS:
        return MODELS[model]
    digest = h.digest(model)
    candidates = []
    for binding in PROTOCOL['bindings']:
        key = binding['key']
        if digest == binding['state']['state_sha256']:
            candidates.append(key)
        progress = OUT/key/'progress.json'
        if progress.exists():
            saved = load(progress)
            if saved['updates'] and digest == saved['updates'][-1]['after_sha256']:
                candidates.append(key)
    assert len(set(candidates)) == 1, 'Exact reference/start identity is required'
    key = candidates[0]
    binding = next(b for b in PROTOCOL['bindings'] if b['key'] == key)
    reference = copy.deepcopy(model).cpu()
    reference.load_state_dict(torch.load(binding['state']['path'], map_location='cpu', weights_only=False)['model'], strict=True)
    assert h.digest(reference) == binding['state']['state_sha256'] == PROTOCOL['gold_initial'][key]['state_sha256']
    shadow = copy.deepcopy(reference).to(device=next(model.parameters()).device, dtype=torch.float64)
    source = PROTOCOL['gold_initial'][key]
    assert sha(source['array_path']) == source['array_sha256']
    full = np.load(source['array_path'])['g_full251'].copy()
    assert full.shape == (9446,) and full.dtype == np.float64 and np.isfinite(full).all()
    folder = OUT/key
    cost_path = folder/'chemistry_cost.json'
    cost = load(cost_path) if cost_path.exists() else {
        'current_sample_backwards': 0, 'reference_sample_backwards': 0,
        'new_full251_reference_backwards': 0, 'reused_full251_reference_backwards': 251,
    }
    scalars = {binding['state']['state_sha256']: source['L_full']}
    if (folder/'progress.json').exists():
        saved = load(folder/'progress.json')
        for row in saved['updates']:
            if row['accepted']:
                trials = row['diagnostics'].get('vector_armijo', {}).get('trials', [])
                if trials:
                    scalars[row['after_sha256']] = trials[-1]['losses']['relchem']
    state = {'key': key, 'reference': reference, 'shadow': shadow, 'full': full,
             'reference_sha256': binding['state']['state_sha256'], 'scalars': scalars,
             'folder': folder, 'cost': cost}
    MODELS[model] = state
    return state


class StaticSVRG(h.core.ChemistryBatchObjective):
    """Gradient-only control variate; no scalar acceptance evaluation."""
    def __init__(self, sampled, rows, context):
        self.model, self.shadow = sampled.model, sampled.shadow
        self.sampled, self.rows, self.context = sampled, rows, context
        self.reference = reference_state(self.model)

    def value_and_grad(self):
        state = self.reference
        if h.digest(self.model) == state['reference_sha256']:
            # At theta_ref the paired correction is identically zero.
            vector = state['full'].copy()
            value = state['scalars'][state['reference_sha256']]
        else:
            identity = batch_identity(self.rows)
            cache = state['folder']/f'reference_sample_{identity}.npz'
            receipt = cache.with_suffix('.json')
            if cache.exists():
                bound = load(receipt)
                assert bound['reference_sha256'] == state['reference_sha256']
                assert bound['rows'] == self.rows and bound['sha256'] == sha(cache)
                ref_gradient = np.load(cache)['gradient']
            else:
                ref_batch = h.core.ChemistryBatchObjective(state['reference'], state['shadow'],
                    self.sampled.reactions, self.sampled.weights, self.sampled.dispersions)
                _, ref = counted(ref_batch, state, 'reference_sample_backwards')
                ref_gradient = flat(ref)
                np.savez_compressed(cache, gradient=ref_gradient)
                dump(receipt, {'reference_sha256': state['reference_sha256'], 'rows': self.rows, 'sha256': sha(cache)})
            value, current = counted(self.sampled, state, 'current_sample_backwards')
            vector = combine(state['full'], flat(current), ref_gradient)
        # This sampled scalar is unused by the fixed-step controller.
        return float(value), mapped(vector, self.model)

    def __call__(self):
        raise RuntimeError('No per-update chemistry scalar evaluation in fixed-step experiment')


def fixed_update(model, optimizer, factories, *, aggregator, aggregator_state,
                 task_order, vector_armijo, **unused):
    """Existing isolated gradients and solver, then one normalized fixed step."""
    assert optimizer is None and task_order == h.ORDER
    snapshot = {n: v.detach().clone() for n, v in model.state_dict().items()}
    rng = h.capture_rng_state()
    try:
        losses, sparse = h.core.compute_isolated_task_gradients(model, factories, task_order=task_order)
        raw = h.core.materialize_task_zeros(model, sparse, task_order=task_order)
        direction, meta, state = aggregator(raw, state=copy.deepcopy(aggregator_state))
        assert meta['feasible'] and all(torch.isfinite(v).all() for v in direction.values())
        eta = vector_armijo['initial_step_size']
        with torch.no_grad():
            for name, parameter in h.named_trainable_parameters(model).items():
                parameter.copy_(snapshot[name] + (-eta * direction[name]))
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite fixed-step model')
        diagnostics = {**meta, 'fixed_step': {'eta': eta, 'backtracks': 0,
                       'chemistry_scalar_acceptance': False},
                       'vector_armijo': {'accepted_t': 1.0, 'trials': []}}
        return SimpleNamespace(accepted=True, losses=losses, stop_reason=None,
                               diagnostics=diagnostics, aggregator_state=state)
    except Exception:
        model.load_state_dict(snapshot, strict=True)
        h.restore_rng_state(rng)
        raise


def objectives(model, shadow, entry, context):
    factories = h.objectives(model, shadow, entry, context)
    rows = entry['per_rank'][0]['task_samples']['relchem']
    counts = {}
    for e in context['entries']:
        db = e['per_rank'][0]['reaction']['database']
        counts[db] = counts.get(db, 0)+1
    assert len(rows) == len(counts) == 8 and {r['database'] for r in rows} == set(counts)
    assert all(r['weight'] == counts[r['database']]/251 for r in rows)
    factories['relchem'] = StaticSVRG(factories['relchem'], rows, context)
    return factories


def verify(protocol):
    for path, expected in protocol['immutable_files'].items():
        assert sha(path) == expected, path
    assert sha(__file__) == protocol['script_sha256']
    return len(protocol['immutable_files'])+1


def freeze():
    assert not (OUT/'protocol.json').exists()
    original = load(GOLD_OUT/'protocol.json')
    gold.verify(original)
    immutable = dict(original['immutable_files'])
    immutable[str(Path(gold.__file__))] = sha(gold.__file__)
    immutable[str(GOLD_OUT/'protocol.json')] = sha(GOLD_OUT/'protocol.json')
    baselines = {}
    for binding in original['bindings']:
        key = binding['key']
        path = GOLD_OUT/key/'monitor_0.json'
        if not path.exists():
            path = OLD_ARENA_OUT/(key+'_baseline.json')
        panel = load(path)
        assert panel['model_sha256'] == binding['state']['state_sha256']
        immutable[str(path)] = sha(path)
        baselines[key] = {'path': str(path), 'sha256': sha(path)}
    protocol = {**original, 'starting_commit': '94f6cbba2024df96da1fa985807976e9cbd88b19',
        'script_sha256': sha(__file__), 'immutable_files': immutable, 'baselines': baselines,
        'scientific_change': 'Static SVRG K1 chemistry gradient; fixed initial full251 reference. Chemistry Armijo value remains true full251, no surrogate.',
        'reference': 'Exactly t0; reuse SHA-bound qualified full251 gradients. Paired sampled terms use identical manifest IDs, variants and n_d/251 weights.',
        'scalar_cache': 'Reuse exact true-full251 values only for identical model digest; no surrogate and no future state reference.',
        'scope': 'Four fresh SVRG UNIT_MAXMIN trajectories, at most ten updates; no gold continuation, refresh, new sampler, optimizer or extension.'}
    OUT.mkdir(exist_ok=True)
    dump(OUT/'protocol.json', protocol)
    (ROOT/'lap_unitmax_svrg_k1_10update_protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
    print('FROZEN', verify(protocol), flush=True)


def run():
    global PROTOCOL
    PROTOCOL = load(OUT/'protocol.json')
    verify(PROTOCOL)
    original_monitor, original_model = h.monitor, h._pilot_model
    def monitor(model, shadow, context, target):
        if target.name != 'monitor_0.json':
            return original_monitor(model, shadow, context, target)
        source = PROTOCOL['baselines'][target.parent.name]
        assert sha(source['path']) == source['sha256']
        panel = copy.deepcopy(load(source['path']))
        assert panel['model_sha256'] == h.digest(model)
        panel['protocol_sha256'] = sha(OUT/'protocol.json')
        panel['reused_baseline'] = source
        dump(target, panel)
        print('REUSED_T0', target.parent.name, flush=True)
        return panel
    def new_model(*args, **kwargs):
        for arm in PROTOCOL['arms']:
            result = OUT/arm['key']/'result.json'
            if result.exists():
                saved = load(result)
                if saved['status'] != 'TRAJECTORY-PASS' or not saved.get('scientific_pass', False):
                    raise RuntimeError('STOP_ON_FAILED_ARM: '+arm['key'])
        return original_model(*args, **kwargs)
    with patch.object(gold, 'OUT', OUT), patch.object(gold, 'verify', verify), \
         patch.object(gold, 'objectives', objectives), patch.object(h, 'monitor', monitor), \
         patch.object(h, '_pilot_model', new_model), patch.object(h, 'train_moo_update', fixed_update):
        gold.run()


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
