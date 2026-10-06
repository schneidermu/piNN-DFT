"""Read-only saved-direction/full-chemistry audit; no training-controller call."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

import lap_symmetric_moo_trajectory_arena as arena

ROOT = Path(__file__).parent
OUT = ROOT.parent/'lap_fullchem_surrogate_fidelity_runs_20261006'
POSITIONS = (0, 5, 9)
ChemistryBatchObjective = arena.core.ChemistryBatchObjective


def classify(rows):
    mismatch = [r for r in rows if r['p_sample'] > 0 and r['p_full'] <= 0]
    nonlinear = [r for r in rows if r['p_full'] > 0 and r['actual_trials'][0]['delta'] >= 0]
    if mismatch and nonlinear:
        return 'CASE C'
    if mismatch:
        return 'CASE A'
    if nonlinear:
        return 'CASE B'
    return 'CASE D'


def directional(raw, full, direction):
    sample = raw[0]
    ns, nf = np.linalg.norm(sample), np.linalg.norm(full)
    assert ns > 0 and nf > 0 and np.isfinite(full).all()
    us, uf = sample/ns, full/nf
    ps, pf = float(us@direction), float(uf@direction)
    return {'sample_norm': float(ns), 'full_norm': float(nf), 'norm_ratio': float(ns/nf),
            'cos_sample_full': float(us@uf), 'p_sample': ps, 'p_full': pf,
            'p_full_minus_sample': pf-ps, 'sign_agreement': bool((ps > 0) == (pf > 0)),
            'unit_gradient_difference_norm': float(np.linalg.norm(us-uf)),
            'e_direction': float((us-uf)@direction)}


def temporary_trial(model, direction, eta, evaluate):
    base = {n: v.detach().clone() for n, v in model.state_dict().items()}
    before, rng = arena.digest(model), arena.capture_rng_state()
    requested, realized = [], []
    try:
        with torch.no_grad():
            offset = 0
            for name, parameter in arena.named_trainable_parameters(model).items():
                n = parameter.numel()
                delta = -eta*torch.from_numpy(direction[offset:offset+n].copy()).to(parameter.device).reshape(parameter.shape)
                parameter.copy_(base[name]+delta)
                requested.append(delta.detach().double().cpu().flatten())
                realized.append((parameter.double()-base[name].double()).detach().cpu().flatten())
                offset += n
            assert offset == len(direction)
        value = evaluate()
        return value, torch.cat(requested).numpy(), torch.cat(realized).numpy(), arena.digest(model)
    finally:
        model.load_state_dict(base, strict=True)
        arena.restore_rng_state(rng)
        assert arena.digest(model) == before
        assert all(torch.equal(model.state_dict()[n], v) for n, v in base.items())
        assert arena.equal(arena.capture_rng_state(), rng)


def verify(protocol):
    for path, value in protocol['immutable_files'].items():
        assert arena.offline.reuse.sha(path) == value, path
    assert arena.offline.reuse.sha(__file__) == protocol['script_sha256']
    return len(protocol['immutable_files'])+1


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = arena.offline.reuse.load(arena.OUT/'protocol.json')
    arena.verify(old)
    immutable = dict(old['immutable_files'])
    for path in (arena.OUT/'protocol.json', arena.FULL, ROOT/'lap_symmetric_moo_trajectory_arena.py'):
        immutable[str(path)] = arena.offline.reuse.sha(path)
    bindings = []
    for arm in old['arms']:
        if arm['method'] != 'UNIT_MAXMIN':
            continue
        folder = arena.OUT/arm['key']
        result = arena.offline.reuse.load(folder/'result.json')
        assert result['status'] == 'TRAJECTORY-PASS' and result['cursor'] == 10
        for path in (folder/'result.json', folder/'checkpoint_0.pt', folder/'checkpoint_5.pt', folder/'checkpoint_10.pt'):
            immutable[str(path)] = arena.offline.reuse.sha(path)
        model = arena._pilot_model(torch.device('cpu'), torch.float32)
        payload = torch.load(folder/'checkpoint_0.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        for cursor, record in enumerate(result['updates']):
            path = Path(record['geometry_path'])
            assert arena.offline.reuse.sha(path) == record['geometry_sha256']
            immutable[str(path)] = record['geometry_sha256']
            arrays = np.load(path)
            raw, direction = arrays['raw'], arrays['unit']
            assert raw.shape == (4, 9446) and direction.shape == (9446,)
            assert arena.digest(model) == record['before_sha256']
            expected = record['diagnostics']['normalized_progress']['p'][0]
            actual = float(raw[0]@direction/np.linalg.norm(raw[0]))
            assert abs(actual-expected) <= 1e-13
            eta = record['diagnostics']['vector_armijo']['initial_step_size']*record['diagnostics']['vector_armijo']['accepted_t']
            if cursor in POSITIONS:
                state_path = OUT/f'{arm["start"]}_before_{cursor}.pt'
                torch.save({'model': {n: v.detach().clone() for n, v in model.state_dict().items()},
                            'model_sha256': arena.digest(model), 'recovered_cursor': cursor,
                            'original_result_sha256': immutable[str(folder/'result.json')]}, state_path)
                immutable[str(state_path)] = arena.offline.reuse.sha(state_path)
                bindings.append({'key': f'{arm["start"]}_u{cursor}', 'start': arm['start'], 'update': cursor,
                                 'state_path': str(state_path), 'state_sha256': arena.digest(model),
                                 'state_file_sha256': immutable[str(state_path)], 'geometry_path': str(path),
                                 'geometry_sha256': record['geometry_sha256'], 'p_sample_reference': expected,
                                 'eta_acc': eta, 'gamma_original': record['diagnostics']['gamma_star'],
                                 'original_after_sha256': record['after_sha256'],
                                 'original_loss_sample': record['losses']['relchem']})
            # Exact receipt replay, not a new trajectory: no functional evaluation.
            _, _, _, after = temporary_trial(model, direction, eta, lambda: 0.)
            assert after == record['after_sha256']
            with torch.no_grad():
                offset = 0
                for parameter in arena.named_trainable_parameters(model).values():
                    n = parameter.numel()
                    delta = -eta*torch.from_numpy(direction[offset:offset+n].copy()).reshape(parameter.shape)
                    parameter.copy_(parameter.detach().clone()+delta)
                    offset += n
            assert arena.digest(model) == record['after_sha256']
        assert arena.digest(model) == result['final_model_sha256']
        del model
    protocol = {'starting_commit': 'd40bbcb5e7797728cdaedf5bcc6ef4950a4c85f4',
                'bindings': bindings, 'parameter_order': old['parameter_order'],
                'inputs': old['inputs'], 'immutable_files': immutable,
                'script_sha256': arena.offline.reuse.sha(__file__),
                'manifest_file_sha256': old['manifest_file_sha256'],
                'manifest_canonical_sha256': old['manifest_canonical_sha256'],
                'positions': list(POSITIONS), 'task_order': list(arena.ORDER),
                'full_scalar': 'Arithmetic mean of 251 corrected singleton batch_fchem losses; exact non-AE canonical variants from full268 manifest, F64 chemistry shadow on widened F32 stored state/source values.',
                'precision': 'Stored F32; chemistry real F64 leaves/forward/autograd; original source values widened; no output-after-F32 cast.',
                'full_scalar_reduction': 'sum singleton losses /251, exactly the Arena monitoring denominator; gradient sum scaled by1/251.',
                'trials_actual_factors': [1, .5, .25], 'trials_counterfactual': '1; additionally .5 iff full displacement does not descend',
                'repeatability': 'Repeat baseline and actual full-eta scalar at every state; retain absolute repeat differences, no hidden epsilon; signal label uses >20*observed max repeat difference, zero floor reported literally.',
                'direction_progress_tolerance': 1e-13,
                'classification': 'A if sampled-descent/full-nondescent exists, B if positive full slope with finite-eta non-descent exists, C if both, D if neither; raw counts, half/quarter outcomes and cross-start replication govern strength, not added promotion thresholds.',
                'gradient_norm_interpretation': 'Unit-gradient angular fidelity and directional error are causal diagnostics; norm ratio reported separately, never used alone to assign causality.',
                'scope': '12 states only, no training/retained trial/no new MOO/no Nash/no per-database gradients'}
    assert len(bindings) == 12
    arena.offline.reuse.dump(OUT/'protocol.json', protocol)
    arena.offline.reuse.dump(ROOT/'lap_fullchem_surrogate_fidelity_audit_protocol.json', protocol)
    print('FROZEN', len(bindings), 'states', verify(protocol), 'hash checks', flush=True)


def fullchem(model, shadow, context, gradient=False):
    before, rng = arena.digest(model), arena.capture_rng_state()
    shadow.load_state_dict(model.state_dict(), strict=True)
    shadow.train(model.training)
    parameters = arena.named_trainable_parameters(shadow)
    accum = {n: torch.zeros_like(p) for n, p in parameters.items()} if gradient else None
    total, count = 0., 0
    try:
        for entry in context['entries']:
            sample = entry['per_rank'][0]
            ident = sample['reaction']
            reaction = context['store'].load_variant((ident['database'], ident['reaction_id']), sample['variant_suffix'])
            with torch.set_grad_enabled(gradient):
                loss = arena.make_reaction_objective(shadow, reaction, device='cuda', dtype=torch.float64,
                                                     dispersions=context['disp'])()
            assert loss.dtype == torch.float64 and torch.isfinite(loss).all()
            total += float(loss.detach())
            if gradient:
                gs = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
                for (name, _), g in zip(parameters.items(), gs):
                    if g is not None:
                        accum[name].add_(g.detach(), alpha=1/251)
                del gs
            count += 1
            del reaction, loss
        assert count == 251
        vector = torch.cat([v.detach().double().cpu().flatten() for v in accum.values()]).numpy() if gradient else None
        assert np.isfinite(total) and (vector is None or np.isfinite(vector).all())
        return total/251, vector
    finally:
        assert arena.digest(model) == before and all(p.grad is None for p in model.parameters())
        arena.restore_rng_state(rng)
        assert arena.equal(arena.capture_rng_state(), rng)


def run():
    protocol = arena.offline.reuse.load(OUT/'protocol.json')
    verify(protocol)
    entries = [e for e in arena.read_sampling_manifest(arena.FULL)['entries'] if e['per_rank'][0]['reaction']['database'] != 'AE17']
    assert len(entries) == 251
    context = {'entries': entries, 'store': arena.MinnesotaGroupStore(protocol['inputs']['store'], cache_groups=1),
               'disp': arena.load_reaction_dispersions(protocol['inputs']['disp'])}
    for binding in protocol['bindings']:
        target = OUT/(binding['key']+'_result.json')
        if target.exists():
            cached = arena.offline.reuse.load(target)
            assert cached['protocol_sha256'] == arena.offline.reuse.sha(OUT/'protocol.json')
            assert cached['state_sha256'] == binding['state_sha256']
            continue
        started = time.perf_counter()
        model = arena._pilot_model(torch.device('cuda'), torch.float32)
        model.load_state_dict(torch.load(binding['state_path'], map_location='cpu', weights_only=False)['model'], strict=True)
        assert arena.digest(model) == binding['state_sha256']
        assert tuple(arena.named_trainable_parameters(model)) == tuple(r['name'] for r in protocol['parameter_order'])
        shadow = copy.deepcopy(model).double()
        arrays = np.load(binding['geometry_path'])
        raw, actual = arrays['raw'].copy(), arrays['unit'].copy()
        assert abs(float(raw[0]@actual/np.linalg.norm(raw[0]))-binding['p_sample_reference']) <= 1e-13
        print('BEGIN', binding['key'], 'full251 gradient', flush=True)
        baseline, full = fullchem(model, shadow, context, True)
        repeated, _ = fullchem(model, shadow, context)
        row = {**binding, **directional(raw, full, actual), 'L_full': baseline,
               'baseline_repeat_absolute_difference': abs(baseline-repeated), 'actual_trials': [],
               'protocol_sha256': arena.offline.reuse.sha(OUT/'protocol.json')}
        counter = raw.copy()
        counter[0] = full
        cf, meta = arena.offline.maxmin(counter, protocol['immutable_files'][str(arena.offline.reuse.SOLVER)])
        original_replay, original_meta = arena.offline.maxmin(raw, protocol['immutable_files'][str(arena.offline.reuse.SOLVER)])
        np.testing.assert_allclose(original_replay, actual, rtol=1e-12, atol=1e-13)
        original_full_p = counter@actual/np.linalg.norm(counter, axis=1)
        counter_p = counter@cf/np.linalg.norm(counter, axis=1)
        row['counterfactual'] = {'gamma': meta['gamma_star'], 'gamma_change': meta['gamma_star']-binding['gamma_original'],
             'original_gamma_replay': original_meta['gamma_star'], 'direction_cosine': float(actual@cf),
             'angle_degrees': float(np.degrees(np.arccos(np.clip(actual@cf, -1, 1)))),
             'task_progress': dict(zip(arena.ORDER, counter_p.tolist())),
             'original_direction_true_task_progress': dict(zip(arena.ORDER, original_full_p.tolist())),
             'progress_changes': dict(zip(arena.ORDER, (counter_p-original_full_p).tolist())),
             'strict_common_descent': bool(np.all(counter_p > 0)),
             'active_tasks': [arena.ORDER[i] for i, p in enumerate(counter_p) if abs(p-meta['gamma_star']) <= 1e-9],
             'solver': meta, 'trials': []}
        def evaluate(model=model, shadow=shadow, context=context):
            return fullchem(model, shadow, context)[0]
        trial_arrays = {}
        for factor in (1., .5, .25):
            eta = binding['eta_acc']*factor
            value, requested, realized, trial_hash = temporary_trial(model, actual, eta, evaluate)
            if factor == 1:
                assert trial_hash == binding['original_after_sha256']
                repeat, _, _, repeat_hash = temporary_trial(model, actual, eta, evaluate)
                assert repeat_hash == trial_hash
                row['full_eta_repeat_absolute_difference'] = abs(value-repeat)
            prediction = float(-eta*(full@actual))
            row['actual_trials'].append({'factor': factor, 'eta': eta, 'loss': value, 'delta': value-baseline,
                'normalized_delta': (value-baseline)/abs(baseline), 'prediction': prediction,
                'prediction_error': value-baseline-prediction, 'realized_prediction': float(full@realized),
                'requested_norm': float(np.linalg.norm(requested)), 'realized_norm': float(np.linalg.norm(realized)),
                'trial_state_sha256': trial_hash, 'restoration_max_abs_difference': 0})
            trial_arrays['actual_realized_'+str(factor)] = realized
            trial_arrays['actual_requested_'+str(factor)] = requested
        for factor in (1., .5):
            if factor == .5 and row['counterfactual']['trials'][0]['delta'] < 0:
                break
            eta = binding['eta_acc']*factor
            value, _, realized, trial_hash = temporary_trial(model, cf, eta, evaluate)
            row['counterfactual']['trials'].append({'factor': factor, 'eta': eta, 'loss': value, 'delta': value-baseline,
                'prediction': float(-eta*(full@cf)), 'realized_prediction': float(full@realized),
                'trial_state_sha256': trial_hash, 'restoration_max_abs_difference': 0})
        floor = max(row['baseline_repeat_absolute_difference'], row['full_eta_repeat_absolute_difference'])
        row['observed_scalar_repeatability_floor'] = floor
        row['actual_change_resolved'] = abs(row['actual_trials'][0]['delta']) > 20*floor
        row['unchanged_state'] = arena.digest(model) == binding['state_sha256']
        assert row['unchanged_state']
        path = OUT/(binding['key']+'_arrays.npz')
        np.savez_compressed(path, raw_original=raw, g_full251=full, d_actual=actual, d_fullchem=cf, **trial_arrays)
        row['array_path'], row['array_sha256'] = str(path), arena.offline.reuse.sha(path)
        row['elapsed_seconds'] = time.perf_counter()-started
        arena.offline.reuse.dump(target, row)
        print('COMPLETE', binding['key'], 'p_sample/full', row['p_sample'], row['p_full'],
              'Delta full eta', row['actual_trials'][0]['delta'], flush=True)
        del evaluate, model, shadow, full
        torch.cuda.empty_cache()
        verify(protocol)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
