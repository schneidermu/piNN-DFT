"""Read-only exact single-state q-Jacobian instrument qualification."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import lap_q_parameter_sketch_diagnostic as reuse

OUT = Path('C:/Dev/readWFN_share_ms/lap_q_exact_jacobian_runs_20261006')
ETA0 = 0.0002118715157188727
ETAS = [ETA0 / divisor for divisor in (1, 4, 16, 64, 256)]
ROOT = Path(__file__).parent


def dump(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def load(path):
    return json.loads(path.read_text())


def unit(size, seed, device):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    vector = torch.randn(size, generator=generator, dtype=torch.float64)
    return (vector / vector.norm()).to(device)


def dense_jacobian(function, zero):
    value, back = torch.func.vjp(function, zero)
    rows = []
    for index in range(value.numel()):
        target = torch.zeros_like(value)
        target[index] = 1
        rows.append(back(target)[0].detach())
    matrix = torch.stack(rows)
    assert torch.isfinite(matrix).all() and torch.isfinite(value).all()
    return value.detach(), matrix


def finite_response(function, zero, direction, eta, reference, parameter_norm, safety):
    plus = function(zero + eta * direction).detach()
    minus = function(zero - eta * direction).detach()
    repeated_plus = function(zero + eta * direction).detach()
    repeated_minus = function(zero - eta * direction).detach()
    floor = float(torch.cat([(plus-repeated_plus).abs(), (minus-repeated_minus).abs()]).max())
    numerator = plus-minus
    actual = numerator / (2 * eta)
    stats = reuse.errors(actual, reference)
    signal = float(numerator.norm())
    resolved = signal > 0 if floor == 0 else signal >= 20 * floor * numerator.numel() ** .5
    stats.update({'eta': eta, 'direction_norm': float(direction.norm()),
                  'prediction_norm': float(reference.norm()), 'FD_norm': float(actual.norm()),
                  'relative_parameter_norm': eta / parameter_norm,
                  'max_coordinate_perturbation': float((eta * direction).abs().max()),
                  'restoration_max_abs': safety(), 'repeatability_max_abs': floor,
                  'numerator_L2': signal, 'smallest_nonzero_numerator':
                  float(numerator[numerator != 0].abs().min()) if bool((numerator != 0).any()) else None,
                  'signal_resolved': resolved})
    stats['PASS'] = bool(resolved and stats['relative_L2'] is not None
                         and stats['relative_L2'] <= .05 and stats['cosine'] is not None
                         and stats['cosine'] >= .995)
    return stats


def classify(records):
    """Frozen per-direction convergence rule; no additional eta evaluation."""
    errors = [record['relative_L2'] for record in records]
    passes = [index for index, record in enumerate(records) if record['PASS']]
    if not passes:
        return {'classification': 'STRUCTURAL-FAIL', 'qualified': False,
                'reason': 'No predeclared eta meets both error/cosine and resolution gates.'}
    first = passes[0]
    tail = errors[first:]
    # A resolved small-error plateau is allowed; significant rebound is disclosed.
    rebound = any(tail[i+1] > max(2 * tail[i], 1e-8) for i in range(len(tail)-1))
    coherent = all(records[i]['PASS'] for i in range(first, len(records)))
    if rebound and first < len(records)-1:
        label = 'NUMERICAL-CANCELLATION'
    elif coherent:
        label = 'CURVATURE-LIMITED' if first else 'PASS-LINEAR'
    else:
        return {'classification': 'STRUCTURAL-FAIL', 'qualified': False,
                'reason': 'An isolated passing eta does not establish stable smaller-step behavior.'}
    return {'classification': label, 'qualified': True,
            'validated_etas': [records[index]['eta'] for index in passes],
            'error_ratios_larger_over_smaller': [errors[i]/errors[i+1]
                if errors[i+1] and errors[i] is not None else None for i in range(len(errors)-1)],
            'cancellation_attribution': 'possible rebound; inspect absolute/repeatability evidence'
                if rebound else 'no significant rebound'}


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Frozen protocol already exists.'
    reuse.validate_contract()
    old = reuse.prior.load(reuse.ARTIFACTS / 'protocol.json')
    states = [next(state for state in old['states'] if state['seed'] == 11
                   and state['regime'] == regime) for regime in ('P536', 'P67')]
    protocol = {'starting_commit': '11b1f6716e24186f371ba65fa6095c21fea5c666',
                'states': states, 'P67_gate': 'Run only after P536 qualification succeeds.',
                'probe_path': old['probe_path'], 'probe_sha256': old['probe_sha256'],
                'h2': reuse.H2, 'etas': ETAS, 'eta0': ETA0, 'point_count': 64,
                'row_order': 'alpha points0..63, then beta points0..63',
                'parameter_order': old['parameter_order'], 'numel': 9446,
                'arithmetic': 'F64 arithmetic on exact widened F32 state/source values; unchanged prior.contrast',
                'script_sha256': reuse.prior.sha(__file__),
                'reuse_script_sha256': reuse.prior.sha(ROOT/'lap_q_parameter_sketch_diagnostic.py'),
                'old_protocol_sha256': reuse.prior.sha(reuse.ARTIFACTS/'protocol.json'),
                'identity_random_seeds': [42, 314159, 271828], 'random_control_seed': 42,
                'identity_relative_gate': 1e-8, 'finite_relative_gate': .05, 'finite_cosine_gate': .995,
                'rank_relative_thresholds': [1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                'convergence': 'All smaller steps should pass or show coherent resolved plateau; rebound greater than2x and1e-8 is cancellation candidate, not automatically proven cancellation.',
                'signal_rule': 'Exact repeat floor0 requires nonzero numerator; otherwise numerator L2>=20*floor*sqrt128; no epsilon denominator.',
                'localization_gate': 'Only STRUCTURAL-FAIL; modules eta0/16,/64,/256; LN gamma/beta only if needed.',
                'coordinate_gate': 'Only failing module; top2 absolute leading-direction coordinates; same3 tiny etas.',
                'no_training': True, 'no_expensive_objective_gradients': True}
    dump(OUT/'protocol.json', protocol)
    print('FROZEN', ETAS, flush=True)


def instance(state, protocol):
    old = reuse.prior.load(reuse.ARTIFACTS/'protocol.json')
    single = copy.deepcopy(old)
    single['states'] = [state]
    env = reuse.FrozenEnsemble(single)  # reuse loader; exactly one state, never ensemble-map.
    fn = lambda delta: env.state_function(0, delta, protocol['h2'])
    return env, fn


def qualify(state, protocol):
    started = time.perf_counter()
    env, fn = instance(state, protocol)
    base = env.zero
    slopes, matrix = dense_jacobian(fn, base)
    assert matrix.shape == (128, 9446)
    print('JACOBIAN', state['regime'], tuple(matrix.shape), flush=True)
    cached = torch.load(reuse.ARTIFACTS/'window_arrays.pt', weights_only=False)
    cached_slopes = next(row['slope'] for row in cached if row['seed'] == state['seed']
                         and row['regime'] == state['regime'] and row['window'] == 'h2').flatten().to(base.device)
    assert torch.equal(slopes, cached_slopes), reuse.errors(slopes, cached_slopes)
    parameters = env.base[0]
    flat = torch.cat([parameters[item['name']].flatten() for item in env.order])
    assert len(flat) == 9446
    assert all(torch.equal(flat[item['start']:item['stop']].reshape(item['shape']), parameters[item['name']])
               for item in env.order)
    identities = []
    _, back = torch.func.vjp(fn, base)
    for seed in protocol['identity_random_seeds']:
        direction = unit(9446, seed, base.device)
        output = unit(128, seed, base.device)
        direct = torch.func.jvp(fn, (base,), (direction,))[1]
        a = reuse.errors(matrix @ direction, direct)
        b = reuse.errors(matrix.T @ output, back(output)[0])
        assert a['relative_L2'] <= protocol['identity_relative_gate'], a
        assert b['relative_L2'] <= protocol['identity_relative_gate'], b
        identities.append({'seed': seed, 'JVP': a, 'VJP': b})
    # Exact small rectangular SVD, no randomized range approximation.
    cpu = matrix.detach().cpu().numpy()
    u, singular, vh = np.linalg.svd(cpu, full_matrices=False)
    spectrum = reuse.spectral_metrics(torch.from_numpy(singular.copy()))
    spectrum.update({'stable_rank': float(np.sum(singular**2)/singular[0]**2),
                     'ranks': {str(threshold): int(np.sum(singular/singular[0] >= threshold))
                               for threshold in protocol['rank_relative_thresholds']}})
    leading = torch.from_numpy(vh[0].copy()).to(base.device)
    random = unit(9446, protocol['random_control_seed'], base.device)
    norm = float(flat.norm())
    responses = {}
    classifications = {}
    for label, direction in [('v1', leading), ('random', random)]:
        reference = matrix @ direction
        responses[label] = [finite_response(fn, base, direction, eta, reference, norm, env.safety)
                            for eta in protocol['etas']]
        classifications[label] = classify(responses[label])
        print('FD', state['regime'], label,
              [row['relative_L2'] for row in responses[label]], classifications[label]['classification'], flush=True)
    qualified = all(row['qualified'] for row in classifications.values())
    localization = []
    if not qualified:
        for name in sorted({reuse.group(item['name']) for item in env.order}):
            vector = torch.zeros_like(leading)
            for item in env.order:
                if reuse.group(item['name']) == name:
                    vector[item['start']:item['stop']] = leading[item['start']:item['stop']]
            size = float(vector.norm())
            if size == 0:
                continue
            vector /= size
            rows = [finite_response(fn, base, vector, eta, matrix@vector, norm, env.safety)
                    for eta in protocol['etas'][2:]]
            localization.append({'module': name, 'v1_projected_norm': size, 'FD': rows})
            if name == 'x_feature_extractor.1':
                for suffix in ('weight', 'bias'):
                    part = torch.zeros_like(leading)
                    for item in env.order:
                        if item['name'] == name+'.'+suffix:
                            part[item['start']:item['stop']] = leading[item['start']:item['stop']]
                    if float(part.norm()) > 0:
                        part /= part.norm()
                        checks = [finite_response(fn, base, part, eta, matrix@part, norm, env.safety)
                                  for eta in protocol['etas'][2:]]
                        localization.append({'module': name+'.'+suffix, 'FD': checks})
        # Final one-coordinate bridge, selected only from failing module projections.
        for module in [row for row in localization if not any(check['PASS'] for check in row['FD'])
                       and row['module'] in {reuse.group(item['name']) for item in env.order}]:
            eligible = [coordinate for item in env.order if reuse.group(item['name']) == module['module']
                        for coordinate in range(item['start'], item['stop'])]
            top = sorted(eligible, key=lambda coordinate: abs(float(leading[coordinate])), reverse=True)[:2]
            module['coordinate_checks'] = []
            for coordinate in top:
                vector = torch.zeros_like(leading)
                vector[coordinate] = 1
                checks = [finite_response(fn, base, vector, eta, matrix[:, coordinate], norm, env.safety)
                          for eta in protocol['etas'][2:]]
                module['coordinate_checks'].append({'flattened_coordinate': coordinate, 'FD': checks})
    model = env.models[0]
    unique = {id(value) for _, value in model.named_parameters(remove_duplicate=True) if value.requires_grad}
    aliases = {}
    for name, value in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(value), []).append(name)
    aliased_names = [names for names in aliases.values() if len(names) > 1]
    assert len(unique) == len(env.order)
    path = OUT/f"seed11_{state['regime']}_exact.npz"
    np.savez_compressed(path, J=cpu, slopes=slopes.cpu().numpy(), U=u, singular=singular,
                        Vh=vh, leading=leading.cpu().numpy(), random=random.cpu().numpy())
    result = {'state': state, 'shape': list(cpu.shape), 'finite': True, 'replay_bitwise': True,
              'parameter_roundtrip_bitwise': True, 'trainable_unique_tensors': len(unique),
              'aliases': aliased_names, 'identities': identities, 'spectrum': spectrum,
              'FD': responses, 'classifications': classifications, 'qualified': qualified,
              'localization': localization, 'parameter_norm': norm,
              'exact_restoration_max_abs': env.safety(), 'path': str(path),
              'sha256': reuse.prior.sha(path), 'seconds': time.perf_counter()-started}
    dump(OUT/f"seed11_{state['regime']}_metrics.json", result)
    return result


def run():
    protocol = load(OUT/'protocol.json')
    assert protocol['script_sha256'] == reuse.prior.sha(__file__)
    assert protocol['reuse_script_sha256'] == reuse.prior.sha(ROOT/'lap_q_parameter_sketch_diagnostic.py')
    assert protocol['etas'] == ETAS and protocol['h2'] == reuse.H2
    reuse.validate_contract()
    old = reuse.prior.load(reuse.ARTIFACTS/'protocol.json')
    before = reuse.prior_window.verify(old)
    first = qualify(protocol['states'][0], protocol)
    results = [first]
    if first['qualified']:
        results.append(qualify(protocol['states'][1], protocol))
    after = reuse.prior_window.verify(old)
    dump(OUT/'results.json', {'states': results, 'hash_before': before, 'hash_after': after,
                             'mismatches': 0, 'state_mutation': 'NONE', 'no_training': True})
    print('DONE', [row['qualified'] for row in results], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.stage]()
