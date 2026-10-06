"""Prospective seed23 replication; reuse qualified observable and array analysis."""
import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

import lap_q_exact_parameter_jacobian_diagnostic as exact
import lap_q_objective_coupling_diagnostic as analysis

ROOT = Path(__file__).parent
OUT = ROOT.parent / 'lap_q_seed23_replication_runs_20261006'
OLD = exact.reuse.ARTIFACTS


def qualification(rows):
    """Require a passing eta and a fully passing smaller-step plateau."""
    passing = [i for i, row in enumerate(rows) if row['PASS']]
    if not passing or not all(row['PASS'] for row in rows[passing[0]:]):
        return {'qualified': False, 'classification': 'STRUCTURAL-FAIL'}
    first = passing[0]
    return {'qualified': True, 'classification': 'CURVATURE-LIMITED' if first else 'PASS-LINEAR',
            'validated_etas': [row['eta'] for row in rows[first:]],
            'error_ratios': [a['relative_L2']/b['relative_L2'] if b['relative_L2'] else None
                             for a, b in pairwise(rows)]}


def replication(gproj, gresp, leading, overlap, cosines):
    gates = {'preserved_subspace': leading >= .95 and overlap >= .80,
             'increased_chemistry_coupling': gproj >= 1.5 and gresp >= 1.5,
             'opposed_secondaries': all(value <= -.90 for value in cosines)}
    label = ('REPLICATED' if all(gates.values()) else
             'PARTIAL REPLICATION' if gproj >= 1.5 or gresp >= 1.5 else 'NOT REPLICATED')
    return {'classification': label, 'gates': gates}


def verify(protocol):
    for path, digest in protocol['immutable_files'].items():
        assert analysis.sha(path) == digest, path
    assert analysis.sha(__file__) == protocol['script_sha256']
    return len(protocol['immutable_files'])


def gradients(binding):
    tensors = torch.load(binding['gradient']['path'], map_location='cpu', weights_only=False)['aggregates']
    assert list(tensors) == list(analysis.TASKS)
    assert all(value.dtype == torch.float64 and tuple(value.shape) == (9446,) for value in tensors.values())
    result = np.stack([tensors[name].numpy() for name in analysis.TASKS])
    receipt = analysis.load(binding['gradient']['receipt_path'])
    for i, name in enumerate(analysis.TASKS):
        assert abs(np.linalg.norm(result[i])/receipt['geometry']['norms'][name]-1) <= 1e-12
    assert np.isfinite(result).all()
    return result


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    exact.reuse.validate_contract()
    old = analysis.load(OLD/'protocol.json')
    init = analysis.load(analysis.INIT/'protocol.json')
    inherited = analysis.load(exact.OUT/'protocol.json')
    assert inherited['parameter_order'] == old['parameter_order']
    immutable = {str(path): analysis.sha(path) for path in
                 [OLD/'protocol.json', OLD/'window_metrics_raw.json', exact.OUT/'protocol.json',
                  analysis.INIT/'protocol.json', analysis.INIT/'matrix.py',
                  ROOT/'lap_q_exact_parameter_jacobian_diagnostic.py',
                  ROOT/'lap_q_parameter_sketch_diagnostic.py',
                  ROOT/'lap_q_objective_coupling_diagnostic.py']}
    immutable[old['probe_path']] = old['probe_sha256']
    window = analysis.load(OLD/'window_metrics_raw.json')
    immutable[str(OLD/'window_arrays.pt')] = window['arrays_sha256']
    bindings = []
    for regime in ('P67', 'P536'):
        state = next(row for row in old['states'] if row['seed'] == 23 and row['regime'] == regime)
        binding = next(row for row in old['gradient_bindings'] if row['seed'] == 23 and row['regime'] == regime)
        receipt = analysis.load(binding['receipt_path'])
        assert receipt['state_sha256'] == binding['state_sha256'] == state['state_sha256']
        assert receipt['state_file_sha256'] == state['file_sha256']
        assert receipt['array_sha256'] == binding['sha256']
        assert receipt['diagnostic_script_sha256'] == analysis.sha(analysis.INIT/'matrix.py')
        assert receipt['protocol_sha256'] == analysis.sha(analysis.INIT/'protocol.json')
        assert receipt['source_sha256'] == init['source_sha256']
        bindings.append({'state': state, 'gradient': binding})
        immutable.update({state['path']: state['file_sha256'], binding['path']: binding['sha256'],
                          binding['receipt_path']: binding['receipt_sha256']})
    for path, digest in init['source_sha256'].items():
        immutable[str(ROOT/path)] = digest
    solver = analysis.load(bindings[0]['gradient']['receipt_path'])['geometry']['MGDA']['solver_source_sha256']
    immutable[str(analysis.SOLVER)] = solver
    protocol = {**inherited, 'starting_commit': 'eed2df1552a54d6a4a6da980523ef22311ba3d8b',
                'states': [row['state'] for row in bindings], 'bindings': bindings,
                'immutable_files': immutable, 'script_sha256': analysis.sha(__file__),
                'qualification_rule': 'All smaller etas after first PASS must PASS; tiny resolved error rebound is disclosed, not automatically classified as cancellation.',
                'replication_rule': 'REPLICATED iff leading>=.95, full overlap>=.80, G_proj>=1.5, G_resp>=1.5, all P536 chemistry-secondary response cosines<=-.90. PARTIAL if at least one chemistry gain reaches1.5 but full conjunction fails; otherwise NOT REPLICATED.',
                'singular_orientation': 'P67 each mode largest-absolute coordinate positive; P536 each corresponding mode dot P67 positive (zero dot: own largest coordinate positive). Higher-mode pairing is descriptive only.',
                'resolved_threshold': 1e-3, 'tiny_relative_component': 1e-8,
                'full_overlap_definition': 'Same seed11 formula: mean squared principal-angle cosines of leading min(resolved ranks) bases; also report full rectangular overlap if ranks differ.',
                'gradient_ordering_evidence': 'SHA-bound executed matrix.py flat(): zip with sorted unique named_trainable_parameters; same bound production helper; tied coordinates once. Legacy arrays lack embedded names.',
                'gradient_sign': '+grad L; induced response=J*(-g/||g||)',
                'scientific_gradient_precision': 'Chemistry F64 shadow; Exc/operator production F32 leaves then widened gradients. No recomputation.',
                'stop_on_qualification_failure': True, 'no_seed41': True}
    verify(protocol)
    for binding in bindings:
        gradients(binding)
    analysis.dump(OUT/'protocol.json', protocol)
    print('FROZEN', len(immutable), 'hashes; seed23 only', flush=True)


def qualify(binding, protocol):
    state = binding['state']
    env, fn = exact.instance(state, protocol)
    slopes, matrix = exact.dense_jacobian(fn, env.zero)
    assert matrix.shape == (128, 9446)
    cached = torch.load(OLD/'window_arrays.pt', weights_only=False)
    replay = next(row['slope'] for row in cached if row['seed'] == 23 and
                  row['regime'] == state['regime'] and row['window'] == 'h2').flatten().to(env.zero.device)
    assert torch.equal(slopes, replay)
    flat = torch.cat([env.base[0][row['name']].flatten() for row in env.order])
    assert env.order == protocol['parameter_order']
    assert all(torch.equal(flat[row['start']:row['stop']].reshape(row['shape']), env.base[0][row['name']])
               for row in env.order)
    identities = []
    _, back = torch.func.vjp(fn, env.zero)
    for seed in protocol['identity_random_seeds']:
        direction = exact.unit(9446, seed, env.zero.device)
        output = exact.unit(128, seed, env.zero.device)
        jvp = exact.reuse.errors(matrix@direction, torch.func.jvp(fn, (env.zero,), (direction,))[1])
        vjp = exact.reuse.errors(matrix.T@output, back(output)[0])
        assert max(jvp['relative_L2'], vjp['relative_L2']) <= 1e-8
        identities.append({'seed': seed, 'JVP': jvp, 'VJP': vjp})
    cpu = matrix.cpu().numpy()
    u, singular, vh = np.linalg.svd(cpu, full_matrices=False)
    rows, gates = {}, {}
    for label, direction in [('v1', torch.from_numpy(vh[0].copy()).to(env.zero.device)),
                             ('random', exact.unit(9446, 42, env.zero.device))]:
        rows[label] = [exact.finite_response(fn, env.zero, direction, eta, matrix@direction,
                       float(flat.norm()), env.safety) for eta in protocol['etas']]
        gates[label] = qualification(rows[label])
        print(state['regime'], label, [row['relative_L2'] for row in rows[label]], gates[label], flush=True)
    spectrum = exact.reuse.spectral_metrics(torch.from_numpy(singular.copy()))
    spectrum['stable_rank'] = float(np.sum(singular**2)/singular[0]**2)
    spectrum['ranks'] = {str(t): int(np.sum(singular/singular[0] >= t)) for t in protocol['rank_relative_thresholds']}
    path = OUT/f"seed23_{state['regime']}_exact.npz"
    np.savez_compressed(path, J=cpu, U=u, singular=singular, Vh=vh, slopes=slopes.cpu().numpy())
    result = {'identity': binding, 'qualified': all(row['qualified'] for row in gates.values()),
              'FD': rows, 'classification': gates, 'identities': identities, 'spectrum': spectrum,
              'shape': [128, 9446], 'replay_bitwise': True, 'ordering_roundtrip': True,
              'restoration_max_abs': env.safety(), 'path': str(path), 'sha256': analysis.sha(path)}
    analysis.dump(OUT/f"{state['regime']}_qualification.json", result)
    return result


def run():
    protocol = analysis.load(OUT/'protocol.json')
    before = verify(protocol)
    qualifications, arrays = {}, {}
    for binding in protocol['bindings']:
        name = binding['state']['regime']
        qualifications[name] = qualify(binding, protocol)
        if not qualifications[name]['qualified']:
            analysis.dump(OUT/'results.json', {'classification': 'QUALIFICATION FAILED',
                          'qualifications': qualifications, 'hash_before': before, 'hash_after': verify(protocol)})
            return
        arrays[name] = dict(np.load(qualifications[name]['path']))
    for i in range(128):
        a, b = arrays['P67'], arrays['P536']
        sa = 1 if a['Vh'][i, np.argmax(np.abs(a['Vh'][i]))] >= 0 else -1
        a['Vh'][i] *= sa
        a['U'][:, i] *= sa
        dot = a['Vh'][i]@b['Vh'][i]
        sb = (1 if dot > 0 else -1) if dot != 0 else (1 if b['Vh'][i, np.argmax(np.abs(b['Vh'][i]))] >= 0 else -1)
        b['Vh'][i] *= sb
        b['U'][:, i] *= sb
    states = {}
    for binding in protocol['bindings']:
        name = binding['state']['regime']
        data, g = arrays[name], gradients(binding)
        metrics, projected, perpendicular = analysis.coupling(data['J'], data['Vh'], data['singular'], g)
        metrics['common_descent'] = {label: analysis.margin(value, np.linalg.norm(g, axis=1), 1e-8,
                                 protocol['immutable_files'][str(analysis.SOLVER)])
                                 for label, value in [('q', projected), ('complement', perpendicular)]}
        metrics['dominant_modes'] = [{'mode': i+1, 'sigma': float(data['singular'][i]),
                                    'energy_fraction': float(data['singular'][i]**2/np.sum(data['singular']**2)),
                                    'coefficients': {task: float(g[j]@data['Vh'][i]) for j, task in enumerate(analysis.TASKS)}}
                                   for i in range(3)]
        metrics['threshold_sensitivity'] = {}
        for threshold in (1e-2, 1e-4):
            diagnostic, _, _ = analysis.coupling(data['J'], data['Vh'], data['singular'], g, threshold)
            metrics['threshold_sensitivity'][str(threshold)] = {'rank': diagnostic['resolved_rank'],
                'fractions': {task: row['projection']['full']['fraction'] for task, row in diagnostic['objectives'].items()}}
        states[name] = metrics
    ranks = [states[name]['resolved_rank'] for name in ('P67', 'P536')]
    overlaps = {str(k): {'used_k': min(k, min(ranks)), **analysis.overlap(arrays['P67']['Vh'][:min(k, min(ranks))],
                arrays['P536']['Vh'][:min(k, min(ranks))])} for k in (1, 2, 3, 4, 8)}
    overlaps['full'] = {'used_k': min(ranks), **analysis.overlap(arrays['P67']['Vh'][:min(ranks)], arrays['P536']['Vh'][:min(ranks)])}
    overlaps['full_rectangular'] = analysis.overlap(arrays['P67']['Vh'][:ranks[0]], arrays['P536']['Vh'][:ranks[1]])
    a, b = states['P67']['objectives'], states['P536']['objectives']
    gproj = b['full251']['projection']['full']['fraction']/a['full251']['projection']['full']['fraction']
    ratios = {task: b[task]['induced']['norm']/a[task]['induced']['norm'] for task in analysis.TASKS}
    leading = float(abs(arrays['P67']['Vh'][0]@arrays['P536']['Vh'][0]))
    cosines = [states['P536']['pairs']['full251__'+task]['induced_q_cosine'] for task in analysis.TASKS[1:]]
    decision = replication(gproj, ratios['full251'], leading, overlaps['full']['mean_squared_overlap'], cosines)
    result = {'protocol_sha256': analysis.sha(OUT/'protocol.json'), 'qualifications': qualifications,
              'states': states, 'overlap': overlaps, 'leading_abs_cosine': leading, 'G_proj': gproj,
              'G_resp': ratios['full251'], 'response_ratios': ratios, **decision,
              'hash_before': before, 'hash_after': verify(protocol), 'mismatches': 0,
              'parameter_order_compatible': True, 'state_mutation': 'NONE',
              'expensive_gradient_recomputations': 0, 'no_training': True}
    analysis.dump(OUT/'results.json', result)
    print(decision, 'G_proj', gproj, 'G_resp', ratios['full251'], 'overlap', overlaps['full'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
