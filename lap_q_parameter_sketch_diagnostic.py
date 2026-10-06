"""Bounded matrix-free q-controllability diagnostic, not a training entrypoint."""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch

ARTIFACTS = Path('C:/Dev/readWFN_share_ms/lap_q_controllability_runs_20261005')
sys.path.insert(0, str(ARTIFACTS))
import probe as prior
import window as prior_window

OUT = Path('C:/Dev/readWFN_share_ms/lap_q_parameter_sketch_h2_runs_20261005')
H2 = [0.023218985940978362, 0.042529125693106434]
H3 = [0.011609492970489181, 0.021264562846553217]


def group(name):
    for prefix in ('x_output_layer', 'c_output_layer', 'c_input_layers',
                   'c_symmetrization_blocks', 'c_post_symm_blocks'):
        if name.startswith(prefix + '.'):
            return prefix
    for index in (0, 1, 3, 4):
        if name.startswith(f'x_feature_extractor.{index}.'):
            return f'x_feature_extractor.{index}'
    raise ValueError(name)


def probes(rows, width, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return (2 * torch.randint(0, 2, (rows, width), generator=generator) - 1).double() / math.sqrt(rows)


def row_views(states):
    views = {'FULL': [], 'NON_HEAD': [], 'PBE_HEAD_ONLY': []}
    for index, state in enumerate(states):
        rows = list(range(index * 128, (index + 1) * 128))
        views['FULL'].extend(rows)
        views['PBE_HEAD_ONLY' if state['regime'] == 'PBE_HEAD' else 'NON_HEAD'].extend(rows)
    return views


def spectral_metrics(s):
    s = s.detach().double().cpu()
    energy = s.square()
    total = float(energy.sum())
    if total == 0:
        return {'singular_values': s.tolist(), 'normalized': None, 'numerical_rank': 0,
                'r99': 0, 'effective_rank': 0., 'sketch_frobenius': 0.}
    probability = energy / total
    positive = probability > 0
    return {'singular_values': s.tolist(), 'normalized': (s / s[0]).tolist(),
            'numerical_rank': int((s / s[0] >= 1e-3).sum()),
            'r99': int(torch.searchsorted(energy.cumsum(0) / total, .99)) + 1,
            'effective_rank': float(torch.exp(-(probability[positive] * probability[positive].log()).sum())),
            'concentration': float(energy[0] / total),
            'sketch_frobenius': math.sqrt(total), 'sigma1': float(s[0]),
            'sigma_median_nonzero': float(s[s > 0].median()),
            'cumulative_energy': (energy.cumsum(0) / total).tolist()}


def errors(actual, reference):
    absolute = float((actual - reference).norm())
    norm = float(reference.norm())
    return {'absolute_L2': absolute, 'relative_L2': absolute / norm if norm > 0 else None,
            'cosine': float(actual @ reference / (actual.norm() * reference.norm()))
            if float(actual.norm()) > 0 and norm > 0 else None}


def orthonormalize(z):
    q, _ = torch.linalg.qr(z.detach().double().cpu(), mode='reduced')
    error = float((q.T @ q - torch.eye(q.shape[1], dtype=torch.float64)).abs().max())
    assert error <= 1e-8, error
    return q.to(z.device), error


class FrozenEnsemble:
    """Common additive coordinate displacement; stored states are never averaged."""
    def __init__(self, protocol):
        self.states = protocol['states']
        self.models = [prior.model_for(state) for state in self.states]
        self.order = prior.ordering(self.models[0])
        assert self.order == protocol['parameter_order']
        self.names = [item['name'] for item in self.order]
        self.numel = self.order[-1]['stop']
        self.base = []
        self.saved = []
        for model in self.models:
            parameters = prior.named_trainable_parameters(model)
            assert list(parameters) == self.names
            assert set(parameters) == {name for name, value in model.named_parameters(remove_duplicate=True) if value.requires_grad}
            assert all(value.dtype == torch.float64 for value in parameters.values())
            self.base.append({name: value.detach().clone() for name, value in parameters.items()})
            self.saved.append(prior.state_hash(model))
        data = torch.load(protocol['probe_path'], weights_only=False)
        self.raw = data['raw'].cuda()
        self.identities = data['identities']
        self.views = row_views(self.states)
        self.zero = torch.zeros(self.numel, dtype=torch.float64, device='cuda')

    def state_function(self, index, delta, halfwidth):
        model = self.models[index]
        mapped = {item['name']: self.base[index][item['name']] + delta[item['start']:item['stop']].reshape(item['shape'])
                  for item in self.order}
        def local(raw):
            return torch.func.functional_call(model, mapped, (raw,), strict=False, tie_weights=True)
        return prior.contrast(local, self.raw, torch.tensor(halfwidth, dtype=torch.float64, device='cuda'))

    def function(self, view, halfwidth=H2):
        indices = [i for i, state in enumerate(self.states)
                   if view == 'FULL' or (view == 'PBE_HEAD_ONLY') == (state['regime'] == 'PBE_HEAD')]
        def evaluate(delta):
            return torch.cat([self.state_function(i, delta, halfwidth) for i in indices])
        return evaluate

    def safety(self):
        difference = 0.
        for index, model in enumerate(self.models):
            assert prior.state_hash(model) == self.saved[index]
            for name, value in prior.named_trainable_parameters(model).items():
                difference = max(difference, float((value.detach() - self.base[index][name]).abs().max()))
                assert value.grad is None
        assert difference == 0
        return difference


def group_fractions(vector, order):
    values = {}
    for item in order:
        name = group(item['name'])
        values[name] = values.get(name, 0.) + float(vector[item['start']:item['stop']].square().sum())
    total = sum(values.values())
    return {name: value / total if total > 0 else None for name, value in values.items()}


def sketch(function, zero, width, seed, order):
    begin = time.perf_counter()
    s, pullback = torch.func.vjp(function, zero)
    omega = probes(len(s), width, seed).to(zero.device)
    z = torch.stack([pullback(omega[:, index])[0] for index in range(width)], 1)
    q, orthogonality = orthonormalize(z)
    y = torch.stack([torch.func.jvp(function, (zero,), (q[:, index],))[1] for index in range(width)], 1)
    u, singular, vh = torch.linalg.svd(y, full_matrices=False)
    leading = q @ vh.T
    metric = spectral_metrics(singular)
    # Hutchinson trace estimator: probes have variance1/M.
    estimate = math.sqrt(len(s) / width * float(z.square().sum()))
    metric.update({'frobenius_estimate': estimate, 'frobenius_estimator': 'sqrt(M/r sum||J^T Omega_k||^2)',
                   'sigma1_over_frobenius_estimate': float(singular[0]) / estimate if estimate > 0 else None,
                   'orthogonality_max_abs': orthogonality, 'width': width, 'seed': seed})
    participation = [group_fractions(leading[:, index], order) for index in range(width)]
    weights = singular.square() / singular.square().sum()
    metric['weighted_group_participation'] = {name: sum(float(weights[i]) * participation[i][name] for i in range(width))
                                             for name in participation[0]}
    metric['leading12_group_participation'] = participation[:12]
    metric['per_row_projected_jacobian_norms'] = torch.linalg.vector_norm(y, dim=1).detach().cpu().tolist()
    norm = float(s.norm())
    gradient_r = pullback(s)[0]
    metric['response_norm_gradient'] = {'norm': float(gradient_r.norm()), 'group_fractions': group_fractions(gradient_r, order)}
    metric['current_response'] = {'s_norm': norm, 'vjp_unit_current_norm': float(pullback(s / norm)[0].norm()) if norm > 0 else None,
                                  'subspace_projection': float((u.T @ (s / norm)).square().sum()) if norm > 0 else None}
    metric['seconds'] = time.perf_counter() - begin
    return metric, {'Z': z.detach().cpu().numpy(), 'Q': q.detach().cpu().numpy(), 'Y': y.detach().cpu().numpy(),
                    'U': u.detach().cpu().numpy(), 'singular': singular.detach().cpu().numpy(),
                    'leading_parameter_directions': leading.detach().cpu().numpy(), 'current_slopes': s.detach().cpu().numpy()}, leading, y @ vh.T


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Protocol immutable'
    old = prior.load(ARTIFACTS / 'protocol.json')
    hashes = prior_window.verify(old)
    ensemble = FrozenEnsemble(old)
    manifest = [{'name': item['name'], 'shape': item['shape'], 'numel': item['numel'], 'requires_grad': True,
                 'production_status': 'trainable', 'start': item['start'], 'stop': item['stop'], 'group': group(item['name'])}
                for item in ensemble.order]
    prior.dump(OUT / 'parameter_manifest.json', {'parameters': manifest, 'numel': ensemble.numel,
               'production_evidence': 'train_lap_moo.py760 loads Lap checkpoint; no freezing before update; lap_moo_training.named_trainable_parameters selects sorted unique trainable leaves.'})
    prior.dump(OUT / 'parameter_group_map.json', {item['name']: item['group'] for item in manifest})
    with torch.no_grad():
        values = ensemble.function('FULL')(ensemble.zero).cpu()
    rows = []
    for i, state in enumerate(ensemble.states):
        for spin in range(2):
            for point in range(64):
                index = i * 128 + spin * 64 + point
                rows.append({'row': index, 'seed': state['seed'], 'regime': state['regime'], 'spin': spin,
                             'probe_point': point, 'source_identity': ensemble.identities[point],
                             'pbe_head': state['regime'] == 'PBE_HEAD', 'current_slope': float(values[index])})
    prior.dump(OUT / 'row_partition.json', {'views': ensemble.views, 'row_metadata': rows,
               'invalid_or_padding_rows': [], 'zero_response_is_not_a_row_exclusion': True})
    contract = {'starting_commit': '8946334d5f3d3d6c77839f13469fe7aeebeb30b7', 'h2': H2, 'h3': H3,
                'states': ensemble.states, 'probe_path': old['probe_path'], 'probe_sha256': old['probe_sha256'],
                'original_protocol_path': str(ARTIFACTS / 'protocol.json'), 'original_protocol_sha256': prior.sha(ARTIFACTS / 'protocol.json'),
                'row_views': {name: len(indices) for name, indices in ensemble.views.items()},
                'parameter_coordinates': '9446 sorted unique trainable coordinates; f(delta)=concat_state S(theta_state+delta); shared additive displacement across aligned states, not independently parameterized block-diagonal models or averaged parameters.',
                'interpretation': 'Ensemble rank is not an individual-state rank. NON_HEAD mixes P67/P536/TANGENT. State image energies and state current-response VJPs are reported; no universal P536 capacity claim from ensemble spectrum.',
                'parameter_manifest_sha256': prior.sha(OUT / 'parameter_manifest.json'),
                'parameter_group_map_sha256': prior.sha(OUT / 'parameter_group_map.json'),
                'row_partition_sha256': prior.sha(OUT / 'row_partition.json'),
                'script_sha256': prior.sha(__file__), 'sketch_width': 40, 'seeds': [42, 314159],
                'repeat_width_reduction': 'Only if timing projection exceeds7.5h: repeat width24; freeze before repeat.',
                'parameter_fd': {'relative_norm': 1e-5, 'reference_norm': 'max RMS audited parameter norm over NON_HEAD states,1; one common displacement for ensemble.', 'gate': .05, 'no_epsilon_search': True},
                'h3_check': {'nonhead_directions': 8, 'head_directions': 2, 'median_gate': .05, 'max_gate': .10},
                'rank': {'relative_numerical_threshold': 1e-3, 'energy_rank': .99, 'entropy_effective_case_threshold': 8},
                'orthogonality_gate': 1e-8, 'head_exact_row_norms': '384 individual scalar VJPs, sequential vectors only, noMxPmatrix; head groups only nonzero expected.',
                'repeat_gate': 'Descriptive; substantial instability defined prospectively as >10% relative difference among top12 resolved singular values OR any top4 principal angle>15degrees. Near-null modes do not drive stop.',
                'case_precedence': 'D if finite-difference/h3/repeat/resolution gate fails; otherwise A/B/C as specified. No optimizer causality proven by local capacity.',
                'numerical_resolution': 'No division by exactzero. VJP/JVP adjoint dot check abs error and relative only if nonzero. Deterministic repeated response VJP; exactzero noise reported plus observed scales, no inventedabsolutegainthreshold.',
                'optional_rho_sigma': 'SKIP: no validated rho/sigma finite windows exist for sameprobe.',
                'preflight_vector': 'Same as original: seed42 CPU Rademacher128 vector dividedsqrt128 for each seed11/23/41.',
                'hash_checks': hashes, 'no_training': True, 'no_dense_jacobian': True}
    assert ensemble.safety() == 0
    prior.dump(OUT / 'protocol.json', contract)
    print('FROZEN', contract['row_views'], 'parameters', ensemble.numel, flush=True)


def preflight(ensemble):
    expected = prior.load(Path('C:/Dev/readWFN_share_ms/lap_q_parameter_sketch_runs_20261005/structural_zero_preflight.json'))['results']
    generator = torch.Generator().manual_seed(42)
    omega = (2 * torch.randint(0, 2, (128,), generator=generator) - 1).double().cuda() / math.sqrt(128)
    output = []
    for i, state in enumerate(ensemble.states):
        if state['regime'] != 'PBE_HEAD':
            continue
        function = lambda delta, i=i: ensemble.state_function(i, delta, H2)
        value, pullback = torch.func.vjp(function, ensemble.zero)
        gradient = pullback(omega)[0]
        norms = {}
        for item in ensemble.order:
            name = group(item['name'])
            norms[name] = norms.get(name, 0.) + float(gradient[item['start']:item['stop']].square().sum())
        old = next(row for row in expected if row['seed'] == state['seed'])
        # Different autograd graph parameter accumulation order may change last bits.
        discrepancy = abs(float(gradient.norm()) - old['vjp_norm']) / old['vjp_norm']
        assert discrepancy <= 1e-12
        assert bool((value == 0).all())
        assert all(norm == 0 for name, norm in norms.items() if 'output_layer' not in name)
        output.append({'seed': state['seed'], 'slope_max_abs': float(value.abs().max()),
                       'vjp_norm': float(gradient.norm()), 'relative_reproduction_error': discrepancy,
                       'group_squared_norms': norms})
    prior.dump(OUT / 'pbe_head_preflight_reproduction.json', {'PASS': True, 'rows': output})
    print('PREFLIGHT PASS', flush=True)


def validate_contract():
    contract = prior.load(OUT / 'protocol.json')
    assert prior.sha(__file__) == contract['script_sha256']
    assert prior.sha(contract['original_protocol_path']) == contract['original_protocol_sha256']
    old = prior.load(contract['original_protocol_path'])
    assert contract['states'] == old['states']
    assert contract['h2'] == H2 and contract['h3'] == H3
    assert contract['probe_sha256'] == old['probe_sha256']
    assert prior.sha(contract['probe_path']) == contract['probe_sha256']
    for name in ('parameter_manifest', 'parameter_group_map', 'row_partition'):
        assert prior.sha(OUT / (name + '.json')) == contract[name + '_sha256'], name
    partition = prior.load(OUT / 'row_partition.json')
    expected = row_views(old['states'])
    assert partition['views'] == expected
    assert contract['row_views'] == {name: len(rows) for name, rows in expected.items()}
    assert len(partition['row_metadata']) == 1536
    for index, row in enumerate(partition['row_metadata']):
        state = old['states'][index // 128]
        assert row['row'] == index and row['seed'] == state['seed'] and row['regime'] == state['regime']
        assert row['spin'] == index % 128 // 64 and row['probe_point'] == index % 64
        assert row['pbe_head'] == (state['regime'] == 'PBE_HEAD')
    manifest = prior.load(OUT / 'parameter_manifest.json')
    assert manifest['numel'] == old['parameter_order'][-1]['stop'] == 9446
    for entry, original in zip(manifest['parameters'], old['parameter_order'], strict=True):
        assert all(entry[key] == original[key] for key in ('name', 'shape', 'numel', 'start', 'stop'))
        assert entry['requires_grad'] is True and entry['production_status'] == 'trainable'
        assert entry['group'] == group(entry['name'])
    assert prior.load(OUT / 'parameter_group_map.json') == {entry['name']: entry['group'] for entry in manifest['parameters']}
    return contract, old


def smoke():
    _, protocol = validate_contract()
    ensemble = FrozenEnsemble(protocol)
    preflight(ensemble)
    index = next(i for i, s in enumerate(ensemble.states) if s['seed'] == 11 and s['regime'] == 'P536')
    function = lambda delta: ensemble.state_function(index, delta, H2)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    s, pullback = torch.func.vjp(function, ensemble.zero)
    omega = probes(len(s), 40, 42).cuda()
    gradient = pullback(omega[:, 0])[0]
    direction = gradient / gradient.norm()
    torch.cuda.synchronize()
    reverse_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    _, image = torch.func.jvp(function, (ensemble.zero,), (direction,))
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - begin
    dot = errors((omega[:, 0] @ image).reshape(1), (gradient @ direction).reshape(1))
    assert dot['relative_L2'] is not None and dot['relative_L2'] <= 1e-10
    projected = 3 * 2 * (12 + 9 + 3) * 40 * (reverse_seconds + forward_seconds) + 300
    budget = {'smoke_state': '11_P536', 'vjp_including_forward_seconds': reverse_seconds,
              'jvp_seconds': forward_seconds, 'projected_compute_seconds': projected,
              'reserve_seconds': 1800, 'primary_width': 40, 'repeat_width': 24 if projected > 27000 else 40,
              'adjoint_dot_error': dot, 'state_restoration_diff': ensemble.safety()}
    prior.dump(OUT / 'timing_budget.json', budget)
    print('TIMING', budget, flush=True)


def run():
    _contract, old = validate_contract()
    assert prior.load(OUT / 'initial_independent_review.json')['status'] == 'PASS'
    before = prior_window.verify(old)
    ensemble = FrozenEnsemble(old)
    preflight(ensemble)
    budget = prior.load(OUT / 'timing_budget.json')
    all_metrics = {}
    artifacts = {}
    models_norms = [sum(float(value.square().sum()) for value in ensemble.base[i].values()) ** .5
                   for i, state in enumerate(ensemble.states) if state['regime'] != 'PBE_HEAD']
    epsilon = 1e-5 * max((sum(x*x for x in models_norms) / len(models_norms)) ** .5, 1.)
    for view, seed in [('NON_HEAD', 42), ('FULL', 42), ('PBE_HEAD_ONLY', 42),
                       ('NON_HEAD', 314159), ('FULL', 314159), ('PBE_HEAD_ONLY', 314159)]:
        key = f'{view}_{seed}'
        width = 40 if seed == 42 else budget['repeat_width']
        fn = ensemble.function(view)
        metric, tensors, directions, images = sketch(fn, ensemble.zero, width, seed, ensemble.order)
        metric['state_output_image_energy'] = {}
        state_order = [state for state in ensemble.states if view == 'FULL' or (view == 'PBE_HEAD_ONLY') == (state['regime'] == 'PBE_HEAD')]
        for i, state in enumerate(state_order):
            metric['state_output_image_energy'][f"{state['seed']}_{state['regime']}"] = float(images[i*128:(i+1)*128].square().sum())
        name = {'NON_HEAD': 'nonhead', 'FULL': 'full', 'PBE_HEAD_ONLY': 'head'}[view]
        array_path = OUT / f'{name}_sketch_seed{seed}.npz'
        np.savez_compressed(array_path, **tensors)
        artifacts[str(array_path)] = prior.sha(array_path)
        with (OUT / f'singular_spectrum_{name}_seed{seed}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['index', 'sigma', 'normalized', 'cumulative_energy'])
            writer.writerows(zip(range(1, width+1), metric['singular_values'], metric['normalized'], metric['cumulative_energy']))
        if seed == 42:
            check_count = 8 if view == 'NON_HEAD' else 2 if view == 'PBE_HEAD_ONLY' else 0
            checks = []
            for i in range(check_count):
                reference = images[:, i]
                shadow = torch.func.jvp(ensemble.function(view, H3), (ensemble.zero,), (directions[:, i],))[1]
                checks.append({'direction': i+1, **errors(shadow, reference)})
            metric['h2_h3'] = checks
            if view == 'NON_HEAD':
                plus = fn(ensemble.zero + epsilon * directions[:, 0])
                minus = fn(ensemble.zero - epsilon * directions[:, 0])
                central = (plus-minus) / (2*epsilon)
                metric['parameter_linearity'] = {'epsilon': epsilon, 'norm_rule': '1e-5 max(RMS NON_HEAD parameter norm,1)',
                     **errors(central, images[:, 0]), 'restoration_max_abs': ensemble.safety()}
                adjoint_s, back = torch.func.vjp(fn, ensemble.zero)
                grad = back(adjoint_s)[0]
                again_s, again_back = torch.func.vjp(fn, ensemble.zero)
                again = again_back(again_s)[0]
                metric['gradient_repeat_max_abs'] = float((grad-again).abs().max())
        all_metrics[key] = metric
        prior.dump(OUT / f'{name}_metrics_seed{seed}.json', metric)
        print('SKETCH', key, 'rank', metric['numerical_rank'], 'effective', metric['effective_rank'],
              'sigma1', metric['sigma1'], 'seconds', metric['seconds'], flush=True)
        del fn, directions, images, tensors
        torch.cuda.empty_cache()
    # HEAD-only row norms are exactly measured one VJP vector at a time; never stack MxP.
    head_fn = ensemble.function('PBE_HEAD_ONLY')
    hs, head_back = torch.func.vjp(head_fn, ensemble.zero)
    row_norms = []
    for i in range(len(hs)):
        unit = torch.zeros_like(hs)
        unit[i] = 1
        row_norms.append(float(head_back(unit)[0].norm()))
    prior.dump(OUT / 'head_exact_row_norms.json', {'norms': row_norms, 'mean': float(np.mean(row_norms)),
        'median': float(np.median(row_norms)), 'p10': float(np.quantile(row_norms, .1)),
        'p90': float(np.quantile(row_norms, .9)), 'max': max(row_norms), 'exact': True})
    # State-level current-response diagnostics prevent assigning ensemble capacity to every state.
    state_control = []
    for i, state in enumerate(ensemble.states):
        fn = lambda delta, i=i: ensemble.state_function(i, delta, H2)
        value, back = torch.func.vjp(fn, ensemble.zero)
        norm = float(value.norm())
        gradient = back(value)[0]
        state_control.append({**state, 's_norm': norm, 'response_norm_gradient': float(gradient.norm()),
                              'unit_current_vjp_norm': float(back(value/norm)[0].norm()) if norm > 0 else None})
    prior.dump(OUT / 'state_current_response_control.json', state_control)
    after = prior_window.verify(old)
    prior.dump(OUT / 'hash_audit.json', {'before': before, 'after': after, 'mismatches': 0,
               'parameter_restoration_max_abs': ensemble.safety(), 'state_mutation': 'NONE'})
    prior.dump(OUT / 'all_metrics.json', {'partitions': all_metrics, 'artifacts': artifacts, 'no_dense_jacobian': True,
               'head_exact_row_norms': prior.load(OUT / 'head_exact_row_norms.json'), 'state_control': state_control})


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'smoke', 'run'])
    args = parser.parse_args()
    {'freeze': freeze, 'smoke': smoke, 'run': run}[args.stage]()
