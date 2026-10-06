"""Controller/state qualification with tiny deterministic fixtures."""
import copy

import numpy as np
import torch

import lap_symmetric_moo_trajectory_arena as a


def fixture(method, reject=False):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1)
    base = model.weight.detach().clone()
    def objective():
        loss = model.weight.square().sum()
        return loss+100 if reject and not torch.equal(model.weight.detach(), base) else loss
    factories = dict.fromkeys(a.ORDER, objective)
    solver_sha = a.offline.reuse.sha(a.offline.reuse.SOLVER)
    def hook(raw, **kwargs):
        joint, meta, state, _ = a.aggregate(raw, method, kwargs['state'], a.ORDER, solver_sha)
        return joint, meta, state
    update = a.train_moo_update(model, None, factories, method='pcd', aggregator=hook,
              aggregator_state=None, task_order=a.ORDER, step_rule='pcd_direct_vector_armijo',
              vector_armijo={'initial_step_size': .01, 'c': 1e-4, 'rho': .5, 'max_backtracks': 20})
    return model, base, update


def test_normalized_budget_identical_and_armijo():
    states = []
    for method in ('UNIT_MAXMIN', 'Nash-MTL'):
        model, base, result = fixture(method)
        assert result.accepted
        assert abs(float((model.weight.detach()-base).norm())-.01) < 1e-12
        trial = result.diagnostics['vector_armijo']['trials'][0]
        assert trial['armijo_pass'] and trial['realized_common_descent']
        states.append(model.weight.detach().numpy())
    np.testing.assert_allclose(states[0], states[1], atol=1e-12)


def test_rejection_restores_model_nash_state_and_cursor():
    model, base, result = fixture('Nash-MTL', reject=True)
    assert not result.accepted and torch.equal(model.weight, base)
    assert result.aggregator_state is None
    assert a.advance_cursor(0, result.accepted) == 0


def test_eta_derivation_immutable_pair_budget():
    norm = 12.34
    eta = a.ALPHA_PCD*norm
    assert eta == a.ALPHA_PCD*norm
    assert a.advance_cursor(5, True) == 6


def test_resume_roundtrip_and_mismatch_fail_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'OUT', tmp_path)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(a, 'MANIFEST', manifest)
    a.offline.reuse.dump(tmp_path/'protocol.json', {'fixture': True})
    protocol = {'manifest_file_sha256': a.offline.reuse.sha(manifest), 'immutable_files': {},
                'script_sha256': a.offline.reuse.sha(a.__file__)}
    model = torch.nn.Linear(2, 1)
    state = {'method': 'nash_mtl', 'alpha': [1., 2., 3., 4.]}
    original = a.digest(model)
    rng = a.capture_rng_state()
    a.checkpoint(tmp_path/'state.pt', model, state, 5, protocol)
    with torch.no_grad():
        model.weight.add_(1)
    cursor, restored = a.resume(tmp_path/'state.pt', model, protocol)
    assert cursor == 5 and a.equal(restored, state) and a.digest(model) == original
    assert a.equal(a.capture_rng_state(), rng)
    manifest.write_text('{"changed":true}')
    try:
        a.resume(tmp_path/'state.pt', model, protocol)
    except AssertionError:
        pass
    else:
        raise AssertionError('manifest mismatch must fail closed')
    assert a.digest(model) == original


def test_stress_aggregation_is_read_only():
    raw = {t: {'weight': torch.tensor([1., 2., 3.], dtype=torch.float64)} for t in a.ORDER}
    _, _, state, _ = a.aggregate(raw, 'Nash-MTL', None, a.ORDER, a.offline.reuse.sha(a.offline.reuse.SOLVER))
    held = copy.deepcopy(state)
    _, _, _, data = a.aggregate(raw, 'Nash-MTL', state, a.ORDER, a.offline.reuse.sha(a.offline.reuse.SOLVER))
    assert a.equal(state, held) and abs(np.linalg.norm(data['unit'])-1) < 1e-12


def test_paired_winner_rules():
    common = {'trajectory_passes': 4, 'scientific_passes': 4, 'worst_Rmax10': .99,
              'paired_wins': 3, 'median_Rmax10': .98, 'severe_events': 0,
              'median_Rmax5': .995, 'systematic_pathology': False}
    summary = {'UNIT_MAXMIN': common.copy(), 'Nash-MTL': {**common, 'worst_Rmax10': .999, 'paired_wins': 1}}
    assert a.winner(summary)['classification'] == 'CLEAR UNIT_MAXMIN WIN'
    summary['UNIT_MAXMIN']['paired_wins'] = 2
    summary['Nash-MTL']['paired_wins'] = 2
    assert a.winner(summary)['classification'] == 'INCONCLUSIVE'


def test_full_monitoring_read_only_and_manifest_repeat(tmp_path, monkeypatch):
    a.offline.reuse.dump(tmp_path/'protocol.json', {'fixture': True})
    monkeypatch.setattr(a, 'OUT', tmp_path)
    model = torch.nn.Linear(1, 1, bias=False)
    shadow = copy.deepcopy(model).double()
    before, rng = a.digest(model), a.capture_rng_state()
    class Store:
        def load_variant(self, identity, variant):
            return {'db': identity[0]}
    class Systems:
        def load(self, name):
            return name
    def reaction_factory(shadow, reaction, **kwargs):
        def value():
            return a.lap_training.batch_fchem([reaction['db']], torch.tensor([2.], dtype=torch.float64),
                                              torch.tensor([1.], dtype=torch.float64))
        return value
    monkeypatch.setattr(a, 'make_reaction_objective', reaction_factory)
    monkeypatch.setattr(a, 'make_mrks_objective_factories', lambda *args, **kwargs:
                        (lambda: torch.tensor(1.), lambda: torch.tensor(2.)))
    entries = [{'per_rank': [{'reaction': {'database': 'NCCE31' if i < 251 else 'AE17', 'reaction_id': i},
                              'variant_suffix': 'frozen'}]} for i in range(268)]
    context = {'full': {'entries': entries}, 'store': Store(), 'systems': Systems(), 'disp': None, 'mrksdisp': None}
    first = a.monitor(model, shadow, context, tmp_path/'t5.json')
    second = a.monitor(model, shadow, context, tmp_path/'t10.json')
    assert first == second and a.digest(model) == before and a.equal(a.capture_rng_state(), rng)
    assert context['full']['entries'] == entries
