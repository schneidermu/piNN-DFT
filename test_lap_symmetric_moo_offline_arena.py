"""Diagnostic geometry and selection tests; no models or objective evaluation."""
import numpy as np
import torch

import lap_symmetric_moo_offline_arena as arena


def solve(g):
    return arena.maxmin(np.array(g, dtype=float), arena.reuse.sha(arena.reuse.SOLVER))


def test_identical_and_orthogonal():
    d, c = solve([[1, 0]]*4)
    np.testing.assert_allclose(d, [1, 0], atol=1e-12)
    assert abs(c['gamma_star']-1) < 1e-12
    d, c = solve(np.eye(4))
    np.testing.assert_allclose(d, [.5]*4, atol=1e-12)
    assert abs(c['gamma_star']-.5) < 1e-12


def test_opposing_has_no_strict_margin():
    _, c = solve([[1, 0], [-1, 0]])
    assert c['stationary_label'] and c['gamma_star'] < 1e-12


def test_scaling_invariance_known_geometry_and_determinism():
    g = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1.]])
    a, c = solve(g)
    b, _ = solve(g*np.array([1, 10, .01, 400])[:, None])
    np.testing.assert_allclose(a, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_array_equal(a, solve(g)[0])
    assert abs(c['gamma_star']-1/np.sqrt(2)) < 1e-12


def test_unit_progress_native_scale_and_sign():
    g = np.eye(4)
    a = arena.progress(g, np.ones(4), .5)
    b = arena.progress(g, np.ones(4)*123, .5)
    assert a['p'] == b['p'] == [.5]*4
    assert a['efficiency'] == 1 and a['common_descent']
    assert not arena.progress(g, -np.ones(4), .5)['common_descent']


def test_tiers_and_redundancy_boundaries():
    row = {'status': 'PASS', 'common_descent': True, 'p_min': .1}
    assert arena.tier([row]*4) == 1
    bad = {**row, 'common_descent': False, 'p_min': -.02}
    assert arena.tier([row]*3+[bad]) == 2
    assert arena.tier([row]*2+[bad]*2) == 3
    assert arena.redundant([{'abs_cosine': .995, 'max_progress_difference': .01}]*4)
    assert not arena.redundant([{'abs_cosine': .9949, 'max_progress_difference': .01}]*4)


def test_selection_lexicographic_and_redundancy():
    base = {'tier': 1, 'worst_p_min': .1, 'minimum_efficiency': .5,
            'efficiency_range': .1, 'median_p_min': .2, 'median_imbalance': .1}
    summary = {name: base.copy() for name in ('IMTL-G', 'CAGrad', 'Nash-MTL', 'UNIT_MEAN')}
    diversity = {name: [{'abs_cosine': .9, 'max_progress_difference': .1}]*4 for name in summary}
    assert arena.selection(summary, diversity)['slot2'] == 'IMTL-G'
    summary['CAGrad']['worst_p_min'] = .11
    assert arena.selection(summary, diversity)['slot2'] == 'CAGrad'
    diversity['CAGrad'] = [{'abs_cosine': 1, 'max_progress_difference': 0}]*4
    assert arena.selection(summary, diversity)['slot2'] == 'IMTL-G'


def test_existing_candidate_cold_start_determinism():
    g = torch.eye(4, dtype=torch.float64)+1
    structured = {name: {'fixture': g[i]} for i, name in enumerate(arena.TASKS)}
    for method, hparams in arena.METHODS.values():
        kwargs = {'method': method, 'hyperparameters': hparams, 'state': None, 'task_order': arena.TASKS}
        a, da, sa = arena.aggregate_task_gradients(structured, **kwargs)
        b, db, sb = arena.aggregate_task_gradients(structured, **kwargs)
        assert torch.equal(a['fixture'], b['fixture'])
        assert da == db and sa == sb
