"""Focused offline sampling/aggregation/gate checks; no scientific evaluations."""
import numpy as np
import pytest

import lap_chemistry_estimator_fidelity_arena as audit


def catalog():
    return [{'database': db, 'reaction_id': str(i), 'variant_suffix': 'canonical'}
            for db, n in [('small', 4), ('large', 247)] for i in range(n)]


def test_frozen_draws_deterministic_nested_uniform_no_replacement():
    a = audit.draw_manifest(catalog(), census=True)
    b = audit.draw_manifest(catalog(), census=True)
    assert a == b and len(a['replicates']) == 16
    for rep in range(16):
        previous = set()
        for k in audit.KS:
            rows = audit.batch_rows(a, rep, k)
            ids = [i for i, _ in rows]
            assert len(ids) == len(set(ids)) and previous <= set(ids)
            assert sum(w for _, w in rows) == pytest.approx(1)
            previous = set(ids)
        for db in a['database_order']:
            actual = sum(w for i, w in rows if a['catalog'][i]['database'] == db)
            assert actual == pytest.approx(a['counts'][db]/251)


def test_exact_K8_infeasible_on_four_reaction_database():
    manifest = audit.draw_manifest(catalog())
    assert audit.batch_rows(manifest, 0, 8) is None
    assert len(audit.batch_rows(manifest, 0, 4)) == 8


def test_cached_singleton_weighted_gradient_equals_direct_linear_sum():
    gradients = np.array([[1., 2.], [4., 3.], [-1., 2.]])
    rows = [(11, .2), (2, .3), (5, .5)]
    result = audit.estimate(gradients, [2, 5, 11], rows)
    np.testing.assert_array_equal(result, .2*gradients[2]+.3*gradients[0]+.5*gradients[1])


def fixture(failures=0, tail=-.01):
    return [{'key': f'state{i//16}', 'K': 1, 'replicate': i % 16,
             'fullchem_descent': i >= failures, 'p_full': tail if i < failures else .1,
             'cosine': .9, 'gamma': .1, 'solver_qualified': True, 'reaction_evaluations': 8}
            for i in range(192)]


def test_qualification_overall_state_tail_and_solver_failures_not_hidden():
    rows = fixture()
    assert audit.summarize(rows)[0]['qualifies']
    # Two failures at one state fail the per-state>=90% gate despite 190/192 success.
    assert not audit.summarize(fixture(2))[0]['qualifies']
    # One catastrophic negative fails despite 191/192 successful directions.
    assert not audit.summarize(fixture(1, -.5))[0]['qualifies']
    rows[0]['solver_qualified'] = False
    assert not audit.summarize(rows)[0]['qualifies']


def test_solver_descent_sign_and_true_gradient_replacement_is_not_sampling():
    gradients = np.array([[1., 0.], [1., 1.], [2., 1.], [3., -1.]])
    direction, _ = audit.prior.arena.offline.maxmin(gradients, audit.sha(audit.prior.arena.offline.reuse.SOLVER))
    assert (gradients@direction > 0).all()
    gold = -gradients[0]
    assert gold@direction < 0
    # Gold is used to score the already constructed direction, not to change it.
    replay, _ = audit.prior.arena.offline.maxmin(gradients, audit.sha(audit.prior.arena.offline.reuse.SOLVER))
    np.testing.assert_array_equal(direction, replay)
