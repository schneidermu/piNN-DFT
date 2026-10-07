"""Bounded vector-only checks of CV pairing, exactification, gates and costs."""
import numpy as np
import pytest

import lap_chemistry_db_variance_svrg_audit as audit


def test_same_ids_weights_control_variate_identity():
    reference = np.arange(12, dtype=float).reshape(4, 3)
    gold = reference.mean(axis=0)
    batch = [(0, .25), (3, .75)]
    np.testing.assert_array_equal(audit.cv_gradient(gold, reference, reference, batch), gold)
    delta = np.array([1., -2., 3.])
    np.testing.assert_array_equal(audit.cv_gradient(gold, reference+delta, reference, batch), gold+delta)


def test_leave_one_exact_changes_only_its_weighted_contribution():
    g = np.array([[1., 0.], [3., 0.], [0., 4.], [0., 8.]])
    m = {'database_order': ['a', 'b'], 'catalog': [{'database': db} for db in ['a', 'a', 'b', 'b']]}
    parts = audit.db_parts(g, m, [(0, .5), (2, .5)])
    estimate = parts['a']+parts['b']
    exact_a = g[:2].mean(axis=0)*.5
    np.testing.assert_array_equal(estimate-parts['a']+exact_a, exact_a+parts['b'])


def fixture():
    return [{'case': f'case{i//16}', 'p_full': .1, 'cosine': .9, 'gamma': .1, 'solver_qualified': True} for i in range(128)]


def test_literal_case_gate_tail_and_solver():
    rows = fixture()
    assert audit.qualify(rows)['qualifies']
    rows[0]['p_full'] = -.01
    assert audit.qualify(rows)['qualifies']
    rows[1]['p_full'] = -.01
    assert not audit.qualify(rows)['qualifies']
    rows = fixture()
    rows[0]['p_full'] = -.03
    assert not audit.qualify(rows)['qualifies']
    rows = fixture()
    rows[0]['solver_qualified'] = False
    assert not audit.qualify(rows)['qualifies']


def test_deployable_cost_not_cached_free():
    assert audit.cost(1, False)['total_backwards_10'] == 411
    assert audit.cost(2, False)['total_backwards_10'] == 571
    assert audit.cost(1, True)['total_backwards_10'] == 662
    assert audit.cost(2, True)['total_backwards_10'] == 822


def test_counterfactual_probability_and_lower_tail_signs():
    result = audit.impact([{'base_p_full': -.3, 'exact_p_full': .1}, {'base_p_full': .2, 'exact_p_full': .3}])
    assert result['descent_probability_uplift'] == .5
    assert result['catastrophic_tail_reduction'] == 1
    assert result['p_full_change']['median'] == pytest.approx(.25)
