"""Deterministic projection/sign/overlap checks without model evaluation."""
import numpy as np

from lap_q_objective_coupling_diagnostic import (
    SOLVER,
    cosine,
    coupling,
    margin,
    overlap,
    sha,
)


def fixture():
    matrix = np.zeros((128, 4))
    matrix[:4] = np.diag([10., 1., 1e-4, 1e-5])
    gradients = np.array([[1., 2., 3., 4.], [-1., -2., 2., 1.],
                          [2., 1., 4., 3.], [-2., -1., 3., 2.]])
    return matrix, np.eye(4), np.array([10., 1., 1e-4, 1e-5]), gradients


def test_projection_fraction_absolute_norm_and_negative_step_sign():
    matrix, vh, s, gradients = fixture()
    result, projected, remainder = coupling(matrix, vh, s, gradients)
    assert result['resolved_rank'] == 2
    np.testing.assert_allclose(projected+remainder, gradients)
    np.testing.assert_allclose(projected[:, 2:], 0)
    first = result['objectives']['full251']
    assert abs(first['projection']['full']['fraction']-5/30) < 1e-15
    assert abs(first['projection']['full']['absolute_norm']-np.sqrt(5)) < 1e-15
    expected = matrix@(-gradients[0]/np.linalg.norm(gradients[0]))
    assert abs(first['induced']['norm']-np.linalg.norm(expected)) < 1e-15
    assert result['pairs']['full251__ae17']['induced_q_cosine'] < 0


def test_subspace_sign_invariance_and_same_coordinate_overlap():
    a = np.eye(4)[:2]
    b = -a
    result = overlap(a, b)
    np.testing.assert_allclose(result['canonical_correlations'], [1., 1.])
    assert result['largest_angle_degrees'] == 0


def test_relative_tiny_component_gate():
    assert cosine(np.array([1e-9, 0]), np.array([1., 0]), 1., 1., 1e-8) is None
    assert cosine(np.array([1e-7, 0]), np.array([1., 0]), 1., 1., 1e-8) == 1


def test_q_and_complement_contributions_reproduce_full_cosine():
    result, _, _ = coupling(*fixture())
    for row in result['pairs'].values():
        assert abs(row['q_dot_over_full_norms']+row['complement_dot_over_full_norms']-
                   row['full_cosine']) < 1e-15


def test_reused_common_descent_certificate_and_update_sign():
    components = np.array([[1., 0.], [0., 1.], [1., 1.], [.5, 1.]])
    result = margin(components, np.linalg.norm(components, axis=1), 1e-8, sha(SOLVER))
    assert result['status'] == 'PASS'
    assert abs(result['gamma']-np.sqrt(.5)) < 1e-12
    assert min(result['gradient_space_products']) >= result['gamma']-1e-12
    assert all(-value < 0 for value in result['gradient_space_products'])
