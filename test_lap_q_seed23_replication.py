"""Prospective decision and qualification regressions; no real-data evaluation."""
import numpy as np

from lap_q_objective_coupling_diagnostic import coupling, overlap
from lap_q_objective_coupling_seed23_diagnostic import qualification, replication


def test_literal_replication_boundaries():
    assert replication(1.5, 1.5, .95, .8, [-.9]*3)['classification'] == 'REPLICATED'
    assert replication(1.499, 1.5, .95, .8, [-.9]*3)['classification'] == 'PARTIAL REPLICATION'
    assert replication(1.2, 1.1, .99, .99, [-.99]*3)['classification'] == 'NOT REPLICATED'
    assert replication(2, 2, .99, .99, [-.9, -.9, -.899])['classification'] == 'PARTIAL REPLICATION'


def test_isolated_pass_cannot_qualify_on_rebound():
    rows = [{'eta': 1/4**i, 'PASS': passed, 'relative_L2': error}
            for i, (passed, error) in enumerate([(False, .9), (True, .01), (False, .1)])]
    assert not qualification(rows)['qualified']
    rows[-1].update(PASS=True, relative_L2=.001)
    assert qualification(rows)['classification'] == 'CURVATURE-LIMITED'


def test_sign_orientation_invariant_observables():
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(128, 150))
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    gradients = rng.normal(size=(4, 150))
    a, _, _ = coupling(matrix, vh, singular, gradients)
    b, _, _ = coupling(matrix, -vh, singular, gradients)
    for name in a['objectives']:
        assert a['objectives'][name]['projection'] == b['objectives'][name]['projection']
        assert a['objectives'][name]['induced'] == b['objectives'][name]['induced']
    assert np.isclose(overlap(vh[:8], -vh[:8])['mean_squared_overlap'], 1)
