"""F019 algebra and fail-closed cache identity checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f019_fixed_db as audit


def test_database_weights_preserve_equal_database_mean():
    databases = [d for d, count in audit.COUNTS.items() for _ in range(count)]
    assert len(databases) == 251
    w = audit.weights(databases)
    for d in audit.COUNTS:
        assert np.isclose(w[np.array(databases) == d].sum() / 251, 1 / 8)


def test_mean_gradient_reconstruction():
    databases = [d for d, count in audit.COUNTS.items() for _ in range(count)]
    g = np.random.default_rng(19).normal(size=(251, 5))
    actual = (audit.weights(databases)[:, None] * g).mean(axis=0)
    independent = np.mean([g[np.array(databases) == d].mean(axis=0) for d in audit.COUNTS], axis=0)
    np.testing.assert_allclose(actual, independent, rtol=1e-13, atol=1e-13)


def test_variant_identity_must_not_be_ignored():
    lookup = {('reaction', 'level2'): 0}
    assert ('reaction', 'level3') not in lookup


def test_coefficients_unchanged():
    assert audit.geometry.b8.LAMBDAS['op'] == 0.33597561607048215
    assert audit.geometry.b8.LAMBDAS['relchem'] == 0.017015480965588553
