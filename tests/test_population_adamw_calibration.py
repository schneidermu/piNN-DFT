from types import SimpleNamespace

import numpy as np
import torch

from tools.calibrate_population_adamw import bootstrap, draws, geometry


def test_population_draws_are_identity_uniform_not_database_balanced():
    rows = [{'id': f'{task}_{i}', 'task': task, 'database': 'rare' if i == 0 else 'common',
             'reaction_id': i, 'variants': {f'v{j}': {} for j in range(8)}}
            for task, count in (('relchem', 251), ('ae17', 17)) for i in range(count)]
    bundle = SimpleNamespace(reactions={r['id']: r for r in rows},
                             systems={f's{i}': {} for i in range(90)})
    manifest = draws(bundle, 5000, 930337)
    assert manifest[:48] == draws(bundle, 48, 930337)
    assert manifest[:48] != draws(bundle, 48, 930338)
    assert len({x['relchem']['identity'] for x in manifest}) == 251
    assert len({x['mrks_id'] for x in manifest}) == 90
    rare_fraction = np.mean([x['relchem']['database'] == 'rare' for x in manifest])
    assert 0 < rare_fraction < .02  # Not 50% equal-database allocation.
    assert all(x['relchem']['weight'] == x['ae17']['weight'] == 1 for x in manifest)


def test_bootstrap_is_deterministic_and_inverse_median_rule_unchanged():
    values = np.ones((48, 4)) * np.array([1., 2., 4., 8.])
    result = bootstrap(values, repetitions=100)
    assert result == bootstrap(values, repetitions=100)
    for j, task in enumerate(('relchem', 'ae17', 'exc', 'op')):
        assert result[task]['median_95pct_interval'] == [2.**j, 2.**j]
        assert result[task]['coefficient_95pct_interval'] == [1 / (4 * 2.**j)] * 2


def test_direction_geometry_retains_signed_contributions_and_raw_dots():
    matrix = torch.tensor([[1., 0.], [-.8, .4], [.5, 1.], [.2, .5]], dtype=torch.float64)
    coefficients = dict(zip(('relchem', 'ae17', 'exc', 'op'), [1., .5, .25, .25]))
    joint, details = geometry(matrix, coefficients)
    torch.testing.assert_close(joint, sum(matrix[j] * x for j, x in enumerate(coefficients.values())))
    np.testing.assert_allclose(details['task_dots_unit_weighted_direction'], (matrix @ (joint / joint.norm())).numpy(), atol=1e-15)
    assert abs(sum(details['signed_fraction_of_joint_squared_norm']) - 1) < 1e-14
    assert details['signed_fraction_of_joint_squared_norm'][1] < 0
