import numpy as np
import pytest

from tools import lap_relchem_gradient_conflict_audit as audit


def vector(value, scale=1.0):
    array = np.full(audit.PARAMETER_COUNT, value, dtype=np.float64)
    array[0] = scale
    return array


def test_database_mean_and_contribution_use_one_total():
    rows = [vector(0.0, 2.0), vector(0.0, 4.0)]
    mean, contribution = audit.database_terms(rows)
    assert mean[0] == pytest.approx(3.0)
    assert contribution[0] == pytest.approx(3.0 * 2 / 251)
    assert np.allclose(contribution, mean * (2 / 251))


def test_zero_gradient_cosine_is_undefined():
    assert audit.cosine(vector(0.0, 0.0), vector(0.0, 1.0)) is None


def test_opposite_vectors_cancel():
    left = vector(0.0, 1.0)
    right = -left
    assert audit.cosine(left, right) == pytest.approx(-1.0)
    assert audit.cancellation_ratio(left, right) == pytest.approx(1.0)
    assert audit.predicted_change(left, right) == pytest.approx(-float(np.dot(left, right)))


def test_system_panel_is_fixed_before_gradients():
    rows = [{'id': f'mrks_{index:02d}', 'n_grid': 1000 - index} for index in range(90)]
    selected = audit.select_systems(rows)
    ordered = sorted(rows, key=lambda row: (row['n_grid'], row['id']))
    assert [row['id'] for row in selected] == [ordered[index]['id'] for index in (11, 33, 56, 78)]


def test_competing_direction_uses_recorded_coefficients_once():
    ae17, exc, operator = vector(0.0, 1.0), vector(0.0, 2.0), vector(0.0, 3.0)
    direction = audit.competing_direction(ae17, exc, operator)
    assert direction[0] == pytest.approx(
        audit.LAMBDAS['ae17'] + 2 * audit.LAMBDAS['exc'] + 3 * audit.LAMBDAS['op']
    )


def test_vector_length_is_rejected():
    with pytest.raises(ValueError):
        audit.as_vector({'weight': np.ones(3)}, ('weight',))


def test_source_does_not_call_training_or_optimizer_steps():
    audit.forbid_training_calls(audit.Path(__file__).resolve().parents[1].joinpath(
        'tools/lap_relchem_gradient_conflict_audit.py'
    ).read_text(encoding='utf-8'))


def test_sign_mismatch_supports_path_hypothesis_locally():
    history = {
        database: {'sign_actual': 1, 'sign_s0': -1, 'sign_s70': -1}
        for database in audit.FOCUS_DATABASES
    }
    states = {}
    for _state in ('s0', 's70'):
        states[_state] = {
            'terms': {
                database: {'cos_other': 0.2, 'predicted_delta_under_minus_other': -1.0}
                for database in audit.FOCUS_DATABASES
            },
            'cosine_matrix': {
                'ABDE4': {'pTC13': 0.4, 'PA8': 0.4},
                'pTC13': {'PA8': 0.4},
            },
        }
    labels = audit.adjudicate(history, states)
    assert labels['H1'] == 'WEAKENED'
    assert labels['H3'] == 'SUPPORTED LOCALLY'
    assert labels['H2_remaining_226'] == 'NOT VERIFIED'
    assert labels['next_experiment'] == 'C'
