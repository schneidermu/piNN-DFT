import ast
import sys
from pathlib import Path

import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_wtmad_gradient_alignment as diagnostic


def test_clean28_is_the_mean_absolute_metric_and_zero_has_subgradient_zero():
    weighted = [1.0, 2.0, 3.0] + [0.0] * 25
    assert diagnostic.clean28_scalar(weighted) == sum(weighted) / 28
    assert diagnostic.subgradient_sign(0.0) == 0.0
    assert diagnostic.subgradient_sign(1.5) == 1.0
    assert diagnostic.subgradient_sign(-0.25) == -1.0
    assert diagnostic.reaction_gradient_scale(4.0, 0.0) == 0.0
    assert diagnostic.reaction_gradient_scale(4.0, 2.0) == 4.0 / 28
    assert diagnostic.reaction_gradient_scale(4.0, -2.0) == -4.0 / 28
    try:
        diagnostic.clean28_scalar([1.0])
    except ValueError:
        pass
    else:
        raise AssertionError('a short Clean28 vector was accepted')


def test_sie_exclusion_keeps_the_original_denominator():
    full = np.array([4.0, 1.0])
    sie = np.array([1.0, -1.0])
    remainder = full - sie
    assert np.allclose(remainder, np.array([3.0, 2.0]))
    assert not np.allclose(remainder, full * 28 / 27)


def test_geometry_helpers_and_adamw_formula_do_not_mutate_moments():
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 2.0])
    assert diagnostic.cosine(left, right) == 0.0
    assert diagnostic.cancellation_ratio((left, -left)) == 1.0
    assert diagnostic.predicted_change(left, left) == -1.0
    assert diagnostic.normalized_change(np.array([3.0, 0.0]), np.array([2.0, 0.0])) == -3.0
    param = np.array([1.0])
    grad = np.array([0.5])
    moment = np.array([0.1])
    second = np.array([0.2])
    moment_before = moment.copy()
    step = 3
    beta1, beta2 = 0.9, 0.999
    updated = beta1 * moment + (1 - beta1) * grad
    second_updated = beta2 * second + (1 - beta2) * grad**2
    corrected = updated / (1 - beta1**step)
    second_hat = second_updated / (1 - beta2**step)
    expected = -0.25 * (corrected / (np.sqrt(second_hat) + 1e-8) + 0.01 * param)
    actual = diagnostic.hypothetical_adamw_displacement(
        param, grad, moment, second, step - 1, 0.25, (beta1, beta2), 1e-8, 0.01)
    assert np.allclose(actual, expected)
    assert np.array_equal(moment, moment_before)


def test_parameter_blocks_partition_normalization_away_from_the_networks():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_feature_extractor = nn.Sequential(nn.Linear(2, 2, bias=False), nn.LayerNorm(2))
            self.x_output_layer = nn.Linear(2, 1, bias=False)
            self.c_input_layers = nn.Sequential(nn.Linear(2, 2, bias=False), nn.LayerNorm(2))
            self.extra = nn.Linear(1, 1)

    assignment = diagnostic.parameter_blocks(Toy())
    assert assignment['x_feature_extractor.0.weight'] == 'exchange'
    assert assignment['x_output_layer.weight'] == 'exchange'
    assert assignment['x_feature_extractor.1.weight'] == 'normalization'
    assert assignment['x_feature_extractor.1.bias'] == 'normalization'
    assert assignment['c_input_layers.0.weight'] == 'correlation'
    assert assignment['c_input_layers.1.weight'] == 'normalization'
    assert assignment['extra.weight'] == 'remaining'
    model = Toy()
    names = [name for name, _parameter in model.named_parameters()]
    numels = [parameter.numel() for _name, parameter in model.named_parameters()]
    indices = diagnostic.block_indices(names, numels, assignment)
    covered = np.concatenate([indices[block] for block in indices if len(indices[block])])
    assert sorted(covered.tolist()) == list(range(sum(numels)))


def test_frozen_clean28_receipts_are_means_of_weighted_absolute_errors():
    for spec in diagnostic.CHECKPOINTS:
        payload = b8.read_json(spec['validation'])
        rows = payload['reaction_rows']
        assert len(rows) == 28
        assert payload['clean28'] == spec['expected_clean28']
        assert abs(sum(row['weighted_absolute_error'] for row in rows) / 28 - payload['clean28']) < 1e-12


def test_diagnostic_file_does_not_call_optimizer_step():
    source = Path(diagnostic.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'step' not in calls
    assert 'adamw_step' not in source
