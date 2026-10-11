import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_f011_normalized_chemistry as arm
import lap_relchem_factorial_gradient_audit as fact


def test_reference_scale_stays_outside_the_penalty_and_c_matches_the_norm():
    residual = 1e-12
    scale = 4.0
    factor = 0.5
    coefficient = 4.0
    penalty = factor * math.sqrt(residual * residual + 1e-20)
    weighted = chemistry_term(penalty, coefficient, scale)
    divided_inside = coefficient * factor * math.sqrt((residual / scale) ** 2 + 1e-20)
    assert weighted == coefficient / scale * penalty
    assert weighted != divided_inside
    old = np.array([3.0, 4.0])
    new = np.array([1.0, 0.0])
    calibrated = arm.calibrate_norm(old, new)
    assert abs(calibrated['scaled_norm'] - 5.0) < 1e-12
    assert abs(calibrated['c'] - 5.0) < 1e-12
    assert calibrated['cosine_old'] == 0.6


def chemistry_term(penalty, coefficient, scale):
    return penalty * arm.chemistry_scale(coefficient, scale)


def test_f011_coefficients_match_the_chain_rule_and_not_a_rescaled_residual():
    residuals = [2.0, -3.0]
    raw = np.array([0.5, 2.0])
    scales = {'DBH76': 4.0, 'NCCE31': 2.0}
    databases = ['DBH76', 'NCCE31']
    coefficients = arm.f011_coefficients(residuals, raw, scales, databases)
    expected = np.array([
        0.5 * fact.deriv_original(2.0) / (4.0 * 2),
        2.0 * fact.deriv_original(-3.0) / (2.0 * 2),
    ])
    assert np.allclose(coefficients, expected)
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0]])
    gradient = coefficients @ jacobian
    rebuilt = np.zeros(2)
    for index in range(2):
        rebuilt += coefficients[index] * jacobian[index]
    assert np.allclose(gradient, rebuilt)


def test_schedule_and_source_do_not_step_during_calibration():
    source = Path(arm.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    calibration = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'cpu_gates')
    calls = [
        node.func.attr for node in ast.walk(calibration)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'step' not in calls
    assert 'adamw_step' not in ast.get_source_segment(source, calibration)
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    assert b8.presentation_counts(batches) == {row['relchem']['identity']: 8 for row in rows}
    assert b8.sha256_file(arm.PARENT_MANIFEST).startswith('b249964d')
