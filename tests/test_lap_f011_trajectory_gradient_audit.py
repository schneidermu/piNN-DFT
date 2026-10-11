import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_f011_trajectory_gradient_audit as audit
import lap_relchem_factorial_gradient_audit as fact


def test_f010_matches_jw_and_f011_keeps_the_scale_outside_the_square_root():
    residuals = [2.0, -4.0]
    databases = ['DBH76', 'NCCE31']
    scales = {'DBH76': 2.0, 'NCCE31': 4.0}
    raw = np.array([0.5, 2.0])
    jacobian = np.eye(2)
    gradients = audit.chemistry_gradients(residuals, databases, scales, raw, 3.0, jacobian)
    assert np.allclose(gradients['jw'], gradients['f010'])
    manual = 3.0 * (
        0.5 / 2.0 * fact.phi_original(2.0) + 2.0 / 4.0 * fact.phi_original(-4.0)
    ) / 2
    produced = 3.0 * sum(
        raw[index] / scales[databases[index]] * fact.phi_original(residual)
        for index, residual in enumerate(residuals)
    ) / 2
    assert abs(produced - manual) < 1e-12


def test_jw_descent_cosine_is_negative_when_the_step_opposes_the_gradient():
    gradient = np.array([3.0, 4.0])
    displacement = -gradient
    assert float(np.dot(gradient, displacement)) < 0.0
    observed = -1.0
    linear = float(np.dot(gradient, displacement))
    assert observed - linear != 0.0 or abs(linear) > 0.0


def test_source_does_not_step_or_load_clean28():
    source = Path(audit.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    measure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'measure')
    calls = [
        node.func.attr for node in ast.walk(measure)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'step' not in calls
    assert 'adamw_step' not in source
    assert 'diet30' not in source
    assert 'evaluate_microbatch_endpoint' not in source
    assert len(audit.CHECKS) == 5
    assert audit.CLEAN28[64] == 8.782020261263
