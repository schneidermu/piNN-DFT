import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_relchem_factorial_gradient_audit as audit


def test_original_penalty_is_outside_the_database_factor_and_matches_its_derivative():
    residual = -2.5
    phi = audit.phi_original(residual)
    assert phi == math.sqrt(1e-20 + residual * residual)
    step = 1e-6
    numerical = (audit.phi_original(residual + step) - audit.phi_original(residual - step)) / (2 * step)
    assert abs(audit.deriv_original(residual) - numerical) < 1e-8
    assert audit.deriv_l1(0.0) == 0.0
    assert audit.deriv_original(0.0) == 0.0
    assert audit.penalty_bits('F010') == (False, True, False)
    assert audit.penalty_bits('F101') == (True, False, True)


def test_factorial_cells_match_mae_jw_and_a_scalar_rescaling():
    residuals = [1.0, -3.0, 2.0]
    databases = ['DBH76', 'DBH76', 'NCCE31']
    scales = {'DBH76': 2.0, 'NCCE31': 4.0}
    raw = np.array([0.5, 0.5, 2.0])
    legacy = audit.mean_one_weights(raw)
    assert abs(legacy.mean() - 1.0) < 1e-12
    mae = audit.diagnostic_scalar(residuals, scales, databases, legacy, 'F000')
    assert abs(mae - (1 + 3 + 2) / 3) < 1e-12
    styled = audit.diagnostic_scalar(residuals, scales, databases, legacy, 'F010')
    assert abs(styled - (1 / 2 + 3 / 2 + 2 / 4) / 3) < 1e-12
    produced = audit.production_scalar(residuals, raw)
    scaled = raw.mean() * audit.diagnostic_scalar(residuals, scales, databases, legacy, 'F101')
    assert abs(produced - scaled) < 1e-12
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    chain = audit.production_coefficients(residuals, raw) @ jacobian
    rebuilt = raw.mean() * (audit.reaction_coefficients(residuals, scales, databases, legacy, 'F101') @ jacobian)
    assert np.allclose(chain, rebuilt)
    total = audit.reaction_coefficients(residuals, scales, databases, legacy, 'F000') @ jacobian
    parts = [
        audit.reaction_coefficients(residuals, scales, databases, legacy, 'F000')[index] * jacobian[index]
        for index in range(3)
    ]
    assert np.allclose(total, np.sum(parts, axis=0))


def test_file_does_not_step_or_recompute_clean28():
    source = Path(audit.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'step' not in calls
    assert 'adamw_step' not in source
    assert '56.84' not in source
    assert len(audit.EDGES) == 12
    assert len(audit.VARIANTS) == 8
