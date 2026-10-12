import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_f010_uniform_normalized_chemistry as f010
import lap_f011_normalized_chemistry as arm
import lap_relchem_factorial_gradient_audit as fact


def test_f010_drops_the_legacy_factor_and_keeps_the_scale_outside_the_square_root():
    residual = 1e-12
    scale = 4.0
    factor = 10.0
    coefficient = 3.0
    penalty = factor * math.sqrt(residual * residual + 1e-20)
    produced = penalty * f010.f010_singleton_scale(coefficient, scale, factor)
    assert abs(produced - coefficient * math.sqrt(residual * residual + 1e-20) / scale) < 1e-18
    f011_value = penalty * arm.chemistry_scale(coefficient, scale)
    assert produced != f011_value
    divided_inside = coefficient * math.sqrt((residual / scale) ** 2 + 1e-20)
    assert produced != divided_inside


def test_f010_coefficients_match_factorial_f110_and_not_f011():
    residuals = [2.0, -3.0]
    raw = np.array([0.5, 10.0])
    scales = {'DBH76': 4.0, 'NCCE31': 2.0}
    databases = ['DBH76', 'NCCE31']
    produced = f010.f010_raw_coefficients(residuals, scales, databases)
    expected = fact.reaction_coefficients(residuals, scales, databases, np.ones(2), 'F110')
    legacy = arm.f011_coefficients(residuals, raw, scales, databases)
    assert np.allclose(produced, expected)
    assert not np.allclose(produced, legacy)
    scalar = f010.f010_raw_scalar(residuals, scales, databases)
    manual = sum(fact.phi_original(residual) / scales[database] for residual, database in zip(residuals, databases, strict=True)) / 2
    assert abs(scalar - manual) < 1e-12


def test_training_does_not_read_clean28_before_the_arm_finishes():
    source = Path(f010.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    assert 'validation' not in ast.get_source_segment(source, train)
    assert 'diet30' not in source
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    segment = ast.get_source_segment(source, execute)
    assert segment.index('train(') < segment.index('evaluate_clean28(')
    assert f010.NEW_UPDATES == 251
    assert f010.NEW_LR == 2.5e-5
