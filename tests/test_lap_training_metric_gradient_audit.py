import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_training_metric_gradient_audit as audit


def test_training_metrics_use_the_251_denominator_and_reject_a_nonpositive_scale():
    residuals = [1.0] * 250 + [-3.0]
    databases = ['DBH76'] * 100 + ['PA8'] * 151
    references = [2.0] * 100 + [4.0] * 151
    scales, counts = audit.database_reference_scales(references, databases)
    assert counts == {'DBH76': 100, 'PA8': 151}
    assert scales['DBH76'] == 2.0
    assert audit.j251_mae(residuals) == (250 + 3) / 251
    expected = (100 * 1.0 / 2.0 + 150 * 1.0 / 4.0 + 3.0 / 4.0) / 251
    assert audit.j251_wtmad_style(residuals, databases, scales) == expected
    assert audit.absolute_gradient_weight(0.0, scales['DBH76']) == 0.0
    assert audit.absolute_gradient_weight(-1.0, 4.0) == -1.0 / (251 * 4.0)
    assert audit.singleton_objective(3.0, 2.0) == 2.0 * math.sqrt(1e-20 + 9.0)
    try:
        audit.database_reference_scales([0.0] * 251, ['DBH76'] * 251)
    except ValueError:
        pass
    else:
        raise AssertionError('a zero reference scale was accepted')


def test_leave_one_subset_out_does_not_renormalize_and_coefficients_apply_once():
    full = np.array([4.0, 1.0])
    subset = np.array([1.0, -1.0])
    assert np.allclose(audit.leave_one_subset_out(full, subset), np.array([3.0, 2.0]))
    assert not np.allclose(audit.leave_one_subset_out(full, subset), full * 251 / 250)
    metric = np.array([1.0, 0.0, 0.0])
    tasks = {
        'relchem': np.array([1.0, 0.0, 0.0]),
        'ae17': np.array([0.0, 2.0, 0.0]),
        'exc': np.array([0.0, 0.0, 3.0]),
        'op': np.array([-1.0, 0.0, 0.0]),
    }
    lambdas = {'relchem': 0.5, 'ae17': 0.25, 'exc': 0.25, 'op': 2.0}
    parts, combined = audit.directional_parts(metric, tasks, lambdas)
    assert parts['relchem'] == -0.5
    assert parts['op'] == 2.0
    assert combined == sum(parts.values())


def test_manifest_has_251_unique_relchem_reactions_and_the_file_does_not_step():
    evaluation = b8.read_json(ROOT / 'relchem_joint_epoch_evaluation_manifest.json')
    rows = [row for row in evaluation['rows'] if row['task'] == 'relchem']
    assert len(rows) == 251
    assert len({row['identity'] for row in rows}) == 251
    assert all(row['variant'] for row in rows)
    source = Path(audit.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert 'step' not in calls
    assert 'adamw_step' not in source
    assert '56.84' not in source
