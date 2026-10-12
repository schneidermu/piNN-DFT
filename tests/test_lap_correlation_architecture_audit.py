import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import lap_correlation_architecture_audit as audit


def test_smooth_activation_matches_the_preregistered_formula():
    z = np.array([1.0, 0.5, 1.5, 3.0])
    produced = audit.smooth_gc(z, 4.0)
    manual = 1.0 - (2.0 / 4.0) * np.log(np.cosh(4.0 * (z - 1.0) / 2.0))
    assert np.allclose(produced, manual)
    assert produced[0] == 1.0
    assert np.all(produced <= 1.0 + 1e-12)
    slope = audit.smooth_slope(np.array([1.0]), 4.0)
    assert abs(float(slope[0])) < 1e-15


def test_positive_correlation_is_case_a_only_when_gc_crosses_a_unit_threshold():
    assert audit.classify_point(-0.1, 0.05, 2.2, 0.01, True) == 'A'
    assert audit.classify_point(-0.01, 0.02, 1.0, 0.01, True) == 'B'
    assert audit.classify_point(float('nan'), 0.1, 1.2, 0.1, False) == 'C'
    assert abs(float(audit.invert_shifted_elu(np.array([2.0]))[0]) - 2.0) < 1e-12
    assert abs(float(audit.original_slope(np.array([1.0]))[0]) - 1.0) < 1e-12


def test_source_does_not_step_or_read_clean28():
    root = Path(audit.__file__).resolve().parent
    for name in (
        'lap_correlation_architecture_audit.py',
        'lap_correlation_architecture_followup.py',
        'lap_correlation_architecture_render.py',
    ):
        source = (root / name).read_text(encoding='utf-8')
        assert 'adamw_step' not in source
        assert 'optimizer.step' not in source
        assert 'validation(' not in source
        assert 'diet30' not in source
    assert math.isclose(audit.SIGN_THRESHOLD, 1e-8)
