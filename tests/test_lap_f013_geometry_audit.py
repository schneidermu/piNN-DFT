import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_f013_geometry_audit as audit


def test_identical_gradients_have_unit_margin():
    vector = np.array([3.0, 4.0])
    solved = audit.maxmin([vector / np.linalg.norm(vector)] * 3)
    assert abs(solved['gamma'] - 1.0) < 1e-8
    assert solved['dual_gap'] < 1e-8


def test_opposite_gradients_have_no_common_descent():
    solved = audit.maxmin([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    assert solved['gamma'] < 1e-8


def test_orthogonal_pair_has_the_bisector_margin():
    solved = audit.maxmin([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert abs(solved['gamma'] - np.sqrt(0.5)) < 1e-7
    assert abs(min(solved['margins']) - solved['gamma']) < 1e-7


def test_flat_perturbation_restores_bytes():
    import lap_f013_local_probes as probe
    import torch

    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    before = probe._hash_tensors([parameter])
    clone = parameter.detach().clone()
    probe._apply([parameter], np.array([0.25, -0.5, 0.125, 0.0]))
    assert not torch.equal(parameter, clone)
    with torch.no_grad():
        parameter.copy_(clone)
    assert probe._hash_tensors([parameter]) == before


def test_nearly_opposite_margin_is_small_but_positive():
    epsilon = 1e-3
    second = np.array([-1.0, epsilon])
    second /= np.linalg.norm(second)
    solved = audit.maxmin([np.array([1.0, 0.0]), second])
    assert 0.0 < solved['gamma'] < 0.02
    assert solved['dual_gap'] < 1e-8
