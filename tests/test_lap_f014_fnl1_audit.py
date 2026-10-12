import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_f014_fnl1_audit as fnl


def test_b8_contributions_are_the_unit_gradients():
    components = {task: np.array([fnl.S[task], 0.0]) for task in fnl.TASKS}
    gradient, _weights, parts = fnl.fnl1_gradient(components, fnl.B)
    for task in fnl.TASKS:
        assert abs(np.linalg.norm(parts[task]) - 1.0) < 1e-12
    assert abs(np.linalg.norm(gradient) - 4.0) < 1e-12


def test_autograd_matches_the_analytical_product_rule():
    theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    raw = {
        'jw': 0.2 + 0.3 * theta[0],
        'ae17': 4.0 + theta[1],
        'exc': 10.0 - 0.5 * theta[2],
        'op': 0.04 + 0.1 * theta[0],
    }
    loss = 0.5 * sum(raw[task] ** 2 / (fnl.B[task] * fnl.S[task]) for task in fnl.TASKS)
    loss.backward()
    losses = {task: float(raw[task].detach()) for task in fnl.TASKS}
    analytical = np.zeros(3)
    analytical[0] = losses['jw'] / (fnl.B['jw'] * fnl.S['jw']) * 0.3 + losses['op'] / (fnl.B['op'] * fnl.S['op']) * 0.1
    analytical[1] = losses['ae17'] / (fnl.B['ae17'] * fnl.S['ae17'])
    analytical[2] = losses['exc'] / (fnl.B['exc'] * fnl.S['exc']) * -0.5
    assert np.allclose(theta.grad.numpy(), analytical)


def test_reset_adamw_is_sign_preconditioned():
    param = np.array([0.0, 0.0])
    grad = np.array([2.0, -4.0])
    delta = fnl.adamw_reset(param, grad)
    expected = -2.5e-5 * grad / (np.abs(grad) + 1e-8)
    assert np.allclose(delta, expected)
