"""Small deterministic tests of the exact diagnostic bridge."""
import torch

from lap_q_exact_parameter_jacobian_diagnostic import (
    classify,
    dense_jacobian,
    finite_response,
    unit,
)


def test_exact_dense_and_directional_bridge():
    zero = torch.zeros(3, dtype=torch.float64)
    matrix = torch.tensor([[1., 2., 3.], [2., -1., 4.]], dtype=torch.float64)
    function = lambda value: torch.sin(matrix @ value)
    _, jacobian = dense_jacobian(function, zero)
    assert torch.equal(jacobian, matrix)
    direction = unit(3, 42, 'cpu')
    a = finite_response(function, zero, direction, 1e-4, jacobian @ direction,
                        1., lambda: 0.)
    b = finite_response(function, zero, direction, 2.5e-5, jacobian @ direction,
                        1., lambda: 0.)
    assert a['PASS'] and b['PASS']
    assert a['relative_L2']/b['relative_L2'] > 10
    assert classify([a, b])['classification'] == 'PASS-LINEAR'


def test_curvature_limited_and_structural_fail_gates():
    failed = {'relative_L2': .5, 'PASS': False, 'eta': 1.}
    passed = {'relative_L2': .02, 'PASS': True, 'eta': .25}
    smaller = {'relative_L2': .001, 'PASS': True, 'eta': .0625}
    assert classify([failed, passed, smaller])['classification'] == 'CURVATURE-LIMITED'
    assert not classify([failed, failed])['qualified']


def test_no_signal_does_not_pass_and_parameter_state_is_unchanged():
    parameter = torch.ones(3, dtype=torch.float64)
    saved = parameter.clone()
    function = lambda delta: parameter.square() + delta * 0
    row = finite_response(function, torch.zeros_like(parameter), unit(3, 42, 'cpu'),
                          1e-4, torch.zeros_like(parameter), float(parameter.norm()), lambda: 0.)
    assert not row['PASS'] and row['relative_L2'] is None
    assert torch.equal(parameter, saved)


def test_control_direction_is_reproducible_and_normalized():
    a = unit(9446, 42, 'cpu')
    assert torch.equal(a, unit(9446, 42, 'cpu'))
    assert abs(float(a.norm())-1) < 1e-15
