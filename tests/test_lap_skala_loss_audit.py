"""CPU checks for the frozen Skala loss audit. No GPU and no optimizer."""
import math
from pathlib import Path

import numpy as np
import pytest

from tools import lap_skala_loss_audit as audit

K = audit.K


def test_skala_equation_scalar_parity():
    residual_h = 0.02
    reference_h = -0.5
    expected = (residual_h ** 2) / (1e-4 + abs(reference_h))
    assert audit.loss_b(residual_h * K, reference_h * K) == pytest.approx(expected)
    assert audit.loss_b(residual_h * K, reference_h * K) == pytest.approx(0.0004 / 0.5001)


def test_hartree_kcal_conversion():
    assert K * 1e-4 == pytest.approx(0.06275095)
    assert audit.loss_b(K, 0.0) == pytest.approx(1.0 / 1e-4)
    assert audit.loss_b(-K, 0.0) == pytest.approx(1.0 / 1e-4)
    assert audit.denominator(0.06275095) == pytest.approx(2e-4)


def test_fchem_singleton_matches_batch_fchem():
    import torch
    factor = audit.fchem_factors()['NCCE31']
    from optuna_joint import FCHEM_DB_WEIGHTS, FREQ_WEIGHTS, MEAN_WEIGHT, batch_fchem
    prediction = torch.tensor([4.0], dtype=torch.float64)
    reference = torch.tensor([1.5], dtype=torch.float64)
    produced = float(batch_fchem(['NCCE31'], prediction, reference))
    assert produced == pytest.approx(factor * math.sqrt(2.5 ** 2 + 1e-20))
    assert factor == pytest.approx(FCHEM_DB_WEIGHTS['NCCE31'] * FREQ_WEIGHTS['NCCE31'] / MEAN_WEIGHT)
    products = [FCHEM_DB_WEIGHTS[db] * FREQ_WEIGHTS[db] for db in FCHEM_DB_WEIGHTS]
    assert 'AE17' in FCHEM_DB_WEIGHTS
    assert MEAN_WEIGHT == pytest.approx(sum(products) / len(products))
    assert audit.fchem_factors()['NCCE31'] > audit.fchem_factors()['MGAE109']


def test_reference_denominator_cases():
    assert audit.denominator(0.0) == pytest.approx(1e-4)
    assert audit.denominator(1e-6) == pytest.approx(1e-4 + 1e-6 / K)
    assert audit.denominator(-1e-6) == audit.denominator(1e-6)
    assert audit.denominator(K) == pytest.approx(1.0 + 1e-4)
    assert audit.denominator(-1000.0) == audit.denominator(1000.0)
    assert audit.loss_b(0.0, 0.0) == 0.0
    assert math.isfinite(audit.loss_b(500.0, 0.0))
    assert math.isfinite(audit.loss_b(-500.0, -1e-8))
    with pytest.raises(ValueError):
        audit.denominator(1.0, floor_h=0.0)


def test_huber_continuity_and_quadratic_match():
    residual = 1.0
    reference = 20.0
    assert abs(audit.normalized_residual(residual, reference)) < audit.DELTA
    assert audit.loss_c(residual, reference) == pytest.approx(audit.loss_b(residual, reference))
    width = audit.denominator(reference)
    boundary = audit.DELTA * math.sqrt(width) * K
    assert audit.loss_c(boundary, reference) == pytest.approx(audit.DELTA ** 2)
    assert audit.loss_c(-boundary, reference) == pytest.approx(audit.DELTA ** 2)
    assert audit.loss_c(boundary, reference) == pytest.approx(audit.loss_b(boundary, reference))
    large = 80.0
    assert abs(audit.normalized_residual(large, 5.0)) > audit.DELTA
    assert audit.loss_c(large, 5.0) < audit.loss_b(large, 5.0)
    assert audit.huber_region(residual, reference) == 'quadratic'
    assert audit.huber_region(large, 5.0) == 'linear'


def test_huber_derivative_continuity():
    reference = 12.0
    width = audit.denominator(reference)
    boundary = audit.DELTA * math.sqrt(width) * K
    quadratic = audit.d_loss_c_de_h(boundary, reference)
    linear = 2.0 * audit.DELTA / math.sqrt(width)
    assert quadratic == pytest.approx(linear)
    assert audit.d_loss_c_de_h(-boundary, reference) == pytest.approx(-linear)
    assert audit.d_loss_c_de_h(boundary * 0.5, reference) == pytest.approx(
        audit.d_loss_b_de_h(boundary * 0.5, reference)
    )


def finite_difference(function, residual, reference):
    step = 1e-6
    return (function(residual + step, reference) - function(residual - step, reference)) / (2.0 * step)


def test_analytic_derivatives_match_finite_differences():
    cases = ((1.5, 10.0), (-2.0, 0.0), (0.25, -0.01), (40.0, 5.0), (-40.0, 80.0), (8.0, 1e-4))
    factor = 1.3
    for residual, reference in cases:
        assert finite_difference(lambda value, ref, factor=factor: audit.loss_a(value, factor), residual, reference) == pytest.approx(
            audit.d_loss_a(residual, factor), rel=1e-6, abs=1e-8,
        )
        assert finite_difference(audit.loss_b, residual, reference) == pytest.approx(
            audit.d_loss_de_kcal(audit.d_loss_b_de_h(residual, reference)), rel=1e-6, abs=1e-8,
        )
        assert finite_difference(audit.loss_c, residual, reference) == pytest.approx(
            audit.d_loss_de_kcal(audit.d_loss_c_de_h(residual, reference)), rel=1e-5, abs=1e-8,
        )


def test_parameter_gradient_rescaling():
    torch = pytest.importorskip('torch')
    weight = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64))
    residual = 3.0 * weight
    reference = torch.tensor(4.0, dtype=torch.float64)
    factor = 1.4
    old = factor * torch.sqrt(residual * residual + 1e-20)
    new = (residual / K) ** 2 / (1e-4 + (reference / K).abs())
    old_gradient = torch.autograd.grad(old, weight, retain_graph=True)[0].detach().numpy()
    new_gradient = torch.autograd.grad(new, weight)[0].detach().numpy()
    scaled = audit.rescale_gradient(
        old_gradient,
        audit.d_loss_de_kcal(audit.d_loss_b_de_h(float(residual.detach()), float(reference))),
        audit.d_loss_a(float(residual.detach()), factor),
    )
    assert np.allclose(scaled, new_gradient, rtol=1e-10, atol=1e-12)
    leaf = torch.nn.Parameter(torch.tensor(30.0, dtype=torch.float64))
    wide = audit.denominator(5.0)
    assert abs(float(leaf.detach()) / K / math.sqrt(wide)) > audit.DELTA
    base = factor * torch.sqrt(leaf * leaf + 1e-20)
    huber_input = leaf / K / math.sqrt(torch.tensor(wide, dtype=torch.float64))
    huber = 2.0 * audit.DELTA * huber_input.abs() - audit.DELTA ** 2
    base_gradient = torch.autograd.grad(base, leaf, retain_graph=True)[0].detach().numpy()
    huber_gradient = torch.autograd.grad(huber, leaf)[0].detach().numpy()
    scaled_huber = audit.rescale_gradient(
        base_gradient,
        audit.d_loss_de_kcal(audit.d_loss_c_de_h(float(leaf.detach()), 5.0)),
        audit.d_loss_a(float(leaf.detach()), factor),
    )
    assert np.allclose(scaled_huber, huber_gradient, rtol=1e-10, atol=1e-12)


def test_near_zero_residual_is_undefined():
    assert audit.d_loss_a(0.0, 2.0) == 0.0
    assert audit.amplification(audit.d_loss_de_kcal(audit.d_loss_b_de_h(0.0, 5.0)), 0.0) is None
    assert audit.rescale_gradient(np.ones(4), 1.0, 1e-12) is None
    assert audit.rescale_gradient(np.array([2.0, -4.0]), 2.0, 4.0).tolist() == pytest.approx([1.0, -2.0])


def test_loss_b_does_not_take_database_factor():
    assert 'factor' not in audit.loss_b.__code__.co_varnames
    assert audit.loss_b(1.25, -3.5) == audit.loss_b(1.25, -3.5)
    weighted = audit.fchem_factors()['NCCE31'] * audit.loss_b(1.25, -3.5)
    assert weighted != pytest.approx(audit.loss_b(1.25, -3.5))
    assert audit.loss_a(1.25, audit.fchem_factors()['ABDE4']) != pytest.approx(
        audit.loss_a(1.25, audit.fchem_factors()['MGAE109'])
    )


def _population():
    rows = []
    for database, count in audit.EXPECTED_COUNTS.items():
        for index in range(count):
            rows.append({
                'identity': f'{database}-{index:03d}',
                'database': database,
                'variant': 'level3' if index % 2 == 0 else 'level2',
                'task': 'relchem',
                'loss': float(index),
            })
    return rows


def test_frozen_panel_selection_ignores_loss_and_keeps_variants():
    rows = _population()
    panel, direct = audit.select_panel(rows)
    again, direct_again = audit.select_panel(rows)
    assert [row['identity'] for row in panel] == [row['identity'] for row in again]
    assert [row['identity'] for row in direct] == [row['identity'] for row in direct_again]
    for row in rows:
        row['loss'] = -row['loss']
    flipped, _ = audit.select_panel(rows)
    assert [row['identity'] for row in flipped] == [row['identity'] for row in panel]
    assert len(panel) == 40
    assert {row['database'] for row in panel[:25]} == set(audit.FOCUS_DATABASES)
    counts = {database: 0 for database in audit.CONTROL_DATABASES}
    for row in panel[25:]:
        counts[row['database']] += 1
        assert row['variant'] in {'level2', 'level3'}
    assert counts == {database: 3 for database in audit.CONTROL_DATABASES}
    assert [row['database'] for row in direct] == list(audit.DIRECT_DATABASES)
    assert all(row['identity'] in {item['identity'] for item in panel} for row in direct)
    with pytest.raises(ValueError):
        audit.select_panel(rows + [dict(rows[0])])


def test_checkpoint_snapshot_detects_mutation(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'frozen')
    before = audit.snapshot_files([path])
    audit.loss_b(1.0, 2.0)
    audit.assert_snapshots(before)
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError):
        audit.assert_snapshots(before)


def test_source_has_no_optimizer_step():
    source = Path(audit.__file__).read_text(encoding='utf-8')
    audit.forbid_training_calls(source)
    with pytest.raises(ValueError):
        audit.forbid_training_calls('def f(optimizer):\n    optimizer.step()\n')
    assert '0 optimizer' not in source or 'optimizer_steps' in source


def _state(share_b, share_c, top_b=0.1, top_c=0.1):
    databases = {
        database: {'derivative_share_b': share_b, 'derivative_share_c': share_c}
        for database in audit.REL_DATABASES
    }
    return {
        'parity_failed': False,
        'summary': {
            'databases': databases,
            'concentration': {'abs_dL_b': {'5': 0.2}, 'abs_dL_c': {'5': 0.2}},
        },
        'gradients': {'top1_b': top_b, 'top1_c': top_c},
    }


def test_decision_rejects_database_imbalance_and_failed_parity():
    checks = [{'status': 'pass'} for _ in range(6)]
    balanced = {'s0': _state(0.125, 0.125), 's70': _state(0.125, 0.125)}
    assert audit.decide(balanced, checks)[0] == 'GO-SKALA'

    def dominated(pattern):
        state = _state(0.0, 0.0)
        for database, value in pattern.items():
            state['summary']['databases'][database]['derivative_share_b'] = value
            state['summary']['databases'][database]['derivative_share_c'] = value
        return state

    pattern = {database: 0.01 for database in audit.REL_DATABASES}
    pattern['DBH76'] = 0.93
    assert audit.decide({'s0': dominated(pattern), 's70': dominated(pattern)}, checks)[0] == 'NO-GO'
    assert audit.decide(balanced, [{'status': 'fail'}])[0] == 'PARTIAL'


def test_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        audit.loss_b(float('nan'), 1.0)
    with pytest.raises(ValueError):
        audit.loss_a(float('inf'), 1.0)
    with pytest.raises(ValueError):
        audit.loss_c(1.0, float('nan'))
    with pytest.raises(ValueError):
        audit.rescale_gradient(np.array([1.0, np.nan]), 1.0, 2.0)
