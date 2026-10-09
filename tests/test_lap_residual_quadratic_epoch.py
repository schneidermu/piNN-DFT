"""CPU checks for the residual-aware chemistry multiplier. No GPU and no optimizer step."""
import ast
import math
from pathlib import Path

import pytest
import torch

from tools import lap_residual_quadratic_epoch as rq

EPS = rq.EPS


def test_original_and_new_losses_match_closed_forms():
    residual, factor, scale = 3.5, 1.25, 6.0
    assert rq.loss_old(residual, factor) == pytest.approx(factor * math.sqrt(residual ** 2 + EPS))
    assert rq.loss_new(residual, factor, scale) == pytest.approx(
        0.5 * factor * (math.sqrt(residual ** 2 + EPS) + residual ** 2 / (2.0 * scale)))
    assert rq.surrogate_loss(rq.loss_old(residual, factor), factor, scale) == pytest.approx(
        rq.loss_new(residual, factor, scale))


def test_gradient_factor_matches_derivative_ratio():
    for residual in (1e-8, 1e-4, 0.2, 2.0, 6.0, 10.0, 100.0, -4.0, -100.0):
        for factor in (0.017, 1.0, 2.245):
            loss = rq.loss_old(residual, factor)
            expected = 0.5 * (1.0 + math.sqrt(residual * residual + EPS) / 6.0)
            assert rq.gradient_factor(loss, factor, 6.0) == pytest.approx(expected)


def test_finite_differences_match_analytical_factor():
    scale = 5.5
    for residual in (1e-6, 0.1, 2.0, 10.0, 100.0, -4.0, -80.0):
        for factor in (0.2, 2.897):
            step = 1e-6
            old_derivative = (rq.loss_old(residual + step, factor) - rq.loss_old(residual - step, factor)) / (2 * step)
            new_derivative = (rq.loss_new(residual + step, factor, scale) - rq.loss_new(residual - step, factor, scale)) / (2 * step)
            predicted = rq.gradient_factor(rq.loss_old(residual, factor), factor, scale) * old_derivative
            assert new_derivative == pytest.approx(predicted, rel=1e-6, abs=1e-8)


def test_signed_zero_and_large_residuals():
    assert rq.loss_old(4.0, 1.5) == pytest.approx(rq.loss_old(-4.0, 1.5))
    assert rq.loss_new(4.0, 1.5, 6.0) == pytest.approx(rq.loss_new(-4.0, 1.5, 6.0))
    assert rq.absolute_residual(rq.loss_old(0.0, 1.5), 1.5) == pytest.approx(0.0)
    assert rq.gradient_factor(rq.loss_old(0.0, 1.5), 1.5, 6.0) == pytest.approx(0.5 * (1.0 + math.sqrt(EPS) / 6.0))
    large = rq.gradient_factor(rq.loss_old(1e6, 0.5), 0.5, 6.0)
    assert large == pytest.approx(0.5 * (1.0 + 1e6 / 6.0), rel=1e-9)
    assert rq.absolute_residual(rq.loss_old(-8.0, 0.4), 0.4) == pytest.approx(8.0)


def test_database_factor_is_applied_once():
    scale = 4.0
    assert rq.loss_new(5.0, 2.0, scale) == pytest.approx(2.0 * rq.loss_new(5.0, 1.0, scale))
    assert rq.loss_new(5.0, 2.0, scale) != pytest.approx(4.0 * rq.loss_new(5.0, 1.0, scale))
    assert rq.gradient_factor(rq.loss_old(5.0, 0.1), 0.1, scale) == pytest.approx(
        rq.gradient_factor(rq.loss_old(5.0, 10.0), 10.0, scale))


def test_singleton_matches_batch_fchem_without_a_second_factor():
    rq.database_factors()
    from optuna_joint import batch_fchem
    factor = rq.database_factors()['ABDE4']
    produced = float(batch_fchem(['ABDE4'], torch.tensor([3.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64)))
    assert produced == pytest.approx(rq.loss_old(2.0, factor))
    assert 'AE17' in rq.database_factors()


def test_synthetic_autograd_matches_every_coordinate():
    parameter = torch.nn.Parameter(torch.tensor([0.7, -1.2, 0.25], dtype=torch.float64))
    weights = torch.tensor([1.5, -0.25, 0.8], dtype=torch.float64)
    factor, scale = 1.7, 4.5
    residual = (parameter * weights).sum()
    loss_a = factor * torch.sqrt(residual * residual + EPS)
    loss_direct = 0.5 * (loss_a + (loss_a.square() / factor - factor * EPS) / (2.0 * scale))
    old, = torch.autograd.grad(loss_a, parameter, retain_graph=True)
    new, = torch.autograd.grad(loss_direct, parameter)
    multiplier = rq.gradient_factor(float(loss_a.detach()), factor, scale)
    reference = float(new.norm())
    assert reference > 0
    assert float((new - multiplier * old).norm()) / reference <= 1e-10
    assert float(loss_direct.detach()) == pytest.approx(rq.surrogate_loss(float(loss_a.detach()), factor, scale))


def test_zero_residual_synthetic_gradient_stays_zero():
    residual = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    loss_a = 1.2 * torch.sqrt(residual * residual + EPS)
    loss_direct = 0.5 * (loss_a + (loss_a.square() / 1.2 - 1.2 * EPS) / 8.0)
    old, = torch.autograd.grad(loss_a, residual, retain_graph=True)
    new, = torch.autograd.grad(loss_direct, residual)
    assert float(old) == pytest.approx(0.0, abs=1e-12)
    assert float(new) == pytest.approx(0.0, abs=1e-12)


def test_other_tasks_and_dtype_are_unchanged_by_one_multiplier():
    raw = {task: {'weight': torch.ones(4, dtype=torch.float64)} for task in rq.TASKS}
    scaled = rq.scale_relchem(raw, 1.5)
    for task in ('ae17', 'exc', 'op'):
        assert scaled[task]['weight'] is raw[task]['weight']
    assert scaled['relchem']['weight'].dtype == torch.float64
    assert torch.equal(scaled['relchem']['weight'], raw['relchem']['weight'] * 1.5)
    assert scaled['relchem']['weight'] is not raw['relchem']['weight']


def test_architecture_gate_counts_parameter_coordinates():
    source = Path(rq.__file__).read_text(encoding='utf-8')
    assert 'coordinates != 9446' in source
    assert 'len(parameters) != 9446' not in source


def test_wrapper_casts_only_through_existing_adamw_boundary():
    source = Path(rq.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert 'backward' not in attributes
    assert 'step' not in attributes
    assert source.count('.to(') == 0
    fixed = (rq.REPO / 'train_models' / 'lap_fixed_adamw.py').read_text(encoding='utf-8')
    assert fixed.count('.to(p.dtype)') == 1


def test_median_uses_the_middle_of_251_residuals():
    assert rq.median_scale([float(value) for value in range(251)]) == 125.0
    assert rq.median_scale([float(value) for value in range(1, 252)]) == 126.0
    with pytest.raises(ValueError):
        rq.median_scale([0.0] * 251)
    with pytest.raises(ValueError):
        rq.median_scale([1.0] * 250)
    values = [1.0] * 251
    values[10] = math.nan
    with pytest.raises(ValueError):
        rq.median_scale(values)


def _entry(index, identity):
    return {'cursor': index, 'mrks_id': f'mrks_{index}',
            'relchem': {'identity': identity, 'variant': 'level3', 'database': 'DBH76'},
            'ae17': {'identity': f'ae-{index}', 'variant': 'level2', 'database': 'AE17'}}


def test_manifest_and_checkpoint_mismatch_are_rejected():
    rows = [_entry(index, f'reaction-{index}') for index in range(251)]
    assert rq.validate_manifest(rows)[125]['relchem']['variant'] == 'level3'
    with pytest.raises(ValueError):
        rq.validate_manifest(rows[:-1])
    broken = [_entry(index, 'same') for index in range(251)]
    with pytest.raises(ValueError):
        rq.validate_manifest(broken)
    missing = [_entry(index, f'reaction-{index}') for index in range(251)]
    missing[4]['relchem']['variant'] = ''
    with pytest.raises(ValueError):
        rq.validate_manifest(missing)
    saved = {'protocol_id': rq.PROTOCOL_ID, 'manifest_sha256': 'different', 'scale_hex': (1.25).hex(),
             'calibration_sha256': 'abc', 'lambdas': dict(rq.LAMBDAS), 'cursor': 0, 'logs': []}
    with pytest.raises(ValueError, match='manifest'):
        rq.require_checkpoint(saved, 1.25, 'abc')
    saved['manifest_sha256'] = rq.MANIFEST_SHA
    saved['scale_hex'] = (9.0).hex()
    with pytest.raises(ValueError, match='scale'):
        rq.require_checkpoint(saved, 1.25, 'abc')


def test_decision_classes_follow_the_predeclared_gate():
    milestones = {cursor: {'clean28': 9.0} for cursor in rq.MILESTONES}
    base = {'updates': 251, 'parity_passed': True, 'numerical_failure': False, 'milestones': milestones, 'qualified': {}}
    assert rq.classify(base) == 'NO-GO'
    assert not rq.beats_historical(rq.HISTORICAL_BEST_CLEAN28)
    milestones[251] = {'clean28': rq.HISTORICAL_BEST_CLEAN28 - 0.01}
    assert rq.classify(base) == 'PARTIAL'
    base['qualified'][251] = {'eligible': False}
    assert rq.classify(base) == 'NO-GO'
    base['qualified'][251] = {'eligible': True}
    assert rq.classify(base) == 'PROGRESS'
    milestones[90] = {'clean28': 6.5}
    base['qualified'][90] = {'eligible': True}
    assert rq.classify(base) == 'BREAKTHROUGH'
    base['updates'] = 250
    assert rq.classify(base) == 'PARTIAL'
    ratios = rq.scientifically_eligible({task: rq.BASELINE[task] * 0.5 for task in rq.TASKS})
    assert ratios['relchem'] == pytest.approx(0.5)
    assert rq.scientifically_eligible({task: rq.BASELINE[task] for task in rq.TASKS}) is False


def test_torn_calibration_line_is_dropped(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"index": 0}\n{"index": 1\n', encoding='utf-8')
    assert rq._jsonl(path) == [{'index': 0}]
    assert path.read_text(encoding='utf-8') == '{"index": 0}\n'


def test_frozen_control_receipts_match_predeclared_values():
    j251 = rq._load(rq.REPO / 'relchem_joint_epoch_metrics.json')['arms']['J']
    assert j251['ratios'] == rq.J251_RATIOS
    b80 = rq._load(rq.REPO / 'iid_adamw_lr_stabilization_metrics.json')['audits']['B80']
    assert b80['ratios_t0'] == rq.B80_RATIOS
    p536 = rq._load(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\R\endpoint_0_validation.json'))
    j251_validation = rq._load(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\J\endpoint_251_validation.json'))
    assert p536['metrics']['clean28'] == rq.P536_CLEAN28
    assert j251_validation['metrics']['clean28'] == rq.J251_CLEAN28


def test_historical_coefficients_match_the_j251_calibration_file():
    payload = rq._load(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\J\calibration.json'))
    assert payload['lambda'] == rq.LAMBDAS
    assert rq.sha256_file(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\J\calibration.json')) == rq.CALIBRATION_SHA
    assert rq.C == 0.5
