"""F022 cache, scalar, strict angular gate and early-stop contracts."""
import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f022_scalarization as exp


def test_unique_strict_interval_and_fail_closed_cases():
    lower, upper = exp.interval(-.662494, .752422, -.984409)
    assert lower == .662494 and upper == pytest.approx(.752422 / .984409)
    for args in ((-.9, .1, -.99), (.5, .5, .1), (0., 0., 0.)):
        with pytest.raises(ValueError):
            exp.interval(*args)


def test_frozen_alpha_and_norm_identity_use_population_S():
    cal = exp.audit.read(exp.ROOT / 'lap_f022_gradient_calibration.json')
    assert cal['population'] == 'S'
    assert cal['alpha'] == (cal['alpha_min'] + cal['alpha_max']) / 2
    assert cal['K'] == pytest.approx(exp.audit.geometry.C010 * cal['JW_norm_S'] / cal['D'], rel=1e-14)
    assert cal['a'] == cal['K'] / cal['JW_norm_S']
    assert cal['b'] == cal['K'] * cal['alpha'] / cal['legacy_norm_S']
    assert cal['chemistry_gradient_norm'] == pytest.approx(exp.audit.geometry.C010 * cal['JW_norm_S'], rel=1e-14)


def test_single_scalar_and_parameter_gradient_equal_two_terms():
    p = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor([[1., 2.], [-3., .4], [2., -.1]], dtype=torch.float64)
    residual = matrix @ p
    phi = torch.sqrt(residual.square() + 1e-20)
    scale = torch.tensor([2., 3., 5.], dtype=torch.float64)
    factors = torch.tensor([exp.audit.chemistry._factor(d) for d in ('ABDE4', 'EA13', 'PA8')], dtype=torch.float64)
    cal = exp.audit.read(exp.ROOT / 'lap_f022_gradient_calibration.json')
    a, b = cal['a'], cal['b']
    combined = ((a/scale + b*factors)*phi).mean()
    separate = a*(phi/scale).mean() + b*(factors*phi).mean()
    torch.testing.assert_close(combined, separate, rtol=1e-14, atol=1e-14)
    cg = torch.autograd.grad(combined, p, retain_graph=True)[0]
    sg = torch.autograd.grad(separate, p)[0]
    torch.testing.assert_close(cg, sg, rtol=1e-14, atol=1e-14)


def test_legacy_is_actual_production_singleton_not_database_RMSE():
    from train_models.optuna_joint import batch_fchem
    for db in ('ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13'):
        prediction = torch.tensor([-.57], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([.3], dtype=torch.float64)
        actual = batch_fchem([db], prediction, target)
        expected = exp.audit.chemistry._factor(db) * torch.sqrt((prediction[0]-target[0]).square()+1e-20)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ag = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
        eg = torch.autograd.grad(expected, prediction)[0]
        torch.testing.assert_close(ag, eg, rtol=0, atol=0)


def test_cache_hashes_and_fixed_source_order_match():
    p = exp.audit.read(exp.ROOT / 'lap_f022_provenance.json')
    assert p['authenticated'] and p['source_records_verified'] == 251
    assert exp.audit.sha(exp.stream.OUT/'stream_jacobian.npz') == p['stream_jacobian_sha256']
    assert exp.audit.sha(exp.stream.OUT/'stream_rows.json') == p['stream_mapping_sha256']
    assert exp.audit.sha(exp.f010.PARENT_CHECKPOINT) == exp.CHECKPOINT
    for relative, digest in p['source_hashes'].items():
        assert exp.audit.sha(exp.ROOT / relative) == digest
    parent = torch.load(exp.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert exp.f010.continuation.moment_sha(parent['optimizer']) == exp.MOMENT


def test_original_stream_population_separation_and_fixed_coefficients():
    protocol = exp.audit.read(exp.ROOT / 'lap_f022_protocol.json')
    assert protocol['positions'] == list(range(64)) and protocol['c010'] == 9.974823077811354
    assert protocol['lambdas'] == exp.b8.LAMBDAS
    assert protocol['lambdas']['op'] == .33597561607048215
    assert protocol['baseline']['relchem'] == 1.2354710978866068
    assert protocol['baseline']['ae17'] == 24.901047104016875
    assert protocol['baseline']['exc'] == 92.17480502000551
    assert protocol['baseline']['op'] == .03314821681605566
    assert protocol['preflight_seconds'] == 900 and protocol['active_seconds'] == 5400 and protocol['wall_seconds'] == 10800
    mapping = exp.audit.read(exp.stream.OUT / 'stream_rows.json')
    lookup = {(r['identity'], r['variant']) for r in mapping}
    batches = exp.f010._batches()[:64]
    assert len(batches) == 64 and sum(len(r['relchem']) for r in batches) == 512
    assert all((r['identity'], r['variant']) in lookup for batch in batches for r in batch['relchem'])
    assert 'S:' in protocol['calibration_population'] and 'E:' in protocol['scientific_population']


def test_failure_stops_without_native_updates_or_development():
    decision = exp.audit.read(exp.ROOT / 'lap_f022_decision.json')
    assert decision['classification'] == 'C' and decision['JW_dot_G'] > 0 and decision['legacy_dot_G'] < 0
    assert decision['legacy_dot_chemistry'] > 0 and decision['legacy_dot_physics'] < 0
    assert not decision['training_launched'] and decision['optimizer_updates'] == 0
    assert decision['native_step'] == 'NOT RUN' and decision['Clean28'] == 'NOT EVALUATED'
    tree = ast.parse(Path(exp.__file__).read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('AdamW', 'step', 'validation') for n in ast.walk(tree))
    assert decision['active_compute_seconds'] < 900 and decision['whole_workflow_wall_seconds'] < 10800


def test_architecture_constraints_reuse_is_authenticated_parent():
    receipt = exp.audit.read(exp.ROOT / 'lap_f021_phys4_resume_parity.json')
    protocol = exp.audit.read(exp.ROOT / 'lap_f021_phys4_protocol.json')
    assert protocol['checkpoint_sha256'] == exp.CHECKPOINT
    assert receipt['moment_sha256'] == exp.MOMENT and receipt['constraints']['passed']
    assert receipt['constraints']['tau_gap'] == receipt['constraints']['spin_max_abs'] == 0
    assert receipt['constraints']['ueg_beta_gap'] == receipt['constraints']['high_density_gamma_gap'] == 0
