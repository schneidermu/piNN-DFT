"""Prefix-exposure and reduction contracts for the PHYS4-only intervention."""
import ast
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f021_phys4 as exp
import lap_f021_phys4_eval as evaluation
import lap_f021_phys4_run as run


def test_prefix_mapping_has_exact_exposures_and_no_outside_positions():
    assert exp.OFFSETS == (0, 31, 62, 93)
    rows = [p for t in range(128) for p in exp.positions(t)]
    assert Counter(rows) == dict.fromkeys(range(128), 4)
    assert len(rows) == 512 and min(rows) == 0 and max(rows) == 127


def test_centered_variance_and_joint_covariance_identity():
    rng = np.random.default_rng(21)
    physical = rng.normal(size=(128, 19))
    chemistry = rng.normal(size=(128, 19))
    averaged = physical[np.array([exp.positions(t) for t in range(128)])].mean(axis=1)
    np.testing.assert_allclose(physical.mean(axis=0), averaged.mean(axis=0), atol=1e-14)
    assert run.variance(averaged) <= run.variance(physical)
    for values in (physical, averaged):
        assert np.isclose(run.variance(chemistry + values),
                          run.variance(chemistry) + run.variance(values) + 2 * run.covariance(chemistry, values), atol=1e-12)


def test_four_scalar_gradient_is_mean_and_coefficients_applied_once():
    p = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    losses = [(p * (i + 1)).square().sum() for i in range(4)]
    individual = [torch.autograd.grad(loss, p, retain_graph=True)[0] for loss in losses]
    average = torch.autograd.grad(sum(losses) / 4, p)[0]
    torch.testing.assert_close(average, torch.stack(individual).mean(0), atol=0, rtol=0)
    coefficient = exp.b8.LAMBDAS['ae17']
    torch.testing.assert_close(average * coefficient, torch.stack([g * coefficient for g in individual]).mean(0), rtol=1e-15, atol=1e-15)


def test_constant_scientific_configuration_and_single_native_step_reuse():
    assert exp.TOTAL == 128 and exp.BUDGET == 36000
    assert exp.audit.geometry.C010 == 9.974823077811354
    assert exp.b8.BASELINE['relchem'] == 1.2354710978866068
    assert exp.b8.BASELINE['op'] == 0.03314821681605566
    assert exp.b8.LAMBDAS['op'] == 0.33597561607048215
    tree = ast.parse(Path(run.__file__).read_text())
    adapter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training_adapter')
    calls = [n for n in ast.walk(adapter) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {'step', 'clip_grad_norm_'} for n in calls)
    assert not any(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == 'np' and n.func.attr == 'load' for n in calls)
    assert any(isinstance(n.func, ast.Name) and n.func.id == 'original_other' for n in calls)


def test_immutable_historical_chemistry_manifest():
    batches = exp.f010._batches()
    assert len(batches[:128]) == 128
    assert sum(len(b['relchem']) for b in batches[:128]) == 1024
    assert exp.audit.sha(exp.f010.continuation.PARENT_MANIFEST) == exp.f010.continuation.MANIFEST_SHA


def test_endpoint_population_and_conditional_development_gate():
    code = Path(evaluation.__file__).read_text()
    assert 'len(rows) == 251 and len(physical) == 90' in code
    assert "continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')" in code
    assert "values[t] / b8.BASELINE[t]" in code
    assert 'all(v < 1 and math.isfinite(v)' in code
    assert "if not final['eligible']:" in code
    assert code.index("if not final['eligible']:") < code.index('result = validation(model, bundle)')
    assert "'evaluations': 0" in code and "'evaluations': 1" in code
    assert 'audit.TrainingBundle' in code


def test_execution_guard_has_frozen_hash_and_whole_workflow_budget():
    path = exp.ROOT / 'tools/lap_f021_phys4_execute.py'
    contract = exp.read('execution_contract')
    assert contract['driver_sha256'] == exp.audit.sha(path)
    assert contract['protocol_sha256'] == exp.audit.sha(exp.ROOT / 'lap_f021_phys4_protocol.json')
    assert contract['budget_seconds'] == 36000
    code = path.read_text()
    assert "time.time() - contract['wall_clock_origin_unix']" in code
    assert 'training_module.PublicationDataset = exp.audit.TrainingBundle' in code
