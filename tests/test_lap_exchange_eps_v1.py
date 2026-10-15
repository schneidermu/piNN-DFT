"""F026 configuration, persistence and bounded-experiment regression fixtures."""

import copy
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lap_exchange_eps_v1 as experiment


@pytest.fixture
def original():
    # Every tensor comes from the authenticated original fresh initialization.
    receipt = experiment.pilot.read("provenance")["pilot_checkpoint_receipts"][0]
    payload = experiment.pilot.resume_payload(receipt)
    return experiment.construct(payload["models"]["initial"], 1e-5)


def test_five_modules_and_only_declared_nonstate_attributes_change(original):
    before = experiment.pilot.state_hash(original)
    experiment.configure_epsilon(original)
    experiment.assert_epsilon(original)
    assert len(experiment.EXCHANGE) == 5
    assert before == experiment.pilot.state_hash(original) == experiment.INITIAL_HASH
    assert sum(p.numel() for p in experiment.audit.names(original).values()) == 9446
    assert len({id(p) for p in experiment.audit.names(original).values()}) == len(
        experiment.audit.names(original)
    )
    for row in experiment.epsilon_rows(original):
        assert row["epsilon"] == (1e-3 if row["owner"] == "exchange" else 1e-5)


def test_tensor_loading_does_not_prove_epsilon_and_wrong_shadow_fails(original):
    native = copy.deepcopy(original)
    experiment.configure_epsilon(native)
    shadow = copy.deepcopy(original).double()
    shadow.load_state_dict(native.state_dict())
    with pytest.raises(AssertionError):
        experiment.assert_epsilon(shadow)
    experiment.configure_epsilon(shadow)
    shadow.load_state_dict(native.state_dict())
    experiment.assert_epsilon(native)
    experiment.assert_epsilon(shadow)
    assert (
        experiment.scientific_identity(native)["scientific_hash"]
        != experiment.scientific_identity(original)["scientific_hash"]
    )


def test_correlation_epsilon_mutation_or_extra_exchange_module_rejected(original):
    experiment.configure_epsilon(original)
    original.c_input_layers[1].eps = 1e-3
    with pytest.raises(AssertionError):
        experiment.assert_epsilon(original)
    original.c_input_layers[1].eps = 1e-5
    original.x_extra_norm = torch.nn.LayerNorm(32)
    with pytest.raises(AssertionError):
        experiment.configure_epsilon(original)


def test_explicit_metadata_checkpoint_roundtrip_and_missing_metadata_fail(
    original, tmp_path
):
    experiment.configure_epsilon(original)
    optimizer = experiment.optimizer_for(original)
    assert not optimizer.state
    receipt = experiment.save(tmp_path / "fixture.pt", original, optimizer, 0)
    restored, shadow, ro, payload = experiment.restore(Path(receipt["path"]))
    assert experiment.scientific_identity(restored) == experiment.scientific_identity(
        original
    )
    experiment.assert_epsilon(restored)
    experiment.assert_epsilon(shadow)
    assert not ro.state and payload["t"] == 0
    assert payload["RNG"][2].device.type == "cpu"
    assert payload["RNG"][2].dtype == torch.uint8
    assert all(r.device.type == "cpu" for r in payload["RNG"][3])
    experiment.assert_optimizer(ro, restored)
    del payload["epsilon_manifest"]
    torch.save(payload, tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="epsilon metadata"):
        experiment.restore(tmp_path / "missing.pt")


def test_complete_functional_constraints_and_shared_exchange_correlation_coupling(
    original,
):
    experiment.configure_epsilon(original)
    assert experiment.pilot.constraints(original)["PASS"]
    model = copy.deepcopy(original).double()
    x = torch.tensor(
        [[0.8, 0.4, 0.03, 0.04, 0.05, 0.0, 0.0, 0.01, -0.02]],
        dtype=torch.float64,
        device=experiment.DEVICE,
    )
    y = model(x)
    # The existing shared exchange hidden vector affects correlation outputs.
    g = torch.autograd.grad(
        y[:, [0, 1, 28]].sum(), model.x_feature_extractor[0].weight
    )[0]
    assert (
        torch.isfinite(y).all() and torch.isfinite(g).all() and bool(g.abs().max() > 0)
    )
    experiment.assert_epsilon(model)


def test_precision_cadence_fixed_coordinates_and_no_extra_objectives():
    assert experiment.COORDINATES == (
        ("x_feature_extractor.1.bias", 6),
        ("x_feature_extractor.3.fc.1.bias", 16),
        ("x_feature_extractor.3.fc.0.weight", 887),
    )
    source = inspect.getsource(experiment.native_step)
    assert source.count("optimizer.step()") == 1
    assert ".to(p.dtype)" in source
    training = inspect.getsource(experiment.train)
    assert "if t == 32:" in training and "range(start, 33)" in training
    assert "theta0.pt" in training and "resume.pt" in training
    assert "kernel(" not in training and "SUBSET" not in training
    assert ".mean(0)" in inspect.getsource(experiment.batch)
    assert "configure_epsilon" not in source  # eps is fixed, not changed per update


def test_metric_definition_and_correct_final_comparison():
    dtype = torch.float64
    S = torch.diag(torch.tensor([1.0, 2.0], dtype=dtype))
    V = torch.tensor([[0.2, 0.1], [0.1, -0.3]], dtype=dtype)
    ref = torch.zeros_like(V)
    X = torch.diag(torch.tensor([1.0, 2**-0.5], dtype=dtype))
    actual = experiment.pilot.op.operator_loss(V, ref, S)
    torch.testing.assert_close(actual, (X @ V @ X).square().sum() / 2)
    control = 0.5199287840361003
    assert 0.99 * control == pytest.approx(0.5147294961957393)
    assert (control - 0.5) / control > 0.01


def test_optimizer_policy_is_fixed_and_rejects_changed_lr(original):
    experiment.configure_epsilon(original)
    optimizer = experiment.optimizer_for(original)
    experiment.assert_optimizer(optimizer, original)
    optimizer.param_groups[0]["lr"] = 1e-5
    with pytest.raises(AssertionError):
        experiment.assert_optimizer(optimizer, original)


def test_frozen_preflight_coordinate_coverage_and_no_tolerance_retuning():
    receipt = experiment.read("preflight")
    assert receipt["PASS"] and receipt["temporary_native_steps"] == 1
    assert receipt["temporary_step_retained"] is False
    assert receipt["recovery_extra_steps"] == 0
    assert len(receipt["derivative"]) == 6
    path = experiment.ROOT / (experiment.PREFIX + "derivative_checks.csv")
    rows = list(csv.DictReader(path.open()))
    for t in (0, 1):
        schedule = experiment.read(f"FD_schedule_theta{t}")["planned"]
        for name, index in experiment.COORDINATES:
            selected = [
                r
                for r in rows
                if r["check"] == "fixed_adjoint_FD"
                and int(r["theta"]) == t
                and r["name"] == name
                and int(r["index"]) == index
            ]
            plan = next(
                p for p in schedule if p["name"] == name and p["index"] == index
            )
            assert len(selected) == len(plan["steps"]) == 7
            assert [float(r["step"]) for r in selected] == plan["steps"]
    assert experiment.read("protocol")["tolerances"]["FD_adjacent_relative"] == 0.01


def test_subnormal_moment_is_nearest_f32_not_relative_precision():
    gradient = float(np.float32(-1.3619584361883007e-19))
    exact = 0.001 * gradient**2
    stored = float(np.float32(exact))
    minimum_step = float(np.nextafter(np.float32(0), np.float32(1)))
    assert stored < np.finfo(np.float32).tiny
    assert abs(stored - exact) <= minimum_step / 2
    assert stored == 1.8548987772267604e-41
    # Keep the reference bound in F64: NumPy weak-scalar F32 promotion would
    # underflow both sides of this scalar comparison to zero.
    assert abs(stored - exact) > 5 * float(np.finfo(np.float32).eps) * exact
