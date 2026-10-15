"""F028 scientific identity, scalarization, stream and native-boundary fixtures."""

import ast
import copy
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "tools"),
    str(Path(__file__).resolve().parents[1] / "train_models"),
]
import lap_f028_joint_eps as exp
import lap_f028_joint_eps_run as execution


def test_exact_seed2028_constructor_reproduction():
    # Isolated unit reproducibility fixture, not a second scientific initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2028)
        fixture = exp.pilot.pcPBELMLOptimizerV2Lap(**exp.CONFIG).float()
    assert (
        exp.pilot.state_hash(fixture)
        == exp.read(exp.file("initialization"))["tensor_hash"]
    )
    assert exp.read(exp.file("initialization"))["scientific_initializations"] == 1


def test_independent_equal_tensors_and_empty_optimizers():
    a, sa, oa = exp.allocate("CONTROL")
    b, sb, ob = exp.allocate("INTERVENTION")
    assert exp.pilot.state_hash(a) == exp.pilot.state_hash(b)
    assert tuple(exp.core.named_trainable_parameters(a)) == tuple(
        exp.core.named_trainable_parameters(b)
    )
    assert (
        sum(p.numel() for p in exp.core.named_trainable_parameters(a).values()) == 9446
    )
    assert all(
        x.data_ptr() != y.data_ptr()
        for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
    assert all(
        x.data_ptr() != y.data_ptr()
        for x, y in zip(a.buffers(), b.buffers(), strict=True)
    )
    assert not oa.state and not ob.state and oa is not ob
    assert (
        exp.scientific_identity(a, "CONTROL")["scientific_configuration_hash"]
        != exp.scientific_identity(b, "INTERVENTION")["scientific_configuration_hash"]
    )
    for arm, model, shadow in (("CONTROL", a, sa), ("INTERVENTION", b, sb)):
        exp.eps_helpers.assert_epsilon(model, exp.ARMS[arm])
        exp.eps_helpers.assert_epsilon(shadow, exp.ARMS[arm])
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(p.dtype == torch.float64 for p in shadow.parameters())


def test_exact_five_lns_and_post_load_hook():
    model, shadow, _optimizer = exp.allocate("INTERVENTION")
    rows = exp.eps_helpers.epsilon_rows(model)
    assert {r["name"] for r in rows if r["epsilon"] == 1e-3} == set(
        exp.eps_helpers.EXCHANGE
    )
    assert len(exp.eps_helpers.EXCHANGE) == 5
    assert all(r["epsilon"] == 1e-5 for r in rows if r["owner"] == "correlation")
    shadow.load_state_dict(model.state_dict())
    next(
        m for n, m in shadow.named_modules() if n == exp.eps_helpers.EXCHANGE[0]
    ).eps = 1e-5
    with pytest.raises(AssertionError):
        shadow.load_state_dict(model.state_dict())


def test_exact_eight_offset_stream_and_shared_physical_presentations():
    rows = exp.b8.load_j251_rows()
    batches = exp.b8.build_batches(rows)
    assert exp.b8.assert_batch_properties(batches, rows)["presentations"] == 2008
    assert batches == exp.read(exp.OUT / "training_manifest.json")
    for t, entry in enumerate(batches):
        assert (
            entry["ae17"] == rows[t]["ae17"] and entry["mrks_id"] == rows[t]["mrks_id"]
        )
        assert len(entry["relchem"]) == 8
        assert entry["relchem"] == [
            rows[(t + offset) % 251]["relchem"] for offset in exp.b8.OFFSETS
        ]


def test_f010_chemistry_formula_removes_legacy_factor_once():
    scales = exp.read(exp.ROOT / "lap_f010_normalization_receipt.json")["scales"]
    for database, scale in scales.items():
        factor = exp.chemistry._factor(database)
        x = torch.tensor(-3.1, dtype=torch.float64, requires_grad=True)
        legacy = factor * torch.sqrt(x.square() + 1e-20)
        actual = legacy * exp.f010.f010_singleton_scale(exp.C010, scale, factor)
        expected = exp.C010 * torch.sqrt(x.square() + 1e-20) / scale
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(actual, x, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, x)[0]
        torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)


def test_f64_sum_zero_coordinates_and_exact_native_adamw_step():
    # Synthetic linear optimizer fixture; no molecular training step.
    model = torch.nn.Linear(2, 1).float()
    optimizer = torch.optim.AdamW(
        exp.core.named_trainable_parameters(model).values(),
        lr=2.5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    parameters = exp.core.named_trainable_parameters(model)
    raw = {
        task: {
            n: torch.full_like(p, float(i + 1), dtype=torch.float64)
            for n, p in parameters.items()
        }
        for i, task in enumerate(exp.TASKS)
    }
    for values in raw.values():
        values["bias"].zero_()
    result = exp.weighted_gradient(raw, exp.b8.LAMBDAS)
    expected = sum(exp.b8.LAMBDAS[t] * raw[t]["weight"] for t in exp.TASKS)
    torch.testing.assert_close(result["weight"], expected)
    exp.adamw_step(model, optimizer, raw, exp.b8.LAMBDAS)
    assert all(int(s["step"]) == 1 for s in optimizer.state.values())
    assert len(optimizer.state) == len(parameters)
    assert all(s["exp_avg"].dtype == torch.float32 for s in optimizer.state.values())
    assert all(p.grad is None for p in parameters.values())
    assert inspect.getsource(exp.adamw_step).count("optimizer.step()") == 1


def test_checkpoint_counter_moments_epsilon_rng_and_hashes(tmp_path):
    model, _shadow, optimizer = exp.allocate("INTERVENTION")
    # Directly populate a synthetic state: no optimizer update is executed.
    for p in model.parameters():
        if p.requires_grad:
            optimizer.state[p] = {
                "step": torch.tensor(7.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.full_like(p, 0.125),
            }
    path = tmp_path / "fixture.pt"
    exp.save(path, "INTERVENTION", model, optimizer, 7, [{}] * 7)
    restored, rs, ro, payload = exp.restore(path, "INTERVENTION")
    assert exp.scientific_identity(restored, "INTERVENTION") == exp.scientific_identity(
        model, "INTERVENTION"
    )
    assert all(int(s["step"]) == 7 for s in ro.state.values())
    assert all(
        torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])
        for a, b in zip(optimizer.state.values(), ro.state.values(), strict=True)
    )
    assert torch.equal(torch.get_rng_state(), payload["RNG"][2])
    assert payload["RNG"][2].device.type == "cpu"
    exp.eps_helpers.assert_epsilon(rs, 1e-3)
    del payload["identity"]["epsilon"]
    torch.save(payload, tmp_path / "invalid.pt")
    with pytest.raises((KeyError, AssertionError)):
        exp.restore(tmp_path / "invalid.pt", "INTERVENTION")


@pytest.mark.parametrize("arm", tuple(exp.ARMS))
def test_actual_arm_constraints_and_existing_density_operator_regressions(
    arm, monkeypatch
):
    import test_lap as physical
    import test_lap_operator_unit as operator_tests

    model = copy.deepcopy(exp.initial_template()).double()
    exp.eps_helpers.configure_epsilon(model, exp.ARMS[arm])
    assert exp.pilot.constraints(model)["PASS"]
    monkeypatch.setattr(physical, "small_model", lambda: copy.deepcopy(model))
    physical.test_tau_exactly_absent_and_spin_symmetry()
    physical.test_every_exact_anchor()
    monkeypatch.setattr(
        operator_tests, "pcPBELMLOptimizerV2Lap", lambda **_kwargs: copy.deepcopy(model)
    )
    operator_tests.test_lap_energy_operator_backpropagates_to_network_parameters()


def test_training_only_boundary_and_no_historical_initializers():
    bundle = exp.TrainingBundle()
    assert not hasattr(bundle, "validation_species") and not hasattr(
        bundle, "validation_reactions"
    )
    with pytest.raises(PermissionError):
        bundle.validation()
    assert len(bundle.systems) == 90 and len(bundle.reactions) == 268
    bundle.close()
    source = inspect.getsource(exp) + inspect.getsource(execution)
    assert ".kernel(" not in source
    assert "_load_parent_weights" not in source and "model_at(" not in source
    assert "for arm in exp.ARMS" in inspect.getsource(execution.train)
    assert "finish_pending_startup" in inspect.getsource(execution.train)
    assert exp.read(exp.file("protocol"))["coefficients"] == {
        **exp.b8.LAMBDAS,
        "c010": exp.C010,
    }
    assert np.isfinite(list(exp.b8.LAMBDAS.values())).all()


def test_executed_matched_stream_and_finite_training_receipts():
    """Audit executed presentations, rather than only the sampler implementation."""
    with (exp.ROOT / (exp.PREFIX + "training_log.csv")).open() as handle:
        rows = list(csv.DictReader(handle))
    manifest = exp.read(exp.OUT / "training_manifest.json")
    assert len(rows) == 502
    for cursor, entry in enumerate(manifest):
        for offset, arm in enumerate(exp.ARMS):
            row = rows[2 * cursor + offset]
            assert row["arm"] == arm
            assert int(row["update"]) == int(row["counter"]) == cursor + 1
            assert int(row["manifest_cursor"]) == cursor
            assert ast.literal_eval(row["chemistry_identity_variants"]) == [
                (s["identity"], s["variant"]) for s in entry["relchem"]
            ]
            assert row["ae17_identity"] == entry["ae17"]["identity"]
            assert row["ae17_variant"] == entry["ae17"]["variant"]
            assert row["mrks_id"] == entry["mrks_id"]
            assert row["finite"] == "True" and row["epsilon"] == "PASS"
            for key, value in row.items():
                if key.endswith(("_loss", "_norm", "_change")) and value:
                    assert np.isfinite(float(value)), (arm, cursor, key)
    assert exp.read(exp.file("training_completion"))["retained_updates"] == 502


def test_endpoint_native_state_finite_and_historical_sources_unchanged():
    for arm in exp.ARMS:
        model, shadow, optimizer, saved = exp.restore(exp.OUT / f"{arm}_t251.pt", arm)
        assert saved["cursor"] == 251
        exp.eps_helpers.assert_epsilon(model, exp.ARMS[arm])
        exp.eps_helpers.assert_epsilon(shadow, exp.ARMS[arm])
        assert all(torch.isfinite(p).all() for p in model.parameters())
        for state in optimizer.state.values():
            assert int(state["step"]) == 251
            assert torch.isfinite(state["exp_avg"]).all()
            assert torch.isfinite(state["exp_avg_sq"]).all()
            assert (state["exp_avg_sq"] >= 0).all()
    for path, expected in exp.read(exp.file("protocol"))["source_hashes"].items():
        assert exp.b8.sha256_file(exp.ROOT / path) == expected, path
