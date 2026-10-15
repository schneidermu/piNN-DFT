"""F029 restoration, LR-only comparison, stream and finite-step contracts."""

import ast
import copy
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lap_f029_lr_continuation as f


def test_exact_source_and_independent_lr_only_copies():
    assert f.base.b8.sha256_file(f.SOURCE) == f.SOURCE_SHA
    a, b = f.clone("CONTROL"), f.clone("LOW_LR")
    f.independent(a, b)
    assert f.identity(a[0]) == f.common()[3]["identity"]
    assert f.digest(f.normalized_optimizer(a[2])) == f.digest(
        f.normalized_optimizer(f.common()[2])
    )
    assert a[2].param_groups[0]["lr"] == 2.5e-5
    assert b[2].param_groups[0]["lr"] == 2.5e-6
    assert all(int(s["step"]) == 251 for s in a[2].state.values())
    assert all(s["exp_avg_sq"].sum() > 0 for s in a[2].state.values())


def test_manifest_original_variants_and_fixed_coefficients():
    source = f.base.read(f.base.OUT / "training_manifest.json")
    actual = f.stream()
    assert len(actual) == 32 and actual == source[:32]
    rows = f.base.b8.load_j251_rows()
    for k, entry in enumerate(actual):
        assert entry["cursor"] == k
        assert entry["relchem"] == [
            rows[(k + offset) % 251]["relchem"] for offset in f.base.b8.OFFSETS
        ]
        assert entry["ae17"] == rows[k]["ae17"]
        assert entry["mrks_id"] == rows[k]["mrks_id"]
    assert f.read("protocol")["coefficients"] == {
        **f.base.b8.LAMBDAS,
        "c010": 9.974823077811354,
    }
    assert (
        f.read("protocol")["precision"]
        == f.base.read(f.base.file("protocol"))["precision"]
    )


def test_authenticated_complete_raw_gradient_parity():
    a = np.load(f.OUT / "CONTROL_first_gradients.npz", allow_pickle=False)
    b = np.load(f.OUT / "LOW_LR_first_gradients.npz", allow_pickle=False)
    for task in f.base.TASKS:
        assert a[task].shape == b[task].shape == (9446,)
        assert np.isfinite(a[task]).all()
        np.testing.assert_allclose(a[task], b[task], atol=1e-10, rtol=1e-9)
    assert f.read("preflight")["molecular_optimizer_steps"] == 0


def test_local0_restoration_moments_rng_and_shadow_epsilon():
    model, shadow, optimizer, payload = f.restore(f.OUT / "LOW_LR_t251.pt", "LOW_LR")
    assert payload["local_cursor"] == 0 and payload["global_cursor"] == 251
    assert payload["RNG"][2].device.type == "cpu"
    assert f.digest(payload["RNG"]) == f.digest(f.base.pilot.rng())
    assert f.identity(model) == f.common()[3]["identity"]
    assert f.digest(f.normalized_optimizer(optimizer)) == f.digest(
        f.normalized_optimizer(f.common()[2])
    )
    name = f.base.eps_helpers.EXCHANGE[0]
    dict(shadow.named_modules())[name].eps = 1e-5
    with pytest.raises(AssertionError):
        shadow.load_state_dict(model.state_dict())


def test_native_boundary_uses_saved_counter_and_moments_synthetic_only():
    model = torch.nn.Linear(1, 1).float()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    for p in model.parameters():
        optimizer.state[p] = {
            "step": torch.tensor(251.0),
            "exp_avg": torch.full_like(p, 0.1),
            "exp_avg_sq": torch.full_like(p, 0.25),
        }
    old_moments = [s["exp_avg"].clone() for s in optimizer.state.values()]
    raw = {
        t: {
            n: torch.full_like(p, float(j + 1), dtype=torch.float64)
            for n, p in f.base.core.named_trainable_parameters(model).items()
        }
        for j, t in enumerate(f.base.TASKS)
    }
    combined = f.base.weighted_gradient(raw, f.base.b8.LAMBDAS)
    f.base.adamw_step(model, optimizer, raw, f.base.b8.LAMBDAS)
    assert all(p.grad is None for p in model.parameters())
    for (name, p), old in zip(model.named_parameters(), old_moments, strict=True):
        assert int(optimizer.state[p]["step"]) == 252
        expected = old.lerp(combined[name].float(), 0.1)
        torch.testing.assert_close(optimizer.state[p]["exp_avg"], expected)
        assert optimizer.state[p]["exp_avg"].dtype == torch.float32


def test_same_presentation_finite_step_classification(monkeypatch):
    entry = f.stream()[0]
    seen = []
    monkeypatch.setattr(f, "stream", lambda: [entry])
    monkeypatch.setattr(f, "save", lambda *_args: None)

    def scalar(_model, _shadow, _arm, _bundle, actual):
        assert actual is entry
        seen.append(actual)
        return {t: 2.0 for t in f.base.TASKS}

    monkeypatch.setattr(f.base, "scalar_tasks", scalar)
    row = {
        "finite_status": "PENDING",
        "local_update": 1,
        "total_linear_change": -0.01,
        **{t + "_loss_before": 1.0 for t in f.base.TASKS},
    }
    f.finish_pending("CONTROL", None, None, None, [row], None)
    assert seen == [entry]
    assert row["finite_step_overshoot"] and not row["locally_favorable_total"]
    assert all(row[t + "_actual_change"] == 1 for t in f.base.TASKS)


def test_exact_evaluator_output_routing_and_no_source_mutation(tmp_path, monkeypatch):
    source_before = f.base.b8.sha256_file(f.SOURCE)
    old_out, old_restore = f.base.OUT, f.base.restore
    monkeypatch.setattr(f, "OUT", tmp_path)

    def fixture(arm, cursor):
        alias = f.base.OUT / f"INTERVENTION_t{cursor}.pt"
        assert f.base.b8.sha256_file(alias) == source_before
        assert arm == "INTERVENTION" and cursor == 251
        return {"routed": True}

    monkeypatch.setattr(f.base, "evaluation", fixture)
    assert f.evaluate("fixture", f.SOURCE, 251) == {"routed": True}
    assert f.base.OUT == old_out and f.base.restore is old_restore
    assert f.base.b8.sha256_file(f.SOURCE) == source_before


def test_closed_clean28_gate_and_no_new_initializer(tmp_path, monkeypatch):
    writes = {}
    monkeypatch.setattr(f, "file", lambda name: tmp_path / (name + ".json"))
    monkeypatch.setattr(f, "write", lambda name, value: writes.update({name: value}))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Ineligible endpoint accessed development data")

    monkeypatch.setattr(f.base, "PublicationDataset", forbidden)
    result = f.clean28({a: {"ratios": {"op": 2}, "eligible": False} for a in f.LRS})
    assert all(v["evaluations"] == 0 for v in result["values"].values())
    assert not result["future_test_access"]
    source = inspect.getsource(f)
    assert ".kernel(" not in source and "manual_seed(" not in source
    assert "pcPBELMLOptimizerV2Lap(" not in source
    assert "2.5e-6" in source and "F029" in source


def test_pairwise_resolution_and_historical_integrity():
    same = {"JW": 0.1, "op": 0.03}
    assert not f.improved("JW", same, same)
    assert not f.improved("op", same, same)
    assert f.improved("JW", {"JW": 0.09}, same)
    for path, expected in f.read("source_verification")["F028_artifacts"].items():
        assert f.base.b8.sha256_file(path) == expected


def test_restored_intervention_constraints_and_variational_fixture(monkeypatch):
    import test_lap as physical
    import test_lap_operator_unit as operator_tests

    model = copy.deepcopy(f.common()[0]).cpu().double()
    assert f.base.pilot.constraints(model)["PASS"]
    monkeypatch.setattr(physical, "small_model", lambda: copy.deepcopy(model))
    physical.test_tau_exactly_absent_and_spin_symmetry()
    physical.test_every_exact_anchor()
    monkeypatch.setattr(
        operator_tests, "pcPBELMLOptimizerV2Lap", lambda **_kwargs: copy.deepcopy(model)
    )
    operator_tests.test_lap_energy_operator_backpropagates_to_network_parameters()


def test_executed_matched_continuation_and_endpoints():
    if not f.file("training_complete").exists():
        pytest.skip("Both retained trajectories have not yet completed")
    with (f.ROOT / (f.PREFIX + "training_log.csv")).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    for k, entry in enumerate(f.stream()):
        for offset, arm in enumerate(f.LRS):
            r = rows[2 * k + offset]
            assert r["arm"] == arm
            assert int(r["global_update"]) == 252 + k
            assert int(r["local_update"]) == k + 1
            assert float(r["LR"]) == f.LRS[arm]
            assert ast.literal_eval(r["chemistry_identity_variants"]) == [
                (s["identity"], s["variant"]) for s in entry["relchem"]
            ]
            assert r["ae17_identity"] == entry["ae17"]["identity"]
            assert r["ae17_variant"] == entry["ae17"]["variant"]
            assert r["mrks_id"] == entry["mrks_id"]
    for arm in f.LRS:
        model, _shadow, optimizer, p = f.restore(f.OUT / f"{arm}_t283.pt", arm)
        assert p["local_cursor"] == 32 and p["global_cursor"] == 283
        assert all(int(s["step"]) == 283 for s in optimizer.state.values())
        assert f.base.pilot.constraints(model)["PASS"]
    with (f.ROOT / (f.PREFIX + "finite_steps.csv")).open() as handle:
        finite = list(csv.DictReader(handle))
    assert len(finite) == 6
    assert {int(r["global_update"]) for r in finite} == {252, 253, 283}
    assert all(r["finite_status"].startswith("COMPLETE") for r in finite)
    for row in finite:
        vectors = np.load(
            f.OUT / f"{row['arm']}_step{row['global_update']}_vectors.npz",
            allow_pickle=False,
        )
        delta = vectors["displacement"]
        weighted_linear = 0.0
        for task in f.base.TASKS:
            actual = float(vectors[task] @ delta)
            np.testing.assert_allclose(
                actual, float(row[task + "_linear_change"]), atol=1e-10, rtol=1e-9
            )
            weighted_linear += f.base.b8.LAMBDAS[task] * actual
            assert float(row[task + "_loss_after"]) - float(
                row[task + "_loss_before"]
            ) == float(row[task + "_actual_change"])
        np.testing.assert_allclose(
            weighted_linear, float(row["total_linear_change"]), atol=1e-10, rtol=1e-9
        )
    _mc, _sc, oc, _pc = f.restore(f.OUT / "CONTROL_t252.pt", "CONTROL")
    _ml, _sl, ol, _pl = f.restore(f.OUT / "LOW_LR_t252.pt", "LOW_LR")
    assert f.digest(f.normalized_optimizer(oc)) == f.digest(f.normalized_optimizer(ol))
