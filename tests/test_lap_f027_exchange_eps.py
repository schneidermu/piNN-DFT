"""F027 identity, isolated epsilon helpers, atomic restore and cadence checks."""

import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lap_f027_exchange_eps as experiment


def test_seed2027_reproducibility_in_isolated_unit_fixture():
    # Not a scientific initialization/arm: this constructor is a reproducibility fixture.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        fixture = experiment.pilot.pcPBELMLOptimizerV2Lap(**experiment.CONFIG)
    assert (
        experiment.pilot.state_hash(fixture)
        == experiment.read("initialization")["tensor_hash"]
    )
    assert experiment.read("initialization")["scientific_initializations"] == 1


def test_identical_native_tensor_order_storage_and_fresh_optimizers():
    models, shadows, optimizers, helpers = experiment.new_models()
    a, b = models.values()
    assert experiment.pilot.state_hash(a) == experiment.pilot.state_hash(b)
    assert list(experiment.names(a)) == list(experiment.names(b))
    assert sum(p.numel() for p in experiment.names(a).values()) == 9446
    assert all(
        x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters())
    )
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a.buffers(), b.buffers()))
    assert all(not o.state for o in optimizers.values())
    assert len({id(o) for o in optimizers.values()}) == 2
    assert (
        experiment.identity(a, helpers["CONTROL"])["scientific_hash"]
        != experiment.identity(b, helpers["INTERVENTION"])["scientific_hash"]
    )
    for arm in experiment.ARMS:
        helpers[arm].assert_epsilon(models[arm])
        helpers[arm].assert_epsilon(shadows[arm])
        assert all(p.dtype == torch.float32 for p in models[arm].parameters())
        assert all(p.dtype == torch.float64 for p in shadows[arm].parameters())


def test_isolated_defaults_and_post_load_manifest_fail_closed():
    models, shadows, _, helpers = experiment.new_models()
    helpers["CONTROL"].assert_epsilon(models["CONTROL"])
    helpers["INTERVENTION"].assert_epsilon(models["INTERVENTION"])
    with pytest.raises(AssertionError):
        helpers["INTERVENTION"].assert_epsilon(models["CONTROL"])
    shadows["CONTROL"].load_state_dict(models["INTERVENTION"].state_dict())
    helpers["CONTROL"].assert_epsilon(shadows["CONTROL"])
    assert helpers["CONTROL"].assert_epsilon.__defaults__ == (1e-5,)
    assert helpers["INTERVENTION"].assert_epsilon.__defaults__ == (1e-3,)
    for arm, model in models.items():
        rows = helpers[arm].epsilon_rows(model)
        assert sum(r["owner"] == "exchange" for r in rows) == 5
        assert all(r["epsilon"] == 1e-5 for r in rows if r["owner"] == "correlation")


def test_atomic_checkpoint_rng_and_epsilon_restoration(tmp_path):
    models, _, optimizers, helpers = experiment.new_models()
    for arm, h in helpers.items():
        path = tmp_path / (arm + ".pt")
        receipt = experiment.checkpoint(path, arm, models[arm], optimizers[arm], 0, h)
        restored, shadow, optimizer, payload = experiment.restore(
            Path(receipt["path"]), arm, h
        )
        assert experiment.identity(restored, h) == experiment.identity(models[arm], h)
        assert payload["RNG"][2].device.type == "cpu"
        assert torch.equal(torch.get_rng_state(), payload["RNG"][2])
        h.assert_epsilon(restored)
        h.assert_epsilon(shadow)
        assert not optimizer.state
        del payload["identity"]["epsilon"]
        torch.save(payload, tmp_path / (arm + "_missing.pt"))
        with pytest.raises(ValueError, match="epsilon"):
            experiment.restore(tmp_path / (arm + "_missing.pt"), arm, h)
        assert not path.with_suffix(".tmp").exists()


def test_no_optimizer_state_crossover_on_synthetic_fixture():
    # Tiny synthetic optimizer operations, not molecular updates or R0 training.
    a, b = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    b.load_state_dict(a.state_dict())
    oa, ob = (
        torch.optim.AdamW(a.parameters(), foreach=False),
        torch.optim.AdamW(b.parameters(), foreach=False),
    )
    for p in a.parameters():
        p.grad = torch.ones_like(p)
    oa.step()
    assert oa.state and not ob.state
    assert all(
        id(s) not in {id(v) for v in ob.state.values()} for s in oa.state.values()
    )


def test_mean_matrix_loss_and_gradient_reduction():
    p = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    matrices = [
        (p * torch.eye(2, dtype=torch.float64) + offset) for offset in (0.1, 0.3, 0.7)
    ]
    losses = [
        experiment.pilot.op.operator_loss(
            M, torch.zeros_like(M), torch.eye(2, dtype=torch.float64)
        )
        for M in matrices
    ]
    individual = [torch.autograd.grad(loss, p, retain_graph=True)[0] for loss in losses]
    direct = torch.autograd.grad(torch.stack(losses).mean(), p)[0]
    torch.testing.assert_close(direct, torch.stack(individual).mean())
    helper = experiment.helper("CONTROL")
    assert inspect.getsource(helper.native_step).count("optimizer.step()") == 1
    assert ".to(p.dtype)" in inspect.getsource(helper.native_step)


def test_actual_constraints_and_fixed_sources():
    models, _, _, helpers = experiment.new_models()
    for arm, model in models.items():
        assert experiment.pilot.constraints(model)["PASS"]
        helpers[arm].assert_epsilon(model)
    source = experiment.read("source_verification")
    assert [s["system"] for s in source["systems"]] == ["H2", "BeH2", "CO"]
    assert [s["points"] for s in source["systems"]] == [47080, 89200, 85960]
    assert "kernel(" not in inspect.getsource(experiment)
    assert "resume_payload" not in inspect.getsource(experiment)


def test_interleaving_and_t1_no_extra_molecular_step():
    assert list(experiment.ARMS) == ["CONTROL", "INTERVENTION"]
    source = inspect.getsource(experiment.train)
    assert "range(1,33)" in source.replace(" ", "")
    assert source.count("h.native_step(") == 1
    assert "native_step" not in inspect.getsource(experiment.qualify)
    assert "native_step" not in inspect.getsource(experiment.preflight)
    assert experiment.MILESTONES == (0, 1, 2, 16, 32)
    assert experiment.read("protocol")["effect_floor"] == 0.01
