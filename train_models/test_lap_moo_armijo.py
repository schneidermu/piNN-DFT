"""Focused acceptance tests for the direct vector-Armijo PCD update path."""

from __future__ import annotations

import copy
import random
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

TRAIN_MODELS = Path(__file__).resolve().parent
for _path in (str(TRAIN_MODELS.parent), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import lap_moo_protocol as _lap_moo_protocol
import lap_moo_training as _lap_moo_training
import train_lap_moo as _train_lap_moo_driver

train_moo_update = _lap_moo_training.train_moo_update


TASKS = ("chem", "exc", "op")
ARMIJO_RULE = "pcd_direct_vector_armijo"
PCD = {"tau": 0.02, "beta": 0.999, "eps": 1e-8, "qp_tolerance": 1e-9}


class ScalarModel(nn.Module):
    def __init__(self, value: float = 1.0, *, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([value], dtype=dtype))
        self.register_buffer("forwards", torch.zeros((), dtype=torch.int64))


class StatefulScalarModel(ScalarModel):
    """A trial forward mutates a buffer and consumes RNG without changing loss."""

    def bump_and_draw(self) -> torch.Tensor:
        with torch.no_grad():
            self.forwards.add_(1)
        random.random()
        np.random.random()
        return torch.rand((), dtype=self.x.dtype, device=self.x.device) * 0.0


def _assert_rng_equal(actual, expected):
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    torch.testing.assert_close(actual["torch_cpu"], expected["torch_cpu"], rtol=0, atol=0)
    assert len(actual["torch_cuda"]) == len(expected["torch_cuda"])
    for left, right in zip(actual["torch_cuda"], expected["torch_cuda"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def quadratic_factories(model: ScalarModel, curvatures=(1.0, 2.0, 3.0)):
    return {
        task: (lambda curvature=curvature: 0.5 * curvature * model.x.square().sum())
        for task, curvature in zip(TASKS, curvatures)
    }


def stateful_quadratic_factories(model: StatefulScalarModel, curvature=4.0):
    return {
        task: (
            lambda: 0.5 * curvature * model.x.square().sum() + model.bump_and_draw()
        )
        for task in TASKS
    }


def conflicting_factories(model: ScalarModel):
    return {
        "chem": lambda: model.x.sum(),
        "exc": lambda: -model.x.sum(),
        "op": lambda: model.x.sum() * 0.0,
    }


def run_armijo(
    model,
    factories,
    *,
    initial_step_size=0.1,
    max_backtracks=20,
    aggregator_state=None,
    aggregator=None,
    world_size=1,
):
    return train_moo_update(
        model,
        None,
        factories,
        method="pcd",
        hyperparameters=PCD,
        aggregator_state=aggregator_state,
        aggregator=aggregator,
        world_size=world_size,
        step_rule=ARMIJO_RULE,
        vector_armijo={
            "c": 1e-4,
            "rho": 0.5,
            "max_backtracks": max_backtracks,
            "initial_step_size": initial_step_size,
        },
    )


def vector_diagnostics(result):
    return result.diagnostics["vector_armijo"]


def test_aligned_quadratics_accept_full_step_and_commit_exact_trial():
    model = ScalarModel()
    x0 = model.x.detach().clone()
    alpha0 = 0.1

    result = run_armijo(
        model,
        quadratic_factories(model),
        initial_step_size=alpha0,
    )

    details = vector_diagnostics(result)
    assert result.accepted is True
    assert details["accepted_t"] == 1.0
    assert details["backtracks"] == 0
    assert len(details["trials"]) == 1
    assert details["trials"][0]["armijo_pass"] is True
    assert details["trials"][0]["strict_all_task_decrease"] is True
    assert all(value < 0.0 for value in details["directional_slopes"].values())
    # The three gradients are positively aligned; PCD rescales to the raw
    # chemistry norm, so the accepted direct displacement is alpha0 * 1.
    torch.testing.assert_close(model.x, x0 - alpha0, rtol=0.0, atol=1e-14)
    assert result.aggregator_state["t"] == 1


def test_high_curvature_overshoot_backtracks_until_all_three_decrease():
    model = ScalarModel()
    alpha0 = 1.0
    factories = quadratic_factories(model, curvatures=(4.0, 4.0, 4.0))

    result = run_armijo(model, factories, initial_step_size=alpha0)

    details = vector_diagnostics(result)
    assert result.accepted is True
    assert details["accepted_t"] == 0.25
    assert details["backtracks"] == 2
    assert [trial["t"] for trial in details["trials"]] == [1.0, 0.5, 0.25]
    assert [trial["armijo_pass"] for trial in details["trials"]] == [False, False, True]
    assert all(trial["strict_all_task_decrease"] is False for trial in details["trials"][:2])
    assert details["trials"][2]["strict_all_task_decrease"] is True
    # At t=1/4, x=0 exactly; the committed parameter must match that trial.
    torch.testing.assert_close(model.x, torch.zeros_like(model.x), rtol=0.0, atol=1e-14)
    assert result.aggregator_state["t"] == 1


def test_accepted_backtracks_restore_trial_buffers_rng_and_commit_ema_once():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected_rng = _lap_moo_training.capture_rng_state()
    for _ in TASKS:
        random.random()
        np.random.random()
        torch.rand((), dtype=torch.float64)
    expected_rng = _lap_moo_training.capture_rng_state()

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model = StatefulScalarModel()
    result = run_armijo(
        model,
        stateful_quadratic_factories(model, curvature=4.0),
        initial_step_size=1.0,
    )

    assert result.accepted is True
    assert vector_diagnostics(result)["accepted_t"] == 0.25
    assert result.aggregator_state["t"] == 1
    # Trial forwards are rolled back; only the three one-per-task gradient
    # forwards are retained in model buffers.
    assert model.forwards.item() == len(TASKS)
    _assert_rng_equal(_lap_moo_training.capture_rng_state(), expected_rng)


def test_non_common_descent_rejects_without_advancing_parameters_or_ema():
    model = ScalarModel()
    before = copy.deepcopy(model.state_dict())
    incoming_ema = None

    result = run_armijo(
        model,
        conflicting_factories(model),
        initial_step_size=0.1,
        aggregator_state=incoming_ema,
    )

    details = vector_diagnostics(result)
    assert result.accepted is False
    assert result.stop_reason
    assert any(value >= 0.0 for value in details["directional_slopes"].values())
    assert details["trials"] == []
    assert result.aggregator_state is incoming_ema
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_bounded_search_rejects_when_no_fraction_passes_and_restores_state_rng():
    model = StatefulScalarModel()
    factories = stateful_quadratic_factories(model, curvature=4.0)
    initial_model = copy.deepcopy(model.state_dict())
    initial_rng = _lap_moo_training.capture_rng_state()

    result = run_armijo(
        model,
        factories,
        initial_step_size=1.0,
        max_backtracks=1,
    )

    details = vector_diagnostics(result)
    assert result.accepted is False
    assert result.stop_reason
    assert [trial["t"] for trial in details["trials"]] == [1.0, 0.5]
    assert all(trial["armijo_pass"] is False for trial in details["trials"])
    assert all(trial["strict_all_task_decrease"] is False for trial in details["trials"])
    _assert_rng_equal(_lap_moo_training.capture_rng_state(), initial_rng)
    for key, value in initial_model.items():
        assert torch.equal(model.state_dict()[key], value)
    assert result.aggregator_state is None


def test_aggregator_runs_once_across_multiple_trial_evaluations():
    model = ScalarModel()
    calls = 0

    def counting_aggregator(*args, **kwargs):
        nonlocal calls
        calls += 1
        from moo_aggregators import aggregate_task_gradients

        return aggregate_task_gradients(*args, **kwargs)

    result = run_armijo(
        model,
        quadratic_factories(model, curvatures=(4.0, 4.0, 4.0)),
        initial_step_size=1.0,
        aggregator=counting_aggregator,
    )

    details = vector_diagnostics(result)
    assert result.accepted is True
    assert details["backtracks"] == 2
    assert calls == 1
    assert result.aggregator_state["t"] == 1


def test_deterministic_replay_selects_same_fraction_and_parameter():
    def replay():
        torch.manual_seed(712)
        model = ScalarModel()
        result = run_armijo(
            model,
            quadratic_factories(model, curvatures=(4.0, 4.0, 4.0)),
            initial_step_size=1.0,
        )
        return model.x.detach().clone(), vector_diagnostics(result)

    first_x, first_details = replay()
    second_x, second_details = replay()
    assert first_details["accepted_t"] == second_details["accepted_t"] == 0.25
    assert first_details["trials"] == second_details["trials"]
    assert torch.equal(first_x, second_x)


def test_committed_parameter_is_bitwise_the_evaluated_float32_candidate():
    base = torch.tensor(0.6284514665603638, dtype=torch.float32)
    nominal_target = torch.tensor(-0.8161681294441223, dtype=torch.float32)
    delta = nominal_target - base
    assert not torch.equal(base + delta, nominal_target)

    model = ScalarModel(float(base), dtype=torch.float32)
    joint_direction = -delta.reshape_as(model.x)
    evaluated_candidates = []

    def objective():
        evaluated_candidates.append(model.x.detach().clone())
        return model.x.sum()

    def prescribed_direction(task_grads, **kwargs):
        return {"x": joint_direction.clone()}, {}, {"test": "prescribed"}

    result = run_armijo(
        model,
        {task: objective for task in TASKS},
        initial_step_size=1.0,
        aggregator=prescribed_direction,
    )

    details = vector_diagnostics(result)
    assert result.accepted is True
    assert details["accepted_t"] == 1.0
    trial_candidates = evaluated_candidates[len(TASKS) :]
    assert len(trial_candidates) == len(TASKS)
    assert all(torch.equal(candidate, trial_candidates[0]) for candidate in trial_candidates)
    assert torch.equal(model.x.detach(), trial_candidates[0])
    assert details["trials"][0]["armijo_pass"] is True


def test_direct_armijo_full_step_matches_canonical_sgd_step():
    alpha0 = 6.632573669086685e-7
    direct_model = ScalarModel()
    direct = run_armijo(
        direct_model,
        quadratic_factories(direct_model, curvatures=(1.0, 1.0, 1.0)),
        initial_step_size=alpha0,
    )

    sgd_model = ScalarModel()
    optimizer = torch.optim.SGD(sgd_model.parameters(), lr=alpha0)
    canonical = train_moo_update(
        sgd_model,
        optimizer,
        quadratic_factories(sgd_model, curvatures=(1.0, 1.0, 1.0)),
        method="pcd",
        hyperparameters=PCD,
        aggregator_state=None,
        world_size=1,
    )

    assert direct.accepted is True
    assert vector_diagnostics(direct)["accepted_t"] == 1.0
    assert direct.learning_rate == canonical.learning_rate == alpha0
    assert torch.equal(direct_model.x, sgd_model.x)
    assert direct.aggregator_state["t"] == canonical.aggregator_state["t"] == 1


def _checkpoint_manifest(*, world_size=1, initial_step_size=0.1):
    grouped = {
        0: [{"Database": "ABDE4", "ReactionID": 1, "component_paths": ["A1.npz"]}],
    }
    manifest = _lap_moo_protocol.build_sampling_manifest(
        grouped,
        ["H2"],
        updates=4,
        seed=73,
        source_hashes={"test-fixture": "1" * 64},
        world_size=world_size,
    )
    config = {
        "c": 1.0e-4,
        "rho": 0.5,
        "max_backtracks": 20,
        "initial_step_size": initial_step_size,
    }
    metadata = _lap_moo_protocol.make_protocol_metadata(
        architecture="scalar-armijo-test",
        method="pcd",
        method_hyperparameters=PCD,
        fixed_scalarization=None,
        optimizer={"name": "none"},
        lr_schedule={"name": "none"},
        predopt_checkpoint_sha256="2" * 64,
        sampling_manifest_sha256=manifest["manifest_sha256"],
        minnesota_data_sha256="3" * 64,
        operator_corpus_manifest_sha256="4" * 64,
        ao_cache_manifest_sha256="5" * 64,
        reaction_dispersions_sha256="6" * 64,
        mrks_dispersions_sha256="7" * 64,
        random_seed=73,
        dtype="torch.float64",
        grid_chunk_size=16,
        ao_cache_chunk_size=4096,
        world_size=world_size,
        sampling_manifest_file_sha256="8" * 64,
        source_code_sha256={
            path: f"{index + 9:064x}"
            for index, path in enumerate(_lap_moo_protocol.PCD_SOURCE_FILE_PATHS)
        },
        step_rule=_lap_moo_protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        vector_armijo=config,
    )
    return manifest, metadata, config


def _stochastic_constant_factories(model):
    def make_objective():
        random.random()
        np.random.random()
        torch.rand((), dtype=model.x.dtype)
        return 0.5 * model.x.square().sum()

    return {task: make_objective for task in TASKS}


def test_actual_direct_checkpoint_resume_matches_uninterrupted_second_update(tmp_path):
    manifest, metadata, config = _checkpoint_manifest()
    torch.manual_seed(313)
    random.seed(314)
    np.random.seed(315)
    model = ScalarModel()
    first = train_moo_update(
        model,
        None,
        _stochastic_constant_factories(model),
        method="pcd",
        hyperparameters=metadata["method_hyperparameters"],
        aggregator_state=None,
        world_size=1,
        step_rule=_lap_moo_protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        vector_armijo=config,
    )
    assert first.accepted is True
    assert first.aggregator_state["t"] == 1

    checkpoint = tmp_path / "direct-armijo.pt"
    _lap_moo_training.save_moo_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        protocol_metadata=metadata,
        sampling_manifest=manifest,
        next_update=1,
        aggregator_state=first.aggregator_state,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_state_dict"] is None
    assert payload["scheduler_state_dict"] is None

    expected_second = train_moo_update(
        model,
        None,
        _stochastic_constant_factories(model),
        method="pcd",
        hyperparameters=metadata["method_hyperparameters"],
        aggregator_state=first.aggregator_state,
        world_size=1,
        step_rule=_lap_moo_protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        vector_armijo=config,
    )
    assert expected_second.accepted is True
    expected_model = copy.deepcopy(model.state_dict())
    expected_ema = copy.deepcopy(expected_second.aggregator_state)
    expected_rng = _lap_moo_training.capture_rng_state()

    resumed_model = ScalarModel()
    cursor, resumed_ema = _lap_moo_training.load_moo_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=None,
        scheduler=None,
        expected_protocol_metadata=metadata,
        sampling_manifest=manifest,
        restore_rng=True,
    )
    assert cursor == 1
    assert resumed_ema == first.aggregator_state
    resumed_second = train_moo_update(
        resumed_model,
        None,
        _stochastic_constant_factories(resumed_model),
        method="pcd",
        hyperparameters=metadata["method_hyperparameters"],
        aggregator_state=resumed_ema,
        world_size=1,
        step_rule=_lap_moo_protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        vector_armijo=config,
    )
    assert resumed_second.accepted is True
    assert resumed_second.aggregator_state == expected_ema
    assert tuple(resumed_model.state_dict()) == tuple(expected_model)
    for name in expected_model:
        assert torch.equal(resumed_model.state_dict()[name], expected_model[name])
    _assert_rng_equal(_lap_moo_training.capture_rng_state(), expected_rng)


def _two_rank_global_trial_resume_worker(rank, port, output_prefix, checkpoint_path, manifest, metadata, config):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        model = ScalarModel()

        def factories(current_model, *, second=False):
            coefficient = 1.0 if rank == 0 else 7.0
            target = 0.0 if rank == 0 else (3.0 if second else 2.0)

            def objective():
                return 0.5 * coefficient * (current_model.x - target).square().sum()

            return {task: objective for task in TASKS}

        first = run_armijo(
            model,
            factories(model),
            initial_step_size=0.5,
            world_size=2,
        )
        first_x = model.x.detach().clone()
        _lap_moo_training.save_moo_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            protocol_metadata=metadata,
            sampling_manifest=manifest,
            next_update=1,
            aggregator_state=first.aggregator_state,
        )
        expected_second = run_armijo(
            model,
            factories(model, second=True),
            initial_step_size=0.5,
            aggregator_state=first.aggregator_state,
            world_size=2,
        )
        expected = {
            "model": copy.deepcopy(model.state_dict()),
            "ema": copy.deepcopy(expected_second.aggregator_state),
            "rng_draws": (random.random(), float(np.random.random()), torch.rand(2)),
        }

        random.seed(900 + rank)
        np.random.seed(1000 + rank)
        torch.manual_seed(1100 + rank)
        resumed_model = ScalarModel()
        cursor, resumed_ema = _lap_moo_training.load_moo_checkpoint(
            checkpoint_path,
            model=resumed_model,
            optimizer=None,
            scheduler=None,
            expected_protocol_metadata=metadata,
            sampling_manifest=manifest,
            restore_rng=True,
        )
        resumed_second = run_armijo(
            resumed_model,
            factories(resumed_model, second=True),
            initial_step_size=0.5,
            aggregator_state=resumed_ema,
            world_size=2,
        )
        resumed = {
            "cursor": cursor,
            "model": copy.deepcopy(resumed_model.state_dict()),
            "ema": copy.deepcopy(resumed_second.aggregator_state),
            "rng_draws": (random.random(), float(np.random.random()), torch.rand(2)),
        }
        torch.save(
            {
                "first": {
                    "accepted": first.accepted,
                    "details": vector_diagnostics(first),
                    "losses": first.losses,
                    "x": first_x,
                    "ema": first.aggregator_state,
                },
                "expected": expected,
                "resumed": resumed,
            },
            output_prefix + str(rank),
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU Gloo is unavailable in this PyTorch build.",
)
def test_two_rank_gloo_uses_global_trial_means_and_resumes_exactly(tmp_path):
    manifest, metadata, config = _checkpoint_manifest(world_size=2, initial_step_size=0.5)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output_prefix = str(tmp_path / "gloo-rank-")
    checkpoint = str(tmp_path / "gloo-direct-armijo.pt")
    torch.multiprocessing.spawn(
        _two_rank_global_trial_resume_worker,
        args=(port, output_prefix, checkpoint, manifest, metadata, config),
        nprocs=2,
        join=True,
    )
    outputs = [
        torch.load(f"{output_prefix}{rank}", map_location="cpu", weights_only=False)
        for rank in range(2)
    ]

    expected_initial_mean = 2.0
    expected_accepted_mean = 0.875
    for output in outputs:
        first = output["first"]
        assert first["accepted"] is True
        assert first["details"]["accepted_t"] == 0.5
        assert [trial["t"] for trial in first["details"]["trials"]] == [1.0, 0.5]
        assert first["losses"] == {task: expected_initial_mean for task in TASKS}
        assert first["details"]["trials"][0]["losses"] == {
            task: expected_initial_mean for task in TASKS
        }
        assert first["details"]["trials"][1]["losses"] == {
            task: expected_accepted_mean for task in TASKS
        }
        assert first["ema"]["t"] == 1
        assert torch.equal(first["x"], torch.tensor([1.75], dtype=torch.float64))
        assert output["resumed"]["cursor"] == 1
        assert output["expected"]["ema"] == output["resumed"]["ema"]
        for name, expected_value in output["expected"]["model"].items():
            assert torch.equal(output["resumed"]["model"][name], expected_value)
        assert output["expected"]["rng_draws"][:2] == output["resumed"]["rng_draws"][:2]
        torch.testing.assert_close(
            output["expected"]["rng_draws"][2],
            output["resumed"]["rng_draws"][2],
            rtol=0,
            atol=0,
        )

    # Local rank 0 gets worse while rank 1 gets better; shared Armijo decisions
    # are made from the global per-objective means above.
    initial_local = (0.5, 3.5)
    accepted_local = (1.53125, 0.21875)
    assert accepted_local[0] > initial_local[0]
    assert accepted_local[1] < initial_local[1]
    assert sum(accepted_local) / 2 < sum(initial_local) / 2
    assert outputs[0]["first"]["details"]["directional_slopes"] == outputs[1]["first"]["details"]["directional_slopes"]
    assert outputs[0]["first"]["ema"] == outputs[1]["first"]["ema"]


def _minimal_driver_args(*extra):
    return [
        "train_lap_moo.py",
        "--predopt-checkpoint",
        "predopt.pt",
        "--minnesota-store-manifest",
        "groups.json",
        "--central-data-dir",
        "central",
        "--ao-cache-dir",
        "cache",
        "--sampling-manifest",
        "sampling.json",
        "--output-dir",
        "out",
        "--method",
        "pcd",
        "--updates",
        "1",
        "--learning-rate",
        "0.1",
        *extra,
    ]


@pytest.mark.parametrize(
    ("extra", "expected_weight_decay"),
    [
        ((), 1.0e-2),
        (("--step-rule", ARMIJO_RULE), 0.0),
        (("--step-rule", ARMIJO_RULE, "--weight-decay", "0"), 0.0),
    ],
)
def test_cli_normalizes_decay_without_changing_historical_optimizer_default(
    monkeypatch, extra, expected_weight_decay
):
    class StopBeforeDataLoad(RuntimeError):
        pass

    observed = []

    def stop_at_runtime(device):
        observed.append(sys._getframe(1).f_locals["args"].weight_decay)
        raise StopBeforeDataLoad

    monkeypatch.setattr(sys, "argv", _minimal_driver_args(*extra))
    monkeypatch.setattr(_train_lap_moo_driver, "_distributed_runtime", stop_at_runtime)
    with pytest.raises(StopBeforeDataLoad):
        _train_lap_moo_driver.main()
    assert observed == [expected_weight_decay]


def test_cli_rejects_nonzero_decay_for_direct_armijo_before_runtime(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_driver_args("--step-rule", ARMIJO_RULE, "--weight-decay", "0.01"),
    )
    monkeypatch.setattr(
        _train_lap_moo_driver,
        "_distributed_runtime",
        lambda _device: pytest.fail("runtime must not start with incompatible direct-mode decay"),
    )
    with pytest.raises(ValueError, match="does not accept weight decay"):
        _train_lap_moo_driver.main()


def test_rejected_first_update_can_be_checkpointed_and_resumed_at_cursor_zero(tmp_path):
    manifest, metadata, config = _checkpoint_manifest()
    model = ScalarModel()
    before = copy.deepcopy(model.state_dict())
    result = run_armijo(
        model,
        conflicting_factories(model),
        initial_step_size=config["initial_step_size"],
    )
    assert result.accepted is False
    assert result.aggregator_state is None
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)

    checkpoint = tmp_path / "stopped-before-first-acceptance.pt"
    _lap_moo_training.save_moo_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        protocol_metadata=metadata,
        sampling_manifest=manifest,
        next_update=0,
        aggregator_state=None,
    )
    resumed_model = ScalarModel()
    cursor, ema = _lap_moo_training.load_moo_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=None,
        scheduler=None,
        expected_protocol_metadata=metadata,
        sampling_manifest=manifest,
        restore_rng=True,
    )
    assert cursor == 0
    assert ema["t"] == 0
    assert ema["v"] == [0.0, 0.0, 0.0]
    for name, value in before.items():
        assert torch.equal(resumed_model.state_dict()[name], value)
