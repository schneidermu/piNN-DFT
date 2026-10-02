"""Focused correctness tests for the isolated one-stage MOO training core."""

from __future__ import annotations

import copy
import json
import random
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import optuna_joint as _REAL_OPTUNA_JOINT
from lap_moo_protocol import (
    LEGACY_READ_ONLY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    SAMPLING_PROTOCOL_VERSION,
    SamplingStream,
    build_sampling_manifest,
    build_sampling_manifest_from_catalog,
    canonical_sha256,
    make_protocol_metadata,
    validate_cursor,
    validate_protocol_metadata,
    validate_sampling_manifest,
)
from lap_moo_training import (
    AOFactorChunk,
    MRKSOperatorSystem,
    average_raw_task_gradients,
    compute_isolated_task_gradients,
    load_moo_checkpoint,
    make_cosine_scheduler,
    make_mrks_objective_factories,
    materialize_task_zeros,
    named_trainable_parameters,
    save_moo_checkpoint,
    train_moo_update,
)
from train_lap_moo import REPO_ROOT as DRIVER_REPO_ROOT
from train_lap_moo import _distributed_runtime, _external_path, _run_scheduled_update


def _has_visible_cuda_device() -> bool:
    # On some Windows builds is_available() can be true with
    # CUDA_VISIBLE_DEVICES='' even though device_count() is zero.
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _reaction(database: str, reaction_id: int, suffix: str) -> dict:
    component_suffix = "" if suffix == "default" else f"__{suffix}"
    return {
        "Database": database,
        "ReactionID": reaction_id,
        "component_paths": [f"A{reaction_id}{component_suffix}.npz"],
    }


def _small_grouped() -> dict[int, list[dict]]:
    return {
        0: [_reaction("ABDE4", 1, "default"), _reaction("ABDE4", 1, "alt")],
        1: [_reaction("AE17", 2, "default"), _reaction("AE17", 2, "alt")],
        2: [_reaction("DBH76", 3, "default"), _reaction("DBH76", 3, "alt")],
    }


def _frozen_production_catalog_v1() -> tuple[list[dict], list[str], dict[str, str]]:
    """Compact input fixture for the immutable shared 150-update stream."""
    variants = [
        "level2",
        "level2_delley",
        "level2_gauss_chebyshev",
        "level2_mura",
        "level3",
        "level3_delley",
        "level3_gauss_chebyshev",
        "level3_mura",
    ]
    reaction_catalog = [
        {"database": database, "reaction_id": reaction_id, "variants": variants}
        for database, reaction_ids in (
            ("ABDE4", (0, 2, 3)),
            ("AE17", (0, 8, 16)),
            ("DBH76", (0, 37, 75)),
            ("EA13", (0, 6, 12)),
            ("IP13", (0, 6, 12)),
            ("MGAE109", (0, 56, 108)),
            ("NCCE31", (0, 15, 29)),
            ("PA8", (0, 4, 7)),
            ("pTC13", (0, 6, 12)),
        )
        for reaction_id in reaction_ids
    ]
    systems = [
        "AlBeH", "BH", "BeH2", "C2H2_iso2", "CH2O", "CH4", "CO", "ClH",
        "ClHS", "H2", "H2O", "H4Si", "HLi", "HPSi_iso2", "N2",
    ]
    source_hashes = {
        "ao_factor_cache_manifest": "60631c23d1683dcf7855ef5897464994addb990c1c8cf33ea2572adbf532a00a",
        "central_operator_manifest": "7005cd869ea8be9636b03f385e7069f4e9023c9582fe7defc5437a7d6609b887",
        "minnesota_group_store_manifest": "ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a",
        "mrks_dispersions": "a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440",
        "panel_definition": "f98c112344dc52df401fb5eff50fd8ce609cc9c8f69085223cc9c0dee9c66131",
        "predopt_checkpoint": "ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8",
        "reaction_dispersions": "f7bd56d6b8ad133b7729dbb9f47013bf2c9d0d14fcb41296ed7a4df100855927",
    }
    return reaction_catalog, systems, source_hashes


def _hashes() -> dict[str, str]:
    return {
        name: f"{index:064x}"
        for index, name in enumerate(
            ("mn", "mrks", "predopt", "manifest", "cache", "reaction_d3", "mrks_d3"), 1
        )
    }


def _metadata(manifest_sha: str) -> dict:
    hashes = _hashes()
    return make_protocol_metadata(
        architecture="tiny-test-architecture",
        method="fixed",
        method_hyperparameters={"fixed_weights": [1.0, 1.0, 1.0]},
        fixed_scalarization={"calibration": "test-only", "fixed_weights": [1.0, 1.0, 1.0]},
        optimizer={"name": "RAdamW", "lr": 0.01, "weight_decay": 0.01},
        lr_schedule={"name": "cosine", "total_updates": 4, "min_lr_ratio": 0.1},
        predopt_checkpoint_sha256=hashes["predopt"],
        sampling_manifest_sha256=manifest_sha,
        minnesota_data_sha256=hashes["mn"],
        operator_corpus_manifest_sha256=hashes["mrks"],
        ao_cache_manifest_sha256=hashes["cache"],
        reaction_dispersions_sha256=hashes["reaction_d3"],
        mrks_dispersions_sha256=hashes["mrks_d3"],
        random_seed=41,
        dtype="torch.float64",
        grid_chunk_size=16,
        ao_cache_chunk_size=4096,
    )


class _CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, dtype=torch.float64)
        self.model_kwargs = {"width": 2}

    def forward(self, value):
        return self.linear(value)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def test_sampling_manifest_is_reproducible_balanced_and_rank_aware():
    systems = ["H2", "CO", "BeH2"]
    hashes = _hashes()
    first = build_sampling_manifest(
        _small_grouped(), systems, updates=12, seed=41, source_hashes=hashes, world_size=2
    )
    second = build_sampling_manifest(
        _small_grouped(), systems, updates=12, seed=41, source_hashes=hashes, world_size=2
    )
    catalog_manifest = build_sampling_manifest_from_catalog(
        [
            {"database": "DBH76", "reaction_id": 3, "variants": ["default", "alt"]},
            {"database": "ABDE4", "reaction_id": 1, "variants": ["default", "alt"]},
            {"database": "AE17", "reaction_id": 2, "variants": ["default", "alt"]},
        ],
        systems,
        updates=12,
        seed=41,
        source_hashes=hashes,
        world_size=2,
    )
    assert first == second
    assert first == catalog_manifest
    digest = validate_sampling_manifest(first)
    stream = SamplingStream(first)
    for rank in range(2):
        rank_entries = [stream.entry(step, rank) for step in range(3)]
        assert {(x["reaction"]["database"], x["reaction"]["reaction_id"]) for x in rank_entries} == {
            ("ABDE4", 1), ("AE17", 2), ("DBH76", 3)
        }
        assert {x["mrks_system"] for x in rank_entries} == set(systems)
        assert all(x["variant_suffix"] in {"default", "alt"} for x in rank_entries)
    cursor = {"sampling_manifest_sha256": digest, "next_update": 5}
    assert validate_cursor(first, cursor) == 5
    cursor["next_update"] = 13
    with pytest.raises(ValueError, match="outside"):
        validate_cursor(first, cursor)
    tampered = json.loads(json.dumps(first))
    tampered["entries"][0]["per_rank"][0]["mrks_system"] = "unknown"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_sampling_manifest(tampered)


def test_v2_metadata_rebuilds_the_immutable_v1_sampling_stream():
    catalog, systems, source_hashes = _frozen_production_catalog_v1()
    manifest = build_sampling_manifest_from_catalog(
        catalog,
        systems,
        updates=150,
        seed=41,
        source_hashes=source_hashes,
        world_size=1,
    )
    assert SAMPLING_PROTOCOL_VERSION == LEGACY_READ_ONLY_PROTOCOL_VERSION
    assert manifest["manifest_sha256"] == (
        "550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928"
    )
    metadata = _metadata(manifest["manifest_sha256"])
    assert metadata["protocol_version"] == PROTOCOL_VERSION
    assert metadata["ao_cache_chunk_size"] == 4096
    validate_protocol_metadata(metadata)


def test_protocol_is_explicitly_one_stage_and_rejects_legacy_control_keys():
    manifest = build_sampling_manifest(
        _small_grouped(), ["H2"], updates=1, seed=41, source_hashes=_hashes()
    )
    metadata = _metadata(manifest["manifest_sha256"])
    validate_protocol_metadata(metadata)
    assert metadata["protocol_version"] == PROTOCOL_VERSION
    assert metadata["one_stage_main_training"] is True
    lowered = {key.lower() for key in _walk_keys(metadata)}
    assert not any("omega" in key or key.startswith("s5_") or "phase_schedule" in key for key in lowered)
    incompatible = dict(metadata)
    incompatible["s5_phases"] = []
    with pytest.raises(ValueError, match="phase schedule"):
        validate_protocol_metadata(incompatible)
    for alias in ("objective_weights_by_epoch", "EpochDependentTaskWeights"):
        nested = dict(metadata)
        nested["method_hyperparameters"] = {alias: [1.0, 1.0, 1.0]}
        with pytest.raises(ValueError, match="Epoch-dependent task weights"):
            validate_protocol_metadata(nested)


def test_protocol_v2_hashes_ao_cache_chunk_and_keeps_v1_read_only_explicit():
    manifest = build_sampling_manifest(
        _small_grouped(), ["H2"], updates=1, seed=41, source_hashes=_hashes()
    )
    metadata = _metadata(manifest["manifest_sha256"])
    assert metadata["ao_cache_chunk_size"] == 4096
    alternate = copy.deepcopy(metadata)
    alternate["ao_cache_chunk_size"] = 1024
    assert canonical_sha256(metadata) != canonical_sha256(alternate)
    with pytest.raises(ValueError, match="ao_cache_chunk_size"):
        make_protocol_metadata(
            architecture="tiny-test-architecture",
            method="fixed",
            method_hyperparameters={"fixed_weights": [1.0, 1.0, 1.0]},
            fixed_scalarization={"fixed_weights": [1.0, 1.0, 1.0]},
            optimizer={"name": "RAdamW"},
            lr_schedule={"name": "cosine", "total_updates": 1},
            predopt_checkpoint_sha256=_hashes()["predopt"],
            sampling_manifest_sha256=manifest["manifest_sha256"],
            minnesota_data_sha256=_hashes()["mn"],
            operator_corpus_manifest_sha256=_hashes()["mrks"],
            ao_cache_manifest_sha256=_hashes()["cache"],
            reaction_dispersions_sha256=_hashes()["reaction_d3"],
            mrks_dispersions_sha256=_hashes()["mrks_d3"],
            random_seed=41,
            dtype="float32",
            grid_chunk_size=16,
            ao_cache_chunk_size=1024.0,
        )

    legacy = copy.deepcopy(metadata)
    legacy["protocol_version"] = LEGACY_READ_ONLY_PROTOCOL_VERSION
    legacy.pop("ao_cache_chunk_size")
    with pytest.raises(ValueError, match="protocol version"):
        validate_protocol_metadata(legacy)
    validate_protocol_metadata(legacy, allow_v1_read_only=True)
    retrofitted = copy.deepcopy(legacy)
    retrofitted["ao_cache_chunk_size"] = 4096
    with pytest.raises(ValueError, match="cannot be retrofitted"):
        validate_protocol_metadata(retrofitted, allow_v1_read_only=True)


def test_driver_output_paths_must_resolve_outside_the_repository(tmp_path):
    with pytest.raises(ValueError, match="outside the repository"):
        _external_path(DRIVER_REPO_ROOT / "pilot-output", "output directory")
    outside = tmp_path / "pilot-output"
    assert _external_path(outside, "output directory") == outside.resolve()


@pytest.mark.skipif(not _has_visible_cuda_device(), reason="CUDA runtime selection requires a visible CUDA device")
def test_auto_cuda_runtime_sets_device_zero(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    device, rank, local_rank, world_size = _distributed_runtime("auto")
    assert device == torch.device("cuda:0")
    assert (rank, local_rank, world_size) == (0, 0, 1)


def test_isolated_task_gradients_match_three_independent_backward_calls():
    torch.manual_seed(4)
    model = _CheckpointModel()
    x = torch.tensor([[0.2, -0.4], [0.7, 0.5]], dtype=torch.float64)
    targets = {
        "chem": torch.tensor([[0.1], [0.8]], dtype=torch.float64),
        "exc": torch.tensor([[0.2], [-0.2]], dtype=torch.float64),
        "op": torch.tensor([[-0.7], [0.3]], dtype=torch.float64),
    }
    factories = {
        name: (lambda name=name: (model(x) - targets[name]).square().mean())
        for name in ("chem", "exc", "op")
    }
    _losses, gradients = compute_isolated_task_gradients(model, factories)
    parameters = named_trainable_parameters(model)
    for task in ("chem", "exc", "op"):
        model.zero_grad(set_to_none=True)
        factories[task]().backward()
        for name, parameter in parameters.items():
            torch.testing.assert_close(gradients[task][name], parameter.grad)


def test_unused_task_parameters_are_zero_before_allreduce():
    class PartiallyUsed(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.unused = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

    model = PartiallyUsed()
    factories = {
        "chem": lambda: model.used.square(),
        "exc": lambda: (model.used + 1).square(),
        "op": lambda: (model.used - 1).square(),
    }
    _, sparse = compute_isolated_task_gradients(model, factories)
    dense = materialize_task_zeros(model, sparse)
    for task in ("chem", "exc", "op"):
        assert torch.equal(dense[task]["unused"], torch.zeros_like(model.unused))
    averaged = average_raw_task_gradients(dense, world_size=1)
    for task in ("chem", "exc", "op"):
        assert torch.equal(averaged[task]["unused"], torch.zeros_like(model.unused))


def test_one_update_calls_each_task_once_and_applies_one_joint_gradient():
    model = _CheckpointModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    calls = []

    def aggregator(task_grads, *, method, hyperparameters, state):
        calls.append((method, tuple(task_grads), hyperparameters, state))
        joint = {
            name: sum(task_grads[task][name] for task in ("chem", "exc", "op")) / 3
            for name in task_grads["chem"]
        }
        return joint, {"solver_status": "test"}, {"updates": 1}

    factories = {
        "chem": lambda: model.linear.weight.square().sum(),
        "exc": lambda: (model.linear.weight - 1).square().sum(),
        "op": lambda: (model.linear.weight + 1).square().sum(),
    }
    result = train_moo_update(
        model,
        optimizer,
        factories,
        method="fixed",
        hyperparameters={"fixed_weights": [1.0, 1.0, 1.0]},
        aggregator_state={"before": True},
        aggregator=aggregator,
        record_update_geometry=True,
    )
    assert tuple(result.losses) == ("chem", "exc", "op")
    assert calls == [("fixed", ("chem", "exc", "op"), {"fixed_weights": [1.0, 1.0, 1.0]}, {"before": True})]
    assert result.aggregator_state == {"updates": 1}
    assert result.diagnostics["parameter_update_norm"] > 0
    assert "task_update_dots" in result.diagnostics


def test_cosine_schedule_is_shared_smooth_and_bounded():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = make_cosine_scheduler(optimizer, total_updates=4, min_lr_ratio=0.2)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.02)
    assert all(left >= right for left, right in pairwise(rates))
    assert all(np.isfinite(rate) for rate in rates)


def test_mrks_objectives_preserve_dispersion_energy_and_hfree_operator_loss(monkeypatch):
    import lap_moo_training
    from lap_operator import assemble_rks_operator, operator_loss
    from lap_vxc import integrated_energy

    # Another historical test module installs an import-time optuna_joint
    # stub. Pin the real dependency for this test and for the objective's lazy
    # imports, then let monkeypatch restore the suite's prior module mapping.
    monkeypatch.setitem(sys.modules, "optuna_joint", _REAL_OPTUNA_JOINT)
    batch_exc = _REAL_OPTUNA_JOINT.batch_exc

    class PolynomialEnergy(nn.Module):
        descriptor_protocol = "rho-sigma-total-lapl-tau-free-v1"

        def __init__(self):
            super().__init__()
            self.coefficients = nn.Parameter(
                torch.tensor([0.7, -0.2, 0.3, 0.11], dtype=torch.float64)
            )

        def forward(self, rho, sigma, lapl):
            c = self.coefficients
            return c[0] * rho.sum(-1) + c[1] * sigma.sum(-1) + c[2] * lapl.sum(-1) + c[3] * lapl.square().sum(-1)

    monkeypatch.setattr(lap_moo_training, "LapEnergy", lambda model: model)
    torch.manual_seed(17)
    model = PolynomialEnergy()
    dtype = torch.float64
    features = torch.randn(4, 10, dtype=dtype)
    features[:, :2] = features[:, :2].abs() + 0.5
    weights = torch.tensor([0.1, 0.2, 0.15, 0.07], dtype=dtype)
    phi = torch.randn(2, 3, dtype=dtype)
    grad_phi = torch.randn(2, 3, 3, dtype=dtype)
    lap_phi = torch.randn(2, 3, dtype=dtype)
    chunks = (
        AOFactorChunk(slice(0, 2), phi, grad_phi, lap_phi),
        AOFactorChunk(slice(2, 4), phi * 0.8, grad_phi * 1.1, lap_phi * 0.9),
    )
    overlap = torch.eye(3, dtype=dtype) * 1.3
    reference = torch.randn(3, 3, dtype=dtype)
    reference = (reference + reference.T) * 0.5
    system = MRKSOperatorSystem(
        name="H2",
        features=features,
        weights=weights,
        exc_target=torch.tensor(-0.4, dtype=dtype),
        reference_operator=reference,
        overlap=overlap,
        ao_chunks=chunks,
    )
    exc_factory, op_factory = make_mrks_objective_factories(
        model,
        system,
        point_chunk_size=2,
        dispersions={"H2": 0.025},
    )

    raw_energy = integrated_energy(model, features[:, None, :], weights, 2)
    expected_exc = batch_exc(
        ["H2"],
        (raw_energy + 0.025).reshape(1),
        system.exc_target.reshape(1),
    )
    expected_operator = reference.new_zeros(reference.shape)
    for chunk in chunks:
        expected_operator = expected_operator + assemble_rks_operator(
            model,
            features[chunk.rows],
            weights[chunk.rows],
            chunk.phi,
            chunk.grad_phi,
            chunk.lap_phi,
            chunk_size=2,
        )
    expected_op = operator_loss(expected_operator, reference, overlap)
    torch.testing.assert_close(exc_factory(), expected_exc)
    torch.testing.assert_close(op_factory(), expected_op)
    for factory in (exc_factory, op_factory):
        gradient = torch.autograd.grad(factory(), model.coefficients)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_checkpoint_roundtrip_restores_cursor_optimizer_rng_and_protocol(tmp_path):
    grouped = _small_grouped()
    manifest = build_sampling_manifest(
        grouped, ["H2", "CO"], updates=4, seed=41, source_hashes=_hashes()
    )
    metadata = _metadata(manifest["manifest_sha256"])
    torch.manual_seed(91)
    random.seed(91)
    np.random.seed(91)
    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scheduler = make_cosine_scheduler(optimizer, total_updates=4)
    loss = model(torch.tensor([[1.0, 0.0]], dtype=torch.float64)).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}

    mismatched_metadata = dict(metadata)
    mismatched_metadata["sampling_manifest_sha256"] = _hashes()["manifest"]
    with pytest.raises(ValueError, match="sampling manifest identities differ"):
        save_moo_checkpoint(
            tmp_path / "mismatched.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            protocol_metadata=mismatched_metadata,
            sampling_manifest=manifest,
            next_update=1,
            aggregator_state={},
        )

    checkpoint = tmp_path / "moo.pt"
    save_moo_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        protocol_metadata=metadata,
        sampling_manifest=manifest,
        next_update=1,
        aggregator_state={"alpha": [0.2, 0.3, 0.5], "updates": 1},
    )
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    alternate_chunk_metadata = copy.deepcopy(metadata)
    alternate_chunk_metadata["ao_cache_chunk_size"] = 1024
    assert canonical_sha256(metadata) != canonical_sha256(alternate_chunk_metadata)
    mismatch_model = _CheckpointModel()
    mismatch_parameters = {
        key: value.detach().clone() for key, value in mismatch_model.state_dict().items()
    }
    mismatch_optimizer = torch.optim.AdamW(mismatch_model.parameters(), lr=0.003)
    mismatch_scheduler = make_cosine_scheduler(mismatch_optimizer, total_updates=4)
    mismatch_optimizer_groups = copy.deepcopy(mismatch_optimizer.state_dict()["param_groups"])
    mismatch_scheduler_state = copy.deepcopy(mismatch_scheduler.state_dict())
    with pytest.raises(ValueError, match="protocol differs"):
        load_moo_checkpoint(
            checkpoint,
            model=mismatch_model,
            optimizer=mismatch_optimizer,
            scheduler=mismatch_scheduler,
            expected_protocol_metadata=alternate_chunk_metadata,
            sampling_manifest=manifest,
            restore_rng=False,
        )
    assert not mismatch_optimizer.state
    assert mismatch_optimizer.state_dict()["param_groups"] == mismatch_optimizer_groups
    assert mismatch_scheduler.state_dict() == mismatch_scheduler_state
    for name, value in mismatch_model.state_dict().items():
        torch.testing.assert_close(value, mismatch_parameters[name])

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "lap-moo-one-stage"
    assert not any(key.lower() in {"h", "h_bohr", "stencil", "stencil_order", "stencil_version"} for key in _walk_keys(payload))
    assert not any("omega" in key.lower() or key.lower().startswith("s5_") for key in _walk_keys(payload["protocol_metadata"]))

    restored_model = _CheckpointModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.003)
    restored_scheduler = make_cosine_scheduler(restored_optimizer, total_updates=4)
    cursor, aggregator_state = load_moo_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_protocol_metadata=metadata,
        sampling_manifest=manifest,
        restore_rng=True,
    )
    assert cursor == 1
    assert aggregator_state == {"alpha": [0.2, 0.3, 0.5], "updates": 1}
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)

    wrong_manifest = build_sampling_manifest(
        grouped, ["H2", "CO"], updates=4, seed=42, source_hashes=_hashes()
    )
    with pytest.raises(ValueError, match="sampling manifest identities differ"):
        load_moo_checkpoint(
            checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            expected_protocol_metadata=metadata,
            sampling_manifest=wrong_manifest,
            restore_rng=False,
        )

    wrong_metadata = dict(metadata)
    wrong_metadata["method"] = "nash_mtl"
    wrong_metadata["fixed_scalarization"] = None
    incompatible_model = _CheckpointModel()
    with pytest.raises(ValueError, match="differs"):
        load_moo_checkpoint(
            checkpoint,
            model=incompatible_model,
            optimizer=torch.optim.AdamW(incompatible_model.parameters(), lr=0.003),
            scheduler=None,
            expected_protocol_metadata=wrong_metadata,
            sampling_manifest=manifest,
            restore_rng=False,
        )

    corrupt_payload = dict(payload)
    corrupt_payload["operator_metadata"] = dict(payload["operator_metadata"])
    corrupt_payload["operator_metadata"]["stencil"] = {"version": "invalid"}
    corrupt_checkpoint = tmp_path / "corrupt.pt"
    torch.save(corrupt_payload, corrupt_checkpoint)
    compatible_model = _CheckpointModel()
    compatible_optimizer = torch.optim.AdamW(compatible_model.parameters(), lr=0.003)
    compatible_scheduler = make_cosine_scheduler(compatible_optimizer, total_updates=4)
    with pytest.raises(ValueError, match="spatial-stencil"):
        load_moo_checkpoint(
            corrupt_checkpoint,
            model=compatible_model,
            optimizer=compatible_optimizer,
            scheduler=compatible_scheduler,
            expected_protocol_metadata=metadata,
            sampling_manifest=manifest,
            restore_rng=False,
        )


def test_cli_scheduled_update_and_checkpoint_cursor_advance_together(tmp_path):
    manifest = build_sampling_manifest(
        _small_grouped(), ["H2"], updates=4, seed=41, source_hashes=_hashes()
    )
    metadata = _metadata(manifest["manifest_sha256"])
    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scheduler = make_cosine_scheduler(optimizer, total_updates=4)
    values = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    objectives = {
        task: (lambda task_scale=scale: model(values).square().sum() * task_scale)
        for task, scale in (("chem", 1.0), ("exc", 2.0), ("op", 3.0))
    }
    hyperparameters = {"fixed_weights": [1.0, 1.0, 1.0]}

    result = _run_scheduled_update(
        model,
        optimizer,
        objectives,
        method="fixed",
        hyperparameters=hyperparameters,
        aggregator_state={},
        scheduler=scheduler,
        world_size=1,
    )
    assert scheduler.last_epoch == 1
    assert result.learning_rate == pytest.approx(optimizer.param_groups[0]["lr"])
    checkpoint = tmp_path / "scheduled.pt"
    save_moo_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        protocol_metadata=metadata,
        sampling_manifest=manifest,
        next_update=1,
        aggregator_state=result.aggregator_state,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["sampling_cursor"]["next_update"] == 1
    assert payload["scheduler_state_dict"]["last_epoch"] == 1

    restored_model = _CheckpointModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.003)
    restored_scheduler = make_cosine_scheduler(restored_optimizer, total_updates=4)
    cursor, _ = load_moo_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_protocol_metadata=metadata,
        sampling_manifest=manifest,
        restore_rng=False,
    )
    assert cursor == restored_scheduler.last_epoch == 1
    _run_scheduled_update(
        restored_model,
        restored_optimizer,
        objective_factories={
            task: (
                lambda task_scale=scale: restored_model(values).square().sum() * task_scale
            )
            for task, scale in (("chem", 1.0), ("exc", 2.0), ("op", 3.0))
        },
        method="fixed",
        hyperparameters=hyperparameters,
        aggregator_state={},
        scheduler=restored_scheduler,
        world_size=1,
    )
    assert restored_scheduler.last_epoch == 2

    stale_scheduler_payload = dict(payload)
    stale_scheduler_payload["scheduler_state_dict"] = dict(payload["scheduler_state_dict"])
    stale_scheduler_payload["scheduler_state_dict"]["last_epoch"] = 0
    stale_scheduler = tmp_path / "stale_scheduler.pt"
    torch.save(stale_scheduler_payload, stale_scheduler)
    stale_model = _CheckpointModel()
    stale_optimizer = torch.optim.AdamW(stale_model.parameters(), lr=0.003)
    stale_lr_scheduler = make_cosine_scheduler(stale_optimizer, total_updates=4)
    with pytest.raises(ValueError, match="scheduler progress does not match"):
        load_moo_checkpoint(
            stale_scheduler,
            model=stale_model,
            optimizer=stale_optimizer,
            scheduler=stale_lr_scheduler,
            expected_protocol_metadata=metadata,
            sampling_manifest=manifest,
            restore_rng=False,
        )


@pytest.mark.skipif(not _has_visible_cuda_device(), reason="CUDA checkpoint restoration requires a visible CUDA device")
def test_cuda_checkpoint_restore_keeps_rng_tensors_on_cpu(tmp_path):
    torch.manual_seed(111)
    random.seed(222)
    np.random.seed(333)
    model = _CheckpointModel().to(device="cuda", dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scheduler = make_cosine_scheduler(optimizer, total_updates=4)
    manifest = build_sampling_manifest(
        _small_grouped(), ["H2"], updates=4, seed=41, source_hashes=_hashes()
    )
    metadata = _metadata(manifest["manifest_sha256"])
    checkpoint = tmp_path / "cuda.pt"
    save_moo_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        protocol_metadata=metadata,
        sampling_manifest=manifest,
        next_update=0,
        aggregator_state={},
    )
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1, device="cuda")

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    restored = _CheckpointModel().to(device="cuda", dtype=torch.float64)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003)
    restored_scheduler = make_cosine_scheduler(restored_optimizer, total_updates=4)
    load_moo_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_protocol_metadata=metadata,
        sampling_manifest=manifest,
        map_location="cuda",
        restore_rng=True,
    )
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1, device="cuda"), expected_torch)


def _two_rank_worker(rank: int, rendezvous: str, output_prefix: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    seen = {}

    def aggregator(task_grads, *, method, hyperparameters, state):
        seen.update(
            {task: task_grads[task]["weight"].clone() for task in task_grads}
        )
        joint = {
            "weight": sum(
                task_grads[task]["weight"] for task in ("chem", "exc", "op")
            )
            / 3
        }
        return joint, {}, {}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    task_scales = {
        "chem": float(rank + 1),
        "exc": float(2 * (rank + 1)),
        "op": float(3 - rank),
    }
    factories = {
        task: (lambda scale=scale: model.weight.sum() * scale)
        for task, scale in task_scales.items()
    }
    result = train_moo_update(
        model,
        optimizer,
        factories,
        method="fixed",
        aggregator=aggregator,
        world_size=2,
        record_update_geometry=False,
    )
    torch.save(
        {"seen": seen, "weight": model.weight.detach(), "losses": result.losses},
        output_prefix + str(rank),
    )
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="Gloo file rendezvous regression runs on Linux.")
def test_two_rank_allreduces_each_raw_task_before_moo(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "rank")),
        nprocs=2,
    )
    expected = {"chem": 1.5, "exc": 3.0, "op": 2.5}
    for rank in range(2):
        payload = torch.load(tmp_path / f"rank{rank}", map_location="cpu", weights_only=False)
        for task, value in expected.items():
            torch.testing.assert_close(
                payload["seen"][task], torch.tensor([[value]], dtype=torch.float64)
            )
        torch.testing.assert_close(payload["weight"], torch.tensor([[1.0 - 0.01 * (7.0 / 3)]], dtype=torch.float64))
