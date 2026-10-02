"""Focused optimizer-group, scheduling, and resume tests for Lap MOO."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lap_moo_optimizers import (
    MuonAdamW,
    _ensure_native_muon_bfloat16,
    build_optimizer,
    partition_muon_parameters,
)
from lap_moo_protocol import make_protocol_metadata, validate_protocol_metadata
from lap_moo_training import make_cosine_scheduler
from NN_models_lap import pcPBELMLOptimizerV2Lap


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))

    def forward(self, value):
        return self.fc(value)


class _TinyRoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_input_layers = nn.Sequential(nn.Linear(3, 4, bias=False), nn.LayerNorm(4))
        self.c_symmetrization_blocks = nn.Sequential(_Residual())
        self.c_post_symm_blocks = nn.Sequential(_Residual())
        self.x_feature_extractor = nn.Sequential(nn.Linear(4, 4, bias=False), _Residual())
        self.c_output_layer = nn.Linear(8, 2)
        self.x_output_layer = nn.Linear(4, 1)
        self.extra = nn.Linear(4, 4)
        self.scalar = nn.Parameter(torch.tensor(0.5))


def _parameter_ids(groups):
    return [id(parameter) for group in groups for parameter in group["params"]]


def test_muon_parameter_eligibility_uses_hidden_module_role_and_covers_all():
    model = _TinyRoleModel()
    muon, fallback = partition_muon_parameters(model)
    muon_names = {name for name, _ in muon}
    fallback_names = {name for name, _ in fallback}

    assert muon_names == {
        "c_input_layers.0.weight",
        "c_symmetrization_blocks.0.fc.0.weight",
        "c_post_symm_blocks.0.fc.0.weight",
        "x_feature_extractor.0.weight",
        "x_feature_extractor.1.fc.0.weight",
    }
    assert {
        "c_output_layer.weight",
        "c_output_layer.bias",
        "x_output_layer.weight",
        "x_output_layer.bias",
        "extra.weight",
        "extra.bias",
        "scalar",
        "c_input_layers.1.weight",
    } <= fallback_names
    assert not (muon_names & fallback_names)
    assert muon_names | fallback_names == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(parameter.ndim == 2 for _, parameter in muon)


def test_real_lap_model_sends_only_hidden_linear_weights_to_muon():
    model = pcPBELMLOptimizerV2Lap(num_layers=6, h_dim=8, use_g_x=True, use_g_c=True)
    muon, fallback = partition_muon_parameters(model)
    muon_names = {name for name, _ in muon}
    fallback_names = {name for name, _ in fallback}

    assert len(model.state_dict()) > len(muon_names)
    assert "c_output_layer.weight" in fallback_names
    assert "x_output_layer.weight" in fallback_names
    assert "c_input_layers.0.weight" in muon_names
    assert "x_feature_extractor.0.weight" in muon_names
    assert all("bias" not in name for name in muon_names)
    assert all(parameter.ndim == 2 for _, parameter in muon)
    assert len(muon_names) + len(fallback_names) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )


def test_radamw_factory_keeps_existing_optimizer_grouping_and_type():
    model = pcPBELMLOptimizerV2Lap(num_layers=4, h_dim=4)
    optimizer, metadata = build_optimizer(
        model, family="radamw", learning_rate=0.003, weight_decay=0.01
    )
    assert isinstance(optimizer, torch.optim.RAdam)
    assert metadata["name"] == "RAdamW"
    assert len(optimizer.param_groups) == 2
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.0]
    assert metadata["learning_rate"] == 0.003


def test_muon_adamw_wrapper_schedules_and_resumes_both_optimizers():
    torch.manual_seed(19)
    model = _TinyRoleModel()
    optimizer, metadata = build_optimizer(
        model,
        family="muon_adamw",
        learning_rate=3e-4,
        muon_learning_rate=1e-2,
        weight_decay=0.01,
    )
    assert isinstance(optimizer, MuonAdamW)
    assert metadata["all_trainable_parameters_included_once"] is True
    assert metadata["adamw_fallback_learning_rate"] == pytest.approx(3e-4)
    group_ids = _parameter_ids(optimizer.param_groups)
    assert len(group_ids) == len(set(group_ids)) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )
    assert len(optimizer.param_groups) == 3
    assert optimizer.learning_rates() == {
        "muon": pytest.approx(1e-2),
        "adamw_decay": pytest.approx(3e-4),
        "adamw_no_decay": pytest.approx(3e-4),
    }

    scheduler = make_cosine_scheduler(optimizer, total_updates=4)

    def update(current_model, current_optimizer, current_scheduler):
        current_optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in current_model.parameters())
        loss.backward()
        current_optimizer.step()
        current_scheduler.step()

    update(model, optimizer, scheduler)
    assert scheduler.last_epoch == 1
    assert optimizer.learning_rates()["muon"] < 1e-2
    assert optimizer.learning_rates()["adamw_decay"] < 3e-4
    saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
    saved_scheduler_state = copy.deepcopy(scheduler.state_dict())
    saved_model_state = copy.deepcopy(model.state_dict())

    resumed_model = _TinyRoleModel()
    resumed_model.load_state_dict(saved_model_state)
    resumed_optimizer, _ = build_optimizer(
        resumed_model,
        family="muon_adamw",
        learning_rate=3e-4,
        muon_learning_rate=1e-2,
        weight_decay=0.01,
    )
    resumed_scheduler = make_cosine_scheduler(resumed_optimizer, total_updates=4)
    resumed_optimizer.load_state_dict(saved_optimizer_state)
    resumed_scheduler.load_state_dict(saved_scheduler_state)

    update(model, optimizer, scheduler)
    update(resumed_model, resumed_optimizer, resumed_scheduler)
    assert scheduler.state_dict() == resumed_scheduler.state_dict()
    assert optimizer.learning_rates() == resumed_optimizer.learning_rates()
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, resumed_model.state_dict()[name])


def test_muon_lr_is_required_and_not_accepted_by_other_families():
    model = _TinyRoleModel()
    with pytest.raises(ValueError, match="requires a positive finite Muon learning rate"):
        build_optimizer(model, family="muon_adamw", learning_rate=1e-3)
    with pytest.raises(ValueError, match="only valid for muon_adamw"):
        build_optimizer(
            model,
            family="adamw",
            learning_rate=1e-3,
            muon_learning_rate=1e-2,
        )


def test_native_muon_rejects_pre_ampere_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 0))
    with pytest.raises(RuntimeError, match="requires CUDA BF16 matrix operations"):
        _ensure_native_muon_bfloat16(torch.device("cuda:0"))


def test_muon_protocol_records_and_rejects_parameter_partition_conflicts():
    model = _TinyRoleModel()
    _, optimizer_metadata = build_optimizer(
        model,
        family="muon_adamw",
        learning_rate=3e-4,
        muon_learning_rate=1e-2,
    )
    digest = "a" * 64

    def metadata_for(optimizer):
        return make_protocol_metadata(
            architecture="tiny-optimizer-test",
            method="nash_mtl",
            method_hyperparameters={},
            fixed_scalarization=None,
            optimizer=optimizer,
            lr_schedule={"name": "cosine", "total_updates": 4},
            predopt_checkpoint_sha256=digest,
            sampling_manifest_sha256=digest,
            minnesota_data_sha256=digest,
            operator_corpus_manifest_sha256=digest,
            ao_cache_manifest_sha256=digest,
            reaction_dispersions_sha256=digest,
            mrks_dispersions_sha256=digest,
            random_seed=41,
            dtype="float32",
            grid_chunk_size=16,
            ao_cache_chunk_size=4096,
        )

    valid = metadata_for(optimizer_metadata)
    validate_protocol_metadata(valid)
    invalid_optimizer = copy.deepcopy(optimizer_metadata)
    invalid_optimizer["adamw_fallback_parameter_names"].append(
        invalid_optimizer["muon_parameter_names"][0]
    )
    with pytest.raises(ValueError, match="disjoint complete parameter partition"):
        validate_protocol_metadata(metadata_for(invalid_optimizer))
