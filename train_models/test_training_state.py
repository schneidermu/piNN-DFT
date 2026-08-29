import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _stub_module(name, attributes):
    module = types.ModuleType(name)
    for attribute in attributes:
        setattr(module, attribute, object)
    sys.modules[name] = module


_stub_module("dataset", ("collate_fn", "fast_collate_fn_predopt"))
_stub_module(
    "NN_models",
    (
        "pcPBELMLOptimizerV2",
        "pcPBELMLOptimizerV2GcSoftplusMirror",
        "pcPBELMLOptimizerV2GcSoftplusMirrorR2ScanAlpha",
        "pcPBELMLOptimizerV2GcSveluMirror",
        "pcPBELMLOptimizerV2Log",
    ),
)
_stub_module("predopt", ("DatasetPredopt", "predopt"))
_stub_module("prepare_data", ("load_chk",))
_stub_module(
    "reaction_energy_calculation",
    ("calculate_reaction_energy", "calculate_xc_energy", "get_local_energies"),
)
_stub_module(
    "utils",
    (
        "_fix_sigma_tot_closed_shell",
        "_grid_to_model_input",
        "configure_optimizers",
        "seed_worker",
        "set_random_seed",
    ),
)

from optuna_joint import (
    GlobalCosineWithConvergenceTail,
    build_scheduler,
    completed_params_signature,
)
from training_state import (
    atomic_torch_save,
    capture_runtime_state,
    load_torch_payload,
    restore_runtime_state,
)


def _optimizer(learning_rate=3.588259475602772e-4):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([parameter], lr=learning_rate)


def test_tail_scheduler_preserves_base_lr_path_and_reaches_floor():
    base_optimizer = _optimizer()
    tail_optimizer = _optimizer()
    base = build_scheduler(base_optimizer, 500)
    tail = GlobalCosineWithConvergenceTail(
        tail_optimizer,
        base_epochs=500,
        tail_epochs=300,
        tail_start_lr=1e-5,
        tail_min_lr=1e-7,
    )
    for epoch in range(1, 501):
        base.step()
        tail.prepare_epoch(epoch)
        tail.step()
        assert tail.get_last_lr() == base.get_last_lr()

    tail.prepare_epoch(501)
    assert tail.get_last_lr() == [1e-5]
    for _ in range(300):
        tail.step()
    assert np.isclose(tail.get_last_lr()[0], 1e-7)
    tail.step()
    assert np.isclose(tail.get_last_lr()[0], 1e-7)


def test_tail_scheduler_state_resumes_identically():
    first_optimizer = _optimizer()
    first = GlobalCosineWithConvergenceTail(
        first_optimizer,
        base_epochs=500,
        tail_epochs=300,
        tail_start_lr=1e-5,
        tail_min_lr=1e-7,
    )
    for epoch in range(1, 621):
        first.prepare_epoch(epoch)
        first.step()

    second_optimizer = _optimizer()
    second_optimizer.load_state_dict(first_optimizer.state_dict())
    second = GlobalCosineWithConvergenceTail(
        second_optimizer,
        base_epochs=500,
        tail_epochs=300,
        tail_start_lr=1e-5,
        tail_min_lr=1e-7,
    )
    second.load_state_dict(first.state_dict())
    for epoch in range(621, 801):
        first.prepare_epoch(epoch)
        second.prepare_epoch(epoch)
        first.step()
        second.step()
        assert first.get_last_lr() == second.get_last_lr()


def test_runtime_state_and_atomic_payload_roundtrip(tmp_path):
    generator = torch.Generator().manual_seed(17)
    loaders = {"train_loader": SimpleNamespace(generator=generator)}
    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(7)
    state = capture_runtime_state(loaders)
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))

    path = tmp_path / "state.pt"
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    atomic_torch_save(
        {
            "runtime": state,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    assert path.is_file()
    assert not path.with_suffix(".pt.tmp").exists()
    restored = load_torch_payload(path, map_location="cpu")
    assert restored["model_state_dict"].keys() == model.state_dict().keys()
    assert restored["optimizer_state_dict"]["state"]
    restore_runtime_state(restored["runtime"], loaders)
    observed = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
    assert torch.equal(observed[3], expected[3])


def test_completed_schedule_signature_allows_only_future_extension():
    saved = {
        "lr_train": 1e-3,
        "epoch_schedule": [
            {"name": "joint", "start_epoch": 1, "end_epoch": 400, "params": {"v": 40}},
            {"name": "finish", "start_epoch": 401, "end_epoch": 800, "params": {"v": 7}},
        ],
    }
    extended = {
        "lr_train": 1e-3,
        "epoch_schedule": [
            {"name": "joint", "start_epoch": 1, "end_epoch": 400, "params": {"v": 40}},
            {"name": "finish", "start_epoch": 401, "end_epoch": 900, "params": {"v": 7}},
        ],
    }
    wrong = {
        **extended,
        "epoch_schedule": [
            {"name": "joint", "start_epoch": 1, "end_epoch": 400, "params": {"v": 80}},
            extended["epoch_schedule"][1],
        ],
    }
    assert completed_params_signature(saved, 800) == completed_params_signature(extended, 800)
    assert completed_params_signature(saved, 800) != completed_params_signature(wrong, 800)
