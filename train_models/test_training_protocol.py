"""Regression coverage for training provenance and external-only selection."""

import inspect
import json
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import optuna_joint as training
import prepare_data as mn
from dataset import get_compounds_coefs_energy, load_component_names, load_ref_energies
from prepare_vxc import prepare_vxc

EXPECTED = {
    "DBH76": {15, 35, 42, 43, 54, 55},
    "MGAE109": {18, 28, 34, 55, 74},
    "EA13": {4, 8},
    "NCCE31": {12, 21, 30},
}


def _base_reactions():
    return get_compounds_coefs_energy(
        load_component_names("data"), load_ref_energies("data")
    )


def test_exact_exclusions_and_every_other_source_row_retained():
    assert mn.DIET_HELD_OUT_MN_REACTIONS == EXPECTED
    pairs = {(db, rid) for db, ids in EXPECTED.items() for rid in ids}
    assert len(pairs) == 16
    base = _base_reactions()
    train = mn.filter_minnesota_training(base)
    source = {(r["Database"], r["ReactionID"]) for r in base.values()}
    retained = {(r["Database"], r["ReactionID"]) for r in train.values()}
    assert len(source) == len(base) == 284
    assert retained == source - pairs
    assert not retained & pairs
    assert len(train) == 268
    # Flattened keys differ from original per-database IDs.
    assert any(key != r["ReactionID"] for key, r in base.items())
    for r in base.values():
        original = load_component_names("data")[r["Database"]][r["ReactionID"]]
        np.testing.assert_array_equal(r["Components"], original["Components"])


def test_predopt_and_training_share_clean_pool(monkeypatch):
    captured = []

    def augment(pool, index):
        captured.append(pool)
        return {key: [{**r, "Grid": np.zeros((1, 12))}] for key, r in pool.items()}

    monkeypatch.setattr(mn, "build_file_index", lambda path: {})
    monkeypatch.setattr(mn, "group_and_augment_reactions", augment)
    predopt, grouped = mn.prepare("unused")
    assert len(captured) == 1
    assert len(captured[0]) == len(predopt) == len(grouped) == 268
    assert {(r["Database"], r["ReactionID"]) for r in predopt.values()} == {
        (r[0]["Database"], r[0]["ReactionID"]) for r in grouped.values()
    }
    assert "test_size" not in inspect.signature(mn.prepare).parameters


def _write_mrks(path, valid=True):
    with h5py.File(path, "w") as handle:
        handle["grid"] = np.zeros((3, 12))
        handle["vrho"] = np.ones((2, 3))
        handle["weights"] = np.ones(3)
        if valid:
            handle["E_xc"] = -1.25


def test_all_valid_mrks_train_and_stale_pickles_are_ignored(tmp_path):
    source = tmp_path / "h5"
    source.mkdir()
    for i in range(5):
        _write_mrks(source / f"system_{i}.h5")
    _write_mrks(source / "invalid.h5", valid=False)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for filename in mn.OBSOLETE_PICKLES:
        (checkpoints / filename).write_bytes(b"obsolete invalid pickle")
    base = next(r for r in _base_reactions().values() if r["Database"] == "AE17")
    mn.save_chk({0: base}, {0: [base]}, str(checkpoints))
    data = prepare_vxc(str(source), str(checkpoints))
    assert len(data) == 5
    assert all(set(("Grid", "Vrho", "Weights", "E_xc")) <= item.keys() for item in data)
    assert not any((checkpoints / name).exists() for name in mn.OBSOLETE_PICKLES)
    predopt, train, loaded_mrks = mn.load_chk(str(checkpoints))
    assert len(loaded_mrks) == 5
    # Even if stale files reappear they cannot be read (invalid pickle bytes).
    for name in mn.OBSOLETE_PICKLES:
        (checkpoints / name).write_bytes(b"obsolete invalid pickle")
    with pytest.warns(UserWarning, match="Ignoring obsolete"):
        assert len(mn.load_chk(str(checkpoints))) == 3
    assert predopt[0]["ReactionID"] == train[0][0]["ReactionID"]
    assert "test_size" not in inspect.signature(prepare_vxc).parameters


def test_unversioned_or_contaminated_pickle_fails_closed(tmp_path):
    for name in (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "data_vxc_train.pickle",
    ):
        (tmp_path / name).write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match="rerun"):
        mn.load_chk(str(tmp_path))
    contaminated = {"Database": "DBH76", "ReactionID": 15}
    mn.save_chk({0: contaminated}, {0: [contaminated]}, str(tmp_path))
    (tmp_path / "mrks_protocol.json").write_text(
        json.dumps({"protocol": mn.TRAINING_PROTOCOL})
    )
    with pytest.raises(ValueError, match="provenance"):
        mn.load_chk(str(tmp_path))


def test_training_loader_contract_uses_entire_mrks_corpus():
    args = SimpleNamespace(
        batch_size=1, vxc_batch_size=1, num_workers_train=0, num_workers_vxc=0
    )
    reaction = {"Energy": torch.tensor([0.0])}
    loaders = training.build_dataloaders(
        {0: [reaction]},
        [{"Name": str(i)} for i in range(5)],
        trial_seed=1,
        args=args,
        rank=0,
        world_size=1,
    )
    assert set(loaders) == {
        "train_loader",
        "train_sampler",
        "vxc_train_loader",
        "vxc_train_sampler",
    }
    assert len(loaders["vxc_train_loader"].dataset) == 5


def test_replay_history_final_checkpoint_and_ten_epoch_snapshots(monkeypatch, tmp_path):
    args = SimpleNamespace(
        seed=41,
        name="PBE-LGxGc_2_8",
        n_train=20,
        weight_decay=0.01,
        training_state_every=10,
        snapshot_start_epoch=10,
        snapshot_every=10,
    )
    loaders = {
        "train_loader": SimpleNamespace(dataset=object()),
        "train_sampler": None,
        "vxc_train_loader": object(),
        "vxc_train_sampler": None,
    }
    monkeypatch.setattr(training, "build_dataloaders", lambda **kwargs: loaders)
    monkeypatch.setattr(training, "build_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(training, "load_state_dict_into_model", lambda *args: None)
    monkeypatch.setattr(
        training, "DDP", lambda model, **kwargs: SimpleNamespace(module=model)
    )
    monkeypatch.setattr(
        training,
        "configure_optimizers",
        lambda model, **kwargs: torch.optim.SGD(model.module.parameters(), lr=0.01),
    )
    monkeypatch.setattr(
        training,
        "build_training_scheduler",
        lambda optimizer, args: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.9
        ),
    )
    monkeypatch.setattr(training, "sync_failure", lambda failed, device: failed)
    monkeypatch.setattr(training, "gather_object", lambda obj, world_size: [obj])
    state_epochs = []
    original_save = training.atomic_torch_save

    def record_save(payload, path):
        if "training_state" in path.name:
            state_epochs.append(payload["completed_epoch"])
        original_save(payload, path)

    monkeypatch.setattr(training, "atomic_torch_save", record_save)
    calls = []

    def train_epoch(**kwargs):
        assert kwargs["train_loader"] is loaders["train_loader"]
        assert kwargs["vxc_train_loader"] is loaders["vxc_train_loader"]
        calls.append(kwargs["epoch"])
        # Increasing diagnostic error must not select an earlier checkpoint.
        with torch.no_grad():
            kwargs["model"].module.weight.fill_(kwargs["epoch"] + 1)
        kwargs["optimizer"].step()
        metrics = {
            key: float(kwargs["epoch"] + 1)
            for key in (
                "train_fchem",
                "train_vxc",
                "train_exc",
                "train_exc_loss",
                "train_full_loss",
                "train_reaction_loss",
                "train_mae",
                "optimizer_steps",
            )
        }
        metrics["train_per_system_exc_rmse"] = {}
        return metrics, {}, False

    monkeypatch.setattr(training, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        training,
        "validate_one_epoch",
        lambda **kwargs: pytest.fail("non-SCF validation called"),
    )
    params = {
        "lr_train": 0.01,
        "gradient_merge_strategy": "sum",
        "accum_iter": 1,
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 1,
        "vxc_grad_clip": 5,
        "vxc_loss_scale": 40,
    }
    result = training.run_trial(
        19,
        params,
        args,
        None,
        {},
        [],
        torch.device("cpu"),
        0,
        1,
        {},
        None,
        tmp_path,
        True,
    )
    assert calls == list(range(20))
    assert state_epochs == [10, 20]
    assert result["final_epoch"] == 20
    assert not any("selected" in key or "val_" in key for key in result)
    for row in result["epoch_history"]:
        assert not any(key.startswith("val_") for key in row)
        assert {
            "train_fchem",
            "train_vxc",
            "train_exc",
            "phase_name",
            "learning_rate",
        } <= row.keys()
    snapshots = list((tmp_path / "checkpoints" / "epoch_snapshots").glob("*.pt"))
    assert {p.name for p in snapshots} == {
        "trial_19_epoch_0010.pt",
        "trial_19_epoch_0020.pt",
    }
    final = torch.load(result["final_checkpoint_path"], weights_only=True)
    assert final["weight"].item() == 20
    state = torch.load(result["training_state_path"], weights_only=False)
    assert state["completed_epoch"] == 20
    assert state["format_version"] == 2
    assert state["training_protocol"] == mn.TRAINING_PROTOCOL
    assert "current_selected_key" not in state
    obsolete_path = tmp_path / "obsolete_state.pt"
    torch.save({**state, "format_version": 1}, obsolete_path)
    args.resume_training_state = str(obsolete_path)
    with pytest.raises(ValueError, match="Unsupported training-state format"):
        training.run_trial(
            19,
            params,
            args,
            None,
            {},
            [],
            torch.device("cpu"),
            0,
            1,
            {},
            None,
            tmp_path,
            True,
        )


def test_optuna_cannot_rank_training_metrics():
    with pytest.raises(SystemExit, match="external.*SCF"):
        training.main()


def test_preopt_metadata_cannot_reuse_old_protocol():
    args = SimpleNamespace(
        name="PBE-LGxGc_6_32",
        model_type="gc_svelu_mirror",
        dropout=0,
        n_predopt=2,
        lr_predopt=0.01,
        batch_size=1,
        preopt_vxc_weight=0,
        preopt_vxc_steps=0,
        preopt_vxc_target="pbe",
    )
    metadata = training.preopt_metadata(args)
    assert metadata["training_protocol"] == mn.TRAINING_PROTOCOL
    historical = {
        key: value for key, value in metadata.items() if key != "training_protocol"
    }
    assert metadata != historical
