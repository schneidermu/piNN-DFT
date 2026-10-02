"""Checkpoint-loading guards for the cursor-100 real-SCF smoke runner."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_lap_moo_scf_smoke as _scf_smoke
from lap_moo_protocol import (
    LEGACY_READ_ONLY_PROTOCOL_VERSION,
    build_sampling_manifest,
    canonical_sha256,
    make_protocol_metadata,
)
from lap_moo_training import make_cosine_scheduler, save_moo_checkpoint
from NN_models_lap import pcPBELMLOptimizerV2Lap
from run_lap_moo_scf_smoke import (
    ARCHITECTURE,
    CANONICAL_MODEL_KWARGS,
    PREDOPT_SHA256,
    _load_moo_model_at_cursor,
    _model_state_sha256,
)
from utils import configure_optimizers


def _canonical_checkpoint(
    tmp_path: Path,
    *,
    cursor: int = 100,
    cursor_override: int | None = None,
    scheduler_epoch_override: int | None = None,
    scheduler_step_count_override: int | None = None,
    architecture: str = ARCHITECTURE,
    protocol_version: str | None = None,
    schedule_updates_override: int | None = None,
) -> tuple[Path, str, dict]:
    grouped = {
        0: [
            {
                "Database": "ABDE4",
                "ReactionID": 1,
                "component_paths": ["A1.npz"],
            }
        ]
    }
    manifest = build_sampling_manifest(
        grouped,
        ["H2"],
        updates=150,
        seed=41,
        source_hashes={"fixture_source": hashlib.sha256(b"source").hexdigest()},
    )
    manifest_sha = manifest["manifest_sha256"]
    weights = [1.0, 1.0, 1.0]
    protocol = make_protocol_metadata(
        architecture=architecture,
        method="fixed",
        method_hyperparameters={"fixed_weights": weights},
        fixed_scalarization={
            "calibration_report_sha256": hashlib.sha256(b"calibration").hexdigest(),
            "fixed_weights": weights,
        },
        optimizer={"name": "RAdamW", "learning_rate": 1e-6},
        lr_schedule={
            "name": "cosine",
            "total_updates": 150,
            "minimum_lr_ratio": 0.1,
            "same_shape_for_all_methods": True,
        },
        predopt_checkpoint_sha256=PREDOPT_SHA256,
        sampling_manifest_sha256=manifest_sha,
        minnesota_data_sha256=hashlib.sha256(b"minnesota").hexdigest(),
        operator_corpus_manifest_sha256=hashlib.sha256(b"operator").hexdigest(),
        ao_cache_manifest_sha256=hashlib.sha256(b"ao-cache").hexdigest(),
        reaction_dispersions_sha256=hashlib.sha256(b"reaction-d3").hexdigest(),
        mrks_dispersions_sha256=hashlib.sha256(b"mrks-d3").hexdigest(),
        random_seed=41,
        dtype="float32",
        grid_chunk_size=256,
        ao_cache_chunk_size=4096,
    )
    model = pcPBELMLOptimizerV2Lap(**CANONICAL_MODEL_KWARGS)
    optimizer = configure_optimizers(
        model, 1e-6, optimizer_str="radamw", weight_decay=0.01
    )
    scheduler = make_cosine_scheduler(optimizer, total_updates=150)
    for _ in range(cursor):
        optimizer.step()
        scheduler.step()

    path = tmp_path / "canonical-moo.pt"
    save_moo_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        protocol_metadata=protocol,
        sampling_manifest=manifest,
        next_update=cursor,
        aggregator_state={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if cursor_override is not None:
        payload["sampling_cursor"]["next_update"] = cursor_override
    if scheduler_epoch_override is not None:
        payload["scheduler_state_dict"]["last_epoch"] = scheduler_epoch_override
    if scheduler_step_count_override is not None:
        payload["scheduler_state_dict"]["_step_count"] = scheduler_step_count_override
    if architecture != ARCHITECTURE:
        payload["protocol_metadata"]["architecture"] = architecture
    if protocol_version is not None:
        payload["protocol_metadata"]["protocol_version"] = protocol_version
        if protocol_version == LEGACY_READ_ONLY_PROTOCOL_VERSION:
            payload["protocol_metadata"].pop("ao_cache_chunk_size")
    if schedule_updates_override is not None:
        payload["protocol_metadata"]["lr_schedule"]["total_updates"] = (
            schedule_updates_override
        )
    payload["protocol_metadata_sha256"] = canonical_sha256(
        payload["protocol_metadata"]
    )
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, payload


def test_valid_cursor100_checkpoint_loads_exact_saved_weights_on_cpu(tmp_path):
    path, digest, payload = _canonical_checkpoint(tmp_path)

    model, metadata = _load_moo_model_at_cursor(
        path, expected_sha256=digest, expected_method="fixed"
    )

    assert metadata["checkpoint_cursor"] == metadata["scheduler_last_epoch"] == 100
    assert metadata["scheduler_step_count"] == 101
    assert metadata["model_state_dict_sha256"] == _model_state_sha256(
        payload["model_state_dict"]
    )
    assert metadata["model_kwargs"] == CANONICAL_MODEL_KWARGS
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float64
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            payload["model_state_dict"][name].to(dtype=value.dtype),
            rtol=0,
            atol=0,
        )


def test_scf_read_only_loader_accepts_historical_v1_without_chunk_size(tmp_path):
    path, digest, _payload = _canonical_checkpoint(
        tmp_path, protocol_version=LEGACY_READ_ONLY_PROTOCOL_VERSION
    )

    model, metadata = _load_moo_model_at_cursor(
        path, expected_sha256=digest, expected_method="fixed"
    )

    assert model.architecture == ARCHITECTURE
    assert metadata["protocol_metadata"]["protocol_version"] == (
        LEGACY_READ_ONLY_PROTOCOL_VERSION
    )
    assert "ao_cache_chunk_size" not in metadata["protocol_metadata"]


def test_loader_rejects_checkpoint_bytes_that_do_not_match_the_pinned_hash(tmp_path):
    path, _digest, _payload = _canonical_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        _load_moo_model_at_cursor(
            path, expected_sha256="0" * 64, expected_method="fixed"
        )


def test_loader_rejects_checkpoint_cursor_that_does_not_match_requested_smoke(tmp_path):
    path, digest, _payload = _canonical_checkpoint(tmp_path, cursor_override=99)

    with pytest.raises(ValueError, match="cursor-100 checkpoint"):
        _load_moo_model_at_cursor(
            path, expected_sha256=digest, expected_method="fixed"
        )


def test_loader_rejects_stale_scheduler_even_when_sampling_cursor_is_100(tmp_path):
    path, digest, _payload = _canonical_checkpoint(
        tmp_path, scheduler_epoch_override=99
    )

    with pytest.raises(ValueError, match="scheduler position"):
        _load_moo_model_at_cursor(
            path, expected_sha256=digest, expected_method="fixed"
        )

    path, digest, _payload = _canonical_checkpoint(
        tmp_path, scheduler_step_count_override=100
    )
    with pytest.raises(ValueError, match="scheduler step count"):
        _load_moo_model_at_cursor(
            path, expected_sha256=digest, expected_method="fixed"
        )


def test_loader_rejects_checkpoint_from_a_different_moo_method(tmp_path):
    path, digest, _payload = _canonical_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="does not match requested method"):
        _load_moo_model_at_cursor(
            path, expected_sha256=digest, expected_method="cagrad"
        )


def test_loader_rejects_noncanonical_architecture_and_protocol_version(tmp_path):
    architecture_path, architecture_digest, _payload = _canonical_checkpoint(
        tmp_path, architecture="other-lap-model"
    )
    with pytest.raises(ValueError, match="Unsupported MOO architecture"):
        _load_moo_model_at_cursor(
            architecture_path,
            expected_sha256=architecture_digest,
            expected_method="fixed",
        )

    protocol_path, protocol_digest, _payload = _canonical_checkpoint(
        tmp_path, protocol_version="lap-moo-invalid-v1"
    )
    with pytest.raises(ValueError, match="Incompatible one-stage Lap MOO protocol"):
        _load_moo_model_at_cursor(
            protocol_path,
            expected_sha256=protocol_digest,
            expected_method="fixed",
        )

    short_horizon_path, short_horizon_digest, _payload = _canonical_checkpoint(
        tmp_path, schedule_updates_override=20
    )
    with pytest.raises(ValueError, match="horizon is shorter"):
        _load_moo_model_at_cursor(
            short_horizon_path,
            expected_sha256=short_horizon_digest,
            expected_method="fixed",
        )


def test_shorter_optimizer_pilot_requires_explicit_expected_cursor(tmp_path):
    path, digest, _payload = _canonical_checkpoint(tmp_path, cursor=25)

    with pytest.raises(ValueError, match="cursor-100 checkpoint"):
        _load_moo_model_at_cursor(
            path, expected_sha256=digest, expected_method="fixed"
        )

    _model, metadata = _load_moo_model_at_cursor(
        path,
        expected_sha256=digest,
        expected_method="fixed",
        expected_cursor=25,
    )
    assert metadata["checkpoint_cursor"] == metadata["expected_cursor"] == 25
    assert metadata["scheduler_last_epoch"] == 25
    assert metadata["scheduler_step_count"] == 26


def test_scf_smoke_cli_accepts_pcd_checkpoint_method(tmp_path, monkeypatch):
    class _ReachedAfterParse(Exception):
        pass

    checkpoint = tmp_path / "pcd-cursor25.pt"
    npz_root = tmp_path / "npz"
    output = tmp_path / "scf.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lap_moo_scf_smoke.py",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            "a" * 64,
            "--method",
            "pcd",
            "--expected-cursor",
            "25",
            "--npz-root",
            str(npz_root),
            "--output",
            str(output),
        ],
    )

    def stop_after_parse(path, label):
        assert Path(path) == checkpoint
        assert label == "MOO checkpoint"
        raise _ReachedAfterParse

    monkeypatch.setattr(_scf_smoke, "_require_external", stop_after_parse)
    with pytest.raises(_ReachedAfterParse):
        _scf_smoke.main()
