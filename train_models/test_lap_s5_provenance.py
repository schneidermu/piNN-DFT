"""Fail-closed checkpoint metadata tests for Lap S5 provenance."""

import copy
import sys
import types
from pathlib import Path

import pytest

_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)


def _install_replay_import_stub():
    # The schedule helper only needs E3_SCHEDULE_PRESETS; avoid importing the
    # distributed training stack while testing this small metadata contract.
    if "optuna_joint" in sys.modules:
        return False
    stub = types.ModuleType("optuna_joint")
    for name in (
        "init_distributed",
        "load_chk",
        "load_mrks_dispersions",
        "run_or_reuse_preoptimization",
        "run_trial",
        "set_random_seed",
    ):
        setattr(stub, name, None)
    stub.DEFAULT_MRKS_DISPERSIONS = ""
    sys.modules["optuna_joint"] = stub
    return True


_installed_training_stub = _install_replay_import_stub()

try:
    from .lap_data import PROTOCOL
    from .lap_s5_protocol import apply_lap_s5_scale_overrides
    from .lap_s5_provenance import (
        build_lap_s5_provenance,
        source_bindings_from_verified_corpus,
        validate_lap_s5_provenance,
    )
    from .lap_vxc import STENCIL_ORDER, STENCIL_VERSION
    from .NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL
except ImportError:
    from lap_data import PROTOCOL
    from lap_s5_protocol import apply_lap_s5_scale_overrides
    from lap_s5_provenance import (
        build_lap_s5_provenance,
        source_bindings_from_verified_corpus,
        validate_lap_s5_provenance,
    )
    from lap_vxc import STENCIL_ORDER, STENCIL_VERSION
    from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL

if _installed_training_stub:
    # Do not leak the temporary replay-bridge stub into training-module tests.
    sys.modules.pop("optuna_joint", None)


_MN_HASH = "a" * 64
_MN_MANIFEST_HASH = "b" * 64
_MRKS_HASH = "c" * 64
_CORPUS_MANIFEST_HASH = "d" * 64
_DISPERSION_HASH = "e" * 64


def _verified_manifest():
    # Identities deliberately contain no source paths; the verified immutable
    # artifact and manifest hashes are sufficient for training-time checks.
    return {
        "manifest_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "descriptor_protocol": DESCRIPTOR_PROTOCOL,
        "stencil_version": STENCIL_VERSION,
        "stencil_order": list(STENCIL_ORDER),
        "units": "Bohr",
        "h_bohr": 0.005,
        "mrks_systems": 90,
        "minnesota_manifest_sha256": _MN_MANIFEST_HASH,
        "artifact_sha256": {
            "data_predopt.pickle": "f" * 64,
            "data_train_grouped.pickle": _MN_HASH,
            "minnesota_protocol.json": "1" * 64,
            "data_full_vxc_train.pickle": _MRKS_HASH,
        },
    }


def _metadata(schedule=None):
    sources = source_bindings_from_verified_corpus(
        _verified_manifest(),
        _CORPUS_MANIFEST_HASH,
        dispersion_identity="dispersions_mrks.pickle",
        dispersion_sha256=_DISPERSION_HASH,
    )
    return build_lap_s5_provenance(
        h_bohr=0.005,
        dtype="float32",
        model_kwargs={
            "num_layers": 6,
            "h_dim": 32,
            "dropout": 0.0,
            "use_g_x": True,
            "use_g_c": True,
        },
        source_bindings=sources,
        schedule=schedule,
    )


def test_build_and_validate_complete_lap_s5_metadata_without_raw_source_paths():
    metadata = _metadata()
    assert validate_lap_s5_provenance(metadata) == metadata
    assert metadata["potential_mode"] == "full_euler"
    assert metadata["potential_expression"] == "C-divA+lapB"
    assert metadata["tau_dependent"] is False
    assert metadata["optimizer"]["name"] == "RAdamW"
    assert metadata["optimizer"]["weight_decay"] == 0.01
    assert metadata["schedule"]["source_preset"] == "simple4_two_step_40_10"
    assert metadata["omega"] == 0.5
    assert metadata["sources"]["minnesota"]["sha256"] == _MN_HASH
    assert metadata["sources"]["mrks_targets"]["sha256"] == _MRKS_HASH
    assert metadata["sources"]["dispersions"]["sha256"] == _DISPERSION_HASH
    assert not any(
        "path" in key for source in metadata["sources"].values() for key in source
    )


def test_v1_7point_stencil_provenance_normalizes_from_explicit_version():
    metadata = _metadata()
    legacy = copy.deepcopy(metadata)
    legacy["metadata_version"] = 1
    legacy["stencil"] = {
        key: legacy["stencil"][key] for key in ("version", "h_bohr", "units")
    }

    restored = validate_lap_s5_provenance(legacy)
    assert restored["metadata_version"] == 2
    assert restored["stencil"]["version"] == STENCIL_VERSION
    assert restored["stencil"]["derivative_order"] == 2
    assert restored["stencil"]["stencil_order"] == list(STENCIL_ORDER)


def test_verified_corpus_manifest_revalidates_embedded_artifact_hashes():
    metadata = _metadata()
    assert (
        validate_lap_s5_provenance(
            metadata,
            corpus_manifest=_verified_manifest(),
            corpus_manifest_sha256=_CORPUS_MANIFEST_HASH,
        )
        == metadata
    )

    changed = _verified_manifest()
    changed["artifact_sha256"]["data_full_vxc_train.pickle"] = "9" * 64
    with pytest.raises(ValueError, match="mrks_targets identity/hash"):
        validate_lap_s5_provenance(
            metadata,
            corpus_manifest=changed,
            corpus_manifest_sha256=_CORPUS_MANIFEST_HASH,
        )


def test_current_source_hashes_can_be_checked_independently_of_file_paths():
    metadata = _metadata()
    expected = copy.deepcopy(metadata["sources"])
    validate_lap_s5_provenance(metadata, expected_source_bindings=expected)
    expected["dispersions"]["sha256"] = "8" * 64
    with pytest.raises(ValueError, match="source identities/hashes"):
        validate_lap_s5_provenance(metadata, expected_source_bindings=expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("architecture", "pcPBELMLOptimizerV2", "Architecture/descriptor"),
        ("descriptor_protocol", "rho-sigma-total-tau-v1", "Architecture/descriptor"),
        ("potential_mode", "partial_vrho", "full Euler potential"),
        ("potential_expression", "partial-rho", "full Euler potential"),
        ("tau_dependent", True, "full Euler potential"),
        ("omega", 1.0, "OMEGA=0.5"),
    ],
)
def test_rejects_legacy_partial_tau_and_wrong_omega(field, value, message):
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        validate_lap_s5_provenance(metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m["stencil"].update(h_bohr=0.0), "h_bohr"),
        (lambda m: m["stencil"].update(version="legacy-grid-vrho"), "stencil version"),
        (lambda m: m.update(dtype="float16"), "dtype"),
        (lambda m: m["model_kwargs"].update(tau=True), "Tau-related"),
        (lambda m: m["model_kwargs"].update(dropout=0.1), "dropout=0"),
        (lambda m: m["optimizer"].update(name="AdamW"), "RAdamW"),
        (lambda m: m["optimizer"].update(weight_decay=0.1), "weight_decay"),
        (lambda m: m["scheduler"].update(warmup_epochs=4), "scheduler.warmup_epochs"),
        (lambda m: m["sources"]["minnesota"].update(sha256="not-a-hash"), "SHA-256"),
        (
            lambda m: m["sources"]["mrks_targets"].pop("manifest_sha256"),
            "identity/hash",
        ),
        (lambda m: m["sources"]["dispersions"].update(identity=""), "identity"),
    ],
)
def test_rejects_incomplete_or_mismatched_run_provenance(mutation, message):
    metadata = _metadata()
    mutation(metadata)
    with pytest.raises(ValueError, match=message):
        validate_lap_s5_provenance(metadata)


def test_allows_only_explicit_objective_scale_or_clip_calibration():
    base = _metadata()["schedule"]
    calibrated = apply_lap_s5_scale_overrides(
        base,
        {"anchor": {"vxc_loss_scale": 65.0}},
        clip_thresholds={"repair": {"exc_grad_clip": 1.5}},
    )
    validate_lap_s5_provenance(_metadata(schedule=calibrated))

    tampered = copy.deepcopy(calibrated)
    tampered["epoch_schedule"][0]["params"]["gradient_merge_strategy"] = "sum"
    with pytest.raises(ValueError, match="differs from protocol"):
        _metadata(schedule=tampered)


def test_requires_manifest_data_and_manifest_hash_as_a_pair():
    with pytest.raises(ValueError, match="Pass both corpus manifest"):
        validate_lap_s5_provenance(_metadata(), corpus_manifest=_verified_manifest())


def test_refuses_unhashed_or_wrong_protocol_corpus_sources():
    manifest = _verified_manifest()
    manifest["protocol"] = "legacy-vrho"
    with pytest.raises(ValueError, match="legacy or non-Lap"):
        source_bindings_from_verified_corpus(
            manifest,
            _CORPUS_MANIFEST_HASH,
            dispersion_identity="dispersions_mrks.pickle",
            dispersion_sha256=_DISPERSION_HASH,
        )

    manifest = _verified_manifest()
    manifest["minnesota_manifest_sha256"] = None
    with pytest.raises(ValueError, match="SHA-256"):
        source_bindings_from_verified_corpus(
            manifest,
            _CORPUS_MANIFEST_HASH,
            dispersion_identity="dispersions_mrks.pickle",
            dispersion_sha256=_DISPERSION_HASH,
        )
