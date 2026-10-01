"""Build and validate immutable run metadata for the Lap full-Vxc S5 trainer.

The public entry points are :func:`source_bindings_from_verified_corpus`,
:func:`build_lap_s5_provenance`, and :func:`validate_lap_s5_provenance`.
At launch, call ``lap_data.verify_corpus(corpus_dir)`` and pass its returned
manifest, the SHA-256 of ``preprocessing_manifest.json``, and the dispersion
artifact identity/hash to ``source_bindings_from_verified_corpus``.  This
captures Minnesota and mRKS artifact hashes from the verified immutable
manifest; original source paths are neither required nor recorded.  Pass the
result into ``build_lap_s5_provenance`` and store the returned mapping in the
checkpoint.  On resume, validate that mapping against the current verified
corpus manifest/file hash and current dispersion hash before restoring
optimizer state.

The validator is deliberately strict: only the tau-free ``C - div(A) +
lap(B)`` potential, explicit positive stencil spacing, RAdamW with the
historical S5 decay, the five-phase ``simple4_two_step_40_10`` schedule, and
hashed Minnesota/mRKS/dispersion sources are accepted.  The metadata itself
is JSON-serializable, so it can also be written beside a checkpoint/report.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

# These research modules retain script-style absolute imports among siblings.
_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from lap_data import PROTOCOL
from lap_s5_protocol import OMEGA_METADATA, build_lap_s5_protocol
from lap_vxc import STENCIL_ORDER, STENCIL_VERSION
from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL

METADATA_VERSION = 1
MODEL_CLASS = "pcPBELMLOptimizerV2Lap"
POTENTIAL_MODE = "full_euler"
POTENTIAL_EXPRESSION = "C-divA+lapB"
SCHEDULE_NAME = "simple4_two_step_40_10"
S5_INITIAL_LR = 3.588259475602772e-4
S5_WEIGHT_DECAY = 0.01
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCALE_KEYS = {"vxc_loss_scale", "exc_loss_scale", "reaction_grad_scale"}
_CLIP_KEYS = {"vxc_grad_clip", "exc_grad_clip", "reaction_grad_clip"}
_PHASE_IDS = (
    "anchor",
    "representation_40",
    "representation_10",
    "repair",
    "consolidation",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest.",
    )
    return value


def _identity(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{label} must be a nonempty immutable artifact identity.",
    )
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be a finite number.",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite.")
    _require(result > 0 if positive else result >= 0, f"{label} is out of range.")
    return result


def _validate_sources(sources: Any) -> dict[str, dict[str, str]]:
    expected_groups = {"minnesota", "mrks_targets", "dispersions"}
    _require(isinstance(sources, Mapping), "Source bindings must be a mapping.")
    _require(
        set(sources) == expected_groups,
        "Source bindings must include Minnesota, mRKS targets, and dispersions exactly.",
    )
    normalized: dict[str, dict[str, str]] = {}
    for group in sorted(expected_groups):
        source = sources[group]
        _require(isinstance(source, Mapping), f"{group} binding must be a mapping.")
        expected_keys = {"identity", "sha256"}
        if group != "dispersions":
            expected_keys |= {"manifest_identity", "manifest_sha256"}
        _require(
            set(source) == expected_keys,
            f"{group} binding is missing identity/hash provenance or has extra fields.",
        )
        normalized[group] = {
            "identity": _identity(source["identity"], f"{group}.identity"),
            "sha256": _hash(source["sha256"], f"{group}.sha256"),
        }
        if group != "dispersions":
            normalized[group].update(
                manifest_identity=_identity(
                    source["manifest_identity"], f"{group}.manifest_identity"
                ),
                manifest_sha256=_hash(
                    source["manifest_sha256"], f"{group}.manifest_sha256"
                ),
            )
    return normalized


def source_bindings_from_verified_corpus(
    corpus_manifest: Mapping[str, Any],
    corpus_manifest_sha256: str,
    *,
    dispersion_identity: str,
    dispersion_sha256: str,
) -> dict[str, dict[str, str]]:
    """Extract source identities from a verified Lap corpus manifest.

    ``corpus_manifest_sha256`` is the digest of the manifest file itself,
    computed by the caller while verifying the corpus.  No source paths are
    needed: Minnesota and mRKS hashes are taken from ``artifact_sha256`` and
    the embedded Minnesota-manifest hash in the immutable Lap manifest.
    """
    _require(isinstance(corpus_manifest, Mapping), "Lap corpus manifest is required.")
    _require(
        corpus_manifest.get("protocol") == PROTOCOL
        and corpus_manifest.get("architecture") == ARCHITECTURE
        and corpus_manifest.get("descriptor_protocol") == DESCRIPTOR_PROTOCOL
        and corpus_manifest.get("stencil_version") == STENCIL_VERSION
        and corpus_manifest.get("manifest_version") == 1
        and corpus_manifest.get("mrks_systems") == 90
        and corpus_manifest.get("units") == "Bohr"
        and corpus_manifest.get("stencil_order") == list(STENCIL_ORDER),
        "Cannot bind sources from a legacy or non-Lap full-Vxc corpus manifest.",
    )
    _finite_number(corpus_manifest.get("h_bohr"), "corpus h_bohr", positive=True)
    artifacts = corpus_manifest.get("artifact_sha256")
    _require(isinstance(artifacts, Mapping), "Lap corpus artifact hashes are missing.")
    mn_hash = artifacts.get("data_train_grouped.pickle")
    target_hash = artifacts.get("data_full_vxc_train.pickle")
    _hash(mn_hash, "data_train_grouped.pickle sha256")
    _hash(target_hash, "data_full_vxc_train.pickle sha256")
    mn_manifest_hash = _hash(
        corpus_manifest.get("minnesota_manifest_sha256"),
        "minnesota_manifest_sha256",
    )
    return _validate_sources(
        {
            "minnesota": {
                "identity": "data_train_grouped.pickle",
                "sha256": mn_hash,
                "manifest_identity": "minnesota_preprocessing_manifest.json",
                "manifest_sha256": mn_manifest_hash,
            },
            "mrks_targets": {
                "identity": "data_full_vxc_train.pickle",
                "sha256": target_hash,
                "manifest_identity": "preprocessing_manifest.json",
                "manifest_sha256": _hash(
                    corpus_manifest_sha256, "preprocessing_manifest.json sha256"
                ),
            },
            "dispersions": {
                "identity": dispersion_identity,
                "sha256": dispersion_sha256,
            },
        }
    )


def _default_optimizer() -> dict[str, Any]:
    return {
        "name": "RAdamW",
        "learning_rate": S5_INITIAL_LR,
        "weight_decay": S5_WEIGHT_DECAY,
        "kwargs": {"betas": [0.9, 0.999], "eps": 1e-8},
    }


def _default_scheduler() -> dict[str, Any]:
    return {
        "name": "SequentialLR(LinearLR,CosineAnnealingLR)",
        "total_epochs": 500,
        "warmup_epochs": 5,
        "warmup_start_factor": 0.001,
        "minimum_lr": 1e-6,
    }


def _normalize_dtype(dtype: Any) -> str:
    value = str(dtype).removeprefix("torch.")
    _require(
        value in {"float32", "float64"}, "Lap S5 dtype must be float32 or float64."
    )
    return value


def build_lap_s5_provenance(
    *,
    h_bohr: float,
    dtype: Any,
    model_kwargs: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    optimizer: Mapping[str, Any] | None = None,
    scheduler: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create validated checkpoint/run provenance for one Lap-S5 configuration.

    ``source_bindings`` normally comes from
    :func:`source_bindings_from_verified_corpus`.  The optimizer/scheduler
    defaults encode the historical S5 RAdamW + five-epoch linear warmup and
    cosine schedule.  ``schedule`` may be an explicitly scale/clip-calibrated
    protocol, but phase boundaries, objective merge rules, and all other
    S5 mechanics remain fixed.
    """
    chosen_schedule = (
        build_lap_s5_protocol() if schedule is None else copy.deepcopy(dict(schedule))
    )
    payload = {
        "metadata_version": METADATA_VERSION,
        "architecture": ARCHITECTURE,
        "model_class": MODEL_CLASS,
        "descriptor_protocol": DESCRIPTOR_PROTOCOL,
        "protocol": PROTOCOL,
        "potential_mode": POTENTIAL_MODE,
        "potential_expression": POTENTIAL_EXPRESSION,
        "tau_dependent": False,
        "stencil": {"version": STENCIL_VERSION, "h_bohr": h_bohr, "units": "Bohr"},
        "dtype": _normalize_dtype(dtype),
        "model_kwargs": copy.deepcopy(dict(model_kwargs)),
        "optimizer": copy.deepcopy(
            dict(_default_optimizer() if optimizer is None else optimizer)
        ),
        "scheduler": copy.deepcopy(
            dict(_default_scheduler() if scheduler is None else scheduler)
        ),
        "schedule": chosen_schedule,
        "omega": OMEGA_METADATA,
        "sources": _validate_sources(source_bindings),
    }
    validate_lap_s5_provenance(payload)
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Lap S5 provenance must be finite and JSON-serializable."
        ) from exc
    return payload


def _validate_schedule(schedule: Any) -> None:
    reference = build_lap_s5_protocol()
    _require(isinstance(schedule, Mapping), "S5 schedule metadata is required.")
    _require(
        set(schedule) == set(reference), "S5 schedule has missing or unexpected fields."
    )
    _require(
        schedule.get("protocol") == reference["protocol"]
        and schedule.get("source_preset") == SCHEDULE_NAME
        and schedule.get("omega") == OMEGA_METADATA,
        "S5 schedule identity/OMEGA differs from the canonical protocol.",
    )
    phases = schedule.get("epoch_schedule")
    reference_phases = reference["epoch_schedule"]
    _require(
        isinstance(phases, list) and len(phases) == len(reference_phases),
        "S5 schedule must contain exactly five phases.",
    )
    for phase, expected in zip(phases, reference_phases):
        _require(isinstance(phase, Mapping), "S5 phase must be a mapping.")
        _require(
            set(phase) == set(expected), "S5 phase fields differ from the protocol."
        )
        for key in ("phase_id", "name", "start_epoch", "end_epoch"):
            _require(
                phase[key] == expected[key], f"S5 phase {key} differs from protocol."
            )
        params = phase["params"]
        expected_params = expected["params"]
        _require(
            isinstance(params, Mapping) and set(params) == set(expected_params),
            "S5 phase objective settings are incomplete or unexpected.",
        )
        for key, expected_value in expected_params.items():
            value = params[key]
            if key in _SCALE_KEYS:
                _finite_number(value, f"{phase['phase_id']}.{key}", positive=True)
            elif key in _CLIP_KEYS:
                if value not in (None, "none"):
                    _finite_number(value, f"{phase['phase_id']}.{key}", positive=True)
            else:
                _require(
                    value == expected_value,
                    f"S5 phase {phase['phase_id']} setting {key!r} differs from protocol.",
                )


def _validate_optimizer(optimizer: Any) -> None:
    _require(isinstance(optimizer, Mapping), "Optimizer provenance is required.")
    _require(
        set(optimizer) == {"name", "learning_rate", "weight_decay", "kwargs"},
        "Optimizer provenance fields are incomplete or unexpected.",
    )
    _require(
        isinstance(optimizer["name"], str) and optimizer["name"].casefold() == "radamw",
        "Lap S5 requires the historical RAdamW optimizer.",
    )
    _require(
        _finite_number(optimizer["learning_rate"], "learning_rate", positive=True) > 0,
        "RAdamW learning_rate must be positive.",
    )
    _require(
        _finite_number(optimizer["weight_decay"], "weight_decay") == S5_WEIGHT_DECAY,
        "Lap S5 RAdamW weight_decay must remain 0.01.",
    )
    _require(
        isinstance(optimizer["kwargs"], Mapping), "Optimizer kwargs must be a mapping."
    )
    _require(
        not any("tau" in str(key).casefold() for key in optimizer["kwargs"]),
        "Tau-related optimizer metadata is forbidden.",
    )


def _validate_scheduler(scheduler: Any) -> None:
    reference = _default_scheduler()
    _require(isinstance(scheduler, Mapping), "Scheduler provenance is required.")
    _require(
        set(scheduler) == set(reference), "Scheduler metadata fields differ from S5."
    )
    for key, expected in reference.items():
        if isinstance(expected, Real):
            _require(
                _finite_number(scheduler[key], f"scheduler.{key}") == expected,
                f"scheduler.{key} differs from the historical S5 scheduler.",
            )
        else:
            _require(scheduler[key] == expected, f"scheduler.{key} differs from S5.")


def _validate_model_kwargs(model_kwargs: Any) -> None:
    _require(isinstance(model_kwargs, Mapping), "Lap model kwargs are required.")
    required = {"num_layers", "h_dim", "dropout", "use_g_x", "use_g_c"}
    _require(required <= set(model_kwargs), "Lap model kwargs are incomplete.")
    _require(
        not any("tau" in str(key).casefold() for key in model_kwargs),
        "Tau-related model kwargs are forbidden.",
    )
    _require(
        isinstance(model_kwargs["dropout"], Real)
        and not isinstance(model_kwargs["dropout"], bool)
        and _finite_number(model_kwargs["dropout"], "model_kwargs.dropout") == 0,
        "Lap functional must use dropout=0.",
    )
    for key in ("num_layers", "h_dim"):
        value = model_kwargs[key]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"model_kwargs.{key} must be a positive integer.",
        )
    for key in ("use_g_x", "use_g_c"):
        _require(
            isinstance(model_kwargs[key], bool), f"model_kwargs.{key} must be boolean."
        )


def validate_lap_s5_provenance(
    metadata: Mapping[str, Any],
    *,
    corpus_manifest: Mapping[str, Any] | None = None,
    corpus_manifest_sha256: str | None = None,
    expected_source_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on protocol drift and optionally bind to a current corpus.

    Providing ``corpus_manifest`` requires its file hash too.  This compares
    the checkpoint's Minnesota and mRKS artifact hashes to the verified
    manifest without needing any original input path.
    ``expected_source_bindings`` optionally binds all three source groups,
    including the external dispersion artifact, to current immutable hashes.
    """
    _require(isinstance(metadata, Mapping), "Lap S5 provenance mapping is required.")
    expected_fields = {
        "metadata_version",
        "architecture",
        "model_class",
        "descriptor_protocol",
        "protocol",
        "potential_mode",
        "potential_expression",
        "tau_dependent",
        "stencil",
        "dtype",
        "model_kwargs",
        "optimizer",
        "scheduler",
        "schedule",
        "omega",
        "sources",
    }
    _require(
        set(metadata) == expected_fields, "Lap S5 checkpoint provenance fields differ."
    )
    _require(
        metadata["metadata_version"] == METADATA_VERSION,
        "Unsupported provenance version.",
    )
    _require(
        metadata["architecture"] == ARCHITECTURE
        and metadata["model_class"] == MODEL_CLASS
        and metadata["descriptor_protocol"] == DESCRIPTOR_PROTOCOL
        and metadata["protocol"] == PROTOCOL,
        "Architecture/descriptor/full-Vxc protocol mismatch; reject legacy checkpoints.",
    )
    _require(
        metadata["potential_mode"] == POTENTIAL_MODE
        and metadata["potential_expression"] == POTENTIAL_EXPRESSION
        and metadata["tau_dependent"] is False,
        "Lap checkpoint must represent the tau-free full Euler potential C-divA+lapB.",
    )
    _require(metadata["omega"] == OMEGA_METADATA, "Lap S5 requires OMEGA=0.5.")
    stencil = metadata["stencil"]
    _require(
        isinstance(stencil, Mapping) and set(stencil) == {"version", "h_bohr", "units"},
        "Explicit stencil version and h metadata are required.",
    )
    _require(
        stencil["version"] == STENCIL_VERSION and stencil["units"] == "Bohr",
        "Unsupported stencil version or coordinate units.",
    )
    _finite_number(stencil["h_bohr"], "stencil.h_bohr", positive=True)
    _normalize_dtype(metadata["dtype"])
    _validate_model_kwargs(metadata["model_kwargs"])
    _validate_optimizer(metadata["optimizer"])
    _validate_scheduler(metadata["scheduler"])
    _validate_schedule(metadata["schedule"])
    stored_sources = _validate_sources(metadata["sources"])
    if expected_source_bindings is not None:
        _require(
            stored_sources == _validate_sources(expected_source_bindings),
            "Checkpoint source identities/hashes differ from the current immutable inputs.",
        )

    _require(
        (corpus_manifest is None) == (corpus_manifest_sha256 is None),
        "Pass both corpus manifest data and its SHA-256, or neither.",
    )
    if corpus_manifest is not None:
        actual_bindings = source_bindings_from_verified_corpus(
            corpus_manifest,
            corpus_manifest_sha256,
            dispersion_identity=metadata["sources"]["dispersions"]["identity"],
            dispersion_sha256=metadata["sources"]["dispersions"]["sha256"],
        )
        for group in ("minnesota", "mrks_targets"):
            _require(
                metadata["sources"][group] == actual_bindings[group],
                f"Checkpoint {group} identity/hash differs from the current immutable corpus.",
            )
        _require(
            metadata["stencil"]["h_bohr"] == corpus_manifest["h_bohr"],
            "Checkpoint stencil h differs from the current immutable corpus.",
        )
    return copy.deepcopy(dict(metadata))


__all__ = [
    "METADATA_VERSION",
    "build_lap_s5_provenance",
    "source_bindings_from_verified_corpus",
    "validate_lap_s5_provenance",
]
