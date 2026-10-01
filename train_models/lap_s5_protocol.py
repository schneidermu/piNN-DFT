"""Pure helpers for the canonical Lap S5 replay protocol.

The reference schedule is the ``simple4_two_step_40_10`` E3 preset. That
preset stores the 208-epoch representation interval as one entry per epoch;
this module folds adjacent entries with identical parameters into five
readable phases without changing any phase parameter.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

SOURCE_PRESET_NAME = "simple4_two_step_40_10"
OMEGA_METADATA = 0.5
EXPECTED_BOUNDARIES = (
    (1, 72),
    (73, 176),
    (177, 280),
    (281, 440),
    (441, 500),
)

_EXPECTED_PHASE_NAMES = (
    "simple_potential_anchor",
    "simple_joint_representation",
    "simple_joint_representation",
    "simple_chemical_repair",
    "simple_unbiased_consolidation",
)
_PHASE_IDS = (
    "anchor",
    "representation_40",
    "representation_10",
    "repair",
    "consolidation",
)
_SCALE_KEYS = frozenset({"vxc_loss_scale", "exc_loss_scale", "reaction_grad_scale"})
_CLIP_KEYS = frozenset({"reaction_grad_clip", "vxc_grad_clip", "exc_grad_clip"})

# Full phase parameter records make the protocol fail closed if the replay
# reference changes its merge, clipping, accumulation, or objective settings.
_EXPECTED_PARAMS = (
    {
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.6,
        "vxc_grad_clip": 5.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
    },
    {
        "accum_iter": 2,
        "gradient_merge_strategy": "sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 1.0,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 40.0,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
    },
    {
        "accum_iter": 2,
        "gradient_merge_strategy": "sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 1.0,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 10.0,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
    },
    {
        "accum_iter": 2,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.75,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 15,
        "exc_loss_scale": 3.0,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
    },
    {
        "accum_iter": 2,
        "gradient_merge_strategy": "sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 1.0,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 7.0,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
    },
)


def _load_reference_schedule() -> Sequence[Mapping[str, Any]]:
    """Import the replay preset lazily so schedule inspection stays lightweight."""
    try:
        from .replay_trial_19_bridge import E3_SCHEDULE_PRESETS
    except ImportError:
        from replay_trial_19_bridge import E3_SCHEDULE_PRESETS
    return E3_SCHEDULE_PRESETS[SOURCE_PRESET_NAME]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_finite_number(
    value: Any, label: str, *, allow_zero: bool = True
) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be a finite number.",
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite.")
    _require(
        number >= 0 if allow_zero else number > 0, f"{label} must be non-negative."
    )
    return number


def _validate_source_schedule(
    source_schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(
        isinstance(source_schedule, Sequence)
        and not isinstance(source_schedule, (str, bytes)),
        "The S5 source schedule must be a sequence of phase mappings.",
    )
    _require(bool(source_schedule), "The S5 source schedule cannot be empty.")

    copied = copy.deepcopy(list(source_schedule))
    expected_start = 1
    normalized: list[dict[str, Any]] = []
    for index, phase in enumerate(copied):
        _require(isinstance(phase, Mapping), f"Source phase {index} must be a mapping.")
        _require(
            set(phase) == {"name", "start_epoch", "end_epoch", "params"},
            f"Source phase {index} has unexpected or missing fields.",
        )
        name = phase["name"]
        start = phase["start_epoch"]
        end = phase["end_epoch"]
        params = phase["params"]
        _require(isinstance(name, str) and name, f"Source phase {index} needs a name.")
        _require(
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool),
            f"Source phase {index} epoch boundaries must be integers.",
        )
        _require(
            start == expected_start,
            "Source schedule must cover epochs 1-500 exactly, without gaps or overlaps.",
        )
        _require(end >= start, f"Source phase {index} has an inverted epoch range.")
        _require(
            isinstance(params, Mapping),
            f"Source phase {index} params must be a mapping.",
        )
        if (
            normalized
            and normalized[-1]["name"] == name
            and normalized[-1]["params"] == params
        ):
            normalized[-1]["end_epoch"] = end
        else:
            normalized.append(
                {
                    "name": name,
                    "start_epoch": start,
                    "end_epoch": end,
                    "params": copy.deepcopy(dict(params)),
                }
            )
        expected_start = end + 1

    _require(expected_start == 501, "Source schedule must end at epoch 500.")
    _require(
        len(normalized) == 5, "The S5 reference must collapse to exactly five phases."
    )

    actual_boundaries = tuple(
        (phase["start_epoch"], phase["end_epoch"]) for phase in normalized
    )
    _require(
        actual_boundaries == EXPECTED_BOUNDARIES,
        "S5 phase boundaries do not match the canonical protocol.",
    )
    actual_names = tuple(phase["name"] for phase in normalized)
    _require(
        actual_names == _EXPECTED_PHASE_NAMES,
        "S5 phase labels do not match the canonical protocol.",
    )
    for index, (phase, expected_params) in enumerate(zip(normalized, _EXPECTED_PARAMS)):
        _require(
            phase["params"] == expected_params,
            f"S5 phase {index + 1} parameters differ from the canonical reference.",
        )
        _require(
            "omega" not in phase["params"] and "OMEGA" not in phase["params"],
            "OMEGA belongs in protocol metadata, not phase params.",
        )
    return normalized


def _annotate_phases(phases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for phase_id, phase in zip(_PHASE_IDS, phases):
        result.append(
            {
                "phase_id": phase_id,
                "name": phase["name"],
                "start_epoch": phase["start_epoch"],
                "end_epoch": phase["end_epoch"],
                "params": copy.deepcopy(phase["params"]),
            }
        )
    return result


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    allow_scale_and_clip_overrides: bool = False,
) -> None:
    _require(isinstance(protocol, Mapping), "Lap S5 protocol must be a mapping.")
    _require(
        set(protocol) == {"protocol", "source_preset", "omega", "epoch_schedule"},
        "Lap S5 protocol has unexpected or missing metadata fields.",
    )
    _require(protocol.get("protocol") == "lap_s5", "Expected a Lap S5 protocol.")
    _require(
        protocol.get("source_preset") == SOURCE_PRESET_NAME,
        "Lap S5 protocol must retain the reference preset name.",
    )
    _require(
        protocol.get("omega") == OMEGA_METADATA,
        "Lap S5 protocol must retain OMEGA=0.5 as separate metadata.",
    )
    phases = protocol.get("epoch_schedule")
    _require(
        isinstance(phases, list) and len(phases) == 5,
        "Lap S5 protocol must contain exactly five phases.",
    )
    for index, phase in enumerate(phases):
        expected_keys = {"phase_id", "name", "start_epoch", "end_epoch", "params"}
        _require(
            isinstance(phase, Mapping) and set(phase) == expected_keys,
            f"Lap S5 phase {index + 1} has unexpected or missing fields.",
        )
        _require(
            phase["phase_id"] == _PHASE_IDS[index],
            f"Lap S5 phase {index + 1} has an invalid phase id.",
        )
        expected_start, expected_end = EXPECTED_BOUNDARIES[index]
        _require(
            phase["name"] == _EXPECTED_PHASE_NAMES[index],
            f"Lap S5 phase {index + 1} has an invalid source label.",
        )
        _require(
            (phase["start_epoch"], phase["end_epoch"])
            == (expected_start, expected_end),
            f"Lap S5 phase {index + 1} has invalid epoch boundaries.",
        )
        params = phase["params"]
        _require(
            isinstance(params, Mapping),
            f"Lap S5 phase {index + 1} params must be a mapping.",
        )
        expected_params = _EXPECTED_PARAMS[index]
        _require(
            set(params) == set(expected_params),
            f"Lap S5 phase {index + 1} has unexpected parameter keys.",
        )
        for key, expected_value in expected_params.items():
            value = params[key]
            if allow_scale_and_clip_overrides and key in _SCALE_KEYS:
                _positive_finite_number(value, f"{_PHASE_IDS[index]}.{key}")
            elif allow_scale_and_clip_overrides and key in _CLIP_KEYS:
                _validate_clip_threshold(value, f"{_PHASE_IDS[index]}.{key}")
            else:
                _require(
                    value == expected_value,
                    f"Lap S5 phase {index + 1} parameter {key!r} differs from the protocol.",
                )
        _require(
            "omega" not in params and "OMEGA" not in params,
            "OMEGA belongs in protocol metadata, not phase params.",
        )


def build_lap_s5_protocol(
    source_schedule: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a deep-copied, validated five-phase S5 protocol.

    ``source_schedule`` may be the raw preset (including its one-epoch entries)
    or an equivalent schedule. When omitted, the exact replay-bridge preset is
    loaded. OMEGA is kept as top-level metadata and never folded into a phase
    loss scale.
    """
    source = _load_reference_schedule() if source_schedule is None else source_schedule
    phases = _validate_source_schedule(source)
    protocol = {
        "protocol": "lap_s5",
        "source_preset": SOURCE_PRESET_NAME,
        "omega": OMEGA_METADATA,
        "epoch_schedule": _annotate_phases(phases),
    }
    _validate_protocol(protocol)
    return protocol


def apply_lap_s5_scale_overrides(
    protocol: Mapping[str, Any],
    scale_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    clip_thresholds: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy an S5 protocol and adjust only explicit scales and clip thresholds.

    Override mappings are keyed by phase id: ``anchor``, ``representation_40``,
    ``representation_10``, ``repair``, or ``consolidation``. The scale mapping
    accepts only ``vxc_loss_scale``, ``exc_loss_scale``, and
    ``reaction_grad_scale``. Optional clip thresholds are separate and accept
    only the three ``*_grad_clip`` fields. Boundaries, optimizer settings,
    accumulation, and all merge settings remain locked to the reference.
    """
    _validate_protocol(protocol, allow_scale_and_clip_overrides=True)
    scales = {} if scale_overrides is None else scale_overrides
    clips = {} if clip_thresholds is None else clip_thresholds
    _require(isinstance(scales, Mapping), "Scale overrides must be a phase-id mapping.")
    _require(isinstance(clips, Mapping), "Clip thresholds must be a phase-id mapping.")

    result = copy.deepcopy(dict(protocol))
    phases = {phase["phase_id"]: phase for phase in result["epoch_schedule"]}
    for section, allowed_keys, label in (
        (scales, _SCALE_KEYS, "scale override"),
        (clips, _CLIP_KEYS, "clip threshold"),
    ):
        for phase_id, overrides in section.items():
            _require(
                phase_id in phases, f"Unknown S5 phase id in {label}: {phase_id!r}."
            )
            _require(
                isinstance(overrides, Mapping),
                f"{label.title()} for {phase_id!r} must be a mapping.",
            )
            invalid = set(overrides) - allowed_keys
            _require(
                not invalid,
                f"Disallowed {label} keys for {phase_id!r}: {sorted(invalid)}.",
            )
            for key, value in overrides.items():
                if key in _SCALE_KEYS:
                    _positive_finite_number(value, f"{phase_id}.{key}")
                else:
                    _validate_clip_threshold(value, f"{phase_id}.{key}")
                phases[phase_id]["params"][key] = copy.deepcopy(value)

    _validate_protocol(result, allow_scale_and_clip_overrides=True)
    return result


def _validate_clip_threshold(value: Any, label: str) -> None:
    if value is None or value == "none":
        return
    _positive_finite_number(value, label)


def build_lap_s5_phase_smoke_view(
    protocol: Mapping[str, Any] | None = None,
    *,
    per_phase_step_limit: int | None = None,
) -> dict[str, Any]:
    """Build a clearly marked smoke-only view, optionally truncating each phase.

    Smoke epochs are renumbered from one. Every row carries the original phase
    name and source epoch span, while its parameters are copied unchanged from
    the protocol. This view is explicitly marked as non-production so a short
    smoke run cannot be mistaken for the full five-hundred-epoch schedule.
    """
    source_protocol = build_lap_s5_protocol() if protocol is None else protocol
    _validate_protocol(source_protocol, allow_scale_and_clip_overrides=True)
    if per_phase_step_limit is not None:
        _require(
            isinstance(per_phase_step_limit, int)
            and not isinstance(per_phase_step_limit, bool)
            and per_phase_step_limit > 0,
            "per_phase_step_limit must be a positive integer or None.",
        )

    smoke_phases = []
    smoke_start = 1
    for source_phase in source_protocol["epoch_schedule"]:
        source_duration = source_phase["end_epoch"] - source_phase["start_epoch"] + 1
        smoke_duration = (
            source_duration
            if per_phase_step_limit is None
            else min(source_duration, per_phase_step_limit)
        )
        smoke_phases.append(
            {
                "phase_id": source_phase["phase_id"],
                "name": source_phase["name"],
                "source_phase_name": source_phase["name"],
                "source_start_epoch": source_phase["start_epoch"],
                "source_end_epoch": source_phase["end_epoch"],
                "start_epoch": smoke_start,
                "end_epoch": smoke_start + smoke_duration - 1,
                "step_limit": smoke_duration,
                "params": copy.deepcopy(source_phase["params"]),
            }
        )
        smoke_start += smoke_duration

    return {
        "protocol": "lap_s5_phase_smoke_view",
        "source_protocol": source_protocol["protocol"],
        "source_preset": source_protocol["source_preset"],
        "omega": source_protocol["omega"],
        "production_schedule": False,
        "per_phase_step_limit": per_phase_step_limit,
        "epoch_schedule": smoke_phases,
    }


__all__ = [
    "EXPECTED_BOUNDARIES",
    "OMEGA_METADATA",
    "SOURCE_PRESET_NAME",
    "apply_lap_s5_scale_overrides",
    "build_lap_s5_phase_smoke_view",
    "build_lap_s5_protocol",
]
