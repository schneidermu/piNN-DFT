import argparse
import copy
import json
import math
import pickle
from pathlib import Path

from optuna_joint import (
    init_distributed,
    load_chk,
    load_mrks_dispersions,
    run_or_reuse_preoptimization,
    run_trial,
    set_random_seed,
    DEFAULT_MRKS_DISPERSIONS,
)


TRIAL_19_PARAMS = {
    "accum_iter": 3,
    "gradient_merge_strategy": "clip_then_sum",
    "lr_train": 3.588259475602772e-4,
    "reaction_grad_clip": "none",
    "reaction_grad_scale": 0.6,
    "vxc_grad_clip": 5.0,
    "vxc_loss_scale": 75,
    "exc_loss_scale": 1.0,
    "exc_grad_clip": "none",
    "exc_grad_scale": 1.0,
    "exc_gradient_merge_strategy": "sum",
    "epoch_schedule": [
        {
            "name": "clip_drive",
            "start_epoch": 1,
            "end_epoch": 72,
            "params": {
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
        },
        {
            "name": "sum_repair",
            "start_epoch": 73,
            "end_epoch": 160,
            "params": {
                "accum_iter": 2,
                "gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 40,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        },
        {
            "name": "fchem_polish",
            "start_epoch": 161,
            "end_epoch": 260,
            "params": {
                "accum_iter": 2,
                "gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 20,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        },
        {
            "name": "fchem_drive",
            "start_epoch": 261,
            "end_epoch": 380,
            "params": {
                "accum_iter": 2,
                "gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 10,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        },
        {
            "name": "fchem_finish",
            "start_epoch": 381,
            "end_epoch": 500,
            "params": {
                "accum_iter": 2,
                "gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 5,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        },
    ],
}


TRIAL_19_PHASE_NAMES = tuple(phase["name"] for phase in TRIAL_19_PARAMS["epoch_schedule"])


MICRO_SCHEDULE_PRESETS = {
    "sum_repair_plus15_finish_minus15": {
        "duration_deltas": {"sum_repair": 15, "fchem_finish": -15},
    },
    "fchem_polish_plus20_finish_minus20": {
        "duration_deltas": {"fchem_polish": 20, "fchem_finish": -20},
    },
    "fchem_polish_plus10_finish_minus10": {
        "duration_deltas": {"fchem_polish": 10, "fchem_finish": -10},
    },
    "exc_repair_compressed": {
        "duration_deltas": {"sum_repair": 10, "fchem_polish": 10, "fchem_drive": 20, "fchem_finish": -40},
        "param_overrides": {
            "sum_repair": {"reaction_grad_scale": 0.9, "vxc_loss_scale": 50, "exc_loss_scale": 1.5},
            "fchem_polish": {"gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.75, "vxc_loss_scale": 35, "exc_loss_scale": 3.0, "exc_grad_clip": 2.0, "exc_gradient_merge_strategy": "clip_then_sum"},
            "fchem_drive": {"reaction_grad_scale": 1.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.5},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "exc_repair_compressed_smooth": {
        "duration_deltas": {"sum_repair": 10, "fchem_polish": 10, "fchem_drive": 20, "fchem_finish": -40},
        "param_overrides": {
            "sum_repair": {"reaction_grad_scale": 0.9, "vxc_loss_scale": 50, "exc_loss_scale": 1.5},
            "fchem_polish": {"gradient_merge_strategy": "sum", "reaction_grad_scale": 0.75, "vxc_loss_scale": 35, "exc_loss_scale": 3.0, "exc_grad_clip": "none", "exc_gradient_merge_strategy": "sum"},
            "fchem_drive": {"reaction_grad_scale": 1.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.5},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "exc_repair_density_guard": {
        "duration_deltas": {"sum_repair": 10, "fchem_polish": 10, "fchem_drive": 20, "fchem_finish": -40},
        "param_overrides": {
            "sum_repair": {"reaction_grad_scale": 0.9, "vxc_loss_scale": 50, "exc_loss_scale": 1.5},
            "fchem_polish": {"gradient_merge_strategy": "sum", "reaction_grad_scale": 0.8, "vxc_loss_scale": 40, "exc_loss_scale": 2.0, "exc_grad_clip": "none", "exc_gradient_merge_strategy": "sum"},
            "fchem_drive": {"reaction_grad_scale": 1.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.5},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "late_exc_repair": {
        "duration_deltas": {"fchem_polish": -20, "fchem_drive": 60, "fchem_finish": -40},
        "param_overrides": {
            "fchem_drive": {"gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.75, "vxc_loss_scale": 15, "exc_loss_scale": 3.0, "exc_grad_clip": 2.0, "exc_gradient_merge_strategy": "clip_then_sum"},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "early_exc_repair": {
        "duration_deltas": {"sum_repair": 20, "fchem_polish": 20, "fchem_finish": -40},
        "param_overrides": {
            "sum_repair": {"reaction_grad_scale": 0.9, "vxc_loss_scale": 50, "exc_loss_scale": 1.5},
            "fchem_polish": {"gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.75, "vxc_loss_scale": 35, "exc_loss_scale": 3.0, "exc_grad_clip": 2.0, "exc_gradient_merge_strategy": "clip_then_sum"},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "finish_vxc7": {
        "param_overrides": {"fchem_finish": {"vxc_loss_scale": 7}},
    },
    "drive12_finish7": {
        "param_overrides": {
            "fchem_drive": {"vxc_loss_scale": 12},
            "fchem_finish": {"vxc_loss_scale": 7},
        },
    },
    "clip_minus10_sum_plus10": {
        "duration_deltas": {"clip_drive": -10, "sum_repair": 10},
    },
}
MICRO_SCHEDULE_PRESET_NAMES = tuple(MICRO_SCHEDULE_PRESETS)

def _phase_from_base(name, start_epoch, end_epoch, overrides=None):
    base_phase = next(phase for phase in TRIAL_19_PARAMS["epoch_schedule"] if phase["name"] == name)
    phase = copy.deepcopy(base_phase)
    phase["start_epoch"] = start_epoch
    phase["end_epoch"] = end_epoch
    if overrides:
        phase["params"].update(overrides)
    return phase


def _h9_repair_phase(
    start_epoch,
    end_epoch,
    *,
    exc_loss_scale=3.0,
    vxc_loss_scale=15,
    reaction_grad_scale=0.75,
):
    return _phase_from_base(
        "fchem_drive",
        start_epoch,
        end_epoch,
        {
            "gradient_merge_strategy": "clip_then_sum",
            "reaction_grad_scale": reaction_grad_scale,
            "vxc_loss_scale": vxc_loss_scale,
            "exc_loss_scale": exc_loss_scale,
            "exc_grad_clip": 2.0,
            "exc_gradient_merge_strategy": "clip_then_sum",
        },
    )


def _h9_soft_landing_phase(
    start_epoch,
    end_epoch,
    *,
    exc_loss_scale,
    vxc_loss_scale,
    reaction_grad_scale,
):
    phase = _h9_repair_phase(
        start_epoch,
        end_epoch,
        exc_loss_scale=exc_loss_scale,
        vxc_loss_scale=vxc_loss_scale,
        reaction_grad_scale=reaction_grad_scale,
    )
    phase["name"] = "h9_soft_landing"
    return phase


def _build_h9_schedule(
    *,
    repair_start=241,
    repair_end=420,
    repair_exc_loss_scale=3.0,
    repair_vxc_loss_scale=15,
    repair_reaction_grad_scale=0.75,
    soft_landing=None,
):
    if repair_start < 241 or repair_end < repair_start:
        raise ValueError("Invalid h9 repair window.")

    schedule = [
        _phase_from_base("clip_drive", 1, 72),
        _phase_from_base("sum_repair", 73, 160),
        _phase_from_base("fchem_polish", 161, 240),
    ]

    if repair_start > 241:
        schedule.append(_phase_from_base("fchem_drive", 241, repair_start - 1))

    schedule.append(
        _h9_repair_phase(
            repair_start,
            repair_end,
            exc_loss_scale=repair_exc_loss_scale,
            vxc_loss_scale=repair_vxc_loss_scale,
            reaction_grad_scale=repair_reaction_grad_scale,
        )
    )

    next_epoch = repair_end + 1
    if soft_landing:
        duration, kwargs = soft_landing
        if duration <= 0:
            raise ValueError("h9 soft-landing duration must be positive.")
        soft_end = next_epoch + duration - 1
        schedule.append(_h9_soft_landing_phase(next_epoch, soft_end, **kwargs))
        next_epoch = soft_end + 1

    if next_epoch > 500:
        raise ValueError("h9 repair and soft-landing phases exceed 500 epochs.")

    schedule.append(
        _phase_from_base(
            "fchem_finish",
            next_epoch,
            500,
            {"vxc_loss_scale": 7},
        )
    )
    return schedule


H9_SCHEDULE_PRESETS = {
    # Exploration: identify which part of h9's repair window and force matters.
    "h9_explore_short_repair": _build_h9_schedule(repair_end=380),
    "h9_explore_long_repair": _build_h9_schedule(repair_end=460),
    "h9_explore_delayed_repair": _build_h9_schedule(repair_start=281, repair_end=440),
    "h9_explore_exc4": _build_h9_schedule(repair_exc_loss_scale=4.0),
    "h9_explore_reaction1": _build_h9_schedule(repair_reaction_grad_scale=1.0),
    # Exploitation: retain h9's repair and replace its destructive hard finish.
    "h9_exploit_soft_exc2_vxc10_r085": _build_h9_schedule(
        soft_landing=(
            40,
            {"exc_loss_scale": 2.0, "vxc_loss_scale": 10, "reaction_grad_scale": 0.85},
        )
    ),
    "h9_exploit_soft_exc2_vxc15_r075": _build_h9_schedule(
        soft_landing=(
            40,
            {"exc_loss_scale": 2.0, "vxc_loss_scale": 15, "reaction_grad_scale": 0.75},
        )
    ),
    "h9_exploit_soft_exc25_vxc12_r080": _build_h9_schedule(
        soft_landing=(
            40,
            {"exc_loss_scale": 2.5, "vxc_loss_scale": 12, "reaction_grad_scale": 0.80},
        )
    ),
    "h9_exploit_repair440_soft30": _build_h9_schedule(
        repair_end=440,
        soft_landing=(30, {"exc_loss_scale": 2.0, "vxc_loss_scale": 12, "reaction_grad_scale": 0.80}),
    ),
    "h9_exploit_soft60": _build_h9_schedule(
        soft_landing=(
            60,
            {"exc_loss_scale": 2.0, "vxc_loss_scale": 10, "reaction_grad_scale": 0.75},
        )
    ),
}
H9_SCHEDULE_PRESET_NAMES = tuple(H9_SCHEDULE_PRESETS)


def _build_e3_schedule(
    *,
    repair_start=281,
    repair_end=440,
    pre_drive_overrides=None,
    repair_exc_loss_scale=3.0,
    repair_vxc_loss_scale=15,
    repair_reaction_grad_scale=0.75,
    finish_overrides=None,
):
    if repair_start < 241 or repair_end < repair_start or repair_end >= 500:
        raise ValueError("Invalid e3 repair window.")

    schedule = [
        _phase_from_base("clip_drive", 1, 72),
        _phase_from_base("sum_repair", 73, 160),
        _phase_from_base("fchem_polish", 161, 240),
    ]
    if repair_start > 241:
        pre_drive = _phase_from_base(
            "fchem_drive",
            241,
            repair_start - 1,
            pre_drive_overrides,
        )
        pre_drive["name"] = "e3_pre_drive"
        schedule.append(pre_drive)
    repair = _h9_repair_phase(
        repair_start,
        repair_end,
        exc_loss_scale=repair_exc_loss_scale,
        vxc_loss_scale=repair_vxc_loss_scale,
        reaction_grad_scale=repair_reaction_grad_scale,
    )
    repair["name"] = "e3_repair"
    schedule.append(repair)
    finish_params = {"vxc_loss_scale": 7}
    if finish_overrides:
        finish_params.update(finish_overrides)
    finish = _phase_from_base("fchem_finish", repair_end + 1, 500, finish_params)
    finish["name"] = "e3_finish"
    schedule.append(finish)
    return schedule


def _build_e3_split_repair_schedule():
    schedule = [
        _phase_from_base("clip_drive", 1, 72),
        _phase_from_base("sum_repair", 73, 160),
        _phase_from_base("fchem_polish", 161, 240),
    ]
    pre_drive = _phase_from_base("fchem_drive", 241, 280)
    pre_drive["name"] = "e3_pre_drive"
    schedule.append(pre_drive)
    first_repair = _h9_repair_phase(281, 360)
    first_repair["name"] = "e3_repair_a"
    schedule.append(first_repair)
    gap_drive = _phase_from_base("fchem_drive", 361, 400)
    gap_drive["name"] = "e3_gap_drive"
    schedule.append(gap_drive)
    second_repair = _h9_repair_phase(401, 440)
    second_repair["name"] = "e3_repair_b"
    schedule.append(second_repair)
    finish = _phase_from_base("fchem_finish", 441, 500, {"vxc_loss_scale": 7})
    finish["name"] = "e3_finish"
    schedule.append(finish)
    return schedule


E3_SCHEDULE_PRESETS = {
    # Temporal map: determine the viable delayed-repair region.
    "e3_onset261_end440": _build_e3_schedule(repair_start=261),
    "e3_onset301_end440": _build_e3_schedule(repair_start=301),
    "e3_onset281_end420": _build_e3_schedule(repair_end=420),
    "e3_onset281_end460": _build_e3_schedule(repair_end=460),
    # Preconditioning: test whether elapsed time or the pre-repair trajectory creates e3's basin.
    "e3_pre_reaction075": _build_e3_schedule(
        pre_drive_overrides={"reaction_grad_scale": 0.75},
    ),
    "e3_pre_density_guard": _build_e3_schedule(
        pre_drive_overrides={
            "reaction_grad_scale": 0.75,
            "vxc_loss_scale": 20,
            "exc_loss_scale": 1.5,
        },
    ),
    # Repair balance: e4 rules out higher E_xc force; test the opposite side and VXC pressure.
    "e3_repair_exc2": _build_e3_schedule(repair_exc_loss_scale=2.0),
    "e3_repair_vxc10": _build_e3_schedule(repair_vxc_loss_scale=10),
    "e3_finish_reaction075": _build_e3_schedule(
        finish_overrides={"reaction_grad_scale": 0.75},
    ),
    # Structure: test whether contiguous clipped repair is necessary.
    "e3_split_repair": _build_e3_split_repair_schedule(),
}
E3_SCHEDULE_PRESET_NAMES = tuple(E3_SCHEDULE_PRESETS)


def _build_simple_guard_schedule(*, guard, repair_end):
    """Build a three-stage homotopy/repair/consolidation schedule."""
    if not 0.0 <= guard <= 1.0:
        raise ValueError("Simple guard must be in [0, 1].")
    if repair_end not in {420, 440}:
        raise ValueError("Simple repair end must be 420 or 440.")

    endpoint_vxc = 10.0 + 30.0 * guard
    endpoint_exc = 1.0 + guard
    endpoint_reaction = 1.0 - 0.2 * guard
    homotopy = []
    for epoch in range(1, 281):
        fraction = (epoch - 1) / 279.0
        phase = _phase_from_base(
            "fchem_drive",
            epoch,
            epoch,
            {
                "accum_iter": 2,
                "gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 0.6 + fraction * (endpoint_reaction - 0.6),
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 75.0 + fraction * (endpoint_vxc - 75.0),
                "exc_loss_scale": 1.0 + fraction * (endpoint_exc - 1.0),
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        )
        phase["name"] = "simple_joint_homotopy"
        homotopy.append(phase)

    repair = _h9_repair_phase(281, repair_end)
    repair["name"] = "simple_energy_repair"
    finish = _phase_from_base(
        "fchem_finish",
        repair_end + 1,
        500,
        {"vxc_loss_scale": 7},
    )
    finish["name"] = "simple_joint_consolidation"
    return [*homotopy, repair, finish]


SIMPLE_SCHEDULE_PRESETS = {
    f"simple_s{2 * index + offset}": _build_simple_guard_schedule(
        guard=guard,
        repair_end=repair_end,
    )
    for index, guard in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
    for offset, repair_end in ((1, 420), (2, 440))
}
SIMPLE_SCHEDULE_PRESET_NAMES = tuple(SIMPLE_SCHEDULE_PRESETS)


OCCAM_VXC_SCALE = 64.0
OCCAM_REPRESENTATION_END = 300
OCCAM_REPAIR_END = 400


def _occam_progress(fraction, path):
    if path == "geometric" or path == "linear":
        return fraction
    if path == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * fraction))
    if path == "quarter_step":
        return 0.0 if fraction < 0.25 else 1.0
    if path == "half_step":
        return 0.0 if fraction < 0.5 else 1.0
    raise ValueError(f"Unknown Occam representation path: {path}")


def _build_occam_schedule(*, path, repair_multiplier):
    """Build a 300/100/100 joint-repair-joint curriculum.

    The Vxc anchor is the nearest power of two to the measured initial
    Fchem/Vxc gradient-norm ratio (~73). All remaining coefficients are tied
    to powers of two, leaving path shape and repair strength as the only
    experimental factors.
    """
    if repair_multiplier not in {2.0, 3.0}:
        raise ValueError("Occam repair multiplier must be 2 or 3.")

    representation = []
    for epoch in range(1, OCCAM_REPRESENTATION_END + 1):
        fraction = (epoch - 1) / (OCCAM_REPRESENTATION_END - 1)
        progress = _occam_progress(fraction, path)
        if path == "linear":
            reaction_scale = 0.5 + 0.5 * progress
            vxc_scale = OCCAM_VXC_SCALE - 0.75 * OCCAM_VXC_SCALE * progress
        else:
            reaction_scale = 2.0 ** (progress - 1.0)
            vxc_scale = OCCAM_VXC_SCALE * 2.0 ** (-2.0 * progress)

        anchored = fraction < 0.25
        phase = _phase_from_base(
            "fchem_drive",
            epoch,
            epoch,
            {
                "accum_iter": 2,
                "gradient_merge_strategy": "clip_then_sum" if anchored else "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": reaction_scale,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": vxc_scale,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
                "exc_gradient_merge_strategy": "sum",
            },
        )
        phase["name"] = "occam_representation"
        representation.append(phase)

    repair = _phase_from_base(
        "fchem_drive",
        OCCAM_REPRESENTATION_END + 1,
        OCCAM_REPAIR_END,
        {
            "accum_iter": 2,
            "gradient_merge_strategy": "clip_then_sum",
            "reaction_grad_clip": "none",
            "reaction_grad_scale": repair_multiplier / (repair_multiplier + 1.0),
            "vxc_grad_clip": 2.0,
            "vxc_loss_scale": OCCAM_VXC_SCALE / 4.0,
            "exc_loss_scale": repair_multiplier,
            "exc_grad_clip": 2.0,
            "exc_grad_scale": 1.0,
            "exc_gradient_merge_strategy": "clip_then_sum",
        },
    )
    repair["name"] = "occam_energy_repair"

    consolidation = _phase_from_base(
        "fchem_finish",
        OCCAM_REPAIR_END + 1,
        500,
        {
            "accum_iter": 2,
            "gradient_merge_strategy": "sum",
            "reaction_grad_clip": "none",
            "reaction_grad_scale": 1.0,
            "vxc_grad_clip": 2.0,
            "vxc_loss_scale": OCCAM_VXC_SCALE / 8.0,
            "exc_loss_scale": 1.0,
            "exc_grad_clip": "none",
            "exc_grad_scale": 1.0,
            "exc_gradient_merge_strategy": "sum",
        },
    )
    consolidation["name"] = "occam_joint_consolidation"
    return [*representation, repair, consolidation]


_OCCAM_PATHS = ("geometric", "cosine", "linear", "quarter_step", "half_step")
OCCAM_SCHEDULE_PRESETS = {
    f"occam_{path}_{repair_name}": _build_occam_schedule(
        path=path,
        repair_multiplier=repair_multiplier,
    )
    for path in _OCCAM_PATHS
    for repair_name, repair_multiplier in (("double", 2.0), ("triple", 3.0))
}
OCCAM_SCHEDULE_PRESET_NAMES = tuple(OCCAM_SCHEDULE_PRESETS)


def _apply_duration_deltas(params, duration_deltas):
    phases = params["epoch_schedule"]
    durations = {
        phase["name"]: phase["end_epoch"] - phase["start_epoch"] + 1
        for phase in phases
    }
    for phase_name, delta in duration_deltas.items():
        if phase_name not in durations:
            raise ValueError(f"Unknown Trial 19 phase in duration delta: {phase_name}")
        durations[phase_name] += delta
        if durations[phase_name] <= 0:
            raise ValueError(f"Non-positive duration for phase {phase_name}: {durations[phase_name]}")

    original_total = sum(
        phase["end_epoch"] - phase["start_epoch"] + 1
        for phase in phases
    )
    new_total = sum(durations.values())
    if new_total != original_total:
        raise ValueError(
            f"Micro schedule duration deltas must preserve total epochs: "
            f"{new_total} != {original_total}"
        )

    start_epoch = 1
    for phase in phases:
        duration = durations[phase["name"]]
        phase["start_epoch"] = start_epoch
        phase["end_epoch"] = start_epoch + duration - 1
        start_epoch = phase["end_epoch"] + 1


def _apply_param_overrides(params, param_overrides):
    phases_by_name = {phase["name"]: phase for phase in params["epoch_schedule"]}
    for phase_name, overrides in param_overrides.items():
        if phase_name not in phases_by_name:
            raise ValueError(f"Unknown Trial 19 phase in param override: {phase_name}")
        phases_by_name[phase_name]["params"].update(overrides)


def apply_micro_schedule_preset(params, preset_name):
    if preset_name is None:
        return params
    if preset_name not in MICRO_SCHEDULE_PRESETS:
        raise ValueError(f"Unknown Trial 19 micro schedule preset: {preset_name}")

    preset = MICRO_SCHEDULE_PRESETS[preset_name]
    if preset.get("duration_deltas"):
        _apply_duration_deltas(params, preset["duration_deltas"])
    if preset.get("param_overrides"):
        _apply_param_overrides(params, preset["param_overrides"])
    return params


def build_trial_19_params(extend_phase=None, extend_epochs=0):
    params = copy.deepcopy(TRIAL_19_PARAMS)
    if extend_epochs < 0:
        raise ValueError("--extend-epochs must be non-negative.")
    if extend_epochs == 0:
        return params
    if extend_phase is None:
        raise ValueError("--extend-phase is required when --extend-epochs is non-zero.")

    found_phase = False
    downstream_shift = 0
    for phase in params["epoch_schedule"]:
        if downstream_shift:
            phase["start_epoch"] += downstream_shift
            phase["end_epoch"] += downstream_shift
        if phase["name"] == extend_phase:
            phase["end_epoch"] += extend_epochs
            downstream_shift += extend_epochs
            found_phase = True

    if not found_phase:
        raise ValueError(f"Unknown Trial 19 phase for extension: {extend_phase}")
    return params


def resolve_trial_19_params(args):
    if args.occam_schedule_preset:
        if (
            args.simple_schedule_preset
            or args.e3_schedule_preset
            or args.h9_schedule_preset
            or args.micro_schedule_preset
            or args.extend_epochs
        ):
            raise ValueError("--occam-schedule-preset cannot be combined with other schedule options.")
        params = copy.deepcopy(TRIAL_19_PARAMS)
        params["epoch_schedule"] = copy.deepcopy(
            OCCAM_SCHEDULE_PRESETS[args.occam_schedule_preset]
        )
        return params
    if args.simple_schedule_preset:
        if (
            args.e3_schedule_preset
            or args.h9_schedule_preset
            or args.micro_schedule_preset
            or args.extend_epochs
        ):
            raise ValueError("--simple-schedule-preset cannot be combined with other schedule options.")
        params = copy.deepcopy(TRIAL_19_PARAMS)
        params["epoch_schedule"] = copy.deepcopy(
            SIMPLE_SCHEDULE_PRESETS[args.simple_schedule_preset]
        )
        return params
    if args.e3_schedule_preset:
        if args.h9_schedule_preset or args.micro_schedule_preset or args.extend_epochs:
            raise ValueError("--e3-schedule-preset cannot be combined with other schedule options.")
        params = copy.deepcopy(TRIAL_19_PARAMS)
        params["epoch_schedule"] = copy.deepcopy(E3_SCHEDULE_PRESETS[args.e3_schedule_preset])
        return params
    if args.h9_schedule_preset:
        if args.micro_schedule_preset or args.extend_epochs:
            raise ValueError("--h9-schedule-preset cannot be combined with micro or extension options.")
        params = copy.deepcopy(TRIAL_19_PARAMS)
        params["epoch_schedule"] = copy.deepcopy(H9_SCHEDULE_PRESETS[args.h9_schedule_preset])
        return params
    if args.micro_schedule_preset and args.extend_epochs:
        raise ValueError("--micro-schedule-preset cannot be combined with --extend-epochs.")
    params = build_trial_19_params(args.extend_phase, args.extend_epochs)
    return apply_micro_schedule_preset(params, args.micro_schedule_preset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Trial 19 bridge schedule and save the final checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_64")
    parser.add_argument("--model-type", type=str, default="base", choices=["base", "log", "gc_svelu_mirror", "gc_softplus_mirror", "gc_softplus_mirror_r2scan_alpha"])
    parser.add_argument("--n-predopt", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--extend-phase", type=str, default=None, choices=TRIAL_19_PHASE_NAMES)
    parser.add_argument("--extend-epochs", type=int, default=0)
    parser.add_argument("--micro-schedule-preset", type=str, default=None, choices=MICRO_SCHEDULE_PRESET_NAMES)
    parser.add_argument("--h9-schedule-preset", type=str, default=None, choices=H9_SCHEDULE_PRESET_NAMES)
    parser.add_argument("--e3-schedule-preset", type=str, default=None, choices=E3_SCHEDULE_PRESET_NAMES)
    parser.add_argument(
        "--simple-schedule-preset",
        type=str,
        default=None,
        choices=SIMPLE_SCHEDULE_PRESET_NAMES,
    )
    parser.add_argument(
        "--occam-schedule-preset",
        type=str,
        default=None,
        choices=OCCAM_SCHEDULE_PRESET_NAMES,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vxc-batch-size", type=int, default=1)
    parser.add_argument("--lr-predopt", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-vxc", type=int, default=2)
    parser.add_argument("--preopt-vxc-weight", type=float, default=0.0)
    parser.add_argument("--preopt-vxc-steps", type=int, default=0)
    parser.add_argument("--preopt-vxc-target", type=str, default="pbe", choices=["pbe"])
    parser.add_argument("--trial-number", type=int, default=19)
    parser.add_argument("--train-fchem-target", type=float, default=40.0)
    parser.add_argument("--val-vxc-target", type=float, default=1.1)
    parser.add_argument("--val-fchem-soft-cap", type=float, default=90.0)
    parser.add_argument("--save-selected-checkpoints", action="store_true", default=True)
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument("--mrks-dispersions-pickle", type=str, default=str(DEFAULT_MRKS_DISPERSIONS))
    parser.add_argument(
        "--no-reaction-dispersion",
        action="store_true",
        help="Do not add precomputed D3 dispersion corrections in reaction-energy training/validation.",
    )
    return parser.parse_args()


def select_last_epoch(epoch_history):
    if not epoch_history:
        raise ValueError("Cannot select the last epoch from empty history.")
    return epoch_history[-1]


def last_epoch_checkpoint_key(row):
    return (-int(row["epoch"]),)


def main() -> None:
    args = parse_args()
    trial_19_params = resolve_trial_19_params(args)
    schedule_end_epoch = trial_19_params["epoch_schedule"][-1]["end_epoch"]
    if args.n_train < schedule_end_epoch:
        raise ValueError(
            f"--n-train={args.n_train} is shorter than the Trial 19 schedule end epoch "
            f"{schedule_end_epoch}."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, world_size, device, rank0 = init_distributed()
    set_random_seed(args.seed + args.trial_number)

    if args.no_reaction_dispersion:
        dispersions = {}
    else:
        with (Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle").open("rb") as handle:
            dispersions = pickle.load(handle)
    mrks_dispersions = (
        load_mrks_dispersions(args.mrks_dispersions_pickle)
        if args.include_mrks_dispersion
        else None
    )

    data_predopt, data_train, data_val, data_vxc_train, data_vxc_val = load_chk(path=args.checkpoints_dir)
    shared_preopt_checkpoint = run_or_reuse_preoptimization(
        args=args,
        output_dir=output_dir,
        data_predopt=data_predopt,
        data_vxc_train=data_vxc_train,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        rank0=rank0,
    )

    result = run_trial(
        trial_number=args.trial_number,
        params=trial_19_params,
        args=args,
        shared_preopt_checkpoint=Path(shared_preopt_checkpoint),
        data_train=data_train,
        data_val=data_val,
        data_vxc_train=data_vxc_train,
        data_vxc_val=data_vxc_val,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        dispersions=dispersions,
        mrks_dispersions=mrks_dispersions,
        output_dir=output_dir,
        rank0=rank0,
        epoch_selector=select_last_epoch,
        checkpoint_row_key=last_epoch_checkpoint_key,
    )

    if rank0:
        final_epoch = select_last_epoch(result["epoch_history"])
        print("Replay complete for Trial 19 bridge schedule.")
        print(f"Final selected epoch: {final_epoch['epoch']}")
        print(
            "Final metrics: "
            f"train_fchem={float(final_epoch['train_fchem']):.8f}, "
            f"val_vxc={float(final_epoch['val_vxc']):.8f}, "
            f"val_exc={float(final_epoch['val_exc']):.8f}, "
            f"val_fchem={float(final_epoch['val_fchem']):.8f}, "
            f"phase={final_epoch.get('phase_name')}"
        )
        print(f"Selected checkpoint: {result.get('selected_checkpoint_path')}")
        print(f"History path: {result.get('history_path')}")
        print(f"Params: {json.dumps(trial_19_params, sort_keys=True)}")


if __name__ == "__main__":
    main()
