import argparse
import copy
import json
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
