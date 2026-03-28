import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import torch.distributed as dist

from optuna_joint import (
    FAIL_TRIAL,
    RUN_TRIAL,
    STOP_TRIALS,
    broadcast_payload,
    init_distributed,
    load_chk,
    run_or_reuse_preoptimization,
    run_trial,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Goal-driven Optuna search for same-epoch train_fchem < target and val_vxc < target. "
            "Includes bridge schedules that finish with a late `sum` repair phase."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_64")
    parser.add_argument("--n-predopt", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vxc-batch-size", type=int, default=1)
    parser.add_argument("--lr-predopt", type=float, default=2e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--save-selected-checkpoints", action="store_true")
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-vxc", type=int, default=2)
    parser.add_argument("--preopt-vxc-weight", type=float, default=0.0)
    parser.add_argument("--preopt-vxc-steps", type=int, default=0)
    parser.add_argument("--preopt-vxc-target", type=str, default="pbe", choices=["pbe"])
    parser.add_argument("--train-fchem-target", type=float, default=40.0)
    parser.add_argument("--val-vxc-target", type=float, default=1.1)
    parser.add_argument(
        "--val-fchem-soft-cap",
        type=float,
        default=90.0,
        help=(
            "Soft tie-break guard for selecting same-epoch goal candidates. "
            "This does not override the main goal; it only prefers less pathological val_fchem when the box score is tied."
        ),
    )
    parser.add_argument("--sampler-startup-trials", type=int, default=12)
    parser.add_argument("--enqueue-recommended", action="store_true")
    return parser.parse_args()


def goal_epoch_components(
    epoch_row: Dict[str, Any],
    train_fchem_target: float,
    val_vxc_target: float,
    val_fchem_soft_cap: Optional[float],
) -> Dict[str, float]:
    train_fchem = float(epoch_row["train_fchem"])
    val_vxc = float(epoch_row["val_vxc"])
    val_fchem = float(epoch_row["val_fchem"])

    train_ratio = train_fchem / train_fchem_target
    vxc_ratio = val_vxc / val_vxc_target
    box_score = max(train_ratio, vxc_ratio)
    l1_score = train_ratio + vxc_ratio

    if val_fchem_soft_cap is None or val_fchem_soft_cap <= 0.0:
        val_fchem_guard_ratio = 1.0
    else:
        val_fchem_guard_ratio = max(val_fchem / val_fchem_soft_cap, 1.0)

    # Scalar objective for targeted optimization.
    objective = box_score + 1.0e-3 * l1_score + 1.0e-5 * val_fchem_guard_ratio

    return {
        "goal_objective": objective,
        "goal_box_score": box_score,
        "goal_l1_score": l1_score,
        "goal_train_ratio": train_ratio,
        "goal_vxc_ratio": vxc_ratio,
        "goal_train_gap": train_fchem - train_fchem_target,
        "goal_vxc_gap": val_vxc - val_vxc_target,
        "goal_val_fchem_guard_ratio": val_fchem_guard_ratio,
        "goal_joint_hit": train_fchem < train_fchem_target and val_vxc < val_vxc_target,
    }


def goal_epoch_key(
    epoch_row: Dict[str, Any],
    train_fchem_target: float,
    val_vxc_target: float,
    val_fchem_soft_cap: Optional[float],
) -> tuple:
    parts = goal_epoch_components(
        epoch_row=epoch_row,
        train_fchem_target=train_fchem_target,
        val_vxc_target=val_vxc_target,
        val_fchem_soft_cap=val_fchem_soft_cap,
    )
    return (
        parts["goal_box_score"],
        parts["goal_l1_score"],
        parts["goal_val_fchem_guard_ratio"],
        float(epoch_row["val_vxc"]),
        float(epoch_row["train_fchem"]),
        float(epoch_row["val_fchem"]),
        int(epoch_row["epoch"]),
    )


def best_epoch_by_goal_box(
    epoch_history: List[Dict[str, Any]],
    train_fchem_target: float,
    val_vxc_target: float,
    val_fchem_soft_cap: Optional[float],
) -> Dict[str, Any]:
    if not epoch_history:
        raise ValueError("Cannot select goal-best epoch from empty history.")
    return min(
        epoch_history,
        key=lambda row: goal_epoch_key(
            epoch_row=row,
            train_fchem_target=train_fchem_target,
            val_vxc_target=val_vxc_target,
            val_fchem_soft_cap=val_fchem_soft_cap,
        ),
    )


def schedule_switch_bounds(n_train: int) -> tuple[int, int]:
    lower = 40 if n_train >= 72 else max(16, n_train // 2)
    raw_upper = max(lower, n_train - 12)
    upper = lower + 4 * ((raw_upper - lower) // 4)
    return lower, upper


def suggest_goal_bridge_params(trial: optuna.trial.Trial, n_train: int) -> Dict[str, Any]:
    family = trial.suggest_categorical(
        "search_family",
        [
            "clip_static_refine",
            "clip_to_sum_repair",
        ],
    )

    if family == "clip_static_refine":
        params = {
            "gradient_merge_strategy": "clip_then_sum",
            "accum_iter": trial.suggest_categorical("clip_static_accum_iter", [1, 2, 3]),
            "vxc_loss_scale": trial.suggest_categorical("clip_static_vxc_loss_scale", [50, 75, 100]),
            "reaction_grad_clip": "none",
            "reaction_grad_scale": trial.suggest_categorical("clip_static_reaction_grad_scale", [0.6, 0.7, 0.8]),
            "vxc_grad_clip": trial.suggest_categorical("clip_static_vxc_grad_clip", [2.0, 3.0, 5.0]),
            "lr_train": trial.suggest_float("clip_static_lr_train", 2.2e-4, 3.8e-4, log=True),
        }
    else:
        switch_min, switch_max = schedule_switch_bounds(n_train)
        switch_epoch = trial.suggest_int("repair_switch_epoch", switch_min, switch_max, step=4)
        drive_accum_iter = trial.suggest_categorical("repair_drive_accum_iter", [1, 2, 3])
        drive_vxc_loss_scale = trial.suggest_categorical("repair_drive_vxc_loss_scale", [50, 75, 100])
        drive_reaction_grad_scale = trial.suggest_categorical("repair_drive_reaction_grad_scale", [0.6, 0.7])
        drive_vxc_grad_clip = trial.suggest_categorical("repair_drive_vxc_grad_clip", [3.0, 5.0])
        repair_accum_iter = trial.suggest_categorical("repair_sum_accum_iter", [2, 3])
        repair_vxc_loss_scale = trial.suggest_categorical("repair_sum_vxc_loss_scale", [40, 50, 60])
        lr_train = trial.suggest_float("repair_lr_train", 2.3e-4, 3.6e-4, log=True)

        params = {
            "gradient_merge_strategy": "clip_then_sum",
            "accum_iter": drive_accum_iter,
            "vxc_loss_scale": drive_vxc_loss_scale,
            "reaction_grad_clip": "none",
            "reaction_grad_scale": drive_reaction_grad_scale,
            "vxc_grad_clip": drive_vxc_grad_clip,
            "lr_train": lr_train,
            "epoch_schedule": [
                {
                    "name": "clip_drive",
                    "start_epoch": 1,
                    "end_epoch": switch_epoch,
                    "params": {
                        "gradient_merge_strategy": "clip_then_sum",
                        "accum_iter": drive_accum_iter,
                        "vxc_loss_scale": drive_vxc_loss_scale,
                        "reaction_grad_clip": "none",
                        "reaction_grad_scale": drive_reaction_grad_scale,
                        "vxc_grad_clip": drive_vxc_grad_clip,
                    },
                },
                {
                    "name": "sum_repair",
                    "start_epoch": switch_epoch + 1,
                    "end_epoch": n_train,
                    "params": {
                        "gradient_merge_strategy": "sum",
                        "accum_iter": repair_accum_iter,
                        "vxc_loss_scale": repair_vxc_loss_scale,
                        # Explicitly retained for saved payload readability.
                        "reaction_grad_clip": "none",
                        "reaction_grad_scale": 1.0,
                        "vxc_grad_clip": 2.0,
                    },
                },
            ],
        }

    params["search_family"] = family
    return params


def recommended_trials(n_train: int) -> List[Dict[str, Any]]:
    switch_min, switch_max = schedule_switch_bounds(n_train)

    def clamp_switch(epoch: int) -> int:
        return min(max(epoch, switch_min), switch_max)

    return [
        {
            "search_family": "clip_static_refine",
            "clip_static_accum_iter": 2,
            "clip_static_vxc_loss_scale": 100,
            "clip_static_reaction_grad_scale": 0.7,
            "clip_static_vxc_grad_clip": 3.0,
            "clip_static_lr_train": 3.351653892770322e-4,
        },
        {
            "search_family": "clip_static_refine",
            "clip_static_accum_iter": 1,
            "clip_static_vxc_loss_scale": 100,
            "clip_static_reaction_grad_scale": 0.7,
            "clip_static_vxc_grad_clip": 3.0,
            "clip_static_lr_train": 2.481397263087066e-4,
        },
        {
            "search_family": "clip_static_refine",
            "clip_static_accum_iter": 3,
            "clip_static_vxc_loss_scale": 50,
            "clip_static_reaction_grad_scale": 0.7,
            "clip_static_vxc_grad_clip": 5.0,
            "clip_static_lr_train": 2.5e-4,
        },
        {
            "search_family": "clip_to_sum_repair",
            "repair_drive_accum_iter": 2,
            "repair_drive_vxc_loss_scale": 100,
            "repair_drive_reaction_grad_scale": 0.7,
            "repair_drive_vxc_grad_clip": 3.0,
            "repair_sum_accum_iter": 3,
            "repair_sum_vxc_loss_scale": 50,
            "repair_lr_train": 3.351653892770322e-4,
            "repair_switch_epoch": clamp_switch(56),
        },
        {
            "search_family": "clip_to_sum_repair",
            "repair_drive_accum_iter": 2,
            "repair_drive_vxc_loss_scale": 100,
            "repair_drive_reaction_grad_scale": 0.7,
            "repair_drive_vxc_grad_clip": 3.0,
            "repair_sum_accum_iter": 3,
            "repair_sum_vxc_loss_scale": 50,
            "repair_lr_train": 3.351653892770322e-4,
            "repair_switch_epoch": clamp_switch(64),
        },
        {
            "search_family": "clip_to_sum_repair",
            "repair_drive_accum_iter": 1,
            "repair_drive_vxc_loss_scale": 100,
            "repair_drive_reaction_grad_scale": 0.7,
            "repair_drive_vxc_grad_clip": 3.0,
            "repair_sum_accum_iter": 3,
            "repair_sum_vxc_loss_scale": 50,
            "repair_lr_train": 2.481397263087066e-4,
            "repair_switch_epoch": clamp_switch(64),
        },
        {
            "search_family": "clip_to_sum_repair",
            "repair_drive_accum_iter": 3,
            "repair_drive_vxc_loss_scale": 50,
            "repair_drive_reaction_grad_scale": 0.7,
            "repair_drive_vxc_grad_clip": 5.0,
            "repair_sum_accum_iter": 3,
            "repair_sum_vxc_loss_scale": 50,
            "repair_lr_train": 2.5e-4,
            "repair_switch_epoch": clamp_switch(64),
        },
    ]


def summarize_goal_study(
    study: optuna.study.Study,
    train_fchem_target: float,
    val_vxc_target: float,
) -> None:
    def fmt_float(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.8f}"

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        print("No completed trials.")
        return

    goal_hits = [trial for trial in completed_trials if trial.user_attrs.get("goal_joint_hit")]
    best_trials = sorted(completed_trials, key=lambda trial: float(trial.value))[:10]

    print(
        "\nGoal summary: "
        f"hits={len(goal_hits)}/{len(completed_trials)} "
        f"for train_fchem < {train_fchem_target:.3f} and val_vxc < {val_vxc_target:.4f}"
    )

    if goal_hits:
        print("\nExact goal hits:")
        for trial in sorted(goal_hits, key=lambda item: float(item.value))[:10]:
            print(
                f"  Trial {trial.number}: objective={float(trial.value):.8f}, "
                f"epoch={trial.user_attrs.get('goal_best_epoch')}, "
                f"train_fchem={fmt_float(trial.user_attrs.get('goal_train_fchem'))}, "
                f"val_vxc={fmt_float(trial.user_attrs.get('goal_val_vxc'))}, "
                f"val_fchem={fmt_float(trial.user_attrs.get('goal_val_fchem'))}, "
                f"phase={trial.user_attrs.get('goal_phase_name')}, "
                f"family={trial.user_attrs.get('search_family')}, params={trial.params}"
            )
    else:
        print("\nNo exact goal hits yet.")

    print("\nBest near-miss trials:")
    for trial in best_trials:
        print(
            f"  Trial {trial.number}: objective={float(trial.value):.8f}, "
            f"box={fmt_float(trial.user_attrs.get('goal_box_score'))}, "
            f"epoch={trial.user_attrs.get('goal_best_epoch')}, "
            f"train_fchem={fmt_float(trial.user_attrs.get('goal_train_fchem'))}, "
            f"val_vxc={fmt_float(trial.user_attrs.get('goal_val_vxc'))}, "
            f"val_fchem={fmt_float(trial.user_attrs.get('goal_val_fchem'))}, "
            f"phase={trial.user_attrs.get('goal_phase_name')}, "
            f"family={trial.user_attrs.get('search_family')}, params={trial.params}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, world_size, device, rank0 = init_distributed()
    set_random_seed(args.seed)

    with (Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle").open("rb") as handle:
        dispersions = pickle.load(handle)

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

    def goal_epoch_selector(epoch_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return best_epoch_by_goal_box(
            epoch_history=epoch_history,
            train_fchem_target=args.train_fchem_target,
            val_vxc_target=args.val_vxc_target,
            val_fchem_soft_cap=args.val_fchem_soft_cap,
        )

    def goal_checkpoint_key(row: Dict[str, Any]) -> tuple:
        return goal_epoch_key(
            epoch_row=row,
            train_fchem_target=args.train_fchem_target,
            val_vxc_target=args.val_vxc_target,
            val_fchem_soft_cap=args.val_fchem_soft_cap,
        )

    if rank0:
        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=args.sampler_startup_trials,
            multivariate=True,
        )
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
        )
        study.set_user_attr("selection_mode", "same_epoch_train_fchem_val_vxc_box")
        study.set_user_attr(
            "goal_targets",
            {"train_fchem": args.train_fchem_target, "val_vxc": args.val_vxc_target},
        )
        study.set_user_attr("val_fchem_soft_cap", args.val_fchem_soft_cap)
        study.set_user_attr(
            "objective_formula",
            "goal_box + 1e-3 * goal_l1 + 1e-5 * max(val_fchem / soft_cap, 1.0)",
        )
        study.set_user_attr(
            "search_space_summary",
            {
                "clip_static_refine": {
                    "gradient_merge_strategy": ["clip_then_sum"],
                    "accum_iter": [1, 2, 3],
                    "vxc_loss_scale": [50, 75, 100],
                    "reaction_grad_clip": ["none"],
                    "reaction_grad_scale": [0.6, 0.7, 0.8],
                    "vxc_grad_clip": [2.0, 3.0, 5.0],
                    "lr_train": [2.2e-4, 3.8e-4],
                },
                "clip_to_sum_repair": {
                    "phase_1": {
                        "gradient_merge_strategy": ["clip_then_sum"],
                        "accum_iter": [1, 2, 3],
                        "vxc_loss_scale": [50, 75, 100],
                        "reaction_grad_clip": ["none"],
                        "reaction_grad_scale": [0.6, 0.7],
                        "vxc_grad_clip": [3.0, 5.0],
                    },
                    "phase_2": {
                        "gradient_merge_strategy": ["sum"],
                        "accum_iter": [2, 3],
                        "vxc_loss_scale": [40, 50, 60],
                    },
                    "switch_epoch": list(schedule_switch_bounds(args.n_train)),
                    "lr_train": [2.3e-4, 3.6e-4],
                },
            },
        )
        if args.enqueue_recommended:
            for params in recommended_trials(args.n_train):
                study.enqueue_trial(params)
    else:
        study = None

    try:
        for _ in range(args.n_trials):
            if rank0:
                trial = study.ask()
                params = suggest_goal_bridge_params(trial, n_train=args.n_train)
                payload = {
                    "command": RUN_TRIAL,
                    "trial_number": trial.number,
                    "params": {key: value for key, value in params.items() if key != "search_family"},
                    "shared_preopt_checkpoint": str(shared_preopt_checkpoint),
                    "search_family": params["search_family"],
                }
                print(f"Starting trial {trial.number} with params: {json.dumps(params, sort_keys=True)}")
            else:
                trial = None
                payload = None

            payload = broadcast_payload(payload)
            if payload["command"] != RUN_TRIAL:
                break
            if payload["command"] in (STOP_TRIALS, FAIL_TRIAL):
                break

            trial_result = run_trial(
                trial_number=int(payload["trial_number"]),
                params=payload["params"],
                args=args,
                shared_preopt_checkpoint=Path(payload["shared_preopt_checkpoint"]),
                data_train=data_train,
                data_val=data_val,
                data_vxc_train=data_vxc_train,
                data_vxc_val=data_vxc_val,
                device=device,
                local_rank=local_rank,
                world_size=world_size,
                dispersions=dispersions,
                output_dir=output_dir,
                rank0=rank0,
                epoch_selector=goal_epoch_selector,
                checkpoint_row_key=goal_checkpoint_key,
            )

            if rank0:
                if trial_result["failed"]:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    print(f"Trial {trial.number} failed.")
                else:
                    goal_best = goal_epoch_selector(trial_result["epoch_history"])
                    goal_summary = goal_epoch_components(
                        epoch_row=goal_best,
                        train_fchem_target=args.train_fchem_target,
                        val_vxc_target=args.val_vxc_target,
                        val_fchem_soft_cap=args.val_fchem_soft_cap,
                    )

                    trial.set_user_attr("search_family", payload["search_family"])
                    trial.set_user_attr("selected_epoch", trial_result["selected_epoch"])
                    trial.set_user_attr("selected_joint_score", trial_result["selected_joint_score"])
                    trial.set_user_attr("selected_train_fchem", trial_result["selected_train_fchem"])
                    trial.set_user_attr("selected_val_fchem", trial_result["selected_val_fchem"])
                    trial.set_user_attr("selected_val_vxc", trial_result["selected_val_vxc"])
                    trial.set_user_attr("min_val_fchem_any_epoch", trial_result["min_val_fchem_any_epoch"])
                    trial.set_user_attr("min_val_vxc_any_epoch", trial_result["min_val_vxc_any_epoch"])
                    trial.set_user_attr("best_val_full_loss_any_epoch", trial_result["best_val_full_loss_any_epoch"])
                    trial.set_user_attr("goal_best_epoch", int(goal_best["epoch"]))
                    trial.set_user_attr("goal_phase_name", goal_best.get("phase_name"))
                    trial.set_user_attr("goal_train_fchem", float(goal_best["train_fchem"]))
                    trial.set_user_attr("goal_val_fchem", float(goal_best["val_fchem"]))
                    trial.set_user_attr("goal_val_vxc", float(goal_best["val_vxc"]))
                    for key, value in goal_summary.items():
                        trial.set_user_attr(key, value)
                    trial.set_user_attr("shared_preopt_checkpoint_path", str(shared_preopt_checkpoint))
                    if trial_result.get("selected_checkpoint_path") is not None:
                        trial.set_user_attr("selected_checkpoint_path", trial_result["selected_checkpoint_path"])
                    if trial_result.get("history_path") is not None:
                        trial.set_user_attr("history_path", trial_result["history_path"])

                    study.tell(trial, float(goal_summary["goal_objective"]))
                    print(
                        f"Completed trial {trial.number}: goal_epoch={goal_best['epoch']} "
                        f"goal_train_fchem={float(goal_best['train_fchem']):.8f} "
                        f"goal_val_vxc={float(goal_best['val_vxc']):.8f} "
                        f"goal_val_fchem={float(goal_best['val_fchem']):.8f} "
                        f"goal_box_score={goal_summary['goal_box_score']:.8f} "
                        f"goal_objective={goal_summary['goal_objective']:.8f}"
                    )
            dist.barrier()

        if rank0:
            summarize_goal_study(
                study=study,
                train_fchem_target=args.train_fchem_target,
                val_vxc_target=args.val_vxc_target,
            )
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
