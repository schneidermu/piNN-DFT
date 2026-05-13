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
)
from replay_trial_19_bridge import (
    last_epoch_checkpoint_key,
    parse_args,
    select_last_epoch,
)


LOG_TUNED_PARAMS = {
    # The log model is a representational superset of the base model, but the
    # appended raw log channels are poorly conditioned relative to the bounded
    # descriptors. Use a lower step size and lower decay so the optimizer can
    # learn to gate those channels instead of immediately damping or overshooting
    # them.
    "accum_iter": 4,
    "gradient_merge_strategy": "clip_then_sum",
    "reaction_gradient_merge_strategy": "clip_then_sum",
    "vxc_gradient_merge_strategy": "clip_then_sum",
    "lr_train": 1.35e-4,
    "reaction_grad_clip": 200.0,
    "reaction_grad_scale": 0.35,
    "vxc_grad_clip": 1.0,
    "vxc_loss_scale": 150,
    "exc_loss_scale": 0.5,
    "exc_grad_clip": 2.0,
    "exc_grad_scale": 0.5,
    "exc_gradient_merge_strategy": "clip_then_sum",
    "epoch_schedule": [
        {
            "name": "log_vxc_anchor",
            "start_epoch": 1,
            "end_epoch": 120,
            "params": {
                "accum_iter": 4,
                "reaction_gradient_merge_strategy": "clip_then_sum",
                "vxc_gradient_merge_strategy": "clip_then_sum",
                "exc_gradient_merge_strategy": "clip_then_sum",
                "reaction_grad_clip": 200.0,
                "reaction_grad_scale": 0.35,
                "vxc_grad_clip": 1.0,
                "vxc_loss_scale": 150,
                "exc_loss_scale": 0.5,
                "exc_grad_clip": 2.0,
                "exc_grad_scale": 0.5,
            },
        },
        {
            "name": "log_joint_repair",
            "start_epoch": 121,
            "end_epoch": 260,
            "params": {
                "accum_iter": 3,
                "reaction_gradient_merge_strategy": "clip_then_sum",
                "vxc_gradient_merge_strategy": "clip_then_sum",
                "exc_gradient_merge_strategy": "clip_then_sum",
                "reaction_grad_clip": 300.0,
                "reaction_grad_scale": 0.70,
                "vxc_grad_clip": 1.5,
                "vxc_loss_scale": 100,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": 2.0,
                "exc_grad_scale": 0.7,
            },
        },
        {
            "name": "log_fchem_entry",
            "start_epoch": 261,
            "end_epoch": 430,
            "params": {
                "accum_iter": 3,
                "reaction_gradient_merge_strategy": "sum",
                "vxc_gradient_merge_strategy": "clip_then_sum",
                "exc_gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 50,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
            },
        },
        {
            "name": "log_fchem_finish",
            "start_epoch": 431,
            "end_epoch": 600,
            "params": {
                "accum_iter": 2,
                "reaction_gradient_merge_strategy": "sum",
                "vxc_gradient_merge_strategy": "clip_then_sum",
                "exc_gradient_merge_strategy": "sum",
                "reaction_grad_clip": "none",
                "reaction_grad_scale": 1.0,
                "vxc_grad_clip": 2.0,
                "vxc_loss_scale": 20,
                "exc_loss_scale": 1.0,
                "exc_grad_clip": "none",
                "exc_grad_scale": 1.0,
            },
        },
    ],
}


def main() -> None:
    args = parse_args()
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
        params=LOG_TUNED_PARAMS,
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
        print("Replay complete for Trial 19 tuned log schedule.")
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
        print(f"Params: {json.dumps(LOG_TUNED_PARAMS, sort_keys=True)}")


if __name__ == "__main__":
    main()
