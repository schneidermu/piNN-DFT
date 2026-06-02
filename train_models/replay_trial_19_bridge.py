import argparse
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
        params=TRIAL_19_PARAMS,
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
        print(f"Params: {json.dumps(TRIAL_19_PARAMS, sort_keys=True)}")


if __name__ == "__main__":
    main()
