import argparse
import collections
import copy
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from optuna_joint import (
    DEFAULT_MRKS_DISPERSIONS,
    FAIL_VALUE,
    OMEGA,
    add_gradient_list_to_parameters,
    allreduce_parameter_grads,
    batch_fchem,
    build_dataloaders,
    build_model,
    build_scheduler,
    compute_exc_from_errors,
    compute_fchem_from_errors,
    exc_loss,
    gather_object,
    get_trainable_parameters,
    grads_are_finite,
    init_distributed,
    load_chk,
    load_mrks_dispersions,
    load_state_dict_into_model,
    prepare_objective_gradients,
    resolve_objective_params,
    run_or_reuse_preoptimization,
    save_trial_history,
    scale_gradient_list,
    sync_failure,
    update_db_errors,
    update_exc_errors,
    validate_one_epoch,
    vxc_loss,
)
from reaction_energy_calculation import calculate_reaction_energy
from replay_trial_19_bridge import TRIAL_19_PARAMS, select_last_epoch, last_epoch_checkpoint_key
from utils import configure_optimizers, set_random_seed


ARML_TASKS = ("vxc", "exc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Trial 19 with ARML auxiliary-task reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_32")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gc_svelu_mirror",
        choices=["base", "log", "gc_svelu_mirror", "gc_softplus_mirror", "gc_softplus_mirror_r2scan_alpha"],
    )
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
    parser.add_argument("--arml-alpha-lr", type=float, default=5e-3)
    parser.add_argument("--arml-temperature", type=float, default=1.0)
    parser.add_argument("--arml-min-multiplier", type=float, default=0.0)
    parser.add_argument("--arml-max-multiplier", type=float, default=0.0, help="Disabled when <= 0.")
    parser.add_argument(
        "--no-reaction-dispersion",
        action="store_true",
        help="Do not add precomputed D3 dispersion corrections in reaction-energy training/validation.",
    )
    return parser.parse_args()


def _dot_grad_lists(
    left: List[Optional[torch.Tensor]],
    right: List[Optional[torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float64)
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        total = total + torch.sum(lhs.detach() * rhs.detach())
    return total


def _reaction_loss_from_batch(
    model: DDP,
    reaction_batch: Dict[str, Any],
    y_batch: torch.Tensor,
    device: torch.device,
    dispersions: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid = reaction_batch["Grid"].to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
    predictions = model(grid)
    reaction_energy, _ = calculate_reaction_energy(
        reaction_batch,
        predictions,
        device,
        rung="GGA",
        dft="PBE",
        dispersions=dispersions,
        return_local_energies=False,
    )
    reaction_loss = batch_fchem(list(reaction_batch["Database"]), reaction_energy, y_batch)
    return reaction_loss, reaction_energy


class ARMLController:
    """Two-task ARML controller for VXC and E_xc auxiliary likelihood weights.

    The paper's auxiliary likelihood weights are represented as learnable logits.
    We use the one-step hypergradient of held-out reaction loss:

        d L_val(theta - eta * sum_i w_i grad L_i) / d w_i
        = -eta <grad L_val, grad L_i>

    This keeps the implementation exact for the first-order one-step ARML update and
    avoids changing the model architecture or introducing nonlocal information.
    """

    def __init__(
        self,
        lr: float,
        temperature: float,
        min_multiplier: float,
        max_multiplier: float,
        device: torch.device,
    ) -> None:
        self.logits = torch.zeros(len(ARML_TASKS), device=device, dtype=torch.float64, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.logits], lr=lr)
        self.temperature = float(temperature)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)

    def multipliers(self) -> torch.Tensor:
        weights = torch.softmax(self.logits / max(self.temperature, 1e-8), dim=0) * len(ARML_TASKS)
        if self.min_multiplier > 0.0:
            weights = torch.clamp(weights, min=self.min_multiplier)
        if self.max_multiplier > 0.0:
            weights = torch.clamp(weights, max=self.max_multiplier)
        return weights

    def update(self, alignment_scores: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        multipliers = self.multipliers()
        meta_loss = -torch.sum(multipliers * alignment_scores.detach())
        meta_loss.backward()
        self.optimizer.step()
        return self.multipliers().detach()


def _next_from_cycling_iterator(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch_arml(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    train_loader,
    val_loader,
    vxc_train_loader,
    params: Dict[str, Any],
    arml: ARMLController,
    device: torch.device,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    include_mrks_dispersion: bool,
    world_size: int,
) -> Tuple[Dict[str, float], Dict[str, List[float]], bool]:
    params = resolve_objective_params(params)
    model.train()
    trainable_parameters = get_trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)

    train_db_errors: Dict[str, List[float]] = collections.defaultdict(list)
    train_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    n_train = len(train_loader)
    n_vxc = len(vxc_train_loader)
    n_val = len(val_loader)
    if n_train == 0 or n_vxc == 0 or n_val == 0:
        raise ValueError("Train, validation, and VXC loaders must be non-empty for ARML.")

    n_steps = max(n_train, n_vxc)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    vxc_iter = iter(vxc_train_loader)

    loss_sum = 0.0
    reaction_loss_sum = 0.0
    vxc_loss_sum = 0.0
    exc_loss_sum = 0.0
    mae_sum = 0.0
    optimizer_steps = 0
    vxc_multiplier_sum = 0.0
    exc_multiplier_sum = 0.0
    vxc_alignment_sum = 0.0
    exc_alignment_sum = 0.0
    failed = False

    for batch_idx in range(n_steps):
        (reaction_batch, y_batch), train_iter = _next_from_cycling_iterator(train_loader, train_iter)
        (val_batch, val_y_batch), val_iter = _next_from_cycling_iterator(val_loader, val_iter)
        X_vxc, vxc_iter = _next_from_cycling_iterator(vxc_train_loader, vxc_iter)

        current_bases = list(reaction_batch["Database"])
        y_batch = y_batch.to(device, non_blocking=True)
        do_step = ((batch_idx + 1) % params["accum_iter"] == 0) or ((batch_idx + 1) == n_steps)

        reaction_loss, reaction_energy = _reaction_loss_from_batch(
            model,
            reaction_batch,
            y_batch,
            device,
            dispersions,
        )
        weighted_reaction_loss = reaction_loss / params["accum_iter"]
        microbatch_failed = (
            not torch.isfinite(reaction_energy).all()
            or not torch.isfinite(weighted_reaction_loss)
        )
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break

        reaction_grads = prepare_objective_gradients(
            trainable_parameters,
            weighted_reaction_loss,
            params["reaction_gradient_merge_strategy"],
            params["reaction_grad_clip"],
            params["reaction_grad_scale"],
        )

        vxc_term = vxc_loss(model, X_vxc, device, rung="GGA", dft="PBE", create_graph=True)
        base_weighted_vxc_loss = OMEGA * params["vxc_loss_scale"] * vxc_term / params["accum_iter"]
        microbatch_failed = not torch.isfinite(base_weighted_vxc_loss)
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        vxc_grads = prepare_objective_gradients(
            trainable_parameters,
            base_weighted_vxc_loss,
            params["vxc_gradient_merge_strategy"],
            params["vxc_grad_clip"],
            1.0,
        )

        exc_term, pred_exc, ref_exc = exc_loss(
            model,
            X_vxc,
            device,
            rung="GGA",
            dft="PBE",
            dispersions=mrks_dispersions,
            include_mrks_dispersion=include_mrks_dispersion,
        )
        base_weighted_exc_loss = params["exc_loss_scale"] * exc_term / params["accum_iter"]
        microbatch_failed = (
            not torch.isfinite(pred_exc).all()
            or not torch.isfinite(base_weighted_exc_loss)
        )
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        exc_grads = prepare_objective_gradients(
            trainable_parameters,
            base_weighted_exc_loss,
            params["exc_gradient_merge_strategy"],
            params["exc_grad_clip"],
            params["exc_grad_scale"],
        )

        val_reaction_loss, val_reaction_energy = _reaction_loss_from_batch(
            model,
            val_batch,
            val_y_batch.to(device, non_blocking=True),
            device,
            dispersions,
        )
        microbatch_failed = (
            not torch.isfinite(val_reaction_energy).all()
            or not torch.isfinite(val_reaction_loss)
        )
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        val_grads = prepare_objective_gradients(
            trainable_parameters,
            val_reaction_loss,
            "sum",
            "none",
            1.0,
        )

        alignment_scores = torch.stack(
            [
                _dot_grad_lists(val_grads, vxc_grads, device),
                _dot_grad_lists(val_grads, exc_grads, device),
            ]
        )
        dist.all_reduce(alignment_scores, op=dist.ReduceOp.SUM)
        alignment_scores.div_(world_size)
        multipliers = arml.update(alignment_scores)

        add_gradient_list_to_parameters(trainable_parameters, reaction_grads)
        add_gradient_list_to_parameters(
            trainable_parameters,
            scale_gradient_list(vxc_grads, float(multipliers[0].item())),
        )
        add_gradient_list_to_parameters(
            trainable_parameters,
            scale_gradient_list(exc_grads, float(multipliers[1].item())),
        )

        update_db_errors(train_db_errors, current_bases, reaction_energy, y_batch)
        update_exc_errors(train_exc_errors, list(X_vxc["Names"]), pred_exc, ref_exc)
        loss_sum += float(
            (
                reaction_loss
                + float(multipliers[0].item()) * OMEGA * params["vxc_loss_scale"] * vxc_term
                + float(multipliers[1].item()) * params["exc_loss_scale"] * exc_term
            ).item()
        )
        reaction_loss_sum += float(reaction_loss.item())
        vxc_loss_sum += float(vxc_term.item())
        exc_loss_sum += float(exc_term.item())
        mae_sum += float(nn.functional.l1_loss(reaction_energy, y_batch).item())
        vxc_multiplier_sum += float(multipliers[0].item())
        exc_multiplier_sum += float(multipliers[1].item())
        vxc_alignment_sum += float(alignment_scores[0].item())
        exc_alignment_sum += float(alignment_scores[1].item())

        del reaction_loss, weighted_reaction_loss, reaction_energy
        del vxc_term, base_weighted_vxc_loss
        del exc_term, base_weighted_exc_loss, pred_exc, ref_exc
        del val_reaction_loss, val_reaction_energy
        del reaction_grads, vxc_grads, exc_grads, val_grads

        if not do_step:
            continue

        allreduce_parameter_grads(trainable_parameters, world_size)
        step_failed = not grads_are_finite(trainable_parameters)
        step_failed = sync_failure(step_failed, device)
        if step_failed:
            failed = True
            break

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    if failed:
        optimizer.zero_grad(set_to_none=True)
        return {}, {}, True

    scalar_tensor = torch.tensor(
        [
            loss_sum,
            reaction_loss_sum,
            vxc_loss_sum,
            exc_loss_sum,
            mae_sum,
            float(n_steps),
            float(optimizer_steps),
            vxc_multiplier_sum,
            exc_multiplier_sum,
            vxc_alignment_sum,
            exc_alignment_sum,
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

    gathered_errors = gather_object(dict(train_db_errors), world_size)
    global_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_errors:
        for db, errs in local_dict.items():
            global_errors[db].extend(errs)
    train_fchem, train_per_db = compute_fchem_from_errors(global_errors)

    gathered_exc_errors = gather_object(dict(train_exc_errors), world_size)
    global_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_exc_errors:
        for system_name, errs in local_dict.items():
            global_exc_errors[system_name].extend(errs)
    train_exc, train_per_system_exc = compute_exc_from_errors(global_exc_errors)

    denom = max(scalar_tensor[5].item(), 1.0)
    metrics = {
        "train_full_loss": float(scalar_tensor[0].item() / denom),
        "train_reaction_loss": float(scalar_tensor[1].item() / denom),
        "train_vxc": float(scalar_tensor[2].item() / denom),
        "train_exc_loss": float(scalar_tensor[3].item() / denom),
        "train_mae": float(scalar_tensor[4].item() / denom),
        "train_fchem": train_fchem,
        "train_exc": train_exc,
        "train_per_system_exc_rmse": train_per_system_exc,
        "optimizer_steps": int(scalar_tensor[6].item()),
        "arml_vxc_multiplier": float(scalar_tensor[7].item() / denom),
        "arml_exc_multiplier": float(scalar_tensor[8].item() / denom),
        "arml_vxc_alignment": float(scalar_tensor[9].item() / denom),
        "arml_exc_alignment": float(scalar_tensor[10].item() / denom),
    }
    return metrics, dict(train_per_db), False


def run_trial_arml(
    trial_number: int,
    params: Dict[str, Any],
    args: argparse.Namespace,
    shared_preopt_checkpoint: Path,
    data_train: dict,
    data_val: dict,
    data_vxc_train: list,
    data_vxc_val: list,
    device: torch.device,
    local_rank: int,
    world_size: int,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    output_dir: Path,
    rank0: bool,
) -> Dict[str, Any]:
    trial_seed = args.seed + trial_number
    set_random_seed(trial_seed)
    loaders = build_dataloaders(
        data_train=data_train,
        data_val=data_val,
        data_vxc_train=data_vxc_train,
        data_vxc_val=data_vxc_val,
        trial_seed=trial_seed,
        args=args,
        rank=local_rank,
        world_size=world_size,
    )

    model = build_model(args, device)
    load_state_dict_into_model(model, shared_preopt_checkpoint, device)
    model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    )
    optimizer = configure_optimizers(
        model=model,
        learning_rate=params["lr_train"],
        optimizer_str="radamw",
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.n_train)
    arml = ARMLController(
        lr=args.arml_alpha_lr,
        temperature=args.arml_temperature,
        min_multiplier=args.arml_min_multiplier,
        max_multiplier=args.arml_max_multiplier,
        device=device,
    )

    epoch_history: List[Dict[str, Any]] = []
    selected_checkpoint_path = None
    current_selected_key = None
    failed = False
    min_val_fchem_any_epoch = math.inf
    min_val_vxc_any_epoch = math.inf
    min_val_exc_any_epoch = math.inf
    best_val_full_loss_any_epoch = math.inf

    if args.save_selected_checkpoints and rank0:
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        selected_checkpoint_path = checkpoints_dir / f"trial_{trial_number}_selected.pt"

    for epoch in range(args.n_train):
        epoch_number = epoch + 1
        effective_params = resolve_objective_params(
            resolve_epoch_params(params, epoch_number=epoch_number, n_train=args.n_train)
        )
        train_dataset = loaders["train_loader"].dataset
        if hasattr(train_dataset, "resample"):
            train_dataset.resample(epoch)
        for sampler_name in ("train_sampler", "vxc_train_sampler", "val_sampler"):
            sampler = loaders.get(sampler_name)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        train_metrics, train_per_db, train_failed = train_one_epoch_arml(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train_loader"],
            val_loader=loaders["val_loader"],
            vxc_train_loader=loaders["vxc_train_loader"],
            params=effective_params,
            arml=arml,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(getattr(args, "include_mrks_dispersion", False)),
            world_size=world_size,
        )
        if train_failed:
            failed = True
            break

        val_metrics, val_per_db, val_failed = validate_one_epoch(
            model=model,
            val_loader=loaders["val_loader"],
            vxc_val_loader=loaders["vxc_val_loader"],
            params=effective_params,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(getattr(args, "include_mrks_dispersion", False)),
            world_size=world_size,
        )
        if val_failed:
            failed = True
            break
        scheduler.step()

        row = {
            "epoch": epoch_number,
            "train_fchem": train_metrics["train_fchem"],
            "train_vxc": train_metrics["train_vxc"],
            "train_exc": train_metrics["train_exc"],
            "train_exc_loss": train_metrics["train_exc_loss"],
            "train_full_loss": train_metrics["train_full_loss"],
            "train_reaction_loss": train_metrics["train_reaction_loss"],
            "train_mae": train_metrics["train_mae"],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "val_fchem": val_metrics["val_fchem"],
            "val_vxc": val_metrics["val_vxc"],
            "val_exc": val_metrics["val_exc"],
            "val_exc_loss": val_metrics["val_exc_loss"],
            "val_full_loss": val_metrics["val_full_loss"],
            "val_reaction_loss": val_metrics["val_reaction_loss"],
            "val_mae": val_metrics["val_mae"],
            "val_joint_score": val_metrics["val_joint_score"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_per_database_rmse": val_per_db,
            "train_per_database_rmse": train_per_db,
            "val_per_system_exc_rmse": val_metrics["val_per_system_exc_rmse"],
            "train_per_system_exc_rmse": train_metrics["train_per_system_exc_rmse"],
            "phase_name": effective_params.get("phase_name", "static"),
            "phase_start_epoch": int(effective_params.get("phase_start_epoch", 1)),
            "phase_end_epoch": int(effective_params.get("phase_end_epoch", args.n_train)),
            "effective_accum_iter": int(effective_params["accum_iter"]),
            "effective_vxc_loss_scale": float(effective_params["vxc_loss_scale"]),
            "effective_exc_loss_scale": float(effective_params["exc_loss_scale"]),
            "arml_vxc_multiplier": train_metrics["arml_vxc_multiplier"],
            "arml_exc_multiplier": train_metrics["arml_exc_multiplier"],
            "arml_vxc_alignment": train_metrics["arml_vxc_alignment"],
            "arml_exc_alignment": train_metrics["arml_exc_alignment"],
        }
        epoch_history.append(row)
        min_val_fchem_any_epoch = min(min_val_fchem_any_epoch, row["val_fchem"])
        min_val_vxc_any_epoch = min(min_val_vxc_any_epoch, row["val_vxc"])
        min_val_exc_any_epoch = min(min_val_exc_any_epoch, row["val_exc"])
        best_val_full_loss_any_epoch = min(best_val_full_loss_any_epoch, row["val_full_loss"])

        candidate_key = last_epoch_checkpoint_key(row)
        if current_selected_key is None or candidate_key < current_selected_key:
            current_selected_key = candidate_key
            if args.save_selected_checkpoints and rank0 and selected_checkpoint_path is not None:
                torch.save(model.module.state_dict(), selected_checkpoint_path)

        if rank0:
            print(
                f"ARML Trial {trial_number} epoch {epoch_number}/{args.n_train}: "
                f"train_fchem={row['train_fchem']:.8f} "
                f"val_fchem={row['val_fchem']:.8f} "
                f"val_vxc={row['val_vxc']:.8f} "
                f"val_exc={row['val_exc']:.8f} "
                f"joint_score={row['val_joint_score']:.8f} "
                f"m_vxc={row['arml_vxc_multiplier']:.4f} "
                f"m_exc={row['arml_exc_multiplier']:.4f}"
            )

    failed = sync_failure(failed, device)
    if failed:
        return {"failed": True, "trial_number": trial_number, "params": params}

    selected = select_last_epoch(epoch_history)
    trial_payload = {
        "failed": False,
        "trial_number": trial_number,
        "params": params,
        "arml": {
            "tasks": list(ARML_TASKS),
            "alpha_lr": args.arml_alpha_lr,
            "temperature": args.arml_temperature,
            "min_multiplier": args.arml_min_multiplier,
            "max_multiplier": args.arml_max_multiplier,
        },
        "selected_epoch": int(selected["epoch"]),
        "selected_joint_score": float(selected["val_joint_score"]),
        "selected_train_fchem": float(selected["train_fchem"]),
        "selected_val_fchem": float(selected["val_fchem"]),
        "selected_val_vxc": float(selected["val_vxc"]),
        "selected_val_exc": float(selected["val_exc"]),
        "selected_phase_name": selected.get("phase_name"),
        "min_val_fchem_any_epoch": float(min_val_fchem_any_epoch),
        "min_val_vxc_any_epoch": float(min_val_vxc_any_epoch),
        "min_val_exc_any_epoch": float(min_val_exc_any_epoch),
        "best_val_full_loss_any_epoch": float(best_val_full_loss_any_epoch),
        "epoch_history": epoch_history,
        "selected_checkpoint_path": str(selected_checkpoint_path) if selected_checkpoint_path is not None else None,
    }
    if rank0:
        history_path = save_trial_history(output_dir, trial_number, trial_payload)
        trial_payload["history_path"] = str(history_path)
    else:
        trial_payload["history_path"] = None
    return trial_payload


def resolve_epoch_params(params: Dict[str, Any], epoch_number: int, n_train: int) -> Dict[str, Any]:
    schedule = params.get("epoch_schedule")
    if not schedule:
        resolved = dict(params)
        resolved.setdefault("phase_name", "static")
        resolved.setdefault("phase_start_epoch", 1)
        resolved.setdefault("phase_end_epoch", n_train)
        return resolved
    for index, phase in enumerate(schedule):
        start_epoch = int(phase.get("start_epoch", 1))
        end_epoch = int(phase.get("end_epoch", n_train))
        if start_epoch <= epoch_number <= end_epoch:
            resolved = {key: value for key, value in params.items() if key != "epoch_schedule"}
            resolved.update(dict(phase.get("params", {})))
            resolved["phase_name"] = str(phase.get("name", f"phase_{index + 1}"))
            resolved["phase_start_epoch"] = start_epoch
            resolved["phase_end_epoch"] = end_epoch
            return resolved
    raise ValueError(f"No scheduled phase covers epoch {epoch_number}.")


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

    params = copy.deepcopy(TRIAL_19_PARAMS)
    result = run_trial_arml(
        trial_number=args.trial_number,
        params=params,
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
    )

    if rank0:
        if result.get("failed"):
            print("ARML replay failed.")
            return
        final_epoch = select_last_epoch(result["epoch_history"])
        print("ARML replay complete for Trial 19 bridge schedule.")
        print(f"Final selected epoch: {final_epoch['epoch']}")
        print(
            "Final metrics: "
            f"train_fchem={float(final_epoch['train_fchem']):.8f}, "
            f"val_vxc={float(final_epoch['val_vxc']):.8f}, "
            f"val_exc={float(final_epoch['val_exc']):.8f}, "
            f"val_fchem={float(final_epoch['val_fchem']):.8f}, "
            f"m_vxc={float(final_epoch['arml_vxc_multiplier']):.4f}, "
            f"m_exc={float(final_epoch['arml_exc_multiplier']):.4f}, "
            f"phase={final_epoch.get('phase_name')}"
        )
        print(f"Selected checkpoint: {result.get('selected_checkpoint_path')}")
        print(f"History path: {result.get('history_path')}")
        print(f"Params: {json.dumps(params, sort_keys=True)}")


if __name__ == "__main__":
    main()
