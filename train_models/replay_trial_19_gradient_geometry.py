"""One-stage Trial 19 replay with parameter-free gradient aggregation.

The three objectives are reaction thermochemistry, exact E_xc, and local V_xc.
Unlike the historical replay, this driver has no epoch schedule and no objective
loss weights. The aggregation rule operates directly on per-objective gradients.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from optuna_joint import (
    DEFAULT_MRKS_DISPERSIONS,
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
    run_or_reuse_preoptimization,
    save_trial_history,
    sync_failure,
    update_db_errors,
    update_exc_errors,
    validate_one_epoch,
    vxc_loss,
)
from reaction_energy_calculation import calculate_reaction_energy
from replay_trial_19_bridge import (
    TRIAL_19_PARAMS,
    last_epoch_checkpoint_key,
    select_last_epoch,
)
from utils import configure_optimizers, set_random_seed


GradientList = List[torch.Tensor]
TASKS = ("reaction", "vxc", "exc")
AGGREGATION_MODES = (
    "primary_pcgrad",
    "primary_pcgrad_auxmean",
    "primary_orthogonal",
    "aligned_aux",
    "drop_conflicting_aux",
    "normalized_sum",
    "symmetric_pcgrad",
    "mgda",
    "raw_primary_pcgrad",
    "log_primary_pcgrad",
)
EPS = 1e-12

# Static bookkeeping values are also used by validation. They are not training
# weights: training gradients are built from the three unweighted raw losses.
ONE_STAGE_PARAMS = {
    "accum_iter": 2,
    "lr_train": TRIAL_19_PARAMS["lr_train"],
    "gradient_merge_strategy": "sum",
    "reaction_grad_clip": "none",
    "reaction_grad_scale": 1.0,
    "vxc_grad_clip": "none",
    "vxc_loss_scale": 1.0,
    "exc_loss_scale": 1.0,
    "exc_grad_clip": "none",
    "exc_grad_scale": 1.0,
    "exc_gradient_merge_strategy": "sum",
    "phase_name": "one_stage_gradient_geometry",
    "phase_start_epoch": 1,
    "phase_end_epoch": 500,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-stage Trial 19 replay with parameter-free gradient geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--aggregation", choices=AGGREGATION_MODES, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_32")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gc_svelu_mirror",
        choices=[
            "base",
            "log",
            "gc_svelu_mirror",
            "gc_softplus_mirror",
            "gc_softplus_mirror_r2scan_alpha",
        ],
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
    parser.add_argument(
        "--save-selected-checkpoints", action="store_true", default=True
    )
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument(
        "--mrks-dispersions-pickle", type=str, default=str(DEFAULT_MRKS_DISPERSIONS)
    )
    parser.add_argument(
        "--no-reaction-dispersion",
        action="store_true",
        help="Do not add precomputed D3 corrections during reaction training/validation.",
    )
    return parser.parse_args()


def _next_from_cycling_iterator(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


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
    reaction_loss = batch_fchem(
        list(reaction_batch["Database"]), reaction_energy, y_batch
    )
    return reaction_loss, reaction_energy


def _compute_global_gradient_list(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
) -> GradientList:
    local = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    global_grads: GradientList = []
    for parameter, grad in zip(parameters, local):
        value = torch.zeros_like(parameter) if grad is None else grad.detach().clone()
        if world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(world_size)
        global_grads.append(value)
    return global_grads


def _dot(left: GradientList, right: GradientList) -> torch.Tensor:
    return sum(
        (torch.sum(lhs * rhs) for lhs, rhs in zip(left, right)),
        start=torch.zeros((), device=left[0].device, dtype=left[0].dtype),
    )


def _norm(grads: GradientList) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_dot(grads, grads), min=0.0))


def _scale(grads: GradientList, factor: torch.Tensor | float) -> GradientList:
    return [grad * factor for grad in grads]


def _add(*gradient_lists: GradientList) -> GradientList:
    return [sum(values) for values in zip(*gradient_lists)]


def _subtract(left: GradientList, right: GradientList) -> GradientList:
    return [lhs - rhs for lhs, rhs in zip(left, right)]


def _normalize(grads: GradientList) -> GradientList:
    return _scale(grads, 1.0 / torch.clamp(_norm(grads), min=EPS))


def _project_conflict(candidate: GradientList, anchor: GradientList) -> GradientList:
    coefficient = torch.minimum(
        torch.zeros((), device=anchor[0].device, dtype=anchor[0].dtype),
        _dot(candidate, anchor) / torch.clamp(_dot(anchor, anchor), min=EPS),
    )
    return _subtract(candidate, _scale(anchor, coefficient))


def _project_all_anchor_component(
    candidate: GradientList, anchor: GradientList
) -> GradientList:
    coefficient = _dot(candidate, anchor) / torch.clamp(_dot(anchor, anchor), min=EPS)
    return _subtract(candidate, _scale(anchor, coefficient))


def _mgda_weights(unit_grads: list[GradientList]) -> torch.Tensor:
    """Solve the three-task minimum-norm simplex problem by active-set enumeration."""
    device, dtype = unit_grads[0][0].device, unit_grads[0][0].dtype
    gram = torch.stack(
        [
            torch.stack([_dot(left, right) for right in unit_grads])
            for left in unit_grads
        ]
    )
    candidates: list[torch.Tensor] = []
    for index in range(3):
        value = torch.zeros(3, device=device, dtype=dtype)
        value[index] = 1.0
        candidates.append(value)
    for left in range(3):
        for right in range(left + 1, 3):
            denominator = (
                gram[left, left] + gram[right, right] - 2.0 * gram[left, right]
            )
            if float(torch.abs(denominator).item()) <= EPS:
                fraction = torch.tensor(0.5, device=device, dtype=dtype)
            else:
                fraction = torch.clamp(
                    (gram[right, right] - gram[left, right]) / denominator, 0.0, 1.0
                )
            value = torch.zeros(3, device=device, dtype=dtype)
            value[left] = fraction
            value[right] = 1.0 - fraction
            candidates.append(value)
    ones = torch.ones(3, device=device, dtype=dtype)
    inverse_ones = torch.linalg.pinv(gram) @ ones
    full = inverse_ones / torch.clamp(torch.sum(inverse_ones), min=EPS)
    if bool(torch.all(full >= -1e-8)):
        candidates.append(
            torch.clamp(full, min=0.0)
            / torch.clamp(torch.sum(torch.clamp(full, min=0.0)), min=EPS)
        )
    objectives = [weights @ gram @ weights for weights in candidates]
    return candidates[
        min(range(len(candidates)), key=lambda index: float(objectives[index].item()))
    ]


def aggregate_gradients(
    mode: str,
    raw_grads: list[GradientList],
    losses: Sequence[torch.Tensor],
) -> tuple[GradientList, dict[str, float]]:
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation mode: {mode}")
    raw_norms = [_norm(grads) for grads in raw_grads]
    unit = [_normalize(grads) for grads in raw_grads]
    cos_fr = _dot(unit[0], unit[1])
    cos_fe = _dot(unit[0], unit[2])
    cos_ve = _dot(unit[1], unit[2])
    nominal_weights = torch.ones(3, device=unit[0][0].device, dtype=unit[0][0].dtype)

    if mode == "primary_pcgrad":
        combined = _add(
            unit[0],
            _project_conflict(unit[1], unit[0]),
            _project_conflict(unit[2], unit[0]),
        )
    elif mode == "primary_pcgrad_auxmean":
        auxiliary_mean = _scale(_add(unit[1], unit[2]), 0.5)
        combined = _add(unit[0], _project_conflict(auxiliary_mean, unit[0]))
        nominal_weights = torch.tensor(
            [1.0, 0.5, 0.5], device=unit[0][0].device, dtype=unit[0][0].dtype
        )
    elif mode == "primary_orthogonal":
        combined = _add(
            unit[0],
            _project_all_anchor_component(unit[1], unit[0]),
            _project_all_anchor_component(unit[2], unit[0]),
        )
    elif mode == "aligned_aux":
        nominal_weights = torch.stack(
            [
                torch.ones_like(cos_fr),
                torch.clamp(cos_fr, min=0.0),
                torch.clamp(cos_fe, min=0.0),
            ]
        )
        combined = _add(
            *[_scale(grads, weight) for grads, weight in zip(unit, nominal_weights)]
        )
    elif mode == "drop_conflicting_aux":
        nominal_weights = torch.stack(
            [
                torch.ones_like(cos_fr),
                (cos_fr >= 0).to(cos_fr.dtype),
                (cos_fe >= 0).to(cos_fe.dtype),
            ]
        )
        combined = _add(
            *[_scale(grads, weight) for grads, weight in zip(unit, nominal_weights)]
        )
    elif mode == "normalized_sum":
        combined = _add(*unit)
    elif mode == "symmetric_pcgrad":
        projected: list[GradientList] = []
        for task_index in range(3):
            task = unit[task_index]
            for other_index in range(3):
                if other_index != task_index:
                    task = _project_conflict(task, unit[other_index])
            projected.append(task)
        combined = _add(*projected)
    elif mode == "mgda":
        nominal_weights = _mgda_weights(unit)
        combined = _add(
            *[_scale(grads, weight) for grads, weight in zip(unit, nominal_weights)]
        )
    elif mode == "raw_primary_pcgrad":
        combined = _add(
            raw_grads[0],
            _project_conflict(raw_grads[1], raw_grads[0]),
            _project_conflict(raw_grads[2], raw_grads[0]),
        )
    else:  # log_primary_pcgrad
        log_grads = [
            _scale(grads, 1.0 / torch.clamp(loss.detach(), min=EPS))
            for grads, loss in zip(raw_grads, losses)
        ]
        combined = _add(
            log_grads[0],
            _project_conflict(log_grads[1], log_grads[0]),
            _project_conflict(log_grads[2], log_grads[0]),
        )

    diagnostics = {
        "raw_norm_reaction": float(raw_norms[0].item()),
        "raw_norm_vxc": float(raw_norms[1].item()),
        "raw_norm_exc": float(raw_norms[2].item()),
        "cos_reaction_vxc": float(cos_fr.item()),
        "cos_reaction_exc": float(cos_fe.item()),
        "cos_vxc_exc": float(cos_ve.item()),
        "combined_norm": float(_norm(combined).item()),
        "weight_reaction": float(nominal_weights[0].item()),
        "weight_vxc": float(nominal_weights[1].item()),
        "weight_exc": float(nominal_weights[2].item()),
    }
    return combined, diagnostics


def _add_to_parameter_grads(
    parameters: Sequence[torch.nn.Parameter], grads: GradientList
) -> None:
    for parameter, grad in zip(parameters, grads):
        if parameter.grad is None:
            parameter.grad = grad.detach().clone()
        else:
            parameter.grad.add_(grad.detach())


def train_one_epoch_geometry(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    train_loader,
    vxc_train_loader,
    params: Dict[str, Any],
    aggregation: str,
    device: torch.device,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    include_mrks_dispersion: bool,
    world_size: int,
) -> Tuple[Dict[str, Any], Dict[str, List[float]], bool]:
    model.train()
    trainable_parameters = get_trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    train_db_errors: Dict[str, List[float]] = collections.defaultdict(list)
    train_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    n_steps = max(len(train_loader), len(vxc_train_loader))
    if n_steps == 0:
        raise ValueError("Training and Vxc loaders must be non-empty.")
    train_iter, vxc_iter = iter(train_loader), iter(vxc_train_loader)
    sums = collections.defaultdict(float)
    optimizer_steps = 0
    failed = False

    for batch_idx in range(n_steps):
        (reaction_batch, y_batch), train_iter = _next_from_cycling_iterator(
            train_loader, train_iter
        )
        X_vxc, vxc_iter = _next_from_cycling_iterator(vxc_train_loader, vxc_iter)
        y_batch = y_batch.to(device, non_blocking=True)
        do_step = ((batch_idx + 1) % params["accum_iter"] == 0) or (
            (batch_idx + 1) == n_steps
        )

        reaction_loss, reaction_energy = _reaction_loss_from_batch(
            model, reaction_batch, y_batch, device, dispersions
        )
        if sync_failure(
            not torch.isfinite(reaction_energy).all()
            or not torch.isfinite(reaction_loss),
            device,
        ):
            failed = True
            break
        reaction_grads = _compute_global_gradient_list(
            reaction_loss / params["accum_iter"], trainable_parameters, world_size
        )

        vxc_term = vxc_loss(
            model, X_vxc, device, rung="GGA", dft="PBE", create_graph=True
        )
        if sync_failure(not torch.isfinite(vxc_term), device):
            failed = True
            break
        vxc_grads = _compute_global_gradient_list(
            vxc_term / params["accum_iter"], trainable_parameters, world_size
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
        if sync_failure(
            not torch.isfinite(pred_exc).all() or not torch.isfinite(exc_term), device
        ):
            failed = True
            break
        exc_grads = _compute_global_gradient_list(
            exc_term / params["accum_iter"], trainable_parameters, world_size
        )

        combined, diagnostics = aggregate_gradients(
            aggregation,
            [reaction_grads, vxc_grads, exc_grads],
            [reaction_loss, vxc_term, exc_term],
        )
        _add_to_parameter_grads(trainable_parameters, combined)

        update_db_errors(
            train_db_errors, list(reaction_batch["Database"]), reaction_energy, y_batch
        )
        update_exc_errors(train_exc_errors, list(X_vxc["Names"]), pred_exc, ref_exc)
        sums["loss"] += float((reaction_loss + vxc_term + exc_term).item())
        sums["reaction_loss"] += float(reaction_loss.item())
        sums["vxc_loss"] += float(vxc_term.item())
        sums["exc_loss"] += float(exc_term.item())
        sums["mae"] += float(nn.functional.l1_loss(reaction_energy, y_batch).item())
        for name, value in diagnostics.items():
            sums[name] += value

        del reaction_loss, reaction_energy, reaction_grads
        del vxc_term, vxc_grads
        del exc_term, exc_grads, pred_exc, ref_exc, combined

        if not do_step:
            continue
        if sync_failure(not grads_are_finite(trainable_parameters), device):
            failed = True
            break
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    if failed:
        optimizer.zero_grad(set_to_none=True)
        return {}, {}, True

    scalar_names = [
        "loss",
        "reaction_loss",
        "vxc_loss",
        "exc_loss",
        "mae",
        "raw_norm_reaction",
        "raw_norm_vxc",
        "raw_norm_exc",
        "cos_reaction_vxc",
        "cos_reaction_exc",
        "cos_vxc_exc",
        "combined_norm",
        "weight_reaction",
        "weight_vxc",
        "weight_exc",
    ]
    scalar = torch.tensor(
        [
            *(sums[name] for name in scalar_names),
            float(n_steps),
            float(optimizer_steps),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    denom = max(float(scalar[-2].item()), 1.0)

    gathered_errors = gather_object(dict(train_db_errors), world_size)
    global_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_errors:
        for database, errors in local_dict.items():
            global_errors[database].extend(errors)
    train_fchem, train_per_db = compute_fchem_from_errors(global_errors)

    gathered_exc_errors = gather_object(dict(train_exc_errors), world_size)
    global_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_exc_errors:
        for system_name, errors in local_dict.items():
            global_exc_errors[system_name].extend(errors)
    train_exc, train_per_system_exc = compute_exc_from_errors(global_exc_errors)

    averaged = {
        name: float(scalar[index].item() / denom)
        for index, name in enumerate(scalar_names)
    }
    metrics: Dict[str, Any] = {
        "train_full_loss": averaged["loss"],
        "train_reaction_loss": averaged["reaction_loss"],
        "train_vxc": averaged["vxc_loss"],
        "train_exc_loss": averaged["exc_loss"],
        "train_mae": averaged["mae"],
        "train_fchem": train_fchem,
        "train_exc": train_exc,
        "train_per_system_exc_rmse": train_per_system_exc,
        "optimizer_steps": int(round(float(scalar[-1].item()) / max(world_size, 1))),
        **{
            f"geometry_{name}": value
            for name, value in averaged.items()
            if name not in {"loss", "reaction_loss", "vxc_loss", "exc_loss", "mae"}
        },
    }
    return metrics, dict(train_per_db), False


def run_trial_geometry(
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
    epoch_history: List[Dict[str, Any]] = []
    selected_checkpoint_path = None
    current_selected_key = None
    failed = False
    minima = {
        "val_fchem": math.inf,
        "val_vxc": math.inf,
        "val_exc": math.inf,
        "val_full_loss": math.inf,
    }

    if args.save_selected_checkpoints and rank0:
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        selected_checkpoint_path = checkpoints_dir / f"trial_{trial_number}_selected.pt"

    for epoch in range(args.n_train):
        epoch_number = epoch + 1
        train_dataset = loaders["train_loader"].dataset
        if hasattr(train_dataset, "resample"):
            train_dataset.resample(epoch)
        for sampler_name in ("train_sampler", "vxc_train_sampler", "val_sampler"):
            sampler = loaders.get(sampler_name)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        train_metrics, train_per_db, train_failed = train_one_epoch_geometry(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train_loader"],
            vxc_train_loader=loaders["vxc_train_loader"],
            params=params,
            aggregation=args.aggregation,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(args.include_mrks_dispersion),
            world_size=world_size,
        )
        if train_failed:
            failed = True
            break
        val_metrics, val_per_db, val_failed = validate_one_epoch(
            model=model,
            val_loader=loaders["val_loader"],
            vxc_val_loader=loaders["vxc_val_loader"],
            params=params,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(args.include_mrks_dispersion),
            world_size=world_size,
        )
        if val_failed:
            failed = True
            break
        scheduler.step()

        row: Dict[str, Any] = {
            "epoch": epoch_number,
            "aggregation": args.aggregation,
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
            "phase_name": "one_stage_gradient_geometry",
            "phase_start_epoch": 1,
            "phase_end_epoch": args.n_train,
            "effective_accum_iter": params["accum_iter"],
            "effective_vxc_loss_scale": 1.0,
            "effective_exc_loss_scale": 1.0,
            **{
                key: value
                for key, value in train_metrics.items()
                if key.startswith("geometry_")
            },
        }
        epoch_history.append(row)
        for metric in minima:
            minima[metric] = min(minima[metric], float(row[metric]))

        candidate_key = last_epoch_checkpoint_key(row)
        if current_selected_key is None or candidate_key < current_selected_key:
            current_selected_key = candidate_key
            if (
                args.save_selected_checkpoints
                and rank0
                and selected_checkpoint_path is not None
            ):
                torch.save(model.module.state_dict(), selected_checkpoint_path)

        if rank0:
            print(
                f"Geometry {args.aggregation} Trial {trial_number} epoch {epoch_number}/{args.n_train}: "
                f"train_fchem={row['train_fchem']:.8f} val_fchem={row['val_fchem']:.8f} "
                f"val_vxc={row['val_vxc']:.8f} val_exc={row['val_exc']:.8f} "
                f"cos(F,V)={row['geometry_cos_reaction_vxc']:.4f} "
                f"cos(F,E)={row['geometry_cos_reaction_exc']:.4f}"
            )

    failed = sync_failure(failed, device)
    if failed:
        return {
            "failed": True,
            "trial_number": trial_number,
            "params": params,
            "aggregation": args.aggregation,
        }
    selected = select_last_epoch(epoch_history)
    payload = {
        "failed": False,
        "trial_number": trial_number,
        "params": params,
        "aggregation": args.aggregation,
        "tasks": list(TASKS),
        "selected_epoch": int(selected["epoch"]),
        "selected_joint_score": float(selected["val_joint_score"]),
        "selected_train_fchem": float(selected["train_fchem"]),
        "selected_val_fchem": float(selected["val_fchem"]),
        "selected_val_vxc": float(selected["val_vxc"]),
        "selected_val_exc": float(selected["val_exc"]),
        "selected_phase_name": selected["phase_name"],
        "min_val_fchem_any_epoch": minima["val_fchem"],
        "min_val_vxc_any_epoch": minima["val_vxc"],
        "min_val_exc_any_epoch": minima["val_exc"],
        "best_val_full_loss_any_epoch": minima["val_full_loss"],
        "epoch_history": epoch_history,
        "selected_checkpoint_path": str(selected_checkpoint_path)
        if selected_checkpoint_path
        else None,
    }
    if rank0:
        history_path = save_trial_history(output_dir, trial_number, payload)
        payload["history_path"] = str(history_path)
    else:
        payload["history_path"] = None
    return payload


def main() -> None:
    args = parse_args()
    if args.n_train != 500:
        raise ValueError(
            "The gradient-geometry comparison is fixed to exactly 500 epochs."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_rank, world_size, device, rank0 = init_distributed()
    set_random_seed(args.seed + args.trial_number)
    if args.no_reaction_dispersion:
        dispersions = {}
    else:
        with (
            Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle"
        ).open("rb") as handle:
            dispersions = pickle.load(handle)
    mrks_dispersions = (
        load_mrks_dispersions(args.mrks_dispersions_pickle)
        if args.include_mrks_dispersion
        else None
    )
    data_predopt, data_train, data_val, data_vxc_train, data_vxc_val = load_chk(
        path=args.checkpoints_dir
    )
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
    params = dict(ONE_STAGE_PARAMS)
    result = run_trial_geometry(
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
            print(f"Gradient-geometry replay failed: {args.aggregation}")
            return
        final = select_last_epoch(result["epoch_history"])
        print(f"Gradient-geometry replay complete: {args.aggregation}")
        print(f"Final selected epoch: {final['epoch']}")
        print(
            f"Final metrics: train_fchem={final['train_fchem']:.8f}, "
            f"val_vxc={final['val_vxc']:.8f}, val_exc={final['val_exc']:.8f}, "
            f"val_fchem={final['val_fchem']:.8f}"
        )
        print(f"Selected checkpoint: {result.get('selected_checkpoint_path')}")
        print(f"History path: {result.get('history_path')}")
        print(f"Params: {json.dumps(params, sort_keys=True)}")


if __name__ == "__main__":
    main()
