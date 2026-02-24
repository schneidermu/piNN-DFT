"""
Main training script for pcPBELMLOptimizerV2.

Two-phase training:
  1. Pre-optimization: warm-starts the model to reproduce true PBE constants.
  2. Main training: jointly minimises reaction energy (Fchem) and Vxc physics loss.

Usage (distributed, 2 GPUs):
    torchrun --nproc_per_node=2 predopt_train.py --name PBE-L_8_32 --omega 0.067
"""

import argparse
import collections
import os
import pickle
import random
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.distributed as dist
from dotenv import find_dotenv, load_dotenv
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import collate_fn, fast_collate_fn_predopt
from NN_models import pcPBELMLOptimizerV2
from predopt import DatasetPredopt, predopt, true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import calculate_reaction_energy, get_local_energies
from utils import configure_optimizers, seed_worker, set_random_seed

set_random_seed(41)
g = torch.Generator()
g.manual_seed(41)

# ---------------------------------------------------------------------------
# Database weighting dictionaries
# ---------------------------------------------------------------------------

FCHEM_VALIDATION = {
    "ABDE4": 1,
    "AE17": 0.25,
    "DBH76": 1,
    "EA13": 1,
    "IP13": 1,
    "MGAE109": 1 / 4.73394495412844,
    "NCCE31": 10,
    "PA8": 1,
    "pTC13": 1,
}

FREQ_WEIGHTS = {
    "ABDE4": 1 / 4,
    "AE17": 1 / 17,
    "DBH76": 1 / 76,
    "EA13": 1 / 13,
    "IP13": 1 / 13,
    "MGAE109": 1 / 109,
    "NCCE31": 1 / 31,
    "PA8": 1 / 8,
    "pTC13": 1 / 13,
}

mean_weight = np.mean(
    np.array([FCHEM_VALIDATION[db] * FREQ_WEIGHTS[db] for db in FCHEM_VALIDATION])
)
mean_freq_weight = np.mean(np.array([FREQ_WEIGHTS[db] for db in FREQ_WEIGHTS]))

PBE_TRAIN_ERRORS = {
    "ABDE4": 4.039,
    "AE17": 45.612,
    "DBH76": 9.558,
    "EA13": 3.184,
    "IP13": 4.829,
    "MGAE109": 19.818,
    "NCCE31": 1.699,
    "PA8": 1.267,
    "pTC13": 7.230,
}

PBE_VALIDATION_ERRORS = {
    "ABDE4": 0.117,
    "AE17": 68.046,
    "DBH76": 11.055,
    "EA13": 1.856,
    "IP13": 3.958,
    "MGAE109": 15.984,
    "NCCE31": 1.540,
    "PA8": 2.634,
    "pTC13": 4.150,
}

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

_VXC_LOSS_SCALE: float = 1000.0        # scaling applied to vxc loss before blending
_WARMUP_EPOCHS: int = 5                 # linear LR warm-up duration
_WARMUP_START_FACTOR: float = 0.001    # initial LR fraction at warm-up start
_MIN_LR: float = 1e-6                  # cosine annealing lower bound
_EARLY_STOP_PATIENCE: int = 50
_DEFAULT_SMOOTHING_WINDOW: int = 10    # epochs before best-model tracking starts
_BEST_MODEL_DIR: str = "best_models/"
_PLOT_DIR: str = "./batch_fchem/"
# Plotting scale factors (for visualization only, not training)
_TRAIN_FCHEM_PLOT_SCALE: float = 50.0
_VAL_FCHEM_PLOT_SCALE: float = 15.0


# ---------------------------------------------------------------------------
# Utility: trainable log-scale toggle
# ---------------------------------------------------------------------------

def set_scales_trainable(model: nn.Module, trainable: bool = True) -> None:
    """
    Toggles the trainability of the descriptor log-scale parameters.

    During pre-optimization these are frozen so that the network learns to
    reproduce PBE constants without distorting the descriptor normalization.
    They are unfrozen for the main training phase.

    Args:
        model: DDP-wrapped or bare pcPBELMLOptimizerV2 model.
        trainable: Whether to enable gradient computation for scale params.
    """
    base_model = model.module if hasattr(model, "module") else model
    for p in [base_model.log_scale_rho, base_model.log_scale_sigma,
              base_model.log_scale_tau, base_model.log_scale_lapl]:
        p.requires_grad = trainable
    status = "ENABLED" if trainable else "DISABLED"
    print(f"--- Scale Parameter Training: {status} ---")


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class VxcDataset(torch.utils.data.Dataset):
    """Dataset wrapper for Vxc data loaded from pickle."""

    def __init__(self, data_list: list) -> None:
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset that returns one randomly chosen grid augmentation per reaction
    at each access. This prevents the model from overfitting to a single
    integration grid per molecule.

    Args:
        data: Dict mapping base reaction key → list of augmented reaction dicts.
    """

    def __init__(self, data: dict) -> None:
        self.reaction_groups = list(data.values())

    def __len__(self) -> int:
        return len(self.reaction_groups)

    def __getitem__(self, idx: int):
        chosen = random.choice(self.reaction_groups[idx])
        return chosen, chosen["Energy"]


class EarlyStopper:
    """Stops training when validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 1, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ---------------------------------------------------------------------------
# Collate function for Vxc data
# ---------------------------------------------------------------------------

def vxc_collate_fn(batch: list) -> dict:
    """Concatenates grid points from multiple systems into one large batch."""
    return {
        "Grid":    torch.cat([item["Grid"]    for item in batch], dim=0),
        "Vrho":    torch.cat([item["Vrho"]    for item in batch], dim=0),
        "Weights": torch.cat([item["Weights"] for item in batch], dim=0),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_function(
    factor_dictionary: dict,
    total_database_errors: dict,
    val: bool = False,
) -> float:
    """
    Computes the weighted sum of per-database RMSEs and prints a comparison
    against PBE baseline errors.

    Args:
        factor_dictionary: Per-database importance weights.
        total_database_errors: Dict mapping db name → list of absolute errors.
        val: If True, compare against PBE_VALIDATION_ERRORS; else PBE_TRAIN_ERRORS.

    Returns:
        Weighted sum of RMSEs across all databases.
    """
    err_ref = PBE_VALIDATION_ERRORS if val else PBE_TRAIN_ERRORS
    fchem = 0.0
    for db in sorted(total_database_errors):
        factor = FCHEM_VALIDATION.get(db, 1)
        error = np.sqrt(np.mean(np.array(total_database_errors[db]) ** 2))
        fchem += error * factor
        print(f"{db}: {error:6.3f}, {100 * (error - err_ref.get(db, 1)) / err_ref.get(db, 1):6.3f}%")
    return fchem


def batch_fchem(
    current_bases: list,
    reaction_energy: torch.Tensor,
    y_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the database-weighted reaction energy loss.

    Loss = mean over databases of { w_db · sqrt(MSE_db + eps) }

    Args:
        current_bases: List of database names for each reaction in the batch.
        reaction_energy: Predicted reaction energies, shape (N,).
        y_batch: Reference reaction energies, shape (N,).

    Returns:
        Scalar loss tensor.
    """
    err_dict: dict = {}
    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict.setdefault(database, [[], []])
        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)

    fchem = []
    for database, (preds, refs) in err_dict.items():
        db_predictions = torch.stack(preds)
        db_ref = torch.stack(refs)
        factor = FCHEM_VALIDATION.get(database, 1) * FREQ_WEIGHTS.get(database, 1) / mean_weight
        mse = nn.functional.mse_loss(db_predictions, db_ref)
        fchem.append(factor * torch.sqrt(1e-20 + mse))

    return torch.sum(torch.stack(fchem)) / len(fchem)


def batch_mse_weighted(
    current_bases: list,
    reaction_energy: torch.Tensor,
    y_batch: torch.Tensor,
    normalization_function=torch.sqrt,
    do_factor: bool = True,
) -> torch.Tensor:
    """
    Computes a frequency-weighted, reference-normalized MSE loss.

    Args:
        current_bases: Database names per reaction.
        reaction_energy: Predicted energies.
        y_batch: Reference energies.
        normalization_function: Applied to |ref| for per-sample normalization.
        do_factor: Whether to apply FREQ_WEIGHTS normalization.

    Returns:
        Scalar loss tensor.
    """
    err_dict: dict = {}
    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict.setdefault(database, [[], []])
        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)

    fchem = []
    for database, (preds, refs) in err_dict.items():
        db_predictions = torch.stack(preds)
        db_ref = torch.stack(refs)
        sq_err = (db_predictions - db_ref) ** 2 / (1e-3 + normalization_function(torch.abs(db_ref)))
        if do_factor:
            fchem.append(FREQ_WEIGHTS.get(database, 1) / mean_freq_weight * sq_err)
        else:
            fchem.append(sq_err)

    return torch.sum(torch.stack(fchem)) / len(y_batch)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extend_bases(X_batch: dict, bases: list) -> tuple:
    """Extracts the database label(s) from the current batch."""
    if len(X_batch["Database"][0]) == 1:
        current_bases = [X_batch["Database"]]
    else:
        current_bases = list(X_batch["Database"])
    bases += current_bases
    return current_bases, bases


def make_total_db_errors(
    pred_energies,
    reaction_energy: torch.Tensor,
    errors,
    ref_energies,
    y_batch: torch.Tensor,
    total_database_errors: dict,
    current_bases: list,
) -> tuple:
    """Accumulates per-database absolute errors over the course of an epoch."""
    if len(pred_energies):
        pred_energies = torch.hstack([pred_energies, reaction_energy])
        ref_energies  = torch.hstack([ref_energies,  y_batch])
        errors        = torch.hstack([errors, reaction_energy - y_batch])
    else:
        pred_energies = reaction_energy
        ref_energies  = y_batch
        errors        = reaction_energy - y_batch

    for base, error in zip(current_bases, reaction_energy - y_batch):
        total_database_errors.setdefault(base, [])
        total_database_errors[base].append(torch.abs(error).item())

    return pred_energies, ref_energies, errors, total_database_errors


# ---------------------------------------------------------------------------
# Vxc physics loss
# ---------------------------------------------------------------------------

def vxc_loss(
    model: nn.Module,
    X_batch: dict,
    device: torch.device,
    rung: str = "GGA",
    dft: str = "PBE",
) -> torch.Tensor:
    """
    Computes the physics-based Vrho loss using automatic differentiation.

    The loss measures the integrated squared difference between the predicted
    exchange-correlation potential (∂(ε_xc·ρ)/∂ρ via autograd) and the target
    Vrho from the training data, weighted by density and integration weights:

        L_vxc = Σ ρ_tot · w · (V_xc_pred − V_xc_ref)² / Σ ρ_tot · w

    The model input is reconstructed with ρ and σ as leaf tensors so that
    autograd can differentiate through the energy with respect to ρ.

    Args:
        model: The DDP-wrapped pcPBELMLOptimizerV2 model.
        X_batch: Dict with keys 'Grid', 'Vrho', 'Weights'.
            Grid columns: [x, y, z, w, ρ_α, ρ_β, σ_αα, σ_tot, σ_ββ, τ_α, τ_β, ...]
        device: Target device.
        rung: DFT rung identifier ("GGA" or "LDA").
        dft: DFT functional identifier ("PBE").

    Returns:
        Scalar loss tensor with gradient graph retained.
    """
    grid_raw = X_batch["Grid"].to(device).clone().detach()
    rho   = grid_raw[:, 4:6].clone().requires_grad_(True)
    sigma = grid_raw[:, 6:9].clone().requires_grad_(True)

    target_vrho = X_batch["Vrho"].to(device)
    weights     = X_batch["Weights"].to(device)

    model_input = torch.cat([rho, sigma, grid_raw[:, 9:]], dim=1)
    constants   = model(model_input)

    calc_data = get_local_energies(
        {"Densities": rho, "Gradients": sigma, "Weights": weights},
        constants, device, rung=rung, dft=dft, enhancement=None,
    )

    rho_tot    = rho[:, 0] + rho[:, 1]
    e_xc_pred  = calc_data["Local_energies"] * rho_tot

    grads = torch.autograd.grad(
        outputs=e_xc_pred,
        inputs=rho,
        grad_outputs=torch.ones_like(e_xc_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    pred_vrho = (grads[:, 0] + grads[:, 1]) / 2.0

    rho_total_detached = rho_tot.detach()
    diff_sq      = (pred_vrho - target_vrho) ** 2
    loss_integral = torch.sum(rho_total_detached * weights * diff_sq)
    norm_factor   = torch.sum(rho_total_detached * weights)
    return loss_integral / (norm_factor + 1e-10)


# ---------------------------------------------------------------------------
# Training sub-functions
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    vxc_train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    vxc_iter,
    omega: float,
    accum_iter: int,
    device: torch.device,
    rung: str,
    dft: str,
    dispersions: dict,
    local_rank: int,
) -> tuple:
    """
    Runs one training epoch.

    Returns:
        (epoch_loss_sum, epoch_mae_sum, epoch_vxc_sum,
         epoch_db_errors, vxc_iter)
        vxc_iter is returned so its state persists across epochs.
    """
    model.train()
    train_loader.sampler.set_epoch(train_loader.sampler.epoch
                                   if hasattr(train_loader.sampler, "epoch") else 0)

    epoch_loss_sum: float = 0.0
    epoch_mae_sum:  float = 0.0
    epoch_vxc_sum:  float = 0.0
    epoch_db_errors: dict = collections.defaultdict(list)

    progress_bar = tqdm(
        train_loader,
        disable=(local_rank != 0),
        mininterval=2.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
        X_batch_grid = X_batch["Grid"].to(device, non_blocking=True)
        y_batch      = y_batch.to(device, non_blocking=True)
        current_bases, _ = extend_bases(X_batch=X_batch, bases=[])

        try:
            X_vxc = next(vxc_iter)
        except StopIteration:
            vxc_iter = iter(vxc_train_loader)
            X_vxc = next(vxc_iter)

        predictions = model(X_batch_grid)
        reaction_energy, _ = calculate_reaction_energy(
            X_batch, predictions, device, rung=rung, dft=dft, dispersions=dispersions
        )

        batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)
        loss_vxc         = vxc_loss(model, X_vxc, device, rung=rung, dft=dft)
        loss             = (1 - omega) * batch_fchem_loss + omega * loss_vxc * _VXC_LOSS_SCALE

        loss.backward()

        if ((batch_idx + 1) % accum_iter == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        MAE = nn.functional.l1_loss(reaction_energy, y_batch).item()
        epoch_loss_sum += loss.item()
        epoch_mae_sum  += MAE
        epoch_vxc_sum  += loss_vxc.item()

        _, _, _, epoch_db_errors = make_total_db_errors(
            [], reaction_energy, [], [], y_batch, epoch_db_errors, current_bases
        )

    return epoch_loss_sum, epoch_mae_sum, epoch_vxc_sum, epoch_db_errors, vxc_iter


def _validate_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    vxc_test_loader: torch.utils.data.DataLoader,
    omega: float,
    device: torch.device,
    rung: str,
    dft: str,
    dispersions: dict,
    local_rank: int,
) -> tuple:
    """
    Runs one validation epoch.

    Returns:
        (val_loss_sum, val_mae_sum, val_vxc_sum, val_samples_count, val_db_errors)
    """
    model.eval()

    val_loss_sum:     float = 0.0
    val_mae_sum:      float = 0.0
    val_vxc_sum:      float = 0.0
    val_samples_count: int  = 0
    val_db_errors:    dict  = collections.defaultdict(list)
    vxc_val_iter = iter(vxc_test_loader)

    progress_bar = tqdm(
        test_loader,
        disable=(local_rank != 0),
        mininterval=2.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for X_batch, y_batch in progress_bar:
            X_batch_grid = X_batch["Grid"].to(device, non_blocking=True)
            y_batch      = y_batch.to(device, non_blocking=True)
            current_bases, _ = extend_bases(X_batch=X_batch, bases=[])

            predictions = model(X_batch_grid)
            reaction_energy, _ = calculate_reaction_energy(
                X_batch, predictions, device, rung=rung, dft=dft, dispersions=dispersions
            )

            try:
                X_vxc = next(vxc_val_iter)
            except StopIteration:
                vxc_val_iter = iter(vxc_test_loader)
                X_vxc = next(vxc_val_iter)

            with torch.enable_grad():
                loss_vxc_val = vxc_loss(model, X_vxc, device, rung=rung, dft=dft)

            batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)
            loss = (1 - omega) * batch_fchem_loss + omega * loss_vxc_val * _VXC_LOSS_SCALE

            curr_batch_size     = y_batch.size(0)
            val_vxc_sum        += loss_vxc_val.item() * curr_batch_size
            val_loss_sum       += loss.item() * curr_batch_size
            val_mae_sum        += nn.functional.l1_loss(reaction_energy, y_batch).item() * curr_batch_size
            val_samples_count  += curr_batch_size

            _, _, _, val_db_errors = make_total_db_errors(
                [], reaction_energy, [], [], y_batch, val_db_errors, current_bases
            )

    return val_loss_sum, val_mae_sum, val_vxc_sum, val_samples_count, val_db_errors


def _sync_metrics(
    train_sums: tuple,
    val_sums: tuple,
    train_db_errors: dict,
    val_db_errors: dict,
    world_size: int,
    device: torch.device,
) -> tuple:
    """
    Performs DDP all_reduce on scalar metrics and all_gather_object on
    per-database error dictionaries.

    Args:
        train_sums: (loss_sum, mae_sum, vxc_sum) from _train_epoch.
        val_sums:   (loss_sum, mae_sum, vxc_sum, n_samples) from _validate_epoch.
        train_db_errors: Local per-database errors from training.
        val_db_errors:   Local per-database errors from validation.
        world_size: Number of DDP processes.
        device: Device for tensor operations.

    Returns:
        (avg_train_loss, avg_train_mae, avg_train_vxc,
         avg_val_loss, avg_val_mae, avg_val_vxc,
         global_train_errors, global_val_errors,
         n_train_batches_per_rank)
        Note: avg_train_* are per-batch averages; avg_val_* are per-sample averages.
    """
    # --- Scalar sync ---
    train_loss_sum, train_mae_sum, train_vxc_sum = train_sums
    val_loss_sum, val_mae_sum, val_vxc_sum, val_samples = val_sums

    metrics_train = torch.tensor(
        [train_loss_sum, train_mae_sum, train_vxc_sum], device=device
    )
    dist.all_reduce(metrics_train, op=dist.ReduceOp.SUM)
    total_train_loss, total_train_mae, total_train_vxc = (metrics_train / world_size).tolist()

    metrics_val = torch.tensor(
        [val_loss_sum, val_mae_sum, val_vxc_sum, val_samples], device=device
    )
    dist.all_reduce(metrics_val, op=dist.ReduceOp.SUM)
    total_val_loss, total_val_mae, total_val_vxc, total_val_samples = metrics_val.tolist()

    # --- Per-database error sync ---
    gathered_train = [None] * world_size
    gathered_val   = [None] * world_size
    dist.all_gather_object(gathered_train, train_db_errors)
    dist.all_gather_object(gathered_val,   val_db_errors)

    global_train_errors: dict = collections.defaultdict(list)
    for local_dict in gathered_train:
        for db, errs in local_dict.items():
            global_train_errors[db].extend(errs)

    global_val_errors: dict = collections.defaultdict(list)
    for local_dict in gathered_val:
        for db, errs in local_dict.items():
            global_val_errors[db].extend(errs)

    # --- Compute averages ---
    # Training: averaged over all ranks × batches (not per-sample)
    avg_train_loss = total_train_loss
    avg_train_mae  = total_train_mae
    avg_train_vxc  = total_train_vxc

    n = int(total_val_samples) if total_val_samples > 0 else 1
    avg_val_loss = total_val_loss / n
    avg_val_mae  = total_val_mae  / n
    avg_val_vxc  = total_val_vxc  / n

    return (avg_train_loss, avg_train_mae, avg_train_vxc,
            avg_val_loss, avg_val_mae, avg_val_vxc,
            global_train_errors, global_val_errors)


def _log_metrics(
    run,
    epoch: int,
    train_metrics: tuple,
    val_metrics: tuple,
    global_train_errors: dict,
    global_val_errors: dict,
    model: nn.Module,
    omega: float,
    train_history: dict,
    val_history: dict,
    n_train_batches: int,
) -> tuple:
    """
    Appends metrics to history lists and logs them to Neptune (rank 0 only).

    Args:
        run: Neptune run object, or None if Neptune is not configured.
        epoch: Current epoch index (0-based).
        train_metrics: (avg_loss, avg_mae, avg_vxc) per-batch averages.
        val_metrics:   (avg_loss, avg_mae, avg_vxc) per-sample averages.
        global_train_errors: Aggregated per-database training errors.
        global_val_errors:   Aggregated per-database validation errors.
        model: DDP-wrapped model (for reading log_scale params).
        omega: Vxc loss weight (used for plot scaling).
        train_history: Dict of lists for tracking train metrics across epochs.
        val_history:   Dict of lists for tracking val metrics across epochs.
        n_train_batches: Number of training batches per rank (for averaging).

    Returns:
        (train_fchem, val_fchem): Weighted database RMSE sums for this epoch.
    """
    avg_train_loss, avg_train_mae, avg_train_vxc = train_metrics
    avg_val_loss,   avg_val_mae,   avg_val_vxc   = val_metrics

    # Normalize training metrics by batch count
    avg_train_loss = avg_train_loss / n_train_batches if n_train_batches > 0 else 0
    avg_train_mae  = avg_train_mae  / n_train_batches if n_train_batches > 0 else 0
    avg_train_vxc  = avg_train_vxc  / n_train_batches if n_train_batches > 0 else 0

    train_history["full_loss"].append(avg_train_loss)
    train_history["mae"].append(avg_train_mae)
    train_history["vxc"].append(avg_train_vxc)
    val_history["full_loss"].append(avg_val_loss)
    val_history["mae"].append(avg_val_mae)
    val_history["vxc"].append(avg_val_vxc)

    if mlflow.active_run() is not None:
        # Log primary losses
        mlflow.log_metric("train/full_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("validation/full_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("train/vxc_loss", avg_train_vxc, step=epoch)
        mlflow.log_metric("validation/vxc_loss", avg_val_vxc, step=epoch)

        # Log per-database RMSEs
        for db, errors_list in global_train_errors.items():
            rmse = np.sqrt(np.mean(np.square(errors_list)))
            mlflow.log_metric(f"train/{db}_rmse", rmse, step=epoch)
        for db, errors_list in global_val_errors.items():
            rmse = np.sqrt(np.mean(np.square(errors_list)))
            mlflow.log_metric(f"validation/{db}_rmse", rmse, step=epoch)

        # Log scaling parameters
        base_model = model.module if hasattr(model, "module") else model
        if hasattr(base_model, "log_scale_rho"):
            mlflow.log_metric("scaling_params/rho", torch.exp(base_model.log_scale_rho).item(), step=epoch)
            mlflow.log_metric("scaling_params/sigma", torch.exp(base_model.log_scale_sigma).item(), step=epoch)
            mlflow.log_metric("scaling_params/tau", torch.exp(base_model.log_scale_tau).item(), step=epoch)
        if hasattr(base_model, "log_scale_lapl"):
            mlflow.log_metric("scaling_params/lapl", torch.exp(base_model.log_scale_lapl).item(), step=epoch)

    print(f"\n--- Epoch {epoch + 1} Summary ---")
    print("Training Set Metrics:")
    train_fchem = loss_function(FCHEM_VALIDATION, global_train_errors, val=False)
    print(f"Global Train Fchem: {train_fchem:.4f}\n")

    print("Validation Set Metrics:")
    val_fchem = loss_function(FCHEM_VALIDATION, global_val_errors, val=True)
    print(f"Global Validation Fchem: {val_fchem:.4f}\n")

    # Composite plot metric (scales chosen for visual clarity)
    train_history["fchem"].append((1 - omega) * train_fchem / _TRAIN_FCHEM_PLOT_SCALE + omega * avg_train_vxc * 100)
    val_history["fchem"].append(  (1 - omega) * val_fchem   / _VAL_FCHEM_PLOT_SCALE   + omega * avg_val_vxc   * 200)

    # Track pure reaction loss (fchem) for plotting
    train_history["reaction_loss"].append(train_fchem)
    val_history["reaction_loss"].append(val_fchem)

    return train_fchem, val_fchem


def _save_checkpoint(
    model: nn.Module,
    epoch: int,
    batch_size: int,
    lr_train: float,
    name: str,
    omega: float,
    avg_train_loss: float,
    avg_val_loss: float,
    train_fchem: float,
    val_fchem: float,
    val_full_loss: list,
    val_loss_window: collections.deque,
    prev: Optional[str],
    prev_best: Optional[str],
    best_model_dir: str,
) -> tuple:
    """
    Saves epoch checkpoint and conditionally a 'BEST' checkpoint.

    Deletes the previous epoch checkpoint (keeping only the latest) and
    replaces the best checkpoint when a new minimum validation loss is reached
    after the initial smoothing window.

    Returns:
        (new_prev, new_prev_best): Updated checkpoint paths.
    """
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    if prev and os.path.exists(prev):
        os.remove(prev)
    prev = (
        f"{best_model_dir}bs_{batch_size}_lr_{lr_train}_{name}_{omega}"
        f"_epoch_{epoch + 1}"
        f"_train_loss_{avg_train_loss:.3f}_val_loss_{avg_val_loss:.3f}"
        f"_train_fchem_{train_fchem:.3f}_val_fchem_{val_fchem:.3f}.pth"
    )
    torch.save(state_dict, prev)

    val_loss_window.append(avg_val_loss)
    if len(val_loss_window) == val_loss_window.maxlen:
        if avg_val_loss <= min(val_full_loss):
            print(f"New best Val loss: {avg_val_loss:.3f}. Saving model.")
            if prev_best and os.path.exists(prev_best):
                try:
                    os.remove(prev_best)
                except OSError:
                    pass
            prev_best = (
                f"{best_model_dir}BEST_EPOCH_bs_{batch_size}_lr_{lr_train}_{name}_{omega}"
                f"_epoch_{epoch + 1}"
                f"_train_loss_{avg_train_loss:.3f}_val_loss_{avg_val_loss:.3f}"
                f"_train_fchem_{train_fchem:.3f}_val_fchem_{val_fchem:.3f}.pth"
            )
            torch.save(state_dict, prev_best)

    return prev, prev_best


def _plot_losses(
    train_full_loss: list,
    val_full_loss: list,
    train_fchem_loss: list,
    test_fchem: list,
    batch_size: int,
    lr_train: float,
    name: str,
    omega: float,
    plot_dir: str,
) -> None:
    """Renders and saves the two-panel loss plot. Closes the figure to free memory."""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[3, 6], sharex=True)
    ax[0].plot(train_full_loss, label="Train Loss")
    ax[0].plot(val_full_loss,   label="Validation Loss")
    ax[1].plot(train_fchem_loss, label="Train Fchem")
    ax[1].plot(test_fchem,       label="Validation Fchem")
    ax[0].legend()
    ax[1].legend()
    plt.savefig(f"{plot_dir}bs_{batch_size}_lr_{lr_train}_{name}_{omega}.png")
    plt.close(fig)


def _plot_reaction_and_vxc_losses(
    train_reaction_loss: list,
    val_reaction_loss: list,
    train_vxc_loss: list,
    val_vxc_loss: list,
) -> str:
    """
    Creates a 2-subplot figure showing reaction loss and vxc loss per epoch.

    Args:
        train_reaction_loss: List of training reaction (fchem) loss values per epoch.
        val_reaction_loss: List of validation reaction (fchem) loss values per epoch.
        train_vxc_loss: List of training vxc loss values per epoch.
        val_vxc_loss: List of validation vxc loss values per epoch.

    Returns:
        Path to the saved plot file.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

    # Subplot 1: Reaction Loss per Epoch
    axes[0].plot(train_reaction_loss, label="Train Reaction Loss", linewidth=2)
    axes[0].plot(val_reaction_loss, label="Validation Reaction Loss", linewidth=2)
    axes[0].set_ylabel("Reaction Loss (kcal/mol)", fontsize=12)
    axes[0].set_title("Reaction Loss per Epoch", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Vxc Loss per Epoch
    axes[1].plot(train_vxc_loss, label="Train Vxc Loss", linewidth=2)
    axes[1].plot(val_vxc_loss, label="Validation Vxc Loss", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Vxc Loss", fontsize=12)
    axes[1].set_title("Vxc Loss per Epoch", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to a temporary file
    plot_path = "loss_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return plot_path


# ---------------------------------------------------------------------------
# Main training coordinator
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    early_stopper: EarlyStopper,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    vxc_train_loader: torch.utils.data.DataLoader,
    vxc_test_loader: torch.utils.data.DataLoader,
    run,
    n_epochs: int = 25,
    accum_iter: int = 1,
    omega: float = 0.067,
    smoothing_window: int = _DEFAULT_SMOOTHING_WINDOW,
    local_rank: int = 0,
    device: torch.device = torch.device("cpu"),
    rung: str = "GGA",
    dft: str = "PBE",
    dispersions: Optional[dict] = None,
    batch_size: int = 3,
    lr_train: float = 1e-4,
    name: str = "",
) -> tuple:
    """
    Orchestrates the main training loop.

    Coordinates per-epoch training/validation, DDP metric synchronization,
    Neptune logging, checkpointing and plotting.

    Args:
        model: DDP-wrapped pcPBELMLOptimizerV2.
        optimizer: Configured optimizer (from configure_optimizers).
        scheduler: LR scheduler (SequentialLR: LinearLR warm-up + CosineAnnealingLR).
        early_stopper: EarlyStopper instance.
        train_loader: DataLoader for training reactions.
        test_loader: DataLoader for validation reactions.
        vxc_train_loader: DataLoader for training Vxc data.
        vxc_test_loader: DataLoader for validation Vxc data.
        run: Neptune run object (or None).
        n_epochs: Number of training epochs.
        accum_iter: Gradient accumulation steps.
        omega: Weight of Vxc physics loss (0 = pure reaction energy, 1 = pure Vxc).
        smoothing_window: Epochs before best-model tracking activates.
        local_rank: DDP local rank.
        device: Compute device.
        rung: DFT rung for energy calculation ("GGA" or "LDA").
        dft: DFT functional identifier ("PBE").
        dispersions: Dispersion correction lookup dict.
        batch_size: Reactions per batch (used only for checkpoint filenames).
        lr_train: Learning rate (used only for checkpoint filenames).
        name: Model identifier string (used for filenames and logging).

    Returns:
        (train_full_loss, val_full_loss, best_model_path)
    """
    train_history = {"full_loss": [], "mae": [], "vxc": [], "fchem": [], "reaction_loss": []}
    val_history   = {"full_loss": [], "mae": [], "vxc": [], "fchem": [], "reaction_loss": []}

    prev, prev_best = None, None
    val_loss_window = collections.deque(maxlen=smoothing_window)
    world_size = dist.get_world_size()

    if local_rank == 0:
        os.makedirs(_BEST_MODEL_DIR, exist_ok=True)
        os.makedirs(_PLOT_DIR, exist_ok=True)

    vxc_iter = iter(vxc_train_loader)

    for epoch in range(n_epochs):
        train_loader.sampler.set_epoch(epoch)

        # ---- Training pass ----
        (train_loss_sum, train_mae_sum, train_vxc_sum,
         train_db_errors, vxc_iter) = _train_epoch(
            model, train_loader, vxc_train_loader, optimizer,
            vxc_iter, omega, accum_iter, device, rung, dft, dispersions, local_rank,
        )
        if scheduler:
            scheduler.step()

        # ---- Validation pass ----
        (val_loss_sum, val_mae_sum, val_vxc_sum,
         val_samples_count, val_db_errors) = _validate_epoch(
            model, test_loader, vxc_test_loader, omega,
            device, rung, dft, dispersions, local_rank,
        )

        # ---- DDP sync ----
        (avg_train_loss, avg_train_mae, avg_train_vxc,
         avg_val_loss, avg_val_mae, avg_val_vxc,
         global_train_errors, global_val_errors) = _sync_metrics(
            (train_loss_sum, train_mae_sum, train_vxc_sum),
            (val_loss_sum, val_mae_sum, val_vxc_sum, val_samples_count),
            train_db_errors, val_db_errors, world_size, device,
        )

        if local_rank == 0:
            # ---- Logging ----
            train_fchem, val_fchem = _log_metrics(
                run, epoch,
                (avg_train_loss, avg_train_mae, avg_train_vxc),
                (avg_val_loss,   avg_val_mae,   avg_val_vxc),
                global_train_errors, global_val_errors,
                model, omega, train_history, val_history,
                n_train_batches=len(train_loader),
            )

            # ---- Checkpointing ----
            prev, prev_best = _save_checkpoint(
                model, epoch, batch_size, lr_train, name, omega,
                avg_train_loss, avg_val_loss, train_fchem, val_fchem,
                val_history["full_loss"], val_loss_window,
                prev, prev_best, _BEST_MODEL_DIR,
            )

            # ---- Plotting ----
            _plot_losses(
                train_history["full_loss"], val_history["full_loss"],
                train_history["fchem"],     val_history["fchem"],
                batch_size, lr_train, name, omega, _PLOT_DIR,
            )

            # ---- MLflow artifact: reaction and vxc loss plots ----
            if mlflow.active_run() is not None:
                plot_path = _plot_reaction_and_vxc_losses(
                    train_history["reaction_loss"],
                    val_history["reaction_loss"],
                    train_history["vxc"],
                    val_history["vxc"],
                )
                mlflow.log_artifact(plot_path, artifact_path="plots")
                # Clean up the temporary file
                if os.path.exists(plot_path):
                    os.remove(plot_path)

        if early_stopper.early_stop(avg_val_loss):
            if local_rank == 0:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    return train_history["full_loss"], val_history["full_loss"], prev_best


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train pcPBELMLOptimizerV2 on reaction energy and Vxc data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name", type=str, default="PBE-L_8_32",
        help="Model name. Format: 'PBE-L_<num_layers>_<h_dim>', e.g. PBE-L_8_32.",
    )
    parser.add_argument("--n_predopt",      type=int,   default=3,      help="Pre-optimization epochs.")
    parser.add_argument("--n_train",        type=int,   default=1000,   help="Main training epochs.")
    parser.add_argument("--batch_size",     type=int,   default=3,      help="Reactions per batch.")
    parser.add_argument("--dropout",        type=float, default=0.6,    help="Dropout rate in ResBlocks.")
    parser.add_argument("--omega",          type=float, default=0.0,    help="Vxc loss weight (0=Fchem only, 1=Vxc only).")
    parser.add_argument("--lr_train",       type=float, default=1e-4,   help="Learning rate for main training.")
    parser.add_argument("--lr_predopt",     type=float, default=2e-2,   help="Learning rate for pre-optimization.")
    parser.add_argument("--weight_decay",   type=float, default=1e-2,   help="AdamW weight decay.")
    parser.add_argument("--optimizer",      type=str,   default="radamw",
                        choices=["radamw", "adamw"],   help="Optimizer variant.")
    parser.add_argument("--vxc_batch_size", type=int,   default=1,      help="Batch size for Vxc DataLoader.")
    return parser


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer: torch.optim.Optimizer, n_train: int):
    """Builds the LinearLR warm-up → CosineAnnealingLR sequential scheduler."""
    warmup = LinearLR(optimizer, start_factor=_WARMUP_START_FACTOR, total_iters=_WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=n_train - _WARMUP_EPOCHS, eta_min=_MIN_LR)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[_WARMUP_EPOCHS])


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _build_dataloaders(
    data_train: dict,
    data_test: dict,
    data_vxc_train: list,
    data_vxc_val: list,
    data_all: dict,
    batch_size: int,
    vxc_batch_size: int,
    generator: torch.Generator,
) -> tuple:
    """
    Constructs and returns all DataLoaders needed for training.

    Returns:
        (train_dataloader, test_dataloader, predopt_dataloader,
         vxc_train_loader, vxc_test_loader)
    """
    train_set    = AugmentedDataset(data=data_train)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4, pin_memory=True,
        shuffle=False, sampler=train_sampler, generator=generator,
        collate_fn=collate_fn, worker_init_fn=seed_worker,
    )

    test_set    = AugmentedDataset(data=data_test)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, num_workers=4, pin_memory=True,
        shuffle=False, sampler=test_sampler, generator=generator,
        collate_fn=collate_fn, worker_init_fn=seed_worker,
    )

    predopt_set    = DatasetPredopt(data=data_all)
    predopt_sampler = DistributedSampler(predopt_set, shuffle=False)
    predopt_dataloader = torch.utils.data.DataLoader(
        predopt_set, batch_size=batch_size, num_workers=4, pin_memory=True,
        shuffle=False, sampler=predopt_sampler, generator=generator,
        collate_fn=fast_collate_fn_predopt, worker_init_fn=seed_worker,
    )

    vxc_train_set    = VxcDataset(data_vxc_train)
    vxc_train_sampler = DistributedSampler(vxc_train_set, shuffle=True)
    vxc_train_loader  = torch.utils.data.DataLoader(
        vxc_train_set, batch_size=vxc_batch_size, num_workers=2, pin_memory=True,
        sampler=vxc_train_sampler, collate_fn=vxc_collate_fn, worker_init_fn=seed_worker,
    )

    vxc_test_set    = VxcDataset(data_vxc_val)
    vxc_test_sampler = DistributedSampler(vxc_test_set, shuffle=False)
    vxc_test_loader  = torch.utils.data.DataLoader(
        vxc_test_set, batch_size=vxc_batch_size, num_workers=2, pin_memory=True,
        sampler=vxc_test_sampler, collate_fn=vxc_collate_fn, worker_init_fn=seed_worker,
    )

    return train_dataloader, test_dataloader, predopt_dataloader, vxc_train_loader, vxc_test_loader


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. DDP initialization
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2. Load data
    data, data_train, data_test, data_vxc_train, data_vxc_val = load_chk(path="checkpoints")

    # 3. Parse arguments
    args = _build_argument_parser().parse_args()
    num_layers, h_dim = map(int, args.name.split("_")[1:])
    name = args.name + "_" + str(args.dropout)  # e.g. PBE-L_8_32_0.6

    # 4. Build model
    base_model = pcPBELMLOptimizerV2(
        num_layers=num_layers, h_dim=h_dim, dropout=args.dropout, DFT="PBE"
    ).to(device)
    model = DDP(base_model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        print(FCHEM_VALIDATION)
        print(f"name={name}, n_predopt={args.n_predopt}, n_train={args.n_train}, "
              f"batch_size={args.batch_size}, dropout={args.dropout}, omega={args.omega}, "
              f"lr_train={args.lr_train}, lr_predopt={args.lr_predopt}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Number of parameters: {sum(p.numel() for p in model.module.parameters())}")

    # 5. Load dispersions
    with open("./dispersions/dispersions.pickle", "rb") as handle:
        dispersions = pickle.load(handle)

    # 6. Build dataloaders
    (train_dataloader, test_dataloader, predopt_dataloader,
     vxc_train_loader, vxc_test_loader) = _build_dataloaders(
        data_train, data_test, data_vxc_train, data_vxc_val, data,
        batch_size=args.batch_size, vxc_batch_size=args.vxc_batch_size,
        generator=g,
    )

    # 7. MLFlow logging (rank 0 only)
    run = None
    if local_rank == 0:
        load_dotenv(find_dotenv())
        enable_mlflow = os.getenv("ENABLE_MLFLOW", "true").lower() in ("true", "1", "yes")

        if enable_mlflow:
            try:
                # Configure tracking URI (defaults to ./mlruns/)
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment("piNN-DFT")

                # Start run with descriptive name
                run_name = f"{name}_omega_{args.omega:.3f}"
                mlflow.start_run(run_name=run_name)

                # Log all hyperparameters
                mlflow.log_params({
                    "name": name,
                    "num_layers": num_layers,
                    "h_dim": h_dim,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "optimizer": args.optimizer,
                    "lr_train": args.lr_train,
                    "omega": args.omega,
                    "n_predopt": args.n_predopt,
                    "n_train": args.n_train,
                    "batch_size": args.batch_size,
                    "lr_predopt": args.lr_predopt,
                    "vxc_batch_size": args.vxc_batch_size,
                })

                print(f"MLFlow tracking enabled. URI: {tracking_uri}, Experiment: piNN-DFT, Run: {run_name}")
            except Exception as e:
                print(f"Warning: MLFlow initialization failed: {e}. Logging disabled.")
        else:
            print("MLFlow logging disabled (ENABLE_MLFLOW=false).")

    # 8. Pre-optimization phase (log-scale params frozen)
    set_scales_trainable(model, trainable=False)
    predopt_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_predopt, betas=(0.9, 0.999)
    )
    predopt(
        model, nn.MSELoss(), predopt_optimizer, predopt_dataloader,
        device, n_epochs=args.n_predopt, accum_iter=1, local_rank=local_rank,
    )

    true_constants_PBE = true_constants_PBE.to(device)

    # 9. Main training phase (all params trainable)
    set_scales_trainable(model, trainable=True)
    optimizer = configure_optimizers(
        model=model, learning_rate=args.lr_train,
        optimizer_str=args.optimizer, weight_decay=args.weight_decay,
    )
    scheduler    = _build_scheduler(optimizer, args.n_train)
    early_stopper = EarlyStopper(patience=_EARLY_STOP_PATIENCE)

    train_full_loss, val_full_loss, best_model_path = train(
        model=model, optimizer=optimizer, scheduler=scheduler,
        early_stopper=early_stopper,
        train_loader=train_dataloader, test_loader=test_dataloader,
        vxc_train_loader=vxc_train_loader, vxc_test_loader=vxc_test_loader,
        run=run,
        n_epochs=args.n_train, accum_iter=1, omega=args.omega,
        local_rank=local_rank, device=device,
        rung="GGA", dft="PBE", dispersions=dispersions,
        batch_size=args.batch_size, lr_train=args.lr_train, name=name,
    )

    if local_rank == 0:
        print(f"Training complete. Best model saved at: {best_model_path}")
        if mlflow.active_run() is not None:
            mlflow.end_run()
