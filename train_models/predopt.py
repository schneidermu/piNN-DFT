"""
Pre-optimization phase: warm-starts pcPBELMLOptimizerV2 to output values
close to the true PBE constants before the main reaction-energy training begins.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals import true_constants_PBE


class DatasetPredopt(torch.utils.data.Dataset):
    """
    Dataset for the pre-optimization phase.

    Returns (reaction_dict, true_constants_PBE) pairs.  The target constants
    are the same for every sample; the model is trained to reproduce them from
    any local density input before fine-tuning on reaction energies.

    Args:
        data: Dictionary mapping reaction indices to reaction data dicts.
    """

    def __init__(self, data: dict) -> None:
        self.data = data

    def __getitem__(self, i: int):
        reaction = dict(self.data[i])   # shallow copy to avoid mutating the cache
        reaction.pop("Database", None)
        return reaction, true_constants_PBE

    def __len__(self) -> int:
        return len(self.data.keys())


def predopt(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs: int = 2,
    accum_iter: int = 1,
    local_rank: int = 0,
) -> tuple[list, list]:
    """
    Runs the pre-optimization phase.

    Trains the model to output values close to the true PBE constants on a
    per-grid-point basis. This warm-starts the model near a physically
    meaningful solution before the main reaction-energy training phase.

    Only the 6 adaptive constants are compared: beta (0), gamma (1),
    kappa_up (22), mu_up (23), kappa_down (24), mu_down (25).

    Args:
        model: DDP-wrapped pcPBELMLOptimizerV2.
        criterion: Loss function (typically MSELoss).
        optimizer: Optimizer (typically Adam with lr_predopt).
        train_loader: DataLoader for pre-optimization data.
        device: Target compute device.
        n_epochs: Number of pre-optimization epochs.
        accum_iter: Gradient accumulation steps (reserved; currently 1).
        local_rank: DDP local rank. Logging and progress bars are gated to rank 0.

    Returns:
        (train_loss_mse, train_loss_mae): Per-epoch MSE and MAE losses (rank 0 only;
        other ranks return empty lists).
    """
    _ADAPTIVE_INDICES = [0, 1, 22, 23, 24, 25]

    train_loss_mse: list = []
    train_loss_mae: list = []

    for epoch in range(n_epochs):
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()

        train_mse_losses_per_epoch = []
        train_mae_losses_per_epoch = []

        progress_bar = tqdm(train_loader, disable=(local_rank != 0))

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch["Grid"].to(device, non_blocking=True)
            y_batch = torch.tile(y_batch, [X_batch.shape[0], 1]).to(
                device, non_blocking=True
            )[:, _ADAPTIVE_INDICES]

            predictions = model(X_batch)[:, _ADAPTIVE_INDICES]

            loss = criterion(predictions, y_batch)
            loss.backward()

            MAE = mean_absolute_error(predictions.cpu().detach(), y_batch.cpu().detach())
            MSE = loss.item()
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)

            if local_rank == 0:
                progress_bar.set_postfix(MAE=MAE, MSE=MSE)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if local_rank == 0:
            train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
            train_loss_mae.append(np.mean(train_mae_losses_per_epoch))
            print(f"train MSE Loss = {train_loss_mse[epoch]:.8f}")
            print(f"train MAE Loss = {train_loss_mae[epoch]:.8f}")

    return train_loss_mse, train_loss_mae
