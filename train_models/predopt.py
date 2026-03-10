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

from utils import _fix_sigma_tot_closed_shell, _grid_to_model_input

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals import true_constants_PBE
from reaction_energy_calculation import get_local_energies


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
    vxc_loader: torch.utils.data.DataLoader = None,
    preopt_vxc_weight: float = 0.0,
    preopt_vxc_steps: int = 0,
    vxc_target_mode: str = "pbe",
    rung: str = "GGA",
    dft: str = "PBE",
) -> tuple[list, list]:
    """
    Runs the pre-optimization phase.

    Trains the model to output values close to the true PBE constants on a
    per-grid-point basis. This warm-starts the model near a physically
    meaningful solution before the main reaction-energy training phase.

    Only the 9 adaptive constants are compared: beta (0), gamma (1),
    kappa_up (22), mu_up (23), kappa_down (24), mu_down (25),
    G_NN_up (26), G_NN_down (27), G_c (28). Target values for G_NN
    are 0.0; target for G_c is 1.0.

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
    _ADAPTIVE_INDICES = [0, 1, 22, 23, 24, 25, 26, 27, 28]  # Added G_NN_up, G_NN_down, G_c
    if vxc_target_mode != "pbe":
        raise ValueError(f"Unsupported vxc_target_mode: {vxc_target_mode}")

    def _vrho_from_constants(
        constants: torch.Tensor,
        grid: torch.Tensor,
        weights: torch.Tensor,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho = grid[:, 4:6].clone().requires_grad_(True)
        sigma = grid[:, 6:9].clone()
        sigma = _fix_sigma_tot_closed_shell(sigma)
        sigma_pbe = torch.stack(
            [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]], dim=1
        )
        calc_data = get_local_energies(
            {"Densities": rho, "Gradients": sigma_pbe, "Weights": weights},
            constants,
            device,
            rung=rung,
            dft=dft,
            enhancement=None,
        )
        rho_tot = rho[:, 0] + rho[:, 1]
        e_xc = calc_data["Local_energies"] * rho_tot
        grads = torch.autograd.grad(
            outputs=e_xc,
            inputs=rho,
            grad_outputs=torch.ones_like(e_xc),
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        vrho = (grads[:, 0] + grads[:, 1]) / 2.0
        return vrho, rho_tot

    train_loss_mse: list = []
    train_loss_mae: list = []

    for epoch in range(n_epochs):
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()

        train_mse_losses_per_epoch = []
        train_mae_losses_per_epoch = []
        train_vxc_losses_per_epoch = []
        vxc_steps_used = 0

        progress_bar = tqdm(train_loader, disable=(local_rank != 0))
        vxc_iter = iter(vxc_loader) if (vxc_loader is not None and preopt_vxc_weight > 0 and preopt_vxc_steps > 0) else None

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch["Grid"].to(device, non_blocking=True)
            y_batch = torch.tile(y_batch, [X_batch.shape[0], 1]).to(
                device, non_blocking=True
            )[:, _ADAPTIVE_INDICES]

            _base = getattr(model, "module", model)
            if getattr(_base, "use_g_x", True):
                y_batch[:, [6, 7]] = y_batch[:, [6, 7]] - 1  # G_NN target: 1→0 (only when learned)

            predictions = model(X_batch)[:, _ADAPTIVE_INDICES]

            loss_constants = criterion(predictions, y_batch)
            loss = loss_constants

            if vxc_iter is not None and vxc_steps_used < preopt_vxc_steps:
                try:
                    X_vxc = next(vxc_iter)
                except StopIteration:
                    vxc_iter = iter(vxc_loader)
                    X_vxc = next(vxc_iter)

                grid_vxc = X_vxc["Grid"].to(device, non_blocking=True)
                weights_vxc = X_vxc["Weights"].to(device, non_blocking=True)

                pbe_constants = true_constants_PBE.to(device).reshape(1, -1).expand(grid_vxc.shape[0], -1)
                target_vrho, _ = _vrho_from_constants(
                    pbe_constants, grid_vxc, weights_vxc, create_graph=False
                )
                target_vrho = target_vrho.detach()

                model_input_vxc = _grid_to_model_input(grid_vxc, fix_closed_shell_sigma=True)
                pred_constants_vxc = model(model_input_vxc)
                pred_vrho, rho_tot = _vrho_from_constants(
                    pred_constants_vxc, grid_vxc, weights_vxc, create_graph=True
                )
                diff_sq = (pred_vrho - target_vrho) ** 2
                norm = torch.sum(rho_tot.detach() * weights_vxc) + 1e-10
                loss_vxc = torch.sum(rho_tot.detach() * weights_vxc * diff_sq) / norm
                loss = loss + preopt_vxc_weight * loss_vxc
                train_vxc_losses_per_epoch.append(loss_vxc.item())
                vxc_steps_used += 1

            loss.backward()

            MAE = mean_absolute_error(predictions.cpu().detach(), y_batch.cpu().detach())
            MSE = loss_constants.item()
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)

            if local_rank == 0:
                if train_vxc_losses_per_epoch:
                    progress_bar.set_postfix(
                        MAE=MAE, MSE=MSE, VXC=np.mean(train_vxc_losses_per_epoch)
                    )
                else:
                    progress_bar.set_postfix(MAE=MAE, MSE=MSE)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if local_rank == 0:
            train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
            train_loss_mae.append(np.mean(train_mae_losses_per_epoch))
            print(f"train MSE Loss = {train_loss_mse[epoch]:.8f}")
            print(f"train MAE Loss = {train_loss_mae[epoch]:.8f}")
            if train_vxc_losses_per_epoch:
                print(
                    f"train preopt Vxc Loss ({vxc_target_mode}) = "
                    f"{np.mean(train_vxc_losses_per_epoch):.8f} "
                    f"(steps={vxc_steps_used})"
                )

    return train_loss_mse, train_loss_mae
