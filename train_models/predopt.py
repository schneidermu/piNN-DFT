import gc

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm

from dft_functionals import true_constants_PBE, true_constants_SVWN3

true_constants_SVWN = true_constants_SVWN3


class DatasetPredopt(torch.utils.data.Dataset):
    def __init__(self, data, dft):
        self.data = data
        self.dft = dft

    def __getitem__(self, i):
        self.data[i].pop("Database", None)
        if self.dft == "PBE":
            y_single = true_constants_PBE
        elif self.dft == "SVWN3":
            y_single = true_constants_SVWN
        elif self.dft == "XALPHA":
            y_single = torch.Tensor([1.05])
        return self.data[i], y_single

    def __len__(self):
        return len(self.data.keys())


def predopt(
    model,
    criterion,
    optimizer,
    train_loader,
    device,
    n_epochs=2,
    accum_iter=1,
    double_star=False,
    xalpha=False,
):
    train_loss_mse = []
    train_loss_mae = []

    for epoch in range(n_epochs):
        print("Epoch", epoch + 1)
        model.train()

        train_mse_losses_per_epoch = []
        train_mae_losses_per_epoch = []

        progress_bar = tqdm(train_loader)

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch["Grid"].to(device, non_blocking=True)
            if not double_star and not xalpha:
                y_batch = torch.tile(y_batch, [X_batch.shape[0], 1]).to(
                    device, non_blocking=True
                )[:, [0, 1, 22, 23, 24, 25]]
                predictions = model(X_batch)[:, [0, 1, 22, 23, 24, 25]]
            elif xalpha:
                predictions = model(X_batch)
                y_batch = 1.05 * torch.ones(X_batch.shape[0], 1, device=device)
            else:
                predictions = torch.stack(model(X_batch), dim=1).to(device)
                y_batch = torch.ones(X_batch.shape[0], 3, device=device)

            loss = criterion(predictions, y_batch)
            loss.backward()

            MAE = mean_absolute_error(
                predictions.cpu().detach(), y_batch.cpu().detach()
            )
            MSE = loss.item()
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            progress_bar.set_postfix(MAE=MAE, MSE=MSE)

            if ((batch_idx + 1) % accum_iter == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            del X_batch, y_batch, predictions, loss, MAE, MSE
            gc.collect()
            torch.cuda.empty_cache()

        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))

        print(f"train MSE Loss = {train_loss_mse[epoch]:.8f}")
        print(f"train MAE Loss = {train_loss_mae[epoch]:.8f}")

        if np.mean(train_mae_losses_per_epoch) < 1e-8:
            return train_loss_mse, train_loss_mae  # Early stopping

    return train_loss_mse, train_loss_mae
