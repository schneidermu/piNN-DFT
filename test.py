import gc
import pickle
from optparse import OptionParser

import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm

from dataset import collate_fn, collate_fn_predopt
from NN_models import MLOptimizer, pcPBEMLOptimizer
from predopt import DatasetPredopt, true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import (calculate_reaction_energy,
                                         get_local_energies)
from utils import set_random_seed

set_random_seed(41)


FCHEM_FACTORS = {
    "MGAE109": 1/4.73394495412844,
    "NCCE31": 10
}


def Fchem(total_database_errors):
    Fchem_ = 0
    for db in total_database_errors:
        error = np.sqrt(np.mean((np.array(total_database_errors[db]))**2))
        if db in FCHEM_FACTORS:
            error *= FCHEM_FACTORS[db]
        Fchem_ += error

    return Fchem_


data, data_train, data_test = load_chk(path="checkpoints")

parser = OptionParser()
parser.add_option('--Name', type=str,
                  help="Name of the functional",
                  default="PBE_8_32")
parser.add_option('--N_preopt', type=int,
                  default=3,
                  help="Number of pre-optimization epochs")
parser.add_option('--N_train', type=int,
                  default=1000,
                  help="Number of training epochs")
parser.add_option('--Batch_size', type=int,
                  default=3,
                  help="Number of reactions in a batch")
parser.add_option('--Dropout', type=float,
                  default=0.4,
                  help="Dropout rate during training")
parser.add_option('--Omega', type=float,
                  default=0.0,
                  help="Omega value in the loss function")
parser.add_option('--LR_train', type=float,
                  default=1e-4,
                  help="Omega value in the loss function")
parser.add_option('--LR_predopt', type=float,
                  default=2e-2,
                  help="Omega value in the loss function")

(Opts, args) = parser.parse_args()

name, n_predopt, n_train, batch_size, dropout, omega, lr_train, lr_predopt = (
    Opts.Name,
    Opts.N_preopt,
    Opts.N_train,
    Opts.Batch_size,
    Opts.Dropout,
    Opts.Omega,
    Opts.LR_train,
    Opts.LR_predopt
)

print('name, n_predopt, n_train, batch_size, dropout, omega, lr_train, lr_predopt')
print(name, n_predopt, n_train, batch_size, dropout, omega, lr_train, lr_predopt)

#2e-2 for predopt was the best by far (0.00088730), but can be bigger
# train lr 1e-4 seems to be fine, maybe lower?
if "PBE" in name:
    rung = "GGA"
    dft = "PBE"

elif "XALPHA" in name:
    rung = "LDA"
    dft = "XALPHA"
    nconstants = 1



num_layers, h_dim = map(int, name.split("_")[1:])
device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")


if dft == "PBE":
    model = pcPBEMLOptimizer(
        num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=dft
    ).to(device)
elif dft == "XALPHA":
    model = MLOptimizer(
        num_layers, h_dim, nconstants, dropout, dft
    ).to(device)

# Load dispersion corrections.
with open("./dispersions/dispersions.pickle", "rb") as handle:
    dispersions = pickle.load(handle)

# Describe custom pytorch Dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
#        self.data[i].pop("Database", None)
        return self.data[i], self.data[i]["Energy"]

    def __len__(self):
        return len(self.data.keys())

# Load train, test and pre-optimization dataloaders.
train_set = Dataset(data=data_train)
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    collate_fn=collate_fn,
)
test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn,
)
train_predopt_set = DatasetPredopt(data=data, dft=dft)
train_predopt_dataloader = torch.utils.data.DataLoader(
    train_predopt_set,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn_predopt,
)

criterion = nn.MSELoss()


name += "_" + str(dropout)

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr_predopt, betas=(0.9, 0.999), weight_decay=0.01
)

from importlib import reload

import predopt

reload(predopt)
import utils

reload(utils)
import dataset

reload(dataset)

from predopt import predopt

# Pre-optimize the model.
train_loss_mse, train_loss_mae = predopt(
    model,
    criterion,
    optimizer,
    train_predopt_dataloader,
    device,
    n_epochs=n_predopt,
    accum_iter=1,
)


torch.cuda.empty_cache()


from tqdm.notebook import tqdm

mae = nn.L1Loss()


def exc_loss(reaction, pred_constants, dft="PBE", true_constants=true_constants_PBE):

    HARTREE2KCAL = 627.5095

    # Turn backsplit indices into slices.
    backsplit_ind = reaction["backsplit_ind"].to(torch.int32)
    indices = list(
        zip(
            torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)),
            backsplit_ind,
        )
    )
    n_molecules = len(indices)

    # Initialize loss.
    loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Calculate predicted local energies.
    predicted_local_energies = get_local_energies(
        reaction, pred_constants, device, rung=rung, dft=dft
    )["Local_energies"]

    # Split them into systems.
    predicted_local_energies = [
        predicted_local_energies[start:stop] for start, stop in indices
    ]

    # Calculate local PBE energies.
    true_local_energies = get_local_energies(
        reaction, true_constants.to(device), device, rung="GGA", dft="PBE"
    )["Local_energies"]

    # Split them into systems.
    true_local_energies = [true_local_energies[start:stop] for start, stop in indices]

    # Calculate local energy loss.
    for i in range(n_molecules):
        loss += (
            1
            / len(predicted_local_energies[i])
            * (
                torch.sum((predicted_local_energies[i] - true_local_energies[i]) ** 2)
            )
        )

    return loss * HARTREE2KCAL / n_molecules

true_constants_PBE = true_constants_PBE.to(device)

def train(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    n_epochs=25,
    accum_iter=1,
    verbose=False,
    omega=0.067,
):
    torch.set_printoptions(precision=2)
    train_loss_mae = []
    train_loss_mse = []
    train_loss_exc = []
    test_loss_mae = []
    test_loss_mse = []
    test_loss_exc = []

    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(True)
        print(f"Epoch {epoch + 1}")
        # train
        model.train()
        progress_bar_train = tqdm(train_loader)
        train_mae_losses_per_epoch = []
        train_mse_losses_per_epoch = []
        train_exc_losses_per_epoch = []
        optimizer.zero_grad()


        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}


        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar_train):
            X_batch_grid, y_batch = X_batch["Grid"].to(device), y_batch.to(device)

            if len(X_batch["Database"]) == 1:
                current_bases = [X_batch["Database"], ]
            else:
                current_bases = [x for x in X_batch["Database"]]

            bases += current_bases

            predictions = model(X_batch_grid)
            reaction_energy = calculate_reaction_energy(
                X_batch,
                predictions,
                device,
                rung=rung,
                dft=dft,
                dispersions=dispersions,
            ).to(device)

            if len(pred_energies): 
                pred_energies = torch.hstack([pred_energies, reaction_energy])
                ref_energies = torch.hstack([ref_energies, y_batch])
                errors = torch.hstack([errors, pred_energies-ref_energies])
            else:
                pred_energies = reaction_energy
                ref_energies = y_batch
                errors = reaction_energy-y_batch
            
            for base, error in zip(bases, errors):
                total_database_errors[base] = total_database_errors.get(base, [])
                total_database_errors[base].append(abs(error.item()))

            local_loss = exc_loss(X_batch, predictions, dft=dft)

            MAE = mae(reaction_energy, y_batch).item()

            for i, db in enumerate(current_bases):
                factor = FCHEM_FACTORS.get(db, 1)
                try:
                    reaction_energy[i] *= factor
                    y_batch[i] *= factor
                except:
                    reaction_energy *= factor
                    y_batch *= factor

            reaction_mse_loss = criterion(reaction_energy, y_batch)

            # Calculate total loss function
            loss = (1 - omega) / 4 * torch.sqrt(reaction_mse_loss) + omega * local_loss * 10

            MSE = reaction_mse_loss.item()

            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            train_exc_losses_per_epoch.append(torch.sqrt(local_loss).item())
            progress_bar_train.set_postfix(MSE=MSE, MAE=MAE)

            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            del X_batch, X_batch_grid, y_batch, reaction_energy
            gc.collect()
            torch.cuda.empty_cache()

        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))
        train_loss_exc.append(np.mean(train_exc_losses_per_epoch))
        print(
            f"train Weighted MSE Loss = {train_loss_mse[epoch]:.8f} MAE Loss = {train_loss_mae[epoch]:.8f}"
        )
        print(f"train Local Energy Loss = {train_loss_exc[epoch]:.8f}")

        errors = []
        for db in sorted(total_database_errors):
            errors.append(np.mean(np.array(total_database_errors[db])))
            print(f'{db}: {np.mean(np.array(total_database_errors[db])):.3f}')

        print()
        print("Train Fchem", Fchem(total_database_errors))
        print()


        # test
        model.eval()
        progress_bar_test = tqdm(test_loader)
        test_mae_losses_per_epoch = []
        test_mse_losses_per_epoch = []
        test_exc_losses_per_epoch = []

        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}


        with torch.no_grad():
            for X_batch, y_batch in progress_bar_test:
                X_batch_grid, y_batch = X_batch["Grid"].to(device), y_batch.to(device)

                if len(X_batch["Database"]) == 1:
                    bases += [X_batch["Database"], ]
                else:
                    bases += [x for x in X_batch["Database"]]

                predictions = model(X_batch_grid)
                reaction_energy = calculate_reaction_energy(
                    X_batch,
                    predictions,
                    device,
                    rung=rung,
                    dft=dft,
                    dispersions=dispersions,
                ).to(device)

                if len(pred_energies): 
                    pred_energies = torch.hstack([pred_energies, reaction_energy])
                    ref_energies = torch.hstack([ref_energies, y_batch])
                    errors = torch.hstack([errors, pred_energies-ref_energies])
                else:
                    pred_energies = reaction_energy
                    ref_energies = y_batch
                    errors = reaction_energy-y_batch
            
                for base, error in zip(bases, errors):
                    total_database_errors[base] = total_database_errors.get(base, [])
                    total_database_errors[base].append(abs(error.item()))

                local_loss = exc_loss(X_batch, predictions, dft=dft)

                loss = criterion(reaction_energy, y_batch)
                MSE = loss.item()
                MAE = mae(reaction_energy, y_batch).item()
                test_mse_losses_per_epoch.append(MSE)
                test_mae_losses_per_epoch.append(MAE)
                test_exc_losses_per_epoch.append(torch.sqrt(local_loss).item())

                progress_bar_test.set_postfix(MSE=MSE, MAE=MAE)
                del (
                    X_batch,
                    X_batch_grid,
                    y_batch,
                    reaction_energy,
                    loss,
                    MAE,
                    MSE,
                )
                gc.collect()
                torch.cuda.empty_cache()

        test_loss_mse.append(np.mean(test_mse_losses_per_epoch))
        test_loss_mae.append(np.mean(test_mae_losses_per_epoch))
        test_loss_exc.append(np.mean(test_exc_losses_per_epoch))

        print(
            f"test Unweighted MSE Loss = {test_loss_mse[epoch]:.8f} MAE Loss = {test_loss_mae[epoch]:.8f}"
        )
        print(f"test Local Energy Loss = {test_loss_exc[epoch]:.8f}")

        errors = []
        for db in sorted(total_database_errors):
            errors.append(np.mean(np.array(total_database_errors[db])))
            print(f'{db}: {np.mean(np.array(total_database_errors[db])):.3f}')

        print()
        print("Validation Fchem", Fchem(total_database_errors))
        print()

    return train_loss_mae, test_loss_mae


from importlib import reload

import NN_models
import dft_functionals.PBE as PBE
import reaction_energy_calculation
import utils

reload(NN_models)
reload(utils)
reload(reaction_energy_calculation)
reload(PBE)

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr_train, betas=(0.9, 0.999), weight_decay=0.01
)

N_EPOCHS = n_train
ACCUM_ITER = 1
VERBOSE = False
train_loss_mae, test_loss_mae = train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    test_dataloader,
    n_epochs=N_EPOCHS,
    accum_iter=ACCUM_ITER,
    omega=omega,
    verbose=VERBOSE
)
torch.save(model.state_dict(), f'best_models/{name}.pth')
