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
import matplotlib.pyplot as plt

set_random_seed(41)


def sum_of_weights(dictionary):
    return np.sum(np.array(list(dictionary.values())))


def loss_function(factor_dictionary, total_database_errors):
    fchem = 0

    for db in sorted(total_database_errors):
        factor = factor_dictionary.get(db, 1)
        error = np.sqrt((np.mean(np.array(total_database_errors[db])**2)))
        fchem += error*factor
        print(f'{db}: {error*factor:.3f}')

    return fchem


def modify_training(current_bases, factor_dictionary, y_batch, reaction_energy):
    for i, db in enumerate(current_bases):
        factor = factor_dictionary.get(db, 1)
        try:
            reaction_energy[i] *= factor
            y_batch[i] *= factor
        except:
            reaction_energy *= factor
            y_batch *= factor

    return reaction_energy, y_batch


def extend_bases(X_batch, bases):
    if len(X_batch["Database"][0]) == 1:
        current_bases = [X_batch["Database"],]
    else:
        current_bases = list(X_batch["Database"])
    bases += current_bases

    return current_bases, bases


def make_total_db_errors(pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases):
    if len(pred_energies): 
        pred_energies = torch.hstack([pred_energies, reaction_energy])
        ref_energies = torch.hstack([ref_energies, y_batch])
        errors = torch.hstack([errors, reaction_energy-y_batch])
    else:
        pred_energies = reaction_energy
        ref_energies = y_batch
        errors = reaction_energy-y_batch

    for base, error in zip(current_bases, reaction_energy-y_batch):
        total_database_errors[base] = total_database_errors.get(base, [])
        total_database_errors[base].append((torch.abs(error).item()))

    return pred_energies, ref_energies, errors, total_database_errors


PBE_TRAIN_ENERGY_FACTORS = {
    "ABDE4": 1/92.10,
    "AE17": 0.00196/10,
    "DBH76": 1/17.54,
    "EA13": 1/34.96,
    "IP13": 1/257.31,
    "MGAE109": 1/508.07,
    "NCCE31": 1/4.02,
    "PA8": 1/159.48,
    "pTC13": 1/162.83,
}
PBE_VAL_ENERGY_FACTORS = {
    "ABDE4": 1/97.39,
    "AE17": 0.00217/10,
    "AE17": 0,
    "DBH76": 1/21.88,
    "EA13": 1/56.72,
    "IP13": 1/253.83,
    "MGAE109": 1/460.79,
    "NCCE31": 1/2.01,
    "PA8": 1/174.85,
    "pTC13": 1/201.57,
}

#PBE_TRAIN_FACTORS = {
#    "ABDE4": 3.681/92.10,
#    "AE17": 0.118/10,
#    "DBH76": 8.504/17.54,
#    "EA13": 2.671/34.96,
#    "IP13": 3.521/257.31,
#    "MGAE109": 15.631/508.07,
#    "NCCE31": 1.020/4.02,
#    "PA8": 1.091/159.48,
#    "pTC13": 6.169/162.83,
#}
#
#PBE_VAL_FACTORS = {
#    "ABDE4": 0.117/97.39,
#    "AE17": 0.130/10,
#    "DBH76": 9.051/21.88,
#    "EA13": 1.432/56.72,
#    "IP13": 3.291/253.83,
#    "MGAE109": 13.709/460.79,
#    "NCCE31": 1.048/2.01,
#    "PA8": 2.612/174.85,
#    "pTC13": 4.068/201.57,
#}



FCHEM_FACTORS = {
    'ABDE4': 1/4,
    'AE17': 1/17,
    'DBH76': 1/76,
    'EA13': 1/13,
    'IP13': 1/13,
    'MGAE109': 1/109/(4.73394495412844)**2,
    'NCCE31': 10**2/31,
    'PA8': 1/8,
    'pTC13': 1/13,
}
#
FCHEM_VALIDATION = {
    'MGAE109': 1/4.73394495412844,
    'NCCE31': 10,
}



#def Fchem(total_database_errors):
#    Fchem_ = 0
#    for db in total_database_errors:
#        error = np.sqrt(np.mean((np.array(total_database_errors[db]))**2))
##        if db in FCHEM_FACTORS:
##            error *= FCHEM_FACTORS[db]
#        Fchem_ += error
#
#    return Fchem_


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


def exc_loss(
    reaction, pred_constants, dft="PBE", true_constants=true_constants_PBE,
    val=False
):

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

    loss = torch.tensor(0.0, requires_grad=True).to(device)

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
    true_local_energies = [
        true_local_energies[start:stop] for start, stop in indices
    ]

    for i in range(n_molecules):
        loss += (
            criterion(predicted_local_energies[i], true_local_energies[i])
        )

    return loss * HARTREE2KCAL**2 / n_molecules


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
    train_full_loss = []
    val_full_loss = []
    test_fchem = []

    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(True)
        print(f"Epoch {epoch + 1}")
        # train
        model.train()
        progress_bar_train = tqdm(train_loader)
        train_mae_losses_per_epoch = []
        train_mse_losses_per_epoch = []
        train_exc_losses_per_epoch = []
        train_full_loss_per_epoch = []
        optimizer.zero_grad()

        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar_train):
            X_batch_grid, y_batch = X_batch["Grid"].to(device), y_batch.to(device)

            current_bases, bases = extend_bases(X_batch=X_batch, bases=bases)

            predictions = model(X_batch_grid)
            reaction_energy = calculate_reaction_energy(
                X_batch,
                predictions,
                device,
                rung=rung,
                dft=dft,
                dispersions=dispersions,
            ).to(device)

            pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(
                pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases
            )

            local_loss = exc_loss(X_batch, predictions, dft=dft)

            MAE = mae(reaction_energy, y_batch).item()

            reaction_energy, y_batch = modify_training(
                current_bases=current_bases, factor_dictionary=FCHEM_FACTORS, y_batch=y_batch, reaction_energy=reaction_energy
            )

            reaction_mse_loss = criterion(reaction_energy, y_batch)

            # Calculate total loss function
            loss = (1 - omega) / 3 * reaction_mse_loss + omega * local_loss / 16

            MSE = reaction_mse_loss.item()

            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            train_exc_losses_per_epoch.append(torch.sqrt(local_loss).item())
            train_full_loss_per_epoch.append(loss.item())
            
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
        train_full_loss.append(np.mean(train_full_loss_per_epoch))
        print(
            f"train MSE Loss = {train_loss_mse[epoch]:.8f} MAE Loss = {train_loss_mae[epoch]:.8f}"
        )
        print(
            f"train Local Energy Loss = {train_loss_exc[epoch]:.8f}"
        )

        plt.plot(range(1, len(train_full_loss)+1), train_full_loss, label='train')

        train_fchem = loss_function(factor_dictionary=FCHEM_VALIDATION, total_database_errors=total_database_errors)
        print("\nTrain Fchem", train_fchem, '\n')


        # test
        model.eval()
        progress_bar_test = tqdm(test_loader)
        test_mae_losses_per_epoch = []
        test_mse_losses_per_epoch = []
        test_exc_losses_per_epoch = []
        val_full_loss_per_epoch = []

        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}


        with torch.no_grad():
            for X_batch, y_batch in progress_bar_test:
                X_batch_grid, y_batch = X_batch["Grid"].to(device), y_batch.to(device)

                current_bases, bases = extend_bases(X_batch=X_batch, bases=bases)

                predictions = model(X_batch_grid)
                reaction_energy = calculate_reaction_energy(
                    X_batch,
                    predictions,
                    device,
                    rung=rung,
                    dft=dft,
                    dispersions=dispersions,
                ).to(device)


                pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(
                    pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases
                )

                local_loss = exc_loss(X_batch, predictions, dft=dft)

                MAE = mae(reaction_energy, y_batch).item()

                reaction_energy, y_batch = modify_training(
                    current_bases=current_bases, factor_dictionary=FCHEM_FACTORS, y_batch=y_batch, reaction_energy=reaction_energy
                )

                reaction_mse_loss = criterion(reaction_energy, y_batch)

                MSE = reaction_mse_loss.item()

                loss = (1 - omega) / 3 * reaction_mse_loss + omega * local_loss / 4

                test_mse_losses_per_epoch.append(MSE)
                test_mae_losses_per_epoch.append(MAE)
                test_exc_losses_per_epoch.append(torch.sqrt(local_loss).item())
                val_full_loss_per_epoch.append(loss.item())

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
        val_full_loss.append(np.mean(val_full_loss_per_epoch))

        print(
            f"test MSE Loss = {test_loss_mse[epoch]:.8f} MAE Loss = {test_loss_mae[epoch]:.8f}"
        )
        plt.plot(range(1, len(val_full_loss)+1), val_full_loss, label='validation')
        plt.legend()
        plt.savefig(f"training_{omega}.png")
        plt.clf()
        print(f"test Local Energy Loss = {test_loss_exc[epoch]:.8f}")

        val_fchem = loss_function(factor_dictionary=FCHEM_VALIDATION, total_database_errors=total_database_errors)
        test_fchem.append(val_fchem)

        if val_full_loss[-1] == min(val_full_loss):
            torch.save(
                model.state_dict(), f'best_models/29 june/{name}_{omega}_epoch_{epoch}_loss_{val_fchem:.2f}.pth'
            )
#        else:
#            test_fchem.pop()
        print("\nValidation Fchem", val_fchem, '\n')

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
