import gc
import pickle
from optparse import OptionParser

import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm
import os

from dataset import collate_fn, collate_fn_predopt
from NN_models import MLOptimizer, pcPBEMLOptimizer
from predopt import DatasetPredopt, true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import (calculate_reaction_energy,
                                         get_local_energies)
from utils import set_random_seed
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


set_random_seed(41)

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
else:
    device = torch.device("cpu")


FCHEM_VALIDATION = {
    'MGAE109': 1/4.73394495412844,
    'NCCE31': 10,
}

TRAIN_WEIGHTS = {
    "ABDE4": 1/93.42/4,
    "AE17": 1/93753.82/17,
    "DBH76": 1/18.40/76,
    "EA13": 1/38.31/13,
    "IP13": 1/253.83/13,
    "MGAE109": 1/498.53/109,
    "NCCE31": 1/3.63/31,
    "PA8": 1/163.33/8,
    "pTC13": 1/168.79/13,
}

mean_train_weight = np.mean(np.array(list(TRAIN_WEIGHTS.values())))

PBE_TRAIN_ERRORS = {
    "ABDE4": 2.347,
    "AE17": 54.272,
    "DBH76": 10.120,
    "EA13": 3.112,
    "IP13": 4.964,
    "MGAE109": 4.041,
    "NCCE31": 16.937,
    "PA8": 1.688,
    "pTC13": 5.784,
}

PBE_VALIDATION_ERRORS = {
    "ABDE4": 5.689,
    "AE17": 66.095,
    "DBH76": 8.904,
    "EA13": 2.451,
    "IP13": 3.358,
    "MGAE109": 3.974,
    "NCCE31": 15.767,
    "PA8": 1.789,
    "pTC13": 10.985,
}

    
# Describe custom pytorch Dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i], self.data[i]["Energy"]

    def __len__(self):
        return len(self.data.keys())


def loss_function(factor_dictionary: dict, total_database_errors: dict, val=False):
    '''
    Function for calculation of sum of RMSEs by databases

    Args:
        factor_dictionary: dict used to add weights for databses if needed
        total_database_errors: dict with db names and corresponding arrays of errors
    
    Returns:
        fchem: weighted sum of RMSEs by database
    '''
    fchem = 0

    for db in sorted(total_database_errors):
        factor = factor_dictionary.get(db, 1)
        error = np.sqrt((np.mean(np.array(total_database_errors[db])**2)))
        fchem += error*factor
        if val:
            err_dict = PBE_VALIDATION_ERRORS
        else:
            err_dict = PBE_TRAIN_ERRORS
        print(f'{db}: {error*factor:6.3f}, {100*(error*factor-err_dict.get(db))/err_dict.get(db):6.3f}%')

    return fchem


def batch_fchem(current_bases: list, reaction_energy: torch.tensor, y_batch: torch.tensor):
    '''
    Function for calculation of the root of sum of weighted MSEs by databases

    Args:
        current_bases: list of databases
        reaction_energy: tensor with predicted energies
        y_batch: tensor with reference energies
    
    Returns:
        fchem: root of weighted sum of MSEs by database
    '''
    mse = nn.MSELoss()
    err_dict = dict()
    errors = []


    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict[database] = err_dict.get(database, [[], []])
        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)
    for database in err_dict:
        if database == 'AE17':
            factor = np.abs((1/err_dict[database][1][-1]).item())/mean_train_weight
        else:
            factor = TRAIN_WEIGHTS.get(database, 1)/mean_train_weight
        errors.append(factor*mse(torch.stack(err_dict[database][0]), torch.stack(err_dict[database][1])))

    fchem = torch.sqrt(1e-20 + torch.sum(torch.stack(errors)))/len(err_dict)
    return fchem


def extend_bases(X_batch, bases) -> tuple[list]:
    '''
    Function for logging the current databases and the overall databases during training

    Args:
        X_batch: current batch
        bases: overall databases before including X_batch
    
    Returns:
        current_bases: bases in the current batch
        bases: overall bases
    '''

    if len(X_batch["Database"][0]) == 1:
        current_bases = [X_batch["Database"],]
    else:
        current_bases = list(X_batch["Database"])
    bases += current_bases

    return current_bases, bases


def make_total_db_errors(pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases):
    '''
    Function for making a dictionary of the format: {database name: list of absolute errors}
    '''
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



def exc_loss(
    reaction, pred_constants, predicted_local_energies,  dft="PBE", true_constants=true_constants_PBE,
    val=False
):
    '''
    Function that calculates local energy loss compared to PBE functional
    '''

    criterion = nn.MSELoss()

    HARTREE2KCAL = 627.5095

    backsplit_ind = reaction["backsplit_ind"].to(torch.int32) # Turn backsplit indices into slices.
    indices = list(
        zip(
            torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)),
            backsplit_ind,
        )
    )
    n_molecules = len(indices)

    loss = torch.zeros(1, requires_grad=True).to(device)

    predicted_local_energies = [
        predicted_local_energies[start:stop] for start, stop in indices
    ] # Split them into systems

    true_local_energies = get_local_energies(
        reaction, true_constants.to(device), device, rung="GGA", dft="PBE"
    )["Local_energies"] # Calculate local PBE energies.

    true_local_energies = [
        true_local_energies[start:stop] for start, stop in indices
    ] # Split them into systems.

    for i in range(n_molecules):
        loss += (
            torch.sqrt(criterion(predicted_local_energies[i], true_local_energies[i]) + 1e-20)/np.sqrt(len(predicted_local_energies[i]))
        )

    del (
        true_local_energies,
        predicted_local_energies
    )
    return loss * HARTREE2KCAL / n_molecules


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
    train_fchem_loss = []

    prev = None

    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(False)
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
            reaction_energy, local_energies = calculate_reaction_energy(
                X_batch,
                predictions,
                device,
                rung=rung,
                dft=dft,
                dispersions=dispersions,
            )
            reaction_energy.to(device)
            local_energies.to(device)

            pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(
                pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases
            )

            local_loss = exc_loss(X_batch, predictions, local_energies, dft=dft)

            MAE = mae(reaction_energy, y_batch).item()

            batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)

            # Calculate total loss function
            loss = (1 - omega) / 2  * batch_fchem_loss + omega * local_loss * 50
            loss.backward()

            MSE = criterion(reaction_energy, y_batch).item()
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            train_exc_losses_per_epoch.append(local_loss.item())
            train_full_loss_per_epoch.append(loss.item())
            
            progress_bar_train.set_postfix(MSE=MSE, MAE=MAE)

            if ((batch_idx + 1) % accum_iter == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            del X_batch, X_batch_grid, y_batch, reaction_energy, local_energies, local_loss, MSE, MAE, loss, predictions
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

        train_fchem = loss_function(factor_dictionary=FCHEM_VALIDATION, total_database_errors=total_database_errors)
        print("\nTrain Fchem", train_fchem, '\n')
        train_fchem_loss.append(train_fchem)


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
                reaction_energy, local_energies = calculate_reaction_energy(
                    X_batch,
                    predictions,
                    device,
                    rung=rung,
                    dft=dft,
                    dispersions=dispersions,
                )
                reaction_energy.to(device)
                local_energies.to(device)


                pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(
                    pred_energies, reaction_energy, errors, ref_energies, y_batch, total_database_errors, current_bases
                )

                local_loss = exc_loss(X_batch, predictions, local_energies, dft=dft)

                MAE = mae(reaction_energy, y_batch).item()

                reaction_mse_loss = criterion(reaction_energy, y_batch)
                batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)

                MSE = reaction_mse_loss.item()

                loss = (1 - omega) / 2 * batch_fchem_loss + omega * local_loss * 50

                test_mse_losses_per_epoch.append(MSE)
                test_mae_losses_per_epoch.append(MAE)
                test_exc_losses_per_epoch.append(local_loss.item())
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
                    local_energies,
                    local_loss,
                    predictions
                )
                gc.collect()
                torch.cuda.empty_cache()

        test_loss_mse.append(np.mean(test_mse_losses_per_epoch))
        test_loss_mae.append(np.mean(test_mae_losses_per_epoch))
        test_loss_exc.append(np.mean(test_exc_losses_per_epoch))
        val_full_loss.append(np.mean(val_full_loss_per_epoch))
        if scheduler:
            scheduler.step()
        print(
            f"test MSE Loss = {test_loss_mse[epoch]:.8f} MAE Loss = {test_loss_mae[epoch]:.8f}"
        )
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[3, 6], sharex=True)
        ax[0].plot(range(1, len(train_full_loss)+1), train_full_loss, label='Train Loss')
        ax[0].plot(range(1, len(val_full_loss)+1), val_full_loss, label='Validation Loss')
        print(f"test Local Energy Loss = {test_loss_exc[epoch]:.8f}")

        val_fchem = loss_function(factor_dictionary=FCHEM_VALIDATION, total_database_errors=total_database_errors, val=True)
        test_fchem.append(val_fchem)

        if val_fchem <= min(test_fchem):
            if prev:
                os.remove(prev)
            prev = f'best_models/RELATIVE_AE17/NEW_FORMULA/lr_{lr_train}_{name}_{omega}_epoch_{epoch+1}_loss_{val_fchem:.2f}_{test_loss_exc[-1]:.4f}.pth'
            torch.save(
                model.state_dict(), prev
            )
        print("\nValidation Fchem", val_fchem, '\n')
        ax[1].plot(range(1, len(val_full_loss)+1), train_fchem_loss, label='Train Fchem')
        ax[1].plot(range(1, len(val_full_loss)+1), test_fchem, label='Validation Fchem')
        ax[0].legend()
        ax[1].legend()
        plt.savefig(f"./batch_fchem/RELATIVE_AE17/NEW_FORMULA/lr_{lr_train}_{name}_{omega}.png")
        plt.clf()
        plt.close()

        del (
            val_fchem,
            train_fchem,
            test_mse_losses_per_epoch,
            test_mae_losses_per_epoch,
            val_full_loss_per_epoch
        )
        gc.collect()
        torch.cuda.empty_cache()

    return train_loss_mae, test_loss_mae



if __name__ == "__main__":

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
                      default=0.6,
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

    # Load train, test and pre-optimization dataloaders.
    train_set = Dataset(data=data_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_set = Dataset(data=data_test)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    train_predopt_set = DatasetPredopt(data=data, dft=dft)
    train_predopt_dataloader = torch.utils.data.DataLoader(
        train_predopt_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn_predopt,
    )

    name += "_" + str(dropout)


    from importlib import reload

    import predopt

    reload(predopt)
    import utils

    reload(utils)
    import dataset

    reload(dataset)

    from predopt import predopt



    from tqdm.notebook import tqdm

    mae = nn.L1Loss()


    from importlib import reload

    import NN_models
    import dft_functionals.PBE as PBE
    import reaction_energy_calculation
    import utils

    reload(NN_models)
    reload(utils)
    reload(reaction_energy_calculation)
    reload(PBE)


    criterion = nn.MSELoss()


    name += "_" + str(dropout)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_predopt, betas=(0.9, 0.999), weight_decay=0.01
    )

    train_loss_mse, train_loss_mae = predopt(
        model,
        criterion,
        optimizer,
        train_predopt_dataloader,
        device,
        n_epochs=n_predopt,
        accum_iter=1,
    )


    true_constants_PBE = true_constants_PBE.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_train, betas=(0.9, 0.999), weight_decay=0.01
    )

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

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
