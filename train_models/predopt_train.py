import collections
import gc
import os
import pickle
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import collate_fn, collate_fn_predopt
from NN_models import (MLOptimizer, pcPBEdoublestar, pcPBELMLOptimizer,
                       pcPBEMLOptimizer, pcPBEstar)
from predopt import DatasetPredopt, predopt, true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import calculate_reaction_energy
from utils import configure_optimizers, seed_worker, set_random_seed

set_random_seed(41)
g = torch.Generator()
g.manual_seed(41)

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
else:
    device = torch.device("cpu")


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

mean_freq_weight = np.mean(
    np.array([FCHEM_VALIDATION[db] * FREQ_WEIGHTS[db] for db in FCHEM_VALIDATION])
)

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


# Describe custom pytorch Dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i], self.data[i]["Energy"]

    def __len__(self):
        return len(self.data.keys())


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def gather_reaction_names(local_name, device, world_size, local_rank):
    """
    Gathers reaction names from all distributed processes to the main process (rank 0).
    """

    MAX_LEN = 256 
    
    encoded = torch.tensor([ord(c) for c in local_name[:MAX_LEN]], dtype=torch.int64, device=device)
    
    padded = torch.zeros(MAX_LEN, dtype=torch.int64, device=device)
    padded[:len(encoded)] = encoded
    
    if local_rank == 0:
        gather_list = [torch.zeros_like(padded) for _ in range(world_size)]
    else:
        gather_list = None
        

    dist.gather(tensor=padded, gather_list=gather_list, dst=0)
    
    if local_rank == 0:
        decoded_names = []
        for tensor in gather_list:
            name = "".join([chr(c) for c in tensor if c != 0])
            decoded_names.append(name)
        return " | ".join(decoded_names)
    else:
        return None


def loss_function(factor_dictionary: dict, total_database_errors: dict, val=False):
    """
    Function for calculation of sum of RMSEs by databases

    Args:
        factor_dictionary: dict used to add weights for databses if needed
        total_database_errors: dict with db names and corresponding arrays of errors

    Returns:
        fchem: weighted sum of RMSEs by database
    """
    fchem = 0

    for db in sorted(total_database_errors):
        factor = FCHEM_VALIDATION.get(db, 1)
        error = np.sqrt((np.mean(np.array(total_database_errors[db]) ** 2)))
        fchem += error * factor
        if val:
            err_dict = PBE_VALIDATION_ERRORS
        else:
            err_dict = PBE_TRAIN_ERRORS
        print(
            f"{db}: {error:6.3f}, {100*(error-err_dict.get(db))/err_dict.get(db):6.3f}%"
        )

    return fchem


def batch_fchem(
    current_bases: list, reaction_energy: torch.tensor, y_batch: torch.tensor
):
    """
    Function for calculation of the root of sum of weighted MSEs by databases

    Args:
        current_bases: list of databases
        reaction_energy: tensor with predicted energies
        y_batch: tensor with reference energies

    Returns:
        fchem: root of weighted sum of MSEs by database
    """
    err_dict = dict()
    mse = nn.MSELoss()

    fchem = []

    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict[database] = err_dict.get(database, [[], []])

        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)

    for database in err_dict:

        db_predictions = torch.stack(err_dict[database][0])
        db_ref = torch.stack(err_dict[database][1])

        factor = (
            FCHEM_VALIDATION.get(database, 1)
            * FREQ_WEIGHTS.get(database, 1)
            / mean_freq_weight
        )

        fchem.append(factor * torch.sqrt(1e-20 + mse(db_predictions, db_ref)))

    batch_error = torch.sum(torch.stack(fchem)) / len(fchem)

    return batch_error


def extend_bases(X_batch, bases) -> tuple[list]:
    """
    Function for logging the current databases and the overall databases during training

    Args:
        X_batch: current batch
        bases: overall databases before including X_batch

    Returns:
        current_bases: bases in the current batch
        bases: overall bases
    """

    if len(X_batch["Database"][0]) == 1:
        current_bases = [
            X_batch["Database"],
        ]
    else:
        current_bases = list(X_batch["Database"])
    bases += current_bases

    return current_bases, bases


def make_total_db_errors(
    pred_energies,
    reaction_energy,
    errors,
    ref_energies,
    y_batch,
    total_database_errors,
    current_bases,
):
    """
    Function for making a dictionary of the format: {database name: list of absolute errors}
    """
    if len(pred_energies):
        pred_energies = torch.hstack([pred_energies, reaction_energy])
        ref_energies = torch.hstack([ref_energies, y_batch])
        errors = torch.hstack([errors, reaction_energy - y_batch])
    else:
        pred_energies = reaction_energy
        ref_energies = y_batch
        errors = reaction_energy - y_batch

    for base, error in zip(current_bases, reaction_energy - y_batch):
        total_database_errors[base] = total_database_errors.get(base, [])
        total_database_errors[base].append((torch.abs(error).item()))

    return pred_energies, ref_energies, errors, total_database_errors


def exc_loss(
    reaction,
    pred_constants,
    predicted_local_energies,
    dft="PBE",
    true_constants=true_constants_PBE,
    val=False,
):
    """
    Function that calculates local energy loss compared to PBE functional
    """

    criterion = nn.MSELoss()

    HARTREE2KCAL = 627.5095

    backsplit_ind = reaction["backsplit_ind"].to(
        torch.int32
    )  # Turn backsplit indices into slices.
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
    ]  # Split them into systems

    true_local_energies_full = reaction["PBE_local_energies"].to(device, non_blocking=True)

    true_local_energies = [
        true_local_energies_full[start:stop] for start, stop in indices
    ]

    for i in range(n_molecules):
        loss += torch.sqrt(
            criterion(predicted_local_energies[i], true_local_energies[i]) + 1e-20
        ) / np.sqrt(len(predicted_local_energies[i]))

    del (true_local_energies, predicted_local_energies)
    return loss * HARTREE2KCAL / n_molecules


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    train_loader,
    test_loader,
    n_epochs=25,
    accum_iter=1,
    verbose=False,
    omega=0.067,
    lambda_grad=0,
    smoothing_window=10,
    local_rank=0
):
    torch.set_printoptions(precision=2)
    train_loss_mae, train_loss_mse, train_loss_exc, train_full_loss, train_fchem_loss = [], [], [], [], []
    val_full_loss, test_loss_mae, test_loss_mse, test_loss_exc, test_fchem = [], [], [], [], []

    prev, prev_best = None, None
    val_loss_window = collections.deque(maxlen=smoothing_window)
    test_fchem_window = collections.deque(maxlen=smoothing_window)

    if local_rank == 0:
        best_model_dir = "best_models/"
        plot_dir = "./batch_fchem/"
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

    scaler = torch.amp.GradScaler()
    world_size = dist.get_world_size()

    for epoch in range(n_epochs):

        model.train()
        train_loader.sampler.set_epoch(epoch)
        progress_bar_train = tqdm(
            train_loader,
            disable=(local_rank != 0),
            mininterval=2.0, 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        epoch_train_total_database_errors = collections.defaultdict(list)

        epoch_train_loss_sum = 0.0
        epoch_train_mae_sum = 0.0
        epoch_train_exc_sum = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar_train):
            X_batch_grid, y_batch = X_batch["Grid"].to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            X_batch_grid.requires_grad_(True)
            current_bases, _ = extend_bases(X_batch=X_batch, bases=[])

            with torch.amp.autocast(device_type="cuda"):
                predictions = model(X_batch_grid)

                if "STARSTAR" in name:
                    reaction_energy, local_energies = calculate_reaction_energy(
                        X_batch,
                        torch.ones(X_batch_grid.shape[0], 26).to(device) * true_constants_PBE,
                        device, rung=rung, dft=dft, dispersions=dispersions,
                        enhancement=torch.stack(predictions, dim=1),
                    )
                else:
                    reaction_energy, local_energies = calculate_reaction_energy(
                        X_batch, predictions, device, rung=rung, dft=dft, dispersions=dispersions
                    )

                local_loss = exc_loss(X_batch, predictions, local_energies, dft=dft)
                batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)
                loss = (1 - omega) * batch_fchem_loss + omega * local_loss * 100

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % accum_iter == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            MAE = mae(reaction_energy, y_batch).item()
            epoch_train_loss_sum += loss.item()
            epoch_train_mae_sum += MAE
            epoch_train_exc_sum += local_loss.item()
            
            _, _, _, epoch_train_total_database_errors = make_total_db_errors(
                [], reaction_energy, [], [], y_batch, epoch_train_total_database_errors, current_bases
            )

            if local_rank == 0:
                try:
                    component_names = [str(c) for c in X_batch['Components']]
                    local_reaction_name = ", ".join(component_names)
                except (IndexError, TypeError):
                    local_reaction_name = "loading..."
                
                all_reaction_names = gather_reaction_names(local_reaction_name, device, world_size, local_rank)

            else:
                try:
                    component_names = [str(c) for c in X_batch['Components']]
                    local_reaction_name = ", ".join(component_names)
                except (IndexError, TypeError):
                    local_reaction_name = "loading..."
                gather_reaction_names(local_reaction_name, device, world_size, local_rank)
        
        model.eval()
        progress_bar_test = tqdm(
            test_loader,
            disable=(local_rank != 0),
            mininterval=2.0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_exc_sum = 0.0
        val_samples_count = 0
        epoch_val_total_database_errors = collections.defaultdict(list)

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
            for X_batch, y_batch in progress_bar_test:
                X_batch_grid, y_batch = X_batch["Grid"].to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                current_bases, _ = extend_bases(X_batch=X_batch, bases=[])

                predictions = model(X_batch_grid)

                if "STARSTAR" in name:
                    reaction_energy, local_energies = calculate_reaction_energy(
                        X_batch,
                        torch.ones(X_batch_grid.shape[0], 26).to(device) * true_constants_PBE,
                        device, rung=rung, dft=dft, dispersions=dispersions,
                        enhancement=torch.stack(predictions, dim=1),
                    )
                else:
                    reaction_energy, local_energies = calculate_reaction_energy(
                        X_batch, predictions, device, rung=rung, dft=dft, dispersions=dispersions
                    )

                local_loss = exc_loss(X_batch, predictions, local_energies, dft=dft)
                batch_fchem_loss = batch_fchem(current_bases, reaction_energy, y_batch)
                loss = (1 - omega) * batch_fchem_loss + omega * local_loss * 100
                MAE = mae(reaction_energy, y_batch).item()

                batch_size = y_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_mae_sum += MAE * batch_size
                val_exc_sum += local_loss.item() * batch_size
                val_samples_count += batch_size
                
                _, _, _, epoch_val_total_database_errors = make_total_db_errors(
                    [], reaction_energy, [], [], y_batch, epoch_val_total_database_errors, current_bases
                )

                if local_rank == 0:
                    try:
                        component_names = [str(c) for c in X_batch['Components']]
                        local_reaction_name = ", ".join(component_names)
                    except (IndexError, TypeError):
                        local_reaction_name = "loading..."

                    all_reaction_names = gather_reaction_names(local_reaction_name, device, world_size, local_rank)
                    
                else:
                    try:
                        component_names = [str(c) for c in X_batch['Components']]
                        local_reaction_name = ", ".join(component_names)
                    except (IndexError, TypeError):
                        local_reaction_name = "loading..."
                    gather_reaction_names(local_reaction_name, device, world_size, local_rank)

        
        metrics_to_sync = torch.tensor([val_loss_sum, val_mae_sum, val_exc_sum, val_samples_count], device=device)
        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)

        gathered_train_errors_list = [None] * world_size
        gathered_val_errors_list = [None] * world_size
        
        dist.all_gather_object(gathered_train_errors_list, epoch_train_total_database_errors)
        dist.all_gather_object(gathered_val_errors_list, epoch_val_total_database_errors)

        if local_rank == 0:
            global_train_errors = collections.defaultdict(list)
            for local_dict in gathered_train_errors_list:
                for db_name, errors in local_dict.items():
                    global_train_errors[db_name].extend(errors)

            global_val_errors = collections.defaultdict(list)
            for local_dict in gathered_val_errors_list:
                for db_name, errors in local_dict.items():
                    global_val_errors[db_name].extend(errors)
            
            total_val_loss, total_val_mae, total_val_exc, total_val_samples = metrics_to_sync.tolist()

            avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
            avg_val_mae = total_val_mae / total_val_samples if total_val_samples > 0 else 0
            avg_val_exc = total_val_exc / total_val_samples if total_val_samples > 0 else 0
            
            avg_train_loss = epoch_train_loss_sum / len(train_loader)
            avg_train_mae = epoch_train_mae_sum / len(train_loader)
            avg_train_exc = epoch_train_exc_sum / len(train_loader)
            
            train_full_loss.append(avg_train_loss)
            train_loss_mae.append(avg_train_mae)
            train_loss_exc.append(avg_train_exc)
            val_full_loss.append(avg_val_loss)
            test_loss_mae.append(avg_val_mae)
            test_loss_exc.append(avg_val_exc)

            print(f"\n--- Epoch {epoch + 1} Summary ---")
            
            print("Training Set Metrics:")
            train_fchem = loss_function(FCHEM_VALIDATION, global_train_errors, val=False)
            print(f"Global Train Fchem: {train_fchem:.4f}\n")
            train_fchem_loss.append((1 - omega) * train_fchem / 50 + omega * train_loss_exc[-1] * 100)

            print("Validation Set Metrics:")
            val_fchem = loss_function(FCHEM_VALIDATION, global_val_errors, val=True)
            print(f"Global Validation Fchem: {val_fchem:.4f}\n")
            test_fchem.append((1 - omega) * val_fchem / 15 + omega * test_loss_exc[-1] * 200)
            

            val_loss_window.append(val_full_loss[-1])
            
            if prev and os.path.exists(prev): os.remove(prev)
            prev = f"{best_model_dir}bs_{batch_size}_lr_{lr_train}_{name}_{omega}_epoch_{epoch+1}_train_loss_{train_full_loss[-1]:.3f}_val_loss_{val_full_loss[-1]:.3f}_train_fchem_{train_fchem:.3f}_val_fchem_{val_fchem:.3f}.pth"
            torch.save(model.module.state_dict(), prev)

            if len(val_loss_window) == smoothing_window:
                if scheduler: scheduler.step()
                if val_full_loss[-1] <= min(val_full_loss):
                    print(f"New best Val loss: {val_full_loss[-1]:.3f}. Saving model.")
                    if prev_best and os.path.exists(prev_best):
                        try: os.remove(prev_best)
                        except OSError: pass
                    prev_best = f"{best_model_dir}BEST_EPOCH_bs_{batch_size}_lr_{lr_train}_{name}_{omega}_epoch_{epoch+1}_train_loss_{train_full_loss[-1]:.3f}_val_loss_{val_full_loss[-1]:.3f}_train_fchem_{train_fchem:.3f}_val_fchem_{val_fchem:.3f}.pth"
                    torch.save(model.module.state_dict(), prev_best)

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[3, 6], sharex=True)
            ax[0].plot(train_full_loss, label="Train Loss")
            ax[0].plot(val_full_loss, label="Validation Loss")
            ax[1].plot(train_fchem_loss, label="Train Fchem")
            ax[1].plot(test_fchem, label="Validation Fchem")
            ax[0].legend(); ax[1].legend()
            plt.savefig(f"./batch_fchem/bs_{batch_size}_lr_{lr_train}_{name}_{omega}.png")
            plt.close(fig)

    return train_loss_mae, test_loss_mae, prev_best


if __name__ == "__main__":

    data, data_train, data_test = load_chk(path="checkpoints")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parser = OptionParser()
    parser.add_option(
        "--Name", type=str, help="Name of the functional", default="PBE_8_32"
    )
    parser.add_option(
        "--N_preopt", type=int, default=3, help="Number of pre-optimization epochs"
    )
    parser.add_option(
        "--N_train", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_option(
        "--Batch_size", type=int, default=3, help="Number of reactions in a batch"
    )
    parser.add_option(
        "--Dropout", type=float, default=0.6, help="Dropout rate during training"
    )
    parser.add_option(
        "--Omega", type=float, default=0.0, help="Omega value in the loss function"
    )
    parser.add_option(
        "--LR_train", type=float, default=1e-4, help="Omega value in the loss function"
    )
    parser.add_option(
        "--LR_predopt",
        type=float,
        default=2e-2,
        help="Omega value in the loss function",
    )
    parser.add_option(
        "--Patience", type=int, default=50, help="Patience for early stopping"
    )

    (Opts, args) = parser.parse_args()

    (
        name,
        n_predopt,
        n_train,
        batch_size,
        dropout,
        omega,
        lr_train,
        lr_predopt,
        patience,
    ) = (
        Opts.Name,
        Opts.N_preopt,
        Opts.N_train,
        Opts.Batch_size,
        Opts.Dropout,
        Opts.Omega,
        Opts.LR_train,
        Opts.LR_predopt,
        Opts.Patience,
    )

    xalpha = False

    if "PBE" in name:
        rung = "GGA"
        dft = "PBE"

    elif "XALPHA" in name:
        rung = "LDA"
        dft = "XALPHA"
        xalpha = True
        nconstants = 1

    num_layers, h_dim = map(int, name.split("_")[1:])

    if dft == "PBE":
        if "STARSTAR" in name:
            base_model = pcPBEdoublestar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=dft).to(device)

        elif "STAR" in name:
            base_model = pcPBEstar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=dft).to(device)

        elif "PBE-L" in name:
            base_model = pcPBELMLOptimizer(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=dft).to(device)

        else:
            base_model = pcPBEMLOptimizer(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=dft).to(device)

    elif dft == "XALPHA":
        base_model = MLOptimizer(num_layers, h_dim, nconstants, dropout, dft).to(device)

    model = DDP(base_model, device_ids=[local_rank])
    model = torch.compile(model)

    if local_rank == 0:
        print(FCHEM_VALIDATION)
        print("name, n_predopt, n_train, batch_size, dropout, omega, lr_train, lr_predopt, patience")
        print(name, n_predopt, n_train, batch_size, dropout, omega, lr_train, lr_predopt, patience)
        print("Number of GPUs:", torch.cuda.device_count())
        total_params = sum(p.numel() for p in model.module.parameters()) 
        print(f"Number of parameters: {total_params}")

    with open("./dispersions/dispersions.pickle", "rb") as handle:
        dispersions = pickle.load(handle)

    train_set = Dataset(data=data_train)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
        generator=g,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    test_set = Dataset(data=data_test)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        sampler=test_sampler,
        generator=g,
        worker_init_fn=seed_worker,
    )
    train_predopt_set = DatasetPredopt(data=data, dft=dft)
    predopt_sampler = DistributedSampler(train_predopt_set, shuffle=False)
    train_predopt_dataloader = torch.utils.data.DataLoader(
        train_predopt_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=predopt_sampler,
        collate_fn=collate_fn_predopt,
        generator=g,
        worker_init_fn=seed_worker,
    )

    name += "_" + str(dropout)

    mae = nn.L1Loss()

    criterion = nn.MSELoss()

    name += "_" + str(dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_predopt, betas=(0.9, 0.999))

    double_star = False

    if "STARSTAR" in name:
        double_star = True

    train_loss_mse, train_loss_mae = predopt(
        model,
        criterion,
        optimizer,
        train_predopt_dataloader,
        device,
        n_epochs=n_predopt,
        accum_iter=1,
        double_star=double_star,
        xalpha=xalpha,
        local_rank=local_rank
    )

    true_constants_PBE = true_constants_PBE.to(device)
    optimizer = configure_optimizers(model=model, learning_rate=lr_train)

    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_epochs
    )

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=n_train - warmup_epochs, eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    early_stopper = EarlyStopper(patience=50)

    N_EPOCHS = n_train
    ACCUM_ITER = 1
    VERBOSE = False

    train_loss_mae, test_loss_mae, best_model_path = train(
        model,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        train_dataloader,
        test_dataloader,
        n_epochs=N_EPOCHS,
        accum_iter=ACCUM_ITER,
        omega=omega,
        verbose=VERBOSE,
        local_rank=local_rank
    )
