import random
import pickle

import numpy as np
import torch
from torch import nn

from predopt import true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import (calculate_reaction_energy,
                                         get_local_energies)

device = torch.device("cpu")

rung = "LDA"
dft = "XALPHA"


def loss_function(factor_dictionary, total_database_errors):
    fchem = 0

    for db in sorted(total_database_errors):
        factor = factor_dictionary.get(db, 1)
        error = np.sqrt((np.mean(np.array(total_database_errors[db])**2)))
        fchem += error*factor
        print(f'{db}: {error*factor:.3f}')

    return fchem


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


def extend_bases(X_batch, bases):
    if len(X_batch["Database"][0]) == 1:
        current_bases = [X_batch["Database"],]
    else:
        current_bases = list(X_batch["Database"])
    bases += current_bases

    return current_bases, bases


FCHEM_FACTORS = {
    "MGAE109": 1/4.73394495412844,
    "NCCE31": 10
}
PBE_TRAIN_ENERGY_FACTORS = {
    "ABDE4": 1/92.10,
    "AE17": 0.00196,
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
    "AE17": 0.00217,
    "DBH76": 1/21.88,
    "EA13": 1/56.72,
    "IP13": 1/253.83,
    "MGAE109": 1/460.79,
    "NCCE31": 1/2.01,
    "PA8": 1/174.85,
    "pTC13": 1/201.57,
}

PBE_TRAIN_FACTORS = {
    "ABDE4": 3.681/92.10,
    "AE17": 0.118/10,
    "DBH76": 8.504/17.54,
    "EA13": 2.671/34.96,
    "IP13": 3.521/257.31,
    "MGAE109": 15.631/508.07,
    "NCCE31": 1.020/4.02,
    "PA8": 1.091/159.48,
    "pTC13": 6.169/162.83,
}

PBE_VAL_FACTORS = {
    "ABDE4": 0.117/97.39,
    "AE17": 0.130/10,
    "DBH76": 9.051/21.88,
    "EA13": 1.432/56.72,
    "IP13": 3.291/253.83,
    "MGAE109": 13.709/460.79,
    "NCCE31": 1.048/2.01,
    "PA8": 2.612/174.85,
    "pTC13": 4.068/201.57,
}


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

criterion = nn.MSELoss()

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

set_random_seed(41)

data, data_train, data_test = load_chk(path="checkpoints")


from dataset import collate_fn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
#        self.data[i].pop("Database", None)
        return self.data[i], self.data[i]["Energy"]

    def __len__(self):
        return len(self.data.keys())


train_set = Dataset(data=data_train)
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    collate_fn=collate_fn,
)


test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn,
)
total_set = Dataset(data=data)
total_dataloader = torch.utils.data.DataLoader(
    total_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn,
)

with open("./dispersions/dispersions.pickle", "rb") as handle:
    dispersions = pickle.load(handle)

#dispersions = {}

mae = nn.L1Loss()
mse = nn.MSELoss()

lst = []
local_lst = []
names = {
    0: "Train",
    1: "Test",
    2: "Total",
}
with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader, total_dataloader]):

        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}

        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            current_bases, bases = extend_bases(X_batch, bases)
            grid_size = len(X_batch["Grid"])
            constants = (torch.ones(grid_size) * 1.05).view(grid_size, 1)
            if index == 1:
                val=True
            else:
                val=False
            local_loss = exc_loss(X_batch, constants, dft="XALPHA", val=val)

            energies, local_energies = calculate_reaction_energy(
                X_batch, constants, device, rung="LDA", dft="XALPHA", dispersions=dispersions
            )
            pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(pred_energies, energies, errors, ref_energies, y_batch, total_database_errors, current_bases)
            

            local_lst.append(torch.sqrt(local_loss).item())

        fchem = loss_function(FCHEM_FACTORS, total_database_errors)

        print("Fchem =", fchem)

        print(f"XAlpha {names[index]} MAE =", mae(pred_energies, ref_energies).item())
        print(f"XAlpha {names[index]} Local Loss =", np.mean(np.array(local_lst)))

# XAlpha-D3(BJ) Train Fchem = 533.50
# XAlpha-D3(BJ) Train Local Loss = 1.21
#ABDE4: 4.969
#AE17: 438.235
#DBH76: 11.007
#EA13: 18.603
#IP13: 19.591
#MGAE109: 27.270
#NCCE31: 3.024
#PA8: 5.587
#pTC13: 5.218

# XAlpha-D3(BJ) Test Fchem = 762.35
# XAlpha-D3(BJ) Test Local Loss = 1.23
# ABDE4: 4.929
# AE17: 687.092
# DBH76: 12.820
# EA13: 10.421
# IP13: 9.051
# MGAE109: 21.203
# NCCE31: 2.570
# PA8: 7.924
# pTC13: 6.344

# XAlpha-D3(BJ) Total Fchem = 689.78
# XAlpha-D3(BJ) Total Local Loss = 1.2172921672463417
# ABDE4: 4.959
# AE17: 496.789
# DBH76: 11.365
# EA13: 17.344
# IP13: 17.159
# MGAE109: 26.045
# NCCE31: 2.936
# PA8: 6.171
# pTC13: 5.391


with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader, total_dataloader]):
        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        total_database_errors = {}


        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            grid_size = len(X_batch["Grid"])
            current_bases, bases = extend_bases(X_batch, bases)
            constants = (torch.ones(grid_size * 24)).view(
                grid_size, 24
            ) * true_constants_PBE
            energies, local_energies = calculate_reaction_energy(
                X_batch, constants, device, rung="GGA", dft="PBE", dispersions=dispersions
            )
            pred_energies, ref_energies, errors, total_database_errors = make_total_db_errors(pred_energies, energies, errors, ref_energies, y_batch, total_database_errors, current_bases)

        fchem = loss_function(FCHEM_FACTORS, total_database_errors)

        print("Fchem =", fchem)

        print(f"PBE {names[index]} MAE =", mae(pred_energies, ref_energies).item())



# WITH D3 Train Fchem = 84.23
# ABDE4: 3.681
# AE17: 41.938
# DBH76: 8.504
# EA13: 2.671
# IP13: 3.521
# MGAE109: 15.631
# NCCE31: 1.020
# PA8: 1.091
# pTC13: 6.169


# WITH D3 Test Fchem = 101.34
# ABDE4: 0.117
# AE17: 66.008
# DBH76: 9.051
# EA13: 1.432
# IP13: 3.291
# MGAE109: 13.709
# NCCE31: 1.048
# PA8: 2.612
# pTC13: 4.068


# Total D3 Fchem = 88.54
# ABDE4: 2.790
# AE17: 47.602
# DBH76: 8.612
# EA13: 2.481
# IP13: 3.468
# MGAE109: 15.243
# NCCE31: 1.026
# PA8: 1.472
# pTC13: 5.846