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


FCHEM_FACTORS = {
    "MGAE109": 1/4.73394495412844,
    "NCCE31": 10
}


def Fchem(total_database_errors):
    Fchem_ = 0
    for db in total_database_errors:
        error = np.sqrt(np.mean((np.array(total_database_errors[db]))**2))
#        if db in FCHEM_FACTORS:
#            error *= FCHEM_FACTORS[db]
        Fchem_ += error

    return Fchem_

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def exc_loss(reaction, pred_constants, dft="PBE", true_constants=true_constants_PBE):
    HARTREE2KCAL = 627.5095
    backsplit_ind = reaction["backsplit_ind"].to(torch.int32)
    indices = list(
        zip(
            torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)),
            backsplit_ind,
        )
    )
    n_molecules = len(indices)
    loss = torch.tensor(0.0, requires_grad=True)
    predicted_local_energies = get_local_energies(
        reaction, pred_constants, device, rung=rung, dft=dft
    )["Local_energies"]
    predicted_local_energies = [
        predicted_local_energies[start:stop] for start, stop in indices
    ]
    true_local_energies = get_local_energies(
        reaction, true_constants, device, rung="GGA", dft="PBE"
    )["Local_energies"]
    true_local_energies = [true_local_energies[start:stop] for start, stop in indices]
    for i in range(n_molecules):
        loss += (
            1
            / len(predicted_local_energies[i])
            * (
                torch.sum((predicted_local_energies[i] - true_local_energies[i]) ** 2)
            )
        )

    return loss * HARTREE2KCAL / n_molecules


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

with open("./dispersions/dispersions.pickle", "rb") as handle:
    dispersions = pickle.load(handle)

#dispersions = {}

mae = nn.L1Loss()

lst = []
local_lst = []
names = {
    0: "Train",
    1: "Test",
}
with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader]):

        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}

        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            bases += [x for x in X_batch["Database"]]
            grid_size = len(X_batch["Grid"])
            constants = (torch.ones(grid_size) * 1.05).view(grid_size, 1)
            local_loss = exc_loss(X_batch, constants, dft="XALPHA")
            energies = calculate_reaction_energy(
                X_batch, constants, device, rung="LDA", dft="XALPHA", dispersions=dispersions
            )
            if len(pred_energies): 
                pred_energies = torch.hstack([pred_energies, energies])
                ref_energies = torch.hstack([ref_energies, y_batch])
                errors = torch.hstack([errors, pred_energies-ref_energies])
            else:
                pred_energies = energies
                ref_energies = y_batch
                errors = energies-y_batch
            
            for base, error in zip(bases, errors):
                total_database_errors[base] = total_database_errors.get(base, [])
                total_database_errors[base].append(abs(error.item()))
            

            local_lst.append(torch.sqrt(local_loss).item())

        for db in sorted(total_database_errors):
            error = (np.mean(np.array(total_database_errors[db])))
            print(f'{db}: {error}')

        print("Fchem =", Fchem(total_database_errors))

        print(f"XAlpha {names[index]} MAE =", mae(pred_energies, ref_energies).item())
        print(f"XAlpha {names[index]} Local Loss =", np.mean(np.array(local_lst)))

# XAlpha-D3(BJ) Train Fchem = 792.4750547841372
# XAlpha-D3(BJ) Train Local Loss = 1.2104060273421438
#ABDE4: 21.062885645816202
#AE17: 52.69695062864395
#DBH76: 44.72368584405827
#EA13: 36.83156373044278
#IP13: 71.66897164706526
#MGAE109: 24.982885147478594
#NCCE31: 43.26800077131104
#PA8: 16.822658454315572
#pTC13: 62.54664613336206

# XAlpha-D3(BJ) Test Fchem = 211.41702118178105
# XAlpha-D3(BJ) Test Local Loss = 1.230004608631134
#ABDE4: 5.036346435546875
#AE17: 22.092093331473215
#DBH76: 22.61078643798828
#EA13: 7.615932464599609
#IP13: 33.42640075683594
#MGAE109: 20.841668540553044
#NCCE31: 21.922976970672607
#PA8: 13.867073059082031
#pTC13: 39.849639892578125
        

        


with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader]):
        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        total_database_errors = {}


        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            grid_size = len(X_batch["Grid"])
            bases += [x for x in X_batch["Database"]]
            constants = (torch.ones(grid_size * 24)).view(
                grid_size, 24
            ) * true_constants_PBE
            constants = constants
            energies = calculate_reaction_energy(
                X_batch, constants, device, rung="GGA", dft="PBE", dispersions=dispersions
            )
            if len(pred_energies): 
                pred_energies = torch.hstack([pred_energies, energies])
                ref_energies = torch.hstack([ref_energies, y_batch])
                errors = torch.hstack([errors, pred_energies-ref_energies])
            else:
                pred_energies = energies
                ref_energies = y_batch
                errors = energies-y_batch

            for base, error in zip(bases, errors):
                total_database_errors[base] = total_database_errors.get(base, [])
                total_database_errors[base].append(abs(error.item()))
            

        for db in sorted(total_database_errors):
            error = (np.mean(np.array(total_database_errors[db])))
            print(f'{db}: {error}')

        print("Fchem =", Fchem(total_database_errors))

        print(f"PBE {names[index]} MAE =", mae(pred_energies, ref_energies).item())


# PBE-D3(BJ) Train Fchem = 131.88553919047612
#ABDE4: 15.637174465439536
#AE17: 8.534409250807666
#DBH76: 10.061235319210004
#EA13: 15.401774725037304
#IP13: 10.713732359053074
#MGAE109: 13.32331867085641
#NCCE31: 11.169366430813351
#PA8: 10.91599115618953
#pTC13: 9.69357694120712

# PBE-D3(BJ) Test Fchem = 121.97746813422154
#ABDE4: 11.834625244140625
#AE17: 18.253914969308035
#DBH76: 13.595369743578361
#EA13: 9.636272430419922
#IP13: 11.319406127929687
#MGAE109: 5.945950959858141
#NCCE31: 14.399985790252686
#PA8: 9.213600158691406
#pTC13: 18.13916015625