import copy
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from utils import stack_reactions

sys.path.insert(0, str(Path(__file__).parent.parent))
from dft_functionals import PBE, true_constants_PBE


def group_and_augment_reactions(base_reactions, file_index):
    """
    Takes a set of base reactions, augments them with all available grid types,
    and returns them grouped by the original base reaction key.
    """
    all_grid_suffixes = set()
    for molecule_grids in file_index.values():
        all_grid_suffixes.update(molecule_grids.keys())

    if not all_grid_suffixes:
        print("Warning: No grid suffixes found for augmentation.")
        return {}

    grouped_augmented_dataset = defaultdict(list)

    for base_key, reaction_data in base_reactions.items():
        for grid_suffix in all_grid_suffixes:
            is_valid_combination = all(
                component in file_index and grid_suffix in file_index[component]
                for component in reaction_data["Components"]
            )
            
            if is_valid_combination:
                augmented_reaction = copy.deepcopy(reaction_data)
                augmented_reaction['component_paths'] = [
                    file_index[comp][grid_suffix] for comp in augmented_reaction["Components"]
                ]
                
                populated_reaction = add_reaction_info_from_h5(augmented_reaction, file_index)

                grouped_augmented_dataset[base_key].append(populated_reaction)
                
    return dict(grouped_augmented_dataset)


def build_file_index(data_path):
    """
    Scans the data directory and groups augmented .h5 files by their base molecule name,
    using a double underscore '__' as the separator.
    """
    file_index = defaultdict(dict)
    print(f"Building file index from: {data_path}")
    
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".h5"):
                file_name_no_ext = file.rsplit('.', 1)[0]

                if '__' in file_name_no_ext:
                    base_name, grid_suffix = file_name_no_ext.split('__', 1)
                else:
                    base_name = file_name_no_ext
                    grid_suffix = 'default'
                
                file_index[base_name][grid_suffix] = os.path.join(root, file)

    if not file_index:
        raise FileNotFoundError(f"No .h5 files found in {data_path}. Check the path.")
        
    print(f"Found {len(file_index)} unique molecule base names.")
    return file_index

def ref(x, y, path):
    """
    returns reference energies for points of a reaction grid from Reference_data.csv
    """

    pathfile = "../MN_dataset/Reference_data.csv"

    hartree2kcal = 627.5095
    with open(pathfile, newline="", encoding="cp1251") as csvfile:
        ref_file = csv.reader(csvfile, delimiter=",")
        k = 1
        if y == 391:
            k = hartree2kcal
        ref = []
        for n, i in enumerate(ref_file):
            if x <= n + 1 <= y:
                ref.append((i[0], float(i[2]) * k))

        return ref


def load_ref_energies(path):
    """Returns {db_name: [equation, energy]}"""
    ref_e = {  # Get the reference energies
        "MGAE109": ref(8, 116, path),
        "IP13": ref(155, 167, path),
        "EA13": ref(180, 192, path),
        "PA8": ref(195, 202, path),
        "DBH76": ref(251, 288, path) + ref(291, 328, path),
        "NCCE31": ref(331, 361, path),
        "ABDE4": ref(206, 209, path),
        "AE17": ref(375, 391, path),
        "pTC13": ref(232, 234, path) + ref(237, 241, path) + ref(244, 248, path),
    }
    return ref_e


def load_component_names(path):
    """
    Returns {db_name: {id: {'Components': [...], 'Coefficients: [...]'
                                }
                            }
                        }
     which is a dictionary with Components and Coefficients data about all reactions
    """
    pathfile = "../MN_dataset/total_dataframe_sorted_final.csv"
  
    with open(pathfile, newline="", encoding="cp1251") as csvfile:
        ref_file = csv.reader(csvfile, delimiter=",")
        ref = dict()
        current_database = None

        for n, line in enumerate(ref_file):
            line = np.array(line)
            if n == 0:
                components = np.array(line)
            else:
                reaction_id = int(line[0])
                reaction_database = line[1]
                reaction_component_num = np.nonzero(list(map(float, line[2:])))[0] + 2
                if reaction_database in ref:
                    ref[reaction_database][reaction_id] = {
                        "Components": components[reaction_component_num],
                        "Coefficients": line[reaction_component_num],
                    }
                else:
                    ref[reaction_database] = {
                        reaction_id: {
                            "Components": components[reaction_component_num],
                            "Coefficients": line[reaction_component_num],
                        }
                    }
        return ref


def get_compounds_coefs_energy(reactions, energies):
    """Returns {id:
                    {'Components': [...], 'Coefficients: [...]', 'Energy: float', Database: str
                                }
                            }
    which is a dictionary from load_component_names with Energy information added
    """
    data_final = dict()
    i = 0
    databases = energies.keys()
    for database in databases:
        data = reactions[database]
        for reaction in data:
            data_final[i] = {
                "Database": database,
                "Components": reactions[database][reaction][
                    "Components"
                ],  # .astype(object),
                "Coefficients": torch.Tensor(
                    reactions[database][reaction]["Coefficients"].astype(np.float32)
                ),
                "Energy": torch.tensor([energies[database][reaction][1]], dtype=torch.float32),
            }
            i += 1

    return data_final


def add_reaction_info_from_h5(reaction, file_index):
    """
    reaction must be from get_compounds_coefs_energy
    returns merged descriptos array X, integration weights,
    a and b densities and indexes for backsplitting
    Values are filtered based on density vanishing
    (rho[0] !~ 0 & rho[1] !~ 0)

    Adds the following information to the reaction dict using h5 files from the dataset:
    Grid : np.array with grid descriptors
    Weights : list with integration weights of grid points
    Densities : np.array with alpha and beta densities data for grid points
    HF_energies : list of Total HF energy (T+V) which needs to be added to E_xc
    backsplit_ind: list of indexes where we concatenate molecules' grids
    """
    eps = 1e-27
    X = np.array([])
    backsplit_ind = []
    HF_energies = np.array([])

    for h5_path in reaction['component_paths']:
        with h5py.File(h5_path, "r") as f:
            HF_energies = np.append(HF_energies, f["ener"][:][0])
            X_raw = np.array(f["grid"][:])
            if len(X) == 0:
                X = X_raw[:, 3:13]
            else:
                X = np.vstack((X, X_raw[:, 3:13]))

            X = X[np.logical_or((X[:, 1] > eps), (X[:, 2] > eps))]
            backsplit_ind.append(len(X))

    weights = X[:, 0]
    densities = X[:, 1:3]
    sigmas = X[:, 3:6]

    device = torch.device("cpu")
    pbe_constants = true_constants_PBE.to(device)
    local_pbe_energies = PBE.F_PBE(
        torch.from_numpy(densities).float(),
        torch.from_numpy(sigmas).float(),
        pbe_constants,
        device
    )

    X = X[:, 1:]  # get the grid descriptors

    # sigma_a_b to norm_grad=sigma_a + sigma_b + 2*sigma_a_b to get positive descriptor for log-transformation
    X = np.copy(X)
    X[:, 3] = X[:, 2] + X[:, 4] + 2 * X[:, 3]

    # Now X is rho_a, rho_b, sigma_aa, norm_sigma, sigma_bb, taua, taub, lapla, laplb

    backsplit_ind = np.array(backsplit_ind)

    labels = [
        "Grid",
        "Weights",
        "Densities",
        "Gradients",
        "HF_energies",
        "backsplit_ind",
        "PBE_local_energies"
    ]
    values = [X, weights, densities, sigmas, HF_energies, backsplit_ind, local_pbe_energies.numpy()]
    for label, value in zip(labels, values):
        reaction[label] = torch.Tensor(value)

    return reaction


def make_reactions_dict(path=None):
    """
    Builds a dictionary of all reactions, augmented and grouped by base reaction.
    """
    file_index = build_file_index(path)
    base_reactions = get_compounds_coefs_energy(
        load_component_names(path), load_ref_energies(path)
    )
    
    grouped_dataset = group_and_augment_reactions(base_reactions, file_index)
    
    total_samples = sum(len(v) for v in grouped_dataset.values())
    print(f"Grouped {len(base_reactions)} base reactions into {total_samples} total augmented samples.")
    return grouped_dataset

def collate_fn(data):
    """
    Custom collate function for torch train and test dataloader
    """
    reactions = []
    energies = []
    for reaction, energy in data:
        energies.append(energy)
        reaction.pop("Energy", None)
        reactions.append(reaction)

    torch_tensor_energy = torch.tensor(energies)
    reactions_stacked = stack_reactions(reactions)

    del energies, reactions, data
    return reactions_stacked, torch_tensor_energy


def collate_fn_predopt(data):
    """
    Custom collate function for torch predopt dataloader
    """
    reactions = []
    for reaction, constant in data:
        reactions.append(reaction)
    reactions_stacked = stack_reactions(reactions)
    del reactions, data
    return reactions_stacked, constant


def fast_collate_fn_predopt(data, chunk_size=1000):
    """
    An optimized collate function specifically for the predopt task.
    """

    all_grid_chunks = []
    
    reactions, constants = zip(*data)
    constant_target = constants[0] 

    for reaction in reactions:
        chunks = torch.split(reaction['Grid'], chunk_size, dim=0)
        all_grid_chunks.extend(chunks)

    final_grid_batch = torch.cat(all_grid_chunks, dim=0)
    
    return {'Grid': final_grid_batch}, constant_target