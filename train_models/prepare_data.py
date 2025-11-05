import os
import pickle
import random

import torch
from sklearn.model_selection import StratifiedKFold

from dataset import (build_file_index, get_compounds_coefs_energy,
                     group_and_augment_reactions, load_component_names,
                     load_ref_energies)


def flatten_grouped_data(grouped_data):
    """
    Converts a grouped dictionary of augmentations into a flat dictionary.
    Input: {base_key_0: [aug_1, aug_2], ...}
    Output: {0: aug_1, 1: aug_2, ...}
    """
    flat_dict = {}
    i = 0
    for group in grouped_data.values():
        for reaction in group:
            flat_dict[i] = reaction
            i += 1
    return flat_dict


def prepare(path="data", test_size=0.2, random_state=42):
    """
    Prepares the dataset by performing a split on the base reactions
    and then applying augmentation to the train and test sets independently.
    This prevents data leakage.
    """
    random.seed(random_state)
    
    print("Loading base reactions...")
    ref_energies = load_ref_energies(path)
    component_names = load_component_names(path)
    base_reactions = get_compounds_coefs_energy(component_names, ref_energies)
    print(f"Found {len(base_reactions)} unique base reactions.")

    X_base_keys = list(base_reactions.keys())
    y_base_labels = [r["Database"] for r in base_reactions.values()]
    
    skf = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    try:
        train_indices, test_indices = next(skf.split(X_base_keys, y_base_labels))
    except ValueError as e:
        print(f"Error during StratifiedKFold. This can happen if a database has fewer members than n_splits. Error: {e}")
        indices = list(range(len(X_base_keys)))
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))
        train_indices, test_indices = indices[:split_point], indices[split_point:]

    train_base_dict, test_base_dict = {}, {}
    
    for i in train_indices:
        key = X_base_keys[i]
        reaction = base_reactions[key]
        if (
            (reaction["Database"] == "AE17" and reaction["Components"][0] not in ["H_ae17", "He_ae17", "Li_ae17", "Be_ae17", "N_ae17", "Ne_ae17", "Na_ae17", "Mg_ae17", "P_ae17", "Ar_ae17"])
            or ("HCl_htbh38" in reaction["Components"])
            or ("HCl_mgae109" in reaction["Components"])
        ):
            print(f"Moving reaction {key} ({reaction['Components']}) from train to test set due to override rule.")
            test_base_dict[key] = reaction 
        else:
            train_base_dict[key] = reaction

    for i in test_indices:
        key = X_base_keys[i]
        reaction = base_reactions[key]
        if reaction["Database"] == "AE17" and reaction["Components"][0] in ["H_ae17", "He_ae17", "Li_ae17", "Be_ae17", "N_ae17", "Ne_ae17", "Na_ae17", "Mg_ae17", "P_ae17", "Ar_ae17"]:
             print(f"Moving reaction {key} ({reaction['Components']}) from test to train set due to override rule.")
             train_base_dict[key] = reaction
        else:
            if key not in test_base_dict:
                test_base_dict[key] = reaction

    print(f"Base reactions split: {len(train_base_dict)} train, {len(test_base_dict)} test.")

    file_index = build_file_index(path)
    
    print("\nAugmenting and grouping training set...")
    data_train_grouped = group_and_augment_reactions(train_base_dict, file_index)
    total_train_samples = sum(len(v) for v in data_train_grouped.values())
    print(f"Generated {total_train_samples} training samples from {len(data_train_grouped)} base reactions.")

    print("\nAugmenting and grouping test set...")
    data_test_grouped = group_and_augment_reactions(test_base_dict, file_index)
    total_test_samples = sum(len(v) for v in data_test_grouped.values())
    print(f"Generated {total_test_samples} test samples from {len(data_test_grouped)} base reactions.")
    
    print("\nGenerating flat augmented dataset for predopt...")
    all_data_grouped = group_and_augment_reactions(base_reactions, file_index)

    data_predopt_flat = flatten_grouped_data(all_data_grouped)
    print(f"Generated {len(data_predopt_flat)} total flat samples for predopt.")
    
    for i in data_predopt_flat:
        data_predopt_flat[i]["Grid"] = torch.Tensor(data_predopt_flat[i]["Grid"])

    return data_predopt_flat, data_train_grouped, data_test_grouped


def save_chk(data, data_train, data_test, path="checkpoints"):
    """
    Save all processed data into pickle.
    `data` is a flat dict for predopt.
    `data_train` and `data_test` are dicts grouped by base reaction.
    """
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/data_predopt.pickle", "wb") as f:
        pickle.dump(data, f)
    with open(f"{path}/data_train_grouped.pickle", "wb") as f:
        pickle.dump(data_train, f)
    with open(f"{path}/data_test_grouped.pickle", "wb") as f:
        pickle.dump(data_test, f)


def load_chk(path="checkpoints"):
    """
    Load processed data from pickle.
    Returns the flat predopt data and the grouped train/test data.
    """
    with open(f"{path}/data_predopt.pickle", "rb") as f:
        data = pickle.load(f)
    with open(f"{path}/data_train_grouped.pickle", "rb") as f:
        data_train = pickle.load(f)
    with open(f"{path}/data_test_grouped.pickle", "rb") as f:
        data_test = pickle.load(f)
    return data, data_train, data_test


if __name__ == "__main__":
    data_flat, data_train_grouped, data_test_grouped = prepare(path="data", test_size=0.2)
    save_chk(data_flat, data_train_grouped, data_test_grouped, path="checkpoints")
    print("\nData preparation complete and saved to pickle files.")