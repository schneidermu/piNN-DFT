import copy
import pickle
import random

import torch
from sklearn.model_selection import StratifiedKFold

from dataset import make_reactions_dict


def rename_keys(data):
    # Turns reaction_data dict keys names into numbers.
    l = len(data)
    keys = data.keys()
    data_new = {}
    for i, key in zip(range(l), keys):
        data_new[i] = data[key]
    return data_new


def train_split(data, test_size, shuffle=False, random_state=42):
    random.seed(random_state)
    # Returns train and test reaction dictionaries.
    X = []
    y = []
    for key in data:
        X.append(key)
        y.append(data[key]["Database"])
    skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=random_state)

    it = skf.split(X, y)
    train_index, test_index = next(it)

    train, test = dict(), dict()

    for i in train_index:
        train[i] = data[i]
    for i in test_index:
        test[i] = data[i]

    return rename_keys(train), rename_keys(test)


def prepare(path="data", test_size=0.2, random_state=41):
    # Make a single dictionary from the whole dataset.
    data = make_reactions_dict(path=path)

    # Train-test split.
    data_train, data_test = train_split(copy.deepcopy(data), test_size, shuffle=True, random_state=random_state)

    for data_t in (data_train, data_test):
        for i in range(len(data_t)):
            data_t[i]["Grid"] = torch.Tensor(data_t[i]["Grid"])

    return data, data_train, data_test


def save_chk(data, data_train, data_test, path="checkpoints"):
    # Save all processed data into pickle.
    with open(f"{path}/data.pickle", "wb") as f:
        pickle.dump(data, f)
    with open(f"{path}/data_train.pickle", "wb") as f:
        pickle.dump(data_train, f)
    with open(f"{path}/data_test.pickle", "wb") as f:
        pickle.dump(data_test, f)


def load_chk(path="checkpoints"):
    # Load processed data from pickle.
    with open(f"{path}/data.pickle", "rb") as f:
        data = pickle.load(f)
    with open(f"{path}/data_train.pickle", "rb") as f:
        data_train = pickle.load(f)
    with open(f"{path}/data_test.pickle", "rb") as f:
        data_test = pickle.load(f)
    return data, data_train, data_test


if __name__ == '__main__':
    data, data_train, data_test = prepare(path='data', test_size=0.2)
    save_chk(data, data_train, data_test, path='checkpoints')