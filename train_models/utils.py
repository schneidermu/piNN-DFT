import inspect
import os
import random
from collections import defaultdict
from itertools import chain
from operator import methodcaller

import matplotlib.pyplot as plt
import numpy as np
import torch


def catch_nan(**kwargs):
    nan_detected = False
    inf_detected = False
    for k, v in kwargs.items():
        if v.isnan().any():
            print(f"{k} is NaN")
            nan_detected = True
        if v.isinf().any():
            print(f"{k} is inf")
            inf_detected = True

    if nan_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f"log/{k}.pt")
        raise ValueError("NaN detected")
    if inf_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f"log/{k}.pt")
        raise ValueError("infinity detected")


def save_tensors(**kwargs):
    for k, v in kwargs.items():
        torch.set_printoptions(precision=25)
        torch.save(v, f"log/{k}.pt")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def set_random_seed(seed):
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def stack_reactions(reactions):
    """
    Robustly stacks a list of reaction dictionaries into a single dictionary of batched tensors/arrays.
    """
    if not reactions:
        return {}

    collated = defaultdict(list)
    for reaction in reactions:
        for key, value in reaction.items():
            collated[key].append(value)

    final_reaction = {}
    reaction_indices = [0]
    stop = 0

    for key, value_list in collated.items():

        first_item = value_list[0]

        if isinstance(first_item, np.ndarray):
            if key == "Components":
                 flattened_components = []
                 for components_array in value_list:
                     flattened_components.extend(components_array)
                     reaction_indices.append(stop + len(components_array))
                     stop += len(components_array)
                 final_reaction[key] = np.array(flattened_components)
            else:
                 final_reaction[key] = np.hstack(value_list)
        
        elif isinstance(first_item, torch.Tensor):
            if key in ("Grid", "Densities", "Gradients"):
                final_reaction[key] = torch.vstack(value_list)
            elif key == "backsplit_ind":
                new_backsplit = []
                current_offset = 0
                for ind_tensor in value_list:
                    new_backsplit.append(ind_tensor + current_offset)
                    current_offset = new_backsplit[-1][-1]
                final_reaction[key] = torch.cat(new_backsplit)
            else:
                processed_list = [v.unsqueeze(0) if v.dim() == 0 else v for v in value_list]
                final_reaction[key] = torch.cat(processed_list, dim=0)

        elif isinstance(first_item, list):

            flattened_list = [item for sublist in value_list for item in sublist]
            final_reaction[key] = flattened_list
            
        else:
            final_reaction[key] = value_list
        

    final_reaction["reaction_indices"] = reaction_indices
    return final_reaction


def configure_optimizers(model, learning_rate):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.PReLU,
        torch.nn.BatchNorm1d,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )
    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": 1e-2,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.RAdam(
        optim_groups, lr=learning_rate, decoupled_weight_decay=True
    )
    return optimizer
