import os

import mlflow
import torch

from .NN_models import NN_PBE_model, NN_XALPHA_model
from .PBE import F_PBE
from .SVWN3 import F_XALPHA

dir_path = os.path.dirname(os.path.realpath(__file__))
relative_path_to_model_state_dict = {
    "NN_PBE": "checkpoints/NN_PBE",
    "NN_XALPHA": "checkpoints/NN_XALPHA",
}

nn_model = {
    "NN_PBE": NN_PBE_model,
    "NN_XALPHA": NN_XALPHA_model,
}

class NN_FUNCTIONAL:

    def __init__(self, name):
        path_to_model_state_dict = (
            dir_path + "/" + relative_path_to_model_state_dict[name]
        )
        model = nn_model[name]()
        state_dict = mlflow.pytorch.load_state_dict(
            path_to_model_state_dict, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        model.eval()
        self.name = name
        self.model = model

    def create_features_from_rhos(self, features, device):
        rho_only_a, grad_a_x, grad_a_y, grad_a_z, _, tau_a = torch.unsqueeze(
            features["rho_a"], dim=1
        )
        rho_only_b, grad_b_x, grad_b_y, grad_b_z, _, tau_b = torch.unsqueeze(
            features["rho_b"], dim=1
        )

        eps = 1e-27
        rho_only_a = torch.where(rho_only_a > eps, rho_only_a, 0.0)
        rho_only_b = torch.where(rho_only_b > eps, rho_only_b, 0.0)

        norm_grad_a = grad_a_x**2 + grad_a_y**2 + grad_a_z**2
        norm_grad_b = grad_b_x**2 + grad_b_y**2 + grad_b_z**2

        grad_x = grad_a_x + grad_b_x
        grad_y = grad_a_y + grad_b_y
        grad_z = grad_a_z + grad_b_z
        norm_grad = grad_x**2 + grad_y**2 + grad_z**2

        keys = [
            "rho_a",
            "rho_b",
            "norm_grad_a",
            "norm_grad",
            "norm_grad_b",
            "tau_a",
            "tau_b",
        ]
        values = [
            rho_only_a,
            rho_only_b,
            norm_grad_a,
            norm_grad,
            norm_grad_b,
            tau_a,
            tau_b,
        ]

        feature_dict = dict()
        for key, value in zip(keys, values):
            feature_dict[key] = value.to(device)
            feature_dict[key].requires_grad = True

        feature_dict["norm_grad_ab"] = (
            feature_dict["norm_grad"]
            - feature_dict["norm_grad_a"]
            - feature_dict["norm_grad_b"]
        ) / 2

        return feature_dict

    def create_features_from_libxc(self, features):
        rho = features["rho"]
        sigma = features["sigma"]
        tau = features["tau"]

        feature_dict = dict()

        feature_dict["rho_a"] = torch.tensor(rho / 2, dtype=torch.float32).view(1, -1)
        feature_dict["rho_b"] = torch.tensor(rho / 2, dtype=torch.float32).view(1, -1)
        feature_dict["tau_a"] = torch.tensor(tau / 2, dtype=torch.float32).view(1, -1)
        feature_dict["tau_b"] = torch.tensor(tau / 2, dtype=torch.float32).view(1, -1)
        feature_dict["norm_grad"] = torch.tensor(sigma, dtype=torch.float32).view(1, -1)
        feature_dict["norm_grad_a"] = torch.tensor(sigma / 4, dtype=torch.float32).view(
            1, -1
        )
        feature_dict["norm_grad_b"] = torch.tensor(sigma / 4, dtype=torch.float32).view(
            1, -1
        )
        feature_dict["norm_grad_ab"] = (
            feature_dict["norm_grad"]
            - feature_dict["norm_grad_a"]
            - feature_dict["norm_grad_b"]
        ) / 2

        return feature_dict

    def __call__(self, features, device, mode=None):

        torch.autograd.set_detect_anomaly(True)

        # Transfer model to device
        self.model = self.model.to(device)

        # Get features for NN and functional
        if mode:
            feature_dict = self.create_features_from_libxc(features)
        else:
            feature_dict = self.create_features_from_rhos(features, device)

        keys = [
            "rho_a",
            "rho_b",
            "norm_grad_a",
            "norm_grad",
            "norm_grad_b",
            "tau_a",
            "tau_b",
            "norm_grad_ab",
        ]

        # Concatenate features to get input for NN
        nn_inputs = torch.cat([feature_dict[key] for key in keys[:7]], dim=0).T

        nn_features = torch.tanh(nn_inputs)

        # Get the NN output
        constants = self.model(nn_features)

        # Get densities for functional input
        functional_densities = torch.cat(
            [feature_dict[key] for key in keys[:2]], dim=0
        ).T

        # Get gradients for functional input
        functional_gradients = torch.cat(
            [feature_dict[key] for key in [keys[2], keys[7], keys[4]]], dim=0
        ).T

        if self.name == "NN_PBE":
            vxc = F_PBE(functional_densities, functional_gradients, constants, device)
        elif self.name == "NN_XALPHA":
            vxc = F_XALPHA(functional_densities, constants)
        else:
            raise NameError(f"Invalid functional name: {self.name}")

        local_xc = vxc * (feature_dict["rho_a"] + feature_dict["rho_b"])

        return local_xc, vxc, feature_dict
