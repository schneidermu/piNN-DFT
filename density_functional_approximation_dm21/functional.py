import os

import torch

from .NN_models import NN_PBE_model, NN_XALPHA_model
from .PBE import F_PBE
from .SVWN3 import F_XALPHA
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
relative_path_to_model_state_dict = {
    "NN_PBE": "checkpoints/NN_PBE/state_dict.pth",
    "NN_XALPHA": "checkpoints/NN_XALPHA/state_dict.pth",
}

omega_str_list = ['0', '0076', '067', '18', '33', '50', '67', '82', '93', '99']

relative_path_to_model_state_dict.update({f"NN_XALPHA_{omega}": f"checkpoints/NN_XALPHA/state_dict_0.{omega}.pth" for omega in omega_str_list})
relative_path_to_model_state_dict.update({f"NN_PBE_{omega}": f"checkpoints/NN_PBE/state_dict_0.{omega}.pth" for omega in omega_str_list})
relative_path_to_model_state_dict.update({f"NN_XALPHA_100": f"checkpoints/NN_XALPHA/state_dict_1.pth"})
relative_path_to_model_state_dict.update({f"NN_PBE_100": f"checkpoints/NN_PBE/state_dict_1.pth"})

nn_model = {
    "NN_PBE": NN_PBE_model,
    "NN_XALPHA": NN_XALPHA_model,
}

omega_str_list.append('100')

nn_model.update({f"NN_XALPHA_{omega}": NN_XALPHA_model for omega in omega_str_list})
nn_model.update({f"NN_PBE_{omega}": NN_PBE_model for omega in omega_str_list})


class NN_FUNCTIONAL:

    def __init__(self, name):
        path_to_model_state_dict = (
            dir_path + "/" + relative_path_to_model_state_dict[name]
        )
        model = nn_model[name]()
        model.load_state_dict(
            torch.load(path_to_model_state_dict, map_location=torch.device("cpu")), strict=False
        )
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
            feature_dict[key] = value
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

        eps = 1e-21

        nn_inputs[:, 0] = 1/(feature_dict["rho_a"] + 1e-10)**(1/3)
        nn_inputs[:, 1] = 1/(feature_dict["rho_b"] + 1e-10)**(1/3) 
        alpha_grad = torch.sqrt(feature_dict["norm_grad_a"] + eps)/(feature_dict["rho_a"] + 1e-10)**(4/3)
        norm_grad = torch.sqrt(feature_dict["norm_grad"])/(feature_dict["rho_a"] + feature_dict["rho_b"] + 1e-10)**(4/3)
        beta_grad = torch.sqrt(feature_dict["norm_grad_b"] + eps)/(feature_dict["rho_b"] + 1e-10)**(4/3)
        nn_inputs[:, 2] = torch.where(feature_dict["rho_a"]>eps, alpha_grad, 0)
        nn_inputs[:, 3] = torch.where((feature_dict["rho_a"]+feature_dict["rho_b"])>eps, norm_grad, 0)
        nn_inputs[:, 4] = torch.where(feature_dict["rho_b"]>eps, beta_grad, 0)

        tau_tf_alpha = 3/10 * (3*np.pi**2)**(2/3) * (feature_dict["rho_a"] + 1e-10)**(5/3)
        tau_tf_beta = 3/10 * (3*np.pi**2)**(2/3) * (feature_dict["rho_b"] + 1e-10)**(5/3)

        nn_inputs[:, 5] = torch.where(tau_tf_alpha>eps, (feature_dict["tau_a"] - tau_tf_alpha)/(tau_tf_alpha + 1e-7), 0)
        nn_inputs[:, 6] = torch.where(tau_tf_beta>eps, (feature_dict["tau_b"] - tau_tf_beta)/(tau_tf_beta + 1e-7), 0)

        nn_features = torch.tanh(nn_inputs)

        # Get the NN output
        constants = self.model(nn_features)

        # Get densities for functional input
        functional_densities = torch.cat(
            [feature_dict[key] for key in keys[:2]], dim=0
        ).T

        # Get gradients for functional input
        functional_gradients = torch.cat(
            [feature_dict["norm_grad_a"], feature_dict["norm_grad_ab"], feature_dict["norm_grad_b"]], dim=0
        ).T

        if "NN_PBE" in self.name:
            vxc = F_PBE(functional_densities, functional_gradients, constants, device)
        elif "NN_XALPHA" in self.name:
            vxc = F_XALPHA(functional_densities, constants)
        else:
            raise NameError(f"Invalid functional name: {self.name}")

        local_xc = vxc * (feature_dict["rho_a"] + feature_dict["rho_b"])

        return local_xc, vxc, feature_dict
