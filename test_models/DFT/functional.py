import os
import sys
from pathlib import Path

import numpy
import numpy as np
import torch

from .NN_models import (NN_PBE_L_model, NN_PBE_model, NN_PBE_star_model,
                        NN_PBE_star_star_model, NN_XALPHA_model)

# Import from shared dft_functionals at project root
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))
from dft_functionals import PBE, SVWN3, true_constants_PBE

F_PBE = PBE.F_PBE
F_XALPHA = SVWN3.F_XALPHA

torch.set_default_tensor_type(torch.DoubleTensor)

dir_path = os.path.dirname(os.path.realpath(__file__))
relative_path_to_model_state_dict = {
    "NN_PBE": "checkpoints/NN_PBE/state_dict.pth",
    "NN_XALPHA": "checkpoints/NN_XALPHA/state_dict.pth",
}

func_names = ["NN_PBE", "NN_XALPHA", "NN_PBE-L", "NN_PBE_star", "NN_PBE_star_star"]

omega_str_list = ["0", "0076", "067", "18", "33", "50", "67", "82", "93", "99"]

for name in func_names:
    relative_path_to_model_state_dict.update(
        {
            f"{name}_{omega}": f"checkpoints/{name}/state_dict_0.{omega}.pth"
            for omega in omega_str_list
        }
    )
    relative_path_to_model_state_dict.update(
        {f"{name}_100": f"checkpoints/{name}/state_dict_1.pth"}
    )

nn_model = {
    "NN_PBE": NN_PBE_model,
    "NN_XALPHA": NN_XALPHA_model,
    "NN_PBE_star": NN_PBE_star_model,
    "NN_PBE_star_star": NN_PBE_star_star_model,
    "NN_PBE-L": NN_PBE_L_model,
}

omega_str_list.append("100")

for name in list(nn_model.keys()):
    nn_model.update({f"{name}_{omega}": nn_model[name] for omega in omega_str_list})


class NN_FUNCTIONAL:

    def __init__(self, name):
        path_to_model_state_dict = (
            dir_path + "/" + relative_path_to_model_state_dict[name]
        )
        model = nn_model[name]()
        print(path_to_model_state_dict)
        model.load_state_dict(
            torch.load(path_to_model_state_dict, map_location=torch.device("cpu"))
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

        feature_dict["rho_a"] = torch.tensor(rho / 2, dtype=torch.float64).view(1, -1)
        feature_dict["rho_b"] = torch.tensor(rho / 2, dtype=torch.float64).view(1, -1)
        feature_dict["tau_a"] = torch.tensor(tau / 2, dtype=torch.float64).view(1, -1)
        feature_dict["tau_b"] = torch.tensor(tau / 2, dtype=torch.float64).view(1, -1)
        feature_dict["norm_grad"] = torch.tensor(sigma, dtype=torch.float64).view(1, -1)
        feature_dict["norm_grad_a"] = torch.tensor(sigma / 4, dtype=torch.float64).view(
            1, -1
        )
        feature_dict["norm_grad_b"] = torch.tensor(sigma / 4, dtype=torch.float64).view(
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

        eps_rho = 1e-27
        eps_sigma = eps_rho ** (8 / 3)

        nn_inputs = torch.cat([feature_dict[key] for key in keys[:7]], dim=0).T
        rho_a_inp = feature_dict["rho_a"] + eps_rho
        rho_b_inp = feature_dict["rho_b"] + eps_rho
        grad_a_inp = feature_dict["norm_grad_a"]
        grad_b_inp = feature_dict["norm_grad_b"]
        grad_inp = feature_dict["norm_grad"]
        tau_a_inp = feature_dict["tau_a"]
        tau_b_inp = feature_dict["tau_b"]

        constants = self.model(
            nn_inputs,
            rho_a_inp,
            rho_b_inp,
            grad_a_inp,
            grad_b_inp,
            grad_inp,
            tau_a_inp,
            tau_b_inp,
        )

        functional_densities = torch.cat(
            [feature_dict[key] for key in keys[:2]], dim=0
        ).T

        functional_gradients = torch.cat(
            [
                feature_dict["norm_grad_a"],
                feature_dict["norm_grad_ab"],
                feature_dict["norm_grad_b"],
            ],
            dim=0,
        ).T

        if "NN_PBE" in self.name:
            if "star_star" in self.name:
                vxc = F_PBE(
                    functional_densities,
                    functional_gradients,
                    true_constants_PBE,
                    "cpu",
                    torch.stack(constants, dim=1),
                )
            else:
                vxc = F_PBE(
                    functional_densities, functional_gradients, constants, "cpu"
                )
        elif "NN_XALPHA" in self.name:
            vxc = F_XALPHA(functional_densities, constants)
        else:
            raise NameError(f"Invalid functional name: {self.name}")

        local_xc = vxc * (feature_dict["rho_a"] + feature_dict["rho_b"])

        return local_xc, vxc, feature_dict

    def torch_grad(self, outputs, inputs):
        grads = torch.autograd.grad(
            outputs,
            inputs,
            create_graph=True,
            only_inputs=True,
        )
        return grads

    def eval_xc(
        self, xc_code, rho, spin, relativity=0, deriv=2, omega=None, verbose=None
    ):
    
        if spin == 0:

            with_lapl = (rho.shape[0] == 6)

            if not with_lapl:
                rho_only_a, grad_a_x, grad_a_y, grad_a_z, tau_a = torch.unsqueeze(
                    torch.tensor(rho / 2, dtype=torch.float64), dim=1
                )
                rho_only_b, grad_b_x, grad_b_y, grad_b_z, tau_b = torch.unsqueeze(
                    torch.tensor(rho / 2, dtype=torch.float64), dim=1
                )
            else:
                rho_only_a, grad_a_x, grad_a_y, grad_a_z, lapl_a, tau_a = torch.unsqueeze(
                    torch.tensor(rho / 2, dtype=torch.float64), dim=1
                )
                rho_only_b, grad_b_x, grad_b_y, grad_b_z, lapl_b, tau_b = torch.unsqueeze(
                    torch.tensor(rho / 2, dtype=torch.float64), dim=1
                )
                
        else:
            
            with_lapl = (rho[0].shape[0] == 6)

            if not with_lapl:
                rho_only_a, grad_a_x, grad_a_y, grad_a_z, tau_a = torch.unsqueeze(
                    torch.tensor(rho[0], dtype=torch.float64), dim=1
                )
                rho_only_b, grad_b_x, grad_b_y, grad_b_z, tau_b = torch.unsqueeze(
                    torch.tensor(rho[1], dtype=torch.float64), dim=1
                )
            else:
                rho_only_a, grad_a_x, grad_a_y, grad_a_z, lapl_a, tau_a = torch.unsqueeze(
                    torch.tensor(rho[0], dtype=torch.float64), dim=1
                )
                rho_only_b, grad_b_x, grad_b_y, grad_b_z, lapl_b, tau_b = torch.unsqueeze(
                    torch.tensor(rho[1], dtype=torch.float64), dim=1
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
        
        if with_lapl:
            values.extend([lapl_a, lapl_b])
            keys.extend(["lapl_a", "lapl_b"])

        feature_dict = dict(zip(keys, values))

        for key in feature_dict:
            feature_dict[key].requires_grad = True

        feature_dict["norm_grad_ab"] = (
            feature_dict["norm_grad"]
            - feature_dict["norm_grad_a"]
            - feature_dict["norm_grad_b"]
        ) / 2

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
        
        if with_lapl:
            keys.extend(["lapl_a", "lapl_b"])

        if not with_lapl: 
            nn_inputs = torch.cat([feature_dict[key] for key in keys[:7]], dim=0).T
        else:
            nn_inputs = torch.cat([feature_dict[key] for key in keys[:9]], dim=0).T

        eps_rho = 1e-10
        eps_sigma = 1e-30

        rho_a_inp = feature_dict["rho_a"] + eps_rho
        rho_b_inp = feature_dict["rho_b"] + eps_rho

        grad_a_inp = feature_dict["norm_grad_a"]
        grad_b_inp = feature_dict["norm_grad_b"]
        grad_inp = feature_dict["norm_grad"]
        tau_a_inp = feature_dict["tau_a"]
        tau_b_inp = feature_dict["tau_b"]
        
        if with_lapl:
            lapl_a_inp = feature_dict["lapl_a"]
            lapl_b_inp = feature_dict["lapl_b"]

            constants = self.model(
                nn_inputs,
                rho_a_inp,
                rho_b_inp,
                grad_a_inp,
                grad_b_inp,
                grad_inp,
                tau_a_inp,
                tau_b_inp,
                lapl_a_inp,
                lapl_b_inp,
            )
        
        else:
            constants = self.model(
                nn_inputs,
                rho_a_inp,
                rho_b_inp,
                grad_a_inp,
                grad_b_inp,
                grad_inp,
                tau_a_inp,
                tau_b_inp,
            )
            

        functional_densities = torch.cat(
            [feature_dict[key] for key in keys[:2]], dim=0
        ).T
        functional_gradients = torch.cat(
            [
                feature_dict["norm_grad_a"],
                feature_dict["norm_grad_ab"],
                feature_dict["norm_grad_b"],
            ],
            dim=0,
        ).T

        if "PBE" in self.name:
            if "star_star" in self.name:
                vxc = F_PBE(
                    functional_densities,
                    functional_gradients,
                    true_constants_PBE,
                    "cpu",
                    torch.stack(constants, dim=1),
                )
            else:
                vxc = F_PBE(
                    functional_densities, functional_gradients, constants, "cpu"
                )
        else:
            vxc = F_XALPHA(functional_densities, constants)

        local_xc = vxc * (feature_dict["rho_a"] + feature_dict["rho_b"])

        unweighted_xc = torch.sum(local_xc, dim=1)

        vrho = (
            torch.stack(
                self.torch_grad(
                    unweighted_xc, [feature_dict["rho_a"], feature_dict["rho_b"]]
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        vsigma = (
            torch.stack(
                self.torch_grad(
                    unweighted_xc,
                    [
                        feature_dict["norm_grad_a"],
                        feature_dict["norm_grad_b"],
                        feature_dict["norm_grad"],
                    ],
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        vtau = (
            torch.stack(
                self.torch_grad(
                    unweighted_xc, [feature_dict["tau_a"], feature_dict["tau_b"]]
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        if with_lapl:
            vlapl = (
                torch.stack(
                    self.torch_grad(
                        unweighted_xc, [feature_dict["lapl_a"], feature_dict["lapl_b"]]
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )

        if spin == 0:
            vxc_0 = (vrho[0][0, :] + vrho[1][0, :]) / 2.0
            vxc_1 = vsigma[0][0, :] / 4.0 + vsigma[1][0, :] / 4.0 + vsigma[2][0, :]
            vxc_3 = (vtau[0][0, :] + vtau[1][0, :]) / 2.0
            
            if not with_lapl:
                vxc_2 = np.zeros_like(vxc_3)
            else:
                vxc_2 = (vlapl[0][0, :] + vlapl[1][0, :]) / 2.0

        else:
            vxc_0 = np.stack([vrho[0][0, :], vrho[1][0, :]], axis=1)
            vxc_1 = np.stack(
                [
                    vsigma[0][0, :] + vsigma[2][0, :],
                    2.0 * vsigma[2][0, :],
                    vsigma[1][0, :] + vsigma[2][0, :],
                ],
                axis=1,
            )
            vxc_3 = np.stack([vtau[0][0, :], vtau[1][0, :]], axis=1)

            if not with_lapl:
                vxc_2 = np.zeros_like(vxc_3)
            else:
                vxc_2 = np.stack([vlapl[0][0, :], vlapl[1][0, :]], axis=1)

        fxc = None  # Second derivative not implemented
        kxc = None  # Second derivative not implemented
        exc = vxc.detach().cpu().numpy().astype(np.float64)
        
        exc = numpy.nan_to_num(exc)
        vxc_0 = numpy.nan_to_num(vxc_0)
        vxc_1 = numpy.nan_to_num(vxc_1)
        vxc_2 = numpy.nan_to_num(vxc_2)
        vxc_3 = numpy.nan_to_num(vxc_3)

        return (
            exc,
            (
                vxc_0.astype(np.float64),
                vxc_1.astype(np.float64),
                vxc_2.astype(np.float64),
                vxc_3.astype(np.float64),
            ),
            fxc,
            kxc,
        )
