import sys
from pathlib import Path

import numpy as np
import torch

# Import from shared dft_functionals at project root
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
from DFT import PBE, SVWN3

F_PBE = PBE.F_PBE
F_XALPHA = SVWN3.F_XALPHA
f_svwn3 = SVWN3.f_svwn3


def compute_local_energy_tensors(reaction, constants, device, rung="GGA", dft="PBE", enhancement=None):
    densities = reaction["Densities"].to(device, non_blocking=True)
    if rung == "LDA":
        if dft == "SVWN3":
            local_energies = f_svwn3(densities, constants)
        if dft == "XALPHA":
            local_energies = F_XALPHA(densities, constants)
    elif rung == "GGA":
        gradients = (reaction["Gradients"]).to(device, non_blocking=True)
        if dft == "PBE":
            local_energies = F_PBE(densities, gradients, constants, device, enhancement=enhancement)
    weights = reaction["Weights"].to(device, non_blocking=True)
    return local_energies, densities, weights


def get_local_energies(reaction, constants, device, rung="GGA", dft="PBE", enhancement=None):
    calc_reaction_data = {}
    local_energies, densities, weights = compute_local_energy_tensors(
        reaction,
        constants,
        device,
        rung,
        dft,
        enhancement=enhancement,
    )
    calc_reaction_data["Local_energies"] = local_energies
    calc_reaction_data["Densities"] = densities
    calc_reaction_data["Weights"] = weights
    del local_energies, densities
    return calc_reaction_data


def backsplit(reaction, calc_reaction_data):
    return backsplit_tensors(
        reaction,
        calc_reaction_data["Local_energies"],
        calc_reaction_data["Weights"],
        calc_reaction_data["Densities"],
    )


def backsplit_tensors(reaction, local_energies, weights, densities):
    backsplit_ind = reaction["backsplit_ind"].type(torch.int)
    splitted_data = dict()
    stop = 0

    for i, component in enumerate(reaction["Components"]):
        splitted_data[component] = dict()
        start = stop
        stop = backsplit_ind[i]
        splitted_data[component]["Local_energies"] = local_energies[start:stop]
        splitted_data[component]["Weights"] = weights[start:stop]
        splitted_data[component]["Densities"] = densities[start:stop]
    del backsplit_ind, start, stop
    return splitted_data


def integrate_xc_energy(calc_reaction_data):
    """Integrate only the XC contribution over one unsplit grid."""
    return torch.sum(
        calc_reaction_data["Local_energies"]
        * (
            calc_reaction_data["Densities"][:, 0]
            + calc_reaction_data["Densities"][:, 1]
        )
        * calc_reaction_data["Weights"]
    )


def integration(
    reaction,
    splitted_calc_reaction_data,
    dispersions=None,
    add_hf_energies: bool = True,
    add_dispersion=None,
):
    if dispersions is None:
        dispersions = {}
    if add_dispersion is None:
        add_dispersion = add_hf_energies

    molecule_energies = dict()
    for i, component in enumerate(reaction["Components"]):
        molecule_energies[component + str(i)] = integrate_xc_energy(
            splitted_calc_reaction_data[component]
        )
        if add_hf_energies:
            molecule_energies[component + str(i)] += reaction["HF_energies"][i]
        if add_dispersion and dispersions:
            dispersion_val = torch.tensor(dispersions.get(component, 0), device=splitted_calc_reaction_data[component]["Local_energies"].device)
            molecule_energies[component + str(i)] += dispersion_val

    del splitted_calc_reaction_data
    return molecule_energies


def get_energy_reaction(reaction, molecule_energies):
    slices = reaction.get("reaction_indices", [0, len(reaction["Components"])])
    hartree2kcal = 627.5095
    reaction_energy_kcal = []
    for i in range(len(slices) - 1):
        s = 0
        for coef, ener in list(
            zip(reaction["Coefficients"], molecule_energies.values())
        )[slices[i] : slices[i + 1]]:
            s += coef * ener
        reaction_energy_kcal.append(s * hartree2kcal)
    del ener, coef, s, slices

    return torch.stack(reaction_energy_kcal)


def calculate_reaction_energy(
    reaction, constants, device, rung, dft, dispersions=None, enhancement=None, return_local_energies=True
):
    local_energies, densities, weights = compute_local_energy_tensors(
        reaction,
        constants,
        device,
        rung,
        dft,
        enhancement=enhancement,
    )
    if local_energies.isnan().any():
        print(local_energies.isnan().sum())
        torch.save(local_energies, "local_energies.pt")
        raise Exception()
    splitted_calc_reaction_data = backsplit_tensors(reaction, local_energies, weights, densities)
    molecule_energies = integration(reaction, splitted_calc_reaction_data, dispersions)
    reaction_energy_kcal = get_energy_reaction(reaction, molecule_energies)
    del molecule_energies, splitted_calc_reaction_data, densities, weights
    if return_local_energies:
        return reaction_energy_kcal, local_energies
    del local_energies
    return reaction_energy_kcal, None


def calculate_xc_energy(reaction, constants, device, rung, dft, enhancement=None):
    """Calculate only integrated E_xc for one unsplit grid; no HF, dispersion, or backsplit."""
    local_energies, densities, weights = compute_local_energy_tensors(
        reaction,
        constants,
        device,
        rung,
        dft,
        enhancement=enhancement,
    )
    if local_energies.isnan().any():
        print(local_energies.isnan().sum())
        torch.save(local_energies, "local_energies.pt")
        raise Exception()
    xc_energy = torch.sum(local_energies * (densities[:, 0] + densities[:, 1]) * weights)
    del densities, weights
    return xc_energy, local_energies


def test_energy_PBE(test_grid, constants):
    local_energies = F_PBE(test_grid["Densities"], test_grid["Gradients"], constants)
    local_scaled_energies = (
        local_energies
        * (test_grid["Densities"][:, 0] + test_grid["Densities"][:, 1])
        * (test_grid["Weights"])
    )
    return local_energies, local_scaled_energies
