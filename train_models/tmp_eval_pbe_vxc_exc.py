import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dft_functionals import true_constants_PBE
from reaction_energy_calculation import calculate_xc_energy, get_local_energies
from utils import _fix_sigma_tot_closed_shell

HARTREE2KCAL = 627.5095
DEFAULT_MRKS_DISPERSIONS = Path(__file__).resolve().parent / "dispersions" / "dispersions_mrks.pickle"


class VxcDataset(Dataset):
    def __init__(self, data_list: list) -> None:
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def vxc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "Grid": torch.cat([item["Grid"] for item in batch], dim=0),
        "Vrho": torch.cat([item["Vrho"] for item in batch], dim=0),
        "Weights": torch.cat([item["Weights"] for item in batch], dim=0),
        "E_xc": torch.stack([item["E_xc"] for item in batch], dim=0),
        "Names": [item["Name"] for item in batch],
        "GridLengths": torch.tensor([item["Grid"].shape[0] for item in batch], dtype=torch.long),
    }


def pbe_constants(n_points: int, device: torch.device) -> torch.Tensor:
    constants = true_constants_PBE.to(device).reshape(1, -1).repeat(n_points, 1)
    # The local PBE implementation uses indices 26 and 27 as additive neural
    # exchange corrections. Pure canonical PBE has no NN exchange correction.
    constants[:, 26:28] = 0.0
    return constants


def batch_exc(
    system_names: List[str],
    pred_exc: torch.Tensor,
    ref_exc: torch.Tensor,
) -> torch.Tensor:
    err_dict: Dict[str, List[List[torch.Tensor]]] = {}
    for system_name, pred, ref in zip(system_names, pred_exc, ref_exc):
        err_dict.setdefault(system_name, [[], []])
        err_dict[system_name][0].append(pred)
        err_dict[system_name][1].append(ref)

    values = []
    for preds, refs in err_dict.values():
        system_predictions = torch.stack(preds)
        system_ref = torch.stack(refs)
        mse = nn.functional.mse_loss(system_predictions, system_ref)
        values.append(torch.sqrt(1e-20 + mse))
    return HARTREE2KCAL * torch.sum(torch.stack(values)) / len(values)


def pbe_vxc_loss(
    X_batch: Dict[str, torch.Tensor],
    device: torch.device,
    rung: str = "GGA",
    dft: str = "PBE",
) -> torch.Tensor:
    grid_raw = X_batch["Grid"].to(device).clone().detach()
    rho = grid_raw[:, 4:6].clone().requires_grad_(True)
    sigma = _fix_sigma_tot_closed_shell(grid_raw[:, 6:9].clone())
    sigma_pbe = torch.stack(
        [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]], dim=1
    )
    target_vrho = X_batch["Vrho"].to(device)
    weights = X_batch["Weights"].to(device)
    constants = pbe_constants(rho.shape[0], device)

    calc_data = get_local_energies(
        {"Densities": rho, "Gradients": sigma_pbe, "Weights": weights},
        constants,
        device,
        rung=rung,
        dft=dft,
        enhancement=None,
    )

    rho_tot = rho[:, 0] + rho[:, 1]
    e_xc_pred = calc_data["Local_energies"] * rho_tot
    grads = torch.autograd.grad(
        outputs=e_xc_pred,
        inputs=rho,
        grad_outputs=torch.ones_like(e_xc_pred),
        create_graph=False,
        retain_graph=False,
    )[0]
    pred_vrho = (grads[:, 0] + grads[:, 1]) / 2.0

    diff_sq = (pred_vrho - target_vrho) ** 2
    loss_integral = torch.sum(rho_tot.detach() * weights * diff_sq)
    norm_factor = torch.sum(rho_tot.detach() * weights)
    return loss_integral / (norm_factor + 1e-10)


def pbe_exc_loss(
    X_batch: Dict[str, torch.Tensor],
    device: torch.device,
    rung: str = "GGA",
    dft: str = "PBE",
    dispersions: Dict[str, float] = None,
    include_mrks_dispersion: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_raw = X_batch["Grid"].to(device).clone().detach()
    weights = X_batch["Weights"].to(device)
    target_exc = X_batch["E_xc"].to(device)
    lengths = X_batch["GridLengths"].to(device)

    pred_exc_values = []
    start = 0
    for length in lengths.tolist():
        stop = start + int(length)
        grid_system = grid_raw[start:stop]
        weights_system = weights[start:stop]
        system_name = X_batch["Names"][len(pred_exc_values)]
        rho = grid_system[:, 4:6]
        sigma = _fix_sigma_tot_closed_shell(grid_system[:, 6:9].clone())
        sigma_pbe = torch.stack(
            [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]], dim=1
        )
        constants = pbe_constants(rho.shape[0], device)
        pred_exc, _ = calculate_xc_energy(
            {"Densities": rho, "Gradients": sigma_pbe, "Weights": weights_system},
            constants,
            device,
            rung=rung,
            dft=dft,
            enhancement=None,
            dispersions=dispersions,
            system_name=system_name,
            add_dispersion=include_mrks_dispersion,
        )
        pred_exc_values.append(pred_exc)
        start = stop

    pred_exc_batch = torch.stack(pred_exc_values)
    loss = batch_exc(list(X_batch["Names"]), pred_exc_batch, target_exc)
    return loss, pred_exc_batch, target_exc


def load_mrks_dispersions(path: str) -> Dict[str, float]:
    with Path(path).open("rb") as handle:
        raw = pickle.load(handle)
    return {key: float(value) for key, value in raw.items()}


def evaluate_split(
    data_path: str,
    device: torch.device,
    batch_size: int,
    dispersions: Dict[str, float] = None,
    include_mrks_dispersion: bool = False,
) -> Dict[str, Any]:
    with Path(data_path).open("rb") as handle:
        data = pickle.load(handle)

    loader = DataLoader(
        VxcDataset(data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vxc_collate_fn,
        num_workers=0,
    )

    vxc_values = []
    exc_abs_errors = []
    per_system = {}
    for batch in loader:
        with torch.enable_grad():
            loss_vxc = pbe_vxc_loss(batch, device, rung="GGA", dft="PBE")
            loss_exc, pred_exc, ref_exc = pbe_exc_loss(
                batch,
                device,
                rung="GGA",
                dft="PBE",
                dispersions=dispersions,
                include_mrks_dispersion=include_mrks_dispersion,
            )

        if not torch.isfinite(loss_vxc) or not torch.isfinite(loss_exc):
            raise RuntimeError(f"Non-finite loss for batch {batch['Names']}")

        vxc_values.append(float(loss_vxc.detach().cpu()))
        errors = (pred_exc.detach().cpu() - ref_exc.detach().cpu()).abs().numpy()
        for name, err in zip(batch["Names"], errors):
            err_value = float(err)
            exc_abs_errors.append(err_value)
            per_system[name] = err_value

    exc_arr = np.asarray(exc_abs_errors, dtype=np.float64)
    return {
        "n_systems": len(data),
        "vxc": float(np.mean(vxc_values)),
        "exc_loss_kcal": float(np.mean(exc_arr) * HARTREE2KCAL),
        "exc_mean_abs_ha": float(np.mean(exc_arr)),
        "exc_mean_abs_kcal": float(np.mean(exc_arr) * HARTREE2KCAL),
        "exc_rmse_ha": float(np.sqrt(np.mean(np.square(exc_arr)))),
        "exc_rmse_kcal": float(np.sqrt(np.mean(np.square(exc_arr))) * HARTREE2KCAL),
        "exc_max_abs_ha": float(np.max(exc_arr)),
        "exc_max_abs_kcal": float(np.max(exc_arr) * HARTREE2KCAL),
        "exc_per_system_abs_ha": dict(sorted(per_system.items())),
        "exc_per_system_abs_kcal": {
            name: value * HARTREE2KCAL for name, value in sorted(per_system.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pickle", required=True)
    parser.add_argument("--val-pickle", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="pbe_new_vxc_exc_eval.json")
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument("--mrks-dispersions-pickle", default=str(DEFAULT_MRKS_DISPERSIONS))
    args = parser.parse_args()

    device = torch.device("cpu")
    dispersions = load_mrks_dispersions(args.mrks_dispersions_pickle) if args.include_mrks_dispersion else None
    result = {
        "functional": "PBE",
        "constant_overrides": {
            "26_G_NN_up": 0.0,
            "27_G_NN_down": 0.0,
            "28_G_c": 1.0,
        },
        "include_mrks_dispersion": bool(args.include_mrks_dispersion),
        "train": evaluate_split(
            args.train_pickle, device, args.batch_size, dispersions=dispersions, include_mrks_dispersion=args.include_mrks_dispersion
        ),
        "val": evaluate_split(
            args.val_pickle, device, args.batch_size, dispersions=dispersions, include_mrks_dispersion=args.include_mrks_dispersion
        ),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
