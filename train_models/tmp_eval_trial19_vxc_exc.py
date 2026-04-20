import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from NN_models import pcPBELMLOptimizerV2
from reaction_energy_calculation import calculate_xc_energy, get_local_energies
from utils import (
    _fix_sigma_tot_closed_shell,
    _grid_to_model_input,
)

HARTREE2KCAL = 627.5095


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


def parse_model_name(name: str) -> Tuple[int, int, bool, bool]:
    prefix, num_layers, h_dim = name.split("_")
    use_g_x = "Gx" in prefix
    use_g_c = "Gc" in prefix
    return int(num_layers), int(h_dim), use_g_x, use_g_c


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    num_layers, h_dim, use_g_x, use_g_c = parse_model_name(args.name)
    return pcPBELMLOptimizerV2(
        num_layers=num_layers,
        h_dim=h_dim,
        dropout=args.dropout,
        DFT="PBE",
        use_g_x=use_g_x,
        use_g_c=use_g_c,
    ).to(device)


def load_state_dict_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("log_scale"):
            continue
        cleaned[clean_key] = value
    cleaned["scaling_array"] = model.scaling_array
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    allowed_missing: List[str] = []
    unexpected_without_allowed = [name for name in unexpected if not name.startswith("log_scale")]
    if missing != allowed_missing or unexpected_without_allowed:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch. Missing={missing}, Unexpected={unexpected_without_allowed}"
        )


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


def vxc_loss(
    model: nn.Module,
    X_batch: Dict[str, torch.Tensor],
    device: torch.device,
    rung: str = "GGA",
    dft: str = "PBE",
    create_graph: bool = True,
) -> torch.Tensor:
    grid_raw = X_batch["Grid"].to(device).clone().detach()
    rho = grid_raw[:, 4:6].clone().requires_grad_(True)
    sigma = grid_raw[:, 6:9].clone()
    sigma = _fix_sigma_tot_closed_shell(sigma)
    sigma_pbe = torch.stack(
        [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]], dim=1
    )
    target_vrho = X_batch["Vrho"].to(device)
    weights = X_batch["Weights"].to(device)

    model_input = _grid_to_model_input(grid_raw, fix_closed_shell_sigma=True)
    model_input[:, 0:2] = rho
    constants = model(model_input)

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
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    pred_vrho = (grads[:, 0] + grads[:, 1]) / 2.0

    rho_total_detached = rho_tot.detach()
    diff_sq = (pred_vrho - target_vrho) ** 2
    loss_integral = torch.sum(rho_total_detached * weights * diff_sq)
    norm_factor = torch.sum(rho_total_detached * weights)
    return loss_integral / (norm_factor + 1e-10)


def exc_loss(
    model: nn.Module,
    X_batch: Dict[str, torch.Tensor],
    device: torch.device,
    rung: str = "GGA",
    dft: str = "PBE",
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
        rho = grid_system[:, 4:6]
        sigma = _fix_sigma_tot_closed_shell(grid_system[:, 6:9].clone())
        sigma_pbe = torch.stack(
            [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]], dim=1
        )
        model_input = _grid_to_model_input(grid_system, fix_closed_shell_sigma=True)
        constants = model(model_input)
        pred_exc, _ = calculate_xc_energy(
            {"Densities": rho, "Gradients": sigma_pbe, "Weights": weights_system},
            constants,
            device,
            rung=rung,
            dft=dft,
            enhancement=None,
        )
        pred_exc_values.append(pred_exc)
        start = stop

    pred_exc_batch = torch.stack(pred_exc_values)
    loss = batch_exc(list(X_batch["Names"]), pred_exc_batch, target_exc)
    return loss, pred_exc_batch, target_exc


def evaluate_split(model, data_path, device, batch_size):
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
            loss_vxc = vxc_loss(model, batch, device, rung="GGA", dft="PBE", create_graph=False)
            loss_exc, pred_exc, ref_exc = exc_loss(model, batch, device, rung="GGA", dft="PBE")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-pickle", required=True)
    parser.add_argument("--val-pickle", required=True)
    parser.add_argument("--name", default="PBE-LGxGc_6_64")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="trial19_vxc_exc_eval.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = build_model(SimpleNamespace(name=args.name, dropout=0.0), device)
    load_state_dict_into_model(model, Path(args.checkpoint), device)
    model.eval()

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "train": evaluate_split(model, args.train_pickle, device, args.batch_size),
        "val": evaluate_split(model, args.val_pickle, device, args.batch_size),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
