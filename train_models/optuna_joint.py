import argparse
import collections
import copy
import json
import math
import os
import pickle
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dataset import collate_fn, fast_collate_fn_predopt
from NN_models import (
    pcPBELMLOptimizerV2,
    pcPBELMLOptimizerV2GcSoftplusMirror,
    pcPBELMLOptimizerV2GcSveluMirror,
    pcPBELMLOptimizerV2Log,
)
from predopt import DatasetPredopt, predopt
from prepare_data import load_chk
from reaction_energy_calculation import calculate_reaction_energy, calculate_xc_energy, get_local_energies
from utils import (
    _fix_sigma_tot_closed_shell,
    _grid_to_model_input,
    configure_optimizers,
    seed_worker,
    set_random_seed,
)

EPS = 1e-10
OMEGA = 0.5
FAIL_VALUE = 1.0e12
WARMUP_EPOCHS = 5
WARMUP_START_FACTOR = 0.001
MIN_LR = 1e-6
RUN_TRIAL = "RUN"
STOP_TRIALS = "STOP"
FAIL_TRIAL = "FAIL"

FCHEM_VALIDATION = {
    "ABDE4": 1,
    "AE17": 1,
    "DBH76": 1,
    "EA13": 1,
    "IP13": 1,
    "MGAE109": 1 / 4.73394495412844,
    "NCCE31": 10,
    "PA8": 1,
    "pTC13": 1,
}

FREQ_WEIGHTS = {
    "ABDE4": 1 / 4,
    "AE17": 1 / 17,
    "DBH76": 1 / 76,
    "EA13": 1 / 13,
    "IP13": 1 / 13,
    "MGAE109": 1 / 109,
    "NCCE31": 1 / 31,
    "PA8": 1 / 8,
    "pTC13": 1 / 13,
}

MEAN_WEIGHT = sum(
    FCHEM_VALIDATION[db] * FREQ_WEIGHTS[db] for db in FCHEM_VALIDATION
) / len(FCHEM_VALIDATION)

HARTREE2KCAL = 627.5095
DEFAULT_MRKS_DISPERSIONS = Path(__file__).resolve().parent / "dispersions" / "dispersions_mrks.pickle"


class VxcDataset(Dataset):
    def __init__(self, data_list: list) -> None:
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def variant_suffix_from_path(path: str) -> str:
    file_name_no_ext = Path(path).stem
    if "__" in file_name_no_ext:
        _, suffix = file_name_no_ext.split("__", 1)
        return suffix
    return "default"


def variant_suffix_from_reaction(reaction: Dict[str, Any]) -> str:
    component_paths = reaction.get("component_paths", [])
    if not component_paths:
        raise ValueError("Reaction variant is missing component_paths.")
    suffixes = {variant_suffix_from_path(path) for path in component_paths}
    if len(suffixes) != 1:
        raise ValueError(f"Inconsistent augmentation suffixes detected: {sorted(suffixes)}")
    return next(iter(suffixes))


def canonical_variant(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    suffix_map = {variant_suffix_from_reaction(reaction): reaction for reaction in group}
    chosen_suffix = "default" if "default" in suffix_map else min(suffix_map)
    return suffix_map[chosen_suffix]


class EpochSampledAugmentedDataset(Dataset):
    def __init__(self, data: dict, base_seed: int) -> None:
        self.reaction_groups = list(data.values())
        self.base_seed = base_seed
        self.sampled_reactions: List[Dict[str, Any]] = []
        self.resample(epoch=0)

    def resample(self, epoch: int) -> None:
        generator = random.Random(self.base_seed + epoch)
        self.sampled_reactions = [
            group[generator.randrange(len(group))]
            for group in self.reaction_groups
        ]

    def __len__(self) -> int:
        return len(self.sampled_reactions)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        chosen = self.sampled_reactions[idx]
        return copy.deepcopy(chosen), chosen["Energy"]


class CanonicalVariantDataset(Dataset):
    def __init__(self, data: dict) -> None:
        self.reactions = [canonical_variant(group) for group in data.values()]

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        chosen = self.reactions[idx]
        return copy.deepcopy(chosen), chosen["Energy"]


def vxc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "Grid": torch.cat([item["Grid"] for item in batch], dim=0),
        "Vrho": torch.cat([item["Vrho"] for item in batch], dim=0),
        "Weights": torch.cat([item["Weights"] for item in batch], dim=0),
        "E_xc": torch.stack([item["E_xc"] for item in batch], dim=0),
        "Names": [item["Name"] for item in batch],
        "GridLengths": torch.tensor([item["Grid"].shape[0] for item in batch], dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone DDP Optuna search for joint piNN-DFT training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_64")
    parser.add_argument("--model-type", type=str, default="base", choices=["base", "log", "gc_svelu_mirror", "gc_softplus_mirror"])
    parser.add_argument("--n-predopt", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vxc-batch-size", type=int, default=1)
    parser.add_argument("--lr-predopt", type=float, default=2e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--save-selected-checkpoints", action="store_true")
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-vxc", type=int, default=2)
    parser.add_argument("--preopt-vxc-weight", type=float, default=0.0)
    parser.add_argument("--preopt-vxc-steps", type=int, default=0)
    parser.add_argument("--preopt-vxc-target", type=str, default="pbe", choices=["pbe"])
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument("--mrks-dispersions-pickle", type=str, default=str(DEFAULT_MRKS_DISPERSIONS))
    parser.add_argument(
        "--no-reaction-dispersion",
        action="store_true",
        help="Do not add precomputed D3 dispersion corrections in reaction-energy training/validation.",
    )
    return parser.parse_args()


def init_distributed() -> Tuple[int, int, torch.device, bool]:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return local_rank, world_size, device, dist.get_rank() == 0


def sync_failure(local_failed: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if local_failed else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def broadcast_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    object_list = [payload]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def gather_object(payload: Any, world_size: int) -> List[Any]:
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)
    return gathered


def parse_model_name(name: str) -> Tuple[int, int, bool, bool]:
    prefix, num_layers, h_dim = name.split("_")
    use_g_x = "Gx" in prefix
    use_g_c = "Gc" in prefix
    return int(num_layers), int(h_dim), use_g_x, use_g_c


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    num_layers, h_dim, use_g_x, use_g_c = parse_model_name(args.name)
    model_type = getattr(args, "model_type", "base")
    model_classes = {
        "base": pcPBELMLOptimizerV2,
        "log": pcPBELMLOptimizerV2Log,
        "gc_svelu_mirror": pcPBELMLOptimizerV2GcSveluMirror,
        "gc_softplus_mirror": pcPBELMLOptimizerV2GcSoftplusMirror,
    }
    model_cls = model_classes[model_type]
    return model_cls(
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
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("log_scale"):
            continue
        cleaned[clean_key] = value
    cleaned["scaling_array"] = model.scaling_array
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    allowed_missing: List[str] = []
    allowed_unexpected = {name for name in unexpected if name.startswith("log_scale")}
    unexpected_without_allowed = [name for name in unexpected if name not in allowed_unexpected]
    if missing != allowed_missing or unexpected_without_allowed:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch. Missing={missing}, Unexpected={unexpected_without_allowed}"
        )


def load_mrks_dispersions(path: str) -> Dict[str, float]:
    with Path(path).open("rb") as handle:
        raw = pickle.load(handle)
    return {key: float(value) for key, value in raw.items()}


def build_scheduler(optimizer: torch.optim.Optimizer, n_train: int):
    warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(n_train - WARMUP_EPOCHS, 1),
        eta_min=MIN_LR,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])


def build_reaction_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int,
    shuffle: bool,
) -> Tuple[DataLoader, DistributedSampler]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader, sampler


def build_vxc_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int,
    shuffle: bool,
) -> Tuple[DataLoader, DistributedSampler]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vxc_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader, sampler


def build_preopt_loader(
    data_predopt: dict,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = DatasetPredopt(data_predopt)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=fast_collate_fn_predopt,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def build_dataloaders(
    data_train: dict,
    data_val: dict,
    data_vxc_train: list,
    data_vxc_val: list,
    trial_seed: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    train_set = EpochSampledAugmentedDataset(data_train, base_seed=trial_seed)
    val_set = CanonicalVariantDataset(data_val)
    vxc_train_set = VxcDataset(data_vxc_train)
    vxc_val_set = VxcDataset(data_vxc_val)

    train_loader, train_sampler = build_reaction_loader(
        train_set,
        args.batch_size,
        trial_seed,
        rank,
        world_size,
        args.num_workers_train,
        shuffle=True,
    )
    val_loader, val_sampler = build_reaction_loader(
        val_set,
        args.batch_size,
        trial_seed,
        rank,
        world_size,
        args.num_workers_train,
        shuffle=False,
    )
    vxc_train_loader, vxc_train_sampler = build_vxc_loader(
        vxc_train_set,
        args.vxc_batch_size,
        trial_seed,
        rank,
        world_size,
        args.num_workers_vxc,
        shuffle=True,
    )
    vxc_val_loader, vxc_val_sampler = build_vxc_loader(
        vxc_val_set,
        args.vxc_batch_size,
        trial_seed,
        rank,
        world_size,
        args.num_workers_vxc,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "train_sampler": train_sampler,
        "val_loader": val_loader,
        "val_sampler": val_sampler,
        "vxc_train_loader": vxc_train_loader,
        "vxc_train_sampler": vxc_train_sampler,
        "vxc_val_loader": vxc_val_loader,
        "vxc_val_sampler": vxc_val_sampler,
    }


def batch_fchem(
    current_bases: List[str],
    reaction_energy: torch.Tensor,
    y_batch: torch.Tensor,
) -> torch.Tensor:
    err_dict: Dict[str, List[List[torch.Tensor]]] = {}
    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict.setdefault(database, [[], []])
        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)

    values = []
    for database, (preds, refs) in err_dict.items():
        db_predictions = torch.stack(preds)
        db_ref = torch.stack(refs)
        factor = FCHEM_VALIDATION.get(database, 1) * FREQ_WEIGHTS.get(database, 1) / MEAN_WEIGHT
        mse = nn.functional.mse_loss(db_predictions, db_ref)
        values.append(factor * torch.sqrt(1e-20 + mse))
    return torch.sum(torch.stack(values)) / len(values)


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


def update_db_errors(
    total_database_errors: Dict[str, List[float]],
    current_bases: List[str],
    reaction_energy: torch.Tensor,
    y_batch: torch.Tensor,
) -> None:
    for base, error in zip(current_bases, reaction_energy.detach() - y_batch.detach()):
        total_database_errors.setdefault(base, [])
        total_database_errors[base].append(float(torch.abs(error).item()))


def update_exc_errors(
    total_exc_errors: Dict[str, List[float]],
    system_names: List[str],
    pred_exc: torch.Tensor,
    ref_exc: torch.Tensor,
) -> None:
    for system_name, error in zip(system_names, pred_exc.detach() - ref_exc.detach()):
        total_exc_errors.setdefault(system_name, [])
        total_exc_errors[system_name].append(float(torch.abs(error).item()) * HARTREE2KCAL)


def compute_fchem_from_errors(total_database_errors: Dict[str, List[float]]) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    per_db = {}
    for db in sorted(total_database_errors):
        errors = np.asarray(total_database_errors[db], dtype=np.float64)
        if errors.size == 0:
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        per_db[db] = rmse
        total += FCHEM_VALIDATION.get(db, 1) * rmse
    return total, per_db


def compute_exc_from_errors(total_exc_errors: Dict[str, List[float]]) -> Tuple[float, Dict[str, float]]:
    per_system = {}
    values = []
    for system_name in sorted(total_exc_errors):
        errors = np.asarray(total_exc_errors[system_name], dtype=np.float64)
        if errors.size == 0:
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        per_system[system_name] = rmse
        values.append(rmse)
    if not values:
        return 0.0, per_system
    return float(np.mean(values)), per_system


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

    model_input = torch.cat([rho, sigma, grid_raw[:, 9:]], dim=1)
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
    dispersions: Optional[Dict[str, float]] = None,
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
        model_input = torch.cat([rho, sigma, grid_system[:, 9:]], dim=1)
        constants = model(model_input)
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


def get_trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    base_model = model.module if hasattr(model, "module") else model
    return [param for param in base_model.parameters() if param.requires_grad]


def grads_are_finite(parameters: List[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def compute_grad_list(
    loss: torch.Tensor,
    parameters: List[torch.nn.Parameter],
) -> List[Optional[torch.Tensor]]:
    return list(torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True))


def clip_gradient_list_by_global_norm(
    grads: List[Optional[torch.Tensor]],
    max_norm: Optional[float],
) -> Tuple[List[Optional[torch.Tensor]], float]:
    norm_sq = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_detached = grad.detach()
        norm_sq += float(torch.sum(grad_detached * grad_detached).item())
    total_norm = math.sqrt(max(norm_sq, 0.0))
    if max_norm is None or total_norm <= max_norm or total_norm <= EPS:
        return [None if grad is None else grad.detach() for grad in grads], total_norm

    scale = max_norm / (total_norm + EPS)
    clipped = []
    for grad in grads:
        if grad is None:
            clipped.append(None)
        else:
            clipped.append(grad.detach() * scale)
    return clipped, total_norm


def scale_gradient_list(
    grads: List[Optional[torch.Tensor]],
    scale: float,
) -> List[Optional[torch.Tensor]]:
    scaled: List[Optional[torch.Tensor]] = []
    for grad in grads:
        if grad is None:
            scaled.append(None)
        else:
            scaled.append(grad.detach() * scale)
    return scaled


def add_gradient_list_to_parameters(
    parameters: List[torch.nn.Parameter],
    grads: List[Optional[torch.Tensor]],
) -> None:
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            continue
        if parameter.grad is None:
            parameter.grad = grad.detach().clone()
        else:
            parameter.grad.add_(grad.detach())


def allreduce_parameter_grads(parameters: List[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def resolve_objective_params(params: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(params)
    global_strategy = resolved.get("gradient_merge_strategy", "sum")
    resolved.setdefault("reaction_gradient_merge_strategy", global_strategy)
    resolved.setdefault("vxc_gradient_merge_strategy", global_strategy)
    resolved.setdefault("exc_gradient_merge_strategy", global_strategy)
    resolved.setdefault("exc_loss_scale", 0.0)
    resolved.setdefault("exc_grad_clip", "none")
    resolved.setdefault("exc_grad_scale", 1.0)
    return resolved


def prepare_objective_gradients(
    parameters: List[torch.nn.Parameter],
    weighted_loss: torch.Tensor,
    merge_strategy: str,
    grad_clip,
    grad_scale: float = 1.0,
) -> List[Optional[torch.Tensor]]:
    grads = compute_grad_list(weighted_loss, parameters)
    if merge_strategy == "clip_then_sum":
        clip_value = None if grad_clip == "none" else float(grad_clip)
        grads, _ = clip_gradient_list_by_global_norm(grads, clip_value)
    elif merge_strategy != "sum":
        raise ValueError(f"Unsupported gradient merge strategy: {merge_strategy}")
    return scale_gradient_list(grads, float(grad_scale))


def add_objective_gradients(
    parameters: List[torch.nn.Parameter],
    weighted_loss: torch.Tensor,
    merge_strategy: str,
    grad_clip,
    grad_scale: float = 1.0,
) -> None:
    grads = prepare_objective_gradients(
        parameters,
        weighted_loss,
        merge_strategy,
        grad_clip,
        grad_scale,
    )
    add_gradient_list_to_parameters(parameters, grads)


def suggest_params(trial: Any) -> Dict[str, Any]:
    return {
        "lr_train": trial.suggest_float("lr_train", 1e-4, 1e-3, log=True),
        "accum_iter": trial.suggest_categorical("accum_iter", [1, 2, 3]),
        "vxc_loss_scale": trial.suggest_categorical("vxc_loss_scale", [50, 100, 150, 200]),
        "exc_loss_scale": trial.suggest_categorical("exc_loss_scale", [0.0, 1.0, 5.0, 10.0]),
        "reaction_grad_scale": trial.suggest_categorical(
            "reaction_grad_scale", [0.1, 0.3, 0.5, 0.7, 1.0]
        ),
        "reaction_grad_clip": trial.suggest_categorical(
            "reaction_grad_clip", ["none", 100.0, 300.0, 1000.0]
        ),
        "vxc_grad_clip": trial.suggest_categorical("vxc_grad_clip", [1.0, 2.0, 3.0, 5.0]),
        "exc_grad_clip": trial.suggest_categorical("exc_grad_clip", ["none", 1.0, 2.0, 5.0]),
        "exc_grad_scale": trial.suggest_categorical("exc_grad_scale", [0.1, 0.3, 0.5, 1.0]),
        "gradient_merge_strategy": trial.suggest_categorical(
            "gradient_merge_strategy", ["sum", "clip_then_sum"]
        ),
        "exc_gradient_merge_strategy": trial.suggest_categorical(
            "exc_gradient_merge_strategy", ["sum", "clip_then_sum"]
        ),
    }


def resolve_shared_preopt_checkpoint(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path]:
    checkpoint_path = Path(args.shared_preopt_checkpoint) if args.shared_preopt_checkpoint else output_dir / "preoptimized_checkpoint.pt"
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    return checkpoint_path, metadata_path


def preopt_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "name": args.name,
        "model_type": getattr(args, "model_type", "base"),
        "dropout": args.dropout,
        "n_predopt": args.n_predopt,
        "lr_predopt": args.lr_predopt,
        "batch_size": args.batch_size,
        "preopt_vxc_weight": args.preopt_vxc_weight,
        "preopt_vxc_steps": args.preopt_vxc_steps,
        "preopt_vxc_target": args.preopt_vxc_target,
    }


def run_or_reuse_preoptimization(
    args: argparse.Namespace,
    output_dir: Path,
    data_predopt: dict,
    data_vxc_train: list,
    device: torch.device,
    local_rank: int,
    world_size: int,
    rank0: bool,
) -> Path:
    checkpoint_path, metadata_path = resolve_shared_preopt_checkpoint(args, output_dir)
    metadata = preopt_metadata(args)

    decided_path: Optional[str] = None
    if rank0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        reuse = False
        if checkpoint_path.exists() and metadata_path.exists() and not args.force_preopt:
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                reuse = existing == metadata
            except Exception:
                reuse = False
        if reuse:
            print(f"Reusing shared preoptimized checkpoint: {checkpoint_path}")
        else:
            print(f"Running shared preoptimization and saving to: {checkpoint_path}")
        decided_path = str(checkpoint_path)
        payload = {"path": decided_path, "reuse": reuse}
    else:
        payload = None

    payload = broadcast_payload(payload)
    checkpoint_path = Path(payload["path"])

    if payload["reuse"]:
        dist.barrier()
        return checkpoint_path

    set_random_seed(args.seed)
    model = build_model(args, device)
    model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    )

    preopt_loader = build_preopt_loader(
        data_predopt=data_predopt,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=local_rank,
        world_size=world_size,
        num_workers=max(1, args.num_workers_vxc),
    )
    vxc_loader = None
    if args.preopt_vxc_weight > 0.0 and args.preopt_vxc_steps > 0 and data_vxc_train:
        vxc_dataset = VxcDataset(data_vxc_train)
        vxc_loader, _ = build_vxc_loader(
            dataset=vxc_dataset,
            batch_size=args.vxc_batch_size,
            seed=args.seed,
            rank=local_rank,
            world_size=world_size,
            num_workers=args.num_workers_vxc,
            shuffle=True,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_predopt, betas=(0.9, 0.999))
    predopt(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        train_loader=preopt_loader,
        device=device,
        n_epochs=args.n_predopt,
        accum_iter=1,
        local_rank=local_rank,
        vxc_loader=vxc_loader,
        preopt_vxc_weight=args.preopt_vxc_weight,
        preopt_vxc_steps=args.preopt_vxc_steps,
        vxc_target_mode=args.preopt_vxc_target,
        rung="GGA",
        dft="PBE",
    )

    if rank0:
        torch.save(model.module.state_dict(), checkpoint_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
    dist.barrier()
    del model, optimizer, preopt_loader, vxc_loader
    return checkpoint_path


def train_one_epoch(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    vxc_train_loader: DataLoader,
    params: Dict[str, Any],
    device: torch.device,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    include_mrks_dispersion: bool,
    world_size: int,
    epoch: int,
) -> Tuple[Dict[str, float], Dict[str, List[float]], bool]:
    params = resolve_objective_params(params)
    model.train()
    trainable_parameters = get_trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)

    train_db_errors: Dict[str, List[float]] = collections.defaultdict(list)
    train_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    n_train = len(train_loader)
    n_vxc = len(vxc_train_loader)
    if n_train == 0 or n_vxc == 0:
        raise ValueError("Both train_loader and vxc_train_loader must be non-empty.")

    n_steps = max(n_train, n_vxc)
    train_iter = iter(train_loader)
    vxc_iter = iter(vxc_train_loader)

    loss_sum = 0.0
    reaction_loss_sum = 0.0
    vxc_loss_sum = 0.0
    exc_loss_sum = 0.0
    mae_sum = 0.0
    optimizer_steps = 0
    failed = False

    for batch_idx in range(n_steps):
        try:
            reaction_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            reaction_batch, y_batch = next(train_iter)

        try:
            X_vxc = next(vxc_iter)
        except StopIteration:
            vxc_iter = iter(vxc_train_loader)
            X_vxc = next(vxc_iter)

        current_bases = list(reaction_batch["Database"])
        grid = reaction_batch["Grid"].to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        do_step = ((batch_idx + 1) % params["accum_iter"] == 0) or ((batch_idx + 1) == n_steps)
        strategies = [
            params["reaction_gradient_merge_strategy"],
            params["vxc_gradient_merge_strategy"],
            params["exc_gradient_merge_strategy"],
        ]
        manual_merge = True
        predictions = model(grid)
        reaction_energy, _ = calculate_reaction_energy(
            reaction_batch,
            predictions,
            device,
            rung="GGA",
            dft="PBE",
            dispersions=dispersions,
            return_local_energies=False,
        )
        reaction_loss = batch_fchem(current_bases, reaction_energy, y_batch)
        weighted_reaction_loss = reaction_loss / params["accum_iter"]
        microbatch_failed = (
            not torch.isfinite(reaction_energy).all()
            or not torch.isfinite(weighted_reaction_loss)
        )
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        add_gradient_list_to_parameters(
            trainable_parameters,
            prepare_objective_gradients(
                trainable_parameters,
                weighted_reaction_loss,
                params["reaction_gradient_merge_strategy"],
                params["reaction_grad_clip"],
                params["reaction_grad_scale"],
            ),
        )

        vxc_term = vxc_loss(model, X_vxc, device, rung="GGA", dft="PBE", create_graph=True)
        weighted_vxc_loss = OMEGA * params["vxc_loss_scale"] * vxc_term / params["accum_iter"]
        microbatch_failed = not torch.isfinite(weighted_vxc_loss)
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        add_gradient_list_to_parameters(
            trainable_parameters,
            prepare_objective_gradients(
                trainable_parameters,
                weighted_vxc_loss,
                params["vxc_gradient_merge_strategy"],
                params["vxc_grad_clip"],
                1.0,
            ),
        )

        exc_term, pred_exc, ref_exc = exc_loss(
            model,
            X_vxc,
            device,
            rung="GGA",
            dft="PBE",
            dispersions=mrks_dispersions,
            include_mrks_dispersion=include_mrks_dispersion,
        )
        weighted_exc_loss = params["exc_loss_scale"] * exc_term / params["accum_iter"]
        full_loss = weighted_reaction_loss + weighted_vxc_loss + weighted_exc_loss
        microbatch_failed = (
            not torch.isfinite(pred_exc).all()
            or not torch.isfinite(full_loss)
        )
        microbatch_failed = sync_failure(microbatch_failed, device)
        if microbatch_failed:
            failed = True
            break
        if float(params["exc_loss_scale"]) != 0.0:
            add_gradient_list_to_parameters(
                trainable_parameters,
                prepare_objective_gradients(
                    trainable_parameters,
                    weighted_exc_loss,
                    params["exc_gradient_merge_strategy"],
                    params["exc_grad_clip"],
                    params["exc_grad_scale"],
                ),
            )

        update_db_errors(train_db_errors, current_bases, reaction_energy, y_batch)
        update_exc_errors(train_exc_errors, list(X_vxc["Names"]), pred_exc, ref_exc)
        loss_sum += float((
            reaction_loss
            + OMEGA * params["vxc_loss_scale"] * vxc_term
            + params["exc_loss_scale"] * exc_term
        ).item())
        reaction_loss_sum += float(reaction_loss.item())
        vxc_loss_sum += float(vxc_term.item())
        exc_loss_sum += float(exc_term.item())
        mae_sum += float(nn.functional.l1_loss(reaction_energy, y_batch).item())
        del full_loss
        del weighted_reaction_loss, weighted_vxc_loss, weighted_exc_loss
        del reaction_loss, vxc_term, exc_term, reaction_energy, pred_exc, ref_exc
        del predictions, grid, y_batch

        if not do_step:
            continue

        if manual_merge:
            allreduce_parameter_grads(trainable_parameters, world_size)

        step_failed = not grads_are_finite(trainable_parameters)
        step_failed = sync_failure(step_failed, device)
        if step_failed:
            failed = True
            break

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    if failed:
        optimizer.zero_grad(set_to_none=True)
        return {}, {}, True

    scalar_tensor = torch.tensor(
        [loss_sum, reaction_loss_sum, vxc_loss_sum, exc_loss_sum, mae_sum, float(n_steps), float(optimizer_steps)],
        device=device,
    )
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

    gathered_errors = gather_object(dict(train_db_errors), world_size)
    global_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_errors:
        for db, errs in local_dict.items():
            global_errors[db].extend(errs)
    train_fchem, train_per_db = compute_fchem_from_errors(global_errors)

    gathered_exc_errors = gather_object(dict(train_exc_errors), world_size)
    global_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_exc_errors:
        for system_name, errs in local_dict.items():
            global_exc_errors[system_name].extend(errs)
    train_exc, train_per_system_exc = compute_exc_from_errors(global_exc_errors)

    metrics = {
        "train_full_loss": float(scalar_tensor[0].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_reaction_loss": float(scalar_tensor[1].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_vxc": float(scalar_tensor[2].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_exc_loss": float(scalar_tensor[3].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_mae": float(scalar_tensor[4].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_fchem": train_fchem,
        "train_exc": train_exc,
        "train_per_system_exc_rmse": train_per_system_exc,
        "optimizer_steps": int(scalar_tensor[6].item()),
    }
    return metrics, dict(train_per_db), False


def validate_one_epoch(
    model: DDP,
    val_loader: DataLoader,
    vxc_val_loader: DataLoader,
    params: Dict[str, Any],
    device: torch.device,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    include_mrks_dispersion: bool,
    world_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    params = resolve_objective_params(params)
    model.eval()

    val_db_errors: Dict[str, List[float]] = collections.defaultdict(list)
    val_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    n_val = len(val_loader)
    n_vxc = len(vxc_val_loader)
    if n_val == 0 or n_vxc == 0:
        raise ValueError("Both val_loader and vxc_val_loader must be non-empty.")

    val_iter = iter(val_loader)
    vxc_iter = iter(vxc_val_loader)
    n_steps = max(n_val, n_vxc)

    reaction_loss_sum = 0.0
    vxc_loss_sum_value = 0.0
    exc_loss_sum_value = 0.0
    full_loss_sum = 0.0
    mae_sum = 0.0
    sample_count = 0
    vxc_step_count = 0
    failed = False

    for _ in range(n_steps):
        try:
            reaction_batch, y_batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            reaction_batch, y_batch = next(val_iter)

        try:
            X_vxc = next(vxc_iter)
        except StopIteration:
            vxc_iter = iter(vxc_val_loader)
            X_vxc = next(vxc_iter)

        current_bases = list(reaction_batch["Database"])
        y_batch = y_batch.to(device, non_blocking=True)
        grid = reaction_batch["Grid"].to(device, non_blocking=True)

        with torch.no_grad():
            predictions = model(grid)
            reaction_energy, _ = calculate_reaction_energy(
                reaction_batch,
                predictions,
                device,
                rung="GGA",
                dft="PBE",
                dispersions=dispersions,
                return_local_energies=False,
            )
            reaction_loss = batch_fchem(current_bases, reaction_energy, y_batch)
            mae = nn.functional.l1_loss(reaction_energy, y_batch)

        with torch.enable_grad():
            loss_vxc_val = vxc_loss(model, X_vxc, device, rung="GGA", dft="PBE", create_graph=False)
            loss_exc_val, pred_exc, ref_exc = exc_loss(
                model,
                X_vxc,
                device,
                rung="GGA",
                dft="PBE",
                dispersions=mrks_dispersions,
                include_mrks_dispersion=include_mrks_dispersion,
            )

        batch_failed = (
            not torch.isfinite(reaction_energy).all()
            or not torch.isfinite(loss_vxc_val)
            or not torch.isfinite(pred_exc).all()
            or not torch.isfinite(loss_exc_val)
        )
        batch_failed = sync_failure(batch_failed, device)
        if batch_failed:
            failed = True
            break

        update_db_errors(val_db_errors, current_bases, reaction_energy, y_batch)
        update_exc_errors(val_exc_errors, list(X_vxc["Names"]), pred_exc, ref_exc)
        curr_batch_size = int(y_batch.size(0))
        reaction_loss_sum += float(reaction_loss.item()) * curr_batch_size
        vxc_loss_sum_value += float(loss_vxc_val.item())
        exc_loss_sum_value += float(loss_exc_val.item())
        full_loss_sum += float((
            reaction_loss
            + OMEGA * params["vxc_loss_scale"] * loss_vxc_val
            + params["exc_loss_scale"] * loss_exc_val
        ).item()) * curr_batch_size
        mae_sum += float(mae.item()) * curr_batch_size
        sample_count += curr_batch_size
        vxc_step_count += 1

    if failed:
        return {}, {}, True

    scalar_tensor = torch.tensor(
        [
            reaction_loss_sum,
            vxc_loss_sum_value,
            exc_loss_sum_value,
            full_loss_sum,
            mae_sum,
            float(sample_count),
            float(vxc_step_count),
        ],
        device=device,
    )
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

    gathered_errors = gather_object(dict(val_db_errors), world_size)
    global_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_errors:
        for db, errs in local_dict.items():
            global_errors[db].extend(errs)
    val_fchem, val_per_db = compute_fchem_from_errors(global_errors)

    gathered_exc_errors = gather_object(dict(val_exc_errors), world_size)
    global_exc_errors: Dict[str, List[float]] = collections.defaultdict(list)
    for local_dict in gathered_exc_errors:
        for system_name, errs in local_dict.items():
            global_exc_errors[system_name].extend(errs)
    val_exc, val_per_system_exc = compute_exc_from_errors(global_exc_errors)

    total_samples = max(int(scalar_tensor[5].item()), 1)
    total_vxc_steps = max(int(scalar_tensor[6].item()), 1)
    metrics = {
        "val_reaction_loss": float(scalar_tensor[0].item() / total_samples),
        "val_vxc": float(scalar_tensor[1].item() / total_vxc_steps),
        "val_exc_loss": float(scalar_tensor[2].item() / total_vxc_steps),
        "val_full_loss": float(scalar_tensor[3].item() / total_samples),
        "val_mae": float(scalar_tensor[4].item() / total_samples),
        "val_fchem": val_fchem,
        "val_exc": val_exc,
        "val_per_system_exc_rmse": val_per_system_exc,
    }
    metrics["val_joint_score"] = max(metrics["val_fchem"] / 50.0, metrics["val_vxc"] / 1.0)
    return metrics, dict(val_per_db), False


def select_representative_epoch(epoch_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not epoch_history:
        raise ValueError("Cannot select representative epoch from empty history.")
    return min(
        epoch_history,
        key=lambda row: (
            row["val_joint_score"],
            row["val_fchem"],
            row["val_vxc"],
            row["epoch"],
        ),
    )


def resolve_epoch_params(
    params: Dict[str, Any],
    epoch_number: int,
    n_train: int,
) -> Dict[str, Any]:
    schedule = params.get("epoch_schedule")
    if not schedule:
        resolved = dict(params)
        resolved.setdefault("phase_name", "static")
        resolved.setdefault("phase_start_epoch", 1)
        resolved.setdefault("phase_end_epoch", n_train)
        return resolved

    for index, phase in enumerate(schedule):
        start_epoch = int(phase.get("start_epoch", 1))
        end_epoch = int(phase.get("end_epoch", n_train))
        if start_epoch <= epoch_number <= end_epoch:
            resolved = {key: value for key, value in params.items() if key != "epoch_schedule"}
            resolved.update(dict(phase.get("params", {})))
            resolved["phase_name"] = str(phase.get("name", f"phase_{index + 1}"))
            resolved["phase_start_epoch"] = start_epoch
            resolved["phase_end_epoch"] = end_epoch
            return resolved

    raise ValueError(
        f"No scheduled phase covers epoch {epoch_number}. "
        f"Configured schedule: {json.dumps(schedule, sort_keys=True)}"
    )


def default_checkpoint_row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row["val_joint_score"],
        row["val_fchem"],
        row["val_vxc"],
        row["epoch"],
    )


def save_trial_history(output_dir: Path, trial_number: int, payload: Dict[str, Any]) -> Path:
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    path = trials_dir / f"trial_{trial_number}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def run_trial(
    trial_number: int,
    params: Dict[str, Any],
    args: argparse.Namespace,
    shared_preopt_checkpoint: Path,
    data_train: dict,
    data_val: dict,
    data_vxc_train: list,
    data_vxc_val: list,
    device: torch.device,
    local_rank: int,
    world_size: int,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    output_dir: Path,
    rank0: bool,
    epoch_selector: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
    checkpoint_row_key: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    trial_seed = args.seed + trial_number
    set_random_seed(trial_seed)

    loaders = build_dataloaders(
        data_train=data_train,
        data_val=data_val,
        data_vxc_train=data_vxc_train,
        data_vxc_val=data_vxc_val,
        trial_seed=trial_seed,
        args=args,
        rank=local_rank,
        world_size=world_size,
    )

    model = build_model(args, device)
    load_state_dict_into_model(model, shared_preopt_checkpoint, device)
    model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    )
    optimizer = configure_optimizers(
        model=model,
        learning_rate=params["lr_train"],
        optimizer_str="radamw",
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.n_train)

    epoch_history: List[Dict[str, Any]] = []
    min_val_fchem_any_epoch = math.inf
    min_val_vxc_any_epoch = math.inf
    min_val_exc_any_epoch = math.inf
    best_val_full_loss_any_epoch = math.inf
    current_selected_key = None
    selected_checkpoint_path = None
    failed = False
    epoch_selector = epoch_selector or select_representative_epoch
    checkpoint_row_key = checkpoint_row_key or default_checkpoint_row_key

    if args.save_selected_checkpoints and rank0:
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        selected_checkpoint_path = checkpoints_dir / f"trial_{trial_number}_selected.pt"

    for epoch in range(args.n_train):
        epoch_number = epoch + 1
        effective_params = resolve_epoch_params(params, epoch_number=epoch_number, n_train=args.n_train)
        effective_params = resolve_objective_params(effective_params)
        train_dataset = loaders["train_loader"].dataset
        if hasattr(train_dataset, "resample"):
            train_dataset.resample(epoch)
        if hasattr(loaders["train_sampler"], "set_epoch"):
            loaders["train_sampler"].set_epoch(epoch)
        if hasattr(loaders["vxc_train_sampler"], "set_epoch"):
            loaders["vxc_train_sampler"].set_epoch(epoch)

        train_metrics, train_per_db, train_failed = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train_loader"],
            vxc_train_loader=loaders["vxc_train_loader"],
            params=effective_params,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(getattr(args, "include_mrks_dispersion", False)),
            world_size=world_size,
            epoch=epoch,
        )
        if train_failed:
            failed = True
            break

        val_metrics, val_per_db, val_failed = validate_one_epoch(
            model=model,
            val_loader=loaders["val_loader"],
            vxc_val_loader=loaders["vxc_val_loader"],
            params=effective_params,
            device=device,
            dispersions=dispersions,
            mrks_dispersions=mrks_dispersions,
            include_mrks_dispersion=bool(getattr(args, "include_mrks_dispersion", False)),
            world_size=world_size,
        )
        if val_failed:
            failed = True
            break

        scheduler.step()

        row = {
            "epoch": epoch_number,
            "train_fchem": train_metrics["train_fchem"],
            "train_vxc": train_metrics["train_vxc"],
            "train_exc": train_metrics["train_exc"],
            "train_exc_loss": train_metrics["train_exc_loss"],
            "train_full_loss": train_metrics["train_full_loss"],
            "train_reaction_loss": train_metrics["train_reaction_loss"],
            "train_mae": train_metrics["train_mae"],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "val_fchem": val_metrics["val_fchem"],
            "val_vxc": val_metrics["val_vxc"],
            "val_exc": val_metrics["val_exc"],
            "val_exc_loss": val_metrics["val_exc_loss"],
            "val_full_loss": val_metrics["val_full_loss"],
            "val_reaction_loss": val_metrics["val_reaction_loss"],
            "val_mae": val_metrics["val_mae"],
            "val_joint_score": val_metrics["val_joint_score"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_per_database_rmse": val_per_db,
            "train_per_database_rmse": train_per_db,
            "val_per_system_exc_rmse": val_metrics["val_per_system_exc_rmse"],
            "train_per_system_exc_rmse": train_metrics["train_per_system_exc_rmse"],
            "phase_name": effective_params.get("phase_name", "static"),
            "phase_start_epoch": int(effective_params.get("phase_start_epoch", 1)),
            "phase_end_epoch": int(effective_params.get("phase_end_epoch", args.n_train)),
            "effective_gradient_merge_strategy": effective_params["gradient_merge_strategy"],
            "effective_reaction_gradient_merge_strategy": effective_params["reaction_gradient_merge_strategy"],
            "effective_vxc_gradient_merge_strategy": effective_params["vxc_gradient_merge_strategy"],
            "effective_accum_iter": int(effective_params["accum_iter"]),
            "effective_reaction_grad_clip": effective_params["reaction_grad_clip"],
            "effective_reaction_grad_scale": float(effective_params["reaction_grad_scale"]),
            "effective_vxc_grad_clip": float(effective_params["vxc_grad_clip"]),
            "effective_vxc_loss_scale": float(effective_params["vxc_loss_scale"]),
            "effective_exc_loss_scale": float(effective_params["exc_loss_scale"]),
            "effective_exc_grad_clip": effective_params["exc_grad_clip"],
            "effective_exc_grad_scale": float(effective_params["exc_grad_scale"]),
            "effective_exc_gradient_merge_strategy": effective_params["exc_gradient_merge_strategy"],
        }
        epoch_history.append(row)

        min_val_fchem_any_epoch = min(min_val_fchem_any_epoch, row["val_fchem"])
        min_val_vxc_any_epoch = min(min_val_vxc_any_epoch, row["val_vxc"])
        min_val_exc_any_epoch = min(min_val_exc_any_epoch, row["val_exc"])
        best_val_full_loss_any_epoch = min(best_val_full_loss_any_epoch, row["val_full_loss"])

        candidate_key = checkpoint_row_key(row)
        if current_selected_key is None or candidate_key < current_selected_key:
            current_selected_key = candidate_key
            if args.save_selected_checkpoints and rank0 and selected_checkpoint_path is not None:
                torch.save(model.module.state_dict(), selected_checkpoint_path)

        if rank0:
            print(
                f"Trial {trial_number} epoch {epoch + 1}/{args.n_train}: "
                f"train_fchem={row['train_fchem']:.8f} "
                f"val_fchem={row['val_fchem']:.8f} "
                f"val_vxc={row['val_vxc']:.8f} "
                f"val_exc={row['val_exc']:.8f} "
                f"joint_score={row['val_joint_score']:.8f}"
            )

    failed = sync_failure(failed, device)
    if failed:
        return {
            "failed": True,
            "trial_number": trial_number,
            "params": params,
        }

    selected = epoch_selector(epoch_history)
    trial_payload = {
        "failed": False,
        "trial_number": trial_number,
        "params": params,
        "selected_epoch": int(selected["epoch"]),
        "selected_joint_score": float(selected["val_joint_score"]),
        "selected_train_fchem": float(selected["train_fchem"]),
        "selected_val_fchem": float(selected["val_fchem"]),
        "selected_val_vxc": float(selected["val_vxc"]),
        "selected_val_exc": float(selected["val_exc"]),
        "selected_phase_name": selected.get("phase_name"),
        "min_val_fchem_any_epoch": float(min_val_fchem_any_epoch),
        "min_val_vxc_any_epoch": float(min_val_vxc_any_epoch),
        "min_val_exc_any_epoch": float(min_val_exc_any_epoch),
        "best_val_full_loss_any_epoch": float(best_val_full_loss_any_epoch),
        "epoch_history": epoch_history,
        "selected_checkpoint_path": str(selected_checkpoint_path) if selected_checkpoint_path is not None else None,
    }
    if rank0:
        history_path = save_trial_history(output_dir, trial_number, trial_payload)
        trial_payload["history_path"] = str(history_path)
    else:
        trial_payload["history_path"] = None
    return trial_payload


def summarize_study(study: Any) -> None:
    completed_trials = [trial for trial in study.trials if trial.values is not None]
    if not completed_trials:
        print("No completed trials.")
        return

    print("\nPareto front trials:")
    for trial in study.best_trials:
        score = max(trial.values[0] / 50.0, trial.values[1] / 1.0)
        print(
            f"  Trial {trial.number}: values={trial.values}, "
            f"selected_epoch={trial.user_attrs.get('selected_epoch')}, "
            f"joint_score={score:.8f}, params={trial.params}"
        )

    best_fchem_trial = min(completed_trials, key=lambda trial: trial.values[0])
    best_vxc_trial = min(completed_trials, key=lambda trial: trial.values[1])
    best_compromise_trial = min(
        completed_trials,
        key=lambda trial: max(trial.values[0] / 50.0, trial.values[1] / 1.0),
    )

    print("\nBest-Fchem selected trial:")
    print(f"  Trial {best_fchem_trial.number}: values={best_fchem_trial.values}, params={best_fchem_trial.params}")

    print("\nBest-Vxc selected trial:")
    print(f"  Trial {best_vxc_trial.number}: values={best_vxc_trial.values}, params={best_vxc_trial.params}")

    print("\nBest-compromise selected trial:")
    print(
        f"  Trial {best_compromise_trial.number}: values={best_compromise_trial.values}, "
        f"score={max(best_compromise_trial.values[0] / 50.0, best_compromise_trial.values[1] / 1.0):.8f}, "
        f"params={best_compromise_trial.params}"
    )


def main() -> None:
    import optuna

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, world_size, device, rank0 = init_distributed()
    set_random_seed(args.seed)

    if args.no_reaction_dispersion:
        dispersions = {}
    else:
        with (Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle").open("rb") as handle:
            dispersions = pickle.load(handle)
    mrks_dispersions = (
        load_mrks_dispersions(args.mrks_dispersions_pickle)
        if args.include_mrks_dispersion
        else None
    )

    data_predopt, data_train, data_val, data_vxc_train, data_vxc_val = load_chk(path=args.checkpoints_dir)
    shared_preopt_checkpoint = run_or_reuse_preoptimization(
        args=args,
        output_dir=output_dir,
        data_predopt=data_predopt,
        data_vxc_train=data_vxc_train,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        rank0=rank0,
    )

    if rank0:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            directions=["minimize", "minimize"],
        )
    else:
        study = None

    try:
        for _ in range(args.n_trials):
            if rank0:
                trial = study.ask()
                params = suggest_params(trial)
                payload = {
                    "command": RUN_TRIAL,
                    "trial_number": trial.number,
                    "params": params,
                    "shared_preopt_checkpoint": str(shared_preopt_checkpoint),
                }
                print(f"Starting trial {trial.number} with params: {json.dumps(params, sort_keys=True)}")
            else:
                trial = None
                payload = None

            payload = broadcast_payload(payload)
            if payload["command"] != RUN_TRIAL:
                break

            trial_result = run_trial(
                trial_number=int(payload["trial_number"]),
                params=payload["params"],
                args=args,
                shared_preopt_checkpoint=Path(payload["shared_preopt_checkpoint"]),
                data_train=data_train,
                data_val=data_val,
                data_vxc_train=data_vxc_train,
                data_vxc_val=data_vxc_val,
                device=device,
                local_rank=local_rank,
                world_size=world_size,
                dispersions=dispersions,
                mrks_dispersions=mrks_dispersions,
                output_dir=output_dir,
                rank0=rank0,
            )

            if rank0:
                if trial_result["failed"]:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    print(f"Trial {trial.number} failed.")
                else:
                    trial.set_user_attr("selected_epoch", trial_result["selected_epoch"])
                    trial.set_user_attr("selected_joint_score", trial_result["selected_joint_score"])
                    trial.set_user_attr("selected_val_fchem", trial_result["selected_val_fchem"])
                    trial.set_user_attr("selected_val_vxc", trial_result["selected_val_vxc"])
                    trial.set_user_attr("selected_val_exc", trial_result["selected_val_exc"])
                    trial.set_user_attr("min_val_fchem_any_epoch", trial_result["min_val_fchem_any_epoch"])
                    trial.set_user_attr("min_val_vxc_any_epoch", trial_result["min_val_vxc_any_epoch"])
                    trial.set_user_attr("min_val_exc_any_epoch", trial_result["min_val_exc_any_epoch"])
                    trial.set_user_attr("best_val_full_loss_any_epoch", trial_result["best_val_full_loss_any_epoch"])
                    trial.set_user_attr("shared_preopt_checkpoint_path", str(shared_preopt_checkpoint))
                    if trial_result.get("selected_checkpoint_path") is not None:
                        trial.set_user_attr("selected_checkpoint_path", trial_result["selected_checkpoint_path"])
                    if trial_result.get("history_path") is not None:
                        trial.set_user_attr("history_path", trial_result["history_path"])
                    study.tell(
                        trial,
                        values=(trial_result["selected_val_fchem"], trial_result["selected_val_vxc"]),
                    )
                    print(
                        f"Completed trial {trial.number}: selected_epoch={trial_result['selected_epoch']} "
                        f"selected_val_fchem={trial_result['selected_val_fchem']:.8f} "
                        f"selected_val_vxc={trial_result['selected_val_vxc']:.8f} "
                        f"selected_val_exc={trial_result['selected_val_exc']:.8f}"
                    )
            dist.barrier()

        if rank0:
            summarize_study(study)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
