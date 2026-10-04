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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dataset import collate_fn, fast_collate_fn_predopt
from NN_models import (
    pcPBELMLOptimizerV2,
    pcPBELMLOptimizerV2GcSoftplusMirror,
    pcPBELMLOptimizerV2GcSoftplusMirrorR2ScanAlpha,
    pcPBELMLOptimizerV2GcSveluMirror,
    pcPBELMLOptimizerV2Log,
)
from predopt import DatasetPredopt, predopt
from prepare_data import TRAINING_PROTOCOL
from prepare_data import load_chk as load_chk
from reaction_energy_calculation import (
    calculate_reaction_energy,
    calculate_xc_energy,
    get_local_energies,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from training_state import (
    atomic_torch_save,
    capture_runtime_state,
    load_torch_payload,
    restore_runtime_state,
)
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

FCHEM_DB_WEIGHTS = {
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
    FCHEM_DB_WEIGHTS[db] * FREQ_WEIGHTS[db] for db in FCHEM_DB_WEIGHTS
) / len(FCHEM_DB_WEIGHTS)

HARTREE2KCAL = 627.5095
DEFAULT_MRKS_DISPERSIONS = (
    Path(__file__).resolve().parent / "dispersions" / "dispersions_mrks.pickle"
)


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
        raise ValueError(
            f"Inconsistent augmentation suffixes detected: {sorted(suffixes)}"
        )
    return next(iter(suffixes))


def canonical_variant(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    suffix_map = {
        variant_suffix_from_reaction(reaction): reaction for reaction in group
    }
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
            group[generator.randrange(len(group))] for group in self.reaction_groups
        ]

    def __len__(self) -> int:
        return len(self.sampled_reactions)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        chosen = self.sampled_reactions[idx]
        return copy.deepcopy(chosen), chosen["Energy"]


def vxc_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    if batch and "StencilFeatures" in batch[0]:
        if len(batch) != 1:
            raise ValueError(
                "Full-Euler Lap batches must contain exactly one mRKS system."
            )
        record = batch[0]
        required = {
            "Name",
            "StencilFeatures",
            "Weights",
            "Vxc",
            "E_xc",
            "HBohr",
            "Protocol",
            "StencilVersion",
            "SourceProvenance",
        }
        if not required <= record.keys() or "Vrho" in record:
            raise ValueError(
                "Lap/full-Euler mode requires a provenance-complete stencil record; "
                "legacy Grid/Vrho data are rejected."
            )
        return {
            "LapRecords": [record],
            "Names": [record["Name"]],
            "GridLengths": torch.tensor([len(record["Weights"])], dtype=torch.long),
            "E_xc": record["E_xc"].reshape(1),
        }
    return {
        "Grid": torch.cat([item["Grid"] for item in batch], dim=0),
        "Vrho": torch.cat([item["Vrho"] for item in batch], dim=0),
        "Weights": torch.cat([item["Weights"] for item in batch], dim=0),
        "E_xc": torch.stack([item["E_xc"] for item in batch], dim=0),
        "Names": [item["Name"] for item in batch],
        "GridLengths": torch.tensor(
            [item["Grid"].shape[0] for item in batch], dtype=torch.long
        ),
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
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=[
            "base",
            "log",
            "lap",
            "gc_svelu_mirror",
            "gc_softplus_mirror",
            "gc_softplus_mirror_r2scan_alpha",
        ],
    )
    parser.add_argument("--n-predopt", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--convergence-base-epochs", type=int, default=500)
    parser.add_argument("--convergence-tail-epochs", type=int, default=0)
    parser.add_argument("--convergence-tail-start-lr", type=float, default=1e-5)
    parser.add_argument("--convergence-tail-min-lr", type=float, default=1e-7)
    parser.add_argument("--resume-training-state", type=str, default="")
    parser.add_argument("--training-state-every", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--snapshot-start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vxc-batch-size", type=int, default=1)
    parser.add_argument("--lr-predopt", type=float, default=2e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-vxc", type=int, default=2)
    parser.add_argument("--preopt-vxc-weight", type=float, default=0.0)
    parser.add_argument("--preopt-vxc-steps", type=int, default=0)
    parser.add_argument("--preopt-vxc-target", type=str, default="pbe", choices=["pbe"])
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument(
        "--mrks-dispersions-pickle", type=str, default=str(DEFAULT_MRKS_DISPERSIONS)
    )
    parser.add_argument(
        "--no-reaction-dispersion",
        action="store_true",
        help="Do not add precomputed D3 dispersion corrections in reaction-energy training.",
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
    if not dist.is_initialized():
        return bool(local_failed)
    flag = torch.tensor([1 if local_failed else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def broadcast_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not dist.is_initialized():
        return payload
    object_list = [payload]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def gather_object(payload: Any, world_size: int) -> List[Any]:
    if not dist.is_initialized():
        return [payload]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)
    return gathered


def parse_model_name(name: str) -> Tuple[int, int, bool, bool]:
    prefix, num_layers, h_dim = name.split("_")
    use_g_x = "Gx" in prefix
    use_g_c = "Gc" in prefix
    return int(num_layers), int(h_dim), use_g_x, use_g_c


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from NN_models_lap import pcPBELMLOptimizerV2Lap

    num_layers, h_dim, use_g_x, use_g_c = parse_model_name(args.name)
    model_type = getattr(args, "model_type", "base")
    if args.name.startswith("PBE-Lap-") != (model_type == "lap"):
        raise ValueError(
            "Lap architecture requires both an explicit PBE-Lap name and model_type=lap."
        )
    model_classes = {
        "lap": pcPBELMLOptimizerV2Lap,
        "base": pcPBELMLOptimizerV2,
        "log": pcPBELMLOptimizerV2Log,
        "gc_svelu_mirror": pcPBELMLOptimizerV2GcSveluMirror,
        "gc_softplus_mirror": pcPBELMLOptimizerV2GcSoftplusMirror,
        "gc_softplus_mirror_r2scan_alpha": pcPBELMLOptimizerV2GcSoftplusMirrorR2ScanAlpha,
    }
    model_cls = model_classes[model_type]
    model = model_cls(
        num_layers=num_layers,
        h_dim=h_dim,
        dropout=args.dropout,
        DFT="PBE",
        use_g_x=use_g_x,
        use_g_c=use_g_c,
    )
    return model.to(device=device, dtype=getattr(args, "dtype", torch.float32))


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
    unexpected_without_allowed = [
        name for name in unexpected if name not in allowed_unexpected
    ]
    if missing != allowed_missing or unexpected_without_allowed:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch. Missing={missing}, Unexpected={unexpected_without_allowed}"
        )


def load_mrks_dispersions(path: str) -> Dict[str, float]:
    with Path(path).open("rb") as handle:
        raw = pickle.load(handle)
    return {key: float(value) for key, value in raw.items()}


def _lap_record_from_batch(X_batch: dict[str, Any]) -> dict[str, Any]:
    records = X_batch.get("LapRecords")
    if not isinstance(records, (list, tuple)) or len(records) != 1:
        raise ValueError(
            "Full-Euler Lap objectives require one explicit full-stencil record."
        )
    record = records[0]
    if "Vrho" in record or "Grid" in record:
        raise ValueError(
            "Legacy partial-Vrho records cannot enter Lap/full-Euler mode."
        )
    if record.get("Protocol") != "diet-clean-mn-all-mrks-lap-fullvxc-v1":
        raise ValueError("Lap/full-Euler record protocol is missing or incompatible.")
    from lap_vxc import (
        STENCIL_VERSION,
        stencil_order_for_version,
        validate_stencil_selection,
    )

    version = record.get("StencilVersion")
    order = record.get("StencilOrder")
    # Older full-Vxc records predate the numeric order field, but carry the
    # immutable persisted 7-point version. Resolve only that exact version;
    # tensor shape is never used to select an operator.
    if "StencilOrder" not in record and version == STENCIL_VERSION:
        order = stencil_order_for_version(version)
    validate_stencil_selection(order, version)
    normalized = dict(record)
    normalized["StencilOrder"] = order
    return normalized


def _require_lap_model(model: nn.Module) -> nn.Module:
    from lap_data import PROTOCOL
    from NN_models_lap import DESCRIPTOR_PROTOCOL

    base_model = model.module if hasattr(model, "module") else model
    if getattr(base_model, "descriptor_protocol", None) != DESCRIPTOR_PROTOCOL:
        raise ValueError("full_euler potential mode requires pcPBELMLOptimizerV2Lap.")
    if getattr(base_model, "protocol", PROTOCOL) not in (None, PROTOCOL):
        raise ValueError("Lap architecture and full-Vxc corpus protocol disagree.")
    return base_model


def _lap_energy_and_features(model: nn.Module, record: dict[str, Any], device):
    from lap_vxc import LapEnergy

    base_model = _require_lap_model(model)
    parameter = next(base_model.parameters())
    features = record["StencilFeatures"].to(
        device=device, dtype=parameter.dtype, non_blocking=True
    )
    target = record["Vxc"].to(device=device, dtype=torch.float64, non_blocking=True)
    weights = record["Weights"].to(
        device=device, dtype=torch.float64, non_blocking=True
    )
    return LapEnergy(base_model), features, target, weights


def _add_mrks_dispersion_once(
    prediction: torch.Tensor,
    system_name: str,
    dispersions: dict[str, float] | None,
    enabled: bool,
) -> torch.Tensor:
    """Match calculate_xc_energy's exact Name lookup and one-addition semantics."""
    if not enabled:
        return prediction
    if dispersions is None:
        raise ValueError(
            "Lap-S5 mRKS dispersion is enabled but no artifact was loaded."
        )
    correction = torch.as_tensor(
        float(dispersions.get(system_name, 0.0)),
        device=prediction.device,
        dtype=prediction.dtype,
    )
    return prediction + correction


def lap_full_vxc_loss(
    model: nn.Module,
    X_batch: dict[str, Any],
    device: torch.device,
    point_chunk_size: int,
) -> torch.Tensor:
    from lap_vxc import full_vxc_loss

    record = _lap_record_from_batch(X_batch)
    energy, features, target, weights = _lap_energy_and_features(model, record, device)
    h = float(record["HBohr"])
    return full_vxc_loss(
        energy,
        features,
        target,
        weights,
        h,
        point_chunk_size=point_chunk_size,
        order=record["StencilOrder"],
        version=record["StencilVersion"],
    )


def lap_exc_loss(
    model: nn.Module,
    X_batch: dict[str, Any],
    device: torch.device,
    dispersions: dict[str, float] | None,
    include_mrks_dispersion: bool,
    point_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from lap_vxc import integrated_energy

    record = _lap_record_from_batch(X_batch)
    energy, features, _, weights = _lap_energy_and_features(model, record, device)
    prediction = integrated_energy(
        energy, features, weights, point_chunk_size=point_chunk_size
    )
    prediction = _add_mrks_dispersion_once(
        prediction, record["Name"], dispersions, include_mrks_dispersion
    )
    target = record["E_xc"].to(device=device, dtype=prediction.dtype).reshape(())
    pred_batch = prediction.reshape(1)
    target_batch = target.reshape(1)
    loss = batch_exc([record["Name"]], pred_batch, target_batch)
    return loss, pred_batch, target_batch


def build_scheduler(optimizer: torch.optim.Optimizer, n_train: int):
    warmup = LinearLR(
        optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(n_train - WARMUP_EPOCHS, 1),
        eta_min=MIN_LR,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )


class GlobalCosineWithConvergenceTail:
    """Preserve the 500-epoch replay LR path, then run a low-LR cosine tail."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_epochs: int,
        tail_epochs: int,
        tail_start_lr: float,
        tail_min_lr: float,
    ) -> None:
        if base_epochs <= WARMUP_EPOCHS:
            raise ValueError("Convergence-tail base epochs must exceed warmup epochs.")
        if tail_epochs <= 0:
            raise ValueError("Convergence-tail epochs must be positive.")
        if not 0.0 < tail_min_lr < tail_start_lr:
            raise ValueError("Convergence-tail LR bounds must satisfy 0 < min < start.")
        self.optimizer = optimizer
        self.base_epochs = int(base_epochs)
        self.tail_epochs = int(tail_epochs)
        self.tail_start_lr = float(tail_start_lr)
        self.tail_min_lr = float(tail_min_lr)
        self.completed_epochs = 0
        self.base_scheduler = build_scheduler(optimizer, self.base_epochs)

    def prepare_epoch(self, epoch_number: int) -> None:
        if epoch_number == self.base_epochs + 1:
            self._set_lr(self.tail_start_lr)

    def step(self) -> None:
        self.completed_epochs += 1
        if self.completed_epochs <= self.base_epochs:
            self.base_scheduler.step()
            return
        tail_step = min(self.completed_epochs - self.base_epochs, self.tail_epochs)
        fraction = tail_step / self.tail_epochs
        learning_rate = self.tail_min_lr + 0.5 * (
            self.tail_start_lr - self.tail_min_lr
        ) * (1.0 + math.cos(math.pi * fraction))
        self._set_lr(learning_rate)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "completed_epochs": self.completed_epochs,
            "base_epochs": self.base_epochs,
            "tail_epochs": self.tail_epochs,
            "tail_start_lr": self.tail_start_lr,
            "tail_min_lr": self.tail_min_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        expected = {
            "base_epochs": self.base_epochs,
            "tail_epochs": self.tail_epochs,
            "tail_start_lr": self.tail_start_lr,
            "tail_min_lr": self.tail_min_lr,
        }
        observed = {key: state_dict[key] for key in expected}
        if observed != expected:
            raise ValueError(
                f"Convergence-tail scheduler mismatch: expected {expected}, got {observed}."
            )
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.completed_epochs = int(state_dict["completed_epochs"])

    def get_last_lr(self) -> List[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)


def build_training_scheduler(optimizer, args):
    tail_epochs = int(getattr(args, "convergence_tail_epochs", 0))
    if tail_epochs <= 0:
        return build_scheduler(optimizer, args.n_train)
    base_epochs = int(getattr(args, "convergence_base_epochs", 500))
    if args.n_train < base_epochs + tail_epochs:
        raise ValueError(
            "Convergence training requires n_train >= convergence_base_epochs + "
            f"convergence_tail_epochs, got {args.n_train} < {base_epochs} + {tail_epochs}."
        )
    return GlobalCosineWithConvergenceTail(
        optimizer,
        base_epochs=base_epochs,
        tail_epochs=tail_epochs,
        tail_start_lr=float(getattr(args, "convergence_tail_start_lr", 1e-5)),
        tail_min_lr=float(getattr(args, "convergence_tail_min_lr", 1e-7)),
    )


def completed_params_signature(params: Dict[str, Any], completed_epoch: int) -> str:
    """Describe the objective schedule already consumed by a saved run."""
    normalized = copy.deepcopy(params)
    schedule = normalized.get("epoch_schedule")
    if schedule:
        completed_schedule = []
        for phase in schedule:
            start_epoch = int(phase.get("start_epoch", 1))
            if start_epoch > completed_epoch:
                continue
            completed_phase = copy.deepcopy(phase)
            completed_phase["end_epoch"] = min(
                int(phase.get("end_epoch", completed_epoch)), completed_epoch
            )
            completed_schedule.append(completed_phase)
        normalized["epoch_schedule"] = completed_schedule
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


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
    data_vxc_train: list,
    trial_seed: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    data_protocol = getattr(args, "data_protocol", "legacy_vrho")
    potential_mode = getattr(args, "potential_mode", "partial_vrho")
    model_type = getattr(args, "model_type", "base")
    if data_protocol == "lap_full_vxc":
        if potential_mode != "full_euler" or model_type != "lap":
            raise ValueError(
                "Lap stencil data requires model_type=lap and potential_mode=full_euler."
            )
        if int(args.vxc_batch_size) != 1:
            raise ValueError("Lap full-Euler training requires vxc_batch_size=1.")
        if not data_vxc_train:
            raise ValueError("Lap full-Euler training requires nonempty stencil data.")
        for record in data_vxc_train:
            if (
                "Vrho" in record
                or record.get("Protocol") != "diet-clean-mn-all-mrks-lap-fullvxc-v1"
                or "StencilFeatures" not in record
            ):
                raise ValueError(
                    "Lap full-Euler training rejected legacy/partial-potential data."
                )
    elif data_protocol == "legacy_vrho":
        if potential_mode != "partial_vrho" or model_type == "lap":
            raise ValueError(
                "Legacy Vrho data requires a non-Lap model and partial_vrho mode."
            )
    else:
        raise ValueError(f"Unsupported data protocol: {data_protocol!r}.")

    train_set = EpochSampledAugmentedDataset(data_train, base_seed=trial_seed)
    vxc_train_set = VxcDataset(data_vxc_train)

    train_loader, train_sampler = build_reaction_loader(
        train_set,
        args.batch_size,
        trial_seed,
        rank,
        world_size,
        args.num_workers_train,
        shuffle=True,
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

    return {
        "train_loader": train_loader,
        "train_sampler": train_sampler,
        "vxc_train_loader": vxc_train_loader,
        "vxc_train_sampler": vxc_train_sampler,
    }


def batch_fchem(
    current_bases: List[str],
    reaction_energy: torch.Tensor,
    y_batch: torch.Tensor,
) -> torch.Tensor:
    if isinstance(current_bases, str):
        raise TypeError("current_bases must be a sequence of labels, not a string.")
    err_dict: Dict[str, List[List[torch.Tensor]]] = {}
    for database, pred, ref in zip(current_bases, reaction_energy, y_batch):
        err_dict.setdefault(database, [[], []])
        err_dict[database][0].append(pred)
        err_dict[database][1].append(ref)

    values = []
    for database, (preds, refs) in err_dict.items():
        db_predictions = torch.stack(preds)
        db_ref = torch.stack(refs)
        factor = (
            FCHEM_DB_WEIGHTS.get(database, 1)
            * FREQ_WEIGHTS.get(database, 1)
            / MEAN_WEIGHT
        )
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
        total_exc_errors[system_name].append(
            float(torch.abs(error).item()) * HARTREE2KCAL
        )


def compute_fchem_from_errors(
    total_database_errors: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    total = 0.0
    per_db = {}
    for db in sorted(total_database_errors):
        errors = np.asarray(total_database_errors[db], dtype=np.float64)
        if errors.size == 0:
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        per_db[db] = rmse
        total += FCHEM_DB_WEIGHTS.get(db, 1) * rmse
    return total, per_db


def compute_exc_from_errors(
    total_exc_errors: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
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
    potential_mode: str = "partial_vrho",
    point_chunk_size: int | None = None,
) -> torch.Tensor:
    if potential_mode == "full_euler":
        if point_chunk_size is None or point_chunk_size <= 0:
            raise ValueError(
                "full_euler requires an explicit positive point chunk size."
            )
        return lap_full_vxc_loss(model, X_batch, device, point_chunk_size)
    if potential_mode != "partial_vrho":
        raise ValueError(f"Unsupported potential mode: {potential_mode!r}.")
    if (
        getattr(getattr(model, "module", model), "descriptor_protocol", None)
        == "rho-sigma-total-lapl-tau-free-v1"
    ):
        raise ValueError(
            "Lap models require full_vxc_loss via potential_mode='full_euler' "
            "and reference stencils."
        )
    grid_raw = X_batch["Grid"].to(device).clone().detach()
    rho = grid_raw[:, 4:6].clone().requires_grad_(True)
    sigma = grid_raw[:, 6:9].clone()
    sigma = _fix_sigma_tot_closed_shell(sigma)
    sigma_pbe = torch.stack(
        [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]],
        dim=1,
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
    potential_mode: str = "partial_vrho",
    point_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if potential_mode == "full_euler":
        if point_chunk_size is None or point_chunk_size <= 0:
            raise ValueError(
                "full_euler requires an explicit positive point chunk size."
            )
        return lap_exc_loss(
            model,
            X_batch,
            device,
            dispersions,
            include_mrks_dispersion,
            point_chunk_size,
        )
    if potential_mode != "partial_vrho":
        raise ValueError(f"Unsupported potential mode: {potential_mode!r}.")
    if (
        getattr(getattr(model, "module", model), "descriptor_protocol", None)
        == "rho-sigma-total-lapl-tau-free-v1"
    ):
        raise ValueError("Lap models require full_euler mode and reference stencils.")
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
            [sigma[:, 0], (sigma[:, 1] - sigma[:, 0] - sigma[:, 2]) / 2.0, sigma[:, 2]],
            dim=1,
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
    return list(
        torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    )


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


def allreduce_parameter_grads(
    parameters: list[torch.nn.Parameter], world_size: int
) -> None:
    if world_size <= 1 or not dist.is_initialized():
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


def accumulation_divisor(
    batch_idx: int,
    n_steps: int,
    accum_iter: int,
    potential_mode: str,
) -> int:
    """Return S5's divisor, correcting only Lap's final partial window.

    The historical non-Lap path intentionally keeps its prior full-window
    divisor for exact trajectory compatibility. Lap-S5 averages the final
    partial window by its actual number of microsteps.
    """
    if accum_iter <= 0 or n_steps <= 0 or not 0 <= batch_idx < n_steps:
        raise ValueError("Invalid gradient-accumulation position or size.")
    if potential_mode == "partial_vrho":
        return accum_iter
    if potential_mode != "full_euler":
        raise ValueError(f"Unsupported potential mode: {potential_mode!r}.")
    window_start = (batch_idx // accum_iter) * accum_iter
    return min(accum_iter, n_steps - window_start)


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
        "vxc_loss_scale": trial.suggest_categorical(
            "vxc_loss_scale", [50, 100, 150, 200]
        ),
        "exc_loss_scale": trial.suggest_categorical(
            "exc_loss_scale", [0.0, 1.0, 5.0, 10.0]
        ),
        "reaction_grad_scale": trial.suggest_categorical(
            "reaction_grad_scale", [0.1, 0.3, 0.5, 0.7, 1.0]
        ),
        "reaction_grad_clip": trial.suggest_categorical(
            "reaction_grad_clip", ["none", 100.0, 300.0, 1000.0]
        ),
        "vxc_grad_clip": trial.suggest_categorical(
            "vxc_grad_clip", [1.0, 2.0, 3.0, 5.0]
        ),
        "exc_grad_clip": trial.suggest_categorical(
            "exc_grad_clip", ["none", 1.0, 2.0, 5.0]
        ),
        "exc_grad_scale": trial.suggest_categorical(
            "exc_grad_scale", [0.1, 0.3, 0.5, 1.0]
        ),
        "gradient_merge_strategy": trial.suggest_categorical(
            "gradient_merge_strategy", ["sum", "clip_then_sum"]
        ),
        "exc_gradient_merge_strategy": trial.suggest_categorical(
            "exc_gradient_merge_strategy", ["sum", "clip_then_sum"]
        ),
    }


def resolve_shared_preopt_checkpoint(
    args: argparse.Namespace, output_dir: Path
) -> tuple[Path, Path]:
    checkpoint_path = (
        Path(args.shared_preopt_checkpoint)
        if args.shared_preopt_checkpoint
        else output_dir / "preoptimized_checkpoint.pt"
    )
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    return checkpoint_path, metadata_path


def preopt_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    metadata = {
        "training_protocol": TRAINING_PROTOCOL,
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
    if getattr(args, "data_protocol", "legacy_vrho") == "lap_full_vxc":
        metadata.update(
            data_protocol="lap_full_vxc",
            potential_mode="full_euler",
            dtype=str(getattr(args, "dtype", torch.float32)).removeprefix("torch."),
            lap_s5_provenance=getattr(args, "lap_s5_provenance", None),
        )
    return metadata


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
    is_lap_mode = getattr(args, "data_protocol", "legacy_vrho") == "lap_full_vxc"
    if is_lap_mode and (
        float(args.preopt_vxc_weight) != 0.0 or int(args.preopt_vxc_steps) != 0
    ):
        raise ValueError(
            "Lap PBE predopt must have zero Vxc weight and zero Vxc steps."
        )
    checkpoint_path, metadata_path = resolve_shared_preopt_checkpoint(args, output_dir)
    metadata = preopt_metadata(args)

    decided_path: Optional[str] = None
    if rank0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        reuse = False
        if (
            checkpoint_path.exists()
            and metadata_path.exists()
            and not args.force_preopt
        ):
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
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_path

    set_random_seed(args.seed)
    model = build_model(args, device)
    if dist.is_initialized():
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_predopt, betas=(0.9, 0.999)
    )
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
        base_model = model.module if hasattr(model, "module") else model
        torch.save(base_model.state_dict(), checkpoint_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
    if dist.is_initialized():
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
    potential_mode: str = "partial_vrho",
    data_protocol: str = "legacy_vrho",
    point_chunk_size: int | None = None,
    max_optimizer_steps: int | None = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]], bool]:
    params = resolve_objective_params(params)
    base_model = model.module if hasattr(model, "module") else model
    descriptor = getattr(base_model, "descriptor_protocol", None)
    if potential_mode == "full_euler":
        if data_protocol != "lap_full_vxc":
            raise ValueError("full_euler mode requires data_protocol=lap_full_vxc.")
        _require_lap_model(model)
        if point_chunk_size is None or point_chunk_size <= 0:
            raise ValueError(
                "Lap-S5 training requires an explicit positive point chunk."
            )
    elif potential_mode == "partial_vrho":
        if data_protocol != "legacy_vrho":
            raise ValueError("partial_vrho mode requires data_protocol=legacy_vrho.")
        if descriptor == "rho-sigma-total-lapl-tau-free-v1":
            raise ValueError("Lap models cannot use the legacy partial-Vrho objective.")
    else:
        raise ValueError(f"Unsupported potential mode: {potential_mode!r}.")
    if max_optimizer_steps is not None and max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive when provided.")
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
    if max_optimizer_steps is not None:
        n_steps = min(n_steps, max_optimizer_steps * int(params["accum_iter"]))
    train_iter = iter(train_loader)
    vxc_iter = iter(vxc_train_loader)

    loss_sum = 0.0
    reaction_loss_sum = 0.0
    vxc_loss_sum = 0.0
    exc_loss_sum = 0.0
    mae_sum = 0.0
    optimizer_steps = 0
    gradient_norm_sum = 0.0
    parameter_update_norm_sum = 0.0
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

        do_step = ((batch_idx + 1) % params["accum_iter"] == 0) or (
            (batch_idx + 1) == n_steps
        )
        accumulation_scale = accumulation_divisor(
            batch_idx,
            n_steps,
            int(params["accum_iter"]),
            potential_mode,
        )
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
        weighted_reaction_loss = reaction_loss / accumulation_scale
        microbatch_failed = not torch.isfinite(
            reaction_energy
        ).all() or not torch.isfinite(weighted_reaction_loss)
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

        vxc_term = vxc_loss(
            model,
            X_vxc,
            device,
            rung="GGA",
            dft="PBE",
            create_graph=True,
            potential_mode=potential_mode,
            point_chunk_size=point_chunk_size,
        )
        weighted_vxc_loss = (
            OMEGA * params["vxc_loss_scale"] * vxc_term / accumulation_scale
        )
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
            potential_mode=potential_mode,
            point_chunk_size=point_chunk_size,
        )
        weighted_exc_loss = params["exc_loss_scale"] * exc_term / accumulation_scale
        full_loss = weighted_reaction_loss + weighted_vxc_loss + weighted_exc_loss
        microbatch_failed = not torch.isfinite(pred_exc).all() or not torch.isfinite(
            full_loss
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
        loss_sum += float(
            (
                reaction_loss
                + OMEGA * params["vxc_loss_scale"] * vxc_term
                + params["exc_loss_scale"] * exc_term
            ).item()
        )
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

        if potential_mode == "full_euler":
            gradient_norm_sum += math.sqrt(
                sum(
                    float(parameter.grad.detach().double().square().sum())
                    for parameter in trainable_parameters
                    if parameter.grad is not None
                )
            )
            parameters_before = [
                parameter.detach().clone() for parameter in trainable_parameters
            ]
        optimizer.step()
        if potential_mode == "full_euler":
            parameter_update_norm_sum += math.sqrt(
                sum(
                    float((parameter.detach() - before).double().square().sum())
                    for parameter, before in zip(
                        trainable_parameters, parameters_before
                    )
                )
            )
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break

    if failed:
        optimizer.zero_grad(set_to_none=True)
        return {}, {}, True

    scalar_tensor = torch.tensor(
        [
            loss_sum,
            reaction_loss_sum,
            vxc_loss_sum,
            exc_loss_sum,
            mae_sum,
            float(n_steps),
            float(optimizer_steps),
        ],
        device=device,
    )
    if dist.is_initialized():
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
        "train_full_loss": float(
            scalar_tensor[0].item() / max(scalar_tensor[5].item(), 1.0)
        ),
        "train_reaction_loss": float(
            scalar_tensor[1].item() / max(scalar_tensor[5].item(), 1.0)
        ),
        "train_vxc": float(scalar_tensor[2].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_exc_loss": float(
            scalar_tensor[3].item() / max(scalar_tensor[5].item(), 1.0)
        ),
        "train_mae": float(scalar_tensor[4].item() / max(scalar_tensor[5].item(), 1.0)),
        "train_fchem": train_fchem,
        "train_exc": train_exc,
        "train_per_system_exc_rmse": train_per_system_exc,
        "optimizer_steps": int(scalar_tensor[6].item()),
    }
    if potential_mode == "full_euler":
        denom = max(optimizer_steps, 1)
        metrics["gradient_norm"] = gradient_norm_sum / denom
        metrics["parameter_update_norm"] = parameter_update_norm_sum / denom
    return metrics, dict(train_per_db), False


def validate_one_epoch(*args: Any, **kwargs: Any) -> None:
    """Historical API disabled: validation requires external self-consistent SCF."""
    raise RuntimeError(
        "Non-SCF validation was removed; use external DietGMTKN30 SCF evaluation."
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
            resolved = {
                key: value for key, value in params.items() if key != "epoch_schedule"
            }
            resolved.update(dict(phase.get("params", {})))
            resolved["phase_name"] = str(phase.get("name", f"phase_{index + 1}"))
            resolved["phase_start_epoch"] = start_epoch
            resolved["phase_end_epoch"] = end_epoch
            return resolved

    raise ValueError(
        f"No scheduled phase covers epoch {epoch_number}. "
        f"Configured schedule: {json.dumps(schedule, sort_keys=True)}"
    )


def save_trial_history(
    output_dir: Path, trial_number: int, payload: dict[str, Any]
) -> Path:
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
    shared_preopt_checkpoint: Optional[Path],
    data_train: dict,
    data_vxc_train: list,
    device: torch.device,
    local_rank: int,
    world_size: int,
    dispersions: Dict[str, float],
    mrks_dispersions: Optional[Dict[str, float]],
    output_dir: Path,
    rank0: bool,
) -> Dict[str, Any]:
    trial_seed = args.seed + trial_number
    set_random_seed(trial_seed)

    loaders = build_dataloaders(
        data_train=data_train,
        data_vxc_train=data_vxc_train,
        trial_seed=trial_seed,
        args=args,
        rank=local_rank,
        world_size=world_size,
    )

    resume_training_state = str(getattr(args, "resume_training_state", "") or "")
    model = build_model(args, device)
    if not resume_training_state:
        load_state_dict_into_model(model, shared_preopt_checkpoint, device)
    if dist.is_initialized():
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
    scheduler = build_training_scheduler(optimizer, args)

    epoch_history: List[Dict[str, Any]] = []
    failed = False
    final_checkpoint_path = (
        output_dir / "checkpoints" / f"trial_{trial_number}_final.pt"
    )

    training_state_every = int(getattr(args, "training_state_every", 0))
    if training_state_every < 0:
        raise ValueError("--training-state-every must be nonnegative.")
    snapshot_every = int(getattr(args, "snapshot_every", 0))
    snapshot_start_epoch = int(getattr(args, "snapshot_start_epoch", 1))
    if snapshot_every < 0:
        raise ValueError("--snapshot-every must be nonnegative.")
    if snapshot_start_epoch < 1:
        raise ValueError("--snapshot-start-epoch must be positive.")
    snapshot_dir = output_dir / "checkpoints" / "epoch_snapshots"
    training_state_path = (
        output_dir / "checkpoints" / f"trial_{trial_number}_training_state.pt"
    )
    start_epoch = 0
    lap_s5_provenance = getattr(args, "lap_s5_provenance", None)
    lap_checkpoint_extra = getattr(args, "lap_checkpoint_extra", {})
    is_lap_mode = getattr(args, "data_protocol", "legacy_vrho") == "lap_full_vxc"
    if is_lap_mode:
        if getattr(args, "potential_mode", None) != "full_euler":
            raise ValueError(
                "Lap-S5 run must explicitly set potential_mode=full_euler."
            )
        if not isinstance(lap_s5_provenance, dict):
            raise ValueError("Lap-S5 run requires hashed checkpoint/run provenance.")
        from lap_s5_provenance import validate_lap_s5_provenance

        validate_lap_s5_provenance(lap_s5_provenance)
    if resume_training_state:
        resume_path = Path(resume_training_state)
        resume_payload = load_torch_payload(resume_path, map_location=device)
        if int(resume_payload.get("format_version", -1)) != 2:
            raise ValueError(
                f"Unsupported training-state format: {resume_payload.get('format_version')}."
            )
        if int(resume_payload["trial_number"]) != int(trial_number):
            raise ValueError(
                f"Training-state trial mismatch: {resume_payload['trial_number']} != {trial_number}."
            )
        if resume_payload.get("model_name") != args.name:
            raise ValueError(
                f"Training-state model mismatch: {resume_payload.get('model_name')} != {args.name}."
            )
        if resume_payload.get("model_type", "base") != getattr(
            args, "model_type", "base"
        ):
            raise ValueError(
                "Training-state model type does not match the requested model type."
            )
        if int(resume_payload.get("world_size", world_size)) != int(world_size):
            raise ValueError(
                "Exact training-state resume requires the original DDP world size."
            )
        if resume_payload.get("training_protocol") != TRAINING_PROTOCOL:
            raise ValueError(
                "Cannot resume training state from the obsolete internal-validation protocol."
            )
        if is_lap_mode:
            if resume_payload.get("potential_mode") != "full_euler":
                raise ValueError(
                    "Cannot resume a Lap run from partial-Vrho training state."
                )
            if resume_payload.get("lap_s5_provenance") != lap_s5_provenance:
                raise ValueError(
                    "Lap-S5 resume provenance differs from the requested data/protocol."
                )
            if resume_payload.get("lap_checkpoint_extra") != lap_checkpoint_extra:
                raise ValueError(
                    "Lap-S5 resume corpus identity differs from the requested source."
                )
        elif resume_payload.get("potential_mode", "partial_vrho") != "partial_vrho":
            raise ValueError(
                "Historical model cannot resume from a full-Euler Lap state."
            )
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        epoch_history = list(resume_payload.get("epoch_history", []))
        start_epoch = int(resume_payload["completed_epoch"])
        if len(epoch_history) != start_epoch:
            raise ValueError(
                f"Training-state history length {len(epoch_history)} does not match epoch {start_epoch}."
            )
        if start_epoch >= args.n_train:
            raise ValueError(
                f"Training state is already at epoch {start_epoch}, not below n_train={args.n_train}."
            )
        saved_signature = completed_params_signature(
            resume_payload["params"], start_epoch
        )
        requested_signature = completed_params_signature(params, start_epoch)
        if saved_signature != requested_signature:
            raise ValueError(
                "Training-state objective schedule does not match the requested run."
            )
        runtime_states = resume_payload.get("runtime_states", [])
        rank = dist.get_rank() if dist.is_initialized() else local_rank
        if len(runtime_states) != world_size:
            raise ValueError(
                "Training-state runtime state count does not match DDP world size."
            )
        restore_runtime_state(runtime_states[rank], loaders)
        if rank0:
            print(
                f"Resuming Trial {trial_number} from epoch {start_epoch}: {resume_path}"
            )

    for epoch in range(start_epoch, args.n_train):
        epoch_number = epoch + 1
        prepare_epoch = getattr(scheduler, "prepare_epoch", None)
        if prepare_epoch is not None:
            prepare_epoch(epoch_number)
        effective_params = resolve_epoch_params(
            params, epoch_number=epoch_number, n_train=args.n_train
        )
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
            include_mrks_dispersion=bool(
                getattr(args, "include_mrks_dispersion", False)
            ),
            world_size=world_size,
            epoch=epoch,
            potential_mode=getattr(args, "potential_mode", "partial_vrho"),
            data_protocol=getattr(args, "data_protocol", "legacy_vrho"),
            point_chunk_size=getattr(args, "point_chunk_size", None),
        )
        if train_failed:
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_per_database_rmse": train_per_db,
            "train_per_system_exc_rmse": train_metrics["train_per_system_exc_rmse"],
            "phase_name": effective_params.get("phase_name", "static"),
            "phase_start_epoch": int(effective_params.get("phase_start_epoch", 1)),
            "phase_end_epoch": int(
                effective_params.get("phase_end_epoch", args.n_train)
            ),
            "effective_gradient_merge_strategy": effective_params[
                "gradient_merge_strategy"
            ],
            "effective_reaction_gradient_merge_strategy": effective_params[
                "reaction_gradient_merge_strategy"
            ],
            "effective_vxc_gradient_merge_strategy": effective_params[
                "vxc_gradient_merge_strategy"
            ],
            "effective_accum_iter": int(effective_params["accum_iter"]),
            "effective_reaction_grad_clip": effective_params["reaction_grad_clip"],
            "effective_reaction_grad_scale": float(
                effective_params["reaction_grad_scale"]
            ),
            "effective_vxc_grad_clip": float(effective_params["vxc_grad_clip"]),
            "effective_vxc_loss_scale": float(effective_params["vxc_loss_scale"]),
            "effective_exc_loss_scale": float(effective_params["exc_loss_scale"]),
            "effective_exc_grad_clip": effective_params["exc_grad_clip"],
            "effective_exc_grad_scale": float(effective_params["exc_grad_scale"]),
            "effective_exc_gradient_merge_strategy": effective_params[
                "exc_gradient_merge_strategy"
            ],
        }
        if is_lap_mode:
            row["omega"] = OMEGA
            row["effective_vxc_coefficient"] = OMEGA * float(
                effective_params["vxc_loss_scale"]
            )
        epoch_history.append(row)

        should_save_snapshot = (
            snapshot_every > 0
            and epoch_number >= snapshot_start_epoch
            and (
                (epoch_number - snapshot_start_epoch) % snapshot_every == 0
                or epoch_number == args.n_train
            )
        )
        if should_save_snapshot and rank0:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                (
                    _lap_s5_model_payload(
                        model,
                        lap_s5_provenance,
                        **lap_checkpoint_extra,
                        epoch=epoch_number,
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                    )
                    if is_lap_mode
                    else (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    )
                ),
                snapshot_dir / f"trial_{trial_number}_epoch_{epoch_number:04d}.pt",
            )

        should_save_training_state = training_state_every > 0 and (
            epoch_number % training_state_every == 0 or epoch_number == args.n_train
        )
        if should_save_training_state:
            runtime_states = gather_object(capture_runtime_state(loaders), world_size)
            if rank0:
                atomic_torch_save(
                    {
                        "format_version": 2,
                        "training_protocol": TRAINING_PROTOCOL,
                        "trial_number": int(trial_number),
                        "model_name": args.name,
                        "model_type": getattr(args, "model_type", "base"),
                        "world_size": int(world_size),
                        "completed_epoch": int(epoch_number),
                        "planned_n_train": int(args.n_train),
                        "params": params,
                        "model_state_dict": (
                            model.module.state_dict()
                            if hasattr(model, "module")
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch_history": epoch_history,
                        "runtime_states": runtime_states,
                        "potential_mode": getattr(
                            args, "potential_mode", "partial_vrho"
                        ),
                        "data_protocol": getattr(args, "data_protocol", "legacy_vrho"),
                        "lap_s5_provenance": lap_s5_provenance if is_lap_mode else None,
                        "lap_checkpoint_extra": lap_checkpoint_extra
                        if is_lap_mode
                        else None,
                    },
                    training_state_path,
                )

        if rank0:
            print(
                f"Trial {trial_number} epoch {epoch + 1}/{args.n_train}: "
                f"train_fchem={row['train_fchem']:.8f} "
                f"train_vxc={row['train_vxc']:.8f} "
                f"train_exc={row['train_exc']:.8f} "
                f"phase={row['phase_name']}"
            )

    failed = sync_failure(failed, device)
    if failed:
        return {
            "failed": True,
            "trial_number": trial_number,
            "params": params,
        }

    if rank0:
        final_payload = (
            _lap_s5_model_payload(
                model,
                lap_s5_provenance,
                **lap_checkpoint_extra,
                epoch=int(epoch_history[-1]["epoch"]),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                training_protocol=TRAINING_PROTOCOL,
                data_protocol="lap_full_vxc" if is_lap_mode else "legacy_vrho",
                potential_mode="full_euler" if is_lap_mode else "partial_vrho",
            )
            if is_lap_mode
            else (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
        )
        atomic_torch_save(final_payload, final_checkpoint_path)
    trial_payload = {
        "failed": False,
        "trial_number": trial_number,
        "params": params,
        "final_epoch": int(epoch_history[-1]["epoch"]),
        "training_protocol": TRAINING_PROTOCOL,
        "data_protocol": "lap_full_vxc" if is_lap_mode else "legacy_vrho",
        "potential_mode": "full_euler" if is_lap_mode else "partial_vrho",
        "lap_s5_provenance": lap_s5_provenance if is_lap_mode else None,
        "lap_checkpoint_extra": lap_checkpoint_extra if is_lap_mode else None,
        "epoch_history": epoch_history,
        "final_checkpoint_path": str(final_checkpoint_path),
        "epoch_snapshots": str(snapshot_dir) if snapshot_every > 0 else None,
        "training_state_path": str(training_state_path)
        if training_state_every > 0
        else None,
    }
    if rank0:
        history_path = save_trial_history(output_dir, trial_number, trial_payload)
        trial_payload["history_path"] = str(history_path)
    else:
        trial_payload["history_path"] = None
    return trial_payload


def _lap_s5_model_payload(model, provenance, **metadata):
    from lap_checkpoint import checkpoint_payload
    from lap_s5_provenance import validate_lap_s5_provenance

    if provenance is None:
        raise ValueError("Lap-S5 checkpoint requires validated protocol provenance.")
    validate_lap_s5_provenance(provenance)
    base_model = model.module if hasattr(model, "module") else model
    return checkpoint_payload(base_model, lap_s5_provenance=provenance, **metadata)


def main() -> None:
    raise SystemExit(
        "Standalone Optuna search is disabled: external DietGMTKN30 SCF validation "
        "is required for model selection. Use replay_trial_19_bridge.py for training "
        "and evaluate epoch snapshots externally."
    )


if __name__ == "__main__":
    main()
