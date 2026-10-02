"""Optimizer families used by the one-stage Lap MOO trainer.

The RAdamW path delegates to the historical optimizer factory unchanged. The
Muon comparison uses PyTorch's native ``torch.optim.Muon`` for hidden 2-D
Linear weights and AdamW for all remaining trainable parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

try:
    from .utils import configure_optimizers
except ImportError:  # pragma: no cover - direct script import
    from utils import configure_optimizers

OPTIMIZER_FAMILIES = ("radamw", "adamw", "muon_adamw")
MUON_HIDDEN_MODULE_PREFIXES = (
    "c_input_layers",
    "c_symmetrization_blocks",
    "c_post_symm_blocks",
    "x_feature_extractor",
)
MUON_DEFAULTS = {
    "momentum": 0.95,
    "nesterov": True,
    "ns_coefficients": (3.4445, -4.775, 2.0315),
    "eps": 1e-7,
    "ns_steps": 5,
    "adjust_lr_fn": "original",
}


class MuonAdamW(torch.optim.Optimizer):
    """A scheduler-compatible wrapper around native Muon and AdamW.

    The child optimizers own their state dictionaries. ``param_groups`` points
    at those same group dictionaries so ordinary PyTorch schedulers update the
    child learning rates in place.
    """

    state_format_version = 1

    def __init__(self, muon_optimizer: torch.optim.Optimizer, adamw_optimizer: torch.optim.Optimizer):
        groups = [
            dict(group)
            for group in (*muon_optimizer.param_groups, *adamw_optimizer.param_groups)
        ]
        super().__init__(groups, defaults={})
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer
        muon_group_count = len(muon_optimizer.param_groups)
        self.muon_optimizer.param_groups = self.param_groups[:muon_group_count]
        self.adamw_optimizer.param_groups = self.param_groups[muon_group_count:]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        muon_loss = self.muon_optimizer.step(closure=closure)
        self.adamw_optimizer.step()
        return muon_loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "lap-muon-adamw-v1",
            "muon": self.muon_optimizer.state_dict(),
            "adamw": self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("format") != "lap-muon-adamw-v1":
            raise ValueError("Incompatible MuonAdamW optimizer checkpoint.")
        if set(state_dict) != {"format", "muon", "adamw"}:
            raise ValueError("MuonAdamW checkpoint has unexpected optimizer state.")
        self.muon_optimizer.load_state_dict(state_dict["muon"])
        self.adamw_optimizer.load_state_dict(state_dict["adamw"])
        # Optimizer.load_state_dict replaces its param_groups list. Keep the
        # scheduler-facing wrapper synchronized with those restored groups.
        self.param_groups = [
            *self.muon_optimizer.param_groups,
            *self.adamw_optimizer.param_groups,
        ]

    def learning_rates(self) -> dict[str, float]:
        return {
            "muon": float(self.muon_optimizer.param_groups[0]["lr"]),
            "adamw_decay": float(self.adamw_optimizer.param_groups[0]["lr"]),
            "adamw_no_decay": float(self.adamw_optimizer.param_groups[1]["lr"]),
        }


def _trainable_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    identities = [id(parameter) for parameter in parameters.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("Model parameter iterator contains a tied parameter more than once.")
    return parameters


def partition_muon_parameters(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """Return (hidden Muon weights, AdamW fallback), preserving unique names.

    Eligibility is based on module role, not shape alone. Only the 2-D weights
    of hidden Linear layers in the Lap architecture's four hidden module trees
    are eligible. Output heads, biases, LayerNorms, and other tensors fall back
    to AdamW.
    """
    named_parameters = _trainable_named_parameters(model)
    hidden_weight_names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_hidden = any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in MUON_HIDDEN_MODULE_PREFIXES
        )
        if is_hidden and module.weight.requires_grad and module.weight.ndim == 2:
            parameter_name = f"{module_name}.weight" if module_name else "weight"
            if parameter_name not in named_parameters:
                raise ValueError(
                    f"Eligible Muon parameter {parameter_name!r} is missing from the unique parameter list."
                )
            hidden_weight_names.add(parameter_name)

    if not hidden_weight_names:
        raise ValueError("MuonAdamW found no eligible hidden 2-D Linear weights.")
    muon = [(name, named_parameters[name]) for name in named_parameters if name in hidden_weight_names]
    adamw = [(name, parameter) for name, parameter in named_parameters.items() if name not in hidden_weight_names]
    all_names = [name for name, _ in (*muon, *adamw)]
    if len(all_names) != len(named_parameters) or set(all_names) != set(named_parameters):
        raise ValueError("Muon/AdamW partition does not cover every trainable parameter exactly once.")
    if any(parameter.ndim != 2 for _, parameter in muon):
        raise ValueError("Muon eligibility unexpectedly included a non-matrix parameter.")
    return muon, adamw


def _adamw_groups(
    model: nn.Module,
    parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Match the repository's decoupled decay policy for fallback parameters."""
    parameter_map = dict(parameters)
    if not parameter_map:
        raise ValueError("AdamW fallback group cannot be empty.")
    decay_names: set[str] = set()
    no_decay_names: set[str] = set()
    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            if full_name not in parameter_map or not parameter.requires_grad:
                continue
            if local_name.endswith("bias"):
                no_decay_names.add(full_name)
            elif local_name.endswith("weight") and isinstance(module, nn.Linear):
                decay_names.add(full_name)
            elif local_name.endswith("weight") and isinstance(
                module, (nn.LayerNorm, nn.PReLU, nn.BatchNorm1d)
            ):
                no_decay_names.add(full_name)
            else:
                no_decay_names.add(full_name)
    if decay_names & no_decay_names:
        raise ValueError("A fallback parameter was assigned both decay policies.")
    if (decay_names | no_decay_names) != set(parameter_map):
        missing = sorted(set(parameter_map) - (decay_names | no_decay_names))
        raise ValueError(f"Fallback parameters have no AdamW decay policy: {missing}")
    return [
        {
            "params": [parameter_map[name] for name in sorted(decay_names)],
            "weight_decay": float(weight_decay),
            "group_name": "adamw_decay",
        },
        {
            "params": [parameter_map[name] for name in sorted(no_decay_names)],
            "weight_decay": 0.0,
            "group_name": "adamw_no_decay",
        },
    ]


def _ensure_native_muon_bfloat16(device: torch.device) -> None:
    """Fail early where Muon's native BF16 Newton-Schulz path is unsupported."""
    if device.type != "cuda":
        return
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError(
            "Native torch.optim.Muon orthogonalization requires CUDA BF16 matrix "
            f"operations; device {device} has compute capability {capability}. "
            "No altered-precision Muon fallback is implemented."
        )
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(
                f"Native BF16 Muon orthogonalization is unavailable on {device}."
            )


def build_optimizer(
    model: nn.Module,
    *,
    family: str = "radamw",
    learning_rate: float,
    weight_decay: float = 1e-2,
    muon_learning_rate: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build an MOO optimizer and its protocol metadata.

    For ``muon_adamw``, ``learning_rate`` is the AdamW fallback LR and
    ``muon_learning_rate`` is independently required for hidden matrices.
    """
    if family not in OPTIMIZER_FAMILIES:
        raise ValueError(f"Unknown optimizer family {family!r}; expected {OPTIMIZER_FAMILIES}.")
    if not (learning_rate > 0 and torch.isfinite(torch.tensor(learning_rate))):
        raise ValueError("Learning rates must be positive and finite.")
    if weight_decay < 0 or not torch.isfinite(torch.tensor(weight_decay)):
        raise ValueError("Weight decay must be finite and nonnegative.")

    if family == "radamw":
        if muon_learning_rate is not None:
            raise ValueError("--muon-learning-rate is only valid for muon_adamw.")
        optimizer = configure_optimizers(
            model, learning_rate, optimizer_str="radamw", weight_decay=weight_decay
        )
        metadata = {
            "family": family,
            "name": "RAdamW",
            "torch_version": torch.__version__,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "decoupled_weight_decay": True,
            "parameter_groups": "Linear weights decayed; biases and norm weights excluded",
        }
        return optimizer, metadata

    if family == "adamw":
        if muon_learning_rate is not None:
            raise ValueError("--muon-learning-rate is only valid for muon_adamw.")
        optimizer = configure_optimizers(
            model, learning_rate, optimizer_str="adamw", weight_decay=weight_decay
        )
        metadata = {
            "family": family,
            "name": "AdamW",
            "torch_version": torch.__version__,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "decoupled_weight_decay": True,
            "parameter_groups": "Linear weights decayed; biases and norm weights excluded",
        }
        return optimizer, metadata

    if muon_learning_rate is None or muon_learning_rate <= 0 or not torch.isfinite(
        torch.tensor(muon_learning_rate)
    ):
        raise ValueError("muon_adamw requires a positive finite Muon learning rate.")
    muon_named, adamw_named = partition_muon_parameters(model)
    parameter_device = muon_named[0][1].device
    _ensure_native_muon_bfloat16(parameter_device)
    muon_optimizer = torch.optim.Muon(
        [parameter for _, parameter in muon_named],
        lr=float(muon_learning_rate),
        weight_decay=float(weight_decay),
        **MUON_DEFAULTS,
    )
    adamw_optimizer = torch.optim.AdamW(
        _adamw_groups(model, adamw_named, weight_decay=weight_decay),
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer = MuonAdamW(muon_optimizer, adamw_optimizer)
    all_names = {name for name, _ in (*muon_named, *adamw_named)}
    metadata = {
        "family": family,
        "name": "torch.optim.Muon + torch.optim.AdamW",
        "torch_version": torch.__version__,
        "muon_learning_rate": float(muon_learning_rate),
        "adamw_fallback_learning_rate": float(learning_rate),
        "muon_weight_decay": float(weight_decay),
        "adamw_weight_decay": float(weight_decay),
        "adamw_fallback_betas": [0.9, 0.999],
        "adamw_fallback_eps": 1e-8,
        "adamw_fallback_decoupled_weight_decay": True,
        "muon": {
            **MUON_DEFAULTS,
            "ns_coefficients": list(MUON_DEFAULTS["ns_coefficients"]),
            "orthogonalization_dtype": "bfloat16 (native PyTorch implementation)",
        },
        "muon_implementation_source": "https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/_muon.py",
        "fallback_rate_reference": "PyTorch Muon example: Muon lr=0.02, AdamW fallback lr=3e-4",
        "muon_parameter_names": [name for name, _ in muon_named],
        "adamw_fallback_parameter_names": [name for name, _ in adamw_named],
        "all_trainable_parameters_included_once": len(all_names)
        == sum(parameter.requires_grad for parameter in model.parameters()),
        "ddp_semantics": "all-reduce each raw task gradient, then aggregate globally, then apply identical optimizer state on each rank",
    }
    if not metadata["all_trainable_parameters_included_once"]:
        raise ValueError("Muon/AdamW parameter groups do not include every trainable parameter exactly once.")
    return optimizer, metadata


__all__ = [
    "MUON_DEFAULTS",
    "MUON_HIDDEN_MODULE_PREFIXES",
    "OPTIMIZER_FAMILIES",
    "MuonAdamW",
    "build_optimizer",
    "partition_muon_parameters",
]
