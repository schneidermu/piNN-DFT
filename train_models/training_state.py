import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_torch_payload(path: Path, map_location) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_runtime_state(loaders: Dict[str, Any]) -> Dict[str, Any]:
    loader_generator_states = {}
    for name, value in loaders.items():
        generator = getattr(value, "generator", None)
        if generator is not None:
            loader_generator_states[name] = generator.get_state()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator_states": loader_generator_states,
    }


def restore_runtime_state(payload: Dict[str, Any], loaders: Dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
    for name, state in payload.get("loader_generator_states", {}).items():
        generator = getattr(loaders.get(name), "generator", None)
        if generator is not None:
            generator.set_state(state)
