"""Explicit architecture/protocol metadata; old tau checkpoints fail closed."""

import torch
from lap_data import PROTOCOL
from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL, pcPBELMLOptimizerV2Lap


def checkpoint_payload(model, **metadata):
    if {
        "architecture",
        "descriptor_protocol",
        "protocol",
        "model_kwargs",
        "model_state_dict",
    } & metadata.keys():
        raise ValueError(
            "Checkpoint metadata cannot override architecture/protocol/state."
        )
    return dict(
        architecture=ARCHITECTURE,
        descriptor_protocol=DESCRIPTOR_PROTOCOL,
        protocol=PROTOCOL,
        model_kwargs=model.model_kwargs,
        model_state_dict=model.state_dict(),
        **metadata,
    )


def load_lap_checkpoint(path, device="cpu", dtype=torch.float64):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("architecture"),
        payload.get("descriptor_protocol"),
        payload.get("protocol"),
    ) != (ARCHITECTURE, DESCRIPTOR_PROTOCOL, PROTOCOL):
        raise ValueError(
            "Architecture/protocol mismatch: explicit Lap checkpoint metadata required."
        )
    model = pcPBELMLOptimizerV2Lap(**payload["model_kwargs"]).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
