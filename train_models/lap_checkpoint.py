"""Explicit architecture/protocol metadata; old tau checkpoints fail closed."""

import torch
from lap_data import PROTOCOL
from lap_vxc import (
    STENCIL_DERIVATIVE_ORDER,
    STENCIL_VERSION,
    stencil_order_for_version,
    stencil_positions_for_version,
    validate_stencil_selection,
)
from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL, pcPBELMLOptimizerV2Lap

_STENCIL_FIELDS = {"stencil_version", "stencil_order", "derivative_order"}


def checkpoint_payload(model, **metadata):
    if {
        "architecture",
        "descriptor_protocol",
        "protocol",
        "model_kwargs",
        "model_state_dict",
        *_STENCIL_FIELDS,
    } & metadata.keys():
        raise ValueError(
            "Checkpoint metadata cannot override architecture/protocol/state."
        )
    provenance = metadata.get("lap_s5_provenance")
    if provenance is not None:
        stencil = provenance.get("stencil")
        if not isinstance(stencil, dict):
            raise ValueError(
                "Lap S5 provenance must include explicit stencil metadata."
            )
        version = stencil.get("version")
        derivative_order = stencil.get("derivative_order")
        positions = stencil.get("stencil_order")
        validate_stencil_selection(derivative_order, version)
        if positions != list(stencil_positions_for_version(version)):
            raise ValueError("Lap S5 stencil offset layout does not match its version.")
    else:
        version = STENCIL_VERSION
        derivative_order = STENCIL_DERIVATIVE_ORDER
        positions = list(stencil_positions_for_version(version))
    return dict(
        architecture=ARCHITECTURE,
        descriptor_protocol=DESCRIPTOR_PROTOCOL,
        protocol=PROTOCOL,
        model_kwargs=model.model_kwargs,
        model_state_dict=model.state_dict(),
        stencil_version=version,
        stencil_order=list(positions),
        derivative_order=derivative_order,
        **metadata,
    )


def _validate_checkpoint_stencil(payload):
    fields_present = _STENCIL_FIELDS & payload.keys()
    if fields_present and fields_present != _STENCIL_FIELDS:
        raise ValueError("Checkpoint stencil version/order metadata is incomplete.")
    provenance = payload.get("lap_s5_provenance")
    if (
        isinstance(provenance, dict)
        and provenance.get("metadata_version") == 1
        and isinstance(provenance.get("stencil"), dict)
        and set(provenance["stencil"]) == {"version", "h_bohr", "units"}
        and provenance["stencil"].get("version") == STENCIL_VERSION
    ):
        # Old S5 checkpoint provenance explicitly names the persisted 7-point
        # schema. Promote only that exact shape of metadata on read.
        provenance = dict(provenance)
        provenance["metadata_version"] = 2
        provenance["stencil"] = {
            **provenance["stencil"],
            "stencil_order": list(stencil_positions_for_version(STENCIL_VERSION)),
            "derivative_order": stencil_order_for_version(STENCIL_VERSION),
        }
        payload["lap_s5_provenance"] = provenance
    provenance_stencil = (
        provenance.get("stencil") if isinstance(provenance, dict) else None
    )
    if fields_present:
        version = payload["stencil_version"]
        derivative_order = payload["derivative_order"]
        positions = payload["stencil_order"]
    elif provenance_stencil:
        version = provenance_stencil.get("version")
        derivative_order = provenance_stencil.get("derivative_order")
        if derivative_order is None and version == STENCIL_VERSION:
            derivative_order = stencil_order_for_version(version)
        positions = provenance_stencil.get("stencil_order")
        if positions is None and version == STENCIL_VERSION:
            positions = list(stencil_positions_for_version(version))
    else:
        # Checkpoints predating stencil metadata are safely bound to the
        # historical 7-point ID because that was the only persisted schema.
        version = STENCIL_VERSION
        derivative_order = STENCIL_DERIVATIVE_ORDER
        positions = list(stencil_positions_for_version(version))
    validate_stencil_selection(derivative_order, version)
    if positions != list(stencil_positions_for_version(version)):
        raise ValueError("Checkpoint stencil offset layout does not match its version.")
    if provenance_stencil is not None and (
        provenance_stencil.get("version") != version
        or provenance_stencil.get("derivative_order", derivative_order)
        != derivative_order
        or provenance_stencil.get("stencil_order", positions) != positions
    ):
        raise ValueError("Checkpoint and Lap S5 stencil metadata disagree.")
    payload.setdefault("stencil_version", version)
    payload.setdefault("derivative_order", derivative_order)
    payload.setdefault("stencil_order", list(positions))


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
    _validate_checkpoint_stencil(payload)
    model = pcPBELMLOptimizerV2Lap(**payload["model_kwargs"]).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
