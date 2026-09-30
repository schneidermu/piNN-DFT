"""Prepare training-only Minnesota data; validation/test are external SCF."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import torch

from dataset import (
    build_file_index,
    get_compounds_coefs_energy,
    group_and_augment_reactions,
    load_component_names,
    load_ref_energies,
)

DIET_HELD_OUT_MN_REACTIONS = {
    "DBH76": {15, 35, 42, 43, 54, 55},
    "MGAE109": {18, 28, 34, 55, 74},
    "EA13": {4, 8},
    "NCCE31": {12, 21, 30},
}
TRAINING_PROTOCOL = "diet-clean-mn-all-mrks-external-scf-v1"
OBSOLETE_PICKLES = ("data_test_grouped.pickle", "data_vxc_val.pickle")


def filter_minnesota_training(base_reactions: dict) -> dict:
    """Exclude by original per-database ID, preserving all other base keys."""
    return {
        key: reaction
        for key, reaction in base_reactions.items()
        if reaction["ReactionID"]
        not in DIET_HELD_OUT_MN_REACTIONS.get(reaction["Database"], set())
    }


def flatten_grouped_data(grouped_data: dict) -> dict:
    return {
        i: reaction
        for i, reaction in enumerate(
            reaction for group in grouped_data.values() for reaction in group
        )
    }


def prepare(path: str = "data") -> tuple[dict, dict]:
    """Augment one Diet-cleaned pool for both Minnesota training and predopt."""
    base_reactions = get_compounds_coefs_energy(
        load_component_names(path), load_ref_energies(path)
    )
    train = filter_minnesota_training(base_reactions)
    excluded = [r for k, r in base_reactions.items() if k not in train]
    expected = {
        (db, rid) for db, ids in DIET_HELD_OUT_MN_REACTIONS.items() for rid in ids
    }
    actual = {(r["Database"], r["ReactionID"]) for r in excluded}
    if actual != expected or len(excluded) != 16:
        raise ValueError(
            f"Expected exactly 16 Diet overlaps; found {actual}. Check Minnesota source data."
        )
    print(f"Minnesota base reactions total: {len(base_reactions)}")
    print(f"Diet-overlap exclusions: {len(excluded)}")
    for db, ids in sorted(DIET_HELD_OUT_MN_REACTIONS.items()):
        print(f"  {db}: {', '.join(map(str, sorted(ids)))}")
    print(f"Minnesota training reactions: {len(train)}")
    grouped = group_and_augment_reactions(train, build_file_index(path))
    # Copy before tensor conversion, so predopt does not mutate grouped samples.
    flat = {
        i: {**r, "Grid": torch.Tensor(r["Grid"])}
        for i, r in flatten_grouped_data(grouped).items()
    }
    print(f"Available augmented training/predopt samples: {len(flat)}")
    return flat, grouped


def remove_obsolete(path: Path, names: tuple[str, ...]) -> None:
    for name in names:
        stale = path / name
        if stale.exists():
            stale.unlink()
            print(f"Removed obsolete non-SCF validation artifact: {stale}")


def save_chk(data: dict, data_train: dict, path: str = "checkpoints") -> None:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    remove_obsolete(directory, ("data_test_grouped.pickle",))
    for name, payload in (
        ("data_predopt.pickle", data),
        ("data_train_grouped.pickle", data_train),
    ):
        with (directory / name).open("wb") as handle:
            pickle.dump(payload, handle)
    (directory / "minnesota_protocol.json").write_text(
        json.dumps({"protocol": TRAINING_PROTOCOL})
    )


def require_protocol(directory: Path, filename: str) -> None:
    marker = directory / filename
    if (
        not marker.exists()
        or json.loads(marker.read_text()).get("protocol") != TRAINING_PROTOCOL
    ):
        raise ValueError(
            f"Stale or unverified training corpus at {directory}; rerun prepare_data.py and prepare_vxc.py."
        )


def load_chk(path: str = "checkpoints") -> tuple[dict, dict, list[dict[str, Any]]]:
    """Load only training corpora; reject unversioned historical random splits."""
    directory = Path(path)
    for name in OBSOLETE_PICKLES:
        if (directory / name).exists():
            warnings.warn(
                f"Ignoring obsolete non-SCF validation artifact: {directory / name}",
                stacklevel=2,
            )
    require_protocol(directory, "minnesota_protocol.json")
    require_protocol(directory, "mrks_protocol.json")
    payloads = []
    for name in (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "data_vxc_train.pickle",
    ):
        with (directory / name).open("rb") as handle:
            payloads.append(pickle.load(handle))
    predopt, train, vxc = payloads
    # Fail closed if a mislabeled corpus contains benchmark-overlap supervision.
    for reaction in [
        *predopt.values(),
        *(r for group in train.values() for r in group),
    ]:
        if "ReactionID" not in reaction or reaction[
            "ReactionID"
        ] in DIET_HELD_OUT_MN_REACTIONS.get(reaction["Database"], set()):
            raise ValueError(
                "Invalid Minnesota training provenance; rerun prepare_data.py."
            )
    print(f"Loaded {len(vxc)} mRKS training systems (both E_xc and v_xc).")
    return predopt, train, vxc


if __name__ == "__main__":
    save_chk(*prepare())
    print("Training-only data preparation complete.")
