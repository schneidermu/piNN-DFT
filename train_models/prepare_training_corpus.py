"""Regenerate and verify the shared, immutable S5/timing training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle

import torch
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dataset import get_compounds_coefs_energy, load_component_names, load_ref_energies
from prepare_data import (
    DIET_HELD_OUT_MN_REACTIONS,
    TRAINING_PROTOCOL,
    filter_minnesota_training,
    prepare,
    require_protocol,
    save_chk,
)
from prepare_vxc import prepare_vxc

DEFAULT_CORPUS_DIR = "checkpoints_dietclean_noval_v1"
ARTIFACTS = (
    "data_predopt.pickle",
    "data_train_grouped.pickle",
    "data_vxc_train.pickle",
    "minnesota_protocol.json",
    "mrks_protocol.json",
)
MANIFEST_NAME = "preprocessing_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusion_pairs() -> list[list[str | int]]:
    return [
        [db, rid]
        for db, ids in sorted(DIET_HELD_OUT_MN_REACTIONS.items())
        for rid in sorted(ids)
    ]


def merge_mrks_splits(train_path: str, val_path: str) -> tuple[list[dict], list[dict]]:
    """Preserve existing reference targets; validate and combine historical splits."""
    merged = []
    sources = []
    names = set()
    for source in (Path(train_path).resolve(), Path(val_path).resolve()):
        digest = sha256(source)
        with source.open("rb") as handle:
            samples = pickle.load(handle)
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError(f"Expected a nonempty mRKS sample list: {source}")
        for sample in samples:
            required = {"Name", "Grid", "Vrho", "Weights", "E_xc"}
            if not isinstance(sample, dict) or not required <= sample.keys():
                raise ValueError(f"Missing mRKS fields including E_xc: {source}")
            name = sample["Name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"Missing or duplicate mRKS system identity: {name!r}")
            values = {key: torch.as_tensor(sample[key]) for key in required - {"Name"}}
            grid = values["Grid"]
            if (
                grid.ndim != 2
                or grid.shape[0] == 0
                or grid.shape[1] < 12
                or values["Vrho"].shape != (grid.shape[0],)
                or values["Weights"].shape != (grid.shape[0],)
                or values["E_xc"].numel() != 1
            ):
                raise ValueError(f"Invalid mRKS target/grid shapes: {name}")
            if not all(torch.isfinite(value).all().item() for value in values.values()):
                raise ValueError(f"Nonfinite mRKS reference data: {name}")
            names.add(name)
            merged.append(sample)
        if sha256(source) != digest:
            raise ValueError(f"mRKS source changed during import: {source}")
        sources.append({"path": str(source), "sha256": digest, "samples": len(samples)})
    print(f"Merged mRKS training systems: {len(merged)}; no internal validation split.")
    return merged, sources


def regenerate(
    mn_dir: str,
    mrks_dir: str,
    output_dir: str,
    mrks_train_pickle: str | None = None,
    mrks_val_pickle: str | None = None,
) -> Path:
    """Build a new directory once; never overwrite a historical/shared corpus."""
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing corpus: {directory}")
    if bool(mrks_train_pickle) != bool(mrks_val_pickle):
        raise ValueError("Supply both mRKS train and validation pickle paths.")
    sources = []
    mrks = None
    if mrks_train_pickle:
        mrks, sources = merge_mrks_splits(mrks_train_pickle, mrks_val_pickle)
    for source in (mn_dir,) if mrks is not None else (mn_dir, mrks_dir):
        if not Path(source).is_dir() or not any(Path(source).glob("*.h5")):
            raise ValueError(f"Raw H5 sources are missing or empty: {source}")
    base = get_compounds_coefs_energy(
        load_component_names(mn_dir), load_ref_energies(mn_dir)
    )
    clean = filter_minnesota_training(base)
    if len(base) != 284 or len(clean) != 268:
        raise ValueError("Expected 284 source and 268 cleaned Minnesota reactions.")
    predopt, train = prepare(mn_dir)
    if not predopt or not train:
        raise ValueError(
            "No augmented Minnesota training samples; check source H5 availability."
        )
    save_chk(predopt, train, str(directory))
    if mrks is None:
        mrks = prepare_vxc(mrks_dir, str(directory))
    else:
        with (directory / "data_vxc_train.pickle").open("wb") as handle:
            pickle.dump(mrks, handle)
        (directory / "mrks_protocol.json").write_text(
            json.dumps(
                {
                    "protocol": TRAINING_PROTOCOL,
                    "valid_systems": len(mrks),
                    "source_mode": "merged-existing-splits",
                    "source_pickles": sources,
                },
                indent=2,
            )
            + "\n"
        )
    if not mrks:
        raise ValueError(
            "No valid mRKS systems; corpus is incomplete and has no manifest."
        )
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    manifest = {
        "manifest_version": 1,
        "training_protocol": TRAINING_PROTOCOL,
        "git_commit": commit,
        "preprocessing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "minnesota_source_reactions": len(base),
        "excluded_minnesota_reactions": exclusion_pairs(),
        "minnesota_training_reactions": len(clean),
        "available_augmented_base_reactions": len(train),
        "augmented_reaction_samples": sum(len(group) for group in train.values()),
        "predopt_samples": len(predopt),
        "mrks_systems": len(mrks),
        "mrks_source_mode": "merged-existing-splits" if sources else "raw-h5",
        "mrks_source_pickles": sources,
        "source_directories": {
            "minnesota": str(Path(mn_dir).resolve()),
            "mrks": None if sources else str(Path(mrks_dir).resolve()),
        },
        "artifact_sha256": {name: sha256(directory / name) for name in ARTIFACTS},
    }
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    verify(directory)
    return path


def verify(directory: Path) -> dict:
    """Fail before torchrun if provenance is missing or artifacts have changed."""
    directory = directory.resolve()
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"Missing {path}; regenerate with prepare_training_corpus.py.")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("manifest_version") != 1
        or manifest.get("training_protocol") != TRAINING_PROTOCOL
    ):
        raise ValueError("Unsupported preprocessing manifest/protocol.")
    if manifest.get("excluded_minnesota_reactions") != exclusion_pairs():
        raise ValueError(
            "Preprocessing manifest does not contain the exact 16 exclusions."
        )
    if (
        manifest.get("minnesota_source_reactions") != 284
        or manifest.get("minnesota_training_reactions") != 268
    ):
        raise ValueError(
            "Minnesota source/training counts violate the exclusion contract."
        )
    if (
        manifest.get("mrks_systems", 0) < 1
        or manifest.get("augmented_reaction_samples", 0) < 1
    ):
        raise ValueError("Training corpus is empty.")
    if manifest.get("mrks_source_mode") == "merged-existing-splits":
        sources = manifest.get("mrks_source_pickles", [])
        marker = json.loads((directory / "mrks_protocol.json").read_text())
        if (
            len(sources) != 2
            or sum(source.get("samples", 0) for source in sources)
            != manifest["mrks_systems"]
            or any(len(source.get("sha256", "")) != 64 for source in sources)
            or marker.get("source_pickles") != sources
            or marker.get("valid_systems") != manifest["mrks_systems"]
        ):
            raise ValueError("Invalid merged mRKS provenance/counts.")
    for marker in ("minnesota_protocol.json", "mrks_protocol.json"):
        require_protocol(directory, marker)
    hashes = manifest.get("artifact_sha256", {})
    if set(hashes) != set(ARTIFACTS):
        raise ValueError("Incomplete artifact hashes in preprocessing manifest.")
    for name in ARTIFACTS:
        if not (directory / name).is_file() or sha256(directory / name) != hashes[name]:
            raise ValueError(f"Training artifact changed after preprocessing: {name}")
    print(f"Preprocessing manifest: {path}")
    print(f"Preprocessing manifest SHA256: {sha256(path)}")
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mn-dir", default="data")
    parser.add_argument("--mrks-dir", default="h5_vrho_from_mrks")
    parser.add_argument("--mrks-train-pickle")
    parser.add_argument("--mrks-val-pickle")
    parser.add_argument("--output-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(Path(args.output_dir))
    else:
        regenerate(
            args.mn_dir,
            args.mrks_dir,
            args.output_dir,
            args.mrks_train_pickle,
            args.mrks_val_pickle,
        )


if __name__ == "__main__":
    main()
