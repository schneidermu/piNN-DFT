"""Regenerate and verify the shared, immutable S5/timing training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def regenerate(mn_dir: str, mrks_dir: str, output_dir: str) -> Path:
    """Build a new directory once; never overwrite a historical/shared corpus."""
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing corpus: {directory}")
    for source in (mn_dir, mrks_dir):
        if not Path(source).is_dir() or not any(Path(source).glob("*.h5")):
            raise ValueError(f"Raw H5 sources are missing or empty: {source}")
    base = get_compounds_coefs_energy(
        load_component_names(mn_dir), load_ref_energies(mn_dir)
    )
    clean = filter_minnesota_training(base)
    predopt, train = prepare(mn_dir)
    if not predopt or not train:
        raise ValueError(
            "No augmented Minnesota training samples; check source H5 availability."
        )
    save_chk(predopt, train, str(directory))
    mrks = prepare_vxc(mrks_dir, str(directory))
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
        "source_directories": {
            "minnesota": str(Path(mn_dir).resolve()),
            "mrks": str(Path(mrks_dir).resolve()),
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
        manifest.get("minnesota_training_reactions")
        != manifest.get("minnesota_source_reactions", 0) - 16
    ):
        raise ValueError(
            "Minnesota source/training counts violate the exclusion contract."
        )
    if (
        manifest.get("mrks_systems", 0) < 1
        or manifest.get("augmented_reaction_samples", 0) < 1
    ):
        raise ValueError("Training corpus is empty.")
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
    parser.add_argument("--output-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(Path(args.output_dir))
    else:
        regenerate(args.mn_dir, args.mrks_dir, args.output_dir)


if __name__ == "__main__":
    main()
