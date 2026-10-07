"""Prepare all canonical mRKS AO caches using the qualified existing builder."""

import argparse
import json
from pathlib import Path

from train_models.lap_moo_panel import build_ao_factor_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.central / "manifest.json").read_text())
    names = tuple(sorted(row["system_name"] for row in manifest["records"]))
    if len(names) != 90 or len(set(names)) != 90:
        raise ValueError("Exactly 90 unique canonical systems required")
    build_ao_factor_cache(args.central, args.output, chunk_size=2048, systems=names)


if __name__ == "__main__":
    main()
