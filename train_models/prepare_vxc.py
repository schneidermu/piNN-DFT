from __future__ import annotations

import os
import pickle
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from prepare_data import TRAINING_PROTOCOL, remove_obsolete
import json


def load_vxc_from_h5(h5_path: str | Path) -> dict | None:
    """
    Loads Grid, Vrho, Weights, and exact E_xc from an H5 file.
    """
    if not os.path.exists(h5_path):
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. Load Vrho (Target)
            if 'vrho' not in f: 
                print(f"Skipping {h5_path}: No 'vrho' dataset.")
                return None
            vrho_raw = f['vrho'][:]
            
            # Handle shape: (2, N) -> (N,)
            # Vxc is intensive. For closed shell (like Ne), rows are identical.
            # We average them to get a single 1D array matching the density grid.
            if vrho_raw.ndim == 2:
                vrho = np.mean(vrho_raw, axis=0)
            else:
                vrho = vrho_raw
            
            # Convert to Tensor
            vrho = torch.tensor(vrho, dtype=torch.float32)

            # 2. Load Weights (Integration weights)
            if 'weights' not in f: return None
            weights = torch.tensor(f['weights'][:], dtype=torch.float32)

            # 3. Load Grid (Input features: rho, grad, tau, etc.)
            if 'grid' not in f: return None
            grid = torch.tensor(f['grid'][:], dtype=torch.float32)

            # 4. Load exact integrated exchange-correlation energy.
            if 'E_xc' not in f:
                print(f"Skipping {h5_path}: No 'E_xc' dataset.")
                return None
            e_xc = torch.tensor(float(f['E_xc'][()]), dtype=torch.float32)
            
            # 5. Check Shapes
            if not (grid.shape[0] == vrho.shape[0] == weights.shape[0]):
                print(f"Skipping {h5_path}: Shape mismatch G{grid.shape} V{vrho.shape} W{weights.shape}")
                return None

            return {
                "Name": Path(h5_path).stem,
                "Grid": grid,
                "Vrho": vrho,
                "Weights": weights,
                "E_xc": e_xc,
            }
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None


def prepare_vxc(h5_dir: str = "h5_vrho", output_dir: str = "checkpoints") -> list[dict]:
    """Save every valid mRKS system for both E_xc and v_xc training."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    remove_obsolete(directory, ("data_vxc_val.pickle",))
    files = sorted(Path(h5_dir).glob("*.h5"))
    valid_data = []
    for file in files:
        data = load_vxc_from_h5(file)
        if data is not None:
            valid_data.append(data)
    # Always overwrite, even for an empty corpus; never leave stale training data.
    with (directory / "data_vxc_train.pickle").open("wb") as handle:
        pickle.dump(valid_data, handle)
    (directory / "mrks_protocol.json").write_text(json.dumps({
        "protocol": TRAINING_PROTOCOL, "valid_systems": len(valid_data),
        "source_files": [str(file.resolve()) for file in files],
    }))
    print(f"mRKS training systems: {len(valid_data)} / {len(files)} H5 files; no internal validation split.")
    return valid_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", default="h5_vrho_from_mrks")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    prepare_vxc(h5_dir=args.h5_dir, output_dir=args.output_dir)
