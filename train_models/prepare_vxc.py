import os
import pickle
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    train_test_split = None


def split_train_val(data, test_size, random_state):
    if train_test_split is not None:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    n_val = int(np.ceil(len(data) * test_size))
    n_val = min(max(n_val, 1), len(data) - 1)
    val_indices = set(indices[:n_val].tolist())
    train_data = [item for idx, item in enumerate(data) if idx not in val_indices]
    val_data = [item for idx, item in enumerate(data) if idx in val_indices]
    return train_data, val_data

def load_vxc_from_h5(h5_path):
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


def prepare_vxc(h5_dir="h5_vrho", output_dir="checkpoints", test_size=0.1, random_state=42):
    """
    Scans directory, loads data, splits it, and saves pickles.
    """
    print(f"Scanning '{h5_dir}' for .h5 files...")
    p = Path(h5_dir)
    files = sorted(list(p.glob("*.h5")))
    
    if not files:
        print("No .h5 files found! Please run gen_h5_with_vrho.py first.")
        return

    print(f"Found {len(files)} files. Loading into memory...")
    
    valid_data = []
    for f in files:
        data = load_vxc_from_h5(f)
        if data is not None:
            valid_data.append(data)
            print(f"  Loaded: {data['Name']} (pts: {data['Grid'].shape[0]})")
    
    if not valid_data:
        print("No valid data loaded.")
        return

    print(f"\nTotal loaded systems: {len(valid_data)}")

    # Split Data
    if len(valid_data) < 2:
        print("Warning: Only 1 system found. Using it for both Train and Validation.")
        train_data = valid_data
        val_data = valid_data
    else:
        train_data, val_data = split_train_val(valid_data, test_size, random_state)

    print(f"Split: {len(train_data)} Train, {len(val_data)} Validation.")

    # Save Pickles
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "data_vxc_train.pickle")
    val_path = os.path.join(output_dir, "data_vxc_val.pickle")

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    
    with open(val_path, "wb") as f:
        pickle.dump(val_data, f)

    print(f"\nSaved checkpoints:\n  {train_path}\n  {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", default="h5_vrho_from_mrks")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prepare_vxc(
        h5_dir=args.h5_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
