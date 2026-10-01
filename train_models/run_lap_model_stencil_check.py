"""Check Lap-NN Euler components across real h/order choices (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_mrks_stencils import (
    _make_mol,
    exact_coordinate_lookup,
    normalize_name,
    sha256,
)
from lap_checkpoint import load_lap_checkpoint
from lap_data import ao_reference_evaluator, evaluate_stencil
from lap_stencil_operators import (
    STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
)
from lap_vxc import LapEnergy, euler_components
from run_lap_stencil_validation import _stratified_indices


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(weights, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    q = np.broadcast_to(q, v.shape)
    return float(np.sqrt((q * v * v).sum() / q.sum()))


def _pairwise(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_rms(np.asarray(left, dtype=np.float64) - right, weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-pickle", required=True, type=Path)
    parser.add_argument("--npz-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--systems", default="H2,BeH2,CO,N2,ClHS")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument(
        "--h-values",
        default="0.08,0.04,0.02,0.01,0.005,0.0025,0.00125,0.000625,0.0003125",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    legacy_path = args.legacy_pickle.resolve()
    npz_root = args.npz_root.resolve()
    output_path = args.output_json.resolve()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    hs = tuple(float(x) for x in args.h_values.split(","))
    if not hs or any(not np.isfinite(h) or h <= 0 for h in hs):
        raise ValueError("h values must be positive and finite")

    model, payload = load_lap_checkpoint(checkpoint, device="cpu", dtype=torch.float32)
    model.eval()
    energy = LapEnergy(model).eval()
    with legacy_path.open("rb") as stream:
        legacy_records = pickle.load(stream)
    legacy_by_name = {normalize_name(r["Name"]): r for r in legacy_records}
    npz_by_name = {
        normalize_name(p.parent.name): p
        for p in sorted(npz_root.glob("*/inp_mrks.npz"))
    }
    if len(legacy_by_name) != 90 or len(npz_by_name) != 90:
        raise ValueError("Expected exactly 90 systems in both mRKS sources")

    output = {
        "schema": "lap-nn-real-stencil-components-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_stencil_metadata": {
            "version": payload["stencil_version"],
            "derivative_order": payload["derivative_order"],
            "training_h_bohr": (
                payload.get("lap_s5_provenance", {}).get("stencil", {}).get("h_bohr")
            ),
        },
        "legacy_sha256": sha256(legacy_path),
        "dtype": "float32 NN local derivatives; float64 outer stencil accumulation",
        "h_values_bohr": list(hs),
        "systems": {},
        "training_or_targets_used": False,
        "status": "running",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))

    for name in (n.strip() for n in args.systems.split(",") if n.strip()):
        key = normalize_name(name)
        if key not in legacy_by_name or key not in npz_by_name:
            raise ValueError(f"Unknown mRKS system {name!r}")
        record = legacy_by_name[key]
        npz_path = npz_by_name[key]
        print(f"Analyzing {record['Name']}", flush=True)
        with np.load(npz_path, allow_pickle=False) as npz:
            mol, _basis, _charge, _spin = _make_mol(npz)
            dm = np.asarray(npz["dm_ks"], dtype=np.float64)
            spin_dm = np.stack((dm / 2, dm / 2))
            source_coords = np.asarray(npz["grid_coords"], dtype=np.float64)
            ids = exact_coordinate_lookup(record["Grid"][:, :3], source_coords)
            all_centers = source_coords[ids]
            atom_coords = np.asarray(npz["atom_coords"], dtype=np.float64)
            distance = np.linalg.norm(
                all_centers[:, None, :] - atom_coords[None, :, :], axis=-1
            ).min(axis=1)
            grid = np.asarray(record["Grid"], dtype=np.float64)
            indices, inclusion, sampling = _stratified_indices(
                distance,
                grid[:, 4:6].sum(-1),
                args.max_points,
                20261001 + len(output["systems"]),
            )
            centers = all_centers[indices]
            sample_weights = (
                np.asarray(record["Weights"], dtype=np.float64).reshape(-1)[indices]
                * inclusion
            )
            density = grid[indices, 4:6]
            q = sample_weights[:, None] * density
            evaluator = ao_reference_evaluator(mol, spin_dm)
            scheme_results = []
            all_predictions: dict[str, dict[float, dict[str, np.ndarray]]] = {}
            for order, version in (
                (2, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1),
                (4, STENCIL_VERSION_13_POINT_FOURTH_ORDER),
            ):
                local = {}
                for h in hs:
                    started = time.perf_counter()
                    _stencil_coords, features = evaluate_stencil(
                        torch.as_tensor(centers, dtype=torch.float64),
                        h,
                        evaluator,
                        chunk_size=256,
                        order=order,
                        version=version,
                    )
                    with torch.no_grad():
                        components = euler_components(
                            energy,
                            features.to(dtype=torch.float32),
                            h,
                            create_graph=False,
                            order=order,
                            version=version,
                        )
                    arrays = {
                        k: components[k].detach().cpu().numpy()
                        for k in ("C", "minus_div_A", "lap_B", "Vxc")
                    }
                    local[float(h)] = arrays
                    scheme_results.append(
                        {
                            "system": record["Name"],
                            "scheme": version,
                            "derivative_order": order,
                            "h_bohr": float(h),
                            "weighted_rms_magnitude": {
                                term: _weighted_rms(arrays[term], q)
                                for term in ("C", "minus_div_A", "lap_B", "Vxc")
                            },
                            "elapsed_seconds": float(time.perf_counter() - started),
                        }
                    )
                all_predictions[version] = local
                descending = sorted(map(float, hs), reverse=True)
                for idx, h in enumerate(descending[:-1]):
                    smaller = descending[idx + 1]
                    current = local[h]
                    next_value = local[smaller]
                    for row in scheme_results:
                        if row["scheme"] == version and row["h_bohr"] == h:
                            row["pairwise_to_next_smaller_h"] = {
                                "next_smaller_h_bohr": smaller,
                                "weighted_rms_difference": {
                                    term: _pairwise(current[term], next_value[term], q)
                                    for term in ("C", "minus_div_A", "lap_B", "Vxc")
                                },
                            }
                for h in descending:
                    by_h = [
                        r
                        for r in scheme_results
                        if r["scheme"] == version and r["h_bohr"] == h
                    ]
                    if by_h:
                        by_h[0]["pairwise_to_next_smaller_h"] = by_h[0].get(
                            "pairwise_to_next_smaller_h"
                        )
            # Same h, same energy model: compare the discrete operators directly.
            scheme7 = all_predictions[STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1]
            scheme13 = all_predictions[STENCIL_VERSION_13_POINT_FOURTH_ORDER]
            order_differences = [
                {
                    "h_bohr": float(h),
                    "weighted_rms_7point_vs_13point": {
                        term: _pairwise(
                            scheme7[float(h)][term], scheme13[float(h)][term], q
                        )
                        for term in ("C", "minus_div_A", "lap_B", "Vxc")
                    },
                }
                for h in hs
            ]
            output["systems"][record["Name"]] = {
                "npz_path": str(npz_path),
                "npz_sha256": sha256(npz_path),
                "sampling": sampling,
                "sampled_points": len(indices),
                "center_coordinates": "recovered float64 via exact legacy float32 coordinate identity",
                "stencil_results": scheme_results,
                "same_h_order_comparison": order_differences,
            }
        output_path.write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
        )
    output["status"] = "complete"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
