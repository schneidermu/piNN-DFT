"""Build real mRKS Cartesian density stencils from dm_ks and legacy targets.

The external NPZ files provide only the reference molecule/basis/density matrix.
The historical pickle remains the sole source of central points, quadrature
weights, full Vxc, and E_xc.  No interpolation or target reconstruction occurs.

Example (pilot; h is intentionally mandatory)::

    python build_mrks_stencils.py --legacy-pickle data_vxc_train.pickle \
      --npz-root mrks_90_ccsd_pt --output-dir mrks_pilot_h005 \
      --pilot-systems H2,BeH2,CO --h-bohr 0.005 --verify-full-centers
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lap_data import (
    PROTOCOL,
    ao_reference_evaluator,
    evaluate_stencil,
    write_stencil_h5,
)
from lap_vxc import sigma_from_gradients


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    """Case/spacing-insensitive key; punctuation in isomer suffixes is retained."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _unique_index(items, label):
    index = {}
    for name, value in items:
        key = normalize_name(name)
        if not key or key in index:
            raise ValueError(f"Duplicate/empty normalized {label} Name: {name!r}")
        index[key] = (name, value)
    return index


def load_csv_index(path: Path | None):
    if path is None:
        return {}, None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "System" not in reader.fieldnames:
            raise ValueError("extrapolation CSV must contain a System column.")
        rows = [(row["System"], row) for row in reader]
    return _unique_index(rows, "CSV"), sha256(path)


def coordinate_key(row: np.ndarray) -> tuple[int, int, int]:
    # The legacy Grid stores float32 coordinates.  Equality is exact in that
    # representation; this is coordinate identity, not a spatial search.
    value = np.asarray(row, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Invalid coordinate used for exact target identity lookup.")
    value[value == 0] = 0.0  # IEEE signed zeros denote the same exact coordinate.
    return tuple(int(x) for x in value.view(np.uint32))


def exact_vxc_match(legacy_grid, legacy_vxc, npz_coords, npz_vxc):
    source = {}
    for i, coord in enumerate(npz_coords):
        source.setdefault(coordinate_key(coord), []).append(float(npz_vxc[i]))
    differences = []
    multiplicities = []
    unmatched = 0
    for coord, target in zip(legacy_grid[:, :3], legacy_vxc):
        values = source.get(coordinate_key(coord))
        if not values:
            unmatched += 1
            continue
        multiplicities.append(len(values))
        differences.extend(abs(value - float(target)) for value in values)
    if not differences:
        raise ValueError("No exact float32-coordinate matches between legacy and NPZ.")
    diff = np.asarray(differences, dtype=np.float64)
    matched = len(multiplicities)
    return {
        "matched_legacy_points": matched,
        "legacy_points": len(legacy_grid),
        "unmatched_legacy_points": unmatched,
        "candidate_source_values_compared": len(diff),
        "max_coordinate_multiplicity": max(multiplicities),
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "median_abs": float(np.median(diff)),
        "match_rule": "exact coordinate identity after casting source coordinates to legacy float32; no interpolation",
    }


def exact_coordinate_lookup(legacy_coords, npz_coords):
    """Recover original float64 centers using exact stored-float32 identity.

    The legacy pickle stored coordinates as float32 after computing descriptors
    at the original float64 grid points.  Matching after casting the NPZ source
    coordinates to float32 is exact identity lookup, not a spatial search.  A
    collision between distinct float64 points fails closed.
    """
    source = {}
    for i, coord in enumerate(npz_coords):
        source.setdefault(coordinate_key(coord), []).append(i)
    indices = []
    for coord in legacy_coords:
        candidates = source.get(coordinate_key(coord), [])
        if not candidates:
            raise ValueError("A legacy central coordinate has no exact NPZ identity.")
        source_points = np.asarray(npz_coords[candidates], dtype=np.float64)
        if not np.all(source_points == source_points[0]):
            raise ValueError(
                "A legacy float32 coordinate maps to distinct NPZ float64 points; recovery is ambiguous."
            )
        indices.append(candidates[0])
    return np.asarray(indices, dtype=np.int64)


def preserved_targets(legacy):
    """Return historical E_xc and common-channel Vxc after float64 promotion."""
    vxc = torch.as_tensor(legacy["Vrho"], dtype=torch.float64).reshape(-1)
    exc = torch.as_tensor(legacy["E_xc"], dtype=torch.float64).reshape(())
    return vxc[:, None].repeat(1, 2), exc


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    if ref.shape != act.shape:
        raise ValueError(f"Center comparison shape mismatch: {ref.shape} vs {act.shape}")
    diff = np.abs(act - ref)
    relative = diff / np.maximum(np.abs(ref), 1e-12)
    atol, rtol = 5e-7, 5e-6
    compatible = diff <= atol + rtol * np.abs(ref)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "max_relative_floor_1e-12": float(relative.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "float32_level_max_normalized_error": float(
            np.max(diff / (atol + rtol * np.abs(ref)), initial=0.0)
        ),
        "float32_level_compatible": bool(np.all(compatible)),
    }


def _make_mol(npz_data):
    from pyscf import gto
    from pyscf.data import elements

    basis = json.loads(str(npz_data["basis"].reshape(-1)[0]))
    charges = np.asarray(npz_data["atom_charges"], dtype=np.int64)
    coords = np.asarray(npz_data["atom_coords"], dtype=np.float64)
    if coords.shape != (len(charges), 3):
        raise ValueError("NPZ atom coordinates/charges have inconsistent shapes.")
    atoms = [(elements.ELEMENTS[int(z)], tuple(xyz)) for z, xyz in zip(charges, coords)]
    charge = int(np.asarray(npz_data["mol_charge"]).reshape(-1)[0])
    spin = int(np.asarray(npz_data["mol_spin"]).reshape(-1)[0])
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    return mol, basis, charge, spin


def center_reconstruction(mol, dm, grid, coords=None, sample_indices=None):
    # dm_ks is the closed-shell total matrix; the spin density matrices are
    # exactly dm_ks/2.  dm1_ao_wf is deliberately never accessed here.
    evaluator = ao_reference_evaluator(mol, np.stack((dm / 2, dm / 2)))
    coords = (
        np.asarray(grid[:, :3], dtype=np.float64)
        if coords is None
        else np.asarray(coords, dtype=np.float64)
    )
    if coords.shape != (len(grid), 3):
        raise ValueError("Recovered source centers do not align with legacy grid rows.")
    if sample_indices is not None:
        coords = coords[sample_indices]
        grid = grid[sample_indices]
    evaluated = evaluator(coords)
    grad = evaluated[:, 2:8].reshape(-1, 2, 3)
    sigma = np.asarray(sigma_from_gradients(torch.as_tensor(grad)).numpy())
    rho_errors = {
        "alpha": metrics(grid[:, 4], evaluated[:, 0]),
        "beta": metrics(grid[:, 5], evaluated[:, 1]),
        "combined": metrics(grid[:, 4:6], evaluated[:, :2]),
    }
    sigma_errors = {
        "aa": metrics(grid[:, 6], sigma[:, 0]),
        "ab": metrics(grid[:, 7], sigma[:, 1]),
        "bb": metrics(grid[:, 8], sigma[:, 2]),
        "combined": metrics(grid[:, 6:9], sigma),
    }
    lapl_errors = {
        "alpha": metrics(grid[:, 11], evaluated[:, 8]),
        "beta": metrics(grid[:, 12], evaluated[:, 9]),
        "combined": metrics(grid[:, 11:13], evaluated[:, 8:10]),
    }
    return {
        "rho": rho_errors,
        "sigma_aa_ab_bb": sigma_errors,
        "lapl": lapl_errors,
        "points_checked": len(grid),
        "density_max_abs": float(np.max(np.abs(grid[:, 4:6] - evaluated[:, :2]))),
    }, evaluated


def deterministic_sample(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, count, dtype=np.int64))


def source_record(legacy, npz_path, npz, legacy_path, legacy_hash, csv_row, csv_hash):
    grid = legacy["Grid"].detach().cpu().numpy()
    vxc = legacy["Vrho"].detach().cpu().numpy()
    weights = legacy["Weights"].detach().cpu().numpy()
    e_xc = legacy["E_xc"].detach().cpu().numpy()
    if grid.dtype != np.float32 or grid.ndim != 2 or grid.shape[1] != 13:
        raise ValueError(f"{legacy['Name']}: expected legacy float32 Grid (N,13).")
    if vxc.shape != (len(grid),) or weights.shape != (len(grid),) or e_xc.size != 1:
        raise ValueError(f"{legacy['Name']}: invalid legacy target shapes.")
    expected_n = np.asarray(npz["dm_ks"]).shape[0]
    if np.asarray(npz["dm_ks"]).shape != (expected_n, expected_n):
        raise ValueError(f"{legacy['Name']}: dm_ks must be square.")
    npz_coords = np.asarray(npz["grid_coords"])
    npz_vxc = np.asarray(npz["vxc_grid"])
    if npz_coords.ndim != 2 or npz_coords.shape[1] != 3 or npz_vxc.shape != (len(npz_coords),):
        raise ValueError(f"{legacy['Name']}: invalid NPZ grid/vxc shapes.")
    if int(np.asarray(npz["mol_spin"]).reshape(-1)[0]) != 0:
        raise ValueError(f"{legacy['Name']}: current real pipeline requires RKS spin=0.")
    if legacy["Name"] != npz_path.parent.name:
        raise ValueError(f"System Name disagrees with NPZ folder: {legacy['Name']} / {npz_path}")
    match = exact_vxc_match(grid, vxc, npz_coords, npz_vxc)
    if match["unmatched_legacy_points"]:
        raise ValueError(f"{legacy['Name']}: legacy targets have unmatched source coordinates.")
    if match["max_abs"] > 1e-5:
        raise ValueError(f"{legacy['Name']}: legacy Vxc does not match NPZ full vxc_grid.")
    return {
        "name": legacy["Name"],
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256(npz_path),
        "legacy_source_path": str(legacy_path.resolve()),
        "legacy_source_sha256": legacy_hash,
        "csv_system": csv_row.get("System") if csv_row else None,
        "csv_source_sha256": csv_hash,
        "number_legacy_training_points": len(grid),
        "number_npz_source_grid_points": len(npz_coords),
        "basis": str(np.asarray(npz["basis"]).reshape(-1)[0]),
        "charge": int(np.asarray(npz["mol_charge"]).reshape(-1)[0]),
        "spin": int(np.asarray(npz["mol_spin"]).reshape(-1)[0]),
        "ao_dimension": int(expected_n),
        "legacy_vxc_layout": "one common RKS channel (N,); new record stores repeated (N,2)",
        "electron_count_from_dm_ks": None,
        "legacy_integrated_electron_count": float(np.sum((grid[:, 4] + grid[:, 5]) * weights)),
        "central_density_reconstruction_errors": None,
        "central_sigma_reconstruction_errors": None,
        "central_laplacian_reconstruction_errors": None,
        "vxc_matching_statistics": match,
        "E_xc_legacy": float(e_xc.reshape(-1)[0]),
        "exc_wf_audit_only": float(dict(zip(np.asarray(npz["meta_keys"]).tolist(), np.asarray(npz["meta"]).tolist())).get("exc_wf", float("nan"))),
        "E_xc_legacy_minus_exc_wf": float(e_xc.reshape(-1)[0] - dict(zip(np.asarray(npz["meta_keys"]).tolist(), np.asarray(npz["meta"]).tolist())).get("exc_wf", float("nan"))),
        "E_xc_source": "exact legacy mRKS training pickle value",
        "E_xc_original_formula": "unresolved; NPZ exc_wf not substituted",
        "csv_E_CBS_1_audit_only": float(csv_row["E_CBS_1"]) if csv_row and csv_row.get("E_CBS_1") else None,
    }


def build(args):
    legacy_path = Path(args.legacy_pickle).resolve()
    npz_root = Path(args.npz_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stencil output directory: {output}")
    npz_files = sorted(npz_root.glob("*/inp_mrks.npz"))
    if len(npz_files) != 90:
        raise ValueError(f"Expected exactly 90 NPZ source systems; found {len(npz_files)}.")
    npz_index = _unique_index(((p.parent.name, p) for p in npz_files), "NPZ")
    csv_index, csv_digest = load_csv_index(Path(args.csv).resolve() if args.csv else None)
    with legacy_path.open("rb") as handle:
        legacy_records = pickle.load(handle)
    if not isinstance(legacy_records, (list, tuple)) or len(legacy_records) != 90:
        raise ValueError("Legacy mRKS source must be a sequence of exactly 90 records.")
    legacy_index = _unique_index(((item.get("Name", ""), item) for item in legacy_records), "legacy")
    if set(npz_index) != set(legacy_index):
        raise ValueError("Normalized NPZ and legacy Name indexes do not match 90/90.")
    if csv_index and set(npz_index) != set(csv_index):
        missing = sorted(set(npz_index) - set(csv_index))
        extra = sorted(set(csv_index) - set(npz_index))
        raise ValueError(f"CSV Name index does not match 90/90: missing={missing}, extra={extra}")
    selected_names = (
        {normalize_name(name) for name in args.pilot_systems.split(",") if name.strip()}
        if args.pilot_systems
        else (set() if args.audit_only else set(npz_index))
    )
    if (not selected_names and not args.audit_only) or not selected_names <= set(npz_index):
        raise ValueError(f"Unknown/empty selected system set: {sorted(selected_names - set(npz_index))}")
    if not math.isfinite(args.h_bohr) or args.h_bohr <= 0:
        raise ValueError("An explicit positive finite --h-bohr is required.")
    if args.chunk_size <= 0 or args.center_sample_count <= 0:
        raise ValueError("Chunk and center sample sizes must be positive.")

    csv_by_key = {key: value[1] for key, value in csv_index.items()}
    legacy_digest = sha256(legacy_path)
    audit = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "normalized_name_rule": "casefold then remove non-ASCII-alphanumeric characters",
        "input_files": {
            "legacy_pickle": {"path": str(legacy_path), "sha256": sha256(legacy_path)},
            "npz_root": str(npz_root),
            "csv": {"path": str(Path(args.csv).resolve()), "sha256": csv_digest} if args.csv else None,
            "mrks_zip": (
                {"path": str(Path(args.source_zip).resolve()), "sha256": sha256(Path(args.source_zip).resolve())}
                if args.source_zip
                else None
            ),
        },
        "matched_systems": 90,
        "built_systems": [],
        "records": [],
        "pilot_h_bohr": args.h_bohr,
        "pilot_mode": bool(args.pilot_systems),
        "audit_only": args.audit_only,
        "center_verification_mode": (
            "full"
            if args.verify_full_centers or args.audit_full_centers
            else f"deterministic sample of at most {args.center_sample_count}"
        ),
    }
    output.mkdir(parents=True)
    for key in sorted(npz_index, key=lambda k: npz_index[k][0].casefold()):
        name, npz_path = npz_index[key]
        legacy_name, legacy = legacy_index[key]
        if legacy_name != name:
            raise ValueError(f"Name case mismatch across sources: {legacy_name} vs {name}")
        with np.load(npz_path, allow_pickle=False) as npz:
            item = source_record(
                legacy,
                npz_path,
                npz,
                legacy_path,
                legacy_digest,
                csv_by_key.get(key),
                csv_digest,
            )
            should_build = key in selected_names
            full_centers = args.verify_full_centers or args.audit_full_centers
            mol, _, _, spin = _make_mol(npz)
            dm = np.asarray(npz["dm_ks"], dtype=np.float64)
            overlap = mol.intor_symmetric("int1e_ovlp")
            item["electron_count_from_dm_ks"] = float(np.einsum("ij,ji->", dm, overlap))
            if mol.nao_nr() != dm.shape[0]:
                raise ValueError(
                    f"{name}: reconstructed AO dimension {mol.nao_nr()} differs from dm_ks {dm.shape[0]}."
                )
            if not np.allclose(dm, dm.T, atol=1e-12, rtol=0):
                raise ValueError(f"{name}: dm_ks is not symmetric.")
            meta = dict(
                zip(
                    np.asarray(npz["meta_keys"]).tolist(),
                    np.asarray(npz["meta"]).tolist(),
                )
            )
            if abs(item["electron_count_from_dm_ks"] - meta["nelec"]) > 1e-6:
                raise ValueError(
                    f"{name}: dm_ks electron count disagrees with NPZ metadata."
                )
            if should_build or args.audit_full_centers:
                grid = legacy["Grid"].detach().cpu().numpy()
                indices = None
                if not full_centers:
                    indices = deterministic_sample(len(grid), args.center_sample_count)
                npz_coords = np.asarray(npz["grid_coords"], dtype=np.float64)
                source_indices = exact_coordinate_lookup(grid[:, :3], npz_coords)
                center_coords = npz_coords[source_indices]
                center_stats, _ = center_reconstruction(
                    mol, dm, grid, center_coords, indices
                )
                item["central_density_reconstruction_errors"] = center_stats["rho"]
                item["central_sigma_reconstruction_errors"] = center_stats["sigma_aa_ab_bb"]
                item["central_laplacian_reconstruction_errors"] = center_stats["lapl"]
                if should_build:
                    widened_stats, _ = center_reconstruction(mol, dm, grid, None, indices)
                    item["legacy_float32_coordinate_widened_errors"] = {
                        "rho": widened_stats["rho"],
                        "sigma_aa_ab_bb": widened_stats["sigma_aa_ab_bb"],
                        "lapl": widened_stats["lapl"],
                    }
                item["central_coordinate_precision_recovery"] = (
                    "legacy row identity exact-matched to the original NPZ float64 coordinate; "
                    "no nearest-neighbor interpolation; ambiguity fails closed"
                )
                item["central_points_checked"] = center_stats["points_checked"]
                if full_centers:
                    # Old source tensors are float32.  Compare against a
                    # combined absolute/relative float32 storage tolerance.
                    for field in ("rho", "sigma_aa_ab_bb", "lapl"):
                        if not center_stats[field]["combined"]["float32_level_compatible"]:
                            raise ValueError(f"{name}: dm_ks fails full center {field} audit: {center_stats[field]}")
                if should_build:
                    coords = torch.as_tensor(center_coords, dtype=torch.float64)
                    evaluator = ao_reference_evaluator(mol, np.stack((dm / 2, dm / 2)))
                    stencil_start = time.perf_counter()
                    stencil_coords, stencil_features = evaluate_stencil(
                        coords, args.h_bohr, evaluator, chunk_size=args.chunk_size
                    )
                    item["stencil_generation_seconds"] = time.perf_counter() - stencil_start
                    item["stencil_evaluation_points"] = 7 * len(coords)
                    weights = torch.as_tensor(legacy["Weights"].detach().cpu().numpy(), dtype=torch.float64)
                    target, e_xc = preserved_targets(legacy)
                    provenance = {
                        "generator": "build_mrks_stencils.py / PySCF AO eval_ao deriv=2; exact Cartesian offsets",
                        "reference_density": "NPZ dm_ks / 2 in each spin channel; dm1_ao_wf not used",
                        "molecule_basis_ao_order": "NPZ atom_coords, atom_charges, mol_charge, mol_spin, basis JSON; PySCF reconstructed molecule; AO dimension checked against dm_ks",
                        "full_vxc_target": "exact legacy pickle Vrho by Name; verified against NPZ vxc_grid at float32 coordinate identity",
                        "legacy_target_source": str(legacy_path),
                        "legacy_target_sha256": legacy_digest,
                        "npz_source": str(npz_path),
                        "npz_sha256": sha256(npz_path),
                        "E_xc_source": "preserved legacy mRKS training target",
                        "E_xc_original_formula": "unresolved",
                        "NPZ_exc_wf_substituted": False,
                        "legacy_training_points": len(grid),
                        "central_dataset_identity": "legacy row indices select the exact historical central population, weights, Vxc, and E_xc; each original float64 coordinate is recovered by exact float32 identity lookup against NPZ grid_coords",
                        "central_coordinate_precision_recovery": "exact identity lookup; distinct float64 collisions fail; no nearest neighbor/interpolation",
                        "legacy_coordinates_float32_preserved": True,
                        "central_density_check": center_stats,
                        "legacy_vxc_matching": item["vxc_matching_statistics"],
                    }
                    record = {
                        "Name": name,
                        "Coordinates": coords,
                        "LegacyCoordinates": torch.as_tensor(
                            grid[:, :3].astype(np.float64), dtype=torch.float64
                        ),
                        "StencilCoordinates": stencil_coords,
                        "StencilFeatures": stencil_features,
                        "Weights": weights,
                        "Vxc": target,
                        "E_xc": e_xc,
                        "Protocol": PROTOCOL,
                        "StencilVersion": "cartesian-7-rho-grad-lapl-v1",
                        "HBohr": args.h_bohr,
                        "SourceSpin": spin,
                        "TargetKind": "common-rks",
                        "GaugeMetadata": {"available": False, "description": "preserved historical Vxc gauge; no shift or projection applied"},
                        "SourceProvenance": provenance,
                    }
                    h5_path = output / f"{name}.h5"
                    write_stencil_h5(h5_path, record)
                    item["generated_stencil_path"] = str(h5_path)
                    item["generated_stencil_sha256"] = sha256(h5_path)
                    item["generated_stencil_points"] = len(coords)
                    audit["built_systems"].append(name)
        audit["records"].append(item)
        if should_build:
            print(f"Built {name}: {item['number_legacy_training_points']} points; h={args.h_bohr:g} Bohr; center max errors rho/sigma/lapl = "
                  f"{item['central_density_reconstruction_errors']['combined']['max_abs']:.3g}/"
                  f"{item['central_sigma_reconstruction_errors']['combined']['max_abs']:.3g}/"
                  f"{item['central_laplacian_reconstruction_errors']['combined']['max_abs']:.3g}")
    if len(audit["records"]) != 90 or len(audit["built_systems"]) != len(selected_names):
        raise RuntimeError("Source/stencil audit did not complete its expected system count.")
    audit_path = output / "mrks_source_audit_v2.json"
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audit: {audit_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-pickle", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--source-zip", help="Optional MRKS.zip generator archive for input provenance.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-systems", help="Comma-separated system subset; omitting means generate all 90.")
    parser.add_argument("--h-bohr", type=float, required=True, help="Explicit finite-difference step in Bohr; there is no default.")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--center-sample-count", type=int, default=4096)
    parser.add_argument("--verify-full-centers", action="store_true", help="Check every central point before writing its stencil.")
    parser.add_argument("--audit-full-centers", action="store_true", help="Check all systems' central points without requiring full stencil generation.")
    parser.add_argument("--audit-only", action="store_true", help="Build no stencils; audit all 90 source records and centers only.")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
