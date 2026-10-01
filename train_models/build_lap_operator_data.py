"""Build the real H2/BeH2/CO h-free central AO-operator pilot records.

The CLI is intentionally restricted to the three reviewed pilot systems.  It
preserves the legacy center order, weights, common RKS Vxc, and E_xc while
matching each legacy float32 coordinate to its original NPZ float64 center.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Existing builder helpers use historical top-level imports such as
# ``from lap_data import ...``; support them without editing those files.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .build_mrks_stencils import _make_mol, exact_vxc_match, metrics, normalize_name
    from .lap_operator_data import (
        FEATURE_LAYOUT,
        OPERATOR_PROTOCOL,
        PROTOCOL,
        SPATIAL_DERIVATIVE_METHOD,
        density_descriptors_from_ao,
        recover_legacy_centers,
        verify_operator_corpus,
        write_central_operator_record,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_mrks_stencils import _make_mol, exact_vxc_match, metrics, normalize_name
    from lap_operator_data import (
        FEATURE_LAYOUT,
        OPERATOR_PROTOCOL,
        PROTOCOL,
        SPATIAL_DERIVATIVE_METHOD,
        density_descriptors_from_ao,
        recover_legacy_centers,
        verify_operator_corpus,
        write_central_operator_record,
    )


PILOT_SYSTEMS = ("H2", "BeH2", "CO")
MAX_VXC_IDENTITY_ERROR = 1e-5


def _validate_external_output(output: Path) -> Path:
    output = output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError(f"Generated corpus must be outside the Git repository: {repository}")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_index(records: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    index = {}
    for record in records:
        name = str(record.get("Name", ""))
        key = normalize_name(name)
        if not key or key in index:
            raise ValueError(f"Empty or duplicate normalized legacy system name {name!r}.")
        index[key] = (name, record)
    return index


def _npz_index(npz_root: Path) -> dict[str, tuple[str, Path]]:
    index = {}
    for path in npz_root.glob("*/inp_mrks.npz"):
        name = path.parent.name
        key = normalize_name(name)
        if not key or key in index:
            raise ValueError(f"Empty or duplicate normalized NPZ system name {name!r}.")
        index[key] = (name, path)
    if len(index) != 90:
        raise ValueError(f"Expected the audited 90-system NPZ source set; found {len(index)}.")
    return index


def _make_molecule(npz):
    return _make_mol(npz)


def _legacy_grid(record: dict[str, Any]) -> np.ndarray:
    value = record["Grid"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    grid = np.asarray(value)
    if grid.ndim != 2 or grid.shape[1] < 13 or grid.dtype.kind != "f":
        raise ValueError("Legacy Grid must contain float coordinates and rho/sigma/lapl columns.")
    return grid


def _legacy_vector(value: Any, name: str, ngrid: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (ngrid,) or not np.isfinite(result).all():
        raise ValueError(f"Legacy {name} must be a finite vector of length {ngrid}.")
    return result


def _descriptor_audit(grid: np.ndarray, descriptors: np.ndarray) -> dict[str, Any]:
    gradients = descriptors[:, 2:8].reshape(-1, 2, 3)
    sigma = np.column_stack(
        (
            np.einsum("gi,gi->g", gradients[:, 0], gradients[:, 0]),
            np.einsum("gi,gi->g", gradients[:, 0], gradients[:, 1]),
            np.einsum("gi,gi->g", gradients[:, 1], gradients[:, 1]),
        )
    )
    rho_result = metrics(grid[:, 4:6], descriptors[:, :2])
    sigma_result = metrics(grid[:, 6:9], sigma)
    lapl_result = metrics(grid[:, 11:13], descriptors[:, 8:10])
    return {
        "rho": rho_result,
        "sigma_aa_ab_bb": sigma_result,
        "lapl": lapl_result,
    }


def _build_record(
    *,
    name: str,
    legacy_name: str,
    legacy: dict[str, Any],
    npz_path: Path,
    npz,
    legacy_path: Path,
    legacy_sha256: str,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pyscf import __version__ as pyscf_version
    from pyscf.dft import numint

    if legacy_name != name:
        raise ValueError(f"System name case mismatch: legacy {legacy_name!r}, NPZ {name!r}.")
    grid = _legacy_grid(legacy)
    ngrid = len(grid)
    weights = _legacy_vector(legacy["Weights"], "Weights", ngrid)
    vxc = _legacy_vector(legacy["Vrho"], "Vrho", ngrid)
    e_xc_value = legacy["E_xc"]
    if hasattr(e_xc_value, "detach"):
        e_xc_value = e_xc_value.detach().cpu().numpy()
    exc = np.asarray(e_xc_value, dtype=np.float64).reshape(())
    if not np.isfinite(exc):
        raise ValueError(f"{name}: legacy E_xc is non-finite.")

    npz_coords = np.asarray(npz["grid_coords"], dtype=np.float64)
    # Keep the shared builder's exact collision-safe implementation as the
    # canonical identity rule; no nearest-neighbor or tolerance search occurs.
    coords64, source_rows = recover_legacy_centers(grid[:, :3], npz_coords)
    if coords64.shape != (ngrid, 3):
        raise ValueError(f"{name}: exact recovered coordinate population is incomplete.")
    vxc_match = exact_vxc_match(grid, vxc, npz_coords, np.asarray(npz["vxc_grid"]).reshape(-1))
    if vxc_match["unmatched_legacy_points"] or vxc_match["max_abs"] > MAX_VXC_IDENTITY_ERROR:
        raise ValueError(f"{name}: legacy Vxc did not pass exact NPZ target identity audit: {vxc_match}")

    mol, basis, charge, spin = _make_molecule(npz)
    dmks = np.asarray(npz["dm_ks"], dtype=np.float64)
    if dmks.shape != (mol.nao_nr(), mol.nao_nr()):
        raise ValueError(f"{name}: dm_ks AO dimension does not match PySCF molecule.")
    if not np.allclose(dmks, dmks.T, atol=1e-12, rtol=0):
        raise ValueError(f"{name}: dm_ks is not symmetric.")
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    nelec = float(np.einsum("ij,ji->", dmks, overlap))
    npz_metadata = dict(zip(np.asarray(npz["meta_keys"]).tolist(), np.asarray(npz["meta"]).tolist()))
    if abs(nelec - float(npz_metadata["nelec"])) > 1e-6:
        raise ValueError(f"{name}: dm_ks electron count disagrees with source metadata.")
    if spin != 0:
        raise ValueError(f"{name}: pilot is intended for closed-shell RKS systems only.")

    descriptors = np.empty((ngrid, len(FEATURE_LAYOUT)), dtype=np.float64)
    ref_ao = np.zeros((mol.nao_nr(), mol.nao_nr()), dtype=np.float64)
    for start in range(0, ngrid, chunk_size):
        stop = min(start + chunk_size, ngrid)
        chunk_coords = coords64[start:stop]
        ao = np.asarray(numint.eval_ao(mol, chunk_coords, deriv=2))
        descriptors[start:stop] = density_descriptors_from_ao(mol, ao, dmks)
        phi = np.asarray(ao[0], dtype=np.float64)
        weighted_vxc = weights[start:stop] * vxc[start:stop]
        ref_ao += phi.T @ (phi * weighted_vxc[:, None])
    ref_ao = 0.5 * (ref_ao + ref_ao.T)
    descriptor_audit = _descriptor_audit(grid, descriptors)
    if not all(
        descriptor_audit[key]["float32_level_compatible"]
        for key in ("rho", "sigma_aa_ab_bb", "lapl")
    ):
        raise ValueError(f"{name}: central descriptors exceed legacy float32 tolerance: {descriptor_audit}")

    charges = np.asarray(npz["atom_charges"], dtype=np.int64)
    atom_coords = np.asarray(npz["atom_coords"], dtype=np.float64)
    molecule = {
        "basis": basis,
        "atom_charges": charges.tolist(),
        "atom_coords_bohr": atom_coords.tolist(),
        "charge": int(charge),
        "spin": int(spin),
        "ao_labels": list(mol.ao_labels()),
    }
    metadata = {
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "spatial_derivative_method": SPATIAL_DERIVATIVE_METHOD,
        "system_name": name,
        "point_count": int(ngrid),
        "nao": int(mol.nao_nr()),
        "feature_layout": list(FEATURE_LAYOUT),
        "spin_convention": "closed-shell RKS dm_ks is total density; density descriptors use dm_ks/2 in each spin channel",
        "coordinate_identity_rule": "legacy float32 coordinate exact identity lookup against NPZ grid_coords cast to float32; original NPZ float64 coordinates restored; ambiguous float32 collisions fail closed",
        "target_preservation": {
            "VxcLegacy": "exact common scalar RKS Vrho from legacy training pickle; no gauge shift",
            "Exc": "exact legacy E_xc training value promoted to float64; original formula unresolved; NPZ exc_wf not substituted",
            "weights": "exact legacy central quadrature weights promoted to float64",
        },
        "source": {
            "legacy_pickle_sha256": legacy_sha256,
            "npz_file_sha256": _sha256(npz_path),
            "npz_relative_id": f"{name}/inp_mrks.npz",
            "legacy_record_name": legacy_name,
            "legacy_point_count": int(ngrid),
            "npz_grid_point_count": len(npz_coords),
            "vxc_matching": vxc_match,
        },
        "molecule": molecule,
        "verification": {
            "electron_count_from_dmks_overlap": nelec,
            "central_descriptor_compatibility": descriptor_audit,
        },
        "pyscf_version": str(pyscf_version),
    }
    arrays = {
        "coords64": coords64,
        "legacycoords32": np.asarray(grid[:, :3], dtype=np.float32),
        "sourcerow": np.arange(ngrid, dtype=np.int64),
        "npzrow": source_rows,
        "weights": weights,
        "VxcLegacy": vxc,
        "Exc": exc,
        "DensityDescriptorsN10": descriptors,
        "dmks": dmks,
        "RefAO": ref_ao,
        "Overlap": overlap,
    }
    return arrays, metadata


def build(args: argparse.Namespace) -> dict[str, Any]:
    legacy_path = Path(args.legacy_pickle).resolve()
    npz_root = Path(args.npz_root).resolve()
    output = _validate_external_output(Path(args.output_dir))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite central operator output: {output}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    requested_names = (
        [name.strip() for name in args.systems.split(",") if name.strip()]
        if args.systems
        else None
    )
    if args.all90 and requested_names:
        raise ValueError("Use either --all90 or --systems, not both.")
    if requested_names is not None:
        requested_keys = [normalize_name(name) for name in requested_names]
        if not requested_keys or len(set(requested_keys)) != len(requested_keys):
            raise ValueError("--systems must be a non-empty comma-separated unique subset.")
        allowed = {normalize_name(name) for name in PILOT_SYSTEMS}
        if not set(requested_keys) <= allowed:
            raise ValueError(
                f"Pilot mode only permits {PILOT_SYSTEMS}; use explicit --all90 after the pilot gate."
            )
    else:
        requested_keys = None

    with legacy_path.open("rb") as handle:
        legacy_records = pickle.load(handle)
    if not isinstance(legacy_records, (list, tuple)) or len(legacy_records) != 90:
        raise ValueError("Legacy mRKS training pickle must contain the audited 90 records.")
    legacy_index = _legacy_index(legacy_records)
    npz_index = _npz_index(npz_root)
    if set(legacy_index) != set(npz_index):
        raise ValueError("Legacy and NPZ source system identities do not match the audited 90-system set.")
    if args.all90:
        selected_keys = sorted(npz_index, key=lambda key: npz_index[key][0].casefold())
    else:
        selected_keys = requested_keys or [normalize_name(name) for name in PILOT_SYSTEMS]
    if not set(selected_keys) <= set(npz_index):
        raise ValueError(f"Unknown requested system names: {sorted(set(selected_keys) - set(npz_index))}")
    selected_names = [npz_index[key][0] for key in selected_keys]

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    legacy_sha256 = _sha256(legacy_path)
    manifest_records = []
    started = time.perf_counter()
    try:
        for key in selected_keys:
            name, npz_path = npz_index[key]
            legacy_name, legacy = legacy_index[key]
            with np.load(npz_path, allow_pickle=False) as npz:
                arrays, metadata = _build_record(
                    name=name,
                    legacy_name=legacy_name,
                    legacy=legacy,
                    npz_path=npz_path,
                    npz=npz,
                    legacy_path=legacy_path,
                    legacy_sha256=legacy_sha256,
                    chunk_size=args.chunk_size,
                )
            filename = f"{name}.h5"
            digests = write_central_operator_record(temp_dir / filename, arrays, metadata)
            manifest_records.append(
                {
                    "system_name": name,
                    "file": filename,
                    **digests,
                    "point_count": len(arrays["coords64"]),
                    "nao": int(arrays["dmks"].shape[0]),
                }
            )
            print(
                f"Built {name}: {len(arrays['coords64'])} centers, "
                f"{arrays['dmks'].shape[0]} AOs, {Path(temp_dir / filename).stat().st_size / 1e6:.1f} MB"
            )

        manifest = {
            "protocol": PROTOCOL,
            "operator_protocol": OPERATOR_PROTOCOL,
            "spatial_derivative_method": SPATIAL_DERIVATIVE_METHOD,
            "built_systems": [item["system_name"] for item in manifest_records],
            "records": manifest_records,
            "builder": "build_lap_operator_data.py",
            "build_runtime_seconds": time.perf_counter() - started,
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        verify_operator_corpus(
            temp_dir,
            expected_systems=selected_names,
            require_all90=args.all90,
        )
        os.replace(temp_dir, output)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    result = verify_operator_corpus(
        output,
        expected_systems=selected_names,
        require_all90=args.all90,
    )
    result["output_dir"] = str(output)
    result["build_runtime_seconds"] = time.perf_counter() - started
    print(f"Verified pilot corpus at {output}; runtime {result['build_runtime_seconds']:.2f} s")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-pickle", required=True, help="Audited 90-system data_vxc_train.pickle")
    parser.add_argument("--npz-root", required=True, help="Audited 90-system mrks_90_ccsd_pt directory")
    parser.add_argument("--output-dir", required=True, help="New output directory outside the git repository")
    parser.add_argument("--systems", help="Pilot subset of H2,BeH2,CO; defaults to all three")
    parser.add_argument(
        "--all90",
        action="store_true",
        help="Explicitly request the exact 90-system corpus; run only after pilot approval",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
