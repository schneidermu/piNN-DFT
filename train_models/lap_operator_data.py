"""Immutable central-grid data for h-free variational XC-operator work.

The records preserve the audited legacy mRKS centers, weights, common scalar
potential, and energy.  AO values and derivatives are regenerated from the
stored molecule metadata and the exact NPZ ``dm_ks`` matrix when needed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:  # supports both ``python -m train_models...`` and direct script imports
    from .lap_operator import FEATURE_LAYOUT, OPERATOR_PROTOCOL
except ImportError:  # pragma: no cover - exercised by direct-script callers
    from lap_operator import FEATURE_LAYOUT, OPERATOR_PROTOCOL


PROTOCOL = "lap-operator-central-ao-noh-v1"
SPATIAL_DERIVATIVE_METHOD = "variational-ao-no-spatial-fd"
REQUIRED_ARRAYS = (
    "coords64",
    "legacycoords32",
    "sourcerow",
    "npzrow",
    "weights",
    "VxcLegacy",
    "Exc",
    "DensityDescriptorsN10",
    "dmks",
    "RefAO",
    "Overlap",
)
FORBIDDEN_METADATA_KEYS = frozenset({"h", "stencil", "h_bohr", "stencil_version"})
ALL90_EXPECTED_POINT_COUNT = 8_271_091


@dataclass(frozen=True)
class CentralOperatorRecord:
    """One system's central operator data; loaded arrays are read-only."""

    coords64: np.ndarray
    legacycoords32: np.ndarray
    sourcerow: np.ndarray
    npzrow: np.ndarray
    weights: np.ndarray
    VxcLegacy: np.ndarray
    Exc: np.ndarray
    DensityDescriptorsN10: np.ndarray
    dmks: np.ndarray
    RefAO: np.ndarray
    Overlap: np.ndarray
    metadata: dict[str, Any]
    path: Path | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _record_digest(metadata: dict[str, Any], dataset_hashes: dict[str, str]) -> str:
    clean = dict(metadata)
    clean.pop("record_sha256", None)
    clean.pop("dataset_sha256", None)
    return hashlib.sha256(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "operator_protocol": OPERATOR_PROTOCOL,
                "metadata": clean,
                "dataset_sha256": dataset_hashes,
            }
        )
    ).hexdigest()


def _recursive_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _recursive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _recursive_keys(nested)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("Central operator metadata must be a JSON object.")
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected central operator protocol {PROTOCOL!r}.")
    if metadata.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError(f"Expected weak-form operator protocol {OPERATOR_PROTOCOL!r}.")
    if metadata.get("spatial_derivative_method") != SPATIAL_DERIVATIVE_METHOD:
        raise ValueError("Central operator record must explicitly identify the h-free AO method.")
    bad = FORBIDDEN_METADATA_KEYS.intersection(_recursive_keys(metadata))
    if bad:
        raise ValueError(f"h/stencil metadata is forbidden in this protocol: {sorted(bad)}")
    if not isinstance(metadata.get("system_name"), str) or not metadata["system_name"]:
        raise ValueError("Central operator metadata needs a system_name.")


def _normalize_arrays(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = set(REQUIRED_ARRAYS) - set(arrays)
    extra = set(arrays) - set(REQUIRED_ARRAYS)
    if missing or extra:
        raise ValueError(f"Invalid central record fields; missing={sorted(missing)}, extra={sorted(extra)}")
    output = {key: np.asarray(arrays[key]) for key in REQUIRED_ARRAYS}
    float_fields = (
        "coords64", "weights", "VxcLegacy", "Exc", "DensityDescriptorsN10",
        "dmks", "RefAO", "Overlap",
    )
    for key in float_fields:
        output[key] = np.asarray(output[key], dtype="<f8")
    output["legacycoords32"] = np.asarray(output["legacycoords32"], dtype="<f4")
    output["sourcerow"] = np.asarray(output["sourcerow"], dtype="<i8")
    output["npzrow"] = np.asarray(output["npzrow"], dtype="<i8")

    ngrid = output["coords64"].shape[0] if output["coords64"].ndim == 2 else -1
    nao = output["dmks"].shape[0] if output["dmks"].ndim == 2 else -1
    expected_shapes = {
        "coords64": (ngrid, 3),
        "legacycoords32": (ngrid, 3),
        "sourcerow": (ngrid,),
        "npzrow": (ngrid,),
        "weights": (ngrid,),
        "VxcLegacy": (ngrid,),
        "Exc": (),
        "DensityDescriptorsN10": (ngrid, len(FEATURE_LAYOUT)),
        "dmks": (nao, nao),
        "RefAO": (nao, nao),
        "Overlap": (nao, nao),
    }
    if ngrid <= 0 or nao <= 0:
        raise ValueError("Central operator arrays must have non-empty grid and AO dimensions.")
    for key, shape in expected_shapes.items():
        if output[key].shape != shape:
            raise ValueError(f"{key} has shape {output[key].shape}; expected {shape}.")
        if output[key].dtype.kind in "fc" and not np.isfinite(output[key]).all():
            raise ValueError(f"{key} contains non-finite values.")
    if not np.array_equal(output["sourcerow"], np.arange(ngrid, dtype=np.int64)):
        raise ValueError("sourcerow must preserve the ordered legacy central-row indices.")
    if not np.allclose(output["dmks"], output["dmks"].T, atol=1e-12, rtol=0):
        raise ValueError("dmks must be a real symmetric RKS density matrix.")
    if not np.allclose(output["RefAO"], output["RefAO"].T, atol=1e-10, rtol=0):
        raise ValueError("RefAO must be a real symmetric AO matrix.")
    return output


def write_central_operator_record(
    path: str | Path, arrays: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, str]:
    """Write a new immutable HDF5 record and return logical/file digests."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite central operator record: {path}")
    values = _normalize_arrays(arrays)
    meta = dict(metadata)
    _validate_metadata(meta)
    if int(meta.get("point_count", -1)) != len(values["coords64"]):
        raise ValueError("point_count metadata disagrees with central arrays.")
    if int(meta.get("nao", -1)) != values["dmks"].shape[0]:
        raise ValueError("nao metadata disagrees with AO arrays.")
    hashes = {key: _array_digest(value) for key, value in values.items()}
    meta["dataset_sha256"] = hashes
    meta["record_sha256"] = _record_digest(meta, hashes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as handle:
        handle.attrs["protocol"] = PROTOCOL
        handle.attrs["operator_protocol"] = OPERATOR_PROTOCOL
        handle.attrs["metadata_json"] = _canonical_json(meta).decode("utf-8")
        for key, value in values.items():
            options = {} if value.shape == () else {
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
            }
            handle.create_dataset(key, data=value, **options)
        handle.flush()
    return {"record_sha256": meta["record_sha256"], "file_sha256": _file_digest(path)}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_central_operator_record(path: str | Path) -> CentralOperatorRecord:
    """Load and verify one HDF5 record; reject altered arrays or metadata."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("protocol") != PROTOCOL:
            raise ValueError(f"{path}: not a {PROTOCOL} record.")
        if handle.attrs.get("operator_protocol") != OPERATOR_PROTOCOL:
            raise ValueError(f"{path}: incompatible operator protocol.")
        metadata = json.loads(handle.attrs["metadata_json"])
        _validate_metadata(metadata)
        arrays = {key: np.asarray(handle[key][...]) for key in REQUIRED_ARRAYS if key in handle}
        if len(arrays) != len(REQUIRED_ARRAYS):
            raise ValueError(f"{path}: central operator record is missing required arrays.")
    arrays = _normalize_arrays(arrays)
    actual_hashes = {key: _array_digest(value) for key, value in arrays.items()}
    if metadata.get("dataset_sha256") != actual_hashes:
        raise ValueError(f"{path}: central record array hash mismatch.")
    if metadata.get("record_sha256") != _record_digest(metadata, actual_hashes):
        raise ValueError(f"{path}: central record logical hash mismatch.")
    if int(metadata.get("point_count", -1)) != len(arrays["coords64"]):
        raise ValueError(f"{path}: point_count metadata mismatch.")
    if int(metadata.get("nao", -1)) != arrays["dmks"].shape[0]:
        raise ValueError(f"{path}: nao metadata mismatch.")
    for array in arrays.values():
        array.setflags(write=False)
    return CentralOperatorRecord(**arrays, metadata=metadata, path=path)


def recover_legacy_centers(
    legacy_coords: np.ndarray, npz_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover float64 source centers by exact float32 identity, without search."""
    try:
        from .build_mrks_stencils import exact_coordinate_lookup
    except ImportError:  # pragma: no cover - direct-script import mode
        from build_mrks_stencils import exact_coordinate_lookup
    legacy_coords = np.asarray(legacy_coords, dtype=np.float32)
    npz_coords = np.asarray(npz_coords, dtype=np.float64)
    if legacy_coords.ndim != 2 or legacy_coords.shape[1] != 3:
        raise ValueError("legacy_coords must have shape (N,3).")
    indices = exact_coordinate_lookup(legacy_coords, npz_coords)
    return npz_coords[indices].copy(), indices


def build_molecule_from_metadata(metadata: dict[str, Any]):
    """Rebuild the exact PySCF molecule and AO ordering described by a record."""
    _validate_metadata(metadata)
    from pyscf import gto
    from pyscf.data import elements

    molecule = metadata["molecule"]
    charges = [int(value) for value in molecule["atom_charges"]]
    coords = np.asarray(molecule["atom_coords_bohr"], dtype=np.float64)
    if coords.shape != (len(charges), 3):
        raise ValueError("Stored molecule coordinates/charges do not align.")
    atoms = [(elements.ELEMENTS[z], tuple(xyz)) for z, xyz in zip(charges, coords)]
    mol = gto.M(
        atom=atoms,
        basis=molecule["basis"],
        unit="Bohr",
        charge=int(molecule["charge"]),
        spin=int(molecule["spin"]),
        verbose=0,
    )
    if mol.nao_nr() != int(metadata["nao"]):
        raise ValueError("Reconstructed molecule AO dimension differs from the record.")
    if list(mol.ao_labels()) != molecule["ao_labels"]:
        raise ValueError("Reconstructed molecule AO order differs from the record.")
    return mol


def iter_ao_factor_chunks(
    record: CentralOperatorRecord, mol, chunk_size: int = 4096
) -> Iterator[tuple[slice, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(slice, phi, grad_phi, lap_phi)`` in PySCF AO component order."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if mol.nao_nr() != record.dmks.shape[0]:
        raise ValueError("Molecule AO dimension does not match the record.")
    from pyscf.dft import numint

    for start in range(0, len(record.coords64), chunk_size):
        stop = min(start + chunk_size, len(record.coords64))
        ao = np.asarray(numint.eval_ao(mol, record.coords64[start:stop], deriv=2))
        if ao.shape[0] < 10 or ao.shape[1:] != (stop - start, mol.nao_nr()):
            raise ValueError(f"Unexpected PySCF deriv=2 AO shape: {ao.shape}.")
        phi = np.asarray(ao[0], dtype=np.float64)
        grad_phi = np.asarray(ao[1:4].transpose(1, 0, 2), dtype=np.float64)
        lap_phi = np.asarray(ao[4] + ao[7] + ao[9], dtype=np.float64)
        yield slice(start, stop), phi, grad_phi, lap_phi


def evaluate_density_descriptors(mol, dmks: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Evaluate the ten stored spin density descriptors using ``dmks / 2``."""
    from pyscf.dft import numint

    dmks = np.asarray(dmks, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if dmks.shape != (mol.nao_nr(), mol.nao_nr()):
        raise ValueError("dmks shape does not match the reconstructed molecule.")
    ao = np.asarray(numint.eval_ao(mol, coords, deriv=2))
    return density_descriptors_from_ao(mol, ao, dmks)


def density_descriptors_from_ao(mol, ao: np.ndarray, dmks: np.ndarray) -> np.ndarray:
    """Evaluate spin descriptors from an already computed PySCF deriv=2 AO block."""
    from pyscf.dft import numint

    dmks = np.asarray(dmks, dtype=np.float64)
    if dmks.shape != (mol.nao_nr(), mol.nao_nr()):
        raise ValueError("dmks shape does not match the reconstructed molecule.")
    ao = np.asarray(ao)
    if ao.ndim != 3 or ao.shape[0] < 10 or ao.shape[2] != mol.nao_nr():
        raise ValueError("AO block must be PySCF deriv=2 with matching AO dimension.")
    spin_dms = (dmks / 2.0, dmks / 2.0)
    rho = [
        numint.eval_rho(mol, ao, dm, xctype="MGGA", hermi=1, with_lapl=True)
        for dm in spin_dms
    ]
    return np.column_stack(
        (rho[0][0], rho[1][0], rho[0][1:4].T, rho[1][1:4].T, rho[0][4], rho[1][4])
    ).astype(np.float64, copy=False)


def project_reference_vxc_ao(
    phi: np.ndarray, weights: np.ndarray, vxc: np.ndarray
) -> np.ndarray:
    """Project the legacy common RKS scalar potential into the AO basis."""
    phi = np.asarray(phi, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    vxc = np.asarray(vxc, dtype=np.float64)
    if phi.ndim != 2 or weights.shape != (len(phi),) or vxc.shape != weights.shape:
        raise ValueError("Expected phi[N,nao] and weights/vxc[N].")
    if not (np.isfinite(phi).all() and np.isfinite(weights).all() and np.isfinite(vxc).all()):
        raise ValueError("AO projection inputs must be finite.")
    weighted = weights * vxc
    result = phi.T @ (phi * weighted[:, None])
    return np.asarray(0.5 * (result + result.T), dtype=np.float64)


def verify_operator_corpus(
    directory: str | Path,
    *,
    expected_systems: Iterable[str] | None = None,
    require_all90: bool = False,
) -> dict[str, Any]:
    """Verify manifest, record hashes, protocol separation, and system set."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError("Manifest protocol is incompatible with central operator records.")
    if FORBIDDEN_METADATA_KEYS.intersection(_recursive_keys(manifest)):
        raise ValueError("Manifest includes a forbidden h/stencil field.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must list one or more system records.")
    observed: dict[str, CentralOperatorRecord] = {}
    for item in records:
        name, filename = item.get("system_name"), item.get("file")
        if not name or not filename or name in observed:
            raise ValueError("Manifest has an empty or duplicate system record.")
        path = directory / filename
        if _file_digest(path) != item.get("file_sha256"):
            raise ValueError(f"{name}: HDF5 file hash mismatch.")
        record = load_central_operator_record(path)
        if record.metadata["system_name"] != name:
            raise ValueError(f"{name}: manifest/record system name mismatch.")
        if item.get("record_sha256") != record.metadata["record_sha256"]:
            raise ValueError(f"{name}: manifest/record logical hash mismatch.")
        observed[name] = record
    listed_systems = manifest.get("built_systems")
    if listed_systems != [item.get("system_name") for item in records]:
        raise ValueError("Manifest built_systems does not match its ordered record list.")
    systems = set(observed)
    expected = None if expected_systems is None else set(expected_systems)
    if expected is not None and systems != expected:
        raise ValueError(
            f"System set mismatch; missing={sorted(expected - systems)}, "
            f"extra={sorted(systems - expected)}."
        )
    if require_all90:
        if len(systems) != 90:
            raise ValueError(f"All-90 verification requires exactly 90 systems; found {len(systems)}.")
        if expected is None or len(expected) != 90:
            raise ValueError("All-90 verification requires the expected 90-system name set.")
        point_total = sum(len(record.coords64) for record in observed.values())
        if point_total != ALL90_EXPECTED_POINT_COUNT:
            raise ValueError(
                f"All-90 central point total {point_total} != {ALL90_EXPECTED_POINT_COUNT}."
            )
    return {
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "systems": sorted(systems),
        "point_count": sum(len(record.coords64) for record in observed.values()),
        "verified_records": len(observed),
        "all90": bool(require_all90),
    }


def write_corpus_manifest(
    directory: str | Path,
    destination: str | Path,
    *,
    expected_systems: Iterable[str] | None = None,
    require_all90: bool = True,
) -> dict[str, Any]:
    """Write hashes and record metadata without copying central-grid arrays."""
    directory = Path(directory).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable corpus manifest: {destination}")
    verified = verify_operator_corpus(
        directory,
        expected_systems=expected_systems,
        require_all90=require_all90,
    )
    external_path = directory / "manifest.json"
    external_bytes = external_path.read_bytes()
    external = json.loads(external_bytes)
    records = []
    total_bytes = 0
    max_vxc_error = 0.0
    max_descriptor_error = 0.0
    for item in external["records"]:
        record_path = directory / item["file"]
        record = load_central_operator_record(record_path)
        metadata = record.metadata
        file_bytes = record_path.stat().st_size
        total_bytes += file_bytes
        max_vxc_error = max(max_vxc_error, metadata["source"]["vxc_matching"]["max_abs"])
        descriptor_metrics = metadata["verification"]["central_descriptor_compatibility"]
        max_descriptor_error = max(
            max_descriptor_error,
            *(
                descriptor_metrics[field]["float32_level_max_normalized_error"]
                for field in ("rho", "sigma_aa_ab_bb", "lapl")
            ),
        )
        records.append(
            {
                "system_name": item["system_name"],
                "file": item["file"],
                "file_sha256": item["file_sha256"],
                "record_sha256": item["record_sha256"],
                "file_bytes": file_bytes,
                "point_count": item["point_count"],
                "nao": item["nao"],
                "Exc_legacy_ha": float(record.Exc),
                "metadata": metadata,
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "spatial_derivative_method": SPATIAL_DERIVATIVE_METHOD,
        "external_corpus_path_relative_to_manifest": os.path.relpath(
            directory, destination.parent
        ).replace("\\", "/"),
        "external_manifest": {
            "file": "manifest.json",
            "sha256": hashlib.sha256(external_bytes).hexdigest(),
            "bytes": len(external_bytes),
        },
        "corpus": {
            "system_count": verified["verified_records"],
            "central_point_count": verified["point_count"],
            "compressed_hdf5_bytes": total_bytes,
            "builder": external.get("builder"),
            "build_runtime_seconds": external.get("build_runtime_seconds"),
            "all90_strict_verification": verified["all90"],
            "vxc_unmatched_legacy_points": sum(
                item["metadata"]["source"]["vxc_matching"]["unmatched_legacy_points"]
                for item in records
            ),
            "max_vxc_identity_error_ha": max_vxc_error,
            "all_density_descriptors_float32_compatible": all(
                item["metadata"]["verification"]["central_descriptor_compatibility"][field][
                    "float32_level_compatible"
                ]
                for item in records
                for field in ("rho", "sigma_aa_ab_bb", "lapl")
            ),
            "max_density_descriptor_normalized_error": max_descriptor_error,
        },
        "records": records,
    }
    if FORBIDDEN_METADATA_KEYS.intersection(_recursive_keys(manifest)):
        raise ValueError("Root corpus manifest includes a forbidden h/stencil field.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary manifest: {temporary}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return manifest


__all__ = [
    "ALL90_EXPECTED_POINT_COUNT",
    "FEATURE_LAYOUT",
    "OPERATOR_PROTOCOL",
    "PROTOCOL",
    "REQUIRED_ARRAYS",
    "SPATIAL_DERIVATIVE_METHOD",
    "CentralOperatorRecord",
    "build_molecule_from_metadata",
    "density_descriptors_from_ao",
    "evaluate_density_descriptors",
    "iter_ao_factor_chunks",
    "load_central_operator_record",
    "project_reference_vxc_ao",
    "recover_legacy_centers",
    "verify_operator_corpus",
    "write_central_operator_record",
    "write_corpus_manifest",
]
