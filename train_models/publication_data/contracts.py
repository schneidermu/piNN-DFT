"""Canonical, unit-explicit contracts for the publication bundle."""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "lap-publication-dataset-v1"
VARIANTS = ("level2", "level2_delley", "level2_gauss_chebyshev", "level2_mura",
            "level3", "level3_delley", "level3_gauss_chebyshev", "level3_mura")
EXCLUSIONS = {"DBH76": [15, 35, 42, 43, 54, 55], "MGAE109": [18, 28, 34, 55, 74],
              "EA13": [4, 8], "NCCE31": [12, 21, 30]}
# Atomic units, except explicitly labelled reaction energies.
FIELDS = {
    "Grid": ("point,descriptor", "mixed: rho bohr^-3; sigma bohr^-8; tau/lapl bohr^-5"),
    "Weights": ("point", "bohr^3"), "Densities": ("point,spin", "bohr^-3"),
    "Gradients": ("point,sigma_aa_ab_bb", "bohr^-8"),
    "PBE_local_energies": ("point", "hartree/electron"),
    "HF_energies": ("component", "hartree"), "fixed_nonxc": ("scalar", "hartree"),
    "features": ("point,rho2_grad6_lapl2", "rho bohr^-3; grad bohr^-4; lapl bohr^-5"),
    "coords64": ("point,xyz", "bohr"), "legacycoords32": ("point,xyz", "bohr"),
    "DensityDescriptorsN10": ("point,rho2_grad6_lapl2", "rho bohr^-3; grad bohr^-4; lapl bohr^-5"),
    "weights": ("point", "bohr^3"), "VxcLegacy": ("point", "hartree"),
    "Exc": ("scalar", "hartree"), "dmks": ("ao,ao", "electron"),
    "dm": ("ao,ao or spin,ao,ao", "electron"), "RefAO": ("ao,ao", "hartree"),
    "Overlap": ("ao,ao", "dimensionless"), "phi": ("point,ao", "bohr^-3/2"),
    "grad_phi": ("point,xyz,ao", "bohr^-5/2"), "lap_phi": ("point,ao", "bohr^-7/2"),
    "sourcerow": ("point", "index"), "npzrow": ("point", "index"),
    "pbe_epsilon": ("point", "hartree/electron"), "nonxc": ("scalar", "hartree"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def canonical_id(kind, identity):
    return kind + "_" + hashlib.sha256(canonical(identity)).hexdigest()[:24]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(canonical({"shape": array.shape, "dtype": array.dtype.str})
                          + array.tobytes()).hexdigest()


def parse_d3(path, required):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Dispersion line must contain one ID and one value")
        key = parts[0].removesuffix(".gif_")
        value = float(parts[1])
        if key in values or not np.isfinite(value):
            raise ValueError("Duplicate/nonfinite dispersion entry")
        values[key] = value
    if set(values) != set(required):
        raise ValueError("Missing/unexpected dispersion identities")
    return values


def safe_child(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in str(relative):
        raise ValueError("Nonportable or escaping dataset path")
    target = (Path(root) / path).resolve()
    if Path(root).resolve() not in target.parents:
        raise ValueError("Escaping dataset path")
    return target


def annotate(dataset, key, production=None, source=None):
    axes, units = FIELDS[key]
    dataset.attrs.update({"semantic_name": key, "axes": axes, "units": units,
                          "storage_dtype": dataset.dtype.str,
                          "source_dtype": source or dataset.dtype.str,
                          "production_dtype": production or dataset.dtype.str})
