from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import TEST_MODELS_ROOT, ensure_dir


def _load_iodens():
    import sys

    den_mol_or_root = TEST_MODELS_ROOT.parent / "den_mol_or"
    if str(den_mol_or_root) not in sys.path:
        sys.path.insert(0, str(den_mol_or_root))
    import iodens  # type: ignore

    return iodens


def resolve_reference_paths() -> dict[str, Path]:
    den_mol_or_root = TEST_MODELS_ROOT.parent / "den_mol_or"
    ref_ccsd = den_mol_or_root / "REF" / "CCSORT.npz"
    lda_npz = den_mol_or_root / "DENS" / "LDA.npz"
    lda_calc_dir = den_mol_or_root / "calc" / "LDA_pyscf"
    return {
        "ref_ccsd": ref_ccsd,
        "lda_npz": lda_npz,
        "lda_calc_dir": lda_calc_dir,
        "den_mol_or_root": den_mol_or_root,
    }


def _assert_references_available(reference_paths: dict[str, Path]) -> None:
    if not reference_paths["ref_ccsd"].exists():
        raise FileNotFoundError(
            f"Missing CCSD reference densities: {reference_paths['ref_ccsd']}"
        )
    if not reference_paths["lda_npz"].exists() and not reference_paths["lda_calc_dir"].exists():
        raise FileNotFoundError(
            "Missing LDA reference densities. Expected either "
            f"{reference_paths['lda_npz']} or {reference_paths['lda_calc_dir']}"
        )


def _read_mwfn_directory(functional_dir: Path) -> dict[str, np.ndarray]:
    iodens = _load_iodens()
    molecule_arrays = {}
    for molecule_dir in sorted(path for path in functional_dir.iterdir() if path.is_dir()):
        required = [molecule_dir / "rho", molecule_dir / "grad", molecule_dir / "lapl"]
        if not all(path.exists() for path in required):
            continue
        molecule_arrays[molecule_dir.name] = iodens.read_mwfn(str(molecule_dir) + "/")
    return molecule_arrays


def _build_npz_from_functional_dir(functional_dir: Path, target_npz: Path) -> Path:
    arrays = _read_mwfn_directory(functional_dir)
    if not arrays:
        raise FileNotFoundError(f"No density grids found under {functional_dir}")
    np.savez_compressed(target_npz, **arrays)
    return target_npz


def ensure_experiment_lda_npz(dens_dir: Path, reference_paths: dict[str, Path]) -> Path:
    lda_target = dens_dir / "LDA.npz"
    if lda_target.exists():
        return lda_target
    if reference_paths["lda_npz"].exists():
        with np.load(reference_paths["lda_npz"]) as lda_data:
            arrays = {key: lda_data[key] for key in lda_data.files}
        np.savez_compressed(lda_target, **arrays)
        return lda_target
    return _build_npz_from_functional_dir(reference_paths["lda_calc_dir"], lda_target)


def build_functional_npz(
    functional: str,
    functional_calc_dir: Path,
    dens_dir: Path,
) -> Path:
    target_npz = dens_dir / f"{functional}.npz"
    return _build_npz_from_functional_dir(functional_calc_dir, target_npz)


def _compute_tables(
    functional_npz: Path,
    lda_npz: Path,
    ref_ccsd_npz: Path,
) -> dict:
    iodens = _load_iodens()

    with np.load(functional_npz) as functional_data, np.load(lda_npz) as lda_data, np.load(ref_ccsd_npz) as ref_data:
        files = list(lda_data.files)
        if not files:
            raise ValueError("LDA reference file contains no systems")

        for system in files:
            if system not in functional_data.files:
                raise KeyError(f"Functional density file is missing system {system}")
            if system not in ref_data.files:
                raise KeyError(f"CCSD reference file is missing system {system}")

        norm = np.array([iodens.niad_mwfn(lda_data[system], ref_data[system]) for system in files])
        rhonorm = float(sum(norm[:, 0]) / len(files))
        grdnorm = float(sum(norm[:, 1]) / len(files))
        lrnorm = float(sum(norm[:, 2]) / len(files))

        niad = np.array(
            [iodens.niad_mwfn(functional_data[system], ref_data[system]) for system in files]
        )

    rmses = {
        "rho": float(max(niad[:, 0]) / rhonorm),
        "grad": float(max(niad[:, 1]) / grdnorm),
        "lapl": float(max(niad[:, 2]) / lrnorm),
    }
    ranes = {
        "rho": float((sum(niad[:, 0]) / len(files)) / rhonorm),
        "grad": float((sum(niad[:, 1]) / len(files)) / grdnorm),
        "lapl": float((sum(niad[:, 2]) / len(files)) / lrnorm),
    }
    summary = {
        "selected_summary_metric": float(sum(ranes.values()) / 3.0),
        "systems": files,
        "norms": {
            "rho": rhonorm,
            "grad": grdnorm,
            "lapl": lrnorm,
        },
        "rmses": rmses,
        "ranes": ranes,
        "rmse_table": [
            {
                "functional": functional_npz.stem,
                "rho": rmses["rho"],
                "grad": rmses["grad"],
                "lapl": rmses["lapl"],
                "max_component": float(max(rmses.values())),
            }
        ],
        "rane_table": [
            {
                "functional": functional_npz.stem,
                "rho": ranes["rho"],
                "grad": ranes["grad"],
                "lapl": ranes["lapl"],
                "mean_component": float(sum(ranes.values()) / 3.0),
            }
        ],
        "per_system_niad": {
            system: {
                "rho": float(values[0]),
                "grad": float(values[1]),
                "lapl": float(values[2]),
            }
            for system, values in zip(files, niad)
        },
    }
    return summary


def run_avrane_reduction(experiment_root: Path, functional: str) -> tuple[dict, list[str], dict[str, str]]:
    reference_paths = resolve_reference_paths()
    _assert_references_available(reference_paths)

    branch_root = experiment_root / "outputs" / "avrane"
    den_mol_or_root = ensure_dir(branch_root / "den_mol_or")
    dens_dir = ensure_dir(den_mol_or_root / "DENS")
    calc_dir = den_mol_or_root / "calc" / functional

    functional_npz = build_functional_npz(functional, calc_dir, dens_dir)
    lda_npz = ensure_experiment_lda_npz(dens_dir, reference_paths)
    metrics = _compute_tables(functional_npz, lda_npz, reference_paths["ref_ccsd"])

    metrics_path = experiment_root / "reports" / "avrane_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts = [
        str(functional_npz),
        str(lda_npz),
        str(metrics_path),
        str(calc_dir),
    ]
    normalized_refs = {key: str(value) for key, value in reference_paths.items()}
    return metrics, artifacts, normalized_refs
