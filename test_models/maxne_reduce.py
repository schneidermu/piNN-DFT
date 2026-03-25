from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from common import TEST_MODELS_ROOT, ensure_dir


MULTIWFN_CMD = os.environ.get("MULTIWFN_CMD", "Multiwfn")
DB2A = 1.0 / 0.529177
NORMALIZERS = {
    "RHO": 0.009943368,
    "GRD": 0.092398036,
    "LR": 1.445110833,
}


def resolve_maxne_reference_paths() -> dict[str, Path]:
    denrho_root = TEST_MODELS_ROOT.parent / "denrho"
    return {
        "denrho_root": denrho_root,
        "ref_atomic_dir": denrho_root / "dtestin" / "CCSD",
        "genrho_template": denrho_root / "content" / "genRHO.txt",
        "gengrd_template": denrho_root / "content" / "genGRD.txt",
        "genlr_template": denrho_root / "content" / "genLR.txt",
    }


def _assert_maxne_references_available(reference_paths: dict[str, Path]) -> None:
    if not reference_paths["ref_atomic_dir"].exists():
        raise FileNotFoundError(
            f"Missing atomic CCSD reference RDFs: {reference_paths['ref_atomic_dir']}"
        )
    for key in ("genrho_template", "gengrd_template", "genlr_template"):
        if not reference_paths[key].exists():
            raise FileNotFoundError(f"Missing Multiwfn template: {reference_paths[key]}")


def _run_multiwfn_for_mode(wfn_path: Path, template_path: Path, suffix: str) -> Path:
    result = subprocess.run(
        [MULTIWFN_CMD, str(wfn_path.name)],
        cwd=wfn_path.parent,
        input=template_path.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Multiwfn failed for {wfn_path.name} [{suffix}]: "
            f"{result.stderr.strip() or result.stdout.strip() or 'unknown error'}"
        )

    rdf_path = wfn_path.with_name(f"{wfn_path.stem}_{suffix}.rdf")
    default_rdf = wfn_path.parent / "RDF.txt"
    if not default_rdf.exists():
        raise FileNotFoundError(f"Multiwfn did not produce RDF.txt for {wfn_path.name} [{suffix}]")
    default_rdf.replace(rdf_path)
    return rdf_path


def generate_atomic_rdfs(functional_dir: Path, reference_paths: dict[str, Path]) -> list[str]:
    ensure_dir(functional_dir)
    wfn_files = sorted(functional_dir.glob("*.wfn"))
    if not wfn_files:
        raise FileNotFoundError(f"No atomic .wfn files found under {functional_dir}")

    template_map = {
        "RHO": reference_paths["genrho_template"],
        "GRD": reference_paths["gengrd_template"],
        "LR": reference_paths["genlr_template"],
    }
    generated = []
    for wfn_path in wfn_files:
        for suffix, template_path in template_map.items():
            generated.append(str(_run_multiwfn_for_mode(wfn_path, template_path, suffix)))
    return generated


def _read_rdf(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r" +", " ", line.strip())
        if not line:
            continue
        parts = [value for value in line.split(" ") if value]
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float64)


def _rdf_type(path: Path) -> str:
    return path.stem.split("_")[-1].replace("MIX-E", "MIX_E")


def _compute_maxne_tables(functional_dir: Path, reference_dir: Path) -> dict:
    normalized: dict[str, list[float]] = {key: [] for key in NORMALIZERS}
    raw: dict[str, list[float]] = {key: [] for key in NORMALIZERS}
    per_file: dict[str, dict[str, float | str]] = {}

    rdf_files = sorted(functional_dir.glob("*.rdf"))
    if not rdf_files:
        raise FileNotFoundError(f"No .rdf files found under {functional_dir}")

    for rdf_path in rdf_files:
        reference_path = reference_dir / rdf_path.name.replace(functional_dir.name, reference_dir.name)
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference RDF: {reference_path}")

        values = _read_rdf(rdf_path)[:, 2] * DB2A
        reference_values = _read_rdf(reference_path)[:, 2] * DB2A
        rms = float(np.sqrt(np.mean((values - reference_values) ** 2)))
        descriptor = _rdf_type(rdf_path)
        if descriptor not in NORMALIZERS:
            continue
        normalized_rms = rms / NORMALIZERS[descriptor]
        raw[descriptor].append(rms)
        normalized[descriptor].append(normalized_rms)
        per_file[rdf_path.name] = {
            "descriptor": descriptor,
            "rmsd_raw": rms,
            "rmsd_normalized": normalized_rms,
        }

    if not any(normalized.values()):
        raise ValueError(f"No supported RDF descriptor files were found under {functional_dir}")

    max_by_component = {key.lower(): float(max(values)) for key, values in normalized.items() if values}
    mean_by_component = {key.lower(): float(sum(values) / len(values)) for key, values in normalized.items() if values}
    summary = {
        "selected_summary_metric": float(max(max_by_component.values())),
        "max_component": max_by_component,
        "mean_component": mean_by_component,
        "max_ne_of_max": float(max(max_by_component.values())),
        "mean_ne_of_max": float(sum(max_by_component.values()) / len(max_by_component)),
        "max_ne_of_mean": float(max(mean_by_component.values())),
        "mean_ne_of_mean": float(sum(mean_by_component.values()) / len(mean_by_component)),
        "per_file": per_file,
    }
    return summary


def run_maxne_reduction(
    experiment_root: Path,
    functional: str,
) -> tuple[dict, list[str], dict[str, str]]:
    reference_paths = resolve_maxne_reference_paths()
    _assert_maxne_references_available(reference_paths)

    branch_root = experiment_root / "outputs" / "avrane"
    denrho_root = ensure_dir(branch_root / "denrho")
    functional_dir = ensure_dir(denrho_root / "dtestin" / functional)

    generated_rdfs = generate_atomic_rdfs(functional_dir, reference_paths)
    metrics = _compute_maxne_tables(functional_dir, reference_paths["ref_atomic_dir"])

    metrics_path = experiment_root / "reports" / "maxne_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts = [str(functional_dir), *generated_rdfs, str(metrics_path)]
    normalized_refs = {key: str(value) for key, value in reference_paths.items()}
    return metrics, artifacts, normalized_refs
