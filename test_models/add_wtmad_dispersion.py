from __future__ import annotations

import json
import re
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from common import GIF_DIR, RESULTS_DIR, TEST_MODELS_ROOT, normalize_gif_layout
from experiment import load_experiment
from reporting import write_reports


DISPERSION_CHOICES = {"pbe0-d3bj": "PBE0", "pbe-d3bj": "PBE"}


def _dispersion_slug(dispersion_correction: str) -> str:
    return dispersion_correction.lower().replace("-", "_")


def _energy_file_name(nfinal: int, functional: str, dispersion_correction: str) -> str:
    return f"EnergyList_{nfinal}_{functional}__disp_{_dispersion_slug(dispersion_correction)}.txt"


def _parse_wtmad_value(text: str) -> float | None:
    for line in text.splitlines():
        if "WTMAD" not in line.upper():
            continue
        match = re.search(r"WTMAD2?\s*=\s*(-?\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"WTMAD-?2?\s*[:=]\s*(-?\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _system_name_from_energy_key(value: str) -> str:
    if value.endswith(".gif_"):
        return value[:-5]
    return value


def _read_raw_energy_list(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Malformed energy-list line in {path}: {line}")
        rows.append((fields[0], fields[1]))
    return rows


def _read_gif_geometry(system_name: str) -> tuple[str, int, int]:
    gif_path = GIF_DIR / system_name / f"{system_name}.gif_"
    with gif_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
    charge, spin = map(int, lines[2].split())
    coords = "\n".join(" ".join(line.split()) for line in lines[3:])
    return coords, charge, spin - 1


def _build_geometry_molecule(system_name: str):
    from pyscf import gto

    coords, charge, spin = _read_gif_geometry(system_name)
    ecp_atoms = []
    if "I" in coords:
        ecp_atoms.append("I")
    if "Sb" in coords:
        ecp_atoms.append("Sb")
    if "Bi" in coords:
        ecp_atoms.append("Bi")

    molecule = gto.Mole()
    molecule.atom = coords
    molecule.ecp = {atom: "def2-qzvp" for atom in ecp_atoms}
    molecule.basis = "def2-qzvp"
    molecule.verbose = 0
    molecule.spin = spin
    molecule.charge = charge
    molecule.symmetry = False
    molecule.build()
    return molecule


def _dispersion_energy(system_name: str, dispersion_correction: str) -> float:
    import dftd3.pyscf as disp

    molecule = _build_geometry_molecule(system_name)
    d3 = disp.DFTD3Dispersion(
        molecule,
        xc=DISPERSION_CHOICES[dispersion_correction],
        version="d3bj",
    )
    return float(d3.kernel()[0])


def _write_corrected_energy_list(
    source_path: Path,
    target_path: Path,
    dispersion_correction: str,
) -> dict[str, float]:
    rows = _read_raw_energy_list(source_path)
    dispersion_values = {}
    with target_path.open("w", encoding="utf-8") as handle:
        for energy_key, raw_energy in rows:
            system_name = _system_name_from_energy_key(energy_key)
            if raw_energy == "ERROR":
                handle.write(f"{energy_key} ERROR\n")
                continue
            correction = _dispersion_energy(system_name, dispersion_correction)
            dispersion_values[system_name] = correction
            handle.write(f"{energy_key} {float(raw_energy) + correction}\n")
    return dispersion_values


def _run_interface_analysis(experiment, energy_file: Path, dispersion_correction: str) -> tuple[dict, Path]:
    interface_script = TEST_MODELS_ROOT / "InterfaceG16.py"
    if not interface_script.exists():
        return {"warning": "InterfaceG16.py not found; WTMAD metric not extracted."}, energy_file

    functional = experiment.manifest.generated_functional_name
    canonical_energy_file = RESULTS_DIR / f"EnergyList_30_{functional}.txt"
    shutil.copy2(energy_file, canonical_energy_file)

    result = subprocess.run(
        ["python", "InterfaceG16.py", "--Functional", functional],
        cwd=TEST_MODELS_ROOT,
        text=True,
        capture_output=True,
    )
    analysis_path = (
        experiment.reports_dir
        / f"wtmad_interface_output__disp_{_dispersion_slug(dispersion_correction)}.txt"
    )
    analysis_path.write_text(
        (result.stdout or "") + ("\nSTDERR:\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    metrics = {
        "interface_exit_code": result.returncode,
        "dispersion_correction": dispersion_correction,
        "wtmad_2": _parse_wtmad_value(result.stdout or ""),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"InterfaceG16.py analysis failed with exit code {result.returncode}. See {analysis_path}"
        )
    return metrics, analysis_path


def main() -> None:
    parser = ArgumentParser(
        description="Add D3(BJ) corrections to existing raw WTMAD energies without rerunning SCF."
    )
    parser.add_argument("--Manifest", required=True, help="Path to experiment manifest.json")
    parser.add_argument(
        "--DispersionCorrection",
        required=True,
        choices=sorted(DISPERSION_CHOICES),
        help="Posthoc correction to add to existing __disp_none energy list.",
    )
    parser.add_argument("--NFinal", type=int, default=30)
    args = parser.parse_args()

    normalize_gif_layout()
    experiment = load_experiment(args.Manifest)
    functional = experiment.manifest.generated_functional_name
    branch_dir = experiment.branch_output_dir("wtmad")
    source_path = branch_dir / _energy_file_name(args.NFinal, functional, "none")
    if not source_path.exists():
        legacy_path = branch_dir / f"EnergyList_{args.NFinal}_{functional}.txt"
        if legacy_path.exists():
            source_path = legacy_path
        else:
            raise FileNotFoundError(f"Raw no-dispersion energy list not found: {source_path}")

    target_path = branch_dir / _energy_file_name(
        args.NFinal,
        functional,
        args.DispersionCorrection,
    )
    dispersion_values = _write_corrected_energy_list(
        source_path,
        target_path,
        args.DispersionCorrection,
    )
    metrics, analysis_path = _run_interface_analysis(
        experiment,
        target_path,
        args.DispersionCorrection,
    )

    dispersion_path = (
        experiment.reports_dir
        / f"wtmad_dispersion_values__disp_{_dispersion_slug(args.DispersionCorrection)}.json"
    )
    dispersion_path.write_text(json.dumps(dispersion_values, indent=2), encoding="utf-8")

    branch = experiment.manifest.branches["wtmad"]
    artifacts = list(
        dict.fromkeys(
            [
                *branch.artifacts,
                str(target_path),
                str(analysis_path),
                str(dispersion_path),
            ]
        )
    )
    experiment.manifest.wtmad_dispersion_correction = args.DispersionCorrection
    experiment.set_branch_status(
        "wtmad",
        "complete",
        message=f"WTMAD branch postprocessed with {args.DispersionCorrection}.",
        job_ids=branch.job_ids,
        artifacts=artifacts,
        metrics=metrics,
    )
    branch_report = experiment.reports_dir / "wtmad.json"
    branch_report.write_text(
        json.dumps(
            {
                "functional": functional,
                "dispersion_correction": args.DispersionCorrection,
                "source_energy_file": str(source_path),
                "corrected_energy_file": str(target_path),
                "metrics": metrics,
                "output_dir": str(branch_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_reports(experiment)
    experiment.write_manifest()

    print(f"Corrected energy list: {target_path}")
    print(f"WTMAD metrics: {metrics}")
    print(f"Summary: {experiment.reports_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
