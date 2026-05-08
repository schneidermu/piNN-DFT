from __future__ import annotations

import subprocess
import shutil
from argparse import ArgumentParser
from pathlib import Path

from common import ensure_dir, run_sbatch
from experiment import load_experiment
from run_molden import MOLECULES, filter_molecules


JOB_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="MWFN {molecule} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output="{log_path}"
python -m extract_avrane_grids --Manifest "{manifest_path}" --Molecule {molecule} --MultiwfnCmd "{multiwfn_cmd}" --Mode run-one
"""


def _functional_calc_dir(manifest_path: Path) -> tuple[Path, str, Path]:
    experiment = load_experiment(str(manifest_path))
    functional = experiment.manifest.generated_functional_name
    calc_dir = (
        experiment.root
        / "outputs"
        / "avrane"
        / "den_mol_or"
        / "calc"
        / functional
    )
    return calc_dir, functional, experiment.root


def _multiwfn_input(grid_dir: Path, molecule_dir: Path) -> str:
    return "\n".join(
        [
            "5",
            "1",
            "100",
            str(grid_dir),
            str(molecule_dir / "rho"),
            "5",
            "2",
            "100",
            str(grid_dir),
            str(molecule_dir / "grad"),
            "5",
            "3",
            "100",
            str(grid_dir),
            str(molecule_dir / "lapl"),
            "q",
            "",
        ]
    )


def resolve_multiwfn_cmd(cmd: str) -> str:
    path = Path(cmd).expanduser()
    if path.parent != Path("."):
        if path.exists():
            return str(path)
    elif shutil.which(cmd):
        return cmd
    raise FileNotFoundError(
        "Multiwfn executable was not found. Pass --MultiwfnCmd /path/to/Multiwfn."
    )


def run_one_molecule(manifest_path: Path, molecule: str, multiwfn_cmd: str) -> None:
    calc_dir, _functional, _experiment_root = _functional_calc_dir(manifest_path)
    molecule_dir = calc_dir / molecule
    wfn_path = molecule_dir / "gamess.wfn"
    grid_dir = manifest_path.parents[2].parent / "den_mol_or" / "grids" / f"grid_{molecule}"
    output_path = molecule_dir / "calc.out"

    if not wfn_path.exists():
        raise FileNotFoundError(f"Missing WFN file: {wfn_path}")
    if not grid_dir.exists():
        raise FileNotFoundError(f"Missing reference grid directory: {grid_dir}")

    resolved_multiwfn = resolve_multiwfn_cmd(multiwfn_cmd)
    with output_path.open("a", encoding="utf-8") as output_file:
        subprocess.run(
            [resolved_multiwfn, wfn_path.name],
            input=_multiwfn_input(grid_dir, molecule_dir),
            text=True,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            cwd=molecule_dir,
            check=True,
        )

    missing = [
        name
        for name in ("rho", "grad", "lapl")
        if not (molecule_dir / name).exists() or (molecule_dir / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Multiwfn finished but did not create non-empty files for {molecule}: {missing}. "
            f"See {output_path}"
        )


def submit_jobs(
    manifest_path: Path,
    molecules: list[str],
    multiwfn_cmd: str,
) -> list[str]:
    _calc_dir, functional, experiment_root = _functional_calc_dir(manifest_path)
    jobs_dir = ensure_dir(experiment_root / "jobs" / "avrane_multiwfn")
    logs_dir = ensure_dir(experiment_root / "logs" / "avrane_multiwfn")
    job_ids = []
    for molecule in molecules:
        slurm_path = jobs_dir / f"extract_avrane_grids_{molecule}.slurm"
        slurm_path.write_text(
            JOB_TEMPLATE.format(
                molecule=molecule,
                functional=functional,
                log_path=(logs_dir / f"{molecule}_%j.out").as_posix(),
                manifest_path=manifest_path.as_posix(),
                multiwfn_cmd=multiwfn_cmd,
            ),
            encoding="utf-8",
        )
        job_ids.append(run_sbatch(slurm_path))
    return job_ids


def main() -> None:
    parser = ArgumentParser(
        description="Run Multiwfn grid extraction from existing avRANE gamess.wfn files."
    )
    parser.add_argument("--Manifest", required=True, help="Path to experiment manifest.json")
    parser.add_argument("--MultiwfnCmd", required=True, help="Path to the Multiwfn executable")
    parser.add_argument(
        "--Mode",
        choices=("run-one", "submit"),
        default="submit",
        help="Submit one SLURM job per molecule, or run one molecule in the current process.",
    )
    parser.add_argument("--Molecule", default="", help="Required for --Mode run-one")
    parser.add_argument(
        "--Molecules",
        default="",
        help="Optional comma-separated subset for submit mode. Defaults to all avRANE molecules.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.Manifest).resolve()
    if args.Mode == "run-one":
        if not args.Molecule:
            raise ValueError("--Molecule is required for --Mode run-one")
        run_one_molecule(manifest_path, args.Molecule, args.MultiwfnCmd)
        return

    subset = [value for value in args.Molecules.split(",") if value]
    molecules = filter_molecules(MOLECULES, subset or None)
    job_ids = submit_jobs(manifest_path, molecules, args.MultiwfnCmd)
    print("Submitted Multiwfn extraction jobs:")
    print(",".join(job_ids))


if __name__ == "__main__":
    main()
