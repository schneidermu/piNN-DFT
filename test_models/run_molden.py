from __future__ import annotations

from optparse import OptionParser
from pathlib import Path

from common import (
    ATOM_LOG_DIR,
    GENERATED_JOBS_DIR,
    MOLDEN_LOG_DIR,
    ensure_dir,
    ensure_runtime_directories,
    run_sbatch,
)

MOLECULES = ["BH3", "CO", "F2", "H2", "H2O", "HF", "Li2", "LiF", "LiH", "N2"]
ATOMS = [
    ("Be", 0),
    ("B", 1),
    ("B", 3),
    ("C", 2),
    ("C", 4),
    ("N", 3),
    ("N", 5),
    ("O", 4),
    ("O", 6),
    ("F", 5),
    ("F", 7),
    ("Ne", 0),
    ("Ne", 6),
    ("Ne", 8),
]

MOLECULE_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="Rho {molecule} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output="{log_dir}/{functional}_{molecule}_%j.out"
# Executable
python -m get_molden --Functional '{functional}' --Molecule {molecule} --ExperimentRoot "{experiment_root}"{checkpoint_args}
"""

ATOM_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="Rho {atom} +{charge} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output="{log_dir}/{functional}_{atom}_+{charge}_%j.out"
# Executable
python -m get_molden --Functional '{functional}' --Atom {atom} --Charge {charge} --ExperimentRoot "{experiment_root}"{checkpoint_args}
"""


def filter_molecules(molecules: list[str], subset_names: list[str] | None) -> list[str]:
    if not subset_names:
        return molecules
    subset = set(subset_names)
    return [molecule for molecule in molecules if molecule in subset]


def build_checkpoint_args(functional: str, checkpoint_path: str | None, model_key: str | None) -> str:
    if not checkpoint_path or not model_key:
        return ""
    if functional == "Nagai":
        return ""
    if functional in {"PBE", "PBE0", "XAlpha", "r2SCAN", "SCAN", "TPSS"}:
        return ""
    return f' --CheckpointPath "{checkpoint_path}" --ModelKey "{model_key}"'


def build_job_dir(jobs_root: Path | None = None) -> Path:
    return ensure_dir((jobs_root or GENERATED_JOBS_DIR / "molden"))


def write_job(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    print(path)


def generate_jobs(
    functional: str,
    *,
    experiment_root: Path,
    jobs_root: Path | None = None,
    molecule_log_dir: Path | None = None,
    atom_log_dir: Path | None = None,
    subset_molecules: list[str] | None = None,
    checkpoint_path: str | None = None,
    model_key: str | None = None,
    include_atoms: bool = True,
) -> list[Path]:
    ensure_runtime_directories()
    job_dir = build_job_dir(jobs_root)
    molecule_log_dir = ensure_dir(molecule_log_dir or MOLDEN_LOG_DIR)
    atom_log_dir = ensure_dir(atom_log_dir or ATOM_LOG_DIR)
    created_jobs = []

    for molecule in filter_molecules(MOLECULES, subset_molecules):
        slurm_file = job_dir / f"get_molden_{functional}_{molecule}.slurm"
        write_job(
            slurm_file,
            MOLECULE_TEMPLATE.format(
                functional=functional,
                molecule=molecule,
                log_dir=molecule_log_dir.as_posix(),
                experiment_root=experiment_root.as_posix(),
                checkpoint_args=build_checkpoint_args(
                    functional, checkpoint_path, model_key
                ),
            ),
        )
        created_jobs.append(slurm_file)

    if include_atoms:
        for atom, charge in ATOMS:
            slurm_file = job_dir / f"get_molden_{functional}_{atom}_plus_{charge}.slurm"
            write_job(
                slurm_file,
                ATOM_TEMPLATE.format(
                    functional=functional,
                    atom=atom,
                    charge=charge,
                    log_dir=atom_log_dir.as_posix(),
                    experiment_root=experiment_root.as_posix(),
                    checkpoint_args=build_checkpoint_args(
                        functional, checkpoint_path, model_key
                    ),
                ),
            )
            created_jobs.append(slurm_file)

    return created_jobs


def submit_jobs(
    functional: str,
    *,
    jobs_root: Path | None = None,
    subset_molecules: list[str] | None = None,
    include_atoms: bool = True,
) -> list[str]:
    job_dir = build_job_dir(jobs_root)
    job_ids = []

    for molecule in filter_molecules(MOLECULES, subset_molecules):
        slurm_file = job_dir / f"get_molden_{functional}_{molecule}.slurm"
        job_ids.append(run_sbatch(slurm_file))

    if include_atoms:
        for atom, charge in ATOMS:
            slurm_file = job_dir / f"get_molden_{functional}_{atom}_plus_{charge}.slurm"
            job_ids.append(run_sbatch(slurm_file))

    return job_ids


if __name__ == "__main__":
    ensure_runtime_directories()

    parser = OptionParser()
    parser.add_option(
        "--Functional", type=str, default="PBE0", help="Functional for calculation"
    )
    parser.add_option("--ExperimentRoot", type=str, default="")
    parser.add_option("--JobsDir", type=str, default="")
    parser.add_option("--MoleculeLogsDir", type=str, default="")
    parser.add_option("--AtomLogsDir", type=str, default="")
    parser.add_option("--SubsetMolecules", type=str, default="")
    parser.add_option("--CheckpointPath", type=str, default="")
    parser.add_option("--ModelKey", type=str, default="")
    parser.add_option("--Mode", type=str, default="generate")
    parser.add_option("--IncludeAtoms", type=str, default="True")
    (Opts, args) = parser.parse_args()

    subset_molecules = [value for value in Opts.SubsetMolecules.split(",") if value]
    jobs_root = Path(Opts.JobsDir) if Opts.JobsDir else None
    molecule_logs_dir = Path(Opts.MoleculeLogsDir) if Opts.MoleculeLogsDir else None
    atom_logs_dir = Path(Opts.AtomLogsDir) if Opts.AtomLogsDir else None
    experiment_root = Path(Opts.ExperimentRoot) if Opts.ExperimentRoot else Path.cwd()
    include_atoms = str(Opts.IncludeAtoms).lower() in {"1", "true", "yes"}

    if Opts.Mode.lower().startswith("gen"):
        generate_jobs(
            Opts.Functional,
            experiment_root=experiment_root,
            jobs_root=jobs_root,
            molecule_log_dir=molecule_logs_dir,
            atom_log_dir=atom_logs_dir,
            subset_molecules=subset_molecules,
            checkpoint_path=Opts.CheckpointPath or None,
            model_key=Opts.ModelKey or None,
            include_atoms=include_atoms,
        )
    else:
        submit_jobs(
            Opts.Functional,
            jobs_root=jobs_root,
            subset_molecules=subset_molecules,
            include_atoms=include_atoms,
        )
