from __future__ import annotations

from optparse import OptionParser
from pathlib import Path

from DFT.functional import omega_str_list

from common import (
    ENERGY_LOG_DIR,
    GENERATED_JOBS_DIR,
    ensure_dir,
    ensure_runtime_directories,
    normalize_gif_layout,
    run_sbatch,
)

NN_FUNCTIONALS = (
    [f"NN_PBE_{omega}" for omega in omega_str_list]
    + [f"NN_XALPHA_{omega}" for omega in omega_str_list]
    + ["Nagai", "NN_PBE_star", "NN_PBE_star_star"]
    + [f"NN_PBE_star_star_{omega}" for omega in omega_str_list]
)
REFERENCE_FUNCTIONALS = ["PBE", "XAlpha", "r2SCAN", "SCAN", "TPSS"]
ALL_FUNCTIONALS = NN_FUNCTIONALS + REFERENCE_FUNCTIONALS

SCRIPT_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="E {system_name} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="{log_dir}/{functional}_{system_name}_%j.out"
# Executable
python -m script --System {system_name} --NFinal {nfinal} --Functional {functional} --OutputDir "{output_dir}"{checkpoint_args}
"""

DISPERSION_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="D3 {system_name}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="{log_dir}/D3BJ_{system_name}_%j.out"
# Executable
python -m script --Dispersion True --System {system_name} --NFinal {nfinal} --OutputDir "{output_dir}"
"""


def filter_system_names(system_names: list[str], subset_prefixes: list[str] | None) -> list[str]:
    if not subset_prefixes:
        return system_names
    return [
        system_name
        for system_name in system_names
        if any(system_name.startswith(prefix) for prefix in subset_prefixes)
    ]


def build_checkpoint_args(
    functional: str,
    checkpoint_path: str | None,
    model_key: str | None,
) -> str:
    if functional in REFERENCE_FUNCTIONALS or functional == "Nagai":
        return ""
    if not checkpoint_path or not model_key:
        return ""
    return f' --CheckpointPath "{checkpoint_path}" --ModelKey "{model_key}"'


def build_job_dir(system_name: str, jobs_root: Path | None = None) -> Path:
    base_dir = jobs_root or (GENERATED_JOBS_DIR / "energy")
    return ensure_dir(base_dir / system_name)


def write_slurm_file(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    print(path)


def generate_jobs(
    nfinal: int,
    *,
    jobs_root: Path | None = None,
    log_dir: Path | None = None,
    output_dir: Path | None = None,
    subset_prefixes: list[str] | None = None,
    explicit_functionals: list[str] | None = None,
    checkpoint_path: str | None = None,
    model_key: str | None = None,
) -> list[Path]:
    ensure_runtime_directories()
    log_dir = ensure_dir(log_dir or ENERGY_LOG_DIR)
    output_dir = ensure_dir(output_dir or Path("Results"))
    functionals = explicit_functionals or ALL_FUNCTIONALS
    system_names = filter_system_names(normalize_gif_layout(), subset_prefixes)
    created_jobs: list[Path] = []

    for system_name in system_names:
        job_dir = build_job_dir(system_name, jobs_root=jobs_root)
        for functional in functionals:
            slurm_path = job_dir / f"calculate_system_energy_{functional}.slurm"
            write_slurm_file(
                slurm_path,
                SCRIPT_TEMPLATE.format(
                    system_name=system_name,
                    functional=functional,
                    log_dir=log_dir.as_posix(),
                    nfinal=nfinal,
                    output_dir=output_dir.as_posix(),
                    checkpoint_args=build_checkpoint_args(
                        functional, checkpoint_path, model_key
                    ),
                ),
            )
            created_jobs.append(slurm_path)

        dispersion_path = job_dir / "calculate_system_dispersion.slurm"
        write_slurm_file(
            dispersion_path,
            DISPERSION_TEMPLATE.format(
                system_name=system_name,
                log_dir=log_dir.as_posix(),
                nfinal=nfinal,
                output_dir=output_dir.as_posix(),
            ),
        )
        created_jobs.append(dispersion_path)
    return created_jobs


def submit_jobs(
    functional: str,
    *,
    jobs_root: Path | None = None,
    subset_prefixes: list[str] | None = None,
) -> list[str]:
    ensure_runtime_directories()
    job_ids = []
    for system_name in filter_system_names(normalize_gif_layout(), subset_prefixes):
        slurm_path = build_job_dir(system_name, jobs_root=jobs_root) / f"calculate_system_energy_{functional}.slurm"
        print(slurm_path)
        job_ids.append(run_sbatch(slurm_path))
    return job_ids


def submit_dispersion_jobs(
    *,
    jobs_root: Path | None = None,
    subset_prefixes: list[str] | None = None,
) -> list[str]:
    ensure_runtime_directories()
    job_ids = []
    for system_name in filter_system_names(normalize_gif_layout(), subset_prefixes):
        slurm_path = build_job_dir(system_name, jobs_root=jobs_root) / "calculate_system_dispersion.slurm"
        print(slurm_path)
        job_ids.append(run_sbatch(slurm_path))
    return job_ids


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--Mode", type="string", default="Analyse", help="Mode")
    parser.add_option(
        "--Functional", type="string", default="NN_PBE_0", help="Functional to evaluate"
    )
    parser.add_option(
        "--NFinal", type="int", default=30, help="Number of systems in the benchmark set"
    )
    parser.add_option("--OutputDir", type="string", default="Results")
    parser.add_option("--JobsDir", type="string", default="")
    parser.add_option("--LogsDir", type="string", default="")
    parser.add_option("--Subset", type="string", default="")
    parser.add_option("--CheckpointPath", type="string", default="")
    parser.add_option("--ModelKey", type="string", default="")
    parser.add_option("--AllFunctionals", action="store_true", default=False)

    (Opts, args) = parser.parse_args()

    subset_prefixes = [value for value in Opts.Subset.split(",") if value]
    jobs_root = Path(Opts.JobsDir) if Opts.JobsDir else None
    log_dir = Path(Opts.LogsDir) if Opts.LogsDir else None
    output_dir = Path(Opts.OutputDir)

    mode = Opts.Mode.upper()[:2]
    if mode == "GE":
        generate_jobs(
            Opts.NFinal,
            jobs_root=jobs_root,
            log_dir=log_dir,
            output_dir=output_dir,
            subset_prefixes=subset_prefixes,
            explicit_functionals=None if Opts.AllFunctionals else [Opts.Functional],
            checkpoint_path=Opts.CheckpointPath or None,
            model_key=Opts.ModelKey or None,
        )
    elif mode == "CE":
        submit_jobs(
            Opts.Functional,
            jobs_root=jobs_root,
            subset_prefixes=subset_prefixes,
        )
    elif mode == "D3":
        submit_dispersion_jobs(
            jobs_root=jobs_root,
            subset_prefixes=subset_prefixes,
        )
