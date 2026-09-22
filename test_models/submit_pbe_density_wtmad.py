from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from calculate_system_energies import filter_system_names
from common import ensure_dir, normalize_gif_layout, run_sbatch


PBE_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="PBE density {system}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="{log_dir}/pbe_density_{system}_%j.out"
python -m script --System {system} --DensityMode pbe-save --DensityCheckpointDir "{density_dir}"
"""


NN_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="NN@PBE {system}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="{log_dir}/nn_on_pbe_{system}_%j.out"
python -m script --System {system} --NFinal 30 --Functional "{functional}" --OutputDir "{output_dir}" --DispersionCorrection pbe-d3bj --CheckpointPath "{checkpoint}" --ModelKey "{model_key}" --DensityMode nn-on-pbe --DensityCheckpointDir "{density_dir}"
"""


FINALIZE_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="Finalize NN@PBE WTMAD"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output="{log_dir}/finalize_%j.out"
set -euo pipefail
cp "{energy_file}" "{test_models_root}/Results/EnergyList_30_{functional}.txt"
cd "{test_models_root}"
python InterfaceG16.py --Functional "{functional}" | tee "{report_path}"
"""


def main() -> None:
    parser = ArgumentParser(
        description="Submit PBE-density generation followed by fixed-density NN WTMAD jobs."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("functional")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-key", default="NN_PBE-L")
    parser.add_argument("--subset", default="")
    parser.add_argument(
        "--reuse-pbe-densities",
        action="store_true",
        help="Reuse checkpoints carrying a .complete marker instead of rerunning PBE.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    root = ensure_dir(args.output_root.resolve())
    jobs_dir = ensure_dir(root / "jobs")
    logs_dir = ensure_dir(root / "logs")
    density_dir = ensure_dir(root / "pbe_density_checkpoints")
    output_dir = ensure_dir(root / "results")
    subset = [item for item in args.subset.split(",") if item]
    systems = filter_system_names(normalize_gif_layout(), subset or None)

    submitted = []
    nn_job_ids = []
    for system in systems:
        system_jobs = ensure_dir(jobs_dir / system)
        pbe_job = system_jobs / "pbe_density.slurm"
        nn_job = system_jobs / "nn_on_pbe.slurm"
        pbe_job.write_text(
            PBE_TEMPLATE.format(
                system=system,
                log_dir=logs_dir.as_posix(),
                density_dir=density_dir.as_posix(),
            ),
            encoding="utf-8",
        )
        nn_job.write_text(
            NN_TEMPLATE.format(
                system=system,
                log_dir=logs_dir.as_posix(),
                density_dir=density_dir.as_posix(),
                output_dir=output_dir.as_posix(),
                functional=args.functional,
                checkpoint=checkpoint.as_posix(),
                model_key=args.model_key,
            ),
            encoding="utf-8",
        )
        density_path = density_dir / f"{system}.pbe.chk"
        marker_path = density_path.with_suffix(density_path.suffix + ".complete")
        reuse_density = (
            args.reuse_pbe_densities
            and density_path.is_file()
            and marker_path.is_file()
        )
        if reuse_density:
            pbe_id = "reused"
            nn_id = run_sbatch(nn_job)
        else:
            pbe_id = run_sbatch(pbe_job)
            nn_id = run_sbatch(nn_job, extra_args=[f"--dependency=afterok:{pbe_id}"])
        nn_job_ids.append(nn_id)
        submitted.append((system, pbe_id, nn_id))
        print(f"{system}: PBE={pbe_id} NN@PBE={nn_id}")

    energy_file = (
        output_dir
        / f"EnergyList_30_{args.functional}__disp_pbe_d3bj.txt"
    )
    finalizer = jobs_dir / "finalize_wtmad.slurm"
    finalizer.write_text(
        FINALIZE_TEMPLATE.format(
            log_dir=logs_dir.as_posix(),
            energy_file=energy_file.as_posix(),
            test_models_root=Path(__file__).resolve().parent.as_posix(),
            functional=args.functional,
            report_path=(root / "wtmad_interface_output.txt").as_posix(),
        ),
        encoding="utf-8",
    )
    finalizer_id = run_sbatch(
        finalizer,
        extra_args=[f"--dependency=afterok:{':'.join(nn_job_ids)}"],
    )
    print(f"Submitted {len(submitted)} PBE -> NN@PBE job pairs under {root}")
    print(f"Finalizer={finalizer_id}; report={root / 'wtmad_interface_output.txt'}")


if __name__ == "__main__":
    main()
