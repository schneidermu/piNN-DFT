from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from branches import run_avrane_branch, run_wtmad_branch
from common import TEST_MODELS_ROOT, ensure_dir, run_sbatch
from experiment import create_experiment
from reporting import write_reports


FINALIZER_TEMPLATE = """#! /bin/bash
#SBATCH --job-name="Finalize {experiment_slug}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output="{log_path}"
python -m finalize_experiment --Manifest "{manifest_path}"
"""


def submit_finalizer_job(experiment) -> tuple[Path, str]:
    jobs_dir = ensure_dir(experiment.jobs_dir / "orchestration")
    logs_dir = ensure_dir(experiment.logs_dir / "orchestration")
    slurm_path = jobs_dir / "finalize_experiment.slurm"
    log_path = logs_dir / "finalize_experiment_%j.out"
    slurm_path.write_text(
        FINALIZER_TEMPLATE.format(
            experiment_slug=experiment.manifest.experiment_slug,
            log_path=log_path.as_posix(),
            manifest_path=experiment.manifest_path.as_posix(),
        ),
        encoding="utf-8",
    )
    job_id = run_sbatch(slurm_path)
    return slurm_path, job_id


def main() -> None:
    parser = ArgumentParser(description="Run staged piNN-DFT evaluation experiments.")
    parser.add_argument("checkpoint", help="Path to the checkpoint to evaluate")
    parser.add_argument("experiment_name", help="Operator-friendly experiment name")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the reduced smoke subset for WTMAD-2 and avRANE",
    )
    parser.add_argument(
        "--inline-wait",
        action="store_true",
        help="Wait in the current terminal instead of a finalize SLURM job",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Deprecated alias; waiting now happens in a finalize SLURM job by default",
    )
    args = parser.parse_args()

    experiment = create_experiment(
        checkpoint=args.checkpoint,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
    )

    wait_for_completion = args.inline_wait
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_wtmad_branch, experiment, wait_for_completion),
            executor.submit(run_avrane_branch, experiment, wait_for_completion),
        ]
        for future in as_completed(futures):
            future.result()

    if wait_for_completion:
        write_reports(experiment)
        experiment.write_manifest()
        print(f"Experiment created at: {experiment.root}")
        print(f"Overall status: {experiment.overall_status()}")
        print(f"Summary: {experiment.reports_dir / 'summary.md'}")
        return

    finalizer_slurm_path, finalizer_job_id = submit_finalizer_job(experiment)

    print(f"Experiment created at: {experiment.root}")
    print("Overall status: running")
    print(f"Summary: {experiment.reports_dir / 'summary.md'}")
    print(f"Finalize job: {finalizer_job_id}")
    print(f"Finalize script: {finalizer_slurm_path}")


if __name__ == "__main__":
    main()
