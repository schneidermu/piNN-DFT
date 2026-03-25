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


def _parse_branches(raw_value: str) -> set[str]:
    selected = {
        value.strip().lower()
        for value in raw_value.split(",")
        if value.strip()
    }
    if not selected or "all" in selected:
        return {"wtmad", "avrane"}
    invalid = selected - {"wtmad", "avrane"}
    if invalid:
        raise ValueError(f"Unknown branches requested: {', '.join(sorted(invalid))}")
    return selected


def _mark_unselected_branches(experiment, selected_branches: set[str]) -> None:
    for branch_name in experiment.manifest.branches:
        if branch_name not in selected_branches:
            experiment.set_branch_status(
                branch_name,
                "complete",
                message="Skipped by evaluate.py branch selection.",
                job_ids=[],
                artifacts=[],
                metrics={},
            )


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
    parser.add_argument(
        "--branches",
        default="all",
        help="Comma-separated branches to run: all, avrane, wtmad",
    )
    parser.add_argument(
        "--include-atoms",
        action="store_true",
        help="For avRANE runs, also generate atomic density jobs/artifacts for Max RMSD or MaxNE workflows",
    )
    args = parser.parse_args()

    experiment = create_experiment(
        checkpoint=args.checkpoint,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
        include_atoms=args.include_atoms,
    )
    selected_branches = _parse_branches(args.branches)
    _mark_unselected_branches(experiment, selected_branches)

    wait_for_completion = args.inline_wait
    branch_jobs = []
    if "wtmad" in selected_branches:
        branch_jobs.append(("wtmad", run_wtmad_branch, (experiment, wait_for_completion)))
    if "avrane" in selected_branches:
        branch_jobs.append(
            ("avrane", run_avrane_branch, (experiment, wait_for_completion, args.include_atoms))
        )

    with ThreadPoolExecutor(max_workers=max(1, len(branch_jobs))) as executor:
        futures = [
            executor.submit(branch_fn, *branch_args)
            for _, branch_fn, branch_args in branch_jobs
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
