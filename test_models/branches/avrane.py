from __future__ import annotations

import json

from avrane_reduce import resolve_reference_paths, run_avrane_reduction
from common import wait_for_slurm_jobs
from experiment import Experiment
from maxne_reduce import resolve_maxne_reference_paths, run_maxne_reduction
from run_molden import generate_jobs, submit_jobs


def finalize_avrane_branch(experiment: Experiment, include_atoms: bool = False) -> None:
    branch_name = "avrane"
    branch = experiment.manifest.branches[branch_name]
    artifacts = list(branch.artifacts)
    resolved_reference_paths = {
        key: str(value) for key, value in resolve_reference_paths().items()
    }
    subset_molecules = (
        experiment.manifest.smoke_avrane_molecules if experiment.manifest.smoke else None
    )
    wait_for_slurm_jobs(branch.job_ids)
    metrics, reduction_artifacts, reference_paths = run_avrane_reduction(
        experiment.root,
        experiment.manifest.generated_functional_name,
        systems=subset_molecules,
    )
    artifacts.extend(reduction_artifacts)
    merged_reference_paths = dict(reference_paths)
    if include_atoms:
        maxne_metrics, maxne_artifacts, maxne_reference_paths = run_maxne_reduction(
            experiment.root,
            experiment.manifest.generated_functional_name,
        )
        artifacts.extend(maxne_artifacts)
        merged_reference_paths.update(maxne_reference_paths)
        metrics = {
            "avrane": metrics,
            "maxne": maxne_metrics,
        }
    experiment.set_reference_paths(merged_reference_paths)

    branch_report = experiment.reports_dir / "avrane.json"
    branch_report.write_text(
        json.dumps(
            {
                "functional": experiment.manifest.generated_functional_name,
                "job_ids": branch.job_ids,
                "subset_molecules": subset_molecules,
                "include_atoms": include_atoms,
                "reference_paths": merged_reference_paths,
                "metrics": metrics,
                "output_dir": str(experiment.branch_output_dir(branch_name)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append(str(branch_report))
    experiment.set_branch_status(
        branch_name,
        "complete",
        message="avRANE branch finished.",
        job_ids=branch.job_ids,
        artifacts=artifacts,
        metrics=metrics,
    )


def run_avrane_branch(
    experiment: Experiment,
    wait: bool = True,
    include_atoms: bool = False,
) -> None:
    branch_name = "avrane"
    functional = experiment.manifest.generated_functional_name
    job_ids: list[str] = []
    artifacts = []
    branch_report = experiment.reports_dir / "avrane.json"
    resolved_reference_paths = {
        key: str(value) for key, value in resolve_reference_paths().items()
    }
    if include_atoms:
        resolved_reference_paths.update(
            {key: str(value) for key, value in resolve_maxne_reference_paths().items()}
        )
    experiment.set_reference_paths(resolved_reference_paths)
    try:
        experiment.set_branch_status(branch_name, "running", message="Generating avRANE jobs.")
        branch_jobs_dir = experiment.branch_jobs_dir(branch_name)
        branch_logs_dir = experiment.branch_logs_dir(branch_name)
        branch_output_dir = experiment.branch_output_dir(branch_name)
        artifacts.append(str(branch_output_dir))
        subset_molecules = (
            experiment.manifest.smoke_avrane_molecules if experiment.manifest.smoke else None
        )

        generate_jobs(
            functional,
            experiment_root=experiment.root,
            jobs_root=branch_jobs_dir,
            molecule_log_dir=branch_logs_dir / "molecules",
            atom_log_dir=branch_logs_dir / "atoms",
            subset_molecules=subset_molecules,
            checkpoint_path=experiment.manifest.checkpoint_copy,
            model_key=experiment.manifest.model_key,
            include_atoms=include_atoms,
        )
        job_ids = submit_jobs(
            functional,
            jobs_root=branch_jobs_dir,
            subset_molecules=subset_molecules,
            include_atoms=include_atoms,
        )
        experiment.set_branch_status(
            branch_name,
            "running",
            message="avRANE jobs submitted.",
            job_ids=job_ids,
        )

        if wait:
            subset_molecules = (
                experiment.manifest.smoke_avrane_molecules if experiment.manifest.smoke else None
            )
            wait_for_slurm_jobs(job_ids)
            metrics, reduction_artifacts, reference_paths = run_avrane_reduction(
                experiment.root,
                functional,
                systems=subset_molecules,
            )
            artifacts.extend(reduction_artifacts)
            merged_reference_paths = dict(reference_paths)
            if include_atoms:
                maxne_metrics, maxne_artifacts, maxne_reference_paths = run_maxne_reduction(
                    experiment.root,
                    functional,
                )
                artifacts.extend(maxne_artifacts)
                merged_reference_paths.update(maxne_reference_paths)
                metrics = {
                    "avrane": metrics,
                    "maxne": maxne_metrics,
                }
            experiment.set_reference_paths(merged_reference_paths)
        else:
            metrics = {}
            merged_reference_paths = resolved_reference_paths

        branch_report.write_text(
            json.dumps(
                {
                    "functional": functional,
                    "job_ids": job_ids,
                    "subset_molecules": subset_molecules,
                    "include_atoms": include_atoms,
                    "reference_paths": merged_reference_paths,
                    "metrics": metrics,
                    "output_dir": str(branch_output_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts.append(str(branch_report))
        if wait:
            experiment.set_branch_status(
                branch_name,
                "complete",
                message="avRANE branch finished.",
                job_ids=job_ids,
                artifacts=artifacts,
                metrics=metrics,
            )
        else:
            experiment.set_branch_status(
                branch_name,
                "running",
                message="avRANE jobs submitted; waiting skipped.",
                job_ids=job_ids,
                artifacts=artifacts,
                metrics=metrics,
            )
    except Exception as exc:
        branch_report.write_text(
            json.dumps(
                {
                    "functional": functional,
                    "job_ids": job_ids,
                    "include_atoms": include_atoms,
                    "reference_paths": resolved_reference_paths,
                    "metrics": {},
                    "error": str(exc),
                    "output_dir": str(experiment.branch_output_dir(branch_name)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts.append(str(branch_report))
        if job_ids:
            artifacts.append(str(experiment.branch_output_dir(branch_name)))
        experiment.set_branch_status(
            branch_name,
            "failed",
            message=str(exc),
            job_ids=job_ids,
            artifacts=artifacts,
        )
