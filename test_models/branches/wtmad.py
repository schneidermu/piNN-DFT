from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from calculate_system_energies import generate_jobs, submit_jobs
from common import RESULTS_DIR, TEST_MODELS_ROOT, wait_for_slurm_jobs
from experiment import Experiment


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


def _run_interface_analysis(
    experiment: Experiment,
    dispersion_correction: str,
) -> tuple[dict, list[str]]:
    interface_script = TEST_MODELS_ROOT / "InterfaceG16.py"
    if not interface_script.exists():
        return {"warning": "InterfaceG16.py not found; WTMAD metric not extracted."}, []

    branch_dir = experiment.branch_output_dir("wtmad")
    functional = experiment.manifest.generated_functional_name
    energy_file = branch_dir / _energy_file_name(30, functional, dispersion_correction)
    if not energy_file.exists():
        legacy_energy_file = branch_dir / f"EnergyList_30_{functional}.txt"
        if legacy_energy_file.exists():
            energy_file = legacy_energy_file
    mirrored_energy_file = RESULTS_DIR / f"EnergyList_30_{functional}.txt"
    shutil.copy2(energy_file, mirrored_energy_file)

    result = subprocess.run(
        ["python", "InterfaceG16.py", "--Functional", functional],
        cwd=TEST_MODELS_ROOT,
        text=True,
        capture_output=True,
    )
    analysis_path = experiment.reports_dir / "wtmad_interface_output.txt"
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
    return metrics, [str(analysis_path), str(energy_file)]


def finalize_wtmad_branch(experiment: Experiment) -> None:
    branch_name = "wtmad"
    branch = experiment.manifest.branches[branch_name]
    artifacts = list(branch.artifacts)
    dispersion_correction = experiment.manifest.wtmad_dispersion_correction
    wait_for_slurm_jobs(branch.job_ids)
    metrics, analysis_artifacts = _run_interface_analysis(experiment, dispersion_correction)
    artifacts.extend(analysis_artifacts)
    branch_report = experiment.reports_dir / "wtmad.json"
    branch_report.write_text(
        json.dumps(
            {
                "functional": experiment.manifest.generated_functional_name,
                "dispersion_correction": dispersion_correction,
                "job_ids": branch.job_ids,
                "subset_prefixes": (
                    experiment.manifest.smoke_wtmad_databases
                    if experiment.manifest.smoke
                    else None
                ),
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
        message="WTMAD branch finished.",
        job_ids=branch.job_ids,
        artifacts=artifacts,
        metrics=metrics,
    )


def run_wtmad_branch(
    experiment: Experiment,
    wait: bool = True,
    dispersion_correction: str = "none",
) -> None:
    branch_name = "wtmad"
    functional = experiment.manifest.generated_functional_name
    job_ids: list[str] = []
    artifacts = []
    branch_report = experiment.reports_dir / "wtmad.json"
    try:
        experiment.set_branch_status(branch_name, "running", message="Generating WTMAD jobs.")
        branch_jobs_dir = experiment.branch_jobs_dir(branch_name)
        branch_logs_dir = experiment.branch_logs_dir(branch_name)
        branch_output_dir = experiment.branch_output_dir(branch_name)
        artifacts.append(str(branch_output_dir))
        subset_prefixes = (
            experiment.manifest.smoke_wtmad_databases if experiment.manifest.smoke else None
        )

        generate_jobs(
            30,
            jobs_root=branch_jobs_dir,
            log_dir=branch_logs_dir,
            output_dir=branch_output_dir,
            subset_prefixes=subset_prefixes,
            explicit_functionals=[functional],
            checkpoint_path=experiment.manifest.checkpoint_copy,
            model_key=experiment.manifest.model_key,
            dispersion_correction=dispersion_correction,
        )
        job_ids = submit_jobs(
            functional,
            jobs_root=branch_jobs_dir,
            subset_prefixes=subset_prefixes,
            dispersion_correction=dispersion_correction,
        )
        experiment.set_branch_status(
            branch_name,
            "running",
            message="WTMAD jobs submitted.",
            job_ids=job_ids,
        )

        if wait:
            wait_for_slurm_jobs(job_ids)
            metrics, analysis_artifacts = _run_interface_analysis(
                experiment,
                dispersion_correction,
            )
            artifacts.extend(analysis_artifacts)
        else:
            metrics = {}
            artifacts = [str(branch_output_dir)]

        branch_report.write_text(
            json.dumps(
                {
                    "functional": functional,
                    "dispersion_correction": dispersion_correction,
                    "job_ids": job_ids,
                    "subset_prefixes": subset_prefixes,
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
                message="WTMAD branch finished.",
                job_ids=job_ids,
                artifacts=artifacts,
                metrics=metrics,
            )
        else:
            experiment.set_branch_status(
                branch_name,
                "running",
                message="WTMAD jobs submitted; waiting skipped.",
                job_ids=job_ids,
                artifacts=artifacts,
                metrics=metrics,
            )
    except Exception as exc:
        branch_report.write_text(
            json.dumps(
                {
                    "functional": functional,
                    "dispersion_correction": dispersion_correction,
                    "job_ids": job_ids,
                    "metrics": {},
                    "error": str(exc),
                    "output_dir": str(experiment.branch_output_dir(branch_name)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts.append(str(branch_report))
        experiment.set_branch_status(
            branch_name,
            "failed",
            message=str(exc),
            job_ids=job_ids,
            artifacts=artifacts,
        )
