from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable


TEST_MODELS_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = TEST_MODELS_ROOT / "Results"
GIF_DIR = TEST_MODELS_ROOT / "GIF"
MOLDEN_DIR = TEST_MODELS_ROOT / "molden"
GBS_PATH = TEST_MODELS_ROOT / "aug-cc-pwcv5z.gbs"
LOGS_DIR = TEST_MODELS_ROOT / "logs"
ENERGY_LOG_DIR = LOGS_DIR / "log_energy"
MOLDEN_LOG_DIR = LOGS_DIR / "log_molden"
ATOM_LOG_DIR = LOGS_DIR / "log_atoms"
GENERATED_JOBS_DIR = TEST_MODELS_ROOT / "generated_jobs"
EXPERIMENTS_DIR = TEST_MODELS_ROOT / "experiments"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_runtime_directories() -> None:
    for path in (
        RESULTS_DIR,
        ENERGY_LOG_DIR,
        MOLDEN_LOG_DIR,
        ATOM_LOG_DIR,
        GENERATED_JOBS_DIR,
        EXPERIMENTS_DIR,
    ):
        ensure_dir(path)


def iter_gif_system_names() -> list[str]:
    if not GIF_DIR.exists():
        return []

    system_names = []
    for entry in sorted(GIF_DIR.iterdir()):
        if entry.is_dir():
            gif_file = entry / f"{entry.name}.gif_"
            if gif_file.exists():
                system_names.append(entry.name)
        elif entry.is_file() and entry.suffix == ".gif_":
            system_names.append(entry.stem)
    return system_names


def normalize_gif_layout() -> list[str]:
    system_names = []
    for system_name in iter_gif_system_names():
        system_dir = ensure_dir(GIF_DIR / system_name)
        legacy_file = GIF_DIR / f"{system_name}.gif_"
        target_file = system_dir / f"{system_name}.gif_"
        if legacy_file.exists() and legacy_file != target_file:
            legacy_file.rename(target_file)
        system_names.append(system_name)
    return system_names


def run_sbatch(slurm_path: Path, extra_args: list[str] | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if extra_args:
        command.extend(extra_args)
    command.append(str(slurm_path))
    result = subprocess.run(
        command,
        check=True,
        cwd=TEST_MODELS_ROOT,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def _exit_code_failed(exit_code: str) -> bool:
    status = exit_code.split(":", 1)[0]
    return bool(status and status != "0")


def _assert_slurm_jobs_succeeded(job_ids: list[str]) -> None:
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,ExitCode",
        ],
        cwd=TEST_MODELS_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sacct status check failed")
    if not result.stdout.strip():
        return

    failed_rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            continue
        job_id, state, exit_code = fields[:3]
        normalized_state = state.split()[0]
        if (
            normalized_state in FAILED_SLURM_STATES
            or _exit_code_failed(exit_code)
        ):
            failed_rows.append(
                f"{job_id}: state={state}, exit_code={exit_code}"
            )

    if failed_rows:
        details = "; ".join(failed_rows[:10])
        if len(failed_rows) > 10:
            details += f"; ... {len(failed_rows) - 10} more"
        raise RuntimeError(f"SLURM jobs did not complete successfully: {details}")


def wait_for_slurm_jobs(job_ids: Iterable[str], poll_interval_seconds: int = 15) -> None:
    normalized = [job_id for job_id in job_ids if job_id]
    if not normalized:
        return

    while True:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(normalized)],
            cwd=TEST_MODELS_ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "squeue polling failed")
        if not result.stdout.strip():
            _assert_slurm_jobs_succeeded(normalized)
            return
        time.sleep(poll_interval_seconds)
