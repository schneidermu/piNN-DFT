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


def run_sbatch(slurm_path: Path) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", str(slurm_path)],
        check=True,
        cwd=TEST_MODELS_ROOT,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


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
            return
        time.sleep(poll_interval_seconds)
