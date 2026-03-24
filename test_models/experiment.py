from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from common import EXPERIMENTS_DIR, ensure_dir, ensure_runtime_directories

SMOKE_WTMAD_DATABASES = ("BH76-5", "SIE4X4")
SMOKE_AVRANE_MOLECULES = ("H2", "N2")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "experiment"


def infer_model_key(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem.lower()
    path_str = str(checkpoint_path).lower()
    if "xalpha" in stem:
        return "NN_XALPHA"
    if (
        "pbe-l" in stem
        or "pbel" in stem
        or "nn_l" in stem
        or "pbe-l" in path_str
        or "nn_pbe-l" in path_str
        or "nn_l" in path_str
    ):
        return "NN_PBE-L"
    if "star_star" in stem or "doublestar" in stem or "double_star" in stem:
        return "NN_PBE_star_star"
    if re.search(r"(^|[_-])star($|[_-])", stem):
        return "NN_PBE_star"
    return "NN_PBE"


def build_generated_functional_name(experiment_name: str, checkpoint_path: Path) -> str:
    experiment_slug = slugify(experiment_name)
    checkpoint_slug = slugify(checkpoint_path.stem)
    return f"EXP_{experiment_slug}_{checkpoint_slug}"


@dataclass
class ExperimentPaths:
    root: str
    input: str
    jobs: str
    outputs: str
    reports: str
    logs: str


@dataclass
class BranchStatus:
    name: str
    status: str = "pending"
    message: str | None = None
    job_ids: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class ExperimentManifest:
    experiment_name: str
    experiment_slug: str
    created_at: str
    checkpoint_source: str
    checkpoint_copy: str
    generated_functional_name: str
    model_key: str
    smoke: bool
    smoke_wtmad_databases: list[str]
    smoke_avrane_molecules: list[str]
    reference_paths: dict[str, str]
    paths: ExperimentPaths
    branches: dict[str, BranchStatus]


class Experiment:
    def __init__(self, manifest_path: Path, manifest: ExperimentManifest):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self._lock = Lock()

    @property
    def root(self) -> Path:
        return Path(self.manifest.paths.root)

    @property
    def input_dir(self) -> Path:
        return Path(self.manifest.paths.input)

    @property
    def jobs_dir(self) -> Path:
        return Path(self.manifest.paths.jobs)

    @property
    def outputs_dir(self) -> Path:
        return Path(self.manifest.paths.outputs)

    @property
    def reports_dir(self) -> Path:
        return Path(self.manifest.paths.reports)

    @property
    def logs_dir(self) -> Path:
        return Path(self.manifest.paths.logs)

    def write_manifest(self) -> None:
        with self._lock:
            payload = asdict(self.manifest)
            self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def branch_output_dir(self, branch_name: str) -> Path:
        return ensure_dir(self.outputs_dir / branch_name)

    def branch_jobs_dir(self, branch_name: str) -> Path:
        return ensure_dir(self.jobs_dir / branch_name)

    def branch_logs_dir(self, branch_name: str) -> Path:
        return ensure_dir(self.logs_dir / branch_name)

    def set_branch_status(
        self,
        branch_name: str,
        status: str,
        *,
        message: str | None = None,
        job_ids: list[str] | None = None,
        artifacts: list[str] | None = None,
        metrics: dict | None = None,
    ) -> None:
        with self._lock:
            branch = self.manifest.branches[branch_name]
            branch.status = status
            if message is not None:
                branch.message = message
            if job_ids is not None:
                branch.job_ids = job_ids
            if artifacts is not None:
                branch.artifacts = artifacts
            if metrics is not None:
                branch.metrics = metrics
            payload = asdict(self.manifest)
            self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_reference_paths(self, reference_paths: dict[str, str]) -> None:
        with self._lock:
            self.manifest.reference_paths = reference_paths
            payload = asdict(self.manifest)
            self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def overall_status(self) -> str:
        statuses = [branch.status for branch in self.manifest.branches.values()]
        if any(status == "failed" for status in statuses):
            return "failed"
        if all(status == "complete" for status in statuses):
            return "complete"
        if any(status == "running" for status in statuses):
            return "running"
        return "pending"


def create_experiment(checkpoint: str, experiment_name: str, smoke: bool) -> Experiment:
    ensure_runtime_directories()

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_slug = slugify(experiment_name)
    root_dir = ensure_dir(EXPERIMENTS_DIR / f"{timestamp}_{experiment_slug}")
    input_dir = ensure_dir(root_dir / "input")
    jobs_dir = ensure_dir(root_dir / "jobs")
    outputs_dir = ensure_dir(root_dir / "outputs")
    reports_dir = ensure_dir(root_dir / "reports")
    logs_dir = ensure_dir(root_dir / "logs")

    checkpoint_copy = input_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, checkpoint_copy)

    manifest = ExperimentManifest(
        experiment_name=experiment_name,
        experiment_slug=experiment_slug,
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_source=str(checkpoint_path),
        checkpoint_copy=str(checkpoint_copy),
        generated_functional_name=build_generated_functional_name(
            experiment_name, checkpoint_path
        ),
        model_key=infer_model_key(checkpoint_path),
        smoke=smoke,
        smoke_wtmad_databases=list(SMOKE_WTMAD_DATABASES),
        smoke_avrane_molecules=list(SMOKE_AVRANE_MOLECULES),
        reference_paths={},
        paths=ExperimentPaths(
            root=str(root_dir),
            input=str(input_dir),
            jobs=str(jobs_dir),
            outputs=str(outputs_dir),
            reports=str(reports_dir),
            logs=str(logs_dir),
        ),
        branches={
            "wtmad": BranchStatus(name="wtmad"),
            "avrane": BranchStatus(name="avrane"),
        },
    )

    experiment = Experiment(root_dir / "manifest.json", manifest)
    experiment.write_manifest()
    return experiment


def load_experiment(manifest_path: str) -> Experiment:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = ExperimentManifest(
        experiment_name=payload["experiment_name"],
        experiment_slug=payload["experiment_slug"],
        created_at=payload["created_at"],
        checkpoint_source=payload["checkpoint_source"],
        checkpoint_copy=payload["checkpoint_copy"],
        generated_functional_name=payload["generated_functional_name"],
        model_key=payload["model_key"],
        smoke=payload["smoke"],
        smoke_wtmad_databases=payload["smoke_wtmad_databases"],
        smoke_avrane_molecules=payload["smoke_avrane_molecules"],
        reference_paths=payload.get("reference_paths", {}),
        paths=ExperimentPaths(**payload["paths"]),
        branches={
            name: BranchStatus(**branch_payload)
            for name, branch_payload in payload["branches"].items()
        },
    )
    return Experiment(path, manifest)
