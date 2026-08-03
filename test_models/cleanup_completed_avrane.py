from __future__ import annotations

import argparse
import json
from pathlib import Path

from avrane_reduce import cleanup_experiment_avrane_artifacts
from experiment import load_experiment
from reporting import write_reports


def _tree_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete persisted avRANE density artifacts for completed experiments."
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion. Without this flag, print a dry-run inventory only.",
    )
    args = parser.parse_args()

    candidates = []
    for manifest_path in sorted(args.experiments_root.glob("*/manifest.json")):
        try:
            experiment = load_experiment(str(manifest_path))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        branch = experiment.manifest.branches.get("avrane")
        metrics_path = experiment.reports_dir / "avrane_metrics.json"
        artifact_root = experiment.root / "outputs" / "avrane" / "den_mol_or"
        if branch is None or branch.status != "complete" or not metrics_path.is_file():
            continue
        if artifact_root.exists():
            candidates.append((experiment, artifact_root, _tree_size(artifact_root)))

    total_bytes = sum(size for _, _, size in candidates)
    for experiment, artifact_root, size in candidates:
        print(f"{size / 1024**3:.2f} GiB  {experiment.manifest.experiment_slug}")
        print(f"  {artifact_root}")
    print(f"Candidates: {len(candidates)}, total: {total_bytes / 1024**3:.2f} GiB")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete these artifacts.")
        return

    for experiment, _, _ in candidates:
        cleanup_experiment_avrane_artifacts(experiment.root)
        branch = experiment.manifest.branches["avrane"]
        branch.artifacts = [
            artifact
            for artifact in branch.artifacts
            if "/outputs/avrane/den_mol_or/" not in artifact.replace("\\", "/")
        ]
        branch.message = "avRANE metrics persisted; raw density artifacts removed."
        experiment.write_manifest()
        write_reports(experiment)


if __name__ == "__main__":
    main()