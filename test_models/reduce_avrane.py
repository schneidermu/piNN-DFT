from __future__ import annotations

import json
from argparse import ArgumentParser

from avrane_reduce import run_avrane_reduction
from experiment import load_experiment
from reporting import write_reports


def main() -> None:
    parser = ArgumentParser(
        description="Reduce existing avRANE density grids without rerunning SCF jobs."
    )
    parser.add_argument("--Manifest", required=True, help="Path to experiment manifest.json")
    parser.add_argument(
        "--Systems",
        default="",
        help="Optional comma-separated molecule subset. Defaults to all LDA reference systems.",
    )
    args = parser.parse_args()

    experiment = load_experiment(args.Manifest)
    branch = experiment.manifest.branches["avrane"]
    systems = [value for value in args.Systems.split(",") if value] or None

    metrics, reduction_artifacts, reference_paths = run_avrane_reduction(
        experiment.root,
        experiment.manifest.generated_functional_name,
        systems=systems,
    )
    artifacts = list(dict.fromkeys([*branch.artifacts, *reduction_artifacts]))
    experiment.set_reference_paths(reference_paths)
    experiment.set_branch_status(
        "avrane",
        "complete",
        message="avRANE reduction finished from existing density grids.",
        job_ids=branch.job_ids,
        artifacts=artifacts,
        metrics=metrics,
    )

    branch_report = experiment.reports_dir / "avrane.json"
    branch_report.write_text(
        json.dumps(
            {
                "functional": experiment.manifest.generated_functional_name,
                "job_ids": branch.job_ids,
                "subset_molecules": systems,
                "include_atoms": experiment.manifest.include_atoms,
                "reference_paths": reference_paths,
                "metrics": metrics,
                "output_dir": str(experiment.branch_output_dir("avrane")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_reports(experiment)
    experiment.write_manifest()

    print(f"avRANE reduction complete: {experiment.reports_dir / 'summary.md'}")
    print(f"Metrics: {experiment.reports_dir / 'avrane_metrics.json'}")


if __name__ == "__main__":
    main()
