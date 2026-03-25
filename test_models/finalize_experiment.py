from __future__ import annotations

from argparse import ArgumentParser

from branches.avrane import finalize_avrane_branch
from branches.wtmad import finalize_wtmad_branch
from experiment import load_experiment
from reporting import write_reports


def main() -> None:
    parser = ArgumentParser(description="Finalize a staged piNN-DFT experiment.")
    parser.add_argument("--Manifest", required=True, help="Path to experiment manifest.json")
    args = parser.parse_args()

    experiment = load_experiment(args.Manifest)

    if experiment.manifest.branches["wtmad"].status == "running":
        try:
            finalize_wtmad_branch(experiment)
        except Exception as exc:
            branch = experiment.manifest.branches["wtmad"]
            experiment.set_branch_status(
                "wtmad",
                "failed",
                message=str(exc),
                job_ids=branch.job_ids,
                artifacts=branch.artifacts,
                metrics=branch.metrics,
            )
    if experiment.manifest.branches["avrane"].status == "running":
        try:
            finalize_avrane_branch(experiment, include_atoms=experiment.manifest.include_atoms)
        except Exception as exc:
            branch = experiment.manifest.branches["avrane"]
            experiment.set_branch_status(
                "avrane",
                "failed",
                message=str(exc),
                job_ids=branch.job_ids,
                artifacts=branch.artifacts,
                metrics=branch.metrics,
            )

    write_reports(experiment)
    experiment.write_manifest()


if __name__ == "__main__":
    main()
