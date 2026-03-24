from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from branches import run_avrane_branch, run_wtmad_branch
from experiment import create_experiment
from reporting import write_reports


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
        "--no-wait",
        action="store_true",
        help="Submit branch jobs without waiting for completion",
    )
    args = parser.parse_args()

    experiment = create_experiment(
        checkpoint=args.checkpoint,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
    )

    wait_for_completion = not args.no_wait
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_wtmad_branch, experiment, wait_for_completion),
            executor.submit(run_avrane_branch, experiment, wait_for_completion),
        ]
        for future in as_completed(futures):
            future.result()

    write_reports(experiment)
    experiment.write_manifest()

    print(f"Experiment created at: {experiment.root}")
    print(f"Overall status: {experiment.overall_status()}")
    print(f"Summary: {experiment.reports_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
