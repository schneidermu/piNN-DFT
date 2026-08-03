from __future__ import annotations

import argparse
import json
from pathlib import Path

from avrane_reduce import _load_iodens, run_avrane_reduction
from common import ensure_dir
from experiment import (
    BranchStatus,
    Experiment,
    ExperimentManifest,
    ExperimentPaths,
    build_generated_functional_name,
    infer_model_key,
    load_experiment,
)
from reporting import write_reports


def _recover_manifest(experiment_root: Path) -> Experiment:
    manifest_path = experiment_root / "manifest.json"
    try:
        return load_experiment(str(manifest_path))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass

    checkpoints = sorted((experiment_root / "input").glob("*.pt"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint under {experiment_root / 'input'}, "
            f"found {len(checkpoints)}."
        )

    checkpoint = checkpoints[0]
    experiment_slug = experiment_root.name.split("_", maxsplit=1)[-1]
    experiment_name = experiment_slug
    paths = ExperimentPaths(
        root=str(experiment_root),
        input=str(ensure_dir(experiment_root / "input")),
        jobs=str(ensure_dir(experiment_root / "jobs")),
        outputs=str(ensure_dir(experiment_root / "outputs")),
        reports=str(ensure_dir(experiment_root / "reports")),
        logs=str(ensure_dir(experiment_root / "logs")),
    )
    manifest = ExperimentManifest(
        experiment_name=experiment_name,
        experiment_slug=experiment_slug,
        created_at="recovered",
        checkpoint_source=str(checkpoint),
        checkpoint_copy=str(checkpoint),
        generated_functional_name=build_generated_functional_name(experiment_name, checkpoint),
        model_key=infer_model_key(checkpoint),
        wtmad_dispersion_correction="pbe-d3bj",
        smoke=False,
        include_atoms=False,
        smoke_wtmad_databases=[],
        smoke_avrane_molecules=[],
        reference_paths={},
        paths=paths,
        branches={
            "wtmad": BranchStatus(name="wtmad"),
            "avrane": BranchStatus(name="avrane"),
        },
    )
    experiment = Experiment(manifest_path, manifest)
    experiment.write_manifest()
    return experiment


def _calc_directory(experiment: Experiment) -> tuple[str, Path]:
    calc_root = experiment.root / "outputs" / "avrane" / "den_mol_or" / "calc"
    candidates = sorted(path for path in calc_root.iterdir() if path.is_dir()) if calc_root.exists() else []
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one functional directory under {calc_root}, found {len(candidates)}."
        )
    return candidates[0].name, candidates[0]


def _valid_systems(calc_dir: Path) -> tuple[list[str], dict[str, str]]:
    iodens = _load_iodens()
    valid: list[str] = []
    invalid: dict[str, str] = {}
    for molecule_dir in sorted(path for path in calc_dir.iterdir() if path.is_dir()):
        required = [molecule_dir / name for name in ("rho", "grad", "lapl")]
        if not all(path.is_file() for path in required):
            invalid[molecule_dir.name] = "missing rho, grad, or lapl"
            continue
        try:
            iodens.read_mwfn(str(molecule_dir) + "/")
        except Exception as exc:
            invalid[molecule_dir.name] = str(exc)
            continue
        valid.append(molecule_dir.name)
    return valid, invalid


def _write_avrane_report(experiment: Experiment, *, systems: list[str], invalid: dict[str, str], metrics: dict) -> None:
    report_path = experiment.reports_dir / "avrane.json"
    report_path.write_text(
        json.dumps(
            {
                "functional": experiment.manifest.generated_functional_name,
                "partial": True,
                "systems": systems,
                "invalid_systems": invalid,
                "metrics": metrics,
                "output_dir": str(experiment.branch_output_dir("avrane")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover partial avRANE results from existing experiment grid files."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
    )
    args = parser.parse_args()

    for manifest_path in sorted(args.experiments_root.glob("*/manifest.json")):
        root = manifest_path.parent
        if not ("h9-explore-" in root.name or "h9-exploit-" in root.name):
            continue
        try:
            experiment = _recover_manifest(root)
            functional, calc_dir = _calc_directory(experiment)
            experiment.manifest.generated_functional_name = functional
            valid, invalid = _valid_systems(calc_dir)
            print(f"\n{experiment.manifest.experiment_slug}")
            print(f"  valid systems ({len(valid)}): {', '.join(valid) or 'none'}")
            if invalid:
                print(f"  invalid systems ({len(invalid)}): {', '.join(sorted(invalid))}")
            if not valid:
                experiment.write_manifest()
                continue

            metrics, artifacts, reference_paths = run_avrane_reduction(
                experiment.root,
                functional,
                systems=valid,
            )
            branch = experiment.manifest.branches["avrane"]
            artifacts = list(dict.fromkeys([*branch.artifacts, *artifacts]))
            experiment.set_reference_paths(reference_paths)
            experiment.set_branch_status(
                "avrane",
                "complete",
                message=f"Partial avRANE from {len(valid)} validated systems.",
                job_ids=branch.job_ids,
                artifacts=artifacts,
                metrics=metrics,
            )
            _write_avrane_report(
                experiment,
                systems=valid,
                invalid=invalid,
                metrics=metrics,
            )
            write_reports(experiment)
            experiment.write_manifest()
            print(f"  partial avRANE: {metrics['selected_summary_metric']:.9f}")
            print(f"  manifest: {experiment.manifest_path}")
        except Exception as exc:
            print(f"\n{root.name}\n  recovery failed: {exc}")


if __name__ == "__main__":
    main()
