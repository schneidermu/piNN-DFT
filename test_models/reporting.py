from __future__ import annotations

import json
from dataclasses import asdict

from experiment import Experiment


def _format_metric_table(title: str, metrics: dict) -> list[str]:
    if not metrics:
        return [f"#### {title}", "", "_No metrics recorded._", ""]

    lines = [f"#### {title}", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"| {key} | `{json.dumps(value, sort_keys=True)}` |")
        elif isinstance(value, list):
            lines.append(f"| {key} | `{json.dumps(value)}` |")
        else:
            lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return lines


def write_reports(experiment: Experiment) -> None:
    summary_json = experiment.reports_dir / "summary.json"
    summary_md = experiment.reports_dir / "summary.md"

    payload = {
        "experiment_name": experiment.manifest.experiment_name,
        "status": experiment.overall_status(),
        "generated_functional_name": experiment.manifest.generated_functional_name,
        "checkpoint_source": experiment.manifest.checkpoint_source,
        "checkpoint_copy": experiment.manifest.checkpoint_copy,
        "smoke": experiment.manifest.smoke,
        "include_atoms": experiment.manifest.include_atoms,
        "reference_paths": experiment.manifest.reference_paths,
        "paths": asdict(experiment.manifest.paths),
        "branches": {
            name: asdict(branch)
            for name, branch in experiment.manifest.branches.items()
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Experiment Summary: {experiment.manifest.experiment_name}",
        "",
        f"- Status: `{experiment.overall_status()}`",
        f"- Generated functional: `{experiment.manifest.generated_functional_name}`",
        f"- Checkpoint source: `{experiment.manifest.checkpoint_source}`",
        f"- Checkpoint copy: `{experiment.manifest.checkpoint_copy}`",
        f"- Smoke mode: `{experiment.manifest.smoke}`",
        f"- Include atoms: `{experiment.manifest.include_atoms}`",
        f"- Reference paths: `{json.dumps(experiment.manifest.reference_paths, sort_keys=True)}`",
        "",
        "## WTMAD-2",
    ]
    wtmad_branch = experiment.manifest.branches.get("wtmad")
    if wtmad_branch:
        lines.extend(
            [
                f"- Status: `{wtmad_branch.status}`",
                f"- Message: {wtmad_branch.message or 'n/a'}",
                f"- Job IDs: {', '.join(wtmad_branch.job_ids) if wtmad_branch.job_ids else 'n/a'}",
                "",
            ]
        )
        lines.extend(_format_metric_table("WTMAD Metrics", wtmad_branch.metrics))
        lines.append("Artifacts:")
        if wtmad_branch.artifacts:
            for artifact in wtmad_branch.artifacts:
                lines.append(f"- `{artifact}`")
        else:
            lines.append("- none")
        lines.append("")

    lines.append("## avRANE")
    avrane_branch = experiment.manifest.branches.get("avrane")
    if avrane_branch:
        lines.extend(
            [
                f"- Status: `{avrane_branch.status}`",
                f"- Message: {avrane_branch.message or 'n/a'}",
                f"- Job IDs: {', '.join(avrane_branch.job_ids) if avrane_branch.job_ids else 'n/a'}",
                "",
            ]
        )
        lines.extend(_format_metric_table("avRANE Metrics", avrane_branch.metrics))
        lines.append("Artifacts:")
        if avrane_branch.artifacts:
            for artifact in avrane_branch.artifacts:
                lines.append(f"- `{artifact}`")
        else:
            lines.append("- none")
        lines.append("")

    lines.append("## Branches")
    for name, branch in experiment.manifest.branches.items():
        lines.extend(
            [
                f"### {name}",
                f"- Status: `{branch.status}`",
                f"- Message: {branch.message or 'n/a'}",
                f"- Job IDs: {', '.join(branch.job_ids) if branch.job_ids else 'n/a'}",
                "- Artifacts:",
            ]
        )
        if branch.artifacts:
            for artifact in branch.artifacts:
                lines.append(f"  - `{artifact}`")
        else:
            lines.append("  - none")
        lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
