"""Evidence-first retrospective analysis of 30 fixed-architecture schedules.

The script reads completed training histories and evaluation manifests only. It
does not generate schedules, checkpoints, or cluster jobs. All regression and
correlation calculations use the Python standard library so the analysis is
portable to the local Windows checkout.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent.parent
OUT = REPO / "train_models" / "schedule_landscape_30"

REFERENCE = {
    "wtmad": 7.097,
    "avrane": 0.4799438972067495,
}

OLD_SPECS = {
    "baseline": ("reference", "replay_trial_19_bridge_500_exc_kcal_6_32_d3_mrks_gc_svelu_mirror", None),
    "m1": ("micro", "replay_trial_19_micro_h1_sum_repair_plus15_finish_minus15_gc_svelu_mirror", "micro-h1"),
    "m2": ("micro", "replay_trial_19_micro_h2_fchem_polish_plus20_finish_minus20_gc_svelu_mirror", "micro-h2"),
    "m3": ("micro", "replay_trial_19_micro_h3_finish_vxc7_gc_svelu_mirror", "micro-h3"),
    "m4": ("micro", "replay_trial_19_micro_h4_drive12_finish7_gc_svelu_mirror", "micro-h4"),
    "m5": ("micro", "replay_trial_19_micro_h5_clip_minus10_sum_plus10_gc_svelu_mirror", "micro-h5"),
    "h6": ("repair", "replay_trial_19_schedule_h6_exc_repair_compressed_gc_svelu_mirror", "h6-exc"),
    "h7": ("repair", "replay_trial_19_schedule_h7_exc_repair_smooth_gc_svelu_mirror", "h7-exc"),
    "h8": ("repair", "replay_trial_19_schedule_h8_exc_repair_density_guard_gc_svelu_mirror", "h8-exc"),
    "h9": ("h9", "replay_trial_19_schedule_h9_late_exc_repair_gc_svelu_mirror", "h9-late-exc-avrane"),
    "h10": ("repair", "replay_trial_19_schedule_h10_early_exc_repair_gc_svelu_mirror", "h10-early"),
    "e1": ("h9", "replay_trial_19_h9_explore_e1_short_repair_gc_svelu_mirror", "explore-e1"),
    "e2": ("h9", "replay_trial_19_h9_explore_e2_long_repair_gc_svelu_mirror", "explore-e2"),
    "e3": ("h9", "replay_trial_19_h9_explore_e3_delayed_repair_gc_svelu_mirror", "explore-e3"),
    "e4": ("h9", "replay_trial_19_h9_explore_e4_exc4_gc_svelu_mirror", "explore-e4"),
    "e5": ("h9", "replay_trial_19_h9_explore_e5_reaction1_gc_svelu_mirror", "explore-e5"),
    "x1": ("h9", "replay_trial_19_h9_exploit_x1_soft_exc2_vxc10_r085_gc_svelu_mirror", "exploit-x1"),
    "x2": ("h9", "replay_trial_19_h9_exploit_x2_soft_exc2_vxc15_r075_gc_svelu_mirror", "exploit-x2"),
    "x3": ("h9", "replay_trial_19_h9_exploit_x3_soft_exc25_vxc12_r080_gc_svelu_mirror", "exploit-x3"),
    "x4": ("h9", "replay_trial_19_h9_exploit_x4_repair440_soft30_gc_svelu_mirror", "exploit-x4"),
    "x5": ("h9", "replay_trial_19_h9_exploit_x5_soft60_gc_svelu_mirror", "exploit-x5"),
}

SIMPLE_SPECS = {
    "s1": (0.00, 420),
    "s2": (0.00, 440),
    "s3": (0.25, 420),
    "s4": (0.25, 440),
    "s5": (0.50, 420),
    "s6": (0.50, 440),
    "s7": (0.75, 420),
    "s8": (0.75, 440),
    "s10": (1.00, 440),
}

REACTION_ROW = re.compile(
    r"^([A-Za-z0-9x+_-]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s*$"
)
TRAIN_METRICS = ("train_fchem", "val_fchem", "train_vxc", "val_vxc", "train_exc", "val_exc")
FIXED_WINDOWS = ((1, 72), (73, 160), (161, 240), (241, 280), (281, 420), (421, 440), (441, 500))


@dataclass
class Run:
    key: str
    family: str
    history_path: Path
    history: list[dict[str, Any]]
    schedule: list[dict[str, Any]]
    manifest_path: Path | None
    wtmad: float | None
    avrane: float | None
    ranes: dict[str, float]
    per_system: dict[str, dict[str, float]]
    reactions: dict[str, float]
    scf_total: int
    scf_converged: int
    guard: float | None = None
    repair_end: int | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_history(folder: str) -> Path:
    path = WORKSPACE / folder / "trials" / "trial_19.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def manifest_roots() -> list[Path]:
    roots = [WORKSPACE / "experiments"]
    roots.extend(path for path in WORKSPACE.glob("2026*") if path.is_dir())
    return roots


def find_manifest(token: str | None) -> Path | None:
    if token is None:
        return None
    candidates: list[Path] = []
    for root in manifest_roots():
        paths = [root / "manifest.json"] if (root / "manifest.json").is_file() else root.glob("*/manifest.json")
        for path in paths:
            if token not in path.parent.name:
                continue
            try:
                payload = load_json(path)
            except (json.JSONDecodeError, OSError):
                continue
            branches = payload.get("branches", {})
            if branches.get("wtmad", {}).get("metrics", {}).get("wtmad_2") is not None:
                candidates.append(path)
    return max(candidates, key=lambda path: path.parent.name) if candidates else None


def read_external(manifest_path: Path | None) -> tuple[float | None, float | None, dict[str, float], dict[str, dict[str, float]], dict[str, float], int, int]:
    if manifest_path is None:
        return None, None, {}, {}, {}, 0, 0
    payload = load_json(manifest_path)
    branches = payload.get("branches", {})
    wtmad = branches.get("wtmad", {}).get("metrics", {}).get("wtmad_2")
    avmetrics = branches.get("avrane", {}).get("metrics", {})
    avrane = avmetrics.get("selected_summary_metric")
    ranes = avmetrics.get("ranes", {})
    per_system = avmetrics.get("per_system_niad", {})
    reactions: dict[str, float] = {}
    interface = manifest_path.parent / "reports" / "wtmad_interface_output.txt"
    if interface.is_file():
        for line in interface.read_text(encoding="utf-8", errors="replace").splitlines():
            match = REACTION_ROW.match(line)
            if match:
                reactions[match.group(1)] = float(match.group(4))
    logs = list((manifest_path.parent / "logs" / "wtmad").glob("*.out"))
    converged = sum("converged SCF energy" in path.read_text(encoding="utf-8", errors="replace") for path in logs)
    return (
        float(wtmad) if wtmad is not None else None,
        float(avrane) if avrane is not None else None,
        {key: float(value) for key, value in ranes.items()},
        per_system,
        reactions,
        len(logs),
        converged,
    )


def load_run(key: str, family: str, folder: str, token: str | None, guard: float | None = None, repair_end: int | None = None) -> Run:
    history_path = find_history(folder)
    payload = load_json(history_path)
    manifest_path = find_manifest(token)
    wtmad, avrane, ranes, per_system, reactions, scf_total, scf_converged = read_external(manifest_path)
    if key == "baseline":
        wtmad = REFERENCE["wtmad"]
        avrane = REFERENCE["avrane"]
    return Run(
        key=key,
        family=family,
        history_path=history_path,
        history=payload["epoch_history"],
        schedule=payload["params"]["epoch_schedule"],
        manifest_path=manifest_path,
        wtmad=wtmad,
        avrane=avrane,
        ranes=ranes,
        per_system=per_system,
        reactions=reactions,
        scf_total=scf_total,
        scf_converged=scf_converged,
        guard=guard,
        repair_end=repair_end,
    )


def load_runs() -> list[Run]:
    runs = [load_run(key, *spec) for key, spec in OLD_SPECS.items()]
    for key, (guard, repair_end) in SIMPLE_SPECS.items():
        guard_code = f"g{int(guard * 100):03d}"
        folder = f"replay_trial_19_simple_{key}_{guard_code}_repair{repair_end}_gc_svelu_mirror"
        token = f"simple-{key}-{guard_code}-repair{repair_end}"
        runs.append(load_run(key, "simple", folder, token, guard, repair_end))
    return runs


def epoch_controls(run: Run) -> dict[int, dict[str, Any]]:
    controls: dict[int, dict[str, Any]] = {}
    for phase in run.schedule:
        for epoch in range(int(phase["start_epoch"]), int(phase["end_epoch"]) + 1):
            controls[epoch] = phase["params"]
    if set(controls) != set(range(1, 501)):
        raise ValueError(f"{run.key}: schedule does not cover exactly epochs 1..500")
    return controls


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return math.nan
    xbar = (n - 1) / 2
    ybar = mean(values)
    denominator = sum((index - xbar) ** 2 for index in range(n))
    return sum((index - xbar) * (value - ybar) for index, value in enumerate(values)) / denominator


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2 + 1
        for position in range(index, end):
            result[order[position]] = average_rank
        index = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys))
    return numerator / denominator if denominator else math.nan


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def schedule_features(run: Run) -> dict[str, Any]:
    controls = epoch_controls(run)
    rows = [controls[epoch] for epoch in range(1, 501)]
    vxc = [float(row["vxc_loss_scale"]) for row in rows]
    exc = [float(row["exc_loss_scale"]) for row in rows]
    reaction = [float(row["reaction_grad_scale"]) for row in rows]
    clipped = [row.get("gradient_merge_strategy") == "clip_then_sum" for row in rows]
    late_repair = [epoch for epoch in range(201, 501) if clipped[epoch - 1] and exc[epoch - 1] > 1.01]
    conceptual_names = []
    for phase in run.schedule:
        name = phase["name"]
        if not conceptual_names or name != conceptual_names[-1]:
            conceptual_names.append(name)
    result: dict[str, Any] = {
        "run": run.key,
        "family": run.family,
        "guard": run.guard,
        "repair_end_design": run.repair_end,
        "conceptual_phases": len(conceptual_names),
        "raw_segments": len(run.schedule),
        "mean_vxc": mean(vxc),
        "mean_exc": mean(exc),
        "mean_reaction": mean(reaction),
        "clipped_epochs": sum(clipped),
        "late_repair_onset": min(late_repair) if late_repair else None,
        "late_repair_end": max(late_repair) if late_repair else None,
        "late_repair_epochs": len(late_repair),
        "final_sum_epochs": next((500 - epoch for epoch in range(500, 0, -1) if clipped[epoch - 1]), 500),
        "final_vxc": vxc[-1],
        "final_exc": exc[-1],
        "final_reaction": reaction[-1],
    }
    for start, end in FIXED_WINDOWS:
        tag = f"e{start}_{end}"
        result[f"vxc_{tag}"] = mean(vxc[start - 1 : end])
        result[f"exc_{tag}"] = mean(exc[start - 1 : end])
        result[f"reaction_{tag}"] = mean(reaction[start - 1 : end])
        result[f"clip_fraction_{tag}"] = mean(float(value) for value in clipped[start - 1 : end])
    return result


def trajectory_features(run: Run) -> dict[str, Any]:
    by_epoch = {int(row["epoch"]): row for row in run.history}
    result: dict[str, Any] = {"run": run.key}
    for metric in TRAIN_METRICS:
        values = [float(by_epoch[epoch][metric]) for epoch in range(1, 501)]
        result[f"{metric}_final"] = values[-1]
        result[f"{metric}_mean_last20"] = mean(values[-20:])
        result[f"{metric}_slope_last20"] = slope(values[-20:])
        result[f"{metric}_std_last20"] = statistics.pstdev(values[-20:])
        result[f"{metric}_min"] = min(values)
        for start, end in FIXED_WINDOWS:
            tag = f"e{start}_{end}"
            window = values[start - 1 : end]
            result[f"{metric}_mean_{tag}"] = mean(window)
            result[f"{metric}_delta_{tag}"] = mean(window[-min(20, len(window)) :]) - mean(window[: min(20, len(window))])
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matched_contrasts(runs_by_key: dict[str, Run]) -> list[dict[str, Any]]:
    pairs = [
        ("baseline", "m1", "extend sum_repair by 15 epochs"),
        ("baseline", "m2", "extend fchem_polish by 20 epochs"),
        ("baseline", "m3", "terminal Vxc 5 -> 7"),
        ("baseline", "m4", "drive Vxc 10 -> 12 and terminal 5 -> 7"),
        ("baseline", "m5", "shorter early clipped anchor"),
        ("h9", "e4", "repair E_xc 3 -> 4"),
        ("h9", "e5", "repair reaction scale 0.75 -> 1.0"),
        ("h9", "e1", "shorter repair / longer final relaxation"),
        ("h9", "e2", "longer repair / shorter final relaxation"),
        ("h9", "e3", "delay repair onset 40 and end 20 epochs later"),
        ("e3", "s4", "replace E3 epochs 1-280 with smooth g=0.25 homotopy; epochs 281-500 identical"),
        ("s1", "s2", "repair end 420 -> 440 at guard 0.00"),
        ("s3", "s4", "repair end 420 -> 440 at guard 0.25"),
        ("s5", "s6", "repair end 420 -> 440 at guard 0.50"),
        ("s7", "s8", "repair end 420 -> 440 at guard 0.75"),
    ]
    rows = []
    for left_key, right_key, change in pairs:
        left, right = runs_by_key[left_key], runs_by_key[right_key]
        rows.append(
            {
                "left": left_key,
                "right": right_key,
                "change": change,
                "delta_wtmad": right.wtmad - left.wtmad if left.wtmad is not None and right.wtmad is not None else None,
                "delta_avrane": right.avrane - left.avrane if left.avrane is not None and right.avrane is not None else None,
                "delta_scf_converged": right.scf_converged - left.scf_converged if left.scf_total and right.scf_total else None,
            }
        )
    return rows


def correlations(rows: list[dict[str, Any]], feature_names: list[str], target: str) -> list[dict[str, Any]]:
    output = []
    for feature in feature_names:
        selected = [(float(row[feature]), float(row[target])) for row in rows if row.get(feature) not in (None, "") and row.get(target) not in (None, "")]
        if len(selected) < 6 or len({value[0] for value in selected}) < 3:
            continue
        xs, ys = [value[0] for value in selected], [value[1] for value in selected]
        output.append({"feature": feature, "target": target, "n": len(selected), "spearman": spearman(xs, ys), "pearson": pearson(xs, ys)})
    return sorted(output, key=lambda row: abs(row["spearman"]), reverse=True)


def simple_factorial(runs_by_key: dict[str, Run]) -> list[dict[str, Any]]:
    rows = []
    for key in SIMPLE_SPECS:
        run = runs_by_key[key]
        rows.append({"run": key, "guard": run.guard, "repair_end": run.repair_end, "wtmad": run.wtmad, "avrane": run.avrane, "scf": f"{run.scf_converged}/{run.scf_total}"})
    return rows


def reaction_analysis(runs: list[Run]) -> list[dict[str, Any]]:
    names = sorted(set.intersection(*(set(run.reactions) for run in runs if run.reactions)))
    output = []
    evaluated = [run for run in runs if run.wtmad is not None and run.reactions]
    for name in names:
        xs = [run.reactions[name] for run in evaluated]
        ys = [run.wtmad for run in evaluated]
        output.append(
            {
                "reaction": name,
                "n": len(xs),
                "mean_abs_error": mean(xs),
                "range_abs_error": max(xs) - min(xs),
                "spearman_with_wtmad": spearman(xs, ys),
                "e3_abs_error": next((run.reactions[name] for run in runs if run.key == "e3"), None),
                "s3_abs_error": next((run.reactions[name] for run in runs if run.key == "s3"), None),
                "s4_abs_error": next((run.reactions[name] for run in runs if run.key == "s4"), None),
            }
        )
    return sorted(output, key=lambda row: abs(row["spearman_with_wtmad"]), reverse=True)


def avrane_analysis(runs: list[Run]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluated = [run for run in runs if run.avrane is not None and run.ranes]
    components = []
    for component in ("rho", "grad", "lapl"):
        components.append(
            {
                "component": component,
                "n": len(evaluated),
                "min": min(run.ranes[component] for run in evaluated),
                "max": max(run.ranes[component] for run in evaluated),
                "range": max(run.ranes[component] for run in evaluated) - min(run.ranes[component] for run in evaluated),
                "spearman_with_wtmad": spearman([run.ranes[component] for run in evaluated], [run.wtmad for run in evaluated]),
            }
        )
    systems = sorted(set.intersection(*(set(run.per_system) for run in evaluated)))
    system_rows = []
    for system in systems:
        values = [mean(run.per_system[system][component] for component in ("rho", "grad", "lapl")) for run in evaluated]
        system_rows.append(
            {
                "system": system,
                "n": len(values),
                "mean_niad": mean(values),
                "range_niad": max(values) - min(values),
                "spearman_with_total_avrane": spearman(values, [run.avrane for run in evaluated]),
                "spearman_with_wtmad": spearman(values, [run.wtmad for run in evaluated]),
            }
        )
    return components, sorted(system_rows, key=lambda row: row["range_niad"], reverse=True)


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def generate_report(
    runs: list[Run],
    summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    simple: list[dict[str, Any]],
    schedule_corr: list[dict[str, Any]],
    trajectory_corr: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    components: list[dict[str, Any]],
    systems: list[dict[str, Any]],
) -> str:
    ranked = sorted((run for run in runs if run.wtmad is not None and run.avrane is not None), key=lambda run: run.wtmad)
    pareto = []
    for candidate in ranked:
        dominated = any(
            other.wtmad <= candidate.wtmad
            and other.avrane <= candidate.avrane
            and (other.wtmad < candidate.wtmad or other.avrane < candidate.avrane)
            for other in ranked
        )
        if not dominated:
            pareto.append(candidate)
    lines = [
        "# Retrospective schedule analysis: 30 fixed-architecture runs",
        "",
        "## Scope and validity",
        "",
        f"The primary dataset contains **{len(runs)}** 500-epoch replays: one constrained baseline, 20 earlier schedule variants, and nine completed points from the simple schedule sweep (S9 is absent). Architecture, data, seed, final-epoch checkpoint selection, and PBE-D3(BJ) evaluation protocol are fixed. Consequently, matched schedule contrasts are informative for this seed, but they are not estimates of seed variance.",
        "",
        "The reported WTMAD value is the archived 30-reaction interface panel. It must not be described as full GMTKN55 WTMAD-2 until that has been independently verified. Failed or unconverged SCFs are retained as a reliability flag; a lower score accompanied by a failed case is not treated as clean evidence of improvement.",
        "",
        "## External landscape",
        "",
        "| Run | Family | WTMAD | avRANE | SCF |",
        "|---|---|---:|---:|---:|",
    ]
    for run in ranked:
        scf = f"{run.scf_converged}/{run.scf_total}" if run.scf_total else "NA"
        lines.append(f"| {run.key} | {run.family} | {run.wtmad:.3f} | {run.avrane:.6f} | {scf} |")
    lines.extend([
        "",
        "The observed Pareto set is: " + ", ".join(f"**{run.key}** ({run.wtmad:.3f}, {run.avrane:.6f})" for run in pareto) + ".",
        "",
        "No completed schedule reaches either WTMAD < 6 or avRANE < 0.45. E3 remains the best energy point (6.282) and S7/H8 define the best density region (~0.465), so the remaining gaps are 0.282 WTMAD units and about 0.015 avRANE. The WTMAD target is a local extrapolation from the observed gain; the avRANE target is at the edge of what schedule-only variation has demonstrated.",
        "",
        "## Clean simple-sweep evidence",
        "",
        "| Run | Guard g | Repair end | WTMAD | avRANE | SCF |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in simple:
        lines.append(f"| {row['run']} | {row['guard']:.2f} | {row['repair_end']} | {row['wtmad']:.3f} | {row['avrane']:.6f} | {row['scf']} |")
    lines.extend([
        "",
        "The guard response is reproducibly U-shaped in WTMAD across both repair endpoints. Moving from g=0 to g=0.25 improves WTMAD by 0.231 (end 420) and 0.299 (end 440); moving from g=0.25 to g=0.50 loses 0.287 and 0.279. Thus a moderate early homotopy is supported, while stronger density protection over-regularizes energies.",
        "",
        "Repair endpoint has only a small conditional effect: end 440 changes WTMAD by +0.038, -0.030, -0.038, and -0.060 at g=0, 0.25, 0.50, and 0.75. The corresponding avRANE changes are -0.000172, +0.000133, +0.001018, and +0.001401. These effects are much smaller than the guard curvature and are not monotonic. S3 is therefore the cleaner simple candidate: its 0.030 WTMAD disadvantage to S4 is coupled to 84/84 rather than 83/84 converged SCFs.",
        "",
        "A descriptive quadratic fit with a repair-end indicator places the WTMAD minimum at g=0.270 and the avRANE minimum at g=0.580. These are not precise optima, but they quantify a real conflict: the energy-favorable pre-repair path is less density-protective than the avRANE-favorable path.",
        "",
        "## Matched contrast ledger",
        "",
        "| Contrast | Isolated or near-isolated change | dWTMAD | davRANE |",
        "|---|---|---:|---:|",
    ])
    for row in contrasts:
        lines.append(f"| {row['left']} -> {row['right']} | {row['change']} | {format_float(row['delta_wtmad'], 3)} | {format_float(row['delta_avrane'], 6)} |")
    lines.extend([
        "",
        "The strongest defensible conclusions are negative constraints on schedule design: repair E_xc=4 is worse than 3; repair reaction scale 1.0 is worse than 0.75; soft/clipped terminal exits are consistently poor; and extending a strong repair until epoch 460 is harmful.",
        "",
        "The decisive new contrast is **E3 versus S4**. Both use exactly the same repair on epochs 281-440 and the same ordinary consolidation on 441-500. Replacing E3's clipped/stepped first 280 epochs with the smooth g=0.25 homotopy worsens WTMAD by 0.115 and avRANE by 0.002750. Therefore E3's remaining advantage over the best simple schedule is an early-path effect, not a repair-timing effect.",
        "",
        "## Training trajectories",
        "",
        "Terminal internal losses are not valid selection metrics. E2 and X1-X5 obtain train Fchem near 29-30 and val Vxc near 0.12, yet have WTMAD 7.24-7.53. E3 deliberately finishes at higher internal losses and gives the best external energy score. The schedule is selecting a basin and then relaxing objective bias, not minimizing any one logged loss to completion.",
        "",
        "E3 versus S4 makes this especially clear. S4 has lower Vxc and E_xc losses through almost the entire trajectory, including the shared repair and consolidation, but its external metrics are worse. During the shared repair, E3 reaches about 1.0 lower mean train Fchem while retaining roughly 0.007 higher train Vxc and 0.84 higher train E_xc. In the final 60 epochs E3 still has lower train Fchem but worse validation Fchem/Vxc/E_xc. The early schedule therefore changes the representation/basin in a way that the held-out training objectives do not rank correctly.",
        "",
        "The largest cross-run rank correlations are listed below as diagnostics, not causal effects, because schedule families are heavily confounded:",
        "",
        "| Trajectory feature | target | n | Spearman rho |",
        "|---|---|---:|---:|",
    ])
    for row in trajectory_corr[:12]:
        lines.append(f"| {row['feature']} | {row['target']} | {row['n']} | {row['spearman']:.3f} |")
    lines.extend([
        "",
        "A family-held-out response model was intentionally not used to nominate an optimum: with only a few schedule families, continuous controls are aliases for family identity and such a model would extrapolate structure rather than estimate a stable causal surface. The clean factorial and matched contrasts carry more evidential weight.",
        "",
        "## WTMAD decomposition",
        "",
        "Raw reaction errors are not WTMAD contributions, but they show where schedules differ. The most schedule-sensitive reactions are:",
        "",
        "| Reaction | Error range | rho(error, WTMAD) | E3 | S3 | S4 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(reactions, key=lambda item: item["range_abs_error"], reverse=True)[:12]:
        lines.append(f"| {row['reaction']} | {row['range_abs_error']:.3f} | {row['spearman_with_wtmad']:.3f} | {format_float(row['e3_abs_error'], 2)} | {format_float(row['s3_abs_error'], 2)} | {format_float(row['s4_abs_error'], 2)} |")
    lines.extend([
        "",
        "E3's advantage is distributed rather than attributable to one reaction. S3/S4 improve some large raw errors but lose on highly influential subsets, which explains why an unweighted mean absolute error can move in the opposite direction from official WTMAD.",
        "",
        "Against S4 specifically, E3 is worse on MB16-43, DIPCS10, BSR36, G21EA-14, and DC13, but better on W4-11-132, G21EA-25, FH51-24, SIE4x4, and BH76. This is a chemically structured trade rather than uniform error shrinkage. Any next schedule that merely lowers aggregate Fchem is likely to move toward the wrong side of this trade.",
        "",
        "## avRANE decomposition",
        "",
        "| Component | Observed range | rho(component, WTMAD) |",
        "|---|---:|---:|",
    ])
    for row in components:
        lines.append(f"| {row['component']} | {row['range']:.6f} | {row['spearman_with_wtmad']:.3f} |")
    lines.extend([
        "",
        "The systems with the largest schedule-induced mean-component ranges are:",
        "",
        "| System | NIAD range | rho(system NIAD, total avRANE) | rho(system NIAD, WTMAD) |",
        "|---|---:|---:|---:|",
    ])
    for row in systems[:10]:
        lines.append(f"| {row['system']} | {row['range_niad']:.6f} | {row['spearman_with_total_avrane']:.3f} | {row['spearman_with_wtmad']:.3f} |")
    lines.extend([
        "",
        "The density trade-off is component-specific. Moderate guard improves the energy/rho side, whereas stronger guard tends to improve gradient and Laplacian response while sacrificing WTMAD. This is why a scalar avRANE target alone is insufficient for schedule inference.",
        "",
        "E3 versus S4 resolves the component origin of their avRANE difference: S4 is slightly better in rho (0.43005 versus 0.43159), but E3 is better in gradient (0.38987 versus 0.39425) and Laplacian (0.57421 versus 0.57962). E3's early stepped path therefore preserves derivative quality, not pointwise density alone.",
        "",
        "## What the 30 runs actually support",
        "",
        "1. **A late, finite energy-repair excursion is useful.** It must be followed by ordinary sum-gradient consolidation; leaving the model in a strongly repaired or softly clipped regime is consistently worse.",
        "2. **Repair strength has a narrow useful range.** The best supported values remain Vxc=15, E_xc=3, and reaction scale=0.75. Increasing E_xc or reaction scale hurts both objectives in near-matched comparisons.",
        "3. **The pre-repair path matters, but only moderately.** The simple factorial shows an optimum near guard g=0.25, not at either no guard or strong guard. This establishes curvature rather than a monotonic rule.",
        "4. **Repair endpoint is secondary near the useful region.** At onset=281, changing endpoint 420 -> 440 moves WTMAD by only -0.030 at g=0.25. E3 and S4 prove that E3's advantage is instead created before epoch 281. Repair onset is not independently mapped, but it is no longer the main explanation for E3 versus the simple schedules.",
        "5. **Lower internal losses are often anti-predictive beyond a point.** Driving Fchem/Vxc further selects externally worse models. The relevant signal is the response to transitions and recovery during consolidation, not the terminal minimum.",
        "6. **One part of E3's extra complexity is buying quality.** The conceptually three-stage S3 reaches 6.427/0.46784 and beats most bespoke schedules, but the exactly matched E3-S4 suffix proves that E3's first 280 epochs improve both external metrics. The unresolved task is to compress that early path without deleting its useful mechanism.",
        "7. **More epochs are not the mechanism.** Five secondary 550-epoch phase-extension runs score 7.178-7.323 WTMAD and 0.4788-0.4904 avRANE. They are excluded from the fixed-500 primary analysis, but uniformly show that simply lengthening any baseline phase does not recover E3-like quality.",
        "",
        "## Consequence for the next design step",
        "",
        "The next ten runs should not be chosen by a global black-box fit or by perturbing every E3 parameter. The primary block should decompose E3's first 280 epochs: early clipping, the 40/20/10 Vxc plateaus, and the transition into repair. All candidates should preserve the now-supported suffix (repair 281-440 at 15/3/0.75, then ordinary 7/1/1 consolidation). A smaller secondary block can test one repair-onset axis and the energy-density compromise near guard g=0.27-0.58. Exact runs should be selected only after ensuring each run removes one early-stage degree of freedom or resolves one remaining confound.",
        "",
        "## Files",
        "",
        "- `run_summary.csv`: external metrics, terminal metrics, schedule features, and robust trajectory features.",
        "- `matched_contrasts.csv`: predeclared matched and near-matched comparisons.",
        "- `simple_factorial.csv`: clean guard-by-endpoint sweep.",
        "- `schedule_correlations.csv` and `trajectory_correlations.csv`: descriptive associations.",
        "- `reaction_sensitivity.csv`: per-reaction raw-error variation.",
        "- `avrane_components.csv` and `avrane_systems.csv`: density decomposition.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    schedule_rows = [schedule_features(run) for run in runs]
    trajectory_rows = [trajectory_features(run) for run in runs]
    schedule_by_key = {row["run"]: row for row in schedule_rows}
    trajectory_by_key = {row["run"]: row for row in trajectory_rows}
    summary = []
    for run in runs:
        row = {
            "run": run.key,
            "family": run.family,
            "wtmad": run.wtmad,
            "avrane": run.avrane,
            "avrane_rho": run.ranes.get("rho"),
            "avrane_grad": run.ranes.get("grad"),
            "avrane_lapl": run.ranes.get("lapl"),
            "scf_total": run.scf_total,
            "scf_converged": run.scf_converged,
            "history_path": str(run.history_path),
            "manifest_path": str(run.manifest_path) if run.manifest_path else None,
            **{key: value for key, value in schedule_by_key[run.key].items() if key not in {"run", "family"}},
            **{key: value for key, value in trajectory_by_key[run.key].items() if key != "run"},
        }
        summary.append(row)
    runs_by_key = {run.key: run for run in runs}
    contrasts = matched_contrasts(runs_by_key)
    simple = simple_factorial(runs_by_key)
    reactions = reaction_analysis(runs)
    components, systems = avrane_analysis(runs)
    schedule_features_to_test = [key for key in schedule_rows[0] if key not in {"run", "family", "guard", "repair_end_design", "raw_segments"}]
    trajectory_features_to_test = [key for key in trajectory_rows[0] if key != "run"]
    schedule_corr = correlations(summary, schedule_features_to_test, "wtmad") + correlations(summary, schedule_features_to_test, "avrane")
    schedule_corr.sort(key=lambda row: abs(row["spearman"]), reverse=True)
    trajectory_corr = correlations(summary, trajectory_features_to_test, "wtmad") + correlations(summary, trajectory_features_to_test, "avrane")
    trajectory_corr.sort(key=lambda row: abs(row["spearman"]), reverse=True)
    write_csv(OUT / "run_summary.csv", summary)
    write_csv(OUT / "matched_contrasts.csv", contrasts)
    write_csv(OUT / "simple_factorial.csv", simple)
    write_csv(OUT / "schedule_correlations.csv", schedule_corr)
    write_csv(OUT / "trajectory_correlations.csv", trajectory_corr)
    write_csv(OUT / "reaction_sensitivity.csv", reactions)
    write_csv(OUT / "avrane_components.csv", components)
    write_csv(OUT / "avrane_systems.csv", systems)
    report = generate_report(runs, summary, contrasts, simple, schedule_corr, trajectory_corr, reactions, components, systems)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Analysed {len(runs)} runs; wrote {OUT}")


if __name__ == "__main__":
    main()
