"""S5-equivalent shared-engine entry point for tau-free Lap/full-Vxc training.

``pilot-smoke`` runs the real 268-group Minnesota PBE warm start followed by
the exact five S5 phase operators for a few local updates on existing pilot
stencils. ``production`` connects the same 500-epoch S5 engine to a strictly
verified 90-system Lap corpus. The production mode is provided for later use;
this task intentionally does not launch it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from dataset import collate_fn_predopt
from lap_checkpoint import checkpoint_payload, load_lap_checkpoint
from lap_data import read_stencil_h5, require_full_center_verification, verify_corpus
from lap_s5_protocol import (
    apply_lap_s5_scale_overrides,
    build_lap_s5_phase_smoke_view,
    build_lap_s5_protocol,
)
from lap_s5_provenance import (
    S5_INITIAL_LR,
    build_lap_s5_provenance,
    source_bindings_from_verified_corpus,
)
from lap_training import canonical_predopt_view, run_predopt
from optuna_joint import (
    DEFAULT_MRKS_DISPERSIONS,
    OMEGA,
    add_gradient_list_to_parameters,
    build_dataloaders,
    build_model,
    build_preopt_loader,
    build_scheduler,
    clip_gradient_list_by_global_norm,
    configure_optimizers,
    lap_exc_loss,
    lap_full_vxc_loss,
    load_mrks_dispersions,
    run_trial,
    scale_gradient_list,
    set_random_seed,
    train_one_epoch,
    vxc_collate_fn,
)
from predopt import DatasetPredopt
from train_lap import DEFAULT_REACTION_DISPERSIONS, load_reaction_dispersions

TRAIN_MODE = "lap_full_vxc"
POTENTIAL_MODE = "full_euler"
MODEL_NAME = "PBE-Lap-LGxGc_6_32"
DISPERSIONS_MRKS_SHA_NAME = DEFAULT_MRKS_DISPERSIONS.name
DEFAULT_POINT_CHUNKS = (128, 256, 512, 1024, 2048)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_minnesota_view(
    pickle_path: Path,
    manifest_path: Path,
    expected_groups: int,
) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256(pickle_path)
    expected_hash = manifest.get("output_sha256")
    if expected_hash != actual_hash and not (
        expected_groups == 1
        and isinstance(manifest.get("pilot_view"), dict)
        and manifest["pilot_view"].get("output_sha256") == actual_hash
    ):
        raise ValueError(f"Minnesota artifact hash does not match {manifest_path}.")
    data = pickle.loads(pickle_path.read_bytes())
    if not isinstance(data, dict) or len(data) != expected_groups:
        raise ValueError(
            f"Minnesota view must contain exactly {expected_groups} groups."
        )
    if manifest.get("base_reaction_count") != 268:
        raise ValueError(
            "Minnesota manifest does not prove the Diet-clean 268-group set."
        )
    if any(index not in data for index in range(expected_groups)):
        raise ValueError("Minnesota groups must use contiguous integer keys from zero.")
    return data, manifest


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        torch.cuda.set_device(device.index or 0)
    return device


def windows_path_to_wsl(path: str | Path) -> str:
    """Translate an absolute Windows drive path for a WSL SCF subprocess."""
    resolved = str(Path(path).resolve())
    match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", resolved)
    if match is None:
        raise ValueError(f"Cannot map this path into WSL: {resolved}")
    drive, tail = match.groups()
    tail = tail.replace("\\", "/")
    return f"/mnt/{drive.lower()}/{tail}"


def choose_scf_runtime(requested: str) -> str:
    if requested not in {"auto", "native", "wsl"}:
        raise ValueError(f"Unsupported SCF runtime: {requested!r}.")
    has_native_pyscf = importlib.util.find_spec("pyscf") is not None
    if requested == "native":
        if not has_native_pyscf:
            raise RuntimeError("Native SCF runtime requested but PySCF is unavailable.")
        return "native"
    if requested == "wsl" and shutil.which("wsl.exe") is None:
        raise RuntimeError("WSL SCF runtime requested but wsl.exe is unavailable.")
    if requested == "auto" and has_native_pyscf:
        return "native"
    if shutil.which("wsl.exe") is None:
        raise RuntimeError("Neither native PySCF nor wsl.exe is available for SCF.")
    probe = subprocess.run(
        ["wsl.exe", "-e", "python3", "-c", "import pyscf, torch"],
        check=False,
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "WSL is present but its Python lacks PySCF/PyTorch: " + probe.stderr[-2000:]
        )
    return "wsl"


def build_scf_command(runtime, scf_script, checkpoint, npz_path, output_path):
    if runtime == "native":
        prefix = [sys.executable, str(scf_script)]
        paths = [str(checkpoint), str(npz_path), str(output_path)]
    elif runtime == "wsl":
        prefix = ["wsl.exe", "-e", "python3", windows_path_to_wsl(scf_script)]
        paths = [
            windows_path_to_wsl(checkpoint),
            windows_path_to_wsl(npz_path),
            windows_path_to_wsl(output_path),
        ]
    else:
        raise ValueError(f"Unsupported selected SCF runtime: {runtime!r}.")
    return [
        *prefix,
        "--checkpoint",
        paths[0],
        "--npz",
        paths[1],
        "--output",
        paths[2],
        "--grid-level",
        "1",
    ]


def parse_chunks(text: str) -> tuple[int, ...]:
    try:
        chunks = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            "Chunk candidates must be comma-separated positive integers."
        ) from exc
    if not chunks or any(chunk <= 0 for chunk in chunks):
        raise ValueError("Chunk candidates must be positive integers.")
    return tuple(dict.fromkeys(chunks))


def evaluate_predopt_metrics(model, reactions, device, dtype, point_chunk_size):
    from predopt_targets import _ADAPTIVE_INDICES, _prepare_predopt_targets

    from dft_functionals import PBE_CONSTANTS

    totals = torch.zeros(2, dtype=torch.float64, device=device)
    point_count = 0
    model.eval()
    with torch.no_grad():
        for reaction in reactions.values():
            raw = reaction["Grid"]
            if raw.ndim != 2 or raw.shape[1] != 9:
                raise ValueError("Minnesota Lap predopt requires raw N×9 grids.")
            for start in range(0, len(raw), point_chunk_size):
                block = raw[start : start + point_chunk_size].to(
                    device=device, dtype=dtype
                )
                pred = model(block)[:, _ADAPTIVE_INDICES]
                target = _prepare_predopt_targets(
                    PBE_CONSTANTS.to(block), len(block), block.device
                )
                delta = pred - target
                totals[0] += delta.double().square().sum()
                totals[1] += delta.double().abs().sum()
                point_count += len(block)
    denominator = point_count * len(_ADAPTIVE_INDICES)
    return {
        "mse": float((totals[0] / denominator).cpu()),
        "mae": float((totals[1] / denominator).cpu()),
        "grid_points": point_count,
        "adaptive_targets": 9,
    }


def _gradient_norm(grads) -> float:
    return math.sqrt(
        sum(
            float(grad.detach().double().square().sum())
            for grad in grads
            if grad is not None
        )
    )


def _gradient_delta(first, second) -> tuple[float, float]:
    diff_sq, reference_sq, max_abs = 0.0, 0.0, 0.0
    for a, b in zip(first, second):
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError(
                "Chunk-size sweep changed which model parameters are used."
            )
        delta = a.detach().double() - b.detach().double()
        diff_sq += float(delta.square().sum())
        reference_sq += float(b.detach().double().square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    return max_abs, math.sqrt(diff_sq / max(reference_sq, 1e-30))


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).casefold()
    )


def benchmark_vxc_chunks(model, record, device, chunks):
    """Benchmark real H2 full-Vxc loss/gradients, retrying OOM with half chunks."""
    batch = vxc_collate_fn([record])
    parameters = tuple(model.parameters())
    baseline_loss = None
    baseline_grads = None
    rows = []
    free_at_start = total_vram = None
    if device.type == "cuda":
        free_at_start, total_vram = torch.cuda.mem_get_info(device)
    for requested in chunks:
        actual = requested
        retries = 0
        while True:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                loss = lap_full_vxc_loss(model, batch, device, actual)
                grads = torch.autograd.grad(loss, parameters, allow_unused=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                max_delta = relative_delta = 0.0
                if baseline_loss is None:
                    baseline_loss = loss.detach().double().cpu()
                    baseline_grads = tuple(
                        None if grad is None else grad.detach().clone()
                        for grad in grads
                    )
                else:
                    max_delta, relative_delta = _gradient_delta(grads, baseline_grads)
                row = {
                    "requested_chunk": requested,
                    "actual_chunk": actual,
                    "oom_retries": retries,
                    "full_vxc_loss": float(loss.detach()),
                    "relative_loss_error": float(
                        abs(float(loss.detach().double().cpu() - baseline_loss))
                        / max(abs(float(baseline_loss)), 1e-30)
                    ),
                    "gradient_norm": _gradient_norm(grads),
                    "max_abs_gradient_delta_from_128_baseline": max_delta,
                    "relative_l2_gradient_delta_from_128_baseline": relative_delta,
                    "seconds": elapsed,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None,
                }
                rows.append(row)
                del loss, grads
                break
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if not _is_cuda_oom(exc) or actual <= 1:
                    raise
                retries += 1
                actual = max(1, actual // 2)
                torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("Full-Vxc chunk benchmark produced no rows.")
    stable = [
        row
        for row in rows
        if row["relative_l2_gradient_delta_from_128_baseline"] <= 2e-4
        and row["relative_loss_error"] <= 2e-6
    ]
    if free_at_start is not None:
        safe = [
            row
            for row in stable
            if row["cuda_peak_reserved_bytes"] <= 0.75 * free_at_start
        ]
        if safe:
            stable = safe
    if not stable:
        raise RuntimeError("No chunk candidate matched the reference within tolerance.")
    selected = min(stable, key=lambda row: row["seconds"])
    return {
        "device": str(device),
        "free_vram_at_start_bytes": free_at_start,
        "total_vram_bytes": total_vram,
        "selected_chunk": int(selected["actual_chunk"]),
        "selection_rule": "fastest gradient-equivalent candidate using at most 75% of starting free VRAM",
        "rows": rows,
    }


def _loss_and_gradient(model, key, loss):
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return {
        "loss": float(loss.detach()),
        "gradient_norm": _gradient_norm(grads),
    }, tuple(None if g is None else g.detach().clone() for g in grads)


def collect_objective_calibration(
    model,
    reactions,
    records,
    device,
    dtype,
    point_chunk_size,
    reaction_dispersions,
    mrks_dispersions,
):
    from lap_training import reaction_loss

    table: dict[str, Any] = {"minnesota_reactions": [], "mrks_systems": []}
    chosen_grads = None
    for index, reaction in enumerate(reactions):
        target = reaction["Energy"]
        value = reaction_loss(
            model, reaction, target, device, dtype, reaction_dispersions
        )
        row, grads = _loss_and_gradient(model, "reaction", value)
        row.update(
            reaction_index=index,
            database=reaction.get("Database"),
            selected_variant=(
                Path(reaction.get("component_paths", [""])[0]).stem
                if reaction.get("component_paths")
                else "canonical"
            ),
        )
        table["minnesota_reactions"].append(row)
        if index == 0:
            chosen_grads = {"reaction": grads}
    for index, record in enumerate(records):
        batch = vxc_collate_fn([record])
        exc_loss, _, _ = lap_exc_loss(
            model,
            batch,
            device,
            mrks_dispersions,
            True,
            point_chunk_size,
        )
        exc_row, exc_grads = _loss_and_gradient(model, "exc", exc_loss)
        vxc_loss = lap_full_vxc_loss(model, batch, device, point_chunk_size)
        vxc_row, vxc_grads = _loss_and_gradient(model, "vxc", vxc_loss)
        table["mrks_systems"].append(
            {"name": record["Name"], "E_xc": exc_row, "full_Vxc": vxc_row}
        )
        if record["Name"] == "H2":
            chosen_grads["exc"] = exc_grads
            chosen_grads["vxc"] = vxc_grads
            chosen_grads["h2_losses"] = {
                "E_xc": exc_row["loss"],
                "full_Vxc": vxc_row["loss"],
            }
    if chosen_grads is None or not {"reaction", "exc", "vxc"} <= chosen_grads.keys():
        raise ValueError(
            "Calibration must include a Minnesota reaction and the H2 stencil."
        )
    return table, chosen_grads


def _transform_gradients(grads, weight, merge_strategy, clip, grad_scale):
    transformed = scale_gradient_list(grads, float(weight))
    if merge_strategy == "clip_then_sum":
        max_norm = None if clip == "none" else float(clip)
        transformed, _ = clip_gradient_list_by_global_norm(transformed, max_norm)
    elif merge_strategy != "sum":
        raise ValueError(f"Unsupported S5 merge strategy: {merge_strategy}.")
    return scale_gradient_list(transformed, float(grad_scale))


def _cosine(a, b):
    dot, aa, bb = 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            raise ValueError("Objective gradient sparsity differs across objectives.")
        xd, yd = x.detach().double(), y.detach().double()
        dot += float((xd * yd).sum())
        aa += float(xd.square().sum())
        bb += float(yd.square().sum())
    return dot / max(math.sqrt(aa * bb), 1e-30)


def _merge_gradients(reference_model, grad_lists):
    clone = copy.deepcopy(reference_model)
    parameters = [
        parameter for parameter in clone.parameters() if parameter.requires_grad
    ]
    for grads in grad_lists:
        add_gradient_list_to_parameters(parameters, grads)
    combined = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    return clone, parameters, combined


def phase_gradient_rows(model, chosen_grads, protocol):
    rows = []
    for phase in protocol["epoch_schedule"]:
        params = phase["params"]
        reaction = _transform_gradients(
            chosen_grads["reaction"],
            1.0,
            params["gradient_merge_strategy"],
            params["reaction_grad_clip"],
            params["reaction_grad_scale"],
        )
        vxc = _transform_gradients(
            chosen_grads["vxc"],
            OMEGA * params["vxc_loss_scale"],
            params["gradient_merge_strategy"],
            params["vxc_grad_clip"],
            1.0,
        )
        exc = _transform_gradients(
            chosen_grads["exc"],
            params["exc_loss_scale"],
            params["exc_gradient_merge_strategy"],
            params["exc_grad_clip"],
            params["exc_grad_scale"],
        )
        clone, clone_params, combined = _merge_gradients(model, (reaction, vxc, exc))
        rows.append(
            {
                "phase_id": phase["phase_id"],
                "reaction_effective_gradient_norm": _gradient_norm(reaction),
                "vxc_effective_gradient_norm": _gradient_norm(vxc),
                "exc_effective_gradient_norm": _gradient_norm(exc),
                "combined_gradient_norm": _gradient_norm(combined),
                "reaction_vxc_cosine": _cosine(reaction, vxc),
                "reaction_exc_cosine": _cosine(reaction, exc),
                "vxc_exc_cosine": _cosine(vxc, exc),
            }
        )
        del clone, clone_params
    return rows


def candidate_scale_sweep(model, chosen_grads, protocol):
    rnorm = _gradient_norm(chosen_grads["reaction"])
    enorm = _gradient_norm(chosen_grads["exc"])
    vnorm = _gradient_norm(chosen_grads["vxc"])
    match_v = max(0.1, 2.0 * rnorm / max(vnorm, 1e-30))
    match_e = max(1e-5, rnorm / max(enorm, 1e-30))
    candidates = [
        ("reference_S5", protocol),
        (
            "reaction_matched_raw_norms_pilot_only",
            apply_lap_s5_scale_overrides(
                protocol,
                {
                    phase["phase_id"]: {
                        "vxc_loss_scale": match_v,
                        "exc_loss_scale": match_e,
                    }
                    for phase in protocol["epoch_schedule"]
                },
            ),
        ),
        (
            "half_reaction_matched_vxc_pilot_only",
            apply_lap_s5_scale_overrides(
                protocol,
                {
                    phase["phase_id"]: {"vxc_loss_scale": max(0.1, match_v / 2)}
                    for phase in protocol["epoch_schedule"]
                },
            ),
        ),
    ]
    rows = []
    for candidate_name, candidate_protocol in candidates:
        for phase in candidate_protocol["epoch_schedule"]:
            params = phase["params"]
            grads = (
                _transform_gradients(
                    chosen_grads["reaction"],
                    1.0,
                    params["gradient_merge_strategy"],
                    params["reaction_grad_clip"],
                    params["reaction_grad_scale"],
                ),
                _transform_gradients(
                    chosen_grads["vxc"],
                    OMEGA * params["vxc_loss_scale"],
                    params["gradient_merge_strategy"],
                    params["vxc_grad_clip"],
                    1.0,
                ),
                _transform_gradients(
                    chosen_grads["exc"],
                    params["exc_loss_scale"],
                    params["exc_gradient_merge_strategy"],
                    params["exc_grad_clip"],
                    params["exc_grad_scale"],
                ),
            )
            clone, clone_params, combined = _merge_gradients(model, grads)
            optimizer = configure_optimizers(
                clone,
                learning_rate=S5_INITIAL_LR,
                optimizer_str="radamw",
                weight_decay=0.01,
            )
            build_scheduler(optimizer, n_train=500)
            before = [parameter.detach().clone() for parameter in clone_params]
            optimizer.step()
            update_norm = math.sqrt(
                sum(
                    float((parameter.detach() - original).double().square().sum())
                    for parameter, original in zip(clone_params, before)
                )
            )
            rows.append(
                {
                    "candidate": candidate_name,
                    "phase_id": phase["phase_id"],
                    "vxc_loss_scale": params["vxc_loss_scale"],
                    "effective_vxc_loss_coefficient": OMEGA * params["vxc_loss_scale"],
                    "exc_loss_scale": params["exc_loss_scale"],
                    "reaction_grad_scale": params["reaction_grad_scale"],
                    "reaction_effective_gradient_norm": _gradient_norm(grads[0]),
                    "vxc_effective_gradient_norm": _gradient_norm(grads[1]),
                    "exc_effective_gradient_norm": _gradient_norm(grads[2]),
                    "combined_gradient_norm": _gradient_norm(combined),
                    "reaction_vxc_cosine": _cosine(grads[0], grads[1]),
                    "reaction_exc_cosine": _cosine(grads[0], grads[2]),
                    "vxc_exc_cosine": _cosine(grads[1], grads[2]),
                    "one_step_parameter_update_norm": update_norm,
                    "finite": all(
                        math.isfinite(value)
                        for value in (_gradient_norm(combined), update_norm)
                    ),
                }
            )
            del clone, optimizer, before, clone_params
    return {
        "derived_vxc_scale_match": match_v,
        "derived_exc_scale_match": match_e,
        "pilot_only": True,
        "candidates": rows,
    }


def _provenance_for_pilot(
    args, predopt_path, predopt_manifest, reaction_path, reaction_manifest_path, record
):
    stencil_path = Path(args.stencil_dir) / f"{record['Name']}.h5"
    source_audit = Path(args.stencil_dir) / "mrks_source_audit_v2.json"
    dispersion_path = Path(args.mrks_dispersions_pickle)
    audit_hash = (
        sha256(source_audit)
        if source_audit.is_file()
        else record["SourceProvenance"]["legacy_target_sha256"]
    )
    sources = {
        "minnesota": {
            "identity": predopt_path.name,
            "sha256": sha256(predopt_path),
            "manifest_identity": predopt_manifest.name,
            "manifest_sha256": sha256(predopt_manifest),
        },
        "mrks_targets": {
            "identity": stencil_path.name,
            "sha256": sha256(stencil_path),
            "manifest_identity": source_audit.name
            if source_audit.exists()
            else "legacy_target_sha256_embedded_in_stencil",
            "manifest_sha256": audit_hash,
        },
        "dispersions": {
            "identity": dispersion_path.name,
            "sha256": sha256(dispersion_path),
        },
    }
    extra_inputs = {
        "minnesota_predopt": {
            "identity": predopt_path.name,
            "sha256": sha256(predopt_path),
            "manifest_identity": predopt_manifest.name,
            "manifest_sha256": sha256(predopt_manifest),
            "base_reaction_count": 268,
            "variant_selection": "default when present; otherwise lexicographically smallest suffix",
        },
        "minnesota_joint_pilot": {
            "identity": reaction_path.name,
            "sha256": sha256(reaction_path),
            "manifest_identity": reaction_manifest_path.name,
            "manifest_sha256": sha256(reaction_manifest_path),
            "base_reaction_groups": 1,
            "variant_sampling": "EpochSampledAugmentedDataset, deterministic seed",
        },
        "mrks_pilot_stencil": {
            "identity": stencil_path.name,
            "sha256": sha256(stencil_path),
            "npz_sha256": record["SourceProvenance"]["npz_sha256"],
            "legacy_target_sha256": record["SourceProvenance"]["legacy_target_sha256"],
            "E_xc_source": record["SourceProvenance"]["E_xc_source"],
            "NPZ_exc_wf_substituted": False,
        },
    }
    return sources, extra_inputs


def _extract_npz_path(args, record):
    if args.npz:
        return Path(args.npz)
    audit_path = Path(args.stencil_dir) / "mrks_source_audit_v2.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        match = next(
            (
                item
                for item in audit.get("records", [])
                if item.get("name") == record["Name"]
            ),
            None,
        )
        if match and match.get("npz_path"):
            source = match["npz_path"]
            if source.startswith("/mnt/"):
                drive, _, tail = source[5:].partition("/")
                source = f"{drive.upper()}:/{tail}"
            return Path(source)
    raise ValueError(
        "Could not locate original NPZ for the requested SCF smoke; pass --npz."
    )


def _build_s5_lap_model(device, dtype):
    """Build the configured Lap architecture through the shared S5 factory."""
    return build_model(
        argparse.Namespace(
            name=MODEL_NAME,
            model_type="lap",
            dropout=0.0,
            dtype=dtype,
        ),
        device,
    )


def _pilot_model(device, dtype):
    return _build_s5_lap_model(device, dtype)


def _s5_training_params(protocol):
    first = protocol["epoch_schedule"][0]["params"]
    params = {"lr_train": S5_INITIAL_LR, **copy.deepcopy(first)}
    params["epoch_schedule"] = [
        {
            "name": phase["name"],
            "start_epoch": phase["start_epoch"],
            "end_epoch": phase["end_epoch"],
            "params": copy.deepcopy(phase["params"]),
        }
        for phase in protocol["epoch_schedule"]
    ]
    return params


def run_pilot_smoke(args):
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite local Lap-S5 output {output}.")
    output.mkdir(parents=True)
    set_random_seed(args.seed)

    canonical_path = Path(args.canonical_predopt_pickle)
    canonical_manifest_path = Path(args.canonical_manifest)
    canonical_grouped, _ = load_verified_minnesota_view(
        canonical_path, canonical_manifest_path, 268
    )
    predopt_view = canonical_predopt_view(canonical_grouped)
    if len(predopt_view) != 268:
        raise ValueError(
            "PBE predopt must use exactly one canonical variant for all 268 groups."
        )
    reaction_path = Path(args.reaction_grouped_pickle)
    reaction_manifest_path = Path(args.reaction_manifest or args.canonical_manifest)
    reaction_grouped, _ = load_verified_minnesota_view(
        reaction_path, reaction_manifest_path, 1
    )
    if not isinstance(reaction_grouped[0], list) or not reaction_grouped[0]:
        raise ValueError(
            "The local reaction subset must preserve one base group's variants."
        )
    stencil_path = Path(args.stencil_dir) / f"{args.system}.h5"
    record = read_stencil_h5(stencil_path)
    require_full_center_verification(record)
    if record["Name"] != args.system:
        raise ValueError("Requested pilot system and stencil identity differ.")
    if (
        record["SourceProvenance"].get("E_xc_source")
        != "preserved legacy mRKS training target"
    ):
        raise ValueError(
            "The local stencil does not preserve the historical E_xc target."
        )
    mrks_dispersions = load_mrks_dispersions(args.mrks_dispersions_pickle)
    if record["Name"] not in mrks_dispersions:
        raise ValueError(
            f"The historical mRKS dispersion map has no {record['Name']} entry."
        )
    reaction_dispersions = load_reaction_dispersions(args.reaction_dispersions_pickle)

    model = _pilot_model(device, dtype)
    predopt_before = evaluate_predopt_metrics(
        model, predopt_view, device, dtype, args.predopt_chunk_size
    )
    predopt_loader = torch.utils.data.DataLoader(
        DatasetPredopt(predopt_view),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_predopt,
        num_workers=0,
    )
    predopt_history = run_predopt(
        model,
        predopt_loader,
        device,
        dtype,
        epochs=2,
        lr=1e-2,
        chunk=args.predopt_chunk_size,
        world_size=1,
    )
    predopt_after = evaluate_predopt_metrics(
        model, predopt_view, device, dtype, args.predopt_chunk_size
    )
    model.train()

    chunk_report = benchmark_vxc_chunks(
        model, record, device, parse_chunks(args.chunk_candidates)
    )
    point_chunk_size = args.point_chunk_size or chunk_report["selected_chunk"]

    # Use deterministic canonical variants from three distinct groups for the
    # Minnesota gradient scale check; training itself retains epoch-wise variant
    # sampling from the original one-group augmentation subset.
    calibration_reactions = [predopt_view[i] for i in range(min(3, len(predopt_view)))]
    calibration_records = [
        read_stencil_h5(Path(args.stencil_dir) / f"{name}.h5")
        for name in ("H2", "BeH2", "CO")
        if (Path(args.stencil_dir) / f"{name}.h5").is_file()
    ]
    for system_record in calibration_records:
        require_full_center_verification(system_record)
    calibration, chosen_gradients = collect_objective_calibration(
        model,
        calibration_reactions,
        calibration_records,
        device,
        dtype,
        point_chunk_size,
        reaction_dispersions,
        mrks_dispersions,
    )
    protocol = build_lap_s5_protocol()
    phase_rows = phase_gradient_rows(model, chosen_gradients, protocol)
    candidate_sweep = candidate_scale_sweep(model, chosen_gradients, protocol)

    # The short run invokes optuna_joint.train_one_epoch for every update. Each
    # call is one local phase-mechanics step over a deterministic Minnesota
    # variant and the selected verified H2 pilot stencil.
    training_args = argparse.Namespace(
        model_type="lap",
        data_protocol="lap_full_vxc",
        potential_mode="full_euler",
        vxc_batch_size=1,
        batch_size=1,
        num_workers_train=0,
        num_workers_vxc=0,
        seed=args.seed,
    )
    loaders = build_dataloaders(
        reaction_grouped,
        [record],
        trial_seed=args.seed,
        args=training_args,
        rank=0,
        world_size=1,
    )
    optimizer = configure_optimizers(
        model,
        learning_rate=S5_INITIAL_LR,
        optimizer_str="radamw",
        weight_decay=0.01,
    )
    scheduler = build_scheduler(optimizer, n_train=500)
    history = []
    step_number = 0
    for smoke_phase in build_lap_s5_phase_smoke_view(
        protocol, per_phase_step_limit=args.steps_per_phase
    )["epoch_schedule"]:
        source = next(
            item
            for item in protocol["epoch_schedule"]
            if item["phase_id"] == smoke_phase["phase_id"]
        )
        for _ in range(smoke_phase["step_limit"]):
            epoch = step_number
            reaction_dataset = loaders["train_loader"].dataset
            reaction_dataset.resample(epoch)
            loaders["train_sampler"].set_epoch(epoch)
            loaders["vxc_train_sampler"].set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            metrics, per_db, failed = train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=loaders["train_loader"],
                vxc_train_loader=loaders["vxc_train_loader"],
                params=source["params"],
                device=device,
                dispersions=reaction_dispersions,
                mrks_dispersions=mrks_dispersions,
                include_mrks_dispersion=True,
                world_size=1,
                epoch=epoch,
                potential_mode=POTENTIAL_MODE,
                data_protocol=TRAIN_MODE,
                point_chunk_size=point_chunk_size,
                max_optimizer_steps=1,
            )
            if failed or metrics.get("optimizer_steps") != 1:
                raise RuntimeError(
                    f"Lap-S5 phase step failed in {smoke_phase['phase_id']}."
                )
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "step": step_number + 1,
                "source_phase": smoke_phase["phase_id"],
                "source_epoch_range": [
                    smoke_phase["source_start_epoch"],
                    smoke_phase["source_end_epoch"],
                ],
                "production_schedule": False,
                "omega": OMEGA,
                "effective_vxc_coefficient": OMEGA
                * float(source["params"]["vxc_loss_scale"]),
                "train_fchem_reported": metrics["train_fchem"],
                "reaction_objective_backward": metrics["train_reaction_loss"],
                "full_vxc_loss": metrics["train_vxc"],
                "E_xc_batch_exc_loss": metrics["train_exc_loss"],
                "gradient_norm": metrics["gradient_norm"],
                "parameter_update_norm": metrics["parameter_update_norm"],
                "optimizer_steps": metrics["optimizer_steps"],
                "seconds": time.perf_counter() - start,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device)
                if device.type == "cuda"
                else None,
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device)
                if device.type == "cuda"
                else None,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None,
                "per_database_rmse": per_db,
            }
            if (
                not all(
                    math.isfinite(float(row[key]))
                    for key in (
                        "train_fchem_reported",
                        "reaction_objective_backward",
                        "full_vxc_loss",
                        "E_xc_batch_exc_loss",
                        "gradient_norm",
                        "parameter_update_norm",
                    )
                )
                or row["gradient_norm"] <= 0
                or row["parameter_update_norm"] <= 0
            ):
                raise FloatingPointError(
                    "Nonfinite/zero objective gradient or optimizer update."
                )
            history.append(row)
            step_number += 1

    sources, extra_inputs = _provenance_for_pilot(
        args,
        canonical_path,
        canonical_manifest_path,
        reaction_path,
        reaction_manifest_path,
        record,
    )
    provenance = build_lap_s5_provenance(
        h_bohr=record["HBohr"],
        stencil_version=record["StencilVersion"],
        derivative_order=record["StencilOrder"],
        dtype=dtype,
        model_kwargs=model.model_kwargs,
        source_bindings=sources,
        schedule=protocol,
    )
    checkpoint_path = output / "lap_s5_phase_smoke.pt"
    torch.save(
        checkpoint_payload(
            model,
            lap_s5_provenance=provenance,
            phase_smoke=True,
            production_schedule=False,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            optimizer_steps=step_number,
            pilot_inputs=extra_inputs,
        ),
        checkpoint_path,
    )
    reloaded, checkpoint_payload_on_disk = load_lap_checkpoint(
        checkpoint_path, device=device, dtype=dtype
    )
    state_equal = all(
        torch.equal(
            model.state_dict()[key].detach().cpu(),
            reloaded.state_dict()[key].detach().cpu(),
        )
        for key in model.state_dict()
    )
    if (
        not state_equal
        or checkpoint_payload_on_disk.get("lap_s5_provenance") != provenance
    ):
        raise AssertionError("Lap-S5 checkpoint save/load changed model or provenance.")

    npz_path = _extract_npz_path(args, record)
    scf_path = output / "scf_smoke.json"
    scf_script = Path(__file__).with_name("run_lap_scf_real_smoke.py")
    scf_runtime = choose_scf_runtime(args.scf_runtime)
    command = build_scf_command(
        scf_runtime,
        scf_script.resolve(),
        checkpoint_path.resolve(),
        npz_path.resolve(),
        scf_path.resolve(),
    )
    pre_scf_evidence = {
        "run_mode": "pilot_only_phase_mechanics_smoke",
        "branch": "lap_full_vxc",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "point_chunk_size": point_chunk_size,
        "predopt": {
            "source_groups": 268,
            "variant_policy": "default if available; otherwise lexicographically smallest augmentation suffix",
            "optimizer": "Adam",
            "epochs": 2,
            "learning_rate": 1e-2,
            "vxc_weight": 0,
            "vxc_steps": 0,
            "history": predopt_history,
            "before": predopt_before,
            "after": predopt_after,
        },
        "mn_pilot_groups": 1,
        "mn_pilot_variants": len(reaction_grouped[0]),
        "mrks_system": record["Name"],
        "mrks_E_xc_source": record["SourceProvenance"]["E_xc_source"],
        "npz_exc_wf_substituted": False,
        "input_hashes": {
            "canonical_minnesota": sha256(canonical_path),
            "canonical_manifest": sha256(canonical_manifest_path),
            "joint_minnesota_subset": sha256(reaction_path),
            "joint_minnesota_manifest": sha256(reaction_manifest_path),
            "mrks_stencil": sha256(stencil_path),
            "mrks_dispersion": sha256(Path(args.mrks_dispersions_pickle)),
            "reaction_dispersion": sha256(Path(args.reaction_dispersions_pickle)),
        },
        "chunk_benchmark": chunk_report,
        "raw_gradient_calibration_after_predopt": calibration,
        "s5_phase_effective_gradients": phase_rows,
        "candidate_scale_sweep_pilot_only": candidate_sweep,
        "phase_mechanics_steps": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_round_trip_equal": state_equal,
        "scf_runtime": scf_runtime,
        "scf_status": "pending",
        "all_phase_history_finite": True,
    }
    (output / "lap_s5_pre_scf_evidence.json").write_text(
        json.dumps(pre_scf_evidence, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scf_run = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        check=False,
        text=True,
        capture_output=True,
    )
    if scf_run.returncode != 0:
        raise RuntimeError(
            "RKS SCF smoke failed; inspect lap_s5_pre_scf_evidence.json and stderr: "
            + scf_run.stderr[-4000:]
        )

    result = {
        "run_mode": "pilot_only_phase_mechanics_smoke",
        "branch": "lap_full_vxc",
        "base_commit": os.popen("git rev-parse HEAD").read().strip(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "point_chunk_size": point_chunk_size,
        "predopt": {
            "source_groups": 268,
            "variant_policy": "default if available; otherwise lexicographically smallest augmentation suffix",
            "optimizer": "Adam",
            "epochs": 2,
            "learning_rate": 1e-2,
            "vxc_weight": 0,
            "vxc_steps": 0,
            "history": predopt_history,
            "before": predopt_before,
            "after": predopt_after,
        },
        "mn_pilot_groups": 1,
        "mn_pilot_variants": len(reaction_grouped[0]),
        "mrks_system": record["Name"],
        "mrks_E_xc_source": record["SourceProvenance"]["E_xc_source"],
        "npz_exc_wf_substituted": False,
        "input_hashes": {
            "canonical_minnesota": sha256(canonical_path),
            "canonical_manifest": sha256(canonical_manifest_path),
            "joint_minnesota_subset": sha256(reaction_path),
            "joint_minnesota_manifest": sha256(reaction_manifest_path),
            "mrks_stencil": sha256(stencil_path),
            "mrks_dispersion": sha256(Path(args.mrks_dispersions_pickle)),
            "reaction_dispersion": sha256(Path(args.reaction_dispersions_pickle)),
        },
        "chunk_benchmark": chunk_report,
        "raw_gradient_calibration_after_predopt": calibration,
        "s5_phase_effective_gradients": phase_rows,
        "candidate_scale_sweep_pilot_only": candidate_sweep,
        "phase_mechanics_steps": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_round_trip_equal": state_equal,
        "scf_runtime": scf_runtime,
        "scf_smoke": json.loads(scf_path.read_text(encoding="utf-8")),
        "all_phase_history_finite": True,
    }
    (output / "lap_s5_pilot_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def run_production(args):
    """Expose the exact full 90-system route without launching it in this task."""
    if args.point_chunk_size is None:
        raise ValueError("Production mode requires an explicit --point-chunk-size.")
    if args.n_train != 500 or args.predopt_epochs != 2 or args.predopt_lr != 1e-2:
        raise ValueError(
            "Production settings are fixed to S5: 500 epochs and 2-epoch LR=.01 predopt."
        )
    corpus_dir = Path(args.lap_corpus)
    manifest, vxc_records = verify_corpus(corpus_dir)
    manifest_path = corpus_dir / "preprocessing_manifest.json"
    grouped_path = corpus_dir / "data_train_grouped.pickle"
    with grouped_path.open("rb") as handle:
        grouped = pickle.load(handle)
    if len(grouped) != 268:
        raise ValueError("Strict Lap corpus did not provide all 268 Minnesota groups.")
    predopt_view = canonical_predopt_view(grouped)
    if len(predopt_view) != 268:
        raise ValueError("Production PBE predopt view must have exactly 268 groups.")

    local_rank, world_size, device, rank0 = (
        (0, 1, resolve_device(args.device), True)
        if not int(os.environ.get("WORLD_SIZE", "1")) > 1
        else __import__("optuna_joint").init_distributed()
    )
    if args.device not in ("auto", str(device)) and device.type != "cuda":
        raise ValueError(
            "Requested production device differs from torchrun local device."
        )
    dtype = getattr(torch, args.dtype)
    output = Path(args.output_dir)
    if rank0:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output directory {output}.")
        output.mkdir(parents=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    set_random_seed(args.seed)
    model = _build_s5_lap_model(device, dtype)
    predopt_before = evaluate_predopt_metrics(
        model, predopt_view, device, dtype, args.predopt_chunk_size
    )
    preopt_loader = build_preopt_loader(
        predopt_view,
        batch_size=1,
        seed=args.seed,
        rank=local_rank,
        world_size=world_size,
        num_workers=0,
    )
    predopt_history = run_predopt(
        model,
        preopt_loader,
        device,
        dtype,
        epochs=2,
        lr=1e-2,
        chunk=args.predopt_chunk_size,
        world_size=world_size,
    )
    predopt_after = evaluate_predopt_metrics(
        model, predopt_view, device, dtype, args.predopt_chunk_size
    )
    mrks_map = load_mrks_dispersions(args.mrks_dispersions_pickle)
    if any(record["Name"] not in mrks_map for record in vxc_records):
        raise ValueError(
            "Historical mRKS dispersion map does not cover the verified 90 names."
        )
    dispersion_path = Path(args.mrks_dispersions_pickle)
    source_bindings = source_bindings_from_verified_corpus(
        manifest,
        sha256(manifest_path),
        dispersion_identity=dispersion_path.name,
        dispersion_sha256=sha256(dispersion_path),
    )
    protocol = build_lap_s5_protocol()
    provenance = build_lap_s5_provenance(
        h_bohr=manifest["h_bohr"],
        stencil_version=manifest["stencil_version"],
        derivative_order=manifest["derivative_order"],
        dtype=dtype,
        model_kwargs=model.model_kwargs,
        source_bindings=source_bindings,
        schedule=protocol,
    )
    if rank0:
        preopt_path = output / "lap_s5_preoptimized.pt"
        torch.save(model.state_dict(), preopt_path)
        preopt_path.with_suffix(".pt.meta.json").write_text(
            json.dumps(
                {
                    "lap_s5_provenance": provenance,
                    "predopt_history": predopt_history,
                    "predopt_before": predopt_before,
                    "predopt_after": predopt_after,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    preopt_path = output / "lap_s5_preoptimized.pt"

    run_args = argparse.Namespace(
        name=MODEL_NAME,
        model_type="lap",
        dtype=args.dtype,
        dropout=0.0,
        weight_decay=0.01,
        batch_size=1,
        vxc_batch_size=1,
        num_workers_train=2,
        num_workers_vxc=2,
        n_train=500,
        seed=args.seed,
        resume_training_state="",
        training_state_every=args.training_state_every,
        snapshot_every=10,
        snapshot_start_epoch=1,
        convergence_tail_epochs=0,
        convergence_base_epochs=500,
        convergence_tail_start_lr=1e-5,
        convergence_tail_min_lr=1e-7,
        data_protocol=TRAIN_MODE,
        potential_mode=POTENTIAL_MODE,
        point_chunk_size=args.point_chunk_size,
        include_mrks_dispersion=True,
        lap_s5_provenance=provenance,
        lap_checkpoint_extra={"lap_corpus_manifest_sha256": sha256(manifest_path)},
        preopt_vxc_weight=0.0,
        preopt_vxc_steps=0,
        preopt_vxc_target="pbe",
        lr_predopt=1e-2,
        n_predopt=2,
        force_preopt=True,
        shared_preopt_checkpoint=str(preopt_path),
    )
    params = _s5_training_params(protocol)
    result = run_trial(
        trial_number=19,
        params=params,
        args=run_args,
        shared_preopt_checkpoint=preopt_path,
        data_train=grouped,
        data_vxc_train=vxc_records,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        dispersions=load_reaction_dispersions(args.reaction_dispersions_pickle),
        mrks_dispersions=mrks_map,
        output_dir=output,
        rank0=rank0,
    )
    result["predopt_history"] = predopt_history
    result["predopt_before"] = predopt_before
    result["predopt_after"] = predopt_after
    if rank0:
        (output / "lap_s5_production_run.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("pilot-smoke", "production"), default="pilot-smoke"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--canonical-predopt-pickle")
    parser.add_argument("--canonical-manifest")
    parser.add_argument("--reaction-grouped-pickle")
    parser.add_argument("--reaction-manifest")
    parser.add_argument("--stencil-dir")
    parser.add_argument("--system", default="H2", choices=("H2", "BeH2", "CO"))
    parser.add_argument("--npz")
    parser.add_argument("--lap-corpus")
    parser.add_argument("--point-chunk-size", type=int)
    parser.add_argument(
        "--chunk-candidates", default=",".join(map(str, DEFAULT_POINT_CHUNKS))
    )
    parser.add_argument("--predopt-chunk-size", type=int, default=4096)
    parser.add_argument("--steps-per-phase", type=int, default=2)
    parser.add_argument(
        "--scf-runtime", choices=("auto", "native", "wsl"), default="auto"
    )
    parser.add_argument(
        "--mrks-dispersions-pickle", default=str(DEFAULT_MRKS_DISPERSIONS)
    )
    parser.add_argument(
        "--reaction-dispersions-pickle", default=str(DEFAULT_REACTION_DISPERSIONS)
    )
    parser.add_argument("--predopt-epochs", type=int, default=2)
    parser.add_argument("--predopt-lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--training-state-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.predopt_chunk_size <= 0:
        raise SystemExit("--predopt-chunk-size must be positive.")
    if args.mode == "pilot-smoke":
        required = (
            args.canonical_predopt_pickle,
            args.canonical_manifest,
            args.reaction_grouped_pickle,
            args.stencil_dir,
        )
        if not all(required):
            raise SystemExit(
                "pilot-smoke requires canonical Minnesota, reaction subset, and stencil paths."
            )
        if args.steps_per_phase < 1 or args.steps_per_phase > 5:
            raise SystemExit("Phase mechanics smoke is limited to 1–5 steps per phase.")
        run_pilot_smoke(args)
        return
    if not args.lap_corpus:
        raise SystemExit("production mode requires --lap-corpus.")
    if args.point_chunk_size is None or args.point_chunk_size <= 0:
        raise SystemExit(
            "production mode requires an explicit positive --point-chunk-size."
        )
    run_production(args)


if __name__ == "__main__":
    main()
