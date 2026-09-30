"""Run the local Minnesota-predopt + real mRKS full-Vxc pilot on one CUDA GPU.

This is an explicitly small overfit/pipeline proof.  It does not train a
production model, tune production objective weights, or weaken verify_corpus().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import torch
from dataset import collate_fn, collate_fn_predopt
from lap_checkpoint import checkpoint_payload
from lap_data import read_stencil_h5
from lap_diagnostics import diagnose, measure_objectives
from lap_training import (
    canonical_predopt_view,
    mrks_losses,
    reaction_loss,
    run_predopt,
)
from lap_vxc import LapEnergy
from optuna_joint import EpochSampledAugmentedDataset
from predopt import DatasetPredopt
from predopt_targets import _ADAPTIVE_INDICES, _prepare_predopt_targets
from torch.utils.data import DataLoader
from train_lap import MODEL_NAME, load_reaction_dispersions, make_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_pickle(
    path: Path, expected_count: int, manifest_path: Path | None = None
):
    manifest_path = manifest_path or path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"Missing Minnesota view manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256(path)
    expected_hash = manifest.get("output_sha256")
    pilot_view = manifest.get("pilot_view")
    if (
        expected_hash != actual_hash
        and not (
            expected_count == 1
            and isinstance(pilot_view, dict)
            and pilot_view.get("output_sha256") == actual_hash
        )
    ):
        raise ValueError("Minnesota canonical view hash differs from its manifest.")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or len(data) != expected_count:
        raise ValueError(f"Minnesota view must contain exactly {expected_count} reaction groups.")
    if manifest.get("base_reaction_count") != 268:
        raise ValueError("Minnesota manifest does not establish the 268-group protocol.")
    if any(not isinstance(data[i], (dict, list)) for i in range(expected_count)):
        raise ValueError("Minnesota views must use contiguous integer keys from zero.")
    return data, manifest


def eval_predopt_metrics(model, reactions, device, dtype, chunk):
    from dft_functionals import PBE_CONSTANTS

    totals = torch.zeros(2, dtype=torch.float64, device=device)
    point_count = 0
    model.eval()
    with torch.no_grad():
        for reaction in reactions.values():
            raw = reaction["Grid"]
            if raw.ndim != 2 or raw.shape[1] != 9:
                raise ValueError("Minnesota Lap input must have its raw N×9 layout.")
            for start in range(0, len(raw), chunk):
                block = raw[start : start + chunk].to(device=device, dtype=dtype)
                prediction = model(block)[:, _ADAPTIVE_INDICES]
                target = _prepare_predopt_targets(
                    PBE_CONSTANTS.to(block), len(block), block.device
                )
                difference = prediction - target
                totals[0] += difference.double().square().sum()
                totals[1] += difference.double().abs().sum()
                point_count += len(block)
    denom = point_count * len(_ADAPTIVE_INDICES)
    totals = totals.cpu()
    return {"mse": float(totals[0] / denom), "mae": float(totals[1] / denom), "grid_points": point_count}


def cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).casefold()
    )


def recover_oom(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def objective_report(model, energy, reaction, target, record, device, dtype, chunk, dispersions):
    while True:
        try:
            values = measure_objectives(
                model, energy, reaction, target, record, device, dtype, chunk, dispersions
            )
            diagnostic = diagnose(energy, record, device, dtype, chunk)
            return values, diagnostic, chunk
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not cuda_oom(exc) or chunk <= 1:
                raise
            recover_oom(device)
            chunk = max(1, chunk // 2)
            print(f"CUDA OOM in pilot diagnostics; retrying with point chunk {chunk}.", flush=True)


def gradient_norm(model):
    return sum(
        float(parameter.grad.detach().double().square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    ) ** 0.5


def train_pilot_step(
    model, energy, reaction_data, mrks_record, optimizer, weights, device, dtype, chunk, dispersions
):
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            reaction, target = reaction_data
            reaction_value = reaction_loss(model, reaction, target, device, dtype, dispersions)
            (weights[0] * reaction_value).backward()
            exc_value, vxc_value = mrks_losses(energy, mrks_record, device, dtype, chunk)
            (weights[1] * exc_value).backward()
            (weights[2] * vxc_value).backward()
            norm = gradient_norm(model)
            if not torch.isfinite(torch.tensor(norm)) or norm == 0:
                raise FloatingPointError("Pilot parameter gradient is zero or nonfinite.")
            if not all(torch.isfinite(value) for value in (reaction_value, exc_value, vxc_value)):
                raise FloatingPointError("Pilot objective loss is nonfinite.")
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            update_norm = sum(
                float((parameter.detach() - original).double().square().sum())
                for parameter, original in zip(model.parameters(), before)
            ) ** 0.5
            if not torch.isfinite(torch.tensor(update_norm)) or update_norm == 0:
                raise FloatingPointError("Pilot optimizer step did not update parameters.")
            return {
                "reaction_loss": float(reaction_value.detach()),
                "E_xc_loss": float(exc_value.detach()),
                "full_Vxc_loss": float(vxc_value.detach()),
                "gradient_norm": norm,
                "parameter_update_norm": update_norm,
            }, chunk
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            optimizer.zero_grad(set_to_none=True)
            if not cuda_oom(exc) or chunk <= 1:
                raise
            recover_oom(device)
            chunk = max(1, chunk // 2)
            print(f"CUDA OOM in pilot step; retrying this step with point chunk {chunk}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-predopt-pickle", required=True)
    parser.add_argument("--canonical-manifest")
    parser.add_argument("--reaction-grouped-pickle", required=True)
    parser.add_argument("--reaction-manifest")
    parser.add_argument("--stencil-dir", required=True)
    parser.add_argument("--system", default="H2")
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predopt-epochs", type=int, default=2)
    parser.add_argument("--predopt-lr", type=float, default=1e-2)
    parser.add_argument("--predopt-chunk", type=int, default=4096)
    parser.add_argument("--pilot-chunk", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reaction-weight", type=float, default=1e-4)
    parser.add_argument("--exc-weight", type=float, default=1.0)
    parser.add_argument("--vxc-weight", type=float, default=1.0)
    parser.add_argument("--reaction-dispersions", required=True)
    parser.add_argument("--log-every", type=int, default=2)
    args = parser.parse_args()
    if min(args.predopt_chunk, args.pilot_chunk, args.steps, args.predopt_epochs) <= 0:
        parser.error("Chunk sizes, predopt epochs, and training steps must be positive.")
    if args.log_every <= 0:
        parser.error("--log-every must be positive.")

    predopt_grouped, mn_manifest = load_verified_pickle(
        Path(args.canonical_predopt_pickle),
        268,
        Path(args.canonical_manifest) if args.canonical_manifest else None,
    )
    # The streamed artifact keeps the repository's grouped-dataset shape
    # (one-element list per canonical group); predopt consumes the existing
    # flat one-record-per-reaction view produced by load_minnesota().
    predopt = canonical_predopt_view(predopt_grouped)
    grouped, reaction_manifest = load_verified_pickle(
        Path(args.reaction_grouped_pickle),
        1,
        Path(args.reaction_manifest) if args.reaction_manifest else None,
    )
    # The pilot grouped view contains exactly one base reaction but retains its
    # eight augmentation variants, so the repository's epoch-wise sampling is
    # exercised unchanged within that deterministic subset.
    if len(grouped) != 1 or not isinstance(grouped[0], list) or not grouped[0]:
        raise ValueError("The local joint pilot view must retain variants for one base group.")
    if args.system not in {p.stem for p in Path(args.stencil_dir).glob("*.h5")}:
        raise ValueError(f"No real stencil H5 for selected system {args.system}.")
    stencil_path = Path(args.stencil_dir) / f"{args.system}.h5"
    mrks_record = read_stencil_h5(stencil_path)
    if mrks_record["SourceProvenance"].get("E_xc_source") != "preserved legacy mRKS training target":
        raise ValueError("mRKS pilot record does not preserve the legacy E_xc target.")
    if args.steps > 100:
        raise ValueError("This runner is restricted to a <=100-step local pilot.")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output {output}.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        free_vram = total_vram = None
        gpu_name = None
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = make_model(args.name, device, dtype)
    model.train()
    pbe_before = eval_predopt_metrics(model, predopt, device, dtype, args.predopt_chunk)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    pre_loader = DataLoader(
        DatasetPredopt(predopt), batch_size=1, shuffle=False, collate_fn=collate_fn_predopt
    )
    predopt_chunk = args.predopt_chunk
    while True:
        try:
            model.train()
            predopt_history = run_predopt(
                model,
                pre_loader,
                device,
                dtype,
                args.predopt_epochs,
                args.predopt_lr,
                predopt_chunk,
            )
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not cuda_oom(exc) or predopt_chunk <= 1:
                raise
            model.load_state_dict(initial_state)
            recover_oom(device)
            predopt_chunk = max(1, predopt_chunk // 2)
            print(f"CUDA OOM in PBE predopt; restarting from fresh initialization with chunk {predopt_chunk}.", flush=True)
    pbe_after = eval_predopt_metrics(model, predopt, device, dtype, predopt_chunk)
    energy = LapEnergy(model)
    model.train()
    reaction_dataset = EpochSampledAugmentedDataset(grouped, args.seed)
    reaction_loader = DataLoader(
        reaction_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    reaction_dataset.resample(0)
    initial_reaction = next(iter(reaction_loader))
    dispersions = load_reaction_dispersions(args.reaction_dispersions)
    initial_objectives, initial_diagnostic, chunk = objective_report(
        model,
        energy,
        initial_reaction[0],
        initial_reaction[1],
        mrks_record,
        device,
        dtype,
        args.pilot_chunk,
        dispersions,
    )
    weights = [args.reaction_weight, args.exc_weight, args.vxc_weight]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for step in range(args.steps):
        reaction_dataset.resample(step)
        reaction_data = next(iter(reaction_loader))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        values, chunk = train_pilot_step(
            model,
            energy,
            reaction_data,
            mrks_record,
            optimizer,
            weights,
            device,
            dtype,
            chunk,
            dispersions,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            values["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            values["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
            values["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            values["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        values["step"] = step + 1
        values["time_seconds"] = time.perf_counter() - start
        values["point_chunk"] = chunk
        history.append(values)
        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.steps:
            print(json.dumps(values, sort_keys=True), flush=True)

    final_objectives, final_diagnostic, chunk = objective_report(
        model,
        energy,
        reaction_data[0],
        reaction_data[1],
        mrks_record,
        device,
        dtype,
        chunk,
        dispersions,
    )
    output.mkdir(parents=True)
    checkpoint_path = output / "lap_real_pilot.pt"
    torch.save(
        checkpoint_payload(
            model,
            epoch=args.steps,
            precision=args.dtype,
            h_bohr=mrks_record["HBohr"],
            optimizer_state_dict=optimizer.state_dict(),
            objective_weights=weights,
            pilot_only=True,
            optimizer_steps=args.steps,
            pilot_system=args.system,
        ),
        checkpoint_path,
    )
    report = {
        "pilot_only": True,
        "branch_commit_at_run": None,
        "gpu": gpu_name,
        "cuda_total_vram_bytes": total_vram,
        "cuda_free_vram_at_start_bytes": free_vram,
        "dtype": args.dtype,
        "point_chunk_size_final": chunk,
        "pilot_h_bohr": mrks_record["HBohr"],
        "mrks_system": args.system,
        "mrks_stencil_h5_sha256": sha256(stencil_path),
        "npz_sha256": mrks_record["SourceProvenance"]["npz_sha256"],
        "legacy_target_sha256": mrks_record["SourceProvenance"]["legacy_target_sha256"],
        "minnesota_source_sha256": mn_manifest["source_sha256"],
        "minnesota_canonical_view": mn_manifest,
        "reaction_pilot_view": reaction_manifest,
        "reaction_pilot_base_groups": 1,
        "predopt_canonical_view": "one default variant if present; otherwise lexical minimum per group; source audit yields level2 in all groups",
        "predopt_metrics_before": pbe_before,
        "predopt_metrics_after": pbe_after,
        "predopt_history": predopt_history,
        "predopt_epochs": args.predopt_epochs,
        "predopt_lr": args.predopt_lr,
        "objectives_after_predopt": initial_objectives,
        "diagnostics_after_predopt": initial_diagnostic,
        "objectives_after_joint_pilot": final_objectives,
        "diagnostics_after_joint_pilot": final_diagnostic,
        "pilot_loss_weights": {
            "reaction": args.reaction_weight,
            "E_xc": args.exc_weight,
            "full_Vxc": args.vxc_weight,
            "label": "pilot-only explicit weights; not production recommendations",
        },
        "optimizer_lr": args.lr,
        "optimizer_steps": args.steps,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_gradient_norm_after_predopt": {
            name: value["parameter_gradient_norm"] for name, value in initial_objectives.items()
        },
        "parameter_gradient_norm_final": history[-1]["gradient_norm"],
        "all_history_finite": all(
            torch.isfinite(torch.tensor(v)).item()
            for row in history
            for key, v in row.items()
            if isinstance(v, (int, float))
        ),
    }
    (output / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pilot checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved pilot report: {output / 'pilot_report.json'}", flush=True)


if __name__ == "__main__":
    main()
