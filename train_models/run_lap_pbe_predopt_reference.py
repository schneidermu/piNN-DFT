"""Run the fresh canonical Minnesota PBE warm start for stencil diagnostics.

This command performs only the frozen two-epoch PBE predopt. It does not run
reaction/Vxc objectives, an S5 phase, or production training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from dataset import collate_fn_predopt
from lap_checkpoint import checkpoint_payload
from lap_training import canonical_predopt_view, run_predopt
from optuna_joint import set_random_seed
from predopt import DatasetPredopt
from torch.utils.data import DataLoader
from train_lap_s5 import (
    _pilot_model,
    evaluate_predopt_metrics,
    load_verified_minnesota_view,
    resolve_device,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-pickle", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source = args.canonical_pickle.resolve()
    manifest = args.canonical_manifest.resolve()

    device = resolve_device(args.device)
    dtype = torch.float32
    set_random_seed(args.seed)
    grouped, _manifest_data = load_verified_minnesota_view(source, manifest, 268)
    view = canonical_predopt_view(grouped)
    if len(view) != 268:
        raise ValueError(f"Expected 268 canonical predopt reactions, got {len(view)}")
    loader = DataLoader(
        DatasetPredopt(view),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_predopt,
        num_workers=0,
    )

    model = _pilot_model(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    before = evaluate_predopt_metrics(model, view, device, dtype, args.chunk_size)
    history = run_predopt(
        model,
        loader,
        device,
        dtype,
        epochs=2,
        lr=1e-2,
        chunk=args.chunk_size,
        world_size=1,
    )
    after = evaluate_predopt_metrics(model, view, device, dtype, args.chunk_size)
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
    else:
        memory = None
    payload = checkpoint_payload(
        model,
        predopt_only=True,
        predopt_source_sha256=sha256(source),
        predopt_manifest_sha256=sha256(manifest),
        predopt_epochs=2,
        predopt_lr=1e-2,
        predopt_seed=args.seed,
        predopt_history=history,
    )
    checkpoint = output_dir / "lap_pbe_predopt.pt"
    torch.save(payload, checkpoint)
    report = {
        "diagnostic_only": True,
        "training_objective": "canonical PBE adaptive-parameter predopt only",
        "no_reaction_or_mrks_objective": True,
        "base_reactions": 268,
        "canonical_variant_rule": "default when present, otherwise stable repository canonical_variant",
        "source_pickle": str(source),
        "source_sha256": sha256(source),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "chunk_size": args.chunk_size,
        "epochs": 2,
        "learning_rate": 1e-2,
        "seed": args.seed,
        "predopt_metrics_before": before,
        "predopt_history": history,
        "predopt_metrics_after": after,
        "elapsed_seconds": elapsed,
        "cuda_memory": memory,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (output_dir / "lap_pbe_predopt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
