"""Read-only Euler/precision diagnostics and objective calibration.

No checkpoint selection, validation split, gauge projection, or production loss
weights are inferred. Calibration may run explicitly requested clean-pool predopt
in memory and reports fresh/after-predopt losses and parameter gradient norms.
"""

import argparse
import json

import torch
from dataset import collate_fn, collate_fn_predopt
from lap_checkpoint import load_lap_checkpoint
from lap_data import read_stencil_h5, verify_corpus
from lap_training import mrks_losses, reaction_loss, run_predopt, tensor_record
from lap_vxc import LapEnergy, euler_components
from optuna_joint import EpochSampledAugmentedDataset
from predopt import DatasetPredopt
from train_lap import (
    DEFAULT_REACTION_DISPERSIONS,
    MODEL_NAME,
    load_minnesota,
    load_reaction_dispersions,
    loader_for,
    make_model,
)


def diagnose(energy, record, device, dtype, chunk):
    d = tensor_record(record, device, dtype)
    sums = {
        key: 0.0
        for key in (
            "error2",
            "abs_error",
            "error",
            "energy",
            "C",
            "minus_div_A",
            "lap_B",
        )
    }
    max_error, nonfinite, total = 0.0, 0, 0
    f, w = d["StencilFeatures"], d["Weights"]
    q = f[:, 0, :2].double() * w.double()[:, None]
    norm = float(q.sum())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(f), chunk):
        sl = slice(start, start + chunk)
        parts = euler_components(energy, f[sl], d["HBohr"], False)
        error = parts["Vxc"].detach() - d["Vxc"][sl]
        sums["error2"] += float((q[sl] * error.square()).sum())
        sums["abs_error"] += float((q[sl] * error.abs()).sum())
        sums["error"] += float((q[sl] * error).sum())
        sums["energy"] += float((parts["energy"].detach().double() * w[sl]).sum())
        max_error = max(max_error, float(error.abs().max()))
        for key in ("C", "minus_div_A", "lap_B"):
            sums[key] += float((q[sl] * parts[key].detach().square()).sum())
        for val in parts.values():
            nonfinite += int((~torch.isfinite(val)).sum())
            total += val.numel()
    components = {k: (sums[k] / norm) ** 0.5 for k in ("C", "minus_div_A", "lap_B")}
    scale = sum(components.values())
    return {
        "system": d["Name"],
        "electron_number": norm,
        "exc_target_hartree": float(d["E_xc"]),
        "exc_prediction_hartree": sums["energy"],
        "vxc_weighted_rmse": (sums["error2"] / norm) ** 0.5,
        "vxc_weighted_mae": sums["abs_error"] / norm,
        "vxc_weighted_bias": sums["error"] / norm,
        "max_abs_vxc_error": max_error,
        "component_weighted_rms": components,
        "component_relative_rms": {
            k: v / scale if scale else 0.0 for k, v in components.items()
        },
        "h_bohr": d["HBohr"],
        "dtype": str(dtype),
        "nonfinite_fraction": nonfinite / total,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }


def convergence():
    # Deliberately independent of training data and production h selection.
    from lap_analytic import PolynomialEnergy, features

    result = []
    for dtype in (torch.float32, torch.float64):
        energy = PolynomialEnergy(b=0.3, c=0.2, dtype=dtype)
        for h in (0.2, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0001):
            f64 = features(h)
            exact = 1.4 * f64[:, 0, :2] - f64[:, 0, 8:]
            parts = euler_components(energy, f64.to(dtype), h, False)
            result.append(
                {
                    "dtype": str(dtype),
                    "h_bohr": h,
                    "max_abs_error": float((parts["Vxc"] - exact).abs().max()),
                }
            )
    return result


def measure_objectives(
    model, energy, reaction, target, record, device, dtype, chunk, dispersions=None
):
    losses = [
        reaction_loss(model, reaction, target, device, dtype, dispersions),
        *mrks_losses(energy, record, device, dtype, chunk),
    ]
    report = {}
    for key, loss in zip(("reaction", "mrks_exc", "full_mrks_vxc"), losses):
        grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        norm = (
            sum(
                float(g.detach().double().square().sum())
                for g in grads
                if g is not None
            )
            ** 0.5
        )
        report[key] = {"loss": float(loss.detach()), "parameter_gradient_norm": norm}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["convergence", "system", "calibrate"])
    p.add_argument("--corpus")
    p.add_argument("--record-h5")
    p.add_argument("--checkpoint")
    p.add_argument("--system-index", type=int, default=0)
    p.add_argument("--name", default=MODEL_NAME)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--point-chunk-size", type=int, default=4096)
    p.add_argument("--predopt-epochs", type=int, default=2)
    p.add_argument("--predopt-lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reaction-dispersions", default=str(DEFAULT_REACTION_DISPERSIONS))
    p.add_argument("--no-reaction-dispersion", action="store_true")
    args = p.parse_args()
    if args.point_chunk_size <= 0 or args.predopt_epochs < 0:
        p.error("Positive chunk size and nonnegative predopt epochs required.")
    if args.mode == "convergence":
        print(json.dumps(convergence(), indent=2))
        return
    if args.corpus:
        _, records = verify_corpus(args.corpus)
        record = records[args.system_index]
    elif args.record_h5 and args.mode == "system":
        record = read_stencil_h5(args.record_h5)
    else:
        p.error(
            "A verified corpus is required for calibration; system mode also accepts a provenance-complete H5."
        )
    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if args.checkpoint:
        model, _ = load_lap_checkpoint(args.checkpoint, device, dtype)
    else:
        model = make_model(args.name, device, dtype)
    energy = LapEnergy(model)
    if args.mode == "system":
        report = diagnose(energy, record, device, dtype, args.point_chunk_size)
    else:
        if args.checkpoint:
            p.error(
                "Fresh/after-predopt calibration requires fresh initialization; omit --checkpoint."
            )
        grouped, flat = load_minnesota(args.corpus)
        dispersions = load_reaction_dispersions(
            args.reaction_dispersions, args.no_reaction_dispersion
        )
        train = EpochSampledAugmentedDataset(grouped, args.seed)
        reaction_loader = loader_for(train, 0, 1, args.seed, collate_fn)
        reaction, target = next(iter(reaction_loader))
        fresh = measure_objectives(
            model,
            energy,
            reaction,
            target,
            record,
            device,
            dtype,
            args.point_chunk_size,
            dispersions,
        )
        pre_loader = loader_for(
            DatasetPredopt(flat), 0, 1, args.seed, collate_fn_predopt
        )
        run_predopt(
            model,
            pre_loader,
            device,
            dtype,
            args.predopt_epochs,
            args.predopt_lr,
            args.point_chunk_size,
        )
        report = {
            "fresh_initialization": fresh,
            "after_predopt": measure_objectives(
                model,
                energy,
                reaction,
                target,
                record,
                device,
                dtype,
                args.point_chunk_size,
                dispersions,
            ),
            "reaction_sample_seed": args.seed,
            "mrks_system": record["Name"],
            "notice": "Single reproducible training sample diagnostic; no final loss weights or checkpoint selection.",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
