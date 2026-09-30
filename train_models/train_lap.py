"""Explicit Lap/full-Vxc training entry point; no inherited S5 loss weights.

Run only with a verified reference-stencil corpus. Calibration is a separate
read-only command. Under torchrun, one system per rank and manual rank-average
gradients preserve the repository's distributed objective-merging semantics.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
from dataset import collate_fn, collate_fn_predopt
from lap_checkpoint import checkpoint_payload
from lap_data import DEFAULT_CORPUS, verify_corpus
from lap_training import average_gradients, mrks_losses, reaction_loss, run_predopt
from lap_vxc import LapEnergy
from NN_models_lap import pcPBELMLOptimizerV2Lap
from optuna_joint import EpochSampledAugmentedDataset, parse_model_name
from predopt import DatasetPredopt
from torch.utils.data import DataLoader, DistributedSampler

MODEL_NAME = "PBE-Lap-LGxGc_6_32"
DEFAULT_REACTION_DISPERSIONS = (
    Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle"
)


def load_reaction_dispersions(path, disabled=False):
    if disabled:
        return {}
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def make_model(name, device, dtype):
    if not name.startswith("PBE-Lap-"):
        raise ValueError("Explicit PBE-Lap model identifier required.")
    layers, hidden, gx, gc = parse_model_name(name)
    return pcPBELMLOptimizerV2Lap(layers, hidden, use_g_x=gx, use_g_c=gc).to(
        device=device, dtype=dtype
    )


def load_minnesota(directory):
    with (Path(directory) / "data_train_grouped.pickle").open("rb") as f:
        grouped = pickle.load(f)
    with (Path(directory) / "data_predopt.pickle").open("rb") as f:
        flat = pickle.load(f)
    if len(grouped) != 268:
        raise ValueError(
            "Lap training requires all 268 cleaned Minnesota base reactions."
        )
    return grouped, flat


def loader_for(dataset, rank, world, seed, collate):
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed
    )
    return DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate)


def epoch_steps(
    model,
    energy,
    reaction_loader,
    records,
    indices,
    optimizer,
    weights,
    device,
    dtype,
    chunk,
    accum_iter,
    world,
    dispersions=None,
):
    """Chunk count never affects optimizer cadence or system normalization."""
    count = max(len(reaction_loader), len(indices))

    def repeat(iterable):
        while True:
            yield from iterable

    reactions, systems = repeat(reaction_loader), repeat(indices)
    optimizer.zero_grad(set_to_none=True)
    history = []
    for step in range(count):
        batch, target = next(reactions)
        reaction = reaction_loss(model, batch, target, device, dtype, dispersions)
        (weights[0] * reaction / accum_iter).backward()
        exc, vxc = mrks_losses(energy, records[next(systems)], device, dtype, chunk)
        (weights[1] * exc / accum_iter).backward()
        (weights[2] * vxc / accum_iter).backward()
        history.append([float(x.detach()) for x in (reaction, exc, vxc)])
        if (step + 1) % accum_iter == 0 or step + 1 == count:
            average_gradients(model, world)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--name", default=MODEL_NAME)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--reaction-weight", type=float, required=True)
    p.add_argument("--exc-weight", type=float, required=True)
    p.add_argument("--vxc-weight", type=float, required=True)
    p.add_argument("--dtype", choices=["float32", "float64"], required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--predopt-epochs", type=int, default=2)
    p.add_argument("--predopt-lr", type=float, default=1e-3)
    p.add_argument("--point-chunk-size", type=int, default=4096)
    p.add_argument("--accum-iter", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reaction-dispersions", default=str(DEFAULT_REACTION_DISPERSIONS))
    p.add_argument("--no-reaction-dispersion", action="store_true")
    args = p.parse_args()
    if (
        min(args.epochs, args.point_chunk_size, args.accum_iter) <= 0
        or args.predopt_epochs < 0
    ):
        p.error(
            "Epoch/chunk/accumulation counts must be positive; predopt epochs nonnegative."
        )
    weights = [args.reaction_weight, args.exc_weight, args.vxc_weight]
    if not all(torch.isfinite(torch.tensor(w)) and w > 0 for w in weights):
        p.error("All three objective weights must be explicitly positive and finite.")
    manifest, records = verify_corpus(args.corpus)  # before any model or optimizer
    if any(d["TargetKind"] != "common-rks" or d["SourceSpin"] != 0 for d in records):
        raise ValueError(
            "Current experiment is RKS-only; schema preserves general spin targets for later use."
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an experiment directory.")
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group(
            "nccl" if device.type == "cuda" else "gloo"
        )
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model = make_model(args.name, device, dtype)
    if world > 1:
        for value in model.state_dict().values():
            torch.distributed.broadcast(value, src=0)
    grouped, flat = load_minnesota(args.corpus)
    dispersions = load_reaction_dispersions(
        args.reaction_dispersions, args.no_reaction_dispersion
    )
    train = EpochSampledAugmentedDataset(grouped, args.seed)
    reaction_loader = loader_for(train, rank, world, args.seed, collate_fn)
    pre_loader = loader_for(
        DatasetPredopt(flat), rank, world, args.seed, collate_fn_predopt
    )
    system_sampler = DistributedSampler(
        records, num_replicas=world, rank=rank, seed=args.seed
    )
    run_predopt(
        model,
        pre_loader,
        device,
        dtype,
        args.predopt_epochs,
        args.predopt_lr,
        args.point_chunk_size,
        world,
    )
    energy = LapEnergy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if rank == 0:
        output.mkdir(parents=True)
        (output / "run.json").write_text(
            json.dumps({"arguments": vars(args), "corpus_manifest": manifest}, indent=2)
            + "\n"
        )
    for epoch in range(args.epochs):
        train.resample(epoch)
        reaction_loader.sampler.set_epoch(epoch)
        system_sampler.set_epoch(epoch)
        history = epoch_steps(
            model,
            energy,
            reaction_loader,
            records,
            list(system_sampler),
            optimizer,
            weights,
            device,
            dtype,
            args.point_chunk_size,
            args.accum_iter,
            world,
            dispersions,
        )
        if rank == 0:
            torch.save(
                checkpoint_payload(
                    model,
                    epoch=epoch + 1,
                    precision=args.dtype,
                    h_bohr=manifest["h_bohr"],
                    optimizer_state_dict=optimizer.state_dict(),
                    objective_weights=weights,
                ),
                output / f"epoch_{epoch + 1:04d}.pt",
            )
            (output / f"train_{epoch + 1:04d}.json").write_text(
                json.dumps(history) + "\n"
            )
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
