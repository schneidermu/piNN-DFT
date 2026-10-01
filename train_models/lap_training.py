"""Shared Lap objectives, clean Minnesota predopt, and rank-average updates."""

import torch
from lap_vxc import (
    STENCIL_VERSION,
    full_vxc_loss,
    integrated_energy,
    sigma_standard_to_total,
    stencil_order_for_version,
)
from optuna_joint import batch_exc, batch_fchem, canonical_variant
from predopt_targets import _ADAPTIVE_INDICES, _prepare_predopt_targets
from reaction_energy_calculation import calculate_reaction_energy
from torch.utils.checkpoint import checkpoint


def tensor_record(record, device, dtype):
    return {
        k: v.to(device=device, dtype=dtype)
        if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v
        for k, v in record.items()
    }


def canonical_predopt_view(grouped):
    """Return exactly one stable augmentation variant for each base reaction."""
    if not isinstance(grouped, dict) or len(grouped) != 268:
        raise ValueError("Lap PBE predopt requires all 268 cleaned base reactions.")
    return {
        index: canonical_variant(group)
        for index, group in enumerate(grouped.values())
    }


def reaction_loss(model, reaction, target, device, dtype, dispersions=None):
    """Keep the existing reaction integration, weights, and augmentation protocol."""
    reaction = tensor_record(reaction, device, dtype)
    raw = reaction["Grid"]
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(
            "Minnesota model grid must use the explicit raw nine-column layout."
        )
    if not torch.allclose(raw[:, :2], reaction["Densities"], rtol=1e-6, atol=1e-12):
        raise ValueError("Minnesota rho boundary disagrees with model input.")
    if not minnesota_sigma_boundary_matches(raw[:, 2:5], reaction["Gradients"]):
        raise ValueError(
            "Minnesota standard sigma/model sigma-total boundary disagrees."
        )
    constants = checkpoint(model, raw, use_reentrant=False)
    prediction, _ = calculate_reaction_energy(
        reaction,
        constants,
        device,
        "GGA",
        "PBE",
        dispersions=dispersions,
        return_local_energies=False,
    )
    return batch_fchem(
        reaction["Database"], prediction, target.to(device=device, dtype=dtype)
    )


def minnesota_sigma_boundary_matches(model_sigma, standard_sigma):
    """Allow source-float32 rounding after cancellation in the total sigma.

    The grouped Minnesota artifact stores both the model's total-sigma column
    and standard ``(aa, ab, bb)`` values as float32.  Near cancellation, their
    independent rounding can exceed a small relative comparison even though
    each value is consistent with the original float32 record.  Scale the
    absolute tolerance by the magnitudes of the terms being added so a nearly
    zero total does not get an unrealistically strict relative tolerance.
    """
    reconstructed = sigma_standard_to_total(standard_sigma)
    scale = (
        standard_sigma[..., 0].abs()
        + 2 * standard_sigma[..., 1].abs()
        + standard_sigma[..., 2].abs()
    ).unsqueeze(-1)
    tolerance = 8 * torch.finfo(torch.float32).eps * scale + 1e-12
    return bool(((model_sigma - reconstructed).abs() <= tolerance).all())


def mrks_losses(energy, record, device, dtype, chunk):
    d = tensor_record(record, device, dtype)
    f, weights = d["StencilFeatures"], d["Weights"]
    prediction = integrated_energy(energy, f, weights, chunk)
    # integrated_energy accumulates in float64 even for a float32 model. Keep
    # the preserved float32 legacy scalar's value while matching prediction's
    # dtype for the standard MSE backward path.
    target = d["E_xc"].reshape(1).to(dtype=prediction.dtype)
    exc = batch_exc([d["Name"]], prediction.reshape(1), target)
    stencil_version = d["StencilVersion"]
    stencil_order = d.get("StencilOrder")
    if "StencilOrder" not in d and stencil_version == STENCIL_VERSION:
        stencil_order = stencil_order_for_version(stencil_version)
    vxc = full_vxc_loss(
        energy,
        f,
        d["Vxc"],
        weights,
        d["HBohr"],
        chunk,
        order=stencil_order,
        version=stencil_version,
    )
    return exc, vxc


def average_gradients(model, world_size):
    """Same SUM/world-size semantics as the historical manual objective merger."""
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if world_size > 1:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world_size)
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError("Nonfinite Lap objective gradient.")


def predopt_loss(model, raw):
    predicted = model(raw)[:, _ADAPTIVE_INDICES]
    from dft_functionals import PBE_CONSTANTS

    targets = _prepare_predopt_targets(PBE_CONSTANTS.to(raw), len(raw), raw.device)
    return (predicted - targets).square().mean()


def run_predopt(model, loader, device, dtype, epochs, lr, chunk, world_size=1):
    """Canonical nine adaptive PBE targets; no legacy partial-Vrho objective.

    Return per-epoch MSE/MAE over the grid points visited.  The data loader is
    expected to contain one deterministic canonical variant per base reaction.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        if hasattr(getattr(loader, "sampler", None), "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch_mse = 0.0
        epoch_mae = 0.0
        reaction_count = 0
        for reaction, _ in loader:
            raw = reaction["Grid"].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            reaction_mse = 0.0
            reaction_mae = 0.0
            for start in range(0, len(raw), chunk):
                block = raw[start : start + chunk]
                prediction = model(block)[:, _ADAPTIVE_INDICES]
                from dft_functionals import PBE_CONSTANTS

                target = _prepare_predopt_targets(
                    PBE_CONSTANTS.to(block), len(block), block.device
                )
                difference = prediction - target
                mse = difference.square().mean()
                mae = difference.abs().mean()
                fraction = len(block) / len(raw)
                (mse * fraction).backward()
                reaction_mse += float(mse.detach()) * fraction
                reaction_mae += float(mae.detach()) * fraction
            average_gradients(model, world_size)
            optimizer.step()
            epoch_mse += reaction_mse
            epoch_mae += reaction_mae
            reaction_count += 1
        if reaction_count == 0:
            raise ValueError("PBE predopt received an empty canonical reaction view.")
        history.append(
            {
                "epoch": epoch + 1,
                "mse": epoch_mse / reaction_count,
                "mae": epoch_mae / reaction_count,
                "reactions": reaction_count,
            }
        )
    return history
