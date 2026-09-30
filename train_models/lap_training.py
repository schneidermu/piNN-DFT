"""Shared Lap objectives, clean Minnesota predopt, and rank-average updates."""

import torch
from lap_vxc import full_vxc_loss, integrated_energy, sigma_standard_to_total
from optuna_joint import batch_exc, batch_fchem
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
    if not torch.allclose(
        raw[:, 2:5],
        sigma_standard_to_total(reaction["Gradients"]),
        rtol=1e-6,
        atol=1e-12,
    ):
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


def mrks_losses(energy, record, device, dtype, chunk):
    d = tensor_record(record, device, dtype)
    f, weights = d["StencilFeatures"], d["Weights"]
    prediction = integrated_energy(energy, f, weights, chunk)
    exc = batch_exc([d["Name"]], prediction.reshape(1), d["E_xc"].reshape(1))
    vxc = full_vxc_loss(energy, f, d["Vxc"], weights, d["HBohr"], chunk)
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
    """Canonical nine adaptive PBE targets; no legacy partial-Vrho objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if hasattr(getattr(loader, "sampler", None), "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for reaction, _ in loader:
            raw = reaction["Grid"].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(raw), chunk):
                block = raw[start : start + chunk]
                (predopt_loss(model, block) * len(block) / len(raw)).backward()
            average_gradients(model, world_size)
            optimizer.step()
