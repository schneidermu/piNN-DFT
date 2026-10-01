"""Focused regression tests for the shared Lap/full-Vxc S5 adapters."""

import lap_vxc
import optuna_joint as training
import pytest
import torch
from lap_data import PROTOCOL
from NN_models_lap import pcPBELMLOptimizerV2Lap
from torch import nn


def _small_lap_model():
    torch.manual_seed(41)
    return pcPBELMLOptimizerV2Lap(
        num_layers=2, h_dim=8, use_g_x=True, use_g_c=True
    ).double()


def _stencil_features(n=1):
    """Small smooth, closed-shell tensors in the real (N,7,10) schema."""
    centers = torch.linspace(0.25, 0.75, n * 3, dtype=torch.float64).reshape(n, 3)
    offsets = torch.zeros((7, 3), dtype=torch.float64)
    offsets[1, 0], offsets[2, 0] = 0.04, -0.04
    offsets[3, 1], offsets[4, 1] = 0.04, -0.04
    offsets[5, 2], offsets[6, 2] = 0.04, -0.04
    xyz = centers[:, None, :] + offsets[None, :, :]

    amplitude = torch.tensor([0.05, 0.05], dtype=torch.float64)
    cosine_sum = xyz.cos().sum(dim=-1, keepdim=True)
    rho = 0.8 + cosine_sum * amplitude[None, None, :]
    grad = -xyz.sin().unsqueeze(-2) * amplitude[None, None, :, None]
    lapl = -cosine_sum * amplitude[None, None, :]
    return torch.cat([rho, grad.flatten(-2), lapl], dim=-1)


def _lap_record(name="H2", n=1):
    return {
        "Name": name,
        "StencilFeatures": _stencil_features(n),
        "Weights": torch.full((n,), 0.7, dtype=torch.float64),
        "Vxc": torch.zeros((n, 2), dtype=torch.float64),
        "E_xc": torch.tensor(1.0, dtype=torch.float64),
        "HBohr": 0.04,
        "Protocol": PROTOCOL,
        "StencilVersion": lap_vxc.STENCIL_VERSION,
        "SourceProvenance": {"source": "synthetic-test-record"},
    }


def _lap_batch(record):
    return {"LapRecords": [record], "Names": [record["Name"]]}


def test_batch_fchem_matches_legacy_database_weighted_rmse_formula():
    bases = ["ABDE4", "ABDE4", "NCCE31"]
    predictions = torch.tensor([0.0, 2.0, 0.5], dtype=torch.float64)
    references = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)

    actual = training.batch_fchem(bases, predictions, references)
    abde4 = torch.sqrt(torch.tensor(0.5, dtype=torch.float64))
    ncce31 = torch.tensor(0.5, dtype=torch.float64)
    factors = {
        db: training.FCHEM_DB_WEIGHTS[db]
        * training.FREQ_WEIGHTS[db]
        / training.MEAN_WEIGHT
        for db in ("ABDE4", "NCCE31")
    }
    expected = (factors["ABDE4"] * abde4 + factors["NCCE31"] * ncce31) / 2
    torch.testing.assert_close(actual, expected)


def test_reported_fchem_uses_historical_per_database_rmse_weights():
    total, per_database = training.compute_fchem_from_errors(
        {"DBH76": [3.0, 4.0], "NCCE31": [6.0, 8.0]}
    )

    assert per_database == {"DBH76": 3.5355339059327378, "NCCE31": 7.0710678118654755}
    expected = (
        training.FCHEM_DB_WEIGHTS["DBH76"] * per_database["DBH76"]
        + training.FCHEM_DB_WEIGHTS["NCCE31"] * per_database["NCCE31"]
    )
    assert total == pytest.approx(expected)


def test_batch_exc_keeps_system_rmse_in_kcal_per_mol():
    names = ["H2", "H2", "CO"]
    predictions = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    references = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

    actual = training.batch_exc(names, predictions, references)
    expected_h2_rmse = (2.5 + 1e-20) ** 0.5
    expected_co_rmse = (9.0 + 1e-20) ** 0.5
    expected = training.HARTREE2KCAL * (expected_h2_rmse + expected_co_rmse) / 2
    assert actual.item() == pytest.approx(expected)


def test_full_euler_exc_adds_exact_name_dispersion_once(monkeypatch):
    record = _lap_record()
    model = _small_lap_model()
    calls = []
    original_add = training._add_mrks_dispersion_once

    def count_dispersion_addition(prediction, name, dispersions, enabled):
        calls.append(name)
        return original_add(prediction, name, dispersions, enabled)

    monkeypatch.setattr(
        training, "_add_mrks_dispersion_once", count_dispersion_addition
    )
    monkeypatch.setattr(
        lap_vxc,
        "integrated_energy",
        lambda energy, features, weights, point_chunk_size: torch.tensor(
            2.0, dtype=torch.float64
        ),
    )
    # The adapter still validates the actual model and record; only energy
    # integration is replaced to make the one-addition arithmetic transparent.
    loss, predicted, target = training.exc_loss(
        model,
        _lap_batch(record),
        torch.device("cpu"),
        dispersions={"H2": 0.25, "h2": 9.0},
        include_mrks_dispersion=True,
        potential_mode="full_euler",
        point_chunk_size=1,
    )

    torch.testing.assert_close(predicted, torch.tensor([2.25], dtype=torch.float64))
    torch.testing.assert_close(target, torch.tensor([1.0], dtype=torch.float64))
    assert loss.item() == pytest.approx(training.HARTREE2KCAL * 1.25)
    assert calls == ["H2"]


def test_public_vxc_adapter_uses_full_euler_stencil_loss_and_backpropagates():
    model = _small_lap_model()
    batch = _lap_batch(_lap_record(n=1))

    loss = training.vxc_loss(
        model,
        batch,
        torch.device("cpu"),
        potential_mode="full_euler",
        point_chunk_size=1,
    )
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)

    assert torch.isfinite(loss)
    present = [gradient for gradient in gradients if gradient is not None]
    assert present
    assert all(torch.isfinite(gradient).all() for gradient in present)
    assert sum(float(gradient.abs().sum()) for gradient in present) > 0


def test_closed_shell_two_spin_potential_loss_equals_scalar_rho_weighted_mse():
    rho_scalar = torch.tensor([0.6, 1.5], dtype=torch.float64)
    weights = torch.tensor([0.2, 2.0], dtype=torch.float64)
    prediction_scalar = torch.tensor([2.0, -1.0], dtype=torch.float64)
    target_scalar = torch.tensor([1.0, 0.5], dtype=torch.float64)
    rho = rho_scalar[:, None].repeat(1, 2)
    prediction = prediction_scalar[:, None].repeat(1, 2)
    target = target_scalar[:, None].repeat(1, 2)

    actual = lap_vxc.potential_loss(prediction, target, rho, weights)
    expected = (
        weights * rho_scalar * (prediction_scalar - target_scalar).square()
    ).sum() / (weights * rho_scalar).sum()
    torch.testing.assert_close(actual, expected)


def test_full_euler_fails_closed_for_partial_data_and_non_lap_model():
    partial_batch = {
        "Grid": torch.zeros((1, 11), dtype=torch.float64),
        "Vrho": torch.zeros(1, dtype=torch.float64),
        "Weights": torch.ones(1, dtype=torch.float64),
        "Names": ["H2"],
    }
    with pytest.raises(ValueError, match="full-stencil"):
        training.vxc_loss(
            _small_lap_model(),
            partial_batch,
            torch.device("cpu"),
            potential_mode="full_euler",
            point_chunk_size=1,
        )

    with pytest.raises(ValueError, match="requires pcPBELMLOptimizerV2Lap"):
        training.vxc_loss(
            nn.Linear(9, 29).double(),
            _lap_batch(_lap_record()),
            torch.device("cpu"),
            potential_mode="full_euler",
            point_chunk_size=1,
        )

    with pytest.raises(ValueError, match="full_vxc_loss"):
        training.vxc_loss(
            _small_lap_model(),
            partial_batch,
            torch.device("cpu"),
            potential_mode="partial_vrho",
        )


def test_s5_objective_gradient_clip_and_scale_remain_objective_local():
    parameter = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    loss = (parameter * torch.tensor([3.0, 4.0], dtype=parameter.dtype)).sum()

    gradients = training.prepare_objective_gradients(
        [parameter],
        loss,
        merge_strategy="clip_then_sum",
        grad_clip=2.0,
        grad_scale=0.5,
    )

    torch.testing.assert_close(
        gradients[0], torch.tensor([0.6, 0.8], dtype=parameter.dtype)
    )
    assert training.OMEGA == 0.5


def test_partial_accumulation_window_only_changes_full_euler_divisor():
    assert training.accumulation_divisor(3, 5, 2, "full_euler") == 2
    assert training.accumulation_divisor(4, 5, 2, "full_euler") == 1
    # Keep the historical full accumulation divisor on the legacy route.
    assert training.accumulation_divisor(4, 5, 2, "partial_vrho") == 2


def test_shared_lap_engine_optimizer_cadence_counts_final_partial_window(monkeypatch):
    model = nn.Linear(1, 1, bias=False).double()
    model.descriptor_protocol = "rho-sigma-total-lapl-tau-free-v1"
    monkeypatch.setattr(training, "_require_lap_model", lambda candidate: candidate)

    def reaction_energy(reaction, predictions, device, **kwargs):
        return predictions[:, 0], None

    def vxc_objective(model, batch, device, **kwargs):
        return model.weight.square().mean()

    def exc_objective(model, batch, device, **kwargs):
        prediction = model.weight.square().sum()
        target = torch.zeros_like(prediction)
        return (
            training.batch_exc(["H2"], prediction.reshape(1), target.reshape(1)),
            prediction.reshape(1),
            target.reshape(1),
        )

    monkeypatch.setattr(training, "calculate_reaction_energy", reaction_energy)
    monkeypatch.setattr(training, "vxc_loss", vxc_objective)
    monkeypatch.setattr(training, "exc_loss", exc_objective)
    reaction_batch = {
        "Database": ["ABDE4"],
        "Grid": torch.ones((1, 1), dtype=torch.float64),
    }
    reaction_loader = [(reaction_batch, torch.zeros(1, dtype=torch.float64))] * 3
    vxc_batch = {"Names": ["H2"]}
    vxc_loader = [vxc_batch] * 3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    params = {
        "accum_iter": 2,
        "gradient_merge_strategy": "sum",
        "reaction_gradient_merge_strategy": "sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 1.0,
        "vxc_gradient_merge_strategy": "sum",
        "vxc_grad_clip": "none",
        "vxc_loss_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_loss_scale": 1.0,
    }
    original = model.weight.detach().clone()

    metrics, _, failed = training.train_one_epoch(
        model=model,
        optimizer=optimizer,
        train_loader=reaction_loader,
        vxc_train_loader=vxc_loader,
        params=params,
        device=torch.device("cpu"),
        dispersions={},
        mrks_dispersions={},
        include_mrks_dispersion=True,
        world_size=1,
        epoch=0,
        potential_mode="full_euler",
        data_protocol="lap_full_vxc",
        point_chunk_size=1,
    )

    assert not failed
    assert metrics["optimizer_steps"] == 2
    assert metrics["gradient_norm"] > 0
    assert metrics["parameter_update_norm"] > 0
    assert not torch.equal(model.weight.detach(), original)
