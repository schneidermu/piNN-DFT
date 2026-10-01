"""Focused h-free checkpoint and pilot-initialization tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lap_checkpoint import checkpoint_payload
from run_lap_operator_pilot import (
    _final_checkpoint_evaluation,
    _fixed_diagnostic_weights,
    _load_operator_checkpoint,
    _load_predopt_start,
    _require_external,
    _save_operator_checkpoint,
    _scf_molecule_for_system,
)
from train_lap_s5 import _pilot_model


def _predopt_checkpoint(path: Path) -> Path:
    model = _pilot_model(torch.device("cpu"), torch.float32)
    torch.save(checkpoint_payload(model, predopt_only=True, predopt_epochs=2), path)
    return path


def test_two_pilot_branches_load_identical_fresh_predopt_state(tmp_path):
    checkpoint = _predopt_checkpoint(tmp_path / "predopt.pt")
    left, _ = _load_predopt_start(checkpoint, torch.device("cpu"), torch.float32)
    right, _ = _load_predopt_start(checkpoint, torch.device("cpu"), torch.float32)
    assert left is not right
    for left_value, right_value in zip(left.state_dict().values(), right.state_dict().values()):
        torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)


def test_operator_pilot_checkpoint_roundtrips_without_stencil_metadata(tmp_path):
    model = _pilot_model(torch.device("cpu"), torch.float32)
    checkpoint = tmp_path / "operator.pt"
    _save_operator_checkpoint(checkpoint, model, {"pilot_only": True})

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    forbidden = {"h", "stencil", "h_bohr", "stencil_version", "stencil_order", "derivative_order"}
    assert not forbidden.intersection(payload)
    assert not forbidden.intersection(payload["operator_metadata"])
    restored, restored_payload = _load_operator_checkpoint(
        checkpoint, torch.device("cpu"), torch.float64
    )
    assert restored_payload["protocol"] == "lap-weakform-ao-v1"
    for original, roundtripped in zip(model.state_dict().values(), restored.state_dict().values()):
        expected = original.double() if original.is_floating_point() else original
        torch.testing.assert_close(roundtripped, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("location", "field"),
    (("top", "h"), ("top", "stencil_order"), ("operator_metadata", "stencil")),
)
def test_operator_pilot_checkpoint_rejects_stencil_metadata(tmp_path, location, field):
    model = _pilot_model(torch.device("cpu"), torch.float32)
    checkpoint = tmp_path / "operator.pt"
    _save_operator_checkpoint(checkpoint, model, {"pilot_only": True})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = payload if location == "top" else payload["operator_metadata"]
    target[field] = 0.1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="h-free|h or stencil|h/stencil"):
        _load_operator_checkpoint(checkpoint, torch.device("cpu"), torch.float32)


def test_diagnostic_weights_are_frozen_inverse_initial_gradient_norms():
    coeff_exc, coeff_operator = _fixed_diagnostic_weights(2.0, 4.0)
    assert coeff_exc == 0.5
    assert coeff_operator == 0.25
    # Later norm measurements produce new values without mutating this branch's weights.
    assert _fixed_diagnostic_weights(8.0, 1.0) == (0.125, 1.0)
    assert (coeff_exc, coeff_operator) == (0.5, 0.25)
    with pytest.raises(ValueError, match="positive and finite"):
        _fixed_diagnostic_weights(0.0, 1.0)


def test_post_update_evaluation_reports_finite_gradients_without_updating_model(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    original = model.weight.detach().clone()

    def losses(current, selected, chunk_size):
        value = current.weight[0, 0]
        exc_error = value - 2.0
        operator_error = value + 1.0
        exc_loss = exc_error.square()
        operator_loss_value = operator_error.square()
        return (
            exc_loss,
            operator_loss_value,
            {
                "H2": {
                    "predicted_exc_hartree": value,
                    "exc_error_kcal_mol": exc_error,
                    "operator_loss_ha2_per_ao": operator_loss_value,
                }
            },
        )

    monkeypatch.setattr("run_lap_operator_pilot._losses", losses)
    evaluation = _final_checkpoint_evaluation(model, [object()], 8, 0.5, 0.25)
    assert evaluation["checkpoint_state"] == "post_update"
    assert evaluation["exc_gradient_norm"] > 0
    assert evaluation["operator_gradient_norm"] > 0
    assert evaluation["combined_gradient_norm"] > 0
    assert evaluation["system_metrics"]["H2"]["combined_objective_gradient_norm"] > 0
    assert model.weight.item() == original.item()
    assert all(
        torch.isfinite(torch.tensor(value))
        for key, value in evaluation.items()
        if isinstance(value, (int, float))
    )


def test_pilot_artifact_paths_must_stay_outside_repository(tmp_path):
    external = _require_external(tmp_path, "test artifact")
    assert external == tmp_path.resolve()
    with pytest.raises(ValueError, match="outside the repository"):
        _require_external(REPO_ROOT / "pilot-output", "test artifact")


def test_cached_system_reconstructs_molecule_for_wsl_scf(monkeypatch):
    metadata = {"molecule": {"system_name": "H2"}}
    sentinel = object()

    def build(value):
        assert value is metadata
        return sentinel

    monkeypatch.setattr("run_lap_operator_pilot.build_molecule_from_metadata", build)
    system = SimpleNamespace(mol=None, record=SimpleNamespace(metadata=metadata))
    assert _scf_molecule_for_system(system) is sentinel
