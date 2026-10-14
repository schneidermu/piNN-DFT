"""Small exact eigenpair/shift fixtures, without molecular or SCF calls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

PATH = Path(__file__).resolve().parents[1] / "tools/lap_gauge_consistency_v1.py"
SPEC = importlib.util.spec_from_file_location("gauge_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture():
    S = np.array([[1.4, 0.2, 0], [0.2, 1.1, 0.1], [0, 0.1, 0.9]])
    values, vectors = eigh(S)
    C = (vectors / np.sqrt(values)) @ vectors.T
    energy = np.array([-0.7, -0.2, -0.2])  # Deliberate degenerate virtual pair.
    F = S @ C @ np.diag(energy) @ C.T @ S
    Q = S + np.diag([0.05, -0.03, 0.01])
    return S, C, energy, F, Q


def test_exact_nonorthogonal_eigenpairs_and_incomplete_space():
    S, C, energy, F, Q = fixture()
    row, _, _ = AUDIT.diagnostics(F, S, C, energy, np.array([2, 0, 0]), Q)
    assert row["raw_fro_ha"] < 1e-14
    assert row["mo_orthogonality_fro"] < 1e-14
    row, _, _ = AUDIT.diagnostics(F, S, C[:, :2], energy[:2], np.array([2, 0]), Q)
    assert not row["complete_mo_space"]
    assert row["raw_fro_ha"] < 1e-14


@pytest.mark.parametrize("kind", ["Q", "S"])
def test_known_shift_and_additive_sign(kind):
    S, C, energy, F, Q = fixture()
    matrix = Q if kind == "Q" else S
    _, shifts, _ = AUDIT.diagnostics(
        F + 0.023 * matrix, S, C, energy, np.array([2, 0, 0]), Q
    )
    fit = next(
        r for r in shifts if r["shift_basis"] == kind and r["fit_space"] == "full"
    )
    assert fit["c_add_ha"] == pytest.approx(-0.023, abs=1e-14)
    assert fit["adjusted_full_fro_ha"] < 1e-14


def test_nonconstant_and_false_EKT_metadata():
    S, C, energy, F, Q = fixture()
    # Metadata HOMO+IP is exactly zero, but this does not qualify the potential.
    assert energy[0] + 0.7 == 0
    row, shifts, _ = AUDIT.diagnostics(
        F + np.diag([0.03, -0.01, 0.05]), S, C, energy, np.array([2, 0, 0]), Q
    )
    assert row["raw_max_orbital_ha"] > 1e-3
    assert all(r["adjusted_max_orbital_ha"] > 1e-3 for r in shifts)


def test_degenerate_subspace_rotation_and_no_mutation():
    S, C, energy, F, Q = fixture()
    saved = [a.copy() for a in (S, C, energy, F, Q)]
    rotation = np.eye(3)
    rotation[1:, 1:] = [[0.6, -0.8], [0.8, 0.6]]
    row, _, _ = AUDIT.diagnostics(F, S, C @ rotation, energy, np.array([2, 0, 0]), Q)
    assert row["raw_fro_ha"] < 1e-14
    for original, copy in zip((S, C, energy, F, Q), saved):
        assert np.array_equal(original, copy)


def test_source_has_no_scf_or_neural_calls():
    source = PATH.read_text()
    assert ".kernel(" not in source
    assert "get_veff(" not in source
    assert "torch.load" not in source
    assert "optimizer.step" not in source


def test_incomplete_space_does_not_hide_external_residual():
    S, C, energy, F, Q = fixture()
    coupling = np.zeros((3, 3))
    coupling[0, 2] = coupling[2, 0] = 0.02
    perturbed = F + S @ C @ coupling @ C.T @ S
    row, _, _ = AUDIT.diagnostics(
        perturbed, S, C[:, :2], energy[:2], np.array([2, 0]), Q
    )
    assert row["offdiagonal_fro_ha"] < 1e-14
    assert row["outside_mo_span_fro_ha"] == pytest.approx(0.02, abs=1e-14)
