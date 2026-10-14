"""Small grid-transfer bookkeeping and matched-loss negative controls."""

import itertools

import numpy as np

from tools.lap_grid_transfer_v1 import close, loss_identity, reference


def test_deletion_and_duplicate_contributions():
    phi = np.array([[1.0, 0.2], [0.3, 1.0], [0.7, 0.6]])
    weights = np.array([0.4, 0.5, 0.2])
    potential = np.array([0.2, -0.7, 0.4])
    counts = np.array([0, 2, 1])
    original = reference(phi, weights, potential)
    changed = reference(phi, weights * counts, potential)
    deletion = reference(phi, -weights * (counts == 0), potential)
    excess = reference(phi, weights * np.maximum(counts - 1, 0), potential)
    assert close(changed - original, deletion + excess)["status"] == "PASS"
    assert np.linalg.norm(deletion) > 0 and np.linalg.norm(excess) > 0


def test_rounding_and_constant_field_is_Q():
    phi = np.array([[1.0, 0.1], [0.4, 0.6]])
    weights = np.array([0.123456789, 0.765432198])
    potential = np.array([-0.333333333, -0.777777777])
    Q = reference(phi, weights, np.ones(2))
    assert np.allclose(reference(phi, weights, np.full(2, 0.4)), 0.4 * Q)
    assert not np.allclose(0.4 * Q, 0.4 * np.eye(2))
    original = reference(phi, weights, potential)
    rounded_weights = reference(
        phi, weights.astype(np.float32).astype(float), potential
    )
    rounded_potential = reference(
        phi,
        weights.astype(np.float32).astype(float),
        potential.astype(np.float32).astype(float),
    )
    assert np.linalg.norm(rounded_weights - original) > 0
    assert (
        close(
            (rounded_weights - original) + (rounded_potential - rounded_weights),
            rounded_potential - original,
        )["status"]
        == "PASS"
    )


def test_prediction_reference_changes_cancel_exactly():
    pred = np.diag([0.8, 0.4])
    target = np.diag([0.5, 0.1])
    shift = np.array([[0.2, 0.1], [0.1, -0.1]])
    old = pred - target
    changed = (pred + shift) - (target + shift)
    cross, square = loss_identity(old, changed - old)
    assert abs(cross + square) < 1e-15
    assert np.allclose(old, changed)


def test_small_target_effect_can_accompany_residual_amplification():
    A = np.diag([0.2, -0.1])
    delta_ref = np.eye(2) * 1e-4
    delta_pred = A * 0.3
    delta = delta_pred - delta_ref
    cross, square = loss_identity(A, delta)
    actual = (np.sum((A + delta) ** 2) - np.sum(A**2)) / 2
    assert actual > 0
    assert close(np.array(actual), np.array(cross + square))["status"] == "PASS"


def test_wrong_cached_reference_is_detected():
    phi64 = np.array([[0.123456789123, 0.5555555555], [0.765432198765, 0.3333333333]])
    weights = np.array([0.4, 0.7])
    potential = np.array([-0.3, -0.8])
    stored = reference(phi64, weights, potential)
    cached = reference(phi64.astype(np.float32).astype(float), weights, potential)
    assert close(cached, stored, 1e-12, 1e-12)["status"] == "FAIL"


def test_telescoping_matrices_and_losses():
    matrices = [np.eye(2) * value for value in [0.1, 0.4, 0.3]]
    increments = [b - a for a, b in itertools.pairwise(matrices)]
    assert np.allclose(sum(increments), matrices[-1] - matrices[0])
    sequential = sum(sum(loss_identity(a, d)) for a, d in zip(matrices, increments))
    assert np.isclose(
        sequential, (np.sum(matrices[-1] ** 2) - np.sum(matrices[0] ** 2)) / 2
    )
