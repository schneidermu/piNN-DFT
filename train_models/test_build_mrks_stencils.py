"""Small invariants for exact legacy-target and coordinate provenance."""

import numpy as np
import pytest
import torch
from build_mrks_stencils import (
    exact_coordinate_lookup,
    exact_vxc_match,
    preserved_targets,
)


def test_preserved_targets_keep_legacy_float32_bits_after_float64_promotion():
    legacy_exc = torch.tensor(-0.7023566365242004, dtype=torch.float32)
    legacy_vxc = torch.tensor([0.12345679104328156, -2.3456788063049316])
    target, exc = preserved_targets({"Vrho": legacy_vxc, "E_xc": legacy_exc})
    assert torch.equal(target[:, 0], legacy_vxc.to(torch.float64))
    assert torch.equal(target[:, 1], legacy_vxc.to(torch.float64))
    assert torch.equal(exc, legacy_exc.to(torch.float64))


def test_vxc_matching_requires_exact_coordinate_identity():
    legacy_grid = np.zeros((1, 13), dtype=np.float32)
    legacy_grid[0, :3] = [1.0, 2.0, 3.0]
    npz_coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    stats = exact_vxc_match(legacy_grid, np.array([0.5]), npz_coords, np.array([0.5]))
    assert stats["matched_legacy_points"] == 1
    npz_coords[0, 0] += 1e-5
    with pytest.raises(ValueError, match="No exact"):
        exact_vxc_match(legacy_grid, np.array([0.5]), npz_coords, np.array([0.5]))


def test_original_float64_center_recovery_uses_identity_and_rejects_collisions():
    legacy = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    source = np.array([[1.0 + 1e-8, 2.0, 3.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        exact_coordinate_lookup(legacy, source), np.array([0])
    )
    ambiguous = np.array(
        [[1.0 + 1e-8, 2.0, 3.0], [1.0 + 2e-8, 2.0, 3.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        exact_coordinate_lookup(legacy, ambiguous)
