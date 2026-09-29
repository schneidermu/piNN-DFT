import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_functionals import PBE_CONSTANTS
from predopt_targets import _prepare_predopt_targets


def test_predopt_targets_use_canonical_adaptive_pbe_constants():
    targets = _prepare_predopt_targets(
        PBE_CONSTANTS,
        n_grid_points=3,
        device=torch.device("cpu"),
    )

    expected = torch.stack(
        (
            PBE_CONSTANTS[0, 0],
            PBE_CONSTANTS[0, 1],
            PBE_CONSTANTS[0, 22],
            PBE_CONSTANTS[0, 23],
            PBE_CONSTANTS[0, 24],
            PBE_CONSTANTS[0, 25],
            PBE_CONSTANTS.new_tensor(0.0),
            PBE_CONSTANTS.new_tensor(0.0),
            PBE_CONSTANTS.new_tensor(1.0),
        )
    )
    torch.testing.assert_close(targets, expected.expand(3, -1), rtol=0, atol=0)
