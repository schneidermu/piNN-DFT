import torch


_ADAPTIVE_INDICES = (0, 1, 22, 23, 24, 25, 26, 27, 28)


def _prepare_predopt_targets(
    constants: torch.Tensor,
    n_grid_points: int,
    device: torch.device,
) -> torch.Tensor:
    """Tile canonical constants over grid points and select adaptive targets."""
    constants_batch = torch.tile(constants, [n_grid_points, 1]).to(
        device, non_blocking=True
    )
    return constants_batch[:, _ADAPTIVE_INDICES]
