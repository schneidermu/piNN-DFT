"""Two-rank CPU reduction and optimizer cadence independent of chunk count."""

import sys

import pytest
import torch
from lap_analytic import PolynomialEnergy, features
from lap_training import average_gradients
from lap_vxc import full_vxc_loss


def distributed_worker(rank, init_path, out_path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + init_path, rank=rank, world_size=2
    )
    model = PolynomialEnergy(b=0.3, c=0.2)
    f = features(0.1, n=3)
    target = torch.ones(3, 2, dtype=torch.float64) * (rank + 1)
    w = torch.ones(3, dtype=torch.float64)
    loss = full_vxc_loss(model, f, target, w, 0.1, point_chunk_size=rank + 1)
    loss.backward()
    average_gradients(model, 2)
    torch.save(model.coefficients.grad, out_path + str(rank))
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32", reason="CPU Gloo process test uses Linux file rendezvous"
)
def test_two_rank_mean_matches_single_process(tmp_path):
    torch.multiprocessing.spawn(
        distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "grad")),
        nprocs=2,
    )
    model = PolynomialEnergy(b=0.3, c=0.2)
    f, w = features(0.1, n=3), torch.ones(3, dtype=torch.float64)
    losses = [
        full_vxc_loss(
            model, f, torch.ones(3, 2, dtype=torch.float64) * (i + 1), w, 0.1, 3
        )
        for i in range(2)
    ]
    ((losses[0] + losses[1]) / 2).backward()
    for rank in (0, 1):
        torch.testing.assert_close(
            torch.load(str(tmp_path / "grad") + str(rank), weights_only=True),
            model.coefficients.grad,
        )


@pytest.mark.parametrize("chunk", [1, 5])
def test_optimizer_cadence_is_per_system_not_point_chunk(monkeypatch, chunk):
    import train_lap

    model = PolynomialEnergy(b=0.3, c=0.2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    steps = []
    original = optimizer.step

    def step():
        steps.append(1)
        original()

    monkeypatch.setattr(optimizer, "step", step)
    monkeypatch.setattr(
        train_lap, "reaction_loss", lambda *a: model.coefficients.square().sum()
    )
    f, w = features(0.1, n=3), torch.ones(3, dtype=torch.float64)

    def losses(*a):
        return model.coefficients.square().sum(), full_vxc_loss(
            model, f, torch.zeros(3, 2), w, 0.1, chunk
        )

    monkeypatch.setattr(train_lap, "mrks_losses", losses)
    history = train_lap.epoch_steps(
        model,
        model,
        [(None, None)] * 5,
        [None],
        [0],
        optimizer,
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        torch.float64,
        chunk,
        2,
        1,
    )
    assert len(history) == 5
    assert len(steps) == 3
