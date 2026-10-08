import numpy as np
import pytest
import torch

from train_lap_microbatch import existing, restore, sample_pair, save


def rows(task, count):
    return [{'id': f'{task}_{i:03}', 'task': task, 'database': 'example', 'reaction_id': i,
             'variants': {f'grid_{v}': {} for v in range(8)}} for i in range(count)]


def test_uniform_identity_population_and_variant_hierarchy():
    catalog = rows('relchem', 251) + rows('ae17', 17)
    for task, count in (('relchem', 251), ('ae17', 17)):
        samples = [sample_pair(catalog, task, i) for i in range(5000)]
        assert len({s['identity'] for s in samples}) == count
        assert all(s['weight'] == 1 and s['variant'] in {f'grid_{v}' for v in range(8)} for s in samples)
        assert samples[:50] == [sample_pair(list(reversed(catalog)), task, i) for i in range(50)]
        assert sample_pair(catalog, task, 13) == sample_pair(catalog, task, 13)
    with pytest.raises(ValueError):
        sample_pair(catalog, 'missing', 0)


def test_uniform_singleton_gradient_is_mean_not_db_rmse():
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    slopes = torch.tensor([1., 2., 4.], dtype=torch.float64)
    targets = torch.tensor([2., -1., 3.], dtype=torch.float64)
    factors = torch.tensor([0.5, 2., 3.], dtype=torch.float64)
    singleton = factors * torch.sqrt((slopes * theta - targets).square() + 1e-20)
    derivatives = [torch.autograd.grad(value, theta, retain_graph=True)[0] for value in singleton]
    mean_gradient = torch.autograd.grad(singleton.mean(), theta, retain_graph=True)[0]
    torch.testing.assert_close(torch.stack(derivatives).mean(), mean_gradient, rtol=0, atol=1e-15)
    rmse_gradient = torch.autograd.grad(torch.sqrt((slopes * theta - targets).square().mean()), theta)[0]
    assert not torch.isclose(mean_gradient, rmse_gradient)


def test_uniform_variant_mean_and_db_stratification_equivalence():
    # Identity-first does not multiply a reaction's weight by eight.
    gradients = np.arange(5 * 8 * 3, dtype=np.float64).reshape(5, 8, 3)
    uniform = gradients.mean(axis=(0, 1))
    stratified = 2 / 5 * gradients[:2].mean(axis=(0, 1)) + 3 / 5 * gradients[2:].mean(axis=(0, 1))
    np.testing.assert_allclose(uniform, stratified, rtol=0, atol=1e-14)
    # Partition a global microbatch across GPUs with task-specific denominators.
    np.testing.assert_array_equal(gradients[:, 0].sum(axis=0) / 5,
                                  (gradients[:2, 0].sum(axis=0) + gradients[2:, 0].sum(axis=0)) / 5)


def test_real_checkpoint_restores_optimizer_scheduler_rng_and_cursor(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    calibration = {'lambda': [0.1, 0.2, 0.3, 0.4]}
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    path = tmp_path / 'latest.pt'
    save(path, model, optimizer, 1, 'manifest', calibration, scheduler, total_updates=10)
    expected_rng = existing.capture_rng_state()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected = {n: v.clone() for n, v in model.state_dict().items()}
    assert restore(path, model, optimizer, 'manifest', calibration, scheduler) == 1
    assert existing.equal(existing.capture_rng_state(), expected_rng)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert all(torch.equal(model.state_dict()[n], v) for n, v in expected.items())
    with pytest.raises(AssertionError):
        restore(path, model, optimizer, 'wrong manifest', calibration, scheduler)


def test_chunked_pointwise_model_preserves_reaction_chain_rule(monkeypatch):
    from train_lap_microbatch import existing
    module = existing.lap_training
    monkeypatch.setattr(module, 'calculate_reaction_energy',
                        lambda reaction, constants, *args, **kwargs: (constants.sum().reshape(1), None))
    raw = torch.randn(19, 9, dtype=torch.float64)
    raw[:, :2] = raw[:, :2].abs() + 1
    raw[:, 2:5] = 0
    reaction = {'Grid': raw, 'Densities': raw[:, :2], 'Gradients': raw[:, 2:5], 'Database': 'EA13'}
    model = torch.nn.Sequential(torch.nn.Linear(9, 8), torch.nn.LayerNorm(8), torch.nn.GELU(),
                                torch.nn.Linear(8, 3)).double()
    def evaluate(chunk):
        loss = module.reaction_loss(model, {**reaction, 'model_point_chunk_size': chunk},
                                    torch.ones(1, dtype=torch.float64), 'cpu', torch.float64)
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        return loss.detach(), torch.cat([g.flatten() for g in grads])
    gold, actual = evaluate(0), evaluate(4)
    torch.testing.assert_close(actual, gold, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        evaluate(-1)
