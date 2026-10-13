import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f015_audit as audit
from lap_f015_inputs import restored


def test_fixed_ball_quadratic_has_independent_kkt_certificate():
    error = np.array([1., 2., 3.])
    delta, cert = audit.solve_probe(np.eye(3), error, np.ones(3), [[0., 0., 1.]], radius=.01)
    assert cert['qualified']
    np.testing.assert_allclose(delta, -.01 * error / np.linalg.norm(error), atol=1e-9)
    assert cert['predicted_quadratic'] < cert['initial_quadratic']


def test_protection_prevents_first_coordinate_improvement_without_relaxation():
    delta, cert = audit.solve_probe(np.eye(3), [1., 2., 3.], np.ones(3), [[-1., 0., 0.]], radius=.01)
    assert cert['qualified']
    assert delta[0] >= -1e-10
    np.testing.assert_allclose(delta[1:], -.01 * np.array([2., 3.]) / np.sqrt(13), atol=1e-9)
    assert np.linalg.norm(delta) <= .01 + 1e-10


def test_invalid_weights_fail_closed():
    with pytest.raises(AssertionError):
        audit.solve_probe(np.eye(3), [1., 2., 3.], [1., 0., 1.], [[1., 0., 0.]])


def test_restoration_is_exact_even_on_exception(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    model.register_buffer('anchor', torch.tensor([1.]))
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError), restored([model]):
        with torch.no_grad():
            model.weight.add_(.01)
            model.anchor.add_(3.)
        torch.rand(3)
        raise RuntimeError('probe interrupted')
    assert all(torch.equal(before[n], value) for n, value in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


def test_linear_reaction_reconstruction_and_jvp():
    coefficients = np.array([-1., 2.])
    species = np.array([1., 2.])
    jacobian = np.array([[.5, .1], [.2, .3]])
    delta = np.array([.01, -.01])
    before = coefficients @ species * 627.5095
    after = coefficients @ (species + jacobian @ delta) * 627.5095
    np.testing.assert_allclose(after - before, (coefficients @ jacobian * 627.5095) @ delta, atol=1e-12)


def test_no_optimizer_or_validation_entrypoints_in_f015():
    for name in ('lap_f015_audit.py', 'lap_f015_inputs.py'):
        source = (audit.ROOT / 'tools' / name).read_text()
        for forbidden in ('optimizer.step(', 'torch.optim.', 'evaluate_clean28(', '.validation('):
            assert forbidden not in source


def test_canonical_order_is_named_order_not_state_dictionary_iteration():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    native = audit.geometry.b8._import_run().existing.named_trainable_parameters(model)
    assert tuple(native) == tuple(sorted(dict(model.named_parameters())))
    assert tuple(native) != tuple(model.state_dict())


def test_training_only_view_does_not_initialize_validation_metadata():
    bundle = audit.TrainingBundle(audit.ROOT.parent / 'publication_dataset_v1')
    try:
        assert len(bundle.chemistry()) == 251
        assert len(bundle.chemistry('train_ae17')) == 17
        assert len(bundle.mrks()) == 90
        assert not hasattr(bundle, 'validation_species')
        assert not hasattr(bundle, 'validation_reactions')
        with pytest.raises(PermissionError):
            bundle.validation()
    finally:
        bundle.close()
