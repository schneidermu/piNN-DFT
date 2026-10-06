"""Focused artifact geometry and temporary-trial tests; no real data."""
import numpy as np
import pytest
import torch

import lap_fullchem_surrogate_fidelity_audit as audit


def test_parameter_order_matches_shadow_and_tied_parameters_once():
    model = torch.nn.Linear(2, 2)
    shadow = torch.nn.Linear(2, 2).double()
    assert tuple(audit.arena.named_trainable_parameters(model)) == tuple(audit.arena.named_trainable_parameters(shadow))
    tied = torch.nn.Module()
    tied.first = model
    tied.second = model
    assert sum(p.numel() for p in audit.arena.named_trainable_parameters(tied).values()) == 6


def test_stored_direction_replay_and_descent_sign(tmp_path):
    raw = np.array([[1., 0.], [1., 1.], [2., 1.], [3., -1.]])
    direction, _ = audit.arena.offline.maxmin(raw, audit.arena.offline.reuse.sha(audit.arena.offline.reuse.SOLVER))
    path = tmp_path/'saved.npz'
    np.savez(path, raw=raw, unit=direction)
    saved = np.load(path)
    replay, _ = audit.arena.offline.maxmin(saved['raw'], audit.arena.offline.reuse.sha(audit.arena.offline.reuse.SOLVER))
    np.testing.assert_array_equal(saved['unit'], replay)
    row = audit.directional(raw, -raw[0], saved['unit'])
    assert row['p_sample'] > 0 and row['p_full'] < 0
    assert row['e_direction'] == pytest.approx(row['p_sample']-row['p_full'])


def test_counterfactual_fullchem_and_finite_prediction():
    raw = np.array([[1., 0.], [1., 1.], [2., 1.], [3., -1.]])
    full = np.array([1., .3])
    raw[0] = full
    d, meta = audit.arena.offline.maxmin(raw, audit.arena.offline.reuse.sha(audit.arena.offline.reuse.SOLVER))
    assert meta['gamma_star'] > 0 and np.all(raw@d > 0)
    eta = .01
    predicted = -eta*(full@d)
    assert predicted < 0
    assert full@(-eta*d) == pytest.approx(predicted)


@pytest.mark.parametrize('raises', [False, True])
def test_temporary_perturbation_restoration_and_no_artifact_mutation(tmp_path, raises):
    model = torch.nn.Linear(2, 1, bias=False).float()
    before = audit.arena.digest(model)
    snapshot = model.weight.detach().clone()
    immutable = tmp_path/'immutable'
    immutable.write_bytes(b'exact original artifact')
    artifact_sha = audit.arena.offline.reuse.sha(immutable)
    def evaluate():
        if raises:
            raise RuntimeError('frozen failure')
        return float(model.weight.detach().sum())
    if raises:
        with pytest.raises(RuntimeError):
            audit.temporary_trial(model, np.array([1., 0.]), .001, evaluate)
    else:
        value, requested, realized, _ = audit.temporary_trial(model, np.array([1., 0.]), .001, evaluate)
        assert value == pytest.approx(float(snapshot.sum())-.001, abs=1e-7)
        assert requested[0] == -.001 and realized[0] < 0
    assert audit.arena.digest(model) == before and torch.equal(model.weight, snapshot)
    assert audit.arena.offline.reuse.sha(immutable) == artifact_sha


@pytest.mark.parametrize('mismatch,nonlinear,case', [(True, False, 'CASE A'), (False, True, 'CASE B'),
                                                    (True, True, 'CASE C'), (False, False, 'CASE D')])
def test_prospective_classifier(mismatch, nonlinear, case):
    rows = [{'p_sample': .2, 'p_full': -.1 if mismatch else .1,
             'actual_trials': [{'delta': -.01}]}]
    if nonlinear:
        rows.append({'p_sample': .2, 'p_full': .1, 'actual_trials': [{'delta': .01}]})
    assert audit.classify(rows) == case
