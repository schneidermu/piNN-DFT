import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import lap_unitmax_svrg_k1 as s


def test_static_formula_and_reference_cancellation():
    full = np.array([2., -3.])
    current = np.array([7., 11.])
    reference = np.array([1., 5.])
    assert np.array_equal(s.combine(full, current, reference), [8., 3.])
    assert np.array_equal(s.combine(full, current, current), full)
    assert np.array_equal(full, [2., -3.])


def test_pairing_unused_sample_scalar_and_persistent_reference(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    shadow = copy.deepcopy(model).double()
    reactions, weights = ({'id': 7},), (1.,)
    sampled = SimpleNamespace(model=model, shadow=shadow, reactions=reactions, weights=weights,
                              dispersions={}, value_and_grad=lambda: (100., {'weight': torch.tensor([[7.]], dtype=torch.float64)}))
    state = {'key': 'test', 'reference': copy.deepcopy(model), 'shadow': copy.deepcopy(shadow),
             'full': np.array([10.]), 'reference_sha256': 'fixed-t0',
             'folder': tmp_path, 'scalars': {s.h.digest(model): 12.},
             'cost': {'current_sample_backwards': 0, 'reference_sample_backwards': 0}}
    captured = []
    def reference_factory(main, hidden, data, w, dispersions):
        captured.append((main, data, w))
        return SimpleNamespace(value_and_grad=lambda: (20., {'weight': torch.tensor([[3.]], dtype=torch.float64)}))
    rows = [{'database': 'DB', 'reaction_id': 7, 'variant_suffix': 'exact', 'weight': 1.}]
    with patch.object(s, 'reference_state', return_value=state), patch.object(s.h.core, 'ChemistryBatchObjective', reference_factory):
        objective = s.StaticSVRG(sampled, rows, {})
        value, gradient = objective.value_and_grad()
        assert value == 100. and gradient['weight'].item() == 14.
        with pytest.raises(RuntimeError, match='No per-update'):
            objective()
        assert captured[0][0] is state['reference'] and captured[0][1] is reactions and captured[0][2] is weights
        objective.value_and_grad()
        assert len(captured) == 1  # Resume-safe paired reference cache reuse.
        assert np.array_equal(state['full'], [10.])
        receipt = next(tmp_path.glob('reference_sample_*.json'))
        bound = s.load(receipt)
        bound['reference_sha256'] = 'future-state'
        s.dump(receipt, bound)
        with pytest.raises(AssertionError):
            objective.value_and_grad()


def test_t0_uses_exact_reference_without_redundant_sample_backward(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    sampled = SimpleNamespace(model=model, shadow=copy.deepcopy(model).double())
    state = {'full': np.array([4.]), 'reference_sha256': s.h.digest(model),
             'scalars': {s.h.digest(model): 3.}}
    with patch.object(s, 'reference_state', return_value=state), patch.object(s, 'counted', side_effect=AssertionError('unnecessary backward')):
        value, gradient = s.StaticSVRG(sampled, [], {}).value_and_grad()
    assert value == 3. and gradient['weight'].item() == 4. and model.weight.dtype == torch.float32


def test_exact_manifest_rows_and_db_weights():
    counts = {'A': 244, **{chr(66+i): 1 for i in range(7)}}
    entries = [{'per_rank': [{'reaction': {'database': db}}]} for db, n in counts.items() for _ in range(n)]
    rows = [{'database': db, 'reaction_id': 9, 'variant_suffix': 'frozen', 'weight': n/251} for db, n in counts.items()]
    factories = {'relchem': object(), 'ae17': object(), 'exc': object(), 'op': object()}
    captured = []
    def adapter(sampled, actual, context):
        captured.append(actual)
        return 'svrg'
    with patch.object(s.h, 'objectives', return_value=dict(factories)), patch.object(s, 'StaticSVRG', adapter):
        result = s.objectives(None, None, {'per_rank': [{'task_samples': {'relchem': rows}}]}, {'entries': entries})
        assert captured[0] is rows and result['ae17'] is factories['ae17'] and result['op'] is factories['op']
        wrong = copy.deepcopy(rows)
        wrong[0]['weight'] = 1.
        with pytest.raises(AssertionError):
            s.objectives(None, None, {'per_rank': [{'task_samples': {'relchem': wrong}}]}, {'entries': entries})


def test_actual_backward_counter(tmp_path):
    parameter = torch.tensor(3., requires_grad=True, dtype=torch.float64)
    class Batch:
        def value_and_grad(self):
            loss = parameter.square()
            gradient, = torch.autograd.grad(loss, parameter)
            return float(loss.detach()), {'x': gradient}
    state = {'folder': tmp_path, 'cost': {'current_sample_backwards': 0}}
    value, gradients = s.counted(Batch(), state, 'current_sample_backwards')
    assert value == 9. and gradients['x'].item() == 6.
    assert s.load(tmp_path/'chemistry_cost.json')['current_sample_backwards'] == 1


def test_fixed_step_exact_displacement_no_trial_and_rng():
    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    rng = s.h.capture_rng_state()
    raw = {t: {'weight': torch.ones(1, 1, dtype=torch.float64)} for t in s.h.ORDER}
    def aggregate(values, state):
        return {'weight': torch.ones(1, 1, dtype=torch.float64)}, {'feasible': True}, {}
    with patch.object(s.h.core, 'compute_isolated_task_gradients', return_value=({}, raw)):
        result = s.fixed_update(model, None, {}, aggregator=aggregate, aggregator_state=None,
                               task_order=s.h.ORDER, vector_armijo={'initial_step_size': 1e-5})
    assert torch.equal(model.weight, (before.double()-1e-5).float())
    assert result.accepted and result.diagnostics['vector_armijo']['trials'] == []
    assert s.h.equal(rng, s.h.capture_rng_state())


def test_fixed_step_solver_failure_restores_exact_state():
    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    rng = s.h.capture_rng_state()
    def fail(*args, **kwargs):
        with torch.no_grad():
            model.weight.add_(1)
        torch.rand(1)
        raise RuntimeError('solver failure')
    with patch.object(s.h.core, 'compute_isolated_task_gradients', side_effect=fail), \
         pytest.raises(RuntimeError, match='solver failure'):
        s.fixed_update(model, None, {}, aggregator=None, aggregator_state=None,
                       task_order=s.h.ORDER, vector_armijo={'initial_step_size': 1e-5})
    assert torch.equal(model.weight, before) and s.h.equal(rng, s.h.capture_rng_state())
