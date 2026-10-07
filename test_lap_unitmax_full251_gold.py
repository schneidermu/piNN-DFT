"""Full chemistry adapter and exact checkpoint boundary, without scientific runs."""
import copy

import numpy as np
import pytest
import torch

import lap_unitmax_full251_gold as g


def test_adapter_uses_same_scalar_and_f64_parameter_order(monkeypatch):
    model = torch.nn.Linear(2, 1).float()
    shadow = copy.deepcopy(model).double()
    calls = []
    def exact(model, shadow, context, gradient):
        calls.append(gradient)
        vector = np.arange(1, 4, dtype=np.float64) if gradient else None
        return 7., vector
    monkeypatch.setattr(g.chem, 'fullchem', exact)
    class Store:
        def load_variant(self, *args):
            return {}
    objective = g.Full251Chemistry(model, shadow, {'entries': [None]*251, 'store': Store()})
    assert isinstance(objective, g.h.core.ChemistryBatchObjective)
    value, gradient = objective.value_and_grad()
    assert value == float(objective()) == 7.
    assert calls == [True, False]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.dtype == torch.float64 for p in shadow.parameters())
    flat = torch.cat([gradient[n].flatten() for n in g.h.named_trainable_parameters(model)])
    np.testing.assert_array_equal(flat.numpy(), np.arange(1, 4, dtype=np.float64))


def test_only_chemistry_factory_changes(monkeypatch):
    model = torch.nn.Linear(1, 1)
    shadow = copy.deepcopy(model).double()
    originals = dict(zip(g.h.ORDER, [object() for _ in range(4)]))
    monkeypatch.setattr(g.h, 'objectives', lambda *args: originals.copy())
    context = {'entries': [None]*251}
    result = g.objectives(model, shadow, {}, context)
    assert tuple(result) == g.h.ORDER
    assert isinstance(result['relchem'], g.Full251Chemistry)
    for t in g.h.ORDER[1:]:
        assert result[t] is originals[t]


def test_resume_restores_cursor_model_rng_and_fails_before_mutation(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    source = tmp_path/'precision_source.py'
    source.write_text('qualified')
    p = {'manifest_file_sha256': g.sha(manifest), 'immutable_files': {str(source): g.sha(source)}, 'script_sha256': g.sha(g.__file__)}
    g.dump(tmp_path/'protocol.json', p)
    monkeypatch.setattr(g.h, 'OUT', tmp_path)
    monkeypatch.setattr(g.h, 'MANIFEST', manifest)
    monkeypatch.setattr(g.h, 'verify', g.verify)
    model = torch.nn.Linear(1, 1)
    before = g.h.digest(model)
    rng = g.h.capture_rng_state()
    g.h.checkpoint(tmp_path/'state.pt', model, {}, 5, p)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    cursor, state = g.h.resume(tmp_path/'state.pt', model, p)
    assert cursor == 5 and state == {} and g.h.digest(model) == before
    assert g.h.equal(g.h.capture_rng_state(), rng)
    source.write_text('changed')
    before = g.h.digest(model)
    with pytest.raises(AssertionError):
        g.h.resume(tmp_path/'state.pt', model, p)
    assert g.h.digest(model) == before
