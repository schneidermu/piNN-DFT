"""Frozen stream invariants and independent cycle algebra."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f020_stream as audit


def test_authenticated_stream_partition():
    batches = audit.audit.geometry.f010._batches()
    rows = [r for b in batches for r in b['relchem']]
    counts = Counter(r['identity'] for r in rows)
    assert len(rows) == 2008 and len(counts) == 251
    assert set(counts.values()) == {8}
    variants = {}
    for r in rows:
        variants.setdefault(r['identity'], set()).add(r['variant'])
    assert all(len(v) == 1 for v in variants.values())
    spec = audit.audit.geometry.STATES[0]
    archived, _ = audit.audit.chemistry._load_reactions(spec['reactions'])
    matching = sum(variants[r['identity']] == {r['variant']} for r in archived)
    assert matching == 39 and 251 - matching == 212


def test_cycle_algebra_without_claiming_real_gradient_qualification():
    g = np.random.default_rng(20).normal(size=(251, 9))
    index = np.array([[(i + j * 31) % 251 for j in range(8)] for i in range(251)])
    np.testing.assert_allclose(g[index].mean(axis=(0, 1)), g.mean(axis=0), atol=1e-14)
    db = [d for d, n in audit.f019.COUNTS.items() for _ in range(n)]
    weighted = audit.f019.weights(db)[:, None] * g
    np.testing.assert_allclose(weighted[index].mean(axis=(0, 1)), weighted.mean(axis=0), atol=1e-14)


def test_no_retained_optimizer_or_development_evaluator():
    import ast
    tree = ast.parse(Path(audit.__file__).read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not {'step', 'validation', 'backward'}.intersection(calls)
    assert audit.BUDGET == 1800


def test_saved_reconstruction_and_frozen_gate():
    receipt = audit.audit.read(audit.ROOT / 'lap_f020_jacobian_reconstruction.json')
    assert receipt['status'] == 'PASS' and receipt['shape'] == [251, 9446]
    assert receipt['reused'] == 39 and receipt['new'] == 212
    path = audit.OUT / 'stream_jacobian.npz'
    assert audit.audit.sha(path) == receipt['sha256']
    with np.load(path) as payload:
        assert payload['gradients'].shape == (251, 9446)
        assert np.isfinite(payload['gradients']).all()
    mapping = audit.audit.read(audit.OUT / 'stream_rows.json')
    assert len({r['identity'] for r in mapping}) == 251
    assert len({r['parameter_order_sha256'] for r in mapping}) == 1
    cycle = audit.audit.read(audit.ROOT / 'lap_f020_cycle_gradient_parity.json')
    assert cycle['status'] == 'PASS'
    assert cycle['JW_gradient_l2'] <= 1e-10 and cycle['DB_gradient_l2'] <= 1e-10
    decision = audit.audit.read(audit.ROOT / 'lap_f020_decision.json')
    assert decision['positive_dot'] >= 24
    assert decision['p10_F019'] < decision['p10_F010'] - .10
    assert decision['stochastic_fidelity'] == 'FAIL'
    assert decision['finite_step'] == 'NOT RUN' and not decision['training_authorized']


def test_historical_receipts_unchanged():
    old = audit.audit.read(audit.ROOT / 'lap_f019_decision.json')
    assert all(audit.audit.sha(audit.ROOT / name) == sha for name, sha in old['artifacts'].items())
