"""Focused contracts for the isolated matched physics-only arm."""
import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import lap_f024_phys64 as exp


def test_authenticated_parent_control_and_sources():
    protocol = exp.read('protocol')
    assert exp.audit.sha(exp.f010.PARENT_CHECKPOINT) == protocol['checkpoint_sha256']
    assert exp.audit.sha(exp.CONTROL) == protocol['control_sha256']
    parent = torch.load(exp.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert exp.f010.continuation.moment_sha(parent['optimizer']) == protocol['moment_sha256']
    for path, sha in protocol['source_hashes'].items():
        assert exp.audit.sha(exp.ROOT/path) == sha


def test_original_stream_and_fixed_coefficients():
    p=exp.read('protocol')
    rows=[{'position':i,'ae17':r['ae17'],'mrks_id':r['mrks_id']}
          for i,r in enumerate(exp.f010._batches()[:64])]
    assert rows==p['physical_stream']
    assert exp.old.previous.key(rows)==p['physical_stream_sha256']
    assert p['lambdas']=={t:exp.b8.LAMBDAS[t] for t in exp.TASKS}
    assert p['cumulative_start']==128 and p['cumulative_final']==192


def test_F64_three_task_sum_and_materialized_zero():
    parameters={'a':torch.nn.Parameter(torch.ones(2)), 'disconnected':torch.nn.Parameter(torch.zeros(2))}
    raw={t:{'a':torch.tensor([i, -i],dtype=torch.float64),
            'disconnected':torch.zeros(2,dtype=torch.float64)} for i,t in enumerate(exp.TASKS,1)}
    g=exp.combine(raw,parameters)
    expected=sum(exp.b8.LAMBDAS[t]*raw[t]['a'] for t in exp.TASKS)
    assert torch.equal(g['a'],expected)
    assert g['a'].dtype==torch.float64 and torch.equal(g['disconnected'],torch.zeros(2,dtype=torch.float64))
    assert g['a'].to(parameters['a']).dtype==torch.float32


def test_no_relative_training_backward_or_development_call():
    tree=ast.parse(Path(exp.__file__).read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    code=ast.unparse(train)
    assert '_singleton' not in code and '_load_reaction' not in code and 'qualified_mean' not in code
    assert 'other_tasks' in code and code.count('optimizer.step()')==1
    assert 'validation(' not in Path(exp.__file__).read_text()
    assert exp.read('preflight')['relative_chemistry_training_calls']==0


def test_technical_gate_not_common_descent_gate():
    assert exp.read('preflight')['passed']
    assert all(r['gradient_gap']<=r['bound'] for r in exp.read('preflight')['rows'])
    assert exp.read('resource_budget')['resource_qualified']
    assert exp.read('protocol')['operator_effect_floor']==.02918097603147132


def test_native_zero_gradient_momentum_and_recurrence():
    p=torch.nn.Parameter(torch.tensor([.3,-.1],dtype=torch.float32))
    optimizer=torch.optim.AdamW([p],lr=2.5e-5,foreach=False)
    p.grad=torch.tensor([1.,2.]); optimizer.step()
    initial=optimizer.state[p]['exp_avg'].clone()
    p.grad=torch.zeros_like(p); optimizer.step()
    assert torch.allclose(optimizer.state[p]['exp_avg'],.9*initial)
    assert int(optimizer.state[p]['step'])==2
    assert np.isfinite(p.detach().numpy()).all()


def test_checkpoint_roundtrip_preserves_state_and_cursor(tmp_path):
    saved=torch.load(exp.OUT/'checkpoint_0.pt',map_location='cpu',weights_only=False)
    target=tmp_path/'roundtrip.pt'
    torch.save(saved,target)
    restored=torch.load(target,map_location='cpu',weights_only=False)
    assert exp.b8._import_run().existing.equal(saved,restored)
    assert saved['cursor']==0
    parent=torch.load(exp.f010.PARENT_CHECKPOINT,map_location='cpu',weights_only=False)
    assert exp.b8._import_run().existing.equal(saved['model'],parent['model'])
    assert exp.f010.continuation.moment_sha(saved['optimizer'])==exp.read('protocol')['moment_sha256']
    assert exp.b8._import_run().existing.equal(saved['rng'],parent['rng'])
    assert {int(s['step']) for s in saved['optimizer']['state'].values()}=={128}
    assert saved['protocol_sha256']==exp.audit.sha(exp.ROOT/(exp.PREFIX+'protocol.json'))


def test_completed_resume_and_native_moment_accounting():
    path=exp.OUT/'checkpoint_64.pt'
    if not path.exists():
        return  # Early technical suite; this assertion executes again after the endpoint.
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['cursor']==64 and len(saved['logs'])==len(saved['native_gradients'])==64
    assert {int(s['step']) for s in saved['optimizer']['state'].values()}=={192}
    assert exp.b8._import_run().existing.equal(saved,torch.load(exp.OUT/'resumable.pt',map_location='cpu',weights_only=False))
    assert all(r['m_recurrence_relative_gap']<2e-5 and r['v_recurrence_relative_gap']<2e-5 for r in saved['history'])
    for index,r in enumerate(saved['logs']):
        assert r['ae17_identity']==exp.read('protocol')['physical_stream'][index]['ae17']['identity']
        assert r['mrks_id']==exp.read('protocol')['physical_stream'][index]['mrks_id']
