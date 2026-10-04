"""Generic PCD and four-task direct-Armijo acceptance tests."""
from __future__ import annotations

import copy
import importlib.util
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lap_moo_protocol as protocol
import lap_moo_training as training
from moo_aggregators import _solve_pcd_qp, aggregate_task_gradients
from test_lap_moo_armijo import _checkpoint_manifest

TASKS = protocol.FOUR_TASK_NAMES
PCD = {"tau": .02, "beta": .999, "eps": 1e-8, "qp_tolerance": 1e-9}
ARMIJO = {"c": 1e-4, "rho": .5, "max_backtracks": 20, "initial_step_size": .1}


def reference():
    path = ROOT.parent.parent / 'lap_pcd_runs_20261002/upstream_pcd/pcd/qp.py'
    if not path.exists():
        pytest.skip('Pinned external author reference unavailable')
    spec = importlib.util.spec_from_file_location('upstream_qp', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_qp


@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_generic_random_qp_author_parity_and_kkt(count):
    solve = reference()
    rng = np.random.default_rng(70441 + count)
    for _ in range(150):
        vectors = rng.normal(size=(count, 8))
        gram = vectors @ vectors.T
        w, active, feasible, _, residual = _solve_pcd_qp(gram, .02)
        expected = solve(gram, .02)
        np.testing.assert_array_equal(w, expected.weights)
        assert active == expected.active and feasible == expected.feasible
        assert np.isfinite(residual)
        if feasible:
            d = w @ vectors
            margins = vectors[1:] @ d - .02 * np.diag(gram)[1:]
            assert np.min(margins) >= -1e-8
            assert np.min(w[1:]) >= 0
            np.testing.assert_allclose(w[1:] * margins, 0, atol=1e-8)
            np.testing.assert_allclose(d - vectors[0] - w[1:] @ vectors[1:], 0, atol=1e-12)


@pytest.mark.parametrize('active_count', [0, 1, 2, 3])
def test_four_active_sets(active_count):
    secondaries = np.eye(3)
    primary = np.ones(3)
    primary[:active_count] = -1
    vectors = np.vstack([primary, secondaries])
    w, active, feasible, _, _ = _solve_pcd_qp(vectors @ vectors.T, .02)
    assert feasible and active == tuple(range(1, active_count + 1))
    np.testing.assert_allclose(w[1:active_count+1], 1.02)


@pytest.mark.parametrize('rows,feasible', [
    ([[1,1], [1,0], [1,0], [0,0]], True),
    ([[0,1], [1,0], [-1,0], [0,0]], False),
    ([[1,0], [0,0], [0,0], [0,0]], True),
])
def test_degenerate_four(rows, feasible):
    vectors = np.array(rows, dtype=float)
    w, _, flag, _, _ = _solve_pcd_qp(vectors @ vectors.T, .02)
    assert flag is feasible
    if not feasible:
        np.testing.assert_array_equal(w, [1,0,0,0])


def gradients(rows, tasks=TASKS):
    return {task:{'p':torch.tensor(row,dtype=torch.float64)} for task,row in zip(tasks, rows, strict=True)}


def test_primary_zero_and_explicit_priority_resume():
    raw = gradients([[0,0], [1,0], [0,1], [1,1]])
    d, info, state = aggregate_task_gradients(raw, method='pcd', task_order=TASKS)
    assert torch.count_nonzero(d['p']) == 0 and info['solver_status'] == 'primary_zero'
    assert len(state['v']) == 4 and state['t'] == 1
    with pytest.raises(ValueError, match='explicit'):
        aggregate_task_gradients(raw, method='pcd')
    with pytest.raises(ValueError, match='order'):
        aggregate_task_gradients(raw, method='pcd', task_order=TASKS[::-1], state=state)
    with pytest.raises(ValueError):
        aggregate_task_gradients(raw, method='pcd', task_order=TASKS, state=state, hyperparameters={'beta':.9})


def test_three_task_prechange_exact_sequence():
    path = ROOT.parent.parent/'lap_s5_ae17_runs_20261004/pre_four_task_moo_aggregators.py'
    if not path.exists():
        pytest.skip('External prechange snapshot unavailable')
    spec = importlib.util.spec_from_file_location('old_aggregator',path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    rng = np.random.default_rng(34141)
    before = after = None
    for _ in range(100):
        raw = gradients(rng.normal(size=(3,7)), protocol.TASK_NAMES)
        a,ai,before = old.aggregate_task_gradients(raw,method='pcd',state=before)
        b,bi,after = aggregate_task_gradients(raw,method='pcd',state=after)
        assert torch.equal(a['p'], b['p']) and ai == bi and before == after


class Model(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__(); self.p = nn.Parameter(torch.tensor([1.],dtype=dtype))


def factories(model):
    return {t:lambda i=i:(i+1)*model.p.square().sum()/2 for i,t in enumerate(TASKS)}


def update(model, state=None):
    return training.train_moo_update(model,None,factories(model), method='pcd',hyperparameters=PCD,
        aggregator_state=state,task_order=TASKS,step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO)


def four_manifest(world_size=1):
    catalog=[{"database":"ABDE4","reaction_id":1,"variants":["default"]},
             {"database":"AE17","reaction_id":0,"variants":["default"]}]
    batches=[[{task:[{"database":db,"reaction_id":ix,"variant_suffix":"default","weight":1.0}]
              for task,db,ix in (("relchem","ABDE4",1),("ae17","AE17",0))}
              for _ in range(world_size)] for _ in range(4)]
    return protocol.build_sampling_manifest_from_catalog(catalog,["H2"],updates=4,seed=73,
        source_hashes={"test-fixture":"1"*64},world_size=world_size,chemistry_task_samples=batches)


def four_metadata(manifest, world_size=1):
    _,metadata,_=_checkpoint_manifest(world_size=world_size)
    metadata['protocol_version']=protocol.PCD_FOUR_TASK_PROTOCOL_VERSION
    metadata['task_order']=list(TASKS)
    metadata['objective_definitions']={t:t for t in TASKS}
    metadata['pcd_algorithm']=protocol._pcd_algorithm_metadata(TASKS)
    metadata['sampling_manifest_sha256']=manifest['manifest_sha256']
    return metadata


def test_four_armijo_checkpoint_resume_and_metadata(tmp_path):
    manifest = four_manifest()
    metadata = four_metadata(manifest)
    protocol.validate_protocol_metadata(metadata)
    model = Model(); first = update(model)
    assert first.accepted
    trial=first.diagnostics['vector_armijo']['trials'][-1]
    assert trial['realized_common_descent'] and len(trial['realized_task_dots'])==4
    assert all(v<0 for v in trial['realized_task_dots'].values())
    path=tmp_path/'four.pt'
    training.save_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,protocol_metadata=metadata,
        sampling_manifest=manifest,next_update=1,aggregator_state=first.aggregator_state)
    second=update(model,first.aggregator_state); expected=copy.deepcopy(model.state_dict())
    resumed=Model();cursor,state=training.load_moo_checkpoint(path,model=resumed,optimizer=None,scheduler=None,
        expected_protocol_metadata=metadata,sampling_manifest=manifest)
    assert cursor==1
    replay=update(resumed,state)
    assert torch.equal(expected['p'],resumed.p) and second.aggregator_state==replay.aggregator_state
    wrong=copy.deepcopy(metadata);wrong['task_order'][0:2]=wrong['task_order'][1::-1]
    with pytest.raises(ValueError):protocol.validate_protocol_metadata(wrong)


def test_quantized_realized_non_descent_is_rejected():
    model=Model(torch.float32);before=model.p.detach().clone()
    options={**ARMIJO,'initial_step_size':1e-15}
    result=training.train_moo_update(model,None,factories(model),method='pcd',task_order=TASKS,
        step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=options)
    assert not result.accepted and torch.equal(model.p,before) and result.aggregator_state is None


def test_chemistry_batch_shadow_mapping_and_scalar_identity(monkeypatch):
    model=Model(torch.float32);shadow=Model(torch.float64)
    def bind(m,reaction,**kwargs):
        assert kwargs['dtype']==torch.float64
        return lambda:reaction['coefficient']*m.p.square().sum()
    monkeypatch.setattr(training,'make_reaction_objective',bind)
    batch=training.ChemistryBatchObjective(model,shadow,({'coefficient':2.},{'coefficient':5.}),(.25,.75),None)
    value, grads=batch.value_and_grad()
    assert value==batch().item()==4.25 and grads['p'].item()==8.5 and grads['p'].dtype==torch.float64
    fs={t:batch for t in TASKS};losses,raw=training.compute_isolated_task_gradients(model,fs,task_order=TASKS)
    assert all(v==4.25 for v in losses.values()) and all(g['p'].dtype==torch.float64 for g in raw.values())
    assert model.p.grad is None and shadow.p.grad is None


def test_stratified_unbiasedness_exact_enumeration():
    from itertools import product
    dbs=[np.array([1.,3.]),np.array([2.,4.,8.])]
    draws=[sum(len(db)/5*db[ix] for db,ix in zip(dbs,indices,strict=True)) for indices in product(range(2),range(3))]
    assert np.mean(draws)==pytest.approx(np.concatenate(dbs).mean(),abs=1e-15)


def _ddp_worker(rank, port, directory):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        model=Model();fs={t:lambda i=i:(i+1+rank)*model.p.square().sum()/2 for i,t in enumerate(TASKS)}
        losses,raw=training.compute_isolated_task_gradients(model,fs,task_order=TASKS)
        avg=training.average_raw_task_gradients(training.materialize_task_zeros(model,raw,task_order=TASKS),task_order=TASKS,world_size=2)
        for i,t in enumerate(TASKS):assert avg[t]['p'].item()==i+1.5
        mean=training._average_task_losses(losses,model)
        for i,t in enumerate(TASKS):assert mean[t]==(i+1.5)/2
        result=training.train_moo_update(model,None,fs,method='pcd',task_order=TASKS,world_size=2,
            step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO)
        gathered=[None,None];dist.all_gather_object(gathered,(model.p.item(),result.aggregator_state,result.diagnostics))
        assert gathered[0]==gathered[1] and result.accepted
        manifest=four_manifest(world_size=2);metadata=four_metadata(manifest,world_size=2)
        checkpoint=Path(directory)/'ddp.pt'
        training.save_moo_checkpoint(checkpoint,model=model,optimizer=None,scheduler=None,
            protocol_metadata=metadata,sampling_manifest=manifest,next_update=1,aggregator_state=result.aggregator_state)
        dist.barrier()
        second=training.train_moo_update(model,None,fs,method='pcd',task_order=TASKS,world_size=2,
            aggregator_state=result.aggregator_state,step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO)
        expected=model.p.detach().clone()
        cursor,state=training.load_moo_checkpoint(checkpoint,model=model,optimizer=None,scheduler=None,
            expected_protocol_metadata=metadata,sampling_manifest=manifest)
        replay=training.train_moo_update(model,None,fs,method='pcd',task_order=TASKS,world_size=2,
            aggregator_state=state,step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO)
        assert cursor==1 and torch.equal(model.p,expected) and replay.aggregator_state==second.aggregator_state
        Path(directory,f'rank{rank}.json').write_text(json.dumps({'passed':True}))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(sys.platform=='win32',reason='Gloo executed under WSL')
def test_four_task_two_rank_raw_before_pcd(tmp_path):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    torch.multiprocessing.spawn(_ddp_worker,args=(port,str(tmp_path)),nprocs=2,join=True)
    assert all(json.loads((tmp_path/f'rank{r}.json').read_text())['passed'] for r in range(2))


def test_four_sampling_identity_and_fail_closed():
    manifest=four_manifest(world_size=2)
    assert manifest==four_manifest(world_size=2)
    protocol.validate_sampling_manifest(manifest)
    tampered=copy.deepcopy(manifest)
    tampered['entries'][0]['per_rank'][0]['task_samples']['relchem'][0]['database']='AE17'
    tampered['manifest_sha256']=protocol.canonical_sha256(protocol._manifest_payload(tampered))
    with pytest.raises(ValueError):protocol.validate_sampling_manifest(tampered)


def test_four_protocol_builder_and_manifest_resume_rejection(tmp_path):
    manifest=four_manifest()
    metadata=protocol.make_protocol_metadata(
        architecture="scalar",method="pcd",method_hyperparameters=PCD,fixed_scalarization=None,
        optimizer={"name":"none"},lr_schedule={"name":"none"},predopt_checkpoint_sha256="1"*64,
        sampling_manifest_sha256=manifest['manifest_sha256'],minnesota_data_sha256="2"*64,
        operator_corpus_manifest_sha256="3"*64,ao_cache_manifest_sha256="4"*64,
        reaction_dispersions_sha256="5"*64,mrks_dispersions_sha256="6"*64,random_seed=73,dtype="float32",
        grid_chunk_size=256,ao_cache_chunk_size=4096,world_size=1,sampling_manifest_file_sha256="7"*64,
        source_code_sha256={p:"8"*64 for p in protocol.PCD_SOURCE_FILE_PATHS},
        step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO,task_order=TASKS)
    protocol.validate_protocol_metadata(metadata)
    assert set(metadata['objective_definitions'])==set(TASKS)
    model=Model(); result=update(model);path=tmp_path/'bound.pt'
    training.save_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,protocol_metadata=metadata,
        sampling_manifest=manifest,next_update=1,aggregator_state=result.aggregator_state)
    altered=copy.deepcopy(manifest);altered['seed']+=1
    altered['manifest_sha256']=protocol.canonical_sha256(protocol._manifest_payload(altered))
    with pytest.raises(ValueError):
        training.load_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
            expected_protocol_metadata=metadata,sampling_manifest=altered)


def test_four_normalizer_sequence_author_parity():
    upstream=ROOT.parent.parent/'lap_pcd_runs_20261002/upstream_pcd'
    if not (upstream/'pcd/optim.py').exists():pytest.skip('Pinned reference unavailable')
    sys.path.insert(0,str(upstream))
    from pcd.optim import PCD as ReferencePCD
    parameter=nn.Parameter(torch.zeros(7,dtype=torch.float64));ref=ReferencePCD([parameter],tau=.02)
    rng=np.random.default_rng(74141);state=None
    for step in range(80):
        rows=rng.normal(size=(4,7))*10.**rng.uniform(-3,6,size=(4,1))
        tensors=[torch.tensor(row,dtype=torch.float64) for row in rows]
        expected=ref.apply_gradients([[g] for g in tensors])
        d,info,state=aggregate_task_gradients(gradients(rows),method='pcd',task_order=TASKS,state=state)
        torch.testing.assert_close(d['p'],parameter.grad,rtol=2e-10,atol=1e-10)
        assert info['active_indices']==list(expected.active) and info['feasible']==expected.feasible
        np.testing.assert_allclose(list(info['mu'].values()),expected.mu,rtol=2e-10,atol=1e-10)
        np.testing.assert_allclose(state['v'],ref.normalizer.v,rtol=2e-10,atol=1e-10)
        assert state['t']==step+1
        state=json.loads(json.dumps(state))


def test_realized_descent_distortion_detected_before_acceptance():
    model=nn.Linear(2,1,bias=False,dtype=torch.float32)
    with torch.no_grad():model.weight.copy_(torch.tensor([[1.,0.]]))
    vectors=[torch.tensor([[1.,1.]]),torch.tensor([[1e10,-1.]]),torch.tensor([[1.,1.]]),torch.tensor([[1.,1.]])]
    fs={task:lambda v=v:(model.weight*v).sum() for task,v in zip(TASKS,vectors,strict=True)}
    def aggregator(*args,**kwargs):
        return {'weight':torch.tensor([[1e-8,1.]],dtype=torch.float64)}, {}, {}
    result=training.train_moo_update(model,None,fs,method='pcd',task_order=TASKS,aggregator=aggregator,
        step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo={**ARMIJO,'initial_step_size':.001,'max_backtracks':0})
    assert not result.accepted
    trial=result.diagnostics['vector_armijo']['trials'][0]
    assert all(v<0 for v in trial['requested_task_dots'].values())
    assert trial['realized_task_dots']['ae17']>0 and not trial['realized_common_descent']


def test_mixed_precision_chemistry_shadow_reaches_real_armijo(monkeypatch):
    model=Model(torch.float32);shadow=Model(torch.float64)
    monkeypatch.setattr(training,'make_reaction_objective',lambda m,r,**kw:lambda:m.p.square().sum()/2)
    batch=training.ChemistryBatchObjective(model,shadow,({},),(1.,),None)
    fs={'relchem':batch,'ae17':batch,'exc':lambda:model.p.square().sum(),
        'op':lambda:2*model.p.square().sum()}
    _losses,raw=training.compute_isolated_task_gradients(model,fs,task_order=TASKS)
    assert all(g['p'].dtype==torch.float64 for g in raw.values())
    result=training.train_moo_update(model,None,fs,method='pcd',task_order=TASKS,
        step_rule=protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,vector_armijo=ARMIJO)
    assert result.accepted and all(v<0 for v in result.diagnostics['task_update_dots'].values())


def test_rank_deficient_working_subset_skipped_with_author_parity():
    rows=np.array([[-1.,-1.],[1.,0.],[1.,0.],[0.,1.]])
    w,active,feasible,_,_= _solve_pcd_qp(rows@rows.T,.02)
    expected=reference()(rows@rows.T,.02)
    assert feasible and active==(1,3) and active==expected.active
    np.testing.assert_array_equal(w,expected.weights)
    np.testing.assert_allclose(w@rows,[.02,.02],atol=1e-14)
