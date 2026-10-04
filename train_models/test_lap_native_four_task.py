"""Native CLI boundaries; lower-level algorithm tests remain in their existing suites."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))

import lap_moo_protocol as protocol
import lap_moo_training as training
import train_lap_moo as cli
from test_lap_four_task_pcd import Model, four_manifest, four_metadata, update


def test_native_manifest_modes_and_inventory(tmp_path):
    manifest = four_manifest()
    path = tmp_path / 'manifest.json'
    protocol.write_sampling_manifest(path, manifest)
    kwargs = {'catalog':manifest['reaction_catalog'], 'system_names':manifest['mrks_systems'],
                  'source_hashes':manifest['source_hashes'], 'updates':4, 'seed':73, 'world_size':1, 'rank':0}
    assert cli._load_sampling_manifest(path, four_task=True, **kwargs) == manifest
    with pytest.raises(ValueError, match='v1'):
        cli._load_sampling_manifest(path, **kwargs)
    for key, value in [('seed', 74), ('world_size', 2), ('updates', 3), ('source_hashes', {'different': 'a'*64})]:
        with pytest.raises(ValueError):
            cli._load_sampling_manifest(path, four_task=True, **{**kwargs, key:value})
    v1 = protocol.build_sampling_manifest_from_catalog(kwargs['catalog'], ['H2'], updates=4,
         seed=73, source_hashes=kwargs['source_hashes'])
    protocol.write_sampling_manifest(path, v1)
    with pytest.raises(ValueError):
        cli._load_sampling_manifest(path, four_task=True, **kwargs)
    assert cli._load_sampling_manifest(path, **kwargs) == v1


def test_native_exact_batch_construction_and_task_order(monkeypatch):
    model = Model(torch.float32)
    shadow = copy.deepcopy(model).double()
    sample = four_manifest()['entries'][0]['per_rank'][0]
    sample['task_samples']['relchem'] *= 2
    for row in sample['task_samples']['relchem']: row['weight'] = .5
    loaded = []
    class Store:
        def load_variant(self, identity, variant):
            loaded.append((identity, variant))
            return {'scale':len(loaded)}
    def reaction_factory(model, reaction, **kwargs):
        assert kwargs['dtype'] == torch.float64
        return lambda: model.p.square().sum() * reaction['scale']
    monkeypatch.setattr(training, 'make_reaction_objective', reaction_factory)
    operator = lambda: model.p.square().sum()*3
    energy = lambda: model.p.square().sum()*2
    monkeypatch.setattr(cli, 'make_mrks_objective_factories', lambda *a, **k:(energy,operator))
    objectives = cli._four_task_objectives(model, shadow, sample, Store(), None, {}, {}, 256)
    assert tuple(objectives) == protocol.FOUR_TASK_NAMES
    assert loaded == [(('ABDE4',1),'default')]*2 + [(('AE17',0),'default')]
    assert objectives['relchem'].weights == (.5,.5)
    value, gradient = objectives['relchem'].value_and_grad()
    assert value == 1.5 and gradient['p'].dtype == torch.float64
    assert model.p.dtype == torch.float32 and shadow.p.dtype == torch.float64
    assert model.p.grad is None
    captured = {}
    def updater(*args, **kwargs): captured.update(kwargs); return 'sentinel'
    monkeypatch.setattr(cli, 'train_moo_update', updater)
    assert cli._run_scheduled_update(model,None,objectives,method='pcd',hyperparameters={},
           aggregator_state=None,scheduler=None,world_size=1,task_order=protocol.FOUR_TASK_NAMES) == 'sentinel'
    assert captured['task_order'] == protocol.FOUR_TASK_NAMES


def test_v6_precision_provenance_resume_and_legacy_readability(tmp_path):
    manifest = four_manifest()
    old = four_metadata(manifest)
    protocol.validate_protocol_metadata(old)
    metadata = copy.deepcopy(old)
    metadata['protocol_version'] = protocol.PCD_FOUR_TASK_PRECISION_PROTOCOL_VERSION
    metadata['operator_precision_source_sha256'] = {
        p:protocol.file_sha256(cli.REPO_ROOT/p) for p in protocol.OPERATOR_PRECISION_SOURCE_PATHS}
    protocol.validate_protocol_metadata(metadata)
    model = Model(); first = update(model)
    path = tmp_path/'v6.pt'
    training.save_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
        protocol_metadata=metadata,sampling_manifest=manifest,next_update=1,aggregator_state=first.aggregator_state)
    cursor,state = training.load_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
        expected_protocol_metadata=metadata,sampling_manifest=manifest)
    assert cursor == 1 and state == first.aggregator_state
    wrong = copy.deepcopy(metadata)
    wrong['operator_precision_source_sha256']['train_models/lap_operator.py'] = 'a'*64
    with pytest.raises(ValueError,match='precision source'):
        training.load_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
            expected_protocol_metadata=wrong,sampling_manifest=manifest)
    retro = copy.deepcopy(old);retro['operator_precision_source_sha256']=metadata['operator_precision_source_sha256']
    with pytest.raises(ValueError,match='retrofitted'):protocol.validate_protocol_metadata(retro)
    # Old v5 stays readable; it cannot be resumed as a newly precision-bound v6.
    training.save_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
        protocol_metadata=old,sampling_manifest=manifest,next_update=1,aggregator_state=state)
    with pytest.raises(ValueError,match='differs'):
        training.load_moo_checkpoint(path,model=model,optimizer=None,scheduler=None,
            expected_protocol_metadata=metadata,sampling_manifest=manifest)


def _native_fixture_worker(rank, port, directory, world_size, reject=False):
    """Fixture replaces only external data I/O/local formulas; native loop owns all state/DDP."""
    import json
    import os
    from types import SimpleNamespace
    from unittest.mock import patch
    D = Path(directory)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    files = {k:D/k for k in ('predopt','store','central','ao','rd','md')}
    source = {'predopt_checkpoint':protocol.file_sha256(files['predopt']),
              'minnesota_group_store_manifest':protocol.file_sha256(files['store']),
              'central_operator_manifest':protocol.file_sha256(files['central']),
              'ao_factor_cache_manifest':protocol.file_sha256(files['ao']),
              'reaction_dispersions':protocol.file_sha256(files['rd']),
              'mrks_dispersions':protocol.file_sha256(files['md'])}
    manifest_path=D/'fixture_manifest.json'
    manifest=protocol.read_sampling_manifest(manifest_path)
    assert manifest['source_hashes']==source
    catalog=manifest['reaction_catalog']
    loaded=[]
    class Store:
        def __init__(self,*args,**kwargs):
            self.manifest_sha256=source['minnesota_group_store_manifest']
            self.manifest={'group_count':268,'groups':[{'database':r['database'],'reaction_id':r['reaction_id'],
                         'variant_suffixes':r['variants']} for r in catalog]}
        def load_variant(self,identity,variant):
            loaded.append((identity,variant))
            return {'scale':(1 if identity[0]=='ABDE4' else 3)+(variant=='alternate')}
    class Cache:
        def __init__(self,*args,**kwargs):
            self.system_names=['H2'];self.central_manifest_sha256=source['central_operator_manifest']
            self.cache_manifest_sha256=source['ao_factor_cache_manifest']
            self.last_load_seconds=0;self.last_system_cache_bytes=0
        def load(self,name):return SimpleNamespace(name=name)
    def load_model(*args,**kwargs):
        m=Model(torch.float32);m.architecture='scalar';m.model_kwargs={}
        return m,{'predopt_only':True,'predopt_epochs':2,'predopt_lr':.01,'predopt_seed':41}
    def reaction_factory(model,reaction,**kwargs):
        return lambda:reaction['scale']*model.p.square().sum()
    def mrks(model,system,**kwargs):
        return (lambda:(rank+1)*model.p.square().sum(),lambda:(-1 if reject else 1)*(rank+2)*model.p.square().sum())
    update_results=[]
    writes=[];real_save=torch.save
    def rank0_save(*args,**kwargs):
        if isinstance(args[1],(str,Path)):
            assert rank==0
            writes.append(str(args[1]))
        return real_save(*args,**kwargs)
    real_update=cli._run_scheduled_update
    def checked_update(model,*args,**kwargs):
        assert kwargs['task_order']==protocol.FOUR_TASK_NAMES
        p=float(model.p.item());result=real_update(model,*args,**kwargs)
        expected=[(sum(2*(1+r)*p for r in range(world_size))/world_size),
                  (sum(2*(3+r)*p for r in range(world_size))/world_size)]
        for offset in (1,2):
            local=[]
            for r in range(world_size):
                leaf=torch.tensor([p],dtype=torch.float32,requires_grad=True)
                g=torch.autograd.grad((r+offset)*leaf.square().sum(),leaf)[0]
                local.append(float(g.double().item()))
            expected.append(sum(local)/world_size)
        for task,value in zip(protocol.FOUR_TASK_NAMES,expected,strict=True):
            assert result.diagnostics['raw_gradient_norms'][task]==value
        assert result.accepted != reject
        update_results.append(result)
        return result
    base=['native','--four-task-pcd','--predopt-checkpoint',str(files['predopt']),
          '--minnesota-store-manifest',str(files['store']),'--central-data-dir',str(files['central']),
          '--ao-cache-dir',str(files['ao']),'--reaction-dispersions',str(files['rd']),
          '--mrks-dispersions',str(files['md']),'--sampling-manifest',str(manifest_path),
          '--method','pcd','--updates','2','--learning-rate','.1','--device','cpu',
          '--step-rule',protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,'--checkpoint-every','1']
    with patch.object(cli,'MinnesotaGroupStore',Store),patch.object(cli,'CentralAOCache',Cache), \
         patch.object(cli,'load_lap_checkpoint',load_model),patch.object(cli,'load_reaction_dispersions',lambda x:{}), \
         patch.object(cli,'load_mrks_dispersions',lambda x:{}),patch.object(cli,'make_mrks_objective_factories',mrks), \
         patch.object(training,'make_reaction_objective',reaction_factory),patch.object(cli,'_run_scheduled_update',checked_update), \
         patch.object(torch,'save',rank0_save):
        runs=[('R',1,False)] if reject else [('A',2,False),('B',1,False),('B',2,True)]
        for session,(label,stop,resume) in enumerate(runs):
            # Each fresh CLI invocation gets a fresh rendezvous, as separate torchrun jobs do.
            os.environ["MASTER_PORT"]=str(port+session)
            argv=base+['--output-dir',str(D/label),'--stop-after',str(stop)]
            if resume:argv+=['--resume',str(D/label/'latest.pt')]
            with patch.object(sys,'argv',argv):cli.main()
    if reject:
        rejected=torch.load(D/'R/latest.pt',weights_only=False)
        assert rejected['sampling_cursor']['next_update']==0
        assert rejected['aggregator_state']['t']==0 and all(v==0 for v in rejected['aggregator_state']['v'])
        assert rejected['model_state_dict']['p'].item()==1.0
        assert update_results[0].stop_reason is not None
        return
    A=torch.load(D/'A/latest.pt',weights_only=False);B=torch.load(D/'B/latest.pt',weights_only=False)
    assert A['protocol_metadata']==B['protocol_metadata']
    assert A['aggregator_state']==B['aggregator_state'] and A['sampling_cursor']==B['sampling_cursor']
    assert A['sampling_cursor']['next_update']==2
    assert torch.equal(A['model_state_dict']['p'],B['model_state_dict']['p'])
    assert A['protocol_metadata']['protocol_version']==protocol.PCD_FOUR_TASK_PRECISION_PROTOCOL_VERSION
    assert A['optimizer_state_dict'] is None and A['scheduler_state_dict'] is None
    assert update_results[1].diagnostics==update_results[3].diagnostics
    for x,y in zip(A['rng_states_by_rank'],B['rng_states_by_rank'],strict=True):
        assert torch.equal(x['torch_cpu'],y['torch_cpu'])
        assert x['python']==y['python']
        import numpy as np
        assert np.array_equal(x['numpy'][1],y['numpy'][1])
        assert x['numpy'][0]==y['numpy'][0] and x['numpy'][2:]==y['numpy'][2:]
        assert len(x['torch_cuda'])==len(y['torch_cuda'])
        assert all(torch.equal(a,b) for a,b in zip(x['torch_cuda'],y['torch_cuda'],strict=True))
    assert len(writes)==4 if rank==0 else not writes
    expected_variant='default' if rank==0 else 'alternate'
    assert all(v==expected_variant for _,v in loaded)
    (D/f'rank{rank}.json').write_text(json.dumps({'passed':True,'loaded':loaded,
        'final_parameter':float(B['model_state_dict']['p'].item()),'EMA':B['aggregator_state'],
        'cursor':B['sampling_cursor'],'last_diagnostics':update_results[-1].diagnostics}))


def _fixture_files(tmp_path,world_size):
    # No scientific training artifacts; tiny deterministic CPU-only fixture.
    for name in ('predopt','store','central','ao','rd','md'):(tmp_path/name).write_text(name)
    source={k:protocol.file_sha256(tmp_path/v) for k,v in [
        ('predopt_checkpoint','predopt'),('minnesota_group_store_manifest','store'),
        ('central_operator_manifest','central'),('ao_factor_cache_manifest','ao'),
        ('reaction_dispersions','rd'),('mrks_dispersions','md')]}
    catalog=[{'database':'ABDE4','reaction_id':1,'variants':['default','alternate']},
             {'database':'AE17','reaction_id':0,'variants':['default','alternate']}]
    batches=[[{t:[{'database':db,'reaction_id':ix,'variant_suffix':'default' if r==0 else 'alternate','weight':1.0}]
             for t,db,ix in [('relchem','ABDE4',1),('ae17','AE17',0)]} for r in range(world_size)] for _ in range(2)]
    manifest=protocol.build_sampling_manifest_from_catalog(catalog,['H2'],updates=2,seed=41,
             source_hashes=source,world_size=world_size,chemistry_task_samples=batches)
    protocol.write_sampling_manifest(tmp_path/'fixture_manifest.json',manifest)


def test_native_main_resume_fixture(tmp_path):
    _fixture_files(tmp_path,1)
    _native_fixture_worker(0,29591,str(tmp_path),1)


@pytest.mark.skipif(sys.platform=='win32',reason='Native two-rank Gloo qualified under WSL')
def test_native_main_two_rank_resume_fixture(tmp_path):
    import json
    import socket
    _fixture_files(tmp_path,2)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    torch.multiprocessing.spawn(_native_fixture_worker,args=(port,str(tmp_path),2),nprocs=2,join=True)
    rows=[json.loads((tmp_path/f'rank{r}.json').read_text()) for r in range(2)]
    assert rows[0]['loaded']!=rows[1]['loaded']
    for key in ('final_parameter','EMA','cursor','last_diagnostics'):assert rows[0][key]==rows[1][key]


@pytest.mark.parametrize("extra", [["--method","fixed"],["--step-rule","optimizer"],
                                   ["--dtype","float64"],["--optimizer-family","adamw"]])
def test_four_task_cli_rejects_incompatible_modes(monkeypatch,extra):
    argv=['native','--four-task-pcd','--predopt-checkpoint','missing','--minnesota-store-manifest','missing',
          '--central-data-dir','missing','--ao-cache-dir','missing','--sampling-manifest','missing',
          '--output-dir','missing','--method','pcd','--updates','2','--learning-rate','.1',
          '--step-rule',protocol.PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE]+extra
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='Four-task mode'):cli.main()


def test_native_rejection_preserves_committed_cursor_model_EMA(tmp_path):
    _fixture_files(tmp_path,1)
    _native_fixture_worker(0,29592,str(tmp_path),1,reject=True)


def test_native_locked_catalog_does_not_resample(tmp_path,monkeypatch):
    manifest=four_manifest()
    path=tmp_path/'locked.json';protocol.write_sampling_manifest(path,manifest)
    catalog=copy.deepcopy(manifest['reaction_catalog'])
    for row in catalog:row['variants'].append('another_available_variant')
    monkeypatch.setattr(cli,'build_sampling_manifest_from_catalog',lambda *a,**k:pytest.fail('v2 must never regenerate draws'))
    assert cli._load_sampling_manifest(path,catalog=catalog,system_names=['H2'],source_hashes=manifest['source_hashes'],
        updates=4,seed=73,world_size=1,rank=0,four_task=True)==manifest
