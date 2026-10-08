import copy

import numpy as np
import torch

from tools import relchem_accumulation_sweep as s


def test_detached_f64_mean_and_normalization():
    total=None
    vectors=[]
    for i in range(6):
        g={'x':torch.tensor([i+1.,2*i-3.],dtype=torch.float64,requires_grad=True)}
        vectors.append(g['x'].detach())
        total=s.add_gradient(total,g)
        assert not total['x'].requires_grad and total['x'].dtype==torch.float64
        if i+1 in s.KS:
            assert torch.equal(s.mean_gradient(total,i+1)['x'],torch.stack(vectors).mean(0))
    assert torch.equal(s.mean_gradient(total,6)['x']*6,total['x'])


def test_nested_samples_preserve_uniform_identity_variant_hierarchy():
    rows=[{'id':str(i),'task':'relchem','database':'db','reaction_id':i,
           'variants':{str(j):{} for j in range(8)}} for i in range(251)]
    a=s.extra_samples(rows,9)
    assert a==s.extra_samples(rows,9) and len(a)==5
    assert a[:1]==a[:3][:1] and a[:3]==a[:5][:3]
    assert all(r['weight']==1. and r['identity'] in {x['id'] for x in rows} for r in a)


def test_accumulation_resume_native_adamw(tmp_path,monkeypatch):
    run=s.run
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    data=tmp_path/'data'; (data/'mrks').mkdir(parents=True)
    run.write(data/'mrks/dispersion.json',{})
    monkeypatch.setattr(run,'DATA',data)
    class Bundle:
        def __init__(self,*args): self.manifest={'logical_sha256':run.DATA_SHA}
        def chemistry_dispersions(self): return {}
        def close(self): pass
    monkeypatch.setattr(run,'PublicationDataset',Bundle)
    def model_at(state):
        m=torch.nn.Linear(2,1)
        return m,copy.deepcopy(m).double()
    monkeypatch.setattr(run,'model_at',model_at)
    calls=[]
    def base(model,shadow,bundle,entry,*args):
        calls.append('base')
        params=run.existing.named_trainable_parameters(model)
        raw={t:{n:p.detach().double()+j+1 for n,p in params.items()} for j,t in enumerate(run.TASKS)}
        return {'sample':entry,'losses':dict.fromkeys(run.TASKS,1.),'norms':dict.fromkeys(run.TASKS,1.),
                'seconds':{'chemistry_loading_transfers':.01,'relchem':.01},'grid_points':{'relchem':10},
                'peak_allocated_bytes':0,'peak_reserved_bytes':0},raw
    def extra(model,shadow,bundle,row,dispersion):
        calls.append('extra')
        return 2.,{n:p.detach().double()+row['value'] for n,p in run.existing.named_trainable_parameters(model).items()},.01,.01,10
    monkeypatch.setattr(s,'BASE_MEASURE',base)
    monkeypatch.setattr(s,'singleton',extra)
    monkeypatch.setattr(run,'measure',s.accumulated_measure(4))
    def folder(name):
        p=tmp_path/name; p.mkdir()
        run.write(p/'sampling_manifest.json',[{'cursor':i,'relchem':{'value':1},'relchem_extra':[{'value':j} for j in (2,3,4)]} for i in range(3)])
        run.write(p/'calibration.json',{'lambda':dict.fromkeys(run.TASKS,.25),'manifest_sha256':run.file_sha256(p/'sampling_manifest.json')})
        run.write(p/'protocol.json',{'dataset_sha256':run.DATA_SHA,'initial_state':{},'exc_chunk_size':4096,
                  'adamw':{'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'foreach':False}})
        return p
    a,b=folder('resume'),folder('whole')
    run.train(a,3,stop_at=1,learning_rate=1e-4,constant_lr=True,diagnostics=True)
    run.train(a,3,learning_rate=1e-4,constant_lr=True,diagnostics=True)
    run.train(b,3,learning_rate=1e-4,constant_lr=True,diagnostics=True)
    x=torch.load(a/'ordinary_sgd_adamw/latest.pt',weights_only=False); y=torch.load(b/'ordinary_sgd_adamw/latest.pt',weights_only=False)
    assert run.existing.equal(x['model'],y['model']) and run.existing.equal(x['optimizer'],y['optimizer'])
    assert run.existing.equal(x['rng'],y['rng']) and x['cursor']==3
    assert calls.count('base')==6 and calls.count('extra')==18
    assert [r['moment_step'] for r in x['logs']]==[1,2,3]
    with np.load(a/'ordinary_sgd_adamw/raw_gradients/update_000.npz') as g:
        assert g['relchem'].dtype==np.float64
        assert np.array_equal(g['ae17']-g['relchem'],np.full_like(g['relchem'],-.5))
from tools.summarize_relchem_accumulation import paired_changes


def test_paired_identical_and_scaled_probe():
    rows={t+str(i):{'task':t,'stratum':'s','weight':.5,'loss':float(i+1)}
          for t in ('relchem','ae17','exc','op') for i in range(2)}
    baseline={'rows':rows}
    half={'rows':{k:{**v,'loss':v['loss']/2} for k,v in rows.items()}}
    result=paired_changes({'1':baseline,'2':half},baseline)
    assert all(r['ratio95']==[1.,1.] for r in result['1'].values())
    assert all(r['ratio95']==[.5,.5] and r['difference_vs_K1_95']==[-.5,-.5] for r in result['2'].values())
