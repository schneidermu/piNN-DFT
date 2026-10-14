"""Bounded exact E251/E17/full90 endpoint, with matching archived controls."""
import time

import lap_f024_phys64 as exp
import numpy as np
import torch
from lap_f021_phys4_eval import constraints
from lap_f023_preflight import atomic_value


def chemistry_vectors(rows, jacobian, scales):
    residual=np.array([r['residual'] for r in rows])
    sign=residual/np.sqrt(residual**2+1e-20)
    jw=jacobian*(sign/np.array([scales[r['database']] for r in rows]))[:,None]
    legacy=jacobian*(sign*np.array([exp.audit.chemistry._factor(r['database']) for r in rows]))[:,None]
    return {'JW':jw.mean(axis=0),'legacy':legacy.mean(axis=0)}


def signed_exc(model,bundle,identity):
    from train_models.lap_vxc import LapEnergy, integrated_energy
    from train_models.optuna_joint import _add_mrks_dispersion_once
    material=bundle.mrks()[bundle.mrks().positions[identity]]
    with torch.no_grad():
        pred=integrated_energy(LapEnergy(model),material['DensityDescriptorsN10'].cuda().float()[:,None,:],
                               material['weights'].cuda().float(),point_chunk_size=4096)
        pred=_add_mrks_dispersion_once(pred,material['name'],exp.audit.read(bundle.root/'mrks/dispersion.json'),True)
        ref=material['Exc'].to(pred).reshape(())
    return {'signed_error_Ha':float(pred-ref),'prediction_Ha':float(pred),'reference_production_Ha':float(ref)}


def endpoint():
    started=time.perf_counter()
    path=exp.OUT/'checkpoint_64.pt'
    assert exp.audit.sha(path)==exp.read('training_receipt')['checkpoints']['64']
    contract={'checkpoint_sha256':exp.audit.sha(path),'driver_sha256':exp.audit.sha(__file__),
              'protocol_sha256':exp.audit.sha(exp.ROOT/(exp.PREFIX+'protocol.json'))}
    saved=exp.OUT/'endpoint_contract.json'
    if saved.exists(): assert exp.audit.read(saved)==contract
    else: exp.audit.write(saved,contract)
    model,shadow,run,bundle,parameters=exp.setup(path)
    before=run.existing.digest(model); rng=exp.b8._capture_rng()
    folder=exp.OUT/'endpoint';folder.mkdir(exist_ok=True)
    specs=exp.audit.chemistry._selected(bundle)
    scales=exp.audit.read(exp.ROOT/'lap_f010_normalization_receipt.json')['scales']
    dispersion=bundle.chemistry_dispersions()
    for stop in range(25,276,25):
        exp.check(120)
        rows,jacobian=exp.audit.chemistry._reaction_pass(model,shadow,bundle,dispersion,
              specs[:min(stop,251)],tuple(parameters),folder)
        if len(rows)==251: break
    assert all((r['identity'],r['variant'])==(s['identity'],s['variant']) for r,s in zip(rows,specs,strict=True))
    vectors=chemistry_vectors(rows,jacobian,scales)
    ev=exp.audit.read(exp.ROOT/'relchem_joint_epoch_evaluation_manifest.json')
    atoms=[{**r,'database':'AE17'} for r in ev['rows'] if r['task']=='ae17']
    atomfile=folder/'atoms.pt'
    atomic=torch.load(atomfile,weights_only=False) if atomfile.exists() else []
    for spec in atoms[len(atomic):]:
        exp.check(120)
        record=exp.b8._load_reaction(bundle,'ae17',spec)
        value,gradient=exp.b8._singleton(model,shadow,record,dispersion)
        data=atomic_value(model,shadow,bundle,spec)
        atomic.append({**spec,**data,'loss':float(value),'gradient':exp.flat(gradient),
                       'species':bundle.reactions[spec['identity']]['components'][0]})
        torch.save(atomic,atomfile)
        del record,gradient
    vectors['ae17']=np.mean([r['gradient'] for r in atomic],axis=0)
    H=next(r for r in atomic if r['species']=='H_ae17')
    vectors['H']=H['gradient']/exp.audit.chemistry._factor('AE17')
    physicalfile=folder/'physical.pt'
    physical=torch.load(physicalfile,weights_only=False) if physicalfile.exists() else {}
    md=exp.audit.read(bundle.root/'mrks/dispersion.json')
    for identity in sorted(bundle.systems):
        if identity in physical: continue
        exp.check(120)
        tick=time.perf_counter()
        system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
        factories=run.existing.core.make_mrks_objective_factories(model,system,point_chunk_size=256,
                                                               exc_chunk_size=4096,dispersions=md)
        values={};grads={}
        for task,objective in zip(('exc','op'),factories,strict=True):
            losses,raw=run.existing.core.compute_isolated_task_gradients(model,{task:objective},task_order=(task,))
            filled=run.existing.core.materialize_task_zeros(model,raw,task_order=(task,))
            values[task]=float(losses[task]);grads[task]=exp.flat(filled[task])
            del raw,filled,losses
        physical[identity]={**values,**grads_to_fields(grads),**signed_exc(model,bundle,identity),
                            'seconds':time.perf_counter()-tick}
        assert all(np.isfinite(x).all() for x in grads.values())
        torch.save(physical,physicalfile)
        del system,factories,grads
        torch.cuda.empty_cache()
        print('F024_ENDPOINT_PHYSICAL',len(physical),90,flush=True)
    for task in ('exc','op'):vectors[task]=np.mean([r[task+'_gradient'] for r in physical.values()],axis=0)
    np.savez_compressed(folder/'vectors.npz',**vectors)
    exp.write('constraint_checks',{'t64':constraints(model,shadow),'model_unchanged':run.existing.digest(model)==before})
    exp.b8._restore_rng(rng)
    assert run.existing.equal(rng,exp.b8._capture_rng())
    exp.audit.write(folder/'receipt.json',{**contract,'chemistry_count':251,'atomic_count':17,'mrks_count':90,
        'active_seconds':time.perf_counter()-started,'model_unchanged':True,'RNG_restored':True,
        'hashes':{p.name:exp.audit.sha(p) for p in folder.iterdir() if p.is_file()}})
    bundle.close()


def grads_to_fields(grads):
    return {task+'_gradient':v for task,v in grads.items()}


def controls():
    """Only missing fixed-control atomic/per-system scalars; no optimization."""
    tick=time.perf_counter()
    model,shadow,run,bundle,_parameters=exp.setup(exp.CONTROL)
    digest=run.existing.digest(model)
    path=exp.OUT/'control_scalars.json'
    data=exp.audit.read(path) if path.exists() else {'atoms':[],'physical':{}}
    atoms=[{**r,'database':'AE17'} for r in exp.audit.read(exp.ROOT/'relchem_joint_epoch_evaluation_manifest.json')['rows'] if r['task']=='ae17']
    for spec in atoms[len(data['atoms']):]:
        exp.check(120)
        result=atomic_value(model,shadow,bundle,spec)
        data['atoms'].append({**spec,**result,'species':bundle.reactions[spec['identity']]['components'][0]})
        exp.audit.write(path,data)
    md=exp.audit.read(bundle.root/'mrks/dispersion.json')
    for identity in sorted(bundle.systems):
        if identity in data['physical']:continue
        exp.check(120)
        system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
        fs=run.existing.core.make_mrks_objective_factories(model,system,point_chunk_size=256,exc_chunk_size=4096,dispersions=md)
        with torch.no_grad():result={t:float(f()) for t,f in zip(('exc','op'),fs,strict=True)}
        data['physical'][identity]=result
        exp.audit.write(path,data)
        del system,fs
        torch.cuda.empty_cache()
        print('F024_CONTROL_PHYSICAL',len(data['physical']),90,flush=True)
    official=exp.audit.read(exp.f010.OUT/'metrics_t64.json')
    for t in ('exc','op'):
        assert abs(np.mean([r[t] for r in data['physical'].values()])-official[t])<1e-8
    assert exp.audit.sha(exp.CONTROL)==exp.CONTROL_SHA and run.existing.digest(model)==digest
    exp.write('control_receipt',{'checkpoint_sha256':exp.CONTROL_SHA,'scalar_cache_sha256':exp.audit.sha(path),
        'active_seconds':time.perf_counter()-tick,'model_unchanged':True})
    bundle.close()


if __name__=='__main__':
    import sys
    if sys.argv[1]=='endpoint':endpoint()
    else:controls()
