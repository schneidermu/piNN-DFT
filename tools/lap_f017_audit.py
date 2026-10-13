"""F017 training-only reference audit. No optimizer and no checkpoint writes."""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as prior
import lap_f016_audit as f016
import numpy as np
import torch
from lap_f015_inputs import restored
from lap_f016_descriptors import DiagnosticEnergy, DiagnosticQModel, descriptors
from torch import nn

from train_models.lap_operator import _inverse_sqrt_overlap, assemble_rks_operator
from train_models.lap_vxc import LapEnergy, integrated_energy

ROOT=prior.ROOT
OUT=ROOT.parent/'lap_f017_audit_20261013'
OUTPUTS=['representation_contract.md','reference_pair_analysis.csv','reference_observable_metrics.csv',
         'new_weight_jacobian_checks.json','tangent_rank_conditioning.csv','reference_residual_alignment.csv',
         'local_attainability.csv','finite_step_validation.csv','operator_safety.csv','chemistry_transfer.csv']


def freeze():
    f016.verify()
    OUT.mkdir(exist_ok=True)
    target=ROOT/'lap_f017_protocol.json'
    if target.exists():
        verify()
        return
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='3c0b9d61d4826e23c185e8e72601f8452774cb12'
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/lap_full_vxc'],cwd=ROOT,check=True)
    metrics=prior.read(ROOT/'lap_all90_operator_preparation_metrics.json')
    records=[r for r in metrics['records'] if r['historical_array_parity']=='PASS']
    assert len(records)==15
    names=[r['system'] for r in sorted(records,key=lambda r:(r['n_grid'],r['n_ao'],r['system']))]
    pins={str(p):prior.sha(p) for p in ROOT.glob('lap_f01*') if p.is_file()}
    for filename in ('lap_all90_operator_preparation_report.md','lap_all90_operator_preparation_protocol.json',
                     'lap_all90_operator_preparation_metrics.json'):
        pins[str(ROOT/filename)]=prior.sha(ROOT/filename)
    for filename in ('NN_models_lap.py','NN_models.py','lap_operator.py','lap_operator_data.py','lap_training.py'):
        pins[str(ROOT/'train_models'/filename)]=prior.sha(ROOT/'train_models'/filename)
    for filename in ('mrks/systems.jsonl','mrks/dispersion.json'):
        p=ROOT.parent/'publication_dataset_v1'/filename
        pins[str(p)]=prior.sha(p)
    prior.write(target,{'experiment':'F017','parent':head,'pins':pins,'updates':0,'training':False,
                       'primary':'b8_t128','robustness':['f010_t128','f010_t251'],
                       'operator_panel':names,'panel_order':'ascending n_grid,nAO,name; independent of reference residuals',
                       'pair_rule':'for each system, nearest other system in fixed weighted first/second descriptor moments; ties lexicographic; not an equivalence certificate',
                       'pair_scaling':'rho log1p; sigma sqrt; lap signed log1p for raw proxy; bounded R0; R2 q blocks /sqrt2; positive electron normalized moments plus total electron mass; signed mass separately',
                       'reference_blocks':{'energy':'Ha; corrected reference = stored Exc minus existing named dispersion correction; no artificial system-size loss normalization',
                                           'operator':'svec(S^-1/2 V S^-1/2)/sqrt(nAO); preserves qualified Frobenius loss'},
                       'energy_analysis':'energy-only squared residual diagnostic; production Exc uses 627.5095*sqrt(1e-20+error_Ha^2), not MSE',
                       'joint_analysis':'not authorized to invent relative energy/operator scale; report separate blocks unless justified fixed objective scaling is available',
                       'radius':prior.RADIUS,'finite_scales':[.25,1.],
                       'rank_thresholds':[1e-6,1e-8,1e-10],'main_rank_threshold':1e-8,
                       'tolerances':{'embedding_energy_Ha':1e-6,'embedding_operator':1e-10,
                                     'jvp_fd_abs':1e-6,'jvp_fd_rel':.005,'fd_parameter_step':1e-4,
                                     'projection_residual_rel':1e-6,'solver_kkt_rel':1e-6},
                       'resource_gate':{'panel_seconds_max':1800,'dense_operator_GiB_max':2,
                                        'rule':'benchmark full-system parameter JVP before 96-column sweep; stop affected branch if projected panel exceeds budget or JVP numerical validation fails'},
                       'effect_size':'report relative residual reduction; benefits below1e-3 are not substantial without numerical separation; no architecture promotion from partial coverage',
                       'intended_outputs':['lap_f017_'+n for n in OUTPUTS],
                       'no_validation_or_test_inputs':True,'one_variant_per_chemistry_identity':True})
    verify()


def verify():
    f016.verify()
    protocol=prior.read(ROOT/'lap_f017_protocol.json')
    for p,digest in protocol['pins'].items():
        assert prior.sha(p)==digest,p
    data=ROOT.parent/'publication_dataset_v1/mrks/systems.jsonl'
    import json
    systems=[json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    prepared=prior.read(ROOT/'lap_all90_operator_preparation_metrics.json')['records']
    lookup={r['system']:r for r in prepared}
    assert len(systems)==len(lookup)==90
    for row in systems:
        oracle=lookup[row['source_id']]
        assert row['source_sha256']==oracle['source_sha256']
        assert row['ao_cache_sha256']==oracle['ao_cache_sha256']
        assert row['n_ao']==oracle['n_ao'] and row['n_grid']==oracle['n_grid']
    return protocol


def svec(matrix):
    n=len(matrix)
    i,j=torch.triu_indices(n,n,device=matrix.device)
    return matrix[i,j]*torch.where(i==j,matrix.new_tensor(1.),matrix.new_tensor(math.sqrt(2.)))


def new_parts(model):
    return (model.c_input_layers[0].weight,model.x_feature_extractor[0].weight)


def new_flat(values):
    return torch.cat([values[0][:,8:].flatten(),values[1][:,2:].flatten()])


class Observations(nn.Module):
    def __init__(self,model,system,correction):
        super().__init__()
        self.model=model
        self.system=system
        self.correction=correction
        self.sinv=_inverse_sqrt_overlap(system.overlap)

    def forward(self,operator=True):
        energy=DiagnosticEnergy(self.model) if isinstance(self.model,DiagnosticQModel) else LapEnergy(self.model)
        system=self.system
        e=integrated_energy(energy,system.features[:,None,:],system.weights,4096)
        e=(e+e.new_tensor(self.correction)).double()
        if not operator:
            return e.reshape(1)
        v=system.reference_operator.new_zeros(system.reference_operator.shape)
        for chunk in system.ao_chunks:
            v=v+assemble_rks_operator(energy,system.features[chunk.rows],system.weights[chunk.rows],
                                     chunk.phi,chunk.grad_phi,chunk.lap_phi,chunk_size=256,create_graph=True)
        vo=self.sinv@v@self.sinv
        return torch.cat([e.reshape(1),svec(vo)/math.sqrt(len(v))])

    def reference(self):
        s=self.system
        return torch.cat([s.exc_target.reshape(1).double(),svec(self.sinv@s.reference_operator@self.sinv)/math.sqrt(len(s.overlap))])


def load():
    spec=next(s for s in prior.STATES if s['name']=='b8_t128')
    source=prior.geometry.b8.verify_sources()
    model,shadow,_=prior.existing._load_eval_model(source,spec['checkpoint'])
    clone=DiagnosticQModel(model,'R2').eval()
    return model,shadow,clone


def pairs():
    verify()
    bundle=prior.TrainingBundle(ROOT.parent/'publication_dataset_v1')
    collection=[]
    vectors={key:[] for key in ('raw','R0','R2')}
    dispersion=prior.read(bundle.root/'mrks/dispersion.json')
    for identity,row in sorted(bundle.systems.items(),key=lambda item:item[1]['source_id']):
        group=bundle.handles.open(row['shard'])[row['group']]
        f=torch.from_numpy(group['DensityDescriptorsN10'][...]).double()
        weight=torch.from_numpy(group['weights'][...]).double()
        grad=f[:,2:8].reshape(-1,2,3)
        aa,bb=(grad[:,0]**2).sum(1),(grad[:,1]**2).sum(1)
        total=((grad[:,0]+grad[:,1])**2).sum(1)
        raw=torch.cat([f[:,:2],torch.stack([aa,total,bb],1),torch.zeros_like(f[:,:2]),f[:,8:]],1)
        mass=weight.clamp_min(0)*f[:,:2].sum(1).clamp_min(0)
        total_mass=mass.sum()
        mass=mass/total_mass
        for key, storage in vectors.items():
            if key=='raw':
                values=torch.cat([raw[:,:2].log1p(),raw[:,2:5].clamp_min(0).sqrt(),
                                  raw[:,7:9].sign()*raw[:,7:9].abs().log1p()],1)
            else:
                values=descriptors(raw,key)
                if key=='R2': values[:,[5,6,8,9]]/=math.sqrt(2)
            summary=torch.cat([(mass[:,None]*values).sum(0),(mass[:,None]*values.square()).sum(0),total_mass.reshape(1)])
            # RMS per coordinate, not unnormalized distance across dimensions.
            storage.append(summary.numpy()/math.sqrt(values.shape[1]))
        name=row['source_id']
        collection.append({'system':name,'n_ao':row['n_ao'],'n_grid':row['n_grid'],
                           'signed_electron_number':float((weight*f[:,:2].sum(1)).sum()),
                           'reference_E_electronic_Ha':float(group['Exc'][()])-float(dispersion.get(name,0)),
                           'full_raw_input_hash':__import__('hashlib').sha256(raw.numpy().tobytes()+weight.numpy().tobytes()).hexdigest()})
    result=[]
    for key,values in vectors.items():
        a=np.stack(values)
        distances=np.linalg.norm(a[:,None,:]-a[None,:,:],axis=2)
        np.fill_diagonal(distances,np.inf)
        for i,first in enumerate(collection):
            j=int(np.argmin(distances[i]))
            other=collection[j]
            result.append({'representation_summary':key,'system_i':first['system'],'system_j':other['system'],
                           'proxy_distance':float(distances[i,j]),
                           'reference_E_difference_Ha':first['reference_E_electronic_Ha']-other['reference_E_electronic_Ha'],
                           'electron_number_i':first['signed_electron_number'],'electron_number_j':other['signed_electron_number'],
                           'exact_complete_raw_measure_match':first['full_raw_input_hash']==other['full_raw_input_hash'],
                           'certified_reference_contradiction':False,
                           'reason':'weighted moments are not complete representations; no parameter-uniform response bound'})
    prior.table('lap_f017_reference_pair_analysis.csv',result)
    prior.write(OUT/'system_reference_metadata.json',collection)
    bundle.close()
    verify()


def panel():
    protocol=verify()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    bundle=prior.TrainingBundle(ROOT.parent/'publication_dataset_v1')
    mrks=bundle.mrks()
    byname={v['source_id']:k for k,v in bundle.systems.items()}
    dispersion=prior.read(bundle.root/'mrks/dispersion.json')
    model,shadow,clone=load()
    output=[]
    jacobians=[]
    residuals=[]
    checks=[]
    began=time.perf_counter()
    with restored([model,shadow,clone]):
        for index,name in enumerate(protocol['operator_panel']):
            system=mrks.operator_system(byname[name],device='cuda',dtype=torch.float32,chunk_size=4096)
            oracle=Observations(model,system,float(dispersion.get(name,0)))
            expanded=Observations(clone,system,float(dispersion.get(name,0)))
            start=time.perf_counter()
            y0=oracle()
            y=expanded()
            assert abs(float((y[:1]-y0[:1]).detach()))<=protocol['tolerances']['embedding_energy_Ha']
            torch.testing.assert_close(y[1:],y0[1:],atol=1e-10,rtol=1e-10)
            reference=expanded.reference()
            residual=reference-y
            params=tuple(clone.parameters())
            grad_e=torch.autograd.grad(y[0],params,retain_graph=True)
            old_names=tuple(sorted(dict(model.named_parameters())))
            gradients=dict(zip(dict(clone.named_parameters()),grad_e,strict=True))
            old_e=[]
            for key in old_names:
                value=gradients[key]
                if key=='c_input_layers.0.weight': value=value[:,:8]
                if key=='x_feature_extractor.0.weight': value=value[:,:2]
                old_e.append(value.flatten())
            new_e=new_flat([gradients['c_input_layers.0.weight'],gradients['x_feature_extractor.0.weight']])
            op_loss=residual[1:].square().sum()
            op_grad=torch.autograd.grad(op_loss,new_parts(clone),retain_graph=False)
            new_alignment=-.5*new_flat(op_grad).double()
            assert len(new_e)==96 and sum(len(g) for g in old_e)==9446
            jacobians.append(torch.cat([*old_e,new_e]).detach().double().cpu().numpy())
            residuals.append(float(residual[0].detach()))
            seconds=time.perf_counter()-start
            output.append({'system':name,'n_grid':len(system.features),'n_ao':len(system.overlap),
                           'E_reference_Ha':float(reference[0]),'E_predicted_with_correction_Ha':float(y[0].detach()),
                           'E_residual_Ha':residuals[-1],'operator_loss':float(op_loss.detach()),
                           'operator_residual_norm':float(residual[1:].detach().norm()),
                           'new_energy_gradient_norm':float(new_e.detach().norm()),
                           'new_operator_residual_VJP_norm':float(new_alignment.norm()),
                           'embedding_energy_gap':float((y[0]-y0[0]).detach().abs()),
                           'embedding_operator_max_gap':float((y[1:]-y0[1:]).detach().abs().max()),'seconds':seconds})
            np.savez_compressed(OUT/f'{name}_reference.npz',residual=residual.detach().cpu().numpy(),
                                new_operator_residual_VJP=new_alignment.detach().cpu().numpy())
            print('F017_PANEL',name,'seconds',seconds,flush=True)
            del y,y0,residual,params,grad_e,gradients,op_grad,oracle
            if index==0:
                # Exact full-system new-coordinate JVP; second-order reverse AD.
                zero=torch.zeros(96,device='cuda',dtype=torch.float64,requires_grad=True)
                direction=torch.linspace(-1,1,96,device='cuda',dtype=torch.float64)
                direction=direction/direction.norm()
                cw,xw=(p.detach().clone() for p in new_parts(clone))
                def functional(vector,cw=cw,xw=xw,expanded=expanded):
                    state={'model.c_input_layers.0.weight':torch.cat([cw[:,:8],vector[:64].reshape(32,2).float()],1),
                           'model.x_feature_extractor.0.weight':torch.cat([xw[:,:2],vector[64:].reshape(32,1).float()],1)}
                    return torch.func.functional_call(expanded,state,(),strict=False)
                start=time.perf_counter()
                try:
                    _,jvp=torch.autograd.functional.jvp(functional,zero,direction,create_graph=False,strict=True)
                    elapsed=time.perf_counter()-start
                    h=protocol['tolerances']['fd_parameter_step']
                    plus=functional(h*direction).detach()
                    minus=functional(-h*direction).detach()
                    finite=(plus-minus)/(2*h)
                    gap=(finite-jvp).norm()
                    relative=float(gap/jvp.norm().clamp_min(1e-30))
                    passed=bool(gap<=1e-6+.005*jvp.norm())
                    checks.append({'system':name,'method':'double-reverse full variational operator JVP;96 native-F32 additional weights',
                                   'seconds':elapsed,'FD_h':h,'relative_gap':relative,'absolute_gap':float(gap),'passed':passed,
                                   'projected_15x96_seconds':elapsed*15*96})
                except (RuntimeError,NotImplementedError) as error:
                    checks.append({'system':name,'passed':False,'error':str(error),'seconds':time.perf_counter()-start})
                prior.write(ROOT/'lap_f017_new_weight_jacobian_checks.json',checks)
            del expanded,system
            torch.cuda.empty_cache()
            if time.perf_counter()-began>protocol['resource_gate']['panel_seconds_max']:
                break
    np.savez_compressed(OUT/'energy_jacobian.npz',jacobian=np.stack(jacobians),residual=np.array(residuals))
    prior.table('lap_f017_reference_observable_metrics.csv',output)
    bundle.close()
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','panel','pairs'))
    args=parser.parse_args()
    globals()[args.stage]()
