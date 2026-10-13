"""F018: single H2/frozen-direction derivative audit, no optimizer or SCF."""
import argparse
import copy
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f017_audit as f017
import numpy as np
import torch
from lap_f015_inputs import restored
from lap_f016_descriptors import DiagnosticEnergy

from dft_functionals import PBE
from train_models.lap_operator import (
    assemble_rks_operator,
    rks_density_features_from_ao,
)
from train_models.lap_operator_data import load_central_operator_record
from train_models.lap_vxc import sigma_from_gradients, sigma_standard_to_total

ROOT=old.ROOT
OUT=ROOT.parent/'lap_f018_audit_20261013'
H=[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]
CENTRAL=ROOT.parent/'lap_operator_runs_20261001/all90/H2.h5'


def require_open_budget():
    if (OUT/'scientific_execution_closed.json').exists():
        raise TimeoutError('F018 numerical execution is closed at the resource limit; no additional model calculations')


def direction():
    # EXACT F017 CUDA/F64 construction, not a freshly sampled direction.
    v=torch.linspace(-1,1,96,device='cuda',dtype=torch.float64)
    return v/v.norm()


def freeze():
    f017.verify()
    OUT.mkdir(exist_ok=True)
    path=ROOT/'lap_f018_protocol.json'
    if path.exists():
        verify()
        return
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='072e98e6ef97f10328aa858aa068e4c938ca17ba'
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/lap_full_vxc'],cwd=ROOT,check=True)
    record=load_central_operator_record(CENTRAL)
    assert record.dmks.shape==(60,60) and len(record.weights)==42508
    row=next(r for r in old.read(ROOT/'lap_all90_operator_preparation_metrics.json')['records'] if r['system']=='H2')
    assert old.sha(CENTRAL)==row['source_sha256']
    pins={str(p):old.sha(p) for p in ROOT.glob('lap_f017_*') if p.is_file()}
    pins.update({str(p):old.sha(p) for p in (ROOT/'tools').glob('lap_f01[67]_*.py')})
    pins[str(CENTRAL)]=old.sha(CENTRAL)
    spec=next(s for s in old.STATES if s['name']=='b8_t128')
    pins[str(spec['checkpoint'])]=spec['sha256']
    v=direction().cpu().numpy()
    np.save(OUT/'frozen_F017_direction.npy',v)
    old.write(path,{'experiment':'F018','parent':head,'pins':pins,'system':'H2','n_grid':42508,'n_ao':60,
                    'checkpoint_sha256':spec['sha256'],'direction_sha256':old.sha(OUT/'frozen_F017_direction.npy'),
                    'coordinate_order':'c_input_layers.0.weight[:,8:10] row-major64; x_feature_extractor.0.weight[:,2:3] row-major32',
                    'parameter_directions':1,'adjoint_probe_seeds':[1801,1802,1803],
                    'FD_steps':H,'F017_original_h':1e-4,'F017_gate':{'absolute':1e-6,'relative':.005},
                    'tolerances':{'reproduce_scalar_abs':1e-10,'reproduce_scalar_rel':1e-6,
                                  'embedding_abs':1e-10,'embedding_rel':1e-10,
                                  'adjoint_abs':1e-12,'adjoint_rel':1e-8,
                                  'mixed_abs':1e-8,'mixed_rel':1e-5},
                    'native_repeats':3,'budget_seconds':1800,'updates':0,'training':False,
                    'counterfactual':'same widened F32 source values and same realized F32 new weights; isolated F64 model/PBE/chain arithmetic, sensitivity diagnostic not exact native oracle',
                    'mixed_method':'parameter-directional scalar energy derivative first, then anchored AO density-matrix derivative; no weak-form contraction or operator JVP called internally',
                    'density_anchor':'features(P)=stored features widened + AO feature map(P-P_source); preserves exact F017 baseline and source DM despite F32 AO caching',
                    'original_observable':'F017 concatenates energy and operator coordinates. Reproduce full vector, then report both blocks; never retrospectively replace F017 gate',
                    'no_validation_test_or_other_molecular_calculation':True})
    verify()


def verify():
    f017.verify()
    p=old.read(ROOT/'lap_f018_protocol.json')
    for name,digest in p['pins'].items():
        assert old.sha(name)==digest,name
    assert old.sha(OUT/'frozen_F017_direction.npy')==p['direction_sha256']
    return p


def difference(a,b):
    gap=(a-b).detach()
    return {'absolute_gap':float(gap.norm()),'relative_gap':float(gap.norm()/b.detach().norm().clamp_min(1e-30)),
            'cosine':float(a.detach()@b.detach()/(a.detach().norm()*b.detach().norm()).clamp_min(1e-30)),
            'max_coordinate_gap':float(gap.abs().max()),'nonfinite_count':int((~torch.isfinite(a)).sum())}


def mapping(vector,cw,xw,high=False):
    c=vector[:64].reshape(32,2).float()
    x=vector[64:].reshape(32,1).float()
    if high: c,x=c.double(),x.double()
    return {'model.c_input_layers.0.weight':torch.cat([cw[:,:8],c],1),
            'model.x_feature_extractor.0.weight':torch.cat([xw[:,:2],x],1)}


def run():
    require_open_budget()
    protocol=verify()
    began=time.perf_counter()
    def budget():
        if time.perf_counter()-began>protocol['budget_seconds']:
            raise TimeoutError('F018 30-minute numerical budget exhausted')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    bundle=old.TrainingBundle(ROOT.parent/'publication_dataset_v1')
    identity=next(k for k,v in bundle.systems.items() if v['source_id']=='H2')
    system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
    assert len(system.features)==42508 and len(system.overlap)==60
    central=load_central_operator_record(CENTRAL)
    model,shadow,clone=f017.load()
    high=copy.deepcopy(clone).double().eval()
    correction=float(old.read(bundle.root/'mrks/dispersion.json').get('H2',0))
    observations=f017.Observations(clone,system,correction)
    cw,xw=(p.detach().clone() for p in f017.new_parts(clone))
    def functional(vector):
        return torch.func.functional_call(observations,mapping(vector,cw,xw),(),strict=False)
    zero=torch.zeros(96,device='cuda',dtype=torch.float64,requires_grad=True)
    v=direction()
    assert torch.equal(v.cpu(),torch.from_numpy(np.load(OUT/'frozen_F017_direction.npy')))
    fd_rows,realized_rows,analytic_rows,adjoints=[],[],[],[]
    arrays={}
    with restored([model,shadow,clone,high]):
        baseline=functional(zero).detach()
        oracle=f017.Observations(model,system,correction)().detach()
        torch.testing.assert_close(baseline,oracle,atol=1e-10,rtol=1e-10)
        start=time.perf_counter()
        _,jvp=torch.autograd.functional.jvp(functional,zero,v,create_graph=False,strict=True)
        analytic_seconds=time.perf_counter()-start
        plus=functional(1e-4*v).detach()
        minus=functional(-1e-4*v).detach()
        original_fd=(plus-minus)/2e-4
        reproduction=difference(original_fd,jvp)
        previous=old.read(ROOT/'lap_f017_new_weight_jacobian_checks.json')[0]
        for field in ('absolute_gap','relative_gap'):
            np.testing.assert_allclose(reproduction[field],previous[field],atol=1e-10,rtol=1e-6)
        old.write(ROOT/'lap_f018_original_reproduction.json',{'full_F017_vector':reproduction,
                    'operator_only':difference(original_fd[1:],jvp[1:]),
                    'energy_coordinate_absolute_gap':float(abs(original_fd[0]-jvp[0])),
                    'full_vector_dimension':len(jvp),'operator_dimension':len(jvp)-1,
                    'direction_norm':float(v.norm()),'energy_embedding_gap':float(abs(baseline[0]-oracle[0])),
                    'operator_embedding_gap':float((baseline[1:]-oracle[1:]).abs().max()),
                    'analytical_seconds':analytic_seconds,'reproduced':True})
        print('F018_REPRODUCTION',reproduction,'OPERATOR_ONLY',difference(original_fd[1:],jvp[1:]),flush=True)
        arrays.update(baseline=baseline.cpu().numpy(),analytic_full=jvp.cpu().numpy(),original_fd=original_fd.cpu().numpy())
        # Adjoint route retapes V independently; correct transformed dual metric.
        for seed in protocol['adjoint_probe_seeds']:
            budget()
            u=torch.randn(len(jvp)-1,generator=torch.Generator().manual_seed(seed),dtype=torch.float64).cuda()
            u=u/u.norm()
            y=functional(zero)[1:]
            grad=torch.autograd.grad(y@u,zero)[0]
            direct=grad@v
            left=jvp[1:]@u
            gap=float(abs(left-direct))
            passed=gap<=1e-12+1e-8*abs(float(left))
            adjoints.append({'seed':seed,'analytic_inner_product':float(left),'independent_scalar_VJP_dot':float(direct),
                             'absolute_gap':gap,'relative_gap':gap/max(abs(float(left)),1e-30),'pass':passed})
        old.table('lap_f018_adjoint_identity_checks.csv',adjoints)
        # Direct mixed energy route, independently differentiating AO density map.
        mixed=torch.zeros((60,60),device='cuda',dtype=torch.float64)
        dm_source=torch.from_numpy(np.array(central.dmks,copy=True)).cuda()
        for chunk in system.ao_chunks:
            budget()
            dp=torch.zeros_like(dm_source,requires_grad=True)
            phi,grad_phi,lap_phi=(getattr(chunk,n).double() for n in ('phi','grad_phi','lap_phi'))
            # Delta-P map is linear. Source DM is retained; baseline remains exact.
            base=system.features[chunk.rows].double()
            features=base+rks_density_features_from_ao(phi,grad_phi,lap_phi,dp)
            rho=features[:,:2]
            grad=features[:,2:8].reshape(-1,2,3)
            sigma_model=sigma_from_gradients(grad)
            sigma_pbe=sigma_from_gradients(grad.float())
            raw=torch.cat([rho,sigma_standard_to_total(sigma_model),torch.zeros_like(rho),features[:,8:]],1)
            def scalar_energy(vector,raw=raw,rho=rho,sigma_pbe=sigma_pbe,rows=chunk.rows):
                local=mapping(vector,cw,xw)
                state={n:t.double() if t.is_floating_point() else t for n,t in clone.state_dict().items()}
                state['c_input_layers.0.weight']=local['model.c_input_layers.0.weight'].double()
                state['x_feature_extractor.0.weight']=local['model.x_feature_extractor.0.weight'].double()
                constants=torch.func.functional_call(clone,state,(raw,),strict=True)
                rp=rho.float()
                e=PBE.F_PBE(rp,sigma_pbe,constants.float(),rho.device)*rp.sum(-1)
                return (e.double()*system.weights[rows].double()).sum()
            _,parameter_derivative=torch.autograd.functional.jvp(scalar_energy,zero,v,create_graph=True,strict=True)
            mixed=mixed+torch.autograd.grad(parameter_derivative,dp)[0].detach()
        mixed=(mixed+mixed.T)/2
        mixed_observation=f017.svec(observations.sinv@mixed@observations.sinv)/math.sqrt(60)
        cmp=difference(mixed_observation,jvp[1:])
        analytic_rows.append({'method':'energy-parameter-direction first; direct AO density-matrix derivative second',
                              **cmp,'pass':cmp['absolute_gap']<=1e-8+1e-5*float(jvp[1:].norm()),
                              'shared_primitives':'R2/PBE/casts/AO input tensors; no weak-form assembly or operator JVP in mixed route'})
        old.table('lap_f018_analytic_jvp_comparison.csv',analytic_rows)
        arrays['mixed_energy_first']=mixed_observation.cpu().numpy()
        print('F018_MIXED',cmp,flush=True)
        repeats=[functional(zero).detach() for _ in range(protocol['native_repeats'])]
        repeat_gap=max(float((a-baseline).abs().max()) for a in repeats)
        # Higher-precision isolated sensitivity counterfactual.
        hc,hx=(p.detach().clone() for p in f017.new_parts(high))
        def high_operator(vector):
            state=mapping(vector,hc,hx,high=True)
            energy=DiagnosticEnergy(high)
            class Whole(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model=high
                def forward(self):
                    out=system.reference_operator.new_zeros((60,60))
                    for ch in system.ao_chunks:
                        out=out+assemble_rks_operator(energy,system.features[ch.rows].double(),system.weights[ch.rows].double(),
                            ch.phi.double(),ch.grad_phi.double(),ch.lap_phi.double(),chunk_size=256,create_graph=True)
                    return f017.svec(observations.sinv@out@observations.sinv)/math.sqrt(60)
            return torch.func.functional_call(Whole(),state,(),strict=False)
        _,high_jvp=torch.autograd.functional.jvp(high_operator,zero,v,strict=True)
        arrays['high_precision_jvp']=high_jvp.detach().cpu().numpy()
        for h in H:
            budget()
            start=time.perf_counter()
            requested=h*v
            positive=requested.float().double()
            negative=(-requested).float().double()
            effective=(positive-negative)/(2*h)
            midpoint=(positive+negative)/2
            yp=plus if h==1e-4 else functional(requested).detach()
            ym=minus if h==1e-4 else functional(-requested).detach()
            fd=(yp-ym)/(2*h)
            _,actual_prediction=torch.autograd.functional.jvp(functional,zero,effective,strict=True)
            hp=high_operator(requested).detach()
            hm=high_operator(-requested).detach()
            hfd=(hp-hm)/(2*h)
            native=difference(fd[1:],jvp[1:])
            precision=difference(hfd,high_jvp)
            fd_rows.append({'h':h,'original_F017_step':h==1e-4,'analytic_operator_norm':float(jvp[1:].norm()),
                            'FD_operator_norm':float(fd[1:].norm()),**native,
                            'full_F017_absolute_gap':float((fd-jvp).norm()),
                            'energy_coordinate_gap':float(abs(fd[0]-jvp[0])),
                            'numerator_to_baseline_operator_norm':float((yp[1:]-ym[1:]).norm()/baseline[1:].norm()),
                            'F32_parameter_JVP_correction_norm':float((actual_prediction[1:]-jvp[1:]).norm()),
                            'realized_direction_corrected_gap':float((fd[1:]-actual_prediction[1:]).norm()),
                            'F64_counterfactual_absolute_gap':precision['absolute_gap'],
                            'F64_counterfactual_relative_gap':precision['relative_gap'],
                            'native_minus_counterfactual_discrepancy_norm':float(((fd[1:]-jvp[1:])-(hfd-high_jvp)).norm()),
                            'original_gate_operator_block':native['absolute_gap']<=1e-6+.005*float(jvp[1:].norm()),
                            'seconds':time.perf_counter()-start})
            realized_rows.append({'h':h,'requested_norm':float(requested.norm()),'positive_realized_norm':float(positive.norm()),
                                  'negative_realized_norm':float(negative.norm()),'positive_rounding_norm':float((positive-requested).norm()),
                                  'negative_rounding_norm':float((negative+requested).norm()),'midpoint_shift_norm':float(midpoint.norm()),
                                  'effective_direction_difference_norm':float((effective-v).norm())})
            arrays[f'fd_{h}']=fd.cpu().numpy()
            arrays[f'high_fd_{h}']=hfd.cpu().numpy()
            old.table('lap_f018_fd_convergence.csv',fd_rows)
            old.table('lap_f018_realized_f32_perturbations.csv',realized_rows)
            print('F018_FD',h,'operator',native['relative_gap'],'energy_gap',fd_rows[-1]['energy_coordinate_gap'],flush=True)
        old.write(ROOT/'lap_f018_precision_floor.json',{'repeat_max_coordinate_gap':repeat_gap,
                  'native_repeats':3,'all_repeat_bitwise_equal':all(torch.equal(a,baseline) for a in repeats),
                  'high_precision_JVP_native_gap':difference(high_jvp,jvp[1:]),
                  'mixed_energy_first_gap':cmp,'elapsed_seconds':time.perf_counter()-began,
                  'uncertainty_status':'measured deterministic differences; no parameter-uniform rigorous rounding bound',
                  'source_dm_sha256':central.metadata['dataset_sha256']['dmks'],
                  'anchored_density_variation':True,'no_new_density_or_SCF':True})
    np.savez_compressed(OUT/'diagnostic_vectors.npz',**arrays)
    bundle.close()
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run'))
    args=parser.parse_args()
    globals()[args.stage]()
