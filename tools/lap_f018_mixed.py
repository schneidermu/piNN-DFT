"""Stage E: independent high-precision mixed derivative on original H2 only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f017_audit as f017
import lap_f018_audit as audit
import numpy as np
import torch
from lap_f015_inputs import restored

from dft_functionals import PBE
from train_models.lap_operator import rks_density_features_from_ao
from train_models.lap_vxc import sigma_from_gradients, sigma_standard_to_total


def main(native_cast=False,native_pbe=False):
    audit.require_open_budget()
    protocol=audit.verify()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    bundle=old.TrainingBundle(old.ROOT.parent/'publication_dataset_v1')
    identity=next(k for k,v in bundle.systems.items() if v['source_id']=='H2')
    system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
    model,shadow,clone=f017.load()
    clone.double()
    cw,xw=(p.detach().clone() for p in f017.new_parts(clone))
    v=audit.direction()
    zero=torch.zeros(96,device='cuda',dtype=torch.float64,requires_grad=True)
    obs=f017.Observations(clone,system,0.)
    mixed=torch.zeros((60,60),device='cuda',dtype=torch.float64)
    with restored([model,shadow,clone]):
        for ch in system.ao_chunks:
            dp=torch.zeros((60,60),device='cuda',dtype=torch.float64,requires_grad=True)
            f=system.features[ch.rows].double()+rks_density_features_from_ao(ch.phi.double(),ch.grad_phi.double(),ch.lap_phi.double(),dp)
            rho=f[:,:2]
            sigma=sigma_from_gradients(f[:,2:8].reshape(-1,2,3))
            sigma_pbe=sigma_from_gradients(f[:,2:8].reshape(-1,2,3).float())
            raw=torch.cat([rho,sigma_standard_to_total(sigma),torch.zeros_like(rho),f[:,8:]],1)
            def scalar(vector,raw=raw,rho=rho,sigma=sigma,sigma_pbe=sigma_pbe,rows=ch.rows):
                state={n:t for n,t in clone.state_dict().items()}
                local=audit.mapping(vector,cw,xw,high=True)
                # Controlled F64 tangent arithmetic; primal zero/new source
                # weights remain exact. Avoid intermediate F32 gradient casts.
                if not native_cast:
                    local['model.c_input_layers.0.weight']=torch.cat([cw[:,:8],vector[:64].reshape(32,2)],1)
                    local['model.x_feature_extractor.0.weight']=torch.cat([xw[:,:2],vector[64:].reshape(32,1)],1)
                state['c_input_layers.0.weight']=local['model.c_input_layers.0.weight']
                state['x_feature_extractor.0.weight']=local['model.x_feature_extractor.0.weight']
                constants=torch.func.functional_call(clone,state,(raw,),strict=True)
                if native_pbe:
                    rp=rho.float()
                    return ((PBE.F_PBE(rp,sigma_pbe,constants.float(),rho.device)*rp.sum(-1)).double()*system.weights[rows].double()).sum()
                return (PBE.F_PBE(rho,sigma,constants,rho.device)*rho.sum(-1)*system.weights[rows].double()).sum()
            energy_value=scalar(zero)
            theta_gradient=torch.autograd.grad(energy_value,zero,create_graph=True)[0]
            de=theta_gradient@v
            mixed=mixed+torch.autograd.grad(de,dp)[0].detach()
    mixed=(mixed+mixed.T)/2
    ym=f017.svec(obs.sinv@mixed@obs.sinv)/np.sqrt(60)
    with np.load(audit.OUT/'diagnostic_vectors.npz') as vectors:
        jhigh=torch.from_numpy(vectors['high_precision_jvp']).cuda()
        jnative=torch.from_numpy(vectors['analytic_full'][1:]).cuda()
        jmixed=torch.from_numpy(vectors['mixed_energy_first']).cuda()
    eig=torch.linalg.eigvalsh(system.overlap)
    result={'high_energy_first_vs_high_operator_first':audit.difference(ym,jhigh),
            'high_energy_first_vs_native_operator_first':audit.difference(ym,jnative),
            'native_energy_first_vs_high_energy_first':audit.difference(jmixed,ym),
            'overlap_eigen_min':float(eig.min()),'overlap_eigen_max':float(eig.max()),
            'overlap_condition':float(eig.max()/eig.min()),
            'pass':float((ym-jhigh).norm())<=protocol['tolerances']['mixed_abs']+protocol['tolerances']['mixed_rel']*float(jhigh.norm()),
            'native_parameter_cast':native_cast,
            'native_PBE_arithmetic':native_pbe,
            'route':'explicit parameter gradient dot original v, then AO density-matrix gradient; no weak-form assembly, no double-reverse JVP; F64 PBE sensitivity diagnostic'}
    filename='lap_f018_native_mixed_explicit.json' if native_pbe else ('lap_f018_native_cast_mixed.json' if native_cast else 'lap_f018_high_precision_mixed.json')
    old.write(old.ROOT/filename,result)
    np.save(audit.OUT/'high_energy_first_mixed_matrix.npy',mixed.cpu().numpy())
    print('F018_HIGH_MIXED',result,flush=True)
    bundle.close()
    audit.verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-cast',action='store_true')
    parser.add_argument('--native-pbe',action='store_true')
    args=parser.parse_args()
    main(args.native_cast,args.native_pbe)
