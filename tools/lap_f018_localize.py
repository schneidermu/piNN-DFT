"""Earliest divergence on first64 original H2 grid points, F64 only."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f017_audit as f017
import lap_f018_audit as audit
import torch
from lap_f015_inputs import restored
from lap_f016_descriptors import DiagnosticEnergy

from dft_functionals import PBE
from train_models.lap_operator import (
    assemble_rks_operator,
    integrated_xc_energy,
    rks_density_features_from_ao,
)
from train_models.lap_vxc import sigma_from_gradients, sigma_standard_to_total


def main():
    audit.require_open_budget()
    audit.verify()
    torch.set_num_threads(1)
    bundle=old.TrainingBundle(old.ROOT.parent/'publication_dataset_v1')
    identity=next(k for k,v in bundle.systems.items() if v['source_id']=='H2')
    system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
    model,shadow,clone=f017.load()
    clone.double()
    cw,xw=(p.detach().clone() for p in f017.new_parts(clone))
    start=len(system.features)//2
    ch=system.ao_chunks[start//4096]
    off=start-ch.rows.start
    phi,grad,lap=(getattr(ch,k)[off:off+64].double() for k in ('phi','grad_phi','lap_phi'))
    base=system.features[start:start+64].double()
    w=system.weights[start:start+64].double()
    zero=torch.zeros(96,device='cuda',dtype=torch.float64,requires_grad=True)
    v=audit.direction()
    dp=torch.zeros((60,60),device='cuda',dtype=torch.float64,requires_grad=True)
    class Calculation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=clone
        def forward(self,route):
            energy=DiagnosticEnergy(self.model)
            if route=='weak':
                return assemble_rks_operator(energy,base,w,phi,grad,lap,chunk_size=256)
            features=base+rks_density_features_from_ao(phi,grad,lap,dp)
            value=integrated_xc_energy(energy,features,w)
            if route=='energy': return value
            return torch.autograd.grad(value,dp,create_graph=True)[0]
    calculate=Calculation()
    feat=base+rks_density_features_from_ao(phi,grad,lap,dp)
    rho=feat[:,:2]
    sigma=sigma_from_gradients(feat[:,2:8].reshape(-1,2,3))
    raw=torch.cat([rho,sigma_standard_to_total(sigma),torch.zeros_like(rho),feat[:,8:]],1)
    def custom(vector):
        state={n:t for n,t in clone.state_dict().items()}
        local=audit.mapping(vector,cw,xw,high=True)
        state['c_input_layers.0.weight']=local['model.c_input_layers.0.weight']
        state['x_feature_extractor.0.weight']=local['model.x_feature_extractor.0.weight']
        constants=torch.func.functional_call(clone,state,(raw,),strict=True)
        return (PBE.F_PBE(rho,sigma,constants,rho.device)*rho.sum(-1)*w).sum()
    def call(vector,route):
        return torch.func.functional_call(calculate,audit.mapping(vector,cw,xw,high=True),(route,),strict=False)
    with restored([model,shadow,clone]):
        direct=call(zero,'density_first')
        weak=call(zero,'weak')
        _,da=torch.autograd.functional.jvp(lambda t:call(t,'density_first'),zero,v,strict=True)
        _,wa=torch.autograd.functional.jvp(lambda t:call(t,'weak'),zero,v,strict=True)
        _,de=torch.autograd.functional.jvp(lambda t:call(t,'energy'),zero,v,create_graph=True,strict=True)
        mixed=torch.autograd.grad(de,dp)[0]
        _,customde=torch.autograd.functional.jvp(custom,zero,v,create_graph=True,strict=True)
        custommixed=torch.autograd.grad(customde,dp)[0]
        result={'points':f'64 frozenH2 points at midpoint{start}','baseline_direct_DP_vs_weak':audit.difference(direct.flatten(),weak.flatten()),
                'operator_first_direct_DP_vs_weak_JVP':audit.difference(da.flatten(),wa.flatten()),
                'energy_first_vs_direct_DP_first':audit.difference(mixed.flatten(),da.flatten()),
                'energy_first_vs_weak_JVP':audit.difference(mixed.flatten(),wa.flatten()),
                'custom_energy_first_vs_weak_JVP':audit.difference(custommixed.flatten(),wa.flatten()),
                'custom_parameter_energy_derivative':float(customde.detach()),'direct_parameter_energy_derivative':float(de.detach())}
        old.write(old.ROOT/'lap_f018_localization.json',result)
        print('F018_LOCALIZE',result,flush=True)
    bundle.close()


if __name__=='__main__':
    main()
