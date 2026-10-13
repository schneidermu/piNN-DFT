"""Continuous promoted tangent coordinates; identical native primal operator."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f017_audit as f017
import lap_f018_audit as audit
import numpy as np
import torch
from lap_f015_inputs import restored
from lap_f016_descriptors import DiagnosticEnergy

from train_models.lap_operator import (
    _assemble_from_local_partials,
    _model_double_operator_partials,
)


def main():
    audit.require_open_budget()
    audit.verify()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    bundle=old.TrainingBundle(old.ROOT.parent/'publication_dataset_v1')
    identity=next(k for k,v in bundle.systems.items() if v['source_id']=='H2')
    system=bundle.mrks().operator_system(identity,device='cuda',dtype=torch.float32,chunk_size=4096)
    model,shadow,clone=f017.load()
    cw,xw=(p.detach().double().clone() for p in f017.new_parts(clone))
    sinv=f017.Observations(clone,system,0).sinv
    class Promoted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=clone
        def forward(self):
            energy=DiagnosticEnergy(self.model)
            matrix=system.reference_operator.new_zeros((60,60))
            for ch in system.ao_chunks:
                _,c,a,b=_model_double_operator_partials(energy,system.features[ch.rows].double(),True)
                factors=[system.weights[ch.rows].double(),ch.phi.double(),ch.grad_phi.double(),ch.lap_phi.double()]
                matrix=matrix+sum(_assemble_from_local_partials(c[:,spin],a[:,spin],b[:,spin],*factors,256) for spin in (0,1))/2
            return f017.svec(sinv@matrix@sinv)/np.sqrt(60)
    promoted=Promoted()
    native_observer=f017.Observations(clone,system,0.)
    def native_functional(z):
        return torch.func.functional_call(native_observer,audit.mapping(z,cw.float(),xw.float()),(),strict=False)[1:]
    def functional(z):
        state={'model.c_input_layers.0.weight':torch.cat([cw[:,:8],z[:64].reshape(32,2)],1),
               'model.x_feature_extractor.0.weight':torch.cat([xw[:,:2],z[64:].reshape(32,1)],1)}
        return torch.func.functional_call(promoted,state,(),strict=False)
    zero=torch.zeros(96,device='cuda',dtype=torch.float64,requires_grad=True)
    # The SAME frozen tangent after the original native F32 cast; no new probe.
    v=audit.direction().float().double()
    with np.load(audit.OUT/'diagnostic_vectors.npz') as cache:
        original=torch.from_numpy(cache['analytic_full'][1:]).cuda()
        baseline=torch.from_numpy(cache['baseline'][1:]).cuda()
    result=[]
    with restored([model,shadow,clone]):
        y=functional(zero).detach()
        torch.testing.assert_close(y,baseline,atol=1e-10,rtol=1e-10)
        _,jvp=torch.autograd.functional.jvp(functional,zero,v,strict=True)
        for seed in (1801,1802,1803):
            u=torch.randn(len(jvp),generator=torch.Generator().manual_seed(seed),dtype=torch.float64).cuda()
            u=u/u.norm()
            gradient=torch.autograd.grad(functional(zero)@u,zero)[0]
            native_gradient=torch.autograd.grad(native_functional(zero)@u,zero)[0]
            lhs=jvp@u
            rhs=gradient@v
            gap=float(abs(lhs-rhs))
            requested=audit.direction()
            native_right=native_gradient@requested
            posterior=gap+float((native_gradient-gradient).abs()@requested.abs()+gradient.abs()@(requested-v).abs())
            result.append({'seed':seed,'left':float(lhs),'right':float(rhs),'absolute_gap':gap,
                           'relative_gap':gap/max(abs(float(lhs)),1e-30),'pass':gap<=1e-12+1e-8*abs(float(lhs)),
                           'native_absolute_gap':float(abs(lhs-native_right)),
                           'native_vs_promoted_pullback_triangle_bound':posterior,
                           'native_gap_within_measured_bound':float(abs(lhs-native_right))<=posterior+1e-15,
                           'parameter_cotangent_rounding_single_cast_bound':float((gradient.float().double()-gradient).abs()@v.abs())})
    old.write(old.ROOT/'lap_f018_promoted_adjoint.json',{'primal_native_parity_gap':float((y-baseline).abs().max()),
                'analytical_native_JVP_gap':audit.difference(jvp,original),'checks':result,
                'interpretation':'Same F32 primal checkpoint/source values and F32 PBE; promoted F64 parameter-tangent coordinates, not production replacement. Native adjoint FAIL flags remain unchanged.'})
    print('F018_PROMOTED_ADJOINT',result,'JVP_NATIVE_GAP',audit.difference(jvp,original),flush=True)
    bundle.close()


if __name__=='__main__':
    main()
