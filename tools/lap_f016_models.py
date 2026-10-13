"""Frozen model parity, physical input sensitivities and weak-form checks."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f016_audit as audit
import numpy as np
import torch
from lap_f015_inputs import models, restored
from lap_f016_descriptors import DiagnosticEnergy, DiagnosticQModel

from dft_functionals.constants import NN_OUTPUT_SCALE_PBE
from train_models.lap_operator import (
    assemble_rks_operator,
    integrated_xc_energy,
    rks_density_features_from_ao,
)
from train_models.lap_vxc import LapEnergy


def parity():
    audit.verify()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nets = models()
    nets = {k: nets[k] for k in old.LOCAL}
    env = np.load(audit.OUT / 'environment_0.001.npz')
    raw = torch.from_numpy(env['raw'][np.linspace(0, len(env['raw'])-1, 128, dtype=int)]).cuda()
    results, operator, stress = [], [], []
    with restored(list(nets.values())):
        for name, source in nets.items():
            for dtype in (torch.float64, torch.float32):
                original = source.to(dtype).eval()
                for rep in ('R1', 'R2'):
                    clone = DiagnosticQModel(original, rep).eval()
                    x = raw.to(dtype).detach().requires_grad_(True)
                    start = time.perf_counter()
                    a, b = original(x), clone(x)
                    ga = torch.autograd.grad(a.sum(), x)[0]
                    gb = torch.autograd.grad(b.sum(), x)[0]
                    forward_gap = float((a-b).abs().max())
                    gradient_gap = float((ga-gb).abs().max())
                    tol = 1e-12 if dtype == torch.float64 else 2e-6
                    if rep == 'R2':
                        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
                        torch.testing.assert_close(ga, gb, rtol=1e-10 if dtype == torch.float64 else tol,
                                                   atol=1e-10 if dtype == torch.float64 else tol)
                    assert torch.count_nonzero(gb[:, 5:7]) == 0
                    d = clone.get_correlation_descriptors(x)
                    ex = clone.get_exchange_descriptors(x*x.new_tensor([2,2,4,4,4,2,2,2,2]))
                    anchor_gaps = []
                    for anchor, index in ((clone.all_sigma_zero, 0), (clone.all_rho_inf, 1), (clone.all_s_inf, 28)):
                        da = anchor(d)
                        ea = anchor(ex) if index == 0 else ex
                        factor = clone.forward_descriptors(da, ea)/NN_OUTPUT_SCALE_PBE.to(x)
                        anchor_gaps.append(float((factor[:, index]-1).abs().max()))
                        assert anchor_gaps[-1] <= tol
                        assert float((factor[:, 28]-1).abs().max()) <= tol
                    swapped = x[:, [1,0,4,3,2,6,5,8,7]]
                    permutation = list(range(22))+[24,25,22,23,27,26,28]
                    torch.testing.assert_close(clone(swapped), b[:, permutation], rtol=tol, atol=tol)
                    results.append({'model': name, 'representation': rep, 'dtype': str(dtype),
                                    'forward_max_abs_gap_R0': forward_gap, 'input_gradient_max_abs_gap_R0': gradient_gap,
                                    'embedding_required': rep == 'R2', 'embedding_pass': rep == 'R2',
                                    'anchor_max_gap': max(anchor_gaps), 'tau_derivative_exact_zero': True,
                                    'parameters': sum(p.numel() for p in clone.parameters()),
                                    'seconds_forward_derivative_pair': time.perf_counter()-start})
                    # Synthetic AO check: exact stored pipeline, no new molecule/grid/SCF.
                    g = torch.Generator(device='cuda').manual_seed(1603)
                    phi = torch.randn(17,3,generator=g,device='cuda',dtype=torch.float64)*.2
                    dp = torch.randn(17,3,3,generator=g,device='cuda',dtype=torch.float64)*.1
                    lp = torch.randn(17,3,generator=g,device='cuda',dtype=torch.float64)*.1
                    w = torch.linspace(.1,.3,17,device='cuda',dtype=torch.float64)
                    dm = torch.eye(3,device='cuda',dtype=torch.float64)
                    features = rks_density_features_from_ao(phi,dp,lp,dm)
                    energy = DiagnosticEnergy(clone)
                    v = assemble_rks_operator(energy,features,w,phi,dp,lp,create_graph=False)
                    v0 = assemble_rks_operator(LapEnergy(original),features,w,phi,dp,lp,create_graph=False)
                    if rep == 'R2':
                        torch.testing.assert_close(v,v0,rtol=1e-10,atol=1e-10)
                    change = torch.tensor([[.1,.03,0],[.03,-.05,.02],[0,.02,.07]],device='cuda',dtype=torch.float64)
                    step = 1e-5 if dtype == torch.float64 else 1e-3
                    def integral(matrix, phi=phi, dp=dp, lp=lp, dtype=dtype, energy=energy, w=w):
                        f = rks_density_features_from_ao(phi,dp,lp,matrix)
                        if dtype == torch.float64:
                            return integrated_xc_energy(energy,f,w)
                        # SAME repaired energy arithmetic as operator partials.
                        from train_models.lap_operator import (
                            _model_double_operator_partials,
                        )
                        return (_model_double_operator_partials(energy,f,False)[0]*w).sum()
                    with torch.no_grad():
                        finite = (integral(dm+step*change)-integral(dm-step*change))/(2*step)
                    predicted = (v*change).sum()
                    atol, rtol = (1e-7,1e-4) if dtype == torch.float64 else (1e-5,.05)
                    passed = bool(abs(finite-predicted) <= atol+rtol*abs(predicted))
                    operator.append({'model':name,'representation':rep,'dtype':str(dtype),
                                     'operator_R0_max_abs_gap':float((v-v0).abs().max()),
                                     'deltaP_step':step,'trace_V_deltaP':float(predicted),'energy_fd':float(finite),
                                     'fd_absolute_gap':float(abs(finite-predicted)),'fd_pass':passed})
                    assert passed, operator[-1]
                source.double()
        # Nonphysical stress inputs are labeled; no new scientific systems.
        source = nets['b8_t128']
        for rep in ('R1','R2'):
            clone = DiagnosticQModel(source,rep)
            for rho in (0.,1e-12,1e-6,1e-3,1.):
                x = torch.zeros(9,9,device='cuda',dtype=torch.float64)
                x[:,:2] = rho
                q = x.new_tensor([-1e12,-100,-10,-1,0,1,10,100,1e12])
                den = 4*(3*np.pi**2)**(2/3)*(rho+1e-10)**(5/3)
                x[:,7:9] = q[:,None]*den
                x.requires_grad_(True)
                y = clone(x).sum()
                g = torch.autograd.grad(y,x,create_graph=True)[0]
                h = torch.autograd.grad(g[:,7:9].sum(),x)[0]
                assert torch.isfinite(g).all() and torch.isfinite(h).all()
                stress.append({'representation':rep,'rho_per_spin':rho,'max_abs_q':1e12,
                               'max_abs_doutput_dlap':float(g[:,7:9].abs().max()),
                               'max_abs_second_lap':float(h[:,7:9].abs().max()),'finite':True})
    old.write(old.ROOT/'lap_f016_architecture_parity.json',results)
    old.write(old.ROOT/'lap_f016_variational_operator_checks.json',operator)
    old.table('lap_f016_tail_stress.csv',stress)
    audit.verify()


if __name__ == '__main__':
    parity()
