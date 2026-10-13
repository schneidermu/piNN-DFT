"""Actual trained R0 sensitivities only; no hypothetical trained R1/R2 outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f016_audit as audit
import numpy as np
import torch
from lap_f015_inputs import models, restored
from lap_f016_descriptors import raw_q

from dft_functionals import PBE
from dft_functionals.constants import NN_OUTPUT_SCALE_PBE


def main():
    audit.verify()
    torch.set_num_threads(1)
    nets = models()
    nets = {k: nets[k] for k in old.LOCAL}
    rows = []
    # Same deterministic physical samples as F015, no new reaction/grid.
    env = np.load(audit.OUT / 'environment_1e-06.npz')
    raw_np, weights = env['raw'], env['weight']
    raw = torch.from_numpy(raw_np).cuda().requires_grad_(True)
    q = raw_q(raw).detach().cpu().numpy()
    denominator = raw_np[:,7:9] / np.where(q == 0, 1, q)
    # q=0 has the same analytical denominator, never infer it by division.
    denominator = 4*(3*np.pi**2)**(2/3)*(raw_np[:,:2]+1e-10)**(5/3)
    with restored(list(nets.values())):
        for name, model in nets.items():
            chunk_receipts = []
            for start in range(0,len(raw),2048):
                x = raw[start:start+2048].detach().requires_grad_(True)
                output = model(x)
                value = output/NN_OUTPUT_SCALE_PBE.to(x)
                rs,z = PBE.rs_z_calc(x[:,:2])
                standard = torch.stack([x[:,2],(x[:,3]-x[:,2]-x[:,4])/2,x[:,4]],1)
                a,b,total = PBE.xs_xt_calc(x[:,:2],standard)
                ex = PBE.PBE_X(rs,z,total,a,b,output)*x[:,:2].sum(1)
                ec = PBE.PBE_C(rs,z,total,output,x.device)*x[:,:2].sum(1)
                fields = {n:value[:,i] for n,i in zip(('beta','gamma','kappa_up','mu_up','kappa_down','mu_down','GNN_up','GNN_down','Gc'),(0,1,22,23,24,25,26,27,28),strict=True)}
                fields.update(exchange_energy_density=ex,correlation_energy_density=ec)
                result = {}
                for field,v in fields.items():
                    derivative = torch.autograd.grad(v.sum(),x,retain_graph=True)[0]
                    assert torch.count_nonzero(derivative[:,5:7]) == 0
                    result[field] = derivative[:,7:9].detach().cpu().numpy()*denominator[start:start+len(x)]
                chunk_receipts.append(result)
            for field in chunk_receipts[0]:
                derivative = np.concatenate([c[field] for c in chunk_receipts])
                for spin in (0,1):
                    for regime,mask in (('rho_ge1e-6',np.ones(len(raw),bool)),('rho_ge1e-3',raw_np[:,:2].sum(1)>=1e-3),
                                       ('qabs_lt1',abs(q[:,spin])<1),('q95',abs(np.tanh(q[:,spin]))>.95),
                                       ('q99',abs(np.tanh(q[:,spin]))>.99),('q999',abs(np.tanh(q[:,spin]))>.999)):
                        v,w = abs(derivative[mask,spin]),weights[mask]
                        rows.append({'model':name,'channel':field,'spin':spin,'region':regime,'samples':len(v),
                                     'median_abs_d_dq':float(np.median(v)), 'p90_abs_d_dq':float(np.quantile(v,.9)),
                                     'electron_mean_abs_d_dq':float(w@v/w.sum()),'tau_derivative_exact_zero':True})
            np.savez_compressed(audit.OUT/f'trained_sensitivity_{name}.npz',**{f:np.concatenate([c[f] for c in chunk_receipts]) for f in chunk_receipts[0]})
            print('F016_SENSITIVITY',name,flush=True)
    old.table('lap_f016_trained_model_sensitivity.csv',rows)
    audit.verify()


if __name__ == '__main__':
    main()
