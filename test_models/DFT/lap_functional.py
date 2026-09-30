"""Variational RKS adapter for the tau-free energy; no spatial FD in SCF."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train_models"))
from lap_checkpoint import load_lap_checkpoint
from lap_vxc import LapEnergy, sigma_from_gradients


class LapFunctional:
    def __init__(self, model):
        self.model = model.eval()
        self.energy = LapEnergy(model)

    @classmethod
    def from_checkpoint(cls, path):
        model, _ = load_lap_checkpoint(path)
        return cls(model)

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, **kwargs):
        if spin != 0 or deriv != 1:
            raise ValueError(
                "Lap SCF adapter currently supports RKS first derivatives only."
            )
        r = np.asarray(rho)
        if r.ndim != 2 or r.shape[0] != 6:
            raise ValueError(
                "Use RKS_with_Laplacian / MGGA to supply density gradients and Laplacian."
            )
        p = next(self.model.parameters())
        # PySCF RKS input rows: total rho, dx,dy,dz,lapl,tau.
        f = np.column_stack(
            [r[0] / 2, r[0] / 2, r[1:4].T / 2, r[1:4].T / 2, r[4] / 2, r[4] / 2]
        )
        t = torch.as_tensor(f, dtype=p.dtype, device=p.device)
        with torch.enable_grad():
            dens = t[:, :2].detach().requires_grad_(True)
            sigma = (
                sigma_from_gradients(t[:, 2:8].reshape(-1, 2, 3))
                .detach()
                .requires_grad_(True)
            )
            lap = t[:, 8:].detach().requires_grad_(True)
            e = self.energy(dens, sigma, lap)
            c, s, b = torch.autograd.grad(
                e.sum(), (dens, sigma, lap), allow_unused=True
            )
            b = torch.zeros_like(lap) if b is None else b
        exc = e.detach().cpu().numpy() / np.maximum(r[0], np.finfo(float).tiny)
        vrho = c.mean(1).detach().cpu().numpy()
        vsigma = (s.sum(1) / 4).detach().cpu().numpy()
        vlapl = b.mean(1).detach().cpu().numpy()
        vtau = np.zeros_like(vrho)  # no computational tau dependency
        if not all(np.isfinite(x).all() for x in (exc, vrho, vsigma, vlapl)):
            raise FloatingPointError("Nonfinite Lap SCF energy/derivatives.")
        return exc, (vrho, vsigma, vlapl, vtau), None, None

    def make_rks(self, mol):
        from .numint import RKS_with_Laplacian

        mf = RKS_with_Laplacian(mol)
        mf.define_xc_(self.eval_xc, xctype="MGGA")
        return mf
