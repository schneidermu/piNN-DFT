"""PySCF smoke and variational AO checks (run on Linux with PySCF)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lap_data import ao_reference_evaluator, evaluate_stencil
from lap_vxc import euler_components
from test_lap import small_model

from test_models.DFT.lap_functional import LapFunctional
from test_models.DFT.numint import RKS_with_Laplacian


def molecule():
    return gto.M(atom="He 0 0 0", basis="6-31g", verbose=0)


def test_lap_scf_smoke_and_variational_matrix():
    mol = molecule()
    functional = LapFunctional(small_model())
    mf = functional.make_rks(mol)
    mf.grids.level = 0
    mf.max_cycle = 15
    mf.conv_tol = 1e-8
    result = mf.kernel()
    assert np.isfinite(result)
    assert mf.converged
    dm = mf.make_rdm1()
    direction = np.array([[0.17, 0.03], [0.03, -0.08]])
    ni = mf._numint
    _, _, mat = ni.nr_rks(mol, mf.grids, mf.xc, dm)
    eps = 1e-5
    ep = ni.nr_rks(mol, mf.grids, mf.xc, dm + eps * direction)[1]
    em = ni.nr_rks(mol, mf.grids, mf.xc, dm - eps * direction)[1]
    np.testing.assert_allclose(
        (ep - em) / (2 * eps), np.sum(mat * direction), rtol=2e-5, atol=1e-7
    )
    ao = dft.numint.eval_ao(mol, mf.grids.coords[:20], deriv=2)
    rho = dft.numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=True)
    output = functional.eval_xc("", rho)
    assert np.count_nonzero(output[1][3]) == 0
    rho[5] = 1e100
    changed = functional.eval_xc("", rho)
    for a, b in zip(output[1], changed[1]):
        assert np.array_equal(a, b)


def test_existing_custom_rks_pbe_limit():
    mol = molecule()
    ordinary = dft.RKS(mol)
    custom = RKS_with_Laplacian(mol)

    def pbe_callback(code, rho, **kw):
        exc, partials, fxc, kxc = dft.libxc.eval_xc("PBE", rho[:4], **kw)
        return exc, (*partials[:2], None, np.zeros_like(partials[0])), fxc, kxc

    custom.define_xc_(pbe_callback, xctype="MGGA")
    ordinary.grids.level = 0
    ordinary.grids.build()
    dm = ordinary.get_init_guess()
    a = ordinary._numint.nr_rks(mol, ordinary.grids, "PBE", dm)
    b = custom._numint.nr_rks(mol, ordinary.grids, "", dm)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)


def test_original_ao_evaluator_and_pbe_full_potential():
    mol = molecule()
    mf = dft.RKS(mol)
    dm = mf.get_init_guess()
    xyz = torch.tensor([[0.3, 0.4, 0.5], [0.9, 0.2, 0.3]], dtype=torch.float64)
    evaluator = ao_reference_evaluator(mol, np.stack([dm / 2, dm / 2]))
    coords, f = evaluate_stencil(xyz, 0.02, evaluator, 1)
    np.testing.assert_allclose(
        f.reshape(-1, 10).numpy(), evaluator(coords.reshape(-1, 3).numpy())
    )
    # Zero NN weights recover canonical beta/gamma/mu/Gx/Gc; kappa stays bounded
    # by the historical sigmoid construction, so use exact PBE constants here.
    from torch import nn

    from dft_functionals import PBE, PBE_CONSTANTS

    class PBEReference(nn.Module):
        def forward(self, rho, sigma, lap):
            constants = PBE_CONSTANTS.to(rho).expand(len(rho), -1)
            return PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(-1)

    energy = PBEReference()
    predictions = []
    for h in (0.02, 0.01, 0.005):
        _, ff = evaluate_stencil(xyz, h, evaluator)
        predictions.append(euler_components(energy, ff, h, False)["Vxc"])
    errors = [float((predictions[i] - predictions[i + 1]).abs().max()) for i in (0, 1)]
    assert 3.8 < errors[0] / errors[1] < 4.2
