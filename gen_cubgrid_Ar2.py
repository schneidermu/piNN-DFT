import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pylibxc as xc
import torch
from pyscf import dft, gto, lib

import density_functional_approximation_dm21 as dm21

func_dict = {
    "NN_PBE": dm21.NN_FUNCTIONAL("NN_PBE_18"),
    "NN_PBE*": dm21.NN_FUNCTIONAL("NN_PBE_star"),
    "NN_XALPHA": dm21.NN_FUNCTIONAL("NN_XALPHA_99"),
}

ldax = xc.LibXCFunctional("lda_x", "unpolarized")


max_memory = 8000
lib.num_threads(6)

ni = dft.numint.NumInt()

mol = gto.Mole()
mol.verbose = 4
mol.atom = """
Ar  0.0  0.0  1.8785
Ar  0.0  0.0 -1.8785
"""
mol.spin = 0
mol.charge = 0
mol.basis = "def2-QZVPPD"
mol.build()

mf = dft.RKS(mol)
mf.xc = "PBE0"
mf.grids.level = 6
mf.run()

spacex = np.linspace(0, 0, int(1))
spacey = np.linspace(0, 0, int(1))
spacez = np.linspace(-7, 7, int(50000))


coords = []
for k in spacex:
    for j in spacey:
        for l in spacez:
            coords.append([k, j, l])
coords = np.array(coords)

ao = ni.eval_ao(mol, coords, deriv=2)

dm = mf.make_rdm1()

rho = ni.eval_rho(mol, ao, dm, xctype="mGGA")

pdat = {}

sigma = rho[1] ** 2 + rho[2] ** 2 + rho[3] ** 2

alpha = (
    5.0
    / 9.0
    * (
        (rho[5] * 2.0 ** (2 / 3)) * (0.1e1 / (rho[0] ** (5 / 3)))
        - (sigma * 2.0 ** (2 / 3) * (1.0 / (rho[0] ** (8 / 3)))) / 8.0
    )
    * 6.0 ** (1 / 3)
    * (1.0 / (np.pi ** (4 / 3)))
)
pdat["alpha"] = alpha

inp = {}
inp["rho"] = rho[0]
inp["sigma"] = sigma
inp["tau"] = rho[5]

retlx = ldax.compute(inp)
ex_lda = retlx["zk"].flatten()

funcs = [
    ["gga_x_pbe", "gga_c_pbe"],
]

df = dict()

for i, func in enumerate(funcs):
    funcx = xc.LibXCFunctional(func[0], "unpolarized")
    funcc = xc.LibXCFunctional(func[1], "unpolarized")

    retx = funcx.compute(inp)
    retc = funcc.compute(inp)
    ex = retx["zk"].flatten()
    ec = retc["zk"].flatten()

    fxc = (ex + ec) / ex_lda

    pdat["PBE Fxc"] = fxc
    df["PBE"] = fxc

nn_funcs = [
    "NN_PBE",
    "NN_PBE*",
    "NN_XALPHA"
]

for i, func in enumerate(nn_funcs):
    functional = func_dict[func]

    exc = (
        functional(features=inp, device=torch.device("cpu"), mode="Enhancement")[1]
        .detach()
        .numpy()
    )

    fxc = exc / ex_lda

    pdat[func + " Fxc"] = fxc
    df[func] = fxc

df["x"] = spacez


def plot_and_save():
    fig = go.Figure()

    for d in pdat:
        fig.add_trace(go.Scatter(x=spacez, y=pdat[d], mode="lines", name=d))

    fig.update_layout(template="plotly_white", title="")

    fig.update_layout(
        font=dict(family="Courier New, monospace", size=20, color="#7f7f7f"),
    )

    plotly.offline.plot(fig, filename="Ar2.html")


df = pd.DataFrame(df)
df.to_csv("Results/Ar2.csv")
