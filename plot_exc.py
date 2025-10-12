from io import StringIO

import numpy as np
import plotly.graph_objects as go
import pylibxc as xc
import torch

import DFT as DFT
from pcNN_mol.dft_pcnn import model as Nagai_model

buf = StringIO()

ldax = xc.LibXCFunctional("lda_x", "unpolarized")
df = {
    "NN_PBE": dict(),
    "NN_PBE*": dict(),
    "NN_PBE_star_star": dict(),
    "NN_XALPHA": dict(),
    "PBE": dict(),
    "Nagai": dict(),
}

func_dict = {
    "NN_PBE": DFT.NN_FUNCTIONAL("NN_PBE_067"),
    "NN_PBE*": DFT.NN_FUNCTIONAL("NN_PBE_star"),
    "NN_PBE_star_star": DFT.NN_FUNCTIONAL("NN_PBE_star_star_18"),
    "NN_XALPHA": DFT.NN_FUNCTIONAL("NN_XALPHA_067"),
    "Nagai": "Nagai",
}


def get_tr(rs, s, alpha, func, color, dash):
    funcx = xc.LibXCFunctional(func[0], "unpolarized")
    funcc = xc.LibXCFunctional(func[1], "unpolarized")

    rho = 3 / (rs**3 * 4 * np.pi)

    sa = np.linspace(0, s, 15000)

    sigma = (sa * (2 * ((3 * (np.pi**2)) ** (1 / 3)) * (rho ** (4 / 3)))) ** 2
    tau = (
        alpha / (5.0 / 9.0 * 6.0 ** (1 / 3) * (1.0 / (np.pi ** (4 / 3))))
        + ((sigma) * 2.0 ** (2 / 3) * (1.0 / (rho ** (8 / 3)))) / 8.0
    ) / (1.0 / ((rho ** (5 / 3))) * (2.0 ** (2 / 3)))

    inp = {}
    inp["rho"] = np.array([rho] * 15000)
    inp["sigma"] = sigma
    inp["tau"] = tau

    retx = funcx.compute(inp)
    retc = funcc.compute(inp)

    retlx = ldax.compute(inp)

    y = retx["zk"].flatten() + retc["zk"].flatten()
    y /= retlx["zk"].flatten()

    return (
        go.Scatter(
            x=sa,
            y=y,
            mode="lines",
            name=func[0] + " alpha=" + str(alpha),
            line=dict(color=color, dash=dash),
        ),
        sa,
        y,
    )


def get_tr_NN(rs, s, alpha, func, color, dash):

    functional = func_dict[func]

    rho = 3 / (rs**3 * 4 * np.pi)

    sa = np.linspace(0, s, 15000)

    sigma = (sa * (2 * ((3 * (np.pi**2)) ** (1 / 3)) * (rho ** (4 / 3)))) ** 2
    tau = (
        alpha / (5.0 / 9.0 * 6.0 ** (1 / 3) * (1.0 / (np.pi ** (4 / 3))))
        + ((sigma) * 2.0 ** (2 / 3) * (1.0 / (rho ** (8 / 3)))) / 8.0
    ) / (1.0 / ((rho ** (5 / 3))) * (2.0 ** (2 / 3)))

    inp = {}
    inp["rho"] = np.array([rho] * 15000)
    inp["sigma"] = sigma
    inp["tau"] = tau

    if func == "Nagai":
        y = Nagai_model.return_y([inp["rho"], np.sqrt(sigma/3), np.sqrt(sigma/3), np.sqrt(sigma/3), tau])
    else:
        y = (
            functional(features=inp, device=torch.device("cpu"), mode="Enhancement")[1]
            .detach()
            .numpy()
        )

    retlx = ldax.compute(inp)

    y /= retlx["zk"].flatten()

    return (
        go.Scatter(
            x=sa,
            y=y,
            mode="lines",
            name=func + " alpha=" + str(alpha),
            line=dict(color=color, dash=dash),
        ),
        sa,
        y,
    )


fig = go.Figure()

max_s = 5

res_pbe = get_tr(1, max_s, 0, ["gga_x_pbe", "gga_c_pbe"], "black", None)
df["PBE"].update({"0": res_pbe[2]})
fig.add_trace(res_pbe[0])

colors = ["red", "orange", "black", "green", "black"]
funcs = [
    "NN_PBE",
    "NN_PBE*",
    "NN_PBE_star_star",
    "NN_XALPHA",
    "Nagai",
]

for n, i in enumerate(funcs):
    res_0 = get_tr_NN(1, max_s, 0, i, colors[n], "dash")
    res_1 = get_tr_NN(1, max_s, 1, i, colors[n], "dot")
    res_100 = get_tr_NN(1, max_s, 100, i, colors[n], None)
    res_inf = get_tr_NN(1, max_s, 1e9, i, colors[n], None)
    df[i].update({"0": res_0[2], "1": res_1[2], "100": res_100[2], "inf": res_inf[2]})
    fig.add_trace(res_0[0])
    fig.add_trace(res_1[0])
    fig.add_trace(res_100[0])
    fig.add_trace(res_inf[0])


fig.update_layout(template="plotly_white", title="")

fig.update_layout(
    font=dict(family="Courier New, monospace", size=20, color="#7f7f7f"),
)

tuples = [
    ("NN_PBE", "0"),
    ("NN_PBE", "1"),
    ("NN_PBE", "100"),
    ("NN_PBE", "inf"),
    ("NN_PBE*", "0"),
    ("NN_PBE*", "1"),
    ("NN_PBE*", "100"),
    ("NN_PBE*", "inf"),
    ("NN_PBE_star_star", "0"), 
    ("NN_PBE_star_star", "1"), 
    ("NN_PBE_star_star", "100"),
    ("NN_PBE_star_star", "inf"),
    ("NN_XALPHA", "0"),
    ("NN_XALPHA", "1"),
    ("NN_XALPHA", "100"),
    ("NN_XALPHA", "inf"),
    ("Nagai", "0"),
    ("Nagai", "1"),
    ("Nagai", "100"),
    ("Nagai", "inf"),
    ("PBE", "0"),
]

data = np.array([df[x][y] for x, y in tuples])

np.save("Results/exc.npy", data)
