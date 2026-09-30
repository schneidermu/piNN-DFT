"""Run a tiny real H2 RKS/Laplacian SCF smoke from a pilot checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from build_mrks_stencils import _make_mol
from lap_checkpoint import load_lap_checkpoint

from test_models.DFT.lap_functional import LapFunctional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--max-cycle", type=int, default=40)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    args = parser.parse_args()

    from pyscf import dft

    with np.load(args.npz, allow_pickle=False) as npz:
        mol, _, _, spin = _make_mol(npz)
        if spin != 0:
            raise ValueError("The real SCF smoke currently expects a closed-shell molecule.")
    model, payload = load_lap_checkpoint(args.checkpoint, device="cpu", dtype=torch.float64)
    functional = LapFunctional(model)
    mf = functional.make_rks(mol)
    mf.grids.level = args.grid_level
    mf.max_cycle = args.max_cycle
    mf.conv_tol = args.conv_tol
    mf.verbose = 4
    trace = []

    def capture_cycle(envs):
        row = {key: envs[key] for key in ("cycle", "e_tot", "norm_gorb", "norm_ddm") if key in envs}
        trace.append({key: float(value) for key, value in row.items()})

    mf.callback = capture_cycle
    energy = float(mf.kernel())
    if not math.isfinite(energy):
        raise FloatingPointError("RKS Lap SCF returned a nonfinite total energy.")

    mf.grids.build(with_non0tab=True)
    dm = mf.get_init_guess()
    ao = dft.numint.eval_ao(mol, mf.grids.coords[:32], deriv=2)
    rho = dft.numint.eval_rho(mol, ao, dm, xctype="MGGA", hermi=1, with_lapl=True)
    before = functional.eval_xc("", rho, spin=0, deriv=1)
    shifted = np.array(rho, copy=True)
    shifted[5] += 1e6
    after = functional.eval_xc("", shifted, spin=0, deriv=1)
    vtau = np.asarray(before[1][3])
    tau_independent = all(np.array_equal(a, b) for a, b in zip(before[1], after[1]))
    if np.any(vtau != 0) or not tau_independent:
        raise AssertionError("LapFunctional SCF adapter depends on tau or returned nonzero vtau.")

    result = {
        "system": "H2",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_protocol": payload["protocol"],
        "grid_level": args.grid_level,
        "max_cycle": args.max_cycle,
        "conv_tol": args.conv_tol,
        "converged": bool(mf.converged),
        "total_energy_hartree": energy,
        "cycles_recorded": len(trace),
        "convergence_trace": trace,
        "vtau_max_abs": float(np.max(np.abs(vtau), initial=0.0)),
        "tau_independent": tau_independent,
        "all_cycle_energies_finite": all(math.isfinite(row["e_tot"]) for row in trace if "e_tot" in row),
        "all_output_finite": all(np.isfinite(np.asarray(x)).all() for x in (before[0], *before[1][:3])),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
