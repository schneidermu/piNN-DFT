"""Independent component checks for the two large MCONF Diet records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from pyscf import dft, lib
    from pyscf.scf import chkfile
    lib.num_threads(4)

    results = []
    for name in ("MCONF-1-1", "MCONF-1-2"):
        chk = args.raw / "chk" / f"{name}.pbe0.chk"
        row_path = args.staging / "validation" / "diet30_species.jsonl"
        # The species index is only emitted after the aggregate builder phase;
        # use the verified worker record while the build is still staged.
        if not row_path.exists():
            candidates = sorted((args.staging.parent / "publication_dataset_v1.validation_work").glob("job_*/validation/diet30_species.jsonl"))
            row = next(item for p in candidates for item in
                       (json.loads(line) for line in p.read_text().splitlines())
                       if item["source_id"] == name)
        else:
            row = next(json.loads(line) for line in row_path.read_text().splitlines() if json.loads(line)["source_id"] == name)
        shard = args.staging / row["shard"] if row["shard"].startswith("validation/") else args.staging.parent / "publication_dataset_v1.validation_work" / row["shard"]
        if not shard.exists():
            # Worker rows use a path relative to their job directory.
            candidates = sorted((args.staging.parent / "publication_dataset_v1.validation_work").glob("job_*/validation/*.h5"))
            shard = next(p for p in candidates if sha256(p) == row["shard_sha256"])

        mol = chkfile.load_mol(str(chk))
        if not shard.exists() or not h5py.is_hdf5(shard):
            shard = None
        if shard is not None:
            with h5py.File(shard, "r") as probe:
                if row["group"] not in probe:
                    shard = None
        if shard is None:
            candidates = sorted((args.staging.parent / "publication_dataset_v1.validation_work").glob("job_*/validation/*.h5"))
            shard = next(p for p in candidates if row["group"] in h5py.File(p, "r"))

        with h5py.File(chk, "r") as source, h5py.File(shard, "r") as handle:
            dm_source = source["scf/dm"][...]
            group = handle[row["group"]]
            dm_stored = group["dm"][...]
            if dm_source.dtype != np.float64 or dm_stored.dtype != np.float64:
                raise ValueError(f"{name}: density is not native F64")
            density_diff = float(np.max(np.abs(dm_source - dm_stored)))
            coords = group["coords64"][...]
            weights = group["weights"][...]
            features = group["features"][...]
            eps_stored = group["pbe_epsilon"][...]
            nonxc_stored = float(group["nonxc"][()])

        mf = dft.UKS(mol) if dm_source.ndim == 3 else dft.RKS(mol)
        mf.xc = "PBE"
        mf.grids.coords = coords
        mf.grids.weights = weights
        mf.grids.non0tab = None
        spin_dm = dm_source if dm_source.ndim == 3 else np.stack((dm_source * 0.5, dm_source * 0.5))
        direct_features = np.empty_like(features, dtype=np.float64)
        eps_direct = np.empty_like(eps_stored, dtype=np.float64)
        for start in range(0, len(coords), 4096):
            end = min(start + 4096, len(coords))
            ao = dft.numint.eval_ao(mol, coords[start:end], deriv=2)
            rho = [dft.numint.eval_rho(mol, ao, matrix, xctype="MGGA", with_lapl=True) for matrix in spin_dm]
            direct_features[start:end, :2] = np.stack((rho[0][0], rho[1][0]), axis=1)
            direct_features[start:end, 2:5] = rho[0][1:4].T
            direct_features[start:end, 5:8] = rho[1][1:4].T
            direct_features[start:end, 8:10] = np.stack((rho[0][4], rho[1][4]), axis=1)
            if dm_source.ndim == 3:
                eps_direct[start:end] = mf._numint.eval_xc("PBE", np.array(rho)[:, :4], spin=1, deriv=0)[0]
            else:
                eps_direct[start:end] = mf._numint.eval_xc("PBE", (rho[0] + rho[1])[:4], spin=0, deriv=0)[0]
        descriptor_diff = float(np.max(np.abs(direct_features - features)))
        xc_diff = float(abs(np.dot(eps_direct * direct_features[:, :2].sum(axis=1), weights) -
                            np.dot(eps_stored * features[:, :2].sum(axis=1), weights)))
        hcore = mf.get_hcore()
        total = dm_source.sum(axis=0) if dm_source.ndim == 3 else dm_source
        one_nuc = float(np.einsum("ij,ji", total, hcore) + mol.energy_nuc())
        # This is an independent scalar Hartree evaluation: get_j is invoked
        # without constructing a V_xc/veff matrix, then immediately contracted
        # with the source density in F64.
        coulomb = mf.get_j(mol, total)
        hartree = float(0.5 * np.einsum("ij,ji", total, coulomb))
        component_sum = one_nuc + hartree
        nonxc_diff = abs(component_sum - nonxc_stored)
        recombined = component_sum + float(np.dot(eps_direct * direct_features[:, :2].sum(axis=1), weights))
        stored_total = nonxc_stored + float(np.dot(eps_stored * features[:, :2].sum(axis=1), weights))
        results.append({"source_id": name, "checkpoint_sha256": sha256(chk),
                        "stored_shard_sha256": sha256(shard), "density_max_abs_diff": density_diff,
                        "descriptor_max_abs_diff": descriptor_diff, "xc_energy_abs_diff": xc_diff,
                        "one_electron_nuclear": one_nuc, "hartree": hartree,
                        "stored_nonxc": nonxc_stored, "nonxc_component_abs_diff": nonxc_diff,
                        "recombined_fixed_density_energy": recombined,
                        "stored_fixed_density_energy": stored_total,
                        "recombined_abs_diff": abs(recombined - stored_total),
                        "status": "PASS" if max(density_diff, descriptor_diff, xc_diff, nonxc_diff,
                                                abs(recombined - stored_total)) <= 1e-8 else "FAIL"})
    if any(row["status"] != "PASS" for row in results):
        raise SystemExit(json.dumps({"status": "FAIL", "systems": results}, indent=2))
    args.output.write_text(json.dumps({"status": "PASS", "method": "F64 component validation; no dense veff",
                                       "systems": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "systems": results}, indent=2))


if __name__ == "__main__":
    main()
