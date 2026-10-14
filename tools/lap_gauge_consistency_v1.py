"""Read-only fixed-density gauge diagnostics; no neural-model dependencies."""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "lap_gauge_consistency_v1_"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name, value):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


def write_csv(name, rows):
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(F, S, C, energy, occupation, Q):
    """All residuals are orthonormal-AO vectors, including outside MO span."""
    eigen, vectors = eigh(S)
    assert eigen.min() > np.finfo(float).eps * len(S) * eigen.max()
    X = (vectors / np.sqrt(eigen)) @ vectors.T
    half = (vectors * np.sqrt(eigen)) @ vectors.T
    R = X @ (F @ C - (S @ C) * energy)
    norms = np.linalg.norm(R, axis=0)
    occupied = occupation > 0
    homo = np.flatnonzero(occupied)[np.argmax(energy[occupied])]
    A = C.T @ F @ C - np.diag(energy)
    U = half @ C
    outside = R - U @ np.linalg.solve(U.T @ U, U.T @ R)
    denom = np.linalg.norm(X @ F @ C) + np.linalg.norm((X @ S @ C) * energy)
    row = {
        "raw_fro_ha": float(np.linalg.norm(R)),
        "raw_rms_per_mo_ha": float(np.linalg.norm(R) / np.sqrt(C.shape[1])),
        "raw_max_orbital_ha": float(norms.max()),
        "homo_residual_ha": float(norms[homo]),
        "occupied_fro_ha": float(np.linalg.norm(R[:, occupied])),
        "virtual_fro_ha": float(np.linalg.norm(R[:, ~occupied])),
        "relative_residual": float(np.linalg.norm(R) / denom),
        "relative_denominator_ha": float(denom),
        "mo_orthogonality_fro": float(np.linalg.norm(C.T @ S @ C - np.eye(C.shape[1]))),
        "outside_mo_span_fro_ha": float(np.linalg.norm(outside)),
        "n_mo": C.shape[1],
        "n_ao": len(S),
        "complete_mo_space": C.shape[1] == len(S),
        "overlap_min": float(eigen.min()),
        "overlap_condition": float(eigen.max() / eigen.min()),
        "diagonal_offsets_ha": json.dumps(np.diag(A).tolist()),
        "offdiagonal_fro_ha": float(np.linalg.norm(A - np.diag(np.diag(A)))),
        "orbital_residuals_ha": json.dumps(norms.tolist()),
        "homo_rayleigh_ha": float(
            C[:, homo] @ F @ C[:, homo] / (C[:, homo] @ S @ C[:, homo])
        ),
        "Q_minus_S_fro": float(np.linalg.norm(Q - S)),
        "B_minus_I_spectral": float(np.linalg.norm(X @ Q @ X - np.eye(len(S)), 2)),
    }
    fits = []
    for kind, matrix in [("Q", Q), ("S", S)]:
        D = X @ matrix @ C
        for scope, mask in [
            ("full", np.ones(C.shape[1], dtype=bool)),
            ("occupied", occupied),
        ]:
            c = -float(
                np.vdot(D[:, mask], R[:, mask]).real / np.linalg.norm(D[:, mask]) ** 2
            )
            adjusted = R + c * D
            bound = (
                64
                * np.finfo(float).eps
                * len(S)
                * (eigen.max() / eigen.min())
                * max(1, denom)
                / np.linalg.norm(D[:, mask])
            )
            fits.append(
                {
                    "shift_basis": kind,
                    "fit_space": scope,
                    "c_add_ha": c,
                    "arithmetic_bound_ha": float(bound),
                    "adjusted_full_fro_ha": float(np.linalg.norm(adjusted)),
                    "adjusted_max_orbital_ha": float(
                        np.linalg.norm(adjusted, axis=0).max()
                    ),
                    "adjusted_occupied_fro_ha": float(
                        np.linalg.norm(adjusted[:, occupied])
                    ),
                    "absolute_fro_reduction_ha": float(
                        np.linalg.norm(R) - np.linalg.norm(adjusted)
                    ),
                }
            )
    return row, fits, R


def project(phi_chunks):
    V = Q = None
    partials = []
    for phi, weights, potential in phi_chunks:
        phi = np.asarray(phi, dtype=np.float64)
        if V is None:
            V = np.zeros((phi.shape[1], phi.shape[1]))
            Q = V.copy()
        pv = phi.T @ (phi * (weights * potential)[:, None])
        pq = phi.T @ (phi * weights[:, None])
        V += pv
        Q += pq
        partials.append((pv, pq))
    reverse_v = np.zeros_like(V)
    reverse_q = np.zeros_like(Q)
    for pv, pq in reversed(partials):
        reverse_v += pv
        reverse_q += pq
    return V, Q, reverse_v, reverse_q


def main():
    from pyscf import gto, scf
    from pyscf.dft import numint

    protocol = json.loads((ROOT / (PREFIX + "protocol.json")).read_text())
    begin = protocol["first_action_unix"]
    active = time.monotonic()

    def budget():
        if time.time() - begin > protocol["budget_seconds"] - 120:
            raise TimeoutError("Reserved integrity/report budget reached")

    for path, expected in protocol["baseline_receipts"].items():
        assert digest(ROOT / path) == expected, path
    sources = json.loads(
        (ROOT / "lap_operator_contract_v1_provenance.json").read_text()
    )["npz_sources"]
    publication = ROOT.parent / "publication_dataset_v1"
    manifest = [
        json.loads(line)
        for line in (publication / "mrks/systems.jsonl").read_text().splitlines()
    ]
    mapping = {row["source_id"]: row for row in manifest}
    conversion = json.loads(
        (publication / "provenance/mrks_conversion.json").read_text()
    )
    hashes = {row["file"]: row["sha256"] for row in conversion["files"]}
    metadata_rows = []
    provenance = {
        "driver_sha256": digest(__file__),
        "protocol_sha256": digest(ROOT / (PREFIX + "protocol.json")),
        "systems_manifest_sha256": digest(publication / "mrks/systems.jsonl"),
        "sources": {},
        "pilot_corpus": {},
        "no_checkpoint_optimizer_or_SCF": True,
        "grid_arithmetic": "F64 accumulation; cached CORPUS phi was F32",
        "python_version": sys.version,
        "versions": protocol["versions"],
        "conversion_receipt_sha256": digest(
            publication / "provenance/mrks_conversion.json"
        ),
    }
    pilot_data = {}
    for name, receipt in sources.items():
        budget()
        path = Path(receipt["path"])
        actual = digest(path)
        assert actual == receipt["sha256"], name
        provenance["sources"][name] = {"path": str(path), "sha256": actual}
        with np.load(path, allow_pickle=False) as z:
            meta = dict(zip(z["meta_keys"].tolist(), z["meta"].tolist()))
            energy, occ = z["mo_energy"], z["mo_occ"]
            assert np.isfinite(energy).all() and np.isfinite(occ).all()
            assert energy.shape == occ.shape and np.any(occ > 0)
            homo = float(energy[occ > 0].max())
            ip = meta.get("i_ekt")
            delta = homo + float(ip) if ip is not None and float(ip) > 0 else None
            metadata_rows.append(
                {
                    "system": name,
                    "source_sha256": actual,
                    "mo_coeff_shape": str(z["mo_coeff"].shape),
                    "mo_coeff_dtype": str(z["mo_coeff"].dtype),
                    "mo_energy_ha": json.dumps(energy.tolist()),
                    "mo_occ": json.dumps(occ.tolist()),
                    "dm_shape": str(z["dm_ks"].shape),
                    "dm_dtype": str(z["dm_ks"].dtype),
                    "electron_count": meta.get("nelec"),
                    "IP_field": "i_ekt" if ip is not None else "MISSING",
                    "IP_raw": ip,
                    "IP_units": "Ha (energy bundle/task convention; exporter unit annotation absent)",
                    "homo_saved_ha": homo,
                    "delta_IP_ha": delta,
                    "abs_delta_IP_ha": abs(delta) if delta is not None else None,
                    "convergence_metadata": json.dumps(meta),
                }
            )
            if name in protocol["pilot"]:
                pilot_data[name] = {key: z[key] for key in z.files}
    write_csv("all90_metadata", metadata_rows)
    assert len(metadata_rows) == 90 and len(mapping) == 90
    write_json("provenance", provenance)
    checks, shifts, grids = [], [], []
    array_dir = ROOT.parent / "lap_gauge_consistency_v1_arrays_20261014"
    array_dir.mkdir(exist_ok=True)
    for name in protocol["pilot"]:
        budget()
        z = pilot_data[name]
        mol = gto.M(
            atom=[
                (int(charge), tuple(coord))
                for charge, coord in zip(z["atom_charges"], z["atom_coords"])
            ],
            basis=json.loads(str(z["basis"].reshape(-1)[0])),
            unit="Bohr",
            charge=int(z["mol_charge"].reshape(-1)[0]),
            spin=int(z["mol_spin"].reshape(-1)[0]),
            verbose=0,
        )
        S = mol.intor_symmetric("int1e_ovlp")
        hcore = scf.hf.get_hcore(mol)
        P = z["dm_ks"]
        C, energy, occ = z["mo_coeff"], z["mo_energy"], z["mo_occ"]
        J = scf.hf.get_jk(mol, P, hermi=1, with_j=True, with_k=False)[0]
        source_coord, source_weight, source_potential = (
            z["grid_coords"],
            z["grid_weights"].reshape(-1),
            z["vxc_grid"].reshape(-1),
        )

        def ao_chunks(coords, weights, potential, mol=mol):
            for start in range(0, len(weights), 4096):
                budget()
                sl = slice(start, start + 4096)
                yield numint.eval_ao(mol, coords[sl]), weights[sl], potential[sl]

        Vs, Qs, Vsr, Qsr = project(
            ao_chunks(source_coord, source_weight, source_potential)
        )
        entry = mapping[name]
        shard = publication / entry["shard"]
        assert digest(shard) == hashes[entry["shard"]], name
        with h5py.File(shard, "r") as hf:
            group = hf[entry["group"]]
            source_meta = json.loads(group.attrs["source_metadata_json"])
            assert source_meta["source"]["npz_file_sha256"] == sources[name]["sha256"]
            assert source_meta["system_name"] == name
            assert np.array_equal(
                z["atom_coords"], source_meta["molecule"]["atom_coords_bohr"]
            )
            assert np.array_equal(
                z["atom_charges"], source_meta["molecule"]["atom_charges"]
            )
            assert mol.charge == source_meta["molecule"]["charge"]
            assert mol.spin == source_meta["molecule"]["spin"]
            assert mol.ao_labels() == source_meta["molecule"]["ao_labels"]
            coords, weights, potential = (
                group["coords64"][:],
                group["weights"][:].reshape(-1),
                group["VxcLegacy"][:].reshape(-1),
            )
            indices = group["npzrow"][:].astype(int).reshape(-1)
            assert np.array_equal(coords, source_coord[indices])
            # Preserve repeated mapped coordinates; historical corpus rows remain distinct.
            assert np.array_equal(
                weights.astype(np.float32), source_weight[indices].astype(np.float32)
            )
            assert np.array_equal(P, group["dmks"][:])
            assert np.allclose(S, group["Overlap"][:], atol=1e-10, rtol=1e-10)
            assert np.array_equal(
                potential.astype(np.float32),
                source_potential[indices].astype(np.float32),
            )

            def cached_chunks(group=group, weights=weights, potential=potential):
                for start in range(0, len(weights), 4096):
                    sl = slice(start, start + 4096)
                    yield group["phi"][sl], weights[sl], potential[sl]

            Vc, Qc, Vcr, Qcr = project(cached_chunks())
            Vf64, _, _, _ = project(ao_chunks(coords, weights, potential))
            reference = group["RefAO"][:]
            ev, u = eigh(S)
            X = (u / np.sqrt(ev)) @ u.T
            refnorm = np.linalg.norm(X @ reference @ X)
            ref_f64 = float(np.linalg.norm(X @ (Vf64 - reference) @ X))
            ref_cached = float(np.linalg.norm(X @ (Vc - reference) @ X))
            assert ref_f64 <= 1e-10 + 1e-10 * refnorm
            grids.append(
                {
                    "system": name,
                    "source_points": len(source_weight),
                    "corpus_points": len(weights),
                    "source_coords_dtype": str(source_coord.dtype),
                    "source_weights_dtype": str(source_weight.dtype),
                    "source_vxc_dtype": str(source_potential.dtype),
                    "corpus_phi_dtype": str(group["phi"].dtype),
                    "corpus_weights_dtype": str(weights.dtype),
                    "corpus_vxc_dtype": str(potential.dtype),
                    "negative_source_weights": int((source_weight < 0).sum()),
                    "negative_corpus_weights": int((weights < 0).sum()),
                    "row_identity": "EXACT",
                    "unique_npz_rows": len(np.unique(indices)),
                    "unique_legacy_rows": len(np.unique(group["sourcerow"][:])),
                    "weights_source_difference_max": float(
                        np.max(np.abs(weights - source_weight[indices]))
                    ),
                    "potential_source_difference_max": float(
                        np.max(np.abs(potential - source_potential[indices]))
                    ),
                    "delta_V_orthogonal_fro_ha": float(
                        np.linalg.norm(X @ (Vc - Vs) @ X)
                    ),
                    "refAO_F64_absolute_ha": ref_f64,
                    "refAO_cached_absolute_ha": ref_cached,
                    "refAO_cached_gate": ref_cached <= 1e-7 + 2e-6 * refnorm,
                    "source_reverse_sum_matrix_difference": float(
                        np.linalg.norm(Vsr - Vs)
                    ),
                    "corpus_reverse_sum_matrix_difference": float(
                        np.linalg.norm(Vcr - Vc)
                    ),
                }
            )
        provenance["pilot_corpus"][name] = {
            "path": str(shard),
            "sha256": hashes[entry["shard"]],
            "group": entry["group"],
            "mapping": entry,
        }
        Pmo = (C * occ) @ C.T
        dmerr = float(np.linalg.norm(Pmo - P))
        count = float(np.trace(P @ S))
        integrity = (
            dmerr <= 1e-10 + 1e-10 * np.linalg.norm(P)
            and abs(count - mol.nelectron) <= 1e-6
        )
        arrays = {
            "S": S,
            "hcore": hcore,
            "J_KS": J,
            "V_SOURCE": Vs,
            "V_CORPUS": Vc,
            "Q_SOURCE": Qs,
            "Q_CORPUS": Qc,
            "delta_V": Vc - Vs,
        }
        source_fails = False
        for path, V, Q in [("SOURCE", Vs, Qs), ("CORPUS", Vc, Qc)]:
            row, fitted, R = diagnostics(hcore + J + V, S, C, energy, occ, Q)
            Vr, Qr = (Vsr, Qsr) if path == "SOURCE" else (Vcr, Qcr)
            _, reverse_fits, _ = diagnostics(hcore + J + Vr, S, C, energy, occ, Qr)
            for fit, reverse_fit in zip(fitted, reverse_fits):
                fit["chunk_order_c_difference_ha"] = abs(
                    fit["c_add_ha"] - reverse_fit["c_add_ha"]
                )
            row.update(
                system=name,
                grid_path=path,
                Hartree="KS",
                DM_absolute=dmerr,
                DM_relative=dmerr / np.linalg.norm(P),
                electron_count=count,
            )
            integral_ok = integrity and row["mo_orthogonality_fro"] <= 1e-8
            if not integral_ok:
                status = "INSUFFICIENT_INPUTS_OR_NUMERICAL_RESOLUTION"
            elif row["raw_max_orbital_ha"] <= 1e-4:
                status = "NUMERICALLY_CONSISTENT_AT_REPORTED_RESOLUTION"
            elif any(
                f["fit_space"] == "full"
                and f["adjusted_max_orbital_ha"] <= 1e-4
                and abs(f["c_add_ha"]) > max(1e-6, f["arithmetic_bound_ha"])
                for f in fitted
            ):
                status = "RELATIVE_CONSTANT_OFFSET_IDENTIFIED"
            else:
                status = "NONCONSTANT_INCONSISTENCY"
            row["classification"] = status
            row["raw_eigenpair_gate"] = row["raw_max_orbital_ha"] <= 1e-4
            row["homo_rayleigh_plus_i_ekt_ha"] = (
                row["homo_rayleigh_ha"]
                + dict(zip(z["meta_keys"].tolist(), z["meta"].tolist()))["i_ekt"]
            )
            checks.append(row)
            shifts.extend(
                dict(system=name, grid_path=path, Hartree="KS", **f) for f in fitted
            )
            arrays["R_" + path] = R
            source_fails |= path == "SOURCE" and not row["raw_eigenpair_gate"]
        if source_fails and "dm1_ao_wf" in z:
            wf = z["dm1_ao_wf"]
            wf_valid = (
                wf.shape == P.shape
                and np.isfinite(wf).all()
                and np.linalg.norm(wf - wf.T) < 1e-10
                and abs(np.trace(wf @ S) - mol.nelectron) <= 1e-6
            )
            provenance["pilot_corpus"][name]["WF_counterfactual_integrity"] = bool(
                wf_valid
            )
            if wf_valid:
                Jwf = scf.hf.get_jk(mol, wf, hermi=1, with_j=True, with_k=False)[0]
                arrays["J_WF"] = Jwf
                for path, V, Q in [("SOURCE", Vs, Qs), ("CORPUS", Vc, Qc)]:
                    row, fitted, _ = diagnostics(hcore + Jwf + V, S, C, energy, occ, Q)
                    row.update(
                        system=name,
                        grid_path=path,
                        Hartree="WF_COUNTERFACTUAL",
                        classification="DIAGNOSTIC_ONLY_NOT_PRODUCER_HISTORY",
                    )
                    checks.append(row)
                    shifts.extend(
                        dict(
                            system=name,
                            grid_path=path,
                            Hartree="WF_COUNTERFACTUAL",
                            **f,
                        )
                        for f in fitted
                    )
        outfile = array_dir / (name + ".npz")
        np.savez_compressed(outfile, **arrays)
        provenance["pilot_corpus"][name]["arrays"] = {
            "path": str(outfile),
            "sha256": digest(outfile),
        }
        write_csv("eigenpair_checks", checks)
        write_csv("shift_decomposition", shifts)
        write_csv("grid_comparison", grids)
        write_json("provenance", provenance)
        print(name, checks[-1]["raw_max_orbital_ha"], flush=True)
    deltas = np.array(
        [r["delta_IP_ha"] for r in metadata_rows if r["delta_IP_ha"] is not None]
    )
    decision = {
        "metadata_coverage": len(metadata_rows),
        "EKT_available": len(deltas),
        "EKT_missing": 90 - len(deltas),
        "delta_IP_signed_median_ha": float(np.median(deltas)),
        "delta_IP_absolute_median_ha": float(np.median(abs(deltas))),
        "delta_IP_absolute_max_ha": float(abs(deltas).max()),
        "bands_counts": {
            str(b): int((abs(deltas) <= b).sum())
            for b in protocol["criteria"]["bands_ha"]
        },
        "pilot_results": [
            {
                k: r[k]
                for k in ["system", "grid_path", "classification", "raw_max_orbital_ha"]
            }
            for r in checks
            if r["Hartree"] == "KS"
        ],
        "historical_exporter_procedure": "NOT RECONSTRUCTED",
        "wall_seconds": time.time() - begin,
        "active_compute_seconds": time.monotonic() - active,
        "immutable_receipts_pass": all(
            digest(ROOT / p) == h for p, h in protocol["baseline_receipts"].items()
        ),
        "checkpoints_loaded": 0,
        "optimizer_states_loaded": 0,
        "training_updates": 0,
        "SCF_executed": False,
    }
    write_json("decision", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
