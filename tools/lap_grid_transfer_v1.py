"""Matched fixed-density PBE/reference grid-transfer attribution, no training."""

import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.lap_gauge_consistency_v1 import diagnostics, digest
from tools.lap_operator_contract_v1_audit import Canonical, ConstantAdapter
from train_models import lap_operator as op
from train_models.lap_operator_data import density_descriptors_from_ao
from train_models.lap_vxc import LapEnergy, local_partials

PREFIX = "lap_grid_transfer_v1_"


def save(name, value):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def table(name, rows):
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float64)


def reference(phi, weights, potential):
    return phi.T @ (phi * (weights * potential)[:, None])


def loss_identity(A, delta):
    n = len(A)
    return 2 * np.vdot(A, delta).real / n, np.vdot(delta, delta).real / n


def close(actual, expected, atol=1e-10, rtol=1e-9):
    gap = float(np.linalg.norm(actual - expected))
    bound = atol + rtol * float(np.linalg.norm(expected))
    return {
        "absolute": gap,
        "bound": bound,
        "status": "PASS" if gap <= bound else "FAIL",
    }


def assemble(partials, w, p, dp, lp):
    e, c, a, b = partials
    v = op._assemble_from_local_partials(
        c.mean(-1),
        a.mean(1),
        b.mean(-1),
        tensor(w),
        tensor(p),
        tensor(dp),
        tensor(lp),
        4096,
    )
    return v.detach().numpy(), float(np.dot(e.detach().numpy(), w))


def main():
    from pyscf import gto
    from pyscf.dft import numint

    protocol = json.loads((ROOT / (PREFIX + "protocol.json")).read_text())
    started = protocol["first_action_unix"]
    active = time.monotonic()

    def budget():
        if time.time() - started > 5100:
            raise TimeoutError("Reserved final integrity/report budget")

    for path, receipt in protocol["source_hashes"].items():
        assert digest(ROOT / path) == receipt
    old = json.loads((ROOT / "lap_gauge_consistency_v1_provenance.json").read_text())
    outdir = ROOT.parent / "lap_grid_transfer_v1_arrays_20261014"
    outdir.mkdir(exist_ok=True)
    provenance = {
        "driver_sha256": digest(__file__),
        "pilots": {},
        "baseline_head": protocol["head"],
        "versions": old["versions"],
        "scientific_inputs_unchanged": False,
    }
    references, predictions, losses, inventory = [], [], [], []
    closure = {}
    profile = []
    for name in protocol["pilot"]:
        budget()
        before = time.monotonic()
        receipt = old["sources"][name]
        assert digest(receipt["path"]) == receipt["sha256"]
        cache = old["pilot_corpus"][name]
        assert digest(cache["path"]) == cache["sha256"]
        assert digest(cache["arrays"]["path"]) == cache["arrays"]["sha256"]
        with np.load(receipt["path"], allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        old_arrays = np.load(cache["arrays"]["path"], allow_pickle=False)
        S, hcore, J = old_arrays["S"], old_arrays["hcore"], old_arrays["J_KS"]
        ev, u = eigh(S)
        X = (u / np.sqrt(ev)) @ u.T
        Shalf = (u * np.sqrt(ev)) @ u.T
        C, eps, occ = data["mo_coeff"], data["mo_energy"], data["mo_occ"]
        mol = gto.M(
            atom=[
                (int(z), tuple(x))
                for z, x in zip(data["atom_charges"], data["atom_coords"])
            ],
            basis=json.loads(str(data["basis"].reshape(-1)[0])),
            unit="Bohr",
            charge=int(data["mol_charge"].reshape(-1)[0]),
            spin=int(data["mol_spin"].reshape(-1)[0]),
            verbose=0,
        )
        with h5py.File(cache["path"], "r") as hf:
            group = hf[cache["group"]]
            metadata = json.loads(group.attrs["source_metadata_json"])
            assert metadata["source"]["npz_file_sha256"] == receipt["sha256"]
            assert mol.ao_labels() == metadata["molecule"]["ao_labels"]
            assert np.array_equal(group["dmks"][:], data["dm_ks"])
            assert np.array_equal(group["Overlap"][:], S)
            mapping = group["npzrow"][:].astype(int).reshape(-1)
            coords = data["grid_coords"]
            w = data["grid_weights"].reshape(-1)
            v = data["vxc_grid"].reshape(-1)
            wc = group["weights"][:].reshape(-1)
            vc = group["VxcLegacy"][:].reshape(-1)
            assert np.array_equal(coords[mapping], group["coords64"][:])
            assert np.array_equal(wc, wc.astype(np.float32).astype(float))
            assert np.array_equal(vc.astype(np.float32), v[mapping].astype(np.float32))
            counts = np.bincount(mapping, minlength=len(w))
            assert (len(w), len(mapping)) == {
                "H2": (47080, 42508),
                "BeH2": (89200, 77240),
                "CO": (85960, 72110),
            }[name]
            unique = counts > 0
            weight_sum = np.bincount(mapping, weights=wc, minlength=len(w))
            reference_sum = np.bincount(mapping, weights=wc * vc, minlength=len(w))
            coord_keys = [tuple(point) for point in coords]
            coordinate_indices = {}
            f32_groups = {}
            for i, point in enumerate(coords):
                coordinate_indices.setdefault(coord_keys[i], []).append(i)
                f32_groups.setdefault(tuple(point.astype(np.float32)), set()).add(
                    coord_keys[i]
                )
            ambiguity = sum(len(value) > 1 for value in f32_groups.values())
            for i in range(len(w)):
                same = coordinate_indices[coord_keys[i]]
                inventory.append(
                    {
                        "system": name,
                        "source_index": i,
                        "selected": bool(unique[i]),
                        "multiplicity": int(counts[i]),
                        "source_weight": float(w[i]),
                        "mapped_corpus_weight_sum": float(weight_sum[i]),
                        "source_vxc_ha": float(v[i]),
                        "same_coordinate_source_count": len(same),
                        "coordinate_source_weight_sum": float(w[same].sum()),
                        "coordinate_corpus_weight_sum": float(weight_sum[same].sum()),
                        "mapping_history": "existing npzrow, first identical coordinate candidate; no new map",
                    }
                )
            weights = [w, w * unique, w * counts, weight_sum, weight_sum]
            ref_weights = [weights[k] * v if k < 4 else reference_sum for k in range(5)]
            n = len(S)
            pred = [np.zeros((n, n)) for _ in range(10)]
            ref = [np.zeros((n, n)) for _ in range(10)]
            Q = [np.zeros((n, n)) for _ in range(10)]
            energy = np.zeros(10)
            nelec = np.zeros(10)
            features_source = np.empty((len(w), 10))
            deletion_pred = np.zeros((n, n))
            excess_pred = deletion_pred.copy()
            deletion_ref = deletion_pred.copy()
            excess_ref = deletion_pred.copy()
            source_blocks = []
            for start in range(0, len(w), 4096):
                budget()
                sl = slice(start, start + 4096)
                ao = numint.eval_ao(mol, coords[sl], deriv=2)
                p, dp, lp = ao[0], ao[1:4].transpose(1, 0, 2), ao[4] + ao[7] + ao[9]
                f = density_descriptors_from_ao(mol, ao, data["dm_ks"])
                features_source[sl] = f
                partials = local_partials(Canonical(), tensor(f), create_graph=False)
                for k in range(5):
                    pp, ee = assemble(partials, weights[k][sl], p, dp, lp)
                    pred[k] += pp
                    energy[k] += ee
                    rr = reference(p, ref_weights[k][sl], np.ones(len(p)))
                    ref[k] += rr
                    Q[k] += reference(p, weights[k][sl], np.ones(len(p)))
                    nelec[k] += np.dot(f[:, :2].sum(-1), weights[k][sl])
                    if k == 0:
                        source_blocks.append((pp, rr))
                wd = -w[sl] * (counts[sl] == 0)
                we = w[sl] * np.maximum(counts[sl] - 1, 0)
                deletion_pred += assemble(partials, wd, p, dp, lp)[0]
                excess_pred += assemble(partials, we, p, dp, lp)[0]
                deletion_ref += reference(p, wd, v[sl])
                excess_ref += reference(p, we, v[sl])
            stored_ref = group["RefAO"][:]
            closure[name] = {
                "SOURCE_reference_replay": close(
                    ref[0], old_arrays["V_SOURCE"], 1e-10, 1e-10
                ),
                "R4_RefAO_replay": close(ref[4], stored_ref, 1e-10, 1e-10),
                "selection_multiplicity_reference": close(
                    deletion_ref + excess_ref, ref[2] - ref[0]
                ),
                "selection_multiplicity_prediction": close(
                    deletion_pred + excess_pred, pred[2] - pred[0]
                ),
                "vxc_only_does_not_change_prediction": close(pred[4], pred[3]),
                "F64_chunk_order_prediction": close(
                    sum((x[0] for x in reversed(source_blocks)), np.zeros((n, n))),
                    pred[0],
                ),
                "F64_chunk_order_reference": close(
                    sum((x[1] for x in reversed(source_blocks)), np.zeros((n, n))),
                    ref[0],
                ),
            }
            corpus_cached_ref = np.zeros((n, n))
            production_replay = np.zeros((n, n))
            for start in range(0, len(mapping), 4096):
                budget()
                sl = slice(start, start + 4096)
                ao = numint.eval_ao(mol, coords[mapping[sl]], deriv=2)
                p, dp, lp = ao[0], ao[1:4].transpose(1, 0, 2), ao[4] + ao[7] + ao[9]
                cp = group["phi"][sl].astype(float)
                cd = group["grad_phi"][sl].astype(float)
                cl = group["lap_phi"][sl].astype(float)
                fs = group["DensityDescriptorsN10"][sl]
                ff = fs.astype(np.float32).astype(float)
                assert np.allclose(
                    features_source[mapping[sl]], fs, atol=1e-10, rtol=1e-9
                )
                for k, feat, pp, dd, ll in [
                    (5, fs, p, dp, lp),
                    (6, ff, p, dp, lp),
                    (7, ff, cp, dp, lp),
                    (8, ff, cp, cd, cl),
                    (9, ff, cp, cd, cl),
                ]:
                    part = (
                        op._model_double_operator_partials(
                            LapEnergy(ConstantAdapter()), tensor(feat), False
                        )
                        if k == 9
                        else local_partials(Canonical(), tensor(feat), False)
                    )
                    block, ee = assemble(part, wc[sl], pp, dd, ll)
                    pred[k] += block
                    energy[k] += ee
                    Q[k] += reference(pp, wc[sl], np.ones(len(pp)))
                    nelec[k] += np.dot(feat[:, :2].sum(-1), wc[sl])
                corpus_cached_ref += reference(cp, wc[sl], vc[sl])
                # Actual independent production entry point with native F32 inputs.
                block = op.assemble_rks_operator(
                    LapEnergy(ConstantAdapter()),
                    torch.as_tensor(fs, dtype=torch.float32),
                    torch.as_tensor(wc[sl], dtype=torch.float32),
                    torch.as_tensor(group["phi"][sl]),
                    torch.as_tensor(group["grad_phi"][sl]),
                    torch.as_tensor(group["lap_phi"][sl]),
                    chunk_size=4096,
                    create_graph=False,
                )
                production_replay += block.detach().numpy()
            for k in range(5, 10):
                ref[k] = stored_ref.copy()
            closure[name]["CORPUS_cached_reference_replay"] = close(
                corpus_cached_ref, old_arrays["V_CORPUS"], 1e-10, 1e-10
            )
            closure[name]["production_prediction_replay"] = close(
                pred[9], production_replay
            )
            closure[name]["production_reference_remains_F64"] = close(
                ref[9], stored_ref, 0, 0
            )
            U = Shalf @ C
            assert np.linalg.norm(U.T @ U - np.eye(len(S))) < 1e-8
            occmask = occ > 0
            A = [X @ (pred[k] - ref[k]) @ X for k in range(10)]
            total_da = A[-1] - A[0]
            total_dr = X @ (ref[-1] - ref[0]) @ X
            total_dp = X @ (pred[-1] - pred[0]) @ X
            increments = []
            stages_arrays = {
                "deletion_reference": deletion_ref,
                "excess_reference": excess_ref,
                "deletion_prediction": deletion_pred,
                "excess_prediction": excess_pred,
                "features_SOURCE": features_source,
            }
            for k in range(10):
                dr = X @ (ref[k] - ref[k - 1]) @ X if k else np.zeros((n, n))
                dd = X @ (pred[k] - pred[k - 1]) @ X if k else dr.copy()
                da = dd - dr
                increments.append(da)
                loss = float(np.vdot(A[k], A[k]).real / n)
                cross, square = loss_identity(A[k - 1], da) if k else (0.0, 0.0)
                row, _fits, _R = diagnostics(hcore + J + ref[k], S, C, eps, occ, Q[k])
                mo_A = U.T @ A[k] @ U
                mo_da = U.T @ da @ U
                losses.append(
                    {
                        "system": name,
                        "stage": "R" + str(k),
                        "loss_ha2": loss,
                        "delta_loss_ha2": cross + square,
                        "cross_term_ha2": cross,
                        "quadratic_term_ha2": square,
                        "PBE_energy_ha": float(energy[k]),
                        "nelec_integral": float(nelec[k]),
                        "Q_error_spectral": row["B_minus_I_spectral"],
                        "reference_max_orbital_ha": row["raw_max_orbital_ha"],
                        "reference_HOMO_residual_ha": row["homo_residual_ha"],
                        "reference_occupied_fro_ha": row["occupied_fro_ha"],
                        "reference_virtual_fro_ha": row["virtual_fro_ha"],
                        "occupied_block_loss_ha2": float(
                            np.sum(mo_A[np.ix_(occmask, occmask)] ** 2) / n
                        ),
                        "virtual_block_loss_ha2": float(
                            np.sum(mo_A[np.ix_(~occmask, ~occmask)] ** 2) / n
                        ),
                        "cross_block_loss_ha2": float(
                            2 * np.sum(mo_A[np.ix_(occmask, ~occmask)] ** 2) / n
                        ),
                        "delta_A_occ_fro": float(
                            np.linalg.norm(mo_da[np.ix_(occmask, occmask)])
                        ),
                        "delta_A_virtual_fro": float(
                            np.linalg.norm(mo_da[np.ix_(~occmask, ~occmask)])
                        ),
                    }
                )
                for result, diff, total in [
                    (references, dr, total_dr),
                    (predictions, dd, total_dp),
                ]:
                    result.append(
                        {
                            "system": name,
                            "stage": "R" + str(k),
                            "delta_orthogonal_fro": float(np.linalg.norm(diff)),
                            "signed_projection_on_total": float(
                                np.vdot(diff, total).real / np.vdot(total, total).real
                            )
                            if np.linalg.norm(total) > 1e-14
                            else None,
                            "delta_A_fro": float(np.linalg.norm(da)),
                        }
                    )
                stages_arrays.update(
                    {
                        f"Vref_R{k}": ref[k],
                        f"Vpred_R{k}": pred[k],
                        f"A_R{k}": A[k],
                        f"delta_ref_R{k}": dr,
                        f"delta_pred_R{k}": dd,
                        f"delta_A_R{k}": da,
                    }
                )
            cross, square = loss_identity(A[0], total_da)
            gap = float(np.sum(A[-1] ** 2) / n - np.sum(A[0] ** 2) / n)
            closure[name].update(
                {
                    "matrix_delta_closure": close(total_da, total_dp - total_dr),
                    "matrix_telescoping": close(sum(increments), total_da),
                    "loss_identity": close(np.array(gap), np.array(cross + square)),
                    "loss_telescoping": close(
                        np.array(
                            sum(
                                r["delta_loss_ha2"]
                                for r in losses
                                if r["system"] == name
                            )
                        ),
                        np.array(gap),
                    ),
                }
            )
            output = outdir / (name + ".npz")
            np.savez_compressed(output, **stages_arrays)
            provenance["pilots"][name] = {
                "source": receipt,
                "corpus": cache,
                "arrays": {"path": str(output), "sha256": digest(output)},
                "n_SOURCE": len(w),
                "n_CORPUS": len(mapping),
                "selected_unique_indices": int(unique.sum()),
                "unselected_indices": int((~unique).sum()),
                "excess_mapped_presentations": int(np.maximum(counts - 1, 0).sum()),
                "F32_distinct_coordinate_collisions": ambiguity,
                "duplicated_SOURCE_coordinates": sum(
                    len(ii) > 1 for ii in coordinate_indices.values()
                ),
                "mapping_scope": "VERIFIED bookkeeping; identical-source-coordinate candidate provenance cannot separate physical deletion/multiplicity history",
                "delta_reference_fro": float(np.linalg.norm(total_dr)),
                "delta_prediction_fro": float(np.linalg.norm(total_dp)),
                "delta_residual_fro": float(np.linalg.norm(total_da)),
                "delta_ref_pred_cosine": float(
                    np.vdot(total_dr, total_dp).real
                    / np.linalg.norm(total_dr)
                    / np.linalg.norm(total_dp)
                ),
                "L_SOURCE": float(np.sum(A[0] ** 2) / n),
                "L_CORPUS": float(np.sum(A[-1] ** 2) / n),
                "delta_loss": gap,
                "relative_delta_loss": gap / float(np.sum(A[0] ** 2) / n),
                "full_cross_term": cross,
                "full_quadratic_term": square,
                "PBE_energy_change_ha": float(energy[-1] - energy[0]),
                "time_seconds": time.monotonic() - before,
            }
        profile.append(provenance["pilots"][name]["time_seconds"])
        print(name, provenance["pilots"][name], flush=True)
        if name == "H2":
            assert sum(profile) * 4 + 600 < 5400 - (time.time() - started), (
                "Profile leaves insufficient mandatory finish budget"
            )
        inventory_path = outdir / "source_row_inventory.csv"
        with inventory_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
        summaries = []
        for identity in protocol["pilot"]:
            subset = [r for r in inventory if r["system"] == identity]
            for multiplicity in sorted({r["multiplicity"] for r in subset}):
                subset_m = [r for r in subset if r["multiplicity"] == multiplicity]
                summaries.append(
                    {
                        "system": identity,
                        "multiplicity": multiplicity,
                        "source_index_count": len(subset_m),
                        "source_weight_sum": sum(r["source_weight"] for r in subset_m),
                        "corpus_weight_sum": sum(
                            r["mapped_corpus_weight_sum"] for r in subset_m
                        ),
                        "full_inventory_path": str(inventory_path),
                        "full_inventory_sha256": digest(inventory_path),
                    }
                )
        for filename, rows in [
            ("row_inventory", summaries),
            ("reference_attribution", references),
            ("prediction_attribution", predictions),
            ("matched_loss", losses),
        ]:
            table(filename, rows)
        save("closure_checks", closure)
        save("provenance", provenance)
    provenance["active_numerical_seconds"] = time.monotonic() - active
    provenance["scientific_inputs_unchanged"] = all(
        digest(ROOT / path) == receipt
        for path, receipt in protocol["source_hashes"].items()
    ) and all(
        digest(p["source"]["path"]) == p["source"]["sha256"]
        and digest(p["corpus"]["path"]) == p["corpus"]["sha256"]
        for p in provenance["pilots"].values()
    )
    save("provenance", provenance)
    save(
        "decision",
        {
            "wall_seconds": time.time() - started,
            "active_numerical_seconds": provenance["active_numerical_seconds"],
            "budget_respected": time.time() - started < 5400,
            "closure_status": "PASS"
            if all(r["status"] == "PASS" for v in closure.values() for r in v.values())
            else "FAIL",
            "production_endpoint_replay": "PASS"
            if all(
                v["production_prediction_replay"]["status"] == "PASS"
                for v in closure.values()
            )
            else "FAIL",
            "row_attribution": "PARTIAL historical equal-coordinate disambiguation; verified fixed-map arithmetic",
            "filter_rule": "RULE_UNKNOWN; available masking helpers not authenticated as this export",
            "trained_checkpoints_loaded": 0,
            "optimizer_states_loaded": 0,
            "training_updates": 0,
            "SCF": False,
        },
    )


if __name__ == "__main__":
    main()
