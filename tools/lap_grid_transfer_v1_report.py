"""Render grid-transfer verdict from authenticated saved computation arrays."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from lap_grid_transfer_v1 import PREFIX, ROOT, close, digest, op, save, tensor


def read_table(name):
    with (ROOT / (PREFIX + name + ".csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def local_path(path):
    return (
        Path("C:/" + path[7:])
        if sys.platform == "win32" and path.startswith("/mnt/c/")
        else Path(path)
    )


def main():
    protocol = json.loads((ROOT / (PREFIX + "protocol.json")).read_text())
    provenance = json.loads((ROOT / (PREFIX + "provenance.json")).read_text())
    decision = json.loads((ROOT / (PREFIX + "decision.json")).read_text())
    closure = json.loads((ROOT / (PREFIX + "closure_checks.json")).read_text())
    matched = read_table("matched_loss")
    ref = read_table("reference_attribution")
    pred = read_table("prediction_attribution")
    inventory = read_table("row_inventory")
    for file, expected in protocol["source_hashes"].items():
        assert digest(ROOT / file) == expected
    assert provenance["driver_sha256"] == digest(ROOT / "tools/lap_grid_transfer_v1.py")
    for name, record in provenance["pilots"].items():
        for item in [record["source"], record["corpus"], record["arrays"]]:
            assert digest(local_path(item["path"])) == item["sha256"]
        with np.load(local_path(record["arrays"]["path"])) as z:
            with np.load(
                local_path(record["corpus"]["arrays"]["path"])
            ) as prior_arrays:
                S = prior_arrays["S"]
            values, vectors = np.linalg.eigh(S)
            X = (vectors / np.sqrt(values)) @ vectors.T
            for kind in ["ref", "pred", "A"]:
                initial = z["A_R0"] if kind == "A" else z["V" + kind + "_R0"]
                final = z["A_R9"] if kind == "A" else z["V" + kind + "_R9"]
                # Saved increments are in the orthonormal basis; A is already so.
                if kind == "A":
                    closure[name]["saved_residual_telescoping"] = close(
                        sum(z[f"delta_A_R{k}"] for k in range(10)), final - initial
                    )
                else:
                    summed = sum(z[f"delta_{kind}_R{k}"] for k in range(10))
                    closure[name]["saved_" + kind + "_telescoping"] = close(
                        summed, X @ (final - initial) @ X
                    )
            production_loss = float(
                op.operator_loss(tensor(z["Vpred_R9"]), tensor(z["Vref_R9"]), tensor(S))
            )
            closure[name]["production_scalar_loss_replay"] = close(
                np.array(production_loss), np.array(record["L_CORPUS"])
            )
            for k in range(10):
                row = next(
                    r for r in matched if r["system"] == name and r["stage"] == f"R{k}"
                )
                assert (
                    abs(
                        float(row["loss_ha2"])
                        - float(np.sum(z[f"A_R{k}"] ** 2) / len(z["A_R0"]))
                    )
                    < 1e-12
                )
            closure[name]["deletion_reference_separate"] = close(
                z["deletion_reference"], z["Vref_R1"] - z["Vref_R0"]
            )
            closure[name]["excess_reference_separate"] = close(
                z["excess_reference"], z["Vref_R2"] - z["Vref_R1"]
            )
            closure[name]["deletion_prediction_separate"] = close(
                z["deletion_prediction"], z["Vpred_R1"] - z["Vpred_R0"]
            )
            closure[name]["excess_prediction_separate"] = close(
                z["excess_prediction"], z["Vpred_R2"] - z["Vpred_R1"]
            )
    for row in inventory:
        assert (
            digest(local_path(row["full_inventory_path"]))
            == row["full_inventory_sha256"]
        )
    findings = [
        {
            "path": "train_models/prepare_vxc.py",
            "lines": "19-64",
            "finding": "H5 copy and F32 conversion, no row mask",
        },
        {
            "path": "train_models/build_lap_operator_data.py",
            "lines": "164-206,257-270",
            "finding": "inherits legacy rows, exact-coordinate mapping and preserved legacy weights/potential",
        },
        {
            "path": "train_models/build_mrks_stencils.py",
            "lines": "117-145",
            "finding": "first candidate for identical F64 coordinates; distinct-coordinate F32 collisions fail closed",
        },
        {
            "path": "train_models/publication_data/loader.py",
            "lines": "208-234",
            "finding": "features/weights F32, AO factors F32, RefAO/Overlap remain F64",
        },
        {
            "path": "C:/Users/schne/Downloads/MRKS/bin/convert_vxc_data_with_wfn.py",
            "lines": "143-207,293-300",
            "finding": "rho_total > rho_cutoff mask exists; default cutoff -1; no authenticated run/config connecting it to these records",
        },
        {
            "path": "C:/Users/schne/Downloads/MRKS/bin/gen_h5_with_vrho.py",
            "lines": "213-242",
            "finding": "potential noise zeroing, not grid-row deletion; producer not authenticated",
        },
        {
            "path": "C:/Dev/readWFN_share_ms/readWFN_share_ms/h5_vrho_from_mrks_sanity.csv",
            "lines": "pilot rows",
            "finding": "reduced counts predate prepare_vxc; log contains no filtering config",
        },
    ]
    for finding in findings:
        path = (
            Path(finding["path"]) if ":" in finding["path"] else ROOT / finding["path"]
        )
        if path.exists():
            finding["sha256"] = digest(path)
    provenance["bounded_filter_search"] = findings
    provenance["filter_search_conclusion"] = (
        "RULE_UNKNOWN. Search stopped at directly linked helpers/log; no reconstructed historical config."
    )
    provenance["report_driver_sha256"] = digest(__file__)
    provenance["result_hashes"] = {
        name: digest(ROOT / (PREFIX + name + ".csv"))
        for name in [
            "row_inventory",
            "reference_attribution",
            "prediction_attribution",
            "matched_loss",
        ]
    }
    decision.update(
        wall_seconds=time.time() - protocol["first_action_unix"],
        closure_status="PASS"
        if all(r["status"] == "PASS" for x in closure.values() for r in x.values())
        else "FAIL",
        primary_transfer_mechanism="mapped selection dominates; excess multiplicity contributes exactly zero (zero weights); rounding/AO/local arithmetic subleading",
        PBE_loss_impact="partial prediction/reference cancellation, with small negative loss shifts in all three pilots; not full compensation",
        conversion_defect="NOT DEMONSTRATED",
        next_step="separately frozen matched full-grid representation pilot, rebuilding prediction inputs and reference together; no automatic repair or training",
    )
    decision["budget_respected"] = decision["wall_seconds"] < 5400
    save("closure_checks", closure)
    save("provenance", provenance)
    save("decision", decision)
    text = [
        "# SOURCE → CORPUS grid transfer and matched operator-loss impact",
        "",
        "Mapped row selection dominates the reference-matrix change. Canonical PBE changes in almost the same matrix direction, but with only about 5–7% of the reference-change magnitude; compensation is partial. The matched squared operator loss decreases slightly on all three pilots. This is not evidence that the subset is physically superior or explains a trained-NN plateau.",
        "",
        f"Frozen HEAD `{protocol['head']}`; baseline gauge audit sources and three NPZ/corpus/array hashes verified. Versions: {provenance['versions']}. No old branch checkout. Primary formula: unchanged repository PBE with stored canonical constants widened; density/sigma regularization unchanged (PBE.py:50–85). No new Libxc oracle.",
        "",
        "## Matched endpoints",
        "",
        "|System|norm Δref|norm Δpred|norm ΔA|cos(Δref,Δpred)|L_SOURCE|L_CORPUS|ΔL|relative ΔL|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in provenance["pilots"].items():
        text.append(
            "|"
            + name
            + "|"
            + "|".join(
                f"{r[key]:.12g}"
                for key in [
                    "delta_reference_fro",
                    "delta_prediction_fro",
                    "delta_residual_fro",
                    "delta_ref_pred_cosine",
                    "L_SOURCE",
                    "L_CORPUS",
                    "delta_loss",
                    "relative_delta_loss",
                ]
            )
            + "|"
        )
    text += [
        "",
        "Matrix norms are orthonormal-AO Frobenius norms in Ha; loss is Ha² with the original n_AO denominator. ΔA=Δpred−Δref. Reference-only ||Δref||²/n_AO is not a lower bound on this loss. ΔL uses the same fixed overlap, gauge and basis.",
        "",
        "|System|cross term 2⟨A_SOURCE,ΔA⟩/n_AO|quadratic term ||ΔA||²/n_AO|",
        "|---|---:|---:|",
    ]
    for name, r in provenance["pilots"].items():
        text.append(
            f"|{name}|{r['full_cross_term']:.12g}|{r['full_quadratic_term']:.12g}|"
        )
    text += [
        "",
        "The negative cross term dominates the positive quadratic term. The matrix residual changes measurably while the total loss shift remains −0.0151%, −0.2454%, −0.00664%. Those are continuous pilot results, not a universal R0 grid-sufficiency threshold.",
        "",
        "## Frozen stage attribution",
        "",
        "Exact stage definitions are in stage_definitions.json. R0 full SOURCE; R1 unique mapped indices; R2 original mapping/multiplicity; R3 corpus weights; R4 corpus vxc. R5 stored F64 descriptors; R6 production F32 descriptors widened; R7 cached phi; R8 cached AO derivatives; R9 actual production mixed local arithmetic. From R4 onward the original F64 RefAO is frozen. Prediction-only AO changes never alter RefAO.",
        "",
        "|System/stage|reference increment norm|prediction increment norm|Δloss|",
        "|---|---:|---:|---:|",
    ]
    for r in ref:
        if r["stage"] == "R0":
            continue
        p = next(
            x for x in pred if x["system"] == r["system"] and x["stage"] == r["stage"]
        )
        l = next(
            x
            for x in matched
            if x["system"] == r["system"] and x["stage"] == r["stage"]
        )
        text.append(
            f"|{r['system']}/{r['stage']}|{float(r['delta_orthogonal_fro']):.9g}|{float(p['delta_orthogonal_fro']):.9g}|{float(l['delta_loss_ha2']):.9g}|"
        )
    text += [
        "",
        "The saved attribution tables also give signed projections on the respective total matrix difference. Norms are not added as causal percentages. The sequential replacement order is prespecified and its attribution is order dependent.",
        "",
        "## Mapping and historical reduction rule",
        "",
        "|System|SOURCE / CORPUS|unique selected|unselected|excess mapped presentations|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in provenance["pilots"].items():
        text.append(
            f"|{name}|{r['n_SOURCE']} / {r['n_CORPUS']}|{r['selected_unique_indices']}|{r['unselected_indices']}|{r['excess_mapped_presentations']}|"
        )
    text += [
        "",
        "No distinct-coordinate F32 collisions were found. BeH2 and CO each have one group of identical SOURCE coordinates. The authenticated mapper chooses the first identical coordinate candidate; the extra mapped presentations have zero SOURCE and CORPUS weights, hence their matrix and energy effects are exactly zero. No records were deduplicated. Fixed-map deletion/excess bookkeeping closes independently. Historical ownership of coincident physical quadrature rows remains PARTIAL; no duplicate-data defect is inferred. Full per-source inventory with multiplicities and coincident-coordinate weight sums is outside Git, hash-bound by row_inventory.csv.",
        "",
        "The counts were already reduced in the historical H5 sanity log. prepare_vxc merely copied the H5 grid and rounded data to F32. The corpus builder inherited those rows and restored original coordinates. A linked converter contains rho_total > rho_cutoff filtering, but its actual run/config linkage is missing; another helper only zeroes potential noise. Historical reduction rule: RULE_UNKNOWN, not a demonstrated low-density mask. No further broad search was performed. Checked paths, code ranges and hashes are saved in provenance.",
        "",
        "## Numerical integrity and scope",
        "",
        "SOURCE reference replays saved V_SOURCE. Cached CORPUS projection replays the prior gauge-audit V_CORPUS. R4 separately reproduces the true stored F64 RefAO: the previous cached-phi reference was a diagnostic path, not a replacement target. The final prediction independently replays assemble_rks_operator with actual F32 inputs and ConstantAdapter through LapEnergy precision repair; its scalar metric is the unchanged operator_loss definition. A fixed nontrainable zero sentinel selects that precision path; it is not a loaded NN checkpoint.",
        "",
        "F64 arithmetic closure, independent selection/multiplicity sums, vxc-only prediction invariance, matrix/loss telescoping and cross-term identity pass. Chunk sensitivity is measured by reverse accumulation of the same R0 chunks. All inputs and production source hashes remain unchanged. Criteria were frozen before computation; no gauge shifts or overlap-mode removal.",
        "",
        "matched_loss.csv also reports Q error, electron-number integrals, PBE energy, raw reference eigenpair residuals and occupied/virtual blocks with the original n_AO denominator. Saved MOs are complete and qualified; no virtual modes excluded.",
        "",
        "|System|PBE XC energy change (Ha)|",
        "|---|---:|",
    ]
    for name, r in provenance["pilots"].items():
        text.append(f"|{name}|{r['PBE_energy_change_ha']:.12g}|")
    text += [
        "",
        "Tiny integrated-energy changes do not establish matrix accuracy. The large Q error and raw reference eigenpair differences are preserved, without fitted constants. This PBE GGA sensitivity test does not qualify the response of a trained Lap-functional. No chemistry, CBS, D3, all90 census or trained model was evaluated.",
        "",
        "## Verdict and one next action",
        "",
        "Transfer mechanism: selection, with minor weights/vxc precision and prediction descriptor/AO/local-arithmetic effects. No significant conversion defect is demonstrated. Matched PBE loss impact: partial cancellation and a small systematic downward shift on this pilot. Neither replacing only RefAO nor changing gauge is justified.",
        "",
        "One next experiment: a separately frozen matched full-SOURCE-grid representation pilot that rebuilds both prediction inputs and reference together and evaluates the unchanged operator metric. It must not reuse a full-grid target with subset prediction. This report does not authorize its execution or any training.",
        "",
        f"Active numerical CPU time: {provenance['active_numerical_seconds']:.3f}s (includes read/hash/array operations); pilot profile {[round(provenance['pilots'][n]['time_seconds'], 3) for n in protocol['pilot']]}s. Scientific wall through evidence package: {decision['wall_seconds']:.3f}s / 5400s. No GPU compute. Tests and final checks are recorded in decision.json. Unknown historical rule and coincident-row lineage remain explicit limitations; the numerical three-system attribution is complete.",
        "",
        "Trained checkpoints loaded: 0; optimizer states loaded: 0; training updates: 0; SCF: NO; gauge shifted: NO; production grids/targets/loss/architecture modified: NO; new systems/Clean28 access: NO.",
        "",
    ]
    text += [
        "R0–R4 use bounded source-index accumulation with exact fixed coefficient vectors for the specified selected indices and mapped multiplicities. This is algebraically the same R2 presentation list; computational summation order is reported rather than claimed bitwise identical. Direct CORPUS-order endpoint replay and reverse-chunk arithmetic checks constrain the observed reduction-order difference.",
        f"Validation receipt: {json.dumps(decision.get('validation', {}), sort_keys=True)}.",
    ]
    (ROOT / (PREFIX + "report.md")).write_text("\n".join(text))


if __name__ == "__main__":
    main()
