"""Hash-bound report and integrity checks for the one fresh matched pilot."""

import hashlib
import json
import sys
import time

import h5py
import numpy as np
import torch
from lap_fresh_grid_pilot_v1 import (
    OUT,
    PREFIX,
    ROOT,
    gradient_metrics,
    local,
    read,
    resume_payload,
    sha,
    table,
    write,
)


def rows(name):
    import csv

    with (ROOT / (PREFIX + name + ".csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def payload_hash(state, names):
    h = hashlib.sha256()
    for name in names:
        value = state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def main():
    p = read("protocol")
    decision = read("decision")
    provenance = read("provenance")
    initial = read("initialization")
    representations = read("representations")
    resource = read("resource_report")
    preflight = read("preflight")
    constraints = read("constraint_checks")
    cross = rows("cross_evaluation")
    logs = rows("training_log")
    geometry = rows("gradient_comparison")
    assert decision["updates"] == {"SUBSET": 32, "FULL_SOURCE": 32}
    assert len(logs) == 64 and len(cross) == 12
    for arm in p["arms"]:
        updates = [int(r["update"]) for r in logs if r["grid"] == arm]
        assert updates == list(range(1, 33))
        assert all(
            int(r["system_presentations"]) == 3 for r in logs if r["grid"] == arm
        )
    for path, expected in p["source_hashes"].items():
        assert sha(ROOT / path) == expected
    assert provenance["driver_sha256"] == sha(ROOT / "tools/lap_fresh_grid_pilot_v1.py")
    for r in representations["records"].values():
        for item in [r, r["source"], r["corpus_mapping_source"]]:
            assert sha(local(item["path"])) == item["sha256"]
        with h5py.File(local(r["path"])) as f:
            assert (
                len(f["weights"]) == r["SOURCE_count"]
                and len(f["map"]) == r["SUBSET_count"]
            )
            assert all(
                f[k].dtype == np.float64
                for k in ["phi", "grad", "lap", "features", "weights", "vxc"]
            )
    for receipt in (
        provenance["pilot_checkpoint_receipts"] + provenance["gradient_receipts"]
    ):
        assert sha(local(receipt["path"])) == receipt["sha256"]
    init_receipt = next(
        r
        for r in provenance["pilot_checkpoint_receipts"]
        if local(r["path"]).name == "initialization.pt"
    )
    initialization = resume_payload(init_receipt)["models"]["initial"]
    param_names = [r["name"] for r in initial["parameter_order"]]
    buffer_names = [name for name in initialization if name not in param_names]
    initial["parameters_sha256"] = payload_hash(initialization, param_names)
    initial["buffers_sha256"] = payload_hash(initialization, buffer_names)
    initial["buffer_names"] = buffer_names
    initial["other_constructor_flags"] = {
        "num_symm_blocks": 1,
        "DFT": None,
        "nconstants_x": 3,
        "nconstants_c": 3,
        "note": "base defaults; Gc overrides correlation output count to 3; NN_models.py:183-202",
    }
    write("initialization", initial)
    resume_receipt = next(
        r
        for r in provenance["pilot_checkpoint_receipts"]
        if local(r["path"]).name == "resume.pt"
    )
    endpoint = resume_payload(resume_receipt)
    assert endpoint["counters"] == decision["updates"]
    moment_vectors = {}
    for arm in p["arms"]:
        assert (
            payload_hash(endpoint["models"][arm], buffer_names)
            == initial["buffers_sha256"]
        )
        opt = endpoint["optimizers"][arm]
        assert len(opt["param_groups"]) == 1
        assert all(int(state["step"]) == 32 for state in opt["state"].values())
        moment_vectors[arm] = {
            key: torch.cat(
                [
                    opt["state"][identifier][key].flatten().double()
                    for identifier in opt["param_groups"][0]["params"]
                ]
            ).numpy()
            for key in ["exp_avg", "exp_avg_sq"]
        }
        assert all(
            state["exp_avg"].dtype == torch.float32
            and state["exp_avg_sq"].dtype == torch.float32
            for state in opt["state"].values()
        )
        assert all(
            torch.isfinite(state[key]).all()
            for state in opt["state"].values()
            for key in ["step", "exp_avg", "exp_avg_sq"]
        )
    decision["moment_comparison"] = {
        key: gradient_metrics(
            moment_vectors["SUBSET"][key], moment_vectors["FULL_SOURCE"][key]
        )
        for key in ["exp_avg", "exp_avg_sq"]
    }
    with np.load(OUT / "initial_gradients.npz") as z:
        for arm in p["arms"]:
            gap = float(
                np.linalg.norm(
                    z[arm + "_system_gradients"].mean(0) - z[arm + "_gradient"]
                )
            )
            assert gap <= 1e-10 + 1e-9 * np.linalg.norm(z[arm + "_gradient"])
        for i, name in enumerate(p["systems"]):
            if not any(
                r["state"] == "theta0" and r["block"] == "system_" + name
                for r in geometry
            ):
                geometry.append(
                    {
                        "state": "theta0",
                        "block": "system_" + name,
                        **gradient_metrics(
                            z["SUBSET_system_gradients"][i],
                            z["FULL_SOURCE_system_gradients"][i],
                        ),
                    }
                )
    table("gradient_comparison", geometry)
    timing = {
        arm: sum(float(r["seconds"]) for r in logs if r["grid"] == arm)
        for arm in p["arms"]
    }
    source_points = sum(r["SOURCE_count"] for r in representations["records"].values())
    subset_points = sum(r["SUBSET_count"] for r in representations["records"].values())
    resource["training_batch_seconds_total"] = timing
    resource["training_time_ratio_FULL_over_SUBSET"] = (
        timing["FULL_SOURCE"] / timing["SUBSET"]
    )
    resource["grid_points_per_batch"] = {
        "SUBSET": subset_points,
        "FULL_SOURCE": source_points,
    }
    resource["grid_point_count_ratio"] = source_points / subset_points
    resource["optimization_presentations_per_arm"] = 96
    resource["measurement_scope"] = (
        "end-to-end batch durations include CPU/HDF5 transfers, CUDA computation, and native step; not isolated GPU kernel times"
    )
    write("resource_report", resource)
    decision["integrity"] = {
        "all_inputs_and_historical_sources_unchanged": True,
        "buffers_unchanged": True,
        "native_parameters_and_moments_F32": True,
        "updates_exactly_32_per_arm": True,
        "cross_table_2x2_at_each_milestone": True,
        "all_architecture_constraints_PASS": all(
            r["PASS"] for r in constraints.values()
        ),
        "all_finite": all(r["finite"] == "True" for r in cross),
        "new_checkpoint_hashes_PASS": True,
    }
    assert all(decision["integrity"].values())
    decision["wall_seconds"] = time.time() - p["first_action_unix"]
    decision["budget_respected"] = decision["wall_seconds"] < 10800
    decision["next_step"] = (
        "No automatic follow-up; if B, close selection-only intervention for this pilot and prioritize qualification of the large fresh normalization-block sensitivity rather than a full90 rebuild."
    )
    write("decision", decision)
    provenance["report_driver_sha256"] = sha(__file__)
    provenance["execution_environment"] = {
        "training_python": sys.version,
        "training_numpy": np.__version__,
        "training_torch": torch.__version__,
        "training_cuda": torch.version.cuda,
        "preparation_PySCF": "2.14.0 (existing WSL runtime)",
        "run_command_policy": "CUBLAS_WORKSPACE_CONFIG=:4096:8; OMP_NUM_THREADS=1; MKL_THREADING_LAYER=SEQUENTIAL",
        "driver_policy": "deterministic_algorithms=True; TF32 disabled for CUDA matmul and cuDNN",
    }
    provenance["compact_result_hashes"] = {
        name: sha(ROOT / (PREFIX + name + ".csv"))
        for name in ["gradient_comparison", "training_log", "cross_evaluation"]
    }
    write("provenance", provenance)
    source_initial = decision["SOURCE_initial_loss"]
    benefit = decision["B"]
    text = [
        "# Fresh R0 matched full-grid/subset training pilot",
        "",
        f"Classification {decision['classification']}. Both predeclared 32-update branches completed. Relative FULL_SOURCE benefit on the common SOURCE evaluation is B={benefit:.12g}; the 1% practical floor {'passes' if decision['effect_floor_PASS'] else 'does not pass'}. This is one seed and three training systems, not chemical generalization or an architecture-capacity test.",
        "",
        "## Initialization and fixed contract",
        "",
        f"Seed 2026 (Python/NumPy/Torch/CUDA), exact original constructor {initial['config']}, 9446 unique native-F32 trainable coordinates. One scientific constructor initialization, then deepcopy into independent SUBSET/FULL_SOURCE models and a synchronized F64 derivative carrier. Initial state hash `{initial['state_hash']}`; parameter/order/buffer receipts are in initialization.json. Old trained weights and optimizer states loaded: 0. New AdamW states empty at t0, identical single parameter-group policy; both native counters end at 32.",
        "",
        f"Frozen HEAD `{p['head']}`. GPU {resource['device']}, Torch {resource['torch_version']}. Preparation uses the already authenticated PySCF 2.14.0 source/basis contract. Original production files and historical artifacts remain hash-identical.",
        "",
        "## Representations",
        "",
        "|System|FULL_SOURCE points|SUBSET points|AO dimension|SOURCE / SUBSET reference replay error, AO Fro|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in representations["records"].items():
        text.append(
            f"|{name}|{r['SOURCE_count']}|{r['SUBSET_count']}|{r['n_AO']}|{r['reference_replay_gaps']}|"
        )
    text += [
        "",
        "Both representations use exact SOURCE F64 coordinates, weights, vxc, AO values/derivatives and descriptors. SUBSET uses the authenticated ordered index list, including repeated and zero-weight rows. No new cutoff or historical-mask reconstruction. Prediction and reference are assembled on the same grid in each case; legacy RefAO/rounded weights are not used. The reference replay is against prior source-precision R0/R2, not legacy R4.",
        "",
        "The learned forward/descriptors and parameter derivatives use F64 shadow coordinates synchronized from live native weights for every batch. The existing _model_double_operator_partials keeps its native F32 PBE rho/sigma/constants and unchanged regularization. AO assembly and gradient sums are F64. The mean gradient is cast once to native F32 immediately before AdamW. This common source-precision experiment is not a byte-identical production-corpus retraining.",
        "",
        "Loss = mean of the three fully assembled original weak-form matrix losses, ||S^(-1/2)(V−Vref)S^(-1/2)||F²/n_AO. No AE17, energy, relative chemistry, gauge shift, clipping, scheduler or coefficient calibration. Each gradient uses two bounded passes: full matrix first, then its one common matrix adjoint accumulated across chunks. The direct-autograd fixture and existing mixed-precision/operator regression tests cover this calculation; no chunk-square loss substitution.",
        "",
        "## Initial gradient sensitivity",
        "",
        "|Scope|norm SUBSET|norm FULL_SOURCE|absolute vector difference|relative difference|cosine|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in geometry:
        if r["state"] == "theta0":
            text.append(
                "|"
                + r["block"]
                + "|"
                + "|".join(
                    str(r[k])
                    for k in [
                        "norm_SUBSET",
                        "norm_FULL_SOURCE",
                        "absolute_difference",
                        "relative_difference",
                        "cosine",
                    ]
                )
                + "|"
            )
    text += [
        "",
        "Exchange/correlation/normalization ownership follows the authenticated F013 mapping: identify LayerNorm by module identity before assigning network prefixes, with no overlap or duplicate tied exchange parameters. Initial per-system vectors and the full gradient are saved outside Git with hashes. No arbitrary epsilon is used for zero norms; undefined metrics are blank in CSV.",
        "",
        f"Initial SOURCE mean loss {source_initial:.12g}. Large raw gradient norms, dominated initially by normalization coordinates, were disclosed and were not used as a scientific rejection gate. Same-grid copy gradient gap was {preflight['same_grid_gradient_gap']}; full source/subset difference is a separate scientific measurement.",
        "",
        "## Final t32 cross-evaluation",
        "",
        "|Trained on|Evaluated on|mean loss|H2|BeH2|CO|improvement from same-grid theta0|displacement norm|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cross:
        if r["t"] == "32":
            text.append(
                "|"
                + r["trained_grid"]
                + "|"
                + r["evaluated_grid"]
                + "|"
                + "|".join(
                    f"{float(r[k]):.12g}"
                    for k in [
                        "mean_loss",
                        "H2_loss",
                        "BeH2_loss",
                        "CO_loss",
                        "improvement_from_theta0",
                        "displacement_norm",
                    ]
                )
                + "|"
            )
    text += [
        "",
        "SOURCE is a shared numerical evaluation grid on the same three training systems, not independent validation/test data. t16 is diagnostic only; no endpoint selection or early stopping used. t0/t16/t32 system values and all updates are saved in cross_evaluation/training_log tables.",
        "",
        "## Endpoint fixed-weight gradient comparisons",
        "",
        "|Fixed endpoint|norm SUBSET|norm FULL_SOURCE|absolute vector difference|relative difference|cosine|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in geometry:
        if r["state"] != "theta0" and r["block"] == "full":
            text.append(
                "|"
                + r["state"]
                + "|"
                + "|".join(
                    str(r[k])
                    for k in [
                        "norm_SUBSET",
                        "norm_FULL_SOURCE",
                        "absolute_difference",
                        "relative_difference",
                        "cosine",
                    ]
                )
                + "|"
            )
    text += [
        "",
        "These gradients are recomputed at each live t32 weight vector on both grids; theta0 gradients are not substituted. Full block results and endpoint vectors are hash-bound.",
        "",
        f"Endpoint parameter distance ||theta_FULL−theta_SUBSET||={decision['endpoint_parameter_distance']:.12g}; displacement-vector cosine={decision['displacement_cosine']:.12g}. Native moment comparison: `{json.dumps(decision['moment_comparison'])}`. No inherited moments, optimizer reset or alternative optimizer.",
        "",
        "## Resource and integrity receipts",
        "",
        f"Complete-batch preflight times: {resource['complete_batch_seconds']} seconds. Projected remaining cost {resource['projected_remaining_seconds']:.3f}s, remaining budget {resource['remaining_wall_seconds']:.3f}s, reserve 900s; GO before updates. Preflight GPU allocator peak {resource['peak_GPU_bytes']} bytes (not a whole-run peak claim). SOURCE preparation {representations['prepare_seconds']:.3f}s.",
        "",
        f"Measured training batch totals: {timing}; FULL/SUBSET end-to-end ratio {resource['training_time_ratio_FULL_over_SUBSET']:.6g}. Grid points per update {resource['grid_points_per_batch']}; full grid count ratio {resource['grid_point_count_ratio']:.6g}. Every branch has 96 optimization system presentations, with two computational passes per gradient; full grid evaluates more points. Disk/indexing/warmup affect observed wall cost, so a grid-count ratio is not asserted to equal compute-time overhead.",
        "",
        f"Run timer (CPU/GPU/transfer/checkpoint work together) {decision['active_run_seconds']:.3f}s; scientific wall through final integrity {decision['wall_seconds']:.3f}s / 10800s. This is not isolated GPU kernel time. All source/hash, cache, counters, fresh-checkpoint receipts, buffers, constraints and finite checks passed. Safe restore of model/optimizer/RNG/cursors was tested on new hash-bound artifacts, with no old checkpoint load. Atomic resume snapshots represent completed paired updates; no scientific resume was needed.",
        "",
        f"Eight scientific checkpoint files: shared initialization, two t0, two t16, two t32 and one atomic resume. Focused optimizer fixtures are not scientific training arms or additional R0 optimization exposures. Validation receipt: {json.dumps(decision.get('validation', {}))}.",
        "",
        "## One next step and limits",
        "",
        "If the measured classification is B, close this selection-only intervention for the tested seed/horizon; the small gradient/trajectory differences do not support an automatic full90 rebuild. A separate bounded derivative/leverage qualification of the large fresh normalization-block sensitivity is the next justified diagnostic. If A, the next step is an independent limited qualification of the matched full-grid path. Neither result authorizes extra seeds, longer training, chemistry stages or architectural changes.",
        "",
        "Chemical generalization, Clean28, SCF stability, scientific promotion and a general explanation of historical NN plateau were not tested. No new systems, target labels, geometry, basis or density. Production corpus/loss/architecture unchanged.",
        "",
    ]
    (ROOT / (PREFIX + "report.md")).write_text("\n".join(text), encoding="utf8")


if __name__ == "__main__":
    main()
