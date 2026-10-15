"""F028 matched execution and reporting; no historical checkpoint loaders."""

import copy
import csv
import math
import time

import lap_f028_joint_eps as exp
import numpy as np
import torch


def norms(optimizer):
    return {
        k: float(
            torch.cat(
                [s[k].flatten().double() for s in optimizer.state.values()]
            ).norm()
        )
        if optimizer.state
        else 0.0
        for k in ("exp_avg", "exp_avg_sq")
    }


def log_tables(logs):
    rows = sorted(
        [r for arm_rows in logs.values() for r in arm_rows],
        key=lambda r: (r["update"], r["arm"]),
    )
    exp.table("training_log", rows)


def finish_pending_startup(arm, model, shadow, optimizer, logs, bundle, manifest):
    if not logs or logs[-1].get("post_step_scalar_status") != "PENDING":
        return
    row = logs[-1]
    entry = manifest[row["update"] - 1]
    after = exp.scalar_tasks(model, shadow, arm, bundle, entry)
    for task in exp.TASKS:
        row[task + "_finite_step_change"] = after[task] - row[task + "_loss"]
    row["weighted_finite_step_change"] = sum(
        exp.b8.LAMBDAS[t] * row[t + "_finite_step_change"] for t in exp.TASKS
    )
    row["post_step_scalar_status"] = "COMPLETE same original presentation before/after"
    exp.save(exp.OUT / f"{arm}_resume.pt", arm, model, optimizer, row["update"], logs)


def train():
    exp.deterministic()
    pre = exp.read(exp.file("preflight"))
    resource = exp.read(exp.file("resource_report"))
    assert pre["technical"] == "PASS" and resource["training_authorized"]
    assert pre["resource_training_authorization"] == "PASS"
    bundle = exp.TrainingBundle()
    manifest = exp.read(exp.OUT / "training_manifest.json")
    models, shadows, optimizers, counters, logs = {}, {}, {}, {}, {}
    geometry_rows = list(
        csv.DictReader(
            (exp.ROOT / (exp.PREFIX + "gradient_geometry.csv")).open(encoding="utf-8")
        )
    )
    constraint_checks = exp.read(exp.file("constraint_checks"))
    for arm in exp.ARMS:
        path = exp.OUT / f"{arm}_resume.pt"
        model, shadow, optimizer, saved = exp.restore(
            path if path.exists() else exp.OUT / f"{arm}_t0.pt", arm
        )
        models[arm], shadows[arm], optimizers[arm] = model, shadow, optimizer
        counters[arm], logs[arm] = saved["cursor"], saved["logs"]
        assert len(logs[arm]) == counters[arm]
        assert all(
            r["update"] == i + 1 and r["manifest_cursor"] == i
            for i, r in enumerate(logs[arm])
        )
        finish_pending_startup(
            arm, model, shadow, optimizer, logs[arm], bundle, manifest
        )
    assert counters["CONTROL"] in (
        counters["INTERVENTION"],
        counters["INTERVENTION"] + 1,
    )
    for cursor, entry in enumerate(manifest):
        for arm in exp.ARMS:
            if counters[arm] > cursor:
                continue
            assert counters[arm] == cursor
            assert counters["CONTROL"] in (
                counters["INTERVENTION"],
                counters["INTERVENTION"] + 1,
            )
            remaining = 502 - sum(counters.values())
            estimated_batch = max(pre["profiles"].values())
            recent = [r["seconds"] for rows in logs.values() for r in rows[-10:]]
            if recent:
                estimated_batch = max(estimated_batch, float(np.median(recent)))
            exp.budget(
                remaining * estimated_batch * 1.15
                + resource["endpoint_evaluation_reserve"]
                + 2500
            )
            model, shadow, optimizer = models[arm], shadows[arm], optimizers[arm]
            assert all(p.grad is None for p in model.parameters())
            tick = time.monotonic()
            losses, raw, combined, infos, _seconds = exp.joint(
                model, shadow, arm, bundle, entry
            )
            startup = (
                exp.geometry(
                    arm, cursor, entry, model, optimizer, raw, combined, losses
                )
                if cursor in (0, 1, 2)
                else None
            )
            before = exp.flatten(exp.core.named_trainable_parameters(model))
            exp.adamw_step(model, optimizer, raw, exp.b8.LAMBDAS)
            delta = exp.flatten(exp.core.named_trainable_parameters(model)) - before
            counters[arm] += 1
            assert {int(s["step"]) for s in optimizer.state.values()} == {cursor + 1}
            step_moments = norms(optimizer)
            row = {
                "arm": arm,
                "update": cursor + 1,
                "manifest_cursor": cursor,
                "counter": cursor + 1,
                "chemistry_presentations": 8,
                "ae17_presentations": 1,
                "mrks_presentations": 1,
                "chemistry_identity_variants": str(
                    [(i["identity"], i["variant"]) for i in infos]
                ),
                "ae17_identity": entry["ae17"]["identity"],
                "ae17_variant": entry["ae17"]["variant"],
                "mrks_id": entry["mrks_id"],
                **{t + "_loss": losses[t] for t in exp.TASKS},
                **{
                    t + "_raw_gradient_norm": float(exp.flatten(raw[t]).norm())
                    for t in exp.TASKS
                },
                "weighted_loss": sum(exp.b8.LAMBDAS[t] * losses[t] for t in exp.TASKS),
                "weighted_gradient_norm": float(exp.flatten(combined).norm()),
                "native_displacement_norm": float(delta.norm()),
                "gradient_dot_actual_displacement": float(
                    exp.flatten(combined) @ delta
                ),
                "seconds": time.monotonic() - tick,
                "first_moment_norm": step_moments["exp_avg"],
                "second_moment_norm": step_moments["exp_avg_sq"],
                "finite": True,
                "epsilon": "PASS",
                "post_step_scalar_status": "PENDING" if cursor < 2 else "NOT REQUIRED",
            }
            for task in exp.TASKS:
                row[task + "_directional_derivative_along_actual_displacement"] = float(
                    exp.flatten(raw[task]) @ delta
                )
            logs[arm].append(row)
            # Commit durable cursor immediately; pending scalar diagnostics resume without another step.
            exp.save(
                exp.OUT / f"{arm}_resume.pt",
                arm,
                model,
                optimizer,
                cursor + 1,
                logs[arm],
            )
            np.save(exp.OUT / f"{arm}_last_displacement.npy", delta.cpu().numpy())
            if cursor < 2:
                exp.save(
                    exp.OUT / f"{arm}_startup_t{cursor + 1}.pt",
                    arm,
                    model,
                    optimizer,
                    cursor + 1,
                    logs[arm],
                )
            if startup is not None:
                startup["native_displacement_norm"] = row["native_displacement_norm"]
                startup["weighted_gradient_dot_displacement"] = row[
                    "gradient_dot_actual_displacement"
                ]
                for task in exp.TASKS:
                    startup[task + "_raw_derivative_along_actual_displacement"] = row[
                        task + "_directional_derivative_along_actual_displacement"
                    ]
                geometry_rows = [
                    r
                    for r in geometry_rows
                    if not (r["arm"] == arm and int(r["theta"]) == cursor)
                ]
                geometry_rows.append(startup)
                exp.table("gradient_geometry", geometry_rows)
                np.savez(
                    exp.OUT / f"{arm}_startup_gradients_theta{cursor}.npz",
                    **{t: exp.flatten(raw[t]).cpu().numpy() for t in exp.TASKS},
                )
            finish_pending_startup(
                arm, model, shadow, optimizer, logs[arm], bundle, manifest
            )
            if cursor + 1 in exp.MILESTONES:
                constraint_checks[arm + f"_t{cursor + 1}"] = exp.pilot.constraints(
                    model
                )
                assert constraint_checks[arm + f"_t{cursor + 1}"]["PASS"]
                exp.save(
                    exp.OUT / f"{arm}_t{cursor + 1}.pt",
                    arm,
                    model,
                    optimizer,
                    cursor + 1,
                    logs[arm],
                )
                exp.write("constraint_checks", constraint_checks)
            log_tables(logs)
            exp.write(
                "progress",
                {
                    "counters": counters,
                    "presentations": {
                        a: {
                            "chemistry": 8 * n,
                            "ae17": n,
                            "mrks": n,
                            "exc": n,
                            "operator": n,
                        }
                        for a, n in counters.items()
                    },
                    "workflow_wall": exp.elapsed(),
                    "retained_updates": sum(counters.values()),
                },
            )
            print(
                "F028_UPDATE",
                arm,
                cursor + 1,
                "losses",
                losses,
                "seconds",
                row["seconds"],
                "workflow",
                exp.elapsed(),
                flush=True,
            )
            del raw, combined, delta, before
            torch.cuda.empty_cache()
    assert counters == {"CONTROL": 251, "INTERVENTION": 251}
    # Required endpoint startup-sensitivity snapshot, without applying update252.
    for arm in exp.ARMS:
        losses, raw, combined, _infos, _seconds = exp.joint(
            models[arm], shadows[arm], arm, bundle, manifest[-1]
        )
        diag = exp.geometry(
            arm, 251, manifest[-1], models[arm], optimizers[arm], raw, combined, losses
        )
        diag["last_actual_native_displacement_norm"] = logs[arm][-1][
            "native_displacement_norm"
        ]
        diag["endpoint_gradient_is_proposal_only"] = True
        geometry_rows = [
            r for r in geometry_rows if not (r["arm"] == arm and int(r["theta"]) == 251)
        ]
        geometry_rows.append(diag)
        np.savez(
            exp.OUT / f"{arm}_endpoint_gradients_theta251.npz",
            **{t: exp.flatten(raw[t]).cpu().numpy() for t in exp.TASKS},
        )
        exp.table("gradient_geometry", geometry_rows)
    exp.write(
        "training_completion",
        {
            "counters": counters,
            "retained_updates": 502,
            "discarded_profiling_steps": 2,
            "no_updates_after251": True,
        },
    )
    exp.write(
        "training_source_receipt",
        {
            "verified_files": {
                name: bundle.checksums[name] for name in bundle.handles.verified
            },
            "mode": "read-only HDF5; checked against qualified manifest before first access",
        },
    )
    bundle.close()


def endpoint():
    assert exp.read(exp.file("training_completion"))["counters"] == {
        "CONTROL": 251,
        "INTERVENTION": 251,
    }
    for arm in exp.ARMS:
        if not (exp.OUT / f"{arm}_evaluation_t251/metrics.json").exists():
            exp.evaluation(arm, 251)


def clean28(metrics):
    # This is the ONLY branch permitted to instantiate development metadata.
    gate = {
        arm: {
            "eligible": m["eligible"],
            "ratios": m["ratios"],
            "checkpoint_sha256": exp.b8.sha256_file(exp.OUT / f"{arm}_t251.pt"),
        }
        for arm, m in metrics.items()
    }
    exp.write("eligibility", gate)
    if exp.file("clean28_receipt").exists():
        result = exp.read(exp.file("clean28_receipt"))
        assert result["gate"] == gate
    else:
        result = {"gate": gate, "values": {}, "future_test_access": False}
    for arm, m in metrics.items():
        if arm in result["values"]:
            assert result["values"][arm]["status"] != "IN_PROGRESS", (
                "Interrupted Clean28: do not evaluate again"
            )
            continue
        if not m["eligible"]:
            result["values"][arm] = {
                "status": "NOT EVALUATED",
                "reason": "original P536 eligibility failed",
                "evaluations": 0,
            }
            continue
        exp.budget(1200)
        result["values"][arm] = {"status": "IN_PROGRESS", "evaluations": 1}
        exp.write("clean28_receipt", result)
        from evaluate_microbatch_endpoint import validation

        model, shadow, _optimizer, _payload = exp.restore(
            exp.OUT / f"{arm}_t251.pt", arm
        )
        before = exp.scientific_identity(model, arm)
        exp.eps_helpers.assert_epsilon(shadow, exp.ARMS[arm])
        bundle = exp.PublicationDataset(exp.DATA)
        values = validation(model, bundle)
        assert exp.scientific_identity(model, arm) == before
        result["values"][arm] = {
            "status": "EVALUATED",
            "evaluations": 1,
            "clean28": values["clean28"],
            "historical_best_beaten": values["clean28"] < 8.619694171538775,
            "below7": values["clean28"] < 7,
            "reaction_rows": [r for r in values["reaction_rows"] if r["clean"]],
            "full30_used_for_selection": False,
        }
        assert len(result["values"][arm]["reaction_rows"]) == 28
        bundle.close()
        exp.write("clean28_receipt", result)
    exp.write("clean28_receipt", result)
    return result


def finish():
    completion = exp.read(exp.file("training_completion"))
    assert completion["retained_updates"] == 502
    metrics, operator, exc, atoms, reactions, databases = {}, [], [], [], [], []
    initial = {}
    for arm in exp.ARMS:
        for cursor in (0, 251):
            folder = exp.OUT / f"{arm}_evaluation_t{cursor}"
            m = exp.read(folder / "metrics.json")
            if cursor == 251:
                metrics[arm] = m
            else:
                initial[arm] = m
            for identity, record in exp.read(folder / "physical.json").items():
                operator.append(
                    {
                        "arm": arm,
                        "cursor": cursor,
                        "identity": identity,
                        "name": record["name"],
                        "loss": record["op"],
                    }
                )
                exc.append(
                    {"arm": arm, "cursor": cursor, "identity": identity, **record}
                )
            atoms += [
                {"arm": arm, "cursor": cursor, **r}
                for r in exp.read(folder / "atoms.json")
            ]
            reactions += [
                {"arm": arm, "cursor": cursor, **r}
                for r in exp.read(folder / "reactions.json")
            ]
            databases += exp.read(folder / "database_metrics.json")
    # Exact identity joins; preserve E251 variants instead of matching by energy.
    op_endpoint = {
        a: {r["identity"]: r for r in operator if r["arm"] == a and r["cursor"] == 251}
        for a in exp.ARMS
    }
    per_system = []
    for identity, c in op_endpoint["CONTROL"].items():
        i = op_endpoint["INTERVENTION"][identity]
        per_system.append(
            {
                "identity": identity,
                "name": c["name"],
                "control": c["loss"],
                "intervention": i["loss"],
                "delta": i["loss"] - c["loss"],
            }
        )
    op_deltas = {r["identity"]: r for r in per_system}
    for row in operator:
        if row["cursor"] == 251:
            row["matched_delta_intervention_minus_control"] = op_deltas[
                row["identity"]
            ]["delta"]
    cr = {(r["arm"], r["identity"]): r for r in reactions if r["cursor"] == 251}
    for row in reactions:
        if row["cursor"] == 251:
            c, i = cr["CONTROL", row["identity"]], cr["INTERVENTION", row["identity"]]
            assert c["variant"] == i["variant"]
            row["matched_absolute_error_delta"] = abs(i["signed_error_kcal"]) - abs(
                c["signed_error_kcal"]
            )
    exp.table(
        "checkpoint_metrics",
        [
            {k: v for k, v in m.items() if not isinstance(v, dict)}
            for m in [*initial.values(), *metrics.values()]
        ],
    )
    exp.table("full90_operator", operator)
    exp.table("exc_metrics", exc)
    exp.table("ae17_metrics", atoms)
    exp.table("chemistry_reactions", reactions)
    exp.table("chemistry_databases", databases)
    exp.write(
        "hydrogen",
        {
            a: {"t0": initial[a]["hydrogen"], "t251": metrics[a]["hydrogen"]}
            for a in exp.ARMS
        },
    )
    clean = clean28(metrics)
    control, intervention = metrics["CONTROL"], metrics["INTERVENTION"]
    assert math.isfinite(control["op"]) and control["op"] > 0
    benefit = (control["op"] - intervention["op"]) / control["op"]
    chemistry_delta = {
        k: intervention[k] - control[k]
        for k in ("JW", "legacy", "MAE", "RMSE", "macro_database_MAE")
    }
    resolved_cost = {
        k: v for k, v in chemistry_delta.items() if v > 1e-6 * max(1, abs(control[k]))
    }
    outcome = "C" if benefit < 0.01 else "B" if resolved_cost else "A"
    promotion = "NOT ELIGIBLE"
    if benefit >= 0.01 and intervention["eligible"]:
        promotion = "D"
        if clean["values"]["INTERVENTION"].get("historical_best_beaten"):
            promotion = "E"
    swaps = []
    if (
        not exp.file("optional_swaps").exists()
        and exp.elapsed()
        + 2
        * sum(
            r["seconds"]
            for r in exp.read(
                exp.OUT / "CONTROL_evaluation_t251/physical.json"
            ).values()
        )
        + 1800
        < exp.LIMIT
    ):
        bundle = exp.TrainingBundle()
        for arm in exp.ARMS:
            model, shadow, _optimizer, _payload = exp.restore(
                exp.OUT / f"{arm}_t251.pt", arm
            )
            other = "INTERVENTION" if arm == "CONTROL" else "CONTROL"
            exp.eps_helpers.configure_epsilon(model, exp.ARMS[other])
            shadow = copy.deepcopy(model).double()
            exp.eps_helpers.assert_epsilon(shadow, exp.ARMS[other])
            values = {}
            for identity in sorted(bundle.systems):
                exp.budget(1800)
                values[identity] = exp.operator_value_gradient(
                    shadow, bundle, identity, False
                )[0]
            swaps.append(
                {
                    "weights": arm,
                    "epsilon": exp.ARMS[other],
                    "operator_mean": float(np.mean(list(values.values()))),
                    "per_system": values,
                }
            )
            exp.write("optional_swaps", {"status": "IN PROGRESS", "values": swaps})
        bundle.close()
        exp.write(
            "optional_swaps",
            {"status": "PERFORMED", "values": swaps, "used_for_selection": False},
        )
    elif not exp.file("optional_swaps").exists():
        exp.write(
            "optional_swaps", {"status": "NOT RUN", "reason": "final integrity reserve"}
        )
    protocol = exp.read(exp.file("protocol"))
    for path, expected in protocol["source_hashes"].items():
        assert exp.b8.sha256_file(exp.ROOT / path) == expected, path
    verified = exp.read(exp.file("training_source_receipt"))["verified_files"]
    for arm in exp.ARMS:
        for cursor in (0, 251):
            verified.update(
                exp.read(exp.OUT / f"{arm}_evaluation_t{cursor}/receipt.json").get(
                    "verified_training_data_files", {}
                )
            )
    for path, expected in verified.items():
        exp.budget(1200)
        assert exp.b8.sha256_file(exp.DATA / path) == expected, path
    exp.write(
        "final_data_integrity",
        {
            "PASS": True,
            "files_rehashed": len(verified),
            "training_sources_modified": False,
        },
    )
    manifests, checks = {}, exp.read(exp.file("constraint_checks"))
    for arm in exp.ARMS:
        for cursor in exp.MILESTONES:
            model, shadow, _optimizer, payload = exp.restore(
                exp.OUT / f"{arm}_t{cursor}.pt", arm
            )
            check = exp.pilot.constraints(model)
            assert check["PASS"]
            checks[arm + f"_restored_t{cursor}"] = check
            manifests[arm + f"_t{cursor}"] = {
                "identity": payload["identity"],
                "checkpoint_sha256": exp.b8.sha256_file(
                    exp.OUT / f"{arm}_t{cursor}.pt"
                ),
            }
    exp.write("epsilon_manifests", manifests)
    exp.write("constraint_checks", checks)
    provenance = exp.read(exp.file("provenance"))
    provenance["large_artifact_receipts"] = [
        {"path": str(p), "sha256": exp.b8.sha256_file(p)}
        for p in exp.OUT.rglob("*")
        if p.is_file()
    ]
    provenance["drivers"] = {
        p: exp.b8.sha256_file(exp.ROOT / p)
        for p in ("tools/lap_f028_joint_eps.py", "tools/lap_f028_joint_eps_run.py")
    }
    provenance["historical_production_sources_unchanged"] = True
    exp.write("provenance", provenance)
    decision = {
        "experiment": "F028",
        "technical": "PASS",
        "operator_transfer_outcome": outcome,
        "promotion": promotion,
        "benefit": benefit,
        "effect_floor_pass": benefit >= 0.01,
        "chemistry_deltas": chemistry_delta,
        "resolved_chemistry_costs": resolved_cost,
        "control": control,
        "intervention": intervention,
        "operator_system_improved": sum(r["delta"] < 0 for r in per_system),
        "operator_system_regressed": sum(r["delta"] > 0 for r in per_system),
        "largest_improvements": sorted(per_system, key=lambda r: r["delta"])[:10],
        "largest_regressions": sorted(
            per_system, key=lambda r: r["delta"], reverse=True
        )[:10],
        "clean28": clean,
        "scientific_wall": exp.elapsed(),
        "budget_respected": exp.elapsed() < exp.LIMIT,
        "next_recommendation": "Review the measured operator/chemistry tradeoff and separately freeze the next qualification; do not continue, retune epsilon or weights automatically.",
    }
    exp.write("decision", decision)
    resource = exp.read(exp.file("resource_report"))
    resource.update(
        {
            "scientific_wall": exp.elapsed(),
            "budget_respected": exp.elapsed() < exp.LIMIT,
            "timings": {
                p.name: exp.read(p) for p in exp.ROOT.glob(exp.PREFIX + "*timing.json")
            },
        }
    )
    exp.write("resource_report", resource)
    db_table = "\n".join(
        f"|{d['database']}|{d['MAE']:.8g}|{next(i for i in databases if i['arm'] == 'INTERVENTION' and i['cursor'] == 251 and i['database'] == d['database'])['MAE']:.8g}|"
        for d in databases
        if d["arm"] == "CONTROL" and d["cursor"] == 251
    )
    metrics_table = "\n".join(
        f"|{k}|{control[k]:.12g}|{intervention[k]:.12g}|{intervention[k] - control[k]:+.8g}|"
        for k in (
            "op",
            "operator_median",
            "operator_max",
            "JW",
            "legacy",
            "MAE",
            "RMSE",
            "median_absolute_error",
            "macro_database_MAE",
            "ae17",
            "exc",
        )
    )
    text = f"""# F028 — matched fresh four-objective epsilon qualification

Technical PASS. Operator-transfer outcome {outcome}; promotion status {promotion}. Both arms completed exactly 251 updates from identical one-time seed2028 tensors and empty independent AdamW states. Full90 matched benefit B_op={benefit:.12g}; predeclared 1% floor {benefit >= 0.01}. Chemistry is reported separately, not inferred from operator progress.

Original R0 9446 coordinates, Gx/Gc enabled; five exchange LN eps1e-5 versus1e-3, correlation eps1e-5. The same tensors define different functions at t0. Every native/shadow/restored epsilon is asserted. Original corpus, targets, gauge, density, quadrature and dispersion retained; no FULL_SOURCE rebuild. Original F010 c010 and four lambdas frozen, no recalibration, /4, per-electron weights, scheduler, clipping or curriculum.

Matched B8/J251 eight-offset stream0..250: each arm 2008 reaction,251 AE17,251 paired mRKS/Exc/operator presentations. Full scientific evaluations use E251/E17/full90 at t0/t251; intermediate checkpoints cannot select endpoints. Startup minibatch gradients remain labelled separately from full-population scalar metrics. Two authorized profiling steps were discarded with model/optimizer/RNG restoration;502 retained steps, no update252.

## Original endpoint metrics

|Metric|CONTROL|INTERVENTION|I-C|
|---|---:|---:|---:|
{metrics_table}

|Database|CONTROL MAE|INTERVENTION MAE|
|---|---:|---:|
{db_table}

Full90 per-system improved/regressed: {decision["operator_system_improved"]}/{decision["operator_system_regressed"]}. Complete90 rows and largest contributions: full90_operator.csv/decision.json. Exc per-system residuals, E17 signed/absolute atom errors, exact H references/variants, all E251 reaction deltas and database MAE/RMSE are in their corresponding CSVs. All original four P536 ratios: CONTROL {control["ratios"]}, INTERVENTION {intervention["ratios"]}. Eligibility: {control["eligible"]}/{intervention["eligible"]}. H absolute errors: {control["hydrogen"]["absolute_error_kcal"]:.12g}/{intervention["hydrogen"]["absolute_error_kcal"]:.12g} kcal/mol; signed values and source targets are retained in hydrogen.json. No analytic H replacement or self-interaction inference.

Clean28 gate receipt: { {a: {k: v for k, v in r.items() if k != "reaction_rows"} for a, r in clean["values"].items()} }. Only eligible frozen t251 may be evaluated once. Clean28 is historical development evidence, not an untouched test set. Future/test access:NO.

## Integrity, precision and interpretation

Task-specific original boundaries are explicit in protocol.json: F010 chemistry/AE17 F64 shadow and integration, original nativeF32 Exc, F64 learned/F32 PBE operator; corpus AO/feature F32 rounding precedes exact widening. Four gradients are summed F64, cast once per native coordinate, one AdamW step. Bounded two-pass matrix adjoint assembles the complete matrix before squaring. Preflight original scalar/gradient sum, small direct/adjoint and independent energy-first parity, constraints, zero-fill/order and restoration pass. No trained historical NN or inherited moments loaded. Exact scientific identities, counter/RNG/source hashes and large receipts live in provenance/epsilon manifests. Historical files and production code unchanged.

F026/F027 operator-only within-seed benefits22.8141%/18.7999% do not establish joint transfer. This experiment's full90 effect is {benefit:.8%}; it neither establishes chemical generalization nor proves the full physics→chemistry strategy. Resolved chemistry changes are {chemistry_delta}; they are not hidden in a composite score. Ratio eligibility rules remain unchanged. Optional fixed-weight epsilon swaps are explanatory only, never causal percentages or model-selection rules.

Next action: {decision["next_recommendation"]}

Scientific wall {exp.elapsed():.3f}s /72000s. Resource report separates measured execution wall from workflow wall; GPU-kernel-active time was not separately metered. No extra seed, epsilon, loss, training arm, SCF or automatic repair. Tests/Ruff/compileall/diff receipts are finalized separately.
"""
    (exp.ROOT / (exp.PREFIX + "report.md")).write_text(
        text, encoding="utf-8", newline="\n"
    )
