"""F029: LR-only continuation; reuse qualified F028 numerical pathways."""

import argparse
import copy
import csv
import hashlib
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import lap_f028_joint_eps as base
import numpy as np
import torch

ROOT = base.ROOT
OUT = ROOT.parent / "lap_f029_lr_continuation_runs_20261015"
PREFIX = "lap_f029_lr_continuation_"
SOURCE = base.OUT / "INTERVENTION_t251.pt"
SOURCE_SHA = "49ccd8da8194cb12a44829e524973c90a5fe11079a82c3e1938e57fb2a067d1e"
ANCESTOR = "b272860ef807fa128064722655ec6fcf79e83226"
START = datetime.fromisoformat("2026-10-15T14:47:20+00:00").timestamp()
LIMIT = 7200
LRS = {"CONTROL": 2.5e-5, "LOW_LR": 2.5e-6}
MILESTONES = (0, 1, 2, 16, 32)
SELECTED = (1, 2, 32)
ORIGINAL_RESTORE = base.restore
_COMMON = None


def file(name):
    return ROOT / (PREFIX + name + ".json")


def read(name):
    return base.read(file(name))


def write(name, value):
    base.b8.write_json(file(name), value)


def table(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def elapsed():
    return time.time() - START


def budget(reserve=0):
    if elapsed() + reserve >= LIMIT:
        raise TimeoutError("F029 hard budget/reserve exhausted")


def stop(reason):
    write("source_failure", {"reason": str(reason), "classification": "E"})
    raise RuntimeError(reason)


# F028's driver-specific 5000/1800s reserves do not apply to this 7200s run.
# F029 outer phases enforce the measured remaining-work reserve instead.
base.budget = lambda _historical_reserve=0: budget()
for helper in (base.b8, base.f010, base.chemistry):
    helper._stop = stop


def digest(value):
    """Exact tensor-aware hash, independent of torch serialization metadata."""
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())

    visit(value)
    return h.hexdigest()


def identity(model):
    return base.scientific_identity(model, "INTERVENTION")


def state_checks(model, shadow, optimizer, global_cursor, lr):
    base.eps_helpers.assert_epsilon(model, 1e-3)
    base.eps_helpers.assert_epsilon(shadow, 1e-3)
    params = base.core.named_trainable_parameters(model)
    assert sum(p.numel() for p in params.values()) == 9446
    assert tuple(params) == tuple(base.core.named_trainable_parameters(shadow))
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == lr and group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0.01
    assert group["foreach"] is False
    assert all(a is b for a, b in zip(group["params"], params.values(), strict=True))
    assert len(optimizer.state) == len(params)
    assert {int(s["step"]) for s in optimizer.state.values()} == {global_cursor}
    for p in params.values():
        assert p.dtype == torch.float32 and torch.isfinite(p).all()
        for k in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[p][k].dtype == torch.float32
            assert torch.isfinite(optimizer.state[p][k]).all()
        assert (optimizer.state[p]["exp_avg_sq"] >= 0).all()


def common():
    global _COMMON
    if _COMMON is None:
        assert base.b8.sha256_file(SOURCE) == SOURCE_SHA
        model, shadow, optimizer, payload = ORIGINAL_RESTORE(SOURCE, "INTERVENTION")
        assert payload["cursor"] == 251 and len(payload["logs"]) == 251
        state_checks(model, shadow, optimizer, 251, 2.5e-5)
        assert identity(model) == payload["identity"]
        _COMMON = (model, shadow, optimizer, payload)
    return _COMMON


def clone(arm):
    # Deep-copy the tuple together: optimizer references follow the copied model.
    model, shadow, optimizer, original = copy.deepcopy(common())
    optimizer.param_groups[0]["lr"] = LRS[arm]
    state_checks(model, shadow, optimizer, 251, LRS[arm])
    base.pilot.restore_rng(original["RNG"])
    return model, shadow, optimizer


def normalized_optimizer(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for group in state["param_groups"]:
        del group["lr"]
    return state


def independent(a, b):
    for x, y in zip(a[0].parameters(), b[0].parameters(), strict=True):
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    for x, y in zip(a[0].buffers(), b[0].buffers(), strict=True):
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    assert identity(a[0]) == identity(b[0])
    assert digest(normalized_optimizer(a[2])) == digest(normalized_optimizer(b[2]))
    for x, y in zip(a[2].state.values(), b[2].state.values(), strict=True):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert (
                torch.equal(x[key], y[key]) and x[key].data_ptr() != y[key].data_ptr()
            )


def stream():
    protocol = read("protocol")
    assert base.b8.sha256_file(Path(__file__)) == protocol["driver_sha256"]
    assert base.b8.sha256_file(file("stream_manifest")) == protocol["stream_sha256"]
    rows = read("stream_manifest")["rows"]
    assert len(rows) == 32
    return rows


def save(arm, model, shadow, optimizer, local, logs, path=None):
    stream()
    state_checks(model, shadow, optimizer, 251 + local, LRS[arm])
    path = path or OUT / f"{arm}_t{251 + local}.pt"
    payload = {
        "experiment": "F029",
        "arm": arm,
        "identity": identity(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "local_cursor": local,
        "global_cursor": 251 + local,
        "logs": logs,
        "RNG": base.pilot.rng(),
        "source_checkpoint_sha256": SOURCE_SHA,
        "protocol_sha256": base.b8.sha256_file(file("protocol")),
        "stream_sha256": base.b8.sha256_file(file("stream_manifest")),
        "coefficients": read("protocol")["coefficients"],
        "source_verification_sha256": base.b8.sha256_file(file("source_verification")),
    }
    base.atomic_save(path, payload)
    receipts = (
        read("checkpoint_receipts") if file("checkpoint_receipts").exists() else {}
    )
    receipts[str(path)] = base.b8.sha256_file(path)
    write("checkpoint_receipts", receipts)


def restore(path, arm):
    stream()
    assert base.b8.sha256_file(path) == read("checkpoint_receipts")[str(path)]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["experiment"] == "F029" and payload["arm"] == arm
    assert payload["source_checkpoint_sha256"] == SOURCE_SHA
    assert payload["protocol_sha256"] == base.b8.sha256_file(file("protocol"))
    assert payload["stream_sha256"] == base.b8.sha256_file(file("stream_manifest"))
    assert payload["source_verification_sha256"] == base.b8.sha256_file(
        file("source_verification")
    )
    assert payload["coefficients"] == read("protocol")["coefficients"]
    model, shadow, optimizer = clone(arm)
    model.load_state_dict(payload["model"], strict=True)
    assert identity(model) == payload["identity"]
    base.synchronize(model, shadow, "INTERVENTION")
    optimizer.load_state_dict(payload["optimizer"])
    assert payload["global_cursor"] == 251 + payload["local_cursor"]
    assert len(payload["logs"]) == payload["local_cursor"]
    state_checks(model, shadow, optimizer, payload["global_cursor"], LRS[arm])
    assert payload["RNG"][2].device.type == "cpu"
    base.pilot.restore_rng(payload["RNG"])
    return model, shadow, optimizer, payload


def prepare():
    assert not file("protocol").exists(), "Never overwrite a frozen F029 protocol"
    OUT.mkdir(exist_ok=True)
    base.deterministic()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ANCESTOR, "HEAD"], cwd=ROOT, check=True
    )
    assert (
        subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip()
        == "lap_full_vxc"
    )
    historical_names = (
        "provenance.json",
        "protocol.json",
        "report.md",
        "training_log.csv",
        "checkpoint_metrics.csv",
        "gradient_geometry.csv",
        "eligibility.json",
        "hydrogen.json",
        "epsilon_manifests.json",
    )
    paths = [ROOT / (base.PREFIX + n) for n in historical_names]
    paths += [
        ROOT / "tools/lap_f028_joint_eps.py",
        ROOT / "tools/lap_f028_joint_eps_run.py",
    ]
    subprocess.run(
        ["git", "diff", "--exit-code", ANCESTOR, "--", *map(str, paths)],
        cwd=ROOT,
        check=True,
    )
    historical_hashes = {str(p): base.b8.sha256_file(p) for p in paths}
    original_protocol = base.read(base.file("protocol"))
    provenance = base.read(base.file("provenance"))
    for name, expected in {
        **original_protocol["source_hashes"],
        **provenance["drivers"],
    }.items():
        assert base.b8.sha256_file(ROOT / name) == expected, name
    assert base.b8.sha256_file(SOURCE) == SOURCE_SHA
    manifests = base.read(base.file("epsilon_manifests"))
    checkpoint_hashes = {
        t: manifests[f"INTERVENTION_t{t}"]["checkpoint_sha256"] for t in (128, 192, 251)
    }
    for t, expected in checkpoint_hashes.items():
        assert base.b8.sha256_file(base.OUT / f"INTERVENTION_t{t}.pt") == expected
    full = base.read(base.OUT / "training_manifest.json")
    assert (
        base.b8.sha256_file(base.OUT / "training_manifest.json")
        == base.f010.continuation.MANIFEST_SHA
    )
    assert full == base.b8.build_batches(base.b8.load_j251_rows())
    write(
        "stream_manifest",
        {
            "definition": "Prospectively defined next-cycle prefix; not a historical post251 receipt",
            "parent_sha256": base.f010.continuation.MANIFEST_SHA,
            "rows": full[:32],
        },
    )
    write(
        "source_verification",
        {
            "F028_artifacts": historical_hashes,
            "historical_checkpoint_hashes": checkpoint_hashes,
            "original_contract": base.read(base.file("source_verification")),
            "source_code": original_protocol["source_hashes"],
            "driver": {p: base.b8.sha256_file(ROOT / p) for p in provenance["drivers"]},
        },
    )
    scales = base.read(ROOT / "lap_f010_normalization_receipt.json")["scales"]
    definition_bundle = base.TrainingBundle()
    specs = base.chemistry._selected(definition_bundle)
    definition_bundle.close()
    jw_uncertainty = original_protocol["tolerances"]["energy_kcal_absolute"] * float(
        np.mean([1 / scales[s["database"]] for s in specs])
    )
    write(
        "protocol",
        {
            "experiment": "F029",
            "ancestor": ANCESTOR,
            "start_unix": START,
            "wall_limit": LIMIT,
            "wall_origin": "2026-10-15T14:47:20Z, conservative earlier bound preceding first attachment read",
            "source_sha256": SOURCE_SHA,
            "lrs": LRS,
            "updates_per_arm": 32,
            "global_start": 251,
            "global_end": 283,
            "milestones_local": MILESTONES,
            "selected_finite_steps_global": [252, 253, 283],
            "order": "CONTROL then LOW_LR at each local cursor0..31",
            "coefficients": original_protocol["coefficients"],
            "precision": original_protocol["precision"],
            "stream_sha256": base.b8.sha256_file(file("stream_manifest")),
            "tolerances": original_protocol["tolerances"],
            "evaluation_uncertainty": {
                "JW_absolute_per_evaluation": jw_uncertainty,
                "operator_relative_per_evaluation": original_protocol["tolerances"][
                    "physical_relative"
                ],
                "operator_absolute_floor": 1e-12,
                "pairwise": "sum of both individual bounds",
            },
            "history": [251, 128, 192],
            "evaluation": "Unchanged F028 evaluator E251/E17/full90; no intermediate selection",
            "clean28": "Only t283 after both endpoints frozen and all original P536 gates pass, once per eligible arm",
            "no_new_initialization": True,
            "driver_sha256": base.b8.sha256_file(Path(__file__)),
        },
    )
    a, b = clone("CONTROL"), clone("LOW_LR")
    independent(a, b)
    write(
        "common_state",
        {
            "source_sha256": SOURCE_SHA,
            "identity": identity(a[0]),
            "optimizer_without_lr_hash": digest(normalized_optimizer(a[2])),
            "original_optimizer_hash": digest(common()[2].state_dict()),
            "RNG_hash": digest(common()[3]["RNG"]),
            "counter": 251,
            "starting_parameter_identity": "PASS",
            "starting_moment_identity": "PASS",
            "independent_storage": "PASS",
            "optimizer_only_difference": "lr",
        },
    )
    for arm, state in zip(LRS, (a, b), strict=True):
        save(arm, *state, 0, [])
    bundle = base.TrainingBundle()
    entry = stream()[0]
    gradients, profiles, checks = {}, {}, {}
    for arm, (model, shadow, optimizer) in zip(LRS, (a, b), strict=True):
        before = (identity(model), digest(optimizer.state_dict()))
        rng = base.pilot.rng()
        tick = time.monotonic()
        losses, raw, combined, _infos, _seconds = base.joint(
            model, shadow, "INTERVENTION", bundle, entry
        )
        profiles[arm] = time.monotonic() - tick
        gradients[arm] = {t: base.flatten(raw[t]).cpu().numpy() for t in base.TASKS}
        np.savez(OUT / f"{arm}_first_gradients.npz", **gradients[arm])
        checks[arm] = {
            "losses": losses,
            "gradient_norm": float(base.flatten(combined).norm()),
            "constraints": base.pilot.constraints(model),
            "optimizer_hash": digest(optimizer.state_dict()),
        }
        assert checks[arm]["constraints"]["PASS"]
        assert before == (identity(model), digest(optimizer.state_dict()))
        base.pilot.restore_rng(rng)
        assert digest(rng) == digest(base.pilot.rng())
    for t in base.TASKS:
        np.testing.assert_allclose(
            gradients["CONTROL"][t], gradients["LOW_LR"][t], atol=1e-10, rtol=1e-9
        )
    write(
        "preflight",
        {
            "status": "PASS",
            "checks": checks,
            "profiles": profiles,
            "initial_raw_task_gradient_parity": "PASS",
            "molecular_optimizer_steps": 0,
            "RNG_restored": True,
            "F64_gradient_sum": "PASS",
            "source_files_verified": list(bundle.handles.verified),
        },
    )
    bundle.close()
    write(
        "provenance",
        {
            "F028_artifacts": historical_hashes,
            "source_checkpoint_sha256": SOURCE_SHA,
            "starting_checkpoint": str(SOURCE),
            "F028_control_loaded": False,
            "new_initialization": False,
            "allocation": "Deepcopy authenticated F028 architecture template followed by exact source restore; no constructor/RNG initialization",
            "environment": {"torch": torch.__version__, "numpy": np.__version__},
        },
    )
    estimate(3)


def estimate(history_remaining):
    profiles = read("preflight")["profiles"]
    old = base.read(base.file("resource_report"))
    evaluation_cost = max(old["full_population_t0_evaluation_seconds"].values())
    measured = [
        base.read(p)["seconds"]
        for p in OUT.glob("historical_t*/INTERVENTION_evaluation_t*/metrics.json")
    ]
    if measured:
        evaluation_cost = max(evaluation_cost, max(measured))
    training_cost = 64 * max(profiles.values()) * 1.2
    final_cost = 2 * evaluation_cost * 1.2
    history_cost = history_remaining * evaluation_cost * 1.2
    overhead = (
        800  # Same-presentation diagnostics, tests, conditional Clean28, integrity.
    )
    projected = training_cost + final_cost + history_cost + overhead
    fits = elapsed() + projected < LIMIT
    write(
        "resource_report",
        {
            "limit": LIMIT,
            "elapsed": elapsed(),
            "profiles": profiles,
            "evaluation_profile_source": "Authenticated F028 complete original-panel measurements; refreshed by F029 historical panels",
            "evaluation_cost": evaluation_cost,
            "training_projection": training_cost,
            "history_remaining_projection": history_cost,
            "endpoint_reserve": final_cost,
            "integrity_diagnostics_clean28_reserve": overhead,
            "projected_remaining": projected,
            "training_authorized": fits,
            "GPU_kernel_active_time": "NOT METERED",
        },
    )
    if not fits:
        write(
            "decision",
            {
                "classification": "D",
                "reason": "Complete matched pair and mandatory evaluations do not fit the frozen budget",
                "retained_updates": 0,
                "cost_report": read("resource_report"),
            },
        )
        raise TimeoutError("F029 complete scope does not fit; stopped before training")


def evaluate(label, checkpoint, cursor, arm=None):
    # Reuse the exact F028 evaluator with output and loader injection only.
    old_out, old_restore = base.OUT, base.restore
    base.OUT = OUT / label
    base.OUT.mkdir(exist_ok=True)
    alias = base.OUT / f"INTERVENTION_t{cursor}.pt"
    expected = base.b8.sha256_file(checkpoint)
    if not alias.exists():
        shutil.copyfile(checkpoint, alias)
    assert base.b8.sha256_file(alias) == expected
    folder = base.OUT / f"INTERVENTION_evaluation_t{cursor}"
    if (folder / "metrics.json").exists():
        base.OUT = old_out
        return base.read(folder / "metrics.json")
    base.restore = lambda _path, _label: (
        restore(checkpoint, arm)
        if arm
        else ORIGINAL_RESTORE(checkpoint, "INTERVENTION")
    )
    try:
        return base.evaluation("INTERVENTION", cursor)
    finally:
        base.OUT, base.restore = old_out, old_restore


def history():
    assert read("preflight")["status"] == "PASS"
    for n, t in enumerate((251, 128, 192)):
        estimate(3 - n)
        path = base.OUT / f"INTERVENTION_t{t}.pt"
        assert (
            base.b8.sha256_file(path)
            == read("source_verification")["historical_checkpoint_hashes"][str(t)]
        )
        m = evaluate(f"historical_t{t}", path, t)
        if t == 251:
            old = base.read(base.OUT / "INTERVENTION_evaluation_t251/metrics.json")
            for k in (
                "JW",
                "legacy",
                "MAE",
                "RMSE",
                "macro_database_MAE",
                "ae17",
                "exc",
                "op",
            ):
                tol = max(
                    1e-6,
                    read("protocol")["tolerances"]["physical_relative"] * abs(old[k]),
                )
                assert abs(m[k] - old[k]) <= tol, (k, m[k], old[k])
            assert m["hydrogen"]["variant"] == "level3_mura"
            assert (
                abs(
                    m["hydrogen"]["signed_error_kcal"]
                    - old["hydrogen"]["signed_error_kcal"]
                )
                <= 1e-6
            )
    estimate(0)
    write(
        "historical_complete",
        {
            "states": [128, 192, 251],
            "baseline_reproduced": True,
            "starting_state_selection": "Fixed t251 regardless of these metrics",
        },
    )


def finish_pending(arm, model, shadow, optimizer, logs, bundle):
    if not logs or logs[-1].get("finite_status") != "PENDING":
        return
    r = logs[-1]
    entry = stream()[r["local_update"] - 1]
    after = base.scalar_tasks(model, shadow, "INTERVENTION", bundle, entry)
    for t in base.TASKS:
        r[t + "_actual_change"] = after[t] - r[t + "_loss_before"]
        r[t + "_loss_after"] = after[t]
    r["total_actual_change"] = sum(
        base.b8.LAMBDAS[t] * r[t + "_actual_change"] for t in base.TASKS
    )
    r["finite_step_overshoot"] = r["total_linear_change"] < 0 < r["total_actual_change"]
    r["locally_favorable_total"] = (
        r["total_linear_change"] < 0 and r["total_actual_change"] < 0
    )
    r["finite_status"] = "COMPLETE same frozen presentation before/after"
    save(
        arm, model, shadow, optimizer, r["local_update"], logs, OUT / f"{arm}_resume.pt"
    )


def train():
    base.deterministic()
    assert read("historical_complete")["baseline_reproduced"]
    estimate(0)
    manifest = stream()
    bundle = base.TrainingBundle()
    states, logs, counters, saved_rng = {}, {}, {}, {}
    for arm in LRS:
        path = OUT / f"{arm}_resume.pt"
        model, shadow, optimizer, saved = restore(
            path if path.exists() else OUT / f"{arm}_t251.pt", arm
        )
        states[arm] = (model, shadow, optimizer)
        logs[arm], counters[arm] = saved["logs"], saved["local_cursor"]
        saved_rng[arm] = saved["RNG"]
        finish_pending(arm, *states[arm], logs[arm], bundle)
    for k, entry in enumerate(manifest):
        for arm, lr in LRS.items():
            if counters[arm] > k:
                continue
            assert counters[arm] == k and counters["CONTROL"] in (
                counters["LOW_LR"],
                counters["LOW_LR"] + 1,
            )
            resource = read("resource_report")
            remaining = 64 - sum(counters.values())
            budget(
                remaining * max(read("preflight")["profiles"].values()) * 1.2
                + resource["endpoint_reserve"]
                + 600
            )
            model, shadow, optimizer = states[arm]
            if logs[arm]:
                base.pilot.restore_rng(saved_rng[arm])
            tick = time.monotonic()
            assert all(p.grad is None for p in model.parameters())
            losses, raw, combined, infos, _seconds = base.joint(
                model, shadow, "INTERVENTION", bundle, entry
            )
            if k == 0:
                initial = np.load(
                    OUT / "CONTROL_first_gradients.npz", allow_pickle=False
                )
                for t in base.TASKS:
                    np.testing.assert_allclose(
                        base.flatten(raw[t]).cpu().numpy(),
                        initial[t],
                        atol=1e-10,
                        rtol=1e-9,
                    )
            before = base.flatten(base.core.named_trainable_parameters(model))
            base.adamw_step(model, optimizer, raw, base.b8.LAMBDAS)
            delta = base.flatten(base.core.named_trainable_parameters(model)) - before
            state_checks(model, shadow, optimizer, 252 + k, lr)
            linear = {t: float(base.flatten(raw[t]) @ delta) for t in base.TASKS}
            row = {
                "arm": arm,
                "local_update": k + 1,
                "global_update": 252 + k,
                "source_position": k,
                "stream_sha256": base.b8.sha256_file(file("stream_manifest")),
                "chemistry_identity_variants": [
                    (s["identity"], s["variant"]) for s in infos
                ],
                "ae17_identity": entry["ae17"]["identity"],
                "ae17_variant": entry["ae17"]["variant"],
                "mrks_id": entry["mrks_id"],
                "LR": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(base.flatten(combined).norm()),
                "native_displacement_norm": float(delta.norm()),
                "total_linear_change": float(base.flatten(combined) @ delta),
                "seconds": time.monotonic() - tick,
                "finite_status": "PENDING" if k + 1 in SELECTED else "NOT REQUESTED",
                "constraints": base.pilot.constraints(model),
            }
            for t in base.TASKS:
                row[t + "_loss_before"] = losses[t]
                row[t + "_linear_change"] = linear[t]
                row[t + "_direction_conflict"] = linear[t] > 0
            for key in ("exp_avg", "exp_avg_sq"):
                row[key + "_norm"] = float(
                    torch.cat(
                        [s[key].double().flatten() for s in optimizer.state.values()]
                    ).norm()
                )
            assert row["constraints"]["PASS"]
            logs[arm].append(row)
            counters[arm] += 1
            save(
                arm,
                model,
                shadow,
                optimizer,
                k + 1,
                logs[arm],
                OUT / f"{arm}_resume.pt",
            )
            if k + 1 in SELECTED:
                np.savez(
                    OUT / f"{arm}_step{252 + k}_vectors.npz",
                    displacement=delta.cpu().numpy(),
                    **{t: base.flatten(raw[t]).cpu().numpy() for t in base.TASKS},
                )
            finish_pending(arm, model, shadow, optimizer, logs[arm], bundle)
            if k + 1 in MILESTONES:
                save(arm, model, shadow, optimizer, k + 1, logs[arm])
            saved_rng[arm] = base.pilot.rng()
            ordered = sorted(
                [r for values in logs.values() for r in values],
                key=lambda r: (r["local_update"], r["arm"]),
            )
            table("training_log", ordered)
            table("finite_steps", [r for r in ordered if r["local_update"] in SELECTED])
            write(
                "progress",
                {
                    "local_counters": counters,
                    "global_counters": {a: 251 + n for a, n in counters.items()},
                    "wall": elapsed(),
                },
            )
            print(
                "F029_UPDATE",
                arm,
                k + 1,
                "global",
                252 + k,
                "losses",
                losses,
                "seconds",
                row["seconds"],
                flush=True,
            )
            del raw, combined, delta, before
            torch.cuda.empty_cache()
    assert counters == {a: 32 for a in LRS}
    write(
        "training_complete",
        {
            "local_counters": counters,
            "global_counters": {a: 283 for a in LRS},
            "new_retained_updates": 64,
        },
    )
    bundle.close()


def uncertainty(metric, value):
    u = read("protocol")["evaluation_uncertainty"]
    return (
        u["JW_absolute_per_evaluation"]
        if metric == "JW"
        else max(
            u["operator_absolute_floor"],
            abs(value) * u["operator_relative_per_evaluation"],
        )
    )


def improved(metric, a, b):
    return a[metric] < b[metric] - uncertainty(metric, a[metric]) - uncertainty(
        metric, b[metric]
    )


def clean28(metrics):
    gates = {
        a: {"ratios": m["ratios"], "eligible": m["eligible"]}
        for a, m in metrics.items()
    }
    write("p536_ratios", gates)
    receipt = (
        read("clean28_receipt")
        if file("clean28_receipt").exists()
        else {"gates": gates, "values": {}, "future_test_access": False}
    )
    assert receipt["gates"] == gates
    for arm, m in metrics.items():
        if arm in receipt["values"]:
            assert receipt["values"][arm]["status"] != "IN_PROGRESS"
            continue
        if not m["eligible"]:
            receipt["values"][arm] = {
                "status": "NOT EVALUATED",
                "reason": "Original P536 gates failed",
                "evaluations": 0,
            }
            continue
        budget(450)
        receipt["values"][arm] = {"status": "IN_PROGRESS", "evaluations": 1}
        write("clean28_receipt", receipt)
        from evaluate_microbatch_endpoint import validation

        model, _shadow, _optimizer, _payload = restore(OUT / f"{arm}_t283.pt", arm)
        before = identity(model)
        bundle = base.PublicationDataset(base.DATA)
        result = validation(model, bundle)
        assert identity(model) == before
        receipt["values"][arm] = {
            "status": "EVALUATED",
            "evaluations": 1,
            "clean28": result["clean28"],
            "reaction_rows": [r for r in result["reaction_rows"] if r["clean"]],
        }
        assert len(receipt["values"][arm]["reaction_rows"]) == 28
        bundle.close()
        write("clean28_receipt", receipt)
    write("clean28_receipt", receipt)
    return receipt


def finish():
    assert read("training_complete")["new_retained_updates"] == 64
    final = {a: evaluate(a, OUT / f"{a}_t283.pt", 283, a) for a in LRS}
    original = base.read(
        OUT / "historical_t251/INTERVENTION_evaluation_t251/metrics.json"
    )
    states = [
        (f"ORIGINAL_t{t}", t, OUT / f"historical_t{t}/INTERVENTION_evaluation_t{t}")
        for t in (128, 192, 251)
    ]
    states += [(a, 283, OUT / f"{a}/INTERVENTION_evaluation_t283") for a in LRS]
    metrics, db, op, rx, atoms, exc = [], [], [], [], [], []
    for label, t, folder in states:
        m = base.read(folder / "metrics.json")
        metrics.append(
            {
                "state": label,
                "global_cursor": t,
                **{k: v for k, v in m.items() if not isinstance(v, dict)},
            }
        )
        db += [
            {**r, "state": label} for r in base.read(folder / "database_metrics.json")
        ]
        rx += [{**r, "state": label} for r in base.read(folder / "reactions.json")]
        atoms += [{**r, "state": label} for r in base.read(folder / "atoms.json")]
        for identity_, r in base.read(folder / "physical.json").items():
            op.append(
                {
                    "state": label,
                    "identity": identity_,
                    "name": r["name"],
                    "loss": r["op"],
                }
            )
            exc.append({"state": label, "identity": identity_, **r})
    originals = {r["identity"]: r for r in rx if r["state"] == "ORIGINAL_t251"}
    controls = {r["identity"]: r for r in rx if r["state"] == "CONTROL"}
    for r in rx:
        if r["state"] in LRS:
            assert (
                r["variant"]
                == originals[r["identity"]]["variant"]
                == controls[r["identity"]]["variant"]
            )
            r["absolute_error_delta_vs_original"] = abs(r["signed_error_kcal"]) - abs(
                originals[r["identity"]]["signed_error_kcal"]
            )
            r["absolute_error_delta_vs_control"] = abs(r["signed_error_kcal"]) - abs(
                controls[r["identity"]]["signed_error_kcal"]
            )
    table("checkpoint_metrics", metrics)
    table("historical_metrics", [r for r in metrics if r["global_cursor"] != 283])
    for name, values in (
        ("chemistry_databases", db),
        ("full90_operator", op),
        ("chemistry_reactions", rx),
        ("ae17_metrics", atoms),
        ("exc_metrics", exc),
    ):
        table(name, values)
    write(
        "hydrogen",
        {
            label: base.read(folder / "metrics.json")["hydrogen"]
            for label, _t, folder in states
        },
    )
    clean = clean28(final)
    low, control = final["LOW_LR"], final["CONTROL"]
    joint_original = all(improved(t, low, original) for t in ("JW", "op"))
    joint_control = all(improved(t, low, control) for t in ("JW", "op"))
    preserve = (
        low["ae17"] < base.b8.BASELINE["ae17"] and low["exc"] < base.b8.BASELINE["exc"]
    )
    classification = (
        "A"
        if joint_original and joint_control and preserve
        else "B"
        if joint_control and not joint_original
        else "C"
    )
    comparisons = []
    for metric in (
        "JW",
        "legacy",
        "MAE",
        "RMSE",
        "median_absolute_error",
        "macro_database_MAE",
        "ae17",
        "exc",
        "op",
    ):
        for name, a, b in (
            ("LOW_LR-original", low, original),
            ("LOW_LR-CONTROL", low, control),
            ("CONTROL-original", control, original),
        ):
            comparisons.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "signed_delta": a[metric] - b[metric],
                    "relative_delta": (a[metric] - b[metric]) / b[metric]
                    if b[metric]
                    else None,
                }
            )
    table("metric_comparisons", comparisons)
    for path, expected in read("source_verification")["F028_artifacts"].items():
        assert base.b8.sha256_file(path) == expected, path
    for name, expected in read("source_verification")["source_code"].items():
        assert base.b8.sha256_file(ROOT / name) == expected, name
    assert base.b8.sha256_file(SOURCE) == SOURCE_SHA
    restored = {}
    for arm in LRS:
        for local in MILESTONES:
            model, _shadow, _opt, _payload = restore(
                OUT / f"{arm}_t{251 + local}.pt", arm
            )
            check = base.pilot.constraints(model)
            assert check["PASS"]
            restored[f"{arm}_t{251 + local}"] = check
    write("constraint_checks", restored)
    decision = {
        "classification": classification,
        "technical": "PASS",
        "original": original,
        "control": control,
        "low_lr": low,
        "joint_progress_vs_original": joint_original,
        "joint_progress_vs_control": joint_control,
        "AE17_Exc_preserved": preserve,
        "clean28": clean,
        "wall": elapsed(),
        "budget_respected": elapsed() < LIMIT,
        "next_step": "Separately decide whether the measured short-horizon result warrants one longer fixed-LR qualification; no automatic extension, scheduler or new objective.",
    }
    write("decision", decision)
    resource = read("resource_report")
    resource.update(
        {
            "scientific_wall": elapsed(),
            "budget_respected": elapsed() < LIMIT,
            "recorded_phase_wall": {
                p.name: base.read(p) for p in ROOT.glob(PREFIX + "*timing.json")
            },
        }
    )
    logs = list(csv.DictReader((ROOT / (PREFIX + "training_log.csv")).open()))
    resource["training_update_wall_by_arm"] = {
        a: sum(float(r["seconds"]) for r in logs if r["arm"] == a) for a in LRS
    }
    write("resource_report", resource)
    provenance = read("provenance")
    provenance["large_artifacts"] = [
        {"path": str(p), "sha256": base.b8.sha256_file(p)}
        for p in OUT.rglob("*")
        if p.is_file()
    ]
    provenance["checkpoints"] = read("checkpoint_receipts")
    provenance["historical_files_unchanged"] = True
    write("provenance", provenance)
    metric_lines = "\n".join(
        f"|{k}|{original[k]:.12g}|{control[k]:.12g}|{low[k]:.12g}|"
        for k in (
            "JW",
            "legacy",
            "MAE",
            "RMSE",
            "median_absolute_error",
            "macro_database_MAE",
            "ae17",
            "exc",
            "op",
        )
    )
    steps = list(csv.DictReader((ROOT / (PREFIX + "finite_steps.csv")).open()))
    step_lines = "\n".join(
        f"|{r['arm']}/{r['global_update']}|{float(r['total_linear_change']):+.9g}|{float(r['total_actual_change']):+.9g}|{r['finite_step_overshoot']}|{r['relchem_direction_conflict']}|{r['op_direction_conflict']}|"
        for r in steps
    )
    report = f"""# F029 — {classification}

Exact F028 INTERVENTION t251 SHA256 `{SOURCE_SHA}` restored once per process and copied into independent matched arms; parameters, buffers and saved moments identical. Only optimizer LR differs:2.5e-5 versus2.5e-6. No new initialization, moment reset, epsilon or objective change. Global counters251→283,32 retained updates per arm,64 total. No temporary molecular optimizer preflight update.

|Original metric|ORIGINAL t251|CONTROL t283|LOW_LR t283|
|---|---:|---:|---:|
{metric_lines}

JW/operator progress relative to original:{joint_original}; relative to control:{joint_control}; low-LR AE17/Exc below original P536 thresholds:{preserve}. Full signed/relative three-way differences are in metric_comparisons.csv. Scientific P536 eligibility is separate:CONTROL {control["eligible"]}, LOW_LR {low["eligible"]}; all ratios in p536_ratios.json, H signed/absolute errors and exact variant in hydrogen.json. Clean28 receipts:{ {a: r["status"] for a, r in clean["values"].items()} }. Ineligible checkpoints were never passed to a development loader. No future/test access.

|Arm/global update|Total linear prediction|Actual same-presentation change|Overshoot|Chemistry direction conflict|Operator direction conflict|
|---|---:|---:|---|---|---|
{step_lines}

All task-wise before/after values, actual rounded displacement derivatives and classifications are in finite_steps.csv. A positive task derivative is a directional conflict; only negative total prediction with positive actual total change is called overshoot. Smaller LR does not itself reverse a fixed direction. No rejection/backtracking occurred.

Read-only original t128/t192/t251 characterization:historical_metrics.csv. No historical state selected a starting checkpoint; fixed t251 only. These sparse snapshots do not establish oscillations or overshoot. Next32 manifest is an explicitly defined next-cycle prefix, not a recovered historical post251 stream. Both arms used the same identities, variants, offsets and original paired physical presentations. F028 E251/E17/full90 evaluator reused unchanged, with only checkpoint loader/output routing injected; original task-specific precision, targets, gauge, overlap, dispersion and reductions retained. `relchem_loss`/chemistry task values denote c010-scaled F010 objective, never legacy metric.

Hard constraints pass at all retained updates and every restored milestone; native/shadow epsilon asserted after tensor loads. First complete raw task gradients pass deterministic parity before training. F64 weighted accumulation and one nativeF32 cast use unchanged F028 helpers. Both optimizers retain original moments/counters; independent storage checked. Scientific wall {elapsed():.3f}s /7200s. Per-update execution wall is measured; GPU-kernel-active time not separately metered. Full per-reaction, eight-database, E17, Exc and90-system operator receipts are retained. Large tensors/checkpoints outside Git, hash-bound in provenance. Historical F028 and production files unchanged. Focused test/Ruff/compileall/diff receipts finalized separately.

This is a short stabilization/progress diagnostic, not convergence or evidence for long-horizon reduced-LR benefit. LR does not establish a new functional or cure directional conflicts by definition.

One next action:{decision["next_step"]}
"""
    (ROOT / (PREFIX + "report.md")).write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "history", "train", "finish"))
    phase = parser.parse_args().phase
    tick = time.monotonic()
    OUT.mkdir(exist_ok=True)
    base.deterministic()
    try:
        globals()[phase]()
    except Exception as error:
        write("failure", {"phase": phase, "error": repr(error), "elapsed": elapsed()})
        raise
    write(
        phase + "_timing",
        {"seconds": time.monotonic() - tick, "workflow_wall": elapsed()},
    )
