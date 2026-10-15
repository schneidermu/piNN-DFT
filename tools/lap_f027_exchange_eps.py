"""F027: one seed-2027 initialization, two strictly interleaved epsilon arms."""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_fresh_grid_pilot_v1 as pilot
import lap_fresh_layernorm_v1 as audit

ROOT = pilot.ROOT
PREFIX = "lap_f027_exchange_eps_"
OUT = ROOT.parent / "lap_f027_exchange_eps_runs_20261015"
ARMS = {"CONTROL": 1e-5, "INTERVENTION": 1e-3}
MILESTONES = (0, 1, 2, 16, 32)
DEVICE = "cuda"
CONFIG = {
    "num_layers": 6,
    "h_dim": 32,
    "dropout": 0.0,
    "use_g_x": True,
    "use_g_c": True,
}


def read(name):
    return json.loads((ROOT / (PREFIX + name + ".json")).read_text())


def write(name, value):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def table(name, rows):
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def elapsed():
    return time.time() - int((ROOT / (PREFIX + "started.txt")).read_text()) / 1000


def budget(reserve=0):
    if elapsed() + reserve >= 10800:
        raise TimeoutError("F027 frozen scientific wall budget")


def deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def helper(arm):
    """Isolated namespaces reuse F026 mathematics without changing historical modules.

    Only output paths, wall budget and epsilon defaults are specialized. No old
    checkpoint/initialization/optimizer loader is ever called.
    """
    name = PREFIX + arm + "_qualified_helpers"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/lap_exchange_eps_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PREFIX = PREFIX + arm + "_"
    module.OUT = OUT / arm
    module.OUT.mkdir(parents=True, exist_ok=True)
    module.budget = budget
    for function in ("assert_epsilon", "configure_epsilon", "shadow_for"):
        getattr(module, function).__defaults__ = (ARMS[arm],)
    return module


def names(model):
    return pilot.named_trainable_parameters(model)


def tensor_subhash(model, kind):
    h = hashlib.sha256()
    iterator = (
        model.named_parameters() if kind == "parameters" else model.named_buffers()
    )
    for name, value in iterator:
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def identity(model, h):
    value = h.scientific_identity(model)
    value["parameters_sha256"] = tensor_subhash(model, "parameters")
    value["buffers_sha256"] = tensor_subhash(model, "buffers")
    return value


def load_initial():
    receipt = read("initialization")["artifact"]
    assert pilot.sha(Path(receipt["path"])) == receipt["sha256"]
    payload = torch.load(receipt["path"], weights_only=False, map_location="cpu")
    assert payload["seed"] == 2027 and payload["protocol_hash"] == pilot.sha(
        ROOT / (PREFIX + "protocol.json")
    )
    assert (
        pilot.state_hash(payload["model_template"])
        == read("initialization")["tensor_hash"]
    )
    return payload


def new_models():
    payload = load_initial()
    helpers = {arm: helper(arm) for arm in ARMS}
    models = {arm: copy.deepcopy(payload["model_template"]).to(DEVICE) for arm in ARMS}
    optimizers, shadows = {}, {}
    for arm, h in helpers.items():
        h.configure_epsilon(models[arm])
        h.assert_epsilon(models[arm])
        optimizers[arm] = h.optimizer_for(models[arm])
        shadows[arm] = h.shadow_for(models[arm])
    return models, shadows, optimizers, helpers


def batch(model, shadow, h, records, gradient=True):
    budget(300)
    h.assert_epsilon(model)
    h.assert_epsilon(shadow)
    assert all(p.dtype == torch.float32 for p in names(model).values())
    assert all(p.dtype == torch.float64 for p in names(shadow).values())
    shadow.load_state_dict(model.state_dict(), strict=True)
    h.assert_epsilon(shadow)
    results = [
        pilot.system_gradient(shadow, r, "FULL_SOURCE", DEVICE, gradient)
        for r in records.values()
    ]
    per = torch.stack([r[1] for r in results])
    g = per.mean(0)
    np.testing.assert_allclose(
        g.detach().cpu().numpy(),
        per.detach().cpu().numpy().mean(0),
        atol=1e-10,
        rtol=1e-9,
    )
    return [r[0] for r in results], g, per


def atomic_save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint(path, arm, model, optimizer, t, h):
    h.assert_epsilon(model)
    h.assert_optimizer(optimizer, model)
    payload = {
        "namespace": PREFIX,
        "arm": arm,
        "seed": 2027,
        "protocol_hash": pilot.sha(ROOT / (PREFIX + "protocol.json")),
        "source_hashes": read("source_verification")["representation_hashes"],
        "identity": identity(model, h),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "t": t,
        "RNG": pilot.rng(),
    }
    atomic_save(path, payload)
    return {"path": str(path), "sha256": pilot.sha(path), "arm": arm, "t": t}


def restore(path, arm, h):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if "identity" not in payload or "epsilon" not in payload["identity"]:
        raise ValueError("Required epsilon/scientific identity metadata missing")
    assert (
        payload["namespace"] == PREFIX
        and payload["arm"] == arm
        and payload["seed"] == 2027
    )
    assert payload["protocol_hash"] == pilot.sha(ROOT / (PREFIX + "protocol.json"))
    assert (
        payload["source_hashes"] == read("source_verification")["representation_hashes"]
    )
    model = copy.deepcopy(load_initial()["model_template"]).to(DEVICE)
    h.configure_epsilon(model)
    model.load_state_dict(payload["model"], strict=True)
    h.assert_epsilon(model)
    assert identity(model, h) == payload["identity"]
    optimizer = h.optimizer_for(model)
    optimizer.load_state_dict(payload["optimizer"])
    h.assert_optimizer(optimizer, model)
    if payload["t"]:
        assert {int(s["step"]) for s in optimizer.state.values()} == {payload["t"]}
    else:
        assert not optimizer.state
    shadow = h.shadow_for(model)
    shadow.load_state_dict(model.state_dict(), strict=True)
    h.assert_epsilon(shadow)
    assert payload["RNG"][2].device.type == "cpu"
    pilot.restore_rng(payload["RNG"])
    return model, shadow, optimizer, payload


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (ROOT / (PREFIX + "protocol.json")).exists()
    mandatory = [
        "lap_exchange_eps_v1_" + n
        for n in [
            "report.md",
            "protocol.json",
            "epsilon_manifest.json",
            "final_comparison.json",
            "derivative_checks.csv",
            "training_log.csv",
            "validation.json",
        ]
    ]
    mandatory += [
        "lap_fresh_layernorm_v1_report.md",
        "lap_fresh_grid_pilot_v1_report.md",
        "lap_fresh_grid_pilot_v1_representations.json",
    ]
    historical = {name: pilot.sha(ROOT / name) for name in mandatory}
    # Read the actual historical contents; numbers are used only AFTER F027 freezes.
    for name in mandatory:
        (ROOT / name).read_text()
    source = {
        k: v
        for k, v in json.loads(
            (ROOT / "lap_exchange_eps_v1_protocol.json").read_text()
        )["source_hashes"].items()
        if k.startswith(("train_models/", "dft_functionals/"))
    }
    for file in [
        "tools/lap_exchange_eps_v1.py",
        "tools/lap_fresh_grid_pilot_v1.py",
        "tools/lap_fresh_layernorm_v1.py",
    ]:
        source[file] = pilot.sha(ROOT / file)
        (ROOT / file).read_text()
    write(
        "protocol",
        {
            "experiment": "F027",
            "namespace": PREFIX,
            "seed": 2027,
            "scientific_wall_limit": 10800,
            "start_unix_ms": int((ROOT / (PREFIX + "started.txt")).read_text()),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            "source_hashes": source,
            "historical_hashes": historical,
            "constructor": CONFIG,
            "arms": ARMS,
            "updates_per_arm": 32,
            "order": "CONTROL then INTERVENTION for t=1..32; no extra molecular preflight updates",
            "t1_qualification": "After both retained update1 steps, qualify actual t1 for both arms before any update2. No reversible extra molecular step.",
            "effect_floor": 0.01,
            "optimizer": pilot.read("protocol")["optimizer"],
            "precision": pilot.read("protocol")["precision"],
            "derivative_procedure": "Reuse unchanged F026 independent LN, independent dE/dP, fixed H2 coordinates and prospective seven-step curvature-aware FD schedule; per-arm helper namespaces specialize paths and epsilon defaults only",
            "tolerances": json.loads(
                (ROOT / "lap_exchange_eps_v1_protocol.json").read_text()
            )["tolerances"],
            "qualification_gate": "Confirmed analytical discrepancy, source/config/constraint failure stops both arms. Unresolved FD with passed independent analytical/path checks may finish as technically LIMITED, classified D regardless of observed benefit. No tolerance changes.",
            "reserve": "Profile both full gradient batches before retained steps; reserve 1300s for t1 derivatives, milestones, final qualification/tests and swaps, plus 25% training-time margin",
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )


def initialize():
    assert not (ROOT / (PREFIX + "initialization.json")).exists(), (
        "Never generate another scientific initialization"
    )
    p = read("protocol")
    for name, expected in {**p["source_hashes"], **p["historical_hashes"]}.items():
        assert pilot.sha(ROOT / name) == expected, name
    previous = json.loads((ROOT / "lap_exchange_eps_v1_provenance.json").read_text())
    assert (
        pilot.sha(ROOT / "tools/lap_exchange_eps_v1.py")
        == previous["new_driver_sha256"]
    )
    records = pilot.read("representations")["records"]
    verification = []
    for name, record in records.items():
        assert pilot.sha(pilot.local(record["path"])) == record["sha256"]
        with h5py.File(pilot.local(record["path"])) as f:
            assert f["features"].dtype == np.dtype("float64")
            reference = f["ref_FULL_SOURCE"][:]
            projected = np.zeros_like(reference)
            for start in range(0, len(f["weights"]), 2048):
                phi = f["phi"][start : start + 2048]
                wv = f["weights"][start : start + 2048] * f["vxc"][start : start + 2048]
                projected += phi.T @ (phi * wv[:, None])
            np.testing.assert_allclose(projected, reference, atol=1e-10, rtol=1e-10)
            verification.append(
                {
                    "system": name,
                    "points": len(f["weights"]),
                    "n_AO": len(reference),
                    "reference_projection_error": float(
                        np.linalg.norm(projected - reference)
                    ),
                    "source_precision": "F64",
                }
            )
    assert [r["points"] for r in verification] == [47080, 89200, 85960]
    write(
        "source_verification",
        {
            "PASS": True,
            "systems": verification,
            "representation_hashes": {name: r["sha256"] for name, r in records.items()},
            "records": records,
            "new_data": False,
        },
    )
    deterministic()
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    initial = pilot.pcPBELMLOptimizerV2Lap(**CONFIG)
    assert sum(p.numel() for p in names(initial).values()) == 9446
    assert len({id(p) for p in names(initial).values()}) == len(names(initial))
    path = OUT / "seed2027_initialization.pt"
    atomic_save(
        path,
        {
            "seed": 2027,
            "protocol_hash": pilot.sha(ROOT / (PREFIX + "protocol.json")),
            "model_template": initial,
            "RNG": pilot.rng(),
        },
    )
    write(
        "initialization",
        {
            "seed": 2027,
            "scientific_initializations": 1,
            "tensor_hash": pilot.state_hash(initial),
            "config": CONFIG,
            "coordinates": 9446,
            "parameter_order": [
                {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
                for n, p in names(initial).items()
            ],
            "artifact": {"path": str(path), "sha256": pilot.sha(path)},
            "seed2026_weights_or_moments_loaded": 0,
        },
    )
    models, _shadows, optimizers, helpers = new_models()
    assert (
        pilot.state_hash(models["CONTROL"])
        == pilot.state_hash(models["INTERVENTION"])
        == pilot.state_hash(initial)
    )
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(
            models["CONTROL"].parameters(), models["INTERVENTION"].parameters()
        )
    )
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(models["CONTROL"].buffers(), models["INTERVENTION"].buffers())
    )
    manifests = {arm: identity(models[arm], helpers[arm]) for arm in ARMS}
    assert manifests["CONTROL"]["order"] == manifests["INTERVENTION"]["order"]
    assert (
        manifests["CONTROL"]["scientific_hash"]
        != manifests["INTERVENTION"]["scientific_hash"]
    )
    assert all(not o.state for o in optimizers.values())
    write(
        "epsilon_manifests",
        {
            "initial": manifests,
            "initial_tensor_identity": "PASS",
            "independent_storage": "PASS",
        },
    )
    pilot.restore_rng(load_initial()["RNG"])
    for arm in ARMS:
        checkpoint(
            OUT / f"{arm}_t0.pt", arm, models[arm], optimizers[arm], 0, helpers[arm]
        )
    write(
        "provenance",
        {
            "source_hashes": p["source_hashes"],
            "historical_hashes": p["historical_hashes"],
            "initialization_receipt": read("initialization")["artifact"],
            "only_models_loaded": "new F027 seed2027 artifacts",
            "seed2026_checkpoints_loaded": 0,
        },
    )


def metrics(arm, t, model, optimizer, g, losses, h):
    layout = audit.layout(model)
    g = g.detach().cpu().numpy()
    exchange_ln = np.concatenate(
        [np.arange(s, e) for n, s, e, ln, _ in layout if ln and n.startswith("x_")]
    )
    non_ln = np.concatenate([np.arange(s, e) for _, s, e, ln, _ in layout if not ln])
    m = [
        s["exp_avg"].detach().flatten().double().cpu() for s in optimizer.state.values()
    ]
    v = [
        s["exp_avg_sq"].detach().flatten().double().cpu()
        for s in optimizer.state.values()
    ]
    initial = load_initial()["model_template"]
    constraint = pilot.constraints(model)
    assert constraint["PASS"]
    h.assert_epsilon(model)
    return {
        "arm": arm,
        "t": t,
        "mean_loss": float(np.mean(losses)),
        "H2_loss": losses[0],
        "BeH2_loss": losses[1],
        "CO_loss": losses[2],
        "maximum_system_loss": max(losses),
        "gradient_norm": audit.norm(g),
        "exchange_LN_gradient_norm": audit.norm(g[exchange_ln]),
        "outside_LayerNorm_gradient_norm": audit.norm(g[non_ln]),
        "displacement_from_t0": audit.norm(pilot.vector(model) - pilot.vector(initial)),
        "first_moment_norm": float(torch.cat(m).norm()) if m else 0.0,
        "second_moment_norm": float(torch.cat(v).norm()) if v else 0.0,
        "finite": True,
        "constraints": "PASS",
        "epsilon": "PASS",
    }, constraint


def qualify(arm, t, model, shadow, optimizer, h, records):
    tic = time.monotonic()
    losses, g, per = batch(model, shadow, h, records)
    seconds = time.monotonic() - tic
    stat = h.statistics(model, records, t)
    table(arm + f"_hidden_t{t}", stat)
    rows, status = h.derivative_checks(model, records["H2"], per, t)
    for row in rows:
        row["arm"] = arm
    for row in status:
        row["arm"] = arm
    checkpoint(OUT / f"{arm}_t{t}.pt", arm, model, optimizer, t, h)
    restored, rs, _ro, payload = restore(OUT / f"{arm}_t{t}.pt", arm, h)
    rv, rg, _ = batch(restored, rs, h, records)
    np.testing.assert_allclose(rv, losses, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
    assert torch.equal(torch.get_rng_state(), payload["RNG"][2])
    return losses, g, rows, status, seconds


def preflight():
    assert not (ROOT / (PREFIX + "preflight.json")).exists()
    deterministic()
    models, shadows, optimizers, helpers = new_models()
    records = read("source_verification")["records"]
    rows, status, profiles, fixture, summary, constraints = [], [], {}, {}, [], {}
    gradients = {}
    for arm, h in helpers.items():
        losses, g, dr, ds, seconds = qualify(
            arm, 0, models[arm], shadows[arm], optimizers[arm], h, records
        )
        rows += dr
        status += ds
        profiles[arm] = seconds
        fixture[arm] = pilot.fixture_parity(shadows[arm], records["H2"], DEVICE)
        row, c = metrics(arm, 0, models[arm], optimizers[arm], g, losses, h)
        summary.append(row)
        constraints[arm + "_t0"] = c
        gradients[arm + "_g1"] = g.detach().cpu().numpy()
        table("derivative_checks", rows)
        print("initial qualified", row, flush=True)
    projected = 32 * sum(profiles.values()) * 1.25 + 1300
    budget(projected)
    table("checkpoint_metrics", summary)
    write("constraint_checks", constraints)
    np.savez(OUT / "initial_gradients.npz", **gradients)
    write(
        "preflight",
        {
            "PASS": True,
            "derivatives": status,
            "batch_seconds": profiles,
            "small_fixture_parity": fixture,
            "projected_remaining_seconds": projected,
            "remaining_wall_seconds": 10800 - elapsed(),
            "extra_molecular_optimizer_updates": 0,
            "t1_policy": "qualify both retained t1 states before any update2",
            "technical": "VALID"
            if all(s["status"] == "QUALIFIED" for s in status)
            else "LIMITED",
        },
    )


def pair_save(models, optimizers, counters, helpers, checked):
    payload = {
        "namespace": PREFIX,
        "protocol_hash": pilot.sha(ROOT / (PREFIX + "protocol.json")),
        "counters": counters.copy(),
        "source_hashes": read("source_verification")["representation_hashes"],
        "optimizer_configuration": read("protocol")["optimizer"],
        "checked_milestones": checked.copy(),
        "RNG": pilot.rng(),
        "models": {arm: models[arm].state_dict() for arm in ARMS},
        "optimizers": {arm: optimizers[arm].state_dict() for arm in ARMS},
        "identities": {arm: identity(models[arm], helpers[arm]) for arm in ARMS},
    }
    atomic_save(OUT / "pair_resume.pt", payload)


def train():
    deterministic()
    assert read("preflight")["PASS"]
    models, shadows, optimizers, helpers = new_models()
    counters = {arm: 0 for arm in ARMS}
    checked = [0]
    history = []
    summaries = list(
        csv.DictReader((ROOT / (PREFIX + "checkpoint_metrics.csv")).open())
    )
    derivative_rows = list(
        csv.DictReader((ROOT / (PREFIX + "derivative_checks.csv")).open())
    )
    constraints = read("constraint_checks")
    qualifications = read("preflight")["derivatives"]
    if (OUT / "pair_resume.pt").exists():
        payload = torch.load(
            OUT / "pair_resume.pt", weights_only=False, map_location="cpu"
        )
        assert payload["namespace"] == PREFIX and payload["protocol_hash"] == pilot.sha(
            ROOT / (PREFIX + "protocol.json")
        )
        counters = payload["counters"]
        checked = payload["checked_milestones"]
        for arm in ARMS:
            models[arm].load_state_dict(payload["models"][arm])
            helpers[arm].assert_epsilon(models[arm])
            assert identity(models[arm], helpers[arm]) == payload["identities"][arm]
            optimizers[arm].load_state_dict(payload["optimizers"][arm])
            helpers[arm].assert_optimizer(optimizers[arm], models[arm])
        history = list(csv.DictReader((ROOT / (PREFIX + "training_log.csv")).open()))
        assert len(history) == sum(counters.values()), (
            "Cursor/log mismatch: stop without repeating steps"
        )
        pilot.restore_rng(payload["RNG"])
        if (ROOT / (PREFIX + "t1_qualification.json")).exists():
            qualifications += read("t1_qualification")["derivatives"]
    records = read("source_verification")["records"]
    for t in range(1, 33):
        for arm in ARMS:
            if counters[arm] >= t:
                continue
            assert counters[arm] == t - 1
            assert counters["CONTROL"] in (
                counters["INTERVENTION"],
                counters["INTERVENTION"] + 1,
            )
            budget(
                (64 - sum(counters.values()))
                * max(read("preflight")["batch_seconds"].values())
                * 1.1
                + 750
            )
            h = helpers[arm]
            tic = time.monotonic()
            losses, g, _ = batch(models[arm], shadows[arm], h, records)
            before = pilot.vector(models[arm])
            h.native_step(models[arm], optimizers[arm], g)
            counters[arm] += 1
            assert {int(s["step"]) for s in optimizers[arm].state.values()} == {t}
            delta = pilot.vector(models[arm]) - before
            row = {
                "arm": arm,
                "update": t,
                "counter": t,
                "mean_preupdate_loss": float(np.mean(losses)),
                "H2_loss": losses[0],
                "BeH2_loss": losses[1],
                "CO_loss": losses[2],
                "gradient_norm": float(g.norm()),
                "displacement_norm": audit.norm(delta),
                "gradient_dot_displacement": float(g.cpu().numpy() @ delta),
                "batch_seconds": time.monotonic() - tic,
                "first_moment_norm": float(
                    torch.cat(
                        [
                            s["exp_avg"].flatten().double()
                            for s in optimizers[arm].state.values()
                        ]
                    ).norm()
                ),
                "second_moment_norm": float(
                    torch.cat(
                        [
                            s["exp_avg_sq"].flatten().double()
                            for s in optimizers[arm].state.values()
                        ]
                    ).norm()
                ),
            }
            history.append(row)
            table("training_log", history)
            pair_save(models, optimizers, counters, helpers, checked)
            print("retained update", row, flush=True)
        if t in MILESTONES and t not in checked:
            for arm, h in helpers.items():
                if t == 1:
                    losses, g, dr, ds, _seconds = qualify(
                        arm, t, models[arm], shadows[arm], optimizers[arm], h, records
                    )
                    derivative_rows += dr
                    qualifications += ds
                    table("derivative_checks", derivative_rows)
                else:
                    losses, g, _ = batch(models[arm], shadows[arm], h, records)
                    checkpoint(
                        OUT / f"{arm}_t{t}.pt", arm, models[arm], optimizers[arm], t, h
                    )
                    if t == 32:
                        rm, rs, ro, payload = restore(OUT / f"{arm}_t{t}.pt", arm, h)
                        rv, rg, _ = batch(rm, rs, h, records)
                        np.testing.assert_allclose(rv, losses, atol=1e-10, rtol=1e-9)
                        torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
                        models[arm], shadows[arm], optimizers[arm] = rm, rs, ro
                row, c = metrics(arm, t, models[arm], optimizers[arm], g, losses, h)
                summaries.append(row)
                constraints[arm + f"_t{t}"] = c
                np.savez(OUT / f"{arm}_gradient_t{t}.npz", gradient=g.cpu().numpy())
                if t in (1, 2):
                    log = next(
                        r for r in history if r["arm"] == arm and int(r["update"]) == t
                    )
                    log["realized_loss_change"] = float(np.mean(losses)) - float(
                        log["mean_preupdate_loss"]
                    )
                    log["step_improved"] = log["realized_loss_change"] < 0
                print("milestone", row, flush=True)
            checked.append(t)
            if t == 1:
                write(
                    "t1_qualification",
                    {
                        "derivatives": [
                            q for q in qualifications if int(q["theta"]) == 1
                        ],
                        "extra_molecular_updates": 0,
                        "both_qualified_before_update2": True,
                    },
                )
            table("checkpoint_metrics", summaries)
            table("training_log", history)
            write("constraint_checks", constraints)
            pair_save(models, optimizers, counters, helpers, checked)
    write(
        "training_completion",
        {
            "counters": counters,
            "retained_updates": sum(counters.values()),
            "extra_molecular_preflight_updates": 0,
            "derivatives": qualifications,
            "milestones": checked,
        },
    )


def finish():
    deterministic()
    completion = read("training_completion")
    assert completion["counters"] == {"CONTROL": 32, "INTERVENTION": 32}
    # Materialize the protocol-bound source/config fields in the atomic cursor too.
    # This changes metadata only: no tensor, moment, RNG or update is modified.
    cursor_path = OUT / "pair_resume.pt"
    cursor = torch.load(cursor_path, weights_only=False, map_location="cpu")
    assert cursor["counters"] == completion["counters"]
    assert cursor["protocol_hash"] == pilot.sha(ROOT / (PREFIX + "protocol.json"))
    cursor["source_hashes"] = read("source_verification")["representation_hashes"]
    cursor["optimizer_configuration"] = read("protocol")["optimizer"]
    atomic_save(cursor_path, cursor)
    rows = list(csv.DictReader((ROOT / (PREFIX + "checkpoint_metrics.csv")).open()))
    metrics_by = {(r["arm"], int(r["t"])): r for r in rows}
    control = float(metrics_by["CONTROL", 32]["mean_loss"])
    intervention = float(metrics_by["INTERVENTION", 32]["mean_loss"])
    assert np.isfinite(control) and control > 0
    benefit = (control - intervention) / control
    qualified = all(q["status"] == "QUALIFIED" for q in completion["derivatives"])
    classification = (
        "D"
        if not qualified
        else "A"
        if benefit >= 0.01
        else "B"
        if benefit > 0
        else "C"
    )
    systems = []
    for name in ("H2", "BeH2", "CO"):
        key = name + "_loss"
        c0, i0, c32, i32 = [
            float(metrics_by[arm, t][key])
            for arm, t in [
                ("CONTROL", 0),
                ("INTERVENTION", 0),
                ("CONTROL", 32),
                ("INTERVENTION", 32),
            ]
        ]
        systems.append(
            {
                "system": name,
                "control_t0": c0,
                "intervention_t0": i0,
                "control_t32": c32,
                "intervention_t32": i32,
                "absolute_delta": i32 - c32,
                "relative_improvement": (c32 - i32) / c32 if c32 > 0 else None,
                "control_reduction": c0 - c32,
                "intervention_reduction": i0 - i32,
                "improved": i32 < c32,
            }
        )
    table("system_comparison", systems)
    records = read("source_verification")["records"]
    swaps = []
    if (ROOT / (PREFIX + "optional_swaps.json")).exists():
        swaps = read("optional_swaps").get("values", [])
    elif elapsed() + 350 < 10800:
        for arm in ARMS:
            h = helper(arm)
            model, _, _, _ = restore(OUT / f"{arm}_t32.pt", arm, h)
            opposite = "INTERVENTION" if arm == "CONTROL" else "CONTROL"
            eh = helper(opposite)
            eh.configure_epsilon(model)
            shadow = eh.shadow_for(model)
            losses, _, _ = batch(model, shadow, eh, records, False)
            swaps.append(
                {
                    "weights": arm + "_t32",
                    "epsilon": ARMS[opposite],
                    "mean_loss": float(np.mean(losses)),
                    "system_losses": losses,
                    "read_only": True,
                }
            )
        write(
            "optional_swaps",
            {"status": "PERFORMED", "values": swaps, "used_for_selection": False},
        )
    else:
        write(
            "optional_swaps",
            {"status": "NOT RUN", "reason": "reserved integrity budget"},
        )
    for name, expected in {
        **read("protocol")["source_hashes"],
        **read("protocol")["historical_hashes"],
    }.items():
        assert pilot.sha(ROOT / name) == expected, name
    for r in records.values():
        assert pilot.sha(pilot.local(r["path"])) == r["sha256"]
    manifests = read("epsilon_manifests")
    restored_constraints = read("constraint_checks")
    for arm in ARMS:
        h = helper(arm)
        for t in MILESTONES:
            restored, restored_shadow, _optimizer, payload = restore(
                OUT / f"{arm}_t{t}.pt", arm, h
            )
            h.assert_epsilon(restored_shadow)
            constraint = pilot.constraints(restored)
            assert constraint["PASS"]
            restored_constraints[arm + f"_restored_t{t}"] = constraint
            assert payload["t"] == t and payload["arm"] == arm
            manifests[arm + f"_t{t}"] = {
                "identity": payload["identity"],
                "checkpoint_sha256": pilot.sha(OUT / f"{arm}_t{t}.pt"),
            }
    write("epsilon_manifests", manifests)
    write("constraint_checks", restored_constraints)
    provenance = read("provenance")
    provenance["new_checkpoint_array_receipts"] = [
        {"path": str(p), "sha256": pilot.sha(p)} for p in OUT.rglob("*") if p.is_file()
    ]
    provenance["driver_sha256"] = pilot.sha(Path(__file__))
    provenance["historical_sources_unchanged"] = True
    write("provenance", provenance)
    previous = json.loads(
        (ROOT / "lap_exchange_eps_v1_final_comparison.json").read_text()
    )
    decision = {
        "experiment": "F027",
        "classification": classification,
        "technical": "VALID" if qualified else "LIMITED",
        "seed": 2027,
        "benefit": benefit,
        "effect_floor_pass": benefit >= 0.01,
        "control_initial": float(metrics_by["CONTROL", 0]["mean_loss"]),
        "intervention_initial": float(metrics_by["INTERVENTION", 0]["mean_loss"]),
        "control_final": control,
        "intervention_final": intervention,
        "control_initial_gradient": float(metrics_by["CONTROL", 0]["gradient_norm"]),
        "intervention_initial_gradient": float(
            metrics_by["INTERVENTION", 0]["gradient_norm"]
        ),
        "startup_reduction": float(metrics_by["CONTROL", 0]["gradient_norm"])
        / float(metrics_by["INTERVENTION", 0]["gradient_norm"]),
        "seed2026_matched_benefit": previous["relative_final_improvement_B"],
        "counters": completion["counters"],
        "retained_updates": 64,
        "extra_molecular_preflight_updates": 0,
        "scientific_wall_seconds": elapsed(),
        "budget_respected": elapsed() < 10800,
        "next_action": "One separately designed broader matched qualification covering full90 operator, original AE17/Exc/relative-chemistry metrics and H accuracy; no automatic full90 training or curriculum."
        if classification == "A"
        else "Close this replication and review the recorded seed-specific startup/trajectory differences before deciding on any further experiment; do not search seeds or eps values.",
    }
    write("decision", decision)
    times = {
        p.stem: json.loads(p.read_text()) for p in ROOT.glob(PREFIX + "*timing.json")
    }
    write(
        "resource_report",
        {
            "scientific_workflow_wall_seconds": elapsed(),
            "limit_seconds": 10800,
            "recorded_phase_process_wall": times,
            "GPU_active_time": "not separately metered; phase process wall is not GPU-active time",
            "budget_respected": elapsed() < 10800,
        },
    )
    system_text = "\n".join(
        f"|{r['system']}|{r['control_t0']:.12g}|{r['intervention_t0']:.12g}|{r['control_t32']:.12g}|{r['intervention_t32']:.12g}|{r['absolute_delta']:+.10g}|{r['relative_improvement']:.8%}|"
        for r in systems
    )
    log = list(csv.DictReader((ROOT / (PREFIX + "training_log.csv")).open()))
    early = "\n".join(
        f"|{r['arm']}|{r['update']}|{float(r['gradient_dot_displacement']):+.10g}|{float(r['realized_loss_change']):+.10g}|"
        for r in log
        if int(r["update"]) <= 2
    )
    (ROOT / (PREFIX + "report.md")).write_text(
        f"""# F027 — independent seed-2027 matched epsilon replication

Classification **{classification}**; technical **{decision["technical"]}**. Matched B2027={benefit:.12g}, with the predeclared 1% floor {benefit >= 0.01}. Exactly 32 CONTROL and 32 INTERVENTION retained steps; no extra molecular preflight steps or seed-2026 weights/moments loaded.

## Frozen setup and initialization

One constructor initialization followed Python/NumPy/Torch CPU/CUDA seed2027, deterministic algorithms and disabled CUDA/cuDNN TF32. Exact R0 constructor 6 layers, width32, dropout0, Gx/Gc enabled, 9446 unique coordinates. Its newly saved initialization artifact is the sole template for all copies/restores; no historical initialization is loaded. Parameter/buffer equality, order and independent storage pass. Tensor hashes match while scientific hashes differ because all five genuine exchange LNs use 1e-5 versus 1e-3. Every correlation LN remains 1e-5. No topology/activation/anchor/PBE-regularization changes. Exchange hidden features still enter correlation.

Data: authenticated original H2/BeH2/CO FULL_SOURCE grids 47080/89200/85960, identical frozen densities, basis/AO order, quadrature, overlap, gauge and targets. Independent full-grid reference reprojection passes frozen 1e-10 criteria. No grids, densities, labels or systems generated. Source/manifest hashes are in protocol/provenance/source_verification.json; historical F026 and earlier files remain unchanged.

The complete operator matrix precedes the original overlap-normalized squared loss, with equal 1/3 system weights. Original bounded two-pass adjoint, F64 learned shadow/AO/gradient accumulation and original F32 PBE path are unchanged. One final gradient cast feeds native F32 AdamW with fresh independent moments, lr2.5e-5, betas.9/.999, eps1e-8, decay.01, foreach false. No energy/chemistry tasks, scheduler, clipping, gradient surgery or resets.

## Numerical qualification and checkpoint contract

F026 helper code is loaded in isolated per-arm namespaces. Only namespace/output paths, wall-budget callback and explicit epsilon default are specialized; the derivative mathematics is unchanged. The five modules and all correlation eps are asserted on native, F64 shadow, every load_state_dict and numerical copies. Every checkpoint stores architecture/order/shapes, per-module epsilon, separate parameter/buffer hashes, scientific hash, optimizer/config/counter, CPU RNG ByteTensors, data hashes and protocol hash. Missing metadata fails closed. Checkpoints t0/t1/t2/t16/t32 and atomic interleaved pair cursor live outside Git with SHA receipts.

t0 preflight includes two-pass/direct small-fixture parity, full mean-of-three gradient closure, full-batch cost profiles, independent LN first/mixed derivatives on actual zero/near-zero/ordinary hidden inputs, independent energy-first dE/dP path, and the fixed H2 coordinates x_feature_extractor.1.bias[6], .3.fc.1.bias[16], .3.fc.0.weight[887]. F026's prospectively fixed seven-step curvature-aware schedule and tolerances are unchanged. Both retained update1 states were qualified and restored before update2; no temporary molecular step was executed. Final checkpoints also reproduce full losses and gradients after restoration. Derivative statuses: {completion["derivatives"]}. Any unresolved necessary check forces classification D rather than claimed replication. Old F026/audit classifications are not rewritten.

## Matched results

Mean losses: CONTROL t0={decision["control_initial"]:.16g}, t32={control:.16g}; INTERVENTION t0={decision["intervention_initial"]:.16g}, t32={intervention:.16g}. Different t0 losses with identical tensors reflect different mathematical functions, not initialization mismatch. Initial gradient norms {decision["control_initial_gradient"]:.12g} versus {decision["intervention_initial_gradient"]:.12g}; startup reduction factor {decision["startup_reduction"]:.12g}. Lower startup norm is not the endpoint success metric.

|System|C t0|I t0|C t32|I t32|I−C t32|relative benefit|
|---|---:|---:|---:|---:|---:|---:|
{system_text}

Maximum final system losses: CONTROL {float(metrics_by["CONTROL", 32]["maximum_system_loss"]):.12g}, INTERVENTION {float(metrics_by["INTERVENTION", 32]["maximum_system_loss"]):.12g}. Individual regressions, if present, remain explicit in system_comparison.csv; all systems improving is not an extra gate. The result is a uniform mean over three training systems, not a held-out generalization score.

|Arm|update|grad dot actual displacement|realized loss change|
|---|---:|---:|---:|
{early}

Negative dot predicts local descent. Milestone metrics contain complete gradients, exchange-LN/outside-LN norms, moments, displacement, finite/configuration and actual hard-constraint checks at all five predetermined states. No earlier checkpoint selection. Optional read-only endpoint epsilon swaps: {swaps if swaps else "NOT RUN"}. They do not choose a model or imply causal startup-memory percentages.

## Cross-seed conclusion and limits

F026 seed2026 within-seed B={previous["relative_final_improvement_B"]:.10%}; F027 seed2027 within-seed B={benefit:.10%}. Different seeds' absolute losses are not treated as matched starts. Two seeds do not justify a confidence interval or universal efficacy claim. UEG recovery/Gx anchors, spin symmetry, high-density/rapid-gradient limits, tau independence and finite variational operators pass on both actual epsilon-configured functions and restored checkpoints.

Next action: {decision["next_action"]} Full90 operator, unweighted AE17 and Exc, original relative chemistry, and H accuracy remain mandatory before broad scientific qualification. Either fixed joint training or a separately qualified physics-first→chemistry strategy would require a new protocol. No automatic continuation, seed search, epsilon sweep, chemistry, full90 rebuild, SCF or Clean28 access occurred.

Scientific workflow wall at report generation {elapsed():.3f}s, budget10800s, respected {elapsed() < 10800}. Recorded process timings are separate; GPU-active time was not metered. Tests/lint/compile/diff-check results are recorded in the final validation receipt.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["freeze", "initialize", "preflight", "train", "finish"]
    )
    phase = parser.parse_args().phase
    tic = time.monotonic()
    getattr(sys.modules[__name__], phase)()
    write(
        phase + "_timing",
        {
            "process_wall_seconds": time.monotonic() - tic,
            "scientific_wall_seconds": elapsed(),
        },
    )
