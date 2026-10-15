"""F026: one isolated exchange-epsilon intervention against a frozen control."""

import argparse
import copy
import csv
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_fresh_grid_pilot_v1 as pilot
import lap_fresh_layernorm_v1 as audit

ROOT = pilot.ROOT
PREFIX = "lap_exchange_eps_v1_"
OUT = ROOT.parent / "lap_exchange_eps_v1_runs_20261015"
DEVICE = "cuda"
EXCHANGE = (
    "x_feature_extractor.1",
    "x_feature_extractor.3.fc.1",
    "x_feature_extractor.3.fc.4",
    "x_feature_extractor.4.fc.1",
    "x_feature_extractor.4.fc.4",
)
COORDINATES = (
    ("x_feature_extractor.1.bias", 6),
    ("x_feature_extractor.3.fc.1.bias", 16),
    ("x_feature_extractor.3.fc.0.weight", 887),
)
INITIAL_HASH = "359088b9713e532be0f811b4bd9f5f4f91b34d531a936db7548f46b2b410aedd"


def read(name):
    return json.loads((ROOT / (PREFIX + name + ".json")).read_text())


def write(name, data):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )


def table(name, rows):
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def elapsed():
    return time.time() - int((ROOT / (PREFIX + "started.txt")).read_text()) / 1000


def budget(reserve=0):
    if elapsed() + reserve >= 7200:
        raise TimeoutError("F026 frozen scientific wall budget")


def epsilon_rows(model):
    return [
        {
            "name": name,
            "epsilon": module.eps,
            "owner": "exchange" if name in EXCHANGE else "correlation",
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    ]


def assert_epsilon(model, exchange=1e-3):
    modules = dict(model.named_modules())
    actual = {
        name
        for name, module in modules.items()
        if name.startswith("x_") and isinstance(module, nn.LayerNorm)
    }
    assert actual == set(EXCHANGE)
    assert len({id(modules[name]) for name in EXCHANGE}) == 5
    for row in epsilon_rows(model):
        expected = exchange if row["owner"] == "exchange" else 1e-5
        assert row["epsilon"] == expected, (row, expected)


def configure_epsilon(model, exchange=1e-3):
    # Independent validation sees both missing/extra modules and accidental c_ changes.
    modules = dict(model.named_modules())
    assert {
        n
        for n, m in modules.items()
        if n.startswith("x_") and isinstance(m, nn.LayerNorm)
    } == set(EXCHANGE)
    for name in EXCHANGE:
        assert isinstance(modules[name], nn.LayerNorm)
        modules[name].eps = exchange
    assert_epsilon(model, exchange)


def scientific_identity(model):
    manifest = {
        "tensor_hash": pilot.state_hash(model),
        "epsilon": epsilon_rows(model),
        "config": model.model_kwargs,
        "order": [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for n, p in audit.names(model).items()
        ],
        "coordinates": sum(p.numel() for p in audit.names(model).values()),
        "precision": pilot.read("protocol")["precision"],
    }
    manifest["scientific_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return manifest


def construct(state, epsilon):
    # Construction is only allocation: every tensor comes from the original artifact.
    saved = pilot.rng()
    model = pilot.pcPBELMLOptimizerV2Lap(**pilot.read("protocol")["constructor"])
    configure_epsilon(model, epsilon)
    model.load_state_dict(state, strict=True)
    assert_epsilon(model, epsilon)
    pilot.restore_rng(saved)
    return model.to(DEVICE)


def optimizer_for(model):
    return torch.optim.AdamW(
        audit.names(model).values(),
        **{**pilot.read("protocol")["optimizer"], "betas": (0.9, 0.999)},
    )


def assert_optimizer(optimizer, model):
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert all(a is b for a, b in zip(group["params"], audit.names(model).values()))
    assert len(group["params"]) == len(audit.names(model))
    for key, expected in pilot.read("protocol")["optimizer"].items():
        actual = tuple(group[key]) if key == "betas" else group[key]
        assert actual == (tuple(expected) if key == "betas" else expected), key


def shadow_for(model, epsilon=1e-3):
    shadow = copy.deepcopy(model).double()
    configure_epsilon(shadow, epsilon)
    assert_epsilon(model, epsilon)
    assert_epsilon(shadow, epsilon)
    return shadow


def batch(model, shadow, records, epsilon=1e-3, need_grad=True):
    budget(300)
    assert_epsilon(model, epsilon)
    assert_epsilon(shadow, epsilon)
    assert all(p.dtype == torch.float32 for p in audit.names(model).values())
    assert all(p.dtype == torch.float64 for p in audit.names(shadow).values())
    shadow.load_state_dict(model.state_dict(), strict=True)
    assert_epsilon(model, epsilon)
    assert_epsilon(shadow, epsilon)
    result = [
        pilot.system_gradient(shadow, record, "FULL_SOURCE", DEVICE, need_grad)
        for record in records.values()
    ]
    assert_epsilon(shadow, epsilon)
    return (
        [r[0] for r in result],
        torch.stack([r[1] for r in result]).mean(0),
        torch.stack([r[1] for r in result]),
    )


def native_step(model, optimizer, gradient):
    assert_epsilon(model)
    assert_optimizer(optimizer, model)
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in audit.names(model).values():
        p.grad = gradient[offset : offset + p.numel()].reshape(p.shape).to(p.dtype)
        offset += p.numel()
    assert offset == 9446
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in audit.names(model).values())
    assert all(
        torch.isfinite(s[k]).all()
        for s in optimizer.state.values()
        for k in ("exp_avg", "exp_avg_sq")
    )


def save(path, model, optimizer, t):
    assert_epsilon(model)
    payload = {
        "namespace": PREFIX,
        "protocol_hash": pilot.sha(ROOT / (PREFIX + "protocol.json")),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "t": t,
        "RNG": pilot.rng(),
        "epsilon_manifest": scientific_identity(model),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": pilot.sha(path), "t": t}


def restore(path):
    # RNG ByteTensors must stay on CPU; optimizer.load_state_dict moves moments.
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if "epsilon_manifest" not in payload:
        raise ValueError("Missing required epsilon metadata")
    assert payload["namespace"] == PREFIX
    assert payload["protocol_hash"] == pilot.sha(ROOT / (PREFIX + "protocol.json"))
    manifest = payload["epsilon_manifest"]
    assert {r["name"]: r["epsilon"] for r in manifest["epsilon"]} == {
        r["name"]: (1e-3 if r["name"] in EXCHANGE else 1e-5)
        for r in manifest["epsilon"]
    }
    model = construct(payload["model"], 1e-3)
    assert scientific_identity(model) == manifest
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(payload["optimizer"])
    assert_optimizer(optimizer, model)
    assert_epsilon(model)
    shadow = shadow_for(model)
    shadow.load_state_dict(model.state_dict(), strict=True)
    assert_epsilon(shadow)
    pilot.restore_rng(payload["RNG"])
    return model, shadow, optimizer, payload


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (ROOT / (PREFIX + "protocol.json")).exists(), (
        "Do not overwrite an experiment"
    )
    source = {
        **pilot.read("protocol")["source_hashes"],
        "tools/lap_fresh_grid_pilot_v1.py": pilot.read("provenance")["driver_sha256"],
    }
    source["tools/lap_fresh_layernorm_v1.py"] = pilot.sha(
        ROOT / "tools/lap_fresh_layernorm_v1.py"
    )
    historical = {
        p.name: pilot.sha(p)
        for p in ROOT.glob("lap_fresh*_*.*")
        if p.is_file() and p.suffix in (".json", ".csv", ".md")
    }
    write(
        "protocol",
        {
            "experiment": "F026",
            "namespace": PREFIX,
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            "wall_seconds": 7200,
            "first_action_ms": int((ROOT / (PREFIX + "started.txt")).read_text()),
            "timer_note": "80-second conservative allowance includes attachment reading and initial inspection",
            "source_hashes": source,
            "historical_hashes": historical,
            "exchange_epsilon": 1e-3,
            "correlation_epsilon": 1e-5,
            "initialization": "Load original initialization.pt artifact; no new initialization or control retraining",
            "systems": ["H2", "BeH2", "CO"],
            "grid": "FULL_SOURCE only",
            "training_updates": 32,
            "preflight_temporary_steps": 1,
            "operator": "Original mean of three complete matrix losses, equal weights, unchanged F64/F32 boundary",
            "tolerances": {
                "F64_atol": 1e-10,
                "F64_rtol": 1e-9,
                "constraints_atol": 1e-12,
                "FD_adjacent_relative": 0.01,
                "independent_weighted_F32_energy_atol": 1e-6,
                "independent_weighted_F32_energy_rtol": 1e-5,
            },
            "mixed_tolerance_basis": "train_models/test_lap_operator_validation.py gradcheck atol=1e-6, rtol=1e-5; weighted F32 energy-first differentiation rounds quadrature at a different place; strict F64 identity is tested separately",
            "FD_rule": "Same fixed H2 coordinates; h0=min(1e-4*max(abs(base adjoint scalar),1)/max(abs(coordinate AD),1e-30),.02*max(abs(parameter),1e-5)); lower bound 64*F64_epsilon*max(1,abs(parameter)); factors [.125,.25,.5,1,2,4,8]. Parameter-scale cap is frozen prospectively to limit curvature, unlike prior unbounded schedule.",
            "derivative_gate": "Confirmed discrepancy, source/config failure or hard-constraint failure blocks training. A frozen FD sweep without two adjacent 1%-agreeing steps is UNRESOLVED, not a bug; it is nondecisive if independent LN first/mixed derivatives, independent energy-first mixed-precision parity, and exact graph/operator reconstruction checks pass. Technical LIMITED if any required intervention FD remains unresolved; old audit limitation is retained separately.",
            "effect_floor": 0.01,
            "optimizer": pilot.read("protocol")["optimizer"],
            "versions": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )


def authenticate():
    protocol = read("protocol")
    for name, expected in protocol["source_hashes"].items():
        assert pilot.sha(ROOT / name) == expected, name
    records = pilot.read("representations")["records"]
    for r in records.values():
        assert pilot.sha(pilot.local(r["path"])) == r["sha256"]
    assert [r["SOURCE_count"] for r in records.values()] == [47080, 89200, 85960]
    receipts = pilot.read("provenance")["pilot_checkpoint_receipts"]
    original = next(r for r in receipts if Path(r["path"]).name == "initialization.pt")
    initialization = pilot.resume_payload(original)
    state = initialization["models"]["initial"]
    native = construct(state, 1e-5)
    assert pilot.state_hash(native) == INITIAL_HASH
    assert (
        scientific_identity(native)["order"]
        == pilot.read("initialization")["parameter_order"]
    )
    control_optimizer = optimizer_for(native)
    assert not control_optimizer.state
    shadow = shadow_for(native, 1e-5)
    initial_losses, g, _per = batch(native, shadow, records, 1e-5)
    np.testing.assert_allclose(
        np.mean(initial_losses), 1.268092669580413, atol=1e-10, rtol=1e-9
    )
    gr = pilot.read("provenance")["gradient_receipts"][0]
    assert pilot.sha(pilot.local(gr["path"])) == gr["sha256"]
    with np.load(pilot.local(gr["path"])) as data:
        np.testing.assert_allclose(
            g.cpu().numpy(), data["FULL_SOURCE_gradient"], atol=1e-10, rtol=1e-9
        )
    endpoint_receipt = next(
        r for r in receipts if Path(r["path"]).name == "FULL_SOURCE_t32.pt"
    )
    endpoint = pilot.resume_payload(endpoint_receipt)
    assert endpoint["counters"]["FULL_SOURCE"] == 32
    native.load_state_dict(endpoint["models"]["FULL_SOURCE"])
    assert_epsilon(native, 1e-5)
    final_losses, _, _ = batch(native, shadow, records, 1e-5, False)
    np.testing.assert_allclose(
        np.mean(final_losses), 0.5199287840361003, atol=1e-10, rtol=1e-9
    )
    original_t0 = pilot.resume_payload(
        next(r for r in receipts if Path(r["path"]).name == "FULL_SOURCE_t0.pt")
    )
    assert all(
        not state
        for state in original_t0["optimizers"]["FULL_SOURCE"]["state"].values()
    )
    assert endpoint["optimizers"]["FULL_SOURCE"]["param_groups"][0]["lr"] == 2.5e-5
    write(
        "control_authentication",
        {
            "PASS": True,
            "initial_artifact": original,
            "control_t32": endpoint_receipt,
            "t0_losses": initial_losses,
            "t32_losses": final_losses,
            "t0_gradient_norm": float(g.norm()),
            "t0_gradient_cache_parity": "PASS",
            "t0_empty_moments": True,
            "t32_counter": 32,
            "historical_epsilon_evidence": "original constructor source + original driver hashes, same Torch version/default LayerNorm eps; explicitly reconstructed both native/shadow at 1e-5; no claim eps was encoded in old state_dict",
        },
    )
    torch.save(
        {"model": state, "RNG": original_t0["RNG"]}, OUT / "initial_tensor_source.pt"
    )
    write(
        "provenance",
        {
            "initial_artifact": original,
            "control_t32": endpoint_receipt,
            "records": records,
            "source_hashes": protocol["source_hashes"],
            "old_pre_fresh_models_loaded": 0,
        },
    )
    return records, state, original_t0["RNG"]


def statistics(model, records, theta):
    """Native LN observer hooks; semantic labelling changes no parameter use."""
    observer = audit.Observer()
    original = shadow_for(model)
    shadow = copy.deepcopy(original)
    source = textwrap.dedent(inspect.getsource(type(shadow).forward_descriptors))
    for expr, kind in [("ex", "exchange_real"), ("ue", "exchange_UEG")]:
        old = f"self.exchange_hidden({expr})"
        assert source.count(old) == 1
        source = source.replace(
            old, f'self.tagged("{kind}", self.exchange_hidden, {expr})'
        )
    namespace = {"torch": torch, "NN_OUTPUT_SCALE_PBE": pilot.NN_OUTPUT_SCALE_PBE}
    # The source method is hash-authenticated local code, never external input.
    exec(compile(source, "<F026-semantic-observer>", "exec"), namespace)  # noqa: S102
    shadow.forward_descriptors = types.MethodType(
        namespace["forward_descriptors"], shadow
    )

    def tagged(self, kind, function, *args):
        before = observer.kind
        observer.kind = kind
        try:
            return function(*args)
        finally:
            observer.kind = before

    shadow.tagged = types.MethodType(tagged, shadow)
    handles = []
    for name, module in shadow.named_modules():
        if isinstance(module, nn.LayerNorm):
            handles.append(
                module.register_forward_pre_hook(
                    lambda m, args, name=name: observer.observe(name, m, args[0])
                )
            )
    arrays = next(pilot.chunk_arrays(records["H2"], "FULL_SOURCE"))
    torch.testing.assert_close(
        pilot.chunk_matrix(original, arrays, DEVICE, False),
        pilot.chunk_matrix(shadow, arrays, DEVICE, False),
        atol=1e-10,
        rtol=1e-9,
    )
    observer.stats.clear()
    observer.samples.clear()
    for r in records.values():
        for arrays in pilot.chunk_arrays(r, "FULL_SOURCE"):
            pilot.chunk_matrix(shadow, arrays, DEVICE, False)
    for handle in handles:
        handle.remove()
    assert_epsilon(shadow)
    torch.save(observer.samples, OUT / f"hidden_samples_theta{theta}.pt")
    return [
        {
            "theta": theta,
            "kind": "hidden",
            "module": n,
            "semantic_call": kind,
            "mean": s["mean_sum"] / s["count"],
            "variance_mean": s["variance_sum"] / s["count"],
            "variance_min": s["min_variance"],
            "variance_max": s["max_variance"],
            "inverse_std_mean": s["inverse_std_sum"] / s["count"],
            "fraction_variance_le_eps": s["variance_le_eps"] / s["count"],
            "eps": s["eps"],
            "points": s["count"],
            "exact_zero_inputs": s["exact_zero_inputs"],
        }
        for (n, kind), s in observer.stats.items()
    ]


def gradient_diagnostics(model, gradient, theta):
    values = gradient.detach().cpu().numpy()
    rows = []
    for name, start, end, ln, _ in audit.layout(model):
        g = values[start:end]
        rows.append(
            {
                "theta": theta,
                "kind": "gradient",
                "module": name,
                "LayerNorm": ln,
                "gradient_norm": audit.norm(g),
                "max_abs_coordinate": float(np.max(np.abs(g))),
                "max_index": int(np.argmax(np.abs(g))),
                "squared_norm_fraction": audit.norm(g) ** 2 / audit.norm(values) ** 2,
            }
        )
    return rows


def derivative_checks(model, record, per_system, theta):
    shadow = shadow_for(model)
    parameters = dict(shadow.named_parameters())
    V = audit.full_matrix(shadow, record)
    with h5py.File(pilot.local(record["path"])) as f:
        ref, S = [
            torch.tensor(f[k][:], device=DEVICE) for k in ("ref_FULL_SOURCE", "S")
        ]
    leaf = V.detach().requires_grad_(True)
    loss = pilot.op.operator_loss(leaf, ref, S)
    adjoint = torch.autograd.grad(loss, leaf)[0].detach()
    scalar = float((V * adjoint).sum())
    h2g = per_system[0].detach().cpu().numpy()
    mapping = {n: start for n, start, _, _, _ in audit.layout(model)}
    planned = []
    for name, index in COORDINATES:
        p0, ad = (
            float(parameters[name].detach().flatten()[index]),
            float(h2g[mapping[name] + index]),
        )
        h0 = min(
            1e-4 * max(abs(scalar), 1) / max(abs(ad), 1e-30), 0.02 * max(abs(p0), 1e-5)
        )
        h0 = max(h0, 64 * np.finfo(float).eps * max(1, abs(p0)))
        planned.append(
            {
                "theta": theta,
                "name": name,
                "index": index,
                "AD": ad,
                "steps": (h0 * np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])).tolist(),
            }
        )
    write(f"FD_schedule_theta{theta}", {"planned": planned, "before_FD": True})
    selected = tuple(parameters[n] for n, _ in COORDINATES)
    independent = [torch.zeros_like(p) for p in selected]
    energy_V = torch.zeros_like(V)
    for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
        budget(1200)
        assert_epsilon(shadow)
        M = audit.energy_first_chunk(shadow, arrays, True)
        energy_V += M.detach()
        grads = torch.autograd.grad((M * adjoint).sum(), selected, allow_unused=True)
        for total, p, g in zip(independent, selected, grads):
            total += torch.zeros_like(p) if g is None else g
    torch.testing.assert_close(energy_V, V, atol=1e-6, rtol=1e-5)
    rows, statuses = [], []
    for plan, independent_gradient in zip(planned, independent):
        name, index, ad = plan["name"], plan["index"], plan["AD"]
        ig = float(independent_gradient.flatten()[index])
        np.testing.assert_allclose(ig, ad, atol=1e-6, rtol=1e-5)
        rows.append(
            {
                "theta": theta,
                "check": "independent_energy_P_parameter_derivative",
                "name": name,
                "index": index,
                "AD": ad,
                "independent": ig,
                "absolute_gap": abs(ig - ad),
                "status": "PASS",
                "tolerance": "atol1e-6 rtol1e-5; separate F32 weighted-energy path",
            }
        )
        p = parameters[name]
        p0 = float(p.detach().flatten()[index])
        results = []
        for h in plan["steps"]:
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    p.flatten()[index] = p0 + sign * h
                assert_epsilon(shadow)
                M = audit.full_matrix(shadow, record)
                values.append(float((M * adjoint).sum()))
            with torch.no_grad():
                p.flatten()[index] = p0
            fd = (values[0] - values[1]) / (2 * h)
            good = abs(fd - ad) <= 0.01 * abs(ad)
            results.append((fd, good))
            rows.append(
                {
                    "theta": theta,
                    "check": "fixed_adjoint_FD",
                    "name": name,
                    "index": index,
                    "AD": ad,
                    "FD": fd,
                    "step": h,
                    "relative_gap": abs(fd - ad) / abs(ad) if ad else None,
                    "scalar_plus": values[0],
                    "scalar_minus": values[1],
                    "status": "WITHIN_1_PERCENT"
                    if good
                    else "NO_AGREEMENT_AT_THIS_STEP",
                }
            )
        qualified = any(
            a[1] and b[1] and abs(a[0] - b[0]) <= 0.01 * abs(ad)
            for a, b in itertools.pairwise(results)
        )
        statuses.append(
            {
                "theta": theta,
                "name": name,
                "index": index,
                "status": "QUALIFIED" if qualified else "UNRESOLVED",
            }
        )
        print("derivative", statuses[-1], flush=True)
    # Native LN versus an independent biased-variance F64 formula on actual inputs.
    samples = torch.load(OUT / f"hidden_samples_theta{theta}.pt", weights_only=False)
    for (name, kind, label), (_, x, w, b, eps) in samples.items():
        x, w, b = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (x, w, b)
        ]
        probe = torch.linspace(-1, 1, len(x), dtype=torch.float64)
        a, c = F.layer_norm(x, (len(x),), w, b, eps), audit.independent_ln(x, w, b, eps)
        torch.testing.assert_close(a, c, atol=1e-10, rtol=1e-9)
        ga = torch.autograd.grad((a * probe).sum(), (x, w, b), create_graph=True)
        gc = torch.autograd.grad((c * probe).sum(), (x, w, b), create_graph=True)
        for u, v in zip(ga, gc):
            torch.testing.assert_close(u, v, atol=1e-10, rtol=1e-9)
        ma = torch.autograd.grad((ga[0] * probe).sum(), w)[0]
        mc = torch.autograd.grad((gc[0] * probe).sum(), w)[0]
        torch.testing.assert_close(ma, mc, atol=1e-10, rtol=1e-9)
        rows.append(
            {
                "theta": theta,
                "check": "independent_LN_forward_first_mixed",
                "name": name,
                "semantic_call": kind,
                "sample": label,
                "eps": eps,
                "status": "PASS",
            }
        )
    assert_epsilon(model)
    assert_epsilon(shadow)
    return rows, statuses


def preflight():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert not (ROOT / (PREFIX + "preflight.json")).exists()
    records, state, saved_rng = authenticate()
    model = construct(state, 1e-5)
    before = scientific_identity(model)
    configure_epsilon(model)
    identity = scientific_identity(model)
    assert identity["tensor_hash"] == before["tensor_hash"] == INITIAL_HASH
    assert identity["scientific_hash"] != before["scientific_hash"]
    assert identity["coordinates"] == 9446
    write(
        "epsilon_manifest",
        {
            "control": before,
            "intervention": identity,
            "only_nonstate_change": list(EXCHANGE),
        },
    )
    optimizer = optimizer_for(model)
    assert not optimizer.state
    pilot.restore_rng(saved_rng)
    initial_receipt = save(OUT / "theta0.pt", model, optimizer, 0)
    shadow = shadow_for(model)
    arrays, metrics, diagnostic, derivatives, qualifications, constraint = (
        {},
        [],
        [],
        [],
        [],
        [],
    )
    for t in (0, 1):
        tic = time.monotonic()
        values, g, per = batch(model, shadow, records)
        seconds = time.monotonic() - tic
        arrays[f"g{t + 1}"] = g.cpu().numpy()
        arrays[f"system_gradients_theta{t}"] = per.cpu().numpy()
        arrays[f"parameters_theta{t}"] = pilot.vector(model)
        metrics.append(
            {
                "theta": t,
                "mean_loss": float(np.mean(values)),
                "losses": values,
                "gradient_norm": float(g.norm()),
                "batch_seconds": seconds,
            }
        )
        diagnostic += gradient_diagnostics(model, g, t)
        diagnostic += statistics(model, records, t)
        check = pilot.constraints(model)
        assert check["PASS"]
        constraint.append({"theta": t, **check})
        drows, q = derivative_checks(model, records["H2"], per, t)
        derivatives += drows
        qualifications += q
        table("derivative_checks", derivatives)
        print("preflight", metrics[-1], flush=True)
        if t == 0:
            native_step(model, optimizer, g)
        else:
            save(OUT / "preflight_theta1.pt", model, optimizer, 1)
            restored, rs, ro, _payload = restore(OUT / "preflight_theta1.pt")
            rv, rg, _ = batch(restored, rs, records)
            np.testing.assert_allclose(rv, values, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
            assert pilot.state_hash(restored) == pilot.state_hash(model)
            assert all(int(s["step"]) == 1 for s in ro.state.values())
    # Do not retain the temporary step as a head start.
    model, shadow, optimizer, _payload = restore(OUT / "theta0.pt")
    assert not optimizer.state
    assert pilot.state_hash(model) == INITIAL_HASH
    assert torch.equal(torch.get_rng_state(), saved_rng[2])
    projected = 32 * max(m["batch_seconds"] for m in metrics) * 1.3 + 600
    budget(projected)
    np.savez(OUT / "preflight_arrays.npz", **arrays)
    table("gradient_diagnostics", diagnostic)
    write("constraint_checks", {"states": constraint})
    write(
        "shadow_parity",
        {
            "native_shadow_eps_assertions": "PASS before every batch, after synchronization, and in temporary derivative models",
            "restored_theta1_full_batch_loss_gradient_parity": "PASS",
            "restored_theta0_model_optimizer_RNG": "PASS",
            "eps_is_part_of_scientific_identity": True,
        },
    )
    write(
        "preflight",
        {
            "PASS": True,
            "technical": "VALID"
            if all(q["status"] == "QUALIFIED" for q in qualifications)
            else "LIMITED",
            "derivative": qualifications,
            "independent_analytical_and_operator_path_checks": "PASS",
            "temporary_native_steps": 1,
            "temporary_step_retained": False,
            "metrics": metrics,
            "initial_checkpoint": initial_receipt,
            "projected_training_plus_endpoint_seconds": projected,
            "wall_before_training": elapsed(),
        },
    )


def recover_preflight():
    """Recover the failed RNG restore check; NEVER execute another temporary step."""
    assert not (ROOT / (PREFIX + "preflight.json")).exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    path = OUT / "preflight_theta1.pt"
    assert path.exists() and not (OUT / "resume.pt").exists()
    derivative_file = ROOT / (PREFIX + "derivative_checks.csv")
    failure = {
        "initial_result": "FAIL",
        "failure": "torch.set_rng_state rejected RNG tensor moved to CUDA by map_location=DEVICE",
        "location": "isolated experimental restore helper",
        "production_modified": False,
        "scientific_training_started": False,
        "temporary_steps_already_executed": 1,
        "fix": "load checkpoint on CPU; construct moves parameters and optimizer.load_state_dict restores device-correct moments",
        "existing_derivative_table_sha256": pilot.sha(derivative_file),
        "saved_theta1_sha256": pilot.sha(path),
        "additional_temporary_steps": 0,
    }
    write("preflight_restore_failure", failure)
    derivative_rows = list(csv.DictReader(derivative_file.open()))
    qualifications = []
    for t in (0, 1):
        for name, index in COORDINATES:
            selected = [
                r
                for r in derivative_rows
                if r["check"] == "fixed_adjoint_FD"
                and int(r["theta"]) == t
                and r["name"] == name
            ]
            assert len(selected) == 7
            ad = float(selected[0]["AD"])
            good = any(
                a["status"] == b["status"] == "WITHIN_1_PERCENT"
                and abs(float(a["FD"]) - float(b["FD"])) <= 0.01 * abs(ad)
                for a, b in itertools.pairwise(selected)
            )
            qualifications.append(
                {
                    "theta": t,
                    "name": name,
                    "index": index,
                    "status": "QUALIFIED" if good else "UNRESOLVED",
                }
            )
    assert all(
        r["status"] == "PASS"
        for r in derivative_rows
        if r["check"] != "fixed_adjoint_FD"
    )
    records = read("provenance")["records"]
    arrays, metrics, diagnostics, constraints = {}, [], [], []
    for t, filename in [(0, "theta0.pt"), (1, "preflight_theta1.pt")]:
        model, shadow, optimizer, payload = restore(OUT / filename)
        assert payload["t"] == t
        assert (
            ({int(s["step"]) for s in optimizer.state.values()} == {1})
            if t
            else not optimizer.state
        )
        tic = time.monotonic()
        values, g, per = batch(model, shadow, records)
        seconds = time.monotonic() - tic
        arrays[f"g{t + 1}"] = g.cpu().numpy()
        arrays[f"system_gradients_theta{t}"] = per.cpu().numpy()
        arrays[f"parameters_theta{t}"] = pilot.vector(model)
        for name, index in COORDINATES:
            start = next(r[1] for r in audit.layout(model) if r[0] == name)
            saved = next(
                r
                for r in derivative_rows
                if r["check"] == "fixed_adjoint_FD"
                and int(r["theta"]) == t
                and r["name"] == name
            )
            np.testing.assert_allclose(
                float(per[0, start + index]), float(saved["AD"]), atol=1e-10, rtol=1e-9
            )
        metrics.append(
            {
                "theta": t,
                "mean_loss": float(np.mean(values)),
                "losses": values,
                "gradient_norm": float(g.norm()),
                "batch_seconds": seconds,
            }
        )
        diagnostics += gradient_diagnostics(model, g, t)
        diagnostics += statistics(model, records, t)
        checks = pilot.constraints(model)
        assert checks["PASS"]
        constraints.append({"theta": t, **checks})
        restored, rs, ro, again = restore(OUT / filename)
        assert (
            ({int(s["step"]) for s in ro.state.values()} == {1}) if t else not ro.state
        )
        rv, rg, _ = batch(restored, rs, records)
        np.testing.assert_allclose(rv, values, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
        assert pilot.state_hash(restored) == pilot.state_hash(model)
        assert torch.equal(torch.get_rng_state(), again["RNG"][2])
        print("restoration recovery", metrics[-1], flush=True)
    model, shadow, optimizer, payload = restore(OUT / "theta0.pt")
    assert not optimizer.state and pilot.state_hash(model) == INITIAL_HASH
    projected = 32 * max(r["batch_seconds"] for r in metrics) * 1.3 + 650
    budget(projected)
    np.savez(OUT / "preflight_arrays.npz", **arrays)
    table("gradient_diagnostics", diagnostics)
    write("constraint_checks", {"states": constraints})
    write(
        "shadow_parity",
        {
            "native_shadow_eps_assertions": "PASS",
            "restored_theta1_full_batch_loss_gradient_parity": "PASS",
            "restored_theta0_model_optimizer_RNG": "PASS",
            "initial_RNG_restore_failure": "retained separately, fixed and requalified before training",
        },
    )
    write(
        "preflight",
        {
            "PASS": True,
            "technical": "VALID"
            if all(r["status"] == "QUALIFIED" for r in qualifications)
            else "LIMITED",
            "derivative": qualifications,
            "independent_analytical_and_operator_path_checks": "PASS",
            "temporary_native_steps": 1,
            "temporary_step_retained": False,
            "recovery_extra_steps": 0,
            "metrics": metrics,
            "projected_training_plus_endpoint_seconds": projected,
            "wall_before_training": elapsed(),
        },
    )


def train():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert read("preflight")["PASS"]
    records = read("provenance")["records"]
    history, checkpoints, cache, metrics, diagnostics = [], [], {}, [], []
    if (OUT / "resume.pt").exists():
        model, shadow, optimizer, payload = restore(OUT / "resume.pt")
        start = payload["t"]
        assert 0 < start <= 32
        history = list(csv.DictReader((ROOT / (PREFIX + "training_log.csv")).open()))
        assert int(history[-1]["counter"]) == start, (
            "Cursor/log mismatch; preserve state and stop, never repeat update"
        )
        metrics = list(
            csv.DictReader((ROOT / (PREFIX + "checkpoint_metrics.csv")).open())
        )
        with np.load(OUT / "training_arrays.npz") as saved:
            cache = {key: saved[key] for key in saved.files}
        for t in (0, 1, 2, 16, 32):
            path = OUT / f"theta{t}.pt"
            if path.exists() and t <= start:
                checkpoints.append(
                    {"path": str(path), "sha256": pilot.sha(path), "t": t}
                )
        theta0 = cache["parameters_theta0"]
    else:
        model, shadow, optimizer, payload = restore(OUT / "theta0.pt")
        start = 0
        theta0 = pilot.vector(model)
    for t in range(start, 33):
        budget(300 + (32 - t) * read("preflight")["metrics"][0]["batch_seconds"] * 1.1)
        tic = time.monotonic()
        values, g, _per = batch(model, shadow, records)
        theta = pilot.vector(model)
        if t in (0, 1, 2, 16, 32):
            if t == 0:
                np.testing.assert_allclose(
                    values,
                    read("preflight")["metrics"][0]["losses"],
                    atol=1e-10,
                    rtol=1e-9,
                )
                with np.load(OUT / "preflight_arrays.npz") as preflight_arrays:
                    np.testing.assert_allclose(
                        g.cpu().numpy(), preflight_arrays["g1"], atol=1e-10, rtol=1e-9
                    )
            checkpoints.append(save(OUT / f"theta{t}.pt", model, optimizer, t))
            metrics.append(
                {
                    "theta": t,
                    "mean_loss": float(np.mean(values)),
                    "H2_loss": values[0],
                    "BeH2_loss": values[1],
                    "CO_loss": values[2],
                    "max_system_loss": max(values),
                    "gradient_norm": float(g.norm()),
                    "displacement_from_theta0": audit.norm(theta - theta0),
                }
            )
            cache[f"g{t + 1}"] = g.cpu().numpy()
            cache[f"parameters_theta{t}"] = theta
            diagnostics += gradient_diagnostics(model, g, t)
            if t == 1:
                restored, rs, ro, _ = restore(OUT / "theta1.pt")
                rv, rg, _ = batch(restored, rs, records)
                np.testing.assert_allclose(rv, values, atol=1e-10, rtol=1e-9)
                torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
                assert pilot.state_hash(restored) == pilot.state_hash(model)
                assert {int(s["step"]) for s in ro.state.values()} == {1}
                model, shadow, optimizer = restored, rs, ro
                write(
                    "training_theta1_restoration",
                    {
                        "PASS": True,
                        "epsilon_and_full_batch_loss_gradient_parity": "PASS",
                        "optimizer_counter": 1,
                        "moments_preserved": True,
                    },
                )
            if t in (2, 32):
                diagnostics += statistics(model, records, t)
            if t == 32:
                restored, rs, _, _ = restore(OUT / "theta32.pt")
                rv, rg, _ = batch(restored, rs, records)
                np.testing.assert_allclose(rv, values, atol=1e-10, rtol=1e-9)
                torch.testing.assert_close(rg, g, atol=1e-10, rtol=1e-9)
                constraints = read("constraint_checks")
                checks = pilot.constraints(restored)
                assert checks["PASS"]
                constraints["states"].append({"theta": 32, **checks})
                write("constraint_checks", constraints)
        if t == 32:
            break
        current = g.cpu().numpy()
        native_step(model, optimizer, g)
        counters = {int(s["step"]) for s in optimizer.state.values()}
        assert counters == {t + 1}
        delta = pilot.vector(model) - theta
        history.append(
            {
                "update": t + 1,
                "counter": t + 1,
                "mean_preupdate_loss": float(np.mean(values)),
                "H2_loss": values[0],
                "BeH2_loss": values[1],
                "CO_loss": values[2],
                "gradient_norm": float(g.norm()),
                "displacement_norm": audit.norm(delta),
                "loss_directional_dot": float(current @ delta),
                "displacement_from_t0": audit.norm(pilot.vector(model) - theta0),
                "first_moment_norm": float(
                    torch.cat(
                        [
                            s["exp_avg"].flatten().double()
                            for s in optimizer.state.values()
                        ]
                    ).norm()
                ),
                "second_moment_norm": float(
                    torch.cat(
                        [
                            s["exp_avg_sq"].flatten().double()
                            for s in optimizer.state.values()
                        ]
                    ).norm()
                ),
                "seconds": time.monotonic() - tic,
            }
        )
        save(OUT / "resume.pt", model, optimizer, t + 1)
        table("training_log", history)
        table("checkpoint_metrics", metrics)
        np.savez(OUT / "training_arrays.npz", **cache)
        print("update", history[-1], flush=True)
    table("checkpoint_metrics", metrics)
    old = list(csv.DictReader((ROOT / (PREFIX + "gradient_diagnostics.csv")).open()))
    for r in old:
        r.setdefault("stage", "preflight")
    for r in diagnostics:
        r["stage"] = "trajectory"
    table("gradient_diagnostics", old + diagnostics)
    np.savez(OUT / "training_arrays.npz", **cache)
    write(
        "checkpoint_receipts",
        {
            "checkpoints": checkpoints,
            "final_restoration_full_batch_parity": "PASS",
            "updates": 32,
        },
    )


def memory_and_comparison():
    """Read-only accounting of actual updates 1/2 and unapplied proposal 33."""
    with np.load(OUT / "training_arrays.npz") as saved:
        data = {k: saved[k] for k in saved.files}
    model, _, _, _ = restore(OUT / "theta0.pt")
    ordering = audit.layout(model)
    first = data["g1"].astype(np.float32).astype(np.float64)
    rows = []
    for t in (1, 2, 32):
        payload = torch.load(
            OUT / f"theta{t}.pt", weights_only=False, map_location="cpu"
        )
        m, v, counter = audit.flatten_moments(payload, ordering)
        assert counter == t
        mf, vf = audit.first_components(first, t)
        mr, vr = m.double().numpy() - mf, v.double().numpy() - vf
        last_gradient = (
            data[f"g{t}"].astype(np.float32).astype(np.float64) if t in (1, 2) else None
        )
        state_gradient = data[f"g{t + 1}"]
        rounding_bound = (
            (t + 4) * np.finfo(np.float32).eps * np.maximum(vf, v.double().numpy())
        )
        relative_failure = vr < -rounding_bound
        # Preserve the relative-only FAIL. Subnormal F32 has absolute, not
        # relative, resolution; this is checked by exact representability.
        minimum_step = float(np.nextafter(np.float32(0), np.float32(1)))
        subnormal_nearest = (
            (np.maximum(vf, v.double().numpy()) < np.finfo(np.float32).tiny)
            & (np.abs(vr) <= minimum_step / 2)
            & (vf.astype(np.float32).astype(np.float64) == v.double().numpy())
        )
        assert not np.any(relative_failure & ~subnormal_nearest), (
            "Unexplained negative second-moment remainder"
        )
        for name, start, end, ln, _ in ordering:
            ix = slice(start, end)
            positive = v.numpy()[ix] >= np.finfo(np.float32).tiny
            ratios = vf[ix][positive] / v.double().numpy()[ix][positive]
            rows.append(
                {
                    "kind": "stored_moment",
                    "t": t,
                    "module": name,
                    "LayerNorm": ln,
                    "m_first_norm": audit.norm(mf[ix]),
                    "m_rest_norm": audit.norm(mr[ix]),
                    "m_total_norm": audit.norm(m.double().numpy()[ix]),
                    "last_update_gradient_index": t
                    if t in (1, 2)
                    else "g32 not retained",
                    "last_gradient_m_contribution_norm": audit.norm(
                        0.1 * last_gradient[ix]
                    )
                    if last_gradient is not None
                    else None,
                    "last_gradient_v_contribution_norm": audit.norm(
                        0.001 * last_gradient[ix] ** 2
                    )
                    if last_gradient is not None
                    else None,
                    "gradient_at_state_index": t + 1,
                    "cosine_first_state_gradient": audit.cosine(
                        mf[ix], state_gradient[ix]
                    ),
                    "cosine_rest_state_gradient": audit.cosine(
                        mr[ix], state_gradient[ix]
                    ),
                    "v_first_norm": audit.norm(vf[ix]),
                    "v_rest_norm": audit.norm(vr[ix]),
                    "v_total_norm": audit.norm(v.double().numpy()[ix]),
                    "v_first_total_ratio_median": float(np.median(ratios))
                    if len(ratios)
                    else None,
                    "negative_v_rest": int((vr[ix] < 0).sum()),
                    "min_v_rest": float(vr[ix].min()),
                    "negative_v_outside_rounding_bound": int(
                        (vr[ix] < -rounding_bound[ix]).sum()
                    ),
                    "explained_by_exact_subnormal_nearest_rounding": int(
                        (relative_failure[ix] & subnormal_nearest[ix]).sum()
                    ),
                    "v_ratio_subnormal_coordinates_excluded": int(
                        ((v.numpy()[ix] > 0) & ~positive).sum()
                    ),
                }
            )
        np.savez(
            OUT / f"moment_t{t}.npz",
            m=m.numpy(),
            v=v.numpy(),
            m_first=mf,
            m_rest=mr,
            v_first=vf,
            v_rest=vr,
        )
    summaries = []
    for t in (1, 2, 33):
        base = t - 1
        payload = torch.load(
            OUT / f"theta{base}.pt", weights_only=False, map_location="cpu"
        )
        if base:
            m, v, counter = audit.flatten_moments(payload, ordering)
            assert counter == base
        else:
            m = v = torch.zeros(9446, dtype=torch.float32)
        theta = torch.tensor(
            data[f"parameters_theta{base}"], dtype=torch.float32, device=DEVICE
        )
        g = data[f"g{t}"]
        proposed, mn, vn, den = audit.proposal(
            theta,
            m.to(DEVICE),
            v.to(DEVICE),
            torch.tensor(g, dtype=torch.float32, device=DEVICE),
            t,
        )
        if t < 3:
            np.testing.assert_array_equal(
                proposed.detach().cpu().numpy(),
                data[f"parameters_theta{t}"].astype(np.float32),
            )
            actual = data[f"parameters_theta{t}"] - data[f"parameters_theta{base}"]
        else:
            actual = (proposed.double() - theta.double()).cpu().numpy()
        mf, _ = audit.first_components(first, t)
        scale = -2.5e-5 / (1 - 0.9**t) / den.double().cpu().numpy()
        first_delta = scale * mf
        rest_delta = scale * (mn.double().cpu().numpy() - mf)
        decay = (
            (theta.clone().mul_(1 - 2.5e-5 * 0.01).double() - theta.double())
            .cpu()
            .numpy()
        )
        rounding = actual - first_delta - rest_delta - decay
        component = {
            "first_gradient": first_delta,
            "rest": rest_delta,
            "decay": decay,
            "rounding": rounding,
            "total": actual,
        }
        for name, vector in component.items():
            rows.append(
                {
                    "kind": "update_direction",
                    "t": t,
                    "module": "ALL",
                    "component": name,
                    "displacement_norm": audit.norm(vector),
                    "loss_directional_dot": float(g @ vector),
                    "gradient_cosine": audit.cosine(g, vector),
                    "applied": t < 3,
                }
            )
            for block, select in [
                ("exchange_LN", lambda r: r[3] and r[0].startswith("x_")),
                ("other_exchange", lambda r: not r[3] and r[0].startswith("x_")),
                ("correlation", lambda r: r[0].startswith("c_")),
            ]:
                ix = np.concatenate(
                    [np.arange(r[1], r[2]) for r in ordering if select(r)]
                )
                rows.append(
                    {
                        "kind": "update_direction",
                        "t": t,
                        "module": block,
                        "component": name,
                        "displacement_norm": audit.norm(vector[ix]),
                        "loss_directional_dot": float(g[ix] @ vector[ix]),
                        "gradient_cosine": audit.cosine(g[ix], vector[ix]),
                        "applied": t < 3,
                    }
                )
        summaries.append(
            {
                "update": t,
                "first_dot": float(g @ first_delta),
                "rest_dot": float(g @ rest_delta),
                "total_dot": float(g @ actual),
                "total_norm": audit.norm(actual),
                "first_norm": audit.norm(first_delta),
                "rest_norm": audit.norm(rest_delta),
                "applied": t < 3,
            }
        )
        np.savez(
            OUT / f"update{t}_components.npz",
            **component,
            m=mn.cpu().numpy(),
            v=vn.cpu().numpy(),
            denominator=den.cpu().numpy(),
        )
    table("adamw_memory", rows)
    metrics = list(csv.DictReader((ROOT / (PREFIX + "checkpoint_metrics.csv")).open()))
    by_t = {int(r["theta"]): r for r in metrics}
    control = read("control_authentication")
    ci, cf = float(np.mean(control["t0_losses"])), float(np.mean(control["t32_losses"]))
    ni, nf = float(by_t[0]["mean_loss"]), float(by_t[32]["mean_loss"])
    final = {
        "control_initial_loss": ci,
        "control_final_loss": cf,
        "intervention_initial_loss": ni,
        "intervention_final_loss": nf,
        "initial_epsilon_effect": ni - ci,
        "final_absolute_delta": nf - cf,
        "relative_final_improvement_B": (cf - nf) / cf,
        "effect_floor_threshold": 0.99 * cf,
        "effect_floor_pass": nf <= 0.99 * cf,
        "control_reduction_from_own_start": ci - cf,
        "intervention_reduction_from_own_start": ni - nf,
        "startup_gradient_reduction_factor": control["t0_gradient_norm"]
        / float(by_t[0]["gradient_norm"]),
        "startup_gradient_relative_reduction": 1
        - float(by_t[0]["gradient_norm"]) / control["t0_gradient_norm"],
        "updates": 32,
        "temporary_preflight_steps": 1,
        "memory": summaries,
        "weight_decay_accounting": "actual native F32 multiplicative displacement; remaining add/update rounding is separate",
        "proposal33_applied": False,
        "systems": [
            {
                "system": name,
                "control_initial": control["t0_losses"][i],
                "control_final": control["t32_losses"][i],
                "intervention_initial": float(by_t[0][name + "_loss"]),
                "intervention_final": float(by_t[32][name + "_loss"]),
                "final_delta": float(by_t[32][name + "_loss"])
                - control["t32_losses"][i],
            }
            for i, name in enumerate(["H2", "BeH2", "CO"])
        ],
        "intervention_max_final_system_loss": float(by_t[32]["max_system_loss"]),
        "control_max_final_system_loss": max(control["t32_losses"]),
        "realized_step1_loss_change": float(by_t[1]["mean_loss"]) - ni,
        "realized_step2_loss_change": float(by_t[2]["mean_loss"])
        - float(by_t[1]["mean_loss"]),
    }
    write("final_comparison", final)
    return final


def rounding_diagnostic():
    data = np.load(OUT / "training_arrays.npz")
    model, _, _, _ = restore(OUT / "theta0.pt")
    ordering = audit.layout(model)
    first = data["g1"].astype(np.float32).astype(np.float64)
    findings = []
    for t in (1, 2, 32):
        payload = torch.load(
            OUT / f"theta{t}.pt", weights_only=False, map_location="cpu"
        )
        _, v, _ = audit.flatten_moments(payload, ordering)
        _, vf = audit.first_components(first, t)
        total = v.double().numpy()
        rest = total - vf
        bound = (t + 4) * np.finfo(np.float32).eps * np.maximum(vf, total)
        indices = np.flatnonzero(rest < -bound)
        findings.append(
            {
                "t": t,
                "relative_bound_failure_count": len(indices),
                "maximum_failure_absolute": float(np.max(np.abs(rest[indices])))
                if len(indices)
                else 0,
                "examples": [
                    {
                        "flat_index": int(i),
                        "g1_native": float(first[i]),
                        "v_first_F64": float(vf[i]),
                        "v_total_native": float(total[i]),
                        "rest": float(rest[i]),
                        "relative_bound": float(bound[i]),
                    }
                    for i in indices[:12]
                ],
            }
        )
    write(
        "rounding_diagnostic",
        {
            "original_relative_only_guard": "FAIL preserved",
            "F32_minimum_normal": float(np.finfo(np.float32).tiny),
            "findings": findings,
        },
    )
    print(json.dumps(findings, indent=2), flush=True)


def coupling_check():
    """Read-only t0 learned-output effect, not another operator-loss census."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    payload = torch.load(
        OUT / "initial_tensor_source.pt", weights_only=False, map_location="cpu"
    )
    control = construct(payload["model"], 1e-5).double()
    intervention = construct(payload["model"], 1e-3).double()
    assert pilot.state_hash(control) == pilot.state_hash(intervention)
    result = []
    for name, record in read("provenance")["records"].items():
        n, square, maximum = 0, np.zeros(29), np.zeros(29)
        for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
            features = torch.tensor(arrays["features"], device=DEVICE)
            rho = features[:, :2]
            grad = features[:, 2:8].reshape(-1, 2, 3)
            sigma = pilot.op.sigma_standard_to_total(
                pilot.op.sigma_from_gradients(grad)
            )
            raw = torch.cat([rho, sigma, torch.zeros_like(rho), features[:, 8:]], -1)
            with torch.no_grad():
                difference = (intervention(raw) - control(raw)).cpu().numpy()
            n += len(difference)
            square += np.square(difference).sum(0)
            maximum = np.maximum(maximum, np.abs(difference).max(0))
        result.append(
            {
                "system": name,
                "points": n,
                "per_output_RMS_difference": np.sqrt(square / n).tolist(),
                "per_output_max_abs_difference": maximum.tolist(),
                "exchange_max": float(maximum[22:28].max()),
                "correlation_max": float(maximum[[0, 1, 28]].max()),
            }
        )
    assert all(r["exchange_max"] > 0 and r["correlation_max"] > 0 for r in result)
    write(
        "coupling_check",
        {
            "PASS": True,
            "scope": "same t0 tensors; complete three SOURCE grids; learned outputs only",
            "output_indices": {
                "exchange": list(range(22, 28)),
                "correlation": [0, 1, 28],
            },
            "systems": result,
        },
    )


def finish():
    final = memory_and_comparison()
    control_authentication = read("control_authentication")
    control_t0_receipt = next(
        r
        for r in pilot.read("provenance")["pilot_checkpoint_receipts"]
        if Path(r["path"]).name == "FULL_SOURCE_t0.pt"
    )
    for receipt in [control_authentication["control_t32"], control_t0_receipt]:
        payload = pilot.resume_payload(receipt)
        group = payload["optimizers"]["FULL_SOURCE"]["param_groups"][0]
        for key, expected in pilot.read("protocol")["optimizer"].items():
            assert (tuple(group[key]) if key == "betas" else group[key]) == (
                tuple(expected) if key == "betas" else expected
            )
    control_authentication["actual_t0_t32_optimizer_hyperparameters"] = (
        "PASS: lr/betas/eps/weight_decay/foreach match the frozen original policy"
    )
    write("control_authentication", control_authentication)
    # The protocol explicitly permits endpoint epsilon swaps on read-only copies.
    # They never select another model and do not enter the primary classification.
    records = read("provenance")["records"]
    for record in records.values():
        assert pilot.sha(pilot.local(record["path"])) == record["sha256"]
    control_payload = pilot.resume_payload(control_authentication["control_t32"])
    intervention_payload = torch.load(
        OUT / "theta32.pt", weights_only=False, map_location="cpu"
    )
    swaps = []
    for label, state, epsilon in [
        ("control_t32_weights", control_payload["models"]["FULL_SOURCE"], 1e-3),
        ("intervention_t32_weights", intervention_payload["model"], 1e-5),
    ]:
        model = construct(state, epsilon)
        shadow = shadow_for(model, epsilon)
        values, _, _ = batch(model, shadow, records, epsilon, False)
        swaps.append(
            {
                "weights": label,
                "exchange_epsilon": epsilon,
                "system_losses": values,
                "mean_loss": float(np.mean(values)),
                "read_only": True,
            }
        )
    final["optional_endpoint_epsilon_swaps"] = swaps
    write("final_comparison", final)
    epsilon_manifest = read("epsilon_manifest")
    epsilon_manifest["checkpoint_parameter_buffer_hashes"] = []
    parameter_names = {
        row["name"] for row in pilot.read("initialization")["parameter_order"]
    }
    for t in (0, 1, 2, 16, 32):
        path = OUT / f"theta{t}.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        hashes = {}
        for kind in ("parameters", "buffers"):
            h = hashlib.sha256()
            for key, value in payload["model"].items():
                if (key in parameter_names) == (kind == "parameters"):
                    a = value.detach().contiguous().numpy()
                    h.update(key.encode())
                    h.update(str(a.dtype).encode())
                    h.update(str(a.shape).encode())
                    h.update(a.tobytes())
            hashes[kind + "_sha256"] = h.hexdigest()
        epsilon_manifest["checkpoint_parameter_buffer_hashes"].append(
            {
                "t": t,
                "checkpoint_sha256": pilot.sha(path),
                "scientific_hash": payload["epsilon_manifest"]["scientific_hash"],
                **hashes,
            }
        )
    write("epsilon_manifest", epsilon_manifest)
    preflight = read("preflight")
    classification = (
        "A"
        if final["effect_floor_pass"]
        else "C"
        if final["final_absolute_delta"] > 0
        else "B"
    )
    for name, expected in {
        **read("protocol")["source_hashes"],
        **read("protocol")["historical_hashes"],
    }.items():
        assert pilot.sha(ROOT / name) == expected, name
    provenance = read("provenance")
    provenance["new_artifacts"] = [
        {"path": str(p), "sha256": pilot.sha(p)}
        for p in sorted(OUT.iterdir())
        if p.is_file()
    ]
    provenance["new_driver_sha256"] = pilot.sha(Path(__file__))
    provenance["historical_and_production_sources_unchanged"] = True
    write("provenance", provenance)
    decision = {
        "experiment": "F026",
        "scientific_classification": classification,
        "technical_status": preflight["technical"],
        "derivative_qualifications": preflight["derivative"],
        "updates": 32,
        "temporary_preflight_steps": 1,
        "new_training_arms": 1,
        "historical_control_retrained": False,
        "effect_floor_pass": final["effect_floor_pass"],
        "B": final["relative_final_improvement_B"],
        "scientific_wall_seconds": elapsed(),
        "budget_respected": elapsed() < 7200,
        "interpretation": "Conditioning, early optimizer displacement, and absolute final loss are evaluated separately; exchange-hidden coupling can also change correlation outputs.",
        "next_experiment": "One separately frozen second-seed matched replication of this exact eps comparison on the same three systems, with the seed fixed prospectively and no epsilon sweep, before any full90 or chemistry claim. No new seed was selected or executed here."
        if classification == "A"
        else "Close this eps=1e-3 intervention at the declared horizon; no additional training follows. The remaining concrete question is whether its endpoint loss difference comes from normalization representation or learned weights; a separately authorized fixed-weight epsilon-swap diagnostic would isolate that gap.",
    }
    write("decision", decision)
    systems = "\n".join(
        f"| {r['system']} | {r['control_initial']:.12g} | {r['intervention_initial']:.12g} | {r['control_final']:.12g} | {r['intervention_final']:.12g} | {r['final_delta']:+.9g} |"
        for r in final["systems"]
    )
    updates = "\n".join(
        f"| {r['update']} | {r['first_dot']:+.10g} | {r['rest_dot']:+.10g} | {r['total_dot']:+.10g} | {r['applied']} |"
        for r in final["memory"]
    )
    report = f"""# F026 — Matched exchange LayerNorm epsilon pilot

Scientific classification **{classification}**; technical **{preflight["technical"]}**. One 32-update intervention completed. Historical FULL_SOURCE control was authenticated, never retrained. This is a single-seed, three-training-system experiment, not chemical or full90 qualification.

## Configuration and control

Only `x_feature_extractor.1`, `.3.fc.1`, `.3.fc.4`, `.4.fc.1`, `.4.fc.4` LayerNorm eps change from 1e-5 to 1e-3. Every module is checked by identity/type; correlation LN eps remains 1e-5. Native F32, F64 shadow, derivative copies and restored models are explicitly configured and independently asserted before every batch and after every tensor synchronization. `_model_double_operator_partials` executes the configured shadow through functional_call; eps remains a Python module attribute. Tensor identity alone is never used as scientific identity.

Original initialization.pt bytes were SHA-authenticated; loaded tensor state has hash `{INITIAL_HASH}`, original parameter order and 9446 unique coordinates. Constructor calls allocate storage only; their random values are discarded and RNG restored. No new scientific initialization was used. Weights and buffers remain exactly equal after epsilon configuration. The epsilon manifest records every LN, architecture config, order, precision and tensor/scientific hashes. Exchange hidden features feed correlation, so the intervention is not claimed to affect only exchange energy.

Control t0 loss/gradient and t32 loss were recomputed once on the original grids/configuration. Values reproduce {final["control_initial_loss"]:.16g} and {final["control_final_loss"]:.16g}; g1 cache parity passes. Old checkpoints did not store eps: reconstruction is supported by the unchanged original driver/constructor hashes and matching Torch version, not invented checkpoint metadata. New checkpoints require explicit eps metadata and reject missing configuration. Actual training theta1 and theta32 checkpoint restores reproduce full-batch losses/gradients, optimizer counters and epsilon; preflight additionally restores model/shadow/empty optimizer/RNG to t0. The first preflight restore FAIL (CPU RNG ByteTensor moved to CUDA) is retained in preflight_restore_failure.json. The isolated driver now loads payloads on CPU, restores optimizer devices, and was requalified from existing theta0/theta1 without another temporary update. Production was not repaired. An initial pytest attempt encountered an unwritable default temporary directory; the identical tests pass with a dedicated workspace basetemp.

## Data, loss, precision and preflight

Only original FULL_SOURCE H2/BeH2/CO caches: 47080/89200/85960 points. Original DM, source precision, overlap, targets, gauge and matrix metric remain unchanged. The original two-pass bounded-memory adjoint sums complete matrices before the squared loss. The three system losses/gradients receive equal 1/3 weight. Learned forward/gradient and AO accumulation use F64; original PBE local arithmetic stays F32; the mean gradient is cast once to native F32 AdamW. AdamW lr2.5e-5, betas .9/.999, eps1e-8, decay.01, foreach false. No chemistry/energy tasks, clipping, scheduler or moment reset.

A separate reversible first preflight update was executed and discarded. It is not one of the 32 retained training updates. The intervention then restarted from the exact t0 weights, empty optimizer and saved RNG. Thus 33 molecular optimizer executions occurred: 1 temporary qualification + 32 declared intervention updates, one arm. theta0/1/2/16/32 plus an atomic resume checkpoint are saved outside Git with SHA receipts.

Preflight full-batch profiles: {[m["batch_seconds"] for m in preflight["metrics"]]} seconds. Reserved training-plus-endpoint projection was {preflight["projected_training_plus_endpoint_seconds"]:.3f} seconds; budget was checked throughout. Final checkpoint restoration includes a repeated full-batch gradient check. Phase timings report active scientific process wall separately from total workflow wall; neither is advertised as metered GPU-active time.

Fixed H2 coordinates remain `.1.bias[6]`, `.3.fc.1.bias[16]`, `.3.fc.0.weight[887]`. Before numerical results, the prior scalar-change FD rule was frozen with a prospective parameter-scale cap to limit curvature. Seven symmetric h values per coordinate/state; unchanged requirement of two adjacent FD values within 1% of AD and each other. No post-result perturbation or tolerance changes. Statuses: {preflight["derivative"]}. Independent LN formula uses actual eps and biased variance, with first/mixed derivatives at real observed zero/near-zero/ordinary inputs. Independent dE/dP path calls no production AO assembler/local-partial helper; its weighted-F32 rounding is judged by separately frozen atol1e-6/rtol1e-5 from existing operator gradcheck, not strict F64 identity. Hard constraints and numerical path checks pass. If an FD limitation is UNRESOLVED, technical status remains LIMITED under the prospectively recorded nondecisive-FD rule. The historical secondary theta1 FD limitation remains UNRESOLVED in its original report regardless of F026's result.

## Absolute endpoints, not normalized progress

| System | control t0 | intervention t0 | control t32 | intervention t32 | final delta |
|---|---:|---:|---:|---:|---:|
{systems}

Mean initial: control {final["control_initial_loss"]:.16g}, intervention {final["intervention_initial_loss"]:.16g}; epsilon alone changes the initial loss by {final["initial_epsilon_effect"]:+.12g}. This is a representational/numerical change at identical tensors, not evidence of better optimization. Mean final: control {final["control_final_loss"]:.16g}, intervention {final["intervention_final_loss"]:.16g}. Absolute delta32 = {final["final_absolute_delta"]:+.12g}; B = {final["relative_final_improvement_B"]:.12g}. One-percent threshold {final["effect_floor_threshold"]:.16g}: {final["effect_floor_pass"]}. Maximum final system loss: control {final["control_max_final_system_loss"]:.12g}, intervention {final["intervention_max_final_system_loss"]:.12g}.

Reductions from own starts: control {final["control_reduction_from_own_start"]:.12g}, intervention {final["intervention_reduction_from_own_start"]:.12g}. The primary comparison remains absolute final original operator loss. Two explicitly permitted read-only fixed-weight epsilon swaps were evaluated after t32 was frozen: control t32 tensors with eps1e-3 give mean {swaps[0]["mean_loss"]:.12g}; intervention t32 tensors with eps1e-5 give mean {swaps[1]["mean_loss"]:.12g}. These distinguish pointwise normalization sensitivity from parameter differences, without causal attribution of the training path. Neither counterfactual selects a model or changes the predeclared classification; no optimizer update was performed.

## Startup conditioning and optimizer dynamics

Startup gradient norm: control 33745589.04345739, intervention {preflight["metrics"][0]["gradient_norm"]:.12g}; reduction factor {final["startup_gradient_reduction_factor"]:.12g}. Per-module gradients, dominant coordinates, squared-norm fractions, actual ordinary/UEG hidden means/variances and inverse standard deviations at theta0/1/2/32 are in gradient_diagnostics.csv. Labels distinguish preflight and retained trajectory. LN pre-hooks only observe; native LayerNorm implementation is unchanged. Constraints at theta0/1/32 cover UEG/Gx anchors, spin, high density, rapid gradient, tau independence and finite outputs/operators.

| update | first-gradient displacement dot | rest dot | total native displacement dot | applied |
|---|---:|---:|---:|---|
{updates}

Negative dot predicts local descent of the true operator loss. Realized step1 loss change {final["realized_step1_loss_change"]:+.12g}; step2 change {final["realized_step2_loss_change"]:+.12g}. Historical control update2 had first/rest/total dots +0.004817969742 / -0.000605736539 / +0.004212087320 and actual loss change +0.00500770340812. Compare these directional/finite-step measurements, not just moment norms. All decompositions use the actual complete proposal's denominator and bias corrections. Native update1/2 arithmetic reproduces realized F32 parameters byte-for-byte. t33 is arithmetic only, not applied. Stored first/rest moments are accounting, not independent causal trajectories; t1 residual is rounding only. Negative v residuals are retained without clipping. No additive causal decomposition of the nonlinear second-moment denominator is claimed.

An additional postprocessing relative-only moment guard initially FAILED at t1 coordinate 322: F64 v_first=1.8549307819044815e-41, native v=1.8548987772267604e-41, remainder=-3.200467772116241e-46. This subnormal difference is less than half the exact minimum F32 step (1.401298464324817e-45), and native v equals the nearest F32 rounding of v_first exactly. The original FAIL remains in rounding_diagnostic.json and per-tensor CSV counts. It is not hidden by widening the relative criterion: a separate exact subnormal-rounding check explains it; any unexplained violation still fails. Subnormal moment ratios are excluded as unresolved rather than treated as meaningful percentages. No scientific derivative, loss or constraint tolerance changed. This accounting-only issue did not alter training or its checkpoints.

H1 is assessed by the measured startup reduction. H2 is assessed separately by early directional and realized changes above. H3 is assessed by absolute final loss and the declared 1% floor. One seed cannot establish general superiority. Production implementation/default architecture and all historical artifacts remain hash-unchanged. Tests, lint, compile and diff-check results are recorded separately.

One next action: {decision["next_experiment"]}

Scientific wall at report generation {elapsed():.3f} seconds, limit 7200, respected {elapsed() < 7200}. No SCF, Clean28/future-test data, new systems, production targets/loss changes, control retraining or chemistry training.
"""
    (ROOT / (PREFIX + "report.md")).write_text(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "phase",
        choices=[
            "freeze",
            "preflight",
            "recover_preflight",
            "train",
            "coupling_check",
            "rounding_diagnostic",
            "finish",
        ],
    )
    phase = p.parse_args().phase
    tic = time.monotonic()
    getattr(sys.modules[__name__], phase)()
    write(
        phase + "_timing",
        {
            "phase_wall_seconds": time.monotonic() - tic,
            "scientific_wall_seconds": elapsed(),
        },
    )
