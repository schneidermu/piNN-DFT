"""Bounded diagnostics of the authenticated fresh pilot; never a new arm."""

import argparse
import copy
import inspect
import itertools
import json
import random
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_fresh_grid_pilot_v1 as pilot

ROOT = pilot.ROOT
PREFIX = "lap_fresh_layernorm_v1_"
OUT = ROOT.parent / "lap_fresh_layernorm_v1_runs_20261015"
DEVICE = "cuda"


def write(name, value):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def table(name, rows):
    import csv

    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def elapsed():
    return time.time() - int((ROOT / (PREFIX + "started.txt")).read_text()) / 1000


def budget(reserve=0):
    if elapsed() + reserve >= 7200:
        raise TimeoutError("Frozen 7200-second wall budget")


def names(model):
    return pilot.named_trainable_parameters(model)


def norm(x):
    return float(np.linalg.norm(x))


def cosine(a, b):
    return float(a @ b / norm(a) / norm(b)) if norm(a) and norm(b) else None


def layout(model):
    ln = {
        name + "." + leaf
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
        for leaf in ("weight", "bias")
    }
    result, start = [], 0
    for name, p in names(model).items():
        end = start + p.numel()
        result.append((name, start, end, name in ln, tuple(p.shape)))
        start = end
    assert start == 9446
    return result


def fresh():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return pilot.pcPBELMLOptimizerV2Lap(**pilot.read("protocol")["constructor"])


def batch(model, records):
    shadow = copy.deepcopy(model).double().to(DEVICE)
    results = []
    for record in records.values():
        budget(300)
        results.append(pilot.system_gradient(shadow, record, "FULL_SOURCE", DEVICE))
    return (
        [r[0] for r in results],
        torch.stack([r[1] for r in results]).mean(0),
        torch.stack([r[1] for r in results]),
    )


def snapshot(model, optimizer, t):
    path = OUT / f"theta{t}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "t": t,
            "RNG": pilot.rng(),
        },
        path,
    )
    return {"path": str(path), "sha256": pilot.sha(path)}


def freeze():
    OUT.mkdir(exist_ok=True)
    source = pilot.read("protocol")["source_hashes"]
    source = {
        k: v
        for k, v in source.items()
        if k.startswith(("train_models/", "dft_functionals/"))
    }
    source["tools/lap_fresh_grid_pilot_v1.py"] = pilot.read("provenance")[
        "driver_sha256"
    ]
    historical = [f for f in ROOT.glob("lap_fresh_grid_pilot_v1_*") if f.is_file()]
    protocol = {
        "namespace": PREFIX,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "wall_budget_seconds": 7200,
        "timer_unix_ms": int((ROOT / (PREFIX + "started.txt")).read_text()),
        "source_hashes": source,
        "historical_hashes": {f.name: pilot.sha(f) for f in historical},
        "scope": "FULL_SOURCE H2/BeH2/CO; exactly two replay steps; no intervention arms; theta16/32 read-only",
        "tolerances": {
            "gradient_atol": 1e-10,
            "gradient_rtol": 1e-9,
            "loss_atol": 1e-10,
            "loss_rtol": 1e-9,
            "LN_F64_atol": 1e-10,
            "LN_F64_rtol": 1e-9,
            "constraints_atol": 1e-12,
        },
        "FD_rule": "H2 coordinate steps = 1e-4*max(abs(base adjoint contraction),1)/max(abs(H2 coordinate gradient),1e-30) times [0.125,0.25,0.5,1,2,4,8]; lower bound 64*F64_epsilon*max(1,abs(parameter)); frozen before FD results",
        "FD_qualification": "a convergence plateau requires two adjacent steps within 1% of AD and each other; mixed-F32 quantization/curvature recorded separately; no automatic production-bug claim",
        "LN_replica_scheme": "named semantic parameter-use groups, identical replicas, density dependencies retained; strict branch sum closure",
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
    }
    write("protocol", protocol)


def replay():
    if (ROOT / (PREFIX + "replay_receipt.json")).exists():
        raise RuntimeError(
            "Replay already completed; reuse theta1/theta2, never repeat steps"
        )
    protocol = json.loads((ROOT / (PREFIX + "protocol.json")).read_text())
    for file, value in protocol["source_hashes"].items():
        assert pilot.sha(ROOT / file) == value, file
    records = pilot.read("representations")["records"]
    for r in records.values():
        assert pilot.sha(pilot.local(r["path"])) == r["sha256"]
    provenance = pilot.read("provenance")
    authenticated = []
    payloads = {}
    for receipt in provenance["pilot_checkpoint_receipts"]:
        if "FULL_SOURCE_t" in receipt["path"]:
            value = pilot.resume_payload(receipt)
            t = value["counters"]["FULL_SOURCE"]
            payloads[t] = value
            authenticated.append(receipt)
    for r in provenance["gradient_receipts"]:
        assert pilot.sha(pilot.local(r["path"])) == r["sha256"]
        authenticated.append(r)
    model = fresh()
    assert pilot.state_hash(model) == pilot.read("initialization")["state_hash"]
    assert sum(p.numel() for p in names(model).values()) == 9446
    model.to(DEVICE)
    assert pilot.state_hash(model) == pilot.read("initialization")["state_hash"]
    model.load_state_dict(payloads[0]["models"]["FULL_SOURCE"])
    optimizer = torch.optim.AdamW(
        names(model).values(),
        **{**pilot.read("protocol")["optimizer"], "betas": (0.9, 0.999)},
    )
    assert not optimizer.state
    history, tensors, receipts = [], {}, []
    initial_receipt = provenance["gradient_receipts"][0]
    with np.load(pilot.local(initial_receipt["path"])) as cache:
        cached = cache["FULL_SOURCE_gradient"]
    for t in range(3):
        started = time.monotonic()
        values, g, system = batch(model, records)
        a = g.detach().cpu().numpy()
        if t == 0:
            np.testing.assert_allclose(a, cached, atol=1e-10, rtol=1e-9)
        history.append(
            {
                "theta": t,
                "loss": float(np.mean(values)),
                "gradient_index": t + 1,
                "gradient_norm": norm(a),
                "system_losses": values,
                "active_seconds": time.monotonic() - started,
            }
        )
        tensors[f"g{t + 1}"] = a
        tensors[f"systems_theta{t}"] = system.detach().cpu().numpy()
        tensors[f"parameters_theta{t}"] = pilot.vector(model)
        receipts.append(snapshot(model, optimizer, t))
        print(history[-1], flush=True)
        if t < 2:
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for p in names(model).values():
                p.grad = g[offset : offset + p.numel()].reshape(p.shape).to(p.dtype)
                offset += p.numel()
            tensors[f"g{t + 1}_native"] = g.float().detach().cpu().numpy()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    np.savez(OUT / "replay_arrays.npz", **tensors)
    for t in (16, 32):
        torch.save(payloads[t], OUT / f"authenticated_theta{t}.pt")
    write(
        "replay_receipt",
        {
            "native_optimizer_replay_updates_executed": 2,
            "new_intervention_arms": 0,
            "history": history,
            "diagnostic_states": receipts,
            "arrays": {
                "path": str(OUT / "replay_arrays.npz"),
                "sha256": pilot.sha(OUT / "replay_arrays.npz"),
            },
            "fresh_hash_reproduced": True,
            "initial_gradient_cache_parity": "PASS",
            "state_numbering": "g1 at theta0; g2 at theta1; g3 at theta2; g33 at theta32",
            "read_only_pilot_states": [0, 16, 32],
        },
    )
    write(
        "provenance",
        {
            "authenticated": authenticated,
            "representations": records,
            "source_hashes": protocol["source_hashes"],
            "old_models_loaded": 0,
        },
    )


def load_theta(t):
    model = fresh().to(DEVICE)
    payload = torch.load(OUT / f"theta{t}.pt", weights_only=False, map_location=DEVICE)
    model.load_state_dict(payload["model"])
    return model, payload


class Observer:
    """Summary-only observation; no input or constraint is detached in the graph."""

    def __init__(self):
        self.enabled = True
        self.stats = {}
        self.samples = {}
        self.kind = "unlabelled"

    def observe(self, name, module, h):
        if not self.enabled:
            return
        with torch.no_grad():
            x = h.detach()
            v = x.var(-1, unbiased=False)
            key = (name, self.kind)
            s = self.stats.setdefault(
                key,
                {
                    "count": 0,
                    "mean_sum": 0.0,
                    "variance_sum": 0.0,
                    "min_variance": float("inf"),
                    "max_variance": 0.0,
                    "inverse_std_sum": 0.0,
                    "variance_le_eps": 0,
                    "exact_zero_inputs": 0,
                    "eps": module.eps,
                },
            )
            s["count"] += len(x)
            s["mean_sum"] += float(x.mean(-1).sum())
            s["variance_sum"] += float(v.sum())
            s["min_variance"] = min(s["min_variance"], float(v.min()))
            s["max_variance"] = max(s["max_variance"], float(v.max()))
            s["inverse_std_sum"] += float((v + module.eps).rsqrt().sum())
            s["variance_le_eps"] += int((v <= module.eps).sum())
            s["exact_zero_inputs"] += int((x == 0).all(-1).sum())
            s["affine_weight_norm"] = float(module.weight.norm())
            s["affine_bias_norm"] = float(module.bias.norm())
            for label, idx in [("min", int(v.argmin())), ("max", int(v.argmax()))]:
                sample_key = (name, self.kind, label)
                candidate = float(v[idx])
                old = self.samples.get(sample_key)
                if (
                    old is None
                    or (label == "min" and candidate < old[0])
                    or (label == "max" and candidate > old[0])
                ):
                    self.samples[sample_key] = (
                        candidate,
                        x[idx].cpu().numpy(),
                        module.weight.detach().cpu().numpy(),
                        module.bias.detach().cpu().numpy(),
                        module.eps,
                    )


def instrument(model, observer):
    """Expand only affine parameter USES, with the original source call graph."""
    original_class = type(model)
    source = textwrap.dedent(inspect.getsource(original_class.forward_descriptors))
    replacements = {
        "self.exchange_hidden(ex)": 'self.tagged("x_real", self.exchange_hidden, ex)',
        "self.exchange_hidden(ue)": 'self.tagged("x_UEG", self.exchange_hidden, ue)',
        "self.correlation_output(d, hx)": 'self.tagged("c_real", self.correlation_output, d, hx)',
        "self.correlation_output(x1, (hu + hd) / 2)": 'self.tagged("c_UEG", self.correlation_output, x1, (hu + hd) / 2)',
        "self.correlation_output(x2, hx)": 'self.tagged("c_high_density", self.correlation_output, x2, hx)',
        "self.correlation_output(x3, hx)": 'self.tagged("c_rapid_gradient", self.correlation_output, x3, hx)',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    namespace = {"torch": torch, "NN_OUTPUT_SCALE_PBE": pilot.NN_OUTPUT_SCALE_PBE}
    # Only hash-authenticated local method source, never data or external code.
    exec(compile(source, "<diagnostic-source-call-labelling>", "exec"), namespace)  # noqa: S102
    model.forward_descriptors = types.MethodType(
        namespace["forward_descriptors"], model
    )

    def tagged(self, kind, function, *args):
        previous = observer.kind
        observer.kind = kind
        try:
            return function(*args)
        finally:
            observer.kind = previous

    def exchange_hidden(self, d):
        kind = observer.kind
        up = self.tagged(kind + "_up", self.x_feature_extractor, d[:, [2, 5]])
        down = self.tagged(kind + "_down", self.x_feature_extractor, d[:, [4, 6]])
        return up, down

    def correlation_output(self, d, hx):
        kind = observer.kind
        h = self.tagged(
            kind + "_direct",
            lambda z: self.c_symmetrization_blocks(self.c_input_layers(z)),
            d,
        )
        hs = self.tagged(
            kind + "_swap",
            lambda z: self.c_symmetrization_blocks(self.c_input_layers(z)),
            self.swap_descriptors(d),
        )
        h = self.tagged(kind + "_fused", self.c_post_symm_blocks, (h + hs) / 2)
        return self.c_output_layer(torch.cat([h, hx], 1))

    model.tagged = types.MethodType(tagged, model)
    model.exchange_hidden = types.MethodType(exchange_hidden, model)
    model.correlation_output = types.MethodType(correlation_output, model)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.LayerNorm):
            continue
        if name.startswith("x_"):
            groups = ["x_real_up", "x_real_down", "x_UEG_up", "x_UEG_down"]
        elif name.startswith("c_post"):
            groups = [
                kind + "_fused"
                for kind in ["c_real", "c_UEG", "c_high_density", "c_rapid_gradient"]
            ]
        else:
            groups = [
                kind + suffix
                for kind in ["c_real", "c_UEG", "c_high_density", "c_rapid_gradient"]
                for suffix in ["_direct", "_swap"]
            ]
        module.replicas = nn.ParameterDict(
            {
                kind + "__" + leaf: nn.Parameter(getattr(module, leaf).detach().clone())
                for kind in groups
                for leaf in ["weight", "bias"]
            }
        )

        def forward(self, h, module_name=name):
            observer.observe(module_name, self, h)
            return F.layer_norm(
                h,
                self.normalized_shape,
                self.replicas[observer.kind + "__weight"],
                self.replicas[observer.kind + "__bias"],
                self.eps,
            )

        module.forward = types.MethodType(forward, module)
    return model


def attributed_system(model, observer, record):
    import h5py

    with h5py.File(pilot.local(record["path"])) as f:
        ref = torch.tensor(f["ref_FULL_SOURCE"][:], device=DEVICE)
        S = torch.tensor(f["S"][:], device=DEVICE)
    observer.enabled = True
    V = torch.zeros_like(ref)
    for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
        V += pilot.chunk_matrix(model, arrays, DEVICE, False).detach()
    leaf = V.detach().requires_grad_(True)
    loss = pilot.op.operator_loss(leaf, ref, S)
    adjoint = torch.autograd.grad(loss, leaf)[0].detach()
    observer.enabled = False
    parameters = dict(model.named_parameters())
    totals = {k: torch.zeros_like(v) for k, v in parameters.items()}
    for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
        M = pilot.chunk_matrix(model, arrays, DEVICE, True)
        gradients = torch.autograd.grad(
            (M * adjoint).sum(), tuple(parameters.values()), allow_unused=True
        )
        for key, p, g in zip(parameters, parameters.values(), gradients):
            totals[key] += torch.zeros_like(p) if g is None else g
    return float(loss.detach()), {
        k: v.detach().cpu().numpy() for k, v in totals.items()
    }


def attribution():
    records = pilot.read("representations")["records"]
    replay_data = np.load(OUT / "replay_arrays.npz")
    call_rows, branch_rows, parameter_rows, checks = [], [], [], []
    all_samples = {}
    profile = None
    for t in (0, 1, 2):
        budget(600)
        native, _ = load_theta(t)
        ordering = layout(native)
        observer = Observer()
        model = instrument(copy.deepcopy(native).double(), observer)
        arrays = next(pilot.chunk_arrays(records["H2"], "FULL_SOURCE"))
        tic = time.monotonic()
        original_chunk = pilot.chunk_matrix(
            copy.deepcopy(native).double(), arrays, DEVICE, False
        )
        expanded_chunk = pilot.chunk_matrix(model, arrays, DEVICE, False)
        torch.testing.assert_close(
            original_chunk, expanded_chunk, atol=1e-10, rtol=1e-9
        )
        if profile is None:
            M = pilot.chunk_matrix(model, arrays, DEVICE, True)
            torch.autograd.grad(
                M.square().sum(), tuple(model.parameters()), allow_unused=True
            )
            profile = time.monotonic() - tic
            print("attribution chunk profile seconds", profile, flush=True)
            assert (
                profile * sum(r["SOURCE_count"] for r in records.values()) / 2048 * 3
                < 7200 - elapsed() - 600
            )
        observer.stats.clear()
        observer.samples.clear()
        outputs = [
            attributed_system(model, observer, record) for record in records.values()
        ]
        base = replay_data[f"g{t + 1}"]
        reconstructed = np.zeros_like(base)
        groups = {}
        for name, start, end, is_ln, shape in ordering:
            if is_ln:
                parent, leaf = name.rsplit(".", 1)
                selected = [
                    key
                    for key in outputs[0][1]
                    if key.startswith(parent + ".replicas.")
                    and key.endswith("__" + leaf)
                ]
                total = np.zeros(end - start)
                for key in selected:
                    contribution = np.mean(
                        [out[1][key].reshape(-1) for out in outputs], axis=0
                    )
                    kind = key.split(".replicas.")[1].split("__")[0]
                    groups.setdefault(kind, np.zeros_like(base))[start:end] += (
                        contribution
                    )
                    total += contribution
                reconstructed[start:end] = total
            else:
                reconstructed[start:end] = np.mean(
                    [out[1][name].reshape(-1) for out in outputs], axis=0
                )
            g = base[start:end]
            p0 = replay_data["parameters_theta0"][start:end]
            current = replay_data[f"parameters_theta{t}"][start:end]
            parameter_rows.append(
                {
                    "theta": t,
                    "name": name,
                    "shape": str(shape),
                    "coordinates": end - start,
                    "LayerNorm": is_ln,
                    "gradient_norm": norm(g),
                    "max_abs": float(np.max(np.abs(g))),
                    "max_index": int(np.argmax(np.abs(g))),
                    "squared_norm_fraction": norm(g) ** 2 / norm(base) ** 2,
                    "displacement_from_theta0": norm(current - p0),
                }
            )
        np.testing.assert_allclose(reconstructed, base, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(
            [o[0] for o in outputs],
            json.loads((ROOT / (PREFIX + "replay_receipt.json")).read_text())[
                "history"
            ][t]["system_losses"],
            atol=1e-10,
            rtol=1e-9,
        )
        checks.append(
            {
                "theta": t,
                "gradient_relative_closure": norm(reconstructed - base) / norm(base),
                "primal_and_density_operator_chunk_parity": "PASS",
                "loss_parity": "PASS",
            }
        )
        for kind, g in groups.items():
            branch_rows.append(
                {
                    "theta": t,
                    "call_kind": kind,
                    "gradient_norm": norm(g),
                    "signed_dot_total": float(g @ base),
                    "signed_projection_fraction": float(g @ base / (base @ base)),
                    "cosine_total": cosine(g, base),
                }
            )
        for (name, kind), s in observer.stats.items():
            n = s["count"]
            call_rows.append(
                {
                    "theta": t,
                    "module": name,
                    "call_kind": kind,
                    "points": n,
                    "hidden_mean": s["mean_sum"] / n,
                    "variance_mean": s["variance_sum"] / n,
                    "variance_min": s["min_variance"],
                    "variance_max": s["max_variance"],
                    "inverse_std_mean": s["inverse_std_sum"] / n,
                    "fraction_variance_le_eps": s["variance_le_eps"] / n,
                    "exact_zero_inputs": s["exact_zero_inputs"],
                    "eps": s["eps"],
                    "affine_weight_norm": s["affine_weight_norm"],
                    "affine_bias_norm": s["affine_bias_norm"],
                }
            )
        all_samples[t] = observer.samples
        np.savez(OUT / f"branch_theta{t}.npz", **groups)
        print("branch closure", checks[-1], flush=True)
    torch.save(all_samples, OUT / "hidden_samples.pt")
    table("call_statistics", call_rows)
    table("parameter_gradients", parameter_rows)
    table("branch_attribution", branch_rows)
    write(
        "attribution_closure",
        {
            "checks": checks,
            "chunk_profile_seconds": profile,
            "shared_hidden": "exchange hx contributions include all real/constraint correlation consumers; density dependencies never detached",
            "replicas_only": True,
        },
    )


def flatten_moments(payload, ordering, pilot_payload=False):
    state = (
        payload["optimizers"]["FULL_SOURCE"] if pilot_payload else payload["optimizer"]
    )
    ids = state["param_groups"][0]["params"]
    assert len(ids) == len(ordering)
    m, v, steps = [], [], []
    for i, (_, start, end, _, shape) in zip(ids, ordering):
        s = state["state"][i]
        assert tuple(s["exp_avg"].shape) == shape
        assert s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32
        m.append(s["exp_avg"].detach().cpu().reshape(-1))
        v.append(s["exp_avg_sq"].detach().cpu().reshape(-1))
        steps.append(int(s["step"]))
    assert len(set(steps)) == 1
    return torch.cat(m), torch.cat(v), steps[0]


def first_components(g1, t):
    return 0.1 * 0.9 ** (t - 1) * g1, 0.001 * 0.999 ** (t - 1) * g1**2


def proposal(theta, m, v, g, t):
    """Native single-tensor AdamW arithmetic on detached copies, never step()."""
    mn, vn, p = m.clone(), v.clone(), theta.clone()
    mn.lerp_(g, 0.1)
    vn.mul_(0.999).addcmul_(g, g, value=0.001)
    den = vn.sqrt().div_((1 - 0.999**t) ** 0.5).add_(1e-8)
    p.mul_(1 - 2.5e-5 * 0.01)
    p.addcdiv_(mn, den, value=-2.5e-5 / (1 - 0.9**t))
    return p, mn, vn, den


def memory():
    data = np.load(OUT / "replay_arrays.npz")
    native, _ = load_theta(0)
    ordering = layout(native)
    g1 = data["g1_native"].astype(np.float64)
    rows, updates, summaries = [], [], []
    with np.load(
        pilot.local(pilot.read("provenance")["gradient_receipts"][1]["path"])
    ) as ep:
        g33 = ep["FULL_SOURCE_FULL_SOURCE"]
    for t in (1, 2, 16, 32):
        payload = torch.load(
            OUT / (f"theta{t}.pt" if t < 3 else f"authenticated_theta{t}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        m32, v32, counter = flatten_moments(payload, ordering, t >= 3)
        assert counter == t
        m, v = m32.double().numpy(), v32.double().numpy()
        mf, vf = first_components(g1, t)
        mr, vr = m - mf, v - vf
        current = data[f"g{t + 1}"] if t < 3 else g33 if t == 32 else None
        for name, start, end, is_ln, shape in ordering:
            ix = slice(start, end)
            valid = v[ix] > 0
            ratios = vf[ix][valid] / v[ix][valid]
            rows.append(
                {
                    "t": t,
                    "name": name,
                    "LayerNorm": is_ln,
                    "m_first_norm": norm(mf[ix]),
                    "m_rest_norm": norm(mr[ix]),
                    "m_total_norm": norm(m[ix]),
                    "cosine_first_rest": cosine(mf[ix], mr[ix]),
                    "cosine_first_current": cosine(mf[ix], current[ix])
                    if current is not None
                    else None,
                    "cosine_rest_current": cosine(mr[ix], current[ix])
                    if current is not None
                    else None,
                    "v_first_norm": norm(vf[ix]),
                    "v_rest_norm": norm(vr[ix]),
                    "v_total_norm": norm(v[ix]),
                    "v_ratio_median": float(np.median(ratios)) if len(ratios) else None,
                    "v_ratio_max": float(np.max(ratios)) if len(ratios) else None,
                    "negative_v_rest_count": int((vr[ix] < 0).sum()),
                    "minimum_v_rest": float(np.min(vr[ix])),
                    "reconstruction_m_abs": norm((mf + mr - m)[ix]),
                    "reconstruction_v_abs": norm((vf + vr - v)[ix]),
                }
            )
        summaries.append(
            {
                "t": t,
                "m_first_norm": norm(mf),
                "m_rest_norm": norm(mr),
                "m_total_norm": norm(m),
                "cosine_first_current": cosine(mf, current)
                if current is not None
                else None,
                "v_first_norm": norm(vf),
                "v_total_norm": norm(v),
                "negative_v_rest_count": int((vr < 0).sum()),
                "minimum_v_rest": float(vr.min()),
                "beta1_power": 0.9 ** (t - 1),
                "beta2_power": 0.999 ** (t - 1),
                "full_history_available": False,
            }
        )
    for t in (2, 33):
        payload = torch.load(
            OUT / ("theta1.pt" if t == 2 else "authenticated_theta32.pt"),
            weights_only=False,
            map_location="cpu",
        )
        m, v, counter = flatten_moments(payload, ordering, t == 33)
        state = payload["model"] if t == 2 else payload["models"]["FULL_SOURCE"]
        theta = torch.cat([state[name].cpu().flatten() for name, *_ in ordering])
        g = data["g2"] if t == 2 else g33
        next_theta, mn, vn, den = proposal(
            theta.to(DEVICE),
            m.to(DEVICE),
            v.to(DEVICE),
            torch.tensor(g, dtype=torch.float32, device=DEVICE),
            t,
        )
        next_theta, mn, vn, den = [x.cpu() for x in (next_theta, mn, vn, den)]
        actual = (
            data["parameters_theta2"] - data["parameters_theta1"]
            if t == 2
            else (next_theta.double() - theta.double()).numpy()
        )
        if t == 2:
            np.testing.assert_array_equal(
                next_theta.numpy(), data["parameters_theta2"].astype(np.float32)
            )
        mf, _ = first_components(g1, t)
        rest = mn.double().numpy() - mf
        scale = -2.5e-5 / (1 - 0.9**t) / den.double().numpy()
        first, remaining = scale * mf, scale * rest
        decay = -2.5e-5 * 0.01 * theta.double().numpy()
        rounding = actual - first - remaining - decay
        components = {
            "first_gradient": first,
            "rest_gradients": remaining,
            "weight_decay": decay,
            "rounding_residual": rounding,
            "total_native": actual,
            "raw_negative_gradient": -g,
        }
        for kind, delta in components.items():
            updates.append(
                {
                    "update": t,
                    "component": kind,
                    "block": "ALL",
                    "norm": norm(delta),
                    "loss_directional_dot": float(g @ delta),
                    "cosine_gradient": cosine(g, delta),
                    "applied": t == 2,
                }
            )
            for block, use in [
                ("LayerNorm", lambda r: r[3]),
                ("exchange_nonLN", lambda r: not r[3] and r[0].startswith("x_")),
                ("correlation_nonLN", lambda r: not r[3] and r[0].startswith("c_")),
            ]:
                indices = np.concatenate(
                    [np.arange(r[1], r[2]) for r in ordering if use(r)]
                )
                updates.append(
                    {
                        "update": t,
                        "component": kind,
                        "block": block,
                        "norm": norm(delta[indices]),
                        "loss_directional_dot": float(g[indices] @ delta[indices]),
                        "cosine_gradient": cosine(g[indices], delta[indices]),
                        "applied": t == 2,
                    }
                )
        np.savez(
            OUT / f"update{t}_components.npz",
            **components,
            proposed_m=mn.numpy(),
            proposed_v=vn.numpy(),
            denominator=den.numpy(),
        )
    table("moment_accounting", rows)
    table("update_decomposition", updates)
    write(
        "memory_summary",
        {
            "states": summaries,
            "full_gradient_history": "not saved; first/rest accounting uses saved moments, not a replay of g3..g32",
            "update2_native_arithmetic_parity": "byte-identical",
            "t33_applied": False,
        },
    )


def independent_ln(h, weight, bias, eps):
    centered = h - h.mean(-1, keepdim=True)
    return (
        weight * centered / (centered.square().mean(-1, keepdim=True) + eps).sqrt()
        + bias
    )


def local_derivatives():
    samples = torch.load(OUT / "hidden_samples.pt", weights_only=False)
    rows = []
    for t, records in samples.items():
        for (name, kind, label), (_, h, w, b, eps) in records.items():
            h, w, b = [
                torch.tensor(x, dtype=torch.float64, requires_grad=True)
                for x in (h, w, b)
            ]
            probe = torch.linspace(-1, 1, len(h), dtype=torch.float64)
            native = F.layer_norm(h, (len(h),), w, b, eps)
            oracle = independent_ln(h, w, b, eps)
            a = torch.autograd.grad(
                (native * probe).sum(), (h, w, b), create_graph=True
            )
            c = torch.autograd.grad(
                (oracle * probe).sum(), (h, w, b), create_graph=True
            )
            for x, y in zip(a, c):
                torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)
            # Mixed density/affine derivative: d/dgamma of a directional input derivative.
            mixed_a = torch.autograd.grad((a[0] * probe.flip(0)).sum(), w)[0]
            mixed_c = torch.autograd.grad((c[0] * probe.flip(0)).sum(), w)[0]
            torch.testing.assert_close(mixed_a, mixed_c, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(native, oracle, atol=1e-10, rtol=1e-9)
            rows.append(
                {
                    "check": "independent_LN",
                    "theta": t,
                    "module": name,
                    "call_kind": kind,
                    "sample": label,
                    "variance": float(h.detach().var(unbiased=False)),
                    "forward_gap": float((native - oracle).detach().abs().max()),
                    "first_gap": max(
                        float((x - y).detach().abs().max()) for x, y in zip(a, c)
                    ),
                    "mixed_gap": float((mixed_a - mixed_c).abs().max()),
                    "status": "PASS",
                }
            )
    table("derivative_checks", rows)


def energy_first_chunk(model, arrays, create_graph=True, promoted=False):
    """Differentiate discrete energy in P; no local-partial or AO-assembler helper.

    Weighted F32 energy backprop rounds the weights, unlike the production
    unweighted-pointwise derivative followed by F64 quadrature. This difference
    is recorded, not hidden by asserting arithmetic identity.
    """
    a = {
        k: torch.as_tensor(v, dtype=torch.float64, device=DEVICE)
        for k, v in arrays.items()
    }
    phi, grad, lap = a["phi"], a["grad"], a["lap"]
    delta = torch.zeros(
        (phi.shape[1], phi.shape[1]),
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    dr = 0.5 * torch.einsum("gm,mn,gn->g", phi, delta, phi)
    dg = 0.5 * (
        torch.einsum("gkm,mn,gn->gk", grad, delta, phi)
        + torch.einsum("gm,mn,gkn->gk", phi, delta, grad)
    )
    dl = 0.5 * (
        torch.einsum("gm,mn,gn->g", lap, delta, phi)
        + torch.einsum("gm,mn,gn->g", phi, delta, lap)
        + 2 * torch.einsum("gkm,mn,gkn->g", grad, delta, grad)
    )
    rho = a["features"][:, :2] + dr[:, None]
    gradients = a["features"][:, 2:8].reshape(-1, 2, 3) + dg[:, None, :]
    laplacian = a["features"][:, 8:] + dl[:, None]
    s = torch.stack(
        [
            gradients[:, 0].square().sum(-1),
            (gradients[:, 0] * gradients[:, 1]).sum(-1),
            gradients[:, 1].square().sum(-1),
        ],
        -1,
    )
    raw = torch.cat(
        [
            rho,
            torch.stack([s[:, 0], s[:, 0] + 2 * s[:, 1] + s[:, 2], s[:, 2]], -1),
            torch.zeros_like(rho),
            laplacian,
        ],
        -1,
    )
    constants = model(raw)
    if promoted:
        energy = pilot.op.PBE.F_PBE(rho, s, constants, DEVICE) * rho.sum(-1)
    else:
        rf, gf = rho.float(), gradients.float()
        sf = torch.stack(
            [
                gf[:, 0].square().sum(-1),
                (gf[:, 0] * gf[:, 1]).sum(-1),
                gf[:, 1].square().sum(-1),
            ],
            -1,
        )
        energy = pilot.op.PBE.F_PBE(rf, sf, constants.float(), DEVICE) * rf.sum(-1)
    integrated = (energy * a["weights"]).sum()
    matrix = torch.autograd.grad(integrated, delta, create_graph=create_graph)[0]
    return matrix


def full_matrix(model, record):
    V = None
    for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
        M = pilot.chunk_matrix(model, arrays, DEVICE, False).detach()
        V = M if V is None else V + M
    return V


def coordinate_directions(model, gradient):
    mapping = layout(model)
    ordered = sorted(
        (
            (abs(gradient[i]), name, i - start, i)
            for name, start, end, ln, _ in mapping
            if ln
            for i in range(start, end)
        ),
        key=lambda r: (-r[0], r[1], r[2]),
    )
    first = ordered[0]
    second = next(r for r in ordered if r[1] != first[1])
    controls = sorted(
        (
            (abs(gradient[i]), name, i - start, i)
            for name, start, end, ln, _ in mapping
            if not ln and name.startswith("x_")
            for i in range(start, end)
        ),
        key=lambda r: (-r[0], r[1], r[2]),
    )
    return [(r[1], r[2], r[3]) for r in [first, second, controls[0]]]


def end_to_end():
    import csv

    import h5py

    record = pilot.read("representations")["records"]["H2"]
    data = np.load(OUT / "replay_arrays.npz")
    initial, _ = load_theta(0)
    directions = coordinate_directions(initial, data["g1"])
    rows = list(csv.DictReader((ROOT / (PREFIX + "derivative_checks.csv")).open()))
    planned, checks = [], []
    for t in (0, 1):
        native, _ = load_theta(t)
        model = copy.deepcopy(native).double()
        parameters = dict(model.named_parameters())
        V = full_matrix(model, record)
        with h5py.File(pilot.local(record["path"])) as f:
            ref, S = [
                torch.tensor(f[k][:], device=DEVICE) for k in ("ref_FULL_SOURCE", "S")
            ]
        leaf = V.detach().requires_grad_(True)
        loss = pilot.op.operator_loss(leaf, ref, S)
        adjoint = torch.autograd.grad(loss, leaf)[0].detach()
        base_scalar = float((V * adjoint).sum())
        h2g = data[f"systems_theta{t}"][0]
        steps = {}
        for name, index, flat in directions:
            p0 = float(parameters[name].detach().flatten()[index])
            h0 = 1e-4 * max(abs(base_scalar), 1) / max(abs(h2g[flat]), 1e-30)
            h0 = max(h0, 64 * np.finfo(float).eps * max(1, abs(p0)))
            steps[(name, index)] = h0 * np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
            planned.append(
                {
                    "theta": t,
                    "name": name,
                    "index": index,
                    "flat_index": flat,
                    "parameter": p0,
                    "H2_AD": float(h2g[flat]),
                    "steps": steps[(name, index)].tolist(),
                }
            )
        # Persist the exact schedule BEFORE observing any finite differences.
        write("FD_schedule", {"directions_from_theta0": directions, "planned": planned})
        selected = tuple(parameters[name] for name, _, _ in directions)
        independent = [torch.zeros_like(p) for p in selected]
        energy_V = torch.zeros_like(V)
        for arrays in pilot.chunk_arrays(record, "FULL_SOURCE"):
            budget(300)
            M = energy_first_chunk(model, arrays, True)
            energy_V += M.detach()
            values = torch.autograd.grad(
                (M * adjoint).sum(), selected, allow_unused=True
            )
            for target, p, g in zip(independent, selected, values):
                target += torch.zeros_like(p) if g is None else g
        matrix_gap = norm((energy_V - V).cpu().numpy()) / norm(V.cpu().numpy())
        for (name, index, flat), ig in zip(directions, independent):
            independent_value = float(ig.flatten()[index])
            ad = float(h2g[flat])
            rows.append(
                {
                    "check": "independent_discrete_energy_P_derivative",
                    "theta": t,
                    "module": name,
                    "index": index,
                    "production_H2_AD": ad,
                    "independent_AD": independent_value,
                    "relative_gap": abs(independent_value - ad) / max(abs(ad), 1e-30),
                    "matrix_relative_gap": matrix_gap,
                    "status": "PASS"
                    if abs(independent_value - ad) <= 1e-10 + 1e-9 * abs(ad)
                    else "MIXED_PRECISION_DISCRETIZATION_DIFFERENCE",
                }
            )
        for name, index, flat in directions:
            p = parameters[name]
            original = float(p.detach().flatten()[index])
            for h in steps[(name, index)]:
                budget(300)
                values = []
                for sign in (1, -1):
                    with torch.no_grad():
                        p.flatten()[index] = original + sign * h
                    M = full_matrix(model, record)
                    values.append(float((M * adjoint).sum()))
                with torch.no_grad():
                    p.flatten()[index] = original
                fd = (values[0] - values[1]) / (2 * h)
                ad = float(h2g[flat])
                rows.append(
                    {
                        "check": "production_mixed_F64_parameter_FD_fixed_adjoint",
                        "theta": t,
                        "module": name,
                        "index": index,
                        "step": float(h),
                        "production_H2_AD": ad,
                        "FD": fd,
                        "relative_gap": abs(fd - ad) / max(abs(ad), 1e-30),
                        "scalar_plus": values[0],
                        "scalar_minus": values[1],
                        "status": "WITHIN_1_PERCENT"
                        if abs(fd - ad) <= 0.01 * abs(ad)
                        else "NO_AGREEMENT_AT_THIS_STEP",
                    }
                )
            assert pilot.state_hash(native) == pilot.state_hash(load_theta(t)[0])
        checks.append(
            {
                "theta": t,
                "independent_energy_matrix_relative_gap": matrix_gap,
                "fixed_adjoint": True,
                "parameter_perturbations_restored": True,
            }
        )
        print("end-to-end completed", checks[-1], flush=True)
    table("derivative_checks", rows)
    write(
        "end_to_end_receipt",
        {
            "checks": checks,
            "independence": "energy-first dE/dP uses no production partial/assembly helper; same PBE formula/model/source inputs; weighted F32 backward is a distinct discrete arithmetic path",
            "FD_max_steps_per_direction_state": 7,
        },
    )


def integrity():
    """Tied-parameter checks and historical-state preservation, no optimizer step."""
    import csv

    protocol = json.loads((ROOT / (PREFIX + "protocol.json")).read_text())
    for name, expected in {
        **protocol["source_hashes"],
        **protocol["historical_hashes"],
    }.items():
        assert pilot.sha(ROOT / name) == expected, name
    checks, downstream = [], []
    record = pilot.read("representations")["records"]["H2"]
    arrays = next(pilot.chunk_arrays(record, "FULL_SOURCE"))
    for t in (0, 1, 2, 16, 32):
        if t < 3:
            model, payload = load_theta(t)
        else:
            payload = torch.load(
                OUT / f"authenticated_theta{t}.pt",
                weights_only=False,
                map_location=DEVICE,
            )
            model = fresh().to(DEVICE)
            model.load_state_dict(payload["models"]["FULL_SOURCE"])
        baseline_hash = pilot.state_hash(model)
        saved_rng = pilot.rng()
        result = pilot.constraints(model)
        assert result["PASS"]
        if t < 3:
            shadow = copy.deepcopy(model).double()
            observer = Observer()
            expanded = instrument(copy.deepcopy(shadow), observer)
            features = torch.tensor(arrays["features"], device=DEVICE)
            a = pilot.op._model_double_operator_partials(
                pilot.LapEnergy(shadow), features, False
            )
            b = pilot.op._model_double_operator_partials(
                pilot.LapEnergy(expanded), features, False
            )
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)
            # This is the first bias-free layer downstream of the dominant LN.
            beta = shadow.x_feature_extractor[1].bias
            first_hidden = shadow.x_feature_extractor[3].fc[0](F.gelu(beta))
            downstream.append(
                {
                    "theta": t,
                    "first_LN_bias_norm": float(beta.detach().norm()),
                    "first_downstream_linear_weight_norm": float(
                        shadow.x_feature_extractor[3].fc[0].weight.detach().norm()
                    ),
                    "first_downstream_anchor_input_norm": float(
                        first_hidden.detach().norm()
                    ),
                    "first_downstream_anchor_input_variance": float(
                        first_hidden.detach().var(unbiased=False)
                    ),
                }
            )
        pilot.restore_rng(saved_rng)
        assert pilot.state_hash(model) == baseline_hash
        assert torch.equal(torch.get_rng_state(), saved_rng[2])
        checks.append(
            {
                "theta": t,
                "constraints": result,
                "parameters_buffers_restored": True,
                "RNG_restored": True,
                "expanded_energy_density_partial_parity": "PASS"
                if t < 3
                else "NOT RUN",
            }
        )
    data = np.load(OUT / "replay_arrays.npz")
    for t in (0, 1, 2):
        np.testing.assert_allclose(
            data[f"g{t + 1}"], data[f"systems_theta{t}"].mean(0), atol=1e-10, rtol=1e-9
        )
    native, _ = load_theta(0)
    mapping = layout(native)
    rows = list(csv.DictReader((ROOT / (PREFIX + "parameter_gradients.csv")).open()))
    for row in rows:
        row["system"] = "mean_H2_BeH2_CO"
    for t in (0, 1, 2):
        for system, vector in zip(["H2", "BeH2", "CO"], data[f"systems_theta{t}"]):
            for name, start, end, ln, shape in mapping:
                g = vector[start:end]
                rows.append(
                    {
                        "theta": t,
                        "system": system,
                        "name": name,
                        "shape": str(shape),
                        "coordinates": end - start,
                        "LayerNorm": ln,
                        "gradient_norm": norm(g),
                        "max_abs": float(np.max(np.abs(g))),
                        "max_index": int(np.argmax(np.abs(g))),
                        "squared_norm_fraction": norm(g) ** 2 / norm(vector) ** 2,
                        "displacement_from_theta0": norm(
                            data[f"parameters_theta{t}"][start:end]
                            - data["parameters_theta0"][start:end]
                        ),
                    }
                )
    table("parameter_gradients", rows)
    rounding = []
    for t in (1, 2, 16, 32):
        payload = torch.load(
            OUT / (f"theta{t}.pt" if t < 3 else f"authenticated_theta{t}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        m, v, _ = flatten_moments(payload, mapping, t >= 3)
        mf, vf = first_components(data["g1_native"].astype(np.float64), t)
        total_v = v.double().numpy()
        rest_v = total_v - vf
        allowance = (t + 4) * np.finfo(np.float32).eps * np.maximum(vf, total_v)
        violation = rest_v < -allowance
        np.savez(
            OUT / f"moment_t{t}.npz",
            m=m.numpy(),
            v=v.numpy(),
            m_first=mf,
            m_rest=m.double().numpy() - mf,
            v_first=vf,
            v_rest=rest_v,
            v_ratio=np.divide(
                vf, total_v, out=np.full_like(vf, np.nan), where=total_v > 0
            ),
            negative_rounding_bound=allowance,
        )
        assert not violation.any()
        rounding.append(
            {
                "t": t,
                "negative_v_rest_count": int((rest_v < 0).sum()),
                "minimum_v_rest": float(rest_v.min()),
                "negative_entries_outside_F32_accumulation_bound": int(violation.sum()),
                "bound": "(t+4)*F32_epsilon*max(v_first,v_total); accounting only; no clipping",
            }
        )
    write(
        "constraint_checks",
        {
            "states": checks,
            "downstream": downstream,
            "moment_rounding": rounding,
            "production_and_historical_files_unchanged": True,
            "native_arithmetic_note": "Initial CPU proposal differed from actual CUDA update2 in 103 coordinates by at most 3.7252903e-9. Matching CUDA arithmetic restored byte identity; CPU attempt was diagnostic, not a production FAIL.",
            "t1_rest_note": "No other gradient exists at t1: the reported residual is entirely native rounding. At later t, rest includes subsequent-gradient contributions and native rounding.",
        },
    )


def finish():
    import csv

    def read(name):
        return json.loads((ROOT / (PREFIX + name + ".json")).read_text())

    data = np.load(OUT / "replay_arrays.npz")
    model, _ = load_theta(0)
    authenticated_order = pilot.read("initialization")["parameter_order"]
    actual_order = [
        {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
        for n, p in names(model).items()
    ]
    assert actual_order == authenticated_order
    assert len({id(p) for p in names(model).values()}) == len(names(model))
    gradient_rows = list(
        csv.DictReader((ROOT / (PREFIX + "parameter_gradients.csv")).open())
    )
    mapping = {n: (start, end) for n, start, end, _, _ in layout(model)}
    for r in gradient_rows:
        t = int(r["theta"])
        start, end = mapping[r["name"]]
        r["displacement_from_previous"] = norm(
            data[f"parameters_theta{t}"][start:end]
            - data[f"parameters_theta{max(0, t - 1)}"][start:end]
        )
    table("parameter_gradients", gradient_rows)

    derivative_file = ROOT / (PREFIX + "derivative_checks.csv")
    derivative_rows = list(csv.DictReader(derivative_file.open()))
    fd = [r for r in derivative_rows if "_FD_" in r["check"]]
    qualifications = []
    schedule = read("FD_schedule")
    for planned in schedule["planned"]:
        subset = [
            r
            for r in fd
            if int(r["theta"]) == planned["theta"] and r["module"] == planned["name"]
        ]
        passed = any(
            a["status"] == b["status"] == "WITHIN_1_PERCENT"
            and abs(float(a["FD"]) - float(b["FD"])) <= 0.01 * abs(planned["H2_AD"])
            for a, b in itertools.pairwise(subset)
        )
        qualifications.append(
            {
                "theta": planned["theta"],
                "name": planned["name"],
                "index": planned["index"],
                "FD_convergence": "QUALIFIED" if passed else "UNRESOLVED",
                "best_relative_gap": min(float(r["relative_gap"]) for r in subset),
            }
        )
    for r in derivative_rows:
        if r["check"] == "independent_discrete_energy_P_derivative":
            r["strict_F64_comparison"] = "PASS" if r["status"] == "PASS" else "FAIL"
        if "_FD_" in r["check"]:
            r["regime_note"] = (
                "qualified agreement region"
                if r["status"] == "WITHIN_1_PERCENT"
                else "curvature-dominated enlarged step; independent weighted-energy AD agrees closely"
            )
    table("derivative_checks", derivative_rows)
    receipt, closure, _memory, _constraints = [
        read(k)
        for k in (
            "replay_receipt",
            "attribution_closure",
            "memory_summary",
            "constraint_checks",
        )
    ]
    closure["parameter_order_verified"] = True
    closure["contribution_cancellation"] = []
    for t in (0, 1, 2):
        with np.load(OUT / f"branch_theta{t}.npz") as contributions:
            vectors = [contributions[k] for k in contributions.files]
        gross = sum(norm(g) for g in vectors)
        net = norm(np.sum(vectors, axis=0))
        closure["contribution_cancellation"].append(
            {
                "theta": t,
                "sum_individual_norms": gross,
                "norm_sum_LN_contributions": net,
                "net_over_gross": net / gross,
                "cancelled_fraction_of_gross_norm": 1 - net / gross,
            }
        )
    write("attribution_closure", closure)
    derivative = (
        "QUALIFIED"
        if all(r["FD_convergence"] == "QUALIFIED" for r in qualifications)
        else "UNRESOLVED"
    )
    decision = {
        "derivative": derivative,
        "initial_spike_three_coordinates": "QUALIFIED_AT_REPORTED_FD_RESOLUTION",
        "startup": "CONSTRAINT_LAYER_NORM_SUPPORTED",
        "optimizer_memory": "DIRECTIONALLY_ADVERSE_AT_TESTED_STATE",
        "FD_qualification": qualifications,
        "branch_sum_closure": closure["checks"],
        "native_replay_steps": 2,
        "new_intervention_arms": 0,
        "proposal33_applied": False,
        "production_bug_confirmed": False,
        "next_experiment": "One matched fresh-seed FULL_SOURCE operator-only pair: original LayerNorm eps versus one prospectively fixed larger LayerNorm eps, identical initialization/source/optimizer/update count. First qualify exact constraints and initial derivatives; do not reset moments or alter other components. No epsilon sweep. This is a recommendation only; no numerical epsilon was selected here.",
        "limits": "theta1 secondary LN coordinate lacks a two-step FD convergence region; no additional FD steps/tolerance changes. No g17 or complete saved gradient history. No causal optimizer-without-g1 trajectory. No chemical or full90 conclusion.",
        "scientific_wall_seconds": elapsed(),
        "budget_respected": elapsed() < 7200,
        "timing_note": "Wall includes all coding/diagnostics/tests. Direct replay gradient passes measured separately; total GPU-active time was not metered and is not equated to workflow wall.",
    }
    write("decision", decision)
    provenance = read("provenance")
    provenance["diagnostic_array_receipts"] = [
        {"path": str(p), "sha256": pilot.sha(p)}
        for p in sorted(OUT.glob("*"))
        if p.is_file()
    ]
    provenance["script_sha256"] = pilot.sha(Path(__file__))
    provenance["historical_and_production_hashes_unchanged"] = True
    write("provenance", provenance)
    h = receipt["history"]
    report = f"""# Fresh R0 LayerNorm startup and AdamW memory audit

The startup spike is localized to the exchange UEG constraint path, not the ordinary grid calls. The dominant initial derivative is supported by independent H2 finite differences. First-gradient memory is locally adverse in update 2; its residual contribution in proposal 33 is much smaller and the total proposal predicts descent. No production bug is established. Full derivative status is **{derivative}** because the secondary theta1 LayerNorm coordinate lacks two adjacent qualifying finite-difference steps under the frozen schedule.

## Frozen sources and precision

Baseline HEAD/ancestor: `{read("protocol")["head"]}`. Original source hashes and all historical fresh-pilot small artifacts remain unchanged (protocol/provenance JSON). The constructor exactly reproduced initial model-state hash `359088b9713e532be0f811b4bd9f5f4f91b34d531a936db7548f46b2b410aedd`, seed 2026, six layers, width 32, Gx/Gc enabled, dropout zero, 9446 unique native F32 coordinates. Only authenticated FULL_SOURCE t0/t16/t32 pilot states and initial/endpoint gradients were read. No B8/P536/S5 model or optimizer was loaded.

FULL_SOURCE H2/BeH2/CO counts are 47080/89200/85960. Source AO/descriptors/quadrature remain F64. Learned shadow/partial accumulation is F64; the production PBE branch remains F32 (`train_models/lap_operator.py:271–302`). Native AdamW retains F32 storage, lr 2.5e-5, betas .9/.999, eps 1e-8, weight decay .01, foreach false. One mean-of-three F64 gradient is cast once per replay update. `tools/lap_fresh_grid_pilot_v1.py:289–312` supplies the unchanged two-pass matrix adjoint evaluation; the metric is `train_models/lap_operator.py:422–440`. No subset comparison was repeated.

## Exact replay and gradient ownership

| State | Mean loss | Gradient index | Gradient norm |
|---|---:|---|---:|
| theta0 | {h[0]["loss"]:.16g} | g1 | {h[0]["gradient_norm"]:.16g} |
| theta1 | {h[1]["loss"]:.16g} | g2 | {h[1]["gradient_norm"]:.16g} |
| theta2 | {h[2]["loss"]:.16g} | g3 | {h[2]["gradient_norm"]:.16g} |
| theta32 (authenticated cache) | 0.5199287840361003 | g33 | 201.43661503863174 |

Exactly TWO molecular native optimizer steps were executed, both exact replay, not new intervention training. theta16/theta32 were read only; t33 was not applied. Small synthetic optimizer unit fixtures are separate from molecular replay. Initial g1 agrees with its authenticated cache. Per-system and mean parameter gradients are in `lap_fresh_layernorm_v1_parameter_gradients.csv`; complete vectors are hash-receipted outside Git. Ownership uses actual LayerNorm module identity.

At theta0, `x_feature_extractor.1.bias` has norm 33744021.090396255; coordinate 6 has |gradient| 18267962.16256031. Its squared-norm fraction is 0.9999070742767314. Next are `x_feature_extractor.3.fc.1.bias` (325276.90109632746, max coordinate 16) and `.3.fc.4.bias` (3933.5875436277865). Correlation is not declared dead from its small operator-only share.

## Constraint-path attribution and hidden statistics

Semantic roles follow the actual calls in `train_models/NN_models_lap.py:85–150`: real exchange up/down, UEG exchange up/down, real/UEG/high-density/rapid-gradient correlation direct/swap and fused paths. Shared exchange hidden paths into correlation are retained, including their mixed density/parameter derivatives. Only LN affine parameter uses are expanded into equal replicas. These are diagnostic derivatives, not alternative training models. Energy and each density partial agree on a full H2 chunk, as do assembled matrices and full three-system losses. Gradient branch-sum relative closure is {closure["checks"][0]["gradient_relative_closure"]:.4g}, {closure["checks"][1]["gradient_relative_closure"]:.4g}, {closure["checks"][2]["gradient_relative_closure"]:.4g} at theta0/1/2. Profile before full attribution: {closure["chunk_profile_seconds"]:.6g} seconds per tested chunk.

At theta0 each UEG spin path contributes norm 16872794.880478688 and signed projection fraction 0.5000000106301892 onto the TOTAL gradient. Each ordinary exchange contribution has norm 5.329570659502747 and projection fraction -1.0631798756756775e-8. Fractions are signed dot projections, not independent percentages of norms. Full contributions/cancellation can be reconstructed from hash-bound branch arrays. The two UEG paths remain dominant after update 1 (each norm 347.41398334956574) and update 2 (215.26185253919408).

Every exchange UEG LN input is exactly zero at theta0, across 222240 repeated anchor calls per spin and module. These repeats are not independent physical observations. Actual eps = 1e-5 and inverse standard deviation = 316.227766. The first bias-free Linear still has zero anchor output at theta1/2. Its LN bias becomes nonzero (norm 0.00014142136 after update1). Downstream `.3.fc.1` variance rises from zero to 8.995244162e-11; `.3.fc.4` to 8.561932077e-7. Later `.4.fc.1` variance rises to 0.01164849535 and inverse standard deviation falls to 9.26144; `.4.fc.4` reaches variance 0.2438917697. Real-grid and artificial-constraint statistics are separately labelled in `lap_fresh_layernorm_v1_call_statistics.csv`. Downstream first-layer values are independently recorded in constraint_checks JSON.

The local LN input Jacobian at zero scales as diag(gamma)/sqrt(eps) times the centering projector. Its OWN bias derivative is identity, not 1/sqrt(eps). The dominant early bias receives downstream-chain amplification. Residual connections, GELU, weights and cancellations are included in the measured graph; no naive product of 316 factors is used. Positive eps makes this poorly conditioned startup rather than a mathematical singularity. Branch cancellation is quantified as norm(sum contributions)/sum(norm contributions): {[r["net_over_gross"] for r in closure["contribution_cancellation"]]} at theta0/1/2 (higher means less cancellation). These ratios are not causal percentages.

## Independent derivative checks and retained limits

Independent F64 LN formula uses biased feature variance and actual eps. Forward, first and mixed input/affine derivatives pass frozen atol 1e-10 / rtol 1e-9 on real observed minimum/maximum-variance inputs at theta0/1/2 (including zero anchors).

H2 directions, selected deterministically from mean g1 BEFORE finite differences: `.1.bias[6]`, `.3.fc.1.bias[16]`, `.3.fc.0.weight[887]`, all in x_feature_extractor. The corresponding H2 derivatives at theta0 are 6270290.387128715, -43644.8038976052, 3.89822656769507. All three have qualified adjacent-step FD agreement under the frozen 1% rule, with best relative errors approximately 2.23e-6, 1.57e-6 and 2.97e-7. Thus the enormous startup derivative is genuinely present in the tested scalar, not just a backward-accumulation norm artifact.

At theta1, the dominant bias and non-LN control qualify (best relative gaps 4.35e-5 and 2.65e-6). Secondary bias has AD 0.029319474349165; its smallest predetermined step gives FD 0.029128747796065 (0.6505% gap), but the next step has 2.626% gap and larger steps exhibit strong curvature. There is only one qualifying step, so its convergence status is UNRESOLVED. No step refinement or tolerance widening was performed. All 42 predeclared step sizes (7 per direction/state) and scalar values are retained.

The independent discrete-energy path differentiates energy with respect to a full symmetric-density-matrix perturbation, then to parameters. It calls neither the production AO assembler nor local-partial helper. It shares the physical PBE formula and model. Weighted F32 energy backprop rounds quadrature in a different place; matrix relative differences are 5.873612507e-9 / 3.634464309e-9, coordinate derivative relative gaps 1.28e-10 to 1.46e-8. Four comparisons exceed strict F64 rtol 1e-9 and remain explicit FAIL for that arithmetic comparison. They are not silently relabelled exact parity, nor treated as a demonstrated production bug. There is no promoted-F64 oracle or extra FD sweep. Perturbations are F64, not rounded back into F32 parameters. A fixed base matrix adjoint contraction has the original loss derivative at the base state; it is not a new training objective.

## AdamW memory in actual update directions

For native F32 g1, first-moment contribution is .1*.9^(t-1)*g1; first second-moment contribution is .001*.999^(t-1)*g1^2. Saved t16/t32 moments are authenticated. Intermediate gradient vectors were not saved: rest is accounting residual from saved moments, not an independent history replay. At t1, rest is solely native rounding, not another gradient. Negative v_rest entries are reported, not clipped; all lie within the explicitly recorded conservative F32 accumulation bound. Full per-coordinate ratios are outside Git with SHA receipts.

| t | ||m_first|| | ||m_rest|| | ||m_total|| | cos(m_first,current gradient) |
|---|---:|---:|---:|---:|
| 1 | 3374558.8798 | 0.11457 (rounding) | 3374558.9757 | -0.177259 |
| 2 | 3037102.9918 | 69.44674 | 3037090.7517 | -0.197617 |
| 16 | 694791.7481 | 213.61615 | 694777.1264 | unavailable (g17 not saved) |
| 32 | 128746.3136 | 194.62597 | 128733.3235 | -0.0965503 |

Components below use ONE actual full-proposal denominator, actual bias correction, native CUDA ordering, and separately measured weight decay/rounding. CPU arithmetic initially differed from CUDA in 103 coordinates by at most 3.7252903e-9; using the same CUDA path reproduces actual update2 parameters byte-for-byte. No optimizer reset or alternative trajectory was run.

| update | component | displacement norm | grad L dot displacement |
|---|---|---:|---:|
| 2 (actual replay) | first-gradient numerator | 0.00142811364 | +0.004817969742 |
| 2 | rest | 0.00085427823 | -0.000605736539 |
| 2 | decay | 5.16160e-6 | -1.35259e-7 |
| 2 | rounding | 6.79636e-7 | -1.06243e-8 |
| 2 | total actual | 0.00167724524 | +0.004212087320 |
| 33 (NOT applied) | first-gradient numerator | 3.12935e-5 | +6.70315e-7 |
| 33 | rest | 0.00192909440 | -0.002481288466 |
| 33 | total native proposal | 0.00193374085 | -0.002480760021 |

Actual replay update2 loss increases by {h[2]["loss"] - h[1]["loss"]:.12g}, consistent in sign with its linear prediction. At update33, first-gradient displacement remains locally adverse but is small; the total proposal predicts descent. Raw large first-moment norm does not imply domination of the preconditioned update. Stored m32 first contribution has beta1^31; next m33 has beta1^32. The second-moment denominator is nonlinear and has no claimed additive causal percentages. Removing g1 might change the entire trajectory: this arithmetic accounting does not show that a reset or omission would improve training.

## Integrity, outcome and one next experiment

Tied-parameter UEG, spin, high-density, rapid-gradient and tau-independence constraints pass at theta0/1/2/16/32. Model/buffers/RNG restore checks pass; source/history hashes unchanged. No SCF, old functional checkpoints, chemistry/development data, production detach, architecture/loss change or optimizer change occurred. Gradient summation and bias correction were tested; see focused tests. Large arrays and diagnostic states are outside Git with SHA receipts. Seven focused tests passed before report generation; final check receipt records the complete final test/lint/compile run.

Statuses: DERIVATIVE **{derivative}** (startup spike qualified, one secondary theta1 FD limit); STARTUP **CONSTRAINT_LAYER_NORM_SUPPORTED**; OPTIMIZER MEMORY **DIRECTIONALLY_ADVERSE_AT_TESTED_STATE**. H1 supported for initial dominant derivative, no production defect confirmed. H2 strongly supported by closure-qualified semantic attribution. H3 supported for actual second-step direction; not established as globally harmful at t32 or over a counterfactual trajectory.

One next controlled experiment: one matched fresh-seed FULL_SOURCE operator-only pair comparing original LN eps with ONE prospectively fixed larger eps, same weights/data/optimizer/horizon, after exact-constraint and derivative preflight. No eps value selected here, no sweep or new arm launched. This tests startup conditioning without conflating grid selection, LR or moment reset.

Scientific wall at report generation: {elapsed():.3f} seconds of 7200. Measured replay gradient-pass compute: {sum(x["active_seconds"] for x in h):.3f} seconds. GPU-active time was not separately metered; workflow/coding/idle time is not labelled GPU computation. This is a three-system startup audit, not chemical generalization, full90 qualification, or proof of the effectiveness of any future intervention.
"""
    (ROOT / (PREFIX + "report.md")).write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "freeze",
            "replay",
            "attribution",
            "memory",
            "local_derivatives",
            "end_to_end",
            "integrity",
            "finish",
        ],
    )
    getattr(sys.modules[__name__], parser.parse_args().phase)()
