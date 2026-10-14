"""One fresh initialization and two matched operator-only grid trajectories."""

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train_models"))
from dft_functionals.constants import NN_OUTPUT_SCALE_PBE
from train_models import lap_operator as op
from train_models.lap_moo_training import named_trainable_parameters
from train_models.lap_vxc import LapEnergy
from train_models.NN_models_lap import pcPBELMLOptimizerV2Lap

PREFIX = "lap_fresh_grid_pilot_v1_"
OUT = ROOT.parent / "lap_fresh_grid_pilot_v1_runs_20261014"
OUT.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def local(path):
    return (
        Path("C:/" + path[7:])
        if sys.platform == "win32" and path.startswith("/mnt/c/")
        else Path(path)
    )


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


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def vector(model):
    return torch.cat(
        [
            p.detach().double().flatten().cpu()
            for p in named_trainable_parameters(model).values()
        ]
    ).numpy()


def rng():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(saved):
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    if saved[3]:
        torch.cuda.set_rng_state_all(saved[3])


def constraints(model):
    m = copy.deepcopy(model).cpu().double()
    raw = torch.tensor(
        [
            [0.8, 0.4, 0.03, 0.04, 0.05, 0.0, 0.0, 0.01, -0.02],
            [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    base = m(raw)
    changed = raw.clone()
    changed[:, 5:7] = 1e100
    swap = raw[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]]
    spin = float(
        (m(swap) - base[:, list(range(22)) + [24, 25, 22, 23, 27, 26, 28]]).abs().max()
    )
    d = m.get_correlation_descriptors(raw)
    ue = m.all_sigma_zero(d)
    hi = m.all_rho_inf(d)
    si = m.all_s_inf(d)
    u = m.forward_descriptors(ue, ue) / NN_OUTPUT_SCALE_PBE.to(d)
    high = m.forward_descriptors(hi, d) / NN_OUTPUT_SCALE_PBE.to(d)
    rapid = m.forward_descriptors(si, d) / NN_OUTPUT_SCALE_PBE.to(d)
    errors = {
        "tau": float((m(changed) - base).abs().max()),
        "spin": spin,
        "UEG": float((u[:, [0, 23, 25, 28]] - 1).abs().max()),
        "Gx_UEG": float(u[:, 26:28].abs().max()),
        "high_density": float((high[:, [1, 28]] - 1).abs().max()),
        "rapid_gradient": float((rapid[:, 28] - 1).abs().max()),
    }
    return {
        "errors": errors,
        "finite": bool(torch.isfinite(base).all()),
        "PASS": max(errors.values()) <= 1e-12 and bool(torch.isfinite(base).all()),
    }


def check_budget(reserve=0):
    p = read("protocol")
    if time.time() - p["first_action_unix"] + reserve >= p["wall_budget_seconds"]:
        raise TimeoutError("Frozen scientific budget reserve reached")


def prepare():
    from pyscf import gto
    from pyscf.dft import numint

    from train_models.lap_operator_data import density_descriptors_from_ao

    p = read("protocol")
    old = json.loads((ROOT / "lap_grid_transfer_v1_provenance.json").read_text())
    records = {}
    start = time.monotonic()
    for name in p["systems"]:
        check_budget(900)
        r = old["pilots"][name]
        for key in ["source", "corpus", "arrays"]:
            assert sha(local(r[key]["path"])) == r[key]["sha256"]
        with np.load(local(r["source"]["path"])) as z:
            data = {k: z[k] for k in z.files}
        with h5py.File(local(r["corpus"]["path"])) as f:
            g = f[r["corpus"]["group"]]
            mapping = g["npzrow"][:].astype(int)
            assert np.array_equal(g["coords64"][:], data["grid_coords"][mapping])
            metadata = json.loads(g.attrs["source_metadata_json"])
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
        assert mol.ao_labels() == metadata["molecule"]["ao_labels"]
        coords = data["grid_coords"]
        w = data["grid_weights"].reshape(-1)
        v = data["vxc_grid"].reshape(-1)
        n = mol.nao_nr()
        counts = np.bincount(mapping, minlength=len(w))
        destination = OUT / (name + "_SOURCE.h5")
        refs = [np.zeros((n, n)), np.zeros((n, n))]
        with h5py.File(destination, "w") as f:
            f.create_dataset("map", data=mapping)
            f.create_dataset("weights", data=w)
            f.create_dataset("vxc", data=v)
            f.create_dataset("coords", data=coords)
            f.create_dataset("S", data=mol.intor_symmetric("int1e_ovlp"))
            for k, shape in [
                ("phi", (len(w), n)),
                ("grad", (len(w), 3, n)),
                ("lap", (len(w), n)),
                ("features", (len(w), 10)),
            ]:
                f.create_dataset(
                    k, shape=shape, dtype="f8", chunks=(min(1024, len(w)),) + shape[1:]
                )
            for startrow in range(0, len(w), 2048):
                check_budget(900)
                sl = slice(startrow, startrow + 2048)
                ao = numint.eval_ao(mol, coords[sl], deriv=2)
                phi = ao[0]
                f["phi"][sl] = phi
                f["grad"][sl] = ao[1:4].transpose(1, 0, 2)
                f["lap"][sl] = ao[4] + ao[7] + ao[9]
                f["features"][sl] = density_descriptors_from_ao(mol, ao, data["dm_ks"])
                for k, weight in enumerate([w[sl], w[sl] * counts[sl]]):
                    refs[k] += phi.T @ (phi * (weight * v[sl])[:, None])
            f.create_dataset("ref_FULL_SOURCE", data=refs[0])
            f.create_dataset("ref_SUBSET", data=refs[1])
        with np.load(local(r["arrays"]["path"])) as previous:
            gaps = [
                float(np.linalg.norm(refs[0] - previous["Vref_R0"])),
                float(np.linalg.norm(refs[1] - previous["Vref_R2"])),
            ]
            assert all(
                g <= 1e-10 + 1e-10 * np.linalg.norm(ref) for g, ref in zip(gaps, refs)
            )
        records[name] = {
            "path": str(destination),
            "sha256": sha(destination),
            "source": r["source"],
            "corpus_mapping_source": r["corpus"],
            "SOURCE_count": len(w),
            "SUBSET_count": len(mapping),
            "n_AO": n,
            "dtype": "F64 source inputs and all AO/descriptor arrays",
            "reference_replay_gaps": gaps,
            "selection_multiplicity_preserved": True,
        }
    write(
        "representations",
        {
            "records": records,
            "prepare_seconds": time.monotonic() - start,
            "old_models_loaded": 0,
        },
    )


def chunk_arrays(record, grid, size=2048):
    with h5py.File(local(record["path"])) as f:
        indices = f["map"][:] if grid == "SUBSET" else np.arange(len(f["weights"]))
        for start in range(0, len(indices), size):
            ix = indices[start : start + size]
            unique, inverse = np.unique(ix, return_inverse=True)
            yield {
                k: f[k][unique][inverse]
                for k in ["phi", "grad", "lap", "features", "weights"]
            }


def chunk_matrix(shadow, arrays, device, graph):
    t = {
        k: torch.as_tensor(v, dtype=torch.float64, device=device)
        for k, v in arrays.items()
    }
    e, c, a, b = op._model_double_operator_partials(
        LapEnergy(shadow), t["features"], graph
    )
    tensors = (e, c, a, b)
    if not all(torch.isfinite(v).all() for v in tensors):
        bad = {
            str(i): torch.nonzero(~torch.isfinite(v).reshape(len(v), -1).all(-1))
            .flatten()
            .cpu()
            .tolist()
            for i, v in enumerate(tensors)
        }
        raise FloatingPointError("Nonfinite local energy/partials: " + json.dumps(bad))
    matrix = op._assemble_from_local_partials(
        c.mean(-1),
        a.mean(1),
        b.mean(-1),
        t["weights"],
        t["phi"],
        t["grad"],
        t["lap"],
        2048,
    )
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Nonfinite AO matrix")
    return matrix


def system_gradient(shadow, record, grid, device, need_grad=True):
    with h5py.File(local(record["path"])) as f:
        ref = torch.tensor(f["ref_" + grid][:], device=device)
        S = torch.tensor(f["S"][:], device=device)
    V = torch.zeros_like(ref)
    for arrays in chunk_arrays(record, grid):
        V += chunk_matrix(shadow, arrays, device, False).detach()
    leaf = V.detach().requires_grad_(True)
    loss = op.operator_loss(leaf, ref, S)
    upstream = torch.autograd.grad(loss, leaf)[0].detach()
    names = named_trainable_parameters(shadow)
    grads = [torch.zeros_like(p) for p in names.values()]
    if need_grad:
        for arrays in chunk_arrays(record, grid):
            matrix = chunk_matrix(shadow, arrays, device, True)
            values = torch.autograd.grad(
                (matrix * upstream).sum(), tuple(names.values()), allow_unused=True
            )
            for total, p, value in zip(grads, names.values(), values):
                total.add_(torch.zeros_like(p) if value is None else value)
    result = torch.cat([g.flatten() for g in grads])
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite parameter gradient " + grid)
    return float(loss), result


def batch(model, shadow, records, grid, device, need_grad=True):
    shadow.load_state_dict(model.state_dict(), strict=True)
    values = []
    grads = []
    for name, record in records.items():
        check_budget(600)
        try:
            value, g = system_gradient(shadow, record, grid, device, need_grad)
        except FloatingPointError as error:
            write(
                "numerical_blocker",
                {
                    "system": name,
                    "grid": grid,
                    "operation": str(error),
                    "subset_mapping": record["corpus_mapping_source"],
                    "no_rows_removed": True,
                },
            )
            raise
        values.append(value)
        grads.append(g)
    return values, torch.stack(grads).mean(0), torch.stack(grads)


def gradient_metrics(a, b):
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    gap = float(np.linalg.norm(a - b))
    return {
        "norm_SUBSET": an,
        "norm_FULL_SOURCE": bn,
        "absolute_difference": gap,
        "relative_difference": gap / bn if bn > 0 else None,
        "cosine": float(a @ b / an / bn) if an > 0 and bn > 0 else None,
    }


def blocks(model):
    # Authenticated F013 mapping: LayerNorm by module identity, then network ownership.
    norms = set()
    for parent, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            norms.update(
                parent + "." + n for n, _ in module.named_parameters(recurse=False)
            )
    result = {}
    offset = 0
    for name, p in named_trainable_parameters(model).items():
        kind = (
            "normalization"
            if name in norms
            else "exchange"
            if name.startswith(("x_feature_extractor.", "x_output_layer."))
            else "correlation"
            if name.startswith("c_")
            else "remaining"
        )
        result.setdefault(kind, []).extend(range(offset, offset + p.numel()))
        offset += p.numel()
    assert offset == 9446
    return result


def fixture_parity(shadow, record, device):
    arrays = next(chunk_arrays(record, "FULL_SOURCE", 7))
    params = tuple(named_trainable_parameters(shadow).values())
    full = chunk_matrix(shadow, arrays, device, True)
    ref = full.detach() * 0.91
    S = torch.eye(len(full), dtype=torch.float64, device=device)
    loss = op.operator_loss(full, ref, S)
    direct = torch.autograd.grad(loss, params, allow_unused=True)
    leaf = full.detach().requires_grad_(True)
    u = torch.autograd.grad(op.operator_loss(leaf, ref, S), leaf)[0].detach()
    acc = [torch.zeros_like(p) for p in params]
    for sl in [slice(0, 3), slice(3, 7)]:
        M = chunk_matrix(shadow, {k: v[sl] for k, v in arrays.items()}, device, True)
        gs = torch.autograd.grad((M * u).sum(), params, allow_unused=True)
        for a, p, g in zip(acc, params, gs):
            a += torch.zeros_like(p) if g is None else g
    dv = torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten()
            for p, g in zip(params, direct)
        ]
    )
    av = torch.cat([g.flatten() for g in acc])
    gap = float((dv - av).norm())
    bound = 1e-10 + 1e-9 * float(dv.norm())
    assert gap <= bound
    return {"adjoint_vs_direct_gradient_gap": gap, "bound": bound, "PASS": True}


def checkpoint(path, models, optimizers, counters):
    payload = {
        "namespace": PREFIX,
        "protocol_sha256": sha(ROOT / (PREFIX + "protocol.json")),
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "counters": counters.copy(),
        "RNG": rng(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha(path)}


def resume_payload(receipt):
    path = local(receipt["path"])
    assert path.parent.resolve() == OUT.resolve() and sha(path) == receipt["sha256"]
    value = torch.load(path, weights_only=False, map_location="cpu")
    assert value["namespace"] == PREFIX and value["protocol_sha256"] == sha(
        ROOT / (PREFIX + "protocol.json")
    )
    return value


def restore_checkpoint(receipt, models, optimizers):
    value = resume_payload(receipt)
    assert set(value["models"]) == set(models) and set(value["optimizers"]) == set(
        optimizers
    )
    for name, model in models.items():
        model.load_state_dict(value["models"][name], strict=True)
        optimizers[name].load_state_dict(value["optimizers"][name])
    restore_rng(value["RNG"])
    return value["counters"].copy()


def run():
    p = read("protocol")
    representations = read("representations")
    records = representations["records"]
    start = time.monotonic()
    for file, expected in p["source_hashes"].items():
        assert sha(ROOT / file) == expected
    for record in records.values():
        assert sha(local(record["path"])) == record["sha256"]
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial = pcPBELMLOptimizerV2Lap(**p["constructor"])
    names = named_trainable_parameters(initial)
    assert (
        sum(v.numel() for v in names.values()) == 9446
        and initial.use_g_x
        and initial.use_g_c
    )
    models = {grid: copy.deepcopy(initial).to(device) for grid in p["arms"]}
    assert (
        state_hash(models["SUBSET"])
        == state_hash(models["FULL_SOURCE"])
        == state_hash(initial)
    )
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(
            models["SUBSET"].parameters(), models["FULL_SOURCE"].parameters()
        )
    )
    shadow = copy.deepcopy(initial).to(device).double()
    optimizers = {
        k: torch.optim.AdamW(
            named_trainable_parameters(m).values(),
            **{**p["optimizer"], "betas": tuple(p["optimizer"]["betas"])},
        )
        for k, m in models.items()
    }
    assert all(not o.state for o in optimizers.values())
    theta0 = vector(initial)
    mapping = [
        {"name": n, "shape": list(v.shape), "dtype": str(v.dtype)}
        for n, v in names.items()
    ]
    write(
        "initialization",
        {
            "seed": 2026,
            "config": initial.model_kwargs,
            "coordinates": 9446,
            "state_hash": state_hash(initial),
            "parameter_order": mapping,
            "parameter_order_sha256": hashlib.sha256(
                json.dumps(mapping).encode()
            ).hexdigest(),
            "independent_equal_copies": True,
            "fresh_optimizer_states": True,
        },
    )
    saved_rng = rng()
    snapshot = state_hash(models["SUBSET"])
    constraint_log = {"t0": constraints(initial)}
    assert constraint_log["t0"]["PASS"]
    parity = fixture_parity(shadow, records["H2"], device)
    grads_cache = {}
    gradient_rows = []
    cross = []
    logs = []
    times = {}
    checkpoints = []
    counters = {k: 0 for k in models}
    for grid in p["arms"]:
        before = time.monotonic()
        vals, g, per = batch(models["SUBSET"], shadow, records, grid, device)
        torch.cuda.synchronize() if device == "cuda" else None
        times[grid] = time.monotonic() - before
        grads_cache[grid] = (vals, g.detach().cpu().numpy(), per.detach().cpu().numpy())
    _same_vals, same_g, _ = batch(
        models["FULL_SOURCE"], shadow, records, "SUBSET", device
    )
    same_gap = float(np.linalg.norm(same_g.cpu().numpy() - grads_cache["SUBSET"][1]))
    assert same_gap <= 1e-10 + 1e-9 * np.linalg.norm(grads_cache["SUBSET"][1])
    assert state_hash(models["SUBSET"]) == snapshot and all(
        not o.state for o in optimizers.values()
    )
    restore_rng(saved_rng)
    assign = blocks(initial)

    def comparisons(label, a, b):
        gradient_rows.append(
            {"state": label, "block": "full", **gradient_metrics(a, b)}
        )
        for block, indices in assign.items():
            gradient_rows.append(
                {
                    "state": label,
                    "block": block,
                    **gradient_metrics(a[indices], b[indices]),
                }
            )

    comparisons("theta0", grads_cache["SUBSET"][1], grads_cache["FULL_SOURCE"][1])
    np.savez_compressed(
        OUT / "initial_gradients.npz",
        **{k + "_gradient": v[1] for k, v in grads_cache.items()},
        **{k + "_system_gradients": v[2] for k, v in grads_cache.items()},
    )
    projected = (32 + 5) * (times["SUBSET"] + times["FULL_SOURCE"]) + 900
    remaining = 10800 - (time.time() - p["first_action_unix"])
    write(
        "resource_report",
        {
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "complete_batch_seconds": times,
            "projected_remaining_seconds": projected,
            "remaining_wall_seconds": remaining,
            "peak_GPU_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else 0,
            "GO": projected < remaining,
            "prepare_seconds": representations["prepare_seconds"],
        },
    )
    write(
        "preflight",
        {
            "sources": "PASS",
            "reference_replay": "PASS",
            "constraints": "PASS",
            "same_grid_gradient_gap": same_gap,
            "matrix_adjoint_parity": parity,
            "fresh_states_unchanged": True,
            "RNG_restored": True,
            "GO": projected < remaining,
        },
    )
    if projected >= remaining:
        raise TimeoutError(
            "Both trajectories and mandatory final qualification do not fit"
        )

    def evaluate(t, with_gradient=False):
        endpoint = {}
        for trained, model in models.items():
            endpoint[trained] = {}
            for grid in p["arms"]:
                if t == 0:
                    vals, g, _ = grads_cache[grid]
                else:
                    vals, gg, _ = batch(
                        model, shadow, records, grid, device, with_gradient
                    )
                    g = gg.detach().cpu().numpy()
                endpoint[trained][grid] = g
                row = {
                    "t": t,
                    "trained_grid": trained,
                    "evaluated_grid": grid,
                    "mean_loss": float(np.mean(vals)),
                    "displacement_norm": float(np.linalg.norm(vector(model) - theta0)),
                    "finite": bool(np.isfinite(vals).all()),
                }
                for name, value in zip(records, vals):
                    row[name + "_loss"] = value
                row["improvement_from_theta0"] = float(
                    np.mean(grads_cache[grid][0]) - np.mean(vals)
                )
                cross.append(row)
            if with_gradient:
                comparisons(
                    trained + "_t32",
                    endpoint[trained]["SUBSET"],
                    endpoint[trained]["FULL_SOURCE"],
                )
        table("cross_evaluation", cross)
        table("gradient_comparison", gradient_rows)
        return endpoint

    evaluate(0)
    checkpoints.append(
        checkpoint(OUT / "initialization.pt", {"initial": initial}, {}, {"initial": 0})
    )
    for grid in p["arms"]:
        checkpoints.append(
            checkpoint(
                OUT / (grid + "_t0.pt"),
                {grid: models[grid]},
                {grid: optimizers[grid]},
                {grid: 0},
            )
        )
    resume_receipt = checkpoint(OUT / "resume.pt", models, optimizers, counters)
    restored_counters = restore_checkpoint(resume_receipt, models, optimizers)
    assert restored_counters == counters and all(
        state_hash(models[k]) == snapshot for k in models
    )
    for t in range(1, 33):
        check_budget((33 - t) * (times["SUBSET"] + times["FULL_SOURCE"]) + 900)
        for grid, model in models.items():
            before = time.monotonic()
            vals, g, _ = batch(model, shadow, records, grid, device)
            optimizer = optimizers[grid]
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for parameter in named_trainable_parameters(model).values():
                parameter.grad = (
                    g[offset : offset + parameter.numel()]
                    .reshape(parameter.shape)
                    .to(parameter.dtype)
                )
                offset += parameter.numel()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            counters[grid] += 1
            assert all(torch.isfinite(v).all() for v in model.parameters())
            assert all(int(s["step"]) == t for s in optimizer.state.values())
            logs.append(
                {
                    "grid": grid,
                    "update": t,
                    "optimizer_counter": t,
                    "mean_preupdate_loss": float(np.mean(vals)),
                    "gradient_norm": float(g.norm()),
                    "displacement_from_t0": float(
                        np.linalg.norm(vector(model) - theta0)
                    ),
                    "seconds": time.monotonic() - before,
                    "system_presentations": 3,
                    "H2_loss": vals[0],
                    "BeH2_loss": vals[1],
                    "CO_loss": vals[2],
                }
            )
        resume_receipt = checkpoint(OUT / "resume.pt", models, optimizers, counters)
        table("training_log", logs)
        if t in [16, 32]:
            for grid in p["arms"]:
                c = constraints(models[grid])
                constraint_log[grid + "_t" + str(t)] = c
                assert c["PASS"]
                checkpoints.append(
                    checkpoint(
                        OUT / (grid + "_t" + str(t) + ".pt"),
                        {grid: models[grid]},
                        {grid: optimizers[grid]},
                        {grid: t},
                    )
                )
            final_gradients = evaluate(t, t == 32)
            if t == 32:
                np.savez_compressed(
                    OUT / "endpoint_gradients.npz",
                    **{
                        trained + "_" + grid: g
                        for trained, values in final_gradients.items()
                        for grid, g in values.items()
                    },
                )
        print("Completed paired update", t, flush=True)
    write("constraint_checks", constraint_log)
    final = {
        r["trained_grid"]: r
        for r in cross
        if r["t"] == 32 and r["evaluated_grid"] == "FULL_SOURCE"
    }
    sub, full = final["SUBSET"]["mean_loss"], final["FULL_SOURCE"]["mean_loss"]
    benefit = (sub - full) / sub if sub > 0 else None
    dsub = vector(models["SUBSET"]) - theta0
    dfull = vector(models["FULL_SOURCE"]) - theta0
    moment = {}
    for grid, opt in optimizers.items():
        moment[grid] = {
            key: float(
                torch.cat(
                    [s[key].flatten().double() for s in opt.state.values()]
                ).norm()
            )
            for key in ["exp_avg", "exp_avg_sq"]
        }
    original = float(np.mean(grads_cache["FULL_SOURCE"][0]))
    resolved = max(original - sub, original - full) > 1e-10 + 1e-9 * original
    classification = (
        "A"
        if benefit is not None
        and benefit >= 0.01
        and full < original
        and abs(sub - full) > 1e-10 + 1e-9 * sub
        else "B"
        if resolved
        else "D"
    )
    write(
        "decision",
        {
            "classification": classification,
            "updates": counters,
            "SOURCE_initial_loss": original,
            "SUBSET_endpoint_SOURCE_loss": sub,
            "FULL_endpoint_SOURCE_loss": full,
            "B": benefit,
            "effect_floor_PASS": classification == "A",
            "endpoint_parameter_distance": float(
                np.linalg.norm(vector(models["SUBSET"]) - vector(models["FULL_SOURCE"]))
            ),
            "displacement_cosine": float(
                dsub @ dfull / np.linalg.norm(dsub) / np.linalg.norm(dfull)
            ),
            "moment_norms": moment,
            "wall_seconds": time.time() - p["first_action_unix"],
            "active_run_seconds": time.monotonic() - start,
            "budget_respected": time.time() - p["first_action_unix"] < 10800,
            "old_trained_checkpoints_loaded": 0,
            "old_optimizer_states_loaded": 0,
        },
    )
    write(
        "provenance",
        {
            "source_hashes_unchanged": all(
                sha(ROOT / file) == expected
                for file, expected in p["source_hashes"].items()
            ),
            "pilot_checkpoint_receipts": checkpoints + [resume_receipt],
            "fresh_checkpoint_files_created": len(checkpoints) + 1,
            "gradient_receipts": [
                {"path": str(OUT / f), "sha256": sha(OUT / f)}
                for f in ["initial_gradients.npz", "endpoint_gradients.npz"]
            ],
            "driver_sha256": sha(__file__),
            "representation_manifest_sha256": sha(
                ROOT / (PREFIX + "representations.json")
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    try:
        prepare() if args.mode == "prepare" else run()
    except Exception as error:
        write(
            "decision",
            {
                "classification": "C" if not isinstance(error, TimeoutError) else "D",
                "error": repr(error),
                "no_automatic_retry": True,
                "wall_seconds": time.time() - read("protocol")["first_action_unix"],
            },
        )
        raise
