"""F028: one fresh seed, original corpus/objectives, matched epsilon trajectories."""

import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "train_models"), str(ROOT)]

import lap_b8_adamw as b8
import lap_exchange_eps_v1 as eps_helpers
import lap_f010_uniform_normalized_chemistry as f010
import lap_f027_exchange_eps as identity_helpers
import lap_fresh_grid_pilot_v1 as pilot
import lap_training_metric_gradient_audit as chemistry
import numpy as np
import torch

from train_models import lap_moo_training as core
from train_models.lap_fixed_adamw import adamw_step, weighted_gradient
from train_models.lap_vxc import LapEnergy, integrated_energy
from train_models.optuna_joint import _add_mrks_dispersion_once, batch_exc
from train_models.publication_data import PublicationDataset
from train_models.publication_data.loader import Handles

PREFIX = "lap_f028_joint_eps_"
OUT = ROOT.parent / "lap_f028_joint_eps_runs_20261015"
DATA = ROOT.parent / "publication_dataset_v1"
ANCESTOR = "86d6bdb6e85319724b134a11e216ff428dc6fbb8"
ARMS = {"CONTROL": 1e-5, "INTERVENTION": 1e-3}
CONFIG = identity_helpers.CONFIG
TASKS = b8.TASKS
C010 = 9.974823077811354
LIMIT = 72000
MILESTONES = (0, 64, 128, 192, 251)
# Conservative earlier timestamp: the last verified F027 remote-HEAD action.
# It includes the interrupted, read-only F028 source checks before the full prompt.
START_MS = 1792053056891


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file(name):
    return ROOT / (PREFIX + name + ".json")


def write(name, value):
    b8.write_json(file(name), value)


def scoped_stop(reason):
    # Reused historical helpers must never write their old stop/report paths.
    write("source_failure", {"reason": str(reason), "training_stopped": True})
    raise RuntimeError(str(reason))


for _historical_helper in (b8, f010, chemistry):
    _historical_helper._stop = scoped_stop


def elapsed():
    return time.time() - START_MS / 1000


def budget(reserve=0):
    if elapsed() + reserve >= LIMIT:
        raise TimeoutError("F028 frozen wall budget/reserve exhausted")


def table(name, rows):
    if not rows:
        return
    path = ROOT / (PREFIX + name + ".csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def scientific_identity(model, arm):
    eps_helpers.assert_epsilon(model, ARMS[arm])
    names = core.named_trainable_parameters(model)
    assert sum(p.numel() for p in names.values()) == 9446
    config = {
        "architecture": model.model_kwargs,
        "epsilon": eps_helpers.epsilon_rows(model),
        "order": [{"name": n, "shape": list(p.shape)} for n, p in names.items()],
        "precision_hash": read(file("protocol"))["precision_hash"],
    }
    return {
        **config,
        "scientific_configuration_hash": b8.sha256_text(
            json.dumps(config, sort_keys=True)
        ),
        "parameter_hash": identity_helpers.tensor_subhash(model, "parameters"),
        "buffer_hash": identity_helpers.tensor_subhash(model, "buffers"),
    }


def synchronize(model, shadow, arm):
    eps_helpers.assert_epsilon(model, ARMS[arm])
    shadow.load_state_dict(model.state_dict(), strict=True)
    eps_helpers.assert_epsilon(shadow, ARMS[arm])
    assert tuple(core.named_trainable_parameters(model)) == tuple(
        core.named_trainable_parameters(shadow)
    )


class TrainingBundle(PublicationDataset):
    """Same F015 training-only index boundary, with F028 hash qualification."""

    def __init__(self):
        self.root = DATA.resolve()
        contract = read(file("source_verification"))
        assert (
            b8.sha256_file(self.root / "dataset_manifest.json")
            == contract["dataset_manifest_sha256"]
        )
        self.manifest = read(self.root / "dataset_manifest.json")
        assert self.manifest["logical_sha256"] == b8.DATA_SHA
        assert self.manifest["status"] == "qualified"
        self.checksums = {r["file"]: r["sha256"] for r in self.manifest["files"]}
        self.reactions = self._index("chemistry/reactions.jsonl")
        self.species = self._index("chemistry/species.jsonl")
        self.systems = self._index("mrks/systems.jsonl")
        splits = read(self.root / "splits.json")
        assert (
            b8.sha256_file(self.root / "splits.json") == self.checksums["splits.json"]
        )
        assert not {"test", "future"} & splits.keys()
        self.splits = {
            k: splits[k] for k in ("train_relchem", "train_ae17", "train_mrks")
        }
        assert len(self.reactions) == 268 and len(self.systems) == 90
        assert len(self.splits["train_relchem"]["ids"]) == 251
        assert len(self.splits["train_ae17"]["ids"]) == 17
        self.handles = Handles(self.root, self.checksums)

    def validation(self, *_args, **_kwargs):
        raise PermissionError(
            "F028 training-only loader; gated evaluation uses a separate loader"
        )


def initial_template():
    receipt = read(file("initialization"))["artifact"]
    path = Path(receipt["path"])
    assert b8.sha256_file(path) == receipt["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["seed"] == 2028
    return payload["template"]


def allocate(arm):
    model = copy.deepcopy(initial_template()).cuda()
    eps_helpers.configure_epsilon(model, ARMS[arm])
    shadow = copy.deepcopy(model).double()
    shadow.register_load_state_dict_post_hook(
        lambda module, _keys: eps_helpers.assert_epsilon(module, ARMS[arm])
    )
    synchronize(model, shadow, arm)
    optimizer = torch.optim.AdamW(
        core.named_trainable_parameters(model).values(),
        lr=2.5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    assert not optimizer.state
    return model, shadow, optimizer


def atomic_save(path, payload):
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def save(path, arm, model, optimizer, cursor, logs):
    payload = {
        "experiment": "F028",
        "arm": arm,
        "seed": 2028,
        "identity": scientific_identity(model, arm),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cursor": cursor,
        "logs": logs,
        "RNG": pilot.rng(),
        "protocol_sha256": b8.sha256_file(file("protocol")),
        "source_verification_sha256": b8.sha256_file(file("source_verification")),
        "manifest_sha256": f010.continuation.MANIFEST_SHA,
        "coefficients": {**b8.LAMBDAS, "c010": C010},
    }
    atomic_save(path, payload)
    return {"path": str(path), "sha256": b8.sha256_file(path)}


def restore(path, arm):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["experiment"] == "F028" and payload["arm"] == arm
    assert payload["protocol_sha256"] == b8.sha256_file(file("protocol"))
    assert payload["source_verification_sha256"] == b8.sha256_file(
        file("source_verification")
    )
    assert payload["manifest_sha256"] == f010.continuation.MANIFEST_SHA
    assert payload["coefficients"] == {**b8.LAMBDAS, "c010": C010}
    assert "epsilon" in payload["identity"]
    model, shadow, optimizer = allocate(arm)
    model.load_state_dict(payload["model"], strict=True)
    eps_helpers.assert_epsilon(model, ARMS[arm])
    assert scientific_identity(model, arm) == payload["identity"]
    synchronize(model, shadow, arm)
    optimizer.load_state_dict(payload["optimizer"])
    if payload["cursor"]:
        assert {int(s["step"]) for s in optimizer.state.values()} == {payload["cursor"]}
    else:
        assert not optimizer.state
    assert payload["RNG"][2].device.type == "cpu"
    pilot.restore_rng(payload["RNG"])
    return model, shadow, optimizer, payload


def freeze():
    assert not file("protocol").exists(), "Never silently overwrite a frozen experiment"
    OUT.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ANCESTOR, "HEAD"], cwd=ROOT, check=True
    )
    rows = b8.load_j251_rows()
    batches = b8.build_batches(rows)
    properties = b8.assert_batch_properties(batches, rows)
    manifest_path = f010.continuation.PARENT_MANIFEST
    assert read(manifest_path) == batches
    assert b8.sha256_file(manifest_path) == f010.continuation.MANIFEST_SHA
    assert (
        b8.sha256_file(ROOT / "relchem_joint_epoch_evaluation_manifest.json")
        == b8.EVAL_MANIFEST_SHA
    )
    scales = read(ROOT / "lap_f010_normalization_receipt.json")
    assert scales["c"] == C010
    dataset = read(DATA / "dataset_manifest.json")
    assert dataset["logical_sha256"] == b8.DATA_SHA and dataset["status"] == "qualified"
    sources = [
        "tools/lap_b8_adamw.py",
        "tools/lap_f010_uniform_normalized_chemistry.py",
        "tools/lap_training_metric_gradient_audit.py",
        "tools/lap_exchange_eps_v1.py",
        "tools/lap_f027_exchange_eps.py",
        "tools/lap_fresh_grid_pilot_v1.py",
        "train_models/NN_models.py",
        "train_models/NN_models_lap.py",
        "train_models/lap_operator.py",
        "train_models/lap_vxc.py",
        "train_models/lap_moo_training.py",
        "train_models/lap_training.py",
        "train_models/lap_fixed_adamw.py",
        "train_models/optuna_joint.py",
        "train_models/publication_data/loader.py",
        "train_models/reaction_energy_calculation.py",
        "dft_functionals/PBE.py",
        "dft_functionals/constants.py",
    ]
    hashes = {s: b8.sha256_file(ROOT / s) for s in sources}
    precision = {
        "native": "F32 parameters and AdamW moments",
        "chemistry_AE17": "Unchanged F010 qualified F64 shadow, source widening, reaction integration/PBE arithmetic; not recast to native F32",
        "Exc": "Unchanged original F32 energy/learned forward and F32-leaf derivatives widened to F64 (historical task-specific precision boundary)",
        "operator": "Original F32 corpus AO/features widened exactly as production; F64 learned partials, original F32 PBE, F64 matrix/two-pass/shadow gradients",
        "sum": "Four task gradients combined F64; exactly one final per-coordinate cast to native F32",
        "point_chunks": {"operator": 2048, "Exc": 4096, "large_reaction": 16384},
    }
    source_verification = {
        "dataset_manifest_sha256": b8.sha256_file(DATA / "dataset_manifest.json"),
        "logical_sha256": b8.DATA_SHA,
        "manifest_sha256": f010.continuation.MANIFEST_SHA,
        "metadata": {
            s: b8.sha256_file(DATA / s)
            for s in (
                "splits.json",
                "chemistry/reactions.jsonl",
                "chemistry/species.jsonl",
                "mrks/systems.jsonl",
                "chemistry/dispersion.json",
                "chemistry/dispersion_metadata.json",
                "mrks/dispersion.json",
            )
        },
        "expected_training_file_hashes": {
            r["file"]: r["sha256"]
            for r in dataset["files"]
            if not r["file"].startswith("validation/")
        },
        "properties": properties,
        "policy": "Metadata authenticated now; every consumed H5 hash checked before first read by Handles; no development tensors",
    }
    for relative, digest in source_verification["metadata"].items():
        expected = next(r["sha256"] for r in dataset["files"] if r["file"] == relative)
        assert expected == digest, relative
    write("source_verification", source_verification)
    write(
        "protocol",
        {
            "experiment": "F028",
            "seed": 2028,
            "ancestor": ANCESTOR,
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "start_unix_ms": START_MS,
            "wall_origin": "Conservative F027 last remote verification, preceding all interrupted F028 read-only actions",
            "wall_limit": LIMIT,
            "constructor": CONFIG,
            "arms": ARMS,
            "source_hashes": hashes,
            "precision": precision,
            "precision_hash": b8.sha256_text(json.dumps(precision, sort_keys=True)),
            "coefficients": {**b8.LAMBDAS, "c010": C010},
            "baseline": b8.BASELINE,
            "order": "CONTROL then INTERVENTION at each cursor 0..250",
            "updates": 251,
            "milestones": MILESTONES,
            "evaluation": "Full E251/E17/full90 t0 and t251; lightweight constraints/checkpoints intermediate; no selection",
            "startup_diagnostics": "Actual presentations at theta0/theta1/theta2; endpoint uses last training presentation250, explicitly not full population gradients; first2 finite changes on same pre-step presentation",
            "effect_floor": 0.01,
            "optional_swaps": "Only frozen t251 full90 if budget permits",
            "clean28": "Only scientifically eligible predetermined t251, after both frozen and all ratios computed; exactly once each; no future/test",
            "tolerances": {
                "sum_atol": 1e-10,
                "sum_rtol": 1e-9,
                "physical_relative": float(16 * np.finfo(np.float32).eps),
                "energy_kcal_absolute": 1e-6,
                "constraints": 1e-12,
            },
            "resource_margin": 1.35,
            "final_reserve_seconds": 5000,
            "optimizer": {
                "lr": 2.5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "foreach": False,
            },
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
            },
        },
    )
    b8.write_json(OUT / "training_manifest.json", batches)
    write(
        "provenance",
        {
            "source_hashes": hashes,
            "training_manifest": {
                "path": str(OUT / "training_manifest.json"),
                "sha256": b8.sha256_file(OUT / "training_manifest.json"),
            },
            "pretrained_weights_loaded": 0,
            "inherited_moments": 0,
        },
    )


def initialize():
    assert not file("initialization").exists()
    deterministic()
    random.seed(2028)
    np.random.seed(2028)
    torch.manual_seed(2028)
    torch.cuda.manual_seed_all(2028)
    model = pilot.pcPBELMLOptimizerV2Lap(**CONFIG).float()
    assert (
        sum(p.numel() for p in core.named_trainable_parameters(model).values()) == 9446
    )
    path = OUT / "seed2028_initialization.pt"
    atomic_save(path, {"template": model, "seed": 2028, "RNG": pilot.rng()})
    write(
        "initialization",
        {
            "seed": 2028,
            "scientific_initializations": 1,
            "tensor_hash": pilot.state_hash(model),
            "artifact": {"path": str(path), "sha256": b8.sha256_file(path)},
        },
    )
    models = [allocate(arm) for arm in ARMS]
    a, b = models[0][0], models[1][0]
    assert pilot.state_hash(a) == pilot.state_hash(b) == pilot.state_hash(model)
    assert all(
        x.data_ptr() != y.data_ptr()
        for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
    manifests = {}
    for arm, (native, shadow, optimizer) in zip(ARMS, models, strict=True):
        assert not optimizer.state
        manifests[arm] = scientific_identity(native, arm)
        assert pilot.constraints(native)["PASS"]
        save(OUT / f"{arm}_t0.pt", arm, native, optimizer, 0, [])
        eps_helpers.assert_epsilon(shadow, ARMS[arm])
    write("epsilon_manifests", manifests)


def flatten(mapping):
    return torch.cat([v.detach().double().flatten() for v in mapping.values()])


def unflatten(vector, model):
    result, start = {}, 0
    for n, p in core.named_trainable_parameters(model).items():
        result[n] = vector[start : start + p.numel()].reshape(p.shape)
        start += p.numel()
    assert start == 9446
    return result


def operator_value_gradient(shadow, bundle, identity, need_grad=True):
    """Qualified F026/F027 two-pass adjoint on ORIGINAL corpus rows/precision."""
    data = bundle.mrks()
    record = data[data.positions[identity]]
    ref, overlap = record["RefAO"].cuda(), record["Overlap"].cuda()

    def matrices(graph):
        for rows, factors in data.ao_chunks(identity, 2048):
            # Match original operator_system(dtype=F32), then its exact promotion.
            arrays = {
                "features": record["DensityDescriptorsN10"][rows].float(),
                "weights": record["weights"][rows].float(),
                "phi": factors["phi"].float(),
                "grad": factors["grad_phi"].float(),
                "lap": factors["lap_phi"].float(),
            }
            yield pilot.chunk_matrix(shadow, arrays, "cuda", graph)

    V = torch.zeros_like(ref)
    for matrix in matrices(False):
        V += matrix.detach()
    leaf = V.detach().requires_grad_(True)
    loss = pilot.op.operator_loss(leaf, ref, overlap)
    upstream = torch.autograd.grad(loss, leaf)[0].detach()
    params = core.named_trainable_parameters(shadow)
    gradients = {n: torch.zeros_like(p) for n, p in params.items()}
    if need_grad:
        for matrix in matrices(True):
            g = torch.autograd.grad(
                (matrix * upstream).sum(), tuple(params.values()), allow_unused=True
            )
            for total, value in zip(gradients.values(), g, strict=True):
                if value is not None:
                    total.add_(value.detach())
    assert torch.isfinite(loss) and all(
        torch.isfinite(g).all() for g in gradients.values()
    )
    return float(loss), gradients


def exc_value_gradient(model, bundle, identity, need_grad=True):
    record = bundle.mrks()[bundle.mrks().positions[identity]]
    features, weights = (
        record["DensityDescriptorsN10"].cuda().float(),
        record["weights"].cuda().float(),
    )
    with torch.set_grad_enabled(need_grad):
        prediction = integrated_energy(
            LapEnergy(model), features[:, None, :], weights, 4096
        )
        prediction = _add_mrks_dispersion_once(
            prediction, record["name"], read(DATA / "mrks/dispersion.json"), True
        )
        target = record["Exc"].to(prediction).reshape(1)
        value = batch_exc([record["name"]], prediction.reshape(1), target)
        params = core.named_trainable_parameters(model)
        g = (
            torch.autograd.grad(value, tuple(params.values()), allow_unused=True)
            if need_grad
            else [None] * len(params)
        )
    gradients = {
        n: torch.zeros_like(p, dtype=torch.float64)
        if v is None
        else v.detach().double()
        for (n, p), v in zip(params.items(), g, strict=True)
    }
    assert torch.isfinite(value) and all(
        torch.isfinite(v).all() for v in gradients.values()
    )
    return (
        float(value),
        gradients,
        {
            "signed_error_Ha": float(prediction.detach() - target.reshape(())),
            "prediction_Ha": float(prediction.detach()),
            "reference_Ha": float(target[0]),
        },
    )


def joint(model, shadow, arm, bundle, entry):
    budget(5000)
    synchronize(model, shadow, arm)
    start = time.monotonic()
    chem, gc, infos, _seconds = f010.f010_mean(
        model,
        shadow,
        bundle,
        entry["relchem"],
        bundle.chemistry_dispersions(),
        read(ROOT / "lap_f010_normalization_receipt.json")["scales"],
        C010,
    )
    reaction = b8._load_reaction(bundle, "ae17", entry["ae17"])
    ae, ga = b8._singleton(model, shadow, reaction, bundle.chemistry_dispersions())
    del reaction
    exc, ge, _ = exc_value_gradient(model, bundle, entry["mrks_id"])
    op, go = operator_value_gradient(shadow, bundle, entry["mrks_id"])
    raw = {"relchem": gc, "ae17": ga, "exc": ge, "op": go}
    params = core.named_trainable_parameters(model)
    assert all(tuple(g) == tuple(params) for g in raw.values())
    assert all(x.dtype == torch.float64 for g in raw.values() for x in g.values())
    combined = weighted_gradient(raw, b8.LAMBDAS)
    independent = sum(b8.LAMBDAS[t] * flatten(raw[t]) for t in TASKS)
    torch.testing.assert_close(flatten(combined), independent, atol=1e-10, rtol=1e-9)
    losses = {"relchem": chem, "ae17": ae, "exc": exc, "op": op}
    assert all(math.isfinite(v) for v in losses.values())
    assert all(torch.isfinite(g).all() for g in combined.values())
    return losses, raw, combined, infos, time.monotonic() - start


def scalar_tasks(model, shadow, arm, bundle, entry):
    synchronize(model, shadow, arm)
    scales = read(ROOT / "lap_f010_normalization_receipt.json")["scales"]
    dispersion = bundle.chemistry_dispersions()
    values = []
    with torch.no_grad():
        for spec in entry["relchem"]:
            reaction = b8._load_reaction(bundle, "relchem", spec)
            pred, ref = chemistry._prediction(shadow, reaction, dispersion)
            values.append(
                float(
                    C010
                    * torch.sqrt((pred - ref).square() + 1e-20)
                    / scales[spec["database"]]
                )
            )
        reaction = b8._load_reaction(bundle, "ae17", entry["ae17"])
        pred, ref = chemistry._prediction(shadow, reaction, dispersion)
        ae = float(
            chemistry._factor("AE17") * torch.sqrt((pred - ref).square() + 1e-20)
        )
    exc, _, _ = exc_value_gradient(model, bundle, entry["mrks_id"], False)
    op, _ = operator_value_gradient(shadow, bundle, entry["mrks_id"], False)
    return {"relchem": float(np.mean(values)), "ae17": ae, "exc": exc, "op": op}


def geometry(arm, theta, entry, model, optimizer, raw, combined, losses, delta=None):
    g = flatten(combined)
    weighted = {t: flatten(raw[t]) * b8.LAMBDAS[t] for t in TASKS}
    ln_names = set()
    exchange = set(eps_helpers.EXCHANGE)
    all_ln, exchange_ln, start = [], [], 0
    for parent, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            ln_names.update(
                parent + "." + n for n, _ in module.named_parameters(recurse=False)
            )
    for name, p in core.named_trainable_parameters(model).items():
        ix = list(range(start, start + p.numel()))
        if name in ln_names:
            all_ln += ix
            if name.rsplit(".", 1)[0] in exchange:
                exchange_ln += ix
        start += p.numel()
    non_ln = sorted(set(range(9446)) - set(all_ln))

    def norm(v):
        return float(torch.linalg.vector_norm(v))

    def moment(k):
        return (
            norm(torch.cat([s[k].flatten().double() for s in optimizer.state.values()]))
            if optimizer.state
            else 0.0
        )

    a, b = flatten(raw["relchem"]), flatten(raw["op"])
    cos = (
        float(a @ b / (a.norm() * b.norm())) if a.norm() > 0 and b.norm() > 0 else None
    )
    row = {
        "arm": arm,
        "theta": theta,
        "presentation_cursor": entry["cursor"],
        "population": "one declared original B8 training minibatch, NOT E251/full90",
        "weighted_total_gradient_norm": norm(g),
        "exchange_LN_gradient_norm": norm(g[exchange_ln]),
        "non_LayerNorm_gradient_norm": norm(g[non_ln]),
        "chemistry_operator_cosine": cos,
        "cancellation_ratio_norm_sum_over_sum_norms": norm(g)
        / sum(norm(v) for v in weighted.values()),
        "first_moment_norm": moment("exp_avg"),
        "second_moment_norm": moment("exp_avg_sq"),
    }
    for t in TASKS:
        row[t + "_loss"] = losses[t]
        row[t + "_weighted_gradient_norm"] = norm(weighted[t])
        row[t + "_raw_derivative_along_negative_joint"] = -float(flatten(raw[t]) @ g)
        if delta is not None:
            row[t + "_raw_derivative_along_actual_displacement"] = float(
                flatten(raw[t]) @ delta
            )
    if delta is not None:
        row["native_displacement_norm"] = norm(delta)
        row["weighted_gradient_dot_displacement"] = float(g @ delta)
    return row


def small_operator_parity(shadow, bundle, identity):
    dataset = bundle.mrks()
    group = dataset.group(bundle.systems[identity])
    indices = np.linspace(0, len(group["weights"]) - 1, 7, dtype=int)
    arrays = {
        k: np.asarray(group[field][indices], dtype=np.float32)
        for k, field in (
            ("features", "DensityDescriptorsN10"),
            ("weights", "weights"),
            ("phi", "phi"),
            ("grad", "grad_phi"),
            ("lap", "lap_phi"),
        )
    }
    params = core.named_trainable_parameters(shadow)
    V = pilot.chunk_matrix(shadow, arrays, "cuda", True)
    reference = V.detach() * 0.91
    overlap = torch.eye(len(V), dtype=torch.float64, device="cuda")
    loss = pilot.op.operator_loss(V, reference, overlap)
    direct = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
    leaf = V.detach().requires_grad_(True)
    upstream = torch.autograd.grad(
        pilot.op.operator_loss(leaf, reference, overlap), leaf
    )[0].detach()
    accumulated = [torch.zeros_like(p) for p in params.values()]
    for sl in (slice(0, 3), slice(3, 7)):
        M = pilot.chunk_matrix(
            shadow, {k: v[sl] for k, v in arrays.items()}, "cuda", True
        )
        pieces = torch.autograd.grad(
            (M * upstream).sum(), tuple(params.values()), allow_unused=True
        )
        for total, piece in zip(accumulated, pieces, strict=True):
            if piece is not None:
                total.add_(piece.detach())
    expected = torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten()
            for p, g in zip(params.values(), direct, strict=True)
        ]
    )
    actual = torch.cat([g.flatten() for g in accumulated])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    # Independent energy-first dE/dP path, already qualified by F026/F027.
    import lap_fresh_layernorm_v1 as derivative_helpers

    independent_V = derivative_helpers.energy_first_chunk(shadow, arrays, True)
    torch.testing.assert_close(independent_V, V.detach(), atol=1e-6, rtol=1e-5)
    return {
        "rows": indices.tolist(),
        "direct_vs_adjoint_gradient_gap": float((actual - expected).norm()),
        "energy_first_matrix_gap": float((independent_V - V.detach()).norm()),
        "PASS": True,
    }


def preflight():
    assert not file("preflight").exists()
    deterministic()
    bundle = TrainingBundle()
    entry = read(OUT / "training_manifest.json")[0]
    results, constraints, profiles = {}, {}, {}
    for arm, epsilon in ARMS.items():
        model, shadow, optimizer, _saved = restore(OUT / f"{arm}_t0.pt", arm)
        rng = pilot.rng()
        before = scientific_identity(model, arm)
        torch.cuda.reset_peak_memory_stats()
        tic = time.monotonic()
        losses, raw, combined, infos, seconds = joint(model, shadow, arm, bundle, entry)
        assert not any(p.grad is not None for p in model.parameters())
        # Independent original F010 scalar, with no legacy database multiplier.
        expected = np.mean(
            [
                i["unscaled_loss"]
                / chemistry._factor(i["database"])
                * C010
                / read(ROOT / "lap_f010_normalization_receipt.json")["scales"][
                    i["database"]
                ]
                for i in infos
            ]
        )
        np.testing.assert_allclose(expected, losses["relchem"], atol=1e-10, rtol=1e-9)
        parity = small_operator_parity(shadow, bundle, entry["mrks_id"])
        constraints[arm] = pilot.constraints(model)
        assert constraints[arm]["PASS"]
        np.savez(
            OUT / f"{arm}_initial_gradients.npz",
            **{t: flatten(raw[t]).cpu().numpy() for t in TASKS},
        )
        diag = geometry(arm, 0, entry, model, optimizer, raw, combined, losses)
        # Authorized discarded profiling step; restore entire original state afterwards.
        adamw_step(model, optimizer, raw, b8.LAMBDAS)
        assert {int(s["step"]) for s in optimizer.state.values()} == {1}
        restored, rs, ro, _payload = restore(OUT / f"{arm}_t0.pt", arm)
        assert scientific_identity(restored, arm) == before and not ro.state
        assert all(
            torch.equal(a, b)
            for a, b in zip(model.buffers(), restored.buffers(), strict=True)
        )
        pilot.restore_rng(rng)
        eps_helpers.assert_epsilon(rs, epsilon)
        profiles[arm] = seconds
        results[arm] = {
            "losses": losses,
            "geometry": diag,
            "parity": parity,
            "complete_gradient_seconds": seconds,
            "profile_total_seconds": time.monotonic() - tic,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "discarded_profile_optimizer_steps": 1,
            "initial_state_optimizer_RNG_restored": True,
        }
        print("F028_PREFLIGHT", arm, results[arm], flush=True)
        del model, shadow, optimizer, raw, combined, restored, rs, ro
        torch.cuda.empty_cache()
    write(
        "preflight",
        {
            "technical": "PASS",
            "results": results,
            "profiles": profiles,
            "retained_steps": 0,
            "discarded_profiling_native_steps": 2,
            "resource_training_authorization": "PENDING complete t0 evaluation cost measurements",
        },
    )
    write("constraint_checks", constraints)
    table("gradient_geometry", [results[a]["geometry"] for a in ARMS])
    bundle.close()


def evaluation(arm, cursor):
    """All original scalar populations; persistent rows are checkpoint-hash bound."""
    deterministic()
    start = time.monotonic()
    path = OUT / f"{arm}_t{cursor}.pt"
    model, shadow, _optimizer, _saved = restore(path, arm)
    folder = OUT / f"{arm}_evaluation_t{cursor}"
    folder.mkdir(exist_ok=True)
    contract = {
        "checkpoint_sha256": b8.sha256_file(path),
        "protocol_sha256": b8.sha256_file(file("protocol")),
        "arm": arm,
        "cursor": cursor,
    }
    if (folder / "contract.json").exists():
        assert read(folder / "contract.json") == contract
    else:
        b8.write_json(folder / "contract.json", contract)
    before = scientific_identity(model, arm)
    rng = pilot.rng()
    bundle = TrainingBundle()
    dispersion = bundle.chemistry_dispersions()
    scales = read(ROOT / "lap_f010_normalization_receipt.json")["scales"]
    specs = chemistry._selected(bundle)
    rows = (
        read(folder / "reactions.json") if (folder / "reactions.json").exists() else []
    )
    assert all(
        (r["identity"], r["variant"]) == (s["identity"], s["variant"])
        for r, s in zip(rows, specs)
    )
    for spec in specs[len(rows) :]:
        budget(1800)
        reaction = b8._load_reaction(bundle, "relchem", spec)
        with torch.no_grad():
            pred, ref = chemistry._prediction(shadow, reaction, dispersion)
        residual = float(pred - ref)
        assert math.isfinite(residual)
        rows.append(
            {
                **spec,
                "prediction": float(pred),
                "reference": float(ref),
                "signed_error_kcal": residual,
                "JW_penalty": math.sqrt(residual**2 + 1e-20) / scales[spec["database"]],
                "legacy_penalty": chemistry._factor(spec["database"])
                * math.sqrt(residual**2 + 1e-20),
            }
        )
        b8.write_json(folder / "reactions.json", rows)
        del reaction
        if len(rows) % 25 == 0:
            print("F028_EVAL_CHEMISTRY", arm, cursor, len(rows), flush=True)
    atom_specs = [
        {**s, "database": "AE17"}
        for s in read(ROOT / "relchem_joint_epoch_evaluation_manifest.json")["rows"]
        if s["task"] == "ae17"
    ]
    atoms = read(folder / "atoms.json") if (folder / "atoms.json").exists() else []
    for spec in atom_specs[len(atoms) :]:
        budget(1800)
        reaction = b8._load_reaction(bundle, "ae17", spec)
        with torch.no_grad():
            pred, ref = chemistry._prediction(shadow, reaction, dispersion)
        residual = float(pred - ref)
        atoms.append(
            {
                **spec,
                "species": bundle.reactions[spec["identity"]]["components"][0],
                "prediction_kcal": float(pred),
                "reference_kcal": float(ref),
                "signed_error_kcal": residual,
                "absolute_error_kcal": abs(residual),
                "loss": chemistry._factor("AE17") * math.sqrt(residual**2 + 1e-20),
            }
        )
        b8.write_json(folder / "atoms.json", atoms)
        del reaction
    physical = (
        read(folder / "physical.json") if (folder / "physical.json").exists() else {}
    )
    for identity in sorted(bundle.systems):
        if identity in physical:
            continue
        budget(1800)
        tic = time.monotonic()
        exc, _, signed = exc_value_gradient(model, bundle, identity, False)
        op, _ = operator_value_gradient(shadow, bundle, identity, False)
        physical[identity] = {
            "exc": exc,
            "op": op,
            **signed,
            "name": bundle.systems[identity]["source_id"],
            "seconds": time.monotonic() - tic,
        }
        b8.write_json(folder / "physical.json", physical)
        print("F028_EVAL_PHYSICAL", arm, cursor, len(physical), 90, flush=True)
    assert len(rows) == 251 and len(atoms) == 17 and len(physical) == 90
    errors = np.array([r["signed_error_kcal"] for r in rows])
    database_rows = []
    for database in sorted({r["database"] for r in rows}):
        e = np.array(
            [r["signed_error_kcal"] for r in rows if r["database"] == database]
        )
        database_rows.append(
            {
                "arm": arm,
                "cursor": cursor,
                "database": database,
                "count": len(e),
                "MAE": float(np.abs(e).mean()),
                "RMSE": float(np.sqrt(np.mean(e**2))),
            }
        )
    metrics = {
        "arm": arm,
        "cursor": cursor,
        "JW": float(np.mean([r["JW_penalty"] for r in rows])),
        "legacy": float(np.mean([r["legacy_penalty"] for r in rows])),
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "macro_database_MAE": float(np.mean([r["MAE"] for r in database_rows])),
        "ae17": float(np.mean([r["loss"] for r in atoms])),
        "exc": float(np.mean([r["exc"] for r in physical.values()])),
        "op": float(np.mean([r["op"] for r in physical.values()])),
        "operator_median": float(np.median([r["op"] for r in physical.values()])),
        "operator_max": float(max(r["op"] for r in physical.values())),
        "seconds": time.monotonic() - start,
        "constraints": pilot.constraints(model),
    }
    metrics["chemistry_training_scalar"] = C010 * metrics["JW"]
    metrics["ratios"] = {
        t: metrics["legacy" if t == "relchem" else t] / b8.BASELINE[t] for t in TASKS
    }
    metrics["eligible"] = all(
        r < 1 and math.isfinite(r) for r in metrics["ratios"].values()
    )
    assert metrics["constraints"]["PASS"] and scientific_identity(model, arm) == before
    metrics["hydrogen"] = next(r for r in atoms if r["species"] == "H_ae17")
    pilot.restore_rng(rng)
    b8.write_json(folder / "metrics.json", metrics)
    b8.write_json(folder / "database_metrics.json", database_rows)
    b8.write_json(
        folder / "receipt.json",
        {
            **contract,
            "complete": True,
            "seconds": metrics["seconds"],
            "model_and_RNG_restored": True,
            "verified_training_data_files": {
                name: bundle.checksums[name] for name in bundle.handles.verified
            },
            "hashes": {
                p.name: b8.sha256_file(p)
                for p in folder.iterdir()
                if p.is_file() and p.name != "receipt.json"
            },
        },
    )
    bundle.close()
    print("F028_EVALUATED", arm, cursor, metrics, flush=True)
    return metrics


def baseline():
    assert read(file("preflight"))["technical"] == "PASS"
    for arm in ARMS:
        if not (OUT / f"{arm}_evaluation_t0/metrics.json").exists():
            evaluation(arm, 0)
    measurements = {
        a: read(OUT / f"{a}_evaluation_t0/metrics.json")["seconds"] for a in ARMS
    }
    profiles = read(file("preflight"))["profiles"]
    projected_training = 251 * sum(profiles.values()) * 1.35
    final_evaluation = sum(measurements.values()) * 1.35
    reserve = 5000
    projected = projected_training + final_evaluation + reserve
    fits = elapsed() + projected < LIMIT
    write(
        "resource_report",
        {
            "limit": LIMIT,
            "elapsed_before_training": elapsed(),
            "complete_gradient_profiles": profiles,
            "full_population_t0_evaluation_seconds": measurements,
            "training_projection_with_margin": projected_training,
            "endpoint_evaluation_reserve": final_evaluation,
            "integrity_startup_and_conditional_clean28_reserve": reserve,
            "projected_remaining": projected,
            "training_authorized": fits,
            "active_computation": "Per-call execution wall measured; GPU kernel-active time not separately metered",
        },
    )
    assert fits, (
        "Complete matched pair does not fit the frozen budget; no retained training"
    )
    pre = read(file("preflight"))
    pre["resource_training_authorization"] = "PASS"
    write("preflight", pre)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "freeze",
            "initialize",
            "preflight",
            "baseline",
            "train",
            "endpoint",
            "finish",
        ],
    )
    phase = parser.parse_args().phase
    tic = time.monotonic()
    OUT.mkdir(exist_ok=True)
    b8.write_json(
        OUT / (phase + "_execution.json"),
        {
            "phase": phase,
            "started_unix": time.time(),
            "driver_sha256": b8.sha256_file(Path(__file__)),
            "execution_driver_sha256": b8.sha256_file(
                ROOT / "tools/lap_f028_joint_eps_run.py"
            )
            if (ROOT / "tools/lap_f028_joint_eps_run.py").exists()
            else None,
        },
    )
    if phase in ("train", "endpoint", "finish"):
        import lap_f028_joint_eps_run as execution

        getattr(execution, phase)()
    else:
        globals()[phase]()
    write(
        phase + "_timing",
        {"seconds": time.monotonic() - tic, "workflow_wall": elapsed()},
    )
