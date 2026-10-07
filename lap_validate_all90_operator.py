"""Read-only all-90 cache integrity and historical-15 regression gate."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "train_models"))
from lap_moo_training import make_mrks_objective_factories
from lap_operator import (
    LapEnergy,
    _inverse_sqrt_overlap,
    assemble_rks_operator,
    operator_loss,
)
from lap_operator_data import _array_digest, load_central_operator_record
from train_lap_moo import CentralAOCache
from train_lap_s5 import _pilot_model


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_arrays(central, cache, oracle):
    manifest = json.loads((cache / "manifest.json").read_text())
    old = json.loads((oracle / "manifest.json").read_text())
    old_rows = {r["system_name"]: r for r in old["records"]}
    if len(manifest["records"]) != 90:
        raise ValueError("All 90 systems required")
    rows = []
    for item in manifest["records"]:
        name = item["system_name"]
        source = central / f"{name}.h5"
        central_manifest = json.loads((central / "manifest.json").read_text())
        source = central / next(r["file"] for r in central_manifest["records"] if r["system_name"] == name)
        if sha(source) != item["source_file_sha256"] or sha(cache / item["file"]) != item["cache_file_sha256"]:
            raise ValueError(f"{name}: source/cache SHA mismatch")
        record = load_central_operator_record(source)
        _inverse_sqrt_overlap(torch.from_numpy(record.Overlap.copy()))
        n, nao = item["point_count"], item["nao"]
        if len(record.weights) != n or record.Overlap.shape != (nao, nao):
            raise ValueError(f"{name}: dimensions")
        with h5py.File(cache / item["file"], "r") as handle:
            expected = {"phi": (n, nao), "grad_phi": (n, 3, nao), "lap_phi": (n, nao)}
            for key, shape in expected.items():
                ds = handle[key]
                if ds.shape != shape or ds.dtype != np.dtype("float32"):
                    raise ValueError(f"{name}: {key} shape/dtype")
                for start in range(0, n, 2048):
                    block = ds[start:start + 2048]
                    if not np.isfinite(block).all():
                        raise ValueError(f"{name}: {key} nonfinite")
            if name in old_rows:
                previous = old_rows[name]
                if previous["source_file_sha256"] != item["source_file_sha256"]:
                    raise ValueError(f"{name}: historical target identity mismatch")
                if sha(oracle / previous["file"]) != previous["cache_file_sha256"]:
                    raise ValueError(f"{name}: historical cache SHA mismatch")
                with h5py.File(oracle / previous["file"], "r") as before:
                    for key in expected:
                        for start in range(0, n, 2048):
                            if not np.array_equal(handle[key][start:start + 2048], before[key][start:start + 2048]):
                                raise ValueError(f"{name}: historical {key} parity failed")
        rows.append({"system": name, "n_grid": n, "n_ao": nao,
                     "source_sha256": item["source_file_sha256"],
                     "overlap_sha256": _array_digest(record.Overlap),
                     "reference_operator_sha256": _array_digest(record.RefAO),
                     "ao_cache_sha256": item["cache_file_sha256"],
                     "historical_array_parity": "PASS" if name in old_rows else "not_applicable"})
        print(json.dumps({"integrity": name}), flush=True)
    return rows, old_rows


def evaluate(model, system, gradient=False):
    if gradient:
        factory = make_mrks_objective_factories(model, system, point_chunk_size=256, dispersions=None)[1]
        loss = factory()
        grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        vector = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                            for p, g in zip(model.parameters(), grads)]).detach().cpu()
        return float(loss.detach()), vector
    prediction = system.reference_operator.new_zeros(system.reference_operator.shape)
    for chunk in system.ao_chunks:
        prediction += assemble_rks_operator(
            LapEnergy(model), system.features[chunk.rows], system.weights[chunk.rows],
            chunk.phi, chunk.grad_phi, chunk.lap_phi, chunk_size=256, create_graph=False).detach()
    return float(operator_loss(prediction, system.reference_operator, system.overlap)), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("central", "cache", "oracle", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    state_sha = sha(args.state)
    if state_sha != "264a8d189d53d49007d7de11d8cfb5382e8b8262e4b48068142c8771e293c834":
        raise ValueError("Frozen seed11 P67 state provenance mismatch")
    torch.set_num_threads(1)
    rows, historical = validate_arrays(args.central, args.cache, args.oracle)
    torch.manual_seed(41)
    model = _pilot_model(torch.device("cuda"), torch.float32)
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    caches = [CentralAOCache(args.central, path, device=torch.device("cuda"), dtype=torch.float32, chunk_size=4096)
              for path in (args.cache, args.oracle)]
    for row in rows:
        name = row["system"]
        loss, grad = evaluate(model, caches[0].load(name), name in historical)
        if not np.isfinite(loss) or (grad is not None and not torch.isfinite(grad).all()):
            raise ValueError(f"{name}: nonfinite operator loss/gradient")
        row["frozen_model_operator_loss"] = loss
        if name in historical:
            old_loss, old_grad = evaluate(model, caches[1].load(name), True)
            if old_loss != loss or not torch.equal(grad, old_grad):
                raise ValueError(f"{name}: exact operator loss/gradient parity failed")
            row["historical_loss_gradient_parity"] = "PASS"
            row["gradient_sha256"] = hashlib.sha256(grad.numpy().tobytes()).hexdigest()
        if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in model.state_dict().items()):
            raise ValueError("Model state mutation")
        args.output.write_text(json.dumps({"status": "RUNNING", "records": rows}, indent=2, allow_nan=False))
        print(json.dumps({"operator_validation": name, "loss": loss}), flush=True)
    # Recheck every immutable input after evaluation.
    checked = 0
    for path in (args.cache, args.oracle):
        for row in json.loads((path / "manifest.json").read_text())["records"]:
            if sha(path / row["file"]) != row["cache_file_sha256"]:
                raise ValueError("Cache mutation")
            if sha(args.central / next(r["file"] for r in json.loads((args.central / "manifest.json").read_text())["records"]
                                       if r["system_name"] == row["system_name"])) != row["source_file_sha256"]:
                raise ValueError("Source mutation")
            checked += 2
    if sha(args.state) != state_sha:
        raise ValueError("Frozen model file mutation")
    args.output.write_text(json.dumps({"status": "PASS", "operator_ready": 90,
        "historical_15_parity": "PASS", "records": rows, "immutable_hash_checks_after": checked,
        "model_state_file_sha256": state_sha, "state_mutation": "NONE",
        "validation_seconds": time.perf_counter() - started}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
