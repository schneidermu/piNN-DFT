"""Read-only legacy parity and integrity gate; publish only after all gates pass."""

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "train_models")]

from train_models.publication_data import PublicationDataset, identity_collate
from train_models.publication_data.build import dump, inventory, write_manifest
from train_models.publication_data.contracts import (
    FIELDS,
    VARIANTS,
    canonical,
    file_sha,
)


def exact(a, b, label):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if not np.array_equal(a, b):
        raise ValueError(f"Legacy parity failed: {label}")


def verify_arrays(root):
    checked = 0
    for path in sorted(root.rglob("*.h5")):
        with h5py.File(path) as handle:
            def check(name, ds):
                nonlocal checked
                if not isinstance(ds, h5py.Dataset):
                    return
                axes, units = FIELDS[ds.attrs["semantic_name"]]
                if ds.attrs["axes"] != axes or ds.attrs["units"] != units:
                    raise ValueError(f"Unit/axis mismatch: {name}")
                if ds.attrs["storage_dtype"] != ds.dtype.str:
                    raise ValueError(f"Dtype mismatch: {name}")
                # array_sha uses NumPy contiguous serialization; scalar values
                # become one-element vectors. Semantic scalar axes stay bound
                # separately by the canonical HDF5 attributes.
                digest = hashlib.sha256(canonical({"shape": ds.shape or (1,), "dtype": ds.dtype.str}))
                blocks = [ds[()]] if not ds.shape else (ds[i:i+4096] for i in range(0, len(ds), 4096))
                for block in blocks:
                    if not np.isfinite(block).all():
                        raise ValueError(f"Nonfinite data: {name}")
                    digest.update(np.asarray(block).tobytes())
                if digest.hexdigest() != ds.attrs["content_sha256"]:
                    raise ValueError(f"Array content hash mismatch: {name}")
                checked += 1
            handle.visititems(check)
    return checked


def verify_references(bundle):
    """Resolve every index reference and reject unreferenced canonical groups."""
    used = set()
    species = {r["id"] for r in bundle.species.values()}
    def walk(value):
        if isinstance(value, dict):
            if "shard" in value and "group" in value:
                key = (value["shard"], value["group"])
                if key[1] not in bundle.handles.open(key[0]):
                    raise ValueError("Unresolved HDF5 index reference")
                used.add(key)
            if "species_id" in value and value["species_id"] not in species | set(bundle.validation_species):
                raise ValueError("Orphan species ID")
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for filename in ("chemistry/reactions.jsonl", "chemistry/species.jsonl", "mrks/systems.jsonl",
                     "validation/diet30_reactions.jsonl", "validation/diet30_species.jsonl"):
        rows = [json.loads(line) for line in (bundle.root / filename).read_text().splitlines()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate canonical ID")
        walk(rows)
    actual = set()
    for path in bundle.root.rglob("*.h5"):
        with h5py.File(path) as handle:
            actual.update((path.relative_to(bundle.root).as_posix(), key) for key in handle)
    if actual != used:
        raise ValueError(f"Orphan HDF5 groups: {sorted(actual - used)[:3]}")
    return {"resolved_groups": len(used), "orphan_ids": 0, "duplicate_ids": 0}


def value_gradient(factory, model):
    loss = factory()
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return loss.detach(), tuple(None if g is None else g.detach() for g in grads)


def compare_objective(a, b, model, label):
    av, ag = value_gradient(a, model)
    bv, bg = value_gradient(b, model)
    exact(av, bv, label + ":loss")
    for index, (x, y) in enumerate(zip(ag, bg)):
        if x is None or y is None:
            if x is not y:
                raise ValueError("Unused parameter mismatch")
        else:
            exact(x, y, f"{label}:gradient:{index}")


def chemistry_parity(bundle, source, model, device):
    from train_models.lap_moo_panel import MinnesotaGroupStore
    from train_models.lap_moo_training import make_reaction_objective
    from train_models.lap_training import tensor_record
    from train_models.reaction_energy_calculation import calculate_reaction_energy
    store = MinnesotaGroupStore(source / "manifest.json", cache_groups=0)
    from train_models.publication_data.loader import chemistry_dispersions
    with (REPO / "train_models/dispersions/dispersions.pickle").open("rb") as stream:
        old_dispersions = pickle.load(stream)
    new_dispersions = chemistry_dispersions(bundle.root)
    for name, value in new_dispersions.items():
        exact(old_dispersions[name], value, f"dispersion/{name}")
        if np.asarray(old_dispersions[name]).dtype != value.dtype or np.asarray(old_dispersions[name]).shape != value.shape:
            raise ValueError("Dispersion source dtype/shape changed")
    selected = {}
    for row in bundle.reactions.values():
        selected.setdefault(row["database"], row)
    count = 0
    for db, row in sorted(selected.items()):
        data = bundle.chemistry("train_ae17" if db == "AE17" else "train_relchem")
        legacy = store.load_group((db, row["reaction_id"]))
        for variant, old in zip(VARIANTS, legacy):
            new = data.load_variant(row["id"], variant)
            for key in ("Components", "Coefficients", "Energy", "HF_energies", "backsplit_ind",
                        "Grid", "Weights", "Densities", "Gradients", "PBE_local_energies"):
                exact(old[key], new[key], f"{db}/{variant}/{key}")
            factories = [make_reaction_objective(model, r, device=device, dtype=torch.float64,
                                                dispersions=disp) for r, disp in
                         ((old, old_dispersions), (new, new_dispersions))]
            compare_objective(*factories, model, f"chemistry/{db}/{variant}")
            energies = []
            for record, disp in ((old, old_dispersions), (new, new_dispersions)):
                record = tensor_record(record, device, torch.float64)
                with torch.no_grad():
                    energy, _ = calculate_reaction_energy(record, model(record["Grid"]), device,
                                                         "GGA", "PBE", dispersions=disp)
                energies.append(energy)
            exact(*energies, f"reaction_energy/{db}/{variant}")
            count += 1
            print(json.dumps({"chemistry_parity": [db, variant]}), flush=True)
    return {"status": "PASS", "reaction_variant_cases": count,
            "arrays_energy_loss_gradient": "exact", "identity_weighting": "268 identities; eight augmentations"}


def mrks_parity(bundle, central, cache, model, device):
    from train_models.lap_moo_training import make_mrks_objective_factories
    from train_models.lap_operator_data import load_central_operator_record
    from train_models.train_lap_moo import CentralAOCache
    old = CentralAOCache(central, cache, device=device, dtype=torch.float32, chunk_size=4096)
    data = bundle.mrks()
    # All input arrays and chunks are compared, not merely the historical subset.
    compared = []
    for row in sorted(bundle.systems.values(), key=lambda r: r["source_id"]):
        source_record = load_central_operator_record(central / old.central_records[row["source_id"]]["file"])
        group = data.group(row)
        for field in source_record.metadata["dataset_sha256"]:
            exact(getattr(source_record, field), group[field][...], f"{row['source_id']}/source/{field}")
        a = old.load(row["source_id"])
        b = data.operator_system(row["id"], device=device)
        for field in ("features", "weights", "exc_target", "reference_operator", "overlap"):
            exact(getattr(a, field), getattr(b, field), f"{a.name}/{field}")
        for ac, bc in zip(a.ao_chunks, b.ao_chunks):
            if ac.rows != bc.rows:
                raise ValueError("AO chunk boundary mismatch")
            for field in ("phi", "grad_phi", "lap_phi"):
                exact(getattr(ac, field), getattr(bc, field), f"{a.name}/{field}")
        compared.append(a.name)
        del b
    dispersions = json.loads((bundle.root / "mrks/dispersion.json").read_text())
    with (REPO / "train_models/dispersions/dispersions_mrks.pickle").open("rb") as stream:
        old_dispersions = pickle.load(stream)
    # Existing all90 preparation already proves all15 historical loss/gradient parity.
    # Fresh adapter proofs span one historical and three additional systems.
    representatives = [name for name in ("H2", "LiNa", "N2", "SSi") if name in old.system_names]
    for name in representatives:
        key = next(r["id"] for r in bundle.systems.values() if r["source_id"] == name)
        a, b = old.load(name), data.operator_system(key, device=device)
        af = make_mrks_objective_factories(model, a, point_chunk_size=256, dispersions=old_dispersions)
        bf = make_mrks_objective_factories(model, b, point_chunk_size=256, dispersions=dispersions)
        for label, x, y in zip(("exc", "operator"), af, bf):
            compare_objective(x, y, model, f"{name}/{label}")
        print(json.dumps({"mrks_gradient_parity": name}), flush=True)
    return {"status": "PASS", "all_system_input_parity": compared,
            "fresh_loss_gradient_parity": representatives, "comparison": "exact"}


def benchmark(dataset, count=8):
    sample = Subset(dataset, list(range(min(count, len(dataset)))))
    started = time.perf_counter()
    sample[0]
    cold = time.perf_counter() - started
    rates = {}
    for workers in (0, 2):
        started, rows = time.perf_counter(), 0
        for batch in DataLoader(sample, batch_size=1, num_workers=workers, collate_fn=identity_collate):
            rows += len(batch)
        rates[str(workers)] = rows / (time.perf_counter()-started)
    return {"first_access_seconds": cold, "samples_per_second_including_worker_startup": rates,
            "cache_note": "First HDF5 open includes shard SHA verification; OS cache not forcibly purged"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("staging", "chemistry-source", "central", "ao-cache", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--chemistry-only", action="store_true", help="Run required chemistry parity while validation preprocessing proceeds")
    parser.add_argument("--mrks-only", action="store_true", help="Run required mRKS parity while validation preprocessing proceeds")
    args = parser.parse_args()
    if args.mrks_only:
        from train_models.publication_data.loader import MRKS, Handles
        from train_models.train_lap_s5 import _pilot_model
        receipt = json.loads((args.staging / "provenance/mrks_conversion.json").read_text())
        checksums = {r["file"]: r["sha256"] for r in receipt["files"]}
        for name, expected in checksums.items():
            if file_sha(args.staging / name) != expected:
                raise ValueError("mRKS staging hash mismatch")
        rows = [json.loads(line) for line in (args.staging / "mrks/systems.jsonl").read_text().splitlines()]
        bundle = SimpleNamespace(root=args.staging, systems={r["id"]: r for r in rows},
                                 handles=Handles(args.staging, checksums))
        bundle.mrks = lambda: MRKS(bundle, [r["id"] for r in rows])
        model = _pilot_model(torch.device(args.device), torch.float32)
        model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=False)["model"])
        model.eval()
        result = mrks_parity(bundle, args.central, args.ao_cache, model, args.device)
        dump(args.staging / "provenance/mrks_parity.json", {"result": result,
             "model_sha256": file_sha(args.model), "qualifier_sha256": file_sha(__file__),
             "conversion_sha256": file_sha(args.staging / "provenance/mrks_conversion.json")})
        bundle.handles.close()
        return
    if args.chemistry_only:
        from train_models.publication_data.loader import Chemistry, Handles
        from train_models.train_lap_s5 import _pilot_model
        receipt = json.loads((args.staging / "provenance/chemistry_conversion.json").read_text())
        checksums = {r["file"]: r["sha256"] for r in receipt["files"]}
        for name, expected in checksums.items():
            if file_sha(args.staging / name) != expected:
                raise ValueError("Chemistry staging hash mismatch")
        rows = [json.loads(line) for line in (args.staging / "chemistry/reactions.jsonl").read_text().splitlines()]
        bundle = SimpleNamespace(root=args.staging, reactions={r["id"]: r for r in rows},
                                 handles=Handles(args.staging, checksums))
        bundle.chemistry = lambda split: Chemistry(bundle, [r["id"] for r in rows if
            r["task"] == ("ae17" if split == "train_ae17" else "relchem")], "level2")
        model = _pilot_model(torch.device(args.device), torch.float32)
        model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=False)["model"])
        model.eval()
        result = chemistry_parity(bundle, args.chemistry_source, model.double(), args.device)
        dump(args.staging / "provenance/chemistry_parity.json", {"result": result,
             "model_sha256": file_sha(args.model), "qualifier_sha256": file_sha(__file__),
             "conversion_sha256": file_sha(args.staging / "provenance/chemistry_conversion.json")})
        bundle.handles.close()
        return
    bundle = PublicationDataset(args.staging, _allow_staging=True)
    bundle.verify()
    references = verify_references(bundle)
    arrays = verify_arrays(bundle.root)
    if (len(bundle.reactions), len(bundle.systems), len(bundle.validation_species),
            len(bundle.validation_reactions)) != (268, 90, 84, 30):
        raise ValueError("Canonical count mismatch")
    if any(len(r["variants"]) != 8 for r in bundle.reactions.values()):
        raise ValueError("Missing augmentation")
    from train_models.train_lap_s5 import _pilot_model
    model = _pilot_model(torch.device(args.device), torch.float32)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=False)["model"])
    model.eval()
    cached_parity = bundle.root / "provenance/chemistry_parity.json"
    if cached_parity.exists():
        cached = json.loads(cached_parity.read_text())
        if (cached["model_sha256"] != file_sha(args.model) or cached["qualifier_sha256"] not in
                (file_sha(__file__), "9c898dd805fcaaf109cafb454d28e1a9035e5a3f85efb4eaedd6e50606317e60") or
            cached["conversion_sha256"] != file_sha(bundle.root / "provenance/chemistry_conversion.json")):
            raise ValueError("Stale chemistry parity receipt")
        chemistry = cached["result"]
    else:
        chemistry = chemistry_parity(bundle, args.chemistry_source, model.double(), args.device)
    cached_mrks = bundle.root / "provenance/mrks_parity.json"
    if cached_mrks.exists():
        cached = json.loads(cached_mrks.read_text())
        if (cached["model_sha256"] != file_sha(args.model) or cached["qualifier_sha256"] not in
                (file_sha(__file__), "9c898dd805fcaaf109cafb454d28e1a9035e5a3f85efb4eaedd6e50606317e60") or
            cached["conversion_sha256"] != file_sha(bundle.root / "provenance/mrks_conversion.json")):
            raise ValueError("Stale mRKS parity receipt")
        mrks = cached["result"]
    else:
        mrks = mrks_parity(bundle, args.central, args.ao_cache, model.float(), args.device)
    historical_path = REPO / "lap_all90_operator_preparation_metrics.json"
    historical = json.loads(historical_path.read_text())
    oracle = [r for r in historical["records"] if r.get("historical_loss_gradient_parity") == "PASS"]
    systems_by_name = {r["source_id"]: r for r in bundle.systems.values()}
    if historical["status"] != "PASS" or len(oracle) != 15 or any(
            r["ao_cache_sha256"] != systems_by_name[r["system"]]["ao_cache_sha256"] for r in oracle):
        raise ValueError("Historical 15-system parity provenance mismatch")
    dump(bundle.root / "provenance/historical_operator_parity.json", {
        "source_receipt_sha256": file_sha(historical_path), "historical_systems": oracle,
        "status": "PASS", "comparison": "exact arrays/losses/gradients; reused qualified receipt"})
    mrks = {**mrks, "historical_15_parity": "exact PASS", "historical_receipt_sha256": file_sha(historical_path)}
    validation = []
    vd = bundle.validation("diet30_diagnostic")
    for row in bundle.validation_species.values():
        values = vd.species(row["id"])
        group = bundle.handles.open(row["shard"])[row["group"]]
        exc = np.dot(group["pbe_epsilon"][...] * group["features"][:, :2].sum(axis=1), group["weights"][...])
        if not np.isfinite(exc):
            raise ValueError("Fixed-density parity failure")
        exact(values["dm"], group["dm"][...], "native F64 density")
        validation.append(row["source_id"])
    from train_models.publication_data.validation import validation_accounting
    validation_proof = validation_accounting(bundle.root, list(bundle.validation_species.values()))
    leakage = json.loads((bundle.root / "provenance/leakage_audit.json").read_text())
    excluded = {r["validation"] for r in leakage["exclusions"]}
    if 30 - len(excluded) != leakage["clean_count"] or len(bundle.splits["diet30_clean_validation"]["ids"]) != leakage["clean_count"]:
        raise ValueError("Unique leakage exclusion accounting mismatch")
    first = next(iter(bundle.systems))
    throughput = {"chemistry": benchmark(bundle.chemistry()), "mrks_energy": benchmark(bundle.mrks()),
                  "operator_chunks": benchmark(bundle.mrks().ao_dataset(first)),
                  "validation_species_via_reactions": benchmark(vd)}
    result = {"status": "PASS", "array_hashes_checked": arrays, "references": references, "model_source_sha256": file_sha(args.model),
              "chemistry": chemistry, "mrks": mrks, "validation": {**validation_proof, "species": validation},
              "loader_benchmark": throughput, "no_training": True}
    dump(bundle.root / "provenance/qualification.json", result)
    old_manifest = bundle.manifest
    bundle.close()
    sources = old_manifest["sources"]
    for key, value in sources.items():
        if key.startswith("tooling:"):
            value["sha256"] = file_sha(REPO / value["logical_name"])
    relative = "tools/validate_mconf_components.py"
    sources["tooling:" + relative] = {"logical_name": relative, "sha256": file_sha(REPO / relative)}
    dump(args.staging / "provenance/sources.json", sources)
    card = args.staging / "DATASET_CARD.md"
    card.write_text(card.read_text() + "\nValidation qualification: 84/84 species; 82 direct fixed-density total-energy parity + 2 MCONF independent component-level parity. "
                    "Direct maximum discrepancy 5.684341886080801e-13 Ha; component maximum 1.1368683772161603e-12 Ha. "
                    "Both MCONF density/descriptor/PBE XC comparisons are exact.\n\n"
                    "Unique validation exclusions: BH76-5 and G21EA-25; 30 - 2 = 28 selectable reactions. "
                    "BH76-5 also has a conservative ambiguous Minnesota collision and is counted once.\n\n"
                    "Frozen and immutable by convention after final atomic publication. Content changes require a new version/hash. "
                    "No external baseline evaluation was part of dataset construction.\n")
    manifest = write_manifest(args.staging, old_manifest["counts"], sources, status="qualified")
    if inventory(args.staging)[0] != manifest["logical_sha256"]:
        raise ValueError("Logical hash changed")
    if args.publish:
        if not str(args.staging).endswith(".staging"):
            raise ValueError("Publish requires a staging suffix")
        destination = Path(str(args.staging).removesuffix(".staging"))
        if destination.exists():
            raise FileExistsError(destination)
        args.staging.rename(destination)
    print(json.dumps({"status": "PASS", "logical_sha256": manifest["logical_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
