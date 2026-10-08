"""Trusted legacy conversion; canonical output contains no Python objects."""

import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .contracts import (
    EXCLUSIONS,
    FIELDS,
    SCHEMA,
    VARIANTS,
    annotate,
    array_sha,
    canonical,
    canonical_id,
    file_sha,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path, rows):
    Path(path).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def portable(value):
    """Historical metadata may contain source-machine paths; preserve logical names."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and (":\\" in value or ":/" in value or value.startswith(("/mnt/", "/home/"))):
        return value.replace("\\", "/").rsplit("/", 1)[-1]
    return value


class Shards:
    def __init__(self, root, modality, limit=512 * 1024**2):
        self.root, self.modality, self.limit = Path(root), modality, limit
        (self.root / modality).mkdir(parents=True, exist_ok=True)
        self.number, self.size, self.handle = -1, 0, None

    def group(self, key, estimated_bytes):
        if self.handle is None or (self.size and self.size + estimated_bytes > self.limit):
            self.close()
            self.number += 1
            self.relative = f"{self.modality}/{self.modality}_{self.number:03d}.h5"
            self.handle = h5py.File(self.root / self.relative, "x")
            self.handle.attrs["schema"] = SCHEMA
            self.size = 0
        self.size += estimated_bytes
        return self.handle.create_group(key), {"shard": self.relative, "group": key}

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def put(group, key, value, production=None, source=None):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {key}")
    opts = {"chunks": (min(4096, len(array)), *array.shape[1:]), "compression": "lzf", "shuffle": True} if array.ndim and len(array) else {}
    dataset = group.create_dataset(key, data=array, **opts)
    annotate(dataset, key, production, source)
    dataset.attrs["content_sha256"] = array_sha(array)


def chemistry(root, source, dispersions):
    from train_models.lap_moo_panel import MinnesotaGroupStore
    store = MinnesotaGroupStore(source / "manifest.json", cache_groups=0)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["group_count"] != 268 or manifest["variant_count"] != 2144:
        raise ValueError("Qualified 268 x 8 corpus required")
    writer, reactions, species = Shards(root, "chemistry"), [], {}
    records, pbe_records, counts, source_hashes = {}, {}, Counter(), []
    try:
        for item in sorted(manifest["groups"], key=lambda r: (r["database"], r["reaction_id"])):
            db, rid = item["database"], item["reaction_id"]
            if rid in EXCLUSIONS.get(db, []):
                raise ValueError("Held-out Minnesota identity reintroduced")
            variants = store.load_group((db, rid))
            if tuple(item["variant_suffixes"]) != VARIANTS:
                raise ValueError("Missing/changed augmentation axis")
            base = variants[0]
            reaction = {"id": canonical_id("reaction", [db, rid]), "database": db, "reaction_id": rid,
                        "task": "ae17" if db == "AE17" else "relchem",
                        "components": base["Components"].tolist(),
                        "coefficients": base["Coefficients"].tolist(),
                        "target_kcal_mol": float(base["Energy"][0]), "variants": {}, "component_paths": {},
                        "source_group_sha256": item["file_sha256"]}
            counts[db] += 1
            source_hashes.append({"file": item["file"], "sha256": item["file_sha256"]})
            for suffix, variant in zip(VARIANTS, variants):
                if (variant["Components"].tolist() != reaction["components"] or
                        variant["Coefficients"].tolist() != reaction["coefficients"] or
                        float(variant["Energy"][0]) != reaction["target_kcal_mol"]):
                    raise ValueError("Grid variant changed chemical identity/target")
                reaction["component_paths"][suffix] = [portable(p) for p in variant["component_paths"]]
                reaction["variants"][suffix] = []
                start = 0
                for i, component in enumerate(reaction["components"]):
                    end = int(variant["backsplit_ind"][i])
                    arrays = {k: variant[k][start:end].detach().cpu().numpy() for k in
                              ("Grid", "Weights", "Densities", "Gradients", "PBE_local_energies")}
                    arrays["fixed_nonxc"] = np.asarray(variant["HF_energies"][i].item(), dtype=np.float32)
                    pbe = arrays.pop("PBE_local_energies")
                    pbe_sha = array_sha(pbe)
                    if pbe_sha not in pbe_records:
                        pbe_id = canonical_id("pbe_grid_values", pbe_sha)
                        pbe_group, pbe_reference = writer.group(pbe_id, pbe.nbytes)
                        put(pbe_group, "PBE_local_energies", pbe,
                            production="<f4 widened to <f8 for matched chemistry")
                        pbe_records[pbe_sha] = pbe_reference
                    signature = {k: array_sha(v) for k, v in arrays.items()}
                    species_id = canonical_id("species", component)
                    key = (component, suffix)
                    if key in records:
                        if records[key]["signature"] != signature:
                            raise ValueError(f"Inconsistent repeated species/grid {key}")
                        reference = records[key]["reference"]
                    else:
                        grid_id = canonical_id("grid", [component, suffix])
                        group, reference = writer.group(grid_id, sum(v.nbytes for v in arrays.values()))
                        for field, array in arrays.items():
                            if array.dtype != np.float32:
                                raise ValueError("Historical chemistry rounding is not F32")
                            put(group, field, array, production="<f4 widened to <f8 for matched chemistry")
                        reference = {**reference, "species_id": species_id, "grid_id": grid_id}
                        records[key] = {"signature": signature, "reference": reference}
                        species.setdefault(species_id, {"id": species_id, "source_id": component, "variants": {}})["variants"][suffix] = reference
                    reaction["variants"][suffix].append({**reference, "pbe_record": pbe_records[pbe_sha]})
                    start = end
                if start != len(variant["Weights"]):
                    raise ValueError("Invalid component grid partition")
            reactions.append(reaction)
            print(json.dumps({"chemistry_identity": [db, rid]}), flush=True)
    finally:
        writer.close()
    for reaction in reactions:
        reaction["identity_sampling_weight"] = 1 / (17 if reaction["task"] == "ae17" else 251)
        reaction["db_weight"] = counts[reaction["database"]] / (17 if reaction["task"] == "ae17" else 251)
    jsonl(root / "chemistry/reactions.jsonl", reactions)
    jsonl(root / "chemistry/species.jsonl", [species[k] for k in sorted(species)])
    with Path(dispersions).open("rb") as stream:
        correction = pickle.load(stream)
    used = {r["source_id"] for r in species.values()}
    dump(root / "chemistry/dispersion.json", {key: float(correction.get(key, 0.0)) for key in sorted(used)})
    dump(root / "chemistry/dispersion_metadata.json", {key: {
        "source_present": key in correction, "source_dtype": np.asarray(correction.get(key, 0)).dtype.str,
        "source_shape": np.asarray(correction.get(key, 0)).shape, "storage_dtype": "JSON float64",
        "production_dtype": "native source scalar dtype through torch.tensor", "units": "hartree"}
        for key in sorted(used)})
    return {"identities": len(reactions), "variants": len(reactions) * 8,
            "relchem": counts.total() - counts["AE17"], "ae17": counts["AE17"],
            "species": len(species), "species_grids": len(records), "databases": dict(counts),
            "source_groups": source_hashes}


def mrks(root, central, cache, dispersions):
    from train_models.lap_operator_data import load_central_operator_record
    central_manifest = json.loads((central / "manifest.json").read_text())
    ao_manifest = json.loads((cache / "manifest.json").read_text())
    if (len(central_manifest["records"]) != 90 or len(ao_manifest["records"]) != 90 or
            ao_manifest["central_data_manifest_sha256"] != file_sha(central / "manifest.json")):
        raise ValueError("90/90 qualified source/cache binding required")
    ao_rows = {r["system_name"]: r for r in ao_manifest["records"]}
    if len(ao_rows) != 90:
        raise ValueError("Duplicate mRKS system")
    writer, systems = Shards(root, "mrks"), []
    try:
        for row in sorted(central_manifest["records"], key=lambda r: r["system_name"]):
            name, target = row["system_name"], central / row["file"]
            ao = ao_rows[name]
            if file_sha(target) != row["file_sha256"] or file_sha(cache / ao["file"]) != ao["cache_file_sha256"]:
                raise ValueError("mRKS source/cache hash mismatch")
            record = load_central_operator_record(target)
            if ao["source_record_sha256"] != record.metadata["record_sha256"]:
                raise ValueError("mRKS logical record mismatch")
            identity = canonical_id("mrks", name)
            group, reference = writer.group(identity, ao["ao_factor_raw_bytes_float32"])
            for field in record.metadata["dataset_sha256"]:
                array = getattr(record, field)
                production = "<f4" if field in ("DensityDescriptorsN10", "weights") else array.dtype.str
                put(group, field, array, production=production)
            group.attrs["source_metadata_json"] = canonical(portable(record.metadata)).decode()
            with h5py.File(cache / ao["file"], "r") as source:
                for field in ("phi", "grad_phi", "lap_phi"):
                    source.copy(source[field], group, name=field)
                    annotate(group[field], field, "<f4 widened to <f8 in AO assembly", "<f8 PySCF eval_ao rounded to <f4")
                    # Content hash independent of HDF5 byte serialization.
                    ds = group[field]
                    import hashlib
                    digest = hashlib.sha256(canonical({"shape": ds.shape, "dtype": ds.dtype.str}))
                    for start in range(0, len(ds), 4096):
                        block = ds[start:start + 4096]
                        if not np.isfinite(block).all():
                            raise ValueError("Nonfinite AO factor")
                        digest.update(block.tobytes())
                    ds.attrs["content_sha256"] = digest.hexdigest()
            systems.append({"id": identity, "source_id": name, **reference,
                            "n_grid": len(record.weights), "n_ao": record.Overlap.shape[0],
                            "has_exc": True, "has_pointwise_vxc": True, "has_operator": True,
                            "source_sha256": row["file_sha256"], "source_record_sha256": row["record_sha256"],
                            "ao_cache_sha256": ao["cache_file_sha256"]})
            print(json.dumps({"mrks_system": name}), flush=True)
    finally:
        writer.close()
    jsonl(root / "mrks/systems.jsonl", systems)
    with Path(dispersions).open("rb") as stream:
        correction = pickle.load(stream)
    dump(root / "mrks/dispersion.json", {r["source_id"]: float(correction.get(r["source_id"], 0)) for r in systems})
    return {"systems": len(systems), "has_exc": 90, "has_pointwise_vxc": 90, "has_operator": 90}


def inventory(root):
    """Canonical array content + scientific metadata, excluding runtime receipts."""
    import hashlib
    parts, files = [], []
    scientific_metadata = {"splits.json", "provenance/schema.json", "provenance/sources.json",
        "provenance/leakage_audit.json", "provenance/training_geometry_identities.json",
        "provenance/validation_protocol.json", "provenance/validation_qualification.json", "chemistry/reactions.jsonl", "chemistry/species.jsonl",
        "chemistry/dispersion.json", "chemistry/dispersion_metadata.json", "mrks/systems.jsonl", "mrks/dispersion.json",
        "validation/diet30_reactions.jsonl", "validation/diet30_species.jsonl"}
    def logical_metadata(value):
        if isinstance(value, dict):
            return {key: logical_metadata(item) for key, item in value.items()
                    if key not in ("shard", "shard_sha256", "component_receipt_sha256")}
        if isinstance(value, list):
            return [logical_metadata(item) for item in value]
        return value
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ("dataset_manifest.json", "checksums.sha256"):
            continue
        relative = path.relative_to(root).as_posix()
        files.append({"file": relative, "sha256": file_sha(path), "bytes": path.stat().st_size})
        if path.suffix == ".h5":
            with h5py.File(path, "r") as handle:
                def collect(name, obj, relative=relative):
                    if isinstance(obj, h5py.Dataset):
                        if obj.attrs.get("semantic_name") not in FIELDS:
                            raise ValueError("Unknown numerical semantic")
                        parts.append(["array", name, dict(obj.attrs)])
                    elif obj.attrs:
                        parts.append(["group", name, dict(obj.attrs)])
                handle.visititems(collect)
        elif relative in scientific_metadata:
            if path.suffix == ".jsonl":
                value = [json.loads(line) for line in path.read_text().splitlines()]
            else:
                value = json.loads(path.read_text())
            parts.append(["metadata", relative, logical_metadata(value)])
    parts.sort(key=canonical)
    return hashlib.sha256(canonical(parts)).hexdigest(), files


def write_manifest(root, counts, sources, status="staging"):
    logical, files = inventory(root)
    repo = Path(__file__).resolve().parents[2]
    try:
        git = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError:
        # Managed Windows worktree .git pointers contain Windows gitdir paths.
        # Under WSL query the same repository using the native Git executable.
        import shutil
        native_git = shutil.which("git.exe")
        if native_git is None or not str(repo).startswith("/mnt/"):
            raise
        native_repo = subprocess.check_output(["wslpath", "-w", str(repo)], text=True).strip()
        git = subprocess.check_output([native_git, "-C", native_repo, "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema": SCHEMA, "status": status, "frozen": status == "qualified", "logical_sha256": logical, "repository_commit": git,
                "counts": counts, "sources": sources, "training_exclusions": EXCLUSIONS,
                "chemistry_sample_unit": "Database,ReactionID; grid variant is augmentation",
                "grid_variants": list(VARIANTS), "test_split": False,
                "immutability_policy": "Freeze after qualification; any content change requires a new version/hash",
                "logical_hash_policy": "canonical scientific metadata and array content; independent of compression and shard layout",
                "validation_qualification": json.loads((root / "provenance/validation_qualification.json").read_text())
                    if (root / "provenance/validation_qualification.json").exists() else None,
                "primary_dispersion": "PBE0-D3(BJ)", "secondary_dispersion": "PBE-D3(BJ)",
                "operator_protocol": "lap-weakform-ao-v1", "units_axes": FIELDS,
                "dtype_policy": "per-dataset attributes; preserve historical F32 chemistry before matched F64; native F64 validation",
                "files": files}
    dump(root / "dataset_manifest.json", manifest)
    (root / "checksums.sha256").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in files))
    return manifest


def schema(root):
    import ast
    source = Path(__file__).resolve().parents[1] / "optuna_joint.py"
    def number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return number(node.left) / number(node.right)
        raise ValueError("Unexpected chemistry weighting expression")
    constants = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ("FCHEM_DB_WEIGHTS", "FREQ_WEIGHTS") for t in node.targets):
            constants[node.targets[0].id] = {ast.literal_eval(k): number(v)
                for k, v in zip(node.value.keys, node.value.values)}
    constants["MEAN_WEIGHT"] = sum(constants["FCHEM_DB_WEIGHTS"][db] * constants["FREQ_WEIGHTS"][db]
                                   for db in constants["FCHEM_DB_WEIGHTS"]) / len(constants["FCHEM_DB_WEIGHTS"])
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    dump(root / "provenance/schema.json", {"schema": SCHEMA, "fields": FIELDS, "variants": VARIANTS,
         "chemistry_grid_columns": ["rho_alpha", "rho_beta", "sigma_aa", "sigma_aa+2sigma_ab+sigma_bb",
                                    "sigma_bb", "tau_alpha", "tau_beta", "lapl_alpha", "lapl_beta"],
         "gradient_columns": ["sigma_aa", "sigma_ab", "sigma_bb"],
         "reaction_energy_units": "kcal/mol", "reaction_energy_conversion": 627.5095,
         "fixed_nonxc_training": "exact historical component ener[0] contribution, stored F32, no reinterpretation",
         "mRKS_Exc": "exact qualified legacy target; do not replace with NPZ exc_wf",
         "metadata_fields": {
             "chemistry.target_kcal_mol": {"source_dtype": "<f4", "storage_dtype": "JSON float64", "production_dtype": "<f4 then matched-F64", "units": "kcal/mol"},
             "chemistry.coefficients": {"source_dtype": "<f4", "storage_dtype": "JSON float64", "production_dtype": "<f4 then matched-F64", "units": "stoichiometric count"},
             "chemistry.dispersion": {"source_dtype": "<f8 zero-dimensional numpy.ndarray", "storage_dtype": "JSON float64 + dtype/shape metadata", "production_dtype": "<f8 through native torch.tensor semantics", "units": "hartree"},
             "validation.reference_energy": {"source_dtype": "YAML float64", "storage_dtype": "JSON float64", "production_dtype": "<f8", "units": "kcal/mol"},
             "validation.dispersion": {"source_dtype": "text decimal parsed F64", "storage_dtype": "JSON float64", "production_dtype": "<f8", "units": "hartree"}},
         "potential_gauge": "source asymptotic gauge unchanged; no constant or trace subtraction",
         "operator": "dExc/dP_total; symmetric-orthogonalized squared Frobenius/nAO",
         "chemistry_loss_constants": constants,
         "full251_scalar": "arithmetic mean of 251 corrected singleton batch_fchem losses",
         "singleton_loss": "sqrt(max(squared kcal/mol error,1e-20)) * FCHEM_DB_WEIGHTS[db] * FREQ_WEIGHTS[db] / MEAN_WEIGHT"})
