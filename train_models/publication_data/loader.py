"""Identity-first PyTorch datasets with process-local lazy HDF5 handles."""

import json
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import SCHEMA, VARIANTS, file_sha, safe_child


def identity_collate(rows):
    """Variable-sized scientific samples are a list, not a padded giant tensor."""
    return rows


def chemistry_collate(rows):
    """Legacy batching with Energy retained for existing objective factories."""
    from train_models.utils import stack_reactions
    return stack_reactions(rows)


def chemistry_dispersions(root):
    """Recreate legacy scalar types; torch.tensor(ndarray) preserves native F64."""
    root = Path(root)
    values = json.loads((root / "chemistry/dispersion.json").read_text())
    metadata = json.loads((root / "chemistry/dispersion_metadata.json").read_text())
    return {name: np.asarray(values[name], dtype=row["source_dtype"]).reshape(row["source_shape"])
            for name, row in metadata.items() if row["source_present"]}


class Handles:
    def __init__(self, root, checksums=None):
        self.root = Path(root)
        self.checksums = checksums or {}
        self.verified = set()
        self.pid = None
        self.handles = OrderedDict()

    def open(self, relative):
        if self.pid != os.getpid():
            self.close()
            self.verified.clear()
            self.pid = os.getpid()
        if relative not in self.handles:
            path = safe_child(self.root, relative)
            if relative not in self.verified:
                if relative in self.checksums and file_sha(path) != self.checksums[relative]:
                    raise ValueError(f"Shard hash mismatch: {relative}")
                self.verified.add(relative)
            self.handles[relative] = h5py.File(path, "r")
            if self.handles[relative].attrs.get("schema") != SCHEMA:
                raise ValueError("HDF5 schema mismatch")
        self.handles.move_to_end(relative)
        while len(self.handles) > 8:
            self.handles.popitem(last=False)[1].close()
        return self.handles[relative]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.pid = None

    def __getstate__(self):
        return {"root": self.root, "checksums": self.checksums, "verified": set(),
                "pid": None, "handles": OrderedDict()}

    def __del__(self):
        self.close()


class PublicationDataset:
    def __init__(self, root, *, _allow_staging=False):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        allowed = ("qualified", "staging") if _allow_staging else ("qualified",)
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("status") not in allowed:
            raise ValueError("Unqualified or incompatible dataset")
        self.splits = json.loads((self.root / "splits.json").read_text())
        if "test" in self.splits:
            raise ValueError("Publication v1 must not contain a test split")
        self.checksums = {r["file"]: r["sha256"] for r in self.manifest["files"]}
        if file_sha(self.root / "splits.json") != self.checksums["splits.json"]:
            raise ValueError("Split metadata hash mismatch")
        for relative in self.checksums:
            if not safe_child(self.root, relative).is_file():
                raise ValueError(f"Missing dataset file: {relative}")
        self.reactions = self._index("chemistry/reactions.jsonl")
        self.species = self._index("chemistry/species.jsonl")
        self.systems = self._index("mrks/systems.jsonl")
        self.validation_species = self._index("validation/diet30_species.jsonl")
        self.validation_reactions = self._index("validation/diet30_reactions.jsonl")
        expected = {"train_relchem": {r["id"] for r in self.reactions.values() if r["task"] == "relchem"},
                    "train_ae17": {r["id"] for r in self.reactions.values() if r["task"] == "ae17"},
                    "train_mrks": set(self.systems), "diet30_diagnostic": set(self.validation_reactions)}
        if expected["train_relchem"] | expected["train_ae17"] != set(self.reactions):
            raise ValueError("Unknown chemistry task")
        for split, identities in expected.items():
            selected = self.splits[split]["ids"]
            if len(selected) != len(set(selected)) or set(selected) != identities:
                raise ValueError(f"Split membership mismatch: {split}")
        clean = self.splits["diet30_clean_validation"]["ids"]
        if len(clean) != len(set(clean)) or not set(clean) <= set(self.validation_reactions):
            raise ValueError("Invalid clean validation membership")
        if self.splits["diet30_diagnostic"]["selection_allowed"] or not self.splits["diet30_clean_validation"]["selection_allowed"]:
            raise ValueError("Diagnostic/validation selection policy mismatch")
        if any(not all(r.get(flag) is True for flag in ("has_exc", "has_pointwise_vxc", "has_operator"))
               for r in self.systems.values()):
            raise ValueError("Incomplete canonical mRKS target availability")
        self.handles = Handles(self.root, self.checksums)

    def _index(self, relative):
        path = safe_child(self.root, relative)
        if file_sha(path) != self.checksums[relative]:
            raise ValueError(f"Metadata hash mismatch: {relative}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        index = {row["id"]: row for row in rows}
        if len(index) != len(rows):
            raise ValueError("Duplicate canonical identity")
        return index

    def verify(self):
        for path, expected in self.checksums.items():
            if file_sha(safe_child(self.root, path)) != expected:
                raise ValueError(f"Dataset hash mismatch: {path}")
        return True

    def chemistry(self, split="train_relchem", variant="level2"):
        if split not in ("train_relchem", "train_ae17") or variant not in VARIANTS:
            raise ValueError("Unknown chemistry split/variant")
        return Chemistry(self, self.splits[split]["ids"], variant)

    def chemistry_dispersions(self):
        return chemistry_dispersions(self.root)

    def mrks(self, split="train", require_operator=True):
        if split != "train":
            raise ValueError("Unknown mRKS split")
        rows = self.splits["train_mrks"]["ids"]
        if require_operator and any(not self.systems[i]["has_operator"] for i in rows):
            raise ValueError("Incomplete operator availability")
        return MRKS(self, rows)

    def validation(self, split="diet30_clean_validation"):
        if split not in ("diet30_clean_validation", "diet30_diagnostic"):
            raise ValueError("Unknown validation split")
        return Validation(self, self.splits[split]["ids"])

    def close(self):
        self.handles.close()


class Base(Dataset):
    def __init__(self, bundle, ids):
        self.bundle, self.ids = bundle, tuple(ids)
        self.positions = {key: i for i, key in enumerate(ids)}

    def __len__(self):
        return len(self.ids)

    def group(self, row):
        return self.bundle.handles.open(row["shard"])[row["group"]]


class Chemistry(Base):
    def __init__(self, bundle, ids, variant):
        super().__init__(bundle, ids)
        self.variant = variant
        self.by_identity = {(bundle.reactions[i]["database"], bundle.reactions[i]["reaction_id"]): i for i in ids}

    def __getitem__(self, index):
        if isinstance(index, tuple):
            identity, variant = index
            return self.load_variant(identity, variant)
        return self.load_variant(self.ids[index], self.variant)

    def load_variant(self, identity, variant):
        key = self.by_identity[identity] if isinstance(identity, tuple) else identity
        if key not in self.positions or variant not in VARIANTS:
            raise KeyError((identity, variant))
        row = self.bundle.reactions[key]
        components = row["variants"][variant]
        arrays = {field: [] for field in ("Grid", "Weights", "Densities", "Gradients", "PBE_local_energies")}
        fixed, ends = [], []
        for component in components:
            group = self.group(component)
            for field, values in arrays.items():
                source_group = self.group(component["pbe_record"]) if field == "PBE_local_energies" else group
                values.append(torch.from_numpy(source_group[field][...]))
            fixed.append(float(group["fixed_nonxc"][()]))
            ends.append(sum(len(a) for a in arrays["Weights"]))
        return {"Database": row["database"], "ReactionID": row["reaction_id"],
                "canonical_id": key, "variant": variant,
                "Components": np.array(row["components"]),
                "Coefficients": torch.tensor(row["coefficients"], dtype=torch.float32, device="cpu"),
                "Energy": torch.tensor([row["target_kcal_mol"]], dtype=torch.float32, device="cpu"),
                "component_paths": row["component_paths"][variant],
                "HF_energies": torch.tensor(fixed, dtype=torch.float32, device="cpu"),
                "backsplit_ind": torch.tensor(ends, dtype=torch.float32, device="cpu"),
                **{k: torch.cat(v) for k, v in arrays.items()}}


class MRKS(Base):
    def __getitem__(self, index):
        row = self.bundle.systems[self.ids[index]]
        group = self.group(row)
        return {"id": row["id"], "name": row["source_id"],
                **{k: torch.from_numpy(np.array(group[k][...], copy=True)).to(
                    dtype=torch.float32 if k in ("DensityDescriptorsN10", "weights") else torch.float64) for k in
                   ("DensityDescriptorsN10", "weights", "Exc", "VxcLegacy", "Overlap", "RefAO")}}

    def ao_chunks(self, identity, chunk_size=4096):
        if chunk_size <= 0:
            raise ValueError("Positive AO chunk size required")
        row = self.bundle.systems[identity]
        group = self.group(row)
        for start in range(0, row["n_grid"], chunk_size):
            rows = slice(start, min(start + chunk_size, row["n_grid"]))
            yield rows, {k: torch.from_numpy(group[k][rows]) for k in ("phi", "grad_phi", "lap_phi")}

    def operator_system(self, identity, device="cpu", dtype=torch.float32, chunk_size=4096):
        from train_models.lap_moo_training import AOFactorChunk, MRKSOperatorSystem
        row = self.bundle.systems[identity]
        values = self[self.positions[identity]]
        chunks = tuple(AOFactorChunk(rows, **{k: v.to(device=device, dtype=dtype) for k, v in factors.items()})
                       for rows, factors in self.ao_chunks(identity, chunk_size))
        return MRKSOperatorSystem(row["source_id"], values["DensityDescriptorsN10"].to(device=device, dtype=dtype),
                                  values["weights"].to(device=device, dtype=dtype), values["Exc"].to(device),
                                  values["RefAO"].to(device), values["Overlap"].to(device), chunks)

    def ao_dataset(self, identity, chunk_size=4096):
        """Grid chunks as CPU DataLoader samples; no worker shares an HDF5 handle."""
        return AOChunks(self.bundle, identity, chunk_size)


class AOChunks(Dataset):
    def __init__(self, bundle, identity, chunk_size):
        if identity not in bundle.systems or chunk_size <= 0:
            raise ValueError("Known system and positive AO chunk size required")
        self.bundle, self.identity, self.chunk_size = bundle, identity, chunk_size

    def __len__(self):
        return (self.bundle.systems[self.identity]["n_grid"] + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, index):
        row = self.bundle.systems[self.identity]
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, stop = index*self.chunk_size, min((index+1)*self.chunk_size, row["n_grid"])
        group = self.bundle.handles.open(row["shard"])[row["group"]]
        return {"system_id": self.identity, "start": start, "stop": stop,
                **{key: torch.from_numpy(group[key][start:stop]).float() for key in
                   ("DensityDescriptorsN10", "weights", "phi", "grad_phi", "lap_phi")}}


class Validation(Base):
    def __getitem__(self, index):
        row = self.bundle.validation_reactions[self.ids[index]]
        return {**row, "species": [self.species(c["species_id"]) for c in row["components"]]}

    def species(self, identity):
        row = self.bundle.validation_species[identity]
        group = self.group(row)
        return {**row, **{k: torch.from_numpy(np.array(group[k][...], copy=True)) for k in
                         ("features", "weights", "nonxc", "dm")}}
