"""Deterministic data preparation helpers for the Lap MOO survey panel.

This module defines only the survey data panel and disposable AO-factor cache.
It does not define objectives, weights, optimizers, or training behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

PANEL_PROTOCOL = "lap-moo-survey-panel-v1"
MN_PANEL_PROTOCOL = "lap-moo-minnesota-panel-v1"
AO_CACHE_PROTOCOL = "lap-ao-factor-cache-f32-v1"
OPERATOR_PROTOCOL = "lap-weakform-ao-v1"
CENTRAL_DATA_PROTOCOL = "lap-operator-central-ao-noh-v1"
MN_GROUP_STORE_PROTOCOL = "lap-moo-minnesota-group-store-v1"
MN_PER_DATABASE = 3
EXPECTED_MINNESOTA_DATABASES = (
    "ABDE4",
    "AE17",
    "DBH76",
    "EA13",
    "IP13",
    "MGAE109",
    "NCCE31",
    "PA8",
    "pTC13",
)

# Fixed before looking at objective values; covers 60–248 AOs and H through
# Cl, with Al, P, Si, and S represented. Names resolve against the audited
# all-90 central operator manifest.
MRKS_PANEL_SYSTEMS = (
    "H2",
    "HLi",
    "BH",
    "BeH2",
    "H2O",
    "CH4",
    "N2",
    "CO",
    "CH2O",
    "C2H2_iso2",
    "H4Si",
    "AlBeH",
    "ClH",
    "ClHS",
    "HPSi_iso2",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_suffix(reaction: dict[str, Any]) -> str:
    component_paths = reaction.get("component_paths")
    if not component_paths:
        raise ValueError("Minnesota reaction variant is missing component_paths.")
    suffixes = set()
    for component_path in component_paths:
        filename = os.path.basename(str(component_path).replace("\\", "/"))
        stem = os.path.splitext(filename)[0]
        suffixes.add(stem.split("__", 1)[1] if "__" in stem else "default")
    if len(suffixes) != 1:
        raise ValueError(f"Reaction components disagree on augmentation suffix: {sorted(suffixes)}")
    return next(iter(suffixes))


def select_minnesota_panel(
    canonical_manifest: dict[str, Any],
    *,
    reactions_per_database: int = MN_PER_DATABASE,
) -> list[dict[str, Any]]:
    """Select evenly spaced immutable group identities from every database."""
    if canonical_manifest.get("schema") != "canonical-minnesota-view-v1":
        raise ValueError("Unsupported Minnesota canonical manifest schema.")
    groups = canonical_manifest.get("groups")
    if not isinstance(groups, list):
        raise TypeError("Minnesota manifest must contain a groups list.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_keys: set[int] = set()
    for group in groups:
        database = group.get("database")
        group_key = group.get("group_key")
        if not isinstance(database, str) or not isinstance(group_key, int):
            raise TypeError("Minnesota group is missing a database or integer group_key.")
        if group_key in seen_keys:
            raise ValueError(f"Duplicate Minnesota source group key {group_key}.")
        seen_keys.add(group_key)
        suffixes = group.get("available_suffixes")
        if (
            not isinstance(suffixes, list)
            or not suffixes
            or len(suffixes) != len(set(suffixes))
            or group.get("variant_count") != len(suffixes)
        ):
            raise ValueError(f"{database} group {group_key} has invalid variant metadata.")
        selected_suffix = group.get("selected_suffix")
        if selected_suffix not in suffixes:
            raise ValueError(f"{database} group {group_key} canonical suffix is unavailable.")
        grouped.setdefault(database, []).append(group)

    if tuple(sorted(grouped)) != tuple(sorted(EXPECTED_MINNESOTA_DATABASES)):
        raise ValueError(
            "Minnesota panel must span the expected nine databases; "
            f"observed={sorted(grouped)}."
        )
    selected: list[dict[str, Any]] = []
    for database in sorted(grouped):
        rows = sorted(grouped[database], key=lambda row: row["group_key"])
        if len(rows) < reactions_per_database:
            raise ValueError(
                f"{database} has {len(rows)} groups; cannot select {reactions_per_database}."
            )
        # Integer round-half-up positions from first to last source group.
        indexes = [
            (2 * rank * (len(rows) - 1) + (reactions_per_database - 1))
            // (2 * (reactions_per_database - 1))
            for rank in range(reactions_per_database)
        ]
        if len(indexes) != len(set(indexes)):
            raise ValueError(f"{database} is too small for distinct panel selections.")
        for rank, index in enumerate(indexes):
            row = rows[index]
            selected.append(
                {
                    "database": database,
                    "reaction_id": row["reaction_id"],
                    "source_group_key": row["group_key"],
                    "canonical_output_index": row["output_index"],
                    "database_group_rank": index,
                    "database_group_count": len(rows),
                    "canonical_point_count": row["point_count"],
                    "canonical_variant_suffix": row["selected_suffix"],
                    "variant_suffixes": sorted(row["available_suffixes"]),
                    "variant_count": row["variant_count"],
                }
            )
    if len(selected) != reactions_per_database * len(EXPECTED_MINNESOTA_DATABASES):
        raise AssertionError("Minnesota panel selector produced an unexpected row count.")
    return selected


def build_panel_definition(
    canonical_manifest_path: str | Path,
    central_data_dir: str | Path,
    *,
    reactions_per_database: int = MN_PER_DATABASE,
) -> dict[str, Any]:
    """Build a compact deterministic panel definition from audited manifests."""
    canonical_manifest_path = Path(canonical_manifest_path).resolve()
    central_data_dir = Path(central_data_dir).resolve()
    canonical = _read_json(canonical_manifest_path)
    central_manifest_path = central_data_dir / "manifest.json"
    central = _read_json(central_manifest_path)
    if canonical.get("base_reaction_count") != 268 or len(canonical.get("groups", [])) != 268:
        raise ValueError("Minnesota source manifest does not describe the verified 268-group view.")
    if central.get("protocol") != CENTRAL_DATA_PROTOCOL or central.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError("Central data manifest uses an incompatible operator protocol.")
    central_records = central.get("records")
    if not isinstance(central_records, list) or len(central_records) != 90:
        raise ValueError("MOO panel requires the audited all-90 central operator corpus.")
    if central.get("built_systems") != [row.get("system_name") for row in central_records]:
        raise ValueError("Central corpus system list does not match its record list.")
    records_by_name = {row.get("system_name"): row for row in central_records}
    if len(records_by_name) != 90 or None in records_by_name:
        raise ValueError("Central corpus manifest has missing or duplicate system names.")
    missing = sorted(set(MRKS_PANEL_SYSTEMS) - records_by_name.keys())
    if missing:
        raise ValueError(f"Central corpus is missing selected mRKS systems: {missing}.")

    source = canonical.get("source")
    if not isinstance(source, dict) or not source.get("sha256") or not source.get("size_bytes"):
        raise ValueError("Canonical manifest is missing full grouped-source identity.")
    selected_mn = select_minnesota_panel(canonical, reactions_per_database=reactions_per_database)
    selected_mrks = []
    for name in MRKS_PANEL_SYSTEMS:
        row = records_by_name[name]
        selected_mrks.append(
            {
                "system_name": name,
                "file": row["file"],
                "central_file_sha256": row["file_sha256"],
                "record_sha256": row["record_sha256"],
                "point_count": row["point_count"],
                "nao": row["nao"],
            }
        )
    return {
        "schema": PANEL_PROTOCOL,
        "panel_definition_protocol": PANEL_PROTOCOL,
        "minnesota": {
            "panel_protocol": MN_PANEL_PROTOCOL,
            "source_manifest_path": str(canonical_manifest_path),
            "source_manifest_sha256": _sha256_file(canonical_manifest_path),
            "canonical_source_path": source.get("path"),
            "canonical_source_sha256": source["sha256"],
            "canonical_source_bytes": source["size_bytes"],
            "canonical_view_sha256": canonical.get("output_sha256"),
            "selection_rule": "three evenly spaced source group ranks per database; first/mid/last; all variant suffixes retained",
            "database_count": len(EXPECTED_MINNESOTA_DATABASES),
            "reactions_per_database": reactions_per_database,
            "selected_reaction_count": len(selected_mn),
            "reactions": selected_mn,
            "variant_policy": "preserve every suffix from the source group; sort suffix identity before materialization",
        },
        "mrks": {
            "central_manifest_path": str(central_manifest_path),
            "central_manifest_sha256": _sha256_file(central_manifest_path),
            "central_protocol": central["protocol"],
            "operator_protocol": central["operator_protocol"],
            "selection_rule": "fixed coverage panel selected before objective evaluation; span basis size and first/second-row/high-Z elements",
            "selected_system_count": len(selected_mrks),
            "systems": selected_mrks,
        },
    }


class _HashingReader:
    """Hash a pickle source while ``pickle.load`` reads it exactly once."""

    def __init__(self, stream: BinaryIO, expected_size: int, *, low_memory_bytes: int):
        self._stream = stream
        self._digest = hashlib.sha256()
        self._expected_size = expected_size
        self._low_memory_bytes = low_memory_bytes
        self.bytes_read = 0
        self._next_progress = 1024**3

    def _track(self, block: bytes) -> bytes:
        if block:
            self._digest.update(block)
            self.bytes_read += len(block)
            if self.bytes_read >= self._next_progress:
                try:
                    import psutil

                    available = psutil.virtual_memory().available
                except ImportError:
                    available = None
                print(
                    json.dumps(
                        {
                            "event": "grouped_pickle_read_progress",
                            "bytes_read": self.bytes_read,
                            "source_bytes": self._expected_size,
                            "available_memory_bytes": available,
                        }
                    ),
                    flush=True,
                )
                if available is not None and available < self._low_memory_bytes:
                    raise MemoryError(
                        "Stopping grouped pickle load before available RAM falls below the safety floor."
                    )
                self._next_progress += 1024**3
        return block

    def read(self, size: int = -1) -> bytes:
        return self._track(self._stream.read(size))

    def readline(self, size: int = -1) -> bytes:
        return self._track(self._stream.readline(size))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _LazyStorageReference:
    """On-disk torch storage payload retained without deserializing its tensor."""

    def __init__(self, spool_path: Path, offset: int, length: int):
        self.spool_path = spool_path
        self.offset = offset
        self.length = length
        self._storage = None

    def materialize(self):
        if self._storage is None:
            import torch

            with self.spool_path.open("rb") as stream:
                stream.seek(self.offset)
                payload = stream.read(self.length)
            if len(payload) != self.length:
                raise ValueError("Truncated lazy torch storage payload.")
            self._storage = torch.storage._load_from_bytes(payload)
        return self._storage

    def release(self) -> None:
        self._storage = None


class _LazyTensorReference:
    """Tensor metadata plus a deferred, exact torch-storage reconstruction."""

    def __init__(self, rebuild_name: str, args: tuple[Any, ...]):
        self.rebuild_name = rebuild_name
        self.args = args

    def materialize(self):
        import torch

        args = list(self.args)
        storage = args[0]
        if isinstance(storage, _LazyStorageReference):
            args[0] = storage.materialize()
        rebuild = getattr(torch._utils, self.rebuild_name)
        return rebuild(*args)


class _StorageSpool:
    """Append serialized tensor-storage payloads to a disposable external file."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("xb")
        self.bytes_written = 0
        self.references: list[_LazyStorageReference] = []

    def store(self, payload: bytes) -> _LazyStorageReference:
        offset = self.stream.tell()
        self.stream.write(payload)
        self.bytes_written += len(payload)
        reference = _LazyStorageReference(self.path, offset, len(payload))
        self.references.append(reference)
        return reference

    def release_materialized(self) -> None:
        for reference in self.references:
            reference.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


class _HashingWriter:
    """Hash a pickle while it is written without rereading the group file."""

    def __init__(self, stream: BinaryIO):
        self._stream = stream
        self._digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, block: bytes) -> int:
        written = self._stream.write(block)
        if written != len(block):
            raise OSError("Short write while writing Minnesota group record.")
        self._digest.update(block)
        self.bytes_written += written
        return written

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _LazyTensorUnpickler(pickle.Unpickler):
    """Keep tensor storage as disk-backed proxies during a grouped source pass."""

    def __init__(self, stream: BinaryIO, storage_spool: _StorageSpool):
        super().__init__(stream)
        self._storage_spool = storage_spool

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return self._storage_spool.store
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return lambda *args: _LazyTensorReference(name, args)
        return super().find_class(module, name)


def _materialize_selected_tensors(value: Any) -> Any:
    if isinstance(value, _LazyTensorReference):
        return value.materialize()
    if isinstance(value, dict):
        for key in tuple(value):
            value[key] = _materialize_selected_tensors(value[key])
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _materialize_selected_tensors(item)
        return value
    if isinstance(value, tuple):
        return tuple(_materialize_selected_tensors(item) for item in value)
    return value


def _contains_lazy_tensor(value: Any) -> bool:
    if isinstance(value, (_LazyStorageReference, _LazyTensorReference)):
        return True
    if isinstance(value, dict):
        return any(_contains_lazy_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_lazy_tensor(item) for item in value)
    return False


def _assert_external_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError(f"Survey data outputs must stay outside the repository: {resolved}")
    return resolved


def materialize_minnesota_panel(
    grouped_pickle_path: str | Path,
    panel_definition: dict[str, Any],
    output_pickle_path: str | Path,
    *,
    minimum_free_memory_bytes: int = 1024**3,
) -> dict[str, Any]:
    """Extract selected full groups with lazy storage spooling.

    Tensor storage bytes are copied to a disposable disk spool during one
    sequential source read. The small group/variant metadata graph remains in
    memory; only selected groups are reconstructed as torch tensors.
    """
    source_path = Path(grouped_pickle_path).resolve()
    output_path = _assert_external_path(output_pickle_path)
    mn_definition = panel_definition.get("minnesota", {})
    expected_size = mn_definition.get("canonical_source_bytes")
    expected_hash = mn_definition.get("canonical_source_sha256")
    if not isinstance(expected_size, int) or not isinstance(expected_hash, str):
        raise TypeError("Panel definition lacks the grouped Minnesota source identity.")
    if source_path.stat().st_size != expected_size:
        raise ValueError("Grouped Minnesota pickle size does not match its source manifest.")
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        available = None
    if available is not None and available < expected_size + minimum_free_memory_bytes:
        raise MemoryError(
            "Insufficient available RAM for one-pass grouped-pickle deserialization: "
            f"available={available}, required_at_least={expected_size + minimum_free_memory_bytes}."
        )
    source_groups = mn_definition.get("reactions")
    if not isinstance(source_groups, list) or not source_groups:
        raise ValueError("Panel definition has no Minnesota reaction identities.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".partial")
    spool_path = output_path.with_name(output_path.name + ".storages.partial")
    if output_path.exists() or temp_path.exists() or spool_path.exists():
        raise FileExistsError(f"Refusing to overwrite Minnesota panel output: {output_path}")

    started = time.perf_counter()
    low_memory_floor = max(minimum_free_memory_bytes, 1024**3)
    selected_data: dict[int, list[dict[str, Any]]]
    output_published = False
    storage_spool_bytes = 0
    try:
        with _StorageSpool(spool_path) as storage_spool:
            with source_path.open("rb") as raw_stream:
                reader = _HashingReader(
                    raw_stream,
                    expected_size,
                    low_memory_bytes=low_memory_floor,
                )
                source_data = _LazyTensorUnpickler(reader, storage_spool).load()
                if reader.bytes_read < expected_size:
                    reader._track(raw_stream.read())
                source_digest = reader.hexdigest()
                if source_digest != expected_hash:
                    raise ValueError(
                        f"Grouped source hash mismatch: expected {expected_hash}, got {source_digest}."
                    )
            storage_spool.close()
            storage_spool_bytes = storage_spool.bytes_written
        if not isinstance(source_data, dict):
            raise TypeError("Grouped Minnesota source must contain a dictionary.")
        selected_data = {}
        for group in source_groups:
            group_key = group["source_group_key"]
            if group_key not in source_data:
                raise ValueError(f"Minnesota source is missing panel group {group_key}.")
            variants = source_data[group_key]
            if not isinstance(variants, list):
                raise TypeError(f"Minnesota source group {group_key} is not a variant list.")
            by_suffix: dict[str, dict[str, Any]] = {}
            for reaction in variants:
                suffix = _variant_suffix(reaction)
                if suffix in by_suffix:
                    raise ValueError(f"Duplicate {suffix!r} variant in source group {group_key}.")
                if reaction.get("Database") != group["database"] or reaction.get("ReactionID") != group["reaction_id"]:
                    raise ValueError(f"Minnesota source identity mismatch in group {group_key}.")
                by_suffix[suffix] = reaction
            expected_suffixes = group["variant_suffixes"]
            if sorted(by_suffix) != expected_suffixes:
                raise ValueError(
                    f"Variant suffix mismatch for group {group_key}: "
                    f"expected={expected_suffixes}, got={sorted(by_suffix)}."
                )
            selected_data[group_key] = [
                _materialize_selected_tensors(by_suffix[suffix])
                for suffix in expected_suffixes
            ]
        if _contains_lazy_tensor(selected_data):
            raise ValueError("Selected Minnesota panel still contains lazy tensor references.")
        with temp_path.open("xb") as output_stream:
            pickle.dump(selected_data, output_stream, protocol=pickle.HIGHEST_PROTOCOL)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temp_path, output_path)
        output_published = True
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        if spool_path.exists():
            spool_path.unlink()
        if output_published and output_path.exists():
            output_path.unlink()
        raise
    if spool_path.exists():
        spool_path.unlink()

    selected_group_count = len(selected_data)
    selected_variant_count = sum(len(group) for group in selected_data.values())
    del selected_data, source_data
    return {
        "protocol": MN_PANEL_PROTOCOL,
        "file": output_path.name,
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "file_bytes": output_path.stat().st_size,
        "selected_group_count": selected_group_count,
        "variant_count": selected_variant_count,
        "temporary_storage_spool_bytes": storage_spool_bytes,
        "variant_order": "lexicographic suffix",
        "source_sha256_verified": source_digest,
        "elapsed_seconds": time.perf_counter() - started,
    }


def verify_materialized_minnesota_panel(
    panel_pickle_path: str | Path,
    panel_definition: dict[str, Any],
) -> dict[str, Any]:
    """Read and verify an extracted panel artifact using standard torch pickle."""
    panel_path = _assert_external_path(panel_pickle_path)
    mn_definition = panel_definition.get("minnesota", {})
    expected_groups = mn_definition.get("reactions", [])
    with panel_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or len(data) != len(expected_groups):
        raise ValueError("Materialized Minnesota panel has an unexpected group count.")
    for expected in expected_groups:
        group_key = expected["source_group_key"]
        variants = data.get(group_key)
        if not isinstance(variants, list):
            raise TypeError(f"Materialized panel lacks group {group_key}.")
        by_suffix = {_variant_suffix(row): row for row in variants}
        if sorted(by_suffix) != expected["variant_suffixes"]:
            raise ValueError(f"Materialized panel suffix mismatch for group {group_key}.")
        for row in by_suffix.values():
            if row.get("Database") != expected["database"] or row.get("ReactionID") != expected["reaction_id"]:
                raise ValueError(f"Materialized panel identity mismatch for group {group_key}.")
    return {
        "path": str(panel_path),
        "sha256": _sha256_file(panel_path),
        "file_bytes": panel_path.stat().st_size,
        "verified_group_count": len(data),
        "verified_variant_count": sum(len(rows) for rows in data.values()),
        "verified_suffixes": True,
        "verified_source_identities": True,
    }


def build_minnesota_group_store(
    grouped_pickle_path: str | Path,
    canonical_manifest_path: str | Path,
    store_dir: str | Path,
    *,
    minimum_free_memory_bytes: int = 1024**3,
) -> dict[str, Any]:
    """Extract all 268 canonical reaction groups one full group at a time."""
    import gc

    source_path = Path(grouped_pickle_path).resolve()
    canonical_path = Path(canonical_manifest_path).resolve()
    store_dir = _assert_external_path(store_dir)
    canonical = _read_json(canonical_path)
    if canonical.get("schema") != "canonical-minnesota-view-v1":
        raise ValueError("Unsupported Minnesota canonical manifest schema.")
    groups = canonical.get("groups")
    if not isinstance(groups, list) or len(groups) != 268:
        raise ValueError("Production Minnesota group store requires exactly 268 groups.")
    source = canonical.get("source", {})
    expected_size = source.get("size_bytes")
    expected_hash = source.get("sha256")
    if source_path.stat().st_size != expected_size:
        raise ValueError("Grouped Minnesota pickle size does not match the canonical manifest.")
    if store_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Minnesota group store: {store_dir}")
    partial_dir = store_dir.with_name(store_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"Refusing to overwrite partial Minnesota group store: {partial_dir}")
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        available = None
    if available is not None and available < expected_size + minimum_free_memory_bytes:
        raise MemoryError(
            "Insufficient available RAM for bounded source pass: "
            f"available={available}, required_at_least={expected_size + minimum_free_memory_bytes}."
        )

    partial_dir.mkdir(parents=True, exist_ok=False)
    group_dir = partial_dir / "groups"
    group_dir.mkdir()
    spool_path = partial_dir / ".storage_spool.partial"
    started = time.perf_counter()
    group_records: list[dict[str, Any]] = []
    output_bytes = 0
    try:
        with _StorageSpool(spool_path) as storage_spool:
            with source_path.open("rb") as raw_stream:
                reader = _HashingReader(
                    raw_stream,
                    expected_size,
                    low_memory_bytes=max(minimum_free_memory_bytes, 1024**3),
                )
                source_data = _LazyTensorUnpickler(reader, storage_spool).load()
                if reader.bytes_read < expected_size:
                    reader._track(raw_stream.read())
                source_digest = reader.hexdigest()
                if source_digest != expected_hash:
                    raise ValueError(
                        f"Grouped source hash mismatch: expected {expected_hash}, got {source_digest}."
                    )
            if not isinstance(source_data, dict):
                raise TypeError("Grouped Minnesota source must contain a dictionary.")
            ordered_groups = sorted(groups, key=lambda row: row["output_index"])
            for index, group in enumerate(ordered_groups, start=1):
                group_key = group["group_key"]
                if group_key not in source_data:
                    raise ValueError(f"Minnesota source is missing group {group_key}.")
                source_variants = source_data[group_key]
                if not isinstance(source_variants, list):
                    raise TypeError(f"Minnesota group {group_key} is not a variant list.")
                by_suffix: dict[str, dict[str, Any]] = {}
                for reaction in source_variants:
                    suffix = _variant_suffix(reaction)
                    if suffix in by_suffix:
                        raise ValueError(f"Duplicate {suffix!r} variant in source group {group_key}.")
                    if reaction.get("Database") != group["database"] or reaction.get("ReactionID") != group["reaction_id"]:
                        raise ValueError(f"Minnesota source identity mismatch in group {group_key}.")
                    by_suffix[suffix] = reaction
                suffixes = sorted(group["available_suffixes"])
                if sorted(by_suffix) != suffixes or len(by_suffix) != group["variant_count"]:
                    raise ValueError(f"Minnesota group {group_key} has incomplete augmentation variants.")

                group_data = [
                    _materialize_selected_tensors(by_suffix[suffix]) for suffix in suffixes
                ]
                variants = []
                for suffix, reaction in zip(suffixes, group_data, strict=True):
                    grid = reaction.get("Grid")
                    if getattr(grid, "ndim", None) != 2 or grid.shape[1] != 9:
                        raise ValueError(f'{group["database"]} group {group_key}/{suffix} has invalid Grid.')
                    variants.append(
                        {
                            "suffix": suffix,
                            "point_count": int(grid.shape[0]),
                            "grid_dtype": str(grid.dtype),
                        }
                    )
                filename = (
                    f"group_{group['output_index']:03d}_"
                    f"key_{group_key:03d}_{group['database']}_{group['reaction_id']}.pickle"
                )
                final_path = group_dir / filename
                temp_path = final_path.with_name(final_path.name + ".partial")
                with temp_path.open("xb") as raw_output:
                    writer = _HashingWriter(raw_output)
                    pickle.dump(group_data, writer, protocol=pickle.HIGHEST_PROTOCOL)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(temp_path, final_path)
                output_bytes += writer.bytes_written
                group_records.append(
                    {
                        "database": group["database"],
                        "reaction_id": group["reaction_id"],
                        "source_group_key": group_key,
                        "canonical_output_index": group["output_index"],
                        "canonical_variant_suffix": group["selected_suffix"],
                        "canonical_point_count": group["point_count"],
                        "file": f"groups/{filename}",
                        "file_sha256": writer.hexdigest(),
                        "file_bytes": writer.bytes_written,
                        "variant_count": len(variants),
                        "variant_suffixes": suffixes,
                        "variants": variants,
                    }
                )
                source_data.pop(group_key)
                del source_variants, group_data, by_suffix
                storage_spool.release_materialized()
                gc.collect()
                if index % 20 == 0 or index == len(ordered_groups):
                    try:
                        import psutil

                        available = psutil.virtual_memory().available
                    except ImportError:
                        available = None
                    print(
                        json.dumps(
                            {
                                "event": "mn_group_store_progress",
                                "groups_written": index,
                                "group_count": len(ordered_groups),
                                "output_bytes": output_bytes,
                                "available_memory_bytes": available,
                            }
                        ),
                        flush=True,
                    )
                    if available is not None and available < minimum_free_memory_bytes:
                        raise MemoryError("Stopping group-store build below its RAM safety floor.")
            storage_spool.close()
            storage_spool_bytes = storage_spool.bytes_written
            del source_data
            gc.collect()
        spool_path.unlink()
        if len(group_records) != 268 or sum(row["variant_count"] for row in group_records) != 2144:
            raise ValueError("Minnesota group store did not contain all 268×8 variants.")
        manifest = {
            "schema": MN_GROUP_STORE_PROTOCOL,
            "source_protocol": canonical.get("schema"),
            "canonical_manifest_path": str(canonical_path),
            "canonical_manifest_sha256": _sha256_file(canonical_path),
            "canonical_view_sha256": canonical.get("output_sha256"),
            "source_pickle_path": str(source_path),
            "source_pickle_bytes": expected_size,
            "source_pickle_sha256_verified": source_digest,
            "group_count": len(group_records),
            "variant_count": sum(row["variant_count"] for row in group_records),
            "variant_policy": "all source variants retained and sorted by suffix per group",
            "group_file_bytes_total": output_bytes,
            "temporary_storage_spool_bytes": storage_spool_bytes,
            "elapsed_seconds": time.perf_counter() - started,
            "groups": group_records,
        }
        manifest_path = partial_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(partial_dir, store_dir)
    except BaseException as error:
        # Leave an explicit .partial directory for inspection/restart; never
        # discard already-written records without an auditable user decision.
        print(
            json.dumps(
                {
                    "event": "mn_group_store_incomplete",
                    "partial_path": str(partial_dir),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )
        raise

    final_manifest = store_dir / "manifest.json"
    return {
        "path": str(store_dir),
        "manifest_path": str(final_manifest),
        "manifest_sha256": _sha256_file(final_manifest),
        "group_count": len(group_records),
        "variant_count": sum(row["variant_count"] for row in group_records),
        "group_file_bytes_total": output_bytes,
        "temporary_storage_spool_bytes": storage_spool_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256_verified": source_digest,
    }


class MinnesotaGroupStore:
    """Hash-verifying, small-LRU reader for the external 268-group store."""

    def __init__(self, manifest_path: str | Path, *, cache_groups: int = 1):
        if cache_groups < 0:
            raise ValueError("cache_groups must be nonnegative.")
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = _read_json(self.manifest_path)
        if self.manifest.get("schema") != MN_GROUP_STORE_PROTOCOL:
            raise ValueError("Minnesota group store uses an incompatible protocol.")
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        groups = self.manifest.get("groups")
        if not isinstance(groups, list) or len(groups) != self.manifest.get("group_count"):
            raise ValueError("Minnesota group store manifest has an invalid group list.")
        self.cache_groups = cache_groups
        self._groups_by_key: dict[int, dict[str, Any]] = {}
        self._groups_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
        self._verified: set[tuple[str, int, int]] = set()
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        for row in groups:
            group_key = row.get("source_group_key")
            identity = (row.get("database"), row.get("reaction_id"))
            filename = row.get("file")
            if (
                not isinstance(group_key, int)
                or not isinstance(identity[0], str)
                or not isinstance(identity[1], int)
                or not isinstance(filename, str)
                or Path(filename).name != filename.split("/")[-1]
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
            ):
                raise ValueError("Minnesota group store contains invalid identity or file path.")
            if group_key in self._groups_by_key or identity in self._groups_by_identity:
                raise ValueError("Minnesota group store contains duplicate identities.")
            self._groups_by_key[group_key] = row
            self._groups_by_identity[identity] = row

    def _resolve(self, identity: int | tuple[str, int] | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(identity, int):
            row = self._groups_by_key.get(identity)
        elif isinstance(identity, tuple) and len(identity) == 2:
            row = self._groups_by_identity.get((identity[0], identity[1]))
        elif isinstance(identity, Mapping):
            if "source_group_key" in identity:
                row = self._groups_by_key.get(identity["source_group_key"])
            else:
                row = self._groups_by_identity.get(
                    (identity.get("database"), identity.get("reaction_id"))
                )
        else:
            raise TypeError("Group identity must be a source key, (database, reaction_id), or mapping.")
        if row is None:
            raise KeyError(f"Minnesota group identity is not in this store: {identity!r}.")
        return row

    def load_group(
        self,
        identity: int | tuple[str, int] | Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Return all variants for one group after file-hash verification."""
        row = self._resolve(identity)
        group_key = row["source_group_key"]
        cached = self._cache.get(group_key)
        if cached is not None:
            self._cache.move_to_end(group_key)
            return cached
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Minnesota group file path escapes its store.")
        path = self.root / relative
        stat = path.stat()
        fingerprint = (str(path), stat.st_size, stat.st_mtime_ns)
        if fingerprint not in self._verified:
            if stat.st_size != row["file_bytes"] or _sha256_file(path) != row["file_sha256"]:
                raise ValueError(f"Minnesota group file hash mismatch for group {group_key}.")
            self._verified.add(fingerprint)
        with path.open("rb") as stream:
            variants = pickle.load(stream)
        if not isinstance(variants, list) or len(variants) != row["variant_count"]:
            raise ValueError(f"Minnesota group {group_key} has an invalid stored variant list.")
        by_suffix = {_variant_suffix(reaction): reaction for reaction in variants}
        if len(by_suffix) != len(variants) or sorted(by_suffix) != row["variant_suffixes"]:
            raise ValueError(f"Minnesota group {group_key} has an invalid stored suffix set.")
        for reaction in variants:
            if reaction.get("Database") != row["database"] or reaction.get("ReactionID") != row["reaction_id"]:
                raise ValueError(f"Minnesota group {group_key} has a mismatched reaction identity.")
        variants = [by_suffix[suffix] for suffix in row["variant_suffixes"]]
        if self.cache_groups:
            self._cache[group_key] = variants
            self._cache.move_to_end(group_key)
            while len(self._cache) > self.cache_groups:
                self._cache.popitem(last=False)
        return variants

    def load_variant(
        self,
        identity: int | tuple[str, int] | Mapping[str, Any],
        variant_suffix: str,
    ) -> dict[str, Any]:
        """Load one explicitly identified augmentation variant from a group."""
        row = self._resolve(identity)
        if variant_suffix not in row["variant_suffixes"]:
            raise KeyError(f"Unknown variant suffix {variant_suffix!r} for {identity!r}.")
        group = self.load_group(identity)
        return group[row["variant_suffixes"].index(variant_suffix)]


def verify_minnesota_group_store(
    manifest_path: str | Path,
    *,
    canonical_manifest_path: str | Path | None = None,
    expected_group_count: int = 268,
) -> dict[str, Any]:
    """Hash-check and load every group independently, one group at a time."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("group_count") != expected_group_count:
        raise ValueError(
            f"Expected {expected_group_count} Minnesota groups, found {manifest.get('group_count')}."
        )
    if canonical_manifest_path is not None:
        canonical_path = Path(canonical_manifest_path).resolve()
        if _sha256_file(canonical_path) != manifest.get("canonical_manifest_sha256"):
            raise ValueError("Group store was built from a different canonical manifest.")
        canonical = _read_json(canonical_path)
        if canonical.get("source", {}).get("sha256") != manifest.get("source_pickle_sha256_verified"):
            raise ValueError("Group store source hash does not match canonical source identity.")

    store = MinnesotaGroupStore(manifest_path, cache_groups=1)
    verified_variants = 0
    verified_grid_points = 0
    for index, row in enumerate(manifest["groups"], start=1):
        variants = store.load_group(row["source_group_key"])
        observed_variant_rows = []
        for reaction in variants:
            suffix = _variant_suffix(reaction)
            grid = reaction.get("Grid")
            if getattr(grid, "ndim", None) != 2 or grid.shape[1] != 9:
                raise ValueError(f"{row['database']} group {row['source_group_key']}/{suffix} has invalid Grid.")
            observed_variant_rows.append(
                {
                    "suffix": suffix,
                    "point_count": int(grid.shape[0]),
                    "grid_dtype": str(grid.dtype),
                }
            )
        observed_variant_rows.sort(key=lambda item: item["suffix"])
        if observed_variant_rows != row["variants"]:
            raise ValueError(
                f"{row['database']} group {row['source_group_key']} differs from group metadata."
            )
        if row["canonical_variant_suffix"] not in row["variant_suffixes"]:
            raise ValueError(f"{row['database']} group {row['source_group_key']} lost its canonical suffix.")
        verified_variants += len(variants)
        verified_grid_points += sum(item["point_count"] for item in observed_variant_rows)
        if index % 20 == 0 or index == len(manifest["groups"]):
            print(
                json.dumps(
                    {
                        "event": "mn_group_store_verification_progress",
                        "groups_verified": index,
                        "group_count": len(manifest["groups"]),
                        "variants_verified": verified_variants,
                    }
                ),
                flush=True,
            )
    if verified_variants != manifest.get("variant_count"):
        raise ValueError("Minnesota store total variant count does not match its manifest.")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_group_count": len(manifest["groups"]),
        "verified_variant_count": verified_variants,
        "verified_variant_grid_points": verified_grid_points,
        "verified_group_file_bytes": manifest["group_file_bytes_total"],
        "source_pickle_sha256_verified": manifest["source_pickle_sha256_verified"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
    }


def _read_central_atom_numbers(record_path: Path) -> list[int]:
    import h5py

    with h5py.File(record_path, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
    molecule = metadata.get("molecule", {})
    charges = molecule.get("atom_charges")
    if not isinstance(charges, list) or not charges:
        raise ValueError(f"Central record has no atomic-number metadata: {record_path.name}.")
    numbers = [round(float(charge)) for charge in charges]
    if any(number < 1 for number in numbers):
        raise ValueError(f"Central record has invalid atomic numbers: {record_path.name}.")
    return numbers


def build_ao_factor_cache(
    central_data_dir: str | Path,
    cache_dir: str | Path,
    *,
    chunk_size: int = 2048,
    systems: tuple[str, ...] = MRKS_PANEL_SYSTEMS,
) -> dict[str, Any]:
    """Generate one disposable float32 AO cache at a time for panel systems."""
    import h5py
    import numpy as np
    try:
        from .lap_operator import OPERATOR_PROTOCOL
        from .lap_operator_data import (
            build_molecule_from_metadata,
            iter_ao_factor_chunks,
            load_central_operator_record,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from lap_operator import OPERATOR_PROTOCOL
        from lap_operator_data import (
            build_molecule_from_metadata,
            iter_ao_factor_chunks,
            load_central_operator_record,
        )

    central_data_dir = Path(central_data_dir).resolve()
    cache_dir = _assert_external_path(cache_dir)
    if chunk_size <= 0:
        raise ValueError("AO cache chunk_size must be positive.")
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite AO cache directory: {cache_dir}")
    central_manifest_path = central_data_dir / "manifest.json"
    central_manifest = _read_json(central_manifest_path)
    if central_manifest.get("protocol") != CENTRAL_DATA_PROTOCOL or central_manifest.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError("Central corpus protocol does not match h-free AO operator contract.")
    central_records = central_manifest.get("records", [])
    if len(central_records) != 90:
        raise ValueError("AO survey cache requires the verified 90-system central corpus.")
    records_by_name = {item["system_name"]: item for item in central_records}
    if len(records_by_name) != 90 or set(systems) - records_by_name.keys():
        raise ValueError("AO survey panel names do not resolve uniquely in all-90 central data.")

    cache_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache_records: list[dict[str, Any]] = []
    for index, name in enumerate(systems, start=1):
        item = records_by_name[name]
        source_path = central_data_dir / item["file"]
        if _sha256_file(source_path) != item["file_sha256"]:
            raise ValueError(f"{name}: central source file hash mismatch.")
        record = load_central_operator_record(source_path)
        if record.metadata["record_sha256"] != item["record_sha256"]:
            raise ValueError(f"{name}: central logical record hash mismatch.")
        molecule = build_molecule_from_metadata(record.metadata)
        point_count = len(record.coords64)
        nao = int(record.dmks.shape[0])
        cache_path = cache_dir / f"{name}_ao_factors.h5"
        with h5py.File(cache_path, "x") as handle:
            handle.attrs["protocol"] = AO_CACHE_PROTOCOL
            handle.attrs["operator_protocol"] = OPERATOR_PROTOCOL
            handle.attrs["system_name"] = name
            handle.attrs["source_record_sha256"] = record.metadata["record_sha256"]
            handle.attrs["source_file_sha256"] = item["file_sha256"]
            handle.attrs["dtype"] = "float32"
            phi_ds = handle.create_dataset(
                "phi",
                shape=(point_count, nao),
                dtype="<f4",
                chunks=(min(chunk_size, point_count), nao),
                compression="lzf",
                shuffle=True,
            )
            grad_ds = handle.create_dataset(
                "grad_phi",
                shape=(point_count, 3, nao),
                dtype="<f4",
                chunks=(min(chunk_size, point_count), 3, nao),
                compression="lzf",
                shuffle=True,
            )
            lap_ds = handle.create_dataset(
                "lap_phi",
                shape=(point_count, nao),
                dtype="<f4",
                chunks=(min(chunk_size, point_count), nao),
                compression="lzf",
                shuffle=True,
            )
            for row_slice, phi, grad_phi, lap_phi in iter_ao_factor_chunks(
                record, molecule, chunk_size
            ):
                phi_ds[row_slice] = np.asarray(phi, dtype=np.float32)
                grad_ds[row_slice] = np.asarray(grad_phi, dtype=np.float32)
                lap_ds[row_slice] = np.asarray(lap_phi, dtype=np.float32)
            handle.flush()
        atomic_numbers = _read_central_atom_numbers(source_path)
        cache_records.append(
            {
                "system_name": name,
                "file": cache_path.name,
                "point_count": point_count,
                "nao": nao,
                "atomic_numbers": atomic_numbers,
                "max_atomic_number": max(atomic_numbers),
                "source_record_sha256": record.metadata["record_sha256"],
                "source_file_sha256": item["file_sha256"],
                "cache_file_sha256": _sha256_file(cache_path),
                "cache_file_bytes": cache_path.stat().st_size,
                "ao_factor_raw_bytes_float32": point_count * nao * 5 * 4,
            }
        )
        print(
            json.dumps(
                {
                    "event": "ao_cache_system_complete",
                    "index": index,
                    "system_name": name,
                    "point_count": point_count,
                    "nao": nao,
                    "cache_file_bytes": cache_records[-1]["cache_file_bytes"],
                }
            ),
            flush=True,
        )
        del record, molecule

    cache_manifest = {
        "protocol": AO_CACHE_PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "cache_dtype": "float32",
        "central_data_manifest": str(central_manifest_path),
        "central_data_manifest_sha256": _sha256_file(central_manifest_path),
        "source_protocol": CENTRAL_DATA_PROTOCOL,
        "panel_protocol": PANEL_PROTOCOL,
        "chunk_size_used_for_generation": chunk_size,
        "records": cache_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(cache_manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {
        **cache_manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "total_cache_file_bytes": sum(row["cache_file_bytes"] for row in cache_records),
        "total_ao_factor_raw_bytes_float32": sum(
            row["ao_factor_raw_bytes_float32"] for row in cache_records
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mn-canonical-manifest", type=Path, required=True)
    parser.add_argument("--central-data-dir", type=Path, required=True)
    parser.add_argument("--grouped-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--skip-mn-materialization", action="store_true")
    parser.add_argument("--skip-ao-cache", action="store_true")
    args = parser.parse_args(argv)

    output_dir = _assert_external_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite survey output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    definition = build_panel_definition(args.mn_canonical_manifest, args.central_data_dir)
    definition_path = output_dir / "panel_definition.json"
    with definition_path.open("x", encoding="utf-8") as stream:
        json.dump(definition, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    final_manifest: dict[str, Any] = {
        "schema": PANEL_PROTOCOL,
        "definition_path": str(definition_path),
        "definition_sha256": _sha256_file(definition_path),
        "minnesota": {"materialized": False},
        "mrks": {"ao_cache_built": False},
    }
    if not args.skip_mn_materialization:
        final_manifest["minnesota"] = {
            "materialized": True,
            **materialize_minnesota_panel(
                args.grouped_pickle,
                definition,
                output_dir / "mn_panel_27x8.pickle",
            ),
        }
    if not args.skip_ao_cache:
        final_manifest["mrks"] = {
            "ao_cache_built": True,
            **build_ao_factor_cache(
                args.central_data_dir,
                output_dir / "mrks_15system_ao_cache",
                chunk_size=args.chunk_size,
            ),
        }
    manifest_path = output_dir / "panel_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(final_manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"panel_manifest": str(manifest_path), **final_manifest}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
