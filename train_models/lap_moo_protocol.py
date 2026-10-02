"""Reproducible one-stage, three-objective Lap MOO protocol primitives.

This module owns only the new MOO protocol.  Historical S5 schedules and
checkpoints remain in their existing modules and are not imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "lap-moo-one-stage-v2"
LEGACY_READ_ONLY_PROTOCOL_VERSION = "lap-moo-one-stage-v1"
# Sampling identity is frozen independently from training/checkpoint metadata.
# The v2 AO-cache chunk field changes checkpoint provenance, not which samples
# each update consumes, so new protocol metadata must keep the v1 seed domain.
SAMPLING_PROTOCOL_VERSION = "lap-moo-one-stage-v1"
TASK_NAMES = ("chem", "exc", "op")
OPTIMIZER_FAMILIES = ("radamw", "adamw", "muon_adamw")
METHOD_IDS = ("fixed", "imtl_g", "cagrad", "nash_mtl")
_FORBIDDEN_OPERATOR_KEYS = {
    "h",
    "h_bohr",
    "stencil",
    "stencil_order",
    "stencil_version",
    "derivative_order",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_suffix(reaction: Mapping[str, Any]) -> str:
    paths = reaction.get("component_paths", ())
    if not paths:
        raise ValueError("Every Minnesota augmentation needs component_paths.")
    suffixes = set()
    for component_path in paths:
        stem = Path(str(component_path)).stem
        suffixes.add(stem.split("__", 1)[1] if "__" in stem else "default")
    if len(suffixes) != 1:
        raise ValueError(f"Reaction components have inconsistent variants: {sorted(suffixes)}")
    return next(iter(suffixes))


def _derive_seed(seed: int, label: str) -> int:
    material = f"{SAMPLING_PROTOCOL_VERSION}:{int(seed)}:{label}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _reaction_catalog(grouped_reactions: Mapping[Any, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if not isinstance(grouped_reactions, Mapping) or not grouped_reactions:
        raise ValueError("Minnesota grouped reactions must be a nonempty mapping.")
    groups: dict[tuple[str, int], set[str]] = {}
    for variants in grouped_reactions.values():
        if not variants:
            raise ValueError("Minnesota base-reaction groups cannot be empty.")
        identities = set()
        suffixes = set()
        for reaction in variants:
            try:
                identity = (str(reaction["Database"]), int(reaction["ReactionID"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Minnesota variants need Database and ReactionID.") from exc
            identities.add(identity)
            suffixes.add(_variant_suffix(reaction))
        if len(identities) != 1:
            raise ValueError("An augmentation group contains multiple reaction identities.")
        identity = next(iter(identities))
        if identity in groups:
            raise ValueError(f"Duplicate Minnesota reaction identity: {identity}.")
        if len(suffixes) != len(variants):
            raise ValueError(f"Duplicate augmentation suffix in Minnesota group {identity}.")
        groups[identity] = suffixes
    return [
        {"database": database, "reaction_id": reaction_id, "variants": sorted(variants)}
        for (database, reaction_id), variants in sorted(groups.items())
    ]


def _cycle_permutation(items: Sequence[Any], seed: int, label: str, cycle: int) -> list[Any]:
    ordered = list(items)
    random.Random(_derive_seed(seed, f"{label}:cycle:{cycle}")).shuffle(ordered)
    return ordered


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def build_sampling_manifest(
    grouped_reactions: Mapping[Any, Sequence[Mapping[str, Any]]],
    system_names: Sequence[str],
    *,
    updates: int,
    seed: int,
    source_hashes: Mapping[str, str],
    world_size: int = 1,
) -> dict[str, Any]:
    """Materialize paired reaction/augmentation/system IDs for every update.

    Each stream independently shuffles a full permutation every cycle.  Each
    reaction group appears once per reaction cycle, and exactly one available
    augmentation is chosen for it.  The mRKS stream cycles over systems with
    equal system probability, independent of grid size.
    """
    if updates <= 0 or world_size <= 0:
        raise ValueError("updates and world_size must be positive.")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    catalog = _reaction_catalog(grouped_reactions)
    systems = sorted({str(name) for name in system_names})
    if not systems or len(systems) != len(system_names):
        raise ValueError("mRKS system names must be nonempty and unique.")
    clean_hashes = dict(sorted((str(k), str(v).lower()) for k, v in source_hashes.items()))
    if not clean_hashes or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in clean_hashes.values()
    ):
        raise ValueError("source_hashes must contain named SHA-256 hex digests.")

    reaction_ids = [(item["database"], item["reaction_id"]) for item in catalog]
    reaction_by_id = {(item["database"], item["reaction_id"]): item for item in catalog}
    entries = []
    for update in range(updates):
        per_rank = []
        for rank in range(world_size):
            reaction_cycle, reaction_offset = divmod(update, len(reaction_ids))
            system_cycle, system_offset = divmod(update, len(systems))
            ordered_reactions = _cycle_permutation(
                reaction_ids,
                seed,
                f"reaction-order:rank:{rank}",
                reaction_cycle,
            )
            ordered_systems = _cycle_permutation(
                systems, seed, f"system-order:rank:{rank}", system_cycle
            )
            database, reaction_id = ordered_reactions[reaction_offset]
            variants = reaction_by_id[(database, reaction_id)]["variants"]
            variant_rng = random.Random(
                _derive_seed(
                    seed,
                    f"variant:rank:{rank}:{reaction_cycle}:{database}:{reaction_id}",
                )
            )
            per_rank.append(
                {
                    "rank": rank,
                    "reaction": {"database": database, "reaction_id": reaction_id},
                    "variant_suffix": variants[variant_rng.randrange(len(variants))],
                    "mrks_system": ordered_systems[system_offset],
                }
            )
        entries.append({"update": update, "per_rank": per_rank})

    manifest = {
        "schema": "lap-moo-sampling-manifest-v1",
        "seed": seed,
        "world_size": world_size,
        "independent_streams": [
            {
                "rank": rank,
                "reaction_order_seed": _derive_seed(seed, f"reaction-order:rank:{rank}"),
                "variant_seed": _derive_seed(seed, f"variant:rank:{rank}"),
                "system_order_seed": _derive_seed(seed, f"system-order:rank:{rank}"),
            }
            for rank in range(world_size)
        ],
        "reaction_cycle_length": len(reaction_ids),
        "mrks_system_cycle_length": len(systems),
        "reaction_catalog": catalog,
        "mrks_systems": systems,
        "source_hashes": clean_hashes,
        "entries": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def build_sampling_manifest_from_catalog(
    reaction_catalog: Sequence[Mapping[str, Any]],
    system_names: Sequence[str],
    *,
    updates: int,
    seed: int,
    source_hashes: Mapping[str, str],
    world_size: int = 1,
) -> dict[str, Any]:
    """Build the same persisted stream from a small metadata inventory.

    This avoids deserializing the large Minnesota tensor corpus just to plan
    reaction identities and augmentation suffixes.
    """
    normalized = []
    for group in reaction_catalog:
        try:
            database = str(group["database"])
            reaction_id = int(group["reaction_id"])
            variants = sorted({str(value) for value in group["variants"]})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Catalog rows need database, reaction_id, and variants.") from exc
        if not database or not variants:
            raise ValueError("Catalog reaction identity and augmentation list cannot be empty.")
        normalized.append(
            {
                "database": database,
                "reaction_id": reaction_id,
                "variants": variants,
            }
        )
    if len({(row["database"], row["reaction_id"]) for row in normalized}) != len(normalized):
        raise ValueError("Reaction catalog contains duplicate identities.")
    grouped = {}
    for index, row in enumerate(normalized):
        grouped[index] = [
            {
                "Database": row["database"],
                "ReactionID": row["reaction_id"],
                "component_paths": [
                    f"source{index}.npz"
                    if suffix == "default"
                    else f"source{index}__{suffix}.npz"
                ],
            }
            for suffix in row["variants"]
        ]
    return build_sampling_manifest(
        grouped,
        system_names,
        updates=updates,
        seed=seed,
        source_hashes=source_hashes,
        world_size=world_size,
    )


def validate_sampling_manifest(manifest: Mapping[str, Any]) -> str:
    if manifest.get("schema") != "lap-moo-sampling-manifest-v1":
        raise ValueError("Incompatible sampling manifest schema.")
    actual = canonical_sha256(_manifest_payload(manifest))
    if manifest.get("manifest_sha256") != actual:
        raise ValueError("Sampling manifest hash mismatch.")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Sampling manifest has no update entries.")
    catalog = {
        (group["database"], group["reaction_id"]): set(group["variants"])
        for group in manifest.get("reaction_catalog", ())
    }
    if len(catalog) != manifest.get("reaction_cycle_length"):
        raise ValueError("Sampling manifest reaction catalog is invalid.")
    for index, entry in enumerate(entries):
        if entry.get("update") != index:
            raise ValueError("Sampling manifest update IDs are invalid.")
        per_rank = entry.get("per_rank")
        if not isinstance(per_rank, list) or len(per_rank) != manifest.get("world_size"):
            raise ValueError("Sampling manifest rank streams are incomplete.")
        for rank, item in enumerate(per_rank):
            identity = item.get("reaction", {})
            if (
                item.get("rank") != rank
                or not item.get("variant_suffix")
                or (identity.get("database"), identity.get("reaction_id")) not in catalog
                or item.get("variant_suffix")
                not in catalog.get((identity.get("database"), identity.get("reaction_id")), set())
            ):
                raise ValueError("Sampling manifest reaction identity/variant is invalid.")
            if item.get("mrks_system") not in manifest.get("mrks_systems", ()):
                raise ValueError("Sampling manifest contains an unknown mRKS system.")
    return actual


def sampling_entry(manifest: Mapping[str, Any], update: int, rank: int = 0) -> dict[str, Any]:
    validate_sampling_manifest(manifest)
    if not 0 <= update < len(manifest["entries"]):
        raise IndexError("update is outside the sampling manifest.")
    if not 0 <= rank < manifest["world_size"]:
        raise IndexError("rank is outside the sampling manifest.")
    return dict(manifest["entries"][update]["per_rank"][rank])


class SamplingStream:
    """Validated O(1) cursor over an immutable, hash-bound update manifest."""

    def __init__(self, manifest: Mapping[str, Any]):
        self.manifest = manifest
        self.manifest_sha256 = validate_sampling_manifest(manifest)

    def entry(self, update: int, rank: int = 0) -> dict[str, Any]:
        if not 0 <= update < len(self.manifest["entries"]):
            raise IndexError("update is outside the sampling manifest.")
        if not 0 <= rank < self.manifest["world_size"]:
            raise IndexError("rank is outside the sampling manifest.")
        return dict(self.manifest["entries"][update]["per_rank"][rank])


def write_sampling_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    digest = validate_sampling_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return digest


def read_sampling_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_sampling_manifest(manifest)
    return manifest


def validate_cursor(manifest: Mapping[str, Any], cursor: Mapping[str, Any]) -> int:
    digest = validate_sampling_manifest(manifest)
    if cursor.get("sampling_manifest_sha256") != digest:
        raise ValueError("Resume cursor belongs to a different sampling manifest.")
    next_update = cursor.get("next_update")
    if type(next_update) is not int or not 0 <= next_update <= len(manifest["entries"]):
        raise ValueError("Resume cursor is outside the sampling manifest.")
    return next_update


def make_protocol_metadata(
    *,
    architecture: str,
    method: str,
    method_hyperparameters: Mapping[str, Any],
    fixed_scalarization: Mapping[str, Any] | None,
    optimizer: Mapping[str, Any],
    lr_schedule: Mapping[str, Any],
    predopt_checkpoint_sha256: str,
    sampling_manifest_sha256: str,
    minnesota_data_sha256: str,
    operator_corpus_manifest_sha256: str,
    ao_cache_manifest_sha256: str,
    reaction_dispersions_sha256: str,
    mrks_dispersions_sha256: str,
    random_seed: int,
    dtype: str,
    grid_chunk_size: int,
    ao_cache_chunk_size: int,
) -> dict[str, Any]:
    """Build the explicit, h-free metadata contract for a new MOO run."""
    required_hashes = (
        predopt_checkpoint_sha256,
        sampling_manifest_sha256,
        minnesota_data_sha256,
        operator_corpus_manifest_sha256,
        ao_cache_manifest_sha256,
        reaction_dispersions_sha256,
        mrks_dispersions_sha256,
    )
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower())
        for value in required_hashes
    ):
        raise ValueError("All source/checkpoint identities must be SHA-256 hex digests.")
    if not architecture or not method:
        raise ValueError("architecture and method are required.")
    if type(grid_chunk_size) is not int or grid_chunk_size <= 0:
        raise ValueError("grid_chunk_size must be a positive integer.")
    if type(ao_cache_chunk_size) is not int or ao_cache_chunk_size <= 0:
        raise ValueError("ao_cache_chunk_size must be a positive integer.")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": architecture,
        "operator_protocol": "lap-weakform-ao-v1",
        "objective_definitions": {
            "chem": "existing Minnesota grouped reaction objective; stoichiometry, database weights, D3 treatment, kcal/mol retained",
            "exc": "existing legacy mRKS integrated Exc target and historical one-addition dispersion treatment; kcal/mol RMSE",
            "op": "h-free variational AO XC operator loss: symmetric-orthogonalized squared Hilbert-Schmidt error divided by nAO",
        },
        "task_order": list(TASK_NAMES),
        "method": method,
        "method_hyperparameters": dict(method_hyperparameters),
        "fixed_scalarization": None if fixed_scalarization is None else dict(fixed_scalarization),
        "optimizer": dict(optimizer),
        "lr_schedule": dict(lr_schedule),
        "predopt_checkpoint_sha256": predopt_checkpoint_sha256.lower(),
        "sampling_manifest_sha256": sampling_manifest_sha256.lower(),
        "minnesota_data_sha256": minnesota_data_sha256.lower(),
        "operator_corpus_manifest_sha256": operator_corpus_manifest_sha256.lower(),
        "ao_cache_manifest_sha256": ao_cache_manifest_sha256.lower(),
        "reaction_dispersions_sha256": reaction_dispersions_sha256.lower(),
        "mrks_dispersions_sha256": mrks_dispersions_sha256.lower(),
        "random_seed": int(random_seed),
        "dtype": str(dtype),
        "grid_chunk_size": int(grid_chunk_size),
        "ao_cache_chunk_size": int(ao_cache_chunk_size),
        "one_stage_main_training": True,
    }


def _validate_no_legacy_controls(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "omega" in lowered or lowered.startswith("s5_") or "phase_schedule" in lowered:
                raise ValueError("Legacy objective weighting or phase schedule detected.")
            if "epoch" in lowered and any(
                token in lowered for token in ("weight", "scalar", "lambda", "coefficient")
            ):
                raise ValueError("Epoch-dependent task weights are forbidden in one-stage MOO metadata.")
            if lowered in _FORBIDDEN_OPERATOR_KEYS:
                raise ValueError("MOO metadata must identify the h-free AO operator path.")
            _validate_no_legacy_controls(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_no_legacy_controls(nested)


def _positive_finite_number(value: Any, label: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0 if allow_zero else float(value) <= 0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"Lap MOO {label} must be finite and {qualifier}.")


def validate_protocol_metadata(
    metadata: Mapping[str, Any], *, allow_v1_read_only: bool = False
) -> None:
    version = metadata.get("protocol_version")
    legacy_read_only = allow_v1_read_only and version == LEGACY_READ_ONLY_PROTOCOL_VERSION
    if version != PROTOCOL_VERSION and not legacy_read_only:
        raise ValueError("Incompatible one-stage Lap MOO protocol version.")
    if legacy_read_only and "ao_cache_chunk_size" in metadata:
        raise ValueError("Historical v1 MOO metadata cannot be retrofitted with an AO-cache chunk size.")
    if metadata.get("operator_protocol") != "lap-weakform-ao-v1":
        raise ValueError("Incompatible operator protocol in Lap MOO metadata.")
    if metadata.get("task_order") != list(TASK_NAMES):
        raise ValueError("Lap MOO task order must be chem, exc, op.")
    if metadata.get("one_stage_main_training") is not True:
        raise ValueError("Checkpoint is not marked as one-stage main training.")
    _validate_no_legacy_controls(metadata)
    if metadata.get("method") not in METHOD_IDS or not metadata.get("architecture"):
        raise ValueError("Lap MOO metadata needs method and architecture identities.")
    objectives = metadata.get("objective_definitions")
    if not isinstance(objectives, Mapping) or set(objectives) != set(TASK_NAMES):
        raise TypeError("Lap MOO objective definitions must be a mapping.")
    if not isinstance(metadata.get("method_hyperparameters"), Mapping):
        raise TypeError("Lap MOO method hyperparameters must be a mapping.")
    if metadata["method"] == "fixed":
        fixed = metadata.get("fixed_scalarization")
        weights = fixed.get("fixed_weights") if isinstance(fixed, Mapping) else None
        if not isinstance(weights, list) or len(weights) != len(TASK_NAMES):
            raise ValueError("Fixed scalarization needs three frozen calibration weights.")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in weights
        ):
            raise ValueError("Fixed scalarization weights must be positive finite numbers.")
        if not math.isclose(math.prod(float(value) for value in weights), 1.0, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Fixed scalarization weights must have unit geometric mean.")
    elif metadata.get("fixed_scalarization") is not None:
        raise ValueError("Only the fixed method may contain scalarization weights.")
    for name in (
        "predopt_checkpoint_sha256",
        "sampling_manifest_sha256",
        "minnesota_data_sha256",
        "operator_corpus_manifest_sha256",
        "ao_cache_manifest_sha256",
        "reaction_dispersions_sha256",
        "mrks_dispersions_sha256",
    ):
        digest = metadata.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            raise ValueError(f"Lap MOO metadata has an invalid {name}.")
    optimizer = metadata.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(metadata.get("lr_schedule"), Mapping):
        raise TypeError("Lap MOO metadata must define optimizer and LR schedule mappings.")
    if not legacy_read_only:
        chunk_size = metadata.get("ao_cache_chunk_size")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("Lap MOO metadata must define a positive integer AO-cache chunk size.")
    family = optimizer.get("family")
    if family is not None and family not in OPTIMIZER_FAMILIES:
        raise ValueError("Lap MOO metadata names an unsupported optimizer family.")
    if family in ("radamw", "adamw"):
        _positive_finite_number(optimizer.get("learning_rate"), "optimizer learning rate")
        _positive_finite_number(optimizer.get("weight_decay"), "optimizer weight decay", allow_zero=True)
    elif family == "muon_adamw":
        _positive_finite_number(optimizer.get("muon_learning_rate"), "Muon learning rate")
        _positive_finite_number(
            optimizer.get("adamw_fallback_learning_rate"), "AdamW fallback learning rate"
        )
        _positive_finite_number(optimizer.get("muon_weight_decay"), "Muon weight decay", allow_zero=True)
        _positive_finite_number(
            optimizer.get("adamw_weight_decay"), "AdamW fallback weight decay", allow_zero=True
        )
        muon = optimizer.get("muon")
        if not isinstance(muon, Mapping) or muon.get("orthogonalization_dtype") != (
            "bfloat16 (native PyTorch implementation)"
        ):
            raise ValueError("Muon metadata must identify native PyTorch BF16 orthogonalization.")
        muon_names = optimizer.get("muon_parameter_names")
        adamw_names = optimizer.get("adamw_fallback_parameter_names")
        if (
            not isinstance(muon_names, list)
            or not muon_names
            or not all(isinstance(name, str) and name for name in muon_names)
            or not isinstance(adamw_names, list)
            or not adamw_names
            or not all(isinstance(name, str) and name for name in adamw_names)
            or len(muon_names + adamw_names) != len(set(muon_names + adamw_names))
            or optimizer.get("all_trainable_parameters_included_once") is not True
        ):
            raise ValueError("Muon/AdamW metadata must list a disjoint complete parameter partition.")
    grid_chunk_size = metadata.get("grid_chunk_size")
    if (
        type(metadata.get("random_seed")) is not int
        or type(grid_chunk_size) is not int
        or grid_chunk_size <= 0
    ):
        raise ValueError("Lap MOO metadata seed and grid chunk size are invalid.")


__all__ = [
    "LEGACY_READ_ONLY_PROTOCOL_VERSION",
    "METHOD_IDS",
    "OPTIMIZER_FAMILIES",
    "PROTOCOL_VERSION",
    "SAMPLING_PROTOCOL_VERSION",
    "TASK_NAMES",
    "SamplingStream",
    "build_sampling_manifest",
    "build_sampling_manifest_from_catalog",
    "canonical_sha256",
    "file_sha256",
    "make_protocol_metadata",
    "read_sampling_manifest",
    "sampling_entry",
    "validate_cursor",
    "validate_protocol_metadata",
    "validate_sampling_manifest",
    "write_sampling_manifest",
]
