"""Tests for deterministic MOO survey-panel data selection."""


import hashlib
import json
import pickle

import pytest
import torch

from train_models.lap_moo_panel import (
    EXPECTED_MINNESOTA_DATABASES,
    MN_GROUP_STORE_PROTOCOL,
    MRKS_PANEL_SYSTEMS,
    MinnesotaGroupStore,
    _LazyTensorUnpickler,
    _materialize_selected_tensors,
    _StorageSpool,
    select_minnesota_panel,
)


def _canonical_manifest():
    groups = []
    group_key = 0
    for database in EXPECTED_MINNESOTA_DATABASES:
        for rank in range(4):
            groups.append(
                {
                    "database": database,
                    "reaction_id": rank,
                    "group_key": group_key,
                    "output_index": group_key,
                    "point_count": 1000 + rank,
                    "selected_suffix": "level2",
                    "available_suffixes": ["level2", "level3"],
                    "variant_count": 2,
                }
            )
            group_key += 1
    return {
        "schema": "canonical-minnesota-view-v1",
        "groups": groups,
    }


def test_panel_selects_three_evenly_spaced_groups_per_database():
    first = select_minnesota_panel(_canonical_manifest())
    second = select_minnesota_panel(_canonical_manifest())

    assert first == second
    assert len(first) == 27
    assert {row["database"] for row in first} == set(EXPECTED_MINNESOTA_DATABASES)
    for database in EXPECTED_MINNESOTA_DATABASES:
        selected = [row for row in first if row["database"] == database]
        assert [row["database_group_rank"] for row in selected] == [0, 2, 3]
        assert all(row["variant_suffixes"] == ["level2", "level3"] for row in selected)
        assert all(row["variant_count"] == 2 for row in selected)


def test_selector_rejects_missing_database_or_duplicate_variants():
    missing_database = _canonical_manifest()
    missing_database["groups"] = [
        row for row in missing_database["groups"] if row["database"] != "ABDE4"
    ]
    with pytest.raises(ValueError, match="expected nine databases"):
        select_minnesota_panel(missing_database)

    duplicate_variant = _canonical_manifest()
    duplicate_variant["groups"][0]["available_suffixes"] = ["level2", "level2"]
    with pytest.raises(ValueError, match="invalid variant metadata"):
        select_minnesota_panel(duplicate_variant)


def test_mrks_panel_is_fixed_distinct_and_has_full_panel_scale():
    assert len(MRKS_PANEL_SYSTEMS) == 15
    assert len(set(MRKS_PANEL_SYSTEMS)) == 15
    assert {"H2", "ClHS", "HPSi_iso2"}.issubset(MRKS_PANEL_SYSTEMS)


def test_lazy_storage_spool_preserves_tensor_values_and_shared_storage(tmp_path):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    view = base[1:]
    expected = {"base": base, "view": view}
    pickle_path = tmp_path / "fixture.pickle"
    with pickle_path.open("wb") as stream:
        pickle.dump(expected, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with pickle_path.open("rb") as stream:
        standard = pickle.load(stream)

    spool = _StorageSpool(tmp_path / "storage.spool")
    try:
        with pickle_path.open("rb") as stream:
            lazy = _LazyTensorUnpickler(stream, spool).load()
    finally:
        spool.close()
    actual = _materialize_selected_tensors(lazy)

    assert torch.equal(actual["base"], standard["base"])
    assert torch.equal(actual["view"], standard["view"])
    assert actual["view"].stride() == standard["view"].stride()
    assert actual["view"].storage_offset() == standard["view"].storage_offset()
    actual_alias = actual["base"].untyped_storage().data_ptr() == actual["view"].untyped_storage().data_ptr()
    standard_alias = standard["base"].untyped_storage().data_ptr() == standard["view"].untyped_storage().data_ptr()
    assert actual_alias == standard_alias


def test_group_store_hash_verifies_and_loads_explicit_variant(tmp_path):
    reactions = [
        {
            "Database": "MGAE109",
            "ReactionID": 5,
            "component_paths": [f"/grid/component__{suffix}.h5"],
            "Grid": torch.tensor([[float(index), 1.0]], dtype=torch.float32),
        }
        for index, suffix in enumerate(("level2", "level3"))
    ]
    group_path = tmp_path / "group_17.pickle"
    with group_path.open("wb") as stream:
        pickle.dump(reactions, stream, protocol=pickle.HIGHEST_PROTOCOL)
    file_bytes = group_path.stat().st_size
    group_hash = hashlib.sha256(group_path.read_bytes()).hexdigest()
    manifest = {
        "schema": MN_GROUP_STORE_PROTOCOL,
        "source_pickle_sha256_verified": "source-sha256",
        "group_count": 1,
        "groups": [
            {
                "database": "MGAE109",
                "reaction_id": 5,
                "source_group_key": 17,
                "file": group_path.name,
                "file_bytes": file_bytes,
                "file_sha256": group_hash,
                "variant_count": 2,
                "variant_suffixes": ["level2", "level3"],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = MinnesotaGroupStore(manifest_path, cache_groups=1)
    group = store.load_group(("MGAE109", 5))
    variant = store.load_variant(
        {"database": "MGAE109", "reaction_id": 5},
        "level3",
    )

    assert [row["component_paths"][0].split("__")[-1][:-3] for row in group] == [
        "level2",
        "level3",
    ]
    assert torch.equal(variant["Grid"], reactions[1]["Grid"])
    assert len(store._cache) == 1


def test_group_store_rejects_changed_record_hash(tmp_path):
    group_path = tmp_path / "group_17.pickle"
    with group_path.open("wb") as stream:
        pickle.dump([], stream)
    manifest = {
        "schema": MN_GROUP_STORE_PROTOCOL,
        "group_count": 1,
        "groups": [
            {
                "database": "MGAE109",
                "reaction_id": 5,
                "source_group_key": 17,
                "file": group_path.name,
                "file_bytes": group_path.stat().st_size,
                "file_sha256": "0" * 64,
                "variant_count": 0,
                "variant_suffixes": [],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = MinnesotaGroupStore(manifest_path)
    with pytest.raises(ValueError, match="file hash mismatch"):
        store.load_group(17)
