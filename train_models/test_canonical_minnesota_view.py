import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).with_name("canonical_minnesota_view.py")
SPEC = importlib.util.spec_from_file_location("canonical_minnesota_view", MODULE_PATH)
canonical_view = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(canonical_view)


SUFFIXES = [f"level{level}" for level in range(2, 10)]


def _reaction(suffix: str, reaction_id: int) -> dict:
    shared = np.asarray([1.0, 2.0], dtype=np.float32)
    return {
        "Database": "MGAE109",
        "ReactionID": reaction_id,
        "Components": np.asarray(["a", "b", "c"]),
        "Coefficients": np.asarray([1.0, -1.0, 1.0]),
        "Energy": float(reaction_id),
        "component_paths": [f"components__{suffix}.npy"] * 3,
        "Grid": np.full((2, 9), reaction_id, dtype=np.float32),
        "Weights": shared,
        "Densities": np.full((2, 2), reaction_id, dtype=np.float32),
        "Gradients": np.full((2, 6), reaction_id, dtype=np.float32),
        "HF_energies": np.full((3,), reaction_id, dtype=np.float64),
        "backsplit_ind": 1,
        "PBE_local_energies": shared,
    }


def _dataset() -> dict[int, list[dict]]:
    return {
        group: [_reaction(suffix, 1000 + group) for suffix in SUFFIXES]
        for group in range(canonical_view.EXPECTED_GROUPS)
    }


def test_stream_reader_selects_canonical_and_preserves_memo_references(tmp_path):
    source = tmp_path / "grouped.pickle"
    with source.open("wb") as handle:
        pickle.dump(_dataset(), handle, protocol=4)

    with source.open("rb") as handle:
        last_use, _ = canonical_view.scan_memo_last_uses(handle)
    with source.open("rb") as handle:
        unpickler = canonical_view._PruningUnpickler(handle, last_use, pilot_group=4)
        actual = canonical_view._validate_root(unpickler.load(), unpickler._metadata)

    assert len(actual) == canonical_view.EXPECTED_GROUPS
    assert actual[4][0]["Grid"][0, 0] == 1004
    assert unpickler._metadata[4]["selected_suffix"] == "level2"
    assert len(unpickler._pilot_variants) == canonical_view.EXPECTED_VARIANTS
    assert (
        unpickler._pilot_variants[0]["Weights"]
        is unpickler._pilot_variants[0]["PBE_local_energies"]
    )


def test_build_emits_contiguous_canonical_and_eight_variant_pilot(tmp_path):
    source = tmp_path / "grouped.pickle"
    output = tmp_path / "canonical.pickle"
    manifest_path = tmp_path / "manifest.json"
    pilot_path = tmp_path / "pilot.pickle"
    with source.open("wb") as handle:
        pickle.dump(_dataset(), handle, protocol=4)

    manifest = canonical_view.build_canonical_view(
        source,
        output,
        manifest_path,
        pilot_group=7,
        pilot_output_path=pilot_path,
    )

    with output.open("rb") as handle:
        canonical = pickle.load(handle)
    assert list(canonical) == list(range(268))
    assert all(len(group) == 1 for group in canonical.values())
    assert manifest["base_reaction_count"] == 268
    assert manifest["source_sha256"] == canonical_view.sha256_file(source)
    assert manifest["output_sha256"] == canonical_view.sha256_file(output)
    assert manifest["groups"][7]["selected_suffix"] == "level2"
    assert manifest["pilot_view"]["selected_reaction_count"] == 1
    with pilot_path.open("rb") as handle:
        pilot = pickle.load(handle)
    assert list(pilot) == [7]
    assert len(pilot[7]) == 8


def test_rejects_unexpected_variant_count():
    group = [_reaction(suffix, 0) for suffix in SUFFIXES[:-1]]
    with pytest.raises(canonical_view.SourceFormatError, match="expected 8"):
        canonical_view._validate_variants(0, group)


def test_rejects_mismatched_schema():
    group = [_reaction(suffix, 0) for suffix in SUFFIXES]
    del group[0]["PBE_local_energies"]
    group[0]["Unexpected"] = 1
    with pytest.raises(canonical_view.SourceFormatError, match="field schema mismatch"):
        canonical_view._validate_variants(0, group)
