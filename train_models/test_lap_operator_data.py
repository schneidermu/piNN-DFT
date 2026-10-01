"""Focused provenance, immutability, and identity checks for central records."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from train_models.build_lap_operator_data import _validate_external_output
from train_models.lap_operator import OPERATOR_PROTOCOL
from train_models.lap_operator_data import (
    PROTOCOL,
    SPATIAL_DERIVATIVE_METHOD,
    load_central_operator_record,
    project_reference_vxc_ao,
    recover_legacy_centers,
    verify_operator_corpus,
    write_central_operator_record,
    write_corpus_manifest,
)


def _fixture_record():
    phi = np.array([[1.0, 0.2], [0.3, 1.1], [-0.4, 0.8]], dtype=np.float64)
    weights = np.array([0.2, 0.5, 0.7], dtype=np.float64)
    vxc = np.array([-0.3, 0.4, 0.7], dtype=np.float64)
    arrays = {
        "coords64": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        "legacycoords32": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
        "sourcerow": np.arange(3),
        "npzrow": np.arange(3),
        "weights": weights,
        "VxcLegacy": vxc,
        "Exc": np.asarray(-0.73),
        "DensityDescriptorsN10": np.ones((3, 10), dtype=np.float64),
        "dmks": np.eye(2, dtype=np.float64),
        "RefAO": project_reference_vxc_ao(phi, weights, vxc),
        "Overlap": np.eye(2, dtype=np.float64),
    }
    metadata = {
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "spatial_derivative_method": SPATIAL_DERIVATIVE_METHOD,
        "system_name": "Fixture",
        "point_count": 3,
        "nao": 2,
        "source": {
            "legacy_pickle_sha256": "legacy-fixture",
            "npz_file_sha256": "npz-fixture",
            "npz_relative_id": "Fixture/inp_mrks.npz",
            "vxc_matching": {"max_abs": 0.0, "unmatched_legacy_points": 0},
        },
        "verification": {
            "central_descriptor_compatibility": {
                field: {
                    "float32_level_compatible": True,
                    "float32_level_max_normalized_error": 0.01,
                }
                for field in ("rho", "sigma_aa_ab_bb", "lapl")
            }
        },
    }
    return arrays, metadata


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_central_record_roundtrip_is_readonly_and_hashed(tmp_path):
    arrays, metadata = _fixture_record()
    record_path = tmp_path / "Fixture.h5"
    written = write_central_operator_record(record_path, arrays, metadata)
    loaded = load_central_operator_record(record_path)

    assert loaded.metadata["protocol"] == "lap-operator-central-ao-noh-v1"
    assert loaded.metadata["operator_protocol"] == "lap-weakform-ao-v1"
    assert loaded.metadata["record_sha256"] == written["record_sha256"]
    assert loaded.coords64.dtype == np.dtype("<f8")
    assert loaded.legacycoords32.dtype == np.dtype("<f4")
    assert not loaded.coords64.flags.writeable
    assert not loaded.RefAO.flags.writeable
    np.testing.assert_array_equal(loaded.VxcLegacy, arrays["VxcLegacy"])
    with pytest.raises(FileExistsError):
        write_central_operator_record(record_path, arrays, metadata)


def test_record_hash_detects_array_tampering(tmp_path):
    h5py = pytest.importorskip("h5py")
    arrays, metadata = _fixture_record()
    record_path = tmp_path / "Fixture.h5"
    write_central_operator_record(record_path, arrays, metadata)
    with h5py.File(record_path, "r+") as handle:
        handle["VxcLegacy"][0] += 0.25
    with pytest.raises(ValueError, match="array hash mismatch"):
        load_central_operator_record(record_path)


def test_h_free_protocol_rejects_h_or_stencil_fields_recursively():
    arrays, metadata = _fixture_record()
    metadata["source"] = {"h_bohr": 0.01}
    with pytest.raises(ValueError, match="forbidden"):
        write_central_operator_record("unused.h5", arrays, metadata)


def test_center_recovery_uses_exact_float32_identity_and_rejects_collision():
    source = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=np.float64)
    recovered, indices = recover_legacy_centers(source.astype(np.float32), source)
    np.testing.assert_array_equal(indices, np.array([0, 1]))
    np.testing.assert_array_equal(recovered, source)

    collide = np.array([[0.1, 0.0, 0.0], [0.10000000001, 0.0, 0.0]])
    with pytest.raises(ValueError, match="distinct NPZ float64"):
        recover_legacy_centers(collide[:1].astype(np.float32), collide)


def test_scalar_reference_projection_has_no_rks_half_factor():
    phi = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    weights = np.array([0.5, 1.5], dtype=np.float64)
    vxc = np.array([2.0, -0.25], dtype=np.float64)
    expected = sum(weights[g] * vxc[g] * np.outer(phi[g], phi[g]) for g in range(2))
    np.testing.assert_allclose(project_reference_vxc_ao(phi, weights, vxc), expected)


def test_manifest_verifier_and_all90_gate(tmp_path):
    arrays, metadata = _fixture_record()
    record_path = tmp_path / "Fixture.h5"
    digests = write_central_operator_record(record_path, arrays, metadata)
    manifest = {
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "built_systems": ["Fixture"],
        "records": [
            {
                "system_name": "Fixture",
                "file": record_path.name,
                **digests,
                "point_count": 3,
                "nao": 2,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_operator_corpus(tmp_path, expected_systems={"Fixture"})
    assert result["verified_records"] == 1
    assert result["point_count"] == 3
    with pytest.raises(ValueError, match="System set mismatch|All-90"):
        verify_operator_corpus(
            tmp_path,
            expected_systems={f"S{i}" for i in range(90)},
            require_all90=True,
        )


def test_builder_rejects_git_repository_output(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside the Git repository"):
        _validate_external_output(repo_root / "generated_corpus")
    assert _validate_external_output(tmp_path / "external-corpus") == (tmp_path / "external-corpus").resolve()


def test_root_corpus_manifest_contains_metadata_and_hashes_only(tmp_path):
    arrays, metadata = _fixture_record()
    record_path = tmp_path / "Fixture.h5"
    digests = write_central_operator_record(record_path, arrays, metadata)
    external = {
        "protocol": PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "built_systems": ["Fixture"],
        "records": [
            {
                "system_name": "Fixture",
                "file": record_path.name,
                **digests,
                "point_count": 3,
                "nao": 2,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(external), encoding="utf-8")

    root_path = tmp_path / "lap_operator_corpus_manifest.json"
    manifest = write_corpus_manifest(
        tmp_path,
        root_path,
        expected_systems={"Fixture"},
        require_all90=False,
    )
    assert manifest["corpus"]["system_count"] == 1
    assert manifest["corpus"]["vxc_unmatched_legacy_points"] == 0
    assert manifest["records"][0]["metadata"]["record_sha256"] == digests["record_sha256"]
    assert "DensityDescriptorsN10" not in manifest["records"][0]
    with pytest.raises(FileExistsError):
        write_corpus_manifest(tmp_path, root_path, require_all90=False)
