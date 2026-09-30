"""Fail-closed stencil provenance, protocol, training and calibration checks."""

import json
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from dataset import collate_fn
from lap_analytic import analytic_features
from lap_data import (
    PROTOCOL,
    build_corpus,
    read_stencil_h5,
    validate_record,
    verify_corpus,
    write_stencil_h5,
)
from lap_diagnostics import diagnose, measure_objectives
from lap_training import mrks_losses, predopt_loss, reaction_loss, run_predopt
from lap_vxc import (
    STENCIL_VERSION,
    LapEnergy,
    integrated_energy,
    sigma_total_to_standard,
    stencil_coordinates,
)
from optuna_joint import build_model, vxc_loss
from prepare_training_corpus import exclusion_pairs, sha256
from test_lap import raw_grid, small_model


def record(name="He", spin=0, kind="common-rks"):
    coords = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]], dtype=torch.float64)
    sc = stencil_coordinates(coords, 0.1)
    f = analytic_features(sc, equal_spin=spin == 0)
    return {
        "Name": name,
        "Coordinates": coords,
        "StencilCoordinates": sc,
        "StencilFeatures": f,
        "Weights": torch.tensor([0.1, 0.2], dtype=torch.float64),
        "Vxc": torch.ones(2, 2, dtype=torch.float64),
        "E_xc": torch.tensor(-0.5, dtype=torch.float64),
        "Protocol": PROTOCOL,
        "StencilVersion": STENCIL_VERSION,
        "HBohr": 0.1,
        "SourceSpin": spin,
        "TargetKind": kind,
        "GaugeMetadata": {"available": False},
        "SourceProvenance": {
            "generator": "synthetic-test",
            "reference_density": "analytic-test",
            "molecule_basis_ao_order": "analytic/no-AO-test",
            "full_vxc_target": "synthetic-test",
        },
    }


def write_h5(path, d, layout="point,spin"):
    with h5py.File(path, "w") as f:
        keys = {
            "Coordinates": "coords",
            "StencilCoordinates": "stencil_coords",
            "StencilFeatures": "stencil_features",
            "Weights": "weights",
            "Vxc": "vxc",
            "E_xc": "E_xc",
        }
        for key, field in keys.items():
            val = d[key].numpy()
            if key == "Vxc" and layout == "spin,point":
                val = val.T
            f[field] = val
        f.attrs.update(
            protocol=d["Protocol"],
            stencil_version=d["StencilVersion"],
            h_bohr=d["HBohr"],
            source_spin=d["SourceSpin"],
            target_kind=d["TargetKind"],
            vxc_layout=layout,
            gauge_metadata=json.dumps(d["GaugeMetadata"]),
            source_provenance=json.dumps(d["SourceProvenance"]),
        )


@pytest.mark.parametrize("layout", ["point,spin", "spin,point"])
def test_h5_preserves_ambiguous_two_point_spin_channels(tmp_path, layout):
    d = record(spin=1, kind="spin-resolved")
    d["Vxc"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    path = tmp_path / "He.h5"
    write_h5(path, d, layout)
    assert torch.equal(read_stencil_h5(path)["Vxc"], d["Vxc"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("Protocol", "legacy", "protocol"),
        ("HBohr", 0.0, "HBohr"),
        ("GaugeMetadata", {}, "available"),
        ("SourceProvenance", {}, "provenance"),
        ("SourceSpin", 1, "closed-shell"),
        ("Vrho", torch.ones(2), "reference spin AO"),
    ],
)
def test_record_scientific_corruption(field, value, match):
    d = record()
    d[field] = value
    with pytest.raises(ValueError, match=match):
        validate_record(d)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    mn, source, output = [tmp_path / x for x in ("mn", "stencils", "lap")]
    mn.mkdir()
    source.mkdir()
    for name in ("data_predopt.pickle", "data_train_grouped.pickle"):
        with (mn / name).open("wb") as f:
            pickle.dump({}, f)
    (mn / "minnesota_protocol.json").write_text("{}")
    manifest = {
        "minnesota_source_reactions": 284,
        "minnesota_training_reactions": 268,
        "excluded_minnesota_reactions": exclusion_pairs(),
        "augmented_reaction_samples": 268,
        "artifact_sha256": {p.name: sha256(p) for p in mn.iterdir()},
    }
    (mn / "preprocessing_manifest.json").write_text(json.dumps(manifest))
    # Synthetic fixture only; production always calls the real legacy verifier.
    monkeypatch.setattr("prepare_training_corpus.verify", lambda path: manifest)
    monkeypatch.setattr("launch_provenance.current_commit", lambda path: "0" * 40)
    for i in range(90):
        write_h5(source / f"system{i:02d}.h5", record(f"system{i:02d}"))
    build_corpus(mn, source, output)
    return output


def test_corpus_roundtrip_and_no_overwrite(corpus):
    manifest, records = verify_corpus(corpus)
    assert len(records) == manifest["mrks_systems"] == 90
    assert manifest["minnesota_training_reactions"] == 268
    with pytest.raises(FileExistsError):
        build_corpus("", "", corpus)


@pytest.mark.parametrize(
    "field,value",
    [
        ("h_bohr", 0.2),
        ("units", "Angstrom"),
        ("stencil_order", ["center"]),
        ("mrks_systems", 89),
        ("minnesota_training_reactions", 267),
        ("excluded_minnesota_reactions", []),
        ("descriptor_protocol", "tau"),
        ("stencil_data_sha256", "invalid"),
        ("minnesota_manifest_sha256", "invalid"),
    ],
)
def test_manifest_corruptions_fail_closed(corpus, field, value):
    path = corpus / "preprocessing_manifest.json"
    m = json.loads(path.read_text())
    m[field] = value
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        verify_corpus(corpus)


def test_artifact_and_target_metadata_corruption(corpus):
    path = corpus / "preprocessing_manifest.json"
    m = json.loads(path.read_text())
    m["systems"][0]["target_kind"] = "spin-resolved"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="scientific metadata"):
        verify_corpus(corpus)
    m["systems"][0]["target_kind"] = "common-rks"
    path.write_text(json.dumps(m))
    with (corpus / "data_full_vxc_train.pickle").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_corpus(corpus)


def test_versioned_writer_preserves_reference_and_refuses_overwrite(tmp_path):
    d = record()
    path = tmp_path / "He.h5"
    write_stencil_h5(path, d)
    restored = read_stencil_h5(path)
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored[key], value)
        else:
            assert restored[key] == value
    with pytest.raises(FileExistsError):
        write_stencil_h5(path, d)


def test_legacy_partial_loss_and_wrong_name_are_rejected():
    model = small_model()
    with pytest.raises(ValueError, match="full_vxc_loss"):
        vxc_loss(model, {}, torch.device("cpu"))
    args = SimpleNamespace(name="PBE-Lap-LGxGc_2_8", model_type="lap", dropout=0.0)
    assert type(build_model(args, torch.device("cpu"))) is type(model)
    args.model_type = "base"
    with pytest.raises(ValueError, match="explicit"):
        build_model(args, torch.device("cpu"))


def test_reaction_mrks_same_energy_and_calibration():
    model = small_model()
    energy = LapEnergy(model)
    raw = raw_grid()
    rx = {
        "Grid": raw,
        "Densities": raw[:, :2],
        "Gradients": sigma_total_to_standard(raw[:, 2:5]),
        "Weights": torch.ones(len(raw), dtype=raw.dtype),
        "HF_energies": torch.tensor([0.0], dtype=raw.dtype),
        "Components": np.array(["He"]),
        "Coefficients": torch.tensor([1.0], dtype=raw.dtype),
        "backsplit_ind": torch.tensor([len(raw)]),
        "Database": "EA13",
    }
    target = torch.tensor([1.0], dtype=raw.dtype)
    batch, target = collate_fn([(rx, target)])
    loss = reaction_loss(model, batch, target, torch.device("cpu"), torch.float64)
    assert torch.isfinite(loss)
    assert torch.isfinite(predopt_loss(model, raw))
    d = record()
    exc, vxc = mrks_losses(energy, d, torch.device("cpu"), torch.float64, 1)
    assert torch.isfinite(exc) and torch.isfinite(vxc)
    report = measure_objectives(
        model, energy, batch, target, d, torch.device("cpu"), torch.float64, 1
    )
    assert all(x["parameter_gradient_norm"] > 0 for x in report.values())
    diag = diagnose(energy, d, torch.device("cpu"), torch.float64, 1)
    assert diag["nonfinite_fraction"] == 0
    prediction = integrated_energy(energy, d["StencilFeatures"], d["Weights"], 2)
    assert diag["exc_prediction_hartree"] == pytest.approx(float(prediction.detach()))


def test_predopt_updates_canonical_constants_in_memory():
    model = small_model()
    raw = raw_grid()
    before = predopt_loss(model, raw).detach()
    run_predopt(
        model,
        [({"Grid": raw}, None)],
        torch.device("cpu"),
        torch.float64,
        epochs=1,
        lr=1e-4,
        chunk=3,
    )
    after = predopt_loss(model, raw).detach()
    assert torch.isfinite(after)
    assert after < before
