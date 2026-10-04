"""Fail-closed stencil provenance, protocol, training and calibration checks."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
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
    require_full_center_verification,
    validate_record,
    verify_corpus,
    write_stencil_h5,
)
from lap_diagnostics import diagnose, measure_objectives
from lap_training import (
    canonical_predopt_view,
    minnesota_sigma_boundary_matches,
    mrks_losses,
    predopt_loss,
    reaction_loss,
    run_predopt,
)
from lap_vxc import (
    STENCIL_ORDER,
    STENCIL_VERSION,
    LapEnergy,
    integrated_energy,
    sigma_total_to_standard,
    stencil_coordinates,
)
from optuna_joint import batch_fchem, build_model, parse_args, vxc_loss
from prepare_training_corpus import exclusion_pairs, sha256
from test_lap import raw_grid, small_model
from train_lap import epoch_steps


def test_minnesota_sigma_boundary_accepts_float32_cancellation_rounding():
    standard = torch.tensor(
        [[0.00027287358534522355, -0.0002746396348811686, 0.0002764188393484801]],
        dtype=torch.float32,
    )
    model_sigma = torch.tensor(
        [[0.00027287358534522355, 1.3165866619146982e-8, 0.0002764188393484801]],
        dtype=torch.float32,
    )
    assert not torch.allclose(
        model_sigma,
        torch.stack(
            [
                standard[:, 0],
                standard[:, 0] + 2 * standard[:, 1] + standard[:, 2],
                standard[:, 2],
            ],
            dim=-1,
        ),
        rtol=1e-6,
        atol=1e-12,
    )
    assert minnesota_sigma_boundary_matches(model_sigma, standard)


def record(name="He", spin=0, kind="common-rks"):
    coords = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]], dtype=torch.float64)
    sc = stencil_coordinates(coords, 0.1)
    f = analytic_features(sc, equal_spin=spin == 0)
    return {
        "Name": name,
        "Coordinates": coords,
        "LegacyCoordinates": coords.clone(),
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
            "npz_sha256": "a" * 64,
            "legacy_target_sha256": "b" * 64,
            "E_xc_source": "preserved legacy mRKS training target",
            "NPZ_exc_wf_substituted": False,
            "legacy_training_points": 2,
            "central_density_check": {
                "points_checked": 2,
                **{
                    key: {"combined": {"float32_level_compatible": True}}
                    for key in ("rho", "sigma_aa_ab_bb", "lapl")
                },
            },
        },
    }


def test_production_center_verifier_requires_every_legacy_row():
    d = record()
    require_full_center_verification(d)
    d["SourceProvenance"]["central_density_check"]["points_checked"] = 1
    with pytest.raises(ValueError, match="every legacy central point"):
        require_full_center_verification(d)


def test_production_center_verifier_accepts_old_independent_full_population_counts():
    d = record()
    provenance = d["SourceProvenance"]
    del provenance["legacy_training_points"]
    provenance["legacy_vxc_matching"] = {
        "legacy_points": 2,
        "matched_legacy_points": 2,
        "unmatched_legacy_points": 0,
    }
    require_full_center_verification(d)
    provenance["legacy_vxc_matching"]["matched_legacy_points"] = 1
    with pytest.raises(ValueError, match="lacks legacy central-point"):
        require_full_center_verification(d)


def write_h5(path, d, layout="point,spin"):
    with h5py.File(path, "w") as f:
        keys = {
            "Coordinates": "coords",
            "LegacyCoordinates": "legacy_coords",
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


def test_legacy_7point_manifest_without_derivative_order_remains_readable(corpus):
    manifest_path = corpus / "preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["stencil_order"] == list(STENCIL_ORDER)
    manifest.pop("derivative_order")
    manifest_path.write_text(json.dumps(manifest))

    restored_manifest, records = verify_corpus(corpus)
    assert restored_manifest["derivative_order"] == 2
    assert restored_manifest["stencil_version"] == STENCIL_VERSION
    assert len(records) == 90


def test_built_corpus_remains_usable_when_source_h5_moves(corpus, tmp_path):
    source_dir = tmp_path / "stencils"
    for path in source_dir.glob("*.h5"):
        path.unlink()
    source_dir.rmdir()
    manifest_path = corpus / "preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["minnesota_manifest_source_path"] = (
        "Z:/source-was-moved/preprocessing_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest))
    manifest, records = verify_corpus(corpus)
    assert len(records) == 90
    assert len(manifest["source_h5_identities"]) == 90


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


def test_optuna_cli_accepts_registered_lap_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "optuna_joint.py",
            "--study-name",
            "test",
            "--storage",
            "sqlite:///unused.db",
            "--n-trials",
            "1",
            "--output-dir",
            "unused",
            "--model-type",
            "lap",
        ],
    )
    assert parse_args().model_type == "lap"


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


@pytest.mark.parametrize(
    ("database", "expected_factor"),
    [("NCCE31", 2.896652477664184), ("AE17", 0.5282130988681748)],
)
def test_reaction_loss_scalar_database_matches_list_loss_and_gradient(
    database, expected_factor, monkeypatch
):
    class ScaledConstants(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, raw):
            return self.scale.expand(len(raw), 1)

    def fake_reaction_energy(reaction, constants, device, *args, **kwargs):
        return constants.mean().reshape(1), None

    monkeypatch.setattr(
        "lap_training.calculate_reaction_energy", fake_reaction_energy
    )
    raw = raw_grid()
    target = torch.zeros(1, dtype=raw.dtype)

    def reaction(database_label):
        return {
            "Grid": raw,
            "Densities": raw[:, :2],
            "Gradients": sigma_total_to_standard(raw[:, 2:5]),
            "Database": database_label,
        }

    scalar_model = ScaledConstants()
    list_model = ScaledConstants()
    scalar_loss = reaction_loss(
        scalar_model,
        reaction(database),
        target,
        torch.device("cpu"),
        torch.float64,
    )
    list_loss = reaction_loss(
        list_model,
        reaction([database]),
        target,
        torch.device("cpu"),
        torch.float64,
    )
    scalar_gradient = torch.autograd.grad(scalar_loss, scalar_model.scale)[0]
    list_gradient = torch.autograd.grad(list_loss, list_model.scale)[0]

    torch.testing.assert_close(scalar_loss, list_loss)
    torch.testing.assert_close(scalar_gradient, list_gradient)
    assert scalar_loss.item() == pytest.approx(expected_factor)
    assert torch.isfinite(scalar_gradient).all()
    assert scalar_gradient.item() == pytest.approx(expected_factor)


def test_batch_fchem_rejects_scalar_database_name():
    with pytest.raises(TypeError, match="string"):
        batch_fchem(
            "NCCE31",
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


def test_legacy_predopt_batch_fchem_preserves_lists_and_rejects_scalar():
    script = """
import sys
import types

sys.modules["mlflow"] = types.ModuleType("mlflow")
import torch
import predopt_train

prediction = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
reference = torch.zeros(1, dtype=torch.float64)
loss = predopt_train.batch_fchem(["NCCE31"], prediction, reference)
loss.backward()
factor = (
    predopt_train.FCHEM_VALIDATION["NCCE31"]
    * predopt_train.FREQ_WEIGHTS["NCCE31"]
    / predopt_train.mean_weight
)
assert abs(loss.item() - 2 * factor) < 1e-12
assert abs(prediction.grad.item() - factor) < 1e-12

try:
    predopt_train.batch_fchem("NCCE31", prediction.detach(), reference)
except TypeError:
    pass
else:
    raise AssertionError("scalar database label was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_predopt_updates_canonical_constants_in_memory():
    model = small_model()
    raw = raw_grid()
    before = predopt_loss(model, raw).detach()
    history = run_predopt(
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
    assert history[0]["mse"] > 0
    assert history[0]["mae"] > 0


def test_lap_predopt_uses_one_canonical_variant_per_base_reaction():
    grouped = {
        index: [
            {"component_paths": [f"m{index}__level2_delley.h5"]},
            {"component_paths": [f"m{index}__level2.h5"]},
        ]
        for index in range(268)
    }
    view = canonical_predopt_view(grouped)
    assert len(view) == 268
    assert all(
        item["component_paths"][0].endswith("__level2.h5") for item in view.values()
    )
    grouped[0].append({"component_paths": ["m0.h5"]})
    assert canonical_predopt_view(grouped)[0]["component_paths"] == ["m0.h5"]
    with pytest.raises(ValueError, match="268"):
        canonical_predopt_view({0: grouped[0]})


def test_final_partial_gradient_accumulation_window_is_not_underweighted(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    coefficients = iter((2.0, 4.0, 8.0))

    def fake_reaction_loss(*args):
        return model.weight.sum() * next(coefficients)

    def fake_mrks_losses(*args):
        return model.weight.sum() * 0, model.weight.sum() * 0

    monkeypatch.setattr("train_lap.reaction_loss", fake_reaction_loss)
    monkeypatch.setattr("train_lap.mrks_losses", fake_mrks_losses)
    epoch_steps(
        model=model,
        energy=None,
        reaction_loader=[(None, torch.tensor(0.0)) for _ in range(3)],
        records=[record()],
        indices=[0],
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
        weights=[1.0, 0.0, 0.0],
        device=torch.device("cpu"),
        dtype=torch.float64,
        chunk=1,
        accum_iter=2,
        world=1,
    )
    assert model.weight.item() == pytest.approx(-11.0)
