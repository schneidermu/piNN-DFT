"""Integration checks for persisted 7- and 13-point Lap stencil schemas."""

import json
import sys

import h5py
import pytest
import torch
from lap_data import (
    PROTOCOL,
    build_corpus,
    read_stencil_h5,
    validate_record,
    verify_corpus,
    write_stencil_h5,
)
from lap_vxc import (
    STENCIL_VERSION,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
    euler_components,
    stencil_coordinates,
    stencil_order_for_version,
    stencil_positions_for_version,
)
from NN_models_lap import DESCRIPTOR_PROTOCOL, pcPBELMLOptimizerV2Lap
from torch import nn


def _features(coords, h, version, order, kind):
    stencil_coords = stencil_coordinates(coords, h, order=order, version=version)
    x, y, z = stencil_coords.unbind(-1)
    if kind == "cubic":
        rho = 1.0 + 0.2 * x**3 + 0.1 * y**3 + 0.05 * z**3
        grad = torch.stack((0.6 * x**2, 0.3 * y**2, 0.15 * z**2), dim=-1)
        lap = 1.2 * x + 0.6 * y + 0.3 * z
    else:
        rho = 1.0 + 0.1 * (x**4 + y**4 + z**4)
        grad = torch.stack((0.4 * x**3, 0.4 * y**3, 0.4 * z**3), dim=-1)
        lap = 0.12 * (x.square() + y.square() + z.square())
    spin_rho = rho.unsqueeze(-1).expand(*rho.shape, 2)
    spin_grad = grad.unsqueeze(-2).expand(*grad.shape[:-1], 2, 3)
    spin_lap = lap.unsqueeze(-1).expand(*lap.shape, 2)
    features = torch.cat((spin_rho, spin_grad.flatten(-2), spin_lap), dim=-1)
    return stencil_coords, features


def _record(version, kind="cubic"):
    order = stencil_order_for_version(version)
    h = 0.075
    coords = torch.tensor([[0.1, -0.2, 0.3], [0.35, 0.25, -0.1]], dtype=torch.float64)
    stencil_coords, features = _features(coords, h, version, order, kind)
    return {
        "Name": "synthetic",
        "Coordinates": coords,
        "LegacyCoordinates": coords.clone(),
        "StencilCoordinates": stencil_coords,
        "StencilFeatures": features,
        "Weights": torch.ones(len(coords), dtype=torch.float64),
        "Vxc": torch.zeros((len(coords), 2), dtype=torch.float64),
        "E_xc": torch.tensor(-0.25, dtype=torch.float64),
        "Protocol": PROTOCOL,
        "StencilVersion": version,
        "StencilOrder": order,
        "HBohr": h,
        "SourceSpin": 0,
        "TargetKind": "common-rks",
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


class _PolynomialEnergy(nn.Module):
    def __init__(self, b, c):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float64))

    def forward(self, rho, sigma, lapl):
        return (
            self.a * rho.square().sum(-1)
            + self.b * (sigma[..., 0] + sigma[..., 2])
            + self.c * lapl.square().sum(-1)
        )


@pytest.mark.parametrize(
    ("version", "kind", "b", "c"),
    [
        (STENCIL_VERSION, "cubic", 0.2, 0.0),
        (STENCIL_VERSION_13_POINT_FOURTH_ORDER, "cubic", 0.2, 0.0),
        (STENCIL_VERSION, "quartic", 0.0, 0.3),
        (STENCIL_VERSION_13_POINT_FOURTH_ORDER, "quartic", 0.0, 0.3),
    ],
)
def test_versioned_paths_compute_cubic_and_quartic_vxc(version, kind, b, c):
    order = stencil_order_for_version(version)
    coords = torch.tensor([[0.1, -0.2, 0.3], [0.35, 0.25, -0.1]], dtype=torch.float64)
    _, features = _features(coords, 0.075, version, order, kind)
    parts = euler_components(
        _PolynomialEnergy(b, c),
        features,
        0.075,
        order=order,
        version=version,
    )
    x, y, z = coords.unbind(-1)
    if kind == "cubic":
        rho = 1.0 + 0.2 * x**3 + 0.1 * y**3 + 0.05 * z**3
        lap = 1.2 * x + 0.6 * y + 0.3 * z
        expected = rho - 0.4 * lap
    else:
        rho = 1.0 + 0.1 * (x**4 + y**4 + z**4)
        expected = rho + 0.432
    torch.testing.assert_close(
        parts["Vxc"], expected[:, None].expand(-1, 2), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize(
    "version",
    [STENCIL_VERSION, STENCIL_VERSION_13_POINT_FOURTH_ORDER],
)
def test_h5_roundtrip_preserves_selected_stencil_schema(tmp_path, version):
    record = _record(version)
    path = tmp_path / "synthetic.h5"
    write_stencil_h5(path, record)
    restored = read_stencil_h5(path)
    assert restored["StencilVersion"] == version
    assert restored["StencilOrder"] == stencil_order_for_version(version)
    assert (
        len(stencil_positions_for_version(version))
        == restored["StencilFeatures"].shape[1]
    )
    torch.testing.assert_close(
        restored["StencilCoordinates"], record["StencilCoordinates"]
    )
    torch.testing.assert_close(restored["StencilFeatures"], record["StencilFeatures"])


@pytest.mark.parametrize(
    ("record_version", "record_order", "replacement_version", "replacement_order"),
    [
        (STENCIL_VERSION, 2, STENCIL_VERSION_13_POINT_FOURTH_ORDER, 4),
        (STENCIL_VERSION_13_POINT_FOURTH_ORDER, 4, STENCIL_VERSION, 2),
    ],
)
def test_record_and_h5_reject_cross_version_shapes(
    tmp_path, record_version, record_order, replacement_version, replacement_order
):
    record = _record(record_version)
    mismatched = dict(record)
    mismatched["StencilVersion"] = replacement_version
    mismatched["StencilOrder"] = replacement_order
    with pytest.raises(ValueError, match="shape"):
        validate_record(mismatched)

    path = tmp_path / "synthetic.h5"
    write_stencil_h5(path, record)
    with h5py.File(path, "r+") as handle:
        handle.attrs["stencil_version"] = replacement_version
        handle.attrs["derivative_order"] = replacement_order
        handle.attrs["stencil_order"] = json.dumps(
            list(stencil_positions_for_version(replacement_version))
        )
    with pytest.raises(ValueError, match="shape"):
        read_stencil_h5(path)


def test_record_and_h5_reject_nonmatching_stencil_coordinates(tmp_path):
    record = _record(STENCIL_VERSION_13_POINT_FOURTH_ORDER)
    record["StencilCoordinates"][0, 1, 0] += 1e-12
    with pytest.raises(ValueError, match="exactly match"):
        validate_record(record)

    record = _record(STENCIL_VERSION_13_POINT_FOURTH_ORDER)
    path = tmp_path / "synthetic.h5"
    write_stencil_h5(path, record)
    with h5py.File(path, "r+") as handle:
        handle["stencil_coords"][0, 1, 0] += 1e-12
    with pytest.raises(ValueError, match="exactly match"):
        read_stencil_h5(path)


def test_s5_batch_routes_explicit_version_and_order_to_vxc_loss(monkeypatch):
    import lap_vxc
    from optuna_joint import lap_full_vxc_loss, vxc_collate_fn

    record = _record(STENCIL_VERSION_13_POINT_FOURTH_ORDER)
    observed = {}

    class _LapModel(nn.Module):
        descriptor_protocol = DESCRIPTOR_PROTOCOL

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def capture_loss(
        energy, features, target, weights, h, point_chunk_size, *, order, version
    ):
        observed.update(shape=tuple(features.shape), order=order, version=version, h=h)
        return features.new_zeros((), dtype=torch.float64)

    monkeypatch.setattr(lap_vxc, "full_vxc_loss", capture_loss)
    result = lap_full_vxc_loss(
        _LapModel(), vxc_collate_fn([record]), torch.device("cpu"), 8
    )
    assert result.item() == 0.0
    assert observed == {
        "shape": (2, 13, 10),
        "order": 4,
        "version": STENCIL_VERSION_13_POINT_FOURTH_ORDER,
        "h": record["HBohr"],
    }


def test_13_point_corpus_manifest_records_layout_and_derivative_order(
    tmp_path, monkeypatch
):
    from prepare_training_corpus import exclusion_pairs, sha256

    mn_dir, stencil_dir, output_dir = (
        tmp_path / "mn",
        tmp_path / "stencils",
        tmp_path / "corpus",
    )
    mn_dir.mkdir()
    stencil_dir.mkdir()
    for filename in (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "minnesota_protocol.json",
    ):
        (mn_dir / filename).write_bytes(b"synthetic " + filename.encode())
    mn_manifest = {
        "minnesota_source_reactions": 284,
        "minnesota_training_reactions": 268,
        "excluded_minnesota_reactions": exclusion_pairs(),
        "augmented_reaction_samples": 268,
        "artifact_sha256": {path.name: sha256(path) for path in mn_dir.iterdir()},
    }
    (mn_dir / "preprocessing_manifest.json").write_text(json.dumps(mn_manifest))
    monkeypatch.setattr("prepare_training_corpus.verify", lambda _path: mn_manifest)
    monkeypatch.setattr("launch_provenance.current_commit", lambda _path: "0" * 40)
    for index in range(90):
        record = _record(STENCIL_VERSION_13_POINT_FOURTH_ORDER)
        record["Name"] = f"system{index:02d}"
        write_stencil_h5(stencil_dir / f"{record['Name']}.h5", record)

    build_corpus(mn_dir, stencil_dir, output_dir)
    manifest, records = verify_corpus(output_dir)
    assert manifest["stencil_version"] == STENCIL_VERSION_13_POINT_FOURTH_ORDER
    assert manifest["derivative_order"] == 4
    assert manifest["stencil_order"] == list(
        stencil_positions_for_version(STENCIL_VERSION_13_POINT_FOURTH_ORDER)
    )
    assert len(records) == 90
    assert all(record["StencilOrder"] == 4 for record in records)


def test_s5_checkpoint_carries_selected_stencil_order(tmp_path):
    from lap_checkpoint import checkpoint_payload, load_lap_checkpoint

    version = STENCIL_VERSION_13_POINT_FOURTH_ORDER
    stencil_metadata = {
        "version": version,
        "stencil_order": list(stencil_positions_for_version(version)),
        "derivative_order": 4,
        "h_bohr": 0.075,
        "units": "Bohr",
    }
    model = pcPBELMLOptimizerV2Lap(2, 8, use_g_x=True, use_g_c=True).double()
    payload = checkpoint_payload(model, lap_s5_provenance={"stencil": stencil_metadata})
    assert payload["stencil_version"] == version
    assert payload["stencil_order"] == list(stencil_positions_for_version(version))
    assert payload["derivative_order"] == 4
    checkpoint = tmp_path / "stencil13.pt"
    torch.save(payload, checkpoint)
    _, restored = load_lap_checkpoint(checkpoint)
    assert restored["stencil_version"] == version
    assert restored["derivative_order"] == 4


def test_checkpoint_read_normalizes_legacy_v1_s5_provenance(tmp_path):
    from lap_checkpoint import checkpoint_payload, load_lap_checkpoint

    model = pcPBELMLOptimizerV2Lap(2, 8, use_g_x=True, use_g_c=True).double()
    payload = checkpoint_payload(model)
    payload["lap_s5_provenance"] = {
        "metadata_version": 1,
        "stencil": {
            "version": STENCIL_VERSION,
            "h_bohr": 0.075,
            "units": "Bohr",
        },
    }
    checkpoint = tmp_path / "legacy-v1-stencil.pt"
    torch.save(payload, checkpoint)

    _, restored = load_lap_checkpoint(checkpoint)
    assert restored["lap_s5_provenance"]["metadata_version"] == 2
    assert restored["lap_s5_provenance"]["stencil"]["derivative_order"] == 2
    assert restored["lap_s5_provenance"]["stencil"]["stencil_order"] == list(
        stencil_positions_for_version(STENCIL_VERSION)
    )


@pytest.mark.parametrize(
    ("nested_version", "checkpoint_version"),
    [
        (STENCIL_VERSION, STENCIL_VERSION_13_POINT_FOURTH_ORDER),
        (STENCIL_VERSION_13_POINT_FOURTH_ORDER, STENCIL_VERSION),
    ],
)
def test_checkpoint_rejects_stencil_version_disagreement(
    tmp_path, nested_version, checkpoint_version
):
    from lap_checkpoint import checkpoint_payload, load_lap_checkpoint

    nested_order = stencil_order_for_version(nested_version)
    checkpoint_order = stencil_order_for_version(checkpoint_version)
    nested = {
        "version": nested_version,
        "stencil_order": list(stencil_positions_for_version(nested_version)),
        "derivative_order": nested_order,
        "h_bohr": 0.075,
        "units": "Bohr",
    }
    model = pcPBELMLOptimizerV2Lap(2, 8, use_g_x=True, use_g_c=True).double()
    payload = checkpoint_payload(model, lap_s5_provenance={"stencil": nested})
    payload["stencil_version"] = checkpoint_version
    payload["derivative_order"] = checkpoint_order
    payload["stencil_order"] = list(stencil_positions_for_version(checkpoint_version))
    checkpoint = tmp_path / "mismatched.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="metadata disagree"):
        load_lap_checkpoint(checkpoint)


def test_builder_exposes_version_option_and_keeps_legacy_default(monkeypatch):
    from build_mrks_stencils import parse_args

    required = [
        "--legacy-pickle",
        "legacy.pickle",
        "--npz-root",
        "npz",
        "--output-dir",
        "out",
        "--h-bohr",
        "0.075",
    ]
    monkeypatch.setattr(sys, "argv", ["build_mrks_stencils.py", *required])
    assert parse_args().stencil_version == STENCIL_VERSION
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mrks_stencils.py",
            *required,
            "--stencil-version",
            STENCIL_VERSION_13_POINT_FOURTH_ORDER,
        ],
    )
    assert parse_args().stencil_version == STENCIL_VERSION_13_POINT_FOURTH_ORDER
