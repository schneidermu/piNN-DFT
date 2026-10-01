"""Versioned reference-density stencils; reject legacy center-only Vrho data."""

import json
import math
import pickle
import re
import shutil
from numbers import Integral
from pathlib import Path

import h5py
import torch
from lap_vxc import (
    STENCIL_DERIVATIVE_ORDER,
    STENCIL_ORDER,
    STENCIL_VERSION,
    stencil_coordinates,
    stencil_order_for_version,
    stencil_positions_for_version,
    validate_stencil_selection,
)
from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL

PROTOCOL = "diet-clean-mn-all-mrks-lap-fullvxc-v1"
DEFAULT_CORPUS = "checkpoints_dietclean_lap_fullvxc_v1"
_LEGACY_STENCIL_POSITIONS = list(STENCIL_ORDER)
MISSING_SOURCE = (
    "Full mRKS Vxc requires independently evaluated (N,n_offsets,10) rho/spin-gradient-vector/"
    "Laplacian stencils. Center-only Grid/Vrho data cannot supply this. Provide the "
    "original mRKS generator plus reference spin AO density matrices and exact molecule/"
    "basis/AO ordering (or a wavefunction evaluator), and unchanged full Vxc/E_xc targets. "
    "Nearest-neighbor interpolation and inferred gradient directions are forbidden."
)


def normalize_target(v, features, source_spin, target_kind):
    n = len(features)
    if target_kind == "common-rks":
        if source_spin != 0 or not torch.allclose(
            features[..., 0], features[..., 1], rtol=1e-8, atol=1e-12
        ):
            raise ValueError(
                "One-channel Vxc requires verified closed-shell source and equal spin densities."
            )
        if v.shape == (n,):
            v = v[:, None].repeat(1, 2)
        elif v.shape != (n, 2) or not torch.allclose(
            v[:, 0], v[:, 1], rtol=1e-8, atol=1e-12
        ):
            raise ValueError("Common RKS Vxc must have identical channels.")
    elif target_kind != "spin-resolved":
        raise ValueError("TargetKind must be common-rks or spin-resolved.")
    if v.shape != (n, 2):
        raise ValueError(
            "Full Vxc target must be (N,2); arbitrary spin averaging is forbidden."
        )
    return v


def validate_record(d):
    required = {
        "Name",
        "Coordinates",
        "LegacyCoordinates",
        "StencilCoordinates",
        "StencilFeatures",
        "Weights",
        "Vxc",
        "E_xc",
        "Protocol",
        "StencilVersion",
        "HBohr",
        "SourceSpin",
        "TargetKind",
        "GaugeMetadata",
        "SourceProvenance",
    }
    if not required <= d.keys() or "Vrho" in d:
        raise ValueError(MISSING_SOURCE)
    if d["Protocol"] != PROTOCOL:
        raise ValueError("Stale/unsupported Lap full-Vxc protocol.")
    version = d["StencilVersion"]
    order = d.get("StencilOrder")
    # Pre-versioned-order 7-point H5/pickle records carried this exact version
    # identifier. Keep those readable by mapping the identifier itself to its
    # registered order; never inspect tensor shape to choose a version.
    if "StencilOrder" not in d and version == STENCIL_VERSION:
        order = stencil_order_for_version(version)
        d["StencilOrder"] = order
    stencil = validate_stencil_selection(order, version)
    if not isinstance(d["Name"], str) or not d["Name"] or d["SourceSpin"] not in (0, 1):
        raise ValueError("Invalid system name/source-spin metadata.")
    h = d["HBohr"]
    if not isinstance(h, (float, int)) or not math.isfinite(h) or h <= 0:
        raise ValueError(
            "Positive finite HBohr is required; no production default exists."
        )
    n = len(d["Coordinates"])
    shapes = {
        "Coordinates": (n, 3),
        "LegacyCoordinates": (n, 3),
        "StencilCoordinates": (n, stencil.point_count, 3),
        "StencilFeatures": (n, stencil.point_count, 10),
        "Weights": (n,),
        "Vxc": (n, 2),
        "E_xc": (),
    }
    for key, shape in shapes.items():
        t = d[key]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.float64
            or t.shape != shape
            or not torch.isfinite(t).all()
        ):
            raise ValueError(f"{key} must be finite float64 with shape {shape}.")
    if n == 0 or (d["Weights"] < 0).any() or (d["StencilFeatures"][..., :2] < 0).any():
        raise ValueError("Empty grid or negative weights/density.")
    expected = stencil_coordinates(
        d["Coordinates"], h, order=order, version=version
    )
    # Reconstruct identically from the recorded centers and h; no nearest neighbors.
    if not torch.equal(expected, d["StencilCoordinates"]):
        raise ValueError(
            "Stencil coordinates do not exactly match recorded Cartesian offsets/h."
        )
    if not torch.equal(
        d["LegacyCoordinates"].to(torch.float32),
        d["Coordinates"].to(torch.float32),
    ):
        raise ValueError(
            "Recovered source centers do not retain exact legacy float32 coordinate identity."
        )
    if (d["StencilFeatures"][:, 0, :2] * d["Weights"][:, None]).sum() <= 0:
        raise ValueError("Nonpositive integrated electron number.")
    normalize_target(d["Vxc"], d["StencilFeatures"], d["SourceSpin"], d["TargetKind"])
    if not isinstance(d["GaugeMetadata"], dict) or not isinstance(
        d["SourceProvenance"], dict
    ):
        raise ValueError(  # noqa: TRY004 -- schema violations consistently raise ValueError
            "Explicit gauge/source metadata dictionaries are required (may report unavailable)."
        )
    if not isinstance(d["GaugeMetadata"].get("available"), bool):
        raise ValueError(  # noqa: TRY004 -- missing availability is a schema violation
            "GaugeMetadata.available must explicitly report alignment metadata availability."
        )
    provenance = d["SourceProvenance"]
    if not all(
        provenance.get(key)
        for key in (
            "generator",
            "reference_density",
            "molecule_basis_ao_order",
            "full_vxc_target",
        )
    ):
        raise ValueError(
            "Missing original generator/reference-density/basis-order/full-target provenance."
        )
    for key in ("npz_sha256", "legacy_target_sha256"):
        digest = provenance.get(key)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Missing or invalid SHA-256 provenance field: {key}.")
    if provenance.get("E_xc_source") != "preserved legacy mRKS training target":
        raise ValueError(
            "The full-Vxc corpus must preserve the historical E_xc target."
        )
    if provenance.get("NPZ_exc_wf_substituted") is not False:
        raise ValueError("NPZ exc_wf may not be substituted for the historical E_xc.")
    return d


def require_full_center_verification(d):
    """Require a builder-measured numeric tie to every legacy central row.

    This check is for production corpus assembly.  Development H5 files may
    carry a deterministic dense sample, but those records cannot be promoted
    into the 90-system corpus until every legacy center has been reconstructed
    and compared with the old rho/sigma/lapl columns.
    """
    provenance = d["SourceProvenance"]
    n = provenance.get("legacy_training_points")
    check = provenance.get("central_density_check")
    if not isinstance(check, dict):
        raise ValueError(  # noqa: TRY004 -- invalid source provenance is a data error.
            "Production stencil lacks legacy central-point verification provenance."
        )
    if type(n) is not int:
        # Earlier H5 builds kept the full checked-row count in the numeric
        # center report and independently in exact-coordinate Vxc matching.
        # Accept that equivalent evidence only when all three populations
        # (coordinates, center checks, and matched legacy targets) are equal.
        checked = check.get("points_checked")
        matching = provenance.get("legacy_vxc_matching")
        if not isinstance(matching, dict):
            raise ValueError(
                "Production stencil lacks legacy central-point verification provenance."
            )
        n = checked
        if (
            type(n) is not int
            or matching.get("legacy_points") != n
            or matching.get("matched_legacy_points") != n
            or matching.get("unmatched_legacy_points") != 0
        ):
            raise ValueError(
                "Production stencil lacks legacy central-point verification provenance."
            )
    if n != len(d["Coordinates"]):
        raise ValueError(
            "Production stencil lacks legacy central-point verification provenance."
        )
    if check.get("points_checked") != n:
        raise ValueError(
            "Production stencil did not numerically verify every legacy central point."
        )
    for group in ("rho", "sigma_aa_ab_bb", "lapl"):
        combined = check.get(group, {}).get("combined", {})
        if combined.get("float32_level_compatible") is not True:
            raise ValueError(
                f"Production stencil failed the legacy center {group} comparison."
            )


def evaluate_stencil(
    coords,
    h,
    reference_evaluator,
    chunk_size=4096,
    *,
    order=STENCIL_DERIVATIVE_ORDER,
    version=STENCIL_VERSION,
):
    """Caller supplies ORIGINAL reference-density evaluator, never an SCF surrogate."""
    if reference_evaluator is None:
        raise ValueError(MISSING_SOURCE)
    if chunk_size <= 0 or len(coords) == 0:
        raise ValueError("Positive chunk size and nonempty coordinates are required.")
    stencil = validate_stencil_selection(order, version)
    sc = stencil_coordinates(coords, h, order=order, version=version)
    values = []
    for start in range(0, len(coords), chunk_size):
        xyz = sc[start : start + chunk_size].reshape(-1, 3)
        f = torch.as_tensor(reference_evaluator(xyz.cpu().numpy()), dtype=torch.float64)
        if f.shape != (len(xyz), 10):
            raise ValueError(
                "Reference evaluator must return rho_a,b,grad_a_xyz,grad_b_xyz,lapl_a,b."
            )
        values.append(f.reshape(-1, stencil.point_count, 10))
    return sc, torch.cat(values)


def ao_reference_evaluator(mol, spin_density_matrices):
    """Exact AO evaluation of PROVIDED reference matrices, no new SCF calculation.

    Matrices must belong to mol's exact basis/AO order and describe the reference
    density. This adapter does not manufacture or validate mRKS provenance.
    """
    import numpy as np
    from pyscf.dft import numint

    dm = np.asarray(spin_density_matrices, dtype=np.float64)
    if dm.shape != (2, mol.nao_nr(), mol.nao_nr()) or not np.isfinite(dm).all():
        raise ValueError(
            "Two reference spin AO density matrices matching the exact molecule are required."
        )
    if not np.allclose(dm, dm.transpose(0, 2, 1), atol=1e-12):
        raise ValueError("Reference density matrices must be real symmetric.")

    def evaluate(coords):
        ao = numint.eval_ao(mol, coords, deriv=2)
        r = [
            numint.eval_rho(mol, ao, spin_dm, xctype="MGGA", hermi=1, with_lapl=True)
            for spin_dm in dm
        ]
        return np.column_stack(
            [r[0][0], r[1][0], r[0][1:4].T, r[1][1:4].T, r[0][4], r[1][4]]
        )

    return evaluate


def read_stencil_h5(path):
    with h5py.File(path, "r") as f:
        names = {
            "Coordinates": "coords",
            "LegacyCoordinates": "legacy_coords",
            "StencilCoordinates": "stencil_coords",
            "StencilFeatures": "stencil_features",
            "Weights": "weights",
            "Vxc": "vxc",
            "E_xc": "E_xc",
        }
        if not all(v in f for v in names.values()):
            raise ValueError(f"{path}: {MISSING_SOURCE}")
        if any(
            f[v].dtype.kind != "f" or f[v].dtype.itemsize != 8 for v in names.values()
        ):
            raise ValueError(
                "Versioned stencil H5 fields must be float64; no silent precision promotion."
            )
        version = f.attrs.get("stencil_version")
        order = f.attrs.get("derivative_order")
        if order is None and version == STENCIL_VERSION:
            # Historical 7-point files used the immutable version ID but did
            # not yet persist the numeric order attribute.
            order = stencil_order_for_version(version)
        declared_positions = f.attrs.get("stencil_order")
        if declared_positions is not None:
            try:
                declared_positions = json.loads(declared_positions)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid H5 stencil_order offset layout.") from exc
            if declared_positions != list(stencil_positions_for_version(version)):
                raise ValueError("H5 stencil_order layout disagrees with its version.")
        if order is not None:
            if isinstance(order, bool) or not isinstance(order, Integral):
                raise ValueError("H5 derivative_order must be an integer.")
            order = int(order)
        d = {
            k: torch.as_tensor(f[v][()], dtype=torch.float64) for k, v in names.items()
        }
        d.update(
            Name=Path(path).stem,
            Protocol=f.attrs.get("protocol"),
            StencilVersion=version,
            StencilOrder=int(order) if order is not None else None,
            HBohr=float(f.attrs.get("h_bohr", float("nan"))),
            SourceSpin=int(f.attrs.get("source_spin", -1)),
            TargetKind=f.attrs.get("target_kind"),
            GaugeMetadata=json.loads(f.attrs.get("gauge_metadata", "{}")),
            SourceProvenance=json.loads(f.attrs.get("source_provenance", "{}")),
        )
        # Explicit axes preserve channels even for ambiguous 2x2 sources.
        layout = f.attrs.get("vxc_layout")
        if layout == "spin,point":
            d["Vxc"] = d["Vxc"].T.contiguous()
        elif layout != "point,spin" and not (
            layout == "point" and d["TargetKind"] == "common-rks"
        ):
            raise ValueError(
                "Explicit vxc_layout=spin,point / point,spin / point is required."
            )
        d["Vxc"] = normalize_target(
            d["Vxc"], d["StencilFeatures"], d["SourceSpin"], d["TargetKind"]
        )
    return validate_record(d)


def write_stencil_h5(path, record):
    """Write a validated original-reference record; never overwrite a source.

    Use evaluate_stencil with the original reference evaluator, then supply
    unchanged central Vxc/E_xc/weights and explicit provenance. No density,
    target, spin identity, or alignment is inferred here.
    """
    d = validate_record(record)
    if Path(path).stem != d["Name"]:
        raise ValueError("Stencil H5 filename must preserve the original system Name.")
    with h5py.File(path, "x") as f:
        names = {
            "Coordinates": "coords",
            "LegacyCoordinates": "legacy_coords",
            "StencilCoordinates": "stencil_coords",
            "StencilFeatures": "stencil_features",
            "Weights": "weights",
            "Vxc": "vxc",
            "E_xc": "E_xc",
        }
        for name, field in names.items():
            f[field] = d[name].detach().cpu().numpy()
        f.attrs.update(
            protocol=PROTOCOL,
            stencil_version=d["StencilVersion"],
            derivative_order=d["StencilOrder"],
            stencil_order=json.dumps(
                list(stencil_positions_for_version(d["StencilVersion"]))
            ),
            h_bohr=d["HBohr"],
            source_spin=d["SourceSpin"],
            target_kind=d["TargetKind"],
            vxc_layout="point,spin",
            gauge_metadata=json.dumps(d["GaugeMetadata"]),
            source_provenance=json.dumps(d["SourceProvenance"]),
        )


def build_corpus(mn_corpus, stencil_dir, output_dir):
    from launch_provenance import current_commit
    from prepare_training_corpus import sha256
    from prepare_training_corpus import verify as verify_mn

    output, sources = Path(output_dir), sorted(Path(stencil_dir).glob("*.h5"))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corpus {output}")
    if not sources:
        raise ValueError(MISSING_SOURCE)
    # Validate ALL stencils before creating any output directory.
    records = [read_stencil_h5(p) for p in sources]
    if len(records) != 90 or len({d["Name"] for d in records}) != 90:
        raise ValueError("Production Lap corpus requires all 90 unique mRKS systems.")
    for record in records:
        require_full_center_verification(record)
    if len({d["HBohr"] for d in records}) != 1:
        raise ValueError("All systems must use the same explicit finite-difference h.")
    if len({(d["StencilOrder"], d["StencilVersion"]) for d in records}) != 1:
        raise ValueError(
            "All systems must use the same explicit stencil order/version."
        )
    derivative_order = records[0]["StencilOrder"]
    stencil_version = records[0]["StencilVersion"]
    stencil_order = list(stencil_positions_for_version(stencil_version))
    mn = verify_mn(Path(mn_corpus))
    output.mkdir(parents=True)
    for name in (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "minnesota_protocol.json",
    ):
        shutil.copyfile(Path(mn_corpus) / name, output / name)
    artifact = output / "data_full_vxc_train.pickle"
    names = (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "minnesota_protocol.json",
        artifact.name,
    )
    mn_manifest = Path(mn_corpus) / "preprocessing_manifest.json"
    mn_manifest_copy = output / "minnesota_preprocessing_manifest.json"
    shutil.copyfile(mn_manifest, mn_manifest_copy)
    source_h5_manifest = []
    for path, record in zip(sources, records):
        digest = sha256(path)
        record["SourceProvenance"]["generated_stencil_h5_name"] = path.name
        record["SourceProvenance"]["generated_stencil_h5_sha256"] = digest
        source_h5_manifest.append(
            {
                "name": record["Name"],
                "filename": path.name,
                "sha256": digest,
                "stencil_version": record["StencilVersion"],
                "derivative_order": record["StencilOrder"],
                "stencil_order": list(
                    stencil_positions_for_version(record["StencilVersion"])
                ),
            }
        )
    # Re-serialize after binding every record to its exact generated stencil.
    with artifact.open("wb") as handle:
        pickle.dump(records, handle)
    manifest = {
        "manifest_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "descriptor_protocol": DESCRIPTOR_PROTOCOL,
        "git_commit": current_commit(Path(__file__).parent.parent),
        "minnesota_source_reactions": mn["minnesota_source_reactions"],
        "minnesota_training_reactions": mn["minnesota_training_reactions"],
        "excluded_minnesota_reactions": mn["excluded_minnesota_reactions"],
        "augmented_reaction_samples": mn["augmented_reaction_samples"],
        "mrks_systems": 90,
        "stencil_version": stencil_version,
        "stencil_order": stencil_order,
        "derivative_order": derivative_order,
        "h_bohr": records[0]["HBohr"],
        "units": "Bohr",
        "systems": [
            {
                "name": d["Name"],
                "target_kind": d["TargetKind"],
                "source_spin": d["SourceSpin"],
                "gauge_metadata": d["GaugeMetadata"],
                "source_provenance": d["SourceProvenance"],
            }
            for d in records
        ],
        "source_h5_identities": source_h5_manifest,
        "stencil_data_sha256": sha256(artifact),
        "artifact_sha256": {n: sha256(output / n) for n in names},
        "minnesota_manifest_source_path": str(mn_manifest.resolve()),
        "minnesota_manifest_sha256": sha256(mn_manifest_copy),
    }
    (output / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    verify_corpus(output)
    return output


def verify_corpus(directory):
    from prepare_training_corpus import exclusion_pairs, sha256

    directory = Path(directory)
    m = json.loads((directory / "preprocessing_manifest.json").read_text())
    if (
        m.get("protocol"),
        m.get("architecture"),
        m.get("descriptor_protocol"),
    ) != (PROTOCOL, ARCHITECTURE, DESCRIPTOR_PROTOCOL):
        raise ValueError(
            "Architecture/full-Vxc protocol mismatch; legacy corpus rejected."
        )
    manifest_order = m.get("stencil_order")
    derivative_order = m.get("derivative_order")
    if (
        m.get("stencil_version") == STENCIL_VERSION
        and manifest_order == _LEGACY_STENCIL_POSITIONS
        and derivative_order is None
    ):
        # Early manifests persisted the exact 7-point position sequence. Its
        # immutable version ID determines second order; no tensor shape is read.
        derivative_order = stencil_order_for_version(m["stencil_version"])
        m["derivative_order"] = derivative_order
    try:
        validate_stencil_selection(derivative_order, m.get("stencil_version"))
        expected_positions = list(stencil_positions_for_version(m["stencil_version"]))
        if manifest_order != expected_positions:
            raise ValueError("Manifest offset layout disagrees with stencil version.")
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported corpus stencil order/version.") from exc
    if m.get("manifest_version") != 1 or m.get("units") != "Bohr":
        raise ValueError("Invalid stencil manifest/version/units.")
    source_manifest = directory / "minnesota_preprocessing_manifest.json"
    if not source_manifest.is_file() or sha256(source_manifest) != m.get(
        "minnesota_manifest_sha256"
    ):
        raise ValueError("Minnesota source manifest provenance mismatch.")
    mn = json.loads(source_manifest.read_text())
    for key in (
        "minnesota_source_reactions",
        "minnesota_training_reactions",
        "excluded_minnesota_reactions",
        "augmented_reaction_samples",
    ):
        if m.get(key) != mn.get(key):
            raise ValueError(f"Minnesota source scientific metadata mismatch: {key}")
    for name in (
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "minnesota_protocol.json",
    ):
        if m.get("artifact_sha256", {}).get(name) != mn.get("artifact_sha256", {}).get(
            name
        ):
            raise ValueError(f"Minnesota source artifact mismatch: {name}")
    if (
        m.get("excluded_minnesota_reactions") != exclusion_pairs()
        or m.get("minnesota_source_reactions") != 284
        or m.get("minnesota_training_reactions") != 268
    ):
        raise ValueError("Minnesota exclusion/count contract changed.")
    artifacts = {
        "data_predopt.pickle",
        "data_train_grouped.pickle",
        "minnesota_protocol.json",
        "data_full_vxc_train.pickle",
    }
    if set(m.get("artifact_sha256", {})) != artifacts:
        raise ValueError("Incomplete immutable artifact hashes.")
    for name, digest in m["artifact_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if (
        m.get("stencil_data_sha256")
        != m["artifact_sha256"]["data_full_vxc_train.pickle"]
    ):
        raise ValueError("Stencil data hash disagrees with artifact hash.")
    sources = m.get("source_h5_identities", [])
    if len(sources) != 90 or len({item.get("name") for item in sources}) != 90:
        raise ValueError("All 90 source H5 hashes are required.")
    source_h5_by_name = {}
    for item in sources:
        digest = item.get("sha256")
        if (
            not item.get("filename")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or (
                "stencil_version" in item
                and item["stencil_version"] != m["stencil_version"]
            )
            or (
                "stencil_order" in item
                and item["stencil_order"] != m["stencil_order"]
            )
            or (
                "derivative_order" in item
                and item["derivative_order"] != m["derivative_order"]
            )
        ):
            raise ValueError("Invalid generated stencil source identity/hash.")
        source_h5_by_name[item["name"]] = item
    if any(
        (directory / name).exists()
        for name in (
            "data_vxc_train.pickle",
            "data_vxc_val.pickle",
            "data_test_grouped.pickle",
            "data_val_grouped.pickle",
            "data_full_vxc_val.pickle",
        )
    ):
        raise ValueError(
            "Legacy potential/internal validation artifacts are forbidden."
        )
    with (directory / "data_full_vxc_train.pickle").open("rb") as f:
        ds = pickle.load(f)
    if (
        len(ds) != 90
        or m.get("mrks_systems") != 90
        or len({d["Name"] for d in ds}) != 90
    ):
        raise ValueError("Incomplete/duplicate mRKS system set.")
    for d, metadata in zip(ds, m["systems"]):
        validate_record(d)
        expected = {
            "name": d["Name"],
            "target_kind": d["TargetKind"],
            "source_spin": d["SourceSpin"],
            "gauge_metadata": d["GaugeMetadata"],
            "source_provenance": d["SourceProvenance"],
        }
        if (
            expected != metadata
            or d["HBohr"] != m["h_bohr"]
            or d["StencilOrder"] != m["derivative_order"]
            or d["StencilVersion"] != m["stencil_version"]
        ):
            raise ValueError(
                "Manifest scientific metadata differs from the actual stencil records."
            )
        source_identity = source_h5_by_name.get(d["Name"])
        provenance = d["SourceProvenance"]
        if (
            source_identity is None
            or provenance.get("generated_stencil_h5_name")
            != source_identity.get("filename")
            or provenance.get("generated_stencil_h5_sha256")
            != source_identity.get("sha256")
        ):
            raise ValueError("Stencil record is not tied to its generated H5 hash.")
    if (
        len(m["systems"]) != 90
        or not m.get("git_commit")
        or m.get("augmented_reaction_samples", 0) <= 0
    ):
        raise ValueError("Incomplete scientific provenance.")
    return m, ds


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mn-corpus", default="checkpoints_dietclean_noval_v1")
    parser.add_argument("--stencil-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_CORPUS)
    args = parser.parse_args()
    build_corpus(args.mn_corpus, args.stencil_dir, args.output_dir)


if __name__ == "__main__":
    main()
