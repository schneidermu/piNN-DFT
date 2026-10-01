"""Compare Cartesian Vxc stencils with analytic PBE on real mRKS centers.

This is a diagnostic only. It reads the legacy center population and the NPZ
``dm_ks`` matrix; it writes summaries to an explicit path outside the repo.
No target fitting, interpolation, corpus generation, or model training occurs.

Run from the repository root under the WSL PySCF environment, for example::

    PYTHONPATH=train_models python train_models/run_lap_stencil_validation.py \
      --legacy-pickle /path/data_vxc_train.pickle \
      --npz-root /path/mrks_90_ccsd_pt \
      --output-json /path/lap_stencil_runs/real_pbe_h_sweep.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_mrks_stencils import (
    _make_mol,
    exact_coordinate_lookup,
    normalize_name,
    sha256,
)
from build_mrks_stencils import (
    metrics as target_metrics,
)
from lap_data import ao_reference_evaluator, evaluate_stencil
from lap_stencil_operators import (
    STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1,
    STENCIL_VERSION_13_POINT_FOURTH_ORDER,
)
from lap_stencil_reference import (
    ao_spin_density_derivatives,
    pbe_euler_potential,
    sigma_from_gradient,
)
from lap_vxc import euler_components

from dft_functionals import PBE
from dft_functionals.constants import PBE_CONSTANTS

H_VALUES = (0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025)
REQUIRED_SYSTEMS = ("H2", "BeH2", "CO", "N2", "ClHS")
DEFAULT_SYSTEMS = (
    "H2",
    "BH",
    "ClH",
    "BeH2",
    "H2O",
    "CO",
    "N2",
    "CH4",
    "AlBeH",
    "CH2O",
    "H4Si",
    "ClHS",
    "C2H2_iso2",
)
SCHEMES = (
    (2, STENCIL_VERSION_7_POINT_RHO_GRAD_LAPL_V1),
    (4, STENCIL_VERSION_13_POINT_FOURTH_ORDER),
)


class CanonicalPBE(nn.Module):
    """Repository canonical PBE energy density, with exact zero Lap dependence."""

    def forward(self, rho: torch.Tensor, sigma: torch.Tensor, lapl: torch.Tensor):
        constants = PBE_CONSTANTS.to(dtype=rho.dtype, device=rho.device).expand(
            len(rho), -1
        )
        return PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(dim=-1)


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _load_sources(legacy_path: Path, npz_root: Path):
    with legacy_path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, (list, tuple)) or len(records) != 90:
        raise ValueError(
            "The historical legacy pickle must contain exactly 90 systems."
        )
    legacy = {normalize_name(r["Name"]): r for r in records}
    files = sorted(npz_root.glob("*/inp_mrks.npz"))
    if len(files) != 90:
        raise ValueError(f"Expected 90 NPZ source systems, found {len(files)}.")
    npzs = {normalize_name(p.parent.name): p for p in files}
    if len(legacy) != 90 or len(npzs) != 90 or set(legacy) != set(npzs):
        raise ValueError("Legacy and NPZ normalized Name indexes do not match 90/90.")
    return legacy, npzs


def _stratified_indices(
    distance: np.ndarray, density: np.ndarray, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample equally across joint distance/density rank strata with HT weights."""
    n = len(distance)
    if n <= max_points:
        return np.arange(n), np.ones(n), {"population": n, "sampled": n, "strata": 1}
    cells = 4
    # Rank bins are stable when many points have exactly equal density/distance.
    dist_rank = np.argsort(np.argsort(distance, kind="mergesort"), kind="mergesort")
    rho_rank = np.argsort(
        np.argsort(np.log10(np.maximum(density, 1e-300)), kind="mergesort"),
        kind="mergesort",
    )
    dist_bin = np.minimum(cells - 1, dist_rank * cells // n)
    rho_bin = np.minimum(cells - 1, rho_rank * cells // n)
    labels = dist_bin * cells + rho_bin
    nonempty = [label for label in range(cells * cells) if np.any(labels == label)]
    quota, remainder = divmod(max_points, len(nonempty))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    inverse_probability: list[np.ndarray] = []
    for ordinal, label in enumerate(nonempty):
        members = np.flatnonzero(labels == label)
        take = min(len(members), quota + (ordinal < remainder))
        local = np.sort(rng.choice(members, size=take, replace=False))
        chosen.append(local)
        inverse_probability.append(np.full(take, len(members) / take, dtype=np.float64))
    indices = np.concatenate(chosen)
    weights = np.concatenate(inverse_probability)
    order = np.argsort(indices)
    return (
        indices[order],
        weights[order],
        {
            "population": n,
            "sampled": len(indices),
            "strata": len(nonempty),
            "distance_bins": cells,
            "density_bins": cells,
            "selection": "deterministic joint rank strata; inverse-inclusion weights applied to metrics",
        },
    )


def _weighted_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    quadrature_weight: np.ndarray,
    inclusion_weight: np.ndarray,
    density: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    error = act - ref
    sampling_q = np.asarray(quadrature_weight, dtype=np.float64) * np.asarray(
        inclusion_weight, dtype=np.float64
    )
    q = sampling_q[:, None] * np.asarray(density, dtype=np.float64)
    total = float(q.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(
            "Density-weighted analytic-PBE metric has invalid normalization."
        )
    return {
        "unweighted_rmse": float(np.sqrt(np.mean(error * error))),
        "unweighted_mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "density_weighted_rmse": float(np.sqrt((q * error * error).sum() / total)),
        "density_weighted_mae": float((q * np.abs(error)).sum() / total),
        "density_weighted_bias": float((q * error).sum() / total),
    }


def _component_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def _source_geometry(npz: Any):
    mol, basis, charge, spin = _make_mol(npz)
    dm = np.asarray(npz["dm_ks"], dtype=np.float64)
    if dm.shape != (mol.nao_nr(), mol.nao_nr()):
        raise ValueError("dm_ks AO dimensions do not match the reconstructed molecule.")
    spin_dm = np.stack((dm / 2, dm / 2))
    atom_coords = np.asarray(npz["atom_coords"], dtype=np.float64)
    atom_charges = np.asarray(npz["atom_charges"], dtype=np.int64)
    return mol, dm, spin_dm, atom_coords, atom_charges, basis, charge, spin


def _distance_density_bins(
    error: np.ndarray,
    nearest_distance: np.ndarray,
    density: np.ndarray,
    quadrature_weight: np.ndarray,
    inclusion_weight: np.ndarray,
) -> list[dict[str, Any]]:
    """Weighted error summaries for joint nearest-nucleus/density quartiles."""
    d_rank = np.argsort(
        np.argsort(nearest_distance, kind="mergesort"), kind="mergesort"
    )
    n = len(nearest_distance)
    d_bin = np.minimum(3, 4 * d_rank // n)
    r_rank = np.argsort(
        np.argsort(np.log10(np.maximum(density.sum(-1), 1e-300)), kind="mergesort"),
        kind="mergesort",
    )
    r_bin = np.minimum(3, 4 * r_rank // n)
    result = []
    for di in range(4):
        for ri in range(4):
            mask = (d_bin == di) & (r_bin == ri)
            if not mask.any():
                continue
            component_error = error[mask]
            q = (quadrature_weight[mask] * inclusion_weight[mask])[:, None] * density[
                mask
            ]
            qsum = float(q.sum())
            result.append(
                {
                    "distance_quartile": di,
                    "density_quartile": ri,
                    "sample_points": int(mask.sum()),
                    "nearest_distance_min_bohr": float(nearest_distance[mask].min()),
                    "nearest_distance_max_bohr": float(nearest_distance[mask].max()),
                    "density_total_min": float(density[mask].sum(-1).min()),
                    "density_total_max": float(density[mask].sum(-1).max()),
                    "density_weighted_rmse": float(
                        np.sqrt((q * component_error**2).sum() / qsum)
                    ),
                    "density_weighted_mae": float(
                        (q * np.abs(component_error)).sum() / qsum
                    ),
                    "density_weighted_bias": float((q * component_error).sum() / qsum),
                }
            )
    return result


def _evaluate_system(
    name: str,
    record: dict[str, Any],
    npz_path: Path,
    legacy_path: Path,
    max_points: int,
    seed: int,
    run_float32: bool,
    h_values: tuple[float, ...],
) -> dict[str, Any]:
    import time as _time

    started = _time.perf_counter()
    with np.load(npz_path, allow_pickle=False) as npz:
        mol, _dm, spin_dm, atom_coords, atom_charges, basis, charge, spin = (
            _source_geometry(npz)
        )
        npz_coords = np.asarray(npz["grid_coords"], dtype=np.float64)
        npz_grid_vxc = np.asarray(npz["vxc_grid"], dtype=np.float64)
        npz_identity = exact_coordinate_lookup(record["Grid"][:, :3], npz_coords)
        centers_all = npz_coords[npz_identity]
        legacy_grid = np.asarray(record["Grid"], dtype=np.float64)
        weights_all = np.asarray(record["Weights"], dtype=np.float64).reshape(-1)
        if len(centers_all) != len(legacy_grid) or len(weights_all) != len(legacy_grid):
            raise ValueError(f"{name}: legacy and NPZ center population mismatch.")
        # The central centers and targets are sourced from legacy data; this
        # exact identity check is used only to recover original float64 coords.
        identity_differences = npz_grid_vxc[npz_identity] - np.asarray(
            record["Vrho"], dtype=np.float64
        ).reshape(-1)
        if np.max(np.abs(identity_differences)) > 1e-5:
            raise ValueError(
                f"{name}: exact-coordinate legacy/NPZ Vxc identity failed."
            )

        nearest = np.linalg.norm(
            centers_all[:, None, :] - atom_coords[None, :, :], axis=-1
        )
        nearest_atom = nearest.argmin(axis=1)
        nearest_distance_all = nearest[np.arange(len(nearest)), nearest_atom]
        density_all = legacy_grid[:, 4:6]
        indices, inclusion, sampling = _stratified_indices(
            nearest_distance_all, density_all.sum(-1), max_points, seed
        )
        centers = centers_all[indices]
        legacy = legacy_grid[indices]
        quadrature = weights_all[indices]
        inclusion = inclusion.astype(np.float64)
        nearest_distance = nearest_distance_all[indices]
        selected_atom = nearest_atom[indices]
        selected_z = atom_charges[selected_atom]

        # Independent central AO/Hessian fields and analytic PBE Euler target.
        center_fields = ao_spin_density_derivatives(mol, spin_dm, centers)
        sigma_center = sigma_from_gradient(
            torch.as_tensor(center_fields.gradient, dtype=torch.float64)
        ).numpy()
        rho_error = center_fields.density - legacy[:, 4:6]
        sigma_error = sigma_center - legacy[:, 6:9]
        lap_error = center_fields.laplacian - legacy[:, 11:13]
        center_metrics = {
            "rho_alpha_beta": target_metrics(legacy[:, 4:6], center_fields.density),
            "sigma_aa_ab_bb": target_metrics(legacy[:, 6:9], sigma_center),
            "lapl_alpha_beta": target_metrics(
                legacy[:, 11:13], center_fields.laplacian
            ),
        }
        if not all(
            item["float32_level_compatible"] for item in center_metrics.values()
        ):
            raise ValueError(
                f"{name}: dm_ks central reconstruction exceeded the old float32 corpus tolerance: "
                f"{center_metrics}"
            )
        center_audit = {
            "sample_points": len(indices),
            "reconstruction_metrics": center_metrics,
            "rho_max_abs": float(np.max(np.abs(rho_error))),
            "rho_rms": float(np.sqrt(np.mean(rho_error**2))),
            "sigma_max_abs": float(np.max(np.abs(sigma_error))),
            "sigma_rms": float(np.sqrt(np.mean(sigma_error**2))),
            "laplacian_max_abs": float(np.max(np.abs(lap_error))),
            "laplacian_rms": float(np.sqrt(np.mean(lap_error**2))),
            "alpha_beta_rho_max_abs": float(
                np.max(
                    np.abs(center_fields.density[:, 0] - center_fields.density[:, 1])
                )
            ),
            "alpha_beta_gradient_max_abs": float(
                np.max(
                    np.abs(center_fields.gradient[:, 0] - center_fields.gradient[:, 1])
                )
            ),
            "alpha_beta_laplacian_max_abs": float(
                np.max(
                    np.abs(
                        center_fields.laplacian[:, 0] - center_fields.laplacian[:, 1]
                    )
                )
            ),
            "exact_float32_coordinate_identity": True,
            "tolerance_reference": "full 90-system source audit previously checked all centers with dm_ks; this sweep independently checks the deterministic stratified sample",
        }
        rho_t = torch.as_tensor(center_fields.density, dtype=torch.float64)
        grad_t = torch.as_tensor(center_fields.gradient, dtype=torch.float64)
        hess_t = torch.as_tensor(center_fields.hessian, dtype=torch.float64)
        analytic = pbe_euler_potential(rho_t, grad_t, hess_t)
        analytic_components = {
            "C": analytic["rho_derivative"].detach().numpy(),
            "minus_div_A": analytic["minus_div_A"].detach().numpy(),
            "lap_B": np.zeros_like(center_fields.density),
            "Vxc": analytic["Vxc"].detach().numpy(),
        }
        evaluator = ao_reference_evaluator(mol, spin_dm)
        energy = CanonicalPBE()

        summaries: list[dict[str, Any]] = []
        predictions: dict[tuple[str, str], dict[float, np.ndarray]] = {}
        for order, version in SCHEMES:
            for h in h_values:
                stencil_start = _time.perf_counter()
                _coords, features = evaluate_stencil(
                    torch.as_tensor(centers, dtype=torch.float64),
                    h,
                    evaluator,
                    chunk_size=256,
                    order=order,
                    version=version,
                )
                stencil_seconds = _time.perf_counter() - stencil_start
                for dtype_name, dtype in (("float64", torch.float64),):
                    feature_tensor = features.to(dtype=dtype)
                    estimated = euler_components(
                        energy,
                        feature_tensor,
                        h,
                        create_graph=False,
                        order=order,
                        version=version,
                    )
                    component_arrays = {
                        key: estimated[key].detach().cpu().numpy()
                        for key in ("C", "minus_div_A", "lap_B", "Vxc")
                    }
                    float64_arrays = component_arrays
                    predictions.setdefault((version, dtype_name), {})[float(h)] = (
                        component_arrays["Vxc"].copy()
                    )
                    target = analytic_components["Vxc"]
                    metrics = _weighted_metrics(
                        target,
                        component_arrays["Vxc"],
                        quadrature,
                        inclusion,
                        center_fields.density,
                    )
                    item = {
                        "system": name,
                        "scheme": version,
                        "derivative_order": order,
                        "h_bohr": float(h),
                        "dtype": dtype_name,
                        "stencil_evaluation_seconds": stencil_seconds,
                        "Vxc": metrics,
                        "components": {
                            key: _component_metrics(
                                analytic_components[key], component_arrays[key]
                            )
                            for key in ("C", "minus_div_A", "lap_B", "Vxc")
                        },
                        "distance_density_bins": _distance_density_bins(
                            component_arrays["Vxc"] - target,
                            nearest_distance,
                            center_fields.density,
                            quadrature,
                            inclusion,
                        ),
                    }
                    abs_error = np.abs(component_arrays["Vxc"] - target)
                    worst = np.unravel_index(np.argmax(abs_error), abs_error.shape)
                    wi, ws = int(worst[0]), int(worst[1])
                    item["worst_point"] = {
                        "sample_index": int(indices[wi]),
                        "spin": "alpha" if ws == 0 else "beta",
                        "coordinate_bohr": centers[wi].tolist(),
                        "nearest_nucleus_distance_bohr": float(nearest_distance[wi]),
                        "nearest_nucleus_Z": int(selected_z[wi]),
                        "rho_alpha_beta": center_fields.density[wi].tolist(),
                        "gradient_alpha_xyz": center_fields.gradient[wi, 0].tolist(),
                        "gradient_beta_xyz": center_fields.gradient[wi, 1].tolist(),
                        "hessian_alpha": center_fields.hessian[wi, 0].tolist(),
                        "hessian_beta": center_fields.hessian[wi, 1].tolist(),
                        "analytic_components": {
                            k: float(analytic_components[k][wi, ws])
                            for k in ("C", "minus_div_A", "lap_B", "Vxc")
                        },
                        "finite_difference_components": {
                            k: float(component_arrays[k][wi, ws])
                            for k in ("C", "minus_div_A", "lap_B", "Vxc")
                        },
                    }
                    summaries.append(item)
                if run_float32 and name in REQUIRED_SYSTEMS:
                    feature_tensor = features.to(dtype=torch.float32)
                    estimated = euler_components(
                        energy,
                        feature_tensor,
                        h,
                        create_graph=False,
                        order=order,
                        version=version,
                    )
                    component_arrays = {
                        key: estimated[key].detach().cpu().numpy()
                        for key in ("C", "minus_div_A", "lap_B", "Vxc")
                    }
                    float32_key = (
                        version,
                        "float32-local-PBE-derivatives+float64-operator",
                    )
                    predictions.setdefault(float32_key, {})[float(h)] = (
                        component_arrays["Vxc"].copy()
                    )
                    summaries.append(
                        {
                            "system": name,
                            "scheme": version,
                            "derivative_order": order,
                            "h_bohr": float(h),
                            "dtype": "float32-local-PBE-derivatives+float64-operator",
                            "Vxc": _weighted_metrics(
                                analytic_components["Vxc"],
                                component_arrays["Vxc"],
                                quadrature,
                                inclusion,
                                center_fields.density,
                            ),
                            "components": {
                                key: _component_metrics(
                                    analytic_components[key], component_arrays[key]
                                )
                                for key in ("C", "minus_div_A", "lap_B", "Vxc")
                            },
                            "float32_vs_float64_local_derivative_difference": {
                                key: _component_metrics(
                                    float64_arrays[key], component_arrays[key]
                                )
                                for key in ("C", "minus_div_A", "lap_B", "Vxc")
                            },
                        }
                    )

        # Add direct h-to-next-smaller comparisons as a diagnostic alongside
        # the independent analytic reference errors. Pairwise agreement alone
        # is not used to select a stable step size.
        descending_h = sorted(set(map(float, h_values)), reverse=True)
        for item in summaries:
            dtype_key = (
                item["scheme"],
                item["dtype"]
                if item["dtype"] == "float64"
                else "float32-local-PBE-derivatives+float64-operator",
            )
            h_index = descending_h.index(item["h_bohr"])
            if h_index + 1 < len(descending_h):
                smaller_h = descending_h[h_index + 1]
                pair = predictions[dtype_key]
                item["pairwise_to_next_smaller_h"] = {
                    "next_smaller_h_bohr": smaller_h,
                    "metrics": _weighted_metrics(
                        pair[smaller_h],
                        pair[item["h_bohr"]],
                        quadrature,
                        inclusion,
                        center_fields.density,
                    ),
                }
            else:
                item["pairwise_to_next_smaller_h"] = None

        # Density/electron information for the tested source system. The
        # quadrature integral is an audit only; no source quantities are changed.
        result = {
            "name": name,
            "npz_path": str(npz_path),
            "npz_sha256": sha256(npz_path),
            "legacy_name": record["Name"],
            "legacy_sha256": sha256(legacy_path),
            "basis": basis,
            "charge": charge,
            "spin": spin,
            "ao_dimension": int(mol.nao_nr()),
            "legacy_points": len(legacy_grid),
            "npz_source_points": len(npz_coords),
            "selected_indices": indices.tolist(),
            "sampling": sampling,
            "central_reconstruction": center_audit,
            "legacy_E_xc": float(np.asarray(record["E_xc"]).reshape(-1)[0]),
            "NPZ_exc_wf_audit_only": float(
                dict(
                    zip(
                        np.asarray(npz["meta_keys"]).tolist(),
                        np.asarray(npz["meta"]).tolist(),
                    )
                ).get("exc_wf", float("nan"))
            ),
            "E_xc_policy": "historical legacy training target is canonical; NPZ exc_wf was not substituted",
            "legacy_integrated_electron_count": float(
                np.sum((legacy_grid[:, 4] + legacy_grid[:, 5]) * weights_all)
            ),
            "finite_difference_summaries": summaries,
            "elapsed_seconds": float(_time.perf_counter() - started),
        }
        if not np.isfinite(result["NPZ_exc_wf_audit_only"]):
            result["NPZ_exc_wf_audit_only"] = None
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-pickle", required=True, type=Path)
    parser.add_argument("--npz-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--h-values", default=",".join(map(str, H_VALUES)))
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    h_values = tuple(float(x) for x in args.h_values.split(","))
    if not h_values or any(not np.isfinite(h) or h <= 0 for h in h_values):
        raise ValueError("Every --h-values item must be positive and finite")
    names = [name.strip() for name in args.systems.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one system is required")
    legacy_path = args.legacy_pickle.resolve()
    npz_root = args.npz_root.resolve()
    output_path = args.output_json.resolve()
    legacy_by_name, npz_by_name = _load_sources(legacy_path, npz_root)
    output: dict[str, Any] = {
        "schema": "lap-stencil-real-pbe-validation-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch_commit": "ad2036799758d6f7dff6b7ef8c94fb020941461c",
        "legacy_source": {
            "path": str(legacy_path),
            "sha256": sha256(legacy_path),
        },
        "npz_root": str(npz_root),
        "density_matrix_source": "dm_ks, split equally into alpha/beta for RKS",
        "target_policy": "analytic PBE reference is independent of mRKS Vxc target; legacy E_xc/Vxc are audit metadata only",
        "precision": {
            "float64": "AO fields and local canonical PBE derivatives in float64; outer operator float64",
            "float32": "AO-generated stencil features cast to float32 for local canonical PBE derivatives, then outer operator accumulates in float64, matching euler_components training behavior",
        },
        "h_values_bohr": list(h_values),
        "schemes": [
            {"version": version, "derivative_order": order}
            for order, version in SCHEMES
        ],
        "required_float32_systems": list(REQUIRED_SYSTEMS) if args.float32 else [],
        "systems": {},
        "status": "running",
    }
    _json_dump(output_path, output)
    for ordinal, name in enumerate(names):
        key = normalize_name(name)
        if key not in legacy_by_name or key not in npz_by_name:
            raise ValueError(
                f"Requested system not present in exact 90-system source set: {name}"
            )
        actual_name = legacy_by_name[key]["Name"]
        print(f"[{ordinal + 1}/{len(names)}] {actual_name}", flush=True)
        result = _evaluate_system(
            actual_name,
            legacy_by_name[key],
            npz_by_name[key],
            legacy_path,
            args.max_points,
            args.seed + ordinal,
            args.float32,
            h_values,
        )
        output["systems"][actual_name] = result
        output["status"] = "running"
        _json_dump(output_path, output)
        print(
            f"  done: {result['elapsed_seconds']:.1f}s; "
            f"central rho/lap max {result['central_reconstruction']['rho_max_abs']:.3e}/"
            f"{result['central_reconstruction']['laplacian_max_abs']:.3e}",
            flush=True,
        )
    output["status"] = "complete"
    output["system_order"] = names
    _json_dump(output_path, output)


if __name__ == "__main__":
    main()
