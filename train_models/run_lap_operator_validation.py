"""Run independent central-grid Lap operator validations and write JSON results.

Recommended runtime is the project's PySCF WSL environment. Example:

    python train_models/run_lap_operator_validation.py \
      --systems H2 BeH2 CO N2 ClH --out-dir /mnt/c/Dev/readWFN_share_ms/lap_operator_runs_20261001

The script uses exact float32 coordinate identity to recover the NPZ float64
coordinates for legacy rows. It never uses displaced coordinates or a spatial
finite difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lap_operator import (
    assemble_rks_operator,
    orthonormalize_operator,
    rks_density_features_from_ao,
)
from lap_stencil_reference import ao_spin_density_derivatives, pbe_euler_potential

from dft_functionals import PBE
from dft_functionals.constants import PBE_CONSTANTS


class CanonicalPBEEnergy(torch.nn.Module):
    """Canonical repository PBE energy density, with no Lap/tau dependence."""

    def forward(self, rho, sigma, lapl):
        del lapl
        constants = PBE_CONSTANTS.to(dtype=rho.dtype, device=rho.device).expand(
            len(rho), -1
        )
        return PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coord_key(row: np.ndarray) -> bytes:
    return np.asarray(row, dtype=np.float32).tobytes()


def _load_legacy_record(records, name: str) -> dict:
    matches = [record for record in records if record.get("Name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one legacy record named {name!r}.")
    return matches[0]


def _load_system(name: str, legacy_record: dict, npz_root: Path) -> dict:
    npz_path = npz_root / name / "inp_mrks.npz"
    with np.load(npz_path, allow_pickle=False) as source:
        source_coords = np.asarray(source["grid_coords"], dtype=np.float64)
        source_weights = np.asarray(source["grid_weights"], dtype=np.float64)
        dm = np.asarray(source["dm_ks"], dtype=np.float64)
        basis = json.loads(str(source["basis"][0]))
        atom_coords = np.asarray(source["atom_coords"], dtype=np.float64)
        atom_charges = np.asarray(source["atom_charges"], dtype=np.int32)
        charge = int(np.asarray(source["mol_charge"]).reshape(-1)[0])
        spin = int(np.asarray(source["mol_spin"]).reshape(-1)[0])
        source_vxc = np.asarray(source["vxc_grid"], dtype=np.float64)

    grid = np.asarray(legacy_record["Grid"].cpu(), dtype=np.float32)
    legacy_xyz32 = grid[:, :3]
    weights = np.asarray(legacy_record["Weights"].cpu(), dtype=np.float64)
    vxc = np.asarray(legacy_record["Vrho"].cpu(), dtype=np.float64)
    if len(legacy_xyz32) != len(weights) or len(weights) != len(vxc):
        raise ValueError(f"{name}: legacy coordinates/weights/Vxc row counts differ.")

    identity = defaultdict(list)
    for index, row in enumerate(source_coords.astype(np.float32)):
        identity[_coord_key(row)].append(index)
    indices = []
    unambiguous_rows = []
    duplicate_rows = 0
    for row in legacy_xyz32:
        candidates = identity.get(_coord_key(row), [])
        if not candidates:
            raise ValueError(f"{name}: legacy row has no exact NPZ float32 coordinate identity.")
        candidate_coords = source_coords[candidates]
        # Some source grids repeat the exact same float64 point with separate
        # quadrature weights. That is a repeated identity, not a spatial guess;
        # retain each legacy row's own weight and require its reference Vxc to
        # agree with every identical-coordinate source row.
        if not np.array_equal(candidate_coords, np.broadcast_to(candidate_coords[0], candidate_coords.shape)):
            raise ValueError(f"{name}: a target float32 coordinate maps to distinct NPZ float64 points.")
        if len(candidates) > 1:
            duplicate_rows += 1
            if np.max(np.abs(source_vxc[candidates] - vxc[len(indices)])) > 1e-6:
                raise ValueError(f"{name}: repeated-coordinate NPZ Vxc values disagree with legacy target.")
        else:
            unambiguous_rows.append(len(indices))
        indices.append(candidates[0])
    indices = np.asarray(indices, dtype=np.int64)
    coords = source_coords[indices]
    weight_abs = float(
        np.max(np.abs(weights[unambiguous_rows] - source_weights[indices[unambiguous_rows]]))
        if unambiguous_rows
        else 0.0
    )
    vxc_abs = float(np.max(np.abs(vxc - source_vxc[indices])))

    from pyscf import gto

    atoms = [(int(z), tuple(xyz)) for z, xyz in zip(atom_charges, atom_coords)]
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
    )
    if mol.nao_nr() != dm.shape[0] or dm.shape != (mol.nao_nr(), mol.nao_nr()):
        raise ValueError(f"{name}: reconstructed basis/AO dimension disagrees with dm_ks.")
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    electrons = float(np.einsum("ij,ji->", dm, overlap))
    expected_electrons = float(np.sum(atom_charges) - charge)
    if abs(electrons - expected_electrons) > 2e-5:
        raise ValueError(
            f"{name}: dm_ks electron count {electrons} != {expected_electrons}."
        )
    return {
        "name": name,
        "npz_path": str(npz_path),
        "npz_sha256": _sha256(npz_path),
        "legacy_points": len(coords),
        "source_points": len(source_coords),
        "matched_rows": len(indices),
        "unmatched_rows": 0,
        "exact_repeated_coordinate_rows": duplicate_rows,
        "weight_identity_checked_rows": len(unambiguous_rows),
        "legacy_weight_npz_match_max_abs": weight_abs,
        "legacy_vxc_npz_match_max_abs": vxc_abs,
        "dm_electron_count": electrons,
        "expected_electron_count": expected_electrons,
        "mol": mol,
        "dm": dm,
        "coords": coords,
        "weights": weights,
        "source_coords_full": source_coords,
        "source_weights_full": source_weights,
        "vxc_target": vxc,
        "source_vxc_full": source_vxc,
        "overlap": overlap,
    }


def _torch_array(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float64), dtype=torch.float64)


def _evaluate_system(
    system: dict,
    chunk_size: int,
    *,
    coords: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    grid_kind: str = "legacy-training-subset",
) -> dict:
    from pyscf.dft import numint

    name = system["name"]
    mol, dm = system["mol"], system["dm"]
    coords = system["coords"] if coords is None else coords
    weights = system["weights"] if weights is None else weights
    spin_dm = np.stack([0.5 * dm, 0.5 * dm])
    energy = CanonicalPBEEnergy()
    weak = np.zeros_like(system["overlap"])
    strong = np.zeros_like(weak)
    spin_vxc_maxdiff = 0.0
    common_vxc_blocks = []
    rho_total_blocks = []
    start_time = time.perf_counter()

    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        xyz = coords[start:stop]
        w_np = weights[start:stop]
        ao = np.asarray(numint.eval_ao(mol, xyz, deriv=2), dtype=np.float64)
        phi_np = ao[0]
        grad_np = ao[1:4].transpose(1, 0, 2).copy()
        lap_np = (ao[4] + ao[7] + ao[9]).copy()
        phi, grad_phi, lap_phi, w = map(
            _torch_array, (phi_np, grad_np, lap_np, w_np)
        )
        dm_t = _torch_array(dm)
        features = rks_density_features_from_ao(phi, grad_phi, lap_phi, dm_t)
        block = assemble_rks_operator(
            energy,
            features,
            w,
            phi,
            grad_phi,
            lap_phi,
            chunk_size=min(512, len(xyz)),
            create_graph=False,
        )
        weak += block.detach().numpy()

        fields = ao_spin_density_derivatives(mol, spin_dm, xyz)
        field_t = [
            torch.as_tensor(x, dtype=torch.float64)
            for x in (fields.density, fields.gradient, fields.hessian)
        ]
        v_spin = pbe_euler_potential(*field_t)["Vxc"].cpu().numpy()
        spin_vxc_maxdiff = max(
            spin_vxc_maxdiff, float(np.max(np.abs(v_spin[:, 0] - v_spin[:, 1])))
        )
        v_common = 0.5 * (v_spin[:, 0] + v_spin[:, 1])
        common_vxc_blocks.append(v_common)
        rho_total_blocks.append(fields.density.sum(axis=1))
        strong += np.einsum(
            "g,gi,gj->ij", w_np * v_common, phi_np, phi_np, optimize=True
        )

    delta = weak - strong
    abs_fro = float(np.linalg.norm(delta, ord="fro"))
    strong_fro = float(np.linalg.norm(strong, ord="fro"))
    relative_fro = abs_fro / max(strong_fro, np.finfo(np.float64).tiny)
    max_abs = float(np.max(np.abs(delta)))
    s_t = _torch_array(system["overlap"])
    delta_orth = orthonormalize_operator(_torch_array(delta), s_t).numpy()
    strong_orth = orthonormalize_operator(_torch_array(strong), s_t).numpy()
    orth_fro = float(np.linalg.norm(delta_orth, ord="fro"))
    orth_reference_fro = float(np.linalg.norm(strong_orth, ord="fro"))
    orth_relative = orth_fro / max(orth_reference_fro, np.finfo(np.float64).tiny)
    common_vxc = np.concatenate(common_vxc_blocks)
    rho_total = np.concatenate(rho_total_blocks)
    target_density_slices = {}
    target_vxc = None
    if grid_kind == "complete-npz-source-grid":
        target_vxc = system["source_vxc_full"]
    elif len(coords) == len(system["vxc_target"]):
        target_vxc = system["vxc_target"]
    if target_vxc is not None:
        target_diff = common_vxc - target_vxc
        target_abs = np.abs(target_diff)
        for label, mask in (
            ("rho_lt_1e-12", rho_total < 1e-12),
            ("rho_1e-12_to_1e-8", (rho_total >= 1e-12) & (rho_total < 1e-8)),
            ("rho_1e-8_to_1e-4", (rho_total >= 1e-8) & (rho_total < 1e-4)),
            ("rho_ge_1e-4", rho_total >= 1e-4),
        ):
            if np.any(mask):
                target_density_slices[label] = {
                    "count": int(mask.sum()),
                    "max_abs_target_difference_Ha": float(target_abs[mask].max()),
                    "rms_target_difference_Ha": float(np.sqrt(np.mean(target_diff[mask] ** 2))),
                }
        target_max = float(target_abs.max())
        target_rms = float(np.sqrt(np.mean(target_diff**2)))
    else:
        target_max = None
        target_rms = None
    return {
        "name": name,
        "grid_kind": grid_kind,
        "n_ao": mol.nao_nr(),
        "legacy_points": len(coords),
        "exact_repeated_coordinate_rows": system["exact_repeated_coordinate_rows"],
        "weight_identity_checked_rows": system["weight_identity_checked_rows"],
        "source_points": system["source_points"],
        "elapsed_seconds": time.perf_counter() - start_time,
        "legacy_weight_npz_match_max_abs": system["legacy_weight_npz_match_max_abs"],
        "legacy_vxc_npz_match_max_abs": system["legacy_vxc_npz_match_max_abs"],
        "dm_electron_count": system["dm_electron_count"],
        "spin_potential_max_abs_difference": spin_vxc_maxdiff,
        "strong_potential_vs_legacy_target_max_abs_Ha": target_max,
        "strong_potential_vs_legacy_target_rms_Ha": target_rms,
        "strong_potential_vs_legacy_target_density_slices": target_density_slices,
        "weak_strong_abs_frobenius_Ha": abs_fro,
        "weak_strong_relative_frobenius": relative_fro,
        "weak_strong_max_abs_element_Ha": max_abs,
        "orthonormalized_abs_frobenius_Ha": orth_fro,
        "orthonormalized_relative_frobenius": orth_relative,
        "weak_matrix_frobenius_Ha": float(np.linalg.norm(weak, ord="fro")),
        "strong_matrix_frobenius_Ha": strong_fro,
        "weak_matrix": weak.tolist(),
        "strong_matrix": strong.tolist(),
    }


def _evaluate_pyscf_parity(system: dict, chunk_size: int) -> dict:
    """Compare the Torch Lap operator to the real custom PySCF RKS integrator."""
    from lap_vxc import LapEnergy
    from pyscf.dft import numint
    from test_lap import small_model

    from test_models.DFT.lap_functional import LapFunctional

    mol, dm = system["mol"], system["dm"]
    coords, weights = system["coords"], system["weights"]
    functional = LapFunctional(small_model())
    mf = functional.make_rks(mol)
    mf.max_memory = 128
    # Reuse the exact source central coordinates and legacy quadrature weights.
    mf.grids.coords = coords
    mf.grids.weights = weights
    mf.grids.non0tab = None

    energy = LapEnergy(functional.model)
    matrix_torch = np.zeros_like(system["overlap"])
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        ao = np.asarray(numint.eval_ao(mol, coords[start:stop], deriv=2), dtype=np.float64)
        phi = _torch_array(ao[0])
        grad_phi = _torch_array(ao[1:4].transpose(1, 0, 2).copy())
        lap_phi = _torch_array((ao[4] + ao[7] + ao[9]).copy())
        w = _torch_array(weights[start:stop])
        features = rks_density_features_from_ao(phi, grad_phi, lap_phi, _torch_array(dm))
        matrix_torch += assemble_rks_operator(
            energy,
            features,
            w,
            phi,
            grad_phi,
            lap_phi,
            chunk_size=min(512, stop - start),
            create_graph=False,
        ).detach().numpy()

    _, _, matrix_pyscf = mf._numint.nr_rks(
        mol, mf.grids, mf.xc, dm, max_memory=128
    )
    delta = matrix_torch - matrix_pyscf
    abs_fro = float(np.linalg.norm(delta, ord="fro"))
    pyscf_fro = float(np.linalg.norm(matrix_pyscf, ord="fro"))
    delta_orth = orthonormalize_operator(
        _torch_array(delta), _torch_array(system["overlap"])
    ).numpy()
    pyscf_orth = orthonormalize_operator(
        _torch_array(matrix_pyscf), _torch_array(system["overlap"])
    ).numpy()
    orth_fro = float(np.linalg.norm(delta_orth, ord="fro"))
    orth_ref = float(np.linalg.norm(pyscf_orth, ord="fro"))
    return {
        "name": system["name"],
        "functional": "same LapEnergy NN instance / actual RKS_with_Laplacian NumIntWithLaplacian.nr_rks",
        "grid_kind": "all exact legacy central rows and weights",
        "n_ao": mol.nao_nr(),
        "n_grid": len(coords),
        "abs_frobenius_Ha": abs_fro,
        "relative_frobenius": abs_fro / max(pyscf_fro, np.finfo(np.float64).tiny),
        "max_abs_element_Ha": float(np.max(np.abs(delta))),
        "orthonormalized_abs_frobenius_Ha": orth_fro,
        "orthonormalized_relative_frobenius": orth_fro
        / max(orth_ref, np.finfo(np.float64).tiny),
    }


def _evaluate_pbe_libxc_parity(system: dict, chunk_size: int) -> dict:
    """Cross-check canonical Torch PBE weak AO matrix against PySCF/libxc."""
    from pyscf import dft
    from pyscf.dft import numint

    mol, dm = system["mol"], system["dm"]
    coords, weights = system["coords"], system["weights"]
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf.max_memory = 128
    mf.grids.coords = coords
    mf.grids.weights = weights
    mf.grids.non0tab = None
    _, _, matrix_libxc = mf._numint.nr_rks(
        mol, mf.grids, mf.xc, dm, max_memory=128
    )

    matrix_torch = np.zeros_like(system["overlap"])
    energy = CanonicalPBEEnergy()
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        ao = np.asarray(numint.eval_ao(mol, coords[start:stop], deriv=2), dtype=np.float64)
        phi_np = ao[0]
        grad_np = ao[1:4].transpose(1, 0, 2).copy()
        lap_np = (ao[4] + ao[7] + ao[9]).copy()
        phi, grad_phi, lap_phi, w = map(
            _torch_array, (phi_np, grad_np, lap_np, weights[start:stop])
        )
        features = rks_density_features_from_ao(phi, grad_phi, lap_phi, _torch_array(dm))
        matrix_torch += assemble_rks_operator(
            energy,
            features,
            w,
            phi,
            grad_phi,
            lap_phi,
            chunk_size=min(512, stop - start),
            create_graph=False,
        ).detach().numpy()
    delta = matrix_torch - matrix_libxc
    abs_fro = float(np.linalg.norm(delta, ord="fro"))
    libxc_fro = float(np.linalg.norm(matrix_libxc, ord="fro"))
    delta_orth = orthonormalize_operator(
        _torch_array(delta), _torch_array(system["overlap"])
    ).numpy()
    libxc_orth = orthonormalize_operator(
        _torch_array(matrix_libxc), _torch_array(system["overlap"])
    ).numpy()
    orth_fro = float(np.linalg.norm(delta_orth, ord="fro"))
    orth_ref = float(np.linalg.norm(libxc_orth, ord="fro"))
    return {
        "name": system["name"],
        "functional": "repository CanonicalPBEEnergy vs PySCF/libxc PBE",
        "grid_kind": "all exact legacy central rows and weights",
        "n_ao": mol.nao_nr(),
        "n_grid": len(coords),
        "abs_frobenius_Ha": abs_fro,
        "relative_frobenius": abs_fro / max(libxc_fro, np.finfo(np.float64).tiny),
        "max_abs_element_Ha": float(np.max(np.abs(delta))),
        "orthonormalized_abs_frobenius_Ha": orth_fro,
        "orthonormalized_relative_frobenius": orth_fro
        / max(orth_ref, np.finfo(np.float64).tiny),
    }


def run(args) -> dict:
    torch.set_num_threads(args.torch_threads)
    legacy_path = Path(args.legacy)
    npz_root = Path(args.npz_root)
    if not legacy_path.is_file() or not npz_root.is_dir():
        raise FileNotFoundError(f"Missing legacy pickle or NPZ root: {legacy_path}, {npz_root}")
    with legacy_path.open("rb") as stream:
        records = pickle.load(stream)
    output = {
        "protocol": "lap-weakform-ao-validation-v1",
        "spatial_finite_difference_used": False,
        "legacy_source": str(legacy_path),
        "legacy_source_sha256": _sha256(legacy_path),
        "npz_root": str(npz_root),
        "chunk_size": args.chunk_size,
        "torch_threads": args.torch_threads,
        "results": [],
        "full_source_grid_results": [],
        "refined_grid_results": [],
        "unpruned_grid_results": [],
        "pyscf_parity_results": [],
        "pbe_libxc_parity_results": [],
    }
    for name in args.systems:
        print(f"D4 {name}: loading exact legacy/NPZ rows", flush=True)
        system = _load_system(name, _load_legacy_record(records, name), npz_root)
        print(
            f"D4 {name}: {system['legacy_points']} points, {system['source_points']} source-grid points",
            flush=True,
        )
        result = _evaluate_system(system, args.chunk_size)
        output["results"].append(result)
        print(
            f"D4 {name}: relF={result['weak_strong_relative_frobenius']:.6e} "
            f"max={result['weak_strong_max_abs_element_Ha']:.6e} Ha "
            f"orthRel={result['orthonormalized_relative_frobenius']:.6e} "
            f"time={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if name in args.full_source_systems:
            full_result = _evaluate_system(
                system,
                args.chunk_size,
                coords=system["source_coords_full"],
                weights=system["source_weights_full"],
                grid_kind="complete-npz-source-grid",
            )
            output["full_source_grid_results"].append(full_result)
            print(
                f"D4 {name} complete NPZ grid: "
                f"relF={full_result['weak_strong_relative_frobenius']:.6e} "
                f"max={full_result['weak_strong_max_abs_element_Ha']:.6e} Ha "
                f"orthRel={full_result['orthonormalized_relative_frobenius']:.6e} "
                f"time={full_result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        if name in args.refined_grid_systems:
            from pyscf.dft.gen_grid import Grids

            for level in args.refined_grid_levels:
                grid = Grids(system["mol"])
                grid.level = level
                grid.build(with_non0tab=False)
                refined = _evaluate_system(
                    system,
                    args.chunk_size,
                    coords=np.asarray(grid.coords, dtype=np.float64),
                    weights=np.asarray(grid.weights, dtype=np.float64),
                    grid_kind=f"pyscf-grid-level-{level}",
                )
                refined["grid_level"] = level
                refined["max_coord_difference_from_npz_source_Bohr"] = (
                    float(
                        np.max(
                            np.abs(
                                grid.coords - system["source_coords_full"]
                            )
                        )
                    )
                    if len(grid.coords) == len(system["source_coords_full"])
                    else None
                )
                refined["elapsed_seconds"] = refined["elapsed_seconds"]
                output["refined_grid_results"].append(refined)
                print(
                    f"D4 {name} PySCF grid L{level} ({len(grid.coords)} points): "
                    f"relF={refined['weak_strong_relative_frobenius']:.6e} "
                    f"max={refined['weak_strong_max_abs_element_Ha']:.6e} Ha "
                    f"orthRel={refined['orthonormalized_relative_frobenius']:.6e} "
                    f"time={refined['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        if name in args.unpruned_grid_systems:
            from pyscf.dft.gen_grid import Grids

            for nrad, nang in args.unpruned_grid_specs:
                grid = Grids(system["mol"])
                grid.atom_grid = (nrad, nang)
                grid.prune = None
                grid.build(with_non0tab=False)
                unpruned = _evaluate_system(
                    system,
                    args.chunk_size,
                    coords=np.asarray(grid.coords, dtype=np.float64),
                    weights=np.asarray(grid.weights, dtype=np.float64),
                    grid_kind=f"pyscf-unpruned-radial-{nrad}-angular-{nang}",
                )
                unpruned["radial_points_per_atom"] = nrad
                unpruned["angular_points_per_shell"] = nang
                output["unpruned_grid_results"].append(unpruned)
                print(
                    f"D4 {name} unpruned radial={nrad} angular={nang} "
                    f"({len(grid.coords)} points): "
                    f"relF={unpruned['weak_strong_relative_frobenius']:.6e} "
                    f"max={unpruned['weak_strong_max_abs_element_Ha']:.6e} Ha "
                    f"orthRel={unpruned['orthonormalized_relative_frobenius']:.6e} "
                    f"time={unpruned['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        if name in args.pyscf_parity_systems:
            parity = _evaluate_pyscf_parity(system, args.chunk_size)
            output["pyscf_parity_results"].append(parity)
            print(
                f"D5 {name}: relF={parity['relative_frobenius']:.6e} "
                f"max={parity['max_abs_element_Ha']:.6e} Ha "
                f"orthRel={parity['orthonormalized_relative_frobenius']:.6e}",
                flush=True,
            )
        if name in args.pbe_libxc_parity_systems:
            parity = _evaluate_pbe_libxc_parity(system, args.chunk_size)
            output["pbe_libxc_parity_results"].append(parity)
            print(
                f"D4 {name} Torch/libxc PBE weak parity: "
                f"relF={parity['relative_frobenius']:.6e} "
                f"max={parity['max_abs_element_Ha']:.6e} Ha "
                f"orthRel={parity['orthonormalized_relative_frobenius']:.6e}",
                flush=True,
            )
    return output


def _defaults():
    if os.name == "nt":
        return (
            r"C:\Users\schne\Downloads\data_vxc_train.pickle",
            r"C:\Users\schne\Downloads\mrks_90_ccsd_pt\mrks_90_ccsd_pt",
            r"C:\Dev\readWFN_share_ms\lap_operator_runs_20261001",
        )
    return (
        "/mnt/c/Users/schne/Downloads/data_vxc_train.pickle",
        "/mnt/c/Users/schne/Downloads/mrks_90_ccsd_pt/mrks_90_ccsd_pt",
        "/mnt/c/Dev/readWFN_share_ms/lap_operator_runs_20261001",
    )


def main():
    legacy_default, npz_default, output_default = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", default=legacy_default)
    parser.add_argument("--npz-root", default=npz_default)
    parser.add_argument("--out-dir", default=output_default)
    parser.add_argument("--systems", nargs="+", default=["H2", "BeH2", "CO", "N2", "ClH"])
    parser.add_argument("--full-source-systems", nargs="*", default=[])
    parser.add_argument("--refined-grid-systems", nargs="*", default=[])
    parser.add_argument("--refined-grid-levels", nargs="*", type=int, default=[])
    parser.add_argument("--unpruned-grid-systems", nargs="*", default=[])
    parser.add_argument(
        "--unpruned-grid-specs",
        nargs="*",
        type=int,
        default=[100, 434],
        help="Flattened nrad/nang pairs, e.g. --unpruned-grid-specs 100 434 140 590.",
    )
    parser.add_argument("--pyscf-parity-systems", nargs="*", default=[])
    parser.add_argument("--pbe-libxc-parity-systems", nargs="*", default=[])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.torch_threads <= 0:
        parser.error("chunk-size and torch-threads must be positive")
    if len(args.unpruned_grid_specs) % 2:
        parser.error("unpruned-grid-specs must contain nrad/nang pairs")
    args.unpruned_grid_specs = list(zip(args.unpruned_grid_specs[::2], args.unpruned_grid_specs[1::2]))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    result_path = output_dir / "d4_real_pbe_ao_validation.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
