"""Run the isolated central-grid E_xc/operator diagnostic pilot.

The runner first replays the established two-epoch canonical Minnesota PBE
predopt, then forks two short diagnostic fits from that same checkpoint:
H2 only and H2+BeH2+CO. It has no reaction, MOO, stencil, or production path.
Use ``--execute`` to start work; without it the command only validates paths
and prints the planned run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TRAIN_MODELS = Path(__file__).resolve().parent
_REPO_ROOT = _TRAIN_MODELS.parent
if str(_TRAIN_MODELS) not in sys.path:
    sys.path.insert(0, str(_TRAIN_MODELS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lap_checkpoint import load_lap_checkpoint
from lap_operator import (
    OPERATOR_PROTOCOL,
    LapEnergy,
    assemble_rks_operator,
    integrated_xc_energy,
    operator_checkpoint_metadata,
    operator_loss,
    rks_density_features_from_ao,
    validate_operator_metadata,
)
from lap_operator_data import (
    build_molecule_from_metadata,
    iter_ao_factor_chunks,
    load_central_operator_record,
    verify_operator_corpus,
)
from lap_vxc import sigma_from_gradients, sigma_standard_to_total
from NN_models_lap import ARCHITECTURE, DESCRIPTOR_PROTOCOL, pcPBELMLOptimizerV2Lap

SYSTEMS = ("H2", "BeH2", "CO")
HARTREE2KCAL = 627.5094740631
AO_CACHE_PROTOCOL = "lap-ao-factor-cache-f32-v1"


@dataclass
class PilotSystem:
    name: str
    record: Any
    mol: Any
    features: torch.Tensor
    weights: torch.Tensor
    ref_v: torch.Tensor
    overlap: torch.Tensor
    ao_chunks: list[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor]]
    density_features_max_abs: float
    density_features_relative_max_error: float
    reference_projection_rel_fro: float
    ao_cache_bytes: int
    ao_prepare_seconds: float


@dataclass
class ScfSystem:
    name: str
    record: Any
    mol: Any


def _require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must stay outside the repository: {resolved}")
    return resolved


def _tensor(value, *, device, dtype) -> torch.Tensor:
    return torch.tensor(np.array(value, copy=True), device=device, dtype=dtype)


def _norm(grads) -> float:
    squared = 0.0
    for grad in grads:
        if grad is not None:
            squared += float(grad.detach().double().square().sum())
    return math.sqrt(squared)


def _grad_cosine(left, right) -> float:
    dot = 0.0
    left2 = 0.0
    right2 = 0.0
    for a, b in zip(left, right):
        if a is not None:
            left2 += float(a.detach().double().square().sum())
        if b is not None:
            right2 += float(b.detach().double().square().sum())
        if a is not None and b is not None:
            dot += float((a.detach().double() * b.detach().double()).sum())
    denom = math.sqrt(left2 * right2)
    return dot / denom if denom else float("nan")


def _fixed_diagnostic_weights(exc_gradient_norm: float, operator_gradient_norm: float) -> tuple[float, float]:
    """Freeze equal-initial-gradient diagnostic coefficients for one branch."""
    if (
        not math.isfinite(exc_gradient_norm)
        or not math.isfinite(operator_gradient_norm)
        or min(exc_gradient_norm, operator_gradient_norm) <= 0
    ):
        raise ValueError("Both pretraining objective gradient norms must be positive and finite.")
    return 1.0 / exc_gradient_norm, 1.0 / operator_gradient_norm


def _load_predopt_start(path: Path, device, dtype):
    """Load an independent branch start from the identical fresh predopt file."""
    return load_lap_checkpoint(path, device=device, dtype=dtype)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_ao_cache(data_dir: Path, cache_dir: Path, chunk_size: int) -> dict[str, Any]:
    """Generate a disposable float32 AO cache in the PySCF environment."""
    import h5py

    cache_dir = _require_external(cache_dir, "AO cache directory")
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite AO cache: {cache_dir}")
    verification = verify_operator_corpus(data_dir, expected_systems=SYSTEMS)
    data_manifest_path = data_dir / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    source_files = {item["system_name"]: data_dir / item["file"] for item in data_manifest["records"]}
    cache_dir.mkdir(parents=True, exist_ok=False)
    cache_records = []
    started = time.perf_counter()
    for name in SYSTEMS:
        source_path = source_files[name]
        record = load_central_operator_record(source_path)
        mol = build_molecule_from_metadata(record.metadata)
        ngrid, nao = record.DensityDescriptorsN10.shape[0], record.dmks.shape[0]
        cache_path = cache_dir / f"{name}_ao_factors.h5"
        with h5py.File(cache_path, "x") as handle:
            handle.attrs["protocol"] = AO_CACHE_PROTOCOL
            handle.attrs["operator_protocol"] = OPERATOR_PROTOCOL
            handle.attrs["system_name"] = name
            handle.attrs["source_record_sha256"] = record.metadata["record_sha256"]
            handle.attrs["source_file_sha256"] = _file_sha256(source_path)
            handle.attrs["dtype"] = "float32"
            phi_ds = handle.create_dataset(
                "phi", shape=(ngrid, nao), dtype="<f4", chunks=(min(chunk_size, ngrid), nao),
                compression="lzf", shuffle=True,
            )
            grad_ds = handle.create_dataset(
                "grad_phi", shape=(ngrid, 3, nao), dtype="<f4", chunks=(min(chunk_size, ngrid), 3, nao),
                compression="lzf", shuffle=True,
            )
            lap_ds = handle.create_dataset(
                "lap_phi", shape=(ngrid, nao), dtype="<f4", chunks=(min(chunk_size, ngrid), nao),
                compression="lzf", shuffle=True,
            )
            for sl, phi, grad_phi, lap_phi in iter_ao_factor_chunks(record, mol, chunk_size):
                phi_ds[sl] = np.asarray(phi, dtype=np.float32)
                grad_ds[sl] = np.asarray(grad_phi, dtype=np.float32)
                lap_ds[sl] = np.asarray(lap_phi, dtype=np.float32)
            handle.flush()
        cache_records.append(
            {
                "system_name": name,
                "file": cache_path.name,
                "point_count": ngrid,
                "nao": nao,
                "source_record_sha256": record.metadata["record_sha256"],
                "source_file_sha256": _file_sha256(source_path),
                "cache_file_sha256": _file_sha256(cache_path),
                "cache_file_bytes": cache_path.stat().st_size,
            }
        )
    manifest = {
        "protocol": AO_CACHE_PROTOCOL,
        "operator_protocol": OPERATOR_PROTOCOL,
        "cache_dtype": "float32",
        "central_data_manifest": str(data_manifest_path.resolve()),
        "central_data_manifest_sha256": _file_sha256(data_manifest_path),
        "source_protocol": verification["protocol"],
        "chunk_size_used_for_generation": chunk_size,
        "records": cache_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result = {
        **manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "total_cache_file_bytes": sum(item["cache_file_bytes"] for item in cache_records),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _load_ao_cache_chunks(
    record,
    data_dir: Path,
    cache_dir: Path,
    device,
    dtype,
    chunk_size: int,
) -> tuple[list[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor]], float, float, float, int]:
    import h5py

    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (
        cache_manifest.get("protocol") != AO_CACHE_PROTOCOL
        or cache_manifest.get("operator_protocol") != OPERATOR_PROTOCOL
        or cache_manifest.get("cache_dtype") != "float32"
    ):
        raise ValueError("AO-factor cache uses an incompatible protocol or dtype.")
    central_manifest = data_dir / "manifest.json"
    if cache_manifest.get("central_data_manifest_sha256") != _file_sha256(central_manifest):
        raise ValueError("AO cache was generated from a different central-data manifest.")
    item = next((value for value in cache_manifest["records"] if value.get("system_name") == record.metadata["system_name"]), None)
    if item is None:
        raise ValueError(f"AO cache is missing {record.metadata['system_name']}.")
    if item.get("source_record_sha256") != record.metadata["record_sha256"]:
        raise ValueError(f"AO cache source record does not match {record.metadata['system_name']}.")
    source_path = data_dir / next(
        value["file"] for value in json.loads(central_manifest.read_text(encoding="utf-8"))["records"]
        if value["system_name"] == record.metadata["system_name"]
    )
    if item.get("source_file_sha256") != _file_sha256(source_path):
        raise ValueError(f"AO cache source-file hash does not match {record.metadata['system_name']}.")
    cache_path = cache_dir / item["file"]
    if item.get("cache_file_sha256") != _file_sha256(cache_path):
        raise ValueError(f"AO cache file hash mismatch for {record.metadata['system_name']}.")
    ao_chunks = []
    density_error_by_feature = np.zeros(record.DensityDescriptorsN10.shape[1], dtype=np.float64)
    feature_scale = np.max(np.abs(record.DensityDescriptorsN10), axis=0)
    reference_projection = np.zeros_like(record.RefAO, dtype=np.float64)
    dm64 = torch.tensor(np.array(record.dmks, copy=True), dtype=torch.float64)
    with h5py.File(cache_path, "r") as handle:
        if (
            handle.attrs.get("protocol") != AO_CACHE_PROTOCOL
            or handle.attrs.get("operator_protocol") != OPERATOR_PROTOCOL
            or handle.attrs.get("system_name") != record.metadata["system_name"]
            or handle.attrs.get("source_record_sha256") != record.metadata["record_sha256"]
            or handle.attrs.get("dtype") != "float32"
        ):
            raise ValueError(f"Invalid AO-factor cache metadata for {record.metadata['system_name']}.")
        ngrid, nao = record.DensityDescriptorsN10.shape[0], record.dmks.shape[0]
        expected_shapes = {"phi": (ngrid, nao), "grad_phi": (ngrid, 3, nao), "lap_phi": (ngrid, nao)}
        for key, shape in expected_shapes.items():
            if key not in handle or handle[key].shape != shape or handle[key].dtype != np.dtype("<f4"):
                raise ValueError(f"Invalid {key} cache shape/dtype for {record.metadata['system_name']}.")
        for start in range(0, ngrid, chunk_size):
            stop = min(start + chunk_size, ngrid)
            sl = slice(start, stop)
            phi_np = np.asarray(handle["phi"][sl])
            grad_np = np.asarray(handle["grad_phi"][sl])
            lap_np = np.asarray(handle["lap_phi"][sl])
            phi64 = torch.as_tensor(phi_np.astype(np.float64))
            grad64 = torch.as_tensor(grad_np.astype(np.float64))
            lap64 = torch.as_tensor(lap_np.astype(np.float64))
            calculated = rks_density_features_from_ao(phi64, grad64, lap64, dm64)
            row_error = (calculated - torch.as_tensor(np.array(record.DensityDescriptorsN10[sl], copy=True))).abs().amax(dim=0).numpy()
            density_error_by_feature = np.maximum(density_error_by_feature, row_error)
            weighted = np.asarray(record.weights[sl] * record.VxcLegacy[sl], dtype=np.float64)
            reference_projection += phi64.numpy().T @ (phi64.numpy() * weighted[:, None])
            ao_chunks.append(
                (
                    sl,
                    _tensor(phi_np, device=device, dtype=dtype),
                    _tensor(grad_np, device=device, dtype=dtype),
                    _tensor(lap_np, device=device, dtype=dtype),
                )
            )
    reference_projection = 0.5 * (reference_projection + reference_projection.T)
    ref_norm = max(float(np.linalg.norm(record.RefAO)), np.finfo(float).tiny)
    ref_rel = float(np.linalg.norm(reference_projection - record.RefAO) / ref_norm)
    density_error = float(density_error_by_feature.max())
    density_relative_error = float(
        np.max(density_error_by_feature / np.maximum(feature_scale, np.finfo(float).tiny))
    )
    if density_relative_error > 2e-6:
        raise ValueError(f"{record.metadata['system_name']}: AO-cache density relative mismatch is {density_relative_error:.3e}.")
    if ref_rel > 1e-5:
        raise ValueError(f"{record.metadata['system_name']}: AO-cache RefAO mismatch is {ref_rel:.3e}.")
    cache_bytes = sum(t.numel() * t.element_size() for chunk in ao_chunks for t in chunk[1:])
    return ao_chunks, density_error, density_relative_error, ref_rel, cache_bytes


def _load_systems(
    data_dir: Path, cache_dir: Path, device, dtype, chunk_size: int
) -> tuple[dict[str, PilotSystem], dict]:
    corpus_verification = verify_operator_corpus(data_dir, expected_systems=SYSTEMS)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_names = [item.get("system_name") for item in cache_manifest.get("records", [])]
    if len(cache_names) != len(SYSTEMS) or set(cache_names) != set(SYSTEMS):
        raise ValueError("AO cache manifest must contain exactly H2, BeH2, and CO.")
    files = {item["system_name"]: data_dir / item["file"] for item in manifest["records"]}
    loaded: dict[str, PilotSystem] = {}
    for name in SYSTEMS:
        path = files[name]
        record = load_central_operator_record(path)
        start_time = time.perf_counter()
        features64 = np.asarray(record.DensityDescriptorsN10, dtype=np.float64)
        ao_chunks, density_error, density_relative_error, ref_rel, cache_bytes = _load_ao_cache_chunks(
            record, data_dir, cache_dir, device, dtype, chunk_size
        )
        loaded[name] = PilotSystem(
            name=name,
            record=record,
            mol=None,
            features=_tensor(features64, device=device, dtype=dtype),
            weights=_tensor(record.weights, device=device, dtype=dtype),
            ref_v=_tensor(record.RefAO, device=device, dtype=dtype),
            overlap=_tensor(record.Overlap, device=device, dtype=dtype),
            ao_chunks=ao_chunks,
            density_features_max_abs=density_error,
            density_features_relative_max_error=density_relative_error,
            reference_projection_rel_fro=ref_rel,
            ao_cache_bytes=cache_bytes,
            ao_prepare_seconds=time.perf_counter() - start_time,
        )
    return loaded, {
        **corpus_verification,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "record_files": {name: str(path.resolve()) for name, path in files.items()},
        "record_file_sha256": {name: _file_sha256(path) for name, path in files.items()},
        "ao_cache_manifest": str(cache_manifest_path.resolve()),
        "ao_cache_manifest_sha256": _file_sha256(cache_manifest_path),
        "ao_cache_protocol": cache_manifest.get("protocol"),
        "ao_cache_total_file_bytes": sum(item["cache_file_bytes"] for item in cache_manifest["records"]),
    }


def _operator_for_system(energy, system: PilotSystem, chunk_size: int, *, create_graph=True):
    result = system.ref_v.new_zeros(system.ref_v.shape)
    for sl, phi, grad_phi, lap_phi in system.ao_chunks:
        result = result + assemble_rks_operator(
            energy,
            system.features[sl],
            system.weights[sl],
            phi,
            grad_phi,
            lap_phi,
            chunk_size=chunk_size,
            create_graph=create_graph,
        )
    return result


def _losses(model, systems: list[PilotSystem], chunk_size: int, *, create_graph=True):
    energy = LapEnergy(model)
    exc_errors = []
    op_errors = []
    per_system = {}
    for system in systems:
        predicted_exc = integrated_xc_energy(energy, system.features, system.weights)
        ref_exc = torch.as_tensor(float(system.record.Exc), device=predicted_exc.device, dtype=predicted_exc.dtype)
        error_kcal = (predicted_exc - ref_exc) * HARTREE2KCAL
        predicted_v = _operator_for_system(energy, system, chunk_size, create_graph=create_graph)
        v_loss = operator_loss(predicted_v, system.ref_v, system.overlap)
        exc_errors.append(error_kcal.square())
        op_errors.append(v_loss)
        per_system[system.name] = {
            "predicted_exc_hartree": predicted_exc,
            "exc_error_kcal_mol": error_kcal,
            "operator_loss_ha2_per_ao": v_loss,
        }
    exc_loss = torch.stack(exc_errors).mean()
    op_loss = torch.stack(op_errors).mean()
    return exc_loss, op_loss, per_system


def _q_tau_diagnostics(model, system: PilotSystem) -> dict[str, Any]:
    count = min(1024, len(system.features))
    density = system.features[:, 0] + system.features[:, 1]
    rank_positions = torch.linspace(
        0, len(density) - 1, count, device=density.device, dtype=torch.float64
    ).round().long()
    population_order = torch.argsort(density)
    indices = population_order.index_select(0, rank_positions)
    features = system.features.index_select(0, indices)
    rho = features[:, :2]
    grad = features[:, 2:8].reshape(-1, 2, 3)
    sigma = sigma_from_gradients(grad)
    lapl = features[:, 8:10].detach().requires_grad_(True)
    raw = torch.cat((rho, sigma_standard_to_total(sigma), torch.zeros_like(rho), lapl), dim=-1)
    tau_shifted = raw.detach().clone()
    tau_shifted[:, 5:7] = 1e4
    raw_qzero = raw.detach().clone()
    raw_qzero[:, 7:9] = 0
    with torch.enable_grad():
        constants = model(raw)
        shifted_constants = model(tau_shifted)
        zero_q_constants = model(raw_qzero)
        local = LapEnergy(model)(rho, sigma, lapl)
        lap_gradient = torch.autograd.grad(local.sum(), lapl, allow_unused=True)[0]
    tau_delta = float((constants.detach() - shifted_constants.detach()).abs().max())
    q_gradient = torch.zeros_like(lapl) if lap_gradient is None else lap_gradient.detach()
    q_zero = LapEnergy(model)(rho, sigma, torch.zeros_like(lapl)).detach()
    q_actual = local.detach()
    values_finite = all(
        bool(torch.isfinite(value).all())
        for value in (raw, constants, shifted_constants, zero_q_constants, q_gradient, q_zero, q_actual)
    )
    if not values_finite or tau_delta != 0.0:
        raise FloatingPointError("Nonfinite q/tau diagnostic or tau input leaked into Lap model output.")
    return {
        "sample_points": count,
        "sampling": "deterministic density-rank stratification over the complete central population",
        "density_range": [float(density.min()), float(density.max())],
        "tau_feature_max_abs": 0.0,
        "tau_perturbation_output_max_abs": tau_delta,
        "lapl_adaptive_constants_delta_max_abs_vs_zero_lapl": float(
            (constants.detach() - zero_q_constants.detach()).abs().max()
        ),
        "lapl_energy_delta_max_abs_vs_zero_lapl": float((q_actual - q_zero).abs().max()),
        "lapl_energy_gradient_max_abs": float(q_gradient.abs().max()),
        "lapl_energy_gradient_l2": float(q_gradient.double().norm()),
        "lapl_energy_gradient_nonzero_entries": int(torch.count_nonzero(q_gradient)),
        "all_finite": values_finite,
    }


def _save_operator_checkpoint(path: Path, model, metadata: dict[str, Any]) -> str:
    operator_metadata = operator_checkpoint_metadata().to_dict()
    validate_operator_metadata(operator_metadata)
    payload = {
        "architecture": ARCHITECTURE,
        "descriptor_protocol": DESCRIPTOR_PROTOCOL,
        "protocol": OPERATOR_PROTOCOL,
        "operator_metadata": operator_metadata,
        "model_kwargs": dict(model.model_kwargs),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "pilot_metadata": metadata,
    }
    forbidden = {"h", "stencil", "h_bohr", "stencil_version", "stencil_order", "derivative_order"}
    if forbidden.intersection(payload) or forbidden.intersection(payload["operator_metadata"]):
        raise ValueError("Operator pilot checkpoints cannot contain h/stencil metadata.")
    torch.save(payload, path)
    return _file_sha256(path)


def _load_operator_checkpoint(path: Path, device, dtype):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("architecture"),
        payload.get("descriptor_protocol"),
        payload.get("protocol"),
    ) != (ARCHITECTURE, DESCRIPTOR_PROTOCOL, OPERATOR_PROTOCOL):
        raise ValueError("Incompatible h-free operator pilot checkpoint.")
    validate_operator_metadata(payload["operator_metadata"])
    forbidden = {"h", "stencil", "h_bohr", "stencil_version", "stencil_order", "derivative_order"}
    if forbidden.intersection(payload) or forbidden.intersection(payload["operator_metadata"]):
        raise ValueError("Operator pilot checkpoint contains forbidden h/stencil fields.")
    model = pcPBELMLOptimizerV2Lap(**payload["model_kwargs"]).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _scf_molecule_for_system(system):
    """Reconstruct the molecule only in a PySCF SCF environment."""
    if system.mol is not None:
        return system.mol
    return build_molecule_from_metadata(system.record.metadata)


def _run_scf(checkpoint: Path, system: ScfSystem, output_dir: Path, *, grid_level: int, max_cycle: int):
    from pyscf import dft

    from test_models.DFT.lap_functional import LapFunctional

    model, checkpoint_payload = _load_operator_checkpoint(
        checkpoint, torch.device("cpu"), torch.float64
    )
    mol = _scf_molecule_for_system(system)
    functional = LapFunctional(model)
    mf = functional.make_rks(mol)
    mf.grids.level = grid_level
    mf.max_cycle = max_cycle
    mf.conv_tol = 1e-8
    trace = []

    def capture_cycle(envs):
        fields = ("cycle", "e_tot", "norm_gorb", "norm_ddm")
        trace.append({key: float(envs[key]) for key in fields if key in envs})

    mf.callback = capture_cycle
    started = time.perf_counter()
    total_energy = float(mf.kernel())
    elapsed = time.perf_counter() - started
    if not math.isfinite(total_energy):
        raise FloatingPointError(f"{system.name}: RKS SCF returned a nonfinite energy.")
    mf.grids.build(with_non0tab=True)
    dm = mf.get_init_guess()
    ao = dft.numint.eval_ao(mol, mf.grids.coords[:32], deriv=2)
    rho = dft.numint.eval_rho(mol, ao, dm, xctype="MGGA", hermi=1, with_lapl=True)
    before = functional.eval_xc("", rho, spin=0, deriv=1)
    shifted = np.array(rho, copy=True)
    shifted[5] += 1e6
    after = functional.eval_xc("", shifted, spin=0, deriv=1)
    vtau = np.asarray(before[1][3])
    tau_independent = all(np.array_equal(left, right) for left, right in zip(before[1], after[1]))
    if np.any(vtau != 0) or not tau_independent:
        raise AssertionError(f"{system.name}: SCF adapter is tau-dependent.")
    result = {
        "system": system.name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_protocol": checkpoint_payload["protocol"],
        "grid_level": grid_level,
        "max_cycle": max_cycle,
        "converged": bool(mf.converged),
        "total_energy_hartree": total_energy,
        "cycles_recorded": len(trace),
        "convergence_trace": trace,
        "elapsed_seconds": elapsed,
        "vtau_max_abs": float(np.max(np.abs(vtau), initial=0.0)),
        "tau_independent": tau_independent,
        "all_cycle_energies_finite": all(math.isfinite(row["e_tot"]) for row in trace if "e_tot" in row),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{checkpoint.parent.name}__{system.name}_scf.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _run_scf_only(
    data_dir: Path,
    output_dir: Path,
    h2_checkpoint: Path,
    combined_checkpoint: Path,
    grid_level: int,
    max_cycle: int,
) -> dict[str, Any]:
    output_dir = _require_external(output_dir, "SCF output directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite SCF output directory: {output_dir}")
    if not h2_checkpoint.is_file() or not combined_checkpoint.is_file():
        raise FileNotFoundError("Both h2-only and combined h-free checkpoints must exist.")
    verification = verify_operator_corpus(data_dir, expected_systems=SYSTEMS)
    central_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    source_paths = {item["system_name"]: data_dir / item["file"] for item in central_manifest["records"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in SYSTEMS:
        record = load_central_operator_record(source_paths[name])
        if int(record.metadata["molecule"]["spin"]) != 0:
            raise ValueError(f"{name}: SCF pilot requires a closed-shell system.")
        records[name] = ScfSystem(name, record, build_molecule_from_metadata(record.metadata))
    results = []
    for checkpoint, names in ((h2_checkpoint, ("H2",)), (combined_checkpoint, SYSTEMS)):
        for name in names:
            results.append(
                _run_scf(
                    checkpoint,
                    records[name],
                    output_dir,
                    grid_level=grid_level,
                    max_cycle=max_cycle,
                )
            )
    report = {
        "operation": "post-pilot RKS SCF checks only",
        "protocol": OPERATOR_PROTOCOL,
        "central_data_verification": verification,
        "central_manifest_sha256": _file_sha256(data_dir / "manifest.json"),
        "checkpoints": {
            "h2_only": {"path": str(h2_checkpoint.resolve()), "sha256": _file_sha256(h2_checkpoint)},
            "h2_beh2_co": {"path": str(combined_checkpoint.resolve()), "sha256": _file_sha256(combined_checkpoint)},
        },
        "results": results,
    }
    report_path = output_dir / "lap_operator_scf_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _final_checkpoint_evaluation(
    model,
    selected: list[PilotSystem],
    chunk_size: int,
    coeff_exc: float,
    coeff_op: float,
) -> dict[str, Any]:
    """Evaluate the saved post-update state without changing model parameters."""
    model.eval()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    exc_loss, op_loss, per_system = _losses(model, selected, chunk_size)
    combined = coeff_exc * exc_loss + coeff_op * op_loss
    exc_grads = torch.autograd.grad(exc_loss, parameters, retain_graph=True, allow_unused=True)
    op_grads = torch.autograd.grad(op_loss, parameters, retain_graph=True, allow_unused=True)
    combined_grads = torch.autograd.grad(combined, parameters, retain_graph=True, allow_unused=True)
    per_system_metrics = {}
    n_systems = len(selected)
    for index, (name, values) in enumerate(per_system.items()):
        contribution = (
            coeff_exc * values["exc_error_kcal_mol"].square() / n_systems
            + coeff_op * values["operator_loss_ha2_per_ao"] / n_systems
        )
        grads = torch.autograd.grad(
            contribution,
            parameters,
            retain_graph=index + 1 < len(per_system),
            allow_unused=True,
        )
        per_system_metrics[name] = {
            "predicted_exc_hartree": float(values["predicted_exc_hartree"].detach()),
            "exc_error_kcal_mol": float(values["exc_error_kcal_mol"].detach()),
            "exc_loss_kcal2_contribution": float(
                values["exc_error_kcal_mol"].detach().square() / n_systems
            ),
            "operator_loss_ha2_per_ao": float(values["operator_loss_ha2_per_ao"].detach()),
            "combined_objective_gradient_norm": _norm(grads),
        }
    result = {
        "checkpoint_state": "post_update",
        "optimizer_updates_completed": None,
        "exc_loss_kcal2": float(exc_loss.detach()),
        "operator_loss_ha2_per_ao": float(op_loss.detach()),
        "combined_loss": float(combined.detach()),
        "exc_gradient_norm": _norm(exc_grads),
        "operator_gradient_norm": _norm(op_grads),
        "exc_operator_gradient_cosine": _grad_cosine(exc_grads, op_grads),
        "combined_gradient_norm": _norm(combined_grads),
        "system_metrics": per_system_metrics,
    }
    scalar_values = [
        value
        for key, value in result.items()
        if isinstance(value, (int, float))
    ]
    scalar_values.extend(
        value
        for metrics in per_system_metrics.values()
        for value in metrics.values()
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise FloatingPointError("Final checkpoint objective or gradient diagnostics are nonfinite.")
    return result


def _refresh_checkpoint_evaluations(
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    h2_checkpoint: Path,
    combined_checkpoint: Path,
    device,
    dtype,
    chunk_size: int,
) -> dict[str, Any]:
    """Refresh reports from existing checkpoints; this operation performs no optimizer updates."""
    output_dir = _require_external(output_dir, "pilot output directory")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Existing pilot output directory required: {output_dir}")
    systems, cache_verification = _load_systems(data_dir, cache_dir, device, dtype, chunk_size)
    refreshed = {}
    for label, checkpoint in (("h2_only", h2_checkpoint), ("h2_beh2_co", combined_checkpoint)):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, payload = _load_operator_checkpoint(checkpoint, device, dtype)
        branch_systems = tuple(payload["pilot_metadata"]["systems"])
        selected = [systems[name] for name in branch_systems]
        report_path = checkpoint.parent / "pilot_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        initial_weights = report["initial_objectives_and_gradients"]["diagnostic_loss_coefficients"]
        final_evaluation = _final_checkpoint_evaluation(
            model,
            selected,
            chunk_size,
            float(initial_weights["exc"]),
            float(initial_weights["operator"]),
        )
        final_evaluation["optimizer_updates_completed"] = int(report["steps"])
        q_tau_by_system = {system.name: _q_tau_diagnostics(model, system) for system in selected}
        for row in report.get("history", []):
            row["measurement_phase"] = "pre_update"
        report["history_semantics"] = (
            "Each history row is evaluated immediately before its named optimizer update; "
            "final_checkpoint_evaluation measures the saved state after all updates."
        )
        report["optimizer_updates_completed"] = int(report["steps"])
        report["final_checkpoint_evaluation"] = final_evaluation
        report["q_tau_diagnostics_after_pilot_by_system"] = q_tau_by_system
        report["q_tau_diagnostics_after_pilot"] = q_tau_by_system[branch_systems[0]]
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        refreshed[label] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "report": str(report_path.resolve()),
            "final_checkpoint_evaluation": final_evaluation,
            "q_tau_diagnostics_after_pilot_by_system": q_tau_by_system,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_path = output_dir / "lap_operator_pilot_report.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["branches"] = [
        {
            "name": label,
            "checkpoint": value["checkpoint"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "report": value["report"],
            "final_checkpoint_evaluation": value["final_checkpoint_evaluation"],
            "q_tau_diagnostics_after_pilot_by_system": value["q_tau_diagnostics_after_pilot_by_system"],
        }
        for label, value in refreshed.items()
    ]
    summary["post_update_metrics_refreshed_without_optimizer_updates"] = True
    summary["evaluation_ao_cache_verification"] = cache_verification
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result = {"optimizer_updates": 0, "branches": refreshed}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _run_branch(
    *,
    label: str,
    system_names: tuple[str, ...],
    systems: dict[str, PilotSystem],
    predopt_checkpoint: Path,
    predopt_sha256: str,
    output_dir: Path,
    device,
    dtype,
    steps: int,
    lr: float,
    chunk_size: int,
) -> tuple[Path, dict[str, Any]]:
    model, _ = _load_predopt_start(predopt_checkpoint, device=device, dtype=dtype)
    model.train()
    selected = [systems[name] for name in system_names]
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    exc0, op0, initial_per_system = _losses(model, selected, chunk_size)
    exc_grads = torch.autograd.grad(exc0, parameters, retain_graph=True, allow_unused=True)
    op_grads = torch.autograd.grad(op0, parameters, retain_graph=True, allow_unused=True)
    exc_norm = _norm(exc_grads)
    op_norm = _norm(op_grads)
    try:
        coeff_exc, coeff_op = _fixed_diagnostic_weights(exc_norm, op_norm)
    except ValueError as exc:
        raise FloatingPointError(f"{label}: objective gradient norm is zero or nonfinite.") from exc
    cosine = _grad_cosine(exc_grads, op_grads)
    if not math.isfinite(cosine):
        raise FloatingPointError(f"{label}: objective gradient cosine is nonfinite.")
    initial = {
        "exc_loss_kcal2": float(exc0.detach()),
        "operator_loss_ha2_per_ao": float(op0.detach()),
        "exc_gradient_norm": exc_norm,
        "operator_gradient_norm": op_norm,
        "exc_operator_gradient_cosine": cosine,
        "system_metrics": {
            name: {
                "predicted_exc_hartree": float(values["predicted_exc_hartree"].detach()),
                "exc_error_kcal_mol": float(values["exc_error_kcal_mol"].detach()),
                "operator_loss_ha2_per_ao": float(values["operator_loss_ha2_per_ao"].detach()),
            }
            for name, values in initial_per_system.items()
        },
        "diagnostic_loss_coefficients": {
            "exc": coeff_exc,
            "operator": coeff_op,
            "rule": "freeze inverse initial parameter-gradient norms so both pilot objectives begin with unit gradient norm; diagnostic only",
        },
    }
    del exc0, op0, exc_grads, op_grads, initial_per_system
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        exc_loss, v_loss, per_system = _losses(model, selected, chunk_size)
        combined = coeff_exc * exc_loss + coeff_op * v_loss
        if not all(bool(torch.isfinite(value)) for value in (exc_loss, v_loss, combined)):
            raise FloatingPointError(f"{label}: nonfinite objective at step {step + 1}.")
        combined.backward()
        grad_norm = _norm([parameter.grad for parameter in parameters])
        if not math.isfinite(grad_norm) or grad_norm == 0:
            raise FloatingPointError(f"{label}: zero/nonfinite combined gradient at step {step + 1}.")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "step": step + 1,
            "measurement_phase": "pre_update",
            "exc_loss_kcal2": float(exc_loss.detach()),
            "operator_loss_ha2_per_ao": float(v_loss.detach()),
            "combined_loss": float(combined.detach()),
            "combined_gradient_norm": grad_norm,
            "exc_errors_kcal_mol": {
                name: float(values["exc_error_kcal_mol"].detach())
                for name, values in per_system.items()
            },
            "step_seconds": time.perf_counter() - started,
            "chunk_size": chunk_size,
        }
        if device.type == "cuda":
            row.update(
                cuda_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                cuda_reserved_bytes=int(torch.cuda.memory_reserved(device)),
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            )
        if not all(math.isfinite(value) for value in row["exc_errors_kcal_mol"].values()):
            raise FloatingPointError(f"{label}: nonfinite E_xc error at step {step + 1}.")
        history.append(row)
        print(json.dumps({"branch": label, **row}, sort_keys=True), flush=True)

    checkpoint_path = output_dir / "lap_operator_pilot.pt"
    checkpoint_hash = _save_operator_checkpoint(
        checkpoint_path,
        model,
        {
            "pilot_only": True,
            "branch": label,
            "systems": list(system_names),
            "objective": "E_xc plus AO operator diagnostic; no reaction/MOO objective",
            "steps": steps,
            "learning_rate": lr,
            "chunk_size": chunk_size,
            "predopt_checkpoint_sha256": predopt_sha256,
        },
    )
    final_evaluation = _final_checkpoint_evaluation(
        model, selected, chunk_size, coeff_exc, coeff_op
    )
    final_evaluation["optimizer_updates_completed"] = steps
    q_tau_by_system = {system.name: _q_tau_diagnostics(model, system) for system in selected}
    q_tau = q_tau_by_system[selected[0].name]
    report = {
        "pilot_only": True,
        "branch": label,
        "systems": list(system_names),
        "predopt_checkpoint": str(predopt_checkpoint.resolve()),
        "predopt_checkpoint_sha256": predopt_sha256,
        "steps": steps,
        "learning_rate": lr,
        "chunk_size": chunk_size,
        "objective": "E_xc plus AO operator diagnostic; no reaction/MOO objective",
        "initial_objectives_and_gradients": initial,
        "history_semantics": (
            "Each history row is evaluated immediately before its named optimizer update; "
            "final_checkpoint_evaluation measures the saved state after all updates."
        ),
        "optimizer_updates_completed": steps,
        "final_checkpoint_evaluation": final_evaluation,
        "q_tau_diagnostics_after_pilot": q_tau,
        "q_tau_diagnostics_after_pilot_by_system": q_tau_by_system,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_protocol": OPERATOR_PROTOCOL,
        "h_or_stencil_metadata_present": False,
        "all_history_finite": all(
            math.isfinite(value)
            for row in history
            for key, value in row.items()
            if isinstance(value, (int, float))
        ),
    }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return checkpoint_path, report


def _run(args) -> dict[str, Any]:
    output_dir = _require_external(args.output_dir, "pilot output directory")
    cache_dir = _require_external(args.ao_cache_dir, "AO cache directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output_dir}")
    for path in (args.canonical_pickle, args.canonical_manifest, args.central_data_dir, cache_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    requested_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; predopt was not started.")
    if not args.skip_scf and importlib.util.find_spec("pyscf") is None:
        raise RuntimeError("PySCF is unavailable here; use --skip-scf, then run --scf-only in WSL.")
    if requested_device.type == "cuda" and requested_device.index is not None:
        torch.cuda.set_device(requested_device)
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.predopt_dir is None:
        predopt_dir = output_dir / "predopt"
        runner = _TRAIN_MODELS / "run_lap_pbe_predopt_reference.py"
        command = [
            sys.executable,
            str(runner),
            "--canonical-pickle",
            str(args.canonical_pickle.resolve()),
            "--canonical-manifest",
            str(args.canonical_manifest.resolve()),
            "--output-dir",
            str(predopt_dir),
            "--device",
            str(requested_device),
            "--seed",
            str(args.seed),
            "--chunk-size",
            str(args.predopt_chunk_size),
        ]
        subprocess.run(command, cwd=_REPO_ROOT, check=True)
    else:
        predopt_dir = _require_external(args.predopt_dir, "fresh predopt directory")
        checkpoint = predopt_dir / "lap_pbe_predopt.pt"
        report_path = predopt_dir / "lap_pbe_predopt_report.json"
        if not checkpoint.is_file() or not report_path.is_file():
            raise FileNotFoundError("--predopt-dir must contain a completed checkpoint and report.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("base_reactions") != 268
            or report.get("epochs") != 2
            or report.get("seed") != args.seed
            or report.get("source_sha256") != _file_sha256(args.canonical_pickle)
            or report.get("source_manifest_sha256") != _file_sha256(args.canonical_manifest)
            or report.get("checkpoint_sha256") != _file_sha256(checkpoint)
        ):
            raise ValueError("--predopt-dir does not match this canonical two-epoch PBE predopt request.")
        load_lap_checkpoint(checkpoint, device="cpu", dtype=torch.float32)
    predopt_checkpoint = predopt_dir / "lap_pbe_predopt.pt"
    predopt_report_path = predopt_dir / "lap_pbe_predopt_report.json"
    predopt_hash = _file_sha256(predopt_checkpoint)
    device = requested_device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        free_vram = total_vram = None
        gpu_name = None
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    systems, data_provenance = _load_systems(
        args.central_data_dir.resolve(), cache_dir, device, dtype, args.chunk_size
    )
    system_provenance = {
        name: {
            "points": len(system.record.coords64),
            "nao": system.record.dmks.shape[0],
            "density_features_max_abs_error": system.density_features_max_abs,
            "density_features_relative_max_error": system.density_features_relative_max_error,
            "reference_projection_relative_frobenius": system.reference_projection_rel_fro,
            "ao_cache_bytes": system.ao_cache_bytes,
            "ao_prepare_seconds": system.ao_prepare_seconds,
        }
        for name, system in systems.items()
    }

    branches = []
    for label, names in (("h2_only", ("H2",)), ("h2_beh2_co", SYSTEMS)):
        branch_dir = output_dir / label
        branch_dir.mkdir()
        checkpoint, report = _run_branch(
            label=label,
            system_names=names,
            systems=systems,
            predopt_checkpoint=predopt_checkpoint,
            predopt_sha256=predopt_hash,
            output_dir=branch_dir,
            device=device,
            dtype=dtype,
            steps=args.steps,
            lr=args.lr,
            chunk_size=args.chunk_size,
        )
        branches.append({"name": label, "checkpoint": str(checkpoint.resolve()), "report": str((branch_dir / "pilot_report.json").resolve()), "checkpoint_sha256": report["checkpoint_sha256"]})

    scf_results = []
    if not args.skip_scf:
        scf_dir = output_dir / "scf"
        for branch in branches:
            checkpoint = Path(branch["checkpoint"])
            selected_names = ("H2",) if branch["name"] == "h2_only" else SYSTEMS
            for name in selected_names:
                scf_results.append(
                    _run_scf(
                        checkpoint,
                        systems[name],
                        scf_dir,
                        grid_level=args.scf_grid_level,
                        max_cycle=args.scf_max_cycle,
                    )
                )
    result = {
        "pilot_only": True,
        "production_training": False,
        "reaction_objective": False,
        "moo_or_slurm_jobs": False,
        "gate2": "passed before run",
        "device": str(device),
        "gpu": gpu_name,
        "cuda_total_vram_bytes": total_vram,
        "cuda_free_vram_at_start_bytes": free_vram,
        "dtype": args.dtype,
        "seed": args.seed,
        "steps_per_branch": args.steps,
        "learning_rate": args.lr,
        "chunk_size": args.chunk_size,
        "central_data": data_provenance,
        "system_provenance": system_provenance,
        "predopt_report": str(predopt_report_path.resolve()),
        "predopt_report_sha256": _file_sha256(predopt_report_path),
        "predopt_checkpoint": str(predopt_checkpoint.resolve()),
        "predopt_checkpoint_sha256": predopt_hash,
        "branches": branches,
        "scf_results": scf_results,
    }
    (output_dir / "lap_operator_pilot_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-pickle", type=Path)
    parser.add_argument("--canonical-manifest", type=Path)
    parser.add_argument("--central-data-dir", type=Path)
    parser.add_argument("--ao-cache-dir", type=Path)
    parser.add_argument("--predopt-dir", type=Path, help="reuse a verified fresh two-epoch canonical PBE predopt run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-ao-cache", action="store_true", help="build the transient AO cache in a PySCF environment")
    parser.add_argument("--scf-only", action="store_true", help="run WSL/PySCF SCF checks on completed h-free checkpoints")
    parser.add_argument("--evaluate-checkpoints-only", action="store_true", help="refresh post-update objectives and q/tau diagnostics without training")
    parser.add_argument("--h2-checkpoint", type=Path)
    parser.add_argument("--combined-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--predopt-chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scf-grid-level", type=int, default=1)
    parser.add_argument("--scf-max-cycle", type=int, default=40)
    parser.add_argument("--skip-scf", action="store_true")
    parser.add_argument("--cache-chunk-size", type=int, default=4096)
    parser.add_argument("--execute", action="store_true", help="execute the selected cache, pilot, or SCF operation")
    args = parser.parse_args()
    if args.prepare_ao_cache and args.scf_only:
        parser.error("--prepare-ao-cache and --scf-only are separate modes")
    if args.prepare_ao_cache:
        if args.central_data_dir is None or args.ao_cache_dir is None:
            parser.error("AO cache preparation requires --central-data-dir and --ao-cache-dir")
        if args.cache_chunk_size <= 0:
            parser.error("--cache-chunk-size must be positive")
        if args.execute:
            _build_ao_cache(args.central_data_dir.resolve(), args.ao_cache_dir, args.cache_chunk_size)
        else:
            print(json.dumps({"dry_run": True, "operation": "prepare_ao_cache", "source": str(args.central_data_dir.resolve()), "ao_cache_dir": str(_require_external(args.ao_cache_dir, "AO cache directory")), "cache_dtype": "float32", "chunk_size": args.cache_chunk_size}, indent=2), flush=True)
        return
    if args.scf_only:
        required = (args.central_data_dir, args.output_dir, args.h2_checkpoint, args.combined_checkpoint)
        if any(path is None for path in required):
            parser.error("--scf-only requires --central-data-dir, --output-dir, --h2-checkpoint, and --combined-checkpoint")
        if not args.execute:
            print(json.dumps({"dry_run": True, "operation": "scf_only", "output_dir": str(_require_external(args.output_dir, "SCF output directory"))}, indent=2), flush=True)
            return
        _run_scf_only(args.central_data_dir.resolve(), args.output_dir, args.h2_checkpoint, args.combined_checkpoint, args.scf_grid_level, args.scf_max_cycle)
        return
    if args.evaluate_checkpoints_only:
        required = (
            args.central_data_dir,
            args.ao_cache_dir,
            args.output_dir,
            args.h2_checkpoint,
            args.combined_checkpoint,
        )
        if any(path is None for path in required):
            parser.error("--evaluate-checkpoints-only requires central data, AO cache, output, and both checkpoints")
        if args.chunk_size <= 0:
            parser.error("--chunk-size must be positive")
        if not args.execute:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "operation": "evaluate_existing_checkpoints_without_optimizer_updates",
                        "output_dir": str(_require_external(args.output_dir, "pilot output directory")),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        requested_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        _refresh_checkpoint_evaluations(
            args.central_data_dir.resolve(),
            args.ao_cache_dir.resolve(),
            args.output_dir,
            args.h2_checkpoint.resolve(),
            args.combined_checkpoint.resolve(),
            requested_device,
            torch.float32 if args.dtype == "float32" else torch.float64,
            args.chunk_size,
        )
        return
    if any(path is None for path in (args.canonical_pickle, args.canonical_manifest, args.central_data_dir, args.ao_cache_dir, args.output_dir)):
        parser.error("pilot mode requires canonical inputs, --central-data-dir, --ao-cache-dir, and --output-dir")
    if min(args.predopt_chunk_size, args.chunk_size, args.steps) <= 0 or args.lr <= 0:
        parser.error("chunk sizes, steps, and learning rate must be positive")
    if not 20 <= args.steps <= 40:
        parser.error("--steps must be between 20 and 40 for this diagnostic pilot")
    if not args.execute:
        if _require_external(args.output_dir, "pilot output directory").exists():
            parser.error(f"--output-dir already exists: {args.output_dir}")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "predopt_epochs": 2,
                    "canonical_reactions": 268,
                    "branches": {"h2_only": ["H2"], "h2_beh2_co": list(SYSTEMS)},
                    "steps_per_branch": args.steps,
                    "chunk_size": args.chunk_size,
                    "ao_cache_dir": str(_require_external(args.ao_cache_dir, "AO cache directory")),
                    "scf": not args.skip_scf,
                    "output_dir": str(args.output_dir.resolve()),
                    "required_execute_flag": "--execute",
                },
                indent=2,
            ),
            flush=True,
        )
        return
    _run(args)


if __name__ == "__main__":
    main()
