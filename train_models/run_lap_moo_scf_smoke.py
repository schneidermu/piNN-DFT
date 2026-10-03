"""Run CPU RKS/Laplacian smoke checks on an explicitly pinned MOO checkpoint.

This loader is intentionally separate from ``run_lap_scf_real_smoke.py``:
one-stage MOO checkpoints use the same saved model weights but a distinct,
hash-bound training payload.  It validates that payload before constructing
the canonical Lap architecture and loading every saved model tensor.  The
default cursor is 100; checkpoints from shorter pilots must pass an explicit
``--expected-cursor`` value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "train_models")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_mrks_stencils import _make_mol
from lap_moo_protocol import (
    PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
    canonical_sha256,
    validate_protocol_metadata,
)
from lap_moo_training import _reject_stencil_checkpoint_fields
from lap_operator import (
    OPERATOR_PROTOCOL,
    operator_checkpoint_metadata,
    validate_operator_metadata,
)
from NN_models_lap import ARCHITECTURE, pcPBELMLOptimizerV2Lap

from test_models.DFT.lap_functional import LapFunctional

DEFAULT_EXPECTED_CURSOR = 100
CANONICAL_MODEL_KWARGS = {
    "num_layers": 6,
    "h_dim": 32,
    "dropout": 0.0,
    "use_g_x": True,
    "use_g_c": True,
}
PREDOPT_SHA256 = "ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8"
NPZ_SHA256 = {
    "H2": "4d99d3ec1f94ccbc3c9caa12d22428e03438dcbaeb07a765d3f30459d7e6df95",
    "BeH2": "f49532fdd6b8f2b57e67c6cd9826bd83cf4a0ddb574787a6f9931dfd6abf6d00",
    "CO": "b5e228068b44f4a9584385c969f514e13164c6c96b1160b5a2e549693e95f18e",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_external(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must stay outside the repository: {resolved}")
    return resolved


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact stored parameter bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"MOO model state entry {name!r} is not a tensor.")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"MOO model state entry {name!r} is nonfinite.")
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_moo_model_at_cursor(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_method: str,
    expected_cursor: int = DEFAULT_EXPECTED_CURSOR,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Validate exact run identity, then strictly load the saved model state."""
    if type(expected_cursor) is not int or expected_cursor <= 0:
        raise ValueError("expected_cursor must be a positive integer.")
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256.lower()
    ):
        raise ValueError("--checkpoint-sha256 must be a 64-character SHA-256 hex digest.")

    digest_before = _sha256_file(checkpoint_path)
    if digest_before != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"found {digest_before}."
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    digest_after = _sha256_file(checkpoint_path)
    if digest_after != digest_before:
        raise RuntimeError("Checkpoint changed while it was being loaded; retry with an immutable file.")
    if not isinstance(payload, dict) or payload.get("checkpoint_kind") != "lap-moo-one-stage":
        raise ValueError("Expected a one-stage MOO checkpoint payload.")
    _reject_stencil_checkpoint_fields(payload)

    protocol = payload.get("protocol_metadata")
    if not isinstance(protocol, dict):
        raise TypeError("MOO checkpoint is missing protocol metadata.")
    validate_protocol_metadata(protocol, allow_v1_read_only=True)
    protocol_sha256 = canonical_sha256(protocol)
    if payload.get("protocol_metadata_sha256") != protocol_sha256:
        raise ValueError("MOO checkpoint protocol metadata hash mismatch.")
    operator_metadata = payload.get("operator_metadata")
    validate_operator_metadata(operator_metadata)
    if operator_metadata != operator_checkpoint_metadata().to_dict():
        raise ValueError("MOO checkpoint operator metadata differs from the canonical AO protocol.")
    if payload.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError("MOO checkpoint operator protocol mismatch.")
    if protocol.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"Unsupported MOO architecture {protocol.get('architecture')!r}; "
            f"expected {ARCHITECTURE!r}."
        )
    if protocol.get("method") != expected_method:
        raise ValueError(
            f"MOO checkpoint method {protocol.get('method')!r} does not match "
            f"requested method {expected_method!r}."
        )
    if protocol.get("predopt_checkpoint_sha256") != PREDOPT_SHA256:
        raise ValueError("MOO checkpoint is not bound to the reviewed canonical predopt source.")

    direct_armijo = protocol.get("step_rule") == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    lr_schedule = protocol.get("lr_schedule")
    if direct_armijo:
        if lr_schedule != {"name": "none"}:
            raise ValueError("Direct vector-Armijo checkpoint must declare no LR scheduler.")
        if payload.get("optimizer_state_dict") is not None or payload.get("scheduler_state_dict") is not None:
            raise ValueError("Direct vector-Armijo checkpoint contains optimizer or scheduler state.")
        total_updates = None
    else:
        if not isinstance(lr_schedule, dict) or lr_schedule.get("name") != "cosine":
            raise ValueError("MOO checkpoint does not declare the reviewed cosine schedule.")
        total_updates = lr_schedule.get("total_updates")
        if type(total_updates) is not int or total_updates < expected_cursor:
            raise ValueError("MOO schedule horizon is shorter than the requested checkpoint cursor.")
        minimum_lr_ratio = lr_schedule.get("minimum_lr_ratio")
        if not isinstance(minimum_lr_ratio, (int, float)) or not math.isclose(
            float(minimum_lr_ratio), 0.1, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("MOO checkpoint does not declare the reviewed 10% cosine floor.")
        if lr_schedule.get("same_shape_for_all_methods") is not True:
            raise ValueError("MOO checkpoint does not bind the shared cosine schedule shape.")

    cursor = payload.get("sampling_cursor")
    if not isinstance(cursor, dict) or cursor.get("next_update") != expected_cursor:
        raise ValueError(f"SCF smoke requires a valid cursor-{expected_cursor} checkpoint.")
    if cursor.get("sampling_manifest_sha256") != protocol.get("sampling_manifest_sha256"):
        raise ValueError("Checkpoint cursor and MOO protocol sampling stream disagree.")
    scheduler = payload.get("scheduler_state_dict")
    if direct_armijo:
        if scheduler is not None:
            raise ValueError("Direct vector-Armijo checkpoint unexpectedly contains scheduler state.")
        scheduler_last_epoch = None
        scheduler_step_count = None
    else:
        if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != expected_cursor:
            raise ValueError("MOO checkpoint scheduler position does not match its cursor.")
        if scheduler.get("_step_count") != expected_cursor + 1:
            raise ValueError("MOO checkpoint scheduler step count does not match its cursor.")
        scheduler_last_epoch = scheduler["last_epoch"]
        scheduler_step_count = scheduler["_step_count"]

    model_kwargs = payload.get("model_kwargs")
    if not isinstance(model_kwargs, dict) or model_kwargs != CANONICAL_MODEL_KWARGS:
        raise ValueError(
            "MOO checkpoint model_kwargs differ from the canonical reviewed architecture: "
            f"{model_kwargs!r}."
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("MOO checkpoint is missing model_state_dict.")

    # The saved architecture parameters are authoritative only after exact
    # comparison with the reviewed architecture.  The state load is strict so
    # no initialized or missing model values can enter the SCF calculation.
    model = pcPBELMLOptimizerV2Lap(**model_kwargs).to(device="cpu", dtype=torch.float64)
    expected_keys = tuple(model.state_dict())
    if tuple(state) != expected_keys:
        raise ValueError("MOO model_state_dict keys/order differ from the canonical model.")
    model.load_state_dict(state, strict=True)
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest_before,
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_cursor": cursor["next_update"],
        "expected_cursor": expected_cursor,
        "sampling_manifest_sha256": cursor["sampling_manifest_sha256"],
        "method": protocol["method"],
        "architecture": protocol["architecture"],
        "model_kwargs": model_kwargs,
        "model_state_dict_sha256": _model_state_sha256(state),
        "protocol_metadata_sha256": protocol_sha256,
        "predopt_checkpoint_sha256": protocol["predopt_checkpoint_sha256"],
        "training_dtype": protocol["dtype"],
        "scf_model_dtype": str(next(model.parameters()).dtype),
        "scheduler_last_epoch": scheduler_last_epoch,
        "scheduler_step_count": scheduler_step_count,
        "lr_schedule_total_updates": total_updates,
        "protocol_metadata": protocol,
    }
    return model, metadata


def _finite_or_none(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _callback_row(envs: dict[str, Any]) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for key in ("cycle", "e_tot", "norm_gorb", "norm_ddm"):
        if key in envs:
            row[key] = _finite_or_none(envs[key])
    return row


def _record_failure(result: dict[str, Any], message: str) -> None:
    result["failures"].append(message)
    result["failure"] = "; ".join(result["failures"])


def _run_one_system(
    system: str,
    npz_path: Path,
    functional: LapFunctional,
    *,
    grid_level: int,
    max_cycle: int,
    conv_tol: float,
    max_memory_mb: int,
    verbose: int,
) -> dict[str, Any]:
    from pyscf import dft

    npz_sha256 = _sha256_file(npz_path)
    trace: list[dict[str, float | None]] = []
    result: dict[str, Any] = {
        "system": system,
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha256,
        "grid_level": grid_level,
        "max_cycle": max_cycle,
        "conv_tol": conv_tol,
        "pyscf_max_memory_mb": max_memory_mb,
        "converged": False,
        "total_energy_hartree": None,
        "cycles_recorded": 0,
        "convergence_trace": trace,
        "all_cycle_energies_finite": False,
        "vtau_max_abs": None,
        "tau_independent_exact": None,
        "tau_test_grid_points": 0,
        "all_xc_outputs_finite": None,
        "failures": [],
        "failure": None,
    }
    try:
        if npz_sha256 != NPZ_SHA256[system]:
            raise ValueError(
                f"{system} NPZ SHA-256 differs from the audited mRKS source: {npz_sha256}."
            )
        with np.load(npz_path, allow_pickle=False) as npz:
            mol, _basis, _charge, spin = _make_mol(npz)
        if spin != 0:
            raise ValueError(f"{system} is not closed shell (spin={spin}).")

        mf = functional.make_rks(mol)
        mf.grids.level = grid_level
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol
        mf.max_memory = max_memory_mb
        mf.verbose = verbose
        mf.callback = lambda envs: trace.append(_callback_row(envs))

        energy = float(mf.kernel())
        result["converged"] = bool(mf.converged)
        result["total_energy_hartree"] = _finite_or_none(energy)
        result["cycles_recorded"] = len(trace)
        result["all_cycle_energies_finite"] = bool(trace) and all(
            row.get("e_tot") is not None and math.isfinite(row["e_tot"])
            for row in trace
        )
        if not math.isfinite(energy):
            _record_failure(result, "PySCF returned a nonfinite total energy.")
        elif not mf.converged:
            _record_failure(result, f"RKS did not converge within {max_cycle} cycles.")

        # Use the final SCF density for the adapter contract check.  The check
        # perturbs only PySCF's tau channel; all reported XC values must remain
        # bitwise identical and vtau must be exactly zero.
        mf.grids.build(with_non0tab=True)
        dm = mf.make_rdm1()
        point_count = min(32, len(mf.grids.coords))
        if point_count == 0:
            raise ValueError(f"{system} generated no SCF grid points.")
        ao = dft.numint.eval_ao(mol, mf.grids.coords[:point_count], deriv=2)
        rho = dft.numint.eval_rho(
            mol, ao, dm, xctype="MGGA", hermi=1, with_lapl=True
        )
        before = functional.eval_xc("", rho, spin=0, deriv=1)
        shifted = np.array(rho, copy=True)
        shifted[5] += 1e6
        after = functional.eval_xc("", shifted, spin=0, deriv=1)
        before_values = (before[0], *before[1])
        after_values = (after[0], *after[1])
        tau_independent = len(before_values) == len(after_values) and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(before_values, after_values)
        )
        vtau = np.asarray(before[1][3])
        all_xc_finite = all(
            np.isfinite(np.asarray(value)).all()
            for value in (*before_values, *after_values)
        )
        vtau_is_zero = bool(np.all(vtau == 0))
        result.update(
            {
                "vtau_max_abs": _finite_or_none(np.max(np.abs(vtau), initial=0.0)),
                "vtau_exactly_zero": vtau_is_zero,
                "tau_independent_exact": bool(tau_independent),
                "tau_test_grid_points": point_count,
                "all_xc_outputs_finite": bool(all_xc_finite),
                "ao_dimension": int(mol.nao_nr()),
                "electron_count": int(mol.nelectron),
            }
        )
        if not tau_independent:
            _record_failure(result, "XC output changed after tau-only perturbation.")
        if not vtau_is_zero:
            _record_failure(result, "LapFunctional returned nonzero vtau.")
        if not all_xc_finite:
            _record_failure(result, "LapFunctional returned nonfinite XC output.")
    except Exception as exc:  # noqa: BLE001 -- Preserve every real SCF failure trace.
        _record_failure(result, f"{type(exc).__name__}: {exc}")
        result["failure_traceback"] = traceback.format_exc()
        result["cycles_recorded"] = len(trace)
        result["all_cycle_energies_finite"] = bool(trace) and all(
            row.get("e_tot") is not None and math.isfinite(row["e_tot"])
            for row in trace
        )
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--method", required=True, choices=("fixed", "imtl_g", "cagrad", "nash_mtl", "pcd")
    )
    parser.add_argument(
        "--expected-cursor",
        type=int,
        default=DEFAULT_EXPECTED_CURSOR,
        help="Required checkpoint/scheduler cursor; defaults to 100.",
    )
    parser.add_argument("--npz-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--max-cycle", type=int, default=40)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--max-memory-mb", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--verbose", type=int, default=3)
    args = parser.parse_args()

    if args.grid_level < 0 or args.max_cycle <= 0 or args.conv_tol <= 0:
        parser.error("grid level, cycle limit, and convergence tolerance must be positive/valid")
    if args.max_memory_mb <= 0 or args.num_threads <= 0:
        parser.error("memory and thread limits must be positive")
    if args.expected_cursor <= 0:
        parser.error("expected cursor must be positive")

    checkpoint = _require_external(args.checkpoint, "MOO checkpoint").resolve(strict=True)
    npz_root = _require_external(args.npz_root, "mRKS NPZ root").resolve(strict=True)
    output = _require_external(args.output, "Output JSON")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing SCF result: {output}")
    if not npz_root.is_dir():
        raise NotADirectoryError(npz_root)

    torch.set_num_threads(args.num_threads)
    from pyscf import __version__ as pyscf_version
    from pyscf import lib

    lib.num_threads(args.num_threads)
    model, checkpoint_metadata = _load_moo_model_at_cursor(
        checkpoint,
        expected_sha256=args.checkpoint_sha256,
        expected_method=args.method,
        expected_cursor=args.expected_cursor,
    )
    functional = LapFunctional(model)
    systems = []
    for name in ("H2", "BeH2", "CO"):
        npz_path = (npz_root / name / "inp_mrks.npz").resolve(strict=True)
        systems.append(
            _run_one_system(
                name,
                npz_path,
                functional,
                grid_level=args.grid_level,
                max_cycle=args.max_cycle,
                conv_tol=args.conv_tol,
                max_memory_mb=args.max_memory_mb,
                verbose=args.verbose,
            )
        )

    for system_result in systems:
        system_result["passed"] = bool(
            system_result.get("converged")
            and system_result.get("total_energy_hartree") is not None
            and system_result.get("all_cycle_energies_finite")
            and system_result.get("vtau_exactly_zero") is True
            and system_result.get("tau_independent_exact") is True
            and system_result.get("all_xc_outputs_finite") is True
            and system_result.get("failure") is None
        )

    report = {
        "schema": "lap-moo-cursor-real-scf-smoke-v1",
        "method": args.method,
        "expected_cursor": args.expected_cursor,
        "execution_device": "cpu",
        "torch_version": torch.__version__,
        "pyscf_version": pyscf_version,
        "torch_cuda_available_but_unused": bool(torch.cuda.is_available()),
        "cpu_threads": args.num_threads,
        "checkpoint": checkpoint_metadata,
        "systems": systems,
        "all_systems_passed": all(system["passed"] for system in systems),
    }
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["all_systems_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
