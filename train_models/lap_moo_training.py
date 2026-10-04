"""One-stage Lap MOO update, objective, optimizer, and checkpoint helpers."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

try:
    from .lap_moo_protocol import (
        PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        TASK_NAMES,
        canonical_sha256,
        validate_cursor,
        validate_protocol_metadata,
        validate_sampling_manifest,
    )
    from .lap_operator import (
        OPERATOR_PROTOCOL,
        LapEnergy,
        assemble_rks_operator,
        operator_checkpoint_metadata,
        operator_loss,
        validate_operator_metadata,
    )
    from .lap_vxc import integrated_energy
except ImportError:  # pragma: no cover - direct script imports
    from lap_moo_protocol import (
        PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
        TASK_NAMES,
        canonical_sha256,
        validate_cursor,
        validate_protocol_metadata,
        validate_sampling_manifest,
    )
    from lap_operator import (
        OPERATOR_PROTOCOL,
        LapEnergy,
        assemble_rks_operator,
        operator_checkpoint_metadata,
        operator_loss,
        validate_operator_metadata,
    )
    from lap_vxc import integrated_energy


@dataclass(frozen=True)
class AOFactorChunk:
    rows: slice
    phi: torch.Tensor
    grad_phi: torch.Tensor
    lap_phi: torch.Tensor


@dataclass(frozen=True)
class MRKSOperatorSystem:
    """One full mRKS system, with AO factors retained in point chunks."""

    name: str
    features: torch.Tensor
    weights: torch.Tensor
    exc_target: torch.Tensor
    reference_operator: torch.Tensor
    overlap: torch.Tensor
    ao_chunks: tuple[AOFactorChunk, ...]


@dataclass(frozen=True)
class UpdateResult:
    losses: dict[str, float]
    diagnostics: dict[str, Any]
    aggregator_state: dict[str, Any] | None
    learning_rate: float
    accepted: bool = True
    stop_reason: str | None = None


def named_trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return one stable name per unique trainable parameter."""
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if parameter.requires_grad
    }
    return dict(sorted(parameters.items()))


def _scalar_loss(value: Any, task: str) -> torch.Tensor:
    loss = value[0] if isinstance(value, tuple) else value
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError(f"Objective {task!r} must return one scalar tensor.")
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(f"Objective {task!r} is nonfinite.")
    return loss.reshape(())


def compute_isolated_task_gradients(
    model: nn.Module,
    objective_factories: Mapping[str, Callable[[], torch.Tensor]],
    *, task_order: tuple[str, ...] = TASK_NAMES,
) -> tuple[dict[str, float], dict[str, dict[str, torch.Tensor | None]]]:
    """Evaluate chem/E/V separately and return their unmodified parameter grads."""
    if tuple(objective_factories.keys()) != task_order:
        raise ValueError(f"Objectives must be ordered exactly as {task_order}.")
    parameters = named_trainable_parameters(model)
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    ordered_parameters = tuple(parameters.values())
    losses: dict[str, float] = {}
    task_grads: dict[str, dict[str, torch.Tensor | None]] = {}
    for task in task_order:
        factory = objective_factories[task]
        if isinstance(factory, ChemistryBatchObjective):
            value, gradients = factory.value_and_grad()
            if set(gradients) != set(parameters):
                raise ValueError("Chemistry shadow gradients have different parameter names.")
            losses[task], task_grads[task] = value, gradients
        else:
            loss = _scalar_loss(objective_factories[task](), task)
            grads = torch.autograd.grad(
                loss,
                ordered_parameters,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
            losses[task] = float(loss.detach().double().cpu())
            task_grads[task] = {
                name: None if grad is None else grad.detach()
                for name, grad in zip(parameters, grads)
            }
            del loss, grads
    if any(isinstance(factory, ChemistryBatchObjective) for factory in objective_factories.values()):
        # Preserve shadow derivatives; widen only existing lower-precision grads.
        task_grads = {
            task: {name: None if grad is None else grad.to(torch.float64) for name, grad in values.items()}
            for task, values in task_grads.items()
        }
    return losses, task_grads


def average_raw_task_gradients(
    task_grads: Mapping[str, Mapping[str, torch.Tensor | None]],
    *,
    world_size: int | None = None,
    task_order: tuple[str, ...] = TASK_NAMES,
) -> dict[str, dict[str, torch.Tensor]]:
    """All-reduce every task/parameter tensor before nonlinear aggregation."""
    if tuple(task_grads.keys()) != task_order:
        raise ValueError(f"Task gradients must be ordered exactly as {task_order}.")
    names = tuple(next(iter(task_grads.values())).keys())
    if any(tuple(task_grads[task].keys()) != names for task in task_order):
        raise ValueError("Every task must provide the same ordered parameter names.")
    distributed = dist.is_available() and dist.is_initialized()
    actual_world_size = dist.get_world_size() if distributed else 1
    if world_size is not None and int(world_size) != actual_world_size:
        raise ValueError(
            f"world_size={world_size} does not match initialized group {actual_world_size}."
        )
    if actual_world_size > 1 and not distributed:
        raise RuntimeError("Distributed task gradients require an initialized process group.")

    averaged: dict[str, dict[str, torch.Tensor]] = {}
    for task in task_order:
        averaged[task] = {}
        for name in names:
            grad = task_grads[task][name]
            if grad is None:
                raise ValueError(
                    "Materialize parameter-shaped zeros before all-reduce; missing "
                    f"gradient tensor for {task}/{name}."
                )
            value = grad.detach().contiguous().clone()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite raw gradient for {task}/{name}.")
            if distributed:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value.div_(actual_world_size)
            averaged[task][name] = value
    return averaged


def materialize_task_zeros(
    model: nn.Module,
    task_grads: Mapping[str, Mapping[str, torch.Tensor | None]],
    *, task_order: tuple[str, ...] = TASK_NAMES,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replace unused entries with exact zeros using model parameter shapes."""
    parameters = named_trainable_parameters(model)
    if tuple(task_grads.keys()) != task_order:
        raise ValueError(f"Task gradients must be ordered exactly as {task_order}.")
    output: dict[str, dict[str, torch.Tensor]] = {}
    for task in task_order:
        if set(task_grads[task]) != set(parameters):
            raise ValueError(f"Task {task!r} has a different parameter-name set.")
        output[task] = {}
        for name, parameter in parameters.items():
            grad = task_grads[task][name]
            if grad is None:
                dtype = next(
                    (task_grads[other][name].dtype for other in task_order if task_grads[other][name] is not None),
                    parameter.dtype,
                )
                output[task][name] = torch.zeros_like(parameter, dtype=dtype)
            elif grad.shape != parameter.shape or grad.device != parameter.device:
                raise ValueError(f"Gradient shape/device mismatch at {task}/{name}.")
            else:
                output[task][name] = grad
    return output


def _dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    value = 0.0
    for name in left:
        value += float(torch.sum(left[name].detach().double() * right[name].detach().double()).cpu())
    return value


def _norm(gradient: Mapping[str, torch.Tensor]) -> float:
    return max(_dot(gradient, gradient), 0.0) ** 0.5


def _gradient_diagnostics(
    raw: Mapping[str, Mapping[str, torch.Tensor]],
    joint: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    tasks = tuple(raw)
    norms = {task: _norm(raw[task]) for task in tasks}
    cosines = {}
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1 :]:
            denominator = norms[left] * norms[right]
            cosines[f"{left}:{right}"] = (
                _dot(raw[left], raw[right]) / denominator if denominator else 0.0
            )
    return {
        "raw_gradient_norms": norms,
        "raw_gradient_cosines": cosines,
        "joint_gradient_norm": _norm(joint),
        "task_directional_dots": {task: _dot(raw[task], joint) for task in tasks},
    }


def _average_task_losses(losses: Mapping[str, float], model: nn.Module) -> dict[str, float]:
    tasks = tuple(losses)
    values = torch.tensor(
        [float(losses[task]) for task in tasks],
        dtype=torch.float64,
        device=next(iter(named_trainable_parameters(model).values())).device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(dist.get_world_size())
    return {task: float(value.cpu()) for task, value in zip(tasks, values)}


def _all_ranks_true(value: bool, model: nn.Module) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return value
    parameter = next(iter(named_trainable_parameters(model).values()))
    flag = torch.tensor([int(value)], dtype=torch.int32, device=parameter.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _evaluate_trial_losses(
    model: nn.Module,
    objective_factories: Mapping[str, Callable[[], torch.Tensor]],
    *, task_order: tuple[str, ...] = TASK_NAMES,
) -> tuple[dict[str, float] | None, str | None]:
    """Evaluate ordered local scalars, then take the existing unweighted rank mean."""
    local_losses: dict[str, float] = {}
    for task in objective_factories:
        error = None
        try:
            value = objective_factories[task]()
            loss = value[0] if isinstance(value, tuple) else value
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError(f"Objective {task!r} must return one scalar tensor.")
            loss_value = float(loss.detach().double().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Objective {task!r} is nonfinite.")
            local_losses[task] = loss_value
        except Exception as exc:  # noqa: BLE001 - sync a rank-local trial failure before loss reduction
            error = exc
        if not _all_ranks_true(error is None, model):
            return None, "nonfinite_or_failed_task_evaluation"
    losses = _average_task_losses(local_losses, model)
    if not all(math.isfinite(value) for value in losses.values()):
        return None, "nonfinite_global_task_mean"
    return losses, None


def _clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _validate_vector_armijo_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(options, Mapping) or set(options) != {
        "c",
        "rho",
        "max_backtracks",
        "initial_step_size",
    }:
        raise ValueError("Direct PCD stepping requires c, rho, max_backtracks, and initial_step_size.")
    c = options["c"]
    rho = options["rho"]
    alpha0 = options["initial_step_size"]
    max_backtracks = options["max_backtracks"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)) or not 0.0 < float(c) < 1.0:
        raise ValueError("Vector-Armijo c must be finite and in (0, 1).")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(float(rho)) or not 0.0 < float(rho) < 1.0:
        raise ValueError("Vector-Armijo rho must be finite and in (0, 1).")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError("Vector-Armijo max_backtracks must be a nonnegative integer.")
    if isinstance(alpha0, bool) or not isinstance(alpha0, (int, float)) or not math.isfinite(float(alpha0)) or float(alpha0) <= 0.0:
        raise ValueError("Vector-Armijo initial_step_size must be finite and positive.")
    return {
        "c": float(c),
        "rho": float(rho),
        "max_backtracks": max_backtracks,
        "initial_step_size": float(alpha0),
    }


def apply_joint_gradient(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    joint_gradient: Mapping[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    parameters = named_trainable_parameters(model)
    if set(joint_gradient) != set(parameters):
        raise ValueError("MOO output must contain every unique trainable parameter.")
    for name, parameter in parameters.items():
        gradient = joint_gradient[name]
        if gradient is None:
            parameter.grad = None
            continue
        if gradient.shape != parameter.shape or gradient.device != parameter.device:
            raise ValueError(f"Joint-gradient shape/device mismatch at {name}.")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite joint gradient at {name}.")
        parameter.grad = gradient.detach().clone()
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    optimizer.step()
    return {
        name: parameter.detach() - before[name]
        for name, parameter in parameters.items()
    }


def train_moo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    objective_factories: Mapping[str, Callable[[], torch.Tensor]],
    *,
    method: str,
    hyperparameters: Mapping[str, Any] | None = None,
    aggregator_state: Mapping[str, Any] | None = None,
    aggregator: Callable[..., tuple[Mapping[str, torch.Tensor | None], dict[str, Any], dict[str, Any]]] | None = None,
    scheduler: Any = None,
    world_size: int | None = None,
    record_update_geometry: bool = True,
    step_rule: str = "optimizer",
    vector_armijo: Mapping[str, Any] | None = None,
    task_order: tuple[str, ...] = TASK_NAMES,
) -> UpdateResult:
    """One canonical update: raw local tasks -> all-reduce -> MOO -> optimizer."""
    if (len(task_order) < 2 or len(set(task_order)) != len(task_order)
            or any(not isinstance(task, str) or not task for task in task_order)):
        raise ValueError("Explicit task order must contain at least two unique names.")
    direct_armijo = step_rule == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    if step_rule not in ("optimizer", PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE):
        raise ValueError(f"Unsupported MOO step rule {step_rule!r}.")
    if direct_armijo:
        if method != "pcd" or optimizer is not None or scheduler is not None:
            raise ValueError("Direct vector-Armijo requires PCD with no optimizer or scheduler.")
        armijo = _validate_vector_armijo_options(vector_armijo)
        model_before = _clone_model_state(model)
        rng_before = capture_rng_state()
    else:
        if optimizer is None:
            raise ValueError("The optimizer step rule requires an optimizer.")
        if vector_armijo is not None:
            raise ValueError("Vector-Armijo parameters require the direct vector-Armijo step rule.")
        optimizer.zero_grad(set_to_none=True)
        model_before = None
        rng_before = None
        armijo = None
    try:
        losses, sparse_grads = compute_isolated_task_gradients(model, objective_factories, task_order=task_order)
        losses = _average_task_losses(losses, model)
        dense_local = materialize_task_zeros(model, sparse_grads, task_order=task_order)
        raw = average_raw_task_gradients(dense_local, world_size=world_size, task_order=task_order)
    except Exception:
        if direct_armijo:
            assert model_before is not None and rng_before is not None
            _restore_model_state(model, model_before)
            restore_rng_state(rng_before)
        raise
    if aggregator is None:
        try:
            from .moo_aggregators import aggregate_task_gradients
        except ImportError:  # pragma: no cover - direct script import
            from moo_aggregators import aggregate_task_gradients

        aggregator = aggregate_task_gradients
    try:
        joint, method_diagnostics, next_state = aggregator(
            raw,
            method=method,
            hyperparameters=hyperparameters,
            state=copy.deepcopy(aggregator_state) if direct_armijo else aggregator_state,
            **({"task_order": task_order} if task_order != TASK_NAMES else {}),
        )
    except Exception:
        if direct_armijo:
            assert model_before is not None and rng_before is not None
            _restore_model_state(model, model_before)
            restore_rng_state(rng_before)
        raise
    parameters = named_trainable_parameters(model)
    dense_joint = {
        name: (torch.zeros_like(parameter) if joint.get(name) is None else joint[name])
        for name, parameter in parameters.items()
    }
    diagnostics = _gradient_diagnostics(raw, dense_joint)
    diagnostics.update(method_diagnostics)
    if direct_armijo:
        assert model_before is not None and rng_before is not None and armijo is not None
        model_at_base = _clone_model_state(model)
        rng_after_gradients = capture_rng_state()
        alpha0 = armijo["initial_step_size"]
        proposal = {name: -alpha0 * gradient for name, gradient in dense_joint.items()}
        slopes = {task: _dot(raw[task], proposal) for task in task_order}
        diagnostics["vector_armijo"] = {
            **armijo,
            "step_rule": PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE,
            "directional_slopes": slopes,
            "trials": [],
        }
        accepted_t: float | None = None
        accepted_delta: dict[str, torch.Tensor] | None = None
        accepted_parameters: dict[str, torch.Tensor] | None = None
        try:
            if method_diagnostics.get("feasible") is False or not all(torch.isfinite(value).all() for value in proposal.values()) or not all(
                math.isfinite(value) and value < 0.0 for value in slopes.values()
            ):
                stop_reason = "no_finite_common_descent_direction"
            else:
                stop_reason = "no_accepted_step_within_backtrack_cap"
                for backtrack in range(armijo["max_backtracks"] + 1):
                    t = armijo["rho"] ** backtrack
                    trial_parameters = {}
                    with torch.no_grad():
                        _restore_model_state(model, model_at_base)
                        parameters = named_trainable_parameters(model)
                        for name, parameter in parameters.items():
                            trial_parameters[name] = model_at_base[name] + t * proposal[name]
                        no_op = all(
                            torch.equal(trial_parameters[name], parameters[name].detach())
                            for name in parameters
                        )
                        if not no_op:
                            for name, parameter in parameters.items():
                                parameter.copy_(trial_parameters[name])
                                trial_parameters[name] = parameter.detach().clone()
                    if no_op:
                        diagnostics["vector_armijo"]["trials"].append(
                            {
                                "t": t, "finite": True, "no_op": True,
                                "requested_delta_norm": _norm({name: t * value for name, value in proposal.items()}),
                                "realized_delta_norm": 0.0, "requested_realized_cosine": 0.0,
                                "zero_coordinate_fraction": 1.0,
                                "requested_task_dots": {task: t * slopes[task] for task in task_order},
                                "realized_task_dots": {task: 0.0 for task in task_order},
                                "realized_common_descent": False,
                                "losses": dict(losses),
                                "actual_reductions": {task: 0.0 for task in task_order},
                                "strict_all_task_decrease": False,
                            }
                        )
                        stop_reason = "candidate_unchanged_at_float_precision"
                        break
                    requested_delta = {name: t * proposal[name] for name in proposal}
                    realized_delta = {
                        name: trial_parameters[name].double() - model_at_base[name].double()
                        for name in parameters
                    }
                    requested_norm, realized_norm = _norm(requested_delta), _norm(realized_delta)
                    realized_dots = {task: _dot(raw[task], realized_delta) for task in task_order}
                    realized_descent = all(math.isfinite(value) and value < 0.0 for value in realized_dots.values())
                    trial_record: dict[str, Any] = {
                        "t": t, "finite": False, "no_op": False,
                        "requested_delta_norm": requested_norm,
                        "realized_delta_norm": realized_norm,
                        "requested_realized_cosine": (
                            _dot(requested_delta, realized_delta) / (requested_norm * realized_norm)
                            if requested_norm and realized_norm else 0.0
                        ),
                        "zero_coordinate_fraction": sum(int((value == 0).sum()) for value in realized_delta.values())
                            / sum(value.numel() for value in realized_delta.values()),
                        "requested_task_dots": {task: t * slopes[task] for task in task_order},
                        "realized_task_dots": realized_dots,
                        "realized_common_descent": realized_descent,
                    }
                    try:
                        restore_rng_state(rng_after_gradients)
                        trial_losses, trial_error = _evaluate_trial_losses(model, objective_factories)
                        if trial_losses is not None:
                            predicted = {task: -t * slopes[task] for task in task_order}
                            actual = {task: losses[task] - trial_losses[task] for task in task_order}
                            margins = {
                                task: losses[task] + armijo["c"] * t * slopes[task] - trial_losses[task]
                                for task in task_order
                            }
                            ratios = {
                                task: actual[task] / predicted[task]
                                for task in task_order
                            }
                            armijo_pass = all(margins[task] >= 0.0 for task in task_order)
                            strict_decrease = all(actual[task] > 0.0 for task in task_order)
                            trial_record.update(
                                {
                                    "finite": True,
                                    "losses": trial_losses,
                                    "predicted_reductions": predicted,
                                    "actual_reductions": actual,
                                    "actual_to_predicted_ratios": ratios,
                                    "armijo_margins": margins,
                                    "armijo_pass": armijo_pass,
                                    "strict_all_task_decrease": strict_decrease,
                                }
                            )
                            if armijo_pass and strict_decrease and realized_descent:
                                accepted_t = t
                                accepted_parameters = trial_parameters
                                accepted_delta = {
                                    name: realized_delta[name]
                                    for name in accepted_parameters
                                }
                                stop_reason = None
                        else:
                            trial_record["error"] = trial_error
                    finally:
                        _restore_model_state(model, model_at_base)
                        restore_rng_state(rng_after_gradients)
                    trial_record["parameter_delta_norm"] = _norm(
                        {name: t * proposal[name] for name in proposal}
                    )
                    diagnostics["vector_armijo"]["trials"].append(trial_record)
                    if stop_reason is None:
                        break
        except Exception:
            _restore_model_state(model, model_before)
            restore_rng_state(rng_before)
            raise
        diagnostics["vector_armijo"]["accepted_t"] = accepted_t
        diagnostics["vector_armijo"]["backtracks"] = (
            len(diagnostics["vector_armijo"]["trials"]) - 1
            if diagnostics["vector_armijo"]["trials"]
            else 0
        )
        if stop_reason is not None:
            _restore_model_state(model, model_before)
            restore_rng_state(rng_before)
            diagnostics["vector_armijo"]["accepted"] = False
            diagnostics["vector_armijo"]["stop_reason"] = stop_reason
            return UpdateResult(
                losses,
                diagnostics,
                copy.deepcopy(aggregator_state) if aggregator_state is not None else None,
                alpha0,
                accepted=False,
                stop_reason=stop_reason,
            )
        assert accepted_t is not None and accepted_delta is not None and accepted_parameters is not None
        with torch.no_grad():
            parameters = named_trainable_parameters(model)
            for name, parameter in parameters.items():
                parameter.copy_(accepted_parameters[name])
        delta_norm = _norm(accepted_delta)
        diagnostics["parameter_update_norm"] = delta_norm
        diagnostics["joint_gradient_update_cosine"] = (
            _dot(dense_joint, accepted_delta)
            / (diagnostics["joint_gradient_norm"] * delta_norm)
            if diagnostics["joint_gradient_norm"] and delta_norm
            else 0.0
        )
        diagnostics["task_update_dots"] = {
            task: _dot(raw[task], accepted_delta) for task in task_order
        }
        diagnostics["vector_armijo"]["accepted"] = True
        diagnostics["vector_armijo"]["stop_reason"] = None
        return UpdateResult(losses, diagnostics, dict(next_state), alpha0)
    if record_update_geometry:
        assert optimizer is not None
        update = apply_joint_gradient(model, optimizer, joint)
        delta_norm = _norm(update)
        diagnostics["parameter_update_norm"] = delta_norm
        diagnostics["joint_gradient_update_cosine"] = (
            _dot(dense_joint, update) / (diagnostics["joint_gradient_norm"] * delta_norm)
            if diagnostics["joint_gradient_norm"] and delta_norm
            else 0.0
        )
        diagnostics["task_update_dots"] = {
            task: _dot(raw[task], update) for task in task_order
        }
    else:
        assert optimizer is not None
        apply_joint_gradient(model, optimizer, joint)
    if scheduler is not None:
        scheduler.step()
    assert optimizer is not None
    learning_rate = float(optimizer.param_groups[0]["lr"])
    return UpdateResult(losses, diagnostics, dict(next_state), learning_rate)


def make_reaction_objective(model: nn.Module, reaction: Mapping[str, Any], *, device, dtype, dispersions=None):
    """Bind the existing Minnesota reaction loss without changing its definition."""
    try:
        from .lap_training import reaction_loss
    except ImportError:  # pragma: no cover
        from lap_training import reaction_loss

    def objective() -> torch.Tensor:
        return reaction_loss(
            model,
            reaction,
            reaction["Energy"],
            device,
            dtype,
            dispersions,
        )

    return objective


def make_mrks_objective_factories(
    model: nn.Module,
    system: MRKSOperatorSystem,
    *,
    point_chunk_size: int,
    dispersions: Mapping[str, float] | None,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """Build unchanged mRKS E and h-free AO-operator scalar objectives."""
    if point_chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive.")
    if system.features.ndim != 2 or system.features.shape[1] != 10:
        raise ValueError("mRKS features must use the ten-column central-grid layout.")
    if system.weights.shape != (len(system.features),):
        raise ValueError("mRKS quadrature weights do not match its full grid.")
    if not system.ao_chunks:
        raise ValueError("mRKS operator objectives require full-system AO chunks.")

    def energy_objective() -> torch.Tensor:
        energy = LapEnergy(model)
        prediction = integrated_energy(
            energy, system.features[:, None, :], system.weights, point_chunk_size
        )
        # This is the same one-time Name lookup/addition used by historical mRKS.
        try:
            from .optuna_joint import _add_mrks_dispersion_once, batch_exc
        except ImportError:  # pragma: no cover
            from optuna_joint import _add_mrks_dispersion_once, batch_exc
        prediction = _add_mrks_dispersion_once(
            prediction, system.name, None if dispersions is None else dict(dispersions), True
        )
        target = torch.as_tensor(
            system.exc_target, device=prediction.device, dtype=prediction.dtype
        ).reshape(1)
        return batch_exc([system.name], prediction.reshape(1), target)

    def operator_objective() -> torch.Tensor:
        energy = LapEnergy(model)
        predicted = system.reference_operator.new_zeros(system.reference_operator.shape)
        next_row = 0
        for chunk in system.ao_chunks:
            rows = chunk.rows
            if rows.start != next_row:
                raise ValueError("AO chunks must cover the grid contiguously and in order.")
            if rows.stop is None or rows.stop <= rows.start:
                raise ValueError("AO chunk slices must be nonempty finite ranges.")
            chunk_operator = assemble_rks_operator(
                energy,
                system.features[rows],
                system.weights[rows],
                chunk.phi,
                chunk.grad_phi,
                chunk.lap_phi,
                chunk_size=point_chunk_size,
                create_graph=True,
            )
            predicted = predicted + chunk_operator.to(
                device=predicted.device, dtype=predicted.dtype
            )
            next_row = rows.stop
        if next_row != len(system.features):
            raise ValueError("AO chunks do not cover the complete mRKS grid.")
        return operator_loss(predicted, system.reference_operator, system.overlap)

    return energy_objective, operator_objective


@dataclass
class ChemistryBatchObjective:
    """Bounded-memory weighted chemistry mean with exact F64 shadow derivatives.

    The shadow is synchronized from the main stored state for every evaluation.
    Local source values are widened, not regenerated at higher precision.
    Parameter names map the derivative of the F64 scalar to the main coordinates;
    gradients remain F64 and never pass through F32 leaf-gradient storage.
    """

    model: nn.Module
    shadow: nn.Module
    reactions: tuple[Mapping[str, Any], ...]
    weights: tuple[float, ...]
    dispersions: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if (not self.reactions or len(self.reactions) != len(self.weights)
                or any(not math.isfinite(w) or w <= 0.0 for w in self.weights)
                or not math.isclose(sum(self.weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Chemistry batch needs positive unit-sum weights aligned with reactions.")
        if tuple(named_trainable_parameters(self.model)) != tuple(named_trainable_parameters(self.shadow)):
            raise ValueError("Chemistry shadow parameter names differ from the main model.")
        if any(p.dtype != torch.float64 for p in self.shadow.parameters()):
            raise ValueError("Chemistry shadow parameters must be F64.")

    def _evaluate(self, *, gradient: bool) -> tuple[float, dict[str, torch.Tensor]]:
        self.shadow.load_state_dict(self.model.state_dict(), strict=True)
        self.shadow.train(self.model.training)
        parameters = named_trainable_parameters(self.shadow)
        device = next(iter(parameters.values())).device
        total = 0.0
        accumulated = {name: torch.zeros_like(p) for name, p in parameters.items()} if gradient else {}
        for reaction, weight in zip(self.reactions, self.weights, strict=True):
            factory = make_reaction_objective(
                self.shadow, reaction, device=device, dtype=torch.float64, dispersions=self.dispersions
            )
            with torch.set_grad_enabled(gradient):
                loss = _scalar_loss(factory(), "chemistry")
            total += weight * float(loss.detach().double().cpu())
            if gradient:
                grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
                for name, value in zip(parameters, grads, strict=True):
                    if value is not None:
                        accumulated[name].add_(value.detach(), alpha=weight)
                del grads
            del loss
        if not math.isfinite(total) or any(not torch.isfinite(g).all() for g in accumulated.values()):
            raise FloatingPointError("Chemistry batch scalar or gradients are nonfinite.")
        return total, accumulated

    def value_and_grad(self) -> tuple[float, dict[str, torch.Tensor]]:
        return self._evaluate(gradient=True)

    def __call__(self) -> torch.Tensor:
        value, _ = self._evaluate(gradient=False)
        return next(self.shadow.parameters()).new_tensor(value)


def make_three_objective_factories(
    model: nn.Module,
    reaction: Mapping[str, Any],
    system: MRKSOperatorSystem,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reaction_dispersions: Mapping[str, Any] | None,
    mrks_dispersions: Mapping[str, float] | None,
    point_chunk_size: int,
) -> dict[str, Callable[[], torch.Tensor]]:
    """Bind one reaction and its paired full-grid E/V mRKS system in task order."""
    energy_factory, operator_factory = make_mrks_objective_factories(
        model,
        system,
        point_chunk_size=point_chunk_size,
        dispersions=mrks_dispersions,
    )
    return {
        "chem": make_reaction_objective(
            model,
            reaction,
            device=device,
            dtype=dtype,
            dispersions=None if reaction_dispersions is None else dict(reaction_dispersions),
        ),
        "exc": energy_factory,
        "op": operator_factory,
    }


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Use one smooth cosine schedule with the same shape for every method."""
    if total_updates <= 0 or not 0.0 <= min_lr_ratio < 1.0:
        raise ValueError("Invalid cosine schedule duration or minimum-LR ratio.")

    def multiplier(update: int) -> float:
        position = min(max(update, 0), total_updates) / total_updates
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + float(np.cos(np.pi * position))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _gather_rng_states() -> list[dict[str, Any]]:
    state = capture_rng_state()
    if not dist.is_available() or not dist.is_initialized():
        return [state]
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, state)
    return [item for item in gathered if item is not None]


def _reject_stencil_checkpoint_fields(value: Any) -> None:
    forbidden = {"h", "h_bohr", "stencil", "stencil_order", "stencil_version", "derivative_order"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                raise ValueError("One-stage MOO checkpoints cannot contain spatial-stencil metadata.")
            _reject_stencil_checkpoint_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_stencil_checkpoint_fields(nested)


def _validate_pcd_world_size(
    protocol_metadata: Mapping[str, Any], sampling_manifest: Mapping[str, Any]
) -> None:
    if protocol_metadata.get("method") != "pcd":
        return
    metadata_world_size = protocol_metadata.get("world_size")
    manifest_world_size = sampling_manifest.get("world_size")
    actual_world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if (
        type(metadata_world_size) is not int
        or type(manifest_world_size) is not int
        or metadata_world_size != manifest_world_size
        or metadata_world_size != actual_world_size
    ):
        raise ValueError(
            "PCD protocol, sampling manifest, and initialized process group world sizes differ."
        )


def _validate_rng_state(state: Any) -> None:
    """Check a captured RNG bundle without changing process-global RNG state."""
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("MOO checkpoint contains a malformed per-rank RNG state.")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
    except (TypeError, ValueError) as exc:
        raise ValueError("MOO checkpoint contains an invalid Python or NumPy RNG state.") from exc
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise ValueError("MOO checkpoint contains an invalid CPU Torch RNG state.")
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except RuntimeError as exc:
        raise ValueError("MOO checkpoint contains an invalid CPU Torch RNG state.") from exc
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)) or any(
        not isinstance(item, torch.Tensor)
        or item.device.type != "cpu"
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in cuda_states
    ):
        raise ValueError("MOO checkpoint contains invalid CUDA RNG states.")


def _validated_pcd_aggregator_state(
    protocol_metadata: Mapping[str, Any],
    aggregator_state: Mapping[str, Any] | None,
    *,
    cursor: int,
) -> dict[str, Any]:
    """Fail closed on PCD EMA state before saving or mutating resumed objects."""
    if protocol_metadata.get("method") != "pcd":
        return copy.deepcopy(dict(aggregator_state or {}))

    hyperparameters = protocol_metadata.get("method_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise TypeError("PCD protocol method hyperparameters must be a mapping.")
    expected = {
        "version": 1,
        "method": "pcd",
        "task_order": list(protocol_metadata["task_order"]),
        "tau": hyperparameters.get("tau"),
        "beta": hyperparameters.get("beta"),
        "eps": hyperparameters.get("eps"),
        "qp_tolerance": hyperparameters.get("qp_tolerance"),
    }
    if aggregator_state is None or (isinstance(aggregator_state, Mapping) and not aggregator_state):
        if cursor != 0:
            raise ValueError("PCD checkpoint is missing EMA state at a nonzero cursor.")
        return {**expected, "v": [0.0] * len(protocol_metadata["task_order"]), "t": 0}
    if not isinstance(aggregator_state, Mapping):
        raise TypeError("PCD checkpoint EMA state must be a mapping.")
    required = {*expected, "v", "t"}
    if set(aggregator_state) != required:
        raise ValueError("PCD checkpoint EMA state has missing or unexpected fields.")
    for name, value in expected.items():
        found = aggregator_state.get(name)
        if name == "version" and type(found) is not int:
            raise ValueError("PCD checkpoint EMA state version is invalid.")
        if name == "task_order":
            try:
                found = list(found)
            except TypeError as exc:
                raise ValueError("PCD checkpoint task order is invalid.") from exc
        elif name in ("tau", "beta", "eps", "qp_tolerance"):
            if (
                isinstance(found, bool)
                or not isinstance(found, (int, float))
                or not math.isfinite(float(found))
                or float(found) != float(value)
            ):
                raise ValueError(f"PCD checkpoint EMA state {name!r} differs from its protocol.")
            continue
        if found != value:
            raise ValueError(f"PCD checkpoint EMA state {name!r} differs from its protocol.")
    raw_v = aggregator_state.get("v")
    if not isinstance(raw_v, (list, tuple)) or len(raw_v) != len(protocol_metadata["task_order"]):
        raise ValueError("PCD checkpoint EMA values must align with protocol task order.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in raw_v
    ):
        raise ValueError("PCD checkpoint EMA values must be finite and nonnegative.")
    state_cursor = aggregator_state.get("t")
    if type(state_cursor) is not int or state_cursor != cursor:
        raise ValueError("PCD EMA step count does not match the saved sampling cursor.")
    return copy.deepcopy(dict(aggregator_state))


def save_moo_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    protocol_metadata: Mapping[str, Any],
    sampling_manifest: Mapping[str, Any],
    next_update: int,
    aggregator_state: Mapping[str, Any] | None,
) -> None:
    """Atomically save an h-free MOO checkpoint plus exact resume state."""
    validate_protocol_metadata(protocol_metadata)
    direct_armijo = protocol_metadata.get("step_rule") == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    if direct_armijo:
        if optimizer is not None or scheduler is not None:
            raise ValueError("Direct vector-Armijo checkpoints cannot contain optimizer or scheduler state.")
    elif optimizer is None:
        raise ValueError("Optimizer-based MOO checkpoints require optimizer state.")
    manifest_hash = validate_sampling_manifest(sampling_manifest)
    if (protocol_metadata.get("task_order") != list(TASK_NAMES)
            and sampling_manifest.get("task_order") != protocol_metadata.get("task_order")):
        raise ValueError("Checkpoint task order differs from sampling manifest.")
    cursor = {
        "sampling_manifest_sha256": manifest_hash,
        "next_update": int(next_update),
    }
    validate_cursor(sampling_manifest, cursor)
    if protocol_metadata.get("sampling_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint protocol and sampling manifest identities differ.")
    _validate_pcd_world_size(protocol_metadata, sampling_manifest)
    saved_aggregator_state = _validated_pcd_aggregator_state(
        protocol_metadata, aggregator_state, cursor=cursor["next_update"]
    )
    if scheduler is not None:
        scheduler_epoch = scheduler.state_dict().get("last_epoch")
        if scheduler_epoch != cursor["next_update"]:
            raise ValueError(
                "MOO checkpoint scheduler progress does not match its sampling cursor: "
                f"last_epoch={scheduler_epoch}, next_update={cursor['next_update']}."
            )
    rng_states = _gather_rng_states()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        payload = {
            "checkpoint_kind": "lap-moo-one-stage",
            "protocol_metadata": dict(protocol_metadata),
            "protocol_metadata_sha256": canonical_sha256(protocol_metadata),
            "operator_metadata": operator_checkpoint_metadata().to_dict(),
            "operator_protocol": OPERATOR_PROTOCOL,
            "model_kwargs": dict(getattr(model, "model_kwargs", {})),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "sampling_cursor": cursor,
            "aggregator_state": saved_aggregator_state,
            "rng_states_by_rank": rng_states,
        }
        _reject_stencil_checkpoint_fields(payload)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_moo_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    expected_protocol_metadata: Mapping[str, Any],
    sampling_manifest: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Fail closed if protocol/data identity differs; return the next cursor."""
    validate_protocol_metadata(expected_protocol_metadata)
    direct_armijo = expected_protocol_metadata.get("step_rule") == PCD_DIRECT_VECTOR_ARMIJO_STEP_RULE
    if direct_armijo:
        if optimizer is not None or scheduler is not None:
            raise ValueError("Direct vector-Armijo resume cannot load optimizer or scheduler state.")
    elif optimizer is None:
        raise ValueError("Optimizer-based MOO resume requires an optimizer.")
    manifest_hash = validate_sampling_manifest(sampling_manifest)
    if (expected_protocol_metadata.get("task_order") != list(TASK_NAMES)
            and sampling_manifest.get("task_order") != expected_protocol_metadata.get("task_order")):
        raise ValueError("Checkpoint task order differs from sampling manifest.")
    if expected_protocol_metadata.get("sampling_manifest_sha256") != manifest_hash:
        raise ValueError("Requested protocol and sampling manifest identities differ.")
    # Keep RNG byte tensors on CPU even when the model is restored to CUDA.
    # load_state_dict copies model weights to the model device, while optimizer
    # state loading moves its tensors to each parameter's device.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _reject_stencil_checkpoint_fields(payload)
    if payload.get("checkpoint_kind") != "lap-moo-one-stage":
        raise ValueError("This is not a clean one-stage MOO checkpoint.")
    metadata = payload.get("protocol_metadata")
    validate_protocol_metadata(metadata)
    if canonical_sha256(metadata) != payload.get("protocol_metadata_sha256"):
        raise ValueError("MOO checkpoint protocol metadata hash mismatch.")
    if canonical_sha256(metadata) != canonical_sha256(expected_protocol_metadata):
        raise ValueError("MOO checkpoint protocol differs from the requested run.")
    if metadata.get("sampling_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint protocol and sampling manifest identities differ.")
    cursor_value = validate_cursor(sampling_manifest, payload["sampling_cursor"])
    _validate_pcd_world_size(metadata, sampling_manifest)
    aggregator_state = _validated_pcd_aggregator_state(
        metadata, payload.get("aggregator_state"), cursor=cursor_value
    )
    scheduler_state = payload.get("scheduler_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if direct_armijo:
        if scheduler_state is not None or optimizer_state is not None:
            raise ValueError("Direct vector-Armijo checkpoint has unexpected optimizer or scheduler state.")
    elif scheduler is not None:
        if scheduler_state is None:
            raise ValueError("Checkpoint is missing the learning-rate scheduler state.")
        scheduler_epoch = scheduler_state.get("last_epoch")
        if scheduler_epoch != cursor_value:
            raise ValueError(
                "MOO checkpoint scheduler progress does not match its sampling cursor: "
                f"last_epoch={scheduler_epoch}, next_update={cursor_value}."
            )
    elif scheduler_state is not None:
        raise ValueError("Checkpoint has an unexpected learning-rate scheduler.")
    if not direct_armijo and not isinstance(optimizer_state, Mapping):
        raise ValueError("Optimizer-based MOO checkpoint is missing optimizer state.")
    operator_metadata = payload.get("operator_metadata")
    validate_operator_metadata(operator_metadata)
    if operator_metadata != operator_checkpoint_metadata().to_dict():
        raise ValueError("MOO checkpoint operator metadata differs from the AO protocol.")
    if payload.get("operator_protocol") != OPERATOR_PROTOCOL:
        raise ValueError("MOO checkpoint operator protocol mismatch.")
    if dict(payload.get("model_kwargs", {})) != dict(getattr(model, "model_kwargs", {})):
        raise ValueError("MOO checkpoint architecture kwargs differ from the model.")
    if tuple(payload["model_state_dict"]) != tuple(model.state_dict()):
        raise ValueError("MOO checkpoint model parameter keys differ.")
    states = payload.get("rng_states_by_rank")
    if not isinstance(states, (list, tuple)):
        raise TypeError("Checkpoint per-rank RNG states must be a list or tuple.")
    if metadata.get("method") == "pcd" and len(states) != metadata["world_size"]:
        raise ValueError(
            "PCD checkpoint per-rank RNG state count does not match its protocol world size."
        )
    for state in states:
        _validate_rng_state(state)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank >= len(states):
        raise ValueError("Checkpoint does not contain RNG state for this rank.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        restore_rng_state(states[rank])
    return cursor_value, aggregator_state


__all__ = [
    "AOFactorChunk",
    "ChemistryBatchObjective",
    "MRKSOperatorSystem",
    "UpdateResult",
    "apply_joint_gradient",
    "average_raw_task_gradients",
    "capture_rng_state",
    "compute_isolated_task_gradients",
    "load_moo_checkpoint",
    "make_cosine_scheduler",
    "make_mrks_objective_factories",
    "make_reaction_objective",
    "make_three_objective_factories",
    "materialize_task_zeros",
    "named_trainable_parameters",
    "restore_rng_state",
    "save_moo_checkpoint",
    "train_moo_update",
]
