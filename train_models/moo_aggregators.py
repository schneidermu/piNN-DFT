"""Deterministic structured gradient aggregation for one-stage multi-task training.

All geometry is accumulated in float64 over the named parameter tensors.  No
concatenated parameter-sized vector is created, so unused parameters and large
models do not require a separate flattened copy.

PCD follows the published construction by Dara Varam and Mohamed I. AlHajri,
"Not All Objectives Are Born Equal: Priority-Constrained Descent for
Hierarchical Multi-Objective Optimization" (TMLR 2026), and the MIT-licensed
reference implementation at
https://github.com/DaraVaram/priority-constrained-descent, commit
e9afdbc9f4cb09343934eadb31dc72ea5ebac9b0. The reference software copyright
is Р’В© 2026 Dara Varam and Mohamed I. AlHajri and is distributed under the MIT
License, retained in ``train_models/PCD_LICENSE``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import torch

_TASK_PRIORITY = ("chem", "exc", "op")
_METHODS = ("fixed", "imtl_g", "cagrad", "nash_mtl", "pcd")
_PCD_QP_TOL = 1.0e-9


def _ordered_tasks(task_grads: Mapping[str, Mapping[str, torch.Tensor | None]]) -> tuple[str, ...]:
    if not isinstance(task_grads, Mapping) or not task_grads:
        raise ValueError("task_grads must be a nonempty task-to-parameter mapping.")
    if any(not isinstance(task, str) or not task for task in task_grads):
        raise ValueError("Task names must be nonempty strings.")
    priority = [task for task in _TASK_PRIORITY if task in task_grads]
    extras = sorted(task for task in task_grads if task not in _TASK_PRIORITY)
    tasks = tuple(priority + extras)
    if len(tasks) < 2:
        raise ValueError("Gradient aggregation requires at least two task gradients.")
    return tasks


def _collect_parameters(
    task_grads: Mapping[str, Mapping[str, torch.Tensor | None]], tasks: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[str, dict[str, torch.Tensor | None]], dict[str, torch.Tensor]]:
    names = tuple(sorted({name for task in tasks for name in task_grads[task]}))
    if not names:
        raise ValueError("No parameter gradients were supplied.")
    values: dict[str, dict[str, torch.Tensor | None]] = {task: {} for task in tasks}
    templates: dict[str, torch.Tensor] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter names must be nonempty strings.")
        for task in tasks:
            grad = task_grads[task].get(name)
            if grad is not None:
                if not isinstance(grad, torch.Tensor):
                    raise TypeError(f"Gradient {task}/{name} is not a tensor or None.")
                if not grad.is_floating_point() or grad.is_sparse:
                    raise TypeError(f"Gradient {task}/{name} must be a dense floating tensor.")
                if not bool(torch.isfinite(grad).all().item()):
                    raise FloatingPointError(f"Gradient {task}/{name} contains nonfinite values.")
                if name in templates:
                    template = templates[name]
                    if grad.shape != template.shape:
                        raise ValueError(f"Gradient shape mismatch for parameter {name!r}.")
                    if grad.device != template.device or grad.dtype != template.dtype:
                        raise ValueError(f"Gradient dtype/device mismatch for parameter {name!r}.")
                else:
                    templates[name] = grad
            values[task][name] = grad
    return names, values, templates


def _task_norms(
    tasks: tuple[str, ...], names: tuple[str, ...], values: Mapping[str, Mapping[str, torch.Tensor | None]]
) -> dict[str, float]:
    norms: dict[str, float] = {}
    for task in tasks:
        parts: dict[torch.device, list[torch.Tensor]] = {}
        for name in names:
            grad = values[task][name]
            if grad is not None:
                part = torch.linalg.vector_norm(grad.detach().to(dtype=torch.float64))
                parts.setdefault(grad.device, []).append(part)
        device_norms = []
        for device_parts in parts.values():
            device_norms.append(float(torch.linalg.vector_norm(torch.stack(device_parts)).item()))
        norm = math.hypot(*device_norms) if device_norms else 0.0
        if not math.isfinite(norm):
            raise FloatingPointError(f"Task gradient norm for {task!r} is not finite.")
        norms[task] = norm
    return norms


def _cosine(
    left: str,
    right: str,
    names: tuple[str, ...],
    values: Mapping[str, Mapping[str, torch.Tensor | None]],
    norms: Mapping[str, float],
) -> float:
    left_norm, right_norm = norms[left], norms[right]
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    parts: dict[torch.device, list[torch.Tensor]] = {}
    for name in names:
        a, b = values[left][name], values[right][name]
        if a is None or b is None:
            continue
        # Per-parameter float64 products avoid both raw-norm overflow and a
        # model-sized flattened temporary.  The normalized dot is in [-1, 1].
        a64 = a.detach().to(dtype=torch.float64).div(left_norm)
        b64 = b.detach().to(dtype=torch.float64).div(right_norm)
        parts.setdefault(a.device, []).append(torch.sum(a64 * b64))
    total = sum(float(torch.stack(group).sum().item()) for group in parts.values())
    if not math.isfinite(total):
        raise FloatingPointError(f"Task cosine for {left!r}/{right!r} is not finite.")
    return min(1.0, max(-1.0, total))


def _geometry(
    tasks: tuple[str, ...], names: tuple[str, ...], values: Mapping[str, Mapping[str, torch.Tensor | None]],
    norms: Mapping[str, float],
) -> tuple[list[list[float]], list[list[float]], list[float], float | None, int, float]:
    n = len(tasks)
    cosines = [[_cosine(a, b, names, values, norms) for b in tasks] for a in tasks]
    scale = max(norms.values(), default=0.0)
    if scale == 0.0:
        scaled_gram = np.zeros((n, n), dtype=np.float64)
    else:
        ratios = np.array([norms[task] / scale for task in tasks], dtype=np.float64)
        scaled_gram = np.asarray(cosines, dtype=np.float64) * np.outer(ratios, ratios)
        scaled_gram = (scaled_gram + scaled_gram.T) * 0.5
    eigvals = np.linalg.eigvalsh(scaled_gram)
    spectral_scale = max(float(np.max(np.abs(eigvals))), 1.0)
    tiny = np.finfo(np.float64).eps * n * spectral_scale * 64.0
    eigvals[np.logical_and(eigvals < 0, eigvals >= -tiny)] = 0.0
    positive = eigvals[eigvals > tiny]
    rank = int(positive.size)
    condition = None if rank < n else float(positive[-1] / positive[0])
    raw_gram: list[list[float]] = []
    for i, task_i in enumerate(tasks):
        row = []
        for j, task_j in enumerate(tasks):
            product = norms[task_i] * norms[task_j] * cosines[i][j]
            row.append(float(product) if math.isfinite(product) else None)
        raw_gram.append(row)
    return raw_gram, scaled_gram.tolist(), eigvals.tolist(), condition, rank, scale


def _combine(
    tasks: tuple[str, ...], names: tuple[str, ...],
    values: Mapping[str, Mapping[str, torch.Tensor | None]],
    coefficients: Mapping[str, float], *, divisors: Mapping[str, float] | None = None,
    final_scale: float = 1.0, output_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for name in names:
        active = [values[task][name] for task in tasks if values[task][name] is not None]
        if not active:
            result[name] = None
            continue
        template = active[0]
        assert template is not None
        total = torch.zeros(template.shape, dtype=torch.float64, device=template.device)
        for task in tasks:
            grad = values[task][name]
            if grad is None:
                continue
            divisor = 1.0 if divisors is None else float(divisors[task])
            coefficient = float(coefficients[task]) / divisor
            if coefficient != 0.0:
                total.add_(grad.detach().to(dtype=torch.float64), alpha=coefficient)
        if final_scale != 1.0:
            total.mul_(float(final_scale))
        if not bool(torch.isfinite(total).all().item()):
            raise FloatingPointError(f"Aggregated gradient for parameter {name!r} is nonfinite.")
        result[name] = total.to(dtype=template.dtype if output_dtype is None else output_dtype)
    return result


def _pcd_combine(
    tasks: tuple[str, ...], names: tuple[str, ...],
    values: Mapping[str, Mapping[str, torch.Tensor | None]],
    coefficients: Mapping[str, float], *, final_scale: float = 1.0,
) -> dict[str, torch.Tensor | None]:
    """Form a PCD direction in gradient dtype, without a model-sized flatten."""
    result: dict[str, torch.Tensor | None] = {}
    for name in names:
        active = [values[task][name] for task in tasks if values[task][name] is not None]
        if not active:
            result[name] = None
            continue
        template = active[0]
        assert template is not None
        total = torch.zeros_like(template)
        for task in tasks:
            grad = values[task][name]
            coefficient = float(coefficients[task])
            if grad is not None and coefficient != 0.0:
                total.add_(grad.detach(), alpha=coefficient)
        if final_scale != 1.0:
            total.mul_(float(final_scale))
        if not bool(torch.isfinite(total).all().item()):
            raise FloatingPointError(f"Aggregated PCD gradient for parameter {name!r} is nonfinite.")
        result[name] = total
    return result


def _fixed_coefficients(
    tasks: tuple[str, ...], hyperparameters: Mapping[str, Any]
) -> dict[str, float]:
    raw = hyperparameters.get("fixed_weights", hyperparameters.get("weights", [1.0] * len(tasks)))
    if isinstance(raw, Mapping):
        if set(raw) != set(tasks):
            raise ValueError("fixed_weights mapping must contain exactly the supplied task names.")
        weights = [float(raw[task]) for task in tasks]
    else:
        if not isinstance(raw, (list, tuple)) or len(raw) != len(tasks):
            raise ValueError("fixed_weights must contain one value per task in task order.")
        weights = [float(value) for value in raw]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Fixed scalarization weights must be positive finite values.")
    return {task: weights[index] / len(tasks) for index, task in enumerate(tasks)}


def _imtl_coefficients(
    tasks: tuple[str, ...], cosines: list[list[float]], norms: Mapping[str, float],
) -> tuple[dict[str, float], str, int, float]:
    n = len(tasks)
    # Constraints are dР’В·(u_i-u_ref)=0 with sum(alpha)=1.  Divide each
    # nonzero constraint row by its norm before the deterministic SVD solve.
    matrix = np.ones((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    rhs[0] = 1.0
    ref = n - 1
    for row in range(n - 1):
        constraint = np.asarray(
            [norms[tasks[column]] * (cosines[column][row] - cosines[column][ref])
             for column in range(n)],
            dtype=np.float64,
        )
        row_norm = float(np.linalg.norm(constraint))
        if row_norm > 64.0 * np.finfo(np.float64).eps:
            constraint /= row_norm
        else:
            constraint[:] = 0.0
        matrix[row + 1] = constraint
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        raise RuntimeError("IMTL-G constraint solver produced no singular values.")
    threshold = np.finfo(np.float64).eps * max(matrix.shape) * singular[0] * 64.0
    keep = singular > threshold
    rank = int(np.count_nonzero(keep))
    solution = vh[keep].T @ ((u[:, keep].T @ rhs) / singular[keep])
    residual = float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))
    if not np.all(np.isfinite(solution)) or residual > 1e-8:
        raise RuntimeError(f"IMTL-G constraints are inconsistent (residual={residual:.3e}).")
    retained = singular[keep]
    condition = float(retained[0] / retained[-1]) if retained.size else math.inf
    if rank and condition > 1e14:
        raise RuntimeError(f"IMTL-G constraints are ill-conditioned (condition={condition:.3e}).")
    status = "converged" if rank == n else "converged_rank_deficient_minimum_norm"
    return {task: float(solution[i]) for i, task in enumerate(tasks)}, status, rank, residual


def _cagrad_weights(
    tasks: tuple[str, ...], scaled_gram: list[list[float]], g0_norm: float,
    *, c: float, max_iter: int, ftol: float,
) -> tuple[np.ndarray, str, int, float]:
    from scipy.optimize import minimize

    n = len(tasks)
    gram = np.asarray(scaled_gram, dtype=np.float64)
    mean = np.full(n, 1.0 / n, dtype=np.float64)
    mean_direction = gram @ mean
    normalized_mean = mean_direction / g0_norm
    # Dividing the paper's objective by ||g0|| preserves its minimizer and
    # avoids a tiny objective when the mean gradient nearly cancels.
    def objective(w: np.ndarray) -> float:
        norm_sq = float(w @ gram @ w)
        norm = math.sqrt(max(norm_sq, 0.0))
        if norm == 0.0:
            if c == 0.0:
                return float(normalized_mean @ w)
            raise RuntimeError("CAGrad encountered an undefined zero weighted gradient.")
        return float(normalized_mean @ w + c * norm)

    def gradient(w: np.ndarray) -> np.ndarray:
        norm_sq = float(w @ gram @ w)
        norm = math.sqrt(max(norm_sq, 0.0))
        if norm == 0.0:
            if c == 0.0:
                return normalized_mean.copy()
            raise RuntimeError("CAGrad encountered an undefined zero weighted gradient.")
        return normalized_mean + c * (gram @ w) / norm

    result = minimize(
        objective,
        mean,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                      "jac": lambda w: np.ones_like(w)},),
        options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": False},
    )
    weights = np.asarray(result.x, dtype=np.float64)
    weights[np.abs(weights) < 1e-14] = 0.0
    weights[np.abs(weights - 1.0) < 1e-14] = 1.0
    primal = max(
        abs(float(weights.sum()) - 1.0),
        max(0.0, float(-np.min(weights))),
        max(0.0, float(np.max(weights) - 1.0)),
    )
    if not result.success or primal > 1e-8 or not np.all(np.isfinite(weights)):
        raise RuntimeError(
            f"CAGrad simplex solver failed: {result.message}; primal_residual={primal:.3e}."
        )
    # Check KKT stationarity on the active face and dual feasibility off it.
    grad = gradient(weights)
    active = weights > 1e-9
    multiplier = -float(np.mean(grad[active])) if np.any(active) else 0.0
    stationarity = float(np.max(np.abs(grad[active] + multiplier))) if np.any(active) else math.inf
    inactive = ~active
    dual = float(max(0.0, -np.min(grad[inactive] + multiplier))) if np.any(inactive) else 0.0
    residual = max(primal, stationarity, dual)
    if residual > 2e-5:
        raise RuntimeError(f"CAGrad KKT check failed (residual={residual:.3e}).")
    return weights, "converged", int(getattr(result, "nit", 0)), residual


def _r_minus_log1p(values: np.ndarray) -> np.ndarray:
    """Evaluate ``r - log1p(r)`` without cancellation near zero."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    small = np.abs(values) < 1.0e-4
    if np.any(small):
        r = values[small]
        power = r * r
        remainder = np.zeros_like(r)
        for order in range(2, 13):
            remainder += power / order if order % 2 == 0 else -power / order
            power *= r
        result[small] = remainder
    if np.any(~small):
        r = values[~small]
        result[~small] = r - np.log1p(r)
    return result


def _nash_potential_change(
    beta: np.ndarray,
    delta: np.ndarray,
    gradient: np.ndarray,
    gram: np.ndarray,
) -> float:
    """Compute ``phi(beta + delta) - phi(beta)`` without subtractive loss.

    For ``phi(x) = .5*x.T@gram@x - sum(log(x))``, expanding the quadratic
    and writing ``r = delta / beta`` gives

        gradient.T@delta + .5*delta.T@gram@delta + sum(r - log1p(r)).

    Evaluating this change directly preserves the tiny Armijo decrements that
    disappear when two full potential values near stationarity are subtracted.
    """
    ratio = delta / beta
    log_remainder = _r_minus_log1p(ratio)
    change = gradient @ delta + 0.5 * (delta @ gram @ delta) + np.sum(log_remainder)
    return float(change)


def _solve_nash_potential(
    gram: np.ndarray,
    *,
    scale: float,
    warm_alpha: np.ndarray | None,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int, float]:
    """Minimize the exact positive Nash potential with damped Newton steps."""
    n = gram.shape[0]
    beta = np.ones(n, dtype=np.float64) if warm_alpha is None else warm_alpha * scale
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0):
        raise ValueError("Nash-MTL preconditioned warm start is not positive and finite.")

    converged = False
    for iteration in range(max_iter + 1):
        kx = gram @ beta
        residual = float(np.max(np.abs(beta * kx - 1.0)))
        if residual <= tol:
            converged = True
            solver_iterations = iteration
            break
        if iteration == max_iter:
            solver_iterations = iteration
            break
        gradient = kx - 1.0 / beta
        hessian = gram + np.diag(1.0 / (beta * beta))
        try:
            step_direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Nash-MTL Newton Hessian is singular; no ridge was applied.") from exc
        directional_derivative = float(gradient @ step_direction)
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise RuntimeError("Nash-MTL Newton step is not a finite descent direction.")
        step = 1.0
        negative = step_direction < 0.0
        if np.any(negative):
            step = min(step, 0.99 * float(np.min(-beta[negative] / step_direction[negative])))
        accepted = False
        for _ in range(80):
            delta = step * step_direction
            trial = beta + delta
            if np.all(np.isfinite(trial)) and np.all(trial > 0.0):
                potential_change = _nash_potential_change(beta, delta, gradient, gram)
                if math.isfinite(potential_change) and potential_change <= 1e-4 * step * directional_derivative:
                    beta = trial
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            raise RuntimeError("Nash-MTL damped Newton line search failed Armijo decrease.")

    solver_residual = float(np.max(np.abs(beta * (gram @ beta) - 1.0)))
    if not converged or solver_residual > tol:
        raise RuntimeError(
            f"Nash-MTL potential solver did not converge in {max_iter} iterations "
            f"(dimensionless residual={solver_residual:.3e})."
        )
    return beta, int(solver_iterations), solver_residual


def _pcd_state_values(
    state: Mapping[str, Any] | None,
    tasks: tuple[str, ...],
    *,
    tau: float,
    beta: float,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Validate and load PCD's per-task EMA state."""
    if state is None:
        return np.zeros(len(tasks), dtype=np.float64), 0
    if not isinstance(state, Mapping):
        raise TypeError("PCD state must be a mapping.")
    if state.get("version") != 1 or state.get("method") != "pcd":
        raise ValueError("PCD state has an unsupported version or method.")
    try:
        task_order = tuple(state.get("task_order", ()))
    except TypeError as exc:
        raise ValueError("PCD state task order is incompatible with this update.") from exc
    if task_order != tasks:
        raise ValueError("PCD state task order is incompatible with this update.")
    for name, expected in (
        ("tau", tau), ("beta", beta), ("eps", eps), ("qp_tolerance", _PCD_QP_TOL)
    ):
        try:
            value = float(state[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"PCD state is missing a valid {name!r} value.") from exc
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"PCD state {name!r} does not match the active configuration.")
    try:
        raw_v = np.asarray(state["v"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PCD state must contain finite, task-aligned EMA values.") from exc
    if raw_v.shape != (len(tasks),) or not np.all(np.isfinite(raw_v)) or np.any(raw_v < 0.0):
        raise ValueError("PCD state must contain finite, nonnegative, task-aligned EMA values.")
    count = state.get("t")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("PCD state step count must be a nonnegative integer.")
    return raw_v.copy(), count


def _pcd_independent(active_gram: np.ndarray, tol: float) -> bool:
    """Match the reference solver's scale-free independence test."""
    if active_gram.shape[0] == 1:
        return True
    diagonal = np.diag(active_gram)
    if np.any(diagonal <= 0.0):
        return False
    norms = np.sqrt(diagonal)
    normalized = active_gram / np.outer(norms, norms)
    return bool(np.linalg.eigvalsh(normalized)[0] > tol)


def _solve_pcd_qp(
    normalized_gram: np.ndarray,
    tau: float,
    *,
    tol: float = _PCD_QP_TOL,
) -> tuple[np.ndarray, tuple[int, ...], bool, int, float]:
    """Solve the ordered K-task PCD QP by the source's KKT enumeration.

    Returned weights define ``d_tilde = sum_i weights[i] * g_tilde_i``.
    Active task indices are 1-based into the task order (primary is index 0).
    """
    gram = np.asarray(normalized_gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] < 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"PCD requires a square KxK Gram matrix, K >= 2, got {gram.shape}.")
    if not np.all(np.isfinite(gram)):
        raise FloatingPointError("PCD normalized Gram matrix contains nonfinite values.")
    weights = np.zeros(len(gram), dtype=np.float64)
    weights[0] = 1.0
    secondary_gram = gram[1:, 1:]
    rhs = tau * np.diag(secondary_gram) - gram[1:, 0]
    if np.all(rhs <= 0.0):
        return weights, (), True, 0, 0.0

    atol = tol * max(float(np.max(np.diag(gram))), np.finfo(np.float64).tiny)
    candidates = [index for index in range(len(gram) - 1) if secondary_gram[index, index] > 0.0]
    checked = 0
    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            checked += 1
            active = list(subset)
            active_gram = secondary_gram[np.ix_(active, active)]
            if not _pcd_independent(active_gram, tol):
                continue
            try:
                multipliers = np.linalg.solve(active_gram, rhs[active])
            except np.linalg.LinAlgError:
                continue
            if np.any(multipliers < -tol):
                continue
            multipliers = np.maximum(multipliers, 0.0)
            slack = secondary_gram[:, active] @ multipliers - rhs
            if np.all(slack >= -atol):
                weights[1 + np.asarray(active)] = multipliers
                residual = max(
                    max(0.0, float(-np.min(slack))),
                    float(np.max(np.abs(multipliers * slack[active]))),
                )
                return weights, tuple(1 + index for index in active), True, checked, residual

    # Canonical deployment fallback: discard the secondary constraints and
    # follow the primary gradient for this update.
    fallback_slack = gram[1:, 0] - tau * np.diag(secondary_gram)
    residual = max(0.0, float(-np.min(fallback_slack)))
    return weights, (), False, checked, residual


def aggregate_task_gradients(
    task_grads: Mapping[str, Mapping[str, torch.Tensor | None]],
    *,
    method: str,
    hyperparameters: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    task_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any], dict[str, Any]]:
    """Aggregate per-task parameter gradients and return JSON-safe diagnostics/state.

    Task order is `chem`, `exc`, `op`, then any extra task names sorted
    lexicographically.  Missing parameter gradients count as zero for geometry;
    an output parameter remains `None` only when every task supplied `None`.
    """
    if method not in _METHODS:
        raise ValueError(f"Unsupported gradient aggregation method {method!r}.")
    hparams = {} if hyperparameters is None else dict(hyperparameters)
    tasks = _ordered_tasks(task_grads)
    if task_order is not None:
        tasks = tuple(task_order)
        if (len(tasks) < 2 or len(set(tasks)) != len(tasks)
                or any(not isinstance(task, str) or not task for task in tasks)
                or set(tasks) != set(task_grads)):
            raise ValueError("Explicit task order must contain every task exactly once.")
    names, values, _templates = _collect_parameters(task_grads, tasks)
    norms = _task_norms(tasks, names, values)
    raw_gram, scaled_gram, eigenvalues, condition, rank, scale = _geometry(
        tasks, names, values, norms
    )
    cosine_matrix = [
        [_cosine(a, b, names, values, norms) for b in tasks] for a in tasks
    ]
    solver_status = "closed_form"
    solver_iterations = 0
    solver_residual = 0.0
    new_state: dict[str, Any] = {}
    pcd_diagnostics: dict[str, Any] = {}

    if method == "fixed":
        coefficients = _fixed_coefficients(tasks, hparams)
        joint = _combine(tasks, names, values, coefficients)
    elif method == "imtl_g":
        zero_tasks = [task for task in tasks if norms[task] == 0.0]
        if zero_tasks:
            raise ValueError(f"IMTL-G requires nonzero task gradients; zero: {zero_tasks}.")
        coefficients, solver_status, solver_iterations, solver_residual = _imtl_coefficients(
            tasks, cosine_matrix, norms
        )
        joint = _combine(tasks, names, values, coefficients)
        joint_norm = _task_norms(("joint",), names, {"joint": joint})["joint"]
        if joint_norm <= 1e-14:
            # The exact raw-gradient equations can yield d=0 for opposing
            # tasks.  Preserve that mathematical result and make the
            # degeneracy explicit instead of inventing a fallback direction.
            solver_status = "degenerate_zero_direction"
    elif method == "cagrad":
        c = float(hparams.get("c", 0.4))
        if not math.isfinite(c) or c < 0.0:
            raise ValueError("CAGrad c must be finite and nonnegative.")
        if hparams.get("rescale", "paper_unscaled") != "paper_unscaled":
            raise ValueError("This implementation supports only CAGrad rescale='paper_unscaled'.")
        max_iter = int(hparams.get("max_iter", 500))
        ftol = float(hparams.get("ftol", 1e-12))
        if max_iter <= 0 or not math.isfinite(ftol) or ftol <= 0:
            raise ValueError("CAGrad max_iter and ftol must be positive.")
        mean_coeff = {task: 1.0 / len(tasks) for task in tasks}
        # Compute ||g0|| after scaling all task gradients by their largest norm.
        mean_vector = np.full(len(tasks), 1.0 / len(tasks), dtype=np.float64)
        if scale == 0.0:
            h0 = {name: None for name in names}
            mean_norm = 0.0
        else:
            h0 = _combine(tasks, names, values, mean_coeff,
                          divisors={task: scale for task in tasks}, output_dtype=torch.float64)
            mean_norm = _task_norms(("h0",), names, {"h0": h0})["h0"]
        if mean_norm == 0.0:
            coefficients = mean_coeff
            joint = _combine(tasks, names, values, coefficients)
            solver_status = "zero_mean_stationary"
        else:
            weights, solver_status, solver_iterations, solver_residual = _cagrad_weights(
                tasks, scaled_gram, mean_norm, c=c, max_iter=max_iter, ftol=ftol
            )
            hgw = _combine(tasks, names, values,
                           {task: float(weights[i]) for i, task in enumerate(tasks)},
                           divisors={task: scale for task in tasks}, output_dtype=torch.float64)
            gw_norm = _task_norms(("hgw",), names, {"hgw": hgw})["hgw"]
            if c > 0.0 and gw_norm == 0.0:
                raise RuntimeError("CAGrad optimum has zero weighted gradient; normalized direction is undefined.")
            coeff_values = mean_vector + (
                (c * mean_norm / gw_norm) * weights if c > 0.0 else np.zeros(len(tasks))
            )
            coefficients = {task: float(coeff_values[i]) for i, task in enumerate(tasks)}
            # Form the paper's direction in scaled coordinates to preserve
            # accuracy when the optimized weighted sum is small.
            joint = {}
            for name in names:
                present = [values[task][name] for task in tasks if values[task][name] is not None]
                if not present:
                    joint[name] = None
                    continue
                template = present[0]
                assert template is not None
                total = torch.zeros(template.shape, dtype=torch.float64, device=template.device)
                if h0[name] is not None:
                    total.add_(h0[name].to(dtype=torch.float64))
                if c > 0.0 and hgw[name] is not None:
                    total.add_(hgw[name].to(dtype=torch.float64), alpha=c * mean_norm / gw_norm)
                total.mul_(scale)
                if not bool(torch.isfinite(total).all().item()):
                    raise FloatingPointError(f"Aggregated gradient for parameter {name!r} is nonfinite.")
                joint[name] = total.to(dtype=template.dtype)
    elif method == "pcd":
        if task_order is None and tasks != _TASK_PRIORITY:
            raise ValueError(
                "Nonlegacy PCD requires an explicit task_order; canonical order is chem, exc, op."
            )
        try:
            tau = float(hparams.get("tau", 0.02))
            beta = float(hparams.get("beta", 0.999))
            eps = float(hparams.get("eps", 1.0e-8))
            configured_tol = float(hparams.get("qp_tolerance", _PCD_QP_TOL))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCD tau, beta, eps, and qp_tolerance must be numeric scalars.") from exc
        if (not math.isfinite(tau) or not 0.0 <= tau <= 1.0
                or not math.isfinite(beta) or not 0.0 <= beta < 1.0
                or not math.isfinite(eps) or eps <= 0.0):
            raise ValueError("PCD requires tau in [0, 1], beta in [0, 1), and positive finite eps.")
        if not math.isfinite(configured_tol) or configured_tol != _PCD_QP_TOL:
            raise ValueError(f"PCD qp_tolerance is pinned to {_PCD_QP_TOL:g}.")

        ema_v, ema_t = _pcd_state_values(
            state, tasks, tau=tau, beta=beta, eps=eps
        )
        squared_norms = np.asarray([norms[task] ** 2 for task in tasks], dtype=np.float64)
        if not np.all(np.isfinite(squared_norms)):
            raise FloatingPointError("PCD squared task gradient norms are nonfinite.")
        next_t = ema_t + 1
        ema_v = beta * ema_v + (1.0 - beta) * squared_norms
        bias_correction = 1.0 - beta ** next_t
        if not np.all(np.isfinite(ema_v)) or bias_correction <= 0.0:
            raise FloatingPointError("PCD EMA update or bias correction is invalid.")
        ema_vhat = ema_v / bias_correction
        scales = 1.0 / np.sqrt(ema_vhat + eps)
        scaled_norms = np.asarray(
            [scales[index] * norms[task] for index, task in enumerate(tasks)],
            dtype=np.float64,
        )
        if not (np.all(np.isfinite(ema_vhat)) and np.all(np.isfinite(scales))
                and np.all(np.isfinite(scaled_norms))):
            raise FloatingPointError("PCD normalization produced nonfinite values.")

        cosine_array = np.asarray(cosine_matrix, dtype=np.float64)
        normalized_gram = cosine_array * np.outer(scaled_norms, scaled_norms)
        normalized_gram = (normalized_gram + normalized_gram.T) * 0.5
        np.fill_diagonal(normalized_gram, scaled_norms * scaled_norms)
        if not np.all(np.isfinite(normalized_gram)):
            raise FloatingPointError("PCD normalized Gram matrix contains nonfinite values.")

        primary_norm = norms[tasks[0]]
        if primary_norm == 0.0:
            # Match the reference deployment branch: update EMA state, then
            # halt with an exact zero direction even when secondaries remain.
            weights = np.zeros(len(tasks), dtype=np.float64)
            weights[0] = 1.0
            active_indices: tuple[int, ...] = ()
            feasible = True
            solver_iterations = 0
            solver_residual = 0.0
            solver_status = "primary_zero"
        else:
            weights, active_indices, feasible, solver_iterations, solver_residual = _solve_pcd_qp(
                normalized_gram, tau
            )
            solver_status = (
                "infeasible_primary_fallback" if not feasible
                else "active_set" if active_indices
                else "inactive"
            )

        normalized_coeff_values = weights * scales
        if not np.all(np.isfinite(normalized_coeff_values)):
            raise FloatingPointError("PCD normalized direction coefficients are nonfinite.")
        normalized_coefficients = {
            task: float(normalized_coeff_values[index])
            for index, task in enumerate(tasks)
        }
        normalized_direction = _pcd_combine(
            tasks, names, values, normalized_coefficients
        )
        pre_rescale_norm = _task_norms(
            ("normalized_direction",), names,
            {"normalized_direction": normalized_direction},
        )["normalized_direction"]
        if not math.isfinite(pre_rescale_norm):
            raise FloatingPointError("PCD pre-rescale direction norm is nonfinite.")
        if primary_norm == 0.0 or pre_rescale_norm == 0.0:
            raw_coeff_values = np.zeros(len(tasks), dtype=np.float64)
            joint = _pcd_combine(
                tasks, names, values,
                {task: 0.0 for task in tasks},
            )
            if primary_norm != 0.0:
                solver_status = "zero_direction"
        else:
            rescale = primary_norm / pre_rescale_norm
            raw_coeff_values = normalized_coeff_values * rescale
            if not np.all(np.isfinite(raw_coeff_values)):
                raise FloatingPointError("PCD raw-equivalent coefficients are nonfinite.")
            joint = _pcd_combine(
                tasks, names, values, normalized_coefficients,
                final_scale=rescale,
            )
        coefficients = {
            task: float(raw_coeff_values[index]) for index, task in enumerate(tasks)
        }

        rhs = tau * np.diag(normalized_gram)[1:]
        lhs = normalized_gram[1:, :] @ weights
        slack = lhs - rhs
        feasibility_atol = _PCD_QP_TOL * max(
            float(np.max(np.diag(normalized_gram))), np.finfo(np.float64).tiny
        )
        pcd_diagnostics = {
            "tau": tau,
            "beta": beta,
            "eps": eps,
            "qp_tolerance": _PCD_QP_TOL,
            "ema_squared_norms": {
                task: float(ema_v[index]) for index, task in enumerate(tasks)
            },
            "bias_corrected_ema_squared_norms": {
                task: float(ema_vhat[index]) for index, task in enumerate(tasks)
            },
            "ema_step": next_t,
            "normalization_scales": {
                task: float(scales[index]) for index, task in enumerate(tasks)
            },
            "normalized_task_gradient_norms": {
                task: float(scaled_norms[index]) for index, task in enumerate(tasks)
            },
            "normalized_gram": normalized_gram.tolist(),
            "mu": {task: float(weights[index]) for index, task in enumerate(tasks[1:], 1)},
            "active": [tasks[index] for index in active_indices],
            "active_indices": list(active_indices),
            "feasible": bool(feasible),
            "pre_final_rescale_norm": float(pre_rescale_norm),
            "raw_equivalent_coefficients": dict(coefficients),
            "constraints": {
                task: {
                    "lhs": float(lhs[index]),
                    "rhs": float(rhs[index]),
                    "slack": float(slack[index]),
                    "satisfied": bool(slack[index] >= -feasibility_atol),
                }
                for index, task in enumerate(tasks[1:])
            },
        }
        new_state = {
            "version": 1,
            "method": "pcd",
            "task_order": list(tasks),
            "tau": tau,
            "beta": beta,
            "eps": eps,
            "qp_tolerance": _PCD_QP_TOL,
            "v": [float(value) for value in ema_v],
            "t": next_t,
        }
    else:
        if any(norms[task] == 0.0 for task in tasks):
            zero_tasks = [task for task in tasks if norms[task] == 0.0]
            raise ValueError(f"Nash-MTL potential is unbounded for zero task gradients: {zero_tasks}.")
        if hparams.get("solver", "newton_potential") != "newton_potential":
            raise ValueError("Nash-MTL supports only the canonical newton_potential solver.")
        if int(hparams.get("update_every", 1)) != 1:
            raise ValueError("Nash-MTL coefficients must be updated every step (update_every=1).")
        max_iter = int(hparams.get("max_iter", 100))
        tol = float(hparams.get("tol", 1e-10))
        if max_iter <= 0 or not math.isfinite(tol) or tol <= 0:
            raise ValueError("Nash-MTL max_iter and tol must be positive.")
        if scale == 0.0:
            raise ValueError("Nash-MTL has no finite potential solution at zero gradient scale.")
        gram = np.asarray(scaled_gram, dtype=np.float64)
        gram = (gram + gram.T) * 0.5
        warm_alpha = None
        if state is not None and "alpha" in state:
            if state.get("method") != "nash_mtl" or tuple(state.get("task_order", ())) != tasks:
                raise ValueError("Nash-MTL warm-start state is incompatible with this task order.")
            candidate = np.asarray(state["alpha"], dtype=np.float64)
            if candidate.shape != (len(tasks),) or not np.all(np.isfinite(candidate)) or np.any(candidate <= 0):
                raise ValueError("Nash-MTL warm-start alpha must be positive, finite, and task-aligned.")
            warm_alpha = candidate
        # Let beta = scale * alpha.  This exact variable change only
        # preconditions the Gram matrix; the stationary equations are unchanged.
        beta, solver_iterations, solver_residual = _solve_nash_potential(
            gram,
            scale=scale,
            warm_alpha=warm_alpha,
            max_iter=max_iter,
            tol=tol,
        )
        alpha = beta / scale
        if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise RuntimeError("Nash-MTL produced nonpositive or nonfinite task coefficients.")
        coefficients = {task: float(alpha[i]) for i, task in enumerate(tasks)}
        joint = _combine(tasks, names, values,
                         {task: float(beta[i]) for i, task in enumerate(tasks)},
                         divisors={task: scale for task in tasks})
        solver_status = "converged"
        solver_iterations = int(solver_iterations)
        new_state = {
            "version": 1,
            "method": "nash_mtl",
            "task_order": list(tasks),
            "alpha": [float(value) for value in alpha],
        }

    directional_dots: dict[str, float | None] = {}
    for task in tasks:
        if norms[task] == 0.0:
            directional_dots[task] = 0.0
        else:
            jnorm = _task_norms(("joint",), names, {"joint": joint})["joint"]
            if jnorm == 0.0:
                directional_dots[task] = 0.0
            else:
                dot_cosine = _cosine(task, "joint", names,
                                     {**values, "joint": joint},
                                     {**norms, "joint": jnorm})
                dot = norms[task] * jnorm * dot_cosine
                directional_dots[task] = float(dot) if math.isfinite(dot) else None

    diagnostics: dict[str, Any] = {
        "method": method,
        "task_order": list(tasks),
        "coefficients": {task: float(coefficients[task]) for task in tasks},
        "task_gradient_norms": {task: float(norms[task]) for task in tasks},
        "task_cosines": {task: {other: float(cosine_matrix[i][j]) for j, other in enumerate(tasks)}
                         for i, task in enumerate(tasks)},
        "gram": raw_gram,
        "gram_scaled_by_max_norm_squared": scaled_gram,
        "gram_eigenvalues": [float(value) for value in eigenvalues],
        "gram_condition_number": condition,
        "gram_rank": rank,
        "directional_task_dots": directional_dots,
        "solver_status": solver_status,
        "solver_iterations": int(solver_iterations),
        "solver_residual": float(solver_residual),
    }
    if pcd_diagnostics:
        pcd_diagnostics["post_final_rescale_norm"] = float(
            _task_norms(("joint",), names, {"joint": joint})["joint"]
        )
        diagnostics.update(pcd_diagnostics)
    return joint, diagnostics, new_state
