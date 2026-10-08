"""Fixed-gradient-normalized scalarization; native AdamW, no gradient surgery."""
from collections.abc import Mapping

import numpy as np
import torch

from .lap_moo_training import named_trainable_parameters

TASKS = ('relchem', 'ae17', 'exc', 'op')


def fixed_coefficients(norms, *, epsilon=1e-12, normalization=4.0):
    values = np.asarray(norms, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError('Calibration needs batch-by-four task gradient norms')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Calibration norms must be finite and nonnegative')
    if epsilon <= 0 or normalization <= 0:
        raise ValueError('epsilon and global normalization must be positive')
    scales = np.median(values, axis=0)
    coefficients = 1 / np.maximum(scales, epsilon) / normalization
    return dict(zip(TASKS, scales.tolist(), strict=True)), dict(zip(TASKS, coefficients.tolist(), strict=True))


def weighted_gradient(raw: Mapping, coefficients: Mapping):
    if tuple(raw) != TASKS or tuple(coefficients) != TASKS:
        raise ValueError('Explicit scientific task order required')
    names = tuple(raw[TASKS[0]])
    if any(tuple(raw[t]) != names for t in TASKS):
        raise ValueError('Task parameter ordering differs')
    if any(not np.isfinite(coefficients[t]) or coefficients[t] <= 0 for t in TASKS):
        raise ValueError('Fixed coefficients must be positive and finite')
    # Qualified task derivatives remain F64 until the native optimizer boundary.
    return {n: sum(raw[t][n].double() * coefficients[t] for t in TASKS) for n in names}


def adamw_step(model, optimizer, raw, coefficients):
    joint = weighted_gradient(raw, coefficients)
    parameters = named_trainable_parameters(model)
    if tuple(joint) != tuple(parameters):
        raise ValueError('Optimizer parameter coordinates differ')
    if not all(torch.isfinite(g).all() for g in joint.values()):
        raise FloatingPointError('Nonfinite scalarized gradient')
    optimizer.zero_grad(set_to_none=True)
    for name, p in parameters.items():
        if joint[name].shape != p.shape or joint[name].device != p.device:
            raise ValueError('Gradient shape/device mismatch')
        # F32 native AdamW storage, exactly one cast after F64 combination.
        p.grad = joint[name].to(p.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not all(torch.isfinite(p).all() for p in parameters.values()):
        raise FloatingPointError('Nonfinite AdamW state; stop, no recovery rule')
    return joint


def eligible(objectives, baseline):
    if set(objectives) != set(TASKS) or set(baseline) != set(TASKS):
        raise ValueError('Selection requires all four exact objectives')
    if any(not np.isfinite(baseline[t]) or baseline[t] <= 0 for t in TASKS):
        raise ValueError('Positive finite baseline objectives required')
    return all(np.isfinite(objectives[t]) and objectives[t] / baseline[t] < 1 for t in TASKS)


def select_checkpoint(records, baseline):
    candidates = [r for r in records if eligible(r['objectives'], baseline)]
    return min(candidates, key=lambda r: (r['validation']['clean28'], r['cursor'])) if candidates else None
