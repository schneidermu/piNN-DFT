import numpy
import torch


def ensure_finite_xc(name, values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = numpy.asarray(values)
    invalid = numpy.argwhere(~numpy.isfinite(values))
    if invalid.size:
        location = tuple(int(index) for index in invalid[0])
        raise FloatingPointError(f"Non-finite XC {name} at index {location}")
