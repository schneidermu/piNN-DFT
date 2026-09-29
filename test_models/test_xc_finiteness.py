import numpy as np
import pytest
import torch

_DEFAULT_DTYPE = torch.get_default_dtype()
from test_models.DFT.numerics import ensure_finite_xc
torch.set_default_dtype(_DEFAULT_DTYPE)


@pytest.mark.parametrize("quantity", ["exc", "vrho", "vsigma", "vlapl", "vtau"])
def test_non_finite_xc_quantity_raises_with_its_name(quantity):
    with pytest.raises(FloatingPointError, match=quantity):
        ensure_finite_xc(quantity, np.array([0.0, np.nan]))


def test_finite_xc_quantity_is_accepted():
    ensure_finite_xc("exc", np.array([0.0, -1.2]))
