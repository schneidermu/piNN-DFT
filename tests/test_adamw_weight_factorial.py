import copy

import numpy as np

from tools import adamw_weight_factorial as f


def test_direct_factorial_coefficients():
    base = {'relchem': .017, 'ae17': .00005, 'exc': .000015, 'op': .336}
    before = copy.deepcopy(base)
    for arm, (chem, op) in f.ARMS.items():
        result = f.coefficients(base, arm)
        assert result == {**base, 'relchem': chem*base['relchem'], 'op': op*base['op']}
    assert base == before
    assert np.isclose(sum(f.coefficients(base, 'D').values()), 2*(base['relchem']+base['op'])+base['ae17']+base['exc'])
