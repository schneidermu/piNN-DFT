import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import lap_f017_audit as audit


def test_symmetric_flattening_preserves_frobenius_inner_product():
    generator=torch.Generator().manual_seed(1701)
    a=torch.randn(7,7,generator=generator,dtype=torch.float64)
    b=torch.randn(7,7,generator=generator,dtype=torch.float64)
    a=(a+a.T)/2
    b=(b+b.T)/2
    torch.testing.assert_close(audit.svec(a)@audit.svec(b),(a*b).sum(),atol=1e-14,rtol=1e-14)


def test_new_coordinate_layout_counts_tied_exchange_once():
    from lap_f016_descriptors import DiagnosticQModel
    from NN_models_lap import pcPBELMLOptimizerV2Lap
    model=DiagnosticQModel(pcPBELMLOptimizerV2Lap(use_g_x=True,use_g_c=True),'R2')
    parts=audit.new_parts(model)
    assert len(audit.new_flat(parts))==96
    assert len({p.data_ptr() for p in parts})==2
    assert sum(p.numel() for p in model.parameters())==9542


def test_reference_sign_and_vjp_loss_identity():
    j=torch.tensor([[1.,2.],[-3.,4.]],dtype=torch.float64)
    target=torch.tensor([.2,-.3],dtype=torch.float64)
    p=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    r=target-j@p
    gradient=torch.autograd.grad(r.square().sum(),p)[0]
    torch.testing.assert_close(-.5*gradient,j.T@target)


def test_matrix_free_jvp_matches_direct_small_jacobian():
    p=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
    direction=torch.tensor([.4,.3],dtype=torch.float64)
    def observable(x):
        return torch.stack([x[0]**2+3*x[1],x[0]*x[1]])
    _,value=torch.autograd.functional.jvp(observable,p,direction,strict=True)
    direct=torch.autograd.functional.jacobian(observable,p)@direction
    torch.testing.assert_close(value,direct,atol=1e-14,rtol=1e-14)


def test_unit_normalization_is_operator_objective_not_ao_weighting():
    a=torch.tensor([[1.,.2],[.2,-1.]],dtype=torch.float64)
    b=torch.tensor([[.1,.3],[.3,.4]],dtype=torch.float64)
    residual=audit.svec(a-b)/np.sqrt(2)
    torch.testing.assert_close(residual.square().sum(),(a-b).square().sum()/2)


def test_frozen_protocol_excludes_validation_and_exhaustive_variants():
    protocol=audit.verify()
    assert protocol['no_validation_or_test_inputs']
    assert protocol['one_variant_per_chemistry_identity']
    assert protocol['updates']==0
    assert len(protocol['operator_panel'])==15
    assert protocol['operator_panel'][0]=='H2'


def test_restoration_on_failed_diagnostic():
    from lap_f015_inputs import restored
    model=torch.nn.Linear(2,1)
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError),restored([model]):
        with torch.no_grad(): model.weight.add_(.01)
        torch.rand(3)
        raise RuntimeError('stop failed branch')
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(),rng)
