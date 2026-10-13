import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import lap_f018_audit as audit


def test_original_joint_gate_is_not_operator_only():
    truth=torch.tensor([.1,.2,.3],dtype=torch.float64)
    fd=truth+torch.tensor([.01,0.,0.],dtype=torch.float64)
    assert audit.difference(fd,truth)['absolute_gap']>0
    assert audit.difference(fd[1:],truth[1:])['absolute_gap']==0


def test_native_zero_weight_plus_minus_storage_is_symmetric():
    v=torch.linspace(-1,1,96,dtype=torch.float64)
    v=v/v.norm()
    for h in audit.H:
        plus=(h*v).float().double()
        minus=(-h*v).float().double()
        assert torch.equal(plus,-minus)
        assert torch.count_nonzero((plus+minus)/2)==0
        assert float(((plus-minus)/(2*h)-v).norm())<1e-7


def test_same_frozen_direction_and_h_ladder_are_pinned():
    protocol=audit.verify()
    assert protocol['FD_steps']==[.01,.003,.001,.0003,.0001,.00003,.00001]
    assert protocol['parameter_directions']==1
    assert protocol['n_ao']==60 and protocol['n_grid']==42508
    assert protocol['F017_gate']=={'absolute':1e-6,'relative':.005}
    saved=np.load(audit.OUT/'frozen_F017_direction.npy')
    np.testing.assert_allclose(np.linalg.norm(saved),1.,atol=1e-15)


def test_explicit_energy_first_mixed_derivative():
    theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    density=torch.tensor([.4,.7],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([.3,-.2],dtype=torch.float64)
    energy=((theta+1)*density.square()).sum()
    theta_grad=torch.autograd.grad(energy,theta,create_graph=True)[0]
    mixed=torch.autograd.grad(theta_grad@v,density)[0]
    torch.testing.assert_close(mixed,2*density*v,atol=1e-14,rtol=1e-14)


def test_measured_triangle_error_bound():
    native=torch.tensor([.01,-.03,.02],dtype=torch.float64)
    higher=torch.tensor([.009,-.031,.021],dtype=torch.float64)
    jhigh=torch.tensor([.011,-.033,.025],dtype=torch.float64)
    jnative=torch.tensor([.0115,-.0329,.0248],dtype=torch.float64)
    upper=(native-higher).norm()+(higher-jhigh).norm()+(jhigh-jnative).norm()
    assert (native-jnative).norm()<=upper


def test_no_scientific_optimizer_or_forbidden_panel_calls():
    for path in (audit.ROOT/'tools').glob('lap_f018_*.py'):
        text=path.read_text()
        for token in ('torch.optim.','optimizer.step(','.scf(','f017.panel(','evaluate_clean28('):
            assert token not in text
