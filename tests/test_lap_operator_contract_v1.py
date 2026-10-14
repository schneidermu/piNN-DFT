"""Independent audit regression proof; fixtures only, no trained weights."""

import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.spec_from_file_location('contract_followup', ROOT/'tools/lap_operator_contract_v1_followup.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
AUDIT = MODULE.audit


def test_independent_symmetric_energy_gradient_and_negative_controls():
    rows = MODULE.corrected_fixtures()
    assert all(row['status']=='PASS' for row in rows)
    assert {row['check'] for row in rows if 'meaning' in row} == {
        'extra_spin_factor', 'missing_laplacian_AO_terms', 'wrong_sigma_mapping',
    }


def test_independent_metric_derivative_and_basis_span_invariance():
    assert all(row['status']=='PASS' for row in AUDIT.metric_checks())


def test_full_r0_operator_loss_gradient_from_energy_first_graph():
    rng = torch.get_rng_state()
    torch.manual_seed(44021)
    model = AUDIT.pcPBELMLOptimizerV2Lap(use_g_c=True).double()
    torch.set_rng_state(rng)
    assert sum(p.numel() for p in model.parameters()) == 9446
    gen = torch.Generator().manual_seed(44022)
    rand = lambda *size: torch.randn(*size, generator=gen, dtype=torch.float64)
    phi, grad, lap = rand(7,3), rand(7,3,3), rand(7,3)
    weights = torch.rand(7, generator=gen, dtype=torch.float64)+.1
    dm = (.3*torch.eye(3,dtype=torch.float64)).requires_grad_()
    energy = AUDIT.LapEnergy(model)
    features = AUDIT.op.rks_density_features_from_ao(phi,grad,lap,dm.detach())
    assembled = AUDIT.op.assemble_rks_operator(energy,features,weights,phi,grad,lap,chunk_size=3)
    n, _, lr, sigma = AUDIT.independent_features(phi,grad,lap,(dm+dm.T)/2)
    integral = (energy(torch.stack([n/2,n/2],-1),sigma[:,None].expand(-1,3)/4,
                       torch.stack([lr/2,lr/2],-1))*weights).sum()
    independent = torch.autograd.grad(integral,dm,create_graph=True)[0]
    target=rand(3,3);target=(target+target.T)/2
    overlap=torch.eye(3,dtype=torch.float64)
    parameters=list(model.parameters())
    actual=torch.autograd.grad(AUDIT.op.operator_loss(assembled,target,overlap),parameters,allow_unused=True)
    expected=torch.autograd.grad((independent-target).square().sum()/3,parameters,allow_unused=True)
    for p,a,b in zip(parameters,actual,expected):
        a=torch.zeros_like(p) if a is None else a
        b=torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)


def test_laplacian_three_product_terms_and_sigma_spin_mapping():
    rng = np.random.default_rng(42024)
    phi = torch.tensor(rng.normal(size=(8,3)))
    grad = torch.tensor(rng.normal(size=(8,3,3)))
    lap = torch.tensor(rng.normal(size=(8,3)))
    dm = torch.eye(3,dtype=torch.float64)
    f = AUDIT.op.rks_density_features_from_ao(phi,grad,lap,dm)
    n, dr, lr, total_sigma = AUDIT.independent_features(phi,grad,lap,dm)
    torch.testing.assert_close(f[:,:2].sum(-1),n)
    torch.testing.assert_close(f[:,8:].sum(-1),lr)
    mapped=AUDIT.op.sigma_from_gradients(f[:,2:8].reshape(-1,2,3))
    torch.testing.assert_close(mapped,total_sigma[:,None].expand(-1,3)/4)
    torch.testing.assert_close(f[:,2:5]+f[:,5:8],dr)


def test_constant_grid_potential_is_cQ_not_assumed_cS():
    phi=torch.tensor([[1.,2.],[.3,.7],[.4,-.2]],dtype=torch.float64)
    w=torch.tensor([.2,.3,.5],dtype=torch.float64)
    q=phi.T@(phi*w[:,None])
    actual=AUDIT.op.project_scalar_vxc(phi,w,torch.full_like(w,.37),chunk_size=1)
    torch.testing.assert_close(actual,.37*q,rtol=1e-14,atol=1e-14)
    assert not torch.allclose(q,torch.eye(2,dtype=torch.float64))


def test_frozen_provenance_and_retained_failures():
    protocol=json.loads((ROOT/(AUDIT.PREFIX+'protocol.json')).read_text())
    assert protocol['pilot']==['H2','BeH2','CO']
    assert protocol['sources']=={k:AUDIT.sha(ROOT/k) for k in protocol['sources']}
    assert protocol['driver_sha256']==AUDIT.sha(ROOT/'tools/lap_operator_contract_v1_audit.py')
    with (ROOT/(AUDIT.PREFIX+'pbe_oracle_checks.csv')).open() as handle:
        rows=list(csv.DictReader(handle))
    assert all(r['status']=='PASS' for r in rows if r['path']=='ideal_f64')
    assert {r['system'] for r in rows if r['path']=='repository_f64' and r['status']=='FAIL'}=={'H2','BeH2'}
    with (ROOT/(AUDIT.PREFIX+'variational_checks.csv')).open() as handle:
        assert any(r['status']=='FAIL' for r in csv.DictReader(handle))


def test_audit_has_no_checkpoint_optimizer_or_scf_execution():
    for name in ['lap_operator_contract_v1_audit.py','lap_operator_contract_v1_followup.py']:
        tree=ast.parse((ROOT/'tools'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                text=ast.unparse(node.func)
                assert text not in {'torch.load','pickle.load','torch.optim.AdamW','scf.RKS','scf.UKS'}
                assert not text.endswith(('.step','.kernel','.load_state_dict'))
