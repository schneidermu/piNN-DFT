"""Focused tests of matrix-free diagnostic contracts; no scientific datasets."""
import numpy as np
import torch

import lap_q_parameter_sketch_diagnostic as diagnostic


def test_row_views_preserve_head():
    states=[{'seed':seed,'regime':regime} for seed in (11,23,41) for regime in ('P67','P536','PBE_TANGENT','PBE_HEAD')]
    views=diagnostic.row_views(states)
    assert len(views['FULL'])==1536 and len(views['NON_HEAD'])==1152 and len(views['PBE_HEAD_ONLY'])==384
    assert set(views['NON_HEAD']).isdisjoint(views['PBE_HEAD_ONLY'])
    assert sorted(views['NON_HEAD']+views['PBE_HEAD_ONLY'])==views['FULL']
    assert views['PBE_HEAD_ONLY'][:128]==list(range(384,512))


def test_zero_response_nonzero_parameter_capacity():
    zero=torch.zeros(2,dtype=torch.float64)
    values,pull=torch.func.vjp(lambda theta:torch.stack([theta[0],2*theta[1]]),zero)
    assert torch.equal(values,zero)
    assert torch.equal(pull(torch.ones(2,dtype=torch.float64))[0],torch.tensor([1.,2.],dtype=torch.float64))


def test_parameter_coordinate_roundtrip_and_rows():
    shapes=[(2,3),(1,)];values=[torch.arange(6,dtype=torch.float64).reshape(shapes[0]),torch.tensor([7.])]
    flat=torch.cat([v.reshape(-1) for v in values]);out=[flat[:6].reshape(shapes[0]),flat[6:].reshape(shapes[1])]
    assert all(torch.equal(a,b) for a,b in zip(values,out))
    assert torch.equal(torch.cat([torch.arange(64),torch.arange(64)+64]),torch.arange(128))


def test_adjoint_vjp_jvp_consistency():
    a=torch.arange(400,dtype=torch.float64).reshape(5,80)/400
    theta=torch.zeros(80,dtype=torch.float64);v=torch.linspace(-1,1,80,dtype=torch.float64);w=torch.ones(5,dtype=torch.float64)
    _,back=torch.func.vjp(lambda x:a@x,theta)
    image=torch.func.jvp(lambda x:a@x,(theta,),(v,))[1]
    assert torch.allclose(image,a@v) and torch.allclose(w@image,back(w)[0]@v)


def test_orthogonality_and_probes_deterministic():
    z=torch.randn(80,40,dtype=torch.float64,generator=torch.Generator().manual_seed(1))
    q,error=diagnostic.orthonormalize(z)
    assert q.shape==(80,40) and error<=1e-8
    assert torch.equal(diagnostic.probes(64,40,42),diagnostic.probes(64,40,42))
    assert not torch.equal(diagnostic.probes(64,40,42),diagnostic.probes(64,40,314159))


def test_full_sketch_synthetic_determinism():
    a=torch.randn(64,80,dtype=torch.float64,generator=torch.Generator().manual_seed(9))
    theta=torch.zeros(80,dtype=torch.float64)
    order=[{'name':'x_output_layer.weight','start':0,'stop':80}]
    first=diagnostic.sketch(lambda x:a@x,theta,40,42,order)
    second=diagnostic.sketch(lambda x:a@x,theta,40,42,order)
    for key in first[1]:assert np.array_equal(first[1][key],second[1][key])
    assert first[0]['orthogonality_max_abs']<=1e-8


def test_functional_plus_minus_preserves_parameters():
    model=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
    saved=model.weight.detach().clone();step=torch.ones_like(saved)*1e-5
    for sign in (-1,1):
        torch.func.functional_call(model,{'weight':saved+sign*step},(torch.ones(1,2,dtype=torch.float64),))
    assert torch.equal(model.weight,saved)


def test_manifest_mismatch_fails_closed(tmp_path,monkeypatch):
    import json
    import shutil
    for name in ('protocol','parameter_manifest','parameter_group_map','row_partition'):
        shutil.copyfile(diagnostic.OUT/(name+'.json'),tmp_path/(name+'.json'))
    monkeypatch.setattr(diagnostic,'OUT',tmp_path)
    diagnostic.validate_contract()
    f=tmp_path/'row_partition.json';value=json.loads(f.read_text());value['views']['FULL'].pop();f.write_text(json.dumps(value))
    import pytest
    with pytest.raises(AssertionError):diagnostic.validate_contract()


def test_h2_change_fails_closed(tmp_path,monkeypatch):
    import json
    import shutil
    for name in ('protocol','parameter_manifest','parameter_group_map','row_partition'):
        shutil.copyfile(diagnostic.OUT/(name+'.json'),tmp_path/(name+'.json'))
    monkeypatch.setattr(diagnostic,'OUT',tmp_path)
    f=tmp_path/'protocol.json';value=json.loads(f.read_text());value['h2'][0]*=2;f.write_text(json.dumps(value))
    import pytest
    with pytest.raises(AssertionError):diagnostic.validate_contract()
