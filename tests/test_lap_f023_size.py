"""Fixed size-law, source identity, prefix reduction and no-training gates."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import lap_f023_size as exp


def test_electron_counts_handle_charge_without_neutrality_guess():
    assert exp.electron_count([1],0)==1
    assert exp.electron_count([8,1,1],0)==10
    assert exp.electron_count([3],1)==2
    assert exp.electron_count([17],-1)==18
    with pytest.raises(ValueError):
        exp.electron_count([1],1)
    with pytest.raises(TypeError):
        exp.electron_count([8],.5)
    with pytest.raises(ValueError):
        exp.electron_count([1.5],0)


def test_exact_authenticated_census_and_H_target():
    census = exp.audit.read(exp.OUT/'authenticated_electrons.json')
    assert len(census['ae17'])==17 and len(census['mrks'])==90
    H = next(r for r in census['ae17'].values() if r['source_id']=='H_ae17')
    assert H['Z']==H['Ne']==1 and H['ionic_charge']==0
    assert H['reference_kcal']==-313.7547607421875
    for r in census['mrks'].values():
        assert abs(r['trace_dm_S']-r['Ne'])<=1e-6
        assert r['n_ao']>0 and r['npz_sha256'] and r['record_sha256']


def test_cache_source_pins_and_inherited_moments():
    p = exp.read('provenance')
    assert p['authenticated'] and p['authenticated_atoms']==17 and p['authenticated_mrks']==90
    assert exp.audit.sha(exp.f010.PARENT_CHECKPOINT)==exp.dual.CHECKPOINT
    parent = torch.load(exp.f010.PARENT_CHECKPOINT,map_location='cpu',weights_only=False)
    assert exp.f010.continuation.moment_sha(parent['optimizer'])==exp.dual.MOMENT
    for relative,digest in exp.read('protocol')['source_hashes'].items():
        assert exp.audit.sha(exp.ROOT/relative)==digest
    assert exp.audit.sha(exp.OUT/'cached_arrays.npz')==p['cached_arrays_sha256']


def test_frozen_population_and_task_coefficients():
    p = exp.read('protocol')
    assert p['size_exponent']==.5 and p['c010']==9.974823077811354
    assert p['lambdas']==exp.b8.LAMBDAS and p['operator_size_weight']==1
    assert p['checkpoints']==[0,64,128] and p['updates']==128 and p['arms']==1
    assert p['lr']==2.5e-5 and p['scheduler'] is None and p['clipping'] is None
    prefix=exp.f010._batches()[:128]
    assert [r['cursor'] for r in prefix]==list(range(128))
    assert sum(len(r['relchem']) for r in prefix)==1024
    assert 'S128' in p['calibration_population']
    assert p['scientific_baselines']==exp.b8.BASELINE


def test_norm_preservation_F64_and_unchanged_chemistry_operator():
    cal=exp.read('calibration')
    counts=exp.audit.read(exp.OUT/'authenticated_electrons.json')
    prefix=exp.f010._batches()[:128]
    with np.load(exp.OUT/'cached_arrays.npz') as d:
        for t,key in [('ae17','c_A'),('exc','c_E')]:
            assert d[t].dtype==d[t+'_new'].dtype==np.float64
            weights=np.array([counts['ae17'][r['ae17']['identity']]['Ne']**-.5 if t=='ae17'
                              else counts['mrks'][r['mrks_id']]['Ne']**-.5 for r in prefix])
            np.testing.assert_allclose(d[t+'_new'],d[t]*weights[:,None]*cal[key],rtol=1e-14,atol=1e-12)
            assert np.linalg.norm(d[t+'_new'].mean(0))==pytest.approx(np.linalg.norm(d[t].mean(0)),rel=1e-14)
            # Independent sum of weighted singleton vectors, not a new gradient calculation.
            np.testing.assert_allclose(d[t+'_new'].mean(0),sum(d[t][i]*weights[i]*cal[key] for i in range(128))/128,
                                       rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(d['op'],d['op_new'])
        expected=exp.census.chemistry_batches(prefix)*exp.b8.LAMBDAS['relchem']*exp.audit.geometry.C010
        np.testing.assert_array_equal(d['chemistry'],expected)
        for label in ('old','new'):
            expected=d['chemistry']+sum(exp.b8.LAMBDAS[t]*d[t+('_new' if label=='new' else '')] for t in ('ae17','exc','op'))
            np.testing.assert_array_equal(d['joint_'+label],expected)
            assert np.isfinite(expected.astype(np.float32)).all()


def test_actual_scalar_gradient_parity_and_operator_cache():
    with (exp.ROOT/'lap_f023_weighted_gradient_parity.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert {r['task'] for r in rows}=={'ae17','exc'}
    assert all(r['passed']=='True' and float(r['absolute_gap'])<=float(r['bound']) for r in rows)
    native=exp.read('preflight_decision')
    assert native['live_gradient_relative_gaps']=={'ae17':0.,'exc':0.,'op':0.}


def test_temporary_steps_restore_model_buffers_moments_rng():
    r=exp.read('restoration')
    assert all(r[k] for k in ('model','buffers','optimizer_moments','rng','parent_checkpoint_unchanged'))
    assert r['retained_updates']==0
    assert exp.read('constraint_checks')['passed']
    with (exp.ROOT/'lap_f023_native_step_preflight.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert {r['arm'] for r in rows}=={'old','new'}
    assert all(r['constraints_passed']=='True' for r in rows)


def test_no_go_is_mechanism_not_pareto_rule():
    p=exp.read('preflight_decision')
    assert p['numerical_pass'] and not p['mechanism_pass']
    assert not p['projection_improved'] and not p['native_H_improved']
    assert not p['training_authorized'] and p['optimizer_updates_retained']==0
    assert p['native_H_error_difference_new_minus_old']>exp.read('protocol')['mechanism_H_comparison_resolution_kcal']
    bounds=exp.read('legacy_conflict_bounds')
    assert bounds['new']['legacy_dot_physics']<bounds['old']['legacy_dot_physics']
    assert bounds['new']['R_conflict']>bounds['old']['R_conflict']>1


def test_no_heldout_or_SCF_access_and_resource_accounting():
    d=exp.read('decision')
    assert d['classification']=='NO-GO' and d['optimizer_updates']==0 and not d['training_performed']
    assert d['Clean28']=='NOT EVALUATED' and d['physical_eligibility']=='NOT RUN'
    assert exp.read('provenance')['Clean28_accessed'] is False
    assert d['scientific_wall_seconds']<exp.BUDGET==36000
    assert d['budget_respected']
    assert not (exp.OUT/'ordinary_sgd_adamw').exists()
    assert exp.read('forward_receipt')['records']==174
