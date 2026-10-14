"""Fixed-input XC contract audit. No checkpoint, optimizer or SCF execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyscf
import torch
from pyscf import dft, scf
from pyscf.dft import libxc, numint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dft_functionals import PBE, PBE_CONSTANTS
from train_models import lap_operator as op
from train_models.lap_operator_data import (
    build_molecule_from_metadata,
    load_central_operator_record,
)
from train_models.lap_vxc import LapEnergy, integrated_energy
from train_models.NN_models_lap import pcPBELMLOptimizerV2Lap
from train_models.optuna_joint import _add_mrks_dispersion_once, batch_exc

PREFIX = "lap_operator_contract_v1_"
DATA = ROOT.parent / "publication_dataset_v1"
PILOT = ("H2", "BeH2", "CO")
DOWNLOADS = Path("C:/Users/schne/Downloads" if sys.platform == 'win32' else '/mnt/c/Users/schne/Downloads')
NPZ_ROOT = DOWNLOADS / "mrks_90_ccsd_pt/mrks_90_ccsd_pt"
TOL = {
    "algebra_atol": 1e-10, "algebra_rtol": 1e-9,
    "projection_relative": 1e-10,
    "independent_energy_atol_ha": 1e-6, "independent_energy_rtol": 2e-6,
    "independent_matrix_relative": 1e-5,
    "mixed_energy_atol_ha": 2e-5, "mixed_energy_rtol": 1e-5,
    "mixed_matrix_relative": 1e-4,
    "ideal_energy_atol_ha": 1e-9, "ideal_matrix_relative": 1e-8,
    "source_vxc_atol_ha": 1e-5,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(name, value):
    (ROOT / (PREFIX + name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / (PREFIX + name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tensor(value):
    return torch.tensor(np.asarray(value), dtype=torch.float64)


class Canonical(torch.nn.Module):
    """Explicit constants, including additive exchange=0 and correlation=1."""

    def __init__(self, mixed=False):
        super().__init__()
        self.mixed = mixed

    def forward(self, rho, sigma, lapl):
        del lapl
        if self.mixed:
            rho, sigma = rho.float(), sigma.float()
        constants = PBE_CONSTANTS.to(rho).expand(len(rho), -1)
        return (PBE.F_PBE(rho, sigma, constants, rho.device) * rho.sum(-1)).double()


class IdealPBE(torch.nn.Module):
    """Diagnostic F64 unpolarized PBE/PW_MOD, without production floors.

    This is a different numerical functional, not a replacement production path.
    Independent oracle remains Libxc, not this formula.
    """

    def forward(self, rho, sigma, lapl):
        del lapl
        n = rho.sum(-1).clamp_min(1e-30)
        s = (sigma[:, 0] + 2 * sigma[:, 1] + sigma[:, 2]).clamp_min(0)
        rs = (3 / (4 * math.pi * n)).pow(1 / 3)
        aux = 7.5957 * rs.sqrt() + 3.5876 * rs + 1.6382 * rs.pow(1.5) + .49294 * rs.square()
        ec = -2 * .0310907 * (1 + .21370 * rs) * torch.log1p(1 / (2 * .0310907 * aux))
        kf = (3 * math.pi**2 * n).pow(1 / 3)
        s2 = s / (4 * kf.square() * n.square())
        ex = -3 / 4 * (3 / math.pi)**(1 / 3) * n.pow(1 / 3) * (1 + .804 - .804 / (1 + .2195149727645171 * s2 / .804))
        gamma = (1 - math.log(2)) / math.pi**2
        beta = .06672455060314922
        a = beta / (gamma * torch.expm1(-ec / gamma))
        t2 = math.pi * s / (16 * kf * n.square())
        h = gamma * torch.log1p(beta / gamma * t2 * (1 + a * t2) / (1 + a * t2 + a.square() * t2.square()))
        return n * (ex + ec + h)


class ConstantAdapter(torch.nn.Module):
    """Fixed canonical outputs; one nontrainable dtype sentinel selects repair path."""

    descriptor_protocol = "rho-sigma-total-lapl-tau-free-v1"

    def __init__(self):
        super().__init__()
        self.sentinel = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)

    def forward(self, raw):
        # Keep zero descriptor dependencies so the production mixed partials can
        # request rho/sigma/lapl derivatives even for a constant functional.
        return PBE_CONSTANTS.to(raw).expand(len(raw), -1) + raw.sum(-1, keepdim=True) * 0


def independent_features(phi, grad, lap, dm):
    """Different energy-first graph: total AO quadratic contractions, then spin split."""
    rho = ((phi @ dm) * phi).sum(-1)
    dr = 2 * torch.einsum("gdi,ij,gj->gd", grad, dm, phi)
    lr = 2 * ((lap @ dm) * phi).sum(-1) + 2 * torch.einsum("gdi,ij,gdj->g", grad, dm, grad)
    sigma = dr.square().sum(-1)
    return rho, dr, lr, sigma


def fixture_checks():
    generator = torch.Generator().manual_seed(24019)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    phi, grad, lap = rand(13, 4), rand(13, 3, 4), rand(13, 4)
    weights = torch.rand(13, generator=generator, dtype=torch.float64) + .1
    dm = (torch.eye(4, dtype=torch.float64) * .3).requires_grad_()
    features = op.rks_density_features_from_ao(phi, grad, lap, dm.detach())
    rows = []

    class Polynomial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(.17, dtype=torch.float64))

        def forward(self, rho, sigma, lapl):
            n = rho.sum(-1)
            s = sigma[:, 0] + 2 * sigma[:, 1] + sigma[:, 2]
            return .7 * n.square() + .13 * s + self.theta * lapl.sum(-1).square()

    rng = torch.get_rng_state()
    torch.manual_seed(24019)
    random_r0 = pcPBELMLOptimizerV2Lap().double()
    torch.set_rng_state(rng)
    for name, energy in (("polynomial", Polynomial()), ("fresh_seed24019_R0", LapEnergy(random_r0))):
        n, dr, lr, s = independent_features(phi, grad, lap, dm)
        del dr
        # Energy-first: calls energy only; never calls operator/local partial helpers.
        e = (energy(torch.stack([n / 2, n / 2], -1), s[:, None].expand(-1, 3) / 4,
                    torch.stack([lr / 2, lr / 2], -1)) * weights).sum()
        reference = torch.autograd.grad(e, dm, create_graph=True)[0]
        predicted = op.assemble_rks_operator(energy, features, weights, phi, grad, lap, chunk_size=5)
        gap = float((reference - predicted).detach().norm())
        bound = TOL['algebra_atol'] + TOL['algebra_rtol'] * float(reference.detach().norm())
        rows.append({'fixture': name, 'check': 'energy_DM_gradient', 'absolute_error': gap,
                     'bound': bound, 'status': 'PASS' if gap <= bound else 'FAIL'})
        direction = rand(4, 4)
        direction = (direction + direction.T) / 2
        contraction_gap = float(((predicted-reference) * direction).sum().detach().abs())
        rows.append({'fixture': name, 'check': 'full_symmetric_trace_contraction',
                     'absolute_error': contraction_gap, 'status': 'PASS' if contraction_gap <= bound * float(direction.norm()) else 'FAIL'})
        if name == 'polynomial':
            target = rand(4, 4)
            target = (target + target.T) / 2
            overlap = torch.eye(4, dtype=torch.float64)
            loss = op.operator_loss(predicted, target, overlap)
            actual = torch.autograd.grad(loss, energy.theta, retain_graph=True)[0]
            other = ((reference-target).square().sum()/4)
            expected = torch.autograd.grad(other, energy.theta)[0]
            error = float((actual-expected).abs())
            rows.append({'fixture': name, 'check': 'operator_loss_parameter_gradient',
                         'absolute_error': error, 'status': 'PASS' if error <= 1e-8 else 'FAIL'})
            # Demonstrable negative controls, assembled on fixture copies only.
            bad_lap = op.assemble_rks_operator(energy, features, weights, phi, grad, torch.zeros_like(lap), chunk_size=5)
            wrong_sigma = features.clone()
            wrong_sigma[:, 5:8] *= .5
            bad_sigma = op.assemble_rks_operator(energy, wrong_sigma, weights, phi, grad, lap, chunk_size=5)
            for control, wrong in [('extra_spin_factor',2*predicted), ('missing_laplacian_AO_terms',bad_lap), ('wrong_sigma_mapping',bad_sigma)]:
                error = float((wrong-reference).detach().norm())
                rows.append({'fixture': name, 'check': control, 'absolute_error': error,
                             'status': 'PASS' if error > 100*bound else 'FAIL', 'meaning': 'PASS means deliberately wrong copy detected'})
    return rows


def metric_checks():
    rng = np.random.default_rng(24020)
    s = rng.normal(size=(5,5)); s = s.T@s + np.eye(5)
    v = rng.normal(size=(5,5)); v = (v+v.T)/2
    r = rng.normal(size=(5,5)); r = (r+r.T)/2
    vt, rt, st = tensor(v).requires_grad_(), tensor(r), tensor(s)
    loss = op.operator_loss(vt, rt, st)
    actual = torch.autograd.grad(loss, vt)[0].numpy()
    si = np.linalg.inv(s)
    derivative = 2*si@(v-r)@si/5
    t = rng.normal(size=(5,5)); t = t + 5*np.eye(5)
    transformed = op.operator_loss(tensor(t.T@v@t), tensor(t.T@r@t), tensor(t.T@s@t))
    same = op.operator_loss(vt,vt.detach(),st)
    zero_grad = torch.autograd.grad(same,vt)[0]
    rows=[]
    for name, gap in [('matrix_gradient', np.linalg.norm(actual-derivative)), ('same_span_basis_invariance',abs(float(transformed-loss))), ('self_loss',float(same)), ('self_gradient',float(zero_grad.norm()))]:
        rows.append({'system':'analytic_fixture','check':name,'absolute_error':gap,'status':'PASS' if gap <= 1e-10 else 'FAIL'})
    c=.23
    shift=op.operator_loss(tensor(c*s),tensor(np.zeros_like(s)),st)
    rows.append({'system':'analytic_fixture','check':'known_overlap_shift_loss','absolute_error':abs(float(shift)-c*c),'status':'PASS' if abs(float(shift)-c*c)<1e-10 else 'FAIL'})
    return rows


def freeze():
    sources = ['train_models/'+name+'.py' for name in ['build_lap_operator_data','lap_operator_data','lap_operator','lap_vxc','lap_moo_training','optuna_joint','prepare_vxc','build_mrks_stencils','compute_mrks_pbe0_d3bj','NN_models_lap']]
    sources += ['dft_functionals/PBE.py','dft_functionals/constants.py','tools/build_publication_dataset.py',
                'train_models/mrks_source_audit_v2.json','lap_operator_corpus_manifest.json']
    corpus = read(ROOT/'lap_operator_corpus_manifest.json')
    corpus_root = (ROOT/corpus['external_corpus_path_relative_to_manifest']).resolve()
    assert sha(corpus_root/'manifest.json') == corpus['external_manifest']['sha256']
    git='git' if sys.platform=='win32' else 'git.exe'
    protocol={'namespace':PREFIX,'head':subprocess.check_output([git,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'branch':subprocess.check_output([git,'branch','--show-current'],cwd=ROOT,text=True).strip(),
              'initial_worktree':subprocess.check_output([git,'status','--short'],cwd=ROOT,text=True),
              'wall_origin_unix':time.time(),'budget_seconds':10800,'pilot':list(PILOT),
              'sources':{name:sha(ROOT/name) for name in sources},'driver_sha256':sha(__file__),
              'versions':{'python':platform.python_version(),'torch':torch.__version__,'pyscf':pyscf.__version__,'libxc':libxc.__version__},
              'tolerances':TOL,'tolerance_basis':'F64 operator unit tests 1e-10/1e-9; independent and mixed checks separately allow explicit source constant rounding/regularization budget, not historical model errors; FAILs retained',
              'canonical_constants_stored_f32':PBE_CONSTANTS.tolist(),
              'production_regularization':{'rs_density_add':1e-7,'xs_density_add':1e-10,'xs_sigma_add':1e-10**(8/3),'zeta_clip':'none; z_thr returns input','expm1_cap':87},
              'oracle':'PySCF nr_rks(PBE), original dm/coords/weights; Libxc conventions inspected from installed source; no SCF',
              'counterfactual':'IdealPBE analytical F64 PW_MOD coefficients, rho floor1e-30 only; numerical functional differs from production',
              'no_checkpoint_or_optimizer_load':True,'no_training_or_scf':True,
              'corpus_root':str(corpus_root),'pilot_files':{r['metadata']['system_name']:{'path':str(corpus_root/r['file']),'sha256':r['file_sha256'],'npz_sha256':r['metadata']['source']['npz_file_sha256']} for r in corpus['records'] if r['metadata']['system_name'] in PILOT}}
    assert protocol['branch']=='lap_full_vxc'
    if (ROOT/(PREFIX+'protocol.json')).exists():
        raise FileExistsError('Audit namespace exists; authenticate before reusing it.')
    write('protocol',protocol)


def census(protocol):
    corpus=read(ROOT/'lap_operator_corpus_manifest.json')
    disp_path=DATA/'mrks/dispersion.json'
    disp=read(disp_path)
    conversions=read(DATA/'provenance/mrks_conversion.json')
    expected=next(r['sha256'] for r in conversions['files'] if r['file']=='mrks/dispersion.json')
    assert sha(disp_path)==expected
    rows=[]; targets=[]; npz_receipts={}
    publication={r['source_id']:r for r in (json.loads(line) for line in (DATA/'mrks/systems.jsonl').read_text().splitlines())}
    for record in corpus['records']:
        meta=record['metadata']; name=meta['system_name']; source=NPZ_ROOT/meta['source']['npz_relative_id']
        assert sha(Path(protocol['corpus_root'])/record['file'])==record['file_sha256']
        mapping=publication[name]
        assert mapping['source_sha256']==record['file_sha256']
        with h5py.File(DATA/mapping['shard'],'r') as handle:
            group=handle[mapping['group']]
            assert json.loads(group.attrs['source_metadata_json'])==meta
            assert float(group['Exc'][()])==record['Exc_legacy_ha']
        assert sha(source)==meta['source']['npz_file_sha256']
        npz_receipts[name]={'path':str(source),'sha256':sha(source)}
        with np.load(source,allow_pickle=False) as z:
            info=dict(zip(z['meta_keys'].tolist(),z['meta'].tolist()))
        target=record['Exc_legacy_ha']; correction=disp.get(name,0.)
        observed=float(_add_mrks_dispersion_once(torch.tensor(0.,dtype=torch.float64),name,disp,True))
        assert observed==correction
        singleton=batch_exc([name],torch.tensor([target+correction],dtype=torch.float64),torch.tensor([target],dtype=torch.float64))
        rows.append({'system':name,'stored_target_ha':target,'dispersion_ha':correction,'lookup_present':name in disp,'additions':1,'effective_pure_xc_target_ha':target-correction,'intended_convention':'PBE0-D3(BJ) prediction correction; original E_xc compatibility UNRESOLVED','dispersion_sha256':sha(disp_path)})
        rows[-1]['loss_shift_with_prediction_at_stored_target_kcal']=float(singleton)
        targets.append({'system':name,'quantity':'E_xc','legacy_value_ha':target,'exc_wf_npz_ha':info['exc_wf'],'legacy_minus_exc_wf_ha':target-info['exc_wf'],'electron_count':info['nelec'],'nao':meta['nao'],'grid_points':meta['point_count'],'formula_status':'TARGET_DEFINITION_UNRESOLVED','gauge_status':'UNRESOLVED','source_npz_sha256':meta['source']['npz_file_sha256'],'definition':'legacy E_xc promoted F32->F64; exc_wf physical formula not authenticated','Vrho_mapping':'legacy common scalar matches NPZ vxc_grid; generator semantic provenance unresolved'})
    table('dispersion_contract',rows); table('target_definitions',targets)
    return {'npz_sources':npz_receipts,'dispersion_sha256':sha(disp_path),'conversion_sha256':sha(DATA/'provenance/mrks_conversion.json'),'census_count':len(rows)}


def molecular(name, protocol, cache_root):
    started=time.perf_counter()
    spec=protocol['pilot_files'][name]; path=Path(spec['path'])
    assert sha(path)==spec['sha256']
    record=load_central_operator_record(path)
    source=NPZ_ROOT/record.metadata['source']['npz_relative_id']
    assert sha(source)==spec['npz_sha256']
    with np.load(source,allow_pickle=False) as z:
        z={key:z[key] for key in z.files}
    assert np.array_equal(record.coords64,z['grid_coords'][record.npzrow])
    assert np.array_equal(record.dmks,z['dm_ks'])
    vxc_error=float(np.max(np.abs(record.VxcLegacy-z['vxc_grid'][record.npzrow])))
    assert vxc_error<=TOL['source_vxc_atol_ha']
    mol=build_molecule_from_metadata(record.metadata)
    ni=numint.NumInt()
    grids=dft.gen_grid.Grids(mol)
    grids.coords=record.coords64.copy(); grids.weights=record.weights.copy(); grids.non0tab=None
    nelec, exc, matrix=ni.nr_rks(mol,grids,'PBE',record.dmks,max_memory=256)
    # UKS oracle is independent of repository's sigma mapping/spin assembly.
    nu, eu, vu=ni.nr_uks(mol,grids,'PBE',np.stack([record.dmks/2,record.dmks/2]),max_memory=256)
    matrices={key:np.zeros_like(matrix) for key in ['repository_f64','repository_mixed','ideal_f64']}
    energies={key:0. for key in matrices}
    q=np.zeros_like(matrix); projected=np.zeros_like(matrix); chunked_ref=np.zeros_like(matrix)
    densities=0.; wf_density_difference=0.; nwf=0.
    energies_obj={'repository_f64':Canonical(),'repository_mixed':LapEnergy(ConstantAdapter()),'ideal_f64':IdealPBE()}
    for start in range(0,len(record.weights),4096):
        assert time.time()-protocol['wall_origin_unix']<protocol['budget_seconds']-120
        sl=slice(start,min(start+4096,len(record.weights)))
        ao=ni.eval_ao(mol,record.coords64[sl],deriv=2)
        phi=tensor(ao[0]); grad=tensor(ao[1:4].transpose(1,0,2)); lap=tensor(ao[4]+ao[7]+ao[9]); w=tensor(record.weights[sl])
        features=op.rks_density_features_from_ao(phi,grad,lap,tensor(record.dmks))
        rt=features[:,:2].sum(-1).numpy()
        densities+=float(np.dot(rt,record.weights[sl]))
        source_rho=z['rho_wf'][record.npzrow[sl]]
        wf_density_difference+=float(np.dot(np.abs(rt-source_rho),record.weights[sl]))
        nwf+=float(np.dot(source_rho,record.weights[sl]))
        for key,energy in energies_obj.items():
            sigma=op.sigma_from_gradients(features[:,2:8].reshape(-1,2,3))
            if key=='repository_mixed':
                f32=features.float()
                sigma32=op.sigma_from_gradients(f32[:,2:8].reshape(-1,2,3))
                local=energy(f32[:,:2],sigma32,f32[:,8:])
            else:
                local=energy(features[:,:2],sigma,features[:,8:])
            energies[key]+=float((local*w).sum().detach())
            matrices[key]+=op.assemble_rks_operator(energy,features,w,phi,grad,lap,chunk_size=256,create_graph=False).detach().numpy()
        pp=ao[0]
        # Independent direct contraction, no production builder/project helper.
        projected+=np.einsum('gm,g,gn->mn',pp,record.weights[sl]*record.VxcLegacy[sl],pp,optimize=True)
        q+=pp.T@(pp*record.weights[sl,None])
        chunked_ref+=op.project_scalar_vxc(phi,w,tensor(record.VxcLegacy[sl]),chunk_size=137).numpy()
    ev,vec=np.linalg.eigh(record.Overlap); x=(vec*ev**-.5)@vec.T
    oracle_orth=x@matrix@x
    checks=[]
    native_integrated=float(integrated_energy(energies_obj['repository_mixed'], tensor(record.DensityDescriptorsN10).float()[:,None,:],tensor(record.weights),4096).detach())
    checks.append({'system':name,'path':'production_integrated_energy_chunk4096','energy_abs_error_ha':abs(native_integrated-energies['repository_mixed']),'status':'PASS' if abs(native_integrated-energies['repository_mixed'])<TOL['mixed_energy_atol_ha'] else 'FAIL'})
    for key, production_matrix in matrices.items():
        delta=production_matrix-matrix
        abs_energy=abs(energies[key]-float(exc))
        orth_gap=float(np.linalg.norm(x@delta@x)); orth_rel=orth_gap/float(np.linalg.norm(oracle_orth))
        if key=='repository_mixed':
            energy_bound=TOL['mixed_energy_atol_ha']+TOL['mixed_energy_rtol']*abs(float(exc)); matrix_bound=TOL['mixed_matrix_relative']
        elif key=='ideal_f64':
            energy_bound=TOL['ideal_energy_atol_ha']; matrix_bound=TOL['ideal_matrix_relative']
        else:
            energy_bound=TOL['independent_energy_atol_ha']+TOL['independent_energy_rtol']*abs(float(exc)); matrix_bound=TOL['independent_matrix_relative']
        checks.append({'system':name,'path':key,'n_grid':len(record.weights),'n_AO':mol.nao_nr(),'nelec_oracle':float(nelec),'nelec_production':densities,'nelec_dm_overlap':float(np.trace(record.dmks@record.Overlap)),'Exc_oracle_ha':float(exc),'Exc_production_ha':energies[key],'energy_abs_error_ha':abs_energy,'energy_bound_ha':energy_bound,'matrix_abs_fro':float(np.linalg.norm(delta)),'orth_abs_fro':orth_gap,'orth_relative':orth_rel,'orth_relative_bound':matrix_bound,'status':'PASS' if abs_energy<=energy_bound and orth_rel<=matrix_bound else 'FAIL'})
    checks.append({'system':name,'path':'Libxc_RKS_vs_equal_spin_UKS','energy_abs_error_ha':abs(float(exc)-float(eu)),'matrix_abs_fro':float(np.linalg.norm(matrix-vu[0])),'spin_beta_abs_fro':float(np.linalg.norm(vu[0]-vu[1])),'nelec_uks':float(np.sum(nu)),'status':'PASS' if abs(float(exc)-float(eu))<1e-10 and np.linalg.norm(matrix-vu[0])<1e-8 else 'FAIL'})
    refgap=float(np.linalg.norm(projected-record.RefAO)); refrel=refgap/np.linalg.norm(record.RefAO)
    projection={'system':name,'source_row_identity':'PASS','source_vxc_max_abs_ha':vxc_error,'independent_ref_abs_fro':refgap,'independent_ref_relative':float(refrel),'chunk137_abs_fro':float(np.linalg.norm(projected-chunked_ref)),'status':'PASS' if refrel<=TOL['projection_relative'] else 'FAIL'}
    a=x@(matrices['repository_mixed']-record.RefAO)@x
    b=x@q@x; n=len(a); total=np.sum(a*a)/n; c=np.trace(a)/n
    cq=np.sum(a*b)/np.sum(b*b)
    trace_part=c*c; qpart=cq*cq*np.sum(b*b)/n
    metrics={'system':name,'check':'PBE_vs_mRKS_metric_gauge','n_AO':n,'overlap_min':float(ev.min()),'overlap_max':float(ev.max()),'overlap_condition':float(ev.max()/ev.min()),'SPD_threshold':float(np.finfo(float).eps*n*ev.max()),'Q_minus_S_relative_fro':float(np.linalg.norm(q-record.Overlap)/np.linalg.norm(record.Overlap)),'XQX_minus_I_fro':float(np.linalg.norm(b-np.eye(n))),'XQX_minus_I_spectral':float(np.linalg.norm(b-np.eye(n),2)),'loss_ha2':float(total),'trace_c_ha':float(c),'trace_constant_loss':float(trace_part),'tracefree_loss':float(np.sum((a-c*np.eye(n))**2)/n),'discrete_cQ_ha':float(cq),'discrete_constant_loss':float(qpart),'discrete_orthogonal_loss':float(np.sum((a-cq*b)**2)/n),'trace_decomposition_error':float(abs(total-trace_part-np.sum((a-c*np.eye(n))**2)/n)),'Q_decomposition_error':float(abs(total-qpart-np.sum((a-cq*b)**2)/n)),'gauge_provenance':'UNRESOLVED','status':'PASS'}
    # Validate reference occupied/virtual projectors before block decomposition.
    coeff=z['mo_coeff']; occ=z['mo_occ']; ortho=float(np.linalg.norm(coeff.T@record.Overlap@coeff-np.eye(coeff.shape[1])))
    occupation_ok=np.all(np.isclose(occ,0,atol=1e-8)|np.isclose(occ,2,atol=1e-8))
    metrics['MO_orthogonality_fro']=ortho
    if ortho<1e-8 and occupation_ok:
        sh=(vec*np.sqrt(ev))@vec.T; co=sh@coeff[:,occ>1]
        po=co@co.T; pv=np.eye(n)-po
        metrics.update({'occupied_block_loss':float(np.sum((po@a@po)**2)/n),'mixed_blocks_loss':float(2*np.sum((po@a@pv)**2)/n),'virtual_block_loss':float(np.sum((pv@a@pv)**2)/n),'occupation_status':'PASS'})
    else:
        metrics['occupation_status']='BLOCKED'
    # Source total-energy candidate reconstruction; does not authenticate exporter.
    hcore=scf.hf.get_hcore(mol); j=scf.hf.get_jk(mol,record.dmks,hermi=1,with_j=True,with_k=False)[0]
    ts=float(np.einsum('ij,ji',record.dmks,mol.intor_symmetric('int1e_kin')))
    vext=float(np.einsum('ij,ji',record.dmks,mol.intor_symmetric('int1e_nuc')))
    hartree=float(.5*np.einsum('ij,ji',record.dmks,j)); enuc=float(mol.energy_nuc())
    csv_path=DOWNLOADS/'extrapolation_e.csv'
    with csv_path.open(encoding='utf-8-sig') as handle:
        csv_rows={r['System']:r for r in csv.DictReader(handle)}
    reconstruction={'system':name,'legacy_exc_ha':float(record.Exc),'Ts_ks_ha':ts,'Vext_ks_ha':vext,'EH_ks_ha':hartree,'Enuc_ha':enuc,'t_wf_metadata_ha':dict(zip(z['meta_keys'].tolist(),z['meta'].tolist()))['t_wf'],'integrated_abs_rho_ks_minus_wf':wf_density_difference,'nelec_wf_grid':nwf,'E_xc_formula_status':'UNRESOLVED','exc_wf_definition_status':'UNRESOLVED','meaning':'numeric recipe comparison is not original exporter authentication; WF/KS density difference reported'}
    del hcore
    if name in csv_rows:
        etotal=float(csv_rows[name]['E_CBS_1'])
        reconstructed=etotal-enuc-ts-vext-hartree
        reconstruction.update({'E_CBS1_total_ha':etotal,'E_CBS1_recipe_exc_ha':reconstructed,'recipe_minus_legacy_ha':reconstructed-float(record.Exc)})
    dest=cache_root/(name+'.npz')
    np.savez_compressed(dest,oracle=matrix,repository_f64=matrices['repository_f64'],repository_mixed=matrices['repository_mixed'],ideal_f64=matrices['ideal_f64'],Q=q,RefAO_independent=projected)
    return checks,projection,metrics,reconstruction,{'path':str(dest),'sha256':sha(dest),'active_seconds':time.perf_counter()-started}


def run():
    protocol=read(ROOT/(PREFIX+'protocol.json'))
    assert protocol['driver_sha256']==sha(__file__)
    assert protocol['sources']=={k:sha(ROOT/k) for k in protocol['sources']}
    torch.set_num_threads(1)
    started=time.perf_counter()
    provenance=census(protocol)
    fixtures=fixture_checks(); metric_rows=metric_checks()
    table('variational_checks',fixtures)
    checks=[]; projections=[]; reconstructions=[]; arrays={}
    cache_root=ROOT.parent/'lap_operator_contract_v1_arrays_20261014'
    cache_root.mkdir(exist_ok=True)
    for name in PILOT:
        c,p,m,r,a=molecular(name,protocol,cache_root)
        checks.extend(c); projections.append(p); metric_rows.append(m); reconstructions.append(r); arrays[name]=a
        table('pbe_oracle_checks',checks); table('reference_projection_checks',projections); table('metric_gauge_checks',metric_rows)
        table('energy_reconstruction_checks',reconstructions)
        print(name,'completed',round(a['active_seconds'],2),flush=True)
    provenance.update({'arrays':arrays,'sources_unchanged':protocol['sources']=={k:sha(ROOT/k) for k in protocol['sources']},'trained_checkpoints_loaded':0,'optimizer_states_loaded':0,'training_updates':0,'SCF_executed':False,'active_seconds':time.perf_counter()-started,'scientific_wall_seconds':time.time()-protocol['wall_origin_unix'],'budget_respected':time.time()-protocol['wall_origin_unix']<=protocol['budget_seconds']})
    write('provenance',provenance)
    decision={'primary_target_definition':'UNRESOLVED','dispersion_contract':'UNRESOLVED','gauge_provenance':'UNRESOLVED','variational_assembly':'PASS' if all(r['status']=='PASS' for r in fixtures) else 'FAIL','reference_projection':'PASS' if all(r['status']=='PASS' for r in projections) else 'FAIL','independent_PBE_oracle':'PASS' if all(r['status']=='PASS' for r in checks) else 'FAIL','scientific_verdict':['TARGET_PROVENANCE_UNRESOLVED'],'stationary_condition':'NOT RUN: generator Fock/gauge convention not authenticated','trained_checkpoint_access':False,'optimizer_access':False}
    if any(r['status']=='FAIL' for r in checks):
        decision['scientific_verdict'].append('NUMERICAL_OR_METRIC_SENSITIVITY_IDENTIFIED')
    if decision['variational_assembly']=='FAIL' or decision['reference_projection']=='FAIL':
        decision['scientific_verdict'].append('PARTIAL: failed comparison requires localization')
    write('decision',decision)


if __name__=='__main__':
    if sys.argv[1:]==['freeze']:
        freeze()
    elif sys.argv[1:]==['run']:
        run()
    else:
        raise SystemExit('Usage: audit.py freeze | run')
