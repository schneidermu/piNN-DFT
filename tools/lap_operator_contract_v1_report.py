"""Generate compact audit report/receipts from saved numerical tables."""

import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'lap_operator_contract_v1_'
DATA = ROOT.parent/'publication_dataset_v1'


def read(name):
    return json.loads((ROOT/(PREFIX+name+'.json')).read_text())


def rows(name):
    with (ROOT/(PREFIX+name+'.csv')).open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(name,value):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def main():
    import h5py
    import numpy as np

    protocol=read('protocol'); provenance=read('provenance'); decision=read('decision')
    assert protocol['sources']=={name:sha(ROOT/name) for name in protocol['sources']}
    assert protocol['driver_sha256']==sha(ROOT/'tools/lap_operator_contract_v1_audit.py')
    downloads=Path('C:/Users/schne/Downloads')
    legacy=downloads/'data_vxc_train.pickle'
    assert sha(legacy)=='0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49'
    # This hash-authenticated pickle contains reference-data records, not NN
    # checkpoints or optimizer states. No model loading API is invoked.
    with legacy.open('rb') as handle:
        legacy_rows=pickle.load(handle)
    legacy_index={r['Name']:r for r in legacy_rows}
    assert len(legacy_index)==90 and len(legacy_rows)==90
    with (ROOT/'train_models/dispersions/dispersions_mrks.pickle').open('rb') as handle:
        dispersion=pickle.load(handle)
    assert dispersion==json.loads((DATA/'mrks/dispersion.json').read_text())
    source_preservation=[]
    corpus=json.loads((ROOT/'lap_operator_corpus_manifest.json').read_text())
    for r in corpus['records']:
        old=legacy_index[r['metadata']['system_name']]
        path=Path(r['file'])
        central=(ROOT/corpus['external_corpus_path_relative_to_manifest']/path).resolve()
        with h5py.File(central,'r') as handle:
            scalar_gap=abs(float(old['E_xc'])-float(handle['Exc'][()]))
            vxc_gap=float(np.max(np.abs(old['Vrho'].numpy().reshape(-1).astype(float)-handle['VxcLegacy'][...])))
            weights_gap=float(np.max(np.abs(old['Weights'].numpy().reshape(-1).astype(float)-handle['weights'][...])))
            coordinates_equal=np.array_equal(old['Grid'].numpy()[:,:3],handle['legacycoords32'][...])
        assert scalar_gap==0 and vxc_gap==0 and weights_gap==0 and coordinates_equal
        source_preservation.append({'system':old['Name'],'E_xc_abs_gap_ha':scalar_gap,
                                    'Vxc_max_abs_gap_ha':vxc_gap,'weights_max_abs_gap':weights_gap,
                                    'legacy_coordinate_rows_exact':bool(coordinates_equal),'status':'PASS'})
    del legacy_rows,legacy_index
    with (ROOT/(PREFIX+'legacy_source_parity.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(source_preservation[0]))
        writer.writeheader();writer.writerows(source_preservation)
    sources=[legacy,downloads/'extrapolation_e.csv',downloads/'MRKS/bin/gen_h5_with_vrho.py',
             downloads/'MRKS/bin/convert_vxc_data_with_wfn.py',downloads/'MRKS/bin/readWFN.py',
             ROOT.parent/'mrks_source_audit/check_energy_definition.py',
             ROOT.parent/'readWFN_share_ms/mrks_pbe0_d3bj.json',
             ROOT/'train_models/dispersions/dispersions_mrks.pickle']
    assert json.loads((ROOT.parent/'readWFN_share_ms/mrks_pbe0_d3bj.json').read_text())==json.loads((DATA/'mrks/dispersion.json').read_text())
    conversion=json.loads((DATA/'provenance/mrks_conversion.json').read_text())
    expected={r['file']:r['sha256'] for r in conversion['files']}
    publication=[json.loads(line) for line in (DATA/'mrks/systems.jsonl').read_text().splitlines()]
    current_pilot={}
    for mapping in publication:
        if mapping['source_id'] not in protocol['pilot']:
            continue
        path=DATA/mapping['shard']; actual=sha(path)
        assert actual==expected[mapping['shard']]
        with h5py.File(path,'r') as handle:
            group=handle[mapping['group']]
            current_pilot[mapping['source_id']]={'file_sha256':actual,'source_id':mapping['source_id'],
                'source_metadata_sha256':hashlib.sha256(group.attrs['source_metadata_json'].encode()).hexdigest(),
                'phi_dtype':str(group['phi'].dtype),'grad_phi_dtype':str(group['grad_phi'].dtype),'lap_phi_dtype':str(group['lap_phi'].dtype)}
    npz_info=[]
    for name in protocol['pilot']:
        path=downloads/'mrks_90_ccsd_pt/mrks_90_ccsd_pt'/name/'inp_mrks.npz'
        with np.load(path,allow_pickle=False) as z:
            meta=dict(zip(z['meta_keys'].tolist(),z['meta'].tolist()))
            homo=float(np.max(z['mo_energy'][z['mo_occ']>0]))
            npz_info.append({'system':name,'HOMO_ha':homo,'i_EKT_metadata_ha':meta['i_ekt'],
                             'HOMO_plus_i_EKT_ha':homo+meta['i_ekt'],'converged':meta['converged'],
                             'delta_rho_metadata':meta['delta_rho'],'delta_evir_metadata':meta['delta_evir'],
                             'gauge_interpretation':'numeric HOMO alignment evidence only; original exporter/gauge specification missing'})
    write('primary_source_availability',{
        'checked_locations':['current repository source and git history','Downloads/MRKS/bin exporters',
                             'Downloads/mrks_90_ccsd_pt frozen NPZ','workspace mrks_source_audit scripts',
                             'WSL PySCF 2.14.0/Libxc 7.0.0 installed sources'],
        'external_sources_sha256':{str(p):sha(p) for p in sources},
        'legacy_numeric_source_preservation':'PASS all90 E_xc/Vxc/weights/coordinates; raw reference-data pickle only',
        'original_mRKS_inversion_exporter':'NOT FOUND in checked locations',
        'original_legacy_E_xc_exporter':'NOT FOUND; available prepare_vxc only copies E_xc',
        'current_publication_pilot_file_integrity':current_pilot,
        'potential_numeric_gauge_evidence':npz_info,
        'dispersion_parameters':'PBE0/d3bj requested by producer; D3 package/parameter table and generating H5 geometry receipt unavailable',
        'oracle_density_conventions':'NumInt CUTOFF=1e-15; Libxc cache density_threshold=None (compiled defaults); no override. Original Libxc per-functional compiled threshold not independently extracted.',
        'historical_documentation_conflict':'tools/build_publication_dataset.py:127 calls potentials gauge-fixed; central builder records no gauge shift, with unresolved original gauge. Preservation does not qualify gauge.',
    })
    checks=rows('pbe_oracle_checks'); metrics=[r for r in rows('metric_gauge_checks') if r['system'] in protocol['pilot']]
    reconstruction=rows('energy_reconstruction_checks')
    text=[
        '# Independent XC operator and target-contract audit',
        '',
        'Verdict: TARGET_PROVENANCE_UNRESOLVED + NUMERICAL_OR_METRIC_SENSITIVITY_IDENTIFIED. No production energy/operator/loss implementation bug demonstrated within tested scope. This is not a proof that the targets are physically correct.',
        '',
        f"Frozen HEAD: `{protocol['head']}`; branch `{protocol['branch']}`. Versions: `{protocol['versions']}`. Initial worktree, hashes, tolerances and source/array receipts are saved in protocol/provenance. No trained checkpoints or optimizer states were loaded. Only H2, BeH2, CO received molecular numerical checks; all90 census was hashes/metadata/scalars/dispersion only.",
        '',
        '## Target chain and limits',
        '',
        'Primary inversion/export code was not found in the bounded checked locations. NPZ → legacy scalar potential row mapping → central corpus → current publication corpus is authenticated. The legacy pickle hash matches the builder; all90 NPZ and central-file hashes match, and current publication pilot file hashes match conversion receipts. Geometry/basis/AO ordering, total RKS dm_ks, restored NPZ coordinates, signed legacy weights and row selection are retained.',
        '',
        'Current E_xc is the legacy F32 value widened to F64, in Hartree. The original mathematical definition remains TARGET_DEFINITION_UNRESOLVED. Builder [target preservation](train_models/build_lap_operator_data.py:229) explicitly says “original formula unresolved; NPZ exc_wf not substituted”. [prepare_vxc](train_models/prepare_vxc.py:49) copies an H5 E_xc, it does not establish its formula.',
        '',
        'On the three pilots, the recipe E_CBS_1 − E_nuc − Tr(P_KS T) − Tr(P_KS V_ext) − 0.5 Tr(P_KS J_KS) reproduces legacy E_xc at F32 source resolution. This is numerical evidence for a total-energy-residual recipe, not authentication of its original exporter or of the CBS energy calculation. Nuclear energy is explicitly excluded before the electronic decomposition. The resulting scalar is not assumed to equal ∫n v_xc.',
        '',
        '| System | Recipe − legacy, Ha | ∫|n_KS−n_WF|, electrons |',
        '|---|---:|---:|',
    ]
    for r in reconstruction:
        text.append(f"| {r['system']} | {float(r['recipe_minus_legacy_ha']):.3g} | {float(r['integrated_abs_rho_ks_minus_wf']):.6g} |")
    text += [
        '',
        'NPZ exc_wf equals 0.5∫n_WF v_xc_hole numerically on the pilots (zero stored-value discrepancy). This does not by itself authenticate the exporter definition; an interaction XC energy W_xc must not be substituted for full E_xc=W_xc+T_c. Stored t_wf and computed T_s use different WF/KS densities; density differences are nonzero. Their difference cannot be asserted as an exact same-density identity here.',
        '',
        'Legacy Vrho matches NPZ vxc_grid at the original row identities, with only F32 preservation error. A discovered gen_h5_with_vrho exporter uses local Libxc vrho; it is NOT authenticated as the producer of these NPZ/legacy records and is not evidence that the current target is a partial derivative. Full common-RKS potential semantics and original gauge remain UNRESOLVED. Numeric HOMO/IP metadata evidence is recorded separately. No stationary-condition check was run because original F_ref Hartree/gauge conventions are not authenticated.',
        '',
        '## Dispersion',
        '',
        'The active path is [integrated_energy](train_models/lap_vxc.py:326) → [make_mrks_objective_factories](train_models/lap_moo_training.py:685) → [_add_mrks_dispersion_once](train_models/optuna_joint.py:418) → [batch_exc](train_models/optuna_joint.py:806). It adds one exact Name-key D scalar in Ha. All90 keys exist; missing keys would silently add zero; enabled with no dictionary raises. batch_exc converts singleton absolute/RMSE error to kcal/mol once. Effective pure-NN target is E_target−D; every scalar and loss shift is in dispersion_contract.csv.',
        '',
        'The current dispersion JSON exactly matches the existing mrks_pbe0_d3bj JSON. Its [producer](train_models/compute_mrks_pbe0_d3bj.py:89) requests xc=PBE0, version=d3bj, Bohr geometry and no SCF. Original generating H5/geometry receipt and installed D3 parameter table were not available. Arithmetic/name/unit contract PASS; physical compatibility with unresolved E_xc and exact generating geometry/parameters UNRESOLVED. Geometry-only D has dD/dP=0; its absence from the AO operator is not a defect. No labels or corrections were changed.',
        '',
        '## Independent PBE',
        '',
        'Oracle: PySCF NumInt.nr_rks(PBE) at exact original dm_ks, coords and weights, no SCF/grid regeneration. nr_uks with dm_alpha=dm_beta=dm_total/2 agrees with RKS. Oracle never calls repository PBE formula/local-partial/AO-assembly helpers; AO evaluation library and source inputs are shared. The production canonical adapter explicitly emits PBE constants (Gx=0, Gc=1); zero NN weights are not used.',
        '',
        '| System | Repo F64 energy error, Ha | Repo F64 orthogonal-matrix relative error | Frozen gate | Mixed gate | Ideal F64 relative error |',
        '|---|---:|---:|---|---|---:|',
    ]
    for name in protocol['pilot']:
        f=next(r for r in checks if r['system']==name and r['path']=='repository_f64')
        m=next(r for r in checks if r['system']==name and r['path']=='repository_mixed')
        i=next(r for r in checks if r['system']==name and r['path']=='ideal_f64')
        text.append(f"| {name} | {float(f['energy_abs_error_ha']):.6g} | {float(f['orth_relative']):.6g} | {f['status']} | {m['status']} | {float(i['orth_relative']):.3g} |")
    text += [
        '',
        'Original independent repository-F64 gate remains FAIL for H2/BeH2; it was not widened. Ideal F64 unpolarized PBE/PW_MOD agrees to machine precision; it is a diagnostic different numerical functional. Production rs adds 1e-7 to total density; xs adds 1e-10 to density and its 8/3 power to sigma; constants are originally F32 and PW92 unpolarized A is rounded to 0.031091 rather than 0.0310907. Production z_thr is a no-op. The isolated source-copy comparison shows that precise constants alone barely reduce the matrix discrepancy; removing density/sigma additions as well reduces it to ~1e-13 relative. Thus conventions, predominantly regularization, explain this oracle discrepancy; widening F32 source values alone does not recover exact PBE. All counterfactuals leave production source untouched.',
        '',
        '## Variational assembly and metric',
        '',
        'Nonzero rho/sigma/Laplacian polynomial: independent symmetric-DM energy derivative versus production weak AO assembly, scalar operator-loss parameter gradient and full symmetric δE=Tr(VδP) contraction PASS. A fresh untrained R0 fixture passes density-chain checks; the separate full-9446-coordinate Gc-enabled regression compares every operator-loss parameter derivative. Initial default-constructor fixture had Gc disabled (9381 coordinates); no model quality was evaluated. Three deliberately wrong copies (extra spin factor, missing Laplacian AO terms, wrong sigma mapping) were detected. Three Laplacian product terms are separately regression-tested.',
        '',
        'Original audit fixture FAIL is preserved in variational_checks.csv. It differentiated a graph assuming symmetric DM using unconstrained independent entries, producing an irrelevant antisymmetric derivative. Explicit symmetric leaf parameterization fixes the diagnostic graph; variational_corrected_checks.csv contains the corrected results. This was an audit defect, not a production bug. Original tests and tolerance limits were not weakened.',
        '',
        'Independent real reference projection uses NumPy contraction ∑w v φφ, not the builder helper. RefAO relative gaps are ~4–6e-17; chunk137 agrees. Constant-field fixture is cQ, not assumed cS. operator_loss self-loss/gradient, known shift, independent 2 S^-1 ΔV S^-1/n gradient, and well-conditioned same-basis-span invariance PASS. No overlap modes were discarded.',
        '',
        '| System | cond(S) | ||XQX−I||₂ | PBE−mRKS loss, Ha² | trace component fraction | discrete-Q component fraction |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in metrics:
        total=float(r['loss_ha2'])
        text.append(f"| {r['system']} | {float(r['overlap_condition']):.1f} | {float(r['XQX_minus_I_spectral']):.6g} | {total:.9f} | {float(r['trace_constant_loss'])/total:.4%} | {float(r['discrete_constant_loss'])/total:.4%} |")
    text += [
        '',
        'All overlaps pass the existing numerical SPD gate. BeH2 quadrature is notably incomplete in some AO directions despite good electron-number checks. The exact cI and c_Q XQX decompositions close to ~1e-17. Trace dominance is a property of this PBE−mRKS residual, not proof of an arbitrary target gauge defect. Q≠S prevents automatic interpretation of cI as a constant grid-potential shift. No c or c_Q was applied to targets.',
        '',
        'Occupied/virtual decomposition uses authenticated saved orbitals/occupations and verified CᵀSC≈I (largest Frobenius gap ~2.1e-12). Most loss is virtual–virtual on this pilot; this is disclosed, not removed or declared physically meaningless. Source-F64 versus preserved-F32 vxc projection errors are only ~1e-8–9e-8 orthogonal Frobenius on the pilot. This does not bound arbitrary basis expansions or explain trained-model plateaus; trained weights were not read.',
        '',
        '## Decision and next step',
        '',
        'No confirmed production implementation bug or authenticated target-contract mismatch. The energy formula and potential gauge/semantics are incompletely qualified; significant quadrature/metric structure and small regularization-driven PBE mismatch are measured. The single next justified task is to recover the original mRKS inversion/export source plus legacy E_xc construction and D3 geometry/parameter receipts, then authenticate the target energy/gauge contract. Do not change operator loss, gauge, energy penalty or training before that evidence is resolved.',
        '',
        'No new molecular systems, training, SCF, trained model or optimizer reads, development/test access, production repairs or target/loss changes. No conclusion is made about the effectiveness of physics→chemistry training.',
    ]
    wall=time.time()-protocol['wall_origin_unix']
    assert wall<protocol['budget_seconds']
    text += ['',f'Measured scientific workflow elapsed since protocol freeze: {wall:.2f} s / 10800 s. Instrumented initial computation: {provenance["active_seconds"]:.2f} s; supplement: {read("supplement_receipt")["active_seconds"]:.2f} s. Preliminary source inspection preceded protocol freeze and was not separately timed; no molecular computation occurred before freeze. Metadata/hash/report/test work is not claimed as GPU compute.']
    (ROOT/(PREFIX+'report.md')).write_text('\n'.join(text)+'\n',encoding='utf-8')
    decision['scope_limits']=['original inversion/export and original gauge not authenticated','D3 parameter/geometry receipt missing','repository-F64 oracle gate fails H2/BeH2; regularization sensitivity measured','no trained model conclusions','no SCF stationary check']
    write('decision',decision)
    write('final_integrity',{'sources_unchanged':True,'publication_pilot_hashes':current_pilot,
        'outputs_sha256':{p.name:sha(p) for p in ROOT.glob(PREFIX+'*') if p.is_file() and p.name!=PREFIX+'final_integrity.json'},
        'scientific_wall_seconds':wall,'budget_respected':True,
        'trained_checkpoints_loaded':0,'optimizer_states_loaded':0,'training_updates':0})


if __name__=='__main__':
    main()
