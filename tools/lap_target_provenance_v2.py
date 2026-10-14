"""CBS arithmetic and exact-name scalar linkage. No electronic/model evaluation."""

import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREFIX='lap_target_provenance_v2_'
CSV_PATH=Path('C:/Users/schne/Downloads/extrapolation_e (1).csv')
EXPECTED='4040c5ddd21cb73edcf6f54356114ff2cf216fe47346e227c5ca83180fde7e4e'
COLUMNS=('System','E_CBS_1','E_CBS_2','delta_formula','E_CBS_corr','delta_1','delta_2',
         'E_scf_5Z','E_scf_QZ','E_corr_5Z','E_corr_QZ')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def indexed_csv(path,key='System',schema=None):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if schema is not None and tuple(reader.fieldnames)!=tuple(schema):
            raise ValueError('Unexpected CSV schema')
        rows={}
        for row in reader:
            identity=row[key]
            if not identity or identity in rows:
                raise ValueError('Missing or duplicate exact identity')
            rows[identity]=row
    return rows


def literal_cbs(s5,s4,c5,c4):
    corr=(125*c5-64*c4)/61
    a=5*math.exp(-18)
    b=6*math.exp(-9*math.sqrt(5))
    scf1=(a*s5-b*s4)/(a-b)
    a2=math.exp(-1.63*5)
    b2=math.exp(-1.63*4)
    scf2=(a2*s5-b2*s4)/(a2-b2)
    model2=s5+(s5-s4)/(math.exp(1.63)-1)
    return {'E_CBS_1':scf1+corr,'E_CBS_2':scf2+corr,
            'E_CBS_corr':s5+corr,'model_E_CBS_2':model2+corr,
            'e_inf_corr':corr,'literal_scf2':scf2,'model_scf2':model2}


def f32_after_subtraction(cbs,noxc):
    return float(np.float32(float(cbs)-float(noxc)))


def require_exact_identities(*populations):
    if any(set(population)!=set(populations[0]) for population in populations[1:]):
        raise ValueError('Source populations do not have identical exact identities')


def check_link(cbs,row,legacy):
    noxc=float(row['E_no_xc'])
    logged_cbs=float(row['E_cbs_1'])
    logged_xc=float(row['E_xc'])
    scale=max(1.,abs(cbs),abs(logged_cbs),abs(noxc))
    cbs_bound=.5e-12+8*np.finfo(float).eps*scale
    residual_bound=1e-10+16*np.finfo(float).eps*scale
    predicted=f32_after_subtraction(cbs,noxc)
    return {'logged_E_CBS1_ha':logged_cbs,'CSV_minus_logged_CBS1_ha':cbs-logged_cbs,
            'CBS_log_bound_ha':cbs_bound,'logged_noXC_ha':noxc,'logged_H5_XC_ha':logged_xc,
            'F64_residual_ha':cbs-noxc,'H5_residual_gap_ha':cbs-noxc-logged_xc,
            'residual_bound_ha':residual_bound,'F32_after_subtraction_ha':predicted,
            'legacy_F32_ha':legacy,'legacy_gap_ha':predicted-legacy,
            'status':'PASS' if abs(cbs-logged_cbs)<=cbs_bound and
                    abs(cbs-noxc-logged_xc)<=residual_bound and predicted==legacy else 'MISMATCH'}


def table(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/(PREFIX+name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def write(name,value):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def main():
    evidence=read(ROOT/(PREFIX+'evidence.json'))
    start=evidence['first_action_unix']
    assert time.time()-start<3600
    assert sha(CSV_PATH)==EXPECTED
    old_csv=Path('C:/Users/schne/Downloads/extrapolation_e.csv')
    assert sha(old_csv)==EXPECTED
    assert all(sha(ROOT/p)==s for p,s in evidence['historical_artifacts'].items())
    rows=indexed_csv(CSV_PATH,schema=COLUMNS)
    assert len(rows)==90
    for row in rows.values():
        if not all(math.isfinite(float(row[c])) for c in COLUMNS[1:]):
            raise ValueError('Nonfinite CBS source')
    assert float(rows['H2']['E_CBS_1'])==-1.1745389973344755
    assert float(rows['BeH2']['E_CBS_1'])==-15.903417721361238
    assert float(rows['CO']['E_CBS_1'])==-113.326472447679
    external=ROOT.parent/'lap_target_provenance_v2_materials_20261014'
    external.mkdir(exist_ok=True)
    destination=external/CSV_PATH.name
    if destination.exists():
        assert sha(destination)==EXPECTED
    else:
        shutil.copyfile(CSV_PATH,destination)
    replay=[]
    for name,row in rows.items():
        s5,s4,c5,c4=(float(row[k]) for k in ('E_scf_5Z','E_scf_QZ','E_corr_5Z','E_corr_QZ'))
        result=literal_cbs(s5,s4,c5,c4)
        scale=max(1.,abs(s5),abs(s4),abs(float(row['E_CBS_1'])))
        bound=1e-10+32*np.finfo(float).eps*scale
        delta_bound=1e-11+64*np.finfo(float).eps*scale
        entry={'System':name,'energy_bound_ha':bound,'delta_bound_ha':delta_bound}
        for column in ('E_CBS_1','E_CBS_2','E_CBS_corr'):
            gap=result[column]-float(row[column])
            entry.update({column+'_stored':float(row[column]),column+'_literal':result[column],
                          column+'_gap':gap,column+'_status':'PASS' if abs(gap)<=bound else 'FAIL'})
        delta_values={'delta_formula':result['E_CBS_2']-result['E_CBS_1'],
                      'delta_1':result['E_CBS_corr']-result['E_CBS_1'],
                      'delta_2':result['E_CBS_2']-result['E_CBS_corr']}
        for column,value in delta_values.items():
            gap=value-float(row[column])
            entry.update({column+'_gap':gap,column+'_signed_difference':value,
                          column+'_status':'PASS' if abs(gap)<=delta_bound else 'FAIL'})
        entry['model_E_CBS_2']=result['model_E_CBS_2']
        entry['literal_CBS2_minus_exponential_model']=result['E_CBS_2']-result['model_E_CBS_2']
        entry['CBS2_model_identity_status']='PASS' if abs(entry['literal_CBS2_minus_exponential_model'])<=bound else 'FAIL'
        replay.append(entry)
    table('cbs_replay',replay)
    sanity_path=ROOT.parent/'readWFN_share_ms/h5_vrho_from_mrks_sanity.csv'
    sanity=indexed_csv(sanity_path,'file')
    if any(not name.endswith('.h5') for name in sanity):
        raise ValueError('Unexpected historical H5 identity suffix')
    sanity={name[:-3]:row for name,row in sanity.items()}
    targets=indexed_csv(ROOT/'lap_operator_contract_v1_target_definitions.csv','system')
    data=ROOT.parent/'publication_dataset_v1'
    systems={r['source_id']:r for r in (json.loads(line) for line in (data/'mrks/systems.jsonl').read_text().splitlines())}
    require_exact_identities(rows,sanity,targets,systems)
    pilot_receipts=indexed_csv(ROOT/'lap_operator_contract_v1_energy_reconstruction_checks.csv','system')
    linkage=[]
    for name,row in rows.items():
        legacy=float(targets[name]['legacy_value_ha'])
        entry={'System':name,'selected_column_evidence':'author E_CBS_1 declaration + historical E_cbs_1 log scalar',
               'CSV_E_CBS1_ha':float(row['E_CBS_1']),'noXC_source':'independent existing sanity-log field, not reconstructed from target',
               'noXC_DM_validation':'PILOT_INDEPENDENT_COMPONENT_RECEIPT' if name in pilot_receipts else 'LOG_ONLY',
               'original_H5_availability':'sanity log survives; original H5 bytes unavailable'}
        entry.update(check_link(float(row['E_CBS_1']),sanity[name],legacy))
        separately_rounded=float(np.float32(float(row['E_CBS_1']))-np.float32(float(sanity[name]['E_no_xc'])))
        entry['separately_rounded_F32_operands_ha']=separately_rounded
        entry['separately_rounded_operand_gap_ha']=separately_rounded-legacy
        for alt in ('E_CBS_2','E_CBS_corr'):
            alt_cbs=float(row[alt])
            entry[alt+'_minus_logged_CBS_ha']=alt_cbs-float(sanity[name]['E_cbs_1'])
            entry[alt+'_F32_target_equal']=f32_after_subtraction(alt_cbs,float(sanity[name]['E_no_xc']))==legacy
        mapping=systems[name]
        with h5py.File(data/mapping['shard'],'r') as handle:
            group=handle[mapping['group']]
            metadata=json.loads(group.attrs['source_metadata_json'])
            assert metadata['system_name']==name
            current=float(group['Exc'][()])
        entry['current_corpus_XC_ha']=current
        entry['current_minus_legacy_ha']=current-legacy
        entry['current_corpus_status']='PASS' if current==legacy else 'MISMATCH'
        entry['current_record_id']=mapping['id']
        if name in pilot_receipts:
            p=pilot_receipts[name]
            components=[float(p[k]) for k in ('Enuc_ha','Ts_ks_ha','Vext_ks_ha','EH_ks_ha')]
            independent=sum(components)
            bound=1e-10+16*np.finfo(float).eps*sum(abs(v) for v in components)
            gap=independent-float(sanity[name]['E_no_xc'])
            entry.update({'independent_total_noXC_ha':independent,'independent_noXC_gap_ha':gap,
                          'independent_noXC_bound_ha':bound,'independent_noXC_status':'PASS' if abs(gap)<=bound else 'FAIL',
                          'noXC_terms':'Enuc+Tr(P T)+Tr(P Vext)+0.5Tr(P J(P)); no exchange/XC/dispersion'})
        linkage.append(entry)
    table('target_linkage',linkage)
    replay_pass=all(r[c+'_status']=='PASS' for r in replay for c in ('E_CBS_1','E_CBS_2','E_CBS_corr','delta_formula','delta_1','delta_2'))
    link_pass=all(r['status']=='PASS' and r['current_corpus_status']=='PASS' for r in linkage)
    pilot_pass=all(r.get('independent_noXC_status','PASS')=='PASS' for r in linkage)
    updates=[
        {'topic':'energy recipe','before':'primary recipe unresolved','new':'user text transmitting supplier recipe E_CBS_1−E_noXC[loaded DM], no SCF','after':'DOCUMENTED_BY_PRIMARY_CORRESPONDENCE','limitation':'original screenshot not available; not exact universal E_xc[n_KS]'},
        {'topic':'CBS column choice','before':'numeric support, author selection unavailable','new':'explicit supplied recommendation E_CBS_1; CSV hash identical to historical source','after':'E_CBS_1_DOCUMENTED','limitation':'author message available as user text, not original image'},
        {'topic':'CBS arithmetic','before':'extrapolation formulas unresolved','new':'literal user-transmitted formulas all90 replay','after':'PASS' if replay_pass else 'FAIL','limitation':'original formula image absent; model-consistency check of alternative CBS2 separate'},
        {'topic':'actual target linkage','before':'pilot residual/log rounding verified','new':'all90 selected CBS scalar/log noXC/H5-log residual/F32/current corpus; independent noXC components on pilot','after':'ALL90_VERIFIED' if link_pass and pilot_pass else 'MISMATCH','limitation':'all90 scalar/log linkage; independent DM noXC qualification only H2/BeH2/CO; original H5/export bytes missing'},
        {'topic':'potential declared semantics','before':'unresolved primary label','new':'supplier via user describes final grid XC potential and converged KS dm_ks','after':'FINAL_XC_POTENTIAL_DOCUMENTED','limitation':'original screenshots absent; no independent inversion implementation qualification'},
        {'topic':'gauge','before':'UNRESOLVED','new':'no gauge rule supplied','after':'UNRESOLVED','limitation':'HOMO/i_ekt approximate alignment not a gauge proof'},
        {'topic':'method/basis','before':'UNRESOLVED','new':'basis metadata exists, SCF/correlation column method and exact family not specified','after':'PARTIAL','limitation':'numerical AO basis/geometry metadata authenticated previously; electronic method/CBS calculation pedigree unresolved'},
        {'topic':'D3 intended convention','before':'UNRESOLVED','new':'new correspondence says nothing about pipeline-added D','after':'UNRESOLVED','limitation':'ask source of D3 addition, not presume mRKS supplier responsibility'},
    ]
    status={'updates':updates,'new_primary_evidence_incorporated':'PARTIAL','energy_recipe_documented':True,
            'final_potential_semantics_documented':True,'source_level':'USER_TEXT_TRANSMISSION; original screenshot/formula image missing',
            'CBS_1_replay':'PASS' if replay_pass else 'FAIL','actual_CBS_1_usage':'ALL90' if link_pass else 'MISMATCH',
            'XC_target_linkage':'ALL90' if link_pass and pilot_pass else 'MISMATCH',
            'independent_fixed_DM_noXC_coverage':'PILOT (3 systems), reused authenticated component receipts',
            'alternative_CBS2_model_consistency':'PASS' if all(r['CBS2_model_identity_status']=='PASS' for r in replay) else 'FAIL',
            'alternative_CBS2_affects_used_targets':'NO' if link_pass else 'UNKNOWN',
            'potential_gauge':'UNRESOLVED','method_basis':'PARTIAL','D3_convention':'UNRESOLVED'}
    write('status_update',status)
    evidence.update({'driver_sha256':sha(__file__),'raw_materials_outside_git':{'CSV_path':str(destination),'CSV_sha256':sha(destination),'private_image_paths':[]},
                     'CSV_byte_identity_with_historical_source':True,'CSV_row_order_and_numeric_text_identity':CSV_PATH.read_bytes()==old_csv.read_bytes(),
                     'csv_count':90,'exact_identities':90,'source_schema':list(COLUMNS),
                     'sanity_log_sha256':sha(sanity_path),'current_systems_manifest_sha256':sha(data/'mrks/systems.jsonl'),
                     'CBS_corr_mapping':'source e_cbs_none -> existing E_CBS_corr column; no column rename',
                     'delta_conventions':'numeric signed differences: formula=CBS2−CBS1; delta1=CBS_corr−CBS1; delta2=CBS2−CBS_corr. All differences positive here, so this alone cannot distinguish an undocumented abs convention.',
                     'max_CBS1_abs_gap_ha':max(abs(r['E_CBS_1_gap']) for r in replay),
                     'max_CBS2_abs_gap_ha':max(abs(r['E_CBS_2_gap']) for r in replay),
                     'max_CBS_corr_abs_gap_ha':max(abs(r['E_CBS_corr_gap']) for r in replay),
                     'CBS2_model_fail_count':sum(r['CBS2_model_identity_status']=='FAIL' for r in replay),
                     'max_CBS2_model_discrepancy_ha':max(abs(r['literal_CBS2_minus_exponential_model']) for r in replay),
                     'separate_F32_operand_rounding_mismatch_count':sum(r['separately_rounded_operand_gap_ha']!=0 for r in linkage),
                     'unresolved_selected_column_records':[r['System'] for r in linkage if r['status']!='PASS'],
                     'noXC_reconstruction_is_not_circular':'all90 consumes saved noXC log; pilot independently sums archived DM-derived components. Never computes noXC=CBS−target.',
                     'historical_lineage_remaining':'original screenshot bytes, original H5/export producer and all90 independently computed noXC provenance not fully recovered',
                     'new_NN_AO_SCF_evaluations':0,'wall_seconds':time.time()-start,'budget_respected':time.time()-start<3600})
    write('evidence',evidence)
    (ROOT/(PREFIX+'remaining_questions.md')).write_text('''# Short remaining request — DRAFT, NOT SENT

To the original mRKS/reference supplier:
1. What rule fixes the additive zero of saved vxc_grid, at which export stage, and are mo_energy shifted consistently? Please provide the relevant assignment/config or concise primary description.
2. What precise electronic method and basis/extrapolation details generated E_scf_5Z, E_scf_QZ, E_corr_5Z, E_corr_QZ? Stored AO basis metadata is already available, but it does not identify these source calculations.

To the source of the D3 addition in the training pipeline:
3. Was fitting NN integral to E_target−D intentional? Please provide the original full-XC/residual convention, and the original geometry/backend/parameters receipt if available.

The recipe E_CBS_1−E_noXC on the loaded fixed KS DM has already been explained and is not requested again. Original correspondence screenshots and the readable formula image are missing from this session; supplying them would authenticate the source excerpts already transmitted in the user message. No messages are sent.
''',encoding='utf-8')
    lines=['# Primary CBS/XC evidence addendum',
           '', 'New evidence incorporated: PARTIAL. CSV original is authenticated; original private screenshots/formula image are unavailable. Definitions/formulas are incorporated from the user’s explicit transmission of supplier correspondence, not claimed as an image reading. No message dates, software versions or method labels are invented.',
           '',f'CSV: `{CSV_PATH}`; SHA256 `{EXPECTED}`. It is byte-identical to the historical CSV, including row order and numeric formatting. All90 exact System names, schema and finite values pass. A raw copy is retained outside Git; no private screenshots are committed.',
           '', '| Topic | BEFORE | NEW EVIDENCE | AFTER |', '|---|---|---|---|']
    lines.extend(f"| {r['topic']} | {r['before']} | {r['new']} | {r['after']} |" for r in updates)
    lines += ['',f'CBS_1 max absolute replay gap: {evidence["max_CBS1_abs_gap_ha"]:.6g} Ha; CBS_2: {evidence["max_CBS2_abs_gap_ha"]:.6g}; CBS_corr: {evidence["max_CBS_corr_abs_gap_ha"]:.6g}. Binary64 operations follow the written formulas literally; predeclared bounds remain unchanged. e_cbs_none maps numerically to E_CBS_corr without renaming source columns. Delta columns reproduce the stated signed differences; original intent between signed/abs is not independently inferred when all differences are positive.',
              '',f'Alternative CBS_2: literal formula reproduces its CSV column, but differs from the two-point exponential-model solution on {evidence["CBS2_model_fail_count"]}/90 rows; maximum difference {evidence["max_CBS2_model_discrepancy_ha"]:.6g} Ha. For E_X=E_inf+A exp(−1.63X), E_inf=E5+(E5−E4)/(exp(1.63)−1). This is a separate algebraic check, not a correction to the supplied formula. CBS_2 was not used in the verified scalar target chain; no labels or CSV values changed.',
              '',f'All90 linkage status: {status["XC_target_linkage"]}. Exact identity mapping joins CSV System to historical {{System}}.h5 rows and current corpus source_id. E_cbs_1 log values agree with CSV_1, logged E_noXC is consumed directly (not obtained from target), F64 subtraction then one F32 storage reproduces legacy, and current Exc equals that legacy value exactly. Original H5 bytes/export script are still absent; this is authenticated scalar/log linkage, not full reconstruction of all90 electronic calculations.',
              '', 'H2/BeH2/CO additionally reuse independently computed E_nuc+Tr(P T)+Tr(P V_ext)+0.5Tr(P J(P)) receipts and agree with logged noXC. Those stored components use the original fixed KS DM, geometry/basis/AO order and include nuclei; no exchange/XC/dispersion is added. This independent physical noXC coverage is PILOT, not all90. No SCF, inversion, new density/grid or matrix census was run. Separately rounding CBS and noXC to F32 is not the qualified storage order.',
              '', 'Target wording: “XC-energy reference, constructed as a CBS total-energy residual after subtraction of the no-XC energy evaluated on the supplied fixed KS density matrix.” This is not a claim of exact universal E_xc[n_KS]. It does not replace targets with exc_wf or invent T_c across differing densities.',
              '', 'Supplier-declared field semantics (via user text): mo_coeff AO-basis coefficients; mo_energy Ha; mo_occ 0/2; vxc_grid final grid XC potential; dm_ks converged KS DM; grid_coords/atom_coords Bohr; grid_weights quadrature; serialized basis. The previously authenticated all90 manifest is consistent with downstream use of these fields. Recommendation to replace old systems is documented, but the named mrks_90_ccsd_pt.tar.zst archive bytes were not provided/authenticated here; its name does not prove CCSD or CCSD(T). Gauge and full inversion implementation qualification remain separate.',
              '', 'No confirmed target mismatch warrants a repair. Keep the previous repository-F64 PBE FAIL, operator/RefAO PASS and metric findings unchanged. Remaining request is limited to gauge, electronic method/basis pedigree, and intended D3 residual convention; original image availability is listed separately. D3 questions are addressed to the source of its training-pipeline addition, not attributed automatically to the mRKS supplier.',
              '', f'Wall at computation/report generation: {time.time()-start:.2f} s / 3600 s. Trained checkpoints/optimizer loads:0; training:0; SCF:NO; production/targets/loss changes:NO; messages:NO; Clean28/future-test access:NO.']
    (ROOT/(PREFIX+'report.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
