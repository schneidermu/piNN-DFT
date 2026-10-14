"""Bounded source recovery and scalar-log replay; no model or physics execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'lap_target_provenance_v1_'
DOWNLOADS = Path('C:/Users/schne/Downloads')
PILOT = ('H2', 'BeH2', 'CO')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_index(path, key):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        result = {}
        for row in csv.DictReader(handle):
            identity = row[key]
            if identity in result:
                raise ValueError(f'Duplicate source identity: {identity}')
            result[identity] = row
    return result


def replay_scalar(row, legacy):
    """Replay only the explicit historical log identity, not guessed CBS formulas."""
    xc, total, noxc = (float(row[k]) for k in ('E_xc', 'E_cbs_1', 'E_no_xc'))
    if not all(math.isfinite(v) for v in (xc, total, noxc, legacy)):
        raise ValueError('Nonfinite source scalar')
    recipe_gap = (total - noxc) - xc
    rounded = float(np.float32(xc))
    # The sanity CSV stores twelve decimal places. Three rounded operands
    # contribute <=1.5e-12 plus binary subtraction resolution; fixed before replay.
    bound = 1.5e-12 + 8 * np.finfo(float).eps * max(1., abs(total), abs(noxc))
    return {'logged_E_xc_ha': xc, 'logged_E_CBS1_ha': total,
            'logged_E_no_xc_ha': noxc, 'explicit_log_recipe_gap_ha': recipe_gap,
            'decimal_arithmetic_bound_ha': bound, 'rounded_F32_ha': rounded,
            'legacy_ha': legacy, 'rounding_gap_ha': rounded-legacy,
            'status': 'PASS' if abs(recipe_gap) <= bound and rounded == legacy else 'SOURCE_MISMATCH'}


def git(*args, binary=False):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=not binary)


def table(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/(PREFIX+name+'.csv')).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def document(name, text):
    (ROOT/(PREFIX+name+'.md')).write_text(text.strip()+'\n', encoding='utf-8')


def write(name, value):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    protocol = read_json(ROOT/(PREFIX+'protocol.json'))
    start = protocol['first_action_unix']
    assert time.time()-start < 7200
    assert git('rev-parse', 'HEAD').strip() == protocol['head']
    assert all(sha(ROOT/name) == digest for name,digest in protocol['previous_audit_receipts'].items())
    old_sources = read_json(ROOT/'lap_operator_contract_v1_protocol.json')['sources']
    assert all(sha(ROOT/name)==digest for name,digest in old_sources.items())
    sanity_path = ROOT.parent/'readWFN_share_ms/h5_vrho_from_mrks_sanity.csv'
    sanity = csv_index(sanity_path, 'file')
    previous = csv_index(ROOT/'lap_operator_contract_v1_energy_reconstruction_checks.csv', 'system')
    availability = read_json(ROOT/'lap_operator_contract_v1_primary_source_availability.json')
    replay=[]
    for name in PILOT:
        row={'system':name,'source':str(sanity_path),'source_sha256':sha(sanity_path),
             'check':'historical scalar-log recipe and F32 loader boundary',
             'evidence_level':'numerical support; original producer/config not authenticated'}
        row.update(replay_scalar(sanity[name+'.h5'], float(previous[name]['legacy_exc_ha'])))
        replay.append(row)
    assert all(r['status']=='PASS' for r in replay)
    table('replay',replay)
    d3_commit='4ed868e860d07b2d142d115008d25e2c7a8a3ca0'
    original_blob=git('show',d3_commit+':train_models/dispersions/dispersions_mrks.pickle',binary=True)
    historical_d3_sha=hashlib.sha256(original_blob).hexdigest()
    active_d3_sha=sha(ROOT/'train_models/dispersions/dispersions_mrks.pickle')
    assert historical_d3_sha==active_d3_sha
    assert active_d3_sha==availability['external_sources_sha256'][str(ROOT/'train_models/dispersions/dispersions_mrks.pickle')]
    archive=DOWNLOADS/'MRKS.zip'
    assert sha(archive)=='15a49f4517c35f2159f1e66d02a817945dbf1e1e66deb46f6e612d553890ad16'
    archive_sources=[]
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if name.endswith(('.py','.sh','.ipynb','.md')):
                raw=handle.read(name)
                archive_sources.append({'member':name,'sha256':hashlib.sha256(raw).hexdigest(),
                                        'producer_anchor_hits':[term for term in ['E_CBS_1','E_no_xc','E_xc','vxc_grid','inp_mrks.npz','i_ekt'] if term.encode() in raw]})
    search_rows=[
        {'location':'current repo and all locally available Git refs','query':'E_CBS_1 / E_no_xc / inp_mrks.npz / exc_wf / i_ekt; source/history only','found':'only downstream loaders/builders/audits; no original exporter','confirmed_transition':'H5 E_xc -> F32 legacy loader at 6f91e72; D3 artifact add at 4ed868e','stop_reason':'all local refs inspected, no tags; no primary-source path/link found'},
        {'location':'main / vxc_training / origin/vxc_training + backup/dbstrat-before-sync-20261009','query':'git grep/ls-tree/history and exact exporter anchors','found':'D3 utility/artifact shared; no NPZ/CBS producer','confirmed_transition':'same D3 consumer/source family','stop_reason':'no extra original exporter in these refs'},
        {'location':'existing vxc_training worktree at '+str(ROOT.parent/'readWFN_share_ms/piNN-DFT'),'query':'h5_vrho_from_mrks, E_no_xc, E_cbs_1 references','found':'untracked finetune_scf_mrks.py/scf_trial19_h2.py consumers and sanity CSV','confirmed_transition':'historical H5 schema E_cbs_1/E_no_xc/E_xc was consumed; scalar identity replay pilot','stop_reason':'consumers do not create energy fields; default H5 directories absent'},
        {'location':str(archive),'query':'source inventory and exact producer identifiers','found':f'{len(archive_sources)} py/sh files; no NPZ/CBS/gauge exporter anchors','confirmed_transition':'manifest archive hash authentic; discovered helpers do not establish current NPZ lineage','stop_reason':'archive has no primary mRKS/CBS producer; generic WFN utility literature links are not dataset source links'},
        {'location':str(DOWNLOADS/'vxc_data_2.tar.gz'),'query':'357-member inventory; first H5 schema only','found':'H5/WFN archive; sample has atom/charge/e_fci/grids/vxc and no E_xc/NPZ/config/gauge','confirmed_transition':'compatible with separate convert_vxc_data_with_wfn helper, not tied to current90','stop_reason':'no authenticated mapping to current names/hashes; no new numerical molecular comparison performed'},
        {'location':str(DOWNLOADS)+' (top-level notebooks only)','query':'source/markdown cells: E_CBS_1, exc_wf, t_wf, vxc_grid, inp_mrks.npz, meta_keys, i_ekt','found':'11 notebooks searched; no matching producer anchors; outputs not examined','confirmed_transition':'none','stop_reason':'no matching cells; unrelated notebook/data subdirectories not scanned'},
        {'location':str(DOWNLOADS/'mrks_90_ccsd_pt/mrks_90_ccsd_pt'),'query':'script/log/notebook/config inventory; pilot input_file metadata reused','found':'NPZ + digest records only; input_file points to /home/mmedvedev/lsvvt/chep/ready/{system}/inp.orca.zst','confirmed_transition':'frozen NPZ data identifiers, not cluster source availability','stop_reason':'remote ORCA input/output and exporter not locally provided'},
        {'location':'historical D3 --h5-dir defaults under the two documented PROJECT_ROOT choices','query':'existence of h5_vrho_from_mrks','found':'both documented local directories absent; sanity CSV and D3 JSON survive','confirmed_transition':'dictionary identity with initial committed artifact','stop_reason':'no geometry/config/backend receipt; no modern D3 replay can qualify historical version'},
    ]
    for row in search_rows:
        row['search_closed_at_elapsed_seconds']=round(time.time()-start,3)
    assert time.time()-start <= protocol['search_budget_seconds']
    table('search_log',search_rows)
    candidate_paths={
        'H5_to_legacy_loader':ROOT/'train_models/prepare_vxc.py',
        'central_preservation_builder':ROOT/'train_models/build_lap_operator_data.py',
        'D3_producer':ROOT/'train_models/compute_mrks_pbe0_d3bj.py',
        'current_energy_consumer':ROOT/'train_models/lap_moo_training.py',
        'legacy_energy_consumer':ROOT/'train_models/reaction_energy_calculation.py',
        'historical_H5_consumer':ROOT.parent/'readWFN_share_ms/piNN-DFT/train_models/finetune_scf_mrks.py',
        'H5_sanity_log':sanity_path,
        'CBS_scalar_table':DOWNLOADS/'extrapolation_e.csv',
        'archive_H5_export_helper':DOWNLOADS/'MRKS/bin/gen_h5_with_vrho.py',
        'archive_conversion_helper':DOWNLOADS/'MRKS/bin/convert_vxc_data_with_wfn.py',
    }
    evidence={'protocol_sha256':sha(ROOT/(PREFIX+'protocol.json')),
        'current_head':protocol['head'],'driver_sha256':sha(__file__),
        'candidate_sources':{key:{'path':str(path),'sha256':sha(path)} for key,path in candidate_paths.items()},
        'line_ranges':{'H5_to_legacy_loader':'19-64 (F32 energy cast at 52)','central_preservation_builder':'155-194,229-242',
                       'D3_producer':'21-38,54-92,96-117','current_energy_consumer':'685-704',
                       'legacy_energy_consumer':'153-188','historical_H5_consumer':'580-599,681-705',
                       'H5_sanity_log':'header and H2.h5/BeH2.h5/CO.h5 rows'},
        'git_source_commits':{'H5_F32_import':'6f91e72fb066cf894242beca60639d5a0f804873',
                              'D3_utility_created':'966ba34a8ceadafc12b5adb0a720c01bdec5d6dd','D3_artifact_and_optional_support':d3_commit},
        'D3_historical_blob_sha256':historical_d3_sha,'D3_active_blob_sha256':active_d3_sha,
        'D3_artifact_identity':'VERIFIED exact bytes; original calculation backend/config not authenticated',
        'MRKS_archive_sha256':sha(archive),'MRKS_archive_sources':archive_sources,
        'sanity_log_record_count':len(sanity),'sanity_log_lineage':'unbound surviving historical CSV; no producer/config hash',
        'metadata_gauge_evidence_reused':availability['potential_numeric_gauge_evidence'],
        'contracts':{
            'energy':{'SEMANTIC_DEFINITION':'UNRESOLVED','HISTORICAL_LINEAGE':'verified H5->legacy->corpus; original exporter and CBS producer unresolved','NUMERICAL_REPLAY':'PASS pilot log identity/F32 rounding; previous matrix recipe receipts reused','COVERAGE':'all90 downstream preservation from previous audit; pilot recipe only'},
            'potential':{'SEMANTIC_DEFINITION':'UNRESOLVED','HISTORICAL_LINEAGE':'verified NPZ values->legacy->RefAO; inversion/export/gauge unresolved','NUMERICAL_REPLAY':'previous pilot source projection and HOMO/i_ekt differences reused, no new potential evaluation','COVERAGE':'all90 preservation, pilot alignment evidence'},
            'dispersion':{'SEMANTIC_DEFINITION':'operational residual equation VERIFIED; author-intended full-XC/residual convention UNRESOLVED','HISTORICAL_LINEAGE':'artifact bytes and support commit VERIFIED; original geometry/backend run unresolved','NUMERICAL_REPLAY':'no new D3 evaluation; existing all90 arithmetic receipts reused','COVERAGE':'all90 artifact/consumer; original computation config unknown'}},
        'search_closed':True,'author_request_sent':False,'new_NN_or_AO_evaluations':0,
        'previous_audit_receipts_unchanged':True,'production_sources_unchanged':True}
    write('evidence',evidence)
    energy_text=f'''# Legacy XC energy contract

Status: UNRESOLVED (primary semantic definition); verified downstream numerical preservation.

Current target is the F32 legacy E_xc (Hartree), widened to F64 by the central builder. Direct source evidence: prepare_vxc.py:49–52 at commit 6f91e72fb066cf894242beca60639d5a0f804873 imports the stored H5 E_xc as torch.float32. It does not calculate it. Builder:155–165,229–242 preserves it and explicitly records the unresolved original formula. Source hashes are in evidence.json.

New source: `{sanity_path}` (SHA256 `{sha(sanity_path)}`), 90 named H5 rows. Columns explicitly contain E_xc, E_cbs_1 and E_no_xc; the pilot identity E_xc=E_cbs_1−E_no_xc closes within fixed decimal-source precision, and casting logged E_xc to F32 reproduces legacy exactly. This surviving log supports a historical residual recipe but has no authenticated generating script/config; it does not independently prove how E_no_xc or the CBS reference were calculated.

Reuse the prior three-system matrix receipt, not new AO work: E_CBS_1−E_nuc−Tr(P_KS T)−Tr(P_KS V_ext)−0.5Tr(P_KS J_KS) agrees with legacy at source precision. Separate components were generated in the previous audit on stored dm_ks. The reference CSV headers mention SCF 5Z/QZ and correlation 5Z/QZ values; neither the method nor extrapolation law, reference basis, charge/geometry binding or empirical corrections are authenticated by headers alone.

A (actually executed exporter): unavailable. B (author physical label): downstream consumers call it exact E_xc, which is not primary proof. C (strictly established): the stored target and importer rounding plus numerical residual-recipe support. D (approximations): finite basis/inversion and nonidentical WF/KS densities measured previously; a CBS scalar cannot be declared exact E_xc[n_KS] without matching reference construction/density contract.

exc_wf numerically equals half the rho_WF*vxc_hole integral on the previous pilot; t_wf is a stored scalar label. Their producer formulas remain unresolved. An interaction XC quantity W_xc is not full E_xc=W_xc+T_c. T_WF−T_s across different densities is not an exact same-density T_c identity. No target is replaced; E_xc=integral(n*v_xc) is not assumed.
'''
    document('energy_contract',energy_text)
    document('potential_gauge_contract','''# Potential and gauge contract

Status: potential semantics UNRESOLVED; GAUGE_UNRESOLVED.

Verified downstream: exact NPZ vxc_grid row identity -> legacy Vrho (F32) -> central common-scalar projection -> RefAO. All90 preservation and pilot independent projection receipts are reused. The consumer applies no new shift. This does not identify the original inversion functional/algorithm or determine whether a shift preceded NPZ export.

The authenticated MRKS.zip contains generic WFN/grid helpers, not a source creating inp_mrks.npz/vxc_grid. gen_h5_with_vrho.py:197–203 uses a Libxc local vrho, but its lineage to these current NPZ records is absent; it must not be substituted as their producer. convert_vxc_data_with_wfn.py:223–267 imports a separate scalar vxc H5 schema; no mapping to current90 is authenticated. The archive vxc_data_2 contains a sample HeH+ H5 with e_fci and vxc but no E_xc/config/gauge; only schema was inspected, not new molecular numerics.

NPZ input_file identifies remote ORCA paths /home/mmedvedev/lsvvt/chep/ready/{H2,BeH2,CO}/inp.orca.zst. No remote source access or author identity is inferred from that username. Original input/output, inversion code, export assignments, and gauge/convergence logs are missing.

Reused pilot HOMO+i_ekt residuals (Ha): H2 +5.898011167193573e-05; BeH2 −3.0889428246916317e-06; CO −1.0894252772397639e-04. These are approximate alignment evidence, not exact equality or proof of an applied vxc_grid shift. A rule tied to EKT/IP, asymptotic zero, or another convention remains unverified. Need the assignment modifying vxc_grid and whether orbital energies were shifted in the same export state.

Stored dm_ks is used as total closed-shell RKS density downstream; original WF/KS inversion semantics, filtering/masking/interpolation/tail treatment require primary source. The consumer's common-RKS use is verified; the upstream potential's physical meaning is not. No constant shift, target subtraction, cQ->cS substitution or metric change occurs.
''')
    document('dispersion_contract',f'''# Dispersion contract

Status: author-intended physical convention UNRESOLVED; operational code/artifact lineage VERIFIED.

New direct historical evidence: utility created at 966ba34a8ceadafc12b5adb0a720c01bdec5d6dd. Optional mRKS-D3 support and the active artifact were added together at {d3_commit}. The artifact at that commit has SHA256 `{historical_d3_sha}`, exactly equal to the active 90-name pickle; this was a binary hash comparison, not optimizer/model deserialization. Current JSON and dictionary equality/arithmetic are already verified in the prior audit.

Producer compute_mrks_pbe0_d3bj.py:54–92 loads H5 atom/charge/spin, assumes Bohr, constructs a geometry carrier, and requests dftd3.pyscf.DFTD3Dispersion(xc='PBE0',version='d3bj'). It does not explicitly set numerical damping parameters, backend version or optional extra terms. The default original H5 directories under documented PROJECT_ROOT choices are absent; the sanity CSV and JSON survive. No actual run command, input H5 geometry hash, backend version/parameter table or settings receipt was found. We do not substitute current D3 defaults or perform a modern replay as evidence of an old version.

Current objective mathematically uses prediction=NN_XC_integral+D and target=stored_E_xc, so the integral is fitted to stored_E_xc−D. In an additional historical H5 consumer, finetune_scf_mrks.py:681–705 explicitly returns E_no_xc+NN_XC_integral+D for comparison with E_cbs_1; this is consumer code, not an authenticated generating run or primary author declaration.

A: whether E_target was intended as pure full ab initio XC is unresolved. B: the operative integral is residual under D addition, verified in code. C: optional inclusion is documented by the source commit/functions, but intended full-XC/residual semantics are not explicitly authenticated. D: current pipeline agrees with that operative equation; agreement with the author's original physical convention cannot be certified. No mismatch is asserted merely from adding D to an ab initio target. Geometry-only D has zero density-matrix derivative, so no D term in the AO operator is not an independent defect.

Reuse dispersion_contract.csv from operator-contract audit for all90 signs, units, name matching, effective targets and shifts. No D3 method sweep, target edit or recomputation was made.
''')
    document('missing_materials','''# Draft request — автору исходного mRKS-экспорта (НЕ ОТПРАВЛЕН)

Нужно аутентифицировать 90 записей, прошедших в data_vxc_train.pickle
(SHA256 0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49).
Для минимального подтверждения достаточно исходника/config и связанных логов
H2, BeH2, CO плюс подтверждения общей процедуры и исключений для остальных 87.

1. Пришлите script/notebook и run config, создававшие inp_mrks.npz, особенно assignments для vxc_grid, exc_wf, t_wf и meta_keys. В NPZ input_file указан /home/mmedvedev/lsvvt/chep/ready/{system}/inp.orca.zst. Нужны версия реализации inversion/mRKS и связь outputs с нашими files/hashes. Это закроет producer→NPZ и физический смысл potential/energy fields; публикация об общем алгоритме не заменяет данный exporter.

2. Пришлите script/формулу, создававшие H5 E_xc, E_cbs_1 и E_no_xc до prepare_vxc.py. Сохранился h5_vrho_from_mrks_sanity.csv. На пилоте E_xc=E_cbs_1−E_no_xc и F32 импорт дают наши legacy targets. Подтвердите, содержит ли E_no_xc E_nuc+Tr(P_KS T)+Tr(P_KS V_ext)+0.5Tr(P_KS J_KS), какая именно плотность/геометрия/заряд использованы и когда округлялись значения. Это закроет рецепт, не подменяя его точным E_xc[n_KS].

3. Укажите method/basis/extrapolation для E_CBS_1 и что означают E_scf_5Z, E_scf_QZ, E_corr_5Z, E_corr_QZ в extrapolation_e.csv; включены ли ядра/эмпирические поправки. Нужны связанный расчетный input/log или краткая первичная формула автора. Это закроет происхождение reference total/electronic energy и CBS приближения.

4. Укажите правило gauge и точный момент его применения к сохранённому vxc_grid: potential shift, orbital-energy shift, состояние итерации, units/spin, masking/filtering/tail clipping/interpolation. HOMO+i_ekt (Ha) равен +5.8980e-5 для H2, −3.08894e-6 для BeH2, −1.08943e-4 для CO; почему это не точный ноль и согласованы ли орбитали с экспортированным потенциалом? Assignment плюс соответствующий export log снимет GAUGE_UNRESOLVED.

5. Для dispersions_mrks.pickle, добавленного в Git 4ed868e (SHA256 a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440), пришлите исходные H5 atom/charge/spin или geometry hashes, команду --h5-dir, backend/library version, фактические PBE0/D3(BJ) numerical parameters и настройки дополнительных членов. Это закроет геометрию и историческую конфигурацию, которые не устанавливаются современным D3 replay.

6. Подтвердите intended energy convention: NN integral представляет полный XC либо остаток E_target−D? Был ли исходный E_target чистым ab initio XC? Сейчас код добавляет D к prediction один раз, сохраняя E_target. Краткое первичное описание этой договорённости закроет semantic blocker; мы не хотим оправдывать ее задним числом.

Личность автора не установлена; имя и адрес получателя не предполагаются.
Никакие targets, potentials, loss или архитектура не изменены.
''')
    wall=time.time()-start
    decision={'energy_contract':'UNRESOLVED','potential_semantics':'UNRESOLVED','potential_gauge':'UNRESOLVED',
              'dispersion_convention':'UNRESOLVED','operative_residual_equation':'VERIFIED',
              'dispersion_artifact_history':'VERIFIED','historical_lineage':evidence['contracts'],
              'verdict':'PRIMARY_MATERIALS_REQUIRED; no confirmed target/code mismatch',
              'next_action':'send the prepared single draft to the original data exporter only after separate explicit user authorization',
              'required_external_materials':['NPZ inversion/export source and config/log','legacy H5 E_xc/component exporter','CBS method/basis/extrapolation and input/log','potential gauge/export assignments','original D3 geometry/backend/parameters/run receipt','author-intended full-XC/residual convention'],
              'new_numerical_checks':'3 pilot scalar-log/F32 checks only; no new molecular evaluation',
              'trained_checkpoints_loaded':0,'optimizer_states_loaded':0,'training_updates':0,
              'SCF_executed':False,'targets_modified':False,'production_loss_modified':False,
              'architecture_modified':False,'messages_sent':False,'Clean28_future_test_accessed':False,
              'wall_seconds':wall,'search_closed_elapsed_seconds':search_rows[0]['search_closed_at_elapsed_seconds'],
              'budget_respected':wall<=7200}
    write('decision',decision)
    document('report',f'''# XC target provenance recovery

Verdict: primary contracts remain UNRESOLVED; concrete external materials are enumerated. No implementation/target mismatch was confirmed. Search is closed, not deferred to an unspecified further search.

Start HEAD `{protocol['head']}`; branch lap_full_vxc; later/untracked work preserved. First-action stopwatch covers search, metadata, scalar replay and evidence preparation. Previous operator-contract audit classifications and calculations were reused unchanged; no PBE/AO/general operator audit repeated.

| Question | Confirmed producer/consumer boundary | Remaining blocker | Evidence scope |
|---|---|---|---|
| legacy E_xc | H5 import into F32 at 6f91e72; exact downstream preservation; surviving sanity log gives E_cbs_1−E_no_xc | original H5 component exporter and CBS method/config | all90 preservation reused; 3 pilot log rounding checks |
| vxc_grid/Vrho | NPZ values→legacy→RefAO | exact inversion/export implementation and gauge application | all90 preservation reused; 3 pilot alignment values reused |
| D3 | active binary artifact exactly matches 4ed868e; PBE0/d3bj utility; integral+D equation | input geometry/backend/params/run and author-intended energy convention | all90 artifact/consumer, historical run unknown |

The new sanity CSV corroborates a historical energy-residual recipe, but does not authenticate its generating source or make a CBS/WF reference an exact functional value at n_KS. exc_wf/t_wf source meanings remain unqualified. Approximate HOMO/i_ekt alignment cannot establish a gauge applied to saved vxc_grid. The code necessarily fits the NN integral to E_target−D; that operative fact does not prove the author's intended physical convention.

New scalar replay: all three named H5 rows satisfy the explicit logged subtraction within a predeclared decimal/F64 bound; their F32 values equal legacy exactly. No alternative CBS formulas were searched for numerical fit. No D3 evaluation was run because the original geometry/config/backend is missing. No new NN, AO, inversion, SCF or reference calculations were performed.

Sources, commits/hashes, code ranges and coverage/evidence levels: evidence.json. Checked locations, queries and stop reasons: search_log.csv. Three focused contracts: energy_contract.md, potential_gauge_contract.md, dispersion_contract.md. The only next action is obtaining the six narrowly specified materials in missing_materials.md from the original exporter. The neutral request is a draft; nothing was sent. Current evidence does not authorize a repair or new training.

Wall time at evidence generation: {wall:.2f} s / 7200 s; search closed at {search_rows[0]['search_closed_at_elapsed_seconds']:.2f} s / 3600 s. Final test/Git elapsed time will be separately recorded from the same first-action clock. No unmeasured preliminary search.
''')


if __name__=='__main__':
    main()
