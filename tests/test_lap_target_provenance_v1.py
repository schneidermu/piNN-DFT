"""Small parser/rounding/receipt checks; no electronic or model calculations."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('target_provenance',ROOT/'tools/lap_target_provenance_v1.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_csv_parser_rejects_duplicate_identities(tmp_path):
    path=tmp_path/'duplicates.csv'
    path.write_text('id,value\na,1\na,2\n')
    with pytest.raises(ValueError,match='Duplicate'):
        MODULE.csv_index(path,'id')


def test_csv_parser_accepts_utf8_bom(tmp_path):
    path=tmp_path/'source.csv'
    path.write_text('id,value\na,1\n',encoding='utf-8-sig')
    assert MODULE.csv_index(path,'id')['a']['value']=='1'


def test_explicit_recipe_and_f32_import_are_separate():
    row={'E_xc':'-0.702356641306','E_cbs_1':'-1.174539999999','E_no_xc':'-0.472183358693'}
    legacy=float(np.float32(float(row['E_xc'])))
    result=MODULE.replay_scalar(row,legacy)
    assert result['status']=='PASS'
    assert result['logged_E_xc_ha']!=legacy
    wrong=dict(row,E_no_xc='0.0')
    assert MODULE.replay_scalar(wrong,legacy)['status']=='SOURCE_MISMATCH'
    assert MODULE.replay_scalar(row,legacy+.01)['status']=='SOURCE_MISMATCH'


@pytest.mark.parametrize('bad',['nan','inf','-inf'])
def test_replay_fails_closed_on_nonfinite_source(bad):
    with pytest.raises(ValueError,match='Nonfinite'):
        MODULE.replay_scalar({'E_xc':bad,'E_cbs_1':'1','E_no_xc':'2'},0.)


def test_saved_pilot_replays_do_not_become_full90_semantics():
    with (ROOT/'lap_target_provenance_v1_replay.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert [r['system'] for r in rows]==['H2','BeH2','CO']
    assert all(r['status']=='PASS' for r in rows)
    decision=json.loads((ROOT/'lap_target_provenance_v1_decision.json').read_text())
    assert decision['energy_contract']=='UNRESOLVED'
    assert decision['potential_gauge']=='UNRESOLVED'
    assert decision['dispersion_convention']=='UNRESOLVED'
    assert decision['training_updates']==0 and not decision['messages_sent']


def test_receipts_and_artifact_lineage():
    protocol=json.loads((ROOT/'lap_target_provenance_v1_protocol.json').read_text())
    assert protocol['wall_budget_seconds']==7200
    assert protocol['search_budget_seconds']==3600
    assert all(MODULE.sha(ROOT/p)==s for p,s in protocol['previous_audit_receipts'].items())
    evidence=json.loads((ROOT/'lap_target_provenance_v1_evidence.json').read_text())
    assert evidence['D3_historical_blob_sha256']==evidence['D3_active_blob_sha256']
    assert all(MODULE.sha(v['path'])==v['sha256'] for v in evidence['candidate_sources'].values())
    assert evidence['search_closed'] and not evidence['author_request_sent']
    assert len(evidence['MRKS_archive_sources'])==17
