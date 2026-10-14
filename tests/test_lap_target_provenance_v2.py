"""CBS parser, literal arithmetic, source identity and noncircular linkage proof."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('target_v2',ROOT/'tools/lap_target_provenance_v2.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parser_rejects_duplicates_and_schema_changes(tmp_path):
    path=tmp_path/'source.csv'
    path.write_text('System,value\nH2,1\nH2,2\n')
    with pytest.raises(ValueError,match='duplicate'):
        MODULE.indexed_csv(path)
    with pytest.raises(ValueError,match='schema'):
        MODULE.indexed_csv(path,schema=MODULE.COLUMNS)


def test_exact_identity_isomers_and_case_are_not_fuzzily_joined():
    MODULE.require_exact_identities({'AlBeH','AlBeH_iso2'},{'AlBeH_iso2','AlBeH'})
    with pytest.raises(ValueError,match='exact identities'):
        MODULE.require_exact_identities({'H2'},{'h2'})
    with pytest.raises(ValueError,match='exact identities'):
        MODULE.require_exact_identities({'AlBeH','AlBeH_iso2'},{'AlBeH'})


def test_literal_formulas_and_existing_column_mapping():
    rows=MODULE.indexed_csv(MODULE.CSV_PATH,schema=MODULE.COLUMNS)
    assert len(rows)==90
    for row in rows.values():
        values=MODULE.literal_cbs(*(float(row[k]) for k in ('E_scf_5Z','E_scf_QZ','E_corr_5Z','E_corr_QZ')))
        assert all(values[c]==float(row[c]) for c in ('E_CBS_1','E_CBS_2','E_CBS_corr'))
        expected_none=float(row['E_scf_5Z'])+(125*float(row['E_corr_5Z'])-64*float(row['E_corr_QZ']))/61
        assert values['E_CBS_corr']==expected_none


def test_exponential_model_solution_is_separate_from_literal_cbs2():
    infinity=-5.; amplitude=3.
    s5=infinity+amplitude*math.exp(-1.63*5)
    s4=infinity+amplitude*math.exp(-1.63*4)
    result=MODULE.literal_cbs(s5,s4,0.,0.)
    assert abs(result['model_scf2']-infinity)<1e-14
    assert abs(result['literal_scf2']-infinity)>1e-4


def test_detects_wrong_column_without_circular_noxc_reconstruction():
    row={'E_no_xc':'-99.459337493252','E_cbs_1':'-113.326472447679','E_xc':'-13.867134954427'}
    legacy=-13.867135047912598
    correct=MODULE.check_link(-113.326472447679,row,legacy)
    assert correct['status']=='PASS'
    wrong=MODULE.check_link(-113.326472447679+.002,row,legacy)
    assert wrong['status']=='MISMATCH'
    changed_target=MODULE.check_link(-113.326472447679,row,legacy+.01)
    assert changed_target['status']=='MISMATCH'
    assert correct['logged_noXC_ha']==wrong['logged_noXC_ha']==changed_target['logged_noXC_ha']
    assert row['E_no_xc']=='-99.459337493252'


def test_f64_subtraction_precedes_f32_storage():
    cbs=-113.326472447679; noxc=-99.459337493252
    actual=MODULE.f32_after_subtraction(cbs,noxc)
    incorrectly_rounded=float(np.float32(cbs)-np.float32(noxc))
    assert actual==float(np.float32(cbs-noxc))
    assert actual!=incorrectly_rounded


def test_all90_linkage_and_separate_pilot_coverage():
    with (ROOT/'lap_target_provenance_v2_target_linkage.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==90
    assert all(r['status']=='PASS' and r['current_corpus_status']=='PASS' for r in rows)
    assert {r['System'] for r in rows if r['independent_noXC_status']=='PASS'}=={'H2','BeH2','CO'}
    assert all(r['E_CBS_2_F32_target_equal']=='False' for r in rows)


def test_old_artifacts_unchanged_and_original_images_not_fabricated():
    evidence=json.loads((ROOT/'lap_target_provenance_v2_evidence.json').read_text())
    assert all(MODULE.sha(ROOT/p)==s for p,s in evidence['historical_artifacts'].items())
    assert MODULE.sha(MODULE.CSV_PATH)==MODULE.EXPECTED
    assert evidence['image_hashes']==[]
    assert evidence['primary_correspondence_source'].startswith('USER_TEXT_TRANSMISSION')
    status=json.loads((ROOT/'lap_target_provenance_v2_status_update.json').read_text())
    assert status['energy_recipe_documented'] and status['final_potential_semantics_documented']
    assert status['potential_gauge']=='UNRESOLVED' and status['D3_convention']=='UNRESOLVED'
