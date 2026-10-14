"""Bounded diagnostic corrections/attribution; preserves original audit outputs."""

import math
import time

import lap_operator_contract_v1_audit as audit
import numpy as np
import torch
from pyscf.dft import numint


def corrected_fixtures():
    original = audit.independent_features

    def symmetric_leaf(phi, grad, lap, dm):
        return original(phi, grad, lap, (dm + dm.T) / 2)

    audit.independent_features = symmetric_leaf
    try:
        return audit.fixture_checks()
    finally:
        audit.independent_features = original


class CounterfactualPBE(torch.nn.Module):
    """Isolated source copy; no production module monkeypatch or file changes."""

    def __init__(self, no_floor):
        super().__init__()
        source = (audit.ROOT / 'dft_functionals/PBE.py').read_text()
        if no_floor:
            assert 'eps_add = 1e-7' in source and 'eps_add_rho = 1e-10' in source
            source = source.replace('eps_add = 1e-7', 'eps_add = 0.0')
            source = source.replace('eps_add_rho = 1e-10', 'eps_add_rho = 0.0')
        namespace = {}
        # Only the hash-authenticated local production module is executed here.
        exec(compile(source, 'diagnostic_only_PBE_source_copy', 'exec'), namespace)  # noqa: S102
        self.formula = namespace['F_PBE']
        constants = audit.PBE_CONSTANTS.double().clone()
        constants[0, [0, 1, 3, 6, 9, 12, 15, 18, 21, 22, 23, 24, 25]] = torch.tensor([
            .06672455060314922, (1-math.log(2))/math.pi**2,
            7.5957, 3.5876, 1.6382, .49294, .0310907, .21370,
            -3/8*(3/math.pi)**(1/3)*4**(2/3), .804,
            .2195149727645171, .804, .2195149727645171,
        ], dtype=torch.float64)
        self.register_buffer('constants', constants)

    def forward(self, rho, sigma, lapl):
        del lapl
        return self.formula(rho, sigma, self.constants.expand(len(rho), -1), rho.device)*rho.sum(-1)


def component_checks(protocol):
    rows = []
    for name in audit.PILOT:
        record = audit.load_central_operator_record(protocol['pilot_files'][name]['path'])
        mol = audit.build_molecule_from_metadata(record.metadata)
        with np.load(audit.NPZ_ROOT / record.metadata['source']['npz_relative_id'], allow_pickle=False) as source:
            z = {key: source[key] for key in source.files}
        ni = numint.NumInt()
        matrices = {key: np.zeros_like(record.RefAO) for key in ('precise_constants', 'precise_no_floors')}
        energies = dict.fromkeys(matrices, 0.)
        functionals = {'precise_constants': CounterfactualPBE(False), 'precise_no_floors': CounterfactualPBE(True)}
        npz_projection = np.zeros_like(record.RefAO)
        for start in range(0, len(record.weights), 4096):
            assert time.time()-protocol['wall_origin_unix'] < protocol['budget_seconds']-120
            sl = slice(start, start+4096)
            ao = ni.eval_ao(mol, record.coords64[sl], deriv=2)
            phi = audit.tensor(ao[0])
            grad = audit.tensor(ao[1:4].transpose(1, 0, 2))
            lap = audit.tensor(ao[4]+ao[7]+ao[9])
            w = audit.tensor(record.weights[sl])
            f = audit.op.rks_density_features_from_ao(phi, grad, lap, audit.tensor(record.dmks))
            sigma = audit.op.sigma_from_gradients(f[:, 2:8].reshape(-1, 2, 3))
            for key, energy in functionals.items():
                energies[key] += float((energy(f[:, :2], sigma, f[:, 8:])*w).sum().detach())
                matrices[key] += audit.op.assemble_rks_operator(energy, f, w, phi, grad, lap, chunk_size=256, create_graph=False).numpy()
            npz_projection += ao[0].T @ (ao[0]*(record.weights[sl]*z['vxc_grid'][record.npzrow[sl]])[:, None])
        saved = np.load(audit.ROOT.parent/'lap_operator_contract_v1_arrays_20261014'/f'{name}.npz')
        ev, vec = np.linalg.eigh(record.Overlap)
        x = (vec*ev**-.5)@vec.T
        oracle = saved['oracle']
        for key, matrix in matrices.items():
            gap = float(np.linalg.norm(x@(matrix-oracle)@x))
            relative = gap/float(np.linalg.norm(x@oracle@x))
            rows.append({'system': name, 'component': key, 'Exc_ha': energies[key],
                         'orth_absolute_fro': gap, 'orth_relative': relative,
                         'status': 'PASS' if key!='precise_no_floors' or relative<=audit.TOL['ideal_matrix_relative'] else 'FAIL',
                         'interpretation': 'sensitivity counterfactual, numerical functional differs; not a production repair'})
        rows.append({'system': name, 'component': 'legacy_F32_vxc_vs_source_NPZ_F64_projection',
                     'orth_absolute_fro': float(np.linalg.norm(x@(npz_projection-record.RefAO)@x)),
                     'AO_absolute_fro': float(np.linalg.norm(npz_projection-record.RefAO)),
                     'status': 'MEASURED'})
        meta = dict(zip(z['meta_keys'].tolist(), z['meta'].tolist()))
        hole_integral = float(.5*np.dot(z['grid_weights'], z['rho_wf']*z['vxc_hole']))
        rows.append({'system': name, 'component': 'exc_wf_vs_half_rho_vxc_hole_integral',
                     'candidate_Wxc_ha': hole_integral, 'exc_wf_ha': meta['exc_wf'],
                     'difference_ha': hole_integral-meta['exc_wf'], 'input_file': str(z['input_file'].reshape(-1)[0]),
                     'status': 'DEFINITION_UNRESOLVED',
                     'interpretation': 'numeric identity cannot alone prove exporter definition or T_c inclusion'})
    return rows


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    protocol = audit.read(audit.ROOT/(audit.PREFIX+'protocol.json'))
    # Freeze correction rationale before numerical recheck. Original results retained.
    audit.write('supplement_protocol', {
        'source_sha256': audit.sha(__file__), 'parent_protocol_sha256': audit.sha(audit.ROOT/(audit.PREFIX+'protocol.json')),
        'original_fixture_results_sha256': audit.sha(audit.ROOT/(audit.PREFIX+'variational_checks.csv')),
        'diagnostic_defect': 'independent energy graph assumed symmetric DM while differentiating independent full-matrix entries; antisymmetric derivative is outside declared DM space',
        'correction': 'explicit symmetric DM parameterization in energy-first fixture only',
        'tolerances': protocol['tolerances'], 'pilot': protocol['pilot'], 'no_production_mutation': True,
    })
    rows = corrected_fixtures()
    audit.table('variational_corrected_checks', rows)
    components = component_checks(protocol)
    audit.table('precision_component_checks', components)
    decision = audit.read(audit.ROOT/(audit.PREFIX+'decision.json'))
    decision['variational_assembly'] = 'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    decision['original_fixture_FAIL_preserved'] = True
    decision['diagnostic_implementation_error'] = 'symmetric DM graph fixed in independently recorded supplement, production unchanged'
    decision['scientific_verdict'] = ['TARGET_PROVENANCE_UNRESOLVED', 'NUMERICAL_OR_METRIC_SENSITIVITY_IDENTIFIED']
    audit.write('decision', decision)
    audit.write('supplement_receipt', {'active_seconds': time.perf_counter()-started,
                                      'scientific_wall_seconds': time.time()-protocol['wall_origin_unix'],
                                      'production_sources_unchanged': protocol['sources']=={k:audit.sha(audit.ROOT/k) for k in protocol['sources']},
                                      'optimizer_or_checkpoint_access': False})


if __name__=='__main__':
    main()
