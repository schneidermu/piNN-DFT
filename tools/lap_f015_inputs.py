"""Frozen S5/Lap energies, physical descriptors and reversible finite probes."""
import argparse
import copy
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_f015_audit as audit
import numpy as np
import torch

from dft_functionals import PBE
from dft_functionals.constants import EPS_RHO, NN_OUTPUT_SCALE_PBE
from train_models.lap_moo_panel import MinnesotaGroupStore
from train_models.optuna_joint import build_model


def models():
    result = {}
    source = audit.geometry.b8.verify_sources()
    for spec in audit.STATES:
        model, shadow, _run = audit.existing._load_eval_model(source, spec['checkpoint'])
        shadow.load_state_dict(model.state_dict(), strict=True)
        result[spec['name']] = shadow.eval()
        del model
    s5 = build_model(argparse.Namespace(name='PBE-LGxGc_6_32', model_type='gc_svelu_mirror', dropout=0., dtype=torch.float64), torch.device('cuda'))
    weights = torch.load(audit.S5, map_location='cpu', weights_only=False)
    s5.load_state_dict(weights, strict=True)
    assert sum(p.numel() for p in s5.parameters()) == 9542
    assert all(torch.equal(s5.state_dict()[n].cpu(), v.double()) for n, v in weights.items())
    result['s5'] = s5.double().eval()
    return result


def integrated_parts(model, raw, rho, sigma, weights):
    """Same F64 PBE functions and signed quadrature as reaction integration."""
    totals = [torch.zeros((), device='cuda', dtype=torch.float64) for _ in range(2)]
    density = []
    with torch.no_grad():
        for start in range(0, len(raw), 8192):
            sl = slice(start, start + 8192)
            constants = model(raw[sl])
            rs, zeta = PBE.rs_z_calc(rho[sl])
            xs0, xs1, xt = PBE.xs_xt_calc(rho[sl], sigma[sl])
            ex = PBE.PBE_X(rs, zeta, xt, xs0, xs1, constants)
            ec = PBE.PBE_C(rs, zeta, xt, constants, raw.device)
            assert torch.allclose(ex + ec, PBE.F_PBE(rho[sl], sigma[sl], constants, raw.device), rtol=1e-12, atol=1e-12)
            factor = rho[sl].sum(-1) * weights[sl]
            totals[0] += (factor * ex).sum()
            totals[1] += (factor * ec).sum()
            density.append(torch.stack([ex, ec], -1).cpu().numpy())
    return float(totals[0]), float(totals[1]), np.concatenate(density)


def atomic_comparison():
    """Separate AE17 atomic-energy comparison, one fixed variant, not part of J251."""
    audit.verify()
    run = audit.geometry.b8._import_run()
    bundle = audit.TrainingBundle(run.DATA)
    nets = models()
    before = {n: run.existing.digest(m) for n, m in nets.items()}
    manifest = audit.read(audit.ROOT / 'relchem_joint_epoch_evaluation_manifest.json')
    chosen = [r for r in manifest['rows'] if r['task'] == 'ae17']
    assert len(chosen) == len({r['identity'] for r in chosen}) == 17
    dispersion = bundle.chemistry_dispersions()
    results = []
    for spec in chosen:
        meta = bundle.reactions[spec['identity']]
        assert len(meta['components']) == 1 and meta['coefficients'] == [1.0]
        reaction = audit.geometry.b8._load_reaction(bundle, 'ae17', {**spec, 'database': 'AE17'})
        for name, model in nets.items():
            with torch.no_grad():
                prediction, target = audit.chemistry._prediction(model, reaction, dispersion)
            residual = float(prediction - target)
            results.append({'identity': spec['identity'], 'variant': spec['variant'], 'species': meta['components'][0],
                            'model': name, 'prediction_kcal': float(prediction), 'reference_kcal': float(target),
                            'signed_error_kcal': residual, 'absolute_error_kcal': abs(residual)})
    audit.table('lap_f015_matched_ae17_errors.csv', results)
    # One independent qualified-forward replay per database, not a new population sweep.
    matched = {r['identity']: r for r in audit.read(audit.OUT / 'matched_s5_rows.json')}
    representatives = {}
    for spec in audit.chemistry._selected(bundle):
        representatives.setdefault(spec['database'], spec)
    parity = []
    for spec in representatives.values():
        reaction = audit.geometry.b8._load_reaction(bundle, 'relchem', spec)
        with torch.no_grad():
            prediction, target = audit.chemistry._prediction(nets['s5'], reaction, dispersion)
        residual = float(prediction - target)
        gap = abs(residual - matched[spec['identity']]['residual'])
        assert gap <= 1e-6
        parity.append({'identity': spec['identity'], 'database': spec['database'], 'qualified_residual': residual,
                       'decomposed_residual': matched[spec['identity']]['residual'], 'absolute_gap': gap})
    audit.write(audit.OUT / 's5_qualified_forward_parity.json', parity)
    assert all(run.existing.digest(m) == before[n] for n, m in nets.items())
    bundle.close()
    audit.verify()


def reaction_rows():
    protocol = audit.verify()
    bundle = audit.TrainingBundle(audit.ROOT.parent / 'publication_dataset_v1')
    selected = audit.chemistry._selected(bundle)
    old_protocol = audit.read(audit.ROOT.parent / 'lap_s5_ae17_runs_20261004/protocol.json')
    store_path = next(path for path in old_protocol['pins'] if path.endswith('mn_group_store_268\\manifest.json'))
    old = MinnesotaGroupStore(store_path, cache_groups=1)
    accepted = [__import__('json').loads(line) for line in (audit.ROOT.parent / 'lap_s5_ae17_runs_20261004/s5_rows.jsonl').read_text().splitlines()]
    accepted = {(r['database'], r['reaction_id']): r for r in accepted}
    cache = {s['name']: audit.load(s)[0] for s in audit.STATES}
    assert all([(r['identity'], r['variant']) for r in value['rows']] == [(s['identity'], s['variant']) for s in selected] for value in cache.values())
    nets = models()
    before = {n: audit.geometry.b8._import_run().existing.digest(m) for n, m in nets.items()}
    dispersion = bundle.chemistry_dispersions()
    reaction_output, species_output, matched_s5, provenance = [], [], [], []
    spatial_rows = []
    cached_species = {}
    began = time.perf_counter()
    for index, spec in enumerate(selected):
        record_path = audit.OUT / f'reaction_{index:03}.json'
        if record_path.exists():
            payload = audit.read(record_path)
        else:
            assert (spec['database'], spec['reaction_id']) in accepted
            reaction = bundle.chemistry().load_variant(spec['identity'], spec['variant'])
            legacy = old.load_variant((spec['database'], spec['reaction_id']), spec['variant'])
            fields = ('Grid', 'Weights', 'Densities', 'Gradients', 'HF_energies', 'Coefficients', 'Energy', 'backsplit_ind')
            assert list(reaction['Components']) == list(legacy['Components'])
            assert all(np.array_equal(np.asarray(reaction[f]), np.asarray(legacy[f])) for f in fields), spec
            record = audit.geometry.b8._import_run().existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
            rows, species, parts, species_meta = [], [], {}, []
            start = 0
            for component_index, end in enumerate(record['backsplit_ind'].int().tolist()):
                reference_component = bundle.reactions[spec['identity']]['variants'][spec['variant']][component_index]
                key = reference_component['grid_id']
                component = str(reaction['Components'][component_index])
                if key not in cached_species:
                    raw, rho, sigma, weights = (record[f][start:end] for f in ('Grid', 'Densities', 'Gradients', 'Weights'))
                    values, densities = {}, {}
                    for name, model in nets.items():
                        ex, ec, density = integrated_parts(model, raw, rho, sigma, weights)
                        values[name] = {'ex': ex, 'ec': ec, 'exc': ex + ec}
                        densities[name] = density
                    # Full-grid energy decomposition of the primary S5/B8 comparison.
                    diff = (densities['b8_t128'] - densities['s5']) * (rho.sum(-1) * weights).cpu().numpy()[:, None]
                    q = raw[:, 7:9] / (4 * (3 * math.pi**2)**(2 / 3) * (raw[:, :2] + EPS_RHO)**(5 / 3))
                    saturated = (torch.tanh(q).abs() > .99).any(-1).cpu().numpy()
                    densities_meta = {'grid_id': key, 'species': component, 'n_grid': len(raw),
                                      'signed_delta_ex_saturated': float(diff[saturated, 0].sum()),
                                      'signed_delta_ec_saturated': float(diff[saturated, 1].sum()),
                                      'abs_delta_exc_mass': float(np.abs(diff.sum(-1)).sum()),
                                      'abs_delta_exc_saturated_mass': float(np.abs(diff.sum(-1))[saturated].sum())}
                    cached_species[key] = (values, densities_meta)
                values, density_meta = cached_species[key]
                parts[component_index] = values
                coefficient = float(record['Coefficients'][component_index])
                hf = float(record['HF_energies'][component_index])
                d3 = float(np.asarray(dispersion.get(component, 0)))
                for name, value in values.items():
                    species.append({'identity': spec['identity'], 'database': spec['database'], 'variant': spec['variant'],
                                    'grid_id': key, 'species': component, 'component_index': component_index,
                                    'coefficient': coefficient, 'model': name, 'ex_hartree': value['ex'],
                                    'ec_hartree': value['ec'], 'exc_hartree': value['exc'],
                                    'fixed_nonxc_hartree': hf, 'dispersion_hartree': d3,
                                    'stoichiometric_xc_kcal_mol': coefficient * value['exc'] * 627.5095})
                species_meta.append({**density_meta, 'identity': spec['identity'], 'database': spec['database'], 'coefficient': coefficient})
                start = end
            reference = float(record['Energy'].reshape(()))
            for name in nets:
                predicted = sum(float(record['Coefficients'][i]) * (parts[i][name]['exc'] + float(record['HF_energies'][i])
                                + float(np.asarray(dispersion.get(str(component), 0))))
                                for i, component in enumerate(reaction['Components'])) * 627.5095
                residual = predicted - reference
                if name != 's5':
                    assert abs(residual - cache[name]['rows'][index]['residual']) <= protocol['tolerances']['energy_kcal_abs'], (spec, name, residual)
                rows.append({**spec, 'model': name, 'reference': reference, 'prediction': predicted,
                             'residual': residual, 'absolute_error': abs(residual),
                             'old_singleton': audit.chemistry.singleton_objective(residual, audit.chemistry._factor(spec['database'])),
                             'jw_contribution': abs(residual) / audit.read(audit.ROOT / 'lap_f010_normalization_receipt.json')['scales'][spec['database']] / 251})
            prior_row = accepted[(spec['database'], spec['reaction_id'])]
            current_s5 = next(r for r in rows if r['model'] == 's5')
            if prior_row['variant_suffix'] == spec['variant']:
                assert abs(current_s5['residual'] - prior_row['signed_error']) <= 1e-6
            payload = {'rows': rows, 'species': species, 'spatial': species_meta,
                       'parity': {'identity': spec['identity'], 'variant': spec['variant'], 'all_input_arrays_exact': True,
                                  'source_group_sha256': bundle.reactions[spec['identity']]['source_group_sha256'],
                                  'same_old_evaluation_variant': prior_row['variant_suffix'] == spec['variant'],
                                  'original_geometry_basis_dm_embedded': False}}
            audit.write(record_path, payload)
            del reaction, legacy, record
        reaction_output.extend(payload['rows'])
        species_output.extend(payload['species'])
        spatial_rows.extend(payload['spatial'])
        provenance.append(payload['parity'])
        matched_s5.append(next(r for r in payload['rows'] if r['model'] == 's5'))
        if index % 40 == 0:
            print('F015_MATCHED', index + 1, 'seconds', time.perf_counter() - began, flush=True)
    lookup = {r['identity']: r for r in matched_s5}
    for row in reaction_output:
        row['s5_advantage_absolute_error'] = row['absolute_error'] - lookup[row['identity']]['absolute_error']
    audit.table('lap_f015_matched_reaction_errors.csv', reaction_output)
    audit.table('lap_f015_species_xc_decomposition.csv', species_output)
    audit.table('lap_f015_spatial_energy_differences.csv', spatial_rows)
    summaries = []
    for name in nets:
        group = [r for r in reaction_output if r['model'] == name]
        for db in ('ALL_REACTION_WEIGHTED',) + tuple(sorted({r['database'] for r in group})):
            selected_rows = [r for r in group if db == 'ALL_REACTION_WEIGHTED' or r['database'] == db]
            errors = [r['residual'] for r in selected_rows]
            summaries.append({'model': name, 'database': db, 'n': len(errors), **audit.stats(errors),
                              's5_wins': sum(r['s5_advantage_absolute_error'] > 1e-10 for r in selected_rows),
                              's5_losses': sum(r['s5_advantage_absolute_error'] < -1e-10 for r in selected_rows),
                              'ties': sum(abs(r['s5_advantage_absolute_error']) <= 1e-10 for r in selected_rows),
                              'legacy_singleton_mean': float(np.mean([r['old_singleton'] for r in selected_rows])),
                              'jw_contribution': sum(r['jw_contribution'] for r in selected_rows),
                              'signed_error_correlation_s5': float(np.corrcoef(errors, [lookup[r['identity']]['residual'] for r in selected_rows])[0, 1])})
    audit.table('lap_f015_database_comparison.csv', summaries)
    audit.write(audit.OUT / 'matched_s5_rows.json', matched_s5)
    audit.write(audit.OUT / 'input_parity.json', {'reactions': provenance, 'matched_identities': 251,
                 'original_geometry_basis_limitation': 'Source group identity/hash and exact numerical arrays shared; original DM/geometry/basis not embedded in reaction grids and not independently reconstructed.',
                 'all_model_states_unchanged': all(audit.geometry.b8._import_run().existing.digest(m) == before[n] for n, m in nets.items()),
                 'seconds': time.perf_counter() - began})
    assert all(audit.geometry.b8._import_run().existing.digest(m) == before[n] for n, m in nets.items())
    bundle.close()
    audit.verify()


@contextmanager
def restored(models):
    saved = [copy.deepcopy(m.state_dict()) for m in models]
    rng = audit.geometry.b8._capture_rng()
    before = [audit.geometry.b8._import_run().existing.digest(m) for m in models]
    try:
        yield
    finally:
        for model, state in zip(models, saved, strict=True):
            model.load_state_dict(state, strict=True)
            assert all(torch.equal(v, model.state_dict()[n]) for n, v in state.items())
        audit.geometry.b8._restore_rng(rng)
        assert audit.geometry.b8._import_run().existing.equal(rng, audit.geometry.b8._capture_rng())
        assert before == [audit.geometry.b8._import_run().existing.digest(m) for m in models]


def descriptor_audit():
    """Full selected grids for saturation; stride101 samples for input derivatives."""
    from scipy.spatial import cKDTree

    audit.verify()
    run = audit.geometry.b8._import_run()
    bundle = audit.TrainingBundle(run.DATA)
    nets = models()
    nets = {n: nets[n] for n in ('b8_t128', 's5')}
    selected = audit.chemistry._selected(bundle)
    records = {}
    for spec in selected:
        meta = bundle.reactions[spec['identity']]
        for source, name in zip(meta['variants'][spec['variant']], meta['components'], strict=True):
            key = source['grid_id']
            records.setdefault(key, {'source': source, 'name': name, 'databases': set()})['databases'].add(spec['database'])
    distributions, sensitivities, samples, fd_receipts = [], [], [], []
    before = {n: run.existing.digest(m) for n, m in nets.items()}
    channels = (0, 1, 22, 23, 24, 25, 26, 27, 28)
    field_names = ('beta', 'gamma', 'kappa_up', 'mu_up', 'kappa_down', 'mu_down', 'GNN_up', 'GNN_down', 'Gc')
    start_time = time.perf_counter()
    for grid_index, (grid_id, info) in enumerate(sorted(records.items())):
        receipt = audit.OUT / f'descriptor_{grid_index:03}.json'
        sample_path = audit.OUT / f'descriptor_{grid_index:03}.npz'
        if receipt.exists():
            payload = audit.read(receipt)
            assert audit.sha(sample_path) == payload['sample_sha256']
            with np.load(sample_path) as a:
                samples.append({k: a[k] for k in a.files})
            distributions.extend(payload['distributions'])
            sensitivities.extend(payload['sensitivities'])
            fd_receipts.extend(payload['fd'])
            continue
        src = info['source']
        group = bundle.handles.open(src['shard'])[src['group']]
        raw_np, w_np = group['Grid'][...].astype(np.float64), group['Weights'][...].astype(np.float64)
        assert np.isfinite(raw_np).all() and np.isfinite(w_np).all()
        sample_index = np.arange(0, len(raw_np), 101)
        raw = torch.from_numpy(raw_np).cuda()
        w = torch.from_numpy(w_np).cuda()
        rho_total = raw[:, :2].sum(-1)
        positive, electrons = w.clamp_min(0), w.clamp_min(0) * rho_total.clamp_min(0)
        regions = {'all': torch.ones(len(raw), dtype=torch.bool, device='cuda'),
                   'rho_lt_1e-6': rho_total < 1e-6, 'rho_1e-6_to_1e-3': (rho_total >= 1e-6) & (rho_total < 1e-3),
                   'rho_ge_1e-3': rho_total >= 1e-3}
        dists = []
        for pathway in ('correlation', 'exchange'):
            transformed_raw = raw if pathway == 'correlation' else raw * raw.new_tensor([2, 2, 4, 4, 4, 2, 2, 2, 2])
            descriptor = nets['b8_t128'].get_correlation_descriptors(transformed_raw).detach()
            for col, field in enumerate(('n_up', 'n_down', 's_up', 's_total', 's_down', 'q_up', 'q_down')):
                value = descriptor[:, col]
                for region, mask in regions.items():
                    if not mask.any():
                        continue
                    row = {'grid_id': grid_id, 'species': info['name'], 'databases': ';'.join(sorted(info['databases'])),
                           'pathway': pathway, 'field': field, 'density_regime': region, 'n_points': int(mask.sum()),
                           'mean_tanh_slope': float((1 - value[mask]**2).mean()),
                           'electron_mean_tanh_slope': float(((1 - value[mask]**2) * electrons[mask]).sum() / electrons[mask].sum()) if electrons[mask].sum() > 0 else None}
                    for threshold in (.95, .99, .999):
                        saturated = (value.abs() > threshold) & mask
                        row[f'point_fraction_gt_{threshold}'] = float(saturated.sum() / mask.sum())
                        row[f'positive_weight_fraction_gt_{threshold}'] = float(positive[saturated].sum() / positive[mask].sum()) if positive[mask].sum() > 0 else None
                        row[f'electron_fraction_gt_{threshold}'] = float(electrons[saturated].sum() / electrons[mask].sum()) if electrons[mask].sum() > 0 else None
                    dists.append(row)
        raw_sample = torch.from_numpy(raw_np[sample_index]).cuda().requires_grad_(True)
        norm_q = 4 * (3 * math.pi**2)**(2/3) * (raw_sample[:, :2] + EPS_RHO)**(5/3)
        sample = {'grid_id': np.full(len(sample_index), grid_index, dtype=np.int32),
                  'lap_descriptors': nets['b8_t128'].get_correlation_descriptors(raw_sample).detach().cpu().numpy(),
                  'tau': raw_np[sample_index, 5:7], 'rho_total': raw_np[sample_index, :2].sum(-1),
                  'electron_weight': np.maximum(w_np[sample_index], 0) * raw_np[sample_index, :2].sum(-1),
                  'q_saturated': (np.abs(raw_np[sample_index, 7:9] / norm_q.detach().cpu().numpy()) > np.arctanh(.99)).any(-1)}
        sens, finite_differences = [], []
        for name, model in nets.items():
            values = model(raw_sample) / NN_OUTPUT_SCALE_PBE.to(raw_sample)
            for index, field in zip(channels, field_names, strict=True):
                derivative = torch.autograd.grad(values[:, index].sum(), raw_sample, retain_graph=True)[0]
                dq = (derivative[:, 7:9] * norm_q).detach().cpu().numpy()
                dt = derivative[:, 5:7].detach().cpu().numpy()
                tau_local = dt * sample['tau']
                sample[f'{name}_{field}_dq'] = dq
                sample[f'{name}_{field}_tau_relative'] = tau_local
                if index in (26, 28) and grid_index < 3:
                    chosen = np.flatnonzero(sample['rho_total'] > 1e-3)[:8]
                    for column, scale in ((7, norm_q[:, 0].detach()), (5, torch.maximum(raw_sample[:, 5].detach().abs(), norm_q[:, 0].detach() * .1))):
                        base = raw_sample[chosen].detach().clone()
                        perturb = torch.zeros_like(base)
                        perturb[:, column] = scale[chosen] * 1e-5
                        with torch.no_grad():
                            finite = ((model(base + perturb) - model(base - perturb)) / NN_OUTPUT_SCALE_PBE.to(base))[:, index] / 2e-5
                        analytic = derivative[chosen, column].detach() * scale[chosen]
                        tolerance = 1e-7 + 1e-4 * analytic.abs()
                        passed = bool(((finite - analytic).abs() <= tolerance).all())
                        finite_differences.append({'model': name, 'grid_id': grid_id, 'channel': field, 'input_column': column,
                                                  'n': len(chosen), 'max_abs_gap': float((finite - analytic).abs().max()) if len(chosen) else 0., 'pass': passed})
                        assert passed, finite_differences[-1]
                for physical, data in (('q', dq), ('tau_physical', dt), ('tau_relative', tau_local)):
                    for regime, mask_np in (('all', np.ones(len(data), dtype=bool)), ('rho_ge_1e-3', sample['rho_total'] >= 1e-3), ('q_saturated_99', sample['q_saturated'])):
                        if not mask_np.any():
                            continue
                        value = np.linalg.norm(data[mask_np], axis=1)
                        ew = sample['electron_weight'][mask_np]
                        sens.append({'model': name, 'grid_id': grid_id, 'species': info['name'], 'channel': field,
                                     'physical_input': physical, 'regime': regime, 'n_points': len(value),
                                     'median_norm': float(np.median(value)), 'p90_norm': float(np.quantile(value, .9)),
                                     'electron_mean_norm': float(np.dot(value, ew) / ew.sum()) if ew.sum() > 0 else None})
        np.savez_compressed(sample_path, **sample)
        audit.write(receipt, {'distributions': dists, 'sensitivities': sens, 'fd': finite_differences,
                             'sample_sha256': audit.sha(sample_path), 'grid_id': grid_id})
        distributions.extend(dists)
        sensitivities.extend(sens)
        fd_receipts.extend(finite_differences)
        samples.append(sample)
        del raw, raw_sample, values, derivative
        if grid_index % 40 == 0:
            print('F015_DESCRIPTORS', grid_index + 1, len(records), 'seconds', time.perf_counter() - start_time, flush=True)
    audit.table('lap_f015_descriptor_distributions.csv', distributions)
    audit.table('lap_f015_descriptor_sensitivities.csv', sensitivities)
    audit.write(audit.OUT / 'descriptor_fd.json', fd_receipts)
    # Non-trained local information diagnostic on deterministic physical points.
    desc = np.concatenate([s['lap_descriptors'] for s in samples])
    tau = np.concatenate([s['tau'] for s in samples])
    rho = np.concatenate([s['rho_total'] for s in samples])
    positive = rho >= 1e-6
    desc, tau = desc[positive], tau[positive]
    indices = np.linspace(0, len(desc) - 1, min(50000, len(desc)), dtype=int)
    desc, tau = desc[indices], tau[indices]
    distance, neighbour = cKDTree(desc).query(desc, k=2, workers=1)
    tf_scale = .3 * (3 * math.pi**2)**(2/3) * (rho[positive][indices] + EPS_RHO)**(5/3)
    tau_relative_difference = np.linalg.norm(tau - tau[neighbour[:, 1]], axis=1) / np.maximum(tf_scale, 1e-12)
    information = []
    for limit in (.001, .01, .05):
        mask = distance[:, 1] <= limit
        information.append({'lap_descriptor_distance_max': limit, 'sample_n': len(desc), 'pair_n': int(mask.sum()),
                            'median_tau_difference_over_total_TF': float(np.median(tau_relative_difference[mask])) if mask.any() else None,
                            'p90_tau_difference_over_total_TF': float(np.quantile(tau_relative_difference[mask], .9)) if mask.any() else None,
                            'TF_floor': 1e-12})
    audit.table('lap_f015_tau_lapl_information.csv', information)
    assert all(run.existing.digest(m) == before[n] for n, m in nets.items())
    bundle.close()
    audit.verify()


def finite_probes():
    """Two fixed scales only; resume complete per-probe receipts without repeats."""
    from lap_f013_local_probes import _apply, _constraints
    protocol = audit.verify()
    source = audit.geometry.b8.verify_sources()
    run = audit.geometry.b8._import_run()
    bundle = audit.TrainingBundle(run.DATA)
    dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = audit.read(run.DATA / 'mrks/dispersion.json')
    specs = audit.chemistry._selected(bundle)
    all_rows, checks = [], []
    for spec in audit.STATES:
        if spec['name'] not in ('b8_t128', 'f010_t128'):
            continue
        state, scales, names = audit.load(spec)
        model, shadow, _run = audit.existing._load_eval_model(source, spec['checkpoint'])
        shadow.load_state_dict(model.state_dict(), strict=True)
        parameters = [dict(model.named_parameters())[n] for n in names]
        base_physical = ({t: audit.read(spec['physical'].with_suffix('.json'))[t] for t in ('ae17', 'exc', 'op')}
                         if spec['physical'].suffix == '.npz' else {'ae17': 3.27546495584018, 'exc': 14.679044990002591, 'op': .030933124717599843})
        for label in ('base', 'protected'):
            delta_path = audit.OUT / f'{spec["name"]}_{label}_delta.npy'
            if not delta_path.exists():
                continue
            delta = np.load(delta_path)
            # Selected live F64 chemistry JVPs, no production model perturbation.
            check_path = audit.OUT / f'jvp_{spec["name"]}_{label}.json'
            if check_path.exists():
                checks.extend(audit.read(check_path))
            else:
                unit = delta / np.linalg.norm(delta)
                selected_indices = (0, next(i for i, r in enumerate(state['rows']) if r['database'] == 'ABDE4'))
                results = []
                for index in selected_indices:
                    reaction = audit.geometry.b8._load_reaction(bundle, 'relchem', specs[index])
                    shadow_parameters = [dict(shadow.named_parameters())[n] for n in names]
                    measured = []
                    for sign in (-1, 1):
                        with restored([shadow]):
                            _apply(shadow_parameters, sign * 1e-5 * unit)
                            with torch.no_grad():
                                p, _target = audit.chemistry._prediction(shadow, reaction, dispersion)
                            measured.append(float(p))
                    fd = (measured[1] - measured[0]) / 2e-5
                    analytic = float(state['jacobian'][index] @ unit)
                    passed = abs(fd - analytic) <= 1e-6 + 5e-3 * abs(analytic)
                    result = {'checkpoint': spec['name'], 'direction': label, 'identity': specs[index]['identity'],
                              'finite_difference': fd, 'analytic_jvp': analytic, 'absolute_gap': abs(fd - analytic), 'pass': passed}
                    assert passed, result
                    results.append(result)
                audit.write(check_path, results)
                checks.extend(results)
            for scale in (.25, 1.):
                receipt = audit.OUT / f'finite_{spec["name"]}_{label}_{scale}.json'
                if receipt.exists():
                    all_rows.extend(audit.read(receipt)['rows'])
                    continue
                began = time.perf_counter()
                with restored([model, shadow]):
                    prior_vector = np.concatenate([p.detach().cpu().double().numpy().reshape(-1) for p in parameters])
                    _apply(parameters, scale * delta)
                    shadow.load_state_dict(model.state_dict(), strict=True)
                    actual_vector = np.concatenate([p.detach().cpu().double().numpy().reshape(-1) for p in parameters]) - prior_vector
                    residuals, references = [], []
                    for chosen in specs:
                        reaction = audit.geometry.b8._load_reaction(bundle, 'relchem', chosen)
                        with torch.no_grad():
                            pred, ref = audit.chemistry._prediction(shadow, reaction, dispersion)
                        residuals.append(float(pred - ref))
                        references.append(float(ref))
                        del reaction, pred, ref
                    ae = audit.existing.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
                    physical = audit.existing.mrks_scalars(model, bundle, mrks_dispersion)
                    constraints = _constraints(model)
                    # Existing Lap descriptor anchors, all evaluated without changing descriptors in production.
                    from test_lap import raw_grid
                    with torch.no_grad():
                        raw = raw_grid().cuda().double()
                        d = shadow.get_correlation_descriptors(raw)
                        ex = shadow.get_exchange_descriptors(raw * raw.new_tensor([2, 2, 4, 4, 4, 2, 2, 2, 2]))
                        ue, ux = shadow.all_sigma_zero(d), shadow.all_sigma_zero(ex)
                        anchors = [shadow.forward_descriptors(ue, ux), shadow.forward_descriptors(shadow.all_rho_inf(d), ex),
                                   shadow.forward_descriptors(shadow.all_s_inf(d), ex)]
                        expected_scale = NN_OUTPUT_SCALE_PBE.to(raw)
                        constraints['ueg_beta_gap'] = float((anchors[0][:, 0] / expected_scale[0, 0] - 1).abs().max())
                        constraints['high_density_gamma_gap'] = float((anchors[1][:, 1] / expected_scale[0, 1] - 1).abs().max())
                        constraints['gc_anchor_gap'] = max(float((v[:, 28] - 1).abs().max()) for v in anchors)
                    assert constraints['output_finite']
                    assert all(v <= protocol['tolerances']['anchor_absolute'] for k, v in constraints.items() if k.endswith(('gap', 'abs'))), constraints
                    e = np.array(state['residuals'])
                    predicted_e = e + state['jacobian'] @ actual_vector
                    values = {'jw': audit.chemistry.j251_wtmad_style(residuals, state['databases'], scales), 'ae17': ae, **physical}
                    base_values = {'jw': state['jw_scalar'], **base_physical}
                    results = []
                    for task in ('jw', 'ae17', 'exc', 'op'):
                        predicted_change = float(state['components'][task] @ actual_vector)
                        actual_change = values[task] - base_values[task]
                        results.append({'checkpoint': spec['name'], 'direction': label, 'scale': scale, 'metric': task,
                                        'baseline': base_values[task], 'observed': values[task], 'predicted_change': predicted_change,
                                        'actual_change': actual_change, 'sign_reversal': predicted_change * actual_change < 0,
                                        'requested_step_norm': scale * np.linalg.norm(delta), 'actual_F32_step_norm': float(np.linalg.norm(actual_vector)),
                                        'restored': True})
                    for db in sorted(scales):
                        mask = np.array(state['databases']) == db
                        for metric in ('mae', 'rmse'):
                            old, pred, observed = audit.stats(e[mask])[metric], audit.stats(predicted_e[mask])[metric], audit.stats(np.array(residuals)[mask])[metric]
                            results.append({'checkpoint': spec['name'], 'direction': label, 'scale': scale, 'metric': f'{db}_{metric}',
                                            'baseline': old, 'observed': observed, 'predicted_change': pred - old, 'actual_change': observed - old,
                                            'sign_reversal': (pred - old) * (observed - old) < 0,
                                            'requested_step_norm': scale * np.linalg.norm(delta), 'actual_F32_step_norm': float(np.linalg.norm(actual_vector)), 'restored': True})
                audit.write(receipt, {'rows': results, 'constraints': constraints, 'restoration_exact': True,
                                     'parameters_buffers_rng_restored': True, 'seconds': time.perf_counter() - began})
                all_rows.extend(results)
                print('F015_FINITE', spec['name'], label, scale, 'seconds', time.perf_counter() - began, flush=True)
        del model, shadow
        torch.cuda.empty_cache()
    audit.table('lap_f015_finite_step_validation.csv', all_rows)
    audit.write(audit.OUT / 'jvp_checks.json', checks)
    bundle.close()
    audit.verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('chemistry', 'descriptors', 'finite', 'atoms'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'chemistry': reaction_rows, 'descriptors': descriptor_audit, 'finite': finite_probes,
     'atoms': atomic_comparison}[args.stage]()
