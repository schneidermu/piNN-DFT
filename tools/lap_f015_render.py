"""Receipt-only F015 summaries/plots; never calls a model or optimizer."""
import argparse
import csv
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lap_f015_audit as audit
import matplotlib
import numpy as np
from scipy.linalg import svd
from scipy.spatial import cKDTree

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(name):
    external = audit.OUT / name.replace('.csv', '_full.csv')
    path = external if external.exists() else audit.ROOT / name
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    if w.sum() <= 0:
        return None
    return float(v[min(np.searchsorted(np.cumsum(w), fraction * w.sum()), len(v) - 1)])


def prepare():
    """Additional density-stratified reductions of already computed samples."""
    audit.verify()
    sample_files = sorted(audit.OUT.glob('descriptor_*.npz'))
    samples = []
    for path in sample_files:
        receipt = audit.read(path.with_suffix('.json'))
        assert audit.sha(path) == receipt['sample_sha256']
        with np.load(path) as data:
            samples.append({k: data[k] for k in data.files})
    sensitivities = []
    for model in ('b8_t128', 's5'):
        for channel in ('GNN_up', 'Gc', 'beta', 'gamma', 'mu_up'):
            for derivative in ('dq', 'tau_relative'):
                value = np.concatenate([np.linalg.norm(s[f'{model}_{channel}_{derivative}'], axis=1) for s in samples])
                weight = np.concatenate([s['electron_weight'] for s in samples])
                rho = np.concatenate([s['rho_total'] for s in samples])
                saturated = np.concatenate([s['q_saturated'] for s in samples])
                for region, mask in (('all', np.ones(len(value), dtype=bool)), ('rho_ge_1e-3', rho >= 1e-3), ('q99', saturated)):
                    w, v = weight[mask], value[mask]
                    sensitivities.append({'model': model, 'channel': channel, 'derivative': derivative, 'region': region,
                                          'n_samples': len(v), 'electron_mean_norm': float(w @ v / w.sum()),
                                          'electron_median_norm': weighted_quantile(v, w, .5),
                                          'electron_p90_norm': weighted_quantile(v, w, .9)})
    audit.table('lap_f015_pooled_sensitivities.csv', sensitivities)
    desc = np.concatenate([s['lap_descriptors'] for s in samples])
    tau = np.concatenate([s['tau'] for s in samples])
    rho = np.concatenate([s['rho_total'] for s in samples])
    weight = np.concatenate([s['electron_weight'] for s in samples])
    information = []
    for density_floor in (1e-6, 1e-3):
        eligible = np.flatnonzero(rho >= density_floor)
        indices = eligible[np.linspace(0, len(eligible) - 1, min(len(eligible), 50000), dtype=int)]
        distance, neighbour = cKDTree(desc[indices]).query(desc[indices], k=2, workers=1)
        denominator = .3 * (3 * np.pi**2)**(2/3) * (rho[indices] + 1e-10)**(5/3)
        differences = np.linalg.norm(tau[indices] - tau[indices[neighbour[:, 1]]], axis=1) / np.maximum(denominator, 1e-12)
        for bound in (.001, .01, .05):
            mask = distance[:, 1] <= bound
            w, v = weight[indices][mask], differences[mask]
            information.append({'density_floor': density_floor, 'descriptor_distance_max': bound, 'sample_count': len(indices),
                                'pair_count': int(mask.sum()), 'median_normalized_tau_difference': float(np.median(v)),
                                'p90_normalized_tau_difference': float(np.quantile(v, .9)),
                                'electron_median_normalized_tau_difference': weighted_quantile(v, w, .5),
                                'electron_fraction_tau_difference_gt_0_1': float(w[v > .1].sum() / w.sum()), 'TF_floor': 1e-12})
    audit.table('lap_f015_tau_lapl_information_stratified.csv', information)
    del samples
    # Denominators only: do not repeat descriptor/model evaluations.
    bundle = audit.TrainingBundle(audit.ROOT.parent / 'publication_dataset_v1')
    grids = {}
    for spec in audit.chemistry._selected(bundle):
        for ref in bundle.reactions[spec['identity']]['variants'][spec['variant']]:
            grids.setdefault(ref['grid_id'], ref)
    mass = {}
    for key, ref in grids.items():
        group = bundle.handles.open(ref['shard'])[ref['group']]
        positive = np.maximum(group['Weights'][...], 0).astype(np.float64)
        rho = group['Grid'][:, :2].astype(np.float64).sum(-1)
        for name, mask in (('all', np.ones(len(rho), dtype=bool)), ('rho_lt_1e-6', rho < 1e-6),
                           ('rho_1e-6_to_1e-3', (rho >= 1e-6) & (rho < 1e-3)), ('rho_ge_1e-3', rho >= 1e-3)):
            mass[key, name] = (positive[mask].sum(), (positive[mask] * np.maximum(rho[mask], 0)).sum())
    distribution = rows('lap_f015_descriptor_distributions.csv')
    pooled = []
    for path in ('exchange', 'correlation'):
        for field in ('n_up', 'n_down', 's_up', 's_total', 's_down', 'q_up', 'q_down'):
            for regime in ('all', 'rho_ge_1e-3'):
                chosen = [r for r in distribution if r['pathway'] == path and r['field'] == field and r['density_regime'] == regime]
                for threshold in (.95, .99, .999):
                    n = sum(int(r['n_points']) for r in chosen)
                    positive_mass = sum(mass[r['grid_id'], regime][0] for r in chosen)
                    electron_mass = sum(mass[r['grid_id'], regime][1] for r in chosen)
                    pooled.append({'pathway': path, 'field': field, 'density_regime': regime, 'threshold': threshold,
                                   'n_points': n,
                                   'point_fraction': sum(int(r['n_points']) * float(r[f'point_fraction_gt_{threshold}']) for r in chosen) / n,
                                   'positive_weight_fraction': sum(mass[r['grid_id'], regime][0] * float(r[f'positive_weight_fraction_gt_{threshold}'] or 0) for r in chosen) / positive_mass,
                                   'electron_fraction': sum(mass[r['grid_id'], regime][1] * float(r[f'electron_fraction_gt_{threshold}'] or 0) for r in chosen) / electron_mass})
    audit.table('lap_f015_pooled_descriptor_distributions.csv', pooled)
    bundle.close()
    # Model-to-model species/reaction cancellation, never exact XC error.
    decomposition = rows('lap_f015_species_xc_decomposition.csv')
    grouped = {}
    for row in decomposition:
        key = (row['identity'], row['component_index'])
        grouped.setdefault(key, {})[row['model']] = row
    cancellation = []
    for model in ('b8_t128', 'f010_t128', 'f010_t251', 'f012_t128', 'f012_t251'):
        reaction = {}
        for (identity, _index), entry in grouped.items():
            a, b = entry[model], entry['s5']
            assert a['fixed_nonxc_hartree'] == b['fixed_nonxc_hartree'] and a['dispersion_hartree'] == b['dispersion_hartree']
            factor = float(a['coefficient']) * 627.5095
            dx, dc = factor * (float(a['ex_hartree']) - float(b['ex_hartree'])), factor * (float(a['ec_hartree']) - float(b['ec_hartree']))
            reaction.setdefault(identity, []).append((dx, dc))
        for identity, pieces in reaction.items():
            ex, ec = np.sum(pieces, axis=0)
            absolute_sum = float(np.abs(np.sum(pieces, axis=1)).sum())
            cancellation.append({'identity': identity, 'model': model, 'delta_ex_kcal': float(ex), 'delta_ec_kcal': float(ec),
                                 'delta_exc_kcal': float(ex + ec), 'sum_abs_stoichiometric_species_delta': absolute_sum,
                                 'cancellation_ratio': float(abs(ex + ec) / max(absolute_sum, 1e-12)),
                                 'denominator_floor_kcal': 1e-12})
    audit.table('lap_f015_reaction_xc_cancellation.csv', cancellation)
    s5 = {r['identity']: r['residual'] for r in audit.read(audit.OUT / 'matched_s5_rows.json')}
    uncertainty = []
    for spec in audit.STATES:
        if spec['name'] not in audit.LOCAL:
            continue
        state, _scales, _names = audit.load(spec)
        u, singular, _vt = svd(state['jacobian'], full_matrices=False)
        target = np.array([s5[r['identity']] for r in state['rows']]) - np.array(state['residuals'])
        for threshold in (1e-6, 1e-8, 1e-10):
            kept = singular >= singular[0] * threshold
            coefficients = u[:, kept].T @ target
            displacement = float(np.linalg.norm(coefficients / singular[kept]))
            uncertainty.append({'checkpoint': spec['name'], 'relative_threshold': threshold, 'rank': int(kept.sum()),
                                's5_projection_squared_fraction': float(coefficients @ coefficients / (target @ target)),
                                's5_minimum_norm_displacement': displacement, 'displacement_over_radius': displacement / audit.RADIUS,
                                'resolved_condition': float(singular[0] / singular[kept][-1])})
    audit.table('lap_f015_rank_threshold_sensitivity.csv', uncertainty)
    print('F015_REDUCTIONS_COMPLETE', len(grids), flush=True)


def figure(name):
    plt.tight_layout()
    plt.savefig(audit.ROOT / name, dpi=150)
    plt.close()


def plots():
    reactions = rows('lap_f015_matched_reaction_errors.csv')
    s5 = {r['identity']: float(r['residual']) for r in reactions if r['model'] == 's5'}
    plt.figure(figsize=(6, 5))
    for model in ('b8_t128', 'f010_t251'):
        chosen = [r for r in reactions if r['model'] == model]
        plt.scatter([s5[r['identity']] for r in chosen], [float(r['residual']) for r in chosen], s=10, alpha=.6, label=model)
    plt.axline((0, 0), slope=1, color='grey', linewidth=.6)
    plt.xlabel('S5 signed error (kcal/mol)'); plt.ylabel('Lap signed error (kcal/mol)'); plt.legend()
    figure('lap_f015_signed_errors.png')
    comparison = rows('lap_f015_database_comparison.csv')
    dbs = sorted({r['database'] for r in comparison if r['database'] != 'ALL_REACTION_WEIGHTED'})
    plt.figure(figsize=(10, 5)); x = np.arange(len(dbs))
    for i, model in enumerate(('s5', 'b8_t128', 'f010_t251')):
        values = {r['database']: float(r['mae']) for r in comparison if r['model'] == model}
        plt.bar(x + .25 * (i - 1), [values[d] for d in dbs], width=.25, label=model)
    plt.xticks(x, dbs); plt.ylabel('Database MAE (kcal/mol)'); plt.legend()
    figure('lap_f015_database_mae.png')
    plt.figure(figsize=(7, 4))
    for model in ('b8_t128', 'f010_t251'):
        plt.hist([float(r['s5_advantage_absolute_error']) for r in reactions if r['model'] == model], bins=30, alpha=.5, label=model)
    plt.xlabel('|Lap error| minus |S5 error| (kcal/mol)'); plt.ylabel('Reactions'); plt.legend()
    figure('lap_f015_advantage_histogram.png')
    distribution = rows('lap_f015_pooled_descriptor_distributions.csv')
    plt.figure(figsize=(7, 4))
    for field in ('q_up', 'q_down'):
        selected = [r for r in distribution if r['pathway'] == 'correlation' and r['field'] == field and r['density_regime'] == 'all']
        plt.plot([float(r['threshold']) for r in selected], [float(r['electron_fraction']) for r in selected], 'o-', label=field)
    plt.xlabel('|tanh(q)| threshold'); plt.ylabel('Pooled electron-weighted saturated fraction'); plt.legend()
    figure('lap_f015_q_saturation.png')
    sensitivity = rows('lap_f015_pooled_sensitivities.csv')
    plt.figure(figsize=(7, 4))
    selected = [r for r in sensitivity if r['region'] == 'rho_ge_1e-3' and r['channel'] in ('GNN_up', 'Gc')]
    labels = [f"{r['model']} {r['channel']} {r['derivative']}" for r in selected]
    for field, marker, label in (('electron_median_norm', 'o', 'Median'), ('electron_p90_norm', '^', 'P90'), ('electron_mean_norm', 's', 'Mean')):
        plt.scatter(np.arange(len(selected)), [float(r[field]) for r in selected], marker=marker, label=label)
    plt.xticks(np.arange(len(selected)), labels, rotation=65, ha='right'); plt.ylabel('Electron-weighted sensitivity norm'); plt.legend()
    figure('lap_f015_sensitivities.png')
    cancellation = rows('lap_f015_reaction_xc_cancellation.csv')
    plt.figure(figsize=(6, 5))
    selected = [r for r in cancellation if r['model'] == 'b8_t128']
    plt.scatter([float(r['delta_ex_kcal']) for r in selected], [float(r['delta_ec_kcal']) for r in selected], s=12)
    plt.xlabel('Reaction exchange difference Lap minus S5 (kcal/mol)'); plt.ylabel('Correlation difference (kcal/mol)')
    figure('lap_f015_xc_decomposition.png')
    rank = audit.read(audit.ROOT / 'lap_f015_jacobian_rank_and_projection.json')
    plt.figure(figsize=(7, 4))
    for name, data in rank.items():
        plt.semilogy(np.arange(1, 252), data['singular_values'], label=name)
    plt.xlabel('Singular mode'); plt.ylabel('sigma (kcal/mol per parameter)'); plt.legend()
    figure('lap_f015_singular_spectrum.png')
    local = rows('lap_f015_database_protection.csv')
    plt.figure(figsize=(8, 4))
    labels = [r['checkpoint'] + ' ' + r['direction'] for r in local]
    plt.bar(np.arange(len(local)), [100 * (1 - float(r['weighted_ls_linear']) / float(r['weighted_ls_before'])) for r in local])
    plt.xticks(np.arange(len(local)), labels, rotation=60, ha='right'); plt.ylabel('Predicted diagnostic weighted-LS reduction (%)')
    figure('lap_f015_local_improvement.png')
    finite = rows('lap_f015_finite_step_validation.csv')
    _fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, task in zip(axes.flat, ('jw', 'ae17', 'exc', 'op'), strict=True):
        selected = [r for r in finite if r['metric'] == task]
        for state in ('b8_t128', 'f010_t128'):
            for direction in ('base', 'protected'):
                group = [r for r in selected if r['checkpoint'] == state and r['direction'] == direction]
                ax.scatter([float(r['predicted_change']) for r in group], [float(r['actual_change']) for r in group],
                           marker='o' if direction == 'base' else '^', label=state + ' ' + direction)
        ax.axhline(0, linewidth=.5, color='grey'); ax.axvline(0, linewidth=.5, color='grey')
        ax.set_title(task); ax.set_xlabel('Predicted change'); ax.set_ylabel('Actual change')
        ax.legend(fontsize=5)
        largest = max(abs(float(r['predicted_change'])) for r in selected)
        near = [r for r in selected if abs(float(r['predicted_change'])) < .005 * largest]
        if task != 'jw' and len(near) > 1:
            zoom = ax.inset_axes([.55, .1, .4, .3])
            zoom.scatter([float(r['predicted_change']) for r in near], [float(r['actual_change']) for r in near], s=12)
            zoom.axhline(0, color='grey', linewidth=.5); zoom.axvline(0, color='grey', linewidth=.5)
            zoom.tick_params(labelsize=5); zoom.set_title('Near-zero slopes', fontsize=6)
    figure('lap_f015_finite_changes.png')


def provenance():
    audit.verify()
    artifacts = {p.name: audit.sha(p) for p in audit.ROOT.glob('lap_f015_*') if p.is_file() and p.name != 'lap_f015_provenance.json'}
    external = {p.name: audit.sha(p) for p in audit.OUT.glob('*') if p.is_file()}
    tools = {p.name: audit.sha(p) for p in (audit.ROOT / 'tools').glob('lap_f015_*.py')}
    audit.write(audit.ROOT / 'lap_f015_provenance.json', {'artifact_sha256': artifacts, 'external_sha256': external,
                'diagnostic_source_sha256': tools, 'python': platform.python_version(), 'numpy': np.__version__,
                'torch': audit.torch.__version__, 'cuda': audit.torch.version.cuda, 'device': audit.torch.cuda.get_device_name(),
                'protocol_sha256': audit.sha(audit.ROOT / 'lap_f015_protocol.json'), 'optimizer_updates': 0,
                'historical_inputs_unchanged': True, 'production_sources_unchanged': True})


def rounding_check():
    """Independent CPU replay of F32 addition only, not an optimizer update."""
    checks = []
    for path in sorted(audit.OUT.glob('finite_*.json')):
        receipt = audit.read(path)
        first = receipt['rows'][0]
        spec = next(s for s in audit.STATES if s['name'] == first['checkpoint'])
        state, _scales, names = audit.load(spec)
        saved = audit.torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        source = np.concatenate([saved['model'][n].numpy().reshape(-1) for n in names])
        intended = float(first['scale']) * np.load(audit.OUT / f'{spec["name"]}_{first["direction"]}_delta.npy')
        actual = (source + intended.astype(np.float32)).astype(np.float64) - source.astype(np.float64)
        assert abs(np.linalg.norm(actual) - first['actual_F32_step_norm']) < 1e-12
        for row in receipt['rows'][:4]:
            task = row['metric']
            predicted = float(state['components'][task] @ actual)
            assert abs(predicted - row['predicted_change']) < 1e-10
            checks.append({'checkpoint': spec['name'], 'direction': first['direction'], 'scale': first['scale'],
                           'task': task, 'intended_F64_derivative': float(state['components'][task] @ intended),
                           'rounded_F32_derivative': predicted, 'actual_objective_change': row['actual_change'],
                           'relative_rounding_error_norm': float(np.linalg.norm(actual - intended) / np.linalg.norm(intended)),
                           'actual_step_norm': float(np.linalg.norm(actual)), 'independent_vector_check_pass': True})
    audit.table('lap_f015_f32_rounding.csv', checks)


def compact():
    """Keep full per-grid tables externally and commit compact stratified tables."""
    for name in ('lap_f015_descriptor_distributions.csv', 'lap_f015_descriptor_sensitivities.csv'):
        external = audit.OUT / name.replace('.csv', '_full.csv')
        if not external.exists():
            (audit.ROOT / name).replace(external)
    pooled = rows('lap_f015_pooled_descriptor_distributions.csv')
    audit.table('lap_f015_descriptor_distributions.csv', pooled)
    full = rows('lap_f015_descriptor_sensitivities.csv')
    grouped = {}
    for row in full:
        key = (row['model'], row['channel'], row['physical_input'], row['regime'])
        grouped.setdefault(key, []).append(row)
    reduced = []
    for (model, channel, physical, regime), group in sorted(grouped.items()):
        values = [float(r['electron_mean_norm']) for r in group if r['electron_mean_norm']]
        reduced.append({'model': model, 'channel': channel, 'physical_input': physical, 'regime': regime,
                        'grid_count': len(group), 'n_stride_samples': sum(int(r['n_points']) for r in group),
                        'equal_grid_mean_of_electron_weighted_norm': float(np.mean(values)) if values else None,
                        'median_of_per_grid_sample_medians': float(np.median([float(r['median_norm']) for r in group]))})
    audit.table('lap_f015_descriptor_sensitivities.csv', reduced)


def report():
    """Numbers come from generated tables; no additional scientific evaluation."""
    from train_models.optuna_joint import FCHEM_DB_WEIGHTS

    audit.verify()
    comp = rows('lap_f015_database_comparison.csv')
    overall = {r['model']: r for r in comp if r['database'] == 'ALL_REACTION_WEIGHTED'}
    atom = rows('lap_f015_matched_ae17_errors.csv')
    finite = rows('lap_f015_finite_step_validation.csv')
    assert len(finite) == 160 and len(rows('lap_f015_f32_rounding.csv')) == 32
    input_receipt = audit.read(audit.OUT / 'input_parity.json')
    spatial = rows('lap_f015_spatial_energy_differences.csv')
    regions = {}
    for r in spatial:
        regions[r['identity']] = regions.get(r['identity'], 0.) + float(r['coefficient']) * (
            float(r['signed_delta_ex_saturated']) + float(r['signed_delta_ec_saturated'])) * 627.5095
    region_rows = []
    for r in rows('lap_f015_reaction_xc_cancellation.csv'):
        if r['model'] == 'b8_t128':
            saturated, total = regions[r['identity']], float(r['delta_exc_kcal'])
            region_rows.append({'identity': r['identity'], 'pair': 'B8_minus_S5', 'correlation_q99_delta_exc_kcal': saturated,
                                'non_q99_delta_exc_kcal': total - saturated, 'total_delta_exc_kcal': total})
    audit.table('lap_f015_region_reaction_decomposition.csv', region_rows)
    region_ratio = sum(abs(r['correlation_q99_delta_exc_kcal']) for r in region_rows) / sum(abs(r['total_delta_exc_kcal']) for r in region_rows)
    fd = audit.read(audit.OUT / 'descriptor_fd.json')
    jvp = audit.read(audit.OUT / 'jvp_checks.json')
    assert len(jvp) == 8 and all(r['pass'] for r in jvp + fd)
    assert input_receipt['all_model_states_unchanged']
    assert all(audit.read(p)['parameters_buffers_rng_restored'] for p in audit.OUT.glob('finite_*.json'))
    decision = {'primary_classification': 'A', 'response_status': 'PARTIAL',
                'stage_A': 'PASS numerical-input matching; BLOCKED independent original basis/geometry/DM reconstruction',
                'stage_B': 'PASS physical-grid sensitivity and information diagnostics; no causal architecture ablation',
                'stage_C': 'PASS numerical/KKT/JVP/restoration qualification; mixed finite scientific progress',
                'geometry_and_spatial_mapping': 'BLOCKED: original component HDF5 files/coordinates are unavailable',
                'strongest_supported_mechanism': 'Local aggregate chemistry trade-offs/direction geometry, with conditioning, rounding and curvature limitations',
                'tau_causes_chemical_advantage': 'NOT ESTABLISHED', 'global_Lap_architecture_incapacity': 'NOT ESTABLISHED',
                'next_experiment': 'One read-only energy-consistent S5 generalized-KS AO-operator audit on the historical qualified15, including tau and Laplacian terms, subject to exact frozen density/orbital/tau provenance. No training, SCF, new references or partial-density-derivative proxy.',
                'optimizer_updates': 0, 'training': False, 'development_metrics_used': False,
                'historical_checkpoints_and_production_sources_unchanged': True}
    audit.write(audit.ROOT / 'lap_f015_decision.json', decision)
    lines = ['# F015 — S5-to-Lap chemistry, information and local attainability', '',
             '## 1. Executive scientific conclusion', '',
             '**PARTIAL — completed with documented evidence limitations. Primary classification A: local chemistry-objective/optimization bottleneck supported.**', '',
             'B8 admits an executed-in-memory, historical-radius direction that lowers JW, AE17, Exc and the operator. Its aggregate chemistry gain sacrifices the four historically problematic databases. Protecting them sharply reduces progress and leaves small finite physical regressions. Thus aggregate progress does not imply balanced chemistry. This supports an optimization/objective-geometry mechanism locally; it does not establish that this mechanism explains the entire S5 gap.', '',
             'S5 actively uses tau in both exchange-output and correlation-output pathways. Similar bounded Lap descriptors sometimes conceal substantial tau variation. This is measurable additional information, not a controlled demonstration that tau causes better chemical accuracy. The model-to-model comparison is confounded by weights, activation/anchors, training history/budget and potential supervision. No global Lap expressivity failure is established.', '',
             '## 2. Provenance and comparability', '',
             f'Exact S5 selected checkpoint verified: `{audit.S5_SHA}`; strict9542-parameter load. All five9446-parameter Lap checkpoints and reaction/physical-gradient caches matched F013 SHA receipts. Source revision: `{audit.read(audit.ROOT / "lap_f015_protocol.json")["git_head"]}` (F014 preserved).', '',
             '**251 canonical reaction identities matched exactly** to the accepted S5 evaluation population. Same source-group hash and exact equality of Grid (including original tau), densities, gradient invariants, signed weights, fixed non-XC/HF terms, coefficients, targets and splits against the accepted legacy Minnesota store. All models consume the same numerical inputs/corrections. Only27 identities share the old S5 audit variant, so historical S5 MAE6.11594 was not reused as a current baseline.', '',
             f'Independent qualified S5 forward replay across all8 databases: maximum error-reconstruction gap {max(r["absolute_gap"] for r in audit.read(audit.OUT / "s5_qualified_forward_parity.json")):.3e} kcal/mol. All1255 Lap reaction predictions match their cached residuals within the predeclared1e-6 kcal/mol tolerance. Canonical parameter order is sorted unique names, not state_dict traversal; exact order/JW/eight database gradient checks passed.', '',
             '**Limitation:** original DM, basis/charge/spin/geometry and spatial coordinates are not embedded in these grouped reaction records. For example `C:/Dev/ML-DFT/piNN-DFT/train_models/data/C2H6_abde4__level2.h5` is missing; the accepted grouped pickle is present but has only numerical reaction fields. Independent original-molecule reconstruction and genuine3D spatial maps are BLOCKED. This is a numerical-input-matched comparison, not a newly geometry/DM-certified benchmark. The S5 historical training-corpus identity is also unavailable.', '',
             'Precision: source/checkpointF32 values widened to matchedF64 chemistry; native F64 dispersion scalars retained. F64 diagnostic algebra/input derivatives; actual finite evaluations retain F32 native parameters, repaired F64 learned operator branch, F32 PBE branch and F64 AO assembly. Widening does not recover lost source precision. Every chemistry evaluation uses exactly one fixed variant per identity. No exhaustive variant sweep.', '',
             '## 3. S5 versus Lap chemistry', '',
             'Reaction errors are kcal/mol. JW is `(1/251)sum |e_r|/A_d`, a training-only scalar; it is neither the legacy singleton mean nor GMTKN55 WTMAD-2. Eight-DB weighted RMSE below uses existing fixed `FCHEM_DB_WEIGHTS`, without renormalization.', '',
             '| Model | MAE | RMSE | Median absolute error | JW | Legacy singleton mean | Eight-DB weighted RMSE | S5 wins/losses |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in ('s5', 'b8_t128', 'f010_t128', 'f010_t251', 'f012_t128', 'f012_t251'):
        r = overall[model]
        weighted = sum(FCHEM_DB_WEIGHTS[x['database']] * float(x['rmse']) for x in comp if x['model'] == model and x['database'] != 'ALL_REACTION_WEIGHTED')
        lines.append(f'| {model} | {float(r["mae"]):.6f} | {float(r["rmse"]):.6f} | {float(r["median_abs"]):.6f} | {float(r["jw_contribution"]):.9f} | {float(r["legacy_singleton_mean"]):.9f} | {weighted:.6f} | {r["s5_wins"]}/{r["s5_losses"]} |')
    lines += ['', '| Database | n | S5 MAE | B8 MAE | F010/t251 MAE |', '|---|---:|---:|---:|---:|']
    dbs = sorted({r['database'] for r in comp if r['database'] != 'ALL_REACTION_WEIGHTED'})
    for db in dbs:
        g = {r['model']: r for r in comp if r['database'] == db}
        lines.append(f'| {db} | {g["s5"]["n"]} | {float(g["s5"]["mae"]):.6f} | {float(g["b8_t128"]["mae"]):.6f} | {float(g["f010_t251"]["mae"]):.6f} |')
    lines += ['', 'S5 improves the B8 reaction-weighted MAE by2.418883 kcal/mol (28.34%), winning187/251 reactions. DBH76 and MGAE109 contribute about97.5% of that population-weighted gap; this is a contribution accounting, not a statement that larger databases are scientifically more important. Macro-average database MAE: S5=3.759863, B8=4.916307, F010/t251=5.168453. F010 improves population MAE while worsening this macro summary. At F010/t251 S5 still wins147/251 and has lower median error, but F010 is better on MGAE109 and IP13. S5 is not uniformly superior in every database; at B8 it is worse on EA13.', '',
             'The signed/absolute distributions, p90/p95, error correlations, paired database wins and each reaction JW contribution are in the CSVs. Largest-error contributions can be reconstructed by sorting the recorded contributions; no reaction was discarded.', '',
             '| Model | Matched AE17 atomic MAE (17, separate population) |', '|---|---:|']
    for model in overall:
        mean = np.mean([float(r['absolute_error_kcal']) for r in atom if r['model'] == model])
        lines.append(f'| {model} | {mean:.6f} |')
    lines += ['', 'These source AE17 records have one coefficient+1 component and supervise absolute atomic energies, not relative reaction cancellation. Lap is substantially better on this atomic population even when S5 is better on relchem. Exact integrated XC reference energies for the Minnesota species are not available: per-species numbers are model-to-model XC differences, not exact XC errors. Fixed non-XC and dispersion differences are zero. Median reaction cancellation ratio `|sum coef*deltaExc|/sum |coef*deltaExc|` is0.035123 for B8/S5 and0.072919 for F010/t251/S5 (denominator floor1e-12 kcal/mol). Large individual model-energy differences cancel strongly; this does not prove either model has more accurate individual molecular XC energies.', '',
             '## 4. Descriptor information and sensitivity', '',
             'Full selected population:495 unique grids,23,231,999 points. Stride101 physical samples are used for sensitivities, never substituted for full energy integrals. Signed quadrature is preserved in energies; statistics use max(w,0) and max(w,0)*max(rho_total,0). Full per-grid/database/species statistics remain externally SHA-bound; compact and pooled tables are committed.', '',
             'Exact Lap formulas: `n_sigma=(rho_sigma+1e-10)^(1/3)`; `s_i=(sqrt(sigma_i+1e-30)-sqrt(1e-30))/(2*(3*pi^2)^(1/3)*(rho_i+1e-10)^(4/3))`; `q_sigma=lap_sigma/(4*(3*pi^2)^(2/3)*(rho_sigma+1e-10)^(5/3))`; zeta=(rho_a-rho_b)/(rho_total+1e-10). tanh applies to n/s/q, not zeta. Exchange uses spin-scaled rho*2, sigma*4, lap*2 and consumes only(s_sigma,q_sigma). Correlation consumes all8 descriptors plus fused exchange features.', '',
             'S5 uses10 correlation descriptors and exchange(s_sigma,alpha_sigma,q_sigma). It does not subtract the zero-sigma square-root regularizer in s. Its `alpha=clamp((tau-tauW)/tauTF-1,min=-1)` with tauW=sigma/(8*(rho+EPS_RHO)), C_TF=(3/10)*(3*pi^2)^(2/3). Correlation tauTF=0.5*C_TF*(2*rho_sigma+EPS_RHO)^(5/3); exchange tauTF=C_TF*(rho_scaled+EPS_RHO)^(5/3)+EPS_RHO, after rho/tau/lap*2 and sigma*4. Its q denominator matches the corresponding raw/spin-scaled convention. n/s/alpha/q are tanh-bounded and zeta is unchanged. These exact transformations are retained from NN_models.py. Lap never reads the tau columns. Gc activation differs too: Lap shiftedELU(z-1), S5 1-|z-1|. No transform or activation was replaced.', '',
             'Correlation q_up saturation at |tanh(q)|>.95/.99/.999: unweighted51.84%/46.24%/42.27%, electron-weighted12.35%/6.15%/3.66%. Thus large point fractions are dominated by weak-density/volume regions. At rho>=1e-3, q_up>.99 covers5.40% of electrons. Any-spin correlation-q99 regions account for5.15% of absolute stoichiometry-weighted B8/S5 grid XC-difference mass. This gross fraction must NOT be used to dismiss chemical relevance: after signed integration and reaction cancellation, the sum of absolute q99 reaction contributions is ' + f'{region_ratio:.6f}' + ' times the sum of absolute total model-to-model reaction differences; their correlation is-0.586638. Saturated and nonsaturated contributions strongly cancel. The signed region decomposition is committed. This establishes important cancellation, not a causal saturation bottleneck. Actual tanh slopes, other generated density/gradient descriptors and both spin-scaled pathways are tabulated; generated n/s_total descriptors in the exchange path are not mistaken for consumed exchange inputs.', '',
             '| Output channel, rho>=1e-3 | Lap mean dq norm | S5 mean dq norm | Lap mean tau-relative norm | S5 mean tau-relative norm |',
             '|---|---:|---:|---:|---:|', '| GNN_up (exchange output) | 0.015171 | 0.007900 | 0 | 0.053585 |', '| Gc (correlation output) | 0.035987 | 0.031029 | 0 | 0.100945 |', '',
             'All reported dq sensitivities use the canonical unscaled physical q_sigma=lap_sigma/[4*(3*pi^2)^(2/3)*(rho_sigma+EPS_RHO)^(5/3)], holding other raw inputs fixed. Exchange heads internally consume the spin-scaled q and their chain rule is included; these are not derivatives solely with respect to neural q_exchange while holding fused pathways fixed. Outputs are normalized by NN_OUTPUT_SCALE_PBE (GNN remains additive with unit scale). Tau-relative sensitivity is the norm of tau_sigma*dG/dtau_sigma; physical tau derivatives have inverse kinetic-density units and are separately tabulated. These are output-channel sensitivities, not isolated parameter-block gradients; normalization does not make their magnitudes a causal importance ratio. All24 selected input FD checks passed; Lap tau derivatives are exactly zero. Other learned beta/gamma/mu/kappa/spin outputs are retained in the full tables.', '',
             'For50,000 deterministic physical samples at rho>=1e-3,9682 nearest-neighbour pairs have bounded8-dimensional Lap descriptor distance<=0.001. Their median normalized tau difference is0.005882, electron-weighted median0.000221, and9.15% of paired electron weight has normalized tau difference>0.1. At distance<=0.01 the corresponding tail is18.07%. Normalization is total TF tau with floor1e-12. This demonstrates a nonuniform conditional information tail, not a uniformly missing tau signal or an impossibility theorem. The stricter rho>=1e-6 analysis has very large unweighted tails but small electron-weighted medians; these are not hidden.', '',
             'In q99 regions the S5 electron-weighted mean tau-relative sensitivity is0.143429 for GNN_up and0.080103 for Gc (Lap both0). Grid-region XC differences and these active channels are observational associations, not a tau-causal ablation. No off-manifold substitution was used as causal evidence; FD input perturbations are mathematical derivative checks only. Missing original geometry/coordinates prevents a verified atom/molecule spatial stratification of all relchem grids and genuine3D maps. The separate source-labelled AE17 atomic comparison does not silently relabel the mixed relchem grid pool.', '',
             '## 5. Local Jacobian geometry', '',
             'Reaction order, units(kcal/mol per parameter),251x9446 shape, finite values, sorted unique/tied parameter ordering and eight DB/JW gradient reconstruction passed. All8 selected live parameter JVPs passed against frozen-input F64 central differences. Raw AE17/Exc/operator gradients are reused without training coefficients.', '',
             '| State | Rank @1e-8 | Ranks @1e-6 /1e-10 | Condition resolved | S5 difference projected squared fraction | Minimum-norm displacement | Displacement /r |',
             '|---|---:|---:|---:|---:|---:|---:|']
    rank = audit.read(audit.ROOT / 'lap_f015_jacobian_rank_and_projection.json')
    for name, data in rank.items():
        p = data['s5_projection']
        lines.append(f'| {name} | {data["rank"]} | {data["rank_sensitivity"]["1e-06"]}/{data["rank_sensitivity"]["1e-10"]} | {data["condition_resolved"]:.3e} | {p["squared_fraction"]:.9f} | {p["minimum_norm_step"]:.3f} | {p["step_over_radius"]:.3e} |')
    lines += ['', 'Nearly all of the numerical-input-matched S5 error difference projects into the resolved column space, but exact pattern matching requires wildly impractical linearized displacements. At B8 the displacement changes from280.88 to6155.89 to8843.85 across the frozen resolution thresholds; it is not a robust practical prescription. The inverse is ill-conditioned and all values exceed the trust radius. S5 is an orientation baseline, not a ground-truth teacher. Neither this local pseudoinverse nor numerical rank proves global architecture incapacity; reproducing S5 signed errors is not required to reduce true errors.', '',
             '## 6. Physically constrained attainability', '',
             '`W_rr=1/(251*A_d)` is fixed from training references; the probe minimizes0.5 sum W*(e+J*delta)^2, not JW. Radius fixed at0.0006498670299681904 for every state. Constraints use raw g_A/g_E/g_O. The economic QR span contains every J row and constraint direction without truncation; projection into it leaves objective/constraints unchanged and cannot increase norm. This is an equivalent reduced convex problem, not a new training algorithm.', '',
             'SLSQP is independently certified by primal feasibility, nonnegative multipliers, stationarity and the convex Lagrangian dual bound. No constraint/radius was relaxed. All8 base/protected solves passed; maximum KKT residual1.30e-7, primal residual at roundoff, dual gap<4e-14. Conditioning/active constraints/multipliers are in external certificates. F64 arithmetic is used; threshold sensitivity is separately reported, not tuned.', '',
             '| State | Base predicted weighted-LS reduction | Protected reduction | Progress lost by protecting4 DBs |', '|---|---:|---:|---:|']
    protection = rows('lap_f015_database_protection.csv')
    for name in rank:
        g = {r['direction']: r for r in protection if r['checkpoint'] == name}
        base_gain = float(g['base']['weighted_ls_before']) - float(g['base']['weighted_ls_linear'])
        protected_gain = float(g['protected']['weighted_ls_before']) - float(g['protected']['weighted_ls_linear'])
        initial = float(g['base']['weighted_ls_before'])
        lines.append(f'| {name} | {100*base_gain/initial:.6f}% | {100*protected_gain/initial:.6f}% | {100*(1-protected_gain/base_gain):.3f}% |')
    lines += ['', 'All four states triggered exactly the predeclared protection solve for ABDE4/EA13/PA8/pTC13. Per-database MAE/RMSE predictions are in the tables. Protection is first-order no-worsening of the existing DB absolute-error objective (positive-scaled global contribution), not a guarantee about finite-step RMSE or every other DB. At B8 protected IP13 worsens: no extra constraint was added after inspection.', '',
             'F013 context, reused: four-task /+problem4 /all8DB+physical normalized feasibility margins: B8=.801/.065/.012; F010/t128=.110/.110/.087; F010/t251=.0081/.0033/.0033; F012/t128=.077/.077/.056. Those max-min directions and this weighted-LS direction solve different problems.', '',
             '## 7. Finite-step validation', '',
             'Exactly8 probes at B8/F010t128, base/protected,0.25r/1r; no step-size search. Predictions below use the **actual rounded F32 displacement**, independently reconstructed on CPU and checked against saved norms/dots. Intended F64 solver slopes and relative rounding errors are in lap_f015_f32_rounding.csv. Parameters, buffers, RNG, spin symmetry, passthrough and UEG/high-density/rapid-gradient anchors restored/verified after every probe. Perturbed weights were never saved.', '',
             '| State | Direction | Scale | Actual delta JW | Actual delta AE17 | Actual delta Exc | Actual delta operator | Strict all4 descent |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for name in ('b8_t128', 'f010_t128'):
        for direction in ('base', 'protected'):
            for scale in ('0.25', '1.0'):
                g = {r['metric']: r for r in finite if r['checkpoint'] == name and r['direction'] == direction and r['scale'] == scale}
                values = [float(g[t]['actual_change']) for t in ('jw', 'ae17', 'exc', 'op')]
                lines.append(f'| {name} | {direction} | {scale} | ' + ' | '.join(f'{v:+.9g}' for v in values) + f' | {all(v < 0 for v in values)} |')
    lines += ['', 'B8 base at r lowers JW by0.61437%, AE17 by0.420329, Exc by1.566145 and operator by0.000241183. ABDE4/EA13/PA8/pTC13 MAEs simultaneously rise by0.08354/0.02534/0.09558/0.14238 kcal/mol. This is real aggregate progress with systematic DB sacrifice.', '',
             'B8 protection reduces JW progress to0.000265871 while protecting the four small DBs, but operator rises by2.0388e-6 at r (0.00659% of baseline). At0.25r it already rises1.2671e-7. This is not strict finite common descent. F010 base has approximately neutral physical slopes: F32 addition can change their signs, and operator curvature remains positive. F010 protected at r lowers JW/Exc/operator but AE17 rises0.000211597 despite a rounded linear prediction-4.223e-5. That reversal is retained. The large F013 Exc reversal does not repeat at its former scale on these different QC directions; no universal curvature cure is inferred.', '',
             'All160 metric rows retain prediction/realization/sign reversals, including eight DB MAE/RMSE summaries. F32 vector rounding changes the intended displacement by0.0786–0.3242% in relative vector norm; actual radius norms differ by-4.41e-6 to+4.14e-5 relatively. No compensating rescale or feasibility repair was applied. Boundary sign changes must be distinguished from broad failures: no arbitrary tolerance converts a positive objective change into a decrease. Intended F64 feasible directions and finite native-F32 protection are separate claims.', '',
             '## 8. Supported mechanisms', '',
             '- Aggregation trades off small databases; a lower population MAE/JW can worsen macro chemistry and legacy weighted RMSE.',
             '- Useful local correction directions exist near B8 and survive finite steps; historical aggregate/AdamW directions need not use them (F013 context).',
             '- Protecting the problematic databases costs substantial progress; rounding/curvature matters near active physical constraints.',
             '- S5 uses additional tau information; bounded Lap descriptors compress some of that information, especially in a conditional tail.', '',
             '## 9. Unsupported or unresolved mechanisms', '',
             'No causal tau/activation ablation; no global Lap incapacity; no proof that saturation explains the gap; no assertion of exact individual molecular XC accuracy; no S5 physical operator score; no generalization/SCF-stability result. The full chemical gap cannot be assigned to one cause by comparing differently trained models. S5 correct energetics does not qualify its historical partial-density potential loss; a flawed potential loss does not automatically invalidate its energy functional.', '',
             '## 10. Scientific limitations', '',
             'Single Lap lineage; differing S5 architecture/weights/training budget/history; historical training-corpus identity missing; original geometry/basis/DM reconstruction blocked. Information statistics are deterministic correlated grid samples, not independent experiments. This is fixed-density training chemistry, not out-of-distribution validation. Mandatory background reports were read, including their historical development discussion; no development-score artifact entered F015 calculations or selection and no validation evaluator was called. Initial execution used the canonical bundle, which incidentally initializes validation metadata even on training access; therefore this audit does not claim that such metadata was never visible. The final isolated TrainingBundle guardrail excludes validation indices/labels and rejects validation access while retaining unchanged canonical training methods and source/shard checks. No future-test dataset or evaluation occurred.', '',
             f'Chemistry comparison recorded{input_receipt["seconds"]:.3f}s; finite probes total{sum(audit.read(p)["seconds"] for p in audit.OUT.glob("finite_*.json")):.3f}s. Every finite invocation is bounded and receipts are reused on resume. No training occurred.', '',
             '## 11. One recommended next intervention', '', decision['next_experiment'], '',
             'This is a separate, bounded physical-consistency diagnostic to determine whether S5 chemistry advantage coexists with correct operator supervision or exploits its weaker historical target. It is not a new optimizer/loss recommendation and was not started.', '',
             '## 12. Reproducibility and artifact hashes', '',
             'Run from lap_full_vxc with the qualified libxc Python environment: audit freeze; inputs chemistry; audit local/check; inputs descriptors/finite/atoms; render prepare/rounding/compact/plots/provenance. Frozen protocol is immutable; completed reaction/grid/finite receipts are reused. Complete per-grid tables and sampled derivative arrays stay in C:/Dev/readWFN_share_ms/lap_f015_audit_20261013, SHA-bound by lap_f015_provenance.json. Compact tables, nine data-backed plots, tests and scripts are committed; no checkpoints/raw molecular grids are committed.', '',
             'Provenance records source revision, exact checkpoint/cache/source/artifact SHA256, Python/NumPy/Torch/CUDA/device. All original dataset/checkpoint/production-source pins reverified. Optimizer updates0; trainingNO. Final test/lint/compile/diff receipts are recorded in lap_f015_validation.json. F014 is preserved and the final response records local/remote commit confirmation.', '']
    for title, filename in [('Signed errors', 'lap_f015_signed_errors.png'), ('Per-database comparison', 'lap_f015_database_mae.png'),
                            ('Reaction advantage distribution', 'lap_f015_advantage_histogram.png'), ('q saturation', 'lap_f015_q_saturation.png'),
                            ('Sensitivity distributions', 'lap_f015_sensitivities.png'), ('Exchange/correlation differences', 'lap_f015_xc_decomposition.png'),
                            ('Singular spectrum', 'lap_f015_singular_spectrum.png'), ('Local constrained progress', 'lap_f015_local_improvement.png'),
                            ('Predicted versus realized', 'lap_f015_finite_changes.png')]:
        lines.append(f'![{title}]({filename})')
    (audit.ROOT / 'lap_f015_s5_to_lap_audit_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'plots', 'provenance', 'rounding', 'compact', 'report'))
    args = parser.parse_args()
    {'prepare': prepare, 'plots': plots, 'provenance': provenance, 'rounding': rounding_check,
     'compact': compact, 'report': report}[args.stage]()
