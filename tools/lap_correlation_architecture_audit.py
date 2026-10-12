"""Read-only correlation-architecture audit for pcPBELMLOptimizerV2Lap.

No optimizer step, no Clean28, and no production-file writes. Counterfactual
forwards use a temporary copy whose G_c activation is the existing smooth
mirrored softplus. Checkpoint files are only read.
"""
import hashlib
import math
from pathlib import Path

import lap_b8_adamw as b8
import lap_f010_uniform_normalized_chemistry as f010
import lap_f011_normalized_chemistry as f011
import numpy as np

PROTOCOL_ID = 'lap-correlation-architecture-audit-v1'
OUT = b8.REPO.parent / 'lap_correlation_architecture_audit_20261012'
SIGN_THRESHOLD = 1e-8
H_FLOOR = 1e-12
STRIDE = 400
CHUNK = 8192
BINS = 3000
GC_MAX = 3.0
CHECKS = (
    {'name': 'b8_t128', 'path': f011.PARENT_CHECKPOINT, 'sha256': '1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b'},
    {'name': 'f010_t128', 'path': f010.OUT / 'ordinary_sgd_adamw' / 'checkpoint_128.pt', 'sha256': 'bd5ecc45fa44793e95d82492e2666d1c40c6fa6068b4f2e266824d3774fbc822'},
    {'name': 'f010_t251', 'path': f010.OUT / 'ordinary_sgd_adamw' / 'checkpoint_251.pt', 'sha256': '1dba8af1a11d5f98fdc5b40d8be6b137d9773af3a6f2bc9eb1482750b3a5feb3'},
    {'name': 'f011_t64', 'path': f011.OUT / 'ordinary_sgd_adamw' / 'checkpoint_64.pt', 'sha256': '31aad5509924d7a61a641a85b4127cb52070dbb5761d64a5c2f817db134abc7a'},
    {'name': 'f011_t251', 'path': f011.OUT / 'ordinary_sgd_adamw' / 'checkpoint_251.pt', 'sha256': '8be1fd6de97455f76575a1b353383dd276e7abda3f67e5a08e7eece48a753486'},
)


def smooth_gc(z, sharpness=4.0):
    """1 - (2/k) log(cosh(k(z-1)/2)), stable via the existing softplus identity."""
    delta = np.asarray(z, dtype=np.float64) - 1.0
    penalty = (
        np.log1p(np.exp(-np.abs(sharpness * delta))) + np.maximum(sharpness * delta, 0.0)
        + np.log1p(np.exp(-np.abs(-sharpness * delta))) + np.maximum(-sharpness * delta, 0.0)
        - 2.0 * math.log(2.0)
    ) / sharpness
    return 1.0 - penalty


def invert_shifted_elu(g_c):
    """Invert G_c = shifted_elu(z-1). Valid while G_c > 0."""
    g_c = np.asarray(g_c, dtype=np.float64)
    return np.where(g_c >= 1.0, g_c, 1.0 + np.log(np.maximum(g_c, 1e-300)))


def original_slope(g_c):
    g_c = np.asarray(g_c, dtype=np.float64)
    return np.where(g_c >= 1.0, 1.0, g_c)


def smooth_slope(z, sharpness=4.0):
    return -np.tanh(sharpness * (np.asarray(z, dtype=np.float64) - 1.0) / 2.0)


def classify_point(pw, h_term, g_c, epsilon, finite):
    if not finite:
        return 'C'
    at_one = pw + h_term
    if epsilon > SIGN_THRESHOLD and h_term > H_FLOOR and pw < 0.0:
        critical = -pw / h_term
        if g_c > critical + 1e-8 and critical >= 1.0 - 1e-6:
            return 'A'
        if at_one > SIGN_THRESHOLD:
            return 'B'
        return 'D'
    if epsilon > SIGN_THRESHOLD and at_one > SIGN_THRESHOLD:
        return 'B'
    if epsilon > SIGN_THRESHOLD:
        return 'D'
    return ''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stop(reason):
    b8.write_json(OUT / 'stop.json', {'reason': reason, 'optimizer_updates': 0})
    raise SystemExit(reason)


def _quantile(edges, counts, probability):
    total = float(np.sum(counts))
    if total <= 0.0:
        return None
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, probability * total, side='left'))
    index = min(index, len(edges) - 2)
    return float(edges[index])


def _hist_add(histogram, values, weights):
    clipped = np.clip(values, 0.0, GC_MAX - 1e-12)
    index = np.minimum((clipped / GC_MAX * BINS).astype(np.int64), BINS - 1)
    np.add.at(histogram, index, weights)


def _load_model(source, path, smooth):
    import torch
    from NN_models import pcPBELMLOptimizerV2GcSoftplusMirror
    run = b8._import_run()
    model, _shadow = run.model_at(source['initial_state'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    if smooth:
        model.G_C_SOFTPLUS_SHARPNESS = pcPBELMLOptimizerV2GcSoftplusMirror.G_C_SOFTPLUS_SHARPNESS
        model.activate_g_c = pcPBELMLOptimizerV2GcSoftplusMirror.activate_g_c.__get__(model, type(model))
    model = model.double().cuda().eval()
    if any(module.p != 0 for module in model.modules() if module.__class__.__name__ == 'Dropout'):
        _stop('Dropout is enabled')
    return model, saved


def _chunk_stats(model, smooth, features, weights, sample, system, offset):
    import torch
    from lap_vxc import sigma_from_gradients, sigma_standard_to_total

    from dft_functionals import PBE
    with torch.no_grad():
        rho = features[:, :2]
        sigma = sigma_from_gradients(features[:, 2:8].reshape(-1, 2, 3))
        lap = features[:, 8:]
        raw = torch.cat([rho, sigma_standard_to_total(sigma), torch.zeros_like(rho), lap], -1)
        constants = model(raw)
        smooth_constants = smooth(raw)
        rs, zeta = PBE.rs_z_calc(rho)
        _xs0, _xs1, xt = PBE.xs_xt_calc(rho, sigma)
        pw = PBE.f_pw(rs, zeta, constants)
        h_term = PBE.fH(rs, zeta, PBE.tt(rs, zeta, xt), constants, rho.device)
        g_c = constants[:, 28]
        epsilon = PBE.PBE_C(rs, zeta, xt, constants, rho.device)
        smooth_eps = PBE.PBE_C(rs, zeta, xt, smooth_constants, rho.device)
        exc = PBE.F_PBE(rho, sigma, constants, rho.device)
        smooth_exc = PBE.F_PBE(rho, sigma, smooth_constants, rho.device)
        gap = (epsilon - (pw + g_c * h_term)).abs().max()
        rho_total = rho.sum(-1)
        weight = weights.double()
        packed = {
            'smooth_Ec': float((weight * rho_total * smooth_eps).sum().detach().cpu()),
            'production_Exc': float((weight * rho_total * exc).sum().detach().cpu()),
            'smooth_Exc': float((weight * rho_total * smooth_exc).sum().detach().cpu()),
            'smooth_eps': smooth_eps.detach().cpu().numpy(),
        }
    beta = constants[:, 0]
    gamma = constants[:, 1]
    density = rho_total * epsilon
    packed.update({
        'gap': float(gap.detach().cpu()),
        'pw': pw.detach().cpu().numpy(),
        'h': h_term.detach().cpu().numpy(),
        'g': g_c.detach().cpu().numpy(),
        'eps': epsilon.detach().cpu().numpy(),
        'rho': rho_total.detach().cpu().numpy(),
        'beta': beta.detach().cpu().numpy(),
        'gamma': gamma.detach().cpu().numpy(),
        'density': density.detach().cpu().numpy(),
        'finite': np.isfinite(pw.detach().cpu().numpy()) & np.isfinite(h_term.detach().cpu().numpy()) & np.isfinite(g_c.detach().cpu().numpy()),
    })
    if sample is not None:
        pick = ((np.arange(offset, offset + len(features)) % STRIDE) == 0)
        if np.any(pick):
            sample.append({key: packed[key][pick] for key in ('g', 'eps', 'pw', 'h', 'rho')})
            sample[-1]['system'] = system
    return packed


def _reduce_system(parts, weights):
    g = np.concatenate([part['g'] for part in parts])
    eps = np.concatenate([part['eps'] for part in parts])
    pw = np.concatenate([part['pw'] for part in parts])
    h_term = np.concatenate([part['h'] for part in parts])
    rho = np.concatenate([part['rho'] for part in parts])
    beta = np.concatenate([part['beta'] for part in parts])
    gamma = np.concatenate([part['gamma'] for part in parts])
    finite = np.concatenate([part['finite'] for part in parts])
    w = np.asarray(weights, dtype=np.float64)
    electron = np.maximum(w, 0.0) * np.maximum(rho, 0.0)
    energy = w * rho * eps
    positive = (eps > SIGN_THRESHOLD) & finite
    negative = (eps < -SIGN_THRESHOLD) & finite
    at_one = pw + h_term
    mechanisms = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    for index in np.flatnonzero(positive | ~finite):
        label = classify_point(float(pw[index]), float(h_term[index]), float(g[index]), float(eps[index]), bool(finite[index]))
        if label:
            mechanisms[label] += float(max(energy[index], 0.0))
    conditioned = (h_term > H_FLOOR) & (pw < 0.0) & finite
    critical = np.divide(-pw, h_term, out=np.full_like(pw, np.nan), where=conditioned)
    return {
        'points': len(g),
        'nonpositive_weights': int(np.sum(w <= 0.0)),
        'electrons': float(electron.sum()),
        'gc_min': float(np.min(g)), 'gc_max': float(np.max(g)),
        'gc_mean_point': float(np.mean(g)),
        'gc_mean_electron': float(np.sum(electron * g) / max(electron.sum(), 1e-300)),
        'fraction_gc_gt_1_point': float(np.mean(g > 1.0)),
        'fraction_gc_gt_1_electron': float(np.sum(electron * (g > 1.0)) / max(electron.sum(), 1e-300)),
        'fraction_gc_gt_1p01_electron': float(np.sum(electron * (g > 1.01)) / max(electron.sum(), 1e-300)),
        'fraction_gc_gt_1p1_electron': float(np.sum(electron * (g > 1.1)) / max(electron.sum(), 1e-300)),
        'fraction_gc_gt_1p5_electron': float(np.sum(electron * (g > 1.5)) / max(electron.sum(), 1e-300)),
        'fraction_gc_lt_0_electron': float(np.sum(electron * (g < 0.0)) / max(electron.sum(), 1e-300)),
        'fraction_gc_near_1_electron': float(np.sum(electron * (np.abs(g - 1.0) <= 1e-3)) / max(electron.sum(), 1e-300)),
        'nonfinite': int(np.sum(~finite)),
        'positive_points': int(np.sum(positive)),
        'positive_electron_fraction': float(np.sum(electron * positive) / max(electron.sum(), 1e-300)),
        'Ec': float(energy.sum()),
        'Ec_positive': float(energy[positive].sum()) if np.any(positive) else 0.0,
        'Ec_negative': float(energy[negative].sum()) if np.any(negative) else 0.0,
        'Ec_abs': float(np.sum(np.abs(energy))),
        'case_B_electron_fraction': float(np.sum(electron * (at_one > SIGN_THRESHOLD) * finite) / max(electron.sum(), 1e-300)),
        'mechanisms': mechanisms,
        'critical_median': float(np.nanmedian(critical)) if np.any(conditioned) else None,
        'above_critical_electron': float(np.sum(electron[conditioned] * (g[conditioned] > critical[conditioned])) / max(electron.sum(), 1e-300)),
        'beta_min': float(np.min(beta)), 'beta_max': float(np.max(beta)),
        'gamma_min': float(np.min(gamma)), 'gamma_max': float(np.max(gamma)),
        'decomposition_max_abs': float(max(part['gap'] for part in parts)),
        'g': g, 'eps': eps, 'electron': electron, 'critical': critical, 'conditioned': conditioned,
    }


def scan_checkpoint(spec, source):
    import torch
    folder = OUT / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'systems.json'
    if done.is_file():
        return b8.read_json(done)
    if _sha(spec['path']) != spec['sha256']:
        _stop(f'{spec["name"]} hash does not match the published receipt')
    before = _sha(spec['path'])
    model, _saved = _load_model(source, spec['path'], smooth=False)
    smooth, _saved_smooth = _load_model(source, spec['path'], smooth=True)
    coordinates = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or coordinates != 9446 or int(model.lap_architecture_version) != 1:
        _stop(f'{spec["name"]} is not the 9446-parameter Lap model')
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    rows = []
    point_hist = np.zeros(BINS, dtype=np.float64)
    electron_hist = np.zeros(BINS, dtype=np.float64)
    sample = []
    try:
        mrks = bundle.mrks()
        for index, identity in enumerate(mrks.ids):
            values = mrks[index]
            features = values['DensityDescriptorsN10'].double().cuda()
            weights = values['weights'].double().cpu().numpy()
            parts = []
            for start in range(0, len(features), CHUNK):
                block = features[start:start + CHUNK]
                weight_block = torch.as_tensor(weights[start:start + CHUNK], device='cuda', dtype=torch.float64)
                parts.append(_chunk_stats(model, smooth, block, weight_block, sample, identity, start))
            reduced = _reduce_system(parts, weights)
            smooth_eps = np.concatenate([part['smooth_eps'] for part in parts])
            rho = np.concatenate([part['rho'] for part in parts])
            smooth_density = weights * rho * smooth_eps
            smooth_positive = float(smooth_density[smooth_eps > SIGN_THRESHOLD].sum()) if np.any(smooth_eps > SIGN_THRESHOLD) else 0.0
            _hist_add(point_hist, reduced['g'], np.ones(len(reduced['g'])))
            _hist_add(electron_hist, reduced['g'], reduced['electron'])
            meta = bundle.systems[identity]
            rows.append({
                'system': meta['source_id'], 'identity': identity, 'n_grid': int(meta['n_grid']),
                **{key: value for key, value in reduced.items() if key not in ('g', 'eps', 'electron', 'critical', 'conditioned')},
                'smooth_Ec': float(sum(part['smooth_Ec'] for part in parts)),
                'smooth_Ec_positive': smooth_positive,
                'production_Exc': float(sum(part['production_Exc'] for part in parts)),
                'smooth_Exc': float(sum(part['smooth_Exc'] for part in parts)),
            })
            print('CORR_SYSTEM', spec['name'], index + 1, meta['source_id'], f"{rows[-1]['Ec_positive']:.6e}", flush=True)
            del features, parts, reduced
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    if _sha(spec['path']) != before:
        _stop('The audit changed a checkpoint file')
    edges = np.linspace(0.0, GC_MAX, BINS + 1)
    payload = {
        'name': spec['name'], 'sha256': before, 'coordinates': coordinates, 'systems': rows,
        'gc_quantiles_electron': {str(p): _quantile(edges, electron_hist, p) for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)},
        'gc_quantiles_point': {str(p): _quantile(edges, point_hist, p) for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)},
        'decomposition_max_abs': max(row['decomposition_max_abs'] for row in rows),
        'optimizer_updates': 0,
    }
    if sample:
        stacked = {key: np.concatenate([item[key] for item in sample]) for key in ('g', 'eps', 'pw', 'h', 'rho')}
        np.savez_compressed(folder / 'sample.npz', **stacked)
    b8.write_json(done, payload)
    del model, smooth
    torch.cuda.empty_cache()
    return payload


def constraints(spec, source):
    import torch
    from test_lap import raw_grid

    from dft_functionals.constants import NN_OUTPUT_SCALE_PBE, PBE_CONSTANTS
    path = OUT / spec['name'] / 'constraints.json'
    if path.is_file():
        return b8.read_json(path)
    rows = []
    for dtype, label in ((torch.float64, 'float64'), (torch.float32, 'float32')):
        model, _saved = _load_model(source, spec['path'], smooth=False)
        model = model.to(dtype=dtype)
        grid = raw_grid().to(device='cuda', dtype=dtype)
        first = model(grid)
        second = model(grid)
        changed = grid.clone()
        changed[:, 5:7] = torch.linspace(-10.0, 10.0, changed[:, 5:7].numel(), device=grid.device, dtype=dtype).reshape_as(changed[:, 5:7])
        tau = model(changed)
        swapped = grid[:, [1, 0, 4, 3, 2, 6, 5, 8, 7]]
        spin = model(swapped)
        descriptors = model.get_correlation_descriptors(grid)
        scale = NN_OUTPUT_SCALE_PBE.to(grid)
        anchors = {
            'ueg': model.forward_descriptors(model.all_sigma_zero(descriptors), model.all_sigma_zero(descriptors)) / scale,
            'rho_inf': model.forward_descriptors(model.all_rho_inf(descriptors), descriptors) / scale,
            's_inf': model.forward_descriptors(model.all_s_inf(descriptors), descriptors) / scale,
        }
        scaled = first / scale
        expected_spin = first[:, list(range(22)) + [24, 25, 22, 23, 27, 26, 28]]
        leaf = torch.tensor([0.5, 1.0, 1.5], device='cuda', dtype=dtype, requires_grad=True)
        slope = torch.autograd.grad(model.activate_g_c(leaf).sum(), leaf)[0]
        rows.append({
            'precision': label,
            'repeat_max_abs': float((first - second).abs().max().detach().cpu()),
            'tau_max_abs': float((first - tau).abs().max().detach().cpu()),
            'spin_max_abs': float((spin - expected_spin).abs().max().detach().cpu()),
            'spin_gc_max_abs': float((first[:, 28] - spin[:, 28]).abs().max().detach().cpu()),
            'ueg_gnn_max_abs': float(anchors['ueg'][:, 26:28].abs().max().detach().cpu()),
            'ueg_gc_max_abs': float((anchors['ueg'][:, 28] - 1).abs().max().detach().cpu()),
            'ueg_beta_max_abs': float((anchors['ueg'][:, 0] - 1).abs().max().detach().cpu()),
            'rho_inf_gc_max_abs': float((anchors['rho_inf'][:, 28] - 1).abs().max().detach().cpu()),
            'rho_inf_gamma_max_abs': float((anchors['rho_inf'][:, 1] - 1).abs().max().detach().cpu()),
            's_inf_gc_max_abs': float((anchors['s_inf'][:, 28] - 1).abs().max().detach().cpu()),
            'passthrough_max_abs': float((first[:, 2:22] - PBE_CONSTANTS.to(first)[0, 2:22]).abs().max().detach().cpu()),
            'passthrough_factor_max_abs': float((scaled[:, 2:22] - 1).abs().max().detach().cpu()),
            'gc_slope_at_0p5_1_1p5': [float(value) for value in slope.detach().cpu()],
            'beta_finite': bool(torch.isfinite(first[:, 0]).all()),
            'gamma_finite': bool(torch.isfinite(first[:, 1]).all()),
            'gc_finite': bool(torch.isfinite(first[:, 28]).all()),
        })
        del model
        torch.cuda.empty_cache()
    b8.write_json(path, {'rows': rows, 'optimizer_updates': 0})
    return {'rows': rows}


def execute():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    source = b8.verify_sources()
    b8._import_run()
    scans = [scan_checkpoint(spec, source) for spec in CHECKS]
    limits = [constraints(spec, source) for spec in CHECKS]
    b8.write_json(OUT / 'summary.json', {'scans': [item['name'] for item in scans], 'constraints': len(limits), 'optimizer_updates': 0})
    print('CORR_SCAN_DONE', len(scans), flush=True)
    return scans, limits


if __name__ == '__main__':
    try:
        execute()
    except SystemExit as error:
        print('CORR_STOP', error, flush=True)
        raise
