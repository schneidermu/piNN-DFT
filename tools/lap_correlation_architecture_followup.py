"""Saved-gradient blocks and 15-system operator counterfactual.

Reads frozen checkpoints and previously saved objective gradients. The only
new evaluations are forward/backward diagnostics on temporary model copies.
"""
import time

import lap_b8_adamw as b8
import lap_correlation_architecture_audit as audit
import lap_f011_normalized_chemistry as arm
import lap_relchem_factorial_gradient_audit as fact
import lap_training_metric_gradient_audit as j251
import lap_wtmad_gradient_alignment as prior
import numpy as np

PANEL_FILE = b8.REPO / 'publication_data_v1_metadata' / 'provenance' / 'historical_operator_parity.json'
F011_VECTORS = b8.REPO.parent / 'lap_f011_trajectory_gradient_audit_20261011'
F010_GEOMETRY = b8.REPO.parent / 'lap_f010_uniform_normalized_chemistry_20261011' / 'geometry'


def _panel_names():
    records = b8.read_json(PANEL_FILE)['historical_systems']
    if len(records) != 15 or any(row['historical_loss_gradient_parity'] != 'PASS' for row in records):
        audit._stop('The historical 15-system operator panel is not the PASS oracle')
    return tuple(row['system'] for row in records)


def _share_row(name, task, vector, indices, counts):
    shares = prior.norm_shares(vector, indices)
    total = float(np.linalg.norm(vector))
    row = {'checkpoint': name, 'task': task, 'norm': total}
    for block, positions in indices.items():
        part = vector[positions] if len(positions) else np.zeros(0)
        row[f'{block}_squared_fraction'] = shares[block]
        row[f'{block}_norm'] = float(np.linalg.norm(part)) if len(part) else 0.0
        row[f'{block}_norm_per_parameter'] = (
            0.0 if counts[block] == 0 else float(np.linalg.norm(part) / counts[block])
        )
    return row


def saved_blocks(model):
    run = b8._import_run()
    names = tuple(run.existing.named_trainable_parameters(model))
    numels = [parameter.numel() for parameter in run.existing.named_trainable_parameters(model).values()]
    indices = prior.block_indices(names, numels, prior.parameter_blocks(model))
    counts = {block: len(positions) for block, positions in indices.items()}
    if counts['remaining'] != 0 or sum(counts.values()) != 9446:
        audit._stop('Parameter blocks do not partition 9446 coordinates')
    scales = b8.read_json(j251.OUT / 'reference_scales.json')['scales']
    rows = []
    sources = {
        'b8_t128': F011_VECTORS / 't0' / 'gradients.npz',
        'f011_t64': F011_VECTORS / 't64' / 'gradients.npz',
        'f011_t251': F011_VECTORS / 't251' / 'gradients.npz',
    }
    for name, path in sources.items():
        with np.load(path) as payload:
            stored = {key: np.asarray(payload[key], dtype=np.float64) for key in ('jw', 'relchem', 'exc', 'op')}
        for task, vector in stored.items():
            if vector.shape != (9446,):
                audit._stop(f'{name} {task} is not a 9446-vector')
            rows.append(_share_row(name, task, vector, indices, counts))
    for name in ('t128', 't251'):
        folder = F010_GEOMETRY / name
        loaded = j251._load_reactions(folder / 'reactions.npz')
        if loaded is None:
            audit._stop(f'F010 {name} reaction Jacobian is missing')
        meta, jacobian = loaded
        residuals = [row['residual'] for row in meta]
        databases = [row['database'] for row in meta]
        raw = np.array([j251._factor(database) for database in databases], dtype=np.float64)
        chemistry = {
            'jw': arm.jw_coefficients(residuals, scales, databases) @ jacobian,
            'relchem': fact.production_coefficients(residuals, raw) @ jacobian,
        }
        with np.load(folder / 'physical.npz') as physical:
            chemistry['exc'] = np.asarray(physical['exc'], dtype=np.float64)
            chemistry['op'] = np.asarray(physical['op'], dtype=np.float64)
        for task, vector in chemistry.items():
            rows.append(_share_row(f'f010_{name}', task, vector, indices, counts))
        dot = float(np.dot(chemistry['jw'], chemistry['relchem']))
        rows.append({
            'checkpoint': f'f010_{name}', 'task': 'jw_relchem_cosine',
            'norm': dot / (np.linalg.norm(chemistry['jw']) * np.linalg.norm(chemistry['relchem'])),
        })
    b8.write_json(audit.OUT / 'saved_blocks.json', {'counts': counts, 'rows': rows, 'optimizer_updates': 0})
    return rows, indices, counts


def _identity_for(bundle, source_id):
    matches = [identity for identity, meta in bundle.systems.items() if meta['source_id'] == source_id]
    if len(matches) != 1:
        audit._stop(f'{source_id} is not a unique mRKS source')
    return matches[0]


def _ec_gradient(model, features, weights):
    import torch
    from lap_vxc import sigma_from_gradients, sigma_standard_to_total

    from dft_functionals import PBE
    model.zero_grad(set_to_none=True)
    total = 0.0
    max_eps = -np.inf
    slope_electron = 0.0
    smooth_slope_electron = 0.0
    electrons = 0.0
    for start in range(0, len(features), audit.CHUNK):
        block = features[start:start + audit.CHUNK]
        weight = weights[start:start + audit.CHUNK]
        rho = block[:, :2]
        sigma = sigma_from_gradients(block[:, 2:8].reshape(-1, 2, 3))
        lap = block[:, 8:]
        raw = torch.cat([rho, sigma_standard_to_total(sigma), torch.zeros_like(rho), lap], -1)
        constants = model(raw)
        rs, zeta = PBE.rs_z_calc(rho)
        _xs0, _xs1, xt = PBE.xs_xt_calc(rho, sigma)
        epsilon = PBE.PBE_C(rs, zeta, xt, constants, rho.device)
        rho_total = rho.sum(-1)
        loss = (weight * rho_total * epsilon).sum()
        loss.backward()
        total += float(loss.detach().cpu())
        g_c = constants[:, 28].detach()
        electron = torch.clamp(weight, min=0.0) * torch.clamp(rho_total, min=0.0)
        z = torch.where(g_c >= 1.0, g_c, 1.0 + torch.log(torch.clamp(g_c, min=1e-30)))
        original = torch.where(z >= 1.0, torch.ones_like(z), torch.exp(z - 1.0))
        smooth = -torch.tanh(4.0 * (z - 1.0) / 2.0)
        slope_electron += float((electron * original).sum().detach().cpu())
        smooth_slope_electron += float((electron * smooth.abs()).sum().detach().cpu())
        electrons += float(electron.sum().detach().cpu())
        max_eps = max(max_eps, float(epsilon.detach().max().cpu()))
        del constants, epsilon, loss
    flat = []
    named = b8._import_run().existing.named_trainable_parameters(model)
    for parameter in named.values():
        grad = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
        flat.append(grad.double().cpu().reshape(-1).numpy())
    model.zero_grad(set_to_none=True)
    return {
        'Ec': total,
        'max_eps': max_eps,
        'gradient': np.concatenate(flat),
        'original_slope_electron': slope_electron / max(electrons, 1e-300),
        'smooth_slope_abs_electron': smooth_slope_electron / max(electrons, 1e-300),
    }


def panel(source):
    import torch
    path = audit.OUT / 'panel.json'
    if path.is_file():
        return b8.read_json(path)
    names = _panel_names()
    run = b8._import_run()
    bundle = run.PublicationDataset(run.DATA)
    rows = []
    try:
        mrks = bundle.mrks()
        identities = {source_id: _identity_for(bundle, source_id) for source_id in names}
        for spec in audit.CHECKS:
            if audit._sha(spec['path']) != spec['sha256']:
                audit._stop(f'{spec["name"]} hash changed before the panel')
            model, _saved = audit._load_model(source, spec['path'], smooth=False)
            smooth, _saved_smooth = audit._load_model(source, spec['path'], smooth=True)
            model32, _saved32 = audit._load_model(source, spec['path'], smooth=False)
            smooth32, _saved_smooth32 = audit._load_model(source, spec['path'], smooth=True)
            model32 = model32.float()
            smooth32 = smooth32.float()
            for source_id in names:
                started = time.perf_counter()
                values = mrks[mrks.ids.index(identities[source_id])]
                features = values['DensityDescriptorsN10'].double().cuda()
                weights = values['weights'].double().cuda()
                original = _ec_gradient(model, features, weights)
                changed = _ec_gradient(smooth, features, weights)
                system = mrks.operator_system(identities[source_id], device='cuda', dtype=torch.float32, chunk_size=4096)
                _exc_factory, op_factory = run.existing.core.make_mrks_objective_factories(
                    model32, system, point_chunk_size=256, dispersions=None, exc_chunk_size=4096)
                _exc_smooth, op_smooth = run.existing.core.make_mrks_objective_factories(
                    smooth32, system, point_chunk_size=256, dispersions=None, exc_chunk_size=4096)
                op_value = float(op_factory().detach().cpu())
                op_smooth_value = float(op_smooth().detach().cpu())
                model32.zero_grad(set_to_none=True)
                smooth32.zero_grad(set_to_none=True)
                rows.append({
                    'checkpoint': spec['name'], 'system': source_id,
                    'Ec': original['Ec'], 'smooth_Ec': changed['Ec'],
                    'max_eps': original['max_eps'], 'smooth_max_eps': changed['max_eps'],
                    'original_slope_electron': original['original_slope_electron'],
                    'smooth_slope_abs_electron': changed['smooth_slope_abs_electron'],
                    'op': op_value, 'smooth_op': op_smooth_value,
                    'ec_gradient_norm': float(np.linalg.norm(original['gradient'])),
                    'smooth_ec_gradient_norm': float(np.linalg.norm(changed['gradient'])),
                    'seconds': time.perf_counter() - started,
                })
                np.savez_compressed(
                    audit.OUT / spec['name'] / f'ec_grad_{source_id}.npz',
                    gradient=original['gradient'], smooth_gradient=changed['gradient'])
                print('CORR_PANEL', spec['name'], source_id, f"{original['max_eps']:.3e}", f"{op_value - op_smooth_value:.3e}", flush=True)
                del features, system
                torch.cuda.empty_cache()
            del model, smooth, model32, smooth32
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    for spec in audit.CHECKS:
        if audit._sha(spec['path']) != spec['sha256']:
            audit._stop(f'{spec["name"]} hash changed during the panel')
    b8.write_json(path, {'rows': rows, 'systems': list(names), 'optimizer_updates': 0})
    return {'rows': rows}


def execute():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    source = b8.verify_sources()
    b8._import_run()
    model, _saved = audit._load_model(source, audit.CHECKS[0]['path'], smooth=False)
    saved_blocks(model.cpu())
    del model
    torch.cuda.empty_cache()
    panel(source)
    print('CORR_FOLLOWUP_DONE', flush=True)


if __name__ == '__main__':
    execute()
