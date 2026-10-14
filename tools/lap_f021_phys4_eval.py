"""Required training-only checkpoints and one conditional fixed-density Clean28."""
import argparse
import math

import lap_f021_phys4 as exp
import numpy as np
import torch

audit, b8 = exp.audit, exp.b8


def constraints(model, shadow):
    from lap_f013_local_probes import _constraints
    from test_lap import raw_grid

    from dft_functionals.constants import NN_OUTPUT_SCALE_PBE
    result = _constraints(model)
    with torch.no_grad():
        raw = raw_grid().cuda().double()
        shadow.load_state_dict(model.state_dict(), strict=True)
        d = shadow.get_correlation_descriptors(raw)
        exchange = shadow.get_exchange_descriptors(raw * raw.new_tensor([2, 2, 4, 4, 4, 2, 2, 2, 2]))
        ue, ux = shadow.all_sigma_zero(d), shadow.all_sigma_zero(exchange)
        anchors = [shadow.forward_descriptors(ue, ux), shadow.forward_descriptors(shadow.all_rho_inf(d), exchange),
                   shadow.forward_descriptors(shadow.all_s_inf(d), exchange)]
        scale = NN_OUTPUT_SCALE_PBE.to(raw)
        result['ueg_beta_gap'] = float((anchors[0][:, 0] / scale[0, 0] - 1).abs().max())
        result['high_density_gamma_gap'] = float((anchors[1][:, 1] / scale[0, 1] - 1).abs().max())
        result['gc_anchor_gap'] = max(float((v[:, 28] - 1).abs().max()) for v in anchors)
        changed = raw.clone()
        changed[:, 5:7] += 1
        result['tau_gap'] = float((shadow(raw) - shadow(changed)).abs().max())
    result['passed'] = (result['output_finite'] and result['spin_max_abs'] <= 1e-6
                        and result['passthrough_max_abs'] == 0 and result['tau_gap'] == 0
                        and max(result['ueg_beta_gap'], result['high_density_gamma_gap'], result['gc_anchor_gap']) <= 1e-10)
    assert result['passed'], 'Architecture constraint regression'
    return result


def evaluate(cursor, source, budget):
    receipt = exp.OUT / f'metrics_{cursor}.json'
    path = exp.OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
    if receipt.exists():
        saved = audit.read(receipt)
        assert saved['checkpoint_sha256'] == audit.sha(path)
        return saved
    model, shadow, run = audit.existing._load_eval_model(source, path)
    shadow.load_state_dict(model.state_dict(), strict=True)
    before = run.existing.digest(model)
    bundle = audit.TrainingBundle(exp.ROOT.parent / 'publication_dataset_v1')
    dispersion = bundle.chemistry_dispersions()
    md = audit.read(bundle.root / 'mrks/dispersion.json')
    scales = audit.read(exp.ROOT / 'lap_f010_normalization_receipt.json')['scales']
    specs = audit.chemistry._selected(bundle)
    reaction_cache = exp.OUT / f'reactions_{cursor}.json'
    rows = audit.read(reaction_cache) if reaction_cache.exists() else []
    assert all(r['identity'] == specs[i]['identity'] and r['variant'] == specs[i]['variant'] for i, r in enumerate(rows))
    for s in specs[len(rows):]:
        budget.check(60)
        reaction = b8._load_reaction(bundle, 'relchem', s)
        with torch.no_grad():
            predicted, reference = audit.chemistry._prediction(shadow, reaction, dispersion)
        residual = float(predicted - reference)
        assert math.isfinite(residual)
        rows.append({**s, 'reference': float(reference), 'residual': residual,
                     'legacy_penalty': audit.chemistry._factor(s['database']) * math.sqrt(residual**2 + 1e-20),
                     'JW_penalty': math.sqrt(residual**2 + 1e-20) / scales[s['database']]})
        audit.write(reaction_cache, rows)
        del reaction, predicted, reference
    ae_path = exp.OUT / f'ae17_{cursor}.json'
    if ae_path.exists():
        ae = audit.read(ae_path)['loss']
    else:
        budget.check(60)
        ae = audit.existing.continuation_chemistry_scalar(model, shadow, bundle, dispersion, 'ae17')
        audit.write(ae_path, {'loss': ae, 'count': 17})
    physical_cache = exp.OUT / f'physical_{cursor}.json'
    physical = audit.read(physical_cache) if physical_cache.exists() else {}
    for identity in sorted(bundle.systems):
        if identity in physical:
            continue
        budget.check(60)
        system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
        factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=md)
        with torch.no_grad():
            values = {t: float(factory()) for t, factory in zip(('exc', 'op'), factories, strict=True)}
        assert all(math.isfinite(v) for v in values.values())
        physical[identity] = values
        audit.write(physical_cache, physical)
        del system, factories
        torch.cuda.empty_cache()
        print('F021_EVAL', cursor, len(physical), 90, flush=True)
    assert len(rows) == 251 and len(physical) == 90
    values = {'relchem': float(np.mean([r['legacy_penalty'] for r in rows])), 'ae17': ae,
              **{t: float(np.mean([r[t] for r in physical.values()])) for t in ('exc', 'op')}}
    result = {'cursor': cursor, 'checkpoint_sha256': audit.sha(path), 'values': values,
              'ratios': {t: values[t] / b8.BASELINE[t] for t in b8.TASKS},
              'JW': float(np.mean([r['JW_penalty'] for r in rows])),
              'chemistry': audit.stats([r['residual'] for r in rows]),
              'constraints': constraints(model, shadow), 'model_unchanged': run.existing.digest(model) == before}
    assert result['model_unchanged']
    result['eligible'] = all(v < 1 and math.isfinite(v) for v in result['ratios'].values())
    audit.write(receipt, result)
    bundle.close()
    return result


def clean(source, budget):
    final = audit.read(exp.OUT / 'metrics_128.json')
    path = exp.OUT / 'ordinary_sgd_adamw/checkpoint_128.pt'
    assert audit.sha(path) == final['checkpoint_sha256']
    target = exp.ROOT / 'lap_f021_phys4_clean28_receipt.json'
    if target.exists():
        assert exp.read('clean28_receipt').get('checkpoint_sha256') == final['checkpoint_sha256']
        return
    if not final['eligible']:
        exp.write('clean28_receipt', {'status': 'NOT EVALUATED — scientific eligibility failed',
                                    'checkpoint_sha256': final['checkpoint_sha256'], 'evaluations': 0})
        return
    budget.check(120)
    from tools.evaluate_microbatch_endpoint import validation
    model, _shadow, run = audit.existing._load_eval_model(source, path)
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    result = validation(model, bundle)
    assert run.existing.digest(model) == before
    exp.write('clean28_receipt', {'checkpoint_sha256': final['checkpoint_sha256'], 'evaluations': 1,
                                'clean28': result['clean28'], 'non_self_consistent': True,
                                'reaction_rows': [r for r in result['reaction_rows'] if r['clean']],
                                'full30_used_for_selection': False})
    bundle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('64', '128', 'clean'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = exp.Budget()
    try:
        _prefix, source, _scales = exp.freeze()
        if args.stage == 'clean':
            clean(source, budget)
        else:
            evaluate(int(args.stage), source, budget)
    finally:
        budget.check()


if __name__ == '__main__':
    main()
