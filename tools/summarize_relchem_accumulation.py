"""Read-only paired analysis of the bounded accumulation screen."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import relchem_accumulation_sweep as s


def paired_changes(probes,baseline,count=5000):
    rng=np.random.default_rng(931002)
    result={k:{} for k in probes}
    for task in s.run.TASKS:
        groups={}
        for key,row in baseline['rows'].items():
            if row['task']==task: groups.setdefault(row['stratum'],[]).append(key)
        totals={k:np.zeros(count) for k in ['t0',*probes]}
        for keys in groups.values():
            ix=rng.integers(0,len(keys),(count,len(keys)))
            weight=sum(baseline['rows'][key]['weight'] for key in keys)
            for k,p in {'t0':baseline,**probes}.items():
                totals[k]+=weight*np.array([p['rows'][key]['loss'] for key in keys])[ix].mean(axis=1)
        ratios={k:totals[k]/totals['t0'] for k in probes}
        for k in probes:
            result[k][task]={'ratio95':np.percentile(ratios[k],[2.5,97.5]).tolist(),
                'difference_vs_K1_95':np.percentile(ratios[k]-ratios['1'],[2.5,97.5]).tolist()}
    return result


def main():
    run=s.run; root=s.OUT
    protocol=run.read(root/'protocol.json')
    qualification=run.read(root/'qualification.json')
    baseline=run.read(s.sweep.OUT/'probe_t0.json')
    probes={'1':run.read(s.CONTROL/'probe_t20.json')}
    for k in (2,4,6): probes[str(k)]=run.read(root/f'K{k}'/'probe_t20.json')
    original=run.read(s.CONTROL/'sampling_manifest.json'); manifest=run.read(root/'sampling_manifest.json')
    assert [{key:value for key,value in e.items() if key!='relchem_extra'} for e in manifest]==original
    control0=torch.load(s.CONTROL/'ordinary_sgd_adamw/checkpoint_0.pt',map_location='cpu',weights_only=False)
    controlraw=np.load(s.CONTROL/'ordinary_sgd_adamw/raw_gradients/update_000.npz')
    ratios={k:{t:p['objectives'][t]/baseline['objectives'][t] for t in run.TASKS} for k,p in probes.items()}
    logs={}; hashes={}; duplicates={}; progress={}
    for k in s.KS:
        folder=s.CONTROL if k==1 else root/f'K{k}'
        path=folder/'ordinary_sgd_adamw/checkpoint_20.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['cursor']==20 and saved['scheduler'] is None
        assert saved['calibration']['lambda']==protocol['lambda']
        assert all(x['learning_rate']==1e-4 for x in saved['logs'])
        assert all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point())
        state0=torch.load(folder/'ordinary_sgd_adamw/checkpoint_0.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,state0['model'][n]) for n,v in control0['model'].items())
        with np.load(folder/'ordinary_sgd_adamw/raw_gradients/update_000.npz') as a:
            assert all(np.array_equal(a[t],controlraw[t]) for t in ('ae17','exc','op'))
        if k>1: assert all(len(x['relchem_batch'])==k for x in saved['logs'])
        logs[str(k)]={'mean_seconds':float(np.mean([x['total_seconds'] for x in saved['logs']])),
            'total_seconds':sum(x['total_seconds'] for x in saved['logs']),
            'wall_seconds_including_checkpoint_io':path.stat().st_mtime-(folder/'ordinary_sgd_adamw/checkpoint_0.pt').stat().st_mtime,
            'wall_seconds_per_update':(path.stat().st_mtime-(folder/'ordinary_sgd_adamw/checkpoint_0.pt').stat().st_mtime)/20,
            'peak_live_gib':max(x['peak_allocated_bytes'] for x in saved['logs'])/2**30,
            'peak_reserved_gib':max(x['peak_reserved_bytes'] for x in saved['logs'])/2**30}
        hashes[str(k)]={'checkpoint20_sha256':run.file_sha256(path),'manifest_sha256':run.file_sha256(folder/'sampling_manifest.json')}
        batches=[[e['relchem'],*e['relchem_extra'][:k-1]] for e in manifest[:20]]
        duplicates[str(k)]={'identities':sum(len(b)-len({r['identity'] for r in b}) for b in batches),
            'identity_variant_pairs':sum(len(b)-len({(r['identity'],r['variant']) for r in b}) for b in batches)}
        progress[str(k)]={t:{'negative_count':sum(x['task_progress_actual_step'][i]<0 for x in saved['logs']),
            'median':float(np.median([x['task_progress_actual_step'][i] for x in saved['logs']]))} for i,t in enumerate(run.TASKS)}
    assert run.file_sha256(s.CONTROL/'ordinary_sgd_adamw/latest.pt')==protocol['historical_cursor59_sha256']
    efficiency={}
    v1=qualification['variance']['1']['mean_pairwise_gradient_variance']
    for k in s.KS:
        st=str(k); v=qualification['variance'][st]['mean_pairwise_gradient_variance']; sec=logs[st]['wall_seconds_per_update']
        efficiency[st]={'relchem_ratio_reduction_per_training_minute':(1-ratios[st]['relchem'])/(logs[st]['wall_seconds_including_checkpoint_io']/60),
            'observed_variance_ratio_to_K1':v/v1,'extra_seconds_per_update_vs_K1':sec-logs['1']['wall_seconds_per_update'],
            'descriptive_variance_reduction_per_extra_update_second':None if k==1 else (v1-v)/(sec-logs['1']['wall_seconds_per_update'])}
    result={'protocol':protocol,'qualification':qualification,'probe_ratios':ratios,'paired_uncertainty':paired_changes(probes,baseline),
        'runtime':logs,'compute_efficiency':efficiency,'duplicates':duplicates,'actual_step_progress':progress,
        'checkpoint_hashes':hashes,'new_optimizer_updates':60,'new_training_relchem_reactions':240,
        'new_training_ae17_reactions':60,'new_training_exc_samples':60,'new_training_operator_samples':60,
        'control_reused':True,'historical_cursor59_unchanged':True,'initial_other_gradients_bitwise_equal':True,
        'no_external_or_full_corpus_evaluation':True}
    run.write(root/'results.json',result)
    run.write(run.ROOT/'relchem_accumulation_sweep_metrics.json',result)
    run.write(run.ROOT/'relchem_accumulation_sweep_protocol.json',protocol)
    print(json.dumps({k:result[k] for k in ('probe_ratios','paired_uncertainty','runtime','compute_efficiency','actual_step_progress')},indent=2))


if __name__=='__main__':
    main()

