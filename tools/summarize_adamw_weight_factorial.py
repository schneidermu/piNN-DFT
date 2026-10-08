import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from tools import adamw_weight_factorial as f

r=f.OUT; run=f.run; sw=f.sweep
baseline=run.read(sw.OUT/'probe_t0.json')
probes={'A':run.read(f.CONTROL/'probe_t20.json')}
for arm in 'BCD': probes[arm]=run.read(r/arm/'probe_t20.json')
t59=run.read(r/'t59/probe_t20.json')
ratios={a:{t:p['objectives'][t]/baseline['objectives'][t] for t in run.TASKS} for a,p in probes.items()}
# Identical resamples for all arms and baseline: paired factorial contrasts.
rng=np.random.default_rng(931002); effects={}; uncertainty={a:{} for a in probes}
for t in run.TASKS:
 keys=[k for k,v in baseline['rows'].items() if v['task']==t]
 groups={}
 for k in keys: groups.setdefault(baseline['rows'][k]['stratum'],[]).append(k)
 sums={a:np.zeros(5000) for a in ['t0',*probes]}
 for group in groups.values():
  ix=rng.integers(0,len(group),(5000,len(group)))
  weight=sum(baseline['rows'][k]['weight'] for k in group)
  for a,p in {'t0':baseline,**probes}.items():
   sums[a]+=weight*np.array([p['rows'][k]['loss'] for k in group])[ix].mean(axis=1)
 q={a:sums[a]/sums['t0'] for a in probes}
 for a in probes:
  uncertainty[a][t]={'ratio95':np.percentile(q[a],[2.5,97.5]).tolist(),'difference_vs_A95':np.percentile(q[a]-q['A'],[2.5,97.5]).tolist(),'difference_vs_A':ratios[a][t]-ratios['A'][t]}
 contrasts={'relchem_main':(.5*(q['B']+q['D']-q['A']-q['C']),.5*(ratios['B'][t]+ratios['D'][t]-ratios['A'][t]-ratios['C'][t])),
 'operator_main':(.5*(q['C']+q['D']-q['A']-q['B']),.5*(ratios['C'][t]+ratios['D'][t]-ratios['A'][t]-ratios['B'][t])),
 'interaction':(q['D']-q['B']-q['C']+q['A'],ratios['D'][t]-ratios['B'][t]-ratios['C'][t]+ratios['A'][t])}
 effects[t]={name:{'estimate':float(point),'bootstrap95':np.percentile(values,[2.5,97.5]).tolist()} for name,(values,point) in contrasts.items()}
logs={}; checkpoints={}
for a in probes:
 folder=f.CONTROL if a=='A' else r/a
 p=folder/'ordinary_sgd_adamw/checkpoint_20.pt'; c=torch.load(p,map_location='cpu',weights_only=False)
 logs[a]={'mean_seconds':float(np.mean([x['total_seconds'] for x in c['logs']])),'total_seconds':sum(x['total_seconds'] for x in c['logs']), 'peak_live_gib':max(x['peak_allocated_bytes'] for x in c['logs'])/2**30,'cursor':c['cursor']}
 checkpoints[a]=run.file_sha256(p)
c=torch.load(f.CONTROL/'ordinary_sgd_adamw/latest.pt',map_location='cpu',weights_only=False)
tail=c['logs'][20:]; conflicts={t:sum(x['task_progress_actual_step'][i]<0 for x in tail) for i,t in enumerate(run.TASKS)}
chem=[]
for k,row in baseline['rows'].items():
 if row['task']=='relchem': chem.append({'identity':row['identity'],'variant':row['variant'],'database':row['stratum'],'t0':row['loss'],'t20':probes['A']['rows'][k]['loss'],'t59':t59['rows'][k]['loss']})
m={'protocol':run.read(r/'protocol.json'),'probe_ratios':ratios,'paired_uncertainty':uncertainty,'factorial_effects':effects,'runtime':logs,'checkpoint20_sha256':checkpoints,'t59_ratios':{t:t59['objectives'][t]/baseline['objectives'][t] for t in run.TASKS},'control_t21_t59_negative_actual_step_counts':conflicts,'chemistry_reaction_distribution':chem,'directions':run.read(r/'directions.json'),'exact_full_objectives_evaluated':False,'external_validation_evaluated':False,'historical_t59_unchanged':run.file_sha256(f.CONTROL/'ordinary_sgd_adamw/latest.pt')==f.EXPECTED59}
run.write(r/'results.json',m)
run.write(run.ROOT/'adamw_weight_factorial_metrics.json',m)
print(json.dumps({k:m[k] for k in ['probe_ratios','paired_uncertainty','factorial_effects','runtime','t59_ratios','control_t21_t59_negative_actual_step_counts']},indent=2))


# Final artifact integrity only; no scientific recomputation.
initial_vectors=np.load(f.CONTROL/'ordinary_sgd_adamw/raw_gradients/update_000.npz')
integrity={}
for arm in 'BCD':
 folder=r/arm; p=folder/'ordinary_sgd_adamw/latest.pt'
 saved=torch.load(p,map_location='cpu',weights_only=False)
 raw=np.load(folder/'ordinary_sgd_adamw/raw_gradients/update_000.npz')
 assert saved['cursor']==20 and saved['scheduler'] is None
 assert all(np.array_equal(initial_vectors[t],raw[t]) for t in run.TASKS)
 assert run.file_sha256(folder/'sampling_manifest.json')==m['protocol']['control_protocol']['manifest_sha256']
 assert all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point())
 assert all(x['learning_rate']==1e-4 for x in saved['logs'])
 assert saved['calibration']['lambda']==m['protocol']['arms'][arm]
 integrity[arm]={'latest_sha256':run.file_sha256(p),'initial_raw_gradients_bitwise_equal_control':True,
 'rng_and_optimizer_preserved':bool(saved['rng'] and saved['optimizer']), 'model_finite':True,
 'manifest_identical':True,'checkpoint0_sha256':run.file_sha256(folder/'ordinary_sgd_adamw/checkpoint_0.pt')}
m['integrity']=integrity
m['t59_bootstrap95']=sw.uncertainty(t59,baseline)
m['tests']={'passed':26,'ruff':'PASS with frozen-driver F401 exemption for unused numpy import','compileall':'PASS','diff_check':'PASS'}
m['recommendation']='Separately approved matched 90-update continuation of A and D at constant LR1e-4: D from t20, A from its preserved t59; no rerun of completed A updates. Do not promote weights before exact endpoint evidence.'
run.write(r/'results.json',m)
run.write(run.ROOT/'adamw_weight_factorial_metrics.json',m)
run.write(run.ROOT/'adamw_weight_factorial_protocol.json',m['protocol'])
