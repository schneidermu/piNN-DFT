"""Sequential normalized relchem accumulation; unchanged qualified task/AdamW code."""
import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import adamw_lr_sweep as sweep

run = sweep.run
BASE_MEASURE = run.measure
OUT = run.ROOT.parent / 'lap_relchem_accumulation_20261008'
CONTROL = sweep.OUT / '1e-4'
KS = (1, 2, 4, 6)


def add_gradient(total, gradient):
    if total is None:
        return {n: g.detach().double().clone() for n, g in gradient.items()}
    assert tuple(total) == tuple(gradient)
    for n, g in gradient.items():
        total[n].add_(g.detach().double())
    return total


def mean_gradient(total, count):
    if count < 1:
        raise ValueError('Positive effective batch required')
    return {n: g / count for n, g in total.items()}


def extra_samples(rows, index):
    return [run.sample_pair(rows, 'relchem', index, seed=941011+j) for j in range(5)]


def singleton(model, shadow, bundle, row, dispersion, chunk=16384):
    start = time.perf_counter()
    reaction = bundle.chemistry('train_relchem').load_variant(row['identity'], row['variant'])
    reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
    ngrid = len(reaction['Grid'])
    if ngrid > 131072:
        reaction['model_point_chunk_size'] = chunk
    torch.cuda.synchronize()
    loading = time.perf_counter()-start
    start = time.perf_counter()
    loss, gradient = run.chemistry(model, shadow, reaction, dispersion).value_and_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    del reaction
    return loss, gradient, loading, elapsed, ngrid


def accumulated_measure(k):
    def measure(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk=None):
        record, raw = BASE_MEASURE(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk)
        if k == 1:
            return record, raw
        total = add_gradient(None, raw['relchem'])
        losses = [record['losses']['relchem']]
        grids = [record['grid_points']['relchem']]
        samples = [entry['relchem'], *entry['relchem_extra'][:k-1]]
        for row in samples[1:]:
            loss, gradient, loading, elapsed, ngrid = singleton(model, shadow, bundle, row, dispersion)
            total = add_gradient(total, gradient)
            losses.append(loss)
            grids.append(ngrid)
            record['seconds']['chemistry_loading_transfers'] += loading
            record['seconds']['relchem'] += elapsed
            del gradient
        raw['relchem'] = mean_gradient(total, k)
        record['losses']['relchem'] = sum(losses)/k
        record['norms']['relchem'] = float(torch.cat([g.flatten() for g in raw['relchem'].values()]).norm())
        record['relchem_batch'] = samples
        record['relchem_grids'] = grids
        record['effective_relchem_batch'] = k
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return record, raw
    return measure


def freeze():
    sweep.verify()
    if OUT.exists():
        return
    OUT.mkdir()
    bundle = run.PublicationDataset(run.DATA)
    try:
        rows = list(bundle.reactions.values())
        original = run.read(CONTROL/'sampling_manifest.json')
        manifest = [{**e, 'relchem_extra': extra_samples(rows,e['cursor'])} for e in original]
        rng = random.Random(941101)
        diagnostic = [[run.sample_pair(rows,'relchem',i,seed=941100+j) for j in range(6)] for i in range(4)]
        reference = [run.sample_pair(rows,'relchem',i,seed=941200) for i in range(12)]
        other = {'cursor':0, 'relchem':diagnostic[0][0], 'ae17':run.sample_pair(rows,'ae17',0,seed=941300),
                 'mrks_id':rng.choice(sorted(bundle.systems))}
    finally:
        bundle.close()
    saved = torch.load(CONTROL/'ordinary_sgd_adamw/latest.pt',map_location='cpu',weights_only=False)
    large = max(saved['logs'],key=lambda x:x['grid_points']['relchem'])['sample']['relchem']
    run.write(OUT/'protocol.json',{'reference_commit':'fda7be602a5f0ec794912e57df328df303fba77a',
        'ks':list(KS),'updates':20,'lr':1e-4,'scheduler':'constant','control_reused':True,
        'lambda':run.read(CONTROL/'calibration.json')['lambda'],
        'trainer_sha256':run.file_sha256(run.ROOT/'train_lap_microbatch.py'),'driver_sha256':run.file_sha256(__file__),
        'control_protocol':run.read(CONTROL/'ordinary_sgd_adamw/protocol.json'),
        'historical_cursor59_sha256':run.file_sha256(CONTROL/'ordinary_sgd_adamw/latest.pt'),
        'sampling':'Primary unchanged; five independent uniform replacement streams seed941011..941015; nested prefixes',
        'diagnostic_batches':diagnostic,'reference_samples':reference,'other_diagnostic_tasks':other,
        'large_chunk_sample':large,'parity_tolerances':{'mean_relative_l2':1e-12,'chunk_gradient_relative_l2':1e-10,'chunk_loss_relative':1e-10},
        'bootstrap':'Existing paired5000 within-stratum resamples seed931002; limited probe only',
        'variance':'Four independent nested batches per K; separate12-sample reference is an estimate, not full251',
        'per_arm_runtime_cap_seconds':1800})
    run.write(OUT/'sampling_manifest.json',manifest)
    for k in (2,4,6):
        folder = OUT/f'K{k}'
        folder.mkdir()
        shutil.copyfile(CONTROL/'protocol.json',folder/'protocol.json')
        shutil.copyfile(OUT/'sampling_manifest.json',folder/'sampling_manifest.json')
        calibration = run.read(CONTROL/'calibration.json')
        calibration['manifest_sha256'] = run.file_sha256(folder/'sampling_manifest.json')
        run.write(folder/'calibration.json',calibration)
        protocol = run.read(folder/'protocol.json')
        protocol.update(effective_relchem_batch=k,accumulation_driver_sha256=run.file_sha256(__file__))
        run.write(folder/'protocol.json',protocol)


def flatten(gradient):
    return torch.cat([g.flatten() for g in gradient.values()]).detach().cpu().numpy()


def qualify():
    if (OUT/'qualification.json').exists():
        return
    protocol = run.read(OUT/'protocol.json')
    model,shadow = run.model_at(run.read(CONTROL/'protocol.json')['initial_state'])
    digest = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        _record,raw = accumulated_measure(1)(model,shadow,bundle,run.read(CONTROL/'sampling_manifest.json')[0],
             dispersion,run.read(run.DATA/'mrks/dispersion.json'),4096)
        with np.load(CONTROL/'ordinary_sgd_adamw/raw_gradients/update_000.npz') as expected:
            assert all(np.array_equal(flatten(raw[t]),expected[t]) for t in run.TASKS)
        del raw
        large = protocol['large_chunk_sample']
        a,ga,*_ = singleton(model,shadow,bundle,large,dispersion,16384)
        b,gb,*_ = singleton(model,shadow,bundle,large,dispersion,32768)
        av,bv = flatten(ga),flatten(gb)
        err = float(np.linalg.norm(av-bv)/np.linalg.norm(av))
        assert err<=1e-10 and abs(a-b)/abs(a)<=1e-10
        del ga,gb
        reference=[]; all_batches=[]; mean_errors=[]; timings=[]
        for row in protocol['reference_samples']:
            _,g,_,_,_=singleton(model,shadow,bundle,row,dispersion)
            reference.append(flatten(g)); del g
        for batch in protocol['diagnostic_batches']:
            total=None; vectors=[]; prefixes={}; seconds=[]
            for j,row in enumerate(batch):
                _,g,loading,elapsed,_=singleton(model,shadow,bundle,row,dispersion)
                vectors.append(flatten(g)); total=add_gradient(total,g); seconds.append(loading+elapsed); del g
                if j+1 in KS:
                    measured=flatten(mean_gradient(total,j+1)); direct=np.stack(vectors).mean(axis=0)
                    error=float(np.linalg.norm(measured-direct)/np.linalg.norm(direct))
                    assert error<=1e-12
                    mean_errors.append(error); prefixes[str(j+1)]=measured
            all_batches.append(prefixes); timings.append(seconds)
        entry=protocol['other_diagnostic_tasks']
        _,other=BASE_MEASURE(model,shadow,bundle,entry,dispersion,run.read(run.DATA/'mrks/dispersion.json'),4096)
        other_sum=sum(flatten(other[t])*protocol['lambda'][t] for t in ('ae17','exc','op'))
        summaries={}; ref=np.stack(reference).mean(axis=0)
        for k in KS:
            g=np.stack([p[str(k)] for p in all_batches]); norms=np.linalg.norm(g,axis=1)
            unit=g/norms[:,None]; cosines=unit@unit.T
            pairs=[np.sum((g[i]-g[j])**2)/2 for i in range(4) for j in range(i)]
            joint=g*protocol['lambda']['relchem']+other_sum
            summaries[str(k)]={'norms':norms.tolist(),'norm_cv':float(norms.std()/norms.mean()),
                'mean_pairwise_gradient_variance':float(np.mean(pairs)),
                'mean_pairwise_direction_cosine':float(np.mean([cosines[i,j] for i in range(4) for j in range(i)])),
                'cosine_to_reference':(g@ref/norms/np.linalg.norm(ref)).tolist(),
                'relchem_progress_joint':np.sum(g*joint,axis=1).tolist(),
                'other_progress_joint':{t:(joint@flatten(other[t])/np.linalg.norm(flatten(other[t]))/np.linalg.norm(joint,axis=1)).tolist() for t in ('ae17','exc','op')},
                'mean_relchem_seconds':float(np.mean([sum(v[:k]) for v in timings]))}
        np.savez(OUT/'initial_relchem_diagnostic_gradients.npz',reference=np.stack(reference),
                 **{f'K{k}':np.stack([p[str(k)] for p in all_batches]) for k in KS})
        assert run.existing.digest(model)==digest
        run.write(OUT/'qualification.json',{'k1_raw_gradients_bitwise_equal':True,'mean_max_relative_error':max(mean_errors),
            'large_chunk_gradient_relative_error':err,'large_chunk_loss_relative_error':abs(a-b)/abs(a),
            'large_sample':large,'large_loss':a,'model_unchanged':True,'variance':summaries,
            'diagnostic_arrays_sha256':run.file_sha256(OUT/'initial_relchem_diagnostic_gradients.npz')})
    finally:
        bundle.close()


def main(stage):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    freeze()
    assert run.read(OUT/'protocol.json')['driver_sha256']==run.file_sha256(__file__)
    if stage=='qualify':
        qualify()
        print('QUALIFICATION_COMPLETE',flush=True)
        return
    assert run.read(OUT/'qualification.json')['k1_raw_gradients_bitwise_equal']
    for k in (2,4,6):
        folder=OUT/f'K{k}'
        run.measure=accumulated_measure(k)
        print('ARM_START',k,flush=True)
        try:
            run.train(folder,90,stop_at=20,runtime_seconds=1800,learning_rate=1e-4,constant_lr=True,diagnostics=True)
        finally:
            run.measure=BASE_MEASURE
        saved=torch.load(folder/'ordinary_sgd_adamw/latest.pt',map_location='cpu',weights_only=False)
        assert saved['cursor']==20
        sweep.probe(folder)
        print('ARM_COMPLETE',k,flush=True)
        torch.cuda.empty_cache()
    assert run.file_sha256(CONTROL/'ordinary_sgd_adamw/latest.pt')==run.read(OUT/'protocol.json')['historical_cursor59_sha256']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('qualify','train'))
    main(parser.parse_args().stage)

