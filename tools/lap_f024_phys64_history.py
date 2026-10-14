"""Read-only native arithmetic trace from saved optimizer inputs, not training."""
import math
import time

import lap_f024_phys64 as exp
import numpy as np
import torch


def main():
    started=time.perf_counter()
    parent=torch.load(exp.f010.PARENT_CHECKPOINT,map_location='cpu',weights_only=False)
    final=torch.load(exp.OUT/'checkpoint_64.pt',map_location='cpu',weights_only=False)
    names=exp.read('protocol')['parameter_names']
    ids=parent['optimizer']['param_groups'][0]['params']
    assert len(ids)==len(names)
    weights={n:parent['model'][n].cuda().clone() for n in names}
    moments={n:parent['optimizer']['state'][i]['exp_avg'].cuda().clone() for n,i in zip(names,ids,strict=True)}
    second={n:parent['optimizer']['state'][i]['exp_avg_sq'].cuda().clone() for n,i in zip(names,ids,strict=True)}
    rows=[];milestones={}
    # Clone-only scalar equations mirror the installed non-fused AdamW arithmetic.
    # No optimizer is constructed or stepped, no model/data/objective is evaluated.
    for t,inputs in enumerate(final['native_gradients'],1):
        exp.check(60)
        decay=[];displacement=[]
        for n in names:
            p=weights[n];before=p.clone();g=torch.as_tensor(inputs[n],device='cuda',dtype=torch.float32)
            p.mul_(1-2.5e-5*.01)
            decay.append((p.double()-before.double()).cpu().numpy().ravel())
            moments[n].lerp_(g,1-.9)
            second[n].mul_(.999).addcmul_(g,g,value=1-.999)
            denom=(second[n].sqrt()/math.sqrt(1-.999**(128+t))).add_(1e-8)
            p.addcdiv_(moments[n],denom,value=-2.5e-5/(1-.9**(128+t)))
            displacement.append((p.double()-before.double()).cpu().numpy().ravel())
        d=np.concatenate(decay);delta=np.concatenate(displacement)
        rows.append({'local_update':t,'actual_F32_decay_displacement_norm':float(np.linalg.norm(d)),
            'actual_F32_trace_step_norm':float(np.linalg.norm(delta)),
            'logged_step_norm_gap':abs(float(np.linalg.norm(delta))-final['logs'][t-1]['step_norm'])})
        if t in (32,64):
            saved=torch.load(exp.OUT/f'checkpoint_{t}.pt',map_location='cpu',weights_only=False)
            milestone={'weights_exact':all(torch.equal(weights[n].cpu(),saved['model'][n]) for n in names),
                'first_moments_exact':all(torch.equal(moments[n].cpu(),saved['optimizer']['state'][i]['exp_avg']) for n,i in zip(names,ids,strict=True)),
                'second_moments_exact':all(torch.equal(second[n].cpu(),saved['optimizer']['state'][i]['exp_avg_sq']) for n,i in zip(names,ids,strict=True))}
            assert all(milestone.values()),'Native accounting trace differs from the actual saved trajectory'
            milestones[str(t)]=milestone
    exp.table('native_decay_accounting',rows)
    exp.write('native_history_receipt',{'milestones':milestones,'trace_only':True,'optimizer_created':False,
        'fresh_gradients_computed':False,'retained_updates':0,'active_seconds':time.perf_counter()-started,
        'native_inputs_checkpoint_sha256':exp.audit.sha(exp.OUT/'checkpoint_64.pt'),
        'driver_sha256':exp.audit.sha(__file__)})


if __name__=='__main__':main()
