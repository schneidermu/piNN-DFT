"""Independent scalar-autograd check of the sole fixed size-weight law."""
import lap_f023_preflight as probe
import numpy as np
import torch


def main():
    exp = probe.exp
    model, shadow, run, bundle = probe.setup()
    rng = exp.b8._capture_rng()
    before = run.existing.digest(model)
    entry = exp.f010._batches()[0]
    cal = exp.read('calibration')
    counts = exp.audit.read(exp.OUT/'authenticated_electrons.json')
    budget = exp.Budget()
    rows = []
    try:
        with np.load(exp.OUT/'cached_arrays.npz') as cache:
            for task in ('ae17','exc'):
                budget.check(600)
                if task=='ae17':
                    reaction = exp.b8._load_reaction(bundle,'ae17',entry['ae17'])
                    factory = run.existing.make_reaction_objective(shadow,reaction,device='cuda',dtype=torch.float64,
                        dispersions=bundle.chemistry_dispersions())
                    weight = cal['c_A']*counts['ae17'][entry['ae17']['identity']]['Ne']**-.5
                    parameters = run.existing.named_trainable_parameters(shadow)
                else:
                    system = bundle.mrks().operator_system(entry['mrks_id'],device='cuda',dtype=torch.float32,chunk_size=4096)
                    factory = run.existing.core.make_mrks_objective_factories(model,system,point_chunk_size=256,
                        exc_chunk_size=4096,dispersions=exp.audit.read(bundle.root/'mrks/dispersion.json'))[0]
                    weight = cal['c_E']*counts['mrks'][entry['mrks_id']]['Ne']**-.5
                    parameters = run.existing.named_trainable_parameters(model)
                scalar = factory()*weight
                grads = torch.autograd.grad(scalar,tuple(parameters.values()),allow_unused=True)
                vector = np.concatenate([(torch.zeros_like(p) if g is None else g).detach().double().cpu().numpy().ravel()
                    for p,g in zip(parameters.values(),grads,strict=True)])
                reference = cache[task+'_new'][0]
                error = float(np.linalg.norm(vector-reference))
                bound = 1e-10+(1e-10*np.linalg.norm(reference) if task=='ae17' else 16*np.finfo(np.float32).eps*np.linalg.norm(reference))
                assert error <= bound and np.isfinite(vector).all()
                rows.append({'task':task,'weight':weight,'weighted_loss':float(scalar.detach()),'absolute_gap':error,
                    'relative_gap':error/max(np.linalg.norm(reference),1e-30),'bound':bound,'passed':True})
                del scalar,grads,vector,factory
                torch.cuda.empty_cache()
        exp.table('weighted_gradient_parity',rows)
    finally:
        exp.b8._restore_rng(rng)
        assert run.existing.equal(rng,exp.b8._capture_rng()) and run.existing.digest(model)==before
        bundle.close()
        budget.check()


if __name__=='__main__':
    main()
