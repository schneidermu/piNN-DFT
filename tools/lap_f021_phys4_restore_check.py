"""Reversible native checkpoint/constraint preflight; optimizer steps zero."""
import copy

import lap_f021_phys4 as exp
import lap_f021_phys4_eval as evaluation
import lap_f021_phys4_run as run
import torch


def main():
    budget = exp.Budget()
    try:
        prefix, source, _scales = exp.freeze()
        model, shadow, producer, _names = exp.loaded(source)
        parameters = exp.f010._check_model(model)
        parent = torch.load(exp.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
        optimizer = exp.audit.geometry.arm._new_optimizer(parameters)
        exp.f010.continuation.prepare_loaded_optimizer(optimizer, parent['optimizer'], 128, 1e-4)
        exp.b8._restore_rng(parent['rng'])
        model_before = producer.existing.digest(model)
        moment_before = exp.f010.continuation.moment_sha(optimizer.state_dict())
        rng_before = exp.b8._capture_rng()
        snapshot = copy.deepcopy(optimizer.state_dict())
        target = exp.OUT / 'ordinary_sgd_adamw'
        target.mkdir(exist_ok=True)
        path = exp.OUT / 'restore_check.pt'
        with run.training_adapter(prefix, budget):
            exp.f010._save_checkpoint(path, model, optimizer, 0, [], exp.audit.geometry.C010, exp.audit.sha(exp.f010.PARENT_CHECKPOINT))
            with torch.no_grad():
                next(iter(parameters.values())).add_(1)
                next(iter(optimizer.state.values()))['exp_avg'].add_(1)
            torch.rand(1)
            cursor, logs = exp.f010._load_checkpoint(path, model, optimizer, exp.audit.geometry.C010)
        assert cursor == 0 and logs == []
        assert producer.existing.digest(model) == model_before
        assert exp.f010.continuation.moment_sha(optimizer.state_dict()) == moment_before == exp.MOMENT
        assert producer.existing.equal(snapshot, optimizer.state_dict())
        assert producer.existing.equal(rng_before, exp.b8._capture_rng())
        checked = evaluation.constraints(model, shadow)
        exp.write('resume_parity', {'passed': True, 'model_restored': True, 'optimizer_restored': True,
                                    'rng_restored': True, 'moment_sha256': moment_before,
                                    'constraints': checked, 'optimizer_updates': 0,
                                    't0_checkpoint_sha256': exp.audit.sha(target / 'checkpoint_0.pt')})
    finally:
        budget.check()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
