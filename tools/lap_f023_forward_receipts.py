"""Resume signed-error extraction with the qualified integration API/layout."""
import lap_f023_preflight as probe

from train_models import lap_vxc


def main():
    model, shadow, run, bundle = probe.setup()
    budget = probe.exp.Budget()
    original = lap_vxc.integrated_energy
    before = run.existing.digest(model)
    rng = probe.exp.b8._capture_rng()

    def compatible(energy, features, weights, *, chunk_size):
        # Diagnostic caller repair only: original factory uses [point,1,10].
        return original(energy, features[:, None, :], weights, point_chunk_size=chunk_size)

    try:
        lap_vxc.integrated_energy = compatible
        values = probe.originals(model, shadow, run, bundle, budget)
        assert len(values) == 174 and run.existing.digest(model) == before
        probe.exp.write('forward_receipt', {'complete': True, 'records': 174,
            'driver_sha256': probe.exp.audit.sha(__file__),
            'source_preflight_driver_sha256': probe.exp.audit.sha(probe.__file__),
            'correction': 'signed-error extraction caller keyword chunk_size mapped to qualified point_chunk_size; point,10 input given singleton axis as in original objective factory',
            'scalar_cache_parity': True, 'model_unchanged': True,
            'source_sha256': probe.exp.audit.sha(probe.exp.ROOT/'train_models/lap_vxc.py'),
            'forward_values_sha256': probe.exp.audit.sha(probe.exp.OUT/'original_forward_values.json')})
    finally:
        lap_vxc.integrated_energy = original
        probe.exp.b8._restore_rng(rng)
        assert run.existing.equal(rng, probe.exp.b8._capture_rng())
        bundle.close()
        budget.check()


if __name__ == '__main__':
    main()
