"""Optional H correlation component on the original fixed evaluation grid."""
import lap_f015_inputs as parts
import lap_f023_preflight as probe
import torch

from train_models.reaction_energy_calculation import calculate_xc_energy


def main():
    exp=probe.exp
    model,shadow,run,bundle=probe.setup()
    before=run.existing.digest(model)
    rng=exp.b8._capture_rng()
    try:
        spec=next(r for r in exp.audit.read(exp.ROOT/'relchem_joint_epoch_evaluation_manifest.json')['rows']
            if r['task']=='ae17' and bundle.reactions[r['identity']]['components']==['H_ae17'])
        reaction=exp.b8._load_reaction(bundle,'ae17',{**spec,'database':'AE17'})
        ex,ec,_density=parts.integrated_parts(shadow,reaction['Grid'],reaction['Densities'],reaction['Gradients'],reaction['Weights'])
        with torch.no_grad():
            direct,_local=calculate_xc_energy(reaction,shadow(reaction['Grid']),'cuda','GGA','PBE',add_dispersion=False)
        assert abs(ex+ec-float(direct)) <= 1e-10
        exp.write('hydrogen_components',{'scope':'B8 fixed evaluation H, matched-F64 chemistry arithmetic',
            'identity':spec['identity'],'variant':spec['variant'],'checkpoint_sha256':exp.dual.CHECKPOINT,
            'exchange_Ha':ex,'correlation_Ha':ec,'xc_Ha':ex+ec,'direct_xc_Ha':float(direct),
            'decomposition_parity_absolute_Ha':abs(ex+ec-float(direct)),
            'Hartree_plus_XC':'UNAVAILABLE; no separately authenticated Hartree field',
            'correlation_reference':'NOT PROVIDED; computed functional component only, not a new target',
            'universal_one_electron_cancellation_claimed':False,'model_unchanged':True})
    finally:
        exp.b8._restore_rng(rng)
        assert run.existing.digest(model)==before and run.existing.equal(rng,exp.b8._capture_rng())
        bundle.close()
        exp.Budget().check()


if __name__=='__main__':
    main()
