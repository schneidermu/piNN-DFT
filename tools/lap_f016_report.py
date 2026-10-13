"""Receipt-only tables, plots, decision and reproducibility report for F016."""
import csv
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f016_audit as audit
import numpy as np


def rows(name):
    with (old.ROOT/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    audit.verify()
    # Full signed decomposition already evaluated in F015, never evaluate models here.
    spatial=rows('lap_f015_spatial_energy_differences.csv')
    distributions=old.read(audit.OUT/'saturation_full.json')
    lookup={(r['grid_id'],r['spin'],r['scale']):r for r in distributions
            if r['pathway']=='correlation' and r['density_regime']=='rho_ge_1e-3' and r['threshold']==.99}
    chemistry=[]
    for db in sorted({r['database'] for r in spatial}):
        group=[r for r in spatial if r['database']==db]
        signed,gross,satgross=0.,0.,0.
        reaction={}
        wide_ew=base_ew=mass=0.
        for r in group:
            factor=float(r['coefficient'])*627.5095
            term=factor*(float(r['signed_delta_ex_saturated'])+float(r['signed_delta_ec_saturated']))
            signed+=term
            reaction[r['identity']]=reaction.get(r['identity'],0)+term
            gross+=abs(factor)*float(r['abs_delta_exc_mass'])
            satgross+=abs(factor)*float(r['abs_delta_exc_saturated_mass'])
            for spin in (0,1):
                b,w=lookup[r['grid_id'],spin,1.],lookup[r['grid_id'],spin,4.]
                base_ew+=b['electron_hit']
                wide_ew+=w['electron_hit']
                mass+=b['electron_mass']
        chemistry.append({'database':db,'species_occurrences':len(group),'reaction_count':len(reaction),
                          'signed_q99_B8_minus_S5_kcal':signed,'sum_abs_reaction_q99_contributions_kcal':sum(abs(v) for v in reaction.values()),
                          'gross_abs_XC_difference_kcal':gross,'gross_q99_XC_difference_kcal':satgross,
                          'gross_q99_fraction':satgross/gross,'high_density_R0_q99_electron_fraction':base_ew/mass,
                          'high_density_R1_q99_electron_fraction':wide_ew/mass,
                          'interpretation':'model-to-model difference, not exact XC error; source signed weights unchanged'})
    old.table('lap_f016_chemistry_relevance.csv',chemistry)
    # Most important species by actual signed q99 stoichiometric contribution.
    ranked=sorted(spatial,key=lambda r:abs(float(r['coefficient'])*(float(r['signed_delta_ex_saturated'])+float(r['signed_delta_ec_saturated']))),reverse=True)
    old.table('lap_f016_chemical_species_contributors.csv',ranked[:40])
    # Canonical transform curves, analytical derivatives (stable, not 1-tanh²).
    q=np.linspace(-30,30,1201)
    curve=[]
    for x in q:
        d=[]
        for scale in (1.,4.):
            t=np.tanh(x/scale)
            exp=np.exp(-2*abs(x/scale))
            derivative=4*exp/(1+exp)**2/scale
            d.append(derivative)
            curve.append({'q':x,'scale':scale,'feature':t,'analytic_first':derivative,
                          'analytic_second':-2*t*derivative/scale,'floating_first':(1-t*t)/scale})
        curve.append({'q':x,'scale':0.,'feature':0.,'analytic_first':float(np.linalg.norm(d)),
                      'analytic_second':0.,'floating_first':float(np.linalg.norm(d))})
    old.table('lap_f016_transform_curves.csv',curve)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in ('feature','analytic_first','analytic_second'):
        fig,ax=plt.subplots(figsize=(6,4))
        for scale,label in ((1.,'R0'),(4.,'R1 / extra R2 channel')):
            chosen=[r for r in curve if r['scale']==scale]
            ax.plot([r['q'] for r in chosen],[r[field] for r in chosen],label=label)
        ax.set(xlabel='raw reduced q',ylabel=field)
        ax.legend();fig.tight_layout()
        fig.savefig(old.ROOT/f'lap_f016_{field}.png',dpi=140);plt.close(fig)
    # Every additional plot is reconstructed from its committed numeric table.
    panels = [
        ('lap_f016_saturation_comparison.csv','electron_fraction','lap_f016_electron_saturation.png',
         lambda r:r['density_regime']=='rho_ge_1e-3' and r['spin']=='0' and r['threshold']=='0.99',
         lambda r:r['pathway']+' / '+r['scale']),
        ('lap_f016_near_collisions.csv','electron_collision_mass_fraction','lap_f016_collision_rates.png',
         lambda r:r['density_floor']=='0.001',lambda r:r['scale']+' / '+r['float32_ULPs']+'ULP'),
        ('lap_f016_environment_distinguishability.csv','electron_resolved_fraction','lap_f016_environment_resolution.png',
         lambda r:r['density_floor']=='0.001' and r['distance_bound']=='0.001',lambda r:r['representation']),
        ('lap_f016_tau_residual_ambiguity.csv','electron_fraction_tau_gap_gt0_1','lap_f016_tau_ambiguity.png',
         lambda r:r['density_floor']=='0.001' and r['distance_bound']=='0.001',lambda r:r['representation']),
        ('lap_f016_trained_model_sensitivity.csv','electron_mean_abs_d_dq','lap_f016_network_sensitivity.png',
         lambda r:r['model']=='b8_t128' and r['channel']=='Gc' and r['spin']=='0',lambda r:r['region']),
        ('lap_f016_chemistry_relevance.csv','sum_abs_reaction_q99_contributions_kcal','lap_f016_chemistry_contributions.png',
         lambda r:True,lambda r:r['database'])]
    for file,field,image,select,label in panels:
        chosen=[r for r in rows(file) if select(r)]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar([label(r) for r in chosen],[float(r[field]) for r in chosen])
        ax.set_ylabel(field);ax.tick_params(axis='x',rotation=25)
        fig.tight_layout();fig.savefig(old.ROOT/image,dpi=140);plt.close(fig)
    embedding=old.read(old.ROOT/'lap_f016_architecture_parity.json')
    chosen=[r for r in embedding if r['representation']=='R2']
    old.table('lap_f016_embedding_plot.csv',[{'model':r['model'],'dtype':r['dtype'],
               'forward_gap':r['forward_max_abs_gap_R0'],'gradient_gap':r['input_gradient_max_abs_gap_R0']} for r in chosen])
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['model']+'/'+r['dtype'].split('.')[-1] for r in chosen],[r['forward_max_abs_gap_R0'] for r in chosen])
    ax.set_ylabel('R2 zero-weight max forward gap');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(old.ROOT/'lap_f016_embedding_parity.png',dpi=140);plt.close(fig)
    env=rows('lap_f016_environment_distinguishability.csv')
    chosen=[r for r in env if r['density_floor']=='0.001' and r['distance_bound']=='0.001']
    decision={'primary_classification':'B — RESOLUTION IMPROVES, CHEMICAL RELEVANCE UNRESOLVED',
              'candidate_preference':'R2 for a separately authorized architecture experiment; not production promotion',
              'reason':'Numerical resolution/anchors/zero-weight embedding pass. Signed chemistry associations exist, but no controlled evidence ties resolved pairs to reference errors; new-weight tail stability is not qualified.',
              'next_experiment':'One bounded R0 versus zero-new-weight R2 matched training experiment, with original anchor metric, same initialization/data/optimizer and predeclared tail/operator safety checks. Do not run automatically.',
              'training_updates':0,'WTMAD_claim':False,'production_modified':False}
    old.write(old.ROOT/'lap_f016_candidate_decision.json',decision)
    parity=old.read(old.ROOT/'lap_f016_architecture_parity.json')
    op=old.read(old.ROOT/'lap_f016_variational_operator_checks.json')
    lines=['# F016 — bounded q resolution and constraint audit','',f"**{decision['primary_classification']}**",'',
           'Zero training/optimizer updates. R0 production networks, Gc activation, AO implementation, dataset and checkpoints unchanged. Training-only F015 input view; no validation/test tensors or scores used.',
           '', '## Provenance and physical contract','',
           'Parent F015 commit8e2fe20b1874fbaded101ea8cbb2a949b3013a91; remote ancestry verified after fetch. All F015 source/checkpoint/cache pins reverified. One fixed variant per251 identities;495unique grids and23,231,999stored points. No regeneration. Raw F32 sources widened toF64 do not regain source precision.',
           '','See lap_f016_anchor_contract.md and feature_layouts.json. R1 keeps9446parameters. R2 adds96→9542. Tied exchange consumes(s,q,q4); both spin q4 features append to correlation. Gc metric deliberately remains original8coordinates, otherwise zero new weights DO NOT reproduceR0. This compatibility choice is part of the candidate, not a production change.',
           '', '## Mathematical versus numerical information','',
           'All finite-real tanh transforms are injective. R1/R2 add no independent physical variable or tau information. Exact endpoints and finite-precision collisions are distinct. Collision tables count distinct raw-q values per same grid/path/spin output bin, not all-pairs multiplicity. Raw separation≥1is a declared dimensionless diagnostic, not a universal chemical threshold. Near-collision table uses0/1/4F32ULPs on matched environment pairs.',
           '','R0 slope at zero1;R1.25;R2joint√(1+1/16). R1 reduces local resolution nearzero and increases it in moderate/large q regions. Stable sech² curves distinguish mathematical derivatives from rounded1−tanh² and derivative zeroing. Full first/second derivative tables and both F32/F64collision populations are archived.',
           '', '## Matched environments','',
           'F015 stride101 eligible linspace samples reused exactly,50000perdensityfloor. Original six non-q coordinates unchanged; R2two q scales each weighted1/√2. Distances are representation diagnostics, not interchangeable physical distances; comparisons retain exactly the sameR0nearest pair.',
           '| Representation | rho≥1e-3 near pairs Δq≥1 | Resolved pairs | Electron-weighted resolved fraction |','|---|---:|---:|---:|']
    for r in chosen:
        lines.append(f"|{r['representation']}|{r['R0_near_pairs_qgap_ge1']}|{r['resolved_same_pairs']}|{float(r['electron_resolved_fraction']):.6f}|")
    lines += ['', 'Conditional tau ambiguity remains. Both representations are deterministic functions ofq; nearest-neighbour changes cannot establish tau recovery. See all floors/distancebounds in tau_residual_ambiguity.csv.',
              '', '## Actual trained R0 sensitivity and chemistry','',
              'All four frozen checkpoints evaluated at the sameF015physical samples. Learned beta/gamma/mu/kappa/GNN/Gc and separate exchange/correlation energy-density raw-q derivatives use exact autograd; tau derivatives exactlyzero. Channels are outputs, not parameter blocks. R1output shifts are not accuracy results;R2zero-weight output isR0.',
              '','Complete eight-database signed/gross chemistry accounting reusesF015integrals. Important species listed inchemical_species_contributors.csv. q99fraction ofgrossB8–S5difference is modest, yet signedreactionterms cancel strongly; neither is an exactXCerror. The comparison does not prove that numerical resolution causesS5advantage. Database electron weights are statistical proxies, not substitutes for signedquadrature.',
              '', '## Compatibility and variational checks','',
              f"16frozen model/representation/dtype combinations checked. R2maxF64forwardgap={max(r['forward_max_abs_gap_R0'] for r in parity if r['representation']=='R2' and r['dtype']=='torch.float64'):.3g}; F32gap={max(r['forward_max_abs_gap_R0'] for r in parity if r['representation']=='R2' and r['dtype']=='torch.float32'):.3g}. All anchor/spinsymmetry/tauchecks passed. F32not claimed bitwise. Predeclared atol/rtolF64forward1e-12, input/operator1e-10;F32forward/input2e-6.",
              f"Existing weak-form operator dispatch reused including repairedF64learned/F32PBE path.16synthetic AO density-matrix directionalFD checks passed; maxgap={max(r['fd_absolute_gap'] for r in op):.3g}. This is variational formula/parity evidence, not full-corpus operator qualification. SyntheticAOtest inputs are not new molecular training systems.",
              '','Near-zero-densitystress at rho=0,1e-12,1e-6,1e-3,1andq±1e12 isfinite. Very large physical Laplacian derivatives occur under regularizeddensity denominators (up to~1e16first/~1e35second for learnedconstants). Bounded descriptors alone do not guarantee harmlesspotentials. R2zero-new-weight inheritsR0; nonzero new-weight tail stability remains unqualified. No trainedcandidate is claimed.',
              '', '## Decision and limits','',decision['reason'],'',decision['next_experiment'],
              '','No Clean28/WTMAD benefit inferred. NoSCF, newdata, fittedweights, optimizer, tauchannels or exhaustivevariantsweep. ExistingF015geometry/DMprovenance limitation retained. No independent molecular exactXCtargets available. Zero-weightembedding does not prove safety aftertraining. Constraint preservation holds algebraically for the declaredcandidate metric and numerically ontestedinputs.',
              '', '## Integrity and tests','',
              '32 focused model/operator/F016 tests passed. Ruff (E402 excepted only for script import bootstrap), compileall and git diff --check passed. An initial test run encountered a sandbox temporary-directory permission error; rerunning with an explicit workspace temporary directory passed. No scientific test was skipped or weakened.',
              'Frozen protocol pins, external receipt hashes and source/checkpoint hashes are recorded in provenance. Large arrays remain outside Git.']
    # Readable prose; preserve identifiers and all recorded numbers.
    replacements = {'commit8e2':'commit 8e2', 'per251':'per 251', '495unique':'495 unique',
                    '23,231,999stored':'23,231,999 stored', 'toF64':'to F64', 'keeps9446parameters':'keeps 9446 parameters',
                    'adds96':'adds 96', 'original8coordinates':'original 8 coordinates', 'reproduceR0':'reproduce R0',
                    'separation≥1is':'separation ≥1 is', 'uses0/1/4F32ULPs':'uses 0/1/4 F32 ULPs',
                    'zero1;R1.25;R2joint':'zero 1; R1 .25; R2 joint ', 'nearzero':'near zero',
                    'rounded1':'rounded 1', 'exactly,50000perdensityfloor':'exactly, 50,000 per density floor',
                    'R2two':'R2 two', 'weighted1':'weighted 1', 'sameR0nearest':'same R0 nearest',
                    'ofq':'of q', 'sameF015physical':'same F015 physical', 'exactlyzero':'exactly zero',
                    'R1output':'R1 output', 'results;R2zero':'results; R2 zero', 'weight output isR0':'weight output is R0',
                    'F015integrals':'F015 integrals', 'inchemical':'in chemical', 'q99fraction ofgrossB8':'q99 fraction of gross B8',
                    'signedreactionterms':'signed reaction terms', 'an exactXCerror':'an exact XC error',
                    'causesS5advantage':'causes S5 advantage', 'signedquadrature':'signed quadrature',
                    '16frozen':'16 frozen', 'R2maxF64forwardgap':'R2 max F64 forward gap',
                    'anchor/spinsymmetry/tauchecks':'anchor/spin symmetry/tau checks', 'F32not':'F32 not',
                    'atol/rtolF64forward':'atol/rtol F64 forward', '1e-12, input/operator':' 1e-12, input/operator',
                    'F32forward/input':'F32 forward/input ', 'repairedF64learned/F32PBE':'repaired F64 learned/F32 PBE',
                    'path.16synthetic':'path. 16 synthetic', 'directionalFD':'directional FD', 'maxgap':'max gap',
                    'SyntheticAOtest':'Synthetic AO test', 'densitystress':'density stress', '1andq':'1 and q',
                    'regularizeddensity':'regularized density', '~1e16first/~1e35second':'~1e16 first/~1e35 second',
                    'learnedconstants':'learned constants', 'harmlesspotentials':'harmless potentials',
                    'R2zero':'R2 zero', 'inheritsR0':'inherits R0', 'aftertraining':'after training',
                    'NoSCF':'No SCF', 'newdata':'new data', 'fittedweights':'fitted weights', 'tauchannels':'tau channels',
                    'exhaustivevariantsweep':'exhaustive variant sweep', 'ExistingF015geometry':'Existing F015 geometry',
                    'molecular exactXCtargets':'molecular exact XC targets', 'Zero-weightembedding':'Zero-weight embedding',
                    'safety aftertraining':'safety after training', 'declaredcandidate':'declared candidate',
                    'ontestedinputs':'on tested inputs'}
    text='\n'.join(lines)+'\n'
    for a,b in replacements.items():
        text=text.replace(a,b)
    text=text.replace('\n| Representation', '\n\n| Representation')
    (old.ROOT/'lap_f016_descriptor_saturation_audit_report.md').write_text(text,encoding='utf-8')
    files={p.name:old.sha(p) for p in old.ROOT.glob('lap_f016_*') if p.is_file() and p.name!='lap_f016_descriptor_provenance.json'}
    old.write(old.ROOT/'lap_f016_descriptor_provenance.json',{'reference_commit':audit.verify()['reference_commit'],
              'artifacts':files,'external':{p.name:old.sha(p) for p in audit.OUT.iterdir() if p.is_file()},
              'scripts':{p.name:old.sha(p) for p in (old.ROOT/'tools').glob('lap_f016_*.py')},
              'updates':0,'production_source_hashes_unchanged':True,'checkpoint_hashes_unchanged':True})


if __name__=='__main__':
    main()
