# Four-task PCD qualification

## Problem
Mixed chemistry hides conflicting AE17 and relative-energy gradients.
## Invariants
Explicit order relchem, ae17, exc, op; relchem primary. Canonical EMA, KKT QP, magnitude rescale and fallback. Matched-F64 chemistry scalar and derivative; unchanged E/operator semantics. No S5 edits or contaminated continuation.
## Interfaces and state
Extend existing aggregators, task helpers, protocol and Armijo. Legacy three-task defaults unchanged. New protocol version pins four-task order and sampling identity. Checkpoints serialize K EMA entries. All-reduce raw gradients before PCD; trial losses use rank mean. Rejected trials restore model/RNG/state; accepted iterations advance once.
## Acceptance
K3 prechange parity; K4 independent reference/KKT; zeros/infeasibility; resume/DDP; all four Armijo conditions plus realized descent. Estimator audit uses all268 individual gradients and 32 pinned stratified draws. Frozen geometry and Armijo must pass before at most5 accepted updates. Stop on failed sample, no tuning.
## Non-goals
New optimizer, architecture/loss changes, tau sweep, Diet, Slurm, production training, all90 AO cache.
