# F016 immutable constraint and feature contract

Production remains `pcPBELMLOptimizerV2Lap`, version 1. F016 clones have separate
architecture and descriptor identifiers and are never serialized as production
checkpoints. R0/R1 consume eight correlation inputs; R2 consumes ten. Exchange
consumes only (s,q) per spin, or (s,q,q4) for R2, through ONE tied network.

Raw input is rho_a,rho_b,sigma_a,sigma_total,sigma_b,tau_a,tau_b,lap_a,lap_b.
Tau columns are never read. n=(rho+EPS_RHO)^(1/3),
s=(sqrt(sigma+EPS_SIGMA)-sqrt(EPS_SIGMA))/(2(3pi²)^(1/3)(rho+EPS_RHO)^(4/3)),
q=lap/(4(3pi²)^(2/3)(rho+EPS_RHO)^(5/3)); zeta=(rho_a-rho_b)/(rho_total+EPS_RHO).
Correlation bounds n,s,q by tanh; zeta is unchanged. Exchange first scales
rho/lap by2 and sigma by4; it does not consume n,s_total,zeta. R1 substitutes
tanh(q/4) only. R2 appends q4_a,q4_b after the original eight coordinates.

## Anchor actions and algebraic preservation

* Zero sigma/UEG: zero all s and both original/widened q channels; preserve n,zeta.
  Real and anchor exchange features coincide. beta=beta_activation(0)=1,
  mu=shifted_elu(0)=1, GNN=tanh(0)=0, Gc=activate_gc(1)=1.
* High density: set only n_a,n_b to1; preserve all s,q,q4,zeta and exchange fusion.
  gamma_activation(real-anchor)=1 at this anchor; Gc=1.
* Rapid gradients: set only s_a,s_total,s_b to1; KEEP curvature and exchange fusion.
  The corresponding distance-weighted Gc correction equals1 at the anchor.
* Spin permutation swaps n_a/n_b, s_a/s_b, q_a/q_b, q4_a/q4_b, negates zeta,
  preserves s_total. Tied exchange weights plus symmetric hidden-feature mean
  and symmetric correlation averaging ensure permutation equivariance.
* Columns2:22 remain constant PBE passthrough. All parameter activations, bounds,
  Gc shifted-ELU, regularizers, source rounding and precision boundaries remain.
* All transforms are smooth odd bounded maps with phi(0)=0 and endpoints ±1.
  Composition with the unchanged scalar XC expression remains tau-independent
  and differentiable in rho/sigma/lapl. The existing AO weak form retains both
  Laplacian terms and defines Vxc=dExc/dP_total. No partial-rho proxy is used.

The three-anchor correction uses nonnegative distance weights and its existing
safe zero-denominator rule. R1 applies that formula to its eight coordinates.
For R2 the metric is explicitly the ORIGINAL FIRST EIGHT coordinates, including
original tanh(q). New q4 features enter every real/anchor network evaluation but
not the metric. At an anchor, the corresponding corrected network differences
vanish independent of weights. This is necessary for exact R0 embedding: including
q4 in the metric changes Gc even with zero new network weights.

## Mapping and limits

R1 keeps9446 parameters but generally cannot reproduce a trained R0 checkpoint.
R2 extends correlation first-layer columns8→10 and tied exchange columns2→3;
append zero weights and copy every original parameter/buffer. At h_dim32 this adds
96 parameters. Old contractions are evaluated separately to preserve reduction
order; zero new contributions make R0 an algebraically exact special case.
Finite-precision parity must still be measured, especially F32 strided kernels.

Bounded features do not guarantee bounded physical Laplacian derivatives in
tails: dq/dlap grows as(rho+EPS_RHO)^(-5/3). Stress checks measure that explicitly.
R1 slope at zero is1/4 vs R0's1; R2 joint slope is sqrt(1+1/16), so widening is
not uniformly better. In exact arithmetic all maps encode the SAME physical q.
No tau recovery, improved chemistry accuracy, or global expressivity follows.
