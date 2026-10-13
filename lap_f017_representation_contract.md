# F017 reference-conditioned sufficiency claim

The actual functional is a signed quadrature sum
`E_xc(theta)=sum_g w_g rho_total,g F_PBE(rho_g,sigma_standard,g,C_theta(raw_g))`.
`C_theta` is the full 29-constant vector, not a histogram of q. PBE density,
polarization, gradient invariants and unchanged PW92/pass-through constants enter
outside the learned network. The learned exchange parameters and GNN use tied
spin-scaled (s,q) branches; correlation uses the eight descriptors, spin-symmetric
hidden exchange fusion, UEG/high-density/large-s anchors and the fixed original
eight-coordinate Gc distance metric. R2 adds bounded q/4 channels to network
inputs, with zero new weights initially, but not to that metric. Tau is excluded.

## Three distinct claims

1. **Exact equivalence:** two weighted physical input measures give the same
   observable for *every* admissible theta. Equality of the complete signed
   measures on raw rho/sigma/lapl (up to justified spin symmetries), including
   density prefactors and all nonlearned PBE ingredients, is a sufficient
   energy-equivalence condition. Equality of q or its histogram is insufficient.
   It is not proved to be a necessary condition for this finite architecture.
2. **Numerical collision:** distinct source tuples collapse under specified
   arithmetic. A q-channel collision is only a partial-input collision, not
   equality of the whole local energy or all functional parameter derivatives.
3. **Local finite-capacity limitation:** at a particular theta, reference residual
   correction is small under the certified Jacobian, frozen observation metric
   and parameter-radius bound. This says nothing about all distant parameters.

The audit tests reference residual alignment and attainable reductions, not a
universal approximation theorem. No parameter-uniform Lipschitz bound is assumed:
hidden weights are not globally norm-bounded. Close weighted feature moments
therefore cannot certify a contradiction with integrated reference energies.

## Operator observation space

`V=dE_xc/dP_total` includes the complete rho, gradient and Laplacian chains.
For local partials c,a,b the AO weak form includes phi*c*phi, both first-derivative
cross terms and `lap_phi*b*phi + phi*b*lap_phi + 2 grad_phi*b*grad_phi`.
Features, AO factors and signed quadrature are necessary. An energy histogram
does not determine AO response. RKS uses the average of both spin operators.
The repaired learned branch is F64, PBE branch F32, contraction F64.

For each molecule define `svec(S^-1/2 V S^-1/2)/sqrt(nAO)`, using diagonal weight1
and off-diagonal sqrt2. Its squared Euclidean residual equals the existing
Frobenius operator loss. Different molecules occupy distinct observation blocks;
their AO coordinates are NEVER directly subtracted. No gauge shift is applied.

Energy observations are Hartree with the qualified named dispersion correction.
The reference stored Exc includes that correction; electronic reference equals
stored target minus correction. Existing singleton Exc training loss is
627.5095*sqrt(1e-20+error_Ha^2), not squared energy residual. Energy-only LS is
explicitly a diagnostic. No convenient joint energy/operator scale is invented.

## Scope of evidence

Exact embedding is an algebraic identity checked numerically on actual systems.
An SVD projection with a declared cutoff is a truncated numerical result.
Full-representation moments and nearest pairs are empirical summaries, not
complete measures or equivalence certificates. No local pointwise XC reference
labels are inferred from integrated targets. Minnesota reaction references do
not provide exact molecular XC energies. Finite training data cannot establish
global sufficiency or impossibility. No architecture-training recommendation may
rest on q collisions alone or on an unqualified incomplete operator projection.
