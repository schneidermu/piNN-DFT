PARTIAL — F018 completed with documented limitations

# H2 operator-JVP precision and mixed derivatives

**Scientific classification A: analytical JVP qualified; native FD precision limit identified.** Original F017 FAIL remains unchanged. No architectural/training superiority claim.

## Original failure and observation definition

Exact B8t128 checkpoint1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b; H2,42508points,60AOs. Exact CUDA/F64 linspace direction and64+32 tied coordinate mapping recovered from pinned F017 source. Source DM/AO/cache/overlap identities verified; only H2 loaded for calculations. R0/R2 zero-weight energy/operator embedding passed; parameters/buffers/RNG restored. No source checkpoints or datasets changed.

F017 evaluated1831 coordinates: **one energy coordinate plus1830operator coordinates**. Its original error is reproduced exactly: absolute1.99747202024e-05, relative0.0167734904151. Pure operator error at1e-4 is9.50963437479e-06, relative0.00801792438472. Energy-coordinate error1.75657707266e-05 contributes77.334452%of squared joint error. Mislabeling that joint error as operator-only was an audit-description defect; it does not erase the remaining operator FD FAIL.

## Independent analytical routes and failed diagnostic helper

Operator-first double-reverse JVP uses the existing complete weak form, learnedF64/PBEF32/AOF64 boundary. Independently, explicit grad_theta(E)·v followed by grad_P uses the direct anchored AO density-matrix map and no weak-form assembly. It preserves the source DM/base features and all rho/gradient/Laplacian chains. Native-PBE mixed route differs by2.34809326752e-11, relative1.97976426067e-08: **PASS** under frozen1e-8+1e-5·norm mixed tolerance. A pureF64 arithmetic copy also agrees; it is a sensitivity check, not an exact native oracle.

The initial energy-first nested `autograd.functional.jvp(create_graph=True)` helper produced a77%mixed discrepancy, and a forward-AD helper produced90%. Those failures are preserved externally. Explicit parameter-gradient dot v produces the correct mixed derivative. The demonstrated divergence is in those higher-order diagnostic graph formulations, not in the physical operator; no general PyTorch bug or particular activation cause is claimed without a minimal proof. Same R2/PBE primitives remain shared; independent AO density differentiation validates the weak-form chain/contraction, not every possible primitive globally.

Qualification concerns the differentiable mathematical/AD functional at this source state. It does not assert that the discrete bitwise Float32 mapping has an ordinary infinitesimal derivative, nor that all parameter states are qualified. The full observation-space/operator identity is tested at the original physical point only.

Three native scalar-adjoint comparisons exceed the deliberately strict1e-12+1e-8·abs tolerance by only5–9e-12; their FAIL flags remain. With promoted F64 tangent/cotangent carriers, the primal operator and JVP are unchanged (JVP difference0), and all three SAME probes pass the SAME gate, errors≤4.53e-13. Measured native/promoted pullback triangle bounds cover native discrepancies. This isolates numerical cotangent reduction/cast effects, not a relaxed threshold.

## Frozen FD ladder and numerical floor

|h|Operator absolute error|Operator relative error|Energy coordinate error|
|---|---:|---:|---:|
|0.01|1.2917379025868198e-07|0.00010891119910218232|8.961050767072225e-08|
|0.003|5.354995350443122e-07|0.0004514994594765887|5.365290616871565e-07|
|0.001|1.2009996242164342e-06|0.0010126071932451543|6.65707988501817e-07|
|0.0003|3.3726198434721258e-06|0.002843580501375555|1.5932150411448402e-06|
|0.0001|9.509634374789425e-06|0.00801792438471888|1.7565770726636197e-05|
|3e-05|5.031505708148252e-05|0.0424224851548064|2.4104181835468732e-05|
|1e-05|9.621695750641712e-05|0.08112407474459234|6.104952121587334e-05|

Native discrepancy scales approximately h^-0.962; error numerator2h·gap ranges1.9e-09–3.21e-09. This supports a rounding-limited regime, not a verified native O(h²) region. The F64 sensitivity-copy errors are much smaller, but realized F32 perturbations and remaining noise prevent a clean certified second-order interval. No best h is substituted into the historical gate.

All requested/realized perturbations are recorded. Midpoint shift is exactly0 because new weights start at0 and±casts are symmetric. At original h, realized-direction JVP correction is1.93e-10, orders below9.51e-6. Native repeated forward operators are bitwise identical (3/3): stochastic forward noise0; deterministic arithmetic resolution is not0.

At original h the saved-vector, counterfactual-conditional triangle interval for native operator discrepancy is[9.50926840756e-06,9.5100853323e-06]. It is a measured local comparison, not a parameter-uniform rigorous floating-point bound. Observed native FD uncertainty is9.51e-6. Final F64 subtraction scale eps64·baseline/h is about9.28e-13, assuming perturbed norms of similar order; this alone cannot explain the measured discrepancy. Native PBE/chain arithmetic jointly explains the difference from the F64 copy; learned source values, AO/overlap arithmetic remain common. Pure AO error is not separately isolated.

Fixed PW92/pass-through contributions are parameter-independent algebraically. Their isolated matrix cancellation was NOT numerically measured before stopping. Parameter-dependent learned coefficients enter PBE arithmetic, so saying “PBE is independent oftheta” would be incorrect. No component is assigned an unsupported measured error.

## Resource limitation and answers

The main numerical batch took93.79s, plus bounded single-system localization jobs. However a late audit-wall check found2083.79s versus the1800s limit: **the overall wall budget was exceeded before checking it**. Scientific work was stopped; this is a limitation, not a claim of budget compliance. First omitted diagnostic: isolated fixed-PW92 matrix cancellation; further continuous-parameter F64 O(h²) study also not run. Remaining work consists only of reporting, required tests/hash checks/Git.

1:Original failure reproduced.2:Explicit independent mixed derivative agrees; initial nested-JVP diagnostic does not.3:Native strict adjoint flags FAIL; identical-primal/JVP promoted checks PASS, with measured numerical bounds.4:FD error grows as h shrinks.5:No certified O(h²) interval.6:Repeated-forward variation0; h-specific FD floor9.51e-6 at1e-4.7:Storage rounding contribution≈1.93e-10.8:FinalF64 subtraction alone is insufficient; native arithmetic/cancellation jointly measured.9:Parameterized PBE arithmetic affects FD; fixed contribution separately unmeasured.10:Independent AO route agrees2.35e-11; no isolated pure-assembly bound.11:Joint-vector mislabel and unsupported nested diagnostic graph identified; no production operator patch.12:Joint1.677349% and operator0.801792% decomposed and consistent with measured rounding-limited native FD.13:Original frozen operator JVP independently qualified locally.14:Global error bounds, primitive-level attribution and clean second-order interval remain uncertain.15:Freeze a bounded H2-only matrix-free JVP/VJP feasibility protocol with promoted carriers and explicit energy-first verification; cost it before any15-system expansion. No serial15×96 sweep or training.

Scientific classification:A
Original F017 FD gate:FAIL
Independent analytical qualification:PASS
Numerical precision floor:observed native operator FD error9.509634e-6 at h1e-4; local conditional bound only
H2 systems evaluated:1
New parameter directions evaluated:1 (realized-rounding corrections belong to the same frozen direction)
Optimizer updates:0
Training performed:NO
Production architecture modified:NO
Clean28 accessed:NO
SCF executed:NO
F018 commit/Git push/remote HEAD:reported after commit and remote verification
