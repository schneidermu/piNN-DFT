# Chemistry-estimator fidelity Arena

Starting commit: `bc3aa4f5e684e0e79b4c71288ce0b751114322e0`. Final commit is the Git commit containing this report. Selection: **NO K≤8 QUALIFIES**.

Read-only local screen on the same twelve frozen pre-update0/5/9 states. Full251 chemistry and AE17/Exc/operator gradients are reused by SHA; no model direction is applied and no production code changes.

Replicate manifest SHA: `59913c53189401e92bcfbd40c07763e598519bd4021f3b0c50fa63669bca9e29`. Protocol SHA: `925ff4fcd5dc7e3c38dc181d856921bbf6ada760f3cc56ad40feb998e78c9eeb`. Sixteen seeds41000–41015 generate independent per-DB uniform permutations, shared across states and nested by K. Sampling never reads full-gradient values. Canonical variants match the deterministic full251 reference; weights are n_d/(251 m_d), where m_d is the legal sample count.

Finite-population constraint: ABDE4 has4 reactions. Literal K8 sampling without replacement is infeasible and was not evaluated. The user explicitly confirmed strict K-per-DB sampling and K8 infeasibility. This matches the already-frozen policy; no samples, weights, or gates changed after results. K1/2/4 retain their exact requested sizes.

The frozen gates require >=95% overall success and >=90% in every state. With16 replicates, every state needs at least15 successes;192 total draws require at least183 successes. The predeclared catastrophic lower-tail floor is p_full>=−0.02, inherited from the offline Arena Tier2 safety boundary. Positive progress and all success counts use raw strict signs, without epsilon. Every replicate must also satisfy the unchanged solver/KKT qualification.

| K | Fullchem descent | Worst state fraction | Cosine min/p05/median | p_full min/p05/median | Median gamma | Backward-count/K1 | Qualifies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 39/192 | 0 | -0.86611757/-0.72366435/0.45037529 | -0.98538372/-0.89387792/-0.39136438 | 0.55066606 | 1.0 | False |
| 2 | 43/192 | 0.0625 | -0.69858816/-0.21160787/0.65194259 | -0.86344942/-0.73890281/-0.24913962 | 0.38758849 | 2.0 | False |
| 4 | 110/192 | 0.4375 | -0.71107635/-0.41147166/0.91758747 | -0.9350563/-0.72745294/0.054256302 | 0.23959158 | 4.0 | False |
| 8 | INFEASIBLE | — | — | — | — | — | — |

Per-state success rates:

| State | K1 | K2 | K4 | K8 |
| --- | --- | --- | --- | --- |
| 11_P67_u0 | 0.1875 | 0.1875 | 0.6875 | INFEASIBLE |
| 11_P67_u5 | 0.1875 | 0.1875 | 0.6875 | INFEASIBLE |
| 11_P67_u9 | 0.25 | 0.1875 | 0.625 | INFEASIBLE |
| 11_P536_u0 | 0.0 | 0.0625 | 0.5 | INFEASIBLE |
| 11_P536_u5 | 0.0 | 0.0625 | 0.4375 | INFEASIBLE |
| 11_P536_u9 | 0.125 | 0.125 | 0.5625 | INFEASIBLE |
| 23_P67_u0 | 0.25 | 0.125 | 0.5 | INFEASIBLE |
| 23_P67_u5 | 0.25 | 0.125 | 0.5 | INFEASIBLE |
| 23_P67_u9 | 0.25 | 0.125 | 0.5 | INFEASIBLE |
| 23_P536_u0 | 0.3125 | 0.5 | 0.625 | INFEASIBLE |
| 23_P536_u5 | 0.3125 | 0.5 | 0.625 | INFEASIBLE |
| 23_P536_u9 | 0.3125 | 0.5 | 0.625 | INFEASIBLE |

Cost: exact reaction-backward counts are8,16,32 for K1/K2/K4, compared with251 for the full reference. K8 is infeasible and has no measured or projected candidate cost. Summed measured singleton-gradient timings are diagnostic prospective costs, not end-to-end throughput benchmarks. FULL251 time is projected from per-DB mean singleton timings and counts; its gold gradient is not recomputed. The actual audit caches each needed reaction gradient once and reuses it across all K/replicate combinations. Cache construction work must not be mistaken for a deployable estimator cost.

Lower-tail failures are retained individually in metrics, separately for every state and replicate. Estimated chemistry common descent is kept distinct from true full251 progress. A high cosine alone never determines selection.

State/input identities and singleton cache SHAs are recorded in the protocol and metrics. Original sampled-gradient reconstruction is checked at every state against its exact stored gradient with relativeL2<=1e−11. MainF32 state tensors, gradients and RNG are asserted unchanged. Chemistry derivatives use the existing realF64 shadow and objective, never F32 leaf-gradient storage.

Validation: PASS. Independent review: PASS. Full receipts are embedded in structured metrics.

Interpretation is local to this frozen panel of states and replicate draws. Finite-step or long-horizon training quality is not established. No initialization is preferred.

The95% threshold is an empirical success-frequency gate, not a95% statistical-confidence guarantee. Shared replicate draws deliberately pair states and K values; the192 state/draw measurements must not be treated as independent population replicates. Correct DB weights target the full251 scalar; finite-draw fidelity after unit normalization and nonlinear MOO aggregation is the operational test.

Next experiment: A separate read-only per-database chemistry-gradient variance decomposition/control-variate feasibility experiment on the frozen states; no K escalation or optimizer change. Not launched.

No training or retained model update occurred; UNIT_MAXMIN/controller/production behavior were unchanged. No new initialization, MOO method, adaptive allocation, control variate, K>8, 25/100-update continuation, full90, SCF, Diet or Slurm run was launched.
