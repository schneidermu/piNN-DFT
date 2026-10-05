# q-selective parameter-controllability audit

STOP at the frozen q-contrast linearity gate. Initialization controllability remains unresolved. No model parameter Jacobian or objective projection was computed; this is a failed-probe receipt, not a completed capacity audit.

Starting commit: `8f69b40f3e22f72d774bddf3158ec5e0469e0023`. Production source changes: zero.

## Frozen probe and states

Seeds 11, 23, 41; P67, P536, PBE_TANGENT and PBE_HEAD: 12 existing SHA-bound states. The deterministic 64-point probe spans 63 canonical groups, from the previous immutable sensitivity pool. Probe SHA256: `bcc3d01872d0181b197800e2a0295178224fe1d81e9ebfb2b8b8e2f64437b81a`. Eight fixed validation rows are shared by all states. Parameter ordering is sorted unique trainable names, 9446 coordinates; tied exchange parameters appear once.

The actual unbounded spin q convention is `q_sigma=lapl_sigma/[4*(3*pi^2)^(2/3)*(rho_sigma+EPS_RHO)^(5/3)]; unbounded correlation convention; exchange uses exact production spin scaling in model.forward`. Perturbations change raw Laplacian at fixed rho/sigma. Both spin windows remain within source 5thвЂ“95th percentile support. Computation uses differentiable F64 parameters and exactly widened F32 source values; no native source precision is restored.

## Failed predeclared linearity gate

Initial delta_q: `[0.1857518875278269, 0.3402330055448515]`. The prescribed global halving selects `[0.09287594376391345, 0.17011650277242574]`. Selected versus half-selected contrasts must agree within 5% in relative L2 on every state/spin validation block. They do not:

| Seed | State | Alpha relative error | Beta relative error |
|---|---|---:|---:|
| 11 | P67 | 0.184607292 | 0.205745505 |
| 11 | P536 | 0.239523872 | 1.00208466 |
| 11 | PBE_TANGENT | 0.0471134449 | 0.0647481219 |
| 11 | PBE_HEAD | 0 | 0 |
| 23 | P67 | 0.0355876334 | 0.0707862478 |
| 23 | P536 | 0.0356495338 | 0.14572114 |
| 23 | PBE_TANGENT | 0.0104039184 | 0.0245426033 |
| 23 | PBE_HEAD | 0 | 0 |
| 41 | P67 | 0.0416893201 | 0.0813286732 |
| 41 | P536 | 0.0772038668 | 0.0719915113 |
| 41 | PBE_TANGENT | 0.0321095342 | 0.00306640346 |
| 41 | PBE_HEAD | 0 | 0 |

The worst error is 1.0020846600660325 (100.208%), seed11 P536 beta. Reusing already-stored autograd descriptor derivatives at identical points independently finds approximately 112.24% error for this selected beta contrast. This does not establish a capacity defect, an optimizer defect, or a specific activation cause. Source-support membership alone does not guarantee local linearity.

The protocol explicitly stops before Jacobians when the globally halved window still fails. No smaller unplanned window was evaluated. The later user request authorized only the fixed dyadic-window localization documented separately; this failed-probe receipt and its STOP remain preserved.

## What remains unmeasured

Jq/J0 norms, singular spectra, effective rank, hidden/head controllability, P67/P536 subspace overlap, scientific-gradient projections and induced q responses were not computed. A/B/C/D interpretation cannot be assigned. Existing stored objective gradients were not recomputed. PBE_HEAD gives exactly zero current contrasts in this probe, but that does not establish zero output-head parameter controllability.

## Validation and provenance

All 34 recorded state/source/probe hash checks pass. Three toy tests validate the exact batched first-parameter Jacobian, spectrum metrics and block classifier. External compileall passes. State files and production sources remain unchanged. Large probe and trace arrays remain external, hash-bound in the results JSON. Independent failed-probe review subsequently passed. Its receipt is preserved in the dyadic-window audit; no parameter-controllability conclusion was reviewed or claimed.

## Single next diagnostic

Prospectively freeze one smaller global q window and pass the existing 5% local-linearity gate before building Jq. Preserve this failed probe; never choose a separate window by state. No initialization or optimizer recommendation is supported yet.

No training trajectory was run, no production optimizer was changed, no old cursor10 state was advanced, no expensive objective gradients were recomputed when stored gradients were available, and no full90/SCF/Diet/Slurm/100-update run was launched.
