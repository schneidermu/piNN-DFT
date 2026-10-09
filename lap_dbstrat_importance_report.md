# Database-stratified importance sampling versus IID AdamW

Decision: **NO-GO**.

Arm B ran 90 new AdamW updates from corrected P536. Arm A was the preserved IID LR=1e-4 trajectory and was not retrained. The relchem draw is uniform over eight databases, then uniform over identities in the drawn database. The qualified singleton relchem gradient is multiplied once by `8*n_d/251`. AE17, Exc, operator, coefficients, and AdamW settings are unchanged.

## Sampling integrity

Database counts in the 90 frozen stratified updates, against the uniform-database expectation 11.25:

| Database | Identities | Sampled | Expected | Importance weight |
|---|---:|---:|---:|---:|
| ABDE4 | 4 | 8 | 11.25 | 0.127490 |
| DBH76 | 70 | 6 | 11.25 | 2.231076 |
| EA13 | 11 | 18 | 11.25 | 0.350598 |
| IP13 | 13 | 9 | 11.25 | 0.414343 |
| MGAE109 | 104 | 11 | 11.25 | 3.314741 |
| NCCE31 | 28 | 14 | 11.25 | 0.892430 |
| PA8 | 8 | 9 | 11.25 | 0.254980 |
| pTC13 | 13 | 15 | 11.25 | 0.414343 |

Importance weights used on the trajectory span 0.127490040 to 3.314741036.
Median raw relchem norm 166.995015840, median corrected relchem norm 66.751843975.
Median joint norm before correction 2.969818284, after correction 1.211459595.
Median native AdamW step norm 0.002479530; cursor-aligned IID control median 0.002501140.
These cursor-aligned norms are not paired samples. The correction changes gradient scale; AdamW moments do not preserve that scale as a proportional step.

## Clean28

Clean28 is the mean Diet-weighted absolute error of the 28 leakage-clean reactions, in kcal/mol. It is not full30 WTMAD-2. Full30 was not used for selection.

| Cursor | Old IID Clean28 | New DB-strat Clean28 | Difference |
|---|---:|---:|---:|
| 0 | 9.553190636 | 9.553190636 | 0.000000000 |
| 20 | Not yet established | 9.731327901 | — |
| 59 | 8.868473327 | 9.121993241 | 0.253519914 |
| 70 | 8.629660230 | 8.909938464 | 0.280278234 |
| 80 | 8.649173724 | 9.013481907 | 0.364308182 |
| 90 | 9.273578777 | 9.324003579 | 0.050424802 |

A negative difference favors the new sampler. The historical IID t20 value is not established and is not imputed.

At t70, 15 of 28 clean reactions have a lower score contribution than IID at the same cursor.
Largest contribution decreases versus that IID checkpoint:
- ACONF-10: -0.028727480
- S66-50: -0.025185331
- G21EA-14: -0.013014943
- WCPT18-15: -0.010229743
- MB16-43-10: -0.010014485
Largest contribution increases:
- HAL59-57: +0.036925008
- PX13-9: +0.044361064
- HAL59-40: +0.046497885
- BHPERI-11: +0.061718112
- HEAVY28-16: +0.066120052

## Scientific objectives

| Checkpoint | relchem/t0 | AE17/t0 | Exc/t0 | operator/t0 | Eligible |
|---|---:|---:|---:|---:|---|
| t70 | 1.031209044 | 0.315539432 | 0.322297455 | 0.919694809 | no |
| t90 | 1.018877965 | 0.194544149 | 0.108227088 | 0.920716660 | no |

Reference t0 objectives remain relchem 1.2354710978866068, AE17 24.901047104016875, Exc 92.17480502000551, operator 0.03314821681605566.
Eligibility requires every finite ratio to be strictly below 1. A lower Clean28 without eligibility is not a promoted functional.

## Runtime

Logged synchronized update time: 2940.271s over 90 new updates. Peak live CUDA 13.150 GiB; peak reserved 23.107 GiB.

## Interpretation

- One stochastic trajectory is not proof of a superior sampler.
- Cursor-aligned IID and DB-stratified samples are different reactions.
- An unbiased singleton correction does not imply the same AdamW trajectory.
- The fixed-panel relchem endpoint is not automatically the singleton-gradient expectation.
- No repeated-sample variance estimate was measured.

Recommended next experiment, not executed: No further database-stratified training. One IID AdamW seed from the same P536, 90 updates at LR 1e-4, to test whether the late Clean28 rebound is stream-specific.

Code revision at execution: `745d0b2abac4f78078048c5a1e0e4e6c02124671`.
Initial tensor SHA256: `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`.
Dataset logical SHA256: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`.
New manifest SHA256: `5d540c6bef03b38bb6cc13c209cb23d2358afc4f90d42fcf4001edba8a285755`.
Historical calibration SHA256: `4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096`.

- t0 checkpoint SHA256: `4689f96a888ba7a3f5bd3170fa5d2ae65e08e7ea5eef35851d2ddd688e0363e1`
- t20 checkpoint SHA256: `6b1c156d1c1706b58fb80fa5f1fe40b79f7fa52e20c22f2782d7464a946a3e77`
- t59 checkpoint SHA256: `462f3100cb83f14e34b644e1622a6b7088e432251440faab771654ece8e474c6`
- t70 checkpoint SHA256: `cf489f6b00d19cc1af33fc5d13acf4c97d2313e99a60e169ca9c340320ccb77c`
- t80 checkpoint SHA256: `042b25de04ae901657a98b9e549060c9828f8c147d742ebbe91c847e55b30049`
- t90 checkpoint SHA256: `444629b00cd5efbd3da039af8770b09691396b9b88866bb342549ff36e8d0305`
