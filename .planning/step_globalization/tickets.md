# Tickets: finite-step globalization

## T1 — Freeze the historical replay

Use the existing external diagnostic inputs and frozen stream. Rebuild canonical predopt/seed 41, replay historical update 0 unchanged, and capture the full state immediately before update 1 (AE17 reaction 16 / H2): model parameters/buffers, RNG, stateless SGD configuration, sample cursor, and PCD EMA. The receipt lacks this snapshot; recreate it instead of inferring it.

**Done when:** the exact identities and baseline losses match the receipt and the pre-update state can be restored exactly.

## T2 — Run the same-batch Armijo response curve

Compute task gradients and PCD direction once at update 1; set `alpha0=6.632573669086685e-7` as the initial upper-bound proposal and `Delta0=-alpha0*d`. Require three finite negative raw-loss slopes. Score canonical `t=1` followed by `t=2^-j`, `j=1..20`, with `c=1e-4`, `rho=.5`, recording predicted/actual reduction, ratios, margins, finite values, and parameter movement. Restore complete state after each trial; do not step the optimizer or advance EMA/RNG/cursor/scheduler. Preserve the five-update fixed-SGD historical replay as the reference. Score two extra smaller fractions after first rescue if within cap.

**Gate:** t=1 reproduces chemistry/E_xc increase and operator decrease; a smaller representable candidate passes componentwise Armijo and strictly lowers all three same-batch losses. Otherwise stop and restore; no T3 integration. Only after pass may a separately labeled safeguarded continuation commit the accepted update once and replay updates 2–4.

## T3 — Add the minimal direct-step path

Only after T2 rescue, add one scalar line-search layer using the existing objective/gradient/PCD engine and commit `theta += -t*alpha0*d` directly after acceptance. `alpha0` stays the initial upper-bound proposal; accepted `t` adapts the effective step. This is PCD with adaptive vector Armijo and a direct parameter update, with no optimizer step, adaptive optimizer transform, momentum, decay, or scheduler. Preserve the no-search historical fixed-SGD run unchanged as its control. Do not advance PCD EMA, cursor, or normal RNG state during trials; commit each exactly once after acceptance.

**Done when:** the line path selects the same `t` as the T2 curve and commits one identical finite displacement `-t*alpha0*d`.

## T4 — Test state, resume, and DDP agreement

Cover zero/adverse slopes, nonfinite candidate, strict all-task decrease, componentwise Armijo, bitwise state restoration, and one-time commit. Persist search constants, accepted multiplier, and backtrack count; verify checkpoint/resume histories. Under DDP, equally average the existing one raw scalar per rank for each task (count one per rank), use globally averaged raw gradients for slopes, and synchronize proposal, `t`, and acceptance across ranks; do not weight by grid points or system size.

**Done when:** resumed and continuous traces match and ranks agree on losses, multiplier, and cursor.

## T5 — Stage fixed-panel evidence

Use the existing panel and reductions at updates 0, 5, 10, and 25. Run through 10 first; continue to 25 only if finite and the chemistry safety gate passes. Consider 100 only if chemistry, E_xc, and AO-operator panel aggregates all improve at update 25. Report chemistry first and include paired rows, median, p90, and maximum.

**Done when:** staged gates and paired panel evidence are recorded without changing the panel or objectives.

## T6 — Apply the single fallback gate

If T2 has no practical same-ray common-decrease step after slope, evaluation, and float32 checks, propose one multiobjective trust-region follow-up through a fresh Arena decision. Do not automatically implement/run it. If later results show panel-only gains or chronic tiny steps, request a fresh Arena comparison for balanced batches/variance reduction or BBDMO/nonmonotone/TR as appropriate; do not stack.

**Done when:** fallback trigger is distinguished from later panel/step-size triggers and only one next mechanism is selected.

## T7 — Close out with an audit

Record exact source/config/data identities, response curve, accepted multipliers, finite changes, rollback proof, commit counts, fixed-panel gates, and limitations. Keep the historical SGD control and any safeguarded continuation separately labeled.

**Done when:** another reviewer can reproduce the decision and no batch result is presented as population generalization.

## Final acceptance status

| Ticket | Status | Evidence / blocker |
|---|---|---|
| T1 | Accepted | Frozen historical inputs and first-step replay match. |
| T2 | Accepted known-step gate | Same-ray rescue at 1/32; linear-window probes 1/256 and 1/512. Later first-five coverage stopped at update 2. |
| T3 | Accepted | Direct Armijo path committed in e017768. |
| T4 | Accepted | Independent synthetic, rollback, resume, and actual two-rank Gloo tests pass. |
| T5 | Stopped at cursor 2 | PCD chemistry-ascent ray; no 5/10/25/100 gates or SCF shortlist. |
| T6 | Decision recorded, unrun | Four-solver judge selects native common-descent/Armijo diagnostic; one conditional full-cycle batching fallback. |
| T7 | Accepted | Independent release review, test receipts, and report/protocol/results verified. |
