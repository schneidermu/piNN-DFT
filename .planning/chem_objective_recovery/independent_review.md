# Independent recovery review

**Verdict:** implementation and numerical gates reviewed; exactly two accepted updates from canonical predopt and the matched 27-row panel diagnostic pass. No blocker remains for this bounded recovery stage. This does not authorize extended training.

## Evidence

- Production patch is minimal: scalar `Database` labels become singleton lists at `lap_training.reaction_loss`; both `batch_fchem` helpers reject direct scalar strings. Existing list reduction/scaling remains unchanged. Regression coverage exercises scalar/list factors and gradients plus valid predopt list behavior.
- Validation receipt: Windows 218 passed/3 skipped; WSL 139 passed/2 skipped; focused 45 passed. Ruff remains 112 inherited findings, unchanged from HEAD; it is not clean. `compileall` and diff checks passed.
- Corrected frozen scalar replay gives factor `2.896652477664184` (legacy first-character factor `8.979622680758972`) and corrected F64 delta `-7.429179997853197e-7`. The F32 forward/backward mismatch remains documented as finite-precision sensitivity. Driver-added gradient-ratio thresholds remain labeled as such; their historical failed result is preserved.
- Clean predopt raw gradients repeat exactly. Chemistry uses persistent F64 leaves; E/op remain F32 and are widened only at aggregation. The NCCE31/BH saved-gradient PCD geometry passes strict common descent. The two actual replay samples independently pass per-step geometry and realized-displacement Armijo.
- `tiny_clean_restart_results.json` records the original manifest order: pTC13/BeH2 reaction 12, then AE17/H2 reaction 16. First update accepted at `alpha=6.632573669086685e-7` with zero backtracks; second at `alpha=3.3162868345433423e-7` after one backtrack. Final PCD EMA step is `t=2`.
- I recomputed all nine task/trial Armijo margins as `L_base + 1e-4*(g_task·s_realized) - L_trial`. Every recorded margin and task-pass flag matches, including the rejected initial AE17 trial. Both accepted stored displacements are nonzero and have strictly negative task slopes and positive componentwise margins for chem, exc, and op. The three model/shadow/caller rollback flags are true.
- Receipt/result/checkpoint hashes agree. All 102 result source files and 11 inputs match recorded start/end/current hashes. The three imported helper hashes match the previously reviewed clean-gradient receipt and during/post-run hashes; six restart data files match pinned-manifest, during-run, and post-run hashes, with loader-time verification. These helper/data comparisons are supplemental preexisting/during/post evidence, not claimed as driver start snapshots.
- Independent panel checks matched all 27 row identities and recomputed every cursor-2 ratio from its baseline row. Chemistry: median `1.01414`, 10/27 wins, p90 `2.31736`, max `32.1977`. Exc: median `0.49918`, 27/27 wins. Op: median `0.95764`, 27/27 wins. The chemistry panel is mixed and includes a large outlier; this bounded result does not establish broad chemistry recovery.
- Historical scope remains as recorded in `validity.json`: scalar-route absolute chemistry losses and gradients were affected; same-row ratios cancel the shared positive factor; old training trajectories remain contaminated. Canonical constants-MSE predopt is unaffected.

## Reviewed artifacts

- Repository: `C:\Dev\readWFN_share_ms\lap_full_vxc`, base `1b740325d8bc120d7da9ef1da117338e6ba98fcc`.
- External run: `C:\Dev\readWFN_share_ms\lap_chem_objective_recovery_runs_20261003\tiny_clean_restart_results.json` and `tiny_clean_restart_verified.json` (`status=pass`).
- External driver SHA-256: `82dcbfcb6632fc5de81592b747293cc52aa350356ee5e0cdd2c59793a7ea98d9`; result SHA-256: `c7112b76d4e6168d0d15b081b5c61025c0be154cefe0b5cf0b81de540885fa04`; checkpoint SHA-256: `14c1c5d50cf1e3f17e55d66d0be353d7d8871291718dbc10d3176746b2bbaf69`.

## Remaining scope

No full training or long-horizon trajectory was run. The two-update restart and 27-row panel are limited diagnostics; broader optimization still requires a separate gate. Torch reported 21.0 GB allocated and 27.7 GB reserved against roughly 16 GB physical VRAM; resident VRAM was not measured, so V100 memory fit remains unverified.

Chemistry objective semantics and precision are corrected; a tiny clean restart from canonical predopt is justified.
