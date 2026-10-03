# Execution tickets

1. **Restore and replay cursor 2.** Verify checkpoint/gradient identities and tau-0.02 slopes; stop on any mismatch.
2. **Run the six-value frozen PCD geometry map.** Reuse identical gradients and scratch EMA; perform the independent tau-zero cone-projection check; select the smallest positive tau meeting all three cosine margins.
3. **Apply the existing Armijo gate to that one direction.** If it passes, evaluate the single candidate on the fixed panel. Do not commit training state or launch training.
4. **Conditional diagnostics:** if no positive PCD tau passes geometry, run the native Fliege–Svaiter diagnostic. If the selected PCD direction passes batch Armijo but panel transfer fails, accumulate gradients over one complete immutable 27-entry training-manifest cycle (training samples only), recompute unchanged canonical PCD, and test its direction geometry and vector Armijo on that cycle. Do not form gradients or select the step from panel rows; do not start long training.
