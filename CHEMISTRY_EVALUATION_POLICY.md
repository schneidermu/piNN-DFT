# Permanent chemistry population policy

Effective 2026-10-09, by explicit scientific decision: **never evaluate all eight
quadrature variants of a chemical identity**. This supersedes the exhaustive
endpoint plans in historical experiments; historical results remain unchanged.

Training uses a shuffled, no-replacement epoch of 251 relchem identities, one
independently drawn variant per identity. Evaluation uses 251 relchem and 17 AE17
identities, exactly one independently preselected fixed variant per identity,
reused at every checkpoint. Other stored variants are augmentation, not extra
samples. No diagnostic, endpoint, validation or final qualification may sweep
all variants. Storage integrity checks may inspect arrays without evaluating
scientific objectives.

The qualified singleton loss and database factors remain unchanged. At fixed
parameters, averaging a randomly reshuffled epoch has the same expectation as
uniform identity/variant sampling. During training parameters evolve: random
reshuffling is not an IID conditional-unbiasedness claim at each update.
The fixed-variant evaluation mean is a declared longitudinal objective, not an
exhaustively computed augmentation expectation or a database RMSE.

Enforcement: `lap_chemistry_sampling.validate_evaluation` rejects duplicate
identities (even with different variants); the endpoint evaluator requires the
frozen manifest and preserves historical exhaustive receipts separately.
