# Native four-task qualification
Invariants: unchanged objectives/PCD/Armijo/samples; F32 main, F64 chemistry shadow; production repaired operator.
Interface: explicit --four-task-pcd, existing v2 immutable manifest required; three-task v1 unchanged.
State: v6 four-task metadata binds precision-critical production sources; v5 remains readable, cannot resume as v6.
Acceptance: manifest/mode fail closed; exact update0 parity; five-update qualification only after update0 passes; exact two-update resume; native two-rank Gloo fixture.
Non-goals: new trainer, estimator, method, schedule, precision changes, >5 trajectories, benchmarks/cluster.
