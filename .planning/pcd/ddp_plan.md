# T7 prospective two-rank CPU proof

**Status:** T7 engineering execution completed at representative `tau=0.02`; this does not select a scientific candidate. The immutable world-size-2 manifest, actual two-rank CLI resume-parity proof, and PCD cursor-25 SCF smoke are recorded in `C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\t7_engineering_tau002_ws2\t7_execution_receipt.json`. The prospective instructions below describe the realized workflow; a later scientific candidate requires a separate manifest and run. Wait for T6 to choose the candidate `tau` and the
predeclared common PCD learning-rate rule. Do not create a manifest, run
`torchrun`, or start training before that choice. The existing world-size-1
manifest remains historical: embedded identity
`550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`, while
its file-byte SHA-256 is `666492800bd307efef1a3ea5f9012d3f32218c9fbcfe7ed7ab768635d28112df`.
Neither value is reusable for T7.

## Inputs and manifest

Use the already verified inventory at
`C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\input_inventory.json`:

- predopt checkpoint `ed4ba823…d1b4b63f8`;
- Minnesota store `ec254952…d8c385d2a`;
- central operator corpus `7005cd86…6609b887`;
- 15-system AO cache `60631c23…f532a00a`;
- 27-reaction/15-system panel `f98c1123…ce9c66131`;
- reaction and mRKS dispersion files with hashes in the inventory.

After T6 selection, create a new, candidate-specific external directory under
`C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\`. Reuse
`MinnesotaGroupStore`, `CentralAOCache`, `_catalog_and_systems_for_panel`,
`build_sampling_manifest_from_catalog`, and `write_sampling_manifest` from the
existing CLI/protocol. Use the frozen panel catalog and systems, `updates=150`,
`seed=41`, `world_size=2`, and the runtime-derived source hashes (including the
panel). Refuse to overwrite an existing manifest. Validate exactly 150 entries,
27 reactions, 15 systems, two correctly labeled rank entries per update, and
distinct rank seed/stream sequences. Record both the embedded canonical digest
and the file-byte digest after writing; the former binds manifest content, the
latter is what PCD protocol metadata records. The CLI logs each update’s
`rank_samples`, which must equal the corresponding manifest entry.

The post-selection generation snippet should load the panel and hash inputs
from their verified paths, open `MinnesotaGroupStore` and `CentralAOCache` with
CPU/float32/4096, assert the 268-group/27-reaction/15-system panel, call
`_catalog_and_systems_for_panel`, then build and write with
`build_sampling_manifest_from_catalog(catalog, systems_names, updates=150,
seed=41, world_size=2, source_hashes=runtime_source_hashes)`. Guard the new path
before writing, read it back with `read_sampling_manifest`, and print both
hashes plus the two full rank identity sequences. This is a manifest-only
preparation process; the existing `_load_sampling_manifest` in the training
CLI will independently rebuild the expected content and reject any mismatch.

## CPU proof

Run from `/mnt/c/Dev/readWFN_share_ms/lap_full_vxc` using WSL Ubuntu 22.04 and
the tested environment `/home/schneidermu/.cache/pinn-lap-tests` (Python
3.10.12, PyTorch `2.14.1+cpu`, Gloo enabled; `torchrun` is in that environment’s
`bin/`). WSL cannot follow this worktree’s
Windows-form `.git` pointer directly. Export the verified worktree identities
before invoking the CLI so its run-level Git receipt works:

```bash
export GIT_DIR=/mnt/c/Dev/readWFN_share_ms/readWFN_share_ms/piNN-DFT/.git/worktrees/lap_full_vxc
export GIT_WORK_TREE=/mnt/c/Dev/readWFN_share_ms/lap_full_vxc
cd "$GIT_WORK_TREE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

This maps to revision `0d422fab03ec3551d860c5aa73b038b7103bd2dd`; Windows and
WSL SHA-256 checks matched byte-for-byte for all five protocol-bound source
files. Use explicit `/mnt/c/...` paths for the inventory inputs and new external
manifest/output directories. Supply a selected-candidate hyperparameter JSON
outside the repository (`tau` from T6, `beta=0.999`, `eps=1e-8`,
`qp_tolerance=1e-9`) and the T5-declared common `--learning-rate`.

After selection, set `RUNS`, `MANIFEST`, `HP_JSON`, and numeric `PCD_LR` to
that new candidate’s external paths/value, then use this shared command array:

```bash
COMMON=(
  --predopt-checkpoint /mnt/c/Dev/readWFN_share_ms/lap_operator_runs_20261001/predopt_fgpu_20261001T192623/lap_pbe_predopt.pt
  --minnesota-store-manifest /mnt/c/Dev/readWFN_share_ms/lap_moo_runs_20261001/mn_group_store_268/manifest.json
  --central-data-dir /mnt/c/Dev/readWFN_share_ms/lap_operator_runs_20261001/all90
  --ao-cache-dir /mnt/c/Dev/readWFN_share_ms/lap_moo_runs_20261001/mrks_15system_ao_cache
  --panel-definition /mnt/c/Dev/readWFN_share_ms/lap_moo_runs_20261001/panel_definition.json
  --sampling-manifest "$MANIFEST" --method pcd --method-hyperparameters "$HP_JSON"
  --updates 150 --seed 41 --learning-rate "$PCD_LR" --weight-decay 0.01
  --optimizer-family radamw --dtype float32 --device cpu
  --point-chunk-size 256 --ao-cache-chunk-size 4096 --checkpoint-every 1
  --reaction-dispersions /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/train_models/dispersions/dispersions.pickle
  --mrks-dispersions /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/train_models/dispersions/dispersions_mrks.pickle
)
TORCHRUN=/home/schneidermu/.cache/pinn-lap-tests/bin/torchrun
"$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=2 train_models/train_lap_moo.py \
  "${COMMON[@]}" --output-dir "$RUNS/resumed" --stop-after 1
"$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=2 train_models/train_lap_moo.py \
  "${COMMON[@]}" --output-dir "$RUNS/resumed" --resume "$RUNS/resumed/latest.pt" --stop-after 2
"$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=2 train_models/train_lap_moo.py \
  "${COMMON[@]}" --output-dir "$RUNS/uninterrupted" --stop-after 2
```

Use the unchanged CLI runs with `--method pcd --updates 150 --seed 41
--dtype float32 --device cpu --optimizer-family radamw --weight-decay 0.01
--point-chunk-size 256 --ao-cache-chunk-size 4096`. First run a fresh segmented
output to `--stop-after 1` and save its cursor-1 checkpoint; resume that same
output directory from `latest.pt` to `--stop-after 2`. Then run a second fresh
output uninterrupted to cursor 2. Both runs consume the first two identities of
the exact 150-update world-size-2 manifest; there is no subsampling, changed
objective, or shortened manifest. Compare final checkpoint protocol, cursor,
model, optimizer, scheduler, PCD EMA (`v`,`t`), and both saved rank RNG states;
compare update identities, losses, PCD diagnostics, and EMA rows while ignoring
timings. The existing Gloo regression directly checks distinct local synthetic
task gradients, identical globally averaged PCD inputs/direction/state, and
exact two-rank resume. Label that test as synthetic-objective engine proof; the
CLI run is the real-predopt/model/objective proof for the actual manifest.

## Practicality and limits

No CPU training time has been measured. The host exposes 12 WSL CPUs and about
14 GiB available memory; the largest cached AO record is about 575 MB raw, loaded
one system per rank, so two ranks appear memory-practical with 4096-row chunks.
Run the CPU proof without overlapping a candidate SCF or another CPU-heavy job.
Prior 7–16 second per-update observations are GPU timings and should not be
presented as CPU estimates. Budget roughly 5–20 minutes for four total update
executions (one segmented update, its resumed update, and a two-update baseline),
including process startup and checkpoint comparison; stop and report measured
timing if substantially slower rather than altering chunks/objectives. This is
a cursor-2 execution proof, not a 150-update performance or chemistry result.
