# Publication dataset v1

Canonical storage is JSON/JSONL plus modality-separated HDF5 shards. Legacy
pickle/checkpoint inputs are trusted conversion sources only. They are never
canonical output. Existing experiment loaders remain unchanged.

```python
from train_models.publication_data import PublicationDataset, identity_collate
from torch.utils.data import DataLoader

bundle = PublicationDataset(dataset_root)
chemistry = bundle.chemistry("train_relchem")
record = chemistry.load_variant(("DBH76", 0), "level3_mura")
reference_record = chemistry.load_variant(("DBH76", 0), "level3_mura")
dispersions = bundle.chemistry_dispersions()  # preserves native NumPy F64 scalars
loader = DataLoader(chemistry, batch_size=1, num_workers=2,
                    collate_fn=identity_collate, pin_memory=True)
systems = bundle.mrks()
chunks = systems.ao_dataset(systems.ids[0], chunk_size=4096)
validation = bundle.validation("diet30_clean_validation")
bundle.close()
```

Select reaction identity before selecting augmentation. `len(chemistry)` counts
identities; eight grid variants do not multiply reaction/database weight.
Explicit `(identity, variant)` lookup gives paired consumers exactly the same
input. This package implements no sampling estimator or optimization algorithm.

HDF5 handles open lazily in the consuming process. Pickling removes handles;
PID changes close inherited handles. Dataset samples are CPU tensors only.
First shard access verifies its SHA256; subsequent access uses a process-local
verified handle. `bundle.verify()` rechecks every file. The variable-size collator
returns sample lists rather than padded molecular arrays. AO chunk datasets avoid
materializing entire operator samples in workers.

## Build and qualification

`tools/build_publication_dataset.py --help` lists explicit source arguments.
The builder verifies source manifests/checkpoints, preserves legacy dtype
boundaries, writes `.staging`, and refuses replacement of a published dataset.
`--resume-staging` reuses completed, SHA-bound conversion phases. It never treats
an interrupted incomplete phase as complete.

Run `tools/qualify_publication_dataset.py` with the same chemistry/central/AO
sources, staging root, and frozen model source. It checks every array content
hash, exact legacy chemistry inputs/energies/losses/gradients, all90 operator
inputs and fresh representative operator objective/gradient parity, fixed-density
parity receipts, and worker throughput. `--publish` renames staging only after
these checks pass. Contract tests, relevant existing tests, lint and compilation
must also pass before publication.

The qualified model source is a validation instrument, not canonical dataset
storage. HDF5 array hashes include dtype/shape/content and are independent of
compression. JSON canonicalization and deterministic IDs avoid filesystem/order
dependence. File hashes additionally bind exact shard bytes.

## Scientific boundaries

Chemistry preserves stored F32 values before matched-F64 objective evaluation.
Chemistry dispersion scalars retain their native zero-dimensional NumPy F64
type through the unchanged reaction objective. JSON values alone must not be
passed as Python floats, because that changes torch.tensor's default dtype.
Context-dependent legacy PBE local-energy rounding is preserved in separately
content-addressed records; descriptor arrays remain deduplicated. Numerical
dataset attributes state source/storage/production dtype, axes and atomic units.
The nine-column legacy chemistry grid includes tau; the current tau-free model
and mRKS ten-column descriptor path are preserved without reinterpretation.

Operator targets retain the physical source gauge, total-density derivative and
symmetric-overlap-orthonormalized squared Frobenius loss divided by nAO.
No constant subtraction or traceless projection is introduced.

Validation uses stored PBE0 F64 densities without SCF. Its precomputed non-XC
energy is `Tr(P hcore) + 0.5 Tr(P J[P]) + E_nuc`. Add model XC and the declared
primary PBE0-D3(BJ) correction. PBE-D3(BJ) is secondary only. Diagnostic Diet30
is nonselectable; clean validation excludes reserved and possible training
overlap. No future test split, arrays or labels are packaged.

Source licensing/redistribution permission remains a prerequisite for external
archival; a local reproducible scientific bundle does not confer those rights.
