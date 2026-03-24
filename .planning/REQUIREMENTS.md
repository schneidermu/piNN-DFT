# Requirements: piNN-DFT Evaluation Automation

**Defined:** 2026-03-24
**Core Value:** A newly trained checkpoint can be evaluated reproducibly with one command and one experiment folder, without manual file shuffling or piecing results together by hand.

## v1 Requirements

### Entry and Staging

- [ ] **EVAL-01**: Maintainer can start a full evaluation run by providing a checkpoint path and experiment name to one CLI entrypoint
- [ ] **EVAL-02**: Pipeline copies the tested checkpoint into the experiment directory before evaluation starts
- [ ] **EVAL-03**: Pipeline validates required input paths and records the resolved run configuration in the experiment workspace

### Experiment Workspace

- [ ] **EXPR-01**: Pipeline creates one dedicated experiment directory for each run
- [ ] **EXPR-02**: Pipeline stores per-branch logs and generated job artifacts inside that experiment directory

### WTMAD-2

- [ ] **WTMAD-01**: Maintainer can run WTMAD-2 evaluation without renaming or manually relocating the checkpoint
- [ ] **WTMAD-02**: Pipeline submits and tracks the required SLURM jobs for the WTMAD-2 branch
- [ ] **WTMAD-03**: Pipeline records WTMAD-2 metrics and branch outputs in the experiment summary

### avRANE

- [ ] **AVRA-01**: Maintainer can run avRANE evaluation from the same entrypoint used for WTMAD-2
- [ ] **AVRA-02**: Pipeline submits and tracks the required SLURM jobs for the avRANE branch
- [ ] **AVRA-03**: Pipeline records avRANE metrics and branch outputs in the experiment summary

### Reporting

- [ ] **REPT-01**: Pipeline writes a human-readable final report with branch metrics, failures or skips, and artifact locations
- [ ] **REPT-02**: Pipeline writes a machine-readable summary for programmatic inspection
- [ ] **REPT-03**: Pipeline completes with a partial final summary when one evaluation branch fails or is skipped

## v2 Requirements

### Extended Coverage

- **DENS-01**: Pipeline evaluates atomic density metrics such as MaxNE or related RMSD-style measures
- **DENS-02**: Pipeline supports additional evaluation branches beyond WTMAD-2 and avRANE

### Reliability Enhancements

- **RSLT-01**: Maintainer can resume an interrupted experiment without rerunning completed branches
- **RSLT-02**: Maintainer can retry only failed branches in an existing experiment

## Out of Scope

| Feature | Reason |
|---------|--------|
| Atomic density metrics in v1 | User explicitly narrowed current scope to WTMAD-2 and avRANE |
| Replacing SLURM with another scheduler | Existing workflows are already SLURM-based and scheduler replacement is not the immediate pain point |
| Full retry and resume framework in v1 | Initial version should optimize for the normal successful path first |
| UI or dashboard for experiment browsing | A file-based experiment folder is sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | Phase 1 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 1 | Pending |
| EXPR-01 | Phase 1 | Pending |
| WTMAD-01 | Phase 2 | Pending |
| WTMAD-02 | Phase 2 | Pending |
| WTMAD-03 | Phase 2 | Pending |
| AVRA-01 | Phase 3 | Pending |
| AVRA-02 | Phase 3 | Pending |
| AVRA-03 | Phase 3 | Pending |
| EXPR-02 | Phase 4 | Pending |
| REPT-01 | Phase 4 | Pending |
| REPT-02 | Phase 5 | Pending |
| REPT-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ?

---
*Requirements defined: 2026-03-24*
*Last updated: 2026-03-24 after initial definition*
