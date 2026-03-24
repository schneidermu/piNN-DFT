# Roadmap: piNN-DFT Evaluation Automation

## Overview

This roadmap turns the current manual evaluation flow into a reusable brownfield experiment pipeline. The work starts by defining stable experiment contracts and artifact layout, then reshapes the two target evaluation branches, and finally unifies orchestration, logging, and partial-failure reporting into one operator-friendly command.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Experiment Contract Foundation** - Define the CLI contract, experiment directory schema, and checkpoint staging model
- [ ] **Phase 2: WTMAD-2 Branch Refactor** - Reshape current WTMAD-2 scripts into a reusable branch pipeline
- [ ] **Phase 3: avRANE Branch Refactor** - Reshape current avRANE scripts into a reusable branch pipeline
- [ ] **Phase 4: Unified Logging and Reports** - Centralize branch logs, artifact collation, and human-readable reporting
- [ ] **Phase 5: Final Orchestration and Partial Failure Handling** - Deliver the one-command runner and machine-readable summaries

## Phase Details

### Phase 1: Experiment Contract Foundation
**Goal**: Establish one stable way to describe, stage, and persist an evaluation run
**Depends on**: Nothing (first phase)
**Requirements**: [EVAL-01, EVAL-02, EVAL-03, EXPR-01]
**Success Criteria** (what must be TRUE):
  1. Maintainer can point the new workflow at any checkpoint path and experiment name without relying on legacy naming conventions
  2. Every run creates a dedicated experiment directory with copied checkpoint and persisted run configuration
  3. Shared path and artifact conventions exist for downstream WTMAD-2 and avRANE branches
**Plans**: 3 plans

Plans:
- [ ] 01-01: Define experiment manifest, folder layout, and CLI argument contract
- [ ] 01-02: Implement checkpoint staging and shared path helpers
- [ ] 01-03: Adapt existing evaluation code to consume explicit experiment metadata

### Phase 2: WTMAD-2 Branch Refactor
**Goal**: Convert the current WTMAD-2 workflow into a reusable evaluation branch
**Depends on**: Phase 1
**Requirements**: [WTMAD-01, WTMAD-02, WTMAD-03]
**Success Criteria** (what must be TRUE):
  1. WTMAD-2 branch can be launched for a staged experiment without manual checkpoint relocation
  2. Required SLURM jobs are rendered, submitted, and tracked from the refactored branch
  3. WTMAD-2 metrics and source outputs are collected into experiment-owned artifacts
**Plans**: 3 plans

Plans:
- [ ] 02-01: Refactor legacy WTMAD-2 script entrypoints into explicit library and CLI boundaries
- [ ] 02-02: Implement branch-specific SLURM submission and completion tracking
- [ ] 02-03: Normalize metric extraction and experiment output collation for WTMAD-2

### Phase 3: avRANE Branch Refactor
**Goal**: Convert the current avRANE workflow into a reusable evaluation branch
**Depends on**: Phase 2
**Requirements**: [AVRA-01, AVRA-02, AVRA-03]
**Success Criteria** (what must be TRUE):
  1. avRANE branch can be launched from the same staged experiment contract as WTMAD-2
  2. Required SLURM jobs and downstream processing steps are tracked through explicit branch status
  3. avRANE metrics and source outputs are collected into experiment-owned artifacts
**Plans**: 3 plans

Plans:
- [ ] 03-01: Refactor avRANE-oriented scripts around explicit experiment inputs and outputs
- [ ] 03-02: Implement avRANE branch SLURM submission and waiting flow
- [ ] 03-03: Normalize avRANE metric extraction and experiment output collation

### Phase 4: Unified Logging and Reports
**Goal**: Make experiment outputs easy to inspect through one folder and one top-level report
**Depends on**: Phase 3
**Requirements**: [EXPR-02, REPT-01]
**Success Criteria** (what must be TRUE):
  1. Every branch writes logs and generated job artifacts into the experiment directory
  2. Human-readable report links each branch outcome to metrics, failures or skips, and artifact locations
  3. Operators can diagnose what happened in a run without hunting across repository folders
**Plans**: 2 plans

Plans:
- [ ] 04-01: Centralize branch log, artifact, and status writing under the experiment folder
- [ ] 04-02: Build a unified human-readable reporting layer for completed or partial runs

### Phase 5: Final Orchestration and Partial Failure Handling
**Goal**: Deliver the single-command runner with machine-readable outputs and resilient completion semantics
**Depends on**: Phase 4
**Requirements**: [REPT-02, REPT-03]
**Success Criteria** (what must be TRUE):
  1. One entry command can launch both evaluation branches and wait for them to finish
  2. Final machine-readable summary captures branch status, metrics, and artifact paths
  3. A failed or skipped branch still yields a completed experiment summary with successful branch outputs preserved
**Plans**: 2 plans

Plans:
- [ ] 05-01: Implement top-level orchestration, wait logic, and branch lifecycle coordination
- [ ] 05-02: Emit machine-readable summary and finalize partial-failure completion rules

## Progress

**Execution Order:**
Phases execute in numeric order: 1 > 2 > 3 > 4 > 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Experiment Contract Foundation | 0/3 | Not started | - |
| 2. WTMAD-2 Branch Refactor | 0/3 | Not started | - |
| 3. avRANE Branch Refactor | 0/3 | Not started | - |
| 4. Unified Logging and Reports | 0/2 | Not started | - |
| 5. Final Orchestration and Partial Failure Handling | 0/2 | Not started | - |
