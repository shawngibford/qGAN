# Roadmap: qGAN Code Review Remediation

## Overview

Three phases of targeted remediation on `qgan_pennylane.ipynb`, ordered by strict dependency tiers. Phase 1 establishes a clean, safe foundation with no training behavior impact. Phase 2 implements WGAN-GP correctness and quantum circuit redesign — the highest-risk changes that alter training dynamics, require a fresh training run, and invalidate all existing checkpoints. Phase 3 removes remaining dead code and debug artifacts after Phase 2 training is validated. Mixing phases is dangerous: normalize() must be updated atomically, WINDOW_LENGTH must be set before circuit redesign, and EMD must be fixed before early stopping is changed.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation and Correctness Infrastructure** - Safe infrastructure fixes with no training behavior impact
- [ ] **Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign** - High-impact changes that restore ML theory compliance and redesign the quantum circuit
- [ ] **Phase 3: Post-Processing Consistency and Cleanup** - Dead code removal and debug artifact cleanup after Phase 2 is validated

## Phase Details

### Phase 1: Foundation and Correctness Infrastructure
**Goal**: The notebook runs top-to-bottom without kernel corruption, checkpoints save/load correctly, and the codebase is free of unsafe or blocking code
**Depends on**: Nothing (first phase)
**Requirements**: BUG-01, BUG-04, BUG-05, BUG-06, BUG-07, PERF-02, PERF-03, QUAL-01, QUAL-02, QUAL-04, QUAL-05, QUAL-07, QUAL-09, QUAL-10
**Success Criteria** (what must be TRUE):
  1. The notebook can be run cell-by-cell from top to bottom without hitting `exit()`, a NameError from a global variable, or a kernel crash
  2. A checkpoint saved and then loaded restores the generator parameters with gradients intact (not a plain tensor that silently disables gradient updates)
  3. All `torch.load` calls use `weights_only=True` and do not trigger a security warning
  4. All evaluation and inference forward passes are wrapped in `torch.no_grad()` so no gradients accumulate during non-training code paths
  5. The data file is loaded via `./data.csv` and the notebook runs correctly from any working directory
**Plans**: 3 plans
  - [ ] 01-01-PLAN.md — Checkpoint system rewrite + unsafe code removal (BUG-01, BUG-07, QUAL-04, QUAL-05, QUAL-09)
  - [ ] 01-02-PLAN.md — DataLoader restructuring + training loop bug fixes (BUG-04, BUG-05, BUG-06, PERF-02, PERF-03)
  - [ ] 01-03-PLAN.md — Naming conventions + notebook organization (QUAL-01, QUAL-02, QUAL-07, QUAL-10)

### Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign
**Goal**: The notebook implements WGAN-GP per Gulrajani et al. (2017) with a well-designed quantum circuit supporting universal approximation via data re-uploading, and a fresh training run produces measurably better distributional fidelity
**Depends on**: Phase 1
**Requirements**: BUG-02, BUG-03, PERF-01, PERF-04, PERF-05, WGAN-01, WGAN-02, WGAN-03, WGAN-04, WGAN-05, WGAN-06, WGAN-07, WGAN-08, QC-01, QC-02, QC-03, QC-04, QC-05, QUAL-06
**Success Criteria** (what must be TRUE):
  1. `WINDOW_LENGTH` is set as `2 * NUM_QUBITS` in the hyperparameter config cell and the quantum circuit output dimension matches the critic's Conv1D input dimension
  2. `N_CRITIC = 5` and `LAMBDA = 10` are set in the hyperparameter config, dropout is absent from the critic network, and the training loop runs the critic 5 times per generator step
  3. EMD is computed via `wasserstein_distance(real_samples, fake_samples)` on raw 1D arrays (not histogram bins), and the early stopping monitor is tracking this EMD value
  4. The quantum circuit uses `diff_method='backprop'` on an explicitly constructed `default.qubit` device, and generator parameter gradients are computed during a training step
  5. The generator output scaling and denormalization strategy are identical between the training loop evaluation path and the standalone generation path — running both produces numerically equivalent outputs for the same input noise
**Plans**: 4 plans
  - [ ] 02-01-PLAN.md — Config cell overhaul + normalize() signature (QUAL-06, QC-05, WGAN-01, WGAN-02, WGAN-07)
  - [ ] 02-02-PLAN.md — Quantum circuit redesign + critic architecture (QC-01, QC-02, QC-03, QC-04, PERF-01, WGAN-03)
  - [ ] 02-03-PLAN.md — Training loop rewrite + evaluation pipeline (BUG-02, BUG-03, PERF-04, PERF-05, WGAN-04, WGAN-05, WGAN-08)
  - [ ] 02-04-PLAN.md — Early stopping + standalone generation + post-training (WGAN-06, BUG-02, BUG-03)

### Phase 3: Post-Processing Consistency and Cleanup
**Goal**: The notebook contains no dead code, debug artifacts, or duplicate visualization cells; normalization constants are protected from variable shadowing; and all edge cases in visualization cells are handled
**Depends on**: Phase 2
**Requirements**: QUAL-03, QUAL-08
**Gap Closure:** Closes gaps from v1.0 milestone audit (2 requirements + 1 integration + 1 flow)
**Success Criteria** (what must be TRUE):
  1. The unused `compute_gradient_penalty` method, Cell 57 debug variable `d`, and Cell 49 data perturbation hack are absent from the notebook
  2. Duplicate plotting cells are consolidated so each visualization appears exactly once
  3. Cells 16/18 use distinct variable names (e.g. `mu_viz`/`sigma_viz`) that do not shadow the normalization constants `mu`/`sigma` from Cell 15
  4. Cell 36 loss visualization handles the edge case where `critic_loss_avg` has exactly 1 entry without raising a NameError
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation and Correctness Infrastructure | 0/3 | Not started | - |
| 2. WGAN-GP Correctness and Quantum Circuit Redesign | 0/4 | Not started | - |
| 3. Post-Processing Consistency and Cleanup | 0/TBD | Not started | - |
