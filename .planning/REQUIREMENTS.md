# Requirements: qGAN Code Review Remediation

**Defined:** 2026-02-26
**Core Value:** The qGAN must produce correct, reproducible results with metrics that accurately reflect output quality

## v1 Requirements

Requirements for this remediation. Each maps to roadmap phases.

### Correctness Bugs

- [x] **BUG-01**: Checkpoint saves and loads `model.critic` (not `model.discriminator`)
- [x] **BUG-02**: Generator output scaling (`*0.1`) applied consistently across training, evaluation, and standalone generation
- [x] **BUG-03**: Denormalization strategy unified between training-time evaluation and standalone generation
- [x] **BUG-04**: Loss values stored as Python floats (`.item()`) not tensors retaining computation graphs
- [x] **BUG-05**: Epoch condition uses `self.num_epochs` instead of hardcoded `3000`
- [x] **BUG-06**: `delta` variable scoped inside class as `self.delta` (no global dependency)
- [x] **BUG-07**: `exit()` call removed from notebook cells

### Performance

- [x] **PERF-01**: Quantum circuit uses `diff_method='backprop'` on `default.qubit` simulator
- [x] **PERF-02**: All evaluation/inference forward passes wrapped in `torch.no_grad()`
- [x] **PERF-03**: DataLoader used with proper batch sampling (not flattened to list)
- [x] **PERF-04**: Evaluation metrics computed every N epochs (not every epoch)
- [x] **PERF-05**: Parameter broadcasting used for batch quantum circuit execution where possible

### ML Theory (WGAN-GP)

- [x] **WGAN-01**: `N_CRITIC = 5` (restored from 1)
- [x] **WGAN-02**: `LAMBDA = 10` (restored from 0.8)
- [x] **WGAN-03**: Dropout removed from critic network
- [x] **WGAN-04**: EMD computed on raw samples via `wasserstein_distance(real, fake)` (not histograms)
- [x] **WGAN-05**: Hardcoded histogram bins removed; bins derived from data range where histograms are still used for visualization
- [ ] **WGAN-06**: Early stopping monitors EMD (not critic loss)
- [x] **WGAN-07**: Learning rate ratio corrected (critic LR >= generator LR)
- [x] **WGAN-08**: Stylized facts implementations audited for correctness

### Quantum Circuit Design

- [x] **QC-01**: Redundant IQP RZ gate removed (before noise encoding)
- [x] **QC-02**: Data re-uploading added — noise re-encoded between variational layers
- [x] **QC-03**: PauliX measurements added alongside PauliZ for richer output
- [x] **QC-04**: Noise range expanded from `[0, 2pi]` to `[0, 4pi]`
- [x] **QC-05**: `WINDOW_LENGTH = 2 * NUM_QUBITS` computed automatically (not independently set)

### Code Quality

- [x] **QUAL-01**: Data path changed to relative `./data.csv`
- [x] **QUAL-02**: Duplicate imports removed (`numpy`, `random`)
- [ ] **QUAL-03**: Dead code removed: unused `compute_gradient_penalty` method, Cell 50 `d`, Cell 49 data perturbation hack
- [x] **QUAL-04**: `eval()` replaced with `globals().get()` or explicit logic
- [x] **QUAL-05**: `torch.load` uses `weights_only=True`
- [x] **QUAL-06**: `normalize()` returns `(normalized_data, mu, sigma)` tuple
- [x] **QUAL-07**: Hyperparameter naming consistent (all UPPER_CASE: `N_CRITIC`, `LAMBDA`, etc.)
- [ ] **QUAL-08**: Duplicate plotting cells consolidated
- [x] **QUAL-09**: Unused `self.measurements` removed from `__init__`
- [x] **QUAL-10**: Variable `data` not silently overwritten (use distinct names)

## v2 Requirements

Deferred to future work. Not in current roadmap.

### Infrastructure
- **INFRA-01**: Train/validation/test split for overfitting assessment
- **INFRA-02**: Learning rate scheduling (cosine annealing or step decay)
- **INFRA-03**: Hyperparameter optimization framework (Optuna)
- **INFRA-04**: Statistical significance testing / confidence intervals on metrics

### Architecture
- **ARCH-01**: Module migration — extract core classes to .py files
- **ARCH-02**: nbstripout / CI/CD pipeline for notebook quality
- **ARCH-03**: Comprehensive seed management with `set_seed()` utility

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Qutrit circuit architectures | Separate experimental notebooks; not in remediation scope |
| Full CSV schema validation | Research tool, not production service |
| Checkpoint compression / delta checkpointing | Operational concern, not correctness |
| Module migration to .py | User chose in-place notebook edits |
| Multiple training runs for statistical significance | Too expensive for single remediation pass |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUG-01 | Phase 1 | Complete |
| BUG-02 | Phase 2 | Complete |
| BUG-03 | Phase 2 | Complete |
| BUG-04 | Phase 1 | Complete |
| BUG-05 | Phase 1 | Complete |
| BUG-06 | Phase 1 | Complete |
| BUG-07 | Phase 1 | Complete |
| PERF-01 | Phase 2 | Complete |
| PERF-02 | Phase 1 | Complete |
| PERF-03 | Phase 1 | Complete |
| PERF-04 | Phase 2 | Complete |
| PERF-05 | Phase 2 | Complete |
| WGAN-01 | Phase 2 | Complete |
| WGAN-02 | Phase 2 | Complete |
| WGAN-03 | Phase 2 | Complete |
| WGAN-04 | Phase 2 | Complete |
| WGAN-05 | Phase 2 | Complete |
| WGAN-06 | Phase 2 | Pending |
| WGAN-07 | Phase 2 | Complete |
| WGAN-08 | Phase 2 | Complete |
| QC-01 | Phase 2 | Complete |
| QC-02 | Phase 2 | Complete |
| QC-03 | Phase 2 | Complete |
| QC-04 | Phase 2 | Complete |
| QC-05 | Phase 2 | Complete |
| QUAL-01 | Phase 1 | Complete |
| QUAL-02 | Phase 1 | Complete |
| QUAL-03 | Phase 3 | Pending |
| QUAL-04 | Phase 1 | Complete |
| QUAL-05 | Phase 1 | Complete |
| QUAL-06 | Phase 2 | Complete |
| QUAL-07 | Phase 1 | Complete |
| QUAL-08 | Phase 3 | Pending |
| QUAL-09 | Phase 1 | Complete |
| QUAL-10 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after roadmap creation*
