# Requirements: qGAN Code Review Remediation

**Defined:** 2026-02-26
**Core Value:** The qGAN must produce correct, reproducible results with metrics that accurately reflect output quality

## v1 Requirements

Requirements for this remediation. Each maps to roadmap phases.

### Correctness Bugs

- [ ] **BUG-01**: Checkpoint saves and loads `model.critic` (not `model.discriminator`)
- [ ] **BUG-02**: Generator output scaling (`*0.1`) applied consistently across training, evaluation, and standalone generation
- [ ] **BUG-03**: Denormalization strategy unified between training-time evaluation and standalone generation
- [ ] **BUG-04**: Loss values stored as Python floats (`.item()`) not tensors retaining computation graphs
- [ ] **BUG-05**: Epoch condition uses `self.num_epochs` instead of hardcoded `3000`
- [ ] **BUG-06**: `delta` variable scoped inside class as `self.delta` (no global dependency)
- [ ] **BUG-07**: `exit()` call removed from notebook cells

### Performance

- [ ] **PERF-01**: Quantum circuit uses `diff_method='backprop'` on `default.qubit` simulator
- [ ] **PERF-02**: All evaluation/inference forward passes wrapped in `torch.no_grad()`
- [ ] **PERF-03**: DataLoader used with proper batch sampling (not flattened to list)
- [ ] **PERF-04**: Evaluation metrics computed every N epochs (not every epoch)
- [ ] **PERF-05**: Parameter broadcasting used for batch quantum circuit execution where possible

### ML Theory (WGAN-GP)

- [ ] **WGAN-01**: `N_CRITIC = 5` (restored from 1)
- [ ] **WGAN-02**: `LAMBDA = 10` (restored from 0.8)
- [ ] **WGAN-03**: Dropout removed from critic network
- [ ] **WGAN-04**: EMD computed on raw samples via `wasserstein_distance(real, fake)` (not histograms)
- [ ] **WGAN-05**: Hardcoded histogram bins removed; bins derived from data range where histograms are still used for visualization
- [ ] **WGAN-06**: Early stopping monitors EMD (not critic loss)
- [ ] **WGAN-07**: Learning rate ratio corrected (critic LR >= generator LR)
- [ ] **WGAN-08**: Stylized facts implementations audited for correctness

### Quantum Circuit Design

- [ ] **QC-01**: Redundant IQP RZ gate removed (before noise encoding)
- [ ] **QC-02**: Data re-uploading added — noise re-encoded between variational layers
- [ ] **QC-03**: PauliX measurements added alongside PauliZ for richer output
- [ ] **QC-04**: Noise range expanded from `[0, 2pi]` to `[0, 4pi]`
- [ ] **QC-05**: `WINDOW_LENGTH = 2 * NUM_QUBITS` computed automatically (not independently set)

### Code Quality

- [ ] **QUAL-01**: Data path changed to relative `./data.csv`
- [ ] **QUAL-02**: Duplicate imports removed (`numpy`, `random`)
- [ ] **QUAL-03**: Dead code removed: unused `compute_gradient_penalty` method, Cell 50 `d`, Cell 49 data perturbation hack
- [ ] **QUAL-04**: `eval()` replaced with `globals().get()` or explicit logic
- [ ] **QUAL-05**: `torch.load` uses `weights_only=True`
- [ ] **QUAL-06**: `normalize()` returns `(normalized_data, mu, sigma)` tuple
- [ ] **QUAL-07**: Hyperparameter naming consistent (all UPPER_CASE: `N_CRITIC`, `LAMBDA`, etc.)
- [ ] **QUAL-08**: Duplicate plotting cells consolidated
- [ ] **QUAL-09**: Unused `self.measurements` removed from `__init__`
- [ ] **QUAL-10**: Variable `data` not silently overwritten (use distinct names)

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
| BUG-01 | — | Pending |
| BUG-02 | — | Pending |
| BUG-03 | — | Pending |
| BUG-04 | — | Pending |
| BUG-05 | — | Pending |
| BUG-06 | — | Pending |
| BUG-07 | — | Pending |
| PERF-01 | — | Pending |
| PERF-02 | — | Pending |
| PERF-03 | — | Pending |
| PERF-04 | — | Pending |
| PERF-05 | — | Pending |
| WGAN-01 | — | Pending |
| WGAN-02 | — | Pending |
| WGAN-03 | — | Pending |
| WGAN-04 | — | Pending |
| WGAN-05 | — | Pending |
| WGAN-06 | — | Pending |
| WGAN-07 | — | Pending |
| WGAN-08 | — | Pending |
| QC-01 | — | Pending |
| QC-02 | — | Pending |
| QC-03 | — | Pending |
| QC-04 | — | Pending |
| QC-05 | — | Pending |
| QUAL-01 | — | Pending |
| QUAL-02 | — | Pending |
| QUAL-03 | — | Pending |
| QUAL-04 | — | Pending |
| QUAL-05 | — | Pending |
| QUAL-06 | — | Pending |
| QUAL-07 | — | Pending |
| QUAL-08 | — | Pending |
| QUAL-09 | — | Pending |
| QUAL-10 | — | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 0
- Unmapped: 35

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after initial definition*
