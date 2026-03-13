# Requirements: qGAN Post-HPO Improvements

**Defined:** 2026-03-13
**Core Value:** The qGAN must generate synthetic OD time series that capture real data's volatility structure — not just the mean trend — with variance, kurtosis, and spectral characteristics that match the training distribution.

## v1.1 Requirements

Requirements for post-HPO improvement milestone. Each maps to roadmap phases.

### Regression Fixes

- [ ] **REG-01**: Training loop uses correct noise range [0, 4π] in all 3 locations (critic training, generator training, evaluation)
- [ ] **REG-02**: QNode uses `diff_method='backprop'` instead of `parameter-shift` (prerequisite for broadcasting)
- [ ] **REG-03**: Training loop uses batched/broadcasted QNode calls instead of per-sample Python loops (~12x speedup)
- [ ] **REG-04**: Evaluation generation uses real PAR_LIGHT values instead of `torch.zeros` for conditioning
- [ ] **REG-05**: mu/sigma variable shadowing eliminated in plotting cells

### Spectral Loss

- [ ] **SPEC-01**: Generator loss includes log-PSD MSE term via `torch.fft.rfft` (differentiable)
- [ ] **SPEC-02**: PSD loss weight (`lambda_psd`) is configurable as a hyperparameter
- [ ] **SPEC-03**: PSD loss computed on same batch of real/fake windows used for WGAN loss

### Conditioning

- [ ] **COND-01**: Intervention test cell generates samples at PAR_LIGHT=0 vs PAR_LIGHT=1 and reports KS test
- [ ] **COND-02**: Sweep test cell generates across PAR_LIGHT grid [0, 0.2, 0.4, 0.6, 0.8, 1.0] with summary statistics
- [ ] **COND-03**: Dropout rate is configurable as a hyperparameter (default matches current 0.2)

## v1.2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Capacity Tuning

- **CAP-01**: NUM_LAYERS configurable (support 6-8 layers with automatic parameter recomputation)
- **CAP-02**: Simpler critic architecture option (fewer conv layers, smaller kernels, ~5.5K params vs 250K)
- **CAP-03**: Barren plateau monitoring for deeper circuits (gradient variance tracking)

### Longer Windows

- **WIN-01**: Support for increased qubit count (6-8 qubits, window_length 12-16)
- **WIN-02**: Autoregressive stitching for generating longer sequences from short windows

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-resolution spectral loss | Overkill for T=10 signal with only 6 frequency bins; designed for T=1024+ |
| Focal Frequency Loss (FFL) | Designed for 2D images; adaptive weighting adds complexity without benefit at T=10 |
| Spectral normalization on critic | WGAN-GP already enforces Lipschitz via gradient penalty; redundant |
| StronglyEntanglingLayers template swap | Current manual circuit supports interleaved data re-uploading + PAR_LIGHT encoding |
| Automated circuit architecture search | Enormous compute cost; manual layer count testing sufficient |
| Removing critic Dropout | Reference study uses dropout; keep but make configurable for tuning |
| Migrating to .py module structure | User chose in-place notebook edits (v1.0 decision) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REG-01 | — | Pending |
| REG-02 | — | Pending |
| REG-03 | — | Pending |
| REG-04 | — | Pending |
| REG-05 | — | Pending |
| SPEC-01 | — | Pending |
| SPEC-02 | — | Pending |
| SPEC-03 | — | Pending |
| COND-01 | — | Pending |
| COND-02 | — | Pending |
| COND-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 11 total
- Mapped to phases: 0
- Unmapped: 11 ⚠️

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after initial definition*
