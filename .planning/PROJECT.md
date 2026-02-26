# qGAN Code Review Remediation

## What This Is

A comprehensive fix-up of the PennyLane-based Quantum GAN (`qgan_pennylane.ipynb`) for time series synthesis of bioprocess Optical Density data. The notebook implements a WGAN-GP with a quantum generator (parameterized quantum circuit) and classical 1D-CNN critic. A code review identified ~40 issues across correctness, performance, ML theory, quantum circuit design, and code quality — all of which will be addressed in-place.

## Core Value

The qGAN must produce correct, reproducible results with a training pipeline where metrics accurately reflect output quality, model checkpoints actually save/restore, and WGAN-GP hyperparameters follow established theory.

## Requirements

### Validated

- ✓ Quantum generator using PennyLane PQC with IQP encoding and strongly entangled layers — existing
- ✓ Classical 1D-CNN critic (WGAN-GP) using PyTorch — existing
- ✓ Data preprocessing pipeline: CSV load → log-returns → Lambert W transform → rolling windows → normalization — existing
- ✓ Stylized facts evaluation: ACF, volatility clustering, leverage effect, EMD — existing
- ✓ Early stopping with checkpoint save/restore — existing (buggy, to be fixed)
- ✓ Visualization: loss curves, distribution comparisons, time series plots, DTW analysis — existing

### Active

- [ ] Fix all 7 correctness bugs (checkpoint naming, scaling consistency, memory leaks, hardcoded epoch, global delta, exit() call)
- [ ] Fix all 5 performance issues (backprop diff_method, periodic evaluation, torch.no_grad, DataLoader usage, sequential circuits)
- [ ] Restore WGAN-GP standard hyperparameters (n_critic=5, LAMBDA=10, balanced LR ratio, remove dropout from critic)
- [ ] Fix EMD computation (raw samples, not histograms) and remove hardcoded histogram bins
- [ ] Improve early stopping to monitor EMD instead of critic loss
- [ ] Redesign quantum circuit: remove redundant IQP RZ, add data re-uploading, expand noise range to [0, 4pi], add multi-qubit measurements
- [ ] Make WINDOW_LENGTH computed from NUM_QUBITS automatically
- [ ] Fix all code quality issues (relative data path, dead code removal, duplicate imports, eval() removal, normalize() returning stats, consistent naming)
- [ ] Remove debug artifacts (Cell 50 `d`, Cell 49 data perturbation hack, duplicate plotting cells)
- [ ] Add torch.load weights_only=True for safe checkpoint loading

### Out of Scope

- Migrating to a .py module structure — user chose in-place notebook edits
- Adding new quantum circuit architectures (qutrits, etc.) — focus is on fixing the existing qubit notebook
- Adding a validation/test split — separate concern from the code review fixes
- Learning rate scheduling — not in the review scope

## Context

- This is a PhD research project implementing qGANs for bioprocess time series synthesis
- The notebook has been through multiple experimental iterations (phase2, phase2b, phase2c variants exist for qutrits)
- The code review was performed against the main `qgan_pennylane.ipynb` notebook
- A codebase map exists at `.planning/codebase/` with 7 documents covering architecture, stack, conventions, testing, concerns, integrations, and structure
- The project uses a Python venv (`qgan_env/`) with PennyLane 0.44.0 and PyTorch 2.8.0
- Data file is `data.csv` containing Optical Density measurements

## Constraints

- **Format**: All fixes go into the existing `qgan_pennylane.ipynb` — no new files except as needed for clean separation
- **Data path**: Use relative `./data.csv` instead of hardcoded absolute path
- **Compatibility**: Must work with current PennyLane 0.44.0 and PyTorch 2.8.0 in qgan_env
- **Quantum circuit**: Circuit redesign must maintain the same output dimensionality (2 * NUM_QUBITS measurements) to keep critic architecture compatible

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Edit existing notebook in-place | Preserve git history and avoid file proliferation | — Pending |
| Restore standard WGAN-GP hyperparameters | n_critic=1 and LAMBDA=0.8 diverge from theory without documented justification | — Pending |
| Redesign quantum circuit (all 5 issues) | Full circuit fix maximizes expressivity and correctness | — Pending |
| Switch diff_method to backprop | ~90x speedup for gradient computation on simulator | — Pending |
| Monitor EMD for early stopping | Critic loss is not a reliable quality metric in WGAN | — Pending |

---
*Last updated: 2026-02-26 after initialization*
