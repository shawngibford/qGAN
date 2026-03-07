# qGAN Code Review Remediation

## What This Is

A PennyLane-based Quantum GAN (`qgan_pennylane.ipynb`) for bioprocess time series synthesis of Optical Density data. The notebook implements a WGAN-GP with a quantum generator (parameterized quantum circuit using data re-uploading) and classical 1D-CNN critic. v1.0 addressed all 35 issues from a comprehensive code review — the training pipeline now follows WGAN-GP theory, the quantum circuit supports universal approximation, and metrics accurately reflect output quality.

## Core Value

The qGAN must produce correct, reproducible results with a training pipeline where metrics accurately reflect output quality, model checkpoints actually save/restore, and WGAN-GP hyperparameters follow established theory.

## Requirements

### Validated

- ✓ Quantum generator using PennyLane PQC with data re-uploading and strongly entangled layers — v1.0
- ✓ Classical 1D-CNN critic (WGAN-GP) using PyTorch with no dropout — v1.0
- ✓ Data preprocessing pipeline: CSV load → log-returns → Lambert W transform → rolling windows → normalization — v1.0
- ✓ Stylized facts evaluation: ACF, volatility clustering, leverage effect, EMD — v1.0
- ✓ EMD-based early stopping with checkpoint save/restore — v1.0
- ✓ Visualization: loss curves, distribution comparisons, time series plots, DTW analysis — v1.0
- ✓ All 7 correctness bugs fixed (checkpoint naming, scaling consistency, memory leaks, etc.) — v1.0
- ✓ All 5 performance issues fixed (backprop diff_method, periodic eval, torch.no_grad, DataLoader, broadcasting) — v1.0
- ✓ WGAN-GP standard hyperparameters restored (N_CRITIC=5, LAMBDA=10, balanced LR ratio) — v1.0
- ✓ EMD computed on raw samples via wasserstein_distance (not histograms) — v1.0
- ✓ Quantum circuit redesigned: data re-uploading, [0, 4pi] noise range, PauliX+Z measurements — v1.0
- ✓ WINDOW_LENGTH computed automatically from NUM_QUBITS — v1.0
- ✓ All code quality issues resolved (dead code, duplicates, naming, eval() removal) — v1.0

### Active

(None — next milestone not yet defined)

### Out of Scope

- Migrating to a .py module structure — user chose in-place notebook edits
- Qutrit circuit architectures — separate experimental notebooks
- Validation/test split — separate concern from code review fixes
- Learning rate scheduling — not in review scope
- Multiple training runs for statistical significance — too expensive for single remediation pass
- Checkpoint compression / delta checkpointing — operational concern, not correctness
- Full CSV schema validation — research tool, not production service

## Context

Shipped v1.0 with 1,814 lines of Python across 44 code cells in `qgan_pennylane.ipynb`.
Tech stack: PennyLane 0.44.0, PyTorch 2.8.0, SciPy (wasserstein_distance), dtw-python.
Net change: 730 insertions, 1,576 deletions (substantial cleanup).
PhD research project — the notebook has qutrit experimental variants (phase2, phase2b, phase2c) that were not in remediation scope.

### Known Tech Debt
- Cell 10 mu/sigma shadowing on re-execution (non-blocking in linear execution)
- Cell 37 diagnostic noise range mismatch (diagnostic-only cell)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Edit existing notebook in-place | Preserve git history and avoid file proliferation | ✓ Good — 1 file modified throughout |
| Restore standard WGAN-GP hyperparameters | n_critic=1 and LAMBDA=0.8 diverge from theory without documented justification | ✓ Good — N_CRITIC=5, LAMBDA=10 restored |
| Redesign quantum circuit (all 5 issues) | Full circuit fix maximizes expressivity and correctness | ✓ Good — data re-uploading + backprop + expanded noise |
| Switch diff_method to backprop | ~90x speedup for gradient computation on simulator | ✓ Good — significant training speedup |
| Monitor EMD for early stopping | Critic loss is not a reliable quality metric in WGAN | ✓ Good — EMD directly measures distributional fidelity |
| Remove model_state_dict from checkpoint | qGAN uses params_pqc + critic separately | ✓ Good — simpler checkpoint format |
| Changed encoding from RZ to RX | Non-commutativity with Rot gate RZ components | ✓ Good — avoids redundant rotation |
| Inline mu/sigma into norm.pdf() calls | Eliminates variable shadowing with zero risk | ✓ Good — cleaner than renaming |
| Keep DTW perturbation as ablation study | Intentional sensitivity analysis, not a bug | ✓ Good — consolidated into single cell |

## Constraints

- **Format**: All fixes in existing `qgan_pennylane.ipynb` — no new files
- **Data path**: Relative `./data.csv`
- **Compatibility**: PennyLane 0.44.0 and PyTorch 2.8.0 in qgan_env
- **Quantum circuit**: Output dimensionality = 2 * NUM_QUBITS measurements

---
*Last updated: 2026-03-07 after v1.0 milestone*
