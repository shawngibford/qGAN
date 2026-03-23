# qGAN Post-HPO Improvements

## What This Is

A PennyLane-based Quantum GAN (`qgan_pennylane.ipynb`) for bioprocess time series synthesis of Optical Density data. The notebook implements a WGAN-GP with a quantum generator (parameterized quantum circuit using data re-uploading) and classical 1D-CNN critic with PAR_LIGHT conditioning. v1.0 remediated all 35 code review issues. Post-HPO evaluation revealed persistent variance collapse (fake std 0.0104 vs real 0.0218), regressions from conditioning work (noise range, broadcasting), and identified high-impact improvements (spectral loss, circuit expressivity, critic balance).

## Core Value

The qGAN must generate synthetic OD time series that capture real data's volatility structure — not just the mean trend — with variance, kurtosis, and spectral characteristics that match the training distribution.

## Current Milestone: v1.1 Post-HPO Improvements — COMPLETE

**Goal:** Fix regressions from conditioning work and add high-impact improvements to address variance collapse
**Status:** All 4 phases (4-7) complete as of 2026-03-23

**Target features:**
- Fix noise range regression ([0, 2π] → [0, 4π] in training loop)
- Restore broadcasting optimization (~12x training speedup)
- Fix mu/sigma shadowing
- Add spectral/PSD loss for mid-frequency volatility
- Parameterize circuit layer count (4 → 6-8)
- Verify PAR_LIGHT conditioning actually modulates output
- Add simpler critic architecture option

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

- ✓ Fix noise range to [0, 4π] in all training loop locations — Phase 4
- ✓ Restore broadcasting optimization for batched QNode calls — Phase 5
- ✓ Clean up mu/sigma variable shadowing — Phase 4
- ✓ Add spectral/PSD mismatch loss term — Phase 6
- ✓ Verify PAR_LIGHT conditioning modulates generator output — Phase 7
- ✓ Make critic dropout configurable — Phase 7
- [ ] Make NUM_LAYERS configurable (support 6-8 layers)
- [ ] Add configurable critic architecture (simpler option)

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
- Noise range regressed to [0, 2π] in training loop (3 locations) — PAR_LIGHT conditioning work reintroduced old values
- Broadcasting optimization lost — per-sample Python loops instead of batched QNode calls (~12x slower)
- Cell 10 mu/sigma shadowing on re-execution (non-blocking in linear execution)

### Post-HPO Findings (2026-03-13)
- Variance collapse persists: fake std 0.0104 vs real 0.0218 (48% of target)
- Mean bias 62% off, kurtosis 84% off (0.22 vs 1.40)
- EMD "EXCELLENT", JSD "GOOD" — but moments tell the real story
- Generator learns drift but not volatility structure
- Classical baselines (TinyVAE, FCVAE) also failed — all learned smooth mean curve

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
*Last updated: 2026-03-23 after v1.1 milestone completion (Phase 7)*
