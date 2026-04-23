# qGAN Post-HPO Improvements

## What This Is

A PennyLane-based Quantum GAN (`qgan_pennylane.ipynb`) for bioprocess time series synthesis of Optical Density data. The notebook implements a WGAN-GP with a quantum generator (parameterized quantum circuit using data re-uploading) and classical 1D-CNN critic with PAR_LIGHT conditioning. v1.0 remediated all 35 code review issues. Post-HPO evaluation revealed persistent variance collapse (fake std 0.0104 vs real 0.0218), regressions from conditioning work (noise range, broadcasting), and identified high-impact improvements (spectral loss, circuit expressivity, critic balance).

## Core Value

The qGAN must generate synthetic OD time series that capture real data's volatility structure — not just the mean trend — with variance, kurtosis, and spectral characteristics that match the training distribution.

## Current Milestone: v2.0 AIChE Major Revision Response

**Goal:** Address all reviewer concerns on AIChE Journal manuscript aic-4719598 so the QWGAN-GP bioprocess paper can be resubmitted — establishing quantum-vs-classical evidence, utility-oriented validation, and calibrated claims.

**Target features (Code — Group A, executes first):**
- Extract shared modules from main notebook into `revision/core/` (data pipeline, PQC generator, critic, WGAN-GP training loop)
- Matched-parameter classical WGAN-GP baseline
- Non-adversarial baseline (VAE or AR)
- TSTR evaluation + predictive score + discriminative score
- Inverse-transform results on original OD scale, ACF on both scales
- Shot-noise sensitivity (1024, 8192 shots) and multi-seed (≥5) sweeps
- Depolarizing / amplitude-damping noise-model sensitivity
- Training-progression + circuit-analysis figures (distributions at 0, N/4, N/2, 3N/4, N; parameter/entanglement evolution)
- 2–3 ansatz comparison (depth, entanglement topology)
- Extract full training protocol + dataset statistics → paper-ready tables
- Zenodo DOI / tagged release freeze

**Target features (Paper — Group B, reads numbers from Group A):**
- Reframe hypothesis; tone down "industrial monitoring" / "computational advantages" language
- Circuit design rationale subsection (why 5 qubits, why this ansatz, why classical critic + quantum generator)
- Justify log-returns in bioprocess context (growth-rate interpretation)
- Move decision-tree workflow + Hybrid-GAN mechanistic material to explicit Outlook section; caveat Table A2
- Fix misplaced references [27][28][39][18][19][41][55–57][59]; add Bernal et al. AIChE perspective
- Report dataset details (raw points, windows, train/val/test splits) in Methods
- State evaluation scale per metric (transformed vs. OD) in Methods
- Clarify Appendix A3 log-GAN vs. Wasserstein discrepancy
- Typos / notation unification / figure sizing

**Past milestones:** v1.0 Code Review Remediation (2026-03-07), v1.1 Post-HPO Improvements (2026-03-23 — all 4 phases complete)

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

### Active (v2.0 — Reviewer Response)

Code group (Group A):
- [ ] Extract shared modules (data, PQC, critic, training loop) into `revision/core/`
- [ ] Matched-parameter classical WGAN-GP baseline
- [ ] Non-adversarial baseline (VAE or AR)
- [ ] TSTR + predictive + discriminative score evaluation
- [ ] OD-scale inverse-transform results + ACF on both scales
- [ ] Shot-noise sensitivity (1024, 8192) + multi-seed (≥5) sweeps
- [ ] Depolarizing / amplitude-damping noise-model sensitivity
- [ ] Training-progression + circuit-analysis figures
- [ ] 2–3 ansatz comparison
- [ ] Training-protocol + dataset-stats extraction
- [ ] Zenodo / tagged-release repository freeze

Paper group (Group B):
- [ ] Reframe hypothesis + tone-down claims throughout
- [ ] Circuit design rationale subsection
- [ ] Log-returns bioprocess justification
- [ ] Move decision-tree + Hybrid-GAN to Outlook
- [ ] Fix misplaced references; add Bernal et al.
- [ ] Report dataset details in Methods
- [ ] Clarify evaluation scale per metric
- [ ] Clarify A3 log-GAN vs Wasserstein discrepancy
- [ ] Typos / notation / figure sizing

### Previously Validated (from v1.1)

- ✓ Fix noise range to [0, 4π] in all training loop locations — Phase 4
- ✓ Restore broadcasting optimization for batched QNode calls — Phase 5
- ✓ Clean up mu/sigma variable shadowing — Phase 4
- ✓ Add spectral/PSD mismatch loss term — Phase 6
- ✓ Verify PAR_LIGHT conditioning modulates generator output — Phase 7
- ✓ Make critic dropout configurable — Phase 7

### Out of Scope

- ~~Migrating to a .py module structure~~ — v2.0 REINSTATES partial module extraction into `revision/core/` (shared modules only; main notebook unchanged)
- Qutrit circuit architectures — separate experimental notebooks
- Hardware execution on real QPU — simulator-only (stated honestly in paper per R1-M5)
- Implementing the full closed-loop decision pipeline — moved to Outlook per R2-3
- First-principles Hybrid-GAN implementation — moved to Outlook per R2-5a
- Full CSV schema validation — research tool, not production service
- Checkpoint compression / delta checkpointing — operational concern, not correctness

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

- **Main notebook untouched**: `qgan_pennylane.ipynb` stays as-is; revision work lives in `revision/`
- **Code structure**: `revision/core/` shared Python modules + `revision/NN_topic.ipynb` parallel notebooks (notebooks orchestrate + plot + write JSON only)
- **Compute**: Local Mac only — statevector simulator, multi-seed sweeps must be sized accordingly
- **Data path**: Relative `./data.csv` (shared with main notebook)
- **Compatibility**: PennyLane 0.44.0 and PyTorch 2.8.0 in qgan_env
- **Results contract**: Each revision notebook writes structured JSON to `revision/results/<name>.json` so paper-writing reads numbers from one place
- **Paper scope**: Manuscript aic-4719598 — AIChE Journal Major Revision

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-23 — v2.0 AIChE Major Revision milestone opened*
