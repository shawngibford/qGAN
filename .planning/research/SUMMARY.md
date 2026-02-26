# Project Research Summary

**Project:** qGAN PennyLane WGAN-GP Remediation
**Domain:** Quantum machine learning — Jupyter notebook code remediation
**Researched:** 2026-02-26
**Confidence:** HIGH

## Executive Summary

This project is a targeted remediation of `qgan_pennylane.ipynb`, a quantum GAN using WGAN-GP to generate synthetic financial time series. The notebook has approximately 40 issues spanning four categories: correctness bugs (wrong hyperparameters, broken checkpoint logic, EMD computed on histograms), performance inefficiencies (parameter-shift where backprop gives ~90x speedup), ML theory violations (dropout in WGAN-GP critic, early stopping on the wrong metric), and code quality problems (dead code, global state, insecure loading). No library upgrades are required — PennyLane 0.44.0, PyTorch 2.8.0, and scipy 1.15.3 already support all recommended fixes.

The recommended approach is a three-phase remediation that respects strict dependency ordering. Phase 1 addresses foundation issues — safe infrastructure changes with no training behavior impact. Phase 2 implements WGAN-GP correctness and quantum circuit redesign — the highest-risk changes that will alter training dynamics and invalidate existing checkpoints. Phase 3 is post-processing consistency and cleanup. Mixing phases is dangerous: WINDOW_LENGTH must be derived from NUM_QUBITS before the circuit is redesigned, EMD must be fixed before early stopping is changed, and normalize() must be updated atomically across all call sites.

The key risk in this remediation is cascading silent failures. Several fixes appear small but have wide-reaching effects: changing normalize() breaks all downstream denormalization; changing the circuit output dimension breaks the critic's Conv1D; saving checkpoints with the wrong type silently disables gradient updates. The research consistently identifies "fix together, test atomically, restart kernel between phases" as the mitigation strategy. All pitfalls have well-documented prevention patterns — the risk is sequencing errors, not unknown unknowns.

---

## Key Findings

### Recommended Stack

The existing library stack is correct and no upgrades are needed. The critical fix is switching `diff_method` from `'parameter-shift'` to `'backprop'` on `default.qubit` — this alone eliminates ~90x unnecessary circuit evaluations per training step. `pennylane-lightning` is installed and available as a future upgrade path (`adjoint` method, 2-8x additional speedup), but switching to `backprop` on `default.qubit` is the minimal-change fix with no backend configuration risk.

**Core technologies:**
- `PennyLane 0.44.0`: QNode definition, circuit design — `backprop` diff_method is the correct choice for simulator-only training
- `pennylane-lightning 0.44.0`: Available as upgrade path for `adjoint` method — do not use with `backprop`
- `PyTorch 2.8.0`: Model training — `weights_only=True` is now the default on `torch.load` and must be used; `.item()` on losses prevents memory leaks
- `scipy 1.15.3`: EMD computation — `wasserstein_distance` takes raw sample arrays, not histogram bins

**Critical version constraint:** Do not mix `shots > 0` with `backprop`. The notebook already uses `shots=None` (analytic mode) — constraint already satisfied.

See: `.planning/research/STACK.md`

### Expected Features

Research identifies a clear two-tier structure: 14 table-stakes correctness items and 10 research-quality differentiators. Roughly 30 of the ~40 issues are correctness problems. The anti-features list is equally important — module migration, qutrit architectures, train/val splits, and CI/CD are all explicitly out of scope.

**Must have (table stakes) — correctness floor:**
- WGAN-GP standard hyperparameters: `n_critic=5`, `lambda=10`, no dropout in critic, Adam betas `(0.0, 0.9)`
- Correct EMD computation on raw sample arrays
- Checkpoint save/load with correct `nn.Parameter` wrapping and `model.critic` naming
- `weights_only=True` on all `torch.load` calls
- Consistent generator output scaling (no ad-hoc `* 0.1` at multiple call sites)
- `torch.no_grad()` around all evaluation forward passes
- Remove `exit()`, `eval()`, hardcoded absolute paths, hardcoded epoch numbers, dead code

**Should have (research quality differentiators):**
- `backprop` diff_method (~90x speedup)
- Data re-uploading in quantum circuit (universal approximation)
- Multi-qubit measurements (PauliX alongside PauliZ)
- EMD-based early stopping (direct distributional metric)
- `WINDOW_LENGTH = 2 * NUM_QUBITS` derived constant
- Expanded noise range `[0, 4pi]` (full RZ period coverage)

**Defer to v2+:**
- Module migration to `.py` files
- Qutrit architectures (separate experimental notebooks exist)
- Train/validation splits (changes experimental protocol)
- Hyperparameter optimization (Optuna)
- LR scheduling
- Statistical significance testing

See: `.planning/research/FEATURES.md`

### Architecture Approach

The notebook has six component boundaries (Imports, Data/Preprocessing, Hyperparameter Config, qGAN Class, Training Execution, Visualization) with a strict left-to-right data flow. Five dependency tiers emerge from this structure, directly mapping to three remediation phases. The most dangerous architectural constraint is that `WINDOW_LENGTH` acts as a coupling constant between three components: the rolling window in the DataLoader, the quantum circuit output dimension, and the critic's Conv1D input — any circuit change that alters measurement count must go through this constant first.

**Major components:**
1. **Component A — Imports**: Trivial deduplication; must be clean before any structural changes
2. **Component B — Data/Preprocessing**: `normalize()` signature change is breaking; must update all call sites atomically
3. **Component C — Hyperparameter Config**: Central configuration cell; `WINDOW_LENGTH = 2 * NUM_QUBITS` must be set before circuit redesign
4. **Component D — qGAN Class**: Highest-risk component; D1 (circuit), D2 (critic), D3 (training loop) each have independent fix paths with cross-dependencies
5. **Component E — Training Execution / Post-Generation**: Depends on D being correct; `delta` global variable must be scoped properly before this component is valid
6. **Component F — Visualization**: Final cleanup; duplicate cells and dead artifacts removed last

See: `.planning/research/ARCHITECTURE.md`

### Critical Pitfalls

1. **Critic input dimension mismatch after circuit redesign** (P-1, Critical) — Fix `WINDOW_LENGTH = 2 * NUM_QUBITS` and verify shape alignment *before* redesigning the circuit. Shape mismatches silently corrupt forward passes.

2. **Checkpoint parameter type mismatch** (P-2, Critical) — Saving `params_pqc.detach().clone()` strips `nn.Parameter`; loading assigns a plain tensor with no gradients. Optimizer makes zero updates silently. Always wrap on save AND load: `nn.Parameter(checkpoint['params_pqc'])`. Also fix `model.discriminator` → `model.critic` naming.

3. **Normalization/scaling pipeline cascades silently** (P-3, Critical) — Fixing `normalize()` return value and `* 0.1` generator scaling in separate passes invalidates stale `real.csv`/`fake.csv` and all in-memory variables. Must fix both in the same pass, delete stale CSVs, and restart the kernel.

4. **`diff_method='backprop'` breaks on Lightning backend** (P-4, High) — `pennylane-lightning` may override `default.qubit`. Lightning does not support `backprop`. Use `qml.device("default.qubit", wires=n)` explicitly and test on a one-qubit toy circuit before full training.

5. **Restoring WGAN-GP hyperparameters forces fresh training run** (P-5, High) — Old checkpoints trained with `n_critic=1, LAMBDA=0.8` are incompatible with restored standard values. Both the hyperparameters and the training loop structure (critic iteration count) must change together. Existing `checkpoints_phase2c/` should be abandoned.

See: `.planning/research/PITFALLS.md`

---

## Implications for Roadmap

Based on combined research, three phases are strongly recommended. The dependency chain from ARCHITECTURE.md directly produces this phase structure. Mixing phases is the primary risk.

### Phase 1: Foundation and Correctness Infrastructure
**Rationale:** Tier 1 and Tier 2 fixes have no inter-dependencies and no training behavior impact. They establish a clean, safe base before any architectural change. The `exit()` call, insecure `torch.load`, duplicate imports, and dead code must go before anything else touches the notebook (P-6: stale kernel state breaks subsequent edits).
**Delivers:** Clean, reproducible, memory-safe notebook that can be run top-to-bottom without kernel corruption. Correct EMD computation unblocked. Checkpoint save/load pipeline correct.
**Addresses:** TS-3, TS-4, TS-6, TS-7, TS-9, TS-10, TS-11, TS-12, TS-13, TS-14 (table stakes from FEATURES.md)
**Avoids:** P-2 (checkpoint type mismatch), P-9 (exit kills kernel), P-15 (memory leaks), P-11 (hardcoded path)

### Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign
**Rationale:** These are the highest-impact, highest-risk changes. They must be done together because they all invalidate existing checkpoints and significantly alter training behavior. WINDOW_LENGTH must be set first (P-1 prevention), then circuit redesign, then hyperparameter restoration, then early stopping switch. The diff_method change (P-4 prevention) must explicitly specify the device.
**Delivers:** A notebook that implements WGAN-GP correctly per Gulrajani et al. (2017), with a well-designed quantum circuit supporting universal approximation via data re-uploading. A fresh training run from this phase should produce meaningfully better distributional fidelity.
**Uses:** PennyLane 0.44.0 `backprop` on `default.qubit`; standard WGAN-GP `n_critic=5, LAMBDA=10`
**Implements:** Component C (hyperparameters), Component D1 (circuit), Component D2 (critic), Component D3 (training loop)
**Addresses:** TS-1, TS-2, TS-5, D-1, D-2, D-3, D-4, D-5, D-6, D-7, D-8 (from FEATURES.md)
**Avoids:** P-1 (dimension mismatch), P-3 (normalize cascade), P-4 (Lightning backend), P-5 (incompatible checkpoints), P-7 (histogram EMD), P-8 (dropout in critic), P-14 (wrong early stopping metric)

### Phase 3: Post-Processing Consistency and Cleanup
**Rationale:** After Phase 2 training is validated, fix delta consistency in generation cells, unify denormalization strategy, extract OUTPUT_SCALE, and remove all debug artifacts. These are low-risk cleanup tasks that require Phase 2 to be complete and verified first — they depend on the correct scaling and circuit pipeline being established.
**Delivers:** A publication-ready notebook with no dead code, consistent data pipeline from training through generation, and no hardcoded magic constants.
**Implements:** Component E (generation cells), Component F (visualization), Component D dead code removal
**Addresses:** TS-7, TS-8, TS-12; remaining anti-features removed (AF items)
**Avoids:** P-6 (cell order dependencies), P-10 (eval), P-12 (requirements pin), P-13 (duplicate plots)

### Phase Ordering Rationale

- **Phase 1 before Phase 2**: The `exit()` call and checkpoint naming bugs can kill a training run. These must be gone before any long training experiment. The EMD fix (prerequisite for early stopping) belongs in Phase 1 so Phase 2 can use it immediately.
- **WINDOW_LENGTH before circuit redesign**: The ARCHITECTURE.md dependency analysis makes this mandatory (P-1). Attempting circuit redesign without fixing this constant first is the single most likely way to introduce a silent shape mismatch.
- **Hyperparameters and training loop together**: P-5 is explicit — these two changes must land in the same commit/pass. Hyperparameter changes alone with the old training loop structure are insufficient.
- **Phase 3 after Phase 2 is validated**: Post-processing consistency depends on the correct scaling pipeline being established and a training run verifying it works.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (circuit redesign sub-task):** Data re-uploading interleaved between variational layers requires careful weight count verification against `WINDOW_LENGTH = 2 * NUM_QUBITS`. The exact interleaving pattern and its effect on circuit parameter count needs validation against the existing Conv1D critic architecture.
- **Phase 2 (stylized facts audit — D-9):** ACF, volatility clustering, and leverage effect implementations should be audited against correct statistical definitions before finalizing visualization.

Phases with standard patterns (skip deeper research):
- **Phase 1:** All changes are well-documented bug fixes with no ambiguity. Standard PyTorch and PennyLane patterns.
- **Phase 3:** Pure cleanup — no research needed.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PennyLane 0.44.0 and PyTorch 2.8.0 docs confirm all recommendations; `backprop` stable since v0.18; `weights_only=True` default since PyTorch 2.6 |
| Features | HIGH | Clear two-tier structure derived from codebase analysis; table stakes are definitively broken, not ambiguous |
| Architecture | HIGH | Component boundaries and dependency tiers derived from actual notebook structure; data flow is explicit |
| Pitfalls | HIGH | Based on existing codebase analysis; most pitfalls are already-observable bugs, not speculative risks |

**Overall confidence:** HIGH

The high confidence across all areas reflects that this is a remediation project on an existing, fully-inspectable codebase. The research is not speculative — the issues are observable. The recommendations come from authoritative sources (Gulrajani et al., PennyLane docs, PyTorch docs, scipy docs).

### Gaps to Address

- **Stylized facts correctness (D-9):** Research flagged that existing ACF/volatility/leverage implementations need auditing against correct definitions, but did not complete this audit. Handle during Phase 2 or 3 planning with explicit review of each visualization cell.
- **Noise range change impact on training stability:** Expanding noise from `[0, 2pi]` to `[0, 4pi]` is theoretically correct (full RZ period) but its effect on training stability at this specific circuit depth is medium-confidence. Monitor first training run after Phase 2 carefully.
- **requirements.txt version pin:** Currently `pennylane>=0.32.0`; should be pinned to `pennylane==0.44.0`. Not blocking but should be addressed in Phase 3 cleanup.

---

## Sources

### Primary (HIGH confidence)
- [PennyLane QNode API — 0.44.0](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) — diff_method options, backprop requirements
- [Quantum Gradients with Backpropagation | PennyLane Demos](https://pennylane.ai/qml/demos/tutorial_backprop) — backprop vs parameter-shift comparison
- [Gulrajani et al. (2017) — Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — n_critic=5, LAMBDA=10, no dropout, Adam betas
- [torch.load — PyTorch 2.10](https://docs.pytorch.org/docs/stable/generated/torch.load.html) — weights_only=True default since 2.6
- [scipy.stats.wasserstein_distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) — raw sample input, not histogram bins

### Secondary (MEDIUM confidence)
- [Perez-Salinas et al. (2020) — Data Re-uploading for a Universal Quantum Classifier](https://quantum-journal.org/papers/q-2020-02-06-226/) — data re-uploading enables universal approximation
- PennyLane 0.44.0 parameter broadcasting documentation — batch execution without Python loop

### Tertiary (LOW confidence)
- Noise range `[0, 4pi]` impact on training stability — theoretically motivated, empirical validation needed on this specific circuit
- Stylized facts implementation correctness — flagged as needing audit, not yet verified

---
*Research completed: 2026-02-26*
*Ready for roadmap: yes*
