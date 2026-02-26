# Features Research: qGAN WGAN-GP Notebook

**Research Date:** 2026-02-26
**Research Type:** Project Research — Features dimension for qGAN WGAN-GP code remediation
**Milestone:** Subsequent — What features/capabilities a well-implemented qGAN notebook should have

---

## Summary

A well-implemented qGAN WGAN-GP notebook has two distinct tiers: a correctness floor that every serious implementation must meet, and a research-quality ceiling that separates publishable work from functional experiments. For this remediation, roughly 30 of the ~40 identified issues are table-stakes correctness problems. The remaining 10 are research-quality differentiators.

---

## Table Stakes (Must Have for Correctness)

### TS-1: WGAN-GP Standard Hyperparameters
n_critic=5, lambda=10, no dropout in critic, balanced learning rate ratio.
**Complexity:** Low — parameter value changes only

### TS-2: Correct EMD Computation on Raw Samples
`scipy.stats.wasserstein_distance` on raw arrays, not histogram-binned distributions.
**Complexity:** Low-Medium

### TS-3: Checkpoint Save/Restore Correctness
Save `nn.Parameter` data correctly; load and re-wrap. Fix discriminator→critic naming.
**Complexity:** Low

### TS-4: Safe Checkpoint Loading (weights_only=True)
PyTorch 2.6+ default. Prevents arbitrary code execution via pickle.
**Complexity:** Trivial

### TS-5: Generator Output Scaling Consistency
Scale factor must be consistent across training, evaluation, and generation. No ad-hoc `*0.1` inconsistencies.
**Complexity:** Low

### TS-6: No exit() in Notebook
Remove — kills kernel, destroys all in-memory state.
**Complexity:** Trivial

### TS-7: Remove Global State Mutation (delta variable)
Scope `delta` inside class or pass as parameter.
**Complexity:** Low

### TS-8: Remove Hardcoded Epoch Number
Use configured `NUM_EPOCHS`, not literal `3000`.
**Complexity:** Trivial

### TS-9: torch.no_grad() for Evaluation
Wrap all inference-time forward passes to prevent memory leak.
**Complexity:** Low

### TS-10: Relative Data Path
`./data.csv` instead of hardcoded absolute path.
**Complexity:** Trivial

### TS-11: Remove eval() Calls
Replace with `globals().get()` or explicit logic.
**Complexity:** Low

### TS-12: Dead Code / Duplicate Import Removal
Remove Cell 50 `d`, Cell 49 perturbation hack, duplicate imports, dead `compute_gradient_penalty` method.
**Complexity:** Low

### TS-13: DataLoader Correctness
Use DataLoader with proper batch sampling instead of flattening to list.
**Complexity:** Low-Medium

### TS-14: normalize() Returns Statistics
Return `(normalized_data, mu, sigma)` for invertible normalization.
**Complexity:** Low

---

## Differentiators (Research Quality Improvements)

### D-1: backprop diff_method
~90x speedup for gradient computation on simulator.
**Complexity:** Low

### D-2: EMD-Based Early Stopping
Monitor EMD instead of critic loss for stopping criterion.
**Complexity:** Medium
**Depends on:** TS-2

### D-3: Data Re-Uploading in Quantum Circuit
Interleave noise encoding between variational layers for universal approximation.
**Complexity:** Medium

### D-4: Multi-Qubit Measurements
Add PauliX alongside PauliZ to capture entanglement information.
**Complexity:** Medium
**Depends on:** D-3

### D-5: Expanded Noise Range [0, 4pi]
Full coverage of RZ gate period.
**Complexity:** Trivial
**Depends on:** D-3

### D-6: Remove Redundant IQP RZ Gate
Eliminate wasted parameters.
**Complexity:** Low

### D-7: WINDOW_LENGTH Derived from NUM_QUBITS
`WINDOW_LENGTH = 2 * NUM_QUBITS` enforced automatically.
**Complexity:** Trivial

### D-8: Periodic Evaluation (Not Every Epoch)
Compute metrics every N epochs to reduce wall-clock time.
**Complexity:** Low
**Depends on:** D-2

### D-9: Stylized Facts Audit
Verify existing ACF, volatility, leverage implementations against correct definitions.
**Complexity:** Medium

### D-10: normalize() Returns Statistics
Enable inverse normalization for original-space comparison.
**Complexity:** Low

---

## Anti-Features (Deliberately NOT Add)

| Anti-Feature | Reason |
|---|---|
| AF-1: Module migration (.py) | Project decision: in-place notebook edits |
| AF-2: Qutrit architectures | Separate experimental notebooks; not in scope |
| AF-3: Train/validation split | Changes experimental protocol |
| AF-4: LR scheduling | Not in review scope; no evidence it's needed |
| AF-5: Hyperparameter optimization (Optuna) | Separate concern; would balloon scope |
| AF-6: Statistical significance testing | Requires multiple training runs |
| AF-7: Checkpoint compression | Operational concern, not correctness |
| AF-8: Full CSV schema validation | Research tool, not production service |
| AF-9: nbstripout / CI/CD | Dev infrastructure, not code remediation |

---

## Dependency Map

```
TS-1 (hyperparams) → TS-2 (EMD meaningful)
TS-2 (correct EMD) → D-2 (early stopping on EMD) → D-8 (eval frequency)
TS-3 (checkpoint) → TS-4 (weights_only)
TS-9 (no_grad) → D-8 (periodic eval feasible)
D-3 (re-uploading) → D-4 (multi-qubit) → D-7 (WINDOW_LENGTH formula)
                    → D-5 (noise range)
                    → D-6 (redundant gates)
```

---
*Research completed: 2026-02-26*
