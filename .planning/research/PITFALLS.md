# Pitfalls Research: qGAN Code Remediation

**Research Date:** 2026-02-26
**Domain:** Quantum GAN refactoring / WGAN-GP best practices
**Confidence:** High (based on existing codebase analysis and known issues)

## Critical Pitfalls

### P-1: Breaking Critic Input Dimensionality During Circuit Redesign
**Severity:** Critical
**Warning Signs:** Shape mismatch errors during forward pass after circuit changes
**Description:** The critic's Conv1D is hardcoded to `[batch, 1, window_length]`. Any circuit change that alters measurement count silently mismatches. `WINDOW_LENGTH = 2 * NUM_QUBITS` is an implicit constraint.
**Prevention:** Fix `WINDOW_LENGTH = 2 * NUM_QUBITS` computation first, then redesign the circuit, then verify shape alignment before touching the critic.
**Phase:** Should be addressed in early phase (before circuit redesign)

### P-2: Checkpoint Parameter Type Mismatch on Load
**Severity:** Critical
**Warning Signs:** Optimizer silently makes no updates after loading checkpoint; loss doesn't change
**Description:** `params_pqc.detach().clone()` strips `nn.Parameter`, storing a plain `Tensor`. Loading assigns a non-gradient Tensor — optimizer silently makes no updates. Current code also references `model.discriminator` instead of `model.critic`.
**Prevention:** Always wrap on save and load: `nn.Parameter(checkpoint['params_pqc'])`. Add `requires_grad` assertion after load. Fix discriminator→critic naming.
**Phase:** Must fix in correctness bugs phase

### P-3: Scaling/Normalization Pipeline Changes Cascade Silently
**Severity:** Critical
**Warning Signs:** Metrics suddenly look very different after a "small" fix; generated data range changes unexpectedly
**Description:** Fixing `normalize()` to return stats, or the `*0.1` generator output scaling, invalidates stale `real.csv`/`fake.csv` and all in-memory evaluation variables. Must restart kernel and rerun all cells top-to-bottom.
**Prevention:** Fix normalize() return value and generator scaling in the same pass. Delete stale CSV outputs. Document that kernel restart is required.
**Phase:** Should be grouped together in a single phase

### P-4: diff_method Change Breaks Gradient Flow on Lightning Backend
**Severity:** High
**Warning Signs:** Error on QNode creation or silent fallback to parameter-shift
**Description:** `pennylane-lightning 0.44.0` is installed and may override `default.qubit`. Lightning does not support `diff_method="backprop"`. Need to verify the active backend before switching diff_method.
**Prevention:** Explicitly use `qml.device("default.qubit", wires=n)` (not lightning). Test on a one-qubit toy circuit first to verify backprop works.
**Phase:** Performance fixes phase

### P-5: Restoring Standard WGAN-GP Hyperparameters Forces Retraining
**Severity:** High
**Warning Signs:** Training diverges immediately after loading old checkpoint with new hyperparameters
**Description:** Old checkpoints are incompatible with `n_critic=5, LAMBDA=10`. The training loop structure must also be updated to actually use n_critic (ensure the critic loop iterates n_critic times). Both changes must happen together, and existing checkpoints should be abandoned.
**Prevention:** Change hyperparameters and training loop together. Start fresh training run.
**Phase:** ML theory fixes phase

## Moderate Pitfalls

### P-6: Jupyter Cell Execution Order Dependencies Break After Edits
**Severity:** Moderate
**Warning Signs:** NameError or stale variable values after editing cells
**Description:** Stale kernel variables shadow fixes. The global `delta` bug fix, class redefinitions, and DataLoader rebuilds all require a kernel restart + full re-run to take effect.
**Prevention:** Remove debug artifacts (Cell 49/50) first. After each phase of fixes, recommend kernel restart + run-all to validate.
**Phase:** Code cleanup phase (early)

### P-7: EMD Computed on Histograms Instead of Raw Samples
**Severity:** Moderate
**Warning Signs:** EMD values that seem disconnected from visual distribution quality
**Description:** Current code passes histogram bin values to `wasserstein_distance()` — result is bin-count-dependent, biased. Replace with `wasserstein_distance(real_samples.flatten(), fake_samples.flatten())`.
**Prevention:** Must be fixed before switching early stopping to monitor EMD (P-7 is prerequisite for P-14).
**Phase:** ML theory fixes phase

### P-8: Dropout in WGAN-GP Critic Corrupts Gradient Penalty
**Severity:** Moderate
**Warning Signs:** Gradient penalty values oscillate wildly; training instability
**Description:** Dropout makes the critic stochastic — gradient penalty is computed on a different sub-network each pass, violating the Lipschitz constraint.
**Prevention:** Remove dropout layers from the critic entirely (shape-preserving change, safe to do).
**Phase:** ML theory fixes phase

### P-9: exit() Call Terminates the Kernel Mid-Training
**Severity:** Moderate
**Warning Signs:** Kernel dies unexpectedly, all in-memory state lost
**Description:** `exit()` kills the IPython kernel — all training history, model parameters lost. No stack trace.
**Prevention:** Replace with `break` or `return` or conditional flow. Remove before any training run.
**Phase:** Correctness bugs phase (early)

### P-10: eval() on Dynamic Input Can Corrupt Global State
**Severity:** Low-Moderate
**Warning Signs:** Unexpected variable values after running debug cells
**Description:** `eval()` with a variable argument executes arbitrary Python. In notebook context, risk is low but pattern is bad.
**Prevention:** Replace with `globals().get(var_name)` or specific type conversions.
**Phase:** Code quality phase

### P-11: Hardcoded Absolute Data Path Breaks Reproducibility
**Severity:** Low
**Warning Signs:** FileNotFoundError on any other machine
**Description:** `pd.read_csv('/Users/shawngibford/...')` fails on any other machine. Also points to wrong directory.
**Prevention:** Replace with `pd.read_csv('./data.csv')`.
**Phase:** Code quality phase

### P-12: PennyLane API Drift Between Required and Installed Version
**Severity:** Low-Moderate
**Warning Signs:** Deprecation warnings or changed function signatures
**Description:** `requirements.txt` says `>=0.32.0` but 0.44.0 is installed. APIs may have changed between versions.
**Prevention:** Pin to `pennylane==0.44.0` in requirements.txt. Baseline the circuit before redesigning.
**Phase:** Should be verified during circuit redesign phase

### P-13: Duplicate Plotting Cells Produce Inconsistent Comparisons
**Severity:** Low
**Warning Signs:** Different cells showing different results for "the same" data
**Description:** After normalization fixes, duplicate cells may use different data pipeline states.
**Prevention:** Remove duplicate cells in the cleanup pass. Keep one canonical generation + visualization pipeline.
**Phase:** Code quality phase

### P-14: Early Stopping on Critic Loss Is Unreliable in WGAN
**Severity:** Moderate
**Warning Signs:** Early stopping triggers at a point where generated data quality is poor
**Description:** Critic loss (Wasserstein distance estimate) should stabilize, not minimize. It's not a direct measure of generation quality.
**Prevention:** Switch to EMD after P-7 is fixed (P-7 is a prerequisite). Consider composite metric.
**Phase:** ML theory fixes phase (after EMD fix)

### P-15: Memory Leaks from Quantum Circuit Tensors
**Severity:** Moderate
**Warning Signs:** Increasingly slow later epochs; OOM errors on long training runs
**Description:** Forward passes outside `torch.no_grad()` accumulate autograd graph tensors. Loss values stored as tensors retain entire computation graphs.
**Prevention:** Wrap all evaluation calls in `torch.no_grad()`. Use `.item()` before appending losses to lists.
**Phase:** Performance/correctness fixes phase

## Dependency Chain

Key ordering constraints for the remediation:
1. Code cleanup (dead code, debug artifacts) → before any structural changes
2. Normalize/scaling fixes → must be done together in one pass
3. Circuit redesign → must fix WINDOW_LENGTH computation first
4. EMD fix (P-7) → must come before early stopping change (P-14)
5. Hyperparameter restoration → must update training loop structure simultaneously
6. diff_method change → verify backend device explicitly

---
*Research completed: 2026-02-26*
