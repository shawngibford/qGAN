# Phase 4: Code Regression Fixes - Research

**Researched:** 2026-03-13
**Domain:** Jupyter notebook code fixes (noise range, PAR_LIGHT eval, variable shadowing) + validation run
**Confidence:** HIGH

## Summary

Phase 4 addresses three code regressions in `qgan_pennylane.ipynb` that were introduced during the PAR_LIGHT conditioning work (v1.0 Phase 2), plus a 200-epoch validation run to confirm HPO hyperparameters transfer to the corrected code. All three bugs are well-localized in the notebook -- the noise range fix is a literal string replacement at 3 locations in cell 26 plus 1 location in cell 45, the par_zeros eval fix requires replacing a `torch.zeros` call with real PAR_LIGHT sampling in the eval section of cell 26, and the mu/sigma shadowing requires inlining values in cell 12's `norm.pdf()` call. Additionally, the ACF loss code must be entirely removed (not just zeroed) per user decision, and a validation run with HPO parameters must be executed with results saved to JSON.

This phase is low-risk mechanically but the validation run is the key unknown -- HPO hyperparameters were tuned on the regressed [0, 2pi] noise range, and switching to [0, 4pi] changes the generator's input distribution. The user has defined a clear failure path: proceed regardless but flag for HPO re-run if EMD exceeds 2x baseline (0.002274).

**Primary recommendation:** Fix all three bugs in a single cell-26 edit pass, remove ACF loss entirely, update cell 28 hyperparameters to HPO values, then run validation. Keep fixes and validation in separate plans so the code changes can be verified before committing to a long training run.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- If EMD > 2x HPO baseline (0.002274): proceed to Phase 5 but flag for HPO re-run after Phase 5/6
- If training diverges (NaN loss or critic loss > 1000): auto-fallback to v1.0 defaults (lambda_gp=10, n_critic=5) and retry
- Accept Phase 4 EMD as interim baseline regardless of whether it meets threshold
- Validation run is 200 epochs (not shortened)
- Claude executes the validation run (not manual)
- Baseline metrics to capture: EMD, moment statistics (mean, std, kurtosis), spectral profile (PSD comparison), training dynamics (loss curves, gradient norms, epoch timing)
- Storage: print in notebook output AND save to JSON file
- JSON includes full config (HPO params, noise range, epoch count, timestamp, git hash) for reproducibility
- PSD depth: Claude's discretion
- Primary attempt uses HPO-tuned values: lr_g=0.003, lr_c=0.0002, lambda_gp=2.16, n_critic=9
- ACF loss disabled: lambda_acf=0 (not just zeroed -- remove ACF loss code entirely)
- Other HPO params (LRs, lambda_gp, n_critic) kept as-is to isolate the noise range variable
- Auto-fallback to v1.0 defaults if NaN/divergence occurs

### Claude's Discretion
- PSD baseline depth (summary stat vs full per-frequency arrays)
- Exact validation cell structure and output formatting
- Error state handling during validation run
- mu/sigma fix implementation (PROJECT.md suggests inline into norm.pdf() calls)

### Deferred Ideas (OUT OF SCOPE)
- HPO re-run on corrected code (after Phase 5/6)
- ACF loss replacement with spectral loss (Phase 6)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REG-01 | Training loop uses correct noise range [0, 4pi] in all 3 locations (critic training, generator training, evaluation) | Three exact locations identified in cell 26 (lines 321, 398, 479) plus cell 45 (line 7) and cell 29 (line 4 - cosmetic). Fix is literal `2 * np.pi` -> `4 * np.pi` replacement. |
| REG-04 | Evaluation generation uses real PAR_LIGHT values instead of `torch.zeros` for conditioning | Bug at cell 26 line 488. Fix: sample from `par_data_list` (already available in `_train_one_epoch` scope), compress via reshape+mean pattern (same as lines 324-327), remap to [0,1]. |
| REG-05 | mu/sigma variable shadowing eliminated in plotting cells | Cell 12 (lines 15-16) sets `mu`/`sigma` as numpy scalars; cell 15 (line 1) sets them as torch tensors. Fix: inline `np.mean(log_delta_np)` and `np.std(log_delta_np)` directly into `norm.pdf()` call in cell 12 line 22. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PennyLane | 0.44.0 | Quantum circuit simulation | Project constraint -- installed in qgan_env |
| PyTorch | 2.8.0 | Classical neural network + autograd | Project constraint -- installed in qgan_env |
| SciPy | (system) | `wasserstein_distance` for EMD | Already used for EMD evaluation |
| NumPy | (system) | Noise sampling, array operations | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.fft` | (built-in) | PSD computation for baseline metrics | Validation baseline capture |
| `json` | (stdlib) | Save validation results to JSON | Validation output |
| `subprocess` | (stdlib) | Capture git hash for reproducibility | Validation JSON metadata |
| `time` | (stdlib) | Epoch timing for training dynamics | Already used in training |

### Alternatives Considered
None -- all fixes are within existing stack. No new dependencies needed.

## Architecture Patterns

### Notebook Cell Organization

The notebook follows a linear execution pattern with hyperparameters defined in cell 28. All code changes for Phase 4 affect:

```
Cell 12  - Data analysis (mu/sigma shadowing fix)
Cell 26  - qGAN class definition (noise range, par_zeros, ACF removal)
Cell 28  - Hyperparameter configuration (HPO values, remove LAMBDA_ACF)
Cell 29  - Circuit diagram (cosmetic noise range fix)
Cell 45  - Standalone generation (noise range fix)
New cell - Validation run + JSON output
```

### Pattern 1: Noise Sampling (Current and Fixed)

**What:** Generator noise is sampled uniformly and fed to the quantum circuit's RZ encoding layer.
**When to use:** Every location that creates noise for generator input.

Current (WRONG -- 3 training locations + cell 45):
```python
noise_values = np.random.uniform(0, 2 * np.pi, size=self.num_qubits)
```

Fixed:
```python
noise_values = np.random.uniform(0, 4 * np.pi, size=self.num_qubits)
```

The 4pi range is correct because the circuit uses RZ encoding gates which have periodicity of 4pi (RZ rotation has period 4pi due to the half-angle formula: RZ(theta) = exp(-i*theta/2 * Z)). The v1.0 circuit redesign intentionally expanded the range to exploit the full period, but the training loop was never updated.

### Pattern 2: PAR_LIGHT Compression for Circuit Input

**What:** PAR_LIGHT windows of length 10 are compressed to 5 values (one per qubit) for the circuit's RY encoding.
**When to use:** Every location that feeds PAR_LIGHT to the generator.

The pattern already exists in critic training (cell 26, lines 324-327):
```python
# Compress PAR_LIGHT window (10 values -> 5) for circuit input
par_for_circuit = par_window.reshape(self.num_qubits, 2).mean(dim=1).float()
# Remap from [-1,1] to [0,1] for RY encoding
par_for_circuit = (par_for_circuit + 1.0) / 2.0
```

The eval section (line 488) must use this same pattern instead of `torch.zeros`.

### Pattern 3: Variable Shadowing Fix (Inline Values)

**What:** Cell 12 creates `mu` and `sigma` as numpy scalars for a Gaussian overlay plot. Cell 15 creates `mu` and `sigma` as torch tensors from `normalize()`. If cell 12 is re-executed after cell 15, the torch tensors get overwritten.
**Fix:** Inline the computation into the `norm.pdf()` call in cell 12:

Current (cell 12, lines 15-16, 22):
```python
mu = np.mean(log_delta_np)
sigma = np.std(log_delta_np)
# ... later ...
pdf = norm.pdf(x, mu, sigma)
```

Fixed (remove lines 15-16, update line 22):
```python
pdf = norm.pdf(x, np.mean(log_delta_np), np.std(log_delta_np))
```

Also update the print statements at lines 45-46 to use `np.mean(log_delta_np)` and `np.std(log_delta_np)` directly, since they currently reference the removed `mu` and `sigma` variables.

### Pattern 4: ACF Loss Removal

**What:** Remove ACF loss computation entirely from the generator loss, not just zero the weight.
**Scope of removal:**
1. **`__init__` parameter** (line 2): Remove `lambda_acf=0.1` from constructor signature
2. **`self.lambda_acf` assignment** (line 12): Remove
3. **`self.acf_avg` list init** (line 68): Keep -- this tracks the eval ACF RMSE metric, not the loss
4. **`diff_acf_lag1` static method** (lines 104-114): Remove entirely
5. **ACF penalty block** (lines 443-457): Remove the real window sampling, per-window ACF computation, and penalty stacking
6. **Combined loss line** (line 460): Change `generator_loss = generator_loss_wgan + self.lambda_acf * acf_penalty` to `generator_loss = generator_loss_wgan`
7. **Cell 28 `LAMBDA_ACF`** (line 11): Remove from hyperparameter cell
8. **Cell 28 `qGAN()` constructor call** (line 60): Remove `lambda_acf=LAMBDA_ACF` argument
9. **Cell 40 HPO retrain**: Remove `lambda_acf` references

NOTE: The `self.acf_avg` list (line 68) and its population at line 506 should be KEPT. This tracks the ACF RMSE evaluation metric (computed via `stylized_facts()`), which is different from the training ACF penalty loss. The stylized facts evaluation remains useful for monitoring.

### Anti-Patterns to Avoid

- **Changing GEN_SCALE to fix the 0.1 scaling mismatch:** The training loop hardcodes `* 0.1` (lines 340, 428, 492) while cell 45 uses `* GEN_SCALE` (= 1.0). This is a real discrepancy but it is NOT in Phase 4 scope. Fixing it now would invalidate the HPO baseline. Note it and move on.
- **Fixing cell 29 noise range "bug":** Cell 29 uses `2 * np.pi` for dummy circuit visualization. This doesn't affect training but should be updated to `4 * np.pi` for consistency. It is cosmetic.
- **Modifying the eval denormalization pipeline:** The eval section (lines 500-502) uses `rescale` + `lambert_w_transform` instead of `full_denorm_pipeline`. This is a pre-existing inconsistency from v1.0. Do not change it in Phase 4 -- it works correctly and changing it risks introducing new bugs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PSD computation | Custom FFT wrapper | `torch.fft.rfft` on 1D signal | Handles windowing, normalization correctly |
| EMD calculation | Custom Wasserstein | `scipy.stats.wasserstein_distance` | Already used, handles edge cases |
| JSON serialization of tensors | Manual conversion | `float(tensor.item())` for scalars, `.tolist()` for arrays | Standard PyTorch-to-JSON pattern |
| Git hash capture | Manual git parsing | `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` | Reliable, standard approach |

## Common Pitfalls

### Pitfall 1: HPO Parameters Were Tuned on Wrong Noise Range
**What goes wrong:** The HPO study (best_params.json) was run with [0, 2pi] noise. Switching to [0, 4pi] changes the generator's input manifold. Learning rates and gradient penalty that worked for [0, 2pi] may not work for [0, 4pi].
**Why it happens:** The noise range regression predates HPO -- HPO optimized around the wrong baseline.
**How to avoid:** The user has defined an explicit failure path: if EMD > 0.002274, proceed but flag for HPO re-run. If NaN or critic loss > 1000, fallback to v1.0 defaults (lambda_gp=10, n_critic=5).
**Warning signs:** NaN in loss, critic loss exploding, EMD increasing monotonically.

### Pitfall 2: PAR_LIGHT Eval Fix Changes EMD Baseline
**What goes wrong:** The par_zeros eval bug means all prior EMD values reflect unconditioned generation. Fixing it means the new EMD is not directly comparable to HPO baseline.
**Why it happens:** With zeros, the generator ignores conditioning. With real PAR_LIGHT, it conditions on actual data, potentially producing better (or different) distributions.
**How to avoid:** Accept this is a known confound. Capture both the conditioned EMD (new) and document the comparison caveat in the validation JSON.
**Warning signs:** EMD significantly different from HPO baseline even if training looks healthy.

### Pitfall 3: mu/sigma Fix Must Also Update Print Statements
**What goes wrong:** If you inline `mu`/`sigma` into `norm.pdf()` but forget to update the print statements at lines 45-46 of cell 12, you get a NameError on re-execution.
**Why it happens:** The print statements reference `mu` and `sigma` by name.
**How to avoid:** Update all references in cell 12 -- both the `norm.pdf()` call AND the print statements.

### Pitfall 4: ACF Removal Must Be Complete
**What goes wrong:** If `lambda_acf` remains in the constructor but `acf_penalty` is removed, the constructor fails. If `diff_acf_lag1` is removed but still referenced, AttributeError.
**Why it happens:** ACF code touches multiple locations: constructor, method, training loop, hyperparameter cell, HPO retrain cell.
**How to avoid:** Systematic removal: constructor param, self assignment, static method, training block, combined loss line, cell 28 config, cell 40 retrain. Keep `self.acf_avg` (eval metric tracking, not loss).

### Pitfall 5: Validation Cell Must Create Fresh qGAN Instance
**What goes wrong:** Running validation in the same qGAN instance that was used for testing means optimizer state, loss histories, and params_pqc carry over.
**Why it happens:** Notebook cells share global state.
**How to avoid:** The validation cell must create a fresh `qGAN()` instance with HPO parameters, fresh optimizers, and fresh early stopping. Pattern: follow cell 40's approach.

### Pitfall 6: Generator Output Scaling Discrepancy
**What goes wrong:** Training loop uses hardcoded `* 0.1` scaling (cell 26 lines 340, 428, 492) but cell 45 standalone generation uses `* GEN_SCALE` which is `1.0`. This means standalone generation produces 10x the training-time scale.
**Why it happens:** `GEN_SCALE` was set to 1.0 in cell 28 but the training loop was never updated to use it.
**How to avoid:** This is OUT OF SCOPE for Phase 4. Do not change either the training loop or `GEN_SCALE` in this phase -- the HPO baseline was tuned with the `* 0.1` scaling in training, and changing it would invalidate comparisons. Flag this for a future fix. The validation run should use the training loop's built-in eval (which uses `* 0.1`), not cell 45.

## Code Examples

### Fix 1: Noise Range (cell 26, 3 locations)

```python
# Line 321 (critic training) - change:
noise_values = np.random.uniform(0, 2 * np.pi, size=self.num_qubits)
# to:
noise_values = np.random.uniform(0, 4 * np.pi, size=self.num_qubits)

# Line 398 (generator training) - same change
# Line 479 (evaluation) - same change
```

### Fix 2: PAR_LIGHT Eval (cell 26, replace lines 487-489)

Current:
```python
for j, generator_input in enumerate(generator_inputs):
    par_zeros = torch.zeros(self.num_qubits, dtype=torch.float32)
    gen_out = self.generator(generator_input, par_zeros, self.params_pqc)
```

Fixed:
```python
for j, generator_input in enumerate(generator_inputs):
    # Use real PAR_LIGHT from dataset (same pattern as training)
    random_idx = torch.randint(0, len(par_data_list), (1,)).item()
    par_window = par_data_list[random_idx]
    par_for_circuit = par_window.reshape(self.num_qubits, 2).mean(dim=0).float()
    par_for_circuit = (par_for_circuit + 1.0) / 2.0
    gen_out = self.generator(generator_input, par_for_circuit, self.params_pqc)
```

**IMPORTANT dimension note:** In the critic training loop (line 325), `par_window` has shape `(1, WINDOW_LENGTH)` due to the reshape at line 312, so `.reshape(self.num_qubits, 2)` works correctly. In the eval fix, `par_window` comes directly from `par_data_list` with shape `(WINDOW_LENGTH,)`, so the reshape also works directly. Verify the actual tensor shape during implementation.

### Fix 3: mu/sigma Shadowing (cell 12)

```python
# Remove lines 15-16 (mu = ..., sigma = ...)
# Change line 22 from:
pdf = norm.pdf(x, mu, sigma)
# to:
pdf = norm.pdf(x, np.mean(log_delta_np), np.std(log_delta_np))

# Change lines 45-46 from:
print(f"Mean: {mu:.4f}")
print(f"Standard Deviation: {sigma:.4f}")
# to:
print(f"Mean: {np.mean(log_delta_np):.4f}")
print(f"Standard Deviation: {np.std(log_delta_np):.4f}")
```

### Fix 4: ACF Loss Removal (cell 26, multiple locations)

```python
# __init__ signature: remove lambda_acf parameter
def __init__(self, num_epochs, batch_size, window_length, n_critic, gp, num_layers, num_qubits, delta=1):

# Remove self.lambda_acf assignment (line 12)
# Remove diff_acf_lag1 static method (lines 104-114)

# Generator training (remove lines 443-457, simplify line 460):
generator_loss = -torch.mean(fake_scores)  # Pure WGAN loss
```

### Fix 5: Noise Range in Cell 45 (standalone generation)

```python
# Change line 6-8 from:
noise = torch.tensor(
    np.random.uniform(0, 2 * np.pi, size=(qgan.num_qubits, num_samples)),
    dtype=torch.float32
)
# to:
noise = torch.tensor(
    np.random.uniform(0, 4 * np.pi, size=(qgan.num_qubits, num_samples)),
    dtype=torch.float32
)
```

### Validation Cell Structure (new cell)

```python
# ── Phase 4 Validation Run (200 epochs) ──
import json as json_mod
import subprocess
import time

# HPO-tuned parameters (from best_params.json)
VAL_LR_CRITIC = 1.8046e-05
VAL_LR_GEN = 6.9173e-05  # lr_critic * lr_gen_ratio (3.833)
VAL_LAMBDA_GP = 2.16
VAL_N_CRITIC = 9
VAL_EPOCHS = 200

# Create fresh model (no lambda_acf -- removed)
qgan_val = qGAN(
    num_epochs=VAL_EPOCHS,
    batch_size=BATCH_SIZE,
    window_length=WINDOW_LENGTH,
    n_critic=VAL_N_CRITIC,
    gp=VAL_LAMBDA_GP,
    num_layers=NUM_LAYERS,
    num_qubits=NUM_QUBITS,
    delta=delta,
)

c_opt = torch.optim.Adam(qgan_val.critic.parameters(), lr=VAL_LR_CRITIC, betas=(0.0, 0.9))
g_opt = torch.optim.Adam([qgan_val.params_pqc], lr=VAL_LR_GEN, betas=(0.0, 0.9))
qgan_val.compile_QGAN(c_opt, g_opt)

early_stopper_val = EarlyStopping(
    patience=50,
    warmup_epochs=50,
    checkpoint_path='results/phase4_validation_checkpoint.pt',
)

start = time.time()
qgan_val.train_qgan(dataloader, log_delta, transformed_norm_log_delta, num_elements, early_stopper=early_stopper_val)
elapsed = time.time() - start

# ... capture metrics, save JSON ...
```

### Validation JSON Structure

```python
validation_results = {
    "phase": "4-code-regression-fixes",
    "timestamp": datetime.now().isoformat(),
    "git_hash": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    "config": {
        "noise_range": [0, "4*pi"],
        "lr_critic": VAL_LR_CRITIC,
        "lr_generator": VAL_LR_GEN,
        "lambda_gp": VAL_LAMBDA_GP,
        "n_critic": VAL_N_CRITIC,
        "lambda_acf": 0,  # removed
        "epochs": VAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_qubits": NUM_QUBITS,
        "num_layers": NUM_LAYERS,
        "window_length": WINDOW_LENGTH,
    },
    "metrics": {
        "emd": float(...),
        "moments": {
            "real": {"mean": ..., "std": ..., "kurtosis": ...},
            "fake": {"mean": ..., "std": ..., "kurtosis": ...},
        },
        "psd": { ... },  # Summary or full arrays at Claude's discretion
        "training_dynamics": {
            "final_critic_loss": ...,
            "final_generator_loss": ...,
            "elapsed_seconds": elapsed,
            "epochs_completed": ...,
        },
    },
    "hpo_baseline": {
        "best_emd": 0.001137,
        "threshold_2x": 0.002274,
    },
    "outcome": "PASS" or "FLAG_FOR_HPO_RERUN" or "FALLBACK_USED",
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| [0, 2pi] noise range | [0, 4pi] noise range | v1.0 circuit redesign (Phase 2) | Training loop never updated -- 3 locations still use 2pi |
| `par_zeros` in eval | Real PAR_LIGHT conditioning | Should have been from Phase 2 | Eval metrics reflect unconditioned generation, masking conditioning issues |
| ACF loss as temporal penalty | Will be replaced by spectral/PSD loss | Phase 6 | ACF targets only lag-1; PSD targets full frequency spectrum |

**Deprecated/outdated:**
- `lambda_acf` parameter: Being removed in this phase. Phase 6 spectral loss replaces it.
- `diff_acf_lag1` method: Being removed. Not needed once ACF penalty is dropped from generator loss.

## Discovered Issues (Not in Phase 4 Scope)

1. **GEN_SCALE vs hardcoded 0.1:** Training loop hardcodes `* 0.1` output scaling (cell 26 lines 340, 428, 492) but cell 45 standalone generation uses `* GEN_SCALE` which is `1.0`. This means standalone generation produces output at 10x training scale. The `full_denorm_pipeline` in cell 45 may compensate but the inconsistency should be resolved in a future phase.

2. **Cell 45 noise shape:** Cell 45 creates noise with shape `(num_qubits, num_samples)` for broadcasting, which assumes the generator supports broadcasting. Phase 5 will restore broadcasting -- until then, cell 45 may need to be updated or used differently.

## Open Questions

1. **HPO Parameter Transfer**
   - What we know: HPO was tuned on [0, 2pi] noise. The best_params.json has lr_critic=1.8e-5, lr_gen=6.9e-5, lambda_gp=2.16, n_critic=9
   - What's unclear: Whether these parameters produce stable training with [0, 4pi] noise. Wider noise range means larger gradients through the RZ gates, which may require lower learning rates
   - Recommendation: Run the 200-epoch validation as specified. The auto-fallback to v1.0 defaults handles divergence. User has accepted that Phase 4 EMD is an interim baseline regardless

2. **PSD Baseline Depth**
   - What we know: User wants PSD comparison between real and fake as a pre-Phase-6 baseline
   - What's unclear: Whether to store full per-frequency arrays or just summary stats
   - Recommendation: Store both -- summary stats (total power ratio, peak frequency match) plus the full PSD arrays. The arrays are tiny (T=10 means only 6 frequency bins via rfft). Storage cost is negligible and downstream Phase 6 can use them for direct comparison

3. **Eval Denormalization Path**
   - What we know: The training eval (cell 26 lines 500-502) uses `rescale` + `lambert_w_transform` but skips the final `denormalize` step. Cell 45 uses `full_denorm_pipeline` which includes `denormalize`
   - What's unclear: Whether the training eval intentionally operates in a different space
   - Recommendation: Do NOT change the training eval denormalization in Phase 4. It has been this way since v1.0 and the HPO EMD baseline was computed using this path. Changing it would make the validation EMD incomparable

## Sources

### Primary (HIGH confidence)
- `qgan_pennylane.ipynb` cell 26 (lines 321, 398, 479, 488) -- direct code inspection of bug locations
- `qgan_pennylane.ipynb` cell 12 (lines 15-16, 22, 45-46) -- mu/sigma shadowing locations
- `qgan_pennylane.ipynb` cell 28 -- current hyperparameter configuration
- `qgan_pennylane.ipynb` cell 45 -- standalone generation noise range
- `results/hpo/best_params.json` -- HPO best parameters (lr_critic=1.8e-5, lr_gen_ratio=3.83, lambda_gp=2.16, n_critic=9, lambda_acf=0.062, best_emd=0.001137)
- `.planning/phases/04-code-regression-fixes/04-CONTEXT.md` -- user decisions and constraints

### Secondary (MEDIUM confidence)
- `.planning/PROJECT.md` -- project context, known tech debt, key decisions
- `post_hpo.md` -- post-HPO analysis findings (variance collapse, broadcasting regression)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, all fixes within existing code
- Architecture: HIGH - all bug locations precisely identified with line numbers
- Pitfalls: HIGH - failure modes well-characterized by user decisions and HPO history
- Validation run outcome: LOW - unknown whether HPO params transfer to [0, 4pi]

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable -- notebook code, not evolving libraries)
