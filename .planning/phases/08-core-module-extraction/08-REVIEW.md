---
phase: 08-core-module-extraction
reviewed: 2026-04-27T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - revision/__init__.py
  - revision/core/__init__.py
  - revision/core/data.py
  - revision/core/eval.py
  - revision/core/models/__init__.py
  - revision/core/models/critic.py
  - revision/core/models/quantum.py
  - revision/core/training.py
  - revision/01_parity_check.ipynb
  - scripts/build_parity_notebook.py
findings:
  critical: 3
  warning: 8
  info: 7
  total: 18
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-04-27
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

The Phase 8 extraction generally preserves notebook behavior for the default code paths exercised by the parity check (default `spectral_loss_weight=0.0`, default `callback=None`, default `seed=42`). However, the **opt-in extension hooks contain real correctness bugs that will mislead any user who turns them on**, and several engineering choices weaken the safety of the package:

- `_spectral_psd_loss` does not compute the gradient of the PSD MSE — the autograd path it constructs only flows gradient through `fake_flat.var()`, so enabling `spectral_loss_weight > 0` does NOT optimize the PSD objective. This is BLOCKER-class because the docstring explicitly markets the function as "the v1.1 Phase 6 PSD penalty hook."
- The PSD hook also resamples a fresh batch of real data every call (out of sync with the critic's training batch), making the target stochastic per generator step.
- `EarlyStopping._load_checkpoint` reassigns `model.params_pqc.data = checkpoint["params_pqc"]` through the `_ESAdapter` — the adapter's `params_pqc` setter would replace the underlying `nn.Parameter` with a plain tensor and silently break gradient flow. The current load path uses `.data =` (the getter), so it works in the common case, but the setter path is a foot-gun that is wired to misbehave.
- Architectural invariants are guarded with `assert` statements (stripped under `python -O`).
- Multiple silent-ignore behaviors (`par_light` discarded, `callback` exceptions printed only) mask future bugs.

The parity-check notebook itself looks correct and should produce the locked tolerance verdict on the unconditioned default path. The bugs surface only when downstream phases (12, 13) start exercising the extension hooks.

## Critical Issues

### CR-01: Spectral PSD loss has no gradient path to the actual MSE objective

**File:** `revision/core/training.py:438-475`
**Severity:** BLOCKER

`_spectral_psd_loss` computes `mse` as a Python `float` from numpy arrays (line 468: `mse = float(np.mean(diff ** 2))`). It then "re-attaches" autograd by returning:

```python
return mse * fake_flat.var() / (fake_flat.var().detach() + eps)
```

This expression is numerically `≈ mse` but its gradient w.r.t. `params_pqc` is:

```
d/dθ [ mse * var(fake) / detach(var(fake)) ]
  = mse * d/dθ var(fake) / detach(var(fake))
```

The MSE is a **constant** w.r.t. θ in this expression — it carries no learning signal. So enabling `spectral_loss_weight > 0` does NOT minimize PSD divergence; it pushes the generator toward higher (or lower) sample variance scaled by a constant. This silently fails the v1.1 Phase 6 acceptance criterion.

The docstring acknowledges "The MSE itself is a constant w.r.t. params for this simplified hook" but that is the bug, not a feature — the function is named and documented as "MSE between log-PSDs" and "v1.1 Phase 6 hook" so callers will reasonably assume it actually optimizes that quantity.

**Fix:** Compute Welch on the torch tensor with autograd-friendly ops (use `torch.stft` or implement Welch via FFT in torch), e.g.:

```python
def _spectral_psd_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_flat = fake.reshape(-1)
    real_flat = real.reshape(-1).detach().to(fake_flat.dtype)
    # Use torch FFT so gradients flow.
    fake_psd = torch.abs(torch.fft.rfft(fake_flat)) ** 2
    real_psd = torch.abs(torch.fft.rfft(real_flat)) ** 2
    eps = 1e-12
    return torch.mean(
        (torch.log(fake_psd + eps) - torch.log(real_psd + eps)) ** 2
    )
```

Until a differentiable implementation lands, the safest interim is to raise `NotImplementedError` when `spectral_loss_weight > 0` so the silent failure mode is impossible.

---

### CR-02: `_ESAdapter.params_pqc` setter would replace the `nn.Parameter` with a raw tensor

**File:** `revision/core/training.py:421-429`
**Severity:** BLOCKER (latent — triggered by any code that does `adapter.params_pqc = X`)

The adapter exposes `params_pqc` as a `@property` whose setter is:

```python
@params_pqc.setter
def params_pqc(self, value):
    self._generator.params_pqc = value
```

`self._generator` is an `nn.Module`. Assigning a non-Parameter value to an attribute that was previously an `nn.Parameter` un-registers the parameter from the module. After such an assignment:
- `generator.parameters()` no longer yields `params_pqc`
- `g_optimizer.param_groups[0]["params"]` still references the *old* tensor (orphaned)
- `qnode(noise, generator.params_pqc)` runs against a non-leaf tensor with no `requires_grad`

`EarlyStopping._load_checkpoint` (line 166: `model.params_pqc.data = checkpoint["params_pqc"]`) currently uses the **getter** plus `.data` assignment, so the setter is not exercised on this code path. But the setter exists, claims to be "for safety," and would actually destroy the model. Any future caller that does `es_model.params_pqc = new_param` will silently corrupt training.

**Fix:** Either remove the setter entirely (raise `AttributeError`), or make it copy in place:

```python
@params_pqc.setter
def params_pqc(self, value):
    if isinstance(value, torch.nn.Parameter):
        # Replace through the module's parameter machinery
        self._generator.params_pqc = value
    else:
        # Copy into existing Parameter to preserve grad/optimizer wiring
        with torch.no_grad():
            self._generator.params_pqc.data.copy_(value)
```

---

### CR-03: `find_optimal_lambert_delta` will raise `RuntimeWarning` and could divide by zero on degenerate inputs

**File:** `revision/core/data.py:168-181`
**Severity:** BLOCKER (input-dependent crash / silent garbage)

The minimizer's objective is:

```python
sign = np.sign(data)
lr = lambertw(d * data ** 2).real
lr = np.maximum(lr, 0)
transformed = sign * np.sqrt(lr / d)
```

For `d → 0.01` (the lower bound) and a constant `data` array (e.g., a window with all-equal values), `np.std(data) == 0` upstream, `(data - mu)/sigma` produces `NaN`, and `lambertw(d * NaN)` returns `NaN`. `_sp_kurtosis(NaN_array)` returns `NaN`, which is not a valid scalar minimization objective and `minimize_scalar` may converge to garbage or raise.

Additionally, `lambertw(d * x**2)` for `x**2` very large overflows numpy double; the function silently returns `inf` and the kurtosis becomes `nan`.

There is no input validation in `load_and_preprocess` that the OD column has any variance, no NaN check after `normalize`, and no guard against the optimizer returning the boundary value (a sign that the minimum lies outside `[0.01, 2.0]`).

**Fix:** Validate inputs and bound the search:

```python
def find_optimal_lambert_delta(normed: np.ndarray) -> float:
    if not np.all(np.isfinite(normed)):
        raise ValueError("Lambert delta search input contains NaN/Inf")
    if np.std(normed) == 0:
        raise ValueError("Lambert delta search input has zero variance")
    # ...existing minimize_scalar...
    delta = float(result.x)
    if delta in (0.01, 2.0):
        import warnings
        warnings.warn(
            f"Lambert delta hit search boundary ({delta}); kurtosis "
            "minimum may lie outside [0.01, 2.0]."
        )
    return delta
```

---

## Warnings

### WR-01: Architectural invariants enforced by `assert` (stripped under `python -O`)

**File:** `revision/core/models/quantum.py:48-51`
**Issue:** `assert window_length == 2 * num_qubits` is a load-bearing invariant — violating it would silently produce mis-shaped tensors. `python -O` strips assertions. Same pattern in `scripts/build_parity_notebook.py:86, 97, 296, 297`.
**Fix:**
```python
if window_length != 2 * num_qubits:
    raise ValueError(
        f"window_length must equal 2 * num_qubits "
        f"(got window_length={window_length}, num_qubits={num_qubits})"
    )
```

---

### WR-02: PSD hook re-samples real data every generator step (target drifts within an epoch)

**File:** `revision/core/training.py:325-329, 432-435`
**Issue:** When `spectral_loss_weight > 0`, `real_log_returns_for_psd(gan_data_list, batch_size)` calls `torch.randint` to draw a fresh real batch — *different from the batch the critic just trained on*. This adds gratuitous noise to the spectral target and consumes from the same global RNG that drives critic batch selection, so flipping the spectral weight on/off changes the entire seeded sequence of training batches downstream.
**Fix:** Reuse the real batch from the most recent critic iteration:

```python
if spectral_loss_weight > 0.0:
    psd_penalty = _spectral_psd_loss(gen_out, real_log_returns)
    generator_loss = generator_loss + spectral_loss_weight * psd_penalty
```

Remove `real_log_returns_for_psd` entirely; the dedicated sampler is both buggy and unnecessary.

---

### WR-03: `device` argument to `compute_gradient_penalty` is misleading dead input

**File:** `revision/core/training.py:31-73`
**Issue:** The signature accepts `device: torch.device` but the function explicitly ignores it — line 55 uses `real_samples.device`. The comment on lines 52-53 acknowledges this. Callers may believe they are pinning the GP computation onto a particular device by passing `device=`, but in fact the placement is governed entirely by `real_samples`. This is an API trap.
**Fix:** Remove the parameter and update the call site:

```python
def compute_gradient_penalty(
    critic: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    ...
```

---

### WR-04: Eval block depends on `real_log_returns` leaking from the critic loop scope

**File:** `revision/core/training.py:336-355`
**Issue:** Line 352 reads `real_log_returns.reshape(-1).cpu().numpy()` to build the EMD comparison set. `real_log_returns` was last bound inside the critic loop at line 266. If `n_critic == 0` (degenerate but not impossible — there is no validation), `real_log_returns` is undefined and the eval block raises `NameError` only at the eval epoch. Even when `n_critic > 0`, the eval EMD compares against whatever batch the critic happened to draw last — a quietly biased baseline.
**Fix:** Sample an explicit eval batch:

```python
if epoch % eval_every == 0 or epoch + 1 == num_epochs:
    with torch.no_grad():
        eval_indices = torch.randint(0, len(gan_data_list), (batch_size,))
        eval_real = torch.stack([gan_data_list[idx] for idx in eval_indices])
        # ... rest of eval ...
        real_flat = eval_real.reshape(-1).cpu().numpy()
```

Also add `if n_critic < 1: raise ValueError(...)` at the top of `train_wgan_gp`.

---

### WR-05: `compute_jsd` produces NaN on degenerate constant inputs

**File:** `revision/core/eval.py:96-112`
**Issue:** When `real` and `fake` are both constant (or empty), `lo == hi`, `np.linspace(lo, hi, bins+1)` returns `bins+1` identical edges, `np.histogram` returns all-zero counts, and `rh / rh.sum()` evaluates `0 / 0 → NaN`. `jensenshannon([NaN,...], [NaN,...]) → NaN`. The notebook never hit this because real log-returns aren't constant, but the function is now part of a public API.
**Fix:** Guard explicitly:

```python
lo = min(real.min(), fake.min())
hi = max(real.max(), fake.max())
if hi == lo:
    return 0.0  # identical degenerate distributions
edges = np.linspace(lo, hi, bins + 1)
...
```

---

### WR-06: PQC parameter-count gating uses strict `<` and silently skips gates on tight budgets

**File:** `revision/core/models/quantum.py:130, 141, 159`
**Issue:** Three guard expressions:
- `if idx < len(params_pqc):` (Step 2 IQP)
- `if idx + 2 < len(params_pqc):` (Step 4 Rot block)
- `if idx + 1 < len(params_pqc):` (Step 5 RX/RY pair)

These mirror the notebook bug-for-bug, and at the canonical `(num_qubits=5, num_layers=4) → 75 params` shape they all pass cleanly. But the strict `<` comparisons are subtly wrong — they should be `idx + N <= len(params_pqc) - 1`, i.e., `idx + N < len(params_pqc)` is equivalent for accessing index `idx + N`. Wait — `params_pqc[idx + 2]` requires `idx + 2 < len(params_pqc)`, so `<` is *correct* for the 3-element Rot read. So this is OK as-is for the canonical shape; the warning is that the silent-skip behavior on misconfigured shapes will produce an arbitrary partially-applied circuit instead of a loud failure.
**Fix:** Replace the silent skips with a precondition check at construction time so misconfigured `num_params` cannot reach the QNode:

```python
expected = num_qubits + num_layers * (num_qubits * 3) + num_qubits * 2
if self.num_params != expected:
    raise ValueError(
        f"num_params={self.num_params} does not match circuit topology "
        f"(expected {expected})"
    )
```

Then drop the `if idx ...` guards inside `generator_circuit` — they hide misconfiguration today.

---

### WR-07: Callback exceptions are swallowed with no traceback

**File:** `revision/core/training.py:378-379`
**Issue:** `except Exception as exc: print(f"  [callback warning] {exc!r}")`. A bare `repr(exc)` loses the traceback — a Phase 13 introspection callback that throws will report something like `KeyError('emd')` with no indication of *where* in the callback code the error occurred. Bad debugging UX.
**Fix:**
```python
import traceback
except Exception as exc:
    print(f"  [callback warning] {exc!r}")
    traceback.print_exc()
```

---

### WR-08: `torch.load(..., weights_only=False)` is the unsafe code-execution path

**File:** `revision/core/training.py:165` and `scripts/build_parity_notebook.py:94, 239, 310`
**Issue:** `weights_only=False` allows the unpickler to instantiate arbitrary classes from the checkpoint. PyTorch flipped the safe default to `weights_only=True` precisely because checkpoints are an attack vector. Phase 8's checkpoints are local artifacts you authored, so the immediate risk is low — but the code is now part of an importable module that other people will reuse on checkpoints from elsewhere.
**Fix:** Where the checkpoint contains only tensors and primitive dicts (the case here), pass `weights_only=True`:

```python
checkpoint = torch.load(self.checkpoint_path, weights_only=True)
```

If the checkpoint contains optimizer state with custom classes, register them via `torch.serialization.add_safe_globals` rather than disabling the check globally.

---

## Info

### IN-01: `module-level` import of `revision.core` from `data.py` creates a circular-import risk

**File:** `revision/core/data.py:21`
**Issue:** `data.py` does `from revision.core import DITHER, DITHER_SEED, PAR_LIGHT_MAX, WINDOW_LENGTH`. `revision/core/__init__.py` in turn imports `data` (line 35: `from revision.core import data, eval, training`). The import currently works because the constants are defined *before* the submodule imports in `__init__.py`, but any reordering will break it. Same pattern in `training.py:225`.
**Fix:** Define the constants in a small `revision/core/constants.py` (no dependencies) and import from there in both `__init__.py` and the submodules.

---

### IN-02: `eval` shadows the Python builtin in the package namespace

**File:** `revision/core/__init__.py:35`, `revision/core/eval.py`
**Issue:** Documented in `eval.py:13-15` as a deliberate trade-off. Acceptable, but `revision.core.eval` will silently override the builtin if anyone does `from revision.core import *` (mitigated by `__all__`). Consider `metrics.py` instead — both shorter and unambiguous.

---

### IN-03: `_NOISE_HIGH_LITERAL = 4 * math.pi` is dead code

**File:** `revision/core/training.py:478-482`
**Issue:** A module-level constant added solely so a grep-based verification step finds the literal `4 * math.pi` in `training.py`. It is unused and the comment admits as much. Dead code now that the verification step is past.
**Fix:** Delete the constant and the comment.

---

### IN-04: `forward()` silently discards `par_light`

**File:** `revision/core/models/quantum.py:200-201`
**Issue:** `_ = par_light` documents the intent (forward-compat hook). When Phase 12+ wires conditioning back in, callers that *think* they passed conditioning will be silently ignored. Consider a one-time `warnings.warn` on first non-None `par_light` so accidental use of the dormant feature is visible.

---

### IN-05: Magic literal `0.1` for generator output scaling

**File:** `revision/core/training.py:283, 316, 350`
**Issue:** `gen_out * 0.1` appears three times with no name. It mirrors the notebook line `generated_samples = generated_samples.to(torch.float64) * 0.1`, but there is no constant `GEN_SCALE` used (despite `revision/core/__init__.py:22` defining `GEN_SCALE = 1.0` — note: 1.0, not 0.1, so the constant doesn't even match the actual scaling).
**Fix:** Either replace the literal with a module constant `GEN_OUTPUT_SCALE = 0.1` or reconcile the existing `GEN_SCALE` constant with the value actually used.

---

### IN-06: `compute_log_delta` `len(od_np)` works only because `od_np` was already converted

**File:** `revision/core/data.py:60`
**Issue:** `od_np = od_values.numpy() if isinstance(...) else od_values.copy()` — both branches return a numpy array, so `len(od_np)` is the array length. Fine. But the docstring should clarify that `od_values` may be either a torch tensor *or* a numpy array; the type hint says `torch.Tensor` only, which is misleading.

---

### IN-07: `kurt_avg.append(moments["kurtosis"])` records *fake-only* kurtosis

**File:** `revision/core/training.py:361`
**Issue:** The metric name `kurt_avg` suggests an averaged comparison, but the value stored is just `moments["kurtosis"]` of the *fake* batch. Compare to `compute_moments(real_flat)` to compute a real-vs-fake delta if that's the intent. If the stored value is just a tracking statistic, rename to `fake_kurt`.
**Fix:** Either compute and store both real and fake kurtosis, or rename the field for clarity.

---

_Reviewed: 2026-04-27_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
