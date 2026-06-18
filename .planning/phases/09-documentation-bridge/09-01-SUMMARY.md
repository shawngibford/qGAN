---
phase: 09-documentation-bridge
plan: 01
subsystem: core
tags: [pytorch, autograd, scipy, lambert-w, differentiable-transform, eval-06]

# Dependency graph
requires:
  - phase: 08-core-module-extraction
    provides: core/data.py with inverse_lambert_w_transform at Phase-8 parity baseline = 0.0
provides:
  - Differentiable inverse_lambert_w_transform via torch.autograd.Function
  - _InverseLambertW class with closed-form backward (dW/dz = W/(z·(1+W)))
  - Forward path bit-identical to scipy-only baseline (Phase 8 parity preserved)
  - Gradient flow through inverse Lambert W for OD-scale optimization
affects: [09-05-roundtrip, 11-tstr, 12-noise-gradients, 14-paper-methods]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "torch.autograd.Function wrapping non-torch (scipy) forward kernel"
    - "Closed-form analytic backward via implicit function theorem"
    - "torch.where mask for division-by-zero edge cases with analytic limit value"
    - "Caller-dtype preservation across autograd boundary (grad_data.to(grad_output.dtype))"

key-files:
  created: []
  modified:
    - core/data.py

key-decisions:
  - "D-03 honored: in-place replacement of inverse_lambert_w_transform (no parallel function)"
  - "D-05 honored: scipy.special.lambertw confined to forward path; backward is pure torch"
  - "D-07 honored: no symbol renames; public wrapper signature preserved verbatim"
  - "Closed-form derivative dW/dz = W/(z·(1+W)) (Corless et al. 1996, IFT on principal branch); chained via implicit diff of out² = W(δ·x²)/δ to get d(out)/dx = W/(out·δ·x·(1+W))"
  - "Edge case x≈0 masked with analytic limit d(out)/dx = 1 (verified: out ≈ x near zero by Taylor)"

patterns-established:
  - "torch.autograd.Function for non-torch numerical kernels (first such class in repo)"
  - "Bit-identical forward preservation idiom: legacy scipy body kept verbatim inside Function.forward"
  - "ctx.save_for_backward(tensors) + ctx.delta = delta (non-tensor) split (Pitfall 6)"

requirements-completed: [EVAL-06]

# Metrics
duration: 26min
completed: 2026-05-15
---

# Phase 09 Plan 01: Differentiable inverse Lambert W via torch.autograd.Function Summary

**Replaced `core/data.py::inverse_lambert_w_transform` with a custom `torch.autograd.Function` (`_InverseLambertW`) — forward path is bit-identical to the legacy scipy-only code (Phase 8 parity = 0.0 preserved verbatim), backward path is pure torch using the closed-form identity `dW/dz = W/(z·(1+W))` (Corless et al. 1996), delivering EVAL-06.**

## Performance

- **Duration:** ~26 min
- **Started:** 2026-05-15T15:34:00Z (approx; plan start)
- **Completed:** 2026-05-15T16:00:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added private `class _InverseLambertW(torch.autograd.Function)` immediately under the existing "Cell 17 — Lambert W transforms" banner in `core/data.py`, preserving the section structure.
- Forward path is a **verbatim** preservation of the legacy code at the old `data.py:80-86` (`data.double()` → `sign` → `data²` → `scipy.special.lambertw(.real)` → `torch.tensor(..., dtype=float64, device=data.device)` → `sign * sqrt(W/δ)`). Verified bit-identical against a pre-edit baseline: `max_abs_diff = 0.0` on a 20-element float64 randn sample.
- Backward path is pure torch on `ctx.saved_tensors = (data64, W, out)` with `ctx.delta` for the non-tensor argument; `scipy` is **not** called in backward (D-05 honored).
- Zero-input safety: `torch.where(data64.abs() < 1e-300, ones_like(data64), W / safe_denom)` returns the analytic limit value `1` at `x = 0`, eliminating 0/0 NaN gradients.
- Caller dtype preserved across the autograd boundary: float32 input → float32 gradient (`grad_data.to(grad_output.dtype)`).
- Device preserved across the numpy/scipy boundary (CPU smoke verified; MPS/CUDA path uses `device=data64.device` after `torch.tensor(lambert_result, ...)`).
- Module-level docstring updated to remove the stale "non-differentiable" claim (Rule 2 cleanup — see Deviations).
- `load_and_preprocess('./data.csv')` end-to-end smoke passed unchanged: OD=778, log_delta=777, windows=384, δ=0.146935.

## Forward / Backward Final Shape

**`_InverseLambertW.forward(ctx, data, delta)`**:

```python
data64 = data.double()
sign = torch.sign(data64)
data_squared = data64 ** 2
lambert_input = (delta * data_squared).cpu().numpy()
lambert_result = lambertw(lambert_input).real
lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data64.device)
out = sign * torch.sqrt(lambert_tensor / delta)
ctx.save_for_backward(data64, lambert_tensor, out)
ctx.delta = delta
return out
```

**`_InverseLambertW.backward(ctx, grad_output)`**:

```python
data64, W, out = ctx.saved_tensors
delta = ctx.delta
denom = out * delta * data64 * (1.0 + W)
zero_mask = data64.abs() < 1e-300
safe_denom = torch.where(zero_mask, torch.ones_like(denom), denom)
deriv = torch.where(zero_mask, torch.ones_like(data64), W / safe_denom)
grad_data = grad_output * deriv
return grad_data.to(grad_output.dtype), None
```

**Public wrapper** (signature preserved verbatim per D-07):

```python
def inverse_lambert_w_transform(data: torch.Tensor, delta: float) -> torch.Tensor:
    return _InverseLambertW.apply(data, delta)
```

## Closed-form Derivative — Derivation

Let `out = sign(x) · sqrt(W(δ·x²)/δ)` and `z = δ·x²`, `w = W(z)`. Squaring both sides:
`out² = w/δ`. Differentiating implicitly with respect to `x`:

```
2·out · d(out)/dx = (1/δ) · dw/dz · dz/dx
                  = (1/δ) · (w / (z·(1+w))) · 2δx           [Lambert W identity, principal branch]
                  = 2x·w / (z·(1+w))
                  = 2x·w / (δ·x²·(1+w))
                  = 2·w / (δ·x·(1+w))
```

Therefore `d(out)/dx = w / (out · δ · x · (1+w))`. The Lambert W identity `dW/dz = W/(z·(1+W))` for `z ≠ 0` is the standard implicit-function-theorem result on the principal branch (Corless et al. 1996, "On the Lambert W function"). At `x = 0`: `out = w = 0`, the form is `0/0`; Taylor-expanding `out` near zero gives `out ≈ sign(x)·|x| = x`, so the analytic limit is `d(out)/dx → 1`. We mask with `torch.where(|x| < 1e-300, 1, …)` to return this limit value cleanly.

## gradcheck Command + Result

```python
torch.autograd.gradcheck(
    lambda v: inverse_lambert_w_transform(v, 0.146932),
    (torch.randn(20, dtype=torch.float64, requires_grad=True),),
    eps=1e-6, atol=1e-6,
)
# Returns True (verified locally on PyTorch 2.9.0, scipy 1.16.2)
```

`gradcheck` returned `True` — the closed-form analytic backward matches central-difference numerical Jacobian within `atol=1e-6` on a 20-element float64 sample.

## Forward Parity vs Legacy Implementation

Pre-edit baseline captured by running the legacy scipy-only `inverse_lambert_w_transform` on `torch.manual_seed(42); x = torch.randn(20, dtype=torch.float64); delta = 0.146932`. After the rewrite, the same call returned a tensor where:

```
max_abs_diff(new_out, baseline_out) == 0.0
```

Sample values (post-rewrite, first 5 elements; identical to pre-edit):

| i | x[i] | out[i] |
|---|------|--------|
| 0 | 0.33669035438385886 | 0.3339431913775438 |
| 1 | 0.1288094051365897 | 0.12865287105311274 |
| 2 | 0.23446236336153173 | 0.23352489585896544 |
| 3 | 0.2303330279146119 | 0.22944391658955976 |
| 4 | -1.1228563767381703 | -1.03748435945593 |

Phase 8 parity baseline (EMD delta = 0.0, moments delta = 0.0) is preserved bit-identically — the new code path differs from the legacy path only in that it routes through `Function.apply`, which is a no-op on the forward output.

## Acceptance Criteria Gates (all pass)

| Gate | Expected | Actual |
|---|---|---|
| `grep -c 'class _InverseLambertW(torch.autograd.Function)' core/data.py` | 1 | 1 |
| `grep -c 'return _InverseLambertW.apply(data, delta)' core/data.py` | 1 | 1 |
| `grep -c 'def inverse_lambert_w_transform' core/data.py` | 1 | 1 |
| `grep -c 'ctx.save_for_backward' core/data.py` | ≥ 1 | 1 |
| `grep -c 'Non-differentiable' core/data.py` | 0 | 0 |
| `grep -c 'torch.where' core/data.py` | ≥ 1 | 3 |
| No new third-party imports | — | None added |
| `gradcheck` exits 0 | True | True |
| `import revision.core.data` clean (no warnings) | OK | OK |

## Task Commits

1. **Task 1: Implement `_InverseLambertW` autograd Function with closed-form backward** — `e702bd4` (feat)

## Files Created/Modified

- `core/data.py` — Inserted `class _InverseLambertW(torch.autograd.Function)` under the existing "Cell 17 — Lambert W transforms" banner; rewrote `def inverse_lambert_w_transform` body to a one-line dispatch `return _InverseLambertW.apply(data, delta)`; updated public-wrapper docstring (removed "Non-differentiable" note, added closed-form-derivative reference); updated module-level docstring to drop the stale "Phase 9 will replace... non-differentiable" forward reference (Rule 2 deviation — documentation now matches code).

## Decisions Made

- **Form of backward chosen:** `d(out)/dx = W / (out · δ · x · (1+W))` (from implicit differentiation of `out² = W/δ`). The alternative algebraic simplification `d(out)/dx = (out/x)·(1/(1+W))` was considered but rejected: it would shift the 0/0 singularity from `x·out` to `x` alone, which is functionally equivalent and adds no numerical benefit since the `zero_mask` already handles the limit. The chosen form is structurally explicit and matches the derivation in 09-RESEARCH.md Pattern 1.
- **Zero-mask threshold = 1e-300** (well below float64 `tiny ≈ 2.225e-308`): catches genuine zeros without false-triggering on small-but-finite inputs (where the closed-form is numerically well-behaved). Single `zero_mask` reused for both `safe_denom` and `deriv` `torch.where` calls — prevents the masked-out lane from producing a NaN that propagates through autograd graph metadata.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Documentation Correctness] Module docstring updated to remove stale "non-differentiable" claim**
- **Found during:** Task 1 (post-implementation file scan)
- **Issue:** The plan's `<action>` Step 3 instructed updating the public wrapper's docstring but did not specify updating the module-level docstring at `data.py:1-10`, which contained the same stale claim: "Phase 9 (EVAL-06) will replace `inverse_lambert_w_transform` with a fully differentiable alternative; the present implementation is a scalar round-trip using `scipy.special.lambertw` and therefore non-differentiable." After Task 1, that text became factually incorrect (the present implementation IS differentiable). Leaving it would directly contradict the new code and violate the acceptance criterion `grep -c 'Non-differentiable' core/data.py == 0`.
- **Fix:** Replaced with: "Phase 9 (EVAL-06) implements `inverse_lambert_w_transform` as a custom `torch.autograd.Function` (`_InverseLambertW`): `scipy.special.lambertw` is called only in the forward path (Phase 8 parity preserved bit-identically), and the backward path is pure torch using the closed-form identity `dW/dz = W/(z·(1+W))`."
- **Files modified:** `core/data.py` (lines 1-12)
- **Verification:** `grep -c 'Non-differentiable' core/data.py == 0` (passes); `grep -in 'non-differentiable' core/data.py` returns no matches; re-ran full Task 1 verification suite (gradcheck, parity, dtype, NaN-at-zero) — all still pass.
- **Committed in:** `e702bd4` (Task 1 commit, same file)

---

**Total deviations:** 1 auto-fixed (1 Rule 2 documentation correctness)
**Impact on plan:** Documentation hygiene only. No code-behavior change; in scope as part of the same in-place file edit. The acceptance criterion that "Non-differentiable" count == 0 was satisfied cleanly only after this cleanup.

## Issues Encountered

None. The plan was prescriptive and the verification script in `<verify>` caught everything in one pass. No iteration on the closed-form derivative (gradcheck green on first run). No NaN issues at zero (mask design was correct on first run).

## Open Items / Deferred to Plan 09-05

The following are explicitly out of scope for plan 09-01 and scheduled for plan 09-05 (round-trip notebook + smoke-test harness) per phase scope:

- **Round-trip 1e-8 verification** on (a) `torch.randn(777, dtype=float64)` synthetic input and (b) real `log_delta` (777 elements live count) via `02_eval06_roundtrip.ipynb` (D-04).
- **`full_denorm_pipeline` end-to-end smoke** — gradient-flow check on `gen_windows.grad` after `od_out.sum().backward()` (D-04b looser ≤1e-6 tolerance to absorb rolling-window un-stitching).
- **`01_parity_check.ipynb` re-run** to confirm parity_check.json still shows `pass=true` after the in-place replacement (regression check on Phase 8 baseline; expected to pass because the forward output is bit-identical).
- **Phase 09.1 scaffolding** (`core/preprocessing.py` with `forward_lambert`/`inverse_lambert` re-exports + `NotImplementedError` stubs) — separate plan (D-06).
- **Documentation deliverables** (`docs/training_protocol.md`, `docs/dataset_stats.md`) — separate plans (DOC-01, DOC-02).

## Threat Model Mitigations Applied

| Threat ID | Mitigation | Where |
|-----------|------------|-------|
| T-09-01 (forward correctness) | Verbatim preservation of legacy forward body; bit-identical parity verified (max_abs_diff = 0.0) | `_InverseLambertW.forward` |
| T-09-02 (NaN at x≈0) | `torch.where(\|data64\| < 1e-300, 1, W/safe_denom)` returns analytic limit value 1 | `_InverseLambertW.backward` |
| T-09-03 (dtype mismatch) | `grad_data.to(grad_output.dtype)` honors caller's dtype across the autograd boundary | `_InverseLambertW.backward` last line |
| T-09-04 (device leak via scipy) | `torch.tensor(lambert_result, dtype=torch.float64, device=data64.device)` preserves device after the numpy round-trip | `_InverseLambertW.forward` |
| T-09-05 (saved-tensor API misuse) | `ctx.save_for_backward(data64, lambert_tensor, out)` for tensors only; `ctx.delta = delta` for the Python float | `_InverseLambertW.forward` |
| T-09-06 (network/PII) | Accepted — pure numerical transform | — |

## Threat Flags

None — no new security-relevant surface introduced beyond the threat-model coverage above.

## Self-Check

**Files claimed created/modified:**
- `core/data.py` — `[ -f core/data.py ] && echo FOUND` → FOUND
- `_InverseLambertW` class present → `grep -c 'class _InverseLambertW' core/data.py` → 1
- public wrapper present → `grep -c 'def inverse_lambert_w_transform' core/data.py` → 1

**Commit claimed:**
- `e702bd4` → `git log --oneline | grep e702bd4` → `e702bd4 feat(09-01): differentiable inverse_lambert_w_transform via torch.autograd.Function` → FOUND

**Verification suite:** all 6 in-process checks pass (class exists, gradcheck, zero-NaN, dtype preservation, device preservation, bit-identical forward parity).

## Self-Check: PASSED

## Next Phase Readiness

- **Plan 09-05 (round-trip verification notebook)** can now run: `inverse_lambert_w_transform` is differentiable, gradcheck-validated, and the forward output is bit-identical to the Phase 8 baseline. The 1e-8 round-trip assertion on real `log_delta` should pass because (a) forward path is unchanged and (b) backward correctness is independent of the round-trip test (round-trip exercises only the forward composition `inverse∘forward`).
- **Phase 11 (TSTR)** can now backprop through `inverse_lambert_w_transform` into OD-scale loss surfaces.
- **Phase 12 (noise gradients)** can use the same path for OD-scale gradient computation.
- **No blockers** for downstream plans.

---
*Phase: 09-documentation-bridge*
*Completed: 2026-05-15*
