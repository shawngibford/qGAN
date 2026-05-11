# Phase 9: Documentation Bridge - Research

**Researched:** 2026-05-11
**Domain:** PyTorch custom autograd, scipy Lambert W, scientific documentation
**Confidence:** HIGH

## Summary

Phase 9 has three concrete deliverables — two markdown documentation files (`training_protocol.md`, `dataset_stats.md`) and one in-place code replacement (`inverse_lambert_w_transform` → custom `torch.autograd.Function`) — plus a scaffold for Phase 09.1 (`revision/core/preprocessing.py`). Every locked decision in CONTEXT.md (D-01 through D-10) is resolvable from existing code: numerical constants live in `revision/core/__init__.py`, the Lambert W math has a well-known closed-form derivative, and the round-trip verification pattern already exists in `revision/01_parity_check.ipynb`. The only **non-trivial** engineering is the autograd Function and its `gradcheck` validation — the rest is content assembly with strict citation discipline.

**Three discrepancies between CONTEXT.md and live code were discovered during research and must be reconciled by the planner** (see Open Questions):
1. CONTEXT.md states the tensor length is `776` ("torch.randn(776, dtype=float64)" in D-04). Live pipeline produces `log_delta` of length **777** (verified by running `load_and_preprocess` on `./data.csv`).
2. CONTEXT.md and `data.csv` row counts: CONTEXT.md claims "777 OD rows / 384 rolling windows"; actual is **778 OD rows → 777 log_delta values → 384 windows** (confirmed by live run).
3. Date range: CONTEXT.md says "2024-03-27 onwards"; actual end date in `data.csv` is **2024-03-31 23:52** (≈4.5 days, not "onwards" indefinitely).

**Primary recommendation:** Implement the custom autograd Function with `ctx.save_for_backward(W_tensor, data_input)`, use `gradcheck` for unit-level correctness, and reuse the `01_parity_check.ipynb` cell pattern for the user-facing 1e-8 round-trip assertion. Honor the **actual** 777 length, not the 776 cited in CONTEXT.md — flag the discrepancy in 09-PLAN before execution.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOC-01 | `revision/docs/training_protocol.md` documents N_CRITIC, λ, optimizer, both LRs, epochs, early-stopping rule, seeds, shot/analytic — traceable to `revision/core/__init__.py` | All 17 constants verified in `__init__.py:1-45`; Adam betas=(0.0,0.9) at `training.py:233-234`; EarlyStopping(patience=50, warmup_epochs=100) at `training.py:96-98` and notebook line 3983; seed=42 default at `training.py:188`; shots=None at `models/quantum.py:64` |
| DOC-02 | `revision/docs/dataset_stats.md` reports raw time-point count, rolling-window count, split convention, campaign count | Live pipeline verified: OD=778, log_delta=777, windows=384 (stride=2, WINDOW_LENGTH=10); single campaign starting 2024-03-27; sampling cadence 10-min (consecutive DATE deltas in `data.csv`) |
| EVAL-06 | `revision/core/data.py` exposes differentiable `inverse_transform`, ≤1e-8 round-trip | Closed-form `dW/dz = W/(z(1+W))` verified numerically to ~1e-10 against `lambertw`; `torch.autograd.Function` is the canonical pattern; `torch.autograd.gradcheck` validates implementation; Phase 8 parity baseline gives 0.0 EMD/moment drift, so the 1e-8 tolerance has full headroom |
</phase_requirements>

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Data Convention**
- **D-01: No train/val/test split.** All 777 OD rows / 384 rolling windows used for training. Justification: single-campaign dataset is too small to justify a held-out split; aligns with bioprocess single-campaign reality. EMD-based early stopping uses the same distribution as comparison — this is acknowledged as a methodological constraint in dataset_stats.md, in line with R1-M5 calibration honesty. *(Note: actual count is 778 OD / 777 log_delta — see Open Question OQ-1.)*
- **D-02: Single campaign acknowledged in DOC-02.** data.csv = exactly 1 campaign starting 2024-03-27, 10-min sampling, no other campaigns available. dataset_stats.md states this plainly with a 1-paragraph "Single-Campaign Limitation" prose block; multi-campaign generalization is referenced as a Phase 14 Outlook item, not a current scope claim.

**EVAL-06 — Differentiable Inverse Transform**
- **D-03: In-place replacement of `inverse_lambert_w_transform`.** No parallel function. Forward output stays at scipy precision (Phase 8 parity = 0.0 delta gives the headroom); backward path becomes a custom `torch.autograd.Function` with the closed-form analytic derivative `dW/dz = W / (z·(1+W))` (implicit-function-theorem identity, well-known for the principal branch).
- **D-04: Round-trip verification covers BOTH synthetic and real inputs.** A single test asserts `max_abs_error(inverse(forward(x)), x) ≤ 1e-8` on (a) a `torch.randn(776, dtype=float64)` synthetic tensor and (b) the full real `log_delta` tensor (776 elements). Synthetic = decoupled correctness; real = data-path correctness. *(See OQ-1: actual length is 777.)*
- **D-05: scipy stays in the forward path, removed from autograd.** `scipy.special.lambertw` is called once inside the `torch.autograd.Function.forward`; the backward path uses only torch ops on the cached `W` value. No new third-party dependencies.

**Phase 09.1 Scaffolding**
- **D-06: Add `revision/core/preprocessing.py` in Phase 9.** Exposes the full ablation contract: `forward_logreturns`/`inverse_logreturns`, `forward_lambert`/`inverse_lambert`, `forward_minmax_od`/`inverse_minmax_od`. Phase 9 implements only the Lambert pair (it IS EVAL-06); the other four raise `NotImplementedError("Phase 09.1")` with one-line docstrings describing the expected behavior.
- **D-07: `revision/core/data.py` keeps existing functions.** No symbol renames, no removals. The differentiable Lambert W implementation lives in `data.py` (where `inverse_lambert_w_transform` currently is); `preprocessing.py` re-exports it under the `inverse_lambert` name to satisfy the unified contract. Single source of truth in `data.py`.

**Documentation Style**
- **D-08: Hybrid format — tables for numbers + 1-paragraph prose for justifications.** Both `training_protocol.md` and `dataset_stats.md` follow this pattern.
- **D-09: Numbers traceable to `revision/core/__init__.py`.** Every constant is sourced from `__init__.py`. Doc cites the source file once at the top.
- **D-10: shot/analytic distinction stated explicitly.** training_protocol.md states clearly: "All Phase 9 results use analytic statevector simulation (PennyLane `default.qubit` with `shots=None`); shot-noise sweeps are reported separately in Phase 12."

### Claude's Discretion

- Exact section ordering inside training_protocol.md and dataset_stats.md.
- Exact wording of the single-campaign limitation paragraph (D-02).
- Whether to include a small "Reproducibility" subsection in training_protocol.md (cite the seed=42 default and `torch.manual_seed` location); add if it fits naturally.
- File-level layout of `preprocessing.py` (one function per pair vs grouped).
- Choice of synthetic tensor dtype/range for round-trip test — float64 + reasonable input distribution that exercises the Lambert W's domain.
- Light docstring + type-hint additions to `data.py` Lambert functions if they don't change behavior.

### Deferred Ideas (OUT OF SCOPE)

- Pipeline A (raw OD) and Pipeline B (log-returns only) implementations → Phase 09.1 (ABL-01).
- Multi-seed run framework → Phase 09.1 builds it for the 3-pipeline ablation; Phase 12 generalizes for shot/noise sweeps (SENS-03).
- Dataset histograms / OD-level moments in dataset_stats.md — could be added if cheap; otherwise covered by Phase 11 EVAL-05.
- Multi-campaign data pipeline → Phase 14 Outlook section.
- Differentiable forward Lambert W — `lambert_w_transform` is already pure-torch and differentiable; no work needed.
- Reproducibility section in training_protocol.md — Claude's discretion to include if it fits naturally; otherwise defer to Phase 14.
- Shot-noise / analytic distinction quantitative reporting — qualitative statement only in Phase 9; quantitative sweep is Phase 12 (SENS-01).
</user_constraints>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Numerical kernel: `lambertw` evaluation | scipy (forward only, per D-05) | — | scipy is the project-standard SciPy 1.x library for special functions; no in-tree dep change |
| Autograd graph: backward gradient | PyTorch `torch.autograd.Function` | — | Closed-form `dW/dz = W/(z(1+W))` is fast, exact, and avoids re-entering scipy in backward — D-05 mandates this |
| Public API surface | `revision/core/data.py::inverse_lambert_w_transform` (in-place per D-03) | `revision/core/preprocessing.py::inverse_lambert` (re-export per D-07) | data.py = single source of truth; preprocessing.py is the unified-API facade Phase 09.1 will fill out |
| Verification harness | New `revision/02_eval06_roundtrip.ipynb` (notebook-orchestrates pattern, INFRA-01 convention) | `revision/results/eval06_roundtrip.json` (JSON artifact) | Established pattern from `01_parity_check.ipynb`: notebook loads → calls modules → asserts → writes JSON |
| Documentation | `revision/docs/training_protocol.md`, `revision/docs/dataset_stats.md` | — | Plain markdown files; consumed by Phase 14 paper drafting |
| Number lineage | `revision/core/__init__.py` (citation target) | docs (citation source) | D-09: single source of truth so future HPO change updates one file |

## Standard Stack

### Core (already installed and verified)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 [VERIFIED: live import] | Custom autograd Function, gradcheck, float64 tensors | Project standard since v1.0 (`qgan_env`); `torch.autograd.Function` is the canonical PyTorch pattern for non-torch kernels |
| scipy | (existing) [VERIFIED: imported in `data.py:17`] | `scipy.special.lambertw` forward call (D-05) | Already in dep tree; principal-branch `.real` extraction matches existing code at `data.py:84` |
| numpy | (existing) [VERIFIED: `data.py:14`] | Buffer between torch and scipy | Project standard |
| PennyLane | 0.44.0 [CITED: PROJECT.md "Tech stack"] | Quantum simulator — doc reference only, no code change in Phase 9 | Project standard |
| pandas | (existing) [VERIFIED: `data.py:15`] | CSV load for dataset_stats verification | Project standard |

### Supporting (no new installs needed)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.autograd.gradcheck` | bundled with PyTorch | Numerical-vs-analytic gradient validation | Phase 9 round-trip unit test for the new autograd Function — runs once per CI invocation |
| Jupyter / nbconvert | (existing) | Execute round-trip notebook for verification | Pattern from `01_parity_check.ipynb` (executed via `jupyter nbconvert --execute` in Phase 8) |

### Alternatives Considered (all REJECTED per CONTEXT.md decisions)

| Instead of | Could Use | Tradeoff | Why Rejected |
|------------|-----------|----------|--------------|
| Custom `torch.autograd.Function` | Halley-iteration pure-torch Lambert W | Removes scipy from forward path entirely | D-05 explicitly keeps scipy in forward; user discretion chose option (a) |
| Custom `torch.autograd.Function` | `torchlambertw` third-party package | Off-the-shelf | D-05 forbids new third-party deps; closed-form derivative is one-liner |
| Parallel function `inverse_lambert_w_differentiable` | New symbol next to existing | Two functions → cleaner diff, but two code paths to maintain | D-03 explicitly: "In-place replacement. No parallel function." |
| 1e-6 round-trip tolerance | Looser tolerance | Easier to hit | D-04 locks 1e-8; Phase 8 parity = 0.0 gives headroom |

**Installation:** No new installs. Verify with:
```bash
python3 -c "import torch; import scipy.special; import pennylane; print(torch.__version__, scipy.__version__, pennylane.__version__)"
```

**Version verification:** [VERIFIED: live import on this machine 2026-05-11]
- PyTorch 2.10.0 (gradcheck and Function API stable since PyTorch 1.3)
- PennyLane 0.44.0 (`models/quantum.py:64` shots=None default.qubit confirmed)

## Architecture Patterns

### System Architecture Diagram

```
                         Phase 9 deliverable surface
                         ───────────────────────────
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  [Documentation]            [Differentiable code]      [09.1 scaffold]
   docs/*.md                  data.py inverse           preprocessing.py
        │                           │                           │
        │                           ▼                           ▼
        │                  ┌────────────────────┐    ┌──────────────────┐
        │                  │ torch.autograd.    │    │ forward_lambert  │──┐
        │                  │ Function           │    │ inverse_lambert  │  │ re-exports
        │                  │                    │    │ (re-exports)     │  │ from
        │                  │ forward:           │    └──────────────────┘  │ data.py
        │                  │  scipy.special.    │                          │
        │                  │  lambertw          │    ┌──────────────────┐  │
        │                  │  (D-05 boundary)   │    │ forward/inverse_ │  │
        │                  │                    │    │ logreturns       │  │
        │                  │  ctx.save: W,data  │    │ forward/inverse_ │  │
        │                  │                    │    │ minmax_od        │  │
        │                  │ backward:          │    │ → raise          │  │
        │                  │  closed-form       │    │   NotImpl(09.1)  │  │
        │                  │  dW/dz = W/(z(1+W))│    └──────────────────┘  │
        │                  │  pure torch        │                          │
        │                  └────────────────────┘                          │
        │                           │                                      │
        │                           ▼                                      │
        │                    [Verification harness]                        │
        │                    revision/02_eval06_roundtrip.ipynb            │
        │                           │                                      │
        │                           ├──> gradcheck (analytic vs numerical) │
        │                           ├──> round-trip on synthetic randn     │
        │                           ├──> round-trip on real log_delta      │
        │                           ├──> end-to-end full_denorm_pipeline   │
        │                           └──> revision/results/                 │
        │                                  eval06_roundtrip.json           │
        ▼                                                                  ▼
   Phase 14 (Paper)                                                   Phase 09.1
   Methods sections                                                   (ABL-01..03)
   (DOC-01, DOC-02)                                                   fills in stubs
```

### Recommended File Layout

```
revision/
├── core/
│   ├── __init__.py            # UNCHANGED — canonical constants (cited by docs)
│   ├── data.py                # MODIFIED — inverse_lambert_w_transform rewritten with autograd.Function
│   ├── preprocessing.py       # NEW — D-06 scaffold; Lambert pair re-exports from data.py
│   ├── eval.py                # UNCHANGED
│   └── training.py            # UNCHANGED
├── docs/
│   ├── training_protocol.md   # NEW — DOC-01
│   └── dataset_stats.md       # NEW — DOC-02
├── 01_parity_check.ipynb      # UNCHANGED (reference pattern)
├── 02_eval06_roundtrip.ipynb  # NEW — round-trip verification harness
└── results/
    └── eval06_roundtrip.json  # NEW — verification artifact (created by 02 notebook)
```

### Pattern 1: Custom `torch.autograd.Function` wrapping a non-differentiable kernel

**What:** PyTorch's documented pattern for inserting an external (here: scipy) operation into the autograd graph by providing an analytic backward. The `Function.forward` runs the non-torch kernel; `Function.backward` uses only torch ops on saved tensors. The implicit-function-theorem identity for Lambert W (`dW/dz = W/(z(1+W))` on the principal branch) makes the backward closed-form and exact.

**When to use:** Any time training requires gradients through a function whose forward implementation is in another library (scipy, numba, a CUDA kernel) but whose derivative has a known analytic form.

**Skeleton (verified pattern — [CITED: PyTorch docs `torch.autograd.Function`]):**

```python
import torch
from scipy.special import lambertw

class _InverseLambertW(torch.autograd.Function):
    """Differentiable inverse Lambert W transform.

    Forward:  z = sign(x) * sqrt(W(δ·x²) / δ)   (matches data.py:80-86)
    Backward: via chain rule and dW/dz = W/(z(1+W))
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, delta: float) -> torch.Tensor:
        # Promote to float64 (matches current data.py:80; required for 1e-8 tolerance)
        data64 = data.double()
        sign = torch.sign(data64)
        x2 = data64 ** 2
        # scipy boundary — D-05 keeps this only in forward
        lambert_arg = (delta * x2).cpu().numpy()
        W_np = lambertw(lambert_arg).real  # principal branch, matches data.py:84
        W = torch.from_numpy(W_np).to(dtype=torch.float64, device=data64.device)
        out = sign * torch.sqrt(W / delta)
        # Save tensors needed by backward
        ctx.save_for_backward(data64, W, out)
        ctx.delta = delta
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        data64, W, out = ctx.saved_tensors
        delta = ctx.delta
        # Chain rule: out = sign(x) * sqrt(W(δx²)/δ)
        # Let z = δ·x², w = W(z). Then sqrt-term = sqrt(w/δ).
        # d(out)/dx = sign(x) * (1 / (2·sqrt(w/δ))) * (1/δ) * dw/dz * dz/dx
        #           = sign(x) * (1 / (2·sqrt(w/δ))) * (1/δ) * (w/(z(1+w))) * 2δx
        # Simplify: dz/dx = 2δx; (1/δ)·(w/(z(1+w)))·2δx = 2x·w/(z(1+w))
        # And z = δ·x², so w/(z(1+w)) = w/(δ·x²·(1+w))
        # Final: d(out)/dx = sign(x) / (2·sqrt(w/δ)) · 2x·w / (δ·x²·(1+w))
        #                 = sign(x) · x · w / (δ·x²·(1+w)·sqrt(w/δ))
        # Note: sign(x)·x = |x|, and sqrt(w/δ) = |out| (since out = sign·sqrt(w/δ))
        # Cleaner form: differentiate out² = w/δ implicitly.
        #   d(out²)/dx = (1/δ)·dw/dx = (1/δ)·dw/dz·2δx = 2x·w/(z(1+w))
        #   2·out·d(out)/dx = 2x·w/(δx²(1+w))
        #   d(out)/dx = x·w / (out · δ · x² · (1+w))
        #             = w / (out · δ · x · (1+w))
        # Edge case: at x=0, out=0, w=0 — use limit. By Taylor: out ≈ sign(x)·|x|·sqrt(1) = x for small x,
        # so d(out)/dx → 1 at x=0. Implement with a stable mask.
        eps = torch.finfo(data64.dtype).tiny  # 2.225e-308 for float64
        # Safe denominators
        denom = out * delta * data64 * (1.0 + W)
        # Where data64 ≈ 0, override derivative with 1 (limit value)
        zero_mask = data64.abs() < 1e-300
        deriv = torch.where(
            zero_mask,
            torch.ones_like(data64),
            W / torch.where(zero_mask, torch.ones_like(denom), denom),
        )
        grad_data = grad_output * deriv
        # Cast back to caller dtype if it was float32
        return grad_data.to(grad_output.dtype), None  # None for delta (non-tensor)


def inverse_lambert_w_transform(data: torch.Tensor, delta: float) -> torch.Tensor:
    """Public API — same signature as before, now differentiable."""
    return _InverseLambertW.apply(data, delta)
```

**Notes on the derivation:**
- The simplest exact form (preferred for numerical stability) is the one derived by implicitly differentiating `out² = W(δ·data²) / δ`. Both forms are mathematically identical; pick the form that minimizes catastrophic cancellation near `data = 0`.
- An equivalent **scalar pre-multiplication** form: `d(out)/dx = (out / x) * (1 / (1 + W))` for `x ≠ 0` — derivable from the above. This may be more numerically stable because `out/x` stays O(1) near zero. The planner can choose either; `gradcheck` will catch errors in whichever form.

[ASSUMED] The "cleaner form" `out/x · 1/(1+W)` simplification — derived algebraically here in this RESEARCH but not verified by a third-party source. The planner MUST `gradcheck` against `torch.autograd.gradcheck` before committing.

### Pattern 2: `torch.autograd.gradcheck` for analytic-vs-numeric validation

**What:** PyTorch ships a built-in `torch.autograd.gradcheck` that compares the custom `backward` against a finite-difference numerical Jacobian. This is THE canonical test for custom autograd Functions.

**When to use:** Always, for any new `torch.autograd.Function`.

**Skeleton:**
```python
import torch
from revision.core.data import _InverseLambertW, inverse_lambert_w_transform

def test_gradcheck():
    # gradcheck needs float64 and inputs requiring grad
    x = torch.randn(20, dtype=torch.float64, requires_grad=True)
    # Wrap to a function of only the differentiable arg
    func = lambda data: _InverseLambertW.apply(data, 0.146932)
    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-6, rtol=1e-4)
```

[CITED: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html] — `gradcheck` is the documented validation tool for `torch.autograd.Function` implementations.

### Pattern 3: Notebook-orchestrates verification (mirrors `01_parity_check.ipynb`)

**What:** Notebooks load modules, run experiments, write JSON to `revision/results/`. No business logic in notebooks. The `01_parity_check.ipynb` cell pattern is `load → forward → inverse → assert → json.dump(artifact)`.

**Verified pattern (from `revision/01_parity_check.ipynb`):**
1. **Repo-root finder cell** — walks up CWD until it finds `data.csv` + `revision/core/`; inserts to `sys.path`. Required because `nbconvert --execute` sets CWD to notebook dir.
2. **Seeded RNG setup** — `torch.manual_seed(SEED); np.random.seed(SEED)`.
3. **Module imports** — `from revision.core.data import load_and_preprocess, inverse_lambert_w_transform, lambert_w_transform, full_denorm_pipeline`.
4. **Assertion cell** — explicit `assert max_abs_error <= 1e-8, f"..."` with informative failure message.
5. **JSON artifact** — `Path("revision/results/eval06_roundtrip.json").write_text(json.dumps(artifact, indent=2))`.
6. **Closing print** — `print("EVAL-06 round-trip PASSED")`.

### Pattern 4: `preprocessing.py` skeleton with `NotImplementedError` stubs

**What:** Lock the API contract for Phase 09.1 *now* so the ablation phase doesn't refactor mid-experiment. Phase 9 implements only the Lambert pair (re-exports from `data.py`); the other four raise `NotImplementedError("Phase 09.1")` with one-line docstrings.

**Skeleton (consumes the contract from `.planning/scratch/09.1-r1-m3-ablation-spec.md` lines 16-19 and lines 95-97):**

```python
"""Preprocessing pipelines — three ablation variants for R1-M3 (Phase 09.1).

Phase 9 implements only the Lambert W pair (it IS EVAL-06); the other four
functions are NotImplementedError stubs reserved for Phase 09.1.

Contract: each forward_X / inverse_X pair must satisfy
    max_abs(inverse_X(forward_X(x), *args), x) <= 1e-8
on a real OD trajectory.
"""
from __future__ import annotations
from typing import Tuple
import torch

# ── Pipeline C (CURRENT PAPER) — log-returns + Lambert W ────────────────────
# Re-export from data.py per D-07 (single source of truth)
from revision.core.data import (
    lambert_w_transform as forward_lambert,
    inverse_lambert_w_transform as inverse_lambert,
)


# ── Pipeline B — log-returns only ────────────────────────────────────────────
def forward_logreturns(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log-returns r_t = ln(OD_{t+1}/OD_t) and standardize. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


def inverse_logreturns(r: torch.Tensor, od_start: torch.Tensor, mu, sigma) -> torch.Tensor:
    """Un-standardize log-returns and integrate cumulatively from od_start. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


# ── Pipeline A — min-max normalized OD ───────────────────────────────────────
def forward_minmax_od(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize OD to [0, 1]; return (scaled, od_min, od_max). Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


def inverse_minmax_od(scaled: torch.Tensor, od_min: torch.Tensor, od_max: torch.Tensor) -> torch.Tensor:
    """Un-normalize scaled OD back to original units. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


__all__ = [
    "forward_lambert", "inverse_lambert",
    "forward_logreturns", "inverse_logreturns",
    "forward_minmax_od", "inverse_minmax_od",
]
```

### Anti-Patterns to Avoid

- **DO NOT** call `scipy.special.lambertw` inside `backward`. Per D-05, scipy is forward-only — backward must be pure torch on saved tensors. Calling scipy in backward also breaks vectorization and CUDA portability.
- **DO NOT** create a parallel `inverse_lambert_w_differentiable` function. D-03 mandates in-place replacement. Two code paths invite divergence.
- **DO NOT** rename or remove any symbol in `data.py`. D-07 mandates single source of truth; downstream code (including `01_parity_check.ipynb` at line `from revision.core.data import ...`) already imports by name.
- **DO NOT** implement the other four `forward_*`/`inverse_*` pairs in Phase 9. D-06 reserves them for Phase 09.1 — implementing now risks contract drift.
- **DO NOT** hand-type numbers in `training_protocol.md`. D-09 mandates traceability to `__init__.py`. Format: "N_CRITIC=9 (`revision/core/__init__.py:11`)".
- **DO NOT** claim "multi-campaign" or "train/val/test split" anywhere in the docs. D-01, D-02 explicitly lock these. The honest framing is "single-campaign, no held-out split, EMD early-stop limitation acknowledged".
- **DO NOT** make `inverse_lambert_w_transform` accept `requires_grad=False` tensors with no autograd path. The forward must register the saved tensors via `ctx.save_for_backward` even when the caller doesn't request grad — PyTorch handles the grad-disabled fast path internally.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lambert W principal-branch eval | Hand-rolled Halley iteration in torch | `scipy.special.lambertw` (D-05: stays in forward) | scipy's implementation handles series expansion near 0, asymptotic for large z, branch points — edge cases that took decades to harden |
| Numerical gradient validation | Hand-rolled finite-difference test | `torch.autograd.gradcheck` | Handles complex-step where applicable, integrates with PyTorch's grad system, has documented tolerances |
| float64 promotion at autograd boundary | Manual `.double()`/`.float()` casts everywhere | Promote on entry, cast back to `grad_output.dtype` in backward | Matches existing pattern at `data.py:80, 101`; the 1e-8 tolerance is unreachable in float32 |
| Closed-form Lambert W derivative | Look it up or derive manually each time | `dW/dz = W / (z·(1+W))` (this is the IFT identity for the principal branch) | Well-known identity since Corless et al. 1996; verified numerically in this research session to ~1e-10 against `scipy.special.lambertw` finite-diff |
| Markdown table numbers | Type values manually | Cite `revision/core/__init__.py:LINE` for each | Tables become drop-in Methods content for Phase 14; one update site for any future HPO change |

**Key insight:** The whole phase is glue — the math is trivial, the scipy and torch APIs are stable, and the project-internal infrastructure (notebook pattern, JSON artifacts, module imports) already exists from Phase 8. The principal risk is **mis-typing** a constant in the markdown docs or **mis-deriving** the chain-rule term in backward. `gradcheck` neutralizes the second risk; the "cite `__init__.py` line" rule neutralizes the first.

## Runtime State Inventory

> Phase 9 is **partially a refactor** (in-place replacement of `inverse_lambert_w_transform`). Most categories are clean because Phase 9 does not rename anything (D-07).

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | None — Phase 9 changes a pure function. No databases, no persistent state, no checkpoints store the Lambert W output. (Checkpoints store `params_pqc`, `critic_state`, `c_optimizer`, `g_optimizer`, `mu`, `sigma`, `epoch`, `emd` — see `training.py:151-160`. None reference the inverse transform.) | None |
| **Live service config** | None — no external services. | None |
| **OS-registered state** | None — no daemons, schedulers, or system services. | None |
| **Secrets and env vars** | None — no API keys or env vars used by `data.py`. | None |
| **Build artifacts / installed packages** | `revision/core/__pycache__/` — bytecode for `data.py`. Will auto-regenerate on next import after the in-place replacement. `revision/__pycache__/` likewise. No installed wheel — this is a project-local package, not a pip-installed dependency. | None — Python auto-invalidates `.pyc` by mtime. |

**Callers of `inverse_lambert_w_transform`:** grepped — only `revision/core/data.py::load_and_preprocess` (line 234) and `revision/01_parity_check.ipynb` (the inline path) call it. The Phase 8 parity check uses an **inline copy** of the function, so it will NOT pick up the differentiable version. **This is OK** because the parity check's purpose is to verify Path A (inline) vs Path B (module), and the module's forward output must still match the inline forward exactly to within the locked tolerance. **Verify after change:** rerun `01_parity_check.ipynb` to confirm parity_check.json still shows pass=true.

**Caller of the wrapper, transitively:** `full_denorm_pipeline` at `data.py:134-162` does NOT call `inverse_lambert_w_transform` — it calls the **forward** `lambert_w_transform` only (line 160). So the differentiable inverse change affects **only one direction** of the pipeline. The forward (Gaussian → heavy-tail) is already pure-torch differentiable.

## Common Pitfalls

### Pitfall 1: NaN gradients at x = 0
**What goes wrong:** `out = sign(x) * sqrt(W(δx²)/δ)`. At `x = 0`, `W(0) = 0`, so `out = 0` and the naïve derivative `W/(out·δ·x·(1+W))` has a `0/0` form. PyTorch will produce NaN unless explicitly masked.
**Why it happens:** Division by `out` and `x` simultaneously; both zero at `x = 0`.
**How to avoid:** Use `torch.where(x.abs() < tiny, ones_like(x), naive_deriv)`. The analytic limit `lim_{x→0} d(out)/dx = 1` (verified: at the origin `out ≈ sign(x)·|x| = x`, so derivative is 1).
**Warning signs:** `gradcheck` fails with NaN-comparison error; training loss becomes NaN after a few epochs because real `log_delta` data has values close to zero.

### Pitfall 2: float32 → float64 dtype mismatch in backward
**What goes wrong:** If caller passes `float32` data, forward promotes to `float64` for stability. Backward must cast the final gradient **back to caller dtype** to avoid breaking the autograd graph upstream.
**Why it happens:** PyTorch's autograd requires that `grad_input` matches `input.dtype` exactly.
**How to avoid:** `return grad_data.to(grad_output.dtype), None` (or `.to(data.dtype)` — both work as long as they match the *original* input dtype). The Phase 9 round-trip test uses float64 throughout, so this won't surface there — **add a float32 unit test** to catch it.
**Warning signs:** `RuntimeError: function _InverseLambertWBackward returned an invalid gradient at index 0 - got [torch.float64] but expected shape compatible with [torch.float32]`.

### Pitfall 3: scipy returns numpy on CPU only; original tensor may be on MPS/CUDA
**What goes wrong:** `(delta * x2).cpu().numpy()` followed by `torch.from_numpy(...)` brings the tensor to CPU. If the caller's input was on MPS or CUDA, the output ends up on CPU and breaks subsequent ops.
**Why it happens:** scipy is CPU-only; numpy arrays don't carry device info.
**How to avoid:** Explicit `.to(device=data.device)` after `from_numpy`. The current `data.py:85` already does this — preserve the pattern.
**Warning signs:** `RuntimeError: Expected all tensors to be on the same device` in downstream code after the inverse.

### Pitfall 4: 1e-8 tolerance vs scipy `lambertw` accuracy
**What goes wrong:** `scipy.special.lambertw` has documented relative accuracy ~1e-15 on the principal branch for positive real input — well within 1e-8. BUT `lambert_w_transform` (forward) applies `clamp(_, -12.0, 11.0)` at `data.py:104`. If the synthetic test tensor has values that round-trip outside `[-12, 11]`, the clamp introduces a non-invertible truncation.
**Why it happens:** D-04 mandates `torch.randn(776, dtype=float64)` synthetic input. randn output is virtually always inside `[-5, +5]`, so this is unlikely — but worth checking.
**How to avoid:** Verify on a controlled synthetic distribution that the *forward* path doesn't clamp. Use `torch.randn(776) * 2` or similar where amplitude stays bounded. Alternatively, document that the round-trip is on the unclamped portion.
**Warning signs:** Round-trip test passes on random but fails on real data, OR passes on synthetic but fails when called via `full_denorm_pipeline` which uses `lambert_w_transform` with clamp.

### Pitfall 5: gradcheck dtype requirement
**What goes wrong:** `torch.autograd.gradcheck` requires `dtype=torch.float64` AND `requires_grad=True` on inputs. Calling with float32 inputs silently passes gradcheck but doesn't actually validate the analytic backward.
**Why it happens:** Numerical derivatives at float32 precision are too noisy.
**How to avoid:** Always `x = torch.randn(N, dtype=torch.float64, requires_grad=True)` in gradcheck tests. [CITED: pytorch.org/docs gradcheck "Inputs need to be of double precision"].
**Warning signs:** `gradcheck` returns True but real-world gradients are wrong.

### Pitfall 6: ctx.save_for_backward with non-tensor values
**What goes wrong:** `delta` is a Python float, not a tensor. Saving it via `ctx.save_for_backward(..., delta)` raises an error.
**Why it happens:** `save_for_backward` only accepts tensors.
**How to avoid:** Store non-tensors directly on `ctx`: `ctx.delta = delta`. The skeleton above does this.
**Warning signs:** `TypeError: save_for_backward can only save tensors`.

### Pitfall 7: Doc numbers drift if `__init__.py` changes
**What goes wrong:** Future HPO retune changes `LR_CRITIC` in `__init__.py`. `training_protocol.md` still shows the old value.
**Why it happens:** Markdown is hand-typed; no programmatic link.
**How to avoid:** D-09 partially mitigates by mandating citations like "(`revision/core/__init__.py:13`)" so a reviewer sees the source. **Stronger mitigation (Claude's discretion):** the Phase 9 plan can include a sanity-check cell in `02_eval06_roundtrip.ipynb` that asserts the constants in the docs match `__init__.py` (string regex). Optional; nice to have.
**Warning signs:** Future phase report has a number that disagrees with the markdown.

## Code Examples

### Round-trip assertion idiom (preserves Phase 8 pattern)

```python
# In revision/02_eval06_roundtrip.ipynb (cell pattern from 01_parity_check.ipynb)
import torch
from revision.core.data import (
    load_and_preprocess,
    inverse_lambert_w_transform,
    lambert_w_transform,
    full_denorm_pipeline,
)

# (1) Synthetic round-trip — decoupled correctness
torch.manual_seed(42)
delta = 0.146932  # Phase 8 parity_check value; could also load_and_preprocess to recompute
x_synth = torch.randn(777, dtype=torch.float64, requires_grad=True)
# Round-trip: forward Lambert (Gaussian→heavy-tail) THEN inverse Lambert (heavy→Gaussian)
y_synth = lambert_w_transform(x_synth, delta)
x_synth_rt = inverse_lambert_w_transform(y_synth, delta)
err_synth = (x_synth_rt - x_synth).abs().max().item()
assert err_synth <= 1e-8, f"Synthetic round-trip failed: {err_synth:.3e}"
print(f"Synthetic round-trip max_abs_error = {err_synth:.3e}  [PASS, ≤ 1e-8]")

# (2) Real-data round-trip — data-path correctness
d = load_and_preprocess("./data.csv")
real = d["norm_log_delta"].double()  # pre-Lambert input (Gaussian-ish)
# transformed_norm_log_delta = inverse_lambert(norm_log_delta) ... but that's the wrong direction.
# Correct: forward is Gaussian→heavy; we want inverse(forward(real)) ≈ real.
y_real = lambert_w_transform(real, d["delta"])
real_rt = inverse_lambert_w_transform(y_real, d["delta"])
err_real = (real_rt - real).abs().max().item()
assert err_real <= 1e-8, f"Real round-trip failed: {err_real:.3e}"
print(f"Real round-trip max_abs_error = {err_real:.3e}  [PASS, ≤ 1e-8]")

# (3) gradcheck — analytic vs numerical backward
ok = torch.autograd.gradcheck(
    lambda v: inverse_lambert_w_transform(v, d["delta"]),
    (torch.randn(20, dtype=torch.float64, requires_grad=True),),
    eps=1e-6, atol=1e-6,
)
assert ok, "gradcheck failed"
print("gradcheck PASSED")

# (4) End-to-end full pipeline differentiability — confirm full_denorm_pipeline still flows gradients
# (This uses lambert_w_transform (forward), not inverse — included as a smoke test that the
# end-to-end OD-scale path remains differentiable after the EVAL-06 change.)
gen_windows = torch.randn(10, 10, dtype=torch.float64, requires_grad=True)
od_out = full_denorm_pipeline(gen_windows, d["transformed_norm_log_delta"],
                              d["mu"].double(), d["sigma"].double(), d["delta"])
od_out.sum().backward()
assert gen_windows.grad is not None and not torch.isnan(gen_windows.grad).any()
print("full_denorm_pipeline gradient flow PASSED")
```

### `training_protocol.md` skeleton (DOC-01) — hybrid format (D-08)

```markdown
# Training Protocol — QWGAN-GP (v1.1 unconditioned baseline)

> **Source of truth:** all numerical constants below are imported from
> `revision/core/__init__.py`. Update that file to change them; this doc
> tracks the file via the line-cited references in the table.

## Optimizer & Schedule

| Constant | Value | Source |
|----------|-------|--------|
| `N_CRITIC` | 9 critic steps per generator step | `revision/core/__init__.py:11` |
| `LAMBDA` (gradient penalty coeff) | 2.16 | `revision/core/__init__.py:12` |
| `LR_CRITIC` | 1.8046 × 10⁻⁵ | `revision/core/__init__.py:13` |
| `LR_GENERATOR` | 6.9173 × 10⁻⁵ | `revision/core/__init__.py:14` |
| Optimizer | Adam, betas=(0.0, 0.9) | `revision/core/training.py:233-234` |
| `NUM_EPOCHS` | 2000 | `revision/core/__init__.py:20` |
| `BATCH_SIZE` | 12 | `revision/core/__init__.py:21` |

(prose: 1-paragraph justification — HPO-tuned values from v1.1 Phase 4; Adam betas chosen for WGAN-GP stability...)

## Early-Stopping

| Property | Value | Source |
|----------|-------|--------|
| Monitored metric | EMD on log-returns | `revision/core/training.py:79-140` |
| `patience` | 50 eval cycles (= 500 epochs at EVAL_EVERY=10) | `revision/core/training.py:96` |
| `warmup_epochs` | 100 epochs (no monitoring during warmup) | `revision/core/training.py:97` |
| Checkpoint scheme | save-best-EMD, reload on stop | `revision/core/training.py:142-175` |

(prose: 1-paragraph — EMD chosen over critic loss; EMD-on-same-distribution caveat per R1-M5...)

## Quantum Circuit

| Property | Value | Source |
|----------|-------|--------|
| Backend | PennyLane `default.qubit`, `shots=None` (analytic statevector) | `revision/core/models/quantum.py:64` |
| Differentiation | `diff_method="backprop"` | `revision/core/models/quantum.py:43, 76` |
| `NUM_QUBITS` | 5 | `revision/core/__init__.py:17` |
| `NUM_LAYERS` | 4 strongly-entangled | `revision/core/__init__.py:18` |
| `WINDOW_LENGTH` | 10 (= 2 × NUM_QUBITS) | `revision/core/__init__.py:19` |
| Noise range | [0, 4π] (NOT [0, 2π]; v1.1 Phase 4) | `revision/core/__init__.py:32-33` |
| PQC parameter count | 75 (= 5 + 4·15 + 10) | verified Phase 8 |

## Critic (1D-CNN)

| Property | Value | Source |
|----------|-------|--------|
| Architecture | Conv1d(1→64)→Conv1d(64→128)→Conv1d(128→128)→AdaptiveAvgPool1d→Linear(128→32)→Dropout→Linear(32→1) | `revision/core/models/critic.py` |
| Kernel size | 10, padding 5 | `revision/core/models/critic.py` |
| Dropout | 0.2 (configurable) | `revision/core/__init__.py:24` |
| Precision | float64 (`.double()`) | `revision/core/models/critic.py` |

## Gradient Penalty

| Property | Value | Source |
|----------|-------|--------|
| Type | Two-sided (mean((‖∇‖₂ − 1)²)) | `revision/core/training.py:30-73` |
| Coefficient λ | 2.16 (= `LAMBDA`) | `revision/core/__init__.py:12` |
| Interpolation α | sampled per-sample, U(0,1), broadcast over remaining dims | `revision/core/training.py:54-60` |

## Reproducibility

| Property | Value | Source |
|----------|-------|--------|
| Seed (default) | 42 | `revision/core/training.py:188` |
| Seeded RNGs | `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all` | `revision/core/training.py:211-215` |
| DITHER (data pipeline) | 0.005 | `revision/core/__init__.py:27` |
| DITHER_SEED | 42 | `revision/core/__init__.py:28` |

(1-line prose: "All Phase 9 results use analytic statevector simulation (`default.qubit`, `shots=None`); shot-noise sweeps are reported separately in Phase 12 SENS-01." — fulfills D-10.)
```

### `dataset_stats.md` skeleton (DOC-02) — hybrid format (D-08)

```markdown
# Dataset Statistics — Single-Campaign LUCY Photobioreactor

> **Source CSV:** `./data.csv` (10-min sampling, columns DATE, PRE, TEMP_EXT,
> TEMP_CULTURE, PAR_LIGHT, PH, DO, OD, DRY, CELL).

## Counts

| Quantity | Value | Source / Derivation |
|----------|-------|---------------------|
| Raw CSV rows | 778 | `wc -l data.csv` minus header |
| OD rows post fillna+dropna | 778 | `revision/core/data.py::load_and_preprocess` cell 5 logic (lines 211-219) |
| Log-return rows (N−1) | 777 | `compute_log_delta`: `log_od[1:] - log_od[:-1]` |
| Rolling windows (m=10, s=2) | 384 | `rolling_window` (`data.py:110-118`): `(777−10)//2 + 1 = 384` |
| Independent campaigns | 1 | LUCY bioreactor, single run |

## Sampling & Date Range

| Property | Value |
|----------|-------|
| Bioreactor | LUCY photobioreactor (Algenuity) |
| Sampling cadence | 10 minutes |
| Start date | 2024-03-27 13:12 |
| End date | 2024-03-31 23:52 |
| Duration | ≈ 4.5 days |

## Split Convention

| Convention | Decision |
|------------|----------|
| Train / val / test | NONE — all 384 windows used for training |
| Held-out evaluation | NONE — EMD early-stop uses the same distribution (acknowledged limitation per R1-M5) |

(1-paragraph "Single-Campaign Limitation" prose block per D-02: justifies no-split decision in single-campaign reality; references multi-campaign generalization → Phase 14 Outlook.)

## Preprocessing Pipeline

(brief reference: log-returns → standardize → Lambert W heavy-tail → min-max to [-1,1] → rolling windows; full ablation of this pipeline → Phase 09.1 R1-M3.)

## PAR_LIGHT Note

PAR_LIGHT is captured in `data.csv` but conditioning was **disabled** in the
final v1.1 unconditioned baseline (`RUN_NAME = "unconditioned_wgan"` —
`qgan_pennylane.ipynb` cell 65). PAR_LIGHT is reserved for Phase 13
conditional-generation introspection if revisited.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| scipy-only inverse (non-differentiable) | Custom `torch.autograd.Function` with closed-form backward | Phase 9 (this phase) | Enables gradient flow through inverse for TSTR (Phase 11) and OD-scale optimization (Phase 12) |
| Numerical gradient approximations | `torch.autograd.gradcheck` for validation | PyTorch 1.3+ (stable since 2019) | Standard pattern; gradcheck is documented [CITED: pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html] |
| Hand-derived chain-rule formulas | Implicit function theorem identity `dW/dz = W/(z(1+W))` | Known since Corless et al. 1996 "On the Lambert W function" | Avoids re-derivation each time; numerically validated to ~1e-10 in this research session |

**Deprecated/outdated:**
- Per-sample QNode loops (replaced by batched broadcasting in v1.1 Phase 5) — not relevant to Phase 9 directly, but `training.py:282` already uses the modern batched call so any code added must respect that.
- `n_critic=1, LAMBDA=0.8` from v0.x — replaced by HPO-tuned `N_CRITIC=9, LAMBDA=2.16` in v1.1 Phase 4. `training_protocol.md` documents the v1.1 values, NOT the v0.x defaults.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Simplified backward form `d(out)/dx = w/(out·δ·x·(1+w))` is equivalent to the chain-rule expansion | Pattern 1 | LOW — `gradcheck` will catch any error; this is mechanical algebra |
| A2 | `lim_{x→0} d(inverse_lambert)/dx = 1` (justifying the `torch.where` mask) | Pitfall 1 | LOW — verifiable by Taylor expansion: `out ≈ x` near zero, so derivative is 1 |
| A3 | `torch.autograd.gradcheck` accepts a lambda wrapping `Function.apply` | Pattern 2 | LOW — standard pattern; if it fails, wrap in `nn.Module` |
| A4 | Cited `data.py` line numbers for the new differentiable function will land near the same lines as the current implementation (68-87) | Architecture diagram | LOW — implementation detail; doc citations in `training_protocol.md` use `__init__.py` lines which are stable |
| A5 | The "Reproducibility" subsection in training_protocol.md (Claude's discretion) fits naturally | Doc skeleton | LOW — optional per CONTEXT.md; can be dropped if it bloats the doc |

## Open Questions

1. **OQ-1: Discrepancy between CONTEXT.md "776" and actual `log_delta` length of 777.**
   - What we know: Live `load_and_preprocess('./data.csv')` returns `log_delta` of length **777** (verified `2026-05-11` on this machine).
   - What's unclear: Whether CONTEXT.md's "776" was a transcription error from the discussion phase, or whether the user is on a slightly different data.csv state.
   - Recommendation: **Honor the live value (777)** in the synthetic round-trip test (`torch.randn(777, dtype=torch.float64)`) and in `dataset_stats.md`. The 1-off difference is meaningless for autograd correctness; the test still validates 776 elements with `torch.randn(776)` if the user prefers. The planner should ASK the user once before locking the doc number, then proceed.

2. **OQ-2: data.csv row count is 778, not 777.**
   - What we know: `wc -l data.csv` = 778 (rows) + 1 (header) = 779 total lines, so 778 data rows. `load_and_preprocess` returns OD tensor of length 778.
   - What's unclear: Whether CONTEXT.md's "777 OD rows" is a transcription error or reflects an older CSV version.
   - Recommendation: **dataset_stats.md uses 778 OD / 777 log_delta / 384 windows** — derived from live pipeline, not from CONTEXT.md prose.

3. **OQ-3: Date range — "2024-03-27 onwards" vs actual end date 2024-03-31 23:52.**
   - What we know: Last row in `data.csv` is `2024-03-31 23:52` (~4.5 days total).
   - What's unclear: Whether the user wants the doc to say "onwards" (implying the campaign continues) or the explicit end timestamp.
   - Recommendation: Be **specific** in `dataset_stats.md`: "2024-03-27 13:12 to 2024-03-31 23:52 (≈ 4.5 days)". Honest, paper-ready.

4. **OQ-4: Should `full_denorm_pipeline` (end-to-end OD scale) be round-tripped as part of EVAL-06?**
   - What we know: D-04 specifies the test on `inverse_lambert_w_transform` only. CONTEXT.md `<specifics>` doesn't expand. The Phase 9 ROADMAP success criterion 3 says "log-return + Lambert W back-transform to OD".
   - What's unclear: Whether the 1e-8 tolerance applies to the bare Lambert pair (forward then inverse) or the full pipeline (rescale + lambert + denormalize).
   - Recommendation: Test **both** — the bare Lambert round-trip with strict 1e-8, AND a smoke test on `full_denorm_pipeline` confirming gradient flow + reasonable round-trip behavior. The latter has accumulated float32 error from the `rescale` step (`data.py:121-128`) that probably exceeds 1e-8 — document the looser tolerance there explicitly (e.g., 1e-6) without weakening the headline 1e-8 claim on the Lambert pair.

5. **OQ-5: Synthetic test distribution range.**
   - What we know: D-04 specifies `torch.randn(N, dtype=torch.float64)`. Standard normal stays within ~[-5, +5] with overwhelming probability.
   - What's unclear: Whether `randn` adequately exercises the Lambert W domain — the real `norm_log_delta` after standardization has range roughly `[-5, +5]` as well, so it's representative.
   - Recommendation: Use `torch.randn(777, dtype=torch.float64)` directly. If a future reviewer questions coverage, add a uniform `[-3, +3]` variant.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All revision code | ✓ | 3.11+ (also 3.14 present) [VERIFIED: live run] | — |
| PyTorch | Custom autograd Function, gradcheck | ✓ | 2.10.0 [VERIFIED] | — |
| scipy | `scipy.special.lambertw` (forward path) | ✓ | (installed; matches data.py import) [VERIFIED] | — |
| numpy | Buffer between torch and scipy | ✓ | 1.26.4 [VERIFIED] | — |
| PennyLane | Doc reference only (no Phase 9 code call) | ✓ | 0.44.0 [CITED: PROJECT.md] | — |
| pandas | CSV stat verification | ✓ | (installed) [VERIFIED] | — |
| Jupyter / nbconvert | Execute `02_eval06_roundtrip.ipynb` | ✓ (already used in Phase 8) [VERIFIED: 01_parity_check.ipynb ran successfully] | — | — |
| git | Phase commit | ✓ | [VERIFIED: git status works] | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

> **Skipped — `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json:12`.**

If re-enabled later, the EVAL-06 test would map cleanly:
- REQ-EVAL-06 → unit + integration: `assert (inverse(forward(x)) - x).abs().max() <= 1e-8` runnable in seconds.
- REQ-DOC-01 → doc-existence + grep checks for required constants.
- REQ-DOC-02 → doc-existence + grep checks for required counts.

## Sources

### Primary (HIGH confidence)
- `revision/core/data.py` (read in full, 257 lines) — current `inverse_lambert_w_transform` at lines 68-87, `lambert_w_transform` at lines 90-104, `full_denorm_pipeline` at lines 134-162, `load_and_preprocess` at lines 187-256. All references in this RESEARCH cite live line numbers.
- `revision/core/__init__.py` (read in full, 45 lines) — all hyperparameter constants. Cited line-by-line in the doc skeletons.
- `revision/core/training.py` (read in full, 483 lines) — Adam betas (lines 233-234), seed setup (lines 211-215), EarlyStopping (lines 79-175).
- `revision/core/models/quantum.py` (line-grep) — `default.qubit`, `shots=None`, `diff_method="backprop"` at lines 64, 76, 43.
- `revision/core/eval.py` (read in full, 164 lines) — EMD on raw samples decision (line 25-36).
- `revision/01_parity_check.ipynb` (read in full) — established notebook pattern that 02_eval06_roundtrip.ipynb will follow.
- `.planning/phases/09-documentation-bridge/09-CONTEXT.md` (read in full, 140 lines) — locked decisions D-01..D-10.
- `.planning/phases/08-core-module-extraction/08-VERIFICATION.md` (read in full) — Phase 8 parity baseline (EMD delta = 0.0).
- `.planning/scratch/09.1-r1-m3-ablation-spec.md` (read in full, 101 lines) — preprocessing.py contract for Phase 09.1 stubs.
- `.planning/REQUIREMENTS.md` — DOC-01, DOC-02, EVAL-06 definitions.
- Live execution: `python3 -c "from revision.core.data import load_and_preprocess; ..."` on 2026-05-11 — confirmed OD=778, log_delta=777, windows=384, delta=0.146932, mu=0.002449, sigma=0.021759.
- Live numerical check: `lambertw` derivative closed-form vs central-difference at z ∈ {0.01, 0.1, 1.0, 10.0} agreed to ~1e-10 absolute.

### Secondary (MEDIUM confidence)
- PyTorch documentation pattern for `torch.autograd.Function` [CITED: https://pytorch.org/docs/stable/notes/extending.html and https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html] — referenced from training knowledge; not re-fetched in this session.
- Corless et al. 1996, "On the Lambert W function" — `dW/dz = W/(z(1+W))` identity [CITED: standard reference]; verified numerically here.

### Tertiary (LOW confidence)
- None — all critical claims are either VERIFIED against live code or CITED to primary docs.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library version verified by live import.
- Architecture: HIGH — every reference to existing code includes a file + line citation.
- Pitfalls: HIGH — derived from active reading of the existing code; the NaN-at-zero issue verified numerically here.
- Doc content: HIGH — every constant in the doc skeletons has a file:line citation.
- Discrepancies (OQ-1, OQ-2, OQ-3): HIGH — verified against live data.csv and live pipeline execution; reconcile with user before locking doc text.

**Research date:** 2026-05-11
**Valid until:** 2026-06-10 (30 days — stable scientific Python stack; `revision/core/` is locked from Phase 8)

## RESEARCH COMPLETE
