# Phase 9: Documentation Bridge - Pattern Map

**Mapped:** 2026-05-11
**Files analyzed:** 5 (1 modify, 4 create)
**Analogs found:** 3 in-repo + 1 PyTorch-docs idiom + 1 no-analog (docs) / 5 total

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `revision/core/data.py` (MODIFY) | utility (numerical transform) | transform / autograd | `revision/core/data.py` existing `inverse_lambert_w_transform` (lines 68–87) + `lambert_w_transform` (lines 90–104) + PyTorch `torch.autograd.Function` docs idiom | exact body, new autograd wrapper (no in-repo Function subclass) |
| `revision/core/preprocessing.py` (CREATE) | module facade / API contract | re-export | `revision/core/data.py` (module structure, `from __future__ import`, docstring style, `__all__`) | role-match (sibling module) |
| `revision/02_eval06_roundtrip.ipynb` (CREATE) | test / verification notebook | request-response (load→assert→json) | `revision/01_parity_check.ipynb` | exact (canonical pattern; same file family) |
| `revision/docs/training_protocol.md` (CREATE) | documentation | static content + traceable refs | none in-repo (`revision/docs/` is empty — `.gitkeep` only) | no analog (RESEARCH skeleton is the template) |
| `revision/docs/dataset_stats.md` (CREATE) | documentation | static content + traceable refs | none in-repo | no analog (RESEARCH skeleton is the template) |

## Pattern Assignments

### `revision/core/data.py` (utility, transform/autograd) — MODIFY

**Analog:** existing `inverse_lambert_w_transform` body at `revision/core/data.py:68–87` (forward output behavior to preserve verbatim) + `lambert_w_transform` at `revision/core/data.py:90–104` (float64 promotion style) + PyTorch docs `torch.autograd.Function` idiom (no in-repo subclass exists — grep returned 0 hits).

**Imports pattern** (file `revision/core/data.py:11–21`):
```python
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis as _sp_kurtosis

from revision.core import DITHER, DITHER_SEED, PAR_LIGHT_MAX, WINDOW_LENGTH
```
Copy: keep the existing import block untouched; no new third-party deps (D-05). The new `torch.autograd.Function` subclass uses only `torch` + `scipy.special.lambertw` (both already imported).

**Forward-path behavior to preserve verbatim** (file `revision/core/data.py:80–87`):
```python
data = data.double()                                       # float64 promote
sign = torch.sign(data)
data_squared = data ** 2
lambert_input = (delta * data_squared).cpu().numpy()       # scipy boundary
lambert_result = lambertw(lambert_input).real              # principal branch
lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
transformed_data = sign * torch.sqrt(lambert_tensor / delta)
return transformed_data
```
The new `_InverseLambertW.forward(ctx, data, delta)` MUST produce a bit-identical tensor to this (Phase 8 parity baseline = 0.0 delta — D-03 in-place replacement, no parallel function).

**Float64 promotion convention** (file `revision/core/data.py:101`, mirroring forward):
```python
transformed_data = transformed_data.double()
```
Apply: promote on entry of `forward`; cast `grad_data.to(grad_output.dtype)` at the end of `backward` to honor caller dtype.

**Device preservation pattern** (file `revision/core/data.py:85`):
```python
lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
```
Apply: the new Function must continue to use `device=data.device` after `torch.from_numpy(...)` so MPS/CUDA tensors round-trip through scipy without leaking onto CPU (Pitfall 3 in RESEARCH).

**Section-header / numbering style** (file `revision/core/data.py:65–67`):
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell 17 — Lambert W transforms
# ─────────────────────────────────────────────────────────────────────────────
```
Apply: keep the existing comment banner directly above `inverse_lambert_w_transform`. Add the `class _InverseLambertW(torch.autograd.Function):` immediately above the public wrapper, under the same "Cell 17" banner.

**Public-API signature to preserve** (file `revision/core/data.py:68`):
```python
def inverse_lambert_w_transform(data: torch.Tensor, delta: float) -> torch.Tensor:
```
Apply: the public wrapper becomes a one-line `return _InverseLambertW.apply(data, delta)`. Same signature, same return shape/dtype. D-07: no rename.

**Docstring template to preserve** (file `revision/core/data.py:69–79`):
```python
"""Inverse Lambert W transform (heavy-tail → Gaussian-ish).

Notebook cell 17. Uses ``scipy.special.lambertw`` on the principal branch
(``.real``). Promotes to float64 for numerical stability — the notebook
does the same (``data.double()``).

Note
----
Non-differentiable. Phase 9 (EVAL-06) replaces this with a differentiable
alternative.
"""
```
Update: drop the "Non-differentiable" note; replace with "Differentiable via custom `torch.autograd.Function` with closed-form backward `dW/dz = W/(z(1+W))`." Keep the cell-17 lineage line for traceability.

**Out-of-repo idiom (no in-repo analog — PyTorch docs):**
```python
class _InverseLambertW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, delta: float) -> torch.Tensor:
        # ... (verbatim of lines 80–87 above) ...
        ctx.save_for_backward(data_float64, W_tensor, out)   # tensors only
        ctx.delta = delta                                     # non-tensor on ctx
        return out

    @staticmethod
    def backward(ctx, grad_output):
        data, W, out = ctx.saved_tensors
        delta = ctx.delta
        # Closed-form via implicit-function-theorem identity dW/dz = W/(z(1+W))
        # masked at x≈0 with the limit value 1 (Pitfall 1 of RESEARCH).
        ...
        return grad_data.to(grad_output.dtype), None         # None for `delta`
```
Source: RESEARCH.md "Pattern 1" lines 193–258 (verbatim skeleton, including the `zero_mask` torch.where idiom).

---

### `revision/core/preprocessing.py` (module facade) — CREATE

**Analog:** `revision/core/data.py` (sibling module, same package, same style conventions).

**Imports pattern** (copy from `revision/core/data.py:11–17`):
```python
from __future__ import annotations
from typing import Tuple
import torch

from revision.core.data import (
    lambert_w_transform as forward_lambert,
    inverse_lambert_w_transform as inverse_lambert,
)
```
Per D-07: single source of truth in `data.py`; `preprocessing.py` re-exports under the unified contract names.

**Module docstring pattern** (mirroring `revision/core/data.py:1–10` and `revision/core/__init__.py:1–6`):
```python
"""Preprocessing pipelines — three ablation variants for R1-M3 (Phase 09.1).

Phase 9 implements only the Lambert W pair (it IS EVAL-06); the other four
functions are NotImplementedError stubs reserved for Phase 09.1.

Contract: each forward_X / inverse_X pair must satisfy
    max_abs(inverse_X(forward_X(x), *args), x) <= 1e-8
on a real OD trajectory.
"""
```
Copy the "module is a refactor not rewrite" voice from `data.py:1–10`. Same triple-quoted module docstring at top, before `from __future__ import`.

**Section-header style** (mirroring `revision/core/data.py:24–26`, `:43–45`, `:65–67`):
```python
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline C (CURRENT PAPER) — log-returns + Lambert W
# ─────────────────────────────────────────────────────────────────────────────
```
Apply three banners: one per pipeline variant. Keep the box-drawing style verbatim — it is the project signature comment header (used 6 times in `data.py`).

**`__all__` pattern** (copy from `revision/core/__init__.py:38–45`):
```python
__all__ = [
    "forward_lambert", "inverse_lambert",
    "forward_logreturns", "inverse_logreturns",
    "forward_minmax_od", "inverse_minmax_od",
]
```
Use `__all__` explicit list to lock the public API surface for Phase 09.1.

**NotImplementedError stub pattern** (no in-repo analog — RESEARCH.md "Pattern 4" lines 305–354 is the template):
```python
def forward_logreturns(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log-returns r_t = ln(OD_{t+1}/OD_t) and standardize. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")
```
Apply identical signature shape to the other 3 stubs (`inverse_logreturns`, `forward_minmax_od`, `inverse_minmax_od`). Each gets one-line docstring stating expected behavior + `raise NotImplementedError("Phase 09.1")`.

**Module registration:** `revision/core/__init__.py:35` currently imports `data, eval, training` and `models`. Phase 9 must add `preprocessing` to that import line and to `__all__` so the contract is package-level accessible (`from revision.core import preprocessing`).

---

### `revision/02_eval06_roundtrip.ipynb` (test / verification notebook) — CREATE

**Analog:** `revision/01_parity_check.ipynb` (canonical pattern; same notebook family in same dir).

**Markdown header cell pattern** (copy structure from `01_parity_check.ipynb` cell `51f2e835`):
```markdown
# Phase 9 EVAL-06 Round-Trip Verification

Verifies the differentiable `inverse_lambert_w_transform` (Phase 9):

- **Synthetic round-trip**: max|inverse(forward(x)) − x| ≤ 1e-8 on torch.randn(777, dtype=float64)
- **Real round-trip**: max|inverse(forward(log_delta)) − log_delta| ≤ 1e-8 on real 777-element log_delta
- **gradcheck**: analytic backward vs finite-difference, atol=1e-6
- **Full pipeline smoke test**: full_denorm_pipeline gradient flow non-NaN, looser ≤ 1e-6 tolerance (D-04b)

Output: `revision/results/eval06_roundtrip.json` with `pass: true`.
```

**Repo-root finder cell — copy verbatim from `01_parity_check.ipynb` cell `2c8bc6c2` lines 1–46**:
```python
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# nbconvert sets CWD to the notebook's directory. Walk upward to repo root
# (the directory containing data.csv + revision/).
def _find_repo_root():
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / "data.csv").exists() and (d / "revision" / "core").is_dir():
            return d
    raise FileNotFoundError(
        "Could not locate repo root from " + str(here) +
        " (looked for data.csv + revision/core)"
    )

REPO_ROOT = _find_repo_root()
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"Repo root: {REPO_ROOT}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```
**MUST copy verbatim** — this is the load-bearing nbconvert compatibility shim (cited as "Pattern 3 step 1" in RESEARCH.md line 292). Do not improvise.

**Module-import cell pattern** (copy from `01_parity_check.ipynb` cell `5d83ed4a` lines 1–9):
```python
from revision.core.data import (
    load_and_preprocess,
    inverse_lambert_w_transform,
    lambert_w_transform,
    full_denorm_pipeline,
)
```

**Assertion + JSON-artifact cell pattern** (copy from `01_parity_check.ipynb` cell `a28db61d` lines 1–46):
```python
tolerance = {"synthetic": 1e-8, "real": 1e-8, "full_pipeline": 1e-6, "gradcheck": True}
delta = {
    "synthetic": err_synth,
    "real": err_real,
    "full_pipeline": err_full,
    "gradcheck_passed": bool(ok),
}
passed = (
    delta["synthetic"] <= tolerance["synthetic"]
    and delta["real"] <= tolerance["real"]
    and delta["full_pipeline"] <= tolerance["full_pipeline"]
    and delta["gradcheck_passed"]
)

def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

artifact = {
    "delta": delta,
    "tolerance": tolerance,
    "pass": bool(passed),
    "seed": SEED,
    "git_sha": _git_sha(),
    "notes": "Phase 9 EVAL-06: differentiable inverse Lambert W round-trip.",
}

out = Path("revision/results/eval06_roundtrip.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2))
print(json.dumps(artifact, indent=2))

assert passed, f"EVAL-06 FAILED: delta={delta} tolerance={tolerance}"
print("EVAL-06 round-trip PASSED")
```
Direct copy of the cell `a28db61d` shape: build `artifact` dict, `out.parent.mkdir(parents=True, exist_ok=True)`, `out.write_text(json.dumps(artifact, indent=2))`, then `assert passed` with informative message, closing `print("... PASSED")`.

**Round-trip + gradcheck core cell** (template from RESEARCH.md "Code Examples" lines 442–492):
```python
# Synthetic
x_synth = torch.randn(777, dtype=torch.float64, requires_grad=True)
delta_const = 0.146932  # verified Phase 8 baseline; or compute via load_and_preprocess
y = lambert_w_transform(x_synth, delta_const)
x_rt = inverse_lambert_w_transform(y, delta_const)
err_synth = (x_rt - x_synth).abs().max().item()

# Real
d = load_and_preprocess("./data.csv")
real = d["norm_log_delta"].double()
y_real = lambert_w_transform(real, d["delta"])
real_rt = inverse_lambert_w_transform(y_real, d["delta"])
err_real = (real_rt - real).abs().max().item()

# gradcheck
ok = torch.autograd.gradcheck(
    lambda v: inverse_lambert_w_transform(v, d["delta"]),
    (torch.randn(20, dtype=torch.float64, requires_grad=True),),
    eps=1e-6, atol=1e-6,
)

# Full-pipeline smoke (D-04b looser tolerance)
gen_windows = torch.randn(10, 10, dtype=torch.float64, requires_grad=True)
od_out = full_denorm_pipeline(
    gen_windows, d["transformed_norm_log_delta"], d["mu"].double(), d["sigma"].double(), d["delta"]
)
od_out.sum().backward()
assert gen_windows.grad is not None and not torch.isnan(gen_windows.grad).any()
err_full = 0.0  # gradient-flow smoke; numeric round-trip computed separately if added
```
Note: real-data length is **777** (RESEARCH OQ-1, OQ-2 verified live). Do NOT use 776.

---

### `revision/docs/training_protocol.md` (documentation) — CREATE

**Analog:** none in-repo (`revision/docs/` contains only `.gitkeep`). Top-level `README.md` is not the right shape (marketing prose, not paper-ready spec). **Use RESEARCH.md "training_protocol.md skeleton" at lines 494–567 as the literal template.**

**Citation pattern (D-09):** Every numerical value must include a `(\`revision/core/__init__.py:LINE\`)` source citation. Verified line numbers from live read of `revision/core/__init__.py`:

| Constant | Value | Source line |
|----------|-------|-------------|
| `N_CRITIC` | 9 | `revision/core/__init__.py:11` |
| `LAMBDA` | 2.16 | `revision/core/__init__.py:12` |
| `LR_CRITIC` | 1.8046e-05 | `revision/core/__init__.py:13` |
| `LR_GENERATOR` | 6.9173e-05 | `revision/core/__init__.py:14` |
| `NUM_QUBITS` | 5 | `revision/core/__init__.py:17` |
| `NUM_LAYERS` | 4 | `revision/core/__init__.py:18` |
| `WINDOW_LENGTH` | 10 | `revision/core/__init__.py:19` |
| `NUM_EPOCHS` | 2000 | `revision/core/__init__.py:20` |
| `BATCH_SIZE` | 12 | `revision/core/__init__.py:21` |
| `GEN_SCALE` | 1.0 | `revision/core/__init__.py:22` |
| `EVAL_EVERY` | 10 | `revision/core/__init__.py:23` |
| `DROPOUT_RATE` | 0.2 | `revision/core/__init__.py:24` |
| `DITHER` | 0.005 | `revision/core/__init__.py:27` |
| `DITHER_SEED` | 42 | `revision/core/__init__.py:28` |
| `PAR_LIGHT_MAX` | 12.5 | `revision/core/__init__.py:29` |
| `NOISE_LOW` | 0.0 | `revision/core/__init__.py:32` |
| `NOISE_HIGH` | 4π | `revision/core/__init__.py:33` |

Non-`__init__.py` citations (verified live):
- Adam betas=(0.0, 0.9): `revision/core/training.py:233-234`
- `torch.manual_seed(seed)`: `revision/core/training.py:211`
- `torch.cuda.manual_seed_all(seed)`: `revision/core/training.py:215`
- `seed: int = 42` default: `revision/core/training.py:188`
- `EarlyStopping(patience=50, warmup_epochs=100)`: `revision/core/training.py:96-97` (class defaults at lines 94–98)
- `shots=None` analytic statevector: `revision/core/models/quantum.py:64` (per RESEARCH citation)
- `diff_method="backprop"`: `revision/core/models/quantum.py:43, 76` (per RESEARCH citation)

**Section ordering** (Claude's discretion per CONTEXT.md):
Suggested order: (1) Optimizer & Schedule, (2) Early-Stopping, (3) Quantum Circuit, (4) Critic (1D-CNN), (5) Gradient Penalty, (6) Reproducibility, (7) Analytic-vs-Shot Distinction (D-10 statement).

**Hybrid format (D-08):** table for numerical values immediately followed by 1-paragraph prose justification. Use the RESEARCH skeleton (lines 503–567) verbatim, swapping in the verified line numbers above.

---

### `revision/docs/dataset_stats.md` (documentation) — CREATE

**Analog:** none in-repo. **Use RESEARCH.md "dataset_stats.md skeleton" at lines 571–616 as the literal template.**

**Verified live counts (RESEARCH OQ-1, OQ-2, OQ-3 — reconciled with reality, NOT with CONTEXT.md prose):**

| Quantity | Value | Source |
|----------|-------|--------|
| Raw CSV rows (data) | 778 | `wc -l data.csv` minus header; verified `load_and_preprocess` returns OD tensor of length 778 |
| OD rows post fillna+dropna | 778 | `revision/core/data.py:211-219` (fillna with 10-row rolling-mean, then dropna) |
| Log-return rows (N−1) | 777 | `revision/core/data.py:62` (`log_od[1:] - log_od[:-1]`) |
| Rolling windows (m=10, s=2) | 384 | `(777 − 10) // 2 + 1 = 384`; `revision/core/data.py:110-118` |
| Independent campaigns | 1 | LUCY bioreactor, single run |
| Sampling cadence | 10 minutes | `data.csv` consecutive DATE deltas |
| Start date | 2024-03-27 13:12 | first row of `data.csv` |
| End date | 2024-03-31 23:52 | last row of `data.csv` (≈ 4.5 days) |

**CONTEXT.md says "777 OD / 384 windows"; live pipeline says "778 OD / 777 log_delta / 384 windows".** Honor the live values per RESEARCH OQ-1/2/3.

**Hybrid format (D-08):** counts table → "Single-Campaign Limitation" 1-paragraph prose block → sampling/date-range table → split-convention table → preprocessing-pipeline-reference subsection → PAR_LIGHT note.

**Single-Campaign Limitation prose anchor (D-02 wording is Claude's discretion):** must acknowledge (a) no train/val/test split, (b) EMD early-stop uses same distribution it compares against (R1-M5 honest framing), (c) multi-campaign generalization → Phase 14 Outlook only.

**PAR_LIGHT note pattern** (from RESEARCH lines 611–615): single line stating `RUN_NAME = "unconditioned_wgan"` disabled conditioning; PAR_LIGHT captured but reserved for Phase 13 conditional-generation introspection.

## Shared Patterns

### Float64 Promotion at Autograd Boundary
**Source:** `revision/core/data.py:80, 101` (both forward and inverse currently call `.double()` on entry)
**Apply to:** New `_InverseLambertW.forward` (promote on entry, cache float64 tensors via `ctx.save_for_backward`, cast `grad_data.to(grad_output.dtype)` in backward).
```python
data = data.double()                        # entry: float32 → float64
# ... compute ...
# backward end:
return grad_data.to(grad_output.dtype), None
```
**Why:** 1e-8 tolerance is unreachable in float32; pattern matches existing forward at line 80.

### Device Preservation Across scipy Boundary
**Source:** `revision/core/data.py:85`
```python
lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
```
**Apply to:** New autograd Function — preserve `data.device` after `torch.from_numpy(...)` so caller's MPS/CUDA tensor doesn't get pinned to CPU (Pitfall 3, RESEARCH lines 408–412).

### Section-Header Box-Drawing Comment Style
**Source:** `revision/core/data.py:24-26, 43-45, 65-67, 107-109, 131-133, 165-167, 184-186` (used 7 times across the module)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell NN — section name
# ─────────────────────────────────────────────────────────────────────────────
```
**Apply to:** `preprocessing.py` (3 section headers, one per pipeline variant), and any new section added to `data.py` for `_InverseLambertW` (keep under existing "Cell 17 — Lambert W transforms" banner; do not introduce a new one).

### Notebook Repo-Root Finder + nbconvert Shim
**Source:** `revision/01_parity_check.ipynb` cell `2c8bc6c2` lines 19–46
**Apply to:** `revision/02_eval06_roundtrip.ipynb` first cell (verbatim copy, including `os.chdir(REPO_ROOT)` + `sys.path.insert`).
**Why:** nbconvert sets CWD to notebook dir; without this shim `data.csv` and `revision/core/` won't resolve.

### JSON Artifact Schema (Phase 8 lineage)
**Source:** `revision/01_parity_check.ipynb` cell `a28db61d` — see `revision/results/parity_check.json` for the realized shape:
```json
{
  "delta": {...},
  "tolerance": {...},
  "pass": true,
  "seed": 42,
  "git_sha": "<sha>",
  "notes": "..."
}
```
**Apply to:** `revision/results/eval06_roundtrip.json` — same top-level keys (`delta`, `tolerance`, `pass`, `seed`, `git_sha`, `notes`). Adds verification-specific deltas (`synthetic`, `real`, `full_pipeline`, `gradcheck_passed`).

### "Source of Truth" Citation Footer for Docs (D-09)
**Source:** D-09 mandates traceability to `revision/core/__init__.py`. No in-repo doc analog — pattern is defined by Phase 9 itself.
**Apply to:** Both `training_protocol.md` and `dataset_stats.md` — top-of-file blockquote naming the source file once, then per-row `(\`<path>:<line>\`)` citations in tables.
```markdown
> **Source of truth:** all numerical constants below are imported from
> `revision/core/__init__.py`. Update that file to change them; this doc
> tracks the file via the line-cited references in the table.
```

### Module Registration in `revision/core/__init__.py`
**Source:** `revision/core/__init__.py:35-39`
```python
from revision.core import data, eval, training  # noqa: F401,E402
from revision.core import models  # noqa: F401,E402

__all__ = [
    "data", "eval", "training", "models",
    ...
]
```
**Apply to:** Phase 9 adds `preprocessing` to both the import line and `__all__`:
```python
from revision.core import data, eval, training, preprocessing  # noqa: F401,E402
__all__ = ["data", "eval", "training", "models", "preprocessing", ...]
```

## No Analog Found

| File | Role | Data Flow | Reason | Fallback |
|------|------|-----------|--------|----------|
| `revision/docs/training_protocol.md` | documentation | static | `revision/docs/` is empty (`.gitkeep` only); top-level `README.md` is marketing prose, wrong shape; `archive/*.md` files are phase-result notes, not paper methods specs | Use RESEARCH.md skeleton (lines 494–567) verbatim as template |
| `revision/docs/dataset_stats.md` | documentation | static | same as above | Use RESEARCH.md skeleton (lines 571–616) verbatim as template |
| `class _InverseLambertW(torch.autograd.Function)` in `data.py` | autograd Function subclass | autograd | `grep -r "torch.autograd.Function"` returned 0 hits in `revision/` and 0 hits in `qgan_pennylane.ipynb` — this is the first such class in the repo | Use RESEARCH.md "Pattern 1" skeleton (lines 193–258) which cites the canonical PyTorch docs idiom |

## Metadata

**Analog search scope:** `revision/core/`, `revision/`, top-level `*.md`, `archive/*.md`
**Codebase grep hits:**
- `torch.autograd.Function`: 0 in-repo hits (verified `grep -r --include="*.py" --include="*.ipynb"`)
- `revision/docs/`: empty (only `.gitkeep`)
- existing Lambert W code: `revision/core/data.py:68-104` (both directions)
- notebook pattern: `revision/01_parity_check.ipynb` (~270 lines, fully read)
- canonical constants: `revision/core/__init__.py` (45 lines, fully read)

**Pattern extraction date:** 2026-05-11

## PATTERN MAPPING COMPLETE
