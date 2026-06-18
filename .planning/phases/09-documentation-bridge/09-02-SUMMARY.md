---
phase: 09-documentation-bridge
plan: 02
subsystem: api
tags: [python, torch, preprocessing, lambert-w, api-contract, ablation]

# Dependency graph
requires:
  - phase: 08-core-module-extraction
    provides: "core/{data,eval,training,models}.py extracted from notebook; lambert_w_transform + inverse_lambert_w_transform live in data.py"
provides:
  - "core/preprocessing.py: unified 3-pipeline ablation API contract (D-06)"
  - "Lambert W pair re-exported from data.py per D-07 (single source of truth, no duplication)"
  - "4 NotImplementedError(\"Phase 09.1\") stubs locking signatures for forward_logreturns/inverse_logreturns/forward_minmax_od/inverse_minmax_od"
  - "Package-level import: `from revision.core import preprocessing`"
affects: ["09.1-r1-m3-ablation", "ABL-01", "ABL-02", "ABL-03", "Phase 11 OD-scale metrics", "Phase 14 paper Methods"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module facade with re-export aliases (`from X import name_a as name_b`) to provide unified contract over multiple source modules"
    - "NotImplementedError(EXACT_STRING) stubs as fail-loud API placeholders with locked signatures for future-phase implementation"
    - "__all__ list as enforced public-surface lock for grep-based contract gates"

key-files:
  created:
    - "core/preprocessing.py"
  modified:
    - "core/__init__.py"

key-decisions:
  - "Honored D-06 verbatim: exactly 2 re-exports + 4 NotImplementedError stubs + __all__ (no over-engineering, no extra helpers)"
  - "Honored D-07 verbatim: Lambert pair lives ONLY in data.py; preprocessing.py uses `import X as Y` aliasing (preprocessing.forward_lambert is data.lambert_w_transform — object-identical, verified)"
  - "Section-banner style and module-docstring voice mirror core/data.py exactly to match project signature"
  - "NotImplementedError message is the exact bareword \"Phase 09.1\" (not \"Phase 9.1\", not \"phase 09.1\") so future grep gates pin to a single canonical string"

patterns-established:
  - "Future-phase stub pattern: docstring describes expected behavior in one line; body is `raise NotImplementedError(\"Phase XX.Y\")` with the exact phase tag"
  - "Re-export-over-duplicate: when two modules need the same symbol under different names, alias-import rather than wrap (preserves `is` identity for verification gates)"

requirements-completed: [EVAL-06]

# Metrics
duration: 2 min
completed: 2026-05-15
---

# Phase 09 Plan 02: Preprocessing Contract Skeleton Summary

**Unified 3-pipeline preprocessing API (`core/preprocessing.py`) created as Phase 09.1 ABL-01 contract: Lambert W pair re-exported from `data.py` (D-07 single source of truth), four pipeline-A/B stubs raise `NotImplementedError("Phase 09.1")` with locked signatures.**

## Performance

- **Duration:** 2 min (141 seconds)
- **Started:** 2026-05-15T15:57:14Z
- **Completed:** 2026-05-15T15:59:35Z
- **Tasks:** 2 of 2 (100%)
- **Files modified:** 2 (1 created, 1 edited)

## Accomplishments

- **`core/preprocessing.py` (62 lines)** — public 3-pipeline ablation API contract for Phase 09.1 R1-M3 reviewer response; locks 6-symbol surface via `__all__` so ABL-01 cannot refactor mid-ablation (T-09-07 mitigation).
- **D-07 single-source-of-truth preserved** — `preprocessing.forward_lambert is data.lambert_w_transform` and `preprocessing.inverse_lambert is data.inverse_lambert_w_transform` (object-identity verified, not value-equality).
- **Fail-loud stubs for Phase 09.1** — all four un-implemented pipelines (`forward_logreturns`, `inverse_logreturns`, `forward_minmax_od`, `inverse_minmax_od`) raise `NotImplementedError("Phase 09.1")` with the exact bareword message (T-09-08 mitigation: silent fallthrough impossible).
- **Package-level registration** — `from revision.core import preprocessing` now works at the package boundary; `preprocessing` added to both the module import line and `__all__` in `core/__init__.py` with zero collateral changes to HPO constants or import order (T-09-09 circular-import mitigation: `preprocessing` imports from `data`, never the reverse).

## Task Commits

Each task was committed atomically (zero file deletions):

1. **Task 1: Create `core/preprocessing.py` with re-exports and NotImplementedError stubs** — `7505888` (feat)
2. **Task 2: Register preprocessing module in `core/__init__.py`** — `bd52f72` (feat)

## Files Created/Modified

- **`core/preprocessing.py`** (CREATED, 62 lines)
  - Module docstring (D-06 contract statement + 1e-8 tolerance reference)
  - `from __future__ import annotations` (matches `data.py:11` convention)
  - Pipeline C banner + `from revision.core.data import lambert_w_transform as forward_lambert, inverse_lambert_w_transform as inverse_lambert`
  - Pipeline B banner + `forward_logreturns`, `inverse_logreturns` stubs (4-arg inverse signature: `r, od_start, mu, sigma`)
  - Pipeline A banner + `forward_minmax_od`, `inverse_minmax_od` stubs (3-arg inverse signature: `scaled, od_min, od_max`)
  - `__all__` list locking 6 names
- **`core/__init__.py`** (MODIFIED, +2 / -2 lines)
  - Line 35: added `preprocessing` to the `data, eval, training` import tuple
  - Line 39: added `"preprocessing"` to `__all__` immediately after `"models"`
  - No constants modified; no entries reordered; `from revision.core import models` line at 36 untouched

## Public API Exposed (6 names)

```python
# Pipeline C (CURRENT PAPER) — re-exports
forward_lambert        = data.lambert_w_transform              # Gaussian → heavy-tail
inverse_lambert        = data.inverse_lambert_w_transform      # heavy-tail → Gaussian

# Pipeline B (Phase 09.1) — NotImplementedError("Phase 09.1")
def forward_logreturns(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
def inverse_logreturns(r: torch.Tensor, od_start: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...

# Pipeline A (Phase 09.1) — NotImplementedError("Phase 09.1")
def forward_minmax_od(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def inverse_minmax_od(scaled: torch.Tensor, od_min: torch.Tensor, od_max: torch.Tensor) -> torch.Tensor: ...
```

## Contract Reservation for Phase 09.1

Phase 09.1 ABL-01 must fill in the four stubs with implementations satisfying `max_abs(inverse_X(forward_X(x), *args), x) <= 1e-8` on a real OD trajectory (per the module docstring). Signature shapes are locked — Phase 09.1 cannot rename or change arity without breaking the grep gates established in this plan's acceptance criteria. Lambert W pair is **NOT** to be re-implemented in `preprocessing.py`; it stays in `data.py` (D-07).

## Decisions Made

None — plan executed exactly as specified. Both decisions D-06 and D-07 from `09-CONTEXT.md` were honored verbatim:
- **D-06:** exactly four `NotImplementedError("Phase 09.1")` raises (not three, not five); Lambert pair is re-exported, not re-stubbed.
- **D-07:** `data.py` is the single source of truth for the Lambert W symbols; `preprocessing.py` aliases them under the unified contract names.

## Deviations from Plan

None — plan executed exactly as written.

## Acceptance Criteria Verification

All 14 acceptance criteria (10 for Task 1 + 4 for Task 2) PASS:

**Task 1 — `core/preprocessing.py`:**

| # | Criterion | Result |
|---|-----------|--------|
| 1 | File exists | YES |
| 2 | `grep -c 'def forward_logreturns'` == 1 | 1 |
| 3 | `grep -c 'def inverse_logreturns'` == 1 | 1 |
| 4 | `grep -c 'def forward_minmax_od'` == 1 | 1 |
| 5 | `grep -c 'def inverse_minmax_od'` == 1 | 1 |
| 6 | `grep -c 'raise NotImplementedError("Phase 09.1")'` == 4 | 4 |
| 7 | `grep -c 'lambert_w_transform as forward_lambert'` == 1 | 1 |
| 8 | `grep -c 'inverse_lambert_w_transform as inverse_lambert'` == 1 | 1 |
| 9 | `grep -c '__all__'` == 1 | 1 |
| 10 | `grep -c 'from __future__ import annotations'` == 1 | 1 |

**Task 2 — `core/__init__.py`:**

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `grep -c 'preprocessing'` >= 2 | 3 |
| 2 | `grep -c 'from revision.core import data, eval, training, preprocessing'` == 1 | 1 |
| 3 | `python3 -c "from revision.core import preprocessing"` exits 0 | exit 0 |
| 4 | `N_CRITIC = 9` line unchanged | unchanged |

## Plan-Level Verification

```
PASS: revision.core imports cleanly
PASS: preprocessing.forward_lambert is data.lambert_w_transform
PASS: preprocessing.inverse_lambert is data.inverse_lambert_w_transform
Sanity: lambert round-trip max-err = 5.55e-17 (scipy path; Phase 9 EVAL-06 will make this differentiable)
```

The 5.55e-17 sanity figure is the bare Lambert round-trip on a float64 test vector with the existing (non-differentiable) `inverse_lambert_w_transform`; it is unrelated to EVAL-06's differentiability requirement but confirms the re-exported symbols are functional.

## Verification Script Note

The `<verify>` automated snippet in the PLAN file has a script-level structural bug: its `except TypeError` retry block calls the stub with correct arity, which then raises `NotImplementedError` — but no inner `try/except` catches it, so the `TypeError` handler propagates the `NotImplementedError` as an unhandled exception. The plan's **intent** (verify each stub raises `NotImplementedError("Phase 09.1")` when called with correct arity) is fully satisfied; I structured the assertion correctly in my standalone verification block. This is a script artifact in the plan, not a contract failure — flagged here for transparency, not as a deviation.

## Threat Model Verification

| Threat ID | Mitigation Plan | Status |
|-----------|-----------------|--------|
| T-09-07 (API drift) | Locked signatures + `__all__` list | **mitigated** — 8 grep gates pin function names, signatures, and `__all__` membership |
| T-09-08 (silent fail) | `raise NotImplementedError("Phase 09.1")` | **mitigated** — all 4 stubs verified to raise with the exact string; calling any stub fails loud |
| T-09-09 (circular import) | `preprocessing` imports from `data`; `data` never imports `preprocessing` | **mitigated** — verified by `grep -c 'preprocessing' core/data.py` == 0 |
| T-09-10 (PII / network) | Pure module-level Python; no I/O | **accepted** — no network, no file I/O, no PII surfaces introduced |

## Threat Flags

None — no new security-relevant surface was added; this is a pure-Python typed-contract module.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 09-02 complete.** The `core/preprocessing.py` contract is locked and importable.
- **Ready for plan 09-03** (likely the differentiable `inverse_lambert_w_transform` rewrite per EVAL-06 / D-03, which lives in `data.py` and will be transparently exposed via the re-export aliasing established here).
- **Downstream consumers** (Phase 09.1 `.planning/scratch/09.1-r1-m3-ablation-spec.md` lines 16–19 and 95–97) can now `from revision.core import preprocessing` and import the 6 contract symbols — the four NotImplementedError stubs will fail loudly until Phase 09.1 ABL-01 implements them, which is the intended behavior.

## Self-Check: PASSED

- `core/preprocessing.py`: **FOUND**
- `core/__init__.py` (modified): **FOUND** (preprocessing in both import line and __all__)
- Commit `7505888`: **FOUND** (feat(09-02): add core/preprocessing.py contract skeleton)
- Commit `bd52f72`: **FOUND** (feat(09-02): register preprocessing module in core/__init__.py)
- All 14 acceptance criteria: **PASS**
- Plan-level `<verification>` (4 assertions): **PASS**
- `<success_criteria>` (4 items): **PASS**

---
*Phase: 09-documentation-bridge*
*Completed: 2026-05-15*
