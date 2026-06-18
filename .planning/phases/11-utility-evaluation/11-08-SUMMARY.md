---
phase: 11-utility-evaluation
plan: 08
subsystem: testing
tags: [gap-closure, wr-06, test-timegan-scores, determinism, pytest-collected, eval-03]

# Dependency graph
requires:
  - phase: 11-utility-evaluation
    provides: "11-07 single-Generator discriminative_score (the contract this test pins)"
provides:
  - "tests/test_timegan_scores.py — pytest-collected test_discriminative_score_deterministic locking discriminative_score determinism"
affects: [14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Determinism locked by a pytest-COLLECTED def test_..., not only an __main__ smoke block"

key-files:
  created: []
  modified:
    - tests/test_timegan_scores.py

key-decisions:
  - "Exact-equality assertion (a == b), not tolerance — the 11-07 single-Generator fix makes the path bit-deterministic, so the test would fail against the old mixed-global-RNG implementation"

patterns-established:
  - "Every risky stochastic scoring path gets a collected determinism test mirroring test_scores_deterministic"

requirements-completed: [EVAL-03]

# Metrics
duration: ~4min
completed: 2026-05-18
---

# Phase 11 Plan 08: WR-06 Collected discriminative_score Determinism Test

**Added `test_discriminative_score_deterministic` to the pytest-collected suite, locking the discriminative path's determinism (proving the 11-07 single-Generator fix) — the suite that "locks the invariants" now actually covers the riskier RNG path.**

## Performance

- **Duration:** ~4 min (inline orchestrator execution — subagents were Bash-denied this session)
- **Started:** 2026-05-18 (base 1f1c186)
- **Completed:** 2026-05-18
- **Tasks:** 1
- **Files modified:** 1 (`tests/test_timegan_scores.py`)

## Accomplishments

- **WR-06 closed:** new module-level `def test_discriminative_score_deterministic()` (pytest-collected, not only in `__main__`) builds fixed-size seeded float64 arrays, calls `discriminative_score(real, synth, 42, 10, iters=40, bs=32)` twice, and asserts exact equality (`a == b`) plus a finite + `[0.0, 0.5 + 1e-6]` range check. It would fail against the old mixed-global-RNG implementation and passes against the 11-07 single-Generator fix.
- `test_discriminative_score_deterministic()` also appended to the `if __name__ == "__main__"` smoke block.
- **Untouched:** `run_timegan_scores.py` and `core/` — test-only addition, no driver/data change.
- **Invariants held:** `git diff --stat -- core/ run_timegan_scores.py` empty; all 4 result JSON `data_hash` remain `91e447d4624e25b3`; isolated test passes; `pytest tests/ -q` → **23 passed** (strictly greater than the prior 22).

## Task Commits

1. **Task 1: add test_discriminative_score_deterministic to the collected pytest suite** - `27c8440` (test)

## Files Created/Modified

- `tests/test_timegan_scores.py` - Added pytest-collected `test_discriminative_score_deterministic` (exact-equality + range) and its `__main__` smoke call.

## Decisions Made

- **Exact equality, not tolerance:** the 11-07 fix makes the discriminative path bit-deterministic, so `assert a == b` is both correct and a strictly stronger regression guard than an approximate check — and is precisely what would have caught the WR-05 defect.

## Deviations from Plan

- None. Test mirrors `test_scores_deterministic` structure exactly as specified; sizes/iters kept small (well under 60s).

## Self-Check: PASSED

- `grep 'def test_discriminative_score_deterministic'` → present (line 61) ✓
- exact-equality `a == b` + finite + `[0, 0.5+1e-6]` asserts ✓
- `__main__` block calls `test_discriminative_score_deterministic()` (line 84) ✓
- `pytest ...::test_discriminative_score_deterministic -q` → 1 passed ✓
- `git diff --stat -- core/ run_timegan_scores.py` → empty ✓
- all 4 result JSON data_hash `91e447d4624e25b3` ✓
- `pytest tests/ -q` → 23 passed (> 22) ✓
