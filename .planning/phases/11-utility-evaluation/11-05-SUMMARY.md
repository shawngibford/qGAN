---
phase: 11-utility-evaluation
plan: 05
subsystem: testing
tags: [gap-closure, wr-01, wr-02, wr-03, wr-04, run-utility, reproducibility, eval-01, eval-04]

# Dependency graph
requires:
  - phase: 11-utility-evaluation
    provides: "11-01 run_utility.py (EVAL-01 TSTR + EVAL-04 augmentation driver)"
provides:
  - "run_utility.py — hardened: derived shape comment, NaN-on-degenerate r2, collision-free (mk,p,label)-qualified subsample seed, grid-collapse assertion"
affects: [14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-loud non-degeneracy guards over silent plausible-looking fallbacks"
    - "Run-identity-qualified crc32 subsample seeds (collision-free, recorded + self-describing)"

key-files:
  created: []
  modified:
    - run_utility.py

key-decisions:
  - "WR-02 assert written as eval_windows[:, 9:10].std() > 0.0 (no float() wrapper) to satisfy the plan's own verify regex; numpy float comparison is semantically identical (Rule-4 minor, plan-internal-consistency)"
  - "WR-04 assert placed before the pre-existing n_synth>=pool guard; that branch is now defensively dead but left in place (plan did not request removal)"

patterns-established:
  - "subsample_rng_seed + subsample_rng_seed_derivation sibling fields make every emitted augmentation seed self-explaining"

requirements-completed: [EVAL-01, EVAL-04]

# Metrics
duration: ~10min
completed: 2026-05-18
---

# Phase 11 Plan 05: WR-01..04 Correctness/Reproducibility Hardening

**Closed the four localized correctness/reproducibility warnings in `run_utility.py` (stale shape comment, degenerate-R2 masking, lossy subsample seed, unguarded injection-grid collapse) with the data_hash invariant and the 22-test suite preserved.**

## Performance

- **Duration:** ~10 min (inline orchestrator execution — subagents were Bash-denied this session)
- **Started:** 2026-05-18 (base 1f1c186)
- **Completed:** 2026-05-18
- **Tasks:** 2
- **Files modified:** 1 (`run_utility.py`)

## Accomplishments

- **WR-01:** Replaced the misleading `# (384,10)` literal on `_real_windowed_od` with a derived annotation stating `((len(OD)-WINDOW_LENGTH)//2 + 1, WINDOW_LENGTH) == (385,10)` and the `HELD_OUT_N=320 → 65` training-window invariant. The `(384,10)` literal no longer appears anywhere in the file.
- **WR-02:** `r2_score_inline` now returns `float("nan")` (matching sklearn `r2_score`) instead of a plausible-looking `0.0` that defeats the strict `<0` leakage sentinel; `train_eval_tstr` asserts `eval_windows[:, 9:10].std() > 0.0` before training so a future degenerate slice fails at its true cause. Current data is non-degenerate → every produced number is byte-identical.
- **WR-03:** Subsample RNG is now `np.random.default_rng(zlib.crc32(f"augsub|{mk}|{p}|{label}".encode()) & 0xFFFFFFFF)` — collision-free and fully qualified by run identity, replacing the lossy `int(ratio*1000)+1`. The exact integer is recorded in `subsample_rng_seed` with a new self-describing `subsample_rng_seed_derivation` sibling field.
- **WR-04:** `assert n_synth < synth_pool.shape[0]` inside the `_INJECTION_GRID` loop makes a shrunken pool fail loudly instead of silently collapsing `+100%` into `synthetic_only`. Always passes at the documented pool size (~3840).
- **Invariants held:** `git diff --stat -- core/` empty after every task; `tstr.json`/`augmentation.json` on-disk `data_hash` still `91e447d4624e25b3` (seed governs only synthetic subsampling, not any data_hash input); `pytest tests/ -q` → 22 passed incl. `test_no_leakage_sentinel`.

## Task Commits

Each task was committed atomically:

1. **Task 1: WR-01 shape comment + WR-02 degenerate-R2 masking** - `0db6c11` (fix)
2. **Task 2: WR-03 lossy subsample seed + WR-04 grid collapse** - `606a949` (fix)

## Files Created/Modified

- `run_utility.py` - Hardened: derived shape comment; NaN-on-degenerate `r2_score_inline` + pre-train non-degeneracy assert; crc32 `(mk,p,label)`-qualified subsample seed + derivation field; injection-grid-collapse assert.

## Decisions Made

- **WR-02 assert form:** wrote `assert eval_windows[:, 9:10].std() > 0.0` without the plan-action's `float(...)` wrapper, because the plan's own `<verify>` regex (`eval_windows\[:, 9:10\]\.std\(\) > 0`) and acceptance criterion require that exact substring. Numpy scalar comparison is semantically identical. Minor Rule-4 deviation resolving an internal plan inconsistency.
- **WR-04 placement:** assertion sits before the pre-existing `if n_synth >= synth_pool.shape[0]` guard, which is now defensively dead code. Left in place — the plan only asked to add the assert, not to prune the branch.

## Deviations from Plan

- WR-02 assertion expression simplified to drop the redundant `float()` cast (see Decisions). No behavioral change; satisfies the plan's verify gate and acceptance criteria exactly.

## Self-Check: PASSED

- `! grep -nF '(384,10)'` → no match ✓
- `grep 'float("nan")'` → present (line 221) ✓
- `grep 'eval_windows[:, 9:10].std() > 0'` → present (line 233) ✓
- `grep crc32 augsub derivation` + `! grep 'int(ratio * 1000) + 1'` → present / absent ✓
- `grep 'assert n_synth < synth_pool.shape[0]'` → present (line 457) ✓
- `git diff --stat -- core/` → empty ✓
- `data_hash` in tstr.json + augmentation.json → `91e447d4624e25b3` ✓
- `pytest tests/ -q` → 22 passed ✓
