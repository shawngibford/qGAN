---
phase: 12-sensitivity-analysis
plan: 03
subsystem: testing
tags: [python, stdlib, json, aggregation, multiseed, sens-03]

requires:
  - phase: 10-headline-results
    provides: five frozen headline JSONs (baseline_comparison, tstr, predictive_discriminative, augmentation, fidelity_dualscale) with shared data_hash 91e447d4624e25b3
provides:
  - run_multiseed_rollup.py — SENS-03 pure stdlib aggregator
  - results/multiseed_summary.json — 5-seed mean±std roll-up (1266 cells)
affects: [sensitivity-analysis, manuscript-revision, verification]

tech-stack:
  added: []
  patterns:
    - "Pure stdlib aggregator: no torch/pennylane/core imports; single invocation, no args"
    - "Cross-artifact data_hash hard gate (D-10-15) before any roll-up math"
    - "Faithful source-null propagation: value:null cells emitted as mean/std null + n_null provenance, never fabricated or dropped"

key-files:
  created:
    - run_multiseed_rollup.py
    - results/multiseed_summary.json
  modified: []

key-decisions:
  - "D-11-09 N/A cells (fidelity_dualscale Pipeline-A log_return, value:null) are excluded from mean/std math and emitted as null with n=0 + n_null, not dropped — preserves grid completeness without fabricating statistics"
  - "Plan's literal Task-2 verify assertion (all rollup cells numeric) is a plan defect; superseded by faithful null propagation (see Deviations)"

patterns-established:
  - "Source-null propagation: aggregator never invents a mean for an upstream value:null cell; records n_null for provenance"

requirements-completed: [SENS-03]

duration: ~25min
completed: 2026-05-18
---

# Phase 12: Sensitivity Analysis — Plan 03 (Multiseed Rollup) Summary

**SENS-03 pure stdlib aggregator emitting `multiseed_summary.json`: 1266 mean±std cells over the 5-seed set {42,43,44,45,46}, gated on the canonical cross-artifact data_hash `91e447d4624e25b3`, with faithful propagation of D-11-09 N/A cells.**

## Performance

- **Duration:** ~25 min (including stall-recovery finalization)
- **Started:** 2026-05-18T12:28:01Z
- **Completed:** 2026-05-18T15:28:05Z
- **Tasks:** 2
- **Files modified:** 2 (1 script, 1 generated artifact)

## Accomplishments
- `run_multiseed_rollup.py`: stdlib-only aggregator, no device/model/torch/pennylane imports
- Hard cross-artifact data_hash assertion (D-10-15) — all five headline JSONs verified mutually equal to `91e447d4624e25b3`
- `multiseed_summary.json`: 1266 cells, full schema per cell, seed_set [42,43,44,45,46], 5-seed headline cells report n==5
- Faithful handling of 168 D-11-09 N/A cells (fidelity_dualscale Pipeline-A log_return) as null with `n_null` provenance

## Task Commits

1. **Task 1: Build run_multiseed_rollup.py (resolver, data_hash gate, groupby roll-up)** — `dd6bd64` (feat)
2. **Task 2: Run aggregator + verify multiseed_summary.json** — `0c47844` (feat)

## Files Created/Modified
- `run_multiseed_rollup.py` — SENS-03 pure aggregator: repo resolver, cross-artifact data_hash hard gate, long-form groupby roll-up with null-safe mean/std
- `results/multiseed_summary.json` — 1266-cell roll-up: provenance header (schema, data_hash, consumed_artifacts, seed_set) + per-cell {source, model_kind, pipeline, metric_name, scale, injection_ratio, mean, std, n, n_null, seeds}

## Decisions Made
- **D-11-09 N/A propagation:** `fidelity_dualscale.json` carries `value: null` with `scale_na_reason` for Pipeline-A log_return rows (no log-return space for raw-OD Pipeline A). The aggregator excludes nulls from `statistics.fmean`/`stdev` (which raise on `None`), emits the cell with `mean/std = null, n = 0, n_null = <count>`. This preserves grid completeness (no headline cell silently dropped) without fabricating statistics — the scientifically correct behavior for a manuscript deliverable.
- `core/` untouched (D-10-13) — verified empty `git diff --stat`.

## Deviations from Plan

### Plan defect — too-strict Task-2 verify assertion (NOT auto-fixed in code; resolved by design)

**1. [Plan defect] Task-2 automated verify asserts every rollup cell has numeric mean/std**
- **Found during:** Task 2 (stall-recovery verification)
- **Issue:** The plan's literal Task-2 `<automated>` assertion `all(isinstance(c['mean'],(int,float)) ... for c in s['rollup'])` does not account for the 168 legitimately-null D-11-09 cells. `fidelity_dualscale.json` itself emits `value: null` (with explicit `scale_na_reason`) for Pipeline-A log_return — these have no defined fidelity metric upstream.
- **Resolution:** The implementation correctly propagates source nulls rather than fabricating means or dropping cells. The plan's "no headline cell silently dropped" intent and D-11-09 are satisfied; the literal numeric-everywhere assertion is the defect and is superseded. All other Task-2 assertions pass: data_hash `91e447d4624e25b3`, consumed_artifacts (all 5 → canonical hash), seed_set [42,43,44,45,46], full schema, ≥1 five-seed headline cell with n==5.
- **Verification:** 1098 numeric cells + 168 source-null cells = 1266 total; null cells exclusively `fidelity_dualscale.json` / `scale: log_return` / Pipeline A, each with upstream `scale_na_reason` "Pipeline A is raw-OD; no log-return space (D-11-09)".

---

**Total deviations:** 1 (plan-verify defect, resolved by faithful null propagation — the correct scientific behavior)
**Impact on plan:** No scope creep. The deliverable is more correct than a numeric-everywhere roll-up would be. Verifier should treat the plan's literal all-numeric assertion as superseded by D-11-09.

## Issues Encountered
- The Wave-1 executor agent was killed by the 600s stall watchdog while committing Task 2. Recovery: Task-2 work (script change + generated artifact) was intact in the worktree; orchestrator verified correctness, committed Task 2, and authored this SUMMARY. The aggregator change itself was complete and correct.

## User Setup Required
None — pure offline stdlib aggregation of existing frozen artifacts.

## Next Phase Readiness
- SENS-03 deliverable `multiseed_summary.json` complete and parallel-safe (zero overlap with Plans 01/02).
- No blockers for the remaining sensitivity-analysis plans.

---
*Phase: 12-sensitivity-analysis*
*Completed: 2026-05-18*
