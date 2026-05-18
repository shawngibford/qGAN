---
phase: 11-utility-evaluation
plan: 06
subsystem: testing
tags: [gap-closure, cr-01, run-dualscale-fidelity, reproducibility, portability, human-uat, eval-05]

# Dependency graph
requires:
  - phase: 11-utility-evaluation
    provides: "11-03 run_dualscale_fidelity.py (EVAL-05 dual-scale fidelity driver)"
provides:
  - "revision/run_dualscale_fidelity.py — portable: QGAN_CANONICAL_REPO opt-in resolver, fail-loud _resolve_run_dir, single-root provenance assertion"
affects: [14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Env-var opt-in fallback (no machine-specific path baked into a reproducibility artifact)"
    - "Single-root provenance assertion guarding cross-checkout artifact mixing"

key-files:
  created: []
  modified:
    - revision/run_dualscale_fidelity.py

key-decisions:
  - "Provenance recorded via a module-level _RESOLVED_ROOTS set populated inside _resolve_run_dir; asserted at end of emit_rows (the single emission boundary) before any JSON write in main()"

patterns-established:
  - "Reproducibility drivers must fail loudly with actionable env-var guidance off-box rather than silently resolving a baked-in home path"

requirements-completed: [EVAL-05]

# Metrics
duration: ~8min
completed: 2026-05-18
---

# Phase 11 Plan 06: CR-01 Portable Canonical-Repo Resolver

**Removed the hardcoded home-directory checkout path from `revision/run_dualscale_fidelity.py`, replaced it with an opt-in `QGAN_CANONICAL_REPO` env-var resolver that fails loudly with guidance, and added a single-root provenance assertion — closing the one critical (CR-01) finding and the open HUMAN-UAT reproducibility blocker.**

## Performance

- **Duration:** ~8 min (inline orchestrator execution — subagents were Bash-denied this session)
- **Started:** 2026-05-18 (base 1f1c186)
- **Completed:** 2026-05-18
- **Tasks:** 1
- **Files modified:** 1 (`revision/run_dualscale_fidelity.py`)

## Accomplishments

- **CR-01 closed:** `_CANONICAL_REPO_FALLBACK` is now `Path(os.environ["QGAN_CANONICAL_REPO"]).resolve()` when the env var is set, else `None`. The literal `/Users/shawngibford/dev/phd/qGAN` no longer appears anywhere in the file — the driver is portable across machines/CI.
- **Fail-loud resolver:** `_resolve_run_dir` returns the in-tree path if present; else, only if `QGAN_CANONICAL_REPO` is set, tries the fallback; else raises `FileNotFoundError` whose message names the missing in-tree path, instructs the operator to set `QGAN_CANONICAL_REPO`, and cites the D-11-08 no-regeneration rule. Verified functionally: env unset → `_CANONICAL_REPO_FALLBACK is None` and the raised message contains both `QGAN_CANONICAL_REPO` and `D-11-08`.
- **Single-root provenance guard:** a module-level `_RESOLVED_ROOTS` set records each resolved checkout root; `emit_rows` asserts `len(_RESOLVED_ROOTS) <= 1` after the 60-run loop and before any JSON is written, with a message naming the mixed checkouts. Cross-checkout artifact mixing can no longer pass silently.
- **HUMAN-UAT test 1** (cross-machine reproducibility) is now resolvable: portable + fail-loud + provenance-asserted.
- **Invariants held:** `git diff --stat -- revision/core/` empty; `fidelity_dualscale.json` `data_hash` still `91e447d4624e25b3` with 3360 rows; `pytest revision/tests/ -q` → 22 passed.

## Task Commits

1. **Task 1: QGAN_CANONICAL_REPO resolver + fail-loud _resolve_run_dir + single-root assertion** - `61c4eb4` (fix)

## Files Created/Modified

- `revision/run_dualscale_fidelity.py` - Env-var opt-in canonical-repo resolver (no baked path), fail-loud `_resolve_run_dir` with actionable guidance, `_RESOLVED_ROOTS` single-root provenance assertion in `emit_rows`.

## Decisions Made

- **Provenance recording site:** populated `_RESOLVED_ROOTS` inside `_resolve_run_dir` (the single chokepoint every run dir passes through) and asserted at the end of `emit_rows` rather than threading a return value through `reconstruct_od`/`_run_base`. This is the minimal-surface implementation the plan explicitly permits ("you may compare ... into a set as runs are resolved").

## Deviations from Plan

- None. Implemented exactly as specified; `import os` added at module top, constant replaced, `_resolve_run_dir` rewritten, single-root assertion added in the `emit_rows` loop boundary.

## Self-Check: PASSED

- `! grep -F '/Users/shawngibford/dev/phd/qGAN'` → no match ✓
- `grep QGAN_CANONICAL_REPO` + `grep os.environ` → present ✓
- `_CANONICAL_REPO_FALLBACK is None` when env unset ✓ (functional check)
- `_resolve_run_dir` raises `FileNotFoundError` naming `QGAN_CANONICAL_REPO` + `D-11-08` ✓ (functional check)
- single-root assertion present with mixed-checkout message ✓
- `git diff --stat -- revision/core/` → empty ✓
- `fidelity_dualscale.json` data_hash `91e447d4624e25b3`, 3360 rows ✓
- `pytest revision/tests/ -q` → 22 passed ✓
