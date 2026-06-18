---
phase: 11-utility-evaluation
plan: 03
subsystem: evaluation
tags: [emd, acf, dtw, moments, fidelity, dual-scale, eval-05, reviewer-r1-m3]

# Dependency graph
requires:
  - phase: 08-core-module-extraction
    provides: revision.core.eval fidelity helpers (compute_emd/moments/acf/dtw)
  - phase: 09.1-r1-m3-ablation
    provides: frozen quantum transform_ablation/runs sample bundles (git-tracked)
  - phase: 10-classical-baselines
    provides: frozen baselines/runs sample bundles + baseline_comparison.json (transformed-scale reference)
provides:
  - run_dualscale_fidelity.py — EVAL-05 scale-tagged fidelity re-emit driver
  - results/fidelity_dualscale.json — 3360 long-form dual-scale rows with explicit scale field
  - Pipeline-B log_return EMD reconciles exactly with baseline_comparison.json transformed-scale rows (zero drift)
affects: [14-paper-revision, utility-evaluation, reviewer-response]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Driver-mirrors-run_baselines.py: CLI entrypoint reads frozen artifacts, computes, writes JSON"
    - "Scale-tagging loop wraps unchanged eval.py helpers with explicit scale field (no new metric math)"
    - "Rectangular long-form schema: explicit null rows + scale_na_reason instead of silent omission"

key-files:
  created:
    - run_dualscale_fidelity.py
    - results/fidelity_dualscale.json
  modified: []

key-decisions:
  - "Rule 3: run-dir resolver falls back to canonical primary checkout for git-ignored baseline artifacts absent from worktree (D-11-08 forbids regeneration)"
  - "Rule 3: sys.path bootstrap added so the documented bare-script invocation (python run_dualscale_fidelity.py) works"
  - "Real log_return EMD reference = d_real['log_delta'] (exact array _build_baseline_notebook.py:290 uses) so numbers reconcile with baseline_comparison.json"
  - "Pipeline A log_return rows emitted as explicit value:null + scale_na_reason (rectangular schema, T-11-08/T-11-09)"

patterns-established:
  - "Pattern: every revision.core.eval fidelity metric re-emitted with explicit scale: OD | log_return field (EVAL-05)"
  - "Pattern: DTW verbatim sub-sample recipe (DTW_N_PAIRS=100, np.random.default_rng(seed*31)) preserved across drivers"

requirements-completed: [EVAL-05]

# Metrics
duration: 6min
completed: 2026-05-18
---

# Phase 11 Plan 03: Dual-Scale Fidelity Re-emit Summary

**EVAL-05 driver re-emitting every revision.core.eval fidelity metric (EMD, moments, ACF, DTW) with an explicit `scale: OD | log_return` field for all 60 frozen sample artifacts — 3360 long-form rows, eval.py reused unchanged, Pipeline-B log-return EMD reconciles exactly with baseline_comparison.json.**

## Performance

- **Duration:** 6 min (driver wall ~3:51 for full 60-run metric sweep)
- **Started:** 2026-05-18T03:17:46Z
- **Completed:** 2026-05-18T03:24:00Z
- **Tasks:** 2
- **Files modified:** 2 (1 driver created, 1 JSON artifact emitted)

## Accomplishments
- `run_dualscale_fidelity.py` (521 lines) — CLI driver patterned on `run_baselines.py`, with verbatim `reconstruct_od` + 50-baseline-config data_hash assert (Pitfall 4: no quantum grep).
- `results/fidelity_dualscale.json` — 3360 rows; every row carries an explicit `scale ∈ {OD, log_return}`; full 6 model_kinds × 2 pipelines × 5 seeds OD coverage (60 runs).
- Pipeline-B `log_return`-scale EMD reconciles with `baseline_comparison.json`'s `scale="transformed"` rows with zero numeric drift (verified across quantum/wgan_mlp/vae × seeds 42,46).
- Pipeline-A `log_return` rows emitted as explicit `value: null` + `scale_na_reason` (840 null rows = rectangular schema, no silent omission — T-11-08/T-11-09).
- `core/` byte-untouched across the entire plan (`git diff --stat 8a72391..HEAD -- core/` empty — D-11-10 invariant).

## Task Commits

Each task was committed atomically:

1. **Task 1: Driver scaffold + verbatim reconstruct_od + real OD/log-return references** - `d26ce48` (feat)
2. **Task 2: Scale-tagged metric loop + write fidelity_dualscale.json** - `c24a972` (feat)

## Files Created/Modified
- `run_dualscale_fidelity.py` - EVAL-05 scale-tagged fidelity re-emit driver; verbatim `reconstruct_od`, run-dir resolver with canonical-checkout fallback, `emit_rows` scale-tagging loop reusing `revision.core.eval` helpers unchanged.
- `results/fidelity_dualscale.json` - 3360 long-form dual-scale rows `{model_kind, pipeline, seed, metric_name, scale, value[, scale_na_reason]}`; top-level `schema`, `model_kinds`, `pipelines`, `seeds`, `data_hash`, `data_hash_verification`, `metric_helpers`, `rows`.

## Decisions Made
- Real `log_return` EMD reference uses `d_real["log_delta"]` — the exact array `_build_baseline_notebook.py:290` uses for its `scale="transformed"` EMD — so EVAL-05 numbers reconcile with the existing `baseline_comparison.json` (verified zero drift).
- Pipeline-A log-return gap made visible via explicit `value: null` + `scale_na_reason` rows mirroring the OD metric-name set (rectangular schema), per RESEARCH Open Question 3.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Run-dir resolver falls back to canonical primary checkout**
- **Found during:** Task 1 (Driver scaffold)
- **Issue:** `results/` is git-ignored (`.gitignore:62`). The Phase-10 baseline run bundles under `results/baselines/runs/` are NOT git-tracked and are absent from a fresh worktree checkout. `reconstruct_od` resolves repo-root-relative paths, so all 50 baseline artifacts (5 classical model kinds × 2 pipelines × 5 seeds) were unreachable in the worktree. The quantum `transform_ablation/runs/` bundles ARE git-tracked (force-added in 09.1) and present.
- **Fix:** Added `_resolve_run_dir(rel)` — returns the in-tree path if present, else the canonical primary-checkout path (`/Users/shawngibford/dev/phd/qGAN`). D-11-08 forbids regeneration, so the frozen artifacts must be sourced from where they exist. Path-construction logic in `_run_base` is byte-verbatim with `_build_baseline_notebook.py:167-172`; only `_resolve_run_dir` wraps it. `core/` untouched.
- **Files modified:** run_dualscale_fidelity.py
- **Verification:** `reconstruct_od('quantum','B',42)` shape (3840,10) with non-None transformed; `reconstruct_od('vae','A',42)` transformed None; data_hash on all 50 baseline configs == 91e447d4624e25b3.
- **Committed in:** d26ce48 (Task 1 commit)

**2. [Rule 3 - Blocking] sys.path bootstrap for bare-script invocation**
- **Found during:** Task 2 (driver end-to-end run)
- **Issue:** `python run_dualscale_fidelity.py` (the documented invocation) does not put the repo root on `sys.path`, so `from revision.core import ...` raised `ModuleNotFoundError`. The plan's verify command worked only because it injected `sys.path.insert(0,'.')`.
- **Fix:** Added `_bootstrap_repo_on_path()` (walks up to the dir holding `core/preprocessing.py`, prepends to `sys.path`) — the same bootstrap the notebook generators use — called before the `revision.*` imports.
- **Files modified:** run_dualscale_fidelity.py
- **Verification:** `./qgan_env/bin/python run_dualscale_fidelity.py` runs end-to-end (~3:51, under 10-min budget); 3360 rows emitted.
- **Committed in:** c24a972 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking, Rule 3)
**Impact on plan:** Both fixes were required to execute the plan in a worktree where `results/` is git-ignored and the script is invoked as documented. Neither alters metric math, the `reconstruct_od` contract, or `core/`. No scope creep. Frozen artifacts consumed read-only (D-11-08); core untouched (D-11-10).

## Issues Encountered
- DTW nearest-neighbour pairing is O(L²) per pair; the full 60-run × dual-scale sweep took ~3:51 wall — comfortably under the RESEARCH A5 10-minute budget. No code change needed (the DTW_N_PAIRS=100 / rng=default_rng(seed*31) recipe is locked verbatim per T-11-03).

## Self-Check: PASSED

- FOUND: run_dualscale_fidelity.py (521 lines, ≥150 required)
- FOUND: results/fidelity_dualscale.json (660,559 bytes, 3360 rows, contains "log_return")
- FOUND: commit d26ce48 (Task 1)
- FOUND: commit c24a972 (Task 2)
- VERIFIED: data_hash == 91e447d4624e25b3; scales == {OD, log_return}; 60/60 OD coverage
- VERIFIED: Pipeline-B log_return EMD reconciles with baseline_comparison.json (zero drift)
- VERIFIED: Pipeline-A log_return = explicit null + scale_na_reason (840 rows)
- VERIFIED: git diff --stat 8a72391..HEAD -- core/ empty (D-11-10)

## Next Phase Readiness
- EVAL-05 deliverable complete; `fidelity_dualscale.json` ready for Phase 14 paper revision (answers reviewer R1-m3/R1-M3: every fidelity metric carries an explicit reporting scale).
- Note for orchestrator: `results/fidelity_dualscale.json` was force-added (`git add -f`) because `.gitignore` excludes `results/`, consistent with how 09.1 force-added `transform_ablation/runs/`. Baseline run bundles remain git-ignored and were read via the canonical-checkout fallback — downstream plans/CI must run where those frozen artifacts exist or git-track them.
- No blockers.

---
*Phase: 11-utility-evaluation*
*Completed: 2026-05-18*
