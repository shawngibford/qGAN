---
phase: 11-utility-evaluation
plan: 04
subsystem: testing
tags: [pytest, scientific-integrity, leakage-sentinel, data-hash, phase10-reproduction, dual-scale, timegan, core-untouched, phase11-closeout]

# Dependency graph
requires:
  - phase: 11-utility-evaluation
    provides: "Wave-1 outputs tstr.json / augmentation.json / predictive_discriminative.json / fidelity_dualscale.json + run_utility.reconstruct_od"
  - phase: 10-classical-baselines
    provides: "baseline_comparison.json — the Phase-10 quantum|B OD-EMD anchor (git-tracked)"
provides:
  - "tests/test_utility.py — Phase-11 cross-artifact scientific-integrity pytest suite (10 test functions, dual-mode)"
  - "Executable closeout proof that ROADMAP Phase-11 SC 1-4 are each artifact-backed"
affects: [14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cross-artifact integrity suite consumes Wave-1 JSONs read-only; only #5/#9 recompute (small subset)"
    - "Artifact-free Phase-10 reproduction proof: fidelity_dualscale.json reconciled row-for-row vs baseline_comparison.json anchor (always hard-asserts)"
    - "Repo-root sys.path bootstrap + pytest-shim so the suite is genuinely dual-mode (system pytest AND qgan_env script fallback)"

key-files:
  created:
    - tests/test_utility.py
  modified: []

key-decisions:
  - "Rule-1: sample-shape sentinel is per-pipeline (B->(3840,10), A->(3850,10)) not a flat (3840,10); a flat assertion would wrongly fail every valid Pipeline-A run"
  - "Rule-3: de-facto verification is system pytest on PATH — the plan's literal './qgan_env/bin/python -m pytest' is unsatisfiable (pytest absent from qgan_env); zero installs (threat T-11-SC accept honored)"
  - "Phase-10 reproduction has an artifact-free hard path (anchor JSON reconciliation) + an artifact-bearing recompute path that skips cleanly in a bare worktree"

patterns-established:
  - "Pattern: scientific-integrity invariants codified as pytest sentinels (data-hash/leakage/shape/dual-scale/metadata/core-untouched/Phase-10-reproduction) — permanent guard, not one-time inspection"
  - "Pattern: dual-mode test file (pytest + plain-script) via graceful pytest-shim so qgan_env without pytest still runs the suite"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05]

# Metrics
duration: ~35min
completed: 2026-05-18
---

# Phase 11 Plan 04: Cross-Artifact Verification Suite Summary

**`tests/test_utility.py` — a 10-function pytest suite locking every Phase-11 scientific-integrity invariant (data-hash, eval/train leakage, per-pipeline sample shape, dual-scale fidelity, TimeGAN reference-pinning, `core/` untouched, Phase-10 quantum|B OD-EMD reproduction) for all four Wave-1 JSON outputs; full `tests/` suite green (22 passed) with zero Phase 8-10 regression and `git diff core/` byte-count 0.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-18
- **Completed:** 2026-05-18
- **Tasks:** 2
- **Files modified:** 1 created (`tests/test_utility.py`)

## Accomplishments

- **`test_utility.py`** (10 test functions) following the `test_classical.py` convention, plus a repo-root `sys.path` bootstrap and a graceful pytest-shim so it runs identically under system pytest and as a plain `./qgan_env/bin/python` script:
  - `test_all_outputs_exist` — all four JSONs exist + parse + non-empty `rows`.
  - `test_data_hash_consistency` — every JSON top-level `data_hash == 91e447d4624e25b3`; `run_utility.EXPECTED_DATA_HASH` matches; recomputed `_compute_data_hash(data.csv)` matches.
  - `test_long_form_schema` — exact `{model_kind,pipeline,seed,metric_name,scale,value}` keys on every row of every JSON (+ documented `injection_ratio` / `scale_na_reason` extras; augmentation rows must all carry `injection_ratio`).
  - `test_no_leakage_sentinel` — TSTR `real_only_baseline` R2 NEGATIVE for every init seed, `n_train_real==65`, `n_eval_real==320`; augmentation lift-block `real_only` R2 NEGATIVE, `n_real_train==65` (Pitfall 5 leakage guard).
  - `test_sample_shape_invariant` — parametrized over `(wgan_mlp|B, quantum|B, quantum|A)`: Pipeline B → (3840,10), Pipeline A → (3850,10), 10 columns, dtype float64.
  - `test_timegan_metadata` — `jsyoon0823/TimeGAN` URL, `hidden_dim==10`, non-trivial `univariate_adaptation` rationale (A1).
  - `test_dualscale_coverage` — both `scale` values present; Pipeline-B `log_return` EMD non-null; Pipeline-A `log_return` explicit `value is None` + `scale_na_reason` (T-11-08/09).
  - `test_core_untouched` — `git diff --stat -- core/` empty (D-11-10, T-11-04).
  - `test_phase10_reproduction_anchor_reconciles` — artifact-free hard proof: `fidelity_dualscale.json` quantum|B OD-EMD reconciles **bit-stable** (<1e-9) row-for-row with the Phase-10 anchor in `baseline_comparison.json`, and its mean (0.027586) sits inside the RESEARCH band `0.0276 ± 0.0046`. Plus an artifact-bearing `test_phase10_reproduction_recompute` that live-recomputes via `reconstruct_od` + `revision.core.eval.compute_emd` (skips cleanly without frozen `samples.npy`).
  - `test_phase11_success_criteria` — aggregator asserting ROADMAP SC-1 (TSTR r2/mae/rmse), SC-2 (predictive+discriminative mean/std over 5 seeds), SC-3 (augmentation delta rows × injection grid), SC-4 (explicit dual `scale` on every fidelity row).
- **Full `tests/` suite green: 22 passed** (test_classical 2 + test_nonadversarial 3 + test_timegan_scores + test_utility 13 incl. parametrized/recompute) — no Phase 8-10 regression. `git diff --stat core/` byte-count **0** after both tasks (D-11-10 final invariant).

## Task Commits

Each task was committed atomically:

1. **Task 1: cross-artifact scientific-integrity suite** - `977648a` (test)
2. **Task 2: full-suite regression gate + dual-mode script fallback** - `23978f0` (test)

## Files Created/Modified

- `tests/test_utility.py` - Phase-11 cross-artifact integrity suite: 10 test functions over the four Wave-1 JSONs; repo-root bootstrap + pytest-shim for dual-mode (pytest / plain-script) execution; recompute tests guarded to skip without frozen artifacts.

## Decisions Made

- **Phase-10 reproduction split into an artifact-free hard path + an artifact-bearing recompute path.** The strongest "verbatim reuse did not drift" evidence is reconciling `fidelity_dualscale.json` against the git-tracked `baseline_comparison.json` anchor — that needs no frozen `samples.npy` and always hard-asserts (bit-stable to <1e-9; mean 0.027586 inside `0.0276 ± 0.0046`). A second live-recompute test (`reconstruct_od` + `compute_emd`) hard-asserts where the canonical checkout's frozen artifacts exist and `pytest.skip`s in a bare worktree.
- **System pytest on PATH is the de-facto verification environment.** The existing Wave-1 / prior-phase tests are run with `/opt/homebrew/bin/pytest` (9.0.2), which resolves both the `revision.*` package and the scientific libs and runs the prior suite green — this is the working invocation, used unchanged here.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Sample-shape sentinel is per-pipeline, not a flat (3840,10)**
- **Found during:** Task 1 (test_sample_shape_invariant verification)
- **Issue:** The plan's acceptance criterion #5 asserts `reconstruct_od(...).shape == (3840,10)` generically. Empirically the frozen artifacts carry a **pipeline-dependent** synth count: Pipeline B → (3840,10), Pipeline A → **(3850,10)** (verified across wgan_mlp/quantum/vae/ar). Plan-01's SUMMARY only ever spot-checked `wgan_mlp|B`==(3840,10), so the flat figure was Pipeline-B-specific. A flat (3840,10) assertion would make the sentinel itself wrong — failing on every valid Pipeline-A run.
- **Fix:** Codified the true invariant: `_EXPECTED_SHAPE = {"A": (3850,10), "B": (3840,10)}`, plus `shape[1]==10` (WINDOW_LENGTH) and `dtype==float64`. The sentinel now guards the real artifact property rather than encoding an incorrect spec.
- **Files modified:** tests/test_utility.py
- **Verification:** parametrized `(wgan_mlp|B, quantum|B, quantum|A)` all pass; full suite 22 passed.
- **Committed in:** 977648a (Task 1)

**2. [Rule 3 - Blocking] Plan's verify command unsatisfiable — pytest absent from qgan_env**
- **Found during:** Task 1 / Task 2 (running the documented `./qgan_env/bin/python -m pytest`)
- **Issue:** The plan/RESEARCH/threat-register all assume "pytest already in qgan_env (zero installs)". It is NOT: `./qgan_env/bin/python -m pytest` → `No module named pytest`. The constraint is correct (no installs — T-11-SC `accept`), but the literal verify command cannot run. The existing `test_classical.py`/`test_nonadversarial.py` are de-facto run via the system `pytest` on PATH (`/opt/homebrew/bin/pytest` 9.0.2), which resolves `revision.*` + scientific libs and runs the prior suite green.
- **Fix:** (a) Verified the suite via system `pytest tests/` (the working, install-free invocation — threat T-11-SC `accept`/zero-install honored; no slopsquat risk since nothing is installed). (b) Added a graceful pytest-shim + repo-root `sys.path` bootstrap so the plain-script fallback (`./qgan_env/bin/python tests/test_utility.py`) also runs the suite — making the file genuinely dual-mode instead of dead code. `import pytest` no longer hard-fails under qgan_env.
- **Files modified:** tests/test_utility.py
- **Verification:** system pytest full suite 22 passed; qgan_env script-mode `ALL test_utility.py checks PASSED` (incl. recompute).
- **Committed in:** 23978f0 (Task 2)

---

**Total deviations:** 2 auto-fixed (1 Rule-1 bug in the plan's literal spec, 1 Rule-3 blocking — broken verify command). Neither touches `core/`, the Wave-1 drivers, or any metric math; both make the sentinels *correct* and *runnable*. No scope creep. No package installs (T-11-SC `accept` upheld).

## Issues Encountered

- **Worktree has no venv and no frozen `samples.npy`.** `results/` is git-ignored; the 60 frozen sample bundles + `qgan_env` exist only in the canonical checkout `/Users/shawngibford/dev/phd/qGAN` (the documented Wave-1 pattern — see Plans 01-03 SUMMARYs). Resolution: the test file was authored and committed in the worktree, mirrored to the canonical checkout to run/verify against real artifacts + env, and the repo-root bootstrap + skip-guards make the committed file run identically in both locations (hard-asserts where artifacts exist, skips the 2 recompute tests in a bare worktree; the 8 artifact-free tests — including the Phase-10 anchor reconciliation — always hard-assert).

## User Setup Required

None - local, offline, no network/auth/PII.

## Verification Results

- `pytest tests/test_utility.py -q` → 13 passed (incl. 3 parametrized shape + recompute, artifact-bearing checkout).
- `pytest tests/ -q` → **22 passed**, 1 warning (no Phase 8-10 regression).
- `./qgan_env/bin/python tests/test_utility.py` (script-mode) → ALL checks PASSED.
- `git diff --stat core/` → empty (byte-count 0) after both tasks (D-11-10).
- Phase-10 anchor: quantum|B OD-EMD mean 0.027586 (std 0.004576) — inside `0.0276 ± 0.0046`; reconciles bit-stable (<1e-9) row-for-row with `baseline_comparison.json`.

## Next Phase Readiness

- All five EVAL requirements (EVAL-01..05) are now executable-suite-backed; Phase 11 closes with a permanent goal-backward verification guard. `tstr.json` / `augmentation.json` / `predictive_discriminative.json` / `fidelity_dualscale.json` are integrity-locked for Phase 14 manuscript consumption.
- Note for orchestrator: the documented `./qgan_env/bin/python -m pytest` command is unsatisfiable (pytest absent from qgan_env). The working CI invocation is system `pytest tests/` on PATH; the file also self-runs via `./qgan_env/bin/python tests/test_utility.py`. No installs were performed (threat T-11-SC `accept` honored).
- No blockers.

## Self-Check: PASSED

- FOUND: tests/test_utility.py
- FOUND: .planning/phases/11-utility-evaluation/11-04-SUMMARY.md
- FOUND commit 977648a (Task 1)
- FOUND commit 23978f0 (Task 2)
- VERIFIED: full tests/ suite 22 passed (no regression)
- VERIFIED: git diff --stat core/ empty (D-11-10)
- VERIFIED: Phase-10 quantum|B OD-EMD reconciles within anchor band

---
*Phase: 11-utility-evaluation*
*Completed: 2026-05-18*
