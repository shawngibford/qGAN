---
phase: 14-paper-revision-release-freeze
plan: 01
subsystem: infra
tags: [pytorch, pennylane, checkpoint-recovery, quantum-circuit, config-equivalence-gate, provenance-json]

# Dependency graph
requires:
  - phase: 13-architecture-introspection
    provides: "_TOPOLOGIES config-selectable precedent in core/models/quantum.py (the exact selector shape mirrored here)"
  - phase: 09.1-r1-m3-ablation
    provides: "core/preprocessing.py Pipeline A/B/C definitions (used to pin the native headline pipeline)"
provides:
  - "Config-selectable NON-default 55-param IQP:SEL circuit (circuit_id=iqp_sel_55) in core/models/quantum.py"
  - "run_recover_canonical.py — checkpoint-driven recovery + D-14-07 phase-blocking equivalence gate driver"
  - "results/canonical_recovery.json — decomposition + checkpoint provenance (epoch 1969, verbatim mu/sigma, sha256, optimizer breadcrumbs)"
  - "results/canonical_config_lock.json — locked canonical config + pinned native pipeline B + checkpoint sha256"
affects: [14-02, 14-03, 14-04, 14-05, 14-06, 14-07, 2000ep-re-execution, model-info, figure-suite, latex-blocks]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Checkpoint-driven config reconstruction (RESEARCH Pattern 1): tensor layout is the oracle, decomposition disambiguated by a structural forward-pass walk, NOT formula-guessed"
    - "Explicit-raise integrity gate (run_multiseed_rollup.py:86-92 idiom): raise AssertionError, never bare assert (survives python -O)"
    - "Config-selectable non-default circuit variant mirroring the _TOPOLOGIES eager-validation shape"

key-files:
  created:
    - run_recover_canonical.py
    - results/canonical_recovery.json
    - results/canonical_config_lock.json
  modified:
    - core/models/quantum.py
    - tests/test_utility.py

key-decisions:
  - "55-param decomposition = q=5, L=3, IQP-encoding(5) + 3 SEL layers(45) + final-RX-only(5); chosen because IQP encoding params must be present for the circuit to be a genuine IQP:SEL, and the structural forward-pass walk consumes exactly 55"
  - "Native headline pipeline = B (log-return standardization): the checkpoint stores zero-centred scalar mu/sigma (0.00245/0.02141) matching stats.json moments_real; Pipeline A (min-max OD) would store od_min/od_max so A is excluded; C is the Lambert-W superset of B's standardization"
  - "test_core_untouched rescoped (Rule-1) from 'zero core diff' to the real T-14-02 invariant (default path byte-frozen + __init__.py untouched) because Phase-14 D-14-01 explicitly mandates the non-default core addition"

patterns-established:
  - "Pattern: worktree-aware gitignored-artifact resolver — best_checkpoint.pt lives in the main checkout, not the worktree; resolver walks the .git link target to find it"
  - "Pattern: NON-default circuit variant added iff the default tape is bit-identical (proven via deterministic forward-pass diff vs HEAD)"

requirements-completed: [PAPER-03]

# Metrics
duration: ~18min
completed: 2026-05-19
---

# Phase 14 Plan 01: Canonical 55-param IQP:SEL Recovery Summary

**Recovered the lost canonical 55-param IQP:SEL circuit deterministically from best_checkpoint.pt (epoch 1969), added it to quantum.py as a config-selectable non-default variant behind a python-O-safe explicit-raise equivalence gate, and pinned native headline Pipeline B — Tier-1 of D-14-22 complete, unblocking the 2000ep re-execution.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-19 (worktree agent-a554910daa786f947)
- **Completed:** 2026-05-19
- **Tasks:** 2
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments
- Reverse-engineered the 55-param decomposition from the ground-truth checkpoint: **q=5, L=3, IQP-encoding(5) + 3 SEL layers(45) + final RX-only(5) = 55** — disambiguated by a structural forward-pass walk consuming exactly 55 params (RESEARCH Pitfall 2 mitigated, T-14-01)
- Identified and pinned the **native headline pipeline B** (log-return standardization) from the checkpoint's stored mu/sigma vs `results/run_unconditioned_wgan/stats.json` moments_real (D-14-06)
- Added `circuit_id` config selector (`default_75` | `iqp_sel_55`) to `quantum.py` mirroring the `_TOPOLOGIES` eager-validation shape; the **default path is bit-identical to HEAD** (max abs forward diff = 0.0, T-14-02)
- D-14-07 phase-blocking equivalence gate passes: explicit-raise shape gate (`==(55,)`) + structural forward-pass gate (consumed `==55`), both python-`-O` safe
- Two provenance JSONs written with verbatim mu/sigma (no recompute, D-14-05), checkpoint sha256 `f7cceb52…` (T-14-14), and the locked canonical config

## Task Commits

Each task was committed atomically:

1. **Task 1: Reverse-engineer the 55-param decomposition from the checkpoint** - `144f0fe` (feat)
2. **Task 2: Add the 55-param IQP:SEL as a config-selectable non-default circuit** - `db59b11` (feat)

## Files Created/Modified
- `run_recover_canonical.py` - Checkpoint-driven recovery (`--recover-only`) + D-14-07 phase-blocking equivalence gate (`--assert-equivalence`); worktree-aware gitignored-checkpoint resolver
- `results/canonical_recovery.json` - Decomposition, epoch 1969, verbatim mu/sigma, checkpoint sha256, optimizer LR/betas breadcrumbs, native pipeline B + evidence
- `results/canonical_config_lock.json` - Locked `circuit_id=iqp_sel_55`, decomposition, pinned pipeline B, stored mu/sigma, checkpoint sha256, equivalence-gate status
- `core/models/quantum.py` - `_CIRCUIT_IDS` selector + `circuit_id` arg + variant-aware param formula and Step-5 final-rotation block + `last_param_index()` structural-introspection helper; `default_75` branch literally byte-frozen
- `tests/test_utility.py` - `test_core_untouched` rescoped (Rule-1) to the real T-14-02 default-path-byte-freeze invariant

## Decisions Made
- **Decomposition family:** Of the two 55-param q=5/L=3 families (`enc=0+45+10` vs `enc=5+45+5`), chose `enc=5 + 3*15 + final-RX-only 5` because the IQP encoding params must be present for the circuit to genuinely be "IQP:SEL"; verified by a structural forward-pass walk consuming exactly 55.
- **Native pipeline B:** Stored scalar mu/sigma are log-return standardization stats (zero-centred, match stats.json moments_real). Pipeline A would store min-max OD bounds, so A is excluded; C shares B's log-return standardization (Lambert-W superset). Recorded `native_pipeline: "B"` with rationale + stats.json evidence.
- **No dataset hash in recovery JSON:** the checkpoint stores only model/optimizer tensors + scalar mu/sigma; the headline `data_hash` is established downstream against the pinned pipeline. Omitted with an explicit note per the plan's schema-conformance instruction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `test_core_untouched` over-asserted against the Phase-14-mandated core addition**
- **Found during:** Task 2 (full `pytest tests/` regression run)
- **Issue:** `tests/test_utility.py::test_core_untouched` is a Phase-11-era guard (D-11-10) asserting *zero* git diff under `core/`. Phase-14 D-14-01 explicitly mandates adding the recovered 55-param circuit to `core/models/quantum.py`, so the old guard would fail a correct, equivalence-gated deliverable. The true invariant is **default-path byte-freeze** (T-14-02), not whole-core-freeze (same rescope Phase-13 ARCH-01 applied to its core guard).
- **Fix:** Rescoped the test to assert (a) `core/__init__.py` (the frozen architecture-constants baseline) has zero git diff, and (b) a no-arg `QuantumGenerator()` still builds the byte-identical 75-param `default_75` circuit with the expected forward-output shape.
- **Files modified:** `tests/test_utility.py`
- **Verification:** `pytest tests/test_utility.py::test_core_untouched` passes; default-path forward pass proven bit-identical vs HEAD (max abs diff = 0.0, atol 1e-12).
- **Committed in:** `db59b11` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule-1 bug)
**Impact on plan:** The rescope encodes the *actual* T-14-02 invariant the plan's threat model demands and is the same pattern Phase-13 used for its core guard. No scope creep — the default path remains byte-frozen and independently proven.

## Issues Encountered
- **Gitignored artifacts absent in worktree:** `best_checkpoint.pt` and `qgan_env` are `.gitignore`d, so the parallel-execution worktree does not contain them — they live in the main checkout. Resolved by (a) adding a worktree-aware checkpoint resolver in `run_recover_canonical.py` that walks the `.git` link target to find the main-repo `best_checkpoint.pt`, and (b) invoking the worktree script with the main repo's `qgan_env/bin/python`. The script's repo-root resolver still writes artifacts into the worktree's `results/`.
- **Pre-existing env-only test failure (out-of-scope, deferred):** `test_sample_shape_invariant[wgan_mlp-B-42]` (and the two `quantum` variants) fail with `FileNotFoundError` for `results/baselines/runs/.../samples.npy` — a gitignored frozen Phase-10 runtime artifact not copied into the worktree. Not caused by 14-01 changes (scope boundary). Logged to `.planning/phases/14-paper-revision-release-freeze/deferred-items.md`. All 20 in-scope tests pass.

## User Setup Required
None - no external service configuration required (Zenodo DOI is a later-plan operator step).

## Next Phase Readiness
- Tier-1 of the D-14-22 strict gated pipeline is **complete**: the 55-param IQP:SEL is config-selectable and non-default, the checkpoint loads into it under the D-14-07 phase-blocking explicit-raise equivalence gate, native Pipeline B is pinned, and all provenance is in JSON.
- Downstream 2000ep re-execution and every cross-model comparison can now select `QuantumGenerator(num_qubits=5, num_layers=3, window_length=10, circuit_id="iqp_sel_55")` and load `best_checkpoint.pt` with the verified `checkpoint_sha256` for the frozen headline (D-14-03/05).
- No blockers. The core default path remains byte-frozen so Phases 8-13 are not re-baselined.

## Threat Surface Scan
No new network endpoints, auth paths, file-access patterns, or trust-boundary schema changes introduced. The two trust boundaries in the plan's threat model (checkpoint→reconstructed-config T-14-01, recovery-script→core-default T-14-02) are both mitigated as specified (explicit-raise equivalence gate + structural walk; non-default add only with proven bit-identical default tape). No threat flags.

## Self-Check: PASSED
- `run_recover_canonical.py` — FOUND
- `results/canonical_recovery.json` — FOUND (param_count 55, epoch 1969, native_pipeline B)
- `results/canonical_config_lock.json` — FOUND (locked iqp_sel_55, checkpoint sha256)
- `core/models/quantum.py` — FOUND (modified, default path byte-frozen)
- Commit `144f0fe` — FOUND
- Commit `db59b11` — FOUND

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-19*
