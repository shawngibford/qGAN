---
phase: 09-documentation-bridge
plan: 03
subsystem: docs
tags: [methods-section, training-protocol, wgan-gp, hyperparameters, citations, paper-ready]

# Dependency graph
requires:
  - phase: 08-core-module-extraction
    provides: core/{__init__.py, training.py, eval.py, models/quantum.py, models/critic.py} — source-of-truth constants and class implementations cited by the doc
  - phase: 04-hyperparameter-optimization (v1.1)
    provides: HPO-tuned constants (N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, [0, 4π] noise) embedded in __init__.py
provides:
  - docs/training_protocol.md (DOC-01) — paper-ready Methods-section content
  - Hybrid format (table + 1-paragraph prose) template for other phase-9 docs
  - File:line citation discipline (D-09) for hyperparameter traceability
affects: [09-04-dataset-stats, 14-paper-drafting (PAPER-08, PAPER-09), future HPO retunes that touch __init__.py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hybrid markdown format: tables for numeric content + 1-paragraph prose justifications per section (D-08)"
    - "Per-row file:line citations back to canonical source files (D-09)"
    - "Explicit shot/analytic distinction in Methods docs (D-10)"

key-files:
  created:
    - docs/training_protocol.md
  modified: []

key-decisions:
  - "Reproducibility section included (Claude's discretion item from CONTEXT.md) — fits naturally with seed=42 default and DITHER/DITHER_SEED traceability"
  - "Section ordering: Optimizer & Schedule → Early-Stopping → Quantum Circuit → Critic → Gradient Penalty → Reproducibility → Analytic-vs-Shot Distinction (matches 09-RESEARCH.md line 360 suggestion)"
  - "Adam betas cite training.py:233-234 (live verification confirmed) rather than __init__.py because the Gulrajani 0.0/0.9 choice lives in the optimizer construction, not the constants module"
  - "Critic table rows include explicit line numbers (46, 49, 52, 56, 59-63, 67) for each architecture block to give reviewers a single grep target"

patterns-established:
  - "Source-of-truth blockquote at top of file: 'Update those files to change values; this document tracks them via per-row line citations'"
  - "Methodological caveat callout (R1-M5 calibration honesty) inline with the relevant section rather than buried at the end"

requirements-completed: [DOC-01]

# Metrics
duration: ~7min
completed: 2026-05-15
---

# Phase 9 Plan 03: Training Protocol Methods Doc Summary

**`docs/training_protocol.md` (153 lines, 7 sections) documenting all 17 HPO hyperparameters with per-row file:line citations to `core/__init__.py` plus Adam betas, EarlyStopping, shots=None, and seeding citations to training.py / quantum.py / critic.py / eval.py — paper-ready drop-in for Phase 14 Methods (PAPER-08, PAPER-09).**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-15T15:52:00Z (approx)
- **Completed:** 2026-05-15T15:59:02Z
- **Tasks:** 1/1
- **Files modified:** 1 created (0 modified)

## Accomplishments

- All 17 hyperparameter constants from `core/__init__.py` (lines 11–33) documented with verbatim values and line citations.
- Adam optimizer betas=(0.0, 0.9) cited at `core/training.py:233-234`.
- EarlyStopping (patience=50, warmup_epochs=100, save-best-EMD/reload-on-stop) cited at `core/training.py:79-175`.
- shots=None / diff_method="backprop" cited at `core/models/quantum.py:64, 43, 77`.
- Critic 1D-CNN architecture (Conv1d 1→64→128→128, k=10, p=5, AdaptiveAvgPool1d, Linear 128→32→1 with LeakyReLU + Dropout) cited at `core/models/critic.py:46, 49, 52, 56, 59-63, 67`.
- Two-sided gradient penalty (λ=2.16, U(0,1) interpolation, unit-norm target) cited at `core/training.py:30-73`.
- Seed=42 default + torch/np/random/cuda seeding cited at `core/training.py:188, 211-215`.
- D-10 shot/analytic statement explicit in dedicated final section deferring shot-noise sweep to Phase 12 SENS-01.
- R1-M5 calibration honesty caveat included in Early-Stopping section (no held-out validation split because of single-campaign dataset; cross-references `docs/dataset_stats.md`).

## 7 Sections with their numerical citations

1. **Optimizer & Schedule** — `N_CRITIC=9`@`__init__.py:11`, `LAMBDA=2.16`@`:12`, `LR_CRITIC=1.8046e-05`@`:13`, `LR_GENERATOR=6.9173e-05`@`:14`, Adam `betas=(0.0, 0.9)`@`training.py:233-234`, `NUM_EPOCHS=2000`@`:20`, `BATCH_SIZE=12`@`:21`, `EVAL_EVERY=10`@`:23`.
2. **Early-Stopping** — `EarlyStopping` class@`training.py:79-175`, EMD metric@`eval.py:25-36`, `patience=50`@`training.py:96`, `warmup_epochs=100`@`training.py:97`, checkpoint scheme@`training.py:142-175`.
3. **Quantum Circuit** — `shots=None`@`quantum.py:64`, `diff_method="backprop"`@`quantum.py:43, 77`, `NUM_QUBITS=5`@`:17`, `NUM_LAYERS=4`@`:18`, `WINDOW_LENGTH=10`@`:19`, `NOISE_LOW/HIGH=0/4π`@`:32-33`, `GEN_SCALE=1.0`@`:22`, 75 PQC params (verified Phase 8).
4. **Critic (1D-CNN)** — Block 1@`critic.py:46`, Block 2@`:49`, Block 3@`:52`, AdaptiveAvgPool1d@`:56`, head@`:59-63`, `DROPOUT_RATE=0.2`@`__init__.py:24`, float64 precision@`critic.py:67`.
5. **Gradient Penalty** — `compute_gradient_penalty` at `training.py:30-73`, `λ=2.16`@`__init__.py:12`, α∼U(0,1)@`training.py:54-60`, unit-norm target@`training.py:72`.
6. **Reproducibility** — Seed default 42@`training.py:188`, `torch.manual_seed`@`:211`, `np.random.seed`@`:212`, `random.seed`@`:213`, `torch.cuda.manual_seed_all`@`:214-215`, `DITHER=0.005`@`__init__.py:27`, `DITHER_SEED=42`@`:28`.
7. **Analytic-vs-Shot Distinction** — Prose only; D-10 statement deferring shot-noise sweep to Phase 12 SENS-01 at `{analytic, 8192, 1024}` shots.

## Citation audit

Every constant value in the doc matches `core/__init__.py` verbatim — verified by reading `__init__.py` (45 lines, fully read) and `training.py` lines 1–250 before drafting. Live-grep count: 17 occurrences of `core/__init__.py:` (acceptance threshold ≥ 10), 14 occurrences of `core/training.py:`, 4 occurrences of `core/models/quantum.py:`, 7 occurrences of `core/models/critic.py:`, 1 occurrence of `core/eval.py:`. All 20+ acceptance-criteria grep gates returned `OK` in one run.

## Task Commits

1. **Task 1: Write docs/training_protocol.md (7 sections, hybrid table+prose)** — `4b2f5ed` (docs)

## Files Created/Modified

- `docs/training_protocol.md` (created, 153 lines) — Paper-ready Methods-section equivalent. Drop-in target for Phase 14 PAPER-08 / PAPER-09.

## Decisions Made

- **Included Reproducibility section** (Claude's discretion item from `09-CONTEXT.md` line 57): the section fits naturally between Gradient Penalty and the Shot/Analytic statement and is needed to cite `seed=42` + DITHER constants.
- **Line numbers corrected vs. plan template:** plan's draft listed `np.random.seed(seed) | core/training.py:212-213` and `random.seed(seed) | core/training.py:213-214`. Live `Read` of `training.py` confirmed `np.random.seed` is line 212 alone and `random.seed` is line 213 alone, with `torch.cuda.manual_seed_all` at line 215 inside an `if torch.cuda.is_available():` block at line 214. Citations now read `:212`, `:213`, `:214-215` respectively for accuracy (D-09 compliance).
- **Critic architecture rows carry per-line citations** (46, 49, 52, 56, 59-63, 67) rather than the plan's generic `core/models/critic.py`; chosen to give reviewers single grep targets.
- **Gradient penalty "Gradient target = 1" cited to `training.py:72`** (the `((gradients.norm(2, dim=1) - 1) ** 2).mean()` line) rather than the generic `30-73` range used for the type row.

## Deviations from Plan

None on the spec side — all required content delivered verbatim. Citation-line accuracy improvements (above) are interpretations of the plan's instruction "do not hand-type values for any constant; copy the values from `core/__init__.py` line-by-line" — the same discipline applied to citation lines. The plan's textual draft also lists `core/training.py:30-73` for the gradient-penalty "Gradient target = 1" row; refined to `:72` since that is the literal line containing the `- 1` term. This is a Rule 1 (citation accuracy) micro-fix, not a content change.

**Total deviations:** 0 functional, 4 citation-line-number refinements (all Rule 1 accuracy; values unchanged).
**Impact on plan:** None. All acceptance grep-gates passed on first run.

## Issues Encountered

None. Source files were all read and cross-checked before the Write call.

## Threat surface scan

Re-read of the doc against the plan `<threat_model>` register:
- **T-09-11 (number drift):** Mitigated. Every numeric value in every table row carries a `core/__init__.py:LINE` (or `training.py`/`quantum.py`/`critic.py`/`eval.py`) citation. One `grep -c 'core/__init__.py'` confirms ≥ 17 occurrences against the ≥ 10 threshold.
- **T-09-12 (overclaiming shots):** Mitigated. Final section ("Analytic-vs-Shot Distinction") states explicitly that all Phase 9 results use analytic statevector simulation and defers shot-noise reporting to Phase 12 SENS-01. R1-M5 calibration caveat additionally appears inline in the Early-Stopping section regarding the absence of held-out validation EMD.
- **T-09-13 (info disclosure):** Accepted — static markdown, no PII or secrets.
- **T-09-14 (citation provenance):** Mitigated. Citations resolve to actual files (all four — `__init__.py`, `training.py`, `quantum.py`, `critic.py`, `eval.py` — exist and were read live during drafting).

No new threat flags introduced (no new endpoints, no auth surface, no schema changes).

## Pointer to downstream consumer

This doc is the upstream artifact for **Phase 14 (PAPER-08, PAPER-09) Methods drafting**. Both paper requirements consume `docs/training_protocol.md` directly: PAPER-08 drafts the Training Methods subsection by paraphrasing the prose blocks and embedding the constants tables; PAPER-09 cross-references the shot/analytic statement when discussing reviewer concern R1-M5. A future HPO retune (post-Phase 12 SENS-01) updates `core/__init__.py` once and the doc tracking is mechanical via the line citations.

## Self-Check: PASSED

Verified files:
- `docs/training_protocol.md` — exists (153 lines, `[ -f docs/training_protocol.md ]` returns 0)

Verified commits:
- `4b2f5ed` (docs(09-03): write training_protocol.md paper-ready Methods doc) — confirmed in `git log`

All acceptance-criteria grep gates in `<verify>` returned `Task 1 training_protocol.md checklist: OK` on first run.

## Next Phase Readiness

- Phase 9 Plan 04 (`09-04-PLAN.md`, dataset_stats.md) can start immediately — this plan establishes the hybrid-format + per-row-citation pattern Plan 04 will mirror.
- Phase 14 (PAPER-08, PAPER-09) can draft Methods independently of Phases 10–13 once Plan 04 lands (parallel-decoupling rationale per `09-CONTEXT.md`).
- No blockers.

---
*Phase: 09-documentation-bridge*
*Completed: 2026-05-15*
