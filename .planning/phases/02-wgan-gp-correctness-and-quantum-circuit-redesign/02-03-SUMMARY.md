---
phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign
plan: 03
subsystem: quantum-ml
tags: [wgan-gp, broadcasting, gradient-penalty, emd, stylized-facts, kurtosis, denormalization]

# Dependency graph
requires:
  - phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign
    plan: 01
    provides: "WGAN-GP config with GEN_SCALE, EVAL_EVERY, LAMBDA=10, N_CRITIC=5"
  - phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign
    plan: 02
    provides: "Data re-uploading circuit with backprop, clean critic"
provides:
  - "Broadcasting-based training loop (single QNode call per batch)"
  - "One-sided gradient penalty with per-sample alpha"
  - "Correct EMD computation on raw 1D arrays via wasserstein_distance"
  - "full_denorm_pipeline() for consistent denormalization"
  - "Stylized facts with kurtosis at stitched and window level"
  - "Periodic evaluation every EVAL_EVERY epochs with 4 logging categories"
  - "Dynamic histogram bins from real data range"
affects: [02-04-checkpoint-system]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parameter broadcasting: noise (num_qubits, batch_size) -> single QNode call"
    - "One-sided GP: clamp(grad_norm - 1, min=0)^2"
    - "EMD on raw 1D arrays in normalized space (not histograms)"
    - "Denormalization pipeline: rescale -> lambert_w_transform -> denormalize"
    - "Periodic eval with all 4 categories: losses+GP, EMD+facts, plot, grads"

key-files:
  created: []
  modified:
    - "qgan_pennylane.ipynb"

key-decisions:
  - "Used lambert_w_transform (not inverse_lambert_w_transform) for denorm pipeline to correctly reverse preprocessing Gaussianization"
  - "Inserted full_denorm_pipeline as new Cell 23 between utility functions and model definition"

patterns-established:
  - "Broadcasting pattern: noise shape (num_qubits, N) -> stack(list(results)).T -> (N, WINDOW_LENGTH)"
  - "All generator calls multiply by GEN_SCALE immediately after stacking"
  - "Evaluation block in train_qgan (not _train_one_epoch) fires every EVAL_EVERY epochs"
  - "full_denorm_pipeline() is the single denormalization path for both training eval and standalone generation"

requirements-completed: [BUG-02, BUG-03, PERF-04, PERF-05, QC-04, WGAN-04, WGAN-05, WGAN-08]

# Metrics
duration: 4min
completed: 2026-03-02
---

# Phase 02 Plan 03: Training Loop and Evaluation Pipeline Summary

**Broadcasting-based training loop with one-sided GP, raw EMD computation, and comprehensive stylized facts including kurtosis at both window and stitched levels**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-02T09:09:33Z
- **Completed:** 2026-03-02T09:14:15Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced per-sample generator loops with parameter broadcasting (single QNode call per batch, ~8x speedup)
- Implemented one-sided gradient penalty with per-sample alpha using actual_batch_size
- GEN_SCALE applied consistently in all three code paths: critic training, generator training, evaluation
- EMD now computed on raw 1D flattened arrays via wasserstein_distance (eliminates histogram binning artifacts)
- Added full_denorm_pipeline() as single denormalization function for training eval (and later standalone generation)
- Stylized facts rewritten with kurtosis, both stitched and window-level computation
- Evaluation block runs every EVAL_EVERY epochs with all four logging categories
- Dynamic histogram bins derived from real data range

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite critic and generator training steps with broadcasting and correct GP** - `6ddc49d` (feat)
2. **Task 2: Add evaluation pipeline with correct EMD, dynamic bins, and comprehensive stylized facts** - `1dd0ff1` (feat)

## Files Created/Modified
- `qgan_pennylane.ipynb` - Cell 3: added kurtosis import; New Cell 23: full_denorm_pipeline function; Cell 26: rewritten _train_one_epoch (broadcasting, one-sided GP), updated train_qgan (evaluation block), rewritten stylized_facts (kurtosis, window-level)

## Decisions Made
- Used `lambert_w_transform` (not `inverse_lambert_w_transform`) in denormalization pipeline: the preprocessing applies `inverse_lambert_w_transform` to Gaussianize data, so reversal requires `lambert_w_transform` which is described as "Transform the Gaussianized data back to its original state" in Cell 17. This matches the RESEARCH.md Pattern 4 and the original code's behavior.
- Inserted `full_denorm_pipeline` as a new cell (Cell 23) after utility functions rather than inside the class, since it references module-level functions (rescale, lambert_w_transform, denormalize) and will be used by standalone generation cells outside the class.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used lambert_w_transform instead of inverse_lambert_w_transform in denorm pipeline**
- **Found during:** Task 2 (denormalization pipeline implementation)
- **Issue:** Plan specified `inverse_lambert_w_transform` for the denormalization Lambert step, but this is mathematically incorrect. Preprocessing applies `inverse_lambert_w_transform` to Gaussianize; to reverse, we need `lambert_w_transform` which de-Gaussianizes.
- **Fix:** Used `lambert_w_transform(rescaled, delta)` in full_denorm_pipeline, matching RESEARCH.md Pattern 4 and the original training code's behavior.
- **Files modified:** qgan_pennylane.ipynb (new Cell 23)
- **Verification:** Function signature and pipeline order match RESEARCH.md and original code
- **Committed in:** 1dd0ff1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Corrected denormalization function to match mathematical pipeline. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training loop fully rewritten with broadcasting, ready for training execution
- Evaluation pipeline produces EMD values needed by Plan 02-04 early stopping
- full_denorm_pipeline() ready for use by standalone generation (Plan 02-04)
- Note: Cell indices shifted by 1 due to new Cell 23 insertion (class now at Cell 26, config at Cell 28)

## Self-Check: PASSED

- FOUND: qgan_pennylane.ipynb
- FOUND: 02-03-SUMMARY.md
- FOUND: 6ddc49d (Task 1 commit)
- FOUND: 1dd0ff1 (Task 2 commit)

---
*Phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign*
*Completed: 2026-03-02*
