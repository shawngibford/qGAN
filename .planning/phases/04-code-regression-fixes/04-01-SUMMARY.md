---
phase: 04-code-regression-fixes
plan: 01
subsystem: training
tags: [quantum-gan, pennylane, noise-range, par-light, acf-removal, hpo]

# Dependency graph
requires:
  - phase: 03-post-processing
    provides: "Trained qGAN with WGAN-GP + IQP circuit"
provides:
  - "Corrected noise range [0, 4pi] in all 5 locations"
  - "Real PAR_LIGHT conditioning in eval (replaces par_zeros)"
  - "ACF loss completely removed from training"
  - "HPO-tuned hyperparameters set in cell 28"
  - "mu/sigma variable shadowing eliminated"
affects: [04-02-validation-run, 05-broadcasting, 06-spectral-loss]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PAR_LIGHT compression: reshape(num_qubits, 2).mean(dim=1) + remap to [0,1]"

key-files:
  created: []
  modified:
    - "qgan_pennylane.ipynb"

key-decisions:
  - "ACF loss code fully removed (not just zeroed) per user decision -- Phase 6 spectral loss replaces it"
  - "self.acf_avg eval metric tracking preserved for monitoring stylized facts"
  - "HPO-tuned values applied: N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05"

patterns-established:
  - "Eval uses same PAR_LIGHT compression pattern as training (reshape+mean+remap)"

requirements-completed: [REG-01, REG-04, REG-05]

# Metrics
duration: 3min
completed: 2026-03-13
---

# Phase 4 Plan 1: Code Regression Fixes Summary

**Fixed noise range to [0, 4pi] in all 5 locations, replaced par_zeros eval with real PAR_LIGHT conditioning, removed ACF loss entirely, updated hyperparameters to HPO-tuned values, and eliminated mu/sigma variable shadowing**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T18:35:18Z
- **Completed:** 2026-03-13T18:38:11Z
- **Tasks:** 2
- **Files modified:** 1 (qgan_pennylane.ipynb -- 6 cells modified)

## Accomplishments
- Noise range corrected from [0, 2pi] to [0, 4pi] in cells 26 (3 training loop locations), 29 (circuit diagram), and 45 (standalone generation)
- Eval block in cell 26 now samples real PAR_LIGHT from par_data_list using the same reshape+mean+remap pattern as critic/generator training
- ACF loss completely removed: constructor param, self.lambda_acf, diff_acf_lag1 static method, penalty block, combined loss line -- while preserving self.acf_avg eval metric tracking
- Cell 28 hyperparameters updated to HPO-tuned values with LAMBDA_ACF removed
- Cell 12 mu/sigma variable shadowing eliminated via inline computation in norm.pdf() and print statements
- Cell 40 HPO retrain cell cleaned of all lambda_acf references

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix qGAN class (cell 26) -- noise range, par_zeros eval, ACF removal** - `60b27e5` (fix)
2. **Task 2: Fix supporting cells -- mu/sigma shadowing, hyperparameters, circuit diagram, HPO retrain, standalone generation** - `d10cc05` (fix)

## Files Created/Modified
- `qgan_pennylane.ipynb` - All regression fixes applied across cells 12, 26, 28, 29, 40, 45

## Decisions Made
- ACF loss fully removed (not just zeroed) per user decision in 04-CONTEXT.md -- Phase 6 spectral loss will replace it
- self.acf_avg list and its population in the eval section preserved -- this tracks the ACF RMSE evaluation metric via stylized_facts(), separate from the training ACF penalty loss
- HPO-tuned hyperparameters applied exactly as specified: N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Notebook is ready for 04-02 validation run with corrected code
- All HPO-tuned hyperparameters are set in cell 28
- qGAN constructor no longer accepts lambda_acf parameter
- Note: HPO parameters were tuned on [0, 2pi] noise range -- validation run will determine if they transfer to [0, 4pi]

## Self-Check: PASSED

- FOUND: qgan_pennylane.ipynb
- FOUND: 04-01-SUMMARY.md
- FOUND: 60b27e5 (Task 1 commit)
- FOUND: d10cc05 (Task 2 commit)

---
*Phase: 04-code-regression-fixes*
*Completed: 2026-03-13*
