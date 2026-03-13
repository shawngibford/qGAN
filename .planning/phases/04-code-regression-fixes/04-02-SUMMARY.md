---
phase: 04-code-regression-fixes
plan: 02
subsystem: training
tags: [quantum-gan, pennylane, validation, emd, baseline-metrics, psd]

# Dependency graph
requires:
  - phase: 04-code-regression-fixes
    plan: 01
    provides: "Corrected noise range [0, 4pi], real PAR_LIGHT eval, ACF removal, HPO hyperparameters"
provides:
  - "200-epoch validation run with PASS outcome (EMD=0.001301)"
  - "Comprehensive baseline metrics in results/phase4_validation.json"
  - "PSD baseline for Phase 6 spectral loss comparison"
  - "Standalone validation script (scripts/phase4_validation.py)"
  - "Validation cell in notebook (cell 41)"
affects: [05-broadcasting, 06-spectral-loss, 07-hpo-rerun]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Standalone validation script pattern: replicate notebook pipeline for headless execution"
    - "Validation JSON structure with config/metrics/hpo_baseline/outcome for downstream comparison"

key-files:
  created:
    - "scripts/phase4_validation.py"
    - "results/phase4_validation.json"
  modified:
    - "qgan_pennylane.ipynb"

key-decisions:
  - "Used standalone Python script for validation execution (avoids running all 70 notebook cells headless)"
  - "Stored full PSD arrays (6 frequency bins) rather than just summary stats -- negligible storage, full downstream utility"
  - "Early stopping warmup set to 50 epochs (not 100) for 200-epoch run to allow sufficient monitoring window"
  - "HPO parameters transfer successfully to corrected [0, 4pi] noise range -- no fallback needed"

patterns-established:
  - "Validation JSON schema: phase/timestamp/git_hash/config/metrics/hpo_baseline/outcome"
  - "EMD evaluation via histogram-based wasserstein_distance on denormalized samples"

requirements-completed: [REG-01, REG-04, REG-05]

# Metrics
duration: 37min
completed: 2026-03-13
---

# Phase 4 Plan 2: Validation Run Summary

**200-epoch validation run with HPO hyperparameters on corrected [0, 4pi] noise range -- PASS outcome with EMD=0.001301 (within 2x of HPO baseline 0.001137)**

## Performance

- **Duration:** 37 min (28.3 min training + setup/verification)
- **Started:** 2026-03-13T18:41:27Z
- **Completed:** 2026-03-13T19:18:58Z
- **Tasks:** 1
- **Files created/modified:** 3

## Accomplishments
- 200-epoch validation run completed successfully with no divergence and no fallback needed
- Best EMD of 0.001301 achieved at epoch 51, confirming HPO parameters transfer to corrected [0, 4pi] noise range
- Comprehensive baseline metrics captured: EMD, moment statistics (mean/std/kurtosis for real vs fake), PSD comparison (6 frequency bins), training dynamics (loss curves, timing, EMD history)
- Validation JSON saved with full config for reproducibility (git hash, all hyperparameters, noise range)
- Standalone validation script created for repeatable execution outside notebook

## Validation Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Best EMD | 0.001301 | PASS (< 2x threshold of 0.002274) |
| HPO Baseline EMD | 0.001137 | Reference |
| EMD Ratio | 1.14x | Close to baseline |
| Epochs Completed | 200/200 | Full run |
| Best EMD Epoch | 51 | Relatively early convergence |
| Fallback Used | No | HPO params stable |

### Moment Comparison

| Moment | Real | Fake | Notes |
|--------|------|------|-------|
| Mean | 0.002449 | 0.060920 | Fake has positive bias |
| Std | 0.021745 | 0.042930 | Fake ~2x wider spread |
| Kurtosis | 1.399 | 0.115 | Fake less heavy-tailed |

### PSD Baseline

- Peak frequency match: True (both at bin 4)
- Total power ratio: 15.86 (fake has significantly more power -- expected with wider spread)
- 6 frequency bins stored for Phase 6 spectral loss comparison

## Task Commits

Each task was committed atomically:

1. **Task 1: Create validation cell and execute 200-epoch training run** - `41f8540` (feat)

## Files Created/Modified
- `scripts/phase4_validation.py` - Standalone validation script replicating full notebook pipeline
- `results/phase4_validation.json` - Complete validation results with config, metrics, and PASS outcome
- `qgan_pennylane.ipynb` - Validation cell added at cell 41 (after HPO retrain cell 40)

## Decisions Made
- Used standalone Python script rather than notebook execution (avoids headless execution of 70 cells with plotting/display dependencies)
- Early stopping warmup set to 50 epochs (vs 100 in main training) to maximize monitoring window in 200-epoch run
- Stored full PSD arrays (6 frequency bins from rfft on T=10 windows) rather than just summary statistics -- negligible storage cost, full utility for Phase 6 comparison
- HPO hyperparameters confirmed working on [0, 4pi] noise range -- no need for early fallback to v1.0 defaults

## Deviations from Plan

None - plan executed exactly as written. The standalone script approach was explicitly allowed by the plan ("Claude should determine the best execution approach").

## Issues Encountered

None. Training was stable throughout all 200 epochs with no NaN values, no critic loss explosion, and no need for auto-fallback.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 4 complete: code regressions fixed and validated
- results/phase4_validation.json provides baseline for Phases 5-7 comparison
- Key concern: fake sample moments show positive mean bias (0.061 vs 0.002) and wider spread (std 0.043 vs 0.022) -- Phase 5 broadcasting fix and Phase 6 spectral loss may help
- PSD power ratio of 15.86 indicates substantial amplitude mismatch that spectral loss (Phase 6) should address
- HPO parameters are validated for [0, 4pi] noise range -- no HPO re-run needed at this stage

## Self-Check: PASSED

- FOUND: scripts/phase4_validation.py
- FOUND: results/phase4_validation.json
- FOUND: qgan_pennylane.ipynb
- FOUND: 04-02-SUMMARY.md
- FOUND: 41f8540 (Task 1 commit)

---
*Phase: 04-code-regression-fixes*
*Completed: 2026-03-13*
