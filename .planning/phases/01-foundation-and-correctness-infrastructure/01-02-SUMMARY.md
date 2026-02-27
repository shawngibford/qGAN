---
phase: 01-foundation-and-correctness-infrastructure
plan: 02
subsystem: training-infrastructure
tags: [dataloader, pytorch, training-loop, memory, torch, pennylane]

# Dependency graph
requires:
  - phase: 01-01
    provides: "Safe checkpoint system and removal of unsafe code patterns"
provides:
  - "Proper batch-based DataLoader iteration (no flatten-to-list)"
  - "Loss values stored as Python floats via .item() (prevents computation graph retention)"
  - "Parameterized epoch condition using self.num_epochs"
  - "delta scoped as self.delta instance attribute (no global dependency)"
  - "Evaluation/inference code wrapped in torch.no_grad()"
affects: [01-foundation-and-correctness-infrastructure]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DataLoader with shuffle=False, drop_last=True for temporal data"
    - "_train_one_epoch receives real_batch tensor directly from DataLoader"
    - "All loss history appends use .item() to detach from graph"
    - "Evaluation block wrapped in torch.no_grad() context manager"
    - "Hyperparameters accessed as self.* inside class methods, not globals"

key-files:
  created: []
  modified:
    - "qgan_pennylane.ipynb"

key-decisions:
  - "Use real_batch.shape[0] for fake batch loop count (adapts to DataLoader's actual batch size including potential last-batch variation)"
  - "Store dataloader reference as self.dataloader on the model for potential reuse"

patterns-established:
  - "DataLoader yields (real_batch,) tuples where real_batch is (batch_size, window_length)"
  - "All training hyperparameters are self.* instance attributes, never bare globals"
  - "Evaluation/inference forward passes always under torch.no_grad()"

requirements-completed: [BUG-04, BUG-05, BUG-06, PERF-02, PERF-03]

# Metrics
duration: 3min
completed: 2026-02-27
---

# Phase 1 Plan 2: DataLoader Restructuring and Training Loop Bug Fixes Summary

**Proper batch-based DataLoader iteration replacing flatten-to-list, with .item() on loss appends, parameterized epoch check, self.delta scoping, and torch.no_grad() on evaluation passes**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-27T09:34:41Z
- **Completed:** 2026-02-27T09:38:17Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced broken DataLoader flatten-to-list pattern with direct batch iteration using shuffle=False, drop_last=True
- Fixed memory leak: loss values now stored as Python floats via .item() instead of retaining full computation graphs
- Scoped delta as self.delta instance attribute (constructor parameter with default=1), eliminating NameError risk from global variable dependency
- Wrapped evaluation/inference block in torch.no_grad() to prevent unnecessary gradient tracking during metrics computation
- Fixed hardcoded epoch condition (3000) to use self.num_epochs

## Task Commits

Each task was committed atomically:

1. **Task 1: Restructure DataLoader for proper batch sampling** - `02c9cbb` (fix)
2. **Task 2: Fix training loop bugs and add torch.no_grad()** - `f4167b4` (fix)

## Files Created/Modified
- `qgan_pennylane.ipynb` - Cell 23 (qGAN class): train_qgan() iterates DataLoader directly, _train_one_epoch() accepts real_batch tensor, loss appends use .item(), self.delta replaces bare delta, evaluation wrapped in torch.no_grad(), epoch condition parameterized. Cell 24: delta=1 added to constructor. Cell 25: DataLoader with BATCH_SIZE, shuffle=False, drop_last=True. Cell 26: train_qgan_with_early_stopping() uses DataLoader iteration. Cell 28: training call passes dataloader variable.

## Decisions Made
- Used `real_batch.shape[0]` instead of `self.batch_size` for the fake batch generation loop, so the fake batch size automatically matches whatever the DataLoader provides (handles drop_last edge cases)
- Stored dataloader reference as `self.dataloader` on the model instance for potential reuse by train_qgan() method

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Cell 26 train_qgan_with_early_stopping also had flatten-to-list**
- **Found during:** Task 1 (DataLoader restructuring)
- **Issue:** Cell 26's `train_qgan_with_early_stopping()` function had its own copy of the flatten-to-list pattern, separate from cell 23's `train_qgan()` method. The plan only explicitly mentioned cell 23.
- **Fix:** Applied the same DataLoader iteration pattern to cell 26, replacing its `gan_data_list` construction with direct `for (real_batch,) in gan_data:` iteration.
- **Files modified:** qgan_pennylane.ipynb (cell 26)
- **Verification:** `gan_data_list` not present anywhere in notebook
- **Committed in:** 02c9cbb

**2. [Rule 3 - Blocking] Cell 28 training call needed variable name update**
- **Found during:** Task 1 (DataLoader restructuring)
- **Issue:** Cell 28 passed `gan_data` to the training function, but cell 25 now creates a variable named `dataloader` instead of `gan_data`.
- **Fix:** Updated the `gan_data=gan_data` keyword argument to `gan_data=dataloader` in cell 28.
- **Files modified:** qgan_pennylane.ipynb (cell 28)
- **Verification:** Variable name matches cell 25's new DataLoader variable
- **Committed in:** 02c9cbb

---

**Total deviations:** 2 auto-fixed (2 blocking - additional flatten-to-list copy and variable name mismatch)
**Impact on plan:** Both auto-fixes necessary to avoid runtime errors. No scope creep.

## Issues Encountered
None beyond the deviations noted above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DataLoader and training loop are now correct, ready for Plan 03 (hyperparameter and metrics fixes)
- The `self.delta` pattern and `torch.no_grad()` convention are established for all future code
- Loss values stored as Python floats means the plotting/checkpoint code in cells 26, 29, 30 will work without `.item()` workarounds
- No training behavior was changed (same hyperparameters, loss functions, architecture)

## Self-Check: PASSED

- FOUND: qgan_pennylane.ipynb
- FOUND: 01-02-SUMMARY.md
- FOUND: 02c9cbb (Task 1 commit)
- FOUND: f4167b4 (Task 2 commit)

---
*Phase: 01-foundation-and-correctness-infrastructure*
*Completed: 2026-02-27*
