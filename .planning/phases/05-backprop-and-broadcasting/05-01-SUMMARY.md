---
phase: 05-backprop-and-broadcasting
plan: 01
subsystem: quantum-generator
tags: [pennylane, backprop, broadcasting, qnode, batching]

# Dependency graph
requires:
  - phase: 04-code-regression-fixes
    provides: correct [0, 4pi] noise range and real PAR_LIGHT conditioning
provides:
  - backprop QNode with shots=None device for analytic gradient computation
  - batched QNode calls at all four loop sites (critic, generator, eval Cell 26, eval Cell 41)
  - broadcasting-ready noise shape (num_qubits, batch_size) at all call sites
affects: [05-02-validation, 06-spectral-loss]

# Tech tracking
tech-stack:
  added: []
  patterns: [pennylane-broadcasting, batched-qnode-calls, torch-stack-transpose-pattern]

key-files:
  created: []
  modified: [qgan_pennylane.ipynb]

key-decisions:
  - "backprop replaces parameter-shift due to PennyLane #4462 broadcasting gradient bugs"
  - "Noise shape (num_qubits, batch_size) not (batch_size, num_qubits) per PennyLane broadcasting spec"
  - "torch.stack(list(results)).T pattern used for all batched QNode output reshaping"
  - "fake_windows_list eliminated from Cell 41 PSD section in favor of batched tensor iteration"

patterns-established:
  - "Batched QNode call: noise (num_qubits, N), par_light (num_qubits, N), results via torch.stack(list(r)).T"
  - "PAR_LIGHT compression: reshape(N, num_qubits, 2).mean(dim=2).float().T for broadcasting"

requirements-completed: [REG-02, REG-03]

# Metrics
duration: 3min
completed: 2026-03-18
---

# Phase 5 Plan 1: Backprop and Broadcasting Summary

**Switched QNode from parameter-shift to backprop and converted all four per-sample Python loops to single batched QNode calls using PennyLane broadcasting**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-18T19:29:11Z
- **Completed:** 2026-03-18T19:33:10Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- QNode diff_method switched to backprop with shots=None device (PennyLane #4462 fix)
- Critic training loop: per-sample for-loop replaced with single batched QNode call
- Generator training loop: per-sample for-loop replaced with single batched QNode call with gradient flow preserved
- Cell 26 evaluation loop: per-sample for-loop replaced with single batched QNode call
- Cell 41 evaluation loop: per-sample for-loop replaced with single batched QNode call
- All isinstance/type-conversion boilerplate removed from converted locations
- Shape comments added at each batched QNode call site

## Task Commits

Each task was committed atomically:

1. **Task 1: Switch QNode to backprop and add shots=None** - `fa811ee` (feat)
2. **Task 2: Convert critic and generator training loops to batched calls** - `61996ea` (feat)
3. **Task 3: Convert evaluation loop (Cell 26) and Cell 41 eval loop** - `51853f3` (feat)

## Files Created/Modified
- `qgan_pennylane.ipynb` - Cell 26: backprop QNode, shots=None device, batched critic/generator/eval loops; Cell 41: batched eval loop and updated PSD section

## Decisions Made
- backprop chosen over parameter-shift because PennyLane issue #4462 documents that parameter-shift explicitly disallows gradients through broadcasted tapes
- Noise shape follows PennyLane convention: (num_qubits, batch_size) with batch dimension as LAST axis
- `torch.stack(list(results)).T` pattern adopted for converting batched QNode tuple output to (batch_size, window_length) tensor
- Cell 41 PSD section consolidated: two-line fake_windows_denorm + torch.stack replaced with single-line comprehension over batched fake_windows tensor

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Notebook is ready for Plan 02 validation: equivalence testing (batched vs sequential output within 1e-6), same-seed reproducibility, and SC4 timing gate
- All four loop sites converted; no per-sample loops remain in training or evaluation code
- Gradient flow through params_pqc preserved in generator training (no torch.no_grad() wrapping)

## Self-Check: PASSED

All files and commits verified:
- `qgan_pennylane.ipynb` exists with all changes applied
- `fa811ee` (Task 1), `61996ea` (Task 2), `51853f3` (Task 3) all present in git log
- Final verification script confirms all 11 success criteria pass

---
*Phase: 05-backprop-and-broadcasting*
*Completed: 2026-03-18*
