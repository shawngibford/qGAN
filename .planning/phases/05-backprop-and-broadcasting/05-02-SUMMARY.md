---
phase: 05-backprop-and-broadcasting
plan: 02
subsystem: quantum-generator
tags: [validation, sc4, broadcasting, timing, equivalence]

# Dependency graph
requires:
  - phase: 05-backprop-and-broadcasting/01
    provides: backprop QNode and batched broadcasting at all 4 loop sites
provides:
  - equivalence verification that batched output matches sequential within 1e-6
  - SC4 investigation results documenting multi-expval broadcasting limitation
affects: [06-spectral-loss]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: [qgan_pennylane.ipynb]

key-decisions:
  - "SC4 hard gate waived: multi-expval circuit return type prevents PennyLane broadcasting vectorization"
  - "Backprop-only gain accepted: backprop is correct for gradient flow even without vectorized speedup"
  - "Broadcasting syntax retained in codebase for future PennyLane improvements"
  - "Circuit architecture (10 separate qml.expval returns) preserved to avoid training quality risk"

patterns-established: []

requirements-completed: []

# Metrics
duration: 15min
completed: 2026-03-19
---

# Phase 5 Plan 2: Validation and SC4 Gate Summary

**Validation cells inserted, run, and removed. Equivalence test passed. SC4 timing gate failed — accepted with backprop-only gain.**

## Performance

- **Duration:** ~15 min (including user validation run)
- **Completed:** 2026-03-19
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Three temporary validation cells inserted after Cell 46 (equivalence, same-seed mini-run, SC4 timing)
- Equivalence test: batched output matches sequential within tolerance
- SC4 timing gate: FAILED — ratio=103.7% (post-broadcasting epoch time is ~equal to pre-broadcasting)
- Root cause identified: circuit returns tuple of 10 separate qml.expval() calls, preventing PennyLane from vectorizing the batch simulation
- Secondary issue: benchmark asymmetry (pre-broadcasting baseline omitted evaluation loop)
- User decision: accept backprop-only gain, proceed to Phase 6
- Validation cells removed from notebook after investigation

## Task Commits

1. **Task 1: Add validation cells** - `58d675c` (feat)
2. **Task 2: Remove validation cells after SC4 investigation** - `a1f1825` (feat)

## Files Created/Modified
- `qgan_pennylane.ipynb` - Validation cells inserted then removed; notebook restored to post-Plan-01 state

## Decisions Made
- SC4 hard gate waived after investigation revealed the multi-expval return type (10 separate PauliX/PauliZ measurements) prevents PennyLane's broadcasting engine from vectorizing the simulation
- The backprop switch (REG-02) remains correct and necessary for proper gradient flow
- The batched call syntax (REG-03) is in place and will benefit from future PennyLane optimization of multi-measurement broadcasting
- Circuit architecture preserved as-is to avoid training quality regression risk

## Deviations from Plan

**SC4 gate failed** — plan specified this as a hard gate for Phase 5. After investigation:
- Root cause: circuit `return (*measurements,)` pattern with 10 separate `qml.expval()` calls forces PennyLane to evaluate each observable independently rather than vectorizing
- The research phase verified speedup on single-return circuits; the actual notebook circuit has structurally different return type
- User accepted backprop-only outcome and waived SC4

## Issues Encountered
- SC4 timing ratio: 103.7% (no speedup from broadcasting)
- Benchmark flaw: pre-broadcasting `_sequential_epoch` omitted evaluation loop, making comparison unfair
- Multi-expval tuple return is a known PennyLane limitation for broadcasting vectorization

## Self-Check: PASSED (with SC4 waiver)

- Validation cells inserted and removed: verified no ONE-TIME cells remain
- Commits present: `58d675c`, `a1f1825`
- SC4 waived by user decision after root cause investigation

---
*Phase: 05-backprop-and-broadcasting*
*Completed: 2026-03-19*
