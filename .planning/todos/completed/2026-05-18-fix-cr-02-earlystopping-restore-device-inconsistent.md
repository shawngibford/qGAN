---
created: 2026-05-18T01:59:33Z
title: Fix CR-02 — EarlyStopping checkpoint restore device/dtype-inconsistent
area: general
resolves_phase: 13
source: 10-REVIEW.md (CR-02)
files:
  - core/training.py:163-171
---

## Problem

Code review of Phase 10 (`10-REVIEW.md`, finding CR-02) flagged the
`EarlyStopping` checkpoint-restore path in `core/training.py` as
**device/dtype-inconsistent with Adam optimizer state after the Phase-10 device
move**. `torch.load(weights_only=False)` is called with no `map_location`, so
optimizer/parameter state is restored to whatever device it was saved on, which
can mismatch the live `params_pqc` device. This contradicts the comment's claim
that the CPU/CUDA path reproduces Phase 09.1 exactly — the pre-MPS code was
device-uniform; this defect was introduced by the Phase-10 MPS device-move
change.

Dormant in Phase 10 (every one of the 50 baseline runs passed
`early_stopper=None`, so the restore path never executed and BASE-01/02/03 are
unaffected). Becomes live for any phase that re-enables EMD early stopping with
checkpoint restore — relevant to **Phase 13** and any later run that uses the
early-stopper.

## Solution

TBD — pass `map_location=device` (and re-cast restored tensors to the live
`compute_dtype`) when restoring the EarlyStopping checkpoint, then move
optimizer state to the active device so the restored state matches
`params_pqc`. Add a regression test that early-stops + restores on MPS and on
CPU and asserts device/dtype consistency. See
`.planning/phases/10-classical-baselines/10-REVIEW.md` for full detail.
