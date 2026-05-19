---
created: 2026-05-18T01:59:33Z
title: Fix CR-01 — spectral-loss hook non-differentiable + device-unsafe
area: general
resolves_phase: 13
source: 10-REVIEW.md (CR-01)
files:
  - revision/core/training.py:356-360
  - revision/core/training.py:470-507
---

## Problem

Code review of Phase 10 (`10-REVIEW.md`, finding CR-01) flagged the opt-in
`_spectral_psd_loss` hook in `revision/core/training.py` as **non-differentiable
w.r.t. `params_pqc`**: `mse` is a frozen Python float derived from detached numpy
arrays, and the `mse * var / var.detach()` construction carries zero
PSD-mismatch gradient. The docstring tells callers they can "opt back in", but
doing so would silently train the wrong objective. It is also **device-unsafe
after the Phase-10 MPS device move** — a CPU target tensor is compared against a
device-resident generator output.

Dormant in Phase 10 (every one of the 50 baseline runs used
`spectral_loss_weight=0.0`, so the hook never executed and BASE-01/02/03 are
unaffected). It becomes live for **Phase 13 (Architecture & Introspection)**,
which exercises the spectral/callback path — must be fixed before that phase
relies on it.

## Solution

TBD — make the spectral PSD loss a real differentiable torch term computed on
the device-resident generator output (no detach()/numpy round-trip for the
gradient path), and ensure the target/reference PSD tensor is moved to the
generator's device + `compute_dtype`. Add a unit test asserting a non-zero
gradient flows into `params_pqc` when `spectral_loss_weight > 0`. See
`.planning/phases/10-classical-baselines/10-REVIEW.md` for full detail.
