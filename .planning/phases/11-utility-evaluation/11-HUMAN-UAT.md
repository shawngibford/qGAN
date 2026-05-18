---
status: resolved
phase: 11-utility-evaluation
source: [11-VERIFICATION.md]
started: 2026-05-18T00:00:00Z
updated: 2026-05-18T00:00:00Z
---

## Current Test

[resolved — CR-01 code fix applied in 11-06 (61c4eb4); the blocking code decision is done. Optional cross-machine human spot-check remains nice-to-have, non-blocking.]

## Tests

### 1. Reproducibility of fidelity_dualscale.json on a different machine
expected: Run `python revision/run_dualscale_fidelity.py` from a machine that is NOT `/Users/shawngibford/dev/phd/qGAN` (or after renaming/moving the repo) without setting `QGAN_CANONICAL_REPO`. Driver should emit `fidelity_dualscale.json` with `data_hash==91e447d4624e25b3` and 3360 rows.
result: RESOLVED (code) — CR-01 fixed in 11-06 (61c4eb4). The hardcoded `/Users/shawngibford/dev/phd/qGAN` fallback is removed; `_CANONICAL_REPO_FALLBACK` is now an opt-in `QGAN_CANONICAL_REPO` env resolver (None when unset). Off-box without the env var, `_resolve_run_dir` now raises a `FileNotFoundError` that names `QGAN_CANONICAL_REPO` and cites D-11-08 (functionally verified on this machine: env-unset → fallback is None and the raised message contains both `QGAN_CANONICAL_REPO` and `D-11-08`); setting `QGAN_CANONICAL_REPO` to a checkout containing the frozen bundles reproduces the artifact. A single-root provenance assertion now blocks silent cross-checkout mixing. The blocking code decision (apply CR-01 before Phase 14) is DONE. A literal second-machine human run is an optional confirmation, not a phase blocker.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
