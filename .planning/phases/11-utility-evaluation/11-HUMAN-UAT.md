---
status: partial
phase: 11-utility-evaluation
source: [11-VERIFICATION.md]
started: 2026-05-18T00:00:00Z
updated: 2026-05-18T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Reproducibility of fidelity_dualscale.json on a different machine
expected: Run `python revision/run_dualscale_fidelity.py` from a machine that is NOT `/Users/shawngibford/dev/phd/qGAN` (or after renaming/moving the repo) without setting `QGAN_CANONICAL_REPO`. Driver should emit `fidelity_dualscale.json` with `data_hash==91e447d4624e25b3` and 3360 rows. Currently `_resolve_run_dir` silently routes through the hardcoded fallback `/Users/shawngibford/dev/phd/qGAN` (run_dualscale_fidelity.py:112, CR-01); on another machine it fails with `FileNotFoundError` and no guidance. The current artifact is correct and verified on the author's machine — the open decision is whether to apply the CR-01 fix (replace the hardcoded constant with `os.environ["QGAN_CANONICAL_REPO"]`) before Phase 14 manuscript work so peers/CI can re-execute.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
