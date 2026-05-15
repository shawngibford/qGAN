---
phase: 09-documentation-bridge
plan: 05
subsystem: revision-verification
tags: [eval-06, round-trip, gradcheck, lambert-w, autograd, regression, phase-8-parity]

# Dependency graph
requires:
  - phase: 09-documentation-bridge
    plan: 01
    provides: differentiable inverse_lambert_w_transform with closed-form backward (gradcheck-validated)
  - phase: 08-core-module-extraction
    provides: revision/01_parity_check.ipynb (regression baseline; deltas=0.0)
provides:
  - revision/02_eval06_roundtrip.ipynb (verification harness for EVAL-06)
  - revision/results/eval06_roundtrip.json (artifact, pass=true)
  - Phase 8 parity regression confirmation (parity_check.json deltas still 0.0)
affects: [09.1-r1-m3-ablation, 11-tstr, 12-noise-gradients, 14-paper-methods]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bare autograd-Function round-trip test via lambert_w_transform(clip_disabled) pair"
    - "torch.autograd.gradcheck as a numerical-Jacobian sanity check on closed-form backward"
    - "True pipeline-level round-trip via full_denorm_pipeline(d['windowed_data']) vs rolling_window(d['log_delta']) reference (D-04b — not a bare-pair masquerade)"
    - "Regression-via-re-execute: re-run prior phase's verification notebook to certify zero drift"

key-files:
  created:
    - revision/02_eval06_roundtrip.ipynb
    - revision/results/eval06_roundtrip.json
  modified:
    - revision/01_parity_check.ipynb  (re-executed inplace; only execution_count + outputs updated)
    - revision/results/parity_check.json  (re-regenerated; only git_sha_pre/post field changed; all numerical metrics identical)

key-decisions:
  - "Bare-pair clip override: pass clip_low=-1e20, clip_high=1e20 to lambert_w_transform in the round-trip cell. The default training-stability clip [-12, 11] saturates ≥2 real norm_log_delta outliers (|norm|>3.80) which can never round-trip. The clip is preserved everywhere else."
  - "Full-pipeline check (D-04b) uses rolling_window(d['log_delta'], 10, 2).reshape(-1) as the elementwise reference — guarantees the test is a TRUE end-to-end pipeline round-trip, not a wrapper around the bare Lambert pair."
  - "Backup file revision/results/parity_check.json.pre-phase9.bak created, used for byte-for-byte equality check, then deleted (no clutter)."

requirements-completed: [EVAL-06]

# Metrics
duration: 28min
completed: 2026-05-15
---

# Phase 09 Plan 05: EVAL-06 Round-Trip Verification + Phase 8 Parity Regression Summary

**Created the EVAL-06 verification harness (`revision/02_eval06_roundtrip.ipynb`), executed it to produce `revision/results/eval06_roundtrip.json` with `pass=true` on all four locked tolerances, and re-ran `revision/01_parity_check.ipynb` to certify the plan 09-01 in-place autograd-Function replacement preserves Phase 8 parity bit-identically (all four deltas remain 0.0).**

## Performance

- **Duration:** ~28 min
- **Started:** 2026-05-15T16:55:16Z
- **Completed:** 2026-05-15
- **Tasks:** 3 (all autonomous)
- **Files created:** 2 (`revision/02_eval06_roundtrip.ipynb`, `revision/results/eval06_roundtrip.json`)
- **Files modified:** 2 (`revision/01_parity_check.ipynb` re-executed; `revision/results/parity_check.json` git_sha refresh only)

## Accomplishments

### Task 1 — `revision/02_eval06_roundtrip.ipynb` created (commit `18c387d`)

5-cell notebook (1 markdown + 4 code), following the `revision/01_parity_check.ipynb` template pattern:

- **Cell 1 (markdown):** Phase 9 EVAL-06 overview with all four check descriptions and target tolerances.
- **Cell 2 (`2c8bc6c2`):** Repo-root finder + sys.path insert + `os.chdir(REPO_ROOT)` + deterministic seed (`SEED=42`, `torch.manual_seed`, `np.random.seed`). Verbatim pattern from `01_parity_check.ipynb` cell 2c8bc6c2.
- **Cell 3 (`5d83ed4a`):** Imports from `revision.core.data` (`load_and_preprocess`, `inverse_lambert_w_transform`, `lambert_w_transform`, `full_denorm_pipeline`, `rolling_window`) + `WINDOW_LENGTH` from `revision.core`; loads real data and asserts shape (777 log_delta, (384, 10) windows).
- **Cell 4 (`b1c3c8fe`):** The four verification checks (see below).
- **Cell 5 (`a28db61d`):** Artifact assembly + JSON write + hard-assert `passed` + `git rev-parse HEAD` lookup. Schema matches 09-PATTERNS.md JSON artifact section: `{delta, tolerance, pass, seed, git_sha, notes}`.

### Task 2 — Executed and produced `eval06_roundtrip.json` (commit `0a638d0`)

Ran via `qgan_env/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace revision/02_eval06_roundtrip.ipynb --ExecutePreprocessor.timeout=600`. (Note: `qgan_env/bin/jupyter` has a stale shebang from a renamed parent dir; used `python3 -m jupyter` fallback.)

After Rule-1 deviation fix (see "Deviations from Plan" below), all four checks pass:

| Check | Measured delta | Tolerance | Status |
|---|---|---|---|
| Synthetic round-trip (`torch.randn(777, fp64)`) | **4.440892e-16** | 1e-8 | PASS |
| Real round-trip (full 777-elem `norm_log_delta`) | **4.440892e-16** | 1e-8 | PASS |
| Full-pipeline round-trip (D-04b, `full_denorm_pipeline` vs `rolling_window(log_delta)`) | **4.803323e-09** | 1e-6 | PASS |
| `torch.autograd.gradcheck` (eps=1e-6, atol=1e-6) | — | True | **PASS** |

`pass: true`. `git_sha: "18c387d351bb3e5a26b3e18316adc8c688c0be40"`. `seed: 42`.

### Task 3 — Phase 8 parity regression confirmed (commit `917e755`)

Re-executed `revision/01_parity_check.ipynb` via the same nbconvert path. `revision/results/parity_check.json` shows:

```
pre  = {emd: 0.12048789057906201, mean: 0.0017183494914040196,
        std: 0.1710686770721286, kurtosis: -0.039478752608490986}
post = {emd: 0.12048789057906201, mean: 0.0017183494914040196,
        std: 0.1710686770721286, kurtosis: -0.039478752608490986}
delta = {emd: 0.0, mean: 0.0, std: 0.0, kurtosis: 0.0}
pass  = true
```

A byte-for-byte equality check against `revision/results/parity_check.json.pre-phase9.bak` (snapshot created before re-execution) confirmed **zero drift** in all numerical fields; only `git_sha_pre`/`git_sha_post` updated to the current HEAD. The backup file was then deleted (no clutter per plan recommendation).

## Acceptance Criteria Gates

### Task 1 (notebook structure)

| Gate | Expected | Actual |
|---|---|---|
| `test -f revision/02_eval06_roundtrip.ipynb` | exists | OK |
| `python3 -c "import json; json.load(...)"` exits 0 | OK | OK |
| `grep -cE '(^\|[^_])lambert_w_transform\('` >= 1 | ≥1 | 2 |
| `grep -c 'inverse_lambert_w_transform'` >= 1 | ≥1 | 5 |
| `grep -c 'torch\.autograd\.gradcheck'` >= 1 | ≥1 | 2 |
| `grep -c 'full_denorm_pipeline'` >= 1 | ≥1 | 5 |
| `grep -cE '1e-6\|1.0e-6\|1e-06'` >= 1 | ≥1 | 5 |
| `grep -c 'rolling_window'` >= 1 | ≥1 | 2 |
| `grep -c 'eval06_roundtrip.json'` >= 1 | ≥1 | 2 |
| `grep -c '_find_repo_root'` >= 1 | ≥1 | 2 |
| `grep -c '_git_sha'` >= 1 | ≥1 | 2 |
| `grep -c 'load_and_preprocess'` >= 1 | ≥1 | 4 |
| Cell count >= 5 | ≥5 | 5 |

### Task 2 (executed artifact)

| Gate | Expected | Actual |
|---|---|---|
| File exists | yes | yes |
| `"pass": true` | true | true |
| `"seed": 42` | 42 | 42 |
| `"git_sha"` non-empty, ≠ `"unknown"` | ok | `18c387d3…0be40` |
| `delta.synthetic ≤ 1e-8` | ≤1e-8 | 4.44e-16 |
| `delta.real ≤ 1e-8` | ≤1e-8 | 4.44e-16 |
| `delta.full_pipeline ≤ 1e-6` | ≤1e-6 | 4.80e-9 |
| `delta.gradcheck_passed == true` | true | true |
| Schema keys present | `{delta, tolerance, pass, seed, git_sha, notes}` | all six present |

### Task 3 (Phase 8 regression)

| Gate | Expected | Actual |
|---|---|---|
| `revision/results/parity_check.json` exists | yes | yes |
| `"pass": true` | true | true |
| `revision/01_parity_check.ipynb` valid JSON | yes | yes |
| No error cells after re-execution | 0 errors | 0 errors |
| All deltas unchanged from Phase 8 baseline | =0.0 (all four) | =0.0 (all four) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in verification harness] `lambert_w_transform` default clipping saturates real outliers, breaks bare-pair round-trip**

- **Found during:** Task 2 (first execution of the notebook).
- **Issue:** The first execution returned `delta.real = 9.81e-2 >> 1e-8`. Investigation: `revision.core.data.lambert_w_transform` clamps its output to `[clip_low=-12.0, clip_high=11.0]` for numerical stability of the `exp(δ/2·x²)` term in the training-data path. Two real values in `d["norm_log_delta"]` (indices 67 and 4, with `norm ≈ 3.90` and `3.83`) map to `lambert_w_transform = 11.0` after the clip. The inverse Lambert W is bijective on the principal branch, so it has no way to distinguish a genuine `+11` from a clip-saturated `+11`; both indices round-trip to ~3.802 (the input value that produces forward = +11 before clipping). This is **not** a bug in the autograd Function; it is the bare-pair test conflating two distinct effects (Lambert W correctness vs. forward-path clipping).
- **Fix:** In Cell 4 (`b1c3c8fe`), pass `clip_low=-1e20, clip_high=1e20` (effectively disabled) to `lambert_w_transform` **only for the two bare-pair checks** (synthetic and real round-trip). The clipping is preserved everywhere else:
  - `full_denorm_pipeline` (check 4) still calls `lambert_w_transform` with default clips inside.
  - The training-data path (`load_and_preprocess` and downstream training scripts) is untouched.
- **Code change:**
  ```python
  _BIG = 1.0e20
  y_synth = lambert_w_transform(x_synth, delta_const, clip_low=-_BIG, clip_high=_BIG)
  y_real  = lambert_w_transform(real,    delta_const, clip_low=-_BIG, clip_high=_BIG)
  ```
  Added a 9-line explanatory comment block above the change.
- **Files modified:** `revision/02_eval06_roundtrip.ipynb` (cell `b1c3c8fe` source only).
- **Verification:** After fix, re-executed → `delta.real = 4.44e-16` (16 orders of magnitude tighter than required tolerance). Full pipeline check unchanged (4.80e-9, well under 1e-6).
- **Why this is correct:** The plan's `<must_haves><truths>` says "Real-data round-trip max|inverse(forward(log_delta_norm)) − log_delta_norm| ≤ 1e-8 on the full 777-element real norm_log_delta tensor". The clipping is a numerical-stability guard on the training-data forward path, not part of the Lambert W mathematical pair. Measuring the autograd Function's bare-pair correctness REQUIRES disabling the clip; otherwise the test conflates two unrelated effects. The plan's tasks were silent on this — the truths only locked the math, not the helper function's defaults. The fix follows the principle "test the thing you mean to test."
- **Committed in:** `0a638d0` (Task 2 commit; the deviation was discovered during Task 2 execution, not Task 1).

---

**Total deviations:** 1 auto-fixed (1 Rule 1 — verification-harness bug, not implementation bug)
**Impact on plan:** Test only. No production-code change. The differentiable `inverse_lambert_w_transform` from plan 09-01 is unchanged and unchallenged by this finding.

## Issues Encountered

### Environment quirk

The `qgan_env/bin/jupyter` shebang points to `/Users/shawngibford/dev/qml/qGAN/qgan_env/bin/python` (an old, no-longer-existing path — the project moved from `dev/qml/qGAN/` to `dev/phd/qGAN/`). Direct invocation fails with `bad interpreter`. **Workaround:** Use `qgan_env/bin/python3 -m jupyter nbconvert ...` instead. This works because `python3 -m jupyter` resolves jupyter via the Python module loader and bypasses the broken script shebang. Both Task 2 and Task 3 used this workaround successfully. No code change needed; documenting for future executors.

## Task Commits

1. **Task 1: Create `revision/02_eval06_roundtrip.ipynb`** — `18c387d` (feat)
2. **Task 2: Execute notebook, produce `eval06_roundtrip.json` pass=true** — `0a638d0` (feat; includes the Rule 1 bare-pair-clip fix)
3. **Task 3: Re-execute `01_parity_check.ipynb`, confirm Phase 8 zero-drift parity** — `917e755` (test)

## Files Created/Modified

- **Created:**
  - `revision/02_eval06_roundtrip.ipynb` (5 cells, ~190 lines of JSON when prettified)
  - `revision/results/eval06_roundtrip.json` (force-added past `.gitignore` for `results/`, matching the prior Phase 8 INFRA-02 commit `c21a90a` pattern for `parity_check.json`)
- **Modified:**
  - `revision/01_parity_check.ipynb` — re-executed via nbconvert; `execution_count` numbers and `outputs` cell metadata regenerated; **no source-code changes**.
  - `revision/results/parity_check.json` — `git_sha_pre`/`git_sha_post` field updated from `79a24cb…1a3a8` to `0a638d0…fa3a5`; all numerical fields (`pre`, `post`, `delta`, `pass`, `tolerance`, `seed`, etc.) byte-identical to pre-Phase-9 baseline (verified by Python equality check vs `.bak` snapshot).

## Decisions Made

- **Bare-pair clip override scope:** Limited to the two bare round-trip checks (synthetic, real); kept the default clip everywhere else, including inside `full_denorm_pipeline`. Rationale: the Phase 8 parity guarantees rest on the clipped forward, and the D-04b pipeline check must run end-to-end-faithful (not idealized).
- **Full-pipeline reference choice:** Used `rolling_window(d["log_delta"].double(), WINDOW_LENGTH, 2).reshape(-1)` as the element-wise reference (length 3840 = 384 × 10). Alternative (chosen against): repeat the windowing inside the test cell as a literal copy of `revision/core/data.py:281–282`. The chosen approach uses the same `rolling_window` function imported from the production code, so any future change to windowing logic is reflected by the test automatically.
- **Backup-file disposition:** Created `.pre-phase9.bak`, used for diff-comparison, then deleted. Kept the SUMMARY text documenting the equality check (so the audit trail survives without persistent clutter).
- **Re-execution rather than separate harness:** Re-executed the existing `revision/01_parity_check.ipynb` (in-place) rather than building a separate regression script. The notebook is small, deterministic, and the canonical Phase 8 attestation; re-executing it is the most defensible regression check possible.

## Threat Model Mitigations Applied

| Threat ID | Mitigation | Where |
|-----------|------------|-------|
| T-09-19 (silent test failure) | Hard `assert passed, ...` on the last line of cell `a28db61d`; nbconvert exits non-zero on failure | `revision/02_eval06_roundtrip.ipynb` cell 5 |
| T-09-20 (provenance loss) | `git_sha = git rev-parse HEAD` recorded in artifact; `seed=42` pinned; `notes` field populated | cell `a28db61d` |
| T-09-21 (NaN gradient at x≈0) | `torch.autograd.gradcheck` (eps=1e-6, atol=1e-6) on a 20-element sample passes — analytic backward matches finite-difference | cell `b1c3c8fe` check (3) |
| T-09-22 (forward-output drift breaking Phase 8 parity) | Re-executed `01_parity_check.ipynb`; all four deltas remain exactly 0.0 (Task 3) | `revision/results/parity_check.json` |
| T-09-23 (network calls) | Accepted: `_git_sha()` is local-only `git rev-parse HEAD` | no network code |
| T-09-24 (path injection on output JSON) | Hardcoded `Path("revision/results/eval06_roundtrip.json")` relative to repo root | cell `a28db61d` |
| T-09-25 (full-pipeline rt is bare-pair masquerade) | Check 4 uses `full_denorm_pipeline(d["windowed_data"], …)` compared against `rolling_window(d["log_delta"], 10, 2).reshape(-1)` — TRUE pipeline round-trip per D-04b, not a re-run of the bare Lambert pair | cell `b1c3c8fe` check (4) |

## Threat Flags

None — no new security-relevant surface introduced beyond the threat model.

## Self-Check

**Files claimed created/modified:**

- `revision/02_eval06_roundtrip.ipynb` — `[ -f ] && echo FOUND` → FOUND
- `revision/results/eval06_roundtrip.json` — `[ -f ] && echo FOUND` → FOUND
- `revision/01_parity_check.ipynb` modification (re-execute) — `git diff --stat HEAD~1 HEAD~0 | grep 01_parity_check.ipynb` → confirmed
- `revision/results/parity_check.json` modification (git_sha refresh only) — confirmed (only 2-line diff)

**Commits claimed:**

- `18c387d` → `git log --oneline | grep 18c387d` → `18c387d feat(09-05): add EVAL-06 round-trip verification notebook` → FOUND
- `0a638d0` → `git log --oneline | grep 0a638d0` → `0a638d0 feat(09-05): execute EVAL-06 round-trip notebook; pass=true` → FOUND
- `917e755` → `git log --oneline | grep 917e755` → `917e755 test(09-05): re-execute 01_parity_check.ipynb; Phase 8 parity preserved` → FOUND

**Plan-level `<verification>` re-run:**

- All three task `<verify>` blocks exit 0 — confirmed in Bash output
- `revision/results/eval06_roundtrip.json` exists with `pass=true` and all deltas strictly within tolerances — confirmed (4.44e-16, 4.44e-16, 4.80e-9)
- `revision/results/parity_check.json` (regression-checked) still `pass=true` — confirmed
- No raised exceptions in either notebook's executed output cells — `error` output count = 0 in both
- `gradcheck` step confirmed analytic backward matches numerical Jacobian to atol=1e-6 — confirmed (`gradcheck_passed = true`)
- Check 4 is a TRUE `full_denorm_pipeline` round-trip — confirmed (calls `full_denorm_pipeline(d["windowed_data"], …)` and compares against `rolling_window(d["log_delta"], 10, 2).reshape(-1)`)

## Self-Check: PASSED

## Next Phase Readiness

- **Plan 09-06+ (DOC-01 training_protocol.md, DOC-02 dataset_stats.md, preprocessing.py scaffold):** all unblocked. The EVAL-06 verification harness is the canonical proof point that the differentiable inverse Lambert W is mathematically sound, and the artifact JSON is ready for citation in the Phase 9 final output.
- **Phase 09.1 (R1-M3 ablation):** `revision/02_eval06_roundtrip.ipynb` is the **template** for ABL-01 round-trip sanity tests on the other four forward/inverse pairs (`forward_logreturns`/`inverse_logreturns`, `forward_minmax_od`/`inverse_minmax_od`). Phase 09.1 should clone this notebook, swap in the pair under test, and assert the same `pass=true` schema. The clip-disabled bare-pair pattern is generic.
- **Phase 11 (TSTR) and Phase 12 (noise gradients):** can now backprop OD-scale losses through `inverse_lambert_w_transform` with confidence that the gradient flow is numerically correct.
- **No blockers** for downstream phases.

---
*Phase: 09-documentation-bridge*
*Completed: 2026-05-15*
