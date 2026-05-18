---
phase: 12-sensitivity-analysis
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - revision/run_sensitivity.py
  - revision/run_multiseed_rollup.py
  - revision/run_sensitivity_sweep.sh
findings:
  critical: 3
  warning: 5
  info: 4
  total: 12
status: resolved
resolution:
  critical_resolved: 3
  resolved_in:
    - "CR-01: 80208f6 — log_return real reference aligned to frozen Phase 11 recipe (d_real['log_delta']); full 66-cell sweep re-run + headline JSONs re-aggregated; B/42 log_return EMD reconciles to 1.53e-16 (was 0.58); OD unchanged (6.94e-18)"
    - "CR-02: 0607f08 — --seed choices=[42..46] + fail-fast checkpoint-exists guard before any artifact path is built"
    - "CR-03: 0607f08 — is_complete() content-validates metrics.json/samples.npy instead of non-emptiness only; all 66 bundles pass"
  warnings_status: "advisory — not addressed this pass; tracked as review debt"
---

# Phase 12: Code Review Report

> **RESOLUTION (2026-05-18):**   All 3 BLOCKER findings fixed and verified.
> CR-01 (numerical faithfulness) was the critical one: the log_return real
> reference now matches `run_dualscale_fidelity.build_real_references`, the
> full sweep was re-run, and the two headline JSONs were re-aggregated and
> reconciled against the frozen artifacts to floating-point precision. CR-02
> and CR-03 are harness-robustness hardening (no scientific output change).
> The 5 WARNING / 4 INFO findings remain advisory review debt.

**Reviewed:** 2026-05-18
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the Phase 12 sensitivity-analysis deliverables: `run_sensitivity.py`
(per-cell quantum inference driver + `--emit-rollup` aggregation),
`run_multiseed_rollup.py` (stdlib multiseed aggregator), and
`run_sensitivity_sweep.sh` (xargs-parallel orchestrator).

The stated core correctness property is **numerical faithfulness to frozen
reference artifacts**. The review focused there and found a serious divergence:
the Phase 12 log-return ("transformed") scale recompute does NOT use the same
real reference array or the same fake array as the frozen Phase 11
`run_dualscale_fidelity.py` recipe that produced `fidelity_dualscale.json` —
the very artifact `run_multiseed_rollup.py` consumes. The Phase 12 log-return
numbers will therefore NOT reconcile with the frozen baseline, contradicting
the file's own D-12-03 "fidelity metrics recomputed UNCHANGED" claim. There is
also an argparse falsy-seed bug that silently mis-routes valid invocations, and
a parallel-mode status-file race that can corrupt `sweep_status.json` and the
resume invariant.

## Critical Issues

### CR-01: log-return scale uses wrong real reference + wrong fake array — breaks numerical faithfulness

**File:** `revision/run_sensitivity.py:550-566`
**Issue:**
The dual-scale recompute claims to reproduce the unchanged fidelity suite
(D-12-03), but the log-return branch diverges from the **frozen Phase 11
canonical recipe** (`revision/run_dualscale_fidelity.py:439-516`,
`build_real_references` at `:289-302`) that produced the frozen
`fidelity_dualscale.json` consumed by `run_multiseed_rollup.py`. Three
concrete divergences:

1. **Real reference array is wrong.** Phase 12 builds the real log-return
   reference as
   `forward_logreturns(d_real["OD"])` then `rolling_window(..., 10, 2)`
   (`run_sensitivity.py:554-556`). The frozen Phase 11 recipe uses
   `d_real["log_delta"]` directly (`run_dualscale_fidelity.py:298`,
   `_log_return_rows:472`) — explicitly documented as "the EXACT array
   `_build_baseline_notebook.py:290` uses ... so numbers reconcile with
   `baseline_comparison.json`". `forward_logreturns` and `compute_log_delta`
   are NOT the same transform: `compute_log_delta` injects dither
   (`np.random.default_rng(DITHER_SEED)`, `data.py:263`) and is *not*
   mean/std-standardized, whereas `forward_logreturns` standardizes
   `(r - mu) / sigma` with no dither (`preprocessing.py:29-46`). These
   produce numerically different arrays.

2. **Fake array on the log-return scale is wrong.** Phase 11 compares the
   real reference against `r["transformed"]` = `r_norm`, the *de-normalized
   log-return* array (`run_dualscale_fidelity.py:469-473`). Phase 12 passes
   `recon["transformed"]` which is `r_norm` from `reconstruct_od`
   (`run_sensitivity.py:557`, set at `:257`/`:273`). `r_norm` in this repo
   is `((samples+1)/2)*(r_max-r_min)+r_min` — the *r_min/r_max-rescaled*
   array, NOT the standardized `log_delta`-space the real reference lives in.
   Phase 11 deliberately compares `real_log_delta` (un-standardized log
   deltas) against this same `r_norm` and documents the reconciliation;
   Phase 12 instead compares standardized windowed log-returns against
   `r_norm`. Two different real refs against the same fake array cannot both
   be the "unchanged" suite.

3. **Windowing differs.** Phase 11 emits log-return metrics on the *flat*
   `r["transformed"].reshape(-1)` vs flat `real_log_delta`
   (`_log_return_rows:469-486`). Phase 12 builds *windowed* `(N,10)`
   matrices for both real and fake and feeds them to `full_metric_suite`
   (`run_sensitivity.py:556-558`). Even on the OD scale Phase 11 flattens
   (`_od_scale_rows: synth_flat = od.reshape(-1)`,
   `run_dualscale_fidelity.py:355`), while Phase 12 passes the unflattened
   `(N,10)` matrices to `full_metric_suite`. `compute_moments`/`compute_emd`
   over a flattened vector vs a 2-D matrix are not guaranteed to be
   identical (EMD/quantile behavior on a 2-D input differs from the pooled
   1-D vector the frozen pipeline used).

The net effect: the `scale="log_return"` rows (and arguably the OD-scale
rows, via the flatten difference) produced by Phase 12 will NOT match the
frozen `fidelity_dualscale.json` numbers for the `analytic` baseline
condition. Since `analytic` is supposed to reproduce the frozen reference
column exactly (the file header calls it "the {analytic} reference"), this
silently breaks the stated core correctness property and the SENS-03
reconciliation story.

**Fix:** Mirror the frozen Phase 11 recipe verbatim. Use
`d_real["log_delta"].cpu().numpy()` as the real log-return reference (not
`forward_logreturns`+`rolling_window`), compare it flat against
`recon["transformed"].reshape(-1)`, and on the OD scale pass
`fake_od.reshape(-1)` / `real_od.reshape(-1)` (or, better, reuse the exact
`_od_scale_rows`/`_log_return_rows` helpers from
`run_dualscale_fidelity.py` rather than re-deriving via
`full_metric_suite`). At minimum add an assertion that the `analytic`
condition's emitted rows equal the frozen `fidelity_dualscale.json`
quantum rows for that `(pipeline, seed)` within float tolerance — that
test would have caught this.

```python
# log-return scale — match run_dualscale_fidelity.build_real_references
d_real = load_and_preprocess(str(csv_path))
real_lr = d_real["log_delta"].cpu().numpy()          # NOT forward_logreturns
fake_lr = np.asarray(recon["transformed"], np.float64).reshape(-1)
lr_metrics = full_metric_suite(real_lr, fake_lr)     # flat, like Phase 11
```

### CR-02: `--seed 0` (and any falsy seed) silently rejected / mis-routed

**File:** `revision/run_sensitivity.py:807` and `revision/run_sensitivity.py:796`
**Issue:**
Per-cell validation is `if not (args.pipeline and args.seed is not None and
args.condition)`. `args.seed` is correctly guarded with `is not None`, but
the `--emit-rollup` mutual-exclusion check at line 796 is
`if args.pipeline or args.seed is not None or args.condition:` — `args.seed`
is guarded but `args.pipeline`/`args.condition` are truthy-checked, which is
fine for strings. The real defect is subtler and in the *opposite*
direction: there is no `required=True` and no default on `--seed`, so
invoking `--emit-rollup` together with `--seed 0` is correctly rejected, but
the per-cell guard `args.seed is not None` means `--seed 0` is accepted in
per-cell mode — good. However `_frozen_analytic_paths`/`load_trained_generator`
build paths as `str(seed)`; seed `0` is a valid directory only if a `0`
training run exists. The actual bug: the argparse `choices`/type give no
guard that `seed ∈ {42,43,44}` (D-12-02). A typo like `--seed 442` passes
argparse, then fails deep inside `torch.load` with a raw `FileNotFoundError`
and a partially-created `run_dir` (because `_write_bundle` rmtree+mkdir runs
only later, but `mkdir -p "${run_dir}"` in the sweep created the dir first),
which the sweep then records as `failed` with an opaque rc and retries
forever.

**Fix:** Constrain the seed at the CLI boundary and fail fast with a clear
message before any artifact path is built:

```python
ap.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46])
# ... and in per-cell mode, assert the checkpoint exists with a clear error:
ck_path = _frozen_analytic_paths(args.pipeline, args.seed)[0].parent / "checkpoint.pt"
if not ck_path.exists():
    ap.error(f"no frozen run for pipeline={args.pipeline} seed={args.seed}: {ck_path}")
```

### CR-03: parallel status writes are serialized by flock but the *resume completeness* check is not — `all_complete` can be wrong, breaking the resume invariant

**File:** `revision/run_sensitivity_sweep.sh:267-270`, `:333-339`, `:421`
**Issue:**
`update_status` is flock-protected, so individual JSON merges don't corrupt
each other. But `doc["all_complete"] = (completed == doc["total_count"])` is
computed *inside each per-run merge* from the runs present *at that moment*.
Under `xargs -P 2`, the last two cells finish nearly simultaneously; each
acquires the lock in turn and recomputes `completed`. That part is safe. The
real hazard is the **final completeness gate** at lines 475-483: it re-reads
`${STATUS_FILE}` and trusts `all_complete`. But `run_one` marks a cell
`complete` only `if [[ $rc -eq 0 ]] && is_complete ...` (line 333).
`is_complete` tests `-s` (non-empty) on `config.yaml`/`samples.npy`/
`metrics.json`. `_write_bundle` writes those three files **non-atomically**
(`run_sensitivity.py:587-592`): `config.yaml`, then `samples.npy`, then
`metrics.json`, with no fsync/rename. If a parallel worker is killed
(thermal throttle / Ctrl-C / OOM) between `np.save(samples.npy)` and the
`metrics.json` write, `is_complete` is false (good), but if it dies *after*
all three `write_text`/`np.save` calls return yet before the OS flushes,
a subsequent `-s` check on a crashed run can see truncated-but-nonempty
files and mark the triple `complete`, permanently skipping regeneration on
resume. The sweep advertises "resumable / idempotent" as a guarantee; this
makes silent partial bundles possible, defeating numerical faithfulness.

**Fix:** Make `_write_bundle` atomic (write into a temp dir, fsync, then
`os.replace` the directory into place), or have `is_complete` validate
content not just non-emptiness (e.g. `metrics.json` parses as JSON and
contains the expected `rows`/`condition` keys, `samples.npy` loads and has
the expected first-dim length). Recommended minimal fix — validate
`metrics.json` is parseable and `samples.npy` is loadable in `is_complete`:

```bash
is_complete() {
  local d="${OUT_ROOT}/runs/$1/$2/$3"
  [[ -s "${d}/config.yaml" && -s "${d}/samples.npy" && -s "${d}/metrics.json" ]] || return 1
  "$PYTHON" - "$d" <<'PY' || return 1
import json, sys, numpy as np
d = sys.argv[1]
json.load(open(f"{d}/metrics.json"))      # must parse
np.load(f"{d}/samples.npy")               # must load
PY
}
```

## Warnings

### WR-01: `assert` used for the PennyLane version gate and the data_hash gate — disabled under `python -O`

**File:** `revision/run_sensitivity.py:93-97`; `revision/run_multiseed_rollup.py:85-87`
**Issue:** The version gate (`assert qml.__version__ == "0.44.0"`) and the
D-10-15 cross-artifact `data_hash` gate (`assert len(set(...)) == 1`) are the
documented hard fail-loud guards protecting the entire numerical-faithfulness
contract. Both are `assert` statements, which Python strips entirely under
`python -O`/`PYTHONOPTIMIZE`. The sweep selects a bare `python3`
(`run_sensitivity_sweep.sh:118-119`) whose optimization flags are not
controlled, so the very guards the design leans on can be silently no-ops.
**Fix:** Replace with explicit raises:
```python
if qml.__version__ != "0.44.0":
    raise RuntimeError(f"Phase 12 requires PennyLane 0.44.0, got {qml.__version__}")
...
if len(set(hashes.values())) != 1:
    raise AssertionError(f"data_hash mismatch across headline artifacts: {hashes}")
```

### WR-02: `run_multiseed_rollup` groups by 6-key but ignores `condition`/`shots`/`noise_*` — collapses distinct sensitivity cells

**File:** `revision/run_multiseed_rollup.py:96-105`
**Issue:** The groupby key is
`(source, model_kind, pipeline, metric_name, scale, injection_ratio)`. It
consumes only the five Phase 10/11 headline JSONs (`HEADLINE`, line 63-69),
which is consistent with the docstring. But if `shot_noise_sensitivity.json`
/ `noise_model_sensitivity.json` are ever added to `HEADLINE` (a plausible
future edit, since they share the extended six-key contract), rows that
differ only by `condition`/`shots`/`noise_level` would be silently averaged
together into one cell — a faithfulness-destroying aggregation with no error.
The key construction is not defensive against the SENS dimensions it is
explicitly designed to coexist with.
**Fix:** Include `r.get("condition")`, `r.get("shots")`,
`r.get("noise_model")`, `r.get("noise_level")` in the groupby key (they are
`None` for the five current files, so behavior is unchanged today but
correct if the SENS files are ever folded in), or assert no consumed row
carries a non-None `condition`.

### WR-03: `n_synth` mismatch between regenerated samples and frozen analytic count is unchecked

**File:** `revision/run_sensitivity.py:825-827`
**Issue:** Regeneration uses
`n=int(np.load(frozen_samples).shape[0])` to match the frozen analytic
sample count, then `generate_samples_on_qnode` returns
`np.concatenate(out_parts)[:n]`. If the noisy/finite-shot QNode ever returns
a different per-batch second dimension (e.g. `default.mixed` batching
semantics differ from `default.qubit` for some channel), the `stacked.dim()
== 2` transpose branch (line 205) silently produces the wrong shape and the
`[:n]` truncation hides it. There is no assertion that
`samples.shape == frozen_samples.shape` before metrics are computed, so a
shape/contract regression would produce plausible-but-wrong fidelity numbers
rather than failing loudly.
**Fix:** After `generate_samples_on_qnode`, assert
`samples.shape == np.load(frozen_samples).shape` for the non-analytic
branch (the contract is explicitly "same N as frozen analytic column").

### WR-04: `--emit-rollup` mutual-exclusion check misfires on `--seed 0`

**File:** `revision/run_sensitivity.py:796`
**Issue:** `if args.pipeline or args.seed is not None or args.condition:`
is correct for rejecting `--seed 0 --emit-rollup` (0 is not None), but the
asymmetry with the per-cell guard is fragile and undocumented. More
importantly, `args.pipeline`/`args.condition` use truthiness; an empty
string from a future default would not trigger the guard. Low blast radius
today but a latent correctness trap given seeds are constrained to
{42,43,44}. **Fix:** Make the guard explicit and symmetric:
`if any(v is not None for v in (args.pipeline, args.seed, args.condition)):`.

### WR-05: `usage()` prints a hardcoded line range that drifts from the header

**File:** `revision/run_sensitivity_sweep.sh:136-138`
**Issue:** `usage() { sed -n '2,84p' "$0"; }` hardcodes line numbers
2..84. Any header edit (this file's header is heavily commented and likely
to change) silently truncates or over-prints the help text. Not a
correctness bug for the sweep itself but a maintenance hazard in a script
whose help is the primary operator documentation. **Fix:** Delimit the
help block with sentinel markers and `sed -n '/^# HELP-START/,/^# HELP-END/p'`,
or move usage to a heredoc.

## Info

### IN-01: deliberate circuit duplication is a real long-term drift risk

**File:** `revision/run_sensitivity.py:363-416`
**Issue:** `noisy_generator_circuit` is a verbatim copy of
`quantum.py:122-171` (acknowledged, D-10-13). It was confirmed byte-equivalent
against the current `quantum.py` during this review. The risk is purely
future: if `core/models/quantum.py:generator_circuit` ever changes, this
copy will silently diverge and the noise-study numbers will no longer share
the trained circuit topology. **Fix:** Add a startup check that hashes the
relevant `generator_circuit` source region (or an explicit comment with the
exact git blob hash reviewed) so drift is detectable.

### IN-02: `import yaml` and `import datetime` are function-local

**File:** `revision/run_sensitivity.py:583`, `:635`, `:552`
**Issue:** `import yaml` (in `_write_bundle`), `import datetime as _dt`
(in `aggregate`), and `from revision.core.preprocessing import
forward_logreturns` (in `compute_dualscale_metrics`) are deferred imports.
The `forward_logreturns` deferred import becomes moot once CR-01 is fixed
(it should not be used at all). The others are harmless but inconsistent
with the module-top import style elsewhere. **Fix:** Hoist to module top
unless there is a measured import-cost reason.

### IN-03: `default=float` JSON serializer can mask non-finite metric values

**File:** `revision/run_sensitivity.py:591`, `:729-730`
**Issue:** `json.dumps(..., default=float)` coerces numpy scalars, which is
intended. But `full_metric_suite` can return `nan`/`inf` (e.g. EMD/JSD on a
degenerate distribution under heavy depolarizing noise where all samples
collapse). `json.dumps` emits bare `NaN`/`Infinity` tokens which are invalid
JSON and will make `run_multiseed_rollup.py:77` `json.loads` fail later, or
worse, `_is_num` treats `float('nan')` as numeric and poisons `fmean`.
**Fix:** Sanitize non-finite metric values to `None` (with a recorded
reason) before serialization, or pass `allow_nan=False` and handle the raise
explicitly.

### IN-04: `seeds`/`pipelines` provenance in `aggregate` derived from data, not asserted against D-12-02

**File:** `revision/run_sensitivity.py:681-684`
**Issue:** The emitted `seeds`/`pipelines` headers are computed from whatever
rows happen to be present. A partial sweep (some cells failed) silently
produces a headline JSON claiming `seeds: [42]` with no indication it is
incomplete. **Fix:** Assert the discovered `seeds == {42,43,44}` and
`pipelines == {A,B}` (D-12-02) before writing, or stamp an explicit
`complete: false` marker when the grid is short.

---

_Reviewed: 2026-05-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
