---
phase: 11-utility-evaluation
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - revision/run_utility.py
  - revision/run_timegan_scores.py
  - revision/run_dualscale_fidelity.py
  - revision/tests/test_timegan_scores.py
  - revision/tests/test_utility.py
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-05-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the four Phase-11 evaluation drivers (TSTR/augmentation, TimeGAN
predictive/discriminative, dual-scale fidelity) and the two pytest suites.
The "verbatim reuse" claims for `reconstruct_od` and `train_eval_tstr` were
checked byte-for-byte against `revision/_build_baseline_notebook.py` and the
OD-scale fidelity emission against the baseline notebook — those reuses are
faithful and bit-stable, and `compute_emd` ravels internally so the test-9
2-D call site is safe.

The headline defect is a hardcoded absolute machine path in
`run_dualscale_fidelity.py` that silently routes frozen-artifact resolution to
one developer's home directory — a reproducibility hole that directly
contradicts the phase's "no silent omission / portable consumer" contract and
will make the driver non-portable and capable of mixing artifacts from a
stale checkout without any loud failure. Several correctness-adjacent warnings
follow (a stale shape comment that conflicts with the test's `n_train_real==65`
invariant, an unguarded `ss_tot>0` R2 fallback that can mask a degenerate eval
set, a non-disjoint augmentation subsample RNG keyed on a lossy `int(ratio*1000)`,
and a discriminative-split RNG-coupling subtlety).

## Critical Issues

### CR-01: Hardcoded absolute home-directory path defeats reproducibility and can silently mix stale artifacts

**File:** `revision/run_dualscale_fidelity.py:112` (used at `:145-147` via `_resolve_run_dir`)
**Issue:**
```python
_CANONICAL_REPO_FALLBACK = Path("/Users/shawngibford/dev/phd/qGAN")
```
This is a machine- and user-specific absolute path baked into a scientific
reproducibility driver. Three concrete problems:

1. **Non-portable / non-reproducible.** On any other machine, CI runner, or
   reviewer checkout the fallback silently does not exist, so behavior diverges
   from the documented "canonical checkout" path with no warning. The whole
   point of the Phase-11 data-hash invariant is cross-machine reproducibility;
   a hardcoded `/Users/shawngibford/...` fallback structurally breaks that
   guarantee. The docstring even advertises this as the intended Rule-3
   behavior, which makes the artifact provenance unverifiable off-box.
2. **Silent stale-artifact mixing.** `_resolve_run_dir` returns the in-tree
   path "if it exists", else the canonical fallback "if it exists". If the
   current worktree has a *partial* set of frozen bundles (some present, some
   absent), the loop in `emit_rows` will transparently read some runs from the
   worktree and others from a *different, possibly older* checkout at the
   hardcoded path. There is no assertion that both sources share the same
   `data_hash`/git revision, so `fidelity_dualscale.json` can be assembled
   from two inconsistent artifact sets while still passing the data-hash
   loop (which only re-reads `config.yaml` from whichever dir resolved). This
   is exactly the "silent omission / silent substitution" failure the phase
   contract forbids.
3. The fallback path also shadows the `_find_repo_root()` result, so a
   correctly-located repo whose `results/` is merely git-ignored-but-present
   still works, but a developer who *moves* the repo loses the fallback with
   no diagnostic beyond a generic `FileNotFoundError`.

**Fix:** Remove the hardcoded constant. Make the fallback an explicit,
opt-in CLI/env input and assert provenance consistency:
```python
import os
# No hardcoded home dir. Operator must declare the canonical checkout
# explicitly when running from a bundle-less worktree.
_CANONICAL_REPO_FALLBACK = (
    Path(os.environ["QGAN_CANONICAL_REPO"]).resolve()
    if os.environ.get("QGAN_CANONICAL_REPO")
    else None
)

def _resolve_run_dir(rel: Path) -> Path:
    repo = _find_repo_root()
    in_tree = repo / rel
    if in_tree.exists():
        return in_tree
    if _CANONICAL_REPO_FALLBACK is not None:
        fb = _CANONICAL_REPO_FALLBACK / rel
        if fb.exists():
            return fb
    raise FileNotFoundError(
        f"frozen run dir not found: {in_tree}. Set QGAN_CANONICAL_REPO to a "
        f"checkout containing the frozen bundles (D-11-08 forbids regeneration)."
    )
```
Additionally, assert that *all 60* runs resolve from the *same* root (record
the resolved root per run and fail loudly if a single emit mixes two roots),
so cross-checkout artifact mixing cannot pass silently.

## Warnings

### WR-01: Stale shape comment contradicts the `n_train_real == 65` test invariant

**File:** `revision/run_utility.py:187` (comment); cross-ref `revision/tests/test_utility.py:168`
**Issue:** `_real_windowed_od` annotates the return as `# (384,10)`. But
`rolling_window` returns `(len(OD)-10)//2 + 1` rows, and
`test_no_leakage_sentinel` hard-asserts `rob["n_train_real"] == 65` with
`HELD_OUT_N = 320`, which requires the windowed array to have **385** rows
(`385 - 320 = 65`), not 384 (`384 - 320 = 64`). The slicing logic
`real_windowed_OD[HELD_OUT_N:]` is correct regardless, so this is not a
runtime bug, but the comment is wrong and actively misleads any reader trying
to reconcile the partition arithmetic — a real hazard in a reproducibility
artifact where the window count is load-bearing. The same stale `(384,10)`
framing appears implicitly in the `D-10-21` lineage.
**Fix:** Correct the comment to the true count and make it derived, e.g.
`# shape ((len(OD)-WINDOW_LENGTH)//2 + 1, 10) == (385,10); [320:] -> 65 train`,
or drop the magic number entirely.

### WR-02: `r2_score_inline` silently returns 0.0 when the eval target is constant — masks a degenerate run

**File:** `revision/run_utility.py:216-219`
**Issue:** `return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0`. The `else 0.0`
branch is reached when the held-out target variance is zero. In this driver an
R2 of exactly `0.0` is indistinguishable from a *legitimately computed* R2 of
0.0, yet it is also the sentinel-relevant region: `test_no_leakage_sentinel`
asserts every real-only R2 is strictly `< 0`. A degenerate eval slice (e.g. a
mis-sliced or empty `eval_windows`) would silently yield `0.0`, which is
**not `< 0`**, so the leakage test would fail with a confusing message rather
than pointing at the true cause (degenerate variance). Verbatim-reuse is the
stated reason this is byte-identical, but the inline copy still warrants a
guard since sklearn's `r2_score` raises/produces NaN here instead of a
plausible-looking 0.0.
**Fix:** Distinguish the degenerate case explicitly, e.g. return
`float("nan")` (or raise) when `ss_tot == 0` and assert non-degeneracy on the
eval set before training:
```python
return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
```
If byte-identity with Phase-10 must be preserved for the number itself, add a
separate `assert eval_windows[:, 9:10].std() > 0` guard in `train_eval_tstr`.

### WR-03: Augmentation subsample RNG seed is lossy and collides across distinct ratios

**File:** `revision/run_utility.py:446` and `:462`
**Issue:** `sub_rng = np.random.default_rng(int(ratio * 1000) + 1)`. The seed
is derived from `int(ratio * 1000)`. This is fragile: any future ratio with
sub-0.001 resolution (or a float like `0.1 + 0.15`) truncates, and more
importantly the seed is **decoupled from `(model_kind, pipeline)`**, so the
*same* subsample index set is drawn for every model/pipeline at a given ratio.
That is defensible as a controlled comparison, but it is undocumented and the
recorded `subsample_rng_seed` field gives a false impression of independence.
It also silently changes meaning if `_INJECTION_GRID` is ever extended with
e.g. `+10%` (`0.10*1000=100`) vs an accidental `0.1001` — both behave
differently from intent without any error.
**Fix:** Use an explicit, collision-free, fully-qualified seed and record the
exact derivation, e.g.
`np.random.default_rng((hash((mk, p, label)) & 0xFFFFFFFF))` or a documented
integer table `{"+25%": 1001, "+50%": 1501, "+100%": 2001}` referenced by
label, not by `int(ratio*1000)`.

### WR-04: `synthetic_only` augmentation condition can equal `+100%` and is not partition-guarded against pool size

**File:** `revision/run_utility.py:447-475`
**Issue:** For the `+100%` row, `n_synth = round(1.0 * n_real_train)` (~65),
which is far below `synth_pool.shape[0]` (~3840×... pooled), so the
`if n_synth >= synth_pool.shape[0]: synth_sel = synth_pool` branch is
effectively dead for the documented data sizes but becomes live (silently)
for any smaller pool, at which point `+100%` and `synthetic_only` produce
identical training sets and identical metrics with no warning that the
injection grid collapsed. The metadata caveat documents the ~60× imbalance
but nothing asserts the grid points remain distinct.
**Fix:** Assert the injection grid is non-degenerate for the actual pool size,
e.g. `assert n_synth < synth_pool.shape[0], (label, n_synth, synth_pool.shape)`
(or explicitly record `grid_collapsed: true`), so a shrunken pool fails loudly
instead of emitting duplicated conditions.

### WR-05: Discriminative score couples two independent RNG streams off one seed; split is order-sensitive

**File:** `revision/run_timegan_scores.py:273-283, :305-309`
**Issue:** `discriminative_score` calls `np.random.seed(seed)` (legacy global
RNG) for the two `_split` permutations, then *separately* constructs
`np.random.default_rng(seed)` for minibatch index draws. The two `_split`
calls consume the global RNG state sequentially, so `r_tr/r_te` and `s_tr/s_te`
are correlated through global-RNG order; if the call order of `_split(rw)` /
`_split(sw)` is ever reordered, or if any upstream code touches the global
`np.random` state between `np.random.seed(seed)` and the splits, the split
changes silently and the "deterministic given seed" contract (asserted only
for `predictive_score` in `test_scores_deterministic`) breaks for the
discriminative path with no test coverage. Mixing legacy global `np.random`
with the `Generator` API in the same function is an error-prone pattern the
project's other drivers avoid.
**Fix:** Use a single explicit `Generator` for both splits and minibatches
(`g = np.random.default_rng(seed); perm = g.permutation(n)`), drop
`np.random.seed`, and add a `test_discriminative_score_deterministic`
analogous to the predictive determinism test.

### WR-06: No determinism / range test for `discriminative_score`; smoke assertions only run under `__main__`

**File:** `revision/tests/test_timegan_scores.py:44-49`; `revision/run_timegan_scores.py:454-472`
**Issue:** `test_scores_deterministic` covers only `predictive_score`. The
discriminative path — which has the riskier RNG coupling (WR-05) and a
`squeeze(-1)`/shape contract between `(B,)` logits and `(B,)` labels — has no
determinism test and no shape-mismatch guard. The only discriminative
exercise beyond `test_discriminative_score_in_range` lives in the
`if __name__ == "__main__":` block of the *driver* (`run_timegan_scores.py`),
which `pytest` never executes, so a regression in the GRU output shape or RNG
coupling would not be caught by the suite that is supposed to "lock the
invariants". A subtle shape bug (e.g. logits `(B,1)` vs labels `(B,)`) would
broadcast in `(preds == yte)` to `(B,B)` and produce a meaningless accuracy
without raising.
**Fix:** Add `test_discriminative_score_deterministic` (two identical calls
must be equal) and an explicit shape assertion in `discriminative_score`
before the accuracy computation:
`assert logits.shape == yte.shape, (logits.shape, yte.shape)`.

## Info

### IN-01: `_find_repo_root` is duplicated four ways with divergent start points

**File:** `revision/run_utility.py:41-51`, `revision/run_timegan_scores.py:56-66`, `revision/run_dualscale_fidelity.py:68-82` & `:115-121`, `revision/tests/test_utility.py:67-72`
**Issue:** Five near-identical repo-root walkers exist; `run_utility` starts
from `Path(__file__).parent`, `run_timegan_scores` from `.parent.parent`, and
`run_dualscale_fidelity` has *two* different implementations
(`_bootstrap_repo_on_path` and `_find_repo_root`). Drift between these is a
latent maintenance hazard and already nearly bit the hardcoded-path issue.
**Fix:** Promote one `_find_repo_root` helper (e.g. into a small
`revision/_pathutil.py`, not `revision/core/` per D-11-10) and import it.

### IN-02: `run_dualscale_fidelity` defines `_find_repo_root` but also `_bootstrap_repo_on_path`; the former re-walks from cwd

**File:** `revision/run_dualscale_fidelity.py:115-121`
**Issue:** `_find_repo_root()` walks from `Path.cwd()`, while
`_bootstrap_repo_on_path()` walks from `__file__`. `main()` uses the
cwd-based one for `--out` anchoring, so running from an unexpected cwd anchors
output differently than the other two drivers (which anchor on `__file__`).
Not a bug for the documented invocation but an inconsistency across the
"locked across all Phase 11 drivers" constants claim.
**Fix:** Use the `__file__`-anchored resolver consistently for both import
bootstrap and output anchoring.

### IN-03: `import yaml` performed inside `verify_data_hash` rather than at module top

**File:** `revision/run_dualscale_fidelity.py:226`
**Issue:** The other two drivers import `yaml` at module scope; here it is a
function-local import with no stated reason. Minor inconsistency / hides a
hard dependency from import-time failure.
**Fix:** Move `import yaml` to the module header to match the sibling drivers.

### IN-04: `real_only_r2` long-form extraction in the leakage test is effectively dead

**File:** `revision/tests/test_utility.py:176-189`
**Issue:** `real_only_r2` filters augmentation rows for
`metric_name == "r2"`, but `run_augmentation` only ever emits `r2_delta`/
`mae_delta`/`rmse_delta` long-form rows (never a bare `"r2"`), so the list is
always empty and the `if real_only_r2:` block never executes. The real check
is the `a["lift"]` loop above it, so the test still has teeth, but the dead
branch implies coverage that does not exist.
**Fix:** Remove the dead `real_only_r2` block or assert against the actual
emitted metric name (`r2_delta == 0` for `real_only`).

### IN-05: Unused parameter / dead `n_synth_subsample` path carried verbatim into three consumers

**File:** `revision/run_utility.py:143-152`, `revision/run_timegan_scores.py:146-155`, `revision/run_dualscale_fidelity.py:171-189`
**Issue:** `reconstruct_od(..., n_synth_subsample=None)` is copied verbatim
(justified by the verbatim-reuse contract) but no Phase-11 call site ever
passes `n_synth_subsample`, so the branch is dead in this phase. Acceptable
under the verbatim mandate; noting it so reviewers do not mistake it for
live behavior.
**Fix:** None required (verbatim reuse is contractually mandated); optionally
add a one-line comment that the kwarg is intentionally unused in Phase 11.

---

_Reviewed: 2026-05-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
