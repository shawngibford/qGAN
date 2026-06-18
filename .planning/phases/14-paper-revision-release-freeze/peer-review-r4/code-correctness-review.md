# Peer Review r4 — Agent 2: Code Correctness Review

Scope: bug-hunt the qGAN revision pipeline emitters + core eval modules, run the
`tests/` suite, hunt scale/off-by-one/RNG/dtype/shape/histogram bugs.
Last gate before the `v2.0-revision` Zenodo DOI freeze.

## Environment note

The assigned git worktree is checked out at an OLD commit (`c82169c`, phase 8) far
behind `main` (`8180a5e`); at that commit only `core/{data,eval,training}.py`
exist. The named review targets (`run_matched2000_dualscale.py`,
`run_distribution_emd.py`, `run_welch_aggregator.py`, `run_matched2000.py`,
`verify_freeze_ready.py`, `verify_number_provenance.py`, `tests/`,
`core/preprocessing.py`, `core/models/`) are tracked in the
main repo at a later commit but absent from the worktree HEAD. They were copied
read-only into `/tmp/peer-review-r4/code/` (a permitted working directory) along
with `data.csv` and the full `results/` artifact tree (311 JSONs, 45
matched2000 sample bundles, 18 transform_ablation bundles) to review and test in
isolation. The main repo was used READ-ONLY (git status / git diff queries and
in-place pytest with `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`,
`--basetemp=/tmp/...`). Nothing was written into the main repo.

## Test suite result — `tests/`

**23 passed / 0 genuine failures.**

- Run in the main-repo environment (the documented verification env, with
  `results/` artifacts + frozen `samples.npy` bundles present):
  **23 passed in 8.81s.**
- Run in the isolated `/tmp` copy: **22 passed, 1 failed.** The single failure
  is `test_utility.py::test_core_untouched`, which shells out to
  `git diff --stat -- core/__init__.py`; in `/tmp` there is no `.git`
  so git exits 129 and the test's `assert out.returncode == 0` trips. This is a
  pure worktree-environment artifact, NOT a code bug — verified independently:
  in the main repo `git diff --stat -- core/` and
  `... core/__init__.py` both return empty with exit 0, i.e. the core
  module IS byte-clean and the test PASSES there. The qgan_env interpreter does
  ship pytest 9.0.3 (the test docstrings claiming "qgan_env ships no pytest" are
  stale, but the dual-mode shim is harmless).

Suite covers: cross-phase data-hash invariant, TSTR/augmentation leakage
sentinels, reconstruct_od shape/dtype invariant, dual-scale coverage +
Pipeline-A explicit-null, classical/non-adversarial generator param-count and
autograd-liveness contracts, Phase-10 OD-EMD anchor reconciliation. No genuine
correctness regressions surfaced.

## Determinism / drift verification

All three named emitters were re-run and compared byte-for-byte against their
committed JSON artifacts:

- `run_matched2000_dualscale.py`: 2576 rows / 560 aggregates re-emitted;
  **max aggregate-mean abs-diff vs committed = 0.0** (exact). R3-CR-2 anchor
  values reproduce (AR log_return EMD 0.00294158, V1 0.01497, etc.).
- `run_welch_aggregator.py`: 40 pairs re-built; **max field abs-diff = 0.0**.
- `run_distribution_emd.py`: 90 rows re-emitted; **max row-value abs-diff = 0.0**.

No RNG leakage or non-determinism detected — the seeded `default_rng(seed*...)`
draws are reproducible and the pipeline is bit-stable.

## Findings

### LOW-1 — `verify_number_provenance.py` Pass-1 text-match is a known weak gate (disclosed)
File: `verify_number_provenance.py:208-227`, `:254-267`
The gate resolves a literal if it appears as a boundary-delimited substring in
ANY `results/*.json` blob (the `<text-match>` path), and Pass 2's
ε-neighborhood can resolve a literal to a numerically-close-but-semantically-
unrelated JSON value. This is the gate-v2 false-positive class disclosed in r2/r3
(sign-flip lookbehind + ε-neighborhood). It only weakens the gate (could PASS a
wrong number); it never produces a false BLOCK. Verified non-material here: all
three paper-blocks files PASS legitimately (`paper_blocks_framing.md` 23 literals,
`paper_blocks_refs_methods.md` 49, `reviewer_response.md` 83), and the emitter
JSONs they resolve against were independently confirmed byte-exact above. The
v2.1 differential-test for the negative-sign lookbehind passes. No regression;
documented item — flagged for completeness only.

### LOW-2 — `run_distribution_emd.py` self-test uses bare `assert` (python -O strips it)
File: `run_distribution_emd.py:344-350`
`emit()` self-tests via `assert self_emd == 0.0` / `assert self_fim == 1.0`.
Every other gate in the codebase deliberately uses `raise AssertionError` for
`python -O` safety (the explicit-raise idiom is cited throughout
`run_matched2000.py`, `verify_*.py`). Under `python -O` this self-test is
silently disabled. The self-test is non-load-bearing (it only catches a gross
histogram-EMD regression, and the metric is exercised by the real rows anyway),
so impact is minor. Cosmetic consistency fix; not freeze-blocking.

### LOW-3 — Welch OD strong-claim thresholds clear by very thin margins
File: `run_welch_aggregator.py:138-141`, summaries at runtime
`OD_floor_welch_p = 0.36521` vs threshold `> 0.36` (margin 0.0014) and
`OD_ceiling_abs_cohen_d = 0.64417` vs threshold `<= 0.65` (margin 0.0058). The
gate passes correctly and the thresholds are legitimate, but the margins are
narrow enough that any future re-seeding of the matched2000 sweep could flip the
verdict. This is a property of the data, not a code bug — noted so the freeze
team is aware the OD strong claim is statistically thin. No action required for
this freeze.

## Checks performed with NO finding (clean)

- Scale handling: the R3-CR-2 fix in `run_matched2000_dualscale._log_return_rows`
  (`:384` `trans_flat_raw = trans_flat * sigma + mu`) is correct — it
  un-standardizes the fake log-returns to raw units to match the raw
  `real_log_delta`. `inverse_logreturns` (`preprocessing.py:64`) confirms the
  stored `r_norm` is standardized, so `r_norm*sigma+mu` recovers raw units.
  The mirror fix in `run_distribution_emd._real_references` (`:199`
  `norm_log_delta = (raw_log_delta - mu)/sigma`) standardizes the REAL side
  instead — the opposite direction, but internally consistent: both sides land
  in the same (standardized) space because the fake `r_norm` from
  `_fake_log_return_flat` is already standardized. Both emitters are unit-coherent.
- Histogram bin edges: `compute_histogram_density_emd` (`run_distribution_emd.py:155-168`)
  correctly derives edges from REAL only, reuses them for FAKE
  (`np.histogram(fake, bins=edges)`), normalizes both to total-mass=1 over the
  SAME edge set (no per-distribution renormalization), and discloses dropped
  out-of-range fake mass via `fake_in_range_mass`. This is the correct R3-CR-1
  fix. (Observed `fake_in_range_mass` 0.95–1.0 across all matched2000 models —
  no severe truncation in this corpus, but the metric is implemented correctly.)
  `compute_jsd` in `core/eval.py:108-111` does `np.histogram(density=True)` then
  renormalizes `rh/rh.sum()` — a JSD-specific normalization, not the EMD path;
  acceptable and unchanged from v1.0.
- Off-by-one: `rolling_window` (`data.py:150-158`) `range(0, len-m+1, s)` is
  correct; the `od[:, :10]` 11→10 trim in `reconstruct_od` is intentional and
  documented; `forward_logreturns` length-(N-1) contract is honored.
- RNG / seeds: `default_rng(seed)`, `default_rng(seed*7919+1)`, `default_rng(s*31)`
  are deterministic and reproduced byte-exact above. `_train_vae` seeds torch +
  numpy + stdlib `random` (HI-7). `generate_wgan_samples` uses a fresh
  `default_rng(seed)`. The sweep uses `xargs -P` (fresh OS process per run), not
  in-process pools — no shared-RNG corruption path.
- dtype: emitters consistently `.astype(np.float64)` on load; quantum statevector
  forced to CPU/float64; `_finite_sanitize`/`_dumps_finite` in `run_matched2000.py`
  convert non-finite floats to None and hard-raise above a 5% threshold.
- Array shapes: `MODEL_ORDER` (figure_suite) == `MODEL_KINDS`
  (run_matched2000_dualscale) == 9 models, identical order. `reconstruct_od`
  exists in two modules with different signatures (`run_utility`:
  `(model,pipeline,seed)`; `run_figure_suite`: `(repo,model,seed)`) but each is
  imported only by code expecting its own signature — no call-site mismatch.
- `except` clauses: all are narrowly typed (`OSError`, `StopIteration`,
  `AttributeError`, `TypeError/ValueError`, `PackageNotFoundError`) — no bare
  `except:`. The one `except Exception: pass` (`verify_number_provenance.py:226`)
  is a fallthrough inside a multi-pass resolver and does not swallow a metric
  error.
- `verify_freeze_ready.py`: gate logic sound. Confirmed in the main repo: 0
  gitignored result JSONs, 799 tracked `revision/results` paths, `qgan_env/` not
  tracked, `data.csv` tracked, largest tracked `.pt` ~6.0 MB (well below the
  25 MB `LARGE_CKPT_BYTES` threshold). All three invariants would pass.
- Strict accept gate (`run_matched2000._strict_accept`): uses explicit
  `raise AssertionError` throughout (python -O safe), checks seed set, data_hash,
  epoch budget, early-stop-off, device manifest, schema, 5-file bundle, and the
  D-14-10 headline/reproduction conflation guard. Correct.

No CRITICAL, HIGH, or MEDIUM correctness bugs were found. No regression of the
closed R3-CR-1 / R3-CR-2 items — both fixes are present, correctly implemented,
and reproduce the documented r3 anchor numbers byte-exactly.

FREEZE VERDICT: GO
