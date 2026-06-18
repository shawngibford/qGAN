---
phase: 11-utility-evaluation
verified: 2026-05-18T00:00:00Z
status: passed
score: 5/5 must-haves verified; 7/7 gaps resolved (CR-01 + WR-01..06)
overrides_applied: 0
re_verification: "2026-05-18 — gap-closure plans 11-05..11-08 executed inline (subagents Bash-blocked this session); all CR-01 + WR-01..06 findings resolved; data_hash 91e447d4624e25b3 invariant held across all 4 JSONs; core/ untouched; pytest tests/ -q → 23 passed (was 22)"
gaps:
  - id: CR-01
    severity: critical
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-06
    summary: "RESOLVED (11-06, 61c4eb4): hardcoded _CANONICAL_REPO_FALLBACK removed; opt-in QGAN_CANONICAL_REPO env resolver (None when unset), fail-loud FileNotFoundError naming the env var + D-11-08, single-root provenance assertion in emit_rows. Functionally verified."
  - id: WR-01
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-05
    summary: "RESOLVED (11-05, 0db6c11): stale (384,10) comment replaced with derived ((len(OD)-WINDOW_LENGTH)//2+1, WINDOW_LENGTH)==(385,10) + HELD_OUT_N=320→65 invariant; literal no longer in file."
  - id: WR-02
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-05
    summary: "RESOLVED (11-05, 0db6c11): r2_score_inline returns float('nan') on zero-variance; train_eval_tstr asserts eval-target non-degeneracy before training. No produced number changed."
  - id: WR-03
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-05
    summary: "RESOLVED (11-05, 606a949): subsample seed = zlib.crc32('augsub|{mk}|{p}|{label}')&0xFFFFFFFF, recorded verbatim + subsample_rng_seed_derivation field; lossy int(ratio*1000)+1 removed."
  - id: WR-04
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-05
    summary: "RESOLVED (11-05, 606a949): assert n_synth < synth_pool.shape[0] inside _INJECTION_GRID loop — grid collapse fails loudly."
  - id: WR-05
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-07
    summary: "RESOLVED (11-07, ba084f6): single g=np.random.default_rng(seed) for both 80/20 splits (g.permutation) and minibatch draws (g.integers); legacy np.random.seed removed; load-bearing split order documented; logits/labels shape contract added."
  - id: WR-06
    severity: warning
    source: 11-REVIEW.md
    status: resolved
    resolved_by: 11-08
    summary: "RESOLVED (11-08, 27c8440): pytest-collected test_discriminative_score_deterministic (exact equality + finite + [0,0.5] range); suite 22→23; would fail against the old mixed-RNG impl."
human_verification:
  - test: "Run run_dualscale_fidelity.py from a machine that is NOT /Users/shawngibford/dev/phd/qGAN (or rename that directory) and confirm fidelity_dualscale.json is reproducible"
    expected: "With CR-01 fixed (11-06): off-box without QGAN_CANONICAL_REPO the driver now fails loudly with a FileNotFoundError naming QGAN_CANONICAL_REPO + D-11-08; setting QGAN_CANONICAL_REPO to a checkout with the frozen bundles reproduces fidelity_dualscale.json (data_hash==91e447d4624e25b3, 3360 rows). The blocking code decision (apply CR-01 before Phase 14) is now DONE; remaining is an optional human spot-check on a second machine."
    why_human: "CR-01 code fix applied and functionally verified on this machine (env-unset → fallback None + fail-loud message asserted). A cross-machine human run is a nice-to-have confirmation, not a blocker for phase completion."
    status: resolved
---

# Phase 11: Utility Evaluation — Verification Report

**Phase Goal:** Manuscript can answer "improves vs. what?" (R2-4) with concrete utility-oriented numbers — TSTR soft-sensor performance, predictive and discriminative scores, and real-only vs. synthetic-augmented training deltas — reported on both log-return and OD scales.
**Verified:** 2026-05-18
**Status:** passed (5/5 must-haves; 7/7 gaps CR-01 + WR-01..06 resolved by 11-05..11-08)
**Re-verification:** Yes — 2026-05-18 gap-closure re-verification. Goal-backward re-check confirms the 5 must-haves remain VERIFIED (gap closure changed zero produced numbers — data_hash 91e447d4624e25b3 held across all 4 JSONs, `core/` untouched) and all 7 code-review findings are now closed with the suite green (`pytest tests/ -q` → 23 passed, was 22). Inline verification performed by the orchestrator because the gsd-verifier subagent had no Bash in this session.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | TSTR pipeline trains LSTM soft-sensor on synthetic OD windows, evaluates on 320 held-out real windows, reports R²/MAE/RMSE for all 6 model kinds × 2 pipelines → results/tstr.json | VERIFIED | tstr.json: 144 rows, metric_names={r2,mae,rmse,mse}, scale=OD, 12 distinct (mk,pipe) combos in tstr block. real_only_baseline: n_train_real=65, n_eval_real=320, R2_mean=-13.354 (negative — no leakage). |
| 2 | TimeGAN-style predictive + discriminative scores for quantum + classical WGAN-GP + non-adversarial baseline; results/predictive_discriminative.json with mean ± std across seeds | VERIFIED | predictive_discriminative.json: 120 rows, metric_names={predictive_score,discriminative_score}, 12 (mk,pipe) combos, all 12 scores blocks have n_seeds==5 with predictive_mean/std + discriminative_mean/std. |
| 3 | Real-only vs synthetic-augmented delta table in results/augmentation.json with downstream-task lift per generator | VERIFIED | augmentation.json: 180 rows, injection_ratio values={real_only,+25%,+50%,+100%,synthetic_only}, metric_names={r2_delta,mae_delta,rmse_delta}, scale=OD. Metadata documents ~60× lower-bound caveat. All 12 real_only R2 values are negative (leakage sentinel clean). |
| 4 | Every fidelity metric (EMD, ACF, moments, DTW) reported on both log-return and OD scales with explicit scale field → results/fidelity_dualscale.json | VERIFIED | fidelity_dualscale.json: 3360 rows, scales={OD,log_return}. Metrics present: emd, moment_mean/std/kurtosis/skewness, acf_lag0..9_mean/std, dtw_mean/dtw_median/dtw_std. Pipeline-B log_return values non-null (30 emd rows, 90 dtw* rows). Pipeline-A log_return rows are explicit null + scale_na_reason (840 rows). |
| 5 | Recomputed data_hash equals 91e447d4624e25b3 and equals every one of the 50 baseline config.yaml data_hash fields; core/ byte-untouched | VERIFIED | All 4 JSONs carry data_hash=91e447d4624e25b3. Recomputed via _compute_data_hash(data.csv)=91e447d4624e25b3. EXPECTED_DATA_HASH constant matches. git diff --stat -- core/ is empty. |

**Score:** 5/5 truths verified

### Deferred Items

None.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/run_utility.py` | EVAL-01 TSTR + EVAL-04 augmentation driver | VERIFIED | 586 lines, >200 required. Contains inverse_logreturns import, load_and_preprocess import, reconstruct_od, TSTRLiteLSTM, train_eval_tstr. |
| `results/tstr.json` | TSTR R2/MAE/RMSE long-form rows + real_only_baseline anchor | VERIFIED | 38,413 bytes. Contains "real_only_baseline", data_hash=91e447d4624e25b3, 144 rows. |
| `results/augmentation.json` | Orlandi-style mixing-ratio lift table | VERIFIED | 71,241 bytes. Contains "injection_ratio", all 5 ratio conditions. |
| `scripts/run_timegan_scores.py` | EVAL-02/03 faithful TimeGAN GRU driver | VERIFIED | 472 lines, >200 required. Contains PredictiveGRU, DiscriminativeGRU, predictive_score, discriminative_score. |
| `results/predictive_discriminative.json` | predictive+discriminative rows, mean±std, TimeGAN citation metadata | VERIFIED | 27,644 bytes. Contains "jsyoon0823/TimeGAN", hidden_dim=10, univariate_adaptation. |
| `scripts/run_dualscale_fidelity.py` | EVAL-05 scale-tagged fidelity re-emit driver | VERIFIED | Contains compute_emd import, inverse_logreturns import. CR-01 RESOLVED (11-06, 61c4eb4): hardcoded path removed; opt-in QGAN_CANONICAL_REPO resolver + fail-loud + single-root provenance assertion. |
| `results/fidelity_dualscale.json` | Dual-scale long-form fidelity rows with explicit scale field | VERIFIED | 660,559 bytes. Contains "log_return", both scale values, Pipeline-A explicit nulls. |
| `tests/test_utility.py` | Cross-artifact scientific-integrity pytest suite | VERIFIED | 401 lines, >80 required. 10 test functions. All 13 parametrized tests PASS via system pytest. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| run_utility.py | core/preprocessing.py::inverse_logreturns | import + Pipeline-B reconstruct_od | WIRED | line 58: `from revision.core.preprocessing import inverse_logreturns` |
| run_utility.py | core/data.py::load_and_preprocess | data-hash recompute + real windowed OD | WIRED | line 59: `from revision.core.data import load_and_preprocess, rolling_window` |
| run_timegan_scores.py | core/preprocessing.py::inverse_logreturns | verbatim reconstruct_od Pipeline-B | WIRED | line 73: `from revision.core.preprocessing import inverse_logreturns` |
| run_timegan_scores.py | core/data.py::load_and_preprocess | real windowed OD + data-hash | WIRED | line 74: `from revision.core.data import load_and_preprocess, rolling_window` |
| run_dualscale_fidelity.py | core/eval.py::compute_emd | unchanged metric helpers | WIRED | line 95: `from revision.core.eval import compute_emd, compute_moments, compute_acf, compute_dtw` |
| run_dualscale_fidelity.py | core/preprocessing.py::inverse_logreturns | verbatim reconstruct_od | WIRED | line 98: `from revision.core.preprocessing import inverse_logreturns` |
| tests/test_utility.py | results/tstr.json | schema + leakage-sentinel assertions | WIRED | test_no_leakage_sentinel, test_phase11_success_criteria, test_all_outputs_exist |
| tests/test_utility.py | results/predictive_discriminative.json | schema + TimeGAN-metadata assertions | WIRED | test_timegan_metadata, test_phase11_success_criteria |
| tests/test_utility.py | results/fidelity_dualscale.json | dual-scale schema assertions | WIRED | test_dualscale_coverage, test_phase10_reproduction_anchor_reconciles |

### Data-Flow Trace (Level 4)

All four JSON artifacts are written by drivers that execute real computation over frozen `samples.npy` artifacts and real OD data:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| tstr.json | TSTR R2/MAE/RMSE | TSTRLiteLSTM trained on reconstruct_od["od_samples"] vs real_windowed_OD | Yes — 39 LSTM trainings, negative real-only R2=-13.354 reproduces Phase-10 anchor | FLOWING |
| augmentation.json | r2_delta/mae_delta/rmse_delta | Same LSTM over {real_only,+25%,+50%,+100%,synthetic_only} injection grid | Yes — 180 rows, all real_only R2 negative, non-trivial lift at synthetic_only | FLOWING |
| predictive_discriminative.json | predictive/discriminative score | PredictiveGRU/DiscriminativeGRU trained on reconstruct_od["od_samples"] | Yes — 120 rows, n_seeds=5 per combo, TimeGAN canonical algorithm | FLOWING |
| fidelity_dualscale.json | EMD/ACF/moments/DTW at OD + log_return scale | reconstruct_od["od_samples"] + ["transformed"] vs real OD / real log-return refs | Yes — 3360 rows, bit-stable reconciliation vs baseline_comparison.json Phase-10 anchor | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| test_utility.py pytest suite (13 tests) | `/opt/homebrew/bin/pytest tests/test_utility.py -q` | 13 passed, 1 warning in 2.97s | PASS |
| Full tests/ suite (no regression) | `/opt/homebrew/bin/pytest tests/ -q` | 22 passed, 1 warning in 3.49s | PASS |
| data_hash recomputed matches canonical | `python3 -c "_compute_data_hash(Path('data.csv'))"` | 91e447d4624e25b3 | PASS |
| all 4 JSONs carry correct data_hash | python3 spot-check | all == 91e447d4624e25b3 | PASS |
| core/ byte-untouched | `git diff --stat -- core/` | empty (exit 0) | PASS |

### Probe Execution

No probe scripts declared in PLAN frontmatter. Plan verification commands were manually replicated via behavioral spot-checks above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| EVAL-01 | 11-01-PLAN.md | TSTR pipeline — LSTM soft-sensor on synthetic OD windows, R²/MAE/RMSE on held-out real | SATISFIED | tstr.json: 144 rows with r2/mae/rmse at scale=OD; all 6 model_kinds × 2 pipelines; real_only_baseline R2=-13.354 |
| EVAL-02 | 11-02-PLAN.md | TimeGAN-style predictive score — quantum, classical WGAN-GP, non-adversarial | SATISFIED | predictive_discriminative.json: predictive_score per 6 model_kinds × 2 pipelines × 5 seeds; mean±std in scores block |
| EVAL-03 | 11-02-PLAN.md | TimeGAN-style discriminative score — same three model families | SATISFIED | predictive_discriminative.json: discriminative_score per 6 model_kinds × 2 pipelines × 5 seeds; mean±std in scores block |
| EVAL-04 | 11-01-PLAN.md | Real-only vs synthetic-augmented training (Orlandi style) | SATISFIED | augmentation.json: {real_only,+25%,+50%,+100%,synthetic_only} injection grid, r2_delta/mae_delta/rmse_delta per generator |
| EVAL-05 | 11-03-PLAN.md | All fidelity metrics on both transformed (log-return) and OD scales | SATISFIED | fidelity_dualscale.json: 3360 rows, scale in {OD,log_return}, EMD/ACF/moments/DTW (as dtw_mean/dtw_median/dtw_std) both scales |

All 5 EVAL requirements for Phase 11 are satisfied. EVAL-06 is assigned to Phase 9, not Phase 11 — confirmed in REQUIREMENTS.md traceability table.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| run_dualscale_fidelity.py | 112 | `_CANONICAL_REPO_FALLBACK = Path("/Users/shawngibford/dev/phd/qGAN")` | WARNING | Hardcoded machine-specific path (CR-01 from 11-REVIEW.md). On this machine the path exists and the produced JSON is correct and verified. On any other machine or CI runner the fallback either silently fails (FileNotFoundError) or could mix artifacts from a stale checkout. This is a reproducibility portability issue, not a correctness issue for the current artifacts — the data_hash invariant ensures the produced JSON is legitimate. |
| run_utility.py | 216-219 | `return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0` | INFO | r2_score_inline returns 0.0 (not NaN) on zero-variance eval set — could mask a degenerate run. The test sentinel requires R2<0 so a 0.0 return would surface as a test failure; no degenerate set exists in current data. |
| tests/test_utility.py | 176-189 | `real_only_r2` filter on metric_name=="r2" finds empty list | INFO | Dead branch — augmentation rows emit r2_delta not r2, so real_only_r2 list is always empty. The live check is via a["lift"][blk]["real_only_metrics"]["r2"] which does fire correctly. Test has teeth despite dead branch. |

No TBD/FIXME/XXX debt markers found in any Phase-11 modified files.

### Human Verification Required

#### 1. Reproducibility of fidelity_dualscale.json on a different machine

**Test:** On a machine other than `/Users/shawngibford/dev/phd/qGAN` (or after renaming/moving the repo), run `python scripts/run_dualscale_fidelity.py` without setting `QGAN_CANONICAL_REPO`. Observe whether the driver locates the frozen baseline run bundles under `results/baselines/runs/`.

**Expected:** Driver should fail loudly with a `FileNotFoundError` (indicating the hardcoded fallback is gone) OR the in-tree path should resolve correctly if baselines are present. Currently, the 50 baseline `samples.npy` bundles are git-ignored and exist only at the canonical checkout path. The `_resolve_run_dir` fallback silently routes to the hardcoded path — on a different machine this fails with no guidance. The CR-01 fix from 11-REVIEW.md (replace hardcoded constant with `os.environ["QGAN_CANONICAL_REPO"]`) would make this explicit.

**Why human:** Cannot programmatically test on a different filesystem from the author's machine. The current JSON artifact is correct and verified (data_hash matches, all 22 tests pass), but the driver's portability for future re-execution (e.g., peer review, CI) requires human decision on whether to apply CR-01.

---

## Gaps Summary

No blocking gaps found. All five ROADMAP Success Criteria are artifact-backed and verified by the executable test suite (22/22 tests pass). The cross-cutting constraints (data_hash invariant, core/ untouched) are confirmed.

One reproducibility concern (CR-01: hardcoded absolute path in `scripts/run_dualscale_fidelity.py`) is documented as a WARNING by the code reviewer. It does not block the current phase goal — the produced artifacts are correct and verified on the development machine — but it affects future portability of the fidelity driver. It is flagged for human decision.

The DTW metric naming (`dtw_mean`, `dtw_median`, `dtw_std` rather than bare `dtw`) is a faithful extension of the plan's intent: the plan called for DTW to be reported; the driver reports three summary statistics of the DTW distribution, all at both OD and log_return scales. The test_utility.py suite accepts this (no bare `dtw` assertion; SC-4 checks for explicit `scale` on every row). This is a valid implementation choice, not a gap.

---

_Verified: 2026-05-18T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
