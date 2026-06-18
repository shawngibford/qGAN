# 14-15-PLAN-CHECK — gsd-plan-checker (goal-backward verification)

**Verdict: BLOCKED**

The plan's structural integrity is sound (D-14-22, D-14-13, D-14-16 preserved; schema T1→T2 aligned; no re-training; no `core/` touches; verification checklist clean). However, **two headline numerical claims that T4 asks reviewers to read in `reviewer_response.md` and `methods_full.md` are contradicted by the actual QQ companion JSONs**. Reviewers would see numbers that don't match the data they can independently re-derive — a credibility-killing finding in a peer-review-response context. This MUST be fixed before execution.

## Findings

### CRITICAL — Marginal-convergence numbers misstate the data (T4 + plan Context §2)

Plan claims (T4, `reviewer_response.md` body + `methods_full.md` cross-reference, plus Context §2): "8 of the 9 models recover the OD marginal to within **≤0.03 OD-units** max-absolute-quantile-difference; only WGAN-CNN deviates with max upper-tail of **~0.69**."

Spot-check from `figures/qq_<model>.json` (real_quantiles vs fake_quantiles, max-abs across the 0.5–99.5% grid):

| model            | max-abs-quantile-diff |
|------------------|-----------------------|
| iqp_sel_55_repro | **0.2415**            |
| V1               | 0.2452                |
| V2               | 0.2453                |
| V3               | 0.2419                |
| wgan_mlp         | 0.2519                |
| wgan_cnn         | **0.8141**            |
| wgan_lstm        | 0.2538                |
| vae              | 0.2805                |
| ar               | 0.2624                |

- The 8 "converged" models cluster around **~0.24–0.28**, an order of magnitude above the claimed `0.03` threshold.
- WGAN-CNN's max-abs is **0.81**, not `~0.69`.
- The qualitative finding survives (8/9 cluster tightly; WGAN-CNN is the outlier with a substantially larger upper-tail deviation), but the literal numbers don't.

The plan's verification checklist item 5 + the gate-resolution machinery for `0.03` and `0.69` (T5 specifically: "the marginal-convergence numbers `0.03` and `0.69` in `reviewer_response.md` + `methods_full.md` resolve to the qq_overlay companion JSON's per-model max-abs-quantile-diff statistics (emit those explicitly in `qq_overlay.json#convergence_stats` for gate resolution)") will **break gate-resolution** once T3 emits the actual stats: the literals in the docs (0.03, 0.69) will not match anything in `qq_overlay.json#convergence_stats`. The v2.1 gate will fail T5.

**Required fix (one of):**
1. **Recommended — replace the literals in T4 templates and Context §2 with the actual values.** Use `~0.25` (or "≤0.28") for the 8/9 cluster and `~0.81` for WGAN-CNN. Update: Context §2 line 9, T4 `reviewer_response.md` template (lines 110, 118), `methods_full.md` template (line 121), `paper_blocks_framing.md` line 123, T5 verification literals (lines 132, 958-region `automated` block which currently asserts `'0.03'` and `'0.69'` substrings).
2. Or — define a different convergence metric whose numerical headline genuinely is `≤0.03` (e.g. mean-abs across the 5–95% body, or median-abs, or a tail-trimmed range), and rewrite the prose to match. This is more invasive and risks losing the original framing intent; option 1 is preferred.

The qualitative claim ("8/9 hug the diagonal at the figure scale, WGAN-CNN visibly deviates") is preserved either way — it's just the numerical literals that must be corrected before any docs are written.

## Other dimensions — PASS

- **Requirement coverage:** all 4 intended outcomes (Context §11) trace to T1 (distribution-EMD aggregator) / T2 (3-column table) / T3 (qq_overlay) / T4 (reviewer comms). Verification checklist (T5, 10 items) trace-clean.
- **D-14-22 byte-freeze:** every `<verify>` gate asserts `[ -z "$(git diff --stat core/)" ]`; T1 additionally asserts `[ ! -f core/run_distribution_emd.py ]`. No task lists a `core/**` file in `<files>`. PASS.
- **D-14-13 / D-14-16 gate semantics:** new column is informational; strict-accept thresholds unchanged; no v2.1 gate edit. PASS.
- **No re-training:** T1 `<action>` loads `samples.npy` + reconstructs OD via `reconstruct_od`; no `train_*` invocations anywhere in the plan. PASS.
- **Schema T1 → T2 alignment:** T1 emits `aggregates[*, scale='OD', mean, std, n]`; T2 reads `distribution_emd.json#aggregates[*, scale='OD']`. Aligned. PASS.
- **Anchor sources confirmed by spot-check:**
  - `core/eval.py:25-36` — v1.0 raw-sample `compute_emd` present (the contrast the new emitter cites). ✓
  - `run_figure_suite.py:106-116` — `MODEL_ORDER` is exactly the 9 listed. ✓
  - `run_figure_suite.py:261-296` — `reconstruct_od` helper, Pipeline-B `seed*7919+1` draw load-bearing. ✓
  - `run_figure_suite.py:403-433` — `render_qq_plot` companion schema includes `real_quantiles`, `fake_quantiles`, `quantile_grid` — usable by `render_qq_overlay` and by T3's `convergence_stats` emit. ✓
  - `run_figure_suite.py:2378-2400` — caller loop iterates `MODEL_ORDER`; T3's new `render_qq_overlay` call slots in cleanly at ~line 2400+. ✓
  - `run_model_info.py:218-302` — `_reconciliation_rows()` reads `matched2000_dualscale.json#aggregates` filtered by `metric_name='emd'` AND `scale='OD'`. T2's extension to add a third column from `distribution_emd.json#aggregates[*, scale='OD']` plugs in via the same row builder. ✓
  - `reconciliation_note.md` C-3 disclosure paragraph cites the **50-bin** density Wasserstein formulation that T1 implements verbatim. ✓

## False-positives skipped (per instructions)

- Did not flag the histogram-density Wasserstein formulation choice (D-14-22 locks it as `wasserstein_distance(bin_centers, bin_centers, real_hist_density, fake_hist_density)` with `n_bins=50`).
- Did not flag the new emitter living outside `core/` (D-14-22 forbids core/ edits).
- Did not flag the absence of a strict-accept threshold for distribution-EMD (D-14-13 / D-14-16 keep gate semantics frozen).
- Did not flag the lack of re-runs (no re-training is locked in).
- Did not flag the lack of new ACF / conditional-moment figures (out of scope per plan §"Out of scope").

