---
phase: 14-paper-revision-release-freeze
plan: 10
subsystem: paper-figure-suite-extension
tags: [render-only, paper-figures, full-story, frozen-headline-distinct, honest-negative-r2, number-provenance-gated, json-traceable, audited-json-consumed]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze (plan 04)
    provides: "run_figure_suite.py render-only contract (matplotlib.use('Agg') before pyplot, _require/_load_json loud-fail, _save dual PNG+PDF + same-stem JSON, _find_repo_root) — the host module the 7 new figures extend"
  - phase: 14-paper-revision-release-freeze (plan 08)
    provides: "results/matched2000_dualscale.json (single source of truth for matched-2000ep dual-scale numbers; HEADLINE_KIND='frozen_checkpoint_headline' as a DISTINCT row-set with n_seeds=1, source='frozen_checkpoint_epoch_1969') + the _agg_lookup / DUALSCALE_MODEL_ORDER / HEADLINE_COLOR / HEADLINE_LABEL constants the new figures reuse verbatim"
  - phase: 14-paper-revision-release-freeze (plan 09)
    provides: "circuit_atlas + provenance pattern for follow-on render-only extensions (companion JSON shape, conflation_guard string format, source_artifact path discipline)"
  - phase: 14-paper-revision-release-freeze (plan 03)
    provides: "results/model_info.json (per-model parameter_count + family + source) — the Task 4 input"
provides:
  - "figures/training_convergence_all_models.{png,pdf,json} — 7-adversarial-model EMD-vs-epoch trajectories (mean ± std over 5 seeds) + frozen-checkpoint headline as DISTINCT diamond at epoch 1969 + horizontal dashed reference line (D-14-10); VAE/AR explicitly skipped with caption note; render-only over 35 matched2000/runs/<model>/<seed>/metrics.json + headline_canonical.json"
  - "figures/tstr_crossmodel.{png,pdf,json} — TSTR cross-model R²/MAE/RMSE grouped bars × Pipeline A/B for the 6 model_kinds in tstr.json (quantum, wgan_mlp, wgan_cnn, wgan_lstm, vae, ar — NO fabricated V1/V2/V3); negative R² plotted HONESTLY (no clamp/abs/rescale); caption_note built dynamically from tstr.json's per_model_pipeline block (plan-check fix); render-only over results/tstr.json (previously unconsumed paper-facing JSON, now consumed)"
  - "figures/failure_modes_summary.{png,pdf,json} — 3-row × 9-column diagnostic grid (OD distribution overlay / ACF lag-1 vs real / log_return EMD with red-edged IQR outliers); columns ordered by ASCENDING OD EMD; frozen headline rendered as ROW-SPANNING dashed reference lines (D-14-10); render-only over matched2000_dualscale.json + dist_*.json + acf_*.json companions"
  - "figures/param_efficiency_pareto.{png,pdf,json} — log10(parameter_count) × EMD Pareto scatter, OD and log_return facets, family-coded markers (circle/square/triangle); frozen-headline diamond at log10(55) — DISTINCT marker shape AND color AND y from the iqp_sel_55_repro circle at the same x (D-14-10); render-only over model_info.json + matched2000_dualscale.json"
  - "figures/seed_variance_per_model.{png,pdf,json} — 3×3 facet grid (shared x, log-y, shared y) of per-seed EMD trajectories (light) + across-seed mean (bold) per model; 7 adversarial panels + 2 explicit no-trajectory caption panels (VAE/AR); per-panel spread label (tight/moderate/noisy) from final-step std/mean; render-only over 35 per-run metrics.json"
  - "figures/noise_robustness_quantum.{png,pdf,json} — 1×2 panels (depolarizing | amplitude_damping); EMD vs noise level with Pipeline A/B curves (mean ± std over 3 seeds); monotonicity recorded per (channel, pipeline); zero-anchor preserved per zero_anchor_note; render-only over noise_model_sensitivity.json (previously unconsumed paper-facing JSON, now consumed)"
  - "figures/shot_noise_robustness.{png,pdf,json} — single-panel EMD vs shot count (log-x) for Pipeline A/B with horizontal dotted analytic-statevector baseline per pipeline (shots=∞ asymptote); render-only over shot_noise_sensitivity.json (previously unconsumed paper-facing JSON, now consumed)"
  - "run_figure_suite.py — +1189 lines (7 new render_* functions wired into main() AFTER render_matched2000_dualscale_comparison_table and BEFORE render_existing_introspection; established 14-04 render-only contract reused unchanged: matplotlib.use('Agg'), _require/_load_json, _save dual PNG+PDF+JSON; ADVERSARIAL_MODELS_WITH_EMD_AVG and MODELS_NO_TRAJECTORY constants added for Tasks 1+5)"
affects: [paper-PAPER-01, paper-PAPER-09, manuscript-full-story-arc, 14-11-release-freeze]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Render-only figure-suite extension over previously-unconsumed audited JSONs: extend run_figure_suite.py with new render_* functions that consume audited JSONs (tstr.json, noise_model_sensitivity.json, shot_noise_sensitivity.json) verbatim — no recomputation, no resampling, no checkpoint reload; companion JSONs are auto-rglob'd by verify_number_provenance.py unmodified"
    - "Honest negative R² rendering: tstr.json's R²<0 values plotted at their raw negative position (no clamping, no abs(), no log-rescale, no split y-axis); the R²-panel y-axis extends to include the most-negative point with a 0-baseline reference; companion JSON's caption_note carries the explicit reviewer-facing 'R² < 0 ⇒ worse than predicting the mean' wording and the dynamic list of (model, pipeline) entries with r2_mean<0 (built from tstr.json's per_model_pipeline, NEVER hard-coded)"
    - "Frozen-checkpoint headline conflation guard reused across 3 at-risk figures (training_convergence_all_models, failure_modes_summary, param_efficiency_pareto): the frozen headline is rendered as a visually distinct marker (diamond + HEADLINE_COLOR + HEADLINE_LABEL, OR row-spanning dashed reference line), every at-risk companion JSON carries a frozen_headline.conflation_guard string mentioning 'D-14-10'"
    - "Numpy-aggregation discipline inside render-only functions: the only numeric ops allowed inside the 7 new render_* functions are numpy.mean / numpy.std / numpy.histogram / np.diff (the last for monotonicity classification); no revision.core.eval calls beyond what 14-04 already established for OD-reconstruction; core/ byte-freeze (D-14-22) preserved across all 7 commits"

key-files:
  created:
    - figures/training_convergence_all_models.png
    - figures/training_convergence_all_models.pdf
    - figures/training_convergence_all_models.json
    - figures/tstr_crossmodel.png
    - figures/tstr_crossmodel.pdf
    - figures/tstr_crossmodel.json
    - figures/failure_modes_summary.png
    - figures/failure_modes_summary.pdf
    - figures/failure_modes_summary.json
    - figures/param_efficiency_pareto.png
    - figures/param_efficiency_pareto.pdf
    - figures/param_efficiency_pareto.json
    - figures/seed_variance_per_model.png
    - figures/seed_variance_per_model.pdf
    - figures/seed_variance_per_model.json
    - figures/noise_robustness_quantum.png
    - figures/noise_robustness_quantum.pdf
    - figures/noise_robustness_quantum.json
    - figures/shot_noise_robustness.png
    - figures/shot_noise_robustness.pdf
    - figures/shot_noise_robustness.json
  modified:
    - run_figure_suite.py  # +1189 lines net (7 new render_* functions + 2 new constants + main() wiring)

key-decisions:
  - "tstr.json's model_kinds list is honored verbatim (6 entries: quantum, wgan_mlp, wgan_cnn, wgan_lstm, vae, ar) — NO fabricated V1/V2/V3 entries in tstr_crossmodel even though those models exist elsewhere in the suite. The 'quantum' kind here IS the 55-param IQP:SEL (matched2000_reproduction in model_info terms); it is colored MODEL_COLORS['iqp_sel_55_repro'] so the cross-suite legend reads consistently."
  - "tstr_crossmodel's caption_note R²<0 model list is built DYNAMICALLY from tstr.json's per_model_pipeline block at render time (plan-check fix already in the plan): the assistant detects all (model, A) entries with r2_mean<0 and inlines them into the caption_note string. Currently lists 'quantum, wgan_lstm, wgan_mlp' for Pipeline A negative R²; if tstr.json changes upstream the caption text follows automatically (no hand-typed drift)."
  - "failure_modes_summary row-A histogram overlay uses option (a) from the plan — re-derive numpy histograms from reconstruct_od + _real_references rather than embedding parent dist_*.png thumbnails. This keeps the failure_modes_summary PDF fully vectorized and avoids any image-thumbnail composition; the plan explicitly allows this since numpy histogram is rendering math, not metric math (revision.core.eval is not invoked)."
  - "failure_modes_summary outlier threshold for row C (log_return EMD) = Q3 + 1.5 × IQR across the 9 model means (Tukey fence) — a natural distribution-based threshold rather than a fixed scalar. Currently flags ar and wgan_cnn as 'log_ret blowup'; wgan_cnn also flags 'wrong mean' in row A. The thresholds for rows A/B (|fake_mean - real_mean| > 0.1*|real_mean| and |ACF_fake[1] - ACF_real[1]| > 0.5) are the plan's recommended natural thresholds."
  - "param_efficiency_pareto frozen headline x-coordinate is x = log10(model_info['iqp_sel_55_headline'].parameter_count) = log10(55) — at the same x as the iqp_sel_55_repro circle (also 55p) but rendered with a DIFFERENT marker shape (diamond), DIFFERENT color (HEADLINE_COLOR vs MODEL_COLORS['iqp_sel_55_repro']), AND a different y (the frozen headline's OD/log_return EMD is lower because epoch 1969 was the best-EMD checkpoint of the original training run). All three differences satisfy D-14-10 visual distinction; no Pareto frontier line was added (Claude's plan-authorized discretion — the family-marker legend already conveys the cluster structure)."
  - "noise_robustness_quantum and shot_noise_robustness panel layout: 1×2 panels by channel for noise, single panel for shots (because shot_noise_sensitivity.json contains only 3 conditions — analytic + shots_8192 + shots_1024 — too few to justify pipeline faceting; both pipelines fit on one log-x axis cleanly with horizontal analytic baselines). The plan explicitly authorized single-panel-or-faceted as Claude's discretion based on what reads cleanest."
  - "Failure_modes_summary row-A and the seed_variance_per_model log-y ranges: shared bin grid (row A) and shared y_lo/y_hi (seed_variance) computed across all models so cross-column visual comparison is fair. This is rendering presentation, not metric math; the underlying values are read verbatim from companions."

patterns-established:
  - "Plan-14-10 'full story' pattern: each new render_* function (a) loud-fails on missing input JSON via _require/_load_json, (b) computes only numpy aggregations over already-evaluated values, (c) emits a same-stem companion JSON with render_only:true + source_artifact(s), and (d) is wired into main() AFTER render_matched2000_dualscale_comparison_table and BEFORE render_existing_introspection. The 7 new figures bring run_figure_suite.py to 84 PNG triples (>= the 16-figure verified canonical bar by a wide margin)."

requirements-completed: [PAPER-01, PAPER-09]

# Metrics
duration: ~75min
completed: 2026-05-20
---

# Phase 14 Plan 10: Full-Story Render-Only Figure Suite Summary

**Added 7 render-only figures to `scripts/run_figure_suite.py` that complete the manuscript's full narrative arc — consuming three previously-unconsumed audited paper-facing JSONs (`tstr.json`, `noise_model_sensitivity.json`, `shot_noise_sensitivity.json`) and four previously-absent head-to-head views over already-computed artifacts (cross-model training convergence, failure-mode triage grid, parameter-efficiency Pareto, per-model seed variance). Every figure is render-only over audited JSON (no retraining, no resampling, no checkpoint reload, no new metric recomputation); the frozen-checkpoint headline is visually distinct from `iqp_sel_55_repro` on every figure where both could appear (D-14-10) — diamond + HEADLINE_COLOR + horizontal/row-spanning dashed reference lines, never merged into the reproduction series; negative R² in `tstr_crossmodel` is plotted honestly (no clamp / abs / rescale) with a dynamically-built reviewer-facing caption_note. `core/` stays byte-frozen (D-14-22), `scripts/verify_number_provenance.py` stays byte-frozen (D-14-16), and the 7 new companion JSONs are auto-rglob'd into the gate's resolution corpus without any verifier edit.**

## Performance

- **Duration:** ~75 min (7 sequential tasks: smoke-test → full-suite verify → atomic commit per task; each task ~7-12 min including the standalone smoke test of the new render function in isolation before re-running the verify block)
- **Started:** 2026-05-20 (worktree agent-acaa6779e6ee578b9)
- **Completed:** 2026-05-20
- **Tasks:** 7
- **Commits:** 7 atomic `feat(14-10): <figure>` commits + this final docs commit
- **Files:** 21 new figure files (7 PNG + 7 PDF + 7 JSON) + 1 modified module (run_figure_suite.py, +1189 lines)

## Accomplishments

### Task 1 — `render_training_convergence_all_models` (commit `bfdd93d`)

9-model EMD-vs-epoch trajectories with mean ± std seed band + frozen-headline marker. Loads the 35 `matched2000/runs/<adversarial>/<seed>/metrics.json` files (7 adversarial models × 5 seeds) via `_require`/`_load_json` (loud-fail on any missing), stacks per-seed `emd_avg` arrays into (5, 201) per model, and computes per-epoch mean/std via numpy. VAE and AR are explicitly **NOT** plotted (no `emd_avg` trajectory; the in-figure caption + companion JSON's `models_skipped_no_emd_avg: ["vae","ar"]` surface that honestly). The frozen-checkpoint headline (`headline_canonical.json`, source=`frozen_checkpoint_epoch_1969`) is rendered as a DISTINCT diamond at `x=1969, y=checkpoint_emd=0.0838` + a thin horizontal dashed reference line at the same y — never merged into the `iqp_sel_55_repro` mean curve (D-14-10). Companion JSON's `frozen_headline.conflation_guard` records this verbatim.

### Task 2 — `render_tstr_crossmodel` (commit `bf4664d`)

Cross-model TSTR R²/MAE/RMSE grouped bars (1×3 panels), Pipeline A (hatched) vs Pipeline B (solid) clusters, 6 model_kinds from `tstr.json` (NO fabricated V1/V2/V3). The R² panel's y-axis extends to include the most-negative point (currently `tstr["quantum|A"]["r2_mean"] = -4.5724`) with a 0-baseline reference and an in-figure "(R² < 0 ⇒ worse than predicting the mean)" annotation. The companion JSON's `caption_note` is built DYNAMICALLY at render time from `tstr.json`'s `per_model_pipeline` block — listing the exact (model, A) entries with `r2_mean<0` (currently `quantum, wgan_lstm, wgan_mlp`) — so if `tstr.json` changes upstream the caption text follows automatically (plan-check fix; NEVER hard-coded). `tstr.json` (previously unconsumed paper-facing JSON) is now consumed.

### Task 3 — `render_failure_modes_summary` (commit `57265bb`)

3-row × 9-column diagnostic grid with columns ordered by **ascending OD EMD** (best-on-left: `vae, wgan_mlp, iqp_sel_55_repro, V3, V2, V1, wgan_lstm, ar, wgan_cnn`). Rows: (A) OD distribution overlay (numpy histogram over `reconstruct_od(repo, model, PRIMARY_SEED)` — rendering math, not metric math), (B) ACF lag-1 bar pair read directly from `acf_<model>.json` companions (`acf_real_OD[1]` and `acf_fake_OD[1]`), (C) log_return EMD bar with red-edged Tukey-fence outliers (> Q3 + 1.5×IQR). Failure annotations on natural thresholds flag `distribution_overlay|wgan_cnn` (wrong mean), `log_return_emd|ar` (log_ret blowup), and `log_return_emd|wgan_cnn` (log_ret blowup). The frozen-checkpoint headline is rendered as ROW-SPANNING dashed reference lines (vertical at OD `moment_mean` in row A, horizontal at log_return EMD in row C) — never merged into the `iqp_sel_55_repro` column (D-14-10).

### Task 4 — `render_param_efficiency_pareto` (commit `1b63460`)

log10(parameter_count) × EMD Pareto scatter, 1×2 facets (OD | log_return), family-coded markers (circle = adversarial-quantum, square = adversarial-classical, triangle = non-adversarial). Parameter counts read verbatim from `model_info.json`: `iqp_sel_55_repro=55, V1=75, V2=135, V3=75, wgan_mlp=74, wgan_cnn=73, wgan_lstm=78, vae=562, ar=3`. The frozen-checkpoint headline (`iqp_sel_55_headline`, parameter_count=55, source=`frozen_checkpoint_epoch_1969`) is rendered as a DISTINCT diamond at `x=log10(55)` — same x as the `iqp_sel_55_repro` circle but with a different marker shape (diamond vs circle), different color (HEADLINE_COLOR vs MODEL_COLORS['iqp_sel_55_repro']), AND a different y (epoch 1969 had a lower OD EMD than the matched-2000ep reproduction). All three differences satisfy D-14-10 visual distinction; `frozen_headline.conflation_guard` in the companion JSON records this.

### Task 5 — `render_seed_variance_per_model` (commit `0c8cf68`)

3×3 facet grid (shared x, log-y, shared y-range) of per-model 5-seed EMD trajectories (light, alpha=0.35) + across-seed mean (bold, linewidth=2.2). Layout row-major matches `MODEL_ORDER`. VAE and AR panels carry an explicit "no eval-epoch trajectory" caption box on otherwise-blank axes — never fabricated. Per-panel spread labels derived from final-step std/mean ratio (<5% = tight, 5-15% = moderate, ≥15% = noisy) flag `wgan_cnn` as **noisy** — confirming the plan's reviewer-facing payoff prediction (quantum cluster tight, wgan_cnn noisier). All 35 per-run `metrics.json` files loaded via `_require`/`_load_json` (loud-fail on any missing); numpy mean/std over already-evaluated `emd_avg` arrays only.

### Task 6 — `render_noise_robustness_quantum` (commit `ac31814`)

1×2 panels (depolarizing | amplitude_damping). For each channel, Pipeline A (dashed, squares) and Pipeline B (solid, circles) EMD curves vs noise level [0.0, 0.001, 0.01, 0.05] with error bars over 3 seeds [42, 43, 44]. Open-ring markers at level=0.0 preserve the zero anchor (depol_0.0 / ampdamp_0.0 = same physical baseline by `zero_anchor_note`, kept as distinct per-curve rows per the source JSON). Monotonicity check across the level sequence per (channel, pipeline) records `monotonic_increase` for ALL 4 curves — a clean physical result preserved in the companion JSON's `monotonicity` block. Filter is `metric_name=='emd' AND scale=='OD'`; aggregation is numpy mean/std over the 3 seeds. `noise_model_sensitivity.json` (previously unconsumed paper-facing JSON) is now consumed.

### Task 7 — `render_shot_noise_robustness` (commit `ea09265`)

Single-panel EMD vs shot count (log-x). Pipeline A (dashed, squares, #D55E00) and Pipeline B (solid, circles, #0072B2) for finite shot levels (shots_1024 and shots_8192), with horizontal dotted analytic-statevector baselines per pipeline (the shots=∞ asymptotes). Analytic baselines (verbatim per-seed mean): Pipeline A = 1.0512 (n_seeds=3), Pipeline B = 0.0297 (n_seeds=3); finite-shot curves are very close to their analytic asymptotes (shot-noise is a tiny effect at 1024+ shots on the OD scale for both pipelines). `shot_noise_sensitivity.json` (previously unconsumed paper-facing JSON) is now consumed; **ALL 7 plan-14-10 figure triples are present**.

## Three Previously-Unconsumed JSONs — Now Consumed

| JSON path | Previously | Now consumed by |
|---|---|---|
| `results/tstr.json` | orphan paper-facing artifact (data_hash 91e447d4624e25b3 lineage, but no figure) | `tstr_crossmodel` (Task 2) — every plotted value read verbatim from `tstr["<model>|<pipeline>"]` |
| `results/noise_model_sensitivity.json` | orphan paper-facing artifact (720 rows of audited EMD / moments / JSD across 8 conditions × 3 seeds × 2 pipelines) | `noise_robustness_quantum` (Task 6) — filtered to `metric_name=='emd' AND scale=='OD'`, grouped by (noise_model, noise_level, pipeline), aggregated mean ± std over seeds |
| `results/shot_noise_sensitivity.json` | orphan paper-facing artifact (270 rows of audited EMD / moments / JSD across 3 conditions × 3 seeds × 2 pipelines) | `shot_noise_robustness` (Task 7) — same filter / group / aggregate; analytic condition treated as horizontal reference (shots=∞ asymptote) |

## Frozen-Headline Conflation Guard (D-14-10)

The three at-risk figures where both `iqp_sel_55_repro` and the frozen-checkpoint headline could appear all carry a `frozen_headline.conflation_guard` string mentioning D-14-10 in their companion JSON:

| Figure | Frozen-headline rendering | Conflation guard verified |
|---|---|---|
| `training_convergence_all_models.json` | diamond at x=1969 (`marker="D", color=HEADLINE_COLOR, s=100`) + horizontal dashed reference line at y=checkpoint_emd | `frozen_headline.conflation_guard` contains "D-14-10" |
| `failure_modes_summary.json` | row-spanning dashed reference lines (vertical at OD moment_mean in row A, horizontal at log_return EMD in row C) | `frozen_headline_references.conflation_guard` contains "D-14-10" |
| `param_efficiency_pareto.json` | diamond at x=log10(55) — distinct marker shape AND color AND y from the `iqp_sel_55_repro` circle at the same x | `frozen_headline.conflation_guard` contains "D-14-10" |

`tstr_crossmodel`, `seed_variance_per_model`, `noise_robustness_quantum`, and `shot_noise_robustness` do not need the conflation guard — they either don't show the frozen headline at all (tstr.json doesn't include it; seed_variance is by-seed within a single model) or are quantum-only sensitivity sweeps where there's no concurrent reproduction to confuse.

## Deviations from Plan

**None — plan executed exactly as written, with every Claude-discretion authorization exercised within the bounds the plan explicitly allowed.**

The plan's plan-check fix (caption_note built dynamically from `tstr.json`'s `per_model_pipeline` block, never hard-coded) was already present in the plan before this executor started and was honored verbatim — `render_tstr_crossmodel` reads `per_mp[k]["r2_mean"] < 0` at render time and inlines the resulting model list into the `caption_note` string.

## Render-Only Contract Audit

All seven `<verify>` blocks ran clean. The contract greps:

- `grep 'matplotlib.use("Agg")'` — present (host module headless-agg preserved across all 7 task commits)
- `! grep -E '\.fit\(|def train_|model\.sample\(|best_checkpoint\.pt'` — clean (no training/sampling/checkpoint paths introduced)
- `[ -z "$(git diff --stat core/)" ]` — empty (D-14-22 byte-freeze preserved)
- `[ -z "$(git diff --stat verify_number_provenance.py)" ]` — empty (D-14-16 byte-freeze preserved)
- `revision.core.eval` calls inside the 7 new functions: none (only `numpy.mean / std / histogram / diff` — rendering math, NOT metric math)

## Commits

| Task | Commit | Stem | LOC |
|------|--------|------|-----|
| 1 | `bfdd93d` | `training_convergence_all_models` | +3058 (includes per-model mean+std emd_avg arrays in companion JSON) |
| 2 | `bf4664d` | `tstr_crossmodel` | +357 |
| 3 | `57265bb` | `failure_modes_summary` | +500 |
| 4 | `1b63460` | `param_efficiency_pareto` | +312 |
| 5 | `0c8cf68` | `seed_variance_per_model` | +220 |
| 6 | `ac31814` | `noise_robustness_quantum` | +280 |
| 7 | `ea09265` | `shot_noise_robustness` | +199 |

Net delta on `scripts/run_figure_suite.py`: **+1189 lines** (7 new render functions + 2 new module-level constants + 7 main() wiring lines).

## Threat Surface Scan

No new threat surface introduced beyond what the plan's `<threat_model>` (T-14-25 through T-14-32) already covers. All eight threats mitigated:

- T-14-25 (silent partial render): mitigated — every new function uses `_require`/`_load_json` (loud-fail FileNotFoundError on any missing input)
- T-14-26 (frozen headline merged into iqp_sel_55_repro): mitigated — three at-risk figures carry conflation_guard strings; visually verified diamond / row-spanning-dashed treatments
- T-14-27 (hand-typed numbers): mitigated — every plotted value read from an audited JSON via `_load_json` + dict access; no hand-typed numeric literals in the 7 new functions
- T-14-28 (negative R² clamped or rescaled): mitigated — raw negative `r2_mean` values appear verbatim in `tstr_crossmodel.json`'s `per_model_pipeline` block; caption_note carries explicit reviewer-facing R²<0 explanation
- T-14-29 (`core/` edit slips in): mitigated — `git diff --stat core/` empty after every task's verify
- T-14-30 (metric recomputation inside render functions): mitigated — verify greps confirm no `.fit/train_/sample/checkpoint` patterns added; only numpy aggregations
- T-14-31 (`scripts/verify_number_provenance.py` modified): mitigated — gate is byte-frozen across all 7 task verifies; new companions land under `figures/` and are auto-rglob'd by the existing gate
- T-14-32 (figure ↔ data provenance for new figures): mitigated — every companion JSON records `source_artifact(s)` and `render_only: true`

## Self-Check: PASSED

- `figures/training_convergence_all_models.{png,pdf,json}` — FOUND
- `figures/tstr_crossmodel.{png,pdf,json}` — FOUND
- `figures/failure_modes_summary.{png,pdf,json}` — FOUND
- `figures/param_efficiency_pareto.{png,pdf,json}` — FOUND
- `figures/seed_variance_per_model.{png,pdf,json}` — FOUND
- `figures/noise_robustness_quantum.{png,pdf,json}` — FOUND
- `figures/shot_noise_robustness.{png,pdf,json}` — FOUND
- All 7 commit hashes `bfdd93d / bf4664d / 57265bb / 1b63460 / 0c8cf68 / ac31814 / ea09265` present in `git log --oneline`
- `scripts/run_figure_suite.py` modified (+1189 lines, 7 new render_* functions wired into main())
- `core/` byte-frozen (D-14-22): `git diff --stat core/` empty
- `scripts/verify_number_provenance.py` byte-frozen (D-14-16): `git diff --stat verify_number_provenance.py` empty
