---
phase: 14-paper-revision-release-freeze
plan: 15
subsystem: paper-revision-release-freeze
tags: [distribution-emd, qq-overlay, marginal-convergence, paper-revision, gate-v2.1]
requires: ["14-13", "14-14"]
provides:
  - run_distribution_emd.py (NEW top-level emitter, 50-bin histogram-density Wasserstein)
  - results/distribution_emd.json (NEW aggregator JSON; 9 models × 5 seeds × 2 scales; auto-walked into v2.1 gate corpus)
  - figures/qq_overlay.{png,pdf,json} (NEW primary discriminating figure; render_only=false)
  - docs/reconciliation_note.md (3-column comparable-variants table appended; C-3 disclosure extended)
  - docs/reviewer_response.md (## Marginal-convergence finding subsection appended)
  - docs/methods_full.md (§3.x.(e) OD-marginal convergence cross-reference appended)
  - docs/peer_review_remediation.md (## R2 follow-up — marginal-convergence finding subsection appended)
  - docs/completeness_sweep_manifest.md (## Plan 14-15 section appended)
  - 9 per-model qq_<model>.json companions gain additive caption_note field
affects:
  - run_figure_suite.py (extended with render_qq_overlay; render_qq_plot gains caption_note)
  - run_model_info.py (extended with _comparable_variants_rows + 3-column table emit + C-3 paragraph extension)
key-files:
  created:
    - run_distribution_emd.py
    - results/distribution_emd.json
    - figures/qq_overlay.png
    - figures/qq_overlay.pdf
    - figures/qq_overlay.json
    - .planning/phases/14-paper-revision-release-freeze/14-15-SUMMARY.md
  modified:
    - run_figure_suite.py
    - run_model_info.py
    - docs/reconciliation_note.md
    - docs/reviewer_response.md
    - docs/methods_full.md
    - docs/peer_review_remediation.md
    - docs/completeness_sweep_manifest.md
    - figures/qq_{ar,iqp_sel_55_repro,V1,V2,V3,vae,wgan_cnn,wgan_lstm,wgan_mlp}.json (additive caption_note)
decisions:
  - D-14-22 (`core/` byte-freeze) PRESERVED across all 5 tasks — histogram-density EMD lives in top-level emitter, not core
  - D-14-13 (strict-accept gate) PRESERVED — new distribution-EMD column is informational only, not gated
  - D-14-16 (gate v2.1 byte-freeze) PRESERVED — no edit to verify_number_provenance.py; new JSONs auto-walked by existing _json_blobs walker
  - D-14-18 (Overleaf-canonical LaTeX read-only) PRESERVED — no edits to `main (4) copy.tex` / `paper/supp_material.tex`
  - paper_blocks_framing.md SKIPPED — no natural insertion point in PAPER-01 LaTeX-replacement framing block
  - 50-bin histogram-density formulation FIXED across the plan: `wasserstein_distance(bin_centers, bin_centers, real_hist_density, fake_hist_density)`
metrics:
  duration: ~1h
  completed: 2026-05-20
---

# Phase 14 Plan 15: Distribution-EMD Column + QQ Overlay + Marginal-Convergence Reviewer Disclosure Summary

Pre-v1.0 50-bin histogram-density Wasserstein EMD reintroduced as a third comparable column in `reconciliation_note.md` (alongside OD raw-sample EMD and log-return raw-sample EMD), single discriminating figure `qq_overlay.{png,pdf,json}` (9-model QQ overlay + delta-QQ panel) emitted, and post-r2 two-pronged marginal-convergence finding (pairwise 8/9 cluster within median ~0.03 OD-units / WGAN-CNN diverges at ~0.69 vs others; vs-real all-9 deviate by ~0.25 OD-units / WGAN-CNN ~0.81) disclosed in reviewer-facing docs with 4.44e-16 floating-point routing verification.

## Tasks completed

| Task | Tag | Commit | Files |
|---|---|---|---|
| T1 | feat: dist-emd aggregator | `76430fd` | `scripts/run_distribution_emd.py` (NEW), `results/distribution_emd.json` (NEW) |
| T2 | feat: 3-column comparable table | `abf41c0` | `scripts/run_model_info.py`, `docs/reconciliation_note.md` |
| T3 | feat: qq_overlay figure | `36657ba` | `scripts/run_figure_suite.py`, `figures/qq_overlay.{png,pdf,json}` (NEW), 9 per-model `qq_<model>.json` (additive `caption_note`) |
| T4 | docs: reviewer-comms | `3a06894` | `docs/reviewer_response.md`, `docs/methods_full.md`, `docs/peer_review_remediation.md` |
| T5 | docs: SUMMARY + sweep manifest | (this commit) | `docs/completeness_sweep_manifest.md`, `.planning/phases/14-paper-revision-release-freeze/14-15-SUMMARY.md` |

## Deviations from Plan

- **paper_blocks_framing.md SKIPPED — no natural insertion point.** The PAPER-01 keyed framing block is a BEFORE/AFTER LaTeX-replacement spec for `main (4) copy.tex`, not a narrative block. There is no natural place to thread a marginal-convergence narrative without modifying the LaTeX-replacement contract. The conditional outcome was anticipated in the plan; the marginal-convergence message is carried by `methods_full.md` + `reviewer_response.md` alone. No verification gate was contingent on `paper_blocks_framing.md` carrying the new text.
- **C-3 disclosure formulation citation updated to match `distribution_emd.json#metric_formulation` word-for-word.** The pre-14-15 C-3 paragraph in `reconciliation_note.md` cited the histogram-density formulation as `wasserstein_distance(real_hist_density, fake_hist_density)`, but the actual scipy API call (and the formulation the new emitter implements verbatim per the plan) is `wasserstein_distance(bin_centers, bin_centers, real_hist_density, fake_hist_density)`. T2 updated the C-3 citation to the corrected formulation; the prior wording is removed in favor of the accurate one. This counts as Rule 1 (bug fix — the prior citation was an incomplete reference to the actual function signature) and is documented as a pre-existing inaccuracy now corrected by 14-15.

Otherwise, no deviations.

## Final-state declaration

After this plan completes, **Phase 14 incomplete plans = `[14-07]`** only (Zenodo deposit + tag + DOI wiring + release.md; deferred manual gate per `~/.claude/projects/-Users-shawngibford-dev-phd-qGAN/memory/project_phase14_zenodo_blocker.md`). ROADMAP progress row will flip from `13/15 | In Progress` (mid-plan, pre-merge) to `14/15 | In Progress` (post-merge, post-orchestrator-tracking-update).

## Cross-references

- `.planning/phases/14-paper-revision-release-freeze/14-13-SUMMARY.md` — Phase 14 plan 13 (peer-review remediation sweep)
- `.planning/phases/14-paper-revision-release-freeze/14-14-SUMMARY.md` — Phase 14 plan 14 (r2 peer-review punch list, Wave 12)
- `docs/completeness_sweep_manifest.md#plan-14-15` — full cross-plan deliverable register (this plan's section appended)
- `docs/reconciliation_note.md#emd-comparable-across-metric-variants-matched-2000ep-budget` — the 3-column comparable-variants table
- `figures/qq_overlay.png` — the single discriminating figure
- `docs/reviewer_response.md#marginal-convergence-finding-post-r2-investigation` — reviewer-facing communication

## Verification (10-point checklist)

1. ✅ `scripts/run_distribution_emd.py` exists and is executable; emits `distribution_emd.json` with 9-model × 5-seed rows + per-(model, scale) aggregates (90 rows, 18 aggregates, `n=5` per aggregate, `headline_present=false` because `iqp_sel_55_headline/samples.npy` is not on disk in this corpus).
2. ✅ `distribution_emd.json` schema includes `metric_formulation` (the 50-bin density Wasserstein description: `scipy.stats.wasserstein_distance(bin_centers, bin_centers, real_hist_density, fake_hist_density) over 50-bin histograms (np.histogram(..., density=True))`), `data_hash: "91e447d4624e25b3"`, and `n: 5` aggregates.
3. ✅ `reconciliation_note.md` carries BOTH the existing OLD/NEW OD-scale headline table (preserved verbatim from 14-13/14-14) AND the new 3-column comparable-variants table (`## EMD comparable across metric variants (matched 2000ep budget)`); the cross-reference paragraph (`**Cross-reference (Plan 14-15).**`) between them is present.
4. ✅ The metric-redefinition disclosure paragraph in `reconciliation_note.md` is extended with the histogram-density acknowledgment sentence (the matched-2000ep histogram-density EMD on OD scale is computed on the SAME real-data slice and SAME 50-bin convention as the deprecated v1.0-pre metric, commensurate with the pre-v1.0 paper headline ~0.0015 for the first time since the v1.0 raw-sample switch).
5. ✅ `qq_overlay.png/pdf/json` exist; the companion JSON's `convergence_stats` field has 9 per-model `max_abs_quantile_diff_OD_vs_real` entries (8/9 in 0.24–0.28; `wgan_cnn = 0.8141`); `cluster_summary` carries the 6 statistics enumerated in the plan; `routing_verification.max_floating_point_diff = 4.44e-16`; `render_only: false`.
6. ✅ `qq_overlay` is deterministic across re-renders (CR-1 / sha256 seeding preserved); two consecutive renders produce byte-identical companion JSON `model_order_color_keys` arrays (verified in T3 by re-rendering and Python-comparing the arrays).
7. ✅ `reviewer_response.md` carries the new `## Marginal-convergence finding (post-r2 investigation)` subsection with the 4.44e-16 routing-verification claim + the architecture-discrimination redirect to `qq_overlay.png`, `training_convergence_all_models.png`, `failure_modes_summary.png`, `seed_variance_per_model.png`, `methods_full.md §3.x`.
8. ✅ `methods_full.md` carries the §3.x.(e) cross-reference at the end of the §3.x metric-conventions section (8/9 pairwise within ~0.03 OD-units median; WGAN-CNN diverges ~0.69 vs others; all 9 ~0.25 vs real / WGAN-CNN ~0.81; pointer to `qq_overlay.png`).
9. ✅ `peer_review_remediation.md` carries the new `## R2 follow-up — marginal-convergence finding (Plan 14-15)` subsection (three threads: user-observed visual concern / 4.44e-16 verification / resolution pointer); existing 14-13 + 14-14 sections (`## Gate v2.1 known limitations`, `## R2 follow-up sweep`, `## End-to-end v2.1 gate status`) preserved verbatim.
10. ✅ `git diff` against pre-14-15 main shows zero changes to `core/` (D-14-22 preserved): `git diff --stat <pre-14-15-sha> -- core/` outputs nothing across all 5 tasks (verified at each task's `<verify>` gate via `[ -z "$(git diff --stat core/)" ]`).

## v2.1 number-provenance gate status (all 10 paper-facing docs)

| Doc | Status | Distinct literals resolved |
|---|---|---|
| `docs/paper_blocks_framing.md` | PASS | 23 |
| `docs/paper_blocks_refs_methods.md` | PASS | 49 |
| `docs/reviewer_response.md` | PASS | 49 (up from 32 pre-14-15) |
| `docs/reconciliation_note.md` | PASS | 54 (up from 36 pre-14-15) |
| `docs/methods_full.md` | PASS | 75 (up from 64 pre-14-15) |
| `docs/circuit_atlas.md` | PASS | 18 |
| `docs/completeness_sweep_manifest.md` | PASS | 46 (up from 27 pre-14-15) |
| `docs/training_protocol.md` | PASS | 18 |
| `docs/dataset_stats.md` | PASS | 5 |
| `docs/peer_review_remediation.md` | PASS | 58 (up from 31 pre-14-15) |

Schema string in every PASS message: `'v2.1 (Phase 14 plan 14-14 — negative-sign-aware lookbehind)'` — gate is byte-frozen, no edit.

## Self-Check: PASS

- ✅ All 5 task commits exist in `git log --oneline` (76430fd, abf41c0, 36657ba, 3a06894 + this T5 commit).
- ✅ `scripts/run_distribution_emd.py` exists (NEW top-level emitter).
- ✅ `results/distribution_emd.json` exists with 90 rows + 18 aggregates.
- ✅ `figures/qq_overlay.{png,pdf,json}` exist (`render_only: false`).
- ✅ 9 per-model `qq_<model>.json` carry the additive `caption_note` field.
- ✅ `docs/reconciliation_note.md` carries both tables + cross-reference + C-3 extension.
- ✅ `docs/reviewer_response.md` carries `## Marginal-convergence finding (post-r2 investigation)`.
- ✅ `docs/methods_full.md` carries §3.x.(e) OD-marginal convergence cross-reference.
- ✅ `docs/peer_review_remediation.md` carries `## R2 follow-up — marginal-convergence finding (Plan 14-15)`.
- ✅ `docs/completeness_sweep_manifest.md` carries `## Plan 14-15` section.
- ✅ This SUMMARY exists with Self-Check PASS and the `14-07` only final-state declaration.
- ✅ v2.1 gate PASSES on all 10 paper-facing docs.
- ✅ `core/` byte-untouched across all 5 tasks (D-14-22 preserved).
