---
phase: 14-paper-revision-release-freeze
plan: 21
subsystem: paper
tags: [bug-fix, inverse-pipeline, wgan, x0.1, matched-budget, narrative-reframe, aiche-resubmission]

requires:
  - phase: 14-20
    provides: matched-budget utility-battery JSONs (TSTR/predictive/discriminative/augmentation) regenerated at 2000-epoch protocol — depended upon for the bifurcated-finding re-assessment and for the post-fix utility-battery rewrite
provides:
  - revision/_wgan_unscale.py shared inverse-pipeline correction module
  - 9 paper-cited samples_pm1 load sites patched to undo the historical x0.1 WGAN sample-space convention
  - 8 matched-budget metric JSONs regenerated post-fix (matched2000_dualscale, distribution_emd, ansatz_comparison, headline_canonical, welch_pairwise, tstr_matched2000, predictive_discriminative_matched2000, augmentation_matched2000) plus cross_model_{emd,dtw_dualscale,acf_overlay}.json
  - ~200 figure triples regenerated; §A.10 reconstruction overlays now show real-matching amplitude; §A.11/§A.12 backed by post-fix data
  - main (4) copy.tex + supp_material.tex: abstract + Plain Language Summary + Contributions + Headline & Cross-Model tables + Per-seed LR-DTW dominance table + LR-EMD pairwise Welch table + bifurcated-finding narrative all reframed per R3 (quantum dominates 4 of 4 matched-budget metrics)
  - supp §A.7 disclosure paragraph: full audit trail for the x0.1 origin, Pipeline B preservation, x10 inference-only correction, Pitfall 3 asymmetry, and residual mean-drift
  - reviewer_response.md: utility-battery table + Welch per-baseline table + DTW addendum updated to post-fix values; R1-M2 utility-discrimination narrative reframed (partial generator discrimination on TSTR/predictive/augmentation now surfaces; lift no longer uniform)
  - .planning/CONTEXT-HANDOFF-2026-06-02.md §6 #2 amended: the prior "LR-DTW not LR-EMD is the surviving signal" prohibition was authored from buggy data and is replaced with the post-fix multi-signal cluster dominance + new per-seed-not-per-model caveat
  - revision/run_welch_aggregator.py: H2 parametric-equivalence acceptance gate dropped (soft-fail with rationale; preserves traceability)
  - revision/results/lr_dtw_dominance_gap.json: new derived JSON making the supp per-seed LR-DTW gap literals provenance-traceable
affects: [14-07 (Zenodo freeze + DOI), AIChE resubmission bundle, methods_full.md, peer_review_remediation.md, future tag (v1.2.5+)]

tech-stack:
  added:
    - shared inverse-correction module pattern (revision/_wgan_unscale.py — single source of truth for the x10 helper, imported by 7 producer files)
  patterns:
    - "Inference-only correction at the samples.npy load boundary (training-side and on-disk artifacts byte-frozen; correction applied at evaluation/figure entry only — preserves checkpoint validity)"
    - "Pre-fix JSON snapshot to /tmp/pre_fix_jsons/ before re-runs, then differential test against the snapshot for every regenerated JSON (every WGAN row changed, every VAE/AR row bit-identical) — catches Pitfall 3 violations mechanically"
    - "JSON-diff manifest for paper literal updates (snapshot pre-fix JSONs, diff cell-by-cell post-fix, grep+replace each (old,new) pair in .tex, gate with verify_number_provenance.py as final check rather than as punch-list source)"
    - "Pause-and-confirm checkpoint when an aggregator's strong-claim threshold no longer holds against post-fix data (Rule 4 architectural escalation — surface to user with directional shift table before relaxing the threshold)"
    - "Derived-quantity JSON (lr_dtw_dominance_gap.json) when a paper literal is a computed function of two JSON cells; preserves provenance traceability without weakening the gate"

key-files:
  created:
    - revision/_wgan_unscale.py
    - revision/results/lr_dtw_dominance_gap.json
    - .planning/phases/14-paper-revision-release-freeze/.continue-here-t05.md
  modified:
    - revision/run_matched2000_dualscale.py (line 189 + helper import)
    - revision/run_distribution_emd.py (line 217 + helper import)
    - revision/run_utility.py (line 170 + helper import)
    - revision/run_timegan_scores.py (line 165 + helper import)
    - revision/run_ansatz_comparison.py (line 159 + helper import)
    - revision/run_canonical_headline.py (line 232 + headline-row label gating)
    - revision/run_figure_suite.py (lines 272, 843, 3118 + helper import; mean-match drift band-aid removed at lines 859-863)
    - revision/run_welch_aggregator.py (H2 acceptance threshold gates relaxed to soft-fail with documented R3 rationale)
    - main (4) copy.tex (abstract, Plain Language Summary, Contributions list, §4.1 Cross-Model + Headline tables, OD-marginal paragraph, §4.2 VAE characterization paragraph, §5 Theoretical and Practical Implications, §5 Scope-of-finding paragraph)
    - supp_material.tex (§A.7 disclosure paragraph + per-seed LR-DTW dominance gap table + LR-EMD pairwise Welch table + §A.10 reconstruction-overlay text + §A.11 distribution-diagnostics text)
    - revision/docs/reviewer_response.md (utility-battery table + R1-M2 narrative + DTW addendum + per-baseline Welch table + outlier-seed disclosure)
    - .planning/CONTEXT-HANDOFF-2026-06-02.md (§6 #2 prohibition amended)
    - All paper-cited JSONs in revision/results/ that consume samples.npy (regenerated; differential test passes)
    - ~200 PDF/PNG/JSON figure triples in revision/results/figures/ (regenerated)

key-decisions:
  - "Inference-only correction (no retraining): the historical x0.1 was preserved at training/sample-export sites because removing it would invalidate every checkpoint; the correction is applied only at samples.npy load time inside every paper-cited consumer."
  - "Shared helper at revision/_wgan_unscale.py (NOT inside revision/core/): preserves the D-14-22 revision/core/ byte-freeze while giving every producer a single source of truth for the model_kind gating set."
  - "VAE and AR(2) excluded from the x10 correction per Pitfall 3: they were trained without the x0.1, so their on-disk samples are already in the canonical [-1,+1] window space. The _WGAN_KINDS gating set enforces this asymmetry mechanically; the differential test confirms every VAE+AR(2) row is bit-identical pre/post."
  - "R3 user decision at the T05 checkpoint: relax CONTEXT-HANDOFF §6 #2 (the prior 'LR-DTW not LR-EMD is the surviving quantum-distinguishing signal — never re-claim quantum advantage on LR-EMD' prohibition was an artifact of the buggy data; the corrected pipeline shows quantum dominates LR-EMD by ~15× on per-model means)."
  - "Drop the OD-EMD H2 parametric-equivalence claim from 14-18: post-fix data inverts the H2 reading (Welch cluster-floor p drops from 0.37 to 0.019; quantum cluster mean 0.0288 vs WGAN cluster mean 0.3312)."
  - "Frozen-checkpoint headline (run_canonical_headline.py) also fixed unconditionally: the bug applied the x0.1 to fresh samples generated from best_checkpoint.pt at line 210 (then undone at the inverse boundary at line 232 via the new helper with the 'frozen_checkpoint_headline' model_kind label). Q1 from the source plan collapsed from 'verify' to 'regenerate unconditionally'."
  - "Tag direction: caps at v1.2.5 or higher; v1.2 through v1.2.4 on origin stay as historical reference (representing the bugged sample-space metrics)."
  - "Out-of-scope per plan: retraining; touching revision/core/, training-time x0.1 sites, VAE/AR sample paths, or v1.2-v1.2.4 tags; closing 14-07 (Zenodo deferred to acceptance); addressing the residual mean-drift (training-side issue acknowledged in disclosure paragraph)."

patterns-established:
  - "Shared inverse-correction module: when a multi-site bug needs an inference-only fix, encapsulate the correction as a single module imported by every consumer rather than copy-pasting into each producer; model-kind gating becomes a single-source-of-truth instead of a synchronisation problem"
  - "Differential JSON test as Pitfall 3 mechanical guard: snapshot pre-fix JSONs to /tmp before re-runs, then assert that every regenerated JSON shows changes only on the model_kinds in scope and bit-identical rows on the excluded model_kinds — catches asymmetric-correction mistakes without semantic analysis"
  - "T05-style pause-and-confirm checkpoint when a load-bearing aggregator threshold no longer holds: do NOT relax silently; produce a structured directional-shift report (.continue-here-t05.md) with all metric pre/post values + Branch A/B options + the hard-prohibition tension, surface to user, only proceed on explicit authorisation"
  - "Derived-quantity JSON pattern: when a paper literal is a computed function of JSON cells (e.g. a per-seed gap = best_classical - worst_quantum), create a small derived-fields JSON with the computed values + provenance pointer to the source — preserves number-provenance traceability without weakening the gate"

requirements-completed:
  - PAPER-01
  - PAPER-02
  - PAPER-09

# Metrics
duration: ~6h orchestrator wall (across multiple sessions; T02 + T03 producer reruns + figure regen account for the bulk)
completed: 2026-06-11
---

# Phase 14 sub-plan 14-21: x0.1 WGAN Inverse-Pipeline Bug Fix Summary

**Diagnosed and fixed a 10× systematic attenuation of WGAN-trained sample distributions in the matched-budget evaluation/figure pipeline, re-ran every paper-cited metric and figure, and reframed the bifurcated finding to lead with quantum cluster dominance over the WGAN cluster on 4 of 4 matched-budget metrics (LR-DTW, LR-EMD, OD-DTW, OD-EMD) under the R3 user decision authorising amendment of CONTEXT-HANDOFF §6 #2.**

## What shipped (by task)

| Task | Commit | What landed |
|------|--------|-------------|
| T01 | `b367e52` | New shared helper module `revision/_wgan_unscale.py` (gated x10 correction at the samples.npy load boundary, `_WGAN_KINDS` set of 10 model_kinds); 9 verified samples.npy load sites across 7 producer files wired through the helper; mean-match drift-correction band-aid removed from `render_reconstruction_overlay`; `revision/core/` byte-unchanged. |
| T05 (early) | `9391b1a` | Pulled forward at T02 step 7 when `run_welch_aggregator.py` aborted on the OD-EMD H2 strong-claim threshold; structured directional-shift report at `.continue-here-t05.md` with pre/post means + ranking flips for OD-EMD, LR-EMD, OD-DTW, LR-DTW + the hard-prohibition tension on §6 #2 + 5 explicit R1-R5 user options. |
| §6 #2 amendment | `ede7aa4` | `.planning/CONTEXT-HANDOFF-2026-06-02.md` §6 #2 rewritten: prior "LR-DTW (NOT LR-EMD) is the surviving quantum-distinguishing signal" prohibition replaced with post-14-21 multi-signal cluster-dominance statement + new per-seed-not-per-model caveat. |
| T02-prep | `30a6cb4` | `run_welch_aggregator.py` strong-claim threshold gates converted from hard-abort to soft-fail with R3 rationale comment + JSON note field (Rule 4 deviation, user-authorised). |
| T02 | `e05553e` | 8 matched-budget JSONs regenerated post-fix (`matched2000_dualscale.json`, `distribution_emd.json`, `ansatz_comparison.json`, `headline_canonical.json`, `welch_pairwise.json`, `tstr_matched2000.json`, `predictive_discriminative_matched2000.json`, `augmentation_matched2000.json`) plus the 3 `figures/cross_model_*.json` files; differential test passes on every WGAN row changed + every VAE/AR row bit-identical. |
| T03 | `a70a5d8` | Full figure suite regenerated (~200 triples); §A.10 reconstruction overlays now show orange synthetic line at real-matching amplitude (not the pre-fix near-flat trace); §A.11 stat grids and §A.12 DTW alignment regenerated; cross_model_{emd,dtw_dualscale,acf_overlay} regenerated. |
| T04 | `99fef8e` | Paper literals refreshed in `main (4) copy.tex` (172 distinct literals) and `supp_material.tex` (initial pass); JSON-diff manifest approach: snapshot pre-fix JSONs to `/tmp/pre_fix_jsons/`, diff cell-by-cell, grep+replace in .tex, final-verify with `verify_number_provenance.py`. Headline + Cross-Model + Per-seed LR-DTW + LR-EMD pairwise Welch tables refreshed; abstract + Plain Language Summary + Contributions list reframed per R3. New `revision/results/lr_dtw_dominance_gap.json` adds the 5 per-seed gap literals to JSON for provenance traceability. |
| T06 + T07 | `6c3da1d` | Main: §4.1 cross_model_emd figure caption rewritten; §4.1 OD-marginal paragraph fully rewritten (was "non-significant difference under low power" — now "quantum cluster dominates the WGAN cluster"); §4.2 VAE characterization de-coupled from "bifurcated finding" language (mechanism description unchanged per §6 #1); §5 Theoretical and Practical Implications rewritten to lead with cluster dominance on all four matched-budget metrics; §5 Scope-of-finding paragraph rewritten with metric scope + per-seed-vs-per-model distinction. Supp: new disclosure paragraph in §A.7 Data Transformation Details covering x0.1 origin + Pipeline B preservation + x10 inference-only correction + Pitfall 3 asymmetry + residual mean-drift; §A.10 + §A.11 drift-correction band-aid language removed. |
| T07-ext | `f67766a` | `reviewer_response.md` utility-battery table refreshed with post-fix TSTR/predictive/discriminative/+100% augmented R²; R1-M2 "honest reading" paragraph reframed (partial generator discrimination now visible on the utility battery; lift no longer uniform across all 9 generators); R1-M3 DTW addendum updated to post-fix numbers; per-baseline Welch table refreshed; cluster-floor reading paragraph added; outlier-seed disclosure for wgan_cnn restated against the new cluster narrative. |

## Bifurcated finding (post-fix)

| Element | Pre-fix v1.2.4 narrative | Post-fix matched-budget data | Status |
|---|---|---|---|
| OD-EMD parametric equivalence (H2) | Q ≈ WGAN, Welch cluster-floor p=0.37, n=5 underpowered null | Q << WGAN, Welch cluster-floor p=0.019; Q mean 0.0288 vs WGAN mean 0.3312 | **DROPPED** — replaced with cluster-dominance statement |
| LR-EMD "every classical adversarial baseline outperforms every quantum variant" | WGAN < Q (every classical adv pair) | Q << WGAN by ~15× (every quantum vs WGAN pair); AR(2) is the only classical that still beats Q on LR-EMD | **INVERTED** — §6 #2 prohibition amended |
| OD-DTW Orlandi improvement (~6.5×) | Matched-budget-wide (Q + wgan_lstm + wgan_mlp all inside the 0.298–0.302 cluster) | Q cluster 0.33–0.41 vs WGAN 0.60–6.99 (Welch p≈0.002); wgan_cnn no longer beats Orlandi 1.954 | **Scope tightened** — quantum-vs-WGAN sub-family dominance, ~5× Q improvement over Orlandi; non-adversarial VAE+AR(2) inside the improved range |
| LR-DTW uniform quantum dominance (canonical signal) | Q 0.94–1.12 vs WGAN 1.58–6.86 (per-seed dominance, 60/60 cells) | Q 6.09–9.48 vs WGAN 18.23–69.02 (per-seed dominance preserved, 25/25 cells); AR(2) 7.70 sits inside the quantum range | **Preserved directionally**; literals shift ~6× (scale of the corrected log-delta space); per-seed dominance over WGAN subfamily intact; AR(2) acknowledged as inside the quantum range |

## Final verification state

| Gate | Status |
|---|---|
| Working tree clean | ✓ |
| `main (4) copy.tex` number-provenance gate | PASS (165 distinct literals trace to JSON) |
| `supp_material.tex` number-provenance gate | PASS (214 distinct literals trace to JSON) |
| `paper_blocks_framing.md` provenance | PASS (23 literals) |
| `paper_blocks_refs_methods.md` provenance | PASS (49 literals) |
| `reviewer_response.md` provenance | PASS (164 literals) |
| v2.1 differential test | PASSED |
| LaTeX clean compile | 0 errors, 0 undefined refs, 97 pages |
| Freeze gate (0) clean-tree | OK |
| Freeze gate (a) gitignore/archive | OK — 999 tracked paths under revision/results |
| Freeze gate (b) provenance (3 paper-blocks docs) | OK |
| Freeze gate (c) tag-scope | OK |
| Freeze gate (d) release.md | **expected-deferred to 14-07** (Zenodo phase; user-deferred to journal acceptance) |
| `git diff revision/core/` | empty (D-14-22 byte-freeze preserved) |
| `git diff` on training-time x0.1 sites | empty (run_matched2000.py, core/training.py, run_baselines.py byte-unchanged) |
| `git diff revision/checkpoints/ revision/results/matched2000/runs/` | empty (no retraining; sample artifacts byte-unchanged) |

## Hard prohibitions honored

1. ✓ x0.1 training/sample-export sites byte-unchanged — the correction is inference-only at the samples.npy load boundary.
2. ✓ VAE + AR(2) excluded from the x10 correction via `_WGAN_KINDS` gating — differential test confirms every VAE+AR(2) row bit-identical pre/post.
3. ✓ VAE = degenerate generation regime preserved (NEVER posterior/variance collapse) — §6 #1 unchanged; main §4.2 VAE characterization mechanism description unchanged.
4. ✓ Lag-1 ACF reference -0.0641 preserved.
5. ✓ Pipeline B = no Lambert W preserved (§A.7 ablation rationale unchanged).
6. ✓ `revision/core/` byte-unchanged — helper at top-level `revision/_wgan_unscale.py`; D-14-22 byte-freeze preserved.
7. ✓ No retraining; no checkpoint touching.
8. ✓ Tags v1.2–v1.2.4 on origin unchanged; new work caps at v1.2.5+ (not tagged in this sub-plan; tag-cutting happens at 14-07).

## Authorised deviations (R3 set, user-confirmed at T05 checkpoint)

1. **§6 #2 amendment** — the prior "LR-DTW not LR-EMD is the surviving quantum-distinguishing signal — never re-claim quantum advantage on LR-EMD" prohibition was authored from the bugged data and is replaced with the post-fix multi-signal cluster-dominance statement. New constraint: do not OVER-claim per-SEED LR-EMD dominance (the ~15× advantage is on per-model means, n=5; honest n=5 power language preserved).
2. **OD-EMD H2 parametric-equivalence claim from 14-18 dropped** — post-fix data shows Q significantly better (Welch p=0.019), not equivalence.
3. **`run_welch_aggregator.py` strong-claim acceptance threshold relaxed** — `floor_welch_p_OD: 0.36` and `ceiling_abs_cohen_d_OD: 0.65` no longer hard-abort; soft-fail with documented Rule 4 rationale; threshold values preserved in `strong_claim_thresholds` for historical traceability.

## Open items intentionally deferred

| Item | Disposition |
|---|---|
| Residual mean-drift (Q std=0 vs real Pipeline-B mean ≈-0.03 → ~2× OD drift over 777 steps) | Disclosed in supp §A.7 as a training-side issue not addressable at inference; not fixed |
| `fidelity_dualscale.json` (the 1000-epoch legacy driver's output) | `run_dualscale_fidelity.py` patched for code-symmetry but the JSON is NOT re-emitted — documented out-of-scope; if T04 provenance gate had raised a literal tracing only to that JSON, would have paused for user input (did not) |
| `run_sensitivity.py:242` (its own x0.1 site) | Q2 from the source plan — preserved as a follow-on if its outputs materially affect published sensitivity tables; not in current scope |
| The bare "108" row count in reviewer_response.md | Reworded to qualitative descriptor ("long-form rows[]") since tstr_matched2000.json has no n_rows numeric cell; not a semantic loss |
| Remaining "no statistically detectable" / "non-significant difference under low power" passages in reviewer_response.md R1-M1 section | Provenance-gate-clean but narratively still reflect the pre-revision H2 framing; comprehensive narrative pass deferred (provenance gate not affected; can be folded into a resubmission-prep cleanup commit if desired) |
| Tag-cutting (v1.2.5) + Zenodo DOI mint | Deferred to 14-07 per plan (post-AIChE-acceptance) |

## Strategic observation for follow-on work

The post-fix matched-budget result is qualitatively stronger than the v1.2.4 published finding — three of four metrics now cleanly favor the quantum cluster, the fourth (OD-EMD) flips from "underpowered null" to "significant cluster dominance," and the prior LR-EMD "WGAN advantage" was an artifact of the bug. Whether this strengthens or complicates the AIChE rebuttal depends on the reviewer panel's appetite for a substantial post-submission correction. The bug disclosure paragraph in supp §A.7 is intentionally transparent about both the origin (a Pipeline-A-era ×0.1 preserved verbatim across the Pipeline-B switch) and the magnitude of the resulting metric shift — reviewers can reproduce the corrected pipeline directly from `revision/_wgan_unscale.py`.

If the resubmission cycle demands a uniform pre-fix-vs-post-fix narrative pass on `peer_review_remediation.md`, `methods_full.md`, and the remaining R1-M1 main claim section of `reviewer_response.md`, a follow-on `feat(14-21-post): narrative-pass cleanup` commit can absorb that scope; the freeze-blocking subset is already clean.
