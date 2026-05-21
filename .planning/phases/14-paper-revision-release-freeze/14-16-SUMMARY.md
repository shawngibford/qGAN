---
phase: 14-paper-revision-release-freeze
plan: 16
subsystem: paper-revision-release-freeze
tags: [r3-forensic-remediation, R3-CR-1, R3-CR-2, R3-HI-1, welch-pairwise, dtw-asymmetry, path-a, gate-v2.1]
requires: ["14-13", "14-14", "14-15"]
provides:
  - revision/run_welch_aggregator.py (NEW top-level emitter; per-pair Welch t/p + Cohen d + MWU p)
  - revision/results/welch_pairwise.json (NEW aggregator JSON; 40 quantum-classical pairs; OD-EMD strong-claim thresholds)
  - revision/results/matched2000_dualscale.json (corrected log-return EMD column; OD subset byte-identical)
  - revision/results/distribution_emd.json (schema v2; fake_in_range_mass disclosure stat; LR-scale sister-fix)
  - revision/docs/reviewer_response.md (## Parametric-efficiency equivalence H2 + ### DTW addendum)
  - revision/docs/methods_full.md (Log-return scale correction + Shared-edges formulation + DTW historical context paragraphs)
  - revision/docs/reconciliation_note.md (C-3 disclosure extended; 4th DTW column)
  - revision/docs/peer_review_remediation.md (## Plan 14-16 r3 forensic remediation + DTW phantom asymmetry sections)
  - revision/docs/completeness_sweep_manifest.md (## Plan 14-16 section appended)
  - revision/results/figures/qq_overlay.json (plan_14_16_verification field appended)
affects:
  - revision/run_matched2000_dualscale.py (R3-CR-2 fix in _log_return_rows; reconstruct_od returns mu/sigma)
  - revision/run_distribution_emd.py (R3-CR-1 fix in compute_histogram_density_emd; R3-HI-1 sister-fix in _real_references)
  - revision/run_model_info.py (_comparable_variants_rows gains DTW column; C-3 disclosure extended)
key-files:
  created:
    - revision/run_welch_aggregator.py
    - revision/results/welch_pairwise.json
    - .planning/phases/14-paper-revision-release-freeze/14-16-SUMMARY.md
  modified:
    - revision/run_matched2000_dualscale.py
    - revision/run_distribution_emd.py
    - revision/run_model_info.py
    - revision/results/matched2000_dualscale.json
    - revision/results/distribution_emd.json
    - revision/docs/reviewer_response.md
    - revision/docs/methods_full.md
    - revision/docs/reconciliation_note.md
    - revision/docs/peer_review_remediation.md
    - revision/docs/completeness_sweep_manifest.md
    - revision/results/figures/qq_overlay.json
decisions:
  - D-14-22 (`revision/core/` byte-freeze) PRESERVED across all 7 tasks — all fixes live in top-level emitters
  - D-14-13 (strict-accept gate) PRESERVED — no edit to gate semantics
  - D-14-16 (gate v2.1 byte-freeze) PRESERVED — no edit to verify_number_provenance.py; corrected JSONs auto-walked
  - D-14-18 (Overleaf-canonical LaTeX read-only) PRESERVED — no edits to `main (4) copy.tex` / `supp_material.tex`
  - Path A (user decision after executor checkpoint) — LR-EMD-vs-WGAN strong claim withdrawn; OD-EMD equivalence + DTW dominance retained
  - R3-CR-2 fix recipe — un-standardize-fake per pipeline-review-r3.md §2 (NOT standardize-real)
metrics:
  duration: ~1 session
  completed: 2026-05-21
---

# Phase 14 Plan 16: r3 Forensic Remediation Summary

The 5-agent r3 forensic peer-review pass (commit `961ee12`) flagged metric
bugs in the matched-2000ep evaluation pipeline. Plan 14-16 executed Path 1
(fix the bugs) and, after an executor checkpoint surfaced that the
synthesis's load-bearing LR-EMD strong claim was itself derived from the
broken column, the user selected Path A: fix the bugs, withdraw the
LR-EMD-vs-WGAN claim, and reframe to the OD-EMD parametric-efficiency
equivalence (Welch p > 0.36, |d| ≤ 0.65, n=5) + DTW dominance, both of
which survive the corrections.

## Tasks completed

| Task | Tag | Commit | Files |
|---|---|---|---|
| T1 | feat: R3-CR-2 LR-EMD scale fix | `5e37f9f` | `revision/run_matched2000_dualscale.py`, `revision/results/matched2000_dualscale.json` |
| T2 | feat: R3-CR-1 + R3-HI-1 (dist-emd v2) | `088a49f` | `revision/run_distribution_emd.py`, `revision/results/distribution_emd.json` |
| T3 | feat: welch_pairwise.json aggregator | `32209da` | `revision/run_welch_aggregator.py` (NEW), `revision/results/welch_pairwise.json` (NEW) |
| T4 | docs: reviewer_response.md Path A H2 | `028ef42` | `revision/docs/reviewer_response.md` |
| T5 | docs: methods_full + reconciliation_note | `4eccc07` | `revision/run_model_info.py`, `revision/docs/methods_full.md`, `revision/docs/reconciliation_note.md` |
| T6 | docs: figure verification + gate sweep | `61c69cb` | `revision/results/figures/qq_overlay.json` |
| T7 | docs: SUMMARY + remediation + manifest | (this commit) | `revision/docs/peer_review_remediation.md`, `revision/docs/completeness_sweep_manifest.md`, `.planning/.../14-16-SUMMARY.md` |

(Plus the scaffold commit `c843171` carrying the plan + ROADMAP/STATE updates.)

## Deviations from Plan

Three empirical findings surfaced during execution where the corrected
data contradicted predictions inherited from the r3 synthesis. All three
are documented in `peer_review_remediation.md`'s Plan 14-16 section.

- **Path A reframe (executor checkpoint).** The plan's original load-bearing
  strong claim included "quantum significantly beats every WGAN on
  log-return EMD." After the R3-CR-2 fix, the corrected LR-EMD ranking
  inverts: every WGAN beats every quantum; AR (Yule-Walker MLE) leads. The
  pre-fix Welch tests in `statistical-honesty-r3.md` §3b were computed on
  the broken column. The user selected Path A: withdraw the LR-EMD-vs-WGAN
  claim, retain the OD-EMD equivalence (byte-stable column) + DTW dominance.
- **R3-CR-1 numerically inert.** The R3-CR-1 "structural bias" fix
  (`density=True` → `density=False` + shared edges) produces byte-identical
  v1→v2 OD-scale EMD values (delta 0.00000 for all 9 models), because
  `scipy.stats.wasserstein_distance` renormalizes weights internally. The
  fix is still landed — it adds the genuine `fake_in_range_mass` disclosure
  stat and bundles the real R3-HI-1 sister-fix — but the synthesis's
  CRITICAL severity for R3-CR-1 was overstated. Documented honestly in
  `peer_review_remediation.md` and `methods_full.md` §3.x.(g).
- **cross_model_emd is OD-only.** T6's plan assumed `cross_model_emd` has a
  log-return bar group needing regeneration. The figure's companion JSON
  carries only OD-scale fields; since T1 left the OD column byte-stable,
  the figure required no re-render. Confirmed byte-stable; only a
  verification note was committed.

The R3-CR-2 fix recipe was clarified to un-standardize-fake (per
`pipeline-review-r3.md` §2), the canonical recipe that matches the §2
anchor table — the plan's earlier "standardize-real" phrasing would have
produced non-anchor-matching values.

## Final-state declaration

After this plan completes, **Phase 14 incomplete plans = `[14-07]`** only
(Zenodo deposit + tag + DOI wiring + release.md; deferred manual gate per
`~/.claude/projects/-Users-shawngibford-dev-phd-qGAN/memory/project_phase14_zenodo_blocker.md`).
ROADMAP progress row flips to `14/16 | In Progress` post-merge.

## Cross-references

- `.planning/phases/14-paper-revision-release-freeze/peer-review-r3/SYNTHESIS.md` — the r3 trigger
- `.planning/phases/14-paper-revision-release-freeze/peer-review-r3/code-review-r3.md` §H3 — R3-CR-1 + R3-CR-2 + R3-HI-1 mechanisms
- `.planning/phases/14-paper-revision-release-freeze/peer-review-r3/pipeline-review-r3.md` §2 — corrected LR-EMD anchors
- `.planning/phases/14-paper-revision-release-freeze/peer-review-r3/statistical-honesty-r3.md` §3a + §3b — Welch numbers (§3b retracted)
- `.planning/phases/14-paper-revision-release-freeze/14-16-DEVIATION-NOTE.md` — executor checkpoint analysis
- `.planning/phases/14-paper-revision-release-freeze/14-13-SUMMARY.md`, `14-14-SUMMARY.md`, `14-15-SUMMARY.md` — prior wave SUMMARYs
- `revision/docs/peer_review_remediation.md` — `## Plan 14-16 — r3 forensic remediation` + `## Plan 14-16 — DTW phantom asymmetry`
- `revision/docs/completeness_sweep_manifest.md` — `## Plan 14-16` section

## Verification (14-point checklist)

1. ✅ `revision/run_matched2000_dualscale.py` `_log_return_rows` carries the R3-CR-2 fix (`trans_flat_raw = trans_flat * sigma + mu` un-standardize-fake).
2. ✅ `matched2000_dualscale.json` re-emitted; OD-scale subset BYTE-IDENTICAL (SHA-256 `560489fa3b44...` preserved); corrected LR-EMD aggregates match `pipeline-review-r3.md` §2 anchors exactly (ar 0.00294 … vae 0.01583).
3. ✅ `revision/run_distribution_emd.py` `compute_histogram_density_emd` carries the R3-CR-1 fix (shared-edges + total-mass=1 + `fake_in_range_mass`); returns 2-tuple; SCHEMA bumped to v2.
4. ✅ `_real_references` carries the R3-HI-1 sister-fix — returns `norm_log_delta`; `_model_seed_rows` consumes the standardized reference for the log-return path.
5. ✅ `distribution_emd.json` re-emitted under schema v2 with `fake_in_range_mass` per row + `fake_in_range_mass_mean` per aggregate.
6. ✅ `revision/run_welch_aggregator.py` (NEW) emits `welch_pairwise.json` with 40 quantum-classical pairs; OD-EMD strong-claim thresholds (floor Welch p 0.36, ceiling |d| 0.65) enforced before write; computed OD floor p = 0.3652, ceiling |d| = 0.6442.
7. ✅ `welch_pairwise.json` `strong_claim_thresholds` block carries only OD-EMD thresholds (Path A); LR-EMD thresholds absent; `notes` field documents the r3-process retraction.
8. ✅ `reviewer_response.md` R1-M1 row preserved verbatim; new `## Parametric-efficiency equivalence (post-r3 corrected metrics)` H2 with Path A strong claim + `### DTW addendum (Plan 14-16)`; withdrawn LR-EMD-vs-WGAN literals absent.
9. ✅ `methods_full.md` §3.x gains three Plan 14-16 paragraphs (Log-return scale correction, Shared-edges formulation, DTW historical context).
10. ✅ `reconciliation_note.md` re-emitted via `run_model_info.py`: OD column byte-stable; columns 2+3 regenerated from corrected JSONs; C-3 disclosure extended; 4th DTW column added.
11. ✅ `cross_model_emd` confirmed OD-only and byte-stable (no re-render needed); `qq_overlay.json` carries `plan_14_16_verification` field; CR-1 determinism confirmed.
12. ✅ `peer_review_remediation.md` carries `## Plan 14-16 — r3 forensic remediation` (with r3-process retraction subsection) + `## Plan 14-16 — DTW phantom asymmetry`; existing 14-13/14-14/14-15 sections preserved verbatim.
13. ✅ `completeness_sweep_manifest.md` carries `## Plan 14-16` section; existing per-plan sections preserved verbatim.
14. ✅ `revision/core/` byte-untouched across all 7 tasks (D-14-22 preserved); `revision/verify_number_provenance.py` byte-untouched (D-14-16 preserved).

## v2.1 number-provenance gate status (all 10 paper-facing docs)

| Doc | Status | Distinct literals resolved |
|---|---|---|
| `revision/docs/paper_blocks_framing.md` | PASS | 23 |
| `revision/docs/paper_blocks_refs_methods.md` | PASS | 49 |
| `revision/docs/reviewer_response.md` | PASS | 83 |
| `revision/docs/reconciliation_note.md` | PASS | 67 |
| `revision/docs/methods_full.md` | PASS | 105 |
| `revision/docs/circuit_atlas.md` | PASS | 18 |
| `revision/docs/completeness_sweep_manifest.md` | PASS | 47 |
| `revision/docs/training_protocol.md` | PASS | 18 |
| `revision/docs/dataset_stats.md` | PASS | 5 |
| `revision/docs/peer_review_remediation.md` | PASS | 105 |

Schema string in every PASS message: `'v2.1 (Phase 14 plan 14-14 — negative-sign-aware lookbehind)'` — gate byte-frozen, no edit.

## Self-Check: PASS

- ✅ All 7 task commits exist in `git log --oneline` (5e37f9f, 088a49f, 32209da, 028ef42, 4eccc07, 61c69cb + this T7 commit).
- ✅ `revision/run_welch_aggregator.py` + `revision/results/welch_pairwise.json` exist (NEW).
- ✅ `matched2000_dualscale.json` OD subset byte-identical; LR-EMD corrected.
- ✅ `distribution_emd.json` schema v2 with `fake_in_range_mass`.
- ✅ `reviewer_response.md` carries the Path A `## Parametric-efficiency equivalence` H2 + DTW addendum.
- ✅ `methods_full.md` + `reconciliation_note.md` carry the Plan 14-16 correction paragraphs.
- ✅ `peer_review_remediation.md` + `completeness_sweep_manifest.md` carry the Plan 14-16 sections.
- ✅ This SUMMARY exists with Self-Check PASS and the `14-07`-only final-state declaration.
- ✅ v2.1 gate PASSES on all 10 paper-facing docs.
- ✅ `revision/core/` + `verify_number_provenance.py` byte-untouched (D-14-22 + D-14-16 preserved).
