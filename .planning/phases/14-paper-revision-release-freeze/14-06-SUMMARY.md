---
phase: 14-paper-revision-release-freeze
plan: 06
subsystem: paper-revision-package
tags: [latex-blocks, reference-surgery, methods-from-json, reviewer-rebuttal, number-provenance-gate, read-only-tex, location-independent]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze (plan 03)
    provides: "model_info.json (dataset block + seed_set + per-model records), reconciliation_note.md (1000ep->2000ep deltas), verify_number_provenance.py (the executable success-criterion-5 gate)"
  - phase: 14-paper-revision-release-freeze (plan 04)
    provides: "results/figures/* (acf_iqp_sel_55_repro, training_progression, param_trajectory, entanglement_trajectory, cross_model_emd) — figure artifact paths cited by reviewer_response.md"
  - phase: 11-utility-eval
    provides: "tstr.json, predictive_discriminative.json, augmentation.json, fidelity_dualscale.json — dual-scale + utility artifacts cited as supporting evidence"
provides:
  - "docs/paper_blocks_refs_methods.md — copy-paste LaTeX blocks for PAPER-06 (ref surgery), PAPER-07 (Bernal et al.), PAPER-08 (dataset details, render-from-JSON), PAPER-09 (per-metric eval scale, render-from-JSON), PAPER-10 (A3 proposed-extension relabel + log-GAN vs Wasserstein clarification), PAPER-11 (R1-m7 typo checklist); cite-key/label/anchor-keyed, location-independent"
  - "docs/reviewer_response.md — AIChE per-reviewer point-by-point rebuttal; one row per comment ID R1-M1..M5, R1-m1..m7, R2-1..6 -> verbatim concern -> change -> manuscript location -> real results/* supporting artifact"
affects: [14-07, paper-resubmission, manuscript-overleaf]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Location-independent paper-revision block: every fix keyed to a \\cite{} key / \\label{} / verbatim anchor sentence so it applies regardless of Overleaf-external .bib/.tex location (RESEARCH A1, D-14-18)"
    - "Render-from-JSON Methods block: every numeric literal carries a '% source: results/<file>.json#<path>' annotation and uses the EXACT stored value so the substring resolves under the number-provenance gate (D-14-16)"
    - "Sourced-row rebuttal table (training_protocol.md discipline): every comment-ID row carries a real results/* supporting-artifact provenance cell; no TODO/TBD/placeholder cells"

key-files:
  created:
    - docs/paper_blocks_refs_methods.md
    - docs/reviewer_response.md
  modified: []

key-decisions:
  - "PAPER-06/07 delivered as .bib-entry + sentence-rewrite blocks keyed to the \\cite{} keys observed in main (4) copy.tex (esteban2017realvaluedmedicaltimeseries, Mugel2022, giraldo2025q2sar, chokwitthaya2020applying, wang2018esrgan..., dimoudis2023utilizing, Cerezo_2021, yoon2019TimeGAN) — Overleaf-external .bib (A1) means location-independent cite-key keying is the only robust handle; reused already-defined keys (yoon2019TimeGAN, dimoudis2023utilizing) where the correct reference is already in the manuscript instead of inventing a new one."
  - "PAPER-06.g resolves [55]-[57],[59] by REMOVAL of the over-reaching quantum-advantage claim (not substitution) — consistent with the PAPER-02 no-overclaiming lock (D-14-20); adding a not-yet-demonstrated quantum-advantage citation would re-introduce the exact overclaim R1-M5 flagged."
  - "PAPER-08/09 numbers use the EXACT full-precision stored JSON value (e.g. emd log_return seed42 = 0.1209437521974767) rather than a rounded display value — the number-provenance gate resolves by substring/precision against fidelity_dualscale.json / model_info.json, so a rounded literal would fail the gate; render-from-JSON is enforced, not decorative."
  - "PAPER-09 reports a single representative seed (42), Pipeline B (native preprocessing) per metric on BOTH scales — concrete dual-scale labeling satisfies R1-m3 without duplicating the full multi-seed table that lives in fidelity_dualscale.json (single source of truth, D-14-16)."

patterns-established:
  - "Pattern: a paper-revision LaTeX-blocks file is shipped together with its executable number-provenance proof (verify_number_provenance.py PASS) in the same plan that authors it — success-criterion-5 is gated, not asserted"

requirements-completed: [PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10, PAPER-11]

# Metrics
duration: ~35min
completed: 2026-05-19
---

# Phase 14 Plan 06: References / Methods / Typos LaTeX Blocks + Per-Reviewer Rebuttal Summary

**Delivered `docs/paper_blocks_refs_methods.md` — cite-key/label/anchor-keyed, location-independent copy-paste LaTeX blocks for PAPER-06 (per-reference surgery + RETAINED-anchor note), PAPER-07 (Bernal et al. AIChE perspective), PAPER-08 (dataset-details Methods, render-from-JSON), PAPER-09 (per-metric evaluation-scale Methods table, render-from-JSON), PAPER-10 (Appendix A3 relabeled a proposed extension + the log-GAN vs Wasserstein discrepancy clarified + Table A2 caveated), and PAPER-11 (one keyed before→after block per R1-m7 typo/notation checklist item) — plus `docs/reviewer_response.md`, the AIChE per-reviewer point-by-point rebuttal mapping every comment ID (R1-M1..M5, R1-m1..m7, R2-1..6) to its verbatim concern, change, manuscript location, and a real `results/*` supporting artifact. Both files pass `verify_number_provenance.py` and the read-only `.tex` is byte-untouched (D-14-18).**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-19 (worktree agent-a75d4ec29869f2d46)
- **Completed:** 2026-05-19
- **Tasks:** 2
- **Files modified:** 2 created, 0 modified

## Accomplishments

### Task 1 — PAPER-06/07/10/11 keyed reference + typo blocks
- `docs/paper_blocks_refs_methods.md` created with a header stating the Overleaf-external/read-only invariant (D-14-18) and the number-provenance contract.
- **PAPER-06:** a discrete keyed block for each reference fix — [27]→`\cite{esteban2017realvaluedmedicaltimeseries}` (RCGAN is classical, sentence rewritten), [28]→`\cite{Mugel2022}` (reassigned optimization-only), [39]→replaced with Havlíček (2019) + Schuld & Killoran (2019) `.bib` entries + `\cite{havlicek2019supervised, schuld2019quantum}`, [18]→Rasmussen & Williams GPR `.bib` + rewrite, [19]→reuse already-defined `\cite{yoon2019TimeGAN}`, [41]→rely on already-present `\cite{dimoudis2023utilizing}`, [55]-[57],[59]→removed (over-reaching claim deleted, not substituted); explicit RETAINED note for [21]-[23],[34]-[36],[61].
- **PAPER-07:** Bernal et al. "Perspectives of quantum computing for chemical engineering" `@article` `.bib` entry + a softened insertion sentence keyed to the §1.3→§1.4 transition (also satisfies R2-2's measured-quantum-jump ask).
- **PAPER-10:** A.3 section header + lead-in relabeled "Proposed Extension (Outlook) … not implemented"; a `\paragraph{Relationship to the trained objective.}` block clarifying that the trained objective is the WGAN-GP Earth-Mover form (Eq. eq:wgangp) while the A.3 Hybrid-GAN objective is the original log-GAN/JS form; Table A2 caption recaptioned as explicitly aspirational.
- **PAPER-11:** one keyed before→after block per R1-m7 checklist item (Laas→Lags, Figure A5).This→. This, LUCY ©→\textregistered, the malformed mid-sentence `\label{fig:lucy}` + 300L/20L sentence rewrite, Dry Biomass→dry biomass, bio-manufacturing→biomanufacturing, Ref[39] Approac→Approach, Ref[51] caps, QWGAN-GPs→QWGAN-GP, single return symbol $r_t$, enlarge Figs 2-6).

### Task 2 — PAPER-08/09 Methods (from JSON) + reviewer_response.md + provenance gate
- **PAPER-08:** a `\paragraph{Dataset and preprocessing.}` Methods block — 778 raw points → 777 log-returns → 384 windows, single campaign, all-train/no-split, 5 seeds — every literal annotated `% source: results/model_info.json#dataset.<field>` / `#seed_set` (11 `% source:` annotations); values are the live `model_info.json` `dataset` block + `seed_set`.
- **PAPER-09:** an `\paragraph{Evaluation scale.}` + `Table~\ref{tbl:eval_scale}` Methods block labeling EMD / DTW / moments / ACF as reported on both transformed-log-return and original-OD scale, each value the EXACT stored `fidelity_dualscale.json` quantum/Pipeline-B/seed-42 value at full precision (so the gate's substring/precision resolution passes).
- **reviewer_response.md:** per-reviewer sections (R1 Major, R1 Minor, R2) with one row per comment ID R1-M1..M5, R1-m1..m7, R2-1..6 — verbatim concern → change made → manuscript location → a real `results/*` (or `docs/*`) supporting-artifact path; all 19 cited paths verified to exist; no TODO/TBD/placeholder table cell.
- **Provenance gate:** `verify_number_provenance.py --target` PASSES for `paper_blocks_refs_methods.md` (93 distinct literals all resolve) AND `reviewer_response.md` (41 distinct literals all resolve).

## Task Commits

1. **Task 1: PAPER-06/07/10/11 keyed reference + typo LaTeX blocks** — `c957060` (feat)
2. **Task 2: PAPER-08/09 Methods blocks (from JSON) + reviewer_response.md** — `df1a44a` (feat)

## Files Created/Modified
- `docs/paper_blocks_refs_methods.md` — PAPER-06..11 copy-paste LaTeX blocks (created, ~640 lines after Task 2)
- `docs/reviewer_response.md` — AIChE per-reviewer point-by-point rebuttal (created)

## Decisions Made
- **Cite-key keying over location keying:** the `.bib` is Overleaf-external (A1) and the `.tex` is read-only (D-14-18), so every PAPER-06/07 fix is keyed to the `\cite{}` key as it appears in `main (4) copy.tex`, not a line number; already-defined keys reused where the correct reference is already present.
- **[55]-[57],[59] removed not replaced:** substituting a quantum-advantage citation would re-introduce the very overclaim R1-M5 flagged; removal is the D-14-20-consistent resolution.
- **Exact stored JSON values in PAPER-08/09:** rounded display numbers fail the number-provenance gate; render-from-JSON is enforced by using the full-precision stored value verbatim.
- **Single representative seed in PAPER-09:** the full multi-seed dual-scale table is the single source of truth in `fidelity_dualscale.json`; the Methods block labels the scale and gives a concrete representative value rather than duplicating the table (D-14-16).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Number-provenance gate false-positive on `PAPER-11.N` document-structure labels**
- **Found during:** Task 1 (running the plan's own `verify_number_provenance.py` acceptance gate).
- **Issue:** The `### PAPER-11.3` … `### PAPER-11.11` subsection headers form bare decimal tokens (`11.3`, `11.7`, `11.8`, `11.10`, `11.11`) that the gate's identifier-strip patterns do not cover (it strips `D-14-NN`, `R1-MN`, plan ids, `.py:NN`, years-before-`)`, but not `PAPER-11.N`). Five of them did not coincidentally resolve to a JSON float, so the gate raised — blocking the plan's own acceptance criterion. This is the same class as 14-03 deviation #2 (acceptance gate tripping on documentation structure, not data).
- **Fix:** Renamed every `### PAPER-11.N —` header to `### PAPER-11 / R1-m7 item N —` (and the one cross-reference) so the subsection labels are no longer bare decimal literals. No content, no fix, and no reviewer rationale changed; the gate's intent (every *data* number traces to JSON) is preserved and now independently true rather than coincidentally true.
- **Files modified:** `docs/paper_blocks_refs_methods.md`
- **Committed in:** `c957060` (Task 1 commit)

**Total deviations:** 1 auto-fixed (1 Rule-3 blocking gate-false-positive on doc structure). No scope creep — the rename only removes a numbering artifact that the data-provenance gate must not police; all reference/typo content is unchanged.

## Issues Encountered
- **`qgan_env` absent in worktree:** identical to 14-03/14-04 — `qgan_env` is gitignored and lives in the main checkout. Resolved with the established `ln -s /…/qGAN/qgan_env qgan_env` symlink (confirmed gitignored via `git check-ignore`; never committed). The provenance gate runs under this interpreter.
- **`.tex`/`.pdf` not git-tracked in the worktree:** `main (4) copy.tex`, `supp_material.tex`, `QGAN_Review_Response_Plan.md.pdf` are untracked in the worktree (they live in the main checkout). They were READ-ONLY (Read tool only); `git diff --stat -- "main (4) copy.tex" supp_material.tex` is empty, so D-14-18 is satisfied byte-exactly. The blocks were authored against the verbatim content read from the main-checkout copies.
- **`grep TODO|TBD` header false-positive:** the only `TODO`/`TBD` occurrence is the header prose describing the no-placeholder discipline ("no `TODO`/`TBD`/placeholder cells"), not a table cell. Verified: zero `TODO|TBD|placeholder|FIXME|XXX` in any `| ... |` table row. The plan's acceptance criterion (no placeholder/TODO *cells*) is satisfied.

## Known Stubs
None — PAPER-08/09 render every number from `model_info.json` / `fidelity_dualscale.json` (exact stored values, `% source:` annotated); `reviewer_response.md` has a substantive change + a real existing artifact path in every comment-ID row (19/19 cited paths verified present). No placeholder, mock, or empty-data cell. PAPER-06.h's RETAINED note is an intentional, reviewer-confirmed no-op, not a stub.

## Threat Surface Scan
No new network endpoints, auth paths, or external file-access patterns. All three plan trust boundaries are mitigated as specified: (T-14-13) JSON→Methods LaTeX block — PAPER-08/09 render from JSON with `% source:` annotations and exact stored values, `verify_number_provenance.py` PASS is a hard pass; (T-14-17) reviewer_response.md→artifact paths — every supporting-artifact cell points at a path verified to exist (19/19), no placeholder cells; (T-14-16) read-only .tex — `git diff --stat` on `main (4) copy.tex`/`supp_material.tex` is empty. No threat flags.

## Self-Check: PASSED
- `docs/paper_blocks_refs_methods.md` — FOUND (PAPER-06/07/08/09/10/11 keyed blocks; Bernal; Lags; QWGAN-GP; biomanufacturing; @article/@book .bib entries; 11 `% source:` annotations)
- `docs/reviewer_response.md` — FOUND (R1-M1..M5, R1-m1..m7, R2-1..6 rows; 19/19 supporting-artifact paths exist; no TODO/TBD table cell)
- `verify_number_provenance.py --target docs/paper_blocks_refs_methods.md` — PASS (93 distinct literals resolve)
- `verify_number_provenance.py --target docs/reviewer_response.md` — PASS (41 distinct literals resolve)
- `git diff --stat -- "main (4) copy.tex" supp_material.tex` — empty (D-14-18, .tex byte-untouched)
- Commit `c957060` — FOUND
- Commit `df1a44a` — FOUND

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-19*
