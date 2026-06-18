---
phase: 14-paper-revision-release-freeze
plan: 17
subsystem: paper
tags: [latex, manuscript, peer-review, de-overclaiming, number-provenance, zenodo-doi]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze
    provides: paper_blocks_framing.md / paper_blocks_refs_methods.md PAPER-01..11 LaTeX blocks, reconciliation_note.md matched-budget DTW, methods_full.md DTW historical framing
provides:
  - Revised `main (4) copy.tex` with all PAPER-01..11 framing/de-overclaiming/methods blocks integrated
  - Revised `supp_material.tex` with PAPER-04/05b/06g/10 blocks, r_t notation, 20L LUCY caption
  - De-overclaimed abstract (no "high fidelity" / "strong performance")
  - Stale DTW 0.6843 headline reconciled to matched-budget OD-DTW ~0.30
  - Zenodo DOI placeholder (ZENODO-DOI-PLACEHOLDER) for plan 14-07 to mint into
affects: [14-07-zenodo-freeze, paper-resubmission]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Number-provenance gate run over .tex targets (not just .md) — closes WARNING 5"
    - "Symbolic DOI placeholder string (ZENODO-DOI-PLACEHOLDER) as 14-07 mint destination"

key-files:
  created:
    - ".planning/phases/14-paper-revision-release-freeze/14-17-SUMMARY.md"
  modified:
    - "main (4) copy.tex"
    - "supp_material.tex"

key-decisions:
  - "DOI placeholder uses a fully symbolic token (ZENODO-DOI-PLACEHOLDER) rather than the literal 10.5281/zenodo.XXXXXX form — the literal DOI-prefix 10.5281 is not a JSON-resolvable result number and would fail verify_number_provenance.py over the .tex; symbolic placeholder keeps the gate green and is unambiguous for 14-07's INFRA-03 substitution"
  - "Stale 45-param circuit count corrected to the locked 55 (canonical_config_lock.json param_count) to remove an internal contradiction introduced by the PAPER-03 block (Rule 1/2 deviation)"
  - "0.6843 retained once in each .tex file but explicitly labelled a 'frozen pre-v1.0 best-case checkpoint' with the matched-budget ~0.30 stated alongside — satisfies the no-bare-0.6843 contract via the plan's option (b)"

patterns-established:
  - "Pattern 1: anchor-by-label/section-heading then apply AFTER-block intent — never blind-match BEFORE text, since the .tex on disk is older than the framing doc's stated baseline"
  - "Pattern 2: .tex numeric literals are provenance-gated as first-class targets, not gate-exempt by silence"

requirements-completed: [PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10, PAPER-11]

# Metrics
duration: 18min
completed: 2026-05-22
---

# Phase 14 Plan 17: Manuscript Revision Integration Summary

**Integrated all PAPER-01..11 r1/r2/r3 revision blocks into `main (4) copy.tex` and `supp_material.tex`, de-overclaimed the abstract, reconciled the stale DTW 0.6843 headline to the matched-budget ~0.30 evidence, and added a Zenodo DOI placeholder — making the manuscript consistent with the certified JSON evidence base before the 14-07 freeze.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-22T05:00:00Z (approx)
- **Completed:** 2026-05-22T05:18:00Z (approx)
- **Tasks:** 3
- **Files modified:** 2 (`main (4) copy.tex`, `supp_material.tex`)

## Accomplishments
- Applied every PAPER-01..05 framing + LOCKED (D-14-20) de-overclaiming block: falsifiable central question, softened quantum-necessity transition, Key Contributions / Theoretical Implications / Concluding Remarks reframes, new Circuit Design Rationale and Outlook subsections
- De-overclaimed the abstract — removed "strong performance" and "high fidelity" superlatives, reframed to a scoped matched-budget proof-of-concept statement under the 150-word register
- Applied PAPER-06..10 reference surgery and Methods blocks: GPR/TimeGAN/quantum-kernel citation fixes, Bernal et al. AIChE perspective, Dataset-and-preprocessing paragraph, Evaluation-scale table, Hybrid-GAN "Proposed Extension" relabel
- Resolved the stale headline DTW 0.6843 at all 3 occurrences (one removed via PAPER-02b, two labelled as a frozen pre-v1.0 checkpoint with matched-budget ~0.30 stated alongside) and re-anchored the Orlandi ~6.5x comparison to the matched-budget value
- Added a Zenodo DOI placeholder string in both Data Availability sections for plan 14-07
- Fixed PAPER-11 notation/typo/LUCY defects: unified return notation on `r_t`, removed the malformed mid-sentence `\label{fig:lucy}`, fixed the 20L/300L mismatch, corrected the LUCY caption to `\textregistered`/20L
- Ran `verify_number_provenance.py` over both `.tex` files: PASS (45 distinct literals in main, 19 in supp — all resolve to `results/*.json`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Integrate PAPER-01/02/03/04/05 framing + de-overclaiming, de-overclaim the abstract** - `c06acc2` (feat)
2. **Task 2: Integrate PAPER-06/07/08/09/10 references + methods blocks** - `e7e6329` (feat)
3. **Task 3: Resolve stale DTW 0.6843, Zenodo DOI placeholder, PAPER-11 defects, provenance-check** - `f644b9e` (feat)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified
- `main (4) copy.tex` - Revised main manuscript: PAPER-01..03/05a/06..09/11 blocks, de-overclaimed abstract, Circuit Design Rationale + Outlook subsections, Dataset-and-preprocessing + Evaluation-scale Methods material, matched-budget DTW headline, Zenodo DOI placeholder, r_t notation, fixed LUCY label
- `supp_material.tex` - Revised supplementary: PAPER-04 growth-rate log-return justification, PAPER-05b/10c Table A2 aspirational caveat, PAPER-06g VQE/QAOA ref removal, PAPER-10a/b Hybrid-GAN "Proposed Extension" relabel + log-GAN-vs-Wasserstein clarification, labelled 0.6843 historical reference, 20L LUCY caption, 55-param decomposition, Zenodo DOI placeholder
- `.planning/phases/14-paper-revision-release-freeze/14-17-SUMMARY.md` - This summary

## Decisions Made
- **Symbolic DOI placeholder.** The plan suggested the literal `10.5281/zenodo.XXXXXX` placeholder string. The literal DOI-namespace prefix `10.5281` is not a JSON-resolvable result number; under the Task 3 number-provenance gate over the `.tex`, `10.5281` was the single unresolved literal. The plan's Task 3 contract explicitly requires either fixing the offending literal or recording an exemption. The cleaner resolution chosen here is a fully symbolic placeholder token `ZENODO-DOI-PLACEHOLDER` (no embedded digits) — this keeps the provenance gate green over the `.tex` and remains an unambiguous, greppable substitution target for plan 14-07's INFRA-03 block. The real Zenodo DOI (10.5281/zenodo.<id>) is minted only by 14-07.
- **0.6843 retention with explicit labelling.** Per the plan's option (b), 0.6843 survives once in each `.tex` file, but only as an explicitly labelled "frozen pre-v1.0 best-case checkpoint" with the matched-budget OD-DTW ~0.30 stated in the same sentence and a pointer to the reconciliation note. No bare unlabelled 0.6843 survives. The Orlandi 1.954 comparison is re-anchored to the matched-budget ~0.30 (~6.5x improvement per `methods_full.md:451-457`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected stale 45-parameter circuit count to the locked 55**
- **Found during:** Task 3 (provenance-check + internal-consistency review)
- **Issue:** The PAPER-03 block (Task 1) introduces the locked 55-parameter circuit (`canonical_config_lock.json` `param_count`=55), but `main (4) copy.tex` §3.1 still read "contains 45 trainable parameters" and `supp_material.tex` §A.6 still read "Total parameters: 45 (5 ... 30 ... 10 ...)". This created an internal contradiction within the same manuscript — the new Circuit Design Rationale states 55 while two other locations state 45.
- **Fix:** Updated the main-text count to "55 trainable parameters" and the supplementary decomposition to "55 (5 for IQP encoding, 45 for the three strongly entangling layers, 5 for measurement-preparation rotations)", consistent with `canonical_config_lock.json` (`num_qubits`=5, `num_layers`=3, `param_count`=55).
- **Files modified:** `main (4) copy.tex`, `supp_material.tex`
- **Verification:** `verify_number_provenance.py` PASSes — the literal 55 resolves to `canonical_config_lock.json`/`model_info.json`; no remaining 45 in a parameter-count context.
- **Committed in:** `f644b9e` (Task 3 commit)

**2. [Rule 3 - Blocking] Symbolic DOI placeholder to keep the provenance gate green**
- **Found during:** Task 3 (running `verify_number_provenance.py` over the `.tex`)
- **Issue:** The literal placeholder `10.5281/zenodo.XXXXXX` introduced `10.5281` as a numeric literal that does not resolve to any `results/*.json` value; the gate flagged it (the only unresolved literal in either file).
- **Fix:** Replaced the literal-prefix placeholder with a fully symbolic token `ZENODO-DOI-PLACEHOLDER` in both Data Availability sections. The token is digit-free, gate-clean, and a clear substitution anchor for plan 14-07.
- **Files modified:** `main (4) copy.tex`, `supp_material.tex`
- **Verification:** `verify_number_provenance.py --target "main (4) copy.tex"` and `--target supp_material.tex` both exit 0 / PASS.
- **Committed in:** `f644b9e` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both auto-fixes are correctness-required — the 45→55 fix removes an internal manuscript contradiction that would otherwise be frozen into a DOI'd paper; the symbolic DOI placeholder is required for the Task 3 provenance gate to pass over the `.tex`. No scope creep — both edits are within the plan's stated Task 3 mandate.

## Number-Provenance Gate Result (Task 3, WARNING 5 resolution)

`verify_number_provenance.py` **accepts a `.tex` `--target`** (its `--help` documents "regenerated doc or paper LaTeX-blocks file"). It was run over both manuscript files after all Task 1-3 edits:

- `main (4) copy.tex`: **PASS** — 45 distinct numeric literals all resolve to `results/*.json` (schema v2.1).
- `supp_material.tex`: **PASS** — 19 distinct numeric literals all resolve to `results/*.json` (schema v2.1).

The `.tex` numbers are therefore NOT gate-exempt by silence. The numeric literals introduced by Tasks 1-3 resolve as follows:
- PAPER-03 structural numbers (`5` qubits, `3` layers, `55` params, `2` observables, `10` window, `75`/`135` ansatz params, `4`/`8` depths, `2000` epochs) → `canonical_config_lock.json` / `model_info.json` / `ansatz_comparison.json`.
- PAPER-08 dataset numbers (`778`, `777`, `384`, `10`, `2`, `42`-`46`) → `model_info.json#dataset` + `seed_set`.
- PAPER-09 evaluation-scale table values (e.g. `0.1209437521974767`, `0.9343404967853801`) → `fidelity_dualscale.json` (quantum, Pipeline B, seed 42), full stored precision, not rounded.
- Matched-budget DTW `0.30` and the labelled historical `0.6843` → resolve via `matched2000_dualscale.json#aggregates` / `fidelity_dualscale.json`.

## Issues Encountered
- The Task 3 `<verify>` block's `! grep -qi "Dry Biomass"` clause is case-insensitive and therefore also matches the correctly-lowercased "dry biomass". The acceptance criterion (the capitalized proper-noun form "Dry Biomass" is gone) is met — confirmed with a case-sensitive `grep -c 'Dry Biomass'` returning 0. The over-broad `-i` flag in the plan's verification command is a plan-authoring artifact, not a manuscript defect.

## Overleaf-Side Follow-up (not blocking the .tex integration)
The `.bib` file (`bib.bib`) is not in the repo working tree. The following `.bib`-entry additions from PAPER-06/07/11 are recorded as Overleaf-side actions and do NOT block this plan or 14-07:
- Add `@article{esteban2017realvaluedmedicaltimeseries}` verification (classical RCGAN), `@article{havlicek2019supervised}`, `@article{schuld2019quantum}`, `@book{rasmussen2006gaussian}`, `@article{bernal2022perspectives}`.
- PAPER-11 item 7/8: `.bib` title typo "Approac"→"Approach" and [51] title-capitalization standardization.
- PAPER-11 item 1/11: regenerated high-DPI ACF/result figures (`results/figures/*.pdf`) replace the Overleaf-embedded bitmaps; the "Laas"→"Lags" x-axis label and Figure 2-6 enlargement are figure-asset swaps, not `.tex` text edits.

## Next Phase Readiness
- The manuscript `.tex` files now carry the complete r1/r2/r3 revision and are consistent with the certified JSON evidence base.
- SYNTHESIS findings C2, C3, H2, H3, H4, M6, M7, M8, M9 are closed in the `.tex`.
- Plan 14-07 (Zenodo freeze / INFRA-03) can now run: it freezes the tagged tree and mints the real DOI into the `ZENODO-DOI-PLACEHOLDER` token in both files.
- No blockers introduced by this plan.

## Self-Check: PASSED

- `14-17-SUMMARY.md` — FOUND
- `main (4) copy.tex` — FOUND
- `supp_material.tex` — FOUND
- Commit `c06acc2` (Task 1) — FOUND
- Commit `e7e6329` (Task 2) — FOUND
- Commit `f644b9e` (Task 3) — FOUND

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-22*
