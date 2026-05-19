---
phase: 14-paper-revision-release-freeze
plan: 05
subsystem: paper-latex-blocks
tags: [latex-blocks, claim-calibration, number-provenance, read-only-tex, reviewer-response, paper-01-05, locked-deoverclaim]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze (plan 03)
    provides: "model_info.json (10-model aggregate, 55-param decomposition), reconciliation_note.md (1000ep->2000ep EMD delta), verify_number_provenance.py (the executable gate)"
  - phase: 14-paper-revision-release-freeze (plan 04)
    provides: "revision/results/figures/ suite (per-model + cross-model + headline_vs_reproduction) referenced by the PAPER-02/03 'see figure suite' framing"
  - phase: 14-paper-revision-release-freeze (plan 02)
    provides: "matched2000 45/45 accepted sweep + headline_canonical.json — the matched-budget evidence the reframed hypothesis and de-overclaim blocks are calibrated against"
  - phase: 14-paper-revision-release-freeze (plan 01)
    provides: "canonical_config_lock.json — locked iqp_sel_55 55-param decomposition driving the PAPER-03 qubit/layer rationale"
provides:
  - "revision/docs/paper_blocks_framing.md — copy-paste LaTeX blocks for PAPER-01..05 keyed to label/anchor + line citation + one-line reviewer rationale, every numeric literal JSON-traceable (verify_number_provenance.py PASS, 23 distinct literals)"
affects: [14-06, 14-07, paper-resubmission, reviewer-response]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Provenance-safe LaTeX block authoring: every quantitative literal in an AFTER block must resolve verbatim/at-precision to a revision/results/*.json value; non-resolving rounded means (e.g. reconciliation EMD averages, which live only in a .md) are NOT introduced into manuscript body — claims using them are stated qualitatively in the result direction (D-14-20)"
    - "Line citations written in the file:NNN / :NNN-NNN form so the number-provenance gate strips them as source-location identifiers, not data; \\label/anchor sentence is the primary key, line ref is secondary"
    - "Read-only .tex by construction: the Overleaf-canonical .tex files are untracked in the worktree, so git diff --stat is empty by construction (D-14-18) — blocks are copy-paste targets, never edits"

key-files:
  created:
    - revision/docs/paper_blocks_framing.md
  modified: []

key-decisions:
  - "PAPER-02 LOCKED applied unconditionally (D-14-20): the matched-2000ep sweep shows the 55-param quantum generator does NOT beat parameter-matched classical WGAN-GP baselines (reconciliation_note.md: iqp_sel_55_repro EMD ~0.155 vs wgan_cnn ~0.102 / wgan_mlp ~0.122), so every overclaim before->after block is mandatory, not contingent on result direction."
  - "PAPER-03 trainability sub-point stated qualitatively, NOT with hand-typed EMD means: the rounded ansatz EMD means (0.155376/0.156328/0.148114) live only in reconciliation_note.md and do NOT resolve against any revision/results/*.json (the gate only checks JSON). Quantitative rationale therefore uses the structural decomposition (5q/3L/55p; V1/V2/V3 75/135/75p, depth 4/8/4) — all of which resolve in model_info.json/ansatz_comparison.json — and the depth/capacity conclusion is stated in the matched-budget result direction (D-14-20)."
  - "PAPER-02a folded into PAPER-01b: 'exponentially more compactly' (main:151) is the same anchor as the quantum-necessity transition; PAPER-01b's AFTER block already removes it. Tracked explicitly as PAPER-02a so the LOCKED phrase is grep-verifiable as addressed without a conflicting double-replacement of one sentence."
  - "Apparatus constants (20L/300L/880nm/120cm) are deliberately confined to BEFORE quotations; PAPER-05c removes the contradictory '300L configuration of the 20L version' and the malformed mid-sentence \\label{fig:lucy} without introducing apparatus numbers into the resolvable numeric body."

patterns-established:
  - "Pattern: a single keyed LaTeX-blocks file gated by verify_number_provenance.py in the same plan that authors it — the manuscript copy-paste artifact ships with its executable success-criterion-5 proof, identical to the 14-03 provenance-doc pattern"

requirements-completed: [PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05]

# Metrics
duration: ~20min
completed: 2026-05-19
---

# Phase 14 Plan 05: PAPER-01..05 Keyed Framing & Claim-Calibration LaTeX Blocks Summary

**Authored `revision/docs/paper_blocks_framing.md` — copy-paste LaTeX blocks for the five claim-calibration + circuit-rationale reviewer requirements (PAPER-01 reframed parameter-parity hypothesis, PAPER-02 LOCKED de-overclaiming, PAPER-03 Circuit Design Rationale subsection, PAPER-04 bioprocess growth-rate log-return justification, PAPER-05 Outlook demotion + Table A2 caveat + 20L/300L fix), each keyed to a `\label`/anchor sentence with a one-line reviewer rationale, every numeric literal resolving to a `revision/results/*.json` value (verify_number_provenance.py PASS, 23 distinct literals) and the read-only Overleaf `.tex` untouched.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-19 (worktree agent-a40e6ce4c5fe5b85e)
- **Completed:** 2026-05-19
- **Tasks:** 2
- **Files modified:** 1 (1 created)

## Accomplishments

### Task 1 — PAPER-01/02/04/05 keyed framing + calibration blocks
- **PAPER-01:** (a) reframed central question keyed to `main:92` (Section 1.4) carrying the verbatim hypothesis "match or exceed a classical generator of equivalent parameter count"; (b) softened quantum-necessity transition keyed to `main:146-151` removing "exponentially more compactly / richer distributions with fewer parameters". R2-1/R2-2/R1-M5 rationale.
- **PAPER-02 (LOCKED, D-14-20):** before→after block-replacements for each named overclaim — "exponentially more compactly" (`main:151`, folded into PAPER-01b), "high fidelity ... industrial bioprocesses" (`main:266`), "industrial bioprocess engineering" (`main:296`), plus "computational advantages" (`main:276`) and "reduced mode collapse" softening at `supp:135`/`supp:334`/`main:174`. Each calibrated to the matched-budget non-superiority result. R1-M5 rationale.
- **PAPER-04:** finance→bioprocess rewrite of the log-return rationale keyed to `supp:352` / span `supp:358-365`, replacing "highly favored in quantitative analysis" with the growth-rate ($\mathrm{OD}\propto e^{\mu t}$, $r_t\approx\mu_t\Delta t$) interpretation and explicitly disowning the finance framing. R1-M3 rationale.
- **PAPER-05:** (a) new `\subsection*{Outlook}` keyed to `main:286` moving the decision-tree (`fig:qgan_schemcatic`, `supp:340`) and Hybrid-GAN (`fig:qgan_hybrid_appraoch`, `supp:151`) material out of the empirical-contribution claims + demotion edit at `main:261`; (b) Table A2 (`tbl:various_approaches`, `supp:226`, "Hybrid-GAN (Proposed)" row `supp:242`) caveated as aspirational with an explicit removal option; (c) 20L/300L mismatch + malformed mid-sentence `\label{fig:lucy}` fixed at `main:178` and supp caption `supp:346`. R2-3/R2-5a rationale.
- Gate PASS at this stage: 18 distinct numeric literals all resolve.

### Task 2 — PAPER-03 Circuit Design Rationale subsection + provenance gate pass
- New `\subsection{Circuit Design Rationale}` keyed to insert after `main:155` (`\subsection{QWGAN-GP Architecture Overview}`), covering all three R2-5b sub-points:
  1. **Why 5 qubits** — coupled to the length-10 window via 5 qubits × 2 Pauli observables = 10 outputs; locked 5q/3L/55-param decomposition (1 IQP-encoding param/qubit + 3 SEL rot params/qubit/layer) sourced from `canonical_config_lock.json`/`model_info.json`, NOT a hard-coded layer count.
  2. **Expressibility–trainability tradeoff** — compared against V1 (75p, depth 4, range), V2 (135p, depth 8, range), V3 (75p, depth 4, linear) over 5 seeds at matched 2000ep; deeper/larger ansatz did not improve fidelity at matched budget (stated in the matched-budget result direction per D-14-20; structural numbers resolve, no hand-typed EMD mean).
  3. **Classical critic + quantum generator** — WGAN-GP gradient-penalty cost/stability argument + clean single-component isolation vs the matched classical baselines.
- Added a number-provenance source-map footer.
- **Full-file gate PASS: `verify_number_provenance.py --target revision/docs/paper_blocks_framing.md` exits 0, 23 distinct numeric literals all resolve to `revision/results/*.json`.**

## Task Commits

1. **Task 1: PAPER-01/02/04/05 keyed framing+calibration LaTeX blocks** — `fba45e7` (feat)
2. **Task 2: PAPER-03 Circuit Design Rationale subsection + provenance gate pass** — `843f89e` (feat)

## Files Created/Modified
- `revision/docs/paper_blocks_framing.md` — copy-paste LaTeX blocks for PAPER-01..05, each keyed to label/anchor + `file:NNN` line citation + one-line R1-M/R2- reviewer rationale, JSON-source annotations + provenance footer (created; ~480 lines)

## Decisions Made
- **PAPER-02 unconditionally LOCKED (D-14-20):** matched-2000ep sweep shows the quantum entrant does not beat parameter-matched classical baselines; every overclaim before→after block is mandatory regardless of result direction.
- **PAPER-03 trainability stated qualitatively (D-14-20 + provenance contract):** rounded ansatz EMD means live only in `reconciliation_note.md` (a `.md`, not a `.json`) and the gate only checks `revision/results/*.json`, so they do NOT resolve; the rationale uses the structural decomposition (which all resolves) and states the depth/capacity conclusion in the matched-budget result direction rather than hand-typing a non-resolving number into the manuscript body.
- **PAPER-02a folded into PAPER-01b:** the "exponentially more compactly" overclaim and the quantum-necessity transition share anchor `main:151`; one AFTER block satisfies both, tracked explicitly so the LOCKED phrase is grep-verifiable.
- **Apparatus constants quarantined to BEFORE blocks:** 20L/300L/880nm/etc. are not model results and are not in any results JSON; PAPER-05c removes the contradiction and the malformed `\label` without introducing them into the resolvable numeric body.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reference `.tex` / review PDF absent in worktree (untracked in main checkout)**
- **Found during:** Task 1 (read_first step — `main (4) copy.tex`, `supp_material.tex`, `QGAN_Review_Response_Plan.md.pdf` not present in the worktree).
- **Issue:** The plan's `read_first` lists the three reference files as READ-ONLY inputs. They are **untracked** in the main checkout (visible in the start-of-conversation git status), so they do not propagate to a git worktree — identical class to the gitignored `qgan_env`/`best_checkpoint.pt` resolved by 14-01/02/03/04.
- **Fix:** Read the `.tex` anchors directly from the main checkout (`/Users/shawngibford/dev/phd/qGAN/main (4) copy.tex` / `supp_material.tex`) as the READ-ONLY Overleaf-canonical reference. This is the *intended* D-14-18 posture: the `.tex` is external to the repo and must never be edited; the worktree `git diff --stat` is empty by construction, so the T-14-16 mitigation holds trivially. The reviewer-memo sub-points (R1-M3/M5, R2-1/2/3/5a/5b) were taken from the RESEARCH "Phase Requirements" table + `14-RESEARCH.md` decisions (the PDF's verbatim itemisation is mirrored there) — the PDF binary itself was not required to author the one-line rationales.
- **Files modified:** none (read-only access only).
- **Committed in:** n/a (no functional change; the .tex/PDF are not repo artifacts).

**2. [Rule 3 - Blocking] `qgan_env` absent in worktree (gitignored, lives in main checkout)**
- **Found during:** Task 1 (the plan's verify command invokes `./qgan_env/bin/python`).
- **Issue:** Identical to the 14-01/02/03/04 precedent — `qgan_env` is gitignored and lives in the main checkout.
- **Fix:** `ln -sfn /Users/shawngibford/dev/phd/qGAN/qgan_env qgan_env` (already covered by `.gitignore`, never committed); the verify command then runs unchanged.
- **Files modified:** none (gitignored symlink).
- **Committed in:** n/a.

**Total deviations:** 2 auto-fixed (both Rule-3 worktree-resource blockers, both resolved by the established 14-01..04 main-checkout-resolution precedent). No scope creep — no `.tex` edited, no number hand-typed, no plan task altered.

## Issues Encountered
- **Rounded reconciliation EMD means do not resolve against JSON:** the per-model 1000ep→2000ep EMD averages (e.g. `0.154999`, `0.155376`, `0.121527`, `0.101747`) are computed in `revision/docs/reconciliation_note.md` and are **not stored verbatim in any `revision/results/*.json`**; the number-provenance gate only scans `revision/results/*.json`. Writing them into a manuscript block would FAIL the gate (success-criterion 5 / Pitfall 5 / T-14-13). Resolved by design (see Decisions): quantitative claims use the structural decomposition (all of which resolves) and the EMD comparison is stated qualitatively in the matched-budget result direction (D-14-20) — the honest, provenance-safe framing the reviewers asked for.
- **Pre-existing env-only test failure (out-of-scope, NOT re-logged):** the Phase-10 `samples.npy`-missing `test_utility.py` failure recorded by 14-01 in `deferred-items.md` is unrelated to this docs-only plan (no test run, no code path touched). No fix attempted (correct per scope-boundary rule).

## Next Phase Readiness
- **Claim-calibration + circuit-rationale core delivered:** PAPER-01..05 are copy-paste LaTeX blocks keyed to `\label`/anchor + line citation + reviewer rationale, every number JSON-traceable, gate-PASS. Downstream 14-06/14-07 (remaining PAPER requirements, reviewer-response doc, release freeze) can cite/extend this file directly; it is gated by the same `verify_number_provenance.py` used in 14-03.
- **D-14-18 invariant intact:** no `.tex` edited (untracked in worktree → empty diff by construction); blocks are external Overleaf copy-paste targets.
- **D-14-20 honoured:** PAPER-02 de-overclaiming applied unconditionally; the matched-budget non-superiority result is stated honestly throughout (PAPER-01b/02b/02c/02d/03).
- **No blockers.** Pure documentation authoring — no training, no re-run; the matched2000 sweep + headline + figure suite remain byte-frozen (14-01..04 invariants intact).

## Known Stubs
None — every numeric literal in `paper_blocks_framing.md` resolves to a real `revision/results/*.json` artifact (gate PASS, 23 distinct literals); no placeholder text, no TODO/FIXME, no unsourced number. The qualitative trainability framing is an intentional, documented provenance-safety + D-14-20 decision, not a stub (the structural numbers backing it all resolve).

## Threat Surface Scan
No new network endpoints, auth paths, or external file-access patterns. Both plan trust boundaries are mitigated as specified:
- **2000ep JSON → LaTeX block (T-14-13):** every numeric literal annotated with its JSON source; `verify_number_provenance.py` explicit-raise gate is a hard pass criterion — full-file exit 0, 23 literals resolved. Non-resolving rounded means were deliberately kept out of the manuscript body.
- **revision package → in-repo .tex (T-14-16):** the `.tex` files are untracked in the worktree; `git diff --stat -- "main (4) copy.tex" supp_material.tex` is empty by construction — no accidental edit possible (D-14-18).
- **T-14-15 (overclaim reintroduction):** PAPER-02 LOCKED before→after blocks present for each named overclaim phrase, grep-verified.
- **T-14-SC (package installs):** none in this plan (documented no-op).
No threat flags.

## Self-Check: PASSED
- `revision/docs/paper_blocks_framing.md` — FOUND (PAPER-01..05 keyed blocks, `\subsection{Circuit Design Rationale}`, fig:lucy/fig:qgan_hybrid_appraoch/fig:qgan_schemcatic/tbl:various_approaches present, verbatim hypothesis present, R1-M/R2- rationale tags present)
- `verify_number_provenance.py --target revision/docs/paper_blocks_framing.md` — PASS, exit 0, 23 distinct literals all resolve
- `git diff --stat -- "main (4) copy.tex" supp_material.tex` — empty (TEX_UNTOUCHED, D-14-18)
- Commit `fba45e7` (Task 1) — FOUND
- Commit `843f89e` (Task 2) — FOUND

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-19*
