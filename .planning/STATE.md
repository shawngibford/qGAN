---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: AIChE Major Revision Response
status: executing
stopped_at: 2026-06-01 -- v1.2.4 loss-diagnostics patch pending tag at HEAD 9d0d15f (2 commits ahead of v1.2.3). Added Main Fig 6 training_convergence_all_models (matched-budget OD-EMD-vs-epoch across 7 WGAN-GP models) and Supp §A.9 Per-Model Training Loss Diagnostics (8-panel grid covering 4 quantum + 3 classical WGAN-GP + 1 VAE) with 4 paragraphs of commentary identifying three qualitatively distinct training regimes (WGAN-GP critic stability across cohort; WGAN-CNN drift instability vs quantum stability; VAE regularization collapse). Main 5→6 figures; supp 14→15 figures (the new one contains 8 sub-panels). All gates PASS (149 main + 198 supp literals; 60-page compile clean; 0 errors; freeze-ready gates a/b/c PASS; gate d release.md expected-deferred to 14-07).
last_updated: "2026-06-01T18:00:00.000Z"
last_activity: 2026-06-01 -- Loss-diagnostics expansion: 1 main convergence figure + 1 supp 8-panel grid + commentary. Two atomic commits (f4a6565 Fig 6 + 9d0d15f Supp §A.9). Commentary surfaces three substantive training observations: (1) WGAN-GP critic stability across all 35 runs (no NaN/divergence); (2) WGAN-CNN critic loss drifts back toward zero over training — quantum-side stability advantage not previously surfaced visually; (3) VAE ELBO+recon snap to constant ~50 eval steps, KLD collapses to zero (training-side signature of the degenerate generation regime in main §4.1). AR(2) closed-form fit noted with σ² = 0.057. Page count: 59 → 60 (+1). Provenance: 147→149 main, 183→198 supp. Pending: tag v1.2.4 + push.
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 25
  completed_plans: 24
  percent: 53
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-23)

**Core value:** The qGAN must generate synthetic OD time series that capture real data's volatility structure
**Current focus:** Phase 14 — paper-revision-release-freeze (v1.2.4 loss-diagnostics patch pending tag at HEAD `9d0d15f`; v1.2.3 + v1.2.2 + v1.2.1 + v1.2 on origin; AIChE upload pending)

## Current Position

Phase: 14 (paper-revision-release-freeze) — PARTIAL (19/20; 14-07 deferred to journal acceptance)
Plan: 19 of 20 complete. **v1.2.4 loss-diagnostics patch** pending tag at HEAD `9d0d15f` (2026-06-01); v1.2.3 (7bf3f2c), v1.2.2 (e89fd04), v1.2.1 (3f4c2ef), and v1.2 (34eb34e) stay on origin as historical reference points; main is 2 commits ahead of origin/main pending push + tag.
Status: Paper submission-ready with **0 LaTeX errors, 60 pages, 149 main + 198 supp literals trace to JSON**, freeze gates a/b/c PASS, gate d expected-deferred. v1.2.4 added the matched-budget training-convergence headline figure (main Fig 6) plus the per-model training-loss diagnostic grid (supp §A.9, 8 sub-panels + commentary). Now totals: Main 6 figures, Supp 15 figures (one of which is an 8-panel grid). The new loss-diagnostics subsection surfaces a previously visual-only observation: WGAN-CNN training instability vs quantum-cluster stability across all 35 runs. AIChE rebuttal letter + revised manuscript + supp + bib + figures + ama.bst — all assembled and verified. The only remaining external items are (i) AIChE-portal upload (rebuild bundle from v1.2.4 artifacts; updated figure count + sub-panel layout), (ii) GitHub release notes at https://github.com/shawngibford/qGAN/releases/new?tag=v1.2.4, and (iii) plan 14-07 Zenodo DOI mint, which is intentionally deferred to journal acceptance (rebuttal cites ZENODO-DOI-PLACEHOLDER).
Last activity: 2026-05-28 -- Post-swarm audit-cleanup session: 4-parallel-audit-agent pre-tag sweep + 8 cleanup commits + handoff doc update + tag v1.2 + push. Caught and fixed a Lambert W misdescription regression the swarm missed (Methods §3.2 prose described Pipeline B as containing inverse Lambert W when actually D-10-05 dropped that path; the provenance gate validates literals not prose, so the regression slipped past). Added 4 new tables/figures, removed 5 stale single-model figures, bundled 7 legacy figs into repo, cleared 10 audit FLAGs. See .planning/PAPER-SUBMISSION-HANDOFF.md §1A for the full audit-cleanup change log.

Progress: [██████████] 96%

## Performance Metrics

**Velocity:**

- Total plans completed: 48 (v1.0 + v1.1 + v2.0 phase 8 + v2.0 phase 9 + v2.0 phase 09.1 partial)
- v2.0 plans: 13 completed (Phase 8: 5, Phase 9: 5, Phase 09.1: 3 of 4)

**Past milestones:**

| Milestone | Phases | Status |
|-----------|--------|--------|
| v1.0 Code Review Remediation | 3 (phases 1-3) | Shipped 2026-03-07 |
| v1.1 Post-HPO Improvements | 4 (phases 4-7) | Shipped 2026-03-23 |

**Plan execution log (v2.0):**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 09.1 P02 (cli-driver + smoke) | ~10 min | 2 | 18 |
| Phase 09.1 P03 (multi-seed sweep resumable) | 29.1 min (sweep wall @ parallel=2) | 3 | 76 (1 driver script + 75 artifacts) |
| Phase 14 P17 | 18min | 3 tasks | 2 files |
| Phase 14 P18 | ~15 min | 3 tasks | 3 files |
| Phase 14 P19 | ~20 min | 3 tasks | 105 files |

## Accumulated Context

### Roadmap Evolution

- Phase 09.1 inserted after Phase 9: R1-M3 Preprocessing Ablation — empirical 3-pipeline preprocessing comparison (raw OD vs log-returns vs log-returns+Lambert W) for reviewer rebuttal; spec at .planning/scratch/09.1-r1-m3-ablation-spec.md (URGENT)

### Decisions

See PROJECT.md Key Decisions table for full log.

v2.0 roadmap decisions:

- **Phase 8:** INFRA-01 + INFRA-02 isolated as foundational phase — every downstream phase imports from `revision/core/`, so extraction + parity check must land first with no other reqs bundled
- **Phase 9:** DOC-01, DOC-02, EVAL-06 grouped as "Documentation Bridge" — cheap, paper-ready numbers front-loaded so Phase 14 paper drafting can begin in parallel with expensive experiments
- **Phases 10-13 sequencing:** Baselines → Utility Eval → Sensitivity → Architecture, so each phase consumes artifacts from the prior one; compute budget (local Mac statevector) respected by separating sensitivity sweeps from architecture sweeps
- **Phase 14:** All PAPER-* requirements plus INFRA-03 (Zenodo freeze) bundled — paper revision reads JSON from all upstream phases and the tag/DOI freeze is the final wrap-up step

v1.1 highlights retained:

- HPO-tuned values: N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05
- backprop replaces parameter-shift due to PennyLane #4462 broadcasting gradient bugs
- **Phase 09.1 P02:** Redefined Pipeline C smoke parity gate from 2% absolute-EMD to structural-evidence gate (Rule 4 deviation, user-approved). Rationale: the 0.12048789 baseline was measured at 2000-epoch convergence — comparing it to a 100-epoch fresh-init run is physically incoherent. Pipeline C code-path identity is independently proven by 09.1-01's ABL-01 round-trip (max_abs_err 4.44e-16).
- **Phase 09.1 P03:** Archived (not deleted) wave-2 100-epoch smoke artifacts at A/42, B/42, C/42 before launching the 1000-epoch sweep — `is_complete()` checks file presence not epoch count, so silent mixed-budget contamination was the alternative. Sweep then ran clean: 15/15 in 29.1 min @ parallel=2 (vs plan estimate ~24 h; v2.0 PennyLane backprop path is ~50× faster per epoch than v1.1). The conditional +2 seeds gate (D-09.1-06) is deferred to plan 04 (analysis-time decision based on observed inter-seed spread).
- [Phase ?]: Plan 14-17: Manuscript .tex revised directly (D-14-18 READ-ONLY superseded by r4 BLOCK verdict); all PAPER-01..11 blocks integrated, abstract de-overclaimed, stale DTW 0.6843 reconciled to matched-budget ~0.30
- [Phase ?]: Plan 14-17: Zenodo DOI uses symbolic placeholder token ZENODO-DOI-PLACEHOLDER (digit-free) so verify_number_provenance.py passes over the .tex; 14-07 mints the real DOI into it
- [Phase ?]: Plan 14-18: OD-EMD claim recalibrated from 'statistically equivalent' to 'no statistically detectable difference at n=5 (underpowered)'; LR-DTW uniform-dominance posture stated, OD-DTW Orlandi improvement reframed matched-budget-wide
- [Phase ?]: D-14-19: drift reverted to HEAD; freeze candidate is committed HEAD 6518323; verify_freeze_ready.py hardened to certify the committed tree

### Pending Todos

None.

### Blockers/Concerns

- Variance collapse (fake std 48% of real) persists from v1.1 — v2.0 will NOT attempt to fully close this gap; instead it contextualizes honestly against matched classical baselines (reviewer-aligned strategy)
- Multi-seed × multi-ansatz × multi-baseline compute on local Mac only — mitigated by splitting sensitivity (Phase 12) and architecture (Phase 13) into separate phases instead of one monster sweep

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Phase 14 | 14-07 (tag v2.0-revision + Zenodo manual deposit + release.md + DOI wire-in) | deferred to acceptance — first-round revision resubmits with ZENODO-DOI-PLACEHOLDER + new freeze-candidate SHA `3c8502c` in Data Availability (supersedes pre-14-20 SHA `6518323` after 14-20 closed the R1-M2 utility-coverage gap); mint real DOI when AIChE accepts and update camera-ready then | 2026-05-24 |

## Session Continuity

**Last Session:** 2026-06-11T00:00:00.000Z
**Stopped At:** Phase 14 sub-plan 14-21 COMPLETE. ×0.1 WGAN inverse-pipeline bug fixed (inference-only x10 correction at 9 samples.npy load sites across 7 producer files via new shared helper `revision/_wgan_unscale.py`); 8 matched-budget JSONs regenerated; ~200 figure triples regenerated; main.tex + supp_material.tex + reviewer_response.md updated per user R3 decision (Branch B narrative rewrite — quantum cluster dominates 4 of 4 matched-budget metrics). T05 human checkpoint resolved at `.continue-here-t05.md`. CONTEXT-HANDOFF §6 #2 amended (prior LR-EMD prohibition was an artifact of the buggy data). All freeze gates a/b/c PASS; gate d expected-deferred to 14-07 (Zenodo). 14-21-SUMMARY.md committed. Phase 14 progress: 20/21 plans complete (14-07 remains deferred to journal acceptance).
**Resume File:** None — 14-21 closed cleanly. Next session can route freely.

**Phase 14 status:** PARTIAL (20/21 complete; 14-07 deferred). The only remaining intra-phase work is 14-07 Zenodo DOI mint + tag v1.2.5 (or higher) + release.md authoring, which is gated on AIChE acceptance per long-standing user deferral.

**AIChE resubmission state:** Camera-ready post-14-21 bundle is technically clean (provenance + freeze gates a/b/c PASS, LaTeX compile 0 errors/0 undefined refs, 97 pages). The bifurcated-finding narrative shifted dramatically vs. v1.2.4 — three of four matched-budget metrics now favor quantum cluster + OD-EMD H2 parametric-equivalence claim from 14-18 has been DROPPED in favor of significant cluster-dominance reading. The bug-disclosure paragraph in supp §A.7 makes the fix reproducible and the prior framing artifact transparent. Resubmission deadline 2026-06-17 (6 days from today).
