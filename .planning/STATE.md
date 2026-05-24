---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: AIChE Major Revision Response
status: executing
stopped_at: Phase 14 plan 14-20 complete — matched-budget utility battery re-run on Pipeline B; R1-M2 doc rewrite + provenance gate PASS across all paper-facing docs; freeze candidate refreshed at 3c8502c; 14-07 remains deferred
last_updated: "2026-05-24T12:00:00.000Z"
last_activity: 2026-05-24 -- Phase 14 plan 14-20 closed: re-ran utility battery (TSTR + augmentation + predictive/discriminative) against matched2000/runs/ artefacts at 2000 epochs, Pipeline B only; emitted *_matched2000.json sibling JSONs (legacy 1000-epoch JSONs preserved as provenance, not cited); R1-M2 in reviewer_response.md + methods_full.md §3.y + completeness_sweep_manifest.md updated against matched-budget numbers; provenance gate PASS on all 7 paper-facing docs; freeze candidate refreshed at 3c8502c (supersedes 6518323); 14-07 still deferred to acceptance
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 24
  completed_plans: 23
  percent: 51
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-23)

**Core value:** The qGAN must generate synthetic OD time series that capture real data's volatility structure
**Current focus:** Phase 14 — paper-revision-release-freeze

## Current Position

Phase: 14 (paper-revision-release-freeze) — PARTIAL (19/20; 14-07 deferred)
Plan: 19 of 20 complete (14-20 closed the R1-M2 utility-coverage gap surfaced during rebuttal drafting)
Status: 14-20 closed the R1-M2 regime mismatch (utility battery re-run on matched2000 artefacts at 2000 epochs, Pipeline B only). R1-M1 and R1-M2 now share a single matched-budget evidence base. 14-07 (tag + Zenodo deposit) remains deferred to acceptance — first-round revision resubmits with the ZENODO-DOI-PLACEHOLDER + the new freeze-candidate SHA `3c8502c` in Data Availability; real DOI minted at acceptance.
Last activity: 2026-05-24 -- 14-20 landed (matched-budget utility battery + figure + R1-M2 rewrite + methods + manifest); freeze candidate refreshed at commit 3c8502c (hardened verify_freeze_ready.py passes every gate except release.md, which is 14-07's deliverable); legacy 1000-epoch utility JSONs preserved on disk for provenance, not cited in rebuttal

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

**Last Session:** 2026-05-22T06:36:21.678Z
**Stopped At:** Completed 14-18-PLAN.md
**Resume File:** None

**Planned Phase:** 8 (Core Module Extraction) — 5 plans — 2026-04-23T16:46:36.017Z
