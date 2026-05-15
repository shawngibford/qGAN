---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: AIChE Major Revision Response
status: executing
stopped_at: Phase 09.1 plan 02 complete — Wave 3 GO
last_updated: "2026-05-15T19:18:51.318Z"
last_activity: 2026-05-15
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 14
  completed_plans: 12
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-23)

**Core value:** The qGAN must generate synthetic OD time series that capture real data's volatility structure
**Current focus:** Phase 09.1 — r1-m3-preprocessing-ablation

## Current Position

Phase: 09.1 (r1-m3-preprocessing-ablation) — EXECUTING
Plan: 3 of 4 (next: 09.1-03 multi-seed sweep)
Status: Wave 2 smoke passed (structural_pass=true); Wave 3 GO
Last activity: 2026-05-15 -- Plan 09.1-02 complete

Progress: [█████████░] 86%

## Performance Metrics

**Velocity:**

- Total plans completed: 24 (v1.0 + v1.1 + v2.0 phase 8 + v2.0 phase 9 + v2.0 phase 09.1 partial)
- v2.0 plans: 12 completed (Phase 8: 5, Phase 9: 5, Phase 09.1: 2 of 4)

**Past milestones:**

| Milestone | Phases | Status |
|-----------|--------|--------|
| v1.0 Code Review Remediation | 3 (phases 1-3) | Shipped 2026-03-07 |
| v1.1 Post-HPO Improvements | 4 (phases 4-7) | Shipped 2026-03-23 |

**Plan execution log (v2.0):**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 09.1 P02 (cli-driver + smoke) | ~10 min | 2 | 18 |

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

### Pending Todos

None.

### Blockers/Concerns

- Variance collapse (fake std 48% of real) persists from v1.1 — v2.0 will NOT attempt to fully close this gap; instead it contextualizes honestly against matched classical baselines (reviewer-aligned strategy)
- Multi-seed × multi-ansatz × multi-baseline compute on local Mac only — mitigated by splitting sensitivity (Phase 12) and architecture (Phase 13) into separate phases instead of one monster sweep

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

**Last Session:** 2026-05-15T19:18:47.053Z
**Stopped At:** Plan 09.1-02 complete — Wave 3 GO
**Resume File:** .planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-03-PLAN.md

**Planned Phase:** 8 (Core Module Extraction) — 5 plans — 2026-04-23T16:46:36.017Z
