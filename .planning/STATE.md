---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: AIChE Major Revision Response
status: executing
stopped_at: 2026-06-12 -- post-14-21 peer-review remediation cycle COMPLETE; ama.bst-restore fix follow-up applied. 22 atomic commits between f85fd38 and 20c38ad closed the peer-review-swarm BLOCKING items + 19 polish items + co-author 6-comment thread. A subsequent follow-up commit restored ama.bst to the repo root (it was never git-tracked, so the post-14-21 compile silently rendered all 60 in-text citations as [?] and dropped the References section) and rebuilt the PDF (99 → 105 pages with References restored). Bundle ready at ~/Desktop/aiche_upload_post14-21/ (65 files, 7.9 MB). All gates PASS (157 main + 347 supp literals; 105-page compile clean; 0 errors; 0 undefined refs/cites; freeze gates a/b/c PASS; gate d release.md expected-deferred to 14-07). NOTE: this remediation cycle landed without a formal 14-22 plan artifact — work was driven by peer-review reports and co-author comments rather than a PLAN.md scaffold. ROADMAP/STATE still show 20/21 plans complete since the milestone counter wasn't updated.
last_updated: "2026-06-12T00:12:32.258Z"
last_activity: 2026-06-11/12 -- Peer-review remediation cycle (22 commits, ~6h wall clock) + ama.bst-restore follow-up. Wave 1 closed PR-4 BLOCKING-1/2/3 (stale supp Welch tables + Abstract three-of-four contradiction + rebuttal narrative drift). Wave 2 swapped refs [42]+[28] + added barren-plateau citations + normalized citation style. Wave 3 stripped caption file-paths + promoted Figs 2-6 to full-width + restructured supp Fig A11 + expanded Data Availability. Wave 4 trimmed Abstract/Contributions/§5 numerics + added log-returns one-liner + rewrote PLS to ≤250 chars. Wave 5 added §A.7 R1/R5 disclosure + OD-EMD inversion footnote + preemptive "convenient-inversion" §5 subsection + per-seed sensitivity tables + Bonferroni footnote. Page count: 96 → 99 (+3 prose) → 105 (+6 References restored by ama.bst track on 2026-06-12). Provenance: 164→157 main, 217→347 supp. Bundle rebuilt + UPLOAD-CHECKLIST.md refreshed. Pending: external portal upload (deadline 2026-06-17).
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
**Current focus:** Phase 14 — paper-revision-release-freeze. Post-14-21 peer-review remediation cycle complete at HEAD `20c38ad`. Bundle ready at `~/Desktop/aiche_upload_post14-21/`. Only external action (portal upload) + Plan 14-07 (Zenodo + tag at acceptance) remain.

## Current Position

Phase: 14 (paper-revision-release-freeze) — PARTIAL (20/21 plans formally complete; 14-07 deferred to journal acceptance). The 22-commit peer-review-remediation cycle is technically part of 14-21 post-work but is NOT captured as a discrete plan artifact (no 14-22-PLAN.md / SUMMARY.md was authored). ROADMAP.md Wave 18 marks 14-21 complete; the remediation wave is unmarked since it wasn't planned through ROADMAP.

Plan: 20 of 21 complete. Post-14-21 peer-review remediation cycle COMPLETE at HEAD `20c38ad`. v1.2.4 (e89fd04 era), v1.2.3 (7bf3f2c), v1.2.2 (e89fd04), v1.2.1 (3f4c2ef), and v1.2 (34eb34e) stay on origin as historical reference points; main is multiple commits ahead of origin/main pending push + tag at acceptance.

Status: Paper submission-ready with **0 LaTeX errors, 0 undefined refs/cites, 105 pages, 157 main + 347 supp literals trace to JSON**, freeze gates a/b/c PASS, gate d expected-deferred. The 22-commit remediation cycle closed all 3 BLOCKING items from the peer-review swarm + all 6 co-author comments + the deferrable per-seed sensitivity tables + Bonferroni footnote. A follow-up commit restored `ama.bst` to repo root (it was never git-tracked) and rebuilt the PDF — the 14-21-post compile had silently rendered all 60 in-text citations as `[?]` and omitted the References section; the rebuild restores them (105 pages, +6 pp from References). AIChE rebuttal letter (`docs/reviewer_response.md`) + revised manuscript + supp + bib + figures + ama.bst — all assembled in the bundle. The only remaining external items are (i) AIChE-portal upload (bundle ready to zip), (ii) GitHub release notes (held until acceptance), and (iii) plan 14-07 Zenodo DOI mint (deferred to journal acceptance).

Last activity: 2026-06-11/12 -- Peer-review remediation cycle. 4-agent peer-review subagent swarm (PR-1 citations BLOCKING, PR-2 figures BLOCKING, PR-3 prose CONCERNS, PR-4 scientific rigor BLOCKING+CONCERNS) ran in parallel; user opted into full-scope remediation including the deferrable items. Two parallel git-worktree agents executed Waves 2-5 (Agent A main.tex + bib.bib; Agent B supp_material.tex). Agent A merged cleanly; Agent B's worktree was rooted at a pre-14-21 ancestor — branch couldn't merge — salvaged high-value content templates and re-executed Agent B's task list directly on main. Bundle rebuilt + verification gates re-run.

Progress: [██████████] 96%

## Performance Metrics

**Velocity:**

- Total plans completed: 48 (v1.0 + v1.1 + v2.0 phase 8 + v2.0 phase 9 + v2.0 phase 09.1 partial)
- v2.0 plans: 13 completed (Phase 8: 5, Phase 9: 5, Phase 09.1: 3 of 4)
- Post-Plan-14-21 peer-review remediation cycle: 22 atomic commits between f85fd38 and 20c38ad — NOT formally counted as a plan since 14-22-PLAN.md was not authored.

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
| Phase 14 P17 | 18 min | 3 tasks | 2 files |
| Phase 14 P18 | ~15 min | 3 tasks | 3 files |
| Phase 14 P19 | ~20 min | 3 tasks | 105 files |
| Phase 14 P21 (×0.1 inverse-pipeline fix) | ~6 h | 7 commits + 3 deferred-optional cleanups | ~200 figure triples + 8 matched-budget JSONs |
| Phase 14 post-21 peer-review remediation | ~6 h | 22 commits across 5 waves + verification | 2 main + 1 supp + 1 bib + ~30 captions refreshed |

## Accumulated Context

### Roadmap Evolution

- Phase 09.1 inserted after Phase 9: R1-M3 Preprocessing Ablation — empirical 3-pipeline preprocessing comparison (raw OD vs log-returns vs log-returns+Lambert W) for reviewer rebuttal; spec at .planning/scratch/09.1-r1-m3-ablation-spec.md (URGENT)

### Decisions

See PROJECT.md Key Decisions table for full log.

v2.0 roadmap decisions:

- **Phase 8:** INFRA-01 + INFRA-02 isolated as foundational phase — every downstream phase imports from `core/`, so extraction + parity check must land first with no other reqs bundled
- **Phase 9:** DOC-01, DOC-02, EVAL-06 grouped as "Documentation Bridge" — cheap, paper-ready numbers front-loaded so Phase 14 paper drafting can begin in parallel with expensive experiments
- **Phases 10-13 sequencing:** Baselines → Utility Eval → Sensitivity → Architecture, so each phase consumes artifacts from the prior one; compute budget (local Mac statevector) respected by separating sensitivity sweeps from architecture sweeps
- **Phase 14:** All PAPER-* requirements plus INFRA-03 (Zenodo freeze) bundled — paper revision reads JSON from all upstream phases and the tag/DOI freeze is the final wrap-up step

v1.1 highlights retained:

- HPO-tuned values: N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05
- backprop replaces parameter-shift due to PennyLane #4462 broadcasting gradient bugs

Phase 14 sub-plan decisions:

- **Plan 14-17**: Manuscript .tex revised directly (D-14-18 READ-ONLY superseded by r4 BLOCK verdict); all PAPER-01..11 blocks integrated, abstract de-overclaimed, stale DTW 0.6843 reconciled to matched-budget ~0.30
- **Plan 14-17**: Zenodo DOI uses symbolic placeholder token ZENODO-DOI-PLACEHOLDER (digit-free) so verify_number_provenance.py passes over the .tex; 14-07 mints the real DOI into it
- **Plan 14-18**: OD-EMD claim recalibrated from 'statistically equivalent' to 'no statistically detectable difference at n=5 (underpowered)'; LR-DTW uniform-dominance posture stated, OD-DTW Orlandi improvement reframed matched-budget-wide
- **D-14-19**: drift reverted to HEAD; freeze candidate is committed HEAD 6518323; verify_freeze_ready.py hardened to certify the committed tree
- **Plan 14-21 (×0.1 inverse-pipeline fix)**: discovered + fixed a WGAN sample-space convention preserved at 9 paper-cited samples.npy load sites; inference-only correction via _wgan_unscale.py (training-side sites byte-frozen; VAE+AR(2) excluded per Pitfall 3). User authorized §6 #2 amendment at T05 R3 checkpoint. Headline reframed from bifurcated to 4-of-4 cluster dominance.
- **Post-14-21 peer-review remediation**: 4-agent swarm + co-author comments → 22 atomic commits across 5 waves + verification + bundle rebuild. NOT a formal plan (no 14-22-PLAN.md authored). Closed all BLOCKING items (stale supp Welch tables, Abstract internal inconsistency, rebuttal narrative drift) and all 6 co-author comments (refs [42][28], caption file-paths, Abstract/Contributions/Conclusions numeric trim, Figs 2-6 full-width, log-returns bioprocess one-liner, citation style consistency).

### Pending Todos

- (Optional) Retrospectively author 14-22-PLAN.md + 14-22-SUMMARY.md for the peer-review remediation cycle. Reverse-engineerable from commit log + the 4 PR reports in .planning/peer-review-2026-06-11/.

### Blockers/Concerns

- Variance collapse (fake std 48% of real) persists from v1.1 — v2.0 will NOT attempt to fully close this gap; instead it contextualizes honestly against matched classical baselines (reviewer-aligned strategy)
- Multi-seed × multi-ansatz × multi-baseline compute on local Mac only — mitigated by splitting sensitivity (Phase 12) and architecture (Phase 13) into separate phases instead of one monster sweep

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Phase 14 | 14-07 (tag v2.0-revision + Zenodo manual deposit + release.md + DOI wire-in) | deferred to acceptance — first-round revision resubmits with ZENODO-DOI-PLACEHOLDER + post-14-21 + post-peer-review-remediation HEAD `20c38ad` in Data Availability; mint real DOI when AIChE accepts and update camera-ready then | 2026-05-24 |

## Session Continuity

**Last Session:** 2026-06-12T00:12:32.258Z
**Stopped At:** Post-14-21 peer-review remediation cycle COMPLETE at HEAD `20c38ad`; follow-up `ama.bst` restore + PDF rebuild applied 2026-06-12. 22 atomic commits between f85fd38 and 20c38ad close all 3 BLOCKING items from the 4-agent peer-review swarm (PR-1 citations, PR-2 figures, PR-3 prose, PR-4 scientific rigor) plus all 6 co-author comments plus the deferrable per-seed sensitivity tables + Bonferroni footnote. Subsequent ama.bst-restore commit added the missing AMA bibliography style file (was never git-tracked → broken bibtex → 60 [?] citations + no References section in the 14-21-post PDF), rebuilt PDF to 105 pages with citations resolved. Bundle ready at `~/Desktop/aiche_upload_post14-21/` (65 files, 7.9 MB). All gates PASS (157 main + 347 supp literals; 105-page compile clean; 0 errors; 0 undefined refs/cites; freeze gates a/b/c PASS; gate d expected-deferred to 14-07).

**Resume File:** `.planning/phases/14-paper-revision-release-freeze/.continue-here.md` (updated 2026-06-12) + `.planning/HANDOFF.json` (machine-readable mirror).

**Phase 14 status:** PARTIAL (20/21 plans formally complete; 14-07 deferred to acceptance). The 22-commit peer-review remediation cycle is technically part of 14-21 post-work but is NOT captured as a discrete plan artifact (no 14-22-PLAN.md / SUMMARY.md was authored). Two options for next session: (A) retrospectively author 14-22 plan artifact from commit log + PR reports; (B) fold the remediation into a 14-23 plan that wraps Phase 14 completion at acceptance.

**AIChE resubmission state:** Camera-ready post-remediation bundle is clean (provenance + freeze gates a/b/c PASS, LaTeX compile 0 errors/0 undefined refs/0 undefined cites, **105 pages** — corrected up from a previously-stale "99 pages" claim that had been certified against a broken compile where ama.bst was missing). The headline reading dominates 4-of-4 matched-budget metrics (quantum cluster vs parameter-matched classical adversarial WGAN cluster). §A.7 disclosure includes the "Note for reviewers comparing to v1.2.4" + "Relative-fidelity impact" + "Acceptance-gate amendment" paragraphs; main §5 has the preemptive "Note on the inference-pipeline correction" subsection (PR-4 R9 — single highest-leverage addition for surviving hostile review). Per-seed sensitivity tables for OD-EMD/OD-DTW/LR-EMD exposed in supp for reviewer leave-one-out re-aggregation; Bonferroni multi-test correction footnote provided. Resubmission deadline 2026-06-17 (5 days from session pause).
