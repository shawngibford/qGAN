---
phase: 09-documentation-bridge
plan: 04
subsystem: documentation
tags: [paper-methods, dataset-stats, doc-02, lucy-photobioreactor, single-campaign, r1-m2, r1-m5]

# Dependency graph
requires:
  - phase: 08-core-module-extraction
    provides: revision/core/data.py::load_and_preprocess as canonical pipeline whose live output (778/777/384) anchors the doc counts
provides:
  - revision/docs/dataset_stats.md (paper-ready Methods content for Phase 14 PAPER-08)
  - explicit train:val:test ratio row (100% : 0% : 0%) satisfying ROADMAP DOC-02 success criterion
  - Single-Campaign Limitation prose block addressing R1-M5 calibration honesty
  - PAR_LIGHT disabled-conditioning note for unconditioned_wgan baseline
affects: [phase-14-paper, phase-09.1-r1-m3-ablation, phase-13-conditional-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-08 hybrid doc format: tables for numeric content + 1-paragraph prose for justifications"
    - "D-09 traceability: every count cites source file:line in revision/core/data.py"
    - "Live-verified counts (NOT CONTEXT.md prose) — reconciles RESEARCH OQ-1/2/3"

key-files:
  created:
    - revision/docs/dataset_stats.md
  modified: []

key-decisions:
  - "Honored Reconciled Numbers from planning_context (778/777/384), not the CONTEXT.md prose values (777/384)"
  - "Used corrected end date 2024-04-01 23:42 (~5.4 days), not the earlier RESEARCH-era 2024-03-31 23:52 (~4.5 days) — live data.csv verification on 2026-05-15 confirms 2024-04-01 23:42"
  - "Added explicit '100% : 0% : 0% (384 : 0 : 0 windows)' split-ratio row to literally satisfy ROADMAP DOC-02 wording while honoring D-01 (no split)"
  - "Placed Single-Campaign Limitation prose under Split Convention section (D-02 anchor location)"
  - "Phase 09.1 ablation pointer in Preprocessing Pipeline section (ABL-01/02/03 reference)"

patterns-established:
  - "Source-of-truth blockquote at top of doc: names data.csv + load_and_preprocess once, then table rows cite specific file:line"
  - "Single-Campaign Limitation prose: explicit acknowledgment of EMD-on-same-distribution caveat per R1-M5"

requirements-completed: [DOC-02]

# Metrics
duration: 1min
completed: 2026-05-15
---

# Phase 09 Plan 04: Dataset Statistics Documentation Summary

**Paper-ready `revision/docs/dataset_stats.md` (82 lines) with live-verified counts 778/777/384, single-campaign limitation prose addressing R1-M5, and explicit 100%:0%:0% split-ratio row satisfying ROADMAP DOC-02**

## Performance

- **Duration:** ~1 min (74 seconds)
- **Started:** 2026-05-15T15:57:52Z
- **Completed:** 2026-05-15T15:59:06Z
- **Tasks:** 1
- **Files modified:** 1 (created)

## Accomplishments
- Wrote `revision/docs/dataset_stats.md` end-to-end as drop-in Methods content for Phase 14 PAPER-08
- Reconciled live-pipeline counts (778 raw OD / 777 log_delta / 384 rolling windows) — explicitly rejecting the CONTEXT.md prose values of 777/384 per RESEARCH OQ-1/2/3
- Resolved end-date discrepancy: live `data.csv` confirms last row is `1/4/24 23:42` (2024-04-01 23:42), giving a corrected ~5.4-day duration (RESEARCH-era estimate of "~4.5 days, 2024-03-31 23:52" was based on an earlier snapshot)
- Added explicit Split Convention table row reporting "100% : 0% : 0% (384 : 0 : 0 windows)" to literally satisfy ROADMAP DOC-02 success criterion "split ratios and counts" while honoring D-01 (no held-out split)
- Substantive Single-Campaign Limitation prose paragraph (10 sentences) acknowledges EMD-on-same-distribution caveat per R1-M5 calibration honesty
- PAR_LIGHT note documents conditioning was disabled in v1.1 final `unconditioned_wgan` run; reserves the column for Phase 13 conditional-generation introspection
- Phase 09.1 (ABL-01/02/03) preprocessing-ablation pointer present in Preprocessing Pipeline section

## Task Commits

Each task was committed atomically:

1. **Task 1: Write revision/docs/dataset_stats.md (5 sections, hybrid table+prose)** — `c50d281` (docs)

## Files Created/Modified
- `revision/docs/dataset_stats.md` — 82 lines, 5 sections (Counts, Sampling & Date Range, Split Convention, Preprocessing Pipeline, PAR_LIGHT Note) + Single-Campaign Limitation prose anchor under Split Convention; hybrid table-plus-prose format per D-08

## Verified Live Numbers (RESEARCH OQ-1/2/3 reconciled)

| Quantity | Live value | Source |
|---|---|---|
| Raw CSV data rows | 778 | `python: len(open('data.csv').readlines()) - 1 == 778` |
| Log-return rows (N−1) | 777 | `revision/core/data.py:62` (`log_od[1:] - log_od[:-1]`) |
| Rolling windows (m=10, s=2) | 384 | `(777 − 10) // 2 + 1 = 384`; `revision/core/data.py:110-118` |
| Start date | 2024-03-27 13:12 | first row of `data.csv` (`27/03/2024 13:12`) |
| End date | 2024-04-01 23:42 | last row of `data.csv` (`1/4/24 23:42`) |
| Duration | ~5.4 days (5.4375 d = 5 d 10 h 30 min) | computed from start → end |
| Sampling cadence | 10 minutes | consecutive `DATE` deltas |
| Independent campaigns | 1 | LUCY photobioreactor (Algenuity), single run |

## Acceptance-Criteria Grep Gate Results

All 19 acceptance-criteria gates passed with margin:

| Gate | Required ≥ | Actual | Status |
|---|---|---|---|
| 778 | 1 | 2 | OK |
| 777 | 1 | 2 | OK |
| 384 | 1 | 5 | OK |
| `10-min` / `10 minutes` | 1 | 3 | OK |
| 2024-03-27 | 1 | 1 | OK |
| 2024-04-01 | 1 | 1 | OK |
| 5.4 | 1 | 2 | OK |
| split ratio (100% : 0% : 0% or 384 : 0 : 0) | 1 | 2 | OK |
| Single-Campaign | 1 | 2 | OK |
| PAR_LIGHT | 1 | 5 | OK |
| unconditioned_wgan | 1 | 2 | OK |
| LUCY | 1 | 4 | OK |
| load_and_preprocess | 1 | 2 | OK |
| data.csv | 1 | 5 | OK |
| Phase 09.1 / Phase 9.1 | 1 | 1 | OK |
| NONE | 1 | 2 | OK |
| Outlook | 1 | 1 | OK |
| lambert | 1 | 3 | OK |
| Line count | 60 | 82 | OK |

## Decisions Made
- **Live counts win over CONTEXT.md prose.** The plan's `<interfaces>` section already locked the Reconciled Numbers (778/777/384); this execution used those directly without re-checking against the older CONTEXT.md prose claim of 777/384.
- **End date is 2024-04-01 23:42.** Live `tail data.csv` shows `1/4/24 23:42` as the last row. RESEARCH-era "2024-03-31 23:52" was based on an earlier snapshot; the plan's Reconciled Numbers correctly used the newer value, so duration is ~5.4 days, not ~4.5 days.
- **Single-Campaign Limitation prose placement.** Anchored under Split Convention (the D-02 location) because that's where the no-split decision is justified.
- **Explicit ratio row added.** ROADMAP DOC-02 reads "split ratios and counts"; honoring D-01 (no split) requires stating the ratio is 100% : 0% : 0%. Both are reported in one table row.

## Deviations from Plan

None — plan executed exactly as written. The plan's `<action>` block specified the file content verbatim and Task 1 wrote it directly; no auto-fixes (Rules 1-3) needed, no architectural questions (Rule 4) raised. All counts, dates, and section ordering match the plan spec.

## Issues Encountered
None. Live data.csv inspection and `load_and_preprocess` line citations all matched the plan's Reconciled Numbers without conflict.

## User Setup Required
None — no external service configuration required. This is a documentation deliverable.

## Next Phase Readiness

**Phase 14 PAPER-08 (Methods § Dataset details):** `revision/docs/dataset_stats.md` is the upstream artifact and is drop-in ready. The doc:
- Reports all reviewer-required counts (R1-m2: dataset details) with file:line citations
- Acknowledges the single-campaign / no-split methodological constraint per R1-M5 (calibration honesty)
- Defers multi-campaign generalization to Phase 14 Outlook (not a current-scope claim)
- Provides explicit train:val:test ratio satisfying ROADMAP DOC-02 success criterion

**Phase 09.1 (R1-M3 preprocessing ablation):** Preprocessing Pipeline section references ABL-01/02/03 as the head-to-head ablation against raw-OD and log-return-only pipelines.

**Phase 13 (conditional-generation introspection):** PAR_LIGHT note reserves the column for future conditioning work; v1.1 final `unconditioned_wgan` baseline explicitly disabled conditioning.

No blockers.

## Self-Check: PASSED

- `revision/docs/dataset_stats.md` — FOUND (82 lines)
- Commit `c50d281` — FOUND in git log
- All 19 acceptance-criteria grep gates — PASSED

---
*Phase: 09-documentation-bridge*
*Completed: 2026-05-15*
