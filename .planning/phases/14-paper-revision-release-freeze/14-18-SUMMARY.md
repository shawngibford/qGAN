---
phase: 14-paper-revision-release-freeze
plan: 18
subsystem: paper-revision
tags: [claims-integrity, statistics, peer-review-r4, gap-closure]
requires:
  - docs/reviewer_response.md
  - docs/methods_full.md
  - results/welch_pairwise.json
  - verify_number_provenance.py
provides:
  - "Recalibrated OD-EMD claim: 'no statistically detectable difference at n=5 (underpowered)' instead of 'statistically equivalent'"
  - "OD-DTW Orlandi claim reframed as matched-budget-wide; LR-DTW stated as sole quantum-distinguishing DTW result"
  - "Explicit, reconciled multiple-comparisons posture for the LR-DTW and OD-EMD pairwise families"
  - "welch_pairwise.json notes field carries machine-readable non-equivalence clarification"
affects:
  - "Plan 14-07 (freeze-ready gate) — number-provenance gate over these docs re-verified PASS; 14-07 may now run"
tech-stack:
  added: []
  patterns: [wording-only-recalibration, no-recomputation]
key-files:
  created: []
  modified:
    - docs/reviewer_response.md
    - docs/methods_full.md
    - results/welch_pairwise.json
decisions:
  - "OD-EMD claim direction retained (quantum OD-EMD not distinguishable from size-matched classical) but 'equivalent' wording and positive-equivalence framing removed — a high p-value at n=5 is an absence-of-detectable-difference result, not evidence of absence"
  - "LR-DTW uniform-dominance claim treated as conjunctive: no multiplicity correction required (unlike a disjunctive >=1-significant-pair claim); reconciled with OD-EMD non-significance reported without positive-equivalence inference"
metrics:
  duration: ~15 min
  completed: 2026-05-22
  tasks: 3
  files: 3
---

# Phase 14 Plan 18: OD-EMD / OD-DTW Claim Recalibration Summary

Recalibrated the surviving OD-EMD and OD-DTW claim wording so it is
statistically defensible before the irreversible Zenodo freeze: the OD-EMD
"parametric-efficiency equivalence" claim is reframed as "no statistically
detectable difference at n=5 (underpowered)", the OD-DTW Orlandi improvement
is framed as matched-budget-wide rather than quantum-specific, the LR-DTW
multiple-comparisons posture is stated and reconciled across pairwise
families, the wgan_cnn seed-42 outlier and n=5 power limitation are disclosed
at the claim site, and `welch_pairwise.json` carries a machine-readable
non-equivalence note — all with zero recomputation and the number-provenance
gate still passing.

## What Was Built

### Task 1 — Reframe the OD-EMD equivalence claim (commit 6eeda6e)

- `reviewer_response.md` section header changed from "## Parametric-efficiency
  equivalence (post-r3 corrected metrics)" to "## Parametric efficiency: no
  detectable OD-EMD difference at matched budget (n=5, underpowered)".
- Both occurrences of "statistically equivalent" in the OD-EMD claim
  (~line 269 headline claim, ~line 287 supporting paragraph) replaced with
  "no statistically detectable OD-EMD difference".
- n=5 power limitation disclosed at the claim site in both docs: ~15% power
  against d=0.65, 80%-power detection floor d ≈ 2.0, TOST equivalence test not
  satisfied. Wording mirrors the honest DTW register at `methods_full.md`
  §3 ("statistically non-significant under the strict-accept gate; no
  equivalence test is computed").
- `methods_full.md` line ~398: "the OD-EMD equivalence claim" → "the OD-EMD
  no-detectable-difference result" with the same power disclosure appended.
- Every numeric literal (p > 0.36, |d| ≤ 0.65, n=5, per-pair table values,
  parameter counts 55/73-562) byte-unchanged.

### Task 2 — OD-DTW alignment + LR-DTW multiplicity posture + outlier disclosure (commit f522d3a)

- OD-DTW claim (~line 278) reframed: the ~6.5x improvement over Orlandi
  (1.954) is achieved by the matched-budget cluster as a whole — wgan_lstm
  (0.301) and wgan_mlp (0.302) sit inside the same 0.298-0.302 cluster as the
  quantum variants — so it is matched-budget-wide, NOT quantum-specific. The
  "temporal-structure capture quantum is specifically designed for" framing
  was removed.
- LR-DTW (every quantum 0.94-1.12 beats every WGAN+AR 1.58-7.70) stated as
  the sole quantum-distinguishing DTW result.
- Multiple-comparisons posture stated explicitly: the LR-DTW dominance claim
  is a uniform-dominance (conjunctive) claim reported as the worst-case margin
  over the pairwise family; a conjunctive "holds for every pair" claim
  requires no multiplicity correction (unlike a disjunctive ">=1 significant
  pair" claim). Reconciled with the OD-EMD non-significance result, which is
  reported without a positive-equivalence inference so multiplicity does not
  inflate a false claim there either. SYNTHESIS H1 closed in the body text,
  not left to the header.
- wgan_cnn outlier disclosure added near the per-pair table: seed 42 = 0.1587
  vs the other four seeds at 0.020-0.034 (~5x); this single seed sets BOTH
  `strong_claim_thresholds` extrema (p-floor 0.3652, |d|-ceiling 0.6442).
- All numeric literals byte-unchanged.

### Task 3 — welch_pairwise.json notes + provenance gate re-verification (commit 4b65dcc)

- Appended a machine-readable clarification to the `welch_pairwise.json`
  `notes` string (notes is a string — appended, did not restructure):
  "no statistically detectable OD-EMD difference at n=5 (underpowered ... not
  an equivalence claim ... TOST equivalence test is not satisfied ...
  80%-power detection floor d ~ 2.0)".
- The pre-existing Path A r3 retraction record and every other key and every
  numeric value preserved byte-identical (git diff confirms only the `notes`
  field changed).
- Number-provenance gate re-run over both edited docs:
  - `verify_number_provenance.py --target reviewer_response.md` →
    PASS (88 distinct numeric literals all resolve, exit 0).
  - `verify_number_provenance.py --target methods_full.md` → PASS
    (107 distinct numeric literals all resolve, exit 0).
- `scripts/run_welch_aggregator.py` not run and not edited.

## Verification

| Check | Result |
|-------|--------|
| No "statistically equivalent" / "parametric-efficiency equivalence" for OD-EMD | PASS (0 matches in either doc) |
| "underpowered" + "not an equivalence claim" present | PASS (both docs) |
| OD-DTW reframed matched-budget-wide; "specifically designed for" removed | PASS |
| LR-DTW + multiplicity/conjunctive posture present | PASS |
| wgan_cnn seed-42 outlier disclosed | PASS |
| welch_pairwise.json valid JSON, notes contains clarification | PASS (NOTES_OK) |
| Provenance gate PASS on reviewer_response.md | PASS (88 literals, exit 0) |
| Provenance gate PASS on methods_full.md | PASS (107 literals, exit 0) |
| welch_pairwise.json computed values byte-unchanged | PASS (git diff: only notes field) |

SYNTHESIS findings C1, H1, M1, M2, M3, M5 closed.

## Deviations from Plan

None — plan executed exactly as written. No recomputation, no retraining, no
metric re-evaluation (constraint D-11-10 honored); only interpretation wording
around existing, already-certified numbers was changed.

## Self-Check: PASSED

- `docs/reviewer_response.md` — FOUND, modified
- `docs/methods_full.md` — FOUND, modified
- `results/welch_pairwise.json` — FOUND, modified, valid JSON
- `.planning/phases/14-paper-revision-release-freeze/14-18-SUMMARY.md` — FOUND
- Commit 6eeda6e (Task 1) — FOUND
- Commit f522d3a (Task 2) — FOUND
- Commit 4b65dcc (Task 3) — FOUND
