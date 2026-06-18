# Peer Review r4 — Synthesis

**Date:** 2026-05-21
**Scope:** Final pre-DOI-freeze review of the DOI-bound surface (revision/ code + results
+ docs, manuscript .tex, freeze/release packaging) before GSD plan 14-07 cuts tag
`v2.0-revision` and mints an irreversible Zenodo DOI.
**Method:** 6 specialist agents, each in an isolated git worktree. Reports:
`math-stats-review.md`, `code-correctness-review.md`, `results-provenance-review.md`,
`claims-integrity-review.md`, `manuscript-consistency-review.md`,
`freeze-readiness-review.md`.

---

## OVERALL FREEZE VERDICT: **BLOCK**

The repo must **not** be frozen / DOI'd in its current state. Two agents returned BLOCK,
one agent returned a CRITICAL, three returned GO / GO-WITH-FIXES. **7 CRITICAL findings**
across two independent failure domains:

1. **The manuscript was never revised.** The r1/r2/r3 revision exists only in markdown
   (`docs/paper_blocks_*.md`). The actual `.tex` files are the un-revised,
   pre-revision version, are not git-tracked, and carry a stale headline number the
   project's own audit docs internally retract. Freezing now mints a DOI citing either
   no manuscript or the un-revised overclaiming manuscript.
2. **The working tree is dirty and the freeze gate is structurally weak.** The
   `.gitignore` protections for the provenance backbone exist only uncommitted;
   `real.csv`/`fake.csv` have drifted on every row; the mandated `release.md` does not
   exist; `scripts/verify_freeze_ready.py` validates the dirty working tree rather than the tag
   candidate.

Separately, the surviving **OD-EMD "parametric-efficiency equivalence" claim is
statistically indefensible as worded** (a non-significant difference test is not evidence
of equivalence) — a contained but mandatory documentation fix.

The metric **code and numbers are sound**: all 5 emitters reproduce byte-identically, the
test suite is 23/23 green, the provenance gate passes, R3-CR-1/R3-CR-2 fixes show no
regression. The evidence base is trustworthy — the blockers are manuscript integration,
working-tree hygiene, and claim wording, not the science pipeline.

### Per-agent verdicts

| Agent | Domain | Verdict | C / H / M / L |
|---|---|---|---|
| 1 | Math & Statistics | GO-WITH-FIXES | 1 / 2 / 3 / 2 |
| 2 | Code Correctness | GO | 0 / 0 / 0 / 3 |
| 3 | Results & Provenance | GO-WITH-FIXES | 0 / 1 / 1 / 2 |
| 4 | Claims & Analysis Integrity | GO | 0 / 0 / 1 / 1 |
| 5 | Manuscript Consistency | **BLOCK** | 3 / 3 / 4 / 3 |
| 6 | Freeze & Release Readiness | **BLOCK** | 3 / 3 / 3 / 2 |

---

## TRIANGULATED FINDINGS (flagged by ≥2 agents — highest confidence)

- **OD-EMD equivalence claim is statistically unsound** — Agent 1 (CRITICAL B1, ran TOST:
  0/20 pairs pass) + Agent 4 (MEDIUM F1). Independent confirmation from the math reviewer
  and the claims reviewer.
- **Manuscript `.tex` files are not git-tracked** — Agent 5 (CRITICAL C-3) + Agent 6
  (HIGH-2). They will be absent from, or wrongly frozen into, the tag.
- **Welch strong-claim thresholds clear by razor-thin margins** (p=0.3652 vs >0.36;
  |d|=0.6442 vs ≤0.65) — Agents 1, 2, 3 all noted independently. Correct arithmetic, but
  fragile and outlier-driven (Agent 1 B4).

---

## CONSOLIDATED FINDINGS

### CRITICAL (7) — all must be resolved before freeze

| ID | Agent | Finding | Where |
|---|---|---|---|
| C1 | 1 (B1) | OD-EMD "equivalence" claim equates a non-significant difference test with evidence of equivalence; n=5 has ~15% power vs d=0.65; proper TOST fails 0/20 pairs | reviewer_response.md:269-323, methods_full.md:398-399, run_welch_aggregator.py:138-182 |
| C2 | 5 (C-1) | Stale unreconciled headline DTW **0.6843** presented as the live result; every matched-budget artifact says ~0.30; project's own methods_full.md:427 retracts it but the disclosure never reaches the manuscript | main (4) copy.tex:190,266; supp_material.tex:290 |
| C3 | 5 (C-2) | **None** of the PAPER-01..11 paper_blocks revisions are integrated into the .tex — including the LOCKED (D-14-20) de-overclaiming set and the Path A reframe. The manuscript is the pre-revision version | main (4) copy.tex, supp_material.tex |
| C4 | 5 (C-3) + 6 (HIGH-2) | The `.tex` manuscripts are not git-tracked — they will not be in the tagged tree, or the un-revised version gets frozen | `git ls-files '*.tex'` → empty |
| C5 | 6 (CRIT-1) | The `!results/` gitignore negations protecting the provenance backbone exist **only in the uncommitted working tree**; committing the `.gitignore` change without the negations in the same atomic commit ignores all provenance JSON | .gitignore (HEAD vs working tree) |
| C6 | 6 (CRIT-2) | `real.csv` / `fake.csv` modified on **every row** (777/770 rows) in the working tree — unintended data drift; freezing tags state inconsistent with the certified results JSON | real.csv, fake.csv |
| C7 | 6 (CRIT-3) | `docs/release.md` (mandated by the 14-07 plan must_haves, ≥30 lines: tag SHA, DOIs, check-ignore result, reproduce steps) **does not exist** | docs/ |

### HIGH (7)

| ID | Agent | Finding |
|---|---|---|
| H1 | 1 (B2) | Multiple-comparisons posture inconsistent across pairwise families; unstated for the surviving LR-DTW claim |
| H2 | 5 (H-1) | Key Contributions sentence (main:266) claims "improved temporal alignment" / "high fidelity" — superiority the matched-budget evidence withdraws |
| H3 | 5 (H-2) | Abstract (main:49) carries the same uncorrected "high fidelity" overclaim |
| H4 | 5 (H-3) | Orlandi-reference DTW comparison is anchored to the stale, non-reproducible 0.6843 |
| H5 | 3 (HIGH-1) | Review worktrees were provisioned at stale commits; the DOI must be minted from current HEAD `8180a5e` (verified freeze-candidate state) |
| H6 | 6 (HIGH-1) | `results/baselines/runs/` (47 MB, 250 files) is untracked — classical-baseline comparison numbers cannot be reproduced from the deposit; commit at least the metrics.json/config.yaml |
| H7 | 6 (HIGH-3) | `scripts/verify_freeze_ready.py` validates the live (dirty) working tree, not the committed tag candidate — a green run does not certify the tag |

### MEDIUM (12)

| ID | Agent | Finding |
|---|---|---|
| M1 | 1 (B3) | OD-DTW "6.5× vs Orlandi" improvement is shared by wgan_lstm/wgan_mlp — not quantum-specific; reviewer_response.md:278-281 overclaims vs methods_full.md |
| M2 | 1 (B4) | Both strong-claim threshold extrema (p-floor, |d|-ceiling) are set by a single wgan_cnn outlier seed (seed 42, ~5× the other seeds) |
| M3 | 1 (B5) | n=5 power limitation is not disclosed at the equivalence-claim site |
| M4 | 3 (MED-1) | `requirements-pinned.txt` incomplete: `fastdtw` missing, `pandas` unpinned → resolves to pandas 3.0 and breaks statsmodels; a clean install from the recipe fails |
| M5 | 4 (F1) | Equivalence wording should soften to "statistically indistinguishable (n=5)" — same root as C1 |
| M6 | 5 (M-1) | Data Availability section has no Zenodo DOI placeholder for 14-07 to mint into |
| M7 | 5 (M-2) | Notation not unified — `log δ` vs `r_t` coexist (PAPER-11 not applied) |
| M8 | 5 (M-3) | 20L vs 300L LUCY photobioreactor mismatch uncorrected (PAPER-05c) |
| M9 | 5 (M-4) | Malformed `\label` used where `\ref` intended (main:178) — broken/duplicate reference |
| M10 | 6 (MED-1) | 18 ablation `checkpoint.pt` files (~36 MB) ship in the archive (sub-threshold, tolerated, flag for awareness) |
| M11 | 6 (MED-2) | `results/phase4_validation.json` modified in working tree — uncommitted drift, confirm intent |
| M12 | 6 (MED-3) | `git archive` is ~112 MB; confirm `.planning/` history belongs in a public DOI deposit (D-14-21 says yes — final confirmation) |

### LOW (13, summarized)

Gate-v2 weak-match class (can only false-PASS, non-material — Agent 2); bare `assert` in
run_distribution_emd.py self-test (Agent 2); MWU computed but unused (Agent 1 B6);
`data_hash` hardcoded not computed (Agent 1 B7); fragile hand-typed figure refs / verify
`\includegraphics` targets resolve (Agent 5 L-1/L-2); confirm `paper/bib.bib` is tracked
(Agent 5 L-3); ` *.csv` leading-space in .gitignore (Agent 6 L-1); untracked junk to
exclude — `qgan_pennylane copy.ipynb`, `datasets.zip`, `amp`, the PDF, `.claude/`
(Agent 6 L-2). **Plus:** `LICENSE` is deleted in the working tree (uncommitted) — present
at HEAD, but must be `git checkout`-reverted before any pre-freeze commit or the deposit
ships license-less.

---

## REMEDIATION PLAN (ordered) — feed into a 14-07 prep / gap-fix cycle

**Manuscript track (resolves C2, C3, H2, H3, H4, M6–M9):**
1. Apply all PAPER-01..11 paper_blocks revisions to `main (4) copy.tex` + `paper/supp_material.tex`
   — especially the LOCKED (D-14-20) PAPER-02 de-overclaiming set and the Path A reframe.
2. Resolve the 0.6843 DTW headline: replace with the matched-budget DTW (~0.30 quantum
   cluster, sourced from `matched2000_dualscale.json`), or label it explicitly as the
   frozen pre-v1.0 checkpoint with the matched-budget number stated alongside.
3. Add a Zenodo DOI placeholder to the Data Availability section.
4. Fix notation (`r_t`), the 20L/300L mismatch, and the malformed `\label`.

**Claims track (resolves C1, H1, M1–M3, M5):**
5. Reframe "equivalent / equivalence" → "no statistically detectable OD-EMD difference at
   n=5 (underpowered — 80%-power floor d≈2.0; not an equivalence claim)" in
   `reviewer_response.md`, `methods_full.md`, and a `notes` field in `welch_pairwise.json`.
   This mirrors language methods_full.md:441-442 already uses for DTW.
6. Align the reviewer_response.md OD-DTW claim to the honest methods_full.md framing
   (OD-DTW improvement is matched-budget-wide, not quantum-specific; only LR-DTW
   distinguishes quantum).
7. Disclose the wgan_cnn outlier-seed dependence and the n=5 power limitation at the
   claim site.

**Freeze / release track (resolves C4, C5, C6, C7, H6, H7, M4, M10–M12):**
8. `git checkout -- LICENSE` — revert the unintended working-tree deletion.
9. Owner decides canonical `real.csv` / `fake.csv` / `results/phase4_validation.json`;
   confirm consistency with the certified `results/*.json`; commit or revert
   deliberately.
10. Commit the `.gitignore` change as ONE atomic commit (the `results/` line AND both
    `!results/` negations together); fix the ` *.csv` leading space.
11. Commit `results/baselines/runs/` metrics.json + config.yaml artifacts.
12. Decide `.tex` tracking; `git add` + commit the revised `.tex` files and `paper/bib.bib`.
13. Author and commit `docs/release.md` per the 14-07 spec.
14. Fix `requirements-pinned.txt`: add `fastdtw`, pin `pandas<3.0`.
15. Re-run `scripts/verify_freeze_ready.py` against the now-clean **committed** tree; confirm
    `git status --porcelain` is empty; verify `git check-ignore results/*.json`
    against the committed `.gitignore`; then cut `git tag v2.0-revision` from HEAD and
    confirm `git archive` contains `results/*.json`, `data.csv`, `LICENSE`.

Steps 1–7 are documentation/wording edits; steps 8–15 are working-tree hygiene and the
freeze sequence. None require retraining or re-computation — every number is already
certified. Recommend routing this through `/gsd-plan-phase 14 --gaps` (or a UAT-style
gap file) so the fixes land as a tracked plan **before** 14-07 cuts the tag.

---

## WHAT IS CONFIRMED SOUND (do not re-touch)

- All 5 metric emitters reproduce byte-identically vs the committed JSONs (Agent 3).
- `tests/` suite: 23 passed / 0 genuine failures; `core/` byte-frozen
  (Agent 2).
- Number-provenance gate PASSES on all 10 paper-facing docs; 480 literals resolve;
  `data_hash` `91e447d4624e25b3` consistent across all 27 results JSONs; checkpoint
  SHA256 matches the config lock (Agent 3).
- R3-CR-1 and R3-CR-2 fixes are correct with no regression (Agents 1, 2).
- Path A scrub is clean — no orphaned "quantum beats WGAN on LR-EMD" remnant survives in
  any of the 7 paper-facing docs (Agent 4).
- No committed secrets; `ZENODO_TOKEN` appears nowhere in tracked files (Agent 6).
