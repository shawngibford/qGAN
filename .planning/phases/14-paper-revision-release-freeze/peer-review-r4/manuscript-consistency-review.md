# Agent 5 — Manuscript Consistency Review (Peer Review r4)

**Scope:** Cross-check `main (4) copy.tex` + `supp_material.tex` against JSON-backed
evidence in `results/` and the paper-block docs in `docs/`.
**Date:** 2026-05-21. **Target freeze:** git tag `v2.0-revision` → irreversible Zenodo DOI.

---

## ENVIRONMENT NOTE (read first)

This agent runs in a git worktree branched from an old commit (`c82169c`); the
worktree HEAD does **not** contain the `.tex` files or `docs/paper_blocks_*`.
All file reads in this report were performed read-only against the main repo at
`/Users/shawngibford/dev/phd/qGAN` (main HEAD `8180a5e`). No writes were made to
the main repo. Line numbers cited are from the main-repo copies of the files.

---

## EXPLICIT ANSWERS TO THE TWO KEY QUESTIONS

### Q1 — Is DTW 0.6843 reconciled in the manuscript? — **NO. NOT RECONCILED.**

The number `0.6843` appears **three times** in the manuscript as a bare, unlabeled
headline result:

- `main (4) copy.tex:190` — "The DTW distance between original and synthetic
  sequences was 0.6843 (Figure~\ref{fig:DTWD})."
- `main (4) copy.tex:266` — "The QWGAN-GP achieved a Dynamic Time Warping (DTW)
  score of 0.6843, representing improved temporal alignment compared to
  previously reported methods."
- `supp_material.tex:290` — "Our score of 0.6843 indicates strong temporal
  fidelity in the generated bioprocess sequences."

The audited matched-2000-epoch evidence flatly contradicts this:

- `matched2000_dualscale.json#aggregates` quantum OD-scale DTW means (n=5 seeds):
  iqp_sel_55_repro **0.3019**, V1 0.3005, V2 0.2984, V3 0.2991; frozen-checkpoint
  headline **0.2608**. (log-return scale: ~0.94–0.99.)
- `reconciliation_note.md` "EMD comparable across metric variants" table reports
  the quantum DTW (OD) column at **0.298–0.302**.

**No value in any current artifact equals or rounds to 0.6843.** The project's
own audit doc says so explicitly — `methods_full.md:427-434`:

> "The manuscript headline DTW=0.6843 at `main (4) copy.tex:190` +
> `main (4) copy.tex:266` + `supp_material.tex:290` originates from a pre-v1.0
> best-case iqp_sel_55 evaluation pipeline; this value is not re-emitted by the
> current matched-budget contract under the strict-accept gate (D-14-13)."

So the project KNOWS 0.6843 is a stale pre-v1.0 best-case number. But that
disclosure lives **only in `docs/methods_full.md` and
`reviewer_response.md` (markdown)** — it is **completely absent from the
manuscript `.tex`**. The manuscript text presents 0.6843 as the live result with
no "frozen pre-v1.0 checkpoint" label, no reconciliation to the 0.30 matched
number, and no pointer to the disclosure. A reader (and the journal) sees an
unqualified `0.6843` that the project's own evidence base contradicts by ~2.3x.

This is exactly the failure mode the task brief warned about: **a stale,
contradictory number about to be frozen into a cited paper.**

### Q2 — Are the paper_blocks revisions integrated into the `.tex`? — **NO. NOT INTEGRATED.**

The `.tex` files are still the **fully un-revised pre-revision version**. Evidence:

- A grep for the revised-block AFTER-text signatures ("matched parameter budget",
  "matched-epoch", "do not exceed", "falsifiable question", "parameter-matched")
  across `main (4) copy.tex` returns **0 matches**.
- `main (4) copy.tex:266` still reads the un-revised
  "successfully developed and validated... high fidelity" Key Contributions
  sentence. PAPER-02b (LOCKED, D-14-20) requires this be replaced with the
  "comparable to, but do not exceed... size-matched classical WGAN-GP baselines"
  AFTER block. **Not applied.**
- `main (4) copy.tex:296` still reads "demonstrates that quantum-enhanced GANs
  are a viable approach to addressing data scarcity challenges in industrial
  bioprocess engineering" — PAPER-02c requires the scoped AFTER block.
  **Not applied.**
- `main (4) copy.tex:276` still reads "quantum computational advantages can be
  effectively applied" — PAPER-02d requires removal. **Not applied.**
- `supp_material.tex:135` / `:334` still contain the "reduced susceptibility to
  mode collapse" overclaim PAPER-02d requires softened. **Not applied.**
- The PAPER-08 "Dataset and preprocessing" paragraph (778 points / 777
  log-returns / 384 windows / 5 seeds) is **absent** from §3.2.
- The PAPER-09 "Evaluation scale" paragraph + `tbl:eval_scale` is **absent**
  from §4.1.
- PAPER-01a/01b reframed-hypothesis text is **absent** from §1.4 / §2.4.

Furthermore, the PAPER-02b BEFORE block in `paper_blocks_framing.md:118` quotes
`main:266` as already reading "achieved an improved Dynamic Time Warping (DTW)
score" (NO numeral) — but the actual `.tex` at line 266 reads "achieved a
Dynamic Time Warping (DTW) score of **0.6843**". The framing doc was authored
against a manuscript state that does not match the file on disk. The `.tex` is
even **older** than the framing doc's stated baseline.

**The entire r1/r2/r3 revision — including the LOCKED de-overclaiming
requirement and the "Path A reframe" — lives only in markdown. The manuscript
itself has not received a single revision edit.**

---

## FINDINGS BY SEVERITY

### CRITICAL

**C-1 — Stale unreconciled DTW headline 0.6843 frozen into a cited paper.**
`main (4) copy.tex:190, 266`; `supp_material.tex:290`. The headline result is a
pre-v1.0 best-case number contradicted ~2.3x by every current matched-budget
artifact (`matched2000_dualscale.json`, `reconciliation_note.md`). The project's
own `methods_full.md:427` and `reviewer_response.md:361,374` flag it as stale,
but the disclosure never reaches the manuscript. Freezing the DOI now mints a
permanent citation to a number the project itself has retracted internally.

**C-2 — None of the r1/r2/r3 paper_blocks revisions are integrated into the
`.tex`.** The manuscript is the pre-revision version. The LOCKED (D-14-20)
de-overclaiming blocks PAPER-02a/b/c/d are not applied; the Path A reframe
(PAPER-01a/b, PAPER-02b/c) is not applied; PAPER-03/05/06/07/08/09/10/11 are not
applied. A DOI freezing this `.tex` freezes overclaiming text the reviewers
explicitly required removed, and contradicts the JSON evidence base the rest of
the v2.0 release is built on.

**C-3 — The manuscript `.tex` files are NOT git-tracked.**
`git ls-files --error-unmatch 'main (4) copy.tex'` → "did not match any file(s)
known to git"; same for `supp_material.tex`. They appear only as untracked (`??`)
in `git status`. **If the repo is frozen at tag `v2.0-revision` by committing
tracked files, the manuscript will not be in the tagged tree at all** — or, if
`git add`-ed at the last moment, the pre-revision version gets frozen (see C-2).
Either outcome is wrong for a DOI that a journal manuscript will cite.

### HIGH

**H-1 — Key Contributions claims direct contradiction of the matched-budget
result.** `main (4) copy.tex:266` claims the QWGAN-GP "achieved... improved
temporal alignment compared to previously reported methods" and "successfully
generates synthetic time series data with high fidelity." The audited result
(`reconciliation_note.md`, `methods_full.md:436-449`) is that quantum DTW is
**statistically indistinguishable** from size-matched classical WGAN baselines
on OD scale (0.298–0.302 cluster, "statistically non-significant under the
strict-accept gate"). The manuscript claims superiority the evidence withdraws.
PAPER-02b is the LOCKED fix; it is not applied.

**H-2 — Abstract overclaims uncorrected.** `main (4) copy.tex:49` —
"Our results demonstrate strong performance... The synthetic data showed high
fidelity to actual historical experimental data." This is the same withdrawn
superiority framing; the Path A reframe requires the abstract scoped to
"comparable to parameter-matched classical baselines." Not applied.

**H-3 — Orlandi reference DTW 1.954 used as a favorable comparator on a stale
basis.** `main (4) copy.tex:191-194` and `:266` frame "lower DTW score" vs
Orlandi's 1.954 using the stale 0.6843. The current evidence (DTW ~0.30) would
make the comparison *stronger* (~6.5x per `methods_full.md:451`), but the
manuscript ties the comparison to a number that is not reproducible under the
current contract. The comparison is currently anchored to a non-existent result.

### MEDIUM

**M-1 — Data Availability section has no DOI placeholder at all.**
`main (4) copy.tex:291-292` and `supp_material.tex:295-298` cite only the GitHub
URL `https://github.com/shawngibford/qgan`. There is **no Zenodo DOI placeholder
field** for plan 14-07 to mint into. Per the task brief, 14-07 is expected to
mint a DOI; the manuscript currently has nowhere to put it and no
"archived snapshot, DOI: 10.5281/zenodo.XXXXX" placeholder. This is a
consistency gap between what 14-07 will produce and what the manuscript exposes.

**M-2 — Notation not unified (PAPER-11 not applied).** `main (4) copy.tex:250`
ACF caption uses "log $\delta$" for the return symbol; `supp_material.tex:356`
defines the return as `r_t`. PAPER-11 (`paper_blocks_refs_methods.md:505-519`)
requires unifying on `r_t` everywhere. Not applied — two return symbols coexist.

**M-3 — 20L vs 300L LUCY photobioreactor mismatch uncorrected.**
`main (4) copy.tex:178` says "20-liter photobioreactor (LUCY...)" then "see
Figure... for the 300L configuration of the 20L version"; `supp_material.tex:346`
caption says "300L LUCY". PAPER-05c flags this as a defect to fix. Not applied.

**M-4 — Malformed `\label` inside running text.** `main (4) copy.tex:178` —
"see Figure \label{fig:lucy} for..." uses `\label` where `\ref` is intended; the
label is also defined again at `supp_material.tex:347`. This will produce a
broken/duplicate-label reference. PAPER-05c flags it. Not applied.

### LOW

**L-1 — Figure references rely on commented-out `\ref` targets.** Numerous
`%~\ref{...}` comment-outs (`main:82, 117, 166, 261, 287`; "Figure A1/A2/A4/A5",
"Table A1/A2" hand-typed instead). Cross-references are hand-typed letter+number
strings, bypassing LaTeX's auto-numbering. `supp_material.tex` figures are
A-prefixed; if any supp float is added/removed the hand-typed "Figure A4" etc.
silently goes stale. Functional but fragile; not provenance-gated.

**L-2 — `concept_diagram.png` (`main:65`) and several figure image files** are
referenced but not verifiable from this worktree; recommend Agent(s) with
repo-figure access confirm all `\includegraphics` targets resolve in the frozen
tree (`dtwd.png`, `pdf.png`, `cdf.png`, `qq.png`, `acf.png`, `quantum_circuit.png`,
`classicalgan.png`, `hybridgan.png`, `mech_rep.png`, `bpm_qgan.drawio.png`,
`lucy_diagram.jpg`, `concept_diagram.png`). The figure regeneration
(`results/figures/*.pdf`, PAPER-11 note) is also not wired into the
`.tex` `\graphicspath`.

**L-3 — `\bibliography{bib}` (`main:304`)** — the `bib` source file was not
located in the repo root; if it is not part of the frozen tree the manuscript
will not compile from the DOI snapshot. Recommend verifying `bib.bib` is present
and tracked.

---

## SEVERITY COUNTS

| Severity | Count |
|---|---|
| CRITICAL | 3 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 3 |

---

## ASSESSMENT

The JSON evidence base (`results/`) and the audit/markdown layer
(`docs/`) are internally consistent, carefully provenance-gated, and
honest about the Path A reframe and the pre-v1.0/matched-budget metric history.
That work is sound.

**But the manuscript `.tex` files were never touched.** They are the
pre-revision version — un-revised, un-tracked, carrying a stale headline DTW that
the project's own audit docs explicitly retract. The revision exists entirely in
markdown that no journal reader will see. Freezing the DOI now would either (a)
freeze a tree with no manuscript in it, or (b) freeze the un-revised
overclaiming manuscript with the contradicted 0.6843 headline. Both are
unacceptable for a DOI that an AIChE major-revision manuscript will cite.

This is a NEW contradiction (manuscript vs. evidence base), not a re-litigation
of a closed item — the closed items (R3-CR-1, R3-CR-2, the EMD claim withdrawal)
are correctly handled in the JSON/markdown layer. The regression is that the
manuscript integration step was never executed.

### Required before freeze

1. Apply ALL paper_blocks revisions (PAPER-01..11) to `main (4) copy.tex` and
   `supp_material.tex`, especially the LOCKED PAPER-02 de-overclaiming set.
2. Resolve the 0.6843 headline: either replace it with the matched-budget DTW
   (~0.30 quantum cluster) sourced from `matched2000_dualscale.json`, or label
   it explicitly as "frozen pre-v1.0 checkpoint" with the matched-budget number
   stated alongside and a pointer to the disclosure — mirroring how
   `methods_full.md` and `reviewer_response.md` already handle it. A bare
   unlabeled 0.6843 must NOT be frozen.
3. Add a Zenodo DOI placeholder to the Data Availability section so plan 14-07's
   minted DOI has a destination.
4. `git add` and commit the revised `.tex` files (and `bib.bib`) so they are in
   the `v2.0-revision` tagged tree.

None of this can be done from this read-only worktree; it must be done in the
main repo before tagging.

---

FREEZE VERDICT: BLOCK
