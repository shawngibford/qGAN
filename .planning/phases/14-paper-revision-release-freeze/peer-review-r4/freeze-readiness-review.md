# Agent 6 — Freeze & Release Readiness Review (Peer Review r4)

Scope: GSD plan 14-07 — final gate before an irreversible Zenodo DOI is minted by
freezing the repo at tag `v2.0-revision`.

Repo: `/Users/shawngibford/dev/phd/qGAN`
HEAD: `8180a5eca61333888c8c1e3bb193f1942cf9edeb` (branch `main`, 319 commits ahead of `origin/main`)
Tags present: `v1.0` only — **`v2.0-revision` has NOT been cut yet.**

---

## EXECUTIVE ANSWERS (explicitly requested)

1. **Is LICENSE present at HEAD?** — YES. `git show HEAD:LICENSE` returns a full
   MIT License (Copyright (c) 2025 shawngibford), committed in `fdbbdfe`. The
   `git status` "deleted: LICENSE" is a **working-tree-only deletion that is NOT
   committed**. A tag/archive cut at HEAD today *would* include LICENSE. **However**,
   if the working tree is committed as-is before tagging, the deletion would be
   captured and the DOI deposit would ship with NO license — CRITICAL. The
   deletion must be reverted (`git checkout -- LICENSE`) before any pre-freeze commit.

2. **Any committed secrets?** — NO. `git grep` over the full HEAD tree for
   `ZENODO_TOKEN`, `ghp_*`, `AKIA*`, `Bearer`, private-key headers, and
   `password=` found nothing. No `.env` file is tracked. The untracked LaTeX
   manuscripts (`main (4) copy.tex`, `supp_material.tex`) and `docs/`
   are also clean. The 14-07 requirement that `ZENODO_TOKEN` never be committed
   is currently satisfied.

3. **What untracked items MUST be committed before freeze?** — One mandatory:
   `results/baselines/runs/` (47 MB, 250 files — the per-seed baseline
   run artifacts that the classical-baseline comparison numbers depend on; see
   HIGH-1). Possibly: the LaTeX manuscripts if D-14-21's "includes .tex reference
   files" is to be honored (see HIGH-2). Everything else untracked is junk and
   should NOT be committed (see Section 3).

---

## FREEZE BLOCKERS — must resolve before tagging

### CRITICAL-1 — `results/` provenance backbone is NOT protected by the committed `.gitignore`

The 14-07 plan's central risk (RESEARCH Pitfall 4) is that the provenance JSON
gets gitignored out of the tag. The fix — the `!results/` negation
block — exists **only in the uncommitted working tree**. The committed
`.gitignore` at HEAD has neither `results/` nor the negations:

- `git show HEAD:.gitignore` → no `results/` line, no `!results/` lines.
- Working-tree `.gitignore` diff adds both `results/` AND the two negations
  `!results/` + `!results/**/*.json`.

Why this is dangerous: `scripts/verify_freeze_ready.py` **PASSED** when I ran it, but it
ran against the *working tree* `.gitignore` (which has the negations). Its
gate-(a) "self-heal" appends the negation and `git add -f`s the JSON — but those
edits are in the working tree and **uncommitted**. The 261 `results/*.json`
files are individually tracked at HEAD, so a `git archive HEAD` *today* does
contain them. But the moment anyone commits the working-tree `.gitignore` change
that adds the bare `results/` line **without also committing the negations in the
same commit**, every untracked `results/*` file becomes ignored.

Required fix: commit the working-tree `.gitignore` **as a single atomic change**
(the `results/` line and both `!results/` negations together), then
re-run `scripts/verify_freeze_ready.py` and `git check-ignore results/*.json`
against the committed state, and confirm `git archive <tagcandidate> | tar -t`
still contains `results/*.json`. Do NOT tag until check-ignore is
verified against the COMMITTED `.gitignore`.

### CRITICAL-2 — uncommitted data drift in `real.csv` / `fake.csv` would freeze inconsistent state

`git status` shows `real.csv` and `fake.csv` modified. This is not cosmetic:
`git diff --numstat` shows **777 of 778 rows changed in real.csv and 770 in
fake.csv** — i.e. *every data value* differs from HEAD (e.g. `2020-01-01` log
return `0.047086746` → `0.05537796`). These are the log-return reference samples
the notebook and analysis consume.

This is unintended drift of the headline datasets. Freezing now would either
(a) tag HEAD's old CSVs while the working notebook/results reflect the new ones,
or (b) commit the new CSVs that may not match the numbers in
`results/*.json` that the provenance gate certified. Either way the
DOI'd archive is internally inconsistent.

Required fix: the project owner must decide which CSV version is canonical,
ensure it is consistent with the certified provenance JSON, and either commit or
revert the change deliberately — not freeze it by accident.

### CRITICAL-3 — `docs/release.md` does not exist

The 14-07 plan `must_haves.artifacts` requires `docs/release.md`
(min 30 lines) recording tag SHA, reserved version + concept DOI, the
check-ignore result, and reproduce steps. `ls docs/` confirms the file
is **absent** (no tracked copy, no untracked copy). `scripts/verify_freeze_ready.py`
does NOT check for it, so the gate passing does not cover this. The release
record mandated by the plan does not yet exist.

Required fix: author `docs/release.md` per the 14-07 spec (Task 3) and
commit it into the tagged tree.

---

## HIGH severity

### HIGH-1 — `results/baselines/runs/` (47 MB, 250 files) is untracked

`results/baselines/runs/wgan_lstm/{A,B}/{42..46}/` contains
`metrics.json`, `samples.npy`, `config.yaml`, `inverse_kwargs.npz`,
`checkpoint.pt` per seed. The aggregate `results/baseline_comparison.json`
(tracked) and `reconciliation_note.md` are derived from these runs. Freezing
without them ships a DOI archive whose classical-baseline comparison numbers
cannot be reproduced from the deposited artifacts. The per-run `metrics.json`
files at minimum belong in the tag (consistent with D-14-22 "provenance backbone
must ship"). The 50 `checkpoint.pt` files inside it (~2 MB each) are below the
25 MB `LARGE_CKPT_BYTES` threshold but inflate the archive — owner should decide
whether to commit metrics+config only or the full runs. **At minimum, commit the
`metrics.json` / `config.yaml` files.**

### HIGH-2 — No `.tex` manuscript files are tracked at HEAD

`git ls-files '*.tex'` returns zero results. D-14-21 (quoted in the 14-07 plan
interfaces) states the tag "includes `revision/` ... + `.tex` reference files".
The manuscripts `main (4) copy.tex` and `supp_material.tex` are present only as
untracked working-tree files. The plan's Task 3 also treats `main (4) copy.tex`
line 292 as the anchor for the reserved-DOI Data Availability edit. If the
intent is for the DOI deposit to be self-contained with the manuscript source,
these `.tex` files must be committed (note the literal space in the filename
`main (4) copy.tex` — quote it). If the manuscript is intentionally deposited
separately on Zenodo, document that decision in `release.md`. Either way this is
an unresolved scope gap.

### HIGH-3 — `scripts/verify_freeze_ready.py` validates the working tree, not the tag candidate

The gate is correct in logic but operates on the live working tree: gate-(a)
self-heals `.gitignore` and `git add -f`s files into the index, gate-(b) reads
working-copy `docs/*.md`, gate-(c) reads `git ls-files`. None of this
proves the *committed* tree that will be tagged satisfies the invariants. A
green run today does NOT certify the tag. The gate must be re-run **after** the
pre-freeze commit and **before** `git tag`, and ideally extended to also assert
`git status --porcelain` is empty (clean tree) so it cannot pass over a dirty
working tree like the current one.

---

## MEDIUM severity

### MEDIUM-1 — 18 ablation `checkpoint.pt` files (~2 MB each, ~36 MB) ship in the tag archive

`results/transform_ablation/{runs,_smoke_100ep_archive}/**/checkpoint.pt`
— 18 tracked `*.pt` files at ~2,011,893 bytes each. They are tracked despite the
`.gitignore *.pt` rule because they were `git add -f`'d by an earlier wave;
tracked files override `.gitignore`. `scripts/verify_freeze_ready.py` deliberately
tolerates them (all below the 25 MB `LARGE_CKPT_BYTES` threshold) and the
destructive-git prohibition means this plan won't delete them. Net effect: the
DOI archive carries ~42 MB of checkpoints (these 18 + the 6 MB
`best_checkpoint.pt`). Not a blocker, but the archive is heavier than D-14-21's
"large checkpoints referenced by hash, not committed" intent implies. Acceptable
to ship; flag for owner awareness.

### MEDIUM-2 — `results/phase4_validation.json` modified in working tree

`git status` shows `results/phase4_validation.json` modified (83-line diff). The
top-level `results/` directory is *not* tracked in bulk, but this specific file
is tracked. Like the CSVs, this is uncommitted drift. The file is in the
phase-4 (`results/`) area, not `results/`, so it is likely not on the
provenance-gate path, but the owner should confirm whether the change is
intended before it is committed or reverted.

### MEDIUM-3 — `git archive HEAD` total is ~112 MB

A `git archive HEAD` tars to 112 MB. Within Zenodo limits (50 GB) but large for
a code+provenance deposit; dominated by checkpoints, `.npy` sample arrays, and
`.planning/` history. Not a blocker. Consider whether `.planning/` (full GSD
planning history) belongs in a public DOI deposit — D-14-21 explicitly includes
it, so this is a documented decision, but worth a final confirmation.

---

## LOW severity

### LOW-1 — `.gitignore` `*.csv` line has a leading space (latent footgun)

Working-tree `.gitignore` line 40 is ` *.csv` (one leading space), changed from
`# *.csv`. A leading space makes the pattern ` *.csv` rather than `*.csv`, so
`git check-ignore fake.csv` → not-ignored, and `data.csv` / `fake.csv` /
`real.csv` remain tracked — which is the *desired* outcome here (the CSVs must
ship). But it is accidental: the line reads as if CSVs are ignored when they are
not. Harmless for this freeze (the CSVs correctly ship) but should be cleaned up
to `# *.csv` to avoid future confusion.

### LOW-2 — Untracked junk that must NOT be committed

The following untracked items are scratch/duplicate/derived and should be left
out of the freeze (add to `.gitignore` or just don't `git add` them):
`qgan_pennylane copy.ipynb` (4.1 MB duplicate), `main (4) copy.tex` is a "copy"
naming but is the actual manuscript — see HIGH-2, `datasets.zip` (932 KB,
redundant with `datasets/`), `amp` (0-byte empty file), `circuit_diagram.png`
(regenerable), `QGAN_Review_Response_Plan.md.pdf` (export artifact),
`Final Results from 2000 epochs - IQP:SEL circuit/` (1.8 MB — confirm not a
numbers source before discarding), `.claude/` (agent worktree metadata),
`.planning/phases/12-sensitivity-analysis/12-PATTERNS.md` (stray planning file —
owner should decide if it belongs in `.planning/`). `datasets/` (1.8 MB
synthetic CSVs + overlay PNG) — confirm with owner whether these synthetic
samples are a published artifact; if so commit, else leave out.

---

## INVARIANT CHECK RESULTS

| Check | Result |
|---|---|
| `git check-ignore results/*.json` (working tree) | empty (not ignored) — PASS, but only because uncommitted negations exist (CRITICAL-1) |
| `results/*.json` tracked at HEAD | 261 files — PASS |
| `data.csv` tracked | YES — PASS |
| `qgan_env/` tracked | NO (gitignored) — PASS |
| Large checkpoints (>25 MB) tracked | NONE — PASS (but see MEDIUM-1: 18 sub-threshold .pt ship) |
| LICENSE at HEAD | PRESENT — PASS (working-tree deletion uncommitted — see CRITICAL re: pre-freeze commit) |
| Committed secrets / ZENODO_TOKEN | NONE found — PASS |
| `.tex` manuscripts tracked | NONE — FAIL vs D-14-21 (HIGH-2) |
| `scripts/verify_freeze_ready.py` exists | YES (9.7 KB, well-formed, `raise AssertionError` idiom) — PASS |
| `scripts/verify_freeze_ready.py` run result | exit 0, all three gates pass — PASS (but validates working tree, not tag — HIGH-3) |
| `docs/release.md` exists | NO — FAIL (CRITICAL-3) |
| Tag `v2.0-revision` exists | NO — not yet cut (expected; this is the gate before cutting) |

---

## RECOMMENDED PRE-FREEZE SEQUENCE

1. Revert the unintended working-tree deletion: `git checkout -- LICENSE`.
2. Resolve CRITICAL-2: owner decides canonical `real.csv`/`fake.csv` (and
   `results/phase4_validation.json`); confirm consistency with the certified
   `results/*.json`; commit or revert deliberately.
3. Commit the `.gitignore` change as ONE atomic commit containing the `results/`
   line AND both `!results/` negations together (CRITICAL-1); fix the
   ` *.csv` leading space to `# *.csv` (LOW-1) in the same commit.
4. Commit `results/baselines/runs/` metrics/config artifacts (HIGH-1).
5. Decide and act on `.tex` manuscript inclusion (HIGH-2).
6. Author and commit `docs/release.md` (CRITICAL-3).
7. Re-run `python scripts/verify_freeze_ready.py` against the now-clean,
   committed tree; confirm `git status --porcelain` is empty (HIGH-3).
8. Verify `git check-ignore results/*.json` is empty against the
   committed `.gitignore`, then cut `git tag -a v2.0-revision`.
9. Confirm `git archive v2.0-revision | tar -t` contains
   `results/*.json`, `data.csv`, and `LICENSE`.

---

FREEZE VERDICT: BLOCK
