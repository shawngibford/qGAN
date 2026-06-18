---
phase: 14-paper-revision-release-freeze
plan: 19
subsystem: release-freeze
tags: [freeze, gitignore, hygiene, verify-gate, zenodo-precondition]
requires:
  - "14-17: revised .tex manuscripts integrated into the committed tree"
  - "14-18: claims recalibration (no .tex touched)"
provides:
  - "Clean, deliberately-committed v2.0-revision tag-candidate tree"
  - "Atomic .gitignore: results/ exclusion + !results/ negations"
  - "Hardened verify_freeze_ready.py certifying the committed tree"
  - "Recorded freeze-candidate HEAD SHA for plan 14-07"
affects:
  - "14-07: cuts v2.0-revision tag from the recorded freeze-candidate HEAD and runs the hardened gate"
tech-stack:
  added:
    - "fastdtw==0.3.4 (pinned — used by core/eval.py DTW)"
    - "pandas<3.0 (pinned — pandas 3.0 breaks statsmodels)"
  patterns:
    - "Atomic .gitignore commit: exclusion + negations land together"
    - "Freeze gate certifies the committed tree (git ls-files / HEAD:.gitignore), not the working tree"
key-files:
  created:
    - ".planning/phases/14-paper-revision-release-freeze/14-19-SUMMARY.md"
  modified:
    - ".gitignore"
    - "requirements-pinned.txt"
    - "verify_freeze_ready.py"
decisions:
  - "D-14-19-A: real.csv/fake.csv/phase4_validation.json/qgan_pennylane.ipynb working-tree drift reverted to HEAD (owner: drift unintended)"
  - "D-14-19-B: .planning/ ships in the Zenodo deposit (owner: honors D-14-21)"
  - "D-14-19-C: baseline run binaries (*.npy/*.npz/*.pt) gitignored; only metrics.json + config.yaml tracked"
  - "D-14-19-D: manuscript bibliography is Overleaf-side — bib.bib is genuinely absent (Agent 5 L-3)"
metrics:
  duration: "~20 min"
  completed: 2026-05-22
  tasks: 3
  files: 105
---

# Phase 14 Plan 19: Pre-Freeze Working-Tree Hygiene & Hardened Freeze Gate Summary

Established a clean, deliberately-committed `v2.0-revision` tag-candidate tree and hardened `verify_freeze_ready.py` to certify the committed tree rather than a dirty working tree — closing SYNTHESIS C4, C5, C6, H5, H6, H7, M4, M10, M11, M12 so plan 14-07 can cut the tag and mint the Zenodo DOI against a trustworthy tree at a known commit.

## Freeze Candidate

**Freeze-candidate HEAD: `651832349cbca2316638db9fc8e50e7e3f20968f` (`6518323`)**

This is the post-14-19 committed HEAD — the commit that contains all of 14-17 (manuscript revision), 14-18 (claims recalibration) and 14-19 (hygiene + hardened gate). **Plan 14-07 MUST cut `v2.0-revision` from THIS exact commit** — not from `8180a5e`, not from `7aa3c58`, and not from any review worktree commit (SYNTHESIS H5 / Agent 3 HIGH-1: the review worktrees were provisioned at stale commits; the DOI must be minted from the current verified freeze-candidate HEAD).

`./qgan_env/bin/python verify_freeze_ready.py` was run against this committed HEAD and confirms it evaluates the committed tree:

- `(0)` clean-tree OK — `git status --porcelain` is empty
- `(a)` gitignore/archive OK — `git check-ignore` empty; 899 tracked paths under `revision/results`
- `(b)` number-provenance OK — all 3 paper-blocks files PASS (23 + 49 + 88 numeric literals resolve)
- `(c)` tag-scope OK — `qgan_env/` not tracked, `data.csv` tracked, no large checkpoint
- `(d)` release.md — **expected FAIL**: `docs/release.md` is plan 14-07's deliverable; the assertion fails until 14-07 authors it. This is the intended ordering guard.

Every gate except `(d)` passes against `6518323`. The `(d)` failure is by design and resolves when 14-07 runs.

## What Was Built

### Task 1: Owner-Decision Checkpoint (resolved by owner before this execution)

1. **Canonical CSVs** — `real.csv`/`fake.csv` reverted to HEAD; the working-tree drift was unintended and the HEAD versions are consistent with the certified `results/*.json`.
2. **phase4_validation.json** — reverted to HEAD; the 2026-05-05 re-run drift is not part of the v2.0 freeze.
3. **.planning/ inclusion** — `.planning/` (full GSD planning history) ships in the Zenodo deposit, honoring D-14-21.

### Task 2: Pre-Freeze Hygiene (commit `b22107d`)

- **LICENSE restored** — `git checkout -- LICENSE` reverted the working-tree-only deletion (T-14-31).
- **requirements-pinned.txt** — added `fastdtw==0.3.4` (used by `core/eval.py` DTW) and `pandas<3.0` (pandas 3.0 breaks statsmodels). Header comments preserved.
- **.gitignore atomic commit** — fixed the ` *.csv` leading-space typo back to `# *.csv` (CSVs stay tracked); the `results/` exclusion and BOTH `!results/` negations land in this one commit (T-14-28). `git check-ignore results/*.json` against the committed `.gitignore` returns empty — no provenance JSON ignored.
- **Baseline artifacts tracked** — 50 `metrics.json` + 50 `config.yaml` under `results/baselines/runs/` force-staged with `git add -f`. The 50 `*.npy`, 60 `*.npz`, 40 `*.pt` large binaries are NOT committed (D-14-21 / Agent 6 HIGH-1).
- **Drift reverted** — `real.csv`, `fake.csv`, `results/phase4_validation.json`, `qgan_pennylane.ipynb` reverted to HEAD per the Task-1 owner decisions.

### Task 3: Revised .tex Confirmation + Hardened Gate (commit `6518323`)

- **.tex tracking** — `main (4) copy.tex` and `supp_material.tex` are already git-tracked (committed by plan 14-17). Confirmed `git show HEAD:"main (4) copy.tex"` contains the 14-17 `Circuit Design Rationale` marker. The unchanged `.tex` files were NOT re-committed — Task 3(a) is satisfied by 14-17's commit.
- **verify_freeze_ready.py hardened** — added `gate_zero_clean_tree()` asserting `git status --porcelain` is empty; added `gate_d_release_md()` asserting `docs/release.md` exists; `_check_ignored_json()` now reads `git ls-files` (committed tree) and `gate_a_gitignore_archive()` reads `HEAD:.gitignore`. The self-heal block was **removed** — self-healing mutates `.gitignore` and would dirty the tree, contradicting gate 0; the gate now fails hard. All assertions use the explicit `raise AssertionError` idiom (`python -O`-proof). The pre-existing gate (b)/(c) explicit-raise checks are semantically unchanged.

### Planning Docs (commit `4592bfb`)

To make the tracked `.planning/` tree complete before the freeze (per D-14-21), committed the untracked `14-17-PLAN.md`, `14-18-PLAN.md`, `14-19-PLAN.md`, the `peer-review-r4/` review set (7 files), and `12-PATTERNS.md`.

## Working-Tree Cleanliness

The hardened gate's `gate_zero_clean_tree()` requires `git status --porcelain` to be empty. To reach genuine cleanliness:

- Reverted out-of-scope drift (`qgan_pennylane.ipynb`, CSVs, json) per owner decisions.
- Added `.gitignore` patterns for junk that must NOT enter the tag archive (Agent 6 L-2): `.claude/`, `amp`, `circuit_diagram.png`, `datasets.zip`, `datasets/`, `qgan_pennylane copy.ipynb`, `QGAN_Review_Response_Plan.md.pdf`, `Final Results from 2000 epochs - IQP:SEL circuit/`.
- Added `.gitignore` patterns for the large baseline binaries (`results/baselines/runs/**/*.npy`, `**/*.npz`, `**/*.pt`) so the 150 large binaries do not break the clean-tree assertion.
- Committed the planning docs.

`git status --porcelain` prints nothing at plan completion — verified.

## Deviations from Plan

### Auto-fixed / Adjusted Items

**1. [Rule 3 - Blocking] Removed the gate's self-heal block**
- **Found during:** Task 3 — hardening `verify_freeze_ready.py`.
- **Issue:** The plan said to harden the gate to assert a clean working tree, but the existing `gate_a_gitignore_archive()` contained a self-heal block that appends to `.gitignore` and runs `git add -f`. A self-heal that mutates the working tree directly contradicts the new clean-tree assertion (a green run would have just dirtied the tree).
- **Fix:** Removed the self-heal block; the gate now reads the committed `HEAD:.gitignore`, verifies the negation is present, and fails hard if any provenance JSON is ignored. The atomic `.gitignore` commit in Task 2 makes self-heal unnecessary.
- **Files modified:** `verify_freeze_ready.py`
- **Commit:** `6518323`

**2. [Plan-anticipated] .tex files not re-committed**
- The plan's Task 3(a) instructed tracking + committing the `.tex` manuscripts, but plan 14-17 already committed them (verified `git show HEAD:"main (4) copy.tex"` carries the `Circuit Design Rationale` marker). Per the execution context note, the unchanged `.tex` files were NOT re-committed — Task 3(a) is satisfied.

**3. [Plan-anticipated] Junk + baseline-binary ignore patterns added in the Task-2 hygiene commit**
- The plan's Task 2 did not explicitly enumerate the junk/baseline-binary ignore patterns, but the owner decisions and the clean-tree requirement made them necessary. They were added in the same pre-freeze `.gitignore` hygiene commit (`b22107d`) — all pre-freeze `.gitignore` hygiene, consistent with the atomicity note.

### Documented Absences

- **bib.bib** — genuinely absent from the repo. The manuscript bibliography is Overleaf-side (documented decision per Agent 5 L-3). 14-07's `release.md` should note this.
- **docs/release.md** — correctly absent; it is plan 14-07's deliverable. The hardened gate's `(d)` assertion fails until 14-07 runs (intended ordering guard). Not created by this plan.

## Boundary Compliance

No tag was cut, no Zenodo deposit performed, `core/` untouched (byte-frozen D-11-10), and no numbers recomputed or retrained. This plan only established the freeze pre-conditions and hardened the gate.

## Self-Check: PASSED

- `LICENSE` present and tracked — FOUND
- `requirements-pinned.txt` contains `fastdtw` + `pandas<3.0` — FOUND
- Committed `.gitignore` contains `!results/`; `git check-ignore` empty for provenance JSON — VERIFIED
- 50 `metrics.json` + 50 `config.yaml` tracked under `results/baselines/runs/` — FOUND
- `verify_freeze_ready.py` asserts `status --porcelain` + `release.md`, reads committed tree — FOUND
- Commits `b22107d`, `4592bfb`, `6518323` exist — FOUND
- `git status --porcelain` empty — VERIFIED
- Hardened gate run against `6518323`: gates 0/a/b/c PASS, gate d fails as designed — VERIFIED
