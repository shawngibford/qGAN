# Peer Review R4 — Agent 3 (Results & Provenance) Review

**Scope:** Re-run metric emitters, verify byte-stable reproduction against committed
JSONs, run the number-provenance gate over all paper-facing docs, verify data_hash and
checkpoint-SHA256 consistency. Last gate before the irreversible Zenodo DOI freeze at
git tag `v2.0-revision`.

**Reviewer environment:** isolated git worktree
`/Users/shawngibford/dev/phd/qGAN/.claude/worktrees/agent-a206af7dccc0a23fa`,
synced to main-repo HEAD `8180a5e` (see HIGH-1 below). Re-emit interpreter: a fresh
venv `/tmp/qgan_env_r4` built from `revision/requirements-pinned.txt`
(numpy 2.3.4, scipy 1.16.2, torch 2.9.0, pennylane 0.43.0, PyYAML 6.0.3,
statsmodels 0.14.5, Python 3.11.14).

---

## 1. Emitter reproduction

All re-emitted JSONs were diffed against the git-committed versions snapshotted in
`/tmp/peer-review-r4/committed/` BEFORE re-running.

| Emitter | Output JSON | Result |
|---|---|---|
| `run_matched2000_dualscale.py` | `matched2000_dualscale.json` | **BYTE-IDENTICAL** (2576 rows, 560 aggregates) |
| `run_distribution_emd.py` | `distribution_emd.json` | **BYTE-IDENTICAL** (90 rows, 18 aggregates) |
| `run_welch_aggregator.py` | `welch_pairwise.json` | **BYTE-IDENTICAL** (40 pairs) |
| `run_model_info.py` | `model_info.json` | **BYTE-IDENTICAL** (10 model records); no doc drift |
| `run_canonical_headline.py` | `headline_canonical.json` | **BYTE-IDENTICAL** (56 rows) |

`git status --short revision/results/` is **clean** after all 5 re-emits — zero
working-tree drift. `git diff --stat revision/core/` is **empty** (D-11-10 / D-14-22
core byte-freeze upheld).

No emitter diverged. No within-tolerance-only matches; every one is exact-byte.

The R3-CR-1 fix (distribution_emd density=False + shared edges) and R3-CR-2 fix
(matched2000 un-standardize-fake LR-EMD) are present in the emitter source and
reproduce the committed numbers exactly — no regression.

## 2. Number-provenance gate

`revision/verify_number_provenance.py` runs one `--target` doc at a time. Ran it over
**all 10 docs** in `revision/docs/`:

| Doc | Literals | Result |
|---|---|---|
| circuit_atlas.md | 18 | PASS |
| completeness_sweep_manifest.md | 47 | PASS |
| dataset_stats.md | 5 | PASS |
| methods_full.md | 105 | PASS |
| paper_blocks_framing.md | 23 | PASS |
| paper_blocks_refs_methods.md | 49 | PASS |
| peer_review_remediation.md | 105 | PASS |
| reconciliation_note.md | 67 | PASS |
| reviewer_response.md | 83 | PASS |
| training_protocol.md | 18 | PASS |

**The provenance gate PASSES for every paper-facing doc.** Every numeric literal
resolves to a `revision/results/*.json` value (schema v2.1, negative-sign-aware
lookbehind). The v2.1 differential test (`--differential-test`) also PASSES — the
R2-prov-HIGH-1 sign-flip false-positive is correctly guarded.

## 3. data_hash and checkpoint-SHA256 consistency

- **data_hash:** All **27** `revision/results/**.json` files carrying a `data_hash`
  field hold the identical value `91e447d4624e25b3`. Zero divergent values. The
  matched2000 emitter additionally re-verified, at emit time, that all 45
  `matched2000/runs/<model>/<seed>/config.yaml` files AND `headline_canonical.json`
  carry `91e447d4624e25b3` (its explicit-raise D-14-16 gate passed). The data_hash was
  also independently recomputed from `data.csv` via `load_and_preprocess` and equals
  `91e447d4624e25b3`.
- **checkpoint SHA256:** `canonical_config_lock.json#checkpoint_sha256` =
  `f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082`. The actual
  SHA256 of `revision/checkpoints/best_checkpoint.pt` matches exactly; the worktree
  checkpoint is byte-identical to the main-repo copy. `run_canonical_headline.py`'s
  in-process checkpoint-identity gate (T-14-14) passed.
- `headline_canonical.json#checkpoint_emd` (0.083843...) equals
  `canonical_config_lock.json#checkpoint_emd` exactly — these are the training-time
  checkpoint-selection EMD carried verbatim. They are intentionally distinct from the
  headline's *recomputed* dual-scale OD-EMD row (0.023072) — separate fields by
  design, not a discrepancy.

---

## Findings

### HIGH-1 — Worktree was provisioned at a stale commit (process issue; resolved by reviewer)
The worktree was created checked out at commit `c82169c0` ("data: add fake.csv and
real.csv log-return reference samples"), **319 commits behind** the main-repo HEAD
`8180a5e`. At `c82169c0` the entire `revision/` tree contained only 12 files — none of
the emitters, results JSONs, or docs under review existed. `git log` inside the
worktree misleadingly reported the main branch's HEAD.
**Resolution:** the reviewer synced the worktree to `8180a5e` (`git checkout 8180a5e`)
before any verification; all results above are against the correct frozen tree.
**Action for freeze:** confirm the DOI is minted from `8180a5e` (or its `v2.0-revision`
tag), NOT from whatever a fresh worktree defaults to. The `v2.0-revision` tag does not
yet exist — only `v1.0` is tagged. Tagging must target `8180a5e`.

### MEDIUM-1 — `requirements-pinned.txt` is incomplete: missing `fastdtw` and `pandas`
`revision/core/eval.py:84` imports `fastdtw` (required by `compute_dtw`, exercised by
`run_matched2000_dualscale.py`), but `fastdtw` is absent from
`revision/requirements-pinned.txt` and from `framework_versions.json#packages`. A
clean install from the pinned file alone cannot reproduce `matched2000_dualscale.json`
(DTW rows) — it fails with `ModuleNotFoundError: No module named 'fastdtw'`.
Additionally, `pandas` is unpinned: pip resolved `pandas==3.0.3`, which **breaks**
`statsmodels==0.14.5` import (`TypeError: deprecate_kwarg() missing 1 required
positional argument` — pandas 3.0 changed the `deprecate_kwarg` signature).
Reproduction required pinning `pandas<3.0` (used 2.3.3). A reader following the
documented `pip install -r revision/requirements-pinned.txt` recipe in
`methods_full.md §4.1` will hit a hard import failure on `compute_acf`.
**Recommended fix before freeze:** add `fastdtw==<version>` and `pandas==<2.x>` to
`requirements-pinned.txt` and `framework_versions.json`. This does not affect any
*number* (all re-emits are byte-identical) — it is a reproducibility-instructions gap.

### LOW-1 — Welch OD strong-claim thresholds clear by thin margins
`welch_pairwise.json` strong-claim gate: OD floor Welch p = 0.36521 vs threshold
`> 0.36` (margin 0.00521); OD ceiling |Cohen d| = 0.64417 vs threshold `<= 0.65`
(margin 0.00583). Both pass and reproduce byte-stably, so there is no reproduction
risk. Flagged only as informational: the headroom is small and the thresholds appear
chosen to fit the data. Not a freeze blocker.

### LOW-2 — `verify_freeze_ready.py` sweeps only 3 of the 10 provenance-bearing docs
`verify_freeze_ready.py` (lines 65-67) runs the provenance gate over only
`paper_blocks_framing.md`, `paper_blocks_refs_methods.md`, `reviewer_response.md`. The
other 7 docs (methods_full, peer_review_remediation, reconciliation_note, etc.) also
contain JSON-resolved literals (this reviewer verified all 10 pass manually). Not a
correctness issue — just incomplete automated coverage in the freeze-ready check.

---

## Summary

- **Emitters:** 5/5 reproduce **byte-identical** (matched2000_dualscale,
  distribution_emd, welch_pairwise, model_info, canonical_headline). Zero divergence.
- **Provenance gate:** **PASSED** for all 10 paper-facing docs + differential test.
- **data_hash:** consistent `91e447d4624e25b3` across all 27 carrying JSONs.
- **Checkpoint SHA256:** matches the lock exactly.
- No CRITICAL findings. No regression of R3-CR-1 / R3-CR-2. Core tree byte-frozen.
- The MEDIUM finding is a packaging/instructions gap (missing `fastdtw`, unpinned
  `pandas`) that does not alter any published number but would block a clean
  third-party reproduction from the documented recipe.

FREEZE VERDICT: GO-WITH-FIXES
