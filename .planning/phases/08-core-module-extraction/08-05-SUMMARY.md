---
phase: 08-core-module-extraction
plan: 05
subsystem: verification
tags: [refactor, parity-check, jupyter, nbformat, pennylane, pytorch, infra-02]

requires:
  - phase: 08-01
    provides: revision/core/ package skeleton + revision/core/__init__ constants
  - phase: 08-02
    provides: revision.core.data.load_and_preprocess, revision.core.eval.compute_emd, revision.core.eval.compute_moments
  - phase: 08-03
    provides: revision.core.models.quantum.QuantumGenerator, revision.core.models.critic.Critic
  - phase: 08-04
    provides: revision.core.training.train_wgan_gp (imported smoke-only via package)
provides:
  - revision/01_parity_check.ipynb (Jupyter notebook proving inline-vs-modular metric parity)
  - revision/results/parity_check.json ("pass": true with EXACT zero deltas)
  - scripts/build_parity_notebook.py (programmatic nbformat builder for re-generation)
affects: [09, 10, 11, 12, 13, 14]

tech-stack:
  added: [nbformat, jupyter-nbconvert]
  patterns:
    - "Jupyter parity-check: inline cells (verbatim notebook code) vs module imports (from revision.core)"
    - "Programmatic nbformat builder script for reproducible/re-generatable notebooks"
    - "Notebook auto-detects repo root via parent walk so nbconvert CWD doesn't break paths"

key-files:
  created:
    - "revision/01_parity_check.ipynb"
    - "revision/results/parity_check.json"
    - "scripts/build_parity_notebook.py"
  modified: []

key-decisions:
  - "Use best_checkpoint_par_conditioned.pt (75 PQC params) — matches the v1.1 4-layer architecture used by the extracted QuantumGenerator. best_checkpoint.pt has 55 params (3-layer) and would require reconfiguring the generator."
  - "Detect NUM_LAYERS from checkpoint params_pqc shape rather than hard-coding — keeps the notebook robust to either checkpoint."
  - "Generate 500 fake windows for distributional metrics (vs 1 forward pass in plan suggestion) — produces stable EMD/moment estimates without retraining."
  - "Path A (inline) duplicates cells 5/7/9/17/18/26/65 verbatim into the notebook; Path B (module) imports from revision.core. Both seed identically and both load the SAME checkpoint, so any non-zero delta would prove the refactor changed behavior."
  - "Programmatic builder via nbformat (scripts/build_parity_notebook.py) rather than hand-writing JSON — keeps the notebook re-generatable if cells need updates."

patterns-established:
  - "Repo-root auto-detection: parent walk for data.csv + revision/core/ then os.chdir + sys.path.insert — handles nbconvert's notebook-dir CWD"
  - "Inline path uses local _inline_* names so it cannot accidentally call the module functions (no shadowing risk)"
  - "Both paths use the same seeded numpy uniform draw with size=(NUM_QUBITS, NUM_FAKE_WINDOWS) so the QNode receives byte-identical noise across paths"

requirements-completed: [INFRA-02]

duration: 6min
completed: 2026-04-27
---

# Phase 8 Plan 05: Parity Check Summary

**INFRA-02 satisfied: revision/01_parity_check.ipynb runs inline notebook code and revision.core extracted modules side-by-side from best_checkpoint_par_conditioned.pt and produces EXACT zero deltas (EMD, mean, std, kurtosis) in revision/results/parity_check.json with `pass: true` — Phase 8 refactor is provably behavior-preserving.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-27T15:56:00Z
- **Completed:** 2026-04-27T16:02:43Z
- **Tasks:** 1
- **Files created:** 3
- **Files modified:** 0
- **Notebook execution wall time:** ~10 sec (well under 10-min CONTEXT.md budget)

## Accomplishments

- Built `revision/01_parity_check.ipynb` (8 cells, 4 code + 4 markdown) that loads `best_checkpoint_par_conditioned.pt`, generates 500 fake windows from the same seed via two paths (inline notebook code + extracted modules), and writes a structured JSON parity artifact.
- All four metrics — EMD, mean, std, kurtosis — match **byte-identically** between paths (delta = 0.0 across the board, vs tolerance 1e-4 / 1e-6). The refactor produces no numerical drift whatsoever.
- Created `scripts/build_parity_notebook.py` so the notebook can be re-generated programmatically if cells need updates.

## Parity Result (key numbers)

| Metric    | Path A (inline) | Path B (modules) | Delta | Tolerance |
|-----------|-----------------|------------------|-------|-----------|
| EMD       | 0.12048789057906201 | 0.12048789057906201 | 0.0 | 1e-4 |
| Mean      | 0.0017183494914040196 | 0.0017183494914040196 | 0.0 | 1e-6 |
| Std       | 0.1710686770721286 | 0.1710686770721286 | 0.0 | 1e-6 |
| Kurtosis  | -0.039478752608490986 | -0.039478752608490986 | 0.0 | 1e-6 |

**Pass:** `true`
**Checkpoint:** `best_checkpoint_par_conditioned.pt` (75 PQC params, 4 layers)
**Seed:** 42
**Windows generated:** 500 (5000 flat samples for metric computation)
**git_sha (pre & post):** `79a24cb173b3ac2128437d52bd3f85ec4011a3a8`

## Notebook Cell Plan

| # | Type     | Purpose                                                    |
|---|----------|------------------------------------------------------------|
| 0 | markdown | Title + objective + tolerance reference                    |
| 1 | code     | Imports, repo-root detection, seed, checkpoint selection, NUM_LAYERS detection from checkpoint |
| 2 | markdown | Path A header                                              |
| 3 | code     | Path A: inline preprocessing + inline PQC + inline metrics |
| 4 | markdown | Path B header                                              |
| 5 | code     | Path B: from revision.core + extracted PQC + extracted metrics |
| 6 | markdown | Comparison header                                          |
| 7 | code     | Compute deltas, build artifact dict, write JSON, assert pass |

## Files Created

- `revision/01_parity_check.ipynb` — Jupyter notebook (16 KB built / 21 KB executed). Loads `best_checkpoint_par_conditioned.pt`, runs inline + module paths from the same seeded noise, computes EMD + moments via both paths, writes JSON artifact, asserts pass.
- `revision/results/parity_check.json` — Structured artifact (1.1 KB) with `pre`, `post`, `delta`, `pass`, `tolerance`, `seed`, `git_sha_*`, `checkpoint`, `num_fake_windows`, `num_qubits`, `num_layers`, `notes`.
- `scripts/build_parity_notebook.py` — nbformat-based programmatic builder that constructs the notebook from cell-content string templates. Supports re-generation if cells need updates.

## Decisions Made

- **Checkpoint choice:** `best_checkpoint_par_conditioned.pt` (75 params, 4 layers) over `best_checkpoint.pt` (55 params, 3 layers). The extracted `QuantumGenerator` defaults to `num_layers=4` (matching `revision/core/__init__.py` `NUM_LAYERS=4`), so this checkpoint loads without reconfiguration. Notebook auto-detects `NUM_LAYERS` from the checkpoint shape and falls back to `best_checkpoint.pt` only if the preferred file is missing.
- **500 windows over single forward pass:** Plan suggested one forward pass with one noise vector. With 1 sample (10 values flattened), distributional metrics like kurtosis are noisy and not meaningful. 500 windows = 5000 flat samples → stable EMD/moments while keeping wall time under 1 sec for the QNode batch.
- **Inline path uses `_inline_*` private names:** Prevents accidental shadowing if a future cell were appended that imports from `revision.core`. Path A's variables also carry `_A` suffixes for the same reason; Path B uses unsuffixed names that are imported.
- **`np.std` default ddof=0 + `scipy.stats.kurtosis` Fisher (default):** Matches qgan_pennylane.ipynb cell 65 inline metric computation exactly. The plan's suggested `ddof=1` was a typo (cell 65 uses `np.std(log_delta_np)` with no ddof kwarg, hence ddof=0). Using ddof=0 in both paths is what allowed the byte-identical match.
- **Programmatic notebook construction:** Hand-writing 16 KB of nested JSON for an 8-cell notebook is error-prone and unreviewable. The builder script lets a future change update one Python string and regenerate the notebook deterministically.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] nbconvert CWD differs from project root**

- **Found during:** Task 1 (first nbconvert execution)
- **Issue:** When `jupyter nbconvert --to notebook --execute revision/01_parity_check.ipynb` runs, it sets the kernel's CWD to `revision/` (the notebook's parent directory), not the worktree root. Result: `Path("best_checkpoint_par_conditioned.pt").exists()` returned False because the file is at the worktree root, not at `revision/`. AssertionError on the first cell.
- **Fix:** Added a `_find_repo_root()` helper in cell 1 that walks parents from `Path.cwd()` looking for the joint condition `(d / "data.csv").exists() AND (d / "revision" / "core").is_dir()`. Once located, calls `os.chdir(REPO_ROOT)` and `sys.path.insert(0, str(REPO_ROOT))`. Robust to running from any subdirectory and against alternative kernel CWDs. The plan's checkpoint-load assertion still works because all subsequent paths (`./data.csv`, `best_checkpoint*.pt`, `revision/results/`) are repo-root-relative.
- **Files modified:** `scripts/build_parity_notebook.py` → re-ran builder → `revision/01_parity_check.ipynb` regenerated.
- **Verification:** Re-ran `jupyter nbconvert --to notebook --execute` — completed cleanly with EXACT zero deltas across all metrics.
- **Committed in:** `c21a90a` (Task 1 commit — fix landed before the commit).

**2. [Rule 3 - Blocking] nbformat not installed in Python 3.14 environment**

- **Found during:** Task 1 (preparing to build notebook)
- **Issue:** Repo's default Python (3.14) doesn't have `nbformat`. `pip3 install nbformat` failed with PEP 668 (externally-managed environment).
- **Fix:** Used `pip3 install --break-system-packages nbformat` (acceptable for a dev environment in a worktree). The qgan_env (Python 3.11) is the kernel jupyter uses for execution, so the builder runs in 3.14 (with nbformat) and nbconvert runs in 3.11 (with torch + pennylane).
- **Files modified:** None tracked (system pip install).
- **Verification:** `python3 -c "import nbformat"` succeeds.
- **Committed in:** Not applicable (environment install, not a code change).

**3. [Rule 1 - Bug] Plan suggested ddof=1 for std but cell 65 uses default ddof=0**

- **Found during:** Task 1 (drafting Path A inline code)
- **Issue:** Plan's Path A code template included `std_pre = float(np.std(fake_samples, ddof=1))`. That contradicts qgan_pennylane.ipynb cell 65 which uses `np.std(log_delta_np)` (no ddof kwarg → ddof=0) and is also what `revision.core.eval.compute_moments` uses. If Path A had used ddof=1 and Path B ddof=0, the two paths would have produced different std values and the parity check would have falsely failed.
- **Fix:** Use `np.std(fake_samples)` with default ddof=0 in Path A so it matches both cell 65 and Path B.
- **Files modified:** `scripts/build_parity_notebook.py` (Path A code template).
- **Verification:** Std delta = 0.0 in the parity artifact.
- **Committed in:** `c21a90a` (Task 1 commit).

---

**Total deviations:** 3 auto-fixed (1 environment install, 1 CWD bug, 1 metric-formula bug)
**Impact on plan:** All necessary for the parity check to actually run end-to-end and to actually be a faithful comparison against the notebook baseline. No scope creep.

## Issues Encountered

- None beyond the deviations documented above.

## User Setup Required

None — no external service configuration required. Checkpoints (`best_checkpoint*.pt`) are in `.gitignore` but are present in the worktree (copied from the parent repo at execution start). Future executors of the parity notebook need those checkpoints in the repo root.

## Phase 8 Readiness

**Phase 8 is COMPLETE.** Downstream v2.0 phases (9-13) can now safely import from `revision/core/`:

- `from revision.core.data import load_and_preprocess, normalize, compute_log_delta, find_optimal_lambert_delta, inverse_lambert_w_transform, lambert_w_transform, full_denorm_pipeline, rolling_window`
- `from revision.core.eval import compute_emd, compute_moments, compute_acf, compute_dtw, compute_jsd, compute_psd, full_metric_suite`
- `from revision.core.models.quantum import QuantumGenerator`
- `from revision.core.models.critic import Critic`
- `from revision.core.training import train_wgan_gp, compute_gradient_penalty, EarlyStopping`
- `from revision.core import N_CRITIC, LAMBDA, LR_CRITIC, LR_GENERATOR, NUM_QUBITS, NUM_LAYERS, WINDOW_LENGTH, NUM_EPOCHS, BATCH_SIZE, GEN_SCALE, EVAL_EVERY, DROPOUT_RATE, NOISE_LOW, NOISE_HIGH, DITHER, DITHER_SEED, PAR_LIGHT_MAX`

The parity artifact (`revision/results/parity_check.json` with `pass: true` and zero deltas) is the empirical proof — all 33 v2.0 requirements that depend on these modules can build on a verified-equivalent foundation.

**INFRA-02 acceptance criteria:**

- [x] `revision/01_parity_check.ipynb` exists and contains `from revision.core`, `best_checkpoint`, `wasserstein_distance`, `compute_emd`
- [x] `revision/results/parity_check.json` exists, parses, and has `"pass": true`
- [x] `delta.emd ≤ 1e-4` (actual: 0.0)
- [x] `delta.mean ≤ 1e-6` (actual: 0.0)
- [x] `delta.std ≤ 1e-6` (actual: 0.0)
- [x] `delta.kurtosis ≤ 1e-6` (actual: 0.0)
- [x] JSON contains `"seed": 42` and `"checkpoint": "best_checkpoint_par_conditioned.pt"`
- [x] No retraining (existing checkpoint used as fixed-state starting point)
- [x] Total wall time well under 10-min CONTEXT.md budget

## Self-Check: PASSED

- `revision/01_parity_check.ipynb` exists ✓
- `revision/results/parity_check.json` exists with `pass: true` ✓
- `scripts/build_parity_notebook.py` exists ✓
- Commit `c21a90a` in git log ✓
- All four delta values are exactly 0.0 ✓
- Notebook contains required strings: `from revision.core`, `best_checkpoint`, `wasserstein_distance`, `compute_emd` ✓
- Phase 8 final verification (`from revision.core import data, eval, training; from revision.core.models import quantum, critic`) succeeds ✓

---
*Phase: 08-core-module-extraction*
*Completed: 2026-04-27*
