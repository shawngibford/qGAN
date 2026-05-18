---
phase: 11-utility-evaluation
plan: 01
subsystem: testing
tags: [tstr, augmentation, soft-sensor, lstm, orlandi, eval-01, eval-04, long-form-json]

# Dependency graph
requires:
  - phase: 10-classical-baselines
    provides: "50 frozen baseline samples.npy + config.yaml data_hash + tstr real_only_baseline anchor (R2=-13.354)"
  - phase: 09.1-preprocessing-ablation
    provides: "10 frozen quantum transform_ablation samples.npy (Pipelines A/B x seeds 42-46)"
provides:
  - "revision/run_utility.py — EVAL-01 TSTR + EVAL-04 augmentation driver (verbatim reconstruct_od + train_eval_tstr reuse)"
  - "revision/results/tstr.json — TSTR R2/MAE/RMSE long-form rows + real_only_baseline anchor (R2=-13.3542, exact Phase-10 reproduction)"
  - "revision/results/augmentation.json — Orlandi-style mixing-ratio lift table (real_only/+25%/+50%/+100%/synthetic_only) with delta metrics"
affects: [12-sensitivity, 14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Driver-mirrors-run_baselines.py (CLI + frozen-artifact consume + long-form JSON)"
    - "Verbatim reuse of reconstruct_od/train_eval_tstr (no re-derivation; D-11-10 core untouched)"
    - "Repo-root resolver for worktree/arbitrary-cwd safety (RESEARCH Pitfall 6)"

key-files:
  created:
    - revision/run_utility.py
    - revision/results/tstr.json
    - revision/results/augmentation.json
  modified: []

key-decisions:
  - "Reused Phase-10 TSTRLiteLSTM (1-layer LSTM-32) verbatim; added MAE/RMSE to return dict (sklearn ABSENT)"
  - "Single cohesive driver with --tstr-only/--augmentation-only flags rather than two drivers"
  - "Augmentation injection grid {+25%,+50%,+100%} + real_only + synthetic_only; subsample RNG seed recorded per row"
  - "base path anchored at resolved REPO root (not relative) so driver runs from worktree or main repo unchanged"

patterns-established:
  - "Long-form JSON header mirrors baseline_comparison.json (schema/model_kinds/pipelines/seeds/data_hash/data_hash_verification)"
  - "T-11-02 leakage guard: eval=OD[:320], train=OD[320:], index-set disjointness asserted at runtime"

requirements-completed: [EVAL-01, EVAL-04]

# Metrics
duration: ~75min
completed: 2026-05-17
---

# Phase 11 Plan 01: Utility Evaluation Summary

**TSTR soft-sensor (R2/MAE/RMSE) and Orlandi-style augmentation lift curve over 60 frozen GAN sample artifacts, reproducing the Phase-10 real-only anchor exactly (R2=-13.3542 ± 0.5833) with zero numeric drift.**

## Performance

- **Duration:** ~75 min (dominated by 2 end-to-end driver runs: TSTR ~39 LSTM trainings, augmentation ~180)
- **Started:** 2026-05-17 (worktree base 8a72391)
- **Completed:** 2026-05-17
- **Tasks:** 3
- **Files modified:** 3 created (1 driver + 2 JSON artifacts)

## Accomplishments

- `revision/run_utility.py`: a single CLI driver patterned on `run_baselines.py`, copying `_run_base`/`reconstruct_od` and `TSTRLiteLSTM`/`r2_score_inline`/`train_eval_tstr` **verbatim** from `_build_baseline_notebook.py` (the Pipeline-B `np.random.default_rng(seed*7919+1)` od_start draw is preserved; MAE/RMSE added inline since sklearn is absent).
- **EVAL-01 (`tstr.json`)**: 144 long-form rows + 12 (model|pipeline) aggregate blocks. `real_only_baseline` reproduces the Phase-10 anchor **exactly**: R2 = -13.3542 ± 0.5833 (anchor: -13.354 ± 0.583), n_train_real=65, n_eval_real=320. Pipeline-B headline TSTR R2 is strong (ar|B 0.998, wgan_mlp|B 0.997, quantum|B in the high-0.99 band); Pipeline A is the expected weak raw-OD control (negative R2 for several models).
- **EVAL-04 (`augmentation.json`)**: 180 long-form rows + 12 lift blocks across conditions {real_only, +25%, +50%, +100%, synthetic_only}, with `r2_delta`/`mae_delta`/`rmse_delta` vs the real-only anchor. All 12 real_only R2 are negative (leakage sentinel clean). Largest lift is `synthetic_only` on Pipeline B (r2_delta ≈ +14.35), consistent with the ~60× synthetic-budget advantage documented in metadata as a lower-bound caveat.
- **Invariants held:** recomputed `data_hash == 91e447d4624e25b3` and equals all 50 baseline `config.yaml` hashes; `git diff --stat revision/core/` empty after every task (D-11-10).

## Task Commits

Each task was committed atomically:

1. **Task 1: Driver scaffold — artifact I/O, data-hash invariant, verbatim reconstruct_od** - `14699e7` (feat)
2. **Task 2: EVAL-01 TSTR — verbatim train_eval_tstr + MAE/RMSE, write tstr.json** - `98d1e0e` (feat)
3. **Task 3: EVAL-04 Orlandi augmentation — mixing-ratio lift curve, write augmentation.json** - `79c4822` (feat)

## Files Created/Modified

- `revision/run_utility.py` - Phase 11 driver: EVAL-01 TSTR + EVAL-04 augmentation; verbatim recon/TSTR reuse; repo-root resolver; data-hash invariant + leakage guard.
- `revision/results/tstr.json` - 144 long-form R2/MAE/RMSE rows + per-(model|pipeline) aggregates + `real_only_baseline` anchor block.
- `revision/results/augmentation.json` - 180 long-form delta rows + per-generator lift blocks + ~60× lower-bound caveat metadata.

## Decisions Made

- **One driver, two flags** (`--tstr-only` / `--augmentation-only`): the plan named one file `revision/run_utility.py` for both EVAL-01 and EVAL-04; a single cohesive driver keeps the verbatim recon/TSTR code defined once and shared by both, matching the plan's stated artifact list.
- **`base` path anchored at resolved `REPO` root** (vs the notebook's relative `Path("revision/results/...")`): required so the driver produces identical numbers whether invoked from the worktree or the main repo. This is a faithfulness-preserving adaptation, not a logic change — the seeded RNG draw, branch math, and inverse-transform calls are byte-identical to `_build_baseline_notebook.py:167-208`.
- **Reused Phase-10 `TSTRLiteLSTM` verbatim** (Open Question 2 RESOLVED in RESEARCH): lowest-risk, maximally comparable to the already-published Phase-10 scaffolding; only MAE/RMSE added to the return dict.

## Deviations from Plan

None - plan executed exactly as written. The repo-root path anchoring is an explicitly-required adaptation called out by the plan/RESEARCH (Pitfall 6: "Resolve `--csv-path` to an ABSOLUTE path anchored at the repo root"; the same anchoring was applied to `_run_base` for the identical worktree-cwd reason), not an unplanned deviation.

## Issues Encountered

- **Frozen artifacts + venv live only in the main repo, not the worktree.** The 60 `samples.npy`/`inverse_kwargs.npz` artifacts and `qgan_env` are large/untracked and exist at `/Users/shawngibford/dev/phd/qGAN`, while this executor runs in a git worktree. The plan's own verification commands all `cd /Users/shawngibford/dev/phd/qGAN`. Resolution: the driver was authored and committed in the worktree, mirrored to the main repo to run the end-to-end driver + verifications against real artifacts, and the resulting JSON artifacts were copied back into the worktree for atomic commit. The repo-root resolver makes the committed driver run identically in both locations, so the committed code and the verified code are the same logic.
- **Buffered stdout on long runs.** Python stdout did not flush until process exit; progress was tracked via output-file polling and the JSON artifact's appearance. No functional impact.

## Verification Results

- `ast.parse` of `revision/run_utility.py`: ok
- `_compute_data_hash(data.csv) == "91e447d4624e25b3"`: PASS (and == all 50 baseline config.yaml hashes)
- `reconstruct_od('wgan_mlp','B',42)["od_samples"].shape == (3840,10)`, dtype float64: PASS
- `reconstruct_od('quantum','B',42)`: PASS (no KeyError; no data_hash assert on quantum)
- `_run_base('quantum','B',42)` resolves to `transform_ablation/runs/B/42`: PASS
- `tstr.json`: data_hash OK; real_only_baseline n_train_real=65, n_eval_real=320, R2=-13.354 (negative — no leakage); rows cover {r2,mae,rmse} at scale=OD; exact 6-key long-form schema; 6×2 aggregate coverage
- `augmentation.json`: 180 rows; injection_ratio covers {real_only,+25%,+50%,+100%,synthetic_only}; metric_name covers {r2_delta,mae_delta,rmse_delta}; scale=OD; all 12 real_only R2 negative; 6×2 covered; ~60× lower-bound caveat present in metadata
- `git diff --stat revision/core/`: empty after every task (D-11-10 invariant held)

## Next Phase Readiness

- EVAL-01 and EVAL-04 requirements satisfied; `tstr.json` + `augmentation.json` are ready for Phase 14 manuscript consumption (ROADMAP SC-1 / SC-3).
- The verbatim-reuse approach is proven (Phase-10 anchor reproduced to 4 decimal places) — Phase 11 plans 02 (TimeGAN scores) and 03 (dual-scale fidelity) can safely follow the same `reconstruct_od`/long-form-JSON contract.
- No blockers.

## Self-Check: PASSED

- FOUND: revision/run_utility.py
- FOUND: revision/results/tstr.json
- FOUND: revision/results/augmentation.json
- FOUND: .planning/phases/11-utility-evaluation/11-01-SUMMARY.md
- FOUND commit 14699e7 (Task 1)
- FOUND commit 98d1e0e (Task 2)
- FOUND commit 79c4822 (Task 3)
- FOUND commit c846d59 (plan metadata)

---
*Phase: 11-utility-evaluation*
*Completed: 2026-05-17*
