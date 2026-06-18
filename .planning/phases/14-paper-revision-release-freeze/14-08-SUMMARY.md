---
phase: 14-paper-revision-release-freeze
plan: 08
subsystem: matched-2000ep-dualscale
tags: [render-only, eval-only, dual-scale, OD-vs-log_return, frozen-headline-distinct, number-provenance-gated, json-traceable]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze (plan 02)
    provides: "matched2000/runs/<model>/<seed>/ accepted 2000ep bundles (45/45) + headline_canonical.json (frozen epoch-1969 dual-scale row-set)"
  - phase: 14-paper-revision-release-freeze (plan 03)
    provides: "run_model_info.py _fmt() JSON->markdown idiom + the explicit-raise data_hash gate pattern"
  - phase: 14-paper-revision-release-freeze (plan 04)
    provides: "run_figure_suite.py render-only contract (matplotlib.use('Agg') before pyplot, _load_json loud-fail, dual PNG+PDF + same-stem JSON, _find_repo_root) — the canonical render-only pattern Task 2 extends"
provides:
  - "run_matched2000_dualscale.py — eval-only matched-2000ep dual-scale aggregator: re-emits EMD + 4 moments + ACF lags 0-9 + DTW at BOTH OD and log_return scales for all 9 matched-2000ep models from the 45 frozen samples.npy bundles, plus the frozen-checkpoint headline as a DISTINCT row-set (D-14-10), with explicit-raise data_hash gate and loud-fail on missing samples"
  - "results/matched2000_dualscale.json — the single source of truth for matched-2000ep dual-scale numbers: 2576 long-form rows[] + 560 (model,scale,metric) aggregates[], data_hash=91e447d4624e25b3, the 10th row-set is model_kind=frozen_checkpoint_headline (n_seeds=1, source=frozen_checkpoint_epoch_1969)"
  - "figures/matched2000_dualscale_sidebyside.{png,pdf,json} — render-only side-by-side dual-scale figure (3 metric panels x 2 scales = 6 panels; frozen headline as distinct dashed reference line + black diamond marker; companion JSON records every plotted tuple + source_artifact path)"
  - "figures/matched2000_dualscale_comparison.md — copy-paste comparison-table doc rendered via _fmt() entirely from matched2000_dualscale.json; 167 distinct literals all resolve through the existing verify_number_provenance.py gate (no gate edit)"
affects: [paper-PAPER-09, manuscript-dual-scale-section, 14-09-zenodo-freeze]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Eval-only re-aggregation over frozen samples.npy bundles: NO retraining, NO resampling — verbatim Pipeline-B reconstruct_od (the seed*7919+1 od_start draw is load-bearing) + revision.core.eval ONLY (D-11-10). git diff --stat core/ stays empty across the whole plan"
    - "Frozen headline carried as a DISTINCT model_kind=frozen_checkpoint_headline row-set with explicit source=frozen_checkpoint_epoch_1969 on every row (n_seeds=1); never merged into iqp_sel_55_repro (D-14-10 / T-14-16) — enforced numerically (zero row-object overlap) AND visually (dashed reference line + diamond marker positioned outside the bar group)"
    - "Explicit-raise data_hash gate (python -O safe): raise AssertionError if recomputed != 91e447d4624e25b3 OR any consumed config.yaml data_hash mismatches; loud-fail lists every offending (model, seed)"
    - "Render-only side-by-side figure + comparison.md sourced SOLELY from the gated JSON: every plotted value pulled from aggregates[]; every markdown cell is _fmt() of an aggregate; zero hand-typed numbers; loud-fail FileNotFoundError on missing source JSON"
    - "Number-provenance gate (verify_number_provenance.py) auto-covers the new dual-scale JSON via its results/*.json rglob — comparison.md passes the existing gate UNMODIFIED (167/167 literals resolve)"

key-files:
  created:
    - run_matched2000_dualscale.py  # 625 lines (Task 1, recovered)
    - results/matched2000_dualscale.json  # 28270 lines / ~697 KB (Task 1, recovered)
    - figures/matched2000_dualscale_sidebyside.png  # Task 2
    - figures/matched2000_dualscale_sidebyside.pdf  # Task 2
    - figures/matched2000_dualscale_sidebyside.json  # Task 2 (60 plotted tuples + source_artifact)
    - figures/matched2000_dualscale_comparison.md  # Task 2 (passes verify_number_provenance.py)
  modified:
    - run_figure_suite.py  # Task 2: +305 lines (new dual-scale routines + main() wiring)

key-decisions:
  - "Task 1 commit landed via orchestrator-led recovery (commit author Shawn Gibford), not via the standard per-task executor commit, after the original executor (agent a6207b9353499ef52) hit an API socket error post-artifact-emission but pre-commit. Orchestrator recovered the byte-identical artifacts from the locked worktree, re-ran Task 1's full <verify> block end-to-end on main (both_scales=True, agg_ok=True, head_agg_ok=True, data_hash_ok=True; revision.core.eval-only / explicit-raise / no-train-or-sample paths all confirmed) before committing as b3235d9 — documented as deviation #1 below"
  - "Task 2 extends run_figure_suite.py rather than adding a separate run_matched2000_dualscale_figure.py — keeps the entire 2000ep figure surface behind a single render entrypoint per 14-04's 'self-contained suite' precedent; the plan explicitly authorizes either structure (Claude's discretion)"
  - "Figure composition (Claude's D-14 figure-discretion): 3 metric panels (EMD + moment_mean + moment_std) x 2 scale columns (OD, log_return) = 6 subplots. Bars are mean+/-std for the 9 matched-2000ep entrants; frozen headline as a dashed black reference line + diamond marker positioned at x=-0.6 (OUTSIDE the bar group) so it cannot be visually fused with the iqp_sel_55_repro bar (D-14-10)"
  - "Comparison.md metric set: EMD + 4 moments (mean, std, skewness, kurtosis) x 10 entrants x 2 scales — the headline metric set most directly responsive to PAPER-09's matched-budget dual-scale ask. ACF and DTW metrics remain in matched2000_dualscale.json (rows[] + aggregates[]) but are not surfaced in the copy-paste table to keep it manuscript-pasteable"
  - "Markdown cell precision: _fmt() formats floats at 4 decimal places (f'{v:.4f}'), confirmed to resolve via the gate's float-precision matcher against the full-precision JSON values — verified empirically by the verifier passing 167 distinct literals"

patterns-established:
  - "Dual-scale eval-only aggregator pattern: copy run_dualscale_fidelity end-to-end and change ONLY (a) the run-dir resolver, (b) the model list, (c) the source-tagging from config.yaml, (d) add a per-(model,scale,metric) seed-aggregate. The Pipeline-B reconstruction and revision.core.eval metric calls remain VERBATIM (D-11-10) so numbers reconcile across artifacts"
  - "Figure suite extension pattern: a new render routine in run_figure_suite.py reads ONLY a gated JSON source and emits PNG+PDF + a same-stem companion JSON that records every plotted tuple + the source artifact path — figure<->data provenance is independently re-derivable (T-14-17)"

requirements-completed: [PAPER-09]

# Metrics
duration: ~25min  # Task 2 only; Task 1 was completed and committed by the orchestrator recovery flow prior to my spawn
completed: 2026-05-20
---

# Phase 14 Plan 08: Matched-2000ep Dual-Scale Summary

**Added `run_matched2000_dualscale.py` (eval-only aggregator) + `results/matched2000_dualscale.json` (the single source of truth for matched-2000ep dual-scale numbers, 2576 rows + 560 aggregates, data_hash=91e447d4624e25b3) — every number computed via `revision.core.eval` ONLY from the 45 already-saved `samples.npy` bundles (NO retraining, NO resampling) — and then a render-only side-by-side dual-scale figure (PNG+PDF+companion JSON) + a copy-paste comparison-table doc, both sourced solely from that JSON, with the frozen-checkpoint headline carried as a DISTINCT row-set (n_seeds=1, source=frozen_checkpoint_epoch_1969) numerically AND visually separate from the iqp_sel_55_repro 2000ep reproduction (D-14-10). The resubmission's PAPER-09 matched-budget dual-scale comparison is now traceable to a single JSON source of truth, gated by both the explicit-raise data_hash check and the existing `verify_number_provenance.py` unmodified (167/167 literals resolve).**

## Performance

- **Duration:** ~25 min (Task 2 only; Task 1 was recovered and committed by the orchestrator before this executor was spawned — see Deviations)
- **Started:** 2026-05-20 (worktree agent-a92036b39100beb3f)
- **Completed:** 2026-05-20
- **Tasks:** 2 (Task 1 recovered + Task 2 newly executed)
- **Files:** 6 created + 1 modified (605 lines net: Task 1 = 625 + 28270 in pre-existing recovered commit; Task 2 = +305 lines on run_figure_suite.py + 4 new figure files)

## Accomplishments

### Task 1 — Eval-only matched-2000ep dual-scale aggregator + JSON (recovered)

This task was completed by a prior executor (agent a6207b9353499ef52) but never committed by that executor due to an API socket error post-artifact-emission. The orchestrator recovered the byte-identical artifacts from the locked worktree, re-ran the plan's `<verify>` block end-to-end on main, and committed as `b3235d9` directly on main. The artifact integrity claims below reflect that re-verified run.

- `run_matched2000_dualscale.py` (625 lines) — eval-only aggregator copied end-to-end from `run_dualscale_fidelity.py` with ONLY the four matched-2000ep-required changes:
  1. `_run_base`/`_resolve_run_dir` resolves `results/matched2000/runs/<model>/<seed>` and **raises `FileNotFoundError`** with a "no retraining/resampling — D-14-08" message on any absent `samples.npy` / `inverse_kwargs.npz` (loud-fail, never a silent partial aggregate — T-14-14).
  2. `MODEL_KINDS = ['iqp_sel_55_repro','V1','V2','V3','wgan_mlp','wgan_cnn','wgan_lstm','vae','ar']`; matched2000 is Pipeline-B ONLY (verified — no Pipeline-A run dir exists for any matched-2000ep model).
  3. Metric math is `revision.core.eval` IMPORT-ONLY: `compute_emd`/`compute_moments`/`compute_acf`/`compute_dtw` are reused UNCHANGED; the VERBATIM `_od_scale_rows`/`_log_return_rows` bodies and `build_real_references` are copied from `run_dualscale_fidelity.py` so every metric reconciles with the audited dualscale recipe (D-11-10 / T-14-15).
  4. **Explicit-raise data_hash gate (python -O safe):** `raise AssertionError` if recomputed `data_hash != 91e447d4624e25b3` OR if ANY consumed `matched2000/runs/<model>/<seed>/config.yaml` `data_hash` differs (parsed via `yaml.safe_load` — pure-aggregator-safe per 14-03). Loud-fail lists every mismatching (model, seed) — T-14-13.
  5. Frozen-checkpoint headline as its own distinct row-set (D-14-10 / T-14-16): `headline_canonical.json`'s existing dual-scale rows are re-emitted **verbatim** (no checkpoint reload — 14-02 owns that) under `model_kind="frozen_checkpoint_headline"` with an explicit `source="frozen_checkpoint_epoch_1969"` field on every headline row. For the 9 matched2000 models, each row's `source` is READ VERBATIM from that run's `config.yaml['source']` (never hardcoded) — iqp_sel_55_repro carries `matched2000_reproduction`, V1/V2/V3 carry `matched2000_ansatz`, and wgan_mlp/wgan_cnn/wgan_lstm/vae/ar carry `matched2000_baseline`.
  6. Per-(model_kind, scale, metric_name) aggregate: `aggregates[]` with `{mean, std, n_seeds}` over the 5 matched-2000ep seeds (n_seeds=5 for the 9 sweep models, n_seeds=1 for the frozen headline). Aggregations use `statistics`/`numpy` over `revision.core.eval` outputs — not new metric math.

- `results/matched2000_dualscale.json` (697 KB / 28270 lines) — the single source of truth for these numbers:
  - **2576 long-form rows** (model_kind, pipeline, seed, metric_name, scale, value, source)
  - **560 aggregates** (model_kind, scale, metric_name, mean, std, n_seeds, source) — covers 10 entrants (9 sweep + 1 headline) × 2 scales × 28 metric_names
  - `schema: "matched-2000ep dual-scale rows[] + per-(model,scale,metric) seed-aggregate; frozen headline DISTINCT, D-14-10"`
  - `model_kinds: ["iqp_sel_55_repro","V1","V2","V3","wgan_mlp","wgan_cnn","wgan_lstm","vae","ar"]`
  - `pipelines: ["B"]`; `seeds: [42,43,44,45,46]`
  - `data_hash: "91e447d4624e25b3"` (canonical, verified across all 45 consumed config.yaml's + the recomputed hash from data.csv)
  - `metric_helpers: { functions: [revision.core.eval.compute_emd, compute_moments, compute_acf, compute_dtw], note: "reused unchanged, D-11-10" }`
  - All 9 matched2000 models have **both** `scale="OD"` AND `scale="log_return"` EMD rows present; the frozen headline is a separate, never-merged row-set (zero row-object overlap with `iqp_sel_55_repro` — verified by the plan's <automated> assertion).

### Task 2 — Render-only matched-2000ep dual-scale side-by-side figure + comparison-table doc

- **Extended `run_figure_suite.py`** (+305 lines, 1139 lines total) with two new render routines (chose extension over a separate `run_matched2000_dualscale_figure.py` because the plan explicitly authorizes either structure and the 14-04 "self-contained suite" precedent argues for keeping the entire 2000ep figure surface behind one entrypoint):
  - `render_matched2000_dualscale_sidebyside(repo, figures_dir)` — emits `matched2000_dualscale_sidebyside.{png,pdf,json}`.
  - `render_matched2000_dualscale_comparison_table(repo, figures_dir)` — emits `matched2000_dualscale_comparison.md`.
- **Figure design** (3 rows × 2 cols = 6 panels):
  - **Row 1**: EMD (OD scale | log-return scale) — the primary side-by-side panel PAPER-09 requests.
  - **Row 2**: moment_mean (OD | log-return) — first-order distributional fit.
  - **Row 3**: moment_std (OD | log-return) — variance-structure fit.
  - Each cell: 9 colored bars (the matched-2000ep entrants in `MODEL_ORDER`) with `mean ± std` error bars over 5 seeds. The frozen-checkpoint headline is overlaid as a **distinct dashed black reference line + a black diamond marker** positioned at `x = -0.6` (visually OUTSIDE the bar group) so the eye cannot fuse it with the `iqp_sel_55_repro` bar — D-14-10 / T-14-16. A figure-suptitle explicitly names the headline as "epoch 1969 — a DISTINCT series".
- **Companion JSON** records every plotted (model_kind, scale, metric_name, mean, std, n_seeds, source) tuple — 60 tuples (3 metrics × 2 scales × 10 entrants) — plus:
  - `source_artifact = "results/matched2000_dualscale.json"` (figure↔data provenance, T-14-17)
  - `source_data_hash = "91e447d4624e25b3"`
  - `headline_kind = "frozen_checkpoint_headline"`, `headline_source = "frozen_checkpoint_epoch_1969"`
  - `conflation_guard = "D-14-10 / T-14-16: frozen-checkpoint headline plotted as a DISTINCT dashed reference line + diamond marker; never merged into the iqp_sel_55_repro 2000ep reproduction bar."`
- **Comparison-table doc** (`matched2000_dualscale_comparison.md`): two markdown tables (OD-scale aggregates, log-return-scale aggregates) covering EMD + 4 moments (mean, std, skewness, kurtosis) × 10 entrants (9 sweep models + the FROZEN headline as a clearly-labelled separate row). Every cell is `_fmt()` of an `aggregates[]` row from the source JSON — **zero hand-typed numbers** (`run_model_info.py:394-406` `_fmt()` idiom). 167 distinct numeric literals.
- **Loud-fail contract**: both routines call `_load_json` on `results/matched2000_dualscale.json`; a missing file raises `FileNotFoundError` with a "render-only (no training/eval) … run run_matched2000_dualscale.py first" message — verified by an explicit probe with the path swapped to a nonexistent file (`FileNotFoundError` raised as expected).
- **No-recompute discipline**: figure renders by reading the JSON only; no metric is recomputed during render. `grep '.fit(/def train_/model.sample('` on the entire `run_figure_suite.py` returns no matches (render-only end-to-end).
- **Number-provenance gate**: `verify_number_provenance.py` **PASSES on the comparison.md** with **167 distinct literals all resolving** to `matched2000_dualscale.json` — and the gate file was **NOT modified** (the gate's existing `results/*.json` rglob auto-covers the new artifact). `git diff --stat verify_number_provenance.py` is empty.

### Verification (plan verify command — verbatim PASS)

- `./qgan_env/bin/python run_figure_suite.py` → RUN OK (77 PNG total, up from 76 — the new sidebyside figure added; existing 76 figure stems unchanged in content though the PDFs re-render with new timestamps, see Issues Encountered)
- `[ -e matched2000_dualscale_sidebyside.png AND .pdf AND .json AND matched2000_dualscale_comparison.md ]` → **PASS** (all 4 artifacts present, exit 0)
- `./qgan_env/bin/python verify_number_provenance.py --target figures/matched2000_dualscale_comparison.md` → **PASS** (167/167 literals resolve)
- `grep 'matplotlib.use("Agg")'` in `run_figure_suite.py` → PASS (headless before pyplot)
- `grep 'FileNotFoundError|render-only'` in `run_figure_suite.py` → PASS (loud-fail wired)
- Loud-fail probe (point `MATCHED2000_DUALSCALE_REL` at a nonexistent file) → raises `FileNotFoundError` with the render-only message
- `git diff --stat core/` → empty (eval module untouched across the whole plan)
- `git diff --stat verify_number_provenance.py` → empty (gate unmodified)
- No training/sampling/checkpoint-reload path in either Task-1's aggregator or Task-2's renderer: `grep -E '\.fit\(|def train_|model\.sample\('` returns no matches in `run_matched2000_dualscale.py` or in the new `run_figure_suite.py` routines.

## Task Commits

1. **Task 1: Eval-only matched-2000ep dual-scale aggregator + JSON** — `b3235d9` (feat)
   - Committed by the orchestrator (author: Shawn Gibford <shawgi@dtu.dk>) directly on `main` via a recovery flow after the original executor hit an API socket error post-artifact-emission. See deviation #1 below for the full recovery narrative.
2. **Task 2: Render-only matched-2000ep dual-scale side-by-side figure + comparison doc** — `e90ad06` (feat)
   - Committed by this executor (worktree agent-a92036b39100beb3f) on its per-agent branch.

## Files Created/Modified

- `run_matched2000_dualscale.py` — 625-line eval-only aggregator (Task 1, recovered)
- `results/matched2000_dualscale.json` — 697 KB / 28270-line dual-scale JSON; 2576 rows + 560 aggregates; data_hash=`91e447d4624e25b3` (Task 1, recovered)
- `run_figure_suite.py` — +305 lines: the two new dual-scale render routines + `main()` wiring (Task 2)
- `figures/matched2000_dualscale_sidebyside.png` — 230 KB, 3×2-panel side-by-side figure (Task 2)
- `figures/matched2000_dualscale_sidebyside.pdf` — 30 KB vector variant (Task 2)
- `figures/matched2000_dualscale_sidebyside.json` — 571-line companion (60 plotted tuples + source_artifact + conflation_guard) (Task 2)
- `figures/matched2000_dualscale_comparison.md` — 41-line copy-paste comparison table, 167 literals (Task 2)

## Decisions Made

- **D-14-10 enforcement (frozen headline ≠ iqp_sel_55_repro):** carried numerically (distinct `model_kind="frozen_checkpoint_headline"` row-set with `source="frozen_checkpoint_epoch_1969"` AND its own `n_seeds=1` aggregate) AND visually (dashed reference line + diamond marker at `x=-0.6`, OUTSIDE the bar group; explicit suptitle naming "frozen headline (epoch 1969) is a DISTINCT series"; companion-JSON `conflation_guard` string). The plan's `<verify>` block asserts zero row-object overlap with `iqp_sel_55_repro` — confirmed.
- **No-retraining / no-resampling lock:** every number in the new JSON is computed from a frozen `samples.npy` bundle (read-only, byte-identical to the 14-02 sweep accept). Missing-bundle case is `FileNotFoundError` with an explicit "no retraining/resampling — D-14-08" message rather than a silent partial.
- **Single-source-of-truth contract:** the side-by-side figure and the comparison.md table both source SOLELY from `matched2000_dualscale.json`. There is no second numeric origin for these matched-budget dual-scale numbers; the figure's companion JSON records the source artifact path so the figure is independently re-derivable.
- **Task-2 entrypoint choice (extend vs. new file):** chose to extend `run_figure_suite.py` rather than add a separate `run_matched2000_dualscale_figure.py`. Rationale: the plan explicitly authorizes either; the 14-04 "self-contained suite" precedent argues for one render entrypoint; the new routines reuse the existing `_load_json`, `_save`, `_find_repo_root`, `MODEL_COLORS`/`MODEL_LABELS` infrastructure without duplication.
- **Markdown precision (`_fmt` at 4 decimals):** verified empirically against the gate — `f"{v:.4f}"` resolves through the verifier's float-precision matcher (mode 2: format both literal and candidate at the literal's stated precision, then equality-compare). 167/167 literals pass.

## Deviations from Plan

### Auto-fixed / Rule-3 Recovery Issues

**1. [Recovery — non-standard commit path for Task 1] Original executor API socket error pre-commit; orchestrator recovered**
- **Found during:** Task 1 by a prior executor (agent `a6207b9353499ef52`), before my spawn.
- **Issue:** The prior executor produced Task 1's artifacts (`run_matched2000_dualscale.py` and `results/matched2000_dualscale.json`) byte-identical to the verified contents but hit an API socket error post-emission and never executed the per-task commit step. The worktree was left clean-content / unstaged.
- **Fix (orchestrator-led, not by this executor):** the orchestrator (a) recovered the artifacts byte-identical from the locked worktree, (b) re-ran the plan's Task-1 `<verify>` block **end-to-end on main** (`both_scales=True`, `agg_ok=True`, `head_agg_ok=True`, `data_hash_ok=True`; `revision.core.eval` import-only / explicit-raise gate / no train-or-sample paths all confirmed), (c) committed the artifacts directly on `main` as `b3235d9` (author: Shawn Gibford <shawgi@dtu.dk>), and (d) retired the orphaned worktree. This executor inherited a clean base with Task 1 already done and re-ran Task 2 only.
- **Note for auditors (executor-protocol honesty):** Task 1's commit author and the on-main commit path differ from the standard worktree-agent executor protocol (which would commit on the per-agent branch as `worktree-agent-<id>`). This deviation is **integrity-equivalent** because (i) the artifacts are byte-identical to what the executor would have committed, (ii) the plan's `<verify>` block was re-run end-to-end on main with all gates green, and (iii) the canonical-base reset (`worktree_branch_check`) on my spawn pins HEAD to `b3235d9` and ensures Task 2's commit is built on the re-verified Task-1 state. No artifact was hand-typed or recomputed during recovery.
- **Files affected:** `run_matched2000_dualscale.py`, `results/matched2000_dualscale.json` (created via recovery commit `b3235d9`).
- **Commit:** `b3235d9` (Task 1 — recovered).

**Total deviations:** 1 recovery-driven commit-path deviation for Task 1 (artifact integrity independently re-verified before commit). Task 2 was executed and committed under the standard executor protocol with zero deviations.

## Issues Encountered

- **`qgan_env` absent in worktree:** `qgan_env` lives in the main checkout and is gitignored. Resolved by the established `ln -s /…/qGAN/qgan_env qgan_env` symlink (already in `.gitignore`, never committed) — same idiom as plans 14-01 / 14-02 / 14-04. The plan's verify command (`./qgan_env/bin/python run_figure_suite.py`) then works from the worktree.
- **`run_figure_suite.py` PDF re-render churn (cosmetic, pre-existing):** running `run_figure_suite.py` end-to-end (Task 2's verify-step requirement) re-renders **all 76 prior figures** with new timestamps in PDF metadata, making them byte-different but content-identical. This is the same matplotlib-timestamp issue 14-04 documented. Additionally, `render_time_series_comparison` uses `np.random.default_rng(model.__hash__() & 0xFFFF)` where `__hash__()` is Python-hash-randomized across invocations (PEP 456), so `timeseries_*.png/json` also re-render with different (still-valid) sampled-window indices. **Both issues are pre-existing in `run_figure_suite.py` and unrelated to Task 2's deliverables.** Resolution: reverted the timestamp-driven PDF re-renders and the hash-randomized timeseries re-renders before commit so Task 2's commit contains ONLY its own deliverables — `git status` was clean (5 files staged) at commit time.
- **Worktree base initially on a pre-Task-1 commit:** the spawn-time HEAD assertion detected the worktree branch was diverged from `b3235d9` (the canonical Task-1 base); the `worktree_branch_check` reset HEAD to `b3235d9` per its documented protocol. No work was lost (no Task-2 changes existed yet); this is the expected behavior of the recovery-aware spawn.

## Known Stubs

None — every number in `matched2000_dualscale.json` is computed via `revision.core.eval` from a frozen `samples.npy` bundle (Task 1); every plotted value in the side-by-side figure and every cell in the comparison.md is `_fmt()` of an `aggregates[]` row from that JSON (Task 2). No hardcoded empty values, no placeholder text, no mock data sources. The aggregator hard-fails (`FileNotFoundError`) rather than emit a partial JSON; the renderer hard-fails (`FileNotFoundError`) rather than emit a stub figure.

## Threat Surface Scan

No new network endpoints, auth paths, or external file-access patterns. The plan's three trust boundaries (saved samples → re-emitted metric; gated JSON → figure/table; frozen headline ↔ 2000ep reproduction) are all mitigated as specified:
- **T-14-13** (wrong-hash / mixed-budget sample silently aggregated) — mitigated by the explicit-raise data_hash gate on both the recomputed hash from `data.csv` AND every consumed `config.yaml['data_hash']`. Both Task-1's re-verification run and Task-2's render confirm `source_data_hash = "91e447d4624e25b3"` in the figure companion JSON.
- **T-14-14** (silent partial aggregate on a missing sample bundle) — mitigated by `_resolve_run_dir` raising `FileNotFoundError` with a no-retraining-or-resampling message; verified loud-fail probe.
- **T-14-15** (hand-typed / re-implemented metric number) — mitigated by `revision.core.eval` import-only metric math (verbatim `_od_scale_rows`/`_log_return_rows`); `git diff --stat core/` empty; comparison.md gated by `verify_number_provenance.py` (167/167 literals resolve).
- **T-14-16** (frozen headline merged into iqp_sel_55_repro) — mitigated by the distinct `model_kind="frozen_checkpoint_headline"` + explicit `source="frozen_checkpoint_epoch_1969"` row-set; zero row-object overlap with `iqp_sel_55_repro` (asserted by the plan's <automated>); figure plots the headline as a distinct dashed line + diamond marker outside the bar group; companion JSON records the `conflation_guard` string.
- **T-14-17** (figure ↔ data provenance) — mitigated by the same-stem companion JSON recording every plotted value + the source artifact path; comparison.md passes the number-provenance gate which auto-covers the new JSON via its rglob.

No threat flags.

## Self-Check: PASSED

- `run_matched2000_dualscale.py` — FOUND (625 lines, on the working tree from commit `b3235d9`)
- `results/matched2000_dualscale.json` — FOUND (697 KB / 28270 lines, data_hash=`91e447d4624e25b3`, 2576 rows, 560 aggregates, on commit `b3235d9`)
- `run_figure_suite.py` — FOUND (1139 lines post-edit, includes new dual-scale routines + `main()` wiring)
- `figures/matched2000_dualscale_sidebyside.png` — FOUND (3×2-panel figure, 230 KB)
- `figures/matched2000_dualscale_sidebyside.pdf` — FOUND (30 KB)
- `figures/matched2000_dualscale_sidebyside.json` — FOUND (60 plotted tuples + source_artifact + conflation_guard)
- `figures/matched2000_dualscale_comparison.md` — FOUND (41 lines, 167 literals)
- Plan verify gate (full sequence) — PASS: `run_figure_suite.py` runs; all 4 Task-2 artifacts present; `verify_number_provenance.py --target ...comparison.md` returns "PASS — 167 distinct numeric literal(s) all resolve to results/*.json"; `matplotlib.use("Agg")` grep present; `FileNotFoundError|render-only` grep present.
- Loud-fail probe — `_load_json` raises `FileNotFoundError` with the render-only message when the dual-scale JSON path is swapped to a nonexistent file.
- `git diff --stat core/` — empty (eval module untouched across whole plan).
- `git diff --stat verify_number_provenance.py` — empty (gate unmodified).
- Commit `b3235d9` (Task 1, recovered) — FOUND on `git log`.
- Commit `e90ad06` (Task 2) — FOUND on `git log`.

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-20*
