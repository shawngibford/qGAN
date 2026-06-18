---
phase: 14-paper-revision-release-freeze
plan: 04
subsystem: figure-suite
tags: [matplotlib, render-only, pennylane-pipeline-b, dual-scale, cross-model, headline-vs-reproduction, json-traceable]

# Dependency graph
requires:
  - phase: 14-paper-revision-release-freeze (plan 02)
    provides: "matched2000/runs/<model>/<seed>/ accepted 2000ep bundles (45/45) + headline_canonical.json (frozen epoch-1969 headline)"
  - phase: 14-paper-revision-release-freeze (plan 01)
    provides: "canonical 55-param IQP:SEL config lock + native Pipeline B pin"
  - phase: 13-architecture-introspection (plan 04)
    provides: "run_introspect_figures.py render-only shape (the pattern D-14-17 names) + the 3 introspection companion JSONs"
provides:
  - "run_figure_suite.py — render-only per-model + cross-model + introspection figure generator (loud-fail on missing companion, dual PNG+PDF + same-stem reproducibility JSON)"
  - "results/figures/ — 76 PNG figures, each with matching PDF + JSON (>= the verified 16-figure canonical bar)"
affects: [14-05, 14-06, 14-07, paper-figures, manuscript]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Render-only figure suite: every figure is a <stem>.png + <stem>.pdf + <stem>.json triple; a missing 2000ep render-input is a hard FileNotFoundError, never a silent partial figure (T-14-10/11)"
    - "Headline-vs-reproduction conflation guard: the FROZEN epoch-1969 headline (headline_canonical.json) and the 2000ep iqp_sel_55_repro reproduction are drawn as separate distinctly-labelled series, never merged (D-14-10/T-14-12)"
    - "Verbatim Pipeline-B OD reconstruction copied from run_dualscale_fidelity:221-236 (the seed*7919+1 od_start draw is load-bearing) so figure-space OD reconciles with fidelity_dualscale.json"

key-files:
  created:
    - run_figure_suite.py
  modified:
    - results/figures/ (76 PNG + 76 PDF + 76 JSON triples; 3 prior introspection PDFs re-rendered byte-identically in content)

key-decisions:
  - "Completeness bar is the VERIFIED 16 Figure_*.png canonical set (gaps at 14/16/17/18), NOT 20 — the context/D-14-17 '20' is a known discrepancy (RESEARCH Runtime State / Open Q3 / Assumption A2). Delivered 76 PNG, far exceeding 16."
  - "8 canonical per-model figure types ported from the notebook savefig routines (distribution / dual-scale ACF / QQ / time-series / loss / EMD-over-training / OD-reconstruction / stylized-facts) for all 9 matched2000 models, family-aware (adversarial vs VAE vs AR)"
  - "PRIMARY_SEED=42 for the single-seed distribution/QQ/time-series panels; cross-model EMD uses the 5-seed spread (mean ± std) so the cross-model bar carries the sweep's full seed variance"
  - "Existing 3 introspection figures re-rendered in-suite (delegating to run_introspect_figures routines) so run_figure_suite is self-contained — 'extend, do not overwrite': the introspection companion JSON/PNG content is unchanged"

patterns-established:
  - "Pattern: a single render-only suite entrypoint that consumes ONLY frozen artifacts (matched2000 bundles + headline JSON) and writes a JSON companion per figure — every manuscript figure is traceable to a revision/results value"

requirements-completed: [PAPER-09]

# Metrics
duration: ~25min
completed: 2026-05-19
---

# Phase 14 Plan 04: Render-Only 2000ep Figure Suite Summary

**Added `run_figure_suite.py` — a render-only figure module that builds, from the accepted 2000ep artifacts, a complete per-model + cross-model + introspection figure suite (76 PNG, each with a matching PDF + same-stem reproducibility JSON), loud-failing on any missing companion and labelling the frozen headline vs the 2000ep reproduction distinctly (D-14-10). The manuscript's figure set is now coherent, JSON-traceable, and far exceeds the verified 16-figure canonical bar.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-19 (worktree agent-a37ed7ad7a5e3e41a)
- **Completed:** 2026-05-19
- **Tasks:** 1
- **Files modified:** 1 created + the `results/figures/` suite (228 figure files: 76 PNG / 76 PDF / 76 JSON)

## Accomplishments

### Task 1 — Render-only per-model + cross-model + analysis figure suite
- `run_figure_suite.py` copies the `run_introspect_figures.py` shape end-to-end: headless `matplotlib.use("Agg")` BEFORE pyplot, the `_require`/`_load_json` loud-fail (`FileNotFoundError` with a render-only message — never a silent partial figure), `_save()` writing `<stem>.png` + `<stem>.pdf` at `dpi=150, bbox_inches="tight"` + `plt.close` PLUS a same-stem `<stem>.json` reproducibility companion, the `_find_repo_root()` resolver, the `argparse --figures-dir` default of `results/figures`, and print-every-written-path.
- Added the verbatim `_bootstrap_repo_on_path()` from `run_dualscale_fidelity.py:69-83` so the plan's bare-script verify command (`./qgan_env/bin/python run_figure_suite.py`) works as well as `-m revision.run_figure_suite`.
- **8 canonical per-model figure types** ported from the notebook's ~11 savefig routines, rendered for **all 9 matched2000 models** (`iqp_sel_55_repro`, V1, V2, V3, wgan_mlp/cnn/lstm, vae, ar): `distribution_comparison`, `acf_comparison` (dual-scale OD + log_return, NLAGS=9 matched to the peer driver), `qq_plot`, `time_series_comparison`, `loss_curves` (family-aware: adversarial critic/gen vs VAE ELBO/recon/KLD vs AR closed-form fit), `emd_over_training` (adversarial only), `od_reconstruction`, `stylized_facts_trajectory`.
- **Cross-model figures:** `cross_model_distribution` (all models overlaid on real OD), `cross_model_emd` (5-seed mean ± std bar with the FROZEN headline EMD as a distinct annotated reference line), and an explicit `headline_vs_reproduction` figure. The 55-param IQP:SEL is the quantum entrant in every cross-model figure (D-14-04).
- **Headline/reproduction conflation guard (D-14-10 / T-14-12):** the frozen-checkpoint headline (`headline_canonical.json`, source=`frozen_checkpoint_epoch_1969`) is rendered in a deliberately distinct black/dashed style and labelled `IQP:SEL 55p FROZEN headline (ckpt epoch 1969)`, never merged into the `iqp_sel_55_repro` 2000ep reproduction series; each companion JSON records the explicit conflation guard string.
- **Render-only / no-recompute:** OD-scale reconstruction is the VERBATIM Pipeline-B logic of `run_dualscale_fidelity.py:221-236` (the `np.random.default_rng(seed*7919+1)` od_start draw is load-bearing). Metrics come from the frozen `metrics.json` / `headline_canonical.json`; figure-derived stats use `revision.core.eval` helpers only (`compute_emd/acf/moments/dtw`). No model `.fit(`, training loop, or sampling call exists in the module.
- The 3 existing introspection figures (`training_progression`, `param_trajectory`, `entanglement_trajectory`) are re-rendered in-suite by delegating to the `run_introspect_figures` routines when their companion JSON is present — "extend, do not overwrite" (introspection JSON/PNG content unchanged).

### Verification (plan verify command — verbatim PASS)
- `./qgan_env/bin/python run_figure_suite.py` → RUN OK
- `>= 16` PNG with every PNG having a matching PDF AND JSON → **76 PNG, triple-complete True**
- `grep matplotlib.use("Agg")` before pyplot → PASS
- `grep FileNotFoundError|render-only` (loud-fail) → PASS; an explicit missing-artifact probe **raises `FileNotFoundError`** with the render-only message
- No training/sampling path (`grep .fit(/.train(/model.sample(`) → none found (render-only)
- Per-model figures for all 9 models across 8 canonical types + cross-model distribution/EMD + explicit headline_vs_reproduction present

## Task Commits

1. **Task 1: Render-only per-model + cross-model + analysis figure suite** — `ab0daaf` (feat)

## Files Created/Modified
- `run_figure_suite.py` — render-only PNG+PDF+JSON figure suite generator (834 lines)
- `results/figures/` — 76 PNG / 76 PDF / 76 JSON figure triples (per-model × 9 models × {6–8 types}, 3 cross-model, 3 introspection); the 3 prior introspection PDFs re-rendered byte-identically in content from the unchanged companion JSON

## Decisions Made
- **16 not 20 (RESEARCH Runtime State / Open Q3 / Assumption A2):** the acceptance bar is the verified 16 `Figure_*.png` canonical set (gaps 14/16/17/18); the context/D-14-17 "20" is the documented discrepancy. The suite delivers 76 PNG, far exceeding the bar with full per-model coverage.
- **Family-aware training curves:** adversarial models plot critic/generator loss + EMD-over-training; VAE plots ELBO/recon/KLD; AR(p) (3 params, closed-form fit) plots a fitted-parameter bar — so every model gets a meaningful training/fit figure rather than a forced/empty adversarial-loss panel.
- **Self-contained suite:** the introspection figures are re-rendered in-suite by delegating to the proven `run_introspect_figures` routines, so the manuscript figure set is producible from one entrypoint without breaking the "extend, do not overwrite" contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bare-script invocation `ModuleNotFoundError: No module named 'revision'`**
- **Found during:** Task 1 (first run via the plan's verify command `./qgan_env/bin/python run_figure_suite.py`).
- **Issue:** Running the module as a bare script (not `-m`) puts only the script's own dir on `sys.path`, so `from revision.core...` failed. The plan's `<verify><automated>` block uses the bare-script form, so this blocked the plan's own acceptance gate.
- **Fix:** Added the **verbatim** `_bootstrap_repo_on_path()` from the canonical peer `run_dualscale_fidelity.py:69-83` (walk up to the dir holding `core/preprocessing.py` and prepend to `sys.path`) before the `revision.*` imports. Both `-m` and bare-script invocation now work; no behavior change beyond import resolution.
- **Files modified:** `run_figure_suite.py`
- **Committed in:** `ab0daaf` (Task 1 commit)

**Total deviations:** 1 auto-fixed (1 Rule-3 blocking import-path fix). No scope creep — the fix is a verbatim copy of the established peer-driver bootstrap and only restores the plan's own verify command.

## Issues Encountered
- **`qgan_env` absent in worktree:** `qgan_env` is gitignored and lives in the main checkout (Plan 01/02 precedent). Resolved by the established `ln -s /…/qGAN/qgan_env qgan_env` symlink (already in `.gitignore`, never committed); the script's repo-root resolver writes figures into the worktree's `results/figures/`.
- **3 introspection PDFs show as modified:** matplotlib embeds run-timestamp metadata in PDF, so a deterministic re-render of the unchanged introspection JSON produces a byte-different (content-identical) PDF. This is the intended "self-contained suite" behavior, not a content change — the introspection PNG/JSON are unchanged. No deletions; clean working tree post-commit.

## Known Stubs
None — every figure is rendered from a real frozen 2000ep artifact (`matched2000/runs/<model>/<seed>/` bundle) or the frozen `headline_canonical.json`; figure-derived statistics use `revision.core.eval` helpers only. No hardcoded empty/placeholder values, no mock data sources. The renderer hard-fails (`FileNotFoundError`) rather than emit a stub figure for any missing artifact.

## Threat Surface Scan
No new network endpoints, auth paths, or external file-access patterns. The plan's single trust boundary (2000ep artifact → figure) is mitigated as specified: `_require`/`_load_json` loud-fail (T-14-10 — no figure without its backing artifact), a same-stem reproducibility JSON per figure (T-14-11 — full figure↔data provenance), and distinct headline-vs-reproduction visual labels + companion conflation-guard strings (T-14-12 — D-14-10). No threat flags.

## Self-Check: PASSED
- `run_figure_suite.py` — FOUND (834 lines, `matplotlib.use("Agg")` before pyplot, `_require`/`_load_json` FileNotFoundError loud-fail, dual `savefig` + JSON companion, repo-root bootstrap, argparse `--figures-dir`)
- `results/figures/*.png` — FOUND (76 PNG; every PNG has a matching PDF + JSON)
- Plan verify command (verbatim) — PASS (run OK, triple-complete, Agg grep, loud-fail grep)
- Loud-fail probe — `_require` on a missing artifact RAISES `FileNotFoundError`
- `headline_vs_reproduction` + `cross_model_emd` + `cross_model_distribution` — FOUND (55-param IQP:SEL entrant; headline distinctly labelled)
- Commit `ab0daaf` — FOUND

---
*Phase: 14-paper-revision-release-freeze*
*Completed: 2026-05-19*
