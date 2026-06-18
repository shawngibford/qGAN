---
phase: 10-classical-baselines
plan: 04
subsystem: testing
tags: [baselines, wgan-gp, vae, autoregressive, tstr, jupyter, aggregation, fidelity-metrics]

# Dependency graph
requires:
  - phase: 10-classical-baselines (plans 01-03)
    provides: 50 baseline runs (wgan_mlp/cnn/lstm/vae/ar x A/B x seeds 42-46) + run_baselines driver/sweep
  - phase: 09.1-r1-m3-preprocessing-ablation
    provides: 10 quantum reference runs (transform_ablation/runs/{A,B}/{42..46}); reconstruct_od + TSTR-lite source
provides:
  - "06_baseline_comparison.ipynb — deterministic aggregation notebook (generator + .ipynb)"
  - "baseline_comparison.{json,md} (BASE-03) — apples-to-apples table: quantum + 5 new models x A/B x 5 seeds"
  - "baseline_classical_wgan.json (BASE-01) — {wgan_mlp,wgan_cnn,wgan_lstm} subset"
  - "baseline_nonadversarial.json (BASE-02) — {vae,ar} subset with train_protocol_notes"
affects: [phase-11-utility, phase-14-recommendation, aiche-revision-response]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic notebook generator (_build_baseline_notebook.py) mirroring 09.1 _build_analysis_notebook.py"
    - "Verbatim copy of reconstruct_od (A+B only, C deleted) + TSTR-lite (D-10-13, not promoted to core)"
    - "Long-form rows[] + models[] aggregate schema with model_kind dimension (D-10-16)"

key-files:
  created:
    - _build_baseline_notebook.py
    - 06_baseline_comparison.ipynb
    - results/baseline_comparison.json
    - results/baseline_comparison.md
    - results/baseline_classical_wgan.json
    - results/baseline_nonadversarial.json
  modified: []

key-decisions:
  - "Notebook content authored via a deterministic generator (_build_baseline_notebook.py), the 09.1 pattern (RESEARCH line 379) — notebook is regenerable"
  - "Metrics computed via revision.core.eval ONLY; zero new helpers added (D-10-20)"
  - "data_hash recomputed once and asserted equal across all 50 new configs; quantum equivalence by construction, no 09.1 config grep (D-10-15, Pitfall 4)"
  - "No Phase-10 recommendation emitted; Phase 14 deferral caption only (D-10-19)"

patterns-established:
  - "Pattern 1: _run_base(model_kind,...) parametrizes reconstruct_od for both the reused 09.1 quantum layout and the new baselines layout without altering the verbatim A/B inverse math"
  - "Pattern 2: BASE-01/02 are filtered projections of the BASE-03 long-form rows[]+models[]+tstr — single source of truth, no recomputation"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 22min
completed: 2026-05-17
---

# Phase 10 Plan 04: Classical Baselines Apples-to-Apples Comparison Summary

**`06_baseline_comparison.ipynb` aggregates all 50 new baseline runs + 10 reused 09.1 quantum runs into the BASE-01/02/03 deliverables: a long-form fidelity table (EMD/moments/ACF/DTW + transformed-EMD + TSTR-lite R²) for quantum + 3 matched-param classical WGAN-GP + VAE + AR across pipelines A and B, with the data-hash invariant verified and no Phase-10 recommendation.**

## Performance

- **Duration:** ~22 min
- **Tasks:** 2
- **Files modified:** 6 created (2 notebook artifacts + 4 result deliverables)

## Accomplishments
- Built `_build_baseline_notebook.py` (deterministic generator) producing `06_baseline_comparison.ipynb` (20 cells), executed end-to-end via nbconvert with no errors
- `reconstruct_od` copied verbatim (A branch + B branch byte-identical to 09.1; C branch deleted per D-10-05); base path parametrized so the same function resolves reused quantum runs (`transform_ablation/runs/<p>/<s>`) and new runs (`baselines/runs/<model>/<p>/<s>`)
- 1710 long-form rows `{model_kind,pipeline,seed,metric_name,scale,value}` over 6 models × 2 pipelines × 5 seeds, all metrics via `revision.core.eval` only
- `data_hash = 91e447d4624e25b3` recomputed once from `load_and_preprocess` and asserted equal across all 50 new configs; quantum equivalence documented by construction (no 09.1 grep)
- TSTR-lite (`TSTRLiteLSTM`/`r2_score_inline`/`train_eval_tstr`) copied verbatim; 13 (model×pipeline) blocks + real-only baseline with mse/r2 mean/std + per_init_seed (init seeds {40,41,42}, HELD_OUT_N=320)
- BASE-01/02/03 artifacts emitted; `baseline_comparison.md` renders one row per model per pipeline incl. the quantum reference, with a Phase-14 deferral caption (no recommendation)

## Task Commits

Each task was committed atomically:

1. **Task 1: Build 06_baseline_comparison.ipynb — reconstruct_od + eval metrics + data-hash verification** - `af01662` (feat)
2. **Task 2: TSTR-lite verbatim + emit BASE-01/02/03 artifacts** - `4e34944` (feat)

## Files Created/Modified
- `_build_baseline_notebook.py` - Deterministic notebook generator (canonical source, regenerable)
- `06_baseline_comparison.ipynb` - Executed aggregation notebook (20 cells)
- `results/baseline_comparison.json` - BASE-03: long-form rows[] + models[] + tstr block
- `results/baseline_comparison.md` - BASE-03: markdown table, one row per model per pipeline
- `results/baseline_classical_wgan.json` - BASE-01: {wgan_mlp,wgan_cnn,wgan_lstm} subset
- `results/baseline_nonadversarial.json` - BASE-02: {vae,ar} subset with train_protocol_notes

## Decisions Made
None beyond the plan — D-10-04/05/13/15/16/17/18/19/20/21 followed exactly as specified. Parameter counts read straight from on-disk Wave-2 `config.yaml` (quantum=75, wgan_mlp=74, wgan_cnn=73, wgan_lstm=78, vae=562, ar=3), matching the matched-parameter contract.

## Deviations from Plan

None - plan executed exactly as written.

The only stale-environment friction (`jupyter` script shebang pointing at an old `/Users/shawngibford/dev/qml/qGAN` path) was worked around by invoking `qgan_env/bin/python -m jupyter` — no code/config change, not a plan deviation.

## Issues Encountered
- The `qgan_env/bin/jupyter` console-script shebang references a stale interpreter path (`dev/qml/qGAN` vs `dev/phd/qGAN`). Resolved by running nbconvert via `python -m jupyter`. The notebook itself executes cleanly; no artifact impact.

## Observations (informational, no action — Phase 14 owns the decision)
- Pipeline B (log-returns) yields near-parity OD-EMD across all 6 families (~0.026-0.11), consistent with the 09.1 finding that B is the strong pipeline.
- VAE shows much lower OD-EMD on Pipeline A but its `train_protocol_notes` flags suspected posterior collapse (sample std << real std) — surfaced via the config note, not interpreted here per D-10-19.
- These are reported as table values only; no recommendation is made (D-10-19 — Phase 14, driven by Phase 11 utility numbers).

## Known Stubs
None. All deliverables are populated from real on-disk run artifacts; no placeholder/empty data paths.

## Next Phase Readiness
- BASE-01/02/03 deliverables complete and force-tracked; raw 47MB gitignored run dirs left untouched on disk (50 dirs intact, none git-added).
- Phase 11 (utility) can consume `baseline_comparison.json` long-form rows + tstr block directly.
- Phase 14 owns the headline baseline recommendation (deliberately not made here).

## Self-Check: PASSED

All 6 created deliverables + SUMMARY verified present on disk; all 3 commits (af01662, 4e34944, 9468c7b) verified in git log. STATE.md and ROADMAP.md confirmed untouched (orchestrator owns those).

---
*Phase: 10-classical-baselines*
*Completed: 2026-05-17*
