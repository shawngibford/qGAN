---
phase: 13-architecture-introspection
plan: 04
subsystem: revision-figures
tags: [introspection, figures, matplotlib, render-only, paper-ready]
requires:
  - "figures/training_progression.json (plan 13-03, INTRO-01)"
  - "figures/param_trajectory.json (plan 13-03, INTRO-02)"
  - "figures/entanglement_trajectory.json (plan 13-03, INTRO-03)"
provides:
  - "run_introspect_figures.py — render-only matplotlib renderer"
  - "figures/training_progression.{png,pdf} — INTRO-01 figure"
  - "figures/param_trajectory.{png,pdf} — INTRO-02 figure"
  - "figures/entanglement_trajectory.{png,pdf} — INTRO-03 figure"
affects:
  - "Phase 14 (paper) consumes these 6 figure files for the R2-6 black-box rebuttal"
tech-stack:
  added: []
  patterns:
    - "render-only script idiom: argparse --figures-dir + _find_repo_root + Path defaults + print written paths (mirrors run_dualscale_fidelity.py)"
    - "matplotlib.use('Agg') before pyplot import for headless rendering"
    - "loud FileNotFoundError on any missing companion JSON (no silent partial figure)"
key-files:
  created:
    - "run_introspect_figures.py"
    - "figures/training_progression.png"
    - "figures/training_progression.pdf"
    - "figures/param_trajectory.png"
    - "figures/param_trajectory.pdf"
    - "figures/entanglement_trajectory.png"
    - "figures/entanglement_trajectory.pdf"
  modified: []
decisions:
  - "Per-row shared x-range (0.5/99.5 percentile clip) for training_progression so quantum-vs-classical distribution shapes are visually comparable within a target row (D-13 figure discretion)"
  - "Entropy/purity reference bounds derived from the 2-qubit smaller subsystem of the {0,1}|{2,3,4} bipartition: max entropy = ln4, min purity = 1/4 (D-13-09)"
metrics:
  duration: "~9 min"
  completed: "2026-05-19"
  tasks: 1
  files: 7
---

# Phase 13 Plan 04: INTRO Figure Rendering Summary

Render-only matplotlib script that turns the three plan-03 reproducibility JSON files into six paper-ready figure artifacts (png+pdf) for the R2-6 black-box rebuttal, with zero training/sampling so every figure is traceable to its companion JSON (ROADMAP criterion 4).

## What Was Built

`run_introspect_figures.py` — a single render-only entry point that:

- Loads the three companion JSON (`training_progression.json`, `param_trajectory.json`, `entanglement_trajectory.json`) and raises a clear `FileNotFoundError` if any is absent (T-13-13 mitigation).
- **INTRO-01 `training_progression.{png,pdf}`** — 4×5 grid: one row per target (quantum + `wgan_mlp`/`wgan_cnn`/`wgan_lstm`), one column per snapshot epoch {0,250,500,750,1000}; each cell a density histogram of that target's generated samples, all four targets visually side-by-side (D-13-08).
- **INTRO-02 `param_trajectory.{png,pdf}`** — (a) PQC parameter L2-norm vs epoch line; (b) 75-parameter angle-distribution step histograms, one per snapshot epoch; annotated variant=V1, depth=4, topology=range.
- **INTRO-03 `entanglement_trajectory.{png,pdf}`** — (a) Von Neumann entanglement entropy vs epoch with ln4 max reference; (b) reduced-state purity Tr(ρ²) vs epoch with 1/4 and 1.0 bounds; bipartition string `{0,1}|{2,3,4}` annotated verbatim from JSON metadata (D-13-09).

## How It Was Verified

- Plan automated check passed: `python -m revision.run_introspect_figures` then all 6 files non-empty → `FIGURES_OK`.
- Render-only assertion: grep shows no `train_wgan_gp` / `QuantumGenerator(` / `.sample(` / `load_and_preprocess` in the script (T-13-12 mitigation).
- `matplotlib.use("Agg")` + `fig.savefig` present; reads all three JSON constants.
- Idempotency: second run exits 0 and overwrites the 6 files.
- Loud-failure: `--figures-dir /tmp/nonexistent` raises the clear FileNotFoundError.
- Visual spot-check: training_progression renders all 4 targets across 5 epochs; entanglement figure shows the verbatim bipartition annotation and reference bounds.

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

None.

## Known Stubs

None — every figure is rendered from real plan-03 JSON data; no placeholder/mock data sources.

## Self-Check: PASSED

- All 7 created files verified present on disk.
- Task commit `e3fb61b` verified present in git history.
