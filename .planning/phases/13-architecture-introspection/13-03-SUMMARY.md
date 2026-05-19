---
phase: 13-architecture-introspection
plan: 03
subsystem: introspection
tags: [pennylane, pytorch, pqc, wgan-gp, callback-instrumentation, entanglement, reproducibility-json, tdd]

# Dependency graph
requires:
  - phase: 13-architecture-introspection
    plan: 01
    provides: QuantumGenerator.introspect() + topology="range" V1 + greenfield tests/ pytest package
  - phase: 08-core-module-extraction
    provides: train_wgan_gp dormant callback= hook + revision/core/ shared modules
  - phase: 10-classical-baselines
    provides: run_baselines.build_dataset_for_pipeline (Pipeline B) + _WGAN_GENERATORS map
provides:
  - revision/run_introspect.py — callback-snapshot driver (4 targets) + --assemble companion-JSON emitter
  - training_progression.json (INTRO-01) — quantum + 3 classical side-by-side generated-distribution snapshots
  - param_trajectory.json (INTRO-02) — V1 PQC param-norm + angle-histogram source
  - entanglement_trajectory.json (INTRO-03) — vn_entropy + purity trajectory + {0,1}|{2,3,4} bipartition metadata
  - tests/test_introspect_callback.py — INTRO-01/02/03 closure regression
affects: [13-04-figure-rendering, phase-14-paper]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "callback= snapshot closure re-generates from the live generator (callback passes only a scalar dict — never samples)"
    - "training noise contract copied verbatim into the closure (training.py:304-313) for distribution fidelity (T-13-08)"
    - "terminal SNAP index relabel (max(snap) -> num_epochs) instead of a hardcoded 999->1000 (T-13-10)"
    - "hasattr(generator,'introspect') guard separates quantum vs classical capture without crashing the try/except-wrapped hook (T-13-09)"
    - "in-driver monkeypatch of torch.backends.mps.is_available (restored in finally) forces train_wgan_gp's CPU path for the quantum target without touching frozen revision/core/"

key-files:
  created:
    - revision/run_introspect.py
    - tests/test_introspect_callback.py
    - revision/results/figures/training_progression.json
    - revision/results/figures/param_trajectory.json
    - revision/results/figures/entanglement_trajectory.json
    - revision/results/figures/_introspect_quantum.json
    - revision/results/figures/_introspect_wgan_mlp.json
    - revision/results/figures/_introspect_wgan_cnn.json
    - revision/results/figures/_introspect_wgan_lstm.json
  modified: []

key-decisions:
  - "Test path is top-level tests/ (pytest.ini testpaths=tests, conftest at tests/conftest.py) — matches plan's tests/test_introspect_callback.py and the Phase 13 Wave-0 greenfield suite, NOT revision/tests/"
  - "make_snapshot_cb factory exposes the closure independently so the TDD test exercises it on a tiny synthetic eval loop (no 1000-epoch training in the test)"
  - "Generic terminal relabel max(snap)->num_epochs (production: 999->1000) so the same closure works for the short-run rescaled SNAP {0,10,20,29}->30 in the test"
  - "_introspect_<target>.json intermediates committed (not just .gitignored) — plan acceptance asserts their existence and they are the idempotent re-assembly inputs"

requirements-completed: [INTRO-01, INTRO-02, INTRO-03]

# Metrics
duration: ~3h (quantum 1000-epoch CPU statevector run dominated wall time)
completed: 2026-05-19
---

# Phase 13 Plan 03: Architecture-Introspection Instrumented Runs Summary

**Wired a `callback=` snapshot closure into the dormant `train_wgan_gp` hook (zero training-loop surgery), ran four instrumented 1000-epoch single-seed-42 Pipeline-B trainings (V1 quantum depth-4 range + wgan_mlp/cnn/lstm), and emitted the three INTRO-01/02/03 reproducibility companion JSON files — the R2-6 "what is it learning?" rebuttal data.**

## What Was Built

- **`revision/run_introspect.py`** — `make_snapshot_cb()` builds a closure that, on SNAP epochs `{0,250,500,750,999}`, re-generates a batch from the live generator under the exact training noise contract (CPU + `rng.uniform(NOISE_LOW,NOISE_HIGH,(NUM_QUBITS,BATCH_SIZE))` float32 → `.to(float64)*0.1`), hasattr-guards `introspect()`, relabels the terminal index 999 → 1000, and appends one record per snapshot. `--target {quantum,wgan_mlp,wgan_cnn,wgan_lstm}` runs one instrumented training and writes an idempotent `_introspect_<target>.json`. `--assemble` reads the four intermediates and emits the three companion JSON files.
- **`tests/test_introspect_callback.py`** — TDD regression exercising the closure on a tiny synthetic eval loop for both a real `QuantumGenerator` (5 captured fields incl. vn_entropy/purity) and a classical `WGANMLPGenerator` (samples-only, no crash), plus the snapshot-std ≈ metrics-std noise-contract sanity (T-13-08).
- **Three companion JSON files** (ROADMAP criterion 4; plan 04 renders figures from these):
  - `training_progression.json` (INTRO-01): quantum + 3 classical side-by-side generated distributions, 5 snapshots each at `[0,250,500,750,1000]`, metadata `pipeline=B seed=42`.
  - `param_trajectory.json` (INTRO-02): quantum-only `param_norm[5]` + `param_angles[5][75]`, metadata `variant=V1 depth=4 topology=range`.
  - `entanglement_trajectory.json` (INTRO-03): `vn_entropy[5]` + `purity[5]`, metadata `bipartition="{0,1}|{2,3,4}"` verbatim (D-13-09 / T-13-11).

## Results (real run data, seed 42, 1000 epochs)

| Snapshot epoch | 0 | 250 | 500 | 750 | 1000 |
|----------------|---|-----|-----|-----|------|
| vn_entropy | 1.2443 | 1.2072 | 1.2526 | 1.1677 | 1.2090 |
| purity | 0.3231 | 0.3334 | 0.3196 | 0.3475 | 0.3313 |
| param_norm | 4.3977 | 4.3990 | 4.4041 | 4.4076 | 4.4140 |

INTRO-03 bounds hold on real data: vn_entropy ∈ [1.17, 1.25] < ln4 ≈ 1.386; purity ∈ [0.32, 0.35] ∈ [0.25, 1]. All four targets produced exactly 5 snapshots at `[0,250,500,750,1000]` (terminal 999→1000 relabel verified).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Snapshot closure left the generator on CPU, crashing the next training epoch**
- **Found during:** Task 2 (first classical smoke run)
- **Issue:** `train_wgan_gp` moves the generator onto MPS and re-generates on it every epoch. The closure's `generator.to("cpu")` mutates the module in-place, so the epoch after a snapshot crashed with a cross-device matmul (`Tensor for argument weight is on cpu but expected on mps`).
- **Fix:** Capture the generator's device before the snapshot, run the CPU snapshot in a `try`, and restore the original device in a `finally`. Training continues unaffected.
- **Files modified:** revision/run_introspect.py
- **Commit:** e2efefb

**2. [Rule 3 - Blocking] QuantumGenerator cannot train on MPS (PennyLane mis-coerces MPS → CUDA)**
- **Found during:** Task 2 (quantum smoke run)
- **Issue:** `train_wgan_gp` unconditionally moves the generator to MPS when available (training.py:264). Once `params_pqc` lands on MPS, PennyLane's `_coerce_types_torch` tries to build a CUDA device from the MPS tensor and raises `AssertionError: Torch not compiled with CUDA enabled` — the quantum forward pass cannot run on MPS at all on this machine. No existing test trains a *quantum* generator through `train_wgan_gp` on an MPS box, so the latent incompatibility surfaced here first.
- **Fix:** `_force_cpu_for_quantum()` context manager monkeypatches `torch.backends.mps.is_available → False` ONLY around the quantum `train_wgan_gp` call (restored in `finally`), so the device selector falls through to the float64 CPU path identical to the frozen 09.1 quantum runs. Classical WGANs are pure `torch.nn` and keep running on MPS. **`revision/core/` is byte-untouched** — the fix lives entirely in the in-scope driver, preserving the frozen Phase 8-12 reproducibility invariant. This was scoped as a Rule-3 blocking-issue fix (not Rule-4 architectural) because no frozen artifact, schema, or core module changed.
- **Files modified:** revision/run_introspect.py
- **Commit:** e2efefb

## TDD Gate Compliance

- RED: `8afe796` — `test(13-03)` failing regression (ModuleNotFoundError, then behavior assertions).
- GREEN: `73305ab` — `feat(13-03)` driver implementation; 4/4 tests pass.
- Follow-up `fix(13-03)` `e2efefb` (device-safety) kept the suite green.
- No REFACTOR commit (implementation clean; no separate refactor needed).

## Threat Surface Scan

No new security-relevant surface (local-Mac scientific compute; no network/auth/PII). All STRIDE register mitigations satisfied:
- T-13-08 (wrong noise contract): closure copies training.py:304-313 verbatim; test asserts snapshot std ≈ metrics std.
- T-13-09 (callback try/except swallows exception): test asserts exact snapshot count; all 4 intermediates have exactly 5 snapshots.
- T-13-10 (off-by-one drops final epoch): SNAP includes 999 with explicit terminal→num_epochs relabel; all intermediates show epoch 1000 present.
- T-13-11 (bipartition not recorded): `entanglement_trajectory.json` metadata `bipartition == "{0,1}|{2,3,4}"`, verified in the automated check.

## Self-Check: PASSED

- FOUND: revision/run_introspect.py
- FOUND: tests/test_introspect_callback.py
- FOUND: revision/results/figures/training_progression.json
- FOUND: revision/results/figures/param_trajectory.json
- FOUND: revision/results/figures/entanglement_trajectory.json
- FOUND commit 8afe796 (test/RED), 73305ab (feat/GREEN), e2efefb (fix), 7600fa7 (runs+JSON)
- 4/4 tests pass; revision/core/ byte-untouched
