---
phase: 12-sensitivity-analysis
plan: 01
subsystem: sensitivity-harness
tags: [pennylane, inference, shot-noise, noise-channels, fidelity, smoke-gate]
requires:
  - results/transform_ablation/runs/<pipeline>/<seed>/checkpoint.pt (frozen params_pqc)
  - results/transform_ablation/runs/<pipeline>/<seed>/samples.npy (frozen analytic reference)
  - results/transform_ablation/runs/<pipeline>/<seed>/inverse_kwargs.npz
  - results/fidelity_dualscale.json (quantum B/42 OD reference)
  - revision.core (HPO constants, QuantumGenerator, full_metric_suite, inverse_logreturns)
provides:
  - run_sensitivity.py (SENS-01/02 per-cell inference driver, CLI)
  - results/sensitivity/runs/analytic/B/42/{config.yaml,samples.npy,metrics.json} (faithful smoke bundle)
affects:
  - Plan 12-02 (full SENS-01/02 grid sweep — consumes this driver)
tech-stack:
  added: []
  patterns:
    - "qml.set_shots transform (0.44 API) for finite-shot QNodes"
    - "default.mixed + per-layer DepolarizingChannel/AmplitudeDamping for noise channels"
    - "PennyLane 0.44.0 startup version assertion (fail-loud)"
key-files:
  created:
    - run_sensitivity.py
    - results/sensitivity/runs/analytic/B/42/config.yaml
    - results/sensitivity/runs/analytic/B/42/samples.npy
    - results/sensitivity/runs/analytic/B/42/metrics.json
  modified: []
decisions:
  - "Per-layer channel insertion (after each entangling block) chosen as the documented SENS-02 default (RESEARCH Assumption A1 / Open Q2 RESOLVED)"
  - "PennyLane 0.44.0 pinned via startup assert; ./qgan_env (0.43.0) explicitly forbidden — deliberate documented deviation (RESEARCH Open Q1 (a))"
metrics:
  duration: ~15 min
  completed: 2026-05-18
---

# Phase 12 Plan 01: SENS-01/02 Inference Harness Summary

Built `scripts/run_sensitivity.py` — the per-cell SENS-01 (shot-noise) /
SENS-02 (noise-channel) inference driver — and proved it byte-faithful to the
frozen Phase 09.1/10 analytic reference via the analytic/B/42 smoke cell.

## What Was Built

A single-file CLI driver, one `(pipeline, seed, condition)` cell per
invocation, that reloads the frozen 75-element `params_pqc` tensor (NO
retraining — D-12-01), builds the appropriate noisy/finite-shot QNode in the
driver (never in `core/` — D-10-13), regenerates samples honoring the
load-bearing `*0.1` + `np.random.default_rng(seed)` contracts (Pitfall 3),
reconstructs the OD scale with the verbatim Pipeline-A/B recipe including the
`seed*7919+1` od_start draw (Pitfall 4), recomputes the unchanged dual-scale
fidelity suite (`full_metric_suite`, D-12-03 / EVAL-05), and writes an
idempotent per-cell bundle.

- **Task 1** (`bf5fd57`): harness skeleton — repo-root resolver, PennyLane
  0.44.0 startup gate, params reload, verbatim generation + reconstruction
  contracts.
- **Task 2** (`8c7c05e`): `make_shot_qnode` (`qml.set_shots` transform, NO
  `shots=` device kwarg), `make_noisy_qnode` (`default.mixed` + per-layer
  `DepolarizingChannel`/`AmplitudeDamping`), 11 condition tokens, dual-scale
  fidelity recompute, idempotent bundle, CLI.
- **Task 3** (`cb511ab`): smoke gate — analytic/B/42 cell produced and proven
  faithful.

## Smoke Gate Result (harness-faithfulness proof)

| Check | Detector | Result | Threshold | Pass |
|-------|----------|--------|-----------|------|
| Analytic cell reuses frozen samples.npy (no regen) | no-regeneration invariant | `True` | — | ✓ |
| Regenerated analytic samples vs frozen `transform_ablation/runs/B/42/samples.npy` | Pitfall 3 (`*0.1` + `default_rng(seed)` + analytic device) | **max abs err 1.7016768716e-08** | < 1e-6 | ✓ |
| Analytic-cell OD-scale EMD vs `fidelity_dualscale.json` quantum B/42 OD | Pitfall 4 (`*0.1` + `seed*7919+1` reconstruction) | computed `0.022937980562900893`, ref `0.022937980562900886`, **abs delta 6.94e-18** | < 1e-6 | ✓ |

All three checks pass. The device-swap harness is faithful to the frozen
Phase 09.1/10 numbers — the full SENS-01/02 grid (Plan 02) is now safe to run.

The ~1.7e-08 sample max-abs-error (not exactly 0) is float32 QNode evaluation
non-determinism on `default.qubit`, far inside the 1e-6 fp tolerance. The
OD-scale EMD delta (~7e-18) is at machine epsilon, confirming the
reconstruction contracts (`*0.1`, `seed*7919+1`, `od[:,:10]` truncation) are
wired exactly as Phase 11.

## Channel-Insertion Strategy (documented default)

**Per-layer insertion** — one noise channel on every wire immediately AFTER
each entangling (range-based CNOT) block, for all `num_layers` layers. This is
the conventional NISQ deployment-noise model and the documented default
(RESEARCH Assumption A1 / Open Q2 RESOLVED: per-layer). Recorded in the
`make_noisy_qnode` docstring and emitted into every cell's `config.yaml` as
`channel_insertion: "per-layer (after each entangling block)"`.

## Deliberate, Documented Deviations

### 1. [Plan-prescribed] PennyLane 0.44.0 pin (T-12-01 mitigation)

The driver asserts `qml.__version__ == "0.44.0"` at import and fails loud
otherwise, with a message explicitly forbidding `./qgan_env` (PennyLane
0.43.0). This is a deliberate deviation from the analog sweeps'
venv-preference: 0.43 vs 0.44 differ in the `qml.set_shots` transform API and
the `shots=` device-kwarg deprecation. `qgan_env` is NOT upgraded (that would
invalidate the frozen 09.1/10 reproduction baseline). The Plan 02 sweep
wrapper must select an explicit 0.44.0 interpreter (system `python3`) and not
prefer the venv. Prescribed by the plan (RESEARCH Open Q1 recommendation (a));
documented in the module docstring.

### 2. [Plan-prescribed] Circuit-body duplication in the driver (D-10-13 preserved)

`make_noisy_qnode` re-emits a verbatim copy of
`QuantumGenerator.generator_circuit` (quantum.py:122-171) as
`noisy_generator_circuit` so per-layer channels can be inserted (the original
ends with `qml.expval` returns and cannot have channels appended after the
call). This copy lives in `scripts/run_sensitivity.py`, NOT `core/`, so it
does NOT violate D-10-13. `git diff --stat core/` is empty across all
three tasks. Documented in the function docstring as a deliberate noise-study
duplication.

### 3. [Rule 1 — cosmetic] Reworded a `qml.device` inline comment

The Task 2 acceptance grep `grep 'qml.device(' | grep -c 'shots='` must be 0.
The original comment `# NO shots= kwarg (Pitfall 1)` contained the literal
`shots=` and produced a false-positive count of 1 (no `qml.device` call
actually passes `shots=`). Reworded to `# analytic device; shot count via
transform only (Pitfall 1)` so the grep is unambiguously 0. Behavior
unchanged. Folded into the Task 2 commit (`8c7c05e`).

## Verification Status

- `python scripts/run_sensitivity.py --help` exits 0; CLI exposes
  `--pipeline {A,B}`, `--seed`, `--condition` (11 choices incl. `ampdamp_0.01`).
- `ast.parse` clean; no `multiprocessing`; no `diff_method="backprop"|"adjoint"`;
  no `shots=` kwarg on any `qml.device(` line.
- 11 condition tokens recognized: `analytic`, `shots_8192`, `shots_1024`,
  `depol_{0.0,0.001,0.01,0.05}`, `ampdamp_{0.0,0.001,0.01,0.05}` (the four
  `ampdamp_*` cover γ ∈ {0,0.001,0.01,0.05} per D-12-02 / SC-2).
- `full_metric_suite` imported from `revision.core.eval`, called unmodified;
  metrics emitted at two `scale` values (OD always; log_return for Pipeline B).
- `analytic` branch reads frozen `transform_ablation/.../samples.npy`, does not
  call the generation function.
- Smoke bundle `results/sensitivity/runs/analytic/B/42/metrics.json`
  exists, non-empty, contains `emd`.
- `git diff --quiet core/` clean (CORE_CLEAN) for all three tasks.

## Known Stubs

None. The driver is a complete, exercised CLI; the analytic smoke cell is a
real produced-and-verified artifact bundle, not a placeholder.

## Deferred Issues

None. All three tasks completed with verification passing; no auto-fix attempt
limit reached.

## Self-Check: PASSED

- FOUND: run_sensitivity.py
- FOUND: results/sensitivity/runs/analytic/B/42/config.yaml
- FOUND: results/sensitivity/runs/analytic/B/42/samples.npy
- FOUND: results/sensitivity/runs/analytic/B/42/metrics.json
- FOUND commit bf5fd57 (Task 1)
- FOUND commit 8c7c05e (Task 2)
- FOUND commit cb511ab (Task 3)
