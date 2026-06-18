---
phase: 13-architecture-introspection
plan: 02
subsystem: experiment-driver
tags: [pennylane, pytorch, wgan-gp, ansatz, sweep, dual-scale, pytest]

# Dependency graph
requires:
  - phase: 13-architecture-introspection
    plan: 01
    provides: QuantumGenerator(topology=...) selector + 75/135 param counts
  - phase: 09.1
    provides: frozen V1 Pipeline-B 5-seed quantum runs (transform_ablation/runs/B/{42..46})
  - phase: 10
    provides: run_baselines.py / run_baselines_sweep.sh structural template; run_dualscale_fidelity.py schema
provides:
  - run_ansatz.py — single (variant,seed) quantum WGAN driver (V2/V3)
  - run_ansatz_sweep.sh — idempotent resumable 10-run sweep (xargs -P 2)
  - run_ansatz_comparison.py — ARCH-02 dual-scale aggregator
  - tests/test_ansatz_json_schema.py — ARCH-02 schema regression
  - results/ansatz_comparison.json — ARCH-02 comparison table (PENDING — see Blocker)
affects: [phase-14-paper]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "quantum process pinned to CPU statevector path (RESEARCH Pitfall 6) via process-local mps.is_available guard in the driver — training.py byte-unchanged"
    - "2-tuple (variant,seed) sweep key cloned from the 3-tuple baseline sweep"

key-files:
  created:
    - run_ansatz.py
    - run_ansatz_sweep.sh
    - run_ansatz_comparison.py
    - tests/test_ansatz_json_schema.py
  modified: []

key-decisions:
  - "Pitfall-6 CPU pin lives in run_ansatz.py (driver), NOT core/training.py — byte-unchanged discipline preserved and the frozen V1 reference is untouched"
  - "V1 rows re-score the frozen transform_ablation/runs/B samples with full_metric_suite UNCHANGED (D-10-20); NO V1 training recompute and NO V1 run-dir created (D-13-01)"
  - "ansatz_comparison.json reuses full_metric_suite (D-10-20) on BOTH scales (OD via inverse_logreturns + log_return) mirroring run_dualscale_fidelity.py"

requirements-completed: []  # ARCH-01/02 pending sweep completion + aggregation (see Blocker)

# Metrics
duration: in-progress (paused at Bash-permission blocker)
completed: PENDING
---

# Phase 13 Plan 02: Ansatz Comparison Sweep Summary

**ARCH-01/02 quantum-vs-quantum ansatz comparison: single-run driver + 10-run
resumable sweep + dual-scale aggregator + schema test delivered; the 10-run
sweep was executing (2/10 complete, healthy) when execution was halted by an
environment Bash-permission denial before sweep completion and aggregation.**

## Status: INCOMPLETE — paused at a blocking environment denial

Task 1 is fully complete and committed. Task 2 source artifacts
(`run_ansatz_comparison.py`, `tests/test_ansatz_json_schema.py`) are written
and syntax-verified, the schema test skips cleanly pre-emission, and the 10-run
sweep was launched and running healthily — but the environment began denying
all `Bash` invocations mid-sweep, so the remaining required steps could not be
performed.

## Accomplishments

- **Task 1 (committed):** `run_ansatz.py` — single `(variant, seed)`
  quantum WGAN driver cloned from `run_baselines.py`'s WGAN branch.
  `QuantumGenerator(num_layers=depth, topology=topology)`, 1000 epochs, NO
  `early_stopper` (D-13-05), `spectral_loss_weight=0.0` default (D-13-06),
  Pipeline-B verbatim, idempotent 5-file bundle with `data_hash` +
  `ansatz/depth/topology/parameter_count`. `V2=(8,range,135)`,
  `V3=(4,linear,75)`; V1 excluded from `--variant` choices (reused per
  D-13-01). `run_ansatz_sweep.sh` — clone of
  `run_baselines_sweep.sh`: 10-run V2/V3 × {42..46} matrix, 2-tuple
  `(variant,seed)` key, `xargs -P 2 -L 1` only, atomic
  `flock`+tmpfile+`os.rename` status, `--parallel 1|2` guard, no-Pool header
  preserved. Verified: `ast.parse` OK, `bash -n` OK, `--dry-run` lists exactly
  the 10-run V2/V3 matrix (V1 absent), zero non-comment `multiprocessing.Pool`
  code usage, source contains `QuantumGenerator(... topology=`,
  `train_wgan_gp(... num_epochs=1000`, no `early_stopper=`.

- **Pitfall-6 fix (committed):** discovered every quantum run failing in ~3s
  because the shared `train_wgan_gp` unconditionally moves the generator to
  MPS, and PennyLane `default.qubit` + torch interface then probes a
  nonexistent CUDA device. Fixed in the driver (process-local
  `torch.backends.mps.is_available = lambda: False` in `_train_wgan`) — pins
  the quantum statevector path to CPU exactly as RESEARCH Pitfall 6
  prescribes, identical to the 09.1/10 V1 reference path. Verified with a full
  1-epoch end-to-end V3 run writing the complete 5-file bundle.

- **Task 2 source (written, not yet committed):**
  - `run_ansatz_comparison.py` — clones the
    `run_dualscale_fidelity.py` envelope + Pipeline-B dual-scale
    reconstruction; emits `ansatz_comparison.json` with `schema`,
    `model_kinds`, `ansatz_variants` (V1 reuse / V2 / V3 with
    depth/topology/parameter_count), `seeds`, `scales`, `metric_helpers`,
    `data_equivalence` (by-construction note), and long-form `rows[]`. V1 rows
    read the frozen `transform_ablation/runs/B/{42..46}` samples and re-score
    with `full_metric_suite` UNCHANGED (D-10-20) — NO new V1 training, NO V1
    run-dir created (D-13-01). V2/V3 rows from the new
    `results/ansatz/runs/<variant>/<seed>/` bundles. Both scales
    emitted (OD via `inverse_logreturns`, plus `log_return`).
  - `tests/test_ansatz_json_schema.py` — asserts top-level keys;
    `ansatz_variants` V1(4/range/75)/V2(8/range/135)/V3(4/linear/75); rows
    non-empty with the 9 required fields; variant set == {V1,V2,V3}; scale set
    ⊆ {log_return,OD}; per-row dims match the variant registry; V1 source
    string contains "no recompute"/"D-13-01". Syntax-verified; skips cleanly
    (7 skipped) while the JSON is not yet emitted.

- **Sweep execution (in progress at halt):** launched
  `./run_ansatz_sweep.sh --parallel 2`. At the point Bash access was
  lost: 2/10 complete (V2/42 wall=362s rc=0, V2/43 wall=360s rc=0), V2/44 and
  V2/45 running, no failures — the Pitfall-6 fix is holding. The sweep is
  resumable and idempotent, so it can be driven to completion and the
  aggregator + test run on resume.

## Task Commits

1. **Task 1: run_ansatz.py driver + run_ansatz_sweep.sh sweep** — `08588c3` (feat)
2. **Rule 3 fix: pin quantum training to CPU (RESEARCH Pitfall 6)** — `6b91a3e` (fix)

Task 2 artifacts are written to the working tree but NOT yet committed (Bash
denied before the Task-2 commit could be made).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Quantum training failed on MPS (RESEARCH Pitfall 6)**
- **Found during:** Task 2 (first sweep launch — all runs failed in ~3s)
- **Issue:** `core/training.py` unconditionally moves any generator
  to MPS when available. For a `QuantumGenerator`, PennyLane's `default.qubit`
  + torch interface then mis-coerces the non-CPU tensors and probes a
  nonexistent CUDA device, raising "Torch not compiled with CUDA enabled" and
  failing every quantum run immediately.
- **Fix:** Process-local guard in `run_ansatz._train_wgan`
  (`torch.backends.mps.is_available = lambda: False`) so the quantum
  statevector path runs on CPU — exactly RESEARCH Pitfall 6, and identical to
  the 09.1/10 V1 reference runs. `core/training.py` is NOT modified
  (byte-unchanged discipline; not in this plan's `files_modified`); the frozen
  V1 artifacts are untouched.
- **Files modified:** `run_ansatz.py`
- **Verification:** 1-epoch end-to-end V3 run wrote the full 5-file bundle;
  the real sweep then completed V2/42, V2/43 with rc=0.
- **Committed in:** `6b91a3e`

## Blocker (execution halted here)

After launching the sweep and writing the Task-2 source artifacts, the
execution environment began **denying all `Bash` tool invocations** (status
checks, the aggregator run, the schema-test run, and all git commits). The
sweep continues running in the background and is resumable, but this agent
cannot:

1. Poll the sweep to `all_complete: true` (currently 2/10).
2. Run `python -m revision.run_ansatz_comparison` to emit
   `results/ansatz_comparison.json`.
3. Run `pytest tests/test_ansatz_json_schema.py -x -q` (the Task-2
   verification gate).
4. Commit Task 2 (`run_ansatz_comparison.py`, `tests/test_ansatz_json_schema.py`).
5. Commit this SUMMARY.md.

### Resume instructions (Bash access restored)

From `/Users/shawngibford/dev/phd/qGAN/.claude/worktrees/agent-a7cd32bbf9bc4982c`
(a convenience symlink `qgan_env -> /Users/shawngibford/dev/phd/qGAN/qgan_env`
already exists locally and is NOT git-tracked):

```bash
# 1. Drive the resumable sweep to completion (idempotent — skips done runs):
./run_ansatz_sweep.sh --parallel 2
#    Wait until results/ansatz/sweep_status.json -> all_complete: true (10 runs).

# 2. Emit the comparison JSON:
./qgan_env/bin/python -m revision.run_ansatz_comparison

# 3. Run the Task-2 verification gate:
./qgan_env/bin/python -m pytest tests/test_ansatz_json_schema.py -x -q
./qgan_env/bin/python -c "import json;d=json.load(open('results/ansatz_comparison.json'));vs={v['variant'] for v in d['ansatz_variants']};assert vs=={'V1','V2','V3'},vs;assert d['rows'];print('OK',len(d['rows']),'rows')"

# 4. Commit Task 2 + this SUMMARY (stage files individually; never `git add .`,
#    never stage the qgan_env symlink; results/ is gitignored so the
#    JSON/run-dirs are intentionally not committed):
git add run_ansatz_comparison.py tests/test_ansatz_json_schema.py
git commit -m "feat(13-02): ansatz_comparison.json aggregator + ARCH-02 schema test"
git add .planning/phases/13-architecture-introspection/13-02-SUMMARY.md
git commit -m "docs(13-02): complete ansatz-comparison-sweep plan"
```

Note: `results/` is git-ignored (`.gitignore`), matching the
Phase-10/11 precedent — the sweep run-dirs and `ansatz_comparison.json` are
local artifacts and are intentionally not committed; the paper-writing phase
reads them from the local results tree.

## Threat Mitigations Applied

- **T-13-04** (V1 mis-attribution / silent recompute): the aggregator resolves
  V1 strictly from `transform_ablation/runs/B/{42..46}`, never creates a V1
  training dir, re-scores frozen samples with `full_metric_suite` UNCHANGED,
  and records the by-construction `data_equivalence` note; the schema test
  asserts the V1 `source` string contains "no recompute"/"D-13-01".
- **T-13-05** (no `multiprocessing.Pool`): driver + sweep use `xargs -P 2 -L 1`
  only; the only `multiprocessing.Pool` text is the preserved ban
  header/docstring (zero code usage — verified by grep).
- **T-13-06** (torn `sweep_status.json`): cloned the proven
  `flock -x 9` + tmpfile + `os.rename` atomic update verbatim.
- **T-13-07** (mixed-budget contamination): `is_complete()` checks the full
  5-file bundle; the driver does idempotent `shutil.rmtree` on rerun.

## Known Stubs

None — all delivered code is wired. `ansatz_comparison.json` is not a stub but
a pending artifact awaiting sweep completion (blocked by the environment
denial, not by missing logic).

## Self-Check

- Created files present: `run_ansatz.py` (committed `08588c3`,
  fixed `6b91a3e`), `run_ansatz_sweep.sh` (committed `08588c3`),
  `run_ansatz_comparison.py` (written, uncommitted),
  `tests/test_ansatz_json_schema.py` (written, uncommitted),
  `13-02-SUMMARY.md` (this file, uncommitted).
- Commits verified during execution: `08588c3`, `6b91a3e`.
- **Self-Check: PARTIAL** — Task 1 + Pitfall-6 fix committed and verified;
  Task 2 source written and syntax-verified; sweep + aggregation + Task-2
  commit + SUMMARY commit could not be completed because `Bash` was denied
  mid-execution. See Blocker → Resume instructions.

---
*Phase: 13-architecture-introspection*
*Status: INCOMPLETE — resume per Blocker section once Bash access is restored*
