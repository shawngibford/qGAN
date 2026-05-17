---
phase: 10-classical-baselines
plan: 03
subsystem: infra
tags: [bash, xargs, flock, sweep-driver, wgan-gp, pytorch, mps, apple-silicon]

# Dependency graph
requires:
  - phase: 10-classical-baselines (plan 10-02)
    provides: revision/run_baselines.py per-run CLI driver + classical/nonadversarial models
provides:
  - revision/run_baselines_sweep.sh — resumable 50-run sweep driver (5 models x 2 pipelines x 5 seeds)
  - WGAN-GP training path validated on Apple MPS (device wiring + float64->float32 GP fix)
affects: [10-04, baseline-comparison, wave-4-aggregation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sweep driver = verbatim mirror of run_ablation_sweep.sh with .npz-aware is_complete + 3-tuple status key"
    - "Device-aware WGAN-GP: compute_dtype=float32 on MPS, float64 on CPU/CUDA; per-seed reproducibility preserved per path"

key-files:
  created:
    - revision/run_baselines_sweep.sh
  modified:
    - revision/core/training.py
    - revision/run_baselines.py

key-decisions:
  - "SPLIT MODE (user directive): build sweep driver + prove MPS WGAN-GP path; full 50-run sweep delegated to orchestrator background execution after merge"
  - "MPS WGAN-GP fix: move generator (device only, native float32) + critic (device + compute_dtype) onto device; compute_dtype=float32 on MPS (no float64), float64 on CPU; no CPU fallback for WGAN runs (user directive)"
  - "Sample generation moves trained generator back to CPU so samples.npy stays bit-identical float64 *0.1 space"

patterns-established:
  - "Sweep status writer keyed on (model,pipeline,seed) 3-tuple, total_count=50, flock+tempfile+os.rename atomic write"
  - "MPS vs CPU dtype split inside train_wgan_gp preserves prior CPU bit-identical reproduction while enabling MPS"

requirements-completed: [BASE-01, BASE-02]

# Metrics
duration: ~25min
completed: 2026-05-17
---

# Phase 10 Plan 03: Classical Baselines Sweep Driver + MPS WGAN-GP Validation Summary

**Resumable 50-run sweep driver (`run_baselines_sweep.sh`) built as a verbatim mirror of the hardened ablation sweep, plus a Rule-1 fix that makes the WGAN-GP gradient-penalty double-backward train on Apple MPS (it was silently CPU-only).**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-17 (worktree agent-a3caaeb8f183fe572)
- **Completed:** 2026-05-17
- **Tasks:** 2 (Task 1 fully; Task 2 split-mode boundary — driver built + MPS validated, full sweep delegated)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- **`revision/run_baselines_sweep.sh`** — resumable 50-run driver: 5 models (`wgan_mlp wgan_cnn wgan_lstm vae ar`) × 2 pipelines (`A B`, no C per D-10-05) × 5 seeds (`42-46`), EPOCHS=1000, OUT_ROOT=`revision/results/baselines` (D-10-08/D-10-14). `.npz`-aware `is_complete()` (`checkpoint.npz` for `ar`, `checkpoint.pt` otherwise), atomic `update_status()` (`flock -x 9` + `tempfile.mkstemp` + `os.rename`) keyed on the `(model,pipeline,seed)` 3-tuple with `total_count=50`, `--parallel {1,2}` guardrail copied verbatim, zero functional `multiprocessing.Pool` (D-10-24/Pitfall 5). Dry-run confirms exactly **50** would-run triples.
- **MPS WGAN-GP path proven working** — a 2-epoch `wgan_mlp` smoke runs with critic inputs on `mps:0`, finite critic/generator losses, and the complete finite 5-file bundle. The fix was required because `train_wgan_gp` computed a `device` variable selecting MPS but never moved the model/tensors onto it (dead code → silent CPU-only training), and a naive `.to(mps)` additionally failed on the float64 Critic (MPS has no float64).
- **No regression** — `revision/tests/test_classical.py` and `test_nonadversarial.py` both print `OK`; the CPU path remains float64 and deterministic across identical seeds (verified run1==run2).

## Task Commits

1. **Task 1: run_baselines_sweep.sh** — `7beb9cb` (feat)
2. **Task 2 (split-mode): MPS WGAN-GP compatibility fix** — `7595389` (fix, Rule-1 deviation)

**Plan metadata:** committed with this SUMMARY (docs).

## Files Created/Modified

- `revision/run_baselines_sweep.sh` (created) — 50-run resumable sweep driver, verbatim mirror of `run_ablation_sweep.sh` with the documented Phase-10 adaptations.
- `revision/core/training.py` (modified) — device wiring + `compute_dtype` split so the WGAN-GP loop trains on MPS in float32 / CPU in float64.
- `revision/run_baselines.py` (modified) — `generate_wgan_samples` moves the trained generator back to CPU before sampling so `samples.npy` stays bit-identical float64 `*0.1` space.

## Decisions Made

- **SPLIT MODE** per explicit user directive: this agent builds the driver and proves the MPS path only; the orchestrator runs the full 50-run sweep in the background after merge. Sweep completion (50/50) and the uniform `data_hash` invariant are verified by the orchestrator before phase advance.
- **MPS, no CPU fallback for WGAN runs** (user directive): `compute_dtype = float32 if device.type=="mps" else float64`. Generator moves device-only (keeps native float32; output recast via `.to(compute_dtype)*0.1`); Critic moves device+dtype (it is `.double()` by construction). CPU/CUDA path keeps float64 untouched for exact prior reproduction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] WGAN-GP loop ran on CPU despite MPS being available; naive MPS move broke on float64 Critic**
- **Found during:** Task 2 (split-mode MPS smoke procedure)
- **Authorization:** Explicit Phase-10 split-mode user directive — "make MPS work no matter what. NO CPU fallback for the WGAN runs… you are explicitly authorized (Rule 1 deviation) to FIX the code until WGAN-GP trains correctly on MPS."
- **Issue:** `train_wgan_gp` (training.py:218-222) selected a `device` (MPS when available) but never moved the generator/critic/batch tensors onto it — the device variable was dead code, so all 3 new Phase-10 classical WGAN generators silently trained on CPU. Moving naively to MPS then raised `TypeError: Cannot convert a MPS Tensor to float64` because the shared `Critic` is constructed `.double()` (critic.py:67) and MPS has no float64. The GP path's `torch.autograd.grad(create_graph=True)` double-backward had never been exercised on MPS.
- **Fix:**
  - `training.py`: introduced `compute_dtype = float32 (MPS) / float64 (CPU,CUDA)`; `generator.to(device)` (device only — preserves native float32); `critic.to(device=device, dtype=compute_dtype)` (no-op cast on CPU, float32 cast on MPS); per-batch real tensor `.to(device=device, dtype=compute_dtype)`; noise tensors created with `device=device`; generator-output recasts changed `.to(torch.float64)*0.1` → `.to(compute_dtype)*0.1` (3 sites). `compute_gradient_penalty` already keys off `real_samples.device/dtype`, so the GP double-backward followed onto MPS automatically.
  - `run_baselines.py`: `generate_wgan_samples` moves the trained generator back to CPU before sampling so `samples.npy` is bit-identical to the pre-fix float64 `*0.1` space the shared `reconstruct_od` inverse consumes.
- **Files modified:** `revision/core/training.py`, `revision/run_baselines.py`
- **Verification:**
  - Instrumented 2-epoch `wgan_mlp` smoke: critic inputs observed on `mps:0`; `critic_loss_avg=[2.1444, 2.1468]`, `generator_loss_avg=[0.1087, 0.1137]` (all finite).
  - End-to-end CLI run on MPS emitted the full 5-file bundle (config.yaml/checkpoint.pt/samples.npy/metrics.json/inverse_kwargs.npz), all non-empty; samples `(3850,10)` all finite.
  - CPU path forced (mps unavailable): critic input stays `(cpu, float64)`, deterministic across identical seeds (run1==run2) — pre-fix behaviour preserved.
  - `revision/tests/test_classical.py` → `OK classical 74/73/78, single-param, autograd-live`; `revision/tests/test_nonadversarial.py` → `OK nonadversarial VAE + AR` (no regression).
- **Committed in:** `7595389` (separate atomic commit per split-mode directive)

**2. [Rule 1 - Verify-command bug] Plan's `! grep -qi 'multiprocessing'` is self-contradictory with its own verbatim-copy directive**
- **Found during:** Task 1 verification
- **Issue:** The plan's automated verify uses `! grep -qi 'multiprocessing'`, but the same task mandates copying `run_ablation_sweep.sh` verbatim including its "RESEARCH Pitfall — never multiprocessing.Pool" anti-pattern documentation comments (the reference template itself contains the word 4×). A faithful verbatim copy necessarily contains the word in warning prose, so the literal check can never pass for a correct copy.
- **Fix:** Applied the check that captures the true acceptance intent (D-10-24/Pitfall 5 = never *use* `multiprocessing.Pool`): strip comment lines, then assert no `multiprocessing` reference remains. Result: **zero functional multiprocessing**; all 5 string matches are anti-pattern warning comments copied verbatim as the plan directed. No code change needed — the script is correct; only the verify expression was buggy.
- **Files modified:** none (verification methodology only)
- **Verification:** `grep -v '^\s*#' run_baselines_sweep.sh | grep -i multiprocessing` → no match; `bash -n` clean; `--dry-run` enumerates exactly 50 would-run triples; `xargs -P`, `flock`, `checkpoint.npz`, MODELS/SEEDS/EPOCHS constants all present; PIPELINES has no C; file executable.
- **Committed in:** `7beb9cb` (Task 1 commit)

---

**Total deviations:** 2 (1 Rule-1 code bug — the load-bearing MPS fix authorized by the split-mode directive; 1 Rule-1 verify-command bug — methodology correction, no code change).
**Impact on plan:** No scope creep. The MPS fix is exactly the Task-2 split-mode deliverable. The verify-command correction does not alter the delivered script, which faithfully mirrors the hardened template.

## Issues Encountered

- `qgan_env` is git-ignored and therefore absent from the worktree; resolved by invoking the main-repo venv binary (`/Users/shawngibford/dev/phd/qGAN/qgan_env/bin/python`) with `PYTHONPATH=.` from the worktree root so `revision/` imports resolve to the worktree copy. This is an execution-environment detail only; the canonical invocation in the script (`./qgan_env/bin/python`) is unchanged and correct for the merged main repo where the orchestrator runs the full sweep.

## Split-Mode Task 2 Status

- Sweep driver built & MPS WGAN-GP path validated via a 2-epoch smoke (device==mps, finite losses, 5-file bundle).
- **Full 50-run sweep delegated to orchestrator background execution per the user split-mode directive.**
- 50/50 completion + uniform `data_hash` invariant to be verified by the orchestrator before phase advance.
- Smoke scratch dir `revision/results/_mps_smoke` removed; no smoke artifacts committed.

## Next Phase Readiness

- `revision/run_baselines_sweep.sh` is ready for the orchestrator to launch (`./revision/run_baselines_sweep.sh --parallel 2`); it is resumable, so partial progress survives interruption.
- WGAN-GP now genuinely uses Apple MPS for all 3 classical WGAN generators across the sweep; non-WGAN paths (VAE local ELBO loop, AR closed-form lstsq) are unaffected by this change.
- Wave-4 aggregation can proceed once the orchestrator confirms 50/50 + uniform `data_hash`.

## Self-Check: PASSED

- Files: `revision/run_baselines_sweep.sh`, `revision/core/training.py`, `revision/run_baselines.py`, `10-03-SUMMARY.md` all present.
- Commits: `7beb9cb` (Task 1 feat), `7595389` (Task 2 MPS fix) both in git log.
- Smoke scratch dir `revision/results/_mps_smoke` removed; full 50-run sweep NOT run (delegated to orchestrator per split mode).
- STATE.md / ROADMAP.md untouched (orchestrator owns those writes).

---
*Phase: 10-classical-baselines*
*Completed: 2026-05-17*
