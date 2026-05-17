---
phase: 10-classical-baselines
plan: 01
subsystem: revision/core/models
tags: [baselines, wgan-gp, vae, autoregressive, model-definitions]
requires:
  - revision/core/models/quantum.py (interface contract analog)
  - revision/core/training.py (train_wgan_gp params_pqc contract)
  - revision/results/transform_ablation/runs/{A,B}/{42..46} (Phase 09.1 quantum reference column)
provides:
  - WGANMLPGenerator (74 params), WGANCNNGenerator (73), WGANLSTMGenerator (78)
  - VAEBaseline (562 params, ELBO-ready interface)
  - ARBaseline (p=2, 3 params, lstsq fit + recursive sample)
  - barrel revision.core.models exposing classical, nonadversarial
affects:
  - Wave 2 (run_baselines.py) — consumes these model defs unchanged
tech-stack:
  added: []
  patterns:
    - single-flat-nn.Parameter functional generator (mirrors quantum.py params_pqc)
    - model-definitions-only in core/ (D-10-13); loops/orchestration in run_baselines.py
key-files:
  created:
    - revision/core/models/classical.py
    - revision/core/models/nonadversarial.py
    - revision/tests/test_classical.py
    - revision/tests/test_nonadversarial.py
    - revision/tests/__init__.py
  modified:
    - revision/core/models/__init__.py
decisions:
  - "Used main-repo qgan_env python via absolute path (env is gitignored, not in worktree)"
  - "AR sample burn-in 50 steps from zeros for stationarity before saved window"
metrics:
  duration: ~13 min
  completed: 2026-05-17
  tasks: 3
  files: 6
---

# Phase 10 Plan 01: Classical Baseline Model Definitions Summary

Implemented the 5 model definitions Phase 10 needs — 3 matched-parameter classical WGAN-GP generators (single-flat-`params_pqc` functional design, counts 74/73/78) plus VAEBaseline (562 params) and ARBaseline (p=2, 3 params) — and verified all 10 Phase 09.1 quantum reference run dirs exist (phase NOT blocked).

## What Was Built

**Task 1 — `revision/core/models/classical.py` + barrel** (TDD: RED `9bf6abe` → GREEN `fd5a786`)
- `WGANMLPGenerator` (74 params): `Linear(5,4)+b → Tanh → Linear(4,10)+b`
- `WGANCNNGenerator` (73 params): `ConvTranspose1d(1,9,k=6,s=1) → LeakyReLU → Conv1d(9,1,1)`
- `WGANLSTMGenerator` (78 params): functional `LSTM(I=2,H=2,1L,bias) → Linear(2,10)+b`
- Each holds ALL trainable weights as a single live `nn.Parameter` named `params_pqc`; `forward` slices that flat vector into per-layer weight/bias views applied via `torch.nn.functional` (mirrors quantum.py "one parameter vector, functional circuit" design — RESEARCH Pitfall 1).
- `count_params() == params_pqc.numel() == sum(p.numel() for p in parameters())`, all in [71,79], within ±5% of quantum's 75.
- Pitfall-1 negative test passes: one Adam step on `[params_pqc]` mutates it (autograd live, not detached).
- `__init__.py` barrel updated to expose `classical, nonadversarial`.

**Task 2 — `revision/core/models/nonadversarial.py`** (TDD: RED `7178d00` → GREEN `a3d0bb3`)
- `VAEBaseline(nn.Module)`: `Linear(10,16)→ReLU→[mu,logvar](16,4)` encoder, `Linear(4,16)→ReLU→Linear(16,10)` decoder; `encode/reparameterize/decode/forward/sample`; 562 params (reported transparently, NOT param-matched per D-10-03).
- `ARBaseline` (plain class, not nn.Module): AR(2), `count_params()==3`, closed-form `np.linalg.lstsq` fit, recursive `sample` with 50-step burn-in.
- Both samplers emit in `[-1,1]` window space and deliberately do NOT apply `*0.1` (quantum-output artifact — RESEARCH Pitfall 3); documented in module + class docstrings.
- D-10-13 honored: no ELBO loop / optimizer / `.backward()` in this file (guarded by `test_no_training_loop_in_module`).

**Task 3 — Phase 09.1 quantum reference precondition gate** (no code change — nominal `<files>` entry)
- Verified all 10 dirs `revision/results/transform_ablation/runs/{A,B}/{42,43,44,45,46}/` contain `config.yaml`, `samples.npy`, `inverse_kwargs.npz`. Command exited 0: `OK all 10 Phase 09.1 quantum run dirs present`. **Phase is NOT blocked** — the Wave-4 comparison-table reference column is available.

## Verification Results

- `from revision.core.models import classical, nonadversarial` — imports clean
- count_params empirical: MLP=74, CNN=73, LSTM=78 (all [71,79], ±5% of 75); VAE=562 (in [540,580]); AR=3
- Each generator: exactly 1 `nn.Parameter`, `forward((5,B))→(B,10)`, autograd-live Pitfall-1 negative test passes
- VAE: forward returns `(x_hat[B,10], mu[B,4], logvar[B,4])`; sample `(n,10)`
- AR: `phi.shape==(2,)`, `sigma2>0`, `p==2`, sample `(n,10)`, distinct seeds → distinct samples
- All 10 Phase 09.1 quantum run dirs present
- Test files `revision/tests/test_classical.py`, `test_nonadversarial.py` both pass

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] qgan_env not present in worktree**
- **Found during:** Task 3 precondition / all verification steps
- **Issue:** The plan's verify commands hardcode `./qgan_env/bin/python`, but `qgan_env/` is gitignored and lives only in the main repo, not in the parallel-executor worktree.
- **Fix:** Used the absolute path `/Users/shawngibford/dev/phd/qGAN/qgan_env/bin/python` with `PYTHONPATH` set to the worktree root for all verification. No environment changes, no package installs (T-10-SC respected — zero new packages).
- **Files modified:** none (verification-harness invocation only)

**2. [Rule 1 - Bug] AR recursive-sample lag-slice wrapped at index 0**
- **Found during:** Task 2 GREEN
- **Issue:** `buf[t-1 : t-1-p : -1]` produced an empty slice when `t-1-p` went negative (Python negative-stop wraps to end of array), raising a numpy matmul dimension-mismatch in `ARBaseline.sample`.
- **Fix:** Replaced with `buf[t-p : t][::-1]` (ascending lag-p..lag-1 window, reversed to lag-1..lag-p so it dots correctly with `phi`). Added an explanatory comment.
- **Files modified:** `revision/core/models/nonadversarial.py`
- **Commit:** `a3d0bb3` (fixed before the GREEN commit; RED test `7178d00` covered this path)

## TDD Gate Compliance

Both behavior-adding tasks followed RED → GREEN:
- Task 1: `test(10-01)` `9bf6abe` (RED, fails: module absent) → `feat(10-01)` `fd5a786` (GREEN)
- Task 2: `test(10-01)` `7178d00` (RED, fails: module absent) → `feat(10-01)` `a3d0bb3` (GREEN)
RED tests failed for the correct reason (feature genuinely absent — `ModuleNotFoundError` / circular import on missing module), not a spurious pass. No REFACTOR commits needed (code clean at GREEN).

## Threat Model Compliance

- T-10-01 (params_pqc detaching autograd): mitigated — empirical Pitfall-1 negative test (one Adam step mutates `params_pqc`) passes for all 3 generators.
- T-10-SC (package installs): mitigated — zero new packages installed; reused pre-verified main-repo `qgan_env`.
- No new security surface introduced (pure model-definition code, reads only project-authored Phase 09.1 artifacts read-only).

## Known Stubs

None. All model definitions are fully implemented and verified. (ELBO loop / AR fit orchestration / checkpoint serialization are intentionally NOT here — D-10-13 places them in `run_baselines.py`, Wave 2; this is by design, not a stub.)

## Self-Check: PASSED

All 6 created/modified files exist on disk; all 4 task commits (`9bf6abe`, `fd5a786`, `7178d00`, `a3d0bb3`) present in git history.
