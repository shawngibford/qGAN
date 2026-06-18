---
phase: 10-classical-baselines
plan: 02
subsystem: revision
tags: [baselines, cli-driver, wgan-gp, vae, autoregressive, artifact-bundle]
requires:
  - run_ablation.py (verbatim A/B + bundle template)
  - core/training.py::train_wgan_gp (WGAN branch, UNCHANGED — D-10-08)
  - core/models/classical.py (WGAN{MLP,CNN,LSTM}Generator — plan 01)
  - core/models/nonadversarial.py (VAEBaseline, ARBaseline — plan 01)
  - core/models/critic.py (shared Critic — D-10-08)
  - core/preprocessing.py (forward_logreturns/minmax + inverse_logreturns)
provides:
  - run_baselines.py (per-(model,pipeline,seed) CLI driver, 5-file bundle + data_hash)
  - cross-family sample-space comparability gate (RESEARCH Pitfall 3) — GREEN
affects:
  - Wave 3 (50-run sweep) — invokes this driver per (model,pipeline,seed)
  - Wave 4 (comparison notebook) — consumes the 5-file bundle + data_hash
tech-stack:
  added: []
  patterns:
    - one-process-per-invocation idempotent run-dir (mirrors run_ablation)
    - model-family dispatch (WGAN via shared loop / VAE local ELBO / AR lstsq)
key-files:
  created:
    - run_baselines.py
  modified: []
decisions:
  - "Task 2 plan verify snippet (inverse_logreturns(s, **ik)) is incompatible with the actual inverse_logreturns(r_norm, od_start, mu, sigma) signature; implemented the gate via the notebook reconstruct_od B-branch logic (the identical Wave-4 inverse path the plan mandates) instead [Rule 1]"
  - "Used absolute main-repo qgan_env python (env gitignored, absent from worktree) — same Rule-3 deviation as plan 01"
  - "WGAN checkpoint stores gen_state_dict (plan text) — equivalent to params_pqc tensor since params_pqc is the sole nn.Parameter"
metrics:
  duration: ~9 min
  completed: 2026-05-17
  tasks: 2
  files: 1
---

# Phase 10 Plan 02: run_baselines.py CLI Driver Summary

Built `run_baselines.py` — the idempotent per-(model,pipeline,seed) Phase 10 driver with three model-family branches (WGAN-GP via the unchanged shared `train_wgan_gp` loop + shared `Critic`, VAE via a local ELBO loop, AR via closed-form lstsq), emitting the same 5-file artifact bundle as Phase 09.1 plus a NEW `data_hash` field — and verified the cross-family sample-space comparability gate (RESEARCH Pitfall 3, the top BASE-03 risk control) is GREEN before any sweep.

## What Was Built

**Task 1 — `run_baselines.py`** (commit `33b8f71`)
- `argparse`: `--model {wgan_mlp,wgan_cnn,wgan_lstm,vae,ar}`, `--pipeline {A,B}` (C dropped, D-10-05), `--seed int`, `--epochs int`, `--out-root` default `results/baselines`, `--csv-path` default `./data.csv`.
- `build_dataset_for_pipeline`: A and B branches copied VERBATIM from `run_ablation.py` (D-10-07 — identical windowed data + `inverse_kwargs` contract + identical `DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)`); C branch deleted (D-10-05). `_save_inverse_kwargs` copied verbatim.
- HPO constants imported from `revision.core` (BATCH_SIZE, EVAL_EVERY, LAMBDA, LR_CRITIC, LR_GENERATOR, N_CRITIC, NOISE_HIGH, NOISE_LOW, NUM_LAYERS, NUM_QUBITS, WINDOW_LENGTH) — never hardcoded (D-10-08).
- **WGAN branch** (wgan_mlp/wgan_cnn/wgan_lstm): `torch.manual_seed(seed)` → matching generator → `Critic(window_length=WINDOW_LENGTH)` → `train_wgan_gp(...)` called UNCHANGED with the HPO constants (D-10-08) → samples via `generator(noise=(NUM_QUBITS,bs)).to(float64) * 0.1` (the `*0.1` is mandatory WGAN/quantum-output scaling) → `checkpoint.pt = {gen_state_dict, critic_state_dict}`, `metrics.json` = per-epoch dict from `train_wgan_gp`.
- **VAE branch**: local ELBO loop — single `Adam(vae.parameters(), lr=1e-3)`, NO critic / NO n_critic / NO gradient penalty; per-epoch `recon=MSE; kld=-0.5*mean(1+logvar-mu^2-exp(logvar)); loss=recon+beta*kld` with `beta=1.0` and a posterior-collapse heuristic (sample std vs real-window std flagged in metrics + notes; warmup hook present, `warmup_epochs=0` by default since Wave-2 smoke showed no collapse). Sampling via `vae.sample(...)` with NO `*0.1` (Pitfall 3). `checkpoint.pt = {vae_state_dict}`, `metrics.json` = per-epoch elbo/recon/kld + collapse diagnostics.
- **AR branch**: `ARBaseline(p=2)`, `fit()` on the flattened windowed series via lstsq, recursive `sample()` with NO `*0.1` (Pitfall 3). `checkpoint.npz = {phi, sigma2, p}` (D-10-14), `metrics.json` = sigma2 / phi / residual diagnostics.
- 5-file bundle for every branch: `config.yaml`, `checkpoint.pt|.npz`, `samples.npy` (shape `(N_synth,10)`, `N_synth = 10 * n_real_windows`), `metrics.json`, `inverse_kwargs.npz`. `config.yaml` extends the run_ablation base with `model_kind`, `data_hash` (= `sha256(load_and_preprocess(csv)["OD"].numpy().tobytes())[:16]`, D-10-15), `parameter_count` (from `count_params()`), `family` (`adversarial-classical` / `non-adversarial`), `train_protocol_notes` (explicitly documents the `*0.1` asymmetry), and WGAN-only fields set to `null` for VAE/AR.
- Idempotent run-dir overwrite (`shutil.rmtree` before write — T-10-04, no stale partial bundle). No `multiprocessing` import/Pool (Pitfall 5; the only `multiprocessing` token is the prohibition note in the module docstring). No new `revision.core.eval` helpers (D-10-20/24).

**Task 2 — Cross-family sample-space reconstruction smoke gate** (verification gate, no code change — gate passed first try)
- Ran 3-epoch Pipeline-B seed-42 smoke runs for `wgan_cnn`, `vae`, `ar`; reconstructed one sample from each through the IDENTICAL Wave-4 inverse path (the `reconstruct_od` B-branch logic from `_build_analysis_notebook.py:112-127`, consuming each run's own `inverse_kwargs.npz`).
- All three reconstructed OD windows land within `[real_OD.min()*0.5, real_OD.max()*1.5]` = `[0.2350, 5.7000]`. Gate GREEN: WGAN (`*0.1`), VAE (no `*0.1`), AR (no `*0.1`) all reconstruct into the real OD range — the Wave-4 comparison table will aggregate comparable numbers. No branch sample-space fix was needed.

## Verification Results

- Task 1 automated check: `python -m revision.run_baselines --model wgan_mlp --pipeline B --seed 42 --epochs 3` exits 0; all 5 bundle files non-empty; `config.yaml` has `data_hash` (16 hex), `model_kind=='wgan_mlp'`, `parameter_count==74`, `family=='adversarial-classical'`; `samples.npy` shape `(3840,10)` (`N_synth==10*384`).
- argparse rejects `--pipeline C` (`invalid choice: 'C' (choose from 'A', 'B')`); accepts all 5 `--model` kinds.
- No `multiprocessing.Pool` call/import; no new `eval.py` helpers.
- Task 2 cross-family gate: `OK ... wgan_cnn/vae/ar reconstruct into real OD range allowed=[0.2350,5.7000]`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 2 plan verify snippet uses an incompatible inverse signature**
- **Found during:** Task 2
- **Issue:** The plan's literal `<automated>` snippet calls `inverse_logreturns(s[:1], **{k:ik[k] for k in ik.files})`. The B-pipeline `inverse_kwargs.npz` keys are `{r_min,r_max,mu,sigma,od_starts}`, but `inverse_logreturns(r_norm, od_start, mu, sigma)` accepts none of `r_min/r_max/od_starts` — the snippet would raise `TypeError` and never exercise the real inverse.
- **Fix:** Implemented the gate using the canonical notebook `reconstruct_od` B-branch logic (`_build_analysis_notebook.py:112-127`) — un-scale `[-1,1]→r_norm` via `r_min/r_max`, draw `od_start` from `od_starts` with the notebook's `rng=default_rng(seed*7919+1)`, then `inverse_logreturns(r_norm, od_start, mu, sigma)`. This IS the "identical inverse path the Wave-4 notebook uses" the plan's `<action>` mandates; the literal snippet was an internally inconsistent transcription.
- **Files modified:** none (verification harness only — gate passed, no driver fix required)

**2. [Rule 3 - Blocking] qgan_env absent from worktree**
- **Found during:** all verification steps
- **Issue:** Plan verify commands use `./qgan_env/bin/python`, but `qgan_env/` is gitignored and lives only in the main repo, not the parallel-executor worktree.
- **Fix:** Used absolute path `/Users/shawngibford/dev/phd/qGAN/qgan_env/bin/python` with `PYTHONPATH=.`. No environment changes, no package installs (T-10-SC respected — zero new packages).
- **Files modified:** none

### Minor implementation notes (not deviations)

- WGAN `checkpoint.pt` stores `{gen_state_dict, critic_state_dict}` exactly as the plan `<action>` specifies; since each classical generator's only `nn.Parameter` is `params_pqc`, `gen_state_dict` is the params_pqc tensor under a state-dict key (equivalent to run_ablation's `params_pqc` entry).

## Threat Model Compliance

- T-10-03 (path traversal via CLI args): mitigated — `--model`/`--pipeline` are `argparse choices=` (5 + 2 fixed tokens), `--seed` is `type=int`; run-dir built only from these constrained tokens (mirrors run_ablation's hardened pattern).
- T-10-04 (partial bundle): mitigated — `shutil.rmtree(run_dir)` before every write makes runs idempotent; no stale "complete-looking" dir.
- T-10-05 (untrusted pickle): accept — checkpoints are produced by this same codebase.
- T-10-SC (package installs): mitigated — zero new packages.

## Known Stubs

None. All three model-family branches are fully implemented and produce verified non-empty 5-file bundles. The VAE KL-warmup is a no-op-by-default hook (`warmup_epochs=0`) — intentional per the plan (only activated if posterior collapse is observed; Wave-2 smoke showed none), not a stub.

## Self-Check: PASSED

`run_baselines.py` exists on disk; commit `33b8f71` present in `git log`.
