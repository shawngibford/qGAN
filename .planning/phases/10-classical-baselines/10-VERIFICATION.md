---
phase: 10-classical-baselines
verified: 2026-05-17T22:00:00Z
status: passed
score: 14/14
overrides_applied: 0
re_verification: false
---

# Phase 10: Classical Baselines Verification Report

**Phase Goal:** Matched-parameter classical WGAN-GP and a non-adversarial baseline (VAE or AR) are trained under identical conditions to the quantum generator, so the manuscript can report a fair quantum-vs-classical comparison in response to R1-M1 and R2-1.
**Verified:** 2026-05-17T22:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Three classical WGAN-GP generators (MLP/CNN/LSTM) expose matched parameter counts within ±5% of quantum (75) | VERIFIED | Empirical: `WGANMLPGenerator.count_params()==74`, `WGANCNNGenerator.count_params()==73`, `WGANLSTMGenerator.count_params()==78`; all satisfy the single-`params_pqc` `nn.Parameter` contract and `forward((5,B))->(B,10)`; autograd live confirmed (Pitfall-1 negative test passes for all 3) |
| 2 | Non-adversarial baselines VAE and AR are implemented and trainable | VERIFIED | Empirical: `VAEBaseline.count_params()==562`; `encode/decode/reparameterize/forward/sample` all return correct shapes; `ARBaseline.fit` sets `phi.shape==(2,)`, `sigma2>0`, `count_params()==3`; `sample` returns `(n,10)` |
| 3 | All 5 models train via `run_baselines.py` under identical data/seed/epoch conditions | VERIFIED | `run_baselines.py` imports `train_wgan_gp` unchanged; WGAN branch calls it with the shared `Critic` and imported HPO constants; VAE/AR use the same `build_dataset_for_pipeline` (A/B only, verbatim copy from `run_ablation.py`); 50 sweep `config.yaml` files all carry `epochs=1000` and `data_hash=91e447d4624e25b3` (uniform across all 50) |
| 4 | The full 50-run sweep (5 models × 2 pipelines × 5 seeds) completes at 1000 epochs each | VERIFIED | `sweep_status.json`: 50/50 complete, 0 failures, wall 23m29s; on-disk: all 50 `runs/<model>/<p>/<s>/` dirs contain 5-file bundles (config.yaml, checkpoint.pt/.npz, samples.npy, metrics.json, inverse_kwargs.npz) — confirmed programmatically |
| 5 | `baseline_comparison.json` has long-form `rows[]` + `models[]` covering quantum + 5 new models × A/B × 5 seeds | VERIFIED | 1710 rows; `models[]` = `{quantum,wgan_mlp,wgan_cnn,wgan_lstm,vae,ar}`; param counts `{75,74,73,78,562,3}`; all 12 `(model_kind,pipeline)` combos present; schema fields `{model_kind,pipeline,seed,metric_name,scale,value}` valid on all rows |
| 6 | `baseline_comparison.md` renders one row per model with required columns | VERIFIED | Table present with columns: model, parameter_count, OD-EMD (mean±std), OD-ACF lag-1, OD-DTW mean, transformed-EMD (Pipeline B), TSTR-lite R²; all 6 model kinds including quantum reference; D-10-19 compliance: no Phase-10 recommendation ("Phase 14 owns the headline baseline decision") |
| 7 | TSTR-lite is reported per model × pipeline with init seeds {40,41,42}, HELD_OUT_N=320 | VERIFIED | TSTR block in `baseline_comparison.json` has 12 required `model|pipeline` keys (plus `real_only_baseline`); each entry has `mse_mean/std`, `r2_mean/std`, `per_init_seed` with keys `{40,41,42}`; `n_eval_real=320` |
| 8 | `baseline_classical_wgan.json` (BASE-01) models == {wgan_mlp,wgan_cnn,wgan_lstm} | VERIFIED | File present (157KB), `models[]` = `{'wgan_mlp','wgan_cnn','wgan_lstm'}` |
| 9 | `baseline_nonadversarial.json` (BASE-02) models == {vae,ar} with `train_protocol_notes` documenting `*0.1` asymmetry | VERIFIED | File present (102KB), `models[]` = `{'vae','ar'}`; both entries have `train_protocol_notes` explicitly documenting no `*0.1` and citing "RESEARCH Pitfall 3" |
| 10 | Metrics computed via `revision.core.eval` only — no new helpers added (D-10-20) | VERIFIED | `eval.py` defines 7 functions, same as before this phase (`compute_emd`, `compute_moments`, `compute_acf`, `compute_dtw`, `compute_jsd`, `compute_psd`, `full_metric_suite`); `baseline_comparison.json` explicitly records `"metric_helpers": "revision.core.eval ONLY (D-10-20)"` |
| 11 | Data-hash invariant: uniform `91e447d4624e25b3` across all 50 new configs; quantum equivalence by construction (D-10-15) | VERIFIED | `baseline_comparison.json` `data_hash_verification` records `all_equal=true`, `n_new_configs_checked=50`, hash `91e447d4624e25b3`; quantum by-construction argument documented; no 09.1 grep (Pitfall 4) |
| 12 | Phase 09.1 quantum reference column available (10 dirs × 3 core artifacts each) | VERIFIED | Programmatic check: `OK all 10 Phase 09.1 quantum run dirs present (A,B x 42-46)` — all 30 artifacts present |
| 13 | CR-01 (spectral hook) and CR-02 (EarlyStopping restore) code-review defects do not affect any of the 50 sweep runs | VERIFIED | `train_wgan_gp` defaults: `spectral_loss_weight=0.0`, `early_stopper=None`; `run_baselines.py` never passes either argument; programmatic check of all 50 `config.yaml` files confirms `spectral_loss_weight` absent/zero in every run — both defects are dormant paths that were never exercised |
| 14 | Sweep driver (`run_baselines_sweep.sh`) is `.npz`-aware, uses `xargs -P` (no `multiprocessing.Pool`), and writes `sweep_status.json` atomically with `flock` | VERIFIED | Grep confirms: `xargs -P` present, `multiprocessing` absent, `checkpoint.npz` conditional in `is_complete()`, `flock` in status writer; MODELS/SEEDS/EPOCHS constants correct (5 models, seeds 42-46, EPOCHS=1000, PIPELINES A/B only) |

**Score:** 14/14 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `revision/core/models/classical.py` | WGANMLPGenerator(74), WGANCNNGenerator(73), WGANLSTMGenerator(78) with `params_pqc` + `count_params` | VERIFIED | 8857 bytes; empirical counts 74/73/78 confirmed; single `nn.Parameter`, autograd live |
| `revision/core/models/nonadversarial.py` | VAEBaseline (~562 params) + ARBaseline (p=2, 3 params) | VERIFIED | 7773 bytes; VAE=562 params, AR count_params=3; both ELBO-ready, no training loop in file |
| `revision/core/models/__init__.py` | Barrel exposing `classical`, `nonadversarial` | VERIFIED | `from revision.core.models import quantum, critic, classical, nonadversarial`; `__all__` matches |
| `revision/run_baselines.py` | Per-(model,pipeline,seed) CLI driver with WGAN/VAE/AR branches + 5-file bundle + data_hash | VERIFIED | 19629 bytes; `train_wgan_gp` imported and called verbatim; HPO constants from `revision.core`; argparse choices A/B only |
| `revision/run_baselines_sweep.sh` | Resumable 50-run sweep, `.npz`-aware, `xargs -P`, atomic `flock` status | VERIFIED | 17444 bytes; executable; syntax-clean; all required properties confirmed |
| `revision/06_baseline_comparison.ipynb` | Aggregation notebook executed end-to-end | VERIFIED | 55189 bytes; 20-cell notebook; emits all 4 result artifacts |
| `revision/results/baseline_comparison.json` | BASE-03: long-form rows[] + models[] + tstr, 6 models × A/B | VERIFIED | 311841 bytes; 1710 rows; 6 models; 12 TSTR entries; data_hash verified |
| `revision/results/baseline_comparison.md` | BASE-03: markdown table, one row per model per pipeline | VERIFIED | 2214 bytes; all 6 model rows per pipeline; required columns present; no Phase-10 recommendation |
| `revision/results/baseline_classical_wgan.json` | BASE-01: {wgan_mlp,wgan_cnn,wgan_lstm} subset | VERIFIED | 157272 bytes; correct model set |
| `revision/results/baseline_nonadversarial.json` | BASE-02: {vae,ar} subset with train_protocol_notes | VERIFIED | 101767 bytes; correct model set; protocol notes present |
| `revision/results/baselines/sweep_status.json` | 50/50 runs complete, 0 failures | VERIFIED | 50 run records, all `status=complete`, all `return_code=0`, epochs=1000 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `revision/core/models/classical.py` | `revision/core/training.py::train_wgan_gp` | `params_pqc` single `nn.Parameter` + `forward((5,B))->(B,10)` | VERIFIED | Interface contract satisfied empirically; `run_baselines.py` passes generator directly to `train_wgan_gp` |
| `revision/run_baselines.py` | `revision/core/training.py::train_wgan_gp` | WGAN branch calls `train_wgan_gp(gen, Critic(), loader, ...HPO consts...)` | VERIFIED | `grep -n "train_wgan_gp"` shows import at line 90 and call at line 249 with verbatim HPO constants |
| `revision/run_baselines.py` | `revision/core/preprocessing.py` | `build_dataset_for_pipeline` A/B + `inverse_kwargs.npz` | VERIFIED | `inverse_kwargs` referenced in `build_dataset_for_pipeline` (lines 135/152/171); `_save_inverse_kwargs` writes `inverse_kwargs.npz`; all 50 on-disk dirs confirmed to have this file |
| `revision/run_baselines_sweep.sh` | `revision/run_baselines.py` | `run_one()` invokes `python -m revision.run_baselines --model --pipeline --seed --epochs` | VERIFIED | `grep` confirms `revision.run_baselines` invocation in `run_one()` |
| `revision/06_baseline_comparison.ipynb` | `revision/results/baselines/runs + transform_ablation/runs` | `reconstruct_od` + `eval.py` over all 60 run dirs | VERIFIED | `baseline_comparison.json` rows include quantum (from `transform_ablation/runs`) and all 5 new models; 1710 rows total covers all combinations |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `baseline_comparison.json` rows[] | `rows` list | 50 on-disk `samples.npy` + 10 reused quantum `samples.npy`; metrics via `revision.core.eval` | Yes — 1710 rows from real computed fidelity metrics | FLOWING |
| `baseline_comparison.json` tstr block | TSTR R² per (model,pipeline) | Reconstructed OD windows from `samples.npy` fed through verbatim `TSTRLiteLSTM/train_eval_tstr`; eval on 320 real held-out windows | Yes — init seeds {40,41,42} produce per_init_seed results | FLOWING |
| `baseline_comparison.md` | Aggregated mean±std per model per pipeline | Derived from `baseline_comparison.json` rows by the notebook | Yes — populated from real 1710-row table | FLOWING |
| `baseline_classical_wgan.json` | Filtered projection of `baseline_comparison.json` | rows[]/models[] filtered to {wgan_mlp,wgan_cnn,wgan_lstm} | Yes — same real data, no recomputation | FLOWING |
| `baseline_nonadversarial.json` | Filtered projection of `baseline_comparison.json` | rows[]/models[] filtered to {vae,ar} | Yes — same real data | FLOWING |

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| BASE-01 | Plans 01, 02, 03, 04 | Classical WGAN-GP matched ±5% PQC params, same critic/optimizer/schedule/seeds | SATISFIED | WGANMLPGenerator(74)/WGANCNNGenerator(73)/WGANLSTMGenerator(78) — all within ±5% of 75; 50 runs at EPOCHS=1000; `baseline_classical_wgan.json` emitted |
| BASE-02 | Plans 01, 02, 03, 04 | Non-adversarial baseline (VAE or AR) on same data, same evaluation metrics | SATISFIED | VAEBaseline(562) via ELBO loop + ARBaseline(p=2) via lstsq; same `build_dataset_for_pipeline` A/B; same `revision.core.eval` metrics; `baseline_nonadversarial.json` emitted |
| BASE-03 | Plan 04 | Parameter-count / expressibility-controlled comparison table as JSON + markdown | SATISFIED | `baseline_comparison.{json,md}` present: quantum + 3 classical WGAN-GP + VAE + AR, pipelines A/B, 5 seeds, full fidelity suite + TSTR-lite |

**No orphaned requirements:** BASE-01/02/03 are the only Phase-10 requirements in `REQUIREMENTS.md` traceability table. All three are satisfied.

---

### Code Review Findings Assessment (from 10-REVIEW.md)

| Finding | Severity per Review | Impact on Phase-10 Goal | Assessment |
|---------|--------------------|-----------------------|------------|
| CR-01: Spectral hook non-differentiable / device-unsafe | Critical | None — `spectral_loss_weight=0.0` default; `run_baselines.py` never passes this arg; confirmed zero in all 50 `config.yaml` files | WARNING: Dormant path, does not affect BASE-01/02/03 results. Carry to Phase 13 fix. |
| CR-02: EarlyStopping restore device/dtype inconsistency | Critical | None — `early_stopper=None` default; `run_baselines.py` never passes this arg; no EarlyStopping in any sweep run | WARNING: Dormant path, does not affect Phase-10 sweep results. |
| WR-01 through WR-07 | Warning | None affect correctness of the 50 committed sweep runs or their reported metrics | INFO for future phases |
| IN-01 through IN-05 | Info | None affect correctness of BASE-01/02/03 deliverables | INFO |

**Ruling on CR-01 / CR-02:** Both defects reside in code paths gated by opt-in arguments (`spectral_loss_weight > 0.0` and `early_stopper is not None`). Neither argument was passed in any of the 50 sweep runs. The sweep ran all WGAN training on the CPU (macOS MPS path; no early stopping). The BASE-01/02/03 result artifacts are unaffected. These defects should be tracked and fixed before any run that activates either opt-in path, but they do not block Phase-10 goal achievement.

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `revision/core/training.py:390-392` | `acf_avg/vol_avg/lev_avg` emitted as `0.0` placeholder | INFO (IN-05 from review) | Downstream readers of per-epoch metrics see flat zeros; does not affect final deliverables which use `revision.core.eval` on final samples |
| `revision/core/training.py:510-514` | `_NOISE_HIGH_LITERAL = 4 * math.pi` sentinel for grep verification | INFO (IN-03 from review) | Dead constant; no functional impact |

No `TBD`, `FIXME`, or `XXX` markers found in any Phase-10 modified files.

---

### Behavioral Spot-Checks

| Behavior | Check | Result | Status |
|----------|-------|--------|--------|
| Classical generators emit correct param counts | `WGANMLPGenerator().count_params()==74`, `WGANCNNGenerator().count_params()==73`, `WGANLSTMGenerator().count_params()==78` | Confirmed 74/73/78 | PASS |
| Single `nn.Parameter` interface + correct output shape | `sum(1 for _ in g.parameters())==1` and `g(randn(5,12)).shape==(12,10)` for all 3 | All pass | PASS |
| Autograd live (Pitfall-1 negative test) | One Adam step on `[g.params_pqc]` changes `params_pqc` | All 3 generators: `autograd live = True` | PASS |
| VAE interface | `forward(randn(7,10))` returns `(xh[7,10], mu[7,4], lv[7,4])`; `sample(5,...).shape==(5,10)` | Confirmed | PASS |
| AR interface | `fit(x_1d)` sets `phi.shape==(2,)`, `sigma2>0`; `sample(6,...).shape==(6,10)` | Confirmed | PASS |
| 50/50 sweep completeness | All 50 on-disk run dirs have 5-file bundles; `sweep_status.json` reports 50 complete, 0 failed | 50/50 confirmed | PASS |
| Comparison artifacts well-formed | `baseline_comparison.json` has 6 models × 2 pipelines × 12 TSTR entries; param counts correct; `data_hash` uniform | All assertions pass | PASS |
| Phase 09.1 quantum reference dirs | All 10 dirs with `config.yaml`, `samples.npy`, `inverse_kwargs.npz` | Exit 0 confirmed | PASS |

---

### Human Verification Required

None. All phase-10 success criteria are programmatically verifiable from on-disk artifacts. The VAE posterior-collapse observation (noted in 10-04-SUMMARY and visible in Pipeline A `train_protocol_notes`) is documented in the artifact and deliberately deferred to Phase 14 — it does not require human verification here.

---

## Gaps Summary

No gaps. All 14 must-haves verified. Phase goal achieved: matched-parameter classical WGAN-GP (MLP/CNN/LSTM, 74/73/78 params) and non-adversarial baselines (VAE 562 params, AR 3 params) are trained at 1000 epochs under identical data/seed conditions (data_hash=91e447d4624e25b3 uniform across all 50 runs), and the apples-to-apples comparison artifacts for R1-M1/R2-1 are committed and well-formed.

The two Critical code-review findings (CR-01, CR-02) are defects in dormant opt-in paths that zero of the 50 sweep runs activated. They are real defects that should be fixed before any run exercises `spectral_loss_weight > 0` or `EarlyStopping`, but they do not constitute gaps against Phase 10's goal.

---

_Verified: 2026-05-17T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
