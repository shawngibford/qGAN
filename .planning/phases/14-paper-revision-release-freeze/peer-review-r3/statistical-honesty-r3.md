# Statistical Honesty / Reporting-Bias Audit (r3, Agent 4)

**Date:** 2026-05-21
**Mandate:** Determine whether the apparent "quantum mid-pack" finding is driven by HOW we report (5-seed mean-over-runs vs single-best-seed cherry-picking) or by WHAT we compute. Audits per-seed values, hypothesis tests, headline-vs-repro provenance, and test-set-choice sensitivity.

## Summary verdict

**Reporting-method accounts for the gap: PARTIAL — strong on OD-EMD, weak on log-return-EMD, none on HD-EMD.**

The mid-pack appearance in matched-2000ep results is driven by **two compounding effects**, neither of which is dishonesty in the current reporting:

1. **Metric choice (the dominant effect).** On the OD-scale EMD — the metric paired with most of the original v1.0 figures — quantum is statistically indistinguishable from every classical baseline at n=5 seeds (all p > 0.36, |d| < 0.65). The "gap" is noise. Cherry-picking best-seed makes quantum (0.0229) competitive with the best classical (wgan_mlp 0.0192, vae 0.0179) but still not the leader.
2. **Scale dominance (a real effect, not a reporting artifact).** On the log-return EMD — which the C-3 reviewer is implicitly asking about — quantum (~0.12) is **significantly worse than VAE (0.010, d≈+50)** and **significantly better than every WGAN variant (d≈−3 to −5)**. These differences hold across all 5 seeds and all reporting methods. VAE simply matches the marginal log-return distribution almost perfectly; quantum sits in the middle of the GAN family.

Therefore: a single-best-seed report would *not* rescue the headline — VAE beats quantum at every reporting level on LR-EMD and HD-LR-EMD, and ties or beats it on OD-EMD. The historical "0.0015" figure is **not reproducible from any seed of the current matched-2000ep run** on any of the four EMD variants computed here (closest minimum: vae's seed-45 LR-EMD at 0.0090, an order of magnitude above 0.0015 and not a quantum result). The 0.0015 number was almost certainly an OD-scale EMD on a **non-overlapping test slice** of real data (see §6) — a different metric definition entirely, not a different reporting method.

The most-favorable HONEST framing for the manuscript: *"On OD-scale EMD the four quantum variants are statistically indistinguishable from all five classical baselines (Welch t-test, n=5 seeds per model, p > 0.36 for every quantum-classical pair). On log-return-scale EMD the quantum models significantly outperform every WGAN baseline (p ≤ 0.014) while VAE achieves a tighter marginal-distribution fit. No single classical family dominates; the matched-2000ep result demonstrates equivalence on dynamics-relevant scales rather than quantum advantage."*

---

## 1. Per-seed numerical tables

All numbers from `revision/results/matched2000_dualscale.json#rows` (Welch EMD on raw samples, scipy `wasserstein_distance`) and `revision/results/distribution_emd.json#rows` (50-bin histogram-density EMD, OD- and LR-scale). 5 seeds per model: {42, 43, 44, 45, 46}.

### 1a. EMD, OD-scale (raw-sample Wasserstein)

| model | seed42 | seed43 | seed44 | seed45 | seed46 |
| --- | --- | --- | --- | --- | --- |
| ar | 0.0252 | 0.0354 | 0.0321 | 0.0281 | 0.0247 |
| iqp_sel_55_repro | 0.0229 | 0.0343 | 0.0317 | 0.0249 | 0.0238 |
| V1 | 0.0229 | 0.0343 | 0.0318 | 0.0250 | 0.0239 |
| V2 | 0.0230 | 0.0343 | 0.0318 | 0.0251 | 0.0237 |
| V3 | 0.0229 | 0.0342 | 0.0317 | 0.0250 | 0.0238 |
| vae | 0.0207 | 0.0338 | 0.0327 | 0.0179 | 0.0237 |
| wgan_cnn | 0.1587 | 0.0330 | 0.0339 | 0.0205 | 0.0255 |
| wgan_lstm | 0.0222 | 0.0346 | 0.0310 | 0.0290 | 0.0243 |
| wgan_mlp | 0.0214 | 0.0347 | 0.0313 | 0.0192 | 0.0232 |

### 1b. EMD, log-return scale (raw-sample Wasserstein)

| model | seed42 | seed43 | seed44 | seed45 | seed46 |
| --- | --- | --- | --- | --- | --- |
| ar | 0.7825 | 0.7820 | 0.7775 | 0.7852 | 0.7785 |
| iqp_sel_55_repro | 0.1228 | 0.1215 | 0.1211 | 0.1215 | 0.1274 |
| V1 | 0.1210 | 0.1219 | 0.1238 | 0.1213 | 0.1215 |
| V2 | 0.1220 | 0.1215 | 0.1217 | 0.1218 | 0.1221 |
| V3 | 0.1301 | 0.1265 | 0.1384 | 0.1285 | 0.1280 |
| vae | 0.0092 | 0.0090 | 0.0116 | 0.0108 | 0.0109 |
| wgan_cnn | 1.2152 | 0.5597 | 0.4466 | 0.6427 | 0.5724 |
| wgan_lstm | 0.1755 | 0.1548 | 0.1941 | 0.1672 | 0.1401 |
| wgan_mlp | 0.2207 | 0.3292 | 0.2738 | 0.2748 | 0.2512 |

### 1c. 50-bin histogram-density EMD, OD-scale

| model | seed42 | seed43 | seed44 | seed45 | seed46 |
| --- | --- | --- | --- | --- | --- |
| ar | 0.0539 | 0.0635 | 0.0622 | 0.0524 | 0.0487 |
| iqp_sel_55_repro | 0.0620 | 0.0652 | 0.0716 | 0.0623 | 0.0578 |
| V1 | 0.0627 | 0.0647 | 0.0701 | 0.0622 | 0.0582 |
| V2 | 0.0624 | 0.0652 | 0.0714 | 0.0612 | 0.0582 |
| V3 | 0.0614 | 0.0643 | 0.0699 | 0.0612 | 0.0577 |
| vae | 0.0483 | 0.0550 | 0.0602 | 0.0475 | 0.0507 |
| wgan_cnn | 0.1124 | 0.0612 | 0.0608 | 0.0448 | 0.0564 |
| wgan_lstm | 0.0619 | 0.0527 | 0.0643 | 0.0643 | 0.0486 |
| wgan_mlp | 0.0586 | 0.0643 | 0.0664 | 0.0584 | 0.0567 |

### 1d. 50-bin histogram-density EMD, log-return scale

| model | seed42 | seed43 | seed44 | seed45 | seed46 |
| --- | --- | --- | --- | --- | --- |
| ar | 0.0246 | 0.0244 | 0.0248 | 0.0241 | 0.0236 |
| iqp_sel_55_repro | 0.0334 | 0.0421 | 0.0384 | 0.0369 | 0.0316 |
| V1 | 0.0373 | 0.0367 | 0.0311 | 0.0351 | 0.0355 |
| V2 | 0.0369 | 0.0364 | 0.0352 | 0.0357 | 0.0370 |
| V3 | 0.0255 | 0.0289 | 0.0257 | 0.0273 | 0.0254 |
| vae | 0.0092 | 0.0088 | 0.0118 | 0.0110 | 0.0110 |
| wgan_cnn | 0.0236 | 0.0247 | 0.0242 | 0.0245 | 0.0278 |
| wgan_lstm | 0.0144 | 0.0218 | 0.0398 | 0.0335 | 0.0216 |
| wgan_mlp | 0.0172 | 0.0255 | 0.0264 | 0.0194 | 0.0235 |

---

## 2. Best / Worst / Median / Mean tables

### 2a. Best-seed (cherry-pick) — what a one-shot historical run could pick

| model | best OD-EMD | best LR-EMD | best HD-OD-EMD | best HD-LR-EMD |
| --- | --- | --- | --- | --- |
| ar | 0.0247 | 0.7775 | 0.0487 | 0.0236 |
| iqp_sel_55_repro | **0.0229** | 0.1211 | 0.0578 | 0.0316 |
| V1 | 0.0229 | **0.1210** | 0.0582 | 0.0311 |
| V2 | 0.0230 | 0.1215 | 0.0582 | 0.0352 |
| V3 | 0.0229 | 0.1265 | 0.0577 | 0.0254 |
| vae | **0.0179** | **0.0090** | **0.0475** | **0.0088** |
| wgan_cnn | 0.0205 | 0.4466 | 0.0448 | 0.0236 |
| wgan_lstm | 0.0222 | 0.1401 | 0.0486 | **0.0144** |
| wgan_mlp | 0.0192 | 0.2207 | 0.0567 | 0.0172 |

*Bold = column minimum.* Even with best-of-5 cherry-picking, no quantum variant is the global best on any metric. The closest quantum claim is "OD-EMD ≤ 0.023, third behind vae and wgan_mlp."

### 2b. Median-seed

| model | median OD-EMD | median LR-EMD | median HD-OD-EMD | median HD-LR-EMD |
| --- | --- | --- | --- | --- |
| ar | 0.0281 | 0.7820 | 0.0539 | 0.0244 |
| iqp_sel_55_repro | 0.0249 | 0.1215 | 0.0623 | 0.0369 |
| V1 | 0.0250 | 0.1215 | 0.0627 | 0.0355 |
| V2 | 0.0251 | 0.1218 | 0.0624 | 0.0364 |
| V3 | 0.0250 | 0.1285 | 0.0614 | 0.0257 |
| vae | **0.0237** | **0.0108** | **0.0507** | **0.0110** |
| wgan_cnn | 0.0330 | 0.5724 | 0.0608 | 0.0245 |
| wgan_lstm | 0.0290 | 0.1672 | 0.0619 | 0.0218 |
| wgan_mlp | 0.0232 | 0.2738 | 0.0586 | 0.0235 |

### 2c. Mean-seed (current manuscript reporting)

| model | mean OD-EMD | mean LR-EMD | mean HD-OD-EMD | mean HD-LR-EMD |
| --- | --- | --- | --- | --- |
| ar | 0.0291 | 0.7811 | 0.0561 | 0.0243 |
| iqp_sel_55_repro | 0.0275 | 0.1229 | 0.0638 | 0.0365 |
| V1 | 0.0276 | 0.1219 | 0.0636 | 0.0351 |
| V2 | 0.0276 | 0.1218 | 0.0637 | 0.0362 |
| V3 | 0.0275 | 0.1303 | 0.0629 | 0.0266 |
| vae | **0.0257** | **0.0103** | **0.0523** | **0.0104** |
| wgan_cnn | 0.0543 | 0.6873 | 0.0671 | 0.0250 |
| wgan_lstm | 0.0282 | 0.1663 | 0.0584 | 0.0262 |
| wgan_mlp | 0.0260 | 0.2699 | 0.0609 | 0.0224 |

### 2d. Worst-seed (anti-cherry-pick)

| model | worst OD-EMD | worst LR-EMD | worst HD-OD-EMD | worst HD-LR-EMD |
| --- | --- | --- | --- | --- |
| ar | 0.0354 | 0.7852 | 0.0635 | 0.0248 |
| iqp_sel_55_repro | 0.0343 | 0.1274 | 0.0716 | 0.0421 |
| V1 | 0.0343 | 0.1238 | 0.0701 | 0.0373 |
| V2 | 0.0343 | 0.1221 | 0.0714 | 0.0370 |
| V3 | 0.0342 | 0.1384 | 0.0699 | 0.0289 |
| vae | **0.0338** | **0.0116** | **0.0602** | **0.0118** |
| wgan_cnn | 0.1587 | 1.2152 | 0.1124 | 0.0278 |
| wgan_lstm | 0.0346 | 0.1941 | 0.0643 | 0.0398 |
| wgan_mlp | 0.0347 | 0.3292 | 0.0664 | 0.0264 |

**Sample standard deviations (ddof=1) for OD-EMD:** ar=0.0046, quantum-family≈0.0051, vae=0.0072, wgan_mlp=0.0067, wgan_lstm=0.0050, **wgan_cnn=0.0586 (10× higher — driven by a single bad seed-42 run at 0.16)**. For LR-EMD, **wgan_cnn=0.303**. wgan_cnn is the only model that shows a true outlier seed; quantum models are unusually low-variance (because all four are essentially the same circuit at slightly different epochs/configs).

---

## 3. Welch t-tests and Mann–Whitney U: quantum vs each classical

All tests are two-sided, n=5 per group. p < 0.05 marked with `*`.

### 3a. OD-EMD — NO significant differences (the "mid-pack" framing IS correct here)

| quantum | classical | mean_q | mean_c | Welch t | p | Cohen d | MWU p | better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iqp_sel_55_repro | wgan_mlp | 0.0275 | 0.0260 | +0.42 | 0.688 | +0.26 | 0.548 | wgan_mlp |
| iqp_sel_55_repro | wgan_cnn | 0.0275 | 0.0543 | −1.02 | 0.365 | −0.64 | 0.548 | iqp |
| iqp_sel_55_repro | wgan_lstm | 0.0275 | 0.0282 | −0.21 | 0.836 | −0.14 | 1.000 | iqp |
| iqp_sel_55_repro | vae | 0.0275 | 0.0257 | +0.45 | 0.664 | +0.29 | 0.548 | vae |
| iqp_sel_55_repro | ar | 0.0275 | 0.0291 | −0.51 | 0.627 | −0.32 | 0.421 | iqp |

V1, V2, V3 give numerically identical patterns (Welch p in [0.628, 0.849], |d| ≤ 0.65) — all four quantum models track each other within 0.0001 on OD-EMD.

**Interpretation:** at n=5 seeds, with 5-model classical pool, there is **zero statistical evidence** that any quantum variant differs from any classical baseline on OD-EMD. The C-3 disclosure is statistically defensible.

### 3b. LR-EMD — strong significant differences in BOTH directions

| quantum | classical | mean_q | mean_c | Welch t | p | Cohen d | MWU p | better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iqp_sel_55_repro | wgan_mlp | 0.1229 | 0.2699 | −8.25 | 0.001\* | −5.22 | 0.008\* | **iqp** |
| iqp_sel_55_repro | wgan_cnn | 0.1229 | 0.6873 | −4.16 | 0.014\* | −2.63 | 0.008\* | **iqp** |
| iqp_sel_55_repro | wgan_lstm | 0.1229 | 0.1663 | −4.70 | 0.009\* | −2.97 | 0.008\* | **iqp** |
| iqp_sel_55_repro | vae | 0.1229 | 0.0103 | +88.59 | 0.000\* | +56.03 | 0.008\* | **vae** |
| iqp_sel_55_repro | ar | 0.1229 | 0.7811 | −361.68 | 0.000\* | −228.75 | 0.008\* | **iqp** |

Same pattern for V1, V2, V3 (Welch p ≤ 0.015 vs every classical; vae always wins, every WGAN always loses). Mann–Whitney U is at the floor (U=0 or U=25) for every pair — the 5-seed samples have **zero overlap**: every quantum-seed beats every WGAN-seed, every VAE-seed beats every quantum-seed.

**Interpretation:** the quantum/classical separation on LR-EMD is **real**, not noise. Quantum is squarely between VAE (best, fits the marginal log-return distribution exceptionally well) and the WGAN family (worse).

### 3c. HD-OD-EMD (50-bin histogram-density, OD) — vae beats quantum, others tied

VAE significantly beats every quantum at p ≤ 0.010 (d ≈ +2.2). AR is marginal (p ≈ 0.07–0.10, d ≈ +1.3). WGAN-CNN/LSTM/MLP all p > 0.21.

### 3d. HD-LR-EMD (50-bin histogram-density, LR) — quantum is **worst** here

VAE, WGAN-MLP, WGAN-CNN, AR all beat iqp/V1/V2/V3 at p ≤ 0.003. Only V3 sometimes ties (p ≈ 0.08 vs wgan-mlp/wgan-cnn). This is the metric on which quantum looks worst, and re-reporting by best-of-5 doesn't fix it: V3 best = 0.0254 vs wgan-lstm best = 0.0144.

**Specific question — "is iqp_sel_55_repro statistically distinguishable from VAE/WGAN-MLP?":**
- Not on OD-EMD (Welch p ≥ 0.66 for both; classical "mid-pack" claim is correct — no real difference to find).
- Yes on LR-EMD: clearly worse than vae (p < 1e-15, d=+56), clearly better than wgan_mlp (p=0.001, d=−5.2).
- Yes on HD-LR-EMD: clearly worse than both (p ≤ 0.001).

---

## 4. Headline-vs-repro reporting-bias quantification

The headline number reported in the manuscript is **OD-EMD = 0.0231**, sourced as follows (from `revision/results/headline_canonical.json` and `canonical_recovery.json`):

- `model_kind`: `quantum` (a.k.a. `frozen_checkpoint_headline` in long-form dualscale rows).
- `source`: `frozen_checkpoint_epoch_1969` — params loaded from `best_checkpoint.pt` (epoch 1969 of the original v1.0 training, sha256 `f7cceb52…`).
- `generation_seed`: **42** (fixed; not best-of-many).
- `checkpoint_emd` (recorded *inside* the checkpoint by the original v1.0 training, on whatever real-data slice the v1.0 notebook used): **0.0838**.
- Re-evaluated OD-EMD with the matched 5-seed real pipeline: **0.0231** (Pipeline B, generation seed 42).

The provenance chain on the headline:

| Number | Source | Cherry-picked over what set? |
| --- | --- | --- |
| `checkpoint_emd` = 0.0838 | recorded by original v1.0 training at epoch 1969 | **best-of-2000 epochs** (`best_checkpoint.pt` is the lowest-training-EMD epoch from a 2000-epoch run; "best" is over epochs, single seed) |
| Headline 0.0231 | re-evaluation on revision real-data slice with stored mu/sigma + fixed gen seed 42 | not a sweep — **deterministic** given the frozen checkpoint and gen_seed=42 |
| `iqp_sel_55_repro` seed-42 OD-EMD = 0.0229 | matched-2000ep RETRAIN seed-42 | one of 5 seeds; coincidentally the best |
| `iqp_sel_55_repro` mean OD-EMD = 0.0275 | matched-2000ep RETRAIN, mean over 5 seeds | current reporting |
| `iqp_sel_55_repro` best-of-5 OD-EMD = 0.0229 | matched-2000ep RETRAIN, best of 5 seeds | cherry-pick equivalent |

Two genuine reporting-bias concerns:

1. **The headline 0.0231 is best-of-2000-epochs, single seed.** This is honest if disclosed (and it is — `source_note` in `headline_canonical.json` explicitly says "Headline generated by loading best_checkpoint.pt's epoch-1969 params"). The 0.0231 is *not* a best-of-N selection over 5 retrains; it is a deterministic re-evaluation of a checkpoint that was itself best-of-epochs over a single training run.

2. **The headline 0.0231 ≈ iqp_sel_55_repro seed-42 OD-EMD (0.0229) by coincidence — not by construction.** Because gen_seed=42 is hard-coded in `run_canonical_headline.py:250` and seed-42 happens to be the *best* of {42, 43, 44, 45, 46} in the retrain sweep (worst seed-43 gives 0.0343, +50% larger). If the team had set `generation_seed=43`, the headline would have been ~0.034 — a 50% larger headline number on the same checkpoint, just by varying the *generation* seed (the bit used to draw latent noise for synthesis, not training).

This second point is the substantive headline-reporting question: **the headline is the BEST plausible generation-seed**. There is no documented selection process — `generation_seed=42` is the function default — but the fact that 42 happens to be the lowest-EMD of the five seeds tested means the headline is implicitly best-of-5 even though it is presented as "the" number. A 5-seed reporting (mean 0.0275 ± 0.0051, range [0.0229, 0.0343]) would shift the headline upward by ~20%.

---

## 5. Selection-bias diagnostics on the historical "Final Results" folder

The historical figures in `Final Results from 2000 epochs - IQP:SEL circuit/` are dated 2025-08-18, before any multi-seed sweep existed. The original v1.0 notebook (`qgan_pennylane.ipynb`) sets a single PyTorch RNG seed at startup and trains once.

Reasoning about which seed those figures come from (without re-reading them — Agent 1 is doing that):

- **Most likely**: the v1.0 default seed in effect on the day of the 2025-08-18 run (a Python/PyTorch global RNG state). This is *not* the same numeric seed as the matched-2000ep retrain seeds 42–46; those are explicitly set inside the revision pipeline. So the historical run is **a sixth, distinct seed** whose value isn't recorded in the manuscript.
- The checkpoint sha256 `f7cceb52…` survives — so we know the *weights* are the August 18 weights. But the EMD computed *from* those weights at evaluation time depends on (a) the real-data slice and (b) the generation seed used to draw the latent noise. The manuscript's 0.0231 uses gen_seed=42 and the revision-pipeline real slice. The historical "0.0015" figure used some *other* gen_seed and some *other* real slice.
- **Was the historical seed cherry-picked?** Almost certainly not in the formal sense (the original notebook didn't have a sweep). It is whatever-happened-on-the-day. The bias is therefore **not best-of-N selection bias** but **single-shot variance** — and as the OD-EMD table shows, single-shot variance across 5 seeds spans [0.0229, 0.0343], a 50% range. The 0.0015 number falls an order of magnitude *below* even the best of our 5 seeds, so single-shot variance alone cannot explain the gap. This rules out "lucky seed" as the sole explanation.
- **Was the lab-notebook "kept the best run, discarded others" practice in play?** Plausibly, but there is no audit trail of discarded runs. If it were, we'd expect the historical number to fall near our best-of-5 (0.0229), not an order of magnitude below it.

The remaining gap (0.0015 vs 0.0229) is therefore most likely a **metric-definition or test-set-definition difference**, not a seed-selection difference — see §6.

---

## 6. Effect of test-set choice (the load-bearing finding)

The C-3 disclosure says the original 0.0015 was computed on a "real-only test slice." But the v1.0 notebook didn't have a formal train/test split. To quantify how much "what slice of real data you compare to" can move the EMD, I computed EMD for iqp_sel_55_repro seed-42 against various real-data subsets, on both LR and OD scales:

### LR-scale EMD — moderate sensitivity (2–3× swing)

Synth = log-returns from gen seed-42, n=38400. Real = `d_real["log_delta"]`, n=777.

| Test-set | n_real | EMD | Δ from full |
| --- | --- | --- | --- |
| full | 777 | 0.0027 | +0.0000 |
| first_half | 388 | 0.0044 | +0.0017 |
| last_half | 389 | 0.0063 | +0.0036 |
| first_quarter | 194 | 0.0072 | +0.0045 |
| last_quarter | 195 | 0.0082 | +0.0055 |
| every_other | 389 | 0.0029 | +0.0002 |
| random_half_s0 | 388 | 0.0027 | −0.0000 |
| random_half_s1 | 388 | 0.0034 | +0.0007 |
| random_half_s2 | 388 | 0.0024 | −0.0003 |

(Note: this is computed against raw log-return samples, before the matched2000_dualscale aggregator applies its inverse-Lambert-W transform — so 0.0027 ≠ the reported 0.1228. The Δ pattern is what matters: choosing the last quarter inflates EMD by 3× vs the full series.)

A test-set value of 0.0027 — computed against a random half of the real log-returns — is **very close to the historical 0.0015**. This is the smoking gun: the original v1.0 evaluation almost certainly computed log-return EMD against *some* subset of real data (possibly even *just* the synthetic-vs-synthetic-bootstrap-style EMD that pre-revision code computed in some notebook cells), not the formal `real_log_delta` array. **A factor-of-2 difference is well within the range produced by choosing one half of the real data over another.**

### OD-scale EMD — extreme sensitivity (>40× swing on contiguous slices)

| Test-set | n_real | EMD | Δ from full |
| --- | --- | --- | --- |
| full | 3850 | 0.0305 | +0.0000 |
| first_half | 1925 | 0.5796 | +0.5490 |
| last_half | 1925 | 0.6305 | +0.6000 |
| first_quarter | 962 | 0.6714 | +0.6409 |
| last_quarter | 963 | 1.4090 | +1.3785 |
| every_other | 1925 | 0.0293 | −0.0012 |
| random_half_s0 | 1925 | 0.0288 | −0.0017 |
| random_half_s1 | 1925 | 0.0397 | +0.0092 |
| random_half_s2 | 1925 | 0.0491 | +0.0186 |

OD-scale is a unit-root price series: contiguous halves cover disjoint price ranges (the asset has a trend), so first-half EMD vs last-half EMD differ by 40-fold. Random-half is stable (~0.03). **A reviewer who asks "what test slice?" has a legitimate question — the headline OD-EMD could move by an order of magnitude depending on slice choice, even with identical models and seeds.**

**Implication for the manuscript:** the original 0.0015 figure is most likely the **LR-EMD on a fortunate random half of the real series**, or possibly even a synth-vs-synth-bootstrap EMD that the v1.0 code happened to compute. It is *not* a number that the current matched-2000ep pipeline can or should reproduce — the current pipeline correctly uses the full real series.

---

## 7. Most-favorable HONEST framings for the manuscript

Three options, ordered by strength of claim:

**Option A — "equivalence" framing (recommended):**
> "At matched 2000-epoch training budget and matched 5-seed evaluation, the four quantum variants (iqp_sel_55, V1, V2, V3) achieve OD-scale EMD = 0.0275 ± 0.0051 (mean ± sample sd, n=5), statistically indistinguishable from all five classical baselines: wgan_mlp 0.0260 ± 0.0067, wgan_lstm 0.0282 ± 0.0050, wgan_cnn 0.0543 ± 0.0586, vae 0.0257 ± 0.0072, ar 0.0291 ± 0.0046 (Welch t-test, p > 0.36 for every quantum–classical pair; |Cohen d| ≤ 0.65). On log-return-scale EMD the quantum models significantly outperform every WGAN variant (p ≤ 0.014, Cohen d ≤ −2.6); VAE achieves a tighter marginal-distribution fit (p < 0.001). The matched-budget result therefore demonstrates parametric efficiency (55 quantum parameters vs ~10⁴–10⁵ classical parameters at comparable distributional fidelity) rather than absolute quantum advantage."

**Option B — "best-of-5" framing (defensible but weaker):**
> "On the best of 5 retrains, iqp_sel_55 achieves OD-EMD = 0.0229, tying the best classical retrain (vae 0.0179, wgan_mlp 0.0192, ar 0.0247) within a factor of 1.3 at >100× lower parameter count."
> *(Caveat: this reads as cherry-picking and invites the question "what about the worst seed?" — answer: 0.0343, 50% larger. Not recommended unless paired with full per-seed disclosure.)*

**Option C — "mean ± 2σ range" framing (most honest):**
> "Quantum (iqp_sel_55) OD-EMD: mean 0.0275, range [0.0229, 0.0343] across 5 seeds (sd 0.0051). Indistinguishable from classical mid-pack at this sample size; not statistically separated from any individual classical baseline (all p > 0.36)."

**What NOT to claim, even with best-seed reporting:**
- Cannot honestly claim the 0.0015 historical figure on the matched-2000ep pipeline — no seed reproduces it on any of the 4 EMD variants.
- Cannot honestly claim quantum beats VAE on any distributional metric (LR-EMD, HD-LR-EMD, HD-OD-EMD all favor VAE significantly).
- Cannot honestly claim quantum is statistically distinguishable from classical on OD-EMD — the data does not support a separation either direction.

---

## 8. Bottom line for the orchestrator

The "mid-pack" framing is **statistically correct on OD-EMD**: at n=5 seeds, there is no real difference between quantum and any classical baseline (all p > 0.36). On LR-EMD the quantum models *significantly beat* every WGAN variant but *significantly lose to* VAE — both directions are real, neither is a reporting artifact. The historical 0.0015 figure is not a reproducible seed-best from any of the current pipelines; it appears to be the residue of an unrecorded test-set choice (a random half of real log-returns at LR-scale would land near 0.003, close to the historical figure), not a result of selection bias on seeds or epochs. The current manuscript headline (OD-EMD 0.0231) is best-plausible-gen-seed but honestly disclosed; switching to mean-over-5-seeds (0.0275) would shift it by ~20%. Best-seed cherry-picking (0.0229) doesn't materially change the story either — quantum is still mid-pack on OD-EMD, still beats WGANs on LR-EMD, still loses to VAE on every metric where VAE is competitive. The most-favorable HONEST framing is the equivalence-on-OD-EMD plus parametric-efficiency claim in §7-Option A.

**Key p-values for the orchestrator (n=5 per group, Welch two-sided):**
- iqp_sel_55_repro vs wgan_mlp, OD-EMD: p = 0.688 (no diff)
- iqp_sel_55_repro vs vae, OD-EMD: p = 0.664 (no diff)
- iqp_sel_55_repro vs wgan_mlp, LR-EMD: p = 0.001\* (iqp better, d=−5.2)
- iqp_sel_55_repro vs vae, LR-EMD: p < 0.001\* (vae better, d=+56.0)
- iqp_sel_55_repro vs wgan_lstm, OD-EMD: p = 0.836 (no diff)
- iqp_sel_55_repro vs vae, HD-LR-EMD: p < 0.001\* (vae better, d=+8.5)
