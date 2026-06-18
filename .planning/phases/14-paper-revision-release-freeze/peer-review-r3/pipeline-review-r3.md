# Data-Pipeline Lineage Audit (r3, Agent 2)

> **Addendum (2026-05-24):** The VAE "posterior collapse (sample std ≈ 0.0004)" characterization in this document was NOT supported by the matched-budget data. The actual matched-budget VAE log-return std is 0.0186 (≈ 1.17× narrower than real 0.0217, not 54× narrower). The VAE's anomalously low LR-DTW = 0.088 reflects a degenerate generation regime (marginal well-aligned, log-return lag-1 ACF = −0.648 vs real −0.064, matched-pipeline reference) rather than posterior collapse. See `docs/peer_review_remediation.md` for the corrected characterization. This document is preserved unchanged below as a record of the prior belief and the bug-discovery timeline.

---

**Scope.** Forensic audit of the data pipeline used by the matched-2000ep
quantum-vs-classical comparison versus the pre-v1.0 / v1.0 / v1.1 notebook
(`qgan_pennylane.ipynb`). The motivating concern is that current quantum
numbers look mid-pack while the pre-v1.0 paper headline (~0.0015) was
visibly better.

**Verdict — pipeline consistent with notebook? PARTIAL NO.**
- The `core/` module is bit-faithful to the v1.0/v1.1 notebook
  (Lambert W still present, log-returns + dither + rolling-window verbatim).
- BUT the matched-2000ep driver (`run_matched2000.py`) **only uses
  Pipeline B** (log-returns + standardize, NO Lambert W) — a *deliberate*
  divergence locked in by D-10-05.
- AND the matched-2000ep "log_return EMD" reported in `matched2000_dualscale.json`
  is computed under a **50× scale mismatch** that silently penalizes generators
  whose output covers the full [-1,1] range and rewards collapsed generators.
- The pre-v1.0 0.0015 headline used a different metric (50-bin
  histogram-density Wasserstein on a real-only test slice). Apples-to-apples
  it is now reported via `run_distribution_emd.py` (Plan 14-15).

Severity: **HIGH for the log-return EMD column** (it ranks VAE above
quantum because VAE collapsed near zero — *metric gaming*, not quality);
**MEDIUM for the headline comparison** (the 0.0015 vs 0.121 gap is a
metric-redefinition + scale-mismatch artifact stacked on top of a real
quality regression of ~5×).

---

## 1. Forward + Inverse Transformation Chains

### 1.1 v1.0/v1.1 Notebook Pipeline (canonical, Pipeline C)
Cells 5, 9, 15, 17, 18, 21, 22, 23, 30 of `qgan_pennylane.ipynb`, mirrored
verbatim in `core/data.py:load_and_preprocess`:

```
FORWARD (data → training samples):
  CSV → OD                                          # cell 5
  OD → log_delta = log(OD[1:]) - log(OD[:-1])       # cell 9 (with dither U(-0.005,0.005), DITHER_SEED=42)
  log_delta → norm_log_delta (zero-mean / unit-std) # cell 15 (mu, sigma)
  norm_log_delta → transformed_norm_log_delta       # cell 18 (inverse_lambert_w(., delta) with delta = argmin |excess kurtosis|; delta ≈ 0.147)
  transformed_norm_log_delta → scaled_data ∈ [-1,1] # cell 21 (linear rescale via min/max)
  scaled_data → windowed_data (M, 10)               # cell 30 (rolling_window, stride 2)

INVERSE (gen_output ∈ [-1,1] → fake on log_delta scale):
  full_denorm_pipeline @ data.py:174-202
    1. flatten gen_windows
    2. rescale [-1,1] → [transformed_norm_log_delta min, max]
    3. lambert_w_transform(., delta)               # Lambert W FORWARD
    4. denormalize(., mu, sigma)                   # multiply by sigma, add mu

Standalone generation cells 46/47: gen_output * GEN_SCALE (=1.0), then
full_denorm_pipeline. EMD computed on the resulting unnormalized log-delta
scale (cells 59, 66): wasserstein_distance(log_delta_np, fake_log_delta_np).
```

### 1.2 Matched-2000ep Pipeline (current paper artifacts, Pipeline B)
`run_matched2000.py:build_dataset_for_pipeline` (lines 213-254):

```
FORWARD (data → training samples):
  CSV → OD                                          # core.data.load_and_preprocess
  OD → r = log(OD[1:]) - log(OD[:-1])               # forward_logreturns @ core/preprocessing.py:29-46
  r → r_norm = (r - mean) / std                     # ddof=1, single global mu/sigma
  r_norm → r_pm1 = 2*(r_norm - r_min)/(r_max - r_min) - 1   # min-max to [-1,1]
  r_pm1 → windowed (M, 10)                          # rolling_window stride 2
  inverse_kwargs saved: {r_min, r_max, mu, sigma, od_starts}

INVERSE (samples.npy ∈ [-1,1] → OD):
  reconstruct_od @ run_figure_suite.py:261-296
    r_norm = ((samples_pm1 + 1) / 2) * (r_max - r_min) + r_min    # un-minmax to standardized log-return space
    od_start_per_window = rng.choice(od_starts_pool, ...)          # rng = np.random.default_rng(seed*7919+1)
    od_full = inverse_logreturns(r_norm, od_start, mu, sigma)      # un-standardize then cumsum-exp from od_start
    trim last column to 10
```

### 1.3 Drift — Forward Side
| Notebook | Matched-2000ep | Comment |
|----------|----------------|---------|
| Pipeline C (Lambert W + min-max) | Pipeline B (no Lambert W, min-max) | **DELIBERATE** (D-10-05). Removes the Lambert W heavy-tail step that the r1 R1-M3 reviewer flagged. |
| `*0.1` in training-loop critic input (cell 26) | `*0.1` in training-loop critic input (training.py:347, 381, 416) | **CONSISTENT** |
| Standalone gen uses `* GEN_SCALE = *1.0` (cells 46-47) | Standalone gen uses `*0.1` (run_matched2000.py:281, run_baselines.py:205) | **DIVERGENT** — generator output is artificially compressed 10× at sample time vs the notebook headline path. |

### 1.4 Drift — Inverse Side
| Step | Notebook | Matched-2000ep | Comment |
|------|----------|----------------|---------|
| Rescale from [-1,1] | `rescale(., transformed_norm_log_delta)` | `(samples+1)/2 * (r_max-r_min) + r_min` | Both linear; B uses standardized log-return min/max, C uses post-Lambert min/max. |
| Heavy-tail step | `lambert_w_transform(., delta)` | (absent) | **B has no Lambert W; the post-min/max array IS the standardized log-return.** |
| Un-standardize | `denormalize(., mu, sigma)` | `inverse_logreturns(..., mu, sigma)` integrates from od_start | **Distinct purposes** — B re-cumsum-exps to OD; C stops at unnormalized log-delta. |

---

## 2. Per-Model Output-Scale Audit (Pitfall 3)

`results/matched2000/runs/<model>/42/samples.npy` empirical ranges
(measured for this audit):

| Model | min(samples_pm1) | max(samples_pm1) | std | `*0.1` applied? |
|-------|------------------|------------------|------|-----------------|
| iqp_sel_55_repro | −0.077 | +0.076 | 0.021 | YES |
| V1 / V2 / V3 (quantum) | ±0.06–0.09 | ±0.06–0.09 | 0.018–0.029 | YES |
| wgan_mlp | −0.243 | +0.321 | 0.074 | YES (gen output drifts past 1) |
| wgan_cnn | **−0.976** | **+1.460** | **0.362** | YES (gen output blew far past [-1,1]) |
| wgan_lstm | −0.110 | +0.097 | 0.053 | YES |
| **vae** | **−0.045** | **−0.011** | **0.005** | NO `*0.1` — **POSTERIOR COLLAPSE** |
| ar | −1.155 | +1.154 | 0.242 | NO `*0.1` |

`*0.1` lives at `training.py:347, 381, 416` (training-loop critic feed) and
`run_matched2000.py:281` / `run_baselines.py:205` (standalone sample
emission). It is applied unconditionally to quantum + WGAN; explicitly NOT
applied to VAE/AR (Pitfall 3, run_baselines.py:28-31, run_matched2000.py:732,780).

**Consequence — the log_return EMD column ranks generators on noise-floor
proximity to the standardized mean, not on quality.** The inverse mapping
`((s+1)/2)*(r_max-r_min) + r_min` puts:

| Model | samples scale | reconstructed r_norm location | log_return EMD reported |
|-------|--------------|-------------------------------|--------------------------|
| Quantum (±0.08 in samples_pm1) | 0.08 → r_norm ≈ 0.5*(r_max-r_min)+r_min ≈ +0.125 ± 0.33 (std≈0.075) | near standardized mean, narrow | **0.121** |
| VAE (centered at −0.03, std 0.005) | mapped tightly around r_norm ≈ +0.005 ± 0.02 | **collapsed near mean** | **0.009** ← *METRIC GAME* |
| AR (full ±1.15 in samples_pm1) | mapped fully across [r_min, r_max] (std ≈ 1.0) | full range | **0.783** |

The 50× scale mismatch — `r_norm` is in STANDARDIZED log-return space
(real std ≈ 1), but EMD is computed against `real_log_delta` (UN-standardized
real, std = 0.022). All values are inflated by ~50×, but the **rank
inversion** comes from the per-model output-scale differences.

When I re-compute EMD on the proper UN-standardized log-delta scale
(`r_norm * sigma + mu`):

| Model | log_return EMD (mismatched, current) | log_delta EMD (proper) |
|-------|-------------------------------------|------------------------|
| V1 (quantum) | 0.121 | **0.015** |
| VAE | 0.009 | 0.016 |
| AR | 0.783 | **0.003** |

The ranking *inverts* once the scale is fixed. **AR (closed-form lstsq) is
in fact the best fit to the marginal log-delta distribution.** This is the
expected behavior — AR(2) fits the linear-Gaussian marginal of a log-return
series perfectly by construction.

---

## 3. Slice-Size Effect (Empirical)

Self-EMD test on `real_log_delta` (n=777, std=0.022):

| Slice size | EMD(full_real, random_slice) median | EMD(slice, slice) median |
|-----------|--------------------------------------|---------------------------|
| 50 | 0.00365 | 0.00541 |
| 100 | 0.00260 | 0.00382 |
| **200** | **0.00168** | 0.00233 |
| 500 | 0.00074 | 0.00103 |
| 777 (full) | 0.000 | — |

**The pre-v1.0 ~0.0015 headline is empirically consistent with a ~200-point
real-only test slice EMD on the unnormalized log-delta scale.** That matches
the C-3 disclosure description (50-bin histogram-density on a "real-only test
slice").

Crucially, **applying the v1.0 pipeline (Lambert W + GEN_SCALE undone) to
the CURRENT V1 quantum samples gives EMD = 0.0084** on the proper log-delta
scale (full-vs-full), and 0.008 on every slice size from 50–500. So even
once the scale + metric formulation are aligned, the matched-2000ep
quantum is genuinely ~5× worse than the historical headline. **The
historical 0.0015 was the best-checkpoint pre-v1.0 quantum on a real-only
test slice; the current matched-2000ep 0.008 is the final-epoch
post-2000-epoch trained model on the full real series.** Both metric
formulation and selection criterion differ.

---

## 4. Lambert W Status

**Lambert W is NOT removed from the codebase** — `core/data.py:70-145`
preserves `inverse_lambert_w_transform` (forward + differentiable backward
via `_InverseLambertW(torch.autograd.Function)`) and `lambert_w_transform`
verbatim from notebook cell 17.

**BUT — Lambert W is bypassed in the matched-2000ep training and inverse**
pipeline by switching to Pipeline B (D-10-05). The Lambert W is only
exercised by:
- `core/preprocessing.py:20-23` (re-export as `forward_lambert` / `inverse_lambert`)
- `core/data.py:full_denorm_pipeline` (cell 23 inverse — UNUSED by the matched-budget driver)
- Any artifact tagged Pipeline C (none in matched2000).

**Impact on quantum-vs-classical comparison.** The r1 R1-M3 concern was
that Lambert W gave the quantum model a representational advantage by
pre-shaping the target distribution into Gaussian. Pipeline B removes that
advantage symmetrically (all models train on the same standardized log-returns
without heavy-tail pre-shaping). **The Lambert W removal is fair to all
models** — it does not preferentially hurt quantum. (Verified: the v1.0
quantum samples processed through Pipeline-C inverse give EMD 0.008
vs Pipeline-B inverse 0.015; both are far worse than the pre-v1.0 0.0015.
The gap is not Lambert-W-explainable.)

---

## 5. Top 3 Hypotheses for the Historical-vs-Current Discrepancy (ranked by evidence)

### Hypothesis 1 — Metric redefinition (HIGH confidence)
The pre-v1.0 paper's ~0.0015 was **50-bin histogram-density Wasserstein**
on a small real-only test slice (the C-3 disclosure in
`reconciliation_note.md`). The current matched-2000ep ~0.121 is **raw-sample
Wasserstein** on a scale-mismatched reconstruction
(`r_norm` standardized vs `log_delta` unnormalized).

Phase 14-15 added `run_distribution_emd.py` to compute the histogram-density
EMD on OD-scale samples — that is the apples-to-apples comparison. The
"log_return" raw-sample column should be considered RETIRED for
cross-paper comparisons.

**Evidence.** `.planning/phases/14-paper-revision-release-freeze/14-15-PLAN.md`
lines 43, 1079, 1237 explicitly state the histogram-density reintroduction
"makes the matched-2000ep numbers commensurate with the pre-v1.0 paper
headline (~0.0015) for the first time since the v1.0 metric switch".

### Hypothesis 2 — Scale mismatch in current "log_return EMD" column (HIGH confidence)
The `matched2000_dualscale.json#aggregates[scale='log_return', emd]`
column compares standardized synth (std ≈ 1 for an ideal generator)
against unnormalized real (std = 0.022). The values are 50× inflated and
the model ranking is **gamed by VAE posterior collapse** (collapse pulls
synth near the standardized mean, which happens to be near the
unnormalized real mean — but only because both are near zero).

**Evidence.** Direct empirical measurement (see §2). VAE samples are
std=0.005 and report 0.009 log_return EMD (best of all models); AR samples
with realistic std=0.24 report 0.783 (worst). On the proper un-standardized
scale, the ranking reverses (VAE 0.016, AR 0.003).

**Action.** This column should not be in the headline table. The dual-scale
recipe is preserved verbatim from `run_dualscale_fidelity.py:_log_return_rows`
(D-11-10), so the bug is structural — both files compare standardized synth
vs unnormalized real. The OD-scale column (post-inverse-logreturns) is the
honest comparison.

### Hypothesis 3 — Genuine quality regression at the 2000ep matched budget (MEDIUM confidence)
Even after correcting for hypothesis 1 + 2 (applying v1.0 Lambert W
pipeline + scale-aligning), the current V1 quantum scores ~0.008 EMD on
unnormalized log-delta. The historical 0.0015 (under any reasonable
small-slice interpretation, see §3) is ~5× better. Even allowing for
slice-size effect on the *real reference*, ~5× is real.

Plausible drivers (require Agent 3's quantum-circuit audit + Agent 4's
training-loop audit to disentangle):
- Best-checkpoint (epoch 1969) vs final-epoch difference: the headline
  uses a frozen best-epoch checkpoint with EMD 0.084 (training-loop
  metric); the matched-2000ep uses the trained-to-completion final state.
  Best-checkpoint always wins by construction.
- `*0.1` at standalone sample time (matched-2000ep) vs `GEN_SCALE=1.0` at
  standalone sample time (notebook). The current samples are compressed
  to 10% of their training-loop magnitude before being shipped to EMD.
  Whether this hurts or helps is generator-dependent: for a generator that
  naturally produces a wide range, compressing 10× makes its distribution
  too narrow vs the standardized real target; for a generator that already
  produces a near-zero output, compression makes no difference.
- HPO drift: `core/__init__.py:11-14` shows
  `N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8e-5, LR_GENERATOR=6.9e-5`
  (v1.1 Phase-4 HPO-tuned). The pre-v1.0 paper used different hyperparams.

---

## 6. File/Line Citations

- `core/data.py:174-202` — `full_denorm_pipeline` (Pipeline C inverse, Lambert W + denormalize)
- `core/data.py:118-145` — `inverse_lambert_w_transform` / `lambert_w_transform` (Lambert W still present)
- `core/data.py:227-296` — `load_and_preprocess` (full v1.0 forward chain)
- `core/preprocessing.py:29-46` — `forward_logreturns` (Pipeline B forward)
- `core/preprocessing.py:49-72` — `inverse_logreturns` (Pipeline B inverse via cumsum-exp)
- `core/eval.py:25-36` — `compute_emd` (raw-sample Wasserstein, v1.0 locked)
- `core/training.py:347, 381, 416` — `*0.1` in training-loop critic feed
- `run_matched2000.py:213-254` — `build_dataset_for_pipeline` (Pipeline B only)
- `run_matched2000.py:257-284` — `generate_wgan_samples` (`*0.1` at sample emission)
- `run_matched2000.py:732, 780` — VAE/AR explicit "(NO *0.1, Pitfall 3)"
- `run_matched2000_dualscale.py:367-373` — `_log_return_rows` (EMD against `real_log_delta` — SCALE MISMATCH)
- `run_figure_suite.py:261-296` — `reconstruct_od` (canonical Pipeline-B inverse helper)
- `run_distribution_emd.py:78-141` — `compute_histogram_density_emd` (50-bin density Wasserstein restored for pre-v1.0 commensurability)
- `results/headline_canonical.json` — checkpoint_emd = 0.084 at epoch 1969 (training-loop metric, not the pre-v1.0 0.0015)
- `.planning/phases/14-paper-revision-release-freeze/14-13-PLAN.md:677, 680` — pre-v1.0 ≈ 0.0015 headline trajectory disclosure
- `.planning/phases/14-paper-revision-release-freeze/14-15-PLAN.md:43, 1079` — histogram-density reintroduction rationale
