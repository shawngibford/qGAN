# Math/Statistics Peer Review — AIChE qGAN paper revision

Reviewer role: mathematical and statistical rigor at a top-tier journal.
Date: 2026-05-20
Scope: `core/*.py`, `run_*.py`, `docs/methods_full.md`,
`docs/reconciliation_note.md`, `results/*.json`, and
`figures/*.json`.

## Executive verdict

**A math-attentive referee would NOT accept these numbers as currently
presented.** The loss functions and core fidelity metrics are correctly
implemented, parameter counts add up exactly, the gradient penalty is the
standard two-sided Gulrajani 2017 form, and the WGAN-GP sign conventions are
correct. **However**, three independent statistical/honesty issues materially
distort the paper-facing narrative:

1. **CRITICAL — scale collision in the OLD-vs-NEW reconciliation table.** The
   reconciliation_note's "+0.127 EMD degradation at 2000ep" is an arithmetic
   artifact of comparing a Wasserstein on the OD scale (OLD column) to a
   Wasserstein on the in-loop log-return-standardized training space (NEW
   column). On the documented OD scale the true delta is ≈ 0 for every model.
2. **CRITICAL — selection-biased + scale-mixed cross_model_emd figure.** The
   bars are `min` over a 201-point training trajectory (downward biased and
   unreported as such) on the in-loop log-return scale, while the dashed
   reference line is the OD-scale headline EMD. Two different metrics on two
   different scales on the same plot.
3. **MEDIUM — population std (ddof=0) used as the error-bar denominator
   throughout.** Every "mean ± std over 5 seeds" in matched2000_dualscale and
   in the figure suite is `statistics.pstdev` / `numpy.std()` default.
   With n=5 the true sample std is ~12% larger. This is uniform across the
   figures so re-running with `ddof=1` would simply widen every error bar by
   `sqrt(5/4)` ≈ 1.118.

Beyond those: the headline "0.0015 EMD" in the original paper and the new
"0.121 EMD" headline are **two different metrics that share a name** —
histogram-density Wasserstein in the original notebook vs. raw-sample
Wasserstein in the v1.0 audited eval — and nothing in the manuscript-facing
docs explains this re-definition.

## Reproducibility math check (one-liner verifications)

Verified by `python3 -c` invocations in this session:

```
# 3 deltas vs reconciliation_note literals
iqp_sel_55_repro: round(0.154999 - 0.027586, 6) = 0.127413   claim +0.127413 ✓
wgan_lstm:         round(0.146192 - 0.029258, 6) = 0.116934   claim +0.116935 (off by 1 ulp on 6th decimal)
wgan_cnn:          round(0.101747 - 0.113033, 6) = -0.011286  claim -0.011286 ✓
wgan_mlp:          round(0.121527 - 0.027580, 6) = +0.093947  ("+0.093946" appears in reconciliation_note as well)

# Quantum param formulas (core/models/quantum.py:105-109)
iqp_sel_55:    5 + 3*5*3 + 5*1 = 55  ✓ (matches canonical_config_lock.json)
default_75/V1: 5 + 4*5*3 + 5*2 = 75  ✓ (matches v1_config_lock.json)
V2:            5 + 8*5*3 + 5*2 = 135 ✓ (matches v2_config_lock.json)
V3:            5 + 4*5*3 + 5*2 = 75  ✓ (matches v3_config_lock.json; topology change doesn't add params)

# Classical param formulas (classical_architectures.json totals)
wgan_mlp:  5*4+4 + 4*10+10 = 74  ✓
wgan_cnn:  1*9*6+9 + 9*1*1+1 = 73 ✓
wgan_lstm: 4*2*(2+2)+4*2+4*2 + 2*10+10 = 78 ✓  (PyTorch LSTM has two bias vectors, hence 4*2 + 4*2 = 16 bias params)
vae:       10*16+16 + 16*4+4 + 16*4+4 + 4*16+16 + 16*10+10 = 562 ✓
ar(p=2):   2 phi + 1 sigma^2 = 3 ✓
```

All four reconciliation_note deltas (the three rejected by the provenance gate
plus the unrejected wgan_mlp delta) are arithmetically correct as subtractions
of the values written on either side of the row. **They are nevertheless
misleading because the OLD and NEW values are on different scales** — see
CRITICAL-1 below.

## CRITICAL findings

### C-1. Scale collision in reconciliation_note.md "EMD (OD scale)" table

`docs/reconciliation_note.md:9` declares "EMD (OD scale) — final-eval
mean over seeds 42-46" as the table title. Reading the row construction in
`run_model_info.py:220-302`:

- **OLD column** (`old_1000ep`) is built from `baseline_comparison.json` rows
  filtered by `scale == "OD"` and `pipeline == "B"` — correctly on the OD
  scale (`run_model_info.py:282-283`).
- **NEW column** (`new_2000ep`) is built from
  `matched2000/runs/<model>/<seed>/metrics.json["emd_avg"][-1]`
  (`run_model_info.py:240`). But `emd_avg` is populated inside
  `core/training.py:415-420`:

  ```python
  eval_gen = generator(eval_noise)
  eval_gen = eval_gen.to(compute_dtype) * 0.1
  fake_flat = eval_gen.reshape(-1).cpu().numpy()
  real_flat = real_log_returns.reshape(-1).cpu().numpy()
  emd_val = compute_emd(real_flat, fake_flat)
  ```

  `real_log_returns` is a batch of **log-return windows** from the DataLoader.
  `eval_gen * 0.1` is the *0.1-scaled generator output (Pauli-expectation
  domain). Neither side is in OD units. The `emd_avg[-1]` value is therefore
  on the in-loop log-return-standardized scale (with a 0.1 multiplier on
  fakes), not the OD scale.

**The two columns are on different scales, so the row-level deltas have no
physical meaning.** A reader of the manuscript would interpret "+0.127413 OD"
as a 0.127 OD-unit degradation in performance — which is false.

**Direct evidence using the audited dual-scale eval (the document's own claim
that fidelity_dualscale + matched2000_dualscale are the "audited" successors
to baseline_comparison):** when the NEW column is replaced by the audited
matched2000_dualscale.json per-seed OD-scale EMD (the correct apples-to-apples
quantity), the deltas are essentially zero for every model:

```
quantum/iqp_sel_55_repro: NEW_OD = 0.027526  →  delta = -0.000060   (≈0)
wgan_mlp:                  NEW_OD = 0.025952  →  delta = -0.001628   (≈0)
wgan_cnn:                  NEW_OD = 0.054323  →  delta = -0.058710   (IMPROVED)
wgan_lstm:                 NEW_OD = 0.028214  →  delta = -0.001044   (≈0)
```

In other words: **on the OD scale the 2000ep training did NOT degrade EMD; the
"+0.127 degradation" narrative does not exist on the OD scale.** The 80× jump
exists only because the NEW column is being measured on a different metric
space.

**Recommended fix.** Either (a) replace the NEW column with the OD-scale
per-seed mean from `matched2000_dualscale.json` (the file the manuscript
already treats as authoritative for "audited" fidelity), and report deltas
≈ 0; or (b) relabel the table "EMD (in-loop training metric — NOT OD scale,
NOT comparable to audited fidelity)" and explicitly do not compute deltas
between two non-commensurate metrics. Option (a) is correct and consistent
with the rest of the audited corpus.

Citation: `docs/reconciliation_note.md:9-23`,
`run_model_info.py:220-302`, `core/training.py:415-420`,
`results/matched2000_dualscale.json` (rows scale="OD").

### C-2. Selection-biased + scale-mixed cross_model_emd figure

`run_figure_suite.py:620-669` renders the cross_model_emd bar chart.
Two independent issues:

1. **Selection bias.** Line 630 takes `float(np.min(mt["emd_avg"]))` per
   seed — `emd_avg` has 201 entries (eval every 10 epochs over 2000 epochs).
   `min` over 201 noisy evaluations is downward-biased relative to the
   model's true generative performance. For a metric with non-zero noise this
   underestimates EMD by a non-trivial amount.
   - The bar y-axis label is "best EMD over training (mean ± std over 5
     seeds)" — at least the label admits "best", but the caveat that "min
     over 201 evaluations is a selection-biased point estimator" is not
     surfaced anywhere I can find. With n=201 the bias scales as roughly
     `~sigma / sqrt(2*log(201))` for Gaussian-noisy metrics.

2. **Scale mixing.** Line 647-651 reads `frozen_headline_OD_emd` from
   `headline_canonical.json` rows where `scale=="OD"` (= 0.023072) and plots
   it as a horizontal dashed reference line. The bars are on the in-loop
   log-return-standardized scale (per C-1). The reference line and the bars
   are on **two different metric spaces**. Confirmed in
   `figures/cross_model_emd.json`:
   - `best_emd_mean` for iqp_sel_55_repro = 0.1127 (in-loop log-return)
   - `frozen_headline_OD_emd` = 0.0231 (OD)
   A reader sees a horizontal dashed line at 0.023 sitting below all the
   bars at ~0.05-0.11 and would conclude the frozen headline outperforms the
   2000ep reproduction by a factor of 5× — which is a comparison the
   underlying data does not support.

**Recommended fix.** Use OD-scale EMD aggregates from
`matched2000_dualscale.json` (mean over seeds, not min over trajectory) for
the bars, and the same OD-scale headline value for the line. Then both
quantities live in the same metric space. The "best-of-trajectory" framing
should be dropped entirely or moved to a supplementary diagnostic with the
selection-bias caveat explicit.

Citation: `run_figure_suite.py:620-669`,
`figures/cross_model_emd.json`,
`results/headline_canonical.json:53` (OD EMD), training.py:415-420.

### C-3. Original paper's 0.0015 "headline EMD" was a different metric

The original `qgan_pennylane.ipynb` cell 26 / `stylized_facts` function defines
EMD as `wasserstein_distance(empirical_real, empirical_fake)` where
`empirical_real` and `empirical_fake` are **histogram density vectors**
normalized to sum to 1 (`qgan_pennylane.ipynb:1561-1569`):

```python
empirical_real, _ = np.histogram(orig_np, bins=bin_edges, density=True)
empirical_real /= np.sum(empirical_real)
empirical_fake, _ = np.histogram(fake_np, bins=bin_edges, density=True)
empirical_fake /= np.sum(empirical_fake)
emd = wasserstein_distance(empirical_real, empirical_fake)
```

That call passes the two PMF vectors as **values** (not as samples and
weights), so `scipy.stats.wasserstein_distance` treats them as one-dimensional
samples in the range [0, max-density]. The result is dimensionally and
mathematically distinct from the raw-sample Wasserstein.

The current `core/eval.py:25-36` `compute_emd` correctly calls
`wasserstein_distance(real_raw, fake_raw)` on raw sample arrays — the
distributionally correct 1D earth-mover distance over the data distribution.
The v1.0 decision to switch is documented in eval.py's docstring and is the
right call. **However, the manuscript-facing reconciliation note's framing
that the metric "degraded from 0.0015 to 0.121" obscures the fact that these
are two different metrics**. A referee asking "what is the headline EMD now?"
would need to be told: "the metric was redefined; the old 0.0015 was a
histogram-density Wasserstein and is not commensurate with the new 0.121
raw-sample Wasserstein."

This is **not a code bug** — the v1.0 redefinition is documented in
`core/eval.py:25-36` — but a referee will demand that the paper
acknowledge the redefinition explicitly rather than presenting the new number
as a continuation of the old one.

Citation: `qgan_pennylane.ipynb:1561-1569`, `core/eval.py:25-36`,
`docs/reconciliation_note.md` (no mention of the metric
redefinition).

## HIGH findings

### H-1. `training_convergence_all_models` axis label is "OD scale" but data are not OD

`run_figure_suite.py:1146` sets `ax.set_ylabel("EMD (avg over eval
window, OD scale)")`, but the underlying `emd_avg` arrays from
`matched2000/runs/<m>/<s>/metrics.json` are the in-loop log-return-
standardized EMD (training.py:415-420), not OD. The frozen-headline
horizontal line uses `head["checkpoint_emd"] = 0.0838`, which is the
checkpoint's stored training-time EMD on the same in-loop scale, so at least
the bars and line are co-scaled. But the axis label is wrong, and a referee
reading the figure caption will be confused when this 0.0838 line value does
not match the 0.0231 OD-scale headline value in
`headline_canonical.json`.

Citation: `run_figure_suite.py:1146`,
`results/canonical_config_lock.json:2`,
`results/headline_canonical.json:53`.

### H-2. Aggregation uses population std (ddof=0) throughout the figure pipeline

`run_matched2000_dualscale.py:522`:
```python
std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
```
`statistics.pstdev` is population std (ddof=0). With n=5 seeds the unbiased
sample std (`stdev`, ddof=1) is `sqrt(5/4) ≈ 1.118` × larger. Every error bar
in `param_efficiency_pareto`, `failure_modes_summary`, and any other figure
that pulls from `matched2000_dualscale.json["aggregates"]` is therefore
~12% narrower than it should be. The same issue appears in
`run_figure_suite.py:634` (`np.std(finals)`), 1129 (`s.std(axis=0)`), 1892
(`final_vals.std()`).

Recommendation: switch to `statistics.stdev` (ddof=1) and to `np.std(..., ddof=1)`
wherever the spread of a small (n=5) sample is being reported. With n=5,
ddof=1 is the established convention for sample std; pstdev is appropriate
only when the 5 values are the **entire** population, which they are not
(they are 5 samples from an underlying training-noise distribution).

Citation: `run_matched2000_dualscale.py:522`,
`run_figure_suite.py:634`, `:1129`, `:1892`.

### H-3. Shared critic (~250,881 params) absent from parameter-efficiency narrative

`classical_architectures.json["models"]["shared_critic"]["total_params"] =
250,881`. This critic participates in training of every WGAN-GP model —
quantum and classical alike — but is excluded from the parameter counts
reported in `methods_full.md` § 2.f-h, model_info.json, and the
param_efficiency_pareto figure (which uses
`model_info.json["parameter_count"]` = generator-only).

The Pareto chart's x-axis is "log10(parameter_count)" with values in the
~1.7-2.7 range (quantum 55 → log10 ≈ 1.74; vae 562 → log10 ≈ 2.75). The 250k
critic is shared by the adversarial models and dwarfs every generator. A
math-attentive referee would ask: "the paper claims quantum is parameter-
efficient at 55 params vs classical at 74. But the WGAN-GP discriminator (250k
params) is identical across both. The claim should be `+74 generator + 250881
critic = 250955` vs `+55 generator + 250881 critic = 250936`. That's a 0.0075%
parameter reduction, not a ~25% reduction."

Methods doc § 2.k describes the shared critic in prose but does not give the
count, and the Pareto figure does not surface it. Recommendation: add a
"total adversarial parameter count" companion table that includes the critic,
and discuss the parameter-efficiency framing in light of it.

Citation: `results/classical_architectures.json` shared_critic,
`results/model_info.json` (generator-only counts),
`docs/methods_full.md:210-219`.

## MEDIUM findings

### M-1. VAE ELBO uses `mean` over latent dim → implicit β re-weighting

`run_baselines.py:315-319`:
```python
recon = torch.nn.functional.mse_loss(x_hat, x)
kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
loss = recon + beta * kld
```

`mse_loss` defaults to mean over all elements (batch × 10). `kld` does
`torch.mean` over batch × 4 (latent dim 4). The canonical ELBO is per-sample
`sum_z` for KL and `sum_x` for reconstruction, then averaged across the batch.
The current formulation effectively scales KL by `4/(batch*4) = 1/batch` and
recon by `1/(batch*10) = 1/(batch*10)` — i.e., the relative weight of KL vs
recon is `(window/latent) = 10/4 = 2.5` compared to the canonical formulation.
With β=1 in code, this is equivalent to β ≈ 0.4 in the canonical form
(approximately a β-VAE with KL down-weighted).

The methods doc § 2.i states the canonical ELBO equation:

```
L_ELBO(x) = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

This is technically NOT the equation being optimized; the code optimizes a
re-weighted form. **For a math-attentive referee this is a minor formulation
discrepancy** — VAE training works with either weighting and the
posterior-collapse check at run_baselines.py:340 monitors for the obvious
failure mode. But the methods doc should be updated to either reflect the
per-element-mean formulation (which is honest about the implicit β) or the
code should switch to `mse_loss(reduction='sum') / batch_size` + `kld =
-0.5 * sum / batch_size` to match the equation.

Citation: `run_baselines.py:315-319`,
`docs/methods_full.md:189-191`.

### M-2. AR(p) noise variance uses ddof=0 (biased ML estimator, not unbiased)

`core/models/nonadversarial.py:157`:
```python
self.sigma2 = float(resid.var(ddof=0))
```

The standard ML estimator for AR(p) noise variance is
`(1/n) * sum(residuals^2)` (ddof=0 on residuals from a fit that consumed p
degrees of freedom) — biased low by factor `(n-p)/n`. The unbiased estimator
is `1/(n-p) * sum(residuals^2)` (ddof=p on the residuals).

For n=777 log-return rows and p=2, the bias is `(775/777) ≈ 0.9974` — about
0.26% downward bias in sigma^2 → ~0.13% downward bias in sigma. Numerically
trivial for AR(2) on this dataset, but mathematically the convention should
match the methods statement. Methods § 2.j writes the residual as
`varepsilon ~ N(0, sigma^2)` and the OLS fit as
`arg min ||X phi - y||_2^2` — both compatible with either estimator, but a
referee may prefer the unbiased convention. Recommendation: change to
`resid.var(ddof=p)` or document the ML convention.

Citation: `core/models/nonadversarial.py:157`,
`docs/methods_full.md:206-208`.

### M-3. `compute_moments` uses ddof=0 std and Fisher kurtosis — undocumented in methods

`core/eval.py:55-57`:
```python
"std": float(np.std(s)),         # ddof=0 (population)
"skewness": float(skew(s)),       # bias=True (sample, scipy default)
"kurtosis": float(kurtosis(s)),   # Fisher (excess), bias=True (scipy default)
```

The eval.py docstring (lines 8-11) does document this is intentional ("ddof=0
matches cell 59"), but the manuscript-facing methods_full.md does NOT specify
which convention (sample vs population std, Pearson vs Fisher kurtosis) is
used. A reviewer comparing the reported "kurtosis" to the original paper's
"kurtosis" cannot tell if Fisher subtraction (k - 3) was applied — and small
errata about whether real-data kurtosis is "5.2" or "8.2" would surface here.
Recommendation: add a one-line statement to methods_full.md § 2 (or in a
"Metric conventions" subsection) that std is ddof=0, kurtosis is Fisher
(excess), and skew/kurtosis use scipy.stats defaults with bias=True.

Citation: `core/eval.py:42-58`,
`docs/methods_full.md` (no metric convention statement).

### M-4. ACF computed via statsmodels FFT — methods doc says "scipy" implicitly

methods_full.md does not pin the ACF library. `core/eval.py:64-72`
calls `statsmodels.tsa.stattools.acf(s, nlags=20, fft=True)`. statsmodels'
ACF uses a divisor of `n` (biased by default), not `n-k` (unbiased). For a
short series this matters — but the v1.0 decision is documented in
`eval.py:7-11` as a notebook-parity choice. A referee may ask whether the
biased ACF estimator vs the unbiased one was the conscious choice; the
methods doc should clarify.

Citation: `core/eval.py:64-72`.

### M-5. EMD `emd_avg[-1]` is one in-loop snapshot, not 5-seed × full-eval mean

The reconciliation_note NEW column averages `emd_avg[-1]` (one scalar per
seed = the EMD at training-epoch 2000 on **one** batch of 12 fake + 12 real
windows) across 5 seeds. With batch size 12 and a noise-floor EMD around
0.1-0.15, the per-seed estimator variance is non-trivial. Compare against
matched2000_dualscale.json which uses `n_synth = 10 * n_real_windows = 3840`
fakes vs all 384 real windows for the OD-scale EMD — orders of magnitude
more samples per EMD evaluation, hence lower per-seed estimator variance.

The dualscale EMD is therefore both more accurate (more samples) and on the
right scale (OD). reconciliation_note's choice of `emd_avg[-1]` is the worse
estimator on the wrong scale.

Citation: `run_model_info.py:240`, `core/training.py:402-420`.

## LOW findings

### L-1. `wgan_lstm` claimed delta `+0.116935` rounds the 6th decimal up

Computed `0.146192 - 0.029258 = 0.116934`, reconciliation_note says
`+0.116935`. Off by 1 ulp on the 6th decimal — almost certainly because the
true underlying numbers carry more precision than the 6-digit row labels and
the actual delta to full precision rounds to 0.116935. Not a real bug.

Verified by reading metrics.json files directly: wgan_lstm seeds give a more
precise per-seed final emd_avg, and the per-seed mean to higher precision is
slightly above 0.146192. The 6-decimal rounding of the printed "new" value
hides the last-digit adjustment that produces the +0.116935 delta. Acceptable
provided the underlying full-precision numbers live in the JSON sources (they
do).

Citation: `docs/reconciliation_note.md:19`.

### L-2. lambda_gp = 2.16 is non-standard (Gulrajani 2017 uses 10)

Methods_full.md § 3 reports `lambda_gp = 2.16` as an HPO-tuned value (D-14
phase). This is fine for honest reporting, but a referee may ask for the HPO
search range and the criterion that selected 2.16. The methods_full.json
buckets should expose this. Not strictly a math issue.

Citation: `docs/methods_full.md:232`,
`core/training.py:219`.

### L-3. `n_critic = 9` — non-standard but accept

Gulrajani 2017 uses n_critic = 5. Methods reports 9 with HPO provenance.
Same as L-2 — methodologically defensible, just non-standard.

### L-4. Gradient penalty `device` argument is unused (cosmetic API smell)

`core/training.py:31-73` accepts `device` but uses
`real_samples.device` instead (line 55). The comment at lines 51-53 admits
this. Cosmetic only; the GP is computed on the correct device, and the
function signature keeps API symmetry with the notebook. Not a math
correctness issue.

## INFORMATIONAL

### I-1. WGAN-GP loss signs and gradient penalty: CORRECT

- Critic loss (`training.py:364`): `fake_score_mean - real_score_mean +
  lambda_gp * gp` — minimizing this maximizes real_score - fake_score +
  drives gradient norm to 1. Matches Gulrajani 2017 eq. (3). ✓
- Generator loss (`training.py:385`): `-mean(fake_scores)` — minimizing this
  maximizes critic's score on fakes. Standard WGAN-GP generator loss. ✓
- Gradient penalty (`training.py:31-73`): per-sample alpha ~ U(0,1),
  interpolation between real and fake, autograd L2 norm, two-sided penalty
  `((norm - 1)^2).mean()`. Standard Gulrajani 2017 two-sided GP. ✓

### I-2. EMD via `scipy.stats.wasserstein_distance` on raw samples: CORRECT

`core/eval.py:25-36` calls
`wasserstein_distance(real.ravel(), fake.ravel())` — the documented 1D
empirical Wasserstein-1 / earth-mover distance. Inputs are properly
flattened. Eval is on the actual scale of the samples (whichever space the
caller passes in). ✓

### I-3. Quantum parameter formulas match config-locks exactly

`core/models/quantum.py:104-109` ↔
`results/{canonical,default_75,v1,v2,v3}_config_lock.json`
`param_count` fields all match by formula `nq + L*nq*3 + nq*final_rot_factor`
where final_rot_factor ∈ {1 for RX_only, 2 for RX_plus_RY}. ✓

### I-4. Classical parameter formulas match `classical_architectures.json` exactly

All five classical models (wgan_mlp 74, wgan_cnn 73, wgan_lstm 78, vae 562,
ar 3) verify by direct arithmetic. The LSTM 78 = 48 + 30 includes PyTorch's
double-bias convention for `nn.LSTM` (one bias for input-hidden, one for
hidden-hidden), which is the same convention `classical_architectures.json`
encodes via `4 * hidden_size + 4 * hidden_size = 16` extra params. ✓

### I-5. Seed independence: CORRECT

`core/training.py:244-249` seeds torch + numpy + random once at the
top of `train_wgan_gp` before any optimizer/data construction. Subsequent
`np.random.uniform(...)` noise draws (lines 339-345, 373-378, 408-414) and
`torch.rand`/`torch.stack`-via-`torch.randint(...)` data sampling all drain
their respective RNG states deterministically. Different seed values produce
independent trajectories. ✓

### I-6. Number-provenance gate catches the 3 manually-computed deltas

`verify_number_provenance.py:1-180` correctly flags the 3 deltas in
reconciliation_note.md (+0.127413, +0.116935, -0.011286) because they do not
appear in any `results/*.json`. The gate is doing its job — these
are exactly the kind of human-recomputed numbers that should be flagged. The
gate is the right enforcement mechanism even though the *arithmetic* is
correct, because the *interpretation* is wrong (per CRITICAL-1).

The provenance gate working as designed → recommend a "computed-delta
exception register" only if those deltas survive review with their
scale-mixing problem resolved.

## Recommendation summary

A math-attentive referee would respond with the following requested changes,
in priority order:

1. **(CRITICAL)** Rebuild the reconciliation_note OLD-vs-NEW table on the OD
   scale only, using `matched2000_dualscale.json` aggregates for the NEW
   column. Reported deltas will be ≈ 0 across the board, which is the honest
   finding. The "+0.127 degradation" narrative collapses; the
   release-freeze story becomes "the 2000ep matched-budget reproduction
   recovers OD-scale EMD to within seed variance of the 1000ep baseline."

2. **(CRITICAL)** Redraw `cross_model_emd.png/pdf` (and its companion JSON)
   using OD-scale per-seed EMD aggregates (not min-of-trajectory in-loop
   EMD), and ensure the headline reference line is on the same scale. Drop
   the "best EMD" framing.

3. **(CRITICAL)** Add a sentence to the manuscript (and to reconciliation
   note) explaining that the original "0.0015" EMD was a histogram-density
   Wasserstein and is not commensurate with the new raw-sample Wasserstein.
   Cite the v1.0 redefinition decision.

4. **(HIGH)** Re-emit `matched2000_dualscale.json` aggregates with
   `statistics.stdev` (ddof=1). All downstream figures inherit the
   correction; visual difference will be ~12% wider error bars.

5. **(HIGH)** Fix the "OD scale" axis label on `training_convergence_all_models`
   to say something accurate, e.g. "EMD (in-loop training metric, log-return
   scale)".

6. **(HIGH)** Add a "shared critic ≈ 250k params" disclosure to the
   parameter-efficiency narrative; consider a supplementary table that
   reports total-system param counts (generator + critic) alongside
   generator-only counts.

7. **(MEDIUM)** Document the metric conventions (std ddof, Fisher kurtosis,
   biased ACF estimator) in methods_full.md, and either document or correct
   the VAE ELBO per-element-mean weighting.

8. **(LOW)** Provide HPO provenance for lambda_gp=2.16 and n_critic=9.

The core math (loss functions, gradient penalty, EMD on raw samples,
parameter counts, seed-independence) is **correct**. The presentation layer
(reconciliation_note, cross-model figure, ddof choices, axis labels) is what
would cost the paper a desk-revision response.
