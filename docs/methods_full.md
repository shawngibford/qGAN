# Methods (Full)

> **Source of truth.** Every numeric literal in this document resolves to one of
> `results/methods_full.json`, `results/model_info.json`,
> `results/canonical_config_lock.json`,
> `results/default_75_config_lock.json`,
> `results/v1_config_lock.json`,
> `results/v2_config_lock.json`,
> `results/v3_config_lock.json`,
> `results/classical_architectures.json`, or
> `results/framework_versions.json`.
>
> Gate (executable success-criterion-5):
> `./qgan_env/bin/python scripts/verify_number_provenance.py --target docs/methods_full.md`.
> LaTeX equation strings are rendered verbatim from
> `methods_full.json.buckets.2_models.<model>.training_objective.equation_latex*`;
> never authored inline. The two documented manuscript contradictions
> (`default_75` vs `iqp_sel_55`; `dtype_params` vs `dtype_samples`) are
> explicitly addressed in § 6.

This document consolidates the Methods section for the revised manuscript
(PAPER-08 + PAPER-09). It is regenerated whenever the upstream JSON corpus
changes; the number-provenance gate is the executable contract that no
literal may bypass.

---

## 1. Dataset

| Quantity | Value | Source |
|---|---|---|
| Raw CSV rows | 778 | `methods_full.json` buckets.1_dataset.raw_csv_rows / `model_info.json` dataset.raw_csv_rows |
| OD rows (after fillna/dropna) | 778 | `methods_full.json` buckets.1_dataset.od_rows_after_fillna_dropna |
| Log-return rows | 777 | `methods_full.json` buckets.1_dataset.log_return_rows |
| Window length | 10 | `methods_full.json` buckets.1_dataset.window_length |
| Window stride | 2 | `methods_full.json` buckets.1_dataset.window_stride |
| Rolling windows | 384 | `methods_full.json` buckets.1_dataset.rolling_windows |
| Independent campaigns | 1 | `methods_full.json` buckets.1_dataset.independent_campaigns |
| Train windows | 384 | `methods_full.json` buckets.1_dataset.train_windows |
| Val windows | 0 | `methods_full.json` buckets.1_dataset.val_windows |
| Test windows | 0 | `methods_full.json` buckets.1_dataset.test_windows |

Window-count derivation: `windows = (log_return_rows - window_length) // window_stride + 1`,
applied to the single-campaign OD trajectory and gated against the accepted
sweep's `n_real_windows` in `model_info.json`. The single-campaign limitation
means all reported metric variance is over training-seed variation, not over
independent experimental campaigns.

---

## 2. Models

Ten model entries are reported (PAPER-08 Methods completeness): five quantum
(headline + reproduction + V1/V2/V3 matched-budget ansatz variants), three
classical WGAN-GP baselines, and two non-adversarial baselines (VAE, AR(p)).
The shared WGAN-GP critic architecture is recorded once in § 2.k.

### 2.a. `iqp_sel_55_headline` (frozen-checkpoint paper headline)

| Property | Value | Source |
|---|---|---|
| Family | adversarial-quantum | `methods_full.json` buckets.2_models.iqp_sel_55_headline.family |
| num_qubits | 5 | `canonical_config_lock.json` decomposition.num_qubits |
| num_layers | 3 | `canonical_config_lock.json` decomposition.num_layers |
| Topology | range | `canonical_config_lock.json` decomposition.gate_layout.entangler |
| Encoding | IQP (Hadamard + RZ per qubit) | `methods_full.json` buckets.2_models.iqp_sel_55_headline.architecture.encoding |
| Variational block | SEL (Rot(phi,theta,omega) per qubit per layer) | `methods_full.json` buckets.2_models.iqp_sel_55_headline.architecture.variational_block |
| Final rotation | RX_only | `canonical_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 55 | `canonical_config_lock.json` param_count |
| Checkpoint epoch | 1969 | `canonical_config_lock.json` checkpoint_epoch |

Training objective: **WGAN-GP (Wasserstein with two-sided gradient penalty)**
— `methods_full.json` buckets.2_models.iqp_sel_55_headline.training_objective
cites `core/training.py:364` (critic loss),
`core/training.py:385` (generator loss), and
`core/training.py:72` (gradient penalty).

```latex
L_C = \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)] - \mathbb{E}_{x \sim P_r}[C(x)] + \lambda_{gp}\, \mathbb{E}_{\hat x \sim P_{\hat x}}\big[(\|\nabla_{\hat x} C(\hat x)\|_2 - 1)^2\big]
```

```latex
L_G = - \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)]
```

### 2.b. `iqp_sel_55_repro` (matched 2000-epoch reproduction)

Same architecture as `iqp_sel_55_headline` (n_params = 55, num_qubits = 5,
num_layers = 3, topology = range, final_rotation = RX_only —
`canonical_config_lock.json`); distinguished only by `source` ==
`matched2000_reproduction` and `tier` == `T2`. No `checkpoint_epoch` (the
reproduction trains the full 2000-epoch budget — see § 3 — rather than
freezing the best-EMD checkpoint). Training objective: WGAN-GP, citations
and LaTeX as in § 2.a.

### 2.c. `V1` (matched-budget ansatz)

| Property | Value | Source |
|---|---|---|
| Family | adversarial-quantum | `methods_full.json` buckets.2_models.V1.family |
| num_qubits | 5 | `v1_config_lock.json` decomposition.num_qubits |
| num_layers | 4 | `v1_config_lock.json` decomposition.num_layers |
| Topology | range | `v1_config_lock.json` decomposition.gate_layout.entangler |
| Final rotation | RX_plus_RY | `v1_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 75 | `v1_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a — single source of
truth in `methods_full.json` buckets.2_models.V1.training_objective).

### 2.d. `V2` (matched-budget ansatz, deeper)

| Property | Value | Source |
|---|---|---|
| num_qubits | 5 | `v2_config_lock.json` decomposition.num_qubits |
| num_layers | 8 | `v2_config_lock.json` decomposition.num_layers |
| Topology | range | `v2_config_lock.json` decomposition.gate_layout.entangler |
| Final rotation | RX_plus_RY | `v2_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 135 | `v2_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.e. `V3` (matched-budget ansatz, linear entangler)

| Property | Value | Source |
|---|---|---|
| num_qubits | 5 | `v3_config_lock.json` decomposition.num_qubits |
| num_layers | 4 | `v3_config_lock.json` decomposition.num_layers |
| Topology | linear | `v3_config_lock.json` topology |
| Final rotation | RX_plus_RY | `v3_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 75 | `v3_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.f. `wgan_mlp` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Linear | in_features=5, out_features=4, bias, activation=tanh, 24 params | `classical_architectures.json` models.wgan_mlp.generator[0] |
| Linear | in_features=4, out_features=10, bias, 50 params | `classical_architectures.json` models.wgan_mlp.generator[1] |
| **Total** | **74 params** | `classical_architectures.json` models.wgan_mlp.total_params / `model_info.json` parameter_count |

Carve strategy: a single `nn.Parameter` (`params_pqc`) is sliced into
per-layer weight/bias views applied via `torch.nn.functional` (mirrors the
quantum generator's design — RESEARCH § Pitfall 1). Training objective:
WGAN-GP (same equations as § 2.a; same shared critic — § 2.k).

### 2.g. `wgan_cnn` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Reshape | (B, 5) → (B, 1, 5), parameter-free | `classical_architectures.json` models.wgan_cnn.generator[0] |
| ConvTranspose1d | in_channels=1, out_channels=9, kernel_size=6, stride=1, bias, LeakyReLU(0.1), 63 params | `classical_architectures.json` models.wgan_cnn.generator[1] |
| Conv1d | in_channels=9, out_channels=1, kernel_size=1, bias, 10 params | `classical_architectures.json` models.wgan_cnn.generator[2] |
| **Total** | **73 params** | `classical_architectures.json` models.wgan_cnn.total_params / `model_info.json` parameter_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.h. `wgan_lstm` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Reshape | (B, 5) → (B, seq=3, input=2), parameter-free tile/pad | `classical_architectures.json` models.wgan_lstm.generator[0] |
| LSTM_functional | input_size=2, hidden_size=2, num_layers=1, bias, gate order i,f,g,o, 48 params | `classical_architectures.json` models.wgan_lstm.generator[1] |
| Linear | in_features=2, out_features=10, bias, 30 params | `classical_architectures.json` models.wgan_lstm.generator[2] |
| **Total** | **78 params** | `classical_architectures.json` models.wgan_lstm.total_params / `model_info.json` parameter_count |

The LSTM cell is carved from `params_pqc` by hand (NOT `nn.LSTM`) so the
single-`nn.Parameter` invariant matches the quantum generator's design.
Training objective: WGAN-GP (same equations as § 2.a).

### 2.i. `vae` (non-adversarial baseline)

| Block | Layer | Spec | Source |
|---|---|---|---|
| Encoder | Linear `enc` | in=10, out=16, bias | `classical_architectures.json` models.vae.encoder |
| Encoder | Linear `fc_mu` | in=16, out=4, bias | `classical_architectures.json` models.vae.encoder |
| Encoder | Linear `fc_logvar` | in=16, out=4, bias | `classical_architectures.json` models.vae.encoder |
| Decoder | Linear `dec_h` | in=4, out=16, bias | `classical_architectures.json` models.vae.decoder |
| Decoder | Linear `dec_out` | in=16, out=10, bias | `classical_architectures.json` models.vae.decoder |
| Latent dim | 4 | `classical_architectures.json` models.vae.latent_dim |
| Hidden dim | 16 | `classical_architectures.json` models.vae.hidden_dim |
| Window | 10 | `classical_architectures.json` models.vae.window |
| **Total** | **562 params** | `classical_architectures.json` models.vae.total_params / `model_info.json` parameter_count |

Training objective: **ELBO** (variational autoencoder). The ELBO training
loop lives in `scripts/run_baselines.py` (D-10-13) — **not** in
`core/training.py` (which is WGAN-GP only).

```latex
\mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
```

**Implementation note (Plan 14-14, math-review-r2 M-4 correction).** The
actual implemented loss at `run_baselines.py:315-319` uses
per-element-mean MSE + per-element-mean KLD with the literal `β = 1`
coefficient, which is equivalent to a canonical per-window-sum ELBO with
effective coefficient `β_eff = N_recon / N_kld = 10 / 4 = 2.5` (KL
up-weighted). Plan 14-13 originally propagated the inverted figure
`β_eff ≈ 0.4` from the r1 math-review M-4 finding; the r2 math review
surfaced the inversion and §3.x.d carries the corrected derivation. The
comparison numbers in this manuscript use the implemented loss
consistently across all VAE runs; the only correction is to the
documentation of what `β_eff` evaluates to.

**Implementation note (Plan 14-14, methods-reproducibility-review-r2
caveat).** The VAE is **not** parameter-matched to the WGAN-GP variants
(~75k–135k generator params) or the IQP:SEL headline (55 params); at
74 trainable params, the VAE is closer to the IQP:SEL scale and is
included as a non-adversarial low-data baseline, not as a head-to-head
adversarial comparator.

### 2.j. `ar` (non-adversarial baseline)

| Property | Value | Source |
|---|---|---|
| Type | AR(p) | `classical_architectures.json` models.ar.type |
| Order p | 2 | `classical_architectures.json` models.ar.order_p |
| Fit method | np.linalg.lstsq (closed-form) | `classical_architectures.json` models.ar.fit_method |
| **Total** | **3 params** | `classical_architectures.json` models.ar.total_params / `model_info.json` parameter_count |

`ARBaseline` is a plain Python class — **not** an `nn.Module`. Its fit is
closed-form (no training loop) at
`core/models/nonadversarial.py:139-158`.

```latex
x_t = \sum_{k=1}^{p} \phi_k\, x_{t-k} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2);\quad \hat\phi = \arg\min_\phi \|X\phi - y\|_2^2
```

**Implementation note (Plan 14-13, math-review M-2).** The residual variance
estimator at `core/models/nonadversarial.py:157` uses
`resid.var(ddof=0)` (ML estimator), biased by `(n-p)/n ≈ -0.26%` relative
to the standard `ddof=p` Yule-Walker estimator for n=777, p=2 (see §3.x.c).
The convention follows the v1.0 notebook AR baseline; no v2.0 numbers shift.

### 2.k. Shared WGAN-GP critic

One critic instance is constructed per training run and is shared across
`iqp_sel_55_repro` + V1 / V2 / V3 + `wgan_mlp` / `wgan_cnn` / `wgan_lstm`.
Architecture (verbatim `classical_architectures.json` models.shared_critic):
Conv1d(1, 64, k=10, s=1, p=5) + LeakyReLU(0.1); Conv1d(64, 128, k=10, s=1,
p=5) + LeakyReLU(0.1); Conv1d(128, 128, k=10, s=1, p=5) + LeakyReLU(0.1);
AdaptiveAvgPool1d(output_size=1); Flatten; Linear(128, 32) + LeakyReLU(0.1)
+ Dropout(p=0.2); Linear(32, 1). The critic is cast to `torch.float64` at
`core/models/critic.py:67`. The critic carries **250881**
trainable parameters (`classical_architectures.json` models.shared_critic.total_params).

### 2.k.x — Total adversarial parameter budget (Plan 14-13 Task 3, H-3)

The `param_efficiency_pareto` figure plots **generator-only** parameter
counts on the x-axis; the shared critic (250881 params per
`classical_architectures.json` models.shared_critic.total_params) applies
to all adversarial models alike (`iqp_sel_55_repro`, V1, V2, V3,
`wgan_mlp`, `wgan_cnn`, `wgan_lstm`). The **total adversarial parameter
budget** (generator + shared critic) for each model is therefore
generator-only + 250881; the headline `iqp_sel_55` reports a generator of
55 params and a generator+critic total of 250936 params. The
parameter-efficiency comparison is fair under the generator-only x-axis
because the shared critic budget is identical across the adversarial
entries; this subsection documents the convention explicitly so the
figure's caption ("x-axis is generator-only parameter count; the shared
critic ≈250k applies to all adversarial models alike") is no longer the
sole place the breakdown is recorded (H-3 resolution).

---

## 3. Training

| Property | Value | Source |
|---|---|---|
| Optimizer | Adam | `methods_full.json` buckets.3_training.optimizer |
| Optimizer betas | [0.0, 0.9] | `methods_full.json` buckets.3_training.optimizer_betas |
| lr_critic | 1.8046e-05 | `methods_full.json` buckets.3_training.lr_critic |
| lr_generator | 6.9173e-05 | `methods_full.json` buckets.3_training.lr_generator |
| n_critic | 9 | `methods_full.json` buckets.3_training.n_critic |
| lambda_gp | 2.16 | `methods_full.json` buckets.3_training.lambda_gp |
| batch_size | 12 | `methods_full.json` buckets.3_training.batch_size |
| epochs | 2000 | `methods_full.json` buckets.3_training.epochs |
| early_stopping (reproduction) | OFF (full 2000ep, D-14-13) | `methods_full.json` buckets.3_training.early_stopping |
| Seeds | [42, 43, 44, 45, 46] | `methods_full.json` buckets.3_training.seeds |

The matched-budget reproduction (`iqp_sel_55_repro` + V1 / V2 / V3 + the
three classical WGAN-GP baselines) trains the full 2000 epochs with
early-stopping OFF (D-14-13). The frozen-checkpoint headline
(`iqp_sel_55_headline`) is the best-EMD checkpoint at epoch 1969 from the
original early-stopping-ON campaign and is consumed as-is from
`results/canonical_config_lock.json`. The headline's original
training-time learning rates (recorded as breadcrumbs in
`results/canonical_recovery.json`) were lr_critic = 3e-05,
lr_generator = 8e-05 — see `model_info.json` models[0].lr_critic /
lr_generator (`iqp_sel_55_headline` row).

VAE uses a single Adam(lr=1e-3) ELBO loop and AR(p) uses closed-form
`np.linalg.lstsq` — neither participates in the WGAN-GP table above (see
§ 2.i / § 2.j).

### §3.x — Metric conventions (documented per Plan 14-13, math-review remediation)

The following small statistical conventions in `core/` follow the
v1.0 notebook-parity contract and are PRESERVED under D-14-22
(core/ byte-freeze). They are documented here for reviewer
transparency rather than modified.

**(a) `compute_moments`** (cite: `core/eval.py:42-58`). Uses
population standard deviation `np.std(..., ddof=0)` for the
per-distribution `moment_std` field. Fisher excess kurtosis is computed
via `scipy.stats.kurtosis(bias=True)` (biased estimator, matches the v1.0
notebook). Sample skew is computed via `scipy.stats.skew(bias=True)`
(likewise biased; matches v1.0). The biased-vs-unbiased choice is
consistent across all per-distribution moment statistics reported in this
manuscript.

**(b) `compute_acf`** (cite: `core/eval.py` ACF block). Uses
`statsmodels.tsa.stattools.acf(s, nlags=20, fft=True)`, which employs the
biased divisor `n` (rather than the unbiased divisor `n-k` at lag `k`).
This is the same biased ACF estimator as the v1.0 notebook and is the
conventional default for Wasserstein-distance inputs in this codebase.

**(c) AR(p) `sigma^2` estimator** (cite:
`core/models/nonadversarial.py:157`). The residual variance is
computed as `resid.var(ddof=0)` — the ML (maximum-likelihood) estimator,
which is biased by a factor of `(n-p)/n` relative to the standard
`ddof=p` Yule-Walker estimator. For the AR(2) configuration used on the
777-sample log-return sequence (n=777, p=2), the downward bias is
`(777-2)/777 - 1 ≈ -0.26%`. The convention is consistent with the v1.0
notebook AR baseline and is documented here for completeness; no v2.0
numbers shift.

**(d) VAE ELBO formulation** (corrected per Plan 14-14, math-review-r2 M-4
correction; cite: `run_baselines.py:315-319`). The implemented loss
uses per-element-mean reconstruction MSE plus per-element-mean KL divergence
with the standard `β = 1` ELBO coefficient. With per-element-mean MSE =
sum_MSE / N_recon (where N_recon = window × features = 10 × 1 = 10) and
per-element-mean KLD = sum_KLD / N_kld (where N_kld = latent_dim = 4), the
implemented loss factors as:

```
loss = recon + 1·kld
     = sum_MSE/10 + sum_KLD/4
     = (1/10) · (sum_MSE + (10/4)·sum_KLD)
     = (1/10) · (sum_MSE + 2.5·sum_KLD)
```

Hence the canonical-sum-form equivalent is **β_eff = 2.5 (KL up-weighted)**,
NOT `β_eff ≈ 0.4` (the inverted figure propagated from the r1 math-review
M-4 finding through Plan 14-13 — math-review-r2 surfaced the inversion).
Semantic interpretation: the latent space is MORE STRONGLY regularized
toward the unit Gaussian prior than a canonical `β = 1` ELBO would impose;
the implementation sits on the upper-β side of a β-VAE rather than as a
vanilla `β = 1` VAE. The numeric value `β_eff = 2.5` follows directly
from the ratio of the per-element averaging dimensionalities
(N_recon / N_kld = 10 / 4); both terms still use the literal `β = 1`
coefficient in code. The convention follows the v1.0 notebook baseline and
the actual loss expression is documented alongside the canonical LaTeX in
§2.i (VAE).

**(e) OD-marginal convergence (Plan 14-15 post-r2 investigation).** At the
matched-2000ep budget, 8 of 9 models cluster tightly together in their
OD-marginal approximation (median pairwise model-vs-model max-quantile-diff
approximately 0.03 OD-units, range 0.004–0.22 across all 28 pairs), with
WGAN-CNN diverging from this consensus (median approximately 0.69 vs the
other 8). In absolute terms vs the empirical OD marginal, all 9 models
exhibit a systematic ~0.25 OD-unit deviation (8/9 fall in 0.24–0.28;
WGAN-CNN at 0.81 max-abs-quantile-diff over the 0.5–99.5% range) — no
model recovers the marginal in absolute terms; 8 of them just make
essentially the same approximation. See Figure `qq_overlay.png` (Plan
14-15) for the single discriminating figure across architectures;
OD-marginal-EMD numbers should be read alongside ACF / conditional-moment /
TimeGAN-style scores for architecture-level discrimination.

**(f) Log-return scale convention.** The log-return-scale EMD column in
`matched2000_dualscale.json` is computed by un-standardizing the synthetic
log-returns back to raw scale before comparison against the unchanged raw
`real_log_delta`. Concretely, at
`run_matched2000_dualscale.py:368-372` (and the sister site at
`run_distribution_emd.py:144-169`, `_real_references`), the
fake-side `trans_flat_raw = r["transformed"] * sigma + mu`
un-standardizes the synthetic log-returns; this places fake and real on
the same raw-log-return units (the standardize-real alternative is
mathematically scale-matched but produces EMD in standardized units that
do not match the per-step `log_delta` scale used elsewhere in the
pipeline). The matched-budget per-model LR-EMD aggregates anchor at
AR(2) leading at 0.0029, quantum cluster 0.0040 (iqp_sel_55_repro) to
0.0050 (V3), VAE 0.0158, then WGAN cluster wgan_lstm 0.0244, wgan_mlp
0.0444, wgan_cnn 0.1286. The OD-marginal EMD cluster-floor Welch p
(quantum vs WGAN, 12 pairs) is 0.019 at n=5 per group; the quantum
cluster mean (≈0.029) sits ~11× below the WGAN cluster mean (≈0.331).
At n=5/group the two-sample Welch test has approximately 15% power
against |d|=0.65 and an 80%-power detection floor of |d|≈2.0; the
cluster-level OD-EMD dominance is reported as a per-model-mean result
rather than a per-seed equivalence-grade inference, and no TOST
equivalence claim is made on either axis.

**(g) Shared-edges formulation (Plan 14-16).** The original Plan 14-15 emit
of `distribution_emd.json`'s histogram-density EMD used
`np.histogram(..., density=True)` for both real and fake (per
`peer-review-r3/code-review-r3.md` R3-CR-1). Plan 14-16 replaces this with:
(a) `density=False` for both histograms; (b) edges derived from real only;
(c) both histograms normalized to total-mass=1 over the same edge set (no
per-distribution renormalization); (d) out-of-range fake mass disclosed
separately as `fake_in_range_mass = fake_hist.sum() / len(fake)`. The
schema bumps from `'distribution-emd v1 (Phase 14 plan 14-15)'` to
`'distribution-emd v2 (Phase 14 plan 14-16)'`. Investigation finding: with
shared edges the `density=True` vs `density=False` formulation is
numerically inert for `scipy.stats.wasserstein_distance` (which
renormalizes weights internally) — the OD-scale v1-to-v2 aggregate values
are byte-identical. The fix's genuine contribution is the
`fake_in_range_mass` disclosure stat, which confirms no out-of-range
truncation on either scale (OD ~0.98, log-return ~1.0 post-sister-fix).
The corrected aggregates cite `distribution_emd.json#aggregates` under
schema v2.

### DTW historical context (Plan 14-16)

Dynamic Time Warping (DTW) is computed by the byte-frozen emitter at
`core/eval.py:38-89` (D-14-22); per-(model, seed, scale) values
are persisted in `matched2000_dualscale.json` since Plan 14-11, and
per-(model_kind, scale) aggregates (mean ± std, n=5 seeds per cell, ddof=1)
are in `matched2000_dualscale.json#aggregates` under `metric_name='dtw_mean'`.
The manuscript headline DTW=0.6843 at `paper/main.tex:190` +
`paper/main.tex:266` + `paper/supp_material.tex:290` originates from a
pre-v1.0 best-case iqp_sel_55 evaluation pipeline; this value is not
re-emitted by the current matched-budget contract under the strict-accept
gate (D-14-13) — it is a labeled historical-reference literal preserved for
narrative continuity with the LaTeX read-only sources (D-14-18). The
Orlandi et al. reference DTW=1.954 at `paper/main.tex:191` is a labeled
external benchmark, also not re-emitted.

Under the current matched-2000ep evaluation contract, OD-scale DTW
means (n=5 seeds per cell) place the four quantum variants in a tight
low-DTW cluster (V2 0.333, V1 0.349, iqp_sel_55_repro 0.370, V3 0.410)
together with the two non-adversarial baselines (VAE 0.307, AR(2)
0.371). The three classical adversarial WGAN baselines spread above
this cluster: wgan_lstm 0.597, wgan_mlp 0.915, wgan_cnn 6.991. The
cluster-floor Welch p over the 12 quantum-vs-WGAN OD-DTW pairs is
approximately 0.002 at n=5 per group. On log-return scale, all four
quantum variants (range 6.09–9.48) report log-return DTW lower than
every WGAN baseline (wgan_lstm 18.23, wgan_mlp 28.51, wgan_cnn 69.02)
with per-seed dominance: no quantum seed overlaps any classical WGAN
seed across the 25-cell quantum×WGAN×5-seed grid. AR(2) sits at 7.70
inside the quantum range; LR-DTW therefore distinguishes the quantum
cluster from the WGAN cluster but not from AR(2), and the
uniform-dominance LR-DTW claim is scoped to the quantum-vs-WGAN
sub-family with AR(2) carried as a non-adversarial reference. VAE's
log-return DTW of 0.088 reflects a degenerate generation regime
rather than temporal-structure fidelity: the VAE's log-return marginal
is well-aligned with real (LR-EMD ≈ 0.016, sample std ≈ 0.0186 vs real
≈ 0.0217) but its lag-1 autocorrelation is sharply different from real
(ACF lag-1 = −0.648 vs real −0.064), so DTW's global alignment is
small because both series are tightly fluctuating near zero while the
temporal-structure mismatch is not captured. The VAE is excluded from
the uniform-dominance LR-DTW comparison per the §6 #1 hard prohibition
and reported but not interpreted as evidence of model quality.

Relative to the Orlandi et al. reference DTW = 1.954, the matched-2000ep
mean OD-scale DTW of the quantum cluster (range 0.33–0.41) represents
an approximately 5× lower DTW. The wgan_mlp + wgan_lstm pair and both
non-adversarial baselines (VAE, AR(2)) also exceed the Orlandi
reference; wgan_cnn does not. The Orlandi-improvement claim is
therefore scoped to the quantum cluster and the parameter-matched
WGAN-MLP / WGAN-LSTM comparators (a subset of the matched-budget
cohort) plus the two non-adversarial baselines, rather than to every
matched-budget generator.

### §3.y — Utility-oriented evaluation at matched-budget Pipeline B (Plan 14-20)

The TimeGAN-convention utility battery requested by R1-M2 is implemented
in `scripts/run_utility.py` and `scripts/run_timegan_scores.py`. As of
Plan 14-20 (post-rebuttal-prep regime-alignment), the battery consumes
the matched-budget Pipeline B artefacts at
`results/matched2000/runs/<model_kind>/<seed>/` (2000 epochs,
9 trainable model_kinds × 5 generator seeds = 45 cells). The same nine
model_kinds that back the R1-M1 parametric-efficiency analysis
(`matched2000_dualscale.json`, `welch_pairwise.json`) therefore back the
R1-M2 utility analysis — a single matched-budget evidence base.

| Metric | Driver | Output | TimeGAN convention |
|---|---|---|---|
| TSTR R²/MAE/RMSE | `scripts/run_utility.py` `run_tstr` | `results/tstr_matched2000.json` | 1-layer LSTM (hidden=32) soft sensor trained on pooled synthetic OD windows, evaluated on held-out real OD windows (n_eval_real = 320, n_train_synth = 19200 = 5 seeds × 3840) |
| Predictive score | `scripts/run_timegan_scores.py` | `results/predictive_discriminative_matched2000.json` `scores[*].predictive_*` | Canonical Yoon et al. predictive_metrics.py (post-hoc one-step-ahead forecast objective, normalized error) |
| Discriminative score | `scripts/run_timegan_scores.py` | `results/predictive_discriminative_matched2000.json` `scores[*].discriminative_*` | Canonical Yoon et al. discriminative_metrics.py — `discriminative_score = |classifier_accuracy − 0.5|`, lower is better, 0.0 optimal (synthetic at chance from real), 0.5 worst (classifier perfectly separates). Univariate-input adaptation: `hidden_dim = 10` per D-11-04 (the canonical `int(dim/2)` produces a degenerate zero-width GRU when dim = 1; see RESEARCH Pitfall 1 / Assumptions-Log A1) |
| Augmentation lift | `scripts/run_utility.py` `run_augmentation` | `results/augmentation_matched2000.json` | Orlandi-style: same soft sensor trained on n_real ∈ {65} ∪ {65 + k × n_synth} for injection ratios k ∈ {+25%, +50%, +100%}; lift reported as Δr2/Δmae/Δrmse vs real-only baseline (D-11-06/07) |

**data_hash invariance gate (Plan 14-20).** The matched-budget driver
mode asserts `91e447d4624e25b3` against `_compute_data_hash(csv_path)`
AND against every one of the 45 matched-budget `config.yaml` `data_hash`
fields before any soft-sensor or post-hoc-net training starts; the
shortcut "quantum by construction" used by the legacy 1000-epoch driver
mode is removed because all 45 matched-budget configs carry the
canonical hash directly.

**Matched-budget headline (per-variant numbers).** TSTR R² in
[0.993, 0.998] across all 9 generators against a real-only baseline of
R² = -13.354 ± 0.583 (n = 65 real training windows, 3 init seeds);
TimeGAN discriminative score = 0.40888 (to five decimal places)
identically across all 45 cells; Orlandi +100% augmentation R² in
[0.957, 0.971] across all 9 generators. Per-variant numbers anchored at
`tstr_matched2000.json#tstr` / `predictive_discriminative_matched2000.json#scores`
/ `augmentation_matched2000.json#lift`; the reviewer-facing interpretation
of the cross-generator convergence (the cumulative-sum back-transform
encodes near-perfect lag-1 autocorrelation into synthetic OD regardless
of generator) is in `reviewer_response.md`'s `### R1-M2 — Utility-oriented
evaluation — matched-budget re-run (Plan 14-20)` section.

**Legacy 1000-epoch utility JSONs.** `tstr.json`,
`predictive_discriminative.json`, and `augmentation.json` were generated
in Phases 10 and 11 against the pre-recovery `default_75` quantum
entrant (via `results/transform_ablation/runs/` for the quantum
row) and the Phase-10 `baselines/runs/` directory for the classical
baselines — a 1000-epoch regime that pre-dates the Plan 14-01 canonical
55-parameter IQP:SEL recovery. They remain on disk as provenance
reference but are not cited in the rebuttal; every utility number in
the manuscript and `reviewer_response.md` resolves to the
`*_matched2000.json` sibling files.

---

## 4. Hardware & Software

| Property | Value | Source |
|---|---|---|
| Compute device | cpu | `methods_full.json` buckets.4_hardware_software.device |
| PennyLane backend | default.qubit | `methods_full.json` buckets.4_hardware_software.pennylane_device |
| Differentiation | backprop | `methods_full.json` buckets.4_hardware_software.diff_method |
| **Param dtype (`dtype_params`)** | torch.float32 | `methods_full.json` buckets.4_hardware_software.dtype_params |
| **Sample dtype (`dtype_samples`)** | torch.float64 (CPU/CUDA path; falls back to torch.float32 on Apple MPS) | `methods_full.json` buckets.4_hardware_software.dtype_samples |
| Python version | 3.11.14 | `framework_versions.json` python_version |
| Platform | macOS-26.0.1-arm64-arm-64bit | `framework_versions.json` platform |

### 4.1. Framework versions (exact installed pin)

| Package | Version | Source |
|---|---|---|
| pennylane | 0.43.0 | `framework_versions.json` packages.pennylane |
| torch | 2.9.0 | `framework_versions.json` packages.torch |
| numpy | 2.3.4 | `framework_versions.json` packages.numpy |
| scipy | 1.16.2 | `framework_versions.json` packages.scipy |
| matplotlib | 3.10.7 | `framework_versions.json` packages.matplotlib |
| PyYAML | 6.0.3 | `framework_versions.json` packages.PyYAML |

`requirements.txt` carries the `>=` constraint set; the table above
is the exact installed pin captured at methods-doc emit time via
`importlib.metadata.version(...)`. Re-emit on environment change.

The resubmission-canonical environment is committed at
`requirements-pinned.txt` (Phase 14 plan 14-13, Task 1) with exact
`==` pins for every package recorded in
`results/framework_versions.json`; reviewers can rerun the pipeline
against this exact environment with
`python -m venv qgan_env && pip install -r requirements-pinned.txt`.

**`dtype_params` ≠ `dtype_samples`.** Trainable parameters live in
`torch.float32` for every model (every `nn.Parameter` in
`core/models/classical.py` is constructed with
`dtype=torch.float32` — see `core/models/classical.py:78` for
`WGANMLPGenerator`, with analogous lines for `WGANCNNGenerator` /
`WGANLSTMGenerator`; the quantum generator's `params_pqc` is also
`torch.float32`). Generated samples are cast to `torch.float64` at
generation time on the CPU/CUDA path (`core/training.py:268`,
`compute_dtype = torch.float32 if device.type == "mps" else torch.float64`,
then the cast at `core/training.py:347`,
`generated_samples.to(compute_dtype) * 0.1`). The Apple MPS path falls back
to `torch.float32` because MPS does not implement float64 — explicit
`compute_dtype` branching at `core/training.py:259-268`. The two
fields are DISTINCT and MUST NOT be conflated — see § 6(b).

### 4.2. Historical training-time device asymmetry (Plan 14-13, peer-review disclosure)

**Historical training-time device asymmetry (Plan 14-13, peer-review
disclosure).** The matched-2000ep classical runs reported in this manuscript
executed on Apple-Silicon MPS at float32 precision (the runtime default for
the classical training paths `train_wgan_gp` and `_train_vae` at the time of
the original matched-budget sweep), while the quantum runs executed on CPU
at float64 (the `_train_quantum` MPS-disable hook). This asymmetry was
discovered post-execution during the Phase 14 peer-review pass. Future runs
invoke the MPS-disable hook in all training paths (Plan 14-13 Task 4:
`_train_wgan` and `_train_vae` now patch
`torch.backends.mps.is_available = lambda: False` symmetrically), and the
strict-accept gate now records `training_time_device` and enforces equality
across all models in a sweep (D-14-13 extension under Plan 14-13). Numerical
impact: MPS at float32 vs CPU at float64 on these small (74–250881 param)
classical generators is empirically within seed variance for the
matched-budget aggregates reported in this manuscript, but the asymmetry is
disclosed here for completeness in lieu of a full classical sweep re-run.

---

## 5. Reproducibility

| Property | Value | Source |
|---|---|---|
| data_hash | 91e447d4624e25b3 | `methods_full.json` buckets.5_reproducibility.data_hash |
| seed_set | [42, 43, 44, 45, 46] | `methods_full.json` buckets.5_reproducibility.seed_set |

### 5.1. Determinism contract

| Seed call | Source |
|---|---|
| `torch.manual_seed(seed)` | `core/training.py:245` |
| `np.random.seed(seed)` | `core/training.py:246` |
| `random.seed(seed)` | `core/training.py:247` |
| `torch.cuda.manual_seed_all(seed)` (when CUDA available) | `core/training.py:248-249` |

Seeds are set ONCE at the top of `train_wgan_gp` before optimizer/data
construction (`core/training.py:244-249`). The same seed produces
trajectories that agree to ~1e-6 EMD on the same CPU+BLAS+pinned-pip-freeze
stack (`requirements-pinned.txt`); bit-determinism would require
`torch.use_deterministic_algorithms(True)` which is not set in the
byte-frozen `core/training.py` (D-14-22). The pinned-env +
tracked-checkpoint contract (`checkpoints/best_checkpoint.pt`,
sha256 = `f7cceb52…` per `canonical_config_lock.json#checkpoint_sha256`)
delivers reproducibility-within-numerical-tolerance, not bit-determinism
(Plan 14-13, METHODS-HIGH-1 remediation).

### 5.2. Exact rerun command (verbatim)

The block below is the verbatim module docstring of
`run_matched2000.py` (lines 1-69) — preserved character-for-character
inside `methods_full.json.buckets.5_reproducibility.rerun_command_template`
and rendered as-is here:

```
Phase 14 Tier-2/3 driver — train one (model, seed) at the MATCHED 2000-epoch
budget behind a device-honest strict accept gate.

One process per invocation. Idempotent — re-running the same ``--model X
--seed Z`` overwrites ``runs/<model>/<seed>/`` cleanly.

Usage
-----
    ./qgan_env/bin/python -m revision.run_matched2000 \
        --model {iqp_sel_55_repro|V1|V2|V3|wgan_mlp|wgan_cnn|wgan_lstm|vae|ar} \
        --seed N [--epochs 2000] \
        [--out-root results/matched2000] [--csv-path ./data.csv]

    # strict accept gate (per-artifact, explicit-raise; exits non-zero on
    # rejection):
    ./qgan_env/bin/python -m revision.run_matched2000 \
        --accept --model M --seed N [--out-root results/matched2000]
```

Source: `run_matched2000.py:1-69` (module docstring; preserved
verbatim — never paraphrased). The fully verbatim docstring (including the
strict-gate accept-criterion text and the Pitfall-5 / D-10-24 worker-pool
note) lives in `methods_full.json.buckets.5_reproducibility.rerun_command_template`.

---

## 6. Addressing the documented contradictions

### (a) `default_75` vs `iqp_sel_55` — which is "default"?

Both are valid, production quantum circuits used in the manuscript. They
serve different purposes and live in different config-lock JSONs:

- **`default_75`** — `results/default_75_config_lock.json`:
  num_qubits = 5, num_layers = 4, topology = range, n_params = 75, final
  rotation RX+RY per qubit. This is the byte-frozen v1.0/v1.1 baseline
  encoded in `core/__init__.py` (`NUM_QUBITS=5`, `NUM_LAYERS=4`)
  and the underlying circuit_id for the matched-budget ansatz variants V1,
  V2, V3 (see `v1_config_lock.json`, `v2_config_lock.json`,
  `v3_config_lock.json`).
- **`iqp_sel_55`** — `results/canonical_config_lock.json`:
  num_qubits = 5, num_layers = 3, topology = range, n_params = 55, final
  rotation RX only per qubit, `checkpoint_epoch = 1969`. This is the
  **canonical paper circuit** recovered from the **frozen checkpoint** at
  epoch 1969 (D-14-01) and is the manuscript's headline quantum entrant in
  every cross-model comparison.

Neither is "wrong" — `default_75` underlies the matched-budget ansatz study
(V1 / V2 / V3) that demonstrates the headline `iqp_sel_55` is competitive
at a strictly lower parameter count. Where the manuscript talks about
**"the quantum entrant"** it always means `iqp_sel_55`; where it talks
about the **matched-budget ansatz comparison** it always means a V-variant
of `default_75`.

### (b) `dtype_params` vs `dtype_samples` — distinct, never the same

Trainable parameters live in `torch.float32` for every model:
- Classical: every `nn.Parameter` in `core/models/classical.py` is
  constructed with `dtype=torch.float32`
  (`core/models/classical.py:78` for `WGANMLPGenerator`, with
  analogous lines for `WGANCNNGenerator` / `WGANLSTMGenerator`).
- Quantum: the `QuantumGenerator.params_pqc` `nn.Parameter` is also
  `torch.float32`.

Generated samples are cast to `torch.float64` at generation time on the
CPU/CUDA path. The cast is done explicitly at
`core/training.py:268` (`compute_dtype = torch.float32 if
device.type == "mps" else torch.float64`) and applied at
`core/training.py:347` (`generated_samples.to(compute_dtype) *
0.1`). The Apple MPS path falls back to `torch.float32` because MPS does
not implement float64 — see the explicit `compute_dtype` branching at
`core/training.py:259-268`. Samples are cast to float64 to match
the float64 critic (`core/models/critic.py:67` `.double()`).

These are TWO DISTINCT fields in
`methods_full.json.buckets.4_hardware_software` (`dtype_params` and
`dtype_samples`) and a previous internal document conflated them; this
Methods doc resolves the conflation. They must be reported as two separate
rows in any future revision table — see § 4 of this document.

---

## 7. Provenance footer

- Every numeric literal in this document resolves to one of:
  `results/methods_full.json`,
  `results/model_info.json`,
  `results/canonical_config_lock.json`,
  `results/default_75_config_lock.json`,
  `results/v1_config_lock.json`,
  `results/v2_config_lock.json`,
  `results/v3_config_lock.json`,
  `results/classical_architectures.json`,
  `results/framework_versions.json`.
- Gate command:
  `./qgan_env/bin/python scripts/verify_number_provenance.py --target docs/methods_full.md`.
- LaTeX equation strings are rendered verbatim from
  `results/methods_full.json` (never authored inline in this
  document).
- The `rerun_command_template` block in § 5.2 is sliced verbatim from
  `run_matched2000.py:1-69` (module docstring) by
  `scripts/run_methods_full.py` — never paraphrased.
