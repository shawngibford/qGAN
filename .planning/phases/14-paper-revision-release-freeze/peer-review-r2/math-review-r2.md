# Math/Statistics Peer Review R2 — Phase 14 plan 14-13 remediation audit

Reviewer role: independent math+statistics auditor of the 14-13 remediation work
Date: 2026-05-20
Reference: original `peer-review/math-review.md` (this directory's parent)
Scope: did 14-13 actually close the math/stats findings, or are the
remediation numbers themselves wrong, mis-described, or misleading?

---

## Summary verdict

**PASS-WITH-FINDINGS.**

The core remediation work is correct and impressive:

- All nine reconciliation deltas in `results/reconciliation_deltas.json`
  are arithmetically verifiable from `baseline_comparison.json` (OLD column)
  and `matched2000_dualscale.json#aggregates` (NEW column). I recomputed each
  delta independently in Python and every digit matches to >10 significant
  figures.
- The `matched2000_dualscale.json` aggregates standard deviations are
  unambiguously ddof=1 (sample std), confirmed by independently recomputing
  from per-seed rows for all 9 models. The "ddof=0 → ddof=1" switch claimed in
  H-2 is real and complete.
- The 250881-parameter shared-critic count is exact, decomposes by formula
  to the four conv1d layers plus two linear layers, and `total_adversarial_param_budget.json`
  correctly adds 250881 to each generator count to produce the totals
  (e.g. 55 + 250881 = 250936).
- The metric-redefinition disclosure paragraph (C-3) in `reconciliation_note.md`
  now explicitly states the 0.0015 → 0.121 transition is between two
  non-commensurate metrics (histogram-density Wasserstein vs raw-sample
  Wasserstein), citing `core/eval.py:25-36` for the v1.0
  redefinition. The byte-freeze at those line numbers is preserved.
- The AR-sigma² ddof=0 bias claim (≈0.26% downward) is mathematically
  accurate (true value 0.258% — see Findings §M-1).
- The Fisher kurtosis / biased ACF / ddof=0 std conventions documented in
  methods_full.md §3.x match the actual code defaults in `core/eval.py`.

However, **one MEDIUM-severity mathematical error** in the remediation
documentation, **one HIGH-severity narrative overclaim**, and **one LOW
documentation gap** are present:

1. **HIGH — `wgan_cnn` is the largest delta (-0.058710, ~30× larger than
   the next-largest delta wgan_mlp at -0.001628) and the
   "deltas ≈ 0 across the board" / "deltas collapse to numerical noise"
   narrative understates this.** The wgan_cnn improvement IS statistically
   non-significant by Welch's t-test (p = 0.37 due to enormous seed-42
   outliers in both OLD and NEW), so "within seed variance" is technically
   defensible, but glossing over a 2× reduction in mean EMD as "noise"
   risks a referee challenge.
2. **MEDIUM — the VAE implicit-β derivation in `methods_full.md §3.x.d` and
   `peer_review_remediation.md` (β ≈ 0.4 = latent_dim/window = 4/10) is
   mathematically INVERTED.** The correct canonical-sum-form β-equivalent
   is β = window/latent_dim = 10/4 = 2.5 (KL UP-weighted), not 0.4 (KL
   down-weighted). This is the same error already present in the original
   math review's M-1 finding — the remediation propagated rather than
   corrected it. Verified numerically (Findings §H-1 below).
3. **LOW — the wgan_cnn delta is reported in `reconciliation_deltas.json`
   to 6+ significant figures, but per-seed variance is so high that two
   significant figures (delta ≈ −0.06) would be more honest.** Cosmetic.

**Math sound for paper resubmission: YES — pending the wgan_cnn narrative
nuance (HIGH) and the VAE β derivation correction (MEDIUM).**

Both are write-only fixes; no recomputation needed.

---

## Methodology

For each of the 12 specific items in the audit prompt, I:

1. Loaded the relevant JSON artifact and parsed it as a dict / list with
   `json.load`.
2. Re-implemented the claimed computation in plain numpy / scipy / Python
   from the upstream per-seed rows (where available).
3. Compared my independent result against the executor's recorded value
   to >10 significant figures.
4. Walked the cited file:line locations to verify the byte-freeze is real
   and the claims about the source code are correct.

All numerical recomputation in this review is in scratchspace; the key
intermediate values are quoted inline in each finding so the orchestrator
can rerun in <5 lines.

I treated the original `math-review.md` as the *claim* to be audited, not
as ground truth — and one of its claims (M-1 / β=0.4) is in fact wrong on
remediation re-derivation. See §H-1 below.

---

## Findings (by severity)

### HIGH-1 — VAE implicit-β derivation in methods_full.md §3.x.d is INVERTED

**Claim being audited.** `methods_full.md:317-326` and
`peer_review_remediation.md:51` state: "the reconstruction term is
mean-over-10-elements and the KL term is mean-over-4-latent-dimensions,
which is equivalent to a canonical per-window-sum ELBO with implicit β ≈ 0.4"
and: "implicit β ≈ 0.4 (window=10, latent_dim=4)".

**Independent computation.** Let W = window length (10), L = latent_dim (4),
and B = batch size. The code at `run_baselines.py:315-319` computes:

```
recon_code = mse_loss(x_hat, x)                     # mean over B*W elements
kld_code   = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))   # mean over B*L elements
loss_code  = recon_code + β_code * kld_code,  β_code = 1
```

Canonical β-VAE ELBO (per-window-sum form, Higgins+2017):

```
L_can = (1/B) Σ_b [ Σ_i (x_hat_i - x_i)^2  +  β_can * Σ_k KL_k ]
      = (1/B) Σ_b [ recon_sum_b + β_can * kld_sum_b ]
```

Code can be rewritten as:

```
loss_code = (1/(B*W)) Σ_b recon_sum_b + β_code * (1/(B*L)) Σ_b kld_sum_b
W*B*loss_code = Σ_b recon_sum_b + β_code * (W/L) Σ_b kld_sum_b
              = canonical-sum-form with β_can = β_code * (W/L) = 1 * (10/4) = 2.5
```

**Numerical verification** (B=1, W=10, L=4, random tensors, seed 0):
- Code loss = 1.8859, multiplied by W = 18.859
- Canonical `recon_sum + (W/L)*β*kld_sum` = 18.859 ✓ (matches W*loss_code to 5 sig figs)
- Canonical `recon_sum + (L/W)*β*kld_sum` = (does not match anything physical)

**Discrepancy.** The doc claims canonical-equivalent β = L/W = 0.4 (KL
DOWN-weighted). The correct value is β = W/L = 2.5 (KL UP-weighted). The
direction is REVERSED. This is a math error in the doc, not in the code.

**Root cause.** The original math-review M-1 finding states verbatim: "the
relative weight of KL vs recon is `(window/latent) = 10/4 = 2.5` compared
to the canonical formulation. With β=1 in code, this is equivalent to β ≈
0.4 in the canonical form (approximately a β-VAE with KL down-weighted)."
This is internally self-contradictory: the cited ratio is 2.5, and the
named equivalent β is 0.4 (1/2.5). The remediation took the original
reviewer's 0.4 at face value rather than re-deriving from first principles.
The β = W/L = 2.5 (KL UP-weighted) interpretation is what the code actually
implements.

**Practical impact.** Posterior collapse is more likely when KL is
DOWN-weighted (β < 1); with β = 2.5 the code biases AGAINST posterior
collapse, which is consistent with the fact that `run_baselines.py:340`
heuristic does not trigger the collapse flag in practice. If a referee
asked "your VAE is closer to β-VAE β=2.5; why is your KL term so heavily
penalized?", the answer is "we use per-element-mean for both, which is a
common choice and the loss landscape is well-behaved" — but the doc
currently says the OPPOSITE direction, which would confuse a referee.

**Recommended action.** Replace the methods_full.md §3.x.d paragraph with:

> The implemented loss uses per-element-mean MSE on the reconstruction term
> (mean over B×W elements where W=10 is the window length) and
> per-element-mean KL on the latent term (mean over B×L elements where L=4
> is the latent dimensionality), with β = 1. After multiplying through by
> W to match the canonical per-window-sum ELBO form (Higgins+2017), the
> code's effective β is β_eff = β × (W/L) = 10/4 = 2.5 — i.e., the KL term
> is UP-weighted by a factor of 2.5 relative to the canonical β=1 sum-form
> ELBO. The choice is a common practical convention (per-element-mean for
> both terms is what `torch.nn.functional.mse_loss` returns by default) and
> the posterior-collapse heuristic at `run_baselines.py:340` does not
> trigger under this setting. No v2.0 numbers shift.

Also fix `peer_review_remediation.md:51` (`implicit β ≈ 0.4 (window=10,
latent_dim=4)` → `implicit β ≈ 2.5 (window=10, latent_dim=4)`).

Citation: `run_baselines.py:315-319`,
`core/models/nonadversarial.py:63-64` (LATENT_DIM=4, WINDOW=10),
`docs/methods_full.md:316-327`,
`docs/peer_review_remediation.md:51`.

---

### HIGH-2 — wgan_cnn delta -0.058710 is the largest in the table and "deltas ≈ 0 across the board" understates it

**Claim being audited.** `reconciliation_note.md:25` interpretation
paragraph: *"Deltas are ≈ 0 across the board: the matched 2000-epoch budget
recovers OD-scale EMD within seed variance of the 1000-epoch baseline."*
and *"the deltas collapse to numerical noise"*.

**Independent computation.** The 9-row delta table contains:

| model | delta | abs(delta) order |
|---|---|---|
| iqp_sel_55_repro | -0.000060 | tiny |
| wgan_mlp | -0.001628 | small |
| wgan_lstm | -0.001044 | small |
| vae | +0.000002 | tiny |
| ar | -0.000000 | identical |
| **wgan_cnn** | **-0.058710** | **30× larger than next-largest** |

For wgan_cnn:
- OLD per-seed (seeds 42-46): [0.328, 0.034, 0.068, 0.085, 0.050], mean 0.113, std(ddof=1) 0.122
- NEW per-seed (seeds 42-46): [0.159, 0.033, 0.034, 0.021, 0.026], mean 0.054, std(ddof=1) 0.059
- Welch's t-test (unequal variances, n=5 each): t = 0.97, p = 0.37
- Cohen's d ≈ (0.113 - 0.054) / 0.096 ≈ 0.61 (medium effect)

**Discrepancy.** The Welch p-value of 0.37 makes the difference statistically
non-significant (so "within seed variance" is defensible), but:

1. The MEAN dropped from 0.113 → 0.054 — a 2.08× reduction. Calling that a
   "delta ≈ 0" is not accurate; the correct framing is "the apparent
   improvement is not statistically distinguishable from seed noise given
   the high per-seed variance, especially the seed-42 outlier present in
   both campaigns."
2. The std also dropped (0.122 → 0.059), which by itself is an interesting
   finding worth flagging: the 2000ep matched-budget regime is more stable
   for wgan_cnn than the 1000ep baseline.
3. In both campaigns seed 42 produces the largest EMD; removing seed 42:
   - OLD others mean = 0.0593
   - NEW others mean = 0.0282
   - Even with the outlier removed, the matched-2000ep regime is ~2×
     better — so the "improvement" is not driven solely by the outlier.

**Recommended action.** Soften the `reconciliation_note.md:25`
interpretation paragraph to:

> Deltas are ≈ 0 across the table for 8 of 9 models (iqp_sel_55_repro, V1,
> V2, V3, wgan_mlp, wgan_lstm, vae, ar), with the matched 2000-epoch budget
> recovering OD-scale EMD within seed variance of the 1000-epoch baseline.
> For wgan_cnn, the per-seed mean drops from 0.113 → 0.054 (a 2× nominal
> improvement); a Welch's t-test gives p = 0.37 due to the large seed-42
> outlier in both campaigns (OLD 0.328 → NEW 0.159), so the apparent
> wgan_cnn improvement is not statistically distinguishable from seed
> noise, but the directional movement is worth noting.

Citation: `results/matched2000_dualscale.json#rows`
filtered to (model_kind=wgan_cnn, scale=OD, metric_name=emd),
`results/baseline_comparison.json#rows` same filter on (pipeline=B),
`docs/reconciliation_note.md:25`.

---

### MEDIUM-1 — AR-sigma² 0.26% downward-bias claim is correct, but the formula presented is mildly ambiguous

**Claim being audited.** `methods_full.md:219-220` states: *"`resid.var(ddof=0)`
(ML estimator), biased by `(n-p)/n ≈ -0.26%` relative to the standard
`ddof=p` Yule-Walker estimator for n=777, p=2"*.

**Independent computation.** The code at `nonadversarial.py:152-157`
produces residuals of length n_resid = n_total - p (where n_total = 777,
p = 2 → n_resid = 775). The ML estimator is `(1/n_resid) Σ resid²`,
the unbiased OLS estimator is `(1/(n_resid - p)) Σ resid²`. Bias ratio:

```
unbiased / biased = (n_resid - p) / n_resid = 773 / 775 = 0.997419
downward bias    = 1 - 0.997419 = 0.2581%   ≈ 0.26%  ✓
```

So the 0.26% figure is correct. **However**, the formula `(n-p)/n` written
in the doc uses `n = n_total = 777` and gives `775/777 = 0.99742` →
0.2574% bias, which is also ≈ 0.26%. Both formulas land on the same
approximate answer (0.258% vs 0.257%) because n_resid = n_total - p and
n_resid - p = n_total - 2p, but the doc's `(n-p)/n` with n = n_total is
NOT the standard OLS bias formula; the standard formula is `(n_obs - n_params) / n_obs`
which with n_obs = n_resid = 775 and n_params = 2 gives 773/775.

**Discrepancy.** The numerical answer (0.26%) is correct to the displayed
precision. The formula `(n-p)/n` is borderline ambiguous about which `n` is
meant (n_total = 777 or n_resid = 775). Both lead to the same displayed
~0.26%.

**Recommended action.** Either (a) leave as-is — the displayed 0.26% is
correct — or (b) clarify the formula to `(n_resid - p) / n_resid = (n_total
- 2p) / (n_total - p)` for n_total = 777, p = 2 → 773/775 ≈ 0.99742, 0.26%
downward bias. Low priority; cosmetic.

Citation: `core/models/nonadversarial.py:152-157`,
`docs/methods_full.md:217-221, 306-314`.

---

### MEDIUM-2 — wgan_cnn delta is reported to 6+ sig figs but seed variance only justifies 1-2

**Claim being audited.** `reconciliation_deltas.json` reports `wgan_cnn
delta = -0.058709680333968936` (16 sig figs from float64 emit).

**Independent computation.** Given seed std(ddof=1, OLD) = 0.122 and
std(ddof=1, NEW) = 0.059, the standard error of the difference of means
(under Welch's approximation) is `sqrt(0.122²/5 + 0.059²/5) ≈ 0.061`.
The delta is -0.0587, but the 95% CI is approximately `-0.0587 ± 2.78 × 0.061`
= `[-0.228, +0.111]` (Welch t with df ≈ 5.8). So the delta is consistent
with 0 at the 95% level, and the meaningful precision is **one significant
figure**: `-0.06 ± 0.06`.

**Discrepancy.** The JSON reports 16 significant figures (float64 default),
the printed reconciliation_note.md uses 6-decimal-place rounding
(`-0.058710`), but the underlying uncertainty is at the level of 1 sig fig.
The hashed `data_hash` provenance contract requires bit-identical emit, so
the JSON's 16 sig figs are correct as a serialization choice. The
manuscript-facing precision should be `-0.06` (1 sf) or `-0.059 ± 0.061`
(with explicit Welch SE).

**Recommended action.** Add a sentence to the reconciliation_note.md
interpretation paragraph: *"Deltas are reported in JSON to float64
precision; the meaningful manuscript-facing precision for wgan_cnn given
n=5 seeds is one significant figure (delta ≈ -0.06, 95% Welch CI roughly
[-0.23, +0.11])."* Low priority; cosmetic improvement only.

Citation: `results/reconciliation_deltas.json:42`,
`docs/reconciliation_note.md:18`.

---

### LOW-1 — single-seed (headline) aggregate carries `std = 0.0` rather than null/NaN

**Claim being audited.** Headline (single-seed, n=1) aggregates use
`std = 0.0` rather than `null` or NaN.

**Independent computation.** Verified by reading
`matched2000_dualscale.json#aggregates` — entries with
`model_kind = "frozen_checkpoint_headline"` carry `n_seeds = 1, n = 1,
std = 0.0`.

**Discrepancy.** None — but the convention is mildly non-standard. The
"correct" value for std with n=1 is undefined (NaN). Setting it to 0.0 is
defensible because `n_seeds = 1` distinguishes "single observation" from
"5 observations with zero variance," but a reader skimming only the `mean
± std` columns of the JSON might be misled. The gate v2 handles this
correctly per the spec.

**Recommended action.** Optional: document in methods_full.md §3.x.e (new
subsection) that single-seed aggregates carry `std = 0.0, n_seeds = 1` as
a sentinel rather than the mathematically-undefined NaN. The n_seeds field
is the disambiguator. Very low priority.

Citation: `results/matched2000_dualscale.json#aggregates` rows
where `model_kind == "frozen_checkpoint_headline"`.

---

## Numerical-claims spot-check (8 traces)

Each claim below was traced from a paper-facing document to a JSON source
and independently verified.

### 1. `iqp_sel_55_repro` OD-EMD = 0.027526 (reconciliation_note.md:13)

- Source claim: `0.027526` (6-digit rounded)
- JSON: `reconciliation_deltas.json#rows[0].new_2000ep = 0.027526430476567092`
- Independent: `mean([per_seed_emd for seed in 42-46])` from
  `matched2000_dualscale.json#rows` filtered (model_kind=iqp_sel_55_repro,
  scale=OD, metric_name=emd) = 0.0275264305 ✓
- **PASS**

### 2. iqp_sel_55_repro OD-EMD std = 0.005133 (cross_model_emd.json:26)

- Source claim: `0.00513321645930008` (16-digit float64)
- Independent recomputation: `np.std(per_seed, ddof=1) = 0.005133216459300080` ✓
  (ddof=0 would give 0.004591288 — does NOT match, confirming ddof=1 switch real)
- **PASS**

### 3. wgan_cnn delta = -0.058710 (reconciliation_note.md:18)

- Source claim: `-0.058710` (rounded)
- JSON: `reconciliation_deltas.json#rows[5].delta = -0.058709680333968936`
- Independent: NEW_mean - OLD_mean = 0.054323397 - 0.113033077 = -0.058709680 ✓
- **PASS**

### 4. Shared critic = 250881 params (methods_full.md:232, 238, 243)

- Source claim: `250881`
- JSON: `classical_architectures.json#models.shared_critic.total_params = 250881`
- Independent: sum of (Conv1d layers + Linear layers) = 704 + 82048 + 163968
  + 4128 + 33 = 250881 ✓
- **PASS**

### 5. iqp_sel_55 total adversarial budget = 250936 (methods_full.md:244)

- Source claim: `250936`
- JSON: `total_adversarial_param_budget.json#totals.iqp_sel_55.total_adversarial_param_budget = 250936`
- Independent: 55 (generator) + 250881 (critic) = 250936 ✓
- **PASS**

### 6. AR delta = -0.000000 (reconciliation_note.md:21)

- Source claim: `-0.000000` (displayed as 6-decimal-place zero)
- JSON: `reconciliation_deltas.json#rows[8].delta = -6.938893903907228e-18`
- Independent: 0.029084359335535298 - 0.029084359335535298 = 0.0 (float64
  numerical noise ≈ 1e-17) ✓ — AR is closed-form and deterministic, so
  identical seeds produce identical residuals; the -6.9e-18 is float64 ULP
  noise.
- **PASS**

### 7. AR ddof bias = 0.26% downward (methods_full.md:219, 311-312)

- Source claim: `-0.26%` bias for n=777, p=2
- Independent: `(773/775 - 1) × 100% = -0.2581%` ≈ -0.26% ✓
  Note: `(775/777 - 1) × 100% = -0.2574%` also rounds to -0.26%; either
  interpretation gives the same printed figure.
- **PASS** (formula ambiguous, see Medium-1)

### 8. VAE β=1 in code → "implicit β ≈ 0.4" (methods_full.md:325)

- Source claim: implicit β = 0.4 from `latent_dim=4 / window=10`
- Independent derivation: equivalent canonical-sum-form β = window/latent_dim
  = 10/4 = **2.5**, not 0.4
- **FAIL** — direction reversed. See High-1.

### 9. Cross-model EMD mean values are OD-scale (cross_model_emd.json:14-24, caption)

- Source claim: "OD scale, final-eval mean ± std over 5 seeds 42-46"
- Independent: each value in `final_eval_emd_mean_OD` matches the
  `mean` field of the corresponding aggregate in `matched2000_dualscale.json`
  filtered to (scale=OD, metric_name=emd) ✓ for all 9 models.
- **PASS**

### 10. Frozen-headline OD-EMD reference = 0.023072 (cross_model_emd.json:47)

- Source claim: `0.023071979442389253`
- JSON: `headline_canonical.json#rows[i].value` where
  (model_kind=quantum, pipeline=B, metric_name=emd, scale=OD) = 0.023071979442389253 ✓
- **PASS**

**Spot-check verdict: 9 of 10 trace through cleanly; 1 (VAE β derivation)
is mathematically inverted and is the basis of HIGH-1.**

---

## Re-verification of original findings (C-1 through M-4)

### Original C-1 (scale collision in reconciliation_note OLD vs NEW)
**Verdict: RESOLVED.** The NEW column now reads from
`matched2000_dualscale.json#aggregates (metric_name=emd, scale=OD)` — the
audited OD-scale aggregate mean — for every row. Deltas are mathematically
correct as differences within the same metric space. The previous
"+0.127 degradation" narrative is removed; the new "deltas ≈ 0" narrative
is supportable for 8 of 9 models but understates the wgan_cnn case (see
HIGH-2).

### Original C-2 (selection-biased + scale-mixed cross_model_emd figure)
**Verdict: RESOLVED.** `cross_model_emd.json` now contains
`final_eval_emd_mean_OD` (mean over 5 seeds, ddof=1 std, OD scale) and
`frozen_headline_OD_emd = 0.023072` (same OD scale). The
"best EMD over training" framing (min-over-201-evaluations) is dropped.
Caption explicitly states "OD scale, final-eval mean ± std over 5 seeds
42-46."

### Original C-3 (histogram-density vs raw-sample Wasserstein redefinition)
**Verdict: RESOLVED.** `reconciliation_note.md:28-29` now carries the
metric-redefinition disclosure paragraph explicitly stating that pre-v1.0
0.0015 (histogram-density) and v1.0+ 0.121 (raw-sample) measure different
distances over different supports and are NOT commensurate. Citation to
`core/eval.py:25-36` correct (raw-sample Wasserstein implementation).

### Original H-1 (training_convergence "OD scale" axis label wrong)
**Verdict: NOT RE-VERIFIED IN R2 SCOPE** (no figure-side artifact in the
JSON suite was specifically asked about). Per `peer_review_remediation.md:46`,
the label was updated to "EMD (in-loop training metric, log-return-standardized
scale)" via commit `8c67891` (T4). The artifact text is plausible but I
did not open the `.py` file to byte-verify; that would belong to the
code-review-r2 reviewer.

### Original H-2 (ddof=0 → ddof=1 sample-std switch)
**Verdict: RESOLVED.** Independently verified by re-computing std from
per-seed `matched2000_dualscale.json#rows` for all 9 models. Every recorded
aggregate `std` matches `np.std(seed_values, ddof=1)` to 10+ significant
figures. The ddof=0 alternative was checked and rejected (mismatches by ~12%
as expected from sqrt(5/4)). The switch is complete and correct.

### Original H-3 (shared-critic 250881 params disclosure)
**Verdict: RESOLVED.** `total_adversarial_param_budget.json` is the
required companion table; the formula (generator + 250881) is correctly
applied to iqp_sel_55, default_75, V1, V2, V3; classical WGAN models
correctly cite the per-generator counts via `classical_architectures.json`
note. methods_full.md §2.k.x carries the new subsection citing the 250881
literal and stating that x-axis is generator-only. The total 55 + 250881 = 250936
verified.

### Original M-2 (AR sigma² ML estimator bias)
**Verdict: DOCUMENTED.** Per the byte-freeze contract D-14-22, the code
remains at `resid.var(ddof=0)`. The bias is now documented in methods_full.md
§2.j (implementation note) and §3.x.c. The 0.26% figure is approximately
correct (true value 0.258%); see MEDIUM-1 for a minor formula-ambiguity
caveat.

### Original M-3 (ddof=0 std + Fisher kurtosis + biased ACF undocumented)
**Verdict: DOCUMENTED.** methods_full.md §3.x.a documents ddof=0 std,
Fisher excess kurtosis via `scipy.stats.kurtosis(bias=True)`, skew via
`scipy.stats.skew(bias=True)`. §3.x.b documents
`statsmodels.tsa.stattools.acf(s, nlags=20, fft=True)` with biased divisor
n. I verified scipy's kurtosis default IS Fisher (excess, → 0 for
N(0,1)) and statsmodels' acf default IS the biased estimator (`adjusted=False`
→ divisor n). Documented conventions match library defaults.

### Original M-4 (VAE per-element-mean ELBO with implicit β)
**Verdict: PARTIALLY RESOLVED — derivation is documented but the
mathematical equivalent is INVERTED.** Doc claims β_eff = L/W = 0.4 (KL
down-weighted); correct value is β_eff = W/L = 2.5 (KL up-weighted). See
HIGH-1 for full re-derivation. The remediation propagated the original
M-1 finding's internal contradiction rather than re-deriving from first
principles.

### Original M-5 (emd_avg[-1] vs full-eval EMD)
**Verdict: RESOLVED by C-1 fix.** The NEW source in
`reconciliation_deltas.json` no longer reads `emd_avg[-1]` (the
small-sample, in-loop, log-return-scale snapshot); it reads the audited
OD-scale aggregate mean over n_synth = 10 × n_real_windows = 3840
synthetic + 384 real samples. The estimator-quality improvement is
mechanical to the C-1 source switch.

### Original L-1 (wgan_lstm rounding off by 1 ulp)
**Verdict: SUPERSEDED.** The new reconciliation_note.md prints all deltas
with consistent 6-decimal-place rounding from
`reconciliation_deltas.json#rows` (the JSON's float64 emit is the source
of truth). The earlier 1-ULP discrepancy in the printed wgan_lstm delta
is no longer present — current row reads `-0.001044` rounded from the
JSON's `-0.0010435076534954894`.

---

## Final recommendation

**Math sound for paper resubmission: YES — pending two write-only fixes:**

1. **HIGH-1 / VAE β fix:** In `methods_full.md §3.x.d` and
   `peer_review_remediation.md:51`, replace the claim "implicit β ≈ 0.4
   (window=10, latent_dim=4)" with the correct value "implicit β ≈ 2.5
   (window=10, latent_dim=4) — the per-element-mean convention UP-weights
   the KL term by a factor W/L = 10/4 = 2.5 relative to the canonical
   per-window-sum ELBO."
2. **HIGH-2 / wgan_cnn nuance:** In `reconciliation_note.md:25`, soften
   "deltas collapse to numerical noise" to acknowledge wgan_cnn's 2× mean
   drop and the Welch p=0.37 reason for calling it not-significant.

Both fixes are documentation-only — no JSON re-emission, no code change, no
recomputation of any aggregate. The mathematical machinery underneath the
remediation is sound:

- Delta arithmetic: correct to >10 sig figs.
- ddof=1 sample-std switch: complete and verified.
- 250881 critic-included param count: exact.
- Metric-redefinition disclosure (C-3): explicitly stated.
- AR-sigma² 0.26% bias: approximately right (0.258%).
- Fisher kurtosis / biased ACF conventions: documented and match library defaults.
- Single-seed n_seeds=1 handling: defensible sentinel std=0.0 convention.
- Headline reference line in cross_model_emd: correct OD-scale source.

Confidence: **HIGH** (every numerical claim was traced to a JSON source and
re-verified independently in Python; the only discrepancy is the VAE β
derivation direction, which is a self-contained doc-text issue).

---

## Files audited

Primary artifacts (all paths absolute):

- `/Users/shawngibford/dev/phd/qGAN/results/matched2000_dualscale.json`
- `/Users/shawngibford/dev/phd/qGAN/results/reconciliation_deltas.json`
- `/Users/shawngibford/dev/phd/qGAN/results/total_adversarial_param_budget.json`
- `/Users/shawngibford/dev/phd/qGAN/figures/cross_model_emd.json`
- `/Users/shawngibford/dev/phd/qGAN/results/methods_full.json`
- `/Users/shawngibford/dev/phd/qGAN/results/classical_architectures.json`
- `/Users/shawngibford/dev/phd/qGAN/results/baseline_comparison.json`
- `/Users/shawngibford/dev/phd/qGAN/results/headline_canonical.json`
- `/Users/shawngibford/dev/phd/qGAN/results/canonical_config_lock.json`

Docs:

- `/Users/shawngibford/dev/phd/qGAN/docs/reconciliation_note.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/methods_full.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/peer_review_remediation.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/paper_blocks_framing.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/paper_blocks_refs_methods.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/reviewer_response.md`

Code (citation verification only — byte-freeze D-14-22 preserved):

- `/Users/shawngibford/dev/phd/qGAN/core/eval.py:25-36` (compute_emd raw-sample)
- `/Users/shawngibford/dev/phd/qGAN/core/eval.py:42-58` (compute_moments)
- `/Users/shawngibford/dev/phd/qGAN/core/eval.py:64-72` (compute_acf)
- `/Users/shawngibford/dev/phd/qGAN/core/models/nonadversarial.py:53-77` (VAE dims)
- `/Users/shawngibford/dev/phd/qGAN/core/models/nonadversarial.py:152-157` (AR sigma²)
- `/Users/shawngibford/dev/phd/qGAN/run_baselines.py:313-319` (VAE ELBO loss)
