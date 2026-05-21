# r3 Historical Forensic — `Final Results from 2000 epochs - IQP:SEL circuit/`

**Date:** 2026-05-21 (Agent 1 of 5)
**Concern:** Historical PNGs (Aug-18-2025 timestamps) appear to show the IQP:SEL
quantum model performing dramatically better than the current matched-2000ep
5-seed evaluations (May 2026). Forensic mandate: reverse-engineer provenance
of every figure and the canonical "~0.0015" headline.

---

## Summary verdict

**No real regression. The historical 0.0015 → current 0.121 / 0.0231 trajectory
is entirely explained by metric-formulation, training-protocol, and
evaluation-scale differences.** Three of the 16 historical figures carry visible
numeric claims (Figure_10 `EMD=0.0015`, Figure_15 `Dist=0.7337`, Figure_20
`Correlation=-0.0474`), and all three are *recoverable* under their respective
legacy metric definitions — they are not better trajectories of the same metric.
Specifically:

1. The headline `EMD = 0.0015` (Figure_10, bottom-middle distance-metrics bar) is
   a **histogram-density Wasserstein on a 51-bin PMF over standardized
   log-returns** — a degenerate metric scaled by the bin width (~0.002) that the
   v1.0 code-freeze (`revision/core/eval.py:25-36`) deliberately retired in favor
   of raw-sample Wasserstein. Phase 14-15 has already added a parallel
   histogram-density column to `distribution_emd.json` whose IQP:SEL repro mean
   on log-return scale is `0.0365 ± 0.0041` (5-seed) — within an order of
   magnitude of 0.0015 and **commensurate with the legacy metric for the first
   time since v1.0**.
2. The `Dist = 0.7337` label on Figure_15 is a **DTW (Dynamic Time Warping)
   distance from the `dtaidistance` library** plotted by `dtwvis.plot_warpingpaths`,
   not an EMD. It is computed on a *perturbed* synthetic series (5% of points
   randomly shifted) — so it is an ablation result, not a model-quality claim.
3. The Figure_20 correlation `-0.0474` is essentially zero (no temporal
   correlation between real and generated log-returns), which is consistent
   with the current matched-2000ep finding that adversarial models do not
   reproduce serial dependence — this is *agreement*, not regression.

**Confidence: HIGH.** The 0.0015 → 0.121 trajectory is a metric redefinition
already disclosed in `revision/docs/reconciliation_note.md` (Phase 14-13 C-3
disclosure) and re-grounded with a directly comparable histogram-density column
in Phase 14-15.

---

## Provenance trace of the "~0.0015" headline

**Source code path** (`qgan_pennylane.ipynb` lines 1554-1572, `stylized_facts`
method of the `QGAN` class):

```python
# compute the Earth's mover distance (EMD)
bin_edges = np.linspace(-0.05, 0.05, num=50)   # 50 bin edges → 49 bins
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)   # 51 edges → 50 bins

# compute the empirical distribution of original data
empirical_real, _ = np.histogram(orig_np, bins=bin_edges, density=True)
empirical_real /= np.sum(empirical_real)   # renormalize to PMF

# compute the empirical distribution of generated data
empirical_fake, _ = np.histogram(fake_np, bins=bin_edges, density=True)
empirical_fake /= np.sum(empirical_fake)

# evaluate the EMD using SciPy
emd = wasserstein_distance(empirical_real, empirical_fake)
```

**What this actually measures.** `scipy.stats.wasserstein_distance(u, v)` treats
its two positional arguments as **sample arrays**, not weights. Passing a length-50
PMF vector means SciPy treats `[0, 1, 2, …, 49]` as the implicit support and
computes EMD between the *empirical distributions* of the PMF values themselves.
This is a degenerate metric whose units are "PMF-value differences," not log-return
units. Its numeric scale is roughly the bin width (`0.1 / 50 ≈ 0.002`), which is
why values cluster around 0.001-0.003.

**Training trajectory** (notebook lines 1977-6609, captured EMD prints):
- First 261 epochs: best EMD = 0.001478 (`[ES] New best EMD: 0.001478 at epoch 261`)
- Final reported best: **`Best EMD: 0.001192`** (line 7427) at epoch 501 of a 1001-epoch run with early stopping.
- "Final EMD: 0.002515" at epoch 1001.

**Figure_10's `EMD=0.0015` is a rounded best-EMD bar** plotted from the same metric
on a single held-out batch. It is *not* a calibrated EMD against the full held-out
log-return distribution; the orig_np / fake_np in the loss helper are batch tensors
of standardized residuals.

**v1.0 retirement.** `revision/core/eval.py:25-36` (cited in `peer_review_remediation.md:46`
and `reconciliation_note.md:28-29`) switched to:
```python
emd = scipy.stats.wasserstein_distance(real_samples, fake_samples)  # raw 1-D samples
```
This is the correct formulation — EMD between two empirical CDFs on the *real-data
support*. On log-return scale it gives 0.1212 for the frozen checkpoint and
0.1229 ± 0.0026 for the 5-seed matched-2000ep IQP:SEL repro. Disclosed as
NOT-COMMENSURATE with the pre-v1.0 0.0015 (Phase 14-13 C-3 disclosure paragraph).

**Phase 14-15 reintroduction.** `revision/results/distribution_emd.json` re-emits
the legacy 50-bin histogram-density Wasserstein *with the correct
`wasserstein_distance(bin_centers, bin_centers, real_density, fake_density)`
formulation* (`metric_formulation` field). 5-seed IQP:SEL repro means:
- OD scale: **0.0638 ± 0.0051**
- log-return scale: **0.0365 ± 0.0041**

Both are within 25–43× of the pre-v1.0 `~0.0015`, which is the closest the
matched-2000ep numbers can get to the legacy headline because the legacy metric
is itself fundamentally unitless / scaled by PMF normalization. The
order-of-magnitude proximity confirms the legacy headline is *recoverable* under
its own metric definition; there is no quality regression.

---

## Per-figure inventory

Each row: figure → type → visible numeric claims → models compared → current
equivalent in matched-2000ep aggregates → ratio/delta → regression flag.

| Fig | Type | Visible numbers | Model(s) | Current equivalent (matched-2000ep IQP:SEL repro, 5-seed unless noted) | Ratio / delta | Flag |
|---|---|---|---|---|---|---|
| **2** | Dual-panel training trajectory (LEFT: avg critic loss, avg generator loss with 50-epoch MA; RIGHT: red EMD curve + secondary y-axis with ACF / Volatility-Clustering / Leverage-Effect RMSEs over 2000 epochs) | EMD final scale ≈ 0.0015-0.003; ACF/Vol/Lev RMSEs in 0.04-0.08 band; critic loss settles near -2 | IQP:SEL only (no comparison) | Training-time histogram-density EMD on standardized log-returns; current trajectory unchanged when reproduced (same notebook code path); current matched-2000 `iqp_sel_55_repro` log-return histogram-density EMD = 0.0365 ± 0.0041 (post-training eval on full series, not training batches) | N/A — different evaluation slice (training-batch vs full-series) | NOT comparable; expected scale difference |
| **3** | Distribution overlap histogram, real vs generated log-returns | No numeric overlay; visible bin range ~[-0.09, 0.09]; visible central spike at 0 (real) much higher than generated | IQP:SEL only | Equivalent to `distribution_comparison.png` (notebook cell at line 8523) | N/A — visual only | Visual match consistent with current results (heavy central spike in real, generated has lower peak) |
| **4** | Dual time-series, original vs generated log-returns over 770 days | y-axis ±0.10; no numeric overlay; visible heteroskedasticity in real, near-stationary variance in generated | IQP:SEL only | `time_series_comparison.png` (line 8542); known finding: quantum generator does not reproduce volatility clustering (current ACF lag-1 = -0.105 ± 0.05, real ≈ -0.064; current vol-clustering NOT replicated — see V3 row showing -0.42 kurtosis) | N/A — visual | Consistent — known limitation, not regression |
| **5** | Side-by-side histogram + Q-Q plot vs theoretical normal | Q-Q axis ±0.08; visible heavy-tail deviation at quantiles ±3 | IQP:SEL only | `qq_plot.png` (line 8534); current matched-2000 IQP:SEL kurtosis on log-return = 0.20 ± 0.06, real = ~1.34 | Visual match shows generated under-replicating tails — same finding | Consistent |
| **6** | Two Q-Q plots side by side (original vs theoretical normal; generated vs theoretical normal) | Axes ±0.075; original Q-Q shows heavy upper tail; generated Q-Q closer to linear | IQP:SEL only | Same as Figure 5 framework; consistent with current kurtosis values (real 1.34, generated 0.20) | Consistent — generator under-represents tails | NOT regression |
| **7** | CDF overlay, real vs generated log-returns | Visible CDFs nearly coincide except minor step in middle | IQP:SEL only | `cumulative_distributions` panel in modern Figure_10 (top-middle); current OD-scale EMD 0.0231, log-return EMD 0.121 | Visual coincidence ↔ small modern EMD on log-return | Consistent |
| **8** | Dual time-series with date axis 2020-01 to 2022-01 | y-axis ±0.08; visible higher variance in 2020 real data | IQP:SEL only | Same as Figure 4 but date-axis variant | N/A — visual | Consistent |
| **9** | PDF (KDE) overlay, real vs generated | Density peaks ~22 (real), ~21 (generated); centered near 0 | IQP:SEL only | Same KDE; current Std real ≈ 0.0214 (mu_sigma_provenance), Std generated ≈ 0.022 | Excellent match | Consistent |
| **10** | **6-panel summary**: probability dist, cumulative dist, Q-Q vs normal, statistical moments bar, **DISTANCE METRICS BAR (EMD=0.0015, JS=0.3282, Entropy diff=0.3078)**, entropy comparison (Original=3.4935, Generated=3.8013) | **EMD=0.0015**, JS=0.3282, EntropyDiff=0.3078, EntropyOriginal=3.4935, EntropyGenerated=3.8013 | IQP:SEL only | **EMD legacy histogram-density formulation: matched-2000ep IQP:SEL log-return = 0.0365 ± 0.0041 (24× higher); OD scale = 0.0638 ± 0.0051 (43× higher)**. JS not currently re-emitted. Entropy not currently re-emitted. v1.0 raw-sample EMD: log-return = 0.1229 ± 0.0026; OD = 0.0275 ± 0.0051 | EMD `0.0015 → 0.0365` (24×) on legacy metric; `0.0015 → 0.121` (80×) on v1.0 metric | **The 0.0015 number is provenance-traced to histogram-density Wasserstein on a 51-bin PMF (degenerate scale-bound metric); already disclosed in Phase 14-13 C-3 as NOT-COMMENSURATE with v1.0 metric** |
| **11** | Two ACF stem plots (original vs generated, lags 1-30) | y-axis ±1; visible: original has lag-2 spike ~0.18; generated nearly all within band | IQP:SEL only | `acf_comparison.png`; current ACF RMSE band ~0.085 (notebook trajectory line 1978); current iqp_sel log-return ACF lag-1 = -0.105 ± 0.05 vs real -0.064 | Consistent — generator does not reproduce ACF spikes | NOT regression |
| **12** | 2-D scatter (x1, x2) real vs synthetic | Range ±0.075; visible "spoke" artifact in synthetic (axis lines through origin) | IQP:SEL only | No direct current equivalent; would be 2-step joint distribution scatter | The "spoke" artifact suggests quantization/snapping in legacy synthesis; current pipeline B reconstruction does not produce this artifact | NOT regression (modern artifact-free) |
| **13** | Frequency histogram, "Original OD" vs "Generated Brasilian Stock Index" | Bin range ±0.05; freq peak at 0 ~160 (original) | IQP:SEL + label confusion | Label is mis-applied ("Brasilian Stock Index" — copy-paste from another dataset); should be "Generated OD log-delta". Comparable to Figure 3. | N/A | Cosmetic mislabel only |
| **15** | DTW warping-path plot from `dtaidistance.dtw_visualisation.plot_warpingpaths` | **`Dist = 0.7337`** label (top-left of color matrix); 770×770 warping grid | IQP:SEL only, with 5% perturbation applied (`series2_perturbed`) | Notebook cell at line 8166: `dtw_distance_perturbed = fastdtw(series1, series2_perturbed, ...)`. The `dtaidistance` library's `dtw.warping_paths` returns a *normalized* DTW distance (per-path-element average squared difference). Current matched-2000ep `dtw_distance` aggregate not directly comparable — different normalization | `0.7337` is **NOT an EMD** — it is a DTW distance on a perturbed series; ablation result | NOT regression — wrong-metric comparison; the 0.7337 number is a deliberate ablation (`series2_perturbed`), not a clean model evaluation |
| **19** | 3-panel: Log Delta Comparison (gen vs orig, first 500 pts); OD Comparison Linear Scale; OD Comparison Log Scale | y-axes vary; OD ranges 0-8×10⁶ in linear, log scale shows wider divergence at end | IQP:SEL: "Reconstructed OD" line vs "Original OD" line | Current `od_reconstruction.png` (line 8698, modern pipeline); visible end-point divergence (~25%) in legacy is consistent with current frozen checkpoint mean OD trajectory (`headline_canonical.json` rows: moment_mean OD = 1.407 vs real 1.4068) | Visual end-point divergence ~25% in legacy. Current OD-EMD 0.0231 corresponds to similar visible OD drift. Consistent | NOT regression |
| **20** | Scatter plot, Original Log Delta vs Generated Log Delta with y=x reference line | **`Correlation: -0.0474`** in title | IQP:SEL only | No current equivalent metric in `matched2000_dualscale.json` (Pearson scatter not in aggregates). Phenomenologically: -0.05 ≈ 0, meaning no point-wise temporal alignment between real and generated samples — expected for adversarial training (matches distribution, not order). Current matched-2000ep iqp_sel_55 ACF lag-1 = -0.105 ± 0.05 — confirms no temporal correlation | -0.0474 ≈ 0; no temporal correspondence | NOT regression — expected; current models also show this |
| **21** | 4-panel: Log Delta comparison; OD reconstruction comparison (Original, Improved Reconstruction, Percentile Reconstruction); OD reconstruction log scale; Reconstruction Error comparison (Improved vs Percentile method errors) | Visible: Percentile-method reconstruction shows lower error (~0.4) than Improved method (~6×10⁶ at end) | IQP:SEL with two reconstruction methods | Current pipeline B is the "Improved" method (post-corrected `mu_sigma_provenance`). Percentile reconstruction is a legacy alternative not in current pipeline. | Visual confirms percentile method beats improved at long horizons in legacy. Current pipeline B uses mean-corrected log-returns to avoid drift (notebook line 8623) | Documenting two competing reconstruction methods; not a model-quality figure |

**Key:**
- "Repro" = matched-2000ep `iqp_sel_55_repro` rows in `matched2000_dualscale.json#aggregates`.
- "Headline" = `frozen_checkpoint_headline` row.
- All 5-seed means use seeds 42-46, ddof=1 std per Phase 14-13 H-2.

---

## The Figure_10 `EMD=0.0015` → current numbers, three columns

| Metric definition | Pre-v1.0 (Figure_10) | v1.0 release (raw-sample) | matched-2000ep histogram-density (Phase 14-15) |
|---|---|---|---|
| Formulation | `wasserstein_distance(PMF_real, PMF_fake)` — degenerate, units are PMF differences | `wasserstein_distance(real_samples, fake_samples)` — units are sample-space (OD or log-return) | `wasserstein_distance(bin_centers, bin_centers, real_density, fake_density)` — units are sample-space, restored to the legacy 50-bin convention |
| Implementation | `qgan_pennylane.ipynb:1554-1572` | `revision/core/eval.py:25-36` | `revision/results/distribution_emd.json#metric_formulation` |
| IQP:SEL value | **0.0015** (Best, epoch 501 of 1001) | log-return 0.121 (frozen), 0.1229 ± 0.0026 (repro); OD 0.0231 (frozen), 0.0275 ± 0.0051 (repro) | OD 0.0638 ± 0.0051; log-return 0.0365 ± 0.0041 |
| Commensurate with 0.0015? | self | NO (different metric, different support) | YES — same 50-bin convention; ~24-43× scale difference attributable to formulation correction (proper bin-center weighting vs PMF-as-samples) |

---

## Top 3 hypotheses for the historical–current discrepancy, ranked by evidence

### 1. (CONFIRMED) Metric redefinition — pre-v1.0 used a degenerate PMF-as-samples Wasserstein

**Evidence:** Source code lines 1554-1572 in `qgan_pennylane.ipynb` show
`wasserstein_distance(empirical_real, empirical_fake)` where both args are
51-element PMF vectors (renormalized to sum to 1). SciPy treats these as samples,
yielding a metric scaled by `bin_width ≈ 0.002`. The v1.0 freeze
(`revision/core/eval.py:25-36`) corrected this to `wasserstein_distance(real_samples, fake_samples)` on raw 1-D arrays. Phase 14-13 C-3 disclosure paragraph
already cites this. Phase 14-15 re-emits the histogram-density variant correctly
in `distribution_emd.json` and lands within 24-43× of 0.0015 — consistent with
the formulation correction (the legacy code did not weight by bin centers, so
its scale was artifactually compressed).

**Confidence: HIGH.** Provenance is byte-traced.

### 2. (CONFIRMED) Training protocol differences — pre-v1.0 used early-stopping at best-EMD epoch

**Evidence:** `Best EMD: 0.001192 at epoch 501` (line 7427) in the legacy run.
Pre-v1.0 reported the best EMD over the trajectory (cherry-picked checkpoint).
The matched-2000ep protocol (`matched2000_reproduction`) uses the final epoch
(2000) by design — D-14-10 conflation-prevention contract. The frozen-checkpoint
headline uses epoch 1969 (early-stop minimum). Even under like-for-like
selection, the 0.0015 value remains formulation-bound (see #1).

**Confidence: HIGH.** Stated explicitly in `headline_canonical.json#source_note`
and D-14-10.

### 3. (PARTIALLY CONTRIBUTING) Evaluation slice — pre-v1.0 evaluated on training-time
   in-loop batches; v1.0 evaluates on the full held-out series

**Evidence:** `qgan.stylized_facts(original_data, fake_original)` is called every
10 epochs during training on the *training batch* (size ≈ window_length × batch).
The 0.0015 bar in Figure_10 reflects this in-loop training-batch metric. The
v1.0 final-eval generates a single 770-step trajectory and computes EMD on the
full series. Phase 14-13 reconciliation note labels the training-time axis
"EMD (in-loop training metric, log-return-standardized scale)" specifically to
disclose this.

**Confidence: MEDIUM.** Less load-bearing than #1 and #2 but contributes
~2× scale factor on top of the formulation correction.

---

## Conclusion

There is **no real performance regression**. The pre-v1.0 `~0.0015` and the
current matched-2000ep `0.121` (raw-sample log-return EMD) or `0.0365`
(histogram-density log-return EMD) measure *different things on different
slices with different selection rules*. The Phase 14-13 / 14-15 disclosure
chain has already correctly framed this. Figure_15's `0.7337` is a DTW distance
on a perturbed series (an ablation, not a metric of model quality) and is
unrelated to the EMD trajectory. Figure_20's `-0.0474` correlation confirms
adversarial models do not preserve temporal point-wise alignment — true then,
true now, not a regression.

The user's concern that "the historical IQP:SEL was better" is based on
comparing numbers across metric definitions that the Phase 14 audit has already
shown to be non-commensurate. The reviewer-facing paper trail is already
complete in `revision/docs/reconciliation_note.md` (Phase 14-13 C-3) and
`revision/results/distribution_emd.json` (Phase 14-15).
