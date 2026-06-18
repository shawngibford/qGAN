# Matched-2000ep dual-scale comparison — copy-paste table

Rendered ENTIRELY from `results/matched2000_dualscale.json` by `run_figure_suite.py` (`render_matched2000_dualscale_comparison_table`). Zero hand-typed numbers; every literal traces to that single JSON source of truth and passes `verify_number_provenance.py` unmodified.

Quantum entrants (IQP:SEL 55p + ansatz V1/V2/V3) vs classical baselines (WGAN-GP × 3, VAE, AR) at the matched 2000-epoch budget. The frozen-checkpoint headline is reported as a DISTINCT row (source = `frozen_checkpoint_epoch_1969`) and is never merged into the iqp_sel_55_repro reproduction row (D-14-10).

Aggregates are mean over the 5 matched-2000ep seeds (42-46) for the 9 sweep models; the frozen headline aggregate is a single-generation value (no seed variance).

## OD-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.0282 ± 0.0043 (n=5) | 1.4039 ± 0.0082 (n=5) | 0.8783 ± 0.0117 (n=5) | 1.3705 ± 0.0173 (n=5) | 0.8431 ± 0.0504 (n=5) |
| Quantum V1 (75p) | 0.0283 ± 0.0053 (n=5) | 1.4038 ± 0.0073 (n=5) | 0.8787 ± 0.0103 (n=5) | 1.3721 ± 0.0247 (n=5) | 0.8445 ± 0.0813 (n=5) |
| Quantum V2 (135p) | 0.0279 ± 0.0051 (n=5) | 1.4040 ± 0.0089 (n=5) | 0.8782 ± 0.0130 (n=5) | 1.3723 ± 0.0228 (n=5) | 0.8468 ± 0.0788 (n=5) |
| Quantum V3 (75p) | 0.0308 ± 0.0054 (n=5) | 1.4046 ± 0.0083 (n=5) | 0.8806 ± 0.0133 (n=5) | 1.3838 ± 0.0223 (n=5) | 0.9199 ± 0.0816 (n=5) |
| WGAN-GP (MLP) | 0.0769 ± 0.0292 (n=5) | 1.3661 ± 0.0646 (n=5) | 0.8772 ± 0.0483 (n=5) | 1.4874 ± 0.0920 (n=5) | 1.6213 ± 0.6399 (n=5) |
| WGAN-GP (CNN) | 0.7989 ± 1.4736 (n=5) | 2.0336 ± 1.5713 (n=5) | 1.7181 ± 1.8313 (n=5) | 1.9332 ± 0.6951 (n=5) | 5.1967 ± 5.9129 (n=5) |
| WGAN-GP (LSTM) | 0.1177 ± 0.0705 (n=5) | 1.3168 ± 0.1118 (n=5) | 0.8268 ± 0.0737 (n=5) | 1.3807 ± 0.0219 (n=5) | 0.9014 ± 0.0965 (n=5) |
| VAE | 0.0257 ± 0.0072 (n=5) | 1.3854 ± 0.0081 (n=5) | 0.8641 ± 0.0114 (n=5) | 1.3600 ± 0.0206 (n=5) | 0.7796 ± 0.0632 (n=5) |
| AR(p) | 0.0291 ± 0.0046 (n=5) | 1.4038 ± 0.0083 (n=5) | 0.8788 ± 0.0109 (n=5) | 1.3740 ± 0.0206 (n=5) | 0.8748 ± 0.0677 (n=5) |
| FROZEN headline (epoch 1969) | 0.0224 ± 0.0000 (n=1) | 1.4078 ± 0.0000 (n=1) | 0.8866 ± 0.0000 (n=1) | 1.3734 ± 0.0000 (n=1) | 0.8181 ± 0.0000 (n=1) |

## log-return-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.0040 ± 0.0009 (n=5) | 0.1231 ± 0.0047 (n=5) | 0.8297 ± 0.1305 (n=5) | -0.0015 ± 0.0283 (n=5) | 0.2039 ± 0.0555 (n=5) |
| Quantum V1 (75p) | 0.0041 ± 0.0005 (n=5) | 0.1242 ± 0.0044 (n=5) | 0.8104 ± 0.0766 (n=5) | -0.0064 ± 0.0274 (n=5) | -0.1700 ± 0.0327 (n=5) |
| Quantum V2 (135p) | 0.0044 ± 0.0001 (n=5) | 0.1276 ± 0.0019 (n=5) | 0.7760 ± 0.0086 (n=5) | -0.0036 ± 0.0213 (n=5) | -0.2385 ± 0.0195 (n=5) |
| Quantum V3 (75p) | 0.0050 ± 0.0017 (n=5) | 0.1258 ± 0.0039 (n=5) | 1.1790 ± 0.1147 (n=5) | 0.0191 ± 0.0392 (n=5) | -0.4193 ± 0.2450 (n=5) |
| WGAN-GP (MLP) | 0.0444 ± 0.0074 (n=5) | -0.3819 ± 0.2685 (n=5) | 3.5297 ± 0.4126 (n=5) | 0.1635 ± 0.2797 (n=5) | -0.1192 ± 0.5355 (n=5) |
| WGAN-GP (CNN) | 0.1286 ± 0.0626 (n=5) | 0.7411 ± 3.2204 (n=5) | 9.0668 ± 3.3685 (n=5) | -0.5028 ± 0.7971 (n=5) | 1.1707 ± 1.2522 (n=5) |
| WGAN-GP (LSTM) | 0.0244 ± 0.0056 (n=5) | -0.5774 ± 0.6847 (n=5) | 2.0602 ± 0.3002 (n=5) | 0.0917 ± 0.4612 (n=5) | -0.4631 ± 0.4946 (n=5) |
| VAE | 0.0158 ± 0.0000 (n=5) | -0.0070 ± 0.0007 (n=5) | 0.0186 ± 0.0019 (n=5) | -0.1778 ± 0.2233 (n=5) | -0.7296 ± 0.5446 (n=5) |
| AR(p) | 0.0029 ± 0.0001 (n=5) | 0.1209 ± 0.0057 (n=5) | 0.9919 ± 0.0036 (n=5) | 0.0011 ± 0.0123 (n=5) | 0.0023 ± 0.0295 (n=5) |
| FROZEN headline (epoch 1969) | 0.4494 ± 0.0000 (n=1) | 0.1230 ± 0.0000 (n=1) | 0.6203 ± 0.0000 (n=1) | 0.0177 ± 0.0000 (n=1) | 0.9885 ± 0.0000 (n=1) |

Source: `results/matched2000_dualscale.json` (schema: `matched-2000ep dual-scale rows[] + per-(model,scale,metric) seed-aggregate; frozen headline DISTINCT, D-14-10`).

Every value above is `_fmt()` of an `aggregates[]` row from that JSON (see `run_figure_suite.py` `render_matched2000_dualscale_comparison_table`). The number-provenance gate (`verify_number_provenance.py --target results/figures/matched2000_dualscale_comparison.md`) auto-covers this doc because its `results/*.json` rglob includes the new dual-scale JSON without any verifier edit.