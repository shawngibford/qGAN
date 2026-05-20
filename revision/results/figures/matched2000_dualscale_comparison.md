# Matched-2000ep dual-scale comparison — copy-paste table

Rendered ENTIRELY from `revision/results/matched2000_dualscale.json` by `revision/run_figure_suite.py` (`render_matched2000_dualscale_comparison_table`). Zero hand-typed numbers; every literal traces to that single JSON source of truth and passes `revision/verify_number_provenance.py` unmodified.

Quantum entrants (IQP:SEL 55p + ansatz V1/V2/V3) vs classical baselines (WGAN-GP × 3, VAE, AR) at the matched 2000-epoch budget. The frozen-checkpoint headline is reported as a DISTINCT row (source = `frozen_checkpoint_epoch_1969`) and is never merged into the iqp_sel_55_repro reproduction row (D-14-10).

Aggregates are mean over the 5 matched-2000ep seeds (42-46) for the 9 sweep models; the frozen headline aggregate is a single-generation value (no seed variance).

## OD-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.0275 ± 0.0046 (n=5) | 1.4026 ± 0.0073 (n=5) | 0.8751 ± 0.0103 (n=5) | 1.3611 ± 0.0184 (n=5) | 0.7862 ± 0.0567 (n=5) |
| Quantum V1 (75p) | 0.0276 ± 0.0046 (n=5) | 1.4026 ± 0.0073 (n=5) | 0.8751 ± 0.0102 (n=5) | 1.3612 ± 0.0187 (n=5) | 0.7860 ± 0.0575 (n=5) |
| Quantum V2 (135p) | 0.0276 ± 0.0046 (n=5) | 1.4026 ± 0.0074 (n=5) | 0.8751 ± 0.0104 (n=5) | 1.3614 ± 0.0186 (n=5) | 0.7871 ± 0.0579 (n=5) |
| Quantum V3 (75p) | 0.0275 ± 0.0045 (n=5) | 1.4026 ± 0.0074 (n=5) | 0.8750 ± 0.0105 (n=5) | 1.3613 ± 0.0182 (n=5) | 0.7871 ± 0.0561 (n=5) |
| WGAN-GP (MLP) | 0.0260 ± 0.0060 (n=5) | 1.3976 ± 0.0073 (n=5) | 0.8720 ± 0.0088 (n=5) | 1.3621 ± 0.0185 (n=5) | 0.7925 ± 0.0560 (n=5) |
| WGAN-GP (CNN) | 0.0543 ± 0.0524 (n=5) | 1.4249 ± 0.0699 (n=5) | 0.8911 ± 0.0505 (n=5) | 1.3679 ± 0.0218 (n=5) | 0.8270 ± 0.0803 (n=5) |
| WGAN-GP (LSTM) | 0.0282 ± 0.0045 (n=5) | 1.3931 ± 0.0162 (n=5) | 0.8691 ± 0.0157 (n=5) | 1.3604 ± 0.0187 (n=5) | 0.7819 ± 0.0574 (n=5) |
| VAE | 0.0257 ± 0.0064 (n=5) | 1.3854 ± 0.0073 (n=5) | 0.8641 ± 0.0102 (n=5) | 1.3600 ± 0.0185 (n=5) | 0.7796 ± 0.0565 (n=5) |
| AR(p) | 0.0291 ± 0.0041 (n=5) | 1.4038 ± 0.0074 (n=5) | 0.8788 ± 0.0097 (n=5) | 1.3740 ± 0.0184 (n=5) | 0.8748 ± 0.0605 (n=5) |
| FROZEN headline (epoch 1969) | 0.0231 ± 0.0000 (n=1) | 1.4074 ± 0.0000 (n=1) | 0.8843 ± 0.0000 (n=1) | 1.3657 ± 0.0000 (n=1) | 0.7772 ± 0.0000 (n=1) |

## log-return-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.1229 ± 0.0023 (n=5) | 0.1237 ± 0.0004 (n=5) | 0.0830 ± 0.0117 (n=5) | -0.0015 ± 0.0253 (n=5) | 0.2039 ± 0.0496 (n=5) |
| Quantum V1 (75p) | 0.1219 ± 0.0010 (n=5) | 0.1238 ± 0.0004 (n=5) | 0.0810 ± 0.0068 (n=5) | -0.0064 ± 0.0245 (n=5) | -0.1700 ± 0.0293 (n=5) |
| Quantum V2 (135p) | 0.1218 ± 0.0002 (n=5) | 0.1241 ± 0.0002 (n=5) | 0.0776 ± 0.0008 (n=5) | -0.0036 ± 0.0190 (n=5) | -0.2385 ± 0.0175 (n=5) |
| Quantum V3 (75p) | 0.1303 ± 0.0042 (n=5) | 0.1239 ± 0.0004 (n=5) | 0.1179 ± 0.0103 (n=5) | 0.0191 ± 0.0351 (n=5) | -0.4193 ± 0.2192 (n=5) |
| WGAN-GP (MLP) | 0.2699 ± 0.0356 (n=5) | 0.0731 ± 0.0240 (n=5) | 0.3530 ± 0.0369 (n=5) | 0.1635 ± 0.2502 (n=5) | -0.1192 ± 0.4790 (n=5) |
| WGAN-GP (CNN) | 0.6873 ± 0.2713 (n=5) | 0.1855 ± 0.2880 (n=5) | 0.9067 ± 0.3013 (n=5) | -0.5028 ± 0.7129 (n=5) | 1.1707 ± 1.1200 (n=5) |
| WGAN-GP (LSTM) | 0.1663 ± 0.0183 (n=5) | 0.0536 ± 0.0612 (n=5) | 0.2060 ± 0.0269 (n=5) | 0.0917 ± 0.4125 (n=5) | -0.4631 ± 0.4424 (n=5) |
| VAE | 0.0103 ± 0.0010 (n=5) | -0.0070 ± 0.0006 (n=5) | 0.0186 ± 0.0017 (n=5) | -0.1778 ± 0.1997 (n=5) | -0.7296 ± 0.4871 (n=5) |
| AR(p) | 0.7811 ± 0.0028 (n=5) | 0.1209 ± 0.0051 (n=5) | 0.9919 ± 0.0033 (n=5) | 0.0011 ± 0.0110 (n=5) | 0.0023 ± 0.0264 (n=5) |
| FROZEN headline (epoch 1969) | 0.1212 ± 0.0000 (n=1) | 0.1236 ± 0.0000 (n=1) | 0.0620 ± 0.0000 (n=1) | 0.0177 ± 0.0000 (n=1) | 0.9885 ± 0.0000 (n=1) |

Source: `revision/results/matched2000_dualscale.json` (schema: `matched-2000ep dual-scale rows[] + per-(model,scale,metric) seed-aggregate; frozen headline DISTINCT, D-14-10`).

Every value above is `_fmt()` of an `aggregates[]` row from that JSON (see `revision/run_figure_suite.py` `render_matched2000_dualscale_comparison_table`). The number-provenance gate (`revision/verify_number_provenance.py --target revision/results/figures/matched2000_dualscale_comparison.md`) auto-covers this doc because its `revision/results/*.json` rglob includes the new dual-scale JSON without any verifier edit.