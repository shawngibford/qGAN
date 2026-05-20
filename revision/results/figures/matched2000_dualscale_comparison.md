# Matched-2000ep dual-scale comparison — copy-paste table

Rendered ENTIRELY from `revision/results/matched2000_dualscale.json` by `revision/run_figure_suite.py` (`render_matched2000_dualscale_comparison_table`). Zero hand-typed numbers; every literal traces to that single JSON source of truth and passes `revision/verify_number_provenance.py` unmodified.

Quantum entrants (IQP:SEL 55p + ansatz V1/V2/V3) vs classical baselines (WGAN-GP × 3, VAE, AR) at the matched 2000-epoch budget. The frozen-checkpoint headline is reported as a DISTINCT row (source = `frozen_checkpoint_epoch_1969`) and is never merged into the iqp_sel_55_repro reproduction row (D-14-10).

Aggregates are mean over the 5 matched-2000ep seeds (42-46) for the 9 sweep models; the frozen headline aggregate is a single-generation value (no seed variance).

## OD-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.0275 ± 0.0051 (n=5) | 1.4026 ± 0.0082 (n=5) | 0.8751 ± 0.0115 (n=5) | 1.3611 ± 0.0205 (n=5) | 0.7862 ± 0.0634 (n=5) |
| Quantum V1 (75p) | 0.0276 ± 0.0051 (n=5) | 1.4026 ± 0.0082 (n=5) | 0.8751 ± 0.0114 (n=5) | 1.3612 ± 0.0209 (n=5) | 0.7860 ± 0.0643 (n=5) |
| Quantum V2 (135p) | 0.0276 ± 0.0051 (n=5) | 1.4026 ± 0.0083 (n=5) | 0.8751 ± 0.0117 (n=5) | 1.3614 ± 0.0209 (n=5) | 0.7871 ± 0.0647 (n=5) |
| Quantum V3 (75p) | 0.0275 ± 0.0051 (n=5) | 1.4026 ± 0.0082 (n=5) | 0.8750 ± 0.0117 (n=5) | 1.3613 ± 0.0204 (n=5) | 0.7871 ± 0.0627 (n=5) |
| WGAN-GP (MLP) | 0.0260 ± 0.0067 (n=5) | 1.3976 ± 0.0082 (n=5) | 0.8720 ± 0.0099 (n=5) | 1.3621 ± 0.0207 (n=5) | 0.7925 ± 0.0626 (n=5) |
| WGAN-GP (CNN) | 0.0543 ± 0.0586 (n=5) | 1.4249 ± 0.0782 (n=5) | 0.8911 ± 0.0565 (n=5) | 1.3679 ± 0.0244 (n=5) | 0.8270 ± 0.0898 (n=5) |
| WGAN-GP (LSTM) | 0.0282 ± 0.0050 (n=5) | 1.3931 ± 0.0182 (n=5) | 0.8691 ± 0.0176 (n=5) | 1.3604 ± 0.0209 (n=5) | 0.7819 ± 0.0642 (n=5) |
| VAE | 0.0257 ± 0.0072 (n=5) | 1.3854 ± 0.0081 (n=5) | 0.8641 ± 0.0114 (n=5) | 1.3600 ± 0.0206 (n=5) | 0.7796 ± 0.0632 (n=5) |
| AR(p) | 0.0291 ± 0.0046 (n=5) | 1.4038 ± 0.0083 (n=5) | 0.8788 ± 0.0109 (n=5) | 1.3740 ± 0.0206 (n=5) | 0.8748 ± 0.0677 (n=5) |
| FROZEN headline (epoch 1969) | 0.0231 ± 0.0000 (n=1) | 1.4074 ± 0.0000 (n=1) | 0.8843 ± 0.0000 (n=1) | 1.3657 ± 0.0000 (n=1) | 0.7772 ± 0.0000 (n=1) |

## log-return-scale aggregates (mean ± std over 5 seeds; n=1 for headline)

| model | EMD | moment_mean | moment_std | moment_skewness | moment_kurtosis |
|---|---|---|---|---|---|
| IQP:SEL 55p (2000ep repro) | 0.1229 ± 0.0026 (n=5) | 0.1237 ± 0.0005 (n=5) | 0.0830 ± 0.0130 (n=5) | -0.0015 ± 0.0283 (n=5) | 0.2039 ± 0.0555 (n=5) |
| Quantum V1 (75p) | 0.1219 ± 0.0011 (n=5) | 0.1238 ± 0.0004 (n=5) | 0.0810 ± 0.0077 (n=5) | -0.0064 ± 0.0274 (n=5) | -0.1700 ± 0.0327 (n=5) |
| Quantum V2 (135p) | 0.1218 ± 0.0002 (n=5) | 0.1241 ± 0.0002 (n=5) | 0.0776 ± 0.0009 (n=5) | -0.0036 ± 0.0213 (n=5) | -0.2385 ± 0.0195 (n=5) |
| Quantum V3 (75p) | 0.1303 ± 0.0047 (n=5) | 0.1239 ± 0.0004 (n=5) | 0.1179 ± 0.0115 (n=5) | 0.0191 ± 0.0392 (n=5) | -0.4193 ± 0.2450 (n=5) |
| WGAN-GP (MLP) | 0.2699 ± 0.0398 (n=5) | 0.0731 ± 0.0268 (n=5) | 0.3530 ± 0.0413 (n=5) | 0.1635 ± 0.2797 (n=5) | -0.1192 ± 0.5355 (n=5) |
| WGAN-GP (CNN) | 0.6873 ± 0.3034 (n=5) | 0.1855 ± 0.3220 (n=5) | 0.9067 ± 0.3369 (n=5) | -0.5028 ± 0.7971 (n=5) | 1.1707 ± 1.2522 (n=5) |
| WGAN-GP (LSTM) | 0.1663 ± 0.0205 (n=5) | 0.0536 ± 0.0685 (n=5) | 0.2060 ± 0.0300 (n=5) | 0.0917 ± 0.4612 (n=5) | -0.4631 ± 0.4946 (n=5) |
| VAE | 0.0103 ± 0.0011 (n=5) | -0.0070 ± 0.0007 (n=5) | 0.0186 ± 0.0019 (n=5) | -0.1778 ± 0.2233 (n=5) | -0.7296 ± 0.5446 (n=5) |
| AR(p) | 0.7811 ± 0.0031 (n=5) | 0.1209 ± 0.0057 (n=5) | 0.9919 ± 0.0036 (n=5) | 0.0011 ± 0.0123 (n=5) | 0.0023 ± 0.0295 (n=5) |
| FROZEN headline (epoch 1969) | 0.1212 ± 0.0000 (n=1) | 0.1236 ± 0.0000 (n=1) | 0.0620 ± 0.0000 (n=1) | 0.0177 ± 0.0000 (n=1) | 0.9885 ± 0.0000 (n=1) |

Source: `revision/results/matched2000_dualscale.json` (schema: `matched-2000ep dual-scale rows[] + per-(model,scale,metric) seed-aggregate; frozen headline DISTINCT, D-14-10`).

Every value above is `_fmt()` of an `aggregates[]` row from that JSON (see `revision/run_figure_suite.py` `render_matched2000_dualscale_comparison_table`). The number-provenance gate (`revision/verify_number_provenance.py --target revision/results/figures/matched2000_dualscale_comparison.md`) auto-covers this doc because its `revision/results/*.json` rglob includes the new dual-scale JSON without any verifier edit.