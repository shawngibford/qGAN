# BASE-03 — Classical Baselines Apples-to-Apples Comparison

Quantum reference (reused from Phase 09.1, D-10-18) + 3 matched-parameter classical WGAN-GP variants + 2 non-adversarial baselines, across pipelines A and B (Pipeline C dropped, D-10-05), 5 training seeds (42-46).

`data_hash` = `91e447d4624e25b3` recomputed once and verified equal across all 50 new configs; quantum equivalence by construction (D-10-15).


## Pipeline A

| model | parameter_count | OD-EMD (mean±std) | OD-ACF lag-1 | OD-DTW mean | transformed-EMD (Pipeline B) | TSTR-lite R² |
|---|---|---|---|---|---|---|
| quantum | 75 | 1.0516 ± 0.0007 | -0.0943 | 1.7914 | — (n/a for Pipeline A) | -4.572 ± 0.085 |
| wgan_mlp | 74 | 0.9554 ± 0.0255 | -0.1249 | 2.0014 | — (n/a for Pipeline A) | -0.253 ± 1.597 |
| wgan_cnn | 73 | 0.7958 ± 0.3309 | 0.2070 | 4.0531 | — (n/a for Pipeline A) | 0.075 ± 0.078 |
| wgan_lstm | 78 | 1.0019 ± 0.0222 | -0.1575 | 1.7887 | — (n/a for Pipeline A) | -0.753 ± 0.373 |
| vae | 562 | 0.1942 ± 0.0076 | 0.4980 | 0.3934 | — (n/a for Pipeline A) | 0.993 ± 0.000 |
| ar | 3 | 0.9395 ± 0.0005 | 0.3627 | 1.3822 | — (n/a for Pipeline A) | 0.991 ± 0.005 |

## Pipeline B

| model | parameter_count | OD-EMD (mean±std) | OD-ACF lag-1 | OD-DTW mean | transformed-EMD (Pipeline B) | TSTR-lite R² |
|---|---|---|---|---|---|---|
| quantum | 75 | 0.0276 ± 0.0046 | 0.6959 | 0.3008 | 0.1215 ± 0.0004 | 0.994 ± 0.000 |
| wgan_mlp | 74 | 0.0276 ± 0.0061 | 0.6138 | 0.3075 | 0.2848 ± 0.0392 | 0.997 ± 0.000 |
| wgan_cnn | 73 | 0.1130 ± 0.1089 | 0.5778 | 0.7417 | 1.0516 ± 0.3722 | 0.997 ± 0.001 |
| wgan_lstm | 78 | 0.0293 ± 0.0044 | 0.7163 | 0.2987 | 0.1592 ± 0.0334 | 0.997 ± 0.000 |
| vae | 562 | 0.0257 ± 0.0064 | 0.7036 | 0.3067 | 0.0092 ± 0.0012 | 0.993 ± 0.000 |
| ar | 3 | 0.0291 ± 0.0041 | 0.4701 | 0.3707 | 0.7811 ± 0.0028 | 0.998 ± 0.000 |

_TSTR-lite real-only reference: R² = -13.354 ± 0.583 (train on real_windowed_OD[320:], eval on [:320], init seeds {40,41,42})._


_No recommendation is made here: per **D-10-19** Phase 14 owns the headline baseline decision, driven by Phase 11 utility numbers. This table is the apples-to-apples fidelity comparison only._

