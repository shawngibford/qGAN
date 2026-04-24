---
phase: 08
plan: 02
requirements: [INFRA-01]
status: complete
completed: 2026-04-23
---

# Phase 8 Plan 02: Data + Eval Module Fill-In — Summary

Replaced all `NotImplementedError` stubs in `revision/core/data.py` and `revision/core/eval.py` with verbatim extractions of the v1.1 notebook pipeline; self-parity EMD = 0 exactly.

## Functions Extracted

### `revision/core/data.py` (216 lines)
| Function | Source cell | Notes |
|---|---|---|
| `normalize`, `denormalize` | 7 | torch default std (ddof=1 / unbiased) |
| `compute_log_delta` | 7 + 9 | dither=0.005, seed=42 via `np.random.default_rng` |
| `inverse_lambert_w_transform` | 17 | float64 (`.double()`), `scipy.special.lambertw` principal branch |
| `lambert_w_transform` | 17 | float64, clip `[-12.0, 11.0]` |
| `rolling_window`, `rescale` | 22 | |
| `full_denorm_pipeline` | 23 | windows → flat → rescale → Lambert → denormalize |
| `find_optimal_lambert_delta` | 18 | `minimize_scalar` on `bounds=(0.01, 2.0)`, bounded method |
| `load_and_preprocess` | 5+9+15+18+21+30 | Returns 11-key dict; `delta` computed dynamically |

### `revision/core/eval.py` (134 lines)
| Function | Source cell | Notes |
|---|---|---|
| `compute_emd` | 59, 65 | `wasserstein_distance` on RAW samples (v1.0 locked) |
| `compute_moments` | 59, 65 | `np.std` default (ddof=0), `scipy.stats.kurtosis` (Fisher) |
| `compute_acf` | 65 | `statsmodels.tsa.stattools.acf` with `fft=True` |
| `compute_dtw` | 64, 65 | `fastdtw` + `euclidean`, 2D `reshape(-1, 1)` |
| `compute_jsd` | 59, 65 | shared-bin histograms, normalized to PMF |
| `compute_psd` | — (not in nb final metrics) | `scipy.signal.welch` conservative default |
| `full_metric_suite` | aggregator | returns flat dict with 10 keys |

## Behavioral Decisions Preserved Verbatim

- **EMD on raw samples** (NOT histograms) — v1.0 decision
- **ACF fft=True** — cell 65
- **Kurtosis Fisher (excess), std ddof=0** — cells 59/65 use `np.std` and `stats.kurtosis` defaults
- **Lambert W float64** — cell 17 uses `.double()` internally
- **Optimal Lambert delta** — computed via `minimize_scalar` bounds=(0.01, 2.0); notebook delta = `0.14693158417899013` for current `data.csv`

## Verification

```
windowed_data shape: torch.Size([384, 10])  # stride-2, WINDOW_LENGTH=10
Lambert W round-trip err (float64): 3.47e-18  (< 1e-5 required)
Self-parity EMD(real, real) = 0.0  (< 1e-10 required)
delta = 0.14693158417899013
```

## Commits

- `99d3195` feat(08-02): fill data.py with notebook-parity implementations
- `721be89` feat(08-02): fill eval.py with notebook-parity metric implementations

## Deferred Items

- Differentiable `inverse_lambert_w_transform` — Phase 9 (EVAL-06)
- PSD-loss-matching `compute_psd` parameterization — Phase 9 if needed by 08-04 training loop

## Next

08-05 (parity check notebook) can now import `revision.core.data.load_and_preprocess` and `revision.core.eval.full_metric_suite` to compare pre/post extraction metrics.
