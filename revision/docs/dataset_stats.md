# Dataset Statistics — Single-Campaign LUCY Photobioreactor

> **Source data:** `data.csv` in the repo root (10-minute sampling; columns
> `DATE, PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO, OD, DRY, CELL`).
> Counts below are produced by `revision/core/data.py::load_and_preprocess`;
> date range is read from the first and last rows of `data.csv`.

This document characterizes the single-campaign dataset that trained the
v1.1 unconditioned QWGAN-GP and continues to back all v2.0 evaluation
work. Numerical counts are sourced from live pipeline execution and
verified against `wc -l data.csv` and direct inspection of the CSV.

## Counts

| Quantity | Value | Source / Derivation |
|----------|-------|---------------------|
| Raw CSV rows (excluding header) | 778 | `wc -l data.csv` − 1 |
| OD rows after fillna + dropna | 778 | `revision/core/data.py:211-219` (10-row rolling-mean fillna, then dropna) |
| Log-return rows (N − 1) | 777 | `revision/core/data.py:62` (`log_od[1:] - log_od[:-1]`) |
| Rolling windows (length 10, stride 2) | 384 | `(777 − 10) // 2 + 1 = 384`; `revision/core/data.py:110-118` |
| Independent campaigns | 1 | LUCY photobioreactor (Algenuity), single run |

## Sampling & Date Range

| Property | Value |
|----------|-------|
| Bioreactor | LUCY photobioreactor (Algenuity) |
| Sampling cadence | 10 minutes |
| Start date | 2024-03-27 13:12 |
| End date | 2024-04-01 23:42 |
| Duration | ~5.4 days (exact: 5.4375 days = 5 d 10 h 30 min) |

## Split Convention

| Convention | Decision |
|------------|----------|
| Train / val / test | NONE — all 384 windows used for training |
| Split ratio (train : val : test) | 100% : 0% : 0% (384 : 0 : 0 windows) — single-campaign dataset; see D-01 justification below |
| Held-out evaluation set | NONE — EMD early-stop uses the same distribution it compares against |

**Single-Campaign Limitation.** This dataset comprises exactly one
LUCY photobioreactor campaign covering ~5.4 days at 10-minute
sampling cadence; no other independent campaigns are available. The
single-campaign reality forces a methodological constraint that we
state openly here in keeping with the R1-M5 calibration honesty
standard: 384 rolling windows is too small to justify a held-out
train/val/test split without severely under-powering the training
set, and the EMD-based early-stopping metric is therefore computed
on the same distribution it compares against rather than on a
disjoint validation distribution. We acknowledge this as a study
limitation that bounds the strength of any "generalization" claim
made on these data. Generalization to multi-campaign data sets and
a proper held-out split convention is a Phase 14 Outlook item; it
is **not** a current-scope claim of the v2.0 revision.

## Preprocessing Pipeline

The training pipeline implemented in `revision/core/data.py::load_and_preprocess`
applies (in order): log-return differencing `r_t = ln(OD_{t+1}/OD_t)`
→ standardization to zero mean / unit variance → Lambert W heavy-tail
correction (`lambert_w_transform`, with optimal δ from
`find_optimal_lambert_delta`) → min-max rescaling to [−1, 1] →
rolling windows of length 10 and stride 2 (yielding 384 windows).
The differentiable inverse (`inverse_lambert_w_transform`) for the
Lambert W step is implemented by Phase 9 EVAL-06. The bioprocess
justification of the log-return choice (specific growth rate
interpretation, μ = d ln(OD) / dt) and a head-to-head ablation
against raw-OD and log-return-only pipelines is the subject of
Phase 09.1 (R1-M3 preprocessing ablation, requirements ABL-01,
ABL-02, ABL-03).

## PAR_LIGHT Note

The `PAR_LIGHT` column (photosynthetically active radiation, capped
at `PAR_LIGHT_MAX = 12.5`) is present in `data.csv` and was used as
a conditioning variable in the earlier `par_conditioned` baseline,
but conditioning was **disabled** for the v1.1 final
`unconditioned_wgan` run (`qgan_pennylane.ipynb` cell 65, `RUN_NAME
= "unconditioned_wgan"`). PAR_LIGHT remains captured in the
preprocessing pipeline and is reserved for Phase 13
conditional-generation introspection if revisited as part of the
architecture / interpretability work.
