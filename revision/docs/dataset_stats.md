# Dataset Statistics — Single-Campaign LUCY Photobioreactor

> **Source of truth:** every count below is rendered FROM `revision/results/model_info.json` (the `dataset` block, DERIVED from `data.csv` + the locked window config) by `revision/run_model_info.py`. NO hand-typed numbers; `revision/verify_number_provenance.py` is the executable gate.

This document characterizes the single-campaign dataset that backs all v2.0 evaluation work. Counts are derived from live data.csv inspection + the locked rolling-window config — never hand-typed.

## Counts

| Quantity | Value | Source / Derivation |
|----------|-------|---------------------|
| Raw CSV rows (excluding header) | 778 | `model_info.json` dataset.raw_csv_rows |
| OD rows after fillna + dropna | 778 | `model_info.json` dataset.od_rows_after_fillna_dropna |
| Log-return rows (N − 1) | 777 | `model_info.json` dataset.log_return_rows |
| Rolling windows (length 10, stride 2) | 384 | `model_info.json` dataset.rolling_windows |
| Independent campaigns | 1 | `model_info.json` dataset.independent_campaigns |

## Split Convention

| Convention | Value | Source |
|------------|-------|--------|
| Train windows | 384 | `model_info.json` dataset.train_windows |
| Val windows | 0 | `model_info.json` dataset.val_windows |
| Test windows | 0 | `model_info.json` dataset.test_windows |

**Single-Campaign Limitation.** Exactly one LUCY photobioreactor campaign; no other independent campaigns are available. 384 rolling windows is too small to justify a held-out train/val/test split without severely under-powering training, so the EMD-based early-stop metric is computed on the same distribution it compares against (stated openly per the R1-M5 calibration-honesty standard). Multi-campaign generalization is a Phase-14 Outlook item, not a current-scope claim.

## Preprocessing Pipeline

`revision/core/data.py::load_and_preprocess` applies (in order): log-return differencing → zero-mean/unit-variance standardization → Lambert-W heavy-tail correction → min-max rescaling to [−1, 1] → rolling windows of length 10 and stride 2 (yielding 384 windows). The bioprocess justification of the log-return choice (specific growth rate, μ = d ln(OD)/dt) is the subject of Phase 09.1 (R1-M3 preprocessing ablation).

