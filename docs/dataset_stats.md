# Dataset Statistics — Single-Campaign LUCY Photobioreactor

> **Source of truth:** every count below is rendered FROM `results/model_info.json` (the `dataset` block, DERIVED from `data.csv` + the locked window config) by `scripts/run_model_info.py`. NO hand-typed numbers; `scripts/verify_number_provenance.py` is the executable gate.

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

The matched-budget runs use Pipeline B (decision D-10-05; see `run_ablation.py::build_dataset_for_pipeline`, pipeline=='B' branch). Pipeline B applies (in order): log-return differencing → zero-mean/unit-variance standardization → linear rescaling to [−1, 1] using the global min/max of the standardized series → rolling windows of length 10 and stride 2 (yielding 384 windows). Pipeline C (the v1.1 published pipeline with an inverse Lambert-W heavy-tail correction between the standardization and rescaling steps) was dropped per D-10-05 because the 09.1 ablation showed it tied with B on every OD-scale metric while introducing an over-Gaussianization concern (R1-M3). `load_and_preprocess` retains the Pipeline C path for reproducibility of the ablation only; the matched-budget pathway is `build_dataset_for_pipeline('B', ...)`. The bioprocess justification of the log-return choice (specific growth rate, μ = d ln(OD)/dt) is the subject of Phase 09.1.

