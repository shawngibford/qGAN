# Legacy Artifacts

This directory holds pre-revision and historical content from the qGAN project that was quarantined here during the post-AIChE-submission repo cleanup (commit following tag `v1.2.5-pre-cleanup`).

Nothing in this directory is part of the current AIChE manuscript build or is referenced by current `core/`, `results/`, `run_*.py`, or `docs/` artifacts. Everything here is kept for archaeology — reproducibility of older milestones, blame-trail continuity, or future archival reference.

## Contents

| Path | Era | What it is |
|---|---|---|
| `archive/` | v1.0/v1.1 | Old notebooks + results from earlier iterations |
| `Final Results from 2000 epochs - IQP:SEL circuit/` | v1.0 | Named results directory from a 2000-epoch IQP:SEL run (pre-revision) |
| `qgan_pennylane.ipynb` | v1.x | Original monolithic notebook (before extraction to `core/`) |
| `qgan_pennylane copy.ipynb` | v1.x | Duplicate of the original notebook |
| `results/` | pre-revision | Old HPO outputs and pre-extraction results |
| `tests/` | pre-revision | 8+ test modules predating `core/` extraction; current tests now at root `tests/` |
| `scripts/` | v1.1-v2.0 | One-off helpers (`build_parity_notebook.py`, `phase4_validation.py`) not needed by current pipeline |
| `datasets/` | pre-revision | Older synthetic-data CSVs (overlay_synthetic_vs_real.png + synthetic_NNN.csv) |
| `datasets.zip` | pre-revision | Zipped datasets snapshot |
| `fake.csv` | pre-revision | Old test/fake data |
| `QGAN_Review_Response_Plan.md.pdf` | pre-revision | Earlier review-response planning doc |
| `amp` | unknown | Empty 0-byte file, kept for completeness |

## Recovery

If you need any of this back in the active tree, check out the recovery tag:

```bash
git checkout v1.2.5-pre-cleanup -- <path>
```

For example:

```bash
git checkout v1.2.5-pre-cleanup -- archive/
```

The pre-cleanup state is fully recoverable from `v1.2.5-pre-cleanup`.
