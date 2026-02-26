# Phase 1: Foundation and Correctness Infrastructure - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Safe infrastructure fixes on `qgan_pennylane.ipynb` with no training behavior impact. Fixes checkpoint save/load bugs, removes unsafe code (`exit()`, `eval()`, insecure `torch.load`), wraps inference in `torch.no_grad()`, restructures DataLoader usage, and standardizes variable/hyperparameter naming. Requirements: BUG-01, BUG-04, BUG-05, BUG-06, BUG-07, PERF-02, PERF-03, QUAL-01, QUAL-02, QUAL-04, QUAL-05, QUAL-07, QUAL-09, QUAL-10.

</domain>

<decisions>
## Implementation Decisions

### Checkpoint Migration
- Clean break: only support new `critic` key in save/load. No backward compatibility with old `discriminator` key — Phase 2 invalidates all checkpoints anyway
- Save weights + full training state: model weights, optimizer states, epoch number, and current losses
- Timestamped filenames: e.g., `checkpoint_2026-02-26_14-30.pt`
- Keep latest checkpoint only — overwrite on each save to prevent disk bloat

### Notebook Cell Layout
- Single hyperparameter config cell near the top of the notebook with all UPPER_CASE constants
- Replace unsafe code in place (don't delete cells) — preserves cell count and ordering for cross-referencing with notes
- Add markdown section header cells to organize the notebook into logical sections (Imports, Config, Data Loading, Model Definition, Training Loop, Evaluation)
- Replace `eval()` with explicit dictionary lookup mapping string names to values

### DataLoader Restructuring
- Use DataLoader for proper batching — remove the flatten-to-list hack where DataLoader is immediately iterated into `gan_data_list`
- Keep current `batch_size` hyperparameter value unchanged
- Sequential window order (shuffle=False) — preserve temporal locality within epochs. The current code uses random indexing, but user prefers sequential processing
- Windows are already independent samples from rolling window preprocessing; DataLoader manages batch assembly

### Variable Naming Convention
- Stage-based naming for data variables: `raw_data` (CSV load), `log_delta` (after log transform), `scaled_data` (after normalization), `windowed_data` (after rolling window)
- Flat UPPER_CASE for all hyperparameters: `N_CRITIC`, `LAMBDA`, `BATCH_SIZE`, `NUM_QUBITS`, `WINDOW_LENGTH`, `NUM_LAYERS`, `NUM_EPOCHS` — no domain prefixes
- Remove unused code AND stale comments that reference removed variables (clean slate)

### Claude's Discretion
- Data file path: Claude decides on `./data.csv` vs `./data/data.csv` based on current file location and project structure
- Exact markdown section header wording
- Ordering of hyperparameters within the config cell
- Specific dict structure for eval() replacement

</decisions>

<specifics>
## Specific Ideas

- User emphasized: time series data preprocessing pipeline must not be altered. The DataLoader fix changes how pre-windowed samples are batched, not how windows are created from the time series
- Checkpoint timestamping format should be human-readable (YYYY-MM-DD_HH-MM)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation-and-correctness-infrastructure*
*Context gathered: 2026-02-26*
