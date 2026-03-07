# Phase 3: Post-Processing Consistency and Cleanup - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove dead code, debug artifacts, and duplicate visualization cells from `qgan_pennylane.ipynb`. Fix variable shadowing of normalization constants. Handle edge cases in loss visualization. Final polish after Phase 2 training validation. Requirements: QUAL-03, QUAL-08. Gap closure from v1.0 milestone audit.

</domain>

<decisions>
## Implementation Decisions

### Dead Code Removal (QUAL-03)
- `compute_gradient_penalty` standalone method — already removed in Phase 2 (no action needed)
- Cell 57 (`d`) — delete entire cell
- Cell 39 (dead comment about `debug_and_fix_generation()` removal) — delete entire cell
- Cell 37 (debug print: `window`/`len(critic_loss)`) — delete entire cell
- Cell 38 (hyperparameter sanity check print) — keep, useful for notebook output review
- Light sweep: also catch obvious dead artifacts (stale debug prints, orphaned variables, unused imports) found during implementation, but don't refactor working code

### Duplicate Plot Consolidation (QUAL-08)
- Keep both normalized-space AND denormalized-space visualizations (both serve distinct purposes: model training space vs real-world interpretability)
- Keep Cells 42-44 (individual normalized-space comparisons: histogram+Q-Q, ACF, leverage effect) AND Cell 45 (6-panel post-training summary) — individual cells give detailed views, summary gives overview
- Remove Cell 54 (histogram with hardcoded bins -0.05 to 0.05) — inferior to Cell 50 (KDE) which uses data-driven density estimation
- Keep Cell 50 (KDE PDF overlay) + Cell 51 (comprehensive statistical analysis) — Cell 50 gives clean visual, Cell 51 gives quantitative rigor
- Keep Cell 48 (CDF in denormalized space) — different space from Cell 51's CDF panel, both valid
- Consolidate Cells 55+56 into single DTW ablation cell: show DTW computation with AND without perturbation side by side, explicitly labeled as ablation comparison (user wants to keep the perturbation as an intentional study, not remove it as dead code)

### Cell 51 Reorganization
- Split Cell 51 (~100 lines) into 3 separate cells:
  - Cell A: Compute all statistical metrics (EMD, KS test, Jensen-Shannon, entropy, moments) + print results
  - Cell B: 6-panel figure (histograms, CDF, Q-Q, moments bar chart, distance metrics, entropy comparison)
  - Cell C: Summary interpretation text (EXCELLENT/GOOD/FAIR/POOR ratings)
- Keep hardcoded evaluation thresholds (EMD < 0.01 = EXCELLENT, etc.) — reasonable defaults, no need to make configurable

### Variable Shadowing Fix
- Cells 16 and 18 currently redefine `mu` and `sigma`, shadowing normalization constants from Cell 15
- Fix: inline the mean/std computation directly in `norm.pdf()` calls — no intermediate variables at all
- Keep the Gaussian PDF overlay plots in both cells (histogram + Gaussian is a useful normality visualization alongside Q-Q plots)

### Cell 36 Edge Case Handling
- When `critic_loss_avg` has exactly 1 entry: show the bar chart, print the message, then early return (skip all moving average / convolve code)
- Simplify `convert_losses_pytorch_to_tf_format` function: after Phase 1 (BUG-04), losses are stored as Python floats via `.item()`. The tensor-handling branches are dead code — simplify to `np.array(losses)`

### Notebook Section Organization
- Keep current cell ordering (don't reorder cells)
- Add markdown section header cells to separate visualization sections:
  - `## Normalized Space Analysis` before Cell 42
  - `## Denormalized Analysis` before Cell 47 (or equivalent after CSV save)
- Follows Phase 1 convention of adding markdown section headers for logical organization

### Claude's Discretion
- Exact wording of markdown section headers
- Whether to add any additional section headers beyond the two specified
- Details of the DTW ablation cell layout (side-by-side subplots vs sequential)
- Light sweep: identification of any additional dead artifacts beyond the explicitly listed ones

</decisions>

<specifics>
## Specific Ideas

- DTW perturbation is intentionally kept as an ablation study — user wants with/without perturbation comparison side by side, not removal as dead code
- Cell 38 (hyperparameter sanity check) is explicitly kept despite being a print-only cell — user finds it useful for reviewing notebook output
- Phase 1 established "replace in place, don't delete cells" but Phase 3 is the cleanup phase where cell deletion is appropriate
- Cell 51 split into 3 cells is the only structural addition; otherwise this phase is purely subtractive

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Cell 45 post-training summary: already implements a 6-panel consolidated view — pattern to reference for Cell 51 figure cell
- Cell 34 markdown header (`# Evaluation and Visualization`): existing section header pattern to follow

### Established Patterns
- Markdown section headers: Phase 1 added `# Imports`, `# Configuration`, `# Model Definition`, `# Training`, `# Evaluation and Visualization`
- Loss storage: Phase 1 (BUG-04) ensures all losses stored as Python floats via `.item()`
- normalize() returns `(normalized_data, mu, sigma)` tuple — Cell 15 unpacks correctly

### Integration Points
- Cells 16/18 variable fix must not affect Cell 15's `mu`/`sigma` which are used downstream (Cell 23 `full_denorm_pipeline`, Cell 40 standalone generation, checkpoint saving)
- Cell 36's `window` variable is used by the moving average plotting code later in the same cell — early return must skip all dependent code
- Cell 46 (CSV save) is the boundary between normalized-space analysis and denormalized-space analysis

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-post-processing-consistency-and-cleanup*
*Context gathered: 2026-03-07*
