# Phase 3: Post-Processing Consistency and Cleanup - Research

**Researched:** 2026-03-07
**Domain:** Jupyter notebook cleanup, dead code removal, variable scoping, visualization consolidation
**Confidence:** HIGH

## Summary

Phase 3 is a focused cleanup phase that addresses the final two v1 requirements (QUAL-03 dead code removal, QUAL-08 duplicate plot consolidation) plus variable shadowing and edge case fixes. The work is entirely within `qgan_pennylane.ipynb` and involves no new libraries, no architectural changes, and no training modifications. All changes are subtractive (deleting cells, simplifying functions) except for one structural addition (splitting Cell 51 into 3 cells) and two markdown section headers.

The critical risk in this phase is the variable shadowing bug in Cells 16 and 18, where `mu` and `sigma` are redefined as local visualization values, overwriting the normalization constants from Cell 15 that are used downstream in Cells 23 (denorm pipeline), 26 (training loop), 30 (early stopping checkpoints), and 40 (standalone generation). The fix must be precise: inline the `norm.pdf()` computation without introducing intermediate variable names.

**Primary recommendation:** Execute changes in dependency order -- fix variable shadowing first (correctness critical), then dead code removal, then visualization consolidation, then edge case handling. Each change should be verified by confirming no downstream cell references are broken.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `compute_gradient_penalty` standalone method -- already removed in Phase 2 (no action needed)
- Cell 57 (`d`) -- delete entire cell
- Cell 39 (dead comment about `debug_and_fix_generation()` removal) -- delete entire cell
- Cell 37 (debug print: `window`/`len(critic_loss)`) -- delete entire cell
- Cell 38 (hyperparameter sanity check print) -- keep, useful for notebook output review
- Light sweep: also catch obvious dead artifacts (stale debug prints, orphaned variables, unused imports) found during implementation, but don't refactor working code
- Keep both normalized-space AND denormalized-space visualizations (both serve distinct purposes)
- Keep Cells 42-44 (individual normalized-space comparisons) AND Cell 45 (6-panel post-training summary)
- Remove Cell 54 (histogram with hardcoded bins -0.05 to 0.05) -- inferior to Cell 50 (KDE)
- Keep Cell 50 (KDE PDF overlay) + Cell 51 (comprehensive statistical analysis)
- Keep Cell 48 (CDF in denormalized space) -- different space from Cell 51's CDF panel
- Consolidate Cells 55+56 into single DTW ablation cell with/without perturbation side by side
- Split Cell 51 (~100 lines) into 3 separate cells (metrics computation, 6-panel figure, summary interpretation)
- Keep hardcoded evaluation thresholds
- Variable shadowing fix: inline mean/std computation directly in `norm.pdf()` calls
- Keep Gaussian PDF overlay plots in Cells 16/18
- Cell 36 edge case: show bar chart, print message, then early return (skip moving average/convolve code)
- Simplify `convert_losses_pytorch_to_tf_format` to `np.array(losses)` since losses are Python floats
- Keep current cell ordering (don't reorder cells)
- Add markdown section header cells: `## Normalized Space Analysis` before Cell 42, `## Denormalized Analysis` before Cell 47

### Claude's Discretion
- Exact wording of markdown section headers
- Whether to add any additional section headers beyond the two specified
- Details of the DTW ablation cell layout (side-by-side subplots vs sequential)
- Light sweep: identification of any additional dead artifacts beyond the explicitly listed ones

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| QUAL-03 | Dead code removed: unused `compute_gradient_penalty` method, Cell 57 `d`, Cell 49 data perturbation hack | Cell-by-cell audit completed below. `compute_gradient_penalty` already removed in Phase 2. Cell 57 (`d`), Cell 39 (dead comment), Cell 37 (debug print) identified for deletion. Cell 56 perturbation kept as intentional ablation study per user decision. `convert_losses_pytorch_to_tf_format` tensor branches are confirmed dead code (training loop stores Python floats). |
| QUAL-08 | Duplicate plotting cells consolidated | Cell 54 (hardcoded histogram) identified for removal. Cells 55+56 consolidated into single DTW ablation cell. Cell 51 split into 3 for readability. All other "duplicates" confirmed as serving distinct purposes (different spaces, different detail levels). |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nbformat | 5.x | Programmatic notebook cell manipulation (if needed) | Standard for Jupyter cell operations |

### Supporting
No new libraries needed. All changes use existing imports. No import additions or removals required -- every currently imported library is used by retained cells.

### Note on Approach
All edits are direct cell content modifications within `qgan_pennylane.ipynb`. No external tooling is needed. The notebook can be edited as a JSON file (`.ipynb` format) using standard text/JSON tools, or by directly modifying cell source arrays.

## Architecture Patterns

### Notebook Cell Structure (Current)
```
Cell  0: [md]   Title
Cell  1: [code] Installs
Cell  2: [md]   # Imports
Cell  3: [code] Library imports
Cell  4: [md]   ## Data Loading and Preprocessing
Cell  5-12:     Data loading, delta analysis, statistics
Cell 13: [md]   ## Data Preprocessing Utilities
Cell 14-23:     normalize/denormalize, Lambert W, scaling, pipeline
Cell 24: [md]   # Model Definition
Cell 25-26:     Device setup, qGAN class
Cell 27: [md]   # Configuration
Cell 28-29:     Hyperparameters, windowed data
Cell 30-31:     EarlyStopping class + markdown
Cell 32: [md]   # Training
Cell 33:        Training invocation
Cell 34: [md]   # Evaluation and Visualization
Cell 35-57:     All evaluation and visualization cells
```

### Target Cell Structure (After Phase 3)
```
Cell 34: [md]   # Evaluation and Visualization
Cell 35:        Training status check
Cell 36:        Loss visualization (simplified, edge-case handled)
Cell 38:        Hyperparameter sanity check (KEPT)
Cell 40:        Standalone generation
Cell 41:        numpy conversions
NEW:     [md]   ## Normalized Space Analysis
Cell 42:        Histogram + Q-Q (normalized)
Cell 43:        ACF comparison (normalized)
Cell 44:        Leverage effect (normalized)
Cell 45:        6-panel post-training summary
Cell 46:        CSV save
NEW:     [md]   ## Denormalized Analysis
Cell 47:        Q-Q plots (denormalized)
Cell 48:        CDF (denormalized)
Cell 49:        Time series comparison
Cell 50:        KDE PDF overlay
Cell 51A:       Statistical metrics computation + print
Cell 51B:       6-panel statistical figure
Cell 51C:       Summary interpretation text
Cell 52:        ACF (denormalized)
Cell 53:        Lagged scatter plot
Cell 55+56:     DTW ablation (consolidated)

DELETED: Cell 37 (debug print)
DELETED: Cell 39 (dead comment)
DELETED: Cell 54 (hardcoded histogram)
DELETED: Cell 57 (debug `d`)
```

### Pattern: Inline Computation in norm.pdf()

**Current (Cell 16, shadows mu/sigma):**
```python
mu = torch.mean(norm_log_delta).item()
sigma = torch.std(norm_log_delta).item()
x = np.linspace(-6, 6, 100)
pdf = norm.pdf(x, mu, sigma)
```

**Fixed (no intermediate variables):**
```python
x = np.linspace(-6, 6, 100)
pdf = norm.pdf(x, torch.mean(norm_log_delta).item(), torch.std(norm_log_delta).item())
```

**Same pattern for Cell 18:**
```python
x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x, torch.mean(transformed_norm_log_delta).item(), torch.std(transformed_norm_log_delta).item())
```

### Pattern: Early Return for Edge Case (Cell 36)

**Current problem:** When `critic_loss_avg` has exactly 1 entry, the code reaches `np.convolve()` with `window=50` (or `window=5` from the reduced-window branch), but the moving average code still runs and the `convolve` results are used in plots. With 1 entry, the `range(window-1, len(critic_loss))` produces an empty range, causing issues.

**Fixed pattern:**
```python
critic_loss, generator_loss = np.array(qgan.critic_loss_avg), np.array(qgan.generator_loss_avg)

if len(critic_loss) == 1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(['Critic Loss', 'Generator Loss'], [critic_loss[0], generator_loss[0]])
    ax.set_title('Single Epoch Results')
    plt.show()
    print("Training complete with limited data.")
    # STOP HERE -- skip all moving average / convolve code below
else:
    window = min(50, len(critic_loss))
    # ... existing moving average and plotting code ...
```

The key insight: the entire block after the early return check must be in the `else` branch, not just part of it.

### Pattern: Simplified Loss Conversion

**Current (dead tensor-handling branches):**
```python
def convert_losses_pytorch_to_tf_format(critic_losses, generator_losses):
    if isinstance(critic_losses, list):
        critic_array = []
        for loss in critic_losses:
            if isinstance(loss, torch.Tensor):  # DEAD: never true after BUG-04
                squeezed = loss.squeeze().detach().cpu().numpy()
                ...
            else:
                critic_array.append(loss)
        critic_loss = np.array(critic_array)
    ...
```

**Simplified:**
```python
def convert_losses_pytorch_to_tf_format(critic_losses, generator_losses):
    """Convert loss lists to numpy arrays for plotting."""
    return np.array(critic_losses), np.array(generator_losses)
```

### Pattern: DTW Ablation Cell (Cells 55+56 Consolidated)

**Structure:** Single cell with two sections:
1. **Without perturbation:** Compute DTW distance on original series
2. **With perturbation:** Apply 5% random perturbation, compute DTW, show warping paths visualization

```python
# DTW Ablation Study: Effect of Random Perturbation on DTW Distance

series1 = real_data['Log_Return'].to_numpy().reshape(-1, 1)
series2 = fake_data['Log_Return'].to_numpy().reshape(-1, 1)

# --- Without perturbation ---
dtw_distance_clean, warping_path_clean = fastdtw(series1, series2, dist=euclidean)
print(f"DTW Distance (no perturbation): {dtw_distance_clean:.4f}")

# --- With perturbation (5% of points shifted) ---
series2_perturbed = series2.copy()
for idx in range(len(series2_perturbed)):
    if random.random() < 0.05:
        series2_perturbed[idx] += (random.random() - 0.5) / 2

dtw_distance_perturbed, warping_path_perturbed = fastdtw(series1, series2_perturbed, dist=euclidean)
print(f"DTW Distance (with perturbation): {dtw_distance_perturbed:.4f}")
print(f"DTW Distance change: {dtw_distance_perturbed - dtw_distance_clean:+.4f}")

# Warping path visualization (perturbed case)
d, paths = dtw.warping_paths(series1, series2_perturbed, window=500, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(series1, series2_perturbed, paths, best_path)

# Warping alignment visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(series1, 'bo-', label='Real Data', alpha=0.25)
ax.plot(series2_perturbed, 'gs-', label='Fake Data (perturbed)', alpha=0.25)
for (i, j) in warping_path_perturbed:
    ax.plot([i, j], [series1[i], series2_perturbed[j]], 'r-', alpha=0.1)
ax.set_xlabel('Time')
ax.set_ylabel('Log Return')
ax.set_title('DTW Warping Path (with perturbation)')
ax.legend()
plt.show()
```

### Anti-Patterns to Avoid
- **Deleting cells without checking downstream references:** Cell 37 defines `window` variable which is also defined in Cell 36 -- safe to delete. But always verify.
- **Breaking cell execution order:** Do not reorder cells. Only delete, modify in place, or insert new cells at specified positions.
- **Partial early return:** In Cell 36, the early return must prevent ALL subsequent code from executing, not just some of it. Use if/else wrapping, not bare return (since notebook cells don't support `return`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Notebook cell manipulation | Manual JSON editing with string concatenation | Python `json` module with proper structure | Notebook cells have specific JSON structure (cell_type, source as list of strings, metadata, outputs) |
| Variable name collision detection | Manual grep | Systematic cell-by-cell audit (already completed) | Need to verify no downstream breakage |

**Key insight:** This phase is exclusively about removing and simplifying code. There is nothing to build -- only things to delete, simplify, and restructure.

## Common Pitfalls

### Pitfall 1: Cell Index Drift
**What goes wrong:** Cell indices shift as cells are deleted or inserted. A plan that says "delete Cell 57" then "modify Cell 55" may be referencing wrong cells after the first deletion.
**Why it happens:** Jupyter notebooks are arrays of cells; deleting one shifts all subsequent indices.
**How to avoid:** Either (a) process cells in reverse index order (highest first), or (b) identify cells by content/first-line rather than index number after any structural change.
**Warning signs:** A cell modification doesn't match expected content.

### Pitfall 2: Variable Shadowing Fix Breaking Downstream
**What goes wrong:** Cells 16 and 18 currently overwrite `mu` and `sigma`. After fix, Cell 18 is the last cell before downstream users. The fix must ensure Cell 15's `mu`/`sigma` (torch.Tensor values) survive through to Cell 23/26/30/40.
**Why it happens:** Cell 10 also sets `mu`/`sigma` but comes BEFORE Cell 15 so is harmless. Cell 18 comes AFTER Cell 15 and is the actual problem.
**How to avoid:** After applying the fix, verify that `mu` and `sigma` at Cell 23 (first downstream consumer) are still the torch.Tensor values from Cell 15, not Python floats from the visualization cells.
**Warning signs:** Cell 23 `full_denorm_pipeline` receives float instead of tensor for `mu`/`sigma`.

### Pitfall 3: Cell 36 Edge Case -- Incomplete Early Return
**What goes wrong:** The current Cell 36 has a complex structure: it defines `convert_losses_pytorch_to_tf_format`, calls it, then has a lengthy conditional block. The edge case handling must prevent ALL downstream code (moving averages, metric conversions, twin axes plots) from executing when there's only 1 data point.
**Why it happens:** Notebook cells don't support `return` statements at module level. Must use if/else structure.
**How to avoid:** Wrap all post-conversion code in proper conditional branches. The `len(critic_loss) == 1` case should have its own complete block followed by the `else` containing all remaining code.
**Warning signs:** NameError on `window`, `generator_ma`, `critic_ma`, `emd_avg`, etc. when critic_loss_avg has 1 entry.

### Pitfall 4: Cell 51 Split -- Import Dependencies
**What goes wrong:** Cell 51 imports `ks_2samp` and `jensenshannon` inline (from scipy.stats and scipy.spatial.distance). When splitting into 3 cells, the computation cell (Cell 51A) needs these imports but they're buried mid-cell.
**Why it happens:** Original author added imports where they were needed rather than at the top.
**How to avoid:** Move inline imports from Cell 51 to the top of Cell 51A (the computation cell). Alternatively, note that `ks_2samp` is not in Cell 3's imports -- either add it to Cell 3 or keep it in Cell 51A.
**Warning signs:** NameError on `ks_2samp` or `jensenshannon` when running split cells.

### Pitfall 5: DTW Cell Consolidation -- Random State
**What goes wrong:** Cell 56's perturbation uses `random.random()` which depends on random state. Consolidating into a single cell means both the clean DTW and perturbed DTW run in sequence, which is correct. But the random perturbation makes results non-reproducible across runs.
**Why it happens:** No local seed setting before perturbation loop.
**How to avoid:** This is acceptable per user decision (ablation study). Document in the cell that perturbation results vary between runs.

## Code Examples

### Cell 16 Fix (Variable Shadowing)

**Before:**
```python
print('Original Data Min-Max')
print(torch.min(log_delta).item(), torch.max(log_delta).item())
print('Normalized Data Min-Max')
print(torch.min(norm_log_delta).item(), torch.max(norm_log_delta).item())
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))

bin_edges = np.linspace(-6, 6, num=100)
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

axes.hist(norm_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Normalized density')
axes.grid()

mu = torch.mean(norm_log_delta).item()   # <-- SHADOWS Cell 15 mu
sigma = torch.std(norm_log_delta).item()  # <-- SHADOWS Cell 15 sigma

x = np.linspace(-6, 6, 100)
pdf = norm.pdf(x, mu, sigma)

axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()
```

**After:**
```python
print('Original Data Min-Max')
print(torch.min(log_delta).item(), torch.max(log_delta).item())
print('Normalized Data Min-Max')
print(torch.min(norm_log_delta).item(), torch.max(norm_log_delta).item())
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))

bin_edges = np.linspace(-6, 6, num=100)
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

axes.hist(norm_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Normalized density')
axes.grid()

x = np.linspace(-6, 6, 100)
pdf = norm.pdf(x, torch.mean(norm_log_delta).item(), torch.std(norm_log_delta).item())

axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()
```

### Cell 18 Fix (Variable Shadowing)

**Before:**
```python
delta = 1
transformed_norm_log_delta = inverse_lambert_w_transform(norm_log_delta, delta)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))

bin_edges = np.linspace(-3, 3, num=50)
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

axes.hist(transformed_norm_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Transformed log-$\\delta$')
axes.grid()

mu = torch.mean(transformed_norm_log_delta).item()   # <-- SHADOWS Cell 15 mu
sigma = torch.std(transformed_norm_log_delta).item()  # <-- SHADOWS Cell 15 sigma

x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x, mu, sigma)

axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()
```

**After:**
```python
delta = 1
transformed_norm_log_delta = inverse_lambert_w_transform(norm_log_delta, delta)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))

bin_edges = np.linspace(-3, 3, num=50)
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

axes.hist(transformed_norm_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Transformed log-$\\delta$')
axes.grid()

x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x, torch.mean(transformed_norm_log_delta).item(), torch.std(transformed_norm_log_delta).item())

axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()
```

### Cell 36 Simplified and Edge-Case-Handled

**After (complete replacement):**
```python
# Handle the loss data for plotting

def convert_losses_pytorch_to_tf_format(critic_losses, generator_losses):
    """Convert loss lists to numpy arrays for plotting."""
    return np.array(critic_losses), np.array(generator_losses)

# Convert the data
critic_loss, generator_loss = convert_losses_pytorch_to_tf_format(
    qgan.critic_loss_avg, qgan.generator_loss_avg
)
print(f"Converted data shapes: critic_loss: {critic_loss.shape}, generator_loss: {generator_loss.shape}")

# Edge case: single epoch
if len(critic_loss) <= 1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(['Critic Loss', 'Generator Loss'], [critic_loss[0], generator_loss[0]])
    ax.set_title('Single Epoch Results')
    plt.show()
    print("Training complete with limited data. Skipping moving average plots.")
else:
    window = min(50, len(critic_loss))

    # Moving averages
    generator_ma = np.convolve(generator_loss, np.ones(window)/window, mode='valid')
    critic_ma = np.convolve(critic_loss, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    # Loss curves
    axes[0].plot(range(window-1, len(critic_loss)), critic_ma, label='Average Critic Loss', color='blue')
    axes[0].plot(critic_loss, color='black', alpha=0.2)
    axes[0].plot(range(window-1, len(generator_loss)), generator_ma, label='Average Generator Loss', color='orange')
    axes[0].plot(generator_loss, color='black', alpha=0.2)
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    # Metric curves
    emd_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.emd_avg])
    emd_ma = np.convolve(emd_avg, np.ones(window)/window, mode='valid')

    axes[1].plot(range(window-1, len(emd_avg)), emd_ma, label='EMD', color='red')
    axes[1].plot(emd_avg, color='red', linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel('EMD')
    axes[1].legend()
    axes[1].grid()

    acf_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.acf_avg])
    vol_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.vol_avg])
    lev_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.lev_avg])

    acf_ma = np.convolve(acf_avg, np.ones(window)/window, mode='valid')
    vol_ma = np.convolve(vol_avg, np.ones(window)/window, mode='valid')
    lev_ma = np.convolve(lev_avg, np.ones(window)/window, mode='valid')

    axes2 = axes[1].twinx()
    axes2.plot(range(window-1, len(acf_avg)), acf_ma, label='ACF', color='green')
    axes2.plot(acf_avg, color='green', linewidth=0.5, alpha=0.4)
    axes2.plot(range(window-1, len(vol_avg)), vol_ma, label='Volatility Clustering', color='black')
    axes2.plot(vol_avg, color='black', linewidth=0.5, alpha=0.3)
    axes2.plot(range(window-1, len(lev_avg)), lev_ma, label='Leverage Effect', color='orange')
    axes2.set_ylabel('Temporal Metrics')
    axes2.legend()
    axes2.grid()

    plt.tight_layout()
    plt.show()
```

### Cell 51 Split

**Cell 51A (Computation + Print):**
```python
# Comprehensive Statistical Analysis
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

original_data = log_delta_np.flatten()
generated_data = fake_log_delta_np.flatten()

print(f"Original data: {len(original_data)} points")
print(f"Generated data: {len(generated_data)} points")

# Earth Mover's Distance
emd_value = wasserstein_distance(original_data, generated_data)
print(f"\nEarth Mover's Distance: {emd_value:.6f}")

# Entropy analysis
bins = np.linspace(min(np.min(original_data), np.min(generated_data)),
                   max(np.max(original_data), np.max(generated_data)), 100)
real_prob, _ = np.histogram(original_data, bins=bins, density=True)
fake_prob, _ = np.histogram(generated_data, bins=bins, density=True)
real_prob = real_prob / np.sum(real_prob)
fake_prob = fake_prob / np.sum(fake_prob)
epsilon = 1e-10
real_prob = np.maximum(real_prob, epsilon)
fake_prob = np.maximum(fake_prob, epsilon)
entropy_real = entropy(real_prob)
entropy_fake = entropy(fake_prob)

print(f"\nEntropy Analysis:")
print(f"Original data entropy: {entropy_real:.6f}")
print(f"Generated data entropy: {entropy_fake:.6f}")
print(f"Entropy difference: {abs(entropy_real - entropy_fake):.6f}")
print(f"Relative entropy difference: {abs(entropy_real - entropy_fake)/entropy_real*100:.2f}%")

# Kolmogorov-Smirnov test
ks_statistic, ks_pvalue = ks_2samp(original_data, generated_data)
print(f"\nKolmogorov-Smirnov test:")
print(f"  Statistic: {ks_statistic:.6f}")
print(f"  P-value: {ks_pvalue:.6f}")
print(f"  Interpretation: {'Distributions are significantly different' if ks_pvalue < 0.05 else 'Distributions are similar'}")

# Jensen-Shannon divergence
js_distance = jensenshannon(real_prob, fake_prob)
print(f"\nJensen-Shannon Distance: {js_distance:.6f}")

# Moments comparison
print(f"\nMOMENT COMPARISONS")
stats_comparison = {
    'Mean': (np.mean(original_data), np.mean(generated_data)),
    'Std': (np.std(original_data), np.std(generated_data)),
    'Skewness': (stats.skew(original_data), stats.skew(generated_data)),
    'Kurtosis': (stats.kurtosis(original_data), stats.kurtosis(generated_data))
}
for stat_name, (real_val, fake_val) in stats_comparison.items():
    diff = abs(real_val - fake_val)
    rel_diff = (diff / abs(real_val)) * 100 if real_val != 0 else float('inf')
    print(f"{stat_name:10}: Real={real_val:8.6f}, Generated={fake_val:8.6f}, Diff={diff:.6f} ({rel_diff:.2f}%)")
```

**Cell 51B (6-Panel Figure):**
```python
# Statistical comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# [... existing 6-panel plotting code from Cell 51, using variables defined in Cell 51A ...]
```

**Cell 51C (Summary Interpretation):**
```python
# Summary interpretation
print(f"\nSUMMARY INTERPRETATION")
# [... existing interpretation code from Cell 51, using emd_value, js_distance, entropy_real, entropy_fake, ks_pvalue ...]
```

### Markdown Section Headers

**Before Cell 42 (insert new markdown cell):**
```markdown
## Normalized Space Analysis

Comparing original and generated distributions in the normalized training space.
```

**Before Cell 47 (insert new markdown cell):**
```markdown
## Denormalized Analysis

Comparing original and generated time series in the original (denormalized) log-return space.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Store losses as tensors | Store losses as Python floats via `.item()` | Phase 1 (BUG-04) | Makes `convert_losses_pytorch_to_tf_format` tensor branches dead code |
| Standalone `compute_gradient_penalty` method | Inline GP computation in training loop | Phase 2 | Already removed, no Phase 3 action needed |
| Cell indices from original notebook | Cell indices shifted by +1 after Phase 2 inserted Cell 23 | Phase 2 | REQUIREMENTS.md cell references (e.g., "Cell 49", "Cell 50") are off by 1 from current notebook |

**Cell index mapping (REQUIREMENTS.md -> Current notebook):**
- REQUIREMENTS.md "Cell 50 `d`" = Current Cell 57 (debug variable `d`)
- REQUIREMENTS.md "Cell 49 data perturbation hack" = Current Cell 56 (DTW with perturbation)
- The +1 shift happened when Phase 2 inserted `full_denorm_pipeline` as new Cell 23

## Open Questions

1. **Cell 10 also defines `mu` and `sigma`**
   - What we know: Cell 10 uses `mu = np.mean(log_delta_np)` and `sigma = np.std(log_delta_np)` for visualization. These are numpy floats, not torch tensors.
   - What's unclear: Should Cell 10 also be fixed? It comes BEFORE Cell 15 (which re-establishes the correct values), so it doesn't cause a runtime bug.
   - Recommendation: Leave Cell 10 as-is. It's in the "Data Analysis" section (before preprocessing), and Cell 15 overwrites it. Fixing it is out of scope for the "light sweep" since it's not broken, just stylistically imperfect.

2. **emd_avg/acf_avg/vol_avg/lev_avg tensor handling in Cell 36**
   - What we know: The metric arrays (`qgan.emd_avg`, `qgan.acf_avg`, etc.) use `x.item() if isinstance(x, torch.Tensor) else x` conversion. After Phase 1, losses are stored as Python floats. But are these metric arrays also stored as floats?
   - What's unclear: Whether `emd_avg`, `acf_avg`, `vol_avg`, `lev_avg` are also converted via `.item()` in the training loop, or if some remain as tensors.
   - Recommendation: Keep the `isinstance` checks for metric arrays in Cell 36 as defensive code. Only simplify `convert_losses_pytorch_to_tf_format` which handles `critic_loss_avg` and `generator_loss_avg` (confirmed as Python floats).

## Sources

### Primary (HIGH confidence)
- Direct notebook inspection (`qgan_pennylane.ipynb`) -- all cell contents verified by reading the actual file
- `.planning/phases/03-post-processing-consistency-and-cleanup/03-CONTEXT.md` -- locked user decisions
- `.planning/REQUIREMENTS.md` -- QUAL-03 and QUAL-08 requirement definitions
- `.planning/STATE.md` -- Phase 1/2 decisions affecting Phase 3

### Secondary (MEDIUM confidence)
- Cell index mapping from REQUIREMENTS.md to current notebook -- inferred from Phase 2 inserting Cell 23, verified by content matching

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all changes within existing notebook
- Architecture: HIGH -- cell structure fully audited, all referenced cells verified
- Pitfalls: HIGH -- variable shadowing verified by tracing mu/sigma through all 58 cells; edge cases identified from actual code inspection
- Code examples: HIGH -- before/after patterns derived directly from current cell contents

**Research date:** 2026-03-07
**Valid until:** Indefinite -- this is a cleanup phase with no external dependency drift
