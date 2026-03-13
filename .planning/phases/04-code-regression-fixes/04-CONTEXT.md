# Phase 4: Code Regression Fixes - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Restore correct training behavior: fix noise range to [0, 4pi] in all 3 training loop locations, replace par_zeros eval bug with real PAR_LIGHT values, eliminate mu/sigma shadowing in plotting cells. Validate with a 200-epoch run that HPO hyperparameters produce reasonable results on corrected code.

</domain>

<decisions>
## Implementation Decisions

### Validation Failure Path
- If EMD > 2x HPO baseline (0.002274): proceed to Phase 5 but flag for HPO re-run after Phase 5/6
- If training diverges (NaN loss or critic loss > 1000): auto-fallback to v1.0 defaults (lambda_gp=10, n_critic=5) and retry
- Accept Phase 4 EMD as interim baseline regardless of whether it meets threshold
- Validation run is 200 epochs (not shortened)
- Claude executes the validation run (not manual)

### Baseline Metrics to Capture
- EMD (required by SC4)
- Moment statistics: mean, std, kurtosis of fake vs real samples
- Spectral profile: PSD comparison between real and fake (establishes pre-Phase-6 baseline)
- Training dynamics: loss curves (critic, generator), gradient norms, epoch timing
- Storage: print in notebook output AND save to JSON file
- JSON includes full config (HPO params, noise range, epoch count, timestamp, git hash) for reproducibility
- PSD depth: Claude's discretion (summary stat vs full arrays based on downstream utility)

### HPO Parameter Treatment
- Primary attempt uses HPO-tuned values: lr_g=0.003, lr_c=0.0002, lambda_gp=2.16, n_critic=9
- ACF loss disabled: lambda_acf=0 (not just zeroed — remove ACF loss code entirely)
- Spectral loss (Phase 6) will replace ACF loss as the frequency-domain signal
- Other HPO params (LRs, lambda_gp, n_critic) kept as-is to isolate the noise range variable
- Auto-fallback to v1.0 defaults if NaN/divergence occurs

### Claude's Discretion
- PSD baseline depth (summary stat vs full per-frequency arrays)
- Exact validation cell structure and output formatting
- Error state handling during validation run
- mu/sigma fix implementation (PROJECT.md suggests inline into norm.pdf() calls)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Training loop already has 3 noise sampling locations (lines 1359, 1436, 1517 in qgan_pennylane.ipynb) — straightforward 2*pi → 4*pi replacement
- Evaluation loop at line 1526 has par_zeros pattern — needs real PAR_LIGHT from dataset batch
- mu/sigma shadowing at lines 526-527 (numpy) conflicts with lines 666 (torch normalize output)
- Existing checkpoint and EMD evaluation infrastructure can be reused for validation

### Established Patterns
- Notebook uses hyperparameter cells at top for configuration
- EMD computed via scipy wasserstein_distance on raw samples
- Checkpoint save/restore pattern already handles params_pqc + critic state

### Integration Points
- Validation JSON output file will be read by Phases 5-7 for baseline comparison
- ACF loss removal affects generator loss computation in training loop
- Noise range fix affects all 3 training phases (critic, generator, eval) plus dummy_noise at line 1855

</code_context>

<specifics>
## Specific Ideas

- HPO re-run should happen after Phase 5/6 on corrected code — not within Phase 4
- ACF loss should be removed entirely, not just disabled — Phase 6 spectral loss replaces it
- Baseline JSON should be a complete record: if someone asks "what config produced this EMD?", the file answers it

</specifics>

<deferred>
## Deferred Ideas

- HPO re-run on corrected code (after Phase 5/6)
- ACF loss replacement with spectral loss (Phase 6)

</deferred>

---

*Phase: 04-code-regression-fixes*
*Context gathered: 2026-03-13*
