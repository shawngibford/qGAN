# Phase 6: Spectral Loss - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add a differentiable log-PSD MSE loss term to the generator loss, computed via torch.fft.rfft on the same batch of real/fake windows used for WGAN loss. Expose lambda_psd as a configurable hyperparameter. The goal is to give the generator an explicit gradient signal penalizing wrong frequency content, directly addressing the root cause of variance collapse where the generator learns mean drift but not volatility structure.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation details are at Claude's discretion. The user's priority is outcome-focused: the model must produce output that is physically and visually realistic. Minimize bloat — keep the implementation lean and focused.

Specific technical decisions Claude should make:
- PSD computation approach (windowing, log transform, log(0) protection, normalization)
- Loss integration strategy (how PSD loss combines with WGAN loss)
- lambda_psd default value and tuning approach (SC5 requires no single loss component exceeds 10x another at equilibrium)
- Training diagnostics format (how PSD loss is reported — keep minimal, avoid bloat)
- Whether to reuse the existing eval PSD computation pattern (torch.fft.rfft + pow(2) + mean) or modify it

### Locked Constraints (from requirements and prior phases)
- PSD loss must use torch.fft.rfft and be fully differentiable (SPEC-01)
- lambda_psd must be a configurable hyperparameter (SPEC-02)
- PSD loss computed on same batch as WGAN loss — no separate forward pass (SPEC-03)
- ACF loss was fully removed in Phase 4 — spectral loss replaces it as the frequency-domain signal
- Must be compatible with Phase 5 backprop + broadcasting (batched QNode calls)
- Generator loss is currently just `generator_loss_wgan = -torch.mean(fake_scores)` — PSD term adds to this
- All changes in existing qgan_pennylane.ipynb — no new files

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training loop
- `qgan_pennylane.ipynb` Cell containing `generator_loss_wgan = -torch.mean(fake_scores)` — Current generator loss computation, PSD term integrates here
- `qgan_pennylane.ipynb` Cell containing `self.generator_loss_avg` — Loss tracking infrastructure

### PSD baseline
- `qgan_pennylane.ipynb` Phase 4 validation cell containing `torch.fft.rfft` PSD computation — Existing PSD computation pattern to potentially reuse
- Phase 4 baseline JSON stores full PSD arrays (6 frequency bins) for pre/post comparison

### Prior context
- `.planning/phases/04-code-regression-fixes/04-CONTEXT.md` — ACF removal decision, baseline metrics captured
- `.planning/phases/05-backprop-and-broadcasting/05-CONTEXT.md` — Broadcasting constraints, backprop compatibility

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 4 validation cell already computes PSD via `torch.abs(torch.fft.rfft(..., dim=1)).pow(2).mean(dim=0)` — same pattern can be adapted for differentiable loss
- Generator loss computation is a single line (`generator_loss = generator_loss_wgan`) — clean insertion point
- Hyperparameter cells at notebook top for lambda_psd configuration
- `self.generator_loss_avg` list for loss tracking

### Established Patterns
- Hyperparameters defined as class __init__ params and set in config cell
- Loss components computed then summed: was `generator_loss_wgan + self.lambda_acf * acf_penalty` before ACF removal
- Training logs print per-epoch with `print(f'  Generator loss: {generator_loss_val}')`

### Integration Points
- Generator training section: PSD loss computed on same fake_windows batch, added to generator_loss before .backward()
- Hyperparameter cell: lambda_psd added alongside existing N_CRITIC, LAMBDA, etc.
- qGAN.__init__: lambda_psd parameter added
- Evaluation/logging: PSD loss reported as separate component

</code_context>

<specifics>
## Specific Ideas

- User priority is physically and visually realistic output — variance collapse (fake std 0.0104 vs real 0.0218) is the core problem this phase addresses
- Keep implementation lean — avoid over-engineering the loss computation
- Phase 4 captured full PSD arrays (6 bins) as baseline — use these for before/after comparison

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-spectral-loss*
*Context gathered: 2026-03-19*
