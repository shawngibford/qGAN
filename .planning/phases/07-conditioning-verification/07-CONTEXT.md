# Phase 7: Conditioning Verification - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Empirically determine whether PAR_LIGHT conditioning modulates generator output -- a thesis-critical question never honestly measured due to the par_zeros bug fixed in Phase 4. Deliver an intervention test (KS test at PAR_LIGHT=0 vs 1), a sweep test across PAR_LIGHT grid [0, 0.2, 0.4, 0.6, 0.8, 1.0], and make critic dropout rate configurable.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation details are at Claude's discretion. The user's priority is outcome-focused: empirical evidence on conditioning effectiveness, presented clearly for thesis use. Minimize bloat.

Specific technical decisions Claude should make:
- Intervention test structure: sample count, how to load trained model checkpoint, statistical interpretation
- Sweep test presentation: table, plots, or both -- whatever best communicates whether output varies systematically with PAR_LIGHT
- Negative result handling: if conditioning is ineffective (p > 0.05), document the finding honestly and note architectural implications for future work
- Dropout parameterization approach: how to expose the hardcoded `nn.Dropout(p=0.2)` as configurable
- Whether to reuse existing `ks_2samp` patterns already in the notebook or write fresh test cells
- Cell organization and placement within the notebook

### Locked Constraints (from requirements and prior phases)
- Intervention test must report KS test statistic with p-value at PAR_LIGHT=0 vs PAR_LIGHT=1 (COND-01)
- Sweep test must cover PAR_LIGHT grid [0, 0.2, 0.4, 0.6, 0.8, 1.0] with mean, std, kurtosis per level (COND-02)
- Dropout rate must be configurable as a hyperparameter with default 0.2 (COND-03)
- All changes in existing qgan_pennylane.ipynb -- no new files
- Must use model trained with spectral loss from Phase 6 for meaningful conditioning measurement
- par_zeros bug was fixed in Phase 4 -- evaluation now uses real PAR_LIGHT values from dataset
- Broadcasting convention from Phase 5: noise shape (num_qubits, batch_size)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Conditioning architecture
- `qgan_pennylane.ipynb` Cell containing `par_light_encoding` -- RY rotation encoding of PAR_LIGHT conditioning
- `qgan_pennylane.ipynb` Cell containing `nn.Dropout(p=0.2)` -- Hardcoded dropout in critic network (line ~1162)
- `qgan_pennylane.ipynb` Cell containing `define_generator_circuit` -- Full circuit with noise encoding + PAR_LIGHT encoding + entangling layers

### Existing statistical tests
- `qgan_pennylane.ipynb` Cells containing `ks_2samp` -- Existing KS test patterns in evaluation cells (multiple locations)
- `qgan_pennylane.ipynb` Cell containing `mannwhitneyu` -- Additional statistical test pattern

### Prior context
- `.planning/phases/04-code-regression-fixes/04-CONTEXT.md` -- par_zeros fix, baseline metrics
- `.planning/phases/05-backprop-and-broadcasting/05-CONTEXT.md` -- Broadcasting convention
- `.planning/phases/06-spectral-loss/06-CONTEXT.md` -- Spectral loss integration, lean implementation precedent

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ks_2samp` from scipy.stats already imported and used in multiple evaluation cells -- same function for intervention test
- PAR_LIGHT encoding via RY rotations in `par_light_encoding()` method -- maps [0,1] to [0, pi]
- DataLoader yields (log_return_batch, par_light_batch) tuples -- PAR_LIGHT data already windowed and available
- Existing evaluation cells generate fake samples with real PAR_LIGHT conditioning (post Phase 4 fix)

### Established Patterns
- Hyperparameters defined as class __init__ params and set in config cell at notebook top
- Statistical evaluation cells use scipy.stats functions with formatted output
- Critic network defined in single cell with Sequential layers

### Integration Points
- Dropout parameterization: `nn.Dropout(p=0.2)` in critic __init__ -- change to `nn.Dropout(p=self.dropout_rate)` with __init__ parameter
- Hyperparameter cell: dropout_rate added alongside N_CRITIC, LAMBDA, etc.
- Intervention and sweep test cells: new cells after training, using trained model checkpoint
- Generator call signature: `generator(noise_params, par_light_params, params_pqc)` -- PAR_LIGHT is already a direct input

</code_context>

<specifics>
## Specific Ideas

- This is a thesis-critical measurement -- the result (positive or negative) is valuable either way
- Phase 4 fixed the par_zeros bug that made all prior conditioning measurements meaningless -- this is the first honest test
- If conditioning is ineffective, that's an honest scientific finding worth documenting, not a failure

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-conditioning-verification*
*Context gathered: 2026-03-23*
