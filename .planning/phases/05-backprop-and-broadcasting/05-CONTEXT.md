# Phase 5: Backprop and Broadcasting - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Switch QNode diff_method from parameter-shift to backprop and convert all 3 per-sample Python loops (critic training, generator training, evaluation) to batched QNode calls. Target ~12x epoch speedup. No changes to model architecture, loss function, or hyperparameters.

</domain>

<decisions>
## Implementation Decisions

### Verification Approach
- One-time validation cell — run during Phase 5, capture results in notebook output, then remove the cell
- Batched vs sequential output equivalence verified within 1e-6 tolerance (element-wise comparison on a test batch)
- 20-epoch same-seed mini-run to confirm training dynamics are preserved
- Same-seed comparison: run 20 epochs with identical seed before and after broadcasting, loss values at each epoch must match within 1e-6
- Both the equivalence test and mini-run are one-time validation — cells removed after confirmation

### Performance Measurement
- Timing comparison printed in notebook output only (no separate JSON file)
- Claude picks measurement approach (before/after in same session vs Phase 4 baseline as reference)
- SC4 (<30% of pre-broadcasting time over 10 consecutive epochs) is a hard gate — phase fails if not met, must investigate before proceeding to Phase 6

### Notebook Documentation
- Brief comment at QNode definition explaining WHY backprop: PennyLane issue #4462 (parameter-shift has broadcasting gradient bugs)
- Replace the existing comment entirely ("Explicit gradient method for better stability and gradient flow") — no history preserved
- Shape comments at key broadcasting points (e.g., `# noise: (batch_size, num_qubits) — batched QNode call`)
- Minimal comments elsewhere — only where pattern is non-obvious

### Claude's Discretion
- Error handling and fallback strategy (if backprop+broadcasting changes training dynamics)
- Pre-broadcasting timing measurement approach (same-session before/after vs Phase 4 reference)
- Exact structure of validation cells
- How to handle the isinstance/type-conversion boilerplate that exists in per-sample loops (may simplify with broadcasting)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- QNode definition at `define_generator_model()` (Cell 26, ~L218): `qml.QNode(..., diff_method='parameter-shift')` — single-point change to `'backprop'`
- Quantum device already `default.qubit` (Cell 26, L20): compatible with backprop, needs `shots=None` added
- Phase 4 validation cell (Cell 41) has a partially batched noise/PAR_LIGHT generation pattern that can inform the broadcasting approach
- Evaluation cell already generates noise as `(num_qubits, num_samples)` array — close to batched shape

### Established Patterns
- Per-sample loops at 3 locations: critic training (L293-L333), generator training (L407-L417), evaluation (L449-L468)
- Each loop does: generate noise → compress PAR_LIGHT → call generator → isinstance check → type conversion → scale by 0.1
- Generator accepts: `generator(noise_params, par_light_params, params_pqc)` — all are per-sample tensors currently
- Output type handling: `isinstance(gen_out, (list, tuple))` → `torch.stack(list(gen_out))` — may simplify with batched output

### Integration Points
- QNode interface='torch' — backprop requires this (already set)
- Generator output feeds into critic as 2-channel input `[batch, 2, window_length]` — broadcasting must preserve this shape
- Gradient flow through `self.params_pqc` must be maintained in generator training loop
- Critic training uses `torch.no_grad()` for fake sample generation — batched version must preserve this

</code_context>

<specifics>
## Specific Ideas

- PennyLane issue #4462 confirms parameter-shift has known broadcasting gradient bugs — backprop is the required path, not just an optimization
- The 1e-6 tolerance for same-seed loss comparison is strict but appropriate — backprop and parameter-shift should compute identical gradients on default.qubit simulator
- Hard gate on performance (SC4) ensures the phase delivers its core promise — if broadcasting doesn't speed things up, something is fundamentally wrong

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-backprop-and-broadcasting*
*Context gathered: 2026-03-18*
