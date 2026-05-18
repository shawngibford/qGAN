# Phase 12: Sensitivity Analysis - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 12-sensitivity-analysis
**Areas discussed:** Noise application point, Sensitivity-grid seed budget, SENS-03 scope (all presented with recommended defaults; user delegated all to recommended defaults per standing technical-phase guidance)

---

## Noise / shot application point

| Option | Description | Selected |
|--------|-------------|----------|
| Inference-only (recommended) | Regenerate samples from trained analytic generator under noisy/finite-shot device; recompute metrics; no retraining | ✓ (default accepted) |
| Discuss alternative | Open the retrain-under-noise question | |

**User's choice:** Recommended default — inference-only.
**Notes:** Technically forced by `default.qubit, shots=None, diff_method="backprop"` (backprop incompatible with shots/noise) and local-Mac compute infeasibility of retraining the grid. Also the more reviewer-defensible narrative (fixed-model deployment-noise robustness).

---

## Sensitivity-grid seed budget

| Option | Description | Selected |
|--------|-------------|----------|
| 3 seeds for grids, 5 for roll-up (recommended) | {42,43,44} trend bands for SENS-01/02; full {42..46} mean±std for SENS-03 | ✓ (default accepted) |
| Discuss tradeoff | Consider full 5 seeds at every grid point | |

**User's choice:** Recommended default — two-tier seed strategy.
**Notes:** Escalation to 5 seeds on a grid point deferred as a planning-time decision if inter-seed spread obscures the degradation trend.

---

## SENS-03 scope

| Option | Description | Selected |
|--------|-------------|----------|
| Aggregation only (recommended) | Roll up existing Phase 10/11 per-seed artifacts into mean±std; no new compute | ✓ (default accepted) |
| Add new seeds/compute | Treat SENS-03 as new training | |

**User's choice:** Recommended default — pure aggregation.
**Notes:** 5-seed per-seed data already exists in tree; data-hash invariant asserted across consumed artifacts.

---

## Claude's Discretion

User explicitly delegated (standing guidance: minimize process on technical phases, Claude's discretion on implementation): noise-channel device wiring, output JSON structure beyond the established long-form schema, driver/sweep file names + CLI, recomputed metric set, pipeline coverage, subsampling strategy.

## Deferred Ideas

- Noise-aware retraining → v3.0 robustness study if reviewers ask.
- Full 5-seed degradation grids → planning-time escalation if 3-seed spread is noisy.
- CR-01 / CR-02 todos reviewed (weak generic match) — not folded; training-loop bugs locked to Phase 13 by the Phase 11 decision.
