# Phase 13: Architecture & Introspection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 13-architecture-introspection
**Areas discussed:** Fold todos, Ansatz comparison axis, Training budget & seeds, Progression figure scope, Entanglement measure

---

## Fold Todos (cross-reference)

| Option | Description | Selected |
|--------|-------------|----------|
| Fold CR-01 | Make spectral PSD loss a real differentiable torch term on device-resident output + non-zero-gradient unit test | ✓ |
| Fold CR-02 | map_location=device + dtype recast on EarlyStopping restore + move optimizer state + MPS/CPU regression test | ✓ |

**User's choice:** Fold CR-01, Fold CR-02 (both)
**Notes:** Both todos tagged `resolves_phase: 13`; Phase 13 is the first v2.0 phase to exercise the training loop (spectral/callback path + early-stop restore become reachable).

---

## Ansatz comparison axis

| Option | Description | Selected |
|--------|-------------|----------|
| Depth sweep {4,6,8} | Fixed range-CNOT topology, vary num_layers; reuses depth-4 production runs; cleanest 'why this depth?' | |
| Topology sweep @ depth-4 | Fixed depth-4, vary entanglement pattern; 'why this entanglement?' | |
| Hybrid (depth + topology) | depth-4 baseline + depth-8 + one alternative topology; spans both axes | ✓ |

**User's choice:** Hybrid (depth + topology)
**Notes:** Resolved to V1 depth-4 range-CNOT (= production baseline, reused), V2 depth-8 range-CNOT (depth axis), V3 depth-4 linear nearest-neighbor (topology axis). V3 exact topology = Claude's discretion within the topology-at-depth-4 axis.

---

## Training budget & seeds

| Option | Description | Selected |
|--------|-------------|----------|
| 1000 epochs × 5 seeds | Matches Phase 09.1/10 protocol; apples-to-apples; established headline standard | ✓ |
| 2000 epochs × 5 seeds | Full NUM_EPOCHS; most defensible but risks blowing local-Mac budget | |
| 1000 × 3 (grid) + 1 seed (figs) | Lightest; weaker uncertainty story | |

**User's choice:** 1000 epochs × 5 seeds
**Notes:** Only V2/V3 need new runs (10 quantum runs); V1 reuses existing. Early stopping disabled (identical-budget + full-trajectory requirements).

---

## Progression figure scope

| Option | Description | Selected |
|--------|-------------|----------|
| Best Phase-10 classical, Pipe B | Single best classical variant, cleanest contrast | |
| All 3 classical, Pipe B | wgan_mlp + wgan_cnn + wgan_lstm + quantum side-by-side | ✓ |
| wgan_lstm, Pipe B | Fix comparator to LSTM regardless of ranking | |

**User's choice:** All 3 classical, Pipe B
**Notes:** Requires fresh instrumented 1000-epoch runs of all 3 classical variants (Pipeline B, seed 42) since Phase-10 runs weren't instrumented. Classical 1000ep is fast.

---

## Entanglement measure

| Option | Description | Selected |
|--------|-------------|----------|
| Entropy + purity cross-check | Von Neumann entropy (balanced bipartition) + Tr(ρ²) | ✓ |
| Entanglement entropy only | Von Neumann entropy only | |
| State purity only | Tr(ρ²) only | |

**User's choice:** Entropy + purity cross-check
**Notes:** 5-qubit balanced bipartition = 2|3 split; exact wire partition = Claude's discretion, recorded in JSON metadata.

---

## Claude's Discretion

- `QuantumGenerator` ansatz-selection API surface (default = byte-unchanged range-based depth-4)
- V3 exact topology within the topology-at-depth-4 axis (default: linear nearest-neighbor)
- Entanglement bipartition wire choice
- Driver/sweep file names + CLI surface (Phase 10 pattern; no multiprocessing.Pool)
- Figure rendering details + callback-snapshot schema (companion JSON mandatory)
- Fidelity metric set for ansatz_comparison.json (reuse eval.py; dual-scale EVAL-05)

## Deferred Ideas

- Automated circuit architecture search — out of scope per REQUIREMENTS.md
- Re-running Phase 10 matched-param targets if a non-depth-4 ansatz becomes the publication circuit — Phase-14-time decision
- Conditioned (PAR_LIGHT) introspection — follow-up, not Phase 13 scope
