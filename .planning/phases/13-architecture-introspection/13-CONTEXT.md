# Phase 13: Architecture & Introspection - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the **circuit-justification + interpretability evidence** AIChE
reviewers demand: R2-5b ("why this circuit?" → ARCH-01/02) and R2-6 ("black-box
feel" → INTRO-01/02/03). It is the **first v2.0 phase that exercises the WGAN-GP
training loop** — all prior v2.0 phases (8–12) consumed frozen Phase 09.1/10
artifacts. Phase 13 trains *new* quantum ansatz variants and produces *new
instrumented training runs* that snapshot intermediate-epoch state.

Concretely, Phase 13 produces:

1. **Ansatz comparison** (ARCH-01/02) — 3 ansatz variants spanning depth AND
   entanglement-topology axes, trained under the identical Phase 09.1/10
   protocol, multi-seed, full fidelity suite → `results/ansatz_comparison.json`
2. **Training-progression figure** (INTRO-01) — generated distribution at epochs
   {0, N/4, N/2, 3N/4, N} for the quantum generator **and all three Phase-10
   classical WGAN-GP variants** side-by-side, Pipeline B →
   `figures/training_progression.*` + underlying JSON
3. **Parameter-trajectory plot** (INTRO-02) — PQC param norms + angle histograms
   across epochs → figure + JSON
4. **Entanglement trajectory** (INTRO-03) — Von Neumann entanglement entropy
   (balanced bipartition) **with state-purity Tr(ρ²) cross-check** across
   training → figure + JSON

**In scope:** new ansatz variant definitions in `core/models/quantum.py`
(config-selectable); new instrumented training runs (via the existing dormant
`callback(epoch, metrics)` hook in `training.py`); new `run_*.py` +
`*_sweep.sh` driver(s) following the Phase 10 pattern; JSON on the established
long-form schema; introspection figures with reproducibility JSON; the two
folded training-loop bug fixes (CR-01, CR-02) + their regression tests.

**Out of scope (other phases own these):**
- Manuscript integration of the circuit-rationale subsection + figures → Phase 14
  (PAPER-03, PAPER-05, etc.)
- Any new model families beyond ansatz variants of the existing PQC
- New variance-collapse remediation (v2.0 reports honestly, does not re-attempt)
- Automated architecture search (REQUIREMENTS.md explicitly: manual comparison is
  sufficient for the reviewer)
- Shot-noise / noise-model / multi-seed roll-up (Phase 12, complete)

**Why Phase 13 exists separately:** R2-5b and R2-6 are distinct headline rebuttal
points from the comparison table (Phase 10), utility suite (Phase 11), and
sensitivity story (Phase 12). It is intentionally sequenced last among the
compute phases so its training sweeps don't contend with Phase 12's sensitivity
sweeps on the local-Mac budget.
</domain>

<decisions>
## Implementation Decisions

### Ansatz comparison axis (ARCH-01/02) — LOCKED
- **D-13-01:** **Hybrid axis — 3 variants spanning both depth and entanglement
  topology** (user-selected over a pure depth sweep). The three variants:
  - **V1 — depth-4, range-based CNOT** = the existing production ansatz
    (`num_layers=4`, range pattern `r = (layer % (num_qubits-1)) + 1`, 75 params).
    This is the manuscript's circuit and the comparison baseline. **Reuses the
    existing Phase 09.1/10 5-seed quantum runs as variant-1 in the comparison
    table — no recompute** (final-metric reuse; identical-conditions invariant
    D-10-08 already holds for these runs).
  - **V2 — depth-8, range-based CNOT** = the depth axis (`num_layers=8`, 135
    params). New 5-seed training runs.
  - **V3 — depth-4, alternative entanglement topology** = the topology axis
    (75 params, same depth as V1 so the comparison isolates topology). New
    5-seed training runs.
- **D-13-02:** The V3 alternative topology is **linear nearest-neighbor**
  (qubit `i ↔ i+1` open chain, replacing the range-based wrap-around CNOT
  pattern) — the canonical, most reviewer-interpretable contrast to the
  range-based pattern. Captured as a decision (not re-asked) per the user's
  minimize-process standing preference; the planner/researcher may substitute an
  equally-defensible standard topology (e.g., circular nearest-neighbor) only if
  research shows it is strictly more reviewer-defensible — but the *axis*
  (topology-at-fixed-depth-4) is locked.
- **D-13-03:** Variants are **config-selectable** in
  `core/models/quantum.py` (ROADMAP success criterion 1). Add an
  ansatz-spec parameter (depth + topology) to `QuantumGenerator`; the
  range-based pattern remains the default so all prior phases' behavior is
  byte-unchanged. Param-count drift across variants (75/135/75) is expected and
  correct — this is a **quantum-vs-quantum** comparison, *not* the
  matched-parameter classical comparison (that was Phase 10 BASE-01).

### Training budget & seeds (ARCH-02 + INTRO-*) — LOCKED
- **D-13-04:** **1000 epochs × 5 seeds {42, 43, 44, 45, 46}** — matches the
  Phase 09.1/10 baseline protocol exactly so the ansatz table is apples-to-apples
  against the existing baseline set, and 5-seed mean ± std is the established
  headline standard (ROADMAP ARCH-02 "identical training budget, multi-seed").
  Only V2 and V3 need new runs (5 seeds each = 10 new quantum training runs);
  V1 reuses existing runs.
- **D-13-05:** **Early stopping is DISABLED for all Phase 13 training runs.**
  Rationale: (a) ARCH-02 requires "identical training budget" — a fixed 1000-epoch
  budget with no early termination guarantees identical budget across all
  variants; (b) INTRO-01/02/03 require the full epoch trajectory to
  {0, N/4, N/2, 3N/4, N} — early stopping would truncate it. N = 1000, so
  snapshot epochs are {0, 250, 500, 750, 1000}.
- **D-13-06:** Headline runs keep **`spectral_loss_weight=0.0`** (matching the
  production unconditioned cell-65 run and Phases 09.1/10) so the ansatz
  comparison stays comparable to the existing baseline set. The CR-01 fix makes
  the spectral hook *correct if ever enabled*; it does NOT change the headline
  objective.

### Introspection run scope (INTRO-01/02/03) — LOCKED
- **D-13-07:** The introspection figures profile the **production ansatz (V1:
  depth-4, range-CNOT)** — the circuit the manuscript actually uses — so
  "what is it learning?" answers for the *published* circuit. Because the
  existing Phase 09.1/10 V1 runs were **not instrumented** (no intermediate-epoch
  snapshots), INTRO-* requires **one fresh instrumented V1 training run** that
  fires the `callback(epoch, metrics)` hook to capture: generated-distribution
  samples (INTRO-01), `params_pqc` norms + angle histograms (INTRO-02), and
  entanglement entropy + purity of the generator state (INTRO-03) at each
  snapshot epoch. Single representative seed (seed 42) for figure clarity;
  underlying data saved as JSON for reproducibility (ROADMAP criterion 4).
- **D-13-08:** INTRO-01 plots quantum **and all three Phase-10 classical
  WGAN-GP variants** (`wgan_mlp`, `wgan_cnn`, `wgan_lstm`) side-by-side on
  **Pipeline B** (headline pipeline). The Phase-10 classical runs were also not
  instrumented, so INTRO-01 requires **fresh instrumented 1000-epoch runs of the
  three classical variants** (Pipeline B, seed 42) capturing the same snapshot
  epochs. Classical 1000-epoch runs are fast — modest added compute.
- **D-13-09:** INTRO-03 entanglement measure = **Von Neumann entanglement
  entropy on a balanced bipartition + state purity Tr(ρ²) as a cross-check**,
  both saved with underlying JSON. For 5 qubits the balanced bipartition is a
  2|3 split (e.g., wires {0,1} vs {2,3,4}); exact wire partition is Claude's
  discretion (technical detail) but must be recorded in the JSON metadata.

### Folded Todos
- **CR-01 — spectral-loss hook non-differentiable + device-unsafe**
  (`core/training.py:356-360, 470-507`). Folded: Phase 13 is the first
  phase that exercises the spectral/callback path, and the todo is tagged
  `resolves_phase: 13`. Fix: make the spectral PSD loss a real differentiable
  torch term computed on the device-resident generator output (no
  detach()/numpy round-trip on the gradient path); move the target/reference PSD
  tensor to the generator's device + `compute_dtype`; add a unit test asserting
  a non-zero gradient flows into `params_pqc` when `spectral_loss_weight > 0`.
  Note: headline runs keep weight=0.0 (D-13-06) — this fix is correctness for
  the now-reachable hook, not a headline-objective change.
- **CR-02 — EarlyStopping checkpoint restore device/dtype-inconsistent**
  (`core/training.py:163-171`). Folded: tagged `resolves_phase: 13`;
  becomes reachable code in any phase that exercises the training loop. Fix:
  pass `map_location=device` and recast restored tensors to the live
  `compute_dtype` on EarlyStopping restore; move optimizer state to the active
  device so it matches `params_pqc`; add a regression test that early-stops +
  restores on MPS and on CPU asserting device/dtype consistency. Note: Phase 13
  headline runs disable early stopping (D-13-05) so the defect isn't exercised
  by the sweeps — but the fix lands here per the todo's resolves_phase and
  because the path is now reachable for future callers.

### Claude's Discretion
Per the user's standing minimize-process guidance, the following are fully
Claude's discretion (locked by prior patterns / technical detail — no user
opinion needed):
- Exact `QuantumGenerator` API surface for ansatz selection (param name, enum
  vs spec-dict) provided the range-based depth-4 default is byte-unchanged.
- Exact V3 topology if research finds a strictly-more-defensible standard
  alternative within the topology-at-fixed-depth-4 axis (D-13-02).
- Bipartition wire choice for the entanglement entropy (D-13-09), recorded in JSON.
- New driver/sweep file names and CLI surface (pattern after
  `run_baselines.py` + `run_baselines_sweep.sh`); idempotent per-cell
  skip logic; `--parallel 2` guardrail; **no `multiprocessing.Pool`**
  (Phase 09.1 Pitfall 4).
- Figure rendering details (panel layout, format, styling) and the exact
  callback-snapshot data schema, provided each figure has companion
  reproducibility JSON (ROADMAP criterion 4) and outputs join the
  `results/*.json` contract.
- Which fidelity metrics populate `ansatz_comparison.json` (reuse
  `core/eval.py` unchanged; dual-scale per EVAL-05 convention; same
  metric set as Phase 10's `baseline_comparison.json`).
- Sweep wall-time budget and `is_complete()` artifact-bundle layout (follow the
  Phase 10/12 run-dir convention D-10-14).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` (Phase 13 entry, lines ~198–207) — Goal + Success
  Criteria 1–4 (the verifiable contract)
- `.planning/REQUIREMENTS.md` — ARCH-01, ARCH-02, INTRO-01, INTRO-02, INTRO-03
  definitions; R2-5b / R2-6 rebuttal mapping; "automated arch search out of
  scope" note
- `.planning/PROJECT.md` — Locked constraints (local-Mac statevector compute,
  results-JSON contract, main-notebook-untouched, no new variance-collapse
  remediation), Key Decisions log

### Folded-todo source detail
- `.planning/todos/pending/2026-05-18-fix-cr-01-spectral-loss-hook-non-differentiable.md`
  — CR-01 problem + solution sketch
- `.planning/todos/pending/2026-05-18-fix-cr-02-earlystopping-restore-device-inconsistent.md`
  — CR-02 problem + solution sketch
- `.planning/phases/10-classical-baselines/10-REVIEW.md` (findings CR-01, CR-02)
  — full defect detail referenced by both todos

### Upstream artifact / decision contracts (Phase 13 consumes / extends)
- `.planning/phases/10-classical-baselines/10-CONTEXT.md` — identical-conditions
  invariant (D-10-08), run-dir layout (D-10-14), data-hash invariant (D-10-15),
  long-form comparison schema (D-10-16/17), code-placement invariant (D-10-13),
  sweep-driver pattern (D-10-22/23/24), HPO constants
- `.planning/phases/12-sensitivity-analysis/12-CONTEXT.md` — driver/sweep
  pattern reuse, no-`multiprocessing.Pool` rule, JSON-schema extension
  convention
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-CONTEXT.md` —
  Pipeline A/B definitions, eval.py contract, Pitfall 4
- `results/baseline_comparison.json` — long-form
  `{model_kind, pipeline, seed, metric_name, scale, value}` schema the ansatz
  table mirrors/extends (add an `ansatz` / `depth` / `topology` dimension)
- `results/baselines/runs/<model>/<pipeline>/<seed>/` — Phase-10
  classical run dirs; INTRO-01 reads which classical variants exist; existing
  V1 quantum runs are reused as ansatz variant-1 (final metrics only)

### Reusable code
- `core/models/quantum.py` — `QuantumGenerator`; `generator_circuit`
  (range-based CNOT pattern at lines ~137–162); `count_params()`. Ansatz
  variants extend this file (D-13-03)
- `core/training.py` — `train_wgan_gp`; the **dormant Phase 13
  `callback(epoch, metrics)` hook** (lines ~192, ~395–411, eval-epoch only,
  try/except-wrapped) is the instrumentation entry point for INTRO-*;
  `EarlyStopping` (lines ~79, ~163–171 = CR-02); `_spectral_psd_loss`
  (lines ~356–360, ~470–507 = CR-01)
- `core/eval.py` — `full_metric_suite` and helpers — reuse unchanged
  for `ansatz_comparison.json`; dual-scale per EVAL-05
- `core/preprocessing.py` — `inverse_minmax_od`, `inverse_logreturns`
  for OD-scale reconstruction in figures
- `run_baselines.py` + `run_baselines_sweep.sh` — reference
  template for the new Phase 13 driver(s)

### External
- PennyLane 0.44.0 — `qml.density_matrix` / `qml.vn_entropy` / `qml.purity`
  measurement APIs for INTRO-03 entanglement entropy + purity on
  `default.qubit` statevector (researcher to pin exact API for extracting
  reduced-state entropy/purity alongside the existing expval QNode)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/training.py::train_wgan_gp` already has a **purpose-built,
  dormant Phase 13 `callback(epoch, metrics)` hook** (eval-epoch only,
  try/except-wrapped so callback bugs can't kill training). INTRO-01/02/03
  instrumentation hangs off this — no training-loop surgery needed for the
  happy path.
- `core/eval.py::full_metric_suite` — full fidelity suite already
  exists; `ansatz_comparison.json` recomputes it on new ansatz samples, no new
  metric math.
- Existing Phase 09.1/10 5-seed V1 (depth-4 range-CNOT) quantum runs are reused
  as ansatz variant-1 final metrics — no recompute (D-13-01).
- `run_baselines.py` + `_sweep.sh` — idempotent per-cell driver +
  atomic `sweep_status.json` + `--parallel 2`; Phase 13 driver(s) follow this
  exact shape.

### Established Patterns
- Code-placement invariant (D-10-13): `core/` = model + eval helpers
  only (ansatz variant definitions go in `quantum.py`); all sweep/figure
  orchestration in new `run_*.py`.
- Long-form metrics schema `{model_kind, pipeline, seed, metric_name, scale,
  value}` — `ansatz_comparison.json` extends it with an ansatz/depth/topology
  dimension, does not replace it.
- Identical-conditions invariant (D-10-08): 5 seeds {42..46}, HPO constants,
  same critic, windowed data — Phase 13's new ansatz runs inherit this; only
  the ansatz (depth/topology) and epoch=1000/early-stop-off vary.
- No `multiprocessing.Pool` — xargs `-P 2` OS-process parallelism only
  (Phase 09.1 Pitfall 4).

### Integration Points
- `ansatz_comparison.json` + the four introspection figures (+ companion JSON)
  join the `results/*.json` + `figures/` contract
  Phase 14 paper-writing reads (PAPER-03 circuit rationale, PAPER-05 outlook).
- CR-01/CR-02 fixes land in `core/training.py` — must preserve the
  byte-unchanged default behavior all prior phases depend on (spectral
  weight=0.0 + callback=None + early_stopper=None defaults).
- Data-hash field (D-10-15) asserted on any reused V1 / classical artifacts
  before they enter the ansatz table / progression figure.

</code_context>

<specifics>
## Specific Ideas

- The introspection figures profile the **production circuit (V1)** specifically
  — reviewers asked "what is *this* circuit learning?", so instrumenting the
  published ansatz (not the best new variant) is the correct rebuttal framing.
- The ansatz comparison is deliberately **quantum-vs-quantum**: param-count
  drift (75/135/75) across variants is expected and is itself part of the "why
  this depth?" evidence — the matched-parameter constraint belonged to Phase 10's
  classical comparison, not here.
- Early stopping off + fixed 1000 epochs is doing double duty: it satisfies
  ARCH-02's "identical training budget" *and* guarantees the full
  {0,250,500,750,1000} introspection trajectory exists.
- Variance collapse (fake std ≈ 48% of real) remains a known, accepted
  limitation — Phase 13 explains the circuit and its learning dynamics
  honestly; it does not attempt to close the gap.

</specifics>

<deferred>
## Deferred Ideas

- **Automated circuit architecture search** — explicitly out of scope per
  REQUIREMENTS.md (manual 3-variant comparison is sufficient for the reviewer).
  Capture as a v3.0 idea only if a reviewer specifically asks.
- **Re-running Phase 10 matched-param targets for a non-depth-4 publication
  ansatz** — if the ansatz comparison surprisingly favors V2 (depth-8) as the
  publication circuit, Phase 10's ±5% matched-param classical baselines would
  need re-running against 135 params. Flagged (carried from Phase 10's deferred
  list) — a Phase-14-time decision, not a Phase-13 change. Default assumption:
  V1 remains the published circuit and the comparison justifies *why*.
- **Conditioned (PAR_LIGHT) introspection** — the conditioning hook exists but
  the published run is unconditioned; conditioned-circuit introspection is a
  follow-up, not Phase 13 scope.

### Reviewed Todos (not folded)
None — both matched todos (CR-01, CR-02) were folded into scope.

</deferred>

---

*Phase: 13-architecture-introspection*
*Context gathered: 2026-05-18*
