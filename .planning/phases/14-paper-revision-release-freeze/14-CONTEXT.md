# Phase 14: Paper Revision & Release Freeze - Context

**Gathered:** 2026-05-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 14 revises manuscript aic-4719598 end-to-end and freezes the repository
with a tagged release + Zenodo DOI. **This discussion deliberately expanded the
phase** (see D-14-21): in addition to the original paper-edit + freeze scope, the
phase now ALSO recovers the lost canonical quantum-circuit configuration and
re-executes the full evaluation suite (Phases 10–13 artifacts) at a matched
2000-epoch budget so the manuscript's numbers are coherent, fair, and traceable.

Concretely, Phase 14 delivers:

1. **Canonical-config recovery** — reverse-engineer the lost 55-parameter
   IQP:SEL circuit (the proven-good 2000-epoch result preserved only as figures
   in `Final Results from 2000 epochs - IQP:SEL circuit/` and as the trained
   tensor in `best_checkpoint.pt`, epoch 1969), lock it as an explicit
   config-selectable circuit in `revision/core`, and freeze the checkpoint as
   the canonical headline artifact.
2. **Matched-budget re-execution** — re-run every model (55-param IQP:SEL
   reproduction, all ansatz variants V1/V2/V3, classical wgan_mlp/cnn/lstm,
   non-adversarial VAE/AR) at 2000 epochs and regenerate every downstream
   evaluation artifact at that single budget.
3. **Unified model-info table + complete figure suite** — one paper-ready
   all-models table and a comprehensive per-model + cross-model + analysis
   figure set, both generated from JSON artifacts (no hand-typed numbers).
4. **Manuscript revision package** — copy-paste LaTeX blocks for PAPER-01..11 +
   a per-reviewer response document; the `.tex` source stays external/read-only.
5. **Release freeze** — tag `v2.0-revision` + Zenodo DOI, strictly gated on all
   cited numbers passing the strict accept gate.

**Out of scope (unchanged from PROJECT.md):** new variance-collapse remediation
or circuit-architecture re-attempts — v2.0 reports honestly, does not re-attempt.
The 2000-epoch re-run is a *matched-budget* re-execution of existing configs,
NOT a new modeling attempt. Hardware/QPU execution remains out (simulator-only).
Full closed-loop decision pipeline and first-principles Hybrid-GAN remain in
Outlook only.

</domain>

<decisions>
## Implementation Decisions

### Canonical Result Recovery & Reproduction

- **D-14-01:** The canonical 2000-epoch result was produced by a **55-parameter
  IQP:SEL circuit** (IQP RZ encoding + Strongly Entangled Layers). This differs
  from the current `revision/core` default (`NUM_QUBITS=5, NUM_LAYERS=4` → **75**
  params). Phase 14's FIRST deliverable is to reverse-engineer the exact 55-param
  architecture from `best_checkpoint.pt`'s tensor layout + the canonical figures
  + git history of `qgan_pennylane.ipynb`, and add it to `revision/core` as an
  explicit, locked, **config-selectable** circuit.
- **D-14-02:** **Config-source authority rule** — `best_checkpoint.pt` tensor
  layout (`params_pqc` shape = 55, plus optimizer `param_groups`) is GROUND
  TRUTH. Notebook git-history is corroborating only. Reconstruction is
  deterministic and driven by the checkpoint.
- **D-14-03:** The **headline quantum result is generated from the frozen
  `best_checkpoint.pt`** (epoch 1969) loaded into the reconstructed 55-param
  IQP:SEL config — NOT from a fresh retrain. A clean 2000ep retrain runs as a
  **non-load-bearing reproducibility demonstration**, cross-checked against the
  canonical figures + checkpoint EMD.
- **D-14-04:** The reconstructed **55-param IQP:SEL is the quantum entrant in
  EVERY cross-model comparison** (baseline_comparison, tstr,
  predictive_discriminative, fidelity_dualscale, sensitivity grids, and as the
  reference in ansatz_comparison) — the 75-param current default is NOT the
  paper's quantum circuit.
- **D-14-05:** **Checkpoint reproduction landmines (LOCKED requirement):**
  canonical headline generation MUST use the checkpoint's stored `mu`/`sigma`
  normalization stats + a fixed generation seed — never freshly-computed stats —
  or the figures will not reproduce byte-for-byte.
- **D-14-06:** **Canonical pipeline pinning** — identify which Phase-09.1
  preprocessing pipeline (A/B) the 55-param checkpoint was trained on (from
  `results/run_unconditioned_wgan/stats.json` + git archaeology) and pin it as
  the canonical headline pipeline. The 55-param IQP:SEL still runs all pipelines
  in cross-model comparisons, but headline figures/numbers are reported on its
  native pipeline.
- **D-14-07:** **Config-equivalence assertion** — before any sweep, hard-assert
  that loading `best_checkpoint.pt` into the reconstructed config succeeds with
  identical param shape (55) + circuit structure. Failure blocks the phase.

### Matched-Budget Re-execution

- **D-14-08:** Re-run **ALL models at matched 2000 epochs** before paper
  integration: 55-param IQP:SEL (reproduction), ansatz variants V1/V2/V3,
  classical wgan_mlp/cnn/lstm, non-adversarial VAE/AR. Same configs/budget only —
  no variance/architecture re-attempt. (Prior comparison runs were 1000ep;
  headline checkpoint is 2000ep — this closes the unfair-comparison gap.)
- **D-14-09:** **Full regeneration** — every downstream artifact
  (fidelity_dualscale, baseline_comparison, tstr, predictive_discriminative,
  multiseed_summary, shot_noise_sensitivity, noise_model_sensitivity,
  ansatz_comparison, augmentation, eval06_roundtrip) is regenerated at 2000ep.
  Single-budget, coherent paper, **zero mixed-budget caveats**.
- **D-14-10:** All quantum **ansatz variants V1/V2/V3 also retrain at 2000ep**
  with identical seed set + data hash → `ansatz_comparison` becomes a fully
  matched-budget comparison, directly strengthening the "why this circuit"
  rebuttal (R2-5b / ARCH-01/02). The 55-param IQP:SEL is the reference variant;
  headline (frozen checkpoint) and the matched-budget reproduction instance are
  reported **distinctly, never conflated**.
- **D-14-11:** **Backend = `default.qubit` + `diff_method="backprop"`** — NO
  swap to `lightning.qubit` (would force `adjoint`, reintroduce v1.1 broadcasting
  bugs, be a frozen-core change, and re-baseline Phases 8–13). Numerically
  continuous with validated phases. Quantum is honestly reported as a CPU
  statevector sim (PennyLane simulators are not Metal-accelerated).
- **D-14-12:** Parallelize via **`xargs -P2`** (validated M-series thermal cap;
  `--parallel ≥3` hard-rejected). Each run emits a **device/dtype manifest** and
  **hard-asserts the actual backend** — fails loudly on silent CPU/dtype
  fallback (e.g., CPU-float64 when MPS-float32 expected) so the table reports
  the true device per model.
- **D-14-13:** **Strict accept gate** — an artifact is trusted only if: device
  manifest assertion passed (no silent fallback), `data_hash` matches the frozen
  Phase-09.1 dataset across ALL artifacts, seed set is identical (`{42..46}`),
  JSON conforms to the long-form schema, and the run completed the full 2000ep
  (no early-stop on headline path). A **reconciliation note records 1000ep→2000ep
  metric deltas** for transparency.
- **D-14-14:** **Run-to-completion, no hard time-box** (correctness over speed
  for a freeze). Resumable `sweep_status.json` (skip-already-done) + stall
  watchdog, background execution. **Tiered priority**, each tier accepted
  independently via the strict gate so the paper progresses tier-by-tier:
  - T1: config recovery + checkpoint freeze + equivalence assertion
  - T2: headline + baseline_comparison + tstr (claim-bearing)
  - T3: sensitivity grids + ansatz_comparison

### Model Info Table & Provenance

- **D-14-15:** **One unified paper-ready table**, every model a row (production
  55-param IQP:SEL, ansatz V1/V2/V3, wgan_mlp/cnn/lstm, VAE, AR). Columns
  (Claude discretion): params, epochs, early-stop state, optimizer/LR/betas,
  batch, N_CRITIC, λ, seeds, device/dtype, window config, data hash, wall-time.
- **D-14-16:** New `revision/run_model_info.py` introspects each model's actual
  training config + run artifacts → emits `revision/results/model_info.json`
  (long-form schema + `data_hash`). The paper table AND `revision/docs/*.md`
  are regenerated FROM that JSON. **No hand-typed numbers** (success criterion
  5); the markdown docs stop being hand-maintained.

### Figure Suite

- **D-14-17:** Add a `revision/` figure module generating, from the 2000ep
  artifacts, a **complete per-model + cross-model + analysis figure suite**
  (PNG + PDF + reproducibility JSON). Port/adapt the notebook's plotting
  (~11 `savefig`; canonical figure types: distribution_comparison, acf_comparison
  dual-scale, qq_plot, time_series_comparison, loss_curves, emd_over_training,
  od_reconstruction, stylized_facts_trajectory) FOR EVERY MODEL, plus the
  cross-model comparison + introspection figures. **Completeness bar: match or
  exceed the 20-figure canonical set** in `Final Results from 2000 epochs -
  IQP:SEL circuit/`. Gap analysis required: `revision/` currently has only
  trajectory/introspection + transform-ablation figures — the full per-model
  suite is missing and must be added.

### Manuscript Revision Package

- **D-14-18:** Manuscript source is **external (Overleaf)**. Phase 14 produces a
  revision package; the in-repo `.tex` files are **read-only reference, never
  edited**. PAPER-01..11 edits are delivered as **copy-paste LaTeX blocks keyed
  to section/`\label`/anchor sentence** + a one-line reviewer-comment rationale
  per change.
- **D-14-19:** `revision/docs/reviewer_response.md` — **per-reviewer sections**;
  each row = comment ID → verbatim concern → change made → manuscript location
  (section/table/fig) → supporting artifact (JSON/figure path). Maps to success
  criterion 1 and AIChE point-by-point rebuttal format.
- **D-14-20:** Final framing **tone deferred to paper-writing** once the 2000ep
  numbers land (capture both result directions). **Non-negotiable constraint:**
  PAPER-02 claim-calibration (no overclaiming "quantum advantage" / "industrial
  monitoring") is a LOCKED reviewer requirement regardless of which way the
  numbers fall — it is not a tone choice.

### Release Freeze & Sequencing

- **D-14-21:** Tag **`v2.0-revision`** freezes `revision/` (core, run scripts,
  results JSON, docs, figures) + the `.tex` reference files + `.planning`.
  Excludes `qgan_env/` and large checkpoints (referenced by hash, not committed);
  `data.csv` included (small, needed for reproducibility). Zenodo DOI minted
  from the GitHub release of the tag; `revision/docs/release.md` records tag SHA
  + DOI + reproduce steps.
- **D-14-22:** **Strict gated pipeline, release freeze LAST.** Hard order:
  (1) recover+lock 55-param config → (2) equivalence assert + freeze
  `best_checkpoint.pt` → (3) tiered 2000ep regen behind strict accept gate →
  (4) `model_info.json` + figure suite → (5) reconciliation note → (6) LaTeX
  blocks + `reviewer_response.md` → (7) release freeze + Zenodo DOI. Steps 6–7
  are HARD-BLOCKED until every cited number passes the gate; the DOI mints only
  over final numbers.
- **D-14-23:** **Roadmap scope reconciliation** — this phase was intentionally,
  user-drivenly expanded from "paper edits + freeze" to also recover the lost
  circuit and re-execute Phases 10–13 at 2000ep. Planning should flag that
  `ROADMAP.md` / `REQUIREMENTS.md` be updated to reflect the enlarged scope
  (this is a recorded deviation, not silent scope creep).

### Claude's Discretion

- Exact `model_info.json` long-form schema fields, table column ordering, and
  markdown rendering layout.
- Figure styling, exact subplot composition, file naming, and which legacy
  notebook plot routines to port vs rewrite.
- Checkpoint-introspection + git-archaeology mechanics to identify the 55-param
  (qubits, layers, gate layout) decomposition.
- Sweep driver structure (following the established `run_*.py` + `*_sweep.sh`
  `xargs -P2` resumable pattern).
- Stall-watchdog / subagent-permission settings per the project's compute-heavy
  phase memory.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §"Phase 14" — goal, 5 success criteria, requirements
  PAPER-01..11 + INFRA-03
- `.planning/PROJECT.md` — out-of-scope lock (no variance re-attempt; honest
  reporting; simulator-only); v2.0 target features
- `.planning/REQUIREMENTS.md` — PAPER-* / INFRA-03 requirement text
- `.planning/phases/13-architecture-introspection/13-CONTEXT.md` — ansatz
  variant V1/V2/V3 definitions and D-13-* decisions (V1 = production ansatz)

### Canonical result artifacts (LOST-config recovery — highest priority)
- `best_checkpoint.pt` — **canonical headline artifact**: epoch 1969,
  `params_pqc (55,)`, stored `mu`/`sigma`, optimizer `param_groups`. Ground
  truth for circuit reconstruction.
- `results/run_unconditioned_wgan/stats.json` — canonical run statistics +
  config breadcrumbs; pipeline identification
- `Final Results from 2000 epochs - IQP:SEL circuit/` — 20 preserved canonical
  figures (Figure_2..21); reproduction cross-check target + figure-completeness
  bar
- `qgan_pennylane.ipynb` — notebook git history (45 IQP/StronglyEntanglingLayers
  refs; ~11 `savefig` plot routines to port); corroborating config source

### Training/dataset protocol (regenerate from JSON per D-14-16)
- `revision/docs/training_protocol.md` — HPO hyperparameters (N_CRITIC=9,
  λ=2.16, LR_C=1.8046e-5, LR_G=6.9173e-5, NUM_EPOCHS=2000, batch=12,
  betas=(0,0.9)); EarlyStopping semantics
- `revision/docs/dataset_stats.md` — 778 raw rows → 777 log-returns → 384
  windows; single-campaign LUCY; data hash basis

### Core implementation
- `revision/core/models/quantum.py` — current IQP:SEL impl (Hadamard → IQP RZ →
  IQP noise → Strongly Entangled Layers → final RX/RY); 75-param default;
  `_TOPOLOGIES = ("range","linear")`
- `revision/core/__init__.py` — `NUM_QUBITS=5, NUM_LAYERS=4, WINDOW_LENGTH=10`
- `revision/core/training.py` — training loop, device auto-select
  (cuda→mps→cpu), `compute_dtype` float32-on-mps, `EarlyStopping`, dormant
  `callback(epoch, metrics)` hook
- `revision/results/parity_check.json` — Phase 8 zero-drift parity baseline

### Manuscript (READ-ONLY — never edit in-repo)
- `main (4) copy.tex` — manuscript main source (note literal spaces in filename)
- `supp_material.tex` — supplementary material source

### Execution patterns
- `revision/run_ansatz_sweep.sh`, `revision/run_baselines_sweep.sh`,
  `revision/run_sensitivity_sweep.sh` — `xargs -P2` resumable sweep pattern
- `revision/results/baselines/sweep_status.json` — resumable sweep-state schema

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `revision/run_*.py` + `*_sweep.sh` `xargs -P2` resumable drivers — the
  established pattern for the new 2000ep sweep + `run_model_info.py`
- `sweep_status.json` resumable state (skip-already-done) — reuse for
  run-to-completion / tier resumption
- Dormant `callback(epoch, metrics)` hook in `training.py` — already used by
  Phase 13 introspection; reuse for any per-epoch capture
- Notebook plot routines (`results/run_*/` shows the 8 canonical figure types) —
  port into the new `revision/` figure module
- `revision/run_introspect_figures.py` — existing PNG+PDF+reproducibility-JSON
  figure pattern to follow for the full suite
- Phase-8 parity-check harness — model for the config-equivalence assertion

### Established Patterns
- Long-form `rows[] + models[] aggregate` JSON schema with `data_hash` — the
  `model_info.json` emitter must conform
- `core/` byte-frozen on the default path — the 55-param IQP:SEL must be added
  as a NON-default config-selectable circuit (do not mutate default)
- Strict no-hand-typed-numbers provenance (Phase 10 D-10-19/20) — table +
  figures render from JSON only
- Seed set `{42..46}`, dual-scale (OD + log_return), Phase-09.1 pipelines A/B

### Integration Points
- New 55-param IQP:SEL circuit slots into `quantum.py` alongside existing
  topology/variant selection (config-selectable, like ARCH-01 topology)
- New sweep driver(s) feed the existing eval modules
  (`revision.core.eval`) → regenerated long-form JSON → table + figures
- `revision/docs/{training_protocol,dataset_stats,release,reviewer_response}.md`
  generated/updated from artifacts

</code_context>

<specifics>
## Specific Ideas

- User's originating concern: "quantum results do not look very good because
  they haven't been trained for the full epoch stuff" → root cause found:
  comparison runs were 1000ep vs the 2000ep headline checkpoint, AND the
  proven-good result used a different (55-param IQP:SEL) circuit than the
  current 75-param default. Both are now addressed by D-14-01/04/08/09.
- "The distributions match VERY VERY well" — the
  `Final Results from 2000 epochs - IQP:SEL circuit/` figures are the visual
  quality bar the reproduction must hit.
- "Make sure everything is treated equally" — equal 2000ep budget, identical
  seeds/data-hash/gate across every model (D-14-10/13).
- "I don't want it silently running on CPU while claiming MPS" → device
  manifest + hard-assert (D-14-12); quantum honestly reported as CPU sim.
- "Complete suite of plots/figures from all models and analysis" → D-14-17
  gap-analysis + full figure module.

</specifics>

<deferred>
## Deferred Ideas

- `lightning.qubit` backend acceleration — rejected for this phase (forces
  adjoint diff_method, frozen-core change, full re-baseline). Could be a future
  performance milestone if a re-baseline is ever acceptable.
- Closed-loop decision pipeline & first-principles Hybrid-GAN — remain in
  manuscript Outlook only (PROJECT.md out-of-scope), not implemented.

None — discussion otherwise stayed within (the intentionally expanded) phase
scope.

</deferred>

---

*Phase: 14-paper-revision-release-freeze*
*Context gathered: 2026-05-19*
