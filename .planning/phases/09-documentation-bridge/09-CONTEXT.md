# Phase 9: Documentation Bridge - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 9 produces three paper-ready artifacts before any expensive code experiments run, so Phase 14 paper drafting can begin in parallel with Phases 10–13:

1. `docs/training_protocol.md` — Methods-section content covering N_CRITIC, λ, optimizer, both LRs, epochs, early-stopping rule, seeds, shot/analytic distinction (DOC-01).
2. `docs/dataset_stats.md` — dataset characterization: raw OD count, rolling-window count, split convention, campaign count (DOC-02).
3. Differentiable `inverse_transform` in `core/data.py` — log-return + Lambert W back-transform with autograd-flowing gradients, verified ≤1e-8 round-trip (EVAL-06).

**Plus minimal scaffolding for Phase 09.1:** a new `core/preprocessing.py` exposing the `forward_X`/`inverse_X` contract for all three ablation pipelines, with only the Lambert-W pair fully implemented in Phase 9. The other four entries are `NotImplementedError` stubs reserved for Phase 09.1.

**In scope:**
- Implement differentiable Lambert W as in-place replacement of `inverse_lambert_w_transform` in `core/data.py`.
- Add `core/preprocessing.py` skeleton with `forward_logreturns`/`inverse_logreturns`, `forward_lambert`/`inverse_lambert`, `forward_minmax_od`/`inverse_minmax_od` signatures. Lambert pair fully implemented; other four raise `NotImplementedError("Phase 09.1")`.
- Round-trip verification harness asserting max abs error ≤ 1e-8 on (a) `torch.randn` synthetic tensor and (b) full real `log_delta` (776 elements).
- Write `docs/training_protocol.md` and `docs/dataset_stats.md` in hybrid format (tables + 1-paragraph prose justifications).

**Out of scope (deferred to later phases):**
- Pipeline A (raw OD) and Pipeline B (log-returns only) implementations → Phase 09.1.
- Preprocessing ablation runs themselves → Phase 09.1.
- Train/val/test split — explicitly NOT applied (see decision D-01).
- Multi-campaign data handling → Phase 14 Outlook.
- Dataset histograms / OD-level summary statistics beyond reviewer-required counts → could be added if cheap; otherwise Phase 11 EVAL-05.

</domain>

<decisions>
## Implementation Decisions

### Data Convention
- **D-01: No train/val/test split.** All 778 raw OD rows / 777 log_delta entries / 384 rolling windows used for training (live counts verified against `load_and_preprocess` 2026-05-15). Justification: single-campaign dataset is too small to justify a held-out split; aligns with bioprocess single-campaign reality. EMD-based early stopping uses the same distribution as comparison — this is acknowledged as a methodological constraint in dataset_stats.md, in line with R1-M5 calibration honesty.
- **D-02: Single campaign acknowledged in DOC-02.** data.csv = exactly 1 campaign covering 2024-03-27 13:12 → 2024-04-01 23:42 (~5.4 days; 778 raw rows at 10-min cadence), no other campaigns available. dataset_stats.md states this plainly with a 1-paragraph "Single-Campaign Limitation" prose block; multi-campaign generalization is referenced as a Phase 14 Outlook item, not a current scope claim.

### EVAL-06 — Differentiable Inverse Transform
- **D-03: In-place replacement of `inverse_lambert_w_transform`.** No parallel function. Forward output stays at scipy precision (Phase 8 parity = 0.0 delta gives the headroom); backward path becomes a custom `torch.autograd.Function` with the closed-form analytic derivative `dW/dz = W / (z·(1+W))` (implicit-function-theorem identity, well-known for the principal branch).
- **D-04: Round-trip verification covers BOTH synthetic and real inputs.** A single test asserts `max_abs_error(inverse(forward(x)), x) ≤ 1e-8` on (a) a `torch.randn(777, dtype=float64)` synthetic tensor and (b) the full real `log_delta` tensor (777 elements, live count). Synthetic = decoupled correctness; real = data-path correctness.
- **D-04b: Tolerance scope.** The 1e-8 EVAL-06 bound applies to the bare Lambert pair `inverse_lambert_w_transform(lambert_w_transform(x)) ≈ x`. A separate smoke-test verifies `full_denorm_pipeline ∘ load_and_preprocess` round-trips to a looser tolerance (≤ 1e-6) to absorb rolling-window un-stitching and chained-op float accumulation; this looser bound does NOT satisfy EVAL-06 but does protect against pipeline regression.
- **D-05: scipy stays in the forward path, removed from autograd.** `scipy.special.lambertw` is called once inside the `torch.autograd.Function.forward`; the backward path uses only torch ops on the cached `W` value. No new third-party dependencies.

### Phase 09.1 Scaffolding
- **D-06: Add `core/preprocessing.py` in Phase 9.** Exposes the full ablation contract: `forward_logreturns`/`inverse_logreturns`, `forward_lambert`/`inverse_lambert`, `forward_minmax_od`/`inverse_minmax_od`. Phase 9 implements only the Lambert pair (it IS EVAL-06); the other four raise `NotImplementedError("Phase 09.1")` with one-line docstrings describing the expected behavior. Rationale: locks the API contract so Phase 09.1 doesn't refactor mid-ablation.
- **D-07: `core/data.py` keeps existing functions.** No symbol renames, no removals. The differentiable Lambert W implementation lives in `data.py` (where `inverse_lambert_w_transform` currently is); `preprocessing.py` re-exports it under the `inverse_lambert` name to satisfy the unified contract. Single source of truth in `data.py`.

### Documentation Style
- **D-08: Hybrid format — tables for numbers + 1-paragraph prose for justifications.** Both `training_protocol.md` and `dataset_stats.md` follow this pattern. Tables make the numerical content scannable; prose blocks become drop-in copy for Phase 14 Methods sections. Avoids duplicating work in Phase 14 while keeping the docs useful as quick reference.
- **D-09: Numbers traceable to `core/__init__.py`.** Every constant in `training_protocol.md` (N_CRITIC, λ, LRs, NUM_EPOCHS, BATCH_SIZE, NOISE_HIGH, etc.) is sourced from `core/__init__.py`. Doc cites the source file once at the top so a future hyperparameter change has a single place to update.
- **D-10: shot/analytic distinction stated explicitly.** training_protocol.md states clearly: "All Phase 9 results use analytic statevector simulation (PennyLane `default.qubit` with `shots=None`); shot-noise sweeps are reported separately in Phase 12." Addresses R1-M5 calibration concern directly.

### Claude's Discretion
- Exact section ordering inside training_protocol.md and dataset_stats.md.
- Exact wording of the single-campaign limitation paragraph (D-02).
- Whether to include a small "Reproducibility" subsection in training_protocol.md (cite the seed=42 default and `torch.manual_seed` location); add if it fits naturally.
- File-level layout of `preprocessing.py` (one function per pair vs grouped).
- Choice of synthetic tensor dtype/range for round-trip test — float64 + reasonable input distribution that exercises the Lambert W's domain.
- Light docstring + type-hint additions to `data.py` Lambert functions if they don't change behavior.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope & Requirements
- `.planning/PROJECT.md` — v2.0 milestone goals, key decisions (HPO values, backprop, EMD early-stop)
- `.planning/REQUIREMENTS.md` — DOC-01, DOC-02, EVAL-06 definitions; reviewer-comment traceability
- `.planning/ROADMAP.md` §"Phase 9: Documentation Bridge" — 4 success criteria
- `.planning/scratch/09.1-r1-m3-ablation-spec.md` — downstream consumer spec; defines the preprocessing.py contract Phase 9 must satisfy

### Source Code (this phase modifies)
- `core/data.py` — current `inverse_lambert_w_transform` lives here (lines 68–87); `lambert_w_transform` (forward) at lines 90–104; both float64-promoted
- `core/__init__.py` — source of truth for all hyperparameter constants; training_protocol.md cites these
- `core/eval.py` — EMD definition (raw samples, not histograms — v1.0 lock); training_protocol.md references this for early-stopping rule

### Prior Decisions (must preserve)
- `.planning/phases/08-core-module-extraction/08-CONTEXT.md` — Phase 8 quality constraints: HPO defaults preserved, backprop preserved, [0, 4π] noise preserved, no symbol renames
- `.planning/phases/08-core-module-extraction/08-VERIFICATION.md` — Phase 8 parity: EMD delta = 0.0, moments delta = 0.0 (gives headroom for the 1e-8 EVAL-06 spec)

### Reviewer Scope (context, not implementation input)
- `QGAN_Review_Response_Plan.md.pdf` — full reviewer concerns; R1-M4 (training details), R1-m2 (dataset details), R1-m3 (eval scale) are the items Phase 9 directly addresses

### Source Data
- `data.csv` — single-campaign OD time series, 777 rows, 10-min sampling starting 2024-03-27; columns DATE, PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO, OD, DRY, CELL

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/data.py::inverse_lambert_w_transform` (lines 68–87) — current scipy-based implementation. Forward result shape, dtype (float64), and sign-handling behavior must be preserved exactly; only the autograd path is changed.
- `core/data.py::lambert_w_transform` (lines 90–104) — forward (Gaussian → heavy-tail). Already differentiable (pure torch ops). No changes needed; preprocessing.py's `forward_lambert` re-exports this.
- `core/data.py::load_and_preprocess` (lines 187–256) — full v1.1 pipeline; emits `delta`, `mu`, `sigma`, `transformed_norm_log_delta` artifacts that the differentiable inverse must accept unchanged.
- `core/__init__.py` — central constants module (N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-5, LR_GENERATOR=6.9173e-5, NOISE_LOW=0.0, NOISE_HIGH=4π, etc.). training_protocol.md tables import + render these.
- `01_parity_check.ipynb` — Phase 8 parity notebook; provides a template for the round-trip verification cell pattern (load → forward → inverse → assert).

### Established Patterns
- **No business logic in notebooks** — Phase 9 implementation lives in `core/`; any verification cell goes in `02_eval06_roundtrip.ipynb` (or extends `01_parity_check.ipynb`) and only orchestrates + asserts + writes JSON to `results/`.
- **float64 promotion for Lambert W path** — both forward and inverse currently `.double()` the input. The new differentiable inverse must also operate in float64 for the 1e-8 tolerance to be achievable; cast back to caller dtype only at the boundary if needed.
- **HPO constants are import-only** — never redefined inside training scripts; `from revision.core import N_CRITIC, LAMBDA, ...`. training_protocol.md docs follow this lineage explicitly.

### Integration Points
- `core/data.py::full_denorm_pipeline` (lines 134–162) — wraps the inverse Lambert W with the rest of the denormalization stack. This becomes the `inverse_lambert` in `preprocessing.py` (or close to it). Verify the wrapped pipeline is end-to-end differentiable after the Lambert W change.
- Phase 09.1's `core/preprocessing.py` (Phase 9 creates skeleton) — the `forward_lambert`/`inverse_lambert` symbols are the primary public API surface. Phase 09.1 fills in the OD and log-returns variants.
- Future Phases 11, 12: any code that needs OD-scale gradients (TSTR with backprop through inverse, Phase 12 noise gradients on OD) consumes `inverse_lambert` via the contract Phase 9 establishes.

</code_context>

<specifics>
## Specific Ideas

- **Closed-form Lambert W derivative** is well-known: for `W(z) = w` on the principal branch, `dW/dz = w / (z·(1+w))` for `z ≠ 0`. Edge cases: at `z = 0`, `W(0) = 0` and `dW/dz = 1` (use a small-`z` branch with `lim_{z→0} W/(z(1+W)) = 1`). The implementation must handle the `z = 0` case to avoid division-by-zero NaN gradients.
- **Sign handling** in current `inverse_lambert_w_transform`: input `data` can be negative; the function uses `sign(data) * sqrt(W(δ·data²) / δ)`. Both the sign and the chain-rule through `data²` must propagate through autograd correctly — the wrapper's backward must compose `dW/dz · 2δ·data` with the outer `sign(data) / (2·sqrt(W/δ))`.
- **dataset_stats.md content checklist** (from R1-m2 + DOC-02): raw OD count (777), rolling-window count (≈384 with stride 2, WINDOW_LENGTH=10), split convention (none, with justification), campaign count (1, with prose), sampling cadence (10-min), bioreactor type (LUCY photobioreactor), date range (2024-03-27 onwards), PAR_LIGHT context (single-line acknowledgment that conditioning was disabled for unconditioned baseline reporting per the latest run).
- **training_protocol.md content checklist** (from R1-M4 + DOC-01): all HPO constants in a single table; optimizer = Adam(betas=(0.0, 0.9)); early-stopping rule cited from `EarlyStopping` class in notebook (patience=50 eval cycles, warmup=100 epochs, monitors EMD); seed = 42 (and how it's set); shot mode = analytic (statevector); circuit description (5 qubits, 4 layers, IQP encoding + strongly entangled, [0, 4π] noise); critic description (1D-CNN, no dropout / configurable dropout); gradient penalty (two-sided, λ=2.16); training duration (NUM_EPOCHS=2000, BATCH_SIZE=12); reference to `core/__init__.py` as canonical source.

</specifics>

<deferred>
## Deferred Ideas

- **Pipeline A (raw OD) and Pipeline B (log-returns only) implementations** — Phase 09.1 (ABL-01).
- **Multi-seed run framework** — Phase 09.1 builds it for the 3-pipeline ablation; Phase 12 generalizes for shot/noise sweeps (SENS-03).
- **Dataset histograms / OD-level moments** in dataset_stats.md — could be added if cheap; otherwise covered by Phase 11 EVAL-05's two-scale reporting.
- **Multi-campaign data pipeline** — Phase 14 Outlook section.
- **Differentiable forward Lambert W** — `lambert_w_transform` is already pure-torch and differentiable; no work needed.
- **Reproducibility section in training_protocol.md** — Claude's discretion to include if it fits naturally; otherwise defer to Phase 14.
- **Shot-noise / analytic distinction quantitative reporting** — qualitative statement only in Phase 9; quantitative sweep is Phase 12 (SENS-01).

</deferred>

---

*Phase: 09-documentation-bridge*
*Context gathered: 2026-05-08 via /gsd-discuss-phase 9*
