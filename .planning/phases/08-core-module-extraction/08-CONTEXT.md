# Phase 8: Core Module Extraction - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Source:** v2.0 milestone scope (conversation-derived — scope locked in QGAN_Review_Response_Plan.md.pdf and finalized in roadmap)

<domain>
## Phase Boundary

This phase extracts the data pipeline, quantum generator, classical critic, WGAN-GP training loop, and evaluation metrics from `qgan_pennylane.ipynb` into importable Python modules under `revision/core/`. It verifies parity: running the main notebook with the imported modules produces numerically identical (within tolerance) evaluation metrics compared to the pre-extraction baseline.

**In scope:**
- Create `revision/core/` package with modules: `data.py`, `eval.py`, `training.py`, `models/quantum.py`, `models/critic.py`
- Replace inline notebook code paths with imports from `revision/core/` where safe (or add a parallel `revision/01_parity_check.ipynb` that imports modules and runs the same pipeline)
- Write a parity-check artifact (`revision/results/parity_check.json`) comparing pre/post metrics

**Explicitly OUT of scope for this phase** (deferred to later phases):
- `models/classical_wgan.py` → Phase 10 (Classical Baselines)
- `models/vae.py` → Phase 10 (Classical Baselines)
- TSTR / predictive-score / discriminative-score code → Phase 11
- Inverse-transform (`inverse_transform`) fully differentiable → Phase 9 (can stub in Phase 8 if notebook needs it for parity)
- Any behavior change — this phase is a REFACTOR ONLY
</domain>

<decisions>
## Implementation Decisions (LOCKED)

### Package Structure
- Root path: `revision/core/` (project root relative)
- `revision/core/__init__.py` exposes top-level API
- `revision/core/models/__init__.py` for submodule
- Modules created in Phase 8: `data.py`, `eval.py`, `training.py`, `models/quantum.py`, `models/critic.py`
- Future modules (stubbed only in Phase 8 if needed for imports): `models/classical_wgan.py`, `models/vae.py`

### Module Responsibilities
- `data.py`: CSV load, preprocessing (log-returns, Lambert W, rolling windows, normalization), train/val/test split, `inverse_transform` (may start as scalar round-trip — full differentiable version is Phase 9)
- `models/quantum.py`: PQC generator class, QNode construction, data re-uploading ansatz, noise sampling, PAR_LIGHT conditioning hook
- `models/critic.py`: 1D-CNN critic class with configurable dropout (preserve v1.1 Phase 7 decision)
- `training.py`: WGAN-GP training loop (N_CRITIC, λ, gradient penalty, spectral loss hook from v1.1 Phase 6, multi-seed support via `seed` kwarg)
- `eval.py`: EMD (wasserstein_distance on raw samples, not histograms — v1.0 decision), ACF, moment statistics (mean, std, kurtosis), DTW, JSD, PSD comparison

### Notebook Orchestration Contract
- No business logic in notebooks — notebooks only: (a) import from `revision/core/`, (b) call high-level functions, (c) plot, (d) write JSON to `revision/results/`
- Main notebook `qgan_pennylane.ipynb` stays where it is but its code cells may be refactored to import from `revision/core/` (minimal edits; do not restructure the notebook)

### Parity Tolerance (Success Criterion)
- EMD: |EMD_pre − EMD_post| ≤ 1e-4
- Moment metrics (mean, std, kurtosis): absolute difference ≤ 1e-6
- Rationale: float32 accumulation plus any reordering of operations may introduce small numerical drift; hard equality is too brittle

### Parity Check Artifact
- Path: `revision/results/parity_check.json`
- Schema: `{"pre": {...metrics...}, "post": {...metrics...}, "delta": {...}, "pass": bool, "tolerance": {...}, "seed": int, "git_sha_pre": str, "git_sha_post": str}`
- At least one seed; ≥5 seed multi-run deferred to Phase 12

### Compute Constraint
- Local Mac statevector simulator only — all Phase 8 work must run locally within a reasonable session (target ≤ 10 min wall time for parity check)

### Quality Constraints
- Modules MUST NOT change observable behavior of main notebook — ONLY move code
- No renaming of existing public symbols unless absolutely required by module boundary
- Preserve HPO hyperparameter defaults from v1.1 Phase 4 (N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05)
- Preserve backprop diff_method (v1.1 Phase 5 decision — parameter-shift has broadcasting bugs)
- Preserve [0, 4π] noise range (v1.1 Phase 4)
- Preserve spectral/PSD loss term (v1.1 Phase 6)
- Preserve dropout-configurable critic (v1.1 Phase 7)

### Claude's Discretion
- Exact file-level breakdown within modules (which functions go in `data.py` vs `eval.py`, etc.)
- Whether to refactor main notebook in place OR add a parallel `revision/01_parity_check.ipynb` that imports modules and produces the parity artifact (either satisfies INFRA-02)
- Whether the training loop exposes a callback API for Phase 13 introspection (nice-to-have; add hook points if cheap, but don't over-engineer)
- Type hints, docstrings, any light code-quality improvement that does not change behavior

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope & Requirements
- `.planning/PROJECT.md` — v2.0 milestone goals and constraints
- `.planning/REQUIREMENTS.md` — v2.0 requirement definitions (INFRA-01, INFRA-02 are Phase 8)
- `.planning/ROADMAP.md` — Phase 8 success criteria (4 observable conditions)

### Source code to extract FROM
- `qgan_pennylane.ipynb` — main experiment notebook (70 cells); all Phase 8 source code lives here
- `data.csv` — the canonical dataset the pipeline consumes

### History / prior decisions that MUST be preserved
- `.planning/MILESTONES.md` — v1.0 + v1.1 decision history
- `.planning/milestones/v1.0-phases/` — v1.0 phase artifacts (for extracted-code provenance)

### Reviewer scope (context, not required for extraction itself)
- `QGAN_Review_Response_Plan.md.pdf` — reviewer feedback driving the overall v2.0 milestone

</canonical_refs>

<specifics>
## Specific Ideas

- The v1.1 notebook already contains clean class boundaries (Generator-like class, Critic class, training function) — extraction should mostly be cut/paste into modules with import statements adjusted.
- `WINDOW_LENGTH = 2 * NUM_QUBITS` invariant must be preserved (v1.0 decision).
- Parity check can use the existing `best_checkpoint.pt` or `best_checkpoint_par_conditioned.pt` as a fixed-state starting point — run one forward pass with deterministic seed and compare EMD + moments before/after extraction, no retraining needed.
- If the notebook uses any globals (e.g., `NUM_QUBITS`, `NUM_LAYERS`, `DEVICE`), those become module constants or function parameters — do NOT replicate notebook-level globals across modules; prefer explicit arguments with sensible defaults.

</specifics>

<deferred>
## Deferred Ideas

- Full differentiable `inverse_transform` — Phase 9 requirement (EVAL-06)
- Classical WGAN-GP generator + VAE/AR modules — Phase 10
- TSTR / predictive / discriminative eval modules — Phase 11
- Shot-count / noise-model parameterization hooks — Phase 12 (Phase 8 may leave simple kwargs if cheap)
- Ansatz variant selector — Phase 13 (Phase 8 should expose the ansatz as a function so Phase 13 can swap it without touching the training loop)
- Multi-seed JSON schema conventions — Phase 12 (Phase 8 parity check uses single seed)

</deferred>

---

*Phase: 08-core-module-extraction*
*Context gathered: 2026-04-23 via conversation + roadmap*
