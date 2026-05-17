# Phase 10: Classical Baselines — Context

**Gathered:** 2026-05-17
**Status:** Ready for research / planning
**Source:** Interactive discussion 2026-05-17 + Phase 09.1 recommendation

<domain>
## Phase Boundary

This phase delivers the **matched-parameter quantum-vs-classical evidence** that AIChE reviewers R1-M1 ("no classical baseline shown") and R2-1 ("preliminary; no comparison") explicitly demand. The output is a side-by-side comparison table — quantum / classical WGAN-GP / non-adversarial — across an identical training protocol, identical data pipeline, identical seed set, and identical fidelity metric suite.

**In scope:**
- Implement **3 classical WGAN-GP generator variants** (MLP, 1D-CNN, RNN/LSTM) — each matched within ±5% of the PQC's 75 trainable parameters (target range 71–79).
- Implement **VAE** and **AR** non-adversarial baselines on the same data.
- Train every model under the **same conditions as Phase 09.1**: 5 seeds {42, 43, 44, 45, 46}, 1000 epochs, identical N_CRITIC/LAMBDA/LR/BATCH_SIZE, statevector / analytic gradients (no shot noise — Phase 12 owns that).
- Train across **2 data pipelines** (A = min-max OD, B = log-returns) per Phase 09.1's analysis (Pipeline B is the recommended pipeline; Pipeline A is the control showing classical can/can't handle raw OD).
- Emit `revision/results/baseline_classical_wgan.json`, `revision/results/baseline_nonadversarial.json`, and `revision/results/baseline_comparison.json` + markdown table.
- All artifacts carry a data-hash field so reviewers can verify identical splits across models.

**Out of scope (other phases own these):**
- Full TSTR / predictive / discriminative score suite → Phase 11 (EVAL-01..05).
- Shot-noise / noise-model sweeps → Phase 12 (SENS-01, SENS-02).
- Ansatz comparison / training-progression / parameter-trajectory figures → Phase 13 (ARCH-01, INTRO-01..03).
- Manuscript integration of the comparison table → Phase 14 (PAPER-01).

**Why Phase 10 exists separately:** R1-M1 and R2-1 are the headline reviewer rebuttal points. Owning baselines in their own phase keeps the comparison table self-contained and lets Phase 11+ build evaluation infrastructure on top of a stable, identical-protocol baseline set.
</domain>

<decisions>
## Implementation Decisions

### Scope and model set (LOCKED)

- **D-10-01:** Phase 10 trains **5 NEW model types** in addition to the existing quantum generator:
  - **Classical WGAN-GP variants** (3): MLP, 1D-CNN, RNN/LSTM
  - **Non-adversarial baselines** (2): VAE, AR
- **D-10-02:** Each classical WGAN-GP variant is matched within **±5% of the PQC's 75 trainable params** (target range 71–79). The "matched-parameter" framing is BASE-01's literal requirement — all three variants share this constraint so the comparison is apples-to-apples on capacity.
- **D-10-03:** The non-adversarial baselines (VAE, AR) are NOT param-matched to the PQC — they are conceptually different model families. Their parameter counts are reported transparently in `baseline_comparison.json` for full disclosure. VAE is sized to be the "smallest deep VAE that trains stably" on length-10 windows; AR is parameter-minimal by definition (order-p coefficients).
- **D-10-04:** The **quantum generator** is the reference. Its 5-seed × 2-pipeline (A and B) runs from Phase 09.1 are reused as-is — no quantum retraining in Phase 10. Phase 10's sweep only trains the 5 new model types.

### Data pipelines (LOCKED — per user)

- **D-10-05:** Models train on **BOTH Pipeline A (min-max OD) and Pipeline B (log-returns standardized)**. Pipeline C (Lambert W) is excluded — Phase 09.1 recommended dropping it.
- **D-10-06:** Pipeline B is the **headline pipeline** for the comparison table — it's the recommended pipeline from Phase 09.1 and the strongest fidelity story for the manuscript. Pipeline A results are reported as supplementary "raw OD" controls in the same comparison table.
- **D-10-07:** Each model trains on the **same windowed data** from `revision/core/data.py::load_and_preprocess` and `rolling_window(WINDOW_LENGTH=10, stride=2)`. Pipeline routing reuses the per-pipeline forward/inverse helpers from `revision/core/preprocessing.py` (already in tree, ABL-01 verified to round-trip within 1e-8).

### Identical-conditions invariant (LOCKED — D-10-08 is BASE-01's core)

- **D-10-08:** **Identical training conditions across all classical WGAN-GP variants and the quantum reference:**
  - 5 seeds: {42, 43, 44, 45, 46} (same as Phase 09.1)
  - 1000 epochs per (model, pipeline, seed)
  - N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, BATCH_SIZE=12, WINDOW_LENGTH=10 — all from `revision/core/__init__.py`
  - **Same critic** (`revision/core/models/critic.py::Critic`) for every WGAN-GP variant (BASE-01 explicitly requires "identical critic architecture")
  - Same Adam betas, same gradient-penalty formulation, same windowed-data loader
- **D-10-09:** VAE and AR have model-family-specific training protocols (VAE uses ELBO loss; AR uses MLE or least-squares fit) but the **data, seeds, epoch budget, and held-out evaluation split** are matched to the WGAN-GP track. Documented per-model in their respective JSON outputs.
- **D-10-10:** The 5 × 2 × 5 = 50 new training runs at ~220s/run (matching Phase 09.1's observed wall time for the quantum path; classical paths likely faster) should complete in ≈ 110 minutes at `--parallel 2`. Total sweep wall budget: ≤ 3 hours. If a classical variant trains > 5× faster than quantum, the budget can be relaxed.

### Code placement (LOCKED)

- **D-10-11:** **New file `revision/core/models/classical.py`** holds all 3 classical WGAN-GP generator variants as `nn.Module` subclasses with a shared `count_params()` method matching `QuantumGenerator.count_params()`. The file is the analog of `quantum.py` for the classical generators.
- **D-10-12:** **New file `revision/core/models/nonadversarial.py`** holds the VAE and AR baselines. Mixed-paradigm model families live together here to keep the file count down — both are trained outside the WGAN-GP loop.
- **D-10-13:** Per Phase 09.1's D-09.1-19 invariant, all training-loop integration and aggregation logic stays out of `revision/core/` — only model definitions go there. Training and sweep orchestration code lives in `revision/run_baselines.py` (new) and `revision/run_baselines_sweep.sh` (new), patterned after `run_ablation.py` and `run_ablation_sweep.sh`.

### Run-directory layout (LOCKED — extends Phase 09.1's schema)

- **D-10-14:** Sweep outputs go under `revision/results/baselines/runs/<model_kind>/<pipeline>/<seed>/`. Model kinds: `wgan_mlp`, `wgan_cnn`, `wgan_lstm`, `vae`, `ar`. Each run dir contains the same 5-file artifact bundle as Phase 09.1: `config.yaml`, `checkpoint.pt` (or `.npz` for AR), `samples.npy`, `metrics.json`, `inverse_kwargs.npz`.
- **D-10-15:** A **data-hash field** is written to every `config.yaml` — computed as `sha256(real_OD.tobytes())[:16]`. The hash must match across all 50 new runs AND across the quantum runs from Phase 09.1, proving identical data splits (BASE-01 / BASE-03 requirement).

### Comparison table (LOCKED — BASE-03)

- **D-10-16:** `revision/results/baseline_comparison.json` aggregates every model × pipeline × seed combination into a long-form schema mirroring Phase 09.1's metrics.csv:
  - `{model_kind, pipeline, seed, metric_name, scale, value}`
  - Plus a top-level `models[]` array with `{kind, parameter_count, family, train_protocol_notes}`.
- **D-10-17:** A companion `revision/results/baseline_comparison.md` renders the JSON as a markdown table with one row per model, columns for parameter count, OD-EMD (mean ± std), OD-ACF lag-1, OD-DTW mean, transformed-space EMD (Pipelines B), and TSTR-lite R² (sanity scaffolding — Phase 11 owns full TSTR).
- **D-10-18:** The table reports BOTH the quantum reference (5 seeds × 2 pipelines from Phase 09.1) AND every new model on the same pipeline rows. So the reader sees quantum-vs-WGAN-MLP-vs-WGAN-CNN-vs-WGAN-LSTM-vs-VAE-vs-AR for both Pipeline A and Pipeline B.
- **D-10-19:** **No new recommendation** in this phase. Phase 14 (PAPER-01) decides which baseline to highlight in the manuscript narrative based on Phase 11's full TSTR + utility numbers. Phase 10 just delivers the apples-to-apples comparison table.

### Evaluation metrics (LOCKED — reuse Phase 09.1)

- **D-10-20:** Every model emits the same per-run fidelity metric set as Phase 09.1's ablation runs: OD-scale EMD, moments (mean/std/skew/kurtosis), per-lag ACF mean+std (lags 0..9), DTW mean/median/std on nearest-neighbor sub-sample, and transformed-space EMD where applicable. All via `revision/core/eval.py` — no new metric helpers.
- **D-10-21:** TSTR-lite (1-layer LSTM-32, 3 init seeds, 320 held-out real windows) is run per model × pipeline as a sanity-scaffolding check. Numbers go into `baseline_comparison.json` and the markdown table; Phase 11 owns the full multi-architecture TSTR.

### Sweep orchestration (LOCKED — pattern from Phase 09.1)

- **D-10-22:** `revision/run_baselines.py` is the per-(model, pipeline, seed) CLI driver: `python -m revision.run_baselines --model {wgan_mlp,wgan_cnn,wgan_lstm,vae,ar} --pipeline {A,B} --seed N --epochs M`. Idempotent — re-running overwrites the run directory cleanly.
- **D-10-23:** `revision/run_baselines_sweep.sh` loops 5 model kinds × 2 pipelines × 5 seeds = 50 pairs, skips already-complete pairs (same `is_complete()` 5-file check as 09.1), writes `sweep_status.json`, supports `--parallel {1,2}` guardrail. Same atomic-status-writer pattern as `run_ablation_sweep.sh`.
- **D-10-24:** Per RESEARCH Pitfall 4 from Phase 09.1: **never `multiprocessing.Pool`**. xargs -P 2 OS-process parallelism only.

</decisions>

<canonical_refs>
## Canonical References

These docs MUST be consulted by downstream research/planning agents:

- `.planning/PROJECT.md` — Project goals, locked constraints, key decisions log
- `.planning/REQUIREMENTS.md` — BASE-01, BASE-02, BASE-03 definitions and rebuttal mapping (R1-M1, R2-1)
- `.planning/ROADMAP.md` (Phase 10 entry) — Success criteria 1–4
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-CONTEXT.md` — Identical-conditions invariant pattern (D-09.1-04), pipeline definitions, eval.py contract
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-04-SUMMARY.md` — Pipeline B recommendation rationale, TSTR-lite spec, conditional gate result
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-RESEARCH.md` — Pitfall 4 (no multiprocessing.Pool); sweep timing estimates
- `revision/core/models/quantum.py` — `QuantumGenerator.count_params()` returns 75; the target the classical generators match
- `revision/core/models/critic.py` — `Critic` class; reused unchanged for every WGAN-GP variant
- `revision/core/__init__.py` — All HPO constants (N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, BATCH_SIZE=12, WINDOW_LENGTH=10, NUM_EPOCHS=2000, EVAL_EVERY=10)
- `revision/core/training.py::train_wgan_gp` — Shared WGAN-GP training loop; the 3 classical variants call this with their own generator
- `revision/core/preprocessing.py` — `forward_minmax_od`, `inverse_minmax_od`, `forward_logreturns`, `inverse_logreturns` (ABL-01 verified)
- `revision/run_ablation.py` — Reference template for the new `revision/run_baselines.py` CLI driver
- `revision/run_ablation_sweep.sh` — Reference template for the new `revision/run_baselines_sweep.sh` sweep driver
</canonical_refs>

<scope>
## What's IN

- 3 classical WGAN-GP variants (MLP, 1D-CNN, RNN/LSTM) — matched ±5% to PQC's 75 params
- 2 non-adversarial baselines (VAE, AR) — sized to their natural minimum
- 2 data pipelines (A, B) — Pipeline C dropped per Phase 09.1 recommendation
- 5 seeds × 1000 epochs (matches Phase 09.1 invariant)
- New code: `revision/core/models/classical.py`, `revision/core/models/nonadversarial.py`, `revision/run_baselines.py`, `revision/run_baselines_sweep.sh`
- New artifacts: `revision/results/baseline_classical_wgan.json`, `revision/results/baseline_nonadversarial.json`, `revision/results/baseline_comparison.{json,md}`, `revision/results/baselines/runs/<model>/<pipeline>/<seed>/` (50 directories)

## What's OUT (deferred)

- Full TSTR / predictive / discriminative scores → Phase 11 (EVAL-01..05)
- Shot-noise / noise-model sweeps → Phase 12 (SENS-01, SENS-02)
- Ansatz variants / training-progression / introspection figures → Phase 13
- Manuscript integration → Phase 14 (PAPER-01)
- Final "which baseline to highlight" decision → Phase 14 (driven by Phase 11's utility numbers)
- Pipeline C (Lambert W) — explicitly dropped per Phase 09.1 recommendation
</scope>

<deferred_ideas>
## Deferred Ideas (Backlog Candidates)

- **Param-matched VAE / AR** — Both non-adversarial baselines are sized naturally rather than matched to 75 params. If reviewers later ask for a strictly param-matched non-adversarial baseline, that becomes a follow-up phase.
- **GAN-style hybrid (classical generator + quantum critic)** — Inverse of the current quantum-generator / classical-critic split. Not in scope for v2.0; capture for a future v3.0 ablation if interest emerges.
- **Larger ansatz variants** — Phase 13 owns ansatz exploration. If a 6-qubit or 6-layer PQC ends up as the publication choice, Phase 10's matched-param targets shift accordingly and would need re-running.

</deferred_ideas>

<next_steps>
## Next Steps

1. Phase research (`/gsd-research-phase 10`) — survey classical-WGAN architecture standards for matched-capacity small-window time-series generation; survey VAE / AR sizing for length-10 univariate windows.
2. Phase planning (`/gsd-plan-phase 10`) — break into ≥3 plans following Phase 09.1's wave pattern (architectures → CLI driver / smoke → sweep → comparison table).
3. Phase execution (`/gsd-execute-phase 10`) — execute the plans wave-by-wave; ~3-hour total compute budget at `--parallel 2`.
</next_steps>
