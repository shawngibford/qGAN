# Phase 12: Sensitivity Analysis - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the **calibrated-uncertainty evidence** AIChE reviewers R1-M4 ("training details / preliminary") and R2-1 ("no comparison / preliminary") demand. It stress-tests the **already-trained quantum generator** under shot noise and hardware-style noise channels, and consolidates seed variation into mean ± std across every headline result. It does **not** retrain any generator.

Concretely, Phase 12 produces three artifacts:

1. **Shot-noise sensitivity** (SENS-01) — fidelity-metric degradation at shots ∈ {analytic, 8192, 1024} → `revision/results/shot_noise_sensitivity.json`
2. **Noise-model sensitivity** (SENS-02) — depolarizing p ∈ {0, 0.001, 0.01, 0.05} and amplitude-damping γ ∈ {0, 0.001, 0.01, 0.05} → `revision/results/noise_model_sensitivity.json`
3. **Multi-seed roll-up** (SENS-03) — every headline comparison table from Phases 10–11 re-emitted with ≥5 seeds, mean ± std in every cell → `revision/results/multiseed_summary.json`

**In scope:** inference-time noise/shot evaluation of the trained analytic quantum generator (regenerate samples under noisy/finite-shot devices, recompute the existing fidelity suite); aggregation of existing Phase 10/11 per-seed artifacts into mean ± std roll-up tables; new `revision/run_*.py` + `*_sweep.sh` driver(s) following the Phase 10 pattern; JSON emission on the established long-form schema.

**Out of scope (other phases own these):**
- Ansatz comparison / training-progression / parameter-trajectory / entanglement figures → Phase 13 (ARCH-01..02, INTRO-01..03)
- CR-01 (spectral-loss hook non-differentiable) / CR-02 (EarlyStopping checkpoint restore) — training-loop bugs locked to Phase 13 by the Phase 11 decision
- Manuscript integration of uncertainty bars → Phase 14 (PAPER-*)
- Any generator retraining or new model families
- New variance-collapse remediation (v2.0 reports honestly, does not re-attempt)

**Why Phase 12 exists separately:** R1-M4 and R2-1 are headline rebuttal points distinct from the bare comparison table (Phase 10) and the utility suite (Phase 11). Owning sensitivity in its own phase keeps the uncertainty story self-contained and respects the local-Mac compute budget by separating sensitivity sweeps from the Phase 13 architecture sweeps.
</domain>

<decisions>
## Implementation Decisions

### Noise / shot application point (LOCKED)
- **D-12-01:** Shot-noise and noise-channel sensitivity are **inference-only** — regenerate samples from the **already-trained analytic quantum generator** (Phase 09.1/10 checkpoints) on a noisy/finite-shot device, then recompute the existing fidelity metric suite. **No retraining.** Rationale: the generator runs `qml.device("default.qubit", shots=None, diff_method="backprop")`; finite shots and noise channels are incompatible with statevector backprop, and retraining the full noise × shot × seed grid under parameter-shift on a local Mac is infeasible. This also yields the correct reviewer narrative: robustness of a fixed trained model to deployment-time noise.

### Sensitivity-grid seed budget (LOCKED)
- **D-12-02:** SENS-01 and SENS-02 **degradation grids** use **3 seeds {42, 43, 44}** (degradation is a trend, not a headline number — 3 seeds give an adequate spread band at the lowest compute). The full **5-seed set {42, 43, 44, 45, 46}** with mean ± std is reserved for the SENS-03 headline roll-up.

### SENS-03 scope (LOCKED)
- **D-12-03:** SENS-03 is **pure aggregation** of the existing Phase 10/11 per-seed artifacts (`baseline_comparison.json`, `tstr.json`, `predictive_discriminative.json`, `augmentation.json`, `fidelity_dualscale.json`) into mean ± std cells. **No new training, no new seeds** — the 5-seed per-seed data already exists in tree. `multiseed_summary.json` is the consolidated roll-up; data-hash invariant (D-10-15) must be asserted across consumed artifacts.

### Claude's Discretion
Per the user's standing guidance to minimize process on technical phases, the following are fully Claude's discretion (locked by prior patterns, no user opinion needed):
- Noise-channel device wiring (e.g., `default.mixed`, channel insertion strategy, finite-shot device construction) and how the trained analytic params are loaded into the noisy QNode.
- Output JSON structure beyond the established long-form schema `{model_kind, pipeline, seed, metric_name, scale, value}`; degradation-curve representation in `shot_noise_sensitivity.json` / `noise_model_sensitivity.json`.
- New driver/sweep file names and CLI surface (pattern after `revision/run_baselines.py` + `run_baselines_sweep.sh`); idempotent per-cell skip logic; `--parallel 2` guardrail; **no `multiprocessing.Pool`** (Phase 09.1 Pitfall 4).
- Which fidelity metrics are recomputed under noise (reuse `revision/core/eval.py` helpers unchanged; EMD/moments/ACF/DTW at minimum, dual-scale per EVAL-05 convention).
- Pipeline coverage for the noise/shot grid (Pipeline B headline; Pipeline A as supplementary control, mirroring Phase 10/11).
- Subsampling strategy if regenerated sample counts differ from the analytic artifacts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` (Phase 12 entry, lines ~170–182) — Goal + Success Criteria 1–4 (the verifiable contract)
- `.planning/REQUIREMENTS.md` — SENS-01, SENS-02, SENS-03 definitions; R1-M4 / R2-1 rebuttal mapping
- `.planning/PROJECT.md` — Locked constraints (local-Mac statevector compute, results-JSON contract, main-notebook-untouched, no new variance-collapse remediation), Key Decisions log

### Upstream artifact contracts (Phase 12 consumes / extends these)
- `.planning/phases/10-classical-baselines/10-CONTEXT.md` — Run-dir layout (D-10-14), data-hash invariant (D-10-15), long-form comparison schema (D-10-16/17), identical-conditions invariant (D-10-08), code-placement invariant (D-10-13), sweep-driver pattern (D-10-22/23/24)
- `.planning/phases/11-utility-evaluation/11-CONTEXT.md` — Headline artifact set + dual-scale (EVAL-05) convention, no-regeneration invariant (D-11-08), driver-placement pattern (D-11-10)
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-CONTEXT.md` — Identical-conditions invariant, Pipeline A/B definitions, eval.py contract
- `revision/results/baseline_comparison.json` — Headline table SENS-03 rolls up (long-form `{model_kind, pipeline, seed, metric_name, scale, value}`)
- `revision/results/tstr.json`, `revision/results/predictive_discriminative.json`, `revision/results/augmentation.json`, `revision/results/fidelity_dualscale.json` — Phase 11 headline tables SENS-03 rolls up
- `revision/results/baselines/runs/<model>/<pipeline>/<seed>/` — Per-seed artifacts SENS-03 aggregates; quantum analytic checkpoints SENS-01/02 reload (no regeneration of analytic baseline)

### Reusable code
- `revision/core/models/quantum.py` — `QuantumGenerator`; `dev = qml.device("default.qubit", shots=None, diff_method="backprop")` is the constraint behind D-12-01; the noisy/finite-shot device is constructed around this for inference
- `revision/core/eval.py` — Fidelity helpers (`compute_emd/moments/acf/dtw/jsd/psd`, `full_metric_suite`) — reuse unchanged; dual-scale `scale` wrapper per EVAL-05
- `revision/core/preprocessing.py` — `inverse_minmax_od`, `inverse_logreturns` for OD-scale reconstruction
- `revision/run_baselines.py` + `revision/run_baselines_sweep.sh` — Reference template for the new Phase 12 driver(s)

### External
- PennyLane 0.44.0 noise documentation — `default.mixed`, `qml.DepolarizingChannel`, `qml.AmplitudeDamping`, finite-shot devices (researcher to pin exact API for the trained-params-into-noisy-QNode path)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `revision/core/eval.py::full_metric_suite` — full fidelity suite already exists; Phase 12 only recomputes it on noise-perturbed samples, no new metric math.
- `revision/run_baselines.py` + `run_baselines_sweep.sh` — idempotent per-cell driver + atomic `sweep_status.json` + `--parallel 2` guardrail; Phase 12 driver(s) follow this exact shape.
- Phase 10/11 per-seed artifact bundles (50 baseline run dirs + quantum runs) — already carry 5-seed data + data-hash; SENS-03 reads and aggregates, never regenerates.

### Established Patterns
- Long-form metrics schema `{model_kind, pipeline, seed, metric_name, scale, value}` — Phase 12 outputs extend it (add a `shots` / `noise_model` / `noise_level` dimension), not replace it.
- Code-placement invariant (D-10-13): `revision/core/` = model + eval helpers only; all noise-sweep orchestration and aggregation in new `revision/run_*.py`.
- No `multiprocessing.Pool` — xargs `-P 2` OS-process parallelism only (Phase 09.1 Pitfall 4).

### Integration Points
- `shot_noise_sensitivity.json`, `noise_model_sensitivity.json`, `multiseed_summary.json` join the `revision/results/*.json` contract Phase 14 paper-writing reads.
- Data-hash field (D-10-15) is the cross-phase consistency check — Phase 12 asserts it matches across every consumed Phase 10/11 artifact before rolling up.
</code_context>

<specifics>
## Specific Ideas

- The quantum device constraint (`shots=None`, `diff_method="backprop"`) is the technical fact that makes inference-only the only feasible AND most reviewer-defensible noise-sensitivity design — the paper claims robustness of a fixed trained model to deployment-time shot/hardware noise, not noise-aware retraining.
- Two-tier seed strategy: cheap 3-seed trend bands for the degradation grids; full 5-seed mean ± std only where the manuscript prints headline numbers (SENS-03).
- Variance collapse (fake std ≈ 48% of real) remains a known, accepted limitation — Phase 12 reports uncertainty honestly, it does not attempt to close the gap.
</specifics>

<deferred>
## Deferred Ideas

- **Noise-aware retraining** — training the generator under finite shots / noise channels (would require parameter-shift + a large compute budget). Out of scope for v2.0; capture as a v3.0 robustness study if reviewers specifically ask for noise-resilient training.
- **Full 5-seed degradation grids** — if the 3-seed SENS-01/02 trend bands show high inter-seed spread that obscures the degradation trend, escalate to 5 seeds on the affected grid points as a planning-time decision (not a default).

### Reviewed Todos (not folded)
- **Fix CR-01 — spectral-loss hook non-differentiable + device-unsafe** — weak generic-keyword match only; it is a training-loop bug locked to Phase 13 by the Phase 11 decision. Phase 12 is inference-only and does not touch the training loop. Not folded.
- **Fix CR-02 — EarlyStopping checkpoint restore device/dtype-inconsistent** — same: training-loop bug, Phase 13. Phase 12 reloads trained analytic params for inference, does not exercise EarlyStopping restore. Not folded.

</deferred>

---

*Phase: 12-sensitivity-analysis*
*Context gathered: 2026-05-18*
