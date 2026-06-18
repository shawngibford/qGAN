# Phase 11: Utility Evaluation - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the **utility-oriented evidence** AIChE reviewers R1-M2 ("utility-oriented tests") and R2-4 ("improves vs. what?") demand. It builds evaluation infrastructure on top of the stable, identical-protocol artifacts produced by Phases 09.1 and 10 — it does **not** retrain any generator.

Concretely, Phase 11 produces three downstream-task verdicts for the quantum generator vs. the classical WGAN-GP variants vs. the non-adversarial baselines:

1. **TSTR soft-sensor** (EVAL-01) — train a soft-sensor on synthetic OD windows, evaluate on held-out real data; report R²/MAE/RMSE → `results/tstr.json`
2. **TimeGAN predictive + discriminative scores** (EVAL-02/03) with mean ± std across seeds → `results/predictive_discriminative.json`
3. **Real-only vs. synthetic-augmented lift** (EVAL-04, Orlandi-style) → `results/augmentation.json`
4. **Both-scale fidelity reporting** (EVAL-05) — every fidelity metric carries an explicit `scale: "log_return" | "OD"` field

**In scope:** TSTR pipeline; faithful TimeGAN post-hoc predictive/discriminative nets; mixing-ratio augmentation study; dual-scale JSON emission; consumption of existing Phase 10 (`results/baselines/runs/...`) and Phase 09.1 quantum sample artifacts.

**Out of scope (other phases own these):**
- Shot-noise / noise-model / multi-seed roll-up sweeps → Phase 12 (SENS-01..03)
- Ansatz comparison / training-progression / introspection figures → Phase 13 (ARCH/INTRO)
- Manuscript integration of utility numbers → Phase 14 (PAPER-*)
- Any generator retraining or new model families
- CR-01 (spectral-loss hook) / CR-02 (EarlyStopping checkpoint restore) — training-loop bugs already deferred to Phase 13

**Why Phase 11 exists separately:** R2-4 ("improves vs. what?") and R1-M2 (utility tests) are headline rebuttal points distinct from the bare comparison table (Phase 10). Owning the full TSTR/score/augmentation suite in its own phase keeps the utility story self-contained and lets it consume a frozen, identical-protocol baseline set.
</domain>

<decisions>
## Implementation Decisions

### TSTR soft-sensor task (EVAL-01)
- **D-11-01:** The soft-sensor task is **one-step-ahead OD forecasting**: given the preceding OD window `OD[t-k..t]`, predict `OD[t+1]`. No PAR_LIGHT conditioning required; this is the canonical TimeGAN/TSTR setup and gives the cleanest, lowest-variance comparison across quantum + all baselines and the most direct alignment with the predictive score (EVAL-02).
- **D-11-02:** TSTR protocol — train the soft-sensor on **synthetic** windows, evaluate on **held-out real** windows. Report R², MAE, RMSE. Reuse the held-out real split convention already established by Phase 10's TSTR-lite (D-10-21, 320 held-out real windows) so numbers are comparable.

### TimeGAN predictive & discriminative scores (EVAL-02/03)
- **D-11-03:** Use **faithful TimeGAN post-hoc networks**, not the lightweight Phase 10 TSTR-lite scaffolding. Canonical definitions:
  - **Predictive score** = MAE of a post-hoc sequence predictor trained on synthetic, tested on real (TRTS), next-step prediction. Lower is better.
  - **Discriminative score** = `|0.5 − test_accuracy|` of a post-hoc real-vs-synthetic classifier. Lower is better.
- **D-11-04:** Post-hoc nets follow the canonical TimeGAN architecture (GRU-based; hidden dim ≈ input_dim, ~1–2 recurrent layers). Exact hyperparameters are a research item for the planner/researcher to pin against the reference implementation — not user-locked.
- **D-11-05:** Scores reported as **mean ± std across the 5-seed set {42,43,44,45,46}** (ROADMAP success criterion 2), reusing the existing per-seed sample artifacts.

### Augmentation study (EVAL-04, Orlandi-style)
- **D-11-06:** **Mixing-ratio sweep**, not a single augmented condition. Downstream task = the **same one-step-ahead OD soft-sensor** as D-11-01 (consistency: one downstream task drives both TSTR and augmentation).
- **D-11-07:** Conditions: `real-only` baseline, then `real + synthetic` at multiple injection ratios producing a **lift curve per generator** (suggested grid `{+25%, +50%, +100%, synthetic-only}`; exact grid is planner discretion). Delta table = downstream R²/MAE/RMSE change vs. the real-only baseline, per generator.

### Sample provenance
- **D-11-08:** **Reuse Phase 10 / Phase 09.1 artifacts as-is.** Read existing `samples.npy` from the 50 Phase 10 baseline run dirs (`results/baselines/runs/<model>/<pipeline>/<seed>/`) plus the Phase 09.1 quantum runs. **No regeneration, no retraining** — preserves the identical-protocol invariant; Phase 11 adds evaluation code only.
- **D-11-09:** Both **Pipeline A and Pipeline B** are evaluated (matching Phase 10's comparison-table coverage). Pipeline B remains the headline pipeline (D-10-06); Pipeline A reported as the supplementary raw-OD control. EVAL-05 dual-scale (`log_return` + `OD`) emission applies to every metric.

### Code placement (carried forward from D-10-13 invariant)
- **D-11-10:** Evaluation/aggregation logic stays **out of `core/`** (which holds model definitions + `eval.py` fidelity helpers only). New TSTR/score/augmentation orchestration lives in new `run_*.py` driver(s) + JSON emitters, patterned after `run_baselines.py`. Reuse `core/eval.py` fidelity helpers unchanged for EVAL-05 dual-scale reporting.

### Claude's Discretion
- Exact post-hoc GRU hyperparameters (depth, hidden dim, epochs) — pin to the cited TimeGAN reference implementation during research.
- Even though the user selected "faithful" (not "faithful + cite"), **pin and record the reference implementation (ydata-synthetic / original TimeGAN repo) in the JSON metadata** — it is zero-cost and strictly more defensible to reviewers. Captured as discretion, not a re-ask.
- Soft-sensor architecture (1D-CNN vs LSTM per EVAL-01's "or") — planner selects; a single architecture used consistently across all generators is preferred over comparing two.
- Augmentation mixing-ratio grid resolution.
- TSTR/score sample sizes drawn from the existing artifacts (subsampling strategy if artifacts are larger/smaller than needed).

### Reviewed Todos (not folded)
- **Fix CR-01 — spectral-loss hook non-differentiable + device-unsafe** — training-loop bug, already deferred to Phase 13 (last commit `50c2dc7`). Out of Phase 11 utility-evaluation scope.
- **Fix CR-02 — EarlyStopping checkpoint restore device/dtype-inconsistent** — same: training-loop bug, Phase 13. Out of scope here.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` (Phase 11 entry) — Goal + Success Criteria 1–4 (the verifiable contract)
- `.planning/REQUIREMENTS.md` — EVAL-01..05 definitions; R1-M2 / R2-4 rebuttal mapping
- `.planning/PROJECT.md` — Locked constraints (local-Mac compute, results-JSON contract, main-notebook-untouched), Key Decisions log

### Upstream artifact contracts (Phase 11 consumes these)
- `.planning/phases/10-classical-baselines/10-CONTEXT.md` — Run-dir layout (D-10-14), data-hash invariant (D-10-15), comparison-table schema (D-10-16/17), TSTR-lite spec (D-10-21), code-placement invariant (D-10-13)
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-CONTEXT.md` — Identical-conditions invariant, Pipeline A/B definitions, eval.py contract
- `.planning/phases/09.1-r1-m3-preprocessing-ablation/09.1-04-SUMMARY.md` — Pipeline B recommendation rationale; held-out real split / TSTR-lite numbers
- `results/baseline_comparison.json` — Existing long-form metrics schema to extend (`{model_kind, pipeline, seed, metric_name, scale, value}`)
- `results/baselines/runs/<model>/<pipeline>/<seed>/samples.npy` — The synthetic samples Phase 11 evaluates (no regeneration)

### Reusable code
- `core/eval.py` — Fidelity helpers (`compute_emd/moments/acf/dtw/jsd/psd`, `full_metric_suite`) — reuse unchanged; EVAL-05 wraps each with a `scale` field
- `core/preprocessing.py` — `inverse_minmax_od`, `inverse_logreturns` for OD-scale reconstruction (ABL-01 verified round-trip)
- `core/data.py` — `load_and_preprocess`, `rolling_window`, EVAL-06 differentiable `inverse_transform`
- `run_baselines.py` — Reference template for the new Phase 11 driver(s)

### External
- TimeGAN reference implementation (ydata-synthetic / original Yoon et al. repo) — canonical predictive/discriminative score definitions; researcher to locate and pin exact version
- Orlandi et al. [26] (AIChE) — augmentation-study methodology reference for EVAL-04
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/eval.py::full_metric_suite` — already computes EMD/moments/ACF/DTW/JSD/PSD; EVAL-05 only needs a `scale` tag wrapper, not new metric math.
- Phase 10 TSTR-lite (1-layer LSTM-32, 3 init seeds, 320 held-out real windows) — the held-out real split convention and a working synthetic→real evaluation harness already exist to build the full TSTR on top of.
- 50 Phase 10 baseline run dirs + Phase 09.1 quantum runs — all carry `samples.npy` + data-hash; Phase 11 reads, never regenerates.

### Established Patterns
- `run_baselines.py` + `..._sweep.sh` idempotent per-(model,pipeline,seed) driver pattern with 5-file artifact bundle and atomic `sweep_status.json` — Phase 11 drivers follow the same shape.
- Code-placement invariant (D-10-13): `core/` = model defs + eval helpers only; orchestration/aggregation in `run_*.py`.
- Long-form metrics schema `{model_kind, pipeline, seed, metric_name, scale, value}` — Phase 11 JSON outputs extend, not replace, this.

### Integration Points
- Phase 11 outputs (`tstr.json`, `predictive_discriminative.json`, `augmentation.json`) join the existing `results/*.json` contract that Phase 14 paper-writing reads.
- Data-hash field (D-10-15) is the cross-phase consistency check — Phase 11 should assert hashes match across consumed artifacts.
</code_context>

<specifics>
## Specific Ideas

- Soft-sensor and augmentation study use **one shared downstream task** (one-step-ahead OD forecast) so TSTR (EVAL-01) and augmentation lift (EVAL-04) are directly comparable rather than measuring different things.
- Predictive/discriminative scores must be **faithful TimeGAN**, not the Phase 10 lite scaffolding, because reviewers explicitly asked for standard utility tests — non-standard reimplementations weaken the rebuttal.
- Variance collapse (fake std ≈ 48% of real) is a known, accepted limitation — Phase 11 reports utility honestly against matched baselines, it does not attempt to close the gap.
</specifics>

<deferred>
## Deferred Ideas

- **PAR_LIGHT-conditioned soft-sensor** — the manuscript's "industrial soft-sensor" framing (predict OD from PAR_LIGHT) was considered but deferred in favor of the cleaner one-step-ahead task. If reviewers specifically want the conditioned-prediction framing, that is a follow-up evaluation, not a Phase 11 change.
- **Small-real-regime augmentation** — deliberately shrinking the real training set to amplify detectable augmentation lift. Not selected (full real set used); capture as a backlog robustness check if the standard sweep shows no lift.
- CR-01 / CR-02 training-loop fixes — owned by Phase 13.

</deferred>

---

*Phase: 11-utility-evaluation*
*Context gathered: 2026-05-17*
