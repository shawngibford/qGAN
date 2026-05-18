# Roadmap: qGAN Post-HPO Improvements

## Milestones

- ✅ **v1.0 qGAN Code Review Remediation** -- Phases 1-3 (shipped 2026-03-07)
- ✅ **v1.1 Post-HPO Improvements** -- Phases 4-7 (shipped 2026-03-23)
- 🚧 **v2.0 AIChE Major Revision Response** -- Phases 8-14 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (8, 9, 10, ...): Planned milestone work
- Decimal phases (8.1, 10.1): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v1.0 qGAN Code Review Remediation (Phases 1-3) -- SHIPPED 2026-03-07</summary>

- [x] Phase 1: Foundation and Correctness Infrastructure (3/3 plans) -- completed 2026-03-01
- [x] Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign (4/4 plans) -- completed 2026-03-05
- [x] Phase 3: Post-Processing Consistency and Cleanup (2/2 plans) -- completed 2026-03-07

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.1 Post-HPO Improvements (Phases 4-7) -- SHIPPED 2026-03-23</summary>

- [x] Phase 4: Code Regression Fixes (2/2 plans) -- completed 2026-03-13
- [x] Phase 5: Backprop and Broadcasting -- completed as part of v1.1
- [x] Phase 6: Spectral Loss -- completed as part of v1.1
- [x] Phase 7: Conditioning Verification (1/1 plans) -- completed 2026-03-23

Full details: `.planning/ROADMAP.md` (prior revision) and git history.

</details>

### 🚧 v2.0 AIChE Major Revision Response

**Milestone Goal:** Address all reviewer concerns on AIChE Journal manuscript aic-4719598 so the QWGAN-GP bioprocess paper can be resubmitted — establishing quantum-vs-classical evidence, utility-oriented validation, and calibrated claims.

**Dependency contract:** Group A (code) executes before Group B (paper). Paper Phase 14 reads JSON artifacts written by Phases 8-13.

- [ ] **Phase 8: Core Module Extraction** - Extract shared logic into `revision/core/` and verify parity with main notebook
- [x] **Phase 9: Documentation Bridge** - Training protocol + dataset stats + differentiable inverse transform — cheap, paper-ready numbers that unblock paper drafting
- [x] **Phase 10: Classical Baselines** - Matched-parameter classical WGAN-GP + non-adversarial baseline (VAE/AR) + side-by-side comparison table (completed 2026-05-18)
- [x] **Phase 11: Utility Evaluation** - TSTR, predictive/discriminative scores, real-only vs synthetic-augmented, fidelity metrics on both scales (all 4 plans executed + verified 5/5 on 2026-05-18; gap-closure pending for CR-01 + 6 code-review warnings) (completed 2026-05-18)
- [x] **Phase 12: Sensitivity Analysis** - Shot-noise sweep, noise-model sensitivity, multi-seed (≥5) mean ± std across all headline results (completed 2026-05-18)
- [ ] **Phase 13: Architecture & Introspection** - 2–3 ansatz comparison + training-progression / parameter-trajectory / entanglement figures
- [ ] **Phase 14: Paper Revision & Release Freeze** - All PAPER-* revisions to manuscript aic-4719598 + Zenodo DOI freeze

## Phase Details

### Phase 8: Core Module Extraction

**Goal**: `revision/core/` package exists and is a drop-in replacement for inline notebook logic, so every downstream v2.0 phase imports from a single verified codebase
**Depends on**: Phase 7 (v1.1 complete)
**Requirements**: INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):

  1. `revision/core/` contains importable modules `data.py`, `eval.py`, `training.py`, `models/quantum.py`, `models/critic.py`, `models/classical_wgan.py`, `models/vae.py` — every function used by downstream revision notebooks is reachable via `from revision.core...` import
  2. Main notebook `qgan_pennylane.ipynb` re-runs using imported `revision/core/` modules and produces EMD and moment (mean, std, kurtosis) metrics matching the pre-extraction baseline within numerical tolerance (≤1e-6 on float metrics, ≤1e-4 on EMD)
  3. No business logic remains inline in a revision notebook — revision notebooks only orchestrate (call module functions), plot, and write JSON to `revision/results/`
  4. A parity-check artifact (`revision/results/parity_check.json`) exists with the side-by-side metric comparison so future regressions are catchable

**Plans:** 5 plans
Plans:

- [x] 08-01-PLAN.md — Package scaffold (revision/core/ directory + signature stubs for all modules)
- [x] 08-02-PLAN.md — Extract data pipeline + evaluation metrics (data.py, eval.py)
- [x] 08-03-PLAN.md — Extract quantum generator + critic models (models/quantum.py, models/critic.py)
- [x] 08-04-PLAN.md — Extract WGAN-GP training loop with seed/spectral/callback hooks (training.py)
- [x] 08-05-PLAN.md — Parity check notebook + parity_check.json artifact (INFRA-02)

### Phase 9: Documentation Bridge

**Goal**: Paper-ready training protocol, dataset statistics, and a differentiable inverse-transform are available before any expensive code experiments run — so paper drafting can begin in parallel with Phases 10-13 and every downstream evaluation can round-trip between log-return and OD scales
**Depends on**: Phase 8 (extraction must land first so protocol/stats reflect the canonical `revision/core/` code path)
**Requirements**: DOC-01, DOC-02, EVAL-06
**Success Criteria** (what must be TRUE):

  1. `revision/docs/training_protocol.md` exists and documents N_CRITIC, λ, optimizer, both learning rates, epochs, early-stopping rule, seeds, and shot/analytic distinction — numbers traceable to `revision/core/` defaults
  2. `revision/docs/dataset_stats.md` exists and reports raw time-point count, rolling-window count, train/val/test split ratios and counts, and number of independent campaign runs
  3. `revision/core/data.py` exposes a differentiable `inverse_transform` (log-return + Lambert W back-transform) verified round-trip on a held-out sample to match input within 1e-8
  4. Both doc files are referenced from Phase 14 paper work without requiring rewrite (paper-ready prose + numbers)

**Plans:** 5 plans
Plans:

- [x] 09-01-PLAN.md — Differentiable inverse Lambert W via torch.autograd.Function (EVAL-06 core)
- [x] 09-02-PLAN.md — preprocessing.py skeleton + module registration (D-06; Phase 09.1 contract)
- [x] 09-03-PLAN.md — training_protocol.md (DOC-01) — paper-ready hybrid format
- [x] 09-04-PLAN.md — dataset_stats.md (DOC-02) — paper-ready hybrid format
- [x] 09-05-PLAN.md — Round-trip verification notebook + eval06_roundtrip.json + Phase 8 parity regression (EVAL-06 acceptance)

### Phase 09.1: R1-M3 Preprocessing Ablation (INSERTED)

**Goal:** Run a controlled three-pipeline preprocessing ablation (A: raw normalized OD, B: log-returns only, C: log-returns + Lambert W) with identical training conditions and ≥5 seeds per pipeline; produce OD-scale comparison artifacts that answer reviewer R1-M3's "transformation strips temporal structure" claim and empirically justify the chosen pipeline for the revised manuscript.
**Depends on:** Phase 9 (EVAL-06 differentiable inverse transform + `revision/core/preprocessing.py` contract are hard prerequisites)
**Requirements**: ABL-01, ABL-02, ABL-03
**Source spec:** `.planning/scratch/09.1-r1-m3-ablation-spec.md` (user-authored PRD, 2026-05-08)
**Success Criteria** (what must be TRUE):

  1. `revision/core/preprocessing.py` exposes three `forward_X`/`inverse_X` pairs (A/B/C) with verified ≤float-eps round-trip on real trajectories (max abs error printed)
  2. `revision/results/transform_ablation/runs/<pipeline>/<seed>/` contains per-seed checkpoints, generated samples, and run config YAML for all 3 × ≥5 = ≥15 runs; smoke run completes successfully before full multi-seed launch
  3. `revision/results/transform_ablation/metrics.csv` (long-form: pipeline, seed, metric_name, scale, value) + 6 figures (`fig_trajectories.png`, `fig_acf_od.png`, `fig_acf_transformed.png`, `fig_qq_od.png`, `fig_pdf_od.png`, `fig_dtw_distribution.png`) generated on OD scale
  4. `revision/results/transform_ablation/summary.md` answers the four R1-M3 rebuttal questions with numbers (mean ± std) and recommends a pipeline for the revised manuscript
  5. Pipeline C reproduces v1.1 published log-return EMD within 1–2% (sanity check that the ablation harness preserves baseline behavior)

**Plans:** 4/4 plans complete

Plans:

- [x] 09.1-01-PLAN.md — Implement Pipeline A/B preprocessing pure-functions + round-trip verification notebook (ABL-01 gate)
- [x] 09.1-02-PLAN.md — CLI driver (revision/run_ablation.py) + smoke notebook (3 pipelines × seed 42 × 100 epochs, Pipeline C parity gate)
- [x] 09.1-03-PLAN.md — Resumable multi-seed sweep (3 × 5 × 1000 epochs = 29.1 min wall @ parallel=2, 15/15 complete) → ABL-02 GREEN
- [x] 09.1-04-PLAN.md — Analysis notebook: metrics.csv + 6 figures + TSTR-lite + summary.md answering R1-M3 Q1-Q4 (ABL-03 gate)

### Phase 10: Classical Baselines

**Goal**: Matched-parameter classical WGAN-GP and a non-adversarial baseline (VAE or AR) are trained under identical conditions to the quantum generator, so the manuscript can report a fair quantum-vs-classical comparison in response to R1-M1 and R2-1
**Depends on**: Phase 8 (uses shared training loop + critic + data modules), Phase 9 (inverse-transform required for OD-scale reporting)
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):

  1. Classical WGAN-GP generator has trainable parameter count within ±5% of the PQC; trained with identical critic architecture, optimizer, schedule, and seed set; training artifacts written to `revision/results/baseline_classical_wgan.json`
  2. Non-adversarial baseline (VAE or AR — choice documented in phase summary) trained on same data with same evaluation metrics; artifacts in `revision/results/baseline_nonadversarial.json`
  3. Side-by-side comparison table (quantum / classical WGAN-GP / VAE-or-AR) emitted as `revision/results/baseline_comparison.json` with a companion markdown rendering — every row carries parameter count and full fidelity metric suite
  4. All three models use the same data split produced by `revision/core/data.py` — verifiable from a data-hash field in each JSON artifact

**Plans**: TBD

### Phase 11: Utility Evaluation

**Goal**: Manuscript can answer "improves vs. what?" (R2-4) with concrete utility-oriented numbers — TSTR soft-sensor performance, predictive and discriminative scores, and real-only vs. synthetic-augmented training deltas — reported on both log-return and OD scales
**Depends on**: Phase 10 (all utility metrics compute across quantum + both baselines, so baselines must exist)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):

  1. TSTR pipeline trains a 1D-CNN or LSTM soft-sensor on synthetic OD windows, evaluates on held-out real data, and reports R², MAE, RMSE for quantum + both baselines to `revision/results/tstr.json`
  2. TimeGAN-style predictive score and discriminative score computed for quantum + classical WGAN-GP + non-adversarial baseline; results in `revision/results/predictive_discriminative.json` with mean ± std across seeds
  3. Real-only vs. synthetic-augmented training comparison (Orlandi et al. style) produces a delta table in `revision/results/augmentation.json` showing downstream-task lift from each generator
  4. Every fidelity metric (EMD, ACF, moments, DTW) is reported on both transformed (log-return) and original OD scales — visible as explicit `scale: "log_return" | "OD"` fields in JSON outputs

**Plans:** 8/8 plans complete
Plans:
**Wave 1**

- [x] 11-01-PLAN.md — run_utility.py: EVAL-01 TSTR soft-sensor + EVAL-04 Orlandi augmentation lift (tstr.json, augmentation.json)
- [x] 11-02-PLAN.md — run_timegan_scores.py: EVAL-02/03 faithful TimeGAN predictive + discriminative scores (predictive_discriminative.json)
- [x] 11-03-PLAN.md — run_dualscale_fidelity.py: EVAL-05 dual-scale (OD + log_return) fidelity re-emit (fidelity_dualscale.json)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 11-04-PLAN.md — test_utility.py: cross-artifact scientific-integrity verification suite + Phase 11 closeout

**Gap closure** *(from 11-VERIFICATION.md — CR-01 + WR-01..06)*

- [x] 11-05-PLAN.md — run_utility.py correctness: WR-01 shape comment, WR-02 NaN-on-degenerate R2, WR-03 collision-free subsample seed, WR-04 grid-collapse guard (gap, wave 1)
- [x] 11-06-PLAN.md — run_dualscale_fidelity.py portability: CR-01 env-var QGAN_CANONICAL_REPO resolver + fail-loud + single-root provenance assertion (gap, wave 1, closes HUMAN-UAT)
- [x] 11-07-PLAN.md — run_timegan_scores.py: WR-05 single-Generator discriminative_score + logits/labels shape contract (gap, wave 1)
- [x] 11-08-PLAN.md — test_timegan_scores.py: WR-06 collected test_discriminative_score_deterministic (gap, wave 2 — depends on 11-07)

**Cross-cutting constraints:**

- Recomputed data_hash equals 91e447d4624e25b3 and equals every one of the 50 baseline config.yaml data_hash fields
- revision/core/ is untouched (git diff --stat revision/core/ empty)

### Phase 12: Sensitivity Analysis

**Goal**: Quantum results are stress-tested under shot noise, hardware-style noise channels, and seed variation — so the manuscript reports calibrated uncertainty bars and directly addresses R1-M4 and R2-1 preliminary-result concerns
**Depends on**: Phase 10 (baselines needed for multi-seed comparison tables); Phase 11 is parallel-safe but sensitivity results layer on top of utility metrics
**Requirements**: SENS-01, SENS-02, SENS-03
**Success Criteria** (what must be TRUE):

  1. Shot-noise sweep at {analytic, 8192, 1024} shots run for quantum generator; metric degradation curve written to `revision/results/shot_noise_sensitivity.json`
  2. Noise-model sensitivity results for depolarizing channel (p ∈ {0, 0.001, 0.01, 0.05}) and amplitude-damping (γ ∈ {0, 0.001, 0.01, 0.05}) written to `revision/results/noise_model_sensitivity.json`
  3. Every headline comparison table (from Phases 10-11) re-emitted with ≥5 seeds, reporting mean ± std in every cell — `revision/results/multiseed_summary.json` consolidates the multi-seed roll-up
  4. Compute budget respected — sensitivity sweeps complete on local Mac statevector simulator within the phase session (documented in phase summary)

**Plans**: 3 plans

Plans:
**Wave 1**

- [x] 12-01-PLAN.md — run_sensitivity.py inference driver: trained-params reload, set_shots/default.mixed QNodes, *0.1+reconstruction contracts, harness-faithfulness smoke gate (wave 1)
- [x] 12-03-PLAN.md — run_multiseed_rollup.py SENS-03 aggregator: D-10-15 cross-artifact data_hash gate, mean±std roll-up -> multiseed_summary.json (wave 1, parallel-safe)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 12-02-PLAN.md — run_sensitivity_sweep.sh + full SENS-01/02 grid execution; shot_noise_sensitivity.json + noise_model_sensitivity.json emitted (wave 2, depends on 12-01)

**Cross-cutting constraints:**

- revision/core/ is byte-untouched

### Phase 13: Architecture & Introspection

**Goal**: Ansatz choice is justified empirically (2–3 variants compared) and the "black-box" feel (R2-6) is addressed with training-progression, parameter-trajectory, and entanglement-entropy figures — giving reviewers both "why this circuit?" and "what is it learning?" evidence
**Depends on**: Phase 8 (shared PQC module), Phase 10 (classical baseline needed for training-progression side-by-side), Phase 12 (multi-seed framework reused for ansatz comparison)
**Requirements**: ARCH-01, ARCH-02, INTRO-01, INTRO-02, INTRO-03
**Success Criteria** (what must be TRUE):

  1. 2–3 alternate ansatz variants (varying depth in {4, 6, 8} and/or entanglement topology) implemented in `revision/core/models/quantum.py` and selectable via config
  2. Ansatz comparison table (identical training budget, multi-seed, full metric suite) written to `revision/results/ansatz_comparison.json`
  3. Training-progression figure (`revision/results/figures/training_progression.*`) shows generated distribution at epochs {0, N/4, N/2, 3N/4, N} for quantum generator and classical WGAN-GP side-by-side
  4. PQC parameter-trajectory plot (norms + angle histograms across epochs) and entanglement-entropy (or state-purity) trajectory saved as figure artifacts — each with underlying data in JSON for reproducibility

**Plans**: TBD

### Phase 14: Paper Revision & Release Freeze

**Goal**: Manuscript aic-4719598 revised end-to-end — hypothesis reframed, claims calibrated, circuit rationale added, references corrected, methods sections complete, typos fixed — and the repository frozen with a tagged release + Zenodo DOI so reviewers can cite the exact code state
**Depends on**: Phases 9-13 (paper reads numbers and figures from all upstream JSON artifacts)
**Requirements**: PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10, PAPER-11, INFRA-03
**Success Criteria** (what must be TRUE):

  1. Hypothesis reframed in Section 1 (PAPER-01) and all overclaiming language (PAPER-02) softened or removed — reviewer-facing checklist maps each change to the reviewer comment it addresses
  2. Manuscript contains the new "Circuit Design Rationale" subsection (PAPER-03), log-returns bioprocess justification (PAPER-04), and the "Outlook" section with decision-tree + Hybrid-GAN material moved out of main claims (PAPER-05)
  3. Reference list corrected (PAPER-06) with Bernal et al. added (PAPER-07); Methods section now reports dataset details (PAPER-08) and per-metric evaluation scale (PAPER-09); Appendix A3 discrepancy clarified (PAPER-10); all typos and notation unified (PAPER-11)
  4. Repository frozen at tag `v2.0-revision`, Zenodo DOI minted and cited in the manuscript (INFRA-03) — tag + DOI resolvable from `revision/docs/release.md`
  5. All numbers cited in the revised manuscript trace back to a JSON artifact in `revision/results/` (no hand-typed numbers)

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 8 → 9 → 10 → 11 → 12 → 13 → 14

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation and Correctness Infrastructure | v1.0 | 3/3 | Complete | 2026-03-01 |
| 2. WGAN-GP Correctness and Quantum Circuit Redesign | v1.0 | 4/4 | Complete | 2026-03-05 |
| 3. Post-Processing Consistency and Cleanup | v1.0 | 2/2 | Complete | 2026-03-07 |
| 4. Code Regression Fixes | v1.1 | 2/2 | Complete | 2026-03-13 |
| 5. Backprop and Broadcasting | v1.1 | 2/2 | Complete | 2026-03-18 |
| 6. Spectral Loss | v1.1 | 1/1 | Complete | 2026-03-21 |
| 7. Conditioning Verification | v1.1 | 1/1 | Complete | 2026-03-23 |
| 8. Core Module Extraction | v2.0 | 0/5 | Not started | - |
| 9. Documentation Bridge | v2.0 | 0/5 | Not started | - |
| 10. Classical Baselines | v2.0 | 4/4 | Complete    | 2026-05-18 |
| 11. Utility Evaluation | v2.0 | 8/8 | Complete    | 2026-05-18 |
| 12. Sensitivity Analysis | v2.0 | 3/3 | Complete   | 2026-05-18 |
| 13. Architecture & Introspection | v2.0 | 0/TBD | Not started | - |
| 14. Paper Revision & Release Freeze | v2.0 | 0/TBD | Not started | - |
