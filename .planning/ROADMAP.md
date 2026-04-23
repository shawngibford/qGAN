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
- [ ] **Phase 9: Documentation Bridge** - Training protocol + dataset stats + differentiable inverse transform — cheap, paper-ready numbers that unblock paper drafting
- [ ] **Phase 10: Classical Baselines** - Matched-parameter classical WGAN-GP + non-adversarial baseline (VAE/AR) + side-by-side comparison table
- [ ] **Phase 11: Utility Evaluation** - TSTR, predictive/discriminative scores, real-only vs synthetic-augmented, fidelity metrics on both scales
- [ ] **Phase 12: Sensitivity Analysis** - Shot-noise sweep, noise-model sensitivity, multi-seed (≥5) mean ± std across all headline results
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
- [ ] 08-01-PLAN.md — Package scaffold (revision/core/ directory + signature stubs for all modules)
- [ ] 08-02-PLAN.md — Extract data pipeline + evaluation metrics (data.py, eval.py)
- [ ] 08-03-PLAN.md — Extract quantum generator + critic models (models/quantum.py, models/critic.py)
- [ ] 08-04-PLAN.md — Extract WGAN-GP training loop with seed/spectral/callback hooks (training.py)
- [ ] 08-05-PLAN.md — Parity check notebook + parity_check.json artifact (INFRA-02)

### Phase 9: Documentation Bridge
**Goal**: Paper-ready training protocol, dataset statistics, and a differentiable inverse-transform are available before any expensive code experiments run — so paper drafting can begin in parallel with Phases 10-13 and every downstream evaluation can round-trip between log-return and OD scales
**Depends on**: Phase 8 (extraction must land first so protocol/stats reflect the canonical `revision/core/` code path)
**Requirements**: DOC-01, DOC-02, EVAL-06
**Success Criteria** (what must be TRUE):
  1. `revision/docs/training_protocol.md` exists and documents N_CRITIC, λ, optimizer, both learning rates, epochs, early-stopping rule, seeds, and shot/analytic distinction — numbers traceable to `revision/core/` defaults
  2. `revision/docs/dataset_stats.md` exists and reports raw time-point count, rolling-window count, train/val/test split ratios and counts, and number of independent campaign runs
  3. `revision/core/data.py` exposes a differentiable `inverse_transform` (log-return + Lambert W back-transform) verified round-trip on a held-out sample to match input within 1e-8
  4. Both doc files are referenced from Phase 14 paper work without requiring rewrite (paper-ready prose + numbers)
**Plans**: TBD

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
**Plans**: TBD

### Phase 12: Sensitivity Analysis
**Goal**: Quantum results are stress-tested under shot noise, hardware-style noise channels, and seed variation — so the manuscript reports calibrated uncertainty bars and directly addresses R1-M4 and R2-1 preliminary-result concerns
**Depends on**: Phase 10 (baselines needed for multi-seed comparison tables); Phase 11 is parallel-safe but sensitivity results layer on top of utility metrics
**Requirements**: SENS-01, SENS-02, SENS-03
**Success Criteria** (what must be TRUE):
  1. Shot-noise sweep at {analytic, 8192, 1024} shots run for quantum generator; metric degradation curve written to `revision/results/shot_noise_sensitivity.json`
  2. Noise-model sensitivity results for depolarizing channel (p ∈ {0, 0.001, 0.01, 0.05}) and amplitude-damping (γ ∈ {0, 0.001, 0.01, 0.05}) written to `revision/results/noise_model_sensitivity.json`
  3. Every headline comparison table (from Phases 10-11) re-emitted with ≥5 seeds, reporting mean ± std in every cell — `revision/results/multiseed_summary.json` consolidates the multi-seed roll-up
  4. Compute budget respected — sensitivity sweeps complete on local Mac statevector simulator within the phase session (documented in phase summary)
**Plans**: TBD

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
| 9. Documentation Bridge | v2.0 | 0/TBD | Not started | - |
| 10. Classical Baselines | v2.0 | 0/TBD | Not started | - |
| 11. Utility Evaluation | v2.0 | 0/TBD | Not started | - |
| 12. Sensitivity Analysis | v2.0 | 0/TBD | Not started | - |
| 13. Architecture & Introspection | v2.0 | 0/TBD | Not started | - |
| 14. Paper Revision & Release Freeze | v2.0 | 0/TBD | Not started | - |
