# Requirements: qGAN v2.0 AIChE Major Revision Response

**Defined:** 2026-04-23
**Manuscript:** aic-4719598 (AIChE Journal — Major Revision)
**Core Value:** The qGAN must generate synthetic OD time series that capture real data's volatility structure — not just the mean trend — with variance, kurtosis, and spectral characteristics that match the training distribution.
**Scope source:** `QGAN_Review_Response_Plan.md.pdf` (19 action items across 3 priority tiers)

---

## v2.0 Requirements — Code Group (Group A — executes first)

### Infrastructure

- [ ] **INFRA-01**: `revision/core/` package exists with importable modules (`data.py`, `eval.py`, `training.py`, `models/quantum.py`, `models/critic.py`, `models/classical_wgan.py`, `models/vae.py`). All logic lives in modules; revision notebooks only orchestrate + plot + write JSON.
- [ ] **INFRA-02**: Extracted modules reproduce main-notebook behavior — EMD and moment metrics match within numerical tolerance when `qgan_pennylane.ipynb` is re-run using the imported `revision/core/` modules (sanity check, no behavior change).
- [ ] **INFRA-03**: Repository frozen via tagged release (`v2.0-revision`) and Zenodo DOI minted; DOI cited in manuscript.

### Baselines (addresses R1-M1, R2-1)

- [ ] **BASE-01**: Classical WGAN-GP generator matched within ±5% of PQC trainable-parameter count, trained with identical critic architecture, optimizer, schedule, and seed set; full metric suite reported alongside quantum.
- [ ] **BASE-02**: Non-adversarial baseline (VAE or AR) trained on same data with same evaluation metrics.
- [ ] **BASE-03**: Parameter-count / expressibility-controlled comparison table produced as JSON + markdown (quantum, classical WGAN-GP, VAE/AR side-by-side).

### Evaluation (addresses R1-M2, R1-M3, R2-4, R1-m3)

- [ ] **EVAL-01**: TSTR pipeline — train 1D-CNN or LSTM soft-sensor on synthetic OD windows, evaluate on held-out real data; report R², MAE, RMSE.
- [ ] **EVAL-02**: TimeGAN-style predictive score computed for quantum, classical WGAN-GP, and non-adversarial baselines.
- [ ] **EVAL-03**: TimeGAN-style discriminative score computed for the same three models.
- [ ] **EVAL-04**: Real-only vs. synthetic-augmented training comparison (Orlandi et al. [26] style).
- [ ] **EVAL-05**: All fidelity metrics (EMD, ACF, moments, DTW) reported on both transformed (log-return) and original OD scales.
- [ ] **EVAL-06**: Differentiable `inverse_transform` exposed in `revision/core/data.py` (log-return + Lambert W back-transform to OD).

### Sensitivity (addresses R1-M4, R2-1)

- [ ] **SENS-01**: Shot-noise sweep at {analytic, 8192, 1024} shots; metric degradation reported.
- [ ] **SENS-02**: Noise-model sensitivity — depolarizing channel at p ∈ {0, 0.001, 0.01, 0.05} and amplitude-damping at γ ∈ {0, 0.001, 0.01, 0.05}.
- [ ] **SENS-03**: Multi-seed runs (≥5 seeds) for every headline result; mean ± std reported in every comparison table.

### Architecture Study (addresses R2-5b)

- [ ] **ARCH-01**: 2–3 alternate ansatz variants implemented (vary depth {4, 6, 8} and/or entanglement topology).
- [ ] **ARCH-02**: Ansatz comparison table (identical training budget, multi-seed, all metrics).

### Training Introspection (addresses R2-6)

- [ ] **INTRO-01**: Training-progression figure — generated distribution at epochs {0, N/4, N/2, 3N/4, N} for quantum and classical WGAN-GP.
- [ ] **INTRO-02**: PQC parameter trajectory plot (norms, angle histograms across epochs).
- [ ] **INTRO-03**: Entanglement-entropy or state-purity trajectory across training.

### Documentation Bridge (addresses R1-M4, R1-m2)

- [ ] **DOC-01**: Full training protocol extracted from code into `revision/docs/training_protocol.md` — N_CRITIC, λ, optimizer, LRs, epochs, stopping rule, seeds, shot/analytic clarification.
- [ ] **DOC-02**: Dataset statistics — raw time points, rolling windows, train/val/test split ratios and counts, number of independent runs — written to `revision/docs/dataset_stats.md`.

---

## v2.0 Requirements — Paper Group (Group B — reads numbers from Group A)

### Framing & Calibration (addresses R1-M5, R2-1, R2-2)

- [ ] **PAPER-01**: Hypothesis reframed in Section 1: "Can a PQC generator, operating in an exponentially large Hilbert space with O(poly(n)) parameters, match or exceed a classical generator of equivalent parameter count on a low-data bioprocess task?" Quantum-necessity transition softened.
- [ ] **PAPER-02**: Overclaiming language ("industrial bioprocess monitoring," "computational advantages," "exponential representational compactness," "reduced mode collapse") removed or explicitly softened to literature-motivated hypotheses.
- [ ] **PAPER-03**: Circuit design rationale subsection added (why 5 qubits, why this ansatz w/ expressibility–trainability tradeoff, why classical critic + quantum generator).
- [ ] **PAPER-04**: Log-returns justified in bioprocess context (growth-rate interpretation, not imported from finance).
- [ ] **PAPER-05**: Decision-tree workflow + Hybrid-GAN mechanistic material moved to explicit "Outlook" section; Supp. Table A2 either removed or explicitly caveated as aspirational; 20L vs 300L caption mismatch fixed.

### References & Methods (addresses R1-m1, R1-m2, R1-m3, R1-m6, R1-M4)

- [ ] **PAPER-06**: Misplaced references corrected — rewrite sentence for [27]; remove/reassign [28]; replace [39] with Havlíček 2019 + Schuld & Killoran 2019; replace [18] with GPR reference; replace [19] with time-series GAN; replace [41] with rolling-window subsequence reference; remove/replace [55]–[57], [59]. Anchors [21]–[23], [34]–[36], [61] retained.
- [ ] **PAPER-07**: Bernal et al. "Perspectives of quantum computing for chemical engineering" cited in Section 1.3/2 transition.
- [ ] **PAPER-08**: Dataset details reported in Methods (raw time points, rolling windows, splits + counts, independent-run count).
- [ ] **PAPER-09**: Every evaluation metric explicitly labeled in Methods as transformed (log-return) or original OD.
- [ ] **PAPER-10**: Appendix A3 log-GAN vs. Wasserstein discrepancy clarified.

### Copy Edits (addresses R1-M5, R1-m7)

- [ ] **PAPER-11**: Typos + notation unified — Fig. 6 "Laas"→"Lags"; "Figure A5).This"→"Figure A5). This"; "LUCY ©photobioreactor"→"LUCY® photobioreactor"; 300L/20L sentence completed; "Dry Biomass"→"dry biomass"; "bio-manufacturing" vs. "biomanufacturing" standardized; Ref [39] "Approac"→"Approach"; Ref [51] title capitalization; "QWGAN-GPs"→"QWGAN-GP" in conclusions; single return-variable symbol (log δ vs ς) chosen; Figures 2–6 enlarged for journal format.

---

## Reviewer-Comment Traceability

| Reviewer Item | Requirements |
|---------------|--------------|
| R1-M1 (No classical baseline) | BASE-01, BASE-02, BASE-03 |
| R1-M2 (Utility-oriented tests) | EVAL-01, EVAL-02, EVAL-03, EVAL-04 |
| R1-M3 (Signal transformation) | EVAL-05, EVAL-06, PAPER-04 |
| R1-M4 (Training details) | SENS-01, SENS-03, DOC-01, PAPER-10 |
| R1-M5 (Claim calibration) | PAPER-01, PAPER-02, PAPER-05, PAPER-11 |
| R1-m1 (Misplaced refs) | PAPER-06 |
| R1-m2 (Dataset details) | DOC-02, PAPER-08 |
| R1-m3 (Eval scale) | EVAL-05, PAPER-09 |
| R1-m4 (Freeze repo) | INFRA-03 |
| R1-m5 (Orlandi comparison) | EVAL-04 |
| R1-m6 (Bernal citation) | PAPER-07 |
| R1-m7 (Typos) | PAPER-11 |
| R2-1 (Preliminary/no comparison) | BASE-01, SENS-02, PAPER-01 |
| R2-2 (Quantum necessity) | PAPER-01, PAPER-07 |
| R2-3 (Decision pipeline) | PAPER-05 |
| R2-4 ("Improves vs. what?") | EVAL-01 |
| R2-5a (A3 done or not) | PAPER-05 |
| R2-5b (Circuit justification) | ARCH-01, ARCH-02, PAPER-03 |
| R2-6 (Black-box feel) | INTRO-01, INTRO-02, INTRO-03 |

All 19 reviewer items mapped.

---

## Out of Scope (for v2.0)

| Feature | Reason |
|---------|--------|
| Hardware (QPU) execution | Simulator-only stated honestly per R1-M5; not a reviewer ask |
| Closed-loop decision pipeline implementation | Reviewer accepted Outlook-only treatment per R2-3 |
| First-principles Hybrid-GAN implementation | Reviewer accepted "proposed extension" label per R2-5a |
| Refactoring main notebook into full .py package | `revision/core/` covers shared modules only; main notebook stays as-is |
| New training variance-collapse remediation | v2.0 reports honestly against classical baselines, not a fresh attempt |
| Full automated circuit architecture search | Manual ansatz comparison (ARCH-01) is sufficient for reviewer |
| Additional datasets / multi-campaign data | Reviewer accepts single-campaign scope if acknowledged |

---

## Historical (v1.0 + v1.1)

See `.planning/MILESTONES.md` for shipped milestones. Legacy requirement IDs (REG-01..05, SPEC-01..03, COND-01..03) are retained in git history and prior revisions of this file but are not part of v2.0 active scope.

---

## Traceability Table (filled by roadmapper)

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01    | Phase 8  | Pending |
| INFRA-02    | Phase 8  | Pending |
| INFRA-03    | Phase 14 | Pending |
| BASE-01     | Phase 10 | Pending |
| BASE-02     | Phase 10 | Pending |
| BASE-03     | Phase 10 | Pending |
| EVAL-01     | Phase 11 | Pending |
| EVAL-02     | Phase 11 | Pending |
| EVAL-03     | Phase 11 | Pending |
| EVAL-04     | Phase 11 | Pending |
| EVAL-05     | Phase 11 | Pending |
| EVAL-06     | Phase 9  | Pending |
| SENS-01     | Phase 12 | Pending |
| SENS-02     | Phase 12 | Pending |
| SENS-03     | Phase 12 | Pending |
| ARCH-01     | Phase 13 | Pending |
| ARCH-02     | Phase 13 | Pending |
| INTRO-01    | Phase 13 | Pending |
| INTRO-02    | Phase 13 | Pending |
| INTRO-03    | Phase 13 | Pending |
| DOC-01      | Phase 9  | Pending |
| DOC-02      | Phase 9  | Pending |
| PAPER-01    | Phase 14 | Pending |
| PAPER-02    | Phase 14 | Pending |
| PAPER-03    | Phase 14 | Pending |
| PAPER-04    | Phase 14 | Pending |
| PAPER-05    | Phase 14 | Pending |
| PAPER-06    | Phase 14 | Pending |
| PAPER-07    | Phase 14 | Pending |
| PAPER-08    | Phase 14 | Pending |
| PAPER-09    | Phase 14 | Pending |
| PAPER-10    | Phase 14 | Pending |
| PAPER-11    | Phase 14 | Pending |

**Coverage:**
- v2.0 requirements: 33 total (22 code, 11 paper)
- Mapped to phases: 33/33 ✓

---
*Requirements defined: 2026-04-23*
*Last updated: 2026-04-23 — roadmap complete, phase assignments filled*
