### Phase 13: Architecture & Introspection

**Goal**: Ansatz choice is justified empirically (2–3 variants compared) and the "black-box" feel (R2-6) is addressed with training-progression, parameter-trajectory, and entanglement-entropy figures — giving reviewers both "why this circuit?" and "what is it learning?" evidence
**Depends on**: Phase 8 (shared PQC module), Phase 10 (classical baseline needed for training-progression side-by-side), Phase 12 (multi-seed framework reused for ansatz comparison)
**Requirements**: ARCH-01, ARCH-02, INTRO-01, INTRO-02, INTRO-03
**Success Criteria** (what must be TRUE):

  1. 2–3 alternate ansatz variants (varying depth in {4, 6, 8} and/or entanglement topology) implemented in `revision/core/models/quantum.py` and selectable via config
  2. Ansatz comparison table (identical training budget, multi-seed, full metric suite) written to `revision/results/ansatz_comparison.json`
  3. Training-progression figure (`revision/results/figures/training_progression.*`) shows generated distribution at epochs {0, N/4, N/2, 3N/4, N} for quantum generator and classical WGAN-GP side-by-side
  4. PQC parameter-trajectory plot (norms + angle histograms across epochs) and entanglement-entropy (or state-purity) trajectory saved as figure artifacts — each with underlying data in JSON for reproducibility

**Plans**: 4 plans

Plans:
**Wave 1**

- [x] 13-01-PLAN.md — core/ edits: QuantumGenerator topology selector + introspect() (ARCH-01/INTRO-03); CR-01 differentiable PSD + CR-02 device-safe ES restore; greenfield tests/ + 4 regression tests (wave 1, foundational)

**Wave 2** *(blocked on Wave 1 — needs topology arg + introspect())*

- [x] 13-02-PLAN.md — run_ansatz.py + run_ansatz_sweep.sh (10-run V2/V3 × 5-seed sweep) + ansatz_comparison.json on extended long-form schema; V1 reused no-recompute (ARCH-01/02) (wave 2)
- [x] 13-03-PLAN.md — run_introspect.py callback-snapshot driver: instrumented V1 quantum + 3 classical WGAN runs (seed 42); 3 reproducibility companion JSON (INTRO-01/02/03) (wave 2, parallel to 13-02)

**Wave 3** *(blocked on 13-03 companion JSON)*

- [x] 13-04-PLAN.md — run_introspect_figures.py: render training-progression / param-trajectory / entanglement-trajectory figures (png+pdf) from companion JSON, render-only (INTRO-01/02/03) (wave 3)

**Cross-cutting constraints:**

- `revision/core/` default behavior byte-unchanged (topology="range", spectral weight 0.0, callback=None, early_stopper=None) — Phases 8–12 must reproduce
- No `multiprocessing.Pool` — xargs -P 2 only (D-10-24)
- V1 quantum reuses existing transform_ablation/runs/B 5-seed metrics — no recompute (D-13-01)

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
| 12. Sensitivity Analysis | v2.0 | 3/3 | Complete    | 2026-05-18 |
| 13. Architecture & Introspection | v2.0 | 4/4 | Complete    | 2026-05-19 |
| 14. Paper Revision & Release Freeze | v2.0 | 0/TBD | Not started | - |
