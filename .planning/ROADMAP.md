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

**Plans**: 16 plans

> Scope note (D-14-23, recorded deviation): this phase was intentionally, user-drivenly expanded from "paper edits + freeze" to also recover the lost canonical 55-param IQP:SEL circuit and re-execute Phases 10–13 at a matched 2000-epoch budget so the manuscript numbers are coherent and traceable. Plans 01–04 are that recovery/re-execution/provenance work; Plans 05–06 are the manuscript revision package; Plan 07 is the gated release freeze (LAST, D-14-22). Plan 08 adds the matched-budget dual-scale (OD + log_return) side-by-side comparison across all 9 models for the resubmission, re-evaluating the already-saved 14-02/03/04 sample artifacts (no retraining). Plan 09 adds the previously-absent circuit-architecture diagrams + V1/V2/V3 config locks closing the PAPER-03 visualization gap. Plan 10 adds the 7 missing story-completeness figures (training convergence, TSTR, failure modes, parameter-efficiency Pareto, seed variance, noise/shot-noise robustness) that consume previously-unconsumed audited JSONs. Plan 11 consolidates a paper-ready Methods document with programmatic classical-architecture extraction, pinned framework versions, and explicit determinism contract — resolving the two documented contradictions. Plan 12 threads the new circuit diagrams, missing figures, and methods doc into paper-blocks/reviewer_response/reconciliation, and emits a completeness-sweep manifest; after 14-12 only the Zenodo gate (14-07) remains. Plan 13 addresses a 5-agent peer-review pass surfacing 12 CRITICAL + 16 HIGH findings — provenance-gate hardening to v2 (LIFTS D-14-16), scale-correct reconciliation + cross-model EMD figure rebuild, aggregator integrity, render-determinism, and reproducibility infrastructure (pinned requirements, tracked checkpoint, CR-4 historical-asymmetry disclosure + future-gate); pure correction sweep, no retraining; after 14-13 only 14-07 remains. Plan 14 closes a 5-agent r2 peer-review pass on 14-13's remediation work (3 triangulated HIGHs + 12 lower-severity items): gate v2 sign-flip lookbehind fix (v2.1), training_time_device captured before .to("cpu") so the D-14-13 future-gate is structurally sound, VAE β_eff=2.5 derivation correction (not 0.4), and a 12-item doc/JSON cleanup batch; after 14-14 only 14-07 remains. Plan 15 reintroduces the pre-v1.0 histogram-density distribution-EMD as a third comparable column in reconciliation_note.md (so the matched-2000ep numbers can be read in the original paper's metric for the first time since the v1.0 raw-sample switch), adds a qq_overlay figure (Option A — single 9-model overlay + delta-QQ panel) that makes the OD-marginal convergence visually obvious, and updates reviewer_response.md + methods_full.md with the marginal-convergence finding (post-r2 investigation: all 9 models recover the OD marginal to within ≤0.03 OD-units max-quantile-diff except WGAN-CNN); after 14-15 only 14-07 remains. Plan 16 closes a 5-agent r3 forensic peer-review pass that triangulated two CRITICAL metric bugs systematically harming quantum: R3-CR-1 (NEW in 14-15) `revision/run_distribution_emd.py` `density=True` per-distribution renormalization rewards posterior-collapse (VAE std=0.0004) and out-of-range (WGAN-CNN 94% out-of-range) distributions, inverting rankings vs raw-sample EMD on identical samples (quantum drops 1–5 ranks, AR jumps +6); R3-CR-2 (INHERITED) `revision/run_matched2000_dualscale.py:368-372` compares standardized synth (std≈1) against unnormalized real (std=0.022), a 50× scale inflation that reverses log-return rankings on the corrected scale; the plan fixed both at root cause (R3-CR-2 via the un-standardize-fake recipe from pipeline-review-r3.md §2, applied at the matched2000_dualscale.py site AND the R3-HI-1 sister site in run_distribution_emd.py; R3-CR-1 via shared-edges-from-real-range with a disclosed `fake_in_range_mass` stat), re-emitted `matched2000_dualscale.json` + `distribution_emd.json` (v2), and added a new `welch_pairwise.json` aggregator. EXECUTION FINDINGS (Path A reframe — see 14-16-DEVIATION-NOTE.md): (1) after the R3-CR-2 fix the LR-EMD ranking inverts — every WGAN beats every quantum — so the r3 synthesis's "quantum beats WGAN on LR-EMD" claim, which was computed on the broken column, is WITHDRAWN; (2) R3-CR-1's density=True change is numerically inert (scipy wasserstein_distance renormalizes weights internally) — the fix's real value is the disclosure stat. Surviving manuscript claims: OD-EMD parametric-efficiency equivalence (55 quantum params ≡ 73-562 classical generator params, Welch p > 0.36, \\|d\\| ≤ 0.65, n=5) + DTW dominance (quantum beats every WGAN+AR on LR-DTW; ~6.5x Orlandi improvement on OD-DTW). reviewer_response.md + methods_full.md + reconciliation_note.md + peer_review_remediation.md updated; v2.1 gate passes on all 10 paper-facing docs; after 14-16 only 14-07 remains.

Plans:
**Wave 1**
- [x] 14-01-PLAN.md — Recover the 55-param IQP:SEL config from best_checkpoint.pt, add as non-default config-selectable circuit, config-equivalence hard-assert (T1, D-14-07)
**Wave 2** *(blocked on 14-01 — needs the locked 55-param config)*
- [x] 14-02-PLAN.md — Frozen-checkpoint headline + tiered resumable 2000ep matched-budget sweep behind the strict accept gate (D-14-08..14)
**Wave 3** *(blocked on 14-02 — needs accepted 2000ep artifacts)*
- [x] 14-03-PLAN.md — run_model_info.py → model_info.json, regenerate provenance docs from JSON, reconciliation note, reusable number-provenance gate (D-14-15/16)
- [x] 14-04-PLAN.md — Render-only per-model + cross-model + analysis figure suite (PNG+PDF+JSON, ≥16-figure canonical bar) (D-14-17)
**Wave 4** *(blocked on 14-03 + 14-04 — needs numbers + figures + provenance gate)*
- [x] 14-05-PLAN.md — PAPER-01/02/03/04/05 keyed framing/calibration LaTeX blocks; passes number-provenance gate
- [x] 14-06-PLAN.md — PAPER-06..11 keyed refs/methods/typo LaTeX blocks + per-reviewer reviewer_response.md; passes number-provenance gate
**Wave 5** *(blocked on 14-05 + 14-06 — release LAST, D-14-22)*
- [ ] 14-07-PLAN.md — Pre-tag freeze-ready gate + tag v2.0-revision + manual Zenodo reserved-DOI deposit + release.md (INFRA-03)
**Wave 6** *(independent — re-evaluates completed 14-02/03/04 sample artifacts; not blocked by 14-07)*
- [x] 14-08-PLAN.md — Render-only matched-2000ep dual-scale (OD + log_return) side-by-side comparison across all 9 models from saved samples; passes number-provenance gate (supports PAPER-09)
**Wave 7** *(independent — render-only circuit visualization; not blocked by 14-07)*
- [x] 14-09-PLAN.md — Render-only circuit diagrams for all 5 production qubit circuits (default_75, iqp_sel_55, V1, V2, V3) + V1/V2/V3 config locks + circuit_atlas.md (supports PAPER-03)
**Wave 8** *(independent — render-only story-completeness figures over existing JSONs; consumes previously-unconsumed tstr/noise/shot-noise data; not blocked by 14-07)*
- [x] 14-10-PLAN.md — Render-only story-completeness figure suite: training_convergence_all_models, tstr_crossmodel, failure_modes_summary, param_efficiency_pareto, seed_variance_per_model, noise_robustness_quantum, shot_noise_robustness (supports PAPER-01/09)
**Wave 9** *(independent — pure-aggregator methods doc consolidation over existing audited JSON + introspection of revision/core source for citations; not blocked by 14-07)*
- [x] 14-11-PLAN.md — Paper-ready Methods document (`revision/docs/methods_full.md`) + 3 new audited JSONs (`classical_architectures.json` via `run_classical_arch_extract.py`, `framework_versions.json` via `run_framework_versions.py`, `methods_full.json` via `run_methods_full.py`); resolves default_75 vs iqp_sel_55 and dtype_params vs dtype_samples contradictions; passes number-provenance gate (supports PAPER-08/09)
**Wave 10** *(integration only — paper-blocks cross-citation + completeness-sweep manifest; not blocked by 14-07)*
- [x] 14-12-PLAN.md — Thread 14-09/14-10/14-11 artifacts into paper_blocks_framing.md (PAPER-03 atlas + PAPER-01/02b param_efficiency_pareto), paper_blocks_refs_methods.md (PAPER-08 methods_full.md cross-ref + PAPER-09 6-figure citations), reviewer_response.md (Completeness sweep section, R1-M4 RESOLVED via methods_full.md), reconciliation_note.md (caveat paragraph); emit `revision/docs/completeness_sweep_manifest.md`; aggregate end-to-end provenance verify across all 6 paper-blocks docs; pure-additive, no `revision/core/` edit, no `verify_number_provenance.py` edit (supports PAPER-01/03/08/09)
**Wave 11** *(independent — peer-review correction sweep over existing artifacts; not blocked by 14-07)*
- [x] 14-13-PLAN.md — Peer-review remediation sweep: provenance-gate v2 (LIFTS D-14-16), scale-correct reconciliation + cross_model_emd rebuild, ddof=0→ddof=1 sample-std switch, aggregator integrity (HI-2..HI-8, MD-3), figure render-determinism (CR-1), paper-blocks phantom-number cleanup, pinned `revision/requirements-pinned.txt` + tracked `revision/checkpoints/best_checkpoint.pt`, CR-4 honest disclosure + future-gate (D-14-13 extension), methods_full.md Metric Conventions section honoring D-14-22, `peer_review_remediation.md` reviewer-facing index; passes v2 gate on all 9 paper-facing docs (supports PAPER-01/02/03/08/09 + INFRA-03)
**Wave 12** *(independent — pre-tag punch list from r2 peer-review pass; not blocked by 14-07)*
- [x] 14-14-PLAN.md — Pre-tag punch list: gate v2.1 (negative-sign-aware lookbehind, LIFTS D-14-16), training_time_device capture-before-`.to(cpu)`, VAE β_eff=2.5 derivation correction + VAE-not-param-matched caveat, wgan_cnn seed-variance honesty, R1-m4 DOI-pending wording, docstring 1-80→1-69, CR-3 line 346→347, apparatus units split, statsmodels pin, REPRODUCE.md, _introspect render_only marks, sensitivity data_hash; passes v2.1 gate on all 10 paper-facing docs (supports PAPER-01/02/08/09 + INFRA-03)
**Wave 13** *(independent — distribution-EMD column + QQ overlay + marginal-convergence reviewer disclosure; not blocked by 14-07)*
- [x] 14-15-PLAN.md — Distribution-EMD column + QQ overlay (Option A) + marginal-convergence finding: new revision/run_distribution_emd.py emitter (50-bin histogram-density Wasserstein, pre-v1.0 formulation), 3-column comparable EMD table in reconciliation_note.md, qq_overlay.{png,pdf,json} single discriminating figure with delta-QQ panel, reviewer_response.md + methods_full.md + peer_review_remediation.md updates carrying the two-pronged convergence finding (pairwise: 8/9 ~0.03 median + WGAN-CNN-vs-others ~0.69 median; vs-real: all-9 ~0.25 + WGAN-CNN ~0.81) and redirecting reviewers to dependence-structure figures (ACF, conditional moments, TimeGAN-style scores) for architecture discrimination (supports PAPER-01/02/08/09 + INFRA-03)
**Wave 14** *(independent — r3 forensic remediation: fix R3-CR-1 + R3-CR-2 + reframe to strong-claim parametric-efficiency-equivalence; not blocked by 14-07)*
- [x] 14-16-PLAN.md — R3 forensic remediation (7 tasks, Path A; HEAD `75f979e`): fixed R3-CR-2 log-return EMD scale mismatch (`revision/run_matched2000_dualscale.py` — un-standardize-fake recipe per pipeline-review-r3.md §2, NOT standardize-real) and R3-HI-1 sister site in `run_distribution_emd.py`; fixed R3-CR-1 histogram-density formulation (shared-edges-from-real + `fake_in_range_mass` disclosure stat — investigation found the density=True change numerically inert for `scipy.stats.wasserstein_distance`, distribution-emd v2); NEW `run_welch_aggregator.py` → `welch_pairwise.json` (JSON-anchored OD-EMD provenance). **Path A reframe** (executor checkpoint finding): the corrected LR-EMD ranking inverts — every WGAN beats every quantum on LR-EMD, so the synthesis's "quantum beats WGAN on LR-EMD" claim (derived from the broken column) is WITHDRAWN. Surviving manuscript claims: OD-EMD parametric-efficiency equivalence (55 quantum params ≡ 73-562 classical generator params, Welch p > 0.36, |d| ≤ 0.65, n=5) + DTW dominance (quantum beats every WGAN+AR on LR-DTW; ~6.5x Orlandi improvement on OD-DTW). reviewer_response.md/methods_full.md/reconciliation_note.md/peer_review_remediation.md updated; v2.1 gate passes on all 10 paper-facing docs (supports PAPER-01/02/08/09 + INFRA-03)

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
| 14. Paper Revision & Release Freeze | v2.0 | 15/16 | In Progress|  |
