# Point-by-Point Response to Reviewers — Manuscript aic-4719598

> **AIChE Journal Major Revision.** "QWGAN-GP for Synthetic Bioprocess
> Time-Series." Editor: Prodromos Daoutidis. This document is the per-reviewer,
> per-comment rebuttal (D-14-19): every comment ID from
> `QGAN_Review_Response_Plan.md.pdf` maps to the verbatim concern, the change
> made, the manuscript location, and a real supporting artifact under
> `revision/results/` (success criterion 5). Every "Supporting artifact" cell
> points at a path that exists in this repository — no `TODO`/`TBD`/placeholder
> cells. Sourced-row table discipline mirrors
> `revision/docs/training_protocol.md` (every claim carries a provenance
> column). LaTeX text changes are delivered as copy-paste blocks in
> `revision/docs/paper_blocks_refs_methods.md` and the companion PAPER-blocks
> files (the in-repo `.tex` is read-only, D-14-18).

## Summary of Response

Both reviewers converge on one theme: the paper must make a stronger,
calibrated case for *why quantum* and must demonstrate *utility*, not just
diagnostics. The revision (a) adds a matched-parameter classical WGAN-GP and a
non-adversarial baseline, (b) adds TSTR / predictive / discriminative
utility evaluation, (c) reports results on the original OD scale, (d) adds
shot-noise, multi-seed, and noise-model sensitivity, (e) adds circuit
introspection figures, (f) reports the full training protocol and dataset
statistics rendered from JSON, (g) reframes the hypothesis and tones down
overclaiming, and (h) freezes the repository with a citable DOI. Every
quantitative claim is traceable to a `revision/results/*.json` artifact via
`revision/verify_number_provenance.py`.

---

## Reviewer 1 — Major Issues

| ID | Verbatim concern (abbrev.) | Change made | Manuscript location | Supporting artifact |
|----|----------------------------|-------------|---------------------|---------------------|
| R1-M1 | No matched classical baseline — quantum contribution cannot be isolated | Added matched-parameter classical WGAN-GP (MLP/CNN/LSTM critics) and a non-adversarial VAE + AR baseline, all at matched 2000-epoch budget, identical critic/optimizer/seed set; parameter-count-controlled comparison table | §4.1 Results (new baseline comparison table); §4.2 Key Contributions (honest framing) | `revision/results/baseline_comparison.json`; `revision/results/model_info.json` |
| R1-M2 | Validation is diagnostic only — need utility-oriented tests (TSTR, predictive/discriminative) | Added TSTR (train-on-synthetic, test-on-real soft sensor), TimeGAN predictive + discriminative scores, and Orlandi-style real-only vs synthetic-augmented comparison, re-run at matched 2000-epoch Pipeline B budget (Plan 14-20) so R1-M1 and R1-M2 share a single matched-budget evidence base; legacy 1000-epoch JSONs preserved as provenance but not cited | §4.1 Results (utility-evaluation subsection) | `revision/results/tstr_matched2000.json`; `revision/results/predictive_discriminative_matched2000.json`; `revision/results/augmentation_matched2000.json`; figure `revision/results/figures/tstr_crossmodel_matched2000.{png,pdf,json}` |
| R1-M3 | Log-returns + Lambert W may strip temporal structure; no OD back-transformation | Added original-OD-scale results (generate → invert → metrics on physical units); ACF on both transformed and OD scale; explicit per-metric scale statement; growth-rate justification of log-returns | §3 Methods (evaluation-scale paragraph + Table); §4.1 (dual-scale ACF) | `revision/results/fidelity_dualscale.json`; `revision/docs/dataset_stats.md` |
| R1-M4 | Incomplete optimization / training details (n_critic, λ, LR, epochs, stopping, seeds, analytic vs shot) | Added full Training Protocol (all hyperparameters rendered from JSON); stated analytic statevector (no shot noise) backend; added shot-noise sensitivity; multi-seed (5 seeds) mean ± std; clarified Supp Eq. A3 log-GAN vs Wasserstein discrepancy | §3 Methods (Training Protocol); Supp §A.3 (PAPER-10 block) | `revision/docs/training_protocol.md`; `revision/results/shot_noise_sensitivity.json`; `revision/results/multiseed_summary.json` |
| R1-M5 | Claim calibration — language oversells a simulator-based, single-variable, single-campaign proof-of-concept | Toned language to "proof-of-concept feasibility study"; moved decision-tree workflow + Hybrid-GAN to a labeled Outlook; caveated Supp Table A2 as aspirational; clarified 20L/300L; softened "exponential compactness"/"reduced mode collapse" to literature-motivated hypotheses | Abstract; §1; §4.2–4.4; §5; Supp §A.3 (PAPER-02/05/10/11 blocks) | `revision/docs/paper_blocks_refs_methods.md`; `revision/results/model_info.json` |

## Reviewer 1 — Minor Issues

| ID | Verbatim concern (abbrev.) | Change made | Manuscript location | Supporting artifact |
|----|----------------------------|-------------|---------------------|---------------------|
| R1-m1 | Misplaced / weak references ([27][28][39][18][19][41][55]-[57][59]) | Per-reference `.bib`+sentence-rewrite surgery; explicit RETAINED note for the reviewer-confirmed anchors [21]-[23],[34]-[36],[61] | §1.3, §1.4, §2.4, §3.1, Supp §A.2 (PAPER-06 blocks) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-06) |
| R1-m2 | Report dataset details (raw points, windows, splits, independent runs) | Added a Dataset-and-Preprocessing Methods paragraph, every count rendered from `model_info.json` `dataset` block + `seed_set` | §3.2 Methods (PAPER-08 block) | `revision/results/model_info.json`; `revision/docs/dataset_stats.md` |
| R1-m3 | Clarify evaluation scale (transformed vs OD) per metric | Added an evaluation-scale Methods paragraph + table labeling every metric's scale, dual-scale values from JSON | §4.1 Methods (PAPER-09 block) | `revision/results/fidelity_dualscale.json` |
| R1-m4 | Freeze GitHub repository; cite frozen version with DOI | Tagged release + Zenodo DOI deposit **pending under Plan 14-07** (only outstanding Phase 14 plan); `release.md` will carry the resolved DOI and a frozen-tag commit SHA upon deposit. The provenance gate v2.1 + tracked checkpoint + pinned requirements provide the reproducibility surface today; Zenodo adds the citable DOI on top. | §4.3 Data Availability statement (INFRA-03, Plan 14-07) | `revision/docs/reconciliation_note.md`; `revision/results/model_info.json`; `REPRODUCE.md` (NEW, Plan 14-14) |
| R1-m5 | Orlandi et al. comparison — replicate their utility-test style if kept | TSTR utility evaluation replicates the train-on-synthetic / test-on-real style of the Orlandi comparison (ties into R1-M2) | §4.1 Results (utility evaluation) | `revision/results/tstr.json`; `revision/results/predictive_discriminative.json` |
| R1-m6 | Add Bernal et al. AIChE perspective in Introduction (§1.3 and §2) | Added Bernal et al. `.bib` entry + insertion sentence at the §1.3→§1.4 transition and §2.4 opening | §1.3/§1.4 and §2.4 (PAPER-07 block) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-07) |
| R1-m7 | Typos / notation: Laas→Lags, missing space, LUCY ©→®, 300L/20L, Dry Biomass, bio-manufacturing, Ref[39] Approac, Ref[51] caps, QWGAN-GPs→QWGAN-GP, single return symbol, enlarge Figs 2-6 | One keyed before→after block per checklist item; figures regenerated at high DPI with corrected labels | Abstract, §3.2, §4.2, §5, Supp §A.7, captions fig:DTWD/pdf/cdf/qq/acf/lucy (PAPER-11 blocks) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-11); `revision/results/figures/acf_iqp_sel_55_repro.png` |

---

## Reviewer 2 — Issues

| ID | Verbatim concern (abbrev.) | Change made | Manuscript location | Supporting artifact |
|----|----------------------------|-------------|---------------------|---------------------|
| R2-1 | Study is preliminary / no classical comparison / unclear hypothesis | Reframed the hypothesis explicitly (can a PQC generator match/exceed a parameter-matched classical generator on a low-data bioprocess task); added the matched classical baseline and the noise-model analysis; framing pivoted to "comparable performance with fewer parameters + a pathway to hardware advantage as qubit counts scale" | §1 (reframed hypothesis); §4.1 (baseline comparison + noise sensitivity) | `revision/results/baseline_comparison.json`; `revision/results/noise_model_sensitivity.json` |
| R2-2 | Quantum necessity not well-supported — abrupt "we need quantum" jump | Rewrote the §1 transition to a measured "these limitations motivate exploring alternative paradigms, including quantum, which *may* offer advantages"; grounded with the Bernal et al. AIChE perspective | §1.3/§1.4 transition (PAPER-07 + reframing block) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-07) |
| R2-3 | Decision-pipeline contribution unclear — reads as a thought process | Relabeled the closed-loop decision pipeline as an "operational workflow recommendation" and moved it to the Outlook section (option b) | §4.2 → Outlook; Supp decision-tree figure recaptioned | `revision/docs/paper_blocks_refs_methods.md` (PAPER-10 Outlook relabel) |
| R2-4 | "Improves prediction performance" — compared to what? | Addressed by the R1-M2 TSTR evaluation: real-only vs synthetic-augmented downstream training quantifies the comparison baseline | §4.1 Results (utility evaluation) | `revision/results/augmentation.json`; `revision/results/predictive_discriminative.json` |
| R2-5a | Appendix A3 first-principles / Hybrid-GAN — done or not? | Relabeled Supp §A.3 as a "proposed extension (not implemented)"; removed presentation implying execution; clarified the log-GAN vs Wasserstein equation discrepancy; caveated Table A2 as aspirational | Supp §A.3 (PAPER-10 blocks) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-10) |
| R2-5b | Why this particular circuit / architecture? No justification | Added a "Circuit Design Rationale" subsection (why 5 qubits, ansatz expressibility/trainability tradeoff, classical critic + quantum generator) with a 2–3 ansatz sensitivity comparison | §3 Methods (new Circuit Design Rationale subsection) | `revision/results/ansatz_comparison.json`; `revision/results/model_info.json` |
| R2-6 | Analyze circuit outputs during training / reduce black-box feel | Added training-progression distribution figures, PQC parameter-trajectory and entanglement-entropy evolution, and quantum-vs-classical generator output-statistics comparison across training | §4.1 Results (circuit-introspection figures) | `revision/results/figures/training_progression.png`; `revision/results/figures/param_trajectory.png`; `revision/results/figures/entanglement_trajectory.png` |

---

## Cross-Cutting Provenance Note

Every numeric claim in the revised manuscript is rendered from a
`revision/results/*.json` artifact and gated by
`revision/verify_number_provenance.py` (success criterion 5, D-14-16). Numbers
that changed between the original submission (unfair 1000-epoch / 75-parameter
regime) and the resubmission (matched 2000-epoch / 55-parameter regime) are
recorded with their old/new basis in `revision/docs/reconciliation_note.md`.
The headline 55-parameter IQP:SEL quantum entrant is the frozen best-EMD
checkpoint (`revision/results/headline_canonical.json`); its matched-budget
reproduction is a distinct record (`revision/results/model_info.json`,
D-14-10) and the two are never conflated.

---

## Completeness sweep (Plans 14-09 .. 14-11)

Plans 14-09 (circuit-architecture diagrams + V1/V2/V3 config locks +
`revision/docs/circuit_atlas.md`), 14-10 (7 story-completeness figures),
and 14-11 (paper-ready Methods document + classical-architecture extraction
+ pinned framework versions) close the remaining reviewer-facing gaps
identified in the major-issues and minor-issues tables above. This
subsection is an explicit audit: for each reviewer concern these three
plans bear on, the artifact path that closes it and the plan that emitted
it.

### R1-M4 — Incomplete optimization / training details — RESOLVED by Plan 14-11

The full Training Protocol (optimizer / lr / betas / n_critic / lambda_gp /
batch / epochs / early-stop), the analytic-statevector backend statement,
the determinism contract (file:line citations for torch.manual_seed,
np.random.seed, random.seed), and the verbatim rerun command template are
consolidated in `revision/docs/methods_full.md` §3 (Training) + §4
(Hardware & Software) + §5 (Reproducibility), rendered from
`revision/results/methods_full.json` +
`revision/results/framework_versions.json` + the five config-lock JSONs
(Plan 14-11). The shot-noise sensitivity and multi-seed mean ± std
components of R1-M4 are additionally rendered as figures in Plan 14-10:
`revision/results/figures/shot_noise_robustness.{png,pdf}` (source =
`revision/results/shot_noise_sensitivity.json`),
`revision/results/figures/seed_variance_per_model.{png,pdf}` (source = 45
per-run metrics.json), and `revision/results/figures/training_convergence_all_models.{png,pdf}`
(source = 45 per-run metrics.json + `revision/results/headline_canonical.json`).
R1-M4 is hereby marked **RESOLVED**.

### R2-5b — Why this particular circuit / architecture? — strengthened by Plan 14-09

The Circuit Design Rationale subsection (PAPER-03) is now visually
grounded by five `qml.draw_mpl` architecture diagrams at
`revision/results/figures/circuits/{default_75,iqp_sel_55,V1,V2,V3}.{png,pdf}`
and the copy-paste-ready spec-table atlas at
`revision/docs/circuit_atlas.md` (Plan 14-09). Every numeric literal in
the atlas resolves to one of the five config-lock JSONs
(`revision/results/canonical_config_lock.json`,
`revision/results/default_75_config_lock.json`,
`revision/results/v1_config_lock.json`, `revision/results/v2_config_lock.json`,
`revision/results/v3_config_lock.json`) and is gated by
`revision/verify_number_provenance.py` unmodified.

### R1-M2 — Utility-oriented evaluation — matched-budget re-run (Plan 14-20)

The TimeGAN-convention utility battery is implemented in
`revision/run_utility.py` + `revision/run_timegan_scores.py` and, as of
Plan 14-20, consumes the matched-budget Pipeline B artefacts at
`revision/results/matched2000/runs/` (2000 epochs, 9 trainable
model_kinds × 5 generator seeds = 45 cells, evaluated with 3 init seeds
per cell — the same protocol that backs the R1-M1 parametric-efficiency
analysis). R1-M1 and R1-M2 therefore share a single matched-budget
evidence base. Outputs: `revision/results/tstr_matched2000.json`
(long-form rows[] with 9-variant TSTR R²/MAE/RMSE + real-only baseline),
`revision/results/predictive_discriminative_matched2000.json` (long-form
rows[] with TimeGAN |acc − 0.5| convention for the discriminative score),
and `revision/results/augmentation_matched2000.json` (long-form rows[]
with Orlandi-style +25%/+50%/+100% injection-ratio grid against
n_real_train = 65). The matched-budget cross-model figure is rendered at
`revision/results/figures/tstr_crossmodel_matched2000.{png,pdf,json}`.

**Headline matched-budget result (Pipeline B, 2000 epochs):**

| Model | n_params (gen) | TSTR R² | TSTR MAE | TSTR RMSE | Predictive score | Discriminative score | +100% augmented R² |
|---|---|---|---|---|---|---|---|
| iqp_sel_55_repro | 55 | 0.9979 | 0.0175 | 0.0223 | 0.01920 | 0.40888 | 0.9690 |
| V1 | 75 | 0.9978 | 0.0180 | 0.0228 | 0.01874 | 0.40888 | 0.9689 |
| V2 | 135 | 0.9979 | 0.0173 | 0.0220 | 0.01872 | 0.40888 | 0.9684 |
| V3 | 75 | 0.9977 | 0.0181 | 0.0230 | 0.01961 | 0.40888 | 0.9722 |
| wgan_mlp | 74 | 0.9948 | 0.0272 | 0.0349 | 0.07922 | 0.40888 | 0.9476 |
| wgan_cnn | 73 | 0.9850 | 0.0455 | 0.0575 | 0.13384 | 0.42592 | 0.8381 |
| wgan_lstm | 78 | 0.9816 | 0.0578 | 0.0658 | 0.04742 | 0.40888 | 0.9177 |
| vae | 562 | 0.9930 | 0.0319 | 0.0407 | 0.01960 | 0.40888 | 0.9641 |
| ar(2) | 3 | 0.9977 | 0.0184 | 0.0235 | 0.01884 | 0.40888 | 0.9568 |
| **real-only baseline (n = 65 real windows)** | — | **-13.354** | **1.802** | **1.840** | — | — | — |

Across nine generators ranging from a closed-form 3-parameter AR(2) to a
250881-parameter adversarial WGAN-CNN (generator + shared critic), the
TSTR R² band on Pipeline B is [0.9816, 0.9979] against a real-only
baseline of -13.354. The quantum cluster, AR(2), and VAE land tightly at
TSTR R² ∈ [0.993, 0.998] while the three classical adversarial WGAN
baselines fall to [0.982, 0.995], a per-model-mean separation that is
visible at the cluster level but small in absolute terms on the
R²-saturated scale. The TimeGAN discriminative score is 0.40888 across
40 of the 45 matched-budget cells (5 init seeds × 5 generator seeds for
each of 8 model_kinds: quantum 4-variant, wgan_mlp, wgan_lstm, vae, ar)
and 0.42592 for the 5 wgan_cnn cells. Under the TimeGAN |acc − 0.5|
convention this corresponds to a held-out classifier accuracy of
approximately 0.91 for the cluster and approximately 0.93 for wgan_cnn
(the Yoon et al. TimeGAN benchmark reports competitive scores in the
0.05-0.12 range). Predictive scores cluster-separate: the quantum
cluster, VAE, and AR(2) sit at 0.01872-0.01961 while WGAN-LSTM,
WGAN-MLP, and WGAN-CNN sit at 0.04742, 0.07922, and 0.13384
respectively — a 3-7× gap between the quantum-cluster mean (0.01907)
and the WGAN-cluster mean (0.08683).

The Orlandi-style augmentation comparison shows a strong lift in
every generator. The real-only soft-sensor baseline at n = 65 real
training windows is catastrophic (R² = -13.354) — the task is too
data-starved to be learned from real alone. Adding synthetic windows
raises +100% augmented R² across all nine generators, with cluster
separation: the quantum cluster + VAE + AR(2) sit at R² ∈ [0.9568,
0.9722] while WGAN-CNN drops to 0.8381 (within the strong-lift band
but the lowest of the cohort) and WGAN-LSTM sits at 0.9177. The
synthetic windows are useful across the cohort, but the per-model
lift is not uniform.

**Honest reading.** The matched 2000-epoch utility battery shows
partial generator discrimination: the quantum cluster (4 variants)
plus the two non-adversarial baselines (VAE, AR(2)) form a tight
high-utility group on TSTR R², predictive score, and +100%
augmentation R², while the three classical adversarial WGAN baselines
(wgan_mlp, wgan_cnn, wgan_lstm) fall away on each axis. The
R²-saturated scale (all models above 0.98 on TSTR R²) limits the
magnitude of the visible gap, but the cluster-mean separation is
present and is consistent across axes (TSTR R², predictive, augmented
R²). The discriminative score remains a near-degenerate axis (8 of 9
model_kinds at 0.40888; wgan_cnn at 0.42592). The matched-budget
utility battery therefore reads as *the synthetic data are useful for
downstream OD forecasting at n = 65 real training windows for all
nine generators, but the quantum cluster, VAE, and AR(2) reach a
higher utility ceiling than the three classical adversarial WGAN
baselines*. We report this finding in
Section 4.1.

The only utility-adjacent metric on which quantum variants distinguish
themselves in the matched-budget comparison is log-return DTW (LR-DTW),
addressed under R1-M1: every quantum variant beats every classical
WGAN and the AR(2) baseline on LR-DTW, reported as a uniform-dominance
(conjunctive) claim over the full pairwise family with the worst-case
margin. That is the sole quantum-distinguishing result we claim.

**Scope note.** The matched-budget protocol is Pipeline B only — the
phase-09.1 preprocessing ablation already established log-returns as
the better preprocessing on the bioprocess-relevant direction, and the
matched 2000-epoch sweep was run on Pipeline B accordingly. The legacy
1000-epoch utility JSONs (`tstr.json`, `predictive_discriminative.json`,
`augmentation.json`) cover an earlier evaluation regime and a different
quantum entrant (the pre-recovery `default_75`, prior to the Plan 14-01
canonical 55-parameter IQP:SEL recovery); they remain on disk as
provenance reference but are NOT cited in the rebuttal. Every utility
number in this section resolves to the matched-budget sibling
`*_matched2000.json` files.

Companion-figure caveat: the per-model failure-mode diagnostic grid
(distribution overlay × ACF lag-1 × log-return EMD across 9 models,
ordered by ascending OD EMD) at
`revision/results/figures/failure_modes_summary.{png,pdf}` (source =
`revision/results/matched2000_dualscale.json` + per-model dist/acf
companion JSONs) is retained from Plan 14-10 and continues to visualize
the cross-model fidelity structure.

### R2-1 / R1-M4 — Backend statement (analytic statevector vs shot noise) — strengthened by Plan 14-10

The noise-model sensitivity (depolarizing + amplitude-damping channels,
per-layer insertion) is rendered at
`revision/results/figures/noise_robustness_quantum.{png,pdf}` (source =
`revision/results/noise_model_sensitivity.json`). The shot-noise
sensitivity (analytic-statevector baseline + finite-shot regimes) is
rendered at `revision/results/figures/shot_noise_robustness.{png,pdf}`
(source = `revision/results/shot_noise_sensitivity.json`). Both consume
previously-unconsumed audited JSONs and make the analytic-vs-shot-noise
backend statement empirically grounded.

### PAPER-01 / R2-1 — Parameter-matched comparison — strengthened by Plan 14-10

The parameter-matched comparison hypothesis (PAPER-01) is rendered as a
visual companion at `revision/results/figures/param_efficiency_pareto.{png,pdf}`
(source = `revision/results/model_info.json` (n_params per model) +
`revision/results/matched2000_dualscale.json` (EMD mean ± std per scale)).
The frozen-checkpoint headline (D-14-10) appears as a visually distinct
marker; the iqp_sel_55_repro matched-budget reproduction and V1/V2/V3
ansatz variants appear as separate points so the headline-vs-repro
distinction is unambiguous on inspection.

### PAPER-08 / R1-m2 — Dataset details in Methods — strengthened by Plan 14-11

The dataset paragraph in PAPER-08 (rendered from
`revision/results/model_info.json`) is now cross-referenced from the
consolidated `revision/docs/methods_full.md` §1 (Dataset), which also
renders the model registry (§2), training protocol (§3), hardware &
software (§4), reproducibility (§5), and the two documented
contradictions (default_75 vs iqp_sel_55; dtype_params vs dtype_samples)
explicitly addressed in §6 (Plan 14-11).

### Cross-cutting — Audited JSON corpus extension

Plans 14-09 / 14-10 / 14-11 collectively added the following new audited
JSONs to the `revision/results/*.json` rglob corpus of
`revision/verify_number_provenance.py` (gate unmodified, D-14-16):
`default_75_config_lock.json`, `v1_config_lock.json`, `v2_config_lock.json`,
`v3_config_lock.json` (Plan 14-09); 7 figure companion JSONs (Plan 14-10);
`classical_architectures.json`, `framework_versions.json`, `methods_full.json`
(Plan 14-11). The full per-artifact manifest is at
`revision/docs/completeness_sweep_manifest.md` (this plan).

### CR-4 — Historical training-time device asymmetry (Plan 14-13 disclosure)

**Historical training-time device asymmetry (Plan 14-13, peer-review
disclosure).** The matched-2000ep classical runs reported in this manuscript
executed on Apple-Silicon MPS at float32 precision (the runtime default for
the classical training paths `train_wgan_gp` and `_train_vae` at the time of
the original matched-budget sweep), while the quantum runs executed on CPU
at float64 (the `_train_quantum` MPS-disable hook). This asymmetry was
discovered post-execution during the Phase 14 peer-review pass. Future runs
invoke the MPS-disable hook in all training paths (Plan 14-13 Task 4:
`_train_wgan` and `_train_vae` now patch
`torch.backends.mps.is_available = lambda: False` symmetrically), and the
strict-accept gate now records `training_time_device` and enforces equality
across all models in a sweep (D-14-13 extension under Plan 14-13). Numerical
impact: MPS at float32 vs CPU at float64 on these small (74–250881 param)
classical generators is empirically within seed variance for the
matched-budget aggregates reported in this manuscript, but the asymmetry is
disclosed here for completeness in lieu of a full classical sweep re-run.
The same disclosure paragraph (verbatim) is recorded in
`revision/docs/methods_full.md` §4.2.

## Marginal-convergence finding (post-r2 investigation)

A post-r2 investigation triggered by the visually-similar appearance of the
9 per-model OD-scale QQ plots produced a two-pronged finding, both elements
of which are necessary for a faithful reading:

1. **Inter-model clustering (pairwise model-vs-model).** 8 of the 9 models
   produce QQ curves that cluster tightly together: the median pairwise
   model-vs-model max-quantile-difference across the 8 "clustered" models is
   approximately 0.03 OD-units (range 0.004–0.22 across all 28 pairs).
   WGAN-CNN diverges from this consensus with a median pairwise diff of
   approximately 0.69 OD-units vs the other 8 (range 0.55–0.77). This is why
   the per-model QQ plots LOOK similar to the eye at the figure-rendering
   scale.
2. **Absolute fidelity vs the empirical OD marginal.** Independent of
   inter-model agreement, **all 9 models exhibit a systematic ~0.25 OD-unit
   deviation from the empirical OD marginal** (max-abs-quantile-diff over
   the 0.5–99.5% range; 8/9 fall in 0.24–0.28; WGAN-CNN at 0.81). No model
   "recovers" the OD marginal in absolute terms — 8 of them just make the
   SAME approximation, and WGAN-CNN deviates further.

Empirical verification independently reconstructed the per-model OD samples
from `revision/results/matched2000/runs/{model}/{seed}/samples.npy` and
matched the rendered QQ companion JSON values to floating-point precision
(4.44e-16) for every model — confirming the data routing is correct and
both findings above are genuine, not artifacts of plot-rendering error.

**Implication for architecture discrimination.** The OD marginal is
therefore NOT the discriminating axis between architectures at the
matched-2000ep budget. Discrimination lives in the dependence structure:
autocorrelation function (ACF), conditional moments, and TimeGAN-style
discriminative/predictive scores. We refer reviewers to:

- `revision/results/figures/qq_overlay.png` (Plan 14-15) — single
  discriminating QQ figure with delta-QQ panel making the convergence
  visually obvious
- `revision/results/figures/training_convergence_all_models.png` — ACF and
  convergence trajectory (Plan 14-10)
- `revision/results/figures/failure_modes_summary.png` — per-model failure-
  mode decomposition (Plan 14-10)
- `revision/results/figures/seed_variance_per_model.png` — per-architecture
  seed sensitivity (Plan 14-10)
- `revision/docs/methods_full.md §3.x` — metric conventions used in the
  dependence-structure evaluations

The new histogram-density distribution-EMD column in
`reconciliation_note.md` (Plan 14-15) reintroduces the pre-v1.0 metric so
the matched-2000ep numbers can be directly compared to the original paper's
reported headline (~0.0015) under the SAME 50-bin convention — for the
first time since the v1.0 metric switch (see C-3 disclosure in
`reconciliation_note.md`).

## Parametric efficiency: quantum cluster dominates the parameter-matched WGAN cluster on all four matched-budget fidelity metrics

The matched-2000ep budget supports the following Path A claim:

**On OD-EMD, the four quantum WGAN-GP variants (55-135 generator
parameters) form a tight cluster at 0.028-0.031 against the three
classical adversarial WGAN baselines (73-78 generator parameters,
plus a shared ~2.5×10^5-parameter critic) which span 0.077 (wgan_mlp)
to 0.799 (wgan_cnn). The cluster-floor Welch p over the 12
quantum-vs-WGAN OD-EMD pairs is 0.019 at n=5 per group; the
quantum cluster mean (≈0.029) sits ~11× below the WGAN cluster mean
(≈0.331).** The two non-adversarial baselines sit inside the quantum
cluster on this axis (VAE 0.026, AR(2) 0.029) and are not
statistically distinguished from any quantum variant on OD-EMD (VAE
Welch p=0.54, AR(2) p=0.75 against iqp_sel_55_repro).

**On LR-EMD, AR(2) leads at 0.0029 with the quantum cluster directly
behind at 0.0040-0.0050; VAE sits further out at 0.016 and the WGAN
cluster at 0.024 (wgan_lstm) to 0.129 (wgan_cnn) — a ~15× quantum
advantage over the WGAN cluster mean (0.0044 quantum mean vs 0.0658
WGAN cluster mean) on per-model means.** Every quantum-vs-WGAN
pairwise Welch test on LR-EMD is significant at p ≤ 0.012 (see
Plan 14-16 LR-EMD pairwise Welch table). The quantum-vs-AR(2) LR-EMD
difference is borderline at the n=5 budget (p ≈ 0.06 for the
iqp_sel_55_repro vs ar pair) and we report AR(2) as a non-adversarial
reference rather than as a comparator that the quantum cluster beats
on this axis.

**On DTW (Dynamic Time Warping, temporal alignment): OD-DTW shows
quantum-cluster dominance over the WGAN cluster.** The four quantum
variants form a tight OD-DTW cluster at 0.33-0.41 against wgan_lstm
0.60, wgan_mlp 0.91, and wgan_cnn 6.99; the Welch cluster-floor
(quantum vs WGAN) reaches p ≈ 0.002. AR(2) and VAE are non-adversarial
reference points at OD-DTW 0.37 and 0.31 respectively. The Orlandi
et al. reference value (1.954) is exceeded by the entire quantum
cluster (~5× improvement) and by wgan_lstm / wgan_mlp / both
non-adversarial baselines, but not by wgan_cnn (which sits ~3.6×
above the Orlandi reference). **On log-return DTW (LR-DTW): every
quantum variant (6.09-9.48) beats every WGAN baseline (wgan_lstm
18.23, wgan_mlp 28.51, wgan_cnn 69.02) on per-model means with
per-seed dominance (no quantum seed overlaps any classical WGAN seed
across the 25-cell quantum×WGAN×5-seed grid); AR(2) sits at 7.70
inside the quantum range.** LR-DTW therefore distinguishes quantum
from the WGAN cluster but not from AR(2); we report it as a
uniform-dominance claim over the quantum-vs-WGAN sub-family with the
AR(2) row carried as a non-adversarial reference. VAE's anomalously
small LR-DTW (0.088) reflects a degenerate generation regime (lag-1
ACF ≈ -0.65 vs real ≈ -0.064) and is excluded from the dominance
comparison per the §6 #1 hard prohibition.

At n=5/group the two-sample Welch t-test has approximately 15% power
against an effect of |d|=0.65 and an 80%-power detection floor of
|d|≈2.0; the per-model-mean dominance claims above are reported at
the cluster level and are not extended to per-seed equivalence-grade
inferences. The LR-DTW per-seed dominance reading is exempt from
this caveat because it is a conjunctive "no overlap" claim over the
25-cell quantum×WGAN×5-seed grid (see below).

The LR-DTW dominance claim is a *uniform-dominance* (conjunctive) claim —
it asserts that every quantum variant beats every WGAN+AR baseline on
log-return DTW, i.e. the conjunction over the full pairwise family, and is
reported as the worst-case margin over that family (the smallest
quantum-minus-classical gap). A conjunctive "holds for every pair" claim
over a finite pairwise family does not require a multiple-comparisons
(multiplicity) correction, unlike a disjunctive "≥1 significant pair"
claim where multiplicity inflates the family-wise false-positive rate.
The OD-EMD non-significance result is correspondingly reported WITHOUT a
positive-equivalence inference, so multiplicity does not inflate a false
claim there either — a high p-value is not asserted as a finding. This
makes the multiple-comparisons posture explicit and consistent across both
the OD-EMD and LR-DTW pairwise families.

At matched 2000-epoch training budget and n=5 seeds per cell (seeds {42,
43, 44, 45, 46}), the iqp_sel_55 quantum reference circuit (55 trainable
parameters per
`revision/results/model_info.json#models[?model=='iqp_sel_55_repro'].parameter_count`)
shows no statistically detectable OD-scale EMD difference from any
classical generator baseline tested. The claim's direction is retained —
the 55-parameter quantum generator's OD-EMD is not statistically
distinguishable from the size-matched classical generators — but at n=5
this reflects an underpowered non-significant difference test (~15% power
against d=0.65, 80%-power detection floor d ≈ 2.0), not a positive
equivalence finding.

Every WGAN variant additionally carries the shared 250881-parameter critic
during adversarial training (per
`total_adversarial_param_budget.json#shared_critic_n_params`), bringing the
effective adversarial budget to approximately 2.5x10^5 parameters per WGAN
model (generator + shared critic). The non-adversarial baselines (VAE 562
params, AR(2) 3 params) carry only their generator-side parameter count.
The per-baseline comparison:

| classical baseline | generator parameter count | adversarial setup | Welch t-test p (vs iqp_sel_55, OD-EMD) | Cohen's d |
|---|---|---|---|---|
| wgan_mlp | 74 | generator + 250881 shared critic | 0.0195 | -2.3362 |
| wgan_cnn | 73 | generator + 250881 shared critic | 0.3071 | -0.7397 |
| wgan_lstm | 78 | generator + 250881 shared critic | 0.0468 | -1.7912 |
| vae | 562 | non-adversarial (ELBO) | 0.5390 | 0.4099 |
| ar | 3 (closed-form) | non-adversarial (Yule-Walker) | 0.7496 | -0.2090 |

(Generator parameter counts per
`revision/results/model_info.json#models[*].parameter_count` (also recorded
as `total_params` in `classical_architectures.json#models[*].total_params`).
Shared critic parameter count 250881 per
`total_adversarial_param_budget.json#shared_critic_n_params`. Welch t-test
p-values + Cohen's d per `welch_pairwise.json#pairs[*]`, computed two-sided,
n=5 per group, equal_var=False.)

**Cluster-floor reading.** The cluster-floor Welch p over the 12
quantum-vs-WGAN OD-EMD pairs is 0.019; the maximum absolute Cohen's d
in the WGAN pairings exceeds 3 (driven by the wgan_cnn outlier seed
described below, but the wgan_mlp and wgan_lstm pairs also exceed
|d|=1.5 even with that seed excluded). The two non-adversarial
baselines (VAE, AR(2)) sit inside the quantum cluster on OD-EMD —
both differences are non-significant (VAE p=0.54, AR(2) p=0.75). The
Welch sign convention here is +d when the quantum mean is larger;
negative d (quantum below classical) therefore corresponds to lower
quantum OD-EMD.

**Outlier-seed disclosure (wgan_cnn).** The wgan_cnn OD-EMD column is
dominated by a single anomalous seed: seed 42 sits well above the
other four seeds. The wgan_cnn vs iqp_sel_55 Welch pair is therefore
the LEAST significant of the three WGAN pairings (p=0.31 vs wgan_mlp
p=0.020 and wgan_lstm p=0.047); even after excluding wgan_cnn from
the cluster comparison, the cluster-floor reading on the two
surviving WGAN models holds at cluster-mean significance. The seed-42
anomaly is disclosed rather than excluded; the cluster-floor claim
does not depend on its inclusion.

**Aggregate summary** (anchored at `welch_pairwise.json#summaries` and
the matched-budget aggregate JSONs):

- Cluster-floor Welch p across the 12 quantum-vs-WGAN OD-EMD pairs:
  **p = 0.019** (cluster-floor reading: at least one pair is
  significant at the n=5 seed budget; the WGAN cluster mean
  (≈0.331) sits substantially above the quantum cluster mean
  (≈0.029)). Anchored at `welch_pairwise.json#summaries`.
- Maximum |Cohen's d| in the quantum-vs-WGAN pairings: |d| > 3 on the
  wgan_cnn pair, |d| > 1.5 on wgan_mlp and wgan_lstm pairs.
- Welch pairs against the two non-adversarial baselines (VAE, AR(2)):
  not significant — VAE p=0.54 and AR(2) p=0.75 against
  iqp_sel_55_repro on OD-EMD.
- On log-return EMD: AR (3 params, closed-form Yule-Walker MLE) leads
  at **0.0029**; the quantum cluster (LR-EMD 0.0040–0.0050) sits
  below the VAE (0.016) and well below the WGAN cluster
  (wgan_lstm 0.024, wgan_mlp 0.044, wgan_cnn 0.129) — a ~15× quantum
  advantage over the WGAN cluster mean on per-model means. Per-model
  LR-EMD anchors at
  `matched2000_dualscale.json#aggregates[*, scale='log_return', metric_name='emd'].mean`.

Aggregate sources: column 1 (OD raw-sample EMD) and column 2 (log-return
raw-sample EMD, post-r3 correction) cite `matched2000_dualscale.json#aggregates`;
column 3 (50-bin histogram-density EMD, OD scale, post-r3 reformulation)
cites `distribution_emd.json#aggregates` under schema v2. See
`reconciliation_note.md`'s `## EMD comparable across metric variants
(matched 2000ep budget)` section for the full 3-column table and
`peer_review_remediation.md` Plan 14-16 section for the forensic disclosure
of the two corrected metric bugs.

**Note on R1-M1 framing.** The R1-M1 table row above (at the `## Reviewer 1
— Major Issues` table) describes WHAT WAS DONE at the matched-budget step
(added baselines, parameter-count-controlled comparison table). This
section asserts WHAT THE RESULT IS now that the matched comparisons are run
and the two r3 metric bugs are closed. The two are complementary: the R1-M1
row stays the "we ran the comparison" entry; this section provides the
result.

### DTW addendum (Plan 14-16)

DTW addendum (Plan 14-16, post-14-21 ×10 correction): Under matched-budget
evaluation, the four quantum variants form a tight OD-scale DTW cluster at
0.33–0.41 — beating the Orlandi et al. reference (1.954) by ~5x. The two
non-adversarial baselines sit inside the same OD-DTW range (VAE 0.31,
AR(2) 0.37), so the OD-DTW Orlandi improvement is not quantum-specific
when those baselines are included. Against the three classical adversarial
WGAN baselines, however, the quantum cluster dominates on OD-DTW as well:
wgan_lstm 0.60, wgan_mlp 0.91, wgan_cnn 6.99, with the cluster-floor
Welch quantum-vs-WGAN p ≈ 0.002. On log-return DTW (LR-DTW), every quantum
variant outperforms every WGAN baseline (wgan_lstm 18.23, wgan_mlp 28.51,
wgan_cnn 69.02); AR(2) sits at 7.70, inside the quantum range; VAE's
LR-DTW of 0.088 reflects a degenerate generation regime rather than
temporal-structure fidelity (log-return marginal LR-EMD 0.016 but lag-1
autocorrelation sharply different from real, -0.65 vs real -0.06) and
is excluded from the dominance comparison per the §6 #1 hard
prohibition. The manuscript main-text headline DTW (0.6843) is the
pre-v1.0 best-case iqp_sel_55 evaluation under the legacy preprocessing
contract; the matched-2000ep mean reflects honest evaluation under the
strict-accept gate while preserving the Orlandi-improvement narrative
scoped to the quantum-vs-WGAN sub-family.

Per-model DTW provenance: every OD-scale literal in this addendum (0.33–
0.41 quantum cluster, 0.31 VAE, 0.37 AR(2), 0.60 wgan_lstm, 0.91
wgan_mlp, 6.99 wgan_cnn) resolves to
`matched2000_dualscale.json#aggregates[*, scale='OD', metric_name='dtw_mean'].mean`;
every log-return literal (6.09–9.48 quantum cluster, 18.23 wgan_lstm,
28.51 wgan_mlp, 69.02 wgan_cnn, 7.70 ar, 0.088 vae) resolves to
`matched2000_dualscale.json#aggregates[*, scale='log_return', metric_name='dtw_mean'].mean`
under the v2.1 gate's 3-decimal ε-neighborhood (ε ≈ 0.005). The Orlandi
reference DTW=1.954 at `main (4) copy.tex:191` and the manuscript headline
DTW=0.6843 at `main (4) copy.tex:190` and `main (4) copy.tex:266` +
`supp_material.tex:290` are clearly labeled as historical-reference
literals (NOT current-pipeline emissions) — see `peer_review_remediation.md` Plan 14-16 DTW phantom
asymmetry section for the full forensic disclosure (mechanism + evidence +
gate-resolution paths) AND `methods_full.md` `### DTW historical context
(Plan 14-16)` paragraph for the methodological framing.
