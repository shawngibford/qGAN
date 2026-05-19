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
| R1-M2 | Validation is diagnostic only — need utility-oriented tests (TSTR, predictive/discriminative) | Added TSTR (train-on-synthetic, test-on-real soft sensor), predictive score, discriminative score, and real-only vs synthetic-augmented comparison | §4.1 Results (new utility-evaluation subsection) | `revision/results/tstr.json`; `revision/results/predictive_discriminative.json`; `revision/results/augmentation.json` |
| R1-M3 | Log-returns + Lambert W may strip temporal structure; no OD back-transformation | Added original-OD-scale results (generate → invert → metrics on physical units); ACF on both transformed and OD scale; explicit per-metric scale statement; growth-rate justification of log-returns | §3 Methods (evaluation-scale paragraph + Table); §4.1 (dual-scale ACF) | `revision/results/fidelity_dualscale.json`; `revision/docs/dataset_stats.md` |
| R1-M4 | Incomplete optimization / training details (n_critic, λ, LR, epochs, stopping, seeds, analytic vs shot) | Added full Training Protocol (all hyperparameters rendered from JSON); stated analytic statevector (no shot noise) backend; added shot-noise sensitivity; multi-seed (5 seeds) mean ± std; clarified Supp Eq. A3 log-GAN vs Wasserstein discrepancy | §3 Methods (Training Protocol); Supp §A.3 (PAPER-10 block) | `revision/docs/training_protocol.md`; `revision/results/shot_noise_sensitivity.json`; `revision/results/multiseed_summary.json` |
| R1-M5 | Claim calibration — language oversells a simulator-based, single-variable, single-campaign proof-of-concept | Toned language to "proof-of-concept feasibility study"; moved decision-tree workflow + Hybrid-GAN to a labeled Outlook; caveated Supp Table A2 as aspirational; clarified 20L/300L; softened "exponential compactness"/"reduced mode collapse" to literature-motivated hypotheses | Abstract; §1; §4.2–4.4; §5; Supp §A.3 (PAPER-02/05/10/11 blocks) | `revision/docs/paper_blocks_refs_methods.md`; `revision/results/model_info.json` |

## Reviewer 1 — Minor Issues

| ID | Verbatim concern (abbrev.) | Change made | Manuscript location | Supporting artifact |
|----|----------------------------|-------------|---------------------|---------------------|
| R1-m1 | Misplaced / weak references ([27][28][39][18][19][41][55]-[57][59]) | Per-reference `.bib`+sentence-rewrite surgery; explicit RETAINED note for the reviewer-confirmed anchors [21]-[23],[34]-[36],[61] | §1.3, §1.4, §2.4, §3.1, Supp §A.2 (PAPER-06 blocks) | `revision/docs/paper_blocks_refs_methods.md` (PAPER-06) |
| R1-m2 | Report dataset details (raw points, windows, splits, independent runs) | Added a Dataset-and-Preprocessing Methods paragraph, every count rendered from `model_info.json` `dataset` block + `seed_set` | §3.2 Methods (PAPER-08 block) | `revision/results/model_info.json`; `revision/docs/dataset_stats.md` |
| R1-m3 | Clarify evaluation scale (transformed vs OD) per metric | Added an evaluation-scale Methods paragraph + table labeling every metric's scale, dual-scale values from JSON | §4.1 Methods (PAPER-09 block) | `revision/results/fidelity_dualscale.json` |
| R1-m4 | Freeze GitHub repository; cite frozen version with DOI | Tagged release + Zenodo DOI workflow; Data Availability statement updated with the DOI; reproduce steps recorded | §4.3 Data Availability statement (INFRA-03, Plan 14-07) | `revision/docs/reconciliation_note.md`; `revision/results/model_info.json` |
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
