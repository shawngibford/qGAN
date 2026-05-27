---
created: 2026-05-27
updated: 2026-05-27 (post-swarm; rewrite complete)
purpose: Session handoff for rewriting main (4) copy.tex + supp_material.tex against the matched-budget evidence base ahead of AIChE aic-4719598 resubmission
status: SUPERSEDED — rewrite swarm complete at commit a50cb0f. Open .planning/PAPER-SUBMISSION-HANDOFF.md for the post-swarm submission-readiness state.
predecessor: .planning/REBUTTAL-HANDOFF.md (rebuttal-drafting state; load this if you want the per-comment response text)
successor: .planning/PAPER-SUBMISSION-HANDOFF.md (Wave 8 / post-swarm / submission readiness)
---

> **⚠️ THIS DOC IS HISTORICAL — DO NOT RE-EXECUTE THE WORK IT DESCRIBES**
>
> The paper-rewrite swarm consumed this handoff and executed it across 10 atomic commits between `d81306f` and `a50cb0f` on 2026-05-27. The manuscript is now submission-ready.
>
> **If you're a fresh session:** open `.planning/PAPER-SUBMISSION-HANDOFF.md` for the current state and Wave 8 (human ACT) checklist. This doc is preserved as the archeological record of what the rewrite plan asked for.
>
> **§4 (prohibitions) and §3 (load-bearing facts) below remain authoritative** as the prohibition list and JSON-traceability standard. The A5 peer-review simulator surfaced two additional empirical findings during the swarm that this handoff did not anticipate, both documented in PAPER-SUBMISSION-HANDOFF.md §5:
>
> 1. **LR-EMD asymmetry**: quantum models are statistically significantly WORSE than every classical adversarial baseline on the log-return marginal (AR=0.003, classical 0.007–0.013, quantum 0.014–0.015, VAE 0.016). The current §4.1 discloses this honestly. This is NOT a "withdrawn LR-EMD claim" violation per §4.1 — it is the OPPOSITE-direction scope-honest disclosure.
> 2. **OD-EMD pipeline-invariance**: the four quantum ansatze produce near-identical OD-EMD per seed (within 0.0002), because the inverse-preprocessing pipeline projects them to a near-identical OD support. Disclosed in §4.1.
>
> Also: **handoff §3 stated Adam β₁ = 0.5; the JSON ground truth (revision/results/model_info.json) is β₁ = 0.0**. The manuscript now uses 0.0; DECISIONS.md was synced.
>
> Also: **handoff §2.3 figure-caption examples list AR(2)=7.70 inside the classical adversarial range**. The actual classical adversarial range is **1.58–6.86** (wgan_lstm to wgan_cnn); AR(2)=7.70 is a 3-parameter non-adversarial reference. The current manuscript splits them correctly.

---

# Paper Rewrite Handoff — AIChE aic-4719598 (HISTORICAL)

## 0. Resume in 90 seconds

A fresh session opening this file should know:

1. **The work**: AIChE Journal submission aic-4719598 ("Quantum WGAN-GP for synthetic bioprocess time series"), in major revision (R1). 3-week extension granted on 2026-05-27; new deadline ≈ 2026-06-17. **Not withdrawing.**
2. **The remaining task**: Rewrite `main (4) copy.tex` and `supp_material.tex` so the *claims in the prose match the matched-budget evidence* that the revision actually produced. The rebuttal letter is already submitted; the manuscript itself still carries some residual framing that the rewrite needs to close.
3. **The freeze tags**: `v1.0-revision` (commit `52f30b9`) and `v1.1` (commit `ab7086c`, current) on `https://github.com/shawngibford/qGAN`. Cite v1.1 in the Data Availability statement.
4. **The non-negotiables**: Three corrections discovered during rebuttal drafting that the rewrite must respect:
   - **VAE is a degenerate generation regime, NOT posterior collapse.** Log-return std is 0.0186 (≈ real 0.0217), not 4×10⁻⁴. The anomaly is the lag-1 ACF (−0.648 vs real −0.064).
   - **LR-DTW**, not LR-EMD, is the surviving quantum-distinguishing signal. The LR-EMD claim was withdrawn during Plan 14-16 forensic remediation.
   - **Real-data lag-1 ACF reference is −0.064** (matched-pipeline, with dither), not −0.029 (raw, no dither).

---

## 1. State of the work (as of 2026-05-27)

### 1.1 What's been done

- **Phase 14 plans 14-01 through 14-20** complete (one plan 14-07 deferred to acceptance — Zenodo deposit).
- **Matched-budget protocol** executed end-to-end: 9 generators × 5 seeds × 2000 epochs × Pipeline B (log-returns); same critic across all WGAN-GP runs; same data; same hyperparameters; only the generator varies.
- **R3 forensic peer review** surfaced two metric bugs in earlier evaluation code; both fixed at root cause (Plan 14-16). The discovery process produced false intermediate claims (notably the VAE "posterior collapse std=0.0004" characterization) that have since been corrected.
- **Manuscript-side PAPER-01..11 keyed blocks** integrated via Plan 14-17 (commit `e7e6329` lineage) into `main (4) copy.tex` and `supp_material.tex`. These cover: de-overclaimed abstract, Circuit Design Rationale §3.1, log-return justification §A.7, Outlook §4.5, Hybrid-GAN relabel §A.3, Supp Table A2 aspirational caveat, 20 L/300 L LUCY fix, `r_t` notation unified, R1-m7 typo checklist.
- **Rebuttal letter** assembled and submitted by the user from drafts in `.planning/REBUTTAL-HANDOFF.md` (14+ reviewer comments answered).
- **bib.bib cleaned and aligned (2026-05-27)**: 59 entries, every cited key has exactly one bib entry, every bib entry is cited at least once. 4 new entries added (havlicek2019supervised, schuld2019quantum, rasmussen2006gaussian, bernal2022perspectives); 5 R1-m1-flagged obsolete keys removed; 17 pre-existing orphan keys removed. `bib.bib` is at the repo root.
- **GitHub release `v1.1`** published at `ab7086c` (https://github.com/shawngibford/qGAN/releases/tag/v1.1). Three new figures shipped with this release: `cross_model_dtw_dualscale`, `cross_model_acf_overlay`, `preprocessing_pipeline_4panel`.

### 1.2 Freeze state

- **Active SHA**: `ab7086c` (v1.1). Supersedes `52f30b9` (v1.0-revision) for any new citation. v1.0-revision is preserved as the original first-round-resubmission tag.
- **Freeze gate** (`./qgan_env/bin/python revision/verify_freeze_ready.py`): PASS on every gate except gate D (`release.md`), which is the post-acceptance Zenodo deliverable. Same posture as v1.0-revision.
- **Provenance gate** (`./qgan_env/bin/python revision/verify_number_provenance.py`): PASS on the 3 reviewer-facing docs (paper_blocks_framing.md, paper_blocks_refs_methods.md, reviewer_response.md). Every numeric literal in those docs resolves to a JSON value under `revision/results/`.
- **Working tree**: clean (post-v1.1).

### 1.3 What's NOT done — the gap this handoff covers

The manuscript prose has been *partially* recalibrated via PAPER-01..11 keyed edits, but several issues likely remain that a careful end-to-end read would surface:

- Title may still carry "industrial bioprocess monitoring" register that overclaims the proof-of-concept scope.
- Abstract de-overclaim removed "high fidelity" / "strong performance" but other residual phrases may need a fresh pass.
- §1.1, §1.4, §3.3 and other sections that weren't part of the PAPER-* keyed sweep may still carry deployable-framework language.
- The new figures (`cross_model_dtw_dualscale`, `cross_model_acf_overlay`, `preprocessing_pipeline_4panel`) ship in v1.1 but are **not yet inserted into the .tex**. Decisions needed: which go in main, which in supp, with what captions.
- The bifurcated empirical finding (OD-marginal no-separation + LR-DTW quantum dominance + lag-1 ACF quantum-closest-to-real) is the *substantive contribution* but is currently spread across several sections rather than centralized in a dedicated Results subsection.
- Some Methods sub-sections may still refer to the legacy 1000-epoch protocol that was superseded by the matched 2000-epoch budget.

---

## 2. What the rewrite must accomplish

### 2.1 Reframe the contribution honestly

The paper as published needs to read as a **proof-of-concept feasibility study** with **two specific empirical findings**, not as an industrial-deployment framework:

**Finding 1 — Parametric-efficiency equivalence on the OD-marginal.**
A 55-parameter PQC generator is statistically indistinguishable from parameter-matched classical WGAN-GP baselines (73–78 parameters) on the OD-marginal EMD at n=5 (max Welch *p* > 0.36, max |Cohen's *d*| ≤ 0.65). Reported as a **non-significant difference under low power**, not as equivalence — with n=5 the test has only ~15% power against d=0.65 and an 80%-power detection floor of d≈2.0. AR(2) at 3 parameters also matches, characterising the task more than the model.

**Finding 2 — Uniform quantum dominance on log-return temporal structure.**
Every quantum variant outperforms every classical adversarial / autoregressive baseline on log-return DTW. Worst-case quantum (V3 mean 1.122, max seed 1.224) < best-case classical (wgan_lstm mean 1.581, min seed 1.307); uniform dominance across the 12 quantum-vs-{WGAN,AR} per-model-mean cells and across the 60-cell per-seed grid. Quantum cluster lag-1 ACF (−0.09 to −0.10) is also closest to real (−0.064) of all 9 generators — an independent structural-fidelity finding consistent with the LR-DTW result. VAE is excluded from the LR-DTW comparison (degenerate regime: LR-DTW=0.088 but lag-1 ACF=−0.648, badly different from real).

**What the paper does NOT claim:**
- No quantum advantage on the OD-marginal distribution.
- No quantum advantage on the utility battery (TSTR R² band [0.993, 0.998] across all 9 generators on +100% augmentation; no separation).
- No demonstrated industrial-deployment capability.
- No demonstrated multivariate or longer-time-series capability.
- No demonstrated benefit from increasing qubit count, depth, or entangler topology past the 55-parameter configuration (V1 75p, V2 135p, V3 75p-linear all fail to improve on the compact circuit at matched budget).

### 2.2 Required structural rewrites

- **Abstract**: replace any remaining over-claims; lead with the bifurcated finding rather than a single headline.
- **§1 Introduction**: ensure §1.1, §1.2, §1.3, §1.4 all read as motivation for the *open empirical question* (per the §1.3 falsifiable framing already in place), not as motivation for a deployable framework.
- **§3 Methods**: confirm every Methods sub-section references the matched-budget protocol (2000 epochs, n=5 seeds {42,43,44,45,46}, shared 250881-param critic across all WGAN-GP runs).
- **§4 Results**: centralize the bifurcated empirical finding in a single subsection backed by the new figures. Currently distributed across multiple paragraphs.
- **§4.2 Key Contributions**: already de-overclaimed (PAPER-02); verify against §2.1 above.
- **§4.5 Outlook**: already in place (PAPER-05); verify the Hybrid-GAN and decision-tree material is clearly demoted.
- **§5 Concluding Remarks**: already de-overclaimed (PAPER-02); cross-check.

### 2.3 Figure integration

Three new figures exist on disk (`v1.1` artifacts) but are **not yet referenced** in either `.tex` file. Suggested integration:

| Figure | Suggested location | Caption sketch |
|---|---|---|
| `cross_model_dtw_dualscale` | Main §4.1 Results | "All 9 generators' DTW under matched 2000-epoch budget, OD-scale (left) and log-return scale (right). The LR-DTW panel shows uniform quantum dominance: every quantum variant (0.94–1.12) outperforms every classical adversarial baseline (WGAN-MLP 2.62, WGAN-CNN 6.86, WGAN-LSTM 1.58) and the AR(2) baseline (7.70)." |
| `cross_model_acf_overlay` | Main §4.1 Results | "Log-return ACF (lags 0–9) across all 9 generators with real-data reference (dashed black). Quantum cluster lag-1 ACF (−0.09 to −0.10) is closest to real (−0.064) of all 9 generators. VAE lag-1 ACF (−0.648) reflects the degenerate generation regime that excludes it from the LR-DTW comparison." |
| `preprocessing_pipeline_4panel` | Supplementary §A.7 | "Four-stage preprocessing pipeline: raw OD (n=778) → log-returns r_t = ln OD_t − ln OD_(t−1) (n=777) → standardized (μ=0, σ=1) → inverse Lambert W (δ=0.147) + rescale to [−1, 1]." |

---

## 3. Load-bearing facts to use in the rewrite

Every number below is verified against `revision/results/matched2000_dualscale.json` or the figure JSON companions. Quote these; do not introduce new literals.

### 3.1 Dataset (verified)

| Quantity | Value | Source |
|---|---|---|
| Raw OD time points | 778 | `model_info.json#dataset.raw_csv_rows` |
| Log-return observations | 777 | `model_info.json#dataset.log_return_rows` |
| Rolling window length | 10 | `model_info.json#dataset.window_length` |
| Rolling window stride | 2 | `model_info.json#dataset.window_stride` |
| Training windows | 384 | `model_info.json#dataset.rolling_windows` |
| Train/val/test split | 384 / 0 / 0 | Disclosed openly — single campaign, no held-out split |
| Independent seeds | n=5: {42,43,44,45,46} | `model_info.json#seed_set` |
| Real-data log-return lag-1 ACF | −0.064 (matched-pipeline, with dither) | `cross_model_acf_overlay.json` |

### 3.2 Models (verified)

| Model | Generator params | Notes |
|---|---|---|
| iqp_sel_55_repro | 55 | Headline quantum entrant — IQP encoding + 3 strongly-entangling layers, range topology |
| V1 | 75 | Depth-4 range — does not improve on iqp_sel_55 |
| V2 | 135 | Depth-8 range — does not improve |
| V3 | 75 | Depth-4 linear — does not improve |
| wgan_mlp | 74 | Parameter-matched classical WGAN-GP |
| wgan_cnn | 73 | Parameter-matched classical WGAN-GP; seed-42 OD-EMD outlier (0.159 vs others 0.020–0.034) |
| wgan_lstm | 78 | Parameter-matched classical WGAN-GP |
| vae | 562 | ELBO objective; degenerate LR-generation regime (excluded from LR-DTW comparison) |
| ar(2) | 3 | Closed-form Yule–Walker / lstsq; degenerate (matches OD-marginal trivially by moment-matching) |
| Shared WGAN-GP critic | 250881 | Conv1d-based; reused identically across all 4 quantum variants + 3 WGAN baselines (only generator varies) |

### 3.3 Training protocol (verified — matches `methods_full.md` §3)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.5, β₂=0.9) |
| Learning rate (generator / critic) | 6.9173×10⁻⁵ / 1.8046×10⁻⁵ (HPO-tuned, v1.1) |
| n_critic | 9 |
| λ_gp | 2.16 |
| Batch size | 12 |
| Training epochs | 2000 (matched budget; full duration; early-stop OFF) |
| Latent noise distribution | Uniform[0, 4π] |
| Backend | PennyLane `default.qubit`, `diff_method="backprop"`, CPU |

### 3.4 Per-model OD-EMD (verified — `matched2000_dualscale.json`)

| Model | Mean ± sample std (n=5) |
|---|---|
| iqp_sel_55_repro | 0.02753 ± 0.00513 |
| V1 / V2 / V3 | comparable to iqp_sel_55 within seed variance |
| wgan_mlp | 0.02595 |
| wgan_cnn | 0.05432 (mean inflated by seed-42 outlier) |
| wgan_lstm | 0.02821 |
| vae | 0.02573 |
| ar(2) | 0.02908 |

### 3.5 Per-model LR-DTW (verified — `cross_model_dtw_dualscale.json`)

| Model | Mean (n=5) |
|---|---|
| V1 | 0.9400 |
| V2 | 0.9495 |
| iqp_sel_55_repro | 0.9855 |
| V3 | 1.1225 |
| wgan_lstm | 1.5812 |
| wgan_mlp | 2.6243 |
| wgan_cnn | 6.8630 (wgan_cnn seed-42 individual outlier = 11.97 — same anomalous seed as OD-EMD) |
| ar(2) | 7.6991 |
| vae | 0.0876 (excluded — degenerate regime) |

**Range claim**: quantum 0.94–1.12, WGAN 1.58–6.86, AR(2) 7.70. Uniform dominance holds at both per-model-mean AND per-seed worst-case (max quantum seed = 1.224 < min WGAN seed = 1.307 < min AR(2) seed = 7.459).

### 3.6 Per-model lag-1 ACF, log-return scale (verified — `cross_model_acf_overlay.json`)

| Model | Lag-1 ACF (mean over 5 seeds) |
|---|---|
| **Real data** | **−0.0641** |
| iqp_sel_55_repro | −0.0949 |
| V3 | −0.0895 |
| V2 | −0.0968 |
| V1 | −0.0997 |
| wgan_cnn | −0.1112 |
| ar(2) | −0.1356 |
| wgan_mlp | −0.1418 |
| wgan_lstm | −0.2422 |
| **vae** | **−0.6482** (anomaly) |

Quantum cluster (−0.089 to −0.100) is closest-to-real of all 9 generators.

### 3.7 Statistical comparison (verified — `welch_pairwise.json`)

- Welch t-test across 20 quantum-vs-classical pairs (OD-EMD): max **p > 0.36**, max **|Cohen's d| ≤ 0.65**.
- Power at n=5 against d=0.65: **~15%**.
- 80%-power detection floor at n=5: **d ≈ 2.0**.
- TOST equivalence: **not satisfied at any defensible margin** — report as "non-significant under low power" rather than "demonstrated equivalence".

### 3.8 Shot-noise / hardware-noise robustness (verified — `shot_noise_sensitivity.json`, `noise_model_sensitivity.json`)

| Regime | OD-EMD (Pipeline B, mean over n=3 seeds) |
|---|---|
| Analytic statevector | 0.029676 |
| 1024 shots / expectation | 0.029682 |
| 8192 shots / expectation | 0.029675 |
| Max cross-regime difference | ≈ 8×10⁻⁶ (against per-seed std ≈ 6×10⁻³) |
| Depolarizing noise, 0% | 0.029676 |
| Depolarizing noise, 5% | 0.029691 |
| Amplitude damping, 5% | 0.029875 (max — 0.7% relative change from 0%) |

Both shot-noise and noise-channel sweeps use n=3 seeds (NOT n=1 — earlier draft said n=1, corrected).

### 3.9 Utility battery (verified — `tstr_matched2000.json`, `predictive_discriminative_matched2000.json`, `augmentation_matched2000.json`)

| Metric | Range across 9 generators (mean over 5 seeds) |
|---|---|
| TSTR R² (n=65-window soft sensor on synthetic OD) | [0.9930, 0.9977] |
| Predictive score (TimeGAN convention) | [0.0188, 0.0254] |
| Discriminative score (Pipeline B, all 45 cells) | **0.40888 fixed point** (structural artifact of cumsum back-transform — disclose, not interpret) |
| Real-only soft-sensor baseline (n_real=65) | R² = −13.354 ± 0.583 (catastrophic) |
| +25% augmentation R² | comparable across generators |
| +100% augmentation R² | [0.957, 0.971] |

**Honest read**: synthetic data are useful for the data-starved downstream task — real-only n=65 R²=−13.354 lifts to R² in [0.957, 0.971] for *every* generator. But no generator separates from any other on this battery at this scale. The Pipeline-B discriminative-score 0.40888 fixed point is a structural artifact of the cumsum back-transform, not a generator-quality signal — flag it as a finding, do not interpret.

### 3.10 Apparatus (verified)

- LUCY photobioreactor (Synoxis Algae), **20 L unit** used in this study (300 L is the geometrically identical production-scale configuration; same vendor, same sensor/actuator topology, same SALT circulation, same automated controller).
- Six vertical borosilicate glass tubes, 6 cm OD × 120 cm length, parallel.
- OD sensor: 880 nm.
- Data logged at 10-minute intervals by integrated DAQ.
- Single cultivation campaign.

### 3.11 Orlandi et al. reference (verified)

- Orlandi et al. (2024) reported DTW = 1.954 on a finance benchmark — the OD-scale DTW reference cited in §3.x.
- The matched-budget OD-DTW cluster (0.298–0.302 across all quantum variants AND the WGAN cluster) reports an ~6.5× improvement vs. this reference; the improvement is **matched-budget-wide, not quantum-specific** (wgan_mlp 0.302, wgan_lstm 0.301 are inside the cluster).
- The original manuscript headline DTW of 0.6843 is a **pre-revision best-case checkpoint** that does NOT reproduce under matched budget; preserved as labelled historical reference only.

---

## 4. Constraints, prohibitions, and lessons learned

### 4.1 Things the rewrite MUST NOT do

- **MUST NOT** claim quantum advantage on the OD-marginal. The OD-marginal can be matched by a 3-parameter AR(2); the equivalence is non-significant at low power, not demonstrated.
- **MUST NOT** characterize VAE as "posterior collapse" or cite "synthetic std ≈ 4×10⁻⁴". The actual log-return std is 0.0186 (≈ real 0.0217). The right framing is "degenerate generation regime — marginal well-aligned (LR-EMD=0.016), lag-1 ACF sharply different from real (−0.648 vs −0.064)".
- **MUST NOT** cite the LR-EMD-beats-WGANs result. That claim was withdrawn during Plan 14-16 forensic remediation (broken `density=True` column inverted on the corrected scale). LR-DTW is the surviving claim.
- **MUST NOT** cite the pre-revision DTW headline 0.6843 as a current result. It is preserved only as a labelled historical reference.
- **MUST NOT** use the real-data lag-1 ACF reference of −0.029. That value was computed without dither and is not apples-to-apples with the per-model evaluation. Use **−0.064** (matched pipeline).
- **MUST NOT** describe shot-noise or noise-channel sweeps as "n=1 representative seed". Both sweeps use **n=3 seeds {42,43,44}**.
- **MUST NOT** introduce numeric literals that don't trace to `revision/results/*.json`. The provenance gate will catch them.
- **MUST NOT** use "industrial bioprocess monitoring framework", "computational advantages", "deployable framework", "high fidelity", or "strong performance" framing anywhere outside an explicitly-labelled Outlook subsection.
- **MUST NOT** reintroduce the closed-loop-feedback-control framing for the decision-tree workflow. It is a decision-tree triage workflow, demoted to the Outlook.
- **MUST NOT** characterize Hybrid-GAN material in §A.3 as anything other than a *proposed extension that was not implemented or evaluated*.

### 4.2 Things the rewrite SHOULD do

- **SHOULD** lead with the bifurcated empirical finding (OD-marginal no-separation + LR-DTW quantum dominance + lag-1 ACF quantum-closest-to-real) as the core scientific contribution.
- **SHOULD** treat the LR-DTW + lag-1 ACF concordance as a single coherent structural-fidelity finding rather than two separate metric observations.
- **SHOULD** state the scope honestly throughout: single 778-point laboratory cultivation, 5 qubits inside the classically-simulable regime, n=5 seeds per cell.
- **SHOULD** scope future work explicitly: multivariate data, qubit counts past the simulable boundary, larger seed budget for TOST-grade equivalence on the OD-marginal, longer time series.
- **SHOULD** insert the three new figures with captions per §2.3 above.
- **SHOULD** preserve and respect the PAPER-01..11 keyed edits already integrated — do not undo de-overclaim work.

### 4.3 Style notes (from rebuttal drafting)

- **Word-paste-ready math**: when the prose target is .tex, use LaTeX. When the prose target is the Word rebuttal letter, use Unicode (`r_t` not `$r_t$`, `±` not `$\pm$`, `[−1, 1]` not `$[-1, 1]$`).
- **Honest acknowledgment of limitations** front-footed — n=5 power, AR(2) degeneracy, discriminative-score uniformity, VAE degeneracy, shot-noise n=3.
- **Provenance-gate-friendly literals**: ASCII minus `-` not Unicode `−` in .tex; no space-separated thousands (`250881` not `250 881`); preserve stored precision.

---

## 5. Resources available

### 5.1 Files of record

| File | Role |
|---|---|
| `main (4) copy.tex` | Main manuscript — primary target for rewrite |
| `supp_material.tex` | Supplementary — secondary target for rewrite |
| `bib.bib` | Cleaned + aligned bibliography (59 entries, compile-clean) |
| `REF.md` | R1-m1 reference surgery record |
| `revision/docs/methods_full.md` | Full methods document — every Methods-section claim traces from here |
| `revision/docs/reviewer_response.md` | Reviewer-facing rebuttal text, per-comment |
| `revision/docs/peer_review_remediation.md` | Forensic-remediation record |
| `revision/docs/paper_blocks_framing.md` | PAPER-01..05 (framing) keyed before/after LaTeX blocks |
| `revision/docs/paper_blocks_refs_methods.md` | PAPER-06..11 (refs + methods + typos) keyed blocks |
| `revision/docs/completeness_sweep_manifest.md` | Artefact-inventory ledger |
| `.planning/REBUTTAL-HANDOFF.md` | Per-comment rebuttal drafts + style guide (read for tone) |

### 5.2 Data sources (every numeric literal must resolve to one of these)

| JSON | Contents |
|---|---|
| `revision/results/matched2000_dualscale.json` | The master matched-budget evaluation: per-model × per-seed × per-scale × per-metric values (DTW, EMD, ACF lags 0-9, moments) |
| `revision/results/welch_pairwise.json` | Welch *p* + Cohen's *d* for the 20 quantum-vs-classical pairs |
| `revision/results/model_info.json` | Per-model architecture metadata, training-protocol notes, dataset shape |
| `revision/results/classical_architectures.json` | Classical generator + critic architecture metadata |
| `revision/results/tstr_matched2000.json` | TSTR utility results (matched-budget) |
| `revision/results/predictive_discriminative_matched2000.json` | Predictive + discriminative TimeGAN scores |
| `revision/results/augmentation_matched2000.json` | Real-only vs synthetic-augmented downstream training |
| `revision/results/shot_noise_sensitivity.json` | Analytic / 1024 / 8192 shots sweep, n=3 seeds |
| `revision/results/noise_model_sensitivity.json` | Depolarizing + amplitude-damping channels at 0/0.1/1/5%, n=3 seeds |

### 5.3 Figures (in `revision/results/figures/`)

Per-model (× 9 models): `loss`, `timeseries`, `dist`, `acf`, `qq`, `stylized`, `odrecon`, `emd` (last only for 7 adversarial models).
Cross-model: `cross_model_distribution`, `cross_model_emd`, `cross_model_dtw_dualscale` ⭐ NEW, `cross_model_acf_overlay` ⭐ NEW, `qq_overlay`, `matched2000_dualscale_sidebyside`, `training_convergence_all_models`, `seed_variance_per_model`, `failure_modes_summary`, `param_efficiency_pareto`, `tstr_crossmodel`, `tstr_crossmodel_matched2000`, `headline_vs_reproduction`.
Sensitivity: `shot_noise_robustness`, `noise_robustness_quantum`.
Preprocessing: `preprocessing_pipeline_4panel` ⭐ NEW, plus `transform_ablation/figures/*`.
Circuits: `circuits/iqp_sel_55`, `default_75`, `V1`, `V2`, `V3`.
Quantum-specific: `entanglement_trajectory`, `param_trajectory`, `training_progression`.

Each figure has same-stem PDF + PNG + JSON companion for traceability.

### 5.4 Verification gates

```bash
# Provenance: every literal in reviewer-facing docs resolves to a JSON value
./qgan_env/bin/python revision/verify_number_provenance.py

# Freeze readiness: clean tree, gitignore, provenance, tag scope, release.md
# Will FAIL on release.md (gate D) until acceptance — by design
./qgan_env/bin/python revision/verify_freeze_ready.py

# Rendering (idempotent — no retraining)
./qgan_env/bin/python -m revision.render_missing_figures
./qgan_env/bin/python -m revision.run_figure_suite  # full suite
```

---

## 6. Suggested next-session workflow

1. **Open this file** (`.planning/PAPER-REWRITE-HANDOFF.md`). Read sections 1–3.
2. **End-to-end read of `main (4) copy.tex`** with a critical eye for residual over-claims (use §4.1 prohibition list as the checklist).
3. **End-to-end read of `supp_material.tex`** with the same checklist.
4. **List the gaps**: every paragraph that needs rewriting, with a one-line note on what needs to change.
5. **Apply rewrites** as a sequence of small, atomic `Edit` operations — each one keyed to a specific anchor sentence — so a diff review is easy.
6. **Insert the three new figures** per §2.3 with captions sourced from the verified numbers in §3.
7. **Run the provenance gate** after every batch of edits: `./qgan_env/bin/python revision/verify_number_provenance.py`. Any new literal must resolve.
8. **Run the figure renderer** if any new figure or any figure-companion JSON needs to be regenerated.
9. **Commit in atomic units** keyed to the rewrite areas (one commit per major section or per coherent edit batch). Use the conventional commit-message style from the project history (`fix(14): ...`, `docs(14): ...`, `refactor(14): ...`).
10. **After all rewrites land**: tag the next release (`v1.2` or `v1.0-revision.final` — your call), push, update the GitHub release page.
11. **Final compile + Word rebuttal sync**: confirm the .tex compiles cleanly on Overleaf with the new bib.bib uploaded.
12. **Submit via AIChE portal** before 2026-06-17.

---

## 7. Open questions worth resolving early in the rewrite

1. **Title**: does the current title still imply a deployable framework? If so, consider a rescoped title centered on the matched-budget protocol or the bifurcated finding. (User decision.)
2. **Abstract**: should it lead with Finding 1 (parametric-efficiency equivalence) or Finding 2 (LR-DTW + lag-1 ACF dominance)? Reviewer 2 favored hypothesis-first framing — leading with the question + the bifurcated answer would align.
3. **Figure placement**: the three new figures suggested for main §4.1 and supp §A.7 — does that match the visual hierarchy the user wants? Alternative: all three in main if the page budget allows, or DTW + ACF in main / preprocessing in supp.
4. **Outlook framing of the LR-DTW + lag-1 ACF finding**: should the Outlook explicitly identify the conditions under which the LR-DTW dominance would extend (multivariate, longer time series, higher qubit count past the simulable boundary), or keep that scoping in §4.4 Limitations?
5. **Whether to add a Bonus subsection on the discriminative-score uniformity** (the 0.40888 fixed point). It's a methodological finding about Pipeline B specifically, not a generator-quality result, but it's interesting and the rebuttal already discloses it. The Methods or Supp would be natural homes.

---

End of paper-rewrite handoff.
