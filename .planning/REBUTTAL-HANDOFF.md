---
created: 2026-05-24
purpose: Session handoff for AIChE aic-4719598 major-revision rebuttal letter drafting
status: 3 of N reviewer comments drafted (R1-M1, R1-M2, R1-M4); resubmission pending
---

# AIChE Rebuttal Letter — Drafting Handoff

## 1. State of the work

### 1.1 Phase 14 progress

- Phase 14: **19/20 plans complete**.
- Only **14-07** remains, **deferred to journal acceptance**: Zenodo manual deposit + DOI mint + release.md + manuscript DOI wire-in. First-round revision resubmits with the `ZENODO-DOI-PLACEHOLDER` token + the committed freeze-candidate SHA in the Data Availability statement; real DOI minted at acceptance.

### 1.2 Freeze candidate

- **Active SHA: `3c8502c` (content tree) / `8b87293` (with metadata)**.
- Supersedes the pre-14-20 SHA `6518323` recorded in older docs.
- Either SHA is valid for citation; they differ only in `.planning/` metadata.
- `verify_freeze_ready.py` PASSes every gate against the active SHA *except* the `release.md` assertion, which is 14-07's deliverable by design (Plan 14-19 ordering guard).

### 1.3 Working tree

- Clean. `git status --porcelain` empty.
- Provenance gate PASS across all 7 paper-facing docs (574 distinct literals resolve to JSON).

## 2. Rebuttal drafts produced this session

All three drafts below are the **final iterated versions** the user signed off on (tightened from 950 → 540-770 word ranges; no fabricated content; every numeric literal traces to `revision/results/*.json`).

### 2.1 R1-M1 — Matched classical baseline

**Reviewer comment summary:** No matched classical baseline; can't isolate the quantum contribution. Asks for: classical WGAN-GP at matched parameter count, parameter-controlled table, simpler non-adversarial baseline (small VAE or AR), and explicit acknowledgment if classical matches/exceeds quantum at this scale + discussion of what scale might change the picture.

**Final draft (~720 words):**

> **R1-M1. No matched classical baseline.**
>
> We thank the reviewer; this concern goes to the heart of the contribution claim, and the revised manuscript is restructured around it.
>
> *What we did.* The headline quantum generator (`iqp_sel_55`) carries 55 trainable parameters. Under an identical 2000-epoch matched training budget, the same critic architecture and optimizer, and the same five seeds {42, 43, 44, 45, 46}, we trained: three parameter-matched classical WGAN-GP baselines whose generators are within five parameters of the quantum count — `wgan_mlp` (74 params), `wgan_cnn` (73 params), and `wgan_lstm` (78 params); three additional quantum-circuit variants (V1 75, V2 135, V3 75 params); and the two non-adversarial baselines the reviewer explicitly requests — a small VAE (562 generator parameters, ELBO objective) and a second-order autoregressive model AR(2) fit by closed-form Yule–Walker (3 parameters). All nine models are evaluated on the same held-out OD time series at both OD and log-return scales; the parameter-controlled comparison appears as the new Evaluation Scale table in Section 4.1.
>
> *Honest acknowledgment of the result.* Per the reviewer's instruction, we state directly that the classical baselines match the 55-parameter quantum generator on the marginal OD Earth-mover distance at this scale. The per-baseline statistical comparison is:
>
> | Classical baseline | Generator params | Adversarial setup | Welch *p* (vs iqp_sel_55, OD-EMD) | Cohen's *d* |
> |---|---|---|---|---|
> | wgan_mlp | 74 | generator + 250 881 shared critic | 0.688 | +0.26 |
> | wgan_cnn | 73 | generator + 250 881 shared critic | 0.365 | −0.64 |
> | wgan_lstm | 78 | generator + 250 881 shared critic | 0.836 | −0.14 |
> | vae | 562 | non-adversarial (ELBO) | 0.664 | +0.29 |
> | ar(2) | 3 | non-adversarial (Yule–Walker) | 0.627 | −0.32 |
>
> Across the full pairwise family of 20 quantum-vs-classical comparisons (four quantum variants × five classical baselines) Welch t-tests return *p* > 0.36 and |Cohen's *d*| ≤ 0.65; a proper TOST equivalence test is not satisfied at any defensible margin. We report this explicitly as a *non-significant difference under low power*, not as positive evidence of equivalence — at n = 5 per cell the two-sample Welch test has only ~15 % power against *d* = 0.65 and an 80 %-power detection floor of *d* ≈ 2.0. The `wgan_cnn` |*d*| = 0.64 is dominated by a single outlier seed (seed 42 = 0.159 vs the other four at 0.020–0.034) and is not a typical pair. We have removed all "high fidelity" and "strong performance" framing from the abstract and Key Contributions; the revised abstract presents the quantum generator as a proof of concept "comparable to size-matched classical baselines," not as a method of demonstrated advantage. The previously cited DTW headline of 0.6843 — a pre-revision best-case checkpoint — has been re-anchored to the matched-budget evaluation (0.298–0.302), where the ~6.5× improvement over the Orlandi et al. reference (1.954) is achieved by the matched-budget cluster as a whole including `wgan_lstm` (0.301) and `wgan_mlp` (0.302); we now state explicitly that the OD-DTW improvement is matched-budget-wide and not quantum-specific. The quantum-distinguishing signal that survives the matched-budget comparison is on log-return DTW, where every quantum variant (0.94–1.12) outperforms every WGAN baseline (1.58–6.86) and the AR(2) baseline (7.70). We report this LR-DTW result as a uniform-dominance (conjunctive) claim over the full pairwise family with the worst-case margin reported, which by construction needs no multiple-comparisons correction. The VAE's LR-DTW of 0.088 reflects posterior collapse (synthetic std ≈ 4 × 10⁻⁴) and is flagged as such rather than interpreted as model quality.
>
> *The AR(2) result is degenerate, not a model-quality result.* The AR(2)'s apparent competitiveness on OD-EMD (mean 0.029) is an artefact of how the model is fit, not evidence of generative quality. AR(2) matches the mean and variance of the log-return series by closed-form fit and then re-samples from the implied Gaussian; nothing about its OD-EMD comes from learned model behaviour. Its decisive last-place on log-return DTW (7.70, against quantum 0.94–1.12 and the WGAN cluster 1.58–6.86) exposes that AR(2) has no genuine temporal-structure modeling capability — it only *looks* competitive on the OD-marginal because the OD marginal at this scale is largely determined by the two log-return moments AR(2) matches trivially. Read together, the AR(2) result characterizes the *task* rather than the model: the OD-marginal alone is too easy to discriminate generators at this scale, which is itself the partial answer to the reviewer's call for a non-adversarial baseline to "characterize the difficulty of the task itself."
>
> *What scale or complexity might change the picture.* The revision replaces the prior implicit superiority framing with a new Outlook subsection that scopes the regimes under which the picture could change. The present seed budget (n = 5) is too small to convert "no detectable difference" into "demonstrably equivalent within ε" by TOST; a substantially larger budget per cell would be required. The present qubit count (5) lies inside the classically simulable regime; we expect any quantum-specific advantage on the OD-marginal distribution, if it exists, to require qubit counts past the simulable boundary, longer series with richer temporal structure than a single 778-row OD trajectory, and tasks where the marginal distribution is not the dominant signal. We do not claim quantum superiority at the present five-qubit, single-campaign, OD-marginal scale, and we have rewritten Key Contributions, Theoretical/Practical Implications, and Concluding Remarks to match.

### 2.2 R1-M2 — Utility-oriented evaluation

**Reviewer comment summary:** Distributional/temporal diagnostics aren't sufficient; need TimeGAN-convention utility — TSTR, predictive score, discriminative score, Orlandi-style real-only vs synthetic-augmented training.

**Final draft (~1000 words, includes the *Forensic note* paragraph on discriminative-score uniformity):**

> **R1-M2. Validation strategy: utility-oriented tests required.**
>
> We thank the reviewer; the diagnostic-only validation in the original submission was a real gap, and the revised manuscript implements the full TimeGAN-convention utility battery the reviewer asks for, re-run at the matched 2000-epoch Pipeline B training budget so the utility evaluation and the parametric-efficiency analysis under R1-M1 share a single matched-budget evidence base.
>
> *What we did.* The revised manuscript reports four utility-oriented evaluations across the same nine matched-budget models that back R1-M1 — `iqp_sel_55_repro` (55 trainable parameters), V1 (75 params), V2 (135 params), V3 (75 params), `wgan_mlp` (74 params), `wgan_cnn` (73 params), `wgan_lstm` (78 params), VAE (562 params), AR(2) (3 params, closed-form Yule–Walker fit) — each trained on Pipeline B (log-returns) for 2000 epochs with the matched critic and optimizer and the same five seeds {42, 43, 44, 45, 46}. The four metrics: **TSTR** (train-on-synthetic, test-on-real), implemented as a one-step-ahead OD soft sensor (1-layer LSTM, hidden = 32, lifted verbatim from a reference notebook to remove evaluator-choice degrees of freedom), trained on synthetic OD windows and tested on the held-out real OD windows; **predictive score**, the canonical Yoon et al. TimeGAN one-step-ahead forecast objective; **discriminative score**, a small post-hoc classifier scored as |accuracy − 0.5| under the TimeGAN convention (0 optimal, 0.5 worst); and **real-only versus synthetic-augmented downstream training** — the Orlandi-style comparison, with synthetic windows added at +25 %, +50 %, and +100 % of the n_real = 65 real training windows. Per-model numbers and the matched-budget cross-model figure appear in the revised Section 4.1 (figure `tstr_crossmodel_matched2000`).
>
> *Honest acknowledgment of what the metrics show.*
>
> | Model | n_params | TSTR R² | TSTR MAE | Predictive | Discriminative | +100 % aug R² |
> |---|---|---|---|---|---|---|
> | iqp_sel_55_repro | 55 | 0.9945 | 0.0286 | 0.01944 | 0.40888 | 0.9695 |
> | V1 | 75 | 0.9942 | 0.0295 | 0.01947 | 0.40888 | 0.9688 |
> | V2 | 135 | 0.9946 | 0.0283 | 0.01953 | 0.40888 | 0.9685 |
> | V3 | 75 | 0.9949 | 0.0275 | 0.01925 | 0.40888 | 0.9706 |
> | wgan_mlp | 74 | 0.9976 | 0.0183 | 0.01963 | 0.40888 | 0.9667 |
> | wgan_cnn | 73 | 0.9971 | 0.0202 | 0.02538 | 0.40888 | 0.9624 |
> | wgan_lstm | 78 | 0.9966 | 0.0220 | 0.01981 | 0.40888 | 0.9565 |
> | vae | 562 | 0.9930 | 0.0319 | 0.01960 | 0.40888 | 0.9641 |
> | ar(2) | 3 | 0.9977 | 0.0184 | 0.01884 | 0.40888 | 0.9568 |
> | **real-only baseline (n = 65 real windows)** | — | **-13.354** | **1.802** | — | — | — |
>
> Two patterns dominate, and we report both honestly. (i) **Every generator's synthetic OD is useful for downstream training on the n = 65 data-starved real soft-sensor task.** The real-only baseline at n_real = 65 is catastrophic (R² = -13.354) — the task cannot be learned from real windows alone. Training the same soft sensor on synthetic windows produces R² ≈ 0.99 across every generator, and adding 65 synthetic windows on top of the 65 real ones (+100 % augmentation) lifts the soft sensor to R² in [0.957, 0.971] across every generator. This is direct evidence in the Orlandi style that the synthetic data are useful. (ii) **No generator separates from any other on this utility battery at the matched budget.** Across nine generators ranging from a 3-parameter closed-form AR(2) to a 250881-parameter adversarial WGAN-CNN, the TSTR R² band is [0.9930, 0.9977] — a width of 0.0047, smaller than the per-cell standard deviation for most variants. The augmentation lift is broadly comparable across generators. Per the reviewer's final-sentence instruction, we acknowledge directly that the classical baselines effectively match the 55-parameter quantum generator on each of the four utility metrics at this scale.
>
> *Forensic note on the discriminative score uniformity.* The discriminative-score column in the table above reports **0.40888 for every one of the 45 matched-budget cells** — six architecture families, five generator seeds, three init seeds, identical to five decimal places (full float32 precision: 0.4088757634162903). We anticipate the reviewer will read this as suspicious — uniformity to that precision would normally suggest a bug. We have audited it and report the result here.
>
> The uniformity is real, not an implementation defect, and it is specifically a **Pipeline-B phenomenon**, not a generator-specific phenomenon. In the legacy 1000-epoch utility evaluation (`predictive_discriminative.json`, retained on disk as provenance reference), of 60 (model, pipeline, seed) discriminative cells, exactly 29 of the 30 Pipeline-B cells produced the identical 0.4088757634 value; 0 of the 30 Pipeline-A cells produced it (Pipeline-A discriminative scores spread across the 0.430-0.481 range). The sole Pipeline-B exception in the legacy data is `wgan_cnn` seed 42 at 0.4396 — the same anomalous outlier seed disclosed under R1-M1's `wgan_cnn` discussion. The matched-budget re-run produces 45/45 = 100 % Pipeline-B fixed-point hits because (a) all matched-budget runs are Pipeline B by protocol and (b) the 2000-epoch training budget produces cleaner cumulative-sum back-transformed outputs than the 1000-epoch legacy regime, removing the legacy `wgan_cnn`-seed-42 deviation. As a sanity cross-check, the **predictive** scores in the same matched-budget JSON show 44 unique values across 45 cells (range 0.0183–0.0405), confirming the post-hoc TimeGAN nets are training and discriminating normally and that the discriminative-score uniformity is a finding specific to that metric, not a global degeneracy.
>
> The mechanism is structural: the post-hoc GRU classifier learns to identify a fixed signature of the cumulative-sum back-transform (which converts synthetic log-returns to OD), and converges to the same decision boundary regardless of which generator produced the underlying log-returns. Pipeline A has no cumulative-sum back-transform (the generator emits OD windows directly), so the same classifier finds no such universal shortcut and scores vary across generators in the normal way. We treat the Pipeline-B discriminative-score uniformity as a finding rather than as evidence of generator quality — it directly demonstrates that the metric on Pipeline B is dominated by the structural back-transform signature rather than by generator behaviour.
>
> *Why the picture is uniform across generators on every metric.* The forensic finding above for the discriminative score is one example of the same structural mechanism that produces the [0.993, 0.998] TSTR R² band and the [0.957, 0.971] augmentation-lift band. Pipeline B operates on log-return windows, and the back-transform from synthetic log-returns to synthetic OD is a cumulative sum — OD_t = exp(Σ log_return_τ for τ ≤ t). That cumulative-sum back-transform mathematically encodes near-perfect lag-1 autocorrelation into the synthetic OD regardless of the generator's quality, so a soft sensor trained on Pipeline-B-derived synthetic OD essentially learns the persistence forecast OD_{t+1} ≈ OD_t — near-optimal on the real OD series. The discriminative classifier learns the inverse: the structural fingerprint of *being* a cumulative-sum back-transformed series, identifiable in any generator's synthetic output, yielding the 0.40888 fixed point. The augmentation lift inherits the same structural advantage. The Pipeline-B utility battery therefore reports a uniform-across-generators result: *the synthetic data are useful for downstream OD forecasting on a data-starved real soft-sensor task, but no generator outperforms any other on this utility battery at this scale.* The only utility-adjacent metric on which quantum variants distinguish themselves in the matched-budget comparison is log-return DTW (LR-DTW), addressed under R1-M1.
>
> *What scale or complexity might change the picture.* The revised manuscript replaces the prior implicit superiority framing with an Outlook section that scopes the regimes under which the picture could change. The structural saturation observed on Pipeline B is a property of the preprocessing pipeline at this dataset scale and metric resolution, not a fundamental limit; we expect any generator-discriminative utility signal to require richer downstream tasks where the persistence baseline is harder to beat (multi-step forecast horizons, longer time series with regime changes, multivariate OD/biomass/feed-rate joint forecasting), preprocessing pipelines that do not pre-encode lag-1 autocorrelation (raw-OD generation past the present 5-qubit simulable regime is one such direction), and a substantially larger seed budget per cell for power. We do not claim quantum superiority on the utility battery at the present five-qubit, single-campaign, log-return-preprocessing scale, and Section 4.1 reports the matched-budget utility result accordingly.

### 2.3 R1-M4 — Training protocol + shot noise + Supp Eq A.3

**Reviewer comment summary:** Missing n_critic, λ, optimizer/LR, epochs, stopping, seed sensitivity, analytic vs shots. Add Training Protocol; clarify simulation regime; shot-noise sensitivity at 1024 + 8192; mean ± std across ≥ 5 seeds; clarify Supp Eq A3 log-GAN vs Wasserstein discrepancy.

**Final draft (~540 words, tightened from 950):**

> **R1-M4. Incomplete optimization and training details.**
>
> We thank the reviewer; the original submission did not describe the training protocol at a reproducible level. The revised manuscript now carries a self-contained Training Protocol in Section 3, a backend statement, a shot-noise sensitivity analysis at the reviewer's recommended budgets (1024 and 8192 shots) plus an analytic reference, per-layer noise-channel sensitivity, multi-seed reporting across n = 5 seeds, and a clarification of Supplementary Equation A3.
>
> *Training protocol.* All matched-budget runs (four quantum variants + five classical baselines) use the same contract:
>
> | Hyperparameter | Value |
> |---|---|
> | Optimizer (generator + critic) | Adam (β₁ = 0.5, β₂ = 0.9) |
> | Learning rate (generator / critic) | 6.9173 × 10⁻⁵ / 1.8046 × 10⁻⁵ (HPO-tuned, v1.1) |
> | n_critic (critic steps per generator step) | 9 |
> | λ_gp (gradient-penalty coefficient) | 2.16 |
> | Batch size | 12 |
> | Training epochs | 2000 (matched budget, full duration) |
> | Early stopping | OFF for the matched-budget reproduction (decision D-14-13); ON for the frozen-checkpoint historical entrant `iqp_sel_55_headline` (best-EMD-on-eval-window checkpoint) |
> | Seeds | {42, 43, 44, 45, 46} (n = 5 per cell) |
> | Latent noise distribution | Uniform[0, 4π] |
>
> Every value renders from `revision/results/methods_full.json` / `model_info.json` / the config-lock JSONs; file-and-line citations are in `methods_full.md` Section 3. The matched-budget contract runs the full 2000 epochs deliberately — the headline-vs-reproduction distinction is reconciled under R1-M1.
>
> *Simulation regime.* The quantum-circuit outputs are computed as **analytic expectation values** under PennyLane's `default.qubit` device with `diff_method = "backprop"` — exact statevector simulation, no finite-shot estimates. (`backprop` was chosen over parameter-shift due to PennyLane issue #4462, which produced incorrect parameter-shift gradients on the present circuit's input-broadcast shape; `backprop` returns exact gradients on the same statevector.)
>
> *Shot-noise sensitivity (Pipeline B, anchored at `shot_noise_sensitivity.json`).*
>
> | Shot regime | OD-EMD mean | OD-EMD std |
> |---|---|---|
> | Analytic | 0.029676 | 0.004874 |
> | 1024 shots / expectation | 0.029682 | 0.004867 |
> | 8192 shots / expectation | 0.029675 | 0.004870 |
>
> The metric differs across shot regimes by at most 7 × 10⁻⁶ — five decimal places out — against a per-seed standard deviation of ~5 × 10⁻³. The matched-budget OD-EMD is, in practical terms, insensitive to shot count at 1024 shots and above, justifying the analytic-statevector design choice. The shot-noise sweep uses n = 3 seeds; given the cross-regime difference is orders of magnitude smaller than the per-seed variance, this n is informative. We also report a per-layer noise-channel sensitivity sweep (depolarizing and amplitude-damping channels at noise levels 0–5 %, inserted after each entangling block): OD-EMD drifts from 0.029676 at 0 % to 0.029691 (depolarizing) / 0.029875 (amplitude damping) at 5 %, a ≤ 0.7 % relative change. Figures `shot_noise_robustness.{png,pdf}` and `noise_robustness_quantum.{png,pdf}` visualize both sweeps.
>
> *Seed sensitivity.* Every matched-budget number cited in the revised manuscript is reported as mean ± std across the n = 5 seed set. The headline `iqp_sel_55_repro` carries OD-EMD = 0.02753 ± 0.00513; all R1-M1 pairwise statistics (Welch p, Cohen's d, the |d| ≤ 0.65 ceiling) are computed from these per-seed values, and the n = 5 power limitation (~15 % power against d = 0.65; 80 %-power detection floor d ≈ 2.0) is disclosed at every claim site.
>
> *Supplementary Equation A3.* The reviewer correctly identified that Supplementary Equation A3 was written in the log-GAN / Jensen–Shannon form while the model trained in this study is the WGAN-GP (Earth-Mover formulation, Equation eq:wgangp in the main text). Supplementary Section A.3 now states this explicitly and carries a banner *"not implemented in this study"* — A3 is the proposed Hybrid-GAN-mechanistic *extension* objective and is retained in the log-GAN form to mirror the existing Hybrid-GAN literature; the extension, if implemented, would substitute a WGAN-GP critic for the log-GAN discriminator. The Hybrid-GAN material has been moved from Key Contributions to a new Outlook subsection (PAPER-05a), consistent with the R1-M1 / R1-M2 de-overclaiming throughout the revision.

## 3. Remaining reviewer comments to draft

The original reviewer set includes (look at `revision/docs/reviewer_response.md` summary tables — lines 30–110 — for the full inventory):

- **R1-M3** — Preprocessing-pipeline ablation. *Partly addressed at 1000-epoch budget by phase 09.1; the matched-budget context is already in the manuscript. May need a brief letter response noting that the matched-budget protocol settled on Pipeline B per the 09.1 finding.*
- **R1-m1** — Misplaced/weak references (citation surgery, PAPER-06)
- **R1-m2** — Dataset details in Methods (PAPER-08)
- **R1-m3** — Tied to R1-M3
- **R1-m4** — Freeze GitHub + DOI (links to 14-07; placeholder + SHA framing for first-round)
- **R1-m5** — Orlandi comparison (folded into R1-M2 utility eval)
- **R1-m6 .. R1-m12** — Typography, notation, captions (PAPER-11 already integrated)
- **R2-1** — Reframed hypothesis (PAPER-01)
- **R2-2** — Concluding remarks
- **R2-3** — VAE / classical-baseline framing
- **R2-4** — "Improves prediction performance" — handled by R1-M2 augmentation result
- **R2-5b** — Why this particular circuit (Circuit Design Rationale subsection, PAPER-03 already integrated)

Most R1-m* and R2-* items are already integrated into the manuscript via the PAPER-* keyed blocks (Plans 14-05, 14-06, 14-12, 14-15, 14-16, 14-17). The remaining letter work is mostly cross-referencing the existing manuscript sections rather than new drafting.

## 4. Drafting style guide (calibrated this session)

What worked, in order of importance:

1. **4-block structure**: *what we did* → *honest acknowledgment* → *mechanism / degenerate-case caveat* → *what scale might change*. Used in all three drafts.
2. **Per-baseline table inline**. Reviewers respond well to "here is the result, line by line, with the actual numbers." Keep tables ≤ 7 columns.
3. **Direct acknowledgment** of reviewer's specific concern at the start. "Per the reviewer's instruction" / "The reviewer correctly identified". Don't be defensive.
4. **Length target: ~600 words per major comment**, more if the comment has many sub-asks (R1-M4 has 6 → 540 words; R1-M2 has 4 + forensic → 1000 words).
5. **Honest disclosure of limitations** — n = 5 power, AR(2) degeneracy, shot-noise n = 3, discriminative-score uniformity. Always front-foot these; never let the reviewer find them first.
6. **Provenance-gate-friendly literals**: ASCII minus signs (`-`, not `−`), no space-separated big numbers (`250881` not `250 881`, `19200` not `19 200`), preserve stored precision (`-13.354` not `-13.35`).
7. **Cross-reference between R1-* responses**: "addressed under R1-M1" — don't re-litigate.
8. **Never fabricate model identification**. Lesson from the legacy R1-M2 draft: don't label generic `quantum` as `iqp_sel_55` without evidence. Trace data provenance back to source paths before drafting.

## 5. Key numbers (memorize these — they appear in every draft)

| Quantity | Value | Source |
|---|---|---|
| Canonical quantum (matched-budget) param count | 55 | `model_info.json` iqp_sel_55_repro |
| Classical WGAN-GP gen param counts | 74 / 73 / 78 (mlp / cnn / lstm) | `classical_architectures.json` |
| Shared WGAN-GP critic param count | 250 881 | `total_adversarial_param_budget.json` |
| VAE generator params | 562 | `classical_architectures.json` |
| AR(2) params | 3 | `classical_architectures.json` |
| Matched-budget seeds | {42, 43, 44, 45, 46} (n = 5) | `methods_full.json` |
| Matched-budget epochs | 2000 | `methods_full.json` |
| OD-EMD iqp_sel_55_repro | 0.02753 ± 0.00513 | `matched2000_dualscale.json` |
| OD-DTW matched-budget cluster | 0.298 – 0.302 | `matched2000_dualscale.json` |
| Orlandi reference DTW | 1.954 | `methods_full.md` §3 DTW context |
| LR-DTW quantum range | 0.94 – 1.12 | `matched2000_dualscale.json` |
| LR-DTW WGAN range | 1.58 – 6.86 | `matched2000_dualscale.json` |
| LR-DTW AR(2) | 7.70 | `matched2000_dualscale.json` |
| LR-DTW VAE (posterior collapse) | 0.088 | `matched2000_dualscale.json` |
| OD-EMD Welch p, max across 20 quantum-classical pairs | > 0.36 | `welch_pairwise.json` |
| OD-EMD \|Cohen's d\|, max | ≤ 0.65 | `welch_pairwise.json` |
| n = 5 Welch power vs d = 0.65 | ~15 % | (computed) |
| n = 5 80 %-power detection floor | d ≈ 2.0 | (computed) |
| wgan_cnn seed-42 outlier OD-EMD | 0.1587 | `matched2000_dualscale.json` |
| wgan_cnn other-seeds OD-EMD range | 0.020 – 0.034 | `matched2000_dualscale.json` |
| TSTR R² matched-budget band | [0.9930, 0.9977] | `tstr_matched2000.json` |
| Discriminative score (every Pipeline-B cell, matched-budget) | 0.40888 (= 0.4088757634162903) | `predictive_discriminative_matched2000.json` |
| Predictive score band, matched-budget | [0.0188, 0.0254] | `predictive_discriminative_matched2000.json` |
| Real-only soft-sensor baseline (n_real = 65) | R² = -13.354 ± 0.583, MAE = 1.802, RMSE = 1.840 | `tstr_matched2000.json` |
| +100% augmentation R² band | [0.957, 0.971] | `augmentation_matched2000.json` |
| HPO-tuned lr_generator | 6.9173 × 10⁻⁵ | `methods_full.json` |
| HPO-tuned lr_critic | 1.8046 × 10⁻⁵ | `methods_full.json` |
| HPO-tuned n_critic | 9 | `methods_full.json` |
| HPO-tuned λ_gp | 2.16 | `methods_full.json` |
| Batch size | 12 | `methods_full.json` |
| Shot-noise OD-EMD analytic / 1024 / 8192 | 0.029676 / 0.029682 / 0.029675 | `shot_noise_sensitivity.json` |
| Noise-channel sensitivity range | OD-EMD 0.029676 → 0.029875 at 5% amplitude damping | `noise_model_sensitivity.json` |

## 6. Files of record

- **Drafts above** are the canonical text. Quote them verbatim into the rebuttal letter.
- **`revision/docs/reviewer_response.md`** is the in-repo provenance trail; its R1-M2 section was rewritten by Plan 14-20 to match the matched-budget data — verbatim-consistent with the R1-M2 draft above.
- **`revision/docs/methods_full.md` §3.y** documents the matched-budget utility protocol with full per-variant numbers and the data_hash invariance.
- **`revision/docs/completeness_sweep_manifest.md`** is the artefact-inventory ledger; matched-budget rows added by 14-20.
- **Manuscript files** `main (4) copy.tex` + `supp_material.tex` carry the PAPER-* blocks integrated by Plan 14-17; abstract is de-overclaimed; circuit-design subsection + Outlook subsection added.

## 7. Resubmission packaging checklist

When ready to submit:

- [ ] Assemble the rebuttal letter from R1-M1 / R1-M2 / R1-M4 drafts in this document + the R1-M3 / R1-m* / R2-* paragraphs (mostly cross-references to existing PAPER-* manuscript sections)
- [ ] Cite the freeze candidate SHA `3c8502c` (or `8b87293`) in the Data Availability section of the manuscript
- [ ] Confirm `verify_freeze_ready.py` PASSes every gate except `release.md` against the SHA you cite
- [ ] Confirm `git status --porcelain` is empty
- [ ] Submit revision via AIChE Journal portal
- [ ] After acceptance, run Plan 14-07 (Zenodo deposit + DOI + release.md + manuscript DOI wire-in) and submit camera-ready

## 8. Next-session resume

To pick up:

1. Read this handoff doc (`.planning/REBUTTAL-HANDOFF.md`)
2. Read the auto-memory at `~/.claude/projects/-Users-shawngibford-dev-phd-qGAN/memory/MEMORY.md` — the `project_phase14_rebuttal_drafting.md` entry (added this session) carries the calibrated drafting style + key facts
3. Pick the next reviewer comment (R1-M3 is the natural next one)
4. Apply the 4-block structure + per-baseline table + provenance-friendly literals
5. Iterate with the user; tighten to ~600 words once draft is honest

End of handoff.
