# PR-4 — Scientific Rigor & Statistical Honesty Audit

## Verdict
**CONCERNS — partially blocking.** The post-14-21 narrative *can* survive hostile peer review with the present technical artefacts, but only after fixing three issues that any thorough reviewer will spot inside 30 minutes:

- **BLOCKING-1**: Supplement Welch tables (`tbl:welch_od_emd` line 380-421 and the caption of `tbl:welch_lr_emd` line 423-470) are STALE pre-14-21 numbers. The OD-EMD table caption literally states "no quantum-vs-classical pair is statistically resolved on the OD marginal" — directly contradicting the main paper's headline OD-EMD Welch p = 0.019. This is the highest-yield "gotcha" for a hostile reviewer.
- **BLOCKING-2**: Main paper abstract (line 49) says "three of the four headline metrics" then mentions OD-EMD separately, while the Contributions list (line 103, 105, 726, 728, 769) consistently says "all four matched-budget metrics" / "four-metric quantum-vs-WGAN dominance". Internal inconsistency between abstract framing and rest-of-paper framing.
- **BLOCKING-3**: `reviewer_response.md` carries two mutually-contradicting paragraphs: lines 431-437 ("no statistically detectable OD-scale EMD difference… reflects an underpowered non-significant difference test") vs lines 487-491 ("Cluster-floor Welch p = 0.019… WGAN cluster mean ≈0.331 sits substantially above the quantum cluster mean ≈0.029"); and the absolute statement at line 215 "[LR-DTW] is the sole quantum-distinguishing result we claim". A reviewer who reads §R1-M1 in the rebuttal then reads §4.1 in the paper will catch the contradiction immediately.

The R3 amendment of the strong-claim threshold (relaxed hard-abort → soft-fail) is **not disclosed in either main or supp** and is the second-highest-yield reviewer gotcha after the stale tables. The §A.7 "Data Transformation Details" disclosure is technically transparent but rhetorically defensive — it discusses *the bug* without explicitly mapping the corrections to *the v1.2.4 narrative they invert*.

DEFERRABLE: multiple-testing correction; explicit "convenient inversion" paragraph; "no quantum seed overlaps" should be re-counted on the new 4×3×5=60 grid (the SUMMARY says 25/25, but tbl:per_seed_dtw_dominance keeps the 4q×3c×5s wording).

6-day deadline implies: do BLOCKING-1/2/3 + R3 threshold disclosure now; defer multiplicity-correction additions to a footnote.

---

## R1 — §A.7 disclosure paragraph adequacy

### Current text (verbatim, supp_material.tex lines 735-786)

> **WGAN sample-space convention and inverse-pipeline scaling.**
> WGAN-trained generators in this study (quantum IQP:SEL plus V1/V2/V3, classical wgan_mlp/cnn/lstm) are trained and saved with their $[-1, +1]$ generator outputs multiplied by $0.1$ at the sample-export site. This convention is inherited from an earlier reference notebook (`archive/qgan_pennylane_SEL.py`; the relevant scaling block carries the explicit code comment "Real data is roughly in $[-0.08, 0.09]$, so scale by $\sim 0.1$"), where it served to magnitude-match the quantum generator's $[-1, +1]$ output to unstandardized log-returns. The same $\times 0.1$ is retained at the WGAN training/sample-export sites in the present pipeline so that the on-disk `samples.npy` bundles under `results/matched2000/runs/` remain byte-stable across runs and consistent with every saved checkpoint; modifying the training-side scaling would invalidate every checkpoint and break reproducibility against the released artifact set. The on-disk WGAN-trained samples are therefore stored in $[-0.1, +0.1]$ space.
>
> The downstream inverse formula $r_\text{norm} = ((s+1)/2)(r_\max-r_\min)+r_\min$ assumes $s \in [-1, +1]$, so every paper-cited consumer applies a $\times 10$ correction at the `samples.npy` load boundary via a shared helper module `_wgan_unscale.py` before invoking the inverse formula. The correction is inference-only and is gated by `model_kind in _WGAN_KINDS`…
>
> *VAE and AR(2) are deliberately excluded from this set.* Per the "Pitfall 3" design note at `run_baselines.py:28`, the VAE and AR(2) generators are trained without the $\times 0.1$ post-scaling…
>
> A residual generator mean-bias in standardized space persists for the quantum cluster (quantum mean $\approx 0$ vs. real Pipeline-B mean $\approx -0.03$) and contributes a $\sim 2\times$ OD drift over 777 contiguous integration steps in the long-horizon overlays of §A.10; this is a training-side artifact not addressable at the inference boundary and is disclosed for transparency.

### Hostile-reviewer worst-case reading

A hostile reviewer reads this paragraph and concludes: *"The authors discovered, in the revision cycle, that their classical WGAN baselines had been outputting samples 10× too small for the entire evaluation pipeline. They fixed it on the inference side without retraining. Conveniently, the fix makes the WGAN baselines look 10× worse on every metric, exactly the direction needed to support their headline finding. The paragraph doesn't quantify how much the v1.2.4 conclusions change, doesn't acknowledge that the OD-EMD finding has flipped sign (from 'no detectable difference' to 'p=0.019'), and treats the residual mean-drift as a footnote despite the fact that it also affects the quantum cluster."*

The phrasing is the giveaway: "preserve checkpoint validity" and "break reproducibility" sound like methodological prudence, but a hostile reviewer will hear "we noticed the bug, didn't retrain (which would have given a cleaner answer), and applied a multiplicative correction at evaluation time that happens to be a 10× lift on the comparators."

### Specific disclosure improvements

Add the following two sentences at the end of the WGAN-convention paragraph (line 752):

> *Relative-fidelity impact.* The pre-correction WGAN samples were attenuated by a factor of 10 in the rescaled $[-1, +1]$ window space, which translated to a $\sim 10\times$ under-statement of every WGAN-vs-real distance metric. Under the corrected inference pipeline, the WGAN cluster mean shifts up on every distance metric reported in main §4.1 (LR-EMD, OD-EMD, LR-DTW, OD-DTW); see Table A.X for the explicit pre/post comparison.

Then add a new "Table A.X" — a 4×2 grid (Q cluster mean / WGAN cluster mean) × (pre-fix / post-fix) for each of the 4 matched-budget metrics, with a column flagging which v1.2.4 conclusion is preserved vs inverted. The 14-21-SUMMARY Bifurcated-finding-table is already nearly this artefact; lift it into the supplement.

Add an explicit sentence at the start of the disclosure paragraph (line 735):

> *Note for reviewers comparing to the v1.2.4 submission.* The first-round submission (origin tag v1.2.4) reported a bifurcated reading on which classical WGAN baselines outperformed the quantum cluster on LR-EMD and matched parametric-equivalence on OD-EMD. Both readings were artefacts of the under-disclosed sample-space convention described below; under the corrected inference pipeline the quantum cluster dominates the WGAN cluster on all four matched-budget metrics. We retain the v1.2.4 release tag and the pre-fix data artefacts for historical traceability and document the directional shifts cell-by-cell in Table A.X.

This preempts the "convenient finding-inversion" line of attack.

---

## R2 — Statistical claim verification

### R2.1 Welch test sample size

**Manuscript claim** (main p.0 line 397): "the WGAN cluster mean ($\approx 0.3312$) at a Welch cluster-floor $p = 0.019$ over the 12 quantum-vs-WGAN pairs, with the maximum-magnitude Cohen's $d$ exceeding $|d|>3$ in the wgan_cnn pairings"

**Actual sample size**: per-pair n=5 vs n=5 Welch t-test. The "cluster-floor p" is the minimum p across the 12 pairs (4 quantum × 3 WGAN), not a single test on cluster means. The driving pair for the 0.019 floor is `iqp_sel_55_repro vs wgan_lstm` (mean_q≈0.028, mean_c≈0.118, n=5+5, welch_p=0.0188, |d|=2.34). The wgan_cnn pairings give p ≈ 0.31 (NOT significant) — high |d| but huge std (1.47) from the seed-42 outlier.

**Verdict**: Sample-size description is technically correct (n=5 per group, per-pair Welch) but the "cluster-floor" verbiage is non-standard and easy to misread as a cluster-mean-vs-cluster-mean test. A reviewer who doesn't dig into the supp will read "p=0.019 over 12 pairs" and assume it's a multiplicity-corrected aggregate. **Add the explicit definition once** in §3.x or §4.1: *"By 'Welch cluster-floor p = X' we mean the minimum two-sided Welch t-test p-value over the M quantum-vs-WGAN pairs, n=5 per group. The floor is reported uncorrected for multiplicity (rationale: §4.1 footnote Y)."*

### R2.2 Multi-test correction

Main text reports cluster-floor p over 12 pairs for OD-EMD (line 625) and over 12 pairs for LR-EMD (line 515) and "p as low as 0.002" for OD-DTW. None of these are Bonferroni / FDR / Holm-corrected.

For OD-EMD (BLOCKING-1 territory): 0.019 × 12 = 0.228 (Bonferroni), 0.019 with Benjamini-Hochberg at rank-1 of 12 ≈ 0.019/1 × 12 ≈ ditto. Bonferroni would push to "not significant at 0.05" but the Holm or BH-corrected supports a *weaker* claim: "at least one pair is significant at q=0.10 after FDR correction." For LR-EMD the floor p ≈ 0.0002 survives Bonferroni × 12 (=0.0024) trivially.

The reviewer_response.md lines 413-425 has a defensible explanation:

> "The OD-EMD non-significance result is correspondingly reported WITHOUT a positive-equivalence inference, so multiplicity does not inflate a false claim there either"

But this paragraph is STALE (it still calls OD-EMD a non-significance result). The defensible *post-14-21* position is: "We report the *floor* p over the pairwise family as a worst-case lower bound on the strongest pairwise quantum-WGAN gap, not as a cluster-level test. This makes multiplicity correction inappropriate because we are not multiplicity-claiming significance across the family; we are reporting the floor as a property of the family. A multiplicity-adjusted multi-comparison statement on 'at least one significant pair' would give Bonferroni-corrected p = X for OD-EMD; we report both for transparency."

**Recommended addition** (supp §A.7 or §3.x Methods): *"All cluster-floor p-values are reported uncorrected for multiplicity. Bonferroni-corrected counterparts (12 pairs × reported p) are: OD-EMD 0.019 → 0.228; LR-EMD 0.0002 → 0.0024; OD-DTW 0.002 → 0.024. The qualitative conclusion (LR-EMD and OD-DTW remain significant under multiplicity adjustment; OD-EMD does not) is preserved at the q=0.05 FDR level (BH-adjusted)."*

### R2.3 Cohen's d for OD-DTW

OD-DTW |d| values are NOT reported in the main paper. The text says "p as low as 0.002" but does not give the d, and there is no welch_pairwise_dtw.json — the Welch JSON is EMD-only. The 0.002 source traces to `ansatz_comparison.json` line 178 (value 0.00245). A hostile reviewer will note: *"You report a Welch p without the corresponding effect size, and the underlying JSON name (ansatz_comparison) doesn't even mention DTW. Provenance is opaque."*

**Recommended addition**: Report |d| for OD-DTW alongside the p (or build a `welch_pairwise_dtw.json` to mirror the EMD version) — and add it as a row in a unified pairwise-Welch summary table. With Q OD-DTW 0.33-0.41 vs WGAN_lstm 0.60 (n=5, std ≈ 0.02 / 0.03 in the data), |d| should be of order 5-10 — very large. Reporting it strengthens the claim.

### R2.4 n=5 power language vs current dominance reading

Main text line 805 retains: *"the $n=5$ seed budget is sufficient for the per-seed-dominance reading of LR-DTW… and for the cluster-floor Welch tests on OD-EMD ($p = 0.019$) and OD-DTW ($p \approx 0.002$), but it does not support either TOST-grade equivalence inference or per-seed claims on OD-EMD / LR-EMD."*

This is the correct framing post-14-21. The pre-14-21 "underpowered null" framing has been excised from main and supp body text (good). The only remaining "underpowered" mention in supp is line 854, which is the *correct* "underpowered for OD-EMD equivalence testing" — defensible.

**Verdict**: main + supp body OK on power language post-14-21. The defect is that the supp pairwise tables (tbl:welch_od_emd / tbl:welch_lr_emd captions) STILL retain the pre-14-21 power-and-TOST framing (line 387-389: *"no quantum-vs-classical pair is statistically resolved on the OD marginal… at $n=5$ the Welch test has $\approx 15\%$ power against $|d|=0.65$ and a two one-sided test (TOST) for equivalence is not satisfied at any defensible margin"*). This is BLOCKING-1: a hostile reviewer reading the supp will find the table caption saying the *exact opposite* of the main paper's claim.

---

## R3 — H2 inversion transparency

### Pre-14-21 framing (from 14-21-SUMMARY line 102)

> OD-EMD parametric equivalence (H2) | Pre-fix v1.2.4 narrative: Q ≈ WGAN, Welch cluster-floor p=0.37, n=5 underpowered null

### Post-14-21 framing (from main lines 616-642 + 769)

> **OD marginal: quantum cluster dominates the WGAN cluster.** […] The quantum-cluster mean ($\approx 0.0288$) is below the WGAN-cluster mean ($\approx 0.3312$) at a Welch cluster-floor $p = 0.019$ over the 12 quantum-vs-WGAN pairs, with the maximum-magnitude Cohen's $d$ exceeding $|d|>3$ in the wgan_cnn pairings.

### Verdict: **OPAQUE on the inversion**

The main paper presents the cluster-dominance reading as if it had always been the conclusion. There is *zero* acknowledgement that this paragraph replaces a "non-significant difference, parametric-equivalence hedge" narrative from v1.2.4. A reviewer who consulted the published v1.2.4 (which is on origin) will catch the inversion and read it as suspicious.

### Recommended disclosure addition

Add to main §4.1 as a numbered footnote on the "OD marginal: quantum cluster dominates" paragraph header:

> *Reviewers comparing against the first-round submission (origin tag v1.2.4) will note that this paragraph replaces a "parametric-equivalence under low power" reading. The directional shift is an artefact of the sample-space correction described in supp §A.7: under the inference pipeline used in v1.2.4 the WGAN samples were under-scaled by 10×, attenuating every WGAN-vs-real distance and inflating the apparent OD-EMD overlap. The corrected pipeline (this paper) restores the WGAN cluster to its true position and the cluster-floor Welch p shifts from 0.37 to 0.019. The data artefacts and the bug fix are tagged at release.*

---

## R4 — Outlier-seed disclosure (wgan_cnn)

**Verdict**: PARTIAL. The seed-42 wgan_cnn outlier IS disclosed in three places:
- Main line 230-233 (loss-grid caption): "the wgan_cnn seed-42 outlier (single seed where OD-EMD transiently fell to $\sim 10^{-4}$ before recovering)" — *N.B. this is a different artefact from the OD-EMD-final outlier*
- Main line 580-583 (cross_model_emd caption): "the wgan_cnn marker is inflated by a single seed-42 outlier (OD-EMD $=0.1587$, with the other four wgan_cnn seeds between $0.020$ and $0.034$)"
- reviewer_response.md lines 474-482: full outlier-seed disclosure paragraph with the leave-one-out reasoning ("even after excluding wgan_cnn from the cluster comparison, the cluster-floor reading on the two surviving WGAN models holds")

Per-seed tables PRESENT for LR-DTW (supp tbl:per_seed_dtw_dominance line 339-363). Per-seed tables ABSENT for OD-DTW, LR-EMD, OD-EMD — the raw JSON has them, but the supp tables don't expose them. A hostile reviewer will demand them: "show me the 5 wgan_cnn OD-EMD values so I can re-aggregate excluding seed 42."

**Recommended additions**:
1. The seed-by-seed wgan_cnn OD-EMD values ARE in the rebuttal line 581 ("0.020 to 0.034" plus the 0.1587 outlier) but should appear in supp as an auditable table: 9 generators × 5 seeds × OD-EMD, sortable / re-aggregatable.
2. The "leave-wgan_cnn-out" sensitivity in reviewer_response.md line 478-481 should be promoted to the main paper §4.1: *"With wgan_cnn excluded, the surviving WGAN-MLP and WGAN-LSTM still give cluster-floor Welch p ≤ 0.05 against the quantum cluster on OD-EMD; the cluster-dominance claim is robust to the seed-42 outlier."* This is the single highest-impact defensive sentence for the outlier-vulnerability attack.

---

## R5 — Welch threshold relaxation disclosure

**Manuscript discloses?** NO — neither main nor supp mentions that `run_welch_aggregator.py` relaxed its strong-claim acceptance threshold from hard-abort to soft-fail.

**Should it?** YES, and the rationale is straightforward to explain. The code change at `run_welch_aggregator.py` lines 138-156 is well-commented internally:

> Post-x0.1-fix data inverts the H2 parametric-equivalence claim. Thresholds preserved for traceability but no longer gate writes.

The undisclosed code-side methodological choice is the kind of thing a thorough rebuttal review will surface. If it surfaces *after* the second-round decision, it looks bad.

**Recommended disclosure text** (add as supp §A.7 paragraph after the existing residual-mean-drift paragraph, line 786):

> *Acceptance-gate amendment.* The v1.2.4 evaluation pipeline included a strong-claim acceptance gate at `run_welch_aggregator.py` that hard-aborted when the OD-EMD cluster-floor Welch p fell below 0.36 or |Cohen's d| exceeded 0.65 — the H2 parametric-equivalence acceptance thresholds. Under the corrected inference pipeline these thresholds no longer correspond to a defensible claim (cluster-floor p drops to 0.019 with $|d|$ exceeding 2, in the cluster-dominance direction). The gate was converted to a soft-fail diagnostic (logs the violation to stderr and the output JSON, no longer aborts the pipeline), with the historical threshold values preserved in `strong_claim_thresholds.{floor_welch_p_OD, ceiling_abs_cohen_d_OD}` for traceability. The amendment was authorized at the R3 user-decision checkpoint; the rationale is recorded in the source-file comment block at lines 138-156.

---

## R6 — Cross-doc consistency findings

### Stale phrases in reviewer_response.md (BLOCKING-3)

| Line | Verbatim quote | Issue |
|---|---|---|
| 211-216 | "The only utility-adjacent metric on which quantum variants distinguish themselves in the matched-budget comparison is log-return DTW (LR-DTW)… That is the sole quantum-distinguishing result we claim." | DIRECTLY CONTRADICTS main paper claim of 4-of-4 metric dominance. Pre-14-21 framing. |
| 431-437 | "no statistically detectable OD-scale EMD difference from any classical generator baseline tested… at n=5 this reflects an underpowered non-significant difference test (~15% power against d=0.65, 80%-power detection floor d ≈ 2.0), not a positive equivalence finding" | INVERTED claim. Main paper now says cluster-floor p=0.019. |
| 421-422 | "The OD-EMD non-significance result is correspondingly reported WITHOUT a positive-equivalence inference, so multiplicity does not inflate a false claim there either" | Premises a "non-significance result" that is now a significance result. |
| 514-518 | "**Note on R1-M1 framing.** The R1-M1 table row above… and the two r3 metric bugs are closed." | The R1-M1 table is the stale tabular block above, so the meta-note compounds the staleness. |

The 14-21-SUMMARY explicitly flagged this in "Open items intentionally deferred":

> Remaining "no statistically detectable" / "non-significant difference under low power" passages in reviewer_response.md R1-M1 section | Provenance-gate-clean but narratively still reflect the pre-revision H2 framing; comprehensive narrative pass deferred

The flag is correct: provenance gate doesn't catch narrative drift. But a reviewer will. **The deferred narrative pass needs to happen before submission**, not after. Fix: rewrite paragraphs 408-437 and 511-518 to mirror the cluster-floor reading at lines 463-482.

### Number mismatches

The reviewer_response.md table at lines 447-453 (per-baseline Welch table) matches the post-fix welch_pairwise.json. ✓

The reviewer_response.md table at lines 149-160 (utility battery, TSTR R² etc.) is post-fix-consistent. ✓

The supp tables `tbl:welch_od_emd` lines 399-418 are PRE-FIX:
- e.g., wgan_cnn OD-EMD mean = 0.0543 in supp; post-fix value is 0.7989 (main line 511, line 623; reviewer_response line 450)
- IQP:SEL OD-EMD mean = 0.0275 in supp; post-fix value is 0.0282 (main line 419)

This is BLOCKING-1.

The supp table `tbl:welch_lr_emd` lines 448-467 has rough internal consistency with post-fix numbers (e.g., V3 vs WGAN-CNN at LR-EMD shows 0.0050 vs 0.1286 — matches main line 511). But the **caption** at line 423-441 still states "the LR-EMD reversal reported in §4.1 is highly statistically significant in the classical-leads direction" — the *direction* is wrong. Post-14-21, the LR-EMD direction is quantum-leads (negative Cohen's d in this table reflects quantum_lower_than_classical, which is *good* for the headline claim). The "classical-leads direction" wording is stale.

---

## R7 — Utility-discrimination scope (R1-M2)

**Main paper** treats the utility battery as part of the supp evidence; the main §4.1 narrative focuses on the 4 matched-budget distance metrics. The R1-M2-relevant "partial generator discrimination on TSTR/predictive/augmented" claim is in the rebuttal at lines 193-209 (reviewer_response.md). This framing is **HONEST**: the cluster-mean separation is reported as "small in absolute terms on the R²-saturated scale" while still "present and consistent across axes." That is exactly the right hedge.

**However**, the SCOPE has a contradiction with the post-14-21 main-paper headline. The honest reading paragraph at line 209 says the utility battery shows "partial generator discrimination" — fine. But the very next paragraph at lines 211-216 reverts to the stale "LR-DTW is the SOLE quantum-distinguishing result" framing. This is a direct contradiction with the immediately preceding paragraph in the same rebuttal section, and with the main paper's 4-of-4 dominance headline. It also makes the §R1-M2 response unable to defend the main paper's headline.

**Verdict**: R1-M2 utility-discrimination framing is HONESTLY UNDER-PLAYED (the "small in absolute terms" hedge is defensible), but the subsequent paragraph contradicts it. Fix is the BLOCKING-3 narrative pass.

---

## R8 — Adversarial alternative explanations

### Alternative 1: WGAN training instability is the real story

A reviewer might argue: *"WGAN-CNN is a known fragile architecture; its seed-42 outlier (OD-EMD 0.1587) and progressive critic-loss drift (supp §A.8 line 881-894) push the WGAN cluster mean dramatically. The quantum cluster's stability across seeds is the real signal, not a fundamental quantum advantage — you've shown that QWGAN is more stable, not better."*

**Is the manuscript robust against this reading?** PARTIALLY. The main caption at line 580-583 discloses the outlier, and supp §A.8 explicitly describes the WGAN-CNN critic-loss drift. The reviewer_response.md leave-one-out sensitivity (lines 478-481) defuses the cluster-floor-depends-on-cnn reading. But the main paper does NOT promote that leave-one-out sensitivity to the §4.1 OD-EMD paragraph; a reviewer skimming the main text alone can plausibly claim "the cluster-dominance hinges on one fragile WGAN architecture's outlier seed."

**Recommended hedge**: promote the leave-wgan_cnn-out sentence from reviewer_response into main §4.1 OD-EMD paragraph (see R4 recommendation 2).

### Alternative 2: Reported metrics correlate strongly enough that "4-of-4 dominance" is "1 underlying effect"

A reviewer might argue: *"LR-EMD and LR-DTW both measure log-return-scale alignment; OD-EMD and OD-DTW both measure OD-scale alignment after the same deterministic inverse-preprocessing pipeline. Your 4-of-4 cluster dominance is closer to 2 underlying axes with correlated metrics inside each axis. The OD-EMD claim might collapse to a redundant projection of the OD-DTW claim once you adjust for metric correlation."*

**Is the manuscript robust?** PARTIALLY. Main §4.1 (lines 521-524) explicitly disclaims: *"LR-DTW and LR-EMD measure different fidelity axes (temporal-alignment cost under non-linear time warping versus single-step marginal-distributional distance); on this dataset the two rankings are now concordant inside the quantum--vs--WGAN comparison."* This addresses LR-DTW vs LR-EMD but does NOT address the OD-DTW vs OD-EMD correlation or the cross-scale (LR vs OD) correlation. Lines 630-638 even note that "the within-seed range over the four quantum configurations is at most 0.0002, reflecting the deterministic component of the Pipeline B inverse-preprocessing pipeline" — i.e., the OD metrics are *known* to inherit deterministic structure from the inverse pipeline. A hostile reviewer will quote this back as evidence that OD-DTW and OD-EMD are not 2 independent axes.

**Recommended hedge** (main §4.1 or §5 caveat): *"The four matched-budget metrics are not mutually independent: OD-scale metrics inherit the deterministic component of the Pipeline B inverse-preprocessing pipeline (see lines XXX), and LR-DTW + lag-1 ACF jointly diagnose temporal-alignment structure. We therefore report the four-metric dominance as evidence concentrated on two underlying fidelity axes (log-return single-step and temporal-alignment, each in both scales) rather than four independent significance tests."*

### Alternative 3: Parameter-regime mismatch (Q 55 vs WGAN 73-78) confounds the comparison

A reviewer might argue: *"You compare a 55-parameter quantum generator against 73-78-parameter WGAN baselines. The WGAN cluster being worse may reflect the WGAN architecture's need for more parameters at this dataset size — not a quantum-vs-classical advantage. The headline finding is "parameter-matched" but 55 vs 78 is a 41% delta. Show the quantum result at 73 parameters."*

**Is the manuscript robust?** PARTIALLY. The quantum entrants span 55-135 parameters (IQP:SEL=55, V1=75, V2=135, V3=75 per Table 2). The 75-parameter quantum variants (V1 and V3) ARE near-matched to the WGAN range (73-78), so the comparison is *not* purely 55 vs 78. The Pareto figure at line 739-761 visualizes parameter-efficiency. But the *headline* repeatedly anchors to the 55p IQP:SEL.

**Recommended hedge**: ensure the abstract/headline explicitly notes that the 75-parameter quantum variants V1 and V3 are at-parameter-budget with the WGAN cluster (73-78p) and *still dominate*. The current abstract framing (line 49) is technically defensible — the Q range 55-135 brackets the WGAN range — but a hostile reviewer skimming the IQP:SEL-55 references will read it as a stacked comparison.

---

## R9 — The "convenient finding-inversion" question

### Does the manuscript address this directly?

**NO**. The supp §A.7 disclosure paragraph discusses the bug (origin, mechanism, scope) but does not say "the v1.2.4 conclusions are inverted on N of the M headline metrics" or "the bug fix happens to favor the headline hypothesis on every affected metric — here is why that should not be interpreted as bug-shopping." There is no preemptive paragraph anywhere in main, supp, or rebuttal.

**Recommended preemptive paragraph** (verbatim draft, suggested placement: main §5 "Limitations and Future Directions" *or* a new "Note on the inference-pipeline correction" subsection in §5):

> *Note on the inference-pipeline correction and the v1.2.4 narrative.* The first-round submission (origin tag v1.2.4) was prepared under an inference pipeline that under-scaled the WGAN-trained samples by a factor of 10 in the rescaled $[-1, +1]$ window space (see supp §A.7 for the technical disclosure). The corrected inference pipeline, applied uniformly at the `samples.npy` load boundary across every paper-cited consumer, shifts the WGAN cluster's distance metrics in the direction of higher (i.e., worse) values — by construction, since the bug attenuated WGAN-vs-real distances and the correction restores them. Three properties of the corrected pipeline make this fix a defensible methodological correction rather than a result-favoring intervention:
>
> *(i) Asymmetry-by-design.* The correction is gated by `model_kind in _WGAN_KINDS` and excludes the VAE and AR(2) baselines, which were trained without the $\times 0.1$ post-scaling and whose on-disk samples were already in the canonical $[-1, +1]$ space. The differential test against the pre-fix JSON snapshots confirms every VAE and AR(2) row is bit-identical pre/post; only WGAN-kind rows shift. This rules out a uniform "rescale-everything" intervention.
>
> *(ii) Quantum entrants are also WGAN-trained.* The four quantum variants (IQP:SEL, V1, V2, V3) are inside the `_WGAN_KINDS` gate and receive the same $\times 10$ correction as the classical WGAN baselines. The shift therefore moves the quantum cluster's pre-fix attenuated values up in the same direction; the cluster-dominance reading emerges *after* the correction restores the quantum cluster's true scale, not because the correction differentially advantages it.
>
> *(iii) Direction and magnitude are inherited from the bug, not chosen.* The 10× correction is the inverse of the documented $\times 0.1$ at the WGAN training/sample-export sites; both the factor and the gated subset are reproducible from the code comment at `archive/qgan_pennylane_SEL.py` and the Pitfall 3 note at `run_baselines.py:28`. We did not select the magnitude or direction of the correction *post hoc*.
>
> The corrected pipeline inverts the v1.2.4 reading on three of the four matched-budget metrics (preserves directionally on LR-DTW; flips OD-EMD from underpowered-null to cluster-significant; inverts the LR-EMD direction). Reviewers wishing to audit the directional shifts cell-by-cell can do so via the pre-fix JSON snapshots preserved in the release artifact set and the differential-test apparatus at `revision/scripts/v2.1_differential_test.py`.

This paragraph does three useful things at once: (a) flags the v1.2.4-vs-current-narrative inversion explicitly, (b) explains why the correction does not differentially advantage the headline hypothesis, and (c) gives the reviewer a concrete reproducibility handle.

---

## Total fix burden

### BLOCKING (must-do before resubmission)

1. **BLOCKING-1**: Refresh supp `tbl:welch_od_emd` (lines 380-421) and `tbl:welch_lr_emd` caption (lines 423-441) to post-fix numbers + post-fix direction language. ~20 numeric cells + 2 captions. Likely 1-2 hours.
2. **BLOCKING-2**: Reconcile main abstract (line 49 "three of the four headline metrics") with Contributions / Conclusions ("all four matched-budget metrics"). Pick one framing. ~3 sentence rewrites. <1 hour.
3. **BLOCKING-3**: Rewrite reviewer_response.md lines 211-216 + 421-437 + 514-518 to remove pre-14-21 "non-significant difference / sole quantum-distinguishing result" framing. ~3 paragraphs. 1-2 hours.
4. **R5 disclosure**: Add ~150-word paragraph to supp §A.7 disclosing the `run_welch_aggregator.py` threshold relaxation (text drafted above). <30 min.
5. **R9 preemptive paragraph**: Add the "Note on the inference-pipeline correction" paragraph to main §5 (text drafted above). ~30 min.
6. **R3 inversion footnote**: Add the "Reviewers comparing against v1.2.4" footnote to main §4.1 OD-EMD paragraph (text drafted above). <30 min.

BLOCKING total: ~6-8 prose paragraphs, 20 numeric cells, ~5-7 hours of focused work. Well within the 6-day deadline.

### DEFERRABLE (nice-to-have)

- R2.2: Bonferroni-corrected p-values footnoted alongside cluster-floor p in main §4.1. ~1 hour.
- R2.3: build a `welch_pairwise_dtw.json` and add a supp pairwise-DTW table mirroring the EMD ones. ~2 hours.
- R4: per-seed OD-EMD / OD-DTW / LR-EMD tables in supp. ~3 hours.
- R8.1 / R8.2 hedges in main §4.1. ~1 hour.

### The 6-day deadline implies

Do all BLOCKING items immediately (Day 1-2). Defer R2.2/R2.3 to a footnote-only Bonferroni statement. Defer R4 per-seed-table additions if and only if BLOCKING items reveal additional consistency work. Do R8.1 (leave-wgan_cnn-out promotion) and R8.2 (metric-independence hedge) on Day 3-4 — they are the highest-impact remaining hedges. The R9 preemptive paragraph is the single highest-leverage addition for surviving hostile review and should be in the first commit, not deferred.

A reviewer reading the current state will catch BLOCKING-1 within ~10 minutes (the supp table caption is the smoking gun). BLOCKING-3 surfaces within ~30 minutes of reading the rebuttal. Without the R9 paragraph, the manuscript is vulnerable to "convenient inversion" framing in any post-hoc reviewer note. With the BLOCKING fixes + R9 paragraph, the post-14-21 narrative is defensible.
