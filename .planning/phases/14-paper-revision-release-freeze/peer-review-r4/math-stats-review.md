# Peer Review r4 — Agent 1: Math & Statistics Review

**Scope:** Verify the mathematics and statistics behind the two surviving headline claims of the qGAN PhD project, as the last gate before the `v2.0-revision` Zenodo freeze.

**Worktree note:** The worktree HEAD (`c82169c`) predates the analysis work. Per task instructions, the eight KEY FILES (`run_welch_aggregator.py`, `run_matched2000_dualscale.py`, `run_distribution_emd.py`, `core/eval.py`, `welch_pairwise.json`, `matched2000_dualscale.json`, `distribution_emd.json`, `reviewer_response.md`, `methods_full.md`) were copied verbatim from the main repo at HEAD (`8180a5e`) into the worktree for analysis. All file:line citations below refer to those copies, which are byte-identical to the main repo's frozen-candidate files.

---

## Summary of findings

| Severity | Count |
|---|---|
| CRITICAL | 1 |
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 2 |

---

## PART A — Metric implementation correctness

### A1. Wasserstein/EMD (raw-sample) — CORRECT — `revision/core/eval.py:25-36`
`compute_emd` calls `scipy.stats.wasserstein_distance(real, fake)` on raveled raw samples. This is the mathematically correct 1-D EMD (L1 Wasserstein-1 between empirical CDFs). No issue.

### A2. R3-CR-2 un-standardize-fake recipe — MATHEMATICALLY CORRECT — `revision/run_matched2000_dualscale.py:384-387`
`trans_flat_raw = trans_flat * r["sigma"] + r["mu"]` un-standardizes the fake log-returns to raw units before `compute_emd(real_log_delta, trans_flat_raw)`. Both sides are now in the same (raw log-return) units. **Verified:** the post-fix LR-EMD aggregates (AR 0.00294, iqp_sel_55 0.01497, VAE 0.01583) match the magnitudes the docs anchor to. The fix is sound. **MEDIUM caveat (M3 below):** un-standardize-fake and standardize-real are *both* valid scale-matched recipes but produce EMD in different units; the choice of un-standardize-fake is unit-convention, not correctness — adequately disclosed at `methods_full.md:381-383`.

### A3. R3-CR-1 shared-edges histogram-density EMD — CORRECT, with a self-undermining finding — `revision/run_distribution_emd.py:124-172`
The v2 formulation (`density=False`, edges from real only, both histograms normalized to total-mass=1 over the same edge set, out-of-range mass disclosed via `fake_in_range_mass`) is mathematically sound and fixes the renormalization bias. The self-test (`self_emd == 0.0`, `self_fim == 1.0`, lines 341-350) is a valid invariant. **However** `methods_full.md:410-416` itself states that with shared edges, `density=True` vs `density=False` is *numerically inert* for `scipy.stats.wasserstein_distance` (which renormalizes weights internally) and the OD-scale v1→v2 aggregates are byte-identical — i.e. the *only* genuine R3-CR-1 contribution is the `fake_in_range_mass` disclosure stat. This is honestly disclosed; not a defect, but it means the "R3-CR-1 bug fix" is largely a disclosure-stat addition rather than a value correction. No action required — flagged only so the swarm does not over-credit it.

### A4. DTW — CORRECT implementation, but see C1 on the claim — `revision/core/eval.py:78-90`
`fastdtw` with Euclidean metric on `(-1,1)`-reshaped windows is a standard DTW. The `min`-over-64-real-windows / 100-synth subsampling recipe (`run_matched2000_dualscale.py:334-352, 403-426`) with `np.random.default_rng(s*31)` is reproducible. Implementation is fine. The *claim* built on it has issues (C1).

### A5. Moments / ACF ddof — CORRECT and consistently applied
`compute_moments` uses `np.std` default `ddof=0` (population) — `eval.py:42-58` — documented as a deliberate v1.0-locked notebook-parity decision. The *aggregator* statistics across seeds correctly use `ddof=1` (sample SD): `run_distribution_emd.py:319`, `run_welch_aggregator.py:69,107-108`, `run_matched2000_dualscale.py:537-538`. The ddof split is the *correct* convention: population SD within a window (descriptive), sample SD across the 5 seeds (inferential). **No issue** — this is right.

### A6. Welch / Cohen's d arithmetic — VERIFIED EXACT
I recomputed all 20 OD-EMD pairs from `matched2000_dualscale.json` source rows: `scipy.stats.ttest_ind(..., equal_var=False)` and pooled-SD Cohen's d. Max absolute deviation from the values in `welch_pairwise.json` = **3.3e-16**. The JSON is an exact, faithful emission of the source data. The aggregator's `_cohen_d` (pooled SD, `ddof=1`, lines 65-72) is the standard formula. **No arithmetic error.**

---

## PART B — PRIMARY FOCUS: the equivalence claim's inference logic

### B1. CRITICAL — The OD-EMD "parametric-efficiency equivalence" claim is **not statistically defensible as stated**: it equates a non-significant difference test with positive evidence of equivalence.

**Where:** `revision/docs/reviewer_response.md:269-272, 283-323`; `revision/docs/methods_full.md:398-399`; `revision/run_welch_aggregator.py:138-182` (the `strong_claim_thresholds` gate).

**The claim verbatim** (`reviewer_response.md:269-272`):
> "55 quantum parameters achieve OD-scale EMD **statistically equivalent** to classical generators of 73-562 generator params ... (Welch p > 0.36, |d| ≤ 0.65, n=5)."

**The defect.** The claim's *entire* statistical backing is: every one of 20 quantum-vs-classical Welch t-tests returned p > 0.36, and the observed |Cohen's d| ceiling is 0.65. This is the textbook **"absence of evidence is not evidence of absence"** fallacy. A non-significant Welch p-value means *the difference test failed to reject H0: means equal* — it is **not** a test of H1: means are equivalent. The aggregator gate at `run_welch_aggregator.py:168-182` enforces `floor_welch_p_OD > 0.36` and `ceiling_abs_cohen_d_OD <= 0.65` and labels passing this `strong_claim_thresholds`. **Asserting a high p-value as a threshold for an "equivalence" claim inverts the logic of hypothesis testing.** A *higher* p-value is *weaker* evidence against difference, not *stronger* evidence for equivalence.

**Quantified — the test is severely underpowered (this is the crux).** I ran a power analysis for the two-sample Welch t-test at n=5/group, α=0.05 two-sided:

| True Cohen's d | Power to detect |
|---|---|
| 0.50 | 0.108 |
| 0.65 | 0.149 |
| 0.80 | 0.201 |
| 1.00 | 0.286 |
| 1.50 | 0.549 |
| 2.00 | 0.791 |

**The minimum effect size detectable at 80% power with n=5/group is d ≈ 2.02.** In other words, a *real* difference of nearly any plausible magnitude (d up to ~2) would almost certainly produce a non-significant p-value at this sample size. A p > 0.36 result is therefore **the expected outcome whether or not the generators are truly equivalent** — it carries essentially zero discriminating information. The claim "p > 0.36 ⟹ equivalence" is not just weak; it is structurally uninformative.

**A proper equivalence test fails.** I ran the correct procedure — Two One-Sided Tests (TOST) — against the |d| ≤ 0.65 margin the doc itself cites as a ceiling:

- **Unpaired TOST, margin |d| = 0.65:** **0 / 20 pairs** pass equivalence at α=0.05; worst (max) TOST p = 0.4964.
- **Unpaired TOST, margin |d| = 0.50:** 0 / 20 pass; worst TOST p = 0.5873.
- The data are strongly seed-paired (Pearson r = 0.92–0.98 between quantum and each classical across the 5 seeds). A **paired TOST** is the statistically correct design. Even paired, only **8 / 20 pairs** pass at a generous absolute margin of 0.65×(typical SD); **0 / 20** pass at a 0.3-SD margin. The absolute margin needed for *all 20* pairs to clear equivalence is **~0.085 EMD units ≈ 303% of the mean OD-EMD** — i.e. one would have to declare differences of up to 3× the metric value itself "equivalent."

**Conclusion:** Under any defensible equivalence test, the OD-EMD equivalence claim **does not hold**. The claim survives *only* because it uses the wrong test (a difference test) in the wrong direction (high p ⟹ equivalence).

**Note on the |d| ≤ 0.65 criterion.** The doc presents |d| ≤ 0.65 as a co-equal threshold, but it is a **post-hoc ceiling on the observed effect**, not a pre-specified equivalence margin. With n=5, the 95% CI on Cohen's d for a single pair spans roughly ±0.5–1.8 d-units (I computed paired-diff CIs: e.g. iqp_sel_55 vs wgan_cnn has a 95% CI of [-0.103, +0.049] EMD, d-units ±1.82). The point estimate |d| = 0.65 is wholly compatible with a true |d| well above 1.0. Citing the observed |d| as if it were a bound on the true effect compounds the B1 error.

**Required remediation (any one of the following makes the claim defensible):**
1. **Reframe** (lowest effort, fully honest): drop the words "equivalent"/"equivalence" and state "**no statistically detectable OD-EMD difference at n=5**; this study is underpowered (80%-power floor d ≈ 2.0) to detect moderate differences and does not establish equivalence." This is consistent with the honest hedging already present at `methods_full.md:530` ("within seed variance") and `:441-442` ("statistically non-significant ... no equivalence test is computed").
2. **Run a proper pre-specified TOST** against a margin justified *before* seeing the data, and report it. (Per my analysis above this will **fail** at any reasonable margin — so option 1 is the realistic path.)
3. Report the **95% CIs on the mean differences** and let the reader see they are wide.

This is the single most important finding of this review. The metric *implementations* are correct; the *inference layer built on top of them* makes an unsupported claim. Because `welch_pairwise.json` is the cited anchor for a "strong claim" and is about to be frozen under a permanent DOI, **the claim wording must be corrected before the freeze.**

### B2. HIGH — Multiple-comparisons posture is internally inconsistent (and used selectively).
`reviewer_response.md:311-312` notes the 20 OD pairs are "computed two-sided." For the **OD-EMD equivalence** claim, no multiple-comparison correction is applied — and here it is genuinely moot, since the *minimum* raw Welch p is 0.3652 (Bonferroni-adjusted → 1.0); nothing is significant either way. **However**, the same project elsewhere relies on the **LR-DTW** family of ~12+ quantum-vs-classical comparisons (B3) and the withdrawn LR-EMD claim, none with a stated family-wise correction. The freeze artifact should state a single, consistent multiple-comparisons policy across all pairwise families. As-is, the absence of correction is harmless for the (null) OD-EMD result but is an unstated gap for the DTW claim. **MEDIUM-to-HIGH**: flag as HIGH because it touches a *surviving headline claim* (DTW).

### B3. MEDIUM — DTW "dominance" claim: the LR-DTW part is statistically sound; the OD-DTW "6.5× vs Orlandi" part is **not quantum-specific**.
**LR-DTW (sound):** I tested quantum-vs-WGAN on log-return DTW. Quantum (iqp_sel_55 0.985, V1 0.940) beats every WGAN (wgan_lstm 1.581, wgan_mlp 2.624, wgan_cnn 6.863) with Welch p = 0.0007–0.0112 and Mann–Whitney p = 0.004 across all pairs. The separation is large, consistent across all 4 quantum variants and all 3 WGANs, and survives even a Bonferroni correction over ~12 pairs (0.0112×12 ≈ 0.13 for the weakest pair — borderline, but MWU p=0.004 holds). The directional claim "every quantum beats every WGAN on LR-DTW" **is supported**. Caveat: VAE LR-DTW = 0.088 is *better* than quantum and is correctly excluded as posterior-collapse (`methods_full.md:446-449`) — honest.

**OD-DTW (overstated):** `reviewer_response.md:278-281` says "best quantum beats Orlandi et al. by 6.5× on OD-DTW. This is the temporal-structure capture quantum is specifically designed for." Arithmetic checks (1.954/0.300 = 6.51×). **But** wgan_lstm (0.301) and wgan_mlp (0.302) achieve the *same* OD-DTW as the quantum cluster (0.298–0.302). The 6.5×-vs-Orlandi improvement is therefore **shared by two classical baselines** and is *not* evidence of quantum-specific temporal-structure capture on the OD scale. The "specifically designed for" framing is unsupported for OD-DTW. `methods_full.md:436-442` is more honest ("mixed with the classical generator cluster ... statistically non-significant"); the **reviewer_response.md wording overclaims relative to the methods doc.** Recommend aligning `reviewer_response.md:278-281` to the methods-doc framing: the OD-DTW Orlandi improvement is a *matched-budget-wide* result, and only the **LR-DTW** result is quantum-distinguishing.

### B4. MEDIUM — wgan_cnn pairs are dominated by a single outlier seed; the "equivalence" with wgan_cnn is an artifact.
For wgan_cnn, seed 42 OD-EMD = 0.1587 vs the other four seeds at 0.020–0.034 (a ~5× outlier). This inflates wgan_cnn's SD to 0.0586 (vs ~0.005 for every other model). The Welch test against this group has its denominator dominated by one point: the iqp_sel_55-vs-wgan_cnn pair shows the *largest* |d| (0.644, the very value that defines the 0.65 ceiling) and the *smallest* p (0.365, the value that defines the 0.36 floor) — i.e. **both `strong_claim_thresholds` extrema are set by a single anomalous seed.** The 95% paired-diff CI for this pair is [-0.103, +0.049] EMD (d-units ±1.82) — uselessly wide. The robust Mann–Whitney p for this pair is 0.548. The claim's headline thresholds are not robust; a sensitivity note (or median-based / outlier-trimmed companion statistic) should accompany the frozen JSON. At minimum, `reviewer_response.md` should disclose that the |d| = 0.65 ceiling is an outlier-driven extremum, not a typical pair.

### B5. MEDIUM — n=5 is below conventional inferential minimums and this limitation is under-disclosed at the claim site.
n=5 seeds/group is acknowledged in passing (`methods_full.md:46` "variance is over training-seed variation"), but the *consequence* — that the design has ~15% power against a moderate d=0.65 effect — is never stated next to the equivalence claim. A reader of `reviewer_response.md:269-323` is given p-values and Cohen's d with no power context and would reasonably (mis)read "p > 0.36" as evidence of similarity. The freeze artifact should carry an explicit power/limitation sentence at the claim site (this is part of the B1 remediation).

### B6. LOW — `mannwhitneyu` is computed and stored but never used in any claim or gate.
`run_welch_aggregator.py:99` emits `mwu_stat`/`mwu_p` for every pair, but the `strong_claim_thresholds` gate (lines 138-182) only enforces Welch p and Cohen's d. The MWU values are good transparency (and for n=5 the MWU is arguably *less* misleadable than Welch for the equivalence framing), but storing an unused statistic alongside the load-bearing one invites a reader to assume it was part of the inference. Minor — recommend a one-line note in the JSON `notes` field that MWU is descriptive-only. Not a blocker.

### B7. LOW — `data_hash` is a hardcoded literal, not a computed digest.
`run_welch_aggregator.py:32` (`DATA_HASH = "91e447d4624e25b3"`) and `run_distribution_emd.py:103` hardcode the corpus hash rather than computing it from the source file. If `matched2000_dualscale.json` were ever regenerated with different content, the aggregator would still stamp the old hash and the `strong_claim_thresholds` gate would not catch the drift. For a permanently-frozen DOI artifact this is a (small) provenance-integrity gap. Recommend the aggregator compute the hash of its actual input. Not a freeze blocker given the repo is about to be tag-frozen anyway.

---

## PART C — Closed r3 items (regression check only — NOT re-litigated)

- **R3-CR-2 (50× LR scale mismatch):** fix at `run_matched2000_dualscale.py:384` is mathematically correct (A2). LR-EMD aggregates land at the disclosed magnitudes. **No regression.**
- **R3-CR-1 (histogram-density renormalization):** fix at `run_distribution_emd.py:124-172` is correct; the project honestly discloses the OD values are byte-identical and the real contribution is the `fake_in_range_mass` stat (A3). **No regression.**
- **Path A reframe (LR-EMD-vs-WGAN claim withdrawn):** the aggregator correctly enforces *only* OD thresholds (`run_welch_aggregator.py:167-182`) and the LR-EMD pairs are emitted as transparency-only. The withdrawal is handled correctly. **No new issue** — except that the *surviving* OD-EMD claim has the B1 problem, which is independent of the Path A withdrawal.
- **VAE/AR not parameter-matched:** disclosed (`reviewer_response.md:294-295`). Not re-litigated.

---

## Verdict rationale

The metric *implementations* (EMD, DTW, the two r3 fixes, ddof choices, Welch/Cohen arithmetic) are all mathematically correct and verified to machine precision — there is **no regression and no new computational bug**.

The blocker is **B1**: the surviving OD-EMD headline claim asserts "statistical equivalence" on the basis of non-significant difference tests at a sample size (n=5) with ~15% power against a moderate effect. A proper equivalence test (TOST) **fails for 0/20 pairs** at any defensible margin. Because this claim is anchored in `welch_pairwise.json`'s `strong_claim_thresholds` block and is about to be frozen under a permanent, irreversible Zenodo DOI, the wording **must** be corrected first. The fix is low-effort and non-destructive: reframe "equivalent / equivalence" → "no statistically detectable difference at n=5, underpowered, not an equivalence claim" — language the project's own `methods_full.md:441-442` already uses for DTW. No re-computation or re-training is required; only the claim text in `reviewer_response.md:269-323` / `methods_full.md:398-399` and ideally a `notes` clarification in `welch_pairwise.json`. B3 (OD-DTW overclaim) and B4 (outlier-driven thresholds) should be fixed in the same editing pass.

This is **GO-WITH-FIXES**, not BLOCK: the underlying data, code, and metrics are sound and frozen-ready; what must change is claim *wording*, which is a contained, documentation-only edit.

FREEZE VERDICT: GO-WITH-FIXES
