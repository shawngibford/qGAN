# Agent 1 — Text↔Evidence Corroboration Findings

**Audit target:** HEAD `50658a6` (v1.2.1)
**Files audited:** main (4) copy.tex, supp_material.tex
**Scope:** Prose claims about trends, orderings, comparisons, qualifiers — verified against JSON data sources (`matched2000_dualscale.json`, `welch_pairwise.json`, `predictive_discriminative_matched2000.json`, `transform_ablation/summary.md`, `transform_ablation/metrics.csv`).

## Summary
- BLOCK: 1 finding (VAE std-collapse mischaracterisation — appears in 4 prose locations)
- FLAG:  3 findings
- NIT:   2 findings

---

## BLOCK findings

### B-1: VAE characterised as "log-return standard deviation collapses toward zero" — the data refutes std-collapse

- **Location:** main (4) copy.tex lines 362 (Table 2 caption), 392 (Table 2 footnote), 433–436 (cross-model VAE exclusion paragraph), 527 (VAE characterization paragraph in §4.1)
- **Verbatim quotes from .tex:**
  - Line 362: "the VAE operates in a degenerate generation regime in which the synthetic log-return standard deviation collapses toward zero, making DTW vacuously small"
  - Line 392: "$^{\dagger}$VAE LR-DTW excluded from row leader: the small value reflects a degenerate generation regime (collapsed synthetic log-return variance)"
  - Lines 433–436: "...regime in which the synthetic log-return standard deviation collapses toward zero"
  - Line 527: "its mean LR-DTW of $0.0876$ is anomalously small because the synthetic log-return standard deviation collapses --- a near-constant sequence is trivially close in DTW to any reference"
- **JSON evidence:**
  - `matched2000_dualscale.json` aggregate `(model_kind=vae, scale=log_return, metric_name=moment_std)`: mean = **0.01855**, std = 0.00186, n_seeds = 5
  - Real-data log-return std reference ≈ 0.0217 (per `.planning/PAPER-SUBMISSION-HANDOFF.md` §5 prohibition #1 and the matched-pipeline preprocessing)
  - VAE synthetic std is ~86% of real (0.0186 / 0.0217) — that is **not** "collapsed toward zero" and the synthetic sequence is **not** "near-constant"
  - The actual anomaly is `acf_lag1_mean` = **−0.648** vs real ≈ −0.064 (a 10× anti-correlation overshoot, not a variance collapse) — and the manuscript correctly cites this number at lines 487, 489, 530
  - The same paragraph at lines 522–524 actually *contradicts* the std-collapse claim by noting "LR-EMD$\,\approx 0.016$ is also in-cluster with the other generators on the marginal axis, indicating that the per-step distribution is captured" — which would not be possible if the std had collapsed
- **Discrepancy:** All four locations attribute VAE's small LR-DTW to a *variance/std collapse*; the data shows the variance is comparable to real. The actual mechanism (and the one the handoff's hard prohibition #1 mandates) is *temporal-structure degeneracy*: strong negative lag-1 autocorrelation produces a high-frequency zig-zag whose DTW path matches the real series under warping despite carrying no real temporal information. The text already documents the −0.648 ACF as the "diagnostic that exposes the collapse" but then misnames the mechanism.
- **Suggested fix (single rewrite, applied at all 4 sites):**
  - Replace "synthetic log-return standard deviation collapses toward zero" / "collapsed synthetic log-return variance" / "near-constant sequence" with: "the synthetic log-return series exhibits a strongly anti-correlated step-to-step structure (lag-1 ACF $\approx -0.65$ vs real $\approx -0.064$), so the high-frequency oscillation is warped-aligned to the real series at low DTW cost despite carrying no real temporal information."
  - This preserves the conclusion ("LR-DTW vacuously small for VAE, exclude from dominance comparison") while making the mechanism match the data and the hard prohibition.

---

## FLAG findings

### F-1: "Every classical adversarial baseline outperforms every quantum variant" on LR-EMD reads as per-seed but only holds at the mean

- **Location:** main (4) copy.tex line 415–416 (§4.1 lead-in), line 691–693 (§4.4 Limitations), line 782–783 (§5 Concluding Remarks)
- **Verbatim quotes:**
  - Line 415–416 (§4.1): "On the log-return single-step marginal axis (LR-EMD) the direction reverses: every classical adversarial baseline outperforms every quantum variant, and the AR(2) reference leads"
  - Line 691–693 (§4.4): "every classical adversarial baseline outperforming every quantum variant (Section~4.1)"
  - Line 782–783 (§5): "every classical adversarial baseline outperforms every quantum variant"
- **JSON evidence:** Per-seed values from `matched2000_dualscale.json#rows`, scale=log_return, metric_name=emd:
  - At seed 42, wgan_cnn LR-EMD = **0.01586** is *higher (worse)* than every quantum variant at the same seed (V1 = 0.01507, V2 = 0.01503, V3 = 0.01428, iqp_sel = 0.01490) — i.e., 4 of 60 cells in the quantum × classical-adversarial × seed grid violate "classical < quantum"
  - At the mean level, classical adversarial means (wgan_cnn = 0.00711, wgan_mlp = 0.01031, wgan_lstm = 0.01272) are all lower than every quantum mean (range 0.01432–0.01502), so the claim is correct **on per-model means only**
- **Discrepancy:** The phrasing "every classical adversarial baseline outperforms every quantum variant" structurally parallels the LR-DTW per-seed dominance claim ("no quantum--classical seed overlap on LR-DTW") in the same section. A reader will naturally infer per-seed dominance on LR-EMD as well, but per-seed dominance does *not* hold (wgan_cnn seed-42 fails). §4.1 line 462–466 does include the correct hedge ("On per-model means... per-seed dominance was not separately tested on this axis"), but the lead-in at 415–416 and the §4.4 / §5 echoes lose that hedge.
- **Suggested fix:** Insert "On per-model means," (or equivalent qualifier) before each occurrence, and/or note "with one per-seed counter-example: wgan_cnn at seed 42 has LR-EMD 0.0159, above every quantum variant at that seed." This is the same hedging discipline the manuscript applies on LR-DTW ("per-seed dominance, no overlap") vs lag-1 ACF ("mean-level dominance, per-seed overlap observed").

### F-2: §4.1 "the four quantum WGAN-GP variants uniformly outperformed... and the AR(2) reference" — "uniformly" elides ACF per-seed overlap

- **Location:** main (4) copy.tex line 408–411 (§4.1 Cross-Model Comparison opening)
- **Verbatim quote:** "On the log-return temporal-alignment axis (Dynamic Time Warping on the log-return scale, and the lag-1 autocorrelation of the log-return series), the four quantum WGAN-GP variants \emph{uniformly} outperformed the three parameter-matched classical adversarial baselines and the AR(2) reference."
- **JSON evidence:** On LR-DTW the "uniformly" claim is supported per-seed (60/60 cells, verified from `matched2000_dualscale.json#rows`). On the lag-1 ACF half, however, per-seed overlap exists: `acf_lag1_mean` at (wgan_lstm, log_return, seed=46) = **−0.0761**, distance from real (−0.064) = 0.012, *closer* than every quantum-cluster mean (V3=0.0255, iqp=0.0309, V2=0.0328, V1=0.0357 absolute distance from real). The manuscript itself acknowledges this at lines 491–492 and 510.
- **Discrepancy:** "Uniformly" suggests every quantum > every classical at *every* level (per-seed). On LR-DTW that's true; on the conjoined "log-return temporal-alignment axis" that pulls in lag-1 ACF as well, "uniformly" is over-strong because ACF dominance is mean-level only.
- **Suggested fix:** Rewrite as "the four quantum WGAN-GP variants \emph{uniformly} outperformed the three parameter-matched classical adversarial baselines and the AR(2) reference on LR-DTW (per-seed dominance), and on lag-1 ACF the quantum-cluster mean is closer to the real reference than any classical-baseline mean (per-seed overlap noted)." The §1.4 bullet (line 103) and the §4.1 closing paragraph (line 627–636) already use exactly this two-clause structure — the lead-in at 408–411 should match.

### F-3: §4.3 "On the log-return temporal-structure axis (LR-DTW + lag-1 ACF closeness to real), the quantum generators uniformly outperform..." — same elision as F-2

- **Location:** main (4) copy.tex line 673 (§4.3 Theoretical and Practical Implications)
- **Verbatim quote:** "On the log-return temporal-structure axis (LR-DTW + lag-1 ACF closeness to real), the quantum generators uniformly outperform the parameter-matched classical adversarial baselines on the evaluated photobioreactor dataset (Section~4.1)."
- **JSON evidence:** Same as F-2 — "uniformly" reads per-seed but ACF dominance is mean-level only (wgan_lstm seed-46 lag-1 = −0.0761 closer to real than any quantum mean).
- **Discrepancy:** When parsed at the mean level, "every quantum mean > every classical mean" holds: quantum max distance from real-ACF reference (V1 = 0.0357) < best classical distance (wgan_cnn = 0.0472). So the claim is technically defensible if "uniformly" = "uniformly at the mean level". But the §4.1 close (line 627–636) explicitly distinguishes per-seed (LR-DTW) from mean-level (ACF); §4.3 should inherit that hedge for consistency.
- **Suggested fix:** Add a one-clause qualifier: "...the quantum generators uniformly outperform the parameter-matched classical adversarial baselines (per-seed on LR-DTW; mean-level on lag-1 ACF closeness, with per-seed overlap noted)..." This costs one line and harmonises §4.3 with §1.4, §4.1 close, §4.2, and §5.

---

## NIT findings

### N-1: "Welch p > 0.36" claim across abstract, §1.4, §4.1, §4.2, §5 is technically correct but tight (floor = 0.36521)

- **Location:** main (4) copy.tex lines 49 (Abstract), 105 (§1.4), 562–563 (§4.1), and elsewhere — all read "Welch p > 0.36"
- **Verbatim quote (Abstract, representative):** "Welch p > 0.36, max |Cohen's d| <= 0.65 at n=5 power approx 15\%, TOST equivalence not satisfied."
- **JSON evidence:** `welch_pairwise.json#summaries.OD_floor_welch_p_quantum_vs_classical` = **0.36521**. So "$p > 0.36$" is true but the floor only exceeds it by 0.005. The 20-pair detail in supp Table A.X+1 lists the tightest pair (V3 vs wgan_cnn) at p = 0.365.
- **Discrepancy:** None — the claim is correct. The NIT is that "p > 0.36" reads as if the floor might be comfortably above (e.g., 0.4 or 0.5), but it's actually 0.365. A reviewer who recomputes will find the floor is essentially at the threshold, and may want the more precise statement.
- **Suggested fix (optional):** Replace "Welch p > 0.36" with "minimum Welch p = 0.37 across 20 quantum-vs-classical OD-EMD pairs" or "Welch p \geq 0.37" (rounded up at 2 d.p.). Same JSON-rounded value, but signals to the reader that the floor is tight, not loose. Skip if abstract word count is binding.

### N-2: §4.1 "LR-EMD ≈ 0.016 is also in-cluster with the other generators on the marginal axis" describes VAE as in-cluster but VAE is the highest LR-EMD model

- **Location:** main (4) copy.tex line 522–524 (VAE characterization paragraph)
- **Verbatim quote:** "On the OD-marginal it is well-aligned (OD-EMD $\approx 0.026$); its LR-EMD$\,\approx 0.016$ is also in-cluster with the other generators on the marginal axis, indicating that the per-step distribution is captured."
- **JSON evidence:** LR-EMD per-model means (from `matched2000_dualscale.json` aggregates): AR=0.0029, wgan_cnn=0.0071, wgan_mlp=0.0103, wgan_lstm=0.0127, V3=0.0143, iqp=0.0150, V1=0.0150, V2=0.0150, **VAE=0.0158 (highest of all 9)**.
- **Discrepancy:** "In-cluster" is loosely defensible (0.0158 is close to the quantum cluster at 0.0143–0.0150, ~6% above iqp), but it under-states that VAE has the *worst* LR-EMD of all 9 models. "Per-step distribution is captured" is also generous when VAE is being characterised as a degenerate regime in the next sentence — a reviewer may object to the asymmetry.
- **Suggested fix (optional):** Hedge to "its LR-EMD ≈ 0.016, while the worst of the 9 models, is still within 6\% of the quantum cluster (~0.015), so the single-step marginal is approximately captured even though the temporal structure (lag-1 ACF) is not." Skip if §4.1 length is binding.

---

## What was NOT flagged (sanity check)

Verified correct against JSON (representative):
- **Abstract LR-DTW range 0.94–1.12 vs 1.58–6.86:** quantum V1=0.94000, V2=0.94949, iqp=0.98548, V3=1.12246; classical wgan_lstm=1.58119, wgan_mlp=2.62428, wgan_cnn=6.86297 — ranges match to displayed precision.
- **Abstract / §1.4 / §4.2 / §5 quantum lag-1 ACF range −0.0997 to −0.0895:** V1=−0.0997, V2=−0.0968, V3=−0.0895, iqp=−0.0949 — bounds exact at 4 d.p.
- **"Closer to real than any classical-baseline mean":** quantum max distance from −0.064 reference = 0.0357 (V1); best classical = 0.0472 (wgan_cnn); all 4 quantum < best classical ✓.
- **Per-seed LR-DTW 60/60 dominance** (supp Table A.X): every cell verified, tightest margin at seed 46 = 0.2052 (≈15.7% of classical), matches "≈16%".
- **wgan_lstm seed-46 lag-1 ACF = −0.0761** ≈ −0.076 as quoted in §4.1 and Fig 3 caption.
- **Table 1 all 14 cells** (IQP:SEL 5-seed mean ± std on both scales): all match `matched2000_dualscale.json` aggregates to displayed precision.
- **Table 2 row leaders** (LR-EMD: AR=0.0029; OD-EMD: VAE=0.0257; LR-DTW: V1=0.94; OD-DTW: V2=0.2984; lag-1 ACF: V3=−0.0895): each is the row minimum / closest-to-real, no false bolding.
- **wgan_cnn seed-42 OD-EMD outlier (0.1587)** and "other four wgan_cnn seeds lie between 0.020 and 0.034": verified (actual 0.0205–0.0339).
- **Quantum OD-EMD within-seed range ≤ 0.0002:** verified, max range across seeds = 0.00016.
- **Discriminative score collapse (0.4089, std=0)** across all 9 generators: verified from `predictive_discriminative_matched2000.json`.
- **Pipeline-B vs Pipeline-C ablation numbers** (OD-EMD 0.0276 vs 0.0261; OD-ACF 0.696 vs 0.697; etc.) at 1000 epochs: verified from `transform_ablation/metrics.csv` and `summary.md`.
- **Orlandi DTW 1.954 vs matched-budget 0.30 ≈ "6.5 times lower":** 1.954/0.30 = 6.51 ✓.
- **AR(2) LR-EMD = 0.0029 "roughly five times lower than the quantum cluster":** quantum mean ≈ 0.0147; 0.0147/0.0029 = 5.07 ✓.

The four hard prohibitions (VAE = degenerate-regime not posterior-collapse; LR-DTW is the surviving signal not LR-EMD; real ACF = −0.064 not −0.029; Pipeline B has no Lambert W) are correctly enforced — except for the *separate* B-1 problem that the manuscript still attributes the VAE LR-DTW anomaly to *variance* collapse rather than to the temporal-structure (lag-1 ACF) anomaly that the prohibition mandates and that the data actually shows.
