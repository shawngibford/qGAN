# 4-Agent Audit — Synthesized Findings

**Audit target:** HEAD `50658a6` (v1.2.1)
**Files audited:** `main (4) copy.tex`, `supp_material.tex` + all 11 figure files + revision/ code
**Audit date:** 2026-05-28
**Agents:** 1 (text↔evidence) · 2 (cross-section) · 3 (figures↔captions) · 4 (prose↔code)

## Headline

| Severity | Count | Verdict |
|---|---|---|
| **BLOCK** | **1** | Must fix — scientific mischaracterization contradicts hard prohibition #1 |
| **FLAG** | **9** | Recommended — calibration honesty / cross-section coherence |
| **NIT** | **9** | Optional polish |

**No prohibition triggered** other than the BLOCK below (which the post-swarm audit's prohibition-sentinel agent did NOT catch because the prohibited phrase "posterior collapse" was correctly removed — but the prohibited *mechanism* (variance collapse) was reintroduced under a different lexical shape).

---

## §1 — BLOCK (must fix before submission)

### B-1: VAE LR-DTW anomaly mis-mechanized as "variance collapse" — data shows the opposite

**Locations (4 sites, single rewrite applies to all):**
- main (4) copy.tex lines **360–362** (Table 2 caption)
- main (4) copy.tex lines **391–394** (Table 2 footnote)
- main (4) copy.tex lines **433–436** (cross-model VAE-exclusion paragraph)
- main (4) copy.tex line **527** (VAE characterization in §4.1)

**Current prose (representative):** "the synthetic log-return standard deviation collapses toward zero, making DTW vacuously small"

**JSON evidence:** `matched2000_dualscale.json` → VAE `moment_std` mean = **0.0186** (sample-std 0.00186, n=5), real-data log-return std ≈ 0.0217. VAE synthetic std is **86%** of real — NOT collapsed. The actual diagnostic is `acf_lag1_mean` = **−0.648** vs real −0.064 (10× anti-correlation overshoot).

**Why it's a BLOCK:** This contradicts hard prohibition #1 from PAPER-SUBMISSION-HANDOFF.md §5, which explicitly states "Log-return std is 0.0186 (≈ real 0.0217). The anomaly is lag-1 ACF = −0.648 vs real −0.064. Never re-claim 'posterior collapse' or 'synthetic std ≈ 4×10⁻⁴'." The same §4.1 paragraph at lines 522–524 also self-contradicts: "LR-EMD ≈ 0.016 is also in-cluster... indicating that the per-step distribution is captured" — which is incompatible with std collapse.

**Single rewrite, applied at all 4 sites:** Replace "synthetic log-return standard deviation collapses toward zero" / "collapsed synthetic log-return variance" / "near-constant sequence" with: *"the synthetic log-return series exhibits a strongly anti-correlated step-to-step structure (lag-1 ACF ≈ −0.65 vs real ≈ −0.064), so the high-frequency oscillation is warped-aligned to the real series at low DTW cost despite carrying no real temporal information."*

---

## §2 — FLAG (recommended for calibration honesty)

### F-1: OD-marginal comparator-set scope drifts narrow in Abstract / PLS / §1.4-bullet-3

**Sections:** Abstract (line 49) + Plain Language Summary (line 59) + §1.4 bullet 3 (line 103) **vs.** §1.4 bullet 4 (line 105), §4.2 (line 658), §5 (lines 777–781), §4.1 statistical-test prose, supp Welch OD-EMD table.

**Drift:** Front-of-paper narrows OD-marginal claim to "adversarial baselines" only; back-of-paper uses wider "adversarial + VAE + AR(2)" (matching the actual 20-pair Welch table). Same M8-class scope-drift that surfaced in the post-swarm A5 audit.

**Fix:** Insert parenthetical in front-of-paper to widen scope:
- Abstract: add "(against the full parameter-matched comparator set: adversarial baselines, VAE, AR(2))"
- §1.4 bullet 3: add "(against the full set of parameter-matched comparators; see bullet 4)"
- PLS: change "adversarial baselines" → "comparator models"

### F-2: §1.4 bullet 3 omits the LR-EMD reversal that §4.2, §4.3, §5 all carry

**Sections:** §1.4 bullet 3 (line 103) **vs.** §4.2 (line 658), §4.3 (line 673), §5 (lines 781–783).

**Drift:** The canonical "what did the paper find" §1.4 bullet reports the LR-DTW+lag-1 ACF positive direction and the OD-marginal null, but **does NOT mention the LR-EMD reversal** (every classical adversarial baseline beats every quantum variant on LR-EMD; AR(2) leads). Every body section that touches the bifurcated finding explicitly discloses this reversal.

**Fix:** Append to §1.4 bullet 3 after "On the optical-density marginal, no advantage is observed.": "*On the log-return single-step marginal (LR-EMD), the direction reverses and every classical adversarial baseline outperforms every quantum variant (Section~4.1).*"

### F-3: §5 bookend drops the "TOST equivalence not satisfied" hedge

**Sections:** §5 (lines 777–780) **vs.** Abstract (line 49), §1.4 bullet 4 (line 105), §4.1 (lines 567–575), §4.4 (line 702).

**Drift:** §5 uses "statistically indistinguishable" on OD-EMD without the immediate "but not positively equivalent" / "TOST not satisfied" guard that §4.1 and Abstract attach. Combined with F-1, a casual reader of just Abstract + §5 could come away thinking equivalence was demonstrated.

**Fix:** Add ", TOST equivalence not satisfied" inside §5's parenthetical at line 779–780.

### F-4: Pipeline C operation order is misdescribed in 3 sites

**Sections:** main line 291, supp lines 581, 604.

**Current prose:** "Pipeline C: Pipeline B followed by inverse Lambert W"

**Code reality (`revision/core/data.py:269-282`):** Pipeline C inserts Lambert W **between** standardize and rescale: `log-returns → standardize → inverse_lambert_w → rescale to [-1,1]`. The current prose implies Lambert W is appended **after** rescale-to-[-1,1] — a literal reproduction would build a different transform.

**Fix:** Reword as "Pipeline C: log-return → standardize → **inverse Lambert W** → rescale to [-1, 1] (i.e., Pipeline B with an inverse Lambert W step inserted between standardization and rescaling, matching the v1.1 published pipeline)".

### F-5: "Every classical adversarial baseline outperforms every quantum variant" on LR-EMD reads per-seed but only holds at the mean

**Sections:** main lines 415–416 (§4.1 lead-in), 691–693 (§4.4), 782–783 (§5).

**JSON evidence:** At seed 42, wgan_cnn LR-EMD = 0.01586 > every quantum variant at the same seed — 4 of 60 cells in the quantum × classical-adversarial × seed grid violate "classical < quantum". The claim is correct at the **mean** level, not per-seed.

**Fix:** Insert "On per-model means," (or equivalent qualifier) before each occurrence. Mirror the LR-DTW hedging discipline.

### F-6: §4.1 "uniformly outperformed" elides ACF per-seed overlap

**Sections:** main lines 408–411 (§4.1 Cross-Model Comparison opening).

**Drift:** "Uniformly" suggests per-seed dominance everywhere on the conjoined LR-DTW + lag-1 ACF axis. On LR-DTW: true (60/60). On lag-1 ACF: per-seed overlap exists (wgan_lstm seed-46 = −0.0761 is closer to real than every quantum mean).

**Fix:** Rewrite to use the two-clause structure §1.4 bullet 3 and §4.1 close already use: "uniformly outperformed on LR-DTW (per-seed dominance); on lag-1 ACF the quantum-cluster mean is closer to the real reference than any classical-baseline mean (per-seed overlap noted)."

### F-7: §4.3 "uniformly outperform" — same elision as F-6

**Sections:** main line 673 (§4.3 Theoretical and Practical Implications).

**Fix:** Add one-clause qualifier: "(per-seed on LR-DTW; mean-level on lag-1 ACF closeness, with per-seed overlap noted)".

### F-8: "The only component that differs is the generator" — true for adversarial only

**Sections:** main lines 192–194 and 259–262.

**Code reality:** VAE uses single-Adam ELBO loop (no critic, no GP), AR(2) is fit via lstsq (no optimizer / no epochs). The claim is true within the WGAN-GP adversarial cohort; loose for VAE and AR(2).

**Fix:** Add parenthetical at lines 194 and 261: "(within the WGAN-GP adversarial cohort — IQP:SEL, V1/V2/V3, wgan_mlp/cnn/lstm; the VAE and AR(2) non-adversarial baselines use their own native training loops as described above/below)".

### F-9: Fig A8 caption omits the second data source actually rendered

**Sections:** supp line 609 (Fig A8 caption).

**Drift:** Caption attributes the entire 4-panel figure to `metrics.csv`, but the TSTR-lite R² panel actually renders from `tstr_lite.json` per sidecar metadata.

**Fix:** Append to caption: "; TSTR-lite R² panel from `revision/results/transform_ablation/tstr_lite.json` (per `summary.md` fallback)."

---

## §3 — NIT (optional polish; safe to ship without)

| # | Location | Description |
|---|---|---|
| N-1 | Abstract / §1.4 / §4.1 / §4.2 / §5 (all "Welch p > 0.36") | Floor is actually 0.36521 — tight, not loose. Optional: change to "Welch p ≥ 0.37". |
| N-2 | main line 522 | "LR-EMD ≈ 0.016 is in-cluster" — VAE is actually the worst of all 9. Optional hedge: "while the worst of the 9, still within 6% of the quantum cluster". |
| N-3 | main line 770 vs §1.3 line 92 | §5 references "the falsifiable question posed in §1.3" but §1.3 doesn't forward-reference §5. Reader navigation polish. |
| N-4 | main line 410 (§4.1) | "the three parameter-matched classical adversarial baselines and the AR(2) reference" omits VAE-exclusion forward reference. Optional: add "(the VAE is excluded as a degenerate generation regime; see characterization paragraph below)". |
| N-5 | supp Fig 3 caption (line 507) | ACF lag-1 caption uses −0.064 (rounded from JSON −0.06411). Consistent across paper — flagged only to confirm intentional rounding. |
| N-6 | Fig A7 PNG (`preprocessing_pipeline_4panel.png`) | Suptitle overlaps Panel 1 title in PNG preview; PDF (used by .tex) likely renders correctly. Optional re-render with `tight_layout()`. |
| N-7 | Fig A2 caption (supp line 157+) | "QGAN and synthetic data implementation within a hybrid-model approach" — hard-prohibition compliance via §A.3 prose, not in-caption. Optional defensive in-caption qualifier: "(proposed extension; not implemented or evaluated in this study — see §A.3)". |
| N-8 | supp line 531 (Fig A7 caption) | "Four-stage preprocessing pipeline" counts raw OD as a stage; main text correctly says 3 transformations. Optional reword. |
| N-9 | Fig A8 visual cosmetic | None of the 9 NITs blocks submission. |

---

## §4 — What was verified clean (sanity)

All four hard prohibitions hold (except the B-1 mechanism slippage — see §1):
- ✓ Lambert W only inside Pipeline C-dropped rationale (Pipeline B definitions are clean across main + supp + figures)
- ✓ LR-DTW correctly framed as the surviving quantum-distinguishing signal; LR-EMD asymmetry disclosed
- ✓ Real-data lag-1 ACF reference = −0.0641 (consistent across all citations)
- ✓ All 15 cross-checked code-vs-prose claims match (n=5 seeds, 2000 epochs, the 9 generators, all hyperparameters, parameter counts, TimeGAN protocol, 0.6843 historical disclosure)

All 11 figures resolve, all source-data attributions resolve to existing JSON files, no stale-figure regressions.

143 main + 156 supp numeric literals all trace to JSON (provenance gate PASS).

Per-seed LR-DTW 60/60 dominance verified, wgan_lstm seed-46 = −0.0761 ACF callout verified, Table 1 / Table 2 cells verified, ablation numbers verified, Orlandi DTW ratio verified.

---

## §5 — Recommended fix sequence

1. **B-1** (BLOCK): single rewrite at 4 sites — fix first
2. **F-1, F-2, F-3** (front-of-paper calibration): batch into one commit (Abstract + §1.4 + §5)
3. **F-4** (Pipeline C order): one commit across main + supp
4. **F-5, F-6, F-7** (per-seed vs mean hedging): batch into one commit (§4.1 + §4.3 + §4.4 + §5)
5. **F-8** (training-loop scope): one commit at lines 194 + 261
6. **F-9** (Fig A8 caption): one commit
7. **NITs**: optional follow-up, user choice
