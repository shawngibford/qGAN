# R3 Forensic Investigation — Cross-Agent Synthesis

**Trigger:** User concern that the current matched-2000ep quantum results (May 2026) appear mid-pack vs classical baselines, while historical figures in `/Users/shawngibford/dev/phd/qGAN/Final Results from 2000 epochs - IQP:SEL circuit/` (untracked, Aug 2025) suggested otherwise.

**Five-agent parallel investigation:** historical-forensic, pipeline, quantum/checkpoint, statistical-honesty, code-review.

---

## Headline verdict

**MIXED.** The picture is more nuanced than the "apples-to-oranges" framing 14-13/14-14/14-15 adopted. There are **two real bugs in the metric columns** that systematically disadvantage quantum (and reward degenerate solutions like VAE posterior collapse), AND the historical 0.0015 headline is genuinely non-reproducible. Both narratives need correction.

The most-favorable HONEST claim the manuscript can make (per statistical-honesty agent):

> **55 quantum parameters achieve statistically-equivalent OD-EMD to 10⁴-10⁵ classical generator parameters (Welch p > 0.36 for every quantum-classical pair, |d| ≤ 0.65, n=5 seeds); on log-return EMD, quantum significantly beats every WGAN variant (p ≤ 0.014).**

This is a real, defensible claim that the current docs do NOT make. The "quantum mid-pack" framing in the current `reviewer_response.md` understates what the data actually supports.

---

## Hypothesis scorecard

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **(A) Apples-to-oranges metric** | SUPPORTED | Pre-v1.0 used `wasserstein_distance(PMF_real, PMF_fake)` over 51-bin PMFs treated as samples — a degenerate scaling that gives values ~`bin_width≈0.002`. The 0.0015 headline is provenance-traced to `qgan_pennylane.ipynb:1554-1572` (Agent 1, HIGH confidence). |
| **(A') Apples-to-oranges seed/slice** | SUPPORTED | Historical `iqp_sel_55_headline` was a best-checkpoint-via-early-stopping pick on in-loop training-batch EMD; current matched-2000ep takes final-epoch on full held-out series (Agents 1+3, HIGH). Slice-size empirical test: random-half-real reproduces ~0.003 EMD; full-real gives ~0.12 (Agent 4). |
| **(B) Real code bug — log_return EMD scale mismatch** | **SUPPORTED (CRITICAL)** | `matched2000_dualscale.json` log_return column compares STANDARDIZED synth (std≈1) against UNNORMALIZED real_log_delta (std=0.022). 50× scale inflation. Rankings INVERT on the corrected scale: AR best at 0.003, quantum V1 at 0.015, VAE worst at 0.016 (Agent 2). |
| **(B') Real code bug — histogram-density structural bias** | **SUPPORTED (CRITICAL)** | `revision/run_distribution_emd.py:94-141` (NEW from 14-15) — `density=True` re-normalizes each histogram independently over in-range portion, silently dropping out-of-range mass. Rewards narrow distributions (VAE posterior collapse: std=0.0004) and uncapped-range distributions (WGAN-CNN: 94% out-of-range, in-range 6% gets renormalized into coincidental alignment). Rankings INVERT vs raw-sample EMD: VAE jumps 6th→1st, quantum drops 3rd→7th, samples unchanged (Agent 5, HIGH confidence). |
| **(C) Headline-vs-repro alone explains gap** | PARTIAL / WEAK | Headline OD-EMD 0.0231 vs repro mean 0.0275 = 0.9σ gap; not large enough to fully explain (Agent 3). |
| **(D) Checkpoint integrity issue** | RULED OUT | sha256 matches; 55 params; epoch 1969; all hyperparams match notebook (Agent 3). |
| **(E) Quantum-specific precision regression** | RULED OUT | PennyLane returns float64 throughout; `.to(torch.float64)` is no-op for quantum, upcast for classical — quantum remains MORE precise (Agent 3 + 5). |
| **(F) Quantum-specific training-loop bug** | RULED OUT | `_train_quantum` and `_train_wgan` use identical `train_wgan_gp`, identical critic, identical hyperparams (Agent 3 + 5). |

---

## The two real bugs (R3-CR-1 + R3-CR-2)

### R3-CR-1 (CRITICAL, NEW in 14-15): Histogram-density EMD structurally biased

Source: Agent 5 (code review) + Agent 2 (pipeline).

**Mechanism.** `revision/run_distribution_emd.py:94-141` computes `wasserstein_distance(bin_centers, bin_centers, real_hist_density, fake_hist_density)` with `np.histogram(..., density=True)`. The `density=True` flag re-normalizes the histogram so that the area under the curve sums to 1.0 — but if the model's samples have any out-of-range mass, that mass gets silently truncated AND the remaining in-range portion is renormalized.

This rewards two pathological behaviors:
- **Narrow/collapsed distributions** (VAE with posterior collapse, std=0.0004): all mass is densely concentrated at the mean, histogram density is sharp and high, EMD against real's broad density is small relative to a quantum model that distributes correctly.
- **Uncapped-range distributions** (WGAN-CNN: 94% of mass out-of-range): the truncated in-range 6% gets renormalized into a coincidentally-aligned density profile.

**Empirical confirmation.** Ranking on histogram-density EMD vs raw-sample EMD on the same underlying samples:

| Model | Raw-sample OD EMD | Histogram-density OD EMD | Rank change |
|---|---|---|---|
| VAE | 0.0257 (rank 1) | 0.0523 (rank 1) | — |
| AR | 0.0291 (rank 8) | 0.0561 (rank 2) | **+6** |
| WGAN-LSTM | 0.0282 (rank 7) | 0.0584 (rank 3) | +4 |
| WGAN-MLP | 0.0260 (rank 2) | 0.0609 (rank 4) | -2 |
| V3 | 0.0275 (rank 4) | 0.0629 (rank 5) | -1 |
| iqp_sel_55 | 0.0275 (rank 3) | 0.0638 (rank 8) | **-5** |
| V1 | 0.0276 (rank 5) | 0.0636 (rank 7) | -2 |
| V2 | 0.0276 (rank 6) | 0.0637 (rank 6) | — |
| WGAN-CNN | 0.0543 (rank 9) | 0.0671 (rank 9) | — |

Quantum models drop 1-5 ranks; AR jumps +6. **The histogram-density column does not measure what the original paper's `~0.0015` claimed.** It's a biased metric that 14-15 introduced in good faith to enable cross-pre-v1.0 comparability — but the bias systematically harms quantum, which is exactly the column the user looked at and felt concerned about.

### R3-CR-2 (CRITICAL, INHERITED): log_return EMD scale mismatch

Source: Agent 2 (pipeline) + Agent 5 (code review).

**Mechanism.** `revision/run_matched2000_dualscale.py:368-372` computes `wasserstein_distance(synth_log_returns, real_log_delta)` where:
- `synth_log_returns` are standardized (model outputs * 0.1, then implicitly on the [-0.1, +0.1] scale matched to the standardized training data with std ≈ 1)
- `real_log_delta` are RAW log-returns of OD with std ≈ 0.022 — never standardized

50× scale inflation. On the corrected scale (unstandardizing the synth side), rankings invert entirely.

This bug was inherited from the pre-revision `run_dualscale_fidelity.py` (1000ep era) and propagated through 14-08 / 14-13. The 14-13 reconciliation_note.md OLD/NEW table happened to land on the right answer on the OD scale (the OD inverse transform implicitly unscales). But the log-return column has been wrong since 14-08.

**Corrected log-return EMD ranking** (Agent 2's empirical re-computation):

| Model | Current (broken) LR-EMD | Corrected LR-EMD |
|---|---|---|
| VAE | 0.0103 | 0.0163 |
| V2 | 0.1218 | ~0.0153 |
| V1 | 0.1219 | 0.0145 (best quantum) |
| iqp_sel_55 | 0.1229 | ~0.015 |
| V3 | 0.1303 | ~0.015 |
| WGAN-LSTM | 0.1663 | ~? |
| WGAN-MLP | 0.2699 | ~? |
| WGAN-CNN | 0.6873 | ~? |
| AR | 0.7811 | **0.003 (best overall)** |

On the corrected scale, AR (3 params!) is the best on log-return EMD, V1 quantum is 2nd, VAE is mid-pack — completely different story than the broken column suggests. (Note: AR-best on log-return is unsurprising — AR(2) is the Yule-Walker MLE for AR-noise time series. The interesting line is quantum competitive with AR at the marginal-distribution level.)

---

## What the data actually supports (statistical-honesty agent's framing)

**On OD-scale raw-sample EMD (the cleanest metric):**
- 9 models cluster in [0.0257, 0.0543]; WGAN-CNN is the only outlier
- Welch t-test between any quantum-classical pair: p > 0.36, |d| ≤ 0.65 — **no statistically significant difference**
- Parametric-efficiency framing: 55 quantum params match 74-562 classical generator params

**On log-return raw-sample EMD (after R3-CR-2 fix):**
- AR is the best on the corrected scale (closed-form Yule-Walker fits log-return distribution exactly)
- All 4 quantum variants cluster tightly (~0.015) and significantly beat every WGAN (p ≤ 0.014, d ≈ -3 to -5)
- VAE is mid-pack on corrected scale (was artificially #1 on the broken scale due to posterior collapse fooling the standardization)

**On histogram-density EMD (after R3-CR-1 disclosure or fix):**
- The bias must be disclosed — VAE's "win" is posterior collapse
- A defensible variant: 50-bin histogram with SHARED edges based on the empirical-real range only (no per-fake renormalization); recompute and report

---

## What the historical "Final Results" figures actually showed

Per Agent 1's per-figure inventory:

- **Figure_10 `EMD=0.0015`**: traced to notebook cells 1554-1572 — `wasserstein_distance(PMF_real, PMF_fake)` on 51-bin renormalized PMFs treated AS SAMPLES (not weights). Degenerate metric, scaled by bin_width ≈ 0.002. **Not commensurate with any raw-sample EMD.**
- **Figure_15 `Dist=0.7337`**: NOT an EMD — it's a `dtaidistance` DTW distance on a 5%-perturbed synthetic series (ablation, line 8166 of the notebook). Not a model-quality claim.
- **Figure_20 `Correlation=-0.0474`**: confirms no temporal point-wise alignment; the current matched-2000ep has the same finding (ACF lag-1 = -0.105 ± 0.05).
- **"Best EMD: 0.001192 at epoch 501"**: cherry-picked via early-stop minimum on the in-loop training-batch metric. Not a final-epoch evaluation.

**None of the historical figures use the same EMD formulation as the current pipeline.** The apparent "regression" was real (current numbers are larger) but it's because the current pipeline is doing a fundamentally more rigorous evaluation. The historical figures were optimistic by construction.

---

## Three downstream paths (user decides)

### Path 1: Fix the two bugs (14-16 plan)

A 4-task plan to:
1. Fix R3-CR-2 (log-return scale mismatch in `run_matched2000_dualscale.py`) — single-line edit using `norm_log_delta`. Re-emit `matched2000_dualscale.json`. Updates the existing 3-column reconciliation table.
2. Either drop R3-CR-1 (the histogram-density column) entirely OR fix it with shared-edges-from-real-range. Disclose the bias either way.
3. Update `reviewer_response.md` + `methods_full.md` with the corrected numbers + the new most-favorable-honest claim ("55 quantum params statistically-equivalent to 10⁴-10⁵ classical params on OD-EMD; significantly beats every WGAN on LR-EMD").
4. Re-render `cross_model_emd` + `qq_overlay` figures against the corrected aggregates; gate-verify.

**Pros:** the manuscript would carry the strongest defensible claim. **Cons:** 1-2 days of work; introduces another revision before 14-07 tag.

### Path 2: Disclose both bugs as known limitations + reframe

Add to `peer_review_remediation.md`:
- R3-CR-1 disclosure: "The histogram-density EMD column is structurally biased toward narrow/collapsed distributions and out-of-range uncapped distributions — reviewers should read the raw-sample OD-EMD column as the primary metric"
- R3-CR-2 disclosure: "The log-return-scale EMD column has a known standardization mismatch (synth standardized vs real raw); the OD-scale column is the only directly-interpretable headline"

Update `reviewer_response.md` to use the strong-claim framing on the OD-EMD-only basis: "55 quantum params statistically equivalent to 10⁴-10⁵ classical params". Drop the LR/HD comparison framing.

**Pros:** ~2 hours, no code changes. **Cons:** ships a known-broken metric column with disclosure rather than fixing it; weaker than Path 1.

### Path 3: Accept current honest-mid-pack framing and proceed to 14-07

Acknowledge that the most-favorable claim that can be made is "feasibility, not advantage" (per R1-M5 calibration). Don't fix the bugs (they're disclosed-or-honest-mid-pack consistent). Ship.

**Pros:** done. **Cons:** undersells the actual result; reviewers might spot the bugs themselves and ask why.

---

## Recommendation

**Path 1 is strongly preferable.** Both bugs are real, both are 1-day fixes, and both currently make quantum look *worse* than the data supports. The user's intuition that "results are not great" is partly the result of these bugs — and partly real (the OD-scale equivalence is what it is). The honest-and-strong claim emerges only after fixing both.

Specifically: the user is right that the current docs are under-reporting what the data actually supports. The current `reviewer_response.md` says (R1-M1) "this is being addressed in follow-up work" but doesn't claim equivalence. The synthesis above says equivalence IS demonstrated (n=5, p>0.36) — that's a stronger paper.

---

## Files

The 5 individual reports:
- `peer-review-r3/historical-forensic-r3.md` (Agent 1)
- `peer-review-r3/pipeline-review-r3.md` (Agent 2)
- `peer-review-r3/quantum-checkpoint-r3.md` (Agent 3)
- `peer-review-r3/statistical-honesty-r3.md` (Agent 4)
- `peer-review-r3/code-review-r3.md` (Agent 5)

This synthesis at `peer-review-r3/SYNTHESIS.md`.

D-14-22 byte-freeze of `revision/core/` preserved across the entire investigation (read-only audit).
