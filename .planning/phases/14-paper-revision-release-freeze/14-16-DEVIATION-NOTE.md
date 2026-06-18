# Plan 14-16 — Deviation Note (executor checkpoint + inline-execution findings)

This note records three empirical findings surfaced during 14-16 execution
where the corrected data contradicted predictions inherited from the r3
forensic synthesis. All three are benign for the manuscript's surviving
claims; all are documented in `docs/peer_review_remediation.md`'s
Plan 14-16 section.

## Finding 1 — LR-EMD ranking inversion → Path A reframe (executor checkpoint)

The plan's load-bearing strong claim asserted "55 quantum params … and
significantly beats every WGAN on log-return EMD (p ≤ 0.014, d ≤ -2.6)".

T1 implemented the R3-CR-2 fix correctly (un-standardize-fake per
`pipeline-review-r3.md` §2). The corrected log-return EMD aggregates match
the §2 anchor table exactly:

| model | corrected LR-EMD |
|---|---|
| ar | 0.00294 |
| wgan_cnn | 0.00711 |
| wgan_mlp | 0.01031 |
| wgan_lstm | 0.01272 |
| V3 | 0.01432 |
| iqp_sel_55_repro / V1 | 0.01497 |
| V2 | 0.01502 |
| vae | 0.01583 |

**Every WGAN beats every quantum on corrected LR-EMD.** The Welch tests in
`statistical-honesty-r3.md` §3b were computed on the broken
(scale-mismatched) LR-EMD column — they do not survive the fix. The
"quantum beats WGAN on LR-EMD" claim is empirically false on honest data.

**Resolution.** User selected **Path A**: withdraw the LR-EMD-vs-WGAN
claim; retain the OD-EMD parametric-efficiency equivalence (OD column
byte-identical pre/post — Welch p > 0.36, |d| ≤ 0.65, n=5) and the DTW
dominance (independent metric, byte-stable). The R3-CR-2 fix itself is
correct; only the plan's prediction about what the corrected metric would
show was wrong.

## Finding 2 — R3-CR-1 numerically inert

The R3-CR-1 fix (`density=True` → `density=False` + shared edges +
total-mass normalization) produces **byte-identical** v1→v2 OD-scale EMD
values (delta 0.00000 for all 9 models). `scipy.stats.wasserstein_distance`
renormalizes its weight arguments internally, so the density-normalization
distinction is numerically inert when both histograms share edges. The
synthesis's CRITICAL severity for R3-CR-1 was overstated. The fix is still
landed — it adds the genuine `fake_in_range_mass` disclosure stat and
bundles the real R3-HI-1 sister-fix — but it does not change the OD-scale
EMD column.

## Finding 3 — cross_model_emd is OD-only

T6's plan assumed `cross_model_emd` carries a log-return bar group. The
figure's companion JSON carries only OD-scale fields; since T1 left the OD
column byte-stable, the figure required no re-render. Confirmed byte-stable.

## Net effect on manuscript claims

None of the three findings changes the manuscript's surviving claims:
- OD-EMD parametric-efficiency equivalence — intact (byte-stable OD column).
- DTW dominance + ~6.5x Orlandi improvement — intact (independent metric).
- The withdrawn claim (LR-EMD beats WGANs) was never in the manuscript —
  it was an r3-synthesis proposal; the retraction is documented as a
  forensic-process win, not a regression.
