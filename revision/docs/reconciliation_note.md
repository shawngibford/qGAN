# 1000ep -> 2000ep Reconciliation Note (D-14-13)

> **Generated** by `revision/run_model_info.py` — every number below is READ from a `revision/results/*.json` artifact, never recomputed or hand-typed (D-14-16, Pitfall 5).

This note is the authoritative record of every headline metric that changed when the budget moved from the unfair 1000ep / 75-param regime to the matched 2000ep / 55-param regime (Tier-2/3 of D-14-22). The **OLD** column is the frozen Phase-10 `baseline_comparison.json` (1000-epoch budget); the **NEW** column is the accepted 2000-epoch `matched2000` sweep. Any manuscript number that moved between submission and resubmission MUST cite this delta.

`data_hash` = `91e447d4624e25b3` — identical across every consumed artifact (cross-artifact explicit-raise gate, run_multiseed_rollup.py:86-92 idiom).

## EMD (OD scale) — final-eval mean over seeds 42-46

| model | old (1000ep) | new (2000ep) | delta | old basis | new basis |
|---|---|---|---|---|---|
| iqp_sel_55_repro | 0.027586 | 0.154999 | +0.127413 | baseline_comparison.json rows[] (model_kind=quantum, pipeline=B, emd, OD) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| V1 | — | 0.155376 | — | no 1000ep matched-budget counterpart (ansatz variant introduced at 2000ep, D-14-10) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| V2 | — | 0.156328 | — | no 1000ep matched-budget counterpart (ansatz variant introduced at 2000ep, D-14-10) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| V3 | — | 0.148114 | — | no 1000ep matched-budget counterpart (ansatz variant introduced at 2000ep, D-14-10) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| wgan_mlp | 0.027580 | 0.121527 | +0.093946 | baseline_comparison.json rows[] (model_kind=wgan_mlp, pipeline=B, emd, OD) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| wgan_cnn | 0.113033 | 0.101747 | -0.011286 | baseline_comparison.json rows[] (model_kind=wgan_cnn, pipeline=B, emd, OD) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| wgan_lstm | 0.029258 | 0.146192 | +0.116935 | baseline_comparison.json rows[] (model_kind=wgan_lstm, pipeline=B, emd, OD) | matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], mean over seeds 42-46 |
| vae | 0.025740 | — | — | baseline_comparison.json rows[] (model_kind=vae, pipeline=B, emd, OD) | no 2000ep EMD trajectory — non-adversarial baseline tracks ELBO/closed-form fit, not adversarial EMD (metrics.json carries no emd_avg; recompute forbidden, D-14-16) |
| ar | 0.029084 | — | — | baseline_comparison.json rows[] (model_kind=ar, pipeline=B, emd, OD) | no 2000ep EMD trajectory — non-adversarial baseline tracks ELBO/closed-form fit, not adversarial EMD (metrics.json carries no emd_avg; recompute forbidden, D-14-16) |

**Interpretation.** A negative delta means the matched 2000ep budget *improved* (lowered) the EMD relative to the unfair 1000ep baseline. The ansatz variants (V1/V2/V3) have no 1000ep matched-budget counterpart — they were introduced directly at the 2000ep budget (D-14-10) — so their OLD column is intentionally blank rather than carrying a non-comparable number.

**Integration caveat (Plan 14-12, recorded post-14-09/14-10).** Two facets of the table above are now backed by additional audited artifacts: (1) the V1/V2/V3 row param-count values (75 / 135 / 75) now resolve directly to `revision/results/v1_config_lock.json`, `revision/results/v2_config_lock.json`, and `revision/results/v3_config_lock.json` (Plan 14-09 — `gate_layout_breakdown` field decomposes each count as IQP encoding (5) + N\*SEL layers (15 each) + final RX+RY (10)), rather than only indirectly through the `_QUANTUM_ANSATZ` dict at `revision/run_matched2000.py:118-122`; (2) the D-14-10 headline-vs-repro distinction (iqp_sel_55_headline as the frozen-checkpoint EMD, iqp_sel_55_repro as the matched-2000ep reproduction) is now visualized as two distinct points in `revision/results/figures/param_efficiency_pareto.{png,pdf}` (Plan 14-10 — the headline appears as a separate dashed/diamond marker per the conflation-guard contract).
