# Provenance / Numerical-Claims Peer Review

**Reviewer role:** numerical-claims / provenance auditor (independent of the 14-12 executor).
**Method:** ran `scripts/verify_number_provenance.py` end-to-end on all 9 paper-facing docs; spot-checked 10 random load-bearing literals against their claimed source JSONs; audited gate internals (`_NUM`, `_ID_PATTERNS`, `_ALLOW`, `_json_blobs` rglob); checked `data_hash` consistency across paper-facing JSONs; checked cross-artifact field consistency; checked headline-vs-repro conflation; checked R²<0 honesty; investigated the reconciliation_note.md failure.

**Bottom line.** The provenance gate is *mostly* faithful and the high-volume paper-facing numbers (methods_full, training_protocol, dataset_stats, circuit_atlas, paper_blocks_framing/refs_methods, reviewer_response) resolve correctly to source JSON fields at the right precision. **However, three substantive issues exist:** (1) `reconciliation_note.md` mixes scales — its "NEW (2000ep)" column is on a different scale than its "OLD (1000ep)" column, making every delta in that table meaningless; (2) the gate has at least two latent false-positive resolution paths that already triggered on paper-facing literals; (3) data_hash invariant is enforced inconsistently — most paper-facing JSONs lack the field, including all five config-lock JSONs and every figure companion JSON.

---

## §1 — Gate end-to-end results

| Doc | Status | Distinct literals | Failed literals |
|---|---|---|---|
| `docs/paper_blocks_framing.md` | PASS | 24 | — |
| `docs/paper_blocks_refs_methods.md` | PASS | 93 | — |
| `docs/reviewer_response.md` | PASS | 45 | — |
| `docs/reconciliation_note.md` | **FAIL** | — | `+0.127413`, `-0.011286` |
| `docs/methods_full.md` | PASS | 57 | — |
| `docs/circuit_atlas.md` | PASS | 18 | — |
| `docs/completeness_sweep_manifest.md` | PASS | 8 | — |
| `docs/training_protocol.md` | PASS | 17 | — |
| `docs/dataset_stats.md` | PASS | 5 | — |

Notable deviation from the 14-12 executor's report: only **2** literals fail in `reconciliation_note.md`, not 3. The third (`+0.116935`) silently resolves via a **coincidental float-precision match** to an unrelated value in `baselines/runs/vae/A/44/metrics.json` (the value `0.11693459987873212` rounds to `0.116935` at 6 dp). That is a **false positive**, not a true resolution — see §4-FP1.

---

## §2 — Investigation of `reconciliation_note.md` failures

### §2.a — Arithmetic check of the four delta literals

| Row | new | old | doc delta | computed delta | arithmetic correct? |
|---|---|---|---|---|---|
| iqp_sel_55_repro | 0.154999 | 0.027586 | `+0.127413` | `0.127413` | yes |
| wgan_mlp | 0.121527 | 0.027580 | `+0.093946` | `0.093947` | **no — off by 1 in last digit** |
| wgan_cnn | 0.101747 | 0.113033 | `-0.011286` | `-0.011286` | yes |
| wgan_lstm | 0.146192 | 0.029258 | `+0.116935` | `0.116934` | **no — off by 1 in last digit** |

Two of the four deltas are rounded inconsistently with the subtraction (last-digit rounding). The other two are correct as subtractions. None is stored as a field in any audited JSON; they are derived quantities.

### §2.b — Gate failure mode

The gate (`scripts/verify_number_provenance.py`) doesn't compute or recognize arithmetic derivations — it only checks substring-presence or float-precision presence in the JSON corpus. Two of these derived deltas (`+0.116935`, `+0.093946`) silently **pass via the float-precision path**, because they happen to round-match unrelated numbers (`0.116934599…` and `0.093946342…`) elsewhere in the corpus. The two that fail (`+0.127413`, `-0.011286`) are unlucky enough to have no coincidental match.

### §2.c — Resolution options

Both are arithmetically correct subtractions (modulo the last-digit rounding for the wgan_mlp and wgan_lstm cells — see §3-CRIT-1 for a deeper concern). Two routes:

- **(a) Add audited delta fields.** Emit a new JSON `results/reconciliation_deltas.json` with explicit `delta` fields computed from the same source numbers; the gate would then resolve them by substring match. Re-run `scripts/run_model_info.py` or a dedicated `scripts/run_reconciliation.py` emitter. This is the cleaner option but adds an artifact.
- **(b) Document the limitation.** Leave as-is; record that the gate cannot recognize arithmetic derivations and the reconciliation_note.md FAIL is expected. This is what `completeness_sweep_manifest.md:105` already does.

**Recommendation:** Option (b) does NOT discharge the deeper §3-CRIT-1 problem (the reconciliation table is wrong-scale on the NEW column). Fix the table first; the gate failure will resolve itself once the table is restated correctly against either `matched2000_dualscale.json` aggregates or the (existing) `multiseed_summary.json` rollup mechanism.

---

## §3 — Findings

### CRITICAL — CRIT-1 — `reconciliation_note.md` mixes scales in the headline EMD table

**Where.** `docs/reconciliation_note.md:9-22` — the "EMD (OD scale) — final-eval mean over seeds 42-46" table.

**Evidence.** The header asserts both columns are "EMD (OD scale)". The OLD column basis is `baseline_comparison.json rows[] (pipeline=B, emd, OD)` — that is OD-scale. The NEW column basis is `matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1]`. But `core/training.py:415-423` shows `emd_avg` is computed from `real_log_returns` — i.e. it is the **log-return-scale** training-trace EMD at the final epoch, NOT the OD-scale final-eval EMD.

**Consequence.** Every delta in the NEW column is the difference of OD-scale (OLD) minus log-return-scale (NEW). The deltas are scale-mixed, not arithmetically meaningful.

**Concrete numbers.** The actual OD-scale matched-2000ep aggregate means already exist in `results/matched2000_dualscale.json` under `aggregates[*, emd, OD].mean`:

| Model | OLD (1000ep OD) | NEW per doc (log-return!) | NEW correct (2000ep OD) | Correct delta |
|---|---|---|---|---|
| iqp_sel_55_repro | 0.027586 | 0.154999 ❌ | **0.027526** | -0.000060 |
| wgan_mlp | 0.027580 | 0.121527 ❌ | **0.025952** | -0.001628 |
| wgan_cnn | 0.113033 | 0.101747 ❌ | **0.054323** | -0.058710 |
| wgan_lstm | 0.029258 | 0.146192 ❌ | **0.028214** | -0.001044 |

The doc-reported deltas (~+0.09 to +0.13 worse) misrepresent the matched-budget result: in reality the matched 2000ep budget makes the OD-scale EMD essentially flat or substantially better (wgan_cnn improves by 52%). The "**Interpretation**" sentence in `reconciliation_note.md:23` ("A negative delta means the matched 2000ep budget *improved* (lowered) the EMD") then reads as if the matched budget mostly *worsened* fidelity, which is the opposite of what the audited data say.

**Downstream impact.** `paper_blocks_framing.md:13-15` and `paper_blocks_framing.md:401-403` cite `reconciliation_note.md` as the authoritative basis for the LOCKED de-overclaim direction (PAPER-02). The LOCKED de-overclaim is the right direction (parameter-matched comparison does not show a clear quantum advantage), but the *quantitative argument* in reconciliation_note is currently wrong-scale and would not survive a reviewer reading the table next to `matched2000_dualscale.json`.

**Recommendation.** Restate the NEW column using `matched2000_dualscale.json` `aggregates[*, emd, OD].mean` so both columns are unambiguously OD-scale. The values are already stored as audited mean fields, so the rewritten table will pass the provenance gate cleanly without any new JSON.

### CRITICAL — CRIT-2 — `paper_blocks_framing.md:520` justifies a phantom resolution for `0.6843`

**Where.** `paper_blocks_framing.md:119` — the BEFORE block quotes the manuscript: "The QWGAN-GP achieved a Dynamic Time Warping (DTW) score of 0.6843, representing improved temporal alignment compared to previously reported methods." The provenance footer at `:520` states `0.6843` resolves "as a substring of the frozen `results/*.json` artifacts where they coincide".

**Evidence.** `0.6843` does not appear as a literal field in any JSON. It only "resolves" because the longer float `0.03006843578171075` (an EMD value in `baseline_comparison.json` / `fidelity_dualscale.json` / `baseline_classical_wgan.json`) **contains** the substring `0.6843`. This is a **substring false positive**, not an actual data source. The number `0.6843` is the manuscript's claimed DTW score; the only place it is sourced from is the original Overleaf manuscript prose, and the reviewer would correctly demand: "show me the JSON field that equals 0.6843."

**Consequence.** The framing doc's PASS status is partly a lie — one of the 24 distinct literals is a phantom resolution. The footer at `:520` openly admits this by hand-waving "substring of frozen ... where they coincide", which is a polite name for the bug.

**Recommendation.** Either (a) delete `0.6843` from the BEFORE quotation (it's only there to be replaced by the AFTER block anyway, and the AFTER block doesn't contain it), or (b) re-emit a DTW field in a new audited JSON that genuinely stores the 0.6843 quantity (only worthwhile if 0.6843 is actually trustworthy, which is unclear), or (c) tighten the gate so substring resolution requires the matched substring to be a complete numeric token (see §4-GATE-1).

### HIGH — HIGH-1 — Gate's float-precision path admits coincidental matches (false positives)

**Where.** `verify_number_provenance.py:110-142` — `_resolves()`.

**Evidence (already triggered).** Two delta literals in `reconciliation_note.md` were silently passed by the gate via this path:
- `+0.116935` ↛ no true source; "resolves" against `baselines/runs/vae/A/44/metrics.json` value `0.11693459987873212` at 6-dp.
- `+0.093946` ↛ no true source; "resolves" against `augmentation.json` value `0.09394634266694386` at 6-dp.

These have nothing to do with the deltas the doc is claiming. The gate accepted them because at 6-dp precision they match unrelated quantities in unrelated files.

**Consequence.** A doc author who hand-types a wrong number has a non-zero probability of being silently approved by the gate, particularly for 6-dp literals where the JSON corpus is dense.

**Recommendation.** Two non-exclusive mitigations:
- Require the resolving JSON path's context to be semantically related (e.g., a sibling/parent key naming the same metric/model), not just any numeric coincidence in the corpus.
- Print the resolution path next to each literal at PASS-time so a human reviewer can spot semantic mismatches; today the PASS message only counts literals.

### HIGH — HIGH-2 — `data_hash` invariant enforced inconsistently across paper-facing JSONs

**Where.** All five config-lock JSONs (`canonical_config_lock.json`, `default_75_config_lock.json`, `v1_config_lock.json`, `v2_config_lock.json`, `v3_config_lock.json`); `classical_architectures.json`; `framework_versions.json`; `noise_model_sensitivity.json`; `shot_noise_sensitivity.json`; `ansatz_comparison.json`; `parity_check.json`; `eval06_roundtrip.json`; `canonical_recovery.json`; every figure companion JSON under `figures/*.json` (≥75 files); `matched2000/sweep_status.json`; `transform_ablation/*.json` — none of these carry `data_hash: 91e447d4624e25b3`.

**Evidence.** Among 12 top-level JSONs that do carry a `data_hash`, every one of them is `91e447d4624e25b3` (consistent across `model_info.json`, `methods_full.json`, `headline_canonical.json`, `matched2000_dualscale.json`, `baseline_comparison.json`, `baseline_classical_wgan.json`, `baseline_nonadversarial.json`, `fidelity_dualscale.json`, `multiseed_summary.json`, `tstr.json`, `predictive_discriminative.json`, `augmentation.json`).

**Consequence.** Paper-facing JSONs the manuscript cites (config locks, classical_architectures, framework_versions, sensitivity sweeps, all figure companions) cannot be cross-checked against the data-hash invariant. A reviewer would reasonably ask "if this is the audited corpus, why don't all artifacts carry the invariant?" The 14-12 / phase-14 architecture documents claim this is the cross-artifact gate, but it only gates 12 of ~120+ paper-facing JSONs.

**Recommendation.** Either extend the relevant emitters to embed `data_hash` in the config-lock JSONs and the figure companions (low cost — they all derive from the same OD data), or change the contract language so `data_hash` is only claimed for the 12 aggregate metrics JSONs.

### HIGH — HIGH-3 — Doc-doc inconsistency on the param dtype

**Where.** `training_protocol.md:34` says **"Param dtype | torch.float64"** sourced from `model_info.json` `models[].dtype`. `methods_full.md:262` says **"Param dtype (`dtype_params`) | torch.float32"** sourced from `methods_full.json` `4_hardware_software.dtype_params`. Both cite `model_info.json` / `methods_full.json` as authoritative.

**Evidence.** `model_info.json#models[iqp_sel_55_repro].dtype` = `"torch.float64"`. `methods_full.json#buckets.4_hardware_software.dtype_params` = `"torch.float32 (classical: nn.Parameter constructed with dtype=torch.float32 — core/models/classical.py:78; quantum: params_pqc nn.Parameter dtype=torch.float32)"`. `methods_full.md:381-403` correctly distinguishes `dtype_params` (float32, the trainable parameters) from `dtype_samples` (float64, samples cast to match the float64 critic) and even calls out at `:403` that a previous doc conflated them.

**Consequence.** `training_protocol.md` is the doc that conflates them — its row 34 attributes "Param dtype" to `model_info.json#dtype`, but `model_info.json#dtype` is actually the sample dtype (the conflated field) and the genuine param dtype is float32 per `classical.py:78` / the quantum nn.Parameter.

**Recommendation.** Fix `training_protocol.md:34` to either split into the two rows (`dtype_params` = float32, `dtype_samples` = float64) matching `methods_full.md:262-263`, or remove the row and reference methods_full.md §4.b. Also consider renaming the `model_info.json#dtype` field to `dtype_samples` so the source-of-truth is unambiguous.

### MEDIUM — MED-1 — Gate's `_ID_PATTERNS` over-strip: 4-digit-in-parens

**Where.** `verify_number_provenance.py:73` — `r"\b\d{4}\b(?=\s*\))"` (intended for citation years like "(Gulrajani 2017)").

**Evidence.** The pattern strips **any** 4-digit number followed by `)`, not just citation years in the 19xx/20xx range. `(epoch 1969)`, `(frozen checkpoint epoch 1969)`, `(2000)`, `(384)` would all be stripped. Currently 3 of 6 `1969` occurrences in the docs are inside parens and would be silently dropped by the strip. The remaining 3 are in non-paren positions (table cells, prose) so the literal is still checked against canonical_config_lock.json#checkpoint_epoch=1969 and resolves. No active failure today.

**Consequence.** Latent vulnerability — a future doc emitter could put a 4-digit data quantity in parens (e.g. `(2000 epochs)`, `(1234 windows)`) and silently bypass the gate.

**Recommendation.** Tighten the pattern to `\b(?:19|20)\d{2}\b(?=\s*\))` (citation years only), or replace with an inline allowlist of bibliography-style contexts.

### MEDIUM — MED-2 — Gate's `_ID_PATTERNS` over-strip: `:\d+(?:-\d+)?\b`

**Where.** `verify_number_provenance.py:72` — `r":\d+(?:-\d+)?\b"`.

**Evidence.** This pattern strips bare line-range citations like `:255-258`, but it also strips colon-prefixed data values. For example, `pipeline:42` (legitimately representing seed 42 on Pipeline B) → `pipeline ` (numeric content removed before extraction). Today no doc uses `:NN` to denote data; the gate cannot detect this either way.

**Consequence.** Latent vulnerability. Adding new emitters that use `key:NN` shorthand would silently bypass the gate.

**Recommendation.** Anchor the pattern to file/line contexts only, e.g. `(?:\.py|\.md|\.json|\.tex):\d+(?:-\d+)?\b`.

### MEDIUM — MED-3 — Render-only companion JSONs are pooled with source JSONs in the resolution corpus

**Where.** `verify_number_provenance.py:99` — `RESULTS.rglob("*.json")`.

**Evidence.** ≥75 companion JSONs under `figures/` (e.g. `headline_vs_reproduction.json`, `param_efficiency_pareto.json`, `tstr_crossmodel.json`, `seed_variance_per_model.json`) are pure render artifacts. They carry `"render_only": true` and reference their source artifacts under `source_artifact` / `source_artifacts`. The gate treats them identically to source JSONs. In §4-FP2 above I verified `seed_variance_per_model.json#per_model_final_emd_mean[iqp_sel_55_repro] = 0.15499896082475875` resolves the `0.154999` literal — and this **is** the actual source for the (wrong-scale) NEW column in reconciliation_note.md.

**Consequence.** Two failure modes:
- A doc literal can "resolve" against a render-only companion that itself derived the number elsewhere — circular reference. Today the docs don't appear to exploit this circularly, but the protection is purely good behavior, not enforcement.
- The render-only companion may *contain hand-typed values* a generator script put there. If the same doc author wrote both the doc and the companion JSON, the gate cannot tell them apart.

**Recommendation.** Either (a) exclude `figures/*.json` from the resolution corpus and require literals to resolve to an audited (non-render-only) source, or (b) skip JSONs that declare `"render_only": true`. (a) is the stricter contract.

### MEDIUM — MED-4 — `matched2000_dualscale.json` aggregates carry `n: None`

**Where.** `results/matched2000_dualscale.json` — every aggregate row has `"n": null`.

**Evidence.** Sampled rows confirm `mean` and `std` are populated but the sample-size field is null. The seeds list is in `seeds: [42,43,44,45,46]` at top level, so the count is inferable to 5 — but a paper-facing aggregate that doesn't carry its own n is a small audit-hostile choice.

**Consequence.** Reviewers fact-checking "mean ± std over 5 seeds" would have to cross-reference the top-level seeds array; if the aggregator were ever filtered (e.g. dropped a NaN), the n=5 inference would silently be wrong.

**Recommendation.** Populate `n` per row in the aggregator.

### LOW — LOW-1 — `circuit_atlas.md` cites `(frozen checkpoint epoch 1969)` thrice; provenance still verified via other occurrences

Cosmetic / robustness — see MED-1 for the gate-side concern.

### INFORMATIONAL — INFO-1 — R²<0 honesty: caption_note is accurate

I recomputed the set of (model, pipeline) with `r2_mean < 0` from `figures/tstr_crossmodel.json#per_model_pipeline`:

| key | r2_mean | sign |
|---|---|---|
| quantum\|A | -4.5724 | NEG ✓ |
| wgan_mlp\|A | -0.2529 | NEG ✓ |
| wgan_lstm\|A | -0.7530 | NEG ✓ |
| wgan_cnn\|A | +0.0754 | pos |
| vae\|A | +0.9934 | pos |
| (all \|B) | ≥+0.99 | pos |

The companion JSON's `negative_r2_observed = ["quantum|A", "wgan_lstm|A", "wgan_mlp|A"]` and `caption_note` "the exact set ... is: quantum, wgan_lstm, wgan_mlp" are both accurate. R²<0 is plotted and disclosed honestly. ✓

### INFORMATIONAL — INFO-2 — Sweep manifest consistency

`matched2000/sweep_status.json` shows `all_complete: true`, `completed_count: 45`, `total_count: 45`, `len(runs) == 45`. ✓
`model_info.json#iqp_sel_55_repro.parameter_count == 55 == canonical_config_lock.json#param_count == 55 == canonical_config_lock.json#decomposition.param_count == 55`. ✓
`model_info.json#wgan_mlp.parameter_count == 74 == classical_architectures.json#models.wgan_mlp.total_params`. ✓ Same for wgan_cnn (73), wgan_lstm (78), vae (562), ar (3). ✓

### INFORMATIONAL — INFO-3 — Headline-vs-repro conflation: not observed in paper-facing docs

I searched all 9 docs for sentences asserting an EMD value without qualifying frozen vs matched. Every EMD literal that names a number is either (a) a table cell with explicit basis citation (e.g. `methods_full.md:241`, `circuit_atlas.md:74`, `paper_blocks_refs_methods.md:644`), (b) a verbatim manuscript BEFORE quotation marked for replacement (e.g. the 0.6843 DTW BEFORE block in paper_blocks_framing.md:119), or (c) a derived/aggregate cell with an explicit "headline" vs "matched2000_reproduction" / "iqp_sel_55_repro" label. `reviewer_response.md:78-81` explicitly states "the headline ... is the frozen best-EMD checkpoint" and "its matched-budget reproduction is a distinct record ... and the two are never conflated." No conflation found. ✓

---

## §4 — Spot-checks performed

| # | Doc location | Quoted literal | Claimed source path | Actual JSON value | Match |
|---|---|---|---|---|---|
| 1 | `methods_full.md:70` | `1969` | `canonical_config_lock.json` checkpoint_epoch | `1969` | ✓ |
| 2 | `training_protocol.md:13` | `1.8046e-05` | `model_info.json` models[iqp_sel_55_repro].lr_critic | `1.8046e-05` | ✓ |
| 3 | `dataset_stats.md:11` | `778` | `model_info.json` dataset.raw_csv_rows | `778` | ✓ |
| 4 | `methods_full.md:229` | `1.8046e-05` | `methods_full.json` buckets.3_training.lr_critic | `1.8046e-05` | ✓ |
| 5 | `paper_blocks_refs_methods.md:570` | `778` | `model_info.json` dataset.raw_csv_rows | `778` | ✓ |
| 6 | `paper_blocks_refs_methods.md:644` | `0.1209437521974767` | `fidelity_dualscale.json` rows[quantum,B,42,emd,log_return].value | `0.1209437521974767` | ✓ |
| 7 | `paper_blocks_refs_methods.md:644` | `0.022937980562900886` | `fidelity_dualscale.json` rows[quantum,B,42,emd,OD].value | `0.022937980562900886` | ✓ |
| 8 | `paper_blocks_framing.md:425` | `55` | `canonical_config_lock.json` param_count | `55` | ✓ |
| 9 | `training_protocol.md:43` | `91e447d4624e25b3` | `model_info.json` data_hash | `91e447d4624e25b3` | ✓ |
| 10 | `reconciliation_note.md:13` | `0.027586` (OLD) | `baseline_comparison.json` rows mean over seeds (computed) | mean=0.02758597806884… (matches at 6dp via `multiseed_summary.json` rollup) | ✓ |

**Spot-check verdict: 10/10 pass against the source the doc cites.** The 10 spot-checked numbers are honest. The substantive problems live in (a) `reconciliation_note.md` (CRIT-1, scale-mix in the NEW column), (b) the gate's false-positive surface (HIGH-1), and (c) the substring resolution path that lets `0.6843` slip through (CRIT-2 — caught when I went beyond the random 10 to spot-check load-bearing manuscript constants).

---

## §5 — Gate internal audit

| Concern | Status |
|---|---|
| `_NUM` regex catches sci-notation (`1.8046e-05`) and signed decimals | ✓ |
| `_NUM` regex handles LaTeX `\times 10^{-5}` style | n/a — no such pattern in docs |
| `_ID_PATTERNS` strips real data accidentally | LATENT (MED-1, MED-2) |
| `_ALLOW` only contains trivially-universal small integers | ✓ ({"0","1","2","-1","+1","0.0","1.0","2.0","-1.0"}) |
| `rglob("*.json")` pools companion JSONs with source JSONs | LATENT (MED-3) |
| Float-precision matching path admits coincidental matches | ACTIVE (HIGH-1, FP1) |
| Substring matching path admits decimal-substring false positives | ACTIVE (CRIT-2 — `0.6843` "resolves" via `0.03006843578…`) |

---

## §6 — Final assessment

> Would a fact-checking reviewer reading every cited number against the source data find any discrepancy?

**Yes — at least one substantive scale-mix in `reconciliation_note.md` (CRIT-1) would not survive review.** A reviewer who opened `matched2000_dualscale.json` (which the doc itself cites as the source-of-truth in `paper_blocks_framing.md:13` and `paper_blocks_refs_methods.md:553`) would find the matched-2000ep OD-scale aggregate mean for iqp_sel_55_repro is `0.027526`, not `0.154999`. The reviewer would conclude either (a) the reconciliation table is on a different scale than the header asserts, or (b) the "matched 2000ep improved/worsened the EMD" interpretation in the doc is computed against the wrong column.

A secondary concern is the phantom resolution of the manuscript's quoted DTW score `0.6843` in `paper_blocks_framing.md:119` (CRIT-2). The provenance footer at `:520` openly admits it's a substring coincidence. A determined reviewer would call this out as a number lacking a real source.

Everything else — methods_full.md, training_protocol.md, dataset_stats.md, circuit_atlas.md, the PAPER-09 fidelity-dualscale table, the cross-artifact field consistency, the matched2000 sweep completeness, the R²<0 honesty, and 10 of 10 random spot-checks — is honest and traces correctly to the cited JSON fields at the cited precision.

**Required fix to clear the audit:** restate the NEW column of `reconciliation_note.md` against `matched2000_dualscale.json#aggregates[*, emd, OD].mean` (already-stored audited fields, so the gate will pass automatically and the deltas will then be small and consistent with the LOCKED de-overclaiming framing). Optionally remove `0.6843` from the BEFORE quotation in `paper_blocks_framing.md` since the AFTER block already replaces that sentence.
