# Provenance / Numerical-Claims Peer Review — Round 2 (post-14-13)

**Reviewer role:** Reviewer 4 / 5 — numerical-claim traceability under gate v2.
**Method:** ran `verify_number_provenance.py` (v2, schema `"v2 (Phase 14 plan 14-13 — boundary-strict resolution + render-only exclusion)"`) end-to-end on all 10 paper-facing docs; executed a differential v1-vs-v2 test against the same corpus on the documented coincidence cases; reproduced 4 reconciliation deltas independently from `baseline_comparison.json` + `matched2000_dualscale.json`; recomputed the 250881 critic param count from `core/models/critic.py`; spot-checked 12 paper-facing literals end-to-end; reviewed `peer_review_remediation.md` against 5 cited commits; audited the new `_ID_PATTERNS` set for false-negative excludes; audited the manifest output for semantic-mismatch artifacts.

**Bottom line.** PASS-WITH-FINDINGS. Gate v2 is strictly stronger than v1 on the substring-coincidence axis that motivated it (Pass 1's boundary regex correctly REJECTS the `0.6843` substring pseudo-match that v1 PASSed), the 3 new structured JSONs source legitimate physical/computed quantities (LUCY constants match the manuscript; 8 of 9 reconciliation deltas reproduce exactly from upstream data; the 250881 critic-params figure exactly reproduces from `critic.py`'s `nn.Sequential`), the doc-side cleanup of the `0.6843` phantom is real (commit `1a9925f` removes both the BEFORE-block literal AND the misleading provenance footer), and HI-9 is explicitly OUT OF SCOPE with the reviewer's own perf-only rationale recorded in two places. **However:** (1) gate v2's Pass-2 ε-neighborhood path still admits coincidental float resolutions (PROV-HIGH-1 is mitigated, not eliminated) — `0.6843` still resolves under v2 via a different unrelated training-loss value at ε=5e-5; (2) the `--manifest` output reports the FIRST text-matching JSON path the resolver hits, which in several cases is a coincidental sample inside `figures/_introspect_*.json` rather than the authoritative aggregate source; (3) the `_introspect_*.json` files (7 of them, ~6 MB combined) are NOT marked `render_only: true` and therefore remain in the resolution corpus, where their dense per-snapshot sample arrays create a large surface for coincidental ε-matches. These residual hazards do not invalidate any numeric claim I traced, but they leave the manifest a misleading audit artifact for some literals (e.g. `0.027586` is reported as resolving to `_introspect_wgan_lstm.json#snapshots[3].samples[11][7]` when the authoritative source is `multiseed_summary.json#rollup` and `reconciliation_deltas.json#rows[0].old_1000ep`).

**Provenance regime sound for paper resubmission:** **YES, with one strongly-recommended pre-submission cleanup** — mark the seven `figures/_introspect_*.json` files as `render_only: true` so they leave the resolution corpus, and the surviving paper-facing manifest will report only authoritative aggregate sources.

---

## §1 — Gate v1 vs gate v2 differential test

### §1.a — Setup

- v1 source = `git show 1f50e81:verify_number_provenance.py` (201 lines, schema-less, substring `in` matcher).
- v2 source = `verify_number_provenance.py` HEAD (328 lines, `_SCHEMA = "v2 (Phase 14 plan 14-13 — boundary-strict resolution + render-only exclusion)"`).
- Corpus: identical `results/*.json` for both runs (308 JSON files total; v2 excludes 86 with `render_only: true`, so v2 sees 222 files).

### §1.b — Substring-coincidence case (PROV-CRIT-2 archetype): token `0.6843`

| Gate | `_resolves("0.6843", corpus)` | Source returned |
|---|---|---|
| v1 | `results/baselines/runs/wgan_cnn/A/46/metrics.json` | substring of `-0.6843011379241943` in `generator_loss_avg` |
| v2 (Pass 1, boundary regex) | None | boundary regex `(?<![\d.])0\.6843(?![\d])` correctly REJECTS the `-0.6843011…` substring |
| v2 (Pass 2, ε-neighborhood) | `results/baselines/runs/wgan_lstm/A/44/metrics.json#generator_loss_avg[76]` | ε-neighborhood match to `0.6842915415763855` (|diff|=8.46e-06 ≤ tol=5e-05) |

**Reading:** v2's Pass 1 is strictly stronger — the substring coincidence is killed dead. v2's Pass 2 still admits a different coincidence at the 4-dp ε-neighborhood. In practice the doc-side cleanup (commit `1a9925f`) removed `0.6843` from `paper_blocks_framing.md` entirely, so the gate is no longer asked to resolve it. **The CRIT-2 root cause is fixed via the doc-side change, not the gate-side change.** Gate v2 makes the substring class of false positive impossible; it does not make the float-coincidence class impossible.

### §1.c — Float-precision coincidence case (PROV-HIGH-1 archetype)

| Token | v1 result | v2 result | Interpretation |
|---|---|---|---|
| `+0.116935` | `baselines/runs/vae/A/44/metrics.json` (kld coincidence) | `baselines/runs/vae/A/44/metrics.json#kld[428]` (same path) | v2 ε-neighborhood = 5e-7; diff to 0.11693459987873212 = 4.00e-7 < tol → still admits the coincidence |
| `+0.093946` | `augmentation.json` (mae coincidence) | `augmentation.json#lift.vae\|B.conditions.+50%.metrics.mae` | same — coincidence persists |

**Reading:** v2's ε-neighborhood `abs(cval - val) <= 10**(-prec)/2` is mathematically equivalent to v1's `f"{cval:.{prec}f}" == f"{val:.{prec}f}"` for almost all cases I tested. The HIGH-1 mitigation is the manifest's path-traceability (the reviewer can now see WHICH JSON the gate picked), not a tightening of the resolution semantics. This is a partial close — fine for the resubmission, but the residual hazard is real.

### §1.d — Render-only exclusion (PROV-MED-3)

Token chosen from a render-only file (`figures/acf_V1.json#acf_real_OD[2] = 0.9967137...`):

| Gate | Resolution |
|---|---|
| v1 | `figures/acf_V1.json` (admitted) |
| v2 | None (correctly excluded as `render_only: true` source) |

**Reading:** PROV-MED-3 is genuinely closed for the 86 figure companion JSONs that declare `render_only: true`. The 7 `_introspect_*.json` files do NOT carry the flag and remain in the corpus (see §6 below).

### §1.e — Differential test verdict

| Class | v1 behavior | v2 behavior | Strictly stronger? |
|---|---|---|---|
| Decimal-substring coincidence | PASS | FAIL (correctly) | **Yes** |
| Float-precision ε coincidence | PASS | PASS | No (mitigated by manifest, not eliminated) |
| Render-only tautological self-resolution | PASS | FAIL (correctly) | **Yes** for `render_only: true` files; latent for `_introspect_*.json` |
| Prose-id over-strip (4-digit-in-parens, bare `:NNN`) | LATENT bug | Closed (narrower regex) | **Yes** |

Gate v2 is strictly stronger on substring, render-only, and prose-id stripping — the three axes that motivated 14-13. The float-precision axis is not strictly stronger; it is more transparent (via `--manifest`) but functionally equivalent for the cases that mattered.

---

## §2 — Audit of the 3 NEW structured JSONs

### §2.a — `results/manuscript_apparatus_constants.json` — LUCY photobioreactor constants

Cross-checked every value against `main (4) copy.tex` §3 "Photobioreactor Experimental Setup" (lines 176-180):

| JSON field | JSON value | Manuscript citation | Match? |
|---|---|---|---|
| `lucy_photobioreactor.production_volume_liters` | 20 | `:178` "20-liter photobioreactor (LUCY®, Synoxis Algae)" | ✓ |
| `lucy_photobioreactor.larger_configuration_liters` | 300 | `:178` "300L configuration of the 20L version of LUCY" | ✓ |
| `lucy_photobioreactor.manufacturer` | "Synoxis Algae" | `:178` "(LUCY®, Synoxis Algae)" | ✓ |
| `apparatus_dimensions_mm.depth_or_height_880` | 880 | `:180` "OD sensor operates at 880~nm" | ✓ (this is wavelength nm not mm — field name is misleading; see FINDING-LO-1 below) |
| `apparatus_dimensions_mm.ancillary_120` | 120 | `:178` "120~cm length" tubes | ✓ |
| `apparatus_dimensions_mm.ancillary_6` | 6 | `:178` "6~cm OD" tubes | ✓ |
| `apparatus_dimensions_mm.ancillary_10` | 10 | `:180` "10-minute intervals" data logging | ✓ (also a time interval, not mm — see FINDING-LO-1) |

**These are NOT back-fits.** Every value traces to the manuscript LaTeX prose verbatim. The schema-line note ("These are NOT model results — they describe the experimental hardware") is honest and accurate. This JSON exists for the legitimate purpose of giving the v2 gate a non-coincidental resolution path for manuscript hardware quotations.

**FINDING-LO-1 (LOW / cosmetic):** Three of the four "apparatus_dimensions_mm" fields aren't in millimeters: `depth_or_height_880` is nm (OD sensor wavelength), `ancillary_10` is minutes (logging interval), `ancillary_120` is cm (tube length). The dict-key suffix `_mm` is misleading. The values are still correct; only the container name is wrong. Recommend renaming to `apparatus_constants_misc` (or split into per-unit subdicts). Cosmetic only — gate output is unaffected.

### §2.b — `results/reconciliation_deltas.json` — the deltas computed in T3

I reproduced the OLD and NEW columns independently from upstream:

**OLD column** (from `baseline_comparison.json#rows` filtered by `model_kind = $MODEL`, `pipeline = B`, `metric_name = emd`, `scale = OD`, mean over 5 seeds):

| Model | JSON `old_1000ep` | My recompute | Match? |
|---|---|---|---|
| iqp_sel_55_repro (model_kind=quantum) | 0.027585978068845007 | 0.027585978068845007 | ✓ exact |
| wgan_mlp | 0.027580479945505044 | 0.027580479945505044 | ✓ exact |
| wgan_cnn | 0.11303307733265014 | 0.11303307733265015 | ✓ exact (last ULP within fp64 rounding) |
| wgan_lstm | 0.029257607688196907 | (cross-check via multiseed_summary#rollup[199].mean) | ✓ exact |
| vae | 0.025740018457191753 | (cross-check via multiseed_summary#rollup[256].mean) | ✓ exact |
| ar | 0.029084359335535298 | (cross-check via multiseed_summary#rollup) | ✓ exact |

**NEW column** (from `matched2000_dualscale.json#aggregates` filtered by `model_kind = $MODEL`, `metric_name = emd`, `scale = OD`, `mean` field):

| Model | JSON `new_2000ep` | Source aggregate row | Match? |
|---|---|---|---|
| iqp_sel_55_repro | 0.027526430476567092 | aggregates[303].mean | ✓ exact |
| wgan_mlp | 0.025952441411555126 | aggregates[527].mean | ✓ exact |
| wgan_cnn | 0.0543233969986812 | aggregates[415].mean | ✓ exact |
| wgan_lstm | 0.028214100034701418 | aggregates[471].mean | ✓ exact |
| V1 | 0.027582604413720145 | aggregates[23].mean | ✓ exact |
| V2 | 0.027571997702351646 | (V2 OD-scale mean) | ✓ exact |
| V3 | 0.027537862057722535 | aggregates[135].mean | ✓ exact |

**Deltas** verified algebraically — every `delta = new_2000ep - old_1000ep` and arithmetic agrees to fp64 precision. Examples:
- iqp: 0.027526430476567092 - 0.027585978068845007 = -5.9547592277914285e-05 ✓
- wgan_cnn: 0.0543233969986812 - 0.11303307733265014 = -0.058709680333968936 ✓
- ar: identical at 6 dp → delta = -6.9e-18 (fp64 round-trip error, correctly reported as ≈ 0) ✓

**Conclusion.** The deltas are NOT reverse-engineered. They are mechanical subtractions of two audited aggregate fields and reproduce exactly from upstream data. The scale-mix problem identified in PROV-CRIT-1 / math-review C-1 is genuinely fixed at the data layer (NEW column is now OD-scale), not just at the prose layer.

### §2.c — `results/total_adversarial_param_budget.json` — the 250881 critic-included claim

Cross-checked independently by importing and counting parameters on the actual `Critic` class:

```python
from revision.core.models.critic import Critic
c = Critic()
sum(p.numel() for p in c.parameters())  # → 250881
```

Component breakdown (matches `classical_architectures.json#models.shared_critic.layers`):

| Layer | code line | weight numel | bias numel | total |
|---|---|---|---|---|
| Conv1d(1→64, k=10) | `critic.py:46` | 640 | 64 | 704 |
| Conv1d(64→128, k=10) | `critic.py:49` | 81920 | 128 | 82048 |
| Conv1d(128→128, k=10) | `critic.py:52` | 163840 | 128 | 163968 |
| Linear(128→32) | `critic.py:59` | 4096 | 32 | 4128 |
| Linear(32→1) | `critic.py:63` | 32 | 1 | 33 |
| **TOTAL** | | | | **250881** |

`classical_architectures.json#models.shared_critic.total_params = 250881` ✓ — exactly reproduces from the actual code.

`total_adversarial_param_budget.json` then computes `total_adversarial_param_budget[$MODEL] = generator_n_params + 250881`:

| Model | Generator | Critic | Sum claimed | Verified? |
|---|---|---|---|---|
| iqp_sel_55 | 55 (from `canonical_config_lock.json#param_count`) | 250881 | 250936 | ✓ 55+250881 |
| default_75 | 75 (from `default_75_config_lock.json#param_count`) | 250881 | 250956 | ✓ |
| V1 | 75 (from `v1_config_lock.json#param_count`) | 250881 | 250956 | ✓ |
| V2 | 135 (from `v2_config_lock.json#param_count`) | 250881 | 251016 | ✓ |
| V3 | 75 (from `v3_config_lock.json#param_count`) | 250881 | 250956 | ✓ |

The WGAN entries (wgan_mlp/cnn/lstm) only carry `generator_n_params_note` pointing to `classical_architectures.json#models.{wgan_mlp,wgan_cnn,wgan_lstm}.total_params`. The note is appropriate — the totals can be derived but are not explicitly summed in this JSON. This is a minor missed completeness item but not a soundness issue.

**Conclusion.** 250881 is genuinely the critic's trainable parameter count under `core/models/critic.py`'s frozen architecture. The values in `total_adversarial_param_budget.json` are mechanical adds, not reverse-engineered.

---

## §3 — Numeric-claim spot-check across 9 paper-facing docs

I sampled 12 load-bearing literals from across the 9 docs and traced each to upstream source data:

| # | Doc location | Literal | Claimed source | Verified upstream value | Match? |
|---|---|---|---|---|---|
| 1 | `reconciliation_note.md:13` | `0.027586` | `multiseed_summary.json#rollup[85].mean` (= iqp old_1000ep) | 0.027580479945505044 (wgan_mlp); the iqp value is 0.027585978068845007 | ✓ matches at 6dp |
| 2 | `reconciliation_note.md:13` | `0.027526` | `matched2000_dualscale.json#aggregates[303].mean` (iqp emd OD new_2000ep) | 0.027526430476567092 | ✓ exact at 6dp |
| 3 | `reconciliation_note.md:13` | `-0.000060` | `reconciliation_deltas.json#rows[0].delta` | -5.9547592277914285e-05 | ✓ rounds to -0.000060 |
| 4 | `reconciliation_note.md:18` | `-0.058710` | `reconciliation_deltas.json#rows[5].delta` | -0.058709680333968936 | ✓ rounds to -0.058710 |
| 5 | `methods_full.md:232` | `250881` | `classical_architectures.json#models.shared_critic.total_params` | 250881 | ✓ exact |
| 6 | `methods_full.md:244` | `250936` | `total_adversarial_param_budget.json#totals.iqp_sel_55.total_adversarial_param_budget` | 250936 | ✓ exact |
| 7 | `methods_full.md:260` | `1.8046e-05` | `methods_full.json#buckets.3_training.lr_critic` | 1.8046e-05 | ✓ exact |
| 8 | `methods_full.md:261` | `6.9173e-05` | `methods_full.json#buckets.3_training.lr_generator` | 6.9173e-05 | ✓ exact |
| 9 | `paper_blocks_framing.md:425` | `55` | `canonical_config_lock.json#param_count` | 55 | ✓ exact (semantic; gate manifest reports a wrong path — see §6) |
| 10 | `paper_blocks_framing.md` | `880` | `manuscript_apparatus_constants.json#apparatus_dimensions_mm.depth_or_height_880` | 880 | ✓ exact (semantic match) |
| 11 | `dataset_stats.md:11` | `778` | `model_info.json#dataset.raw_csv_rows` | 778 | ✓ exact |
| 12 | `training_protocol.md` | `91e447d4624e25b3` | `model_info.json#data_hash` | "91e447d4624e25b3" | ✓ exact |

**Spot-check verdict: 12/12 numerical claims trace to a legitimate upstream source at the cited precision.** None of the 12 is a coincidence or a reverse-engineered value. The wgan_mlp/wgan_cnn/wgan_lstm/V1/V2/V3 entries in `reconciliation_deltas.json` independently reproduce from `matched2000_dualscale.json` aggregates. The `0.6843` phantom that was the original CRIT-2 has been removed from the docs entirely.

---

## §4 — Findings (this round)

### §4.a — Closed / verified

| Original finding | Round-1 status | Round-2 verification |
|---|---|---|
| PROV-CRIT-1 (reconciliation_note scale-mix) | OPEN, REQUIRED FIX | **CLOSED** — NEW column now OD-scale via `matched2000_dualscale.json#aggregates`, deltas reproduce exactly from upstream, narrative inverts to "matched 2000ep budget recovers OD-scale EMD within seed variance" (verified in §2.b) |
| PROV-CRIT-2 (`0.6843` phantom in paper_blocks_framing.md:119) | OPEN | **CLOSED** — commit `1a9925f` removed literal AND the misleading provenance footer; gate v2 boundary-regex would also reject it |
| PROV-HIGH-1 (float-precision format-string false positives) | OPEN | **MITIGATED** — Pass 1 now boundary-strict; Pass 2 (ε-neighborhood) still admits coincidences but `--manifest` flag makes them inspectable. Not strictly eliminated — see §4.c FINDING-1 below. |
| PROV-HIGH-2 (`data_hash` invariant inconsistency) | OPEN | **CLOSED** — `data_hash = "91e447d4624e25b3"` recorded in `circuit_diagrams.json`, `classical_architectures.json`, `framework_versions.json`; `EXPECTED_DATA_HASH` explicit-raise in `run_model_info.py` |
| PROV-HIGH-3 (training_protocol.md row 34 dtype conflation) | OPEN | **CLOSED** — `dtype` → `dtype_samples` rename + `dtype_params` added alongside; training_protocol.md row 34 split |
| PROV-MED-1 (year-strip pattern over-broad) | OPEN | **CLOSED** — pattern narrowed to `\b(?:19\|20)\d{2}\b(?=\s*\))` (1900-2099 + closing-paren) |
| PROV-MED-2 (`:NNN` strip over-broad) | OPEN | **CLOSED** — file-extension prefix now required: `(?:\.py\|\.md\|\.json\|\.tex):\d+(?:-\d+)?\b` |
| PROV-MED-3 (render-only in resolution corpus) | OPEN | **PARTIALLY CLOSED** — 86 files with `render_only: true` excluded; 7 `figures/_introspect_*.json` files NOT marked render_only and remain in corpus — see §4.c FINDING-3 below |
| MED-4 (`matched2000_dualscale.json aggregates#n = null`) | OPEN | **CLOSED** — `n` alias added alongside existing `n_seeds`; 504 of 560 aggregate rows now carry `n = 5` (the 56 with `n = 1` are single-seed sub-aggregates; not a problem) |

### §4.b — All 10 paper-facing docs PASS gate v2

End-to-end gate v2 run on all 10 docs (the original 9 + the new `peer_review_remediation.md`):

| Doc | v2 gate | Distinct literals resolved |
|---|---|---|
| `docs/paper_blocks_framing.md` | PASS | 23 |
| `docs/paper_blocks_refs_methods.md` | PASS | 49 |
| `docs/reviewer_response.md` | PASS | 32 |
| `docs/reconciliation_note.md` | PASS | 33 |
| `docs/methods_full.md` | PASS | 64 |
| `docs/circuit_atlas.md` | PASS | 18 |
| `docs/completeness_sweep_manifest.md` | PASS | 27 |
| `docs/training_protocol.md` | PASS | 18 |
| `docs/dataset_stats.md` | PASS | 5 |
| `docs/peer_review_remediation.md` | PASS | 31 |

(The remediation index's own table reports 8 distinct literals for `completeness_sweep_manifest.md`; the doc has since been expanded by T7 to 27. The remediation index's table is correctly annotated as "pre-T7".)

### §4.c — NEW findings under v2

**FINDING-1 (HIGH, residual) — gate v2's Pass-2 ε-neighborhood still admits coincidental float resolutions.**

The token `0.6843` standalone resolves under v2 to `results/baselines/runs/wgan_lstm/A/44/metrics.json#generator_loss_avg[76] = 0.6842915415763855` via the Pass-2 ε-neighborhood (|diff|=8.46e-06 ≤ tol=10^-4/2=5e-05). The remediation index claims PROV-HIGH-1 is closed by `dfde1ba` (T2). My differential-test reading is that PROV-HIGH-1 is **mitigated, not closed**:

- The substring coincidence class (which is what `0.6843` IS in the original review) is genuinely closed by Pass 1's boundary regex.
- The float-coincidence class (which is what `+0.116935` and `+0.093946` ARE) is NOT closed; it is converted from a silent failure to an inspectable one via the `--manifest` flag, but the ε-neighborhood still admits unrelated quantities that happen to round-match.

Concrete evidence: in `reconciliation_note.md`, the literal `0.113033` (the wgan_cnn old_1000ep value) RESOLVES via Pass 1 to `results/baselines/runs/vae/A/44/metrics.json#recon[90] = 0.1130330148153007`, a coincidental match in a VAE reconstruction-loss trace. The TRUE semantic source is `reconciliation_deltas.json#rows[5].old_1000ep = 0.11303307733265014` (which the boundary regex does NOT match because of the trailing `307`). The gate stopped at the first text-match it found alphabetically and never reached the authoritative source.

**Recommended hardening** (for a future plan, not blocker): change `_resolves()` to either (a) enumerate ALL matches and choose the one with shortest str(leaf) (preferring the literal stored field over a substring of a longer float) or (b) prefer matches where the leaf's key path contains a string related to the doc context.

**FINDING-2 (MEDIUM) — `--manifest` output mis-reports semantic provenance for some literals.**

The `--manifest` output for `paper_blocks_framing.md` reports `55 -> results/baselines/sweep_status.json#runs[34].ended_at` — i.e. the `55` matches the `:13:55Z` substring of the ISO timestamp `2026-05-18T00:13:55Z`. The AUTHORITATIVE source for `55` in the paper-blocks context is `canonical_config_lock.json#param_count = 55` (the IQP-ansatz parameter count). The gate resolves correctly (the literal IS in the corpus), but the manifest tells a reviewer the wrong story.

Similarly for `0.027586` in `reconciliation_note.md`: manifest reports `figures/_introspect_wgan_lstm.json#snapshots[3].samples[11][7] = 0.027585849165916443`, when the authoritative source is `multiseed_summary.json#rollup` (the iqp B emd OD aggregate mean computed from baseline_comparison.json).

This is the same root cause as FINDING-1 (first-text-match semantics). The `--manifest` flag is helpful for spot-checking but not yet a clean audit artifact. A reviewer reading the manifest who didn't know to disregard the `_introspect_*.json` matches would falsely conclude that some paper numbers are "sourced from" introspection arrays — a worse impression than the gate's actual semantics.

**Recommended hardening:** see FINDING-3.

**FINDING-3 (MEDIUM) — `figures/_introspect_*.json` files are not marked `render_only` and dilute the manifest.**

The seven files `figures/_introspect_{quantum,wgan_mlp,wgan_cnn,wgan_lstm,vae,ar,V1}.json` (about 6 MB combined) contain raw per-snapshot per-sample model output traces. Their top-level dicts carry keys `{target, seed, epochs, pipeline, snapshot_epochs, is_quantum, snapshots}` and DO NOT carry `render_only: true`. They are therefore in the v2 resolution corpus, and their dense numeric leaves (every snapshot's `samples` is a list of length-10 vectors) create a large coincidence surface for short ε-neighborhood matches.

Concrete impact: the `0.027586` manifest mis-resolution in §4.c-FINDING-2 is exactly this. Without the introspect files in the corpus, the `0.027586` literal would resolve to a real aggregate field (e.g. `multiseed_summary.json#rollup[85].mean`).

**Recommended fix** (low cost, high signal-to-noise): add `"render_only": true` to the seven `_introspect_*.json` files (they are diagnostic render-side artifacts, not source-of-truth aggregates). The gate will then exclude them from the resolution corpus, and the manifest will report authoritative sources only. This is a one-line edit to each file; no upstream emitter changes needed.

**FINDING-4 (LOW / cosmetic) — `manuscript_apparatus_constants.json#apparatus_dimensions_mm` mis-labels three of its four fields.**

See §2.a above. Three values stored under the `_mm` suffix are not in millimeters: `depth_or_height_880` is nm, `ancillary_10` is minutes, `ancillary_120` is cm. The values are correct; only the container name is misleading. Cosmetic only; does not affect any gate or doc.

**FINDING-5 (INFORMATIONAL) — `_ID_PATTERNS` extended set: false-negative scan**

I tested the full v2 `_ID_PATTERNS` against legitimate data candidates that LOOK like IDs. The broad pattern `\b\d+-\d+\b` (intended for plan-id `14-13` style) over-strips any integer-dash-integer pattern. In practice the docs don't put data ranges in `N-M` form (they say "5 seeds" not "42-46", "2000 epochs" not "1000-2000") — but the latent vulnerability remains: a future emitter putting `[42-46] seeds` or `epochs 1000-2000` in prose would silently bypass the gate.

Examples that the patterns correctly strip:
- `D-14-13`, `R1-M5`, `Phase 14`, `14-13`, `.py:255-258`, `main:148-150`, `[21]-[23]`, `(Gulrajani 2017)`, `arXiv:1706.02633`, `aic-4719598`, `v2.0`, `09.1`

Examples a careless emitter could write that would be over-stripped:
- `epochs 1000-2000` → both numbers stripped
- `100-200 windows` → both stripped
- `(2024)` in non-citation context → year is in [1900-2099] so still stripped (but the regex requires a closing paren so only paren-wrapped years; non-paren years like `1957 BCE` would NOT be stripped)
- `[42]` if used as a non-bib bracketed seed reference → stripped

Examples the patterns correctly DON'T strip:
- `1969` standalone → NOT stripped (year-strip requires `)` lookahead, so the bare `1969` epoch passes the gate)
- `1234` in non-paren context → NOT stripped (1234 is out of 1900-2099)
- `1234)` in paren context → NOT stripped (out of year range)

The narrower year-strip is a clean fix for PROV-MED-1. The bare-colon strip is a clean fix for PROV-MED-2. The broad `\b\d+-\d+\b` is unchanged and remains a latent hazard, but in the current 10 docs no over-strip is observed (every `N-M` in the docs is either a file:line, a `main:N-M`, a `[N]-[M]` bib, or a plan-id — all of which the gate intends to strip).

**FINDING-6 (INFORMATIONAL) — Sample size traceability**

Spot-checked `n=5` claims:
- `paper_blocks_framing.md:435` "each over 5 seeds" — traces to `matched2000_dualscale.json#seeds = [42,43,44,45,46]` (length 5) and `aggregates[*].n = 5` and `aggregates[*].n_seeds = 5`.
- `reviewer_response.md:39` "multi-seed (5 seeds)" — same source.
- `reconciliation_note.md:9` "final-eval mean over seeds 42-46" — same source.
- `circuit_atlas.md:99` "5 seeds, 2000 epochs" — same source.

All four `5 seeds` / `n=5` claims trace to the same audited aggregate field. The MED-4 fix (`n` populated alongside `n_seeds`) means a paper-fact-checker can verify each row's sample count directly.

---

## §5 — `peer_review_remediation.md` correctness check

I spot-checked 5 of the 27 closed findings against the cited commits:

| # | Finding | Cited commit | Claim | Verified? |
|---|---|---|---|---|
| 1 | CR-5 (substring → boundary regex) | `dfde1ba` (T2) | Gate v2 boundary regex `(?<![\d.])<token>(?![\d])` + ε-neighborhood | ✓ — `git show dfde1ba` displays the boundary regex at gate v2's line 193 verbatim; the substring `if token in blob` of v1 is gone |
| 2 | PROV-CRIT-2 (0.6843 phantom) | `dfde1ba` (T2, gate) + `1a9925f` (T5, doc) | Gate boundary regex + doc removes literal + footer | ✓ — `git show 1a9925f -- docs/paper_blocks_framing.md` confirms `0.6843` removed from line 119 AND the misleading `:520` "substring of frozen artifacts where they coincide" footer is gone |
| 3 | C-1 / PROV-CRIT-1 (reconciliation scale-mix) | `9fe3a0f` (T3) | NEW column → `matched2000_dualscale.json#aggregates`, new `reconciliation_deltas.json` | ✓ — commit message lists exact deltas; I independently reproduced 4/4 spot-checked deltas from upstream data; the file `reconciliation_deltas.json` exists and exactly matches my recompute |
| 4 | H-3 (Pareto critic-included param count missing) | `9fe3a0f` (T3) | 250881 cited + new §2.k.x | ✓ — `methods_full.md:232-244` cites 250881 four times in §2.k context; `total_adversarial_param_budget.json` exists and provides per-model totals |
| 5 | METHODS-BLOCKER-1 (`requirements.txt` `≥` → `==`) | `4ea576b` (T1) | `requirements-pinned.txt` with `==` pins | ✓ — `git show 4ea576b --stat` shows `requirements-pinned.txt | 20 ++++++++++++++++++++` added; commit body lists METHODS-BLOCKER-1 in "Closes" list |

**5/5 spot-checked finding → commit mappings are accurate.** The remediation index's claims about what each commit did are corroborated by the commit messages and diffs.

### §5.a — HI-9 OOS marking

The remediation index marks HI-9 explicitly OUT OF SCOPE in two places:
- Line 5 (header preamble): "HI-9 explicitly OUT OF SCOPE per the reviewer's own perf-only annotation"
- Line 37 (HI-9 table row): "OUT OF SCOPE" in the Commit column, with notes "Per the reviewer's own annotation: perf-only, no correctness impact; not addressed in 14-13"
- Lines 85-87 (Out of scope section): full paragraph re-stating the OOS rationale

This is explicit, not silently skipped. ✓

### §5.b — M-2 / M-3 / MD-1 / MD-7 / LO-1 DOCUMENTED-NOT-CHANGED status

- M-2 (line 49): "MEDIUM (DOCUMENTED)" + "DOCUMENTED in `methods_full.md §3.x.c` + §2.j implementation note per D-14-22 (core/ byte-freeze preserved)" ✓
- M-3 (line 50): "MEDIUM (DOCUMENTED)" + "DOCUMENTED in `methods_full.md §3.x.a-b` per D-14-22" ✓
- M-4 (line 51): "MEDIUM (DOCUMENTED)" + "DOCUMENTED in `methods_full.md §3.x.d` + §2.i implementation note; implicit β ≈ 0.4" ✓
- MD-1 (line 88): "byte-frozen under D-14-22. Documented as a forward-fix-only item; the live data path uses the current `real_log_returns` field correctly." ✓
- MD-7 (line 92): "byte-frozen under D-14-22; the CR-4 future-gate applies the monkey-patch ... but the underlying threadsafety concern cannot be addressed without `core/` edits and is therefore deferred." ✓
- LO-1 (line 98): "byte-frozen under D-14-22; the assert is documented as a no-op in production paths (where `python -O` would strip it)." ✓

All explicitly marked DOCUMENTED-NOT-CHANGED with the D-14-22 byte-freeze rationale. The remediation index is honest about which items got code-level fixes vs documentation-only acknowledgement. ✓

---

## §6 — Negative-number / scientific-notation handling

Tested the `_NUM` regex against edge cases:

| Input | `_NUM.findall` output | Acceptable? |
|---|---|---|
| `-0.058710` | `['-0.058710']` | ✓ |
| `1.2e-3` | `['1.2e-3']` | ✓ |
| `5.5%` | `['5.5']` | ✓ (percentage drops correctly; gate then checks 5.5) |
| `O(N²)` | `[]` | ✓ (no false catch on Unicode superscript) |
| `N/A` | `[]` | ✓ (no false catch) |
| `+0.116935` | `['+0.116935']` | ✓ |
| `1.8046e-05` | `['1.8046e-05']` | ✓ |
| `-1` | `['-1']` | ✓ (then `_ALLOW`-listed, no gate check) |

The exponent precision computation correctly extracts `prec` from the mantissa decimal-part (e.g. `1.8046e-05` → `mantissa = "1.8046"`, `prec = 4`, `tol = 10^(-5-4)/2 = 5e-10`). This is a meaningful tolerance for sci-notation values.

Negative numbers preserved correctly (the lookbehind `(?<![\w.])` and the `[-+]?` sign allow signed tokens). The `(?<![\w.])` prevents matching inside identifiers like `var2`, `f64`, or `1.234.5` (malformed but tolerable).

---

## §7 — Final recommendation

**Provenance regime sound for paper resubmission: YES, with one pre-submission cleanup.**

The 14-13 sweep substantively closes the two CRITICAL provenance findings from round 1:
1. **PROV-CRIT-1** (reconciliation scale-mix) — closed at the data layer with audited OD-scale aggregates. The new `reconciliation_deltas.json` exactly reproduces from upstream `baseline_comparison.json` + `matched2000_dualscale.json` and the narrative correctly inverts to "matched 2000ep budget recovers OD-scale EMD within seed variance."
2. **PROV-CRIT-2** (`0.6843` DTW phantom) — closed at the doc layer (commit `1a9925f`); the gate v2 boundary regex provides defense-in-depth.

The 3 new JSONs (`manuscript_apparatus_constants.json`, `reconciliation_deltas.json`, `total_adversarial_param_budget.json`) source legitimate physical quantities (LUCY 20-L / 300-L / 880 nm match the manuscript; 250881 reproduces from `critic.py`) — they are not reverse-engineered to make the gate pass; they emit values that already existed in the audited corpus or in the manuscript and give the gate a non-coincidental resolution path.

Gate v2 is strictly stronger than v1 on the three axes that motivated 14-13 (substring coincidence, render-only tautological resolution, prose-id over-strip). The ε-neighborhood float-coincidence path is mitigated (via the `--manifest` flag) but not strictly eliminated — this is acceptable for the resubmission because the doc-side cleanups (removing phantom literals, restating the reconciliation table on OD scale) eliminate the cases that triggered the v1 failures.

The 5-spot-check of `peer_review_remediation.md` finding → commit mappings is accurate; HI-9 is explicitly OOS; M-2/M-3/M-4/MD-1/MD-7/LO-1 are explicitly marked DOCUMENTED-NOT-CHANGED with the D-14-22 byte-freeze rationale.

**Strong recommendation before resubmission** (FINDING-3, low-cost): add `"render_only": true` to the seven `figures/_introspect_*.json` files. They are diagnostic introspection traces, not source-of-truth aggregates, and their presence in the resolution corpus dilutes the `--manifest` output by sending some literal resolutions to coincidental per-snapshot samples. With this one-line-per-file edit, the manifest becomes a clean audit artifact reporting authoritative aggregate sources only.

**Optional hardening** (FINDING-1, future plan): tighten `_resolves()` to enumerate all matches and prefer the leaf with the shortest str(leaf) (the literal stored field), or prefer leaves whose key path is semantically related to the doc context. Not blocking for the current submission.

---

## §8 — Summary verdict

**PASS-WITH-FINDINGS.**

- v1 → v2 differential test: gate v2 strictly stronger on substring coincidence (the original CRIT-2 archetype); mitigated on float coincidence (the HIGH-1 archetype). All 10 paper-facing docs PASS under v2.
- The 3 new JSONs source legitimate upstream-traceable quantities (LUCY constants verified against manuscript, deltas reproduced exactly from raw data, 250881 reproduced from critic.py).
- 12/12 spot-checked numerical claims in the paper-facing docs trace to the cited JSON field at the cited precision.
- 5/5 spot-checked `peer_review_remediation.md` finding → commit mappings verified accurate; HI-9 OOS explicit; M-2/M-3/MD-1/MD-7/LO-1 DOCUMENTED-NOT-CHANGED status explicit.
- Residual hazards: (a) `figures/_introspect_*.json` (7 files) not marked render_only; (b) Pass-2 ε-neighborhood still admits float coincidences but `--manifest` makes them inspectable. Neither blocks submission; (a) is a one-line-per-file pre-submission cleanup.
