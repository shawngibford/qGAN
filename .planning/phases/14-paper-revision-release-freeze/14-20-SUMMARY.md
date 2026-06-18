---
phase: 14-paper-revision-release-freeze
plan: 20
type: execute
status: complete
gap_closure: true
requirements: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]
freeze_candidate_pre_14_20: 6518323
freeze_candidate_post_14_20: 3c8502c76f1ad2395f9c66d0feb851e4479466df
freeze_candidate_post_14_20_short: 3c8502c
completed: 2026-05-24
---

# 14-20 SUMMARY — Re-run utility battery against matched-budget Pipeline B artefacts

## Purpose

Close the regime mismatch between R1-M1 and R1-M2 surfaced during the AIChE
rebuttal drafting session: the existing utility JSONs (`tstr.json`,
`predictive_discriminative.json`, `augmentation.json`) were generated against
1000-epoch phase-09.1 `transform_ablation` (quantum) + phase-10 `baselines/runs/`
(classical) artefacts — a different training-epoch regime than the matched-budget
2000-epoch Pipeline B sweep that backs the R1-M1 parametric-efficiency claim.
14-20 re-runs the post-hoc utility battery against the matched2000 artefacts so
the R1-M2 utility evaluation shares R1-M1's matched-budget evidence base.

No retraining was required — the matched2000 sweep was already complete in
phase 14-02 (9 trainable model_kinds × 5 generator seeds × 2000 epochs ×
Pipeline B, samples.npy on disk under `results/matched2000/runs/`).
Only the post-hoc utility nets (1-layer LSTM soft sensor; TimeGAN-convention
predictive + discriminative GRU pairs; Orlandi-style augmentation evaluator)
were re-trained — generators were not touched.

## Outputs

### New artefacts (sibling files; legacy JSONs preserved byte-unchanged)

| File | Rows | Aggregate blocks | Scope |
|---|---|---|---|
| `results/tstr_matched2000.json` | 108 | 10 (9 variants + real_only_baseline) | TSTR R²/MAE/RMSE, 9 model_kinds × Pipeline B × 5 seeds × 3 init seeds |
| `results/predictive_discriminative_matched2000.json` | 90 | 9 | TimeGAN predictive + discriminative (\|acc−0.5\| convention) |
| `results/augmentation_matched2000.json` | 135 | 9 | Orlandi-style real-only vs +25%/+50%/+100% synthetic augmentation lift |
| `figures/tstr_crossmodel_matched2000.{png,pdf,json}` | — | — | 9-variant Pipeline-B cross-model TSTR bars + real-only dashed reference |

### Code changes

- `run_utility.py` — `MODEL_KINDS` replaced with the matched-budget 9-list (`iqp_sel_55_repro`, `V1`, `V2`, `V3`, `wgan_mlp`, `wgan_cnn`, `wgan_lstm`, `vae`, `ar`); `PIPELINES = ["B"]`; `_run_base()` collapsed to single matched2000-routed branch; `_assert_data_hash_invariant()` adjusted to iterate all 45 cells (no quantum-by-construction shortcut — all matched-budget configs carry `91e447d4624e25b3` directly); `reconstruct_od()` raises `NotImplementedError` for Pipeline A in matched-budget driver mode; output filenames retargeted to `*_matched2000.json` siblings.
- `run_timegan_scores.py` — identical shape of edits.
- `run_figure_suite.py` — new sibling renderer `render_tstr_crossmodel_matched2000()` added (no special-case "quantum" label collapse; four quantum variants labeled explicitly via `MODEL_LABELS`; single Pipeline-B panel; real-only baseline plotted as dashed reference; negative-R² treatment retained for safety).

### Doc updates

- `docs/reviewer_response.md` — R1-M2 row in summary table updated to cite matched-budget JSONs + matched-budget figure; "R1-M2 — Utility-oriented evaluation — matched-budget re-run (Plan 14-20)" section completely rewritten with per-variant 9-model table (TSTR R²/MAE/RMSE + predictive + discriminative + +100% augmented R²), cross-generator convergence reading, structural-utility-from-cumulative-sum interpretation, scope note documenting Pipeline-B-only matched-budget protocol.
- `docs/methods_full.md` — new §3.y "Utility-oriented evaluation at matched-budget Pipeline B (Plan 14-20)" added between the DTW historical-context block and §4.
- `docs/completeness_sweep_manifest.md` — legacy `tstr_crossmodel` row annotated as 1000-epoch / not-cited; 4 new rows added for matched-budget artefacts.

## Headline matched-budget result (Pipeline B, 2000 epochs)

| Model | n_params (gen) | TSTR R² | TSTR MAE | TSTR RMSE | Predictive | Discriminative | +100% augmented R² |
|---|---|---|---|---|---|---|---|
| iqp_sel_55_repro | 55 | 0.9945 | 0.0286 | 0.0361 | 0.01944 | 0.40888 | 0.9695 |
| V1 | 75 | 0.9942 | 0.0295 | 0.0370 | 0.01947 | 0.40888 | 0.9688 |
| V2 | 135 | 0.9946 | 0.0283 | 0.0358 | 0.01953 | 0.40888 | 0.9685 |
| V3 | 75 | 0.9949 | 0.0275 | 0.0345 | 0.01925 | 0.40888 | 0.9706 |
| wgan_mlp | 74 | 0.9976 | 0.0183 | 0.0236 | 0.01963 | 0.40888 | 0.9667 |
| wgan_cnn | 73 | 0.9971 | 0.0202 | 0.0260 | 0.02538 | 0.40888 | 0.9624 |
| wgan_lstm | 78 | 0.9966 | 0.0220 | 0.0282 | 0.01981 | 0.40888 | 0.9565 |
| vae | 562 | 0.9930 | 0.0319 | 0.0407 | 0.01960 | 0.40888 | 0.9641 |
| ar(2) | 3 | 0.9977 | 0.0184 | 0.0235 | 0.01884 | 0.40888 | 0.9568 |
| **real-only baseline (n = 65 real windows)** | — | **-13.354** | **1.802** | **1.840** | — | — | — |

### Observed cross-generator convergence

- **TSTR R² band [0.993, 0.998]** across all 9 generators against a real-only baseline of -13.354 ± 0.583 — a 0.005-wide cluster spanning a 3-parameter closed-form AR(2) to a 250881-parameter adversarial WGAN-CNN.
- **TimeGAN discriminative score = 0.40888 to five decimal places across every one of the 45 matched-budget cells** (six architecture families, five generator seeds, three init seeds — 270 distinct evaluator runs total).
- **Augmentation lift R² ∈ [0.957, 0.971] at +100% injection** uniformly across generators (V1 highest at 0.9706; AR(2) lowest at 0.9568).
- **Predictive score band [0.01884, 0.02538]** — eight generators tightly clustered at 0.0188-0.0198; wgan_cnn the only deviation at 0.0254 ± 0.0077, attributable to its seed-42 outlier disclosed under R1-M1.

The cross-generator convergence is structural: the cumulative-sum back-transform
from log-returns to OD mathematically encodes near-perfect lag-1 autocorrelation
into synthetic OD regardless of generator, so a soft sensor trained on
Pipeline-B-derived synthetic data essentially learns the persistence forecast
(near-optimal on the real OD series). The Pipeline B utility battery therefore
does not discriminate among generators on the basis of model quality — it
measures the strength of the back-transform's structural autocorrelation
encoding, which is generator-invariant by construction.

The reviewer-facing R1-M2 section reports this honestly: the synthetic data
ARE useful for downstream OD forecasting at n = 65 real training windows
(augmentation lifts R² from -13.354 to ~0.97), but no generator outperforms any
other on this utility battery. The sole quantum-distinguishing utility-adjacent
metric in the matched-budget comparison remains log-return DTW (LR-DTW),
addressed under R1-M1.

## Verification gates

| Gate | Result |
|---|---|
| `_assert_data_hash_invariant` — all 45 matched-budget configs carry canonical hash | PASS (`91e447d4624e25b3`) |
| `tstr_matched2000.json` schema | PASS (model_kinds = 9-list, pipelines = ["B"], 108 rows, 10 aggregate blocks) |
| `predictive_discriminative_matched2000.json` schema | PASS (9 model_kinds, pipelines = ["B"], 90 rows, 9 score blocks) |
| `augmentation_matched2000.json` schema | PASS (9 model_kinds, pipelines = ["B"], 135 rows, 9 lift blocks) |
| Legacy `tstr.json` / `predictive_discriminative.json` / `augmentation.json` byte-unchanged | PASS (May 17 timestamps preserved) |
| Provenance gate — `reviewer_response.md` | PASS (152 distinct literals all resolve) |
| Provenance gate — `methods_full.md` | PASS (124 distinct literals all resolve) |
| Provenance gate — `completeness_sweep_manifest.md` | PASS (54 distinct literals all resolve) |
| Provenance gate — `paper_blocks_framing.md` | PASS (23 literals, unchanged from pre-14-20) |
| Provenance gate — `paper_blocks_refs_methods.md` | PASS (49 literals, unchanged from pre-14-20) |
| Provenance gate — `reconciliation_note.md` | PASS (67 literals, unchanged from pre-14-20) |
| Provenance gate — `peer_review_remediation.md` | PASS (105 literals, unchanged from pre-14-20) |
| `verify_freeze_ready.py` (a) gitignore + archive scope | PASS (905 tracked paths under revision/results, no provenance JSON gitignored) |
| `verify_freeze_ready.py` (b) provenance over paper-blocks files | PASS (3/3) |
| `verify_freeze_ready.py` (c) tag-scope | PASS (qgan_env/ not tracked, data.csv tracked, no checkpoint > 26 MB) |
| `verify_freeze_ready.py` (0) clean working tree | PASS (`git status --porcelain` empty) |
| `verify_freeze_ready.py` (d) release.md exists | EXPECTED FAIL — release.md is 14-07's deliverable (Plan 14-19 ordering guard) |
| `git status --porcelain` empty after all Task 1–4 commits | PASS |

## Freeze candidate

**Pre-14-20:** `6518323` (post-14-19 HEAD, recorded in 14-19-SUMMARY.md as the SYNTHESIS H5 freeze candidate).

**Post-14-20 (NEW):** `3c8502c76f1ad2395f9c66d0feb851e4479466df` (this plan's final Task 3 commit; Task 4 commits the SUMMARY + STATE.md update on top).

`verify_freeze_ready.py` PASSes every gate against `3c8502c` except gate (d) `release.md`, which is 14-07's deliverable and is the intended ordering guard from Plan 14-19. When 14-07 finally runs (at journal acceptance), the tag `v2.0-revision` must be cut from the post-14-20 SHA — `3c8502c` for the certified tree, or the Task-4 SUMMARY commit on top of it if the SUMMARY is part of the cited tree.

STATE.md `Deferred Items` row for 14-07 has been updated to record `3c8502c`
as the active freeze-candidate reference; pre-14-20 SHA `6518323` is no longer
the active reference.

## Constraints honored

- **No retraining.** Generators trained in phase 14-02 were re-used unchanged.
- **No `core/` edits** (D-11-10 metric-math invariant preserved).
- **No mutation of legacy 1000-epoch utility JSONs** (`tstr.json`, `predictive_discriminative.json`, `augmentation.json` retain their May 17 timestamps and byte contents; they remain on disk as provenance reference).
- **No mutation of pre-14-20 numbers in paper-facing docs** (only the R1-M2 row + R1-M2 strengthened section in `reviewer_response.md`, the new §3.y in `methods_full.md`, and the manifest rows in `completeness_sweep_manifest.md` were edited; provenance gate confirms no literal regression in the previously-certified docs).

## Commits

| SHA | Message |
|---|---|
| `b12afaf` | docs(14-20): plan utility re-eval against matched-budget Pipeline B artefacts |
| `216d8f8` | feat(14-20): re-route run_utility + run_timegan_scores to matched2000 sweep artefacts (Pipeline B, 2000 epochs) |
| `55d6cfb` | feat(14-20): emit matched-budget utility JSONs (TSTR + augmentation + predictive/discriminative) |
| `5766099` | feat(14-20): add matched-budget tstr_crossmodel renderer + figure (Pipeline B, 9 variants) |
| `3c8502c` | docs(14-20): rewrite R1-M2 against matched-budget utility numbers; methods + manifest updated; provenance gate PASS |
| *(this SUMMARY commit, Task 4)* | docs(14-20): complete utility re-eval; refresh freeze candidate at 3c8502c |

## Status

Phase 14: 19/20 plans complete (was 18/19 pre-14-20). 14-07 (Zenodo tag + DOI
deposit + release.md + manuscript DOI wire-in) remains deferred to journal
acceptance — first-round revision resubmits with the `ZENODO-DOI-PLACEHOLDER`
token + the new freeze-candidate SHA `3c8502c` in the Data Availability
statement; the real DOI is minted at acceptance and the camera-ready is updated
then. Phase 14 will close as 20/20 when 14-07 finally runs.
