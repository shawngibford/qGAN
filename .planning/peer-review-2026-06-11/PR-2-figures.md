# PR-2 — Figure & Caption Audit Report

## Verdict
**BLOCKING.** The coauthor's two flagged issues are real, pervasive, and visually verifiable in the compiled PDF. Both must be fixed before the 2026-06-17 resubmission. Independent confirmation from rendering main pages 11, 17–20, 25: every caption from Fig 2 onward contains `\protect\path{revision/...}` traces that render in typewriter font as raw filesystem paths, and Figs 2/3/4/5/6 are all in single-column `figure` env at column-width — axis-tick text and bar-cluster x-labels are unreadable at print size. A reader sees code dumps next to thumbnail plots. This is the highest-severity finding in the figure pass.

---

## C1 — File paths in captions (coauthor's "most repeated note")

### Pattern
The author has consistently appended a `Source: \protect\path{results/...json}` (sometimes 2-line) provenance trace to every figure/table caption. These render in monospace as `figures/<name>.json` etc. and read as leftover code. The reviewer is right: they belong in a Data Availability section, not in reader-facing captions. Two additional "code-dump" patterns appear in body captions: `\texttt{model\_kind=...}` filter snippets and `\texttt{metric\_name==...}` filter clauses.

### Found in main paper ("main (4) copy.tex")

| Figure/Table | Lines | Verbatim trace | Plain-language rewrite |
|---|---|---|---|
| Fig 2 (training_convergence) | 237–240 | `Source: \protect\path{figures/training_convergence_all_models.json}; per-seed trajectories from \protect\path{results/matched2000/runs/<model>/<seed>/metrics.json}.` | **Delete.** Provenance moves to Data Availability §. Replace with nothing (caption already conveys the finding). |
| Tbl 1 (eval_scale) | 362–364 | `Source: \protect\path{results/matched2000_dualscale.json}, filtered to \texttt{model\_kind=iqp\_sel\_55\_repro}.` | **Delete.** Move to Data Availability. |
| Tbl 2 (cross_model_comparison) | 404–406 | `Source: \protect\path{results/matched2000_dualscale.json}, rows aggregated by \texttt{(model\_kind, metric\_name, scale)} over seeds 42--46.` | **Delete.** Move to Data Availability. |
| (orphan comment) | 382 | `% source: results/matched2000_dualscale.json#rows aggregated by ...` | Drop (comment, harmless but signals same pattern). |
| Fig 3 (cross_model_dtw_dualscale) | 497–499 | `Source: \protect\path{results/matched2000_dualscale.json}, filtered to \texttt{metric\_name=dtw\_mean}.` | **Delete.** |
| Fig 4 (cross_model_acf_overlay) | 564–567 | `Source: \protect\path{figures/cross_model_acf_overlay.json}; real-data ACF computed via \protect\path{revision.core.data.load_and_preprocess}.` | **Delete.** The dotted-method reference (`revision.core.data.load_and_preprocess`) is the most egregious case — Python module path in a caption. |
| Fig 5 (cross_model_emd) | 587–591 | `Source: \protect\path{figures/cross_model_emd.json}; rows aggregated from \protect\path{results/matched2000_dualscale.json} filtered to \texttt{metric\_name=emd}, \texttt{scale=OD}.` | **Delete.** |
| §4.1 body text | 436 | `(\path{figures/acf_iqp_sel_55_repro}, dual-scale)` | Reword: "the corresponding original-OD-scale plots are provided in the supplement (Fig.~A11)." |
| Fig 6 (param_efficiency_pareto) | 755–759 | `Source: \protect\path{figures/param_efficiency_pareto.json}; per-model means from \protect\path{results/matched2000_dualscale.json}, parameter counts from \protect\path{results/model_info.json}.` | **Delete.** |

**Main-paper code-dump count: 7 captions + 1 inline body reference + 1 source comment.**

### Found in supplement (supp_material.tex)

| Figure/Table | Lines | Verbatim trace | Plain-language rewrite |
|---|---|---|---|
| Fig A1 (tstr_crossmodel) | 502–505 | `Source: \protect\path{figures/tstr_crossmodel_matched2000.json}; underlying data \protect\path{results/tstr_matched2000.json}.` | **Delete.** |
| Fig A2 (quantum_circuit) | 583–585 | `Rendered with PennyLane's \texttt{qml.draw\_mpl} from the locked configurations at \protect\path{figures/circuits/<name>.json}.` | Replace with `"Rendered with PennyLane's qml.draw_mpl from the locked per-variant configurations."` Drop file-path glob. |
| Tbl A2 (per_seed_dtw_dominance) | 348–350 | `Source: \protect\path{results/matched2000_dualscale.json}, filtered to \texttt{metric\_name=dtw\_mean} and \texttt{scale=log\_return}.` | **Delete.** |
| (orphan) | 338 | `% source: results/matched2000_dualscale.json#rows filtered to ...` | Drop. |
| Tbl A3 (welch_od_emd) | 389–391 | `Source: \protect\path{results/welch_pairwise.json}, filtered to \texttt{scale=='OD'}.` | **Delete.** |
| (orphan) | 380 | `% source: results/welch_pairwise.json#pairs filtered to ...` | Drop. |
| Tbl A4 (welch_lr_emd) | 438–440 | `Source: \protect\path{results/welch_pairwise.json}, filtered to \texttt{scale=='log\_return'}.` | **Delete.** |
| (orphan) | 423 | `% source: results/welch_pairwise.json#pairs filtered to ...` | Drop. |
| Fig A6 (preprocessing_pipeline) | 652–653 | `Source: \protect\path{figures/preprocessing_pipeline_4panel.json}.` | **Delete.** |
| §A.7 body | 713–714 | `Full per-seed metrics ... are in \path{results/transform_ablation/metrics.csv} and \texttt{summary.md}.` | "Full per-seed metrics and the ablation summary are available in the released artifact set (Data Availability)." |
| Fig A7 (preprocessing_ablation) | 728–731 | `Source: OD-scale panels from \protect\path{results/transform_ablation/metrics.csv}; TSTR-lite $R^2$ panel from \protect\path{results/transform_ablation/tstr_lite.json}.` | **Delete.** |
| §A.7 body | 740–741 | `(\protect\path{archive/qgan_pennylane_SEL.py}; the relevant scaling block carries the explicit code comment ...)` | Reword: "(an earlier reference notebook that magnitude-matches the generator output to unstandardized log-returns)." |
| §A.7 body | 747–748 | `\protect\path{results/matched2000/runs/}` | "the released per-run sample bundles" |
| §A.7 body | 758 | `via a shared helper module \protect\path{_wgan_unscale.py}` | "via a shared inference-time helper module (released with the code)." |
| Fig A8 (training_progression) | 820–821 | `Phase 13 INTRO-01 deliverable. Source: \protect\path{figures/training_progression.json}.` | **Delete** both. "Phase 13 INTRO-01 deliverable" is internal GSD-tracking ID — **must not** appear in published captions. |
| Fig A9 (entanglement_trajectory) | 838–839 | `Phase 13 INTRO-03 deliverable. Source: \protect\path{figures/entanglement_trajectory.json}.` | **Delete.** Same Phase-ID problem. |
| Fig A10 (param_trajectory) | 855–856 | `Phase 13 INTRO-02 deliverable. Source: \protect\path{figures/param_trajectory.json}.` | **Delete.** Same Phase-ID problem. |
| Fig A11 (per_model_loss_grid) | 967–970 | `Source: per-model trajectories from \protect\path{results/matched2000/runs/<model>/42/metrics.json} via \protect\path{figures/loss_<model>.json}.` | **Delete.** |
| §A.9 body | 914–915 | `recorded in \protect\path{figures/loss_ar.json#fit_summary}.` | "recorded in the released artifact set." |
| Figs A12–A20 (reconstruction × 9) | 1038–1039, 1048–1049, 1058–1059, 1068–1069, 1077–1078, 1089–1090, 1098–1099, 1108–1109, 1120–1121 | `Source: \protect\path{figures/reconstruction_<model>.json}.` | **Delete all 9.** |
| §A.10 body | 1131–1133 | `(777 points from \protect\path{data.csv} via \protect\path{core/data.py::compute_log_delta})` and `each model's \protect\path{samples.npy}` | Reword to "from the released log-return series" and "each model's released sample bundle". |
| §A.11 panel descriptors | 1144–1158 | `\texttt{scipy.stats.probplot(dist="norm")}`, `\texttt{ddof=0}`, `\texttt{scipy.stats.kurtosis}`, `\protect\path{core/eval.py::compute_moments}`, `\protect\path{core/eval.py::compute_emd}`, `\protect\path{core/eval.py::compute_jsd}` | Body paragraph, not caption. Keep the *scipy method* references but drop the dotted-source paths (`revision.core.eval.compute_*`). |
| Figs A21–A29 (stat_grid × 9) | 1174–1175, 1184–1185, 1194–1195, 1204–1205, 1214–1215, 1224–1225, 1234–1235, 1244–1245, 1257–1258 | `Source: \protect\path{figures/stat_grid_<model>.json}.` | **Delete all 9.** |
| Figs A30–A38 (dtw_alignment × 9) | 1297–1298, 1306–1307, 1315–1316, 1325–1326, 1334–1335, 1344–1345, 1353–1354, 1363–1364, 1372–1373 | `Source: \protect\path{figures/dtw_alignment_<model>.json}.` plus `(\texttt{dtaidistance}, \texttt{window=500}, \texttt{psi=2}).` | **Delete file paths.** Keep the library/parameter pair for reproducibility but in plain prose: "computed using the `dtaidistance` Python package with `window=500`, `psi=2`." |
| §A.12 intro body | 1282–1283 | `the canonical fastdtw-based LR-DTW reported in main-text \S 4.2 (which is computed under the matched-budget windowed-evaluation protocol per \protect\path{core/eval.py::compute_dtw}).` | Drop dotted path: "per the matched-budget windowed-evaluation protocol (see Data Availability)." |
| §A.8 source comments | 1381–1382 | `% source: results/total_adversarial_param_budget.json#shared_critic_n_params (=250881)` etc. | Drop comments. |
| Fig A39 (shot_noise_robustness) | 1440–1444 | `Phase~12 SENS-01 deliverable. Source: \protect\path{figures/shot_noise_robustness.json}; underlying data \protect\path{results/shot_noise_sensitivity.json}.` | **Delete.** Phase ID + paths. |
| Fig A40 (noise_robustness_quantum) | 1460–1464 | `Phase~12 SENS-02 deliverable. Source: \protect\path{figures/noise_robustness_quantum.json}; underlying data \protect\path{results/noise_model_sensitivity.json}.` | **Delete.** Phase ID + paths. |

**Supplement code-dump count: ~30+ captions and embedded body refs; 5 "Phase XX-YY deliverable" leakage instances (Phase 12 SENS-01, SENS-02, Phase 13 INTRO-01/02/03). The Phase-ID leakage is its own line-item severity: these are internal task-tracking tags that have no place in a journal supplement.**

### Data Availability plan

**Current state.**
- Main paper has a brief §"Data Availability and Reproducibility Statement" at lines 870–871 (one paragraph, between Limitations and §Concluding Remarks). It points to the GitHub repo and a Zenodo DOI placeholder. **It does not enumerate per-figure data files or describe directory layout.**
- Supplement has §A.5 "Code and Data Availability" at lines 509–514. Three sentences. Repo + Zenodo placeholder. **Equally bare.**
- Both sections are too terse to absorb the ~40+ file-path traces being removed from captions.

**Proposed.** Expand the supplement §A.5 "Code and Data Availability" (the natural home for figure-by-figure provenance) with a new subsection. Draft below (verbatim, drop in after the existing §A.5 contents at line 514):

> **Per-figure data provenance.** Every figure and table in the main paper and this supplement is regenerable from the released artifact set. The figure scripts and the JSON/CSV data files they consume are organised under the `results/` tree of the released repository (Zenodo archive at the DOI above). The mapping is:
>
> - Main Figure 2 (training convergence): `figures/training_convergence_all_models.json`; per-seed trajectories under `results/matched2000/runs/<model>/<seed>/metrics.json`.
> - Main Figures 3, 5; Tables 1, 2: `results/matched2000_dualscale.json`, filtered by `model_kind`, `metric_name`, `scale` as appropriate.
> - Main Figure 4 (cross-model ACF): `figures/cross_model_acf_overlay.json`; real-data ACF computed via `revision.core.data.load_and_preprocess`.
> - Main Figure 6 (parameter-efficiency frontier): `figures/param_efficiency_pareto.json`; parameter counts from `results/model_info.json`.
> - Supplementary Figure A1 (TSTR cross-model): `figures/tstr_crossmodel_matched2000.json`; underlying data `results/tstr_matched2000.json`.
> - Supplementary Tables A3–A4 (pairwise Welch tests): `results/welch_pairwise.json`.
> - Supplementary Figures A6–A7 (preprocessing pipeline and ablation): `figures/preprocessing_pipeline_4panel.json`; ablation data `results/transform_ablation/metrics.csv` and `tstr_lite.json`.
> - Supplementary Figures A8–A10 (training progression, entanglement, parameter trajectory): `figures/{training_progression,entanglement_trajectory,param_trajectory}.json`.
> - Supplementary Figure A11 (per-model loss grid): per-model JSONs at `figures/loss_<model>.json`, drawn from `results/matched2000/runs/<model>/42/metrics.json`. AR(2) fit summary at `figures/loss_ar.json#fit_summary`.
> - Supplementary Figures A12–A20 (reconstruction overlays): `figures/reconstruction_<model>.json`.
> - Supplementary Figures A21–A29 (log-return distribution grids): `figures/stat_grid_<model>.json`. Metric implementations live in `core/eval.py` (`compute_emd`, `compute_jsd`, `compute_moments`).
> - Supplementary Figures A30–A38 (DTW alignments): `figures/dtw_alignment_<model>.json`.
> - Supplementary Figure A39 (shot-noise sensitivity): `figures/shot_noise_robustness.json`; underlying data `results/shot_noise_sensitivity.json`.
> - Supplementary Figure A40 (noise-model sensitivity): `figures/noise_robustness_quantum.json`; underlying data `results/noise_model_sensitivity.json`.
> - WGAN sample-space convention and `×10` inverse correction: applied at the `samples.npy` load boundary via the helper `_wgan_unscale.py` (see §A.7).
>
> All script entry points are documented in `revision/README.md`.

This absorbs **every** path that was previously in captions, gives reviewers a single audit point, and is appropriate to the supplement.

---

## C2 — Figure 2–6 size assessment

All five are currently `\begin{figure}[!htbp]` (single-column) with `\includegraphics[width=\columnwidth]{...}`. Visual confirmation from PDF pages 11, 17, 19, 20, 25 below.

### Fig 2 — `figures/training_convergence_all_models.pdf` (`\label{fig:training_convergence_all_models}`)
- Current: `figure` + `\columnwidth`, line 218–220.
- Subpanels: 1 panel, 7 curves + 1 marker, multi-line legend overlaying curves at top-right; log-y; 5-seed ±std bands.
- Verdict (page 11): **illegible.** Legend covers ~30% of plot area, axis ticks are sub-pixel-typeface. The headline "tight cluster of mean OD-EMD ≈ 0.026" is the central finding of §3.1 — invisible at print size.
- **Recommended: `figure*` full-width.** Headline finding, 7-line legend, multi-band std envelopes.

### Fig 3 — `figures/cross_model_dtw_dualscale.pdf` (`\label{fig:cross_model_dtw_dualscale}`)
- Current: `figure` + `\columnwidth`, line 483–485.
- Subpanels: 2 panels (OD scale, LR scale), each is a 9-bar cluster bar chart with model names on x-axis.
- Verdict (page 17): **illegible.** The 9 model x-tick labels (`iqp_sel_55`, `wgan_mlp` etc.) are an unreadable diagonal smudge in both panels. This is the figure that visualises the headline dominance claim.
- **Recommended: `figure*` full-width.** Two side-by-side panels with 9-category x-axes cannot fit in 84 mm.

### Fig 4 — `figures/cross_model_acf_overlay.pdf` (`\label{fig:cross_model_acf_overlay}`)
- Current: `figure` + `\columnwidth`, line 553–555.
- Subpanels: 1 panel, 9 colored ACF curves + real-data dashed reference, 10 lags.
- Verdict (page 19): **legibility marginal.** Curves visible, legend in upper-right corner has ~9 entries small but readable. Lag-1 anomaly markers visible. Still cramped.
- **Recommended: `figure*` full-width OR keep single-column.** This is the borderline case. Has a 9-entry legend and is the headline-corroborating figure for lag-1 ACF. Recommend full-width for safety and consistency with Fig 3/5.

### Fig 5 — `figures/cross_model_emd.pdf` (`\label{fig:cross_model_emd}`)
- Current: `figure` + `\columnwidth`, line 571–573.
- Subpanels: 1 panel, 9-bar cluster chart, OD-EMD with wgan_cnn outlier driving y-axis to 2.0.
- Verdict (page 20): **illegible.** The 9 x-tick labels are unreadable; the bars for 8 of 9 models are visually flat against the y-axis baseline because the wgan_cnn seed-42 outlier extends to 2.0+. The "8 of 9 within ±0.005" claim in the caption is invisible.
- **Recommended: `figure*` full-width AND axis-break or log-y.** Even at full-width the wgan_cnn outlier crushes the rest of the data. Consider a broken y-axis or inset that zooms on the 0.0–0.05 range so the within-cluster separation is visible. At minimum, full-width.

### Fig 6 — `figures/param_efficiency_pareto.pdf` (`\label{fig:param_efficiency_pareto}`)
- Current: `figure` + `\columnwidth`, line 739–741.
- Subpanels: 2 panels (OD scale, LR scale), each a scatter with ~9 points + a diamond marker for the frozen reference, axis-labelled by parameter count.
- Verdict (page 25): **illegible.** Two side-by-side scatters at column-width make markers and axis numerics tiny; the headline upper-tier legend describing marker shapes (circle/square/triangle/diamond) is sub-pixel.
- **Recommended: `figure*` full-width.** Two-panel scatter requires it.

**Summary**: every single one of Figs 2–6 needs to be promoted from `figure` to `figure*` (and `\columnwidth` → `\linewidth` inside it). All five are headline-result figures. The two-panel ones (Figs 3 and 6) are particularly hopeless in single-column. The bar-chart ones (Figs 3 OD-panel, 5) need the full width for x-tick legibility.

### Supplement figures — illegibility / sizing issues

- **Fig A6 `preprocessing_pipeline_4panel.pdf` (line 643, `figure` + `\columnwidth`)**: 4-stage pipeline visualisation in a single column. **Promote to `figure*`** — 4 panels in 84 mm is impossible.
- **Fig A7 `preprocessing_ablation_comparison.pdf` (line 717, `figure` + `\columnwidth`)**: 3-pipeline ablation, multi-bar. **Promote to `figure*`.**
- **Fig A9 `entanglement_trajectory.pdf` (line 825, `figure` + `\columnwidth`)**: 2 stacked panels (entanglement entropy, purity). Borderline; keep single-column **only if** axis labels were rendered at a font size that survives 84 mm; otherwise promote.
- **Fig A10 `param_trajectory.pdf` (line 843, `figure` + `\columnwidth`)**: 2 stacked panels (L2 norm, angle distribution). Borderline; same recommendation as A9.
- **Fig A39 `shot_noise_robustness.pdf` (line 1430, `figure` + `\columnwidth`)** and **Fig A40 `noise_robustness_quantum.pdf` (line 1448, `figure` + `\columnwidth`)**: noise sensitivity figures with multi-line curves over noise levels and two pipelines. **Promote to `figure*`** — these are reviewer R1-M5 deliverables and need to be reader-resolvable.
- **Figs A12–A20 (reconstruction overlays, 9 figures, already `figure*` + `\linewidth`)**: OK structurally, but each has 3 vertically stacked panels of 777-point time series. Visual sanity-check recommended; if the bottom (log-scale OD) panel's y-tick labels are illegible, the 3-panel stacking needs taller aspect.
- **Figs A21–A29 (stat grids, 9 figures, already `figure*` + `\linewidth`)**: 6-panel grids. OK as structured but at full-width across a 4-column-wide layout (since this is a `figure*` in a two-column doc) each panel ends up ≈ 27 mm wide — verify the moment bar-chart's numeric labels are legible.
- **Figs A11 (per-model loss grid, `figure*` + 0.24\linewidth × 8 subfigures)**: 8 thumbnails at 24% of full-width = ~41 mm each. This is the grid that supports the "WGAN-CNN critic drift" diagnostic. Subpanels are almost certainly illegible. **Recommend: 4×2 grid at 0.48\linewidth (82 mm each) or 2-page spread.**
- **Figs A30–A38 (DTW alignments, 9 figures, `figure*` + `0.85\linewidth`)**: OK structurally.

---

## C3 — Other findings

### Caption length and informativeness
- Most captions are appropriately self-contained (good — the per-model panel captions in the supp correctly cross-reference the parent figure for the panel-layout description).
- Several main-text captions are **too long** because they re-state §4.1 prose verbatim. Tbl 2's caption (lines 385–406) is 22 lines and re-explains the VAE characterization and Welch test rationale already in the prose. Cut by ~50%.
- Fig 2's caption (lines 221–240) explains the "visible deep dip" (wgan_cnn seed-42 outlier) — useful — but also re-states the VAE/AR(2) exclusion rationale that the figure does not show. Trim.

### Subpanel labels (a)/(b)/(c)/...
- Fig A11 caption (line 960) references "(a)–(d)" for the quantum and "(e)–(g)" for the classical — but the subfigure environments at lines 925–959 have no `\label` and the subcaptions are just the model identifier (`IQP:SEL (55p, Q)` etc.). No `(a)` etc. is printed under each panel. The body §A.9 text at lines 876, 882, 897 then references `\ref{fig:per_model_loss_grid}a`, `f`, `h`. **Broken cross-reference.** Either add `\label{a}` style sub-labels with `\thesubfigure` or rewrite the body refs to name the model directly (e.g., "the WGAN-CNN panel (Fig.~A11)").

### Colormap / accessibility
- Could not verify all per-model figures, but cross-model overlay figures (Fig 4 ACF overlay) appear to use a **rainbow-style multi-category palette** (visible on page 19). Recommend swap to a categorical-safe ColorBrewer palette (e.g., Set1 or Dark2) or a perceptually-uniform sequential for ordered categories. The 9-curve overlay with VAE highlighted is a known color-blindness failure mode if relying on hue alone — make sure VAE is distinguishable by line style as well.
- Fig 5 (cross_model_emd) bar chart on page 20: the wgan_cnn bar (which carries the headline outlier story) and the other 8 bars need printer-grayscale separation; could not verify from screen rendering. Likely currently distinguished only by hue.

### Post-14-21 framing accordance
- Captions of Figs 3, 4, 5 all explicitly state the quantum-cluster dominance finding in the post-14-21 framing (e.g., Fig 3: "the quantum cluster ... is below the WGAN cluster"; Fig 4: "quantum-cluster mean ... is closer to real than any classical-baseline mean"; Fig 5: "quantum-cluster mean is ≈ 0.0288 ... WGAN cluster mean is ≈ 0.3312"). No stale "bifurcated finding" framing detected in captions.
- Fig 6 caption is neutral and consistent.
- Fig 2 caption uses "tight cluster of mean OD-EMD ≈ 0.026" — consistent with post-14-21 cluster framing.
- **No caption-level framing stalenesses found.** The reframing pass appears to have reached every caption.

### Phase-ID leakage (severity flag)
Captions on Figs A8, A9, A10, A39, A40 contain internal Phase/task IDs ("Phase 13 INTRO-01 deliverable", "Phase 12 SENS-01 deliverable", etc.) — these are GSD planning-system tags that have no place in a journal supplement. **Mandatory removal** independent of the path-trace fix.

---

## C4 — Total fix burden estimate

**File-path move-out:**
- Main paper: 7 caption traces + 1 inline §4.1 body sentence + 1 `% source` comment ≈ 9 edits, single text block in each (~30 lines edited).
- Supplement: ~38 caption traces (including 9× reconstruction, 9× stat_grid, 9× dtw_alignment, plus 11 unique-figure traces and 5 Phase-ID-tagged ones) + 6 inline §-body path mentions + several `% source` comments ≈ 50+ edits (~100 lines edited).
- New Data Availability subsection: insert at supp line 514, ~30 lines drafted above.
- Many of the per-model reconstruction/stat_grid/dtw_alignment traces are template-substitutable with a find-replace pass (`Source: \protect\path{figures/<X>.json}.` → empty). Bulk-deletable.

**Figure-size adjustments:**
- Main: 5 `\begin{figure}` → `\begin{figure*}` + 5 `\columnwidth` → `\linewidth` (Figs 2–6). 10 edits.
- Supplement: A6, A7, A39, A40 (4 promotions). A9, A10 inspect-then-decide (2 conditional). A11 grid restructure (1 layout edit). 5–7 edits.
- Total figure-env edits: ~17.

**Wall-clock estimate (15-min increments):**

| Task | Time |
|---|---|
| Find-replace pass on all caption `Source: \protect\path{...}` traces (template-form) | 30 min |
| Hand-edit non-template captions (Fig 4 dotted method ref, Fig A2 PennyLane glob, §A.7 archive ref, etc.) | 30 min |
| Phase-ID leakage removal (5 captions) | 15 min |
| Draft + insert expanded Data Availability subsection in supplement §A.5 | 30 min |
| Inline §4.1 body and §A.7 / §A.9 / §A.10 / §A.12 body path-reference rewrites | 30 min |
| Promote Figs 2–6 to `figure*` (and \columnwidth → \linewidth) | 15 min |
| Promote supplement A6, A7, A39, A40 and inspect A9/A10/A11 | 30 min |
| Restructure Fig A11 8-panel loss grid (4×2 layout) | 30 min |
| Re-compile and verify legibility, fix overflows/floats | 45 min |
| Drop `% source` orphan comments (cleanup) | 15 min |
| Fig A11 subfigure (a)/(b)/(c) label fix + body-cross-ref reconciliation | 30 min |
| Caption-length trim (Fig 2 and Tbl 2 long captions) | 30 min |

**Total: ~5.5 hours.** All achievable inside one focused session before the 2026-06-17 deadline. The find-replace bulk-deletion is the biggest one-shot reduction; the figure-size promotions are mechanical; the new Data Availability paragraph is the single highest-leverage edit (one location absorbs ~40 fragments).

---

## Key files (absolute paths)
- Manuscript: `/Users/shawngibford/dev/phd/qGAN/main (4) copy.tex`
- Supplement: `/Users/shawngibford/dev/phd/qGAN/supp_material.tex`
- Compiled PDF inspected: `/Users/shawngibford/dev/phd/qGAN/main (4) copy.pdf` (pages 11, 17–20, 25)
- Existing main-paper Data Availability: `main (4) copy.tex` lines 870–871
- Existing supp Data Availability (§A.5): `supp_material.tex` lines 509–514 — primary expansion target
