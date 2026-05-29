# Agent 3 — Figure↔Caption Correspondence Findings

**Audit target:** HEAD `50658a6` (v1.2.1)
**Files audited:** `main (4) copy.tex`, `supp_material.tex`, all 11 figure files + 4 sidecar `.json` metadata files
**Scope:** Figure-to-caption correspondence, source-data attribution accuracy, stale-figure detection, prose↔caption consistency.

## Summary
- **BLOCK:** 0 findings
- **FLAG:** 1 finding
- **NIT:** 3 findings

All 11 figures exist at their cited paths. All 4 revision-era PDFs have sidecar JSONs whose `source` fields resolve correctly (`matched2000_dualscale.json` and `transform_ablation/metrics.csv` confirmed present). No stale single-model figures detected (the 5 removed in audit-cleanup — `dtwd.png`, `pdf.png`, `cdf.png`, `qq.png`, `acf.png` — are not referenced). Hybrid-GAN (Fig A2) is correctly qualified as "proposed". Decision-tree triage (Fig A5) is correctly framed as "future work in §4.5 Outlook". Pipeline B figure (Fig A7) makes no Lambert W mention.

## Per-figure verdict table

| Figure | File | Verdict | Notes |
|--------|------|---------|-------|
| Fig 1 | `concept_diagram.png` | OK | Generation→Monitoring→Optimization→Feedback loop with Process/Data/Results/Analysis/Decision nodes; caption "Conceptual diagram of how we envision using quantum generative AI within industrial bioprocess engineering" matches visual content. |
| Fig 2 | `revision/results/figures/cross_model_dtw_dualscale.pdf` | OK | Two-panel bar chart with OD scale (left) and LR scale (right), n=5 seeds error bars, 9 generators labeled (IQP:SEL 55p, V1, V2, V3, WGAN-GP MLP/CNN/LSTM, VAE, AR(p)). Numerical values cited in caption (V1=0.94, V2=0.95, iqp_sel=0.99, V3=1.12, wgan_lstm=1.58, wgan_mlp=2.62, wgan_cnn=6.86, AR=7.70, VAE=0.0876) match sidecar JSON exactly. Source `matched2000_dualscale.json` resolves. |
| Fig 3 | `revision/results/figures/cross_model_acf_overlay.pdf` | OK (NIT) | Lag 0–9 ACF overlay for all 9 generators + dashed black real-data reference. Real-data lag-1 = −0.0641 in JSON, caption says −0.064. Per-model mean values match JSON. See N-1 below for minor unit-label nit. |
| Fig A1 | `classicalgan.png` | OK | Generator/Discriminator schematic with Z input, gradient feedback paths; caption generic but accurate. |
| Fig A2 | `hybridgan.png` | OK | Hybrid data-driven + physics-informed model schematic with GAN inset; caption "QGAN and synthetic data implementation within a hybrid-model approach to process monitoring" — proposal-qualified context throughout §A.3 ("proposed", "not implemented"). Hard-prohibition compliance verified. |
| Fig A3 | `mech_rep.png` | OK | Balance↔Constitutive↔Constraint equations triangle, reprinted-permission attribution present. |
| Fig A4 | `quantum_circuit.png` | OK | 5 qubits (rows 0–4), each starts H, RZ, RZ, Rot followed by ladder CNOTs, then Rot, more CNOTs, terminal RX, RY, measurement — matches Section 3 architecture description (IQP encoding + strongly entangling layers + measurement-prep rotations). |
| Fig A5 | `bpm_qgan.drawio.png` | OK | 8-step decision-tree triage flowchart (Step 1 Process Control, Step 2 Monitoring, Step 3 Exit, Step 4 Add Sensor, Step 5 Mechanistic, Step 6 Data-Driven, Step 7 Quantum Synthetic, Step 8 Real-time) with 4 decision diamonds. Caption "Decision-tree triage schematic… outlined as future work in §4.5 Outlook of the main text. Not an empirical contribution of the present study." — hard-prohibition compliance verified. |
| Fig A6 | `lucy_diagram.jpg` | OK | 20L photobioreactor schematic with OD/pH/DO/temperature/PAR sensors, pumps, harvest bottle; caption matches. |
| Fig A7 | `revision/results/figures/preprocessing_pipeline_4panel.pdf` | OK (NIT) | 4-panel sequence: raw OD (n=778), log-returns (n=777), standardized (μ=0,σ=1), rescaled to [−1,1]. JSON `pipeline: "B"`, no Lambert W. Hard-prohibition compliance verified. See N-2 for rendering nit. |
| Fig A8 | `revision/results/figures/preprocessing_ablation_comparison.pdf` | OK (FLAG) | 4-panel grid: OD-EMD, OD-ACF lag-1, OD-DTW, TSTR-lite R² — bars for A/B/C pipelines, 5 seeds, 1000 epochs, B selected, C dropped. JSON `source` points to `revision/results/transform_ablation/metrics.csv` but caption cites only that CSV (correct). See F-1 below for source-attribution mismatch between caption and figure rendering. |

## FLAG findings

### F-1: Fig A8 caption omits the second derived data source actually rendered in the figure
- **Figure:** Fig A8 (`revision/results/figures/preprocessing_ablation_comparison.pdf`)
- **Caption excerpt:** "Source: `revision/results/transform_ablation/metrics.csv`."
- **Figure actual content:** Sidecar JSON `source` field declares **three** input artifacts: `metrics.csv` (the long-form sweep), `seed_spread.json`, and `tstr_lite.json or summary.md fallback` for the TSTR-lite R² panel. The TSTR-lite R² panel in the figure draws from a different file than the OD-scale panels.
- **Discrepancy:** The caption attributes the entire figure to `metrics.csv` only, but the TSTR-lite R² panel (bottom-right) was rendered from `tstr_lite.json` per the sidecar metadata. A reader trying to reproduce the bottom-right panel from `metrics.csv` alone will not find the LSTM R² column there.
- **Suggested fix:** Append to the caption's source line: "; TSTR-lite R² panel from `revision/results/transform_ablation/tstr_lite.json` (per `summary.md` fallback)."

## NIT findings

### N-1: Fig 3 caption ACF lag-1 real value precision mismatch with sidecar
- **Figure:** Fig 3 (`revision/results/figures/cross_model_acf_overlay.pdf`)
- **Caption excerpt:** "the real-data ACF shown as a dashed reference (lag-1 $= -0.064$, matched-pipeline)"
- **Figure actual content:** Sidecar JSON `real_acf[1] = -0.06411182880401611`; caption rounds to −0.064 (correct to 2 sig figs). Main-text prose at line 479 uses identical "−0.064". The figure embedded annotation says "real-data reference: dashed black; VAE lag-1 anomaly visible at lag=1".
- **Discrepancy:** None — but the rounded value −0.064 is propagated identically through main prose (line 479), caption (line 507), and ACF overlay caption — confirm consistent.
- **Suggested fix:** No change required; flagged only to verify the unrounded JSON value (−0.06411…) is intentionally rounded.

### N-2: Fig A7 panel titles overlap in PNG preview
- **Figure:** Fig A7 (`revision/results/figures/preprocessing_pipeline_4panel.pdf`)
- **Caption excerpt:** "Four-stage preprocessing pipeline (Pipeline~B, native): raw OD ($n=778$), log-returns…"
- **Figure actual content:** In the rendered PNG preview, the top-of-figure title text "Preprocessing pipeline (Pipeline B): raw OD to model-ready [−1, 1] in 4 panels" overlaps with the Panel 1 title "Panel 1: Raw OD (arbitrary units, 778 samples)". The PDF version likely renders correctly with `bbox_inches='tight'` but the PNG sidecar shows visible text collision.
- **Discrepancy:** Cosmetic — the PDF (cited in `\includegraphics`) likely renders cleanly; this is only flagging that the PNG preview at `preprocessing_pipeline_4panel.png` has a layout artifact.
- **Suggested fix:** No paper-text change needed. If a future re-render is done, add `plt.tight_layout()` or increase the top margin so the suptitle doesn't collide with Panel 1's axes title.

### N-3: Fig A2 caption could more strongly signal "proposed" status
- **Figure:** Fig A2 (`hybridgan.png`)
- **Caption excerpt:** "QGAN and synthetic data implementation within a hybrid-model approach to process monitoring."
- **Figure actual content:** Schematic showing data-driven model + physics-informed model + GAN inset with generator/discriminator and "Valid/Invalid" check.
- **Discrepancy:** The caption itself does not contain the words "proposed", "not implemented", "aspirational", or "future". Hard-prohibition compliance is achieved *only* through the surrounding §A.3 prose (which is explicit and repeated). A reader scanning the figure list independently would not see the qualification.
- **Suggested fix:** Append "(proposed extension; not implemented or evaluated in this study — see §A.3)" to the caption for defensive in-caption qualification.

## Cross-cutting verifications (all passed)

- All 11 `\includegraphics{...}` paths resolve to files on disk.
- All 4 revision-era PDFs have sidecar `.json` metadata.
- All cited source-data paths in captions (`matched2000_dualscale.json`, `cross_model_acf_overlay.json`, `preprocessing_pipeline_4panel.json`, `transform_ablation/metrics.csv`) exist.
- The 5 stale single-model figures removed in audit-cleanup (`dtwd.png`, `pdf.png`, `cdf.png`, `qq.png`, `acf.png`) are **not** referenced anywhere in main or supp.
- Fig 2 numerical values in caption (V1=0.94, V2=0.95, iqp=0.99, V3=1.12, wgan_lstm=1.58, wgan_mlp=2.62, wgan_cnn=6.86, AR=7.70, VAE=0.0876) match `cross_model_dtw_dualscale.json` to 2 decimal places.
- Fig 3 quantum-cluster lag-1 band ("−0.09 to −0.10") matches JSON per_model_mean_acf: V1=−0.0997, V2=−0.0968, V3=−0.0895, iqp=−0.0949 — all within the cited band.
- Fig A7 caption says "Pipeline B, native" — JSON confirms `pipeline: "B"`, no Lambert W. Hard prohibition #3 satisfied.
- Fig A5 caption says "Decision-tree triage schematic… outlined as future work in §4.5 Outlook". Hard prohibition #2 satisfied. Main text line 669 and 759 reinforce non-empirical-contribution framing.
- Fig A2 surrounding §A.3 prose includes "is a proposed extension and was not implemented, trained, or evaluated in this work" (line 153). Hard prohibition #1 satisfied (via prose, not caption — see N-3).
- Cross-model sources (Fig 2, Fig 3): both correctly cite `matched2000_dualscale.json` as source. Hard prohibition #4 satisfied.
