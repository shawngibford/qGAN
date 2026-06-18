# Paper Submission Handoff — Wave 8 (Human ACT)

**Status as of 2026-05-28**: Paper-rewrite swarm complete + post-swarm audit-cleanup session complete + read-through complete. Manuscript is submission-ready and tagged `v1.2` at commit `a4cfc1a`. All gates green except the expected `(d) release.md` block (deferred to journal acceptance per plan 14-07). Only `git push` of main + tag, AIChE-portal upload, and GitHub release-notes update remain.

**Deadline**: ≈ 2026-06-17 (three-week extension granted 2026-05-27; ~20 days remaining as of 2026-05-28).

> If you're a fresh session resuming this work: **read this doc end-to-end first**, then `git log a0f932b..HEAD --oneline` to see the full trail (swarm + audit cleanup), then run the verification gates in §2.3. **Do NOT re-execute the swarm OR the audit cleanup** — both are already done. Don't re-introduce the corrections they made (catalogued in §5).

---

## §0 — Resume in 90 seconds

| Field | Value |
|---|---|
| Repo | `/Users/shawngibford/dev/phd/qGAN/` |
| Branch | `main`, 18 commits ahead of `origin/main` |
| Manuscript | `main (4) copy.tex` (filename has literal space + paren — quote in shell) |
| Supplement | `supp_material.tex` |
| Bibliography | `bib.bib` (59 entries; `yoon2019TimeGAN` re-typed as @inproceedings with pages added) |
| Last commit | `a4cfc1a chore(paper-rewrite): bundle 7 legacy figures into repo + clear 10 audit FLAGs` |
| Tag | `v1.2` (local, not yet pushed) |
| Provenance gate | v2.2, PASS — **143 main + 156 supp** literals all resolve to `results/*.json` |
| pdflatex compile (no TEXINPUTS needed) | PASS — **52 pages**, 0 undefined cites, 0 undefined refs, **0 hyperref duplicate-destination warnings**, all 11 figures in PDF render from repo-local files |
| Audit verdicts (all 4 sub-audits) | LaTeX/bib: SUBMISSION-READY • Cleanliness/framing: SUBMISSION-READY • Figure verification: SUBMISSION-READY • Prohibition sentinel: SENTINEL-CLEAN |
| Working tree | Clean |

**Next action**: `git push origin main` + `git push origin v1.2` → AIChE portal upload (instructions in §2.4) → GitHub release-notes update.

---

## §1 — What the swarm produced

The paper now leads with **Finding 2** (uniform quantum dominance on log-return temporal alignment + lag-1 ACF) with **Finding 1** (OD-marginal non-significant under low power) and **LR-EMD asymmetry** (quantum is worse than every classical adversarial baseline on the LR marginal) as scope-honest caveats.

### What changed, section by section

| Section | Before swarm | After swarm |
|---|---|---|
| **Title** | "Quantum Synthetic Data Generation for Industrial Bioprocess Monitoring" | Unchanged (W1 proposed 3 rescope candidates; deferred to human — see §6) |
| **Abstract** (line 49) | "fidelity comparable to size-matched classical baselines" (unhedged parity framing) | 146 words; leads with Finding 2 (LR-DTW 0.94–1.12 vs 1.58–6.86; lag-1 ACF cluster −0.089 to −0.100 vs real −0.064); Finding 1 follows with full scope hedge (Welch p > 0.36, max \|Cohen's d\| ≤ 0.65, n=5 power ≈ 15%, TOST not satisfied); proof-of-concept positioning preserved |
| **Plain Language Summary** (line 59) | "developed a quantum computing method to create realistic artificial data... industrial biotechnology" | 242 chars; "reproduced log-return temporal-alignment cost and one-step autocorrelation more faithfully than parameter-matched classical adversarial baselines, while overall single-point distributions were statistically indistinguishable" |
| **§1.4 Principal Contributions** (lines 95–111) | 4 bullets, all parity framing; "closed-loop deployment" outlook reference | 5 bullets; new "Bifurcated Empirical Finding" bullet before Empirical Evaluation; Empirical Evaluation bullet rewritten with full scope hedge; "future closed-loop deployment" → "future decision-tree triage workflow" |
| **§3 Methods** (lines 155–315) | Matched-budget mentioned implicitly | Full training protocol now spelled out: 2000 epochs, n=5 seeds {42-46}, shared 250881-param critic, Adam (β₁=**0.0**, β₂=0.9), LR_gen=6.9173e-5, LR_critic=1.8046e-5, n_critic=9, λ_gp=2.16, batch=12. Pipeline B explicitly defined. |
| **§4.1 Cross-Model Comparison** (new subsection) | Single-model diagnostics only (IQP:SEL_55 only) | New subsection centralizes the bifurcated finding: LR-DTW dominance + LR-EMD asymmetry + hedged lag-1 ACF (per-seed overlap acknowledged) + VAE-as-degenerate-regime + OD-marginal scope hedge (Welch + power + TOST) + OD-EMD pipeline-invariance note. Existing single-model diagnostics preserved. |
| **Table 1** (`tbl:eval_scale`) | Legacy seed-42 quantum representative values (LR-EMD=0.1209 — **8× off §4.1's headline**) | Regenerated from `matched2000_dualscale.json#iqp_sel_55_repro` as 5-seed mean ± sample-std at 4-decimal precision; source-declaring caption; LR-EMD cell now matches §4.1 (0.01497 ± 0.00020) |
| **§4.2 Key Contributions** (lines 642+) | "fidelity comparable to" / "comparable to, but do not exceed" / stale matched-capacity-ablation caveat | Leads with bifurcated finding; "exceed" disambiguated as per-seed (LR-DTW) vs mean-level (lag-1 ACF); stale "ablation needed" caveat removed (wgan_mlp/cnn/lstm at 73–78 params ARE the matched-capacity comparators) |
| **§4.3 Implications** | "do not provide evidence of a computational advantage at matched capacity" + "high-fidelity synthetic data... enables soft sensors" | Rewritten to acknowledge LR-DTW + lag-1 ACF benefit on this dataset; soft-sensor claim reframed as future work ("demonstrating downstream utility requires multivariate process data") |
| **§4.4 Limitations** | Hardware + single-variable scope only | Adds explicit "Scope of the matched-budget finding" itemize listing 5 things the protocol does NOT establish (LR-marginal asymmetry, lag-1 only, n=5 power limits, 5-qubit regime, comparator set) |
| **§4.5 Outlook** | "Closed-loop decision-driven pipeline" + Hybrid-GAN | "Decision-tree triage workflow" (renamed); explicit "Conditions under which the LR-DTW + lag-1 ACF finding is expected to extend" itemize (multivariate / longer-series / higher-qubit-count / larger-seed-budget) |
| **§5 Concluding Remarks** | "fidelity comparable to" + "whether any quantum-specific benefit exists" (under-claim) | Opens by answering the §1.4 falsifiable question with the exceed/match/fall-short trifurcation; dataset envelope named (778 OD points, 384 windows, 1 campaign, 5 qubits) |
| **3 new figures inserted** | — | `cross_model_dtw_dualscale` (main §4.1), `cross_model_acf_overlay` (main §4.1), `preprocessing_pipeline_4panel` (supp §A.7) |
| **Supp §A.7** | preprocessing prose only | Adds preprocessing_pipeline_4panel figure |
| **Supp §A.5 caption** (Figure A5) | "decision-driven workflow ... unified feedback loop" | "decision-tree triage schematic outlined as future work in §4.5 Outlook" (matches main demote) |
| **Supp DTW subsection** (around line 307) | (no reconciliation note) | Adds reconciliation note explaining 0.6843 pre-v1.0 vs 0.302 matched-budget gap (Pipeline A vs B, single-seed best-case vs 5-seed mean, different epoch budget) |
| **Supp label typos** | `appraoch`, `schemcatic` | `approach`, `schematic` |

### Commit trail (swarm-era, 10 atomic commits)

```
a50cb0f  refactor(paper-rewrite): A6 style polish + structural cleanup
94ea5a0  fix(paper-rewrite): correct Table 1 to matched-budget aggregates + AR(2) attribution
0f47c25  refactor(paper-rewrite): align back-matter (§4.2–§5) + insert preprocessing figure
2123a06  docs(decisions): correct Adam beta_1 0.5 -> 0.0 per JSON ground truth
ee31a41  refactor(paper-rewrite): spell out matched-budget training protocol + fix supp typos
3e0cd2d  refactor(paper-rewrite): disclose LR-EMD asymmetry + hedge lag-1 ACF + OD-pipeline note
d03d35f  refactor(paper-rewrite): centralize bifurcated finding in §4.1 + insert cross-model figures
771d338  fix(prov-gate): v2.2 — split LaTeX en-dash ranges before tokenization
277dec4  refactor(paper-rewrite): reframe abstract + §1.4 with Finding-2 lead
d81306f  docs(14): pin §7 framing decisions for paper-rewrite swarm
```

---

## §1A — What the post-swarm audit-cleanup session (2026-05-28) added

The post-swarm read-through surfaced a regression the swarm's provenance gate
couldn't catch (the gate checks numeric literals, not prose descriptions of
preprocessing), prompted addition of two reviewer-ergonomics artefacts (Table 2 +
per-seed dominance table), and prompted a four-agent parallel audit that
surfaced 4 BLOCK + 10 FLAG findings. All addressed.

### What changed, by category

| Category | Before audit | After audit |
|---|---|---|
| **Lambert W preprocessing** | Methods §3.2 + supp §A.7 described "Pipeline B" using a chain that included inverse Lambert W heavy-tail correction. **This was wrong** — D-10-05 dropped Pipeline C (the Lambert path); matched-budget runs use Pipeline B = log-returns → standardize → rescale to [−1,1], NO Lambert. The preprocessing figure was Pipeline C. | All Lambert W mentions moved into the explicit "Pipeline C dropped per D-10-05" rationale paragraph. New supp §A.7 ablation paragraph + new ablation figure (`preprocessing_ablation_comparison`) show the 5-seed A/B/C comparison that motivated D-10-05. Preprocessing figure regenerated as Pipeline B (panel 4 now shows rescale-to-[−1,1], not Lambert correction). |
| **Cross-model summary table** | §4.1 prose + Figs 1+2 contained the cross-model evidence but no consolidated table. Table 1 was single-model-only (IQP:SEL on both scales). Readers scanning tables first would find no cross-model summary. | New **Table 2** (`tbl:cross_model_comparison`) — full-page `table*`, 5 metrics × 9 generators, 5-seed means with bolded per-row leaders. Bifurcated finding visually obvious: quantum (V1/V2/V3) wins 3 temporal-structure rows; AR(2) wins LR-EMD; VAE wins OD-EMD. Cited 3 places in main text. |
| **Per-seed LR-DTW dominance** | §4.1 + §4.2 claim "no quantum-classical seed overlap on LR-DTW" was assertion-only; reviewers had to trust the prose or load `matched2000_dualscale.json`. | New supp Table A.X (`tbl:per_seed_dtw_dominance`) — 5 rows × 4 cols (Seed, Worst Q (V3), Best C (wgan_lstm), Gap). Verified 60/60 cells satisfy quantum < classical; tightest margin is seed 46 at 0.205 (≈16% relative). |
| **Pairwise statistical evidence** | §4.1 OD-EMD null and LR-EMD reversal stated in aggregate ("Welch p > 0.36"; "every classical adversarial baseline outperforms every quantum variant"). The 40 per-pair Welch tests lived only in `welch_pairwise.json`. | New supp Tables A.X+1 (OD-EMD) and A.X+2 (LR-EMD) — 20 pairs each, 7 cols (Quantum, Classical, Mean_q, Mean_c, Welch p, Cohen's d, MWU p). OD: all 20 pairs p > 0.36, \|d\| < 0.65. LR: 17 of 20 pairs p < 0.001, d ranges from +2.15 to +151.5 (positive d = quantum loses), with negative d only for quantum-vs-VAE pairs. |
| **§4.2 contribution order** | Led with "First, we outline a decision-tree triage workflow… not evaluated empirically here" — a future-work disclaimer in slot 1. | Reordered: QWGAN-GP architecture (1), bifurcated finding (2, cites Table 2 + Figs 1+2), open science (3), decision-tree as future-work organising concept (4). Matches §1.4's correct structure. |
| **Stale single-model figures** | Main carried 5 single-model diagnostic figures (Figs 4–8: `dtwd.png`, `pdf.png`, `cdf.png`, `qq.png`, `acf.png`) from `~/Documents/main_qgan/`, dated Oct 20 2025. Embedded chart titles ("Lucy Log Returns", "Log δ" notation) were from the pre-revision single-best-seed Pipeline-A era; captions had been retrofitted to claim matched-budget multi-seed framing. | All 5 figures removed. Replaced with one bridging paragraph deferring per-model diagnostics to Table 2 + Figs 1+2 (which now carry the cross-model evidence). PDF dropped from 54→52 pages. |
| **Table column overflow** | 4 tables declared `\begin{table}[h]` in twocolumn layout overflowed column width by 78–182pt (visible text spillover): Table 1, per-seed dominance, both Welch tables. | All 4 changed to `\begin{table*}[!htbp]` (full-page width spanning both columns). Overflows resolved. |
| **Legacy-figure packaging** | All 12 legacy figures resolved via local `TEXINPUTS=…~/Documents/main_qgan/…` — paths AIChE compile won't have. | After Figs 4–8 removal, 7 legacy figures remain; all 7 copied into repo root. Compile works without `TEXINPUTS`. |
| **Hyperref duplicate-destination warnings** | 8 `pdfTeX warning (dest): name{equation.N} has been referenced but does not exist` warnings from supp re-using main-text equation anchors. | Added `\theHfigure / \theHtable / \theHequation` disambiguators after supp counter reset. All 8 warnings gone. |
| **`\texttt{revision/.../*.json}` caption overfulls** | Long-path `\texttt{}` cites in 14 figure/table captions didn't break at `/`, causing 6 caption-line overfulls including 95pt on Table 2. | All 14 instances converted to `\path{revision/...}` (breaks at slashes; verbatim, no `\_` escaping needed). Overfulls resolved. |
| **OD-EMD comparator-set scope** | Abstract / §1.4 / §4.2 / §5 wrote "Welch p > 0.36 between quantum and parameter-matched classical *adversarial* baselines" but the 20-pair test in §4.1 includes VAE + AR(2) — scope drift. | Each section's OD-EMD claim updated to "full set of parameter-matched classical comparators (adversarial baselines plus VAE and AR(2))". Matches `welch_pairwise.json` scope. |
| **Minor cleanliness drift** | §4.3 "demonstrated here … are prerequisites for downstream applications"; supp §A.3 "deployed QWGAN-GP"; §1.4 bullet 3 carried AR(2) reference parenthetical that abstract/§4.2/§5 omitted. | §4.3: "demonstrated here" → "reported here", "are prerequisites for" → "may be relevant to". §A.3: "deployed" → "evaluated in this study". §1.4: AR(2) parenthetical removed for symmetry. |
| **Bib entry quality** | `yoon2019TimeGAN` was `@inbook` with no `chapter` or `pages` (had only `articleno` + `numpages`) → BibTeX warning. | Re-typed as `@inproceedings` with `pages = {5508--5518}` (NeurIPS 2019). Warning gone. |

### Commit trail (audit-cleanup era, 8 atomic commits)

```
a4cfc1a  chore(paper-rewrite): bundle 7 legacy figures into repo + clear 10 audit FLAGs
b2ceb43  refactor(paper-rewrite): remove Fig 4 (dtwd.png) — last stale single-model figure
efb05c7  refactor(paper-rewrite): remove stale single-model diagnostic figures + fix table column overflow
c3a1733  feat(paper-rewrite): add pairwise Welch + Cohen's d tables for OD-EMD and LR-EMD to supplement
f7e5dff  feat(paper-rewrite): add per-seed LR-DTW dominance table to supplement
80ad0a6  refactor(paper-rewrite): reorder §4.2 contributions — lead with QWGAN-GP, end with decision-tree future-work
e28fb49  feat(paper-rewrite): add Table 2 cross-model comparison with bolded row-leaders
2cdb558  fix(paper-rewrite): remove Lambert W (Pipeline C, dropped per D-10-05) from manuscript + figures
```

(plus `69c077e docs(14): post-swarm handoff update — Wave 8 submission readiness` between the two eras, and this commit when the present handoff update is committed.)

---

## §2 — Wave 8: human ACT checklist

### 2.1 — End-to-end manuscript read-through ✓ COMPLETE (2026-05-28)

The full main + supp read-through was completed in the audit-cleanup session, covering all 14 sections below. Each was verified clean against the prohibition list (§5), the JSON data sources, and cross-section consistency. The 14 cleanup-driven changes catalogued in §1A above were applied during the read-through.

If you need to re-do the read-through later, use this section list as the index:

```
main (4) copy.tex:
  Abstract (lines 47–50)       — leads Finding 2, Finding 1 + LR-EMD scope-hedged
  Plain Language Summary (58–59)
  §1.4 Principal Contributions (95–111)
  §3 Methods                    — full training protocol + Pipeline B definition (NO Lambert W)
  §4.1 Cross-Model Comparison   — bifurcated finding centralized + Table 2 cross-model summary
  Table 1                       — single-model dual-scale (IQP:SEL only); LR-EMD = 0.0150 ± 0.0002
  Table 2 (NEW)                 — cross-model 9-generator × 5-metric, bolded row-leaders
  §4.2 Key Contributions        — QWGAN-GP first, decision-tree last (reordered)
  §4.3 Implications
  §4.4 Limitations              — "Scope of the matched-budget finding" itemize
  §4.5 Outlook                  — "Decision-tree triage workflow" + "Conditions to extend" itemize
  §5 Concluding Remarks         — opens with §1.4 falsifiable-question answer

supp_material.tex:
  §A.3 Hybrid-GAN               — 11 mentions, all qualified as "proposed/not implemented"
  §A.4 Validation Metrics       — DTW reconciliation note (0.6843 pre-v1.0 vs 0.302 matched)
                                  + Table A.X per-seed LR-DTW dominance (NEW)
                                  + Tables A.X+1/A.X+2 pairwise Welch on OD/LR-EMD (NEW)
  §A.5/A.6 Figure A5 caption    — "decision-tree triage schematic"
  §A.7 Data Transformation      — preprocessing_pipeline_4panel (Pipeline B, no Lambert)
                                  + preprocessing_ablation_comparison (NEW; A/B/C 5-seed)
                                  + "why no Lambert W" ablation rationale paragraph
```

### 2.2 — Eyeball checks ✓ ALL PASSING at HEAD (a4cfc1a)

- **Table 1 LR-EMD cell** reads `0.0150 ± 0.0002` (4-decimal rendering of 0.01497 ± 0.00020 from JSON). The legacy value `0.1209` is absent.
- **Table 1 ACF lag-1 mean cell** reads `-0.0949 ± 0.0092` (4-decimal rendering of -0.09490 ± 0.00923). The legacy `-0.0814` is absent.
- **Table 2** (`tbl:cross_model_comparison`) is in §4.1 immediately after Table 1, full-page-width, 9 generators × 5 metrics with bolded row-winners. Row-winner pattern: V3 (lag-1 ACF), V1 (LR-DTW), V2 (OD-DTW), VAE (OD-EMD), AR(2) (LR-EMD); VAE LR-DTW dagger-marked as degenerate-regime exclusion.
- **Abstract "1.58 – 6.86"** (not "1.58 – 7.70" — that conflates AR(2) with adversarial baselines).
- **§4.5 Outlook header** reads "Decision-tree triage workflow" (NOT "Closed-loop decision-driven pipeline").
- **§5 first sentence** answers the §1.4 falsifiable question with the exceed/match/fall-short trifurcation.
- **`grep -in lambert "main (4) copy.tex"`** returns hits ONLY inside the §3.2 Pipeline C dropped-pipeline rationale (currently lines ~291, 297). Any hit outside that block is a regression. Pipeline B description must NOT mention Lambert W.
- **`grep -in lambert supp_material.tex`** returns hits ONLY inside the §A.7 "Preprocessing ablation: why no Lambert W transform" subsection. Any hit elsewhere (e.g., the preprocessing figure caption) is a regression.
- **Preprocessing figure caption** (supp Figure A7) reads "linearly rescaled to [−1, 1] using the global min and max of the standardized log-return series" — NO Lambert W mention.
- **§4.2 first bullet** (now post-reorder) leads with QWGAN-GP architecture + matched-budget outperforms statement. The decision-tree workflow is now the LAST bullet (#4), explicitly framed as "organising concept for future work — not as an empirical contribution".
- **Per-seed dominance table** (supp Table A.X) shows 5 rows (seeds 42–46), all gap values positive (range 0.205 to 0.640), confirming "no quantum-classical seed overlap on LR-DTW" claim.
- **Welch OD-EMD table** (supp Table A.X+1): every p-value > 0.36, every \|d\| < 0.65 (matches §4.1 aggregate claim).
- **Welch LR-EMD table** (supp Table A.X+2): 17 of 20 pairs marked `***` (p < 0.001), 16 of 20 with positive d (quantum loses), 4 with negative d (the 4 quantum-vs-VAE rows where quantum beats VAE).

### 2.3 — Verification gates ✓ ALL PASSING at HEAD (a4cfc1a)

```bash
cd /Users/shawngibford/dev/phd/qGAN

./qgan_env/bin/python verify_number_provenance.py --target "main (4) copy.tex"
# Expect: PASS — 143 distinct numeric literal(s) (was 122 pre-audit; +21 from Table 2 cells)

./qgan_env/bin/python verify_number_provenance.py --target "supp_material.tex"
# Expect: PASS — 156 distinct numeric literal(s) (was 26 pre-audit;
# +25 from per-seed dominance table, +93 from Welch tables, +11 from ablation prose, +1 from Welch caption scope tag)

./qgan_env/bin/python verify_number_provenance.py --differential-test
# Expect: v2.1 differential test PASSED

./qgan_env/bin/python verify_freeze_ready.py
# Expect: all gates PASS except (d) release.md (plan 14-07's deliverable, deferred to acceptance)

# Compile (no TEXINPUTS needed — all 11 figures are in the repo)
pdflatex -interaction=nonstopmode "main (4) copy.tex" && \
  bibtex "main (4) copy" && \
  pdflatex -interaction=nonstopmode "main (4) copy.tex" && \
  pdflatex -interaction=nonstopmode "main (4) copy.tex"
# Expect: 52 pages, 0 undefined refs, 0 undefined cites, 0 hyperref duplicate-destination warnings
```

If any gate fails, **stop**. The gate failures from before the swarm + audit cleanup are documented in §5; any new failure indicates regression.

### 2.4 — Pre-submission tasks

**All 11 figures are now in the repo** (7 legacy + 4 fresh revision-era). No external `TEXINPUTS` needed for AIChE compile.

**Figure inventory:**

| Where | File | Role |
|---|---|---|
| Repo root | `concept_diagram.png` | Fig 1 main conceptual diagram |
| Repo root | `classicalgan.png` | Supp Fig A1 — classical GAN architecture |
| Repo root | `hybridgan.png` | Supp Fig A2 — hybrid GAN (proposed) |
| Repo root | `mech_rep.png` | Supp Fig A3 — mechanistic representation |
| Repo root | `quantum_circuit.png` | Supp Fig A4 — quantum circuit |
| Repo root | `bpm_qgan.drawio.png` | Supp Fig A5 — decision-tree triage schematic (future work) |
| Repo root | `lucy_diagram.jpg` | Supp Fig A6 — LUCY photobioreactor |
| `results/figures/` | `cross_model_dtw_dualscale.{pdf,png}` | Main Fig 2 — 9-model DTW dual-scale |
| `results/figures/` | `cross_model_acf_overlay.{pdf,png}` | Main Fig 3 — 9-model log-return ACF overlay |
| `results/figures/` | `preprocessing_pipeline_4panel.{pdf,png}` | Supp Fig A7 — Pipeline B preprocessing chain |
| `results/figures/` | `preprocessing_ablation_comparison.{pdf,png}` | Supp Fig A8 — A/B/C ablation comparison |

**Tag** — chosen: `v1.2` (matched-budget release version bump).

```bash
git tag v1.2     # already done locally
git push origin main
git push origin v1.2
```

**GitHub release notes**: at `https://github.com/shawngibford/qgan/releases`. Suggested release-notes headlines: matched-budget cross-model evaluation; bifurcated finding (exceed on LR-DTW + lag-1 ACF, match on OD-EMD, fall short on LR-EMD); Pipeline B preprocessing finalized (Lambert W dropped per D-10-05 / R1-M3); statistical evidence — Table 2 + per-seed dominance + 40-pair Welch tests.

**AIChE submission portal**:
- Upload `main (4) copy.tex`, `supp_material.tex`, `bib.bib`, `ama.bst` (bibliography style file lives at `~/Documents/main_qgan/ama.bst` — copy alongside the .tex), and all **11 figure files** listed above.
- Compile is `pdflatex + bibtex + pdflatex × 2` (no special env needed — all paths repo-relative).
- Total upload: 4 source files + 11 figures = 15 files.

---

## §3 — Open questions deferred to human

These were NOT decided by the swarm:

### §3.1 — Title rescope

Current title: **"Quantum Synthetic Data Generation for Industrial Bioprocess Monitoring"**

W1 proposed 3 candidates:

1. *"Matched-Budget Comparison of Quantum and Classical WGAN-GPs on Laboratory-Scale Bioprocess Time Series"* — names the protocol; scope-conservative.
2. *"Quantum WGAN-GPs Capture Log-Return Temporal Structure of a Photobioreactor Cultivation: A Parameter-Matched Proof of Concept"* — names the bifurcated finding; slightly long.
3. *"A Parameterized Quantum Circuit Generator for Synthetic Bioprocess Time Series: A Matched-Parameter, Matched-Epoch Evaluation"* — closest to current; scope-conservative.

The A2 prohibition sentinel classifies the current title's "Industrial Bioprocess Monitoring" as **ALLOWED** (sanctioned title exception). Rescoping is optional. If you keep the current title, no action needed.

### §3.2 — Tag name ✓ DECIDED

`v1.2` chosen (matched-budget release version bump). Tagged locally; push pending user `git push origin v1.2`.

### §3.3 — Word counts to verify on Overleaf

The Abstract is **146 words** (under the 150 limit). The Plain Language Summary is **242 characters** (under the 250 limit). Both pass locally; Overleaf may format slightly differently.

---

## §4 — Audit verdict trajectory (record of what the swarm caught)

### A5 (Reviewer 2 simulator) verdict trajectory

| Commit | Verdict | Notes |
|---|---|---|
| `d81306f` (pre-swarm) | n/a — A4 said FAIL: Finding 2 entirely absent from manuscript | Trigger for the swarm |
| `d03d35f` (post-W2) | **MAJOR_REVISIONS_STILL_NEEDED** | 3 new substantive findings: M1 LR-EMD asymmetry, M2 lag-1 ACF mean-only, M3 OD-pipeline invariance |
| `3e0cd2d` (post-W2b) | (not re-run) | M1+M2+M3 addressed inline; subsequent gates clean |
| `0f47c25` (post-W4) | **MAJOR_REVISIONS_STILL_NEEDED** | 6 prior M's all RESOLVED; 2 NEW critical: M7 (Table 1 sources legacy seed-42), M8 (AR(2) attribution in Abstract/§1.4) |
| `94ea5a0` (post-W4b) | (not re-run) | M7 + M8 + 3 minor items addressed |
| `a50cb0f` (post-A6) | (final state — A5 not re-run, but A5 explicitly said "with M7 + M8 + m-list polish applied, the next pass would warrant ACCEPT_MINOR_REVISIONS") | All 6 high-value minor items addressed |

### Provenance gate v2.2

`771d338` patched the gate to handle LaTeX `--` numeric ranges (it was tokenizing `0.94--1.12` as `[0.94, -1.12]`). v2.1 differential self-test still PASSES under v2.2; no other behavior changed.

---

## §5 — Hard prohibitions (must not be re-introduced)

These three corrections are **load-bearing** and were the foundation of the swarm's work. If a future session changes them back, the manuscript becomes scientifically incorrect.

1. **VAE is a degenerate generation regime, NOT "posterior collapse"**. Log-return std is 0.0186 (≈ real 0.0217). The anomaly is lag-1 ACF = −0.648 vs real −0.064. **Never re-claim "posterior collapse" or "synthetic std ≈ 4×10⁻⁴".**

2. **LR-DTW (not LR-EMD) is the surviving quantum-distinguishing signal**. The LR-EMD quantum-advantage claim was withdrawn during Plan 14-16 forensic remediation (broken `density=True` column). On the corrected scale, **every classical adversarial baseline outperforms every quantum variant on LR-EMD** (AR=0.003, classical adversarial 0.007–0.013, quantum 0.014–0.015, VAE 0.016). The current §4.1 discloses this asymmetry honestly; **never re-claim quantum advantage on LR-EMD**.

3. **Real-data lag-1 ACF reference is −0.0641** (matched-pipeline, with dither), **NOT −0.029** (legacy unmatched).

4. **Pipeline B = log-returns + standardize + linear rescale to [-1, 1] — NO Lambert W.** The matched-budget runs use Pipeline B exclusively per decision D-10-05 (5-seed ablation showed Pipeline C tied with B on every OD-scale metric while introducing an over-Gaussianization concern flagged by reviewer R1-M3). The inverse Lambert W transform belongs to dropped Pipeline C and may only appear in the manuscript inside the explicit "Pipeline C dropped" rationale (Methods §3.2, Supp §A.7). **Never re-introduce Lambert W into the matched-budget Pipeline B description or the preprocessing figure.** The `lambert_w_transform` / `inverse_lambert_w_transform` functions in `core/data.py` are retained for ablation reproducibility only; do not delete them.

Additional prohibitions enforced by the A2 sentinel regex sweep:

- "deployable framework", "industrial bioprocess monitoring" (outside title), "high fidelity", "strong performance", "computational advantages" — only allowed inside the explicitly-labelled §4.5 Outlook subsection.
- "Hybrid-GAN" must always be qualified as "proposed", "not implemented", "aspirational", or "future-conditional". Never "implemented", "evaluated", "demonstrated", "validated".
- "closed-loop feedback control" for the AI workflow — replaced everywhere with "decision-tree triage workflow".
- "0.6843" allowed ONLY inside a pre-v1.0 historical-reference clause.
- "n=1" or "single representative seed" must NOT be used for shot-noise / noise-channel context (both use n=3 seeds {42, 43, 44}).
- "demonstrated equivalence" / "TOST equivalence is satisfied" — TOST is NOT satisfied at any defensible margin.

---

## §6 — Outstanding

| Item | Status | Action |
|---|---|---|
| Origin/main is 18 commits behind local | Local main has swarm + audit-cleanup work; tag v1.2 also local | `git push origin main && git push origin v1.2` when ready |
| AIChE portal upload | Compile verified clean without TEXINPUTS | Upload 15 files per §2.4 |
| GitHub release notes | n/a | Update at `https://github.com/shawngibford/qgan/releases` after push (suggested copy in §2.4) |
| `ama.bst` bibliography style | Lives at `~/Documents/main_qgan/ama.bst`, not git-tracked | Copy alongside the .tex for AIChE submission (or upload to portal as a 16th file) |
| Plan 14-07 (Zenodo DOI mint) | Deferred to journal acceptance per `project_phase14_zenodo_blocker` memory | Mint Zenodo DOI at acceptance; rebuttal currently cites `ZENODO-DOI-PLACEHOLDER` |
| Phase 13 verification debt | 6 `human_needed` items in `.planning/phases/13-architecture-introspection/13-VERIFICATION.md` | Orthogonal to submission; address before/after at your discretion |
| Word-document rebuttal sync (if AIChE requires it) | Rebuttal letter is in `.planning/REBUTTAL-HANDOFF.md` | If portal wants it, convert to .docx at upload |

---

## §7 — Lessons from the swarm + audit cleanup (for future paper-rewrite efforts)

### From the swarm
- **Sequential writers, parallel auditors** worked. Voice drift was minimal; 4 writers in sequence each took ~3–5 minutes of agent time.
- **A5 (peer-review simulator) earned its slot twice**: at end-of-W2 it surfaced 3 substantive issues that none of the deterministic auditors caught (LR-EMD asymmetry, lag-1 ACF mean-only, OD-pipeline invariance); at end-of-W4 it surfaced 2 critical issues (Table 1 sourcing, AR(2) attribution). Without A5 these would have shipped.
- **The provenance gate v2.1 had a tokenizer bug** for LaTeX `--` ranges. v2.2 patch is upstreamable to any future paper-rewrite work.
- **The auto-memory and `.planning/DECISIONS.md` artifacts** were load-bearing for keeping 5 writer agents and 6 audit agents in sync. The "every numeric literal must trace to a JSON cell" discipline plus the explicit prohibition regex set kept the manuscript drift-free across 10 commits.
- **A5's prior critique was preserved verbatim in the next-wave A5 prompt**, which let A5 explicitly say "M1 RESOLVED — [verbatim quote of new content]". That continuity made the audit chain auditable end-to-end.

### From the post-swarm audit cleanup (2026-05-28)
- **The provenance gate validates literals, not prose.** It caught zero issues with the swarm's Lambert W misdescription because the affected numbers (778, 384, 5 seeds, etc.) were all correct in isolation — the prose chaining them as "Pipeline B = log-returns → standardize → Lambert W → rescale" was wrong, but no individual number triggered the gate. **Lesson:** add a "pipeline-name prose check" gate that verifies every "Pipeline B" mention's described chain matches the actual `run_ablation.py::build_dataset_for_pipeline('B', ...)` code path. The next swarm iteration should include a prose-vs-code consistency auditor as a deterministic gate.
- **Four parallel audit agents was the right number for a 14-section paper.** Each had a tight, distinct scope (structural / cleanliness / figures / prohibitions). Together they surfaced 4 BLOCK + 10 FLAG findings in ~12 min wall time; the same audit by a single agent would have taken 4× longer and missed cross-cutting issues from the narrow scope per agent.
- **Stale-figure-in-cap detection** was the agent's most valuable single contribution. The 5 single-model figures (`pdf.png`, `cdf.png`, etc.) had retrofitted captions but Oct-2025 file timestamps and embedded "Lucy Log Returns" / "Log δ" titles that no human reader scanning the manuscript would have spotted without opening the underlying image files. The agent's `stat` + Read-the-PDF combination caught this where prose-reading alone would not have.
- **Adding artefacts (Table 2, per-seed dominance, Welch pairwise tables) addressed a "make the strong claim auditable" reviewer-ergonomics gap.** The original §4.1 prose was scope-honest but the strong "no quantum-classical seed overlap on LR-DTW" claim was assertion-only. Adding the per-seed table converted it into a cell-counting test the reviewer can verify in one glance. Same logic applied to the 40-pair Welch table for the OD-EMD null and the LR-EMD reversal. **Lesson:** for any strong claim of the form "every X is Y", produce the artefact that lists all X cells so the reviewer can re-verify by inspection.

---

**End of submission handoff.** When in doubt, prefer the JSON sources in `results/` over any prose summary. Every quantitative claim in the manuscript is supposed to trace back to one of them.

**Cumulative session state at HEAD (a4cfc1a, tag v1.2):**
- Manuscript: submission-ready
- Numerical traceability: every literal traces to `results/*.json`
- Statistical artefacts: Table 2 + per-seed dominance + 40-pair pairwise Welch tests
- Reviewer ergonomics: Table 2 bolded row-leaders; full-page-width supp stat tables; per-claim auditable artefacts
- Scope hedging: all 4 hard prohibitions + 14 sentinel-phrase prohibitions hold
- Pipeline B (no Lambert W): enforced across main + supp + figures
- Figures: 0 stale assets; all 11 in PDF reflect current matched-budget narrative; all repo-local
- AIChE packaging: ✓ compile clean without TEXINPUTS
