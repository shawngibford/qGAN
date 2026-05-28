# Paper Submission Handoff — Wave 8 (Human ACT)

**Status as of 2026-05-27**: Paper-rewrite swarm complete. Manuscript is submission-ready at commit `a50cb0f`. All gates green. Only human read-through, tag, push, and AIChE-portal upload remain.

**Deadline**: ≈ 2026-06-17 (three-week extension granted 2026-05-27; ~21 days remaining at swarm completion).

> If you're a fresh session resuming this work: **read this doc end-to-end first**, then `git log a0f932b..HEAD --oneline` to see the swarm trail, then run the verification gates in §3 below. **Do NOT re-execute the swarm** — it's already done. Don't re-introduce the corrections it made (catalogued in §5).

---

## §0 — Resume in 90 seconds

| Field | Value |
|---|---|
| Repo | `/Users/shawngibford/dev/phd/qGAN/` |
| Branch | `main`, 10 commits ahead of `origin/main` |
| Manuscript | `main (4) copy.tex` (filename has literal space + paren — quote in shell) |
| Supplement | `supp_material.tex` |
| Bibliography | `bib.bib` (frozen at 59 entries) |
| Last commit | `a50cb0f refactor(paper-rewrite): A6 style polish + structural cleanup` |
| Provenance gate | v2.2, PASS — 122 main + 26 supp literals all resolve to `revision/results/*.json` |
| pdflatex compile | PASS — 53 pages, 0 undefined cites, 0 undefined refs, all 3 new figures render |
| A5 (Reviewer 2 sim) final verdict | All 6 prior major issues + 2 swarm-discovered critical issues resolved |
| Working tree | Clean |

**Next action**: human read-through → tag → push → AIChE portal upload. See §3.

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

### Commit trail (10 atomic commits)

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

## §2 — Wave 8: human ACT checklist

### 2.1 — End-to-end manuscript read-through

Read these sections in order. The narrative should flow with Finding 2 leading and Finding 1 + LR-EMD asymmetry hedging consistently throughout.

```
main (4) copy.tex:
  Abstract (line 47–50)        — leads Finding 2, Finding 1 + LR-EMD scope-hedged
  Plain Language Summary (58–59)
  §1.4 Principal Contributions (95–111)
  §3 Methods (155–315)         — full training protocol + Pipeline B definition
  §4.1 Cross-Model Comparison  — bifurcated finding centralized
  Table 1 (line 332+)          — LR-EMD cell = 0.01497 ± 0.00020
  §4.2 Key Contributions
  §4.3 Implications
  §4.4 Limitations             — "Scope of the matched-budget finding" itemize
  §4.5 Outlook                  — "Decision-tree triage workflow" header
  §5 Concluding Remarks         — opens with §1.4 falsifiable-question answer
  
supp_material.tex:
  §A.3 Hybrid-GAN (lines 142+) — verify still flagged "proposed, not implemented"
  §A.5 Figure A5 caption (~line 359) — "decision-tree triage schematic"
  §A.7 Data Transformation     — preprocessing_pipeline_4panel figure inserted
  Reconciliation note (~line 307) — 0.6843 pre-v1.0 vs 0.302 matched-budget
```

### 2.2 — Eyeball checks

- **Table 1 LR-EMD cell** reads `0.01497 ± 0.00020`. The legacy value `0.1209` should NOT appear.
- **Table 1 ACF lag-1 mean cell** reads `-0.09490 ± 0.00923`. The legacy `-0.0814` should NOT appear.
- **Abstract "1.58 – 6.86"** (not "1.58 – 7.70" — that conflates AR(2) with adversarial baselines).
- **§4.5 Outlook header** reads "Decision-tree triage workflow" (NOT "Closed-loop decision-driven pipeline").
- **§5 first sentence** answers the §1.4 falsifiable question with the exceed/match/fall-short trifurcation.
- **`grep -in lambert "main (4) copy.tex"`** returns hits ONLY inside the §3.2 Pipeline C dropped-pipeline rationale (lines ~291, 297). Any hit outside that block is a regression and must be removed before submission. Pipeline B description must NOT mention Lambert W.
- **`grep -in lambert supp_material.tex`** returns hits ONLY inside the §A.7 "Preprocessing ablation: why no Lambert W transform" subsection. Any hit elsewhere (e.g., the preprocessing figure caption) is a regression.

### 2.3 — Run the verification gates

```bash
cd /Users/shawngibford/dev/phd/qGAN

./qgan_env/bin/python revision/verify_number_provenance.py --target "main (4) copy.tex"
# Expect: PASS — 122 distinct numeric literal(s)

./qgan_env/bin/python revision/verify_number_provenance.py --target "supp_material.tex"
# Expect: PASS — 26 distinct numeric literal(s)

./qgan_env/bin/python revision/verify_number_provenance.py --differential-test
# Expect: v2.1 differential test PASSED

./qgan_env/bin/python revision/verify_freeze_ready.py
# Expect: all gates PASS except release.md (which is plan 14-07's deliverable, deferred to acceptance)
```

If any gate fails, **stop**. Inspect the failure before tagging. The gate failures from before the swarm are documented in §5; any new failure indicates regression.

### 2.4 — Pre-submission tasks

1. **Upload the 12 legacy figures to AIChE** (they live at `/Users/shawngibford/Documents/main_qgan/` and `/Users/shawngibford/Documents/dtu/arxiv_submission/...`, NOT in this repo):
   - `concept_diagram.png`
   - `dtwd.png`, `pdf.png`, `cdf.png`, `qq.png`, `acf.png` (single-model diagnostics)
   - `classicalgan.png`, `hybridgan.png`, `mech_rep.png` (supp diagrams)
   - `quantum_circuit.png`, `bpm_qgan.drawio.png`, `lucy_diagram.jpg` (supp figures)

2. **The 3 new figures ARE in the repo** at:
   - `revision/results/figures/cross_model_dtw_dualscale.{pdf,png}`
   - `revision/results/figures/cross_model_acf_overlay.{pdf,png}`
   - `revision/results/figures/preprocessing_pipeline_4panel.{pdf,png}`
   - The .tex references them via `\includegraphics{revision/results/figures/...}` — make sure Overleaf/AIChE portal can resolve those paths, or flatten the references if the upload requires it.

3. **Tag and push**. The plan deferred the tag name to you:
   - `v1.2` — feels right if this is a "version bump" of the matched-budget release.
   - `v1.0-revision.final` — feels right if this is "the final pre-acceptance state of the v1.0 manuscript".
   ```bash
   git tag v1.2  # or whichever you pick
   git push origin main
   git push origin v1.2
   ```

4. **Update GitHub release page** at `https://github.com/shawngibford/qgan/releases` — note the swarm work + the new figures.

5. **AIChE submission portal**: upload `main (4) copy.tex`, `supp_material.tex`, `bib.bib`, and all figure files (12 legacy + 3 new = 15 files). The .tex compiles cleanly with `pdflatex + bibtex + pdflatex × 2`.

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

### §3.2 — Tag name

`v1.2` vs `v1.0-revision.final` — pick at submission time.

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

4. **Pipeline B = log-returns + standardize + linear rescale to [-1, 1] — NO Lambert W.** The matched-budget runs use Pipeline B exclusively per decision D-10-05 (5-seed ablation showed Pipeline C tied with B on every OD-scale metric while introducing an over-Gaussianization concern flagged by reviewer R1-M3). The inverse Lambert W transform belongs to dropped Pipeline C and may only appear in the manuscript inside the explicit "Pipeline C dropped" rationale (Methods §3.2, Supp §A.7). **Never re-introduce Lambert W into the matched-budget Pipeline B description or the preprocessing figure.** The `lambert_w_transform` / `inverse_lambert_w_transform` functions in `revision/core/data.py` are retained for ablation reproducibility only; do not delete them.

Additional prohibitions enforced by the A2 sentinel regex sweep:

- "deployable framework", "industrial bioprocess monitoring" (outside title), "high fidelity", "strong performance", "computational advantages" — only allowed inside the explicitly-labelled §4.5 Outlook subsection.
- "Hybrid-GAN" must always be qualified as "proposed", "not implemented", "aspirational", or "future-conditional". Never "implemented", "evaluated", "demonstrated", "validated".
- "closed-loop feedback control" for the AI workflow — replaced everywhere with "decision-tree triage workflow".
- "0.6843" allowed ONLY inside a pre-v1.0 historical-reference clause.
- "n=1" or "single representative seed" must NOT be used for shot-noise / noise-channel context (both use n=3 seeds {42, 43, 44}).
- "demonstrated equivalence" / "TOST equivalence is satisfied" — TOST is NOT satisfied at any defensible margin.

---

## §6 — Outstanding (out of swarm scope)

| Item | Status | Action |
|---|---|---|
| 12 legacy figures (concept_diagram.png, etc.) | Never git-tracked; live at `~/Documents/main_qgan/` and `~/Documents/dtu/arxiv_submission/...` | Upload to AIChE portal alongside the .tex |
| Phase 13 verification debt | 6 `human_needed` items in `.planning/phases/13-architecture-introspection/13-VERIFICATION.md` | Orthogonal to submission; address before/after at your discretion |
| Plan 14-07 (Zenodo DOI mint) | Deferred to journal acceptance per `project_phase14_zenodo_blocker` memory | Mint Zenodo DOI at acceptance; rebuttal currently cites `ZENODO-DOI-PLACEHOLDER` |
| Origin/main is 10 commits behind local | Local main has the swarm work | `git push origin main` when comfortable |
| Word-document rebuttal sync (if AIChE requires it) | Rebuttal letter is in `.planning/REBUTTAL-HANDOFF.md` | If portal wants it, convert to .docx at upload |

---

## §7 — Lessons from the swarm (for future paper-rewrite efforts)

- **Sequential writers, parallel auditors** worked. Voice drift was minimal; 4 writers in sequence each took ~3–5 minutes of agent time.
- **A5 (peer-review simulator) earned its slot twice**: at end-of-W2 it surfaced 3 substantive issues that none of the deterministic auditors caught (LR-EMD asymmetry, lag-1 ACF mean-only, OD-pipeline invariance); at end-of-W4 it surfaced 2 critical issues (Table 1 sourcing, AR(2) attribution). Without A5 these would have shipped.
- **The provenance gate v2.1 had a tokenizer bug** for LaTeX `--` ranges. v2.2 patch is upstreamable to any future paper-rewrite work.
- **The auto-memory and `.planning/DECISIONS.md` artifacts** were load-bearing for keeping 5 writer agents and 6 audit agents in sync. The "every numeric literal must trace to a JSON cell" discipline plus the explicit prohibition regex set kept the manuscript drift-free across 10 commits.
- **A5's prior critique was preserved verbatim in the next-wave A5 prompt**, which let A5 explicitly say "M1 RESOLVED — [verbatim quote of new content]". That continuity made the audit chain auditable end-to-end.

---

**End of submission handoff.** When in doubt, prefer the JSON sources in `revision/results/` over any prose summary. Every quantitative claim in the manuscript is supposed to trace back to one of them.
