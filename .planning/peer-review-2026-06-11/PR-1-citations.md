# PR-1 — Citation Audit Report

**Reviewer role:** adversarial. Manuscript: `main (4) copy.tex` + `paper/supp_material.tex` (AMA / `paper/ama.bst`, numerical brackets, citation-order numbering).

## Verdict
**BLOCKING.** Two coauthor-flagged miscitations confirmed; both are factually wrong and survive the [R1-m1] R&R fix the source comments claim was applied. Additional BLOCK-level finding: DTW is introduced in the supplement with a citation to an anomaly-detection paper that doesn't discuss DTW at all. Citation style is inconsistent in three independent dimensions (before/after period, grouped/separate brackets, ~ vs space, comma/semicolon spacing). Two missing canonical references that any AIChE reviewer with the field will demand: Gulrajani-correct attribution is fine, but **Sakoe & Chiba 1978** (or Berndt & Clifford 1994) for DTW and a **TimeGAN / COT-GAN / TTS-GAN benchmarking reference set** to anchor the time-series-GAN landscape are absent.

Citation-number → bibkey mapping (from `main (4) copy.bbl`, citation order) used throughout:
- [22] Dallaire_Demers_2018, [23] Lloyd_2018, [24] Zoufal_2019, [25] rudolph2022, [26] he2025qgan, [27] orlandi2024enhancing, **[28] Mugel2022**, [29] esteban2017, [30] Cerezo_2021, [31] goodfellow2014, [34] gulrajani2017improved, [35] arjovsky17a, [36] villani2008optimal, [41] Lin_2020, **[42] dimoudis2023utilizing**.

---

## C1 — Ref [42] anomaly-detection misuse (coauthor flag) — **CONFIRMED**

- **Bib key:** `dimoudis2023utilizing`
- **Bib title (verbatim, bib.bib:408):** "Utilizing an adaptive window rolling median methodology for time series anomaly detection"
- **Cited at (4 occurrences):**
  1. `main (4) copy.tex:177` — rolling-window subsequence extraction
  2. `supp_material.tex:287` — **DTW intro** (introduces the method itself)
  3. `supp_material.tex:790` — "rolling window technique"
  4. `supp_material.tex:1399` — "rolling window approach" (critic input shape)
- **Verbatim prose (worst offender, supp:287):**
  > "Dynamic Time Warping (DTW) represents a powerful technique for measuring similarity between temporal sequences that may exhibit varying lengths or temporal misalignments. \cite{dimoudis2023utilizing}"

  The Dimoudis paper is about **adaptive rolling-median anomaly detection** in a sensor stream. It does not introduce, derive, or even discuss DTW. Citing it to introduce DTW is a category error.

- **Verbatim prose (main:177):**
  > "Overlapping subsequences of length 10 with stride 2 were extracted using a rolling window approach \cite{dimoudis2023utilizing}.  % [41] removed: it was adaptive rolling-median anomaly detection, not subsequence extraction (R1-m1)"

  **The inline LaTeX comment is the smoking gun.** The author acknowledges *in source* that this citation is for an anomaly-detection rolling-median paper and that R1 already flagged it (R1-m1). The fix removed an old `[41]` but the actual offending key (`dimoudis2023utilizing`) was left in place. The R&R remediation is **incomplete and self-contradicting** — a hostile reviewer will quote this comment directly.

- **Proposed replacements:**
  - For **rolling-window subsequence extraction in time-series ML** (main:177, supp:790, supp:1399): Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation.* Information Sciences 191, 192–213 (rolling-origin / sliding-window canon). Alternatively for time-series-GAN-specific windowing, cite **Yoon et al. TimeGAN [19]** already in the bib — it's the canonical reference for windowing time series prior to GAN training, and it is already cited at main:86 and main:133.
  - For **DTW (supp:287)**: Sakoe, H., & Chiba, S. (1978). *Dynamic programming algorithm optimization for spoken word recognition.* IEEE Trans. ASSP 26(1), 43–49. Or Berndt, D. J., & Clifford, J. (1994). *Using dynamic time warping to find patterns in time series.* AAAI Workshop. Either is the universally accepted citation for DTW introduction.

- **Recommended action:** Replace all four occurrences. Keep `dimoudis2023utilizing` in the bib only if it is actually used for anomaly detection somewhere; otherwise delete the entry.

---

## C2 — Ref [28] QGAN-applications miscategorization (coauthor flag) — **CONFIRMED**

- **Bib key:** `Mugel2022`
- **Bib title (verbatim, bib.bib:537):** "Dynamic portfolio optimization with real datasets using quantum processors and quantum-inspired tensor networks"
- **Cited at (2 occurrences):**
  1. `main (4) copy.tex:92` (Introduction §1, QGAN application list)
  2. `main (4) copy.tex:155` (§2.2 Quantum Machine Learning and QGANs, again QGAN application list)
- **Verbatim prose (main:92, identical pattern at main:155):**
  > "Prior work has reported QGAN results in low-data regimes and on multimodal distributions \cite{rudolph2022..., he2025...}, with applications in finance \cite{orlandi2024enhancing} and optimization \cite{Mugel2022}"

  Mugel et al. 2022 is a paper on **quantum-inspired tensor networks and quantum annealing for dynamic portfolio optimization**. It does not contain a QGAN, a quantum generator, or any adversarial training. Grouping it as a "QGAN application … in optimization" is misleading — it implies a QGAN solved a portfolio problem, when in fact the methods are tensor-network/annealing-based.

- **Verdict:** confirmed. Both occurrences are in QGAN-application context and both are wrong.

- **Proposed replacements (QGAN-for-optimization-related papers):**
  - **Niu, M. Y., Zlokapa, A., Broughton, M., et al. (2022).** *Entangling quantum generative adversarial networks.* PRL 128, 220505 — actual quantum GAN with combinatorial applications.
  - **Stein, S. A., L'Abbate, R., Mu, W., et al. (2022).** *A hybrid system for learning classical data in quantum states.* IEEE BigData — QGAN applied to classification-adjacent optimization.
  - Or simpler: **delete the "and optimization \cite{Mugel2022}" clause** — the QGAN-finance citation (Orlandi) already covers the applied-domain breadth needed for §1 and §2.2. The "optimization" claim is not core to the paper and removing it strengthens, rather than weakens, the prior-art framing.

- **Recommended action:** Drop `Mugel2022` from both `\cite{...}` calls (main:92, main:155). If you want to retain an "optimization" application, replace with one of the suggestions above and add the new bib entry. Otherwise simply remove the phrase " and optimization \cite{Mugel2022}" from both sentences.

---

## C3 — Expanded sweep findings

### BLOCK findings (must fix before resubmission)

**B1. supp:287 DTW intro misattributed (also covered in C1).** See above — the same `dimoudis2023utilizing` is used to "define" DTW. Worst single offender. Already counted.

**B2. supp:91 stray newline-leading citation.** `supp_material.tex:90-91`:
> "and difficulty with complex multimodal distributions when applied to bioprocess data
> \cite{Arjovsky2017, gulrajani2017improved}. These challenges motivate..."

`\cite{...}` is the first thing on its own line, after the sentence is broken across two source lines. Compiles fine but renders as awkward spacing in some BibTeX styles. Fix: move citation to end of preceding source line.

**B3. supp:85 mixed author-year + numerical cite.**
> "The original GAN formulation by Goodfellow et al. (2014) introduced several key innovations \cite{goodfellow2014generativeadversarialnetworks}:"

The "(2014)" is redundant with `\cite{...}` and inconsistent with the numerical AMA style used throughout. Fix: drop the "(2014)" — "The original GAN formulation by Goodfellow et al. \cite{goodfellow2014generativeadversarialnetworks} introduced..." (this matches the style used at main:673, main:677, supp:153, supp:796).

### FLAG findings (should fix; reviewer-charitability borderline)

**F1. supp:280 `rice2020overfittingadversariallyrobustdeep` for "physics term reduces overfitting."**
> "the physics term $M(G(z))$ would impose mechanistic constraints that reduce overfitting \cite{rice2020overfittingadversariallyrobustdeep}"

Rice et al. is about *adversarial robustness / overfitting in adversarially robust training*, not about physics-informed mechanistic regularization. The cited paper supports the general claim "overfitting in adversarial DL is a known problem", but the in-text claim is specifically about physics-informed regularization reducing overfitting. Stronger choices: Karniadakis et al. 2021 (*Nature Rev. Physics*, "Physics-informed machine learning"), Raissi et al. 2019 (PINNs, *J. Comp. Physics*), or Sharma & Liu 2022 already in the bib as `sharma2022hybrid` [47] (hybrid science-guided ML).

**F2. supp:103 `villani2008optimal` for EMD definition.**
> "The Wasserstein distance, also known as the Earth Mover's Distance (EMD), provides a principled way to measure the dissimilarity between two probability distributions. \cite{villani2008optimal}"

Villani 2008 is correct for *optimal transport theory* but is a 1000-page textbook — for the EMD-as-distance claim a more focused citation would be Rubner, Tomasi & Guibas 2000 (*IJCV*, the canonical EMD paper) or kept as-is with a page reference. Charitable, but a hostile reviewer will note Villani never uses the term "EMD" except in passing.

**F3. main:152 `Cerezo_2021` overload.** Cerezo et al. is cited 3 times in the manuscript (main:92, main:152, supp:151) for three distinct claims: NISQ-scalability constraints, VQA benefits for bioprocess applications, and again NISQ noise limits. The middle claim — "variational quantum algorithms… offering potential benefits for data-scarce bioprocess applications" — is **not** supported by Cerezo 2021 (the Nature Rev. Physics paper makes no bioprocess claim whatsoever). FLAG for overciting beyond what the source supports.

**F4. supp:153 redundant double-citation.**
> "Recent experimental implementations have demonstrated proof-of-concept QGANs on noisy intermediate-scale quantum (NISQ) devices.  \cite{rudolph2022..., Huang_2021} Rudolph et al. demonstrated that QGANs can generate high-resolution images using an ion trap quantum computer... \cite{rudolph2022...}."

`rudolph2022` is cited twice in the same paragraph, once in a group and again at the end. Consolidate or drop the second.

**F5. main:74 fractured citation commas.**
> "applications spanning process monitoring, \cite{peng2025...} optimization, \cite{sharma2025ai} and control. \cite{mondal2023review}"

The trailing-comma-before-cite-then-next-word pattern (`, \cite{X} word,`) is non-standard. AMA convention would be `monitoring [9], optimization [10], and control [11]`. Stylistic, but cumulative effect across §1 reads sloppy.

### NIT findings (defer to camera-ready)

**N1. main:70 four un-grouped consecutive `\cite{}` calls.** `\cite{A} \cite{B} \cite{C} \cite{D}` should be `\cite{A,B,C,D}`. Same pattern at main:78 (`\cite{shariatifar2025digital} \cite{exploring_hernndezromero_2025}`). AMA `paper/ama.bst` will collapse and sort these correctly, but the source-level inconsistency is a flag.

**N2. main:177 inline TeX comment leaks intent.** The "% [41] removed: it was adaptive rolling-median anomaly detection..." comment exists *in the source file* the journal will receive if you submit the .tex. Strip all `% ... R1-m1 / R2-m1 ...` revision notes before submission — they are internal scaffolding that reveals the R&R process to reviewers and risks signaling that fixes were partial. (Also present at supp:151: "% [55]-[57],[59] (VQE / option-pricing / QAOA / adversarial-robustness) removed: ...")

**N3. supp:1399 and supp:1403** use `~\cite{}` (non-breaking tilde) while most of the rest of the doc uses ` \cite{}` (space). Pick one and replace globally; tilde is the technically-correct LaTeX choice.

---

## C4 — Citation style consistency

**Dominant style:** numerical AMA brackets via `\cite{}`, placed **after** the closing period (`...sentence. \cite{key}`). This is the count winner across §1, §2, §3. AMA `paper/ama.bst` will render as `[N]` superscript-or-bracket numerical.

**Deviation taxonomy + line numbers:**

| Deviation | Lines | Verbatim fragment |
|---|---|---|
| Before-period cite (`\cite{key}.`) | main:88, main:135, main:148, main:155, main:177, main:866; supp:280, supp:1403, supp:1415 | e.g., main:88 "...biochemical engineering \cite{bernal2022perspectives}." |
| Author-year + numerical mixed | supp:85 | "Goodfellow et al. (2014) ... \cite{goodfellow2014generativeadversarialnetworks}" |
| Consecutive un-grouped (`\cite{A} \cite{B}`) | main:70 (×3), main:78 (×2) | "(Figure...). \cite{A} \cite{B} \cite{C} \cite{D}" |
| Newline-leading cite | supp:91 | "\cite{Arjovsky2017, gulrajani2017improved}. These challenges..." |
| `~\cite{}` (non-breaking) vs ` \cite{}` (space) | supp:691, supp:790, supp:796, supp:1399, supp:1403, supp:1415 | "~\cite{goerg2015lambert}", "~\cite{dimoudis2023utilizing}" — main file uses space |
| Trailing-comma-before-cite | main:74 (×3) | "monitoring, \cite{peng2025...}" |

**Recommendation:** standardize on `... sentence \cite{key1, key2}.` — numerical brackets, inside the sentence, before the closing period, comma-separated grouping. AMA-typical. Apply globally.

**Edit count estimate:** ~35 source-line edits across both files (mostly mechanical sed-able: move `\cite{X}.` to `\cite{X}.` → `\cite{X}.` is already-format, leave; move `. \cite{X}` to ` \cite{X}.` requires word-by-word context). Realistic wall-clock with author-eye review: **45–60 min**.

---

## C5 — Missing prior art (adversarial)

Specific gaps a reviewer will demand:

1. **DTW canonical reference (highest priority).** Sakoe & Chiba 1978 OR Berndt & Clifford 1994 (see C1/B1). Section: supp §A.4.1 (DTW Implementation), supp:287. **Bibkey suggestion: `sakoe1978dtw` or `berndt1994dtw`.**

2. **Rolling-window time-series-ML canon.** Bergmeir & Benítez 2012 (*Information Sciences*) for sliding-window evaluation; or rely on TimeGAN [19] already in the bib for windowing-before-GAN. Section: main §3 / supp §A.4.2 (Rolling Window Implementation). **Bibkey: `bergmeir2012rolling`.**

3. **Time-series GAN benchmark family.** The paper benchmarks QWGAN-GP against classical WGAN but doesn't cite the time-series-GAN benchmarking landscape. **COT-GAN** (Xu et al. NeurIPS 2020, "COT-GAN: Generating Sequential Data via Causal Optimal Transport"), **RCGAN** (Esteban [29] is partially this), and **TTS-GAN** (Li et al. 2022) belong in §2.2 alongside TimeGAN [19] and DoppelGANger [41]. A hostile reviewer will ask: "why these two classical baselines and not the broader benchmark suite?" Section: main:155 and main §2.2. **Bibkeys: `xu2020cotgan`, `li2022ttsgan`.**

4. **Quantum-GAN expressivity / barren-plateau caveat.** The manuscript discusses NISQ scalability via Cerezo 2021 [30] but doesn't cite the barren-plateau literature that is *the* central caveat for any PQC-generator claim. **McClean et al. 2018** (*Nature Comms*, "Barren plateaus in quantum neural network training landscapes") and **Holmes et al. 2022** (*PRX Quantum*, "Connecting ansatz expressibility to gradient magnitudes and barren plateaus") are mandatory citations when reporting a 55–135-parameter PQC generator outperforming classical baselines — without them, the dominant question from a quantum reviewer ("is this in the barren-plateau regime?") is unaddressed. Section: main §2.2 (after NISQ-constraints sentence at main:152) or supp §A.3.3. **Bibkeys: `mcclean2018barren`, `holmes2022barrenexpressibility`.**

5. **Schuld expressivity (mentioned in coauthor brief but absent).** `schuld2019quantum` is already in the bib as [40], cited once at main:152 for quantum kernels. Schuld & Petruccione 2021 textbook chapter on expressivity, or Schuld, Sweke & Meyer 2021 (*PRA*, "Effect of data encoding on the expressive power of variational quantum-machine-learning models") would be a stronger expressivity citation than the 2019 kernel paper. Section: main §2.2.

6. **WGAN-GP theory is well-cited** ([34] Gulrajani; [33] Arjovsky2017; [35] arjovsky17a; [36] villani2008optimal). No gap here.

7. **Bioprocess monitoring literature** ([5–14] sweep): broadly OK, no obvious cherry-picking. Mansouri et al. [43] is self-citation (author is on paper) but contextually appropriate and disclosed.

---

## Total fix burden estimate

| Item | Action | Wall-clock |
|---|---|---|
| C1: Replace `dimoudis2023utilizing` (×4) + add Sakoe-Chiba & Bergmeir bib entries | 4 line edits, 2 bib entries, recompile | 30 min |
| C2: Drop `Mugel2022` from main:92 + main:155, or replace with Niu et al. 2022 | 2 line edits (+1 bib entry if replacing) | 15 min |
| C3-B2/B3: supp:91 line-merge + supp:85 drop "(2014)" | 2 line edits | 5 min |
| C3-F1: Swap `rice2020...` for `karniadakis2021pinns` or use existing `sharma2022hybrid` | 1 line edit (+ optional 1 bib entry) | 10 min |
| C3-F3/F4: Cerezo overload review + remove redundant Rudolph at supp:153 | 2 line edits | 10 min |
| C3-N2: Strip all `% [N] removed: ... (R1-m1)` source comments from main + supp | grep-and-delete sweep, ~15 lines | 15 min |
| C4: Citation-style standardization (35 line edits, all mechanical) | sed pass + author-eye review | 45–60 min |
| C5: Add 4 missing-prior-art citations (Sakoe-Chiba, Bergmeir, COT-GAN, McClean barren) | 4 bib entries + 4–6 in-text edits + 2–3 sentences of new prose to integrate | 60–90 min |
| Recompile + verify .bbl + sanity-check `[N]` numbering downstream | bibtex + 2 pdflatex passes | 15 min |
| **TOTAL** | | **3.5–4.5 hours** |

**Critical path for resubmission (2026-06-17):** C1 and C2 are non-negotiable — both are directly identifiable mismatches that any first-pass reviewer will catch and that the author's own source comments document as known. C5-#4 (barren plateaus) is the highest-value addition for defending the headline quantum-cluster-dominates claim against a quantum-ML reviewer. Everything else is polish.
