# Paper Revision Blocks — References, Methods, Typos (PAPER-06..11)

> **Manuscript of record is Overleaf-external (RESEARCH Assumption A1).** The
> in-repo `paper/main.tex` / `paper/supp_material.tex` are READ-ONLY reference
> (D-14-18) and are NEVER edited. Every block below is a self-contained
> copy-paste unit keyed to a `\cite{}` key, a `\label{}`, or a verbatim anchor
> sentence so it applies regardless of the `.bib` / `.tex` physical location.
>
> **Number provenance (success criterion 5, D-14-16).** Every numeric literal
> in this file resolves to a value in some `results/*.json` and is
> proven by `verify_number_provenance.py --target
> docs/paper_blocks_refs_methods.md`. The authoritative source for any
> number that moved between submission and resubmission is
> `docs/reconciliation_note.md`.
>
> **Reviewer-comment IDs** map to `QGAN_Review_Response_Plan.md.pdf`; the
> per-comment point-by-point rebuttal lives in `docs/reviewer_response.md`.

---

## PAPER-06 — Reference Surgery (Reviewer comment: R1-m1)

Reviewer R1-m1 ("Misplaced / Weak References"): several `\cite{}` keys point at
work that does not support the attached claim. Each fix below is delivered as
**(a)** a `.bib` entry to add to the Overleaf `paper/bib.bib` and **(b)** a
sentence-rewrite keyed to the `\cite{}` key as it appears in
`paper/main.tex` — location-independent (A1).

> The bracketed numbers `[27] [28] [39] [18] [19] [41] [55]-[57] [59]` are the
> reviewer's numbering from the compiled PDF; the corresponding source
> `\cite{}` keys are identified from `paper/main.tex` and are the load-bearing
> handle for each fix.

### PAPER-06.a — Ref [27] (`\cite{esteban2017realvaluedmedicaltimeseries}`) — RCGAN is classical, not a QGAN

**R1-m1 rationale:** [27] is Esteban et al. RCGAN (a *classical* recurrent
conditional GAN for medical time series); it is currently attached to a
sentence implying *QGANs* have been applied to healthcare. Rewrite the claim so
the citation supports a *classical* GAN healthcare application, not a quantum one.

**Anchor (`paper/main.tex` §1.4 line ~92 and §2.4 line ~151), before:**

```latex
... with successful applications in finance \cite{orlandi2024enhancing}, healthcare \cite{esteban2017realvaluedmedicaltimeseries}, and optimization \cite{Mugel2022}.
```

**After (sentence-rewrite — keep the key, fix the claim):**

```latex
... with successful applications in finance \cite{orlandi2024enhancing} and optimization \cite{Mugel2022}; GANs more broadly, including classical recurrent variants, have also been applied to healthcare time series \cite{esteban2017realvaluedmedicaltimeseries}.
```

`.bib` entry (unchanged target — verify it reads as the classical RCGAN paper):

```bibtex
@article{esteban2017realvaluedmedicaltimeseries,
  title   = {Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs},
  author  = {Esteban, Crist{\'o}bal and Hyland, Stephanie L. and R{\"a}tsch, Gunnar},
  journal = {arXiv preprint arXiv:1706.02633},
  year    = {2017}
}
```

### PAPER-06.b — Ref [28] (`\cite{Mugel2022}`) — portfolio optimization, not QGAN evidence

**R1-m1 rationale:** [28] is a portfolio-optimization application and does not
establish a *QGAN* result. Reassign it to the optimization-application slot only
(it legitimately supports "optimization"), and remove it from any sentence that
implies it is QGAN-method evidence.

**Anchor, before (the `\cite{Mugel2022}` appears alongside QGAN method claims):**

```latex
... healthcare \cite{esteban2017realvaluedmedicaltimeseries}, and optimization \cite{Mugel2022}.
```

**After (reassign — keep as an *application* citation only, drop from method-claim sentences):**

```latex
... and optimization \cite{Mugel2022}.  % retained ONLY as an optimization-application example; removed from any QGAN-method-evidence sentence (R1-m1)
```

### PAPER-06.c — Ref [39] — too specialized for the quantum-kernel background; replace with Havlíček + Schuld & Killoran

**R1-m1 rationale:** replace the over-specialized [39] with the two canonical
quantum-kernel references the reviewer names.

**Anchor (`paper/main.tex` §2.4 line ~148, quantum kernels sentence), before:**

```latex
Relevant approaches include quantum-enhanced sampling, quantum kernels \cite{giraldo2025q2sar}, and variational quantum algorithms ...
```

**After (replace the quantum-kernel citation with the two canonical refs):**

```latex
Relevant approaches include quantum-enhanced sampling, quantum kernels \cite{havlicek2019supervised, schuld2019quantum}, and variational quantum algorithms ...
```

`.bib` entries to add:

```bibtex
@article{havlicek2019supervised,
  title   = {Supervised learning with quantum-enhanced feature spaces},
  author  = {Havl{\'\i}{\v{c}}ek, Vojt{\v{e}}ch and C{\'o}rcoles, Antonio D. and Temme, Kristan and Harrow, Aram W. and Kandala, Abhinav and Chow, Jerry M. and Gambetta, Jay M.},
  journal = {Nature},
  volume  = {567},
  year    = {2019}
}

@article{schuld2019quantum,
  title   = {Quantum Machine Learning in Feature Hilbert Spaces},
  author  = {Schuld, Maria and Killoran, Nathan},
  journal = {Physical Review Letters},
  volume  = {122},
  year    = {2019}
}
```

### PAPER-06.d — Ref [18] — GMM in construction, not GPR; replace with an actual GPR reference

**R1-m1 rationale:** [18] is a Gaussian-mixture construction application; the
sentence claims Gaussian *process* regression. Replace with the canonical GPR
reference (Rasmussen & Williams).

**Anchor (`paper/main.tex` §1.3 line ~84), before:**

```latex
Methods such as Gaussian process regression and multivariate statistical models can generate synthetic data based on estimated probability distributions.  \cite{chokwitthaya2020applying}
```

**After (point the GPR claim at the canonical GPR reference):**

```latex
Methods such as Gaussian process regression \cite{rasmussen2006gaussian} and multivariate statistical models can generate synthetic data based on estimated probability distributions.
```

`.bib` entry to add:

```bibtex
@book{rasmussen2006gaussian,
  title     = {Gaussian Processes for Machine Learning},
  author    = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  publisher = {MIT Press},
  year      = {2006}
}
```

### PAPER-06.e — Ref [19] — image super-resolution GAN; replace with a time-series GAN

**R1-m1 rationale:** [19] (`\cite{wang2018esrganenhancedsuperresolutiongenerative}`,
ESRGAN) is an *image* super-resolution GAN cited in a *time-series* synthetic
data context. Replace with a time-series-relevant GAN reference (TimeGAN — the
`\cite{yoon2019TimeGAN}` key is already present in the manuscript, so reuse it).

**Anchor (`paper/main.tex` §1.3 line ~86), before:**

```latex
Machine learning models such as variational autoencoders (VAEs) and generative adversarial networks (GANs) can learn directly from available data to generate synthetic samples.  \cite{wang2018esrganenhancedsuperresolutiongenerative, akkem2024comprehensive}
```

**After (swap the image-SR GAN for the already-defined time-series GAN key):**

```latex
Machine learning models such as variational autoencoders (VAEs) and generative adversarial networks (GANs), including time-series-specific architectures \cite{yoon2019TimeGAN}, can learn directly from available data to generate synthetic samples.  \cite{akkem2024comprehensive}
```

> `\cite{yoon2019TimeGAN}` is already defined in the manuscript (used at
> §2.1 and §2.4) — no new `.bib` entry required; the `wang2018esrgan...` key is
> simply removed from this sentence (R1-m1).

### PAPER-06.f — Ref [41] — adaptive rolling-median anomaly detection; replace with a rolling-window subsequence reference

**R1-m1 rationale:** [41] is an adaptive rolling-*median anomaly detection*
paper; the sentence is about rolling-*window subsequence extraction* for
training-data construction. Replace with a proper rolling-window subsequence
reference. The manuscript already cites `\cite{dimoudis2023utilizing}` for the
rolling-window technique at §3.1 / Supp §A.7 — reuse that key as the rolling-window
subsequence reference and drop the misattributed [41].

**Anchor (`paper/main.tex` §3.1 line ~172), before:**

```latex
Overlapping subsequences of length 10 with stride 2 were extracted using a rolling window approach \cite{dimoudis2023utilizing}.
```

**After (no change needed at this anchor — the correct key is already here;
the FIX is to remove [41] wherever it is attached to the rolling-window claim
and rely on `\cite{dimoudis2023utilizing}`):**

```latex
Overlapping subsequences of length 10 with stride 2 were extracted using a rolling window approach \cite{dimoudis2023utilizing}.  % [41] removed: it was adaptive rolling-median anomaly detection, not subsequence extraction (R1-m1)
```

> The rolling-window numbers (length 10, stride 2, 384 windows) are JSON-sourced
> — see `results/model_info.json` `dataset.window_length`,
> `dataset.window_stride`, `dataset.rolling_windows` — and are stated in the
> PAPER-08 block below. No hand-typed dataset number is introduced here.

### PAPER-06.g — Refs [55]-[57], [59] — VQE / option-pricing / QAOA / adversarial robustness; remove or replace

**R1-m1 rationale:** [55]-[57] and [59] are VQE, quantum option-pricing, QAOA,
and adversarial-robustness papers. None establishes that quantum interference
helps *bioprocess* generative learning; the reviewer asks for removal or
replacement with directly relevant citations.

**Anchor (`paper/supp_material.tex` §A.2.3 "Quantum Advantage for Generative Models",
the interference/optimization-landscape sentences ~line 138), before:**

```latex
... making them particularly suited for complex temporal dependencies found in bioprocess systems. \cite{Liu_2019, Stamatopoulos_2020} Furthermore, quantum algorithms demonstrate theoretical advantages in certain optimization landscapes, suggesting that QGANs may overcome some of the training instabilities ... \cite{farhi2014quantumapproximateoptimizationalgorithm, Cerezo_2021}
```

**After (remove the non-supporting keys; retain only the directly relevant
QGAN/NISQ-constraint citation `\cite{Cerezo_2021}`):**

```latex
... making them particularly suited for complex temporal dependencies found in bioprocess systems. Practical scalability nonetheless remains constrained by current NISQ-device noise and limited circuit depth \cite{Cerezo_2021}.  % [55]-[57],[59] (VQE / option-pricing / QAOA / adversarial-robustness) removed: they do not support a quantum-advantage-for-bioprocess-generation claim (R1-m1)
```

> No replacement `.bib` entries are added — the reviewer's accepted resolution
> is *removal* of the over-reaching claim, not substitution of a
> not-yet-demonstrated quantum-advantage citation (consistent with the
> PAPER-02 no-overclaiming lock, D-14-20).

### PAPER-06.h — Anchors RETAINED (reviewer-confirmed appropriate)

Per R1-m1, the reviewer explicitly confirms the following are correctly placed
and **MUST be retained unchanged**: `[21]-[23]`, `[34]-[36]`, `[61]`. No edit;
recorded here so the Overleaf editor does not "fix" a correct citation.

```text
KEEP (do NOT touch — R1-m1 "Keep as anchors"): [21]-[23], [34]-[36], [61]
```

---

## PAPER-07 — Add Bernal et al. AIChE Perspective (Reviewer comments: R1-m6, R2-2)

**R1-m6 / R2-2 rationale:** the reviewers ask for Bernal, Ajagekar, Harwood,
Stober, and Trenev, "Perspectives of quantum computing for chemical
engineering" to be cited in the Introduction around §1.3/§2 to ground the
quantum-for-chemical-engineering motivation (R2-2: the "classical limitations →
we need quantum" jump must be softened and *grounded in a domain reference*, not
asserted).

`.bib` entry to add:

```bibtex
@article{bernal2022perspectives,
  title   = {Perspectives of quantum computing for chemical engineering},
  author  = {Bernal, David E. and Ajagekar, Akshay and Harwood, Stuart M. and Stober, Spencer T. and Trenev, Dimitar and You, Fengqi},
  journal = {AIChE Journal},
  volume  = {68},
  year    = {2022}
}
```

**Insertion sentence — keyed to the §1.4 transition anchor (`paper/main.tex`
line ~88-92, end of §1.3 "Synthetic Data Generation Approaches" / start of §1.4
"Quantum Generative Adversarial Networks"):**

Before (current §1.3→§1.4 transition, line ~88):

```latex
Addressing these challenges requires novel methodological approaches that can effectively combine the strengths of both classical and quantum computational paradigms.

\subsection{Quantum Generative Adversarial Networks}
```

After (insert the grounded, *softened* motivation sentence citing Bernal et al.
— this simultaneously satisfies R2-2's "make the quantum jump measured"):

```latex
Addressing these challenges requires novel methodological approaches. These limitations motivate exploring alternative computational paradigms, including quantum approaches, which \emph{may} offer advantages for certain structured learning tasks in chemical and biochemical engineering \cite{bernal2022perspectives}.

\subsection{Quantum Generative Adversarial Networks}
```

> The same `\cite{bernal2022perspectives}` may also be added at the §2.4
> opening (`paper/main.tex` line ~146-148) where QML is introduced; one
> citation insertion, two valid anchor points (R1-m6 "Sections 1.3 and 2").

---

## PAPER-10 — Appendix A3 / Hybrid-GAN: relabel as proposed extension + clarify log-GAN vs Wasserstein discrepancy (Reviewer comment: R2-5a)

**R2-5a rationale:** Appendix A.3's systematic mathematical layout
(`paper/supp_material.tex` §A.3, `eq:balance`, `eq:constraint1`, `eq:constraint2`,
`eq:constitutive`, and the Hybrid-GAN objective at line ~158-165) reads as if
the Hybrid-GAN were *implemented and executed*, but the text says it is future
work — confusing. Resolution: relabel as a clearly-marked **proposed
extension**, remove any presentation that implies execution, AND clarify the
equation discrepancy — the **main-text WGAN-GP objective (Eq. `eq:wgangp`,
Earth-Mover/Wasserstein form)** is what was actually trained, whereas the
**Supplementary A.3 Hybrid-GAN objective uses the original log-GAN
(Jensen-Shannon) `\log D` formulation** plus a physics-residual term and is
*not* the trained objective.

### PAPER-10.a — A.3 section-header + lead-in relabel

**Anchor (`paper/supp_material.tex` line ~142 `\subsection{Hybrid-GAN Framework for
Future Work}`), before:**

```latex
\subsection{Hybrid-GAN Framework for Future Work}

\subsubsection{Mathematical Formulation}

The Hybrid-GAN-mechanistic structure represents an advanced architecture that integrates generative adversarial networks with physics-informed mechanistic components ...
```

**After (explicit "proposed extension — not implemented" banner):**

```latex
\subsection{Proposed Extension (Outlook): Hybrid-GAN-Mechanistic Framework}

\subsubsection{Proposed Mathematical Formulation (not implemented in this study)}

\emph{The following Hybrid-GAN-mechanistic structure is a proposed extension and was not implemented, trained, or evaluated in this work; it is included to motivate future research and is presented as a formulation only.} The proposed Hybrid-GAN-mechanistic structure would integrate generative adversarial networks with physics-informed mechanistic components ...
```

### PAPER-10.b — log-GAN vs Wasserstein discrepancy clarification (insert immediately after the A.3 Hybrid-GAN objective equation, ~line 165)

```latex
\paragraph{Relationship to the trained objective.} The objective above is
written in the original (log-GAN / Jensen--Shannon) form,
$\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$, augmented with a
mechanistic-residual penalty. This is deliberately distinct from the objective
actually trained in this study, which is the Wasserstein GAN with gradient
penalty (Eq.~\ref{eq:wgangp}, Earth-Mover formulation). The log-GAN form is
retained here only because it most transparently exposes where a mechanistic
residual term would attach; any future implementation of this proposed
extension would substitute the Wasserstein--gradient-penalty critic used
throughout the present work (Section~3) for the $\log D$ discriminator, so that
the Hybrid-GAN inherits the training stability documented for the deployed
QWGAN-GP rather than the mode-collapse and vanishing-gradient pathologies of
the original GAN objective.
```

> No numbers introduced in PAPER-10 (pure formulation/relabel). The deployed
> objective's hyperparameters (N_CRITIC, λ, LR) are reported in the
> PAPER-08-adjacent Methods/Training-Protocol material and sourced from
> `results/model_info.json`.

### PAPER-10.c — Supp Table A2 caveat (R1-M5 / R2-5a — aspirational, not validated)

**Anchor (`paper/supp_material.tex` §A.3.3 Table `\label{tbl:various_approaches}`,
caption line ~225 and trailing sentence line ~248), before:**

```latex
This table lists a comparative overview of individual methods and demonstrates the advantages of Hybrid-GAN structures for future work based on the current study.
```

**After (explicit aspirational caveat — the Hybrid-GAN "Proposed" row is not an
experimental result):**

```latex
This table is an \emph{aspirational} qualitative comparison: the ``Hybrid-GAN (Proposed)'' row reflects expected properties of the proposed extension and was \emph{not} experimentally validated in this study. It is provided to motivate future work, not as a demonstrated result.
```

---

## PAPER-11 — Typos + Notation Unification (Reviewer comment: R1-m7)

Each R1-m7 checklist item is one keyed before→after block.

### PAPER-11 / R1-m7 item 1 — Fig. 6 x-axis "Laas" → "Lags"

**Where:** ACF figure x-axis label (rendered figure; in this repo the
regenerated ACF figures are `figures/acf_*.png` whose
companion JSON carries the corrected `xlabel`). In the manuscript the relevant
caption is `\label{fig:acf}` (`paper/main.tex` line ~251).

```text
BEFORE: x-axis tick/label reads "Laas"
AFTER : x-axis tick/label reads "Lags"
```

> The regenerated figure suite already emits the corrected label; verify the
> Overleaf-embedded `acf.png` is replaced by the regenerated
> `figures/acf_iqp_sel_55_repro.png` (PAPER-09 / Plan 04).

### PAPER-11 / R1-m7 item 2 — Missing space: "Figure A5).This" → "Figure A5). This"

**Anchor (`paper/main.tex` §4.2 line ~261), before:**

```latex
... within a unified feedback loop (Figure A5).%~\ref{fig:qgan_schemcatic}).
This framework provides a systematic approach ...
```

**After (ensure a space after the parenthesis when the comment is removed/Overleaf renders it):**

```latex
... within a unified feedback loop (Figure~A5). This framework provides a systematic approach ...
```

### PAPER-11 / R1-m7 item 3 — "LUCY ©photobioreactor" → "LUCY® photobioreactor"

**Anchor (`paper/supp_material.tex` Figure `\label{fig:lucy}` caption line ~346), before:**

```latex
\caption{Schematic of the 300L LUCY \textcopyright  photobioreactor, sensors, and actuators.}
```

**After (use the registered-trademark mark, not copyright, and fix spacing):**

```latex
\caption{Schematic of the LUCY\textregistered{} photobioreactor, sensors, and actuators.}
```

> Note `\textcopyright` (©) is replaced by `\textregistered` (®) to match the
> main-text usage `LUCY\textregistered` at `paper/main.tex` line ~178. The
> "300L" descriptor is corrected by PAPER-11 / R1-m7 item 4.

### PAPER-11 / R1-m7 item 4 — Fix the incomplete 300L/20L sentence + the malformed mid-sentence `\label`

**Anchor (`paper/main.tex` §3.2 line ~178), before (a literal
`\label{fig:lucy}` is incorrectly embedded mid-sentence, and the 300L/20L
description is malformed):**

```latex
Time-series data were collected from a 20-liter photobioreactor (LUCY\textregistered, Synoxis Algae) designed for laboratory and pilot-scale cultivation of microalgae, see Figure \label{fig:lucy}  for the 300L configuration of the 20L version of LUCY.
```

**After (remove the malformed mid-sentence `\label`, use `\ref`, and fix the
scale description so it is a complete, correct sentence):**

```latex
Time-series data were collected from a 20-liter photobioreactor (LUCY\textregistered, Synoxis Algae) designed for laboratory- and pilot-scale cultivation of microalgae; the system is also available in a larger 300-liter configuration. The 20-liter unit used in this study is shown schematically in Figure~\ref{fig:lucy}.
```

> The orphan `\label{fig:lucy}` belongs on the actual figure environment in
> `paper/supp_material.tex` (line ~347), which already declares `\label{fig:lucy}`;
> the mid-sentence occurrence in the main text is a typo and is deleted.

### PAPER-11 / R1-m7 item 5 — "Dry Biomass" → "dry biomass"

**Anchor (`paper/main.tex` abstract line ~49), before:**

```latex
... focusing on Optical Density as a key measurement for Dry Biomass estimation.
```

**After (lowercase, not a proper noun):**

```latex
... focusing on optical density as a key measurement for dry biomass estimation.
```

### PAPER-11 / R1-m7 item 6 — Standardize "bio-manufacturing" vs "biomanufacturing" → "biomanufacturing"

**Anchors:** `paper/main.tex` abstract line ~49 ("bio-manufacturing") and
Plain Language Summary line ~59 ("biomanufacturing"). Standardize on the closed
form **biomanufacturing** everywhere.

```latex
% line ~49 BEFORE: Data scarcity in bio-manufacturing poses challenges ...
% line ~49 AFTER : Data scarcity in biomanufacturing poses challenges ...
% line ~59 already "biomanufacturing" — leave as the canonical form
```

### PAPER-11 / R1-m7 item 7 — Ref [39] title typo "Approac" → "Approach"

**Where:** the `.bib` entry for the reference the manuscript numbered [39]
(replaced in PAPER-06.c). If the *replacement* references are used the typo
disappears; if the original [39] entry is retained anywhere its title must read
"Approach" not "Approac".

```bibtex
% BEFORE (typo in title): ... Approac ...
% AFTER  (corrected)     : ... Approach ...
```

### PAPER-11 / R1-m7 item 8 — Ref [51] title capitalization

**Where:** the `.bib` entry the manuscript numbered [51]. Standardize the title
to sentence case consistent with the rest of `paper/bib.bib` (protect proper nouns
with `{}`).

```bibtex
% Standardize [51] title capitalization to match bib.bib house style:
% wrap acronyms/proper nouns in braces, e.g. title = {... {GAN} ... {WGAN-GP} ...}
```

### PAPER-11 / R1-m7 item 9 — "QWGAN-GPs" → "QWGAN-GP" (Concluding Remarks)

**Anchor (`paper/main.tex` §5 Concluding Remarks line ~296), before:**

```latex
By successfully demonstrating that QWGAN-GPs can generate synthetic time-series data ...
```

**After (singular — the model name is not pluralized):**

```latex
By successfully demonstrating that the QWGAN-GP can generate synthetic time-series data ...
```

### PAPER-11 / R1-m7 item 10 — Unify the return-variable symbol (log δ vs ς)

**Anchor (`paper/main.tex` `\label{fig:acf}` caption line ~250, and
`paper/supp_material.tex` §A.7 Eq.~`eq:lambert_w_s9` / Data Transformation line
~354-379), before (the ACF caption uses "log $\delta$" while Supp uses $\nu$ /
$\varsigma$ inconsistently):**

```latex
% main caption: ... log returns (log $\delta$), where $\delta$ is the return calculation.
```

**After (pick ONE symbol — use $r_t$, matching the Supp definition
$r_t = \ln(\mathrm{OD}_t) - \ln(\mathrm{OD}_{t-1})$ at supp line ~356, and use
it everywhere):**

```latex
% main caption: ... log returns $r_t$, where $r_t = \ln(\mathrm{OD}_t) - \ln(\mathrm{OD}_{t-1})$ is the per-step log return (Supp. Eq.~A8).
% Replace every stray "log $\delta$" / "$\varsigma$" return symbol with $r_t$ for a single consistent notation.
```

### PAPER-11 / R1-m7 item 11 — Enlarge Figures 2-6

**Where:** the main-text result figures `\label{fig:DTWD}`, `\label{fig:pdf}`,
`\label{fig:cdf}`, `\label{fig:qq}`, `\label{fig:acf}` (`paper/main.tex`
lines ~196-252). Increase rendered size for legibility.

```latex
% For each of fig:DTWD, fig:pdf, fig:cdf use a wider box:
%   \includegraphics[width=\columnwidth]{...}  ->  \includegraphics[width=\linewidth]{...}
% For the full-width figure* environments (fig:qq, fig:acf) keep width=\linewidth
% and, if still small, promote single-column figures to figure* (two-column span).
% The regenerated high-DPI sources (figures/*.pdf, dpi=150,
% bbox_inches="tight", Plan 04) replace the Overleaf-embedded bitmaps so the
% enlargement does not pixelate.
```

---

## PAPER-08 — Dataset Details in Methods (Reviewer comment: R1-m2)

**R1-m2 rationale:** the reviewer asks for the raw number of time points, the
number of rolling windows, the train/val/test split ratios and counts, and the
number of independent runs to be reported in Methods. Every number in the block
below is rendered FROM `results/model_info.json` (the `dataset` block,
DERIVED from `data.csv` + the locked window config — D-14-16, success criterion
5) and from `model_info.json` `seed_set`; none is hand-typed. Companion doc:
`docs/dataset_stats.md` (also rendered from the same JSON).

> **Consolidated Methods document.** The same dataset numbers — plus the
> model registry, training protocol, hardware/software stack, and
> reproducibility contract — are consolidated in `docs/methods_full.md`
> §1 (Dataset) through §5 (Reproducibility), rendered from
> `results/methods_full.json` + `results/model_info.json` +
> the 5 config-lock JSONs + `results/classical_architectures.json`
> + `results/framework_versions.json` (Plan 14-11). The
> copy-paste LaTeX block below remains the single load-bearing manuscript
> insertion; methods_full.md is the reviewer-facing audit companion.

**Insertion point:** `paper/main.tex` §3.2 "Photobioreactor Experimental
Setup", immediately after the data-logging sentence (line ~180, "...data logged
at 10-minute intervals by an internal data acquisition system."), as a new
"Dataset and Preprocessing" paragraph before §4.

**Copy-paste LaTeX block:**

```latex
\paragraph{Dataset and preprocessing.}
The study uses a single LUCY photobioreactor cultivation campaign
% source: results/model_info.json#dataset.independent_campaigns (=1)
% source: results/model_info.json#dataset.raw_csv_rows (=778)
comprising 778 raw optical-density time points logged at 10-minute intervals.
% source: results/model_info.json#dataset.log_return_rows (=777)
First differencing into log-returns yields 777 log-return observations
($r_t = \ln \mathrm{OD}_t - \ln \mathrm{OD}_{t-1}$), which are standardized to
zero mean and unit variance, passed through an inverse Lambert~$W$ heavy-tail
correction, and rescaled to $[-1, 1]$.
% source: results/model_info.json#dataset.window_length (=10)
% source: results/model_info.json#dataset.window_stride (=2)
% source: results/model_info.json#dataset.rolling_windows (=384)
Overlapping subsequences of length 10 with stride 2 are then extracted with a
rolling window, producing 384 training windows.
% source: results/model_info.json#dataset.train_windows (=384)
% source: results/model_info.json#dataset.val_windows (=0)
% source: results/model_info.json#dataset.test_windows (=0)
Because only one independent campaign is available, all 384 windows are used for
training; no held-out validation or test split is carved out, as a 384-window
single campaign is too small to support a held-out split without severely
under-powering training (stated openly per the calibration-honesty standard,
R1-M5). Multi-campaign generalization is identified as future work.
% source: results/model_info.json#seed_set ([42,43,44,45,46])
All reported quantitative results are aggregated over 5 independent random
seeds (42, 43, 44, 45, 46) and reported as mean $\pm$ standard deviation.
```

> **Render-from-JSON contract.** Every numeric literal above carries a
> `% source: results/model_info.json#<path>` annotation. The values
> are: `independent_campaigns`=1, `raw_csv_rows`=778, `log_return_rows`=777,
> `window_length`=10, `window_stride`=2, `rolling_windows`=384,
> `train_windows`=384, `val_windows`=0, `test_windows`=0, and the 5-element
> `seed_set` [42, 43, 44, 45, 46]. To regenerate, re-run
> `scripts/run_model_info.py`; `scripts/verify_number_provenance.py` proves
> every literal here resolves to `model_info.json`.

---

## PAPER-09 — Per-Metric Evaluation Scale in Methods (Reviewer comment: R1-m3)

**R1-m3 rationale:** the reviewer asks the Methods to state explicitly, for
every evaluation metric, whether it is computed on the transformed (log-return,
training-space) scale or the original optical-density (OD) scale. Source:
`results/fidelity_dualscale.json` (dual-scale rows, 2000ep, both
`scale = "log_return"` and `scale = "OD"`, `metric_helpers =
revision.core.eval`). The block below is a Methods table that labels every
metric family with its evaluation scale and gives the headline 55-param IQP:SEL
quantum value on each scale (single representative seed 42, Pipeline~B — the
native preprocessing pipeline) so the dual-scale reporting is concrete.

**Insertion point:** `paper/main.tex` §4.1 "Results", immediately before the
"Dynamic Time Warping." paragraph (line ~189), as an "Evaluation scale" Methods
paragraph + table.

**Copy-paste LaTeX block:**

```latex
\paragraph{Evaluation scale.}
Every fidelity metric is reported on \emph{both} the transformed log-return
scale (the space the generator is trained in) and the original optical-density
(OD) scale (obtained by inverting the preprocessing pipeline), so that the
physical-unit fidelity is explicit (R1-m3). Table~\ref{tbl:eval_scale} states
the scale of each metric family and the headline 55-parameter IQP:SEL quantum
generator value on each scale.

\begin{table}[h]
\centering
\caption{Evaluation metrics and the scale on which each is computed. Values are
for the 55-parameter IQP:SEL quantum generator, Pipeline~B (native
preprocessing), representative seed.}
\label{tbl:eval_scale}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{Transformed (log-return)} & \textbf{Original (OD)} & \textbf{Reported on} \\
\midrule
EMD                 & 0.1209437521974767 & 0.022937980562900886 & both scales \\
DTW (mean)          & 0.9343404967853801 & 0.2648187239898106   & both scales \\
Moment: mean        & 0.1232816505352802 & 1.4070339987298917   & both scales \\
Moment: std         & 0.07206523080095577 & 0.8839933147241738  & both scales \\
Moment: skewness    & -0.03262545792264619 & 1.3655798412279025 & both scales \\
Moment: kurtosis    & -0.08214151764010014 & 0.7768219209344931 & both scales \\
ACF (lag~1, mean)   & -0.0814285239177223 & 0.6965233188661055  & both scales \\
\bottomrule
\end{tabular}
\end{table}

PDF/CDF, Q--Q, and ACF diagnostic plots in the main text are shown on the
transformed log-return scale (the training space); the corresponding
original-OD-scale versions are provided as regenerated figures
(\texttt{figures/acf\_iqp\_sel\_55\_repro}, dual-scale). DTW,
EMD, and the distributional moments are reported on both scales as above.
```

> **Render-from-JSON contract.** Every value in the table is the exact stored
> `value` for `model_kind = "quantum"`, `pipeline = "B"`, `seed = 42` in
> `results/fidelity_dualscale.json` at the named `metric_name` /
> `scale`: `emd`, `dtw_mean`, `moment_mean`, `moment_std`, `moment_skewness`,
> `moment_kurtosis`, `acf_lag1_mean` — each present on both
> `scale = "log_return"` and `scale = "OD"`. No value is hand-typed or rounded
> (full stored precision is used so the substring resolves);
> `scripts/verify_number_provenance.py` proves every literal resolves to
> `fidelity_dualscale.json`.

### PAPER-09 — Story-completeness figure citations (Plan 14-10)

> The Methods evaluation-scale table above reports a single representative
> seed (seed 42, Pipeline B). Six story-completeness figures from Plan
> 14-10 support specific claims in the surrounding §3 Methods / §4 Results
> narrative and should be cited at the points they each support:
>
> - **Training protocol claim (R1-M4):** the per-epoch convergence behavior
>   of every matched-2000ep model is rendered at
>   `figures/training_convergence_all_models.{png,pdf}`
>   (companion JSON `training_convergence_all_models.json`; source = 45
>   per-run metrics.json + `results/headline_canonical.json`).
>   Cite alongside the Training Protocol paragraph (PAPER-08 / methods_full.md §3).
>
> - **TSTR utility claim (R1-M2):** the cross-model TSTR R²/MAE/RMSE bars
>   for Pipelines A and B are rendered at
>   `figures/tstr_crossmodel.{png,pdf}` (companion JSON
>   `tstr_crossmodel.json`; source = `results/tstr.json`).
>   Negative R² is plotted honestly per the companion JSON's
>   `caption_note`. Cite alongside the §4.1 utility-evaluation subsection.
>
> - **Per-model fidelity / failure modes (R1-M5):** the diagnostic grid
>   (distribution overlay × ACF lag-1 × log-return EMD, 9 models ordered
>   by ascending OD EMD) is rendered at
>   `figures/failure_modes_summary.{png,pdf}` (companion
>   JSON `failure_modes_summary.json`; source =
>   `results/matched2000_dualscale.json` + per-model dist/acf
>   companion JSONs). Cite alongside the §4.1 per-model fidelity claim
>   and the R1-M5 calibration discussion.
>
> - **Multi-seed mean ± std claim (R1-M4):** the per-seed EMD trajectories
>   underneath the seed-aggregated mean ± std are rendered as a 3×3 facet
>   grid at `figures/seed_variance_per_model.{png,pdf}`
>   (companion JSON `seed_variance_per_model.json`; source = 45 per-run
>   metrics.json). Cite alongside the multi-seed reporting paragraph
>   (PAPER-08 / methods_full.md §3).
>
> - **Noise-model sensitivity claim (R1-M4):** EMD vs depolarizing /
>   amplitude-damping noise level is rendered at
>   `figures/noise_robustness_quantum.{png,pdf}`
>   (companion JSON `noise_robustness_quantum.json`; source =
>   `results/noise_model_sensitivity.json`). Cite alongside the
>   noise-sensitivity discussion (R2-1 / R1-M4 backend statement).
>
> - **Shot-noise sensitivity claim (R1-M4):** EMD vs shot count (log-x)
>   with the analytic-statevector reference line is rendered at
>   `figures/shot_noise_robustness.{png,pdf}` (companion
>   JSON `shot_noise_robustness.json`; source =
>   `results/shot_noise_sensitivity.json`). Cite alongside the
>   §3 backend statement (analytic statevector, no shot noise in the
>   reported headline values) and the R1-M4 shot-noise paragraph.

---

<!-- End of PAPER-06..11 LaTeX-blocks file. The AIChE per-reviewer
     point-by-point rebuttal is docs/reviewer_response.md. -->
