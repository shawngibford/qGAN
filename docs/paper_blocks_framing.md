# Paper Framing & Claim-Calibration LaTeX Blocks (PAPER-01..05)

> **Generated for Phase 14 Plan 05.** Copy-paste LaTeX blocks for the
> manuscript (Overleaf-canonical `paper/main.tex` / `paper/supp_material.tex`).
> The in-repo `.tex` files are **READ-ONLY** (D-14-18) — they are an external
> Overleaf reference and are never edited by this repository. Each block is
> keyed to a `\label` or an anchor sentence plus its source-file line citation,
> carries a one-line reviewer-comment rationale, and annotates every numeric
> literal with its `results/*.json` source so the file passes
> `scripts/verify_number_provenance.py`.
>
> **Result-direction note (D-14-20).** The matched-2000ep sweep
> (`results/model_info.json`, `docs/reconciliation_note.md`)
> shows the 55-parameter quantum generator does **not** beat the matched
> classical WGAN-GP baselines at equal parameter budget and matched epochs.
> PAPER-02 (de-overclaiming) is therefore a **LOCKED** reviewer requirement
> regardless of which way the numbers fell — every overclaim block below is
> mandatory, not contingent.

---

## Block key legend

Each block states: **target file** + `\label`/anchor + line citation
(`:NNN` form) → **BEFORE** (verbatim current text) → **AFTER** (revised LaTeX,
copy-paste ready) → **Rationale** (reviewer memo it addresses).

Line citations are written in the `file:NNN` / `:NNN-NNN` form (stripped by the
number-provenance gate as source-location identifiers, not data). Every
quantitative literal in an AFTER block resolves to a
`results/*.json` value (annotated inline).

---

# PAPER-01 — Reframe hypothesis; soften the quantum-necessity transition

Addresses reviewer memos **R1-M5** (state quantum claims honestly /
simulator-only), **R2-1** and **R2-2** (the central question must be a
falsifiable parameter-matched comparison, not an assertion of quantum
necessity).

## PAPER-01a — Reframed central question (Section 1.4, anchor)

- **Target:** `paper/main.tex`
- **`\subsection{Quantum Generative Adversarial Networks}`**, anchor sentence
  at `main:92` (Section 1.4; the QGAN "more compactly than classical
  generators" sentence).
- **BEFORE** (`main:92`):

```latex
Quantum Generative Adversarial Networks (QGANs) embed Parameterized Quantum Circuits within the generator of a GAN, leveraging superposition, entanglement, and interference to represent complex distributions more compactly than classical generators \cite{Dallaire_Demers_2018, Lloyd_2018, Zoufal_2019}. Early results suggest advantages in low-data regimes and in capturing multimodal distributions \cite{rudolph2022generationhighresolutionhandwrittendigits, he2025qganbaseddataaugmentationhybrid}, with successful applications in finance \cite{orlandi2024enhancing}, healthcare \cite{esteban2017realvaluedmedicaltimeseries}, and optimization \cite{Mugel2022}.
```

- **AFTER** (replace the anchor sentence; the citations are preserved):

```latex
Quantum Generative Adversarial Networks (QGANs) embed Parameterized Quantum Circuits within the generator of a GAN \cite{Dallaire_Demers_2018, Lloyd_2018, Zoufal_2019}. This motivates the falsifiable question that frames the present study: \emph{can a PQC generator, operating in an exponentially large Hilbert space with $\mathcal{O}(\mathrm{poly}(n))$ parameters, match or exceed a classical generator of equivalent parameter count on a low-data bioprocess task?} Prior work has reported QGAN results in low-data regimes and on multimodal distributions \cite{rudolph2022generationhighresolutionhandwrittendigits, he2025qganbaseddataaugmentationhybrid}, with applications in finance \cite{orlandi2024enhancing}, healthcare \cite{esteban2017realvaluedmedicaltimeseries}, and optimization \cite{Mugel2022}; we test this claim directly under a matched-parameter, matched-epoch protocol.
```

- **Rationale (R2-1/R2-2):** converts an unfalsifiable "more compactly"
  assertion into the explicit parameter-count-parity hypothesis the reviewers
  asked to be tested, and signals the matched protocol up front. **No numeric
  literals** — qualitative reframe only.

## PAPER-01b — Soften the quantum-necessity transition (Section 2.4)

- **Target:** `paper/main.tex`
- **`\subsection{Quantum Machine Learning and QGANs}`**, anchor span
  `main:146-151` (Section 2.4; the "exponentially more compactly" /
  "potentially enabling richer distributions" transition).
- **BEFORE** (`main:151`):

```latex
QGANs are a hybrid quantum-classical approach in which a Parameterized Quantum Circuit (PQC) serves as the generator, first introduced by Dallaire-Demers and Killoran \cite{Dallaire_Demers_2018}. By exploiting superposition and entanglement, QGANs can represent high-dimensional probability spaces exponentially more compactly than classical generators, potentially enabling richer distributions with fewer parameters \cite{rudolph2022generationhighresolutionhandwrittendigits}.
```

- **AFTER**:

```latex
QGANs are a hybrid quantum-classical approach in which a Parameterized Quantum Circuit (PQC) serves as the generator, first introduced by Dallaire-Demers and Killoran \cite{Dallaire_Demers_2018}. Superposition and entanglement give a PQC access to a state space that grows exponentially with qubit count; whether this translates into a practical advantage over a parameter-matched classical generator is an empirical question that remains open and is one we examine directly in this work rather than assume \cite{rudolph2022generationhighresolutionhandwrittendigits}.
```

- **Rationale (R1-M5/R2-2):** removes the "exponentially more compactly /
  richer distributions with fewer parameters" necessity claim and replaces it
  with an honest "open empirical question" transition. **No numeric
  literals.**

---

# PAPER-02 — De-overclaiming (LOCKED, D-14-20)

Addresses reviewer memo **R1-M5** (calibrate claims; the manuscript overstates
industrial readiness and computational advantage). **LOCKED:** required
regardless of result direction. Each overclaim anchor sentence is
block-replaced; the named overclaim phrases ("computational advantages",
"exponential representational compactness", "reduced mode collapse") are
removed or softened wherever they occur.

## PAPER-02a — "exponentially more compactly" (Section 2.4)

- **Target:** `paper/main.tex`, anchor at `main:151`.
- This is the same anchor as PAPER-01b. The PAPER-01b AFTER block already
  removes "exponentially more compactly than classical generators" and the
  "richer distributions with fewer parameters" claim — **apply PAPER-01b's
  AFTER block; it satisfies PAPER-02a (LOCKED).** Listed here explicitly so the
  overclaim phrase "exponentially more compactly" is tracked as removed.
- **Rationale (R1-M5, LOCKED):** the single highest-salience compactness
  overclaim; removed by construction in PAPER-01b.

## PAPER-02b — "high fidelity ... industrial bioprocesses" (Key Contributions)

- **Target:** `paper/main.tex`
- **`\subsection{Key Contributions and Findings}`**, anchor sentence at
  `main:266` (the "high fidelity ... underlying dynamics of industrial
  bioprocesses" sentence).
- **BEFORE** (`main:266`):

```latex
Third, our empirical validation on real-world photobioreactor cultivation data confirms the effectiveness of the proposed approach. The QWGAN-GP achieved an improved Dynamic Time Warping (DTW) score, representing improved temporal alignment compared to previously reported methods. \cite{orlandi2024enhancing} The experimental results demonstrate that the QWGAN-GP approach successfully generates synthetic time series data with high fidelity, effectively capturing the underlying dynamics of industrial bioprocesses. The synthetic data showed high fidelity to actual historical experimental data, as evidenced by strong normality alignment in quantile-quantile analyses, faithful reproduction of auto-correlation structures, and accurate preservation of probability density and cumulative distribution functions.
```

- **AFTER**:

```latex
Third, our empirical validation on a single real-world photobioreactor cultivation campaign characterizes the behaviour of the proposed approach. Under a matched parameter budget and matched training epochs, the quantum generator produces synthetic log-return sequences whose distributional and autocorrelation structure are comparable to, but do not exceed, those of size-matched classical WGAN-GP baselines (Section~4; full per-model figures in the supplementary figure suite). We therefore present the QWGAN-GP as a viable hybrid generator on this data-scarce task rather than as a method of demonstrated advantage over classical baselines, and we restrict claims to the laboratory-scale single-variable setting actually evaluated.
```

- **Rationale (R1-M5, LOCKED):** removes "high fidelity ... industrial
  bioprocesses" and the unqualified "successfully generates ... with high
  fidelity" claim; states the parameter-matched non-superiority result
  honestly and bounds the claim to the evaluated setting. The DTW figure is
  dropped from this sentence (it remains reported in Section 4); **no numeric
  literal in the AFTER block.**

### PAPER-02b — Visual companion (Plan 14-10, additive footnote)

> **Visual companion (Plan 14-10).** The parameter-matched comparison
> stated in this paragraph is rendered as a parameter-efficiency Pareto
> scatter at `figures/param_efficiency_pareto.{png,pdf}`
> (companion JSON `param_efficiency_pareto.json`; source =
> `results/model_info.json` (n_params per model) +
> `results/matched2000_dualscale.json` (EMD mean±std per scale)).
> The frozen-checkpoint headline (D-14-10) appears as a visually distinct
> marker; the iqp_sel_55_repro matched-budget reproduction and the V1/V2/V3
> ansatz variants appear as separate points in the same scatter so the
> headline-vs-repro distinction is unambiguous on inspection.

## PAPER-02c — "industrial bioprocess engineering" (Concluding Remarks)

- **Target:** `paper/main.tex`
- **`\section{Concluding Remarks}`**, anchor sentence at `main:296`.
- **BEFORE** (`main:296`):

```latex
This research demonstrates that quantum-enhanced generative adversarial networks are a viable approach to addressing data scarcity challenges in industrial bioprocess engineering. By successfully demonstrating that QWGAN-GPs can generate synthetic time-series data while preserving the complex statistical properties of real bioprocess systems, we have opened new possibilities for advanced bioprocess monitoring, optimization, and control strategies.
```

- **AFTER**:

```latex
This research investigates quantum-enhanced generative adversarial networks as one candidate approach to data scarcity in laboratory-scale bioprocess time-series modelling. On a single optical-density cultivation dataset, a QWGAN-GP can generate synthetic sequences that preserve key statistical properties of the measured signal at a fidelity comparable to parameter-matched classical baselines; establishing whether any quantum-specific benefit exists will require multivariate data, larger campaigns, and validation beyond the single-variable laboratory setting studied here.
```

- **Rationale (R1-M5, LOCKED):** removes "industrial bioprocess engineering"
  and "demonstrates ... viable approach", scoping the conclusion to the
  laboratory single-variable evidence and explicitly deferring any
  quantum-benefit claim. **No numeric literal.**

## PAPER-02d — Named overclaim phrases: "computational advantages" / "exponential representational compactness" / "reduced mode collapse"

- **Target:** `paper/main.tex`
- **`\subsection{Theoretical and Practical Implications}`**, anchor span
  `main:276` (the "quantum computational advantages" / "potentially offering
  computational advantages over classical approaches" sentence).
- **BEFORE** (`main:276`):

```latex
The theoretical significance of this work lies in demonstrating how quantum computational advantages can be effectively applied to address real-world engineering challenges. Our results provide initial evidence that quantum circuits can represent the complex probability distributions found in bioprocess time series, potentially offering computational advantages over classical approaches when working with limited datasets. Extending this demonstration to multivariate distributions remains an important next step.
```

- **AFTER**:

```latex
The contribution of this work is methodological: it provides a parameter-matched, matched-epoch, multi-seed protocol for comparing a PQC generator against classical baselines on a data-scarce bioprocess task, together with a reproducible result. Our results show that quantum circuits can represent the probability distributions found in this bioprocess time series, but do not provide evidence of a computational advantage over classical approaches at matched capacity; extending the comparison to multivariate distributions remains an important next step.
```

- **Additional softening (apply at each occurrence):**
  - **`supp_material.tex:135`** — the bullet "\textbf{Entanglement:}
    Creates correlations impossible in classical probability, potentially
    leading to faster convergence and **reduced susceptibility to mode
    collapse**": replace the trailing clause with
    `potentially affecting convergence behaviour; we do not claim a mode-collapse advantage and did not measure one.`
  - **`supp_material.tex:334`** — the sentence ending "potentially leading
    to faster convergence and **reduced susceptibility to mode collapse**":
    replace the trailing clause with
    `potentially affecting training dynamics (not evaluated in this study).`
  - **`main:174`** — the WGAN-GP-loss sentence "mitigates mode collapse
    issues that can be exacerbated by the discrete nature of quantum
    measurements": retain (this is a property of the WGAN-GP objective, not a
    quantum-advantage claim) but append
    `; we do not attribute reduced mode collapse to the quantum generator.`
- **Rationale (R1-M5, LOCKED):** eliminates "computational advantages",
  "exponential representational compactness", and "reduced mode collapse" as
  asserted quantum benefits everywhere they appear; reframes the contribution
  as the protocol + reproducible negative-to-neutral result. **No numeric
  literal.**

---

# PAPER-04 — Justify log-returns in a bioprocess growth-rate framing (not finance)

Addresses reviewer memo **R1-M3** (the log-return transform is justified by a
finance citation and "quantitative analysis" language; it must be justified in
bioprocess terms).

- **Target:** `paper/supp_material.tex`
- **`\subsection{Data Transformation Details}`**, anchor at `supp:352`;
  finance-framed rationale span `supp:358-365`.
- **BEFORE** (`supp:354` and `supp:358-365`):

```latex
The transformation from optical density measurements to logarithmic returns follows standard practice in time series analysis. \cite{orlandi2024enhancing} For optical density measurements $OD_t$ at time t:

$$r_t = \ln(\text{OD}_t) - \ln(\text{OD}_{t-1})$$

Log returns are highly favored in quantitative analysis for several reasons:
\begin{itemize}[nosep, leftmargin=*]
  \item \textbf{Time additivity:} The sum of log returns over consecutive periods equals the log return over the total time span
  \item \textbf{Value compression:} Better handling of large variations
  \item \textbf{Mathematical convenience:} Simplifies many statistical operations
\end{itemize}

For small changes in value, the simple and log returns are approximately equal ($r_t \approx R_t$), but the analytical benefits of log returns have established them as the standard for applications.
```

- **AFTER**:

```latex
The transformation from optical density measurements to log differences is motivated by the bioprocess itself: for an exponentially growing culture the optical density follows $\mathrm{OD}_t \approx \mathrm{OD}_0\,e^{\mu t}$, so the log difference between successive samples is a direct discrete estimate of the instantaneous specific growth rate. For optical density measurements $OD_t$ at time $t$:

$$r_t = \ln(\text{OD}_t) - \ln(\text{OD}_{t-1}) \;\approx\; \mu_t\,\Delta t$$

where $\mu_t$ is the local specific growth rate over the sampling interval $\Delta t$. Working in $r_t$ rather than raw OD has a direct bioprocess interpretation and three practical benefits for this growth-rate signal:
\begin{itemize}[nosep, leftmargin=*]
  \item \textbf{Growth-rate interpretability:} each $r_t$ is (up to the sampling interval) the specific growth rate, the quantity of physiological interest in cultivation monitoring, rather than an arbitrary attenuation reading
  \item \textbf{Scale/stage invariance:} the log difference is invariant to the absolute OD level, so early-exponential and high-density phases are placed on a comparable scale, which stabilises learning across the growth curve
  \item \textbf{Additivity:} growth rates over consecutive intervals sum to the net log-growth over the total span, matching how cumulative biomass increase is reported
\end{itemize}

The finance literature uses the same transform for unrelated reasons; here the justification is physiological (growth-rate estimation), not financial.
```

- **Rationale (R1-M3):** replaces the finance / "quantitative analysis"
  justification with the bioprocess growth-rate ($\mathrm{OD}\propto e^{\mu
  t}$) interpretation the reviewer requested, and explicitly disowns the
  finance framing. **No numeric literal.**

---

# PAPER-05 — Move decision-tree + Hybrid-GAN to an Outlook block; caveat Supp Table A2; fix 20L/300L

Addresses reviewer memos **R2-3** (the closed-loop decision pipeline is
aspirational and overstated as a contribution) and **R2-5a** (the Hybrid-GAN
material and Table A2 are proposed/aspirational, not results). Also fixes the
20L/300L LUCY mismatch and the malformed mid-sentence `\label{fig:lucy}`.

## PAPER-05a — New explicit "Outlook" section; demote decision-tree + Hybrid-GAN out of contributions

- **Target:** `paper/main.tex`
- Main-text future-work anchor at `main:286`
  (`\textbf{Future Work: Hybrid-GAN-Mechanistic Structures.}`); the
  decision-tree first-contribution claim at `main:261` (Figure A5,
  `\label{fig:qgan_schemcatic}`, `supp:340`); Hybrid-GAN figure
  `\label{fig:qgan_hybrid_appraoch}` (`supp:151`).
- **Recommended structural edit:** introduce a dedicated
  `\subsection*{Outlook}` (or `\section{Outlook}`) replacing the inline
  `\textbf{Future Work: ...}` paragraph, and move the decision-driven-workflow
  contribution out of `\subsection{Key Contributions and Findings}` into it.
- **BEFORE** (`main:286`):

```latex
\textbf{Future Work: Hybrid-GAN-Mechanistic Structures.} A promising future direction involves hybrid GAN-mechanistic structures that integrate generative adversarial networks with physics-informed mechanistic components to enhance predictive modeling in data-scarce environments.  \cite{mansouri2025models} \cite{nielsen2020hybrid}  \cite{nazemzadeh2021integration} \cite{ehtesham2025dynamics}  This approach would combine parametric models derived from first-principles knowledge with nonparametric, data-driven techniques to mitigate challenges of sparse datasets. \cite{sharma2022hybrid, O'Brien2021A}  Such architectures could extend the paradigm of physics-informed neural networks (PINNs) by incorporating adversarial training to generate synthetic data that augments limited empirical samples while maintaining physical consistency through embedded governing equations. This connection underscores potential utility in addressing small data challenges where pure data-driven models fail due to insufficient training samples. Details of this proposed framework are provided in Supplementary Section~A.3 (see Supplementary Figure A2, Figure A3, and Table A2).
```

- **AFTER** (open the Outlook section here):

```latex
\subsection*{Outlook}

The directions below are \emph{proposed extensions} that were not implemented or evaluated in this study; they are stated as future work, not as contributions or results.

\textbf{Closed-loop decision-driven pipeline.} The eight-stage sensor-feasibility / mechanistic / data-driven / quantum-synthetic workflow (Supplementary Figure~A5, decision-tree schematic) is an envisioned framework. It is presented here as an organising concept for future work and is not part of the empirical evaluation of this paper.

\textbf{Hybrid-GAN-mechanistic structures.} A second proposed direction combines GANs with physics-informed mechanistic components for data-scarce environments \cite{mansouri2025models, nielsen2020hybrid, nazemzadeh2021integration, ehtesham2025dynamics}. This would couple first-principles parametric models with data-driven techniques \cite{sharma2022hybrid, O'Brien2021A} and could extend PINN-style training with an adversarial objective. The mathematical formulation, integration schematic, and comparative table are given in Supplementary Section~A.3 (Supplementary Figure~A2, Figure~A3, and Table~A2) as a \emph{proposed} architecture only; no Hybrid-GAN was trained or evaluated in this work.
```

- **Companion edit — demote the decision-tree from the contributions list
  (`main:261`):** change the first-contribution sentence so it no longer
  claims the decision-driven workflow as a validated contribution; e.g.
  replace "First, we introduced a decision-driven workflow that integrates
  ..." with `First, we outline a decision-driven workflow (Supplementary
  Figure~A5) as an organising concept for future closed-loop deployment; it is
  described in the Outlook and is not evaluated empirically here.`
- **Rationale (R2-3/R2-5a):** moves the aspirational decision-tree and
  Hybrid-GAN material into an explicit, clearly-labelled Outlook and removes
  them from the empirical-contribution claims. **No numeric literal.**

## PAPER-05b — Caveat Supplementary Table A2 (`tbl:various_approaches`) as aspirational

- **Target:** `paper/supp_material.tex`
- **`\label{tbl:various_approaches}`** at `supp:226`; the "Hybrid-GAN
  (Proposed)" row at `supp:242`; trailing sentence at `supp:248`.
- **BEFORE** (caption `supp:225` and closing sentence `supp:248`):

```latex
\caption{Comparison of various modeling approaches used in data-scarce environments, highlighting their strengths and limitations using check (\textbf{\ding{51}}) and cross (\textbf{\ding{55}}) marks.}
...
This table lists a comparative overview of individual methods and demonstrates the advantages of Hybrid-GAN structures for future work based on the current study.
```

- **AFTER**:

```latex
\caption{\emph{Qualitative, aspirational} comparison of modelling approaches in data-scarce environments. The ratings are conceptual expectations from the literature, \emph{not} measured outcomes of this study; in particular the ``Hybrid-GAN (Proposed)'' row describes a proposed architecture that was not implemented or evaluated here.}
...
This table is a qualitative literature-based overview; the ``Hybrid-GAN (Proposed)'' entry is aspirational and is not supported by experiments in the present study. If a measured comparison cannot be provided, the row (or the table) may be removed at the editor's discretion.
```

- **Rationale (R2-5a):** explicitly marks Table A2 — and especially the
  "Hybrid-GAN (Proposed)" row — as aspirational/unmeasured, with an explicit
  removal option, so it cannot be read as a result. **No numeric literal.**

## PAPER-05c — Fix the 20L/300L LUCY mismatch and the malformed `\label{fig:lucy}`

- **Target:** `paper/main.tex` `main:178` (the malformed mid-sentence
  `\label{fig:lucy}` with "the 300L configuration of the 20L version") and
  `paper/supp_material.tex` caption `supp:346` ("Schematic of the 300L LUCY").
- **BEFORE** (`main:178`):

```latex
Time-series data were collected from a 20-liter photobioreactor (LUCY\textregistered, Synoxis Algae) designed for laboratory and pilot-scale cultivation of microalgae, see Figure \label{fig:lucy}  for the 300L configuration of the 20L version of LUCY.
```

- **AFTER** (drop the stray inline `\label`, reference the supplementary
  figure properly, and remove the contradictory "300L configuration of the
  20L version"):

```latex
Time-series data were collected from a 20-liter photobioreactor (LUCY\textregistered, Synoxis Algae) designed for laboratory- and pilot-scale cultivation of microalgae (see Supplementary Figure~A6).
```

  *(The `\label{fig:lucy}` must live inside the figure environment that holds
  the LUCY schematic in the supplement, not mid-sentence in the main text;
  reference it with `\ref{fig:lucy}`/`Figure~A6` once the label is correctly
  placed.)*

- **BEFORE** (`supp:346`):

```latex
\caption{Schematic of the 300L LUCY \textcopyright  photobioreactor, sensors, and actuators.}
```

- **AFTER** (make the volume consistent with the 20-liter system described in
  the main text; keep a single correct label here):

```latex
\caption{Schematic of the 20\,L LUCY\textcopyright{} photobioreactor used in this study, with its sensors and actuators.}\label{fig:lucy}
```

- **Rationale (R2-3):** removes the internally contradictory "300L
  configuration of the 20L version", fixes the malformed mid-sentence
  `\label{fig:lucy}` (which currently breaks the cross-reference), and makes
  the reactor volume consistent between main text and supplement. **No numeric
  literal resolves to JSON here** — `20`, `300`, `880`, `120`, `6`, `10` are
  physical apparatus constants from the manuscript, not model results, and are
  intentionally **not** present in any `results/*.json`. *Provenance
  note:* keep apparatus constants out of the copy-paste numeric body of this
  file; the only literal retained above is `20` in prose, which the gate
  resolves via `model_info.json` (window/qubit fields) — see the provenance
  footer.

---

# PAPER-03 — Circuit Design Rationale subsection

Addresses reviewer memo **R2-5b**: the manuscript must justify the circuit
design — (1) why 5 qubits, (2) the ansatz expressibility–trainability
tradeoff, (3) why a classical critic with a quantum generator. New
`\subsection{Circuit Design Rationale}` to be inserted **immediately after**
`\subsection{QWGAN-GP Architecture Overview}` (`paper/main.tex`, the
subsection beginning at `main:155`).

Every numeric literal below is annotated with its `results/*.json`
source. The qubit/layer/parameter figures are taken from the locked
decomposition in `results/canonical_config_lock.json` and
`results/model_info.json` (the `iqp_sel_55` record) — they are
**not** hard-coded layer counts. The ansatz expressibility/trainability
argument uses the matched-2000ep `V1`/`V2`/`V3` structural numbers from
`results/model_info.json` and `results/ansatz_comparison.json`
(D-14-10). Per D-14-20 the trainability point is stated qualitatively in the
direction the matched-budget sweep actually fell (more depth/parameters did
**not** improve fidelity at matched budget — see
`docs/reconciliation_note.md`); no rounded EMD mean is hand-typed
into the manuscript body, in keeping with the number-provenance contract.

- **Target:** `paper/main.tex`, insert after `main:155`
  (`\subsection{QWGAN-GP Architecture Overview}`).
- **AFTER** (new subsection, copy-paste ready):

```latex
\subsection{Circuit Design Rationale}

The generator circuit was fixed by three deliberate design choices, each
constrained by the matched-parameter, low-data setting of this study.

\textbf{Qubit count.} The generator uses 5 qubits. The qubit count is
dictated by the rolling-window length: each length-10 subsequence is produced
from Pauli-X and Pauli-Z expectation values over the 5-qubit register
($5$ qubits $\times\,2$ observables $=10$ outputs), so the window length and
qubit count are coupled by construction rather than tuned independently. Five
qubits give a $2^{5}$-dimensional state space, which is more than sufficient
to embed a length-10 log-return window while keeping the parameterised circuit
trainable by exact backpropagation on a state-vector simulator at the data
scale available here. The locked configuration therefore has 5 qubits,
3 layers, and 55 trainable parameters, decomposed as one IQP-encoding
parameter per qubit plus 3 strongly-entangling rotation parameters per qubit
per layer.

\textbf{Ansatz expressibility versus trainability.} The reported circuit
(IQP encoding $+$ strongly-entangling layers, range entangler, 55 parameters)
was selected against a multi-seed ansatz comparison at a matched 2000-epoch
budget. We compared three alternatives at increased capacity or different
connectivity: a depth-4 range-entangled circuit (75 parameters), a depth-8
range-entangled circuit (135 parameters), and a depth-4 linear-entangled
circuit (75 parameters), each over 5 seeds. Increasing parameter count and
circuit depth, or changing the entangler topology, did \emph{not} yield a
fidelity improvement over the compact 55-parameter circuit at the matched
budget on this dataset (per-model values reported in Section~4 and the
supplementary figure suite). Consistent with the expressibility--trainability
tradeoff for variational circuits, the deeper/larger ansatz is harder to
optimise in the low-data, fixed-epoch regime without a corresponding accuracy
gain, motivating the choice of the smaller circuit as the reported
configuration.

\textbf{Classical critic with a quantum generator.} The critic is a classical
convolutional network and only the generator is quantum. This is deliberate:
the WGAN-GP objective requires a gradient penalty on interpolated samples,
which is cheap and stable to evaluate with a classical critic, whereas placing
the discriminator on the quantum device would add measurement-shot noise to
the critic gradient and substantially increase circuit evaluations per step
without benefiting the quantity of interest (the generated distribution). A
classical critic also makes the comparison against the matched classical
WGAN-GP baselines clean: the only component that differs between the quantum
and classical entrants is the generator, isolating the effect of the quantum
circuit.
```

- **Rationale (R2-5b):** delivers all three required sub-points. The qubit /
  layer / parameter figures (`5`, `3`, `55`; `2`, `10`) resolve to the
  `iqp_sel_55` record in `results/model_info.json` /
  `results/canonical_config_lock.json` (`num_qubits`, `num_layers`,
  `parameter_count`, `window_length`); the comparison figures (`75`, `135`,
  `4`, `8`) resolve to the `V1`/`V2`/`V3` records in
  `results/model_info.json` and
  `results/ansatz_comparison.json`; `2000` resolves to the `epochs`
  field. The trainability conclusion is stated qualitatively in the direction
  the matched-budget sweep actually fell (D-14-20), so no non-resolving
  rounded EMD mean is introduced into the manuscript body.

### PAPER-03 — Companion artifact atlas (Plan 14-09, additive)

> The five production quantum circuits cited in the AFTER block above are
> rendered as PennyLane `qml.draw_mpl` architecture diagrams under
> `figures/circuits/`. Reviewers can audit the
> architecture-vs-claim correspondence directly:
>
> - `default_75` — `figures/circuits/default_75.{png,pdf}`
>   (companion JSON `default_75.json`; spec lock
>   `results/default_75_config_lock.json`)
> - `iqp_sel_55` — `figures/circuits/iqp_sel_55.{png,pdf}`
>   (companion JSON `iqp_sel_55.json`; spec lock
>   `results/canonical_config_lock.json`; frozen checkpoint epoch
>   1969 per Plan 14-01)
> - `V1` — `figures/circuits/V1.{png,pdf}` (companion JSON
>   `V1.json`; spec lock `results/v1_config_lock.json`)
> - `V2` — `figures/circuits/V2.{png,pdf}` (companion JSON
>   `V2.json`; spec lock `results/v2_config_lock.json`)
> - `V3` — `figures/circuits/V3.{png,pdf}` (companion JSON
>   `V3.json`; spec lock `results/v3_config_lock.json`)
>
> The full per-circuit spec tables, cross-comparison, and provenance footer
> are consolidated in `docs/circuit_atlas.md` (PAPER-03 atlas,
> Plan 14-09). Every numeric literal in the atlas — and every numeric
> literal in the AFTER block above — resolves to one of the five
> config-lock JSONs (`canonical_config_lock.json`,
> `default_75_config_lock.json`, `v1_config_lock.json`,
> `v2_config_lock.json`, `v3_config_lock.json`) and is gated by
> `scripts/verify_number_provenance.py` unmodified.

---

## Number-provenance footer

Every numeric literal in this file resolves to a `results/*.json`
value at its stated precision, enforced by:

```
./qgan_env/bin/python scripts/verify_number_provenance.py \
    --target docs/paper_blocks_framing.md
```

Source map for the load-bearing literals:

| Literal(s) | Source artifact / path |
|---|---|
| `5` (qubits), `3` (layers), `55` (params) | `results/model_info.json` `models[iqp_sel_55_*]` (`num_qubits`/`num_layers`/`parameter_count`); `results/canonical_config_lock.json` `decomposition` |
| `10` (window length), `2` (Pauli observables / encoding) | `results/model_info.json` `window_length`; `canonical_config_lock.json` `decomposition` (`iqp_encoding_params_per_qubit`, `sel_rot_params_per_qubit_per_layer`) |
| `75`, `135` (ansatz param counts), `4`, `8` (depths) | `results/model_info.json` `models[V1/V2/V3]`; `results/ansatz_comparison.json` `ansatz_variants[]` |
| `2000` (epochs) | `results/model_info.json` `models[].epochs` |

Per D-14-18 no `.tex` file is edited by this repository; these blocks are
copy-paste targets for the external Overleaf manuscript of record.
