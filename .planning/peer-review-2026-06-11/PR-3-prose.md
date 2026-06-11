# PR-3 — Prose Quality Audit Report

## Verdict
**CONCERNS** — manuscript is technically sound but front-loads numeric findings into Abstract, Plain Language Summary, §1.4 Contributions, and §5 Concluding Remarks in violation of AIChE-Journal house style. The §1.4 Contributions bullets are non-parallel and read as a reviewer-pitch ledger, not a contributions list. A one-sentence bioprocess-domain justification of log-returns is missing from §3 (it exists in supp §A.7 lines 657–665 but is not surfaced in the main text). Tone is mostly well-calibrated but contains residual defensive holdover from the v1.2.4 bifurcated-finding era that under-claims the post-14-21 result.

---

## C1 — Numeric literals to trim

### Abstract (lines 47–50)

The Abstract is a single 247-word block carrying SEVEN numeric finding-claims that all belong in §4 / Table 2.

| Line | Verbatim fragment | Verdict | Suggested rewrite |
|---|---|---|---|
| 49 | "Under a matched parameter budget and 2000 training epochs (n=5 seeds)" | **KEEP** (scope) | unchanged — "2000 epochs (n=5 seeds)" is load-bearing protocol scope |
| 49 | "quantum WGAN-GPs with 55--135 parameters dominate parameter-matched classical adversarial WGAN baselines (73--78 parameters)" | **TRIM ranges** | "the quantum WGAN-GP cluster dominates the parameter-matched classical adversarial WGAN cluster" (parameter ranges belong in Table 2 / Section 3) |
| 49 | "on three of the four headline metrics at the matched 2000-epoch budget" | **REWRITE** — under-claims | "on all four matched-budget fidelity metrics" (post-14-21 the finding is 4/4, not 3/4 — see C3 below) |
| 49 | "log-return temporal alignment (LR-DTW 6.09--9.48 vs WGAN 18.23--69.02)" | **TRIM** | "log-return temporal alignment (LR-DTW)" — numeric ranges to §4.1 / Table 2 |
| 49 | "log-return single-step marginal (LR-EMD 0.0040--0.0050 vs WGAN 0.0244--0.1286)" | **TRIM** | "log-return single-step marginal (LR-EMD)" |
| 49 | "optical-density temporal alignment (OD-DTW 0.33--0.41 vs WGAN 0.60--6.99, Welch $p$ as low as $0.002$)" | **TRIM** numbers; **KEEP** Welch significance qualifier in words | "optical-density temporal alignment (OD-DTW, Welch-significant)" |
| 49 | "On the optical-density single-step marginal (OD-EMD 0.028--0.031 vs WGAN 0.077--0.799) the quantum cluster also leads the WGAN cluster significantly (Welch $p = 0.019$)" | **TRIM** ranges; **KEEP** "significantly" | "and on optical-density single-step marginal (OD-EMD, Welch-significant)" |
| 49 | "The mean lag-1 ACF of the quantum cluster ($-0.0997$ to $-0.0895$) is closer to the real reference ($-0.064$) than any classical-baseline mean" | **TRIM numbers** | "The quantum-cluster mean lag-1 ACF is closer to the real reference than any classical-baseline mean" |

**Proposed Abstract rewrite (single paragraph, ~130 words):**

> Data scarcity in biomanufacturing constrains model development for process monitoring and optimization. We evaluate a Quantum Wasserstein Generative Adversarial Network with Gradient Penalty (QWGAN-GP), whose generator is a Parameterized Quantum Circuit (PQC), against parameter-matched classical adversarial baselines on a single laboratory-scale photobioreactor cultivation. Under a matched parameter budget, 2000 training epochs, and n=5 seeds, the quantum WGAN-GP cluster dominates the parameter-matched classical adversarial WGAN cluster on all four matched-budget fidelity metrics — log-return and optical-density temporal alignment (DTW), and log-return and optical-density single-step marginals (EMD) — with the optical-density EMD and DTW separations Welch-significant. The quantum-cluster mean lag-1 autocorrelation is also closer to the real reference than any classical-baseline mean. We report between-cluster gaps with explicit n=5 power caveats; per-model dominance is not extended to per-seed claims on every axis. Specific values appear in Section 4 and Table 2.

---

### Plain Language Summary (line 58–59)

| Line | Verbatim sentence | Trim verdict | Suggested rewrite |
|---|---|---|---|
| 59 | "On a small bioprocess dataset, a quantum generator reproduced both single-step distributions and short-term temporal structure more faithfully than parameter-matched classical WGAN baselines on every matched-budget metric we measured, with statistical-power caveats noted from the five-seed protocol." | **KEEP** — no numeric finding-literals; correctly says "every matched-budget metric" | unchanged. Char count: 308 (over the stated 242 / 250 limit — needs trimming for length, not for numeric leakage) |

**Char-count rewrite to fit 242 limit (240 chars):**

> On a small bioprocess dataset, a quantum generator reproduced both single-step distributions and short-term temporal structure more faithfully than parameter-matched classical WGAN baselines on every matched-budget metric we measured.

(Drops the trailing "with statistical-power caveats noted from the five-seed protocol" — caveat already lives in §4 + Limitations and Plain Language Summary should carry the headline finding, not the caveats.)

---

### §1.4 Principal Contributions (lines 95–109)

This section was flagged correctly by the coauthor — it reads as a 6-paragraph reviewer-rebuttal block, not as a contributions list. The third bullet ("Headline Empirical Finding", line 103) is 11 lines of dense numerics that duplicate the Abstract.

| Line | Verbatim fragment | Verdict | Suggested rewrite |
|---|---|---|---|
| 103 | "(55--135 parameters) dominate parameter-matched classical WGAN adversarial baselines (73--78 parameters)" | **TRIM** | "the quantum cluster dominates the parameter-matched classical WGAN cluster" |
| 103 | "log-return temporal alignment (LR-DTW 6.09--9.48 vs WGAN 18.23--69.02)" | **TRIM** | "log-return temporal alignment (LR-DTW, per-seed dominance)" |
| 103 | "log-return single-step marginal (LR-EMD 0.0040--0.0050 vs WGAN 0.0244--0.1286)" | **TRIM** | "log-return single-step marginal (LR-EMD, ~15× cluster-mean separation)" — qualitative scale only |
| 103 | "optical-density temporal alignment (OD-DTW 0.33--0.41 vs WGAN 0.60--6.99, Welch $p$ as low as $0.002$)" | **TRIM** numbers; **KEEP** Welch in words | "optical-density temporal alignment (OD-DTW, Welch-significant)" |
| 103 | "optical-density single-step marginal (OD-EMD 0.028--0.031 vs WGAN 0.077--0.799, Welch $p = 0.019$)" | **TRIM** numbers | "optical-density single-step marginal (OD-EMD, Welch-significant)" |
| 103 | "quantum cluster mean $-0.0997$ to $-0.0895$ vs real $-0.064$" | **TRIM** | "the quantum-cluster mean is closer to the real lag-1 ACF reference than any classical-baseline mean" |
| 105 | "the quantum cluster's matched-budget mean is closer to the real reference than every parameter-matched classical WGAN cluster mean on each of the four matched-budget metrics, with statistical significance reaching Welch $p = 0.019$ on OD-EMD and $p \approx 0.002$ on OD-DTW" | **TRIM** $p$-values; **KEEP** "Welch-significant" qualifier | "...on each of the four matched-budget metrics, with Welch significance on both optical-density axes" |
| 105 | "the Welch $t$-test affords approximately 15\% statistical power to detect a moderate effect size of $d = 0.5$" | **MOVE** to Limitations (§4.3) | This is a power caveat, not a contribution — it belongs in §4.3 Limitations, not §1.4 Contributions |

**Proposed §1.4 restructure** — four parallel bullets, each a single clean noun-phrase contribution:

1. **A matched-parameter, matched-epoch, multi-seed comparison protocol** for PQC vs. classical generators on data-scarce bioprocess time series.
2. **A hybrid QWGAN-GP architecture** with a Parameterized Quantum Circuit generator and a classical critic, evaluated on a laboratory-scale photobioreactor cultivation.
3. **A matched-budget empirical finding**: the quantum cluster dominates the parameter-matched classical WGAN cluster on all four matched-budget fidelity metrics (log-return and OD DTW; log-return and OD EMD), with Welch significance on both OD axes; specific values in §4.1 / Table 2.
4. **Reproducible open-science artifact** (code + dataset + matched-budget reproduction script) for independent verification.

Drop the standalone "Empirical Evaluation on a Photobioreactor Campaign" bullet (line 105) — it duplicates bullet 3. The decision-tree-triage paragraph (line 109) correctly disclaims itself as non-contribution and should be kept as the trailing clarifying sentence.

---

### §5 Concluding Remarks (lines 873–912)

| Line | Verbatim fragment | Verdict | Suggested rewrite |
|---|---|---|---|
| 881–883 | "log-return temporal alignment (LR-DTW 6.09--9.48 vs 18.23--69.02), log-return single-step marginal (LR-EMD 0.0040--0.0050 vs 0.0244--0.1286, Welch $p \approx 0.0002$)" | **TRIM ranges**; **KEEP** Welch in words | "log-return temporal alignment (LR-DTW), log-return single-step marginal (LR-EMD, Welch-significant)" |
| 884–887 | "optical-density temporal alignment (OD-DTW 0.33--0.41 vs 0.60--6.99, Welch $p \approx 0.002$...), and optical-density single-step marginal (OD-EMD 0.0279--0.0308 vs 0.0769--0.7989, Welch $p = 0.019$)" | **TRIM** | "optical-density temporal alignment (OD-DTW, Welch-significant), and optical-density single-step marginal (OD-EMD, Welch-significant)" |
| 887–889 | "with lag-1 autocorrelation closeness to the real series ($-0.064$) corroborating the temporal structure result" | **TRIM** numeric | "with lag-1 autocorrelation closeness corroborating the temporal-structure result" |
| 897–898 | "a single 778-point optical-density cultivation campaign, 384 length-10 rolling windows, five qubits, five seeds" | **KEEP** | scope-defining; these are envelope numbers, not finding numbers |

**Proposed §5 opening rewrite (lines 875–895):**

> Returning to the falsifiable question posed in Section 1.3 — whether a PQC generator can match or exceed a parameter-matched classical generator on a low-data bioprocess task — the matched-budget evidence on this single dataset answers in the affirmative against the classical WGAN adversarial cluster. The quantum generators exceed the parameter-matched classical WGAN adversarial baselines on every matched-budget metric measured: log-return temporal alignment (LR-DTW, per-seed dominance), log-return single-step marginal (LR-EMD, Welch-significant), optical-density temporal alignment (OD-DTW, Welch-significant), and optical-density single-step marginal (OD-EMD, Welch-significant); lag-1 autocorrelation closeness to the real reference corroborates the temporal-structure result at the cluster-mean level (per-seed overlap noted). The AR(2) reference leads on LR-EMD; the VAE LR-DTW reflects a degenerate generation regime and is excluded from the dominance claim. We treat this as a positive answer scoped to the laboratory-scale, single-variable photobioreactor cultivation evaluated and to the n=5 power budget actually exercised.

---

## C2 — Missing log-returns plain-language justification

**Currently present in main text:** line 305, "First differencing into log-returns yields 777 log-return observations ($r_t = \ln \mathrm{OD}_t - \ln \mathrm{OD}_{t-1}$), which are standardized to zero mean and unit variance, then linearly rescaled to $[-1, 1]$..." — purely mechanical, no domain-meaning rationale.

**Currently present in supp:** §A.7 lines 657–665 has the full bioprocess justification ("for an exponentially growing culture the optical density follows $\mathrm{OD}_t \approx \mathrm{OD}_0\,e^{\mu t}$, so the log difference between successive samples is a direct discrete estimate of the instantaneous specific growth rate", and three bulleted benefits: growth-rate interpretability, additivity, scale-stability). This is excellent prose — the main text just needs the one-sentence headline of it.

**Currently absent from main text:** the bioprocess-domain one-liner connecting log-returns to specific growth rate μ.

**Proposed sentence (verbatim, 1 sentence):**

> Log-returns are the natural rate variable for this signal: for an exponentially growing culture $\mathrm{OD}_t \approx \mathrm{OD}_0\,e^{\mu t}$, so $r_t = \ln\mathrm{OD}_t - \ln\mathrm{OD}_{t-1}$ is (up to the 10-minute sampling interval) the local specific growth rate $\mu_t$, the quantity of physiological interest in cultivation monitoring (supp \S A.7).

**Insertion point:** §3 Methods, line 305, immediately after `($r_t = \ln \mathrm{OD}_t - \ln \mathrm{OD}_{t-1}$),` and before `which are standardized to zero mean...`. This places the domain-meaning one-liner adjacent to the equation that defines it, without disturbing the rest of the preprocessing description.

**Augmentation of existing prose** — the existing line 305 should become:

> First differencing into log-returns yields 777 log-return observations ($r_t = \ln \mathrm{OD}_t - \ln \mathrm{OD}_{t-1}$); log-returns are the natural rate variable for this signal because for an exponentially growing culture $\mathrm{OD}_t \approx \mathrm{OD}_0\,e^{\mu t}$, so $r_t$ is (up to the 10-minute sampling interval) the local specific growth rate $\mu_t$, the quantity of physiological interest in cultivation monitoring (supp \S A.7). These are then standardized to zero mean and unit variance, then linearly rescaled to $[-1, 1]$...

Also recommended: add a single forward-pointer in §1 Introduction around the QIV / specific-growth-rate paragraph (line 74), e.g. at the end of line 74 append: "Section 3 returns to specific growth rate as the natural rate variable that the generator is trained on." — one sentence, ties the §1 bioprocess context to the §3 modeling choice.

---

## C3 — Tone calibration

### Over-claim instances
None severe. The manuscript is well-disciplined on scope qualifiers ("on this single dataset", "n=5 power caveats", "single-variable", "laboratory-scale"). One soft over-claim:

- **Line 771:** "may be relevant to downstream applications such as soft-sensor training and process-monitoring augmentation" — "may be relevant" is acceptably hedged. **No change.**
- **Line 49 (Abstract):** "potential relevance to soft-sensor and control applications" — same hedge, acceptable.

### Under-claim instances (defensive holdover from v1.2.4 bifurcated-finding era)

- **Line 49 (Abstract):** "dominate parameter-matched classical adversarial WGAN baselines (73--78 parameters) **on three of the four headline metrics** at the matched 2000-epoch budget" — **THIS IS A STALE HOLDOVER FROM v1.2.4.** Post-14-21 the result is 4/4 (LR-DTW, LR-EMD, OD-DTW, OD-EMD all show quantum-cluster dominance; the four-of-four is restated correctly in §1.4 line 103, §4.2 line 726, §5 lines 879–887, and §4.3 line 769). The Abstract still says "three of the four" then describes the fourth (OD-EMD) in the next sentence with "the quantum cluster also leads the WGAN cluster significantly" — this is internally contradictory and reads as nervous hedging. **Fix:** rewrite line 49 to say "on all four matched-budget fidelity metrics" up front, then carry the metric list. The corrected Abstract draft in C1 already does this.

- **Line 728 (§4.2):** "We present the QWGAN-GP as outperforming size-matched classical WGAN adversarial baselines on every matched-budget metric on this dataset, while restricting positive claims to between-cluster means under the n=5 power budget and to the laboratory-scale single-variable setting actually evaluated." — the "while restricting positive claims" clause is a 30-word defensive parenthetical that buries the headline. **Fix:** split into two sentences. "We present the QWGAN-GP as outperforming size-matched classical WGAN adversarial baselines on every matched-budget metric on this dataset. Positive claims are restricted to between-cluster means under the n=5 power budget and to the laboratory-scale single-variable setting actually evaluated."

- **Line 49 (Abstract):** "We present a proof-of-concept for quantum-enhanced synthetic time-series generation" — "proof-of-concept" is weaker than the 4/4 + Welch-significance result warrants. **Suggested alternative:** "We present matched-budget evidence that a PQC generator can match or exceed parameter-matched classical adversarial baselines on data-scarce bioprocess time-series synthesis, with potential relevance to soft-sensor and control applications."

---

## C4 — Top 5 worst sentences (clarity)

1. **Line 49 (Abstract, single sentence covers 4 metrics + statistical caveats + power scope):** the Abstract is one block of 247 words with three sentences of 60+ words each. **Rewrite:** break into the proposed 7-sentence version in C1. Currently the reader has to parse "dominate ... on three of the four ... On the optical-density single-step marginal (OD-EMD ...) the quantum cluster also leads the WGAN cluster significantly" as a four-way claim hiding inside a three-way claim — actively confusing.

2. **Line 105 (§1.4 contributions, second-to-last sentence):** "With only 5 seeds per model the Welch $t$-test affords approximately 15\% statistical power to detect a moderate effect size of $d = 0.5$, so per-model means rather than per-seed cells anchor the cluster-level claims." → **Rewrite:** "At n=5 seeds, the Welch $t$-test is underpowered for per-seed claims, so cluster-level claims are anchored on per-model means (full power analysis in §4.3)." The "15% power for $d=0.5$" detail belongs in §4.3 Limitations, not Contributions.

3. **Line 92 (§1.3, falsifiable question, run-on):** "This motivates the falsifiable question that frames the present study: \emph{can a PQC generator, operating in an exponentially large Hilbert space with $\mathcal{O}(\mathrm{poly}(n))$ parameters, match or exceed a classical generator of equivalent parameter count on a low-data bioprocess task?}" → the embedded "operating in an exponentially large Hilbert space with $\mathcal{O}(\mathrm{poly}(n))$ parameters" is jargon-dense and not pulling its weight. **Rewrite:** "This motivates the falsifiable question that frames the present study: \emph{can a PQC generator match or exceed a parameter-matched classical generator on a low-data bioprocess task?} The PQC operates in an exponentially large Hilbert space with $\mathcal{O}(\mathrm{poly}(n))$ parameters; whether that translates into a practical advantage is what we test."

4. **Line 88 (§1.2, ambiguous antecedent + buried message):** "These limitations motivate exploring alternative computational paradigms, including quantum approaches, which \emph{may} offer advantages for certain structured learning tasks in chemical and biochemical engineering" → "which may" antecedent is ambiguous (refers to "quantum approaches" or "alternative computational paradigms"?). **Rewrite:** "These limitations motivate exploring alternative computational paradigms. Quantum approaches \emph{may} offer advantages for certain structured learning tasks in chemical and biochemical engineering [cite]."

5. **Line 769 (§4.3, 5-clause comma-glued list):** "The quantum WGAN-GP cluster dominates the parameter-matched classical adversarial WGAN cluster on all four matched-budget metrics evaluated on the photobioreactor dataset: log-return temporal alignment (LR-DTW, per-seed dominance), log-return single-step marginal (LR-EMD, $\sim 15\times$ quantum advantage on per-model means), optical-density temporal alignment (OD-DTW, Welch cluster-floor $p \approx 0.002$), and optical-density single-step marginal (OD-EMD, Welch cluster-floor $p = 0.019$); the lag-1 autocorrelation of the real series corroborates the cluster-mean separation (Section~4.1)." — This is one sentence of 95 words with two semicolon-spliced subordinate clauses. **Rewrite:** "The quantum WGAN-GP cluster dominates the parameter-matched classical adversarial WGAN cluster on all four matched-budget metrics: LR-DTW (per-seed dominance), LR-EMD (cluster-mean separation), OD-DTW (Welch-significant), and OD-EMD (Welch-significant). Lag-1 autocorrelation closeness to the real series corroborates the cluster-mean temporal-structure result (§4.1)."

---

## C5 — Plain Language Summary verdict

- **Current text (line 59):** "On a small bioprocess dataset, a quantum generator reproduced both single-step distributions and short-term temporal structure more faithfully than parameter-matched classical WGAN baselines on every matched-budget metric we measured, with statistical-power caveats noted from the five-seed protocol."
- **Char count:** 308 chars (over the 242 target stated in the prompt; the file comment at line 57 says "max 250 characters" — either way, over).
- **Carries headline finding?** Yes — "every matched-budget metric we measured" correctly states the post-14-21 4/4 finding.
- **Plain language?** Mostly. "Single-step distributions" and "parameter-matched" are still jargon. "Matched-budget metric" is internal terminology.
- **Verdict:** **REWRITE for length + plainer language. 238 chars:**

> On a small bioprocess dataset, a quantum generator reproduced both the per-step value distribution and the short-term temporal pattern more faithfully than equally-sized classical WGAN baselines on every fidelity metric we measured.

(Drops the power caveat — caveats live in the body; the Plain Language Summary's job is the headline finding in plain words.)

---

## C6 — Contributions parallelism (lines 95–109)

Current §1.4 has 5 bullets (lines 99, 101, 103, 105, 107) plus a trailing disclaimer paragraph (line 109).

| # | Current head | Form | Parallel? | Single clean idea? |
|---|---|---|---|---|
| 1 (line 99) | "Matched-Parameter Comparison Protocol:" | Title-Case noun phrase + colon + 1-sentence definition | ✓ form | ✓ |
| 2 (line 101) | "QWGAN-GP for Process Data Synthesis:" | Title-Case noun phrase + colon + 1 sentence | ✓ form | ✓ |
| 3 (line 103) | "Headline Empirical Finding:" | Title-Case noun phrase + colon + **11 lines of dense numerics + power caveats** | ✗ — 10× longer than others | ✗ — fuses finding + power caveat |
| 4 (line 105) | "Empirical Evaluation on a Photobioreactor Campaign:" | Title-Case noun phrase + colon + 4 sentences | partial — overlaps bullet 3 | ✗ — duplicates bullet 3's content |
| 5 (line 107) | "Open Science and Reproducibility:" | Title-Case noun phrase + colon + 1 sentence | ✓ form | ✓ |

**Issues:**
- Bullet 3 and bullet 4 are the **same contribution** (the matched-budget finding) said twice with overlapping numerics.
- Bullet 3 mixes the **finding** ("quantum dominates on 4/4 metrics") with the **power caveat** ("15% power at $d=0.5$") — different ideas, should be separated or the caveat moved to §4.3.
- Order: protocol → architecture → finding → evaluation → open science. Bullets 3+4 should collapse to one "finding" bullet; the order then becomes protocol → architecture → finding → reproducibility, with the strongest claim (the finding) in the middle. **This ordering is fine**; merging bullets 3+4 is the dominant fix.

**Recommended restructure:** four parallel bullets per the C1 proposed §1.4 restructure, each ~25–40 words, no embedded statistical-power discussion (move to §4.3).

---

## Total fix burden estimate

- **~28 numeric-literal removals** across Abstract (8), Plain Language Summary (length trim only, 0 numeric), §1.4 Contributions (7), §5 Concluding Remarks (5), plus the line-49 "three of the four" → "all four" correction. **Mechanical.**
- **1 sentence insertion** at §3 line 305 (the log-return / specific-growth-rate bioprocess one-liner). Optional companion forward-pointer at line 74.
- **5 sentence-level rewrites** per C4 (Abstract sentence-split, line 105 power-clause shorten, line 92 falsifiable-question split, line 88 antecedent fix, line 769 sentence-split).
- **1 §1.4 structural restructure** — collapse 5 bullets to 4 parallel ones, move power caveat to §4.3.
- **1 Plain Language Summary length trim** to fit 242 chars and plainer language.
- **1 tone correction**: change "three of the four" to "all four" in Abstract (line 49).

**Wall-clock estimate:** 60–90 minutes for one author working in the .tex source. Mechanical numeric trimming ~30 min; the §1.4 restructure and 5 sentence rewrites ~30 min; PLS rewrite + log-return insertion ~15 min; one read-through for consistency ~15 min. No re-running of figures or numerics required — all changes are prose-only and the underlying §4 / Table 2 numbers stay unchanged.

**Files touched:** `main (4) copy.tex` only (lines 47–50, 58–59, 95–109, 305, 728, 769, 875–895). Supplement is unchanged.
