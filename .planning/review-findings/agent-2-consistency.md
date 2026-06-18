# Agent 2 — Cross-Section Consistency Findings

**Audit target:** HEAD `50658a6` (v1.2.1)
**Files audited:** `main (4) copy.tex`, `paper/supp_material.tex`
**Scope:** Cross-section agreement on numbers, scope hedges, orderings, comparator-set scope, future-work tense.

## Summary

- BLOCK: 0 findings
- FLAG: 3 findings
- NIT: 2 findings

The literals (LR-DTW 0.94–1.12 vs 1.58–6.86, AR(2)=7.70, lag-1 ACF −0.064, OD-DTW ≈0.30, Welch p>0.36, |d|≤0.65) and the orderings agree across Abstract / §1.4 / Table 1 / Table 2 / §4.1 / §4.2 / §4.3 / §4.5 / §5 / supp §A.4 / per-seed dominance table / Welch pairwise tables. The decision-tree triage workflow is consistently flagged as future work in §1.4, §4.2, §4.5, and supp Figure A5 caption. The remaining drift is scope-of-comparator drift on the OD-marginal “indistinguishable” claim — same family of issue as the post-swarm M8 finding.

## BLOCK findings

(none)

## FLAG findings

### F-1: OD-marginal comparator-set scope drifts between front-of-paper and §4.2/§5/§1.4-bullet-4

The §4.2 and §5 bookends both state the OD-marginal claim against the **full** comparator set (adversarial + VAE + AR(2)) — this matches the supp Welch OD-EMD table (20 pairs, 4 quantum × 5 classical including VAE and AR(2)) and §4.1's prose ("20 quantum-vs-classical OD-EMD pairs ... including VAE and AR(2)"). The Abstract, Plain Language Summary, and the §1.4 "Bifurcated Empirical Finding" bullet (bullet 3) narrow that scope to "parameter-matched classical adversarial baselines" only. The §1.4 bullet 4 ("Empirical Evaluation") has the correct wider scope. So §1.4 is inconsistent with itself.

- **Sections involved:** Abstract (line 49), Plain Language Summary (line 59), §1.4 bullet 3 (line 103) — vs §1.4 bullet 4 (line 105), §4.2 (line 658), §5 (lines 777–781), §4.1 (line 561).
- **Abstract quote (line 49):** "On the optical-density marginal, no parametric-efficiency advantage is observed: Welch p > 0.36, max |Cohen's d| <= 0.65 …" — the only comparator scope active in the Abstract is the opening "parameter-matched classical adversarial baselines".
- **Plain Language Summary quote (line 59):** "while overall single-point distributions were statistically indistinguishable" — inherits "parameter-matched classical adversarial baselines" from the preceding clause; no wider comparator set named.
- **§1.4 bullet 3 quote (line 103):** "On the optical-density marginal, no advantage is observed." — no comparator scope at all.
- **§1.4 bullet 4 quote (line 105):** "the full set of parameter-matched classical comparators (adversarial baselines plus VAE and AR(2)) are not statistically significant …"
- **§4.2 quote (line 658):** "*match* the full set of parameter-matched classical comparators (adversarial baselines plus VAE and AR(2)) on the OD single-step marginal".
- **§5 quote (lines 777–781):** "they *match* the full set of parameter-matched classical comparators (adversarial baselines plus VAE and AR(2); statistically indistinguishable under the achievable n=5 power, Welch p > 0.36 and |Cohen's d| ≤ 0.65)".
- **Discrepancy:** Abstract / PLS / §1.4 bullet 3 imply the OD-marginal null is only against adversarial baselines; §4.2 / §5 / §1.4 bullet 4 / §4.1 statistical-test prose / supp Welch table all extend it to the full 5-comparator set (including VAE and AR(2)). This is the same M8-class scope-drift that surfaced in the post-swarm A5 audit.
- **Suggested fix:** Pull §4.2/§5's wording back into Abstract / PLS / §1.4 bullet 3:
  - Abstract: "On the optical-density marginal, no parametric-efficiency advantage is observed *against the full set of parameter-matched comparators (adversarial baselines, VAE, and AR(2))*: Welch p > 0.36 …" (room permitting given 150-word cap; a compact parenthetical "(adversarial + VAE + AR(2))" preserves the scope).
  - PLS: "*from the parameter-matched comparator set (adversarial, VAE, AR(2))*" or, given the 250-character PLS cap, change "adversarial baselines" → "comparator models" so the wider scope is not falsely narrowed.
  - §1.4 bullet 3 line 103: add a parenthetical "(against the full set of parameter-matched comparators; see bullet 4)" after "no advantage is observed".

### F-2: §1.4 bullet 3 omits the LR-EMD reversal that bullet 4, §4.1, §4.2, §4.3, §4.4, and §5 all carry

The "Bifurcated Empirical Finding" bullet (§1.4 bullet 3, line 103) is the §1.4 sentence a reader looking for the headline finding will pull. It reports the LR-DTW+lag-1 ACF positive direction and the OD-marginal null, but it does NOT mention the LR-EMD reversal direction (every classical adversarial baseline beats every quantum variant on LR-EMD; AR(2) leads). Every other front-matter framing of the bifurcated finding — Abstract notwithstanding (Abstract omits LR-EMD too), §4.1 (lines 414–417, 627–642), §4.2 (line 658 "*fall short* on the log-return single-step marginal (LR-EMD)"), §4.3 (line 673 "On the log-return single-step marginal axis (LR-EMD) the direction reverses"), §4.4 (lines 690–693), §5 (lines 781–783) — explicitly discloses the LR-EMD direction. The §1.4 contributions bullet is the canonical "what did the paper find" statement; omitting the LR-EMD direction here while §4.2/§4.3/§5 carry it is a scope-honest framing inconsistency.

- **Sections involved:** §1.4 bullet 3 (line 103) vs §4.2 (line 658), §4.3 (line 673), §5 (lines 781–783).
- **§1.4 bullet 3 quote (line 103):** "On the optical-density marginal, no advantage is observed." — end of bullet. No LR-EMD mention.
- **§4.2 quote (line 658):** "…and *fall short* on the log-return single-step marginal (LR-EMD)."
- **§5 quote (lines 781–783):** "and they *fall short* on the log-return single-step marginal (LR-EMD), where every classical adversarial baseline outperforms every quantum variant."
- **Discrepancy:** The principal-contributions bullet — which a reader will quote when describing the paper's findings — frames the bifurcated finding as two-pronged (LR-DTW win + OD null), but the body sections frame it as three-pronged (LR-DTW win + OD null + LR-EMD loss). A reader who reads only the Abstract+§1.4 will miss the LR-EMD scope hedge that §4.2/§5 carry.
- **Suggested fix:** Append a half-sentence to §1.4 bullet 3 line 103, after "On the optical-density marginal, no advantage is observed." → "On the log-return single-step marginal (LR-EMD), the direction reverses and every classical adversarial baseline outperforms every quantum variant (Section~4.1)." Also consider an Abstract addendum if word count allows; otherwise rely on §1.4 bullet 3 to surface the LR-EMD scope.

### F-3: Abstract drops "TOST equivalence not satisfied" hedge in §5's bookend; §5 leans on "statistically indistinguishable" alone

§5 (lines 778–780) describes the OD-marginal result as "statistically indistinguishable under the achievable n=5 power, Welch p > 0.36 and |Cohen's d| ≤ 0.65" — but does NOT in that same sentence say "TOST equivalence not satisfied". Abstract (line 49), §1.4 bullet 4 (line 105), §4.1 (lines 567–575), and §4.4 limitations (line 702) all explicitly say TOST is not satisfied / not demonstrated. §5 mentions "TOST-grade equivalence testing" only in the *future-work* sentence two paragraphs later (line 793), not in the OD-marginal claim itself.

The §5 bookend therefore reads as the strongest equivalence-supportive framing in the paper ("statistically indistinguishable" without the TOST hedge attached). Combined with F-1, a casual reader of just the Abstract and §5 could come away with "the quantum and classical OD-EMD are equivalent" — which §4.1 and the handoff's calibration-honesty standard explicitly forbid.

- **Sections involved:** §5 (lines 777–780) vs Abstract (line 49), §1.4 bullet 4 (line 105), §4.1 (lines 567–575), §4.4 (line 702).
- **§5 quote (lines 777–780):** "they *match* the full set of parameter-matched classical comparators (adversarial baselines plus VAE and AR(2); statistically indistinguishable under the achievable n=5 power, Welch p > 0.36 and |Cohen's d| ≤ 0.65) on the optical-density single-step marginal".
- **Abstract quote (line 49):** "TOST equivalence not satisfied."
- **§4.1 quote (lines 571–575):** "the matched-capacity quantum and classical adversarial generators are statistically indistinguishable on this single-step distributional axis, **but the data do not positively support an equivalence claim**."
- **Discrepancy:** §5 uses "statistically indistinguishable" without the immediate "but not positively equivalent" / "TOST not satisfied" guard that §4.1 and the Abstract attach. The §5 bookend therefore reads slightly stronger than the §4.1 source it bookends.
- **Suggested fix:** Add a clause inside §5's parenthetical at line 779–780. Change:
  - `…statistically indistinguishable under the achievable n=5 power, Welch p > 0.36 and |Cohen's d| ≤ 0.65) on the optical-density single-step marginal`
  to:
  - `…statistically indistinguishable under the achievable n=5 power, Welch p > 0.36 and |Cohen's d| ≤ 0.65, TOST equivalence not satisfied) on the optical-density single-step marginal`
  This keeps the §5 bookend faithful to the §4.1 source.

## NIT findings

### N-1: §5 references "Section 1.3" for the falsifiable question, but the §1.4 contributions bullet does not cross-reference §5

§5 (line 770) explicitly answers "the falsifiable question posed in Section~1.3". This is correct — the question is in §1.3 (Quantum Generative Adversarial Networks, line 92). §1.4 (Principal Contributions) and §4.1 / §4.2, however, restate the bifurcated finding without using the same "falsifiable question" framing, so a reader cross-referencing §5→§1.3 will not see a matching forward reference from §1.3 or §1.4 to §5. Minor — improves reader navigation; not a scientific error.

- **Sections involved:** §1.3 (line 92), §1.4 (lines 95–111), §5 (line 770).
- **§1.3 quote (line 92):** "This motivates the falsifiable question that frames the present study: *can a PQC generator … match or exceed a classical generator of equivalent parameter count on a low-data bioprocess task?*"
- **§5 quote (line 770):** "Returning to the falsifiable question posed in Section~1.3 …"
- **Suggested fix:** Optional. Add at end of §1.3 paragraph: "…we examine this question directly under a matched-parameter, matched-epoch protocol; the answer is given in Section~5." OR add at the end of §1.4 bullet 3: "(see Section~5 for the framed answer to the falsifiable question)".

### N-2: §4.1 line 410 says "three parameter-matched classical adversarial baselines and the AR(2) reference" — but bolded row leaders in Table 2 include VAE-row leader on OD-EMD; the §4.1 prose narrative under-references the full Table-2 contents

§4.1 (line 408–414) frames the LR-DTW + lag-1 ACF win as "the four quantum WGAN-GP variants uniformly outperformed the three parameter-matched classical adversarial baselines and the AR(2) reference". Table 2 (which §4.1 explicitly references) actually shows nine generators (4 quantum + 3 classical adversarial + VAE + AR(2)), and the VAE wins the OD-EMD row (bolded). The §4.1 prose at line 410 omits the VAE from the LR-DTW comparison cluster — which is intentional (VAE is excluded from the LR-DTW dominance comparison per the degenerate-regime characterization, marked with the † in Table 2). But the prose does not say "and the VAE (excluded as degenerate; see paragraph below)" the way Table 2's caption does. A reader cross-checking Table 2 → §4.1 prose will see VAE bolded for OD-EMD and ask why §4.1 narrative does not mention it at the same time. Minor — the VAE characterization paragraph two paragraphs later (lines 520–536) does explain it; the issue is only that the line-410 sentence omits the VAE-exclusion forward reference.

- **Sections involved:** §4.1 (line 410), Table 2 footnote (line 391), §4.1 VAE characterization (lines 520–536).
- **§4.1 line 410 quote:** "the four quantum WGAN-GP variants *uniformly* outperformed the three parameter-matched classical adversarial baselines and the AR(2) reference."
- **Suggested fix:** Optional. Insert "(the VAE is excluded as a degenerate generation regime; see characterization paragraph below)" after "AR(2) reference" on line 410.

## What was NOT flagged (sanity check)

Bookends are otherwise tight: (i) Abstract↔§5 agree on LR-DTW 0.94–1.12 vs 1.58–6.86, lag-1 ACF cluster −0.0997 to −0.0895 vs real −0.064, Welch p>0.36, |Cohen's d|≤0.65, n=5; (ii) §1.4 contributions↔§4.2 Key Contributions↔§4.5 Outlook agree that the decision-tree triage workflow is future-work-only, not an empirical contribution (all four mentions consistently qualified); (iii) §4.1 prose↔Table 2↔Figure 1 caption↔supp per-seed dominance table↔supp Welch OD-EMD/LR-EMD tables agree on per-model means, the V3-vs-wgan\_lstm worst/best pairing, and the 20-pair Welch-vs-TOST framing. Pipeline B prose contains no Lambert W outside the explicit "Pipeline C dropped" rationale in both main §3.2 and supp §A.7. Hybrid-GAN is qualified as proposed/not-implemented in all 11 supp §A.3 mentions plus the §4.5 Outlook reference.
