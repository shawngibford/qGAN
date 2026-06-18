# Paper-Rewrite Framing Decisions (locked 2026-05-27)

These decisions resolve open questions from `PAPER-REWRITE-HANDOFF.md` §7. Every writer agent (W1–W4) and audit agent (A1–A7) references this file. Do **not** relitigate per-wave; if a writer wants to deviate, escalate to the orchestrator.

---

## §7.2 Abstract Lead — **Finding 2 leads**

Lead sentence of the abstract emphasizes the **uniform quantum dominance on log-return temporal structure** (LR-DTW + lag-1 ACF concordance). Finding 1 (parametric-efficiency equivalence on OD-marginal) follows as a scope-honest caveat.

Anchor opening (writer may polish — not rewrite the substance):

> We report that quantum WGAN-GPs with 55–135 parameters uniformly outperform parameter-matched classical adversarial baselines on log-return temporal structure (LR-DTW 0.94–1.12 vs 1.58–7.70) while reproducing the lag-1 ACF of the real series (−0.064) more closely than any classical generator. On the OD-marginal, no parametric-efficiency advantage is observed (Welch p > 0.36, max |Cohen's d| ≤ 0.65, power ≈ 15%).

Rationale: matches Reviewer 2's hypothesis-first preference; the LR-DTW + ACF concordance is the most distinctive contribution.

## §7.3 Figure Placement

| Figure | Location |
|---|---|
| `cross_model_dtw_dualscale` | **Main §4.1 Results** |
| `cross_model_acf_overlay` | **Main §4.1 Results** |
| `preprocessing_pipeline_4panel` | **Supp §A.7** |

Both finding-defining figures (DTW + ACF) land in the main results subsection. The preprocessing 4-panel is methodological detail and belongs in supp. If A7 (compile-tester) reports a page-budget overrun, escalate — do not silently demote.

## §7.4 Outlook (§4.5) Framing

LR-DTW extension conditions are **scoped explicitly in Outlook**: multivariate data, longer time series, higher qubit count past the simulable boundary, larger seed budget for TOST-grade equivalence. §4.4 Limitations briefly cross-refs Outlook for detail; does not duplicate the scope list.

## §7.5 Discriminative-Score Uniformity (0.40888 fixed point)

**Brief mention in Methods §3** describing the discriminative-score interpretation under Pipeline B. **No separate subsection** in Results or Supp. One or two sentences only — the finding is methodological (about the evaluation protocol), not a generator-quality result.

## §7.1 Title — DEFERRED to W1

W1 (Main-Reframer) inspects the current title against the §4.1 prohibition on "deployable-framework / industrial-monitoring" framing and proposes 2–3 rescope candidates to the orchestrator. Human picks. Do not edit the title without human approval.

## Tag Name — DEFERRED to Wave 8

`v1.2` vs `v1.0-revision.final` — orchestrator's call at submission time, not the swarm's call.

---

## Cross-Cutting Constraints (every agent)

1. **Numeric literals**: every value in the .tex must trace to a JSON cell in `results/`. The provenance gate (`scripts/verify_number_provenance.py`) enforces this. Writers must include a `json_source` reference in their return for every new literal.
2. **Prohibition list (§4.1)**: 8 hard rules. A2 enforces deterministically. Hard-blocking phrases:
   - "posterior collapse" (VAE characterization)
   - "4×10⁻⁴" / `4 \times 10^{-4}` (the withdrawn VAE std)
   - "−0.029" / "-0.029" as real-ACF reference (use −0.064 / −0.0641)
   - "LR-EMD" near "quantum/outperform/beats" (withdrawn finding)
   - "0.6843" outside an explicit historical-reference clause
   - "n=1" or "single representative seed" for shot-noise / noise sweeps (both use n=3)
   - "deployable framework", "industrial bioprocess monitoring", "high fidelity", "strong performance", "computational advantages" outside the explicitly-labelled Outlook subsection
   - "Hybrid-GAN" as "implemented", "evaluated", or "demonstrated" (it is a *proposed extension that was not implemented or evaluated*)
   - "closed-loop feedback control" for the decision-tree workflow (it is a decision-tree triage workflow demoted to Outlook)
3. **Bifurcated finding coherence**: LR-DTW and lag-1 ACF must be treated as **one** structural-fidelity finding, not two metric observations. Every section that references them references both jointly.
4. **VAE characterization**: degenerate generation regime — marginal well-aligned (LR-EMD = 0.016), lag-1 ACF sharply different (−0.648 vs real −0.064). Never "posterior collapse".
5. **Bib.bib is frozen**: 59 entries. No new citations without orchestrator approval.

---

## Reference numbers (load-bearing, from `results/*.json`)

### OD-EMD (matched 2000 epochs, n=5 seeds)
- iqp_sel_55_repro: 0.02753 ± 0.00513
- wgan_mlp: 0.02595
- wgan_lstm: 0.02821
- wgan_cnn: 0.05432 (seed-42 outlier inflates mean)

### LR-DTW (matched 2000 epochs)
- V1: 0.9400 | V2: 0.9495 | iqp_sel_55_repro: 0.9855 | V3: 1.1225
- wgan_lstm: 1.5812 | wgan_mlp: 2.6243 | wgan_cnn: 6.8630
- ar(2): 7.6991
- vae: 0.0876 (excluded from comparison — degenerate)

### Lag-1 ACF (matched-pipeline, real-data reference includes dither)
- Real: **−0.0641** (NOT −0.029)
- Quantum cluster: V1 −0.0997, V2 −0.0968, iqp_sel_55 −0.0949, V3 −0.0895
- VAE: −0.6482 (anomaly)

### Welch t-test (20 quantum-vs-classical pairs)
- Max p > 0.36
- Max |Cohen's d| ≤ 0.65
- Power at n=5 against d=0.65: ≈ 15%
- 80%-power detection floor: d ≈ 2.0

### Training protocol (matched-budget contract)
- 2000 epochs (NOT 1000)
- n=5 seeds: {42, 43, 44, 45, 46}
- Shared 250881-parameter critic
- Adam β₁=0.0, β₂=0.9 (per `results/model_info.json#models[*].optimizer_betas`; handoff §3 had stale β₁=0.5 transcription); LR_gen = 6.9173×10⁻⁵; LR_critic = 1.8046×10⁻⁵
- n_critic = 9; λ_gp = 2.16; batch = 12

### Robustness sweeps
- Shot-noise: n=3 seeds {42, 43, 44} across analytic / 1024 / 8192 shots
- Noise channels: n=3 seeds across depolarizing (0%, 5%) and amplitude-damping (5%)

### Dataset shape
- Raw OD: 778 points, single campaign, LUCY 20L photobioreactor, 880nm sensor
- Log-returns: 777
- Rolling windows: 384 (length 10, stride 2)
- Train/val/test: 384/0/0

### Parameter counts
- Quantum: iqp_sel_55: 55, V1: 75, V2: 135, V3: 75
- Classical: wgan_mlp: 74, wgan_cnn: 73, wgan_lstm: 78
- VAE: 562 | AR(2): 3 | Critic: 250881
