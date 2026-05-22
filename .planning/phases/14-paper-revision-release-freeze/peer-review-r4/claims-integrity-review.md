# Claims & Analysis Integrity Review (Agent 4) — qGAN v2.0-revision Freeze Gate

**Reviewer:** Agent 4 (Claims & Analysis Integrity), 6-agent r4 final review swarm
**Date:** 2026-05-21
**Scope:** Verify the two surviving headline claims are honestly supported by the
data; confirm the withdrawn Path A claim (LR-EMD beats WGANs) is fully scrubbed;
flag residual overclaiming.

## Environment note (procedural, not a finding)

The assigned git worktree (`agent-a325d2a77e14eae29`, HEAD `c82169c`) is stale at
Phase 08 — none of the Phase 14 files exist there. All review was performed by
**reading** (never writing) the main repo at `/Users/shawngibford/dev/phd/qGAN`,
HEAD `8180a5e`, which carries the complete Phase 14 plan-16 state. This is the
intended freeze state. No file in the main repo was modified.

---

## 1. SURVIVING CLAIM (a) — OD-EMD parametric-efficiency equivalence

**Claim text** (`revision/docs/reviewer_response.md:269-272`): "55 quantum
parameters achieve OD-scale EMD statistically equivalent to classical generators
of 73-562 generator params AND the full ~2.5x10^5 adversarial budget ... (Welch
p > 0.36, |d| ≤ 0.65, n=5)."

**Verification — VERIFIED, claim is honest.**

- `welch_pairwise.json` carries all 20 quantum×classical OD-EMD pairs (4 quantum
  variants × 5 baselines). Recomputed from the per-cell means/stds in the file:
  **min Welch p = 0.36521, max |Cohen's d| = 0.64417** (`welch_pairwise.json`
  lines 305-318, `summaries` block 674-675). The claim's "p > 0.36, |d| ≤ 0.65"
  is exactly the observed floor/ceiling — tight but truthful.
- Independent spot-check: recomputed the worst pair (iqp_sel_55 vs wgan_cnn, OD)
  by hand from the stored means/stds → Welch t = -1.0185, p = 0.3652, pooled
  Cohen d = -0.6442. Byte-matches `welch_pairwise.json:314-316`. The aggregator
  arithmetic is sound.
- Parameter counts verified against `revision/results/model_info.json`:
  wgan_cnn 73, wgan_mlp 74, wgan_lstm 78, vae 562, ar 3, iqp_sel_55 55. The
  "73-562" range correctly spans min generator (wgan_cnn) to max generator (vae).
- Shared critic 250881 params verified at
  `total_adversarial_param_budget.json#shared_critic_n_params`; the per-baseline
  table at `reviewer_response.md:298-304` matches `welch_pairwise.json` exactly.
- OD column byte-stability across the R3-CR-2 fix is asserted in
  `14-16-SUMMARY.md` (SHA-256 `560489fa3b44...` preserved) and consistent with
  the deviation note; the OD-EMD half of the claim is therefore independent of
  the LR-EMD bug.

**MEDIUM — n=5 / equivalence-by-non-rejection caveat (disclosure adequacy).**
The "statistically equivalent" claim rests on *failure to reject* a two-sided
Welch test at n=5 per group — i.e. an absence-of-evidence argument, not a TOST /
equivalence-bounds argument. With n=5 the test is badly underpowered, so a
non-significant p is weak evidence of equivalence. The docs are mostly honest
about this: `reviewer_response.md:273` says "no pair shows a statistically
significant OD-EMD difference" (correct framing), and the 14-15
marginal-convergence finding (`reviewer_response.md:208-258`) independently
documents that ALL 9 models make the same ~0.25-OD-unit marginal approximation —
which is the real reason the OD-EMD values cluster. That second finding actually
*supports* equivalence on independent grounds. Recommendation, not a blocker:
the word "equivalent" in the headline sentence is slightly stronger than a
non-rejection at n=5 strictly licenses; "statistically indistinguishable at the
matched-2000ep budget (n=5)" would be the fully calibrated phrasing. Not a
freeze blocker because the surrounding paragraphs already disclose the n and the
non-rejection logic, and the marginal-convergence finding corroborates it.

## 2. SURVIVING CLAIM (b) — DTW dominance

**Claim text** (`reviewer_response.md:278-281, 354-364`): quantum 0.298-0.302
OD-scale DTW beats Orlandi (1.954) ~6.5x; every quantum variant beats every WGAN
on log-return DTW.

**Verification — VERIFIED, claim is honest.**

From `matched2000_dualscale.json#aggregates`, `metric_name='dtw_mean'`:

| model | OD-DTW mean | LR-DTW mean |
|---|---|---|
| V1 | 0.300 | 0.940 |
| V2 | 0.298 | 0.949 |
| V3 | 0.299 | 1.122 |
| iqp_sel_55_repro | 0.302 | 0.985 |
| wgan_lstm | 0.301 | 1.581 |
| wgan_mlp | 0.302 | 2.624 |
| wgan_cnn | 0.438 | 6.863 |
| ar | 0.371 | 7.699 |
| vae | 0.307 | 0.088 |

- OD-DTW quantum cluster = 0.298-0.302 ✓ (matches claim literally).
- Orlandi ratio: 1.954 / 0.298 = 6.56x, 1.954 / 0.302 = 6.47x → "~6.5x" ✓.
- LR-DTW: max quantum = 1.122 (V3) < min WGAN = 1.581 (wgan_lstm). **Every
  quantum variant beats every WGAN on LR-DTW** ✓.
- **VAE artifact correctly handled (checked per task instruction).** VAE LR-DTW
  = 0.088 is lower than the quantum cluster, but the dominance claim is scoped
  to "every WGAN+AR" — VAE is explicitly EXCLUDED. `reviewer_response.md:359-361`,
  `methods_full.md:446-449`, and `peer_review_remediation.md:483-491` all flag
  the 0.088 as posterior collapse (sample std ≈ 0.0004) and state it is "NOT
  interpreted as evidence of model quality." The DTW dominance claim does **not**
  silently rely on the VAE anomaly. CLEAN.

**LOW — OD-DTW dominance is a near-tie, not a clean win, vs two WGANs.**
On OD-scale DTW, wgan_lstm (0.301) and wgan_mlp (0.302) sit inside the quantum
cluster (0.298-0.302); only ar (0.371) and wgan_cnn (0.438) are clearly worse.
The docs handle this correctly — `methods_full.md:441-442` explicitly says "the
OD-scale ordering is statistically non-significant under the strict-accept gate;
no equivalence test is computed for DTW" — and the OD-scale claim is framed only
as the Orlandi-ratio, not as a quantum-beats-WGAN claim. So the OD-DTW dominance
is correctly NOT claimed; the LR-DTW dominance IS the real win and it is genuine.
No fix needed; noted so the freeze record is complete.

## 3. PATH A SCRUB — withdrawn LR-EMD-vs-WGAN claim

**Result: SCRUB IS CLEAN.** No orphaned remnant found.

Searched all 7 paper-facing docs (`reviewer_response.md`, `methods_full.md`,
`peer_review_remediation.md`, `reconciliation_note.md`, `paper_blocks_framing.md`,
`paper_blocks_refs_methods.md`, `completeness_sweep_manifest.md`) for: (a) any
"quantum beats/outperforms/exceeds WGAN on LR/log-return EMD" sentence; (b) the
withdrawn pre-fix statistics (p ≤ 0.014, d ≤ -2.6); (c) any table cell or number
implying quantum < WGAN on LR-EMD.

- Every surviving mention of LR-EMD states the **correct, inverted** direction:
  "every WGAN beats every quantum on the corrected LR-EMD scale"
  (`methods_full.md:389`, `peer_review_remediation.md:354`), AR leads at 0.003,
  quantum/WGAN/VAE cluster 0.007-0.016 "with no statistically meaningful
  quantum-vs-WGAN separation" (`reviewer_response.md:274-276, 326-333`).
- The withdrawn pre-fix stats (p ≤ 0.014, d ≤ -2.6) appear **nowhere** as a live
  claim. The only occurrence of the withdrawn-claim text is inside explicit
  retraction prose: `peer_review_remediation.md:294` quotes
  `"significantly beats every WGAN on LR-EMD"` precisely to label it withdrawn.
- `welch_pairwise.json` emits the LR-EMD pairs as per-pair stats for
  transparency but its `strong_claim_thresholds` block (lines 679-682) contains
  **only OD-EMD thresholds** — no LR-EMD threshold — and the `notes` field
  (line 683) carries a machine-readable retraction record.
- `paper_blocks_framing.md:14` and `:125` (the manuscript-facing LaTeX framing
  blocks) state the quantum generator does **not** beat the classical WGAN-GP
  baselines — fully consistent with Path A; the de-overclaiming block PAPER-02
  is marked LOCKED regardless of which way the numbers fell (`:16-18`).
- The withdrawn claim was never in the published manuscript (`.tex` is
  read-only, D-14-18) — it was an r3-synthesis proposal. There is nothing in the
  LaTeX to scrub.

No CRITICAL or HIGH finding on the Path A scrub.

## 4. Residual overclaiming scan

Scanned all 7 docs for stronger-than-calibrated language ("quantum advantage",
"demonstrates advantage", "outperforms classical", "beats every classical").
All hits are benign:
- The hypothesis is phrased as a *falsifiable question* ("can a PQC generator
  match or exceed ...", `paper_blocks_framing.md:57`) — appropriate.
- `paper_blocks_framing.md:125` explicitly: structure is "comparable to, but do
  not exceed" classical baselines; "rather than as a method of demonstrated
  advantage." Correctly de-overclaimed.
- The Hybrid-GAN "demonstrates the advantages ... for future work"
  (`paper_blocks_framing.md:318`, `paper_blocks_refs_methods.md:352`) is the
  aspirational Supp Table A2, already caveated as "proposed extension (not
  implemented)" per R2-5a — pre-existing, closed item.
- `paper_blocks_refs_methods.md:211` quantum-advantage citation is flagged at
  `:223` as a "not-yet-demonstrated quantum-advantage citation" — honest.

No residual overclaiming finding.

## Findings summary

| # | Severity | Finding |
|---|----------|---------|
| F1 | MEDIUM | OD-EMD "equivalent" rests on non-rejection at n=5 (underpowered); "statistically indistinguishable (n=5)" is the fully calibrated wording. Disclosure is adequate (n stated, non-rejection logic stated, marginal-convergence finding corroborates) — recommend wording softening, not a blocker. |
| F2 | LOW | OD-DTW is a near-tie vs wgan_lstm/wgan_mlp; correctly NOT claimed as dominance (only the Orlandi ratio + LR-DTW dominance are claimed). Noted for completeness. |

No CRITICAL, no HIGH. Path A scrub is clean. VAE posterior-collapse artifact is
correctly excluded from the DTW dominance claim. Both surviving headline claims
are numerically verified against the frozen JSONs.

## Verdict rationale

Both surviving claims (a) OD-EMD equivalence and (b) DTW dominance are honestly
supported by `matched2000_dualscale.json` and `welch_pairwise.json` — spot-checks
reproduce the published numbers exactly. The withdrawn LR-EMD-vs-WGAN claim is
fully scrubbed from every paper-facing doc; the only surviving mentions are
correct (inverted) directional statements or explicit retraction prose. No
orphaned remnant would be frozen into the DOI. The two findings are MEDIUM/LOW
wording-calibration items that do not misrepresent the data and are already
substantially disclosed — they do not warrant blocking an irreversible freeze.

FREEZE VERDICT: GO
