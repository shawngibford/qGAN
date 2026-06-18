# Phase 14: Paper Revision & Release Freeze - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-19
**Phase:** 14-paper-revision-release-freeze
**Areas discussed:** Epoch-parity remediation, Speed/device strategy, Model info table, Table provenance, Manuscript & change-tracking, Downstream artifact regeneration scope, Release-freeze mechanics, Reviewer-response checklist format, Re-run validation gate, Quantum-result honest framing, Paper-edit deliverable format, Sweep sequencing & figure suite, Canonical-config recovery, Compute time-box & overflow, Phase-internal ordering & release gating, Checkpoint reproduction landmines, Canonical pipeline pinning, Roadmap scope reconciliation, Config-source authority rule, Figure-suite gap analysis

---

## Epoch-parity remediation

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run all comparison at 2000ep | Re-run classical + non-adversarial + ansatz (incl. quantum comparison) at matched 2000ep | ✓ |
| Re-run only cheap classical + caveat quantum | Classical/VAE/AR to 2000ep, caveat quantum budget | |
| Document budgets + caveat only | No re-run, record budgets + Methods limitation | |

**User's choice:** Re-run all at 2000ep.
**Notes:** "is all of this being run on MPS? or in parallel? we should re-run all of them at 2000ep. so, i pick number 1, but i want this to run as fast as possible. i dont want it to silently run on cpu while claiming MPS." → drove the speed/device discussion.

---

## Speed/device strategy

| Option | Description | Selected |
|--------|-------------|----------|
| default.qubit + xargs -P2 + device manifest | No core change, numerically continuous, hard-assert backend | ✓ |
| Switch to lightning.qubit | C++ speedup but forces adjoint, frozen-core change, re-baseline | |
| default.qubit + xargs -P2, existing logging | No new manifest assertion | |

**User's choice:** default.qubit + xargs -P2 + device manifest (Recommended).
**Notes:** Established that the quantum generator is inherently a CPU statevector sim (PennyLane not Metal-accelerated); only the classical critic touches MPS.

---

## Model info table

| Option | Description | Selected |
|--------|-------------|----------|
| One unified table, all models as rows | Single paper-ready table, parity visible at a glance | ✓ |
| Grouped sub-tables by family | Per-family detail, cross-reference needed | |
| Unified table + shared-protocol callout | Unified + prose parity box | |

**User's choice:** One unified table, all models as rows.

---

## Table provenance

| Option | Description | Selected |
|--------|-------------|----------|
| New model_info.json emitter, table rendered from it | JSON source-of-truth; docs regenerated | ✓ |
| Extend existing markdown docs as source | No new JSON; criterion 5 only partial | |
| JSON emitter, docs hand-maintained separately | Two sources, drift risk | |

**User's choice:** New model_info.json emitter, table rendered from it.

---

## Manuscript & change-tracking

| Option | Description | Selected |
|--------|-------------|----------|
| Source external (Overleaf) — produce revision package | Edit instructions + response doc + tables/figures | ✓ |
| Bring .tex into repo and edit directly | Self-contained, edits .tex in repo | |
| Markdown working draft + response doc | Avoids LaTeX tooling | |

**User's choice:** Option 1, but user adds `main (4) copy.tex` + `paper/supp_material.tex` as READ-ONLY reference — "do not directly edit them."
**Notes:** Both files confirmed present at repo root.

---

## Downstream artifact regeneration scope

| Option | Description | Selected |
|--------|-------------|----------|
| Tiered: headline + utility at 2000ep; sensitivity/ansatz caveated | Bounded compute, some caveats | |
| Full regeneration at 2000ep | Every artifact regenerated, zero caveats, largest cost | ✓ |
| Headline comparison + model_info table only | Smallest, weakest coherence | |

**User's choice:** Full regeneration at 2000ep.

---

## Release-freeze mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Code + results + docs + .tex refs; exclude env/data/checkpoints | Hash-referenced checkpoints, data.csv included | ✓ |
| Everything including env + checkpoints | Maximum reproducibility, heavy archive | |
| Minimal: code + final paper-cited JSON only | Smallest archive | |

**User's choice:** Code + results + docs + .tex refs; exclude env/checkpoints.

---

## Reviewer-response checklist format

| Option | Description | Selected |
|--------|-------------|----------|
| Per-comment table grouped by reviewer | comment ID → concern → change → location → artifact | ✓ |
| Single flat change-log table | All changes in change-order | |
| Prose rebuttal + appendix table | Narrative letter + traceability table | |

**User's choice:** Per-comment table grouped by reviewer.

---

## Re-run validation gate

| Option | Description | Selected |
|--------|-------------|----------|
| Strict: manifest + data_hash + seeds + structural parity | + 1000→2000ep reconciliation note | ✓ |
| Standard: manifest + schema only | Weaker provenance | |
| Strict + per-model sanity bounds | Strongest but bounds risk arbitrariness | |

**User's choice:** Strict gate: manifest + data_hash + seeds + structural parity.

---

## Quantum-result honest framing

| Option | Description | Selected |
|--------|-------------|----------|
| Evidence-led, claim-calibrated | Report what numbers show, calibrate claims | |
| Quantum-favorable emphasis | Lead with quantum strengths (overclaim risk) | |
| Defer framing to paper-writing | Capture both directions, decide once numbers land | ✓ |

**User's choice:** Defer framing to paper-writing.
**Notes:** Captured with the constraint that PAPER-02 claim-calibration is a locked non-negotiable reviewer requirement regardless of tone.

---

## Paper-edit deliverable format

| Option | Description | Selected |
|--------|-------------|----------|
| Copy-paste LaTeX blocks keyed to section/label | + reviewer-comment rationale per change | ✓ |
| old→new diff snippets | Awkward for new subsections/insertions | |
| Prose change instructions | User must translate to LaTeX | |

**User's choice:** Copy-paste LaTeX blocks keyed to section/label.

---

## Sweep sequencing & figure suite

| Option | Description | Selected |
|--------|-------------|----------|
| Resumable background sweeps + comprehensive per-model figure suite | xargs -P2, stall-watchdog, full PDF+PNG suite | ✓ |
| Same sweeps, paper-bound figures only | Less output, incomplete suite | |
| Let me adjust specifics | — | |

**User's choice:** Option 1.
**Notes:** "will this train all of the quantum variants again as well, we need to make sure everything is being treated equally, and this will back up the claim about the quantum circuit selection" → confirmed all ansatz variants retrain at 2000ep; headline vs reproduction instance reported distinctly.

---

## Canonical-config recovery (55-param IQP:SEL)

| Option | Description | Selected |
|--------|-------------|----------|
| Reverse-engineer + lock 55-param IQP:SEL as selectable variant | Reconstruct from checkpoint+notebook, hard-assert load | ✓ |
| Treat 75-param current core as the circuit; checkpoint as legacy | Abandons proven-good result | |
| Investigate first, decide in planning | Defers lock decision | |

**User's choice:** Option 1, "but I want this 55-param circuit to also be run against the other models."
**Notes:** Triggered by the concrete 55 ≠ 75 param finding. The 55-param IQP:SEL becomes the quantum entrant in every cross-model comparison.

---

## Headline guarantee (best_checkpoint.pt)

| Option | Description | Selected |
|--------|-------------|----------|
| Freeze best_checkpoint.pt as canonical + reproduction run as evidence | Checkpoint = headline; retrain non-load-bearing | ✓ |
| Retrain everything fresh, trust Phase-8 parity | Gambles headline on exact reproduction | |
| Checkpoint-only, no retrain | Forfeits reproducibility claim | |

**User's choice:** Freeze best_checkpoint.pt as canonical headline + reproduction run as evidence.

---

## Compute time-box & overflow

| Option | Description | Selected |
|--------|-------------|----------|
| Run-to-completion, resumable, tiered priority, no hard time-box | Correctness over speed, tier-independent acceptance | ✓ |
| Hard wall-time box with partial-acceptance fallback | Bounds time, reintroduces caveats | |
| Estimate first, then decide | Adds a timing-probe step | |

**User's choice:** Run-to-completion, resumable, tiered priority, no hard time-box.

---

## Phase-internal ordering & release gating

| Option | Description | Selected |
|--------|-------------|----------|
| Strict gated pipeline, release freeze last | Steps 6–7 hard-blocked until gate passes | ✓ |
| Parallelize paper-prep with late sweeps | Faster, rework risk | |
| Let me adjust the sequence | — | |

**User's choice:** Strict gated pipeline, release freeze last.

---

## Final de-risking bundle (confirmed together)

| Item | Resolution | Selected |
|------|------------|----------|
| Checkpoint reproduction landmines | Use stored mu/sigma + fixed generation seed | ✓ |
| Canonical pipeline pinning | Pin native pipeline for headline; all pipelines in comparisons | ✓ |
| Roadmap scope reconciliation | Record intentional scope expansion; flag ROADMAP/REQUIREMENTS update | ✓ |
| Config-source authority rule | Checkpoint tensor = ground truth; notebook = corroborating | ✓ |
| Figure-suite gap deliverable | Full per-model+cross-model suite; match/exceed 20-figure canonical bar | ✓ |

**User's choice:** Confirm all five — write CONTEXT.md.
**Notes:** "take a full sweep of the repo and look for plots/figures that we are currently missing and add them, and then add all of the remaining code for figures/plots that are not included that the reviewers and other readers would want to see." → folded into D-14-17.

---

## Claude's Discretion

- model_info.json schema fields, table column ordering, markdown layout
- Figure styling, subplot composition, naming, port-vs-rewrite of legacy plots
- Checkpoint-introspection + git-archaeology mechanics for the 55-param decomposition
- Sweep driver structure (established run_*.py + *_sweep.sh xargs -P2 pattern)
- Stall-watchdog / subagent-permission settings per project compute-heavy memory

## Deferred Ideas

- lightning.qubit acceleration — rejected (adjoint forced, frozen-core, re-baseline)
- Closed-loop decision pipeline & first-principles Hybrid-GAN — Outlook only, not implemented
