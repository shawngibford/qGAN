# Phase 11: Utility Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 11-utility-evaluation
**Areas discussed:** Soft-sensor task, Predictive/discriminative fidelity, Augmentation design, Sample provenance

---

## Soft-sensor task (EVAL-01)

| Option | Description | Selected |
|--------|-------------|----------|
| PAR_LIGHT → OD mapping | Predict OD from PAR_LIGHT conditioning; mirrors manuscript soft-sensor framing | |
| One-step-ahead forecast | Predict OD[t+1] from preceding window; canonical TSTR setup | ✓ |
| Window → next-window | Seq-to-seq next-window prediction; heavier, noisier | |

**User's choice:** One-step-ahead forecast
**Notes:** Cleanest comparison, no conditioning needed, aligns with predictive score.

---

## Predictive/discriminative fidelity (EVAL-02/03)

| Option | Description | Selected |
|--------|-------------|----------|
| Faithful TimeGAN post-hoc nets | Canonical GRU predictor + discriminator definitions | ✓ |
| Lite reuse of Phase 10 infra | Reuse 1-layer LSTM-32 TSTR-lite scaffolding | |
| Faithful + cite reference impl | Faithful nets + pinned reference implementation in metadata | |

**User's choice:** Faithful TimeGAN post-hoc nets
**Notes:** Reviewers asked for standard utility tests. Claude added reference-impl pinning as zero-cost discretion (strictly more defensible).

---

## Augmentation design (EVAL-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Mixing-ratio sweep | Real-only → real+synthetic at several ratios; lift curve | ✓ |
| Single augmented condition | Real-only vs real+synthetic at one fixed ratio | |
| Sweep, small-real regime | Mixing sweep with reduced real set to amplify detectable lift | |

**User's choice:** Mixing-ratio sweep
**Notes:** Downstream task = same one-step-ahead soft-sensor as EVAL-01. Small-real regime captured as deferred robustness check.

---

## Sample provenance

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse Phase 10 / 09.1 artifacts | Read existing samples.npy as-is; no regeneration | ✓ |
| Reuse, Pipeline B headline only | Reuse but full suite on B only; A as subset | |
| Regenerate fresh samples | Re-run generators for larger sample sets | |

**User's choice:** Reuse Phase 10 / 09.1 artifacts
**Notes:** Both Pipeline A and B evaluated (matches Phase 10 coverage); B remains headline. Preserves identical-protocol / no-retraining invariant.

---

## Claude's Discretion

- Exact post-hoc GRU hyperparameters (pin to cited TimeGAN reference impl during research).
- Record TimeGAN reference implementation in JSON metadata even though "faithful" (not "faithful + cite") was selected — zero cost, more defensible.
- Soft-sensor architecture choice (1D-CNN vs LSTM) — single consistent architecture preferred.
- Augmentation mixing-ratio grid resolution.
- Subsampling strategy from existing artifacts.

## Deferred Ideas

- PAR_LIGHT-conditioned soft-sensor — follow-up if reviewers want the conditioned framing.
- Small-real-regime augmentation — backlog robustness check if standard sweep shows no lift.
- CR-01 / CR-02 training-loop fixes — owned by Phase 13 (reviewed, not folded).
