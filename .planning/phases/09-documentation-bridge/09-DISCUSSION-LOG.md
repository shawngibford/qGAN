# Phase 9: Documentation Bridge - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-08
**Phase:** 09-documentation-bridge
**Areas discussed:** Train/val/test split, Campaign count, Differentiable Lambert W, Doc style + 9.1 prep

---

## Train/val/test split

| Option | Description | Selected |
|--------|-------------|----------|
| Chronological + gap | Time-ordered split with WINDOW_LENGTH-1=9 step gap to prevent overlap leakage | |
| Chronological no gap | Time-ordered, no gap (allows up to 9 timesteps of overlap leakage) | |
| Random window-level | Shuffle rolling windows then split (max info, max contamination) | |

**User's choice:** Free-text override — "we do not need a train/test/val split. we have 1 dataset that is small. we need all the data we can get."
**Notes:** Strong principled stance: low-data + single-campaign reality means held-out splits cost more than they yield. EMD early-stopping uses same-distribution evaluation — accepted as a documented methodological constraint per R1-M5 calibration honesty.

### Round-trip target for EVAL-06

| Option | Description | Selected |
|--------|-------------|----------|
| Full log_delta series | Round-trip the real 776-element tensor; data-path correctness | |
| Synthetic random tensor | torch.randn matching shape; pure correctness, decoupled from data | |
| Both | Synthetic + real; covers both axes | ✓ |

**User's choice:** Both
**Notes:** Sets the verification pattern Phase 09.1 reuses for its three pipelines.

---

## Campaign count

| Option | Description | Selected |
|--------|-------------|----------|
| Exactly 1 | Single bioreactor campaign; honest single-campaign framing | ✓ |
| More than 1, only 1 used | Other campaigns exist but excluded with explanation | |
| More than 1, all should be used | Multi-campaign expansion (significant scope creep) | |

**User's choice:** Exactly 1
**Notes:** dataset_stats.md will state this plainly with a one-paragraph "Single-Campaign Limitation" prose block; multi-campaign generalization → Phase 14 Outlook.

---

## Differentiable Lambert W

| Option | Description | Selected |
|--------|-------------|----------|
| Claude's discretion | torch.autograd.Function wrapping scipy with closed-form derivative dW/dz=W/(z(1+W)) | ✓ |
| Pure-torch Halley iteration | 5-10 Halley iterations, no scipy dependency, slightly slower | |
| Third-party lib | torchlambertw or similar; adds external dep | |

**User's choice:** Claude's discretion → option (a) locked
**Notes:** In-place replacement of `inverse_lambert_w_transform` in `revision/core/data.py`. scipy stays in forward path, autograd path is pure torch on the cached W value. No new dependencies.

---

## Doc style + 9.1 prep

### Doc style

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid — tables + prose | Tables for numbers + 1-paragraph prose for justifications; drop-in for Methods | ✓ |
| Terse structured tables | Tables only, one-line explanations | |
| Full Methods prose | Full prose drop-in for manuscript | |

**User's choice:** Hybrid

### 9.1 scaffolding

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — minimal scaffolding | Add preprocessing.py with full contract; only Lambert pair implemented in Phase 9 | ✓ |
| No — strictly Phase 9 scope | Phase 9 only modifies data.py; 9.1 owns all preprocessing.py work | |
| Yes — implement all 6 now | Phase 9 implements all forward/inverse pairs; 9.1 just trains+evaluates | |

**User's choice:** Yes — minimal scaffolding
**Notes:** Locks the API contract. Phase 9 does not refactor `data.py` symbols; preprocessing.py re-exports the Lambert pair under unified names while data.py remains the single source of truth.

---

## Wrap-up

**Question:** "Any remaining gray areas before I write CONTEXT.md?"
**User's choice:** Ready for context
**Notes:** Decisions sufficient — proceed to write 09-CONTEXT.md and route to /gsd-plan-phase 9.

---

## Claude's Discretion

- Implementation choice for differentiable Lambert W (locked option a from explicit user delegation).
- Section ordering inside training_protocol.md and dataset_stats.md.
- Exact wording of single-campaign limitation paragraph.
- Whether to include a "Reproducibility" subsection in training_protocol.md.
- File-level layout of `preprocessing.py` (one function per pair vs grouped).
- Synthetic tensor distribution + dtype for round-trip test.
- Light docstring + type-hint additions to `data.py` Lambert functions if non-behavioral.

## Deferred Ideas

- Pipeline A (raw OD) and Pipeline B (log-returns only) implementations → Phase 09.1.
- Multi-seed run framework → Phase 09.1 builds it; Phase 12 generalizes.
- Dataset histograms / OD-level moments in dataset_stats.md → Phase 11 if needed.
- Multi-campaign data pipeline → Phase 14 Outlook.
- Shot-noise quantitative reporting → Phase 12 (SENS-01).
