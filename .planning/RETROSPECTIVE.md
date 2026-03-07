# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — qGAN Code Review Remediation

**Shipped:** 2026-03-07
**Phases:** 3 | **Plans:** 9 | **Sessions:** ~9

### What Was Built
- Safe checkpoint system with gradient-preserving restore and weights_only security
- Redesigned quantum circuit with data re-uploading, backprop differentiation, and expanded noise range
- WGAN-GP training loop with paper-correct hyperparameters, broadcasting, and one-sided gradient penalty
- EMD-based early stopping with correct Wasserstein distance on raw samples
- Unified denormalization pipeline ensuring training eval and standalone generation are identical
- Clean notebook with dead code removed, variable shadowing fixed, and duplicate plots consolidated

### What Worked
- Strict dependency-ordered phases (foundation → ML theory → cleanup) prevented cascading rework
- Single-file editing scope kept changes focused and git history clean (730 ins, 1,576 del net)
- Phase 3 gap-closure pattern: audit identified missed requirements (QUAL-03, QUAL-08) which were added as a targeted phase rather than reopening earlier phases
- Verification at each phase caught issues early (e.g., NUM_NUM_EPOCHS typo in Phase 1)
- Broadcasting-based training loop rewrite in Phase 2 was the highest-risk change but went smoothly due to Phase 1 infrastructure being solid

### What Was Inefficient
- ROADMAP.md progress table never got updated during execution (all rows show "Not started" at milestone end)
- Initial milestone audit flagged 2 gaps that could have been caught during Phase 2 planning if requirements coverage was checked more carefully
- Nyquist validation was added retroactively as manual-only strategies — integrating validation earlier would have been smoother

### Patterns Established
- In-place Jupyter notebook editing with cell-index tracking across plans
- `full_denorm_pipeline()` pattern for ensuring eval/generation consistency
- UPPER_CASE config cell as single source of truth for all hyperparameters
- Phase verification using python3 -c assertions on notebook JSON (no persistent test suite needed for research notebooks)

### Key Lessons
1. For research notebooks, a dependency-tiered remediation approach (safe changes first, breaking changes second, cleanup last) is the right structure — it minimizes risk of compounding errors
2. Milestone audits before completion are valuable — they caught 2 requirements that would have been missed
3. Variable shadowing in Jupyter notebooks is a persistent hazard — inlining values is safer than renaming when cells may be re-executed out of order
4. Cell index tracking across plans requires careful accounting when new cells are inserted (e.g., full_denorm_pipeline shifted all subsequent indices)

### Cost Observations
- Model mix: primarily quality profile (opus for execution, sonnet for research/planning)
- Sessions: ~9 plan executions across 3 phases
- Notable: Each plan averaged 3 minutes — tight scope kept execution fast

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~9 | 3 | First milestone — established dependency-tiered remediation pattern |

### Cumulative Quality

| Milestone | Requirements | Satisfied | Tech Debt Items |
|-----------|-------------|-----------|-----------------|
| v1.0 | 35 | 35 (100%) | 2 (non-blocking) |

### Top Lessons (Verified Across Milestones)

1. Dependency-tiered phases prevent cascading rework in notebook remediation
2. Milestone audits catch gaps that phase-level verification misses
