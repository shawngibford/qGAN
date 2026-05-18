---
phase: 12-sensitivity-analysis
verified: 2026-05-18T00:00:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
---

# Phase 12: Sensitivity Analysis — Verification Report

**Phase Goal:** Quantum results are stress-tested under shot noise, hardware-style noise channels, and seed variation — so the manuscript reports calibrated uncertainty bars and directly addresses R1-M4 and R2-1 preliminary-result concerns.
**Verified:** 2026-05-18
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Shot-noise sweep at {analytic, 8192, 1024} shots run for quantum generator; metric degradation curve written to revision/results/shot_noise_sensitivity.json | VERIFIED | File exists; 270 rows; schema complete; shots set exactly {None, 8192, 1024}; seeds {42,43,44}; both pipelines; dual-scale |
| 2 | Noise-model sensitivity for depolarizing (p in {0,0.001,0.01,0.05}) and amplitude-damping (gamma in {0,0.001,0.01,0.05}) written to revision/results/noise_model_sensitivity.json | VERIFIED | File exists; 720 rows; noise_model {depolarizing, amplitude_damping}; noise_level {0.0,0.001,0.01,0.05} for each model; ampdamp_0.01 present; per-layer channel_insertion recorded in provenance |
| 3 | Every headline comparison table (Phases 10-11) re-emitted with >=5 seeds, mean +/- std per cell — revision/results/multiseed_summary.json consolidates the roll-up | VERIFIED | File exists; 1266 cells; 870 five-seed cells with n==5 and seeds=[42,43,44,45,46]; data_hash 91e447d4624e25b3 asserted across all five consumed artifacts; 168 D-11-09 N/A null cells faithfully propagated (not fabricated or dropped) |
| 4 | Compute budget respected — sweeps complete on local Mac statevector simulator within the phase session (documented in summary) | VERIFIED | 12-02-SUMMARY.md documents sweep wall time 8m 22s against < 10-min budget; sweep_status.json: all_complete=true, completed_count==total_count==66, failed=0 |

**Score:** 4/4 truths verified

**Cross-cutting constraint — revision/core/ byte-untouched:**

`git diff --stat revision/core/` is empty. `git status revision/core/` reports nothing to commit, working tree clean. VERIFIED across all three plans.

---

### CR-01 Reconciliation (Independent Verification)

The REVIEW.md claims that after the CR-01 fix (commit 80208f6), the full 66-cell sweep was re-run and the Pipeline-B seed-42 log_return EMD now reconciles with the frozen `fidelity_dualscale.json` value within < 1e-6.

Independent measurement from the actual artifacts:

| Metric | Frozen (fidelity_dualscale.json) | Regenerated (shot_noise_sensitivity.json, B/42 analytic) | Abs Delta | Threshold | Status |
|--------|----------------------------------|----------------------------------------------------------|-----------|-----------|--------|
| log_return EMD | 0.1209437521974767 | 0.12094375219747686 | 1.53e-16 | < 1e-6 | VERIFIED |
| OD-scale EMD | 0.022937980562900886 | 0.022937980562900893 | 6.94e-18 | < 1e-6 | VERIFIED |

Both headline numbers reconcile to machine epsilon. The CR-01 reconciliation claim is independently confirmed. Both JSONs share the same `generated_at: 2026-05-18T22:11:15Z`, confirming they were re-aggregated together after the CR-01 fix.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `revision/run_sensitivity.py` | SENS-01/02 per-cell CLI driver | VERIFIED | 900+ lines; AST-parses clean; PennyLane 0.44.0 assert at startup; _find_repo_root present; params_pqc reload; set_shots; default.mixed; DepolarizingChannel; AmplitudeDamping; ampdamp_0.01; diff_method=None (6 instances); no multiprocessing; *0.1 generation contract; seed*7919+1 reconstruction |
| `revision/run_sensitivity_sweep.sh` | Idempotent xargs -P 2 sweep orchestrator | VERIFIED | Syntax-valid (bash -n exits 0); xargs -P; flock/os.fsync/os.rename atomic status; no qgan_env; no multiprocessing; 11-condition CONDITIONS list incl. ampdamp_0.01; SEEDS="42 43 44" only; is_complete content-validates metrics.json and samples.npy (CR-03 fix present) |
| `revision/run_multiseed_rollup.py` | SENS-03 pure stdlib aggregator | VERIFIED | AST-parses clean; _find_repo_root; data_hash assert before any rollup math; statistics.fmean/stdev; no torch/pennylane/core.models imports; HEADLINE lists all 5 files; injection_ratio in groupby key |
| `revision/results/shot_noise_sensitivity.json` | SENS-01 deliverable | VERIFIED | 270 rows; extended long-form schema; shots {None,8192,1024}; seeds {42,43,44}; pipelines {A,B}; scales {OD,log_return} |
| `revision/results/noise_model_sensitivity.json` | SENS-02 deliverable | VERIFIED | 720 rows; extended long-form schema; noise_model {depolarizing,amplitude_damping}; noise_level {0.0,0.001,0.01,0.05} per model; seeds {42,43,44}; pipelines {A,B}; scales {OD,log_return}; channel_insertion provenance present |
| `revision/results/multiseed_summary.json` | SENS-03 deliverable | VERIFIED | 1266 cells; data_hash 91e447d4624e25b3; all 5 consumed_artifacts mapped to canonical hash; seed_set [42,43,44,45,46]; 870 five-seed cells (n==5); 168 null cells = D-11-09 N/A cells (fidelity_dualscale/log_return/Pipeline-A only) — correct faithful propagation |
| `revision/results/sensitivity/runs/analytic/B/42/metrics.json` | Harness-faithfulness smoke cell | VERIFIED | Exists; 20 rows; OD-scale EMD 0.022937980562900893 (delta 6.94e-18 vs frozen); log_return EMD 0.12094375219747686 (delta 1.53e-16 vs frozen); dual-scale confirmed |
| `revision/results/sensitivity/sweep_status.json` | Sweep completion record | VERIFIED | all_complete: true; completed_count=66; total_count=66; failed cells: 0 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| run_sensitivity.py | transform_ablation/runs/<pipeline>/<seed>/checkpoint.pt | torch.load params_pqc reload | WIRED | `ck["params_pqc"]` assigned to `g.params_pqc.data`; path anchored at REPO |
| run_sensitivity.py | revision.core.eval.full_metric_suite | dual-scale fidelity recompute | WIRED | Imported from revision.core.eval; called in compute_dualscale_metrics |
| run_sensitivity_sweep.sh | run_sensitivity.py | xargs -P 2, one cell per python invocation | WIRED | `$PYTHON revision/run_sensitivity.py --pipeline $p --seed $s --condition $c --out-root` |
| run_sensitivity_sweep.sh | xargs -P | OS-process parallelism | WIRED | `xargs -P "$PARALLEL" -L 1`; --parallel guardrail rejects >2 |
| run_multiseed_rollup.py | revision/results/{baseline_comparison,tstr,predictive_discriminative,augmentation,fidelity_dualscale}.json | json.load + cross-artifact data_hash assert | WIRED | All 5 loaded; assert before rollup math; canonical_hash 91e447d4624e25b3 |
| run_multiseed_rollup.py | revision/results/multiseed_summary.json | groupby -> mean +/- std | WIRED | statistics.fmean/stdev; output written via Path.write_text |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| shot_noise_sensitivity.json | rows[] | 66-cell sweep via run_sensitivity.py; params from frozen checkpoints | Yes — 270 rows from real QNode forward passes + frozen analytic samples | FLOWING |
| noise_model_sensitivity.json | rows[] | Same sweep, noise-channel conditions | Yes — 720 rows from default.mixed QNode runs with real DepolarizingChannel/AmplitudeDamping | FLOWING |
| multiseed_summary.json | rollup[] | Five frozen Phase 10/11 headline JSONs (1710+3360+144+120+180 rows) | Yes — aggregated from existing 5-seed data; 1266 cells; 870 with n==5 | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| shot_noise_sensitivity.json has correct schema and shot levels | python3 schema assert | AGG_OK 270 720; all assertions pass | PASS |
| noise_model_sensitivity.json covers all ampdamp levels including 0.01 | python3 noise_level check | {0.0, 0.001, 0.01, 0.05} confirmed for both models | PASS |
| multiseed_summary.json 5-seed cells present | python3 n==5 check | 870 five-seed cells; seed_set [42,43,44,45,46] | PASS |
| CR-01 reconciliation: log_return EMD delta vs frozen | abs(regen - frozen) | 1.53e-16 < 1e-6 | PASS |
| CR-01 reconciliation: OD-scale EMD unchanged | abs(regen - frozen) | 6.94e-18 < 1e-6 | PASS |
| sweep_status.json all_complete | python3 assert check | all_complete=true; 66/66; 0 failed | PASS |
| revision/core/ untouched | git diff --stat | Empty output; exit 0 | PASS |

---

### Probe Execution

No formal `scripts/*/tests/probe-*.sh` probes declared for this phase. Behavioral spot-checks above serve as the equivalent functional verification.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|------------|-------------|--------|---------|
| SENS-01 | 12-01-PLAN, 12-02-PLAN | Shot-noise sweep at {analytic, 8192, 1024} shots; metric degradation reported | SATISFIED | shot_noise_sensitivity.json exists; 270 rows; correct shot levels; degradation trend documented in 12-02-SUMMARY |
| SENS-02 | 12-01-PLAN, 12-02-PLAN | Noise-model sensitivity — depolarizing p in {0,0.001,0.01,0.05} and amplitude-damping gamma in {0,0.001,0.01,0.05} | SATISFIED | noise_model_sensitivity.json exists; 720 rows; both noise models; both 4-level sets confirmed including gamma=0.01 |
| SENS-03 | 12-03-PLAN | Multi-seed runs (>=5 seeds) for every headline result; mean +/- std reported in every comparison table | SATISFIED | multiseed_summary.json exists; 1266 cells; 870 n==5 headline cells; seed_set [42,43,44,45,46]; all 5 Phase 10/11 headline files consumed |

Note: REQUIREMENTS.md traceability table still shows SENS-01/02/03 as "Pending" (not updated to "Complete"). This is a documentation metadata gap — the artifacts demonstrably satisfy the requirements. Advisory only; not a blocker.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER found in any phase-12 modified file | — | None |

No unreferenced debt markers. No stub patterns in production code paths. No empty return stubs.

**Open review items (from 12-REVIEW.md, status: advisory):**

WR-01 through WR-05 and IN-01 through IN-04 are advisory warnings/info findings declared as "not addressed this pass; tracked as review debt" in the REVIEW.md resolution block. None are blockers for the phase goal or for manuscript use of the generated artifacts. They represent hardening opportunities (replace `assert` with `raise`, add groupby defensive keys, etc.) for future phases.

---

### Human Verification Required

None. All success criteria are verifiable programmatically against the artifact contents, and all have been verified.

---

### Gaps Summary

No gaps. All four success criteria are met:

1. shot_noise_sensitivity.json — 270 rows, correct shot levels, correct schema, seeds, pipelines, dual-scale.
2. noise_model_sensitivity.json — 720 rows, correct noise models, correct 4-level sets for both depolarizing and amplitude-damping (including gamma=0.01), correct schema, seeds, pipelines, dual-scale; per-layer channel insertion documented.
3. multiseed_summary.json — 1266 cells, all 5 Phase 10/11 headline files consumed, data_hash 91e447d4624e25b3 asserted and confirmed, 870 five-seed headline cells with n==5, 168 D-11-09 N/A cells faithfully propagated as null (not a gap per phase instructions).
4. Compute budget — 8m 22s sweep wall time, documented in 12-02-SUMMARY.md.

Cross-cutting constraint satisfied: revision/core/ is byte-untouched (git diff --stat empty, working tree clean).

CR-01 numerical faithfulness independently confirmed: Pipeline-B seed-42 log_return EMD reconciles to 1.53e-16 (within < 1e-6) and OD-scale EMD reconciles to 6.94e-18 (within < 1e-6) against the frozen fidelity_dualscale.json values. CR-02 (seed choices=[42..46] + fail-fast checkpoint guard) and CR-03 (is_complete content-validates metrics.json and samples.npy) are both present in the current code.

---

_Verified: 2026-05-18_
_Verifier: Claude (gsd-verifier)_
