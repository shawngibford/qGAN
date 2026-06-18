---
phase: 13-architecture-introspection
verified: 2026-05-19T00:00:00Z
status: human_needed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification: null
human_verification:
  - test: "Inspect training_progression figures (PNG) for visual correctness"
    expected: "4-row × 5-column grid showing recognisable distribution shape evolution for quantum and all three classical variants side-by-side; quantum row visually distinct from classical"
    why_human: "Figure rendering verified (non-empty png/pdf exists, JSON source valid), but histogram/KDE visual correctness and 'side-by-side' layout quality cannot be confirmed by grep"
  - test: "Inspect entanglement_trajectory figure for bipartition annotation"
    expected: "Figure labels show the verbatim string '{0,1}|{2,3,4}' and reference bounds ln4 / 0.25 / 1.0 are annotated"
    why_human: "Cannot verify matplotlib text annotations from file contents; only that the figure file is non-empty and the JSON source contains the bipartition string"
  - test: "Inspect param_trajectory figure for angle histogram correctness"
    expected: "Panel (a) shows a rising/stable L2-norm curve; panel (b) shows 75-parameter angle distributions across 5 epochs with visually distinct profiles"
    why_human: "Cannot verify subplot layout and histogram fidelity programmatically"
  - test: "Confirm REQUIREMENTS.md traceability table updated to mark ARCH-01, ARCH-02, INTRO-01, INTRO-02, INTRO-03 as Complete"
    expected: "All five Phase-13 requirement IDs show 'Complete' in .planning/REQUIREMENTS.md"
    why_human: "REQUIREMENTS.md still shows all five as 'Pending'; updating traceability is a documentation step that the verifier cannot make without human sign-off"
  - test: "Decide disposition of REVIEW BLOCKER CR-01 (--epochs dead-code in run_ansatz.py)"
    expected: "Either (a) fix: thread args.epochs into train_wgan_gp num_epochs=int(epochs), or (b) intentional-lock: remove --epochs flag from driver and sweep to avoid provenance confusion"
    why_human: "The sweep ran at 1000 epochs (default) so the deliverables are correct; but the dead-code knob remains — a future --epochs invocation would silently misrecord config.yaml. Human decides fix-or-remove."
  - test: "Decide disposition of REVIEW BLOCKER CR-02 (missing V1 npz schema guard in run_ansatz_comparison.py)"
    expected: "Either (a) add early-fail validation guard per the review's suggested code, or (b) accept as-is with a documented comment that the V1 key contract is verified by construction"
    why_human: "The aggregator ran successfully (V1 npz keys matched exactly), but the absence of a guard is a latent crash path if the V1 bundle is ever regenerated with different internals. Human decides guard-or-accept."
---

# Phase 13: Architecture-Introspection Verification Report

**Phase Goal:** Ansatz choice is justified empirically (2-3 variants compared) and the "black-box" feel (R2-6) is addressed with training-progression, parameter-trajectory, and entanglement-entropy figures — giving reviewers both "why this circuit?" and "what is it learning?" evidence
**Verified:** 2026-05-19
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 2-3 alternate ansatz variants (V1/V2/V3 varying depth/topology) implemented in quantum.py and selectable via config | VERIFIED | `QuantumGenerator(num_layers=4, topology='range')` → 75 params; `(num_layers=8, topology='range')` → 135 params; `(num_layers=4, topology='linear')` → 75 params. Confirmed live via import. `topology` kwarg in `__init__` with ValueError guard verified. All 30 tests green. |
| 2 | Ansatz comparison table (identical training budget, multi-seed, full metric suite) written to results/ansatz_comparison.json | VERIFIED | File exists (75,177 bytes). 300 rows: V1×100, V2×100, V3×100. V2/V3 each have 5 seeds (42-46). Both scales (log_return + OD). 9 required fields per row. V1 reuse noted "D-13-01, no recompute". `full_metric_suite` called for all rows. Schema test green (30 passing). |
| 3 | Training-progression figure shows generated distribution at epochs {0,250,500,750,1000} for quantum generator and classical WGAN-GP side-by-side | VERIFIED | `training_progression.json` contains 4 targets (quantum, wgan_mlp, wgan_cnn, wgan_lstm), each with `epochs=[0,250,500,750,1000]` and `samples[5][12]` of real float data. `training_progression.png` (71,084 bytes) and `.pdf` (36,504 bytes) exist and non-empty. Figure reads from companion JSON (no training code in renderer). |
| 4 | PQC parameter-trajectory (norms + angle histograms) and entanglement-entropy/purity trajectory saved as figure artifacts, each with underlying data in JSON | VERIFIED | `param_trajectory.json`: `param_norm[5]`, `param_angles[5][75]`, metadata V1/depth=4/topology=range. `entanglement_trajectory.json`: `vn_entropy[5]` (values 1.17-1.25 all < ln4=1.386), `purity[5]` (values 0.32-0.35 all in [0.25,1]), metadata `bipartition="{0,1}|{2,3,4}"`. Both `.png` + `.pdf` exist and non-empty. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `core/models/quantum.py` | topology-selectable ansatz + introspect() | VERIFIED | 317 lines. `topology` in `__init__` (line 52), `self.topology` stored (line 75), `_TOPOLOGIES` constant, `if self.topology == "range"` literal first branch (range block at lines 183, 247), `elif self.topology == "linear"`, `qml.vn_entropy(wires=[0, 1])`, `qml.purity(wires=[0, 1])`, `introspect()` method, `INTROSPECT_BIPARTITION = ((0,1),(2,3,4))`. |
| `core/training.py` | CR-01 torch.fft.rfft + CR-02 map_location restore | VERIFIED | `torch.fft.rfft` at lines 516-517 inside `_spectral_psd_loss`. No `from scipy.signal import welch`. `if spectral_loss_weight > 0.0` guard at line 376. `map_location` at line 178 inside `_load_checkpoint`. |
| `results/ansatz_comparison.json` | ARCH-02 comparison (V1 reuse + V2/V3 new, dual-scale) | VERIFIED | 300 rows. V1/V2/V3 variants with depth=4/8/4, topology=range/range/linear, param_count=75/135/75. V2+V3 each have 5 seeds × both scales. `full_metric_suite` UNCHANGED used. |
| `figures/training_progression.json` | INTRO-01 companion JSON | VERIFIED | 4 targets, 5 epochs each, real sample arrays. metadata pipeline=B, seed=42. |
| `figures/param_trajectory.json` | INTRO-02 companion JSON | VERIFIED | `param_norm[5]`, `param_angles[5][75]`, metadata variant=V1. |
| `figures/entanglement_trajectory.json` | INTRO-03 companion JSON | VERIFIED | `vn_entropy[5]`, `purity[5]`, metadata `bipartition="{0,1}|{2,3,4}"`. INTRO-03 bounds confirmed. |
| `figures/training_progression.{png,pdf}` | INTRO-01 figure | VERIFIED | 71,084 bytes PNG / 36,504 bytes PDF. Non-empty. |
| `figures/param_trajectory.{png,pdf}` | INTRO-02 figure | VERIFIED | 87,511 bytes PNG / 21,855 bytes PDF. Non-empty. |
| `figures/entanglement_trajectory.{png,pdf}` | INTRO-03 figure | VERIFIED | 84,226 bytes PNG / 28,946 bytes PDF. Non-empty. |
| `run_ansatz.py` | single (variant,seed) driver | VERIFIED | `QuantumGenerator(` with `topology=`, `train_wgan_gp(` with `num_epochs=1000`, no `early_stopper=`, `choices=["V2","V3"]`. |
| `run_ansatz_sweep.sh` | 10-run sweep (V2/V3 × 5 seeds) | VERIFIED | VARIANTS/SEEDS/EPOCHS=1000 defined. xargs -P 2. No multiprocessing.Pool. |
| `run_ansatz_comparison.py` | ARCH-02 aggregator | VERIFIED | `full_metric_suite` imported. `transform_ablation/runs/B` path for V1 reuse. V1 no-recompute note. |
| `run_introspect.py` | callback-snapshot driver (4 targets) | VERIFIED | `SNAP = {0, 250, 500, 750, 999}`, 999→1000 relabel, `hasattr(gen_model, "introspect")` guard, `callback=cb` passed to `train_wgan_gp`. |
| `run_introspect_figures.py` | render-only matplotlib renderer | VERIFIED | `matplotlib.use("Agg")` at line 26. `savefig` present. Reads `training_progression.json`, `param_trajectory.json`, `entanglement_trajectory.json`. No `train_wgan_gp` / `QuantumGenerator(` in source. |
| `tests/test_ansatz_variants.py` | ARCH-01 param-count + byte-unchanged regression | VERIFIED | Present; 30 tests pass. |
| `tests/test_entropy_purity.py` | INTRO-03 entropy/purity bounds | VERIFIED | Present; 30 tests pass. |
| `tests/test_cr01_spectral_grad.py` | CR-01 grad regression | VERIFIED | Present; 30 tests pass. |
| `tests/test_cr02_es_restore.py` | CR-02 device/dtype regression | VERIFIED | Present; 30 tests pass. |
| `tests/test_ansatz_json_schema.py` | ARCH-02 schema regression | VERIFIED | Present; 30 tests pass. |
| `tests/test_introspect_callback.py` | INTRO-01/02/03 callback regression | VERIFIED | Present; 30 tests pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `quantum.py::generator_circuit` | topology switch | `if self.topology == "range"` literal first branch | WIRED | `range_param = (layer % (self.num_qubits - 1)) + 1` found at lines 184 and 248 (both inside `if self.topology == "range"`). Default path byte-identical. |
| `quantum.py::introspect` | `_introspect_qnode` | `qml.vn_entropy(wires=[0, 1])`, `qml.purity(wires=[0, 1])` | WIRED | Both measurements present at lines 267-268; `_introspect_qnode` built in `__init__` line 102. |
| `run_ansatz.py` | `QuantumGenerator` | `QuantumGenerator(num_layers=depth, topology=topology)` | WIRED | Confirmed in source. V2=(8,range), V3=(4,linear) mapping verified. |
| `run_ansatz_comparison.py` | `full_metric_suite` | `full_metric_suite` UNCHANGED (D-10-20) | WIRED | Import at line 67; called for both OD and log_return scales. |
| `run_ansatz_comparison.py` | `transform_ablation/runs/B/{42..46}` | V1 reuse NO recompute | WIRED | Path resolution at line 133; V1 source string contains "no recompute". |
| `run_introspect.py::snapshot_cb` | `train_wgan_gp` callback hook | `callback=cb` kwarg | WIRED | `SNAP = {0, 250, 500, 750, 999}` at line 74; `callback=cb` passed at line 256. |
| `run_introspect.py::snapshot_cb` | `generator.introspect` | `hasattr` guard | WIRED | `hasattr(gen_model, "introspect")` at line 150. |
| `run_introspect_figures.py` | `training_progression.json` | `json.load` | WIRED | Constants at lines 32-34; `savefig` confirmed present. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `training_progression.json` | `samples[4 targets][5 epochs][N windows]` | 4 instrumented 1000-epoch runs via `run_introspect.py` | Yes — 12 real float windows per snapshot confirmed | FLOWING |
| `param_trajectory.json` | `param_norm[5]`, `param_angles[5][75]` | V1 quantum run `params_pqc.detach().cpu().numpy()` at each SNAP | Yes — values 4.397-4.414 (real, non-trivial) | FLOWING |
| `entanglement_trajectory.json` | `vn_entropy[5]`, `purity[5]` | `generator.introspect(noise[:,0])` at each SNAP | Yes — values 1.17-1.25 / 0.32-0.35 within bounds | FLOWING |
| `ansatz_comparison.json` | `rows[300]` | V1: frozen transform_ablation samples re-scored; V2/V3: 10 new runs re-scored | Yes — 100% non-zero values, multi-metric, dual-scale | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| QuantumGenerator param counts | `from revision.core.models.quantum import QuantumGenerator; g.count_params()` for V1/V2/V3 | 75 / 135 / 75 | PASS |
| Invalid topology raises ValueError | `QuantumGenerator(topology='star')` | `ValueError: Unknown topology 'star'; expected one of ('range', 'linear')` | PASS |
| Test suite all green | `./qgan_env/bin/python -m pytest tests/ -q` | 30 passed in 3.08s | PASS |
| INTRO-03 bounds on real data | entanglement_trajectory.json vn_entropy/purity values | vn_entropy all < ln4=1.386; purity all in [0.25,1] | PASS |
| ansatz_comparison.json schema | V1/V2/V3 variants, 300 rows, 9-field rows, dual scale | All checks pass | PASS |
| 4 intermediates have 5 snapshots at correct epochs | json.load _introspect_*.json | epochs=[0,250,500,750,1000] for all 4 targets | PASS |
| Figure files non-empty | `ls -la *.png *.pdf` | 6 files, 21KB-87KB each | PASS |

### Probe Execution

Step 7c: SKIPPED — no `scripts/*/tests/probe-*.sh` files exist in this project. Phase-level verification is via pytest suite (30 passed) and direct JSON/artifact checks above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ARCH-01 | 13-01, 13-02 | 2-3 alternate ansatz variants (depth {4,6,8} and/or entanglement topology) | SATISFIED | V1(4/range/75), V2(8/range/135), V3(4/linear/75) constructible via `topology`+`num_layers`; all param counts verified; all 10 V2/V3 sweep runs produced data |
| ARCH-02 | 13-02 | Ansatz comparison table (identical training budget, multi-seed, all metrics) | SATISFIED | `ansatz_comparison.json` — 300 rows, V1/V2/V3 × {42-46} × dual-scale, `full_metric_suite` UNCHANGED, schema test green |
| INTRO-01 | 13-03, 13-04 | Training-progression figure — generated distribution at {0,N/4,N/2,3N/4,N} for quantum + classical WGAN-GP | SATISFIED | `training_progression.json` + `.png`/`.pdf` — 4 targets × 5 epochs × real sample arrays |
| INTRO-02 | 13-03, 13-04 | PQC parameter trajectory (norms, angle histograms across epochs) | SATISFIED | `param_trajectory.json` + `.png`/`.pdf` — `param_norm[5]` + `param_angles[5][75]` |
| INTRO-03 | 13-01, 13-03, 13-04 | Entanglement-entropy or state-purity trajectory | SATISFIED | `entanglement_trajectory.json` + `.png`/`.pdf` — `vn_entropy[5]` + `purity[5]`, bipartition metadata, bounds hold on real data |

**Note:** REQUIREMENTS.md traceability table still shows all five IDs as "Pending" (Phase 13). Updating the table is a documentation step included in human verification items.

### Cross-Phase Reproducibility Invariant

The verifier assessed the `core/` byte-behavior-unchanged invariant vs commit b7c84d3 (Phases 8-12 reproducibility):

- `quantum.py`: The `range` CNOT block is the LITERAL first branch (`if self.topology == "range"`). `range_param = (layer % (self.num_qubits - 1)) + 1` confirmed present at both lines 184 and 248. `topology="range"` is the constructor default. `test_default_forward_byte_unchanged` pins this with atol=1e-12. **Invariant PRESERVED.**
- `training.py`: `_spectral_psd_loss` and `_load_checkpoint` were rewritten, but both sit behind runtime guards (`if spectral_loss_weight > 0.0` and `early_stopper is not None`) that are off at default values. Default headline runs (Phase 8-12) do not use early stopping (D-13-05) or spectral loss (D-13-06). **Invariant PRESERVED for the Phases 8-12 forward path.** (Caveat per REVIEW WR-01: any prior early-stopped run now follows a different `_load_checkpoint` code path — acceptable because Phase-13 headline runs have early-stop OFF; documented in REVIEW for future reproductions.)

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `core/training.py` | 191 | `checkpoint = ckpt` alias dead style (IN-01) | Info | None — cosmetic only |
| `run_ansatz.py` | 257 | `num_epochs=1000` hardcoded, ignores `args.epochs` parameter (REVIEW BLOCKER CR-01) | Warning | No data corruption in as-produced deliverables (sweep used default 1000); latent provenance risk for non-default `--epochs` invocations |
| `run_ansatz_comparison.py` | 148-154 | V1 `inverse_kwargs.npz` read with no key-presence guard (REVIEW BLOCKER CR-02) | Warning | Aggregator ran successfully (V1 npz keys match exactly); latent crash risk if V1 bundle is ever regenerated with different internals |

No `TBD`, `FIXME`, or `XXX` debt markers found in any phase-modified file.

### Review Blockers Assessment (13-REVIEW.md)

The 13-REVIEW.md identifies 2 BLOCKER findings in the driver layer. Assessment against as-produced deliverables:

**BLOCKER CR-01 — `run_ansatz.py --epochs` dead-code:** The sweep was run with `EPOCHS=1000` (shell default). The hardcoded `num_epochs=1000` in `_train_wgan` exactly matches. `config.yaml` records `epochs: 1000` truthfully. `ansatz_comparison.json` has 300 rows from real 1000-epoch training. **The phase-goal deliverable is correct.** The dead-code path (a mismatch) would only materialize with `--epochs <non-1000>`. This is a WARNING for future use, not a BLOCKER against the phase goal.

**BLOCKER CR-02 — unvalidated V1 npz read:** The V1 `inverse_kwargs.npz` files have keys `['mu', 'od_starts', 'r_max', 'r_min', 'sigma']` — exactly what the aggregator reads. The aggregator ran successfully and produced 100 V1 rows with real non-zero metric values. The missing validation guard is a latent crash path but it was NOT triggered. **The phase-goal deliverable is correct.** This is a WARNING for defensive coding, not a BLOCKER against the phase goal as delivered.

### Human Verification Required

#### 1. Figure Visual Correctness — training_progression

**Test:** Open `figures/training_progression.png` and confirm it shows a 4×5 grid (4 targets × 5 epochs) with distribution histograms/KDEs for quantum and the 3 classical models side-by-side; quantum row visually distinct from classical.
**Expected:** Each cell shows a meaningful distribution shape; the quantum row shows non-trivial structure distinct from the classical rows; axes are properly labeled.
**Why human:** matplotlib rendering is confirmed (non-empty file, no training code in renderer), but histogram/KDE visual correctness cannot be verified by grep.

#### 2. Figure Visual Correctness — entanglement_trajectory bipartition annotation

**Test:** Open `figures/entanglement_trajectory.png` and confirm the bipartition string `{0,1}|{2,3,4}` is annotated, and reference bounds (ln4 ≈ 1.386, 0.25, 1.0) are shown.
**Expected:** Two panels (entropy + purity vs epoch); bipartition label visible; reference lines drawn at ln4 and 0.25.
**Why human:** The JSON source contains the bipartition string; cannot verify matplotlib `ax.text()` / `ax.axhline()` output without viewing the file.

#### 3. Figure Visual Correctness — param_trajectory

**Test:** Open `figures/param_trajectory.png` and confirm two panels: (a) L2-norm of PQC params vs epoch and (b) angle-distribution histograms for each of the 5 snapshot epochs.
**Expected:** Panel (a) shows norm values ~4.4 rising slightly over training; panel (b) shows 75-parameter angle distributions that change across epochs.
**Why human:** Cannot verify subplot layout and histogram per-epoch content from file contents alone.

#### 4. REQUIREMENTS.md traceability update

**Test:** Update `.planning/REQUIREMENTS.md` traceability table to mark ARCH-01, ARCH-02, INTRO-01, INTRO-02, INTRO-03 as Complete (Phase 13).
**Expected:** All five IDs show `Complete` in the table.
**Why human:** This is a documentation decision that should be made by the human after confirming the deliverables meet the reviewer's bar.

#### 5. REVIEW BLOCKER CR-01 disposition (--epochs dead-code)

**Test:** In `run_ansatz.py` line 257: either (a) change `num_epochs=1000` to `num_epochs=int(epochs)` and update the `train_protocol_notes` string accordingly, or (b) remove the `--epochs` argparse argument from both `run_ansatz.py` and `run_ansatz_sweep.sh` to eliminate the misleading knob.
**Expected:** No silent discrepancy between `config.yaml epochs` and actual training budget on any invocation.
**Why human:** Both options are valid; the choice depends on whether the 1000-epoch budget is an intentional Phase-13 lock (D-13 decision) or a parameterizable value.

#### 6. REVIEW BLOCKER CR-02 disposition (V1 npz schema guard)

**Test:** In `run_ansatz_comparison.py` around line 148: either (a) add the key-validation guard from the REVIEW's suggested code block, or (b) add an inline comment documenting that the V1 key contract is verified by construction (the 09.1/10 driver and this aggregator share the same `_save_inverse_kwargs` function, making a mismatch structurally impossible absent a schema migration).
**Expected:** Future maintainers have a clear signal about whether the schema contract is validated or trusted by construction.
**Why human:** The choice between a runtime guard and a documented assertion is an architectural/maintenance preference.

## Gaps Summary

No gaps blocking the phase goal. All 4 success criteria are verified in the codebase:
- V1/V2/V3 quantum generator variants are implemented and selectable (75/135/75 params confirmed live)
- `ansatz_comparison.json` exists with 300 rows, real data, correct schema, V1 reused per D-13-01
- Training-progression JSON + figures exist with all 4 targets × 5 epochs of real data
- Parameter-trajectory and entanglement-entropy JSON + figures exist with real run data and correct bipartition metadata

The 2 REVIEW BLOCKERs are driver-layer code quality issues that did NOT corrupt the phase deliverables (sweep ran at the intended 1000 epochs; V1 npz keys matched exactly). They require human disposition but do not block the phase goal as stated.

Status is `human_needed` due to figure visual correctness checks and the requirement/disposition decisions above.

---

_Verified: 2026-05-19_
_Verifier: Claude (gsd-verifier)_
