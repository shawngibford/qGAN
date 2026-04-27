---
phase: 08-core-module-extraction
verified: 2026-04-27T17:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 1
overrides:
  - must_have: "INFRA-01: revision/core/ package includes models/classical_wgan.py and models/vae.py"
    reason: "08-CONTEXT.md (locked) explicitly defers classical_wgan.py and vae.py to Phase 10 (Classical Baselines). The REQUIREMENTS.md INFRA-01 text was written before the CONTEXT.md locked the phase boundary. The scope clarification is the authoritative constraint for Phase 8."
    accepted_by: "phase-scope-clarification (08-CONTEXT.md)"
    accepted_at: "2026-04-23T00:00:00Z"
---

# Phase 8: Core Module Extraction Verification Report

**Phase Goal:** `revision/core/` package exists and is a drop-in replacement for inline notebook logic, so every downstream v2.0 phase imports from a single verified codebase
**Verified:** 2026-04-27T17:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `revision/core` package is importable as a Python module | VERIFIED | Live import `from revision.core import data, eval, training; from revision.core.models import quantum, critic` exits 0 with "imports OK" |
| 2 | All five core submodules exist and contain working implementations (zero NotImplementedError) | VERIFIED | `grep -rn "NotImplementedError" revision/core/` returns no output; all five files read as substantive implementations |
| 3 | HPO hyperparameter constants match v1.1 Phase 4 values | VERIFIED | N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, NOISE_HIGH=4π all assert-pass programmatically |
| 4 | `QuantumGenerator` PQC parameter count matches notebook (75 for 5 qubits, 4 layers) | VERIFIED | `gen.count_params() == 75` confirmed live; formula: 5 + 4×15 + 10 = 75 |
| 5 | `train_wgan_gp` signature preserves all HPO defaults and three extension hooks are no-ops at defaults | VERIFIED | inspect.signature confirms seed=42, spectral_loss_weight=0.0, callback=None, n_critic=9, lambda_gp=2.16, lr_critic=1.8046e-05, lr_generator=6.9173e-05; body contains torch.manual_seed, betas=(0.0, 0.9), spectral hook conditional |
| 6 | `revision/results/parity_check.json` exists with `"pass": true` and all deltas within tolerance | VERIFIED | JSON confirmed: pass=true; EMD delta=0.0 (≤1e-4), mean delta=0.0 (≤1e-6), std delta=0.0 (≤1e-6), kurtosis delta=0.0 (≤1e-6); seed=42, checkpoint=best_checkpoint_par_conditioned.pt |
| 7 | `revision/01_parity_check.ipynb` imports from revision.core and runs both inline and module paths | VERIFIED | Notebook contains all four required strings: `from revision.core`, `best_checkpoint`, `wasserstein_distance`, `compute_emd`; 8 cells |

**Score:** 7/7 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | `models/classical_wgan.py` — classical WGAN-GP generator module | Phase 10 | REQUIREMENTS.md BASE-01/02/03 mapped to Phase 10 (Classical Baselines); 08-CONTEXT.md explicitly defers this |
| 2 | `models/vae.py` — VAE baseline module | Phase 10 | REQUIREMENTS.md BASE-02 mapped to Phase 10; 08-CONTEXT.md explicitly defers this |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `revision/__init__.py` | Top-level package marker | VERIFIED | Exists, contains docstring |
| `revision/core/__init__.py` | Re-exports + HPO constants | VERIFIED | Contains all 17 constants including N_CRITIC=9, LAMBDA=2.16, NOISE_HIGH=4π; imports data, eval, training, models |
| `revision/core/models/__init__.py` | models subpackage marker | VERIFIED | Imports quantum, critic; `__all__` defined |
| `revision/core/data.py` | Functional data pipeline | VERIFIED | 257 lines; load_and_preprocess, normalize, denormalize, compute_log_delta, lambert_w_transform, inverse_lambert_w_transform, rolling_window, rescale, full_denorm_pipeline, find_optimal_lambert_delta — all implemented |
| `revision/core/eval.py` | Functional evaluation metrics | VERIFIED | 164 lines; compute_emd (raw-sample wasserstein_distance), compute_moments, compute_acf, compute_dtw, compute_jsd, compute_psd, full_metric_suite — all implemented |
| `revision/core/training.py` | WGAN-GP training loop | VERIFIED | 483 lines; train_wgan_gp, compute_gradient_penalty, EarlyStopping, _ESAdapter, _spectral_psd_loss — all implemented; no NotImplementedError |
| `revision/core/models/quantum.py` | QuantumGenerator PQC class | VERIFIED | QuantumGenerator with diff_method="backprop", qml.Rot, qml.CNOT, qml.Hadamard, qml.RX, qml.RY; count_params(5,4)=75; batched forward produces (12,10) tensor; gradient flows |
| `revision/core/models/critic.py` | 1D-CNN Critic class | VERIFIED | Three Conv1d layers (1→64→128→128, k=10, p=5), AdaptiveAvgPool1d, Linear(128→32), Dropout(p=dropout_rate), Linear(32→1); cast to double(); configurable dropout |
| `revision/results/.gitkeep` | Directory tracking | VERIFIED | Exists |
| `revision/docs/.gitkeep` | Directory tracking | VERIFIED | Exists |
| `revision/01_parity_check.ipynb` | Parity check notebook | VERIFIED | 8 cells; all four required strings present |
| `revision/results/parity_check.json` | Parity artifact with pass=true | VERIFIED | pass=true; all deltas=0.0 (exact match, well within tolerance) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `revision/core/__init__.py` | `revision/core/data, eval, training` | module-level imports | VERIFIED | `from revision.core import data, eval, training` present at line 35 |
| `revision/core/__init__.py` | `revision/core/models` | module-level import | VERIFIED | `from revision.core import models` present at line 36 |
| `revision/core/models/__init__.py` | `revision/core/models/quantum, critic` | module-level imports | VERIFIED | `from revision.core.models import quantum, critic` at line 2 |
| `revision/core/data.py` | `./data.csv` | pd.read_csv in load_and_preprocess | VERIFIED | `pd.read_csv(str(csv_path))` at data.py line 211 |
| `revision/core/eval.py` | `scipy.stats.wasserstein_distance` | raw-sample call (v1.0 decision) | VERIFIED | `from scipy.stats import wasserstein_distance` + called as `wasserstein_distance(real, fake)` (no histograms inside compute_emd) |
| `revision/core/models/quantum.py` | pennylane | `qml.QNode(diff_method='backprop')` | VERIFIED | `diff_method=diff_method` with default `"backprop"`; qml.Rot, qml.CNOT, qml.Hadamard all present |
| `revision/core/training.py` | `revision.core.eval.compute_emd` | eval-loop import | VERIFIED | `from revision.core.eval import compute_emd, compute_moments` at line 226 |
| `revision/core/training.py` | `compute_gradient_penalty` | call inside critic loop | VERIFIED | `gp = compute_gradient_penalty(...)` at line 295 |
| `revision/01_parity_check.ipynb` | `revision/core (data, eval, models)` | module imports in notebook | VERIFIED | `from revision.core` string confirmed in code cells |
| `revision/01_parity_check.ipynb` | `best_checkpoint_par_conditioned.pt` | checkpoint load for fixed-state pass | VERIFIED | `best_checkpoint` string confirmed in code cells |
| `revision/01_parity_check.ipynb` | `revision/results/parity_check.json` | json.dump at notebook end | VERIFIED | parity_check.json exists and has correct content |

### Data-Flow Trace (Level 4)

These are Python modules (not UI components); data-flow trace applies to the load_and_preprocess pipeline and parity artifact.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `revision/core/data.py::load_and_preprocess` | OD, windowed_data | `pd.read_csv(csv_path)` + pipeline | Yes — reads actual CSV, returns windowed_data shape (384, 10) confirmed in 08-02 SUMMARY | FLOWING |
| `revision/core/eval.py::compute_emd` | float result | `scipy.stats.wasserstein_distance(real, fake)` on raw arrays | Yes — calls wasserstein_distance on actual sample arrays | FLOWING |
| `revision/results/parity_check.json` | all metric fields | executed notebook producing real checkpoint forward-pass outputs | Yes — pass=true with zero deltas confirmed live | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Package imports cleanly | `qgan_env/bin/python3 -c "from revision.core import data, eval, training; from revision.core.models import quantum, critic; print('imports OK')"` | "imports OK" | PASS |
| Parity JSON is valid with pass=true and zero deltas | python3 json.load + assert checks | pass=true, all deltas=0.0 | PASS |
| HPO constants match v1.1 Phase 4 | assert N_CRITIC==9 etc. | all pass | PASS |
| QuantumGenerator count_params(5,4)==75 | `gen.count_params() == 75` | 75 | PASS |
| QuantumGenerator unbatched forward produces (10,) tensor | `out.numel() == 10` | shape (10,) | PASS |
| QuantumGenerator batched forward produces (12,10) tensor | `out_batch.shape == (12, 10)` | shape (12, 10) | PASS |
| Gradient flows to params_pqc | `gen.params_pqc.grad is not None` after backward | grad exists | PASS |
| Critic forward on (12,1,10) float64 produces (12,1) | `y.shape == (12, 1)` | (12, 1) | PASS |
| Critic dropout_rate configurable | p=0.5 propagates to nn.Dropout module | confirmed | PASS |
| train_wgan_gp signature defaults match HPO values | inspect.signature asserts | all pass | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| INFRA-01 | 08-01, 08-02, 08-03, 08-04 | revision/core/ package with importable modules | SATISFIED (with override for classical_wgan.py + vae.py deferred to Phase 10) | All five in-scope modules exist, are importable, and contain working implementations; classical_wgan.py and vae.py explicitly deferred per 08-CONTEXT.md |
| INFRA-02 | 08-05 | Extracted modules reproduce notebook behavior within tolerance | SATISFIED | parity_check.json: pass=true; EMD delta=0.0, mean delta=0.0, std delta=0.0, kurtosis delta=0.0 (all exact zero, well within 1e-4/1e-6 tolerances) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `revision/core/training.py` | 358-361 | `acf_avg`, `vol_avg`, `lev_avg` appended as `0.0` placeholders per eval epoch | INFO | Intentional — documented in 08-04 SUMMARY; full stylized_facts() pipeline is invoked by 08-05 on final generator state, not per-epoch; dict shape preserved for downstream consumers |
| `revision/core/eval.py` | 116 | Comment "not implemented in notebook's metric section" in compute_psd docstring | INFO | Non-blocking — compute_psd function body is fully implemented (scipy.signal.welch); the comment refers to the notebook not having this metric in its final evaluation section, not to missing code |

No BLOCKER or WARNING anti-patterns found. Both flagged items are informational only and documented decisions.

### Human Verification Required

None. All observable truths are verifiable programmatically and have been confirmed via live execution.

### Gaps Summary

No gaps. All seven must-have truths are VERIFIED. The two items from INFRA-01 raw text (classical_wgan.py, vae.py) are confirmed deferred to Phase 10 per the locked 08-CONTEXT.md and are tracked in the Deferred Items section above with an applied override.

The phase goal is fully achieved: `revision/core/` is a drop-in replacement for inline notebook logic with zero numerical drift (exact parity, delta=0.0 on all metrics), and every downstream v2.0 phase can safely import from this single verified codebase.

---

_Verified: 2026-04-27T17:00:00Z_
_Verifier: Claude (gsd-verifier)_
