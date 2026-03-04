---
phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign
verified: 2026-03-04T00:00:00Z
status: passed
score: 19/19 requirements verified, 5/5 success criteria verified
re_verification: false
human_verification:
  - test: "Run training for at least 10 epochs and confirm EMD value appears in eval output every 10 epochs"
    expected: "EMD printed as 'EMD: X.XXXXXX' at epoch 10, 20, etc. with numeric value"
    why_human: "Cannot execute the notebook to confirm runtime behavior"
  - test: "Run training for at least 1 step and confirm generator parameter gradients are non-zero"
    expected: "Generator grad value printed in eval block is non-zero (not 0.0000)"
    why_human: "Gradient flow through backprop QNode cannot be verified statically"
  - test: "Run standalone generation (Cell 40) then compare numerically with training eval output for same noise input"
    expected: "Outputs are numerically equivalent when using the same noise tensor — equivalence guaranteed by shared code path"
    why_human: "Numerical equivalence proof requires execution"
---

# Phase 02: WGAN-GP Correctness and Quantum Circuit Redesign — Verification Report

**Phase Goal:** The notebook implements WGAN-GP per Gulrajani et al. (2017) with a well-designed quantum circuit supporting universal approximation via data re-uploading, and a fresh training run produces measurably better distributional fidelity.
**Verified:** 2026-03-04
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | WINDOW_LENGTH is computed as `2 * NUM_QUBITS` in the config cell | VERIFIED | Cell 28: `WINDOW_LENGTH = 2 * NUM_QUBITS` literal; Cell 26 `__init__` asserts `2 * self.num_qubits == self.window_length` |
| 2 | N_CRITIC = 5 and LAMBDA = 10 in the config cell | VERIFIED | Cell 28: `N_CRITIC = 5` and `LAMBDA = 10` confirmed |
| 3 | LR_CRITIC >= LR_GENERATOR (critic learns faster) | VERIFIED | Cell 28: `LR_CRITIC = 8e-5`, `LR_GENERATOR = 3e-5`; 8e-5 >= 3e-5 |
| 4 | normalize() returns a 3-tuple and call site unpacks all three | VERIFIED | Cell 14: `return (data - mu) / sigma, mu, sigma`; Cell 15: `norm_log_delta, mu, sigma = normalize(log_delta)` |
| 5 | GEN_SCALE = 0.1 defined as named constant | VERIFIED | Cell 28: `GEN_SCALE = 0.1` present |
| 6 | Quantum circuit encodes noise via RX at every variational layer (data re-uploading) | VERIFIED | Cell 26: `self.encoding_layer(noise_params)` called once before layer loop AND once inside layer loop after CNOTs (count = 2 for NUM_LAYERS=2); encoding_layer uses `qml.RX` |
| 7 | Redundant parameterized RZ layer before noise encoding removed | VERIFIED | Cell 26: `iqp_params` absent from `count_params()`; no RZ noise encoding step in circuit definition |
| 8 | Both PauliX and PauliZ measurements produce output dim = 2 * NUM_QUBITS | VERIFIED | Cell 26: `qml.PauliX` and `qml.PauliZ` both present in measurements loop |
| 9 | diff_method='backprop' on default.qubit device | VERIFIED | Cell 26: `qml.device("default.qubit", ...)` and `diff_method='backprop'` in QNode creation |
| 10 | Critic has no Dropout, uses LeakyReLU(0.2) | VERIFIED | Cell 26: no `Dropout` string anywhere in class; `negative_slope=0.2` on all 4 activations; no `negative_slope=0.1` |
| 11 | count_params() returns NUM_LAYERS * NUM_QUBITS * 3 + NUM_QUBITS * 2 | VERIFIED | Cell 26: `rotation_params = self.num_layers * self.num_qubits * 3`, `final_params = self.num_qubits * 2`, no IQP term |
| 12 | Training loop uses parameter broadcasting (single QNode call per batch) | VERIFIED | Cell 26: noise shaped `(self.num_qubits, actual_batch_size)` in critic loop; `(self.num_qubits, self.batch_size)` in generator step; no per-sample Python loops |
| 13 | GEN_SCALE applied in all 3 code paths | VERIFIED | Cell 26: `* GEN_SCALE` appears 3 times — critic training, generator training, eval block |
| 14 | One-sided gradient penalty with per-sample alpha | VERIFIED | Cell 26: `torch.clamp(grad_norms - 1, min=0) ** 2`; `alpha = torch.rand(actual_batch_size, 1, 1, ...)`; uses `real_batch.shape[0]` |
| 15 | EMD computed via wasserstein_distance on raw 1D arrays | VERIFIED | Cell 26: `emd = wasserstein_distance(all_real_flat, fake_flat)` on flattened numpy arrays; no histogram binning |
| 16 | Evaluation fires every EVAL_EVERY epochs with all 4 logging categories | VERIFIED | Cell 26: `if (epoch + 1) % EVAL_EVERY == 0 or epoch + 1 == self.num_epochs:`; losses+GP, EMD+stylized facts, inline plot, gradient norms all present |
| 17 | Early stopping monitors EMD with 50 eval-cycle patience and 100-epoch warmup | VERIFIED | Cell 30: `EarlyStopping` class with `best_emd`, `patience=50`, `warmup_epochs=100`; `check()` method called with `emd` value after eval computation |
| 18 | Checkpoint saves mu, sigma alongside model weights | VERIFIED | Cell 30 `_save_checkpoint`: `'mu': mu, 'sigma': sigma` in checkpoint dict; `best_checkpoint.pt` path |
| 19 | Standalone generation uses identical GEN_SCALE and full_denorm_pipeline as training eval | VERIFIED | Cell 40: `* GEN_SCALE`, `full_denorm_pipeline(...)`, noise `[0, 4*pi]`, broadcasting — identical pattern to class eval block |

**Score:** 19/19 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `qgan_pennylane.ipynb` | All phase 2 changes across 4 plans | VERIFIED | 58 cells; config cell (28), class (26), full_denorm_pipeline (23), EarlyStopping (30), standalone generation (40), post-training summary (45) all substantively implemented |
| `qgan_pennylane.ipynb` Cell 14 | normalize() 3-tuple | VERIFIED | Returns `(data - mu) / sigma, mu, sigma` |
| `qgan_pennylane.ipynb` Cell 23 | full_denorm_pipeline function | VERIFIED | `rescale -> lambert_w_transform -> denormalize` pipeline defined as module-level function |
| `qgan_pennylane.ipynb` Cell 26 | qGAN class with redesigned circuit | VERIFIED | Data re-uploading, backprop, one-sided GP, broadcasting, stylized_facts with kurtosis |
| `qgan_pennylane.ipynb` Cell 28 | Config with WGAN-GP paper values | VERIFIED | N_CRITIC=5, LAMBDA=10, LR_CRITIC=8e-5, LR_GENERATOR=3e-5, GEN_SCALE=0.1, EVAL_EVERY=10 |
| `qgan_pennylane.ipynb` Cell 30 | EMD-based EarlyStopping class | VERIFIED | best_emd, warmup_epochs, checkpoint_path='best_checkpoint.pt' |
| `qgan_pennylane.ipynb` Cell 40 | Standalone generation with identical pipeline | VERIFIED | Broadcasting, GEN_SCALE, full_denorm_pipeline, 4*pi noise range |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| normalize() (Cell 14) | normalize call site (Cell 15) | 3-tuple return unpacking | WIRED | `norm_log_delta, mu, sigma = normalize(log_delta)` — exact pattern |
| Config cell (Cell 28) | qGAN instantiation (Cell 28) | WINDOW_LENGTH, N_CRITIC, LAMBDA passed to constructor | WIRED | `qgan = qGAN(NUM_EPOCHS, BATCH_SIZE, WINDOW_LENGTH, N_CRITIC, LAMBDA, NUM_LAYERS, NUM_QUBITS, delta=1)` |
| count_params() | define_generator_circuit idx usage | params_pqc size must match total idx consumed | WIRED | count = NUM_LAYERS*NUM_QUBITS*3 + NUM_QUBITS*2 = 40; circuit consumes exactly 40 via Rot(3 per qubit per layer) + RX+RY(2 per qubit final) |
| define_generator_circuit (2*NUM_QUBITS measurements) | critic Conv1D input | WINDOW_LENGTH = 2 * NUM_QUBITS | WIRED | Runtime assert in `__init__`: `assert 2 * self.num_qubits == self.window_length` |
| define_generator_model | QNode creation | diff_method='backprop' on default.qubit | WIRED | `qml.QNode(self.define_generator_circuit, self.quantum_dev, interface='torch', diff_method='backprop')` |
| EMD computation (eval block) | early_stopper.check() | emd value triggers patience/improvement | WIRED | `if early_stopper.check(epoch, emd, self, mu, sigma)` immediately after `emd = wasserstein_distance(...)` |
| full_denorm_pipeline (Cell 23) | training eval block (Cell 26) | called with same args as standalone | WIRED | `fake_denorm = full_denorm_pipeline(gen_output, preprocessed_data, mu, sigma, self.delta)` |
| full_denorm_pipeline (Cell 23) | standalone generation (Cell 40) | same function, same args | WIRED | `fake_original = full_denorm_pipeline(gen_output, transformed_norm_log_delta, mu, sigma, qgan.delta)` |
| Adam optimizers | betas=(0, 0.9) per Gulrajani et al. | creation site (Cell 28) | WIRED | `betas=(0.0, 0.9)` on both c_optimizer and g_optimizer |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BUG-02 | 02-03, 02-04 | Generator output scaling (*0.1) applied consistently | SATISFIED | `* GEN_SCALE` in critic training, generator training, eval, and standalone generation |
| BUG-03 | 02-03, 02-04 | Denormalization unified between training eval and standalone | SATISFIED | `full_denorm_pipeline` used in both paths |
| PERF-01 | 02-02 | `diff_method='backprop'` on default.qubit | SATISFIED | QNode definition in Cell 26 |
| PERF-04 | 02-03 | Evaluation every N epochs, not every epoch | SATISFIED | `(epoch + 1) % EVAL_EVERY == 0` conditional in train_qgan |
| PERF-05 | 02-03 | Parameter broadcasting for batch quantum execution | SATISFIED | noise shape `(num_qubits, batch_size)` for single QNode call |
| WGAN-01 | 02-01 | N_CRITIC = 5 | SATISFIED | Cell 28 config |
| WGAN-02 | 02-01 | LAMBDA = 10 | SATISFIED | Cell 28 config |
| WGAN-03 | 02-02 | Dropout removed from critic | SATISFIED | No Dropout string in Cell 26 |
| WGAN-04 | 02-03 | EMD on raw samples via wasserstein_distance | SATISFIED | `wasserstein_distance(all_real_flat, fake_flat)` on 1D arrays |
| WGAN-05 | 02-03 | Dynamic histogram bins derived from data range | SATISFIED | `hist_bin_edges = np.linspace(all_real_flat.min(), all_real_flat.max(), ...)` in class; no hardcoded -0.05 bins in Cell 26 |
| WGAN-06 | 02-04 | Early stopping monitors EMD, not critic loss | SATISFIED | `EarlyStopping` tracks `best_emd`; `check()` receives `emd` value |
| WGAN-07 | 02-01 | LR_CRITIC >= LR_GENERATOR | SATISFIED | LR_CRITIC=8e-5, LR_GENERATOR=3e-5 |
| WGAN-08 | 02-03 | Stylized facts audited (ACF, vol clustering, leverage, kurtosis) | SATISFIED | `stylized_facts` method: ACF, abs-ACF, leverage effect, excess kurtosis at stitched and window levels |
| QC-01 | 02-02 | Redundant IQP RZ gate removed | SATISFIED | No `iqp_params` in `count_params()`; no parameterized RZ before noise encoding |
| QC-02 | 02-02 | Data re-uploading — noise re-encoded between variational layers | SATISFIED | `self.encoding_layer(noise_params)` called inside the layer loop after CNOTs |
| QC-03 | 02-02 | PauliX measurements added alongside PauliZ | SATISFIED | Both `qml.PauliX` and `qml.PauliZ` in measurements list |
| QC-04 | 02-02, 02-03 | Noise range expanded to [0, 4pi] | SATISFIED | All 3 `np.random.uniform` calls in class use `0, 4 * np.pi`; Cell 40 likewise |
| QC-05 | 02-01 | WINDOW_LENGTH = 2 * NUM_QUBITS computed automatically | SATISFIED | Cell 28 config |
| QUAL-06 | 02-01 | normalize() returns (normalized_data, mu, sigma) tuple | SATISFIED | Cell 14 return statement |

**Orphaned requirements (in roadmap but no plan claims them):** None

---

### Anti-Patterns Found

| File | Location | Pattern | Severity | Impact |
|------|----------|---------|----------|--------|
| `qgan_pennylane.ipynb` | Cell 57 | `'d'` — single character dead code variable | Info | Leftover from pre-Phase-2 debug code; not part of any execution path. Phase 3 (QUAL-03) scheduled cleanup. |
| `qgan_pennylane.ipynb` | Cell 54 | `np.linspace(-0.05, 0.05, num=50)` hardcoded bins for visualization with generated data | Info | Legacy post-training comparison cell. WGAN-05 requirement scope was "within the class" — Cell 26 uses dynamic bins. This is a duplicate visualization cell targeted by Phase 3 (QUAL-08). Does not affect EMD computation or training. |
| `qgan_pennylane.ipynb` | Cell 10 | `np.linspace(-0.05, 0.05, num=50)` hardcoded bins for original data EDA | Info | Pre-training exploratory analysis cell, not modified by Phase 2. Visualization only, no ML impact. |

No blockers. No warnings. Three informational items, all scoped to Phase 3 cleanup.

---

### Human Verification Required

#### 1. Generator Gradient Flow via Backprop

**Test:** Run at least one training step with the new circuit (execute Cells 1-33 sequentially in a fresh kernel).
**Expected:** The gradient norm printed at eval time shows a non-zero value for "Generator grad" (e.g., `Generator grad: 0.1234`). If it prints `0.0000`, backprop through the QNode is not working.
**Why human:** Static analysis cannot confirm gradient flow — requires actual execution of the backprop-enabled QNode.

#### 2. Evaluation Fires Every EVAL_EVERY Epochs

**Test:** Run training with `NUM_EPOCHS = 30` and `EVAL_EVERY = 10`. Observe console output.
**Expected:** Eval headers appear at epochs 10, 20, and 30 — not at every epoch. No eval output at epochs 1-9 or 11-19.
**Why human:** The conditional `(epoch + 1) % EVAL_EVERY == 0` is structurally verified but runtime behavior requires execution.

#### 3. Standalone Generation Produces Numerically Equivalent Output to Training Eval

**Test:** During training, capture `gen_output` from the eval block. Then run Cell 40 with the exact same `noise` tensor.
**Expected:** Both outputs are identical to floating-point precision (same `* GEN_SCALE`, same `full_denorm_pipeline` call).
**Why human:** The code paths share identical structure (verified statically), but actual numerical equivalence requires runtime comparison.

---

### Notes on Cell Index Shifts

The SUMMARY files for plans 02-01 and 02-02 reference "Cell 25" as the qGAN class and "Cell 27" as config. After plan 02-03 inserted `full_denorm_pipeline` as a new Cell 23, indices shifted by one. The actual notebook has:

- Cell 23: `full_denorm_pipeline`
- Cell 26: qGAN class (was Cell 25)
- Cell 28: config (was Cell 27)
- Cell 30: EarlyStopping (was Cell 29)

All verifications above used the **actual** cell indices confirmed from direct inspection, not plan-documented indices.

---

### Gaps Summary

No gaps found. All 19 requirements pass verification. All 5 ROADMAP success criteria are satisfied by the implementation:

1. WINDOW_LENGTH = 2 * NUM_QUBITS with runtime assertion — SC1
2. N_CRITIC=5, LAMBDA=10, no Dropout, n_critic loop — SC2
3. wasserstein_distance on raw 1D, EMD fed to early_stopper — SC3
4. default.qubit with backprop — SC4
5. GEN_SCALE + full_denorm_pipeline used identically in both paths — SC5

The three informational anti-patterns (Cells 57, 54, 10) are scoped to Phase 3 cleanup (QUAL-03, QUAL-08) and do not block Phase 2 goal achievement.

---

_Verified: 2026-03-04_
_Verifier: Claude (gsd-verifier)_
