---
phase: 08
plan: 03
requirements: [INFRA-01]
status: complete
completed: 2026-04-23
---

# Phase 8 Plan 03: Quantum Generator + Critic Extraction — Summary

Extracted PQC generator and 1D-CNN critic from `qgan_pennylane.ipynb` cell 26 into `revision/core/models/quantum.py` and `revision/core/models/critic.py`. Identity-preserving refactor; v1.0/v1.1 decisions locked in.

## Extracted Classes

**`QuantumGenerator(num_qubits=5, num_layers=4, window_length=10, diff_method="backprop")`**
- `count_params(5, 4) == 75` verified (IQP 5 + 4×15 Rot + 10 RX/RY final).
- Init scale `0.5` (cell 26 literal `torch.randn(...) * 0.5`).
- Circuit: Hadamard → IQP RZ encoding → noise `encoding_layer` → strongly-entangled Rot+range-CNOT layers → final RX/RY → PauliX+PauliZ on every wire.
- Output: 1D noise `(num_qubits,)` → `(window_length,)`; batched noise `(num_qubits, batch)` → `(batch, window_length)` via `torch.stack(list(results)).T`.
- `par_light` kwarg is a reserved no-op hook — final v1.1 notebook (`RUN_NAME = "unconditioned_wgan"`, cell 65) runs unconditioned, so cell 26 has no PAR_LIGHT wiring to port. Kwarg preserved for future conditioning phases.

**`Critic(window_length=10, dropout_rate=0.2)`**
- Conv1d(1→64, k=10, s=1, p=5) + LeakyReLU(0.1)
- Conv1d(64→128, k=10, s=1, p=5) + LeakyReLU(0.1)
- Conv1d(128→128, k=10, s=1, p=5) + LeakyReLU(0.1)
- AdaptiveAvgPool1d(1) + Flatten
- Linear(128→32) + LeakyReLU(0.1) + **Dropout(p=dropout_rate)** (only Dropout in the net; v1.1 Phase 7 configurability verified)
- Linear(32→1)
- Model cast to `double()` to match notebook float64 tensors.

## Critic Input Shape (for plan 08-04)

**Expected input: `(batch_size, 1, window_length)`, `dtype=torch.float64`.**

Source: cell 26 training loop (`real_batch_tensor.reshape(batch_size, 1, window_length).double()` and `fake_batch_tensor.reshape(batch_size, 1, window_length)`). Plan 08-04 must reshape generator output from `(batch, window_length)` → `(batch, 1, window_length)` and cast to `double` before feeding the critic (notebook also multiplies generator output by `0.1` — that scalar scaling lives in the training loop, not the model).

Output: `(batch_size, 1)` scalar score per sample.

## Decisions Preserved

- `diff_method="backprop"` default (v1.1 Phase 5).
- IQP RZ encoding; no RZ→Rot substitution.
- Range-CNOT: `r = (layer % (num_qubits - 1)) + 1`; `target = (q + r) % num_qubits`.
- Measurement order `(<X_0>, <Z_0>, <X_1>, <Z_1>, …)` matches cell 26.
- Noise range `[0, 4π]` is consumed upstream (training loop); `NOISE_HIGH = 4π` already in `revision/core/__init__.py`.
- Dropout rate plumbed through constructor kwarg; exactly one `nn.Dropout` module.

## Verification

- `QuantumGenerator(5, 4, 10).count_params() == 75` ✓
- Unbatched forward: 1D noise → `(10,)` numel 10 ✓
- Batched forward: `(5, 12)` noise → `(12, 10)` ✓
- Gradient flows to `params_pqc` ✓
- `Critic` accepts `(12, 1, 10)` float64 → returns `(12, 1)` ✓
- `dropout_rate=0.5` propagates to module ✓
- End-to-end gen→reshape→critic→`backward` flows gradient to PQC params ✓

## Commits

- `7d5cef4` feat(08-03): implement QuantumGenerator with data re-uploading PQC
- `91dbbf1` feat(08-03): implement Critic 1D-CNN with configurable dropout

## Self-Check: PASSED

- `revision/core/models/quantum.py` exists ✓
- `revision/core/models/critic.py` exists ✓
- Commits `7d5cef4`, `91dbbf1` in git log ✓
