# Training Protocol — QWGAN-GP (matched 2000ep, 55-param IQP:SEL)

> **Source of truth:** every numerical constant below is rendered FROM `revision/results/model_info.json` by `revision/run_model_info.py` — there are NO hand-typed numbers and NO `core/__init__.py:NN` line citations. Re-run the emitter to update; `revision/verify_number_provenance.py` is the executable gate that proves every literal here resolves to a `revision/results/*.json` value (success criterion 5).

This protocol describes the matched-budget 2000-epoch training run for the canonical 55-param IQP:SEL quantum generator (`source=matched2000_reproduction`) — the quantum entrant in every cross-model comparison (D-14-04). The frozen-checkpoint headline (`source=frozen_checkpoint_epoch_1969`) is a SEPARATE record in `model_info.json` (D-14-10).

## Optimizer & Schedule

| Constant | Value | Source |
|----------|-------|--------|
| `N_CRITIC` | 9 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (n_critic) |
| `LAMBDA` (gradient penalty coeff) | 2.16 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (lambda_gp) |
| `LR_CRITIC` | 1.8046e-05 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (lr_critic) |
| `LR_GENERATOR` | 6.9173e-05 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (lr_generator) |
| Optimizer | Adam, betas=(0.0, 0.9) — WGAN-GP | `model_info.json` models[] kind=quantum source=matched2000_reproduction (optimizer, optimizer_betas) |
| `NUM_EPOCHS` | 2000 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (epochs) |
| `BATCH_SIZE` | 12 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (batch_size) |

Early-stopping state for the matched-budget run: OFF (full 2000ep, D-14-13) (`model_info.json` models[] kind=quantum source=matched2000_reproduction, early_stop). The frozen-checkpoint headline instead uses the best-EMD checkpoint from the original EarlyStopping-enabled campaign (see `model_info.json` iqp_sel_55_headline record).

## Quantum Circuit

| Property | Value | Source |
|----------|-------|--------|
| Backend | default.qubit (analytic statevector) | `model_info.json` models[] kind=quantum source=matched2000_reproduction (pennylane_device) |
| Differentiation | backprop | `model_info.json` models[] kind=quantum source=matched2000_reproduction (diff_method) |
| `NUM_QUBITS` | 5 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (num_qubits) |
| `NUM_LAYERS` | 3 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (num_layers) |
| `WINDOW_LENGTH` | 10 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (window_length) |
| `circuit_id` | iqp_sel_55 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (circuit_id) |
| Entangler topology | range | `model_info.json` models[] kind=quantum source=matched2000_reproduction (topology) |
| PQC trainable parameter count | 55 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (parameter_count) |
| Compute device | cpu | `model_info.json` models[] kind=quantum source=matched2000_reproduction (device) |
| dtype_params | torch.float32 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (dtype_params); see methods_full.md §4.b |
| dtype_samples | torch.float64 | `model_info.json` models[] kind=quantum source=matched2000_reproduction (dtype_samples); see methods_full.md §4.b |
| Backend assertion | PASSED | `model_info.json` models[] kind=quantum source=matched2000_reproduction (backend_assertion) |

## Reproducibility

| Property | Value | Source |
|----------|-------|--------|
| Seed set | [42, 43, 44, 45, 46] | `model_info.json` models[] kind=quantum source=matched2000_reproduction (seeds) |
| Training windows | 384 | `model_info.json` dataset.rolling_windows |
| `data_hash` | `91e447d4624e25b3` | `model_info.json` data_hash (cross-artifact gate) |

All seeds in [42, 43, 44, 45, 46] share the identical config (the strict accept gate D-14-13 enforced this); the data_hash `91e447d4624e25b3` is identical across every consumed 2000ep artifact (cross-artifact explicit-raise gate, run_multiseed_rollup.py:86-92 idiom).

