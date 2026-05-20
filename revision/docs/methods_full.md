# Methods (Full)

> **Source of truth.** Every numeric literal in this document resolves to one of
> `revision/results/methods_full.json`, `revision/results/model_info.json`,
> `revision/results/canonical_config_lock.json`,
> `revision/results/default_75_config_lock.json`,
> `revision/results/v1_config_lock.json`,
> `revision/results/v2_config_lock.json`,
> `revision/results/v3_config_lock.json`,
> `revision/results/classical_architectures.json`, or
> `revision/results/framework_versions.json`.
>
> Gate (executable success-criterion-5):
> `./qgan_env/bin/python revision/verify_number_provenance.py --target revision/docs/methods_full.md`.
> LaTeX equation strings are rendered verbatim from
> `methods_full.json.buckets.2_models.<model>.training_objective.equation_latex*`;
> never authored inline. The two documented manuscript contradictions
> (`default_75` vs `iqp_sel_55`; `dtype_params` vs `dtype_samples`) are
> explicitly addressed in § 6.

This document consolidates the Methods section for the revised manuscript
(PAPER-08 + PAPER-09). It is regenerated whenever the upstream JSON corpus
changes; the number-provenance gate is the executable contract that no
literal may bypass.

---

## 1. Dataset

| Quantity | Value | Source |
|---|---|---|
| Raw CSV rows | 778 | `methods_full.json` buckets.1_dataset.raw_csv_rows / `model_info.json` dataset.raw_csv_rows |
| OD rows (after fillna/dropna) | 778 | `methods_full.json` buckets.1_dataset.od_rows_after_fillna_dropna |
| Log-return rows | 777 | `methods_full.json` buckets.1_dataset.log_return_rows |
| Window length | 10 | `methods_full.json` buckets.1_dataset.window_length |
| Window stride | 2 | `methods_full.json` buckets.1_dataset.window_stride |
| Rolling windows | 384 | `methods_full.json` buckets.1_dataset.rolling_windows |
| Independent campaigns | 1 | `methods_full.json` buckets.1_dataset.independent_campaigns |
| Train windows | 384 | `methods_full.json` buckets.1_dataset.train_windows |
| Val windows | 0 | `methods_full.json` buckets.1_dataset.val_windows |
| Test windows | 0 | `methods_full.json` buckets.1_dataset.test_windows |

Window-count derivation: `windows = (log_return_rows - window_length) // window_stride + 1`,
applied to the single-campaign OD trajectory and gated against the accepted
sweep's `n_real_windows` in `model_info.json`. The single-campaign limitation
means all reported metric variance is over training-seed variation, not over
independent experimental campaigns.

---

## 2. Models

Ten model entries are reported (PAPER-08 Methods completeness): five quantum
(headline + reproduction + V1/V2/V3 matched-budget ansatz variants), three
classical WGAN-GP baselines, and two non-adversarial baselines (VAE, AR(p)).
The shared WGAN-GP critic architecture is recorded once in § 2.k.

### 2.a. `iqp_sel_55_headline` (frozen-checkpoint paper headline)

| Property | Value | Source |
|---|---|---|
| Family | adversarial-quantum | `methods_full.json` buckets.2_models.iqp_sel_55_headline.family |
| num_qubits | 5 | `canonical_config_lock.json` decomposition.num_qubits |
| num_layers | 3 | `canonical_config_lock.json` decomposition.num_layers |
| Topology | range | `canonical_config_lock.json` decomposition.gate_layout.entangler |
| Encoding | IQP (Hadamard + RZ per qubit) | `methods_full.json` buckets.2_models.iqp_sel_55_headline.architecture.encoding |
| Variational block | SEL (Rot(phi,theta,omega) per qubit per layer) | `methods_full.json` buckets.2_models.iqp_sel_55_headline.architecture.variational_block |
| Final rotation | RX_only | `canonical_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 55 | `canonical_config_lock.json` param_count |
| Checkpoint epoch | 1969 | `canonical_config_lock.json` checkpoint_epoch |

Training objective: **WGAN-GP (Wasserstein with two-sided gradient penalty)**
— `methods_full.json` buckets.2_models.iqp_sel_55_headline.training_objective
cites `revision/core/training.py:364` (critic loss),
`revision/core/training.py:385` (generator loss), and
`revision/core/training.py:72` (gradient penalty).

```latex
L_C = \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)] - \mathbb{E}_{x \sim P_r}[C(x)] + \lambda_{gp}\, \mathbb{E}_{\hat x \sim P_{\hat x}}\big[(\|\nabla_{\hat x} C(\hat x)\|_2 - 1)^2\big]
```

```latex
L_G = - \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)]
```

### 2.b. `iqp_sel_55_repro` (matched 2000-epoch reproduction)

Same architecture as `iqp_sel_55_headline` (n_params = 55, num_qubits = 5,
num_layers = 3, topology = range, final_rotation = RX_only —
`canonical_config_lock.json`); distinguished only by `source` ==
`matched2000_reproduction` and `tier` == `T2`. No `checkpoint_epoch` (the
reproduction trains the full 2000-epoch budget — see § 3 — rather than
freezing the best-EMD checkpoint). Training objective: WGAN-GP, citations
and LaTeX as in § 2.a.

### 2.c. `V1` (matched-budget ansatz)

| Property | Value | Source |
|---|---|---|
| Family | adversarial-quantum | `methods_full.json` buckets.2_models.V1.family |
| num_qubits | 5 | `v1_config_lock.json` decomposition.num_qubits |
| num_layers | 4 | `v1_config_lock.json` decomposition.num_layers |
| Topology | range | `v1_config_lock.json` decomposition.gate_layout.entangler |
| Final rotation | RX_plus_RY | `v1_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 75 | `v1_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a — single source of
truth in `methods_full.json` buckets.2_models.V1.training_objective).

### 2.d. `V2` (matched-budget ansatz, deeper)

| Property | Value | Source |
|---|---|---|
| num_qubits | 5 | `v2_config_lock.json` decomposition.num_qubits |
| num_layers | 8 | `v2_config_lock.json` decomposition.num_layers |
| Topology | range | `v2_config_lock.json` decomposition.gate_layout.entangler |
| Final rotation | RX_plus_RY | `v2_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 135 | `v2_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.e. `V3` (matched-budget ansatz, linear entangler)

| Property | Value | Source |
|---|---|---|
| num_qubits | 5 | `v3_config_lock.json` decomposition.num_qubits |
| num_layers | 4 | `v3_config_lock.json` decomposition.num_layers |
| Topology | linear | `v3_config_lock.json` topology |
| Final rotation | RX_plus_RY | `v3_config_lock.json` decomposition.gate_layout.final_rotation |
| n_params | 75 | `v3_config_lock.json` param_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.f. `wgan_mlp` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Linear | in_features=5, out_features=4, bias, activation=tanh, 24 params | `classical_architectures.json` models.wgan_mlp.generator[0] |
| Linear | in_features=4, out_features=10, bias, 50 params | `classical_architectures.json` models.wgan_mlp.generator[1] |
| **Total** | **74 params** | `classical_architectures.json` models.wgan_mlp.total_params / `model_info.json` parameter_count |

Carve strategy: a single `nn.Parameter` (`params_pqc`) is sliced into
per-layer weight/bias views applied via `torch.nn.functional` (mirrors the
quantum generator's design — RESEARCH § Pitfall 1). Training objective:
WGAN-GP (same equations as § 2.a; same shared critic — § 2.k).

### 2.g. `wgan_cnn` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Reshape | (B, 5) → (B, 1, 5), parameter-free | `classical_architectures.json` models.wgan_cnn.generator[0] |
| ConvTranspose1d | in_channels=1, out_channels=9, kernel_size=6, stride=1, bias, LeakyReLU(0.1), 63 params | `classical_architectures.json` models.wgan_cnn.generator[1] |
| Conv1d | in_channels=9, out_channels=1, kernel_size=1, bias, 10 params | `classical_architectures.json` models.wgan_cnn.generator[2] |
| **Total** | **73 params** | `classical_architectures.json` models.wgan_cnn.total_params / `model_info.json` parameter_count |

Training objective: WGAN-GP (same equations as § 2.a).

### 2.h. `wgan_lstm` (classical baseline)

| Layer | Spec | Source |
|---|---|---|
| Reshape | (B, 5) → (B, seq=3, input=2), parameter-free tile/pad | `classical_architectures.json` models.wgan_lstm.generator[0] |
| LSTM_functional | input_size=2, hidden_size=2, num_layers=1, bias, gate order i,f,g,o, 48 params | `classical_architectures.json` models.wgan_lstm.generator[1] |
| Linear | in_features=2, out_features=10, bias, 30 params | `classical_architectures.json` models.wgan_lstm.generator[2] |
| **Total** | **78 params** | `classical_architectures.json` models.wgan_lstm.total_params / `model_info.json` parameter_count |

The LSTM cell is carved from `params_pqc` by hand (NOT `nn.LSTM`) so the
single-`nn.Parameter` invariant matches the quantum generator's design.
Training objective: WGAN-GP (same equations as § 2.a).

### 2.i. `vae` (non-adversarial baseline)

| Block | Layer | Spec | Source |
|---|---|---|---|
| Encoder | Linear `enc` | in=10, out=16, bias | `classical_architectures.json` models.vae.encoder |
| Encoder | Linear `fc_mu` | in=16, out=4, bias | `classical_architectures.json` models.vae.encoder |
| Encoder | Linear `fc_logvar` | in=16, out=4, bias | `classical_architectures.json` models.vae.encoder |
| Decoder | Linear `dec_h` | in=4, out=16, bias | `classical_architectures.json` models.vae.decoder |
| Decoder | Linear `dec_out` | in=16, out=10, bias | `classical_architectures.json` models.vae.decoder |
| Latent dim | 4 | `classical_architectures.json` models.vae.latent_dim |
| Hidden dim | 16 | `classical_architectures.json` models.vae.hidden_dim |
| Window | 10 | `classical_architectures.json` models.vae.window |
| **Total** | **562 params** | `classical_architectures.json` models.vae.total_params / `model_info.json` parameter_count |

Training objective: **ELBO** (variational autoencoder). The ELBO training
loop lives in `revision/run_baselines.py` (D-10-13) — **not** in
`revision/core/training.py` (which is WGAN-GP only).

```latex
\mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
```

### 2.j. `ar` (non-adversarial baseline)

| Property | Value | Source |
|---|---|---|
| Type | AR(p) | `classical_architectures.json` models.ar.type |
| Order p | 2 | `classical_architectures.json` models.ar.order_p |
| Fit method | np.linalg.lstsq (closed-form) | `classical_architectures.json` models.ar.fit_method |
| **Total** | **3 params** | `classical_architectures.json` models.ar.total_params / `model_info.json` parameter_count |

`ARBaseline` is a plain Python class — **not** an `nn.Module`. Its fit is
closed-form (no training loop) at
`revision/core/models/nonadversarial.py:139-158`.

```latex
x_t = \sum_{k=1}^{p} \phi_k\, x_{t-k} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2);\quad \hat\phi = \arg\min_\phi \|X\phi - y\|_2^2
```

### 2.k. Shared WGAN-GP critic

One critic instance is constructed per training run and is shared across
`iqp_sel_55_repro` + V1 / V2 / V3 + `wgan_mlp` / `wgan_cnn` / `wgan_lstm`.
Architecture (verbatim `classical_architectures.json` models.shared_critic):
Conv1d(1, 64, k=10, s=1, p=5) + LeakyReLU(0.1); Conv1d(64, 128, k=10, s=1,
p=5) + LeakyReLU(0.1); Conv1d(128, 128, k=10, s=1, p=5) + LeakyReLU(0.1);
AdaptiveAvgPool1d(output_size=1); Flatten; Linear(128, 32) + LeakyReLU(0.1)
+ Dropout(p=0.2); Linear(32, 1). The critic is cast to `torch.float64` at
`revision/core/models/critic.py:67`.

---

## 3. Training

| Property | Value | Source |
|---|---|---|
| Optimizer | Adam | `methods_full.json` buckets.3_training.optimizer |
| Optimizer betas | [0.0, 0.9] | `methods_full.json` buckets.3_training.optimizer_betas |
| lr_critic | 1.8046e-05 | `methods_full.json` buckets.3_training.lr_critic |
| lr_generator | 6.9173e-05 | `methods_full.json` buckets.3_training.lr_generator |
| n_critic | 9 | `methods_full.json` buckets.3_training.n_critic |
| lambda_gp | 2.16 | `methods_full.json` buckets.3_training.lambda_gp |
| batch_size | 12 | `methods_full.json` buckets.3_training.batch_size |
| epochs | 2000 | `methods_full.json` buckets.3_training.epochs |
| early_stopping (reproduction) | OFF (full 2000ep, D-14-13) | `methods_full.json` buckets.3_training.early_stopping |
| Seeds | [42, 43, 44, 45, 46] | `methods_full.json` buckets.3_training.seeds |

The matched-budget reproduction (`iqp_sel_55_repro` + V1 / V2 / V3 + the
three classical WGAN-GP baselines) trains the full 2000 epochs with
early-stopping OFF (D-14-13). The frozen-checkpoint headline
(`iqp_sel_55_headline`) is the best-EMD checkpoint at epoch 1969 from the
original early-stopping-ON campaign and is consumed as-is from
`revision/results/canonical_config_lock.json`. The headline's original
training-time learning rates (recorded as breadcrumbs in
`revision/results/canonical_recovery.json`) were lr_critic = 3e-05,
lr_generator = 8e-05 — see `model_info.json` models[0].lr_critic /
lr_generator (`iqp_sel_55_headline` row).

VAE uses a single Adam(lr=1e-3) ELBO loop and AR(p) uses closed-form
`np.linalg.lstsq` — neither participates in the WGAN-GP table above (see
§ 2.i / § 2.j).

---

## 4. Hardware & Software

| Property | Value | Source |
|---|---|---|
| Compute device | cpu | `methods_full.json` buckets.4_hardware_software.device |
| PennyLane backend | default.qubit | `methods_full.json` buckets.4_hardware_software.pennylane_device |
| Differentiation | backprop | `methods_full.json` buckets.4_hardware_software.diff_method |
| **Param dtype (`dtype_params`)** | torch.float32 | `methods_full.json` buckets.4_hardware_software.dtype_params |
| **Sample dtype (`dtype_samples`)** | torch.float64 (CPU/CUDA path; falls back to torch.float32 on Apple MPS) | `methods_full.json` buckets.4_hardware_software.dtype_samples |
| Python version | 3.11.14 | `framework_versions.json` python_version |
| Platform | macOS-26.0.1-arm64-arm-64bit | `framework_versions.json` platform |

### 4.1. Framework versions (exact installed pin)

| Package | Version | Source |
|---|---|---|
| pennylane | 0.43.0 | `framework_versions.json` packages.pennylane |
| torch | 2.9.0 | `framework_versions.json` packages.torch |
| numpy | 2.3.4 | `framework_versions.json` packages.numpy |
| scipy | 1.16.2 | `framework_versions.json` packages.scipy |
| matplotlib | 3.10.7 | `framework_versions.json` packages.matplotlib |
| PyYAML | 6.0.3 | `framework_versions.json` packages.PyYAML |

`revision/requirements.txt` carries the `>=` constraint set; the table above
is the exact installed pin captured at methods-doc emit time via
`importlib.metadata.version(...)`. Re-emit on environment change.

**`dtype_params` ≠ `dtype_samples`.** Trainable parameters live in
`torch.float32` for every model (every `nn.Parameter` in
`revision/core/models/classical.py` is constructed with
`dtype=torch.float32` — see `revision/core/models/classical.py:78` for
`WGANMLPGenerator`, with analogous lines for `WGANCNNGenerator` /
`WGANLSTMGenerator`; the quantum generator's `params_pqc` is also
`torch.float32`). Generated samples are cast to `torch.float64` at
generation time on the CPU/CUDA path (`revision/core/training.py:268`,
`compute_dtype = torch.float32 if device.type == "mps" else torch.float64`,
then the cast at `revision/core/training.py:347`,
`generated_samples.to(compute_dtype) * 0.1`). The Apple MPS path falls back
to `torch.float32` because MPS does not implement float64 — explicit
`compute_dtype` branching at `revision/core/training.py:259-268`. The two
fields are DISTINCT and MUST NOT be conflated — see § 6(b).

---

## 5. Reproducibility

| Property | Value | Source |
|---|---|---|
| data_hash | 91e447d4624e25b3 | `methods_full.json` buckets.5_reproducibility.data_hash |
| seed_set | [42, 43, 44, 45, 46] | `methods_full.json` buckets.5_reproducibility.seed_set |

### 5.1. Determinism contract

| Seed call | Source |
|---|---|
| `torch.manual_seed(seed)` | `revision/core/training.py:245` |
| `np.random.seed(seed)` | `revision/core/training.py:246` |
| `random.seed(seed)` | `revision/core/training.py:247` |
| `torch.cuda.manual_seed_all(seed)` (when CUDA available) | `revision/core/training.py:248-249` |

Seeds are set ONCE at the top of `train_wgan_gp` before optimizer/data
construction (`revision/core/training.py:244-249`). The same seed produces
bit-identical training trajectories on the same device/dtype path.

### 5.2. Exact rerun command (verbatim)

The block below is the verbatim module docstring of
`revision/run_matched2000.py` (lines 1-80) — preserved character-for-character
inside `methods_full.json.buckets.5_reproducibility.rerun_command_template`
and rendered as-is here:

```
Phase 14 Tier-2/3 driver — train one (model, seed) at the MATCHED 2000-epoch
budget behind a device-honest strict accept gate.

One process per invocation. Idempotent — re-running the same ``--model X
--seed Z`` overwrites ``runs/<model>/<seed>/`` cleanly.

Usage
-----
    ./qgan_env/bin/python -m revision.run_matched2000 \
        --model {iqp_sel_55_repro|V1|V2|V3|wgan_mlp|wgan_cnn|wgan_lstm|vae|ar} \
        --seed N [--epochs 2000] \
        [--out-root revision/results/matched2000] [--csv-path ./data.csv]

    # strict accept gate (per-artifact, explicit-raise; exits non-zero on
    # rejection):
    ./qgan_env/bin/python -m revision.run_matched2000 \
        --accept --model M --seed N [--out-root revision/results/matched2000]
```

Source: `revision/run_matched2000.py:1-80` (module docstring; preserved
verbatim — never paraphrased). The fully verbatim docstring (including the
strict-gate accept-criterion text and the Pitfall-5 / D-10-24 worker-pool
note) lives in `methods_full.json.buckets.5_reproducibility.rerun_command_template`.

---

## 6. Addressing the documented contradictions

### (a) `default_75` vs `iqp_sel_55` — which is "default"?

Both are valid, production quantum circuits used in the manuscript. They
serve different purposes and live in different config-lock JSONs:

- **`default_75`** — `revision/results/default_75_config_lock.json`:
  num_qubits = 5, num_layers = 4, topology = range, n_params = 75, final
  rotation RX+RY per qubit. This is the byte-frozen v1.0/v1.1 baseline
  encoded in `revision/core/__init__.py` (`NUM_QUBITS=5`, `NUM_LAYERS=4`)
  and the underlying circuit_id for the matched-budget ansatz variants V1,
  V2, V3 (see `v1_config_lock.json`, `v2_config_lock.json`,
  `v3_config_lock.json`).
- **`iqp_sel_55`** — `revision/results/canonical_config_lock.json`:
  num_qubits = 5, num_layers = 3, topology = range, n_params = 55, final
  rotation RX only per qubit, `checkpoint_epoch = 1969`. This is the
  **canonical paper circuit** recovered from the **frozen checkpoint** at
  epoch 1969 (D-14-01) and is the manuscript's headline quantum entrant in
  every cross-model comparison.

Neither is "wrong" — `default_75` underlies the matched-budget ansatz study
(V1 / V2 / V3) that demonstrates the headline `iqp_sel_55` is competitive
at a strictly lower parameter count. Where the manuscript talks about
**"the quantum entrant"** it always means `iqp_sel_55`; where it talks
about the **matched-budget ansatz comparison** it always means a V-variant
of `default_75`.

### (b) `dtype_params` vs `dtype_samples` — distinct, never the same

Trainable parameters live in `torch.float32` for every model:
- Classical: every `nn.Parameter` in `revision/core/models/classical.py` is
  constructed with `dtype=torch.float32`
  (`revision/core/models/classical.py:78` for `WGANMLPGenerator`, with
  analogous lines for `WGANCNNGenerator` / `WGANLSTMGenerator`).
- Quantum: the `QuantumGenerator.params_pqc` `nn.Parameter` is also
  `torch.float32`.

Generated samples are cast to `torch.float64` at generation time on the
CPU/CUDA path. The cast is done explicitly at
`revision/core/training.py:268` (`compute_dtype = torch.float32 if
device.type == "mps" else torch.float64`) and applied at
`revision/core/training.py:347` (`generated_samples.to(compute_dtype) *
0.1`). The Apple MPS path falls back to `torch.float32` because MPS does
not implement float64 — see the explicit `compute_dtype` branching at
`revision/core/training.py:259-268`. Samples are cast to float64 to match
the float64 critic (`revision/core/models/critic.py:67` `.double()`).

These are TWO DISTINCT fields in
`methods_full.json.buckets.4_hardware_software` (`dtype_params` and
`dtype_samples`) and a previous internal document conflated them; this
Methods doc resolves the conflation. They must be reported as two separate
rows in any future revision table — see § 4 of this document.

---

## 7. Provenance footer

- Every numeric literal in this document resolves to one of:
  `revision/results/methods_full.json`,
  `revision/results/model_info.json`,
  `revision/results/canonical_config_lock.json`,
  `revision/results/default_75_config_lock.json`,
  `revision/results/v1_config_lock.json`,
  `revision/results/v2_config_lock.json`,
  `revision/results/v3_config_lock.json`,
  `revision/results/classical_architectures.json`,
  `revision/results/framework_versions.json`.
- Gate command:
  `./qgan_env/bin/python revision/verify_number_provenance.py --target revision/docs/methods_full.md`.
- LaTeX equation strings are rendered verbatim from
  `revision/results/methods_full.json` (never authored inline in this
  document).
- The `rerun_command_template` block in § 5.2 is sliced verbatim from
  `revision/run_matched2000.py:1-80` (module docstring) by
  `revision/run_methods_full.py` — never paraphrased.
