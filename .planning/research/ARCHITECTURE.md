# Architecture Research: v1.1 Post-HPO Integration Architecture

**Domain:** Hybrid quantum-classical GAN for bioprocess time series synthesis
**Researched:** 2026-03-13
**Confidence:** HIGH (based on direct codebase analysis + verified PennyLane/PyTorch docs)

---

## Current Architecture Snapshot

```
qGAN(nn.Module)
  __init__(num_epochs, batch_size, window_length, n_critic, gp, num_layers, num_qubits, delta, lambda_acf)
    |
    +-- quantum_dev: qml.device("default.qubit", wires=num_qubits)
    +-- params_pqc: nn.Parameter(torch.randn(num_params) * 0.5)  [75 params for 5q/4L]
    +-- critic: nn.Sequential(Conv1d stack)  [2-channel input: log-returns + PAR_LIGHT]
    +-- generator: qml.QNode(define_generator_circuit, quantum_dev, interface='torch', diff_method='parameter-shift')
    |
    +-- train_qgan(gan_data, original_data, preprocessed_data, num_elements, early_stopper)
    |     +-- _train_one_epoch(gan_data_list, par_data_list, original_data, preprocessed_data, epoch)
    |           +-- Critic training: n_critic iterations, per-sample Python loop
    |           +-- Generator training: 1 iteration, per-sample Python loop
    |           +-- Evaluation: every eval_every epochs, stylized facts + EMD
    |
    +-- compile_QGAN(c_optimizer, g_optimizer)
    +-- stylized_facts(original_data, fake_original)
    +-- diff_acf_lag1(x) [static]
    +-- count_params()
```

### Quantum Circuit Data Flow (Current)

```
H gates (all qubits)
    |
IQP encoding: trainable RZ(params_pqc[0..4])
    |
Noise encoding: RZ(noise[0..4])         <-- noise from U[0, 2*pi] (REGRESSION)
    |
PAR_LIGHT encoding: RY(par_light * pi)  <-- compressed 10->5 values, remapped [-1,1]->[0,1]
    |
Strongly entangled layers x4:
    Rot(phi, theta, omega) per qubit     <-- params_pqc[5..64]
    CNOT range-based entangling
    |
Final prep: RX, RY per qubit            <-- params_pqc[65..74]
    |
Measurements: PauliX + PauliZ per qubit = 10 values (WINDOW_LENGTH)
```

### Critic Network (Current)

```
Input: [batch_size, 2, 10]  (channel 0: log-returns, channel 1: PAR_LIGHT)
    |
Conv1d(2->64, k=10, pad=5) -> LeakyReLU(0.1)
Conv1d(64->128, k=10, pad=5) -> LeakyReLU(0.1)
Conv1d(128->128, k=10, pad=5) -> LeakyReLU(0.1)
    |
AdaptiveAvgPool1d(1) -> Flatten
    |
Linear(128->32) -> LeakyReLU(0.1) -> Dropout(0.2)
Linear(32->1)
```

### Training Loop Data Flow (Current -- Per-Sample)

```
Critic training (n_critic iterations):
    for i in range(batch_size):           <-- REGRESSION: per-sample loop
        noise = U[0, 2*pi]               <-- REGRESSION: should be [0, 4*pi]
        par_for_circuit = compress(par_window)
        gen_out = generator(noise, par_for_circuit, params_pqc)  <-- single QNode call
        fake_batch.append(gen_out * 0.1)
    critic_loss = W-distance + lambda*GP

Generator training (1 iteration):
    for i in range(batch_size):           <-- REGRESSION: per-sample loop
        noise = U[0, 2*pi]               <-- REGRESSION: should be [0, 4*pi]
        gen_out = generator(noise, par_circuit, params_pqc)
    gen_loss = -E[D(fake)] + lambda_acf * ACF_penalty
```

---

## Feature Integration Points

### Feature 1: Noise Range Fix ([0, 2*pi] -> [0, 4*pi])

**Type:** Bug fix (regression)
**Integration complexity:** Trivial
**Files modified:** Cell 26 (qGAN class, _train_one_epoch method)

**Exact locations to change (3 sites in _train_one_epoch):**

| Location | Current | Fixed |
|----------|---------|-------|
| Critic loop (line ~1359) | `np.random.uniform(0, 2 * np.pi, ...)` | `np.random.uniform(0, 4 * np.pi, ...)` |
| Generator loop (line ~1436) | `np.random.uniform(0, 2 * np.pi, ...)` | `np.random.uniform(0, 4 * np.pi, ...)` |
| Evaluation loop (line ~1517) | `np.random.uniform(0, 2 * np.pi, ...)` | `np.random.uniform(0, 4 * np.pi, ...)` |

**Additional location outside class (Cell 32, line ~1855):**
- Circuit visualization dummy input: already uses `U[0, 2*pi]` -- should match but is cosmetic

**Data flow impact:** None. Noise values are consumed by `encoding_layer()` via `qml.RZ(noise[i])`, which has period 4*pi. Expanding the range allows the full rotation space.

**Dependency:** None. Can be done first.

**Architectural recommendation:** Extract noise range to a class attribute or constant:
```python
# In __init__:
self.noise_range = 4 * np.pi  # Full RZ rotation period

# In training:
noise_values = np.random.uniform(0, self.noise_range, size=self.num_qubits)
```

---

### Feature 2: Broadcasting Optimization Restoration

**Type:** Performance fix (regression, ~12x speedup)
**Integration complexity:** Medium
**Files modified:** Cell 26 (_train_one_epoch method -- critic loop, generator loop, eval loop)

**Current architecture (per-sample):**
```python
# 3 separate loops, each calling generator once per sample
for i in range(self.batch_size):
    gen_out = self.generator(generator_inputs[i], par_circuits_batch[i], self.params_pqc)
```

**Target architecture (broadcasted):**
```python
# Single QNode call with batched noise and par_light
noise_batch = torch.tensor(
    np.random.uniform(0, 4 * np.pi, size=(self.batch_size, self.num_qubits)),
    dtype=torch.float32
)
par_batch = torch.stack(par_circuits_batch)  # [batch_size, num_qubits]

# Broadcasted call: noise_batch has shape [batch_size, num_qubits]
# PennyLane broadcasts over the first dimension automatically
gen_outputs = self.generator(noise_batch, par_batch, self.params_pqc)
# Returns tuple of batch_size-length tensors (one per measurement)
generated_samples = torch.stack(gen_outputs, dim=-1)  # [batch_size, window_length]
```

**PennyLane broadcasting requirements (HIGH confidence, from docs):**
1. `default.qubit` natively supports parameter broadcasting
2. The broadcasted input has exactly one more axis than the operator expects
3. Multiple broadcasted parameters must have matching batch dimensions
4. Works with `interface='torch'` and `diff_method='parameter-shift'`
5. `params_pqc` is NOT broadcasted (shared across batch) -- only noise and par_light are

**Critical integration detail -- PAR_LIGHT compression must be vectorized:**
```python
# Current: per-sample compression
par_for_circuit = par_window.reshape(self.num_qubits, 2).mean(dim=1).float()
par_for_circuit = (par_for_circuit + 1.0) / 2.0

# Broadcasted: batch compression
par_windows = torch.stack(par_windows_batch)  # [batch_size, window_length]
par_compressed = par_windows.reshape(self.batch_size, self.num_qubits, 2).mean(dim=2).float()
par_compressed = (par_compressed + 1.0) / 2.0  # [batch_size, num_qubits]
```

**Three loops to convert:**

| Loop | Location | Gradient context | Notes |
|------|----------|-----------------|-------|
| Critic training | Inner for-loop in n_critic | `torch.no_grad()` | Easiest -- no gradient flow needed |
| Generator training | for-loop after g_optimizer.zero_grad() | Requires grad flow | Must keep `params_pqc` in graph |
| Evaluation | for-loop in eval block | `torch.no_grad()` | Currently uses `par_zeros` (no conditioning) |

**Output handling change:**
- Current: `generator()` returns tuple of 10 scalars -> `torch.stack(list(gen_out))`
- Broadcasted: returns tuple of 10 tensors each of shape `[batch_size]` -> `torch.stack(gen_outputs, dim=-1)` gives `[batch_size, 10]`

**Dependency:** Should be done AFTER noise range fix (so the fix is included in the refactored loops). Should be done BEFORE spectral loss (faster iteration).

**Architectural risk:** The `isinstance(gen_out, (list, tuple))` checks in the current code handle different output formats. Broadcasting changes the output shape, so all consumers must be updated atomically.

---

### Feature 3: mu/sigma Shadowing Fix

**Type:** Bug fix (non-blocking in linear execution)
**Integration complexity:** Trivial
**Files modified:** Cell 10 (normalization cell) or wherever mu/sigma are computed

**Current issue:** The `mu` and `sigma` variables used in `normalize()` / early stopping are overwritten on re-execution of the normalization cell. This doesn't break linear execution but breaks re-runs.

**Fix approach:** Inline mu/sigma into their usage site, or rename to unique names:
```python
# Option A: unique names
norm_mu = transformed_norm_log_delta.mean()
norm_sigma = transformed_norm_log_delta.std()

# Option B: inline into function call (preferred -- already decided in v1.0)
```

**Dependency:** None. Independent of all other features.

---

### Feature 4: Spectral/PSD Loss

**Type:** New feature (addresses variance collapse)
**Integration complexity:** Medium
**Files modified:** Cell 26 (qGAN class -- new method + _train_one_epoch generator loss section)

**New component -- `diff_psd_loss` method:**
```python
@staticmethod
def diff_psd_loss(fake_batch, real_batch):
    """Differentiable power spectral density mismatch loss.

    Uses torch.fft.rfft which is fully differentiable (autograd-supported).
    Computes log-PSD to prevent large-magnitude frequency bins from
    dominating the loss.

    Args:
        fake_batch: [batch_size, window_length] generated samples
        real_batch: [batch_size, window_length] real samples
    Returns:
        Scalar loss (mean squared log-PSD difference)
    """
    # Compute one-sided FFT
    fake_fft = torch.fft.rfft(fake_batch, dim=-1)
    real_fft = torch.fft.rfft(real_batch, dim=-1)

    # Power spectral density (magnitude squared)
    fake_psd = torch.abs(fake_fft) ** 2 + 1e-8
    real_psd = torch.abs(real_fft) ** 2 + 1e-8

    # Log-PSD for scale-invariant comparison
    log_fake_psd = torch.log(fake_psd)
    log_real_psd = torch.log(real_psd)

    # Mean squared error across frequencies and batch
    return torch.mean((log_fake_psd - log_real_psd) ** 2)
```

**Integration into generator loss (in _train_one_epoch):**
```python
# Current generator loss:
generator_loss = generator_loss_wgan + self.lambda_acf * acf_penalty

# New generator loss:
psd_penalty = self.diff_psd_loss(generated_samples, real_batch_for_psd)
generator_loss = generator_loss_wgan + self.lambda_acf * acf_penalty + self.lambda_psd * psd_penalty
```

**New __init__ parameter:**
```python
def __init__(self, ..., lambda_psd=0.1):
    self.lambda_psd = lambda_psd
```

**Data flow change in generator training section:**
- Currently, real windows are sampled only for ACF comparison
- PSD loss needs the same real windows -- reuse `real_batch_acf` (rename to `real_batch_aux`)
- Generated samples `generated_samples` already have shape `[batch_size, window_length]`

**Why torch.fft.rfft (HIGH confidence, from PyTorch docs):**
- Fully differentiable via autograd (since PyTorch 1.7)
- GPU-accelerated
- `rfft` returns compact one-sided representation for real signals
- Window length of 10 gives 6 frequency bins (including DC) -- sufficient for mid-frequency structure

**Window length consideration:** With WINDOW_LENGTH=10, we get 6 frequency bins from rfft. This is enough to capture the volatility structure (bins 2-4 represent mid-frequency variations). The PSD loss is most valuable for frequencies that the ACF lag-1 penalty misses.

**Dependency:** Logically independent, but best implemented AFTER broadcasting (easier to test with faster training). Needs `lambda_psd` added to `__init__` signature, HPO cell, and retrain cell.

---

### Feature 5: Configurable Circuit Layer Count

**Type:** Enhancement (parameterization, already partially configurable)
**Integration complexity:** Low
**Files modified:** Cell 28 (hyperparameter config) + Cell 37/40 (HPO/retrain)

**Current state:** `NUM_LAYERS = 4` is already configurable via the `num_layers` parameter to `qGAN.__init__()`. The circuit construction in `define_generator_circuit()` already loops `for layer in range(self.num_layers)`. The `count_params()` method already computes `self.num_layers * params_per_layer`.

**What needs to change:**
1. **Config cell (Cell 28):** Change `NUM_LAYERS = 4` to `NUM_LAYERS = 6` (or 8)
2. **HPO objective (Cell 37):** Add `num_layers` to search space:
   ```python
   num_layers = trial.suggest_int('num_layers', 4, 8)
   ```
3. **Model instantiation (Cell 40):** Use best num_layers from HPO
4. **Parameter count verification:** Update `expected_params` calculation in config cell

**Parameter count impact:**

| Layers | Params (5 qubits) | Formula |
|--------|--------------------|---------|
| 4 | 75 | 5 + 4*15 + 10 |
| 6 | 105 | 5 + 6*15 + 10 |
| 8 | 135 | 5 + 8*15 + 10 |

**Architectural concern:** More layers = more parameters = potentially harder to train with limited data (384 windows). The parameter-to-data ratio jumps from 75/384 (0.20) to 135/384 (0.35). This is a concern but not a blocker -- quantum circuit expressivity doesn't map directly to classical overfitting risk.

**Checkpoint incompatibility:** Changing NUM_LAYERS invalidates all existing checkpoints (params_pqc tensor size changes). Use a new checkpoint path.

**Dependency:** Independent. Can be combined with HPO in the search space.

---

### Feature 6: PAR_LIGHT Conditioning Verification

**Type:** Validation/diagnostic (not a code change to the training pipeline)
**Integration complexity:** Low
**Files modified:** New diagnostic cell(s) after training

**Architecture of the test:**
```python
# Generate samples with different PAR_LIGHT values, same noise
fixed_noise = torch.tensor(np.random.uniform(0, 4 * np.pi, size=NUM_QUBITS), dtype=torch.float32)

par_light_values = [0.0, 0.25, 0.5, 0.75, 1.0]
outputs = []
for par_val in par_light_values:
    par_input = torch.full((NUM_QUBITS,), par_val, dtype=torch.float32)
    with torch.no_grad():
        out = qgan.generator(fixed_noise, par_input, qgan.params_pqc)
        if isinstance(out, (list, tuple)):
            out = torch.stack(list(out))
        outputs.append(out.numpy())

# Compare outputs -- if PAR_LIGHT modulates, outputs should differ
outputs = np.array(outputs)
max_variation = np.max(np.std(outputs, axis=0))
print(f"Max cross-PAR_LIGHT variation: {max_variation:.6f}")
# If max_variation < 1e-4, conditioning is effectively ignored
```

**Current PAR_LIGHT encoding path:**
```
par_light_params (5 values, [0,1]) -> RY(val * pi) on each qubit
```
This is placed AFTER noise encoding but BEFORE the strongly entangled layers. The RY rotations are NOT parameterized by trainable weights -- they directly use the PAR_LIGHT values. This means the conditioning CAN modulate the output, but whether the trained model has learned to USE this modulation depends on the critic's ability to distinguish PAR_LIGHT-dependent patterns.

**Potential failure mode:** The critic receives PAR_LIGHT as a second channel. If the critic learns to ignore channel 1 (PAR_LIGHT), then the generator has no incentive to condition on PAR_LIGHT. The verification test above detects this.

**If conditioning is broken -- mitigation options:**
1. Increase critic's PAR_LIGHT channel weight (not straightforward with Conv1d)
2. Add PAR_LIGHT-dependent loss term (generate for known PAR_LIGHT, compare distribution shift)
3. Move PAR_LIGHT encoding deeper in circuit (interleave with entangling layers)

**Dependency:** Requires trained model. Purely diagnostic -- does not block other features.

---

### Feature 7: Simpler Critic Architecture Option

**Type:** New feature (alternative architecture)
**Integration complexity:** Medium
**Files modified:** Cell 26 (qGAN class -- new method + __init__ parameter)

**Current critic:**
```
Conv1d(2->64, k=10) -> Conv1d(64->128, k=10) -> Conv1d(128->128, k=10)
-> AdaptiveAvgPool1d(1) -> Linear(128->32) -> Dropout(0.2) -> Linear(32->1)
```
**Parameter count:** ~100K+ parameters (grossly overparameterized for 384 training windows of length 10)

**Proposed simpler critic:**
```python
def define_critic_model_simple(self, window_length):
    """Simpler critic: fewer params, less risk of overpowering generator."""
    model = nn.Sequential(
        nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.AdaptiveAvgPool1d(output_size=1),
        nn.Flatten(),
        nn.Linear(in_features=32, out_features=1),
    )
    model = model.double()
    return model
```

**Integration via __init__ parameter:**
```python
def __init__(self, ..., critic_type='standard'):
    ...
    if critic_type == 'simple':
        self.critic = self.define_critic_model_simple(window_length)
    else:
        self.critic = self.define_critic_model(window_length)
```

**No other code changes needed** -- the critic is used as `self.critic(input)` everywhere, and the input/output shapes are identical (`[batch, 2, window_length]` -> `[batch, 1]`).

**Key design decisions:**
- Keep `in_channels=2` (PAR_LIGHT conditioning must still work)
- Reduce kernel size from 10 to 3 (kernel_size=10 with window_length=10 means the first conv sees the ENTIRE window in one kernel -- essentially a fully connected layer disguised as a conv)
- Remove Dropout(0.2) -- violates WGAN-GP Lipschitz requirement (this was already flagged in v1.0 but appears to still be present)
- Reduce depth from 3 conv layers to 2

**Dependency:** Independent. Can be added to HPO search space alongside num_layers.

---

## Suggested Build Order

The build order is driven by two principles: (1) fix regressions before adding features, (2) restore performance before adding compute-intensive features.

```
Phase 1: Regression Fixes (no architectural changes)
    |
    +-- 1a. Noise range fix [0,2pi] -> [0,4pi]         (3 line changes)
    +-- 1b. mu/sigma shadowing fix                       (trivial rename)
    |
Phase 2: Performance Restoration
    |
    +-- 2a. Broadcasting optimization                    (refactor 3 loops)
    |       Depends on: 1a (noise range baked into new loops)
    |
Phase 3: New Features (order flexible, but suggested)
    |
    +-- 3a. Spectral/PSD loss                            (new method + loss term)
    |       Depends on: 2a (faster iteration for tuning lambda_psd)
    |
    +-- 3b. Configurable circuit layers                  (config change + HPO)
    |       Independent, but group with 3c for HPO
    |
    +-- 3c. Simpler critic option                        (new method + __init__ param)
    |       Independent, but group with 3b for HPO
    |
Phase 4: Validation
    |
    +-- 4a. PAR_LIGHT conditioning verification          (diagnostic cell)
    |       Depends on: trained model from Phase 3
    |
    +-- 4b. Full evaluation with new metrics             (eval cells)
            Depends on: all above
```

**Rationale for ordering:**
1. **1a before 2a:** The noise range fix is 3 line changes. Doing it first means the broadcasting refactor includes the correct range from the start, avoiding a second pass through the same code.
2. **2a before 3a:** Broadcasting gives ~12x speedup. Spectral loss adds compute to each epoch. Without broadcasting, the combined training time becomes prohibitive for iteration.
3. **3a before 3b/3c:** Spectral loss directly addresses the variance collapse problem. Circuit layers and critic architecture are secondary tuning knobs. However, if HPO is planned, 3a+3b+3c can all be added before the HPO run.
4. **4a last:** PAR_LIGHT verification is purely diagnostic. It needs a trained model. No point doing it before the model is improved.

---

## Component Modification Summary

| Component | New | Modified | Unchanged |
|-----------|-----|----------|-----------|
| `__init__` | `lambda_psd`, `critic_type`, `noise_range` | `critic` initialization (conditional) | All other attributes |
| `define_generator_circuit` | -- | -- | Unchanged (layers already parameterized) |
| `define_critic_model` | -- | -- | Unchanged (kept as 'standard' option) |
| `define_critic_model_simple` | NEW METHOD | -- | -- |
| `diff_psd_loss` | NEW METHOD (static) | -- | -- |
| `diff_acf_lag1` | -- | -- | Unchanged |
| `count_params` | -- | -- | Unchanged (already uses self.num_layers) |
| `_train_one_epoch` (critic) | -- | Broadcasting refactor, noise range | Loss calculation unchanged |
| `_train_one_epoch` (generator) | PSD loss term | Broadcasting refactor, noise range, loss equation | WGAN loss calculation |
| `_train_one_epoch` (eval) | -- | Broadcasting refactor, noise range | Stylized facts unchanged |
| `train_qgan` | -- | -- | Unchanged (delegates to _train_one_epoch) |
| `stylized_facts` | -- | -- | Unchanged |
| `compile_QGAN` | -- | -- | Unchanged |
| Config cell (28) | `LAMBDA_PSD`, `CRITIC_TYPE` | `NUM_LAYERS` value | Other hyperparams |
| HPO cell (37) | `num_layers`, `critic_type` in search | `lambda_psd` in search | Trial structure |
| Retrain cell (40) | Use new HPO params | -- | Training orchestration |
| Diagnostic cells | NEW CELLS for PAR_LIGHT test | -- | -- |

---

## Data Flow Changes

### Generator Loss (Modified)

```
BEFORE:
    gen_loss = -E[D(fake)] + lambda_acf * ACF_penalty

AFTER:
    gen_loss = -E[D(fake)] + lambda_acf * ACF_penalty + lambda_psd * PSD_penalty
```

The PSD penalty needs real samples for comparison. Currently, real samples are already fetched for ACF comparison. Reuse them:

```python
# Existing code (rename for clarity):
real_windows_for_aux = []
for _ in range(self.batch_size):
    random_idx = torch.randint(0, len(gan_data_list), (1,))
    real_windows_for_aux.append(gan_data_list[random_idx.item()].to(torch.float64))
real_batch_aux = torch.stack(real_windows_for_aux)  # [batch_size, window_length]

# ACF penalty (unchanged):
acf_penalty = ...  # uses real_batch_aux

# PSD penalty (new):
psd_penalty = self.diff_psd_loss(generated_samples, real_batch_aux)
```

### Broadcasting Output Shape Change

```
BEFORE (per-sample):
    generator(input_i) -> tuple of 10 scalars -> torch.stack -> [10]
    Loop batch_size times -> torch.stack -> [batch_size, 10]

AFTER (broadcasted):
    generator(input_batch) -> tuple of 10 tensors, each [batch_size] -> torch.stack(dim=-1) -> [batch_size, 10]
```

All downstream consumers expect `[batch_size, window_length]` which is preserved.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Separate Broadcasting for Critic vs Generator

**What people do:** Implement broadcasting separately in the critic training loop and the generator training loop with different approaches.
**Why it's wrong:** The output parsing logic must be identical. If one loop handles `(list, tuple)` differently, bugs emerge silently as shape mismatches.
**Do this instead:** Extract a `_generate_batch(noise_batch, par_batch)` helper method that handles broadcasting and output conversion once. Call it from both loops.

### Anti-Pattern 2: Spectral Loss on Raw FFT (Not Log-PSD)

**What people do:** Compute `MSE(fft_fake, fft_real)` directly.
**Why it's wrong:** FFT magnitudes vary by orders of magnitude across frequencies. DC component dominates the loss, mid-frequency structure gets zero gradient signal.
**Do this instead:** Use log-PSD: `MSE(log(|FFT|^2 + eps), log(|FFT|^2 + eps))`. The epsilon prevents log(0).

### Anti-Pattern 3: Changing Critic Architecture Without Resetting Optimizer

**What people do:** Switch `critic_type` between runs but reuse optimizer state from a previous architecture.
**Why it's wrong:** Adam optimizer momentum is shaped to the old parameter set. Loading old optimizer state into a new architecture causes dimension mismatches or silent performance degradation.
**Do this instead:** Always create fresh optimizers when changing critic architecture. Checkpoint paths should encode the architecture config.

### Anti-Pattern 4: Broadcasting params_pqc

**What people do:** Try to broadcast `params_pqc` along with noise to get "diverse" circuits.
**Why it's wrong:** `params_pqc` are the trainable weights -- they must be shared across the batch for meaningful gradient computation. Only the noise and conditioning inputs should be broadcasted.
**Do this instead:** Pass `params_pqc` as-is (unbatched). PennyLane handles the broadcasting correctly when some inputs are batched and others are not.

---

## Notebook Cell Organization

The notebook currently has ~44 code cells. The v1.1 changes affect these cell groups:

| Cell(s) | Current Content | v1.1 Changes |
|---------|----------------|--------------|
| 10 | Normalization (mu/sigma) | Fix shadowing |
| 26 | qGAN class (653 lines) | Noise range, broadcasting, PSD loss method, simple critic method, new __init__ params |
| 28 | Hyperparameter config | NUM_LAYERS, LAMBDA_PSD, CRITIC_TYPE |
| 37 | HPO objective | Add num_layers, critic_type, lambda_psd to search space |
| 40 | Retrain cell | Use new best params |
| NEW | PAR_LIGHT diagnostic | After training, before evaluation |

Cell 26 is the largest change. Consider whether to split the class across multiple cells for readability, but this is explicitly out of scope per project constraints ("edit existing notebook in-place").

---

## Sources

- PennyLane parameter broadcasting: [QNode docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html), [Broadcasting blog](https://pennylane.ai/blog/2022/10/how-to-execute-quantum-circuits-in-collections-and-batches/)
- PennyLane StronglyEntanglingLayers: [API docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html)
- PyTorch torch.fft: [API docs](https://docs.pytorch.org/docs/stable/fft.html), [Autograd FFT blog](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/)
- PennyLane PyTorch interface: [Interface docs](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)
- Direct codebase analysis: `qgan_pennylane.ipynb` Cell 26 (qGAN class, 653 lines)

---
*Architecture research for: v1.1 Post-HPO Integration*
*Researched: 2026-03-13*
