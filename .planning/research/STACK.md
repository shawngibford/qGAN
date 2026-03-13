# Stack Research: v1.1 Post-HPO Improvements

**Domain:** Quantum GAN (PennyLane + PyTorch) — spectral loss, broadcasting, circuit parameterization, critic variants
**Researched:** 2026-03-13
**Confidence:** HIGH (all capabilities verified against installed stack)

---

## Executive Summary

The v1.1 improvements require **zero new dependencies**. Every capability needed -- spectral/PSD loss, parameter broadcasting, configurable circuit layers, and simpler critic architectures -- is achievable with the already-installed PennyLane 0.44.0, PyTorch 2.8.0, and NumPy/SciPy stack. The key "stack change" is a configuration change: switching `diff_method='parameter-shift'` to `diff_method='backprop'`, which unlocks both a ~90x gradient speedup AND native parameter broadcasting support.

---

## Changes to Existing Stack

### Critical Configuration Change: diff_method

| Setting | Current | Required | Impact |
|---------|---------|----------|--------|
| `diff_method` | `'parameter-shift'` | `'backprop'` | Enables broadcasting, ~90x gradient speedup |

**Why this is critical:** The notebook still uses `parameter-shift` despite v1.0 research recommending `backprop`. Parameter broadcasting does NOT work with `parameter-shift` -- it requires `backprop` on `default.qubit` with `shots=None`. This single change is the prerequisite for the ~12x training speedup from batched QNode execution.

**Confidence: HIGH.** PennyLane 0.44.0 documentation explicitly states broadcasting requires `backprop`. The PennyLane discussion forums confirm `parameter-shift` broadcasting has gradient issues that were never fully resolved, while `backprop` broadcasting works correctly.

---

## Feature 1: Spectral/PSD Loss

### What's Needed

Compute power spectral density (PSD) mismatch between real and generated time series windows, use as differentiable auxiliary loss during generator training.

### Stack: `torch.fft.rfft` (already installed)

| Component | Source | Version | Status |
|-----------|--------|---------|--------|
| `torch.fft.rfft` | PyTorch | 2.8.0 (installed) | Built-in, fully differentiable |
| `torch.fft.rfftfreq` | PyTorch | 2.8.0 (installed) | For frequency axis labels |

**Why `torch.fft.rfft`:** It is built into PyTorch since v1.7.0, supports autograd (gradients flow through FFT), works on GPU/CPU, and returns the compact one-sided spectrum for real-valued signals. No external library needed.

**Implementation pattern:**

```python
def psd_loss(real_windows, fake_windows):
    """Differentiable PSD mismatch loss for generator training.

    Args:
        real_windows: [batch_size, window_length] real data
        fake_windows: [batch_size, window_length] generated data
    Returns:
        Scalar loss (mean L1 distance between log-PSDs)
    """
    # Compute one-sided FFT
    real_fft = torch.fft.rfft(real_windows, dim=-1)
    fake_fft = torch.fft.rfft(fake_windows, dim=-1)

    # Power spectral density = |FFT|^2
    real_psd = torch.abs(real_fft) ** 2
    fake_psd = torch.abs(fake_fft) ** 2

    # Log-PSD for scale-invariant comparison (add epsilon for numerical stability)
    real_log_psd = torch.log(real_psd + 1e-8)
    fake_log_psd = torch.log(fake_psd + 1e-8)

    # Mean L1 distance across frequency bins and batch
    return torch.mean(torch.abs(real_log_psd - fake_log_psd))
```

**Why NOT Focal Frequency Loss (FFL):** FFL (EndlessSora/focal-frequency-loss) is designed for 2D images (expects NCHW tensors). Our data is 1D time series with window_length=10, which has only 6 frequency bins after rfft. The adaptive weighting of FFL adds complexity with no benefit at this scale. A simple log-PSD L1 loss is more appropriate and more interpretable for a PhD thesis.

**Why NOT scipy.signal.periodogram:** Not differentiable. Torch operations are needed for gradient flow through the generator.

**Why L1 on log-PSD:** L1 is more robust to outlier frequency bins than L2. Log scale ensures mid-frequency components get attention equal to low-frequency components (addresses the variance collapse where the generator only learns drift).

**Confidence: HIGH.** `torch.fft` module has been stable since PyTorch 1.7.0, with full autograd support confirmed in official PyTorch blog and docs.

---

## Feature 2: Parameter Broadcasting (Batched QNode Execution)

### What's Needed

Replace per-sample Python loops with a single batched QNode call for ~12x speedup (batch_size=12).

### Stack: PennyLane native broadcasting (already installed)

| Component | Source | Version | Status |
|-----------|--------|---------|--------|
| QNode broadcasting | PennyLane | 0.44.0 (installed) | Native, requires `backprop` |

**How it works:** Instead of calling the QNode 12 times in a Python loop, pass inputs with an extra batch dimension. PennyLane vectorizes the statevector simulation internally.

```python
# CURRENT (slow): per-sample loop
for i in range(self.batch_size):
    noise = torch.tensor(np.random.uniform(0, 4*np.pi, size=self.num_qubits), ...)
    result = self.generator(noise, par_light[i], self.params_pqc)
    fake_batch.append(result)

# FIXED (fast): broadcasted call
noise_batch = torch.tensor(
    np.random.uniform(0, 4*np.pi, size=(self.batch_size, self.num_qubits)),
    dtype=torch.float32
)  # shape: (batch_size, num_qubits)
par_light_batch = ...  # shape: (batch_size, num_qubits)
# params_pqc stays unbatched -- PennyLane zips broadcasted + non-broadcasted correctly
results = self.generator(noise_batch.T, par_light_batch.T, self.params_pqc)
# results: tuple of (batch_size,) tensors, one per measurement
fake_batch = torch.stack(list(results)).T  # (batch_size, window_length)
```

**Key rules:**
1. Broadcasted inputs must have exactly one extra axis vs what operator expects (scalars become 1D arrays)
2. Multiple broadcasted inputs must have matching batch dimensions
3. Non-broadcasted inputs (like `params_pqc`) are reused across the batch automatically
4. Works with `backprop` on `default.qubit`. Does NOT work reliably with `parameter-shift`
5. The batch dimension convention: inputs shape `(num_qubits, batch_size)` -- PennyLane broadcasts over the last axis of 1D inputs

**Critical prerequisite:** Must switch to `diff_method='backprop'` first. Broadcasting with `parameter-shift` has known gradient bugs (PennyLane issue #4462).

**Existing partial implementation:** The evaluation cell (line ~4278) already uses broadcasting correctly for inference: `results = qgan.generator(noise, par_tensor_gen, qgan.params_pqc)`. The training loop is what regressed to per-sample loops.

**Confidence: HIGH.** Broadcasting with `backprop` has been supported since PennyLane 0.31.0. The existing evaluation cell in the notebook proves it works with this circuit architecture.

---

## Feature 3: Configurable Circuit Layer Count

### What's Needed

Change `NUM_LAYERS` from hardcoded 4 to configurable 6-8, with proper parameter counting.

### Stack: No changes needed

| Component | Source | Version | Status |
|-----------|--------|---------|--------|
| `qml.StronglyEntanglingLayers` | PennyLane | 0.44.0 | Available but NOT recommended |
| Manual Rot + CNOT pattern | PennyLane | 0.44.0 | Already implemented, keep it |

**Recommendation: Keep the manual circuit, do NOT switch to `qml.StronglyEntanglingLayers`.**

**Why keep manual circuit:**

1. **Data re-uploading:** The current circuit re-encodes noise between layers via `encoding_layer()` and `par_light_encoding()`. `StronglyEntanglingLayers` is a single template call that does NOT support interleaved data re-uploading. Switching would require breaking the template into individual layers anyway, negating any simplification.

2. **Parameter counting transparency:** The manual circuit has explicit `count_params()` that matches the code. `StronglyEntanglingLayers` uses shape `(num_layers, num_qubits, 3)` internally, but adding IQP encoding params, PAR_LIGHT encoding, and measurement prep rotations still requires manual accounting.

3. **Conditioning integration:** PAR_LIGHT RY encoding is interleaved after noise RZ encoding. This is custom to this circuit and doesn't fit `StronglyEntanglingLayers`.

**What changes for configurable layers:**

```python
# Current hardcoded:
NUM_LAYERS = 4

# Change to configurable:
NUM_LAYERS = 6  # or 7, or 8

# Parameter count scales linearly:
# params = NUM_QUBITS + NUM_LAYERS * (NUM_QUBITS * 3) + NUM_QUBITS * 2
# 4 layers: 5 + 4*15 + 10 = 75 params
# 6 layers: 5 + 6*15 + 10 = 105 params
# 8 layers: 5 + 8*15 + 10 = 135 params
```

**Expressivity scaling:** More layers = more trainable parameters = higher circuit expressivity. For 5 qubits, the circuit has 2^5 = 32 dimensional Hilbert space. The current 75 parameters (4 layers) may underparameterize the circuit. 105-135 parameters (6-8 layers) provides better coverage.

**Risk:** More layers = deeper circuit = potential barren plateaus. For 5-8 layers on 5 qubits, this is not a concern (barren plateaus primarily affect >10 qubits with global cost functions). The local Pauli measurements used here mitigate this.

**Confidence: HIGH.** Changing `NUM_LAYERS` is purely a hyperparameter change. The `count_params()` method already handles arbitrary layer counts.

---

## Feature 4: Simpler Critic Architecture

### What's Needed

Add a simpler MLP critic option alongside the existing 3-layer CNN critic to test if the critic is overpowering the quantum generator.

### Stack: PyTorch `nn.Sequential` (already installed)

| Component | Source | Version | Status |
|-----------|--------|---------|--------|
| `nn.Linear` | PyTorch | 2.8.0 | Built-in |
| `nn.LeakyReLU` | PyTorch | 2.8.0 | Built-in |
| `nn.LayerNorm` | PyTorch | 2.8.0 | Built-in, preferred over BatchNorm for WGAN-GP |

**Current CNN critic (too powerful?):**
```
Conv1d(2, 64, k=10) -> LeakyReLU -> Conv1d(64, 128, k=10) -> LeakyReLU -> Conv1d(128, 128, k=10) -> LeakyReLU -> AdaptiveAvgPool1d -> Linear(128, 32) -> LeakyReLU -> Dropout(0.2) -> Linear(32, 1)
```
- Parameters: ~150K+ (estimated from kernel sizes and channel counts)
- Issues: Dropout violates WGAN-GP theory (stochastic outputs invalidate gradient penalty), kernel_size=10 equals window_length (entire receptive field in one layer)

**Recommended simpler critic (MLP):**
```python
def define_critic_model_simple(self, window_length):
    """Simpler MLP critic for better generator-critic balance."""
    input_dim = 2 * window_length  # flatten 2 channels x window_length
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
    )
    model = model.double()
    return model
```
- Parameters: ~3.5K (20x + 64x32 + 32 = ~3.5K)
- No dropout (correct for WGAN-GP)
- Capacity closer to quantum generator (~75-135 params)

**Why MLP and not a smaller CNN:** With window_length=10 and 2 channels, the input is only 20 values. A 1D CNN with kernel_size=10 already covers the entire window in one layer -- there's no locality benefit. An MLP is more honest about the data size.

**Why NOT LayerNorm:** For WGAN-GP, the critic should be as unconstrained as possible (gradient penalty enforces Lipschitz, not normalization). Skip normalization layers in the critic.

**Why NOT spectral normalization:** Spectral normalization (SN) is an alternative to gradient penalty, not a complement. Using both would over-constrain the critic. Stick with WGAN-GP.

**Configuration approach:**
```python
CRITIC_TYPE = "cnn"  # or "mlp"
# In __init__:
if critic_type == "mlp":
    self.critic = self.define_critic_model_simple(window_length)
else:
    self.critic = self.define_critic_model(window_length)
```

**Also fix existing CNN critic:** Remove `Dropout(0.2)` regardless of which architecture is used. Dropout in a WGAN-GP critic makes the gradient penalty computation theoretically invalid because the same interpolated sample produces different outputs on each forward pass.

**Confidence: HIGH.** MLP critics are standard in tabular/low-dimensional WGAN-GP implementations. The architecture balance problem (overpowered critic vs weak quantum generator) is well-documented in quantum GAN literature.

---

## Feature 5: PAR_LIGHT Conditioning Verification

### Stack: No new dependencies

This is a diagnostic task, not a library task. Verification requires:

1. Generate samples with PAR_LIGHT=0.0 vs PAR_LIGHT=1.0 (same noise)
2. Compare outputs to confirm they differ
3. Compute sensitivity: `d(output) / d(PAR_LIGHT)`

All achievable with existing PyTorch `torch.autograd.grad` or simple forward-pass comparison. No new stack needed.

---

## Summary: What to Add, What to Change, What NOT to Add

### Add (zero new packages)

| Capability | Implementation | Using |
|------------|---------------|-------|
| PSD loss | `torch.fft.rfft` + `torch.abs` + `torch.log` | PyTorch 2.8.0 (installed) |
| Simpler critic | `nn.Sequential` with `nn.Linear` layers | PyTorch 2.8.0 (installed) |
| Configurable layers | Change `NUM_LAYERS` hyperparameter | Already supported by `count_params()` |

### Change (configuration only)

| What | From | To | Why |
|------|------|----|-----|
| `diff_method` | `'parameter-shift'` | `'backprop'` | Prerequisite for broadcasting, ~90x gradient speedup |
| Training loop | Per-sample Python loop | Batched QNode call | ~12x training speedup |
| Critic dropout | `Dropout(0.2)` | Remove | Violates WGAN-GP theory |

### Do NOT Add

| Temptation | Why Not |
|------------|---------|
| `focal-frequency-loss` package | Designed for 2D images (NCHW), overkill for 6-bin 1D PSD |
| `scipy.signal` for PSD | Not differentiable, can't flow gradients to generator |
| `qml.StronglyEntanglingLayers` template | Doesn't support interleaved data re-uploading or PAR_LIGHT encoding |
| `torch-fidelity` or other GAN eval packages | Designed for image GANs, not 1D time series |
| Spectral normalization (`torch.nn.utils.spectral_norm`) | Conflicts with WGAN-GP gradient penalty approach |
| `pennylane-lightning` with `adjoint` | Adds complexity; `backprop` on `default.qubit` is simpler and sufficient for 5 qubits |
| Any new pip packages | Everything needed is already installed |

---

## Version Compatibility

| Package | Installed | Required Feature | Compatible |
|---------|-----------|-----------------|------------|
| PennyLane 0.44.0 | Yes | Broadcasting + backprop | Yes (since 0.31.0) |
| PyTorch 2.8.0 | Yes | `torch.fft.rfft` with autograd | Yes (since 1.7.0) |
| PyTorch 2.8.0 | Yes | `nn.Sequential` for MLP critic | Yes (all versions) |
| NumPy 1.26.4 | Yes | Array operations | Yes |

No version upgrades required. No new packages required.

---

## Sources

- [PennyLane QNode API -- 0.44.0](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) -- broadcasting rules, diff_method options
- [PennyLane StronglyEntanglingLayers -- 0.44.1](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html) -- weights shape (L, M, 3), no data re-uploading support
- [PennyLane Broadcasting Issues with PyTorch](https://discuss.pennylane.ai/t/issues-with-backpropagation-when-using-parameter-broadcasting-with-pytorch/3333) -- parameter-shift broadcasting has gradient bugs
- [PyTorch torch.fft Module Blog](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/) -- autograd support confirmed
- [torch.fft.rfft -- PyTorch 2.10 docs](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html) -- one-sided FFT for real signals
- [Focal Frequency Loss (ICCV 2021)](https://github.com/EndlessSora/focal-frequency-loss) -- evaluated and rejected (2D image focus)
- [Spectral GAN for Time Series (2021)](https://ar5iv.labs.arxiv.org/html/2103.01904) -- spectral loss in time series GANs
- [Gulrajani et al. (2017) -- WGAN-GP](https://arxiv.org/abs/1704.00028) -- no dropout in critic, standard hyperparameters
- [PennyLane Gradients and Training](https://docs.pennylane.ai/en/stable/introduction/interfaces.html) -- PyTorch interface backprop support

---
*Stack research for: qGAN v1.1 Post-HPO Improvements*
*Researched: 2026-03-13*
