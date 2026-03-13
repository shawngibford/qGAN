# Feature Research: v1.1 Post-HPO Improvements

**Domain:** Quantum GAN time series synthesis -- variance collapse remediation and regression fixes
**Researched:** 2026-03-13
**Confidence:** MEDIUM-HIGH (spectral loss well-documented in literature; PennyLane layer parameterization verified from source; conditioning verification methods standard practice; critic simplification well-understood for WGAN-GP)

---

## Context

v1.0 shipped a correct, clean WGAN-GP with quantum generator. Post-HPO evaluation (2000 epochs with Optuna-optimized hyperparameters) revealed that variance collapse persists: fake std 0.0104 vs real std 0.0218 (48% of target). The generator learns the drift (mean trend) but not the volatility structure. Classical baselines (TinyVAE, FCVAE) also failed identically -- all produce smooth mean curves -- confirming this is a fundamental mode collapse problem, not a hyperparameter issue.

Additionally, the PAR_LIGHT conditioning work introduced three regressions: noise range reverted from [0, 4pi] to [0, 2pi] in the training loop (3 locations), broadcasting optimization was lost (per-sample Python loops instead of batched QNode calls), and mu/sigma variable shadowing appeared on cell re-execution.

The v1.1 features split cleanly into two categories: (A) regression fixes that restore v1.0 correctness, and (B) new capabilities targeting the variance collapse root cause.

---

## Table Stakes (Regressions That Must Be Fixed)

These are not new features -- they are regressions from the PAR_LIGHT conditioning work that broke previously-working functionality.

| Feature | Why Expected | Complexity | Impact on Variance Collapse | Dependencies |
|---------|--------------|------------|----------------------------|--------------|
| **TS-1: Noise range [0, 4pi] restoration** | v1.0 validated [0, 4pi] covers full RX gate period. Regressed to [0, 2pi] in 3 training loop locations during conditioning work. Half the noise range = half the generator's expressivity. | LOW -- 3 literal changes: `np.random.uniform(0, 2 * np.pi, ...)` to `np.random.uniform(0, 4 * np.pi, ...)` in critic training loop, generator training, and evaluation generation | MODERATE -- Restores full rotation coverage. Won't solve variance collapse alone but removing it halves the exploration space the generator can use. | None -- standalone fix |
| **TS-2: Broadcasting optimization** | v1.0 had batched QNode calls giving ~12x training speedup. Regressed to per-sample Python `for i in range(self.batch_size)` loops during conditioning refactor. Training is ~12x slower than it should be. | MEDIUM -- Must restructure noise generation and PAR_LIGHT compression to produce (num_qubits, batch_size) tensors, then call `qgan.generator(noise_batch, par_batch, params_pqc)` once instead of batch_size times. PennyLane broadcasting requires all array inputs to have a consistent batch dimension. | NO DIRECT IMPACT -- This is a performance fix, not a quality fix. But faster training enables more epochs, more experiments, and faster iteration on the features that do address variance collapse. | TS-1 (noise range must be correct before broadcasting it) |
| **TS-3: mu/sigma variable shadowing** | Cell 10 `mu`/`sigma` variables shadow on re-execution. Non-blocking in linear execution but violates v1.0 correctness standard. | TRIVIAL -- Inline the mu/sigma values into `norm.pdf()` calls or rename to `mu_plot`/`sigma_plot` | NONE -- Cosmetic correctness issue | None |

### Notes on Table Stakes

TS-1 and TS-2 are the only table stakes with any interaction with variance collapse. TS-1 restores the generator's ability to explore the full rotation manifold. TS-2 enables the iteration speed needed to test improvements effectively. TS-3 is a cleanup item.

---

## Differentiators (New Capabilities Targeting Variance Collapse)

### D-1: Spectral/PSD Mismatch Loss

**Value Proposition:** Directly penalizes the generator for producing time series with wrong frequency content. The ACF lag-1 penalty only captures first-order temporal correlation. A PSD loss captures the full frequency spectrum, specifically targeting the missing mid-frequency volatility that makes generated curves smooth.

**Complexity:** MEDIUM

**How it works:**

The standard approach is to compute the power spectral density of both real and generated windows, then penalize the mismatch. For a 1D time series window of length T:

1. Compute FFT: `F = torch.fft.rfft(x)` -- produces (T//2 + 1) complex coefficients
2. Compute PSD: `P = |F|^2 / T` -- power at each frequency bin
3. Take log: `log_P = torch.log(P + epsilon)` -- log-scale for numerical stability
4. Loss: `L_psd = MSE(log_P_fake, log_P_real)` -- mean squared error in log-PSD space

The entire pipeline is differentiable through `torch.fft.rfft` (autograd-supported since PyTorch 1.7). For WINDOW_LENGTH=10, this gives 6 frequency bins (DC through Nyquist), which is sparse but still captures the low/mid/high frequency structure.

**Combined loss:** `L_gen = L_wgan + lambda_acf * L_acf + lambda_psd * L_psd`

**Why log-PSD, not raw PSD:** Raw PSD is dominated by low-frequency (DC) components. Log-scale equalizes the contribution across frequencies, ensuring the generator is penalized for mid/high frequency mismatches. This is analogous to the Focal Frequency Loss (ICCV 2021) insight that vanilla frequency losses collapse once low frequencies converge.

**Why this over Focal Frequency Loss (FFL):** FFL was designed for 2D images with 2D DFT. Our signal is 1D and only 10 timesteps. The simpler log-PSD MSE is appropriate for our dimensionality. FFL's adaptive weighting mechanism adds complexity without clear benefit at T=10.

**Expected impact on variance collapse:** HIGH. The generator currently optimizes WGAN loss (overall distributional similarity) + ACF penalty (lag-1 correlation). Neither directly penalizes the flat power spectrum of smooth curves. A PSD loss will create explicit gradient signal pushing the generator toward the real data's frequency profile, which includes the mid-frequency volatility content that creates the "jaggedness" in real OD time series.

**Risk:** With T=10, the frequency resolution is limited (6 bins). The PSD loss may not have enough frequency granularity to fully resolve the volatility structure. The ACF penalty and PSD loss are somewhat redundant at lag-1 (both capture first-order temporal structure), but the PSD loss adds information at higher frequency bins.

**Depends on:** None (additive loss term). Should be implemented after TS-1/TS-2 so that experiments with the new loss run at correct noise range and full speed.

---

### D-2: Parameterizable Circuit Layer Count (4 -> 6-8)

**Value Proposition:** More variational layers = more expressivity = generator can represent more complex functions. The current 4-layer circuit may not have sufficient capacity to represent the volatility structure the critic is demanding.

**Complexity:** LOW-MEDIUM

**How it works in practice:**

The current circuit manually builds IQP encoding + per-layer (Rot + CNOT) blocks. The `NUM_LAYERS` constant already controls the loop count. Making it configurable means:

1. Change `NUM_LAYERS = 4` to a higher value (6 or 8)
2. Recompute `expected_params = iqp_params + NUM_LAYERS * per_layer_params + final_params`
3. Reinitialize `params_pqc` with the new shape
4. The circuit `define_generator_circuit` already loops `for layer in range(self.num_layers)`, so it scales automatically

For 5 qubits: 4 layers = 75 params, 6 layers = 105 params, 8 layers = 135 params. The parameter shape is `(iqp_params=5) + (layers * 15) + (final=10)`.

**Barren plateau risk:** This is the critical tradeoff. Research shows that for hardware-efficient ansatze (which StronglyEntanglingLayers is), the gradient variance decreases exponentially with circuit depth for random initialization. At 5 qubits:
- 4 layers: safe territory, gradients trainable
- 6 layers: likely still fine for 5 qubits (barren plateaus scale with qubit count more than depth for small circuits)
- 8 layers: approaching the regime where initialization strategy matters

**Mitigation:** The current circuit uses `np.random.uniform(0, 2*pi)` initialization. For deeper circuits, consider identity-block initialization: initialize new layers so they evaluate to identity, preserving the gradient landscape of the shallower circuit. PennyLane's data re-uploading structure (noise re-encoded each layer) partially mitigates barren plateaus by breaking the randomization that causes them.

**Expected impact on variance collapse:** MODERATE. More layers means the generator can express sharper, higher-frequency features in its output. However, the post-HPO results showed that classical baselines with far more parameters also failed. The expressivity bottleneck may not be the primary issue -- the loss landscape (what the generator is optimizing for) may matter more. This is why D-1 (PSD loss) is higher priority.

**Range parameter cycling:** StronglyEntanglingLayers cycles the CNOT range as `r = (l % (n_wires - 1)) + 1` per layer. For 5 qubits: range cycles through [1, 2, 3, 4, 1, 2, 3, 4]. At 4 layers this covers ranges 1-4 once; at 8 layers it covers them twice, giving the circuit two passes at each entanglement pattern.

**Depends on:** None directly. But should be tested alongside D-1 to separate the effect of expressivity from loss signal quality.

---

### D-3: PAR_LIGHT Conditioning Verification

**Value Proposition:** If the conditioning signal is being ignored by the generator, the entire PAR_LIGHT contribution is wasted. The thesis contribution depends on light conditioning actually modulating the output. Verification is not optional -- it is a research requirement.

**Complexity:** LOW

**How to verify conditioning effectiveness (standard approaches):**

1. **Intervention test (primary method):** Generate N samples with PAR_LIGHT=0.0 (no light) and N samples with PAR_LIGHT=1.0 (max light). Compare the output distributions. If the distributions are statistically indistinguishable (e.g., KS test p-value > 0.05, EMD delta near zero), the conditioning is being ignored.

2. **Gradient magnitude check:** Compute `d(output)/d(par_light_input)` for a batch of inputs. If the gradient magnitudes are near-zero relative to `d(output)/d(noise)`, the circuit is not sensitive to the conditioning signal.

3. **Sweep test:** Generate samples across a grid of PAR_LIGHT values [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. Plot a summary statistic (mean, std, ACF) of the generated windows as a function of PAR_LIGHT. If the curve is flat, conditioning is ineffective.

4. **Ablation test:** Train two models -- one with PAR_LIGHT conditioning, one without. Compare EMD on held-out data. If no improvement, conditioning adds no value.

**Why the conditioning might be ignored:**

The PAR_LIGHT encoding uses `qml.RY(par_light_params[i] * np.pi, wires=i)` applied once after IQP encoding, before the variational layers. This is a single RY rotation per qubit, sandwiched between the Hadamard+RZ(IQP)+RX(noise) encoding and the Rot+CNOT variational layers. The variational layers have 60 parameters (4 layers x 5 qubits x 3 rotations) that can effectively "absorb" or "cancel out" the RY rotation. The conditioning signal may not be strong enough relative to the variational freedom.

**Potential fixes if conditioning is ineffective:**
- Re-encode PAR_LIGHT in every layer (data re-uploading for conditioning, not just noise)
- Increase rotation range from `[0, pi]` to `[0, 2pi]`
- Use a different encoding axis (RZ instead of RY, if that creates more non-commutativity)

**Expected impact on variance collapse:** LOW-MODERATE. Effective conditioning could help if different PAR_LIGHT regimes have different volatility signatures. But variance collapse is present across all conditions, suggesting it is a generator capacity or loss signal problem, not a conditioning problem.

**Depends on:** TS-1 (noise range must be correct to isolate PAR_LIGHT effect), TS-2 (fast experiments needed for ablation testing).

---

### D-4: Simpler Critic Architecture Option

**Value Proposition:** If the critic is too powerful relative to the quantum generator, the critic can "perfectly" distinguish real from fake early in training, providing vanishing gradient signal to the generator. A simpler critic may provide more useful gradients for longer.

**Complexity:** MEDIUM

**Current critic architecture (parameter count):**
```
Conv1d(2, 64, k=10, pad=5)   ->   640 + 64   = 704
Conv1d(64, 128, k=10, pad=5) -> 81,920 + 128  = 82,048
Conv1d(128, 128, k=10, pad=5)-> 163,840 + 128 = 163,968
AdaptiveAvgPool1d(1)         ->     0
Flatten                      ->     0
Linear(128, 32)              -> 4,096 + 32     = 4,128
Dropout(0.2)                 ->     0
Linear(32, 1)                -> 32 + 1         = 33
Total critic parameters: ~250,881
```

**Current generator parameters:** 75 (quantum circuit)

**Ratio: 3,345:1** -- The critic has over 3,000 times more parameters than the generator. This is an extreme imbalance.

**Simpler critic option:**
```
Conv1d(2, 32, k=5, pad=2)    ->   320 + 32    = 352
LeakyReLU(0.1)
Conv1d(32, 32, k=5, pad=2)   -> 5,120 + 32    = 5,152
LeakyReLU(0.1)
AdaptiveAvgPool1d(1)
Flatten
Linear(32, 1)                -> 32 + 1         = 33
Total: ~5,537 parameters
```

This reduces the ratio to ~74:1, still in the critic's favor but much less extreme.

**WGAN-GP specific considerations:**

1. **No batch normalization in critic:** The WGAN-GP paper (Gulrajani et al., 2017) explicitly states batch normalization should not be used in the critic because it creates correlations between batch samples, making the gradient penalty less effective. The current implementation correctly avoids BatchNorm but incorrectly includes Dropout(0.2), which also introduces stochasticity in the critic's output. Dropout in the WGAN-GP critic is non-standard and should be removed.

2. **Critic should be "good enough, not perfect":** In WGAN-GP, the critic approximates the Wasserstein distance. An overpowered critic can approximate it so well that the gradient signal saturates -- the critic always says "clearly fake" with maximum confidence. The n_critic=5 ratio gives the critic 5x more gradient updates, compounding the capacity advantage.

3. **Fewer layers, smaller kernels:** The current kernel_size=10 on a WINDOW_LENGTH=10 signal means each convolution sees the entire window in one pass. A kernel_size=5 forces the critic to build up its representation hierarchically, which may produce more spatially-distributed gradient feedback.

**Expected impact on variance collapse:** MODERATE-HIGH. The 3,345:1 parameter ratio is a red flag. If the critic perfectly separates real/fake early on, the generator gets no useful gradient to improve volatility structure. A weaker critic may force the generator to learn finer distributional features to "fool" the critic, rather than just matching the mean trend. However, making the critic too weak risks the opposite problem: the generator can fool it without learning anything meaningful.

**Risk:** Critic simplification is a delicate balance. The "right" critic capacity depends on the generator's current ability. This should be implemented as a configurable option (not a replacement), tested empirically.

**Depends on:** None directly. Can be tested independently. But the combined effect of D-1 (PSD loss) + D-4 (simpler critic) may be synergistic: a weaker critic plus a stronger auxiliary loss could be the right combination.

---

## Anti-Features (Explicitly NOT Building)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Multi-resolution spectral loss** | Used in audio GANs (auraloss) for comprehensive frequency coverage | Overkill for T=10 signal with only 6 frequency bins. Multi-resolution makes sense for T=1024+ signals. | Single-resolution log-PSD MSE is sufficient at this window length. |
| **Focal Frequency Loss (FFL)** | ICCV 2021 paper shows adaptive frequency weighting helps | Designed for 2D images. The adaptive weighting adds complexity and a tunable alpha parameter. At T=10, the frequency space is too small for adaptive weighting to have meaning. | Simple log-PSD MSE with uniform weighting across 6 bins. |
| **Spectral normalization on critic weights** | Different "spectral" technique -- normalizes weight matrices for Lipschitz constraint | WGAN-GP already enforces Lipschitz via gradient penalty. Adding spectral normalization is redundant and the interaction between GP and SN is not well-studied. | Keep gradient penalty as sole Lipschitz enforcement. |
| **StronglyEntanglingLayers template replacement** | PennyLane provides `qml.StronglyEntanglingLayers()` as a convenience template | The current manual circuit has IQP encoding + PAR_LIGHT encoding interleaved between layers, which the template cannot express. Switching to the template would lose the custom encoding structure. | Keep manual circuit construction. Use the template's `shape()` method only for parameter counting verification. |
| **Circuit architecture search (automated)** | Automated NAS for quantum circuits | Enormous computational cost. Each architecture evaluation requires full training. Not feasible in a notebook workflow. | Manual testing of 4, 6, 8 layer configurations is sufficient. |
| **Autoregressive stitching for longer windows** | Generate longer time series by chaining windows | Changes the generation paradigm fundamentally. Would require a new training procedure and different evaluation metrics. | Stay with WINDOW_LENGTH=10 for v1.1. Longer windows are a v2.0 concern. |
| **Wasserstein distance as spectral loss** | Use optimal transport in frequency domain | Adds scipy dependency in the training loop hot path. Not differentiable through torch autograd without custom implementation. | MSE in log-PSD space is simpler and fully differentiable. |

---

## Feature Dependencies

```
TS-1 (noise range fix)
    |
    +---> TS-2 (broadcasting) -- needs correct noise range
    |         |
    |         +---> D-1 (PSD loss) -- needs fast training for tuning lambda_psd
    |         +---> D-2 (layer count) -- needs fast training for experiments
    |         +---> D-3 (conditioning verify) -- needs fast experiments for ablation
    |         +---> D-4 (critic simplification) -- needs fast experiments
    |
    +---> D-3 (conditioning verify) -- needs correct noise range to isolate PAR_LIGHT

TS-3 (mu/sigma fix) -- standalone, no dependencies

D-1 (PSD loss) --enhances--> D-4 (critic simplification)
    Rationale: PSD provides explicit frequency feedback, allowing a simpler
    critic that focuses on overall distributional quality rather than needing
    to implicitly learn frequency discrimination.

D-2 (layer count) --tested-with--> D-1 (PSD loss)
    Rationale: More layers without better loss signal may not help.
    More layers WITH PSD loss may be synergistic.

D-3 (conditioning verify) --potentially-leads-to--> conditioning refactor
    Rationale: If conditioning is ineffective, a refactor is needed.
    This is gated on the verification result.

D-4 (critic simplification) --conflicts-with--> aggressive critic weakening
    Rationale: Must maintain critic quality sufficient to estimate
    Wasserstein distance. Too weak = meaningless loss.
```

### Dependency Notes

- **TS-1 before TS-2:** Broadcasting the wrong noise range would bake the regression into the optimized code path.
- **TS-2 before D-1 through D-4:** All new features require experimental iteration. At ~12x slowdown, experiments that should take 1 hour take 12 hours. Broadcasting is a force multiplier.
- **D-1 has highest standalone impact:** The PSD loss addresses the root cause (loss signal does not penalize missing volatility) and can be tested independently.
- **D-4 should be tested after D-1:** If PSD loss alone resolves variance collapse, critic simplification may be unnecessary. If not, the combination is the next step.

---

## Prioritized Build Order

### Phase 1: Regression Fixes (Prerequisite)

- [x] **TS-1: Noise range restoration** -- 3 literal changes, immediate correctness win
- [x] **TS-2: Broadcasting optimization** -- ~12x speedup enables all experiments
- [x] **TS-3: mu/sigma shadowing** -- Trivial cleanup, do it while touching the file

### Phase 2: High-Impact Additions (Core of v1.1)

- [ ] **D-1: Spectral/PSD loss** -- Highest expected impact on variance collapse. Addresses root cause: loss function does not penalize missing frequency content.
- [ ] **D-3: PAR_LIGHT conditioning verification** -- Must-have for thesis. Quick to implement (intervention test is ~20 lines of code + plots). Results gate whether conditioning refactoring is needed.

### Phase 3: Capacity Tuning (If Variance Collapse Persists)

- [ ] **D-2: Layer count parameterization** -- Test 6 and 8 layers with PSD loss active. Compare EMD and std ratio.
- [ ] **D-4: Critic architecture option** -- Test simpler critic if the 3,345:1 parameter ratio is confirmed as problematic. Implement as a flag, not a replacement.

### Phase Ordering Rationale

1. **Regressions first** because broken code invalidates all experiments.
2. **PSD loss before capacity changes** because the loss signal is the more fundamental problem. Adding capacity (layers) without fixing what the generator optimizes for is like giving someone a bigger brush when they are painting the wrong picture.
3. **Conditioning verification early** because it is low-effort, thesis-critical, and gates a potential refactoring decision.
4. **Critic simplification last** because it is the most uncertain intervention -- the "right" critic capacity is empirical and may require multiple A/B tests.

---

## Feature Prioritization Matrix

| Feature | Variance Collapse Impact | Implementation Cost | Experiment Cost | Priority |
|---------|-------------------------|---------------------|-----------------|----------|
| TS-1: Noise range | MODERATE (restores expressivity) | LOW (3 literals) | NONE | P0 |
| TS-2: Broadcasting | NONE (performance only) | MEDIUM (tensor reshaping) | NONE | P0 |
| TS-3: mu/sigma | NONE (cosmetic) | TRIVIAL | NONE | P0 |
| D-1: PSD loss | HIGH (addresses root cause) | MEDIUM (~30 lines) | MEDIUM (tune lambda_psd) | P1 |
| D-3: Conditioning verify | LOW-MOD (gates thesis claim) | LOW (~20 lines + plots) | LOW (single run) | P1 |
| D-2: Layer count | MODERATE (more expressivity) | LOW (change constant) | HIGH (retrain at each depth) | P2 |
| D-4: Critic simplification | MOD-HIGH (balance ratio) | MEDIUM (new architecture) | HIGH (A/B test) | P2 |

**Priority key:**
- P0: Must fix before any experiments (regressions)
- P1: Highest-impact new capabilities, implement first
- P2: Test after P1 results are in, implement if needed

---

## Implementation Sketches

### D-1: PSD Loss (core addition)

```python
def psd_loss(self, fake_windows, real_windows):
    """Compute log-PSD MSE between generated and real windows.

    Args:
        fake_windows: (batch_size, window_length) generated samples
        real_windows: (batch_size, window_length) real samples
    Returns:
        scalar loss value (differentiable)
    """
    # Compute FFT (real-valued input -> one-sided spectrum)
    fake_fft = torch.fft.rfft(fake_windows, dim=-1)  # (batch, T//2+1)
    real_fft = torch.fft.rfft(real_windows, dim=-1)   # (batch, T//2+1)

    # Power spectral density: |F|^2 / T
    fake_psd = torch.abs(fake_fft) ** 2 / fake_windows.shape[-1]
    real_psd = torch.abs(real_fft) ** 2 / real_windows.shape[-1]

    # Log-scale for balanced frequency contribution
    eps = 1e-8
    fake_log_psd = torch.log(fake_psd + eps)
    real_log_psd = torch.log(real_psd + eps)

    # MSE in log-PSD space
    return torch.mean((fake_log_psd - real_log_psd) ** 2)
```

### D-3: Conditioning Intervention Test

```python
# Generate samples at extreme PAR_LIGHT values
num_test = 100
par_low = torch.zeros(NUM_QUBITS)       # PAR_LIGHT = 0 (no light)
par_high = torch.ones(NUM_QUBITS)       # PAR_LIGHT = 1 (max light)

samples_low, samples_high = [], []
with torch.no_grad():
    for _ in range(num_test):
        noise = torch.tensor(np.random.uniform(0, 4*np.pi, NUM_QUBITS), dtype=torch.float32)
        out_low = qgan.generator(noise, par_low, qgan.params_pqc)
        out_high = qgan.generator(noise, par_high, qgan.params_pqc)
        samples_low.append(torch.stack(list(out_low)))
        samples_high.append(torch.stack(list(out_high)))

samples_low = torch.stack(samples_low)   # (100, WINDOW_LENGTH)
samples_high = torch.stack(samples_high) # (100, WINDOW_LENGTH)

# Statistical test: are distributions different?
from scipy.stats import ks_2samp
ks_stat, ks_pval = ks_2samp(samples_low.flatten().numpy(), samples_high.flatten().numpy())
print(f"KS test: stat={ks_stat:.4f}, p={ks_pval:.4e}")
print(f"Conditioning {'EFFECTIVE' if ks_pval < 0.05 else 'INEFFECTIVE'}")
```

---

## Sources

### Spectral/PSD Loss
- [A Spectral Enabled GAN for Time Series Data Generation (arXiv 2103.01904)](https://arxiv.org/abs/2103.01904) -- adversarial spectral loss combining time and frequency domains
- [Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)](https://github.com/EndlessSora/focal-frequency-loss) -- adaptive frequency weighting, log-PSD stability insights
- [PyTorch torch.fft module](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/) -- differentiable FFT with autograd support (HIGH confidence)
- [torch.fft.rfft documentation](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html) -- one-sided FFT for real signals (HIGH confidence)
- [EEG Signal Reconstruction with Temporal-Spatial-Frequency Loss](https://www.frontiersin.org/articles/10.3389/fninf.2020.00015/full) -- PSD features in frequency domain for temporal characteristics

### PennyLane Circuit Parameterization
- [PennyLane StronglyEntanglingLayers API (v0.44.1)](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html) -- shape(n_layers, n_wires) returns (n_layers, n_wires, 3) (HIGH confidence, verified from source)
- [PennyLane Barren Plateaus demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/) -- gradient variance vs circuit depth
- [Quantum models as Fourier series (PennyLane)](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series) -- expressivity through Fourier lens
- [Quantum GANs PennyLane demo](https://pennylane.ai/qml/demos/tutorial_quantum_gans) -- reference implementation

### WGAN-GP Critic Architecture
- [Improved Training of Wasserstein GANs (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028) -- no batch normalization in critic, gradient penalty specification (HIGH confidence)
- [WGAN-GP batch normalization discussion](https://community.deeplearning.ai/t/batch-norm-in-the-critic-in-wgan-gp-assignment/348063) -- practical guidance on critic architecture
- [Quantum GAN with WGAN-GP (EPJ Quantum Technology, 2025)](https://link.springer.com/article/10.1140/epjqt/s40507-025-00372-z) -- quantum-classical hybrid WGAN-GP architecture

### Conditioning Verification
- [GANs Conditioning Methods: A Survey (arXiv 2408.15640)](https://arxiv.org/abs/2408.15640) -- comprehensive survey of conditioning approaches and evaluation
- [On the Evaluation of Conditional GANs (arXiv 1907.08175)](https://arxiv.org/abs/1907.08175) -- Frechet Joint Distance for conditional consistency evaluation
- [Barren Plateau Lie Algebraic Theory](https://www.nature.com/articles/s41467-024-49909-3) -- depth-expressivity-trainability tradeoff (HIGH confidence)

---
*Feature research for: qGAN v1.1 Post-HPO Improvements*
*Researched: 2026-03-13*
