# Architecture

**Analysis Date:** 2026-02-26

## Pattern Overview

**Overall:** Hybrid Classical-Quantum Generative Adversarial Network (WGAN-GP)

**Key Characteristics:**
- Adversarial training loop with quantum generator and classical discriminator
- Wasserstein GAN with Gradient Penalty for stable training
- Parameterized quantum circuits (PQC) for data generation
- Time series synthesis for bioprocess data
- Modular architecture separating quantum and classical components

## Layers

**Quantum Generator:**
- Purpose: Generate synthetic time series samples from noise using quantum circuits
- Location: Embedded in `qGAN` class in main notebooks
- Contains: PennyLane quantum circuits with parameterized rotation gates, entangling layers, and measurement operations
- Depends on: PennyLane quantum framework, parameter tensors (`params_pqc`)
- Used by: Training loop for generating fake samples during adversarial training

**Classical Discriminator (Critic):**
- Purpose: Distinguish between real and generated time series windows
- Location: Defined in `qGAN.define_critic_model()` method (1D CNN architecture)
- Contains: Sequential PyTorch model with Conv1D layers, LeakyReLU activations, adaptive pooling, and dense layers
- Depends on: PyTorch nn.Module, real and generated batch samples
- Used by: Training loop to compute Wasserstein loss and gradient penalty

**Data Preprocessing Layer:**
- Purpose: Transform raw time series into normalized format suitable for quantum circuits
- Location: Preprocessing code in main notebooks (data transformation, Lambert W transform, normalization)
- Contains: Statistical preprocessing, log-return calculation, rolling window generation
- Depends on: NumPy, PyTorch, SciPy for Lambert W transform
- Used by: Training pipeline to prepare data batches

**Training Loop:**
- Purpose: Alternately train critic and generator in adversarial fashion
- Location: `_train_one_epoch()` method in `qGAN` class
- Contains: Critic training iterations (n_critic steps), generator training, batch sampling, gradient flow
- Depends on: Quantum generator, classical discriminator, optimizers (Adam)
- Used by: Main training orchestration in notebooks

## Data Flow

**Training Data Flow:**

1. Load raw CSV data (`data.csv`)
2. Preprocess: Calculate log-returns from optical density measurements
3. Apply Lambert W transformation to normalize non-Gaussian features
4. Generate rolling windows with stride=2 (window_length=10)
5. Normalize to [-1, 1] range using min-max scaling
6. Create DataLoader for batch sampling

**Forward Pass (Single Training Step):**

1. **Critic Training Loop (n_critic iterations per epoch):**
   - Sample real batch: Random windows from preprocessed data
   - Generate noise: Uniform random values [0, 2π] for num_qubits
   - Forward through quantum generator: IQP encoding → strongly entangled layers → measurements
   - Scale generator output from [-1, 1] to match real data range (×0.1)
   - Critic scores real batch: D(real) via Conv1D network
   - Critic scores fake batch: D(fake) via same network
   - Calculate gradient penalty via interpolation between real/fake
   - Compute Wasserstein loss: E[D(fake)] - E[D(real)] + λ·GP
   - Backprop and optimizer step

2. **Generator Training (1 iteration per epoch):**
   - Sample noise batch: Same distribution as critic training
   - Forward through quantum generator with trainable parameters
   - Critic scores fake batch: D(fake)
   - Compute generator loss: -E[D(fake)] (generator wants to minimize critic)
   - Backprop through gradient flow (PennyLane → PyTorch integration)
   - Optimizer step on params_pqc

3. **Evaluation (per epoch):**
   - Generate full-length synthetic series using trained generator
   - Compute Earth Mover Distance (Wasserstein metric)
   - Calculate stylized facts RMSEs: ACF, volatility clustering, leverage effects
   - Log metrics for convergence monitoring

**State Management:**
- Model parameters: `params_pqc` (quantum circuit weights), critic network weights
- Optimizer states: Adam optimizers for both generator and critic
- Loss history: Lists tracking critic_loss_avg, generator_loss_avg, emd_avg, acf_avg, vol_avg, lev_avg
- Early stopping: Checkpoint saving on improvement, patience counter for convergence detection

## Key Abstractions

**qGAN Class:**
- Purpose: Unified container for hybrid quantum-classical GAN
- Examples: `qgan_pennylane.ipynb`, `archive/qgan_pennylane.py`
- Pattern: PyTorch nn.Module subclass combining quantum device (PennyLane) and classical network (PyTorch)

**Quantum Circuit Components:**
- `encoding_layer()`: IQP encoding with RZ rotations for noise input
- `define_generator_circuit()`: Multi-layer strongly entangled circuit with parameterized rotations
- `count_params()`: Static parameter counting logic (IQP + rotations + ZZ interactions)
- Pattern: PennyLane QNode decorators with measurement specifications

**Discriminator (Critic) Network:**
- Architecture: Conv1D(1→64) → Conv1D(64→128) → Conv1D(128→128) → AdaptiveAvgPool1d → Linear(128→32) → Linear(32→1)
- Pattern: Sequential convolution stack with LeakyReLU(0.1) and dropout
- Input shape: [batch_size, 1, window_length] (1D time series)
- Output: Single scalar score per sample

**Data Transformations:**
- Lambert W Transform: Removes excess kurtosis from log-returns
- Normalization: Min-max scaling to [-1, 1]
- Rolling Window: Stride-2 windows for temporal structure preservation
- Pattern: NumPy/PyTorch operations in preprocessing pipeline

## Entry Points

**Main Notebook (`qgan_pennylane.ipynb`):**
- Location: `/Users/shawngibford/dev/phd/qGAN/qgan_pennylane.ipynb`
- Triggers: User execution of cells in sequence
- Responsibilities: Data loading, preprocessing, model initialization, training orchestration, evaluation, visualization

**Training Execution:**
- Location: `qGAN.train_qgan()` method
- Triggers: Called from notebook after model instantiation
- Responsibilities: Main training loop over epochs, coordinator for critic/generator updates, metric computation

**Phase-Specific Notebooks:**
- `qgan_pennylane_qutrit_phase2c.ipynb`: Latest qutrit implementation (3 qutrits, 1 layer, 57 parameters)
- `qgan_pennylane_qutrit_phase2b.ipynb`: Previous qutrit variant (4 qutrits, 1 layer)
- `qgan_pennylane_qutrit_phase2.ipynb`: Original qutrit phase
- Purpose: Different quantum circuit architectures and qubit/qutrit configurations

**Checkpoint Loading:**
- Location: `load_checkpoint_phase2c.py`
- Triggers: Called to restore trained model from checkpoints
- Responsibilities: Load saved parameter tensors and optimizer states, restore model for inference or continued training

## Error Handling

**Strategy:** Try-catch with manual validation and early stopping

**Patterns:**
- Device fallback: CUDA → MPS → CPU for hardware compatibility
- Type conversion handling: Explicit conversion between PyTorch tensor types and PennyLane outputs
- Gradient flow validation: Checks that generated samples properly propagate gradients through quantum circuit
- Data validation: Type checking for generated samples (list, tuple, tensor conversion)
- Early stopping: Monitor discriminator loss with patience counter, save best checkpoint on improvement

## Cross-Cutting Concerns

**Logging:** Print statements tracking epoch progress, loss values, improvement status, model restoration steps

**Validation:** Seed initialization (random, numpy, torch) for reproducibility; min-max scaling bounds checking

**Authentication:** Not applicable (no external services)

**Device Management:** PyTorch device selection (cuda/mps/cpu), PennyLane quantum device initialization (default.qubit or default.qutrit)

---

*Architecture analysis: 2026-02-26*
