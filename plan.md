# QGAN Synthetic Data Generation Project

## Notes
- All tensor-to-numpy conversion errors (requires_grad) are fixed by using .detach().cpu().numpy().
- All TensorFlow code replaced with PyTorch equivalents.
- Variable naming for generated data is now consistently fake_OD_log_delta_np.
- DataFrames real_data and fake_data are created with DATE and Log_Return columns for downstream analysis and plotting.
- Added missing import for entropy from scipy.stats.
- The script now runs to completion, producing all visualizations and CSV outputs.
- Seaborn's distplot is deprecated; warnings are non-blocking but should be updated in future.
- requirements.txt, README.md, and ALGORITHM.md (scientific algorithm description) have been generated and saved in the project directory.
- Generator is learning (parameters update, losses evolve), but post-training generation pipeline has issues (blank plots, possible data/processing bugs). Debugging and fixes required for meaningful synthetic outputs.
- Quantum expressivity may be insufficient: current PQC is only alternating rotation + CNOT; consider more expressive ansatz.
- Parameter imbalance: ~75 quantum parameters vs ~390K discriminator parameters leads to learning imbalance; need strategies for balancing.
- Gradient flow may be broken between quantum circuit and PyTorch autograd; ensure end-to-end differentiability.
- Sample-by-sample quantum processing prevents batch optimization; implement batch processing for quantum parts if possible.
- Preprocessing (rescale/Lambert W/denormalization) and variable scope issues can cause generation bugs; verify each step.

## Task List
- [x] Fix tensor-to-numpy conversion errors throughout the code.
- [x] Replace all TensorFlow code with PyTorch equivalents.
- [x] Ensure variable naming consistency for all generated data variables.
- [x] Create real_data and fake_data DataFrames with correct structure.
- [x] Add missing imports (e.g., entropy).
- [x] Run and verify full script execution and outputs.
- [x] Generate requirements.txt from all code imports and usage.
- [x] Write README.md with installation, venv, usage, and project description.
- [x] Write scientific algorithm description suitable for publication.
- [x] Debug and fix post-training generation pipeline to ensure meaningful synthetic data and plots (diagnose variable scope, data range, preprocessing, and plotting logic).
- [ ] Improve quantum circuit expressivity (try deeper/more complex ansatz, e.g., entangling layers, more rotation gates).
- [ ] Address parameter imbalance (e.g., regularization, discriminator simplification, or generator parameter increase).
- [ ] Ensure gradient flow is preserved between quantum circuit and PyTorch (no tensor breaks, avoid .detach() in forward pass).
- [ ] Implement batch processing for quantum generator if possible.
- [ ] Audit and robustly test preprocessing (rescale, Lambert W, denormalization) and variable scope in generation code.

## Current Goal
Address QGAN learning and architecture limitations for robust training.

## 📋 Phase 1: Quantum Circuit Enhancement (High Priority)

### 1.1 Improve Quantum Expressivity
- **Current Issue**: Simple alternating RX/RY/RZ + CNOT may be insufficient
- **Solutions**:
  - Implement Hardware Efficient Ansatz with more entangling layers
  - Add Strongly Entangling Layers (all-to-all connectivity)
  - Try QAOA-inspired circuits with problem-specific structure
  - Experiment with Variational Quantum Eigensolver (VQE) style ansatz

### 1.2 Circuit Architecture Experiments
- **Deeper circuits**: Increase layers from 3 to 6-10
- **More rotation gates**: Add RZ rotations between entangling layers
- **Different entangling patterns**: Ring, all-to-all, brick-wall patterns
- **Parameter re-uploading**: Multiple data encoding layers

## 📋 Phase 2: Parameter Balance & Training Dynamics (Critical)

### 2.1 Address Parameter Imbalance (75 vs 390K)
- **Generator Enhancement**:
  - Increase qubits from 5 to 8-12 (exponential parameter growth)
  - Add classical post-processing layers after quantum circuit
  - Implement hybrid quantum-classical generator
- **Discriminator Simplification**:
  - Reduce discriminator complexity (fewer layers/parameters)
  - Use spectral normalization to stabilize training
  - Implement progressive growing (start simple, add complexity)

### 2.2 Training Balance Strategies
- **Learning rate scheduling**: Different rates for G/D
- **Training frequency**: Train generator more often (n_gen > n_critic)
- **Gradient penalty tuning**: Adjust LAMBDA parameter
- **Loss function experiments**: Try different GAN variants (LSGAN, WGAN-GP variants)

## 📋 Phase 3: Gradient Flow & Batch Processing (Technical)

### 3.1 Ensure End-to-End Differentiability
- **Audit gradient flow**: Remove any .detach() calls in forward pass
- **PennyLane integration**: Verify autograd compatibility
- **Gradient monitoring**: Add gradient norm tracking
- **Backpropagation validation**: Test gradients reach quantum parameters

### 3.2 Implement Batch Processing
- **Vectorized quantum circuits**: Use PennyLane's batch execution
- **Parallel circuit evaluation**: Multiple quantum devices if available
- **Memory optimization**: Batch size tuning for quantum circuits
- **Hybrid batching**: Classical batching + quantum vectorization

## 📋 Phase 4: Preprocessing & Pipeline Robustness (Maintenance)

### 4.1 Preprocessing Pipeline Audit
- **Lambert W transform**: Verify mathematical correctness
- **Rescaling function**: Test edge cases and numerical stability
- **Denormalization**: Ensure proper inverse operations
- **Data validation**: Add comprehensive checks at each step

### 4.2 Variable Scope & Code Quality
- **Refactor generation code**: Clean separation of concerns
- **Add unit tests**: Test each preprocessing function independently
- **Error handling**: Robust exception handling throughout pipeline
- **Documentation**: Clear docstrings for all functions

## 📋 Phase 5: Advanced Optimization Techniques (Future)

### 5.1 Training Enhancements
- **Curriculum learning**: Start with simpler patterns, increase complexity
- **Multi-scale training**: Train on different time windows simultaneously
- **Regularization techniques**: Add noise, dropout for quantum circuits
- **Ensemble methods**: Multiple quantum generators with different ansatz

### 5.2 Evaluation & Metrics
- **Advanced metrics**: Implement more sophisticated time series metrics
- **Statistical tests**: Kolmogorov-Smirnov, Anderson-Darling tests
- **Financial metrics**: Volatility clustering, fat tails, autocorrelations
- **Benchmark comparisons**: Compare against classical GANs

## 🎯 Implementation Priority Order

### Week 1: Quick Wins
1. **Increase quantum circuit depth** (3→6 layers)
2. **Add more qubits** (5→8 qubits)
3. **Implement gradient flow monitoring**
4. **Test batch processing for quantum circuits**

### Week 2: Architecture Changes
1. **Implement Hardware Efficient Ansatz**
2. **Add classical post-processing to generator**
3. **Reduce discriminator complexity**
4. **Tune training balance (learning rates, frequencies)**

### Week 3: Advanced Features
1. **Implement strongly entangling layers**
2. **Add curriculum learning**
3. **Comprehensive preprocessing audit**
4. **Advanced evaluation metrics**

## 🔧 Success Metrics

- **Generator Loss**: Should decrease and stabilize
- **Discriminator Balance**: Neither should dominate completely
- **Gradient Norms**: Should be non-zero and stable
- **Synthetic Data Quality**: Better statistical properties matching real data
- **Training Stability**: Consistent performance across multiple runs

## 💡 Key Implementation Notes

1. **Start with circuit expressivity** - this is likely the biggest bottleneck
2. **Monitor gradients carefully** - quantum circuits can have vanishing gradients
3. **Test incrementally** - make one change at a time to isolate effects
4. **Keep baseline comparisons** - always compare against current working version
5. **Document everything** - quantum ML debugging is complex