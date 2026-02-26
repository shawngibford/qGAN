# Testing Patterns

**Analysis Date:** 2026-02-26

## Test Framework

**Runner:**
- Not formally configured
- pytest not in requirements.txt (commented out as optional)
- Tests run as standalone Python scripts with direct execution

**Assertion Library:**
- print statements and manual assertions
- torch assertions for numerical validation (e.g., `torch.isnan()`, `torch.isinf()`)

**Run Commands:**
```bash
python archive/test_quantum_gradients.py              # Run quantum gradient tests
python archive/test_generator_comparison.py           # Run comprehensive generator comparison
# Notebooks executed as-is in Jupyter environment
jupyter notebook qgan_pennylane.ipynb                 # Main qGAN implementation
```

## Test File Organization

**Location:**
- Dedicated test files in `archive/` directory for validation
- No co-located test files within main codebase
- Notebooks contain inline validation cells

**Naming:**
- `test_*.py` pattern for standalone test scripts
  - Example: `archive/test_quantum_gradients.py`
  - Example: `archive/test_generator_comparison.py`
- No unit test files alongside source code

**Structure:**
```
archive/
├── test_quantum_gradients.py         # Validate gradient flow
├── test_generator_comparison.py       # Benchmark generators
└── gradient_flow_diagnostic.py        # Diagnostic utilities
```

## Test Structure

**Suite Organization:**
Test scripts function as executable main modules with function-based organization.

From `archive/test_quantum_gradients.py`:
```python
class QuantumGenerator(nn.Module):
    """Test implementation with gradient tracking"""
    def __init__(self, ...): ...
    def forward(self, x): ...

def test_quantum_gradients():
    """Main test function"""
    print("🧪 Testing Quantum Generator Gradient Flow")
    generator = QuantumGenerator(...)
    # Test forward pass
    output = generator(test_input)
    # Test backward pass
    loss.backward()
    # Test optimization step
    optimizer.step()
    # Verify results with print statements and assertions
```

**Patterns:**

1. **Setup Pattern:**
   ```python
   # Set random seeds for reproducibility
   torch.manual_seed(42)
   np.random.seed(42)

   # Create model instances
   generator = QuantumGenerator(n_qubits=5, n_layers=3, output_dim=10)
   ```

2. **Teardown Pattern:**
   - No explicit cleanup needed
   - Models are garbage collected after test completion
   - Checkpoints saved to filesystem during training (side effects)

3. **Assertion Pattern:**
   - Manual checks with print output and conditionals
   - Example from `test_quantum_gradients.py`:
     ```python
     if generator.params.grad is not None:
         print(f"generator.params.grad shape: {generator.params.grad.shape}")
         # Check if all parameters got gradients
         if len(grad_norms) == generator.n_params:
             print("✅ SUCCESS: All quantum parameters received gradients!")
         else:
             print(f"❌ PROBLEM: Only {len(grad_norms)}/{generator.n_params} parameters got gradients")
     else:
         print("❌ CRITICAL: No gradients computed at all!")
     ```

## Mocking

**Framework:**
- No mocking library detected (unittest.mock not in dependencies)
- Actual components tested, no mocking layer

**Patterns:**
- Real data used for testing
  - Example: `generate_synthetic_timeseries()` in `test_generator_comparison.py` creates realistic test data
  - Example from `archive/test_generator_comparison.py`:
    ```python
    def generate_synthetic_timeseries(n_samples=1000, window_length=10):
        """Generate synthetic time series with known temporal patterns"""
        t = np.linspace(0, 4*np.pi, n_samples)
        trend = 0.001 * t
        seasonal = 0.05 * np.sin(t) + 0.03 * np.cos(2*t)
        noise = 0.02 * np.random.randn(n_samples)
        # ... combine components
        return np.array(windowed_data)
    ```

**What to Mock:**
- Nothing is mocked in current test suite
- Tests exercise full pipeline with real tensors and models

**What NOT to Mock:**
- Quantum circuits (use PennyLane default.qubit backend - simple but real)
- Neural network modules (test actual PyTorch layers)
- Optimizers (use actual Adam optimizer)
- Loss calculations (compute real metrics)

## Fixtures and Factories

**Test Data:**
Generated on-the-fly within test functions. Example from `archive/test_generator_comparison.py`:

```python
def generate_synthetic_timeseries(n_samples=1000, window_length=10):
    """Generate synthetic time series with known temporal patterns"""
    t = np.linspace(0, 4*np.pi, n_samples)

    # Create a time series with multiple components
    trend = 0.001 * t
    seasonal = 0.05 * np.sin(t) + 0.03 * np.cos(2*t)
    noise = 0.02 * np.random.randn(n_samples)

    # Add AR(1) component for temporal dependency
    ar_component = np.zeros(n_samples)
    ar_component[0] = noise[0]
    for i in range(1, n_samples):
        ar_component[i] = 0.7 * ar_component[i-1] + noise[i]

    series = trend + seasonal + ar_component

    # Create windowed data
    windowed_data = []
    for i in range(len(series) - window_length + 1):
        windowed_data.append(series[i:i+window_length])

    return np.array(windowed_data)
```

**Location:**
- Factories are module-level functions, not in separate fixture file
- Example: `archive/test_generator_comparison.py` contains `MLPGenerator`, `QuantumGenerator`, `Critic` as factory classes
- Real data loaded from CSV files in notebooks (hardcoded paths to data.csv)

**Seed Management:**
- Seeds set at module initialization for reproducibility
  ```python
  # Set random seeds
  torch.manual_seed(42)
  np.random.seed(42)
  ```
- Consistent seeds ensure reproducible test results

## Coverage

**Requirements:**
- No coverage requirements enforced
- No coverage tool configured (coverage.py, pytest-cov not in dependencies)

**View Coverage:**
- Not applicable - no coverage tooling present

## Test Types

**Unit Tests:**
- Implicit in gradient flow tests
- `test_quantum_gradients()` validates individual quantum parameter updates
  - Checks each parameter receives gradients
  - Verifies parameter values change after optimizer step
  - Scope: Single QuantumGenerator module with batch processing

**Integration Tests:**
- `test_generator_comparison()` is comprehensive integration test
- Tests two complete generator implementations (MLP vs Quantum)
- Components: generator → critic → loss → optimizer chain
- Scope: Full training pipeline with real data

**End-to-End Tests:**
- Entire training loops in Jupyter notebooks
- Real data from CSV files
- Output: trained models, visualizations, metrics
- Validation through manual inspection of plots and loss curves

**Manual Testing Approach:**
- Notebooks include extensive visualization cells
- Early stopping implementation (`early_stopping_code.py`) provides checkpoint/recovery mechanism
- Training history plots show convergence visually
- Discriminator loss monitoring indicates model learning

## Common Patterns

**Async Testing:**
- Not applicable - synchronous PyTorch execution
- No async/await patterns in codebase

**Gradient Testing:**
Pattern for validating gradient flow in quantum circuits (from `archive/test_quantum_gradients.py`):

```python
# Forward pass
output = generator(test_input)
loss = output.sum()
print(f"Loss: {loss.item():.6f}")

# Backward pass
loss.backward()

# Check gradients
print(f"\n🔍 GRADIENT ANALYSIS:")
print(f"generator.params.grad shape: {generator.params.grad.shape}")
print(f"generator.params.grad is not None: {generator.params.grad is not None}")

if generator.params.grad is not None:
    grad_norms = []
    for i, grad_val in enumerate(generator.params.grad):
        if torch.isnan(grad_val) or torch.isinf(grad_val):
            print(f"⚠️  Parameter {i}: gradient is {grad_val}")
        else:
            grad_norms.append(abs(grad_val.item()))

    print(f"Parameters with finite gradients: {len(grad_norms)} / {generator.n_params}")
    if grad_norms:
        print(f"Gradient norm range: [{min(grad_norms):.8f}, {max(grad_norms):.8f}]")
        print(f"Average gradient norm: {np.mean(grad_norms):.8f}")
```

**Error/Exception Testing:**
Pattern from `archive/test_quantum_gradients.py` - checking for problematic conditions:

```python
# Test parameter update
optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
original_params = generator.params.data.clone()
optimizer.step()
param_changes = torch.abs(generator.params.data - original_params)
changed_params = (param_changes > 1e-8).sum().item()

if changed_params == generator.n_params:
    print("✅ SUCCESS: All parameters updated!")
else:
    print(f"❌ PROBLEM: Only {changed_params} parameters were updated")
```

**Time Dependency Analysis Pattern:**
From `archive/test_generator_comparison.py` - validating temporal properties:

```python
def analyze_time_dependency(real_data, generated_data, window_length=10):
    """Analyze temporal patterns preservation"""

    def compute_autocorrelation(data, max_lags=5):
        """Compute autocorrelation for each time series"""
        autocorrs = []
        for series in data:
            if len(series) > max_lags:
                acf_values = acf(series, nlags=max_lags, fft=False)
                autocorrs.append(acf_values[1:])  # Skip lag 0
        return np.array(autocorrs)

    def compute_volatility_clustering(data):
        """Measure volatility clustering (GARCH-like behavior)"""
        volatilities = []
        for series in data:
            returns = np.diff(series)
            if len(returns) > 1:
                vol = np.std(returns)
                volatilities.append(vol)
        return np.array(volatilities)

    # Compare properties between real and generated data
    real_acf = compute_autocorrelation(real_data)
    gen_acf = compute_autocorrelation(generated_data)
    acf_mse = np.mean((real_acf_mean - gen_acf_mean)**2)

    real_vol = compute_volatility_clustering(real_data)
    gen_vol = compute_volatility_clustering(generated_data)
    vol_corr = np.corrcoef(real_vol, gen_vol)[0, 1]

    return {
        'acf_mse': acf_mse,
        'vol_correlation': vol_corr,
        'real_lag_corr': np.mean(real_lag_corr),
        'gen_lag_corr': np.mean(gen_lag_corr)
    }
```

## Validation Approach

**In Notebooks:**
- Training loss curves plotted to verify convergence
- Generated samples visualized against real data
- Statistical properties (mean, std, skewness, kurtosis) computed and displayed
- Early stopping checkpoints saved for model recovery

**Checkpoint/Recovery Testing:**
- `early_stopping_code.py` provides checkpoint save/load mechanism
- `load_checkpoint_phase2c.py` demonstrates recovery from saved state
- Testing done by: loading checkpoint → verifying parameters restored → resuming training

**Metrics Tracked:**
- Discriminator loss (critic loss)
- Generator loss
- Gradient norms per parameter
- Autocorrelation preservation (time dependency)
- Earth Mover's Distance (Wasserstein distance) to real data
- Distribution statistics (mean, std, skew, kurtosis)

---

*Testing analysis: 2026-02-26*
