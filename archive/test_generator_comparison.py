#!/usr/bin/env python3
"""
Generator Comparison Test: Quantum vs MLP
Tests gradient flow, time dependency preservation, and training dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from torch.autograd import grad
import pandas as pd
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ==================== DATA PREPARATION ====================
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

# ==================== MLP GENERATOR ====================
class MLPGenerator(nn.Module):
    def __init__(self, input_dim=5, output_dim=10, hidden_dims=[64, 32]):
        super(MLPGenerator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ==================== QUANTUM GENERATOR ====================
class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits=5, n_layers=3, output_dim=10):
        super(QuantumGenerator, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim
        
        # Calculate number of parameters
        self.n_params = n_layers * n_qubits * 3  # 3 rotations per qubit per layer
        self.params = nn.Parameter(torch.randn(self.n_params) * 0.5)
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Create quantum circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, params):
            # Encode inputs
            for i, inp in enumerate(inputs):
                if i < n_qubits:
                    qml.RY(inp * np.pi, wires=i)
            
            # Parameterized layers
            param_idx = 0
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[param_idx], wires=qubit)
                    qml.RY(params[param_idx + 1], wires=qubit)
                    qml.RZ(params[param_idx + 2], wires=qubit)
                    param_idx += 3
                
                # Entangling gates
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, output_dim))]
        
        self.quantum_circuit = quantum_circuit
        
    def forward(self, x):
        # Process entire batch at once to maintain gradient flow
        batch_size = x.shape[0]
        
        # Prepare inputs for the entire batch
        inputs_batch = []
        for i in range(batch_size):
            # Pad or truncate input to match n_qubits
            inp = x[i].float()  # Ensure float32
            if len(inp) > self.n_qubits:
                inp = inp[:self.n_qubits]
            elif len(inp) < self.n_qubits:
                inp = torch.cat([inp, torch.zeros(self.n_qubits - len(inp))])
            inputs_batch.append(inp)
        
        # Stack all inputs - this maintains gradient flow
        inputs_tensor = torch.stack(inputs_batch)
        
        # Process each input but keep gradients flowing
        results = []
        for i in range(batch_size):
            # Each circuit call must maintain connection to self.params
            result = self.quantum_circuit(inputs_tensor[i], self.params)
            
            # Convert to tensor and ensure proper shape
            if isinstance(result, list):
                result = torch.stack(result)
            
            # Pad or truncate output to match output_dim
            if len(result) < self.output_dim:
                padding = torch.zeros(self.output_dim - len(result))
                result = torch.cat([result, padding])
            else:
                result = result[:self.output_dim]
            
            results.append(result)
        
        # Stack results - this preserves gradients for all parameters
        output = torch.stack(results).float()
        
        return output

# ==================== SIMPLE CRITIC ====================
class Critic(nn.Module):
    def __init__(self, input_dim=10):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# ==================== GRADIENT ANALYSIS ====================
def analyze_gradients(model, loss, model_name):
    """Analyze gradient flow through the model"""
    gradients = []
    grad_norms = []
    
    # Check if this is a quantum generator (has single params tensor)
    is_quantum = hasattr(model, 'params') and hasattr(model, 'n_params')
    
    if is_quantum:
        # For quantum generator, analyze individual parameter gradients
        if model.params.grad is not None:
            individual_grads = model.params.grad
            for i, grad_val in enumerate(individual_grads):
                if not (torch.isnan(grad_val) or torch.isinf(grad_val)):
                    grad_norm = abs(grad_val.item())
                    gradients.append((f"quantum_param_{i}", grad_norm))
                    grad_norms.append(grad_norm)
        
        print(f"\n=== {model_name} GRADIENT ANALYSIS ===")
        print(f"Total parameters: {model.n_params}")
        print(f"Parameters with gradients: {len(grad_norms)}")
        if grad_norms:
            print(f"Average gradient norm: {np.mean(grad_norms):.8f}")
            print(f"Max gradient norm: {np.max(grad_norms):.8f}")
            print(f"Min gradient norm: {np.min(grad_norms):.8f}")
            print(f"Gradient std: {np.std(grad_norms):.8f}")
            print(f"Non-zero gradients: {sum(1 for g in grad_norms if g > 1e-10)}")
        else:
            print("❌ No gradients found!")
    else:
        # For regular models, use original logic
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients.append((name, grad_norm))
                grad_norms.append(grad_norm)
        
        print(f"\n=== {model_name} GRADIENT ANALYSIS ===")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Parameters with gradients: {len(grad_norms)}")
        if grad_norms:
            print(f"Average gradient norm: {np.mean(grad_norms):.8f}")
            print(f"Max gradient norm: {np.max(grad_norms):.8f}")
            print(f"Min gradient norm: {np.min(grad_norms):.8f}")
            print(f"Gradient std: {np.std(grad_norms):.8f}")
    
    # Check for exploding/vanishing gradients
    if grad_norms:
        if np.max(grad_norms) > 10:
            print("⚠️  WARNING: Possible exploding gradients!")
        elif np.max(grad_norms) < 1e-7:
            print("⚠️  WARNING: Possible vanishing gradients!")
        else:
            print("✅ Gradients appear healthy")
    
    return gradients, grad_norms

# ==================== TIME DEPENDENCY ANALYSIS ====================
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
    
    print("\n=== TIME DEPENDENCY ANALYSIS ===")
    
    # 1. Autocorrelation comparison
    real_acf = compute_autocorrelation(real_data)
    gen_acf = compute_autocorrelation(generated_data)
    
    real_acf_mean = np.mean(real_acf, axis=0)
    gen_acf_mean = np.mean(gen_acf, axis=0)
    
    acf_mse = np.mean((real_acf_mean - gen_acf_mean)**2)
    
    print(f"Real data autocorrelation: {real_acf_mean}")
    print(f"Generated autocorrelation: {gen_acf_mean}")
    print(f"ACF MSE: {acf_mse:.6f}")
    
    # 2. Volatility clustering
    real_vol = compute_volatility_clustering(real_data)
    gen_vol = compute_volatility_clustering(generated_data)
    
    vol_corr = np.corrcoef(real_vol, gen_vol)[0, 1] if len(real_vol) > 1 and len(gen_vol) > 1 else 0
    
    print(f"Real volatility mean/std: {np.mean(real_vol):.4f} / {np.std(real_vol):.4f}")
    print(f"Generated volatility mean/std: {np.mean(gen_vol):.4f} / {np.std(gen_vol):.4f}")
    print(f"Volatility correlation: {vol_corr:.4f}")
    
    # 3. Sequential dependency (lag-1 correlation)
    def lag_correlation(data):
        lag_corrs = []
        for series in data:
            if len(series) > 1:
                lag_corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                if not np.isnan(lag_corr):
                    lag_corrs.append(lag_corr)
        return np.array(lag_corrs)
    
    real_lag_corr = lag_correlation(real_data)
    gen_lag_corr = lag_correlation(generated_data)
    
    print(f"Real lag-1 correlation: {np.mean(real_lag_corr):.4f} ± {np.std(real_lag_corr):.4f}")
    print(f"Generated lag-1 correlation: {np.mean(gen_lag_corr):.4f} ± {np.std(gen_lag_corr):.4f}")
    
    return {
        'acf_mse': acf_mse,
        'vol_correlation': vol_corr,
        'real_lag_corr': np.mean(real_lag_corr),
        'gen_lag_corr': np.mean(gen_lag_corr)
    }

# ==================== TRAINING FUNCTION ====================
def train_generator(generator, critic, real_data, generator_type, epochs=50, lr_g=1e-3, lr_c=1e-4):
    """Train generator and return metrics"""
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.999))
    
    # Convert data to tensors
    real_tensor = torch.FloatTensor(real_data).float()
    
    losses_g = []
    losses_c = []
    gradient_norms_g = []
    gradient_norms_c = []
    
    print(f"\n=== TRAINING {generator_type} GENERATOR ===")
    
    for epoch in range(epochs):
        # Train Critic
        optimizer_c.zero_grad()
        
        # Generate noise for input
        batch_size = min(32, len(real_data))
        noise = torch.randn(batch_size, 5)  # 5D noise input
        
        # Generate fake data
        with torch.no_grad():
            fake_data = generator(noise)
        
        # Get batch of real data
        idx = torch.randperm(len(real_data))[:batch_size]
        real_batch = real_tensor[idx]
        
        # Critic losses
        real_scores = critic(real_batch)
        fake_scores = critic(fake_data)
        
        # Wasserstein loss
        critic_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        
        # Gradient penalty (simplified)
        alpha = torch.rand(batch_size, 1)
        interpolated = alpha * real_batch + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = critic(interpolated)
        gradients = grad(outputs=d_interpolated, inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interpolated),
                        create_graph=True, retain_graph=True)[0]
        
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1)**2)
        
        critic_total_loss = critic_loss + 10 * gradient_penalty
        critic_total_loss.backward()
        optimizer_c.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        
        fake_data = generator(noise)
        fake_scores = critic(fake_data)
        generator_loss = -torch.mean(fake_scores)
        
        generator_loss.backward()
        
        # Analyze gradients before step
        if epoch % 10 == 0:
            grad_info_g, grad_norms_g = analyze_gradients(generator, generator_loss, f"{generator_type} Generator")
            grad_info_c, grad_norms_c = analyze_gradients(critic, critic_total_loss, "Critic")
            gradient_norms_g.append(np.mean(grad_norms_g))
            gradient_norms_c.append(np.mean(grad_norms_c))
        
        optimizer_g.step()
        
        losses_g.append(generator_loss.item())
        losses_c.append(critic_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: G_loss={generator_loss.item():.4f}, C_loss={critic_loss.item():.4f}")
    
    return {
        'generator': generator,
        'losses_g': losses_g,
        'losses_c': losses_c,
        'gradient_norms_g': gradient_norms_g,
        'gradient_norms_c': gradient_norms_c
    }

# ==================== MAIN COMPARISON TEST ====================
def run_comparison_test():
    """Run the complete comparison test"""
    print("🚀 Starting Generator Comparison Test")
    
    # 1. Generate synthetic data
    window_length = 10
    real_data = generate_synthetic_timeseries(n_samples=500, window_length=window_length)
    print(f"Generated {len(real_data)} time series windows of length {window_length}")
    
    # 2. Initialize models
    mlp_generator = MLPGenerator(input_dim=5, output_dim=window_length, hidden_dims=[64, 32])
    quantum_generator = QuantumGenerator(n_qubits=5, n_layers=3, output_dim=window_length)
    
    print(f"MLP Generator parameters: {sum(p.numel() for p in mlp_generator.parameters())}")
    print(f"Quantum Generator parameters: {sum(p.numel() for p in quantum_generator.parameters())}")
    
    # 3. Train both generators
    critic_mlp = Critic(input_dim=window_length)
    critic_quantum = Critic(input_dim=window_length)
    
    mlp_results = train_generator(mlp_generator, critic_mlp, real_data, "MLP", epochs=50)
    quantum_results = train_generator(quantum_generator, critic_quantum, real_data, "QUANTUM", epochs=50)
    
    # 4. Generate samples for comparison
    test_noise = torch.randn(100, 5)
    
    with torch.no_grad():
        mlp_samples = mlp_results['generator'](test_noise).numpy()
        quantum_samples = quantum_results['generator'](test_noise).numpy()
    
    # 5. Analyze time dependency preservation
    print("\n" + "="*50)
    print("MLP GENERATOR TIME DEPENDENCY:")
    mlp_time_metrics = analyze_time_dependency(real_data[:100], mlp_samples)
    
    print("\n" + "="*50) 
    print("QUANTUM GENERATOR TIME DEPENDENCY:")
    quantum_time_metrics = analyze_time_dependency(real_data[:100], quantum_samples)
    
    # 6. Plot results
    plot_comparison_results(real_data, mlp_samples, quantum_samples, 
                          mlp_results, quantum_results, mlp_time_metrics, quantum_time_metrics)
    
    return {
        'mlp_results': mlp_results,
        'quantum_results': quantum_results,
        'mlp_time_metrics': mlp_time_metrics,
        'quantum_time_metrics': quantum_time_metrics,
        'real_data': real_data,
        'mlp_samples': mlp_samples,
        'quantum_samples': quantum_samples
    }

def plot_comparison_results(real_data, mlp_samples, quantum_samples, 
                          mlp_results, quantum_results, mlp_time, quantum_time):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. Sample time series
    axes[0,0].plot(real_data[0], 'b-', label='Real', alpha=0.7)
    axes[0,0].plot(mlp_samples[0], 'r--', label='MLP', alpha=0.7)
    axes[0,0].plot(quantum_samples[0], 'g:', label='Quantum', alpha=0.7)
    axes[0,0].set_title('Sample Time Series')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. Distribution comparison
    axes[0,1].hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0,1].hist(mlp_samples.flatten(), bins=50, alpha=0.5, label='MLP', density=True)
    axes[0,1].hist(quantum_samples.flatten(), bins=50, alpha=0.5, label='Quantum', density=True)
    axes[0,1].set_title('Value Distributions')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3. Training losses
    axes[0,2].plot(mlp_results['losses_g'], 'r-', label='MLP Generator')
    axes[0,2].plot(quantum_results['losses_g'], 'g-', label='Quantum Generator')
    axes[0,2].set_title('Generator Losses')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 4. Gradient norms
    if mlp_results['gradient_norms_g']:
        axes[1,0].plot(mlp_results['gradient_norms_g'], 'r-', label='MLP')
    if quantum_results['gradient_norms_g']:
        axes[1,0].plot(quantum_results['gradient_norms_g'], 'g-', label='Quantum')
    axes[1,0].set_title('Generator Gradient Norms')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 5. Autocorrelation comparison
    real_acf = [acf(series, nlags=5, fft=False)[1:] for series in real_data[:20]]
    mlp_acf = [acf(series, nlags=5, fft=False)[1:] for series in mlp_samples[:20]]
    quantum_acf = [acf(series, nlags=5, fft=False)[1:] for series in quantum_samples[:20]]
    
    axes[1,1].plot(np.mean(real_acf, axis=0), 'b-o', label='Real')
    axes[1,1].plot(np.mean(mlp_acf, axis=0), 'r--s', label='MLP')
    axes[1,1].plot(np.mean(quantum_acf, axis=0), 'g:^', label='Quantum')
    axes[1,1].set_title('Average Autocorrelation')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 6. Variance over time
    axes[1,2].plot(np.var(real_data, axis=0), 'b-', label='Real')
    axes[1,2].plot(np.var(mlp_samples, axis=0), 'r--', label='MLP')
    axes[1,2].plot(np.var(quantum_samples, axis=0), 'g:', label='Quantum')
    axes[1,2].set_title('Variance by Time Step')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    # 7. Summary metrics
    metrics_text = f"""
    TIME DEPENDENCY METRICS:
    
    ACF MSE:
    MLP: {mlp_time['acf_mse']:.6f}
    Quantum: {quantum_time['acf_mse']:.6f}
    
    Lag-1 Correlation:
    Real: {mlp_time['real_lag_corr']:.4f}
    MLP: {mlp_time['gen_lag_corr']:.4f}
    Quantum: {quantum_time['gen_lag_corr']:.4f}
    
    Volatility Correlation:
    MLP: {mlp_time['vol_correlation']:.4f}
    Quantum: {quantum_time['vol_correlation']:.4f}
    """
    
    axes[2,0].text(0.1, 0.1, metrics_text, transform=axes[2,0].transAxes, 
                   fontsize=10, verticalalignment='bottom', fontfamily='monospace')
    axes[2,0].set_xlim(0, 1)
    axes[2,0].set_ylim(0, 1)
    axes[2,0].axis('off')
    axes[2,0].set_title('Summary Metrics')
    
    # 8. Wasserstein distances
    emd_real_mlp = wasserstein_distance(real_data.flatten(), mlp_samples.flatten())
    emd_real_quantum = wasserstein_distance(real_data.flatten(), quantum_samples.flatten())
    
    axes[2,1].bar(['MLP vs Real', 'Quantum vs Real'], [emd_real_mlp, emd_real_quantum])
    axes[2,1].set_title('Earth Mover\'s Distance to Real Data')
    axes[2,1].grid(True)
    
    # 9. Parameter statistics
    with torch.no_grad():
        mlp_params = torch.cat([p.flatten() for p in mlp_results['generator'].parameters()])
        quantum_params = quantum_results['generator'].params.flatten()
    
    axes[2,2].hist(mlp_params.detach().numpy(), bins=30, alpha=0.5, label='MLP', density=True)
    axes[2,2].hist(quantum_params.detach().numpy(), bins=30, alpha=0.5, label='Quantum', density=True)
    axes[2,2].set_title('Parameter Distributions')
    axes[2,2].legend()
    axes[2,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('generator_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = run_comparison_test()
    print("\n🎉 Comparison test completed! Check 'generator_comparison.png' for detailed plots.") 