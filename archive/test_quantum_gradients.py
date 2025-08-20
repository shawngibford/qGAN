#!/usr/bin/env python3
"""
Quick test to verify quantum generator gradient flow
"""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

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

def test_quantum_gradients():
    print("🧪 Testing Quantum Generator Gradient Flow")
    
    # Create generator
    generator = QuantumGenerator(n_qubits=5, n_layers=3, output_dim=10)
    print(f"Total quantum parameters: {generator.n_params}")
    print(f"Parameter shape: {generator.params.shape}")
    
    # Create test input
    batch_size = 4
    input_dim = 5
    test_input = torch.randn(batch_size, input_dim)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # Forward pass
    output = generator(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Create a simple loss (sum of all outputs)
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
        
        print(f"\nParameters with finite gradients: {len(grad_norms)} / {generator.n_params}")
        if grad_norms:
            print(f"Gradient norm range: [{min(grad_norms):.8f}, {max(grad_norms):.8f}]")
            print(f"Average gradient norm: {np.mean(grad_norms):.8f}")
            print(f"Non-zero gradients: {sum(1 for g in grad_norms if g > 1e-10)}")
        
        # Check if all parameters got gradients
        if len(grad_norms) == generator.n_params:
            print("✅ SUCCESS: All quantum parameters received gradients!")
        else:
            print(f"❌ PROBLEM: Only {len(grad_norms)}/{generator.n_params} parameters got gradients")
    else:
        print("❌ CRITICAL: No gradients computed at all!")
    
    # Test parameter update
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
    
    # Store original parameters
    original_params = generator.params.data.clone()
    
    # Take optimizer step
    optimizer.step()
    
    # Check if parameters changed
    param_changes = torch.abs(generator.params.data - original_params)
    changed_params = (param_changes > 1e-8).sum().item()
    
    print(f"\nParameters that changed after optimizer step: {changed_params} / {generator.n_params}")
    if changed_params == generator.n_params:
        print("✅ SUCCESS: All parameters updated!")
    else:
        print(f"❌ PROBLEM: Only {changed_params} parameters were updated")
    
    print(f"Max parameter change: {param_changes.max().item():.8f}")
    print(f"Average parameter change: {param_changes.mean().item():.8f}")

if __name__ == "__main__":
    test_quantum_gradients() 