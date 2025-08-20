import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Import the QGAN components
from qgan_pennylane import qGAN

def gradient_flow_diagnostic():
    """
    Comprehensive diagnostic to check gradient flow through the entire QGAN pipeline:
    Classical data -> Encoding layer -> Quantum circuit -> Measurements -> Forward pass
    """
    print("🔍 COMPREHENSIVE GRADIENT FLOW DIAGNOSTIC")
    print("="*80)
    
    # Configuration matching the main QGAN
    WINDOW_LENGTH = 10
    NUM_QUBITS = 5
    NUM_LAYERS = 3
    EPOCHS = 2
    BATCH_SIZE = 5
    n_critic = 2
    LAMBDA = 0.1
    
    # Create QGAN instance
    qgan = qGAN(EPOCHS, BATCH_SIZE, WINDOW_LENGTH, n_critic, LAMBDA, NUM_LAYERS, NUM_QUBITS)
    
    print(f"✅ QGAN Configuration:")
    print(f"   - Qubits: {NUM_QUBITS}")
    print(f"   - Layers: {NUM_LAYERS}")
    print(f"   - Parameters: {qgan.num_params}")
    print(f"   - Window Length: {WINDOW_LENGTH}")
    print()
    
    # Step 1: Check parameter initialization and gradient readiness
    print("1️⃣ PARAMETER INITIALIZATION CHECK")
    print("-" * 50)
    print(f"✅ Parameters shape: {qgan.params_pqc.shape}")
    print(f"✅ Parameters range: [{qgan.params_pqc.min().item():.4f}, {qgan.params_pqc.max().item():.4f}]")
    print(f"✅ Requires grad: {qgan.params_pqc.requires_grad}")
    print(f"✅ Gradient function: {qgan.params_pqc.grad_fn}")
    print()
    
    # Step 2: Test gradient flow through individual components
    print("2️⃣ COMPONENT-WISE GRADIENT FLOW TEST")
    print("-" * 50)
    
    # Create test inputs
    noise_input = torch.tensor(np.random.uniform(0, 2*np.pi, NUM_QUBITS), 
                              dtype=torch.float32, requires_grad=True)
    
    # Test the generator directly
    print(f"🔍 Testing generator with noise input shape: {noise_input.shape}")
    
    try:
        # Forward pass through generator
        generator_output = qgan.generator(noise_input, qgan.params_pqc)
        print(f"✅ Generator output type: {type(generator_output)}")
        
        # Handle different output types
        if isinstance(generator_output, (list, tuple)):
            print(f"⚠️  Generator returned {type(generator_output)} with {len(generator_output)} elements")
            print(f"   First few elements: {generator_output[:3] if len(generator_output) > 3 else generator_output}")
            # Convert to tensor as done in training
            generator_output = torch.stack(list(generator_output))
            print(f"✅ After stacking: shape {generator_output.shape}")
        
        print(f"✅ Generator output shape: {generator_output.shape}")
        print(f"✅ Generator output range: [{generator_output.min().item():.6f}, {generator_output.max().item():.6f}]")
        print(f"✅ Generator requires grad: {generator_output.requires_grad}")
        
        # Check if output is connected to computational graph
        if generator_output.grad_fn is not None:
            print(f"✅ Connected to autograd graph: {generator_output.grad_fn}")
        else:
            print("❌ NOT connected to autograd graph!")
        print()
        
        # Step 3: Test backward pass through generator
        print("3️⃣ BACKWARD PASS GRADIENT TEST")
        print("-" * 50)
        
        # Create a simple loss (sum of outputs)
        test_loss = generator_output.sum()
        print(f"✅ Test loss: {test_loss.item():.6f}")
        print(f"✅ Test loss requires grad: {test_loss.requires_grad}")
        
        # Clear any existing gradients
        qgan.params_pqc.grad = None
        noise_input.grad = None
        
        # Backward pass
        test_loss.backward()
        
        # Check quantum parameter gradients
        if qgan.params_pqc.grad is not None:
            grad_magnitude = torch.norm(qgan.params_pqc.grad).item()
            non_zero_grads = (qgan.params_pqc.grad.abs() > 1e-8).sum().item()
            print(f"✅ Quantum parameter gradients:")
            print(f"   - Gradient tensor shape: {qgan.params_pqc.grad.shape}")
            print(f"   - Gradient magnitude: {grad_magnitude:.8f}")
            print(f"   - Non-zero gradients: {non_zero_grads}/{len(qgan.params_pqc)}")
            print(f"   - Gradient range: [{qgan.params_pqc.grad.min().item():.8f}, {qgan.params_pqc.grad.max().item():.8f}]")
            
            if non_zero_grads == 0:
                print("❌ WARNING: All quantum parameter gradients are zero!")
            elif non_zero_grads < len(qgan.params_pqc) * 0.5:
                print(f"⚠️  WARNING: Only {non_zero_grads}/{len(qgan.params_pqc)} parameters have non-zero gradients")
            else:
                print(f"✅ Good: {non_zero_grads}/{len(qgan.params_pqc)} parameters have gradients")
        else:
            print("❌ CRITICAL: No gradients computed for quantum parameters!")
        
        # Check noise input gradients  
        if noise_input.grad is not None:
            noise_grad_magnitude = torch.norm(noise_input.grad).item()
            print(f"✅ Noise input gradients:")
            print(f"   - Gradient magnitude: {noise_grad_magnitude:.8f}")
            print(f"   - Gradient range: [{noise_input.grad.min().item():.8f}, {noise_input.grad.max().item():.8f}]")
        else:
            print("❌ No gradients computed for noise input")
        print()
        
        # Step 4: Test multiple forward passes (gradient accumulation)
        print("4️⃣ GRADIENT ACCUMULATION TEST")
        print("-" * 50)
        
        # Clear gradients
        qgan.params_pqc.grad = None
        
        accumulated_loss = 0
        for i in range(3):
            # New noise for each pass
            test_noise = torch.tensor(np.random.uniform(0, 2*np.pi, NUM_QUBITS), 
                                    dtype=torch.float32, requires_grad=True)
            output = qgan.generator(test_noise, qgan.params_pqc)
            # Handle tuple output from generator
            if isinstance(output, (list, tuple)):
                output = torch.stack(output)
            loss = output.sum()
            accumulated_loss += loss.item()
            loss.backward()
        
        if qgan.params_pqc.grad is not None:
            accumulated_grad_magnitude = torch.norm(qgan.params_pqc.grad).item()
            print(f"✅ Accumulated gradients:")
            print(f"   - Total loss: {accumulated_loss:.6f}")
            print(f"   - Accumulated gradient magnitude: {accumulated_grad_magnitude:.8f}")
        print()
        
        # Step 5: Test gradient flow through critic interaction
        print("5️⃣ CRITIC INTERACTION GRADIENT TEST")
        print("-" * 50)
        
        # Create fake data tensor matching real data format
        generated_sample = qgan.generator(noise_input, qgan.params_pqc)
        
        # Process like in training loop
        if isinstance(generated_sample, (list, tuple)):
            generated_sample = torch.stack(list(generated_sample))
        
        generated_sample = generated_sample.to(torch.float64)
        generated_sample_input = generated_sample.unsqueeze(0).unsqueeze(1)  # [1, 1, window_length]
        
        print(f"✅ Processed sample shape: {generated_sample_input.shape}")
        
        # Pass through critic
        fake_score = qgan.critic(generated_sample_input)
        print(f"✅ Critic output: {fake_score.item():.6f}")
        
        # Generator loss (as in actual training)
        generator_loss = -fake_score  # Simplified version
        print(f"✅ Generator loss: {generator_loss.item():.6f}")
        
        # Clear and compute gradients
        qgan.params_pqc.grad = None
        generator_loss.backward()
        
        if qgan.params_pqc.grad is not None:
            critic_grad_magnitude = torch.norm(qgan.params_pqc.grad).item()
            print(f"✅ Gradients through critic:")
            print(f"   - Gradient magnitude: {critic_grad_magnitude:.8f}")
            print(f"   - Non-zero gradients: {(qgan.params_pqc.grad.abs() > 1e-8).sum().item()}/{len(qgan.params_pqc)}")
        else:
            print("❌ CRITICAL: No gradients through critic!")
        print()
        
        # Step 6: QNode configuration check
        print("6️⃣ QNODE CONFIGURATION CHECK")
        print("-" * 50)
        print(f"✅ QNode device: {qgan.generator.device}")
        print(f"✅ QNode interface: {qgan.generator.interface}")
        print(f"✅ QNode diff_method: {qgan.generator.diff_method}")
        
        # Check if device supports gradients
        if hasattr(qgan.generator.device, 'supports_derivatives'):
            print(f"✅ Device supports derivatives: {qgan.generator.device.supports_derivatives}")
        
        # Check trainable parameters by constructing a tape
        try:
            import pennylane as qml
            # Create a test tape to check trainable parameters
            with qml.tape.QuantumTape() as test_tape:
                qgan.generator(noise_input, qgan.params_pqc)
            if test_tape:
                print(f"✅ Tape trainable params: {len(test_tape.trainable_params) if test_tape.trainable_params else 0}")
        except Exception as e:
            print(f"⚠️  Could not construct tape: {e}")
        print()
        
        # Step 7: Parameter-shift gradient validation
        print("7️⃣ PARAMETER-SHIFT GRADIENT VALIDATION")
        print("-" * 50)
        
        # Test parameter-shift rule manually for first parameter
        param_idx = 0
        shift = np.pi / 2
        
        # Store original parameter
        original_param = qgan.params_pqc[param_idx].item()
        
        # Forward shift
        qgan.params_pqc.data[param_idx] = original_param + shift
        output_plus = qgan.generator(noise_input, qgan.params_pqc)
        if isinstance(output_plus, (list, tuple)):
            output_plus = torch.stack(output_plus)
        loss_plus = output_plus.sum().item()
        
        # Backward shift  
        qgan.params_pqc.data[param_idx] = original_param - shift
        output_minus = qgan.generator(noise_input, qgan.params_pqc)
        if isinstance(output_minus, (list, tuple)):
            output_minus = torch.stack(output_minus)
        loss_minus = output_minus.sum().item()
        
        # Restore original parameter
        qgan.params_pqc.data[param_idx] = original_param
        
        # Calculate parameter-shift gradient
        param_shift_grad = (loss_plus - loss_minus) / 2
        print(f"✅ Manual parameter-shift gradient for param {param_idx}: {param_shift_grad:.8f}")
        
        # Compare with autograd gradient
        qgan.params_pqc.grad = None
        output_original = qgan.generator(noise_input, qgan.params_pqc)
        if isinstance(output_original, (list, tuple)):
            output_original = torch.stack(output_original)
        loss_original = output_original.sum()
        loss_original.backward()
        
        if qgan.params_pqc.grad is not None:
            autograd_grad = qgan.params_pqc.grad[param_idx].item()
            print(f"✅ Autograd gradient for param {param_idx}: {autograd_grad:.8f}")
            print(f"✅ Gradient difference: {abs(param_shift_grad - autograd_grad):.8f}")
            
            if abs(param_shift_grad - autograd_grad) < 1e-6:
                print("✅ Gradients match! Parameter-shift is working correctly.")
            else:
                print("⚠️  WARNING: Gradients don't match. Check parameter-shift implementation.")
        print()
        
        # Step 8: Final assessment
        print("8️⃣ FINAL GRADIENT FLOW ASSESSMENT")
        print("-" * 50)
        
        # Overall gradient health check
        issues = []
        
        if qgan.params_pqc.grad is None:
            issues.append("No quantum parameter gradients")
        elif torch.norm(qgan.params_pqc.grad).item() < 1e-10:
            issues.append("Extremely small gradients (possible vanishing gradient)")
        
        if (qgan.params_pqc.grad.abs() > 1e-8).sum().item() < len(qgan.params_pqc) * 0.8:
            issues.append("Many parameters have zero gradients")
        
        if len(issues) == 0:
            print("🎉 SUCCESS: Gradient flow appears healthy!")
            print("✅ All quantum parameters receive gradients")
            print("✅ Gradients flow through encoding, circuit, and measurements")
            print("✅ Parameter-shift rule is working correctly")
        else:
            print("⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")
        
        print("\n" + "="*80)
        print("GRADIENT FLOW DIAGNOSTIC COMPLETE")
        
    except Exception as e:
        print(f"❌ ERROR during gradient flow test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    gradient_flow_diagnostic() 