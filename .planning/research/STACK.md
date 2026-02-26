# Stack Research: PennyLane qGAN WGAN-GP Remediation

**Research Date:** 2026-02-26
**Research Type:** Project Research — Subsequent milestone
**Question:** What are the current best practices for PennyLane quantum circuit design, WGAN-GP training, and PyTorch memory management as of 2025-2026?
**Scope:** Fixing ~40 issues in `qgan_pennylane.ipynb` using PennyLane 0.44.0 and PyTorch 2.8.0

---

## Summary

The existing notebook uses non-standard hyperparameters, a suboptimal differentiation method, insecure checkpoint loading, and a suboptimal quantum circuit design. Each area has well-established best practices that the remediation must implement.

---

## 1. PennyLane Differentiation Method (`diff_method`)

### Recommendation: Use `diff_method="backprop"` on `default.qubit`

```python
@qml.qnode(dev, interface="torch", diff_method="backprop")
def generator_circuit(inputs, weights):
    ...
```

**Why:** For a simulator-only workflow, backpropagation through the statevector requires only one forward pass regardless of parameter count, whereas parameter-shift requires 2p circuit executions. For ~45 parameters, that is an ~90x reduction.

**Constraint:** Requires `shots=None` (analytic mode). Already satisfied.

**Alternative: `diff_method="adjoint"` on `lightning.qubit`** — lower memory, C++ backend adds 2-8x speedup. `pennylane-lightning 0.44.0` is already installed.

| Method | Circuit evals/step | Memory | Requires |
|---|---|---|---|
| `parameter-shift` (current) | 2p | Low | Any device |
| `backprop` on `default.qubit` | 1 | High | `shots=None` |
| `adjoint` on `lightning.qubit` | 1 | Low | `shots=None`, Lightning |

**Decision:** Switch to `backprop` on `default.qubit` as minimal-change fix.

**Confidence: High.** PennyLane 0.44.0 docs confirm backprop is the preferred method for `default.qubit` + PyTorch.

**What NOT to do:**
- Do NOT use `parameter-shift` for simulator training
- Do NOT use `finite-diff` — numerically noisy
- Do NOT mix `shots > 0` with `backprop`

---

## 2. PennyLane Parameter Broadcasting

### Recommendation: Use parameter broadcasting for batch execution

```python
# Instead of looping:
outputs = torch.stack([generator_circuit(noise[i], params) for i in range(batch_size)])

# Use broadcasting:
outputs = generator_circuit(noise, params)  # single simulation for whole batch
```

Works with `backprop`. Known gradient issues exist with `parameter-shift`.

**Confidence: High.** Documented in PennyLane 0.44.0.

---

## 3. Quantum Circuit Design

### 3a. Remove redundant RZ after IQPEmbedding
`qml.IQPEmbedding` already applies RZ rotations. Adding another RZ per qubit immediately after conflates encoding and variational parameters.

### 3b. Expand noise range from [0, 2pi] to [0, 4pi]
RZ has a period of 4pi. Sampling from [0, 2pi] covers only half the parameter space.

### 3c. Add data re-uploading between variational layers
Re-uploading between each layer enables universal approximation (Perez-Salinas et al., 2020).

### 3d. Add PauliX measurements alongside PauliZ
```python
return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)] + \
       [qml.expval(qml.PauliX(i)) for i in range(NUM_QUBITS)]
```

**Confidence: High** (expressibility). **Medium** (specific GAN impact).

---

## 4. WGAN-GP Hyperparameters

### Restore Gulrajani et al. (2017) standard values

```python
N_CRITIC = 5    # was 1
LAMBDA = 10     # was 0.8
```

- **N_CRITIC=5:** Critic must be sufficiently trained before generator updates
- **LAMBDA=10:** Enforces Lipschitz constraint at theoretically motivated strength
- **Remove dropout from critic:** Stochastic output invalidates gradient penalty computation
- **Adam betas (0.0, 0.9):** Prevents momentum-based oscillation

**Confidence: High.** Published defaults from Gulrajani et al.

---

## 5. PyTorch Memory Management

### 5a. `torch.no_grad()` during evaluation
Prevents unnecessary computation graph creation during inference.

### 5b. `torch.load` with `weights_only=True` and `map_location`
```python
checkpoint = torch.load(path, map_location=device, weights_only=True)
```
Default since PyTorch 2.6. Prevents arbitrary code execution via malicious checkpoints.

### 5c. Checkpoint save/load pattern
**Save:** `model.params_pqc.data` (not the `nn.Parameter` wrapper) for `weights_only=True` compatibility.
**Load:** Re-wrap as `nn.Parameter` on load.

### 5d. Loss values: use `.item()` before appending to lists
Prevents retention of entire computation graph across epochs.

---

## 6. EMD Computation

```python
emd = wasserstein_distance(real_samples.flatten(), fake_samples.flatten())
```

Use raw sample arrays, not histogram-binned distributions.

**Confidence: High.** Scipy docs explicitly define inputs as empirical distribution samples.

---

## 7. Early Stopping: Monitor EMD, Not Critic Loss

EMD directly measures distributional distance. Critic loss is an indirect proxy. Compute every 10 epochs due to cost.

**Confidence: Medium.** Theoretically correct; stability at small sample sizes is uncertain.

---

## 8. Version Verification

| Library | Installed | Notes |
|---|---|---|
| PennyLane | 0.44.0 | `backprop` with PyTorch stable since v0.18 |
| pennylane-lightning | 0.44.0 | `adjoint` available as upgrade path |
| PyTorch | 2.8.0 | `weights_only=True` default since 2.6 |
| scipy | 1.15.3 | `wasserstein_distance` raw input since 1.0.0 |

No upgrades required.

---

## Sources

- [PennyLane QNode API — 0.44.0](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html)
- [Quantum Gradients with Backpropagation | PennyLane Demos](https://pennylane.ai/qml/demos/tutorial_backprop)
- [Gulrajani et al. (2017) — Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [torch.load — PyTorch 2.10](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [scipy.stats.wasserstein_distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)

---
*Research completed: 2026-02-26*
