# Domain Pitfalls: Post-HPO Improvements (v1.1)

**Domain:** Adding spectral loss, broadcasting, circuit depth, critic simplification, and noise range fix to existing WGAN-GP qGAN
**Researched:** 2026-03-13
**Overall Confidence:** HIGH (based on direct codebase analysis, HPO results, and verified research)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalidate HPO results, or reintroduce variance collapse.

---

### P-1: Noise Range Fix Invalidates HPO Results

**What goes wrong:** The HPO was run with `noise_values = np.random.uniform(0, 2 * np.pi, ...)` (confirmed at 4 locations in `hpo_runner.ipynb` and `hpo_retrain_runner.ipynb`). The optimized hyperparameters -- `lr_critic=1.80e-05`, `lr_gen_ratio=3.83`, `lambda_gp=2.16`, `n_critic=9`, `lambda_acf=0.062` -- were tuned for a generator whose input space covers `[0, 2pi]`. Switching to `[0, 4pi]` doubles the noise input range, fundamentally changing the loss landscape. The quantum circuit's rotation gates (RZ) have period `2pi`, so `[0, 4pi]` makes the noise distribution wrap around twice, creating a qualitatively different input manifold. HPO-optimized learning rates and penalty weights will not transfer.

**Why it happens:** The noise range regression from `[0, 4pi]` to `[0, 2pi]` happened during PAR_LIGHT conditioning work. HPO was then run on the regressed code. Fixing the regression is correct, but the HPO results become stale.

**Consequences:**
- Learning rates tuned for one noise distribution produce different gradient magnitudes with another
- `lambda_gp=2.16` (unusually low vs standard 10) was found optimal for `[0, 2pi]` -- may cause critic instability at `[0, 4pi]`
- `n_critic=9` was tuned for the old generator expressivity -- may be too many/few for the expanded range
- At best: suboptimal training. At worst: complete training divergence with no explanation

**Prevention:**
1. Fix the noise range FIRST as its own isolated change
2. Run a validation training (100-200 epochs) with the HPO hyperparameters but the corrected noise range
3. Compare EMD convergence curve against the HPO retrain baseline (`best_emd=0.001137`)
4. If EMD is >2x worse or training diverges, HPO must be re-run with corrected noise range
5. Keep HPO results as starting point for a focused re-search around the optimized values

**Detection:** Compare epoch-100 EMD of corrected-noise run vs HPO-retrain epoch-100 EMD. If divergence >50%, hyperparameters are not transferring.

**Phase:** Must be Phase 1 (fix noise range, then validate HPO transferability before any other changes)

**Confidence:** HIGH -- confirmed by examining `results/hpo/best_params.json` and all HPO notebook code

---

### P-2: Spectral Loss Gradient Dominance Kills WGAN Training

**What goes wrong:** Adding a PSD/spectral loss term to the generator creates a multi-objective optimization: `generator_loss = -E[D(fake)] + lambda_acf * acf_penalty + lambda_psd * psd_loss`. The PSD loss operates in frequency domain (computed via `torch.fft.rfft`) and produces gradients on a completely different scale than the Wasserstein critic loss. If `lambda_psd` is even slightly too large, the spectral gradients dominate and the generator ignores the critic entirely -- leading to mode collapse or a generator that matches PSD but produces physically implausible time series. Conversely, if `lambda_psd` is too small, it has no effect.

**Why it happens:**
- The Wasserstein loss `E[D(fake)]` produces gradients of magnitude ~O(1) in a well-trained WGAN-GP
- PSD mismatch (mean squared error between log-PSD curves) produces gradients proportional to the total spectral energy difference, which can be O(10-100) depending on normalization
- The existing `lambda_acf=0.062` was already HPO-tuned -- adding a third loss term disrupts this balance
- FFT-based losses have sharp gradients at specific frequencies, creating "spiky" gradient landscapes that can destabilize Adam optimizer momentum

**Consequences:**
- Generator learns to match frequency spectrum but loses distributional fidelity (EMD degrades)
- Gradient competition between critic loss, ACF penalty, and PSD loss causes oscillation
- Variance collapse can WORSEN because the generator allocates capacity to matching spectral shape at the expense of amplitude

**Prevention:**
1. Compute PSD loss in log-space: `psd_loss = MSE(log(PSD_fake + eps), log(PSD_real + eps))` -- this normalizes gradient magnitudes across frequencies
2. Start with `lambda_psd = 0` and increase gradually (warmup schedule over 200 epochs)
3. Log all three loss components separately (Wasserstein, ACF, PSD) every eval epoch -- monitor that no single component dominates
4. Normalize PSD by window length before comparison: `PSD = |FFT(x)|^2 / len(x)`
5. Use `eps = 1e-8` in log to prevent `-inf` gradients at zero-power frequencies
6. Verify `torch.fft.rfft` gradient flow through PennyLane: run `loss.backward()` and check `params_pqc.grad` is not None and not NaN

**Detection:** If generator loss decreases but EMD worsens for >50 consecutive epochs, spectral loss is dominating. If PSD loss component is >10x the Wasserstein loss component, rebalance weights.

**Phase:** Should be Phase 3 (after noise range fix and broadcasting are validated)

**Confidence:** HIGH -- PSD loss gradient scale mismatch is well-documented in frequency-domain GAN literature (see Sources)

---

### P-3: Broadcasting Optimization Breaks with Conditional PAR_LIGHT Inputs

**What goes wrong:** PennyLane parameter broadcasting requires all batched inputs to have the batch dimension as their FIRST axis, and all batch dimensions must have the same size. The current generator signature is `generator(noise_params, par_light_params, params_pqc)` where:
- `noise_params`: shape `(num_qubits,)` per sample, batched would be `(batch_size, num_qubits)`
- `par_light_params`: shape `(num_qubits,)` per sample, batched would be `(batch_size, num_qubits)`
- `params_pqc`: shape `(num_params,)` -- shared across batch, NOT batched

The problem is that PennyLane broadcasting requires operators to receive inputs with exactly one additional axis. But `params_pqc` is indexed element-by-element inside the circuit (`params_pqc[idx]`), not passed directly to operators. This manual indexing breaks broadcasting because PennyLane cannot infer which axis is the batch dimension when parameters are sliced.

Additionally, the existing evaluation cell (line ~4278) uses the TRANSPOSED convention: `par_tensor_gen = torch.tensor(par_compressed.T, ...)` with shape `(5, num_samples)`. This means `(num_qubits, batch_size)` -- NOT `(batch_size, num_qubits)`. The PennyLane broadcasting axis convention for 1D operator inputs appends the batch axis as the LAST dimension, not the first. This contradicts PyTorch convention and is a common source of silent errors.

**Why it happens:** The original broadcasting optimization (commit 6ddc49d) was implemented before PAR_LIGHT conditioning was added. The conditioning added a second input argument to the circuit. The per-sample loop was reintroduced (possibly during debugging) and never reverted. Attempting to naively stack batched inputs will fail because the circuit uses explicit Python indexing (`for qubit in range(self.num_qubits): qml.RZ(phi=params_pqc[idx], ...)`) rather than vectorized operator calls.

**Consequences:**
- Naive batching attempt causes shape mismatch errors or silent incorrect results
- The circuit's manual parameter indexing (`idx` counter) is incompatible with broadcasting semantics
- Training remains ~12x slower than necessary (batch_size=12 samples sequentially)
- If broadcasting is "partially" implemented (noise batched but PAR_LIGHT not), generated samples will have wrong conditioning

**Prevention:**
1. Use `qml.batch_input` transform instead of raw broadcasting -- it handles non-trainable batched inputs separately from trainable parameters
2. Specify `argnum` carefully: noise (arg 0) and par_light (arg 1) should be batched; params_pqc (arg 2) should NOT
3. Test on a single batch first: compare output of batched call vs loop-of-individual-calls element-by-element (must match within floating-point tolerance ~1e-6)
4. Verify gradient flow: after batched forward pass, call `.backward()` and confirm `params_pqc.grad.shape == params_pqc.shape` (no extra batch dimension)
5. The `diff_method='parameter-shift'` in current code is incompatible with some broadcasting optimizations -- may need to switch to `diff_method='backprop'` first

**Detection:** If batched output shape is not `(batch_size, window_length)`, broadcasting failed. If any NaN appears in gradients after batching, parameter indexing is broken.

**Phase:** Should be Phase 2 (after noise range fix is validated, before adding spectral loss)

**Confidence:** MEDIUM -- PennyLane broadcasting with conditional inputs is not extensively documented; `batch_input` transform should work but needs empirical validation

---

### P-4: Evaluation Section Uses `par_zeros` -- Conditioning Never Tested During Training

**What goes wrong:** In `_train_one_epoch`, the per-epoch evaluation section (around line 1526 of the notebook JSON) generates samples with `par_zeros = torch.zeros(self.num_qubits, dtype=torch.float32)` instead of actual PAR_LIGHT values. This means EMD, ACF RMSE, VOL RMSE, and LEV RMSE metrics -- the exact metrics used for HPO and early stopping -- are computed on UNCONDITIONED samples. The entire HPO study optimized hyperparameters for a generator that produces unconditioned output during evaluation, even though training uses real PAR_LIGHT values.

**Why it happens:** When PAR_LIGHT conditioning was added to the training loops (critic and generator), the evaluation/metrics section was not updated to match. Using zeros for conditioning means `qml.RY(0 * pi, wires=i)` which is an identity operation -- the conditioning has zero effect during evaluation.

**Consequences:**
- ALL reported metrics (EMD, stylized facts RMSEs) reflect unconditioned generation quality
- HPO optimized for unconditioned generation -- the optimal hyperparameters may not be optimal for conditioned generation
- PAR_LIGHT conditioning verification (planned feature) may "fail" not because conditioning doesn't work, but because it was never used during metric computation
- If conditioning is actually effective, enabling it during evaluation could make metrics WORSE initially (because the generator was optimized to produce good output with zero conditioning)

**Prevention:**
1. Fix evaluation section to use real PAR_LIGHT values from the data (same as training)
2. Run a comparison: evaluate the CURRENT best checkpoint with zeros vs real PAR_LIGHT -- if metrics are identical, conditioning is not modulating output (a different problem)
3. If metrics differ significantly, the HPO results reflect unconditioned optimization and may need re-running
4. Log both conditioned and unconditioned metrics during training for comparison

**Detection:** Generate 100 samples with `par_light = zeros` and 100 with `par_light = real_values`. If output distributions are statistically indistinguishable (KS test p > 0.05), conditioning is not working at all.

**Phase:** Must be diagnosed in Phase 1 (before any other changes, as it affects baseline measurements)

**Confidence:** HIGH -- confirmed by direct code inspection at line 1526 of notebook JSON

---

### P-5: Circuit Layer Increase Triggers Barren Plateau

**What goes wrong:** Increasing `NUM_LAYERS` from 4 to 6-8 in a strongly entangled PQC with 5 qubits risks entering the barren plateau regime where gradient variance decreases exponentially with circuit depth. The current circuit uses Rot(phi, theta, omega) gates with CNOT entanglement in a range-based pattern. At 4 layers, the circuit has 75 parameters. At 8 layers, it has 135 parameters. The variance of the cost function gradient scales as `O(1/2^n)` for sufficiently deep circuits with `n` qubits -- but the depth threshold depends on the entanglement pattern and observable locality.

**Why it happens:**
- With 5 qubits and strongly entangling layers, the circuit approaches a 2-design (uniform distribution over unitaries) at depth ~O(poly(n))
- The range-based CNOT pattern `r = (layer % (num_qubits - 1)) + 1` creates full entanglement coverage by layer 4 -- additional layers push toward expressivity saturation
- PauliX + PauliZ measurements are global observables, making barren plateau onset earlier than for local observables
- The current circuit already has a data re-uploading structure (trainable RZ + noise RZ), which partially mitigates barren plateaus -- but only for the encoding parameters, not the entangling layer parameters

**Consequences:**
- Gradients become exponentially small -- optimizer makes no meaningful updates to deep-layer parameters
- Training appears to converge (loss plateaus) but generator quality doesn't improve
- The additional parameters consume capacity (memory and compute) with no benefit
- Worse: if only shallow-layer parameters receive meaningful gradients, the deep layers act as random noise, reducing output quality

**Prevention:**
1. Add gradient norm monitoring: log `torch.norm(params_pqc.grad)` every eval epoch
2. Start from 4 layers (known working), increase to 5, validate, then 6 -- do NOT jump to 8
3. Compare gradient norms for first-layer vs last-layer parameters at each depth
4. If gradient norm of last-layer parameters is <1% of first-layer parameters, stop increasing depth
5. Consider layer-wise learning rate scaling: higher LR for deeper layers to compensate for smaller gradients
6. Alternative to more layers: use data re-uploading at EVERY layer (not just encoding) to break the barren plateau

**Detection:** Plot gradient norm per layer across training. If deeper layers show gradient norms <1e-6 while shallow layers show >1e-3, barren plateau is active.

**Phase:** Should be Phase 4 (after spectral loss and broadcasting are working -- circuit depth is an optimization, not a fix)

**Confidence:** HIGH -- barren plateau theory for strongly entangling circuits is well-established (see Sources)

---

## Moderate Pitfalls

Mistakes that cause wasted training runs or incorrect conclusions but are recoverable.

---

### P-6: Critic Simplification Destabilizes Gradient Penalty

**What goes wrong:** The current critic is a 3-layer CNN (Conv1d: 2->64->128->128, ~20K parameters). Simplifying to 2 layers or fewer filters reduces the critic's capacity to be a useful 1-Lipschitz function. In WGAN-GP, the gradient penalty `lambda * E[(||grad_D(x_hat)||_2 - 1)^2]` enforces the 1-Lipschitz constraint. A simpler critic has fewer parameters to satisfy this constraint, meaning the gradient penalty can become the dominant loss term -- the critic spends all its capacity enforcing Lipschitz smoothness rather than learning to discriminate real from fake.

**Why it happens:** The intuition "the critic is too strong, make it simpler" is partially correct -- an overpowered critic can provide uninformative gradients. But WGAN-GP already handles this through the gradient penalty. Making the critic too weak creates the opposite problem: gradients become noisy and uninformative because the critic can't model the real/fake boundary.

**Consequences:**
- Critic loss oscillates without converging (gradient penalty dominates)
- Generator receives low-quality gradient signal -- training is effectively random walking
- The gradient penalty coefficient `lambda_gp=2.16` (HPO-optimized) may be too high or too low for a different architecture
- EMD improves spuriously (weak critic can't detect differences)

**Prevention:**
1. Make critic architecture a configurable parameter, not a replacement
2. Test simplified critic on 200 epochs alongside original -- compare BOTH EMD and moment statistics (std, kurtosis)
3. Never reduce critic below 2 Conv1d layers for window_length=10 input
4. If simplifying, reduce filters first (64->32->32) before removing layers
5. Re-run HPO (or at minimum re-tune `lambda_gp` and `n_critic`) after any critic architecture change
6. Monitor gradient penalty magnitude: if `GP_loss > 2 * wasserstein_loss`, critic is struggling to be Lipschitz

**Detection:** If critic loss is dominated by the GP term (>70% of total critic loss), the architecture is too constrained.

**Phase:** Should be Phase 5 (last priority -- requires stable baseline from all other changes first)

**Confidence:** MEDIUM -- interaction between critic capacity and GP is well-understood theoretically but the specific thresholds for this circuit/dataset need empirical validation

---

### P-7: PSD Loss Computation Cost with parameter-shift Gradients

**What goes wrong:** The PSD loss requires: `generated_samples -> FFT -> |FFT|^2 -> log -> MSE`. The generated samples come from a PennyLane QNode. The gradient path is: `PSD_loss -> torch.fft.rfft -> quantum_output -> PennyLane parameter-shift`. If `diff_method='parameter-shift'` (current setting), PennyLane evaluates the circuit at `theta + pi/2` and `theta - pi/2` for each parameter. This doubles the number of circuit evaluations per backward pass. Adding PSD loss doesn't break differentiability, but it multiplies the gradient computation cost because the parameter-shift rule must evaluate each shifted circuit, then the shifted outputs must pass through the FFT loss.

**Why it happens:** `torch.fft.rfft` is fully differentiable in PyTorch's autograd. The issue is the compound chain: parameter-shift already makes backward passes expensive (2 circuit evaluations per parameter). With 75 parameters and batch_size=12, that's `2 * 75 * 12 = 1,800` circuit evaluations per generator step, each followed by FFT.

**Consequences:**
- Training becomes ~2x slower per generator step (FFT itself is fast, but it runs on each shifted circuit output)
- If using `backprop` diff_method instead (which the codebase was designed for but currently uses parameter-shift), this is NOT an issue -- backprop computes all gradients in one pass
- Memory usage increases because intermediate FFT results must be retained for backprop

**Prevention:**
1. Switch from `diff_method='parameter-shift'` to `diff_method='backprop'` BEFORE adding PSD loss
2. Verify `backprop` works with `default.qubit` device (it should -- this was tested in v1.0)
3. Compute PSD loss only on the batch-level (average PSD across batch), not per-sample, to reduce FFT calls
4. Profile: measure wall-clock time per epoch with and without PSD loss

**Detection:** If epoch time increases >3x after adding PSD loss, the parameter-shift + FFT chain is the bottleneck.

**Phase:** Switch diff_method in Phase 2 (with broadcasting), add PSD loss in Phase 3

**Confidence:** HIGH -- PyTorch FFT autograd support is confirmed; parameter-shift cost model is well-known

---

### P-8: Loss Weight Interaction -- Three Penalties Cannot Be Independently Tuned

**What goes wrong:** The generator loss will be `L = -E[D(fake)] + lambda_acf * ACF + lambda_psd * PSD`. The HPO optimized `lambda_acf=0.062` for a two-term loss. Adding a third term changes the effective weight of the first two. Even if `lambda_psd` is small, it shifts the gradient balance and the HPO-optimal `lambda_acf` is no longer optimal.

**Why it happens:** In multi-objective optimization, the Pareto front shifts when objectives are added. A weight that was optimal for 2 objectives is not optimal for 3 objectives because the gradient directions are not orthogonal.

**Consequences:**
- ACF penalty becomes less effective (its relative weight drops)
- The generator may sacrifice temporal structure (ACF) to match spectral structure (PSD), even though both matter
- Tuning `lambda_psd` in isolation while keeping `lambda_acf=0.062` fixed may degrade ACF metrics

**Prevention:**
1. After adding PSD loss, do a small grid search over `(lambda_acf, lambda_psd)` pairs -- not just `lambda_psd` alone
2. Normalize all loss terms to similar scales before weighting: `loss_term / running_mean(loss_term)`
3. Alternatively, use a scheduled approach: train with ACF-only for 200 epochs, then add PSD loss
4. Log ACF RMSE and PSD RMSE separately to detect when one degrades as the other improves

**Detection:** If ACF RMSE increases >20% when PSD loss is added, the weight balance is wrong.

**Phase:** Phase 3 (when adding spectral loss)

**Confidence:** HIGH -- multi-objective loss interaction is fundamental to optimization theory

---

### P-9: `diff_method='parameter-shift'` Negates Broadcasting Speedup

**What goes wrong:** The current code uses `diff_method='parameter-shift'`. Even if broadcasting is correctly implemented for the forward pass (reducing from 12 sequential circuit calls to 1 batched call), the backward pass still requires 2 circuit evaluations per parameter for parameter-shift gradients. With 75 parameters, that's 150 circuit evaluations per backward pass -- the forward-pass speedup from broadcasting is dwarfed by the backward-pass cost.

**Why it happens:** Parameter-shift is a hardware-compatible gradient method that doesn't require access to the quantum state. It was chosen for "better stability and gradient flow" (comment in code). But on a simulator (`default.qubit`), `backprop` is both faster and more numerically stable.

**Consequences:**
- Broadcasting optimization provides ~12x forward speedup but only ~1.1x overall speedup (backward dominates)
- The expected "12x training speedup" from broadcasting will not materialize
- Developers may blame broadcasting implementation rather than diff_method

**Prevention:**
1. Switch to `diff_method='backprop'` alongside broadcasting implementation
2. Verify on `default.qubit` (not `lightning.qubit` -- lightning doesn't support backprop)
3. Benchmark: forward-only time vs forward+backward time, with and without backprop

**Detection:** If total epoch time with broadcasting is >50% of sequential epoch time, backward pass is the bottleneck.

**Phase:** Phase 2 (fix together with broadcasting)

**Confidence:** HIGH -- PennyLane diff_method performance characteristics are well-documented

---

### P-10: PAR_LIGHT Compression Loses Temporal Information

**What goes wrong:** The current compression `par_window.reshape(num_qubits, 2).mean(dim=1)` averages every 2 adjacent PAR_LIGHT values to map a 10-element window to 5 qubit inputs. PAR_LIGHT has only 6 unique values `[0.0, 2.5, 5.0, 7.5, 10.0, 12.5]` after normalization to `[0, 1]`, these are `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. Averaging adjacent pairs loses the temporal pattern of light changes within a window. If PAR_LIGHT changes from 0.4 to 0.8 within a window, the mean is 0.6 -- but so is a constant 0.6. The generator cannot distinguish these.

**Why it happens:** The quantum circuit has 5 qubits and takes 5 noise values + 5 PAR_LIGHT values as input. The window length is 10 (2 * num_qubits). There's no natural way to encode 10 conditioning values into 5 qubit inputs without compression.

**Consequences:**
- PAR_LIGHT conditioning verification may show weak modulation not because conditioning is broken, but because the compressed values are uninformative
- The generator cannot capture light-transition dynamics within windows
- This is unlikely to be the primary cause of variance collapse but will limit the ceiling for conditional generation quality

**Prevention:**
1. Try alternative compression: use first and last PAR_LIGHT value (captures transitions), or max and min
2. Consider dedicating one qubit to PAR_LIGHT encoding with data re-uploading
3. Defer this optimization -- the immediate priority is verifying that conditioning works AT ALL before optimizing how it is encoded

**Detection:** Compare generation quality with 6 distinct PAR_LIGHT levels. If output is identical for PAR_LIGHT=0.0 and PAR_LIGHT=1.0, conditioning is not working (independent of compression).

**Phase:** Phase 4 (during conditioning verification, not as a standalone fix)

**Confidence:** MEDIUM -- the impact depends on how much PAR_LIGHT actually varies within 10-step windows

---

### P-11: Checkpoint Incompatibility When Changing NUM_LAYERS

**What goes wrong:** Increasing `NUM_LAYERS` from 4 to 6 changes `num_params` from 75 to 105. Loading a 75-parameter checkpoint into a 105-parameter model fails with a size mismatch error, or worse, silently loads partial parameters if the checkpoint loading code does not validate shape.

**Why it happens:** `params_pqc` is a flat 1D tensor. Its size is `iqp_params + NUM_LAYERS * per_layer_params + final_params`. Any layer count change changes this size.

**Consequences:** Training cannot resume from existing checkpoints. All prior training is invalidated and must restart from random initialization.

**Prevention:**
1. When changing `NUM_LAYERS`, always start a fresh training run with new checkpoint directory
2. Add assertion on checkpoint load: `assert checkpoint['params_pqc'].shape[0] == self.num_params`
3. Consider partial warm-start: load first 75 params from old checkpoint, randomly initialize the extra 30

**Detection:** Shape assertion on checkpoint load catches this immediately.

**Phase:** Phase 4 (when changing circuit depth)

**Confidence:** HIGH -- parameter count formula is deterministic

---

## Minor Pitfalls

Issues that cause confusion or minor quality degradation.

---

### P-12: FFT Window Length Artifacts at Window Length 10

**What goes wrong:** Computing PSD on 10-sample windows gives only 6 frequency bins (for real FFT: `N/2 + 1`). The frequency resolution is `1/(N*dt)`. With only 6 bins, the PSD loss has very coarse spectral resolution -- it can distinguish DC from Nyquist but cannot resolve mid-frequency volatility structure (which is the entire purpose of adding PSD loss).

**Prevention:**
1. Compute PSD on concatenated windows (e.g., 4 consecutive windows = 40 samples, 21 frequency bins) rather than individual windows
2. Use Welch's method with overlapping segments
3. Accept that PSD loss will primarily target the lag-1 and lag-2 frequency components -- overlap with ACF penalty is expected
4. Consider whether PSD loss adds value over ACF penalty given this window length

**Detection:** If PSD loss and ACF penalty have correlation >0.9 during training, they are measuring nearly the same thing.

**Phase:** Phase 3 (implementation decision for spectral loss)

**Confidence:** HIGH -- FFT resolution for short windows is a mathematical fact

---

### P-13: `LAMBDA=0.8` in Main Notebook vs `lambda_gp=2.16` from HPO

**What goes wrong:** The main training cell (line 1751) sets `LAMBDA = 0.8` as the gradient penalty coefficient. But the HPO found `lambda_gp = 2.16` as optimal. The HPO retrain section uses `OPT_LAMBDA_GP`, but the default training section above uses `LAMBDA = 0.8`. If someone runs the default training instead of the HPO retrain, they will use the wrong lambda.

**Prevention:** After HPO, update the default hyperparameters to match HPO results. Or better: load HPO results as defaults with a clear comment.

**Phase:** Phase 1 (part of hyperparameter alignment)

**Confidence:** HIGH -- confirmed by reading both code sections

---

### P-14: GEN_SCALE Interaction with PSD Loss

**What goes wrong:** The generator output is scaled by `GEN_SCALE` (currently 0.1) before being passed to the critic. If the PSD loss is computed on scaled outputs but compared against unscaled real data (or vice versa), the frequency content comparison is invalid because scaling by constant `c` multiplies PSD by `c^2`.

**Prevention:** Ensure PSD loss is computed on the same representation as the real data comparison -- either both scaled or both unscaled. The natural choice is to compute PSD AFTER scaling (same representation the critic sees).

**Phase:** Phase 3 (during PSD loss implementation)

**Confidence:** HIGH

---

### P-15: Kernel Restart Required After Class Redefinition

**What goes wrong:** After modifying the `qGAN` class (adding PSD loss method, changing critic architecture), existing `qgan` instances in memory still use the old class definition. Calling `qgan.psd_loss()` fails with `AttributeError`.

**Prevention:** Always restart kernel and re-run all cells after modifying the qGAN class. Document this in the notebook.

**Phase:** All phases

**Confidence:** HIGH

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Severity | Mitigation |
|---|---|---|---|
| **Noise range fix** | HPO results don't transfer (P-1) | Critical | Validation run before proceeding |
| **Noise range fix** | Eval section uses par_zeros (P-4) | Critical | Fix eval to use real PAR_LIGHT |
| **Noise range fix** | Hyperparameter mismatch LAMBDA vs HPO (P-13) | Minor | Align defaults |
| **Broadcasting** | Conditional inputs break batching (P-3) | Critical | Use `batch_input` transform, not raw broadcasting |
| **Broadcasting** | parameter-shift negates speedup (P-9) | Moderate | Switch to backprop simultaneously |
| **Spectral loss** | Gradient dominance (P-2) | Critical | Log-space PSD, warmup schedule, separate monitoring |
| **Spectral loss** | Loss weight interaction (P-8) | Moderate | Joint tuning of lambda_acf and lambda_psd |
| **Spectral loss** | Short FFT windows (P-12) | Minor | Concatenate windows before FFT |
| **Spectral loss** | Differentiability chain cost (P-7) | Moderate | Switch diff_method first |
| **Spectral loss** | GEN_SCALE interaction (P-14) | Minor | Compute PSD on scaled output |
| **Circuit depth** | Barren plateau (P-5) | Critical | Incremental increase with gradient monitoring |
| **Circuit depth** | Checkpoint incompatibility (P-11) | Moderate | Fresh training, shape assertion |
| **PAR_LIGHT verification** | Compression loses info (P-10) | Moderate | Test conditioning effect before optimizing encoding |
| **Critic simplification** | GP instability (P-6) | Moderate | Test alongside original, re-tune lambda_gp |

---

## Dependency Chain

The order in which pitfalls should be addressed, based on dependencies:

```
Phase 1: Noise Range + Evaluation Fix + HPO Validation
  P-1  (noise range invalidates HPO) -- must be validated first
  P-4  (eval uses par_zeros) -- must be fixed to get honest metrics
  P-13 (LAMBDA mismatch) -- align hyperparameters
  GATE: Run 200-epoch validation. If EMD < 2x HPO baseline, proceed.
        If worse, re-run focused HPO with corrected noise.

Phase 2: Broadcasting + diff_method
  P-3  (broadcasting with PAR_LIGHT) -- use batch_input transform
  P-9  (parameter-shift negates speedup) -- switch to backprop
  GATE: Verify batched output matches sequential output within 1e-6.
        Verify epoch time is <30% of pre-broadcasting time.

Phase 3: Spectral Loss
  P-2  (gradient dominance) -- requires working training loop first
  P-7  (differentiability chain cost) -- requires backprop from Phase 2
  P-8  (loss weight interaction) -- tune jointly
  P-12 (FFT window length) -- implementation decision
  P-14 (GEN_SCALE interaction) -- implementation detail
  GATE: EMD stays within 1.5x of Phase 1 baseline after adding PSD loss.
        Fake std improves toward real std (0.0218).

Phase 4: Circuit Depth + Conditioning Verification
  P-5  (barren plateau) -- needs stable baseline
  P-10 (PAR_LIGHT compression) -- verify conditioning first
  P-11 (checkpoint incompatibility) -- fresh training required
  GATE: Gradient norms for new layers are >1% of first-layer norms.

Phase 5: Critic Simplification
  P-6  (GP instability) -- needs everything else stable
  GATE: EMD and moment statistics do not degrade vs Phase 4 baseline.
```

---

## Sources

### Spectral Loss and Frequency-Domain GANs
- [A Spectral Enabled GAN for Time Series Data Generation](https://ar5iv.labs.arxiv.org/html/2103.01904) -- spectral loss architecture for time series GANs
- [EEG Signal Reconstruction Using WGAN with Temporal-Spatial-Frequency Loss](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2020.00015/full) -- multi-domain loss weighting for signal generation
- [PyTorch torch.fft Module with Autograd](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/) -- confirms FFT differentiability in PyTorch

### PennyLane Broadcasting
- [PennyLane QNode Documentation (0.44.0)](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) -- parameter broadcasting rules and axis conventions
- [PennyLane batch_input Documentation (0.44.0)](https://docs.pennylane.ai/en/stable/code/api/pennylane.batch_input.html) -- batch_input transform for non-trainable inputs
- [How to Execute Quantum Circuits in Batches (PennyLane Blog)](https://pennylane.ai/blog/2022/10/how-to-execute-quantum-circuits-in-collections-and-batches/) -- broadcasting performance and gotchas
- [PennyLane broadcast_expand Transform](https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.broadcast_expand.html) -- expansion rules for broadcasting

### Barren Plateaus and Circuit Depth
- [A Lie algebraic theory of barren plateaus for deep PQCs](https://www.nature.com/articles/s41467-024-49909-3) -- gradient variance scaling with depth
- [Parameterized quantum circuits as universal generative models](https://www.nature.com/articles/s41534-025-01064-3) -- expressivity of PQC generators
- [Quantum Models as Fourier Series (PennyLane Demo)](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series) -- relationship between circuit depth, input range, and Fourier frequencies

### WGAN-GP Theory
- [Improved Training of Wasserstein GANs (Gulrajani et al., 2017)](https://arxiv.org/pdf/1704.00028) -- original WGAN-GP paper, gradient penalty theory, critic capacity requirements
- [GAN Training Stability and Convergence](https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html) -- multi-loss balancing considerations
- [Google ML: Common GAN Problems](https://developers.google.com/machine-learning/gan/problems) -- mode collapse and training instability patterns

### Multi-Objective GAN Training
- [MCGAN: Enhancing GAN Training with Regression-Based Generator Loss](https://arxiv.org/html/2405.17191v1) -- auxiliary loss interaction
- [Adaptive Weighted Discriminator for Training GANs](https://pmc.ncbi.nlm.nih.gov/articles/PMC8963430/) -- loss weighting strategies
- [Understanding GAN Loss Functions](https://neptune.ai/blog/gan-loss-functions) -- gradient competition in multi-term losses

---

*Research completed: 2026-03-13*
*Supersedes v1.0 PITFALLS.md (2026-02-26) which covered code remediation pitfalls -- those issues are resolved*
