# Project Research Summary

**Project:** qGAN v1.1 Post-HPO Improvements
**Domain:** Hybrid quantum-classical GAN for bioprocess time series synthesis (PennyLane + PyTorch)
**Researched:** 2026-03-13
**Confidence:** HIGH

## Executive Summary

This project is a v1.1 improvement pass on a working WGAN-GP with a quantum generator, motivated by two problems revealed after HPO: (1) three code regressions introduced during PAR_LIGHT conditioning work that must be corrected before any meaningful experimentation, and (2) persistent variance collapse where the generator learns mean drift but not volatility structure (fake std 0.0104 vs real std 0.0218, 48% of target). Classical baselines (TinyVAE, FCVAE) failed identically, confirming the problem is a loss signal deficiency, not a hyperparameter issue — the generator has no explicit incentive to match the frequency content of the real data.

The recommended approach is a strict phase-ordered remediation. Fix regressions first (noise range [0,2pi] -> [0,4pi], mu/sigma shadowing, and the par_zeros evaluation bug that caused all HPO metrics to reflect unconditioned generation), then switch `diff_method` from `parameter-shift` to `backprop` and restore parameter broadcasting for a combined ~12x+ training speedup. Only then introduce the primary new capability: a differentiable spectral/PSD loss using `torch.fft.rfft` that directly penalizes the generator for producing time series with wrong frequency content. Secondary capacity tuning (configurable circuit layer count 4->6-8, simpler critic architecture option) follows as a validation-gated phase. Every capability needed is achievable with the already-installed stack — no new packages are required.

The dominant risk is ordering-sensitive: fixing the noise range regression invalidates the HPO-optimized hyperparameters (`lambda_gp=2.16`, `n_critic=9`, `lambda_acf=0.062`), which were tuned on the regressed code. A 200-epoch validation run is required after Phase 1 before proceeding, and HPO may need re-running if EMD degrades more than 2x. A second critical risk is the par_zeros evaluation bug: all HPO and early-stopping metrics were computed on unconditioned generation (PAR_LIGHT=0 throughout eval), meaning the thesis claim about conditioning has never been measured, and fixing this may change reported results significantly.

## Key Findings

### Recommended Stack

The entire v1.1 feature set is achievable with the installed stack: PennyLane 0.44.0, PyTorch 2.8.0, and NumPy/SciPy. The only "stack change" is a configuration change: switching `diff_method='parameter-shift'` to `diff_method='backprop'`, which unlocks a ~90x gradient speedup on `default.qubit` AND native parameter broadcasting support. Focal Frequency Loss, `scipy.signal.periodogram`, `qml.StronglyEntanglingLayers` as a replacement, spectral normalization, and `pennylane-lightning` are all explicitly rejected as inappropriate or unnecessary for this problem scale.

**Core technologies:**
- `torch.fft.rfft` (PyTorch 2.8.0, installed): differentiable PSD loss — fully autograd-supported since PyTorch 1.7, 6 frequency bins for window_length=10, log-PSD MSE appropriate for this scale
- PennyLane native broadcasting (PennyLane 0.44.0, installed): batched QNode execution — requires `backprop` on `default.qubit`, ~12x forward pass speedup, existing evaluation cell proves it works with this circuit architecture
- `nn.Sequential` with `nn.Linear` (PyTorch 2.8.0, installed): simpler critic option — ~5.5K params vs ~250K for current CNN, reduces 3,345:1 parameter ratio, no dropout (required for WGAN-GP theory)
- `diff_method='backprop'` (configuration change only): prerequisite for broadcasting — on `default.qubit` with `shots=None`, both faster and more numerically stable than `parameter-shift` on simulator

### Expected Features

Research establishes a clear two-tier structure: regressions that must be fixed (P0) and new capabilities that address the root cause (P1/P2).

**Must have — regression fixes (P0, table stakes):**
- TS-1: Noise range restoration [0,2pi] -> [0,4pi] — 3 literal changes in `_train_one_epoch`; also fix HPO notebooks at 4 locations
- TS-2: Broadcasting optimization restoration — refactor 3 training loops from per-sample Python for-loops to single batched QNode call; requires switching `diff_method` to `backprop` simultaneously
- TS-3: mu/sigma variable shadowing fix — trivial rename to avoid re-execution bugs
- TS-4 (unlabeled in FEATURES but confirmed critical in PITFALLS P-4): Fix evaluation section to use real PAR_LIGHT values instead of `par_zeros` — all HPO metrics currently reflect unconditioned generation

**Must have — high-impact new capabilities (P1):**
- D-1: Spectral/PSD loss (`lambda_psd * log-PSD MSE`) — addresses root cause of variance collapse by penalizing wrong frequency content; highest expected impact; requires `lambda_psd` warmup schedule and separate loss component logging
- D-3: PAR_LIGHT conditioning verification — intervention test (PAR_LIGHT=0.0 vs 1.0, KS test + sweep plot); thesis requirement; gates whether conditioning refactoring is needed

**Should have — capacity tuning if variance collapse persists (P2):**
- D-2: Configurable circuit layer count (4->6 or 8) — `NUM_LAYERS` constant already used throughout; linear parameter scaling (75->105->135 params); test incrementally from 4->5->6, not a direct jump to 8
- D-4: Simpler critic architecture option — configurable `critic_type` flag (not a replacement); must re-tune `lambda_gp` after any critic change

**Defer to v2+:**
- Multi-resolution spectral loss — overkill for T=10 (only 6 frequency bins)
- Autoregressive stitching for longer windows — changes generation paradigm fundamentally
- Automated circuit architecture search (NAS) — infeasible in notebook workflow

### Architecture Approach

All changes are contained within the existing `qGAN(nn.Module)` class in Cell 26 of `qgan_pennylane.ipynb`, plus config cell 28, HPO cell 37, and retrain cell 40. The class structure is sound — the regressions are in the training loop internals, not the class design. New additions are two methods (`diff_psd_loss` as a static method, `define_critic_model_simple`), two new `__init__` parameters (`lambda_psd`, `critic_type`), and broadcasting refactors of the three loops in `_train_one_epoch`. A `_generate_batch` helper should be extracted to avoid duplicating broadcasting and output-parsing logic across the three loops.

**Major components and their v1.1 changes:**
1. `qGAN.__init__`: gains `lambda_psd=0.1`, `critic_type='standard'`, `noise_range=4*pi`; conditional critic initialization based on `critic_type`
2. `_train_one_epoch` (all three loops): broadcasting refactor — noise and par_light become `(batch_size, num_qubits)` tensors; `params_pqc` stays unbatched; output becomes `(batch_size, window_length)` directly; eval section fixed to use real PAR_LIGHT
3. `diff_psd_loss` (new static method): `torch.fft.rfft -> |FFT|^2 -> log(+1e-8) -> MSE`; operates on `(batch_size, window_length)` tensors; epsilon prevents log(0)
4. `define_critic_model_simple` (new method): 2-layer CNN (Conv1d 2->32 k=3; Conv1d 32->32 k=3; AdaptiveAvgPool1d; Linear 32->1); no dropout; ~5.5K params vs ~250K current
5. Config cell 28: new `LAMBDA_PSD`, `CRITIC_TYPE` constants; updated `NUM_LAYERS` value
6. New diagnostic cells: PAR_LIGHT conditioning intervention test (after training, before final evaluation)

### Critical Pitfalls

1. **Noise range fix invalidates HPO results (P-1)** — `lambda_gp=2.16`, `n_critic=9`, `lambda_acf=0.062` were tuned on `[0,2pi]` noise; switching to `[0,4pi]` changes the loss landscape fundamentally; run 200-epoch validation comparing EMD curves before proceeding; if EMD degrades >2x baseline (`best_emd=0.001137`), re-run focused HPO before adding any new features
2. **Evaluation section uses par_zeros (P-4)** — confirmed by direct code inspection at line 1526; all HPO and early-stopping metrics reflect unconditioned generation; fix eval to use real PAR_LIGHT; compare before/after to understand whether conditioning was ever working; this affects validity of all reported results
3. **Broadcasting breaks with conditional PAR_LIGHT inputs (P-3)** — `parameter-shift` broadcasting has known gradient bugs (PennyLane issue #4462); use `batch_input` transform with explicit `argnum` specification, not raw broadcasting; switch to `backprop` first; verify batched output matches sequential output element-by-element within 1e-6 before training
4. **Spectral loss gradient dominance kills WGAN training (P-2)** — PSD loss gradients in frequency domain can be O(10-100x) the Wasserstein loss; use log-space PSD with epsilon; start `lambda_psd=0` and ramp up over ~200 epochs; log all three loss components separately every eval epoch; if PSD component exceeds 10x Wasserstein component, reduce `lambda_psd`
5. **Circuit depth increase triggers barren plateau (P-5)** — at 5 qubits with strongly entangling layers, gradient variance decreases exponentially with depth; increase from 4->5->6 incrementally; monitor gradient norms per layer; stop if last-layer gradient norms fall below 1% of first-layer norms

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Regression Fixes and Baseline Validation

**Rationale:** Broken code and incorrect evaluation metrics invalidate all experiments. The noise range regression, par_zeros evaluation bug, and LAMBDA mismatch must be corrected before any metric can be trusted. HPO transferability must be empirically validated — this is a decision gate, not just a cleanup step.

**Delivers:** Correct training code with trustworthy metrics; fixed PAR_LIGHT evaluation; validated HPO baseline; decision on whether HPO re-run is required

**Addresses:** TS-1 (noise range — 3 line changes), TS-3 (mu/sigma rename), TS-4 (par_zeros eval fix), P-13 (LAMBDA mismatch — align default LAMBDA to HPO value)

**Avoids:** P-1 (HPO invalidation without awareness), P-4 (all metrics computed incorrectly)

**Gate before Phase 2:** 200-epoch validation run. EMD must be within 2x of HPO baseline. If not, re-run focused HPO with corrected noise range before proceeding.

### Phase 2: diff_method Switch and Broadcasting Restoration

**Rationale:** A ~12x training slowdown is a force multiplier against all subsequent experimentation. Without broadcasting, every experiment takes 12x longer — making the iterative tuning required for spectral loss (lambda_psd schedule, joint lambda_acf/lambda_psd grid search) prohibitively slow. `diff_method='backprop'` is a prerequisite for both broadcasting correctness and efficient spectral loss gradient computation, and must be implemented simultaneously with broadcasting.

**Delivers:** ~12x faster training epochs; correct gradient computation via backprop; prerequisite for all feature experiments

**Addresses:** TS-2 (broadcasting — refactor 3 loops), P-9 (parameter-shift negates speedup), P-7 (PSD cost with parameter-shift)

**Avoids:** P-3 (use `batch_input` transform with explicit `argnum`, not raw broadcasting; verify output matches sequential within 1e-6 before any training)

**Gate before Phase 3:** Batched output matches sequential output within 1e-6. Epoch time is less than 30% of pre-broadcasting time.

### Phase 3: Spectral/PSD Loss

**Rationale:** The root cause of variance collapse is that the generator has no loss signal penalizing wrong frequency content. The ACF lag-1 penalty captures only first-order temporal correlation — the PSD loss adds explicit gradient signal targeting the mid-frequency volatility that makes real OD curves "jagged." This is the highest-impact single addition. Requires Phase 2 for efficient training.

**Delivers:** Explicit frequency-domain gradient signal for generator; potential improvement in fake std toward real std (0.0104 -> 0.0218 target); new `LAMBDA_PSD` hyperparameter tuned empirically

**Addresses:** D-1 (PSD loss — `diff_psd_loss` static method + generator loss term), P-14 (compute PSD on scaled output, same representation critic sees), P-12 (acknowledge 6-bin limitation; consider concatenating windows if resolution is insufficient)

**Avoids:** P-2 (gradient dominance — warmup schedule + separate loss logging + log-PSD normalization), P-8 (loss weight interaction — joint `lambda_acf`/`lambda_psd` grid search, not isolated `lambda_psd` tuning)

**Gate before Phase 4:** EMD stays within 1.5x of Phase 2 baseline after adding PSD loss. Fake std trend moves toward 0.0218. No single loss component exceeds 10x another.

### Phase 4: Conditioning Verification and Capacity Tuning

**Rationale:** Conditioning verification is thesis-critical but requires a trained model and is purely diagnostic — it belongs after a stable baseline is established. Circuit depth tuning is secondary to loss signal quality (more layers without the right loss signal is "a bigger brush painting the wrong picture"). Both require the stable Phase 3 baseline to isolate effects.

**Delivers:** Empirical evidence of conditioning effectiveness (KS test, sweep plot across 5 PAR_LIGHT levels); verified or refuted thesis claim; optionally improved generator expressivity via 6-layer circuit (incremental increase from 4->5->6)

**Addresses:** D-3 (conditioning verification — intervention test, sweep test, gradient magnitude check), D-2 (layer count — change `NUM_LAYERS`, validate gradient norms at each depth), P-10 (PAR_LIGHT compression — test conditioning effect before optimizing encoding), P-5 (barren plateau — incremental increase with gradient monitoring), P-11 (checkpoint incompatibility — fresh training required, shape assertion on load)

**Gate and branch:** Conditioning verification is a binary decision point. If KS test shows conditioning is effective (p < 0.05), proceed to Phase 5. If ineffective, a conditioning refactor (data re-uploading for PAR_LIGHT at every layer, not just once after IQP encoding) may be required — this adds scope and should be scoped separately.

### Phase 5: Critic Architecture Simplification (Conditional)

**Rationale:** The 3,345:1 generator/critic parameter ratio is a red flag, but critic simplification is the most uncertain intervention. WGAN-GP already handles overpowered critics via the gradient penalty, and too-weak critics create their own instability. This phase is explicitly gated on whether Phases 1-3 have resolved variance collapse. If fake std approaches 0.0218 by Phase 3, Phase 5 may be unnecessary.

**Delivers:** Configurable `critic_type` flag (not a replacement of the existing critic); empirical A/B comparison over 200 epochs; fresh `lambda_gp` tuning for new architecture; removal of existing Dropout(0.2) from current CNN critic (valid regardless of which architecture is used)

**Addresses:** D-4 (critic simplification), P-6 (GP instability from too-simple critic — monitor GP term vs Wasserstein term ratio)

**Gate:** EMD and moment statistics (std, kurtosis) do not degrade vs Phase 4 baseline. GP loss does not dominate critic loss (should be less than 70% of total critic loss).

### Phase Ordering Rationale

- **Regressions before features:** The noise range regression and par_zeros bug mean all existing metrics are computed incorrectly. Adding new features on top of incorrect baselines produces meaningless results.
- **`diff_method` + broadcasting together in Phase 2:** They are co-dependent — `backprop` is required for broadcasting correctness (parameter-shift has known bugs), and switching to `backprop` without restoring broadcasting leaves the speedup unrealized.
- **PSD loss before capacity changes:** Loss signal quality is more fundamental than generator capacity. Post-HPO classical baselines with far more parameters also failed — the problem is what the generator optimizes for, not how much it can express.
- **Conditioning verification after stable baseline:** Requires a trained model; running it before Phase 3 improvements would measure conditioning quality against a suboptimal model.
- **Critic simplification last and conditional:** Most uncertain intervention, most dependent on empirical results from earlier phases. May be unnecessary if PSD loss resolves variance collapse.

### Research Flags

Phases requiring empirical validation during execution (not additional pre-planning research):

- **Phase 1 (HPO validation gate):** Whether `lambda_gp=2.16`, `n_critic=9`, `lambda_acf=0.062` transfer after noise range correction cannot be predicted. This is a genuine empirical gate — plan for the possibility of a focused HPO re-run.
- **Phase 2 (broadcasting with PAR_LIGHT inputs):** Confidence is MEDIUM on `batch_input` transform behavior with this specific circuit. Verify batched vs sequential output numerically before running any training.
- **Phase 3 (lambda_psd schedule and joint tuning):** The right `lambda_psd` and re-tuned `lambda_acf` cannot be determined analytically. Budget time for a small grid search over `(lambda_acf, lambda_psd)` pairs (e.g., 3x3 grid, 200 epochs each).
- **Phase 4 (conditioning refactor branch):** If the KS test shows conditioning is ineffective, a data re-uploading refactor for PAR_LIGHT is needed. This is a branch point that adds scope not yet sized.

Phases with standard patterns (implementation is straightforward):

- **Phase 1 (code fixes):** All 3 locations for noise range, mu/sigma rename, par_zeros fix, and LAMBDA alignment are concrete edits with known locations.
- **Phase 3 (PSD loss implementation):** `torch.fft.rfft` usage pattern is well-documented. Implementation sketch exists in ARCHITECTURE.md. Integration points are precise.
- **Phase 5 (critic architecture):** Standard WGAN-GP architecture patterns apply. `define_critic_model_simple` design is finalized in ARCHITECTURE.md.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All capabilities verified against installed PennyLane 0.44.0 and PyTorch 2.8.0. Zero new packages required. `torch.fft` autograd support confirmed in official PyTorch docs. `backprop` broadcasting confirmed working since PennyLane 0.31.0. |
| Features | MEDIUM-HIGH | Regression fixes are certain (confirmed by code inspection). PSD loss spectral theory is well-documented. PAR_LIGHT conditioning effectiveness is unknown until verification runs. Critic simplification impact is empirically uncertain. |
| Architecture | HIGH | Based on direct codebase analysis of the 653-line Cell 26 class. All integration points are concrete (specific lines, method signatures, tensor shapes). Existing evaluation cell demonstrates correct broadcasting convention. |
| Pitfalls | HIGH | 5 critical pitfalls, 6 moderate, 4 minor identified with specific detection methods. P-1 and P-4 confirmed by direct code inspection. P-3 supported by PennyLane issue tracker. P-5 supported by published barren plateau theory. |

**Overall confidence:** HIGH

### Gaps to Address

- **HPO transferability after noise range fix:** Whether optimized hyperparameters transfer to corrected noise range is not predictable. Phase 1 validation run resolves this — plan for either outcome.
- **PAR_LIGHT conditioning effectiveness:** The par_zeros bug means this has never been measured honestly. Phase 4 KS test is the first real measurement. If conditioning is broken, a refactor adds significant scope.
- **PSD loss at T=10 (6-bin resolution):** With only 6 frequency bins, PSD loss may overlap substantially with ACF lag-1 penalty. If correlation between the two terms exceeds 0.9 during training, consider concatenating consecutive windows (4x10=40 samples, 21 bins) before FFT to improve frequency resolution.
- **Correct broadcasting axis convention:** PennyLane uses `(num_qubits, batch_size)` convention (last axis is batch) rather than PyTorch's standard `(batch_size, num_qubits)`. The existing evaluation cell already demonstrates the correct convention and should be used as the reference implementation.

## Sources

### Primary (HIGH confidence)

- [PennyLane QNode API 0.44.0](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) — broadcasting rules, diff_method options, backprop requirements
- [PyTorch torch.fft Module Blog](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/) — autograd FFT support confirmed since PyTorch 1.7
- [torch.fft.rfft docs](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html) — one-sided FFT for real signals
- [Gulrajani et al., Improved WGAN (2017)](https://arxiv.org/abs/1704.00028) — no dropout in critic, gradient penalty theory, Lipschitz constraint
- [Lie algebraic theory of barren plateaus (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-49909-3) — gradient variance scaling with depth
- Direct codebase analysis: `qgan_pennylane.ipynb` Cell 26 (qGAN class, 653 lines), `hpo_runner.ipynb`, `hpo_retrain_runner.ipynb`, `results/hpo/best_params.json`

### Secondary (MEDIUM confidence)

- [PennyLane batch_input docs 0.44.0](https://docs.pennylane.ai/en/stable/code/api/pennylane.batch_input.html) — batch_input transform for non-trainable inputs; behavior with conditional inputs needs empirical validation
- [PennyLane Broadcasting Blog](https://pennylane.ai/blog/2022/10/how-to-execute-quantum-circuits-in-collections-and-batches/) — performance characteristics and axis conventions
- [Spectral GAN for Time Series (arXiv 2103.01904)](https://ar5iv.labs.arxiv.org/html/2103.01904) — spectral loss for time series GANs
- [PennyLane Barren Plateaus demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/) — gradient variance vs circuit depth
- [Quantum GAN with WGAN-GP (EPJ Quantum Technology 2025)](https://link.springer.com/article/10.1140/epjqt/s40507-025-00372-z) — hybrid quantum-classical WGAN-GP architecture
- [PennyLane Broadcasting Issues Forum](https://discuss.pennylane.ai/t/issues-with-backpropagation-when-using-parameter-broadcasting-with-pytorch/3333) — parameter-shift broadcasting gradient bugs

### Tertiary (LOW confidence / needs empirical validation)

- Optimal `lambda_psd` value — no literature precedent for this specific problem; must be determined by grid search during Phase 3
- Conditioning refactor via PAR_LIGHT data re-uploading at every layer — sound in principle but not tested in this specific circuit architecture; scoped only if Phase 4 verification shows conditioning is ineffective

---
*Research completed: 2026-03-13*
*Ready for roadmap: yes*
