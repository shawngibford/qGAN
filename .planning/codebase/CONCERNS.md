# Codebase Concerns

**Analysis Date:** 2026-02-26

## Tech Debt

### Parameter Type Mismatch Bug (Fixed but Recurring Risk)
- **Issue:** When loading checkpoints, quantum parameters (`params_pqc`) are stored as Tensors but require `nn.Parameter` wrapping for gradient computation. This bug occurred in Phase 2C after 13+ hours of training.
- **Files:** `load_checkpoint_phase2c.py` (line 33), `early_stopping_code.py` (line 144), `archive/qgan_pennylane.py`
- **Impact:** Model loading fails at end of training with `TypeError`, requiring manual parameter wrapping. Training completes successfully but evaluation becomes blocked.
- **Fix approach:**
  - Wrap parameters as `nn.Parameter` immediately when saving checkpoints
  - Update checkpoint serialization to preserve parameter types
  - Add type assertion in load functions to validate parameter types before usage
  - Current fix in place but not automated in checkpoint save logic

### Hardcoded File Paths in Archive Code
- **Issue:** Archived Python scripts contain hardcoded absolute paths specific to original developer's machine
- **Files:** `archive/qwxlstmgan_pennylane.py` (multiple instances), `archive/qgan_pennylane.py`
- **Impact:** Archive code cannot be reused without manual path editing. Makes reproducibility difficult if reverting to older approaches.
- **Fix approach:** Extract paths to configuration variables at top of file, support relative paths from project root, or use `pathlib.Path` with `__file__` for relative resolution

### Incomplete Error Handling in Checkpoint Loading
- **Issue:** `load_checkpoint_phase2c.py` lacks try/except blocks around `torch.load()` and state dict restoration
- **Files:** `load_checkpoint_phase2c.py` (lines 18-48)
- **Impact:** If checkpoint file is corrupted, optimizer state is missing, or discriminator architecture changes, loading fails with cryptic error messages
- **Fix approach:** Add try/except with detailed error messages, validate checkpoint structure before loading, provide graceful fallbacks

### Missing Gradient Computation Path Validation
- **Issue:** No explicit validation that gradients flow through quantum circuit to classical discriminator after checkpoint loading
- **Files:** Notebooks and training loop code
- **Impact:** Could load model with broken gradient connections undetected, resulting in zero parameter updates
- **Fix approach:** After loading, run test forward/backward pass to verify gradients, log gradient norms

---

## Known Bugs

### Parameter Type Conversion on Checkpoint Load (Phase 2C)
- **Symptoms:**
  - Training completes successfully (535 epochs, early stopped at epoch 485)
  - Model checkpoints saved with 13+ hours of training data
  - Loading best model triggers `TypeError: expected Tensor, got Parameter` or similar
  - Evaluation code cannot execute
- **Files:** `load_checkpoint_phase2c.py`, `early_stopping_code.py`, `qgan_pennylane_qutrit_phase2c.ipynb`
- **Trigger:** Run checkpoint loading after training completes
- **Workaround:** Manually wrap tensor as parameter: `model.params_pqc = nn.Parameter(checkpoint['params_pqc'])`
- **Status:** Documented in `PHASE2C_RECOVERY_INSTRUCTIONS.md` but requires manual intervention each time

### Potential Data Leakage Between Train/Test Splits
- **Issue:** Data preprocessing (normalization, Lambert W transformation) applied globally before train/test split
- **Files:** Notebook cells in `qgan_pennylane.ipynb`, `qgan_pennylane_qutrit_phase2c.ipynb`
- **Impact:** Statistics from test data influence training, causing optimistic performance metrics
- **Status:** Not explicitly verified if split happens before or after preprocessing

---

## Security Considerations

### No Input Validation on Data Files
- **Risk:** CSV files (`data.csv`, `fake.csv`, `real.csv`) loaded without schema validation
- **Files:** Data loading cells in main notebooks
- **Current mitigation:** Assumes data format is correct, relies on pandas error handling
- **Recommendations:**
  - Add CSV schema validation (expected columns, data types)
  - Validate data ranges (no Inf/NaN after preprocessing)
  - Add checksum or file integrity verification for cached data

### Checkpoint Files Not Encrypted or Signed
- **Risk:** Checkpoint files contain full model parameters and optimizer state. No integrity verification.
- **Files:** `checkpoints_phase2c/` directory
- **Current mitigation:** None - file system permissions only
- **Recommendations:**
  - Add SHA256 hash to checkpoint metadata for integrity verification
  - Document security implications of checkpoint sharing
  - Consider encryption if checkpoints contain sensitive bioprocess data

### Hardcoded Random Seeds
- **Risk:** While reproducibility-focused, fixed seeds disable any statistical testing
- **Files:** Multiple notebooks and scripts (seed = 42)
- **Current mitigation:** None - by design for reproducibility
- **Recommendations:**
  - Add parameter to enable seed randomization for uncertainty quantification
  - Document seed fixing requirement for paper/publication

---

## Performance Bottlenecks

### Checkpoint Storage Explosion
- **Problem:** Each training epoch saves ~5.8MB checkpoint files, creating 129MB+ for 535 epochs (11 checkpoints saved)
- **Files:** `checkpoints_phase2c/` (11 × 5.8MB = 63.8MB actual, more with older phases)
- **Cause:** Full model state dict, optimizer states, and parameter copies stored per checkpoint
- **Current capacity:** Repository is 1.2GB total; checkpoints consume ~10%
- **Improvement path:**
  - Implement differential/delta checkpoints (only save changes from previous)
  - Compress checkpoint files with gzip (likely 30-50% size reduction)
  - Archive old checkpoints after training completes
  - Keep only best + last N checkpoints, delete intermediate saves

### Large Notebook Files (2+ MB each)
- **Problem:** 5 main notebook files, each 2-2.3MB due to output cells with plots and training logs
- **Files:** `qgan_pennylane.ipynb`, `qgan_pennylane_qutrit_phase2*.ipynb`, `qgan_pennylane qutrit.ipynb`
- **Cause:** Notebook output cells store execution history (plots, numeric output) in base64
- **Impact:** Slow to load, slow to edit, inefficient git tracking
- **Improvement path:**
  - Clear all outputs before committing: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace`
  - Move plots to separate directory, reference from notebook
  - Consider converting to Python scripts for non-interactive code
  - Use `nbstripout` in git hooks to auto-clean outputs

### Early Stopping Convergence May Be Premature
- **Problem:** Early stopping with patience=50 stopped training at epoch 535 (best at 485), preventing potential further improvements
- **Files:** `PHASE2C_RECOVERY_INSTRUCTIONS.md`, `early_stopping_code.py`
- **Cause:** 50-epoch patience is arbitrary; no learning curve analysis to validate it's optimal
- **Impact:** May leave 0.1-0.5% performance on table
- **Improvement path:**
  - Analyze loss curve to detect true plateau vs temporary stagnation
  - Plot loss with 10/25/50/100 epoch moving averages
  - Implement adaptive patience (increase if training continues improving)
  - Set separate patience for different phases (higher early, lower late)

### No GPU/Device Optimization Documented
- **Problem:** Code uses PyTorch but doesn't specify device (CPU vs GPU) selection strategy
- **Files:** Training loop in notebooks
- **Impact:** If running on CPU, training is 100-1000x slower; Phase 2C took 13+ hours
- **Improvement path:**
  - Add explicit device selection: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
  - Profile performance to find bottlenecks
  - Consider mixed precision training (float16) to reduce memory

---

## Fragile Areas

### Quantum Circuit Execution (Dependency on PennyLane Internals)
- **Files:** All circuit definition code in notebooks
- **Why fragile:**
  - Custom `TRX`, `TRY`, `TRZ` operations on qutrit subspaces are PennyLane-specific
  - Circuit depends on internal tensor contraction and gradient autodiff
  - Any PennyLane version upgrade could break tensor shape assumptions
  - No explicit compatibility testing
- **Safe modification:**
  - Before updating PennyLane version, run full training pipeline as regression test
  - Add explicit version pinning: `pennylane==0.32.0` (currently `>=0.32.0`)
  - Wrap circuit in integration tests with fixed expected gradient values
- **Test coverage:** No automated circuit output validation; relies on manual DTW metric check

### Data Preprocessing Pipeline (Lambert W + Normalization)
- **Files:** Data transformation cells in main notebooks
- **Why fragile:**
  - Multiple sequential transformations (log returns → Lambert W → z-score normalization)
  - Lambert W transformation poorly documented - unclear what arguments are optimal
  - No validation that output distribution matches expected properties
  - Inverse transform not documented, making data reconstruction difficult
- **Safe modification:**
  - Implement and test inverse transformations for all preprocessing steps
  - Add assertions checking output statistics (mean ≈ 0, std ≈ 1 after norm)
  - Document Lambert W parameter choices with justification
  - Create reusable preprocessing module with docstrings
- **Test coverage:** Gaps - no unit tests for data transformations

### Early Stopping Integration
- **Files:** `early_stopping_code.py`, `EARLY_STOPPING_GUIDE.md`, notebook training loops
- **Why fragile:**
  - Early stopping logic coupled to specific qgan_model attribute access (`.critic_loss_avg`)
  - Assumes model has `discriminator`, `params_pqc`, `c_optimizer`, `g_optimizer` attributes
  - Will fail silently if model structure changes (e.g., optimizer renamed)
  - Checkpoint restore assumes discriminator can be serialized/loaded
- **Safe modification:**
  - Add assertions checking model has required attributes before training
  - Create model interface/protocol defining required attributes
  - Add verbose logging of what's being checkpointed
- **Test coverage:** No tests for edge cases (missing attributes, corrupted checkpoints)

### Generator/Discriminator Architecture Assumptions
- **Files:** QGAN implementation in notebooks
- **Why fragile:**
  - Discriminator architecture likely hardcoded (not clear from partial code review)
  - Window length assumptions (10 for Phase 1, 8 for Phase 2, 6 for Phase 2C) not parameterized
  - Circuit structure tightly coupled to specific qubit/qutrit counts
- **Safe modification:**
  - Extract all hyperparameters (window_size, num_qubits, num_layers) to config section at top
  - Make discriminator architecture configurable
  - Add validation that circuit inputs match window size
- **Test coverage:** No parametric tests verifying different configurations work

---

## Scaling Limits

### Quantum Circuit Expressibility Plateau
- **Current capacity:** 3-5 qutrits, 1-3 layers achieves good results
- **Limit:** Increasing qutrits/layers shows diminishing returns or degradation (Phase 1 worse than Phase 2C despite more resources)
- **Scaling path:**
  - Investigate why 5 qubits + 3 layers underperforms 3 qutrits + 1 layer (design issue, not just expressibility)
  - Test hybrid approaches (classical preprocessing before quantum circuit)
  - Consider trainable data encoding instead of fixed IQP
  - Explore entanglement patterns beyond all-to-all

### Dataset Size Dependency
- **Current capacity:** ~770 training samples (from bioprocess time series)
- **Unknown:** Minimum and maximum effective sample counts
- **Impact:** DTW metric may be unstable with small test sets; confidence intervals not computed
- **Scaling path:**
  - Quantify how DTW varies with train/test split ratio
  - Generate synthetic bootstrap samples to estimate metric variance
  - Add statistical significance testing (confidence intervals on DTW)
  - Determine if more data improves performance

### Training Time vs Convergence
- **Current capacity:** 535 epochs takes ~13.3 hours on available hardware
- **Limit:** Early stopping at 485 epochs suggests further improvements may exist with different patience settings
- **Scaling path:**
  - Profile to identify which operations dominate training time (quantum circuit evaluation vs discriminator)
  - Implement batch-mode quantum circuit evaluation
  - Consider classical-only baselines to compare training speed

### Memory Footprint of Checkpoints
- **Current capacity:** 1.2GB total repository, checkpoints taking ~10%
- **Limit:** 100+ epochs of checkpoints would exceed typical development machine storage
- **Scaling path:** Implement checkpoint pruning strategy (keep best + last 5, delete rest), compress old checkpoints

---

## Dependencies at Risk

### Outdated Testing and Linting Infrastructure
- **Risk:** `requirements.txt` comments out pytest, black, flake8 (development dependencies)
- **Impact:** No enforced code quality, no automated testing pipeline
- **Files:** `requirements.txt`
- **Migration plan:**
  - Uncomment dev dependencies, create separate `requirements-dev.txt`
  - Set up pre-commit hooks for linting/formatting
  - Add pytest configuration with basic sanity tests

### PennyLane Version Pinning Too Loose
- **Risk:** `pennylane>=0.32.0` allows major version jumps with breaking changes
- **Impact:** Future notebooks may fail with newer PennyLane without warning
- **Files:** `requirements.txt`
- **Migration plan:**
  - Pin minor version: `pennylane>=0.32.0,<0.40.0` to limit breaking changes
  - Document tested PennyLane versions
  - Add CI to test against multiple PennyLane versions

### PyTorch Version Dependency
- **Risk:** `torch>=2.0.0` may have breaking changes in tensor/gradient API between minor versions
- **Impact:** Code assumes PyTorch 2.x behavior; 1.x users cannot use it
- **Files:** All notebook training code
- **Migration plan:**
  - Add torch version compatibility layer if supporting 1.x is needed
  - Document minimum tested torch version (currently appears to be 2.0+)
  - Add device handling abstraction in case CUDA/CPU behavior differs between versions

---

## Missing Critical Features

### No Model Validation Strategy
- **Problem:** Only DTW metric used for validation; no cross-validation, no confidence intervals
- **Blocks:** Statistical significance claims; cannot determine if improvements are within noise
- **Files:** All training/evaluation code
- **Solution:** Implement k-fold cross-validation, compute DTW standard deviation

### No Baseline Comparisons
- **Problem:** GAN performance only compared against Phase 1/2 variants, not against other synthetic data methods (VAE, classical AR models, etc.)
- **Blocks:** Publication-ready results; cannot claim advantage over alternatives
- **Files:** Evaluation sections in notebooks
- **Solution:** Implement classical baselines (ARIMA, VAE), compare DTW/statistical metrics

### No Hyperparameter Optimization Framework
- **Problem:** All hyperparameters (learning rates, circuit depth, layer count) manually tuned
- **Blocks:** Scaling to new datasets; reproducibility of optimization process
- **Files:** Training configuration in notebooks
- **Solution:** Use Optuna or similar for systematic hyperparameter search

### Incomplete Documentation of Lambert W Transform
- **Problem:** `scipy.special.lambertw` used for data preprocessing but parameters and purpose not documented
- **Blocks:** Reproducibility; cannot explain methodology to readers
- **Files:** Data preprocessing cells
- **Solution:** Add docstring explaining Lambert W (handles skewness in bioprocess data), justify parameter choices

### No Data Provenance Tracking
- **Problem:** `data.csv` included but origin, collection method, and preprocessing steps not documented
- **Blocks:** Others cannot verify results; bioprocess data may be proprietary
- **Files:** `data.csv` and related notebooks
- **Solution:** Add DATA_README.md documenting source, collection date, any deidentification applied

---

## Test Coverage Gaps

### No Unit Tests for Preprocessing
- **What's not tested:** Lambert W transformation, z-score normalization, log returns calculation
- **Files:** Data transformation cells in notebooks (not in library)
- **Risk:** If preprocessing logic changes, no automated validation
- **Priority:** High - preprocessing is critical to reproducibility

### No Circuit Architecture Tests
- **What's not tested:**
  - Quantum circuit produces expected output dimensions
  - Gradients flow correctly through circuit
  - Circuit works with different qutrit/qubit counts
- **Files:** Circuit definition code in notebooks
- **Risk:** Architecture bugs only caught by full training (expensive)
- **Priority:** High - circuit is core component

### No Checkpoint Serialization Tests
- **What's not tested:**
  - Checkpoint save/load cycle preserves model state exactly
  - Parameter types preserved through serialization
  - Optimizer state recovers correctly
- **Files:** Checkpoint code in notebooks and `early_stopping_code.py`
- **Risk:** Data loss, training inconsistencies (caught only at end)
- **Priority:** High - Phase 2C bug was due to missing this test

### No Data Validation Tests
- **What's not tested:**
  - CSV files have expected schema and no corruption
  - Data ranges are valid (no Inf/NaN in critical columns)
  - Preprocessing produces expected output statistics
- **Files:** Data loading and preprocessing cells
- **Risk:** Silent failures with corrupted data
- **Priority:** Medium - catch input problems early

### No Integration Tests for Full Training Pipeline
- **What's not tested:**
  - Full training from data load to model evaluation runs without errors
  - DTW metric calculation is reproducible
  - Different configurations (Phase 1/2/2C) all work
- **Files:** All notebooks
- **Risk:** Integration bugs only found by manual testing
- **Priority:** Medium - expensive but valuable for regression detection

### No Performance/Regression Tests
- **What's not tested:**
  - Training time stays within expected bounds
  - DTW scores don't degrade unexpectedly
  - Convergence speed remains consistent
- **Files:** Training loop in notebooks
- **Risk:** Silent regressions in performance (no early warning)
- **Priority:** Low - nice-to-have but not critical

---

## Documentation Issues

### Notebook Code Not Modularized
- **Issue:** All code lives in notebooks; no reusable library modules
- **Impact:** Code duplication across notebooks, difficult to maintain
- **Solution:** Extract core functions (circuit building, training loop, preprocessing) into `qgan/` package

### Early Stopping Embedded in Multiple Files
- **Issue:** Early stopping logic defined in `early_stopping_code.py`, `EARLY_STOPPING_GUIDE.md`, and copied into notebooks
- **Impact:** Changes to early stopping require updates in multiple places
- **Solution:** Move to single `qgan/training/early_stopping.py` module

### Phase-Specific Documentation Scattered
- **Issue:** Phase 1, 2, 2B, 2C results spread across separate markdown files; no unified architecture document
- **Impact:** Difficult to understand evolution and current state
- **Solution:** Create `ARCHITECTURE.md` documenting all phases and when each is used

---

*Concerns audit: 2026-02-26*
