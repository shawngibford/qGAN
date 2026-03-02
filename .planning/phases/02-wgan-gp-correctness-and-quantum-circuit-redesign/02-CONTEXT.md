# Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Restore ML theory compliance per Gulrajani et al. (2017), redesign the quantum circuit for universal approximation via data re-uploading, and unify output scaling so training and standalone generation produce identical results. All existing checkpoints are invalidated — this is a fresh training run. Requirements: BUG-02, BUG-03, PERF-01, PERF-04, PERF-05, WGAN-01, WGAN-02, WGAN-03, WGAN-04, WGAN-05, WGAN-06, WGAN-07, WGAN-08, QC-01, QC-02, QC-03, QC-04, QC-05, QUAL-06.

</domain>

<decisions>
## Implementation Decisions

### Quantum Circuit Architecture
- Keep IQP template — remove only the redundant pre-noise RZ gate (QC-01), preserve the IQP-style structure
- Data re-uploading uses the current NUM_LAYERS value (no change to layer count)
- Identical noise re-encoding at each layer (no affine transforms between re-uploads)
- Noise encoded via RX gates (different basis from the RZ variational gates, creates non-commutativity for expressiveness)
- Keep current entangling gate pattern between variational layers
- PauliX and PauliZ measurements concatenated: output dimension = 2 * NUM_QUBITS = WINDOW_LENGTH
- Uniform noise distribution over [0, 4pi], one independent noise value per qubit (latent dim = NUM_QUBITS)
- Use PennyLane parameter broadcasting for batch quantum circuit execution (PERF-05)
- Use `diff_method='backprop'` on explicitly constructed `default.qubit` device (PERF-01)

### Output Scaling and Denormalization
- Keep `*0.1` multiplicative scaling pattern, but make it a named hyperparameter: `GEN_SCALE = 0.1` in the config cell
- Single `denormalize()` function called by both training evaluation and standalone generation — consistency by construction
- Save mu and sigma in checkpoints so loaded models can denormalize without re-running preprocessing
- Claude's Discretion: storage location for mu/sigma (class attributes vs module-level), exact denormalization pipeline ordering (Claude traces preprocessing and constructs the correct inverse)

### WGAN-GP Training Configuration
- N_CRITIC = 5, LAMBDA = 10 (restored to paper values)
- Keep current learning rate values, ensure critic LR >= generator LR (WGAN-07)
- Adam optimizer betas = (0, 0.9) per Gulrajani et al.
- Remove ALL regularization from critic: dropout, batch normalization, and weight decay. Gradient penalty is the sole constraint
- LeakyReLU(0.2) activation in critic
- Claude's Discretion: critic layer sizes/depth — Claude examines current architecture and suggests changes only if clearly undersized/oversized

### Gradient Penalty
- Per-sample interpolation coefficients (each sample gets its own epsilon ~ Uniform(0,1))
- One-sided penalty: only penalize when gradient norm > 1 (not two-sided)
- Correct the inline GP computation in the training loop (no new standalone function)
- Apply GP to interpolated samples in Conv1D input shape (batch, channels, length)

### EMD Computation
- Compute via `wasserstein_distance(real_samples, fake_samples)` on raw 1D arrays (WGAN-04)
- Flatten all windows into a single 1D array for comparison (not per-position)
- Generate same number of samples as real training set for fair comparison
- Use all real data every evaluation (no subsampling)
- Compute in normalized space (same space the model trains in)

### Early Stopping
- Monitor EMD (not critic loss) (WGAN-06)
- Patience: 50 evaluation cycles (500 epochs at eval-every-10)
- Warmup: 100 epochs before early stopping starts monitoring
- Any EMD decrease counts as improvement (no min_delta threshold)
- On trigger: revert to best checkpoint (load best_checkpoint.pt)
- Print detailed message: epoch, best EMD value, best epoch, patience exhaustion info

### Evaluation and Monitoring
- Evaluate every 10 epochs (PERF-04)
- Log all four categories every eval cycle: loss values (critic, generator, GP magnitude), EMD + stylized facts, sample comparison plot (inline in notebook), gradient norm stats (mean/max for critic and generator)
- Text-based progress updates printed every eval cycle (every 10 epochs)
- Post-training summary cell: consolidates loss curves, EMD trajectory, final stylized facts comparison, real vs generated histograms

### Stylized Facts Validation
- All four essential facts: heavy tails (kurtosis), volatility clustering (ACF of squared returns), absence of autocorrelation (ACF of returns), leverage effect (cross-correlation of returns and squared returns)
- Compute at both window-level AND stitched time series level, report both for comparison
- Verify formula correctness (mathematical audit, no specific paper reference required)
- Compute during training (every eval cycle) AND in post-training summary cell

### Histogram and Visualization
- Bin edges computed from real data range, shared between real and generated (WGAN-05)
- Compute bins once before training, reuse every evaluation
- Density normalization (area = 1) for fair comparison regardless of sample sizes
- Post-training summary shows histograms in both normalized and denormalized (original price) space

### Checkpoint Strategy
- Save on best EMD only (single file: `best_checkpoint.pt`)
- Model state only: weights, optimizer states, epoch, mu, sigma. No training history
- Fixed filename (overwritten on each improvement)

### WINDOW_LENGTH and Dimension Coupling
- WINDOW_LENGTH = 2 * NUM_QUBITS, computed automatically in config cell (QC-05)
- Critic Conv1D auto-derives input dimension from WINDOW_LENGTH
- Keep Conv1D architecture (not switching to MLP)
- Assert at model initialization that generator output dimension matches critic input dimension
- Claude's Discretion: whether data pipeline rolling window step needs to be updated to reference WINDOW_LENGTH from config

### Claude's Discretion
- mu/sigma storage location (class attributes vs module-level constants)
- Exact denormalization pipeline ordering (Claude traces preprocessing and constructs correct inverse)
- Critic layer sizes/depth adjustments (only if current architecture is clearly wrong for the task)
- Data pipeline alignment with new WINDOW_LENGTH value

</decisions>

<specifics>
## Specific Ideas

- User wants to change as little as possible while making results correct — minimal structural changes, maximum theory compliance
- One-sided gradient penalty chosen over standard two-sided (penalize only when gradient norm exceeds 1)
- User wants all four stylized facts computed at both window-level and stitched series level to compare whether stitching introduces or destroys statistical properties
- Post-training summary cell should be suitable for papers/presentations — both normalized and denormalized histograms
- Checkpoint is model-state-only (no training history) — training history lives in notebook output

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-wgan-gp-correctness-and-quantum-circuit-redesign*
*Context gathered: 2026-03-02*
