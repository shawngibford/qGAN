# Training Protocol — QWGAN-GP (v1.1 unconditioned baseline)

> **Source of truth:** all numerical constants below are imported from
> `revision/core/__init__.py` (with non-`__init__` quantities cited from
> `revision/core/training.py`, `revision/core/models/quantum.py`, and
> `revision/core/models/critic.py`). Update those files to change values;
> this document tracks them via per-row line citations.

This protocol describes the QWGAN-GP training run that produced the
`unconditioned_wgan` checkpoint (`qgan_pennylane.ipynb` cell 65,
HPO-tuned in v1.1 Phase 4). All numerical values below are HPO-tuned
defaults that remain in force for v2.0 unless explicitly noted.

## Optimizer & Schedule

| Constant | Value | Source |
|----------|-------|--------|
| `N_CRITIC` | 9 critic steps per generator step | `revision/core/__init__.py:11` |
| `LAMBDA` (gradient penalty coeff) | 2.16 | `revision/core/__init__.py:12` |
| `LR_CRITIC` | 1.8046 × 10⁻⁵ (= `1.8046e-05`) | `revision/core/__init__.py:13` |
| `LR_GENERATOR` | 6.9173 × 10⁻⁵ (= `6.9173e-05`) | `revision/core/__init__.py:14` |
| Optimizer | Adam, `betas=(0.0, 0.9)` | `revision/core/training.py:233-234` |
| `NUM_EPOCHS` | 2000 | `revision/core/__init__.py:20` |
| `BATCH_SIZE` | 12 | `revision/core/__init__.py:21` |
| `EVAL_EVERY` | 10 epochs | `revision/core/__init__.py:23` |

`N_CRITIC=9` and `LAMBDA=2.16` are v1.1 HPO-tuned values; the high
`N_CRITIC` reflects the WGAN-GP convention that the critic should
approximate the Wasserstein distance well before each generator update.
Both learning rates are also HPO-tuned and intentionally asymmetric —
the generator updates roughly 4× more aggressively than the critic per
Adam step to compensate for the 9:1 critic-to-generator step ratio.
Adam betas `(0.0, 0.9)` follow Gulrajani et al. 2017's WGAN-GP
recommendation (zero first-moment memory avoids stale-gradient drift
against the moving critic target). The 2000-epoch budget with
`BATCH_SIZE=12` and 384 training windows yields ≈ 32 mini-batches per
epoch and ≈ 64 k generator updates total.

## Early-Stopping

| Property | Value | Source |
|----------|-------|--------|
| Class | `EarlyStopping` (custom, EMD-monitored) | `revision/core/training.py:79-175` |
| Monitored metric | EMD on log-returns (raw-sample Wasserstein) | `revision/core/eval.py:25-36` |
| `patience` (default) | 50 eval cycles ≡ 500 epochs at `EVAL_EVERY=10` | `revision/core/training.py:96` |
| `warmup_epochs` (default) | 100 epochs (no monitoring during warmup) | `revision/core/training.py:97` |
| Checkpoint scheme | save-best-EMD, reload on stop | `revision/core/training.py:142-175` |

EMD is the headline distributional-fidelity metric for this study; it is
computed via `scipy.stats.wasserstein_distance` on raw real-vs-synthetic
samples (NOT histograms — v1.0 design lock). Monitoring EMD rather than
critic loss avoids the moving-target ambiguity intrinsic to adversarial
objectives. **Methodological caveat (R1-M5 calibration):** the EMD
early-stop metric is computed on the same distribution used for training;
a true held-out validation EMD is not available because the
single-campaign dataset is too small to support a held-out split (see
`revision/docs/dataset_stats.md`). The 100-epoch warmup avoids
selecting checkpoints from the noisy early-training regime; the
50-eval-cycle patience (≈ 500 epochs) is generous on the 2000-epoch
budget, accepting longer training runs in exchange for stable best-EMD
selection.

## Quantum Circuit

| Property | Value | Source |
|----------|-------|--------|
| Backend | PennyLane `default.qubit`, `shots=None` (analytic statevector) | `revision/core/models/quantum.py:64` |
| Differentiation | `diff_method="backprop"` | `revision/core/models/quantum.py:43, 77` |
| `NUM_QUBITS` | 5 | `revision/core/__init__.py:17` |
| `NUM_LAYERS` | 4 (strongly-entangled blocks) | `revision/core/__init__.py:18` |
| `WINDOW_LENGTH` | 10 (= 2 × `NUM_QUBITS`) | `revision/core/__init__.py:19` |
| Encoding | IQP-style data-reuploading + strongly-entangling ansatz | `revision/core/models/quantum.py` |
| Latent-noise range | [0, 4π] (NOT [0, 2π]; v1.1 Phase 4) | `revision/core/__init__.py:32-33` (`NOISE_LOW`, `NOISE_HIGH = 4*math.pi`) |
| `GEN_SCALE` | 1.0 (output rescaling) | `revision/core/__init__.py:22` |
| PQC trainable parameter count | 75 (= 5 + 4 × 15 + 10) | verified Phase 8 (`08-VERIFICATION.md`) |

The 5-qubit / 4-layer choice yields a statevector of dimension 2⁵ = 32,
well within local-Mac simulator memory, while 75 trainable parameters
target an expressibility regime that is non-trivial without exceeding
classical PQC trainability limits. `diff_method="backprop"` replaced
parameter-shift in v1.1 Phase 5 because of PennyLane #4462 broadcasting
gradient bugs in the parameter-shift path; backprop is exact on the
analytic statevector and gradient-stable under batched inputs. The
[0, 4π] noise range (v1.1 Phase 4) doubles the standard [0, 2π]
convention to give the encoding layer more dynamic range; this was
HPO-validated against [0, 2π] and showed improved generator diversity.

## Critic (1D-CNN)

| Property | Value | Source |
|----------|-------|--------|
| Block 1 | `Conv1d(1 → 64, kernel_size=10, padding=5)` | `revision/core/models/critic.py:46` |
| Block 2 | `Conv1d(64 → 128, kernel_size=10, padding=5)` | `revision/core/models/critic.py:49` |
| Block 3 | `Conv1d(128 → 128, kernel_size=10, padding=5)` | `revision/core/models/critic.py:52` |
| Pooling | `AdaptiveAvgPool1d(1)` | `revision/core/models/critic.py:56` |
| Head | `Linear(128 → 32) → LeakyReLU → Dropout → Linear(32 → 1)` | `revision/core/models/critic.py:59-63` |
| `DROPOUT_RATE` | 0.2 (configurable) | `revision/core/__init__.py:24` |
| Precision | `float64` (`.double()`) | `revision/core/models/critic.py:67` |

The critic is a 1D-CNN with three convolutional blocks at fixed
`kernel_size=10` matching `WINDOW_LENGTH`; receptive fields span the
full input window in the first layer. `AdaptiveAvgPool1d(1)` reduces
the temporal axis to a single per-channel summary before the linear
head. Lipschitz constraint is enforced via the two-sided gradient
penalty (see § Gradient Penalty), NOT weight clipping. The critic
runs in `float64` for numerical-gradient stability under the high
`N_CRITIC=9` ratio.

## Gradient Penalty

| Property | Value | Source |
|----------|-------|--------|
| Type | Two-sided: `mean(((‖∇‖₂ − 1)²))` | `revision/core/training.py:30-73` |
| Coefficient λ | 2.16 (= `LAMBDA`) | `revision/core/__init__.py:12` |
| Interpolation α | sampled per-sample from `U(0, 1)`, broadcast over remaining dims | `revision/core/training.py:54-60` |
| Gradient target | 1 (unit-norm penalty) | `revision/core/training.py:72` |

Two-sided gradient penalty (Gulrajani et al. 2017) enforces the
1-Lipschitz constraint required by the Wasserstein dual. The penalty
coefficient `λ=2.16` is HPO-tuned (v1.1 Phase 4) — slightly above the
canonical λ=10 baseline; the lower value paired with `N_CRITIC=9` was
empirically more stable than the canonical λ=10 / N_CRITIC=5 setting
on this dataset.

## Reproducibility

| Property | Value | Source |
|----------|-------|--------|
| Seed (default) | 42 | `revision/core/training.py:188` |
| `torch.manual_seed(seed)` | yes | `revision/core/training.py:211` |
| `np.random.seed(seed)` | yes | `revision/core/training.py:212` |
| `random.seed(seed)` | yes | `revision/core/training.py:213` |
| `torch.cuda.manual_seed_all(seed)` | yes (no-op on CPU/MPS) | `revision/core/training.py:214-215` |
| `DITHER` (data pipeline) | 0.005 | `revision/core/__init__.py:27` |
| `DITHER_SEED` | 42 | `revision/core/__init__.py:28` |

All randomness sources are seeded inside `train_wgan_gp` before
parameter initialization to guarantee per-seed determinism on a fixed
machine. `DITHER=0.005` adds U(−0.005, +0.005) noise during data
preprocessing to break OD-value ties before log-ratio computation;
`DITHER_SEED=42` keeps the dither itself reproducible. Multi-seed
sweeps (≥ 5 seeds) for headline results are produced by Phase 12
(SENS-03), reusing this single-seed framework.

## Analytic-vs-Shot Distinction

All Phase 9 results use **analytic statevector simulation** (PennyLane
`default.qubit` with `shots=None`). Shot-noise behavior is reported
**separately in Phase 12 (SENS-01)** at `{analytic, 8192, 1024}`
shots; that analysis is out of scope for Phase 9. This protocol
document, the Phase 8 parity check, the EVAL-06 round-trip check
(Phase 9), and the Phase 09.1 preprocessing ablation all run in the
analytic-statevector regime.
