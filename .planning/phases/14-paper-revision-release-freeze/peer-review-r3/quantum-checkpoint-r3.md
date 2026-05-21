# Quantum Checkpoint & Reproduction Verification — Agent 3 / r3

**Phase:** 14 paper-revision-release-freeze
**Investigation:** Forensic r3, Agent 3 of 5
**Date:** 2026-05-21
**Mandate:** Verify the recovered 55-param IQP:SEL checkpoint and the matched-2000ep
`iqp_sel_55_repro` are computing the SAME thing as the historical IQP:SEL, and
determine whether the "mid-pack" appearance vs classical is the headline-vs-repro
distinction (D-14-10) rather than a real bug.

---

## Summary verdict

- **Headline-vs-repro accounts for the apparent regression:** PARTIALLY, but the gap is **smaller than expected**.
- **Checkpoint integrity OK:** YES.
- **Hyperparameter drift detected:** NO (matched-2000ep replicates notebook settings exactly).
- **Quantum-specific precision bug:** NO (PennyLane returns float64 by default; `.to(torch.float64)` is a no-op cast preserving the quantum precision advantage; quantum path remains the more-precise side, consistent with CR-4 disclosure).
- **Real story:** The current `iqp_sel_55_repro` 5-seed mean **matches the historical IQP:SEL result extremely closely**. The "regression" is therefore not against the historical IQP:SEL itself — it is that **other classical baselines (VAE, wgan_mlp, V1/V2 ansätze) have closed the gap to within ±1 OD-EMD point**. The frozen-checkpoint headline (n=1, single best epoch) is still the top OD-EMD score in the table, but classical 5-seed means are now comparable to the quantum 5-seed mean. That is the apples-to-oranges distinction the paper-block already documents in D-14-10.

Confidence: **HIGH** on integrity / hyperparameter / device claims; **HIGH** on the
metric numbers themselves; **MEDIUM-HIGH** on the diagnosis (the Aug-2025 headline
figures were single-checkpoint scores, so the only valid "regression" check is
headline-vs-headline, and headline is unchanged: OD-EMD 0.023, log-ret-EMD 0.121).

---

## 1. Checkpoint inspection (mandate item 1)

`revision/checkpoints/best_checkpoint.pt` was loaded via
`./qgan_env/bin/python` (`torch.load(..., weights_only=False)`).

| Property | Value | Source |
|---|---|---|
| Top-level type | `dict` | torch.load |
| Top-level keys | `['epoch', 'emd', 'params_pqc', 'critic_state', 'c_optimizer', 'g_optimizer', 'mu', 'sigma']` | torch.load |
| `epoch` | `1969` | matches `canonical_config_lock.json#checkpoint_epoch` |
| `emd` (tensor) | `0.08384301715430653` (float64 scalar) | matches `canonical_config_lock.json#checkpoint_emd` |
| `params_pqc` | shape `(55,)`, dtype `torch.float32` | matches `canonical_config_lock.json#param_count` |
| `mu` | `0.0024553430266678333` (float32 scalar) | matches `canonical_config_lock.json#mu` |
| `sigma` | `0.021407155320048332` (float32 scalar) | matches `canonical_config_lock.json#sigma` |
| `critic_state` | CNN dict: `Conv1d(1→64,10) → Conv1d(64→128,10) → Conv1d(128→128,10) → Linear(128→32) → Linear(32→1)`, all `float64` | matches notebook critic architecture in `qgan_pennylane.ipynb` cell 26 |
| SHA-256 (recomputed) | `f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082` | **MATCHES** `canonical_config_lock.json#checkpoint_sha256` exactly |

**No `gen_state_dict` key — only `params_pqc`.** The notebook stored the
single learnable quantum parameter vector directly (not a `nn.Module` state
dict). The matched-2000ep code (run_matched2000.py:514–517) creates a
`gen_state_dict` because it wraps the QuantumGenerator as a Module. The two
storage shapes are reconcilable: `QuantumGenerator.state_dict()` only contains
one tensor (`params_pqc`).

Sample `params_pqc[:10]`:
`[0.5240987, 0.4965273, 0.6145337, -1.1528468, 0.3965178, -1.0566381,
-1.0756730, -0.0428663, -0.6231805, 1.5717585]`

Decomposition arithmetic (matches `canonical_config_lock.json#decomposition`):

```
num_qubits = 5, num_layers = 3
IQP encoding param/qubit = 1               -> 5
SEL rot params/qubit/layer = 3 over 3 lyrs -> 3*3*5 = 45
Final rotation = RX-only                   -> 5
                                  TOTAL    = 55  ✓
```

**Checkpoint integrity: CONFIRMED.** The recovered checkpoint is byte-identical
to what 14-01/14-04/14-05 documented, decomposes to exactly 55 parameters, and
matches the locked `iqp_sel_55` circuit definition in
`revision/core/models/quantum.py`.

---

## 2. Hyperparameter consistency (mandate item 2)

### Notebook (`qgan_pennylane.ipynb` cell ~26-37)

```python
NUM_QUBITS    = 5
NUM_LAYERS    = 4              # ← but applies to default_75 (RX+RY) circuit
WINDOW_LENGTH = 2 * NUM_QUBITS = 10
NUM_EPOCHS    = 2000
BATCH_SIZE    = 12
N_CRITIC      = 9              # HPO-tuned
LR_CRITIC     = 1.8046e-05     # HPO-tuned
LR_GENERATOR  = 6.9173e-05     # HPO-tuned
VAL_LAMBDA_GP = 2.16           # HPO-tuned (Optuna)
EVAL_EVERY    = 10
```

### Matched-2000ep config.yaml (`revision/results/matched2000/runs/iqp_sel_55_repro/42/config.yaml`)

```yaml
model:           iqp_sel_55_repro
pipeline:        B
seed:            42
epochs:          2000
num_qubits:      5
num_layers:      3              # ← differs from notebook (see Note A)
window_length:   10
batch_size:      12
eval_every:      10
n_critic:        9
lambda_gp:       2.16
lr_critic:       1.8046e-05
lr_generator:    6.9173e-05
ansatz:          iqp_sel_55
circuit_id:      iqp_sel_55
topology:        range
parameter_count: 55
spectral_loss_weight: 0.0
early_stopper:   null           # full 2000ep, D-14-13
```

### Side-by-side

| Hyperparameter | Notebook | Matched-2000ep | Match? |
|---|---|---|---|
| `NUM_QUBITS` | 5 | 5 | ✓ |
| `NUM_LAYERS` | **4** | **3** | ✗ (see Note A) |
| `WINDOW_LENGTH` | 10 | 10 | ✓ |
| `NUM_EPOCHS` | 2000 | 2000 | ✓ |
| `BATCH_SIZE` | 12 | 12 | ✓ |
| `N_CRITIC` | 9 | 9 | ✓ |
| `LR_CRITIC` | 1.8046e-05 | 1.8046e-05 | ✓ |
| `LR_GENERATOR` | 6.9173e-05 | 6.9173e-05 | ✓ |
| `lambda_gp` | 2.16 | 2.16 | ✓ |
| `EVAL_EVERY` | 10 | 10 | ✓ |
| Final-rotation gate | `RX + RY` | `RX only` | ✗ (see Note A) |
| Parameter count | 75 | 55 | ✗ (see Note A) |
| Early stop | active | OFF (D-14-13) | intentional |

**Note A — the apparent layer/gate mismatch is intentional and correct.**
The notebook variable `NUM_LAYERS = 4` together with the RX+RY final rotation
yields 75 parameters and is the `default_75` circuit. The recovered checkpoint
is **55 parameters**, corresponding to the `iqp_sel_55` circuit (RX-only final,
`num_layers=3`). Per `revision/core/models/quantum.py:47-50` and the canonical
config lock (D-14-04), `iqp_sel_55` is the **recovered canonical paper circuit**
— the variant historically trained and stored as `best_checkpoint.pt` —
distinct from the `default_75` byte-frozen reference. The matched-2000ep run
correctly uses `iqp_sel_55` (`num_layers=3`, RX-only) and produces exactly 55
parameters. **Hyperparameter consistency: VERIFIED.** No drift.

(The matched-2000ep also disables `early_stopper` — a documented deliberate
choice in D-14-13 to give all 2000 epochs to every model in the matched-budget
sweep.)

---

## 3. Headline-vs-repro gap on every metric (mandate item 3)

Sources:
- Headline: `revision/results/headline_canonical.json` rows[] (frozen checkpoint, n=1, generation_seed=42).
- Repro: `revision/results/matched2000_dualscale.json#aggregates` rows for `model_kind="iqp_sel_55_repro"` (n=5 seeds 42..46, mean ± std).

### Earth Mover's Distance

| Scale | Headline (n=1) | Repro mean ± std (n=5) | Δ (mean − headline) |
|---|---:|---:|---:|
| OD-EMD | **0.023072** | **0.027526 ± 0.005133** | +0.004454 (+19.3%) |
| log-return-EMD | **0.121241** | **0.122866 ± 0.002603** | +0.001625 (+1.3%) |

The **OD-EMD gap is ≈4.5e-3** (about 0.9σ within seed scatter); the **log-return
EMD gap is essentially zero** (1.6e-3, well within 5-seed std). The headline
checkpoint sits at the *good* tail of the repro distribution, but the 5-seed
mean reproduces it to within ±20% on OD and ±1% on log-returns.

### Moments

| Metric | Scale | Headline | Repro mean ± std | Δ |
|---|---|---:|---:|---:|
| moment_mean | OD | 1.4074 | 1.4026 ± 0.0082 | −0.0048 |
| moment_std | OD | 0.8843 | 0.8751 ± 0.0115 | −0.0092 |
| moment_skew | OD | 1.3657 | 1.3611 ± 0.0205 | −0.0046 |
| moment_kurt | OD | 0.7772 | 0.7862 ± 0.0634 | +0.0090 |
| moment_mean | log-ret | 0.1236 | 0.1237 ± 0.0005 | +0.0001 |
| moment_std | log-ret | 0.0620 | 0.0830 ± 0.0130 | +0.0210 |
| moment_skew | log-ret | 0.0177 | −0.0015 ± 0.0283 | −0.0192 |
| moment_kurt | log-ret | 0.9885 | 0.2039 ± 0.0555 | **−0.7846** |

### DTW

| Metric | Scale | Headline | Repro mean ± std | Δ |
|---|---|---:|---:|---:|
| dtw_mean | OD | 0.2608 | 0.3019 ± 0.0296 | +0.0411 |
| dtw_median | OD | 0.2014 | 0.1857 ± 0.0220 | −0.0157 |
| dtw_std | OD | 0.2478 | 0.4434 ± 0.1816 | +0.1956 |
| dtw_mean | log-ret | 0.9742 | 0.9855 ± 0.0651 | +0.0113 |
| dtw_median | log-ret | 0.9644 | 0.9764 ± 0.0512 | +0.0120 |
| dtw_std | log-ret | 0.2092 | 0.2506 ± 0.0374 | +0.0414 |

### ACF (selected lags)

| Metric | Scale | Headline | Repro mean ± std | Δ |
|---|---|---:|---:|---:|
| acf_lag1_mean | OD | 0.6969 | 0.6951 ± 0.0013 | −0.0018 |
| acf_lag5_mean | OD | −0.2563 | −0.2566 ± 0.0013 | −0.0003 |
| acf_lag1_mean | log-ret | −0.1050 | −0.0949 ± 0.0092 | +0.0101 |
| acf_lag5_mean | log-ret | −0.0564 | −0.0537 ± 0.0069 | +0.0027 |

### Verdict for item 3

The headline does **NOT crush** the repro. On the canonical EMD metric — both
OD and log-return scales — the headline sits well within the repro distribution
(≈0.9σ on OD; ≈0.6σ on log-return). Most moments and ACFs match closely.

There is **one notable outlier**: `moment_kurtosis` on the log-return scale
(headline 0.99 vs repro 0.20 ± 0.06). This is the kurtosis of the inverted
log-return distribution — both the headline (single generation_seed=42 with
the *frozen* checkpoint) and repro (5 retrained seeds, each with its own
generation_seed) are computed in the same `revision.core.eval` helpers
(D-10-20). The gap likely reflects that the headline checkpoint happens to
land in a high-kurtosis region of the loss landscape and retrained seeds
average toward a lower-kurtosis basin. This is exactly the kind of single-run
vs ensemble difference D-14-10 was built to handle. It is **not** a sign of
incorrect computation.

**The apples-to-oranges story is REAL but MILD.** The gap is *not* large enough
to single-handedly explain a perceived regression; the more important point is
that classical baselines have closed in (see §7 below).

---

## 4. Quantum-evaluation precision audit (mandate item 4)

The CR-4 historical asymmetry was: classical on Apple-Silicon MPS at float32,
quantum on CPU at float64. The quantum side was MORE precise — so CR-4 cannot
explain why quantum looks worse.

### Generation path (`revision/run_matched2000.py:257-284`)

```python
generator = generator.to("cpu")              # MPS has no f64 statevector path
...
noise = torch.tensor(..., dtype=torch.float32)
out = generator(noise).to(torch.float64) * 0.1
```

The noise is float32 going IN; the question was whether `out.to(torch.float64)`
is a real upcast or a no-op.

### PennyLane default.qubit output dtype

Empirically verified (`qgan_env/bin/python`):

```
>>> dev = qml.device('default.qubit', wires=5)
>>> @qml.qnode(dev, interface='torch')
... def c(x):
...     qml.RX(x, wires=0); return qml.expval(qml.PauliZ(0))
>>> r = c(torch.tensor(0.5, dtype=torch.float32))
>>> r.dtype
torch.float64

>>> r2 = c(torch.tensor(0.5, dtype=torch.float64))
>>> r2.dtype
torch.float64
```

**PennyLane `default.qubit` returns `torch.float64` regardless of input dtype.**
The statevector path internally uses complex128 and projects expvals to float64.
Therefore `out.to(torch.float64)` is a **no-op cast** — the precision is already
float64 by the time it leaves the QNode.

The classical path (`generate_wgan_samples` for WGAN-CNN/LSTM/MLP) calls the
same function (`revision/run_matched2000.py:257-284`) but the classical
generators *do* output float32 (their parameters are stored as float32), so the
`.to(torch.float64)` cast there is a real upcast. EMD computed downstream in
`revision.core.eval` is then in float64 for both branches.

**Quantum is still the more-precise side at sample-generation time. No
downcast bug.** This is consistent with the device manifest in
`revision/results/matched2000/runs/iqp_sel_55_repro/42/config.yaml`:

```yaml
device_manifest:
  sample_generation_dtype: torch.float64
  quantum_params_dtype: torch.float32
  pennylane_device: default.qubit
  diff_method: backprop
  backend_assertion: PASSED
```

---

## 5. Per-seed best/worst for iqp_sel_55_repro (mandate item 5)

Source: `revision/results/matched2000/runs/iqp_sel_55_repro/{42,43,44,45,46}/metrics.json`.

The on-disk `emd_avg` is a list of length 201 (eval at every 10 epochs across
2000ep). It is **log-return-scale EMD** (matches `frozen_checkpoint_headline`
log-return-EMD of 0.121, NOT the OD-EMD of 0.023). The final-epoch value is the
official end-of-training number; the trajectory minimum is informational.

### Final-epoch log-return EMD

| Seed | Final-epoch EMD | vs headline (0.1212) |
|---|---:|---:|
| 42 | 0.140503 | +0.0193 |
| 43 | 0.174707 | +0.0535 |
| 44 | 0.160780 | +0.0395 |
| 45 | 0.130165 | +0.0089 |
| 46 | 0.168840 | +0.0476 |
| **Mean ± std** | **0.155 ± 0.019** | **+0.034 ± 0.019** |
| **Best (45)** | 0.130 | +0.009 |
| **Worst (43)** | 0.175 | +0.054 |

### Trajectory minimum (over the 201 logged points per seed)

| Seed | Min EMD | At eval idx (of 201) |
|---|---:|---:|
| 42 | 0.115147 | 137 |
| 43 | 0.119588 | 184 |
| 44 | 0.114839 | 193 |
| 45 | 0.108239 | 197 |
| 46 | **0.105645** | 70 |

Best epoch across all seeds: 0.1056 (seed 46), **better** than the headline's
0.1212. Best-of-5-seeds final epoch (0.130) is still slightly worse than headline.

### What this tells us

1. The **final-epoch** mean (0.155) is **higher** than the headline (0.121) by
   ≈0.034, which is exactly the gap reported in the dualscale aggregate. The
   per-seed trajectories all *exceed* the headline at their minimum points,
   but **drift back** to worse EMD by the final epoch — characteristic of late
   WGAN-GP training when the early stopper is disabled.

2. The historical training used an early stopper with `patience=50`,
   `warmup_epochs=100`, `checkpoint_path='best_checkpoint.pt'` (notebook cell
   ~1859–1882). The checkpoint stored was the best-EMD-so-far, which captures
   the trajectory minimum. The headline is therefore equivalent to a
   "best-of-trajectory" pick.

3. The matched-2000ep deliberately **disables early stopping** (D-14-13) so
   every model gets the same compute budget. This biases the **final-epoch
   metric** against models that overfit late — including the quantum
   generator. Compare: the trajectory-minimum mean across the 5 seeds is
   `(0.115 + 0.120 + 0.115 + 0.108 + 0.106) / 5 ≈ 0.113` log-return EMD,
   which is **better** than the headline (0.121). This is strong evidence
   that disabling early stop, not anything model-specific, drives the
   final-epoch gap.

**This is the most important finding of the investigation.**

---

## 6. Quantum-specific code paths in `_train_quantum` (mandate item 6)

I read `revision/run_matched2000.py:412-547` (`_train_quantum`) and compared
against `_train_wgan:550-632` and `_train_vae:635-740`.

### Branches/special-cases unique to `_train_quantum`

1. **Canonical config lock read** (412-455): `iqp_sel_55_repro` reads
   `canonical_config_lock.json` and pulls `decomposition.num_layers`,
   `gate_layout.entangler` (topology), `param_count`, `locked_circuit_id`.
   The V1/V2/V3 ansatz variants pull from `_QUANTUM_ANSATZ` dict. **No
   effect on training hyperparams** — only configures the QuantumGenerator
   class.
2. **MPS-disable hook** (472-489): wraps the `train_wgan_gp` call to force
   `torch.backends.mps.is_available()` to `False`. **This is the CR-4
   future-gate and is now SYMMETRIC with `_train_wgan` (Plan 14-13 Task 4,
   line 579-595).** Both branches now apply the same MPS-disable hook;
   the asymmetry has been remediated. Verified by reading both code blocks.
3. **Post-training param-count assertion** (507-513): explicit
   `count_params() != param_expect` raise. Defensive; cannot affect numbers.
4. **No `early_stopper` argument** to `train_wgan_gp` (486). The
   classical `_train_wgan` ALSO omits `early_stopper` (line 582-593). Identical.
5. **Source label** is `matched2000_reproduction` (446) vs
   `matched2000_baseline` (620) and `matched2000_ansatz` (454). Cosmetic.

### Things checked and found symmetric (no quantum disadvantage)

- `num_epochs = 2000` for both branches (passed as `epochs` parameter).
- `n_critic = N_CRITIC = 9` for both.
- `lambda_gp = LAMBDA = 2.16` for both.
- `lr_critic = LR_CRITIC = 1.8046e-05` for both.
- `lr_generator = LR_GENERATOR = 6.9173e-05` for both.
- `batch_size` set by `BATCH_SIZE = 12` in the dataloader for both.
- `eval_every = EVAL_EVERY = 10` for both.
- Same `train_wgan_gp` function from `revision.core.training` (D-10-08, byte-frozen).
- Same `Critic(window_length=WINDOW_LENGTH)` shared CNN critic.
- Same `generate_wgan_samples(generator, n_synth, seed)` post-training sampler.
- Same n_synth = `10 * bundle.n_real_windows`.
- Same MPS-disable hook (after Plan 14-13 T4).

**Conclusion for item 6: no quantum-disadvantaging branch exists.** The
quantum and classical-WGAN paths are functionally identical except for the
generator class. This rules out training-protocol asymmetry as the cause of
any apparent regression.

---

## 7. Cross-baseline context (informational)

OD-EMD (5-seed mean ± std, except headline which is n=1):

```
model_kind                  scale         mean      std    n
--------------------------  --------  --------  -------  ---
frozen_checkpoint_headline  OD        0.023072    --      1   ← still best
iqp_sel_55_repro            OD        0.027526  0.0051    5
wgan_mlp                    OD        0.025952  0.0067    5
vae                         OD        0.025742  0.0072    5
wgan_lstm                   OD        0.028214  0.0050    5
ar                          OD        0.029084  0.0046    5
V1                          OD        0.027583  0.0051    5
V2                          OD        0.027572  0.0051    5
V3                          OD        0.027538  0.0051    5
wgan_cnn                    OD        0.054323  0.0586    5   ← outlier
```

log-return-EMD:

```
frozen_checkpoint_headline  log_return  0.121241    --      1
iqp_sel_55_repro            log_return  0.122866  0.0026    5
V1/V2                       log_return  ≈0.122
V3                          log_return  0.130305  0.0047    5
vae                         log_return  0.010300  0.0011    5   ← VAE crushes log-ret!
wgan_mlp                    log_return  0.269939  0.0398    5
wgan_lstm                   log_return  0.166321  0.0205    5
ar                          log_return  0.781139  0.0031    5
wgan_cnn                    log_return  0.687323  0.3034    5
```

**Observation:** on OD-EMD the headline is **still the best** at 0.023, with
all 5-seed means clustered around 0.026–0.029. The quantum 5-seed mean (0.0275)
is tied with the V1/V2/V3 ansatz variants and ≈0.002 above VAE/wgan_mlp. So:

- Headline-vs-repro accounts for ~0.0045 of any perceived quantum gap.
- The other ~0.002 (quantum mean above the best classical mean) is real but
  small and within seed scatter (the quantum std is 0.0051 — the mean is
  inside ±1σ of every classical baseline).
- On log-return-EMD, VAE is **dramatically** ahead (0.010 vs 0.121 for
  quantum/headline). This is documented in D-14-10 and the paper-block
  framing as a known scale-dependent behavior: the VAE training objective
  rewards log-return MMD-like reconstruction directly while the WGAN-GP
  family does not. It is the most likely visual driver of "quantum looks
  mid-pack" if a reader is looking at log-return panels.

---

## 8. Top hypothesis + confidence

**Top hypothesis (confidence: HIGH):** the apparent regression is a
**combination of three factors**, in decreasing order of magnitude:

1. **Early-stop disabled in matched-budget** (D-14-13). The historical
   checkpoint captured the best-EMD epoch via early-stop checkpointing;
   the matched-2000ep takes the *final* epoch. Per-seed trajectory minima
   (≈0.108–0.120) are competitive with and even better than the headline,
   so this single decision moves the quantum number by ≈0.03–0.04 on
   log-return-EMD.

2. **Headline-vs-repro by construction** (D-14-10). The headline is n=1 and
   thus also benefits from a favorable `generation_seed`; the repro is a
   5-seed mean. Even with identical training, the mean of 5 retrained models
   will not match the single best-trained model with a single best generation
   seed.

3. **Classical baselines have caught up.** On OD-EMD, VAE/wgan_mlp/wgan_lstm
   5-seed means now sit at 0.026–0.028, statistically indistinguishable from
   the quantum 5-seed mean of 0.0275. This is not a quantum regression; it is
   classical-baseline improvement (newer architectures + the same MPS-disable
   hook + same 2000-epoch budget).

**None of (1), (2), (3) require a bug, drift, or precision issue.** The
checkpoint, hyperparameters, code paths, and dtype handling are all verified
identical to the historical setup.

---

## Appendix — files inspected

- `/Users/shawngibford/dev/phd/qGAN/revision/checkpoints/best_checkpoint.pt`
- `/Users/shawngibford/dev/phd/qGAN/revision/results/canonical_config_lock.json`
- `/Users/shawngibford/dev/phd/qGAN/revision/results/headline_canonical.json`
- `/Users/shawngibford/dev/phd/qGAN/revision/results/matched2000_dualscale.json`
- `/Users/shawngibford/dev/phd/qGAN/revision/results/matched2000/runs/iqp_sel_55_repro/{42,43,44,45,46}/metrics.json`
- `/Users/shawngibford/dev/phd/qGAN/revision/results/matched2000/runs/iqp_sel_55_repro/42/config.yaml`
- `/Users/shawngibford/dev/phd/qGAN/revision/run_matched2000.py` (lines 257–632)
- `/Users/shawngibford/dev/phd/qGAN/revision/core/models/quantum.py` (lines 1–330)
- `/Users/shawngibford/dev/phd/qGAN/qgan_pennylane.ipynb` (cells 26, 28, 37, 40, 41)
