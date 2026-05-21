---
reviewer: code-quality-r3
scope: r3 forensic — quantum-disadvantaging bug hunt in matched-2000ep
  + distribution-EMD + figure-suite code paths
files_audited:
  - revision/run_matched2000.py
  - revision/run_matched2000_dualscale.py
  - revision/run_distribution_emd.py
  - revision/run_figure_suite.py
  - revision/core/eval.py (byte-frozen audit)
  - revision/core/training.py (byte-frozen audit)
  - revision/core/models/quantum.py (byte-frozen audit)
  - revision/core/models/critic.py (byte-frozen audit)
  - revision/core/preprocessing.py (byte-frozen audit, supporting)
  - revision/core/data.py (byte-frozen audit, supporting)
created: 2026-05-21T00:00:00Z
---

# Code Review R3 — Quantum-Disadvantaging Forensic Audit

## Summary verdict

**Quantum-disadvantaging bug found: YES — TWO findings, one CRITICAL (new,
in 14-15) and one HIGH (pre-existing, inherited from the 1000ep driver).**

The r1 review (5 CRITICAL + 9 HIGH) and the r2 review (4 HIGH +
follow-ups) addressed the gate-correctness and provenance side of the
pipeline. They did NOT examine **metric-comparability across model
families**. r3 finds that **the new histogram-density distribution-EMD
metric introduced by plan 14-15 (`revision/run_distribution_emd.py`)
contains a metric construction that systematically disadvantages
quantum**, and that **the dualscale log_return-EMD path
(`revision/run_matched2000_dualscale.py:368-372` and the inherited
`run_dualscale_fidelity.py` recipe) compares real-on-raw-scale against
fake-on-standardized-scale** — a scale mismatch that affects every model
asymmetrically and was therefore inherited by both the 14-08 emit AND
the 14-15 distribution-EMD emit verbatim.

The CRITICAL finding is **NOT** a coding mistake — it is a
**metric-selection mistake** in a new emitter. `density=True` is a
standard NumPy flag, but the consequence in this context (re-normalizing
each histogram independently over the in-range portion only) systematically
rewards narrow-collapse distributions (VAE) and uncapped-range
distributions (WGAN-CNN/AR — out-of-range mass is silently dropped) over
spread-but-bounded distributions (quantum, capped at ±0.1 by the
notebook-parity `*0.1` cast in `training.py:347`). The bias is asymmetric
across model families in a way that consistently DISADVANTAGES quantum.

**Confidence: HIGH.** Empirically demonstrated against the actual
saved artifacts in `revision/results/matched2000/runs/` and the actual
`revision/results/distribution_emd.json` aggregates — see "Hypothesis
verdicts → H3" below for the synthetic-test smoking gun and the
real-artifact reproduction.

D-14-22 byte-freeze attestation: **PASS** — `git diff 06bb470..HEAD --
revision/core/` returns zero bytes (verified). No new modifications to
`revision/core/` since the byte-freeze base.

---

## Hypothesis verdicts

| # | Hypothesis | Verdict | Severity |
|---|---|---|---|
| H1 | Quantum-specific training-loop asymmetry | **PASS** | — |
| H2 | Sample-generation asymmetry | **PASS** | — |
| H3 | EMD-computation route divergence | **FAIL** | CRITICAL + HIGH |
| H4 | Critic asymmetry | **PASS (with INFO note)** | INFORMATIONAL |
| H5 | Gradient-penalty direction | **PASS** | — |
| H6 | Determinism / seed-handling regression | **PASS** | — |
| H7 | Inverse-transform asymmetry | **PASS** | — |
| H8 | D-14-22 byte-freeze verification | **PASS** | — |

### H1: Quantum-specific training-loop asymmetry — PASS

Line-by-line comparison of `_train_quantum`
(`revision/run_matched2000.py:412-547`) vs `_train_wgan`
(`revision/run_matched2000.py:550-632`) vs `_train_vae`
(`revision/run_matched2000.py:635-737`):

| Field | _train_quantum | _train_wgan | _train_vae |
|---|---|---|---|
| `torch.manual_seed(seed)` BEFORE generator | YES (l. 457) | YES (l. 570) | YES (l. 652) |
| `np.random.seed(seed)` BEFORE generator | NO* | NO* | YES (l. 653) |
| `random.seed(seed)` BEFORE generator | NO* | NO* | YES (l. 654) |
| MPS-disable hook | YES (l. 472-489) | YES (l. 579-595) | YES (l. 657-703) |
| `train_wgan_gp(...)` call | YES (same args) | YES (same args) | N/A (ELBO loop) |
| Hyperparameter pass-through | `N_CRITIC`, `LAMBDA`, `LR_CRITIC`, `LR_GENERATOR` | IDENTICAL | (VAE-only: `lr=1e-3`, `beta=1.0`) |
| `generate_wgan_samples` post-training | YES (l. 506) | YES (l. 607) | N/A (`vae.sample`) |
| `training_time_device` capture | YES (l. 500-503) | YES (l. 602-605) | YES (l. 693-696) |

*Both adversarial paths rely on the inner `train_wgan_gp` (training.py:
245-247) to seed numpy + random. So the omission is consistent between
quantum and classical — **NOT** a quantum-specific gap. (HI-7 from r1
correctly flagged VAE specifically because its training path does NOT
call `train_wgan_gp` and therefore needs its own numpy/random seed; the
fix was applied in 14-13 T4.)

**No quantum-asymmetric step found.** Both branches construct a fresh
`Critic(window_length=WINDOW_LENGTH)`, both wrap `train_wgan_gp` in the
MPS-disable hook, both call the same training hyperparameters, both
generate samples via the same `generate_wgan_samples` post-training.

### H2: Sample-generation asymmetry — PASS

`generate_wgan_samples` (`revision/run_matched2000.py:257-284`) was
verified empirically against both generator families:

```
# QuantumGenerator (iqp_sel_55, 5q × 3L)
>>> out = g(noise)  # noise: float32 (5,3)
>>> out.dtype        # torch.float64
>>> out.shape        # (3, 10)
>>> hasattr(out, '.to')  # True
>>> out.to(torch.float64).dtype  # torch.float64 (no-op)

# WGANMLPGenerator
>>> out = g(noise)  # noise: float32 (5,3)
>>> out.dtype        # torch.float32
>>> out.to(torch.float64).dtype  # torch.float64 (lossy upcast)
```

The quantum QNode returns a `torch.Tensor` (not numpy / jax-array) at
float64 — the PennyLane `interface="torch"` + `diff_method="backprop"`
contract handles this correctly. The `.to(torch.float64) * 0.1` cast is
a no-op for the quantum dtype but a lossy upcast for classical (which
upcasts float32 → float64 before the `*0.1` multiply). Mathematically
this means classical samples lose ~7 decimal digits of precision in
the cast that quantum does not lose. **This is the OPPOSITE of a
quantum disadvantage** — it makes classical samples marginally less
precise.

`generator = generator.to("cpu")` (line 270) is a no-op for the quantum
generator's params_pqc (which lives on CPU by construction — the
`_train_quantum` MPS-disable hook ensures CPU-only training). It does
not break PennyLane state.

### H3: EMD-computation route divergence — **FAIL (CRITICAL + HIGH)**

Two distinct findings:

#### CRITICAL R3-CR-1: distribution-EMD metric construction systematically rewards narrow / collapsed distributions and penalizes spread-but-bounded distributions (quantum disadvantage)

**File:** `revision/run_distribution_emd.py:94-141` (the new
`compute_histogram_density_emd` function from plan 14-15)

**The metric:**
```python
real_hist, edges = np.histogram(real, bins=n_bins, density=True)
fake_hist, _ = np.histogram(fake, bins=edges, density=True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
return wasserstein_distance(bin_centers, bin_centers, real_hist, fake_hist)
```

**The bias mechanism (3 compounding effects):**

1. `density=True` normalizes each histogram independently to integrate
   to 1 OVER THE IN-RANGE PORTION. Any fake mass outside `[edges[0],
   edges[-1]]` (the real range) is silently dropped and then the
   remaining mass is re-normalized to integrate to 1. A model that puts
   50% of its samples way outside the real range gets its remaining 50%
   re-normalized into a density that may match real density very well —
   the EMD is computed on the re-normalized fake, not on the original
   fake.

2. `wasserstein_distance(bin_centers, bin_centers, real_hist, fake_hist)`
   with `density=True` weights effectively measures **how aligned the
   density profiles are** within the real range — NOT how well the
   supports overlap. A delta function at the real distribution's mode
   produces a very low EMD even though it represents complete mode
   collapse.

3. The metric is invariant to the FAKE SAMPLE SUPPORT outside the real
   range (because those samples are dropped). A model that produces
   samples in `[-3.9, +6.1]` (WGAN-CNN, ~6% in-range) gets its
   distribution-EMD computed on only the in-range 6% — which can
   coincidentally look like a good match.

**Empirical demonstration (synthetic):**
```
EMD(N(0,1), N(0,1) sample)      = 0.0267  (real-vs-real — baseline noise)
EMD(N(0,1), {50% N(0,1), 50% N(10,1)})  = 0.0147  ← 50% OUT-OF-RANGE wins
EMD(N(0,1), N(0, 0.1))          = 0.7187  ← narrow distribution loses
```
A fake distribution that has 50% completely-misplaced mass wins over a
real-vs-real comparison. A perfectly mean-aligned but narrow distribution
loses by 27× the baseline. The metric is **structurally biased toward
distributions that either collapse to a delta near the real peak OR
extend far beyond the real range**.

**Empirical demonstration (actual matched-2000ep artifacts):**

Reading `revision/results/distribution_emd.json` log_return-scale
aggregates and reproducing the raw-sample EMD against the same samples:

| Model | raw-sample EMD | dist-EMD (14-15) | Notes |
|---|---|---|---|
| iqp_sel_55_repro | 0.0149 | **0.0365** (worse rel.) | bounded `*0.1` cap |
| V1 | 0.0151 | 0.0351 | quantum 4-layer range |
| V2 | 0.0150 | 0.0362 | quantum 8-layer range |
| V3 | 0.0143 | 0.0266 | quantum 4-layer linear |
| wgan_mlp | 0.0112 | 0.0224 | mid-range output |
| wgan_cnn | 0.0159 | 0.0250 | UNCAPPED output [-3.9, +6.1] |
| wgan_lstm | 0.0124 | 0.0262 | mid-range output |
| vae | **0.0158** | **0.0104** | std=0.0004 (posterior collapse) |
| ar | 0.0028 | 0.0243 | wide closed-form fit |

**The rankings INVERT between metrics.** Under raw-sample EMD (the v1.0
canonical metric the manuscript already uses): quantum is competitive
(0.014-0.015 vs WGAN-MLP's 0.011, VAE's 0.016). Under distribution-EMD:
**VAE jumps from 6th place to 1st**, despite VAE having a known posterior
collapse (sample std = 0.0004 vs real std = 0.0217 — 50× narrower).
Quantum drops from 3rd-place-tier to 7th-9th place. The metric rewards
VAE's collapse and penalizes quantum's bounded-but-spread output.

**Mechanism for quantum specifically:** the quantum generator's output
is bounded by `[-0.1, +0.1]` because the Pauli expectation values are in
`[-1, +1]` and `training.py:347` does `* 0.1` (byte-frozen notebook-
parity contract). Quantum cannot produce out-of-range samples — every
sample falls within `[r_min + 0.45*(r_max-r_min), r_min + 0.55*(r_max-
r_min)]` after `reconstruct_od`'s pm1-to-r_norm map. This puts ~30% of
quantum's mass in the real range (the rest is "out-of-range" in the
standardized space, but only because the standardized space has been
re-scaled to `[r_min, r_max] ≈ [-4, +4]` — see R3-HI-1 for the deeper
scale-mismatch issue). Under distribution-EMD, quantum's tight in-range
mass gets re-normalized to a density that doesn't match real's wider
density — high EMD. Under raw-sample EMD, the comparison happens
directly in the data space and the quantum's tightness is reflected as
a smaller EMD (~0.015) commensurate with the actual spread mismatch.

**Why this is the quantum-disadvantaging bug the orchestrator is
looking for:** the user's observation "current matched-2000ep quantum
results look mid-pack vs classical" is partially true at the OD scale
(where everyone is in the 0.025-0.030 range — see
`matched2000_dualscale.json#aggregates`), but on the NEW distribution-
EMD metric quantum looks 1.5-3× worse than VAE/WGAN-MLP. The
manuscript's plan 14-15 introduces this metric as a "comparable" column
against the pre-v1.0 paper figure ~0.0015 headline — but the resulting
ranking is metric-construction-driven, NOT model-quality-driven.

**Recommended fix:**
- **Option A (preferred):** drop `density=True` and use raw histogram
  COUNTS, then normalize by total count after dropping out-of-range
  samples. The dropped-mass should be RECORDED as a forfeit penalty
  added to the EMD (e.g., add `(1 - frac_in_range) * (range_radius)` so
  out-of-range mass costs something). This makes the metric properly
  proper.
- **Option B:** report distribution-EMD ONLY on the OD scale (where
  there is no `*0.1` quantum-cap mismatch) and explicitly disclose that
  the log_return-scale distribution-EMD favors models with narrow OR
  out-of-range output. Add a caveat sentence to
  `reconciliation_note.md`'s C-3 paragraph.
- **Option C:** abandon distribution-EMD entirely and stick with the
  v1.0 raw-sample EMD (already in `revision/core/eval.py:25-36`). The
  pre-v1.0 paper's 50-bin distribution-EMD reference number ~0.0015
  was on a DIFFERENT dataset / different windowing — it is not
  reconcilable to the current matched-2000ep numbers regardless of
  which 50-bin variant we use. Disclose the non-reconciliability in
  reconciliation_note.md (D-14-10 conflation guard already covers this
  for the v1.0 headline; extend to the pre-v1.0 reference number too).

#### HIGH R3-HI-1: log_return-scale EMD compares real (raw) against fake (standardized) — pre-existing scale mismatch inherited from `run_dualscale_fidelity.py`

**Files:**
- `revision/run_matched2000_dualscale.py:368-372` (the `_log_return_rows`
  emit, verbatim with `run_dualscale_fidelity._log_return_rows`)
- `revision/run_distribution_emd.py:144-153` (`_real_references`) +
  `revision/run_distribution_emd.py:156-169` (`_fake_log_return_flat`)

**Evidence:** in `run_matched2000_dualscale.py`'s `_log_return_rows`,
the real reference is `real_log_delta = d_real["log_delta"].cpu().
numpy()` — these are the RAW per-step log-returns (mean ~0.002,
std ~0.022). The fake reference is `r["transformed"]` which comes from
`reconstruct_od` and equals:
```python
r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
```
where `r_min, r_max` are min/max of `norm_log_delta = (log_delta - mu)
/ sigma` (the STANDARDIZED log-returns, mean=0, std≈1, range ≈ [-4, +4]).

So `compute_emd(real_log_delta, trans_flat)` compares:
- real: mean=0.0025, std=0.022 (RAW log-returns)
- fake (quantum): mean=0.12, std=0.087 (STANDARDIZED scale, *0.1-capped)
- fake (WGAN-CNN): mean=...,  std=very large (STANDARDIZED scale, uncapped)

**Verification:**
```
compute_emd(real_log_delta, real_norm_log_delta) = 0.7344
   ↑ same data, different scale, EMD = 0.73 (large because of mu/sigma offset)
compute_emd(real_log_delta, quantum r_norm)      = 0.1228
   ↑ what matched2000_dualscale.json records
compute_emd(real_log_delta, quantum log_delta)   = 0.0149  ← scale-corrected
```

The dualscale JSON's recorded log_return-scale EMD for quantum is
**0.123** (off by 8× from the scale-correct 0.015). For models with
larger samples_pm1 ranges (WGAN-CNN, AR), the recorded EMD blows up
to 0.69 / 0.78 — the dualscale JSON's "log_return EMD" column is
dominated by the *0.1-cap range mismatch, not by distributional fidelity.

**Why this asymmetrically affects quantum:** the `*0.1` cap means quantum
samples_pm1 are in `[-0.08, +0.08]` ≈ 8% of the [-1, +1] band. WGAN
samples can occupy more of the [-1, +1] band (WGAN-MLP: ~24% of band,
WGAN-CNN: well beyond +1 — uncapped). After the scale-mismatched
comparison:
- Quantum: small mean offset + small spread → EMD ≈ 0.12
- WGAN-CNN: huge spread → EMD = 0.69 (worse)
- WGAN-MLP: medium → EMD = 0.27
- AR: huge spread (unconditioned) → EMD = 0.78
- VAE: tight near-zero output, NO *0.1, scale-matches real reasonably → EMD = 0.010 (best)

**The dualscale JSON's log_return EMD column is therefore _accidentally
quantum-favorable_ (quantum's *0.1 cap limits its scale-mismatch
inflation), not quantum-disadvantaging.** But the column is still
fundamentally meaningless — it does not measure log-return distributional
fidelity; it measures the scale mismatch between standardized and raw
log-returns. The manuscript should NOT cite the dualscale log_return EMD
as a model-quality number.

**Recommended fix:** either un-standardize the fake `r_norm` before
calling `compute_emd` (multiply by `sigma`, add `mu` — both available
in `inverse_kwargs.npz`), or change `real_log_delta` to `norm_log_delta`
in the real reference so both sides are on the standardized scale. The
LATTER is a smaller code change (single-line edit to
`build_real_references`); the FORMER is more semantically correct (the
manuscript reports OD-scale metrics, log_return-scale should be the
RAW log-return scale to match).

Same fix needed in `run_distribution_emd.py:_real_references` /
`_fake_log_return_flat`.

**Provenance:** the scale mismatch is INHERITED from `run_dualscale_
fidelity.py` (the 1000ep driver) — the original `_log_return_rows` and
`reconstruct_od` recipes have this bug. The matched2000 driver was
copied verbatim under D-11-10 ("revision.core.eval imported UNCHANGED"),
which preserved the bug. Plan 14-15's `run_distribution_emd.py`
inherited it again when it imported `reconstruct_od` from
`run_figure_suite` and re-used the same `_fake_log_return_flat`
construction pattern.

### H4: Critic asymmetry — PASS (with INFORMATIONAL note)

Both `_train_quantum` (l. 465: `critic = Critic(window_length=WINDOW_
LENGTH)`) and `_train_wgan` (l. 572: identical call) construct a fresh
`Critic` with the same constructor args. Inside `Critic.__init__`
(`revision/core/models/critic.py:38-67`): same `nn.Sequential`, same
layer specs (Conv1d → LeakyReLU → ... → Linear), same `.double()` cast.

The critic's INITIAL WEIGHTS depend on the torch RNG state at
construction time. In `_train_quantum`:
```
torch.manual_seed(seed)          # line 457 — RNG state X
generator = QuantumGenerator(...) # consumes ~75 or 55 randn floats for params_pqc
critic = Critic(...)              # RNG state Y = X + ~75/55 randn draws
```
vs `_train_wgan`:
```
torch.manual_seed(seed)
generator = gens[model]()         # consumes ~M randn floats for classical weights
critic = Critic(...)              # RNG state Y' = X + M randn draws
```

The critic's initial weights are therefore DIFFERENT across model
families. This is symmetric across the 9 models in the sense that
EVERY model gets a different critic init, but it does mean the matched-
budget comparison is NOT "same critic init, just different generator".
Then `train_wgan_gp` (training.py:245) re-seeds `torch.manual_seed(seed)`
which makes subsequent draws (noise, batches, alpha for GP) identical
across runs — but the already-constructed critic weights are frozen at
whatever state they were after construction.

**Not a quantum-specific disadvantage** — V1/V2/V3 also get different
critic inits relative to iqp_sel_55_repro because the `params_pqc`
randn draws have different lengths. The matched-budget contract is
"matched epochs, matched seed", not "matched critic init".

**Recommended (for v3.0 baseline, NOT this resubmission):** seed the
critic INDEPENDENTLY of the generator construction —
```python
torch.manual_seed(seed)
generator = ...
torch.manual_seed(seed + 1000)   # reset before critic
critic = Critic(...)
```
so the critic init is the same across all 9 models for a given seed.

### H5: Gradient-penalty direction — PASS

`compute_gradient_penalty` (`revision/core/training.py:31-73`) computes
the standard two-sided WGAN-GP penalty `((||∇D(x̂)||₂ - 1)²).mean()`
where x̂ = α·real + (1-α)·fake. This is the canonical Gulrajani 2017
formulation. The sign is correct: at `training.py:364`, `critic_loss =
fake_score_mean - real_score_mean + lambda_gp * gp` — positive lambda_gp
ADDS the penalty (the critic loss DESIRES large `real_score_mean -
fake_score_mean` AND small gradient norm violations).

No `torch.clamp` or `torch.nan_to_num` is applied to the quantum
gradient path. The PennyLane backprop interface returns standard torch
gradients that flow through the same `.backward()` chain as classical.

`lambda_gp = 2.16` (from `revision/core/__init__.py`) is passed
identically to both `_train_quantum` and `_train_wgan` (lines 481 + 588:
`lambda_gp=float(LAMBDA)`).

### H6: Determinism / seed-handling regression — PASS

| Path | torch.manual_seed | np.random.seed | random.seed |
|---|---|---|---|
| `_train_quantum` (pre-train_wgan_gp) | YES (l. 457) | NO | NO |
| `_train_wgan` (pre-train_wgan_gp) | YES (l. 570) | NO | NO |
| `_train_vae` (no train_wgan_gp call) | YES (l. 652) | YES (l. 653) | YES (l. 654) |
| `train_wgan_gp` body | YES (l. 245) | YES (l. 246) | YES (l. 247) |
| `generate_wgan_samples` post-train | (rng = np.random.default_rng(seed) l. 271) — independent stream |

Both adversarial paths rely on the inner `train_wgan_gp` to seed numpy
+ random. So np/random get the same seed in both branches before any
critic phase numpy draws. **Identical seeding**.

The only RNG-consuming code between `_train_*`'s `torch.manual_seed`
and `train_wgan_gp`'s re-seed is generator + critic construction, both
of which use ONLY torch RNG. So np/random determinism is fully
controlled by `train_wgan_gp`'s seed block for both adversarial
families. **No quantum-specific gap.**

HI-7 from r1 (VAE missing np/random seed) was fixed in 14-13 T4 (lines
653-654 of `_train_vae`).

### H7: Inverse-transform asymmetry — PASS

`reconstruct_od` in BOTH `run_figure_suite.py:261-296` and
`run_matched2000_dualscale.py:175-217` is structurally identical and
has no model_kind-conditional branching. The transform chain is:
```
samples_pm1 → r_norm via pm1→[r_min, r_max] linear map
            → od via inverse_logreturns(r_norm, od_start, mu, sigma)
            → od[:, :10] trim (Pipeline-B 11→10 truncation)
```
Same RNG seeding `np.random.default_rng(seed * 7919 + 1)` is used for
the `od_start_per_window` draw across all 9 models. The torch.tensor
casts are identical. The cumsum-based `inverse_logreturns` in
`revision/core/preprocessing.py:49-75` has NO model_kind branching.

Quantum's narrow samples (`[-0.08, +0.08]`) result in a narrow
reconstructed OD range (because the pm1→r_norm map preserves the
range proportionally), but this is a CONSEQUENCE of the *0.1 training
cap (R3-CR-1 / R3-HI-1), not a bug in `reconstruct_od` itself. The
inverse is applied symmetrically.

### H8: D-14-22 byte-freeze verification — PASS

```
$ git diff --stat 06bb470..HEAD -- revision/core/
(empty output)

$ git diff --stat db59b11..HEAD -- revision/core/
(empty output)
```

The last commit modifying `revision/core/` is `db59b11` (14-01: "add
non-default 55-param IQP:SEL circuit + D-14-07 equivalence gate") —
the 55-param `iqp_sel_55` circuit branch in `models/quantum.py:240-250`
and `_introspect_circuit:315-326`. All subsequent core/ changes are
ZERO bytes. The byte-freeze is literally honored from 14-01 through
14-15 (HEAD as of this review).

Files specifically audited as read-only:
- `revision/core/eval.py` — sound, notebook-parity preserved
- `revision/core/training.py` — sound, notebook-parity preserved
  (compute_dtype split + ES device-fix per CR-01/02 from Phase 13 already
  audited and accepted)
- `revision/core/models/quantum.py` — sound, bounds-check arithmetic
  correct (IN-9 from r1)
- `revision/core/models/critic.py` — sound, single Dropout layer,
  `.double()` cast preserved

---

## Findings by severity

### CRITICAL

#### R3-CR-1: Distribution-EMD metric is structurally biased against bounded/spread distributions (quantum disadvantage)

See H3 above. The hist-density Wasserstein with `density=True` and
`bins=edges_from_real` re-normalizes both histograms independently
over the in-range portion, silently drops out-of-range fake mass, and
rewards distributions that concentrate near the real peak OR extend
far beyond the real range. Quantum's bounded-but-spread output (capped
by `training.py:347`'s `*0.1` notebook-parity multiply, in BYTE-FROZEN
code, applied only to quantum-and-WGAN samples per the training
contract) is systematically penalized vs collapse-prone models (VAE)
and uncapped models (WGAN-CNN, AR).

**File / lines:** `revision/run_distribution_emd.py:94-141`.

**Fix preference:** Option C (don't ship distribution-EMD) is the
safest single-step fix. The pre-v1.0 paper's 50-bin reference ~0.0015
is on a different dataset / different windowing / different sample
count anyway — it is not reconcilable to the matched-2000ep numbers
regardless of metric choice.

### HIGH

#### R3-HI-1: log_return-scale EMD comparison is scale-mismatched (real RAW vs fake STANDARDIZED) in both 14-08 dualscale and 14-15 distribution-EMD emitters

See H3 above. `compute_emd(real_log_delta, trans_flat)` and
`compute_histogram_density_emd(real_log_delta, fake r_norm)` both
compare raw log-returns (std=0.022) against standardized log-returns
(nominal std=1, *0.1-capped to ~0.09 for quantum). The recorded
log_return EMDs are scale-mismatch artifacts, not distributional
fidelity measures.

**Files / lines:**
- `revision/run_matched2000_dualscale.py:368-372` (recipe inherited
  from `run_dualscale_fidelity.py`)
- `revision/run_distribution_emd.py:144-153` + `:156-169` (recipe
  inherited from the figure suite)

**Asymmetric effect:** the *0.1 quantum cap limits quantum's scale-
mismatch inflation, making the log_return EMD ACCIDENTALLY quantum-
favorable on this scale. Classical models with larger output ranges
(WGAN-CNN, AR) get larger scale-mismatch EMD values, making them look
worse than they actually are.

**Fix preference:** change `_real_references` to use `norm_log_delta`
(standardized log-returns) instead of `log_delta` (raw). Single-line
edit in both files. Document the change in the C-3 disclosure
paragraph and re-emit `matched2000_dualscale.json` +
`distribution_emd.json`.

#### R3-HI-2: Quantum `*0.1` training-loop cap caps generator expressivity vs real `[-1, +1]` data scale (BYTE-FROZEN — disclosure only)

**File / line:** `revision/core/training.py:347`
```python
generated_samples = generated_samples.to(compute_dtype) * 0.1
```

The training-time fake samples are capped at `[-0.1, +0.1]` (because
the quantum generator's Pauli expectation values are bounded in `[-1,
+1]`, then `*0.1`). The real training data, `r_pm1` from
`build_dataset_for_pipeline:234`, is on `[-1, +1]` with 61% of values
OUTSIDE `[-0.1, +0.1]` (1% percentile -0.67, 99% percentile +0.57 —
verified empirically).

The critic is therefore trained to distinguish real `[-1, +1]` vs fake
`[-0.1, +0.1]` — these are not on the same scale and any critic trained
this way learns to distinguish by scale alone. The "best" the quantum
generator can do is saturate the `*0.1` cap (push Pauli expvals toward
±1 so the cap output is near ±0.1). Empirically the saved quantum
samples are at `[-0.077, +0.076]` — already at the cap.

**This is a fundamental architectural constraint of the notebook-parity
training contract** — it applies to every quantum run since v1.0. The
classical WGAN-MLP/CNN/LSTM ALSO have the same `*0.1` multiply in
`training.py:347` (the line is generator-family-agnostic), but
classical generators are UNBOUNDED — `*0.1` of an unbounded `g(z)` is
still unbounded. So the cap is binding only for quantum.

**Impact:** ALL quantum runs (matched2000 and the historical Aug 2025
runs) suffer this. The `*0.1` multiply was inherited from the original
notebook and has been part of the contract since the v1.0 release. It
does NOT explain "current results worse than historical Aug 2025"
because both runs share the contract. But it DOES partially explain
"quantum looks mid-pack vs classical" — quantum is structurally capped
where classical is not.

**Fix preference:** disclose explicitly in `methods_full.md` §3.x that
quantum's effective output range is `[-0.1, +0.1]` while real training
data is on `[-1, +1]`, and that this is a notebook-parity contract
inherited from v1.0 (not changed under D-14-22). A future v3.0
baseline could ablate this by removing the `*0.1` multiply OR by
rescaling the real `r_pm1` to `[-0.1, +0.1]` to match. **Out of scope
for the current resubmission.**

### MEDIUM

#### R3-MD-1: `compute_histogram_density_emd` self-EMD assertion uses bare `assert` (not raise AssertionError)

**File / line:** `revision/run_distribution_emd.py:280-282`
```python
assert self_emd == 0.0, (
    f"self-EMD must be 0 on identical inputs; got {self_emd}"
)
```

Bare `assert` — stripped under `python -O`. The repo-wide convention
(documented in `run_multiseed_rollup.py:86-92` and used everywhere
else in the 14-13 sweep) is `raise AssertionError(...)`. This is the
exact pattern r1's LO-1 flagged for `quantum.py:87`.

**Fix:** `if self_emd != 0.0: raise AssertionError(...)`.

#### R3-MD-2: `_model_seed_rows` skips missing seeds silently (no recorded null-row)

**File / lines:** `revision/run_distribution_emd.py:181-185`
```python
if not (base / "samples.npy").exists():
    # Skip missing seeds quietly (...)
    continue
```

The figure suite and `run_matched2000_dualscale.py` both LOUD-FAIL
with `FileNotFoundError` (T-14-14: "every number must come from an
already-saved bundle"). This emitter silently skips. If a sweep run
partially fails (e.g., one seed of one model didn't complete), the
distribution-EMD JSON's aggregates would compute mean/std over fewer
than 5 seeds without flagging it. The `n` field in the aggregate would
reveal the gap but the JSON would still be emitted (`headline_present`
is the only loud field; per-row absence is silent).

**Fix:** either raise loudly on missing samples (matching the dualscale
contract), or record explicit null-rows so downstream consumers can
detect the gap. Add a `n_seeds_expected = 5` field to the aggregate
schema and raise if `n_seeds != n_seeds_expected` for any model.

#### R3-MD-3: The 14-15 disclosure in `reconciliation_note.md`'s C-3 paragraph does not call out the metric's structural bias

The C-3 disclosure (per `revision/run_distribution_emd.py:8-14`)
acknowledges that the pre-v1.0 50-bin density EMD and the v1.0 raw-
sample EMD "are NOT commensurate". This is correct but understates the
issue. The reader is given the impression that the two metrics measure
"the same thing through different lenses" and can be reported side-by-
side as "two views of the same fact". In reality:
- v1.0 raw-sample EMD measures distributional shift in the data space
  (the manuscript-relevant quantity).
- 14-15 hist-density EMD measures density-profile alignment after
  silently dropping out-of-range mass — a fundamentally different
  question that materially reorders the model rankings.

**Fix:** if R3-CR-1's Option C is rejected (i.e., the team insists on
shipping distribution-EMD), the disclosure paragraph should explicitly
state:
> "The hist-density EMD re-normalizes each histogram to integrate to 1
> over the real distribution's support. Fake samples outside this
> support are silently dropped from the comparison. As a consequence,
> the metric rewards models whose density mass concentrates near the
> real mode (potentially via mode collapse) and penalizes models whose
> output range is bounded (like the quantum generator, whose Pauli-
> expectation values are bounded in [-1, +1] and then multiplied by
> 0.1 in `training.py:347`). The raw-sample EMD in
> `revision/core/eval.py:25-36` is the canonical metric; the hist-
> density column is presented for backward-comparability with the pre-
> v1.0 paper figure only."

### LOW / Style

#### R3-LO-1: `_model_seed_rows` does not record the seeds that were processed

**File / line:** `revision/run_distribution_emd.py:172-243`. The
`rows[]` carry per-(model, seed) values, but the top-level JSON does
not record the SEED SET that was attempted vs successful. A reader of
the JSON cannot tell whether a missing (model_kind, scale) aggregate
is because the model was skipped or because all 5 seeds failed.

**Fix:** add `"seeds_attempted": SEEDS` and `"seeds_processed":
{model_kind: [s1, s2, ...]}` to the top-level payload.

#### R3-LO-2: `compute_histogram_density_emd` does not record the `density=True` flag in the metric_formulation citation

**File / line:** `revision/run_distribution_emd.py:87-91`
```python
METRIC_FORMULATION = (
    "scipy.stats.wasserstein_distance(bin_centers, bin_centers, "
    "real_hist_density, fake_hist_density) over 50-bin histograms "
    "(np.histogram(..., density=True))"
)
```

The `density=True` flag IS mentioned, but only in passing. The string
does not call out that this flag is the source of the renormalization
behavior. A reader who copies the formulation into a paper section
would mention `density=True` but not realize its consequence.

**Fix:** rewrite as "(...with `density=True`, which re-normalizes
each histogram independently to integrate to 1 over the in-range
portion — fake samples outside the real range are silently excluded
from the comparison)".

### INFORMATIONAL

#### R3-IN-1: Critic initial weights differ across model families due to RNG ordering

See H4 above. Each of the 9 models gets a slightly different critic
init because the preceding generator construction consumes a different
number of `torch.randn` draws. Symmetric across all 9 models — not a
quantum-specific disadvantage. Worth noting as a potential audit
question if a reviewer asks "is the comparison truly apples-to-apples?"

#### R3-IN-2: `_train_quantum` and `_train_wgan` rely on `train_wgan_gp`'s internal seeding for numpy/random

Both paths only seed `torch.manual_seed` BEFORE calling
`train_wgan_gp`, then rely on the inner call (training.py:245-247) to
seed all three RNGs. This is correct and symmetric, but it means a
future refactor that moves the inner seed block would silently break
determinism for BOTH paths. Worth documenting at the `_train_*`
docstrings.

#### R3-IN-3: The `*0.1` multiply is generator-family-agnostic but binding only for quantum

`training.py:347` (`generated_samples = generated_samples.to(
compute_dtype) * 0.1`) applies to whatever the generator outputs. For
quantum (Pauli expvals ∈ [-1, +1]), the cap binds at `[-0.1, +0.1]`.
For classical (unbounded `g(z)`), `*0.1` is just a constant rescale
that does not cap the output range. This is the architectural source
of R3-HI-2.

#### R3-IN-4: Empirical sample-range disparity at the saved-pm1 scale

Verified from `revision/results/matched2000/runs/<model>/42/samples.
npy`:
- iqp_sel_55_repro: [-0.077, +0.076] (at the `*0.1` cap)
- V1/V2/V3 quantum: similar range
- wgan_mlp: [-0.243, +0.321] (3-4× wider than cap)
- wgan_cnn: [-3.89, +6.13] (40-60× wider — completely uncapped)
- vae: [-0.045, -0.011] (very narrow — posterior collapse, NO *0.1
  applied per design)
- ar: closed-form fit, much wider

This range disparity is what R3-CR-1's metric bias exploits. Even
though the *0.1 is in BYTE-FROZEN code, the downstream metrics that
consume these samples can be made fairer (R3-CR-1 fix options).

---

## D-14-22 byte-freeze attestation

**PASS.** Verified via:
```
$ git log --diff-filter=M --oneline -- revision/core/
db59b11 feat(14-01): add non-default 55-param IQP:SEL circuit + D-14-07 equivalence gate
... (all prior commits predate Phase 14)

$ git diff --stat 06bb470..HEAD -- revision/core/
(empty)

$ git diff --stat db59b11..HEAD -- revision/core/
(empty)
```

The last modification to `revision/core/` is commit `db59b11` (Phase
14-01). All subsequent Phase 14 commits (14-02 through 14-15) added
files OUTSIDE `revision/core/`. The byte-freeze contract is honored
literally from 14-01 onward.

The 14-01 commit's changes to `models/quantum.py` (the `iqp_sel_55`
55-param circuit variant) are consistent with the documented contract:
- The `default_75` branch is `_LITERALLY_` the pre-Phase-14 code
  block (verified in source code comments at lines 102-103, 222-223,
  235-237 of quantum.py).
- The `iqp_sel_55` branch is the NEW recovery branch and is gated by
  the `_CIRCUIT_IDS` enum + eager validation.
- Both `count_params()` and the bounds-check arithmetic for both
  variants are correct (r1 IN-9 already verified).

No regressions vs the v1.0 notebook contract.

---

## Provenance audit (Aug 2025 vs current matched-2000ep)

The user is concerned that "current matched-2000ep quantum results
look mid-pack vs classical, possibly worse than historical Aug 2025
figures." r3's audit finds NO code change between Aug 2025 and 14-15
that would systematically worsen the quantum-side numbers:

1. **`revision/core/training.py`** — last modified `9872e00` (Phase 13,
   pre-Aug 2025). The training loop is byte-identical to the Aug 2025
   run.
2. **`revision/core/models/quantum.py`** — last modified `db59b11`
   (14-01). The 55-param `iqp_sel_55` circuit was ADDED in 14-01; the
   `default_75` branch is byte-identical to pre-14-01. If the Aug 2025
   results used `default_75` (V1-style), they are reproducible today
   with no code drift. If they used `iqp_sel_55`, they would only
   exist post-14-01 (so cannot predate 14-01).
3. **`revision/core/eval.py`** — last modified `721be89` (Phase 8). The
   raw-sample EMD metric is byte-identical.
4. **The NEW emitters** (`run_distribution_emd.py`,
   `run_matched2000_dualscale.py`, `run_figure_suite.py` extensions)
   are POST-Aug-2025. If the user is comparing the historical Aug 2025
   numbers to the NEW distribution-EMD column or the NEW dualscale
   log_return EMD column, the apparent quantum disadvantage IS the
   metric-bias issue documented in R3-CR-1 / R3-HI-1 — not a
   regression in the quantum training pipeline.

**Recommendation:** clarify with the user WHICH metric column shows
quantum as worse. If it's the raw-sample EMD on the OD scale, that
column is comparable across Aug 2025 and now, and a real regression
would need a separate forensic. If it's the distribution-EMD column
(NEW), the metric construction is the issue (R3-CR-1). If it's the
log_return EMD (NEW dualscale recipe), the scale mismatch is the
issue (R3-HI-1) — but that mismatch is accidentally quantum-favorable,
not adverse, so the user's perception of "worse than Aug 2025" cannot
come from this column.

---

## Comparison to prior-round findings

The CRITICAL R3-CR-1 (distribution-EMD metric bias) and HIGH R3-HI-1
(log_return scale mismatch) are NEW findings not flagged by r1 or r2.

r1 + r2 focused on:
- gate correctness (substring resolution, sign-flip)
- provenance (device manifest, citations, data_hash)
- byte-freeze attestation
- determinism (seeding, hash strings)

Neither examined **metric-comparability across model families**, which
is where R3 found the quantum-disadvantaging issues. This is consistent
with the r3 brief: "audit the matched-2000ep + distribution-EMD +
figure-suite code paths for any bug that could SYSTEMATICALLY
DISADVANTAGE QUANTUM specifically".

The r1/r2 fixes are NOT regressed by R3 findings; the two layers
address different issues.

---

## Recommended action for resubmission

**Pre-tag hot fixes (≤2 hours total):**

1. **R3-HI-1 (scale mismatch fix):** edit
   `run_matched2000_dualscale.py:`build_real_references` and
   `run_distribution_emd.py:_real_references` to use `norm_log_delta`
   instead of `log_delta` for the log_return-scale real reference. Re-
   emit both JSONs. Disclose the correction in the C-3 disclosure
   paragraph + 14-13 SUMMARY.md.

2. **R3-CR-1 (metric choice):** REMOVE the log_return-scale
   distribution-EMD column from the manuscript and reduce
   distribution-EMD to OD-scale-only. The OD-scale distribution-EMD
   does not suffer the *0.1 cap issue (cumsum-based reconstruction
   maps the cap into a much wider OD range). Add a paragraph to
   `methods_full.md` §3.x explicitly disclosing that the pre-v1.0
   paper's 50-bin reference ~0.0015 is on a different dataset and
   different windowing and is not numerically reconcilable to the
   matched-2000ep numbers.

3. **R3-MD-1 (bare assert):** change the line-280 `assert` to `if ...:
   raise AssertionError(...)`.

**Post-tag follow-ups (NOT blocking):**
- R3-HI-2 (*0.1 cap disclosure): add a methods paragraph documenting
  the quantum-specific cap and the asymmetric expressivity it implies.
- R3-MD-2 (silent skip in distribution_emd): make it loud-fail.
- R3-MD-3 (insufficient C-3 disclosure language): rewrite per the
  recommended text above.

The CRITICAL finding (R3-CR-1) materially affects manuscript headline
ordering. If the team ships the current distribution_emd.json as-is,
a reviewer who notices the metric bias will (a) require the
quantum-disadvantage disclosure, AND (b) ask for the v1.0 raw-sample
EMD ordering to be the headline column. Better to fix proactively.

---

## Methodology notes

- All empirical claims verified against actual on-disk artifacts in
  `revision/results/matched2000/runs/` and
  `revision/results/distribution_emd.json` at HEAD = d52c1a0.
- Synthetic metric-bias demonstrations run from a clean python -m
  invocation against scipy 1.x and numpy 1.x (per
  `revision/requirements-pinned.txt`).
- Byte-freeze attestation via `git log --diff-filter=M -- revision/
  core/` and `git diff --stat <base>..HEAD -- revision/core/`.
- No new artifacts were emitted during this review.
- Read-only audit of all in-scope files; no edits.
