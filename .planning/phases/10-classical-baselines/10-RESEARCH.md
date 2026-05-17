# Phase 10: Classical Baselines — Research

**Researched:** 2026-05-17
**Domain:** Matched-parameter classical generative models (WGAN-GP MLP/CNN/LSTM, VAE, AR) for length-10 univariate windows; sweep orchestration mirroring Phase 09.1
**Confidence:** HIGH (fully scoped by 24 locked decisions + an existing, verified Phase 8/9/09.1 codebase; the only genuine design work is param-arithmetic recipes and the `train_wgan_gp` generator-contract adapter)

## Summary

Phase 10 is glue + small new model definitions on top of an already-verified infrastructure. The 24 locked D-10-XX decisions settle every architectural question; this research supplies the *implementation knowledge* the planner needs so neither planner nor executor has to guess: exact parameter arithmetic to land each classical generator in 71–79 params, the precise (and slightly awkward) interface contract `revision/core/training.py::train_wgan_gp` imposes on a generator, VAE/AR sizing, the 5-file artifact bundle + data-hash formula, the reusable TSTR-lite helper, and the pitfalls.

The single most important finding — and the one most likely to derail the executor if not surfaced now — is that **`train_wgan_gp` is hard-coded to the quantum generator's interface in two places**: line 234 builds the generator optimizer as `torch.optim.Adam([generator.params_pqc], ...)` (a single named tensor, not `generator.parameters()`), and lines 282/315 call `generator(noise_batch)` where `noise_batch` has shape `(num_qubits, batch_size)` and the generator must return `(batch, window_length)`. A standard `nn.Module` classical generator has neither a `.params_pqc` attribute nor a `(num_qubits, batch)`→`(batch, 10)` forward signature. The classical generators must therefore expose a `.params_pqc` shim (an `nn.Parameter`-flattening view is not viable; see Pitfall 1) **OR** the plan must add a minimal CONTEXT-authorized adapter. This is the central plan-shaping decision and is analyzed in detail below.

**Primary recommendation:** Implement the 3 classical WGAN-GP generators in `revision/core/models/classical.py` with a forward signature `forward(noise: Tensor[num_qubits, B]) -> Tensor[B, 10]` and a `params_pqc` **property that aliases the module's single trainable parameter bundle** so `train_wgan_gp` plugs in unchanged (D-10-13 forbids touching `revision/core/`). Size each to land in 71–79 params using the exact formulas in §Standard Stack. VAE/AR live in `revision/core/models/nonadversarial.py` with their own training loops in `revision/run_baselines.py` (D-10-11/12/13). Mirror `run_ablation.py`/`run_ablation_sweep.sh` exactly for the 50-run sweep, adding only a `data_hash` field (new — Phase 09.1 has none).

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-10-01:** 5 NEW model types — WGAN-GP {MLP, 1D-CNN, RNN/LSTM} + non-adversarial {VAE, AR}.
- **D-10-02:** Each classical WGAN-GP variant matched within ±5% of PQC's 75 trainable params (target range 71–79).
- **D-10-03:** VAE/AR NOT param-matched; sized to natural minimum; counts reported transparently. VAE = "smallest deep VAE that trains stably" on length-10 windows; AR = order-p coefficients (parameter-minimal by definition).
- **D-10-04:** Quantum generator is the reference; its 5-seed × 2-pipeline (A,B) runs from Phase 09.1 are reused as-is. NO quantum retraining in Phase 10.
- **D-10-05:** Train on BOTH Pipeline A (min-max OD) and Pipeline B (log-returns standardized). Pipeline C dropped.
- **D-10-06:** Pipeline B is the headline; Pipeline A is the supplementary "raw OD" control. Both reported in the comparison table.
- **D-10-07:** Same windowed data from `load_and_preprocess` + `rolling_window(WINDOW_LENGTH=10, stride=2)`; per-pipeline forward/inverse from `revision/core/preprocessing.py`.
- **D-10-08:** Identical training conditions across all WGAN-GP variants and the quantum reference: seeds {42,43,44,45,46}; 1000 epochs; N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, BATCH_SIZE=12, WINDOW_LENGTH=10; **same `Critic`** for every WGAN-GP variant; same Adam betas, same GP formulation, same windowed loader.
- **D-10-09:** VAE/AR have model-family-specific training (VAE→ELBO; AR→MLE/least-squares) but the data, seeds, epoch budget, and held-out eval split are matched to the WGAN-GP track. Per-model protocol documented in JSON.
- **D-10-10:** 5×2×5 = 50 new runs; ≈110 min at `--parallel 2`; ≤3 h sweep budget; relaxable if classical >5× faster than quantum.
- **D-10-11:** NEW `revision/core/models/classical.py` — all 3 classical WGAN-GP generators as `nn.Module` subclasses with a shared `count_params()` matching `QuantumGenerator.count_params()`.
- **D-10-12:** NEW `revision/core/models/nonadversarial.py` — VAE + AR; both trained outside the WGAN-GP loop.
- **D-10-13:** All training-loop/aggregation/orchestration logic stays OUT of `revision/core/` — only model definitions go there. Orchestration in NEW `revision/run_baselines.py` + `revision/run_baselines_sweep.sh`, patterned after `run_ablation.py`/`run_ablation_sweep.sh`.
- **D-10-14:** Sweep outputs at `revision/results/baselines/runs/<model_kind>/<pipeline>/<seed>/`. Model kinds: `wgan_mlp`, `wgan_cnn`, `wgan_lstm`, `vae`, `ar`. Each run dir = same 5-file bundle as 09.1: `config.yaml`, `checkpoint.pt` (or `.npz` for AR), `samples.npy`, `metrics.json`, `inverse_kwargs.npz`.
- **D-10-15:** A `data_hash` field written to every `config.yaml`, computed `sha256(real_OD.tobytes())[:16]`. Must match across all 50 new runs AND across the Phase 09.1 quantum runs.
- **D-10-16:** `baseline_comparison.json` aggregates every model×pipeline×seed into long-form `{model_kind, pipeline, seed, metric_name, scale, value}` + a top-level `models[]` array `{kind, parameter_count, family, train_protocol_notes}`.
- **D-10-17:** Companion `baseline_comparison.md` markdown table, one row per model, columns: parameter count, OD-EMD (mean±std), OD-ACF lag-1, OD-DTW mean, transformed-space EMD (Pipeline B), TSTR-lite R².
- **D-10-18:** Table reports BOTH the quantum reference (5 seeds × 2 pipelines from Phase 09.1) AND every new model on the same pipeline rows.
- **D-10-19:** NO new recommendation in this phase. Phase 14 decides the highlighted baseline. Phase 10 only delivers the apples-to-apples table.
- **D-10-20:** Every model emits the same per-run fidelity metric set as 09.1 (OD-scale EMD, moments, per-lag ACF mean+std lags 0..9, DTW mean/median/std on NN sub-sample, transformed-space EMD where applicable). All via `revision/core/eval.py` — NO new metric helpers.
- **D-10-21:** TSTR-lite (1-layer LSTM-32, 3 init seeds {40,41,42}, 320 held-out real windows) per model×pipeline as sanity scaffolding. Phase 11 owns full TSTR.
- **D-10-22:** `revision/run_baselines.py` per-(model,pipeline,seed) CLI driver: `python -m revision.run_baselines --model {wgan_mlp,wgan_cnn,wgan_lstm,vae,ar} --pipeline {A,B} --seed N --epochs M`. Idempotent.
- **D-10-23:** `revision/run_baselines_sweep.sh` loops 5×2×5=50, skips complete pairs (same 5-file `is_complete()`), writes `sweep_status.json`, `--parallel {1,2}` guardrail, same atomic-status-writer pattern as `run_ablation_sweep.sh`.
- **D-10-24:** NEVER `multiprocessing.Pool`. `xargs -P 2` OS-process parallelism only (Phase 09.1 Pitfall 4).

### Claude's Discretion

- Exact layer dimensions of each classical generator within the 71–79 window (this research gives concrete recipes; planner may pick among them).
- VAE encoder/decoder/latent dims (research gives a concrete "smallest stable" recipe).
- AR order-p selection method (research recommends a concrete default).
- Whether the `params_pqc` contract is satisfied by a property-alias on the classical module or a thin local adapter in `run_baselines.py` (research recommends the property-alias; both are valid).
- Wave grouping (research suggests a structure; planner has final call).

### Deferred Ideas (OUT OF SCOPE)

- Param-matched VAE/AR — future phase if reviewers ask.
- Classical-generator + quantum-critic hybrid — v3.0.
- Larger ansatz variants — Phase 13 (would shift Phase 10's matched-param target).
- Full TSTR/predictive/discriminative suite → Phase 11. Shot-noise → Phase 12. Manuscript integration → Phase 14. Final baseline-to-highlight decision → Phase 14.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **BASE-01** | Classical WGAN-GP generator matched within ±5% of PQC param count; identical critic/optimizer/schedule/seeds; full metric suite alongside quantum | §Standard Stack (param arithmetic for MLP/CNN/LSTM), §`train_wgan_gp` Contract, §Pitfall 1 (the `params_pqc` shim), D-10-08 invariant |
| **BASE-02** | Non-adversarial baseline (VAE or AR) on same data + same metrics | §VAE Sizing, §AR Baseline, D-10-09 protocol |
| **BASE-03** | Param-count-controlled comparison table JSON + markdown (quantum / classical WGAN-GP / VAE-AR side-by-side) | §Comparison Table Schema, §Data-Hash Cross-Check, reuse of 09.1 quantum runs |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 3 classical WGAN-GP generator definitions | `revision/core/models/classical.py` (NEW module) | — | D-10-11 locks; analog of `quantum.py` |
| VAE + AR model definitions | `revision/core/models/nonadversarial.py` (NEW module) | — | D-10-12 locks |
| WGAN-GP training | `revision.core.training.train_wgan_gp` (UNCHANGED) | classical generator's `params_pqc` shim | D-10-08/13: reuse loop verbatim; classical gen must satisfy the loop's hard-coded interface |
| VAE training (ELBO) | `revision/run_baselines.py` (NEW orchestration) | `revision.core.models.nonadversarial.VAE` | D-10-09/13: VAE loop ≠ WGAN-GP loop; loop logic stays out of `core/` |
| AR fit (MLE/Yule-Walker) + sampling | `revision/run_baselines.py` (NEW orchestration) | `revision.core.models.nonadversarial.AR` | D-10-13: fit/sample orchestration is not a "model definition" |
| Per-(model,pipeline,seed) run driver | `revision/run_baselines.py` (NEW) | `revision.core.preprocessing`, `train_wgan_gp` | D-10-22; mirror `run_ablation.py` |
| 50-run sweep + status JSON | `revision/run_baselines_sweep.sh` (NEW) | xargs -P 2 | D-10-23/24; mirror `run_ablation_sweep.sh` |
| Per-run fidelity metrics | `revision.core.eval` (UNCHANGED — no new helpers) | run driver / analysis notebook | D-10-20 forbids new metric helpers |
| Comparison table + markdown + TSTR-lite | analysis notebook (NEW, e.g. `06_baseline_comparison.ipynb`) | matplotlib/pandas | D-10-13/16/17/21; notebooks aggregate+render, `core/` is logic-only |
| OD-scale reconstruction from samples.npy | analysis notebook (reuse 09.1 `reconstruct_od` pattern) | `revision.core.preprocessing.inverse_*` | identical inverse contract as 09.1 |

## Standard Stack

### Core (all already installed and used by Phase 8/9/09.1)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | 2.9.0 | `nn.Module` generators, WGAN-GP autograd, VAE, AR-via-lstsq | Project standard `[VERIFIED: ./qgan_env import check 2026-05-17]` |
| `numpy` | 2.3.4 | sample arrays, AR Yule-Walker, data-hash `tobytes()` | Project standard `[VERIFIED]` |
| `scipy` | 1.16.2 | `wasserstein_distance` (EMD), already wrapped in `eval.py` | Existing dependency `[VERIFIED]` |
| `statsmodels` | 0.14.5 | ACF (`fft=True`); **optionally** `statsmodels.tsa.ar_model.AutoReg` for AR | Already wrapped in `eval.py:64-72` `[VERIFIED]` |
| `fastdtw` | (installed, import OK) | DTW, wrapped in `eval.py:78-90` | Existing `[VERIFIED]` |
| `PyYAML` | 6.0.3 | per-run `config.yaml` | Used by `run_ablation.py:42` `[VERIFIED]` |
| `pennylane` | 0.43.0 | only to instantiate the quantum reference for the comparison table (no retrain) | Existing `[VERIFIED]` — note: 09.1 RESEARCH said 0.44.0; current env is **0.43.0** |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.nn.{Linear,Conv1d,ConvTranspose1d,LSTM}` | bundled | classical generator + VAE building blocks | core of `classical.py`/`nonadversarial.py` |
| inline R² (`1 - ss_res/ss_tot`) | n/a | TSTR-lite R² | **`sklearn` is NOT installed** `[VERIFIED: import sklearn → ModuleNotFoundError]`. Precedent: `_build_analysis_notebook.py:442 r2_score_inline` already does this. Reuse verbatim. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled AR Yule-Walker | `statsmodels.tsa.ar_model.AutoReg` | `AutoReg` is robust + gives the order-p coefficients and noise variance directly; hand-rolled `np.linalg.lstsq` is ~6 lines and avoids depending on a less-used statsmodels API. **Recommend least-squares fit** (see §AR Baseline) — minimal, transparent, parameter count is exactly p+1. |
| `statsmodels` AR sampling | manual recursive simulation | Manual recursion is required anyway to emit `(N, 10)` windowed `samples.npy` matching the 09.1 sample contract; `statsmodels` `.simulate` exists but the manual loop is clearer and matches the windowed-output shape. |
| sklearn `r2_score` | inline `r2_score_inline` | sklearn missing; inline is the established project pattern. Do NOT add sklearn as a dependency. |
| One classical generator, vary by config | 3 distinct `nn.Module` subclasses | D-10-11 locks 3 subclasses (MLP, CNN, LSTM). No latitude. |

**Installation:** None required. Every package is already present in `./qgan_env`. `[VERIFIED: ./qgan_env/bin/python import check 2026-05-17]`

**Version verification performed 2026-05-17** via `./qgan_env/bin/python -c "import ...; print(__version__)"`:
torch 2.9.0, numpy 2.3.4, pennylane 0.43.0, scipy 1.16.2, statsmodels 0.14.5, pyyaml 6.0.3, fastdtw OK, **sklearn MISSING**.

## Package Legitimacy Audit

> Not applicable — Phase 10 installs **zero** new packages. Every dependency is already in the verified `./qgan_env` from Phase 8/9/09.1. No registry lookup or slopcheck needed (nothing to install). All model code is built from `torch.nn` primitives + existing `revision.core` modules.

## The `train_wgan_gp` Generator Contract (CRITICAL — read before planning)

The 3 classical WGAN-GP generators must drop into `revision.core.training.train_wgan_gp` **unchanged** (D-10-08 "same critic ... same windowed loader"; D-10-13 "training-loop logic stays out of `core/`"). Verbatim from `revision/core/training.py`:

1. **Generator optimizer (line 234):**
   ```python
   g_opt = torch.optim.Adam([generator.params_pqc], lr=lr_generator, betas=(0.0, 0.9))
   ```
   The loop builds the generator optimizer over the **single attribute `generator.params_pqc`**, NOT `generator.parameters()`. `[VERIFIED: revision/core/training.py:234]`

2. **Forward call (lines 282, 315, 349):**
   ```python
   noise_batch = torch.tensor(np.random.uniform(NOISE_LOW, NOISE_HIGH,
                              size=(num_qubits, batch_size)), dtype=torch.float32)
   generated_samples = generator(noise_batch)        # expected: (batch, window_length)
   generated_samples = generated_samples.to(torch.float64) * 0.1
   ```
   Input noise shape is `(num_qubits, batch_size)` = `(5, 12)`; the loop transposes nothing — it expects `generator(noise)` to return `(batch_size, window_length)` = `(12, 10)`, then casts to float64 and **multiplies by 0.1**. `[VERIFIED: revision/core/training.py:275-286, 309-317]`

3. **`num_qubits` / `window_length` resolution (lines 228-229):**
   ```python
   num_qubits = getattr(generator, "num_qubits", NUM_QUBITS)       # falls back to 5
   window_length = getattr(generator, "window_length", WINDOW_LENGTH)  # falls back to 10
   ```
   So a classical generator either exposes `.num_qubits`/`.window_length` or the loop uses 5/10. Set them explicitly to `5`/`10` for clarity. `[VERIFIED: revision/core/training.py:228-229]`

4. **EarlyStopping adapter (line 255):** `_ESAdapter` reads `generator.params_pqc`. Not triggered (`run_ablation.py` passes no `early_stopper`), but the attribute must exist. `[VERIFIED: revision/core/training.py:255, 406-429]`

5. **`generate_samples` in `run_ablation.py` (lines 195-209):** uses the same `(NUM_QUBITS, bs)` noise → `generator(noise).to(float64) * 0.1`. `run_baselines.py` must mirror this so `samples.npy` is in the same `[-1,1]·0.1`-style space the 09.1 `reconstruct_od` helper expects. `[VERIFIED: revision/run_ablation.py:180-209]`

**Implication for `classical.py`:** Each classical generator must expose:
- `forward(noise: Tensor) -> Tensor` accepting `(5, B)`-shaped noise and returning `(B, 10)`. The `(5, B)` noise is the *latent input*; the generator may flatten/transpose it internally (it is just a `(5,B)` uniform-noise tensor — treat it as a 5-dim latent per sample, i.e. transpose to `(B, 5)` as the latent vector). **Latent-dim convention: 5** (= `num_qubits`), matching the quantum generator's input contract.
- A `params_pqc` attribute that **is the single `nn.Parameter`/tensor the generator optimizer should update**. See Pitfall 1 for why a naive property returning a flattened copy breaks autograd, and the recommended pattern.
- `count_params() -> int` returning the total trainable count (D-10-11), and `num_qubits=5`, `window_length=10`.

## Standard Stack — Parameter Arithmetic Recipes (the core of BASE-01)

Target: **71 ≤ trainable params ≤ 79** (PQC = 75, ±5% per D-10-02). Latent dim = 5 (the `(5,B)` noise contract). Output = length-10 window. All counts are `sum(p.numel() for p in module.parameters())` — verify in an acceptance test against `count_params()`.

Param formulas:
- **Linear** `nn.Linear(in, out)`: `in*out + out` (bias on). Set `bias=False` to drop the `+out`.
- **Conv1d** `nn.Conv1d(C_in, C_out, k)`: `C_in*C_out*k + C_out`.
- **ConvTranspose1d** `nn.ConvTranspose1d(C_in, C_out, k)`: `C_in*C_out*k + C_out`.
- **LSTM** `nn.LSTM(input_size=I, hidden_size=H, num_layers=1)`: `4*(I*H + H*H + 2*H)` (PyTorch has two bias vectors `b_ih`, `b_hh` → the `+2*H` per gate-block, i.e. `4*(I*H+H*H+H+H)`). Verify empirically — PyTorch LSTM bias count is the classic gotcha.

### (a) MLP generator — recommended primary

Latent 5 → hidden h → output 10. Two `Linear` layers.
Params = `(5*h + h) + (h*10 + 10) = 16h + 10`.
Solve `71 ≤ 16h + 10 ≤ 79` → `3.81 ≤ h ≤ 4.31` → **h = 4 → 16·4+10 = 74 params** ✓ (in range, 1.3% under PQC).
Alternative single-layer: `nn.Linear(5,10)` = `5*10+10 = 60` (too low). With `bias=False` on layer 2: `(5*h+h) + (h*10) = 16h` — h=5 → 80 (just over); h=4→64 (under). **Use h=4, both biases on → 74 params.** Activation between layers (e.g. `nn.Tanh()` — bounds output toward [-1,1], matching the generator-output range the loop scales by 0.1; Tanh has zero params).

```
Latent(5) --Linear(5,4)+bias--> Tanh --Linear(4,10)+bias--> (B,10)
params = (5*4+4) + (4*10+10) = 24 + 50 = 74   [target 71-79 ✓]
```

### (b) 1D-CNN generator

Reshape latent `(B,5)` → `(B,1,5)`, upsample to length 10 via a transpose conv, then a 1×1 conv to mix. Keep channels tiny.
Design: `ConvTranspose1d(1, C, k)` to go length 5→10, then `Conv1d(C, 1, 1)`.
- ConvTranspose1d length: `L_out = (L_in-1)*stride - 2*pad + k`. For `L_in=5, stride=2, pad=0`: `L_out = 8 + k - ... ` — pick `k`/`stride`/`pad` to hit 10. Simpler: `stride=1, pad=0, k=6` → `L_out = 5-1 + 6 = 10`. Params `1*C*6 + C = 7C`.
- `Conv1d(C, 1, kernel=1)`: `C*1*1 + 1 = C + 1`.
Total = `7C + C + 1 = 8C + 1`. Solve `71 ≤ 8C+1 ≤ 79` → `8.75 ≤ C ≤ 9.75` → **C = 9 → 8·9+1 = 73 params** ✓.

```
Latent(5)→view(B,1,5) --ConvTranspose1d(1,9,k=6,s=1)+bias--> (B,9,10)
                       --Conv1d(9,1,k=1)+bias--> (B,1,10) → view (B,10)
params = (1*9*6 + 9) + (9*1*1 + 1) = 63 + 10 = 73   [target 71-79 ✓]
```
(Add a parameter-free `nn.Tanh()` or `nn.LeakyReLU(0.1)` between — LeakyReLU/Tanh have 0 params.)

### (c) RNN/LSTM generator

Feed the 5-dim latent as a length-5 input sequence of 1 feature each, or as `(B, 1, 5)`. Smallest stable: `LSTM(input_size=1, hidden_size=H, num_layers=1)` consuming the latent as a length-5 sequence, then `Linear(H, 10)` from the last hidden state.
- LSTM params (PyTorch, num_layers=1, bias=True): `4*(1*H + H*H + H + H) = 4*(H² + 3H)`.
- `Linear(H, 10)`: `H*10 + 10`.
Total `T(H) = 4H² + 12H + 10H + 10 = 4H² + 22H + 10`.
- H=2 → `16 + 44 + 10 = 70` (just under 71; 6.7% under — borderline outside ±5%).
- H=3 → `36 + 66 + 10 = 112` (way over).
70 is 6.7% below 75 (outside ±5% lower bound of 71.25). Tighten with `Linear(H,10, bias=False)`:
`T(H) = 4H²+12H + H*10` → H=2 → `16+24+20 = 60` (worse). Or drop LSTM bias (`bias=False`): LSTM params `4*(I*H + H*H) = 4*(H²+H)`; H=2 → `4*6=24`; + `Linear(2,10)+bias = 30` → 54 (under). H=3 LSTM no-bias `4*12=48` + `Linear(3,10)+bias=40` → 88 (over).
**Best fit:** `LSTM(input_size=2, hidden_size=2, num_layers=1, bias=True)` consuming latent reshaped to `(B, seq=?, 2)` (use the 5-dim latent padded/projected — see note) → params `4*(2*2 + 2*2 + 2 + 2) = 4*12 = 48`; + `Linear(2,10)+bias = 30` → **78 params** ✓ (4% over PQC, in 71–79).

```
Latent(5) → pad/slice to (B, seq=3, 2)  (use first 6 of a 5-dim latent
            tiled/padded to 6, or project — keep it parameter-free: tile/pad)
  --LSTM(input_size=2, hidden_size=2, num_layers=1, bias=True)-->
  last hidden (B,2) --Linear(2,10)+bias--> (B,10)
params = 4*(2*2 + 2*2 + 2 + 2) + (2*10+10) = 48 + 30 = 78   [71-79 ✓]
```

> **Acceptance-criterion guidance for the planner:** Specify the *exact* layer dims above and assert `module.count_params() == <value>` AND `71 <= count_params() <= 79` AND `count_params()` within ±5% of `QuantumGenerator().count_params()` (==75). The PyTorch LSTM bias count is the #1 arithmetic trap — the acceptance test MUST compute `sum(p.numel() for p in m.parameters())` empirically, never trust the formula alone (Pitfall 4).

> **Latent-shape note:** the loop hands the generator `(num_qubits=5, B)` noise. The classical generators should transpose to `(B,5)` and treat it as a 5-dim latent. The CNN/LSTM reshapes above need a parameter-free deterministic map from `(B,5)` to their expected input shape (e.g. `x = noise.T; x = x.reshape(B, ...)` with tiling/padding, NOT a learnable projection — a learnable projection would add params and blow the budget). Document the exact reshape in the module docstring.

## VAE Sizing (BASE-02, D-10-03/09)

"Smallest deep VAE that trains stably" on length-10 windows. Not param-matched (D-10-03) — report its count transparently. Recommended minimal-but-stable architecture (encoder→latent→decoder, fully-connected, Gaussian latent, standard ELBO):

```
Encoder:  Linear(10, 16) → ReLU → [Linear(16, Lz) (mu), Linear(16, Lz) (logvar)]
Latent:   Lz = 4   (z = mu + eps*exp(0.5*logvar), eps~N(0,I))
Decoder:  Linear(4, 16) → ReLU → Linear(16, 10)   (no final activation; data in [-1,1])
```
Param count (report exactly in JSON; not constrained):
- enc Linear(10,16)=176; mu Linear(16,4)=68; logvar Linear(16,4)=68
- dec Linear(4,16)=80; Linear(16,10)=170
- Total ≈ **562** params. (Disclosed transparently per D-10-03; this is the natural "smallest deep VAE" size — a single 16-unit hidden layer each side. Going smaller (hidden=8) trains less stably; going larger is unnecessary for length-10.)

**ELBO loss** (the training objective — lives in `run_baselines.py`, NOT `core/`, per D-10-13):
```
recon = MSE(x_hat, x)            # or sum over dims; use mean for scale stability
kld   = -0.5 * mean( 1 + logvar - mu^2 - exp(logvar) )
loss  = recon + beta * kld       # beta = 1.0 (standard VAE); document in JSON
```
**Training loop differences vs WGAN-GP:** no critic, no gradient penalty, no n_critic inner loop, single optimizer over `vae.parameters()` (Adam, lr e.g. 1e-3). Same data loader / windowed input / seed / epoch budget as the WGAN-GP track (D-10-09). Sampling: draw `z ~ N(0, I)` of shape `(N, Lz)`, run decoder → `(N, 10)` → that is `samples.npy` (same windowed `[-1,1]`-ish space the pipeline inverse expects; do NOT apply the `*0.1` scaling — that scaling is a quantum-generator-output artifact in `training.py:283`. Document this asymmetry in the VAE JSON `train_protocol_notes`).

> **Pitfall surfaced (see Pitfall 3):** the `*0.1` post-scaling at `training.py:283/316` and `run_ablation.py:205` is applied to *generator output before it hits the critic and before samples are saved*. WGAN-GP classical generators go through `train_wgan_gp` so they inherit `*0.1` automatically via the loop and must replicate it in their `generate_samples`. VAE/AR do NOT go through that loop, so their `samples.npy` must be produced in whatever space the pipeline-inverse + 09.1 `reconstruct_od` expects. The 09.1 `reconstruct_od` treats `samples.npy` as already in `[-1,1]` window space (it maps `(samples+1)/2`). The WGAN path produces `[-1,1]·0.1 ≈ [-0.1,0.1]` then reconstruct maps that — meaning the *quantum* 09.1 samples are also in the `·0.1` space and `reconstruct_od` handles them. To keep VAE/AR comparable, **VAE/AR samples must be emitted in the same scaled space the pipeline inverse expects** — i.e. produce outputs whose value range matches what `reconstruct_od` consumes. Concretely: the safest, most defensible choice is for VAE/AR to emit samples in the `[-1,1]` window space and apply the **same `inverse_kwargs` reconstruction path** as the WGAN models, and to NOT replicate the `*0.1` (since `*0.1` is a quantum-output-magnitude correction, not part of the preprocessing inverse). The planner MUST add an explicit acceptance check: reconstruct one VAE sample and one WGAN sample through the identical pipeline inverse and confirm both land in the real OD range. This is the single largest cross-model-comparability risk in the phase — flag it as a Wave-2 smoke gate.

## AR Baseline (BASE-02, D-10-03/09)

Autoregressive linear model of order p on the windowed series. Parameter-minimal by definition: `p` AR coefficients + 1 noise variance = **p+1 params**.

**Order-p selection:** windows are length 10, so p must be small. **Recommend p = 2** (AR(2)) as the default — a defensible minimal order for length-10 windows; report the choice and rationale in the JSON `train_protocol_notes`. (AR order selection by AIC/BIC over p∈{1,2,3} on the flattened training series is an acceptable alternative; the planner may pick. p=2 → 3 params total.)

**Fit (least-squares MLE-equivalent for Gaussian innovations) — lives in `run_baselines.py`:**
```
# x: flattened standardized training series (per pipeline transformed space)
# Build design matrix from p lags, solve via np.linalg.lstsq:
#   x[t] ≈ phi_1 x[t-1] + ... + phi_p x[t-p] + eps
X = np.stack([x[p-1-k : -1-k] for k in range(p)], axis=1)  # (T-p, p)
y = x[p:]
phi, *_ = np.linalg.lstsq(X, y, rcond=None)                # (p,)
resid = y - X @ phi
sigma2 = float(resid.var(ddof=0))                          # noise variance
```
Equivalent: `statsmodels.tsa.ar_model.AutoReg(x, lags=p).fit()` — gives `.params` and `.sigma2`. Either is acceptable; **least-squares is recommended** (no extra API surface, exactly p+1 reportable params, fully transparent).

**Sample generation → `samples.npy` of shape `(N, 10)`:** recursively simulate windows:
```
rng = np.random.default_rng(seed)
for each of N windows:
    seed the first p values from the real-window starting context (or zeros + burn-in)
    for t in p..9: x[t] = phi @ x[t-1..t-p] + rng.normal(0, sqrt(sigma2))
```
Emit in the same transformed/windowed space as the other models so the shared pipeline-inverse + 09.1 `reconstruct_od` reconstructs OD identically (see Pitfall 3). Checkpoint is a `.npz` (D-10-14: "`checkpoint.pt` (or `.npz` for AR)") storing `phi`, `sigma2`, `p`.

## Run-Directory + Artifact Contract (mirror Phase 09.1 exactly)

Phase 09.1's 5-file bundle per run dir, verified from `run_ablation.py` + the `is_complete()` check in `run_ablation_sweep.sh:164-172`:

| File | Phase 09.1 content | Phase 10 adaptation |
|------|--------------------|---------------------|
| `config.yaml` | pipeline, seed, epochs, num_qubits, num_layers, window_length, batch_size, n_critic, lambda_gp, lr_critic, lr_generator, noise_low/high, eval_every, n_real_windows, inverse_kwargs_keys, csv_path `[VERIFIED: runs/B/42/config.yaml]` | + `model_kind`, + **`data_hash`** (NEW, D-10-15), + `parameter_count`, + `family`. For VAE/AR, WGAN-only fields (n_critic, lambda_gp, lr_critic) become null/omitted with a `train_protocol_notes` string. |
| `checkpoint.pt` | `{params_pqc, critic_state_dict}` | WGAN: `{gen_state_dict, critic_state_dict}`. **AR: `.npz`** with `{phi, sigma2, p}`. VAE: `.pt` with `{vae_state_dict}`. |
| `samples.npy` | `(N_synth, 10)` in scaled window space; `N_synth = 10 * n_real_windows` `[VERIFIED: run_ablation.py:316-318]` | same shape + N_synth rule; same scaled space (see Pitfall 3) |
| `metrics.json` | per-epoch training metrics dict from `train_wgan_gp` | WGAN: same. VAE: per-epoch ELBO/recon/kld. AR: fit diagnostics (sigma2, residual stats). |
| `inverse_kwargs.npz` | per-pipeline aux: A→{od_min,od_max}; B→{r_min,r_max,mu,sigma,od_starts} `[VERIFIED: run_ablation.py:123-164]` | **identical per-pipeline contract** (pipeline determines this, not model). C dropped (D-10-05). |

**`is_complete()` check (mirror `run_ablation_sweep.sh:164-172` verbatim):** all 5 files exist AND non-empty (`-s`). For AR, the bundle is `config.yaml, checkpoint.npz, samples.npy, metrics.json, inverse_kwargs.npz` — the sweep's `is_complete()` must check `checkpoint.npz` for `model_kind=ar` and `checkpoint.pt` otherwise (small conditional in the bash helper; D-10-14 explicitly anticipates the `.npz` variant).

**Data-hash formula (D-10-15) — NEW, no Phase 09.1 precedent:**
```python
import hashlib
raw = load_and_preprocess(str(csv_path))           # same entry point as 09.1
real_OD = raw["OD"].cpu().numpy()                   # float32, shape (778,)
data_hash = hashlib.sha256(real_OD.tobytes()).hexdigest()[:16]
```
> **CRITICAL — `[VERIFIED: grep over revision/ found zero data_hash/sha256/tobytes]`:** Phase 09.1 wrote **no** data-hash. The quantum 09.1 runs at `revision/results/transform_ablation/runs/{A,B}/{seed}/config.yaml` do **not** contain a `data_hash` field. Therefore the D-10-15 cross-check "the hash must match across the Phase 09.1 quantum runs" cannot be done by reading 09.1 configs. It must be done by **recomputing `sha256(real_OD.tobytes())[:16]` from the same `load_and_preprocess(csv_path)` call the 09.1 quantum runs used, and asserting equality**. Practically: the comparison-table step recomputes the hash once from `load_and_preprocess` and verifies all 50 new `config.yaml` hashes equal it; the *quantum* equivalence is established by construction (same code path, same CSV) and documented, not by reading a non-existent field. The planner must phrase the BASE-03 acceptance criterion accordingly — surface this so the executor does not waste time grepping 09.1 configs for a field that isn't there. The `csv_path` in 09.1 configs is `data.csv` `[VERIFIED: runs/B/42/config.yaml]` — Phase 10 must use the identical CSV path.

## Comparison Table Schema (BASE-03, D-10-16/17/18)

`baseline_comparison.json`:
```json
{
  "models": [
    {"kind": "quantum",   "parameter_count": 75,  "family": "adversarial-quantum",
     "train_protocol_notes": "PQC 5q×4L; reused from Phase 09.1 (no retrain, D-10-04)"},
    {"kind": "wgan_mlp",  "parameter_count": 74,  "family": "adversarial-classical", ...},
    {"kind": "wgan_cnn",  "parameter_count": 73,  "family": "adversarial-classical", ...},
    {"kind": "wgan_lstm", "parameter_count": 78,  "family": "adversarial-classical", ...},
    {"kind": "vae",       "parameter_count": 562, "family": "non-adversarial", ...},
    {"kind": "ar",        "parameter_count": 3,   "family": "non-adversarial", ...}
  ],
  "rows": [
    {"model_kind":"wgan_mlp","pipeline":"B","seed":42,
     "metric_name":"emd","scale":"OD","value":0.027},
    ...
  ]
}
```
- Long-form `rows` exactly mirrors 09.1's `metrics.csv` schema (`pipeline, seed, metric_name, scale, value`) plus `model_kind` (D-10-16).
- Quantum rows are produced by re-reading the Phase 09.1 quantum `samples.npy` + `inverse_kwargs.npz` at `runs/{A,B}/{seed}/` and running the **identical** OD-reconstruction + `eval.py` metric path used for the new models (D-10-18). The 09.1 `reconstruct_od` pattern (`_build_analysis_notebook.py:95-149`) is the reusable template — copy it into the new analysis notebook (D-10-13: aggregation lives in the notebook, not `core/`).
- `baseline_comparison.md` columns (D-10-17): model | params | OD-EMD (mean±std) | OD-ACF lag-1 | OD-DTW mean | transformed-EMD (Pipeline B) | TSTR-lite R². One row per model (aggregated over 5 seeds, per pipeline — table is per-pipeline blocks or a pipeline column).

## TSTR-lite Scaffolding (D-10-21) — REUSE, do not reinvent

The exact helper already exists at `revision/_build_analysis_notebook.py:431-477` and produced `revision/results/transform_ablation/tstr_lite.json`. `[VERIFIED: file read 2026-05-17]`. Spec confirmed:
- `TSTRLiteLSTM(hidden=32)`: 1-layer `nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)` + `nn.Linear(32,1)`.
- Input: window `[:, :9]` → predict `[:, 9:10]` (9→1 next-step).
- 3 init seeds **{40, 41, 42}** (NOT the training seeds 42-46).
- Eval on **first 320** real windows; `real_train_for_baseline = real_windowed_OD[320:]` (D-09.1 used `HELD_OUT_N=320`).
- `r2_score_inline(y_true, y_pred) = 1 - ss_res/ss_tot` (sklearn-free; sklearn is not installed).
- Train: Adam lr=1e-3, MSE, 50 epochs, batch 64, `np.random.default_rng(lstm_seed)` for shuffling.
- Per model×pipeline: pool synthetic OD windows across the 5 seeds (`np.concatenate([recon[(m,p,s)] for s in SEEDS])`), train 3 LSTMs (one per init seed), report mse_mean/std, r2_mean/std + per_init_seed (matches `tstr_lite.json` schema exactly).

**Planner directive:** the new analysis notebook copies `TSTRLiteLSTM`, `r2_score_inline`, `train_eval_tstr` verbatim from `_build_analysis_notebook.py:432-477`. Do NOT promote to `revision/core/` (D-10-13: scaffolding stays in the notebook, exactly as 09.1 did).

## Architecture Patterns

### System Architecture Diagram

```
              revision/core/preprocessing.py  (UNCHANGED — A & B only, C dropped)
                 forward_minmax_od / inverse_minmax_od        (Pipeline A)
                 forward_logreturns / inverse_logreturns      (Pipeline B)
                                  │
   ┌──────────────────────────────┼─────────────────────────────────────────┐
   ▼                               ▼                                          ▼
revision/core/models/        revision/core/models/                  revision/results/
 classical.py (NEW)           nonadversarial.py (NEW)                 transform_ablation/
  WGAN_MLP   (74p)             VAE  (~562p, ELBO)                      runs/{A,B}/{42..46}/
  WGAN_CNN   (73p)             AR   (p+1 params, lstsq)                  (Phase 09.1 QUANTUM
  WGAN_LSTM  (78p)                  │                                     runs — reused, D-10-04)
  + params_pqc shim                 │                                          │
   │  (count_params==target)        │                                          │
   ▼                                ▼                                          │
revision/run_baselines.py (NEW)  — one (model,pipeline,seed) per invocation     │
   ├─ WGAN path: train_wgan_gp(gen, Critic(), loader, …HPO consts…)  ──┐        │
   ├─ VAE path:  local ELBO loop (Adam, no critic)                     │        │
   └─ AR path:   np.linalg.lstsq fit + recursive simulate              │        │
        writes 5-file bundle + data_hash to                            │        │
        revision/results/baselines/runs/<kind>/<pipe>/<seed>/          │        │
                                  │                                    │        │
revision/run_baselines_sweep.sh (NEW) — 5×2×5=50, xargs -P{1,2},       │        │
   is_complete() 5-file skip, atomic sweep_status.json (mirror 09.1)   │        │
                                  │                                    │        │
                                  ▼                                    ▼        ▼
              06_baseline_comparison.ipynb (NEW) — load all 50 new runs
                + reuse 09.1 quantum runs; reconstruct_od (copied from
                _build_analysis_notebook.py); eval.py metrics; TSTR-lite
                (copied verbatim); recompute+verify data_hash →
                baseline_classical_wgan.json, baseline_nonadversarial.json,
                baseline_comparison.{json,md}
```

### Recommended Project Structure (all NEW unless noted)

```
revision/
├── core/models/
│   ├── classical.py            # NEW — WGAN_MLP/CNN/LSTM + params_pqc shim + count_params
│   ├── nonadversarial.py       # NEW — VAE (ELBO-ready nn.Module) + AR (fit/sample helpers)
│   └── __init__.py             # MODIFY — add classical, nonadversarial to imports/__all__
├── run_baselines.py            # NEW — mirror run_ablation.py; 3 model paths
├── run_baselines_sweep.sh      # NEW — mirror run_ablation_sweep.sh; 50 pairs; .npz-aware is_complete
├── 06_baseline_comparison.ipynb# NEW — comparison table + md + TSTR-lite + data-hash verify
├── _build_baseline_notebook.py # NEW (optional) — deterministic notebook generator (09.1 precedent)
└── results/
    ├── baselines/runs/<model_kind>/<pipeline>/<seed>/{config.yaml,checkpoint.pt|.npz,
    │       samples.npy,metrics.json,inverse_kwargs.npz}     # 50 dirs
    ├── baseline_classical_wgan.json
    ├── baseline_nonadversarial.json
    └── baseline_comparison.{json,md}
```
Note `revision/core/models/__init__.py` currently imports only `quantum, critic` `[VERIFIED: file read]` — it MUST be updated to expose `classical` and `nonadversarial` or `from revision.core.models.classical import ...` will still work (direct submodule import) but the package `__all__` should be kept consistent (matches 09.1 D-style conventions).

### Pattern: `params_pqc` shim for classical WGAN generators

**What:** `train_wgan_gp` optimizes `[generator.params_pqc]` (a single tensor). A classical `nn.Module` has multiple `nn.Parameter`s. The cleanest contract-satisfying pattern that preserves autograd:

```python
class WGAN_MLP(nn.Module):
    num_qubits = 5
    window_length = 10
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5,4), nn.Tanh(), nn.Linear(4,10))
    @property
    def params_pqc(self):
        # Return the LIVE parameter list-as-single-tensor view is NOT possible
        # (params are separate tensors). Instead expose the parameter LIST so
        # torch.optim.Adam([generator.params_pqc]) ... see Pitfall 1 — the
        # supported pattern is a single nn.Parameter OR adapting the optimizer.
        ...
    def forward(self, noise):           # noise: (5, B)
        x = noise.t()                   # (B, 5)
        return self.net(x)              # (B, 10)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
```
See **Pitfall 1** for the resolved, correct approach (a single flat `nn.Parameter` with functional forward, OR the local-adapter alternative).

### Anti-Patterns to Avoid

- **Re-implementing the WGAN-GP loop per model.** D-10-08/13 forbid. Reuse `train_wgan_gp` verbatim; only the generator changes.
- **Adding new metric helpers to `eval.py`.** D-10-20 forbids. Use existing `compute_emd/compute_acf/compute_dtw/compute_moments`.
- **Putting ELBO/AR-fit/aggregation logic in `revision/core/`.** D-10-13 forbids. Loops live in `run_baselines.py`; aggregation in the notebook.
- **`multiprocessing.Pool` for the sweep.** D-10-24 / 09.1 Pitfall 4 forbid. `xargs -P {1,2}` only.
- **Grepping 09.1 configs for `data_hash`.** It does not exist there (see Data-Hash section). Recompute from `load_and_preprocess`.
- **Trusting the LSTM param formula without empirical check.** PyTorch LSTM has two bias vectors; verify with `sum(p.numel() ...)`.
- **A learnable latent-projection layer to reshape `(B,5)` noise for CNN/LSTM.** Adds params, breaks the 71–79 budget. Use parameter-free tile/pad/reshape.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WGAN-GP training loop | Per-model reimplementation | `revision.core.training.train_wgan_gp` | D-10-08/13; subtle Adam-betas/GP/float64-critic/`*0.1` invariants `[VERIFIED: training.py:233-317]` |
| EMD / ACF / DTW / moments | New metric code | `revision.core.eval.*` | D-10-20 forbids new helpers; already v1.0-locked-correct |
| Per-pipeline forward/inverse | New transform code | `revision.core.preprocessing.{forward,inverse}_{minmax_od,logreturns}` | D-10-07; ABL-01 verified ≤1e-8 round-trip |
| OD reconstruction from samples.npy | New inverse glue | Copy `reconstruct_od` from `_build_analysis_notebook.py:95-149` | Identical inverse_kwargs contract; already verified for A/B |
| TSTR-lite LSTM + R² | New forecaster | Copy `TSTRLiteLSTM`/`train_eval_tstr`/`r2_score_inline` from `_build_analysis_notebook.py:432-477` | D-10-21 spec is exactly this; sklearn-free R² already there |
| Sweep status JSON / resume / locking | New orchestration | Mirror `run_ablation_sweep.sh` atomic-writer + flock + `is_complete()` | D-10-23; battle-tested in 09.1's 15-run sweep |
| AR fit | Custom optimizer | `np.linalg.lstsq` (or `statsmodels.AutoReg`) | Closed-form; p+1 reportable params |
| HPO constants | Hardcoded literals | `from revision.core import N_CRITIC, LAMBDA, LR_CRITIC, LR_GENERATOR, BATCH_SIZE, WINDOW_LENGTH, NOISE_LOW, NOISE_HIGH, EVAL_EVERY` | D-10-08 identical-conditions |

**Key insight:** Phase 10's only genuinely new code is ~3 tiny model classes + a VAE/AR loop + a notebook. Everything else is a faithful copy of the 09.1 pattern. The risk is not algorithmic — it is *interface-matching* (the `params_pqc` contract and the sample-space consistency across model families).

## Runtime State Inventory

> Phase 10 is greenfield (new files, new results dir). Reuses immutable existing code/data. Included for completeness; this is NOT a rename/refactor phase.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — writes to fresh `revision/results/baselines/`. Reuses (read-only) Phase 09.1 quantum runs at `revision/results/transform_ablation/runs/{A,B}/{42..46}/` (D-10-04). | None — read-only reuse |
| Live service config | None — no DB, no n8n, no scheduler | None — verified by absence of any service config in `revision/` |
| OS-registered state | None — no daemons/cron/launchd; sweep is a foreground/tmux bash script | None |
| Secrets/env vars | None — no API keys, no auth in this phase | None |
| Build artifacts | None new — no `pyproject.toml`/package-name change. `revision/core/models/__init__.py` is *edited* (add 2 submodule imports) but that is a source edit, not a stale build artifact. | None — no reinstall needed (pure-source package, run via `./qgan_env/bin/python -m revision...`) |

**Verified by:** grep over `revision/` for service/secret/scheduler markers; inspection of `run_ablation_sweep.sh` (pure bash + tmux/nohup, no OS registration).

## Common Pitfalls

### Pitfall 1: `train_wgan_gp` optimizes `generator.params_pqc`, not `generator.parameters()`

**What goes wrong:** A classical `nn.Module` has its weights in multiple `nn.Parameter`s. `train_wgan_gp:234` does `torch.optim.Adam([generator.params_pqc], ...)`. If `params_pqc` is a `@property` that returns a *fresh concatenated/flattened copy* of the weights, gradients computed in `forward` flow into the original `nn.Parameter`s, NOT the copy the optimizer holds → **the optimizer steps a detached tensor and the generator never learns** (silent failure: loss curves look plausible early then plateau; samples stay near init).

**Why it happens:** The loop was extracted verbatim from the quantum code where `params_pqc` is literally the *one and only* `nn.Parameter` (a `(75,)` tensor) `[VERIFIED: quantum.py:67-70, count_params==75, parameters()→75 numel]`. The contract is "one trainable tensor."

**How to avoid (recommended):** Make each classical generator hold its trainable weights as a **single flat `nn.Parameter`** named `params_pqc`, and implement `forward` *functionally* by slicing that flat vector into the layer weight/bias tensors (à la `torch.nn.functional.linear(x, W, b)` with `W,b` carved from `params_pqc`). This exactly mirrors the quantum generator's "one parameter vector, functional circuit" design and makes `count_params()==params_pqc.numel()` trivially the matched count. The arithmetic recipes in §Standard Stack already give the exact numel (74/73/78); lay them out as one contiguous vector.
**Alternative (also valid, planner's discretion per CD):** keep a normal `nn.Sequential` and, in `run_baselines.py` (NOT `core/` — D-10-13), build the generator optimizer over `generator.parameters()` *instead of* calling `train_wgan_gp`'s internal optimizer — but this requires NOT using `train_wgan_gp` for the optimizer step, which contradicts "reuse the loop verbatim." **Therefore the single-flat-`nn.Parameter` functional design is strongly recommended** — it is the only approach that keeps `train_wgan_gp` truly unchanged.
**Warning signs:** generator loss flat; `samples.npy` variance ≈ init variance; `params_pqc` unchanged after training (`torch.allclose(before, after)` → True).

### Pitfall 2: PyTorch LSTM parameter count miscalculated

**What goes wrong:** Hand-computed LSTM param count omits the second bias vector (`b_hh`), so the planned `hidden_size` lands outside 71–79 and the BASE-01 acceptance test fails.
**Why:** PyTorch `nn.LSTM` has BOTH `bias_ih` and `bias_hh` (4H each) → total bias = 8H per layer, not 4H. Full: `4*(I*H + H*H) + 4*H + 4*H`.
**How to avoid:** Always compute `sum(p.numel() for p in lstm.parameters())` empirically in the acceptance test; never trust the formula. The §Standard-Stack LSTM recipe (I=2,H=2 → 48 LSTM params + 30 Linear = 78) was derived with the two-bias formula — re-verify in code before locking the acceptance criterion.
**Warning signs:** `count_params()` off by a multiple of `hidden_size` from the target.

### Pitfall 3: Sample-space inconsistency across model families breaks the apples-to-apples table

**What goes wrong:** WGAN classical generators go through `train_wgan_gp`, whose generation path casts to float64 and multiplies by `0.1` (`training.py:283/316`; `run_ablation.py:205` replicates it for saved samples). VAE/AR do NOT go through that loop. If VAE/AR `samples.npy` is in a different magnitude/space than the WGAN/quantum samples, the shared `reconstruct_od` (which maps `(samples+1)/2` for A, etc.) reconstructs them onto a different OD scale → the comparison table compares incomparable numbers and BASE-03 is invalid.
**Why:** The `*0.1` is a quantum-output-magnitude artifact (PauliX/PauliZ expectations are in [-1,1]; ×0.1 shrinks them), not part of the preprocessing inverse. It is baked into the WGAN path but meaningless for VAE/AR.
**How to avoid:** Define ONE canonical "saved-sample space" = the `[-1,1]` window space the 09.1 `reconstruct_od` consumes, and make every model emit `samples.npy` in that space. For WGAN models the loop already produces it (inherit unchanged). For VAE/AR, emit decoder/simulated outputs directly in `[-1,1]` window space and run them through the **identical** `reconstruct_od` + `eval.py` path. Add a Wave-2 smoke gate: reconstruct one sample from each of {wgan_mlp, vae, ar} and assert all land within `[real_OD.min()*0.5, real_OD.max()*1.5]`. Document the `*0.1` asymmetry explicitly in each model's `train_protocol_notes` (D-10-16).
**Warning signs:** VAE/AR OD-EMD orders of magnitude off from WGAN/quantum; reconstructed VAE OD values clustered far outside real OD range.

### Pitfall 4: Data-hash cross-check against non-existent 09.1 field

**What goes wrong:** Executor tries to verify D-10-15 by reading `data_hash` from Phase 09.1 quantum `config.yaml` files and finds no such key (it was never written).
**Why:** D-10-15's hash is NEW to Phase 10. `[VERIFIED: grep -rl 'data_hash\|sha256\|tobytes' revision/ → empty]`.
**How to avoid:** Recompute `sha256(load_and_preprocess(csv).["OD"].numpy().tobytes())[:16]` once; assert all 50 new configs equal it; establish quantum equivalence *by construction* (same `load_and_preprocess`, same `data.csv` `[VERIFIED: 09.1 config csv_path==data.csv]`) and document it. Phrase the BASE-03 criterion as "all new runs share one hash, computed from the same data entry point the 09.1 quantum runs used."
**Warning signs:** Executor blocked looking for `data_hash` in 09.1 outputs; spurious "hash mismatch — 09.1 has no hash" error.

### Pitfall 5: `multiprocessing.Pool` corrupts per-seed RNG (inherited from 09.1 Pitfall 4 — RESTATED, LOCKED by D-10-24)

**What goes wrong:** Forked Pool workers share the parent's warm numpy global RNG → seeds 42 and 43 produce identical noise streams → 5-seed mean±std is fake.
**Why:** `os.fork()` copies numpy global state; `train_wgan_gp:212` reseeds but a Pool that pre-imports/pre-runs corrupts ordering.
**How to avoid:** `xargs -P {1,2}` OS-process parallelism only (each invocation = fresh interpreter + fresh RNG; `run_baselines.py` reseeds via `train_wgan_gp(seed=N)` / `np.random.default_rng(seed)`). Mirror `run_ablation_sweep.sh` exactly. NEVER introduce a Python `Pool` in `run_baselines.py` or the sweep.
**Warning signs:** two seeds → bit-identical loss curves / identical `samples.npy`.

### Pitfall 6: VAE posterior collapse on tiny length-10 windows

**What goes wrong:** With a 4-dim latent and strong decoder, KL term drives `mu→0, logvar→0`, decoder ignores `z` → all samples ≈ the data mean (mode collapse), making VAE look artificially bad/good vs WGAN.
**Why:** Classic VAE pathology, exacerbated by small data + small latent.
**How to avoid:** Use `beta=1.0` standard ELBO first; if collapse observed in the Wave-2 smoke (sample variance ≪ real variance), apply a short KL warm-up (linear `beta: 0→1` over first ~20% of epochs) and document it in `train_protocol_notes`. Keep it minimal — D-10-03 says "smallest that trains *stably*", so a documented KL-warmup is in-scope; a bigger network is not the first lever.
**Warning signs:** VAE `samples.npy` std ≈ 0; reconstructed OD nearly constant.

## Code Examples

### `train_wgan_gp` invocation for a classical WGAN generator (mirror run_ablation.py:302-318)

```python
# Source: revision/run_ablation.py:288-318 (verified pattern), generator swapped
from revision.core import (N_CRITIC, LAMBDA, LR_CRITIC, LR_GENERATOR,
                           EVAL_EVERY, WINDOW_LENGTH)
from revision.core.models.critic import Critic
from revision.core.models.classical import WGAN_MLP   # NEW
from revision.core.training import train_wgan_gp

torch.manual_seed(seed)
generator = WGAN_MLP()                # exposes .params_pqc (single flat nn.Parameter),
                                      # .num_qubits=5, .window_length=10, .count_params()==74
critic = Critic(window_length=WINDOW_LENGTH)   # SAME critic, D-10-08
metrics = train_wgan_gp(generator, critic, bundle.dataloader,
                        num_epochs=epochs, n_critic=N_CRITIC, lambda_gp=LAMBDA,
                        lr_critic=LR_CRITIC, lr_generator=LR_GENERATOR,
                        seed=seed, eval_every=EVAL_EVERY)   # loop UNCHANGED
```

### VAE training loop (lives in run_baselines.py, NOT core/ — D-10-13)

```python
# Source: standard ELBO; per-protocol notes documented in JSON (D-10-09)
vae = VAE(in_dim=10, hidden=16, latent=4)              # ~562 params (reported)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for epoch in range(epochs):
    for (x,) in dataloader:                            # x: (B,10) in [-1,1]
        x_hat, mu, logvar = vae(x)
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = min(1.0, epoch / max(1, int(0.2*epochs)))  # KL warm-up (Pitfall 6)
        (recon + beta*kld).backward(); opt.step(); opt.zero_grad()
# sampling -> samples.npy in [-1,1] window space (Pitfall 3)
z = torch.randn(n_synth, 4); samples = vae.decode(z).detach().numpy()  # (n_synth,10)
```

### AR fit + simulate (run_baselines.py)

```python
# Source: closed-form least-squares; checkpoint -> .npz (D-10-14)
p = 2
X = np.stack([x[p-1-k:-1-k] for k in range(p)], axis=1); y = x[p:]
phi, *_ = np.linalg.lstsq(X, y, rcond=None)
sigma2 = float((y - X @ phi).var(ddof=0))              # params = p + 1 = 3
rng = np.random.default_rng(seed)
W = np.empty((n_synth, 10))
for i in range(n_synth):
    w = list(rng.normal(0, np.sqrt(sigma2), size=p))   # burn-in init
    for t in range(p, 10):
        w.append(float(np.dot(phi, w[-p:][::-1]) + rng.normal(0, np.sqrt(sigma2))))
    W[i] = w
np.savez(run_dir/"checkpoint.npz", phi=phi, sigma2=sigma2, p=p)
np.save(run_dir/"samples.npy", W)                      # (n_synth,10), [-1,1]-ish space
```

### Data-hash (D-10-15) — written into every config.yaml

```python
import hashlib
raw = load_and_preprocess(str(csv_path))               # SAME entry point as 09.1 quantum
data_hash = hashlib.sha256(raw["OD"].cpu().numpy().tobytes()).hexdigest()[:16]
config["data_hash"] = data_hash                         # NEW field; 09.1 has none
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 09.1 RESEARCH listed pennylane 0.44.0 | Env has **pennylane 0.43.0** | env as of 2026-05-17 | Quantum reference is only *loaded* (not retrained, D-10-04); version drift is informational, not blocking — re-verify if any quantum re-instantiation is needed |
| 09.1 RESEARCH listed torch 2.10.0 / statsmodels 0.14.6 | Env has **torch 2.9.0 / statsmodels 0.14.5** | env as of 2026-05-17 | Minor; all APIs used (Linear/Conv1d/LSTM/lstsq/acf) are stable across these versions |
| sklearn assumed "likely installed" in 09.1 | **sklearn confirmed MISSING** | 2026-05-17 import check | TSTR-lite R² MUST use inline `r2_score_inline` (already the 09.1 pattern) — do not add sklearn |
| `np.random.seed` global | `np.random.default_rng(seed)` per-call | locked in 09.1 D-09.1-18 | new code uses `default_rng`; `train_wgan_gp:212` global reseed is the one sanctioned exception |

**Deprecated/outdated:** none new for this phase. The 09.1 RESEARCH version table is stale by minor versions (see above) — Phase 10 RESEARCH versions supersede it for this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The single-flat-`nn.Parameter` functional design is the cleanest way to satisfy `train_wgan_gp`'s `params_pqc` contract while keeping the loop unchanged | Pitfall 1 | If executor prefers the local-adapter route, that contradicts "reuse loop verbatim" — planner must explicitly choose; both are CD. Low risk: arithmetic recipes are design-agnostic. |
| A2 | VAE ~562-param "single 16-unit hidden layer each side" is the "smallest deep VAE that trains stably" on length-10 windows | VAE Sizing | If it collapses (Pitfall 6) the documented KL-warmup is the first lever; if still unstable, hidden=24 (still small). Count is disclosed, not constrained (D-10-03), so exact number is informational. |
| A3 | AR(2) (p=2 → 3 params) is a defensible minimal order for length-10 windows | AR Baseline | Planner may instead select p by AIC over {1,2,3}; either is defensible and disclosed. Low risk — D-10-03 only requires "parameter-minimal + transparent". |
| A4 | VAE/AR samples emitted in `[-1,1]` window space (no `*0.1`) reconstruct comparably to WGAN samples via the shared `reconstruct_od` | Pitfall 3 | HIGH-IMPACT if wrong → invalid BASE-03 table. Mitigation: mandatory Wave-2 smoke reconstruction gate across all 3 families. Surfaced as the phase's top risk. |
| A5 | The 09.1 quantum runs at `runs/{A,B}/{seed}/` are present, complete, and reconstructable with the copied `reconstruct_od` | Comparison Table | If 09.1 outputs were not committed/retained, the quantum column cannot be built — verify the 50→ + 10 quantum dirs exist as a Wave-0/Wave-1 precondition. (09.1-04-SUMMARY says ABL-02 GREEN, full sweep complete — likely present.) |

## Open Questions

1. **`params_pqc` contract resolution (Pitfall 1 / A1).**
   - Known: `train_wgan_gp:234` optimizes `[generator.params_pqc]`; D-10-08/13 demand the loop stay unchanged and live outside `core/`.
   - Unclear: single-flat-`nn.Parameter`+functional-forward vs a local optimizer-adapter in `run_baselines.py`.
   - Recommendation: single-flat-`nn.Parameter` functional design (only approach keeping the loop truly verbatim). Planner should lock this in the architecture plan; acceptance test asserts `params_pqc.numel() == count_params() == <74|73|78>` and `torch.allclose(params_pqc_before, params_pqc_after) is False` post-smoke.

2. **VAE/AR sample-space canonicalization (Pitfall 3 / A4).**
   - Known: WGAN path applies `*0.1`; VAE/AR do not pass through it.
   - Unclear: whether emitting VAE/AR in plain `[-1,1]` reconstructs onto the same OD scale as WGAN/quantum.
   - Recommendation: define one canonical saved-sample space; add a mandatory cross-family reconstruction smoke gate in Wave 2; document the `*0.1` asymmetry in every `train_protocol_notes`. This is the planner's must-include verification step.

3. **Existence/retention of Phase 09.1 quantum runs (A5).**
   - Known: 09.1-04-SUMMARY reports ABL-02 GREEN, full 3×5×1000 sweep complete.
   - Unclear: whether all `runs/{A,B}/{42..46}/` dirs (10 quantum runs) are retained on disk in this checkout.
   - Recommendation: Wave-1 precondition check — assert the 10 quantum run dirs exist with their 5-file bundles before the comparison plan; if missing, the phase is BLOCKED on re-running the 09.1 quantum sweep (out of D-10-04's "reuse as-is" assumption).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch | all models + training | ✓ | 2.9.0 | — |
| numpy | AR fit, data-hash, arrays | ✓ | 2.3.4 | — |
| scipy | EMD (via eval.py) | ✓ | 1.16.2 | — |
| statsmodels | ACF (via eval.py); optional AR | ✓ | 0.14.5 | hand-rolled lstsq AR (recommended anyway) |
| fastdtw | DTW (via eval.py) | ✓ | OK | — |
| PyYAML | config.yaml | ✓ | 6.0.3 | — |
| pennylane | load quantum reference (no retrain) | ✓ | 0.43.0 | — |
| sklearn | (would-be) TSTR R² | ✗ | — | **inline `r2_score_inline` (already the 09.1 pattern; mandatory)** |
| xargs -P | sweep parallelism | ✓ (macOS ships xargs; used by 09.1 sweep) | — | `--parallel 1` sequential bash loop |
| flock | atomic sweep_status.json | ✓ (used by run_ablation_sweep.sh) | — | sequential mode avoids contention |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `sklearn` → inline R² (no behavior change; established pattern).

`[VERIFIED: ./qgan_env/bin/python import checks 2026-05-17]`

## Sources

### Primary (HIGH confidence — direct codebase verification, this session)
- `revision/core/training.py:228-317, 234, 406-429` — `train_wgan_gp` generator contract (`params_pqc`, `(num_qubits,B)`→`(B,10)`, `*0.1`, `_ESAdapter`)
- `revision/core/models/quantum.py:31-202` — quantum generator; `count_params()==75`; single `params_pqc` `(75,)` `nn.Parameter`
- `revision/core/models/critic.py:19-77` — the shared `Critic` (unchanged for all WGAN variants, D-10-08)
- `revision/core/__init__.py:11-45` — locked HPO constants
- `revision/core/preprocessing.py:29-103` — A & B forward/inverse (C dropped per D-10-05)
- `revision/core/eval.py:25-163` — EMD/ACF/DTW/moments (no new helpers, D-10-20)
- `revision/core/data.py:227-296` — `load_and_preprocess` (data-hash source; `OD` float32 (778,))
- `revision/run_ablation.py:94-340` — per-(pipeline,seed) driver template; 5-file bundle; `generate_samples` `*0.1`
- `revision/run_ablation_sweep.sh:1-456` — sweep template; `is_complete()` 5-file check; atomic status; xargs -P; Pitfall-4 note
- `revision/_build_analysis_notebook.py:95-149, 432-521` — `reconstruct_od` + TSTR-lite (`TSTRLiteLSTM`, `r2_score_inline`, `train_eval_tstr`, HELD_OUT_N=320, init seeds 40/41/42) — reusable verbatim
- `revision/results/transform_ablation/tstr_lite.json` — confirms TSTR-lite output schema
- `revision/results/transform_ablation/runs/B/42/config.yaml` — confirms 09.1 config schema and **absence of data_hash**; `csv_path: data.csv`
- `.planning/phases/.../09.1-04-SUMMARY.md` — Pipeline B headline, ABL-02 GREEN (quantum runs complete), TSTR-lite spec, Pitfall-4
- `.planning/phases/.../09.1-RESEARCH.md` — Pitfall 4 lineage, sweep timing, reusable patterns
- `./qgan_env` import checks (2026-05-17) — torch 2.9.0, numpy 2.3.4, pennylane 0.43.0, scipy 1.16.2, statsmodels 0.14.5, pyyaml 6.0.3, fastdtw OK, **sklearn MISSING**; `QuantumGenerator().count_params()==75`

### Secondary (MEDIUM — established conventions)
- PyTorch `nn.LSTM` parameter layout (two bias vectors `bias_ih`+`bias_hh`) — standard PyTorch behavior; flagged for empirical re-verification (Pitfall 2)
- Standard VAE ELBO + KL-warmup for posterior-collapse mitigation — well-established VAE practice (Pitfall 6)

### Tertiary (LOW)
- None relied upon.

## Metadata

**Confidence breakdown:**
- Standard stack / param arithmetic: HIGH — formulas derived + `count_params()==75` verified; LSTM count flagged for empirical re-check (Pitfall 2)
- `train_wgan_gp` contract: HIGH — verbatim file:line verification of all 5 interface points
- Artifact/sweep/TSTR reuse: HIGH — direct reuse of verified 09.1 code; templates read in full
- Data-hash: HIGH — absence in 09.1 verified by grep; formula is unambiguous
- Sample-space comparability (Pitfall 3): MEDIUM — the risk is real and identified; the exact reconstruction equivalence must be proven by the mandated Wave-2 smoke gate, not assumed
- VAE/AR sizing: MEDIUM — defensible minimal recipes; counts are disclosed-not-constrained (D-10-03) so exact numbers are informational

**Research date:** 2026-05-17
**Valid until:** 2026-06-17 (30 days — claims verified against immutable codebase + a stable installed env; re-verify env versions only if a quantum re-instantiation becomes necessary)

> Validation Architecture section intentionally omitted: `.planning/config.json` has `workflow.nyquist_validation: false`.
