# Phase 10: Classical Baselines - Pattern Map

**Mapped:** 2026-05-17
**Files analyzed:** 9 (3 new modules/scripts + 4 result artifacts + 1 modified `__init__.py` + 1 new analysis notebook)
**Analogs found:** 9 / 9 (every new file has a direct in-tree analog — Phase 10 is a faithful copy of the 09.1 pattern)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `revision/core/models/classical.py` | model | transform (latent → window) | `revision/core/models/quantum.py` | role-match (nn.Module generator; classical vs quantum internals) |
| `revision/core/models/nonadversarial.py` | model | transform (VAE) / batch-fit (AR) | `revision/core/models/quantum.py` (interface/count_params contract only) | partial (interface contract; no in-tree VAE/AR training analog) |
| `revision/core/models/__init__.py` | config | n/a (barrel import) | existing `revision/core/models/__init__.py` (3-line pattern) | exact |
| `revision/run_baselines.py` | route/driver (CLI) | request-response (one run per invocation) | `revision/run_ablation.py` | exact |
| `revision/run_baselines_sweep.sh` | route/driver (sweep) | batch (50-pair fan-out) | `revision/run_ablation_sweep.sh` | exact |
| `revision/06_baseline_comparison.ipynb` | utility (aggregation) | batch (load runs → aggregate → render) | `revision/_build_analysis_notebook.py` (reconstruct_od + TSTR-lite cells) | role-match |
| `revision/results/baseline_classical_wgan.json` | artifact | n/a (output) | 09.1 `metrics.csv` long-form schema (via notebook) | role-match |
| `revision/results/baseline_nonadversarial.json` | artifact | n/a (output) | same long-form schema | role-match |
| `revision/results/baseline_comparison.{json,md}` | artifact | n/a (output) | 09.1 `tstr_lite.json` schema + notebook md render | role-match |

## Pattern Assignments

### `revision/core/models/classical.py` (model, transform)

**Analog:** `revision/core/models/quantum.py`

The classical generators must satisfy the *exact same `train_wgan_gp` interface contract* the `QuantumGenerator` satisfies. Copy the interface surface from `quantum.py`, replace the PQC internals with `torch.nn` layers.

**Interface contract to copy (the four attributes `train_wgan_gp` reads):**

`quantum.py` lines 53-70 — `num_qubits`/`window_length`/`params_pqc` declaration:
```python
self.num_qubits = num_qubits          # train_wgan_gp:228 reads getattr(generator,"num_qubits")
self.num_layers = num_layers
self.window_length = window_length    # train_wgan_gp:229 reads getattr(generator,"window_length")
self.params_pqc = nn.Parameter(       # train_wgan_gp:234 builds Adam([generator.params_pqc])
    torch.randn(self.num_params, dtype=torch.float32) * _INIT_SCALE,
    requires_grad=True,
)
```

`quantum.py` lines 80-85 — `count_params()` contract (D-10-11: classical `count_params()` must match this signature; for QuantumGenerator(5,4) it returns 75 — the matched-param target):
```python
def count_params(self) -> int:
    """Return total PQC parameter count.
    For (num_qubits=5, num_layers=4): 5 + 4*15 + 10 = 75.
    """
    return self.num_params
```

`quantum.py` lines 173-202 — `forward(noise) -> (batch, window_length)` contract. The loop passes `(num_qubits, batch)` noise and expects `(batch, window_length)` back (note the `stacked.T` transpose at line 199 that produces `(batch, window_length)`):
```python
def forward(self, noise_params, par_light=None):
    results = self.qnode(noise_params, self.params_pqc)
    stacked = torch.stack(list(results))
    if stacked.dim() == 2:
        stacked = stacked.T          # (window_length, batch) -> (batch, window_length)
    return stacked
```

**The `params_pqc` shim — CRITICAL (RESEARCH Pitfall 1, lines 456-459):**
`train_wgan_gp:234` builds `torch.optim.Adam([generator.params_pqc], ...)` over a *single* tensor, not `generator.parameters()`. A naive `@property` returning a flattened copy detaches gradients (silent non-learning). Each classical generator must expose `params_pqc` as the single live `nn.Parameter` the optimizer steps, with a functional `forward` that uses it (RESEARCH §"Pattern: params_pqc shim" lines 389-413; resolution in Pitfall 1 lines 456+). Set `num_qubits = 5`, `window_length = 10` as class attributes (RESEARCH §contract item 3, lines 140-145). Param-arithmetic recipes: MLP h=4 → 74 params; CNN C=9 → 73; LSTM(2,2)+Linear(2,10) → 78 (RESEARCH §"Parameter Arithmetic Recipes" lines 156-215).

**Acceptance anchor:** assert `module.count_params() == sum(p.numel() for p in module.parameters())` AND `71 <= count_params() <= 79` empirically (RESEARCH lines 213, Pitfall: LSTM bias count trap).

---

### `revision/core/models/nonadversarial.py` (model, transform / batch-fit)

**Analog:** `revision/core/models/quantum.py` (for `nn.Module` shape + `count_params()` convention only — no in-tree VAE/AR training analog exists; see "No Analog Found")

**Copy from quantum.py:** the `nn.Module` skeleton + `count_params()` (lines 80-85) signature so `VAE.count_params()` reports its ~562 params for the comparison-table `models[]` array (D-10-03/16). AR is not an `nn.Module` — it is `{phi, sigma2, p}` fit/sample helpers; its "param count" is `p+1` reported as a plain int.

**Architecture spec (from RESEARCH, not from a code analog):**
- VAE: encoder `Linear(10,16)→ReLU→[Linear(16,4) mu, Linear(16,4) logvar]`, decoder `Linear(4,16)→ReLU→Linear(16,10)`, ~562 params (RESEARCH §"VAE Sizing" lines 217-229). ELBO loss lives in `run_baselines.py`, NOT here (D-10-13).
- AR: fit/sample helpers; least-squares `np.linalg.lstsq` design-matrix recipe (RESEARCH §"AR Baseline" lines 241-267). Fit/sample orchestration lives in `run_baselines.py` (D-10-13); only the model definition lives here.

**Sample-space asymmetry (RESEARCH Pitfall 3, lines 237-239):** VAE/AR do NOT go through `train_wgan_gp` so they do NOT inherit the `*0.1` scaling at `training.py:283`. Emit samples in `[-1,1]` window space so the shared `reconstruct_od` inverse consumes them identically. Flag as a Wave-2 smoke gate.

---

### `revision/core/models/__init__.py` (config, barrel)

**Analog:** the existing 3-line file (exact pattern):
```python
"""revision.core.models — PQC generator and classical critic."""
from revision.core.models import quantum, critic  # noqa: F401
__all__ = ["quantum", "critic"]
```
**Modify to:** add `classical, nonadversarial` to the import line and `__all__` (RESEARCH line 387).

---

### `revision/run_baselines.py` (route/driver, request-response)

**Analog:** `revision/run_ablation.py` — copy structure verbatim, branch on `--model`.

**Imports + HPO-constant import block** (`run_ablation.py` lines 32-65) — copy verbatim; D-10-08 identical-conditions requires the same `from revision.core import BATCH_SIZE, EVAL_EVERY, LAMBDA, LR_CRITIC, LR_GENERATOR, N_CRITIC, NOISE_HIGH, NOISE_LOW, NUM_LAYERS, NUM_QUBITS, WINDOW_LENGTH` block and `from revision.core.models.critic import Critic` (same critic for every WGAN-GP variant per D-10-08).

**`build_dataset_for_pipeline`** (`run_ablation.py` lines 94-177) — copy A and B branches verbatim (D-10-07 same windowed data + inverse_kwargs contract); DELETE the C branch (D-10-05 Pipeline C dropped). Keep the identical `DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)` call site (lines 169-171).

**`generate_samples`** (`run_ablation.py` lines 180-209) — copy verbatim for the WGAN path:
```python
noise = torch.tensor(rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, bs)), dtype=torch.float32)
out = generator(noise).to(torch.float64) * 0.1
```
The `* 0.1` is mandatory for WGAN variants (mirrors `training.py:283`). For VAE/AR, write a *separate* sampler that does NOT apply `*0.1` (RESEARCH Pitfall 3).

**`_save_inverse_kwargs`** (`run_ablation.py` lines 212-225) — copy verbatim.

**`main()` + 5-file artifact bundle** (`run_ablation.py` lines 228-336) — copy the argparse + run-dir + config.yaml + checkpoint + metrics.json write sequence. Adaptations:
- argparse: replace `--pipeline {A,B,C}` choices with `{A,B}`; add `--model {wgan_mlp,wgan_cnn,wgan_lstm,vae,ar}` (D-10-22).
- run-dir: `args.out_root / "runs" / args.model / args.pipeline / str(args.seed)` (D-10-14, vs 09.1's `runs/<pipeline>/<seed>/`).
- config dict (lines 266-284): add `model_kind`, `data_hash`, `parameter_count`, `family`; null WGAN-only fields for VAE/AR (RESEARCH §"Run-Directory + Artifact Contract" lines 269-281).
- `data_hash` (NEW, no 09.1 precedent — RESEARCH lines 283-290): `hashlib.sha256(load_and_preprocess(str(csv_path))["OD"].cpu().numpy().tobytes()).hexdigest()[:16]`.
- WGAN branch: `train_wgan_gp(generator, Critic(window_length=WINDOW_LENGTH), bundle.dataloader, num_epochs=..., n_critic=N_CRITIC, lambda_gp=LAMBDA, lr_critic=LR_CRITIC, lr_generator=LR_GENERATOR, seed=..., eval_every=EVAL_EVERY)` — copied verbatim from `run_ablation.py` lines 302-313 (D-10-08).
- VAE branch: local ELBO loop (Adam over `vae.parameters()`, no critic, no n_critic) — RESEARCH lines 231-237.
- AR branch: `np.linalg.lstsq` fit + recursive simulate; checkpoint is `.npz` `{phi,sigma2,p}` (D-10-14, RESEARCH lines 247-267).
- `torch.manual_seed(args.seed)` before model construction (line 291) — copy verbatim.

---

### `revision/run_baselines_sweep.sh` (route/driver, batch)

**Analog:** `revision/run_ablation_sweep.sh` — copy verbatim, change the worklist dimensions and `is_complete()` to be `.npz`-aware.

**`set -euo pipefail` + python-interpreter detection** (`run_ablation_sweep.sh` lines 77-102) — copy verbatim (`./qgan_env/bin/python` preference).

**`--parallel` 1|2 guardrail** (lines 140-156) — copy verbatim (D-10-24 / Pitfall 4: xargs -P 2, never multiprocessing.Pool).

**`is_complete()`** (lines 164-172) — copy and make `.npz`-aware:
```bash
is_complete() {
  local m="$1" p="$2" s="$3"
  local d="${OUT_ROOT}/runs/${m}/${p}/${s}"
  local ckpt="checkpoint.pt"; [[ "$m" == "ar" ]] && ckpt="checkpoint.npz"
  [[ -s "${d}/config.yaml" && -s "${d}/${ckpt}" && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" && -s "${d}/inverse_kwargs.npz" ]]
}
```
(09.1 keyed only on `(p,s)` with fixed `checkpoint.pt`; Phase 10 keys on `(model,p,s)` with the conditional `.npz` — D-10-14, RESEARCH line 281.)

**`update_status()` atomic writer** (lines 187-266) — copy verbatim including the `flock -x 9` advisory lock + `tempfile.mkstemp` + `os.rename` atomic write. Change `total_count` 15 → 50 and add `model` to the per-run record key (replace `r["pipeline"]==p and r["seed"]==s` with a 3-tuple match on model/pipeline/seed).

**`run_one()`** (lines 274-318) — copy verbatim; change the inner call to `"$PYTHON" -m revision.run_baselines --model "$m" --pipeline "$p" --seed "$s" --epochs "$EPOCHS"`.

**Main dispatch + xargs -P 2 worklist** (lines 369-395) — copy verbatim; expand the nested loop to `MODELS="wgan_mlp wgan_cnn wgan_lstm vae ar"` × `PIPELINES="A B"` × `SEEDS="42 43 44 45 46"` = 50 lines; the `xargs -P 2 -L 1 bash -c 'run_one "$0" "$1" "$2"'` invocation gets a third positional.

**Constants block** (lines 82-87) — adapt: `MODELS`, `PIPELINES="A B"` (no C), `SEEDS="42 43 44 45 46"`, `OUT_ROOT="revision/results/baselines"`.

---

### `revision/06_baseline_comparison.ipynb` (utility, batch aggregation)

**Analog:** `revision/_build_analysis_notebook.py` (the deterministic notebook-generator pattern — RESEARCH line 379 suggests an optional `_build_baseline_notebook.py`)

**Copy `reconstruct_od` VERBATIM** from `_build_analysis_notebook.py` lines 95-149 — keep the A and B branches (lines 105-127) exactly (identical inverse_kwargs contract); drop the C branch (lines 129-147). Re-point `base = Path(...)` from `runs/{pipeline}/{seed}` to `runs/{model}/{pipeline}/{seed}` for new runs, but keep the original 09.1 path `revision/results/transform_ablation/runs/{pipeline}/{seed}` for the reused quantum rows (D-10-04/18).

**Copy TSTR-lite VERBATIM** from `_build_analysis_notebook.py` lines 432-477 — `TSTRLiteLSTM` (lines 432-440), `r2_score_inline` (lines 442-445; sklearn-free, sklearn not installed), `train_eval_tstr` (lines 447-477). 3 init seeds {40,41,42}, `HELD_OUT_N = 320` (line 483), eval on `real_windowed_OD[:320]`, train on `[320:]` (D-10-21, RESEARCH lines 317-328). Do NOT promote to `revision/core/` (D-10-13).

**Outputs:** long-form `rows` schema mirroring 09.1 `metrics.csv` plus `model_kind`; top-level `models[]` array; markdown render (RESEARCH §"Comparison Table Schema" lines 292-315). Recompute `data_hash` once and assert all 50 new `config.yaml` hashes equal it; quantum equivalence established by construction, NOT by grepping 09.1 configs (RESEARCH lines 283-290, anti-pattern line 421).

---

## Shared Patterns

### HPO Constants (identical-conditions invariant, D-10-08)
**Source:** `revision/run_ablation.py` lines 45-57
**Apply to:** `run_baselines.py` (WGAN path) and the WGAN-GP `config.yaml` fields
```python
from revision.core import (
    BATCH_SIZE, EVAL_EVERY, LAMBDA, LR_CRITIC, LR_GENERATOR,
    N_CRITIC, NOISE_HIGH, NOISE_LOW, NUM_LAYERS, NUM_QUBITS, WINDOW_LENGTH,
)
```
Never hardcode literals (RESEARCH "Don't Hand-Roll" line 436).

### `train_wgan_gp` Generator Contract
**Source:** `revision/core/training.py` lines 228-234, 282-283, 315-316
**Apply to:** all 3 classical WGAN generators in `classical.py`
```python
num_qubits = getattr(generator, "num_qubits", NUM_QUBITS)      # :228
window_length = getattr(generator, "window_length", WINDOW_LENGTH)  # :229
g_opt = torch.optim.Adam([generator.params_pqc], lr=lr_generator, betas=(0.0, 0.9))  # :234
generated_samples = generator(noise_batch)        # :282 expects (batch, window_length)
generated_samples = generated_samples.to(torch.float64) * 0.1   # :283 the *0.1 scaling
```
The classical generator MUST expose `num_qubits=5`, `window_length=10`, a single live `params_pqc` `nn.Parameter`, and `forward((5,B)) -> (B,10)`.

### 5-File Artifact Bundle
**Source:** `revision/run_ablation.py` lines 285-330 (writes) + `run_ablation_sweep.sh` lines 164-172 (`is_complete`)
**Apply to:** `run_baselines.py` (every model path) and `run_baselines_sweep.sh`
Bundle: `config.yaml, checkpoint.pt|.npz, samples.npy, metrics.json, inverse_kwargs.npz`. WGAN/VAE → `.pt`; AR → `.npz` (D-10-14).

### Atomic Sweep-Status Writer
**Source:** `revision/run_ablation_sweep.sh` lines 187-266 (`update_status` + `flock -x 9` + `tempfile.mkstemp`/`os.rename`)
**Apply to:** `run_baselines_sweep.sh` — copy verbatim, change `total_count` to 50 and key records on `(model, pipeline, seed)`.

### sklearn-free R²
**Source:** `revision/_build_analysis_notebook.py` lines 442-445 (`r2_score_inline`)
**Apply to:** TSTR-lite in `06_baseline_comparison.ipynb`. sklearn is NOT installed (RESEARCH line 101); use this inline form, do not add a dependency.

### Sample-Space Consistency (cross-model comparability — highest risk)
**Source:** `revision/run_ablation.py` line 205 (`* 0.1`) vs RESEARCH Pitfall 3 (lines 237-239)
**Apply to:** WGAN samplers (replicate `*0.1`); VAE/AR samplers (do NOT replicate `*0.1`). Wave-2 smoke gate: reconstruct one VAE sample and one WGAN sample through the identical pipeline inverse; both must land in real OD range.

## No Analog Found

| File / Concern | Role | Data Flow | Reason | Planner Action |
|----------------|------|-----------|--------|----------------|
| `nonadversarial.py` VAE ELBO training | model+loop | transform | No non-WGAN training loop exists anywhere in `revision/` | Use RESEARCH §"VAE Sizing" lines 217-239 spec; loop in `run_baselines.py` |
| `nonadversarial.py` AR fit/sample | batch-fit | batch | No autoregressive / `np.linalg.lstsq` precedent in tree | Use RESEARCH §"AR Baseline" lines 241-267 recipe |
| `data_hash` field | config | n/a | grep over `revision/` found ZERO `data_hash`/`sha256`/`tobytes`; 09.1 wrote none | RESEARCH lines 283-290: recompute from `load_and_preprocess`, verify by construction — do NOT grep 09.1 configs |

## Metadata

**Analog search scope:** `revision/core/models/`, `revision/core/training.py`, `revision/run_ablation*.{py,sh}`, `revision/_build_analysis_notebook.py`, `revision/core/models/__init__.py`, `revision/core/models/critic.py`
**Files scanned:** 7 source files read (quantum.py, run_ablation.py, training.py:220-360, run_ablation_sweep.sh, _build_analysis_notebook.py:90-154 & 425-484, models/__init__.py, critic.py grep)
**Pattern extraction date:** 2026-05-17
**Key constraint:** D-10-13 — only model definitions in `revision/core/`; all loop/aggregation/orchestration logic in `run_baselines.py`, `run_baselines_sweep.sh`, and the notebook.
