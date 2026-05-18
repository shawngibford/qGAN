# Phase 11: Utility Evaluation - Pattern Map

**Mapped:** 2026-05-17
**Files analyzed:** 5 new (3 drivers + 1 optional notebook builder + 1 optional pytest) + 3 new JSON outputs
**Analogs found:** 5 / 5 (all exact or strong role-matches; codebase is the canonical source)

> Phase 11 is ~90% wiring existing verified components. Every new file copies a
> pinned in-tree analog. The risk is numeric drift vs. the Phase 10 anchor, so
> `reconstruct_od` / `train_eval_tstr` / `r2_score_inline` are copied **verbatim**,
> not re-derived (RESEARCH "Don't Hand-Roll" / Pitfall 3).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `revision/run_utility.py` (NEW) | driver | batch / transform (read frozen artifacts → TSTR + augmentation → JSON) | `revision/run_baselines.py` | exact (driver shape) + `_build_baseline_notebook.py` (TSTR/recon logic) |
| `revision/run_timegan_scores.py` (NEW) | driver | batch / transform (frozen artifacts → post-hoc GRU nets → JSON) | `revision/run_baselines.py` | role-match (driver shape; nets are net-new faithful port) |
| `revision/run_dualscale_fidelity.py` (NEW, or folded into a notebook) | driver | transform (recon → `eval.py` metrics → scale-tagged rows) | `_build_baseline_notebook.py` Cell 6 (long-form metric loop) | exact (same metric suite, adds `scale`) |
| `revision/_build_utility_notebook.py` (OPTIONAL) | utility (notebook source generator) | file-I/O | `revision/_build_baseline_notebook.py` | exact |
| `revision/tests/test_utility.py` (OPTIONAL, recommended) | test | request-response (assertions) | `revision/tests/test_classical.py` / `test_nonadversarial.py` | role-match |
| `revision/results/tstr.json` (NEW output) | config/artifact | — | `revision/results/baseline_comparison.json` | exact (long-form schema) |
| `revision/results/predictive_discriminative.json` (NEW output) | config/artifact | — | `revision/results/baseline_comparison.json` | exact (long-form schema) |
| `revision/results/augmentation.json` (NEW output) | config/artifact | — | `revision/results/baseline_comparison.json` | exact (long-form schema + `injection_ratio`) |

**Constants:** All drivers use `MODEL_KINDS = ["quantum","wgan_mlp","wgan_cnn","wgan_lstm","vae","ar"]`, `PIPELINES = ["A","B"]`, `SEEDS = [42,43,44,45,46]` → 6×2×5 = 60 run dirs (all verified present on disk; `samples.npy` confirmed `(3840,10)` float64, range ≈ `[-0.23, 0.28]` for wgan_mlp/B/42).

## Pattern Assignments

### `revision/run_utility.py` (driver — EVAL-01 TSTR + EVAL-04 augmentation)

**Primary analog:** `revision/run_baselines.py` (driver shape) + `revision/_build_baseline_notebook.py` (recon + TSTR logic, copied verbatim).

**CLI / driver skeleton** — copy from `revision/run_baselines.py:430-529`:
```python
# argparse → resolve run dir → load frozen artifacts → compute → write JSON.
# One concern per driver. Idempotent. NO multiprocessing.Pool (RESEARCH Pitfall 5).
ap = argparse.ArgumentParser(description="Phase 11 utility (TSTR + augmentation)")
ap.add_argument("--out", type=Path, default=Path("revision/results"))
ap.add_argument("--csv-path", type=Path, default=Path("./data.csv"))
args = ap.parse_args()
```
Note `revision/run_baselines.py:451-453` defaults `--csv-path` to `./data.csv` (repo-root relative — RESEARCH Pitfall 6: resolve to an absolute repo-root path; recomputed hash MUST equal `91e447d4624e25b3`).

**Imports pattern** — copy from `revision/_build_baseline_notebook.py:67-99` (the notebook Cell-2 import block; drivers use the same set, no `pandas` required):
```python
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from revision.core.preprocessing import inverse_logreturns
from revision.core.data import load_and_preprocess, rolling_window
from revision.core import WINDOW_LENGTH   # WINDOW_LENGTH == 10
# (eval.py helpers NOT needed in run_utility.py — only in the dual-scale driver)
```

**Data-hash invariant** — copy from `revision/run_baselines.py:226-234` AND the assert loop from `_build_baseline_notebook.py:124-149`:
```python
def _compute_data_hash(csv_path: Path) -> str:
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]
# assert == "91e447d4624e25b3" AND == every one of the 50 baseline config.yaml
# data_hash fields. Quantum runs carry NO data_hash field — equivalence is
# by-construction; do NOT grep transform_ablation configs (RESEARCH Pitfall 4).
```

**Run-dir resolver + `reconstruct_od`** — copy **VERBATIM** from `revision/_build_baseline_notebook.py:167-210` (do not "clean up" — the Pipeline-B seeded `od_start` draw is load-bearing, RESEARCH Pitfall 3):
```python
def _run_base(model_kind, pipeline, seed):
    if model_kind == "quantum":
        return Path(f"revision/results/transform_ablation/runs/{pipeline}/{seed}")
    return Path(f"revision/results/baselines/runs/{model_kind}/{pipeline}/{seed}")

def reconstruct_od(model_kind, pipeline, seed, n_synth_subsample=None):
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)   # (3840,10)
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)
    # ... subsample branch (lines 181-184) ...
    if pipeline == "A":
        od_min=float(inv["od_min"]); od_max=float(inv["od_max"])
        od = ((samples_pm1+1.0)/2.0)*(od_max-od_min)+od_min
        return {"od_samples": od, "transformed": None, ...}
    if pipeline == "B":
        r_min=float(inv["r_min"]); r_max=float(inv["r_max"])
        mu=float(inv["mu"]); sigma=float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])
        r_norm = ((samples_pm1+1.0)/2.0)*(r_max-r_min)+r_min
        rng = np.random.default_rng(seed * 7919 + 1)          # LOAD-BEARING seed
        od_start = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
        od_full = inverse_logreturns(torch.tensor(r_norm), torch.tensor(od_start),
                                     torch.tensor(mu), torch.tensor(sigma))
        od = od_full.cpu().numpy()
        if od.shape[1] == 11: od = od[:, :10]
        return {"od_samples": od, "transformed": r_norm, ...}
```

**Real windowed OD + held-out split** — copy from `_build_baseline_notebook.py:104-110` and `:458-462`:
```python
d_real = load_and_preprocess("./data.csv")
real_windowed_OD = rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()  # (384,10)
HELD_OUT_N = 320
real_eval  = real_windowed_OD[:HELD_OUT_N]   # eval set (identical to Phase 10 D-10-21)
real_train_for_baseline = real_windowed_OD[HELD_OUT_N:]   # n_train_real == 65 (verified)
```
**Augmentation leakage guard (EVAL-04, RESEARCH Pitfall 5):** the real portion mixed
into augmented training MUST come from `real_windowed_OD[320:]` (the 65-window train
partition), NEVER `[:320]`. Assert `set(train_idx) ∩ set(eval_idx) == ∅`. A positive
real-only R² is the leakage warning sign (Phase 10 anchor is R²=-13.354 ± 0.583).

**Core TSTR pattern + inline metrics** — copy **VERBATIM** from `_build_baseline_notebook.py:395-440`, then add MAE/RMSE to the return dict (sklearn is ABSENT — RESEARCH Pitfall 2):
```python
class TSTRLiteLSTM(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden,
                                  num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

def r2_score_inline(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def train_eval_tstr(train_windows, eval_windows, lstm_seed=40, hidden=32,
                    epochs=50, bs=64):
    torch.manual_seed(lstm_seed); rng = np.random.default_rng(lstm_seed)
    Xtr = torch.tensor(train_windows[:, :9], dtype=torch.float32).unsqueeze(-1)
    ytr = torch.tensor(train_windows[:, 9:10], dtype=torch.float32)
    Xev = torch.tensor(eval_windows[:, :9], dtype=torch.float32).unsqueeze(-1)
    yev = torch.tensor(eval_windows[:, 9:10], dtype=torch.float32)
    # ... Adam(lr=1e-3), MSELoss, permuted-minibatch loop (lines 419-432) ...
    return {
        "mse":  float(np.mean((yev_np - yev_pred) ** 2)),
        "r2":   r2_score_inline(yev_np, yev_pred),
        "mae":  float(np.mean(np.abs(yev_np - yev_pred))),          # PHASE 11 ADD
        "rmse": float(np.sqrt(np.mean((yev_np - yev_pred) ** 2))),  # PHASE 11 ADD
    }
```
**TSTR roll-up loop** — copy from `_build_baseline_notebook.py:464-501`: per `(model_kind,pipeline)` pool synthetic OD across the 5 training seeds, train 3 LSTMs at init seeds `{40,41,42}` (NOT the training seeds), report `*_mean/*_std` + `per_init_seed`. Reproduce the Phase 10 `real_only_baseline` block verbatim (`n_train_real=65`, R²=-13.354) as the augmentation anchor.

**JSON emission** — see Shared Patterns → Long-form JSON.

---

### `revision/run_timegan_scores.py` (driver — EVAL-02/03 faithful TimeGAN nets)

**Primary analog:** `revision/run_baselines.py` (driver shape). The two post-hoc nets are net-new but follow the same `torch.nn.Module` + `torch.optim.Adam(lr=1e-3)` + manual minibatch-loop idiom already used by `TSTRLiteLSTM`/`train_eval_tstr` (`_build_baseline_notebook.py:395-440`) and the VAE loop (`run_baselines.py:285-377`).

**Predictive net** — pattern from `_build_baseline_notebook.py:395-403` (`TSTRLiteLSTM` shape), retargeted to GRU per the pinned Yoon et al. definition (RESEARCH "TimeGAN Score Definitions"):
```python
# Algorithm source: jsyoon0823/TimeGAN/metrics/predictive_metrics.py (CITE in JSON)
class PredictiveGRU(torch.nn.Module):
    def __init__(self, H):                       # H pinned & RECORDED in JSON metadata
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=H, num_layers=1,
                                batch_first=True)
        self.fc  = torch.nn.Linear(H, 1)
    def forward(self, x):                        # x: (B, 9, 1)
        out, _ = self.gru(x); return self.fc(out)
# train: synthetic windows X=window[:,:-1] y=window[:,-1]; Adam(lr=1e-3),
#        5000 iters, batch 128. test: real_windowed_OD.
# score = mean(|y_real - y_pred|). Lower = better.
```
**Discriminative net:** same single-layer-GRU shape; `Linear(H,1)` → sigmoid logit;
labels real=1 / synth=0; 80/20 split of each pool independently
(`np.random.permutation` with a recorded seed); 2000 iters; batch 128;
`score = abs(0.5 - test_accuracy)`. Lower = better.

**Univariate adaptation (RESEARCH Pitfall 1, Assumption A1):** canonical `hidden_dim = int(dim/2)` is degenerate at `dim=1`. Planner must lock one `H` (recommend `H = WINDOW_LENGTH = 10` or `H = 5`), use the same `H` for both nets, and **record `H` + the adaptation rationale + the `jsyoon0823/TimeGAN` repo URL + `master` commit in `predictive_discriminative.json` metadata**.

**Per-seed roll-up (D-11-05):** compute both scores per training seed `{42,43,44,45,46}` (each seed's synthetic pool), then report `mean ± std` across the 5 seeds per `(model_kind, pipeline)`.

**Recon + data-hash + JSON:** identical to `run_utility.py` (reuse the verbatim `reconstruct_od` and the Shared Patterns below).

---

### `revision/run_dualscale_fidelity.py` (driver — EVAL-05 scale-tagged re-emit)

**Primary analog:** `revision/_build_baseline_notebook.py:234-296` (Cell-6 long-form metric loop) — copy the loop structure; do NOT add new metric math (D-11-10).

**Imports** — add the `eval.py` helpers to the `run_utility.py` import block:
```python
from revision.core.eval import compute_emd, compute_acf, compute_dtw, compute_moments
```

**Core scale-tagging loop** — copy from `_build_baseline_notebook.py:242-291`. The existing loop already emits `scale="OD"` and (Pipeline B) `scale="transformed"`. Phase 11 EVAL-05 renames/duplicates so **every** `revision.core.eval` metric carries an explicit `scale: "OD" | "log_return"` field:
```python
for mk in MODEL_KINDS:
  for p in PIPELINES:
    for s in SEEDS:
      r = reconstruct_od(mk, p, s)
      od = r["od_samples"]; synth_flat = od.reshape(-1)
      rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                       metric_name="emd", scale="OD",
                       value=compute_emd(real_flat, synth_flat)))
      # ... compute_moments / compute_acf (lags 0..9 mean+std) / compute_dtw ...
      if r["transformed"] is not None:                    # Pipeline B only
          rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                           metric_name="emd", scale="log_return",
                           value=compute_emd(real_log_delta,
                                             r["transformed"].reshape(-1))))
      # Pipeline A: log_return scale is n/a — emit explicit null OR omit w/ reason
```
DTW uses the bounded `DTW_N_PAIRS=100` nearest-neighbour sub-sampling recipe with `np.random.default_rng(s*31)` exactly as `_build_baseline_notebook.py:266-284` (matches the 09.1 notebook; do not change the seed or pair count or numbers drift).

**Reuse-unchanged mandate (D-11-10):** `git diff revision/core/` MUST be empty after Phase 11. All EVAL-05 logic lives in this driver, never in `revision/core/eval.py`.

---

### `revision/_build_utility_notebook.py` (OPTIONAL notebook source)

**Exact analog:** `revision/_build_baseline_notebook.py` (entire file). Copy the `md()`/`code()` cell-builder helpers (`:32-43`), the `CELLS = []` accumulator, the `_find_repo_root()` + `os.chdir(REPO)` Cell-2 preamble (`:74-93`), and the final `nb = {...}; NB_PATH.write_text(json.dumps(nb, indent=1))` writer (`:625-636`). Not gitignored — kept as the canonical deterministic notebook source.

---

### `revision/tests/test_utility.py` (OPTIONAL, recommended)

**Analog:** `revision/tests/test_classical.py` / `revision/tests/test_nonadversarial.py` (existing pytest convention; run via `./qgan_env/bin/python -m pytest revision/tests/ -x -q`). Assert: every loaded `samples.npy` is `(3840,10)` float64; recomputed data-hash == `91e447d4624e25b3`; TSTR round-trip vs the Phase 10 anchor (quantum|B R² ≈ 0.994, real-only R² ≈ -13.354); augmentation partition disjointness; `git diff --stat revision/core/` empty.

---

## Shared Patterns

### Long-form JSON schema (extend, never replace)
**Source:** `revision/results/baseline_comparison.json` (verified: 1710 rows) + the writer at `revision/_build_baseline_notebook.py:352-378`.
**Apply to:** all three Phase 11 outputs (`tstr.json`, `predictive_discriminative.json`, `augmentation.json`).
```python
# Verified top-level keys: schema, model_kinds, pipelines, seeds, data_hash,
#   data_hash_verification, metric_helpers, recommendation, models, rows[, tstr]
# Verified row shape:
{"model_kind": "quantum", "pipeline": "A", "seed": 42,
 "metric_name": "emd", "scale": "OD", "value": 1.0520125260633941}
# Verified models[] entry:
{"kind": "quantum", "parameter_count": 75, "family": "adversarial-quantum",
 "train_protocol_notes": "QuantumGenerator(...) ... *0.1 (training.py:283)."}
schema = "long-form rows[] + models[] aggregate (D-10-16)"
data_hash = "91e447d4624e25b3"
```
- `tstr.json`: rows with `metric_name ∈ {r2,mae,rmse,mse}`, `scale="OD"`; plus a `tstr` block mirroring `baseline_comparison.json["tstr"]` (per-`{mk|p}` `*_mean/*_std/per_init_seed`, and `real_only_baseline` with `n_train_real:65`, `n_eval_real:320`, R²=-13.354 anchor).
- `predictive_discriminative.json`: rows `metric_name ∈ {predictive_score, discriminative_score}`, `mean ± std` over seeds {42-46}; metadata records TimeGAN repo URL + commit + chosen `H` + univariate-adaptation rationale.
- `augmentation.json`: rows `metric_name ∈ {r2_delta, mae_delta, rmse_delta}`, add an `injection_ratio` field per row (`real_only`, `+25%`, `+50%`, `+100%`, `synthetic_only`), `scale="OD"`; metadata states the ~60× synthetic-budget caveat (lift is a lower bound — RESEARCH Orlandi section).

Write with `json.dumps(obj, indent=2)` via the `Path(...).write_text(...)` idiom (`_build_baseline_notebook.py:374`). Atomic-write/idempotent-overwrite shape: `run_baselines.py:456-463` (`shutil.rmtree` then recreate) if a sweep wrapper is added.

### Run-dir resolution + frozen-artifact contract
**Source:** `revision/_build_baseline_notebook.py:167-210`.
**Apply to:** every driver (`run_utility.py`, `run_timegan_scores.py`, `run_dualscale_fidelity.py`).
Verified on disk: baseline run dir = `revision/results/baselines/runs/<model>/<pipeline>/<seed>/{config.yaml,checkpoint.pt|.npz,samples.npy,metrics.json,inverse_kwargs.npz}`; quantum run dir = `revision/results/transform_ablation/runs/<pipeline>/<seed>/` (extra `_stdout.log`/`_stderr.log`, no `data_hash` in `config.yaml`). `samples.npy` confirmed `(3840,10)` float64 in `[-1,1]`-ish window space.

### Inverse-transform helpers (reuse unchanged — D-11-10)
**Source:** `revision/core/preprocessing.py` — `inverse_logreturns` (`:49-72`, Pipeline B), `inverse_minmax_od` (`:92-96`, Pipeline A). Round-trip ≤1e-8 (ABL-01 verified). Called only through the verbatim `reconstruct_od` wrapper; never modify `revision/core/`.

### Fidelity metric helpers (reuse unchanged — D-11-10)
**Source:** `revision/core/eval.py` — `compute_emd` (`:25`), `compute_moments` (`:42`), `compute_acf` (`:64`, FFT), `compute_dtw` (`:78`, fastdtw+euclidean), `compute_jsd` (`:96`), `full_metric_suite` (`:143`). v1.0/v1.1 behavioral locks (raw-sample EMD, FFT ACF, Fisher kurtosis, ddof=0 std) baked in. EVAL-05 only wraps each call with a `scale` field at the driver level.

### Sweep / parallelism guardrail (only if a sweep wrapper is added)
**Source:** `revision/run_baselines_sweep.sh`. Phase 11 work is fast aggregation over frozen artifacts (<15 min, RESEARCH A5) so a sweep is likely unnecessary; if added, copy the `is_complete`/`update_status`/atomic-`os.rename` status pattern and the `xargs -P 2` (NEVER `multiprocessing.Pool`) idiom (`:174-184`, `:401-419`). Venv-binary invocation pattern: `./qgan_env/bin/python` (`:97-107`).

## No Analog Found

| File | Role | Data Flow | Reason / Mitigation |
|------|------|-----------|---------------------|
| (none) | — | — | Every Phase 11 file has a strong in-tree analog. The only net-new logic is the two faithful TimeGAN post-hoc GRU nets in `run_timegan_scores.py` — these have no in-tree analog by *exact algorithm*, but their **structural** pattern (single-layer recurrent net + `Linear` head + `Adam(lr=1e-3)` manual minibatch loop) is directly modeled on `TSTRLiteLSTM`/`train_eval_tstr` (`_build_baseline_notebook.py:395-440`); the *algorithm* is pinned to `jsyoon0823/TimeGAN` (RESEARCH, cite in JSON metadata). Not a true "no analog". |

## Metadata

**Analog search scope:** `revision/`, `revision/core/`, `revision/results/`, `revision/results/baselines/runs/`, `revision/results/transform_ablation/runs/`
**Files scanned/read:** `run_baselines.py`, `_build_baseline_notebook.py`, `core/eval.py`, `core/preprocessing.py`, `run_baselines_sweep.sh`, `baseline_comparison.json` (schema probe), on-disk run-dir + `samples.npy` shape verification
**Key verified facts:** `samples.npy` = `(3840,10)` float64; `data_hash` = `91e447d4624e25b3`; `real_only_baseline.n_train_real` = `65`, R² = `-13.354 ± 0.583`; `baseline_comparison.json` = 1710 long-form rows; quantum run dirs carry no `data_hash` field
**Pattern extraction date:** 2026-05-17
