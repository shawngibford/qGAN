"""Generate revision/06_baseline_comparison.ipynb from inlined cell sources.

This script is the SOURCE of the notebook content for plan 10-04 (Phase 10
classical baselines). It produces a single .ipynb by assembling cell sources
defined below. Run once; the .ipynb is then executed via
`jupyter nbconvert --execute --inplace`.

Mirrors revision/_build_analysis_notebook.py (the 09.1 generator pattern).
Not gitignored — kept as the canonical, deterministic notebook source.

Key contracts (Phase 10 plan 10-04):
* reconstruct_od copied VERBATIM from _build_analysis_notebook.py:95-149,
  keeping ONLY the A (105-114) and B (115-127) branches; the C branch
  (129-147) is DELETED (D-10-05). The run-dir base is parametrized so it
  accepts BOTH the new baselines layout
  revision/results/baselines/runs/<model>/<pipeline>/<seed>/ AND the reused
  09.1 quantum layout revision/results/transform_ablation/runs/<pipeline>/<seed>/.
* Metrics computed ONLY via revision.core.eval (D-10-20) — no new helpers.
* TSTRLiteLSTM / r2_score_inline / train_eval_tstr copied VERBATIM from
  _build_analysis_notebook.py:432-477 (D-10-13 — not promoted to core).
* data_hash recomputed ONCE from load_and_preprocess and asserted equal across
  all 50 new config.yaml; quantum equivalence by construction (D-10-15,
  Pitfall 4 — no 09.1 config grep).
* NO recommendation emitted (D-10-19).
"""
import json
from pathlib import Path

NB_PATH = Path("revision/06_baseline_comparison.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s,
    }


CELLS = []

# Cell 1 — title
CELLS.append(md("""\
# Phase 10 BASE-03 — Classical Baselines Apples-to-Apples Comparison

Quantum (reused from Phase 09.1) + 3 matched-parameter classical WGAN-GP
variants (MLP / CNN / LSTM) + 2 non-adversarial baselines (VAE, AR) across
pipelines A and B, 5 seeds (42-46), 1000 epochs.

Produces `baseline_classical_wgan.json` (BASE-01), `baseline_nonadversarial.json`
(BASE-02), and `baseline_comparison.{json,md}` (BASE-03).

Per **D-10-19** this notebook emits NO recommendation on which baseline is best
— Phase 14 owns that decision, driven by Phase 11 utility numbers. Per
**D-10-20** every fidelity metric is computed via `revision.core.eval` only (no
new helpers). Per **D-10-13** the TSTR-lite forecaster is copied verbatim from
the 09.1 notebook generator and is NOT promoted to `revision/core/`.
"""))

# Cell 2 — imports + repo root (mirrors _build_analysis_notebook.py:42-81)
CELLS.append(code("""\
import json, yaml, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Locate repo root by walking up until we find revision/core/
def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")

REPO = _find_repo_root()
import os
os.chdir(REPO)
print("repo:", REPO)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.core.preprocessing import inverse_logreturns
from revision.core.data import load_and_preprocess, rolling_window
from revision.core.eval import compute_emd, compute_acf, compute_dtw, compute_moments
from revision.core import WINDOW_LENGTH

# 6 model_kinds: quantum reused from 09.1; the 5 new ones from Phase 10 sweep.
MODEL_KINDS = ["quantum", "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
PIPELINES = ["A", "B"]            # Pipeline C dropped (D-10-05)
SEEDS = [42, 43, 44, 45, 46]
print("models:", MODEL_KINDS, "pipelines:", PIPELINES, "seeds:", SEEDS)
"""))

# Cell 3 — real data (mirrors _build_analysis_notebook.py:83-91)
CELLS.append(code("""\
d_real = load_and_preprocess("./data.csv")
real_OD = d_real["OD"].cpu().numpy()
real_log_delta = d_real["log_delta"].cpu().numpy()
real_windowed_OD = rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()
real_od_starts = real_windowed_OD[:, 0]
print(f"real OD: shape={real_OD.shape}, log_delta: shape={real_log_delta.shape}, "
      f"windowed_OD: shape={real_windowed_OD.shape}")
"""))

# Cell 4 — data_hash invariant (D-10-15, Pitfall 4)
CELLS.append(md("""\
## Data-hash invariant (D-10-15, RESEARCH Pitfall 4)

Recompute the OD-tensor SHA-256 prefix ONCE from `load_and_preprocess` and
assert all **50 new** `config.yaml` `data_hash` fields equal it. The 09.1
quantum runs are equivalent **by construction**: they were produced by the same
`load_and_preprocess` on the same `data.csv`. 09.1 wrote no `data_hash` field,
so we deliberately do NOT grep the 09.1 configs (anti-pattern — Pitfall 4).
"""))
CELLS.append(code("""\
expected_data_hash = hashlib.sha256(
    load_and_preprocess("data.csv")["OD"].cpu().numpy().tobytes()
).hexdigest()[:16]
print("recomputed data_hash:", expected_data_hash)

NEW_MODELS = ["wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
checked = 0
mismatches = []
for m in NEW_MODELS:
    for p in PIPELINES:
        for s in SEEDS:
            cfg_path = Path(f"revision/results/baselines/runs/{m}/{p}/{s}/config.yaml")
            cfg = yaml.safe_load(cfg_path.read_text())
            checked += 1
            if cfg.get("data_hash") != expected_data_hash:
                mismatches.append((m, p, s, cfg.get("data_hash")))
assert checked == 50, f"expected 50 new configs, checked {checked}"
assert not mismatches, f"data_hash mismatch (UNEXPECTED — report loudly): {mismatches}"
print(f"data_hash invariant OK: all {checked} new configs == {expected_data_hash}")
quantum_equiv_note = (
    "Quantum (09.1) data equivalence established BY CONSTRUCTION: the 09.1 "
    "transform_ablation runs used the identical load_and_preprocess(data.csv) "
    "OD tensor; 09.1 wrote no data_hash field, so no grep is performed "
    "(D-10-15, RESEARCH Pitfall 4)."
)
print(quantum_equiv_note)
"""))

# Cell 5 — reconstruct_od (VERBATIM A+B from _build_analysis_notebook.py:95-127,
# C branch DELETED per D-10-05; base path parametrized for both layouts).
CELLS.append(md("""\
## `reconstruct_od` — copied VERBATIM (A + B branches only)

Copied verbatim from `_build_analysis_notebook.py:95-127`. The A branch
(min-max OD inverse) and B branch (log-returns inverse via
`inverse_logreturns`) are byte-identical to 09.1. The **C branch is DELETED**
(D-10-05). The only change is the `base` path: it is parametrized by
`model_kind` so it resolves the reused quantum runs at
`revision/results/transform_ablation/runs/<pipeline>/<seed>` and the 5 new
models at `revision/results/baselines/runs/<model>/<pipeline>/<seed>`
(D-10-04 / D-10-18). The inverse-kwargs contract is otherwise unchanged.
"""))
CELLS.append(code("""\
def _run_base(model_kind: str, pipeline: str, seed: int) -> Path:
    if model_kind == "quantum":
        # reused 09.1 quantum runs (D-10-18)
        return Path(f"revision/results/transform_ablation/runs/{pipeline}/{seed}")
    # new Phase 10 baseline runs (D-10-14 layout: runs/<model>/<pipeline>/<seed>)
    return Path(f"revision/results/baselines/runs/{model_kind}/{pipeline}/{seed}")


def reconstruct_od(model_kind: str, pipeline: str, seed: int,
                   n_synth_subsample: int | None = None) -> dict:
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)

    if n_synth_subsample is not None and samples_pm1.shape[0] > n_synth_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(samples_pm1.shape[0], n_synth_subsample, replace=False)
        samples_pm1 = samples_pm1[idx]

    if pipeline == "A":
        od_min = float(inv["od_min"]); od_max = float(inv["od_max"])
        od01 = (samples_pm1 + 1.0) / 2.0
        od = od01 * (od_max - od_min) + od_min
        return {"od_samples": od, "transformed": None, "n_synth": od.shape[0],
                "pipeline": pipeline, "seed": seed}

    if pipeline == "B":
        r_min = float(inv["r_min"]); r_max = float(inv["r_max"])
        mu = float(inv["mu"]); sigma = float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])
        r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
        rng = np.random.default_rng(seed * 7919 + 1)
        od_start_per_window = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
        r_norm_t = torch.tensor(r_norm)
        od_start_t = torch.tensor(od_start_per_window)
        od_full = inverse_logreturns(r_norm_t, od_start_t,
                                     torch.tensor(mu), torch.tensor(sigma))
        od = od_full.cpu().numpy()
        if od.shape[1] == 11:
            od = od[:, :10]
        return {"od_samples": od, "transformed": r_norm, "n_synth": od.shape[0],
                "pipeline": pipeline, "seed": seed}

    raise ValueError(f"unknown pipeline {pipeline} (Pipeline C dropped, D-10-05)")


# Smoke-test on one (model_kind, pipeline, seed) per family
for mk in ("quantum", "wgan_mlp", "vae", "ar"):
    for p in PIPELINES:
        r = reconstruct_od(mk, p, 42)
        print(f"{mk:>9} {p}: od_samples={r['od_samples'].shape}, "
              f"transformed={'None' if r['transformed'] is None else r['transformed'].shape}")
"""))

# Cell 6 — long-form metrics rows via revision.core.eval ONLY (D-10-20).
# Mirrors _build_analysis_notebook.py:159-209 (same metric suite + DTW recipe).
CELLS.append(md("""\
## Long-form metric rows — `revision.core.eval` ONLY (D-10-20)

For every (model_kind × pipeline × seed) reconstruct OD and compute the locked
metric suite using **only** `revision.core.eval` helpers (D-10-20 — no new
helpers): OD-scale EMD, moments (mean/std/skewness/kurtosis), per-lag ACF
mean+std (lags 0..9), nearest-neighbour DTW mean/median/std, plus
transformed-space EMD for Pipeline B. Rows are
`{model_kind, pipeline, seed, metric_name, scale, value}` — the 09.1
`metrics.csv` long-form schema plus `model_kind` (D-10-16).
"""))
CELLS.append(code("""\
NLAGS = 9
DTW_N_PAIRS = 100  # matches 09.1 notebook (bounded runtime)
rows = []
recon_cache = {}   # (model_kind, pipeline, seed) -> reconstruct_od dict (invert once)

real_flat = real_windowed_OD.reshape(-1)

for mk in MODEL_KINDS:
    for p in PIPELINES:
        for s in SEEDS:
            print(f"  metrics: model={mk} pipeline={p} seed={s}")
            r = reconstruct_od(mk, p, s)
            recon_cache[(mk, p, s)] = r
            od = r["od_samples"]
            synth_flat = od.reshape(-1)
            # EMD on OD scale (pooled across windows)
            rows.append(dict(model_kind=mk, pipeline=p, seed=s, metric_name="emd",
                             scale="OD", value=compute_emd(real_flat, synth_flat)))
            for k, v in compute_moments(synth_flat).items():
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name=f"moment_{k}", scale="OD", value=v))
            # ACF per window aggregated (lags 0..NLAGS)
            acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in od])
            for lag in range(NLAGS + 1):
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name=f"acf_lag{lag}_mean", scale="OD",
                                 value=float(acfs[:, lag].mean())))
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name=f"acf_lag{lag}_std", scale="OD",
                                 value=float(acfs[:, lag].std())))
            # DTW nearest-neighbor sub-sampled (slow O(L^2) per pair)
            rng = np.random.default_rng(s * 31)
            synth_idx = rng.choice(od.shape[0],
                                   size=min(DTW_N_PAIRS, od.shape[0]), replace=False)
            real_idx = rng.choice(real_windowed_OD.shape[0],
                                  size=min(64, real_windowed_OD.shape[0]),
                                  replace=False)
            dtw_vals = []
            for i in synth_idx:
                best = min(compute_dtw(od[i], real_windowed_OD[j]) for j in real_idx)
                dtw_vals.append(best)
            rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                             metric_name="dtw_mean", scale="OD",
                             value=float(np.mean(dtw_vals))))
            rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                             metric_name="dtw_median", scale="OD",
                             value=float(np.median(dtw_vals))))
            rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                             metric_name="dtw_std", scale="OD",
                             value=float(np.std(dtw_vals))))
            # Transformed-space EMD (Pipeline B only — C dropped)
            if r["transformed"] is not None:
                trans_flat = r["transformed"].reshape(-1)
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name="emd", scale="transformed",
                                 value=compute_emd(real_log_delta, trans_flat)))

df = pd.DataFrame(rows, columns=["model_kind", "pipeline", "seed",
                                 "metric_name", "scale", "value"])
print(f"\\nlong-form rows: {len(df)}")
print(df.head(8))
"""))

# Cell 7 — models[] aggregate array (D-10-16). Parameter counts + family +
# train_protocol_notes are read straight from the on-disk config.yaml.
CELLS.append(md("""\
## `models[]` aggregate — parameter counts + family + train-protocol notes

The top-level `models[]` array carries `{kind, parameter_count, family,
train_protocol_notes}` (D-10-16). `parameter_count` is read from each model's
own `config.yaml` (written by the Wave-2 sweep from `count_params()`); the
quantum count (75) is the matched-param target from Phase 09.1
(QuantumGenerator(5,4) = 5 + 4*15 + 10). `train_protocol_notes` carries the
`*0.1` asymmetry note for VAE/AR and any KL-warmup applied.
"""))
CELLS.append(code("""\
def _read_new_cfg(model_kind: str) -> dict:
    # Read one representative config (A/42) — schema is identical across the
    # 5 seeds x 2 pipelines for a given model.
    return yaml.safe_load(
        Path(f"revision/results/baselines/runs/{model_kind}/A/42/config.yaml").read_text()
    )

models = []
# Quantum reference: 75 params by construction (5 + 4*15 + 10), matched-param
# target from Phase 09.1. No config.yaml data_hash in 09.1 (by-construction).
models.append({
    "kind": "quantum",
    "parameter_count": 75,
    "family": "adversarial-quantum",
    "train_protocol_notes": (
        "QuantumGenerator(num_qubits=5, num_layers=4) PQC = 5 + 4*15 + 10 = 75 "
        "params; trained via train_wgan_gp (Phase 09.1, reused D-10-18); "
        "generator output scaled by *0.1 (training.py:283)."
    ),
})
for mk in ("wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"):
    cfg = _read_new_cfg(mk)
    models.append({
        "kind": mk,
        "parameter_count": int(cfg["parameter_count"]),
        "family": cfg["family"],
        "train_protocol_notes": cfg["train_protocol_notes"],
    })

for m in models:
    print(f"  {m['kind']:>9}  params={m['parameter_count']:>4}  {m['family']}")

# Sanity: matched-parameter band (RESEARCH: 71..79 for the WGAN/quantum group)
_pc = {m["kind"]: m["parameter_count"] for m in models}
assert _pc["quantum"] == 75 and _pc["wgan_mlp"] == 74 and _pc["wgan_cnn"] == 73 \\
    and _pc["wgan_lstm"] == 78 and _pc["ar"] == 3 and 540 <= _pc["vae"] <= 580, _pc
print("parameter-count contract OK")
"""))

# Cell 8 — write the partial baseline_comparison.json (Task 1 deliverable;
# TSTR-lite block + md render are added by Task 2 cells).
CELLS.append(code("""\
out_dir = Path("revision/results")
out_dir.mkdir(parents=True, exist_ok=True)

comparison = {
    "schema": "long-form rows[] + models[] aggregate (D-10-16)",
    "model_kinds": MODEL_KINDS,
    "pipelines": PIPELINES,
    "seeds": SEEDS,
    "data_hash": expected_data_hash,
    "data_hash_verification": {
        "recomputed_from": "load_and_preprocess('data.csv')['OD'].tobytes() sha256[:16]",
        "n_new_configs_checked": 50,
        "all_equal": True,
        "quantum_equivalence": quantum_equiv_note,
    },
    "metric_helpers": "revision.core.eval ONLY (D-10-20): "
                      "compute_emd / compute_moments / compute_acf / compute_dtw",
    "recommendation": "NONE — Phase 14 owns the highlight decision (D-10-19)",
    "models": models,
    "rows": df.to_dict(orient="records"),
}
(out_dir / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2))
print("wrote revision/results/baseline_comparison.json "
      f"({len(comparison['rows'])} rows, {len(models)} models) "
      "[partial — TSTR-lite block + md added below]")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — TSTR-lite (VERBATIM from _build_analysis_notebook.py:432-477) +
# emit BASE-01/02/03 artifacts.
# ─────────────────────────────────────────────────────────────────────────────

# Cell 9 — TSTR-lite definitions (VERBATIM, D-10-13: not promoted to core)
CELLS.append(md("""\
## TSTR-lite — copied VERBATIM (D-10-13)

`TSTRLiteLSTM`, `r2_score_inline`, and `train_eval_tstr` are byte-identical
copies of `_build_analysis_notebook.py:432-477`. sklearn is **not** installed
(R²computed inline). These are NOT promoted to `revision/core/` (D-10-13).
"""))
CELLS.append(code("""\
# Task 2 — TSTR-lite LSTM forecaster
class TSTRLiteLSTM(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden,
                                  num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def r2_score_inline(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def train_eval_tstr(train_windows: np.ndarray, eval_windows: np.ndarray,
                    lstm_seed: int = 40, hidden: int = 32,
                    epochs: int = 50, bs: int = 64) -> dict:
    torch.manual_seed(lstm_seed)
    rng = np.random.default_rng(lstm_seed)
    Xtr = torch.tensor(train_windows[:, :9], dtype=torch.float32).unsqueeze(-1)
    ytr = torch.tensor(train_windows[:, 9:10], dtype=torch.float32)
    Xev = torch.tensor(eval_windows[:, :9], dtype=torch.float32).unsqueeze(-1)
    yev = torch.tensor(eval_windows[:, 9:10], dtype=torch.float32)
    model = TSTRLiteLSTM(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    n = Xtr.shape[0]
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, bs):
            j = idx[start:start + bs]
            xb, yb = Xtr[j], ytr[j]
            opt.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        yev_pred = model(Xev).cpu().numpy()
    yev_np = yev.cpu().numpy()
    return {
        "mse": float(np.mean((yev_np - yev_pred) ** 2)),
        "r2": r2_score_inline(yev_np, yev_pred),
    }
print("TSTR-lite definitions OK")
"""))

# Cell 10 — TSTR-lite run per (model_kind, pipeline) (D-10-21).
# Mirrors _build_analysis_notebook.py:482-521 (HELD_OUT_N=320, init seeds
# {40,41,42}, pool synthetic OD across the 5 training seeds).
CELLS.append(md("""\
## Run TSTR-lite per (model_kind, pipeline) — D-10-21

`HELD_OUT_N = 320`: eval on `real_windowed_OD[:320]`, train the real-only
baseline on `real_windowed_OD[320:]`. For each (model_kind, pipeline) the
synthetic OD windows are pooled across the 5 training seeds, then 3 LSTMs are
trained with init seeds {40,41,42} (NOT training seeds). Reports
`mse_mean/std`, `r2_mean/std`, `per_init_seed` — the 09.1 `tstr_lite.json`
schema.
"""))
CELLS.append(code("""\
HELD_OUT_N = 320
real_eval = real_windowed_OD[:HELD_OUT_N]
real_train_for_baseline = real_windowed_OD[HELD_OUT_N:]
print(f"TSTR: real_eval={real_eval.shape}, "
      f"real_train_for_baseline={real_train_for_baseline.shape}")

tstr = {}
for mk in MODEL_KINDS:
    for p in PIPELINES:
        print(f"  TSTR model={mk} pipeline={p} ...")
        synth_pool = np.concatenate(
            [recon_cache[(mk, p, s)]["od_samples"] for s in SEEDS], axis=0
        )
        per_seed = []
        for lstm_seed in (40, 41, 42):
            res = train_eval_tstr(synth_pool, real_eval, lstm_seed=lstm_seed)
            per_seed.append(res)
            print(f"    init_seed={lstm_seed} mse={res['mse']:.4f} r2={res['r2']:.3f}")
        mses = [x["mse"] for x in per_seed]; r2s = [x["r2"] for x in per_seed]
        tstr[f"{mk}|{p}"] = {
            "model_kind": mk,
            "pipeline": p,
            "n_train_synth": int(synth_pool.shape[0]),
            "n_eval_real": HELD_OUT_N,
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "per_init_seed": {str(k): v for k, v in zip([40, 41, 42], per_seed)},
        }

print("  TSTR real-only baseline ...")
per_seed_real = [train_eval_tstr(real_train_for_baseline, real_eval, lstm_seed=ls)
                 for ls in (40, 41, 42)]
mses_r = [x["mse"] for x in per_seed_real]; r2s_r = [x["r2"] for x in per_seed_real]
tstr["real_only_baseline"] = {
    "n_train_real": int(real_train_for_baseline.shape[0]),
    "n_eval_real": HELD_OUT_N,
    "mse_mean": float(np.mean(mses_r)),
    "mse_std": float(np.std(mses_r)),
    "r2_mean": float(np.mean(r2s_r)),
    "r2_std": float(np.std(r2s_r)),
    "per_init_seed": {str(k): v for k, v in zip([40, 41, 42], per_seed_real)},
}
print("TSTR-lite complete:", list(tstr.keys()))
"""))

# Cell 11 — emit BASE-01/02/03 artifacts + finalize comparison.json
CELLS.append(md("""\
## Emit BASE-01 / BASE-02 / BASE-03 artifacts

* **BASE-01** `baseline_classical_wgan.json` — rows[]+models[] filtered to
  {wgan_mlp, wgan_cnn, wgan_lstm}.
* **BASE-02** `baseline_nonadversarial.json` — rows[]+models[] filtered to
  {vae, ar}, carrying each model's `train_protocol_notes` (ELBO/beta/KL-warmup,
  AR p=2, the `*0.1` asymmetry note).
* **BASE-03** `baseline_comparison.json` (finalized with the `tstr` block) +
  `baseline_comparison.md` (one row per model per pipeline, no recommendation
  prose beyond the Phase-14 deferral caption — D-10-19).
"""))
CELLS.append(code("""\
# Finalize comparison.json with the TSTR-lite block (D-10-21)
comparison["tstr"] = tstr
(out_dir / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2))
print(f"finalized baseline_comparison.json ({len(comparison['rows'])} rows, "
      f"{len(models)} models, tstr keys={len(tstr)})")

def _subset(kinds: set) -> dict:
    return {
        "schema": comparison["schema"],
        "data_hash": expected_data_hash,
        "metric_helpers": comparison["metric_helpers"],
        "recommendation": comparison["recommendation"],
        "models": [m for m in models if m["kind"] in kinds],
        "rows": [r for r in comparison["rows"] if r["model_kind"] in kinds],
        "tstr": {k: v for k, v in tstr.items()
                 if v.get("model_kind") in kinds},
    }

# BASE-01 — classical WGAN-GP variants
wgan_kinds = {"wgan_mlp", "wgan_cnn", "wgan_lstm"}
(out_dir / "baseline_classical_wgan.json").write_text(
    json.dumps(_subset(wgan_kinds), indent=2))
print("wrote baseline_classical_wgan.json (BASE-01):", sorted(wgan_kinds))

# BASE-02 — non-adversarial baselines
na_kinds = {"vae", "ar"}
(out_dir / "baseline_nonadversarial.json").write_text(
    json.dumps(_subset(na_kinds), indent=2))
print("wrote baseline_nonadversarial.json (BASE-02):", sorted(na_kinds))
"""))

# Cell 12 — render baseline_comparison.md (D-10-17/18)
CELLS.append(code("""\
# BASE-03 markdown render: one row per model per pipeline (D-10-17/18).
def _agg(mk: str, p: str, metric_name: str, scale: str = "OD"):
    vals = [r["value"] for r in comparison["rows"]
            if r["model_kind"] == mk and r["pipeline"] == p
            and r["metric_name"] == metric_name and r["scale"] == scale]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

param_of = {m["kind"]: m["parameter_count"] for m in models}
lines = []
lines.append("# BASE-03 — Classical Baselines Apples-to-Apples Comparison\\n")
lines.append(
    "Quantum reference (reused from Phase 09.1, D-10-18) + 3 matched-parameter "
    "classical WGAN-GP variants + 2 non-adversarial baselines, across pipelines "
    "A and B (Pipeline C dropped, D-10-05), 5 training seeds (42-46).\\n")
lines.append(
    f"`data_hash` = `{expected_data_hash}` recomputed once and verified equal "
    "across all 50 new configs; quantum equivalence by construction "
    "(D-10-15).\\n")

for p in PIPELINES:
    lines.append(f"\\n## Pipeline {p}\\n")
    lines.append("| model | parameter_count | OD-EMD (mean±std) | OD-ACF lag-1 "
                 "| OD-DTW mean | transformed-EMD (Pipeline B) | TSTR-lite R² |")
    lines.append("|---|---|---|---|---|---|---|")
    for mk in MODEL_KINDS:
        emd_m, emd_s = _agg(mk, p, "emd", "OD")
        acf1_m, _ = _agg(mk, p, "acf_lag1_mean", "OD")
        dtw_m, _ = _agg(mk, p, "dtw_mean", "OD")
        temd_m, temd_s = _agg(mk, p, "emd", "transformed")
        t = tstr.get(f"{mk}|{p}")
        emd_c = f"{emd_m:.4f} ± {emd_s:.4f}" if emd_m is not None else "—"
        acf_c = f"{acf1_m:.4f}" if acf1_m is not None else "—"
        dtw_c = f"{dtw_m:.4f}" if dtw_m is not None else "—"
        temd_c = (f"{temd_m:.4f} ± {temd_s:.4f}"
                  if temd_m is not None else "— (n/a for Pipeline A)")
        tstr_c = (f"{t['r2_mean']:.3f} ± {t['r2_std']:.3f}"
                  if t is not None else "—")
        lines.append(f"| {mk} | {param_of.get(mk, '—')} | {emd_c} | {acf_c} "
                     f"| {dtw_c} | {temd_c} | {tstr_c} |")

rb = tstr["real_only_baseline"]
lines.append(
    f"\\n_TSTR-lite real-only reference: R² = {rb['r2_mean']:.3f} ± "
    f"{rb['r2_std']:.3f} (train on real_windowed_OD[320:], eval on [:320], "
    "init seeds {40,41,42})._\\n")
lines.append(
    "\\n_No recommendation is made here: per **D-10-19** Phase 14 owns the "
    "headline baseline decision, driven by Phase 11 utility numbers. This table "
    "is the apples-to-apples fidelity comparison only._\\n")

(out_dir / "baseline_comparison.md").write_text("\\n".join(lines) + "\\n")
print("wrote baseline_comparison.md")
print("\\n".join(lines[:14]))
"""))

# Cell 13 — final artifact presence check
CELLS.append(code("""\
required = [
    "revision/results/baseline_comparison.json",
    "revision/results/baseline_comparison.md",
    "revision/results/baseline_classical_wgan.json",
    "revision/results/baseline_nonadversarial.json",
]
print("Final artifact check:")
for r in required:
    sz = Path(r).stat().st_size
    print(f"  {sz:>10} bytes  {r}")
    assert sz > 0, r
print("\\nALL BASE-01/02/03 ARTIFACTS PRESENT")
"""))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH}: {len(CELLS)} cells")
