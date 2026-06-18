"""Phase 11 utility driver — EVAL-01 (TSTR soft-sensor) + EVAL-04 (Orlandi-style
real-only-vs-augmented lift) on the single shared one-step-ahead OD-forecast
downstream task (D-11-01 / D-11-06).

This driver is a pure *consumer* (D-11-08): it reads the 60 frozen `samples.npy`
artifacts (50 Phase 10 baseline runs + 10 Phase 09.1 quantum runs), reconstructs
OD-scale windows with the *verbatim* `reconstruct_od` recipe copied from
`_build_baseline_notebook.py`, trains the `TSTRLiteLSTM` soft-sensor
(also verbatim, D-11-10: NOT promoted to `core/`), and writes two
long-form JSON artifacts:

  * `results/tstr.json`        — EVAL-01 (R2/MAE/RMSE + Phase-10 anchor)
  * `results/augmentation.json` — EVAL-04 (mixing-ratio lift curve)

Pattern source: `run_baselines.py` (CLI/driver shape) +
`_build_baseline_notebook.py:167-501` (recon + TSTR logic, copied
verbatim — the Pipeline-B seeded `od_start` draw is load-bearing, RESEARCH
Pitfall 3; sklearn is ABSENT so R2/MAE/RMSE are computed inline, Pitfall 2).

`core/` is NEVER modified by this driver (D-11-10 / D-10-13 invariant).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (RESEARCH Pitfall 6 — `data.csv` is repo-root relative;
# the driver may run from a worktree / arbitrary cwd). Mirrors the
# `_find_repo_root()` preamble of `_build_baseline_notebook.py:74-93`.
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    # Fallback: walk up from cwd (covers exotic invocations).
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (core/preprocessing.py)")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.preprocessing import inverse_logreturns  # noqa: E402
from core.data import load_and_preprocess, rolling_window  # noqa: E402
from core import WINDOW_LENGTH  # noqa: E402  (== 10)
from core._wgan_unscale import _unscale_wgan_samples  # noqa: E402  Plan 14-21 T01


# ─────────────────────────────────────────────────────────────────────────────
# Module constants (matched-budget driver mode, 14-20):
# 9 trainable model_kinds × 1 pipeline (B) × 5 seeds = 45 frozen run dirs under
# results/matched2000/runs/<model_kind>/<seed>/. All 45 cells carry the
# canonical data_hash 91e447d4624e25b3 directly (no quantum-by-construction
# shortcut needed — the matched2000 sweep wrote data_hash into every config.yaml).
# ─────────────────────────────────────────────────────────────────────────────
MODEL_KINDS = ["iqp_sel_55_repro", "V1", "V2", "V3",
               "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
PIPELINES = ["B"]                      # Pipeline B only (matched-budget driver mode, 14-20)
SEEDS = [42, 43, 44, 45, 46]           # 5 training seeds
INIT_SEEDS = (40, 41, 42)              # LSTM init seeds (NOT training seeds)
HELD_OUT_N = 320                       # D-10-21 real held-out split
EXPECTED_DATA_HASH = "91e447d4624e25b3"  # cross-phase invariant (D-10-15)

# Matched-budget driver mode: every cell under matched2000/runs/<mk>/<seed>/
# carries the canonical data_hash in its config.yaml (verified 2026-05-24
# subagent sweep audit) — the assert loop iterates all 9 model_kinds × 5 seeds
# = 45 cells (no quantum-by-construction shortcut).
ALL_MODELS = ["iqp_sel_55_repro", "V1", "V2", "V3",
              "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]


# ─────────────────────────────────────────────────────────────────────────────
# Data-hash invariant (copied from `run_baselines.py:226-234` + the assert loop
# from `_build_baseline_notebook.py:124-149`). T-11-01 mitigation.
# ─────────────────────────────────────────────────────────────────────────────
def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from `load_and_preprocess` (D-10-15)."""
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _assert_data_hash_invariant(csv_path: Path) -> tuple[str, str]:
    """Recompute the OD-tensor hash and assert it == EXPECTED_DATA_HASH AND ==
    every one of the 45 matched-budget `config.yaml` `data_hash` fields (9
    trainable model_kinds × 1 pipeline (B) × 5 seeds). Every cell under
    matched2000/runs/<mk>/<seed>/ carries the canonical data_hash directly —
    no quantum-by-construction shortcut needed (14-20).

    Returns (recomputed_hash, matched_budget_note).
    """
    recomputed = _compute_data_hash(csv_path)
    assert recomputed == EXPECTED_DATA_HASH, (
        f"data_hash drift: recomputed {recomputed!r} != "
        f"expected {EXPECTED_DATA_HASH!r} (RESEARCH Pitfall 6 — wrong csv?)"
    )
    checked = 0
    mismatches: list[tuple[str, int, object]] = []
    for m in ALL_MODELS:
        for s in SEEDS:
            cfg_path = (
                REPO / "results" / "matched2000" / "runs"
                / m / str(s) / "config.yaml"
            )
            cfg = yaml.safe_load(cfg_path.read_text())
            checked += 1
            if cfg.get("data_hash") != recomputed:
                mismatches.append((m, s, cfg.get("data_hash")))
    assert checked == 45, f"expected 45 matched-budget configs, checked {checked}"
    assert not mismatches, (
        f"data_hash mismatch (UNEXPECTED — report loudly): {mismatches}"
    )
    matched_budget_note = (
        "Matched-budget driver mode (14-20): all 45 cells under "
        "results/matched2000/runs/<model_kind>/<seed>/ carry the "
        "canonical data_hash 91e447d4624e25b3 in their config.yaml directly. "
        "The 4 quantum variants (iqp_sel_55_repro, V1, V2, V3) and 5 classical "
        "baselines (wgan_mlp, wgan_cnn, wgan_lstm, vae, ar) all assert by-value, "
        "not by-construction; no shortcut applied (D-14-10 + 14-20)."
    )
    return recomputed, matched_budget_note


# ─────────────────────────────────────────────────────────────────────────────
# Run-dir resolver + `reconstruct_od` — copied VERBATIM from
# `_build_baseline_notebook.py:167-210` (RESEARCH Pitfall 3 — the Pipeline-B
# `np.random.default_rng(seed*7919+1)` od_start draw is load-bearing; do NOT
# refactor). The only adaptation: anchor `base` at the resolved REPO root so the
# driver works from any cwd (Pitfall 6).
# ─────────────────────────────────────────────────────────────────────────────
def _run_base(model_kind: str, pipeline: str, seed: int) -> Path:
    # Matched-budget driver mode (14-20): single branch, all 9 trainable
    # model_kinds resolve to results/matched2000/runs/<mk>/<seed>/.
    # No pipeline subdir (Pipeline B is implicit; an A-branch lookup raises in
    # reconstruct_od via the NotImplementedError guard below).
    return REPO / "results" / "matched2000" / "runs" / model_kind / str(seed)


def reconstruct_od(model_kind: str, pipeline: str, seed: int,
                   n_synth_subsample: int | None = None) -> dict:
    # 14-20 guard: matched-budget driver mode is Pipeline B only. Pipeline A
    # has no matched2000 artefacts — every cell under matched2000/runs/ was
    # trained as Pipeline B with log-return transform (see config.yaml). The
    # Pipeline-A branch below is retained as dead-but-not-deleted reference
    # but is unreachable in matched-budget driver mode.
    if pipeline != "B":
        raise NotImplementedError(
            f"Matched-budget driver mode supports Pipeline B only; "
            f"got pipeline={pipeline!r}. The matched2000 sweep "
            f"(results/matched2000/runs/) has no Pipeline-A cells. "
            f"To run utility against legacy Pipeline-A artefacts, revert "
            f"_run_base() and _assert_data_hash_invariant() to the pre-14-20 "
            f"layout."
        )
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model_kind)  # Plan 14-21 T01
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


# ─────────────────────────────────────────────────────────────────────────────
# Real windowed OD + held-out split (copied from
# `_build_baseline_notebook.py:104-110, :458-462`).
# ─────────────────────────────────────────────────────────────────────────────
def _real_windowed_od(csv_path: Path) -> np.ndarray:
    d_real = load_and_preprocess(str(csv_path))
    # shape == ((len(OD)-WINDOW_LENGTH)//2 + 1, WINDOW_LENGTH) == (385, 10);
    # downstream [HELD_OUT_N:] with HELD_OUT_N=320 leaves 65 training windows.
    return rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()


def _resolve_csv(csv_arg: Path) -> Path:
    """Resolve --csv-path to an ABSOLUTE repo-root-anchored path (Pitfall 6)."""
    csv_arg = Path(csv_arg)
    if csv_arg.is_absolute():
        return csv_arg
    # Anchor a relative path (e.g. the default `./data.csv`) at the repo root.
    return (REPO / csv_arg).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# TSTR soft-sensor — copied VERBATIM from `_build_baseline_notebook.py:395-440`,
# then MAE/RMSE added to the return dict (sklearn ABSENT — Pitfall 2). `mse`/`r2`
# kept byte-identical so Phase-10 numbers reproduce exactly.
# ─────────────────────────────────────────────────────────────────────────────
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
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_eval_tstr(train_windows: np.ndarray, eval_windows: np.ndarray,
                    lstm_seed: int = 40, hidden: int = 32,
                    epochs: int = 50, bs: int = 64) -> dict:
    torch.manual_seed(lstm_seed)
    rng = np.random.default_rng(lstm_seed)
    # WR-02: fail loudly at the true cause if a future eval slice is degenerate
    # (zero-variance target) — otherwise R2 is undefined and would masquerade as
    # a confusing <0 leakage-sentinel failure. Current data is non-degenerate so
    # every computed number stays byte-identical.
    assert eval_windows[:, 9:10].std() > 0.0, \
        "degenerate (zero-variance) eval target — TSTR R2 undefined"
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
        # PHASE 11 additions for ROADMAP SC-1 (R2/MAE/RMSE) — sklearn ABSENT.
        "mae": float(np.mean(np.abs(yev_np - yev_pred))),
        "rmse": float(np.sqrt(np.mean((yev_np - yev_pred) ** 2))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Long-form JSON header (mirrors `results/baseline_comparison.json`
# top-level keys — RESEARCH Pattern 4).
# ─────────────────────────────────────────────────────────────────────────────
def _json_header(schema: str, data_hash: str, matched_budget_note: str) -> dict:
    return {
        "schema": schema,
        "model_kinds": MODEL_KINDS,
        "pipelines": PIPELINES,
        "seeds": SEEDS,
        "data_hash": data_hash,
        "data_hash_verification": {
            "recomputed_from": "load_and_preprocess('data.csv')['OD'].tobytes() sha256[:16]",
            "n_matched_budget_configs_checked": 45,
            "all_equal": True,
            "matched_budget_note": matched_budget_note,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-01 — TSTR roll-up (copied structure from
# `_build_baseline_notebook.py:464-501`; adds MAE/RMSE aggregates).
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(per_seed: list[dict], keys=("mse", "r2", "mae", "rmse")) -> dict:
    agg: dict = {}
    for k in keys:
        vals = [x[k] for x in per_seed]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def run_tstr(csv_path: Path, out_dir: Path, epochs: int,
             init_seeds: tuple[int, ...]) -> dict:
    """EVAL-01: train soft-sensor on synthetic OD windows pooled across the 5
    training seeds, evaluate on the 320 held-out real windows. Reproduces the
    Phase-10 `real_only_baseline` anchor (n_train_real=65, R2 ~ -13.354)."""
    data_hash, matched_budget_note = _assert_data_hash_invariant(csv_path)
    real_windowed_OD = _real_windowed_od(csv_path)
    real_eval = real_windowed_OD[:HELD_OUT_N]
    real_train_for_baseline = real_windowed_OD[HELD_OUT_N:]
    print(f"[tstr] real_eval={real_eval.shape}, "
          f"real_train_for_baseline={real_train_for_baseline.shape} "
          f"(n_train_real={real_train_for_baseline.shape[0]})")

    tstr: dict = {}
    rows: list[dict] = []
    for mk in MODEL_KINDS:
        for p in PIPELINES:
            print(f"[tstr] model={mk} pipeline={p} ...")
            synth_pool = np.concatenate(
                [reconstruct_od(mk, p, s)["od_samples"] for s in SEEDS], axis=0
            )
            per_seed = []
            for ls in init_seeds:
                res = train_eval_tstr(synth_pool, real_eval, lstm_seed=ls,
                                      epochs=epochs)
                per_seed.append(res)
                for metric_name in ("r2", "mae", "rmse", "mse"):
                    rows.append({
                        "model_kind": mk, "pipeline": p, "seed": int(ls),
                        "metric_name": metric_name, "scale": "OD",
                        "value": float(res[metric_name]),
                    })
                print(f"    init_seed={ls} mse={res['mse']:.4f} "
                      f"r2={res['r2']:.3f} mae={res['mae']:.4f} "
                      f"rmse={res['rmse']:.4f}")
            blk = {
                "model_kind": mk,
                "pipeline": p,
                "n_train_synth": int(synth_pool.shape[0]),
                "n_eval_real": HELD_OUT_N,
                "per_init_seed": {str(k): v for k, v in zip(init_seeds, per_seed)},
            }
            blk.update(_aggregate(per_seed))
            tstr[f"{mk}|{p}"] = blk

    print("[tstr] real-only baseline ...")
    per_seed_real = [
        train_eval_tstr(real_train_for_baseline, real_eval, lstm_seed=ls,
                        epochs=epochs)
        for ls in init_seeds
    ]
    rob = {
        "n_train_real": int(real_train_for_baseline.shape[0]),
        "n_eval_real": HELD_OUT_N,
        "per_init_seed": {str(k): v for k, v in zip(init_seeds, per_seed_real)},
    }
    rob.update(_aggregate(per_seed_real))
    tstr["real_only_baseline"] = rob
    print(f"[tstr] real_only_baseline R2={rob['r2_mean']:.3f} "
          f"+/- {rob['r2_std']:.3f} (Phase-10 anchor ~ -13.354 +/- 0.583)")

    obj = _json_header(
        "long-form rows[] + tstr block (D-10-21, EVAL-01, matched-budget 14-20)",
        data_hash, matched_budget_note,
    )
    obj["soft_sensor"] = (
        "TSTRLiteLSTM (1-layer LSTM, hidden=32) copied verbatim from "
        "_build_baseline_notebook.py:395-440; MAE/RMSE added (sklearn ABSENT, "
        "RESEARCH Pitfall 2). One-step-ahead OD forecast: X=window[:,:9], "
        "y=window[:,9:10] (D-11-01)."
    )
    obj["init_seeds"] = list(init_seeds)
    obj["epochs"] = int(epochs)
    obj["rows"] = rows
    obj["tstr"] = tstr

    out_path = out_dir / "tstr_matched2000.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"[tstr] wrote {out_path} ({len(rows)} rows, "
          f"{len(tstr)} aggregate blocks)")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-04 — Orlandi-style augmentation mixing-ratio lift curve (D-11-06/07).
# ─────────────────────────────────────────────────────────────────────────────
# Injection grid: synthetic windows appended = round(ratio * |real_train|).
_INJECTION_GRID = {"+25%": 0.25, "+50%": 0.50, "+100%": 1.0}


def run_augmentation(csv_path: Path, out_dir: Path, epochs: int,
                     init_seeds: tuple[int, ...]) -> dict:
    """EVAL-04: reuse the SAME one-step-ahead OD soft-sensor and the SAME
    `real_eval = real_windowed_OD[:320]` evaluation set. Conditions per
    (model_kind, pipeline): `real_only`, `+25%`, `+50%`, `+100%`,
    `synthetic_only`. Delta table = (augmented metric) - (real_only metric).

    Partition-disjointness guard (RESEARCH Pitfall 5 / T-11-02): the real
    portion mixed in is ALWAYS `real_windowed_OD[320:]`, NEVER `[:320]`.
    """
    data_hash, matched_budget_note = _assert_data_hash_invariant(csv_path)
    real_windowed_OD = _real_windowed_od(csv_path)
    n_total = real_windowed_OD.shape[0]
    eval_idx = set(range(0, HELD_OUT_N))
    train_idx = set(range(HELD_OUT_N, n_total))
    # T-11-02 leakage guard — assert eval/train index sets are disjoint.
    assert eval_idx.isdisjoint(train_idx), (
        f"LEAKAGE: eval/train partition overlap {eval_idx & train_idx}"
    )
    real_eval = real_windowed_OD[:HELD_OUT_N]
    real_train_for_baseline = real_windowed_OD[HELD_OUT_N:]
    n_real_train = real_train_for_baseline.shape[0]
    print(f"[aug] real_eval={real_eval.shape}, "
          f"real_train={real_train_for_baseline.shape} "
          f"(n_real_train={n_real_train}); partition-disjoint OK")

    rows: list[dict] = []
    lift: dict = {}
    for mk in MODEL_KINDS:
        for p in PIPELINES:
            print(f"[aug] model={mk} pipeline={p} ...")
            # Pool synthetic OD windows across the 5 training seeds {42..46}.
            synth_pool = np.concatenate(
                [reconstruct_od(mk, p, s)["od_samples"] for s in SEEDS], axis=0
            )

            def _mean_metrics(train_windows: np.ndarray) -> dict:
                ps = [
                    train_eval_tstr(train_windows, real_eval, lstm_seed=ls,
                                    epochs=epochs)
                    for ls in init_seeds
                ]
                return {
                    "r2": float(np.mean([x["r2"] for x in ps])),
                    "mae": float(np.mean([x["mae"] for x in ps])),
                    "rmse": float(np.mean([x["rmse"] for x in ps])),
                }

            # ── real_only condition (= the Phase-10 anchor) ──────────────────
            base_metrics = _mean_metrics(real_train_for_baseline)
            print(f"    real_only R2={base_metrics['r2']:.3f} "
                  f"(Phase-10 anchor ~ -13.354; positive => LEAKAGE)")
            conditions: dict = {
                "real_only": {
                    "injection_ratio": "real_only",
                    "n_train": int(n_real_train),
                    "n_synth_added": 0,
                    "metrics": base_metrics,
                }
            }

            # ── real + synthetic @ +25% / +50% / +100% ──────────────────────
            for label, ratio in _INJECTION_GRID.items():
                n_synth = int(round(ratio * n_real_train))
                # WR-04: a shrunken synth_pool must fail loudly, never silently
                # collapse +100% into an identical-to-synthetic_only training
                # set. Always passes at the documented pool size (~3840).
                assert n_synth < synth_pool.shape[0], (
                    label, n_synth, synth_pool.shape,
                    "injection grid collapsed — +100% would equal synthetic_only",
                )
                # WR-03: collision-free subsample seed fully qualified by
                # (model_kind, pipeline, label) — the old int(ratio*1000)+1
                # was lossy and not qualified by run identity.
                seed_int = zlib.crc32(
                    f"augsub|{mk}|{p}|{label}".encode()
                ) & 0xFFFFFFFF
                sub_rng = np.random.default_rng(seed_int)
                if n_synth >= synth_pool.shape[0]:
                    synth_sel = synth_pool
                else:
                    sel_idx = sub_rng.choice(synth_pool.shape[0], n_synth,
                                             replace=False)
                    synth_sel = synth_pool[sel_idx]
                mixed = np.concatenate(
                    [real_train_for_baseline, synth_sel], axis=0
                )
                m = _mean_metrics(mixed)
                conditions[label] = {
                    "injection_ratio": label,
                    "ratio": float(ratio),
                    "n_train": int(mixed.shape[0]),
                    "n_synth_added": int(synth_sel.shape[0]),
                    "subsample_rng_seed": int(seed_int),
                    "subsample_rng_seed_derivation":
                        "crc32('augsub|{mk}|{p}|{label}') & 0xFFFFFFFF",
                    "metrics": m,
                }
                print(f"    {label} R2={m['r2']:.3f} "
                      f"(n_synth={synth_sel.shape[0]})")

            # ── synthetic_only condition ────────────────────────────────────
            so = _mean_metrics(synth_pool)
            conditions["synthetic_only"] = {
                "injection_ratio": "synthetic_only",
                "n_train": int(synth_pool.shape[0]),
                "n_synth_added": int(synth_pool.shape[0]),
                "metrics": so,
            }
            print(f"    synthetic_only R2={so['r2']:.3f}")

            # ── delta table vs the real_only anchor ─────────────────────────
            for label, cond in conditions.items():
                m = cond["metrics"]
                deltas = {
                    "r2_delta": m["r2"] - base_metrics["r2"],
                    "mae_delta": m["mae"] - base_metrics["mae"],
                    "rmse_delta": m["rmse"] - base_metrics["rmse"],
                }
                cond["deltas"] = deltas
                for dn, dv in deltas.items():
                    rows.append({
                        "model_kind": mk, "pipeline": p, "seed": 0,
                        "metric_name": dn, "scale": "OD",
                        "injection_ratio": label, "value": float(dv),
                    })
            lift[f"{mk}|{p}"] = {
                "model_kind": mk, "pipeline": p,
                "n_real_train": int(n_real_train),
                "real_only_metrics": base_metrics,
                "conditions": conditions,
            }

    obj = _json_header(
        "long-form rows[] + lift block (EVAL-04, Orlandi-style, matched-budget 14-20)",
        data_hash, matched_budget_note,
    )
    obj["soft_sensor"] = (
        "Same one-step-ahead OD TSTRLiteLSTM as EVAL-01 (D-11-06). Eval set is "
        "always real_windowed_OD[:320]; the real portion mixed in is always "
        "real_windowed_OD[320:] (T-11-02 leakage guard asserted)."
    )
    obj["init_seeds"] = list(init_seeds)
    obj["epochs"] = int(epochs)
    obj["injection_grid"] = {
        "real_only": "real_windowed_OD[320:] only (Phase-10 anchor)",
        "+25%": "real_train + round(0.25*|real_train|) synthetic windows",
        "+50%": "real_train + round(0.50*|real_train|) synthetic windows",
        "+100%": "real_train + round(1.00*|real_train|) synthetic windows",
        "synthetic_only": "full pooled synthetic (5 training seeds)",
    }
    obj["metadata"] = {
        "caveat": (
            "LOWER BOUND, NOT a matched-budget comparison. The pooled "
            "synthetic set (~3840 windows x 5 seeds) is ~60x larger than the "
            "real train partition (~65 windows). Per Phase 09.1 FLAG-E / "
            "RESEARCH Orlandi section, the reported lift is a lower bound on "
            "the achievable augmentation benefit and MUST NOT be read as a "
            "budget-matched gain."
        ),
        "partition_guard": (
            "T-11-02: eval=real_windowed_OD[:320], train=real_windowed_OD[320:]; "
            "index-set disjointness asserted at runtime (raises on overlap). "
            "A positive real_only R2 is the leakage sentinel (Pitfall 5)."
        ),
        "downstream_task": "one-step-ahead OD forecast (D-11-01/D-11-06)",
    }
    obj["rows"] = rows
    obj["lift"] = lift

    out_path = out_dir / "augmentation_matched2000.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"[aug] wrote {out_path} ({len(rows)} rows, "
          f"{len(lift)} (model|pipeline) lift blocks)")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint (driver shape from `run_baselines.py:430-454`).
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 11 utility driver — EVAL-01 TSTR + EVAL-04 augmentation."
    )
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help=("Output dir for tstr_matched2000.json / "
                          "augmentation_matched2000.json (matched-budget driver, 14-20)."))
    ap.add_argument("--csv-path", type=Path, default=Path("./data.csv"),
                    help="OD CSV (repo-root relative; resolved to absolute).")
    ap.add_argument("--epochs", type=int, default=50,
                    help="TSTR LSTM training epochs (Phase-10 default 50).")
    ap.add_argument("--init-seeds", type=str, default="40,41,42",
                    help="Comma-separated LSTM init seeds (NOT training seeds).")
    ap.add_argument("--tstr-only", action="store_true",
                    help="Run only EVAL-01 (tstr.json).")
    ap.add_argument("--augmentation-only", action="store_true",
                    help="Run only EVAL-04 (augmentation.json).")
    args = ap.parse_args()

    csv_path = _resolve_csv(args.csv_path)
    out_dir = args.out if args.out.is_absolute() else (REPO / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_seeds = tuple(int(x) for x in str(args.init_seeds).split(","))

    print(f"[run_utility] repo={REPO}")
    print(f"[run_utility] csv={csv_path} out={out_dir} "
          f"epochs={args.epochs} init_seeds={init_seeds}")

    run_tstr_flag = not args.augmentation_only
    run_aug_flag = not args.tstr_only

    if run_tstr_flag:
        run_tstr(csv_path, out_dir, args.epochs, init_seeds)
    if run_aug_flag:
        run_augmentation(csv_path, out_dir, args.epochs, init_seeds)

    print("[run_utility] done.")


if __name__ == "__main__":
    main()
