"""CLI driver: EVAL-05 scale-tagged fidelity re-emit (Phase 11).

One process per invocation. Idempotent — re-running overwrites
``revision/results/fidelity_dualscale.json`` cleanly.

Re-emits every ``revision.core.eval`` fidelity metric (EMD, moments, ACF, DTW)
with an explicit ``scale: "OD" | "log_return"`` field, for all 60 frozen sample
artifacts (6 model_kinds x 2 pipelines x 5 seeds). NO regeneration (D-11-08):
the frozen ``samples.npy`` / ``inverse_kwargs.npz`` bundles are read-only. The
fidelity math is reused UNCHANGED from ``revision/core/eval.py`` (D-11-10);
only this driver-level scale-tagging loop is new.

Directly answers reviewer R1-m3 / R1-M3 ("which scale are metrics reported
on?") by making the scale explicit on every metric.

Per-scale contract (D-11-09):
* OD scale: every metric for every (model_kind, pipeline, seed).
* log_return scale: Pipeline B emits real values (log-return space);
  Pipeline A emits explicit ``value: null`` rows with a ``scale_na_reason``
  so the schema stays rectangular and the gap is visible (RESEARCH Open
  Question 3 — emit explicit nulls, never silently omit).

Key contracts:
    D-11-08 — no regeneration; frozen artifacts only.
    D-11-09 — both Pipeline A and B; EVAL-05 dual-scale on every metric.
    D-11-10 — ``revision.core.eval`` reused UNCHANGED (imported, never edited);
              ``git diff --stat revision/core/`` must be empty.

Pitfall 4 (RESEARCH): the data_hash assert loop covers the 50 baseline
configs ONLY. The 09.1 quantum runs wrote no ``data_hash`` field; their data
equivalence is BY CONSTRUCTION (same ``load_and_preprocess(data.csv)`` OD
tensor) — no grep on quantum configs.

Pitfall 6 (RESEARCH): ``--csv-path`` defaults to ``./data.csv`` but is
resolved to an absolute repo-root path; the recomputed data_hash MUST equal
``91e447d4624e25b3``.

Frozen-artifact resolution (Rule 3 deviation — worktree execution):
``revision/results/`` is git-ignored (``.gitignore:62``); the baseline run
bundles under ``revision/results/baselines/runs/`` are NOT tracked and are
absent from a fresh worktree checkout. Since D-11-08 forbids regeneration, the
run-dir resolver falls back to the canonical primary checkout when a frozen
bundle is missing in the current tree. The quantum
``transform_ablation/runs/`` bundles ARE git-tracked (force-added) and present
locally. ``revision/core/`` is never touched.

Usage
-----
    python -m revision.run_dualscale_fidelity \\
        [--out revision/results] [--csv-path ./data.csv]

Writes ``revision/results/fidelity_dualscale.json`` (long-form dual-scale
rows[] with an explicit ``scale`` field on every row).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys

import numpy as np
import torch


def _bootstrap_repo_on_path() -> Path:
    """Ensure the repo root is importable when run as a bare script.

    ``python revision/run_dualscale_fidelity.py`` does not put the repo root
    on ``sys.path`` (only the script's own dir). Walk up to the dir holding
    ``revision/core/preprocessing.py`` and prepend it (same bootstrap the
    notebook generators use).
    """
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_bootstrap_repo_on_path()

from revision.core import WINDOW_LENGTH
from revision.core.data import load_and_preprocess, rolling_window

# revision.core.eval helpers — REUSE UNCHANGED (D-11-10). Import only; the
# eval module is NEVER edited by this driver.
from revision.core.eval import (
    compute_acf,
    compute_dtw,
    compute_emd,
    compute_moments,
)
from revision.core.preprocessing import inverse_logreturns

# ── Constants (locked across all Phase 11 drivers — 11-PATTERNS) ─────────────
MODEL_KINDS = ["quantum", "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
PIPELINES = ["A", "B"]  # Pipeline C dropped (D-10-05)
SEEDS = [42, 43, 44, 45, 46]
EXPECTED_DATA_HASH = "91e447d4624e25b3"
NLAGS = 9
DTW_N_PAIRS = 100  # matches 09.1 / baseline notebook (bounded runtime)

# Baseline run dirs are git-ignored (results/) and may be absent from a fresh
# worktree checkout. D-11-08 forbids regeneration, so fall back to the canonical
# primary checkout where the frozen bundles live. The quantum
# transform_ablation runs are git-tracked and present in-tree.
_CANONICAL_REPO_FALLBACK = Path("/Users/shawngibford/dev/phd/qGAN")


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from load_and_preprocess.

    Identical contract to ``run_baselines._compute_data_hash`` — pins the
    metric run to the exact OD tensor the frozen samples trained against.
    """
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _resolve_run_dir(rel: Path) -> Path:
    """Resolve a frozen run dir, falling back to the canonical checkout.

    ``rel`` is repo-root-relative. Returns the in-tree path if it exists,
    else the canonical primary-checkout path (Rule 3 — git-ignored frozen
    artifacts absent from worktree; D-11-08 forbids regeneration).
    """
    repo = _find_repo_root()
    in_tree = repo / rel
    if in_tree.exists():
        return in_tree
    fallback = _CANONICAL_REPO_FALLBACK / rel
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"frozen run dir not found in worktree ({in_tree}) nor canonical "
        f"checkout ({fallback}); D-11-08 forbids regeneration"
    )


def _run_base(model_kind: str, pipeline: str, seed: int) -> Path:
    """Resolve the frozen run dir for (model_kind, pipeline, seed).

    Quantum reuses the 09.1 ``transform_ablation`` layout (D-10-18); the 5
    classical baselines use the Phase-10 ``baselines`` layout (D-10-14).
    Path identity is byte-verbatim with ``_build_baseline_notebook.py:167-172``;
    only ``_resolve_run_dir`` wraps it for worktree fallback.
    """
    if model_kind == "quantum":
        rel = Path(f"revision/results/transform_ablation/runs/{pipeline}/{seed}")
    else:
        rel = Path(
            f"revision/results/baselines/runs/{model_kind}/{pipeline}/{seed}"
        )
    return _resolve_run_dir(rel)


def reconstruct_od(model_kind: str, pipeline: str, seed: int,
                   n_synth_subsample: int | None = None) -> dict:
    """Reconstruct OD-scale windows from a frozen sample bundle.

    Copied VERBATIM from ``revision/_build_baseline_notebook.py:175-210``
    (identical to Plan 01/02). The Pipeline-B
    ``np.random.default_rng(seed*7919+1)`` od_start draw is load-bearing — do
    NOT change it. Returns ``{"od_samples", "transformed", "n_synth",
    "pipeline", "seed"}``; ``transformed`` is the log-return array for
    Pipeline B (EVAL-05 ``scale="log_return"``) and ``None`` for Pipeline A.
    """
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


def verify_data_hash(csv_path: Path) -> dict:
    """Recompute data_hash; assert == EXPECTED across the 50 baseline configs.

    Pitfall 4: the assert loop covers the 50 baseline configs ONLY. The 09.1
    quantum runs wrote no ``data_hash`` field — their equivalence is BY
    CONSTRUCTION (same ``load_and_preprocess(data.csv)`` OD tensor); no grep on
    quantum configs.
    """
    import yaml

    recomputed = _compute_data_hash(csv_path)
    assert recomputed == EXPECTED_DATA_HASH, (
        f"data_hash {recomputed} != expected {EXPECTED_DATA_HASH}"
    )

    new_models = ["wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
    checked = 0
    mismatches = []
    for m in new_models:
        for p in PIPELINES:
            for s in SEEDS:
                cfg_dir = _run_base(m, p, s)
                cfg = yaml.safe_load((cfg_dir / "config.yaml").read_text())
                checked += 1
                if cfg.get("data_hash") != recomputed:
                    mismatches.append((m, p, s, cfg.get("data_hash")))
    assert checked == 50, f"expected 50 baseline configs, checked {checked}"
    assert not mismatches, f"data_hash mismatch (report loudly): {mismatches}"
    quantum_equiv_note = (
        "Quantum (09.1) data equivalence established BY CONSTRUCTION: the 09.1 "
        "transform_ablation runs used the identical load_and_preprocess("
        "data.csv) OD tensor; 09.1 wrote no data_hash field, so no grep is "
        "performed (RESEARCH Pitfall 4)."
    )
    return {
        "recomputed_from": (
            "load_and_preprocess('data.csv')['OD'].tobytes() sha256[:16]"
        ),
        "n_baseline_configs_checked": checked,
        "all_equal": True,
        "quantum_equivalence": quantum_equiv_note,
    }


def build_real_references(csv_path: Path) -> dict:
    """Build the real OD + log-return reference arrays.

    Reuses the EXACT construction ``_build_baseline_notebook.py`` Cell-3/6
    uses: ``real_windowed_OD`` (rolling_window of OD, step 2) flattened for the
    OD-scale EMD/DTW reference, and ``real_log_delta`` (= ``d_real["log_delta"]``)
    for the Pipeline-B ``scale="log_return"`` EMD reference — so the numbers
    reconcile with ``baseline_comparison.json``'s ``scale="transformed"`` rows.
    """
    d_real = load_and_preprocess(str(csv_path))
    real_windowed_OD = rolling_window(
        d_real["OD"], WINDOW_LENGTH, 2
    ).cpu().numpy()
    real_log_delta = d_real["log_delta"].cpu().numpy()
    return {
        "real_windowed_OD": real_windowed_OD,
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": real_log_delta,
    }


def main() -> None:
    """CLI entry point — emit revision/results/fidelity_dualscale.json."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 11 EVAL-05 scale-tagged fidelity re-emit "
            "(reuses revision.core.eval UNCHANGED — D-11-10)."
        )
    )
    ap.add_argument(
        "--out", type=Path, default=Path("revision/results"),
        help="Directory for fidelity_dualscale.json.",
    )
    ap.add_argument(
        "--csv-path", type=Path, default=Path("./data.csv"),
        help="Path to the OD CSV (same data the frozen runs used).",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    # Pitfall 6: resolve --csv-path to an absolute repo-root path.
    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = (repo / csv_path).resolve()

    recomputed = _compute_data_hash(csv_path)
    print(f"[run_dualscale_fidelity] recomputed data_hash: {recomputed}")
    assert recomputed == EXPECTED_DATA_HASH, (
        f"data_hash {recomputed} != expected {EXPECTED_DATA_HASH}"
    )

    dh_verification = verify_data_hash(csv_path)
    print(
        f"[run_dualscale_fidelity] data_hash invariant OK: all 50 baseline "
        f"configs == {recomputed}"
    )

    refs = build_real_references(csv_path)
    real_OD_flat = refs["real_OD_flat"]
    real_windowed_OD = refs["real_windowed_OD"]
    real_log_delta = refs["real_log_delta"]
    print(
        f"[run_dualscale_fidelity] real refs: windowed_OD="
        f"{real_windowed_OD.shape}, log_delta={real_log_delta.shape}"
    )

    rows = emit_rows(real_OD_flat, real_windowed_OD, real_log_delta)

    out_dir = (
        args.out if args.out.is_absolute() else (repo / args.out)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "long-form dual-scale rows[] (EVAL-05, D-11-09)",
        "model_kinds": MODEL_KINDS,
        "pipelines": PIPELINES,
        "seeds": SEEDS,
        "data_hash": EXPECTED_DATA_HASH,
        "data_hash_verification": dh_verification,
        "metric_helpers": {
            "functions": [
                "revision.core.eval.compute_emd",
                "revision.core.eval.compute_moments",
                "revision.core.eval.compute_acf",
                "revision.core.eval.compute_dtw",
            ],
            "note": "reused unchanged, D-11-10",
        },
        "rows": rows,
    }
    out_path = out_dir / "fidelity_dualscale.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(
        f"[run_dualscale_fidelity] wrote {out_path} ({len(rows)} rows, "
        f"scales={sorted({r['scale'] for r in rows})})"
    )


_PIPELINE_A_NA_REASON = (
    "Pipeline A is raw-OD; no log-return space (D-11-09)"
)


def _od_scale_rows(mk: str, p: str, s: int, od: np.ndarray,
                    real_OD_flat: np.ndarray,
                    real_windowed_OD: np.ndarray) -> list:
    """Emit every eval.py metric on the OD scale for one run.

    Structure copied from ``_build_baseline_notebook.py:249-284`` (same metric
    suite + DTW sub-sample recipe). NO new metric math (D-11-10). Every row
    carries ``scale="OD"``.
    """
    out = []
    synth_flat = od.reshape(-1)
    # EMD on OD scale (pooled across windows)
    out.append(dict(model_kind=mk, pipeline=p, seed=s, metric_name="emd",
                     scale="OD", value=compute_emd(real_OD_flat, synth_flat)))
    # Moments (mean/std/skewness/kurtosis) — each its own metric_name
    for k, v in compute_moments(synth_flat).items():
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"moment_{k}", scale="OD", value=v))
    # ACF per window aggregated (lags 0..NLAGS), mean + std
    acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in od])
    for lag in range(NLAGS + 1):
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_mean", scale="OD",
                         value=float(acfs[:, lag].mean())))
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_std", scale="OD",
                         value=float(acfs[:, lag].std())))
    # DTW nearest-neighbor sub-sampled — verbatim recipe (DTW_N_PAIRS=100,
    # np.random.default_rng(s*31)); do NOT change seed or pair count (T-11-03).
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
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_mean", scale="OD",
                     value=float(np.mean(dtw_vals))))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_median", scale="OD",
                     value=float(np.median(dtw_vals))))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_std", scale="OD",
                     value=float(np.std(dtw_vals))))
    return out


def _log_return_rows(mk: str, p: str, s: int, r: dict,
                      real_log_delta: np.ndarray) -> list:
    """Emit the SAME metric set on the log-return scale.

    Pipeline B: real values vs ``r["transformed"]`` (log-return array). The
    real reference is ``real_log_delta`` (= ``d_real["log_delta"]``) — the
    EXACT array ``_build_baseline_notebook.py:290`` uses for its
    ``scale="transformed"`` EMD, so numbers reconcile with
    ``baseline_comparison.json``.

    Pipeline A: explicit ``value: null`` rows with ``scale_na_reason`` — no
    silent omission (T-11-08 / T-11-09, RESEARCH Open Question 3). The metric
    name set mirrors the OD-scale set so the schema stays rectangular.
    """
    metric_names = ["emd"]
    metric_names += [f"moment_{k}"
                     for k in ("mean", "std", "skewness", "kurtosis")]
    for lag in range(NLAGS + 1):
        metric_names += [f"acf_lag{lag}_mean", f"acf_lag{lag}_std"]
    metric_names += ["dtw_mean", "dtw_median", "dtw_std"]

    if p == "A" or r["transformed"] is None:
        return [
            dict(model_kind=mk, pipeline=p, seed=s, metric_name=name,
                 scale="log_return", value=None,
                 scale_na_reason=_PIPELINE_A_NA_REASON)
            for name in metric_names
        ]

    out = []
    trans = r["transformed"]
    trans_flat = trans.reshape(-1)
    out.append(dict(model_kind=mk, pipeline=p, seed=s, metric_name="emd",
                     scale="log_return",
                     value=compute_emd(real_log_delta, trans_flat)))
    for k, v in compute_moments(trans_flat).items():
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"moment_{k}", scale="log_return",
                         value=v))
    acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in trans])
    for lag in range(NLAGS + 1):
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_mean",
                         scale="log_return",
                         value=float(acfs[:, lag].mean())))
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_std",
                         scale="log_return",
                         value=float(acfs[:, lag].std())))
    # DTW on log-return windows — verbatim recipe (rng seed s*31, 100 pairs).
    rng = np.random.default_rng(s * 31)
    synth_idx = rng.choice(trans.shape[0],
                           size=min(DTW_N_PAIRS, trans.shape[0]),
                           replace=False)
    # Real log-return reference reshaped into NLAGS+1-length windows for the
    # DTW nearest-neighbour pairing (same window length as OD windows).
    wl = trans.shape[1]
    n_full = real_log_delta.shape[0] // wl
    real_lr_windows = real_log_delta[: n_full * wl].reshape(n_full, wl)
    real_idx = rng.choice(real_lr_windows.shape[0],
                          size=min(64, real_lr_windows.shape[0]),
                          replace=False)
    dtw_vals = []
    for i in synth_idx:
        best = min(compute_dtw(trans[i], real_lr_windows[j])
                   for j in real_idx)
        dtw_vals.append(best)
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_mean", scale="log_return",
                     value=float(np.mean(dtw_vals))))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_median", scale="log_return",
                     value=float(np.median(dtw_vals))))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_std", scale="log_return",
                     value=float(np.std(dtw_vals))))
    return out


def emit_rows(real_OD_flat, real_windowed_OD, real_log_delta) -> list:
    """Scale-tagged metric loop over all 60 frozen artifacts.

    For each (model_kind, pipeline, seed): reconstruct OD, emit every
    ``revision.core.eval`` metric on ``scale="OD"``, then the same metric set
    on ``scale="log_return"`` (real values for Pipeline B; explicit nulls with
    a reason for Pipeline A). NO new metric math (D-11-10).
    """
    rows = []
    for mk in MODEL_KINDS:
        for p in PIPELINES:
            for s in SEEDS:
                print(f"  metrics: model={mk} pipeline={p} seed={s}")
                r = reconstruct_od(mk, p, s)
                od = r["od_samples"]
                rows.extend(
                    _od_scale_rows(mk, p, s, od, real_OD_flat,
                                   real_windowed_OD)
                )
                rows.extend(
                    _log_return_rows(mk, p, s, r, real_log_delta)
                )
    return rows


if __name__ == "__main__":
    main()
