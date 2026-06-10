"""CLI driver: ARCH-02 ansatz comparison aggregator (Phase 13).

One process per invocation. Idempotent — re-running overwrites
``revision/results/ansatz_comparison.json`` cleanly.

Emits the ARCH-02 quantum-vs-quantum ansatz comparison on the extended
long-form ``rows[]`` + ``models[]`` schema (D-10-16) extended with
``ansatz``/``depth``/``topology`` dims (Phase 13). Clones the
``run_dualscale_fidelity.py`` envelope + dual-scale reconstruction; metrics are
computed via ``revision.core.eval.full_metric_suite`` UNCHANGED (D-10-20).

Variant provenance (D-13-01, D-10-15):
    * V1 (depth-4, range, 75 params) — REUSED, NOT recomputed-by-training. Its
      frozen 5-seed Pipeline-B sample bundles under
      ``revision/results/transform_ablation/runs/B/{42..46}`` are read and the
      SAME ``full_metric_suite`` is computed on those existing samples (no new
      training). The 09.1 quantum configs carry NO ``data_hash`` field; the
      data equivalence is BY CONSTRUCTION (identical
      ``load_and_preprocess(data.csv)`` OD tensor) and is recorded verbatim-in-
      spirit in the JSON (RESEARCH Pitfall 4 / run_dualscale_fidelity.py
      precedent).
    * V2 (depth-8, range, 135 params) — NEW Phase-13 runs under
      ``revision/results/ansatz/runs/V2/{42..46}``.
    * V3 (depth-4, linear, 75 params) — NEW Phase-13 runs under
      ``revision/results/ansatz/runs/V3/{42..46}``.

Dual-scale (RESEARCH Open Q1): every metric is emitted on both
``scale="log_return"`` and ``scale="OD"`` (OD via the Pipeline-B
``inverse_logreturns`` reconstruction, mirroring run_dualscale_fidelity.py).

Usage
-----
    python -m revision.run_ansatz_comparison \\
        [--out revision/results] [--csv-path ./data.csv] \\
        [--ansatz-root revision/results/ansatz]

Writes ``revision/results/ansatz_comparison.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap_repo_on_path() -> Path:
    """Ensure the repo root is importable when run as a bare script."""
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

# revision.core.eval — REUSE UNCHANGED (D-10-20). Import only; never edited.
from revision.core.eval import full_metric_suite
from revision.core.preprocessing import inverse_logreturns
from revision._wgan_unscale import _unscale_wgan_samples  # Plan 14-21 T01

# ── ARCH-02 ansatz variant registry ─────────────────────────────────────────
# V1 is REUSED from the 09.1/10 transform_ablation Pipeline-B runs (D-13-01, no
# recompute of training); V2/V3 are the Phase-13 new runs.
ANSATZ_VARIANTS = [
    {
        "variant": "V1",
        "depth": 4,
        "topology": "range",
        "parameter_count": 75,
        "source": (
            "reused 09.1/10 5-seed final metrics (D-13-01, no recompute); "
            "frozen samples re-scored with full_metric_suite UNCHANGED"
        ),
    },
    {
        "variant": "V2",
        "depth": 8,
        "topology": "range",
        "parameter_count": 135,
        "source": "Phase 13 new runs",
    },
    {
        "variant": "V3",
        "depth": 4,
        "topology": "linear",
        "parameter_count": 75,
        "source": "Phase 13 new runs",
    },
]
SEEDS = [42, 43, 44, 45, 46]

# By-construction data-equivalence note (09.1 quantum configs carry no
# data_hash field) — recorded verbatim-in-spirit per run_dualscale_fidelity.py.
QUANTUM_EQUIVALENCE_NOTE = (
    "V1 data equivalence established BY CONSTRUCTION: the 09.1/10 "
    "transform_ablation Pipeline-B runs used the identical "
    "load_and_preprocess(data.csv) OD tensor as the Phase-13 V2/V3 runs. The "
    "09.1 quantum configs wrote no data_hash field, so equivalence is "
    "by-construction (NO recompute of V1 training — D-13-01); the frozen V1 "
    "samples.npy bundles are only re-scored with the SAME full_metric_suite "
    "(D-10-20). No V1 training run-dir is created by this aggregator "
    "(RESEARCH Pitfall 4 / run_dualscale_fidelity.py:241-281 precedent)."
)


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _run_base(variant: str, seed: int, ansatz_root: Path) -> Path:
    """Resolve the frozen run dir for (variant, seed).

    V1 reuses the 09.1/10 ``transform_ablation`` Pipeline-B layout (D-13-01 —
    NO recompute, NO new V1 training dir). V2/V3 use the Phase-13 ``ansatz``
    layout written by ``run_ansatz.py``.
    """
    repo = _find_repo_root()
    if variant == "V1":
        return repo / "revision" / "results" / "transform_ablation" / "runs" / "B" / str(seed)
    return ansatz_root / "runs" / variant / str(seed)


def reconstruct_dualscale(variant: str, seed: int, ansatz_root: Path) -> dict:
    """Reconstruct OD-scale + log-return-scale windows from a frozen bundle.

    Pipeline-B reconstruction copied VERBATIM from
    ``run_dualscale_fidelity.reconstruct_od`` (the
    ``np.random.default_rng(seed*7919+1)`` od_start draw is load-bearing — do
    NOT change it). Returns ``{"od_samples", "transformed"}`` where
    ``transformed`` is the log-return array (Pipeline B always emits it).
    """
    base = _run_base(variant, seed, ansatz_root)
    spath = base / "samples.npy"
    ipath = base / "inverse_kwargs.npz"
    # CR-02: fail loudly + early if the frozen bundle is absent. The V1 branch
    # reads transform_ablation/runs/B/<seed> written by a *different* driver
    # (run_baselines._save_inverse_kwargs) in a prior phase, so its existence
    # is a cross-module assumption with no compile-time guarantee.
    if not spath.is_file() or not ipath.is_file():
        raise FileNotFoundError(
            f"{variant} seed={seed}: frozen bundle missing under {base} "
            f"(expected samples.npy + inverse_kwargs.npz) — V1 reuse "
            f"contract (D-13-01) broken; aborting before partial re-score"
        )
    samples_pm1 = np.load(spath).astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, variant)  # Plan 14-21 T01
    inv = np.load(ipath, allow_pickle=True)

    # CR-02: validate the on-disk schema BEFORE scoring so a key/name drift in
    # the 09.1/10 run_baselines layout raises an actionable message up front
    # instead of a bare KeyError deep in the aggregator, *after* V2/V3 have
    # already been re-scored (which would silently fail the ARCH-02 JSON).
    required = {"r_min", "r_max", "mu", "sigma", "od_starts"}
    missing = required - set(inv.files)
    if missing:
        raise KeyError(
            f"{variant} seed={seed}: inverse_kwargs.npz missing keys "
            f"{sorted(missing)} (found {sorted(inv.files)}) — V1 reuse "
            f"contract (D-13-01) broken; aborting before partial re-score"
        )

    r_min = float(inv["r_min"])
    r_max = float(inv["r_max"])
    mu = float(inv["mu"])
    sigma = float(inv["sigma"])
    od_starts_pool = np.asarray(inv["od_starts"])

    r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
    rng = np.random.default_rng(seed * 7919 + 1)
    od_start_per_window = rng.choice(
        od_starts_pool, size=r_norm.shape[0], replace=True
    )
    r_norm_t = torch.tensor(r_norm)
    od_start_t = torch.tensor(od_start_per_window)
    od_full = inverse_logreturns(
        r_norm_t, od_start_t, torch.tensor(mu), torch.tensor(sigma)
    )
    od = od_full.cpu().numpy()
    if od.shape[1] == 11:
        od = od[:, :10]
    return {"od_samples": od, "transformed": r_norm}


def build_real_references(csv_path: Path) -> dict:
    """Build the real OD-window + log-return reference arrays.

    Reuses the EXACT construction ``run_dualscale_fidelity.build_real_references``
    uses so the numbers reconcile with the Phase-11 dual-scale emission.
    """
    d_real = load_and_preprocess(str(csv_path))
    real_windowed_OD = (
        rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()
    )
    real_log_delta = d_real["log_delta"].cpu().numpy()
    return {
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": real_log_delta,
    }


def _rows_for_run(
    variant: str,
    depth: int,
    topology: str,
    seed: int,
    od: np.ndarray,
    trans: np.ndarray,
    real_OD_flat: np.ndarray,
    real_log_delta: np.ndarray,
) -> list:
    """Emit full_metric_suite rows for one run on BOTH scales.

    Each row carries the 9 required fields: model_kind, variant, depth,
    topology, pipeline, seed, metric_name, scale, value (D-10-16 extended).
    Metric math is ``revision.core.eval.full_metric_suite`` UNCHANGED
    (D-10-20).
    """
    out = []
    # ── scale = OD ───────────────────────────────────────────────────────────
    od_metrics = full_metric_suite(real_OD_flat, od.reshape(-1))
    for name, val in od_metrics.items():
        out.append(
            dict(
                model_kind="quantum",
                variant=variant,
                depth=depth,
                topology=topology,
                pipeline="B",
                seed=seed,
                metric_name=name,
                scale="OD",
                value=(float(val) if val is not None else None),
            )
        )
    # ── scale = log_return ───────────────────────────────────────────────────
    lr_metrics = full_metric_suite(real_log_delta, trans.reshape(-1))
    for name, val in lr_metrics.items():
        out.append(
            dict(
                model_kind="quantum",
                variant=variant,
                depth=depth,
                topology=topology,
                pipeline="B",
                seed=seed,
                metric_name=name,
                scale="log_return",
                value=(float(val) if val is not None else None),
            )
        )
    return out


def emit_rows(real_OD_flat, real_log_delta, ansatz_root: Path) -> list:
    """Build every (variant, seed) dual-scale row.

    V1 reads the frozen ``transform_ablation/runs/B/{seed}`` bundles (NO
    recompute of training, D-13-01); V2/V3 read the Phase-13 ``ansatz`` runs.
    """
    rows = []
    for spec in ANSATZ_VARIANTS:
        v = spec["variant"]
        d = int(spec["depth"])
        t = str(spec["topology"])
        for s in SEEDS:
            print(f"  metrics: variant={v} depth={d} topology={t} seed={s}")
            r = reconstruct_dualscale(v, s, ansatz_root)
            rows.extend(
                _rows_for_run(
                    v, d, t, s,
                    r["od_samples"], r["transformed"],
                    real_OD_flat, real_log_delta,
                )
            )
    return rows


def main() -> None:
    """CLI entry point — emit revision/results/ansatz_comparison.json."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 13 ARCH-02 ansatz comparison aggregator "
            "(reuses revision.core.eval.full_metric_suite UNCHANGED — "
            "D-10-20; V1 reused per D-13-01, no recompute)."
        )
    )
    ap.add_argument(
        "--out", type=Path, default=Path("revision/results"),
        help="Directory for ansatz_comparison.json.",
    )
    ap.add_argument(
        "--csv-path", type=Path, default=Path("./data.csv"),
        help="Path to the OD CSV (same data the frozen runs used).",
    )
    ap.add_argument(
        "--ansatz-root", type=Path,
        default=Path("revision/results/ansatz"),
        help="Root holding the Phase-13 V2/V3 runs/<variant>/<seed>/ bundles.",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = (repo / csv_path).resolve()
    ansatz_root = args.ansatz_root
    if not ansatz_root.is_absolute():
        ansatz_root = (repo / ansatz_root).resolve()

    refs = build_real_references(csv_path)
    print(
        f"[run_ansatz_comparison] real refs: OD_flat="
        f"{refs['real_OD_flat'].shape}, log_delta="
        f"{refs['real_log_delta'].shape}"
    )

    rows = emit_rows(refs["real_OD_flat"], refs["real_log_delta"], ansatz_root)

    out_dir = args.out if args.out.is_absolute() else (repo / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": (
            "long-form rows[] + models[] (D-10-16) + ansatz/depth/topology "
            "(Phase 13)"
        ),
        "model_kinds": ["quantum"],
        "ansatz_variants": ANSATZ_VARIANTS,
        "seeds": SEEDS,
        "scales": ["OD", "log_return"],
        "metric_helpers": {
            "functions": ["revision.core.eval.full_metric_suite"],
            "note": "reused unchanged, D-10-20",
        },
        "data_equivalence": QUANTUM_EQUIVALENCE_NOTE,
        "rows": rows,
    }
    out_path = out_dir / "ansatz_comparison.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(
        f"[run_ansatz_comparison] wrote {out_path} ({len(rows)} rows, "
        f"variants={sorted({r['variant'] for r in rows})}, "
        f"scales={sorted({r['scale'] for r in rows})})"
    )


if __name__ == "__main__":
    main()
