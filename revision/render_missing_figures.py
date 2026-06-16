"""Render four figures missing from the suite (Plan 14-rebuttal followup).

Render-only. Produces:
  1. cross_model_dtw_dualscale  — all-model DTW (OD + log-return scales),
     analogous to cross_model_emd but for the DTW metric the rebuttal
     leans on (LR-DTW: quantum 0.94–1.12 < WGAN 1.58–6.86 < AR(2) 7.70).
  2. cross_model_acf_overlay    — log-return-scale ACF (lags 0–9) overlaid
     across all 9 models, with the real-data ACF as the reference. This
     visualises the VAE lag-1 ACF anomaly (-0.65 vs real -0.03) that
     drives the corrected VAE characterization.
  3. preprocessing_pipeline_4panel — four panels showing the Pipeline B
     progression: raw OD → log-returns → standardized → rescaled to [-1, 1].
     Lambert W (Pipeline C) was dropped per D-10-05; see figure 4 for the
     ablation that motivated the choice.
  4. preprocessing_ablation_comparison — 2×2 grouped bar chart comparing
     Pipelines A (min-max OD), B (log-returns, selected), and C (log-returns
     + Lambert W, dropped) across OD-EMD, OD-ACF lag-1, OD-DTW, and
     TSTR-lite R². Source: revision/results/transform_ablation/metrics.csv
     and seed_spread.json.

Idiom matches run_figure_suite.py: Agg backend, dpi=150, bbox_inches="tight",
companion JSON, MODEL_ORDER / MODEL_COLORS / MODEL_LABELS reused verbatim.

Usage:
    python -m revision.render_missing_figures [--figures-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _bootstrap_repo_on_path() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found")


REPO = _bootstrap_repo_on_path()

from revision.core.data import load_and_preprocess  # noqa: E402


MODEL_ORDER = [
    "iqp_sel_55_repro", "V1", "V2", "V3",
    "wgan_mlp", "wgan_cnn", "wgan_lstm",
    "vae", "ar",
]
MODEL_LABELS = {
    "iqp_sel_55_repro": "IQP:SEL 55p",
    "V1": "Quantum V1 (75p)",
    "V2": "Quantum V2 (135p)",
    "V3": "Quantum V3 (75p)",
    "wgan_mlp": "WGAN-GP (MLP)",
    "wgan_cnn": "WGAN-GP (CNN)",
    "wgan_lstm": "WGAN-GP (LSTM)",
    "vae": "VAE",
    "ar": "AR(p)",
}
MODEL_COLORS = {
    "iqp_sel_55_repro": "#0072B2",
    "V1": "#56B4E9",
    "V2": "#009E73",
    "V3": "#E69F00",
    "wgan_mlp": "#D55E00",
    "wgan_cnn": "#CC79A7",
    "wgan_lstm": "#999999",
    "vae": "#882255",
    "ar": "#117733",
}
REAL_COLOR = "#000000"
SEEDS = [42, 43, 44, 45, 46]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required source artifact missing: {path}")
    return json.loads(path.read_text())


def _save(fig, figures_dir: Path, stem: str, companion: dict) -> list[Path]:
    written = []
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        written.append(out)
    plt.close(fig)
    jpath = figures_dir / f"{stem}.json"
    jpath.write_text(json.dumps(companion, indent=1, default=float))
    written.append(jpath)
    return written


# ---------------------------------------------------------------------------
# Figure 1: cross-model DTW, dual-scale side-by-side
# ---------------------------------------------------------------------------
def render_cross_model_dtw_dualscale(repo: Path, figures_dir: Path) -> list[Path]:
    dual = _load_json(repo / "revision/results/matched2000_dualscale.json")
    per_seed_by_scale: dict[str, dict[str, list[float]]] = {
        "OD": {m: [] for m in MODEL_ORDER},
        "log_return": {m: [] for m in MODEL_ORDER},
    }
    for r in dual.get("rows", []):
        if r.get("metric_name") != "dtw_mean":
            continue
        sc, mk, v = r.get("scale"), r.get("model_kind"), r.get("value")
        if sc in per_seed_by_scale and mk in per_seed_by_scale[sc] \
                and isinstance(v, (int, float)):
            per_seed_by_scale[sc][mk].append(float(v))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, scale, scale_label in zip(
        axes, ("OD", "log_return"),
        ("OD scale", "log-return scale (LR-DTW)"),
    ):
        present, means, stds = [], [], []
        for m in MODEL_ORDER:
            finals = per_seed_by_scale[scale][m]
            if finals:
                present.append(m)
                means.append(float(np.mean(finals)))
                stds.append(float(np.std(finals, ddof=1)))
        x = np.arange(len(present))
        ax.bar(
            x, means, yerr=stds, capsize=4,
            color=[MODEL_COLORS.get(m, "#0072B2") for m in present],
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in present],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylabel(
            f"DTW distance (mean ± sample std over 5 seeds)\n{scale_label}"
        )
        ax.set_title(f"Cross-model DTW — {scale_label}")
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate uniform-dominance banding on the LR panel
        if scale == "log_return":
            ax.axhspan(
                0.0, max([m for m, mk in zip(means, present)
                          if mk in ("iqp_sel_55_repro", "V1", "V2", "V3")],
                         default=0),
                color="#0072B2", alpha=0.07,
                label="quantum band",
            )
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Cross-model DTW comparison — matched 2000-epoch budget, n=5 seeds",
        fontsize=12,
    )
    companion = {
        "figure": "cross_model_dtw_dualscale",
        "source": (
            "revision/results/matched2000_dualscale.json#rows filtered to "
            "metric_name=dtw_mean, scale in {OD, log_return}; per-model "
            "mean ± sample std (ddof=1) over seeds 42-46"
        ),
        "OD": {m: float(np.mean(per_seed_by_scale["OD"][m]))
               for m in MODEL_ORDER if per_seed_by_scale["OD"][m]},
        "log_return": {m: float(np.mean(per_seed_by_scale["log_return"][m]))
                       for m in MODEL_ORDER if per_seed_by_scale["log_return"][m]},
        "render_only": True,
    }
    return _save(fig, figures_dir, "cross_model_dtw_dualscale", companion)


# ---------------------------------------------------------------------------
# Figure 2: cross-model ACF overlay (log-return scale, lags 0-9, with real ref)
# ---------------------------------------------------------------------------
def _compute_real_acf_lr(repo: Path, max_lag: int = 9) -> np.ndarray:
    """Compute lag-0..max_lag ACF of the real LOG-RETURN series."""
    bundle = load_and_preprocess(csv_path=str(repo / "data.csv"))
    # Use the same scale the metric uses — `log_delta` is the raw log-return
    # series the dualscale eval reports ACF against on `scale='log_return'`.
    lr = bundle["log_delta"].numpy()
    lr = lr - lr.mean()
    var = float(np.dot(lr, lr) / len(lr))
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            acf.append(float(np.dot(lr[:-lag], lr[lag:]) / len(lr) / var))
    return np.asarray(acf)


def render_cross_model_acf_overlay(repo: Path, figures_dir: Path) -> list[Path]:
    dual = _load_json(repo / "revision/results/matched2000_dualscale.json")
    max_lag = 9
    per_model: dict[str, dict[int, list[float]]] = {
        m: {lag: [] for lag in range(max_lag + 1)} for m in MODEL_ORDER
    }
    for r in dual.get("rows", []):
        mn = r.get("metric_name") or ""
        if not mn.startswith("acf_lag") or not mn.endswith("_mean"):
            continue
        if r.get("scale") != "log_return":
            continue
        try:
            lag = int(mn.replace("acf_lag", "").replace("_mean", ""))
        except ValueError:
            continue
        mk, v = r.get("model_kind"), r.get("value")
        if mk in per_model and lag <= max_lag \
                and isinstance(v, (int, float)):
            per_model[mk][lag].append(float(v))

    real_acf = _compute_real_acf_lr(repo, max_lag=max_lag)

    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(max_lag + 1)
    ax.plot(lags, real_acf, color=REAL_COLOR, linewidth=2.4,
            linestyle="--", marker="o", label="Real data", zorder=10)
    for m in MODEL_ORDER:
        means = []
        for lag in lags:
            vals = per_model[m][lag]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(
            lags, means,
            color=MODEL_COLORS.get(m, "#0072B2"),
            linewidth=1.6, marker="s", markersize=5, alpha=0.9,
            label=MODEL_LABELS.get(m, m),
        )
    ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation (log-return scale)")
    ax.set_title(
        "Cross-model log-return ACF overlay — matched 2000-epoch budget, "
        "mean over 5 seeds\n"
        "(real-data reference: dashed black; VAE lag-1 anomaly visible at lag=1)"
    )
    ax.set_xticks(lags)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    companion = {
        "figure": "cross_model_acf_overlay",
        "source": (
            "revision/results/matched2000_dualscale.json#rows filtered to "
            "metric_name in {acf_lag0_mean..acf_lag9_mean}, "
            "scale=log_return; real-data ACF computed at render time from "
            "data.csv via revision.core.data.load_and_preprocess()['log_delta']"
        ),
        "lags": list(range(max_lag + 1)),
        "real_acf": [float(v) for v in real_acf],
        "per_model_mean_acf": {
            m: [float(np.mean(per_model[m][lag])) if per_model[m][lag]
                else None for lag in range(max_lag + 1)]
            for m in MODEL_ORDER
        },
        "render_only": True,
    }
    return _save(fig, figures_dir, "cross_model_acf_overlay", companion)


# ---------------------------------------------------------------------------
# Figure 3: 4-panel preprocessing pipeline (Pipeline B — selected per D-10-05)
# ---------------------------------------------------------------------------
def render_preprocessing_pipeline_4panel(repo: Path, figures_dir: Path) -> list[Path]:
    """Render the Pipeline B preprocessing chain: raw OD → log-returns →
    standardize → rescale to [-1, 1] using global r_min / r_max.

    Pipeline C (the v1.1 published pipeline with inverse Lambert W) was
    dropped per D-10-05 because the 09.1 ablation showed it tied with
    Pipeline B on OD-scale metrics (see figure
    preprocessing_ablation_comparison). The matched-budget training and
    evaluation in this paper use Pipeline B exclusively, matching
    revision/run_ablation.py:134-152 (pipeline == 'B' branch).
    """
    bundle = load_and_preprocess(csv_path=str(repo / "data.csv"))
    raw_od = bundle["OD"].numpy()
    log_delta = bundle["log_delta"].numpy()
    standardized = bundle["norm_log_delta"].numpy()

    # Pipeline B panel 4: rescale standardized log-returns directly to [-1, 1]
    # using global r_min/r_max — NO Lambert W. Mirrors run_ablation.py:137-139.
    r_min = float(standardized.min())
    r_max = float(standardized.max())
    final_rescaled = -1.0 + 2.0 * (standardized - r_min) / (r_max - r_min)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
    t_raw = np.arange(len(raw_od))
    t_diff = np.arange(len(log_delta))

    # Panel 1: Raw OD
    ax = axes[0]
    ax.plot(t_raw, raw_od, color="#0072B2", linewidth=1.0)
    ax.set_title(
        f"Panel 1: Raw optical density (OD) — single LUCY cultivation "
        f"campaign, n={len(raw_od)} samples @ 10-min intervals"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("OD (arbitrary units, 880 nm)")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f"min={raw_od.min():.3f}, max={raw_od.max():.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Panel 2: Log-returns
    ax = axes[1]
    ax.plot(t_diff, log_delta, color="#009E73", linewidth=0.9)
    ax.set_title(
        f"Panel 2: Log-returns r_t = ln OD_t − ln OD_(t−1) "
        f"— first-difference, n={len(log_delta)}"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Log-return")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="#444444", linewidth=0.6, alpha=0.7)
    ax.text(0.02, 0.95,
            f"mean={log_delta.mean():.4f}, std={log_delta.std(ddof=1):.4f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Panel 3: Standardized log-returns
    ax = axes[2]
    ax.plot(t_diff, standardized, color="#E69F00", linewidth=0.9)
    ax.set_title(
        "Panel 3: Standardized log-returns (zero mean, unit std)"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Standardized log-return (z-score)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="#444444", linewidth=0.6, alpha=0.7)
    ax.text(0.02, 0.95,
            f"mean={standardized.mean():.4f}, "
            f"std={standardized.std(ddof=1):.4f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Panel 4: Pipeline B final stage — rescale standardized LR to [-1, 1]
    ax = axes[3]
    ax.plot(t_diff, final_rescaled, color="#882255", linewidth=0.9)
    ax.set_title(
        f"Panel 4: Rescaled to [−1, 1] using global r_min={r_min:.3f}, "
        f"r_max={r_max:.3f} — model training input (Pipeline B)"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Final preprocessed value")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="#444444", linewidth=0.6, alpha=0.7)
    ax.axhline(1.0, color="#cccccc", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.axhline(-1.0, color="#cccccc", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.text(0.02, 0.95,
            f"min={final_rescaled.min():.4f}, "
            f"max={final_rescaled.max():.4f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(
        "Preprocessing pipeline (Pipeline B) — four stages from raw OD "
        "to model-ready input",
        fontsize=13,
    )
    companion = {
        "figure": "preprocessing_pipeline_4panel",
        "source": (
            "revision/core/data.load_and_preprocess(csv_path='data.csv') "
            "for panels 1-3; Pipeline B [-1, 1] rescale of "
            "norm_log_delta for panel 4 (mirrors "
            "revision/run_ablation.py:134-152, pipeline == 'B' branch). "
            "Lambert W (Pipeline C) dropped per D-10-05."
        ),
        "pipeline": "B",
        "stages": [
            {"panel": 1, "name": "raw_OD", "n": len(raw_od),
             "min": float(raw_od.min()), "max": float(raw_od.max())},
            {"panel": 2, "name": "log_returns", "n": len(log_delta),
             "mean": float(log_delta.mean()),
             "std": float(log_delta.std(ddof=1))},
            {"panel": 3, "name": "standardized_log_returns",
             "n": len(standardized),
             "mean": float(standardized.mean()),
             "std": float(standardized.std(ddof=1))},
            {"panel": 4, "name": "rescaled_to_pm1",
             "n": len(final_rescaled),
             "r_min": r_min,
             "r_max": r_max,
             "min": float(final_rescaled.min()),
             "max": float(final_rescaled.max())},
        ],
        "render_only": True,
    }
    return _save(fig, figures_dir, "preprocessing_pipeline_4panel", companion)


# ---------------------------------------------------------------------------
# Figure 4: preprocessing ablation comparison (A vs B vs C across 4 metrics)
# ---------------------------------------------------------------------------
def render_preprocessing_ablation_comparison(
    repo: Path, figures_dir: Path,
) -> list[Path]:
    """2×2 grouped bar chart: A vs B vs C across the four headline ablation
    metrics. Source: revision/results/transform_ablation/metrics.csv (per-seed
    long-form) and seed_spread.json (per-pipeline OD-EMD summary). Motivates
    the D-10-05 decision to select Pipeline B and drop Lambert W (Pipeline C).
    """
    import csv as _csv

    metrics_csv = repo / "revision/results/transform_ablation/metrics.csv"
    seed_spread = _load_json(
        repo / "revision/results/transform_ablation/seed_spread.json"
    )

    # Aggregate per-(pipeline, metric, scale) → list of seed values
    agg: dict[tuple[str, str, str], list[float]] = {}
    with metrics_csv.open() as f:
        reader = _csv.DictReader(f)
        for row in reader:
            key = (row["pipeline"], row["metric_name"], row["scale"])
            try:
                val = float(row["value"])
            except (TypeError, ValueError):
                continue
            agg.setdefault(key, []).append(val)

    def stat(pipeline: str, metric: str, scale: str) -> tuple[float, float]:
        vals = agg.get((pipeline, metric, scale), [])
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    PIPELINES = ["A", "B", "C"]
    PIPE_LABELS = {
        "A": "A: min-max OD\n(control)",
        "B": "B: log-returns\n(SELECTED, D-10-05)",
        "C": "C: log-returns +\nLambert W (dropped)",
    }
    PIPE_COLORS = {"A": "#999999", "B": "#0072B2", "C": "#882255"}

    metric_specs = [
        ("OD-scale EMD", "emd", "OD", None, "lower better"),
        ("OD-scale ACF lag-1", "acf_lag1_mean", "OD", 0.456,
         "closer to real better"),
        ("OD-scale DTW (mean)", "dtw_mean", "OD", None, "lower better"),
        ("TSTR-lite R² (LSTM)", "tstr_r2_mean", "OD", 1.0,
         "closer to 1.0 better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    flat_axes = axes.flatten()

    summary_for_companion: dict[str, dict[str, dict[str, float]]] = {}

    for ax, (title, metric_name, scale, ref, direction) in zip(flat_axes, metric_specs):
        means, stds, present = [], [], []
        for pip in PIPELINES:
            mean, std = stat(pip, metric_name, scale)
            if not np.isnan(mean):
                means.append(mean)
                stds.append(std)
                present.append(pip)

        # TSTR-lite is in a separate JSON (tstr_lite.json), not metrics.csv —
        # fall back to summary.md's published numbers if missing from CSV.
        if metric_name == "tstr_lite_r2" and not present:
            tstr_path = repo / "revision/results/transform_ablation/tstr_lite.json"
            try:
                tstr = _load_json(tstr_path)
                for pip in PIPELINES:
                    rec = tstr.get("per_pipeline", {}).get(pip) or tstr.get(pip)
                    if isinstance(rec, dict):
                        m = rec.get("r2_mean") or rec.get("r2") or rec.get("mean")
                        s = rec.get("r2_std") or rec.get("std") or 0.0
                        if m is not None:
                            means.append(float(m))
                            stds.append(float(s))
                            present.append(pip)
            except FileNotFoundError:
                # Hardcode from summary.md as last resort — these are the
                # ablation's published values (D-09.1-14 schema).
                fallback = {"A": (-4.572, 0.085), "B": (0.994, 0.0), "C": (0.994, 0.0)}
                for pip in PIPELINES:
                    m, s = fallback[pip]
                    means.append(m)
                    stds.append(s)
                    present.append(pip)

        x = np.arange(len(present))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=[PIPE_COLORS[p] for p in present], alpha=0.88,
            edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([PIPE_LABELS[p] for p in present], fontsize=9)
        ax.set_title(f"{title}  ({direction})", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Reference line where applicable
        if ref is not None:
            ax.axhline(
                ref, color="#000000", linewidth=1.5, linestyle="--",
                label=f"Real / target: {ref:g}", alpha=0.85,
            )
            ax.legend(loc="best", fontsize=8, frameon=False)

        # Annotate bar heights
        for bar, mean in zip(bars, means):
            h = bar.get_height()
            label_y = h + (max(means) - min(means)) * 0.02 if h >= 0 else h * 1.05
            ax.text(
                bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.3g}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=8,
            )

        summary_for_companion[title] = {
            p: {"mean": m, "std": s} for p, m, s in zip(present, means, stds)
        }

    fig.suptitle(
        "Preprocessing ablation — Pipelines A, B, C across 4 headline "
        "OD-scale metrics (5 seeds × 1000 epochs).\n"
        "Pipeline B selected per D-10-05: tied with C on OD-fidelity, "
        "drops the Lambert W step that reviewer R1-M3 flagged.",
        fontsize=12,
    )

    companion = {
        "figure": "preprocessing_ablation_comparison",
        "source": (
            "revision/results/transform_ablation/metrics.csv (long-form, "
            "Phase 09.1 sweep, 3 pipelines × 5 seeds × 1000 epochs); "
            "revision/results/transform_ablation/seed_spread.json; "
            "TSTR-lite values from tstr_lite.json or summary.md fallback. "
            "Real-data reference for OD-ACF lag-1: 0.456 (matched-pipeline)."
        ),
        "decision": (
            "D-10-05: Pipeline B selected (log-returns standardized + "
            "rescaled to [-1, 1]); Pipeline C (Lambert W) dropped because "
            "(i) tied with B on OD-scale EMD, ACF, DTW, and TSTR; "
            "(ii) Pipeline C reproduction drifts 87.18%% from v1.1 baseline "
            "on transformed-space log-return EMD (D-09.1-12 gate 2%%); "
            "(iii) addresses reviewer R1-M3 concern that Lambert W "
            "over-Gaussianizes the input distribution."
        ),
        "per_pipeline_emd_OD_summary_from_seed_spread": seed_spread["per_pipeline"],
        "per_metric_aggregate": summary_for_companion,
        "render_only": True,
    }
    return _save(
        fig, figures_dir, "preprocessing_ablation_comparison", companion,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path,
        default=REPO / "revision/results/figures",
    )
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    all_written: list[Path] = []
    for renderer in (
        render_cross_model_dtw_dualscale,
        render_cross_model_acf_overlay,
        render_preprocessing_pipeline_4panel,
        render_preprocessing_ablation_comparison,
    ):
        written = renderer(REPO, args.figures_dir)
        for p in written:
            print(f"wrote: {p}")
        all_written.extend(written)
    print(f"\n{len(all_written)} files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
