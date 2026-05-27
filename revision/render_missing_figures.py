"""Render three figures missing from the suite (Plan 14-rebuttal followup).

Render-only. Produces:
  1. cross_model_dtw_dualscale  — all-model DTW (OD + log-return scales),
     analogous to cross_model_emd but for the DTW metric the rebuttal
     leans on (LR-DTW: quantum 0.94–1.12 < WGAN 1.58–6.86 < AR(2) 7.70).
  2. cross_model_acf_overlay    — log-return-scale ACF (lags 0–9) overlaid
     across all 9 models, with the real-data ACF as the reference. This
     visualises the VAE lag-1 ACF anomaly (-0.65 vs real -0.03) that
     drives the corrected VAE characterization.
  3. preprocessing_pipeline_4panel — four panels showing the raw OD →
     log-returns → standardized → Lambert-W + rescaled progression.

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
# Figure 3: 4-panel preprocessing pipeline
# ---------------------------------------------------------------------------
def render_preprocessing_pipeline_4panel(repo: Path, figures_dir: Path) -> list[Path]:
    bundle = load_and_preprocess(csv_path=str(repo / "data.csv"))
    raw_od = bundle["OD"].numpy()
    log_delta = bundle["log_delta"].numpy()
    standardized = bundle["norm_log_delta"].numpy()
    final_rescaled = bundle["scaled_data"].numpy()
    delta = float(bundle["delta"])

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

    # Panel 4: After inverse Lambert W heavy-tail correction + rescale to [-1, 1]
    ax = axes[3]
    ax.plot(t_diff, final_rescaled, color="#882255", linewidth=0.9)
    ax.set_title(
        f"Panel 4: Inverse Lambert W heavy-tail correction "
        f"(δ={delta:.4f}) + rescale to [−1, 1] — model training input"
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
        "Preprocessing pipeline — four stages from raw OD to model-ready input",
        fontsize=13, y=1.005,
    )
    companion = {
        "figure": "preprocessing_pipeline_4panel",
        "source": (
            "revision/core/data.load_and_preprocess(csv_path='data.csv') — "
            "Pipeline C (current paper): log-returns + standardize + "
            "inverse Lambert W + rescale to [-1, 1]"
        ),
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
            {"panel": 4, "name": "lambert_w_corrected_rescaled",
             "n": len(final_rescaled),
             "delta": delta,
             "min": float(final_rescaled.min()),
             "max": float(final_rescaled.max())},
        ],
        "render_only": True,
    }
    return _save(fig, figures_dir, "preprocessing_pipeline_4panel", companion)


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
    ):
        written = renderer(REPO, args.figures_dir)
        for p in written:
            print(f"wrote: {p}")
        all_written.extend(written)
    print(f"\n{len(all_written)} files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
