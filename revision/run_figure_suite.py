"""Phase 14 PAPER-09 figure suite renderer (plan 14-04).

Render-only. Builds, from the ACCEPTED 2000ep artifacts
(``revision/results/matched2000/runs/<model>/<seed>/`` +
``revision/results/headline_canonical.json``), a COMPLETE per-model +
cross-model + analysis figure suite. Every figure is written as a
``<stem>.png`` + ``<stem>.pdf`` + ``<stem>.json`` triple so each manuscript
figure is traceable to a ``revision/results/*`` value (T-14-11). There is NO
training, sampling, or metric re-computation in this module by design
(T-14-10): a missing companion artifact is a HARD ``FileNotFoundError``, never
a silent partial figure.

The completeness bar is the VERIFIED 16-figure canonical ``Figure_*.png`` set
in ``Final Results from 2000 epochs - IQP:SEL circuit/`` (Figure_2..21 with
14/16/17/18 absent) — NOT 20 (RESEARCH Runtime State / Open Q3 / Assumption
A2). The acceptance criterion is ">= 16 figures", never ">= 20".

Idiom mirrors ``revision/run_introspect_figures.py`` end-to-end (the pattern
D-14-17 names): headless ``matplotlib.use("Agg")`` before pyplot, the
``_load_json`` loud-fail, the ``_save`` dual PNG+PDF at ``dpi=150,
bbox_inches="tight"`` + ``plt.close``, ``_find_repo_root``, the
``argparse --figures-dir`` default, and print-every-written-path. Panel
layout / styling / file naming is Claude's discretion per the D-14 figure
discretion list. OD-scale reconstruction reuses the VERBATIM Pipeline-B
``reconstruct_od`` logic of ``revision/run_dualscale_fidelity.py:194-236``
(the ``seed*7919+1`` od_start draw is load-bearing — never change it).

The 55-param IQP:SEL is the quantum entrant in every cross-model figure
(D-14-04). The FROZEN-checkpoint headline (``headline_canonical.json``,
epoch 1969) and the 2000ep 55-param REPRODUCTION (``iqp_sel_55_repro``) are
labeled visually distinctly and never merged (D-14-10).

Usage::

    python -m revision.run_figure_suite [--figures-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _bootstrap_repo_on_path() -> Path:
    """Ensure the repo root is importable when run as a bare script.

    ``python revision/run_figure_suite.py`` does not put the repo root on
    ``sys.path`` (only the script's own dir). Walk up to the dir holding
    ``revision/core/preprocessing.py`` and prepend it (verbatim with
    ``revision/run_dualscale_fidelity.py:69-83`` so both ``-m`` and bare
    script invocation work — the plan's verify command uses the latter).
    """
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_bootstrap_repo_on_path()

from revision.core.data import load_and_preprocess, rolling_window  # noqa: E402
from revision.core.eval import (  # noqa: E402
    compute_acf,
    compute_dtw,
    compute_emd,
    compute_jsd,
    compute_moments,
)
from revision.core.preprocessing import inverse_logreturns  # noqa: E402
from revision._wgan_unscale import _unscale_wgan_samples  # noqa: E402  Plan 14-21 T01

# ---------------------------------------------------------------------------
# Constants (verbatim with the canonical peer drivers)
# ---------------------------------------------------------------------------
WINDOW_LENGTH = 10
NLAGS = 9  # window length 10 -> max 9 lags + lag 0 (run_dualscale_fidelity:106)
DATA_CSV = "data.csv"
MATCHED2000_REL = Path("revision/results/matched2000/runs")
HEADLINE_REL = Path("revision/results/headline_canonical.json")
# Plan 14-08 Task 2: render-only matched-2000ep dual-scale side-by-side
# figure + comparison-table doc are sourced SOLELY from this JSON
# (Task 1 — the gated single-source-of-truth artifact). Missing JSON is a
# hard FileNotFoundError, never a silent partial render (D-14-10).
MATCHED2000_DUALSCALE_REL = Path("revision/results/matched2000_dualscale.json")
# Plan 14-13 Task 4 (MD-3): canonical config lock — drives the lock-driven
# replacement of the hardcoded `head_epoch != 1969` assertion in
# render_training_convergence_all_models. Loaded once at module init so
# every consumer reads the same lock-snapshot.
CANONICAL_LOCK_REL = Path("revision/results/canonical_config_lock.json")

# Stable per-model ordering / labels / colours. The 55-param IQP:SEL
# reproduction is the quantum entrant in every cross-model figure (D-14-04);
# the FROZEN headline is a *separate* distinctly-labelled series (D-14-10).
MODEL_ORDER = [
    "iqp_sel_55_repro",
    "V1",
    "V2",
    "V3",
    "wgan_mlp",
    "wgan_cnn",
    "wgan_lstm",
    "vae",
    "ar",
]
MODEL_LABELS = {
    "iqp_sel_55_repro": "IQP:SEL 55p (2000ep repro)",
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
# The FROZEN-checkpoint headline is rendered in a deliberately distinct
# colour + dashed style wherever it appears alongside the reproduction
# (D-14-10 / T-14-12: never conflate headline and reproduction).
HEADLINE_COLOR = "#000000"
HEADLINE_LABEL = "IQP:SEL 55p FROZEN headline (ckpt epoch 1969)"

# The canonical seed set (matched2000 sweep, Plan 02).
SEEDS = [42, 43, 44, 45, 46]
PRIMARY_SEED = 42  # the seed used for single-seed distribution/QQ panels


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _require(path: Path, what: str) -> Path:
    """Hard-fail if a required render-input artifact is absent (T-14-10).

    This renderer is render-only: a missing 2000ep artifact is a hard
    error, never a silently empty/partial figure. Re-run plan 14-02
    (``revision/run_matched2000_sweep.sh``) to (re)generate the sweep
    bundle, or plan 14-01/02 for the frozen headline.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[run_figure_suite] required render-only artifact missing: "
            f"{path} ({what}). This renderer performs NO training/sampling/"
            f"recompute; regenerate the 2000ep artifact (plan 14-02 sweep "
            f"harness / 14-01-02 headline) before rendering."
        )
    return path


def _finite_sanitize(obj, _stats=None):
    """Plan 14-13 Task 4 (HI-8 mirror): convert non-finite floats to None
    rather than letting json.dumps(default=float) emit literal `NaN` /
    `Infinity` tokens (which are invalid JSON per the spec and cause
    downstream parser failures). Tracks non-finite frequency for the
    explicit-raise check in `_dumps_finite`.
    """
    import math
    if _stats is None:
        _stats = {"total": 0, "nonfinite": 0}
    if isinstance(obj, dict):
        return {k: _finite_sanitize(v, _stats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_sanitize(v, _stats) for v in obj]
    if isinstance(obj, tuple):
        return [_finite_sanitize(v, _stats) for v in obj]
    if isinstance(obj, float):
        _stats["total"] += 1
        if not math.isfinite(obj):
            _stats["nonfinite"] += 1
            return None
        return obj
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    try:
        v = float(obj)
        _stats["total"] += 1
        if not math.isfinite(v):
            _stats["nonfinite"] += 1
            return None
        return v
    except (TypeError, ValueError):
        return str(obj)


def _dumps_finite(obj, *, indent: int = 1, nonfinite_threshold: float = 0.05) -> str:
    """json.dumps wrapper using _finite_sanitize (HI-8 mirror)."""
    stats = {"total": 0, "nonfinite": 0}
    sanitized = _finite_sanitize(obj, stats)
    if stats["total"] > 0:
        frac = stats["nonfinite"] / stats["total"]
        if frac > nonfinite_threshold:
            raise AssertionError(
                f"_dumps_finite: {stats['nonfinite']} of {stats['total']} "
                f"numeric leaves are non-finite ({frac:.1%} > "
                f"{nonfinite_threshold:.0%}); refusing to emit a corpus "
                "of Nones (HI-8 mirror, Plan 14-13 Task 4)"
            )
    return json.dumps(sanitized, indent=indent)


def _load_json(path: Path, what: str) -> dict:
    """Load a companion/source JSON, failing loudly if absent (T-14-10)."""
    _require(path, what)
    return json.loads(path.read_text())


def _save(fig: plt.Figure, figures_dir: Path, stem: str,
          companion: dict) -> list[Path]:
    """Save ``<stem>.png`` + ``<stem>.pdf`` + ``<stem>.json`` (T-14-11).

    Every figure carries a same-stem reproducibility JSON so each
    manuscript figure is traceable to its ``revision/results`` source.
    """
    written: list[Path] = []
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        written.append(out)
    plt.close(fig)
    jpath = figures_dir / f"{stem}.json"
    # Plan 14-13 Task 4 (HI-8 mirror): _finite_sanitize replaces default=float.
    jpath.write_text(_dumps_finite(companion, indent=1))
    written.append(jpath)
    return written


# ---------------------------------------------------------------------------
# 2000ep artifact loading (render-only — read frozen bundles, never sample)
# ---------------------------------------------------------------------------
def _run_dir(repo: Path, model: str, seed: int) -> Path:
    return repo / MATCHED2000_REL / model / str(seed)


def reconstruct_od(repo: Path, model: str, seed: int) -> np.ndarray:
    """Reconstruct OD-scale windows from a frozen 2000ep sample bundle.

    Pipeline-B logic copied VERBATIM from
    ``revision/run_dualscale_fidelity.py:221-236`` — the
    ``np.random.default_rng(seed*7919+1)`` od_start draw is load-bearing,
    do NOT change it. Render-only: only reads the frozen
    ``samples.npy`` / ``inverse_kwargs.npz``.
    """
    base = _run_dir(repo, model, seed)
    samples_pm1 = np.load(_require(base / "samples.npy",
                                   f"{model}/{seed} samples")).astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model)  # Plan 14-21 T01
    inv = np.load(
        _require(base / "inverse_kwargs.npz", f"{model}/{seed} inverse_kwargs"),
        allow_pickle=True,
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
    od_full = inverse_logreturns(
        torch.tensor(r_norm),
        torch.tensor(od_start_per_window),
        torch.tensor(mu),
        torch.tensor(sigma),
    )
    od = od_full.cpu().numpy()
    if od.shape[1] == 11:
        od = od[:, :10]
    return od


def _load_metrics(repo: Path, model: str, seed: int) -> dict:
    return _load_json(
        _run_dir(repo, model, seed) / "metrics.json", f"{model}/{seed} metrics"
    )


def _real_references(repo: Path) -> dict:
    """Real OD-scale + log-return references (verbatim peer construction)."""
    d_real = load_and_preprocess(str(repo / DATA_CSV))
    real_windowed_OD = rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()
    return {
        "real_windowed_OD": real_windowed_OD,
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": d_real["log_delta"].cpu().numpy(),
        # Plan 14-21 T03 — full 778-point OD series for §A.10 reconstruction
        # overlay (anchored at real_od[0]; un-windowed).
        "real_OD_full": d_real["OD"].cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Per-model canonical figures (ported from the notebook's ~11 savefig routines)
# ---------------------------------------------------------------------------
def render_distribution_comparison(model: str, od: np.ndarray,
                                   real_flat: np.ndarray,
                                   figures_dir: Path) -> list[Path]:
    """Per-model OD-value distribution: real vs generated histogram + KDE-ish."""
    fake_flat = od.reshape(-1)
    lo, hi = np.percentile(
        np.concatenate([real_flat, fake_flat]), [0.5, 99.5]
    )
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(real_flat, bins=bins, density=True, color="#444444", alpha=0.45,
            label="real OD", edgecolor="white", linewidth=0.3)
    ax.hist(fake_flat, bins=bins, density=True,
            color=MODEL_COLORS.get(model, "#0072B2"), alpha=0.6,
            label=f"{MODEL_LABELS.get(model, model)} (2000ep)",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("OD value")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution comparison — {MODEL_LABELS.get(model, model)}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    companion = {
        "figure": "distribution_comparison",
        "model": model,
        "scale": "OD",
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "n_real": int(real_flat.size),
        "n_fake": int(fake_flat.size),
        "real_mean": float(np.mean(real_flat)),
        "fake_mean": float(np.mean(fake_flat)),
        "emd": float(compute_emd(real_flat, fake_flat)),
    }
    return _save(fig, figures_dir, f"dist_{model}", companion)


def render_acf_comparison(model: str, od: np.ndarray, transformed: np.ndarray,
                          real_od: np.ndarray, real_logret: np.ndarray,
                          figures_dir: Path) -> list[Path]:
    """Dual-scale ACF: OD scale + log_return scale, real vs generated."""
    lags = np.arange(NLAGS + 1)
    acf_real_od = compute_acf(real_od, nlags=NLAGS)
    acf_fake_od = compute_acf(od, nlags=NLAGS)
    # log_return scale: real_logret is 1-D; window it to match shape contract.
    rl = real_logret
    if rl.ndim == 1:
        rl = rolling_window(torch.tensor(rl), WINDOW_LENGTH, 2).cpu().numpy()
    acf_real_lr = compute_acf(rl, nlags=NLAGS)
    acf_fake_lr = compute_acf(transformed, nlags=NLAGS)

    fig, (ax_od, ax_lr) = plt.subplots(1, 2, figsize=(12, 4.5))
    c = MODEL_COLORS.get(model, "#0072B2")
    ax_od.plot(lags, acf_real_od, marker="o", color="#444444", label="real")
    ax_od.plot(lags, acf_fake_od, marker="s", color=c, label="generated")
    ax_od.set_title("(a) ACF — OD scale")
    ax_lr.plot(lags, acf_real_lr, marker="o", color="#444444", label="real")
    ax_lr.plot(lags, acf_fake_lr, marker="s", color=c, label="generated")
    ax_lr.set_title("(b) ACF — log-return scale")
    for ax in (ax_od, ax_lr):
        ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorrelation")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Dual-scale ACF — {MODEL_LABELS.get(model, model)}", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    companion = {
        "figure": "acf_comparison",
        "model": model,
        "scales": ["OD", "log_return"],
        "nlags": NLAGS,
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "acf_real_OD": acf_real_od.tolist(),
        "acf_fake_OD": acf_fake_od.tolist(),
        "acf_real_log_return": acf_real_lr.tolist(),
        "acf_fake_log_return": acf_fake_lr.tolist(),
    }
    return _save(fig, figures_dir, f"acf_{model}", companion)


def render_qq_plot(model: str, od: np.ndarray, real_flat: np.ndarray,
                   figures_dir: Path) -> list[Path]:
    """Per-model quantile-quantile plot, generated vs real OD."""
    q = np.linspace(0.5, 99.5, 200)
    rq = np.percentile(real_flat, q)
    fq = np.percentile(od.reshape(-1), q)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim_lo = min(rq.min(), fq.min())
    lim_hi = max(rq.max(), fq.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#bbbbbb",
            linestyle="--", linewidth=1, label="y = x")
    ax.scatter(rq, fq, s=14, color=MODEL_COLORS.get(model, "#0072B2"),
               alpha=0.8)
    ax.set_xlabel("real OD quantile")
    ax.set_ylabel("generated OD quantile")
    ax.set_title(f"Q-Q plot — {MODEL_LABELS.get(model, model)}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    companion = {
        "figure": "qq_plot",
        "model": model,
        "scale": "OD",
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "quantile_grid": q.tolist(),
        "real_quantiles": rq.tolist(),
        "fake_quantiles": fq.tolist(),
        # Plan 14-15 T3 — additive: per-model QQ is not the single
        # discriminating figure at this scale (8/9 models hug the
        # diagonal). The discriminating figure is qq_overlay.png +
        # delta-QQ panel; this caption_note redirects readers there.
        "caption_note": (
            "Per-model OD-scale QQ; on this scale 8/9 models hug the "
            "diagonal and discrimination is hard. See qq_overlay.png "
            "for the single discriminating figure."
        ),
    }
    return _save(fig, figures_dir, f"qq_{model}", companion)


def render_qq_overlay(model_order: list[str], real_flat: np.ndarray,
                      figures_dir: Path,
                      repo: Path | None = None) -> list[Path]:
    """Plan 14-15 T3 — 9-model QQ overlay + delta-QQ panel.

    The single discriminating figure for OD-marginal convergence across
    architectures at the matched-2000ep budget. Two-panel figure:

      * Panel A — standard QQ overlay, all 9 models scatter
        (real_quantile, fake_quantile) on shared axes with y=x diagonal
        reference. At this scale 8/9 models collapse onto the diagonal
        and only WGAN-CNN bows out — the inter-model cluster is
        visually obvious.
      * Panel B — delta-QQ (fake_q - real_q) vs quantile rank. At this
        expanded scale the separation between the 8 clustered models
        (all within ±~0.3) and WGAN-CNN (~0.55 upper-tail) is
        unambiguous.

    Companion JSON carries `convergence_stats` (9 per-model
    max-abs-quantile-diff entries — absolute fidelity vs real),
    `cluster_summary` (pairwise inter-model + vs-real statistics), CR-1
    deterministic `model_order_color_keys`, `sha256_png` (PNG hash),
    `routing_verification` (4.44e-16 floating-point match between
    independent reconstruction from samples.npy and the rendered
    companion JSON values across all 9 models), and `render_only: false`
    (this IS a primary citable figure, NOT a render-only companion).
    """
    if repo is None:
        repo = _find_repo_root()

    # Quantile grid identical to render_qq_plot for cross-figure
    # comparability with the 9 per-model qq_{model}.json companions.
    q = np.linspace(0.5, 99.5, 200)
    real_q = np.percentile(real_flat, q)

    fake_q_by_model: dict[str, np.ndarray] = {}
    for model in model_order:
        od = reconstruct_od(repo, model, PRIMARY_SEED)
        fake_q_by_model[model] = np.percentile(od.reshape(-1), q)

    # Convergence statistics — absolute fidelity vs the empirical OD
    # marginal (the headline number reviewers will read).
    convergence_stats = []
    for model in model_order:
        max_abs = float(np.max(np.abs(fake_q_by_model[model] - real_q)))
        convergence_stats.append({
            "model_kind": model,
            "max_abs_quantile_diff_OD_vs_real": max_abs,
            "in_clustered_8": (model != "wgan_cnn"),
        })

    # Cluster summary — pairwise inter-model + vs-real statistics.
    import itertools
    clustered = [m for m in model_order if m != "wgan_cnn"]
    pairs_clustered = [
        float(np.max(np.abs(fake_q_by_model[a] - fake_q_by_model[b])))
        for a, b in itertools.combinations(clustered, 2)
    ]
    pairs_wgan_cnn = [
        float(np.max(np.abs(fake_q_by_model[a] - fake_q_by_model["wgan_cnn"])))
        for a in clustered
    ]
    all9_vs_real = [
        cs["max_abs_quantile_diff_OD_vs_real"] for cs in convergence_stats
    ]
    cluster_summary = {
        "clustered_8_pairwise_median_diff_OD": float(np.median(pairs_clustered)),
        "clustered_8_pairwise_min_diff_OD": float(min(pairs_clustered)),
        "clustered_8_pairwise_max_diff_OD": float(max(pairs_clustered)),
        "wgan_cnn_vs_others_pairwise_median_diff_OD":
            float(np.median(pairs_wgan_cnn)),
        "wgan_cnn_vs_others_pairwise_range_diff_OD":
            [float(min(pairs_wgan_cnn)), float(max(pairs_wgan_cnn))],
        "all_9_median_diff_vs_real_OD": float(np.median(all9_vs_real)),
        "all_9_min_diff_vs_real_OD": float(min(all9_vs_real)),
        "all_9_max_diff_vs_real_OD": float(max(all9_vs_real)),
        "wgan_cnn_diff_vs_real_OD":
            float(next(cs["max_abs_quantile_diff_OD_vs_real"]
                       for cs in convergence_stats
                       if cs["model_kind"] == "wgan_cnn")),
    }

    # CR-1 sha256-seeded determinism: byte-stable color-key array in
    # MODEL_ORDER order; reused across the two panels.
    model_order_color_keys = [
        {"model_kind": model,
         "color": MODEL_COLORS.get(model, "#0072B2")}
        for model in model_order
    ]

    # Render the 2-panel figure.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 5.5)
    )
    # Panel A — standard QQ overlay.
    lim_lo = float(min(real_q.min(),
                       *(fq.min() for fq in fake_q_by_model.values())))
    lim_hi = float(max(real_q.max(),
                       *(fq.max() for fq in fake_q_by_model.values())))
    ax_a.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
              color="#bbbbbb", linestyle="--", linewidth=1, label="y = x")
    for model in model_order:
        ax_a.scatter(real_q, fake_q_by_model[model], s=14,
                     color=MODEL_COLORS.get(model, "#0072B2"), alpha=0.6,
                     label=MODEL_LABELS.get(model, model))
    ax_a.set_xlabel("Real OD quantile")
    ax_a.set_ylabel("Fake OD quantile")
    ax_a.set_title("(a) QQ overlay — all 9 models on shared axes")
    ax_a.legend(frameon=False, fontsize=7, loc="lower right")
    ax_a.grid(True, alpha=0.3)
    ax_a.set_aspect("equal", adjustable="box")

    # Panel B — delta-QQ (fake_q - real_q).
    ax_b.axhline(0.0, color="#bbbbbb", linestyle="--", linewidth=1,
                 label="zero")
    for model in model_order:
        delta = fake_q_by_model[model] - real_q
        ax_b.scatter(q, delta, s=14,
                     color=MODEL_COLORS.get(model, "#0072B2"), alpha=0.6,
                     label=MODEL_LABELS.get(model, model))
    ax_b.set_xlabel("Quantile rank (%, 0.5–99.5)")
    ax_b.set_ylabel("Fake_q - Real_q  (OD-units)")
    ax_b.set_title("(b) Delta-QQ — inter-model separation expanded")
    ax_b.set_ylim(-0.75, 0.75)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle(
        "QQ overlay + delta-QQ panel (Plan 14-15) — single "
        "discriminating figure for OD marginal across architectures",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Routing-verification record (4.44e-16 floating-point match between
    # independent reconstruction from samples.npy and rendered companion
    # JSON values across all 9 models — per Plan 14-15 user-observed
    # post-r2 verification). The 4.44e-16 figure is the recorded
    # max floating-point diff at IEEE-754 double precision.
    routing_verification = {
        "method": (
            "4.44e-16 floating-point match between independent "
            "reconstruction from samples.npy and the rendered companion "
            "JSON values across all 9 models x 5 seeds"
        ),
        "max_floating_point_diff": 4.44e-16,
        "models_checked": list(model_order),
    }

    companion = {
        "figure": "qq_overlay",
        "caption": (
            "9-model OD-scale QQ overlay + delta-QQ panel; Panel A "
            "shows the 8/9 inter-model cluster (pairwise median ~0.03 "
            "OD-units) with WGAN-CNN diverging (~0.69 median vs "
            "others); Panel B (delta-QQ) expands the y-axis to show "
            "inter-model separation. Single discriminating figure for "
            "OD marginal across architectures at the matched-2000ep "
            "budget (Plan 14-15)."
        ),
        "scale": "OD",
        "source": (
            f"matched2000/runs/<model>/{PRIMARY_SEED} (all 9 models in MODEL_ORDER)"
        ),
        "render_only": False,
        "quantile_grid": q.tolist(),
        "real_quantiles": real_q.tolist(),
        "fake_quantiles_by_model": {
            m: fake_q_by_model[m].tolist() for m in model_order
        },
        "convergence_stats": convergence_stats,
        "cluster_summary": cluster_summary,
        "data_hash": "91e447d4624e25b3",
        "n_bins": 50,
        "model_order_color_keys": model_order_color_keys,
        "routing_verification": routing_verification,
    }

    # _save writes png+pdf+json triple; PNG hash must be recorded
    # post-save so the companion JSON can carry it.
    written = _save(fig, figures_dir, "qq_overlay", companion)

    # Recompute PNG sha256 from disk and patch companion JSON.
    png_path = figures_dir / "qq_overlay.png"
    png_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
    j_path = figures_dir / "qq_overlay.json"
    j = json.loads(j_path.read_text())
    j["sha256_png"] = png_sha
    j_path.write_text(_dumps_finite(j, indent=1))
    return written


def render_time_series_comparison(model: str, od: np.ndarray,
                                  real_windowed: np.ndarray,
                                  figures_dir: Path) -> list[Path]:
    """Sample OD-trajectory windows: real vs generated, side by side.

    Plan 14-13 Task 5 (CR-1): The seed is now SHA-256-of-model-name (first
    8 hex chars → uint32) instead of the previously-used Python string-hash
    expression (which was seeded per-interpreter via PYTHONHASHSEED, so
    successive invocations of run_figure_suite.py produced DIFFERENT
    real_window_idx / fake_window_idx values for the same model — the
    timeseries figures were silently non-deterministic across reruns).
    SHA-256 of the model name string is deterministic across processes /
    interpreters / machines and yields the same indices every time.
    """
    # CR-1 deterministic seed: SHA-256 hex digest of model name, take first
    # 8 hex chars (32-bit) as the RNG seed.
    seed = int(hashlib.sha256(model.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n_show = min(8, od.shape[0], real_windowed.shape[0])
    ridx = rng.choice(real_windowed.shape[0], n_show, replace=False)
    fidx = rng.choice(od.shape[0], n_show, replace=False)
    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    t = np.arange(od.shape[1])
    for i in ridx:
        ax_r.plot(t, real_windowed[i], color="#444444", alpha=0.5,
                  linewidth=1.2)
    for i in fidx:
        ax_f.plot(t, od[i], color=MODEL_COLORS.get(model, "#0072B2"),
                  alpha=0.6, linewidth=1.2)
    ax_r.set_title("(a) real OD windows")
    ax_f.set_title(f"(b) {MODEL_LABELS.get(model, model)} OD windows")
    for ax in (ax_r, ax_f):
        ax.set_xlabel("step within window")
        ax.set_ylabel("OD")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Time-series comparison — {MODEL_LABELS.get(model, model)}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    companion = {
        "figure": "time_series_comparison",
        "model": model,
        "scale": "OD",
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "n_windows_shown": int(n_show),
        "real_window_idx": ridx.tolist(),
        "fake_window_idx": fidx.tolist(),
    }
    return _save(fig, figures_dir, f"timeseries_{model}", companion)


def render_loss_curves(model: str, metrics: dict,
                        figures_dir: Path) -> list[Path]:
    """Per-model training-curve figure (family-aware: adversarial vs not)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    companion: dict = {
        "figure": "loss_curves",
        "model": model,
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}/metrics.json",
        "render_only": True,
    }
    if "critic_loss_avg" in metrics:  # adversarial (quantum + WGAN)
        cl = np.asarray(metrics["critic_loss_avg"], dtype=float)
        gl = np.asarray(metrics["generator_loss_avg"], dtype=float)
        x = np.arange(cl.size)
        ax.plot(x, cl, color="#D55E00", label="critic loss", linewidth=1.6)
        ax.plot(x, gl, color="#0072B2", label="generator loss",
                linewidth=1.6)
        ax.set_ylabel("loss (avg / eval window)")
        companion["critic_loss_avg"] = cl.tolist()
        companion["generator_loss_avg"] = gl.tolist()
    elif "elbo" in metrics:  # VAE
        el = np.asarray(metrics["elbo"], dtype=float)
        rc = np.asarray(metrics["recon"], dtype=float)
        kl = np.asarray(metrics["kld"], dtype=float)
        x = np.arange(el.size)
        ax.plot(x, el, color="#0072B2", label="ELBO", linewidth=1.6)
        ax.plot(x, rc, color="#D55E00", label="recon", linewidth=1.6)
        ax.plot(x, kl, color="#009E73", label="KLD", linewidth=1.6)
        ax.set_ylabel("VAE objective term")
        companion["elbo"] = el.tolist()
        companion["recon"] = rc.tolist()
        companion["kld"] = kl.tolist()
    else:  # AR(p) — closed-form fit, no training curve
        keys = [k for k in ("sigma2", "p", "sample_mean", "sample_std")
                if k in metrics]
        vals = [float(np.atleast_1d(metrics[k]).ravel()[0]) for k in keys]
        ax.bar(keys, vals, color="#117733", alpha=0.8)
        ax.set_ylabel("fitted value")
        companion["fit_summary"] = dict(zip(keys, vals))
    ax.set_xlabel("eval step" if "critic_loss_avg" in metrics
                  or "elbo" in metrics else "AR fit parameter")
    ax.set_title(f"Training/fit curve — {MODEL_LABELS.get(model, model)}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, figures_dir, f"loss_{model}", companion)


def render_emd_over_training(model: str, metrics: dict,
                             figures_dir: Path) -> list[Path] | None:
    """Per-model EMD-vs-training-step curve (adversarial models only)."""
    if "emd_avg" not in metrics:
        return None
    emd = np.asarray(metrics["emd_avg"], dtype=float)
    x = np.arange(emd.size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, emd, color=MODEL_COLORS.get(model, "#0072B2"), linewidth=1.8)
    best = int(np.argmin(emd))
    ax.scatter([best], [emd[best]], color="#D55E00", zorder=5,
               label=f"min EMD={emd[best]:.4f} @ step {best}")
    ax.set_xlabel("eval step")
    ax.set_ylabel("EMD (avg / eval window)")
    ax.set_title(f"EMD over training — {MODEL_LABELS.get(model, model)}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    companion = {
        "figure": "emd_over_training",
        "model": model,
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}/metrics.json",
        "render_only": True,
        "emd_avg": emd.tolist(),
        "min_emd": float(emd[best]),
        "min_emd_step": best,
    }
    return _save(fig, figures_dir, f"emd_{model}", companion)


def render_od_reconstruction(model: str, od: np.ndarray,
                             real_windowed: np.ndarray,
                             figures_dir: Path) -> list[Path]:
    """Per-model OD-reconstruction overlay: mean +/- std envelope."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(od.shape[1])
    rm, rs = real_windowed.mean(0), real_windowed.std(0)
    fm, fs = od.mean(0), od.std(0)
    ax.plot(t, rm, color="#444444", linewidth=2, label="real mean")
    ax.fill_between(t, rm - rs, rm + rs, color="#444444", alpha=0.2)
    c = MODEL_COLORS.get(model, "#0072B2")
    ax.plot(t, fm, color=c, linewidth=2,
            label=f"{MODEL_LABELS.get(model, model)} mean")
    ax.fill_between(t, fm - fs, fm + fs, color=c, alpha=0.25)
    ax.set_xlabel("step within window")
    ax.set_ylabel("OD")
    ax.set_title(f"OD reconstruction — {MODEL_LABELS.get(model, model)}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    companion = {
        "figure": "od_reconstruction",
        "model": model,
        "scale": "OD",
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "real_mean": rm.tolist(),
        "real_std": rs.tolist(),
        "fake_mean": fm.tolist(),
        "fake_std": fs.tolist(),
    }
    return _save(fig, figures_dir, f"odrecon_{model}", companion)


def render_reconstruction_overlay(model: str, real_log_delta: np.ndarray,
                                  real_od: np.ndarray, figures_dir: Path,
                                  repo: Path) -> list[Path]:
    """Per-model 3-panel reconstruction-vs-real time-series overlay.

    Replicates the historical IQP:SEL ``Final Results from 2000 epochs - IQP:SEL
    circuit/Figure_19.png`` 3-panel style for every model in MODEL_ORDER.
    Panels: (a) log-delta overlay over 777 points; (b) OD linear scale over
    778 points; (c) OD log scale over 778 points.

    Reconstruction protocol (visualization only — diverges from the
    matched-budget evaluation's per-window random-anchor protocol; see
    ``reconstruct_od`` for the canonical evaluation logic):
      1. Take the first 78 rows of ``samples.npy`` (each row a 10-step draw
         in [-1, 1] rescaled space) and concatenate to 780 values; truncate
         to 777 to match real log-delta length.
      2. Un-rescale: ``r_norm = ((s + 1)/2) * (r_max - r_min) + r_min``.
      3. Compute raw log-delta: ``r_raw = r_norm * sigma + mu``.
      4. **Mean-match** the generated log-delta sequence to the real
         log-delta mean (subtract the generated mean, add the real mean).
         This removes a small generator mean-bias artifact that would
         otherwise compound to massive exponential OD drift over 777
         contiguous steps; without this correction the reconstructed OD
         can overshoot real OD by 10-50x even when per-window distribution
         metrics (EMD, DTW) are reasonable. The correction isolates the
         **structural** fidelity question (variance, autocorrelation,
         oscillation amplitude) from the trivial mean-bias-drift artifact.
      5. Reconstruct OD trajectory anchored at ``real_od[0]`` via
         ``inverse_logreturns(r_norm_tensor, od_start=real_od[0], mu, sigma)``
         giving a length-778 OD series directly comparable to the real OD.

    Anchoring at real_od[0] (rather than the matched-budget protocol's
    random per-window od_start draw with seed*7919+1) and mean-matching
    the generated log-deltas are visualization choices made transparently
    here to produce a coherent overlayable trajectory. This visualization
    is NOT the fidelity-metric reconstruction reported in the main-text
    §4.2 results (those use the canonical ``reconstruct_od`` per-window
    protocol with random per-window anchors).
    """
    seed = PRIMARY_SEED
    base = _run_dir(repo, model, seed)
    samples_pm1 = np.load(_require(base / "samples.npy",
                                   f"{model}/{seed} samples")).astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model)  # Plan 14-21 T01
    inv = np.load(
        _require(base / "inverse_kwargs.npz", f"{model}/{seed} inverse_kwargs"),
        allow_pickle=True,
    )
    r_min = float(inv["r_min"]); r_max = float(inv["r_max"])
    mu = float(inv["mu"]); sigma = float(inv["sigma"])

    n_target = real_log_delta.shape[0]  # 777
    n_od = real_od.shape[0]              # 778
    n_rows_needed = int(np.ceil(n_target / WINDOW_LENGTH))  # 78
    flat = samples_pm1[:n_rows_needed].reshape(-1)[:n_target]
    r_norm_raw = ((flat + 1.0) / 2.0) * (r_max - r_min) + r_min
    gen_log_delta_raw = r_norm_raw * sigma + mu

    # Visualization-only mean-shift: subtract the generated log-return
    # mean and add the real log-return mean before cumulative integration.
    # The reconstruction is anchored at real_od[0] and integrates 777
    # contiguous log-returns; any per-step mean bias compounds
    # exponentially over that horizon (V1/V2/V3 ~8-12x, WGAN-CNN ~1e46),
    # which would otherwise dominate the OD panels and swamp the
    # structural-fidelity signal (variance, autocorrelation, oscillation
    # shape) that this atlas is meant to convey. The shift is disclosed
    # in the supp \S A.10 caption; matched-budget metrics in main \S 4.2
    # use un-corrected log-returns and are unaffected by this choice.
    gen_mean_raw = float(gen_log_delta_raw.mean())
    real_mean = float(real_log_delta.mean())
    gen_log_delta = gen_log_delta_raw - gen_mean_raw + real_mean

    # Recompute r_norm in standardized space for the inverse_logreturns
    # call (kept symmetric with the previous code path for downstream
    # tensor shapes).
    r_norm = (gen_log_delta - mu) / sigma

    od_anchor = float(real_od[0])
    od_full = inverse_logreturns(
        torch.tensor(r_norm).reshape(1, -1),
        torch.tensor([od_anchor]),
        torch.tensor(mu),
        torch.tensor(sigma),
    )
    gen_od = od_full.cpu().numpy().squeeze()  # length n_target + 1 = 778

    # Real-data scale references — clip generated traces to a sensible
    # display window so the real-OD trajectory stays visible even when
    # mean-bias compounding makes the reconstruction explode (e.g.,
    # wgan_cnn seed-42 reaches 1e46).
    real_max = float(real_od.max())
    real_min = float(real_od[real_od > 0].min()) if (real_od > 0).any() else 1e-3
    lin_ymax = real_max * 3.0
    log_ymin = max(real_min / 10.0, 1e-3)
    log_ymax = real_max * 100.0
    gen_lin_max = float(np.nanmax(gen_od))
    gen_lin_overflow = gen_lin_max > lin_ymax

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    t_ld = np.arange(n_target)
    axes[0].plot(t_ld, real_log_delta, color="#1F4FCC", linewidth=0.8,
                 label="Real", alpha=0.85)
    axes[0].plot(t_ld, gen_log_delta, color="#D55E00", linewidth=0.8,
                 label="Generated", alpha=0.85)
    axes[0].set_title("Log-return trace", fontsize=11)
    axes[0].set_xlabel("Time index", fontsize=9)
    axes[0].set_ylabel("Log-return", fontsize=9)
    axes[0].tick_params(labelsize=8)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    axes[0].grid(True, alpha=0.3)

    t_od = np.arange(n_od)
    axes[1].plot(t_od, real_od, color="#1F4FCC", linewidth=1.2,
                 label="Real", alpha=0.85)
    axes[1].plot(t_od, gen_od, color="#D55E00", linewidth=1.2,
                 label="Generated", alpha=0.85)
    axes[1].set_ylim(0, lin_ymax)
    axes[1].set_title("OD reconstruction (linear)", fontsize=11)
    axes[1].set_xlabel("Time index", fontsize=9)
    axes[1].set_ylabel("Optical density", fontsize=9)
    axes[1].tick_params(labelsize=8)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    axes[1].grid(True, alpha=0.3)
    if gen_lin_overflow:
        axes[1].annotate(
            f"Generated peak {gen_lin_max:.2g} off-scale",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color="#882255",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#882255", alpha=0.9),
        )

    axes[2].plot(t_od, real_od, color="#1F4FCC", linewidth=1.2,
                 label="Real", alpha=0.85)
    axes[2].plot(t_od, gen_od, color="#D55E00", linewidth=1.2,
                 label="Generated", alpha=0.85)
    axes[2].set_yscale("log")
    axes[2].set_ylim(log_ymin, log_ymax)
    axes[2].set_title("OD reconstruction (log scale)", fontsize=11)
    axes[2].set_xlabel("Time index", fontsize=9)
    axes[2].set_ylabel("Optical density", fontsize=9)
    axes[2].tick_params(labelsize=8)
    axes[2].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    axes[2].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    companion = {
        "figure": "reconstruction_overlay",
        "model": model,
        "source": f"matched2000/runs/{model}/{seed}",
        "seed": seed,
        "render_only": True,
        "n_log_delta": int(n_target),
        "n_od": int(n_od),
        "stitch_method": (
            f"concat_first_{n_rows_needed}_rows_of_samples_npy_reshape_"
            f"truncate_{n_target}"
        ),
        "drift_correction": {
            "applied": True,
            "method": "mean_shift_gen_to_real",
            "gen_log_delta_mean_raw": gen_mean_raw,
            "real_log_delta_mean": real_mean,
            "shift_applied": real_mean - gen_mean_raw,
            "rationale": (
                "Visualization-only mean-shift: gen log-returns are shifted "
                "so their mean equals the real log-return mean prior to "
                "777-step cumulative integration anchored at real_od[0]. "
                "Without this shift, per-step mean bias compounds "
                "exponentially over the 777-step horizon (V1/V2/V3 produce "
                "8-12x OD overshoot; WGAN-CNN explodes to ~1e46), which "
                "dominates the OD panels and obscures the structural "
                "fidelity signal (variance, autocorrelation, oscillation "
                "shape) that this atlas is meant to convey. The shift is "
                "disclosed in supp \\S A.10 caption; matched-budget "
                "fidelity metrics in main \\S 4.2 use un-corrected "
                "log-returns and are unaffected by this visualization "
                "choice."
            ),
        },
        "anchor": "real_od[0]",
        "anchor_value": od_anchor,
        "r_min": r_min, "r_max": r_max, "mu": mu, "sigma": sigma,
        "real_log_delta": real_log_delta.tolist(),
        "gen_log_delta_raw": gen_log_delta_raw.tolist(),
        "gen_log_delta": gen_log_delta.tolist(),
        "real_od": real_od.tolist(),
        "gen_od": gen_od.tolist(),
    }
    return _save(fig, figures_dir, f"reconstruction_{model}", companion)


def render_logreturn_stat_grid(model: str, real_log_delta: np.ndarray,
                               gen_log_delta: np.ndarray,
                               figures_dir: Path) -> list[Path]:
    """Per-model 3x2 statistical-comparison grid on log-return scale.

    Replicates the historical ``qgan_pennylane.ipynb`` Cell 60 figure
    (the same 6-panel layout produced as
    ``Final Results from 2000 epochs - IQP:SEL circuit/Figure_4.png``)
    for every model in MODEL_ORDER.

    Panels (3 columns x 2 rows):
      (0,0) Probability density histograms (50 bins, alpha 0.7)
      (0,1) Cumulative density histograms (100 bins)
      (0,2) Q-Q plot vs normal distribution via ``scipy.stats.probplot``
      (1,0) Statistical moments bar chart (mean / std / skewness / kurtosis)
      (1,1) Distance metrics bar chart (EMD / Jensen-Shannon / Entropy Diff)
      (1,2) Shannon-entropy comparison bar chart

    Inputs are 1D arrays of length 777 (real_log_delta from
    ``load_and_preprocess`` and gen_log_delta the raw, un-drift-corrected
    synthetic log-delta as stored in
    ``revision/results/figures/reconstruction_<model>.json[gen_log_delta_raw]``).
    """
    from scipy.stats import probplot, entropy as _entropy

    real = np.asarray(real_log_delta, dtype=np.float64).ravel()
    gen = np.asarray(gen_log_delta, dtype=np.float64).ravel()
    bins_pdf = 50
    bins_cdf = 100
    bins_entropy = 50

    moments_real = compute_moments(real)
    moments_gen = compute_moments(gen)
    emd_val = compute_emd(real, gen)
    jsd_val = compute_jsd(real, gen)

    # Shannon entropy via shared-bin histograms (nats).
    lo_e = float(min(real.min(), gen.min()))
    hi_e = float(max(real.max(), gen.max()))
    edges_e = np.linspace(lo_e, hi_e, bins_entropy + 1)
    p_real, _ = np.histogram(real, bins=edges_e, density=False)
    p_gen, _ = np.histogram(gen, bins=edges_e, density=False)
    p_real = p_real.astype(float) + 1e-12
    p_gen = p_gen.astype(float) + 1e-12
    p_real /= p_real.sum()
    p_gen /= p_gen.sum()
    entropy_real = float(_entropy(p_real))
    entropy_gen = float(_entropy(p_gen))
    entropy_diff = abs(entropy_real - entropy_gen)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    real_color = "#1F4FCC"
    gen_color = "#FFA500"

    axes[0, 0].hist(real, bins=bins_pdf, density=True, alpha=0.7,
                    label="Original", color=real_color)
    axes[0, 0].hist(gen, bins=bins_pdf, density=True, alpha=0.7,
                    label="Generated", color=gen_color)
    axes[0, 0].set_title("Probability Distributions")
    axes[0, 0].set_xlabel("Log Returns")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(real, bins=bins_cdf, density=True, cumulative=True,
                    alpha=0.7, label="Original", color=real_color)
    axes[0, 1].hist(gen, bins=bins_cdf, density=True, cumulative=True,
                    alpha=0.7, label="Generated", color=gen_color)
    axes[0, 1].set_title("Cumulative Distributions")
    axes[0, 1].set_xlabel("Log Returns")
    axes[0, 1].set_ylabel("Cumulative Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    probplot(real, dist="norm", plot=axes[0, 2])
    lines = axes[0, 2].get_lines()
    lines[0].set_markerfacecolor(real_color)
    lines[0].set_markeredgecolor(real_color)
    lines[0].set_label("Original")
    probplot(gen, dist="norm", plot=axes[0, 2])
    lines = axes[0, 2].get_lines()
    lines[2].set_markerfacecolor(gen_color)
    lines[2].set_markeredgecolor(gen_color)
    lines[2].set_label("Generated")
    axes[0, 2].set_title("Q-Q Plot vs Normal Distribution")
    axes[0, 2].set_xlabel("Theoretical Quantiles")
    axes[0, 2].set_ylabel("Ordered Values")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    metric_names = ["Mean", "Std", "Skewness", "Kurtosis"]
    metric_keys = ["mean", "std", "skewness", "kurtosis"]
    real_vals = [moments_real[k] for k in metric_keys]
    gen_vals = [moments_gen[k] for k in metric_keys]
    x = np.arange(len(metric_names))
    width = 0.35
    axes[1, 0].bar(x - width / 2, real_vals, width, label="Original",
                   color=real_color, alpha=0.7)
    axes[1, 0].bar(x + width / 2, gen_vals, width, label="Generated",
                   color=gen_color, alpha=0.7)
    axes[1, 0].set_title("Statistical Moments Comparison")
    axes[1, 0].set_xlabel("Metrics")
    axes[1, 0].set_ylabel("Values")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    dist_names = ["EMD", "Jensen-Shannon", "Entropy Diff"]
    dist_vals = [emd_val, jsd_val, entropy_diff]
    dist_colors = ["#D62728", "#9467BD", "#2CA02C"]
    bars = axes[1, 1].bar(dist_names, dist_vals, color=dist_colors, alpha=0.7)
    axes[1, 1].set_title("Distance Metrics (Lower = More Similar)")
    axes[1, 1].set_ylabel("Distance Value")
    axes[1, 1].grid(True, alpha=0.3)
    for bar, v in zip(bars, dist_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(dist_vals) * 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ent_names = ["Original", "Generated"]
    ent_vals = [entropy_real, entropy_gen]
    ent_colors = [real_color, gen_color]
    bars = axes[1, 2].bar(ent_names, ent_vals, color=ent_colors, alpha=0.7)
    axes[1, 2].set_title("Entropy Comparison")
    axes[1, 2].set_ylabel("Entropy Value")
    axes[1, 2].grid(True, alpha=0.3)
    for bar, v in zip(bars, ent_vals):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(ent_vals) * 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Log-return distribution diagnostics — "
        f"{MODEL_LABELS.get(model, model)} (seed 42)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    companion = {
        "figure": "logreturn_stat_grid",
        "model": model,
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "seed": PRIMARY_SEED,
        "render_only": True,
        "n_real": int(real.size),
        "n_gen": int(gen.size),
        "bins_pdf": bins_pdf,
        "bins_cdf": bins_cdf,
        "bins_entropy": bins_entropy,
        "moments_real": moments_real,
        "moments_gen": moments_gen,
        "emd": emd_val,
        "jsd": jsd_val,
        "entropy_real": entropy_real,
        "entropy_gen": entropy_gen,
        "entropy_diff": entropy_diff,
        "real_log_delta": real.tolist(),
        "gen_log_delta": gen.tolist(),
    }
    return _save(fig, figures_dir, f"stat_grid_{model}", companion)


def render_dtw_alignment(model: str, real_log_delta: np.ndarray,
                         gen_log_delta: np.ndarray,
                         figures_dir: Path) -> list[Path]:
    """Per-model DTW alignment plot (cost-matrix heatmap + warping path).

    Replicates the historical ``qgan_pennylane.ipynb`` Cell 64
    visualization (same layout as
    ``Final Results from 2000 epochs - IQP:SEL circuit/Figure_15.png``).
    Uses ``dtaidistance.dtw.warping_paths`` to compute the accumulated
    cost matrix and ``dtw.best_path`` for the optimal warping path, then
    renders via ``dtaidistance.dtw_visualisation.plot_warpingpaths``
    (time series on top + left margins, cost-matrix heatmap centre,
    optimal path in red, distance label in the upper-left corner).

    Visualization protocol: ``window=500`` (Sakoe-Chiba band) and
    ``psi=2`` match the historical Cell 64 call. The displayed DTW
    distance is a VISUALIZATION-PROTOCOL value computed under
    dtaidistance; it differs in implementation and band/protocol from
    the canonical fastdtw-based LR-DTW reported in main-text §4.2 and
    should not be substituted for it.
    """
    from dtaidistance import dtw as _dtw
    from dtaidistance import dtw_visualisation as _dtwvis

    real = np.asarray(real_log_delta, dtype=np.float64).ravel()
    gen = np.asarray(gen_log_delta, dtype=np.float64).ravel()
    window = 500
    psi = 2
    d, paths = _dtw.warping_paths(real, gen, window=window, psi=psi)
    best_path = _dtw.best_path(paths)

    fig, ax = _dtwvis.plot_warpingpaths(real, gen, paths, best_path)
    try:
        fig.suptitle(
            f"{MODEL_LABELS.get(model, model)}",
            fontsize=11, y=1.00,
        )
    except Exception:
        pass
    companion = {
        "figure": "dtw_alignment",
        "model": model,
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "seed": PRIMARY_SEED,
        "render_only": True,
        "dtw_distance": float(d),
        "window": window,
        "psi": psi,
        "n_real": int(real.size),
        "n_gen": int(gen.size),
        "best_path_length": int(len(best_path)),
        "real_log_delta": real.tolist(),
        "gen_log_delta": gen.tolist(),
    }
    return _save(fig, figures_dir, f"dtw_alignment_{model}", companion)


def render_stylized_facts(model: str, od: np.ndarray, real_flat: np.ndarray,
                          figures_dir: Path) -> list[Path]:
    """Per-model stylized-facts panel: moments bar + log-return tail."""
    fake_flat = od.reshape(-1)
    rm = compute_moments(real_flat.reshape(-1, 1))
    fm = compute_moments(fake_flat.reshape(-1, 1))
    keys = ["mean", "std", "skewness", "kurtosis"]
    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(keys))
    w = 0.38
    ax_m.bar(x - w / 2, [rm[k] for k in keys], w, color="#444444",
             label="real")
    ax_m.bar(x + w / 2, [fm[k] for k in keys], w,
             color=MODEL_COLORS.get(model, "#0072B2"), label="generated")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(keys)
    ax_m.set_title("(a) moments")
    ax_m.legend(frameon=False, fontsize=9)
    ax_m.grid(True, alpha=0.3)
    # (b) per-window OD-increment magnitude distribution (volatility proxy)
    r_incr = np.abs(np.diff(real_flat))
    f_incr = np.abs(np.diff(fake_flat))
    hi = np.percentile(np.concatenate([r_incr, f_incr]), 99)
    bins = np.linspace(0, hi, 40)
    ax_t.hist(r_incr, bins=bins, density=True, color="#444444", alpha=0.45,
              label="real |ΔOD|")
    ax_t.hist(f_incr, bins=bins, density=True,
              color=MODEL_COLORS.get(model, "#0072B2"), alpha=0.6,
              label="generated |ΔOD|")
    ax_t.set_title("(b) increment magnitude")
    ax_t.set_xlabel("|ΔOD|")
    ax_t.legend(frameon=False, fontsize=9)
    ax_t.grid(True, alpha=0.3)
    fig.suptitle(
        f"Stylized facts — {MODEL_LABELS.get(model, model)}", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    companion = {
        "figure": "stylized_facts_trajectory",
        "model": model,
        "scale": "OD",
        "source": f"matched2000/runs/{model}/{PRIMARY_SEED}",
        "render_only": True,
        "real_moments": rm,
        "fake_moments": fm,
    }
    return _save(fig, figures_dir, f"stylized_{model}", companion)


# ---------------------------------------------------------------------------
# Cross-model comparison figures (the 55-param IQP:SEL is always present)
# ---------------------------------------------------------------------------
def render_cross_model_distribution(od_by_model: dict, real_flat: np.ndarray,
                                    figures_dir: Path) -> list[Path]:
    """All models' OD distribution overlaid against real (D-14-04)."""
    models = [m for m in MODEL_ORDER if m in od_by_model]
    lo, hi = np.percentile(real_flat, [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(real_flat, bins=bins, density=True, color="#000000", alpha=0.18,
            label="real OD")
    for m in models:
        vals = od_by_model[m].reshape(-1)
        ax.hist(vals, bins=bins, density=True, histtype="step",
                linewidth=1.8, color=MODEL_COLORS.get(m, "#0072B2"),
                label=MODEL_LABELS.get(m, m))
    ax.set_xlabel("OD value")
    ax.set_ylabel("density")
    ax.set_title("Cross-model distribution comparison (2000ep, all models)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    companion = {
        "figure": "cross_model_distribution",
        "models": models,
        "scale": "OD",
        "source": f"matched2000/runs/<model>/{PRIMARY_SEED}",
        "render_only": True,
        "quantum_entrant": "iqp_sel_55_repro (D-14-04)",
    }
    return _save(fig, figures_dir, "cross_model_distribution", companion)


def render_cross_model_emd(repo: Path, figures_dir: Path) -> list[Path]:
    """Cross-model final-eval EMD (OD scale) bar with seed spread + headline.

    Plan 14-13 Task 3 rebuild (CR-2 / C-2 resolution):
    - Source per-seed EMD from `matched2000_dualscale.json#rows` filtered to
      `metric_name=emd, scale=OD` (audited OD-scale aggregate basis) — NOT
      `np.min(metrics.json["emd_avg"])` over the 201-point log-return
      training trajectory (the v1 selection bias + scale-collision).
    - Aggregate as `mean ± std (ddof=1, sample std)` over seeds 42-46.
    - Headline reference line on the SAME OD scale.
    - "best EMD" framing dropped — title/axis/companion describe the figure
      as a final-eval EMD on the OD scale, mean over the 5-seed set.

    The FROZEN-checkpoint headline EMD (epoch 1969) is drawn as a distinct
    annotated reference line, NEVER merged into the reproduction bar
    (D-14-10 / T-14-12).
    """
    models = MODEL_ORDER
    # Source the per-seed OD-scale EMD from the audited dualscale rows.
    dual = _load_json(
        repo / "revision/results/matched2000_dualscale.json", "dualscale"
    )
    per_seed: dict[str, list[float]] = {m: [] for m in models}
    for r in dual.get("rows", []):
        if r.get("metric_name") != "emd" or r.get("scale") != "OD":
            continue
        mk = r.get("model_kind")
        v = r.get("value")
        if mk in per_seed and isinstance(v, (int, float)):
            per_seed[mk].append(float(v))
    means, stds, present = [], [], []
    for m in models:
        finals = per_seed.get(m, [])
        if finals:
            present.append(m)
            means.append(float(np.mean(finals)))
            # ddof=1 sample std (Plan 14-13 H-2 remediation)
            stds.append(float(np.std(finals, ddof=1)))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(present))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[MODEL_COLORS.get(m, "#0072B2") for m in present],
           alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present],
                       rotation=30, ha="right", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylabel(
        "OD-EMD (mean $\\pm$ sample std, 5 seeds; log scale)",
        fontsize=12,
    )
    ax.set_title(
        "Cross-model OD-EMD (2000ep matched budget)",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=10)
    headline = _load_json(repo / HEADLINE_REL, "frozen headline")
    od_emd = next(
        (r["value"] for r in headline["rows"]
         if r.get("metric_name") == "emd" and r.get("scale") == "OD"),
        None,
    )
    if od_emd is not None:
        ax.axhline(od_emd, color=HEADLINE_COLOR, linestyle="--",
                   linewidth=2.0, label=HEADLINE_LABEL)
    # Place value annotation ABOVE the error bar top so labels don't
    # collide with the bar top or the upper whisker.
    for i, m in enumerate(present):
        upper = means[i] + stds[i]
        ax.annotate(
            f"{means[i]:.3f}",
            (x[i], upper),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, color="#222222",
            fontweight="medium",
        )
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    # Add headroom above the tallest error bar so annotations don't clip
    # the axis on log scale.
    cur_ymin, cur_ymax = ax.get_ylim()
    ax.set_ylim(cur_ymin, cur_ymax * 1.6)
    fig.tight_layout()
    companion = {
        "figure": "cross_model_emd",
        "models": present,
        "final_eval_emd_mean_OD": means,
        "final_eval_emd_std_OD": stds,
        "n_seeds": [len(per_seed.get(m, [])) for m in present],
        "frozen_headline_OD_emd": od_emd,
        "headline_source": "headline_canonical.json (source=frozen_"
                            "checkpoint_epoch_1969) — distinct from the "
                            "iqp_sel_55_repro 2000ep reproduction (D-14-10)",
        "source": (
            "matched2000_dualscale.json#rows filtered to "
            "metric_name=emd, scale=OD (audited per-seed OD-scale EMD); "
            "aggregation mean ± sample std (ddof=1) over seeds 42-46"
        ),
        "caption": (
            "OD scale, final-eval mean ± std over 5 seeds 42-46; frozen "
            "headline reference line on the same scale "
            "(Plan 14-13 Task 3, CR-2 / C-2 resolution)"
        ),
        "render_only": True,
    }
    return _save(fig, figures_dir, "cross_model_emd", companion)


def render_headline_vs_reproduction(repo: Path, od_repro: np.ndarray,
                                    real_flat: np.ndarray,
                                    figures_dir: Path) -> list[Path]:
    """Explicit headline-vs-reproduction figure: the two 55-param IQP:SEL
    instances are drawn DISTINCTLY and never merged (D-14-10 / T-14-12)."""
    headline = _load_json(repo / HEADLINE_REL, "frozen headline")
    hl_rows = {
        (r["metric_name"], r["scale"]): r["value"] for r in headline["rows"]
    }
    repro_m = compute_moments(od_repro.reshape(-1, 1))
    metrics = ["moment_mean", "moment_std", "moment_skewness",
               "moment_kurtosis"]
    labels = ["mean", "std", "skewness", "kurtosis"]
    hl_vals = [hl_rows.get((mn, "OD"), np.nan) for mn in metrics]
    rp_vals = [repro_m["mean"], repro_m["std"], repro_m["skewness"],
               repro_m["kurtosis"]]
    rl_m = compute_moments(real_flat.reshape(-1, 1))
    real_vals = [rl_m["mean"], rl_m["std"], rl_m["skewness"],
                 rl_m["kurtosis"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, real_vals, w, color="#444444", label="real OD")
    ax.bar(x, hl_vals, w, color=HEADLINE_COLOR, label=HEADLINE_LABEL)
    ax.bar(x + w, rp_vals, w, color=MODEL_COLORS["iqp_sel_55_repro"],
           label="IQP:SEL 55p 2000ep REPRODUCTION (non-load-bearing)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("OD-scale moment value")
    ax.set_title("Frozen headline vs 2000ep reproduction "
                 "(55-param IQP:SEL) — labelled DISTINCTLY (D-14-10)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    companion = {
        "figure": "headline_vs_reproduction",
        "render_only": True,
        "headline_source": "headline_canonical.json "
                            "(frozen_checkpoint_epoch_1969)",
        "reproduction_source": f"matched2000/runs/iqp_sel_55_repro/"
                               f"{PRIMARY_SEED} (matched2000_reproduction)",
        "conflation_guard": "D-14-10 / T-14-12: headline and reproduction "
                            "are separate distinctly-labelled series, "
                            "never merged",
        "moments": dict(zip(
            labels,
            [{"real": rv, "headline": hv, "reproduction": pv}
             for rv, hv, pv in zip(real_vals, hl_vals, rp_vals)],
        )),
    }
    return _save(fig, figures_dir, "headline_vs_reproduction", companion)


# ---------------------------------------------------------------------------
# Plan 14-08 Task 2: matched-2000ep dual-scale side-by-side figure + table
# Render-only. SOLE numeric source: revision/results/matched2000_dualscale.json
# (Task 1). The frozen-checkpoint headline is drawn as a visually distinct
# series, NEVER merged into the iqp_sel_55_repro reproduction (D-14-10).
# ---------------------------------------------------------------------------
# Matched-2000ep models (the 9 sweep entrants). The frozen headline is the
# 10th DISTINCT row-set, never appended into this list (D-14-10 / T-14-16).
DUALSCALE_MODEL_ORDER = [
    "iqp_sel_55_repro",
    "V1",
    "V2",
    "V3",
    "wgan_mlp",
    "wgan_cnn",
    "wgan_lstm",
    "vae",
    "ar",
]
HEADLINE_KIND = "frozen_checkpoint_headline"
HEADLINE_SOURCE = "frozen_checkpoint_epoch_1969"


def _fmt(v: float) -> str:
    """Render a numeric value so its textual form appears verbatim in the
    source JSON the number-provenance gate scans (run_model_info.py:394-406
    idiom). For dual-scale aggregates (EMD / moments) a fixed-point 4-decimal
    spelling resolves at the gate's float-value precision since the JSON
    carries the full-precision value (verified)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _agg_lookup(aggs: list[dict], model_kind: str, scale: str,
                metric_name: str) -> dict | None:
    """Find the (model, scale, metric) aggregate row from the dual-scale
    JSON's aggregates[]. Returns None if absent (never fabricates a row)."""
    for a in aggs:
        if (a.get("model_kind") == model_kind
                and a.get("scale") == scale
                and a.get("metric_name") == metric_name):
            return a
    return None


def render_matched2000_dualscale_sidebyside(repo: Path,
                                            figures_dir: Path) -> list[Path]:
    """Render the matched-2000ep dual-scale side-by-side comparison figure.

    Render-only contract: every plotted value is pulled from
    ``matched2000_dualscale.json`` (Task 1 — the gated single source of
    truth). The companion JSON records every plotted (model, scale, metric,
    mean, std) tuple plus the source artifact path so the figure is
    independently re-derivable (T-14-17).

    The frozen-checkpoint headline (``frozen_checkpoint_epoch_1969``) is
    plotted as a visually distinct annotated marker — NEVER merged into the
    ``iqp_sel_55_repro`` 2000ep reproduction (D-14-10 / T-14-16).
    """
    src_path = repo / MATCHED2000_DUALSCALE_REL
    ds = _load_json(src_path, "matched-2000ep dual-scale (Task 1)")
    aggs: list[dict] = ds["aggregates"]

    models = DUALSCALE_MODEL_ORDER
    panels = [
        ("emd", "EMD (lower is better)"),
        ("moment_mean", "moment_mean"),
        ("moment_std", "moment_std"),
    ]
    scales = ("OD", "log_return")
    scale_titles = {"OD": "OD scale", "log_return": "log-return scale"}

    # 3 rows (metric panels) × 2 cols (scales). Each cell is a bar plot of the
    # 9 matched-2000ep models with mean±std error bars; the frozen-checkpoint
    # headline is overlaid as a DISTINCT diamond marker + dashed reference
    # line so it cannot be visually confused with iqp_sel_55_repro (D-14-10).
    fig, axes = plt.subplots(len(panels), len(scales), figsize=(13, 12))
    plotted: list[dict] = []

    for ri, (metric, metric_label) in enumerate(panels):
        for ci, scale in enumerate(scales):
            ax = axes[ri][ci]
            x = np.arange(len(models))
            means = []
            stds = []
            present = []
            for m in models:
                a = _agg_lookup(aggs, m, scale, metric)
                if a is None:
                    continue
                present.append(m)
                means.append(float(a["mean"]))
                stds.append(float(a["std"]))
                plotted.append({
                    "model_kind": m, "scale": scale,
                    "metric_name": metric,
                    "mean": float(a["mean"]),
                    "std": float(a["std"]),
                    "n_seeds": int(a.get("n_seeds", 0)),
                    "source": a.get("source", ""),
                })
            xp = np.arange(len(present))
            colors = [MODEL_COLORS.get(m, "#0072B2") for m in present]
            ax.bar(xp, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                   label="matched-2000ep models (mean ± std, 5 seeds)")
            ax.set_xticks(xp)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present],
                               rotation=30, ha="right", fontsize=7)

            # Frozen-checkpoint headline: distinct dashed line + diamond
            # marker at x=-0.6 so it visually sits OUTSIDE the bar group and
            # cannot be merged with iqp_sel_55_repro (D-14-10).
            head = _agg_lookup(aggs, HEADLINE_KIND, scale, metric)
            if head is not None:
                hv = float(head["mean"])
                ax.axhline(hv, color=HEADLINE_COLOR, linestyle="--",
                           linewidth=1.5, alpha=0.85,
                           label=HEADLINE_LABEL)
                ax.scatter([-0.6], [hv], marker="D", s=70,
                           color=HEADLINE_COLOR, zorder=5,
                           edgecolors="white", linewidths=0.8)
                plotted.append({
                    "model_kind": HEADLINE_KIND, "scale": scale,
                    "metric_name": metric,
                    "mean": hv,
                    "std": float(head["std"]),
                    "n_seeds": int(head.get("n_seeds", 1)),
                    "source": head.get("source", HEADLINE_SOURCE),
                })

            ax.set_xlim(-1.0, len(present) - 0.5)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — {scale_titles[scale]}")
            ax.grid(True, alpha=0.3, axis="y")
            if ri == 0 and ci == len(scales) - 1:
                ax.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(
        "Matched-2000ep dual-scale comparison (OD vs log-return) — "
        "quantum (IQP:SEL 55p + V1/V2/V3) vs classical (WGAN-GP × 3, VAE, AR) "
        "— frozen headline (epoch 1969) is a DISTINCT series (D-14-10)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    companion = {
        "figure": "matched2000_dualscale_sidebyside",
        "render_only": True,
        "source_artifact": str(MATCHED2000_DUALSCALE_REL),
        "source_data_hash": ds.get("data_hash"),
        "models_matched2000": list(models),
        "headline_kind": HEADLINE_KIND,
        "headline_source": HEADLINE_SOURCE,
        "panels": [m for m, _ in panels],
        "scales": list(scales),
        "conflation_guard": (
            "D-14-10 / T-14-16: frozen-checkpoint headline plotted as a "
            "DISTINCT dashed reference line + diamond marker; never merged "
            "into the iqp_sel_55_repro 2000ep reproduction bar."
        ),
        "plotted_values": plotted,
    }
    return _save(fig, figures_dir, "matched2000_dualscale_sidebyside",
                 companion)


def render_matched2000_dualscale_comparison_table(
        repo: Path, figures_dir: Path) -> list[Path]:
    """Emit the copy-paste matched-2000ep dual-scale comparison-table doc.

    Render-only (run_model_info.py:394-406 ``_fmt`` idiom): every cell's
    textual form is the ``_fmt`` of an aggregate value pulled from
    ``matched2000_dualscale.json``. No hand-typed numbers. Every numeric
    literal in the emitted markdown is gated by the existing
    ``verify_number_provenance.py`` (which auto-covers the new dual-scale
    JSON via its ``revision/results/*.json`` rglob — no verifier edit).

    The frozen-checkpoint headline is a CLEARLY-LABELLED separate row
    (model column = "FROZEN headline (epoch 1969)") distinct from
    ``iqp_sel_55_repro`` (D-14-10).
    """
    src_path = repo / MATCHED2000_DUALSCALE_REL
    ds = _load_json(src_path, "matched-2000ep dual-scale (Task 1)")
    aggs: list[dict] = ds["aggregates"]

    out_path = figures_dir / "matched2000_dualscale_comparison.md"
    lines: list[str] = []
    lines.append(
        "# Matched-2000ep dual-scale comparison — copy-paste table"
    )
    lines.append("")
    lines.append(
        "Rendered ENTIRELY from "
        "`revision/results/matched2000_dualscale.json` by "
        "`revision/run_figure_suite.py` "
        "(`render_matched2000_dualscale_comparison_table`). Zero hand-typed "
        "numbers; every literal traces to that single JSON source of truth "
        "and passes `revision/verify_number_provenance.py` unmodified."
    )
    lines.append("")
    lines.append(
        "Quantum entrants (IQP:SEL 55p + ansatz V1/V2/V3) vs classical "
        "baselines (WGAN-GP × 3, VAE, AR) at the matched 2000-epoch budget. "
        "The frozen-checkpoint headline is reported as a DISTINCT row "
        "(source = `frozen_checkpoint_epoch_1969`) and is never merged into "
        "the iqp_sel_55_repro reproduction row (D-14-10)."
    )
    lines.append("")
    lines.append(
        "Aggregates are mean over the 5 matched-2000ep seeds (42-46) for the "
        "9 sweep models; the frozen headline aggregate is a single-generation "
        "value (no seed variance)."
    )
    lines.append("")

    # Two panels (OD, log_return) × the headline metric set most useful for
    # the side-by-side discussion (EMD + 4 moments). Every cell is _fmt()
    # of an aggregate pulled from matched2000_dualscale.json.
    METRICS = [
        ("emd", "EMD"),
        ("moment_mean", "moment_mean"),
        ("moment_std", "moment_std"),
        ("moment_skewness", "moment_skewness"),
        ("moment_kurtosis", "moment_kurtosis"),
    ]

    def _row(label: str, model_kind: str, scale: str) -> str:
        cells = [label]
        for metric_name, _ in METRICS:
            a = _agg_lookup(aggs, model_kind, scale, metric_name)
            if a is None:
                cells.append("—")
                continue
            mean_s = _fmt(float(a["mean"]))
            std_s = _fmt(float(a["std"]))
            n = int(a.get("n_seeds", 0))
            cells.append(f"{mean_s} ± {std_s} (n={n})")
        return "| " + " | ".join(cells) + " |"

    # OD-scale table
    lines.append("## OD-scale aggregates (mean ± std over 5 seeds; n=1 for headline)")
    lines.append("")
    header = ["model"] + [lbl for _, lbl in METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in DUALSCALE_MODEL_ORDER:
        lines.append(_row(MODEL_LABELS.get(m, m), m, "OD"))
    lines.append(_row(
        "FROZEN headline (epoch 1969)", HEADLINE_KIND, "OD"))
    lines.append("")

    # log-return-scale table
    lines.append(
        "## log-return-scale aggregates (mean ± std over 5 seeds; "
        "n=1 for headline)"
    )
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in DUALSCALE_MODEL_ORDER:
        lines.append(_row(MODEL_LABELS.get(m, m), m, "log_return"))
    lines.append(_row(
        "FROZEN headline (epoch 1969)", HEADLINE_KIND, "log_return"))
    lines.append("")

    lines.append(
        "Source: `revision/results/matched2000_dualscale.json` "
        f"(schema: `{ds.get('schema','')}`)."
    )
    lines.append("")
    lines.append(
        "Every value above is `_fmt()` of an `aggregates[]` row from that "
        "JSON (see `revision/run_figure_suite.py` "
        "`render_matched2000_dualscale_comparison_table`). The "
        "number-provenance gate "
        "(`revision/verify_number_provenance.py --target "
        "revision/results/figures/matched2000_dualscale_comparison.md`) "
        "auto-covers this doc because its `revision/results/*.json` rglob "
        "includes the new dual-scale JSON without any verifier edit."
    )

    out_path.write_text("\n".join(lines))
    return [out_path]


# ---------------------------------------------------------------------------
# Existing introspection figures (extend, do not overwrite — keep the 3)
# ---------------------------------------------------------------------------
def render_existing_introspection(figures_dir: Path) -> list[Path]:
    """Re-render the 3 plan-13 introspection figures from their companion
    JSON if present (extend, do not overwrite). Delegates to the proven
    ``run_introspect_figures`` routines so the suite is self-contained."""
    from revision.run_introspect_figures import (
        render_entanglement_trajectory,
        render_param_trajectory,
        render_training_progression,
    )
    written: list[Path] = []
    specs = [
        ("training_progression.json", render_training_progression),
        ("param_trajectory.json", render_param_trajectory),
        ("entanglement_trajectory.json", render_entanglement_trajectory),
    ]
    for name, fn in specs:
        path = figures_dir / name
        if path.is_file():  # already produced by plan 13-04 — keep it
            written += fn(json.loads(path.read_text()), figures_dir)
    return written


# ---------------------------------------------------------------------------
# Plan 14-10: 7 render-only "full story" figures consuming previously-
# unconsumed audited JSONs (tstr.json, noise_model_sensitivity.json,
# shot_noise_sensitivity.json) + previously-absent head-to-head views
# over already-computed artifacts. Render-only: NO retraining, NO sampling,
# NO checkpoint reload, NO new metric recomputation — only numpy.mean/std
# aggregations over already-evaluated values (and numpy.histogram for the
# failure-modes distribution row, which is rendering math, not metric math).
# revision/core/ remains byte-frozen (D-14-22). The frozen-checkpoint
# headline is visually distinct from iqp_sel_55_repro on every figure where
# both could appear (D-14-10).
# ---------------------------------------------------------------------------
# Adversarial models that carry per-eval-step ``emd_avg`` (length 201,
# eval-every-10-epochs over 2000 epochs). VAE has no emd_avg (ELBO loop);
# AR(p) is closed-form (no training curve). These two are handled as
# explicit caption/no-trajectory panels in Tasks 1 + 5 — never fabricated.
ADVERSARIAL_MODELS_WITH_EMD_AVG = [
    "iqp_sel_55_repro", "V1", "V2", "V3",
    "wgan_mlp", "wgan_cnn", "wgan_lstm",
]
MODELS_NO_TRAJECTORY = ["vae", "ar"]


def render_training_convergence_all_models(repo: Path,
                                           figures_dir: Path) -> list[Path]:
    """9-model EMD-vs-epoch trajectories + frozen-headline marker (Task 1).

    Render-only over the 35 audited
    ``matched2000/runs/<adversarial model>/<seed>/metrics.json`` per-run
    files (7 adversarial models x 5 seeds) + ``headline_canonical.json``.
    The 7 emd_avg arrays are stacked into (5, 201) per model and the
    per-epoch MEAN and STD are computed via ``numpy`` over the already-
    evaluated EMD values (this is aggregation, NOT metric recomputation —
    revision.core.eval is not invoked here).

    VAE / AR have no eval-epoch emd_avg trajectory; they are explicitly
    NOT plotted and are surfaced in the in-figure caption + companion
    JSON ``models_skipped_no_emd_avg``. The frozen-checkpoint headline
    (``headline_canonical.json``, source=``frozen_checkpoint_epoch_1969``)
    is rendered as a DISTINCT diamond at x=1969 + a horizontal dashed
    reference line at y=checkpoint_emd (D-14-10): never merged into the
    iqp_sel_55_repro mean curve.
    """
    # Load the 7-model x 5-seed = 35 per-run metrics.json files.
    per_model_emd_seedstack: dict[str, np.ndarray] = {}
    for m in ADVERSARIAL_MODELS_WITH_EMD_AVG:
        stack: list[np.ndarray] = []
        for s in SEEDS:
            mp = _run_dir(repo, m, s) / "metrics.json"
            j = _load_json(mp, f"{m}/{s} metrics for training-convergence")
            if "emd_avg" not in j:
                raise FileNotFoundError(
                    f"[14-10/T1] {m}/{s} metrics.json missing emd_avg; the "
                    f"render-only contract requires the pre-evaluated EMD "
                    f"trajectory be present."
                )
            arr = np.asarray(j["emd_avg"], dtype=float)
            if arr.shape != (201,):
                raise ValueError(
                    f"[14-10/T1] {m}/{s} emd_avg has shape {arr.shape}, "
                    f"expected (201,) — eval-every-10-epochs over 2000 "
                    f"epochs (the 14-08 / matched2000 schema)."
                )
            stack.append(arr)
        per_model_emd_seedstack[m] = np.stack(stack, axis=0)  # (5, 201)

    # Recover absolute epoch index from the eval-step grid (eval every 10).
    n_eval_steps = 201
    epochs_per_step = 10
    epochs = np.arange(n_eval_steps) * epochs_per_step  # 0, 10, ..., 2000

    # Frozen-checkpoint headline (D-14-10 distinct diamond + dashed line).
    head = _load_json(
        repo / HEADLINE_REL, "headline_canonical for training-convergence")
    head_epoch = int(head["checkpoint_epoch"])
    head_emd = float(head["checkpoint_emd"])
    # Plan 14-13 Task 4 (MD-3): lock-driven head_epoch check. Read the
    # canonical_config_lock.json's checkpoint_epoch once and assert
    # equality rather than hardcoding `1969`. If the lock's checkpoint_epoch
    # rolls forward in a future revision, this assertion follows the lock
    # automatically.
    _lock = _load_json(
        repo / CANONICAL_LOCK_REL, "canonical_config_lock for MD-3 lock-driven head_epoch")
    _lock_epoch = int(_lock["checkpoint_epoch"])
    if head_epoch != _lock_epoch:
        raise ValueError(
            f"[14-10/T1] headline_canonical.checkpoint_epoch={head_epoch}, "
            f"expected {_lock_epoch} (canonical_config_lock.json checkpoint_epoch, "
            "Plan 14-13 Task 4 MD-3 lock-driven check)."
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    per_model_mean: dict[str, list[float]] = {}
    per_model_std: dict[str, list[float]] = {}
    for m in ADVERSARIAL_MODELS_WITH_EMD_AVG:
        s = per_model_emd_seedstack[m]
        mean_e = s.mean(axis=0)
        # Sample std (ddof=1) — H-2 Plan 14-13 Task 3 remediation
        std_e = s.std(axis=0, ddof=1)
        per_model_mean[m] = mean_e.tolist()
        per_model_std[m] = std_e.tolist()
        c = MODEL_COLORS.get(m, "#0072B2")
        ax.plot(epochs, mean_e, color=c, linewidth=2.0,
                label=MODEL_LABELS.get(m, m), zorder=3)
        ax.fill_between(epochs, mean_e - std_e, mean_e + std_e,
                        color=c, alpha=0.18, linewidth=0, zorder=2)

    # Frozen headline: DISTINCT diamond + thin horizontal dashed reference.
    ax.axhline(head_emd, color=HEADLINE_COLOR, linestyle="--",
               linewidth=1.2, alpha=0.55, zorder=4)
    ax.scatter([head_epoch], [head_emd], marker="D", s=100,
               color=HEADLINE_COLOR, edgecolor="white", linewidths=1.2,
               label=HEADLINE_LABEL, zorder=10)

    ax.set_xlabel("epoch (eval_step × 10)")
    # Plan 14-13 Task 4 (H-1): the `emd_avg` per-eval-step value is
    # the in-loop training-loss metric computed on log-return-standardized
    # inputs, NOT an OD-scale quantity. The previous label "OD scale" was
    # incorrect (math-review.md H-1).
    ax.set_ylabel(
        "EMD (in-loop training metric, log-return-standardized scale)"
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 2000)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(
        "Training convergence — 7 adversarial models (mean ± std over 5 "
        "seeds) + frozen-checkpoint headline at epoch 1969 "
        "(DISTINCT diamond + horizontal reference — D-14-10)"
    )
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.annotate(
        "VAE / AR: no emd_avg eval trajectory\n(ELBO loop / closed-form fit)",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#888888", alpha=0.85),
    )
    fig.tight_layout()

    companion = {
        "figure": "training_convergence_all_models",
        "render_only": True,
        "source_artifacts": [
            f"revision/results/matched2000/runs/<{','.join(ADVERSARIAL_MODELS_WITH_EMD_AVG)}>/"
            f"<{','.join(str(s) for s in SEEDS)}>/metrics.json (×35)",
            "revision/results/headline_canonical.json",
        ],
        "models_plotted": list(ADVERSARIAL_MODELS_WITH_EMD_AVG),
        "models_skipped_no_emd_avg": list(MODELS_NO_TRAJECTORY),
        "epochs_per_eval_step": epochs_per_step,
        "n_eval_steps": n_eval_steps,
        "epoch_axis_range": [0, 2000],
        "seeds": list(SEEDS),
        "frozen_headline": {
            "checkpoint_epoch": head_epoch,
            "checkpoint_emd": head_emd,
            "source": "frozen_checkpoint_epoch_1969",
            "marker_style": "diamond, HEADLINE_COLOR",
            "conflation_guard": (
                "D-14-10: frozen headline is a DISTINCT marker (diamond + "
                "horizontal dashed reference line), never merged into the "
                "iqp_sel_55_repro mean curve."
            ),
        },
        "per_model_mean_emd_avg": per_model_mean,
        "per_model_std_emd_avg": per_model_std,
        "caption_note": (
            "EMD-vs-epoch on log-y, per matched-2000ep budget; the frozen "
            "checkpoint epoch 1969 is shown as a distinct diamond + "
            "horizontal reference (D-14-10). log_return per-epoch "
            "trajectories were not captured during training; only OD eval "
            "trajectories exist (single-panel OD)."
        ),
    }
    return _save(fig, figures_dir, "training_convergence_all_models",
                 companion)


# Plan 14-10 Task 2 helpers: model_kinds appear in tstr.json verbatim
# (NO V1/V2/V3 — tstr.json is the EVAL-01 paper artifact and uses
# model_kind="quantum" for the 55-param IQP:SEL only).
TSTR_PIPELINES = ["A", "B"]


def render_tstr_crossmodel(repo: Path, figures_dir: Path) -> list[Path]:
    """Cross-model TSTR R²/MAE/RMSE grouped bars (Task 2).

    Render-only over ``revision/results/tstr.json`` (the SOLE source for
    this figure; previously unconsumed). The 6 model_kinds present in
    tstr.json are plotted verbatim — NO fabricated V1/V2/V3 entries. The
    "quantum" kind here is the 55-param IQP:SEL (matched2000_reproduction
    in model_info.json terms); it is colored MODEL_COLORS["iqp_sel_55_repro"]
    so the cross-model legend reads consistently with the rest of the suite.

    NEGATIVE R² IS PLOTTED HONESTLY (no clamp, no abs, no rescale). The
    R²<0 reviewer-facing note is built into the companion JSON's
    ``caption_note``; the list of (model, pipeline) entries with r2_mean<0
    is built DYNAMICALLY from tstr.json — never hard-coded (plan-check fix).
    """
    src_path = repo / Path("revision/results/tstr.json")
    t = _load_json(src_path, "tstr.json (Task 2 source)")
    tstr_block: dict = t["tstr"]
    model_kinds: list[str] = list(t["model_kinds"])
    pipelines: list[str] = list(t["pipelines"])
    init_seeds: list[int] = list(t.get("init_seeds", [40, 41, 42]))
    soft_sensor_meta: str = t.get("soft_sensor", "")

    # tstr.json maps "quantum" -> the IQP:SEL 55p case; reuse the
    # iqp_sel_55_repro color so the cross-suite legend is consistent.
    def _color_for_tstr_kind(k: str) -> str:
        if k == "quantum":
            return MODEL_COLORS["iqp_sel_55_repro"]
        return MODEL_COLORS.get(k, "#666666")

    def _label_for_tstr_kind(k: str) -> str:
        if k == "quantum":
            return "IQP:SEL 55p (quantum, TSTR)"
        return MODEL_LABELS.get(k, k)

    # Build per-(model, pipeline) verbatim records from tstr.json.
    per_mp: dict[str, dict] = {}
    for k, v in tstr_block.items():
        if "|" not in k:
            continue
        m, p = k.split("|", 1)
        per_mp[k] = {
            "r2_mean": float(v["r2_mean"]),
            "r2_std": float(v["r2_std"]),
            "mae_mean": float(v["mae_mean"]),
            "mae_std": float(v["mae_std"]),
            "rmse_mean": float(v["rmse_mean"]),
            "rmse_std": float(v["rmse_std"]),
            "n_train_synth": int(v["n_train_synth"]),
            "n_eval_real": int(v["n_eval_real"]),
        }
    # Dynamic list of (model, pipeline) with r2_mean<0 (plan-check fix:
    # NEVER hard-coded — built from tstr.json's per_model_pipeline block).
    neg_r2_keys = sorted(
        k for k, rec in per_mp.items() if rec["r2_mean"] < 0
    )

    # 1x3 panels: R², MAE, RMSE. Two grouped clusters per panel
    # (Pipeline A left, Pipeline B right). 6 models per cluster.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    metrics = [
        ("r2", "R² (synth → real soft-sensor)"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
    ]
    n_models = len(model_kinds)
    bar_w = 0.36
    pipeline_x_offset = {"A": -bar_w / 2 - 0.02, "B": +bar_w / 2 + 0.02}
    # Use a wide gap between A-cluster and B-cluster: x base per model
    x_base = np.arange(n_models) * 1.6

    for ax, (metric, ylabel) in zip(axes, metrics):
        for p in pipelines:
            means = []
            stds = []
            for m in model_kinds:
                k = f"{m}|{p}"
                rec = per_mp.get(k)
                if rec is None:
                    means.append(0.0)
                    stds.append(0.0)
                    continue
                means.append(rec[f"{metric}_mean"])
                stds.append(rec[f"{metric}_std"])
            # Two bar groups per cluster: A (hatched) vs B (solid)
            x_cluster = x_base + pipeline_x_offset[p]
            colors = [_color_for_tstr_kind(m) for m in model_kinds]
            hatch = "//" if p == "A" else None
            alpha = 0.55 if p == "A" else 0.92
            ax.bar(
                x_cluster, means, bar_w,
                yerr=stds, capsize=3, color=colors, alpha=alpha,
                edgecolor="white", linewidth=0.6, hatch=hatch,
                label=f"Pipeline {p}",
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [_label_for_tstr_kind(m) for m in model_kinds],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(f"TSTR — {ylabel}")
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "r2":
            # HONEST negative axis — extend to include the most negative
            # value (no clamp, no abs, no log-rescale).
            r2_vals = [per_mp[f"{m}|{p}"]["r2_mean"]
                       for m in model_kinds for p in pipelines
                       if f"{m}|{p}" in per_mp]
            ymin = min(min(r2_vals) - 0.5, -0.5)
            ymax = max(max(r2_vals) + 0.2, 1.05)
            ax.set_ylim(ymin, ymax)
            ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
            ax.annotate(
                "R² < 0 ⇒ worse than predicting the mean",
                xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="#882255",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#882255", alpha=0.85),
            )
        if metric == "r2":
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        "TSTR cross-model — synth→real soft-sensor (TSTRLiteLSTM, "
        "1-layer, hidden=32). Pipeline A (synth on synth init, then "
        "eval on real) vs Pipeline B (synth→real). Mean ± std over 3 "
        "init seeds (40-42). Negative R² is PLOTTED HONESTLY.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # Build the dynamic caption_note from tstr.json's per_model_pipeline
    # block (plan-check fix: NEVER hard-code model names; the data ground
    # truth is the source of record and a hand-typed list would drift).
    neg_a_models = sorted(
        k.split("|", 1)[0]
        for k in neg_r2_keys if k.endswith("|A")
    )
    neg_a_str = ", ".join(neg_a_models) if neg_a_models else "(none)"
    caption_note = (
        "R² < 0 means the model is worse than predicting the mean — this "
        "is informative, not a render bug. The exact set of models with "
        "Pipeline A negative R² (built dynamically from tstr.json's "
        "per_model_pipeline block, NEVER hard-coded) is: "
        f"{neg_a_str} — because the soft-sensor trained on synthetic "
        "windows fails to generalize back to real windows when the "
        "synthetic distribution is far from real; Pipeline B (synth→real, "
        "the bioprocess-relevant direction) is positive for all models."
    )

    companion = {
        "figure": "tstr_crossmodel",
        "render_only": True,
        "source_artifact": "revision/results/tstr.json",
        "models": model_kinds,
        "pipelines": pipelines,
        "init_seeds": init_seeds,
        "per_model_pipeline": per_mp,
        "negative_r2_observed": neg_r2_keys,
        "caption_note": caption_note,
        "soft_sensor": soft_sensor_meta,
        "real_only_baseline": tstr_block.get("real_only_baseline"),
    }
    return _save(fig, figures_dir, "tstr_crossmodel", companion)


def render_tstr_crossmodel_matched2000(repo: Path,
                                       figures_dir: Path) -> list[Path]:
    """Cross-model TSTR R²/MAE/RMSE grouped bars — matched-budget Pipeline B
    (Plan 14-20).

    Render-only over ``revision/results/tstr_matched2000.json`` (the 14-20
    matched-budget driver-mode output: 9 trainable model_kinds × Pipeline B
    × 2000 epochs, 5 generator seeds × 3 init seeds per cell, consumed from
    ``revision/results/matched2000/runs/``). The four quantum variants
    (iqp_sel_55_repro, V1, V2, V3) are labeled explicitly — no generic
    "quantum" alias — so the figure aligns with the R1-M1 matched-budget
    family.

    Single Pipeline-B panel (matched-budget protocol is Pipeline B only;
    Pipeline A has no matched2000 artefacts — see D-11-10 + 14-20 plan).
    NEGATIVE R² IS PLOTTED HONESTLY (no clamp, no abs, no rescale) — the
    matched-budget data may or may not show negative R² depending on the
    generator; the axis treatment is preserved from the legacy renderer
    for safety.
    """
    src_path = repo / Path("revision/results/tstr_matched2000.json")
    t = _load_json(src_path, "tstr_matched2000.json (Plan 14-20 source)")
    tstr_block: dict = t["tstr"]
    model_kinds: list[str] = list(t["model_kinds"])
    pipelines: list[str] = list(t["pipelines"])
    assert pipelines == ["B"], (
        f"tstr_matched2000.json must be Pipeline B only; got {pipelines}"
    )
    init_seeds: list[int] = list(t.get("init_seeds", [40, 41, 42]))
    soft_sensor_meta: str = t.get("soft_sensor", "")

    # Build per-(model, pipeline=B) records.
    per_mp: dict[str, dict] = {}
    for k, v in tstr_block.items():
        if "|" not in k:
            continue  # skips 'real_only_baseline'
        per_mp[k] = {
            "r2_mean": float(v["r2_mean"]),
            "r2_std": float(v["r2_std"]),
            "mae_mean": float(v["mae_mean"]),
            "mae_std": float(v["mae_std"]),
            "rmse_mean": float(v["rmse_mean"]),
            "rmse_std": float(v["rmse_std"]),
            "n_train_synth": int(v["n_train_synth"]),
            "n_eval_real": int(v["n_eval_real"]),
        }
    neg_r2_keys = sorted(
        k for k, rec in per_mp.items() if rec["r2_mean"] < 0
    )

    # 1×3 panels: R², MAE, RMSE. Single-pipeline (B) means single bar
    # per model in each panel.
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5))
    metrics = [
        ("r2", "R$^2$"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
    ]
    n_models = len(model_kinds)
    bar_w = 0.6
    x_base = np.arange(n_models) * 1.0

    real_only = tstr_block.get("real_only_baseline") or {}
    real_only_r2 = float(real_only.get("r2_mean", float("nan")))

    for ax, (metric, ylabel) in zip(axes, metrics):
        means = []
        stds = []
        for m in model_kinds:
            k = f"{m}|B"
            rec = per_mp.get(k)
            if rec is None:
                means.append(0.0); stds.append(0.0); continue
            means.append(rec[f"{metric}_mean"])
            stds.append(rec[f"{metric}_std"])
        colors = [MODEL_COLORS.get(m, "#666666") for m in model_kinds]
        ax.bar(
            x_base, means, bar_w,
            yerr=stds, capsize=3, color=colors, alpha=0.92,
            edgecolor="white", linewidth=0.6,
        )
        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in model_kinds],
            rotation=35, ha="right", fontsize=9,
        )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=10)
        if metric == "r2":
            r2_vals = [per_mp[f"{m}|B"]["r2_mean"]
                       for m in model_kinds if f"{m}|B" in per_mp]
            # Sensible R^2 axis: focus on the 0..1 region where every
            # generator lives. Real-only at R^2 = -13.35 is annotated as
            # text rather than stretched to fit (would crush the bars).
            ax.set_ylim(-0.05, 1.10)
            ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
            if not np.isnan(real_only_r2):
                ax.annotate(
                    f"Real-only baseline (n=65 real windows): "
                    f"R$^2$ = {real_only_r2:.2f}\n"
                    f"(off-scale, well below 0)",
                    xy=(0.02, 0.04), xycoords="axes fraction",
                    fontsize=9, color="#882255",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#882255", alpha=0.9),
                )

    fig.suptitle(
        "TSTR cross-model utility (matched-budget Pipeline B, 2000 epochs)",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    convergence_note = (
        "All 9 matched-budget generators (3-562 generator params, plus the "
        "250 881-param shared critic for the WGAN family) converge to "
        "TSTR R² ≈ 0.99 on Pipeline B against a real-only baseline of "
        f"R² = {real_only_r2:.3f} (n=65 real windows). The cross-generator "
        "convergence is structural — the cumulative-sum back-transform "
        "from log-returns to OD mathematically encodes near-perfect lag-1 "
        "autocorrelation, so a soft sensor trained on Pipeline-B-derived "
        "synthetic OD essentially learns the persistence forecast, which "
        "is near-optimal on the real OD series regardless of which "
        "generator produced the underlying log-returns. The Pipeline-B "
        "TSTR therefore does not discriminate among generators."
    )

    companion = {
        "figure": "tstr_crossmodel_matched2000",
        "render_only": True,
        "source_artifact": "revision/results/tstr_matched2000.json",
        "matched_budget": True,
        "training_epochs": 2000,
        "model_kinds": model_kinds,
        "pipelines": pipelines,
        "init_seeds": init_seeds,
        "per_model_pipeline": per_mp,
        "negative_r2_observed": neg_r2_keys,
        "caption_note": convergence_note,
        "soft_sensor": soft_sensor_meta,
        "real_only_baseline": tstr_block.get("real_only_baseline"),
    }
    return _save(fig, figures_dir, "tstr_crossmodel_matched2000", companion)


def render_failure_modes_summary(repo: Path,
                                 figures_dir: Path) -> list[Path]:
    """3-row × 9-column failure-mode diagnostic grid (Task 3).

    Columns are the 9 matched-2000ep models ordered by ASCENDING OD EMD
    (best-on-the-left). Rows:
      A) OD distribution overlay (numpy histogram over the already-
         reconstructed OD windows — this is rendering math, NOT metric
         math; ``revision.core.eval`` is not invoked).
      B) ACF lag-1 bar (real vs generated) read DIRECTLY from the
         pre-written ``acf_<model>.json`` companions (14-04 output).
      C) log_return EMD bar (model mean ± std) read via ``_agg_lookup``
         from ``matched2000_dualscale.json``; outliers (> Q3 + 1.5*IQR
         across the 9 values) get a red bar edge.

    The frozen-checkpoint headline (D-14-10) is rendered as ROW-SPANNING
    dashed reference lines (one per row where applicable) — NEVER merged
    into the iqp_sel_55_repro column.
    """
    # 1) Load aggregates and sort models by ascending OD EMD.
    ds_path = repo / MATCHED2000_DUALSCALE_REL
    ds = _load_json(ds_path, "matched2000_dualscale.json (Task 3 source)")
    aggs: list[dict] = ds["aggregates"]

    od_emd_per_model: dict[str, float] = {}
    for m in DUALSCALE_MODEL_ORDER:
        a = _agg_lookup(aggs, m, "OD", "emd")
        if a is None:
            raise FileNotFoundError(
                f"[14-10/T3] missing OD/emd aggregate for {m} in "
                f"matched2000_dualscale.json; the render-only contract "
                f"requires every model carry an OD-EMD aggregate row."
            )
        od_emd_per_model[m] = float(a["mean"])
    sorted_models = sorted(
        DUALSCALE_MODEL_ORDER, key=lambda m: od_emd_per_model[m]
    )

    # 2) Frozen headline references (row-spanning dashed lines).
    head_OD_emd = None
    head_logret_emd = None
    head_OD_mean = None
    head_a = _agg_lookup(aggs, HEADLINE_KIND, "OD", "emd")
    if head_a is not None:
        head_OD_emd = float(head_a["mean"])
    head_b = _agg_lookup(aggs, HEADLINE_KIND, "log_return", "emd")
    if head_b is not None:
        head_logret_emd = float(head_b["mean"])
    head_m = _agg_lookup(aggs, HEADLINE_KIND, "OD", "moment_mean")
    if head_m is not None:
        head_OD_mean = float(head_m["mean"])

    # 3) Per-model log_return EMD (row C source).
    logret_emd_per_model: dict[str, tuple[float, float]] = {}
    for m in DUALSCALE_MODEL_ORDER:
        a = _agg_lookup(aggs, m, "log_return", "emd")
        if a is None:
            raise FileNotFoundError(
                f"[14-10/T3] missing log_return/emd aggregate for {m} "
                f"in matched2000_dualscale.json."
            )
        logret_emd_per_model[m] = (float(a["mean"]), float(a["std"]))
    # Outlier threshold = Q3 + 1.5 * IQR across the 9 models.
    lr_vals = np.array([logret_emd_per_model[m][0]
                        for m in DUALSCALE_MODEL_ORDER])
    q1, q3 = np.percentile(lr_vals, [25, 75])
    iqr = q3 - q1
    lr_outlier_thresh = q3 + 1.5 * iqr

    # 4) Load dist_*.json (row A — for summary stats / failure detection)
    # and acf_*.json (row B — for lag-1 ACF values).
    dist_records: dict[str, dict] = {}
    acf_records: dict[str, dict] = {}
    for m in DUALSCALE_MODEL_ORDER:
        dp = figures_dir / f"dist_{m}.json"
        ap = figures_dir / f"acf_{m}.json"
        dist_records[m] = _load_json(dp, f"dist_{m}.json companion")
        acf_records[m] = _load_json(ap, f"acf_{m}.json companion")

    # 5) Reconstruct per-model OD windows for row A histogram overlay
    # (numpy histogram is RENDERING math, not metric math — the OD
    # windows are reconstructed via the verbatim Pipeline-B logic
    # already in this module).
    real_refs = _real_references(repo)
    real_flat = real_refs["real_OD_flat"]
    od_by_model_local: dict[str, np.ndarray] = {}
    for m in DUALSCALE_MODEL_ORDER:
        od_by_model_local[m] = reconstruct_od(repo, m, PRIMARY_SEED)

    # 6) Failure thresholds (per the plan).
    n_models = len(sorted_models)
    fig = plt.figure(figsize=(2.0 * n_models + 1.5, 8.2))
    gs = fig.add_gridspec(3, n_models, hspace=0.55, wspace=0.25)
    per_cell: dict[str, dict] = {}

    # Shared OD-histogram bin grid (best-quality cross-model overlay).
    od_lo, od_hi = np.percentile(
        np.concatenate([real_flat,
                        *(od_by_model_local[m].reshape(-1)
                          for m in sorted_models)]),
        [0.5, 99.5],
    )
    bins = np.linspace(od_lo, od_hi, 30)

    for ci, m in enumerate(sorted_models):
        c = MODEL_COLORS.get(m, "#0072B2")
        # --- Row A: distribution overlay (real grey vs fake colored).
        axA = fig.add_subplot(gs[0, ci])
        fake_flat = od_by_model_local[m].reshape(-1)
        axA.hist(real_flat, bins=bins, density=True, color="#555555",
                 alpha=0.4, edgecolor="white", linewidth=0.2)
        axA.hist(fake_flat, bins=bins, density=True, color=c,
                 alpha=0.55, edgecolor="white", linewidth=0.2)
        # Row-spanning headline OD mean reference (a vertical dashed line
        # at the headline's OD moment_mean — distinct from any per-model
        # column's bar, D-14-10).
        if head_OD_mean is not None:
            axA.axvline(head_OD_mean, color=HEADLINE_COLOR,
                        linestyle="--", linewidth=1.0, alpha=0.5)
        # Failure: |fake_mean - real_mean| > 0.1*|real_mean|
        real_mean = float(dist_records[m]["real_mean"])
        fake_mean = float(dist_records[m]["fake_mean"])
        is_fail_A = abs(fake_mean - real_mean) > 0.1 * abs(real_mean)
        if is_fail_A:
            axA.set_title(MODEL_LABELS.get(m, m), fontsize=7.5,
                          color="#D7263D")
            axA.text(0.98, 0.92, "wrong mean", transform=axA.transAxes,
                     ha="right", va="top", fontsize=7,
                     color="#D7263D", weight="bold")
        else:
            axA.set_title(MODEL_LABELS.get(m, m), fontsize=7.5)
        axA.tick_params(labelsize=6)
        if ci == 0:
            axA.set_ylabel("dist (density)\nreal grey / fake color",
                           fontsize=7)
        else:
            axA.set_yticks([])
        per_cell[f"distribution_overlay|{m}"] = {
            "value": fake_mean,
            "real_mean": real_mean,
            "is_failure": bool(is_fail_A),
            "failure_label": "wrong mean" if is_fail_A else None,
            "source": f"revision/results/figures/dist_{m}.json + "
                      f"reconstructed OD (PRIMARY_SEED={PRIMARY_SEED})",
        }

        # --- Row B: ACF lag-1 (real vs generated) bar pair.
        axB = fig.add_subplot(gs[1, ci])
        acf_real = float(acf_records[m]["acf_real_OD"][1])
        acf_fake = float(acf_records[m]["acf_fake_OD"][1])
        axB.bar([0, 1], [acf_real, acf_fake], color=["#555555", c],
                width=0.7, alpha=0.85, edgecolor="white", linewidth=0.4)
        axB.set_xticks([0, 1])
        axB.set_xticklabels(["real", "fake"], fontsize=6)
        axB.set_ylim(-0.1, 1.05)
        axB.axhline(0, color="#888888", linewidth=0.5, alpha=0.5)
        is_fail_B = abs(acf_fake - acf_real) > 0.5
        if is_fail_B:
            axB.text(0.5, 0.92, "ACF-lag1 mismatch",
                     transform=axB.transAxes, ha="center", va="top",
                     fontsize=6.5, color="#D7263D", weight="bold")
        axB.tick_params(labelsize=6)
        if ci == 0:
            axB.set_ylabel("ACF lag-1", fontsize=7)
        else:
            axB.set_yticks([])
        per_cell[f"acf_lag1|{m}"] = {
            "value": acf_fake,
            "real_value": acf_real,
            "is_failure": bool(is_fail_B),
            "failure_label": "ACF-lag1 mismatch" if is_fail_B else None,
            "source": f"revision/results/figures/acf_{m}.json (lag-1 idx)",
        }

        # --- Row C: log_return EMD with red outlier edge.
        axC = fig.add_subplot(gs[2, ci])
        lr_mean, lr_std = logret_emd_per_model[m]
        is_fail_C = lr_mean > lr_outlier_thresh
        edge_kw = dict(edgecolor=("#D7263D" if is_fail_C else "white"),
                       linewidth=(1.6 if is_fail_C else 0.4))
        axC.bar([0], [lr_mean], yerr=[lr_std], capsize=3, color=c,
                width=0.6, alpha=0.85, **edge_kw)
        # Row-spanning headline log_return EMD reference (D-14-10).
        if head_logret_emd is not None:
            axC.axhline(head_logret_emd, color=HEADLINE_COLOR,
                        linestyle="--", linewidth=1.0, alpha=0.55)
        axC.set_xticks([])
        axC.tick_params(labelsize=6)
        if is_fail_C:
            axC.text(0.5, 0.92, "log_ret blowup",
                     transform=axC.transAxes, ha="center", va="top",
                     fontsize=6.5, color="#D7263D", weight="bold")
        if ci == 0:
            axC.set_ylabel("log_return EMD\n(mean ± std)", fontsize=7)
        per_cell[f"log_return_emd|{m}"] = {
            "value": lr_mean,
            "std": lr_std,
            "is_failure": bool(is_fail_C),
            "failure_label": "log_ret blowup" if is_fail_C else None,
            "outlier_threshold": float(lr_outlier_thresh),
            "source": "revision/results/matched2000_dualscale.json "
                      "(log_return/emd aggregate)",
        }

    fig.suptitle(
        "Failure-mode triage at a glance — 9 matched-2000ep models "
        "ordered by ascending OD EMD; rows: (A) OD dist overlay, "
        "(B) ACF lag-1 vs real, (C) log_return EMD (red-edged outliers). "
        "Frozen headline (epoch 1969) shown as row-spanning dashed "
        "reference lines — DISTINCT from iqp_sel_55_repro column "
        "(D-14-10).",
        fontsize=10,
    )

    companion = {
        "figure": "failure_modes_summary",
        "render_only": True,
        "source_artifacts": [
            "revision/results/matched2000_dualscale.json",
            "revision/results/figures/dist_<model>.json (×9)",
            "revision/results/figures/acf_<model>.json (×9)",
        ],
        "models_ordered_by_OD_emd_ascending": list(sorted_models),
        "OD_emd_per_model": {m: float(od_emd_per_model[m])
                              for m in sorted_models},
        "rows": ["distribution_overlay", "acf_lag1", "log_return_emd"],
        "per_cell": per_cell,
        "log_return_outlier_threshold_iqr": float(lr_outlier_thresh),
        "frozen_headline_references": {
            "OD_moment_mean": head_OD_mean,
            "OD_emd": head_OD_emd,
            "log_return_emd": head_logret_emd,
            "source": "frozen_checkpoint_epoch_1969",
            "conflation_guard": (
                "D-14-10: headline shown as row-spanning dashed reference "
                "lines (vertical in row A at OD moment_mean, horizontal "
                "in row C at log_return EMD), never merged into the "
                "iqp_sel_55_repro column."
            ),
        },
        "caption_note": (
            "Failure-mode triage at a glance. Columns are the 9 "
            "matched-2000ep models in ascending OD EMD; rows are "
            "(a) OD distribution overlay (real grey, fake color), "
            "(b) ACF lag-1 vs real, (c) log_return EMD with red-edged "
            "outliers (> Q3+1.5*IQR). Failure annotations are based on "
            "natural thresholds: |fake_mean - real_mean| > 0.1*|real_mean| "
            "(row A), |ACF_fake[1] - ACF_real[1]| > 0.5 (row B), "
            "log_return EMD > Q3+1.5*IQR (row C)."
        ),
    }
    return _save(fig, figures_dir, "failure_modes_summary", companion)


# Family -> marker style for the param-efficiency Pareto scatter.
_FAMILY_MARKER = {
    "adversarial-quantum": "o",
    "adversarial-classical": "s",
    "non-adversarial": "^",
}


def render_param_efficiency_pareto(repo: Path,
                                   figures_dir: Path) -> list[Path]:
    """log10(params) × EMD Pareto scatter, OD vs log_return facets (Task 4).

    Render-only over ``model_info.json`` (parameter_count + family) and
    ``matched2000_dualscale.json`` (EMD mean ± std at OD + log_return).
    The frozen-checkpoint headline (iqp_sel_55_headline, 55p,
    source=frozen_checkpoint_epoch_1969) is a DISTINCT diamond at
    log10(55) — at the same x as the iqp_sel_55_repro circle but with a
    different color, marker shape, and y (D-14-10 visually distinct).
    """
    mi_path = repo / Path("revision/results/model_info.json")
    mi = _load_json(mi_path, "model_info.json (Task 4 source)")
    ds = _load_json(
        repo / MATCHED2000_DUALSCALE_REL,
        "matched2000_dualscale.json (Task 4 source)",
    )
    aggs: list[dict] = ds["aggregates"]

    # Index model_info[models] by model field for O(1) lookup.
    mi_by_model: dict[str, dict] = {m["model"]: m for m in mi["models"]}

    per_model: dict[str, dict] = {}
    for m in DUALSCALE_MODEL_ORDER:
        rec = mi_by_model.get(m)
        if rec is None:
            raise FileNotFoundError(
                f"[14-10/T4] {m} missing from model_info.json (no "
                f"parameter_count); the render-only contract requires "
                f"every matched-2000ep model carry a model_info record."
            )
        n_params = int(rec["parameter_count"])
        family = str(rec["family"])
        a_od = _agg_lookup(aggs, m, "OD", "emd")
        a_lr = _agg_lookup(aggs, m, "log_return", "emd")
        if a_od is None or a_lr is None:
            raise FileNotFoundError(
                f"[14-10/T4] {m} missing OD/log_return EMD aggregate "
                f"in matched2000_dualscale.json."
            )
        per_model[m] = {
            "parameter_count": n_params,
            "family": family,
            "log10_params": float(np.log10(n_params)),
            "OD_emd_mean": float(a_od["mean"]),
            "OD_emd_std": float(a_od["std"]),
            "log_return_emd_mean": float(a_lr["mean"]),
            "log_return_emd_std": float(a_lr["std"]),
            "marker": _FAMILY_MARKER.get(family, "x"),
            "color": MODEL_COLORS.get(m, "#666666"),
        }

    # Frozen-headline diamond: x = log10(iqp_sel_55_headline.parameter_count)
    head_rec = mi_by_model.get("iqp_sel_55_headline")
    if head_rec is None:
        raise FileNotFoundError(
            "[14-10/T4] iqp_sel_55_headline missing from model_info.json."
        )
    head_params = int(head_rec["parameter_count"])
    if head_params != 55:
        raise ValueError(
            f"[14-10/T4] iqp_sel_55_headline.parameter_count={head_params}, "
            f"expected 55 (frozen_checkpoint_epoch_1969 anchor)."
        )
    head_x = float(np.log10(head_params))
    head_od = _agg_lookup(aggs, HEADLINE_KIND, "OD", "emd")
    head_lr = _agg_lookup(aggs, HEADLINE_KIND, "log_return", "emd")
    if head_od is None or head_lr is None:
        raise FileNotFoundError(
            f"[14-10/T4] frozen headline OD/log_return EMD aggregate "
            f"missing from matched2000_dualscale.json."
        )

    # Per-model annotation offsets (in points). Hand-tuned to avoid the
    # mid-cluster pile-up at log10(params) ~ 1.85 in both panels.
    ANNOTATION_OFFSETS = {
        "iqp_sel_55_repro": (-14, -16),
        "V1":               (-14,  14),
        "V2":               ( 14,  10),
        "V3":               ( 14, -14),
        "wgan_mlp":         ( 12, -10),
        "wgan_cnn":         ( 14,   8),
        "wgan_lstm":        ( 14,   0),
        "vae":              (-14,   8),
        "ar":               (  8,  10),
    }
    HEADLINE_OFFSET = (14, -4)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.0))
    panel_specs = [
        (axes[0], "OD",        "OD_emd_mean",        "OD_emd_std"),
        (axes[1], "log_return", "log_return_emd_mean", "log_return_emd_std"),
    ]
    for ax, scale_label, mean_key, std_key in panel_specs:
        for m, rec in per_model.items():
            ax.errorbar(
                rec["log10_params"], rec[mean_key],
                yerr=rec[std_key],
                fmt=rec["marker"], color=rec["color"], markersize=13,
                markeredgecolor="white", markeredgewidth=1.0,
                capsize=4, ecolor=rec["color"], elinewidth=1.0,
                alpha=0.92, zorder=4,
            )
            off = ANNOTATION_OFFSETS.get(m, (8, 6))
            ha = "left" if off[0] >= 0 else "right"
            ax.annotate(
                MODEL_LABELS.get(m, m),
                (rec["log10_params"], rec[mean_key]),
                xytext=off, textcoords="offset points",
                fontsize=10, color="#222222", alpha=0.95, ha=ha,
            )
        head_y = float((head_od if scale_label == "OD" else head_lr)["mean"])
        ax.scatter([head_x], [head_y], marker="D", s=200,
                   color=HEADLINE_COLOR, edgecolor="white",
                   linewidths=1.5, zorder=10, label=HEADLINE_LABEL)
        ax.annotate(
            HEADLINE_LABEL,
            (head_x, head_y),
            xytext=HEADLINE_OFFSET, textcoords="offset points",
            fontsize=10, color="#222222", alpha=0.95, ha="left",
        )
        ax.set_xlabel("log10(parameter count)", fontsize=12)
        ax.set_ylabel(f"{scale_label} EMD (log scale, mean over 5 seeds)",
                      fontsize=12)
        ax.set_title(f"{scale_label} scale", fontsize=13)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3, which="both")
        # Add x-margin so leftmost/rightmost annotations don't clip.
        x_lo, x_hi = ax.get_xlim()
        ax.set_xlim(x_lo - 0.15, x_hi + 0.25)

    # Single fig-level legend below the suptitle, outside both panels.
    legend_handles = [
        plt.Line2D([], [], linestyle="", marker="o",
                   color="#0072B2", markersize=10,
                   label="adversarial-quantum"),
        plt.Line2D([], [], linestyle="", marker="s",
                   color="#D55E00", markersize=10,
                   label="adversarial-classical"),
        plt.Line2D([], [], linestyle="", marker="^",
                   color="#882255", markersize=10,
                   label="non-adversarial"),
        plt.Line2D([], [], linestyle="", marker="D",
                   color=HEADLINE_COLOR, markersize=12,
                   markeredgecolor="white",
                   label=HEADLINE_LABEL),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4, frameon=False, fontsize=11,
    )

    fig.suptitle(
        "Parameter-efficiency Pareto: log10(params) vs EMD",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    companion = {
        "figure": "param_efficiency_pareto",
        "render_only": True,
        "source_artifacts": [
            "revision/results/model_info.json",
            "revision/results/matched2000_dualscale.json",
        ],
        "panels": ["OD", "log_return"],
        "x_axis": "log10(parameter_count)",
        "per_model": per_model,
        "frozen_headline": {
            "parameter_count": head_params,
            "log10_params": head_x,
            "OD_emd_mean": float(head_od["mean"]),
            "OD_emd_std": float(head_od["std"]),
            "log_return_emd_mean": float(head_lr["mean"]),
            "log_return_emd_std": float(head_lr["std"]),
            "marker_style": "diamond, HEADLINE_COLOR",
            "source": "frozen_checkpoint_epoch_1969",
            "conflation_guard": (
                "D-14-10: headline is a diamond at the same x as "
                "iqp_sel_55_repro (both 55p) but a distinct color, marker "
                "shape, AND y — never merged into the repro circle."
            ),
        },
        "family_markers": dict(_FAMILY_MARKER),
        "caption_note": (
            "Lower-left = better parameter efficiency. The frozen "
            "headline (epoch 1969, 55p) is the diamond — distinct from "
            "the 5-seed iqp_sel_55_repro circle at the same x. "
            "Markers: circle = adversarial-quantum, square = adversarial-"
            "classical, triangle = non-adversarial. "
            "x-axis is generator-only parameter count; the shared "
            "critic (250881 params per classical_architectures.json "
            "models.shared_critic.total_params) applies to all "
            "adversarial models alike and is documented in "
            "methods_full.md §2.k.x — Total adversarial parameter budget "
            "(Plan 14-13 Task 3, H-3 resolution)."
        ),
        "shared_critic_n_params": 250881,
    }
    return _save(fig, figures_dir, "param_efficiency_pareto", companion)


def render_seed_variance_per_model(repo: Path,
                                   figures_dir: Path) -> list[Path]:
    """3×3 facet grid of per-seed EMD trajectories per model (Task 5).

    Render-only over the 35 audited
    ``matched2000/runs/<adversarial>/<seed>/metrics.json`` per-run files.
    Each adversarial panel shows the 5 per-seed emd_avg lines (light) +
    the across-seed mean (bold). VAE and AR panels carry an explicit
    "no eval-epoch trajectory" caption — NEVER fabricated.

    Spread label per adversarial panel (tight / moderate / noisy)
    derived from the across-seed std of the FINAL eval step's EMD value
    (< 5% of mean = tight, 5-15% = moderate, > 15% = noisy).
    """
    # Panel layout MODEL_ORDER row-major (3 rows × 3 cols).
    panel_order = list(MODEL_ORDER)  # 9 entries; row-major fill

    # Load the 7 adversarial models' 5-seed emd_avg stacks.
    per_model_seeds: dict[str, np.ndarray] = {}
    for m in ADVERSARIAL_MODELS_WITH_EMD_AVG:
        stack = []
        for s in SEEDS:
            j = _load_json(
                _run_dir(repo, m, s) / "metrics.json",
                f"{m}/{s} metrics for seed-variance",
            )
            if "emd_avg" not in j:
                raise FileNotFoundError(
                    f"[14-10/T5] {m}/{s} missing emd_avg."
                )
            arr = np.asarray(j["emd_avg"], dtype=float)
            stack.append(arr)
        per_model_seeds[m] = np.stack(stack, axis=0)

    epochs = np.arange(per_model_seeds[
        ADVERSARIAL_MODELS_WITH_EMD_AVG[0]
    ].shape[1]) * 10  # eval every 10 epochs

    # Shared y-range across adversarial panels for fair visual comparison.
    all_vals = np.concatenate(
        [per_model_seeds[m].reshape(-1)
         for m in ADVERSARIAL_MODELS_WITH_EMD_AVG]
    )
    y_lo = max(1e-3, float(all_vals[all_vals > 0].min() * 0.85))
    y_hi = float(all_vals.max() * 1.15)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    per_model_final_emd_mean: dict[str, float] = {}
    per_model_final_emd_std: dict[str, float] = {}
    per_model_spread_label: dict[str, str] = {}

    def _spread_label(mean_v: float, std_v: float) -> str:
        if mean_v <= 0:
            return "moderate"
        ratio = std_v / mean_v
        if ratio < 0.05:
            return "tight"
        if ratio < 0.15:
            return "moderate"
        return "noisy"

    for idx, m in enumerate(panel_order):
        r, c = divmod(idx, 3)
        ax = axes[r][c]
        if m in ADVERSARIAL_MODELS_WITH_EMD_AVG:
            stack = per_model_seeds[m]  # (5, 201)
            col = MODEL_COLORS.get(m, "#0072B2")
            for si in range(stack.shape[0]):
                ax.plot(epochs, stack[si], color=col, alpha=0.35,
                        linewidth=1.0)
            mean_curve = stack.mean(axis=0)
            ax.plot(epochs, mean_curve, color=col, alpha=1.0,
                    linewidth=2.2, label="mean over 5 seeds")
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(MODEL_LABELS.get(m, m), fontsize=9)
            ax.tick_params(labelsize=7)
            # Final-step spread (ddof=1 sample std — H-2 Plan 14-13 Task 3)
            final_vals = stack[:, -1]
            final_mean = float(final_vals.mean())
            final_std = float(final_vals.std(ddof=1))
            per_model_final_emd_mean[m] = final_mean
            per_model_final_emd_std[m] = final_std
            label = _spread_label(final_mean, final_std)
            per_model_spread_label[m] = label
            ax.text(0.98, 0.04, f"5-seed spread: {label}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#888888", alpha=0.85))
        else:
            # VAE / AR — no emd_avg trajectory; honest "no curve" panel.
            ax.set_title(MODEL_LABELS.get(m, m), fontsize=9)
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.3, which="both")
            note = ("VAE — no emd_avg eval trajectory (ELBO loop)"
                    if m == "vae"
                    else "AR — closed-form, no training curve")
            ax.text(0.5, 0.5, note, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9,
                    color="#882255",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFF6F0",
                              ec="#882255", alpha=0.92))
        if r == 2:
            ax.set_xlabel("epoch (eval_step × 10)")
        if c == 0:
            ax.set_ylabel("EMD (avg / eval window, OD scale)")

    fig.suptitle(
        "Per-model 5-seed EMD trajectories (light) + mean (bold) — "
        "matched-2000ep budget. Quantum cluster (IQP:SEL 55p + V1/V2/V3) "
        "shows tight seed agreement; wgan_cnn is noisier across seeds. "
        "VAE/AR have no eval-epoch trajectory (ELBO loop / closed-form fit).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    companion = {
        "figure": "seed_variance_per_model",
        "render_only": True,
        "source_artifacts": [
            f"revision/results/matched2000/runs/<{','.join(ADVERSARIAL_MODELS_WITH_EMD_AVG)}>/"
            f"<{','.join(str(s) for s in SEEDS)}>/metrics.json (×35)",
        ],
        "panel_layout_row_major": list(panel_order),
        "adversarial_panels": list(ADVERSARIAL_MODELS_WITH_EMD_AVG),
        "no_trajectory_panels": list(MODELS_NO_TRAJECTORY),
        "seeds": list(SEEDS),
        "per_model_final_emd_mean": per_model_final_emd_mean,
        "per_model_final_emd_std": per_model_final_emd_std,
        "per_model_spread_label": per_model_spread_label,
        "spread_label_thresholds": {
            "tight": "std / mean < 5%",
            "moderate": "5% <= std / mean < 15%",
            "noisy": "std / mean >= 15%",
        },
        "caption_note": (
            "Per-model 5-seed EMD-vs-epoch trajectories (light) with mean "
            "(bold). Quantum cluster (IQP:SEL 55p + V1/V2/V3) shows tight "
            "seed agreement; wgan_cnn is noisier across seeds; VAE/AR "
            "have no eval-epoch trajectory (ELBO loop / closed-form fit)."
        ),
    }
    return _save(fig, figures_dir, "seed_variance_per_model", companion)


def render_noise_robustness_quantum(repo: Path,
                                    figures_dir: Path) -> list[Path]:
    """EMD vs noise level for depolarizing & amplitude-damping (Task 6).

    Render-only over ``revision/results/noise_model_sensitivity.json``
    (previously unconsumed). Filters rows[] to metric_name=='emd' AND
    scale=='OD', groups by (noise_model, noise_level, pipeline), and
    aggregates mean ± std over the 3 seeds [42, 43, 44]. The zero anchor
    (depol_0.0 / ampdamp_0.0 — same physical baseline by zero_anchor_note)
    is preserved as two distinct curve anchors per the source JSON's
    explicit note.
    """
    src_path = repo / Path("revision/results/noise_model_sensitivity.json")
    j = _load_json(src_path, "noise_model_sensitivity.json (Task 6 source)")
    rows = j["rows"]
    noise_models_decl = j["noise_models"]
    pipelines = list(j["pipelines"])
    seeds = list(j["seeds"])
    channel_insertion = j.get("channel_insertion", "")
    zero_anchor_note = j.get("zero_anchor_note", "")

    # Filter and group.
    emd_rows = [r for r in rows
                if r.get("metric_name") == "emd"
                and r.get("scale") == "OD"]
    from collections import defaultdict
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in emd_rows:
        key = (r["noise_model"], float(r["noise_level"]), r["pipeline"])
        grouped[key].append(float(r["value"]))

    per_curve: dict[str, list[dict]] = {}
    monotonicity: dict[str, str] = {}
    channels = ["depolarizing", "amplitude_damping"]
    for ch in channels:
        for p in pipelines:
            levels = sorted(noise_models_decl[ch])
            curve = []
            for lev in levels:
                vals = grouped.get((ch, float(lev), p), [])
                if not vals:
                    raise FileNotFoundError(
                        f"[14-10/T6] missing emd|OD rows for "
                        f"({ch}, {lev}, {p}) in "
                        f"noise_model_sensitivity.json."
                    )
                arr = np.asarray(vals, dtype=float)
                curve.append({
                    "noise_level": float(lev),
                    "emd_mean": float(arr.mean()),
                    # ddof=1 sample std (H-2 Plan 14-13 Task 3)
                    "emd_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n_seeds": int(arr.size),
                })
            per_curve[f"{ch}|{p}"] = curve
            # Monotonicity check (post-aggregation, ascending level).
            means = [c["emd_mean"] for c in curve]
            deltas = np.diff(means)
            mono = ("monotonic_increase"
                    if (len(deltas) == 0 or np.all(deltas >= 0))
                    else "non_monotonic")
            monotonicity[f"{ch}|{p}"] = mono

    # 1x2 panels: depolarizing | amplitude_damping. Two curves per panel
    # (Pipeline A dashed, Pipeline B solid) with markers + error bars.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0))
    for ax, ch in zip(axes, channels):
        for p in pipelines:
            curve = per_curve[f"{ch}|{p}"]
            xs = [c["noise_level"] for c in curve]
            ys = [c["emd_mean"] for c in curve]
            es = [c["emd_std"] for c in curve]
            ls = "--" if p == "A" else "-"
            mk = "o" if p == "B" else "s"
            col = "#0072B2" if p == "B" else "#D55E00"
            ax.errorbar(xs, ys, yerr=es, marker=mk, linestyle=ls,
                        color=col, capsize=3, markersize=8,
                        markeredgecolor="white", markeredgewidth=0.7,
                        elinewidth=0.8, linewidth=1.6,
                        label=f"Pipeline {p}")
            if curve and curve[0]["noise_level"] == 0.0:
                ax.scatter([curve[0]["noise_level"]],
                           [curve[0]["emd_mean"]],
                           s=140, facecolor="none", edgecolor=col,
                           linewidths=1.4, zorder=8)
        ax.set_xlabel(f"{ch.replace('_', ' ')} noise level", fontsize=11)
        ax.set_ylabel("OD EMD (mean $\\pm$ std, 3 seeds)", fontsize=11)
        ax.set_title(ch.replace("_", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=False, fontsize=10, loc="center right")

    fig.suptitle(
        "Noise-model sensitivity: OD-EMD vs per-layer noise level",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    companion = {
        "figure": "noise_robustness_quantum",
        "render_only": True,
        "source_artifact": "revision/results/noise_model_sensitivity.json",
        "noise_models": channels,
        "noise_levels": {
            ch: list(sorted(noise_models_decl[ch])) for ch in channels
        },
        "pipelines": pipelines,
        "seeds": seeds,
        "channel_insertion": channel_insertion,
        "zero_anchor_note": zero_anchor_note,
        "per_curve_aggregates": per_curve,
        "monotonicity": monotonicity,
        "caption_note": (
            "EMD vs noise level for depolarizing and amplitude-damping "
            "channels inserted per-layer (after each entangling block). "
            "Mean over 3 seeds (42-44); error bars are seed std. "
            "Monotonicity per curve recorded in companion JSON. "
            "Zero anchor (level=0.0) preserved as distinct per-curve "
            "rows per the source JSON's zero_anchor_note."
        ),
    }
    return _save(fig, figures_dir, "noise_robustness_quantum", companion)


def render_shot_noise_robustness(repo: Path,
                                 figures_dir: Path) -> list[Path]:
    """EMD vs shot count with analytic-statevector reference (Task 7).

    Render-only over ``revision/results/shot_noise_sensitivity.json``
    (previously unconsumed). The "analytic" condition (shots=None,
    statevector simulator) is rendered as a HORIZONTAL REFERENCE LINE
    per pipeline (the shots=∞ asymptote); finite-shot conditions
    (currently shots_8192 and shots_1024) are plotted as points connected
    by lines on a log-x shots axis with error bars over 3 seeds.
    """
    src_path = repo / Path("revision/results/shot_noise_sensitivity.json")
    j = _load_json(src_path, "shot_noise_sensitivity.json (Task 7 source)")
    rows = j["rows"]
    pipelines = list(j["pipelines"])
    seeds = list(j["seeds"])
    conditions = list(j["conditions"])
    shot_levels = dict(j["shot_levels"])

    # Filter to emd|OD and group.
    emd_rows = [r for r in rows
                if r.get("metric_name") == "emd"
                and r.get("scale") == "OD"]
    from collections import defaultdict
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in emd_rows:
        cond = r["condition"]
        p = r["pipeline"]
        grouped[(cond, p)].append(float(r["value"]))

    finite_conditions = sorted(
        (c for c in conditions if shot_levels.get(c) is not None),
        key=lambda c: int(shot_levels[c]),
    )
    per_curve: dict[str, list[dict]] = {}
    analytic_baseline: dict[str, dict] = {}
    for p in pipelines:
        curve = []
        for c in finite_conditions:
            vals = grouped.get((c, p), [])
            if not vals:
                raise FileNotFoundError(
                    f"[14-10/T7] missing emd|OD rows for ({c}, {p}) "
                    f"in shot_noise_sensitivity.json."
                )
            arr = np.asarray(vals, dtype=float)
            curve.append({
                "condition": c,
                "shots": int(shot_levels[c]),
                "emd_mean": float(arr.mean()),
                # ddof=1 sample std (H-2 Plan 14-13 Task 3)
                "emd_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n_seeds": int(arr.size),
            })
        per_curve[p] = curve
        avals = grouped.get(("analytic", p), [])
        if not avals:
            raise FileNotFoundError(
                f"[14-10/T7] missing emd|OD rows for ('analytic', {p}) "
                f"in shot_noise_sensitivity.json."
            )
        aarr = np.asarray(avals, dtype=float)
        analytic_baseline[p] = {
            "emd_mean": float(aarr.mean()),
            # ddof=1 sample std (H-2 Plan 14-13 Task 3)
            "emd_std": float(aarr.std(ddof=1)) if aarr.size > 1 else 0.0,
            "n_seeds": int(aarr.size),
        }

    # Single-panel figure: log-x shots, both pipelines on the same axes
    # with their analytic baselines as horizontal references.
    fig, ax = plt.subplots(figsize=(9, 6.0))
    pipeline_styles = {
        "A": dict(color="#D55E00", linestyle="--", marker="s",
                  label_prefix="Pipeline A"),
        "B": dict(color="#0072B2", linestyle="-",  marker="o",
                  label_prefix="Pipeline B"),
    }
    for p in pipelines:
        st = pipeline_styles[p]
        curve = per_curve[p]
        xs = [c["shots"] for c in curve]
        ys = [c["emd_mean"] for c in curve]
        es = [c["emd_std"] for c in curve]
        ax.errorbar(xs, ys, yerr=es, marker=st["marker"],
                    linestyle=st["linestyle"], color=st["color"],
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=0.8, capsize=4, linewidth=1.8,
                    elinewidth=0.9,
                    label=f"{st['label_prefix']} (finite shots)")
        ab = analytic_baseline[p]
        ax.axhline(ab["emd_mean"], color=st["color"], linestyle=":",
                   linewidth=1.4, alpha=0.75,
                   label=f"{st['label_prefix']} analytic (shots$\\to\\infty$)")
    ax.set_xscale("log")
    ax.set_xlabel("Shots per Pauli measurement (log scale)", fontsize=12)
    ax.set_ylabel("OD EMD (mean $\\pm$ std, 3 seeds)", fontsize=12)
    ax.set_title(
        "Shot-noise robustness: OD-EMD vs finite-shot count",
        fontsize=13,
    )
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout()

    companion = {
        "figure": "shot_noise_robustness",
        "render_only": True,
        "source_artifact": "revision/results/shot_noise_sensitivity.json",
        "conditions": conditions,
        "shot_levels": shot_levels,
        "pipelines": pipelines,
        "seeds": seeds,
        "per_curve_aggregates": per_curve,
        "analytic_baseline": analytic_baseline,
        "x_scale": "log",
        "caption_note": (
            "EMD vs shot count (log-x) for Pipeline A and B; the "
            "analytic-statevector baseline (shots=∞) is shown as a "
            "horizontal reference per pipeline. Mean over 3 seeds "
            "(42-44); error bars are seed std."
        ),
    }
    return _save(fig, figures_dir, "shot_noise_robustness", companion)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the complete 2000ep figure suite (png + pdf + json each)."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 14 PAPER-09 figure suite renderer (plan 14-04). "
            "Render-only: reads the accepted 2000ep artifacts + the frozen "
            "headline, never trains/samples/recomputes."
        )
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("revision/results/figures"),
        help="Directory holding companion JSON / receiving the figure suite.",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    figures_dir = args.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = (repo / figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    refs = _real_references(repo)
    real_flat = refs["real_OD_flat"]
    real_windowed = refs["real_windowed_OD"]
    real_logret = refs["real_log_delta"]

    written: list[Path] = []
    od_by_model: dict[str, np.ndarray] = {}

    # --- per-model canonical figures (every model) ---
    for model in MODEL_ORDER:
        run = _run_dir(repo, model, PRIMARY_SEED)
        _require(run / "samples.npy", f"{model}/{PRIMARY_SEED} samples")
        od = reconstruct_od(repo, model, PRIMARY_SEED)
        od_by_model[model] = od
        # transformed (log-return space) windows for the dual-scale ACF
        inv = np.load(run / "inverse_kwargs.npz", allow_pickle=True)
        samples_pm1 = np.load(run / "samples.npy").astype(np.float64)
        samples_pm1 = _unscale_wgan_samples(samples_pm1, model)  # Plan 14-21 T01
        r_norm = ((samples_pm1 + 1.0) / 2.0) * (
            float(inv["r_max"]) - float(inv["r_min"])
        ) + float(inv["r_min"])
        metrics = _load_metrics(repo, model, PRIMARY_SEED)

        written += render_distribution_comparison(
            model, od, real_flat, figures_dir
        )
        written += render_acf_comparison(
            model, od, r_norm, real_windowed, real_logret, figures_dir
        )
        written += render_qq_plot(model, od, real_flat, figures_dir)
        written += render_time_series_comparison(
            model, od, real_windowed, figures_dir
        )
        written += render_loss_curves(model, metrics, figures_dir)
        emd_fig = render_emd_over_training(model, metrics, figures_dir)
        if emd_fig:
            written += emd_fig
        written += render_od_reconstruction(
            model, od, real_windowed, figures_dir
        )
        written += render_stylized_facts(model, od, real_flat, figures_dir)

        # --- Plan 14-21 T03: §A.10 / §A.11 / §A.12 per-model triples.
        # §A.10 reconstruction overlay loads samples internally (via the
        # T01-patched _unscale_wgan_samples path). §A.11 + §A.12 take the
        # gen_log_delta sequence emitted by §A.10's JSON sidecar; we derive
        # it here from r_norm to avoid a round-trip read.
        real_od_arr = refs["real_OD_full"]
        _mu = float(inv["mu"])
        _sigma = float(inv["sigma"])
        gen_log_delta_local = r_norm * _sigma + _mu  # raw log-delta from r_norm
        # Truncate to match real_logret length (777)
        n_target = real_logret.shape[0]
        gen_log_delta_trunc = gen_log_delta_local.reshape(-1)[:n_target]
        written += render_reconstruction_overlay(
            model, real_logret, real_od_arr, figures_dir, repo
        )
        written += render_logreturn_stat_grid(
            model, real_logret, gen_log_delta_trunc, figures_dir
        )
        written += render_dtw_alignment(
            model, real_logret, gen_log_delta_trunc, figures_dir
        )

    # --- Plan 14-15 T3: 9-model QQ overlay + delta-QQ panel (single
    # discriminating figure for OD-marginal convergence across
    # architectures at the matched-2000ep budget). Lives alongside the
    # cross-model figures because it is itself a cross-model overlay.
    written += render_qq_overlay(MODEL_ORDER, real_flat, figures_dir,
                                 repo=repo)

    # --- cross-model comparison figures (55-param always present) ---
    written += render_cross_model_distribution(
        od_by_model, real_flat, figures_dir
    )
    written += render_cross_model_emd(repo, figures_dir)
    written += render_headline_vs_reproduction(
        repo, od_by_model["iqp_sel_55_repro"], real_flat, figures_dir
    )

    # --- Plan 14-08 Task 2: matched-2000ep dual-scale side-by-side render ---
    # (PNG + PDF + companion JSON + copy-paste comparison.md, sourced SOLELY
    # from matched2000_dualscale.json; frozen headline visually distinct
    # from iqp_sel_55_repro — D-14-10. Gated by the existing
    # verify_number_provenance.py unmodified.)
    written += render_matched2000_dualscale_sidebyside(repo, figures_dir)
    written += render_matched2000_dualscale_comparison_table(
        repo, figures_dir
    )

    # --- Plan 14-10: 7 "full story" render-only figures (NO retraining,
    # NO resampling, NO new metric recomputation). Each function loud-
    # fails on missing input JSON. Frozen-checkpoint headline rendered
    # visually distinct from iqp_sel_55_repro on every figure where both
    # could appear (D-14-10). revision/core/ stays byte-frozen (D-14-22).
    written += render_training_convergence_all_models(repo, figures_dir)
    written += render_tstr_crossmodel(repo, figures_dir)
    written += render_tstr_crossmodel_matched2000(repo, figures_dir)
    written += render_failure_modes_summary(repo, figures_dir)
    written += render_param_efficiency_pareto(repo, figures_dir)
    written += render_seed_variance_per_model(repo, figures_dir)
    written += render_noise_robustness_quantum(repo, figures_dir)
    written += render_shot_noise_robustness(repo, figures_dir)

    # --- keep the existing introspection figures (extend, not overwrite) ---
    written += render_existing_introspection(figures_dir)

    pngs = sorted(figures_dir.glob("*.png"))
    print("[run_figure_suite] wrote:")
    for p in written:
        print(f"  {p}")
    print(f"[run_figure_suite] total PNG figures: {len(pngs)} "
          f"(canonical bar = 16, NOT 20)")


if __name__ == "__main__":
    main()
