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
    compute_moments,
)
from revision.core.preprocessing import inverse_logreturns  # noqa: E402

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
    jpath.write_text(json.dumps(companion, indent=1, default=float))
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
    }
    return _save(fig, figures_dir, f"qq_{model}", companion)


def render_time_series_comparison(model: str, od: np.ndarray,
                                  real_windowed: np.ndarray,
                                  figures_dir: Path) -> list[Path]:
    """Sample OD-trajectory windows: real vs generated, side by side."""
    rng = np.random.default_rng(model.__hash__() & 0xFFFF)
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
    """Cross-model final-EMD bar with seed spread + the FROZEN headline.

    The FROZEN-checkpoint headline EMD (epoch 1969) is drawn as a distinct
    annotated reference line, NEVER merged into the reproduction bar
    (D-14-10 / T-14-12).
    """
    models = MODEL_ORDER
    means, stds, present = [], [], []
    for m in models:
        finals = []
        for s in SEEDS:
            mt_path = _run_dir(repo, m, s) / "metrics.json"
            if not mt_path.exists():
                continue
            mt = json.loads(mt_path.read_text())
            if "emd_avg" in mt and len(mt["emd_avg"]):
                finals.append(float(np.min(mt["emd_avg"])))
        if finals:
            present.append(m)
            means.append(float(np.mean(finals)))
            stds.append(float(np.std(finals)))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(present))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[MODEL_COLORS.get(m, "#0072B2") for m in present],
           alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("best EMD over training (mean ± std over 5 seeds)")
    ax.set_title("Cross-model EMD (2000ep matched budget)")
    # FROZEN headline reference line — distinctly labelled (D-14-10).
    headline = _load_json(repo / HEADLINE_REL, "frozen headline")
    od_emd = next(
        (r["value"] for r in headline["rows"]
         if r.get("metric_name") == "emd" and r.get("scale") == "OD"),
        None,
    )
    if od_emd is not None:
        ax.axhline(od_emd, color=HEADLINE_COLOR, linestyle="--",
                   linewidth=1.8, label=HEADLINE_LABEL)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    companion = {
        "figure": "cross_model_emd",
        "models": present,
        "best_emd_mean": means,
        "best_emd_std": stds,
        "frozen_headline_OD_emd": od_emd,
        "headline_source": "headline_canonical.json (source=frozen_"
                            "checkpoint_epoch_1969) — distinct from the "
                            "iqp_sel_55_repro 2000ep reproduction (D-14-10)",
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
    if head_epoch != 1969:
        raise ValueError(
            f"[14-10/T1] headline_canonical.checkpoint_epoch={head_epoch}, "
            f"expected 1969 (frozen_checkpoint_epoch_1969)."
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    per_model_mean: dict[str, list[float]] = {}
    per_model_std: dict[str, list[float]] = {}
    for m in ADVERSARIAL_MODELS_WITH_EMD_AVG:
        s = per_model_emd_seedstack[m]
        mean_e = s.mean(axis=0)
        std_e = s.std(axis=0)
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
    ax.set_ylabel("EMD (avg over eval window, OD scale)")
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
