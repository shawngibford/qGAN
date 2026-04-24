"""Data pipeline: CSV load, log-returns, Lambert W, rolling windows, split.

Extracted verbatim from ``qgan_pennylane.ipynb`` (cells 5, 7, 9, 15, 17, 18,
21, 22, 23, 30). Behavior is identical to v1.1 notebook; this module is a
refactor, not a rewrite.

Phase 9 (EVAL-06) will replace ``inverse_lambert_w_transform`` with a fully
differentiable alternative; the present implementation is a scalar round-trip
using ``scipy.special.lambertw`` and therefore non-differentiable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis as _sp_kurtosis

from revision.core import DITHER, DITHER_SEED, PAR_LIGHT_MAX, WINDOW_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — normalization
# ─────────────────────────────────────────────────────────────────────────────
def normalize(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize to zero-mean / unit-variance.

    Notebook cell 7. Uses torch.std default (unbiased=True, ddof=1).
    Returns ``(normalized_data, mu, sigma)``.
    """
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma, mu, sigma


def denormalize(norm_data: torch.Tensor, mu_original: torch.Tensor, std_original: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`normalize`. Notebook cell 7."""
    return norm_data * std_original + mu_original


# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 / cell 9 — log-returns with dither
# ─────────────────────────────────────────────────────────────────────────────
def compute_log_delta(
    od_values: torch.Tensor,
    dither: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Compute log-returns with optional dithering.

    Notebook cell 7. ``log_delta[t] = log(od[t+1]) - log(od[t])``.
    If ``dither > 0``, add ``U(-dither, +dither)`` noise before the log.
    """
    od_np = od_values.numpy() if isinstance(od_values, torch.Tensor) else od_values.copy()
    if dither > 0:
        if rng is None:
            rng = np.random.default_rng()
        od_np = od_np + rng.uniform(-dither, dither, size=len(od_np))
    log_od = np.log(od_np)
    return torch.tensor(log_od[1:] - log_od[:-1], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 17 — Lambert W transforms
# ─────────────────────────────────────────────────────────────────────────────
def inverse_lambert_w_transform(data: torch.Tensor, delta: float) -> torch.Tensor:
    """Inverse Lambert W transform (heavy-tail → Gaussian-ish).

    Notebook cell 17. Uses ``scipy.special.lambertw`` on the principal branch
    (``.real``). Promotes to float64 for numerical stability — the notebook
    does the same (``data.double()``).

    Note
    ----
    Non-differentiable. Phase 9 (EVAL-06) replaces this with a differentiable
    alternative.
    """
    data = data.double()
    sign = torch.sign(data)
    data_squared = data ** 2
    lambert_input = (delta * data_squared).cpu().numpy()
    lambert_result = lambertw(lambert_input).real
    lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
    transformed_data = sign * torch.sqrt(lambert_tensor / delta)
    return transformed_data


def lambert_w_transform(
    transformed_data: torch.Tensor,
    delta: float,
    clip_low: float = -12.0,
    clip_high: float = 11.0,
) -> torch.Tensor:
    """Forward Lambert W transform (Gaussian-ish → heavy-tail).

    Notebook cell 17. Inverse of :func:`inverse_lambert_w_transform`.
    Promoted to float64 like the notebook; clipped to ``[clip_low, clip_high]``.
    """
    transformed_data = transformed_data.double()
    exp_term = torch.exp((delta / 2) * transformed_data ** 2)
    reversed_data = transformed_data * exp_term
    return torch.clamp(reversed_data, clip_low, clip_high)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 22 — rolling window + rescale
# ─────────────────────────────────────────────────────────────────────────────
def rolling_window(data: torch.Tensor, m: int, s: int) -> torch.Tensor:
    """Stride-``s`` rolling windows of length ``m``.

    Notebook cell 22. Returns tensor of shape ``((len(data)-m)//s + 1, m)``.
    """
    windows = []
    for i in range(0, len(data) - m + 1, s):
        windows.append(data[i:i + m])
    return torch.stack(windows)


def rescale(scaled_data: torch.Tensor, original_data: torch.Tensor) -> torch.Tensor:
    """Rescale ``scaled_data`` from ``[-1, 1]`` back to the range of ``original_data``.

    Notebook cell 22.
    """
    min_val = torch.min(original_data)
    max_val = torch.max(original_data)
    return 0.5 * (scaled_data + 1.0) * (max_val - min_val) + min_val


# ─────────────────────────────────────────────────────────────────────────────
# Cell 23 — full denormalization pipeline
# ─────────────────────────────────────────────────────────────────────────────
def full_denorm_pipeline(
    gen_windows: torch.Tensor,
    preprocessed_data: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Full reverse pipeline: ``windows → flat → rescale → Lambert_W → denormalize``.

    Notebook cell 23. Used by both training-time evaluation and standalone
    generation (BUG-03 fix: identical pipeline in both contexts).

    Args
    ----
    gen_windows
        Generated windows in scaled ``[-1, 1]`` space, shape ``(N, m)``.
    preprocessed_data
        Pre-Lambert (transformed_norm_log_delta) tensor used for :func:`rescale`
        reference range.
    mu, sigma
        Output of :func:`normalize` on the original log-delta.
    delta
        Lambert W tail parameter from :func:`find_optimal_lambert_delta`.
    """
    gen_flat = gen_windows.reshape(-1).double()
    rescaled = rescale(gen_flat, preprocessed_data)
    after_lambert = lambert_w_transform(rescaled, delta)
    original_scale = denormalize(after_lambert, mu, sigma)
    return original_scale


# ─────────────────────────────────────────────────────────────────────────────
# Cell 18 — optimal delta selection
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_lambert_delta(normed: np.ndarray) -> float:
    """Find Lambert W ``delta`` that minimizes ``|excess kurtosis|``.

    Notebook cell 18. Bounded scalar optimization over ``[0.01, 2.0]``.
    """
    def _kurt(d: float, data: np.ndarray) -> float:
        sign = np.sign(data)
        lr = lambertw(d * data ** 2).real
        lr = np.maximum(lr, 0)
        transformed = sign * np.sqrt(lr / d)
        return abs(_sp_kurtosis(transformed, fisher=True))

    result = minimize_scalar(_kurt, bounds=(0.01, 2.0), args=(normed,), method="bounded")
    return float(result.x)


# ─────────────────────────────────────────────────────────────────────────────
# Cells 5 + 9 + 15 + 18 + 21 + 30 — high-level entry point
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str | Path = "./data.csv") -> dict:
    """Run the full v1.1 data pipeline end-to-end.

    Reproduces notebook cells 5 (CSV load), 9 (log-delta + PAR_LIGHT),
    15 (normalize), 18 (find delta + apply inverse Lambert W), 21 (scale to
    ``[-1, 1]``), and 30 (rolling windows).

    Returns
    -------
    dict with keys:
        OD                           : torch.Tensor, shape (N,), float32
        PAR_LIGHT                    : torch.Tensor, shape (N,), float32
        log_delta                    : torch.Tensor, shape (N-1,), float32
        norm_log_delta               : torch.Tensor, normalized log_delta
        mu, sigma                    : torch scalars from normalize()
        delta                        : float, Lambert W tail parameter
                                       (computed by minimize_scalar)
        transformed_norm_log_delta   : torch.Tensor, post-inverse-Lambert-W
        scaled_data                  : torch.Tensor, rescaled to [-1, 1]
        windowed_data                : torch.Tensor, rolling windows of shape
                                       (M, WINDOW_LENGTH) with stride 2
        par_light_norm               : torch.Tensor, PAR_LIGHT[1:] / PAR_LIGHT_MAX
    """
    # Cell 5 — CSV load, OD column cleanup, PAR_LIGHT extraction
    full_data = pd.read_csv(str(csv_path))
    raw_data = full_data[["OD"]].copy()
    raw_data.columns = ["value"]
    raw_data["value"] = pd.to_numeric(raw_data["value"], errors="coerce")
    raw_data["value"] = raw_data["value"].fillna(
        raw_data["value"].rolling(window=10, min_periods=10).mean()
    )
    raw_data = raw_data.dropna()
    OD = torch.tensor(raw_data["value"].values, dtype=torch.float32)
    PAR_LIGHT = torch.tensor(full_data["PAR_LIGHT"].values, dtype=torch.float32)

    # Cell 9 — log-delta with dither + PAR_LIGHT alignment
    log_delta = compute_log_delta(
        OD, dither=DITHER, rng=np.random.default_rng(DITHER_SEED)
    )
    par_light_aligned = PAR_LIGHT[1:]
    par_light_norm = par_light_aligned / PAR_LIGHT_MAX

    # Cell 15 — normalize log-delta
    norm_log_delta, mu, sigma = normalize(log_delta)

    # Cell 18 — find optimal Lambert W delta, apply inverse Lambert
    delta = find_optimal_lambert_delta(norm_log_delta.numpy())
    transformed_norm_log_delta = inverse_lambert_w_transform(norm_log_delta, delta)

    # Cell 21 — rescale to [-1, 1]
    min_val = torch.min(transformed_norm_log_delta)
    max_val = torch.max(transformed_norm_log_delta)
    scaled_data = -1.0 + 2.0 * (transformed_norm_log_delta - min_val) / (max_val - min_val)

    # Cell 30 — rolling window (stride 2)
    windowed_data = rolling_window(scaled_data, WINDOW_LENGTH, 2)

    return {
        "OD": OD,
        "PAR_LIGHT": PAR_LIGHT,
        "log_delta": log_delta,
        "norm_log_delta": norm_log_delta,
        "mu": mu,
        "sigma": sigma,
        "delta": delta,
        "transformed_norm_log_delta": transformed_norm_log_delta,
        "scaled_data": scaled_data,
        "windowed_data": windowed_data,
        "par_light_norm": par_light_norm,
    }
