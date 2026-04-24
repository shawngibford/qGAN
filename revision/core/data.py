"""Data pipeline: CSV load, log-returns, Lambert W, rolling windows, split."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd  # noqa: F401  (reserved for plan 08-02 CSV load)
import torch


def normalize(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (normalized_data, mu, sigma). Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def denormalize(norm_data: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize(). Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_log_delta(od_values: torch.Tensor, dither: float = 0.005, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Compute log-returns with dither. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def lambert_w_transform(transformed_data: torch.Tensor, delta: float, clip_low: float = -12.0, clip_high: float = 11.0) -> torch.Tensor:
    """Forward Lambert W transform (heavy-tail -> Gaussian-ish). Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def inverse_lambert_w_transform(data: torch.Tensor, delta: float) -> torch.Tensor:
    """Inverse Lambert W transform.

    Plan 08-02 installs the notebook-inline scalar version; Phase 9 (EVAL-06)
    replaces with differentiable version.
    """
    raise NotImplementedError("Filled in by plan 08-02")


def rolling_window(data: torch.Tensor, m: int, s: int) -> torch.Tensor:
    """Rolling window of length m, stride s. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def rescale(scaled_data: torch.Tensor, original_data: torch.Tensor) -> torch.Tensor:
    """Min-max rescale to match original data range. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def full_denorm_pipeline(
    gen_windows: torch.Tensor,
    preprocessed_data: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Full denormalization chain (norm -> Lambert -> log-return -> OD). Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def load_and_preprocess(csv_path: str | Path = "./data.csv") -> dict:
    """High-level entrypoint.

    Returns dict with keys:
    {OD, PAR_LIGHT, log_delta, norm_log_delta, transformed_norm_log_delta,
     scaled_data, windowed_data, mu, sigma, delta}.

    Filled in by plan 08-02.
    """
    raise NotImplementedError("Filled in by plan 08-02")
