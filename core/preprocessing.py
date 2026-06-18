"""Preprocessing pipelines — three ablation variants for R1-M3 (Phase 09.1).

Phase 9 implements only the Lambert W pair (it IS EVAL-06); the other four
functions are NotImplementedError stubs reserved for Phase 09.1.

Contract: each ``forward_X`` / ``inverse_X`` pair must satisfy
    max_abs(inverse_X(forward_X(x), *args), x) <= 1e-8
on a real OD trajectory (Phase 09.1 ABL-01).

D-10-05 (matched-budget pipeline selection)
--------------------------------------------
The matched-budget runs reported in the paper use **Pipeline B** exclusively
(log-returns standardized + linear rescale to [-1, 1]). Pipeline C
(log-returns + Lambert W, the v1.1 published pipeline) was dropped because
the Phase 09.1 ablation (results/transform_ablation/) showed it
tied with Pipeline B on every OD-scale metric while introducing an
over-Gaussianization concern flagged by reviewer R1-M3.

The Lambert W forward/inverse pair is **retained for reproducibility of
the ablation only** — it is not on the matched-budget training path. Do
not re-introduce it into Pipeline B or the matched-budget evaluation.
"""
from __future__ import annotations
from typing import Tuple
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline C (CURRENT PAPER) — log-returns + Lambert W
# ─────────────────────────────────────────────────────────────────────────────
# Re-export from data.py per D-07 (single source of truth).
# Phase 9 (EVAL-06) made inverse_lambert_w_transform differentiable; the
# forward (lambert_w_transform) was already pure-torch.
from core.data import (
    lambert_w_transform as forward_lambert,
    inverse_lambert_w_transform as inverse_lambert,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — log-returns only
# ─────────────────────────────────────────────────────────────────────────────
def forward_logreturns(
    od: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-returns r_t = ln(OD_{t+1}/OD_t), then standardize to zero-mean / unit-std.

    Length contract: returns length N-1 (drops one timestep). Returns
    (r_norm, mu, sigma). Uses torch.std default ddof=1 to match v1.1
    normalize() in data.py:36.

    Per CONTEXT.md D-09.1-01 / RESEARCH.md Q3 — single global mu/sigma across
    the full series; CD-5 anchors the inverse at a real per-window OD₀ supplied
    by the caller (see ``inverse_logreturns``).
    """
    log_od = torch.log(od)
    r = log_od[1:] - log_od[:-1]
    mu = torch.mean(r)
    sigma = torch.std(r)  # default ddof=1 — matches v1.1 normalize() at data.py:36
    return (r - mu) / sigma, mu, sigma


def inverse_logreturns(
    r_norm: torch.Tensor,
    od_start: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Un-standardize log-returns and cumulative-integrate from od_start.

    OD_t = OD_0 * exp(cumsum(r_1, ..., r_t)) along the last dim.

    Shape contract:
      - r_norm: (..., L-1) — last dim is the standardized log-return sequence
      - od_start: scalar OR (...,) broadcastable to r_norm.shape[:-1]
      - Returns: (..., L) — first entry along last dim equals od_start
    """
    r = r_norm * sigma + mu                            # un-standardize
    cum = torch.cumsum(r, dim=-1)                       # (..., L-1)
    pad = torch.zeros_like(cum[..., :1])                # (..., 1)
    cum_full = torch.cat([pad, cum], dim=-1)            # (..., L)
    log_od_start = torch.log(od_start)
    if log_od_start.dim() < cum_full.dim():
        log_od_start = log_od_start.unsqueeze(-1)
    log_od = log_od_start + cum_full
    return torch.exp(log_od)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — min-max normalized OD
# ─────────────────────────────────────────────────────────────────────────────
def forward_minmax_od(
    od: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize OD to [0, 1] using GLOBAL extrema (per RESEARCH.md Q4).

    Single-campaign data (docs/dataset_stats.md: 1 campaign, 778 rows)
    makes global min-max the natural choice. Returns (scaled, od_min, od_max).
    """
    od_min = torch.min(od)
    od_max = torch.max(od)
    scaled = (od - od_min) / (od_max - od_min)  # -> [0, 1]
    return scaled, od_min, od_max


def inverse_minmax_od(
    scaled: torch.Tensor, od_min: torch.Tensor, od_max: torch.Tensor,
) -> torch.Tensor:
    """Exact inverse of forward_minmax_od."""
    return scaled * (od_max - od_min) + od_min


__all__ = [
    "forward_lambert", "inverse_lambert",
    "forward_logreturns", "inverse_logreturns",
    "forward_minmax_od", "inverse_minmax_od",
]
