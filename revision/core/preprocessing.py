"""Preprocessing pipelines — three ablation variants for R1-M3 (Phase 09.1).

Phase 9 implements only the Lambert W pair (it IS EVAL-06); the other four
functions are NotImplementedError stubs reserved for Phase 09.1.

Contract: each ``forward_X`` / ``inverse_X`` pair must satisfy
    max_abs(inverse_X(forward_X(x), *args), x) <= 1e-8
on a real OD trajectory (Phase 09.1 ABL-01).
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
from revision.core.data import (
    lambert_w_transform as forward_lambert,
    inverse_lambert_w_transform as inverse_lambert,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — log-returns only
# ─────────────────────────────────────────────────────────────────────────────
def forward_logreturns(od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log-returns r_t = ln(OD_{t+1}/OD_t) and standardize; return (r_norm, (mu, sigma)). Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


def inverse_logreturns(
    r: torch.Tensor, od_start: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Un-standardize log-returns and integrate cumulatively from od_start to recover OD. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — min-max normalized OD
# ─────────────────────────────────────────────────────────────────────────────
def forward_minmax_od(
    od: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize OD to [0, 1]; return (scaled, od_min, od_max). Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


def inverse_minmax_od(
    scaled: torch.Tensor, od_min: torch.Tensor, od_max: torch.Tensor
) -> torch.Tensor:
    """Un-normalize scaled OD back to original units. Phase 09.1."""
    raise NotImplementedError("Phase 09.1")


__all__ = [
    "forward_lambert", "inverse_lambert",
    "forward_logreturns", "inverse_logreturns",
    "forward_minmax_od", "inverse_minmax_od",
]
