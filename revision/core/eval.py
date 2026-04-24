"""Evaluation metrics: EMD, ACF, moments, DTW, JSD, PSD — all pure functions.

Note: this module's name shadows the Python builtin ``eval`` inside the package
namespace. That is acceptable because we always access it as
``revision.core.eval`` (a module attribute), never as a free function.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def compute_emd(real: np.ndarray, fake: np.ndarray) -> float:
    """Earth Mover's Distance via scipy.stats.wasserstein_distance on RAW samples
    (not histograms — v1.0 decision). Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_moments(samples: np.ndarray) -> Dict[str, float]:
    """Return dict with keys: mean, std, skewness, kurtosis. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_acf(samples: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Autocorrelation function via statsmodels. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_dtw(real: np.ndarray, fake: np.ndarray) -> float:
    """Dynamic time warping distance via fastdtw. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_jsd(real: np.ndarray, fake: np.ndarray, bins: int = 100) -> float:
    """Jensen-Shannon divergence on histograms. Filled in by plan 08-02."""
    raise NotImplementedError("Filled in by plan 08-02")


def compute_psd(real: np.ndarray, fake: np.ndarray) -> Dict[str, np.ndarray]:
    """Power spectral density comparison.

    Returns dict with keys: freqs, psd_real, psd_fake. Filled in by plan 08-02.
    """
    raise NotImplementedError("Filled in by plan 08-02")


def full_metric_suite(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    """Compute EMD + moments + JSD in one call.

    Returns flat dict:
    {emd, mean_real, mean_fake, std_real, std_fake,
     skew_real, skew_fake, kurt_real, kurt_fake, jsd}.

    Filled in by plan 08-02.
    """
    raise NotImplementedError("Filled in by plan 08-02")
