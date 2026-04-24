"""Evaluation metrics: EMD, ACF, moments, DTW, JSD, PSD — all pure functions.

Extracted verbatim from ``qgan_pennylane.ipynb`` (cells 59, 60, 64, 65).
This module preserves the v1.0/v1.1 locked behavioral decisions:

* **EMD uses raw samples** via ``scipy.stats.wasserstein_distance`` — NOT
  histogram-based (v1.0 decision; see QGAN_Review_Response_Plan.md.pdf).
* **ACF uses FFT** (notebook cell 65: ``compute_acf(..., fft=True)``).
* **Kurtosis is Fisher (excess)** and **std uses ddof=0** — matches cell 59 /
  cell 65 which call ``np.std(data)`` and ``scipy.stats.kurtosis(data)`` with
  defaults.

Note: the module name shadows the builtin ``eval`` in the package namespace.
That is acceptable because we always access it as ``revision.core.eval`` (a
module attribute), never as a free function.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Cell 59 / cell 65 — EMD on raw samples (v1.0 decision)
# ─────────────────────────────────────────────────────────────────────────────
def compute_emd(real: np.ndarray, fake: np.ndarray) -> float:
    """Earth Mover's Distance via :func:`scipy.stats.wasserstein_distance`.

    CRITICAL: called on **raw samples**, NOT on histograms (v1.0 locked
    decision). Notebook cells 59 and 65 both do
    ``wasserstein_distance(log_delta_np, fake_log_delta_np)``.
    """
    from scipy.stats import wasserstein_distance

    real = np.asarray(real).ravel()
    fake = np.asarray(fake).ravel()
    return float(wasserstein_distance(real, fake))


# ─────────────────────────────────────────────────────────────────────────────
# Cell 59 / cell 65 — moments (ddof=0 std, Fisher kurtosis)
# ─────────────────────────────────────────────────────────────────────────────
def compute_moments(samples: np.ndarray) -> Dict[str, float]:
    """Return ``{mean, std, skewness, kurtosis}``.

    Matches the notebook:
    * ``std`` uses ``np.std(x)`` default (``ddof=0``) — cells 59, 65
    * ``kurtosis`` is Fisher (excess) — ``scipy.stats.kurtosis`` default
    * ``skewness`` uses ``scipy.stats.skew`` default
    """
    from scipy.stats import kurtosis, skew

    s = np.asarray(samples).ravel()
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "skewness": float(skew(s)),
        "kurtosis": float(kurtosis(s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 65 — ACF with FFT
# ─────────────────────────────────────────────────────────────────────────────
def compute_acf(samples: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Autocorrelation function via :func:`statsmodels.tsa.stattools.acf`.

    Notebook cell 65 uses ``fft=True``; preserved here.
    """
    from statsmodels.tsa.stattools import acf as _acf

    s = np.asarray(samples).ravel()
    return np.asarray(_acf(s, nlags=nlags, fft=True))


# ─────────────────────────────────────────────────────────────────────────────
# Cell 26 / cell 64 / cell 65 — DTW via fastdtw
# ─────────────────────────────────────────────────────────────────────────────
def compute_dtw(real: np.ndarray, fake: np.ndarray) -> float:
    """Dynamic time warping distance via ``fastdtw`` with Euclidean metric.

    Notebook cell 64/65: ``fastdtw(real.reshape(-1,1), fake.reshape(-1,1),
    dist=euclidean)``. 2D reshape matches cell 65 call site.
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    real_a = np.asarray(real).ravel().reshape(-1, 1)
    fake_a = np.asarray(fake).ravel().reshape(-1, 1)
    distance, _ = fastdtw(real_a, fake_a, dist=euclidean)
    return float(distance)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 59 / cell 65 — Jensen-Shannon divergence on shared-bin histograms
# ─────────────────────────────────────────────────────────────────────────────
def compute_jsd(real: np.ndarray, fake: np.ndarray, bins: int = 100) -> float:
    """Jensen-Shannon divergence on histograms with shared bin edges.

    Notebook cells 59 + 65. Histograms normalized to probability mass (sum=1).
    """
    from scipy.spatial.distance import jensenshannon

    real = np.asarray(real).ravel()
    fake = np.asarray(fake).ravel()
    lo = min(real.min(), fake.min())
    hi = max(real.max(), fake.max())
    edges = np.linspace(lo, hi, bins + 1)
    rh, _ = np.histogram(real, bins=edges, density=True)
    fh, _ = np.histogram(fake, bins=edges, density=True)
    rh = rh / rh.sum()
    fh = fh / fh.sum()
    return float(jensenshannon(rh, fh))


# ─────────────────────────────────────────────────────────────────────────────
# PSD — conservative default (not implemented in notebook's metric section)
# ─────────────────────────────────────────────────────────────────────────────
def compute_psd(real: np.ndarray, fake: np.ndarray) -> Dict[str, np.ndarray]:
    """Power spectral density comparison via :func:`scipy.signal.welch`.

    Implementation
    --------------
    The notebook does not include a PSD metric in its final evaluation cells
    (cell 65). This conservative default uses ``scipy.signal.welch`` with its
    default parameters. If a downstream consumer (e.g., the PSD spectral-loss
    term in v1.1 Phase 6) requires a specific window / nperseg, adjust here.
    """
    from scipy.signal import welch

    freqs_real, psd_real = welch(np.asarray(real).ravel())
    freqs_fake, psd_fake = welch(np.asarray(fake).ravel())
    return {
        "freqs_real": freqs_real,
        "psd_real": psd_real,
        "freqs_fake": freqs_fake,
        "psd_fake": psd_fake,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def full_metric_suite(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    """Compute EMD + moments + JSD in one call (flat dict output).

    Returns keys:
    ``emd, mean_real, mean_fake, std_real, std_fake,
    skew_real, skew_fake, kurt_real, kurt_fake, jsd``.
    """
    m_r = compute_moments(real)
    m_f = compute_moments(fake)
    return {
        "emd": compute_emd(real, fake),
        "mean_real": m_r["mean"],
        "mean_fake": m_f["mean"],
        "std_real": m_r["std"],
        "std_fake": m_f["std"],
        "skew_real": m_r["skewness"],
        "skew_fake": m_f["skewness"],
        "kurt_real": m_r["kurtosis"],
        "kurt_fake": m_f["kurtosis"],
        "jsd": compute_jsd(real, fake),
    }
