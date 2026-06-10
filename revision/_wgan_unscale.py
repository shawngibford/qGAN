"""Shared inverse-pipeline correction for the x0.1 WGAN sample-space convention.

This module is the SINGLE SOURCE OF TRUTH for the inference-time correction
that undoes the historical ``* 0.1`` training/sample-export scaling applied
to every WGAN-trained generator (quantum IQP:SEL + V1/V2/V3 + classical
wgan_mlp/cnn/lstm).

Origin (smoking-gun)
--------------------
``archive/qgan_pennylane_SEL.py:661-663`` introduced ``generated_sample = generated_sample * 0.1``
with the explicit comment "Real data is roughly in [-0.08, 0.09], so scale by ~0.1".
That ``* 0.1`` was a Pipeline-A artifact: when the critic compared a [-1, +1]
quantum output against unstandardized log-returns in [-0.08, +0.09], the
attenuation was the magnitude-matching trick that made the WGAN loss numerically
meaningful.

Pipeline-B preservation (where the bug bites)
---------------------------------------------
When the revision pipeline switched to Pipeline B (log-returns → standardize →
linear rescale of real to [-1, +1]), the ``* 0.1`` should have been removed.
Instead it was preserved verbatim at five sites:
  * ``revision/run_matched2000.py:281``  (training-time / sample-export)
  * ``revision/core/training.py:347,381,416``  (per-WGAN-variant training loop)
  * ``revision/run_baselines.py:205``  (WGAN sample-export)
So every WGAN-trained ``samples.npy`` on disk under
``revision/results/matched2000/runs/<model>/<seed>/`` is stored at one-tenth
of its canonical [-1, +1] magnitude. The downstream inverse formula
``r_norm = ((s + 1) / 2) * (r_max - r_min) + r_min`` assumes ``s ∈ [-1, +1]``,
so consuming the on-disk samples directly produces ~10× attenuated
log-deltas — and ~10× under-reported standard deviations, EMD, and DTW.

Inference-only correction
-------------------------
The training-side ``* 0.1`` sites are byte-frozen (their removal would
invalidate every checkpoint and every saved sample bundle). The correction
is applied INSIDE every paper-cited consumer at the ``samples.npy`` load
boundary, before any un-rescale math:

    samples_pm1 = np.load(... / "samples.npy").astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model_kind)

Pitfall 3 asymmetry (VAE/AR excluded)
-------------------------------------
Per ``revision/run_baselines.py:28`` "Pitfall 3 (RESEARCH)", VAE and AR were
trained WITHOUT the ``* 0.1`` post-scaling — their ``sample()`` already emits
in canonical [-1, +1] window space. Applying ``* 10`` to their ``samples.npy``
would break their published metrics. The ``_WGAN_KINDS`` frozenset below
gates the correction.

The headline-row labels (``frozen_checkpoint_headline``,
``iqp_sel_55_headline``) and the bare ``quantum`` label used by the ansatz
driver are added as members because the underlying generator is WGAN-trained
quantum even though the row label differs from the seven primary strings.

Residual mean drift (known limitation; not fixed here)
------------------------------------------------------
Even after ``* 10``, the WGAN generator's mean in [-1, +1] space is ≈ 0
while real Pipeline-B data has mean ≈ -0.03. This produces a small residual
OD-trajectory drift. The drift is a training-side artifact and is documented
in the supplement methodology disclosure as out-of-scope for this
inference-time correction.
"""

from __future__ import annotations

import numpy as np

# Membership predicate for the x10 correction. Includes:
#   * the seven primary model_kind strings emitted by run_matched2000.py
#     (iqp_sel_55_repro = quantum IQP:SEL repro; V1/V2/V3 = quantum
#     architecture-ablation variants; wgan_mlp/cnn/lstm = classical
#     adversarial baselines)
#   * the three headline-row / ansatz-driver labels that wrap a WGAN-trained
#     quantum generator under a different label.
# VAE and AR are intentionally EXCLUDED per Pitfall 3
# (revision/run_baselines.py:28) — they were trained without the * 0.1.
_WGAN_KINDS: frozenset[str] = frozenset(
    {
        "iqp_sel_55_repro",
        "V1",
        "V2",
        "V3",
        "wgan_mlp",
        "wgan_cnn",
        "wgan_lstm",
        "frozen_checkpoint_headline",
        "iqp_sel_55_headline",
        "quantum",
    }
)


def _unscale_wgan_samples(samples_pm1: np.ndarray, model_kind: str) -> np.ndarray:
    """Undo the x0.1 WGAN training-time scaling (Pitfall 3 inverse).

    ``samples.npy`` for any ``model_kind`` in ``_WGAN_KINDS`` was written with
    ``* 0.1`` applied (see ``run_matched2000.py:281`` / ``core/training.py``
    / ``run_baselines.py:205``). The downstream un-rescale formula
    ``((s + 1) / 2) * (r_max - r_min) + r_min`` assumes ``s ∈ [-1, +1]``, so
    this helper multiplies the on-disk samples by 10 before they are fed to
    the un-rescale. VAE and AR samples (Pitfall 3 — not in ``_WGAN_KINDS``)
    are returned unchanged.

    Parameters
    ----------
    samples_pm1 :
        On-disk samples loaded from ``samples.npy`` (any shape; broadcast-safe).
    model_kind :
        Row label for the source artifact. Membership in ``_WGAN_KINDS`` is
        the only branch; case-sensitive.

    Returns
    -------
    np.ndarray
        ``samples_pm1 * 10.0`` for WGAN-trained kinds; ``samples_pm1`` unchanged
        for VAE / AR (Pitfall 3 design).
    """
    if model_kind in _WGAN_KINDS:
        return samples_pm1 * 10.0
    return samples_pm1
