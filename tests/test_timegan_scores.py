"""RED/GREEN test for Phase 11 Plan 02 Task 1 — faithful TimeGAN post-hoc nets.

Asserts the pinned jsyoon0823/TimeGAN algorithm contract (Pitfall 1 — the
canonical ``hidden_dim = int(dim/2)`` is degenerate at the univariate dim=1, so
H is locked to WINDOW_LENGTH=10 and used identically for both nets):

  - ``PredictiveGRU(10)`` / ``DiscriminativeGRU(10)`` construct with >0 params
    at ``input_size=1`` (regression guard for the ``int(dim/2)=0`` trap).
  - ``PredictiveGRU`` forward on (B, 9, 1) produces a finite tensor.
  - ``predictive_score`` returns a finite non-negative float (next-step MAE,
    train-on-synthetic / test-on-real).
  - ``discriminative_score`` returns a float in [0, 0.5] (== ``|0.5 - acc|``).
  - Both are deterministic given the recorded seed.

Run: <main-repo>/qgan_env/bin/python -m pytest tests/test_timegan_scores.py
or as a plain script:   ... tests/test_timegan_scores.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

# scripts/ holds the runners; add it to sys.path so we can import them
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_timegan_scores as m  # noqa: E402


def test_gru_nets_not_degenerate():
    pg = m.PredictiveGRU(10)
    dg = m.DiscriminativeGRU(10)
    n_pg = sum(p.numel() for p in pg.parameters())
    n_dg = sum(p.numel() for p in dg.parameters())
    assert n_pg > 0, n_pg
    assert n_dg > 0, n_dg
    out = pg(torch.zeros(4, 9, 1))
    assert torch.isfinite(out).all(), out


def test_predictive_score_finite_nonnegative():
    rng = np.random.default_rng(0)
    synth = rng.random((200, 10)).astype("float64")
    real = rng.random((120, 10)).astype("float64")
    s = m.predictive_score(synth, real, 42, 10, iters=50, bs=32)
    assert np.isfinite(s), s
    assert s >= 0.0, s


def test_discriminative_score_in_range():
    rng = np.random.default_rng(1)
    real = rng.random((200, 10)).astype("float64")
    synth = rng.random((200, 10)).astype("float64")
    d = m.discriminative_score(real, synth, 42, 10, iters=50, bs=32)
    assert 0.0 <= d <= 0.5 + 1e-6, d


def test_scores_deterministic():
    rng = np.random.default_rng(2)
    synth = rng.random((160, 10)).astype("float64")
    real = rng.random((96, 10)).astype("float64")
    a = m.predictive_score(synth, real, 42, 10, iters=40, bs=32)
    b = m.predictive_score(synth, real, 42, 10, iters=40, bs=32)
    assert a == b, (a, b)


def test_discriminative_score_deterministic():
    """WR-06: lock discriminative_score determinism in the COLLECTED suite.

    Mirrors ``test_scores_deterministic`` for the riskier discriminative path
    (the WR-05 single-Generator fix in 11-07). Two identical seeded calls must
    return the EXACT same value (bit-deterministic — not approximate); this
    would fail against the old mixed-global-RNG implementation.
    """
    rng = np.random.default_rng(2)
    real = rng.random((160, 10)).astype("float64")
    synth = rng.random((160, 10)).astype("float64")
    a = m.discriminative_score(real, synth, 42, 10, iters=40, bs=32)
    b = m.discriminative_score(real, synth, 42, 10, iters=40, bs=32)
    assert a == b, (a, b)
    assert np.isfinite(a), a
    assert 0.0 <= a <= 0.5 + 1e-6, a


if __name__ == "__main__":
    test_gru_nets_not_degenerate()
    test_predictive_score_finite_nonnegative()
    test_discriminative_score_in_range()
    test_scores_deterministic()
    test_discriminative_score_deterministic()
    print("all timegan-score tests passed")
