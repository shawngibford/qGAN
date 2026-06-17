"""INTRO-01/02/03 callback-wiring regression (short synthetic run).

Exercises the ``run_introspect`` snapshot closure directly (no 1000-epoch
training) on a tiny synthetic loop for BOTH a real ``QuantumGenerator`` and a
classical WGAN generator, asserting the Task-1 behavior block:

    - snapshot_cb returns immediately unless ``epoch`` is in the SNAP set;
      it appends exactly one record per matching epoch and relabels the
      terminal SNAP index to ``num_epochs`` (the 999 -> 1000 contract,
      rescaled here to 29 -> 30).
    - Each record has keys ``epoch`` / ``samples`` (non-empty list) /
      ``emd`` / ``std``.
    - A ``QuantumGenerator`` record additionally carries ``param_norm``
      (float>0), ``param_angles`` (len == count_params()),
      ``vn_entropy`` (in [0, ln4+eps]) and ``purity`` (in [0.25-eps, 1+eps]).
    - A classical generator (no ``introspect()``) is hasattr-guarded: the
      record has ``samples`` but NO vn_entropy / purity / param fields, and
      the closure does not crash (callback try/except in training.py would
      otherwise swallow the failure -> T-13-09).
    - The generated-snapshot std is the same order of magnitude as the
      ``metrics["std"]`` passed at that epoch (noise-contract sanity,
      Pitfall 2 / T-13-08).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from revision.core.models.classical import WGANMLPGenerator
from revision.core.models.quantum import QuantumGenerator
from revision.run_introspect import make_snapshot_cb

_EPS = 1e-6
# Short-run rescale of the production SNAP {0,250,500,750,999} @ 1000 epochs.
_SHORT_EPOCHS = 30
_SHORT_SNAP = {0, 10, 20, 29}


def _drive(generator, snap, num_epochs, seed=42):
    """Tiny synthetic eval loop mimicking training.py's callback contract.

    Fires the closure on epochs ``e % 10 == 0 or e + 1 == num_epochs``
    (the real eval-branch predicate with eval_every=10), passing the same
    scalar metrics dict shape training.py builds.
    """
    cb, records = make_snapshot_cb(
        generator, snap=snap, num_epochs=num_epochs, seed=seed
    )
    rng = np.random.default_rng(0)
    for e in range(num_epochs):
        if e % 10 == 0 or e + 1 == num_epochs:
            # Synthetic "real" std proxy for the noise-contract sanity check.
            std = float(0.05 + 0.001 * e)
            cb(
                e,
                {
                    "epoch": e,
                    "emd": float(rng.uniform(0.0, 0.2)),
                    "critic_loss": -1.0,
                    "generator_loss": 1.0,
                    "mean": 0.0,
                    "std": std,
                    "kurtosis": 0.0,
                },
            )
    return records


def test_closure_fires_only_on_snap_and_relabels_terminal():
    g = QuantumGenerator()
    records = _drive(g, _SHORT_SNAP, _SHORT_EPOCHS)
    # SNAP {0,10,20,29} -> exactly 4 snapshots.
    assert len(records) == 4, f"expected 4 snapshots, got {len(records)}"
    epochs = [r["epoch"] for r in records]
    # 29 is the terminal SNAP index -> relabelled to num_epochs (30).
    assert epochs == [0, 10, 20, 30], f"bad epoch labels: {epochs}"


def test_quantum_record_has_full_introspection_fields():
    torch.manual_seed(0)
    g = QuantumGenerator()
    records = _drive(g, _SHORT_SNAP, _SHORT_EPOCHS)
    for r in records:
        assert isinstance(r["samples"], list) and len(r["samples"]) > 0
        assert "emd" in r and "std" in r
        assert isinstance(r["param_norm"], float) and r["param_norm"] > 0.0
        assert isinstance(r["param_angles"], list)
        assert len(r["param_angles"]) == g.count_params()
        assert -_EPS <= r["vn_entropy"] <= math.log(4) + _EPS
        assert 0.25 - _EPS <= r["purity"] <= 1.0 + _EPS


def test_classical_record_is_hasattr_guarded_no_crash():
    torch.manual_seed(0)
    g = WGANMLPGenerator()
    records = _drive(g, _SHORT_SNAP, _SHORT_EPOCHS)
    assert len(records) == 4
    for r in records:
        assert isinstance(r["samples"], list) and len(r["samples"]) > 0
        assert "emd" in r and "std" in r
        # Classical generator has no introspect() -> no PQC/entropy fields.
        assert "param_norm" not in r
        assert "param_angles" not in r
        assert "vn_entropy" not in r
        assert "purity" not in r


def test_snapshot_std_same_order_as_metrics_std():
    torch.manual_seed(0)
    g = QuantumGenerator()
    records = _drive(g, _SHORT_SNAP, _SHORT_EPOCHS)
    for r in records:
        gen = np.asarray(r["samples"], dtype=np.float64)
        gen_std = float(gen.std())
        # Both the metrics std and the regenerated std are ~1e-2 (the *0.1
        # scaled generator output). Same order of magnitude (T-13-08).
        assert gen_std > 0.0
        ratio = gen_std / r["std"]
        assert 1e-2 < ratio < 1e2, (
            f"snapshot std {gen_std} not same order as metrics std "
            f"{r['std']} (ratio {ratio})"
        )
