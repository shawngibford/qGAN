"""Phase 12 SENS-01 / SENS-02 per-cell inference driver.

One CLI invocation == one ``(pipeline, seed, condition)`` cell. The trained
analytic quantum generator stores all learned state in a single 75-element
``params_pqc`` tensor (``checkpoint.pt`` in every
``revision/results/transform_ablation/runs/<pipeline>/<seed>/``). Phase 12 is
**inference-only** (D-12-01): we reload that frozen tensor into a freshly
constructed QNode on a noisy / finite-shot device, regenerate samples honoring
the load-bearing ``*0.1`` + ``np.random.default_rng(seed)`` contracts,
reconstruct the OD scale with the verbatim Pipeline-A/B recipe, and recompute
the *unchanged* fidelity suite (``revision.core.eval.full_metric_suite``,
D-12-03). **No retraining anywhere.**

Locked decisions honored here:

  * **D-12-01** — inference-only; reload ``params_pqc``, never retrain. The
    generator's training device is ``default.qubit shots=None
    diff_method=backprop``; finite shots / noise channels are incompatible with
    statevector backprop, so the alternate QNode is built here with
    ``diff_method=None`` under ``torch.no_grad()``.
  * **D-12-02** — degradation grids use seeds {42,43,44}; the 5-seed headline
    roll-up is SENS-03's job, not this driver's.
  * **D-12-03** — fidelity metrics are recomputed via the existing
    ``full_metric_suite`` UNCHANGED; this driver only tags rows with
    ``scale`` / ``shots`` / ``noise_*`` dimensions.
  * **D-10-13** — ``revision/core/`` is byte-untouched. The noisy circuit body
    is a *deliberate, documented duplication* of
    ``quantum.py:generator_circuit`` that lives in THIS file (not ``core/``)
    purely for the noise study (see ``noisy_generator_circuit``).

Deliberate, documented deviation (RESEARCH Open Q1 / Pitfall 5)
---------------------------------------------------------------
This driver asserts ``qml.__version__ == "0.44.0"`` at startup and FAILS LOUD
otherwise. The analog sweep wrappers hard-prefer ``./qgan_env/bin/python``
which carries PennyLane **0.43.0** — a version where the ``qml.set_shots``
transform API and the ``shots=`` device-kwarg deprecation differ. The Phase 12
sweep MUST NOT run via ``./qgan_env``; ``qgan_env`` is intentionally NOT
upgraded (that would invalidate the frozen 09.1/10 reproduction baseline).

Pitfalls guarded
----------------
  * Pitfall 1 — ``shots=`` device kwarg is deprecated in 0.44; we use the
    ``qml.set_shots(qnode, shots=N)`` transform instead.
  * Pitfall 2 — ``backprop`` is incompatible with finite shots / mixed
    devices; every QNode here uses ``diff_method=None``.
  * Pitfall 3 — the ``* 0.1`` output scaling is load-bearing (part of the
    trained sample contract); the generation body is copied verbatim from
    ``run_ablation.generate_samples``.
  * Pitfall 4 — the Pipeline-B ``np.random.default_rng(seed * 7919 + 1)``
    od_start draw is load-bearing; ``reconstruct_od`` is copied verbatim from
    ``run_utility.py``.
  * Pitfall 5 — version skew (handled by the 0.44.0 startup assertion above).
  * Pitfall 6 — cwd-dependent paths; every artifact path is anchored at the
    resolved ``REPO`` root.

Usage
-----
    python revision/run_sensitivity.py --pipeline {A|B} --seed N \\
        --condition <one of the 11 tokens> \\
        [--out-root revision/results/sensitivity] [--csv-path ./data.csv]

The 11 condition tokens::

    analytic
    shots_8192  shots_1024
    depol_0.0   depol_0.001   depol_0.01   depol_0.05
    ampdamp_0.0 ampdamp_0.001 ampdamp_0.01 ampdamp_0.05

Each run writes ``out_root/runs/<condition>/<pipeline>/<seed>/`` with::

    config.yaml   — frozen cell config (condition + shots/noise dims)
    samples.npy   — (N_synth, WINDOW_LENGTH) regenerated samples in [-1, 1]
    metrics.json  — dual-scale long-form rows + full_metric_suite output
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import pennylane as qml

# ─────────────────────────────────────────────────────────────────────────────
# PennyLane version gate (RESEARCH Open Q1 recommendation (a) / Pitfall 5).
# Fail LOUD if not exactly 0.44.0 — the set_shots transform / shots= kwarg
# deprecation differ between 0.43 and 0.44. Do NOT run via ./qgan_env (0.43.0).
# ─────────────────────────────────────────────────────────────────────────────
assert qml.__version__ == "0.44.0", (
    f"Phase 12 requires PennyLane 0.44.0 (set_shots transform / default.mixed "
    f"API); got {qml.__version__}. Do NOT run via ./qgan_env (0.43.0) — that "
    f"venv carries PennyLane 0.43.0 and silently changes shot/noise semantics."
)


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolver (RESEARCH Pitfall 6) — copied verbatim from
# revision/run_utility.py:42-57. Anchor EVERY artifact path at REPO.
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    # Fallback: walk up from cwd (covers exotic invocations).
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (revision/core/preprocessing.py)")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.core import (  # noqa: E402
    BATCH_SIZE,
    NOISE_HIGH,
    NOISE_LOW,
    NUM_LAYERS,
    NUM_QUBITS,
    WINDOW_LENGTH,
)
from revision.core.models.quantum import QuantumGenerator  # noqa: E402
from revision.core.preprocessing import inverse_logreturns  # noqa: E402
from revision.core.data import load_and_preprocess, rolling_window  # noqa: E402
from revision.core.eval import full_metric_suite  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Trained-params load (RESEARCH Code Example 1). The trained state is a single
# 75-element tensor. Phase 12 does NOT use ``g.qnode`` (Pitfall 2 — its bound
# device is default.qubit/backprop, incompatible with shots/mixed). NO training
# loop, NO optimizer, NO .backward() anywhere in this module (D-12-01).
# ─────────────────────────────────────────────────────────────────────────────
def load_trained_generator(pipeline: str, seed: int) -> QuantumGenerator:
    """Reload the frozen ``params_pqc`` 75-tensor into a fresh generator.

    NO retraining (D-12-01). The checkpoint is opened read-only; the only
    mutation is ``g.params_pqc.data = ck["params_pqc"]``.
    """
    g = QuantumGenerator(
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        window_length=WINDOW_LENGTH,
    )
    ck_path = (
        REPO
        / "revision"
        / "results"
        / "transform_ablation"
        / "runs"
        / pipeline
        / str(seed)
        / "checkpoint.pt"
    )
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    g.params_pqc.data = ck["params_pqc"]  # 75-element trained tensor
    g.eval()
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Generation contract — copied VERBATIM from
# revision/run_ablation.py:195-208 (Pitfall 3 — the ``* 0.1`` scaling and
# ``np.random.default_rng(seed)`` are part of the trained sample contract).
# The ONLY change vs the analog: the per-batch call site uses an injected
# ``qnode`` argument instead of ``generator(noise)``, and the returned tuple
# of 10 expvals is stacked + transposed exactly as ``QuantumGenerator.forward``
# does (quantum.py:194-199).
# ─────────────────────────────────────────────────────────────────────────────
def generate_samples_on_qnode(
    g: QuantumGenerator,
    qnode,
    n: int,
    seed: int,
) -> np.ndarray:
    """Generate ``n`` synthetic windows in [-1, 1] via the injected ``qnode``.

    Verbatim port of ``run_ablation.generate_samples`` — the ``* 0.1`` cast
    and ``np.random.default_rng(seed)`` are LOAD-BEARING (Pitfall 3). The only
    deviation is the call site: ``qnode(noise, g.params_pqc)`` instead of
    ``generator(noise)``, plus the ``(window_length, batch) -> .T`` transpose
    replicated from ``QuantumGenerator.forward`` (quantum.py:194-199).
    """
    rng = np.random.default_rng(seed)
    out_parts: list[np.ndarray] = []
    remaining = n
    with torch.no_grad():
        while remaining > 0:
            bs = min(BATCH_SIZE, remaining)
            noise = torch.tensor(
                rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, bs)),
                dtype=torch.float32,
            )
            res = qnode(noise, g.params_pqc)  # tuple of 10 expvals
            stacked = torch.stack(list(res))
            # Batched: (window_length, batch) -> transpose to (batch, window_length).
            # Unbatched: (window_length,) -> already correct.
            if stacked.dim() == 2:
                stacked = stacked.T
            out = stacked.to(torch.float64) * 0.1  # *0.1 LOAD-BEARING (Pitfall 3)
            out_parts.append(out.cpu().numpy())
            remaining -= bs
    samples = np.concatenate(out_parts, axis=0)[:n]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# OD-scale reconstruction — copied VERBATIM from
# revision/run_utility.py:144-179 (Pitfall 4 — the Pipeline-B
# ``np.random.default_rng(seed * 7919 + 1)`` od_start draw is load-bearing; do
# NOT refactor. Note the ``od[:, :10]`` truncation when ``inverse_logreturns``
# returns length-11). ``inverse_kwargs.npz`` is the frozen per-pipeline aux
# state written by the 09.1 ablation runs (read-only).
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_od(
    pipeline: str,
    seed: int,
    samples_pm1: np.ndarray,
    inverse_kwargs_path: Path,
) -> dict:
    """Invert [-1, 1] samples back to OD scale (Pipeline A or B).

    Verbatim recipe from ``run_utility.reconstruct_od``; the only adaptation is
    that ``samples_pm1`` is passed in (regenerated under noise) rather than
    re-loaded from disk, and ``inverse_kwargs`` is read from the frozen
    09.1 ``inverse_kwargs.npz``.
    """
    samples_pm1 = samples_pm1.astype(np.float64)
    inv = np.load(inverse_kwargs_path, allow_pickle=True)

    if pipeline == "A":
        od_min = float(inv["od_min"])
        od_max = float(inv["od_max"])
        od01 = (samples_pm1 + 1.0) / 2.0
        od = od01 * (od_max - od_min) + od_min
        return {
            "od_samples": od,
            "transformed": None,
            "n_synth": od.shape[0],
            "pipeline": pipeline,
            "seed": seed,
        }

    if pipeline == "B":
        r_min = float(inv["r_min"])
        r_max = float(inv["r_max"])
        mu = float(inv["mu"])
        sigma = float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])
        r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
        rng = np.random.default_rng(seed * 7919 + 1)  # load-bearing — do NOT refactor
        od_start_per_window = rng.choice(
            od_starts_pool, size=r_norm.shape[0], replace=True
        )
        r_norm_t = torch.tensor(r_norm)
        od_start_t = torch.tensor(od_start_per_window)
        od_full = inverse_logreturns(
            r_norm_t, od_start_t, torch.tensor(mu), torch.tensor(sigma)
        )
        od = od_full.cpu().numpy()
        if od.shape[1] == 11:
            od = od[:, :10]
        return {
            "od_samples": od,
            "transformed": r_norm,
            "n_synth": od.shape[0],
            "pipeline": pipeline,
            "seed": seed,
        }

    raise ValueError(
        f"unknown pipeline {pipeline} (Pipeline C dropped, D-10-05)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real windowed OD (copied from run_utility.py:186-190). The same
# load_and_preprocess + rolling_window(window=10, stride=2) the 09.1 quantum
# runs used — gives the (385, 10) real OD window matrix the fidelity suite
# compares the regenerated fake samples against.
# ─────────────────────────────────────────────────────────────────────────────
def real_windowed_od(csv_path: Path) -> np.ndarray:
    d_real = load_and_preprocess(str(csv_path))
    return rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()


def _resolve_csv(csv_arg: Path) -> Path:
    """Resolve --csv-path to an ABSOLUTE repo-root-anchored path (Pitfall 6)."""
    csv_arg = Path(csv_arg)
    if csv_arg.is_absolute():
        return csv_arg
    return (REPO / csv_arg).resolve()
