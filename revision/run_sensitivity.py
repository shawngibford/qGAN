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


# ─────────────────────────────────────────────────────────────────────────────
# SENS-01 — finite-shot QNode via the 0.44 ``qml.set_shots`` transform
# (RESEARCH Code Example 2). NO ``shots=`` device kwarg (Pitfall 1 — deprecated
# in 0.44, emits PennyLaneDeprecationWarning). ``diff_method=None`` — inference
# only, NOT backprop (Pitfall 2).
# ─────────────────────────────────────────────────────────────────────────────
def make_shot_qnode(g: QuantumGenerator, shots: int | None):
    """Build an analytic-or-finite-shot QNode on ``default.qubit``.

    ``shots is None`` -> exact analytic expectations (the {analytic} reference
    device). ``shots`` an int -> the 0.44 ``qml.set_shots`` transform wraps the
    analytic QNode (the deprecation-free finite-shot path).
    """
    dev = qml.device("default.qubit", wires=NUM_QUBITS)  # analytic device; shot count via transform only (Pitfall 1)
    qn = qml.QNode(
        g.generator_circuit, dev, interface="torch", diff_method=None
    )  # NOT backprop (Pitfall 2)
    if shots is not None:
        qn = qml.set_shots(qn, shots=shots)  # 0.44 transform API
    return qn


# ─────────────────────────────────────────────────────────────────────────────
# SENS-02 — noise-channel QNode on ``default.mixed`` (RESEARCH Code Example 3).
#
# DELIBERATE, DOCUMENTED DUPLICATION (D-10-13 NOT violated): the gate body
# below is a verbatim copy of ``QuantumGenerator.generator_circuit``
# (revision/core/models/quantum.py:122-171) reproduced in THIS driver — not in
# ``core/`` — purely for the noise study. ``g.generator_circuit`` ends with
# ``qml.expval`` returns, so channels cannot be appended after calling it
# inside another QNode; the body must be re-emitted with channel insertion.
#
# Channel-insertion strategy: **PER-LAYER** — one channel on every wire
# immediately AFTER each entangling (range-based CNOT) block. This is the
# conventional NISQ deployment-noise model and the documented default
# (RESEARCH Assumption A1 / Open Q2 RESOLVED: per-layer). ``level`` is the
# depolarizing ``p`` or amplitude-damping ``gamma``; ``level == 0.0`` reduces
# the mixed-device circuit to the analytic result (sanity-checkable).
# ─────────────────────────────────────────────────────────────────────────────
def make_noisy_qnode(g: QuantumGenerator, channel: str, level: float):
    """Build a ``default.mixed`` QNode with per-layer noise channels.

    ``channel`` in {'depolarizing', 'amplitude_damping'}; ``level`` is ``p``
    (depolarizing) or ``gamma`` (amplitude damping). Per-layer insertion: the
    channel is applied to every wire after each entangling block (Assumption
    A1, RESOLVED default). ``diff_method=None`` (Pitfall 2).
    """
    if channel not in ("depolarizing", "amplitude_damping"):
        raise ValueError(
            f"unknown channel {channel!r} "
            f"(expected 'depolarizing' or 'amplitude_damping')"
        )
    dev = qml.device("default.mixed", wires=NUM_QUBITS)

    def _apply_channel() -> None:
        for q in range(NUM_QUBITS):
            if channel == "depolarizing":
                qml.DepolarizingChannel(level, wires=q)
            else:
                qml.AmplitudeDamping(level, wires=q)

    def noisy_generator_circuit(noise_params, params_pqc):
        # ── verbatim copy of quantum.py:122-171 generator_circuit body ──
        idx = 0

        # Step 1: Hadamard initialization for superposition.
        for qubit in range(NUM_QUBITS):
            qml.Hadamard(wires=qubit)

        # Step 2: IQP encoding with parameterized RZ rotations.
        for qubit in range(NUM_QUBITS):
            if idx < len(params_pqc):
                qml.RZ(phi=params_pqc[idx], wires=qubit)
                idx += 1

        # Step 3: Apply noise encoding (IQP-style) — reuse g.encoding_layer
        # (reads params only; does not touch core/ state).
        g.encoding_layer(noise_params)

        # Step 4: Strongly Entangled Layers (+ per-layer noise channel).
        for layer in range(NUM_LAYERS):
            for qubit in range(NUM_QUBITS):
                if idx + 2 < len(params_pqc):
                    qml.Rot(
                        phi=params_pqc[idx],
                        theta=params_pqc[idx + 1],
                        omega=params_pqc[idx + 2],
                        wires=qubit,
                    )
                    idx += 3

            if NUM_QUBITS > 1:
                range_param = (layer % (NUM_QUBITS - 1)) + 1
                for qubit in range(NUM_QUBITS):
                    target_qubit = (qubit + range_param) % NUM_QUBITS
                    qml.CNOT(wires=[qubit, target_qubit])

            # PER-LAYER noise insertion: channel on every wire after each
            # entangling block (Assumption A1 — documented default).
            _apply_channel()

        # Step 5: Final measurement-preparation rotations.
        for qubit in range(NUM_QUBITS):
            if idx + 1 < len(params_pqc):
                qml.RX(phi=params_pqc[idx], wires=qubit)
                idx += 1
                qml.RY(phi=params_pqc[idx], wires=qubit)
                idx += 1

        # Step 6: Pauli-X and Pauli-Z expectations on each qubit.
        measurements = []
        for i in range(NUM_QUBITS):
            measurements.append(qml.expval(qml.PauliX(i)))
            measurements.append(qml.expval(qml.PauliZ(i)))
        return (*measurements,)

    qn = qml.QNode(
        noisy_generator_circuit, dev, interface="torch", diff_method=None
    )  # NOT backprop (Pitfall 2)
    return qn


# ─────────────────────────────────────────────────────────────────────────────
# Condition parsing — the 11 canonical tokens (D-12-02 / ROADMAP SC-2).
#   analytic
#   shots_8192  shots_1024
#   depol_{0.0,0.001,0.01,0.05}      (depolarizing p)
#   ampdamp_{0.0,0.001,0.01,0.05}    (amplitude-damping gamma)
# ─────────────────────────────────────────────────────────────────────────────
_SHOT_LEVELS = {"shots_8192": 8192, "shots_1024": 1024}
_DEPOL_LEVELS = {
    "depol_0.0": 0.0,
    "depol_0.001": 0.001,
    "depol_0.01": 0.01,
    "depol_0.05": 0.05,
}
_AMPDAMP_LEVELS = {
    "ampdamp_0.0": 0.0,
    "ampdamp_0.001": 0.001,
    "ampdamp_0.01": 0.01,
    "ampdamp_0.05": 0.05,
}
CONDITIONS: list[str] = (
    ["analytic"]
    + list(_SHOT_LEVELS)
    + list(_DEPOL_LEVELS)
    + list(_AMPDAMP_LEVELS)
)


def _build_qnode_for_condition(g: QuantumGenerator, condition: str):
    """Return the QNode for a non-analytic condition (analytic handled upstream)."""
    if condition in _SHOT_LEVELS:
        return make_shot_qnode(g, _SHOT_LEVELS[condition])
    if condition in _DEPOL_LEVELS:
        return make_noisy_qnode(g, "depolarizing", _DEPOL_LEVELS[condition])
    if condition in _AMPDAMP_LEVELS:
        return make_noisy_qnode(g, "amplitude_damping", _AMPDAMP_LEVELS[condition])
    raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def _condition_dims(condition: str) -> dict:
    """Extra long-form row dimensions for a condition (SENS-01 vs SENS-02)."""
    if condition == "analytic":
        return {"shots": None, "noise_model": None, "noise_level": None}
    if condition in _SHOT_LEVELS:
        return {
            "shots": _SHOT_LEVELS[condition],
            "noise_model": None,
            "noise_level": None,
        }
    if condition in _DEPOL_LEVELS:
        return {
            "shots": None,
            "noise_model": "depolarizing",
            "noise_level": _DEPOL_LEVELS[condition],
        }
    if condition in _AMPDAMP_LEVELS:
        return {
            "shots": None,
            "noise_model": "amplitude_damping",
            "noise_level": _AMPDAMP_LEVELS[condition],
        }
    raise ValueError(f"unknown condition {condition!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Dual-scale fidelity recompute (EVAL-05). ``full_metric_suite`` is reused
# UNCHANGED (D-12-03) — this driver only emits long-form rows tagging each
# metric with ``scale`` ∈ {"OD", "log_return"} plus the condition dims.
# ─────────────────────────────────────────────────────────────────────────────
def _emit_rows(
    metric_dict: dict,
    *,
    pipeline: str,
    seed: int,
    scale: str,
    condition: str,
) -> list[dict]:
    dims = _condition_dims(condition)
    rows: list[dict] = []
    for metric_name, value in metric_dict.items():
        rows.append(
            {
                "model_kind": "quantum",
                "pipeline": pipeline,
                "seed": int(seed),
                "metric_name": metric_name,
                "scale": scale,
                "value": float(value),
                "condition": condition,
                **dims,
            }
        )
    return rows


def compute_dualscale_metrics(
    pipeline: str,
    seed: int,
    condition: str,
    fake_samples_pm1: np.ndarray,
    csv_path: Path,
    inverse_kwargs_path: Path,
) -> tuple[dict, list[dict]]:
    """Recompute the unchanged fidelity suite on OD scale AND log-return scale.

    Returns ``(metrics_summary, long_form_rows)`` where ``metrics_summary``
    nests the raw ``full_metric_suite`` output per scale and ``long_form_rows``
    is the EVAL-05 dual-scale long-form list.
    """
    # Real reference windows (OD scale) — the (385,10) matrix the 09.1 runs used.
    real_od = real_windowed_od(csv_path)
    recon = reconstruct_od(pipeline, seed, fake_samples_pm1, inverse_kwargs_path)
    fake_od = recon["od_samples"]

    rows: list[dict] = []
    summary: dict = {}

    # OD scale (always available, both pipelines).
    od_metrics = full_metric_suite(real_od, fake_od)  # UNCHANGED (D-12-03)
    summary["OD"] = od_metrics
    rows += _emit_rows(
        od_metrics, pipeline=pipeline, seed=seed, scale="OD", condition=condition
    )

    # log-return / transformed scale (Pipeline B exposes a transformed array;
    # Pipeline A has no separate log-return scale -> OD is its only scale).
    if recon.get("transformed") is not None:
        # CR-01 fix: the log-return real reference MUST be the frozen Phase 11
        # recipe. run_dualscale_fidelity.build_real_references uses
        # `d_real["log_delta"]` (raw 1D, un-windowed, un-standardized) as the
        # Pipeline-B scale="log_return" reference so the numbers reconcile with
        # fidelity_dualscale.json / baseline_comparison.json's
        # scale="transformed" rows. The previous forward_logreturns(d_real["OD"])
        # + rolling_window construction is a DIFFERENT transform (standardized,
        # no dither) and did not reconcile with the frozen headline artifacts.
        # compute_emd/compute_moments flatten their inputs, so the raw 1D real
        # reference vs the (N,W) fake array reproduces the frozen emd/moment
        # rows exactly (same invariance proven on the OD scale to <1e-17).
        d_real = load_and_preprocess(str(csv_path))
        real_lr = d_real["log_delta"].cpu().numpy()
        fake_lr = np.asarray(recon["transformed"], dtype=np.float64)
        lr_metrics = full_metric_suite(real_lr, fake_lr)  # UNCHANGED (D-12-03)
        summary["log_return"] = lr_metrics
        rows += _emit_rows(
            lr_metrics,
            pipeline=pipeline,
            seed=seed,
            scale="log_return",
            condition=condition,
        )

    return summary, rows


# ─────────────────────────────────────────────────────────────────────────────
# Idempotent per-cell run-dir bundle (copied shape from
# run_baselines.py:456-522): rmtree on rerun (no stale partial bundle), then
# write config.yaml / samples.npy / metrics.json.
# ─────────────────────────────────────────────────────────────────────────────
def _write_bundle(
    run_dir: Path,
    *,
    config: dict,
    samples: np.ndarray,
    metrics: dict,
) -> None:
    import yaml

    if run_dir.exists():
        shutil.rmtree(run_dir)  # idempotent — no stale partial bundle
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    np.save(run_dir / "samples.npy", samples)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Headline aggregation (Plan 12-02 Task 3 — SENS-01 / SENS-02 deliverables).
#
# Walks every per-cell metrics.json under ``<out_root>/runs/*/{B,A}/{42,43,44}``
# and partitions the already-extended long-form ``rows[]`` (each row carries
# {model_kind,pipeline,seed,metric_name,scale,value,condition,shots,
# noise_model,noise_level} — emitted by the per-cell driver) into the two
# headline JSONs anchored at REPO/revision/results:
#
#   shot_noise_sensitivity.json   — SENS-01: conditions analytic/shots_8192/
#                                    shots_1024. Row keys include ``shots``
#                                    (None|8192|1024); noise_model/noise_level
#                                    dropped (not meaningful for SENS-01).
#   noise_model_sensitivity.json  — SENS-02: conditions depol_*/ampdamp_*.
#                                    Row keys include ``noise_model`` ∈
#                                    {depolarizing, amplitude_damping} and
#                                    ``noise_level`` (p or γ). depol_0.0 and
#                                    ampdamp_0.0 are the same physical p=0/γ=0
#                                    baseline but are kept as TWO distinct rows
#                                    (one per noise_model) so each degradation
#                                    curve owns its own zero anchor (documented
#                                    choice — Plan 12-02 Task 1 header note).
#
# This is an EXTEND-not-replace aggregation: the canonical baseline_comparison
# six-key contract {model_kind,pipeline,seed,metric_name,scale,value} is left
# byte-intact in every row; only condition + the relevant SENS dims are added.
# No mean±std here — the 3-seed degradation grids are RAW per-seed rows
# (D-12-02; the 5-seed mean±std roll-up is Plan 12-03 territory only).
# ─────────────────────────────────────────────────────────────────────────────
_SHOT_CONDITIONS = ("analytic",) + tuple(_SHOT_LEVELS)
_NOISE_CONDITIONS = tuple(_DEPOL_LEVELS) + tuple(_AMPDAMP_LEVELS)


def aggregate(out_root: Path) -> tuple[Path, Path]:
    """Aggregate per-cell bundles into the two headline SENS deliverables.

    Returns the (shot_noise_path, noise_model_path) written under
    REPO/revision/results. Idempotent — fully recomputed from the per-cell
    metrics.json bundles on every call.
    """
    import datetime as _dt

    runs_root = out_root / "runs"
    shot_rows: list[dict] = []
    noise_rows: list[dict] = []

    for metrics_path in sorted(runs_root.glob("*/*/*/metrics.json")):
        doc = json.loads(metrics_path.read_text())
        condition = doc["condition"]
        for row in doc["rows"]:
            if condition in _SHOT_CONDITIONS:
                # SENS-01: keep the six-key contract + condition + shots.
                shot_rows.append(
                    {
                        "model_kind": row["model_kind"],
                        "pipeline": row["pipeline"],
                        "seed": row["seed"],
                        "metric_name": row["metric_name"],
                        "scale": row["scale"],
                        "value": row["value"],
                        "condition": row["condition"],
                        "shots": row["shots"],
                    }
                )
            elif condition in _NOISE_CONDITIONS:
                # SENS-02: keep the six-key contract + condition +
                # noise_model + noise_level.
                noise_rows.append(
                    {
                        "model_kind": row["model_kind"],
                        "pipeline": row["pipeline"],
                        "seed": row["seed"],
                        "metric_name": row["metric_name"],
                        "scale": row["scale"],
                        "value": row["value"],
                        "condition": row["condition"],
                        "noise_model": row["noise_model"],
                        "noise_level": row["noise_level"],
                    }
                )
            else:  # pragma: no cover - defensive; CONDITIONS is closed
                raise ValueError(f"unpartitionable condition {condition!r}")

    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    seeds = sorted({r["seed"] for r in shot_rows} | {r["seed"] for r in noise_rows})
    pipelines = sorted(
        {r["pipeline"] for r in shot_rows} | {r["pipeline"] for r in noise_rows}
    )

    shot_doc = {
        "schema": (
            "long-form; extends baseline_comparison six-key contract "
            "{model_kind,pipeline,seed,metric_name,scale,value} with "
            "{condition,shots}"
        ),
        "deliverable": "SENS-01 shot-noise degradation",
        "model_kinds": sorted({r["model_kind"] for r in shot_rows}),
        "pipelines": pipelines,
        "seeds": seeds,
        "conditions": list(_SHOT_CONDITIONS),
        "shot_levels": {"analytic": None, **_SHOT_LEVELS},
        "generated_at": generated_at,
        "rows": shot_rows,
    }
    noise_doc = {
        "schema": (
            "long-form; extends baseline_comparison six-key contract "
            "{model_kind,pipeline,seed,metric_name,scale,value} with "
            "{condition,noise_model,noise_level}"
        ),
        "deliverable": "SENS-02 noise-channel degradation",
        "model_kinds": sorted({r["model_kind"] for r in noise_rows}),
        "pipelines": pipelines,
        "seeds": seeds,
        "conditions": list(_NOISE_CONDITIONS),
        "noise_models": {
            "depolarizing": sorted(set(_DEPOL_LEVELS.values())),
            "amplitude_damping": sorted(set(_AMPDAMP_LEVELS.values())),
        },
        "channel_insertion": "per-layer (after each entangling block)",
        "zero_anchor_note": (
            "depol_0.0 and ampdamp_0.0 are the same physical p=0/gamma=0 "
            "baseline; kept as two distinct rows (one per noise_model) so "
            "each degradation curve owns its own zero anchor"
        ),
        "generated_at": generated_at,
        "rows": noise_rows,
    }

    results_dir = REPO / "revision" / "results"
    shot_path = results_dir / "shot_noise_sensitivity.json"
    noise_path = results_dir / "noise_model_sensitivity.json"
    shot_path.write_text(json.dumps(shot_doc, indent=2, default=float))
    noise_path.write_text(json.dumps(noise_doc, indent=2, default=float))
    return shot_path, noise_path


def _frozen_analytic_paths(pipeline: str, seed: int) -> tuple[Path, Path]:
    """Frozen 09.1 ``samples.npy`` + ``inverse_kwargs.npz`` for a (pipeline, seed)."""
    base = (
        REPO
        / "revision"
        / "results"
        / "transform_ablation"
        / "runs"
        / pipeline
        / str(seed)
    )
    return base / "samples.npy", base / "inverse_kwargs.npz"


def main() -> None:
    """CLI entry point — one (pipeline, seed, condition) cell per invocation."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 12 SENS-01/02 per-cell inference driver — one "
            "(pipeline, seed, condition) cell per invocation. Requires "
            "PennyLane 0.44.0 (asserted at import; do NOT run via ./qgan_env)."
        )
    )
    ap.add_argument("--pipeline", choices=["A", "B"])
    ap.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46])
    ap.add_argument(
        "--condition",
        choices=CONDITIONS,
        help="One of the 11 tokens: analytic / shots_* / depol_* / ampdamp_*.",
    )
    ap.add_argument(
        "--emit-rollup",
        action="store_true",
        help=(
            "Aggregation mode (NOT a third driver file): walk every per-cell "
            "metrics.json under <out-root>/runs and emit the two headline "
            "deliverables shot_noise_sensitivity.json + "
            "noise_model_sensitivity.json. Mutually exclusive with the "
            "per-cell --pipeline/--seed/--condition args."
        ),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("revision/results/sensitivity"),
        help="Root under which runs/<condition>/<pipeline>/<seed>/ is created.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="Path to the OD CSV (same data the 09.1 quantum runs used).",
    )
    args = ap.parse_args()

    out_root = (
        args.out_root
        if Path(args.out_root).is_absolute()
        else (REPO / args.out_root)
    )

    if args.emit_rollup:
        if args.pipeline or args.seed is not None or args.condition:
            ap.error(
                "--emit-rollup is mutually exclusive with "
                "--pipeline/--seed/--condition"
            )
        shot_path, noise_path = aggregate(out_root)
        print(
            f"[run_sensitivity] emit-rollup -> {shot_path} , {noise_path}"
        )
        return

    if not (args.pipeline and args.seed is not None and args.condition):
        ap.error(
            "per-cell mode requires --pipeline, --seed and --condition "
            "(or use --emit-rollup for aggregation)"
        )

    # CR-02: fail fast with a clear message if no frozen run exists for this
    # (pipeline, seed) BEFORE any artifact path/dir is created — otherwise a
    # mis-typed seed dies deep inside torch.load with a raw FileNotFoundError
    # and the sweep retries the opaque failure forever.
    _ck_path = (
        REPO / "revision" / "results" / "transform_ablation" / "runs"
        / args.pipeline / str(args.seed) / "checkpoint.pt"
    )
    if not _ck_path.exists():
        ap.error(
            f"no frozen run for pipeline={args.pipeline} seed={args.seed}: "
            f"{_ck_path} not found"
        )

    run_dir = out_root / "runs" / args.condition / args.pipeline / str(args.seed)
    csv_path = _resolve_csv(args.csv_path)
    frozen_samples, frozen_inv = _frozen_analytic_paths(args.pipeline, args.seed)

    if args.condition == "analytic":
        # No regeneration (no-regeneration invariant, mirrors D-11-08): reuse
        # the frozen 09.1 samples.npy as the analytic reference column.
        samples = np.load(frozen_samples).astype(np.float64)
        regenerated = False
    else:
        g = load_trained_generator(args.pipeline, args.seed)
        qnode = _build_qnode_for_condition(g, args.condition)
        samples = generate_samples_on_qnode(
            g, qnode, n=int(np.load(frozen_samples).shape[0]), seed=args.seed
        )
        regenerated = True

    metrics_summary, long_form_rows = compute_dualscale_metrics(
        args.pipeline,
        args.seed,
        args.condition,
        samples,
        csv_path,
        frozen_inv,
    )

    dims = _condition_dims(args.condition)
    config = {
        "pipeline": args.pipeline,
        "seed": int(args.seed),
        "condition": args.condition,
        "regenerated": regenerated,
        "num_qubits": int(NUM_QUBITS),
        "num_layers": int(NUM_LAYERS),
        "window_length": int(WINDOW_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "noise_low": float(NOISE_LOW),
        "noise_high": float(NOISE_HIGH),
        "pennylane_version": qml.__version__,
        "channel_insertion": "per-layer (after each entangling block)",
        "csv_path": str(csv_path),
        **dims,
    }
    metrics = {
        "condition": args.condition,
        "pipeline": args.pipeline,
        "seed": int(args.seed),
        "metric_suite": metrics_summary,
        "rows": long_form_rows,
    }
    _write_bundle(run_dir, config=config, samples=samples, metrics=metrics)

    print(
        f"[run_sensitivity] pipeline={args.pipeline} seed={args.seed} "
        f"condition={args.condition} regenerated={regenerated} -> {run_dir} "
        f"(n_synth={samples.shape[0]}, samples.shape={samples.shape})"
    )


if __name__ == "__main__":
    main()
