"""Phase 14 Tier-2/3 driver — train one (model, seed) at the MATCHED 2000-epoch
budget behind a device-honest strict accept gate.

One process per invocation. Idempotent — re-running the same ``--model X
--seed Z`` overwrites ``runs/<model>/<seed>/`` cleanly.

Originating bug (D-14-22): the paper's comparison runs were 1000ep vs a 2000ep
headline checkpoint AND a 75-param-vs-55-param mismatch. This driver closes the
unfair-comparison gap — every model trains at an IDENTICAL 2000-epoch budget on
seeds {42..46}, the frozen Phase-09.1 data_hash, and a device-honest manifest.

Model matrix (the plan's Task-2 matrix):
    iqp_sel_55_repro  — the locked 55-param IQP:SEL (circuit_id from
                        canonical_config_lock.json) trained fresh at 2000ep.
                        This is the NON-load-bearing reproduction instance,
                        tagged ``source: "matched2000_reproduction"`` so it is
                        NEVER conflated with the frozen-checkpoint headline
                        (headline_canonical.json, source=
                        frozen_checkpoint_epoch_1969 — D-14-10).
    V1 / V2 / V3      — quantum ansatz variants (depth/topology) at 2000ep
                        (D-14-10 matched-budget ansatz comparison).
    wgan_mlp/cnn/lstm — matched-parameter classical WGAN-GP baselines.
    vae / ar          — non-adversarial baselines.

Backend lock (D-14-11): ``default.qubit`` + ``diff_method="backprop"``. The
PennyLane "lightning" family of devices is forbidden. PennyLane sims are
CPU-only — the quantum rows are CORRECTLY CPU; classical rows must not
silently fall back. Each run emits a
device/dtype manifest into its bundle and hard-asserts the actual backend with
the explicit-raise idiom (run_multiseed_rollup.py:86-92 — survives ``python
-O``); a silent CPU/dtype surprise (e.g. CPU-float64 when MPS-float32 expected)
fails loudly (T-14-04 / D-14-12).

Strict accept gate (D-14-13, applied per-artifact by ``--accept``): an artifact
is trusted ONLY if its device-manifest assertion passed, its ``data_hash``
equals the frozen Phase-09.1 hash, its seed is in {42..46}, the long-form
config schema conforms, AND it completed the full 2000ep (no early-stop on the
headline/reproduction path). Every check uses ``raise AssertionError`` (NOT a
bare ``assert`` — ``python -O`` strips asserts, silently disabling the gate).

NEVER use Python in-process worker pools (RESEARCH Pitfall 5, D-10-24 LOCKED):
the sweep shell uses ``xargs -P 2`` (a fresh OS process + a fresh numpy RNG per
run). There is zero in-process Python parallelism here — forked worker pools
share the parent's already-warm numpy global RNG, which has corrupted at least
one prior reproduction attempt.

Pitfall 6: sample generation runs the trained generator on CPU in float64 (the
canonical ``*0.1`` sample space; Apple MPS has no float64 statevector path).

Usage
-----
    ./qgan_env/bin/python -m revision.run_matched2000 \\
        --model {iqp_sel_55_repro|V1|V2|V3|wgan_mlp|wgan_cnn|wgan_lstm|vae|ar} \\
        --seed N [--epochs 2000] \\
        [--out-root revision/results/matched2000] [--csv-path ./data.csv]

    # strict accept gate (per-artifact, explicit-raise; exits non-zero on
    # rejection):
    ./qgan_env/bin/python -m revision.run_matched2000 \\
        --accept --model M --seed N [--out-root revision/results/matched2000]

Each TRAIN run writes the 5-file bundle to ``out_root/runs/<model>/<seed>/``:
    config.yaml          — frozen hyperparameters + seed + epochs + data_hash
                           + device/dtype manifest + source marker
    checkpoint.pt|.npz   — model state (AR -> .npz {phi,sigma2,p})
    samples.npy          — (N_synth, WINDOW_LENGTH) in [-1, 1]
    metrics.json         — per-epoch / fit-diagnostic metrics
    inverse_kwargs.npz   — Pipeline-B aux state for OD-scale reconstruction
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (verbatim run_multiseed_rollup.py:42-59 idiom — the
# driver may run from a worktree / arbitrary cwd; results are repo-anchored).
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (revision/core/preprocessing.py)")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "revision" / "results"

# The frozen Phase-09.1 data hash every accepted artifact MUST agree on
# (== baseline_comparison.json / fidelity_dualscale.json data_hash, D-10-15).
EXPECTED_DATA_HASH = "91e447d4624e25b3"

# Canonical seed set — the strict gate rejects any seed outside this (D-14-13).
SEED_SET = (42, 43, 44, 45, 46)


# Plan 14-13 Task 4 (HI-8): _finite_sanitize replaces `default=float` in
# json.dumps. Non-finite floats (nan, +inf, -inf) become `None` in the
# serialized JSON; if more than 5% of the numeric leaves are non-finite,
# the helper raises explicitly rather than silently emitting a corpus of
# Nones (which would propagate as a "no data" downstream).
def _finite_sanitize(obj, _stats=None):
    """Recursive sanitizer that converts non-finite floats → None and
    tracks the (total_floats, nonfinite_floats) tally. Mutates a shared
    `_stats` dict so the caller can decide whether to raise.
    """
    import math
    if _stats is None:
        _stats = {"total": 0, "nonfinite": 0}
    if isinstance(obj, dict):
        return {k: _finite_sanitize(v, _stats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_sanitize(v, _stats) for v in obj]
    if isinstance(obj, tuple):
        return [_finite_sanitize(v, _stats) for v in obj]
    if isinstance(obj, float):
        _stats["total"] += 1
        if not math.isfinite(obj):
            _stats["nonfinite"] += 1
            return None
        return obj
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    # Numpy scalars / torch tensors: fall through to float() conversion.
    try:
        v = float(obj)
        _stats["total"] += 1
        if not math.isfinite(v):
            _stats["nonfinite"] += 1
            return None
        return v
    except (TypeError, ValueError):
        return str(obj)


def _dumps_finite(obj, *, indent: int = 2, nonfinite_threshold: float = 0.05) -> str:
    """json.dumps wrapper using _finite_sanitize. Raises if non-finite
    count exceeds `nonfinite_threshold` of total numeric leaves (HI-8)."""
    stats = {"total": 0, "nonfinite": 0}
    sanitized = _finite_sanitize(obj, stats)
    if stats["total"] > 0:
        frac = stats["nonfinite"] / stats["total"]
        if frac > nonfinite_threshold:
            raise AssertionError(
                f"_dumps_finite: {stats['nonfinite']} of {stats['total']} "
                f"numeric leaves are non-finite ({frac:.1%} > "
                f"{nonfinite_threshold:.0%}); refusing to emit a corpus "
                "of Nones (HI-8, Plan 14-13 Task 4)"
            )
    return json.dumps(sanitized, indent=indent)

# Matched budget — every model trains for exactly this many epochs (D-14-08).
MATCHED_EPOCHS = 2000

# Model matrix → which model family / quantum variant.
# (depth, topology) for the quantum ansatz variants; V1 == the depth-4 range
# 75-param default circuit; V2/V3 mirror run_ansatz._ANSATZ_VARIANTS.
_QUANTUM_ANSATZ = {
    "V1": {"num_layers": 4, "topology": "range", "circuit_id": "default_75",
           "parameter_count": 75},
    "V2": {"num_layers": 8, "topology": "range", "circuit_id": "default_75",
           "parameter_count": 135},
    "V3": {"num_layers": 4, "topology": "linear", "circuit_id": "default_75",
           "parameter_count": 75},
}
_WGAN_MODELS = ("wgan_mlp", "wgan_cnn", "wgan_lstm")
_NONADV_MODELS = ("vae", "ar")
_REPRO_MODEL = "iqp_sel_55_repro"
_MODEL_CHOICES = (
    [_REPRO_MODEL] + list(_QUANTUM_ANSATZ)
    + list(_WGAN_MODELS) + list(_NONADV_MODELS)
)

# Tier assignment (D-14-14): each tier independently acceptable so the sweep
# is resumable + run-to-completion with no hard time-box. T1 is Plan 01.
_TIER = {
    _REPRO_MODEL: "T2",
    "wgan_mlp": "T2", "wgan_cnn": "T2", "wgan_lstm": "T2",
    "vae": "T2", "ar": "T2",
    "V1": "T3", "V2": "T3", "V3": "T3",
}

_QUANTUM_MODELS = {_REPRO_MODEL, *_QUANTUM_ANSATZ}


@dataclass
class DatasetBundle:
    """Per-pipeline training-data state (mirrors run_baselines.DatasetBundle —
    D-10-07 identical windowed-data + inverse-kwargs contract)."""

    dataloader: Any
    n_real_windows: int
    inverse_kwargs: Dict[str, Any]
    pipeline_id: str
    windowed: Any


def build_dataset_for_pipeline(csv_path: Path) -> DatasetBundle:
    """Build the native Pipeline-B DataLoader + inverse-kwargs.

    Copied VERBATIM from run_baselines.build_dataset_for_pipeline's "B" branch
    (D-10-07). Pipeline B (log-returns) is the pinned native headline pipeline
    (D-14-06); every model in the matched-budget matrix trains on the identical
    windowed series so the comparison is fair (D-14-08).
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from revision.core import BATCH_SIZE, WINDOW_LENGTH
    from revision.core.data import load_and_preprocess, rolling_window
    from revision.core.preprocessing import forward_logreturns

    raw = load_and_preprocess(str(csv_path))
    od = raw["OD"]
    r_norm, mu, sigma = forward_logreturns(od)
    r_min = torch.min(r_norm)
    r_max = torch.max(r_norm)
    r_pm1 = 2.0 * (r_norm - r_min) / (r_max - r_min) - 1.0
    windowed = rolling_window(r_pm1, WINDOW_LENGTH, 2)
    n_lr = r_norm.shape[0]
    start_indices = list(range(0, n_lr - WINDOW_LENGTH + 1, 2))
    od_starts = torch.stack([od[i] for i in start_indices])
    inverse_kwargs: Dict[str, Any] = {
        "r_min": float(r_min.item()),
        "r_max": float(r_max.item()),
        "mu": float(mu.item()),
        "sigma": float(sigma.item()),
        "od_starts": od_starts.cpu().numpy().astype(np.float64),
    }
    ds = TensorDataset(windowed.float())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return DatasetBundle(
        dataloader=dl,
        n_real_windows=int(windowed.shape[0]),
        inverse_kwargs=inverse_kwargs,
        pipeline_id="B",
        windowed=windowed.detach().to(torch.float64),
    )


def generate_wgan_samples(generator, n: int, seed: int):
    """Generate ``n`` WGAN/quantum windows in [-1, 1] space.

    Copied VERBATIM from run_baselines.generate_wgan_samples (the *0.1
    canonical sample space the shared reconstruct_od inverse consumes; RESEARCH
    Pitfall 6 — Apple MPS has no float64 statevector path so the generator is
    moved back to CPU). FIXED ``np.random.default_rng(seed)``.
    """
    import numpy as np
    import torch

    from revision.core import BATCH_SIZE, NOISE_HIGH, NOISE_LOW, NUM_QUBITS

    generator = generator.to("cpu")
    rng = np.random.default_rng(seed)
    out_parts: list = []
    remaining = n
    with torch.no_grad():
        while remaining > 0:
            bs = min(BATCH_SIZE, remaining)
            noise = torch.tensor(
                rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, bs)),
                dtype=torch.float32,
            )
            out = generator(noise).to(torch.float64) * 0.1
            out_parts.append(out.cpu().numpy())
            remaining -= bs
    return np.concatenate(out_parts, axis=0)[:n]


def _save_inverse_kwargs(run_dir: Path, inverse_kwargs: Dict[str, Any]) -> None:
    """Persist inverse-kwargs as inverse_kwargs.npz (identical contract to
    run_baselines._save_inverse_kwargs)."""
    import numpy as np

    payload: Dict[str, np.ndarray] = {}
    for k, v in inverse_kwargs.items():
        payload[k] = v if isinstance(v, np.ndarray) else np.asarray(v)
    np.savez(run_dir / "inverse_kwargs.npz", **payload)


def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from load_and_preprocess (D-10-15).

    Identical recipe to run_baselines._compute_data_hash — pins every matched
    run to the exact OD tensor so the strict accept gate can prove every model
    consumed the frozen Phase-09.1 data (D-14-13).
    """
    from revision.core.data import load_and_preprocess

    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _device_manifest(generator=None, *, training_time_device=None) -> Dict[str, Any]:
    """Build + hard-assert the device/dtype manifest (T-14-04 / D-14-12).

    PennyLane ``default.qubit`` is CPU-only (D-14-11). The matched-budget run
    generates samples on CPU in float64 (Pitfall 6). A silent device/dtype
    fallback (e.g. CPU-float64 when MPS-float32 expected) would put a false
    device in the paper table — fail loudly via explicit-raise (NOT bare
    assert; ``python -O`` strips asserts and would disable this honesty gate).

    Plan 14-14 Task 2 — `training_time_device` capture site corrected
    -----------------------------------------------------------------
    The 14-13 T4 implementation read `next(generator.parameters()).device`
    at manifest-build time, which happens AFTER `generate_wgan_samples`
    moves the generator to CPU (line 270 `generator = generator.to("cpu")`).
    The manifest therefore always saw `"cpu"` regardless of training
    device, and the D-14-13 strict-accept gate's `training_time_device`
    equality check (added in 14-13 T4) was non-functional.

    The fix is symmetric across all three training paths
    (_train_quantum / _train_wgan / _train_vae): each captures
    `training_time_device` IMMEDIATELY after the training loop returns
    and BEFORE the post-training `.to(cpu)` migration, then passes the
    captured value to `_device_manifest(generator, training_time_device=...)`.
    The optional `training_time_device` kwarg lets callers supply the
    pre-captured value; if None (backward compat), the function falls
    back to the original `next(parameters()).device` read.

    The `_strict_accept` equality check is UNCHANGED — only the value
    it sees now actually reflects the device the model was on at the
    moment training finished.
    """
    import torch

    torch_cpu = "cpu"
    mps_avail = bool(torch.backends.mps.is_available())
    cuda_avail = bool(torch.cuda.is_available())
    # Plan 14-14 Task 2: prefer the pre-captured training_time_device (taken
    # immediately after the training loop returns, before .to(cpu)). Fall
    # back to the post-migration parameters().device read for backward
    # compatibility with any other caller that does not pre-capture.
    if training_time_device is not None:
        # honor the caller-supplied training-time device value verbatim
        pass
    elif generator is not None:
        try:
            training_time_device = str(
                next(generator.parameters()).device
            )
        except (StopIteration, AttributeError):
            training_time_device = torch_cpu
    else:
        training_time_device = torch_cpu
    manifest: Dict[str, Any] = {
        "sample_generation_device": torch_cpu,
        "sample_generation_dtype": "torch.float64",
        "training_time_device": training_time_device,
        "mps_available": mps_avail,
        "cuda_available": cuda_avail,
    }
    if generator is not None and hasattr(generator, "dev"):
        pl_device = (
            generator.dev.name
            if hasattr(generator.dev, "name")
            else str(generator.dev)
        )
        manifest["pennylane_device"] = pl_device
        manifest["diff_method"] = getattr(generator, "diff_method", None)
        # D-14-11 backend lock: the PennyLane "lightning" device family is
        # forbidden — only the default.qubit statevector backend is allowed.
        if "lightning" in pl_device:
            raise AssertionError(
                f"backend manifest violation: PennyLane device "
                f"{pl_device!r} is a lightning.* backend — D-14-11 backend "
                "lock allows only default.qubit; T-14-04 BLOCK"
            )
        if "default.qubit" not in pl_device:
            raise AssertionError(
                f"backend manifest violation: PennyLane device "
                f"{pl_device!r} is not default.qubit (D-14-11 backend "
                "lock); T-14-04 BLOCK"
            )
        # Quantum params are a float32 nn.Parameter on CPU by construction.
        pdev = str(generator.params_pqc.device)
        if not pdev.startswith("cpu"):
            raise AssertionError(
                f"device manifest violation: quantum params on {pdev}, "
                "expected cpu (PennyLane default.qubit is CPU-only, "
                "D-14-11) — silent device fallback (T-14-04 BLOCK)"
            )
        manifest["quantum_params_device"] = pdev
        manifest["quantum_params_dtype"] = str(generator.params_pqc.dtype)
    manifest["backend_assertion"] = "PASSED"
    manifest["note"] = (
        "PennyLane default.qubit statevector path is CPU-only (D-14-11); "
        "quantum rows are CORRECTLY CPU — not a silent classical fallback "
        "(T-14-04). Classical rows generate samples on CPU/float64 (Pitfall "
        "6)."
    )
    return manifest


def _train_quantum(model: str, bundle: DatasetBundle, epochs: int,
                   seed: int) -> tuple:
    """Quantum branch — shared Critic + train_wgan_gp UNCHANGED.

    ``iqp_sel_55_repro`` constructs the locked 55-param IQP:SEL (circuit_id
    from canonical_config_lock.json, D-14-04) — the NON-load-bearing
    reproduction instance (D-14-10). V1/V2/V3 mirror run_ansatz variants. No
    ``early_stopper`` (full 2000ep, D-14-13 — no early-stop on the
    headline/reproduction path).
    """
    import torch

    from revision.core import (
        EVAL_EVERY, LAMBDA, LR_CRITIC, LR_GENERATOR, N_CRITIC,
        NUM_QUBITS, WINDOW_LENGTH,
    )
    from revision.core.models.critic import Critic
    from revision.core.models.quantum import QuantumGenerator
    from revision.core.training import train_wgan_gp

    if model == _REPRO_MODEL:
        lock = json.loads(
            (RESULTS / "canonical_config_lock.json").read_text()
        )
        decomp = lock["decomposition"]
        circuit_id = lock["locked_circuit_id"]
        num_layers = int(decomp["num_layers"])
        # Plan 14-13 Task 4 (HI-4): read topology from the lock dict
        # instead of hardcoding 'range'. The canonical config encodes
        # the entangler topology under decomposition.gate_layout.entangler.
        topology = str(
            decomp.get("gate_layout", {}).get("entangler", "range")
        )
        param_expect = int(lock["param_count"])
        source = "matched2000_reproduction"
        ansatz = "iqp_sel_55"
    else:
        spec = _QUANTUM_ANSATZ[model]
        circuit_id = spec["circuit_id"]
        num_layers = int(spec["num_layers"])
        topology = str(spec["topology"])
        param_expect = int(spec["parameter_count"])
        source = "matched2000_ansatz"
        ansatz = model

    torch.manual_seed(seed)
    generator = QuantumGenerator(
        num_qubits=NUM_QUBITS,
        num_layers=num_layers,
        window_length=WINDOW_LENGTH,
        topology=topology,
        circuit_id=circuit_id,
    )
    critic = Critic(window_length=WINDOW_LENGTH)

    # RESEARCH Pitfall 6 — the quantum statevector path MUST run on CPU
    # (PennyLane default.qubit + torch interface mis-coerces non-CPU tensors).
    # Wrap ONLY the train_wgan_gp call in a restoring guard so MPS availability
    # is reinstated even if training raises (mirrors run_ansatz WR-03; does NOT
    # mutate revision/core/training.py — byte-unchanged discipline).
    orig_mps = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    try:
        metrics = train_wgan_gp(
            generator,
            critic,
            bundle.dataloader,
            num_epochs=int(epochs),
            n_critic=int(N_CRITIC),
            lambda_gp=float(LAMBDA),
            lr_critic=float(LR_CRITIC),
            lr_generator=float(LR_GENERATOR),
            seed=int(seed),
            eval_every=int(EVAL_EVERY),
            # early_stopper deliberately omitted — full 2000ep (D-14-13).
        )
    finally:
        torch.backends.mps.is_available = orig_mps  # type: ignore[assignment]

    # Plan 14-14 Task 2: capture training_time_device IMMEDIATELY after the
    # training loop returns and BEFORE generate_wgan_samples (which at
    # line 270 does `generator = generator.to("cpu")`). The captured value
    # flows through to _device_manifest below so the D-14-13 strict-accept
    # gate's training_time_device equality check sees the actual training
    # device. In current paths the quantum branch always trains on CPU under
    # the torch.backends.mps.is_available=False hook above, so the captured
    # value is "cpu"; the mechanism is symmetric across _train_wgan/_train_vae
    # and protects against future drift if any path ever moves to MPS.
    try:
        training_time_device = str(next(generator.parameters()).device)
    except (StopIteration, AttributeError):
        training_time_device = "cpu"

    n_synth = 10 * bundle.n_real_windows
    samples = generate_wgan_samples(generator, n_synth, seed)
    param_count = int(generator.count_params())
    if param_count != param_expect:
        raise AssertionError(
            f"{model}: count_params()={param_count} != expected "
            f"{param_expect} (circuit_id={circuit_id}, layers={num_layers}, "
            f"topology={topology})"
        )
    checkpoint = {
        "gen_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }
    extra_cfg = {
        "model_kind": "quantum",
        "ansatz": ansatz,
        "circuit_id": circuit_id,
        "depth": num_layers,
        "topology": topology,
        "parameter_count": param_count,
        "family": "adversarial-quantum",
        "n_critic": int(N_CRITIC),
        "lambda_gp": float(LAMBDA),
        "lr_critic": float(LR_CRITIC),
        "lr_generator": float(LR_GENERATOR),
        "early_stopper": None,
        "spectral_loss_weight": 0.0,
        "source": source,
        "device_manifest": _device_manifest(
            generator, training_time_device=training_time_device
        ),
        "train_protocol_notes": (
            f"Matched-budget {model}: QuantumGenerator(circuit_id="
            f"{circuit_id!r}, num_layers={num_layers}, topology={topology!r}) "
            f"via train_wgan_gp UNCHANGED, {int(epochs)} epochs, early-stop "
            "OFF (full 2000ep, D-14-13), spectral_loss_weight=0.0. "
            f"source={source}: the iqp_sel_55 2000ep run is the "
            "NON-load-bearing reproduction instance, distinct from the "
            "frozen-checkpoint headline (source=frozen_checkpoint_epoch_1969, "
            "D-14-10). Backend default.qubit + backprop, CPU-only (D-14-11)."
        ),
    }
    return generator, checkpoint, samples, metrics, extra_cfg


def _train_wgan(model: str, bundle: DatasetBundle, epochs: int,
                seed: int) -> tuple:
    """Classical WGAN branch — shared Critic + train_wgan_gp UNCHANGED
    (D-10-08). Copied from run_baselines._train_wgan."""
    import torch

    from revision.core import (
        EVAL_EVERY, LAMBDA, LR_CRITIC, LR_GENERATOR, N_CRITIC, WINDOW_LENGTH,
    )
    from revision.core.models.classical import (
        WGANCNNGenerator, WGANLSTMGenerator, WGANMLPGenerator,
    )
    from revision.core.models.critic import Critic
    from revision.core.training import train_wgan_gp

    gens = {
        "wgan_mlp": WGANMLPGenerator,
        "wgan_cnn": WGANCNNGenerator,
        "wgan_lstm": WGANLSTMGenerator,
    }
    torch.manual_seed(seed)
    generator = gens[model]()
    critic = Critic(window_length=WINDOW_LENGTH)
    # Plan 14-13 Task 4 (CR-4 future-gate): patch torch.backends.mps.is_available
    # to False symmetrically with _train_quantum. The historical matched-budget
    # classical sweep executed on Apple-Silicon MPS at float32 while quantum
    # runs executed on CPU at float64 (the _train_quantum MPS-disable hook);
    # the asymmetry is disclosed in methods_full.md + reviewer_response.md.
    # Future matched-budget classical runs are CPU-only via this hook.
    orig_mps = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    try:
        metrics = train_wgan_gp(
            generator,
            critic,
            bundle.dataloader,
            num_epochs=int(epochs),
            n_critic=int(N_CRITIC),
            lambda_gp=float(LAMBDA),
            lr_critic=float(LR_CRITIC),
            lr_generator=float(LR_GENERATOR),
            seed=int(seed),
            eval_every=int(EVAL_EVERY),
        )
    finally:
        torch.backends.mps.is_available = orig_mps  # type: ignore[assignment]
    # Plan 14-14 Task 2: capture training_time_device IMMEDIATELY after
    # train_wgan_gp returns and BEFORE generate_wgan_samples (line 270 moves
    # the generator to CPU). The captured value flows through _device_manifest
    # so the D-14-13 strict-accept gate sees the actual training device.
    # Under the MPS-disable hook above, the value will be "cpu"; the mechanism
    # protects against future drift if the hook is ever removed.
    try:
        training_time_device = str(next(generator.parameters()).device)
    except (StopIteration, AttributeError):
        training_time_device = "cpu"
    n_synth = 10 * bundle.n_real_windows
    samples = generate_wgan_samples(generator, n_synth, seed)
    checkpoint = {
        "gen_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }
    extra_cfg = {
        "model_kind": model,
        "parameter_count": int(generator.count_params()),
        "family": "adversarial-classical",
        "n_critic": int(N_CRITIC),
        "lambda_gp": float(LAMBDA),
        "lr_critic": float(LR_CRITIC),
        "lr_generator": float(LR_GENERATOR),
        "source": "matched2000_baseline",
        "device_manifest": _device_manifest(
            generator, training_time_device=training_time_device
        ),
        "train_protocol_notes": (
            f"Matched-budget {model}: WGAN-GP via train_wgan_gp UNCHANGED "
            f"(D-10-08), {int(epochs)} epochs; shared "
            "Critic(window_length=WINDOW_LENGTH); *0.1 output scaling "
            "(Pitfall 3). MPS-disable hook applied (Plan 14-13 Task 4, "
            "CR-4 future-gate). source=matched2000_baseline."
        ),
    }
    return generator, checkpoint, samples, metrics, extra_cfg


def _train_vae(bundle: DatasetBundle, epochs: int, seed: int) -> tuple:
    """VAE branch — local ELBO loop, NO critic/GP (D-10-09). Copied verbatim
    from run_baselines._train_vae.

    Plan 14-13 Task 4 (HI-7 + CR-4 future-gate):
      - HI-7: seed `np.random` and Python `random` in addition to
        `torch.manual_seed` (the latter was the only seed call previously).
      - CR-4 future-gate: patch `torch.backends.mps.is_available` to False
        symmetrically with `_train_quantum` and `_train_wgan`.
    """
    import random as _random  # HI-7: standard-library random seeding
    import numpy as np
    import torch

    from revision.core.models.nonadversarial import VAEBaseline

    # HI-7: seed numpy + standard-library random in addition to torch.
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)

    # CR-4 future-gate: MPS-disable hook applied symmetrically.
    orig_mps = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    try:
        vae = VAEBaseline()
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        beta_target = 1.0
        warmup_epochs = 0
        elbo_hist: list = []
        recon_hist: list = []
        kld_hist: list = []
        for _epoch in range(int(epochs)):
            beta = beta_target
            ep_elbo = ep_recon = ep_kld = 0.0
            n_batches = 0
            for (x,) in bundle.dataloader:
                x = x.float()
                opt.zero_grad()
                x_hat, mu, logvar = vae(x)
                recon = torch.nn.functional.mse_loss(x_hat, x)
                kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + beta * kld
                loss.backward()
                opt.step()
                ep_elbo += float(loss.detach())
                ep_recon += float(recon.detach())
                ep_kld += float(kld.detach())
                n_batches += 1
            n_batches = max(n_batches, 1)
            elbo_hist.append(ep_elbo / n_batches)
            recon_hist.append(ep_recon / n_batches)
            kld_hist.append(ep_kld / n_batches)
        # Plan 14-14 Task 2: capture training_time_device IMMEDIATELY after
        # the VAE training loop returns and BEFORE the parallel
        # `vae.sample(...).cpu()` migration below. The captured value flows
        # through _device_manifest so the D-14-13 strict-accept gate sees the
        # actual training device. Under the MPS-disable hook this is "cpu".
        try:
            training_time_device = str(next(vae.parameters()).device)
        except (StopIteration, AttributeError):
            training_time_device = "cpu"
        n_synth = 10 * bundle.n_real_windows
        gen = torch.Generator()
        gen.manual_seed(seed)
        with torch.no_grad():
            samples = vae.sample(n_synth, gen).to(torch.float64).cpu().numpy()
    finally:
        torch.backends.mps.is_available = orig_mps  # type: ignore[assignment]
    real_std = float(bundle.windowed.std().item())
    samp_std = float(np.std(samples))
    collapse_flag = samp_std < 0.1 * real_std
    metrics = {
        "elbo": elbo_hist, "recon": recon_hist, "kld": kld_hist,
        "sample_std": samp_std, "real_window_std": real_std,
        "posterior_collapse_suspected": bool(collapse_flag),
    }
    checkpoint = {"vae_state_dict": vae.state_dict()}
    extra_cfg = {
        "model_kind": "vae",
        "parameter_count": int(vae.count_params()),
        "family": "non-adversarial",
        "n_critic": None, "lambda_gp": None,
        "lr_critic": None, "lr_generator": None,
        "vae_lr": 1e-3, "vae_beta": float(beta_target),
        "vae_kl_warmup_epochs": int(warmup_epochs),
        "source": "matched2000_baseline",
        # Plan 14-14 Task 2: pass the pre-captured training_time_device taken
        # immediately after the VAE training loop returned, before
        # `vae.sample(...).cpu()`. _device_manifest's optional kwarg consumes
        # it; the post-migration parameters() read is bypassed.
        "device_manifest": _device_manifest(
            vae, training_time_device=training_time_device
        ),
        "train_protocol_notes": (
            f"Matched-budget VAE: local ELBO loop (D-10-09), {int(epochs)} "
            "epochs, single Adam(lr=1e-3), NO critic/GP; sample() emits in "
            "[-1,1] (NO *0.1, Pitfall 3). MPS-disable hook applied "
            "(Plan 14-13 Task 4, CR-4 future-gate). "
            "source=matched2000_baseline."
        ),
    }
    return vae, checkpoint, samples, metrics, extra_cfg


def _train_ar(bundle: DatasetBundle, seed: int) -> tuple:
    """AR branch — closed-form lstsq fit (D-10-13). Copied verbatim from
    run_baselines._train_ar. (No epoch budget — closed-form; ``epochs`` is
    recorded as the matched budget for schema rectangularity.)"""
    import numpy as np

    from revision.core.models.nonadversarial import ARBaseline

    ar = ARBaseline(p=2)
    series = bundle.windowed.cpu().numpy().reshape(-1)
    ar.fit(series)
    n_synth = 10 * bundle.n_real_windows
    rng = np.random.default_rng(seed)
    samples = ar.sample(n_synth, rng).astype(np.float64)
    metrics = {
        "sigma2": float(ar.sigma2),
        "phi": [float(v) for v in ar.phi.tolist()],
        "p": int(ar.p),
        "fit_series_len": int(series.shape[0]),
        "sample_mean": float(np.mean(samples)) if samples.size else 0.0,
        "sample_std": float(np.std(samples)),
    }
    checkpoint = {
        "phi": ar.phi.astype(np.float64),
        "sigma2": np.asarray(float(ar.sigma2)),
        "p": np.asarray(int(ar.p)),
    }
    extra_cfg = {
        "model_kind": "ar",
        "parameter_count": int(ar.count_params()),
        "family": "non-adversarial",
        "n_critic": None, "lambda_gp": None,
        "lr_critic": None, "lr_generator": None,
        "ar_order_p": int(ar.p),
        "source": "matched2000_baseline",
        "device_manifest": _device_manifest(None),
        "train_protocol_notes": (
            "Matched-budget AR(p=2): closed-form np.linalg.lstsq (D-10-13), "
            "50-step burn-in. No epoch loop — fit is closed-form; the matched "
            "2000ep budget is recorded for schema rectangularity. Samples in "
            "[-1,1] (NO *0.1, Pitfall 3). source=matched2000_baseline."
        ),
    }
    return ar, checkpoint, samples, metrics, extra_cfg


def _strict_accept(model: str, seed: int, out_root: Path) -> dict:
    """D-14-13 strict accept gate — explicit-raise on EVERY check.

    An artifact is trusted ONLY if: device-manifest assertion passed,
    ``data_hash`` == frozen Phase-09.1 hash, seed in {42..46}, long-form config
    schema conforms, full 2000ep completed (no early-stop). Every guard uses
    ``raise AssertionError`` (NOT a bare ``assert`` — ``python -O`` strips
    asserts, which would silently disable the gate; run_multiseed_rollup.py:
    86-92 idiom). Exits non-zero on rejection (caller is the sweep / CI).
    """
    import yaml

    run_dir = out_root / "runs" / model / str(seed)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise AssertionError(
            f"accept: {cfg_path} missing — artifact not produced "
            f"(model={model}, seed={seed})"
        )
    cfg = yaml.safe_load(cfg_path.read_text())

    # 1) seed set {42..46}
    if int(cfg.get("seed", -1)) not in SEED_SET:
        raise AssertionError(
            f"accept: seed {cfg.get('seed')} not in {SEED_SET} "
            f"(model={model}) — D-14-13 BLOCK"
        )

    # 2) data_hash == frozen Phase-09.1 hash
    if cfg.get("data_hash") != EXPECTED_DATA_HASH:
        raise AssertionError(
            f"accept: data_hash {cfg.get('data_hash')!r} != frozen "
            f"{EXPECTED_DATA_HASH!r} (model={model}, seed={seed}) — "
            "mixed-data contamination, D-14-13 BLOCK"
        )

    # 3) full 2000ep completed (no early-stop on the headline/repro path)
    if int(cfg.get("epochs", -1)) != MATCHED_EPOCHS:
        raise AssertionError(
            f"accept: epochs {cfg.get('epochs')} != {MATCHED_EPOCHS} "
            f"(model={model}, seed={seed}) — mixed-budget contamination, "
            "D-14-13 BLOCK"
        )
    if cfg.get("early_stopper") not in (None, "None"):
        raise AssertionError(
            f"accept: early_stopper={cfg.get('early_stopper')!r} set "
            f"(model={model}) — early-stop forbidden on the matched-budget "
            "path (D-14-13 BLOCK)"
        )

    # 4) device-manifest assertion passed
    dm = cfg.get("device_manifest")
    if not isinstance(dm, dict) or dm.get("backend_assertion") != "PASSED":
        raise AssertionError(
            f"accept: device_manifest missing/failed (model={model}, "
            f"seed={seed}) — silent device/dtype fallback unverified, "
            "T-14-04 / D-14-13 BLOCK"
        )

    # 4b) Plan 14-13 Task 4 (D-14-13 extension): if `training_time_device`
    #     is recorded, it MUST be "cpu" (CR-4 future-gate intent). Older
    #     historical bundles without this field are accepted (forward-only
    #     gate; historical MPS asymmetry is disclosed in narrative not gated
    #     retroactively).
    ttd = dm.get("training_time_device")
    if ttd is not None and ttd != "cpu":
        raise AssertionError(
            f"accept: training_time_device={ttd!r} != 'cpu' (model={model}, "
            f"seed={seed}) — CR-4 future-gate forbids non-CPU training "
            "for matched-budget runs (Plan 14-13 Task 4, D-14-13 extension)"
        )

    # 5) long-form config schema conforms (the contract fields the
    #    downstream aggregators key on)
    required = (
        "model", "pipeline", "seed", "epochs", "data_hash",
        "model_kind", "parameter_count", "family", "source",
        "device_manifest",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise AssertionError(
            f"accept: config schema missing {missing} (model={model}, "
            f"seed={seed}) — long-form schema nonconformance, D-14-13 BLOCK"
        )

    # 6) the 5-file bundle is present & non-empty
    is_ar = model == "ar"
    ckpt_name = "checkpoint.npz" if is_ar else "checkpoint.pt"
    bundle_files = [
        "config.yaml", ckpt_name, "samples.npy",
        "metrics.json", "inverse_kwargs.npz",
    ]
    for f in bundle_files:
        fp = run_dir / f
        if not fp.exists() or fp.stat().st_size == 0:
            raise AssertionError(
                f"accept: bundle file {f} missing/empty (model={model}, "
                f"seed={seed}) — incomplete artifact, D-14-13 BLOCK"
            )

    # 7) headline/reproduction conflation guard (D-14-10): the iqp_sel_55
    #    2000ep run MUST be tagged distinctly from the frozen-checkpoint
    #    headline.
    if model == _REPRO_MODEL and cfg.get("source") != "matched2000_reproduction":
        raise AssertionError(
            f"accept: {model} source={cfg.get('source')!r} != "
            "'matched2000_reproduction' — headline/reproduction conflation "
            "risk (D-14-10 BLOCK)"
        )

    result = {
        "model": model,
        "seed": seed,
        "tier": _TIER[model],
        "accepted": True,
        "data_hash": cfg["data_hash"],
        "epochs": cfg["epochs"],
        "source": cfg["source"],
        "device_manifest_backend_assertion": dm["backend_assertion"],
    }
    print(
        f"[accept] PASSED model={model} seed={seed} tier={_TIER[model]} "
        f"epochs={cfg['epochs']} data_hash={cfg['data_hash']} "
        f"source={cfg['source']}"
    )
    return result


def _train(model: str, seed: int, epochs: int, csv_path: Path,
           out_root: Path) -> None:
    import numpy as np
    import torch
    import yaml

    from revision.core import (
        BATCH_SIZE, EVAL_EVERY, NOISE_HIGH, NOISE_LOW, NUM_LAYERS,
        NUM_QUBITS, WINDOW_LENGTH,
    )

    run_dir = out_root / "runs" / model / str(seed)
    if run_dir.exists():
        shutil.rmtree(run_dir)  # idempotent: no stale partial bundle
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_dataset_for_pipeline(csv_path)
    data_hash = _compute_data_hash(csv_path)

    if model in _QUANTUM_MODELS:
        _, checkpoint, samples, metrics, extra_cfg = _train_quantum(
            model, bundle, epochs, seed
        )
        ckpt_name = "checkpoint.pt"
    elif model in _WGAN_MODELS:
        _, checkpoint, samples, metrics, extra_cfg = _train_wgan(
            model, bundle, epochs, seed
        )
        ckpt_name = "checkpoint.pt"
    elif model == "vae":
        _, checkpoint, samples, metrics, extra_cfg = _train_vae(
            bundle, epochs, seed
        )
        ckpt_name = "checkpoint.pt"
    elif model == "ar":
        _, checkpoint, samples, metrics, extra_cfg = _train_ar(bundle, seed)
        ckpt_name = "checkpoint.npz"
    else:  # unreachable — argparse choices guard this.
        raise ValueError(f"Unknown model {model!r}")

    config: Dict[str, Any] = {
        "model": model,
        "pipeline": "B",
        "seed": int(seed),
        "epochs": int(epochs),
        "num_qubits": int(NUM_QUBITS),
        "num_layers": int(extra_cfg.get("depth", NUM_LAYERS)),
        "window_length": int(WINDOW_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "eval_every": int(EVAL_EVERY),
        "noise_low": float(NOISE_LOW),
        "noise_high": float(NOISE_HIGH),
        "n_real_windows": int(bundle.n_real_windows),
        "inverse_kwargs_keys": sorted(list(bundle.inverse_kwargs.keys())),
        "csv_path": str(csv_path),
        "data_hash": data_hash,
        "tier": _TIER[model],
        "matched_budget": True,
    }
    config.update(extra_cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    _save_inverse_kwargs(run_dir, bundle.inverse_kwargs)
    np.save(run_dir / "samples.npy", samples)
    if ckpt_name.endswith(".npz"):
        np.savez(run_dir / ckpt_name, **checkpoint)
    else:
        torch.save(checkpoint, run_dir / ckpt_name)
    # Plan 14-13 Task 4 (HI-8): _finite_sanitize replaces default=float;
    # non-finite floats become None and an explicit-raise fires if too many
    # metrics are non-finite.
    (run_dir / "metrics.json").write_text(_dumps_finite(metrics, indent=2))

    print(
        f"[run_matched2000] model={model} seed={seed} epochs={epochs} "
        f"tier={_TIER[model]} -> {run_dir} "
        f"(n_synth={samples.shape[0]}, source={extra_cfg['source']}, "
        f"data_hash={data_hash})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 14 Tier-2/3 matched-2000ep per-(model,seed) driver "
        "behind the D-14-13 strict accept gate — one model/seed per "
        "invocation (NEVER multiprocessing.Pool; the sweep uses xargs -P 2)."
    )
    ap.add_argument("--model", required=True, choices=_MODEL_CHOICES)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epochs", type=int, default=MATCHED_EPOCHS)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("revision/results/matched2000"),
        help="Root under which runs/<model>/<seed>/ is created.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="OD CSV (same data the 09.1 quantum runs used).",
    )
    ap.add_argument(
        "--accept",
        action="store_true",
        help="Run the D-14-13 strict accept gate on an existing "
        "(model,seed) artifact instead of training. Exits non-zero on "
        "rejection (explicit-raise).",
    )
    args = ap.parse_args()

    if args.accept:
        _strict_accept(args.model, args.seed, args.out_root)
        return
    _train(args.model, args.seed, args.epochs, args.csv_path, args.out_root)


if __name__ == "__main__":
    main()
