"""CLI driver: train one (variant, seed) quantum ansatz for Phase 13 (ARCH-01).

One process per invocation. Idempotent — re-running the same
``--variant V --seed Z`` overwrites ``runs/<variant>/<seed>/`` cleanly.

Structural clone of ``revision/run_baselines.py``'s WGAN branch (same windowed
Pipeline-B data + the same 5-file artifact bundle + ``data_hash``) so the
Phase-13 ansatz comparison table aggregates V2/V3 against the reused 09.1/10
Pipeline-B quantum reference column (V1) on identical footing.

ARCH-01 ansatz variants (Phase 13 CONTEXT decisions):
    D-13-01 — V1 (depth-4, range, 75 params) is NOT a sweep target here; its
              5-seed final metrics are REUSED from
              ``revision/results/transform_ablation/runs/B/{42..46}`` (no
              recompute). Only V2/V3 are trained by this driver.
    D-13-05 — headline runs pass NO ``early_stopper`` (early-stop OFF).
    D-13-06 — ``spectral_loss_weight`` defaults to 0.0 (byte-unchanged path).
    D-10-15 — every config.yaml carries ``data_hash`` =
              sha256(real_OD bytes)[:16] recomputed from load_and_preprocess.
    D-10-20/24 — NO new metric helpers here; per-run fidelity metrics are
              computed by the Wave-2 aggregator via ``revision.core.eval``.

Variant → (num_layers, topology, parameter_count):
    V2 = (8, "range",  135)   # deeper, wrap-around CNOTs
    V3 = (4, "linear",  75)   # same depth as V1, open-chain CNOTs

Pitfall 5 (RESEARCH): one variant per invocation, fresh interpreter — never
``multiprocessing.Pool`` anywhere; ``torch.manual_seed`` before construction.

Pitfall 6 (RESEARCH): sample generation runs the trained generator on CPU in
float64 (the canonical ``*0.1`` sample space the shared reconstruct_od inverse
consumes; Apple MPS has no float64 statevector path).

Usage
-----
    python -m revision.run_ansatz --variant {V2|V3} --seed N [--epochs M] \\
        [--out-root revision/results/ansatz] [--csv-path ./data.csv]

Each run writes these five artifacts to ``out_root/runs/<variant>/<seed>/``:
    config.yaml          — frozen hyperparameters + seed + epochs + data_hash
                           + ansatz/depth/topology/parameter_count
    checkpoint.pt        — generator + critic state dicts
    samples.npy          — (N_synth, WINDOW_LENGTH) in [-1, 1]
    metrics.json         — per-epoch training metrics
    inverse_kwargs.npz   — Pipeline-B aux state for OD-scale reconstruction
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from revision.core import (
    BATCH_SIZE,
    EVAL_EVERY,
    LAMBDA,
    LR_CRITIC,
    LR_GENERATOR,
    N_CRITIC,
    NOISE_HIGH,
    NOISE_LOW,
    NUM_LAYERS,
    NUM_QUBITS,
    WINDOW_LENGTH,
)
from revision.core.data import load_and_preprocess, rolling_window
from revision.core.models.quantum import QuantumGenerator  # <- swapped (D-13-01)
from revision.core.models.critic import Critic
from revision.core.preprocessing import forward_logreturns
from revision.core.training import train_wgan_gp

# ARCH-01 ansatz variants. V1 (4, "range", 75) is reused, NOT trained here
# (D-13-01) — so it is intentionally absent from this map and the argparse
# choices. Parameter count is num_qubits + num_layers*(num_qubits*3) +
# num_qubits*2 (== QuantumGenerator.count_params); topology touches only the
# CNOT wiring, never the param count (V1/V3=75, V2=135).
_ANSATZ_VARIANTS = {
    "V2": {"num_layers": 8, "topology": "range", "parameter_count": 135},
    "V3": {"num_layers": 4, "topology": "linear", "parameter_count": 75},
}
_VARIANT_CHOICES = ["V2", "V3"]


@dataclass
class DatasetBundle:
    """Container for per-pipeline training-data state.

    Mirrors ``run_baselines.DatasetBundle`` (identical windowed-data +
    inverse-kwargs contract — D-10-07).
    """

    dataloader: DataLoader
    n_real_windows: int
    inverse_kwargs: Dict[str, Any]
    pipeline_id: str
    windowed: torch.Tensor


def build_dataset_for_pipeline(
    pipeline: str,
    csv_path: Path,
) -> DatasetBundle:
    """Build the DataLoader + inverse-kwargs for the headline pipeline (B).

    The Pipeline-B branch is copied VERBATIM from ``run_baselines.py``
    (D-10-07); Phase 13's headline is Pipeline B (log-returns). Emits
    ``(M, WINDOW_LENGTH)`` windows in [-1, 1] so the quantum generator's
    [-1, 1] output range matches.
    """
    raw = load_and_preprocess(str(csv_path))

    if pipeline == "B":
        od = raw["OD"]
        r_norm, mu, sigma = forward_logreturns(od)
        r_min = torch.min(r_norm)
        r_max = torch.max(r_norm)
        r_pm1 = 2.0 * (r_norm - r_min) / (r_max - r_min) - 1.0
        windowed = rolling_window(r_pm1, WINDOW_LENGTH, 2)
        # window i starts at log-return index 2*i; log-return r_t uses OD[t],
        # so the real per-window OD₀ for window i is OD[2*i] (CD-5).
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
    else:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}; Phase 13 headline is B only "
            f"(D-13-01)."
        )

    ds = TensorDataset(windowed.float())
    # Identical DataLoader signature as run_baselines (D-10-07 mitigation):
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return DatasetBundle(
        dataloader=dl,
        n_real_windows=int(windowed.shape[0]),
        inverse_kwargs=inverse_kwargs,
        pipeline_id=pipeline,
        windowed=windowed.detach().to(torch.float64),
    )


def generate_wgan_samples(
    generator: torch.nn.Module,
    n: int,
    seed: int,
) -> np.ndarray:
    """Generate ``n`` WGAN synthetic windows in [-1, 1] space.

    Copied VERBATIM from ``run_baselines.generate_wgan_samples`` including the
    ``.to("cpu")`` + ``*0.1`` (RESEARCH Pitfall 6 — Apple MPS has no float64
    statevector path; the trained generator is moved back to CPU so samples.npy
    stays bit-identical to the canonical *0.1 sample space the shared
    reconstruct_od inverse consumes). Uses ``np.random.default_rng(seed)``
    (no global ``np.random.seed`` here).
    """
    generator = generator.to("cpu")
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
            out = generator(noise).to(torch.float64) * 0.1
            out_parts.append(out.cpu().numpy())
            remaining -= bs
    return np.concatenate(out_parts, axis=0)[:n]


def _save_inverse_kwargs(run_dir: Path, inverse_kwargs: Dict[str, Any]) -> None:
    """Persist the inverse-kwargs dict as ``inverse_kwargs.npz``.

    Scalars are stored as 0-D arrays; numpy arrays pass through unchanged
    (identical contract to ``run_baselines._save_inverse_kwargs``).
    """
    payload: Dict[str, np.ndarray] = {}
    for k, v in inverse_kwargs.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        else:
            payload[k] = np.asarray(v)
    np.savez(run_dir / "inverse_kwargs.npz", **payload)


def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from load_and_preprocess (D-10-15).

    Pins every ansatz run to the exact OD tensor it trained against so the
    Wave-2 table can prove V1 (reused) and V2/V3 (new) consumed the same data.
    """
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


@contextlib.contextmanager
def _force_cpu_for_quantum():
    """Force ``train_wgan_gp`` onto its CPU path for the quantum statevector.

    WR-03 (Phase 13): the previous implementation mutated the global
    ``torch.backends.mps.is_available`` unconditionally at function-body level
    and NEVER restored it — fine for the one-process-per-invocation driver
    today, but a latent footgun if ``_train_wgan`` is ever imported into a
    longer-lived process (a future aggregator or a test), where MPS would be
    silently disabled for the whole interpreter with no way to undo it. This
    mirrors ``run_introspect._force_cpu_for_quantum`` exactly: monkey-patch
    for the duration of the wrapped call, then restore the original in a
    ``finally``.

    RESEARCH Pitfall 6 — PennyLane's default.qubit + torch interface
    mis-coerces non-CPU tensors (probes a CUDA device that does not exist on
    Apple silicon and raises "Torch not compiled with CUDA enabled"). The
    09.1/10 V1 quantum reference runs are CPU-only by construction, so pinning
    this quantum-training process to CPU keeps V2/V3 on the SAME numeric path
    as the reused V1 column. This is a process-local guard inside the driver —
    it does NOT modify revision/core/training.py (byte-unchanged discipline)
    and cannot affect the frozen V1 artifacts.
    """
    orig = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.backends.mps.is_available = orig  # type: ignore[assignment]


def _train_wgan(
    variant: str,
    bundle: DatasetBundle,
    epochs: int,
    seed: int,
) -> tuple[
    torch.nn.Module, Dict[str, Any], np.ndarray, Dict[str, Any], Dict[str, Any]
]:
    """Quantum WGAN branch — shared Critic + ``train_wgan_gp`` UNCHANGED.

    Constructs ``QuantumGenerator(num_layers=depth, topology=topology)`` per the
    selected ARCH-01 variant. No ``early_stopper`` (D-13-05); the default
    ``spectral_loss_weight=0.0`` (D-13-06) keeps the training path
    byte-unchanged versus the reused V1 09.1/10 runs.
    """
    spec = _ANSATZ_VARIANTS[variant]
    depth = int(spec["num_layers"])
    topology = str(spec["topology"])

    torch.manual_seed(seed)
    generator = QuantumGenerator(
        num_qubits=NUM_QUBITS,
        num_layers=depth,
        window_length=WINDOW_LENGTH,
        topology=topology,
    )
    critic = Critic(window_length=WINDOW_LENGTH)
    # WR-03 (Phase 13): RESEARCH Pitfall 6 — the quantum statevector path MUST
    # run on CPU. Wrap ONLY the train_wgan_gp call in a restoring guard (was: a
    # permanent, non-restoring global monkey-patch at function-body top level)
    # so MPS availability is reinstated even if training raises and the global
    # is never silently disabled for the rest of the interpreter.
    with _force_cpu_for_quantum():
        metrics = train_wgan_gp(
            generator,
            critic,
            bundle.dataloader,
            num_epochs=int(epochs),  # CR-01: honor --epochs (default 1000)
            n_critic=int(N_CRITIC),
            lambda_gp=float(LAMBDA),
            lr_critic=float(LR_CRITIC),
            lr_generator=float(LR_GENERATOR),
            seed=int(seed),
            eval_every=int(EVAL_EVERY),
        )
    n_synth = 10 * bundle.n_real_windows
    samples = generate_wgan_samples(generator, n_synth, seed)
    checkpoint = {
        "gen_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }
    param_count = int(generator.count_params())
    assert param_count == int(spec["parameter_count"]), (
        f"variant {variant}: count_params()={param_count} != expected "
        f"{spec['parameter_count']} (topology must not change the param count)"
    )
    extra_cfg = {
        "model_kind": "quantum",
        "ansatz": variant,
        "depth": depth,
        "topology": topology,
        "parameter_count": param_count,
        "family": "adversarial-quantum",
        "n_critic": int(N_CRITIC),
        "lambda_gp": float(LAMBDA),
        "lr_critic": float(LR_CRITIC),
        "lr_generator": float(LR_GENERATOR),
        "early_stopper": None,
        "spectral_loss_weight": 0.0,
        "train_protocol_notes": (
            f"ARCH-01 ansatz variant {variant}: QuantumGenerator("
            f"num_layers={depth}, topology={topology!r}) trained via "
            f"train_wgan_gp UNCHANGED, {int(epochs)} epochs, early-stop OFF "
            "(D-13-05), "
            "spectral_loss_weight=0.0 (D-13-06). Shared "
            "Critic(window_length=WINDOW_LENGTH); generator output is scaled "
            "by *0.1 (quantum-output magnitude artifact, training.py) before "
            "saving samples (RESEARCH Pitfall 6)."
        ),
    }
    return generator, checkpoint, samples, metrics, extra_cfg


def main() -> None:
    """CLI entry point — one (variant, seed) per invocation."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 13 ARCH-01 quantum-ansatz training driver — one "
            "(variant, seed) per invocation. V1 is reused (D-13-01), not "
            "a sweep target; only V2/V3 are trained here."
        )
    )
    ap.add_argument("--variant", required=True, choices=_VARIANT_CHOICES)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("revision/results/ansatz"),
        help="Root under which runs/<variant>/<seed>/ is created.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="Path to the OD CSV file (same data the 09.1 quantum runs used).",
    )
    args = ap.parse_args()

    run_dir: Path = (
        args.out_root / "runs" / args.variant / str(args.seed)
    )
    # Idempotent: clean/overwrite the dir on rerun (T-13-07 — no stale
    # "complete-looking" partial bundle / mixed-budget contamination).
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build the headline Pipeline-B dataset bundle (D-10-07 windowed data).
    bundle = build_dataset_for_pipeline("B", args.csv_path)

    # 2) data_hash recomputed from the SAME csv the quantum runs used (D-10-15).
    data_hash = _compute_data_hash(args.csv_path)

    # 3) Train the selected quantum ansatz variant.
    _, checkpoint, samples, metrics, extra_cfg = _train_wgan(
        args.variant, bundle, args.epochs, args.seed
    )

    # 4) config.yaml — base fields then the Phase-13 ansatz extension.
    config: Dict[str, Any] = {
        "model": "quantum",
        "pipeline": "B",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "num_qubits": int(NUM_QUBITS),
        "num_layers": int(extra_cfg["depth"]),
        "window_length": int(WINDOW_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "eval_every": int(EVAL_EVERY),
        "noise_low": float(NOISE_LOW),
        "noise_high": float(NOISE_HIGH),
        "n_real_windows": int(bundle.n_real_windows),
        "inverse_kwargs_keys": sorted(list(bundle.inverse_kwargs.keys())),
        "csv_path": str(args.csv_path),
        "data_hash": data_hash,
    }
    config.update(extra_cfg)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    # 5) Persist the 5-file bundle.
    _save_inverse_kwargs(run_dir, bundle.inverse_kwargs)
    np.save(run_dir / "samples.npy", samples)
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )

    print(
        f"[run_ansatz] variant={args.variant} "
        f"depth={extra_cfg['depth']} topology={extra_cfg['topology']} "
        f"seed={args.seed} epochs={args.epochs} -> {run_dir} "
        f"(n_synth={samples.shape[0]}, samples.shape={samples.shape}, "
        f"params={extra_cfg['parameter_count']}, data_hash={data_hash})"
    )


if __name__ == "__main__":
    main()
