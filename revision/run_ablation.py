"""CLI driver: train one (pipeline, seed) pair for the Phase 09.1 ablation.

One process per invocation. Idempotent — re-running the same
``--pipeline X --seed Y`` overwrites ``runs/X/Y/`` cleanly.

Per CONTEXT.md decisions:
    D-09.1-04 — identical training conditions across pipelines; HPO constants
                imported from ``revision.core``.
    D-09.1-07 — N_synth ≈ 10 × N_real_windows per run.
    D-09.1-08 — every run writes a YAML config alongside its artifacts.
    D-09.1-16 — reuses ``revision.core.training.train_wgan_gp``; no duplicated
                training logic.
    D-09.1-17 — pipelines are pure-function pairs from
                ``revision.core.preprocessing``.
    D-09.1-18 — uses ``np.random.default_rng(seed)`` for in-driver noise; the
                training loop at ``training.py:212`` already seeds globally.
    D-09.1-19 — type hints throughout; NO matplotlib import.

Usage
-----
    python -m revision.run_ablation --pipeline {A|B|C} --seed N \\
        [--epochs 1000] [--out-root revision/results/transform_ablation] \\
        [--csv-path ./data.csv]

Each run writes these five artifacts to ``out_root/runs/<pipeline>/<seed>/``:
    config.yaml          — frozen hyperparameters + seed + epochs
    checkpoint.pt        — params_pqc tensor + critic state_dict
    samples.npy          — (N_synth, WINDOW_LENGTH) generator output in [-1, 1]
    metrics.json         — per-epoch training metrics dict from train_wgan_gp
    inverse_kwargs.npz   — per-pipeline aux state for OD-scale reconstruction
"""
from __future__ import annotations

import argparse
import json
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
from revision.core.models.critic import Critic
from revision.core.models.quantum import QuantumGenerator
from revision.core.preprocessing import (
    forward_logreturns,
    forward_minmax_od,
)
from revision.core.training import train_wgan_gp


@dataclass
class DatasetBundle:
    """Container for per-pipeline training-data state.

    Attributes
    ----------
    dataloader
        Torch DataLoader yielding tuples ``(window,)`` of shape
        ``(BATCH_SIZE, WINDOW_LENGTH)`` in [-1, 1] (training-loop input
        contract — see ``training.py:269``).
    n_real_windows
        Count of real windows after rolling. Used to size synthetic sample
        generation at N_synth = 10 × n_real_windows (D-09.1-07).
    inverse_kwargs
        Per-pipeline auxiliary state required to invert generator output back
        to OD scale. Serialized to ``inverse_kwargs.npz``.
    pipeline_id
        Copy of the pipeline identifier ("A", "B", or "C").
    """

    dataloader: DataLoader
    n_real_windows: int
    inverse_kwargs: Dict[str, Any]
    pipeline_id: str


def build_dataset_for_pipeline(
    pipeline: str,
    csv_path: Path,
) -> DatasetBundle:
    """Build the DataLoader + inverse-kwargs for one pipeline.

    All three pipelines emit ``(M, WINDOW_LENGTH)`` windows in [-1, 1] so the
    training loop is invariant across the choice of transform (T-09.1.02-02
    mitigation: identical DataLoader call site).

    Pipeline A
        Global min-max OD → [0, 1] → linearly remapped to [-1, 1] (matches the
        v1.1 generator output range; see RESEARCH.md Pitfall 2).
    Pipeline B
        Standardized log-returns → linearly remapped to [-1, 1] using their own
        min/max. ``od_starts`` captures the real per-window OD₀ used by the
        inverse (CD-5 — anchored at real values to keep ACF shift-invariance
        unchanged; see RESEARCH.md Q3).
    Pipeline C
        v1.1 path verbatim — ``windowed_data`` from ``load_and_preprocess`` is
        already in [-1, 1] post-Lambert-W.

    Returns
    -------
    DatasetBundle
        Loader + auxiliary state for the chosen pipeline.
    """
    raw = load_and_preprocess(str(csv_path))

    if pipeline == "A":
        od = raw["OD"]
        scaled01, od_min, od_max = forward_minmax_od(od)
        # Map [0, 1] -> [-1, 1] so the generator's [-1, 1] output range matches.
        scaled_pm1 = 2.0 * scaled01 - 1.0
        windowed = rolling_window(scaled_pm1, WINDOW_LENGTH, 2)
        inverse_kwargs: Dict[str, Any] = {
            "od_min": float(od_min.item()),
            "od_max": float(od_max.item()),
        }

    elif pipeline == "B":
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
        inverse_kwargs = {
            "r_min": float(r_min.item()),
            "r_max": float(r_max.item()),
            "mu": float(mu.item()),
            "sigma": float(sigma.item()),
            "od_starts": od_starts.cpu().numpy().astype(np.float64),
        }

    elif pipeline == "C":
        windowed = raw["windowed_data"]
        inverse_kwargs = {
            "transformed_norm_log_delta": raw["transformed_norm_log_delta"]
            .cpu()
            .numpy()
            .astype(np.float64),
            "mu": float(raw["mu"].item()),
            "sigma": float(raw["sigma"].item()),
            "delta": float(raw["delta"]),
        }

    else:
        raise ValueError(f"Unknown pipeline {pipeline!r}; expected A, B, or C")

    ds = TensorDataset(windowed.float())
    # Identical DataLoader signature across pipelines (T-09.1.02-02 mitigation):
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return DatasetBundle(
        dataloader=dl,
        n_real_windows=int(windowed.shape[0]),
        inverse_kwargs=inverse_kwargs,
        pipeline_id=pipeline,
    )


def generate_samples(
    generator: torch.nn.Module,
    n: int,
    seed: int,
) -> np.ndarray:
    """Generate ``n`` synthetic windows in [-1, 1] space.

    Mirrors the training-loop generation contract (``training.py:275-286``):
    output is cast to float64 and multiplied by 0.1, matching the scaling that
    feeds the critic (T-09.1.02-01 mitigation — any deviation here breaks the
    Pipeline C parity sanity check in Task 2).

    Uses ``np.random.default_rng(seed)`` per D-09.1-18 (no global
    ``np.random.seed`` calls in this file).
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
            out = generator(noise).to(torch.float64) * 0.1
            out_parts.append(out.cpu().numpy())
            remaining -= bs
    samples = np.concatenate(out_parts, axis=0)[:n]
    return samples


def _save_inverse_kwargs(run_dir: Path, inverse_kwargs: Dict[str, Any]) -> None:
    """Persist the inverse-kwargs dict as ``inverse_kwargs.npz``.

    Scalars are stored as 0-D arrays; numpy arrays pass through unchanged.
    Loading via ``np.load(...)`` yields a dict-like object — scalars come back
    as 0-D arrays callable via ``.item()``.
    """
    payload: Dict[str, np.ndarray] = {}
    for k, v in inverse_kwargs.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        else:
            payload[k] = np.asarray(v)
    np.savez(run_dir / "inverse_kwargs.npz", **payload)


def main() -> None:
    """CLI entry point.

    Parses args, builds the per-pipeline dataset, trains, then writes the
    five-file artifact bundle (config.yaml, checkpoint.pt, samples.npy,
    metrics.json, inverse_kwargs.npz) to ``runs/<pipeline>/<seed>/``.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Phase 09.1 ablation training driver — one (pipeline, seed) "
            "per invocation."
        )
    )
    ap.add_argument("--pipeline", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("revision/results/transform_ablation"),
        help="Root directory under which runs/<pipeline>/<seed>/ is created.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="Path to the OD CSV file.",
    )
    args = ap.parse_args()

    run_dir: Path = args.out_root / "runs" / args.pipeline / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build per-pipeline dataset bundle.
    bundle = build_dataset_for_pipeline(args.pipeline, args.csv_path)

    # 2) Write the YAML run config (D-09.1-08). Hyperparameters MUST come from
    #    revision/core/__init__.py constants (D-09.1-04 identical-conditions).
    config: Dict[str, Any] = {
        "pipeline": args.pipeline,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "num_qubits": int(NUM_QUBITS),
        "num_layers": int(NUM_LAYERS),
        "window_length": int(WINDOW_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "n_critic": int(N_CRITIC),
        "lambda_gp": float(LAMBDA),
        "lr_critic": float(LR_CRITIC),
        "lr_generator": float(LR_GENERATOR),
        "noise_low": float(NOISE_LOW),
        "noise_high": float(NOISE_HIGH),
        "eval_every": int(EVAL_EVERY),
        "n_real_windows": int(bundle.n_real_windows),
        "inverse_kwargs_keys": sorted(list(bundle.inverse_kwargs.keys())),
        "csv_path": str(args.csv_path),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _save_inverse_kwargs(run_dir, bundle.inverse_kwargs)

    # 3) Build models (identical architecture across pipelines per D-09.1-04).
    #    train_wgan_gp re-seeds at training.py:211-215; the manual_seed here is
    #    belt-and-suspenders so parameter init is also deterministic.
    torch.manual_seed(args.seed)
    generator = QuantumGenerator(
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        window_length=WINDOW_LENGTH,
    )
    critic = Critic(window_length=WINDOW_LENGTH)

    # 4) Train via the existing extracted loop (D-09.1-16). No early stopper or
    #    spectral loss — D-09.1-04 demands identical settings; both default to
    #    no-op in train_wgan_gp.
    metrics: Dict[str, list] = train_wgan_gp(
        generator,
        critic,
        bundle.dataloader,
        num_epochs=int(args.epochs),
        n_critic=int(N_CRITIC),
        lambda_gp=float(LAMBDA),
        lr_critic=float(LR_CRITIC),
        lr_generator=float(LR_GENERATOR),
        seed=int(args.seed),
        eval_every=int(EVAL_EVERY),
    )

    # 5) Generate N_synth = 10 × n_real_windows samples (D-09.1-07).
    n_synth = 10 * bundle.n_real_windows
    samples_pm1 = generate_samples(generator, n_synth, args.seed)
    np.save(run_dir / "samples.npy", samples_pm1)

    # 6) Save checkpoint + metrics.
    torch.save(
        {
            "params_pqc": generator.params_pqc.detach().cpu(),
            "critic_state_dict": critic.state_dict(),
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )

    print(
        f"[run_ablation] Pipeline {args.pipeline} seed={args.seed} "
        f"epochs={args.epochs} -> {run_dir} "
        f"(n_synth={n_synth}, samples.shape={samples_pm1.shape})"
    )


if __name__ == "__main__":
    main()
