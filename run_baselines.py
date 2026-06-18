"""CLI driver: train one (model, pipeline, seed) baseline for Phase 10.

One process per invocation. Idempotent — re-running the same
``--model X --pipeline Y --seed Z`` overwrites
``runs/<model>/<pipeline>/<seed>/`` cleanly.

Mirrors ``run_ablation.py`` structure (same windowed data + the same
5-file artifact bundle) so the Wave-4 comparison table aggregates the classical
baselines against the Phase 09.1 quantum reference column on identical footing.

Three model families (Phase 10 CONTEXT decisions):
    D-10-05 — Pipelines A and B only (the C / Lambert-W path is dropped).
    D-10-07 — same windowed data + ``inverse_kwargs`` contract as run_ablation.
    D-10-08 — WGAN variants train via ``train_wgan_gp`` UNCHANGED, with the
              shared ``Critic`` and HPO constants imported from
              `core` (never hardcoded literals).
    D-10-09 — VAE trains via a local ELBO loop (no critic / GP), same
              dataloader / windowed input / seed / epoch budget as WGAN.
    D-10-13 — AR fits closed-form via lstsq; model defs are import-only.
    D-10-14 — run-dir is ``runs/<model>/<pipeline>/<seed>``; AR checkpoint is
              ``checkpoint.npz`` (the rest use ``checkpoint.pt``).
    D-10-15 — every config.yaml carries ``data_hash`` =
              sha256(real_OD bytes)[:16] recomputed from load_and_preprocess.
    D-10-20/24 — NO new metric helpers here; per-run fidelity metrics are
              computed in the Wave-4 notebook via ``core.eval``.
    D-10-22 — every model kind emits the same 5-file bundle.

Pitfall 3 (RESEARCH): VAE/AR do NOT inherit the WGAN loop's ``*0.1``
post-scaling (a quantum-output-magnitude artifact). Their ``sample()`` already
emits in the canonical ``[-1, 1]`` window space the shared ``reconstruct_od``
inverse consumes — the ``*0.1`` is applied ONLY in the WGAN branch.

Pitfall 5 (RESEARCH): one model per invocation, fresh interpreter — never
``multiprocessing.Pool`` anywhere; ``torch.manual_seed`` before construction.

Usage
-----
    python -m run_baselines --model {wgan_mlp|wgan_cnn|wgan_lstm|vae|ar} \\
        --pipeline {A|B} --seed N [--epochs M] \\
        [--out-root results/baselines] [--csv-path ./data.csv]

Each run writes these five artifacts to
``out_root/runs/<model>/<pipeline>/<seed>/``:
    config.yaml          — frozen hyperparameters + seed + epochs + data_hash
    checkpoint.pt|.npz   — model state (AR -> .npz {phi,sigma2,p})
    samples.npy          — (N_synth, WINDOW_LENGTH) in [-1, 1]
    metrics.json         — per-epoch / fit-diagnostic metrics
    inverse_kwargs.npz   — per-pipeline aux state for OD-scale reconstruction
"""
from __future__ import annotations

import argparse
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

from core import (
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
from core.data import load_and_preprocess, rolling_window
from core.models.classical import (
    WGANCNNGenerator,
    WGANLSTMGenerator,
    WGANMLPGenerator,
)
from core.models.critic import Critic
from core.models.nonadversarial import ARBaseline, VAEBaseline
from core.preprocessing import (
    forward_logreturns,
    forward_minmax_od,
)
from core.training import train_wgan_gp

# Model-kind -> generator class for the WGAN branch (D-10-08: shared Critic).
_WGAN_GENERATORS = {
    "wgan_mlp": WGANMLPGenerator,
    "wgan_cnn": WGANCNNGenerator,
    "wgan_lstm": WGANLSTMGenerator,
}
_MODEL_CHOICES = ["wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]


@dataclass
class DatasetBundle:
    """Container for per-pipeline training-data state.

    Attributes mirror ``run_ablation.DatasetBundle`` (D-10-07 identical
    windowed-data + inverse-kwargs contract).
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
    """Build the DataLoader + inverse-kwargs for one pipeline (A or B only).

    Pipelines A and B are copied VERBATIM from ``run_ablation.py`` (D-10-07);
    the C / Lambert-W branch is intentionally absent (D-10-05). Both emit
    ``(M, WINDOW_LENGTH)`` windows in [-1, 1] so every model family trains on
    the identical windowed series.
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

    else:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}; expected A or B (C dropped, D-10-05)"
        )

    ds = TensorDataset(windowed.float())
    # Identical DataLoader signature across pipelines (D-10-07 mitigation):
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

    Mirrors the training-loop generation contract (``training.py:282-283``):
    output cast to float64 and multiplied by 0.1 (the ``*0.1`` is the
    quantum/WGAN-output scaling that feeds the critic — RESEARCH Pitfall 3).
    Uses ``np.random.default_rng(seed)`` (no global ``np.random.seed`` here).
    """
    # train_wgan_gp moves the generator onto MPS/CUDA when available. Sample
    # generation runs on CPU in float64 (the canonical *0.1 sample space the
    # shared reconstruct_od inverse consumes; MPS has no float64). Moving the
    # trained generator back to CPU here keeps samples.npy bit-identical to the
    # pre-MPS-fix CPU path (Phase-10 split-mode Rule-1 fix).
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
    (identical contract to ``run_ablation._save_inverse_kwargs``).
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

    No 09.1 precedent — this field is NEW (RESEARCH Pitfall 4): it pins every
    baseline run to the exact OD tensor it trained against so the Wave-4 table
    can prove all families consumed the same data.
    """
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _train_wgan(
    model_kind: str,
    bundle: DatasetBundle,
    epochs: int,
    seed: int,
) -> tuple[
    torch.nn.Module, Dict[str, Any], np.ndarray, Dict[str, Any], Dict[str, Any]
]:
    """WGAN branch — shared Critic + ``train_wgan_gp`` UNCHANGED (D-10-08)."""
    torch.manual_seed(seed)
    generator = _WGAN_GENERATORS[model_kind]()
    critic = Critic(window_length=WINDOW_LENGTH)
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
    n_synth = 10 * bundle.n_real_windows
    samples = generate_wgan_samples(generator, n_synth, seed)
    checkpoint = {
        "gen_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }
    extra_cfg = {
        "model_kind": model_kind,
        "parameter_count": int(generator.count_params()),
        "family": "adversarial-classical",
        "n_critic": int(N_CRITIC),
        "lambda_gp": float(LAMBDA),
        "lr_critic": float(LR_CRITIC),
        "lr_generator": float(LR_GENERATOR),
        "train_protocol_notes": (
            "WGAN-GP via train_wgan_gp UNCHANGED (D-10-08); shared "
            "Critic(window_length=WINDOW_LENGTH); generator output is scaled "
            "by *0.1 (quantum/WGAN-output magnitude artifact, training.py:283) "
            "before saving samples."
        ),
    }
    return generator, checkpoint, samples, metrics, extra_cfg


def _train_vae(
    bundle: DatasetBundle,
    epochs: int,
    seed: int,
) -> tuple[
    torch.nn.Module, Dict[str, Any], np.ndarray, Dict[str, Any], Dict[str, Any]
]:
    """VAE branch — local ELBO loop, NO critic / GP / n_critic (D-10-09)."""
    torch.manual_seed(seed)
    vae = VAEBaseline()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    beta_target = 1.0
    warmup_epochs = 0  # set >0 only if posterior collapse is observed below.

    elbo_hist: list[float] = []
    recon_hist: list[float] = []
    kld_hist: list[float] = []

    for epoch in range(int(epochs)):
        if warmup_epochs > 0:
            beta = beta_target * min(1.0, (epoch + 1) / warmup_epochs)
        else:
            beta = beta_target
        ep_elbo = ep_recon = ep_kld = 0.0
        n_batches = 0
        for (x,) in bundle.dataloader:
            x = x.float()
            opt.zero_grad()
            x_hat, mu, logvar = vae(x)
            recon = torch.nn.functional.mse_loss(x_hat, x)
            kld = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - logvar.exp()
            )
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

    n_synth = 10 * bundle.n_real_windows
    gen = torch.Generator()
    gen.manual_seed(seed)
    with torch.no_grad():
        samples = vae.sample(n_synth, gen).to(torch.float64).cpu().numpy()

    # Posterior-collapse heuristic (Pitfall 6): sample std vs real-window std.
    real_std = float(bundle.windowed.std().item())
    samp_std = float(np.std(samples))
    collapse_flag = samp_std < 0.1 * real_std

    metrics = {
        "elbo": elbo_hist,
        "recon": recon_hist,
        "kld": kld_hist,
        "sample_std": samp_std,
        "real_window_std": real_std,
        "posterior_collapse_suspected": bool(collapse_flag),
    }
    checkpoint = {"vae_state_dict": vae.state_dict()}
    notes = (
        "VAE trained via a LOCAL ELBO loop (D-10-09): single "
        "Adam(lr=1e-3), NO critic / NO n_critic / NO gradient penalty; "
        "loss = MSE(x_hat,x) + beta*KLD, beta=1.0 (no KL warmup applied). "
        "sample() emits directly in the [-1,1] window space — the WGAN "
        "*0.1 scaling is a quantum-output artifact and is deliberately "
        "NOT applied to VAE samples (RESEARCH Pitfall 3)."
    )
    if collapse_flag:
        notes += (
            " WARNING: sample std << real-window std — posterior collapse "
            "suspected; consider linear KL warmup (set warmup_epochs>0)."
        )
    extra_cfg = {
        "model_kind": "vae",
        "parameter_count": int(vae.count_params()),
        "family": "non-adversarial",
        "n_critic": None,
        "lambda_gp": None,
        "lr_critic": None,
        "lr_generator": None,
        "vae_lr": 1e-3,
        "vae_beta": float(beta_target),
        "vae_kl_warmup_epochs": int(warmup_epochs),
        "train_protocol_notes": notes,
    }
    return vae, checkpoint, samples, metrics, extra_cfg


def _train_ar(
    bundle: DatasetBundle,
    seed: int,
) -> tuple[
    ARBaseline, Dict[str, Any], np.ndarray, Dict[str, Any], Dict[str, Any]
]:
    """AR branch — closed-form lstsq fit, recursive simulate (D-10-13)."""
    ar = ARBaseline(p=2)
    # Fit on the flattened windowed training series (same [-1,1] space).
    series = bundle.windowed.cpu().numpy().reshape(-1)
    ar.fit(series)
    n_synth = 10 * bundle.n_real_windows
    rng = np.random.default_rng(seed)
    samples = ar.sample(n_synth, rng).astype(np.float64)

    resid_mean = float(np.mean(samples)) if samples.size else 0.0
    metrics = {
        "sigma2": float(ar.sigma2),
        "phi": [float(v) for v in ar.phi.tolist()],
        "p": int(ar.p),
        "fit_series_len": int(series.shape[0]),
        "sample_mean": resid_mean,
        "sample_std": float(np.std(samples)),
    }
    # AR checkpoint is a .npz (D-10-14), not a torch .pt.
    checkpoint = {
        "phi": ar.phi.astype(np.float64),
        "sigma2": np.asarray(float(ar.sigma2)),
        "p": np.asarray(int(ar.p)),
    }
    extra_cfg = {
        "model_kind": "ar",
        "parameter_count": int(ar.count_params()),
        "family": "non-adversarial",
        "n_critic": None,
        "lambda_gp": None,
        "lr_critic": None,
        "lr_generator": None,
        "ar_order_p": int(ar.p),
        "train_protocol_notes": (
            "AR(p=2) fit closed-form via np.linalg.lstsq (D-10-13), "
            "recursively simulated with a 50-step burn-in. Samples are in "
            "the [-1,1] window space — the WGAN *0.1 scaling is a "
            "quantum-output artifact and is deliberately NOT applied "
            "(RESEARCH Pitfall 3)."
        ),
    }
    return ar, checkpoint, samples, metrics, extra_cfg


def main() -> None:
    """CLI entry point — one (model, pipeline, seed) per invocation."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 10 classical-baseline training driver — one "
            "(model, pipeline, seed) per invocation."
        )
    )
    ap.add_argument("--model", required=True, choices=_MODEL_CHOICES)
    ap.add_argument("--pipeline", required=True, choices=["A", "B"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/baselines"),
        help="Root under which runs/<model>/<pipeline>/<seed>/ is created.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="Path to the OD CSV file (same data the 09.1 quantum runs used).",
    )
    args = ap.parse_args()

    run_dir: Path = (
        args.out_root / "runs" / args.model / args.pipeline / str(args.seed)
    )
    # Idempotent: clean/overwrite the dir on rerun (T-10-04 — no stale
    # "complete-looking" partial bundle).
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build the per-pipeline dataset bundle (D-10-07 windowed data).
    bundle = build_dataset_for_pipeline(args.pipeline, args.csv_path)

    # 2) data_hash recomputed from the SAME csv the quantum runs used (D-10-15).
    data_hash = _compute_data_hash(args.csv_path)

    # 3) Dispatch to the model-family branch.
    if args.model in _WGAN_GENERATORS:
        _, checkpoint, samples, metrics, extra_cfg = _train_wgan(
            args.model, bundle, args.epochs, args.seed
        )
        ckpt_name = "checkpoint.pt"
    elif args.model == "vae":
        _, checkpoint, samples, metrics, extra_cfg = _train_vae(
            bundle, args.epochs, args.seed
        )
        ckpt_name = "checkpoint.pt"
    elif args.model == "ar":
        _, checkpoint, samples, metrics, extra_cfg = _train_ar(
            bundle, args.seed
        )
        ckpt_name = "checkpoint.npz"
    else:  # unreachable — argparse choices guard this.
        raise ValueError(f"Unknown model {args.model!r}")

    # 4) config.yaml — run_ablation base fields then the Phase-10 extension.
    config: Dict[str, Any] = {
        "model": args.model,
        "pipeline": args.pipeline,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "num_qubits": int(NUM_QUBITS),
        "num_layers": int(NUM_LAYERS),
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
    if ckpt_name.endswith(".npz"):
        np.savez(run_dir / ckpt_name, **checkpoint)
    else:
        torch.save(checkpoint, run_dir / ckpt_name)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )

    print(
        f"[run_baselines] model={args.model} pipeline={args.pipeline} "
        f"seed={args.seed} epochs={args.epochs} -> {run_dir} "
        f"(n_synth={samples.shape[0]}, samples.shape={samples.shape}, "
        f"data_hash={data_hash})"
    )


if __name__ == "__main__":
    main()
