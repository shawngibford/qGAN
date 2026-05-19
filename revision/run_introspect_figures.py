"""Phase 13 INTRO-01/02/03 figure renderer (plan 13-04).

Render-only. Reads the three plan-03 companion JSON files and writes six
paper-ready figure files (png + pdf). There is NO training, sampling, or
recompute in this module by design (threat T-13-12 / T-13-13, ROADMAP
criterion 4): every figure is traceable to its reproducibility JSON.

Idiom mirrors ``revision/run_dualscale_fidelity.py`` (argparse + Path
defaults + ``_find_repo_root`` + print written paths). The matplotlib half
has no in-repo analog (run_ablation.py:17 "NO matplotlib import"); panel
layout/styling is Claude's discretion per the D-13 figure discretion list.

Usage::

    python -m revision.run_introspect_figures [--figures-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Companion JSON file names (the plan-03 contract this script consumes).
TRAINING_PROGRESSION_JSON = "training_progression.json"
PARAM_TRAJECTORY_JSON = "param_trajectory.json"
ENTANGLEMENT_TRAJECTORY_JSON = "entanglement_trajectory.json"

# Stable, readable target ordering / labels for INTRO-01.
TARGET_LABELS = {
    "quantum": "Quantum (PQC)",
    "wgan_mlp": "WGAN-GP (MLP)",
    "wgan_cnn": "WGAN-GP (CNN)",
    "wgan_lstm": "WGAN-GP (LSTM)",
}
TARGET_COLORS = {
    "quantum": "#0072B2",
    "wgan_mlp": "#D55E00",
    "wgan_cnn": "#009E73",
    "wgan_lstm": "#CC79A7",
}


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _load_json(figures_dir: Path, name: str) -> dict:
    """Load a companion JSON, failing loudly if absent (T-13-13).

    This script is render-only: a missing JSON is a hard error, never a
    silent empty/partial figure.
    """
    path = figures_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"[run_introspect_figures] required companion JSON missing: "
            f"{path}. This renderer is render-only (no training/sampling); "
            f"run plan 13-03 to (re)generate the introspection JSON first."
        )
    return json.loads(path.read_text())


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> list[Path]:
    """Save a figure as both ``<stem>.png`` and ``<stem>.pdf``."""
    written: list[Path] = []
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        written.append(out)
    plt.close(fig)
    return written


def render_training_progression(data: dict, figures_dir: Path) -> list[Path]:
    """INTRO-01: generated distribution per target × snapshot epoch (D-13-08).

    Grid: one row per target (quantum + 3 classical WGAN-GP variants), one
    column per snapshot epoch {0,250,500,750,1000}; each cell is a histogram
    of that target's generated samples at that epoch — the quantum generator
    and the three classical baselines visually side-by-side.
    """
    meta = data["metadata"]
    snapshot_epochs = meta["snapshot_epochs"]
    target_order = meta["targets"]
    by_target = {t["target"]: t for t in data["targets"]}

    n_rows = len(target_order)
    n_cols = len(snapshot_epochs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.2 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    # Shared x-range per row so the quantum vs classical shapes are comparable.
    for r, tgt in enumerate(target_order):
        rec = by_target[tgt]
        all_vals = np.concatenate(
            [np.asarray(s, dtype=float).reshape(-1) for s in rec["samples"]]
        )
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        color = TARGET_COLORS.get(tgt, "#444444")
        for c, epoch in enumerate(rec["epochs"]):
            ax = axes[r][c]
            vals = np.asarray(rec["samples"][c], dtype=float).reshape(-1)
            ax.hist(
                vals,
                bins=bins,
                density=True,
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.3,
            )
            if r == 0:
                ax.set_title(f"epoch {epoch}", fontsize=10)
            if c == 0:
                ax.set_ylabel(
                    TARGET_LABELS.get(tgt, tgt),
                    fontsize=9,
                    rotation=90,
                    labelpad=8,
                )
            if r == n_rows - 1:
                ax.set_xlabel("generated value", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "INTRO-01  Generated distribution across training "
        f"(pipeline={meta['pipeline']}, seed={meta['seed']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, figures_dir, "training_progression")


def render_param_trajectory(data: dict, figures_dir: Path) -> list[Path]:
    """INTRO-02: PQC parameter L2-norm curve + per-epoch angle histograms."""
    meta = data["metadata"]
    epochs = data["epochs"]
    param_norm = data["param_norm"]
    param_angles = data["param_angles"]

    fig, (ax_norm, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) parameter L2-norm vs epoch
    ax_norm.plot(
        epochs,
        param_norm,
        marker="o",
        color="#0072B2",
        linewidth=2,
    )
    ax_norm.set_xlabel("epoch")
    ax_norm.set_ylabel(r"PQC parameter $\ell_2$ norm")
    ax_norm.set_title("(a) Parameter norm vs epoch")
    ax_norm.grid(True, alpha=0.3)

    # (b) angle histograms — small-multiples overlaid, one per snapshot epoch
    cmap = plt.get_cmap("viridis")
    n = len(epochs)
    angle_lo = min(min(a) for a in param_angles)
    angle_hi = max(max(a) for a in param_angles)
    bins = np.linspace(angle_lo, angle_hi, 30)
    for i, (epoch, angles) in enumerate(zip(epochs, param_angles)):
        ax_hist.hist(
            np.asarray(angles, dtype=float),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=cmap(i / max(n - 1, 1)),
            label=f"epoch {epoch}",
        )
    ax_hist.set_xlabel("parameter angle (rad)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title(
        f"(b) Angle distribution ({len(param_angles[0])} params)"
    )
    ax_hist.legend(fontsize=8, frameon=False)
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        "INTRO-02  PQC parameter trajectory  "
        f"(variant={meta['variant']}, depth={meta['depth']}, "
        f"topology={meta['topology']}, seed={meta['seed']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures_dir, "param_trajectory")


def render_entanglement_trajectory(
    data: dict, figures_dir: Path
) -> list[Path]:
    """INTRO-03: Von Neumann entanglement entropy + state purity (D-13-09).

    Annotates the bipartition string verbatim from the JSON metadata and the
    [0, ln 4] entropy / [0.25, 1] purity reference bounds.
    """
    meta = data["metadata"]
    epochs = data["epochs"]
    vn_entropy = data["vn_entropy"]
    purity = data["purity"]
    bipartition = meta["bipartition"]

    # 5-qubit state, bipartition {0,1}|{2,3,4}: smaller subsystem = 2 qubits,
    # so max Von Neumann entropy = ln(4); minimum purity = 1/4.
    max_entropy = np.log(4.0)
    min_purity = 0.25

    fig, (ax_e, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_e.plot(
        epochs, vn_entropy, marker="o", color="#0072B2", linewidth=2
    )
    ax_e.axhline(
        max_entropy,
        color="#888888",
        linestyle="--",
        linewidth=1,
        label=r"max $= \ln 4$",
    )
    ax_e.set_ylim(0, max_entropy * 1.08)
    ax_e.set_xlabel("epoch")
    ax_e.set_ylabel("Von Neumann entanglement entropy (nats)")
    ax_e.set_title("(a) Entanglement entropy vs epoch")
    ax_e.legend(fontsize=9, frameon=False)
    ax_e.grid(True, alpha=0.3)

    ax_p.plot(
        epochs, purity, marker="s", color="#D55E00", linewidth=2
    )
    ax_p.axhline(
        min_purity,
        color="#888888",
        linestyle="--",
        linewidth=1,
        label="min $= 1/4$ (max mixed)",
    )
    ax_p.axhline(
        1.0,
        color="#bbbbbb",
        linestyle=":",
        linewidth=1,
        label="max $= 1$ (pure)",
    )
    ax_p.set_ylim(0, 1.05)
    ax_p.set_xlabel("epoch")
    ax_p.set_ylabel(r"reduced-state purity $\mathrm{Tr}(\rho^2)$")
    ax_p.set_title("(b) State purity vs epoch")
    ax_p.legend(fontsize=9, frameon=False)
    ax_p.grid(True, alpha=0.3)

    fig.suptitle(
        "INTRO-03  Entanglement trajectory  "
        f"bipartition {bipartition}  "
        f"(variant={meta['variant']}, seed={meta['seed']})",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        f"Measure: {meta['measure']}",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, figures_dir, "entanglement_trajectory")


def main() -> None:
    """CLI entry point — render the 3 INTRO figures (png + pdf each)."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 13 INTRO figure renderer (plan 13-04). Render-only: "
            "reads the plan-03 companion JSON, never trains/samples."
        )
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("revision/results/figures"),
        help="Directory holding the companion JSON / receiving the figures.",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    figures_dir = args.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = (repo / figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    tp = _load_json(figures_dir, TRAINING_PROGRESSION_JSON)
    pt = _load_json(figures_dir, PARAM_TRAJECTORY_JSON)
    et = _load_json(figures_dir, ENTANGLEMENT_TRAJECTORY_JSON)

    written: list[Path] = []
    written += render_training_progression(tp, figures_dir)
    written += render_param_trajectory(pt, figures_dir)
    written += render_entanglement_trajectory(et, figures_dir)

    print("[run_introspect_figures] wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
