"""CLI driver: INTRO-01/02/03 instrumented training + companion-JSON assembler.

The R2-6 "what is it learning?" rebuttal. One process per ``--target`` per
invocation (Pitfall 5 — never multiprocessing); each invocation runs ONE
1000-epoch single-seed (42) training with a ``callback=`` snapshot closure
wired into ``train_wgan_gp`` (the existing dormant hook — NO training-loop
surgery, RESEARCH Pattern 2).

Four instrumented targets (D-13-08 side-by-side, D-13-07 production V1):
    quantum   -> QuantumGenerator(num_layers=4, topology="range")  (V1)
    wgan_mlp  -> WGANMLPGenerator   (Pipeline B classical, matched-param)
    wgan_cnn  -> WGANCNNGenerator
    wgan_lstm -> WGANLSTMGenerator

Snapshots fire at epoch indices ``{0,250,500,750,999}`` (eval_every=10 ->
{0,250,500,750} satisfy ``e%10==0``; index 999 satisfies ``e+1==1000``).
The terminal index 999 is relabelled to 1000 in the stored record (T-13-10
off-by-one mitigation). Each snapshot captures:
    - generated-distribution samples (INTRO-01, all targets)
    - params_pqc norm + angle list (INTRO-02, quantum only)
    - introspect() vn_entropy + purity (INTRO-03, quantum only)

The closure RE-GENERATES from the live generator (the callback receives only
a scalar metrics dict — training.py:415-431 does NOT pass samples) using the
EXACT training noise contract (training.py:304-313 / run_baselines.py:198-207
copied verbatim — T-13-08): CPU + ``rng.uniform(NOISE_LOW, NOISE_HIGH,
(NUM_QUBITS, BATCH_SIZE))`` float32 -> ``.to(float64) * 0.1``.

Persists an idempotent per-target intermediate
``revision/results/figures/_introspect_<target>.json`` (re-runnable).
``--assemble`` reads the four intermediates and emits the three companion
reproducibility JSON files plan-04 renders into figures (ROADMAP criterion 4).

Usage
-----
    python -m revision.run_introspect --target quantum  [--seed 42] \\
        [--epochs 1000] [--out-dir revision/results/figures] [--csv-path ./data.csv]
    python -m revision.run_introspect --assemble [--out-dir ...]
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch

from revision.core import (
    BATCH_SIZE,
    EVAL_EVERY,
    LAMBDA,
    LR_CRITIC,
    LR_GENERATOR,
    N_CRITIC,
    NOISE_HIGH,
    NOISE_LOW,
    NUM_QUBITS,
)
from revision.core.models.classical import (
    WGANCNNGenerator,
    WGANLSTMGenerator,
    WGANMLPGenerator,
)
from revision.core.models.critic import Critic
from revision.core.models.quantum import QuantumGenerator
from revision.core.training import train_wgan_gp
from revision.run_baselines import build_dataset_for_pipeline

# Production snapshot epochs (D-13-07/08). eval_every=10 -> {0,250,500,750}
# hit e%10==0; 999 hits e+1==num_epochs (T-13-10: relabel 999 -> 1000).
SNAP = {0, 250, 500, 750, 999}

_TARGET_CHOICES = ["quantum", "wgan_mlp", "wgan_cnn", "wgan_lstm"]
_WGAN_GENERATORS = {
    "wgan_mlp": WGANMLPGenerator,
    "wgan_cnn": WGANCNNGenerator,
    "wgan_lstm": WGANLSTMGenerator,
}


def make_snapshot_cb(
    generator: torch.nn.Module,
    *,
    snap: set[int],
    num_epochs: int,
    seed: int = 42,
) -> Tuple[Callable[[int, Dict[str, float]], None], List[Dict[str, Any]]]:
    """Build the snapshot closure + the shared records list.

    The returned ``cb(epoch, metrics)`` returns immediately unless ``epoch``
    is in ``snap``. On a matching epoch it RE-GENERATES one batch from the
    live generator under ``torch.no_grad()`` using the EXACT training noise
    contract (training.py:304-313 — T-13-08), then appends one record:

        {epoch, samples, emd, std,
         [param_norm, param_angles, vn_entropy, purity  if quantum]}

    The terminal SNAP index (``max(snap)`` == 999 in production) is relabelled
    to ``num_epochs`` (T-13-10 off-by-one mitigation: 999 -> 1000). A classical
    generator with no ``introspect()`` is hasattr-guarded -> samples only, no
    crash (the training.py callback try/except would otherwise swallow a
    closure exception and silently drop the snapshot — T-13-09).

    Caller passes the returned ``cb`` as ``callback=`` to ``train_wgan_gp``.
    """
    records: List[Dict[str, Any]] = []
    terminal = max(snap)
    # Closure-local RNG so re-generation is deterministic per run (seed 42).
    rng = np.random.default_rng(seed)

    def cb(epoch: int, metrics: Dict[str, float]) -> None:
        if epoch not in snap:
            return
        label = num_epochs if epoch == terminal else epoch
        # Re-generation: training noise contract copied verbatim
        # (training.py:304-313 / run_baselines.generate_wgan_samples) — MPS
        # has no float64 statevector path (Pitfall 6) so force CPU first.
        # train_wgan_gp keeps the generator resident on `device` (MPS here)
        # and re-generates on it EVERY epoch, so we MUST restore the original
        # device after the snapshot or the next training epoch crashes on a
        # cross-device matmul (Rule 1 fix; mirrors generate_wgan_samples which
        # only runs post-training).
        # WR-04 (Phase 13): capture orig_device BEFORE any mutation, then put
        # the .to("cpu") move itself inside the try so the finally restore
        # always runs even if the device round-trip or generation raises. The
        # device probe is its own try/except (params always exist in practice)
        # and happens before the mutating block, so a failure here cannot
        # strand the generator on CPU.
        try:
            orig_device = next(generator.parameters()).device
        except StopIteration:  # pragma: no cover - generators always have params
            orig_device = torch.device("cpu")
        try:
            gen_model = generator.to("cpu")
            with torch.no_grad():
                noise = torch.tensor(
                    rng.uniform(
                        NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, BATCH_SIZE)
                    ),
                    dtype=torch.float32,
                )
                gen = (
                    gen_model(noise).to(torch.float64) * 0.1
                ).cpu().numpy()
                record: Dict[str, Any] = {
                    "epoch": int(label),
                    "samples": gen.tolist(),
                    "emd": float(metrics["emd"]),
                    "std": float(metrics["std"]),
                }
                # INTRO-02/03: quantum only (hasattr guard — classical has no
                # introspect()). Order matters: params then introspect.
                if hasattr(gen_model, "introspect"):
                    p = gen_model.params_pqc.detach().cpu().numpy()
                    ent, pur = gen_model.introspect(noise[:, 0])
                    record["param_norm"] = float(np.linalg.norm(p))
                    record["param_angles"] = [float(v) for v in p.tolist()]
                    record["vn_entropy"] = float(ent)
                    record["purity"] = float(pur)
            records.append(record)
        finally:
            # Restore training-loop device placement (in-place, like .to()).
            generator.to(orig_device)

    return cb, records


@contextlib.contextmanager
def _force_cpu_for_quantum() -> Iterator[None]:
    """Make ``train_wgan_gp`` take its CPU path for the quantum target.

    Rule-3 blocking-issue fix (NOT architectural — ``revision/core/`` stays
    byte-untouched). ``train_wgan_gp`` (training.py:238-264) unconditionally
    moves the generator to MPS when ``torch.backends.mps.is_available()`` is
    True. A ``QuantumGenerator``'s PennyLane QNode, once its ``params_pqc``
    lands on MPS, mis-coerces the MPS tensor to a CUDA device inside
    ``pennylane.math.single_dispatch._coerce_types_torch`` and raises
    ``AssertionError: Torch not compiled with CUDA enabled`` — the quantum
    forward pass cannot run on MPS at all on this machine.

    The classical WGAN generators are pure ``torch.nn`` and run fine on MPS,
    so this guard is applied ONLY around the quantum training call. We monkey-
    patch ``torch.backends.mps.is_available -> False`` for the duration so the
    training.py device selector falls through to the float64 CPU path
    (identical to the path the frozen 09.1 quantum runs used), then restore it
    in a ``finally`` so the classical targets are unaffected.
    """
    orig = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.backends.mps.is_available = orig  # type: ignore[assignment]


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _intermediate_path(out_dir: Path, target: str) -> Path:
    return out_dir / f"_introspect_{target}.json"


def _run_one_target(
    target: str,
    seed: int,
    epochs: int,
    out_dir: Path,
    csv_path: Path,
) -> Path:
    """Run ONE instrumented training; persist the per-target intermediate.

    Idempotent: re-running ``--target X`` overwrites
    ``_introspect_<target>.json`` cleanly.
    """
    torch.manual_seed(seed)

    # Pipeline B is the headline side-by-side dataset (D-13-08).
    bundle = build_dataset_for_pipeline("B", csv_path)

    if target == "quantum":
        # Production V1 circuit (D-13-07): depth-4, range topology.
        generator: torch.nn.Module = QuantumGenerator(
            num_layers=4, topology="range"
        )
    else:
        generator = _WGAN_GENERATORS[target]()
    critic = Critic(window_length=bundle.windowed.shape[1])

    cb, records = make_snapshot_cb(
        generator, snap=SNAP, num_epochs=int(epochs), seed=seed
    )

    # Quantum needs the CPU/float64 path (PennyLane QNode breaks on MPS —
    # Rule-3 fix); classical WGANs run fine on MPS, so guard ONLY quantum.
    train_ctx = (
        _force_cpu_for_quantum()
        if target == "quantum"
        else contextlib.nullcontext()
    )
    with train_ctx:
        train_wgan_gp(
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
            early_stopper=None,
            callback=cb,
        )

    payload: Dict[str, Any] = {
        "target": target,
        "seed": int(seed),
        "epochs": int(epochs),
        "pipeline": "B",
        "snapshot_epochs": sorted(
            int(epochs) if e == max(SNAP) else e for e in SNAP
        ),
        "is_quantum": target == "quantum",
        "snapshots": records,
    }
    if target == "quantum":
        # D-13-09: pin the exact bipartition the introspect() probe used.
        bp = QuantumGenerator.INTROSPECT_BIPARTITION
        a = ",".join(str(w) for w in bp[0])
        b = ",".join(str(w) for w in bp[1])
        payload["bipartition"] = "{" + a + "}|{" + b + "}"
        payload["variant"] = "V1"
        payload["depth"] = 4
        payload["topology"] = "range"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _intermediate_path(out_dir, target)
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[run_introspect] target={target} seed={seed} epochs={epochs} -> "
        f"{out_path} ({len(records)} snapshots @ "
        f"{[r['epoch'] for r in records]})"
    )
    return out_path


def _assemble(out_dir: Path) -> None:
    """Read the 4 per-target intermediates -> 3 companion JSON files.

    ROADMAP criterion 4: reproducibility companions plan-04 renders into
    figures (NO figure rendering here). Uses the
    ``out_path.write_text(json.dumps(obj, indent=2))`` emit pattern
    (run_dualscale_fidelity.py:355-376).
    """
    inter: Dict[str, Dict[str, Any]] = {}
    for t in _TARGET_CHOICES:
        p = _intermediate_path(out_dir, t)
        if not p.exists():
            raise FileNotFoundError(
                f"missing intermediate {p} — run --target {t} first"
            )
        inter[t] = json.loads(p.read_text())

    # WR-05 (Phase 13): the quantum payload's introspection fields
    # (bipartition / param_norm / vn_entropy / purity) are only written when
    # the generator exposed introspect() (make_snapshot_cb hasattr guard). The
    # p.exists() check above only proves the file is present, not complete — a
    # partially-written or refactor-produced intermediate would otherwise die
    # with a bare KeyError naming an internal dict key. Validate the shape
    # up front with an actionable "re-run --target quantum" message.
    q_doc = inter["quantum"]
    if not q_doc.get("is_quantum") or "bipartition" not in q_doc:
        raise ValueError(
            "quantum intermediate is missing introspection fields "
            "(is_quantum/bipartition) — re-run `--target quantum` to "
            "regenerate a complete intermediate"
        )
    for s in q_doc["snapshots"]:
        missing = {
            "param_norm", "param_angles", "vn_entropy", "purity"
        } - set(s)
        if missing:
            raise ValueError(
                f"quantum snapshot epoch={s.get('epoch')} missing "
                f"{sorted(missing)} — re-run `--target quantum` to "
                f"regenerate a complete intermediate"
            )

    snap_epochs = inter["quantum"]["snapshot_epochs"]

    # INTRO-01: training_progression.json — quantum + 3 classical side-by-side
    # generated-distribution snapshots (D-13-08).
    tp_targets = []
    for t in _TARGET_CHOICES:
        snaps = inter[t]["snapshots"]
        tp_targets.append(
            {
                "target": t,
                "epochs": [s["epoch"] for s in snaps],
                "samples": [s["samples"] for s in snaps],
            }
        )
    training_progression = {
        "schema": "INTRO-01 generated-distribution snapshots (D-13-08)",
        "metadata": {
            "pipeline": "B",
            "seed": int(inter["quantum"]["seed"]),
            "snapshot_epochs": snap_epochs,
            "targets": _TARGET_CHOICES,
        },
        "targets": tp_targets,
    }

    # INTRO-02: param_trajectory.json — quantum only (PQC norms + angles).
    q = inter["quantum"]["snapshots"]
    param_trajectory = {
        "schema": "INTRO-02 PQC parameter trajectory (quantum only)",
        "metadata": {
            "variant": inter["quantum"].get("variant", "V1"),
            "depth": inter["quantum"].get("depth", 4),
            "topology": inter["quantum"].get("topology", "range"),
            "seed": int(inter["quantum"]["seed"]),
            "snapshot_epochs": snap_epochs,
        },
        "epochs": [s["epoch"] for s in q],
        "param_norm": [s["param_norm"] for s in q],
        "param_angles": [s["param_angles"] for s in q],
    }

    # INTRO-03: entanglement_trajectory.json — quantum only (vn_entropy +
    # purity); bipartition recorded verbatim (D-13-09 — T-13-11).
    entanglement_trajectory = {
        "schema": "INTRO-03 entanglement trajectory (quantum only)",
        "metadata": {
            "variant": inter["quantum"].get("variant", "V1"),
            "bipartition": inter["quantum"]["bipartition"],
            "measure": (
                "Von Neumann entanglement entropy + state purity Tr(rho^2)"
            ),
            "seed": int(inter["quantum"]["seed"]),
            "snapshot_epochs": snap_epochs,
        },
        "epochs": [s["epoch"] for s in q],
        "vn_entropy": [s["vn_entropy"] for s in q],
        "purity": [s["purity"] for s in q],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("training_progression.json", training_progression),
        ("param_trajectory.json", param_trajectory),
        ("entanglement_trajectory.json", entanglement_trajectory),
    ):
        out_path = out_dir / name
        out_path.write_text(json.dumps(obj, indent=2))
        print(f"[run_introspect] wrote {out_path}")


def main() -> None:
    """CLI entry point — one ``--target`` run, or ``--assemble``."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 13 INTRO-01/02/03 instrumented training driver + "
            "companion-JSON assembler."
        )
    )
    ap.add_argument("--target", choices=_TARGET_CHOICES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("revision/results/figures"),
        help="Directory for intermediate + companion JSON files.",
    )
    ap.add_argument(
        "--csv-path",
        type=Path,
        default=Path("./data.csv"),
        help="Path to the OD CSV (same data the frozen runs used).",
    )
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="Assemble the 3 companion JSON files from the 4 intermediates.",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    out_dir = (
        args.out_dir
        if args.out_dir.is_absolute()
        else (repo / args.out_dir)
    )
    csv_path = (
        args.csv_path
        if args.csv_path.is_absolute()
        else (repo / args.csv_path).resolve()
    )

    if args.assemble:
        _assemble(out_dir)
        return

    if args.target is None:
        ap.error("--target is required unless --assemble is given")

    _run_one_target(
        args.target, int(args.seed), int(args.epochs), out_dir, csv_path
    )


if __name__ == "__main__":
    main()
