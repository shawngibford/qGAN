"""Phase 14 PAPER-03 circuit-diagram emitter (plan 14-09).

Render-only. Builds the missing architecture visualizations for ALL 5
production quantum circuits used in the manuscript:

    default_75   — the byte-frozen v1.0/v1.1 baseline circuit
                   (5 qubits x 4 SEL layers + final RX+RY = 75 params)
    iqp_sel_55   — the recovered canonical paper circuit
                   (frozen checkpoint epoch 1969 — D-14-01;
                    5 qubits x 3 SEL layers + final RX-only = 55 params)
    V1, V2, V3   — the matched-budget ansatz variants from
                   ``revision/run_matched2000.py:118-122``
                   (V1: range/4L/75p, V2: range/8L/135p, V3: linear/4L/75p)

Two outputs:

    1. Four config-lock JSONs under ``revision/results/``
       (``v1_config_lock.json``, ``v2_config_lock.json``,
       ``v3_config_lock.json``, ``default_75_config_lock.json``) mirroring
       the schema of ``canonical_config_lock.json`` so every numeric literal
       in the circuit-atlas doc gates through the existing
       ``revision/verify_number_provenance.py`` (rglob over
       ``revision/results/*.json`` — zero verifier edit).

    2. 15 figure artifacts under ``revision/results/figures/circuits/``:
       ``{default_75,iqp_sel_55,V1,V2,V3}.{png,pdf,json}`` — every PNG drawn
       via ``qml.draw_mpl(qnode, style="pennylane")`` (never a bespoke
       matplotlib gate drawing), every companion JSON records
       (figure / circuit_id / ansatz_name / source_config_lock_path /
       n_params / depth / topology / num_qubits / renderer /
       generation_timestamp) so each figure is independently re-derivable.

Render-only contract — NO retraining, NO sampling, NO checkpoint reload:
``revision.core.models.quantum.QuantumGenerator`` is imported only to
reconstruct the QNode tape for drawing (no model fit / sample /
checkpoint reload paths whatsoever). The
``QuantumGenerator(circuit_id=...)`` public constructor path is the ONLY
mechanism used to build the QNode; no monkey-patch / subclass override.
``revision/core/`` byte-freeze (D-14-22) is preserved across the whole
plan (``git diff --stat revision/core/`` stays empty).

Idiom mirrors ``revision/run_figure_suite.py`` (the 14-04 canonical pattern):
headless ``matplotlib.use("Agg")`` BEFORE pyplot, ``_bootstrap_repo_on_path``
to support both ``-m`` and bare-script invocation, ``_find_repo_root``,
``_require``/``_load_json`` loud-fail (``FileNotFoundError`` with a
render-only message — never a silent partial figure), ``_save`` dual
PNG+PDF (dpi=150, bbox_inches="tight") + same-stem companion JSON,
``argparse --figures-dir`` default ``revision/results/figures/circuits`` +
print-every-written-path.

Param-count consistency is enforced by an explicit ``raise AssertionError``
(python -O safe — ``run_multiseed_rollup.py:86-92`` idiom): each computed
param_count from the source dict / core constants MUST equal the
param_count it writes (V1=5+4*15+10=75, V2=5+8*15+10=135, V3=5+4*15+10=75,
default_75=5+4*15+10=75); loud-fail lists EVERY offending variant in a
single raise.

Usage::

    python -m revision.run_circuit_diagrams             # locks + diagrams
    python revision/run_circuit_diagrams.py             # bare-script form
    python revision/run_circuit_diagrams.py --config-locks-only
    python revision/run_circuit_diagrams.py --diagrams-only
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: F401,E402  (kept for parity with run_figure_suite)
import torch  # noqa: E402


def _bootstrap_repo_on_path() -> Path:
    """Ensure the repo root is importable when run as a bare script.

    ``python revision/run_circuit_diagrams.py`` does not put the repo root
    on ``sys.path`` (only the script's own dir). Walk up to the dir holding
    ``revision/core/preprocessing.py`` and prepend it (verbatim with
    ``revision/run_figure_suite.py:54-69`` so both ``-m`` and bare-script
    invocation work — the plan's verify command uses the latter).
    """
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_bootstrap_repo_on_path()

# Imports below the bootstrap are intentional (peer driver mirrors this).
import pennylane as qml  # noqa: E402

from revision.core import NUM_LAYERS, NUM_QUBITS, WINDOW_LENGTH  # noqa: E402
# Pure-aggregator import (D-14-22 safety; verified per 14-03 SUMMARY
# deviation #1): ``revision.run_matched2000``'s top-level imports are
# stdlib + dataclass + pathlib + typing only. The torch / numpy / pennylane
# imports live INSIDE functions, so grabbing the module-level
# ``_QUANTUM_ANSATZ`` dict does NOT trigger any model-fit /
# sampling / training-loop / checkpoint-reload path.
from revision.run_matched2000 import _QUANTUM_ANSATZ  # noqa: E402


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
RESULTS_REL = Path("revision/results")
FIGURES_REL = Path("revision/results/figures/circuits")

# Mirrors the schema of revision/results/canonical_config_lock.json
# (Phase 14 D-14-05/06/07) with a new schema-string for the matched-budget
# ansatz locks added by plan 14-09.
SCHEMA_ANSATZ_LOCK = "ansatz-config-lock v1 (Phase 14 D-14-04, plan 14-09)"
SCHEMA_DEFAULT_LOCK = "default-config-lock v1 (Phase 14 D-14-04, plan 14-09)"

# Source-of-truth line numbers in revision/run_matched2000.py for V1/V2/V3
# (the _QUANTUM_ANSATZ dict literally lives at lines 117-124).
_QUANTUM_ANSATZ_LINES = {
    "V1": "revision/run_matched2000.py:118",
    "V2": "revision/run_matched2000.py:120",
    "V3": "revision/run_matched2000.py:122",
}

# default_75 derives from core constants + the default_75 branch of
# revision/core/models/quantum.py — recorded verbatim in the source_path
# field so the regression-guard assertion has a deterministic origin.
_DEFAULT_75_SOURCE_PATH = (
    "revision/core/__init__.py (NUM_QUBITS=5, NUM_LAYERS=4) + "
    "revision/core/models/quantum.py (default_75 branch)"
)

# Render order matters for output stability (use this exact order in
# render_diagrams and any downstream verification).
_RENDER_ORDER = ("default_75", "iqp_sel_55", "V1", "V2", "V3")


# ---------------------------------------------------------------------------
# Repo-root + load-only helpers (mirror run_figure_suite.py:144-174)
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    # Fall back to the script-relative bootstrap (same anchor file).
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _require(path: Path, what: str) -> Path:
    """Hard-fail if a required render-only artifact is absent.

    This emitter is render-only (no training/sampling/recompute). A
    missing config-lock JSON is a hard ``FileNotFoundError`` with an
    explicit render-only message — never a silent partial render.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[run_circuit_diagrams] required render-only artifact missing: "
            f"{path} ({what}). This emitter performs NO training/sampling/"
            f"recompute; (re)run ``--config-locks-only`` to emit the lock "
            f"JSONs before rendering the diagrams."
        )
    return path


def _load_json(path: Path, what: str) -> dict:
    """Load a companion / source JSON, failing loudly if absent."""
    _require(path, what)
    return json.loads(path.read_text())


def _save(fig: plt.Figure, figures_dir: Path, stem: str,
          companion: dict) -> list[Path]:
    """Save ``<stem>.png`` + ``<stem>.pdf`` + ``<stem>.json``.

    Mirrors ``run_figure_suite.py:177-193`` (dpi=150,
    bbox_inches="tight", plt.close). The companion JSON records
    everything a future reader needs to independently re-derive the
    figure (source_config_lock_path / n_params / depth / topology /
    num_qubits / renderer / generation_timestamp).
    """
    written: list[Path] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        written.append(out)
    plt.close(fig)
    jpath = figures_dir / f"{stem}.json"
    jpath.write_text(json.dumps(companion, indent=2, sort_keys=True))
    written.append(jpath)
    return written


# ---------------------------------------------------------------------------
# Step 1: build config-lock JSONs (V1/V2/V3 + default_75)
# ---------------------------------------------------------------------------
def _final_rot_factor(circuit_id: str) -> int:
    """Mirror ``revision/core/models/quantum.py:104`` exactly.

    ``default_75`` carries final RX AND RY per qubit (factor 2);
    ``iqp_sel_55`` carries final RX-only (factor 1). The matched-budget
    V1/V2/V3 entries all use ``circuit_id="default_75"`` so they share
    the factor-2 final rotation.
    """
    return 2 if circuit_id == "default_75" else 1


def _expected_param_count(num_qubits: int, num_layers: int,
                          circuit_id: str) -> int:
    """Mirror ``revision/core/models/quantum.py:105-109`` exactly.

    num_params = num_qubits + num_layers * (num_qubits * 3)
                          + num_qubits * _final_rot_factor(circuit_id)
    """
    return (
        num_qubits
        + num_layers * (num_qubits * 3)
        + num_qubits * _final_rot_factor(circuit_id)
    )


def _gate_layout_breakdown(num_qubits: int, num_layers: int,
                           final_rot: str, total: int) -> str:
    """Human-readable summand string carried in the lock JSON.

    Every summand (num_qubits, num_layers, num_qubits*3, total) appears
    verbatim so ``verify_number_provenance.py``'s substring resolver
    finds the literals quoted in ``circuit_atlas.md``.
    """
    sel_per_layer = num_qubits * 3  # SEL = Rot(phi, theta, omega) per qubit
    if final_rot == "RX_plus_RY":
        final_count = num_qubits * 2
        final_label = "RX+RY"
    elif final_rot == "RX_only":
        final_count = num_qubits
        final_label = "RX"
    else:  # pragma: no cover — guarded upstream
        raise AssertionError(f"unknown final_rot {final_rot!r}")
    return (
        f"IQP encoding ({num_qubits}) + {num_layers}*SEL layers "
        f"({sel_per_layer} each) + final {final_label} ({final_count}) = {total}"
    )


def _build_lock_object(
    *,
    ansatz_name: str,
    num_qubits: int,
    num_layers: int,
    topology: str,
    circuit_id: str,
    final_rot: str,
    source_path: str,
    schema: str,
) -> dict:
    """Construct one config-lock dict mirroring canonical_config_lock.json.

    Schema mirrored from ``revision/results/canonical_config_lock.json``:
    decomposition.{num_qubits, num_layers, param_count,
                   gate_layout.{hadamard_init, iqp_encoding_params_per_qubit,
                                sel_rot_params_per_qubit_per_layer,
                                entangler, final_rotation}},
    top-level param_count (canonical mirrors at both levels),
    native_pipeline="B", locked_circuit_id, schema. Extends the canonical
    schema with three NEW fields for this plan: ``source_path`` (origin
    of the numbers), ``ansatz_name`` (V1/V2/V3/default_75 label), and
    ``gate_layout_breakdown`` (human-readable summand string surfaced in
    the circuit-atlas doc).
    """
    param_count = _expected_param_count(num_qubits, num_layers, circuit_id)
    return {
        "schema": schema,
        "locked_circuit_id": circuit_id,
        "ansatz_name": ansatz_name,
        "decomposition": {
            "num_qubits": num_qubits,
            "num_layers": num_layers,
            "param_count": param_count,
            "gate_layout": {
                "hadamard_init": True,
                "iqp_encoding_params_per_qubit": 1,
                "sel_rot_params_per_qubit_per_layer": 3,
                "entangler": topology,
                "final_rotation": final_rot,
            },
        },
        "param_count": param_count,
        "topology": topology,
        "native_pipeline": "B",
        "source_path": source_path,
        "gate_layout_breakdown": _gate_layout_breakdown(
            num_qubits, num_layers, final_rot, param_count
        ),
    }


def build_config_locks(repo: Path) -> list[Path]:
    """Emit V1/V2/V3 + default_75 config-lock JSONs under revision/results/.

    Loud-fails with an explicit ``raise AssertionError`` (python -O safe,
    ``run_multiseed_rollup.py:86-92`` idiom) on ANY per-variant
    computed-vs-written param_count mismatch — collects every mismatch
    into a list and raises after the loop so the operator sees ALL
    offending variants in one shot, not just the first.

    Returns the sorted list of written paths.
    """
    results_dir = repo / RESULTS_REL
    results_dir.mkdir(parents=True, exist_ok=True)

    # V1/V2/V3 from the source-of-truth dict in run_matched2000.py
    # (pure-aggregator import — verified safe per 14-03; no training import
    # triggered by reading the module-level dict).
    plan = []  # list of (ansatz_name, lock_object, out_path, expected, written)
    mismatches: list[str] = []

    for name in ("V1", "V2", "V3"):
        src = _QUANTUM_ANSATZ[name]
        num_layers = int(src["num_layers"])
        topology = str(src["topology"])
        circuit_id = str(src["circuit_id"])
        # parameter_count in the source dict is the operator-declared value;
        # we compute from the formula and assert equality so any future
        # source-dict edit that drifts from the core formula loud-fails.
        source_param_count = int(src["parameter_count"])

        lock = _build_lock_object(
            ansatz_name=name,
            num_qubits=NUM_QUBITS,
            num_layers=num_layers,
            topology=topology,
            circuit_id=circuit_id,
            # All three matched-budget variants use circuit_id="default_75",
            # so they carry RX+RY final rotation (factor 2).
            final_rot="RX_plus_RY",
            source_path=_QUANTUM_ANSATZ_LINES[name],
            schema=SCHEMA_ANSATZ_LOCK,
        )
        computed = lock["param_count"]

        if computed != source_param_count:
            mismatches.append(
                f"{name}: computed param_count {computed} != source-dict "
                f"parameter_count {source_param_count} "
                f"(formula: {NUM_QUBITS} + {num_layers}*({NUM_QUBITS}*3) + "
                f"{NUM_QUBITS}*{_final_rot_factor(circuit_id)} = {computed})"
            )

        out = results_dir / f"{name.lower()}_config_lock.json"
        plan.append((name, lock, out, source_param_count, computed))

    # default_75 lock derived from core/__init__.py constants
    # (NUM_QUBITS=5, NUM_LAYERS=4) + the default_75 branch of quantum.py
    # (final RX+RY per qubit; range topology is the v1.0/v1.1 default).
    default_lock = _build_lock_object(
        ansatz_name="default_75",
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        topology="range",
        circuit_id="default_75",
        final_rot="RX_plus_RY",
        source_path=_DEFAULT_75_SOURCE_PATH,
        schema=SCHEMA_DEFAULT_LOCK,
    )
    default_computed = default_lock["param_count"]
    # Regression guard against future formula edits — if the core
    # constants / quantum.py formula ever drifts from a 75-param
    # default_75, this loud-fails before any lock is written.
    if default_computed != 75:
        mismatches.append(
            f"default_75: computed param_count {default_computed} != "
            f"expected 75 (NUM_QUBITS={NUM_QUBITS}, NUM_LAYERS={NUM_LAYERS})"
        )
    plan.append((
        "default_75",
        default_lock,
        results_dir / "default_75_config_lock.json",
        75,
        default_computed,
    ))

    if mismatches:
        # Explicit raise (NOT bare assert) — python -O safe.
        raise AssertionError(
            "[run_circuit_diagrams] param_count consistency gate FAILED for "
            f"{len(mismatches)} variant(s) — refusing to emit any lock JSON. "
            "Offending variants: " + " ; ".join(mismatches)
        )

    written: list[Path] = []
    for name, lock, out, _src_pc, _comp in plan:
        out.write_text(json.dumps(lock, indent=2, sort_keys=True))
        written.append(out)

    return written


# ---------------------------------------------------------------------------
# Step 2: render 5 circuit diagrams via qml.draw_mpl (render-only)
# ---------------------------------------------------------------------------
# Variant table — order matters for output stability. Path is relative to
# the repo root; render_diagrams resolves it via repo / lock_path.
_VARIANT_LOCK_PATHS: dict[str, str] = {
    "default_75": "revision/results/default_75_config_lock.json",
    "iqp_sel_55": "revision/results/canonical_config_lock.json",
    "V1":         "revision/results/v1_config_lock.json",
    "V2":         "revision/results/v2_config_lock.json",
    "V3":         "revision/results/v3_config_lock.json",
}


def _extract_lock_fields(lock: dict, name: str) -> dict:
    """Pull the rendering-relevant fields from a config-lock JSON.

    Reads from both the top-level and ``decomposition`` sub-object;
    when both spell the same field, asserts they agree. NEVER hand-picks
    or recomputes a value — every number comes from the lock JSON.
    """
    decomp = lock.get("decomposition", {})
    num_qubits = int(decomp["num_qubits"])
    num_layers = int(decomp["num_layers"])
    locked_circuit_id = str(lock["locked_circuit_id"])

    # param_count is mirrored at top level + decomposition in the canonical
    # schema; when both exist, they MUST agree.
    decomp_pc = int(decomp["param_count"])
    top_pc = int(lock.get("param_count", decomp_pc))
    if decomp_pc != top_pc:
        raise AssertionError(
            f"[run_circuit_diagrams] {name} config lock: decomposition."
            f"param_count ({decomp_pc}) != top-level param_count ({top_pc})"
        )
    param_count = top_pc

    # Topology: canonical_config_lock.json carries the entangler under
    # decomposition.gate_layout.entangler; the new locks carry both that
    # AND a top-level ``topology`` field — read from whichever exists.
    topology = None
    if "topology" in lock:
        topology = str(lock["topology"])
    if topology is None:
        topology = str(decomp.get("gate_layout", {}).get("entangler", "range"))

    return {
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "topology": topology,
        "locked_circuit_id": locked_circuit_id,
        "param_count": param_count,
    }


def _draw_one(
    *,
    name: str,
    repo: Path,
    figures_dir: Path,
    lock_rel: str,
) -> list[Path]:
    """Render a single circuit diagram via qml.draw_mpl and emit triple.

    Reads num_qubits / num_layers / topology / locked_circuit_id /
    param_count EXCLUSIVELY from the loaded config-lock JSON. Hard-asserts
    ``model.num_params == lock.param_count`` before drawing so the
    rendered tape is the exact tape the lock describes (T-14-19).
    """
    lock_path = repo / lock_rel
    lock = _load_json(lock_path, f"{name} config lock")
    spec = _extract_lock_fields(lock, name)

    num_qubits = spec["num_qubits"]
    num_layers = spec["num_layers"]
    topology = spec["topology"]
    locked_circuit_id = spec["locked_circuit_id"]
    param_count = spec["param_count"]

    # Render-only contract: build the QNode AND execute the qml.draw_mpl
    # tape walk under torch.no_grad() so neither the constructor's PQC
    # init nor the 5 forward passes draw_mpl runs through default.qubit
    # leak into PyTorch's autograd graph. Mirrors QuantumGenerator.introspect
    # at revision/core/models/quantum.py:344. The placeholder param
    # tensor's numerical values do NOT affect the drawn tape structure —
    # only the (num_qubits, num_layers, topology, circuit_id) tuple does —
    # so any deterministic tensor of the locked shape is acceptable.
    with torch.no_grad():
        # Defer the import to inside the render path so the lock-only
        # step never pulls in the heavy revision.core.models.quantum
        # module (which transitively imports pennylane + torch — fine
        # here since we are about to use them anyway).
        from revision.core.models.quantum import QuantumGenerator  # noqa: E402

        model = QuantumGenerator(
            num_qubits=num_qubits,
            num_layers=num_layers,
            window_length=2 * num_qubits,
            topology=topology,
            circuit_id=locked_circuit_id,
        )

        if model.num_params != param_count:
            raise AssertionError(
                f"[run_circuit_diagrams] {name}: QuantumGenerator built "
                f"with circuit_id={locked_circuit_id!r}, num_qubits="
                f"{num_qubits}, num_layers={num_layers}, topology="
                f"{topology!r} consumed model.num_params={model.num_params}, "
                f"but config-lock declares param_count={param_count}. "
                f"Refusing to render — fix the lock or the variant table."
            )

        noise = torch.zeros(num_qubits)
        # model.params_pqc is the constructor-initialized random tensor;
        # use it as-is (param values do not affect tape topology).
        # detach()+clone() is extra-paranoid about no autograd leaks.
        params = model.params_pqc.detach().clone()

        # qml.draw_mpl returns a callable; calling it with the QNode's
        # expected args (noise, params) returns (fig, ax). style="pennylane"
        # is the recommended publication style. Never a bespoke
        # matplotlib gate drawing (no rectangle / circle DSL).
        fig, _ax = qml.draw_mpl(model.qnode, style="pennylane")(noise, params)

        suptitle = (
            f"{name} - {num_qubits} qubits x {num_layers} layers "
            f"x {topology} topology - {param_count} parameters"
        )
        if name == "iqp_sel_55":
            # Continuity with 14-01 framing. D-14-10 (headline vs
            # reproduction conflation) does NOT apply here — this is
            # an architecture diagram, no generation numbers.
            suptitle += (
                " (canonical paper circuit, frozen checkpoint epoch 1969)"
            )
        fig.suptitle(suptitle, fontsize=11)

    companion = {
        "figure": "circuit_diagram",
        "circuit_id": locked_circuit_id,
        "ansatz_name": name,
        "source_config_lock_path": str(lock_rel),
        "n_params": param_count,
        "depth": num_layers,
        "topology": topology,
        "num_qubits": num_qubits,
        # Plan 14-13 Task 4 (HIGH-2 / PROV-HIGH-2): record the audited
        # dataset hash so data_hash corpus consistency holds for figure
        # companions too.
        "data_hash": "91e447d4624e25b3",
        "render_only": True,
        "renderer": "qml.draw_mpl(style=\"pennylane\")",
        "generation_timestamp": (
            datetime.datetime.utcnow().isoformat() + "Z"
        ),
    }
    return _save(fig, figures_dir, name, companion)


def render_diagrams(repo: Path, figures_dir: Path) -> list[Path]:
    """Render all 5 circuit diagrams in the canonical _RENDER_ORDER.

    Each variant produces a (PNG, PDF, companion JSON) triple under
    ``figures_dir``. Loud-fails (``FileNotFoundError``) on any missing
    config-lock JSON via ``_load_json`` — never a silent partial render.
    """
    written: list[Path] = []
    for name in _RENDER_ORDER:
        lock_rel = _VARIANT_LOCK_PATHS[name]
        written += _draw_one(
            name=name,
            repo=repo,
            figures_dir=figures_dir,
            lock_rel=lock_rel,
        )
    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Emit config locks + render circuit diagrams (PAPER-03)."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 14 PAPER-03 circuit-diagram emitter (plan 14-09). "
            "Render-only: builds V1/V2/V3 + default_75 config-lock JSONs "
            "from the source dict + core constants and renders all 5 "
            "production circuits via qml.draw_mpl. NEVER trains, samples, "
            "or reloads the frozen checkpoint."
        )
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_REL,
        help="Directory to receive the {png,pdf,json} circuit-diagram "
        "triples (default: revision/results/figures/circuits).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--config-locks-only",
        action="store_true",
        help="Only emit the V1/V2/V3 + default_75 config-lock JSONs "
        "(skip the qml.draw_mpl render step).",
    )
    mode.add_argument(
        "--diagrams-only",
        action="store_true",
        help="Only render the 5 circuit diagrams (assumes locks already "
        "exist; loud-fails with FileNotFoundError if any are absent).",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    figures_dir = args.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = (repo / figures_dir).resolve()

    written: list[Path] = []
    if not args.diagrams_only:
        written += build_config_locks(repo)
    if not args.config_locks_only:
        figures_dir.mkdir(parents=True, exist_ok=True)
        written += render_diagrams(repo, figures_dir)

    print("[run_circuit_diagrams] wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
