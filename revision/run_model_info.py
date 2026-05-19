"""Phase 14 PAPER-08 driver — the unified per-model info aggregator.

This is a **pure consumer / pure aggregator** (D-14-16): it reads ONLY the
accepted 2000ep result artifacts —

  * `revision/results/headline_canonical.json`
        (the frozen-checkpoint headline, source=frozen_checkpoint_epoch_1969),
  * `revision/results/matched2000/runs/<model>/<seed>/config.yaml`
        (the strict-gate-accepted 2000ep sweep configs, all 9 models x 5 seeds),
  * `revision/results/matched2000/sweep_status.json`
        (resumable sweep state — per-run wall-time provenance),
  * `revision/results/canonical_recovery.json`
        (the frozen-checkpoint optimizer LR/betas breadcrumbs, D-14-01),

and emits ONE unified `revision/results/model_info.json` where every model is
a single `models[]` record (D-14-15): the 55-param IQP:SEL frozen-checkpoint
headline and its 2000ep reproduction are SEPARATE rows (D-14-10), the V1/V2/V3
ansatz variants, the three classical WGAN-GP baselines (mlp/cnn/lstm), and the
two non-adversarial baselines (VAE, AR). Each record carries the D-14-15
columns (params, epochs=2000, early-stop state, optimizer/LR/betas, batch,
N_CRITIC, lambda, seeds {42..46}, window config, device/dtype, data_hash,
wall-time).

It ALSO writes `revision/docs/reconciliation_note.md` recording, per model /
metric, the 1000ep -> 2000ep delta (old value from the frozen Phase-10
`baseline_comparison.json` 1000ep artifact, new value from the accepted 2000ep
`matched2000` metrics) — the authoritative record for any manuscript number
that changed (D-14-13, Pitfall 5).

This module performs ZERO deep-learning-framework imports and ZERO core-package
imports — it is a pure JSON/YAML consumer (D-14-16): the PyTorch tensor library,
the PennyLane quantum library, and the project's shared model package are
deliberately never imported here. The cross-artifact `data_hash` gate uses an
explicit `raise AssertionError` (run_multiseed_rollup.py:86-92 idiom) so
`python -O` cannot silently strip it.

Pattern source: `revision/run_multiseed_rollup.py` — repo-root resolver
(42-59), cross-artifact `data_hash` explicit-raise gate (85-92), output write
idiom (176-187). Anti-Pattern (forbidden): re-deriving `data_hash`, recomputing
any metric, or hand-typing a numeric literal into a doc — every number in the
regenerated docs is pulled from `model_info.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (copied verbatim from run_multiseed_rollup.py:42-59 — the
# driver may run from a worktree / arbitrary cwd; results paths are repo-root
# anchored). This driver is a pure aggregator: it performs zero deep-learning
# framework or shared-package imports, so the sys.path.insert exists only to
# keep the resolver shape identical to peers (the directory components below
# are a filesystem anchor path, not a Python import).
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
    raise RuntimeError(
        "repo root not found (anchor file revision-slash-core-slash-"
        "preprocessing.py missing)"
    )


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "revision/results"
DOCS = REPO / "revision/docs"
MATCHED = RESULTS / "matched2000"

# Canonical training seed set re-emitted per model row.
SEED_SET = [42, 43, 44, 45, 46]

# The accepted 2000ep matched-budget sweep models (9-model x 5-seed matrix; the
# 55-param IQP:SEL reproduction + V1/V2/V3 ansatz + 3 classical WGAN baselines +
# 2 non-adversarial baselines). The frozen-checkpoint headline is NOT in this
# list — it is a SEPARATE row sourced from headline_canonical.json (D-14-10).
SWEEP_MODELS = [
    "iqp_sel_55_repro",
    "V1",
    "V2",
    "V3",
    "wgan_mlp",
    "wgan_cnn",
    "wgan_lstm",
    "vae",
    "ar",
]

# Map each sweep model dir name to the model_kind used in the frozen 1000ep
# baseline_comparison.json (the reconciliation old-value basis). Only the rows
# that have a 1000ep counterpart reconcile; ansatz V1/V2/V3 have no 1000ep
# matched-budget counterpart so they carry an explicit "no 1000ep basis" note.
RECON_KIND = {
    "iqp_sel_55_repro": "quantum",
    "wgan_mlp": "wgan_mlp",
    "wgan_cnn": "wgan_cnn",
    "wgan_lstm": "wgan_lstm",
    "vae": "vae",
    "ar": "ar",
}


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader for the flat config.yaml the sweep emits.

    The sweep config.yaml is a flat scalar/list/one-level-nested mapping (no
    anchors, no multi-doc). We parse it without importing PyYAML to keep the
    aggregator dependency-free (RESEARCH Open Q3 idiom: stdlib only). Values are
    json-decoded where possible (so `1.8046e-05` -> float, `null` -> None,
    `true` -> bool); otherwise kept as the raw stripped string.
    """
    out: dict = {}
    cur_key: str | None = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if line.startswith("- "):
            # list item under the most-recent top-level key
            if cur_key is not None:
                out.setdefault(cur_key, [])
                if isinstance(out[cur_key], list):
                    out[cur_key].append(_coerce(line[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if indent == 0:
            cur_key = key
            if val == "":
                out[key] = {}  # nested block or list follows
            else:
                out[key] = _coerce(val)
        else:
            # one-level-nested block (e.g. device_manifest:)
            parent = out.get(cur_key)
            if isinstance(parent, dict):
                parent[key] = _coerce(val)
    return out


def _coerce(s: str):
    s = s.strip()
    if s.startswith(('"', "'")) and s.endswith(('"', "'")) and len(s) >= 2:
        return s[1:-1]
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def _build_model_record(model: str) -> dict:
    """Emit ONE models[] record from the accepted 2000ep sweep config (seed 42
    is the canonical config; every seed shares the identical config modulo the
    seed field — the strict accept gate D-14-13 enforced this)."""
    cfg = _load_yaml(MATCHED / "runs" / model / "42" / "config.yaml")
    dm = cfg.get("device_manifest", {}) or {}
    return {
        "model": model,
        "kind": cfg.get("model_kind"),
        "parameter_count": cfg.get("parameter_count"),
        "family": cfg.get("family"),
        "source": cfg.get("source"),
        "circuit_id": cfg.get("circuit_id"),
        "ansatz": cfg.get("ansatz"),
        "depth": cfg.get("depth"),
        "topology": cfg.get("topology"),
        "epochs": cfg.get("epochs"),
        "early_stop": "OFF (full 2000ep, D-14-13)",
        "optimizer": _optimizer_for(cfg),
        "lr_critic": cfg.get("lr_critic"),
        "lr_generator": cfg.get("lr_generator"),
        "optimizer_betas": [0.0, 0.9],
        "batch_size": cfg.get("batch_size"),
        "n_critic": cfg.get("n_critic"),
        "lambda_gp": cfg.get("lambda_gp"),
        "seeds": SEED_SET,
        "num_qubits": cfg.get("num_qubits"),
        "num_layers": cfg.get("num_layers"),
        "window_length": cfg.get("window_length"),
        "n_real_windows": cfg.get("n_real_windows"),
        "device": dm.get("sample_generation_device"),
        "dtype": dm.get("sample_generation_dtype"),
        "pennylane_device": dm.get("pennylane_device"),
        "diff_method": dm.get("diff_method"),
        "backend_assertion": dm.get("backend_assertion"),
        "data_hash": cfg.get("data_hash"),
        "tier": cfg.get("tier"),
        "train_protocol_notes": cfg.get("train_protocol_notes"),
    }


def _optimizer_for(cfg: dict) -> str:
    family = cfg.get("family", "")
    if family == "non-adversarial":
        if cfg.get("model_kind") == "ar":
            return "closed-form np.linalg.lstsq (no optimizer)"
        return "Adam (single, lr=1e-3) — VAE ELBO loop"
    return "Adam, betas=(0.0, 0.9) — WGAN-GP"


def _wall_seconds_by_model(sweep_status: dict) -> dict:
    """Per-model wall-time from sweep_status.json. The accepted sweep ran to
    completion across kill/resume cycles; the *final* idempotent re-invocation
    reports wall_seconds=0 with skipped_already_done=true for already-accepted
    runs. We sum only NON-skipped wall_seconds; if every run for a model was
    skipped on the recorded invocation we emit null with a provenance note
    (the strict-gate-accepted artifact bundle does not store per-run wall time
    independently of sweep_status — honest null beats a fabricated number)."""
    agg: dict[str, list[int]] = {}
    for r in sweep_status.get("runs", []):
        if r.get("skipped_already_done"):
            continue
        ws = r.get("wall_seconds")
        if isinstance(ws, (int, float)) and ws > 0:
            agg.setdefault(r["model"], []).append(int(ws))
    out = {}
    for m in SWEEP_MODELS:
        vals = agg.get(m)
        out[m] = sum(vals) if vals else None
    return out


def _final_eval_value(metrics: dict, key: str):
    """Final (last-eval-step) value of a trajectory metric, or None."""
    v = metrics.get(key)
    if isinstance(v, list) and v:
        return v[-1]
    if isinstance(v, (int, float)):
        return v
    return None


def _reconciliation_rows() -> list[dict]:
    """Per-model 1000ep -> 2000ep delta on the headline EMD metric.

    OLD basis: the frozen Phase-10 `baseline_comparison.json` (1000ep budget) —
    mean OD-EMD over Pipeline B, seeds 42-46 (the canonical headline cell).
    NEW basis: the accepted 2000ep `matched2000/runs/<model>/<seed>/metrics.json`
    final (last-eval-step) `emd_avg`, averaged over seeds 42-46.

    Both numbers are READ, never recomputed. Ansatz V1/V2/V3 have no 1000ep
    matched-budget counterpart in baseline_comparison.json -> recorded with an
    explicit "no 1000ep basis" marker rather than a fabricated old value.
    """
    old = json.loads((RESULTS / "baseline_comparison.json").read_text())
    rows = []
    for model in SWEEP_MODELS:
        recon_kind = RECON_KIND.get(model)
        # NEW: mean final emd_avg over the 5 accepted 2000ep seeds.
        new_vals = []
        for s in SEED_SET:
            mp = MATCHED / "runs" / model / str(s) / "metrics.json"
            if mp.exists():
                fv = _final_eval_value(json.loads(mp.read_text()), "emd_avg")
                if isinstance(fv, (int, float)):
                    new_vals.append(fv)
        new_mean = statistics.fmean(new_vals) if new_vals else None
        # Non-adversarial baselines (VAE ELBO loop D-10-09 / AR closed-form
        # D-10-13) carry NO training-trajectory `emd_avg` in metrics.json — and
        # recomputing EMD from samples.npy is forbidden (pure aggregator, no
        # metric recompute, D-14-16). Annotate the absence explicitly so a
        # blank NEW for a model that has an OLD basis is self-explaining
        # (faithful provenance, never a silent gap).
        if new_mean is None and not new_vals:
            new_basis = (
                "no 2000ep EMD trajectory — non-adversarial baseline "
                "tracks ELBO/closed-form fit, not adversarial EMD "
                "(metrics.json carries no emd_avg; recompute forbidden, "
                "D-14-16)"
            )
        else:
            new_basis = (
                "matched2000/runs/<model>/<seed>/metrics.json emd_avg[-1], "
                "mean over seeds 42-46"
            )
        if recon_kind is None:
            rows.append(
                {
                    "model": model,
                    "metric": "emd (OD, final-eval mean over seeds 42-46)",
                    "old_1000ep": None,
                    "old_basis": "no 1000ep matched-budget counterpart "
                    "(ansatz variant introduced at 2000ep, D-14-10)",
                    "new_2000ep": new_mean,
                    "new_basis": new_basis,
                    "delta": None,
                }
            )
            continue
        # OLD: mean OD-EMD, Pipeline B, seeds 42-46, from frozen 1000ep file.
        old_vals = [
            r["value"]
            for r in old["rows"]
            if r["model_kind"] == recon_kind
            and r["pipeline"] == "B"
            and r["metric_name"] == "emd"
            and r["scale"] == "OD"
        ]
        old_mean = statistics.fmean(old_vals) if old_vals else None
        delta = (
            (new_mean - old_mean)
            if (new_mean is not None and old_mean is not None)
            else None
        )
        rows.append(
            {
                "model": model,
                "metric": "emd (OD, final-eval mean over seeds 42-46)",
                "old_1000ep": old_mean,
                "old_basis": "baseline_comparison.json rows[] "
                f"(model_kind={recon_kind}, pipeline=B, emd, OD)",
                "new_2000ep": new_mean,
                "new_basis": new_basis,
                "delta": delta,
            }
        )
    return rows


def _write_reconciliation_note(recon: list[dict], data_hash: str) -> None:
    lines: list[str] = []
    lines.append("# 1000ep -> 2000ep Reconciliation Note (D-14-13)\n")
    lines.append(
        "> **Generated** by `revision/run_model_info.py` — every number below "
        "is READ from a `revision/results/*.json` artifact, never recomputed "
        "or hand-typed (D-14-16, Pitfall 5).\n"
    )
    lines.append(
        "This note is the authoritative record of every headline metric that "
        "changed when the budget moved from the unfair 1000ep / 75-param "
        "regime to the matched 2000ep / 55-param regime (Tier-2/3 of "
        "D-14-22). The **OLD** column is the frozen Phase-10 "
        "`baseline_comparison.json` (1000-epoch budget); the **NEW** column is "
        "the accepted 2000-epoch `matched2000` sweep. Any manuscript number "
        "that moved between submission and resubmission MUST cite this "
        "delta.\n"
    )
    lines.append(
        f"`data_hash` = `{data_hash}` — identical across every consumed "
        "artifact (cross-artifact explicit-raise gate, "
        "run_multiseed_rollup.py:86-92 idiom).\n"
    )
    lines.append("## EMD (OD scale) — final-eval mean over seeds 42-46\n")
    lines.append(
        "| model | old (1000ep) | new (2000ep) | delta | old basis "
        "| new basis |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in recon:
        o = "—" if r["old_1000ep"] is None else f"{r['old_1000ep']:.6f}"
        n = "—" if r["new_2000ep"] is None else f"{r['new_2000ep']:.6f}"
        d = "—" if r["delta"] is None else f"{r['delta']:+.6f}"
        lines.append(
            f"| {r['model']} | {o} | {n} | {d} | {r['old_basis']} "
            f"| {r['new_basis']} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation.** A negative delta means the matched 2000ep "
        "budget *improved* (lowered) the EMD relative to the unfair 1000ep "
        "baseline. The ansatz variants (V1/V2/V3) have no 1000ep "
        "matched-budget counterpart — they were introduced directly at the "
        "2000ep budget (D-14-10) — so their OLD column is intentionally "
        "blank rather than carrying a non-comparable number.\n"
    )
    (DOCS / "reconciliation_note.md").write_text("\n".join(lines))


def main() -> None:
    # ── Load the accepted artifacts ───────────────────────────────────────────
    headline = json.loads((RESULTS / "headline_canonical.json").read_text())
    recovery = json.loads((RESULTS / "canonical_recovery.json").read_text())
    sweep_status = json.loads((MATCHED / "sweep_status.json").read_text())

    sweep_cfgs = {
        m: _load_yaml(MATCHED / "runs" / m / "42" / "config.yaml")
        for m in SWEEP_MODELS
    }

    # ── Cross-artifact data_hash gate (HARD, explicit-raise, python -O safe) ───
    # run_multiseed_rollup.py:86-92 idiom: assert ONLY mutual equality of the
    # frozen `data_hash` fields across every consumed 2000ep artifact (the
    # headline + all 9 accepted sweep configs). Do NOT re-derive the hash.
    hashes = {"headline_canonical.json": headline["data_hash"]}
    for m, c in sweep_cfgs.items():
        hashes[f"matched2000/{m}/config.yaml"] = c.get("data_hash")
    if len(set(hashes.values())) != 1:
        raise AssertionError(
            f"data_hash mismatch across consumed 2000ep artifacts: {hashes}"
        )
    canonical_hash = next(iter(hashes.values()))  # expect 91e447d4624e25b3

    # ── Build the unified models[] table ──────────────────────────────────────
    wall = _wall_seconds_by_model(sweep_status)
    breadcrumbs = recovery.get("optimizer_breadcrumbs", {})
    hdev = headline.get("device", {})

    models: list[dict] = []

    # Row 1 — the FROZEN-checkpoint headline (load-bearing, D-14-03/05/10).
    # Distinct `source` marker from the 2000ep reproduction below.
    models.append(
        {
            "model": "iqp_sel_55_headline",
            "kind": "quantum",
            "parameter_count": headline["param_count"],
            "family": "adversarial-quantum",
            "source": headline["source"],  # frozen_checkpoint_epoch_1969
            "circuit_id": headline["locked_circuit_id"],
            "ansatz": "iqp_sel_55",
            "depth": recovery["decomposition"]["num_layers"],
            "topology": recovery["decomposition"]["gate_layout"]["entangler"],
            "epochs": 2000,
            "early_stop": "EarlyStopping ON during original training; headline "
            "is the FROZEN best-EMD checkpoint at epoch "
            f"{headline['checkpoint_epoch']} (D-14-03)",
            "optimizer": "Adam, betas=(0.0, 0.9) — WGAN-GP (frozen-checkpoint "
            "training-time optimizer breadcrumbs)",
            "lr_critic": breadcrumbs.get("c_optimizer_lr"),
            "lr_generator": breadcrumbs.get("g_optimizer_lr"),
            "optimizer_betas": breadcrumbs.get("g_optimizer_betas", [0.0, 0.9]),
            "batch_size": None,
            "n_critic": None,
            "lambda_gp": None,
            "seeds": [headline["generation_seed"]],
            "num_qubits": recovery["decomposition"]["num_qubits"],
            "num_layers": recovery["decomposition"]["num_layers"],
            "window_length": 10,
            "n_real_windows": None,
            "device": hdev.get("torch_device"),
            "dtype": headline.get("dtype"),
            "pennylane_device": hdev.get("pennylane_device"),
            "diff_method": hdev.get("diff_method"),
            "backend_assertion": hdev.get("backend_assertion"),
            "data_hash": headline["data_hash"],
            "tier": "T1 (frozen checkpoint)",
            "wall_seconds": None,
            "checkpoint_epoch": headline["checkpoint_epoch"],
            "checkpoint_sha256": headline["checkpoint_sha256"],
            "train_protocol_notes": headline.get("source_note"),
        }
    )

    # Rows 2..10 — the accepted 2000ep sweep models (reproduction + ansatz +
    # classical baselines + non-adversarial baselines).
    for m in SWEEP_MODELS:
        rec = _build_model_record(m)
        rec["wall_seconds"] = wall.get(m)
        models.append(rec)

    out = {
        "schema": "long-form rows[] + models[] aggregate (D-10-16)",
        "metric_helpers": "revision.core.eval ONLY (D-10-20)",
        "data_hash": canonical_hash,
        "consumed_artifacts": {
            "headline_canonical.json": headline["data_hash"],
            "canonical_recovery.json": "optimizer_breadcrumbs (LR/betas)",
            "matched2000/sweep_status.json": (
                f"{sweep_status.get('completed_count')}/"
                f"{sweep_status.get('total_count')} accepted"
            ),
            **{
                f"matched2000/runs/{m}/*/config.yaml": c.get("data_hash")
                for m, c in sweep_cfgs.items()
            },
        },
        "seed_set": SEED_SET,
        "model_kinds": sorted({m["kind"] for m in models}),
        "models": models,
        # rows[] kept for long-form schema conformance; the model-info table is
        # carried by models[] (one record per model, D-14-15). The numeric
        # headline rows live in headline_canonical.json (the load-bearing
        # evaluation artifact) — duplicating them here would create a second
        # source of truth, which D-14-16 explicitly forbids.
        "rows": [],
        "rows_note": "model-info is a models[]-only aggregate (D-14-15); "
        "numeric evaluation rows are the load-bearing "
        "headline_canonical.json — not duplicated here (single "
        "source of truth, D-14-16).",
    }
    (RESULTS / "model_info.json").write_text(json.dumps(out, indent=2))

    # ── Reconciliation note (1000ep -> 2000ep deltas, D-14-13) ────────────────
    recon = _reconciliation_rows()
    _write_reconciliation_note(recon, canonical_hash)

    print(
        f"model_info.json written: {len(models)} model records, "
        f"data_hash={canonical_hash}"
    )
    print(
        f"reconciliation_note.md written: {len(recon)} model deltas "
        "(1000ep -> 2000ep, EMD OD)"
    )


if __name__ == "__main__":
    main()
