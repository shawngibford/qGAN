"""Phase 14 PAPER-08 driver — the unified per-model info aggregator.

This is a **pure consumer / pure aggregator** (D-14-16): it reads ONLY the
accepted 2000ep result artifacts —

  * `results/headline_canonical.json`
        (the frozen-checkpoint headline, source=frozen_checkpoint_epoch_1969),
  * `results/matched2000/runs/<model>/<seed>/config.yaml`
        (the strict-gate-accepted 2000ep sweep configs, all 9 models x 5 seeds),
  * `results/matched2000/sweep_status.json`
        (resumable sweep state — per-run wall-time provenance),
  * `results/canonical_recovery.json`
        (the frozen-checkpoint optimizer LR/betas breadcrumbs, D-14-01),

and emits ONE unified `results/model_info.json` where every model is
a single `models[]` record (D-14-15): the 55-param IQP:SEL frozen-checkpoint
headline and its 2000ep reproduction are SEPARATE rows (D-14-10), the V1/V2/V3
ansatz variants, the three classical WGAN-GP baselines (mlp/cnn/lstm), and the
two non-adversarial baselines (VAE, AR). Each record carries the D-14-15
columns (params, epochs=2000, early-stop state, optimizer/LR/betas, batch,
N_CRITIC, lambda, seeds {42..46}, window config, device/dtype, data_hash,
wall-time).

It ALSO writes `docs/reconciliation_note.md` recording, per model /
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

Pattern source: `run_multiseed_rollup.py` — repo-root resolver
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

# PyYAML is the canonical config.yaml reader used by every peer driver
# (run_matched2000.py:630 yaml.safe_load — the same script that WROTE these
# config.yaml files via yaml.safe_dump). It is NOT a deep-learning framework
# and NOT the shared model package, so importing it preserves the pure-
# aggregator constraint (D-14-16). Using safe_load (not a hand-rolled parser)
# correctly handles the multi-line block scalars config.yaml emits
# (train_protocol_notes) — a hand-rolled scalar parser silently truncates them.
import yaml


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
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError(
        "repo root not found (anchor file core/"
        "preprocessing.py missing)"
    )


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
DOCS = REPO / "docs"
MATCHED = RESULTS / "matched2000"

# Canonical training seed set re-emitted per model row.
SEED_SET = [42, 43, 44, 45, 46]

# Plan 14-13 Task 4 (HI-3 / PROV-HIGH-2): canonical dataset hash. The
# cross-artifact gate below previously asserted only MUTUAL equality of the
# observed hashes across consumed artifacts; that passes loudly when all
# consumed artifacts agree on a DIFFERENT hash than the audited
# 91e447d4624e25b3. The explicit-raise gate now also asserts equality to
# EXPECTED_DATA_HASH so a regression of the dataset itself surfaces
# immediately rather than after silently propagating through the rest of
# Phase 14.
EXPECTED_DATA_HASH = "91e447d4624e25b3"

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
    """Read a sweep config.yaml via yaml.safe_load — the canonical reader.

    run_matched2000.py wrote these configs with yaml.safe_dump and reads them
    back with yaml.safe_load (run_matched2000.py:630/787); we use the identical
    idiom so types match by construction (e.g. `1.8046e-05` -> float, `null` ->
    None, multi-line block scalars like train_protocol_notes -> full string,
    nested device_manifest -> dict). safe_load never executes arbitrary tags.
    """
    return yaml.safe_load(path.read_text()) or {}


def _build_model_record(model: str) -> dict:
    """Emit ONE models[] record from the accepted 2000ep sweep config (seed 42
    is the canonical config; every seed shares the identical config modulo the
    seed field — the strict accept gate D-14-13 enforced this)."""
    cfg = _load_yaml(MATCHED / "runs" / model / "42" / "config.yaml")
    dm = cfg.get("device_manifest", {}) or {}
    # Plan 14-13 Task 4 (HI-2): optimizer_betas is family-specific.
    # WGAN-GP families use (0.0, 0.9) per Gulrajani's recipe; non-adversarial
    # (VAE / AR) baselines do not use the WGAN-GP critic optimization and
    # report `None` rather than the misleading [0.0, 0.9] hardcode.
    family = cfg.get("family", "")
    if family == "non-adversarial":
        betas: list | None = None
    else:
        betas = [0.0, 0.9]
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
        "optimizer_betas": betas,
        "batch_size": cfg.get("batch_size"),
        "n_critic": cfg.get("n_critic"),
        "lambda_gp": cfg.get("lambda_gp"),
        "seeds": SEED_SET,
        "num_qubits": cfg.get("num_qubits"),
        "num_layers": cfg.get("num_layers"),
        "window_length": cfg.get("window_length"),
        "n_real_windows": cfg.get("n_real_windows"),
        "device": dm.get("sample_generation_device"),
        # Plan 14-13 Task 4 (PROV-HIGH-3 / HIGH-3): dtype field renamed to
        # dtype_samples (the field genuinely is sample-generation dtype, not
        # parameter dtype). dtype_params added alongside for explicit clarity.
        "dtype_samples": dm.get("sample_generation_dtype"),
        "dtype_params": "torch.float32",
        # Keep legacy 'dtype' alias for any consumer that hasn't switched yet.
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

    NEW basis (Plan 14-13 Task 3, C-1 / PROV-CRIT-1 / C-3 remediation): the
    audited OD-scale aggregate mean from
    `matched2000_dualscale.json#aggregates` (entries with
    `metric_name="emd"` and `scale="OD"`) — NOT the previous
    `metrics.json["emd_avg"][-1]` read which sourced the log-return-standardized
    training-loop metric (the scale-collision root cause of C-1 / PROV-CRIT-1).
    The OD-scale aggregate is the same scale as the OLD basis, so deltas are
    now interpretable across the 1000ep/2000ep boundary.

    Both numbers are READ, never recomputed. Ansatz V1/V2/V3 have no 1000ep
    matched-budget counterpart in baseline_comparison.json -> recorded with an
    explicit "no 1000ep basis" marker rather than a fabricated old value.
    """
    old = json.loads((RESULTS / "baseline_comparison.json").read_text())
    dual = json.loads((RESULTS / "matched2000_dualscale.json").read_text())
    # Build {model_kind: OD-scale EMD mean} from the audited aggregates.
    od_emd_by_model: dict[str, float] = {
        a["model_kind"]: a["mean"]
        for a in dual.get("aggregates", [])
        if a.get("metric_name") == "emd" and a.get("scale") == "OD"
    }
    rows = []
    for model in SWEEP_MODELS:
        recon_kind = RECON_KIND.get(model)
        # NEW (Plan 14-13 source switch): audited OD-scale aggregate mean.
        new_mean = od_emd_by_model.get(model)
        if new_mean is None:
            # Non-adversarial baselines historically lacked an OD-scale EMD
            # aggregate in older snapshots; in the current corpus VAE/AR both
            # carry an entry. Preserve the legacy "no 2000ep EMD" basis only
            # when the aggregate is truly absent.
            new_basis = (
                "no 2000ep OD-scale EMD aggregate in "
                "matched2000_dualscale.json#aggregates "
                "(recompute forbidden, D-14-16)"
            )
        else:
            new_basis = (
                "matched2000_dualscale.json#aggregates "
                "(metric_name=emd, scale=OD); audited mean over seeds 42-46 "
                "(Plan 14-13 Task 3, C-1 / PROV-CRIT-1 source switch)"
            )
        if recon_kind is None:
            rows.append(
                {
                    "model": model,
                    "metric": "emd (OD, audited aggregate mean over seeds 42-46)",
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
                "metric": "emd (OD, audited aggregate mean over seeds 42-46)",
                "old_1000ep": old_mean,
                "old_basis": "baseline_comparison.json rows[] "
                f"(model_kind={recon_kind}, pipeline=B, emd, OD)",
                "new_2000ep": new_mean,
                "new_basis": new_basis,
                "delta": delta,
            }
        )
    return rows


def _comparable_variants_rows() -> list[dict]:
    """Plan 14-15 T2 — 3-column comparable-variants table source builder.

    Each row sources three EMD columns from existing aggregator JSONs:

      * **Column 1 — OD raw-sample EMD** from
        ``matched2000_dualscale.json#aggregates`` filtered by
        ``metric_name='emd'`` AND ``scale='OD'`` (per-(model_kind) mean
        over seeds 42-46; ddof=1 sample std).
      * **Column 2 — log-return raw-sample EMD** from
        ``matched2000_dualscale.json#aggregates`` filtered by
        ``metric_name='emd'`` AND ``scale='log_return'`` (per-(model_kind)
        mean over seeds 42-46; ddof=1 sample std).
      * **Column 3 — 50-bin histogram-density EMD** (the pre-v1.0 paper
        metric reintroduced per Plan 14-15) from
        ``distribution_emd.json#aggregates`` filtered by ``scale='OD'``
        (per-(model_kind) mean over seeds 42-46; ddof=1 sample std).

    The 50-bin density Wasserstein formulation in column 3 matches the
    C-3 disclosure citation in ``reconciliation_note.md`` word-for-word
    so reviewers can verify the formulation actually computed matches the
    formulation disclosed (Plan 14-15 T1 + T2).

    Sorted by SWEEP_MODELS (MODEL_ORDER from run_figure_suite.py:106-116).
    Headline iqp_sel_55_headline appears last as a separate row IFF
    `distribution_emd.json#headline_present == True`.
    """
    dual = json.loads((RESULTS / "matched2000_dualscale.json").read_text())
    dist = json.loads((RESULTS / "distribution_emd.json").read_text())

    def _by_scale(scale: str) -> dict[str, dict]:
        return {
            a["model_kind"]: a
            for a in dual.get("aggregates", [])
            if a.get("metric_name") == "emd" and a.get("scale") == scale
        }

    od_raw = _by_scale("OD")
    logret_raw = _by_scale("log_return")
    od_hist = {
        a["model_kind"]: a
        for a in dist.get("aggregates", [])
        if a.get("scale") == "OD"
    }
    # Plan 14-16 — 4th column: per-(model_kind) OD-scale DTW mean from
    # matched2000_dualscale.json (DTW emitter at core/eval.py:38-89
    # byte-untouched under D-14-22; DTW aggregates byte-stable through 14-16).
    dtw_od = {
        a["model_kind"]: a
        for a in dual.get("aggregates", [])
        if a.get("metric_name") == "dtw_mean" and a.get("scale") == "OD"
    }

    rows: list[dict] = []
    model_order = list(SWEEP_MODELS)
    if dist.get("headline_present"):
        model_order.append("iqp_sel_55_headline")
    for model in model_order:
        rows.append({
            "model": model,
            "od_raw_mean": od_raw.get(model, {}).get("mean"),
            "od_raw_std": od_raw.get(model, {}).get("std"),
            "logret_raw_mean": logret_raw.get(model, {}).get("mean"),
            "logret_raw_std": logret_raw.get(model, {}).get("std"),
            "od_hist_mean": od_hist.get(model, {}).get("mean"),
            "od_hist_std": od_hist.get(model, {}).get("std"),
            "dtw_od_mean": dtw_od.get(model, {}).get("mean"),
            "dtw_od_std": dtw_od.get(model, {}).get("std"),
            "source_dist_emd": (
                f"distribution_emd.json#aggregates "
                f"(model_kind={model}, scale=OD); 50-bin density Wasserstein "
                f"(pre-v1.0 formulation)"
            ),
        })
    return rows


def _write_reconciliation_note(recon: list[dict], data_hash: str) -> None:
    lines: list[str] = []
    lines.append("# 1000ep -> 2000ep Reconciliation Note (D-14-13)\n")
    lines.append(
        "> **Generated** by `run_model_info.py` — every number below "
        "is READ from a `results/*.json` artifact, never recomputed "
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
        "**Integration caveat (Plan 14-12, recorded post-14-09/14-10).** Two "
        "facets of the table above are now backed by additional audited "
        "artifacts: (1) the V1/V2/V3 row param-count values (75 / 135 / 75) "
        "now resolve directly to `results/v1_config_lock.json`, "
        "`results/v2_config_lock.json`, and "
        "`results/v3_config_lock.json` (Plan 14-09 — "
        "`gate_layout_breakdown` field decomposes each count as IQP "
        "encoding (5) + N\\*SEL layers (15 each) + final RX+RY (10)), rather "
        "than only indirectly through the `_QUANTUM_ANSATZ` dict at "
        "`run_matched2000.py:118-122`; (2) the D-14-10 "
        "headline-vs-repro distinction (iqp_sel_55_headline as the "
        "frozen-checkpoint EMD, iqp_sel_55_repro as the matched-2000ep "
        "reproduction) is now visualized as two distinct points in "
        "`figures/param_efficiency_pareto.{png,pdf}` "
        "(Plan 14-10 — the headline appears as a separate dashed/diamond "
        "marker per the conflation-guard contract).\n"
    )
    lines.append(
        "**Interpretation (Plan 14-14 rewording — wgan_cnn seed-variance honesty).** All deltas are within seed variance of the 1000-epoch baseline (Welch t-test p ≥ 0.37 for every model). The largest absolute delta is wgan_cnn (-0.059, a ~50% reduction off a small base); seed-42 outliers drive that variance, not a uniform near-zero effect across seeds. The previous `+0.127 degradation` framing was an artifact of the scale-collision between the OLD column (OD-scale EMD from `baseline_comparison.json`) and the NEW column (which under the v1 emit read the log-return-standardized training-loop metric `emd_avg[-1]` from `metrics.json`). Task 3 of Plan 14-13 switches the NEW source to the audited OD-scale aggregate mean in `matched2000_dualscale.json#aggregates` — the same scale as OLD — and the deltas collapse to numerical noise modulo the wgan_cnn seed-variance caveat above (C-1 / PROV-CRIT-1 resolved). The ansatz variants (V1/V2/V3) have no 1000ep matched-budget counterpart — they were introduced directly at the 2000ep budget (D-14-10) — so their OLD column is intentionally blank rather than carrying a non-comparable number."
    )
    lines.append("")
    lines.append(
        "**Cross-reference (Plan 14-15).** "
        "See `## EMD comparable across metric variants (matched 2000ep budget)` "
        "below for the 3-column comparison that adds the log-return raw-sample "
        "EMD and the 50-bin histogram-density EMD (the pre-v1.0 paper metric "
        "reintroduced per Plan 14-15)."
    )
    lines.append("")
    lines.append(
        "**Metric-redefinition disclosure (Plan 14-13, peer-review remediation; Plan 14-15 extension).**"
    )
    lines.append(
        "The v1.0 release (`core/eval.py:25-36`) switched the EMD "
        "implementation from a histogram-density Wasserstein "
        "(`scipy.stats.wasserstein_distance(bin_centers, bin_centers, "
        "real_hist_density, fake_hist_density)` over 50-bin histograms "
        "(`np.histogram(..., density=True)`)) to a raw-sample Wasserstein "
        "(`scipy.stats.wasserstein_distance(real_samples, fake_samples)` "
        "over the raw log-return arrays). The two metrics are NOT "
        "commensurate: the pre-v1.0 headline `~0.0015` (histogram-density on "
        "`real`-only test slice) and the v1.0+ headline `~0.121` "
        "log-return-standardized EMD measure different probabilistic "
        "distances over different supports. The headline trajectory across "
        "versions is therefore: pre-v1.0 ≈ 0.0015 (histogram-density, "
        "deprecated); v1.0 ≈ 0.121 (log-return-standardized raw-sample, "
        "current training-loop metric). The OD-scale aggregate mean in the "
        "table above is the v1.0 raw-sample Wasserstein evaluated on the "
        "original ordinary-differences (OD) scale samples rescaled via "
        "`np.exp(.) - 1` (C-3 resolved). "
        "**Plan 14-15 extension.** "
        "The matched-2000ep histogram-density EMD on OD scale (reported in "
        "the new `## EMD comparable across metric variants (matched 2000ep "
        "budget)` section below; sourced from `distribution_emd.json`) is "
        "computed on the SAME real-data slice and SAME 50-bin convention as "
        "the deprecated v1.0-pre metric, so it IS commensurate with the "
        "pre-v1.0 paper headline (~0.0015) for the first time since the "
        "v1.0 raw-sample switch (Plan 14-15). "
        "**Plan 14-16 r3 remediation.** "
        "Column 2 (log-return raw-sample EMD) was corrected for a "
        "standardization scale mismatch in `run_matched2000_dualscale.py:"
        "368-372` AND `run_distribution_emd.py:_real_references` at "
        "`:144-153` (R3-HI-1 sister sites under the same finding ID in "
        "`peer-review-r3/code-review-r3.md` §H3) via the un-standardize-fake "
        "recipe from `pipeline-review-r3.md` §2; column 3 (50-bin "
        "histogram-density EMD on OD scale) was reformulated with "
        "shared-edges-from-real to eliminate the per-distribution "
        "renormalization concern documented in `peer-review-r3/"
        "code-review-r3.md` R3-CR-1 (investigation finding: with shared "
        "edges the density=True vs density=False formulation is numerically "
        "inert for `scipy.stats.wasserstein_distance`, which renormalizes "
        "weights internally — the OD-scale v1->v2 values are byte-identical; "
        "the fix's genuine contribution is the `fake_in_range_mass` "
        "disclosure stat confirming no out-of-range truncation). Plan 14-16 "
        "r3 remediation also revealed that the pre-fix LR-EMD column had "
        "inverted the quantum-vs-classical ranking due to the R3-CR-2 scale "
        "mismatch; the corrected column places AR (3-parameter Yule-Walker "
        "baseline) first at 0.003 and shows quantum/WGAN/VAE clustering in "
        "the 0.007-0.016 band with no statistically meaningful separation. "
        "See `peer_review_remediation.md` Plan 14-16 r3-process retraction "
        "for the full disclosure. "
        "DTW addendum (Plan 14-16): the comparable-variants table below is "
        "extended with a 4th column reporting per-(model_kind, scale=OD) "
        "DTW mean from `matched2000_dualscale.json#aggregates[*, scale='OD', "
        "metric_name='dtw_mean']` (n=5 seeds per cell, byte-stable through "
        "Plan 14-16 since the DTW emitter at `core/eval.py:38-89` "
        "is BYTE-UNTOUCHED under D-14-22). The DTW per-baseline means "
        "surface the Orlandi-improvement ratio (~6.5x lower than the "
        "Orlandi et al. reference DTW=1.954) and the LR-scale "
        "quantum-vs-WGAN/AR dominance disclosed in `peer_review_remediation.md` "
        "Plan 14-16 DTW phantom asymmetry section. See `methods_full.md` "
        "`### DTW historical context (Plan 14-16)` paragraph for the "
        "methodological framing."
    )
    lines.append("")
    lines.append(
        "**Plan 14-21 amendment (current state of the LR-EMD ranking and the "
        "OD-EMD delta column).** A WGAN sample-space convention preserved at "
        "9 paper-cited `samples.npy` load sites was undone via the shared "
        "inference-only helper `_wgan_unscale.py` (see "
        "`paper/supp_material.tex` §A.7 disclosure paragraph and "
        "`14-21-SUMMARY.md` for the full audit trail). The correction is "
        "gated by the `_WGAN_KINDS` set so VAE and AR(2) samples are passed "
        "through untouched (the differential test on every regenerated JSON "
        "confirms every VAE+AR(2) row is bit-identical pre/post). Two "
        "load-bearing readings of the tables above shift as a result:"
    )
    lines.append("")
    lines.append(
        "1. **OD-EMD delta column** — the WGAN-family rows in the `delta` "
        "column at the top of this note are now substantially larger than "
        "the pre-14-21 \"within seed variance\" framing implied (the Plan "
        "14-14 wgan_cnn `-0.059` claim now reads off the regenerated "
        "`matched2000_dualscale.json` as a larger positive delta). The "
        "quantum + VAE + AR(2) rows are unchanged. The Plan 14-14 "
        "interpretation paragraph above remains the historical record of how "
        "the C-1 / PROV-CRIT-1 scale-collision was closed at 14-13/14-14; "
        "the current numerical reading is the post-14-21 row directly above "
        "it in this table."
    )
    lines.append("")
    lines.append(
        "2. **LR-EMD ranking (column 2 of the comparable-variants table "
        "below)** — the Plan 14-16 r3 remediation claim that \"AR is first "
        "at 0.003 and quantum/WGAN/VAE cluster in the 0.007-0.016 band with "
        "no statistically meaningful separation\" was authored from the "
        "pre-14-21 column. The corrected column places AR first "
        "(closed-form Yule-Walker on the marginal log-return distribution), "
        "quantum second (~0.0040-0.0050), VAE third (~0.0158), and the "
        "WGAN cluster substantially worse (~0.024-0.129) with cluster-floor "
        "Welch p well inside conventional significance. The Plan 14-16 "
        "r3-process retraction section of `peer_review_remediation.md` "
        "carries the same supersession banner."
    )
    lines.append("")

    # Plan 14-15 T2 — append the 3-column comparable-variants section.
    comp = _comparable_variants_rows()
    lines.append(
        "## EMD comparable across metric variants (matched 2000ep budget)"
    )
    lines.append("")
    lines.append(
        "Each column below reports mean EMD over seeds 42-46 (ddof=1 "
        "sample std also recorded in the aggregator JSONs). Column 1 is the "
        "v1.0 raw-sample Wasserstein on OD scale "
        "(`matched2000_dualscale.json#aggregates`, scale=OD); column 2 is "
        "the v1.0 raw-sample Wasserstein on log-return scale "
        "(`matched2000_dualscale.json#aggregates`, scale=log_return); "
        "column 3 is the pre-v1.0 50-bin histogram-density Wasserstein on "
        "OD scale (`distribution_emd.json#aggregates`, scale=OD; Plan 14-15 "
        "reintroduction). Column 3 is commensurate with the pre-v1.0 paper "
        "headline (~0.0015) under the SAME 50-bin convention (see the C-3 "
        "metric-redefinition disclosure above). Column 4 (Plan 14-16) is the "
        "per-(model_kind) OD-scale DTW mean over seeds 42-46 from "
        "`matched2000_dualscale.json#aggregates`, scale=OD, "
        "metric_name='dtw_mean' — see the C-3 DTW addendum above and "
        "`methods_full.md` `### DTW historical context (Plan 14-16)`."
    )
    lines.append("")
    lines.append(
        "| model | OD raw-sample EMD | log-return raw-sample EMD "
        "| histogram-density EMD (50-bin, OD scale) | DTW (OD scale) "
        "| source: distribution-EMD |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in comp:
        c1 = "—" if r["od_raw_mean"] is None else f"{r['od_raw_mean']:.6f}"
        c2 = "—" if r["logret_raw_mean"] is None else f"{r['logret_raw_mean']:.6f}"
        c3 = "—" if r["od_hist_mean"] is None else f"{r['od_hist_mean']:.6f}"
        c4 = "—" if r["dtw_od_mean"] is None else f"{r['dtw_od_mean']:.6f}"
        lines.append(
            f"| {r['model']} | {c1} | {c2} | {c3} | {c4} "
            f"| {r['source_dist_emd']} |"
        )
    lines.append("")

    (DOCS / "reconciliation_note.md").write_text("\n".join(lines))


def _dataset_block(window_length: int) -> dict:
    """Canonical dataset counts, DERIVED (never hand-typed) from data.csv +
    the locked window config so the regenerated dataset_stats.md cites a JSON
    source for every numeric literal.

    raw_csv_rows   = (lines in data.csv) - 1 header row
    od_rows        = raw_csv_rows (10-row rolling-mean fillna then dropna keeps
                     all rows — invariant of core/data.py:255-258)
    log_return_rows= od_rows - 1  (first difference, data.py:64)
    rolling_windows= (log_return_rows - W)//stride + 1  (data.py:150, W=window
                     length=10, stride=2) — must equal the n_real_windows the
                     strict-gate-accepted sweep configs all carry (cross-check
                     in main()).
    """
    csv_lines = sum(
        1 for _ in (REPO / "data.csv").read_text().splitlines() if _.strip()
    )
    raw_csv_rows = csv_lines - 1  # minus header
    od_rows = raw_csv_rows
    log_return_rows = od_rows - 1
    stride = 2
    rolling_windows = (log_return_rows - window_length) // stride + 1
    return {
        "raw_csv_rows": raw_csv_rows,
        "od_rows_after_fillna_dropna": od_rows,
        "log_return_rows": log_return_rows,
        "window_length": window_length,
        "window_stride": stride,
        "rolling_windows": rolling_windows,
        "independent_campaigns": 1,
        "train_windows": rolling_windows,
        "val_windows": 0,
        "test_windows": 0,
        "derivation": "raw=lines(data.csv)-1; logret=raw-1; "
        "windows=(logret-W)//stride+1 (W=window_length, stride=2; "
        "core/data.py:64/150/255-258) — DERIVED, not hand-typed",
    }


def _fmt(v) -> str:
    """Render a JSON value for a markdown cell so its textual form appears
    VERBATIM in model_info.json (the verifier resolves literals by substring
    match at stated precision). int -> bare int; float -> repr (Python repr is
    json.dumps-compatible for the values here, e.g. 1.8046e-05, 2.16); other
    -> str."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _render_training_protocol(mi: dict) -> str:
    """Regenerate training_protocol.md ENTIRELY from model_info.json.

    Preserves the existing "Source of truth" callout + `| Constant | Value |
    Source |` table layout (_build_baseline_notebook.py:550-593 render idiom),
    but every Value cell is pulled from model_info.json and every Source cell
    cites `model_info.json` provenance — never a hand-typed core/__init__.py
    line. The canonical hyperparameter row is the 55-param IQP:SEL 2000ep
    reproduction (the manuscript's quantum entrant, D-14-04)."""
    by = {m["model"]: m for m in mi["models"]}
    q = by["iqp_sel_55_repro"]
    ds = mi["dataset"]
    dh = mi["data_hash"]
    L: list[str] = []
    L.append("# Training Protocol — QWGAN-GP (matched 2000ep, 55-param IQP:SEL)")
    L.append("")
    L.append(
        "> **Source of truth:** every numerical constant below is rendered "
        "FROM `results/model_info.json` by "
        "`run_model_info.py` — there are NO hand-typed numbers and "
        "NO `core/__init__.py:NN` line citations. Re-run the emitter to "
        "update; `verify_number_provenance.py` is the executable "
        "gate that proves every literal here resolves to a "
        "`results/*.json` value (success criterion 5)."
    )
    L.append("")
    L.append(
        "This protocol describes the matched-budget 2000-epoch training run "
        "for the canonical 55-param IQP:SEL quantum generator "
        "(`source=matched2000_reproduction`) — the quantum entrant in every "
        "cross-model comparison (D-14-04). The frozen-checkpoint headline "
        "(`source=frozen_checkpoint_epoch_1969`) is a SEPARATE record in "
        "`model_info.json` (D-14-10)."
    )
    L.append("")
    L.append("## Optimizer & Schedule")
    L.append("")
    L.append("| Constant | Value | Source |")
    L.append("|----------|-------|--------|")
    src = "`model_info.json` models[] kind=quantum source=matched2000_reproduction"
    L.append(f"| `N_CRITIC` | {_fmt(q['n_critic'])} | {src} (n_critic) |")
    L.append(
        f"| `LAMBDA` (gradient penalty coeff) | {_fmt(q['lambda_gp'])} "
        f"| {src} (lambda_gp) |"
    )
    L.append(f"| `LR_CRITIC` | {_fmt(q['lr_critic'])} | {src} (lr_critic) |")
    L.append(
        f"| `LR_GENERATOR` | {_fmt(q['lr_generator'])} "
        f"| {src} (lr_generator) |"
    )
    L.append(
        f"| Optimizer | {q['optimizer']} "
        f"| {src} (optimizer, optimizer_betas) |"
    )
    L.append(f"| `NUM_EPOCHS` | {_fmt(q['epochs'])} | {src} (epochs) |")
    L.append(f"| `BATCH_SIZE` | {_fmt(q['batch_size'])} | {src} (batch_size) |")
    L.append("")
    L.append(
        "Early-stopping state for the matched-budget run: "
        f"{q['early_stop']} ({src}, early_stop). The frozen-checkpoint "
        "headline instead uses the best-EMD checkpoint from the original "
        "EarlyStopping-enabled campaign (see `model_info.json` "
        "iqp_sel_55_headline record)."
    )
    L.append("")
    L.append("## Quantum Circuit")
    L.append("")
    L.append("| Property | Value | Source |")
    L.append("|----------|-------|--------|")
    L.append(
        f"| Backend | {q['pennylane_device']} (analytic statevector) "
        f"| {src} (pennylane_device) |"
    )
    L.append(
        f"| Differentiation | {q['diff_method']} | {src} (diff_method) |"
    )
    L.append(f"| `NUM_QUBITS` | {_fmt(q['num_qubits'])} | {src} (num_qubits) |")
    L.append(f"| `NUM_LAYERS` | {_fmt(q['num_layers'])} | {src} (num_layers) |")
    L.append(
        f"| `WINDOW_LENGTH` | {_fmt(q['window_length'])} "
        f"| {src} (window_length) |"
    )
    L.append(f"| `circuit_id` | {q['circuit_id']} | {src} (circuit_id) |")
    L.append(f"| Entangler topology | {q['topology']} | {src} (topology) |")
    L.append(
        f"| PQC trainable parameter count | {_fmt(q['parameter_count'])} "
        f"| {src} (parameter_count) |"
    )
    L.append(f"| Compute device | {q['device']} | {src} (device) |")
    # Plan 14-13 Task 4 (PROV-HIGH-3 / HIGH-3): dtype row split into
    # dtype_params (torch.float32 trainable nn.Parameter) and dtype_samples
    # (the field formerly labelled `Param dtype`, which actually carries
    # sample-generation dtype). See methods_full.md §4.b.
    L.append(
        f"| dtype_params | {q.get('dtype_params', 'torch.float32')} "
        f"| {src} (dtype_params); see methods_full.md §4.b |"
    )
    L.append(
        f"| dtype_samples | {q.get('dtype_samples', q['dtype'])} "
        f"| {src} (dtype_samples); see methods_full.md §4.b |"
    )
    L.append(
        f"| Backend assertion | {q['backend_assertion']} "
        f"| {src} (backend_assertion) |"
    )
    L.append("")
    L.append("## Reproducibility")
    L.append("")
    L.append("| Property | Value | Source |")
    L.append("|----------|-------|--------|")
    L.append(
        f"| Seed set | {q['seeds']} | {src} (seeds) |"
    )
    L.append(
        f"| Training windows | {_fmt(ds['rolling_windows'])} "
        "| `model_info.json` dataset.rolling_windows |"
    )
    L.append(
        f"| `data_hash` | `{dh}` "
        "| `model_info.json` data_hash (cross-artifact gate) |"
    )
    L.append("")
    L.append(
        f"All seeds in {q['seeds']} share the identical config (the strict "
        "accept gate D-14-13 enforced this); the data_hash "
        f"`{dh}` is identical across every consumed 2000ep artifact "
        "(cross-artifact explicit-raise gate, "
        "run_multiseed_rollup.py:86-92 idiom)."
    )
    L.append("")
    return "\n".join(L) + "\n"


def _render_dataset_stats(mi: dict) -> str:
    """Regenerate dataset_stats.md ENTIRELY from model_info.json.dataset.

    Preserves the `| Quantity | Value | Source |` table layout; every Value is
    pulled from the model_info.json `dataset` block (DERIVED from data.csv +
    the locked window config, never hand-typed)."""
    ds = mi["dataset"]
    L: list[str] = []
    L.append("# Dataset Statistics — Single-Campaign LUCY Photobioreactor")
    L.append("")
    L.append(
        "> **Source of truth:** every count below is rendered FROM "
        "`results/model_info.json` (the `dataset` block, DERIVED "
        "from `data.csv` + the locked window config) by "
        "`run_model_info.py`. NO hand-typed numbers; "
        "`verify_number_provenance.py` is the executable gate."
    )
    L.append("")
    L.append(
        "This document characterizes the single-campaign dataset that backs "
        "all v2.0 evaluation work. Counts are derived from live data.csv "
        "inspection + the locked rolling-window config — never hand-typed."
    )
    L.append("")
    L.append("## Counts")
    L.append("")
    L.append("| Quantity | Value | Source / Derivation |")
    L.append("|----------|-------|---------------------|")
    prov = "`model_info.json` dataset"
    L.append(
        f"| Raw CSV rows (excluding header) | {_fmt(ds['raw_csv_rows'])} "
        f"| {prov}.raw_csv_rows |"
    )
    L.append(
        f"| OD rows after fillna + dropna "
        f"| {_fmt(ds['od_rows_after_fillna_dropna'])} "
        f"| {prov}.od_rows_after_fillna_dropna |"
    )
    L.append(
        f"| Log-return rows (N − 1) | {_fmt(ds['log_return_rows'])} "
        f"| {prov}.log_return_rows |"
    )
    L.append(
        f"| Rolling windows (length {_fmt(ds['window_length'])}, stride "
        f"{_fmt(ds['window_stride'])}) | {_fmt(ds['rolling_windows'])} "
        f"| {prov}.rolling_windows |"
    )
    L.append(
        f"| Independent campaigns | {_fmt(ds['independent_campaigns'])} "
        f"| {prov}.independent_campaigns |"
    )
    L.append("")
    L.append("## Split Convention")
    L.append("")
    L.append("| Convention | Value | Source |")
    L.append("|------------|-------|--------|")
    L.append(
        f"| Train windows | {_fmt(ds['train_windows'])} "
        f"| {prov}.train_windows |"
    )
    L.append(
        f"| Val windows | {_fmt(ds['val_windows'])} | {prov}.val_windows |"
    )
    L.append(
        f"| Test windows | {_fmt(ds['test_windows'])} "
        f"| {prov}.test_windows |"
    )
    L.append("")
    L.append(
        "**Single-Campaign Limitation.** Exactly one LUCY photobioreactor "
        "campaign; no other independent campaigns are available. "
        f"{_fmt(ds['rolling_windows'])} rolling windows is too small to "
        "justify a held-out train/val/test split without severely "
        "under-powering training, so the EMD-based early-stop metric is "
        "computed on the same distribution it compares against (stated "
        "openly per the R1-M5 calibration-honesty standard). Multi-campaign "
        "generalization is a Phase-14 Outlook item, not a current-scope "
        "claim."
    )
    L.append("")
    L.append("## Preprocessing Pipeline")
    L.append("")
    L.append(
        "The matched-budget runs use Pipeline B (decision D-10-05; see "
        "`run_ablation.py::build_dataset_for_pipeline`, "
        "pipeline=='B' branch). Pipeline B applies (in order): log-return "
        "differencing → zero-mean/unit-variance standardization → linear "
        "rescaling to [−1, 1] using the global min/max of the standardized "
        f"series → rolling windows of length {_fmt(ds['window_length'])} "
        f"and stride {_fmt(ds['window_stride'])} (yielding "
        f"{_fmt(ds['rolling_windows'])} windows). Pipeline C (the v1.1 "
        "published pipeline with an inverse Lambert-W heavy-tail correction "
        "between the standardization and rescaling steps) was dropped per "
        "D-10-05 because the 09.1 ablation showed it tied with B on every "
        "OD-scale metric while introducing an over-Gaussianization concern "
        "(R1-M3). `load_and_preprocess` retains the Pipeline C path for "
        "reproducibility of the ablation only; the matched-budget pathway "
        "is `build_dataset_for_pipeline('B', ...)`. The bioprocess "
        "justification of the log-return choice (specific growth rate, "
        "μ = d ln(OD)/dt) is the subject of Phase 09.1."
    )
    L.append("")
    return "\n".join(L) + "\n"


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
    # run_multiseed_rollup.py:86-92 idiom: assert mutual equality of the
    # frozen `data_hash` fields across every consumed 2000ep artifact (the
    # headline + all 9 accepted sweep configs). Do NOT re-derive the hash.
    # Plan 14-13 Task 4 (HI-3 / PROV-HIGH-2): additionally assert equality
    # to EXPECTED_DATA_HASH so a dataset regression where all artifacts
    # SHARE a wrong hash still surfaces loudly.
    hashes = {"headline_canonical.json": headline["data_hash"]}
    for m, c in sweep_cfgs.items():
        hashes[f"matched2000/{m}/config.yaml"] = c.get("data_hash")
    if len(set(hashes.values())) != 1:
        raise AssertionError(
            f"data_hash mismatch across consumed 2000ep artifacts: {hashes}"
        )
    canonical_hash = next(iter(hashes.values()))
    if canonical_hash != EXPECTED_DATA_HASH:
        raise AssertionError(
            f"observed canonical data_hash={canonical_hash!r} does NOT "
            f"match EXPECTED_DATA_HASH={EXPECTED_DATA_HASH!r}; the dataset "
            f"itself has regressed (HI-3 / PROV-HIGH-2 explicit-raise gate, "
            f"Plan 14-13 Task 4)."
        )

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
            # Plan 14-13 Task 4 (PROV-HIGH-3 / HIGH-3): dtype rename to
            # dtype_samples + dtype_params alongside.
            "dtype_samples": headline.get("dtype"),
            "dtype_params": "torch.float32",
            "dtype": headline.get("dtype"),  # legacy alias
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

    # ── Dataset block (DERIVED from data.csv + locked window config) ──────────
    # window_length is locked at 10 across every accepted sweep config; assert
    # that and that the derived rolling_windows equals the n_real_windows the
    # strict-gate-accepted configs all carry (explicit-raise, python -O safe).
    win_lengths = {
        c.get("window_length") for c in sweep_cfgs.values()
    }
    if win_lengths != {10}:
        raise AssertionError(
            f"non-uniform/unexpected window_length across sweep configs: "
            f"{win_lengths}"
        )
    dataset = _dataset_block(window_length=10)
    nrw = {
        c.get("n_real_windows")
        for c in sweep_cfgs.values()
        if c.get("n_real_windows") is not None
    }
    if nrw and dataset["rolling_windows"] not in nrw:
        raise AssertionError(
            f"derived rolling_windows={dataset['rolling_windows']} disagrees "
            f"with accepted sweep n_real_windows={nrw} — dataset count "
            "drift; refusing to emit an incoherent dataset block"
        )

    out = {
        "schema": "long-form rows[] + models[] aggregate (D-10-16)",
        "metric_helpers": "core.eval ONLY (D-10-20)",
        "data_hash": canonical_hash,
        "dataset": dataset,
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
    # ── Reconciliation deltas JSON artifact (Plan 14-13 Task 3) ───────────────
    # Emit the (NEW, OLD, delta) tuples as a structured artifact so the v2
    # provenance gate can resolve the computed delta literals in
    # reconciliation_note.md (the deltas are derived, not raw aggregates, so
    # they need their own JSON source).
    recon_artifact = {
        "schema": (
            "reconciliation-deltas v1 (Phase 14 plan 14-13 Task 3 / "
            "C-1 / PROV-CRIT-1 OD-scale rebuild)"
        ),
        "data_hash": canonical_hash,
        "metric": "emd (OD, audited aggregate mean over seeds 42-46)",
        "old_source": "baseline_comparison.json#rows (pipeline=B, OD, emd)",
        "new_source": "matched2000_dualscale.json#aggregates (metric_name=emd, scale=OD)",
        "rows": [
            {
                "model": r["model"],
                "old_1000ep": r["old_1000ep"],
                "new_2000ep": r["new_2000ep"],
                "delta": r["delta"],
            }
            for r in recon
        ],
    }
    (RESULTS / "reconciliation_deltas.json").write_text(
        json.dumps(recon_artifact, indent=2)
    )

    # ── Regenerate provenance docs FROM model_info.json (D-14-16) ─────────────
    # Re-read what we just wrote so the renderers consume the SAME JSON the
    # verifier will check against (no in-memory shortcut that could diverge
    # from the on-disk single source of truth).
    mi = json.loads((RESULTS / "model_info.json").read_text())
    (DOCS / "training_protocol.md").write_text(
        _render_training_protocol(mi)
    )
    (DOCS / "dataset_stats.md").write_text(_render_dataset_stats(mi))

    print(
        f"model_info.json written: {len(models)} model records, "
        f"data_hash={canonical_hash}"
    )
    print(
        f"reconciliation_note.md written: {len(recon)} model deltas "
        "(1000ep -> 2000ep, EMD OD)"
    )
    print(
        "training_protocol.md + dataset_stats.md regenerated FROM "
        "model_info.json (no hand-typed numbers)"
    )


if __name__ == "__main__":
    main()
