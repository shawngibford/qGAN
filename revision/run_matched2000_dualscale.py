"""CLI driver: matched-2000ep dual-scale fidelity re-emit (Phase 14, plan 14-08).

The matched-2000ep COUNTERPART of ``revision/run_dualscale_fidelity.py``
(which is the OLD 1000-epoch / 60-frozen-Phase-11-bundle artifact). This
driver is a pure RE-EVALUATION + render: it recomputes every
``revision.core.eval`` fidelity metric (EMD, moments, ACF, DTW) at BOTH the
``OD`` and ``log_return`` scales for the 45 already-saved
``revision/results/matched2000/runs/<model>/<seed>/samples.npy`` bundles
(9 models x seeds 42-46, all accepted by the 14-02 strict gate) PLUS the
frozen-checkpoint headline, then emits
``revision/results/matched2000_dualscale.json`` — the single source of truth
for these matched-budget dual-scale numbers.

One process per invocation. Idempotent — re-running overwrites
``revision/results/matched2000_dualscale.json`` cleanly. NO retraining, NO
resampling, NO checkpoint reload (14-02 owns the headline checkpoint reload):
the 45 ``samples.npy`` / ``inverse_kwargs.npz`` bundles are read-only and the
frozen headline rows are re-emitted VERBATIM from
``revision/results/headline_canonical.json``.

The reconstruction + scale-tagged metric math is COPIED VERBATIM from the
audited ``run_dualscale_fidelity.py`` (D-11-10: ``revision.core.eval`` is
imported UNCHANGED, never re-implemented); only what the matched2000 layout
requires is changed:
  (1) the run-dir resolver -> ``matched2000/runs/<model>/<seed>`` (loud-fail
      ``FileNotFoundError`` on any absent ``samples.npy`` / ``inverse_kwargs``);
  (2) ``MODEL_KINDS`` -> the 9 matched2000 models, all Pipeline-B ONLY (verified
      on disk — there is NO Pipeline-A matched2000 run, so no Pipeline-A null
      rows are fabricated);
  (3) an explicit-raise cross-artifact ``data_hash`` gate (python -O safe);
  (4) the frozen-checkpoint headline carried as its OWN distinct row-set
      (``model_kind="frozen_checkpoint_headline"`` + explicit
      ``source="frozen_checkpoint_epoch_1969"``) — NEVER merged into
      ``iqp_sel_55_repro`` (D-14-10);
  (5) a per-(model_kind, scale, metric_name) seed-aggregate
      (mean/std over the 5 matched2000 seeds, n_seeds=5; the frozen headline's
      aggregate is n_seeds=1, single generation seed).

Key contracts:
    D-11-08 — no regeneration; frozen artifacts only.
    D-11-10 — ``revision.core.eval`` reused UNCHANGED (imported, never edited);
              ``git diff --stat revision/core/`` must be empty.
    D-14-08 — no retraining/resampling; render of saved samples only.
    D-14-10 — frozen-checkpoint headline is a DISTINCT, never-merged row-set.
    D-14-16 — cross-artifact data_hash equality enforced by an explicit raise.

Usage
-----
    python -m revision.run_matched2000_dualscale \\
        [--out revision/results] [--csv-path ./data.csv]

Writes ``revision/results/matched2000_dualscale.json`` (long-form dual-scale
rows[] + per-(model,scale,metric) seed-aggregate + a distinct frozen-headline
row-set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import sys

import numpy as np
import torch


def _bootstrap_repo_on_path() -> Path:
    """Ensure the repo root is importable when run as a bare script.

    Verbatim with ``run_dualscale_fidelity.py:69-83`` so both ``-m`` and
    bare-script invocation work (the plan's verify command uses the latter).
    """
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_bootstrap_repo_on_path()

from revision.core import WINDOW_LENGTH
from revision.core.data import load_and_preprocess, rolling_window

# revision.core.eval helpers — REUSE UNCHANGED (D-11-10). Import only; the
# eval module is NEVER edited by this driver.
from revision.core.eval import (
    compute_acf,
    compute_dtw,
    compute_emd,
    compute_moments,
)
from revision.core.preprocessing import inverse_logreturns

# ── Constants ────────────────────────────────────────────────────────────────
# The 9 matched-2000ep models (verified on disk). ALL are Pipeline-B (no
# Pipeline-A matched2000 run exists — do NOT fabricate Pipeline-A null rows).
MODEL_KINDS = [
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
PIPELINES = ["B"]  # matched2000 sweep is Pipeline-B only (verified)
SEEDS = [42, 43, 44, 45, 46]
EXPECTED_DATA_HASH = "91e447d4624e25b3"
NLAGS = 9  # window length 10 -> max 9 lags + lag 0 (run_dualscale_fidelity:106)
DTW_N_PAIRS = 100  # verbatim with run_dualscale_fidelity (bounded runtime)

MATCHED2000_RUNS_REL = Path("revision/results/matched2000/runs")
HEADLINE_REL = Path("revision/results/headline_canonical.json")

# The frozen-checkpoint headline is carried under this DISTINCT model_kind and
# explicit source so it is numerically + visually separable from the 2000ep
# iqp_sel_55_repro reproduction and is NEVER merged (D-14-10).
HEADLINE_MODEL_KIND = "frozen_checkpoint_headline"
HEADLINE_SOURCE = "frozen_checkpoint_epoch_1969"


def _find_repo_root() -> Path:
    """Walk up until ``revision/core/preprocessing.py`` is found."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from load_and_preprocess.

    Identical contract to ``run_dualscale_fidelity._compute_data_hash`` — pins
    the metric run to the exact OD tensor the frozen samples trained against.
    """
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _run_base(model_kind: str, pipeline: str, seed: int) -> Path:
    """Resolve the matched2000 run dir for (model_kind, pipeline, seed).

    The ONLY thing that changes vs ``run_dualscale_fidelity._run_base`` is the
    layout: ``revision/results/matched2000/runs/<model>/<seed>``. ``pipeline``
    is always "B" for matched2000 (verified). Loud-fail (``FileNotFoundError``)
    with a no-retraining/resampling message if either ``samples.npy`` or
    ``inverse_kwargs.npz`` is absent — NEVER a silent partial aggregate
    (T-14-14).
    """
    repo = _find_repo_root()
    base = repo / MATCHED2000_RUNS_REL / model_kind / str(seed)
    for fname in ("samples.npy", "inverse_kwargs.npz"):
        if not (base / fname).exists():
            raise FileNotFoundError(
                f"matched2000 bundle missing: {base / fname} "
                f"(model={model_kind}, pipeline={pipeline}, seed={seed}). "
                f"D-14-08 forbids retraining/resampling — every number must "
                f"come from an already-saved samples.npy bundle. Re-run the "
                f"14-02 sweep harness (revision/run_matched2000_sweep.sh) to "
                f"regenerate this accepted bundle; this driver NEVER trains, "
                f"samples, or silently emits a partial aggregate."
            )
    return base


def reconstruct_od(model_kind: str, pipeline: str, seed: int,
                   n_synth_subsample: int | None = None) -> dict:
    """Reconstruct OD-scale windows from a frozen matched2000 sample bundle.

    Pipeline-B branch copied VERBATIM from
    ``run_dualscale_fidelity.reconstruct_od:221-236`` — the
    ``np.random.default_rng(seed*7919+1)`` od_start draw and the ``od[:, :10]``
    11->10 trim are load-bearing; do NOT change them. matched2000 is Pipeline-B
    only, so the Pipeline-A branch is intentionally absent (there is no
    Pipeline-A matched2000 run to reconstruct). Returns ``{"od_samples",
    "transformed", "n_synth", "pipeline", "seed"}``; ``transformed`` is the
    log-return array for the EVAL-05 ``scale="log_return"`` rows.
    """
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)

    if n_synth_subsample is not None and samples_pm1.shape[0] > n_synth_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(samples_pm1.shape[0], n_synth_subsample, replace=False)
        samples_pm1 = samples_pm1[idx]

    if pipeline == "B":
        r_min = float(inv["r_min"]); r_max = float(inv["r_max"])
        mu = float(inv["mu"]); sigma = float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])
        r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
        rng = np.random.default_rng(seed * 7919 + 1)
        od_start_per_window = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
        r_norm_t = torch.tensor(r_norm)
        od_start_t = torch.tensor(od_start_per_window)
        od_full = inverse_logreturns(r_norm_t, od_start_t,
                                     torch.tensor(mu), torch.tensor(sigma))
        od = od_full.cpu().numpy()
        if od.shape[1] == 11:
            od = od[:, :10]
        return {"od_samples": od, "transformed": r_norm, "n_synth": od.shape[0],
                "pipeline": pipeline, "seed": seed,
                "mu": mu, "sigma": sigma}

    raise ValueError(
        f"matched2000 is Pipeline-B only; got pipeline={pipeline} "
        f"(model={model_kind}, seed={seed}) — no Pipeline-A matched2000 run"
    )


def verify_data_hash(csv_path: Path) -> dict:
    """Recompute data_hash; explicit-raise on ANY cross-artifact mismatch.

    Cross-artifact equality gate (D-14-16, T-14-13): the recomputed
    ``load_and_preprocess(csv)`` hash AND every consumed
    ``matched2000/runs/<model>/<seed>/config.yaml`` ``data_hash`` MUST equal
    ``91e447d4624e25b3``. Uses ``raise AssertionError`` (NOT bare ``assert`` —
    python -O safe). The headline is verified separately against its own
    ``data_hash`` field. Loud-fails listing every offending (model, seed).
    """
    import yaml  # PyYAML 6.0.3 in qgan_env — pure-aggregator-safe (14-03 #1)

    recomputed = _compute_data_hash(csv_path)
    if recomputed != EXPECTED_DATA_HASH:
        raise AssertionError(
            f"recomputed data_hash {recomputed} != expected "
            f"{EXPECTED_DATA_HASH} (D-14-16 cross-artifact equality gate). "
            f"Refusing to emit matched2000_dualscale.json from a mismatched "
            f"OD tensor — this would put an unfair/fabricated number in the "
            f"resubmission (T-14-13)."
        )

    checked = 0
    mismatches: list[tuple] = []
    for m in MODEL_KINDS:
        for s in SEEDS:
            cfg_dir = _run_base(m, "B", s)
            cfg = yaml.safe_load((cfg_dir / "config.yaml").read_text())
            checked += 1
            if cfg.get("data_hash") != recomputed:
                mismatches.append((m, s, cfg.get("data_hash")))
    if checked != len(MODEL_KINDS) * len(SEEDS):
        raise AssertionError(
            f"expected {len(MODEL_KINDS) * len(SEEDS)} matched2000 configs, "
            f"checked {checked}"
        )
    if mismatches:
        raise AssertionError(
            f"matched2000 config.yaml data_hash mismatch (report loudly, "
            f"T-14-13): {mismatches} — every consumed bundle must equal "
            f"{recomputed}"
        )

    headline = json.loads((_find_repo_root() / HEADLINE_REL).read_text())
    head_hash = headline.get("data_hash")
    if head_hash != recomputed:
        raise AssertionError(
            f"headline_canonical.json data_hash {head_hash} != expected "
            f"{recomputed} (D-14-16) — the frozen-checkpoint headline must "
            f"share the byte-frozen OD tensor with the matched2000 sweep"
        )

    return {
        "recomputed_from": (
            "load_and_preprocess('data.csv')['OD'].tobytes() sha256[:16]"
        ),
        "n_matched2000_configs_checked": checked,
        "headline_data_hash_checked": True,
        "all_equal": True,
        "gate": "explicit raise AssertionError (python -O safe), D-14-16",
    }


def build_real_references(csv_path: Path) -> dict:
    """Build the real OD + log-return reference arrays.

    Copied VERBATIM from ``run_dualscale_fidelity.build_real_references`` so the
    matched2000 dual-scale numbers reconcile with the audited dual-scale recipe
    (D-11-10): ``real_windowed_OD`` (rolling_window of OD, step 2) flattened for
    the OD-scale EMD/DTW reference, and ``real_log_delta`` (= ``d_real
    ["log_delta"]``) for the Pipeline-B ``scale="log_return"`` EMD reference.
    """
    d_real = load_and_preprocess(str(csv_path))
    real_windowed_OD = rolling_window(
        d_real["OD"], WINDOW_LENGTH, 2
    ).cpu().numpy()
    real_log_delta = d_real["log_delta"].cpu().numpy()
    return {
        "real_windowed_OD": real_windowed_OD,
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": real_log_delta,
    }


def _od_scale_rows(mk: str, p: str, s: int, od: np.ndarray,
                    real_OD_flat: np.ndarray,
                    real_windowed_OD: np.ndarray,
                    source: str) -> list:
    """Emit every eval.py metric on the OD scale for one run.

    Body copied VERBATIM from ``run_dualscale_fidelity._od_scale_rows`` (same
    metric suite + DTW sub-sample recipe ``np.random.default_rng(s*31)``, 100
    synth / 64 real). NO new metric math (D-11-10). The only addition is the
    per-row ``source`` field (read from config.yaml, never hardcoded). Every
    row carries ``scale="OD"``.
    """
    out = []
    synth_flat = od.reshape(-1)
    out.append(dict(model_kind=mk, pipeline=p, seed=s, metric_name="emd",
                     scale="OD", value=compute_emd(real_OD_flat, synth_flat),
                     source=source))
    for k, v in compute_moments(synth_flat).items():
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"moment_{k}", scale="OD", value=v,
                         source=source))
    acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in od])
    for lag in range(NLAGS + 1):
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_mean", scale="OD",
                         value=float(acfs[:, lag].mean()), source=source))
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_std", scale="OD",
                         value=float(acfs[:, lag].std()), source=source))
    rng = np.random.default_rng(s * 31)
    synth_idx = rng.choice(od.shape[0],
                           size=min(DTW_N_PAIRS, od.shape[0]), replace=False)
    real_idx = rng.choice(real_windowed_OD.shape[0],
                          size=min(64, real_windowed_OD.shape[0]),
                          replace=False)
    dtw_vals = []
    for i in synth_idx:
        best = min(compute_dtw(od[i], real_windowed_OD[j]) for j in real_idx)
        dtw_vals.append(best)
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_mean", scale="OD",
                     value=float(np.mean(dtw_vals)), source=source))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_median", scale="OD",
                     value=float(np.median(dtw_vals)), source=source))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_std", scale="OD",
                     value=float(np.std(dtw_vals)), source=source))
    return out


def _log_return_rows(mk: str, p: str, s: int, r: dict,
                      real_log_delta: np.ndarray,
                      source: str) -> list:
    """Emit the SAME metric set on the log-return scale.

    Body copied VERBATIM from ``run_dualscale_fidelity._log_return_rows``
    Pipeline-B branch (real values vs ``r["transformed"]``; real reference is
    ``real_log_delta``; DTW recipe ``np.random.default_rng(s*31)``, 100 synth /
    64 real). matched2000 is Pipeline-B only, so the Pipeline-A null branch is
    intentionally absent. NO new metric math (D-11-10). The only addition is
    the per-row ``source`` field.

    Plan 14-16 r3 remediation (Path A): the log-return EMD's FAKE side is now
    un-standardized via ``trans_flat_raw = r["transformed"].reshape(-1) *
    r["sigma"] + r["mu"]`` before comparison against the unchanged raw
    ``real_log_delta`` (both sides now in raw log-return units, matching
    ``pipeline-review-r3.md`` §2 anchor table: AR 0.00294, V1 0.01497, VAE
    0.01583). The standardize-real alternative inflates EMD by ~47× and does
    NOT match the §2 anchors; the un-standardize-fake recipe per §2 lines
    124-135 is canonical. The OD-scale rows are byte-untouched. The moment /
    ACF / DTW rows are also byte-untouched (the scale mismatch was specific
    to the EMD call). See R3-CR-2 mechanism in
    ``peer-review-r3/code-review-r3.md`` §H3 and the corrected aggregates
    anchored against ``pipeline-review-r3.md`` §2.
    """
    out = []
    trans = r["transformed"]
    trans_flat = trans.reshape(-1)
    trans_flat_raw = trans_flat * r["sigma"] + r["mu"]  # Plan 14-16 R3-CR-2 fix
    out.append(dict(model_kind=mk, pipeline=p, seed=s, metric_name="emd",
                     scale="log_return",
                     value=compute_emd(real_log_delta, trans_flat_raw),
                     source=source))
    for k, v in compute_moments(trans_flat).items():
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"moment_{k}", scale="log_return",
                         value=v, source=source))
    acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in trans])
    for lag in range(NLAGS + 1):
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_mean",
                         scale="log_return",
                         value=float(acfs[:, lag].mean()), source=source))
        out.append(dict(model_kind=mk, pipeline=p, seed=s,
                         metric_name=f"acf_lag{lag}_std",
                         scale="log_return",
                         value=float(acfs[:, lag].std()), source=source))
    rng = np.random.default_rng(s * 31)
    synth_idx = rng.choice(trans.shape[0],
                           size=min(DTW_N_PAIRS, trans.shape[0]),
                           replace=False)
    wl = trans.shape[1]
    n_full = real_log_delta.shape[0] // wl
    real_lr_windows = real_log_delta[: n_full * wl].reshape(n_full, wl)
    real_idx = rng.choice(real_lr_windows.shape[0],
                          size=min(64, real_lr_windows.shape[0]),
                          replace=False)
    dtw_vals = []
    for i in synth_idx:
        best = min(compute_dtw(trans[i], real_lr_windows[j])
                   for j in real_idx)
        dtw_vals.append(best)
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_mean", scale="log_return",
                     value=float(np.mean(dtw_vals)), source=source))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_median", scale="log_return",
                     value=float(np.median(dtw_vals)), source=source))
    out.append(dict(model_kind=mk, pipeline=p, seed=s,
                     metric_name="dtw_std", scale="log_return",
                     value=float(np.std(dtw_vals)), source=source))
    return out


def _config_source(model_kind: str, seed: int) -> str:
    """Read the ``source`` field VERBATIM from this run's config.yaml.

    Never hardcoded: iqp_sel_55_repro -> matched2000_reproduction,
    V1/V2/V3 -> matched2000_ansatz, wgan_*/vae/ar -> matched2000_baseline
    (whatever the accepted config.yaml actually carries).
    """
    import yaml

    cfg_dir = _run_base(model_kind, "B", seed)
    cfg = yaml.safe_load((cfg_dir / "config.yaml").read_text())
    src = cfg.get("source")
    if not src:
        raise AssertionError(
            f"config.yaml for {model_kind}/{seed} carries no 'source' field "
            f"— refusing to emit a row with an unknown provenance"
        )
    return str(src)


def _headline_rows() -> list:
    """Re-emit the frozen-checkpoint headline rows VERBATIM as a DISTINCT set.

    Loads ``headline_canonical.json`` and re-emits its existing dual-scale rows
    (already computed by 14-02 via ``revision.core.eval``; this driver does NOT
    recompute or reload any checkpoint — 14-02 owns that). Each row is carried
    under the distinct ``model_kind="frozen_checkpoint_headline"`` AND an
    explicit ``source="frozen_checkpoint_epoch_1969"`` so the headline is a
    separate, NEVER-merged row-set with its own aggregate (D-14-10, T-14-16).
    The single generation seed is recorded so the aggregate is n_seeds=1.
    """
    headline = json.loads((_find_repo_root() / HEADLINE_REL).read_text())
    gen_seed = headline.get("generation_seed")
    if headline.get("source") != HEADLINE_SOURCE:
        raise AssertionError(
            f"headline_canonical.json source {headline.get('source')!r} != "
            f"{HEADLINE_SOURCE!r} — refusing to carry an unexpected headline "
            f"provenance (D-14-10)"
        )
    rows = []
    for hr in headline["rows"]:
        rows.append(dict(
            model_kind=HEADLINE_MODEL_KIND,
            pipeline=hr.get("pipeline", "B"),
            seed=gen_seed,
            metric_name=hr["metric_name"],
            scale=hr["scale"],
            value=hr["value"],
            source=HEADLINE_SOURCE,
        ))
    return rows


def emit_rows(real_OD_flat, real_windowed_OD, real_log_delta) -> list:
    """Scale-tagged metric loop over the 45 saved matched2000 bundles.

    For each (model_kind in MODEL_KINDS, pipeline="B", seed in SEEDS):
    reconstruct OD from the saved samples, emit every ``revision.core.eval``
    metric on ``scale="OD"`` then the same set on ``scale="log_return"``
    (real values — Pipeline B has a log-return space). Then append the
    frozen-checkpoint headline rows as a DISTINCT model_kind (D-14-10). NO new
    metric math, NO retraining, NO resampling (D-11-10 / D-14-08).
    """
    rows = []
    for mk in MODEL_KINDS:
        source = _config_source(mk, SEEDS[0])
        for s in SEEDS:
            print(f"  metrics: model={mk} pipeline=B seed={s} source={source}")
            r = reconstruct_od(mk, "B", s)
            od = r["od_samples"]
            rows.extend(
                _od_scale_rows(mk, "B", s, od, real_OD_flat,
                               real_windowed_OD, source)
            )
            rows.extend(
                _log_return_rows(mk, "B", s, r, real_log_delta, source)
            )
    head = _headline_rows()
    print(f"  headline: {len(head)} frozen-checkpoint rows (DISTINCT, D-14-10)")
    rows.extend(head)
    return rows


def build_aggregates(rows: list) -> list:
    """Per-(model_kind, scale, metric_name) mean/std over seeds.

    These are aggregations of ``revision.core.eval`` outputs (mean/std via the
    stdlib ``statistics`` module / numpy) — NOT new metric math. The 9
    matched2000 models aggregate over the 5 seeds (n_seeds=5); the frozen
    headline is a single generation seed so its aggregate is n_seeds=1 and it
    stays a DISTINCT model_kind (never merged into iqp_sel_55_repro, D-14-10).
    """
    groups: dict[tuple, list[float]] = {}
    src_of: dict[tuple, str] = {}
    seeds_of: dict[tuple, set] = {}
    for r in rows:
        if r["value"] is None:
            continue
        key = (r["model_kind"], r["scale"], r["metric_name"])
        groups.setdefault(key, []).append(float(r["value"]))
        src_of[key] = r.get("source")
        seeds_of.setdefault(key, set()).add(r["seed"])

    aggs = []
    for (mk, scale, metric), vals in sorted(groups.items()):
        n_seeds = len(seeds_of[(mk, scale, metric)])
        mean = statistics.fmean(vals)
        # Sample std (statistics.stdev = sqrt of unbiased variance, ddof=1)
        # — Plan 14-13 Task 3 H-2 remediation (was statistics.pstdev = ddof=0)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # `n` alias alongside existing `n_seeds` for paper-facing
        # convention (Plan 14-13 Task 3 MED-4 remediation): n=5 for matched
        # budget aggregates; n=1 for the headline (single frozen-checkpoint
        # eval). KEEPS `n_seeds` for backward compat with 14-08/14-12.
        n_alias = n_seeds
        aggs.append(dict(
            model_kind=mk,
            scale=scale,
            metric_name=metric,
            mean=mean,
            std=std,
            n_seeds=n_seeds,
            n=n_alias,
            source=src_of[(mk, scale, metric)],
        ))
    return aggs


def main() -> None:
    """CLI entry point — emit revision/results/matched2000_dualscale.json."""
    ap = argparse.ArgumentParser(
        description=(
            "Phase 14 plan 14-08 matched-2000ep dual-scale re-emit "
            "(reuses revision.core.eval UNCHANGED — D-11-10; no "
            "retraining/resampling — D-14-08)."
        )
    )
    ap.add_argument(
        "--out", type=Path, default=Path("revision/results"),
        help="Directory for matched2000_dualscale.json.",
    )
    ap.add_argument(
        "--csv-path", type=Path, default=Path("./data.csv"),
        help="Path to the OD CSV (same data the frozen runs used).",
    )
    args = ap.parse_args()

    repo = _find_repo_root()
    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = (repo / csv_path).resolve()

    recomputed = _compute_data_hash(csv_path)
    print(f"[run_matched2000_dualscale] recomputed data_hash: {recomputed}")
    if recomputed != EXPECTED_DATA_HASH:
        raise AssertionError(
            f"data_hash {recomputed} != expected {EXPECTED_DATA_HASH} "
            f"(D-14-16 explicit-raise gate)"
        )

    dh_verification = verify_data_hash(csv_path)
    print(
        f"[run_matched2000_dualscale] data_hash invariant OK: all "
        f"{dh_verification['n_matched2000_configs_checked']} matched2000 "
        f"configs + headline == {recomputed}"
    )

    refs = build_real_references(csv_path)
    real_OD_flat = refs["real_OD_flat"]
    real_windowed_OD = refs["real_windowed_OD"]
    real_log_delta = refs["real_log_delta"]
    print(
        f"[run_matched2000_dualscale] real refs: windowed_OD="
        f"{real_windowed_OD.shape}, log_delta={real_log_delta.shape}"
    )

    rows = emit_rows(real_OD_flat, real_windowed_OD, real_log_delta)
    aggregates = build_aggregates(rows)

    out_dir = (
        args.out if args.out.is_absolute() else (repo / args.out)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": (
            "matched-2000ep dual-scale rows[] + per-(model,scale,metric) "
            "seed-aggregate; frozen headline DISTINCT, D-14-10"
        ),
        # Plan 14-13 Task 4 (HI-5): include HEADLINE_MODEL_KIND in the
        # model_kinds top-level field. The headline IS a distinct model_kind
        # in the aggregates (D-14-10); listing it alongside the 9 sweep
        # models matches the actual rows[] / aggregates[] content. The
        # `headline_model_kind` field below remains as the explicit pointer
        # to the headline entry.
        "model_kinds": MODEL_KINDS + [HEADLINE_MODEL_KIND],
        "pipelines": PIPELINES,
        "seeds": SEEDS,
        "data_hash": EXPECTED_DATA_HASH,
        "data_hash_verification": dh_verification,
        "headline_model_kind": HEADLINE_MODEL_KIND,
        "headline_source": HEADLINE_SOURCE,
        "metric_helpers": {
            "functions": [
                "revision.core.eval.compute_emd",
                "revision.core.eval.compute_moments",
                "revision.core.eval.compute_acf",
                "revision.core.eval.compute_dtw",
            ],
            "note": "reused unchanged, D-11-10",
        },
        "rows": rows,
        "aggregates": aggregates,
    }
    out_path = out_dir / "matched2000_dualscale.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(
        f"[run_matched2000_dualscale] wrote {out_path} ({len(rows)} rows, "
        f"{len(aggregates)} aggregates, scales="
        f"{sorted({r['scale'] for r in rows})}, model_kinds="
        f"{sorted({r['model_kind'] for r in rows})})"
    )


if __name__ == "__main__":
    main()
