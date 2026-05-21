"""Phase 14 plan 14-15 — histogram-density distribution-EMD emitter (NEW).

Reintroduces the pre-v1.0 paper's 50-bin histogram-density Wasserstein
"distribution EMD" as a comparable column alongside the v1.0 raw-sample
EMD already present in ``revision/results/matched2000_dualscale.json``.

The pre-v1.0 paper reported a 50-bin histogram-density Wasserstein. The
v1.0 release (`revision/core/eval.py:25-36`) switched to a raw-sample
Wasserstein over the raw log-return arrays. The 14-13 T3 C-3 disclosure
paragraph in `revision/docs/reconciliation_note.md` acknowledged the two
are NOT commensurate; this emitter reintroduces the deprecated metric
verbatim so the matched-2000ep numbers can be read against the
pre-v1.0 paper headline (~0.0015) under the SAME 50-bin convention
for the first time since the v1.0 raw-sample switch.

Formulation (matches the C-3 disclosure citation in
`reconciliation_note.md` word-for-word)::

    scipy.stats.wasserstein_distance(
        bin_centers, bin_centers,
        real_hist_density, fake_hist_density,
    )

over 50-bin histograms taken with ``np.histogram(..., density=True)``.
The bin edges are taken from the REAL distribution and reused for the
fake distribution so the two histograms share x-axis support.

This emitter lives at the repo TOP LEVEL (NOT in ``revision/core/``)
to preserve D-14-22 (the ``revision/core/`` byte-freeze) — the v1.0
raw-sample ``compute_emd`` at ``revision/core/eval.py:25-36`` is
byte-untouched. See ``.planning/phases/14-paper-revision-release-freeze/14-15-PLAN.md``
for the full plan.

Render-only / aggregator-only: reads existing per-seed sample bundles
from ``revision/results/matched2000/runs/{model}/{seed}/samples.npy``;
NO retraining, NO resampling, NO new metric recomputation against the
v1.0 raw-sample metric. The new aggregator JSON is auto-walked into
the v2.1 number-provenance gate's resolution corpus by the existing
``_json_blobs()`` walker in ``revision/verify_number_provenance.py``
(no gate edit; D-14-16 byte-freeze preserved).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap_repo_on_path() -> Path:
    """Find repo root and put it on sys.path (mirrors run_figure_suite)."""
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_REPO = _bootstrap_repo_on_path()

# Reuse MODEL_ORDER + reconstruct_od from the figure suite (single source of
# truth — do NOT duplicate the implementation here).
from revision.run_figure_suite import (  # noqa: E402
    MODEL_ORDER,
    SEEDS,
    reconstruct_od,
)
from revision.core.data import load_and_preprocess, rolling_window  # noqa: E402
import torch  # noqa: E402

WINDOW_LENGTH = 10
DATA_CSV = "data.csv"
DATA_HASH = "91e447d4624e25b3"  # corpus-consistent (matches matched2000_dualscale)
RESULTS_REL = Path("revision/results")
OUT_REL = RESULTS_REL / "distribution_emd.json"
HEADLINE_MODEL_KIND = "iqp_sel_55_headline"  # n=1, included only if samples.npy present

# Schema string referenced by paper-facing docs (Plan 14-15).
SCHEMA = "distribution-emd v1 (Phase 14 plan 14-15)"

# The metric formulation citation. MUST match the C-3 disclosure citation in
# `revision/docs/reconciliation_note.md` word-for-word.
METRIC_FORMULATION = (
    "scipy.stats.wasserstein_distance(bin_centers, bin_centers, "
    "real_hist_density, fake_hist_density) over 50-bin histograms "
    "(np.histogram(..., density=True))"
)


def compute_histogram_density_emd(
    real: np.ndarray, fake: np.ndarray, n_bins: int = 50
) -> float:
    """Pre-v1.0 paper metric — 50-bin histogram-density Wasserstein.

    Body matches the formulation cited in the C-3 disclosure paragraph of
    ``revision/docs/reconciliation_note.md`` (Plan 14-13 T3) verbatim:
    take the real distribution's density histogram, reuse the edges for the
    fake distribution, take the bin centers (midpoints), then call
    ``scipy.stats.wasserstein_distance(bin_centers, bin_centers,
    real_hist_density, fake_hist_density)``.

    Parameters
    ----------
    real, fake : np.ndarray
        Sample arrays (any shape; flattened internally). Must be non-empty.
    n_bins : int
        Histogram bin count. Default 50 per the pre-v1.0 convention.

    Returns
    -------
    float
        The 50-bin density Wasserstein distance.

    Notes
    -----
    Contrast with the v1.0 raw-sample EMD at ``revision/core/eval.py:25-36``
    (``scipy.stats.wasserstein_distance(real, fake)``) — same scipy
    function, different formulation, BOTH are reported side-by-side in
    ``revision/docs/reconciliation_note.md``'s 3-column comparable-variants
    table (Plan 14-15 T2). D-14-22 (`revision/core/` byte-freeze) preserved:
    this function lives in the top-level emitter, NOT in core.
    """
    from scipy.stats import wasserstein_distance

    real = np.asarray(real).ravel().astype(np.float64)
    fake = np.asarray(fake).ravel().astype(np.float64)
    if real.size == 0 or fake.size == 0:
        raise ValueError(
            "compute_histogram_density_emd: empty input "
            f"(real.size={real.size}, fake.size={fake.size})"
        )
    real_hist, edges = np.histogram(real, bins=n_bins, density=True)
    fake_hist, _ = np.histogram(fake, bins=edges, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return float(
        wasserstein_distance(bin_centers, bin_centers, real_hist, fake_hist)
    )


def _real_references(repo: Path) -> dict:
    """Real OD-scale + log-return references (verbatim with figure suite)."""
    d_real = load_and_preprocess(str(repo / DATA_CSV))
    real_windowed_OD = (
        rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()
    )
    return {
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": d_real["log_delta"].cpu().numpy().ravel(),
    }


def _fake_log_return_flat(repo: Path, model: str, seed: int) -> np.ndarray:
    """Reconstruct fake log-return-scale flattened array.

    Per the figure suite's transformed-space construction at lines 2384-2388
    (samples are stored on [-1, 1] and rescaled to the per-seed [r_min, r_max]
    log-return interval) — same idiom; no retraining; no resampling.
    """
    base = repo / "revision" / "results" / "matched2000" / "runs" / model / str(seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)
    r_min = float(inv["r_min"])
    r_max = float(inv["r_max"])
    r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
    return r_norm.reshape(-1)


def _model_seed_rows(repo: Path, real_refs: dict) -> tuple[list[dict], bool]:
    """Build the per-(model, seed, scale) rows list + headline_present flag."""
    rows: list[dict] = []
    real_OD_flat = real_refs["real_OD_flat"]
    real_logret = real_refs["real_log_delta"]

    for model in MODEL_ORDER:
        for seed in SEEDS:
            base = repo / "revision" / "results" / "matched2000" / "runs" / model / str(seed)
            if not (base / "samples.npy").exists():
                # Skip missing seeds quietly (the figure suite would loud-fail;
                # here we are aggregating and want the row absent rather than
                # an exception breaking the whole emit).
                continue
            od = reconstruct_od(repo, model, seed)
            fake_OD_flat = od.reshape(-1)
            fake_logret_flat = _fake_log_return_flat(repo, model, seed)

            emd_OD = compute_histogram_density_emd(
                real_OD_flat, fake_OD_flat, n_bins=50
            )
            emd_logret = compute_histogram_density_emd(
                real_logret, fake_logret_flat, n_bins=50
            )
            rows.append({
                "model_kind": model,
                "seed": int(seed),
                "scale": "OD",
                "value": emd_OD,
            })
            rows.append({
                "model_kind": model,
                "seed": int(seed),
                "scale": "log_return",
                "value": emd_logret,
            })

    # Optional headline (n=1) — present only if samples.npy is on disk.
    headline_present = False
    head_dir = repo / "revision" / "results" / "matched2000" / "runs" / HEADLINE_MODEL_KIND
    if head_dir.exists():
        # Use the first seed present (typically 42 or unique headline seed).
        for cand_seed in [42, *SEEDS]:
            cand = head_dir / str(cand_seed)
            if (cand / "samples.npy").exists():
                od = reconstruct_od(repo, HEADLINE_MODEL_KIND, cand_seed)
                fake_OD_flat = od.reshape(-1)
                fake_logret_flat = _fake_log_return_flat(
                    repo, HEADLINE_MODEL_KIND, cand_seed
                )
                emd_OD = compute_histogram_density_emd(
                    real_OD_flat, fake_OD_flat, n_bins=50
                )
                emd_logret = compute_histogram_density_emd(
                    real_logret, fake_logret_flat, n_bins=50
                )
                rows.append({
                    "model_kind": HEADLINE_MODEL_KIND,
                    "seed": int(cand_seed),
                    "scale": "OD",
                    "value": emd_OD,
                })
                rows.append({
                    "model_kind": HEADLINE_MODEL_KIND,
                    "seed": int(cand_seed),
                    "scale": "log_return",
                    "value": emd_logret,
                })
                headline_present = True
                break

    return rows, headline_present


def _aggregate(rows: list[dict]) -> list[dict]:
    """Per-(model_kind, scale) mean / std (ddof=1) / n aggregator."""
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (r["model_kind"], r["scale"])
        groups.setdefault(key, []).append(float(r["value"]))
    aggs: list[dict] = []
    for (model_kind, scale), vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = int(arr.size)
        mean = float(np.mean(arr))
        # ddof=1 sample standard deviation (matches 14-13 T3 convention; H1-2 fix)
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        aggs.append({
            "model_kind": model_kind,
            "scale": scale,
            "mean": mean,
            "std": std,
            "n": n,
        })
    # Sort by MODEL_ORDER then scale for human readability.
    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    order_idx[HEADLINE_MODEL_KIND] = len(MODEL_ORDER)
    aggs.sort(key=lambda a: (order_idx.get(a["model_kind"], 999), a["scale"]))
    return aggs


def emit(repo: Path, out_path: Path) -> dict:
    real_refs = _real_references(repo)

    # Self-test: a sample distribution against itself has zero EMD.
    self_emd = compute_histogram_density_emd(
        real_refs["real_OD_flat"], real_refs["real_OD_flat"], n_bins=50
    )
    assert self_emd == 0.0, (
        f"self-EMD must be 0 on identical inputs; got {self_emd}"
    )

    rows, headline_present = _model_seed_rows(repo, real_refs)
    aggs = _aggregate(rows)

    payload = {
        "schema": SCHEMA,
        "metric_formulation": METRIC_FORMULATION,
        "data_hash": DATA_HASH,
        "n_bins": 50,
        "headline_present": headline_present,
        "rows": rows,
        "aggregates": aggs,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 14 plan 14-15 — emit revision/results/distribution_emd.json "
            "(50-bin histogram-density Wasserstein EMD)"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit to a temp path and verify shape without overwriting "
             "the canonical JSON.",
    )
    args = parser.parse_args()

    repo = _REPO
    if args.dry_run:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        payload = emit(repo, tmp)
        print(
            f"[run_distribution_emd] dry-run wrote {tmp} "
            f"(rows={len(payload['rows'])}, aggregates={len(payload['aggregates'])}, "
            f"headline_present={payload['headline_present']})"
        )
        return

    out_path = repo / OUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = emit(repo, out_path)
    print(
        f"[run_distribution_emd] wrote {out_path} "
        f"(rows={len(payload['rows'])}, aggregates={len(payload['aggregates'])}, "
        f"headline_present={payload['headline_present']}, "
        f"data_hash={payload['data_hash']})"
    )


if __name__ == "__main__":
    main()
