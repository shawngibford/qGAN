"""Phase 14 plan 14-15 — histogram-density distribution-EMD emitter (NEW).

Reintroduces the pre-v1.0 paper's 50-bin histogram-density Wasserstein
"distribution EMD" as a comparable column alongside the v1.0 raw-sample
EMD already present in ``results/matched2000_dualscale.json``.

The pre-v1.0 paper reported a 50-bin histogram-density Wasserstein. The
v1.0 release (`core/eval.py:25-36`) switched to a raw-sample
Wasserstein over the raw log-return arrays. The 14-13 T3 C-3 disclosure
paragraph in `docs/reconciliation_note.md` acknowledged the two
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

This emitter lives at the repo TOP LEVEL (NOT in ``core/``)
to preserve D-14-22 (the ``core/`` byte-freeze) — the v1.0
raw-sample ``compute_emd`` at ``core/eval.py:25-36`` is
byte-untouched. See ``.planning/phases/14-paper-revision-release-freeze/14-15-PLAN.md``
for the full plan.

Render-only / aggregator-only: reads existing per-seed sample bundles
from ``results/matched2000/runs/{model}/{seed}/samples.npy``;
NO retraining, NO resampling, NO new metric recomputation against the
v1.0 raw-sample metric. The new aggregator JSON is auto-walked into
the v2.1 number-provenance gate's resolution corpus by the existing
``_json_blobs()`` walker in ``verify_number_provenance.py``
(no gate edit; D-14-16 byte-freeze preserved).

**Plan 14-16 r3 remediation.** The original Plan 14-15 formulation used
``np.histogram(..., density=True)``, which renormalizes each histogram
independently OVER THE IN-RANGE PORTION ONLY — silently dropping
out-of-range fake mass AND re-normalizing the remainder. This rewards
narrow-collapse distributions (e.g., VAE posterior collapse, std=0.0004)
and uncapped distributions (e.g., WGAN-CNN with 94% out-of-range mass at
log-return scale) — see R3-CR-1 mechanism in
``peer-review-r3/code-review-r3.md`` §H3. The v2 formulation (this plan)
computes both histograms with ``density=False`` over edges DERIVED FROM
REAL ONLY, then normalizes both to total-mass=1 OVER THE SAME EDGE SET
(no per-distribution renormalization). Out-of-range fake mass is disclosed
separately as ``fake_in_range_mass = fake_hist.sum() / len(fake)``,
surfacing the WGAN-CNN truncation that v1 silently dropped.

**Plan 14-16 R3-HI-1 sister-fix (Step E2).** Per ``code-review-r3.md``
§H3, R3-HI-1 names BOTH ``run_matched2000_dualscale.py:368-372`` (R3-CR-2,
closed by Plan 14-16 T1) AND ``run_distribution_emd.py:_real_references``
+ ``_fake_log_return_flat`` at ``:144-169`` as edit sites under the SAME
finding ID. T2 Step E2 applies the analogous ``norm_log_delta =
(log_delta - mu) / sigma`` substitution to the LR-scale real reference in
``_real_references``, so the LR-scale rows in ``distribution_emd.json`` v2
consume a standardized real reference against the already-standardized
``r_norm`` fake side. The LR-scale aggregates in v2 are therefore at the
corrected magnitude — they are NOT inherited-broken. The OD-scale path is
byte-untouched by Step E2 (OD inverse cumsum-exps to OD price scale).
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
        if (cand / "core" / "preprocessing.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("repo root not found for sys.path bootstrap")


_REPO = _bootstrap_repo_on_path()

# Reuse MODEL_ORDER + reconstruct_od from the figure suite (single source of
# truth — do NOT duplicate the implementation here).
from run_figure_suite import (  # noqa: E402
    MODEL_ORDER,
    SEEDS,
    reconstruct_od,
)
from core.data import load_and_preprocess, rolling_window  # noqa: E402
import torch  # noqa: E402
from core._wgan_unscale import _unscale_wgan_samples  # noqa: E402  Plan 14-21 T01

WINDOW_LENGTH = 10
DATA_CSV = "data.csv"
DATA_HASH = "91e447d4624e25b3"  # corpus-consistent (matches matched2000_dualscale)
RESULTS_REL = Path("results")
OUT_REL = RESULTS_REL / "distribution_emd.json"
HEADLINE_MODEL_KIND = "iqp_sel_55_headline"  # n=1, included only if samples.npy present

# Schema string referenced by paper-facing docs (Plan 14-16, r3-remediated).
SCHEMA = "distribution-emd v2 (Phase 14 plan 14-16)"

# The metric formulation citation. MUST match the C-3 disclosure citation in
# `docs/reconciliation_note.md` word-for-word.
METRIC_FORMULATION = (
    "np.histogram(real, bins=50, density=False) -> edges from real; "
    "np.histogram(fake, bins=edges, density=False); "
    "both histograms normalized to total-mass=1 over the same edge set "
    "(no per-distribution renormalization); "
    "bin_centers = 0.5*(edges[:-1] + edges[1:]); "
    "scipy.stats.wasserstein_distance(bin_centers, bin_centers, real_norm, fake_norm); "
    "out-of-range fake mass disclosed separately as fake_in_range_mass = fake_hist.sum() / len(fake)"
)


def compute_histogram_density_emd(
    real: np.ndarray, fake: np.ndarray, n_bins: int = 50
) -> tuple[float, float]:
    """Plan 14-16 r3-remediated histogram-density EMD.

    Shared-edges-from-real + total-mass=1 normalization over the same edge
    set (NO per-distribution renormalization). Out-of-range fake mass
    disclosed separately as fake_in_range_mass.

    Returns (emd, fake_in_range_mass) where:
    - emd is the wasserstein_distance between bin_centers weighted by the
      normalized histograms (sharing edges from real).
    - fake_in_range_mass is fake_hist.sum() / len(fake) — the fraction of
      fake samples that fell within real's empirical range. 1.0 = no
      truncation; 0.06 = 94% out-of-range (e.g., WGAN-CNN log-return scale
      per code-review-r3.md §H3).

    Fixes R3-CR-1 from peer-review-r3/code-review-r3.md §H3: the prior
    density=True formulation silently dropped out-of-range fake mass AND
    re-normalized each histogram independently, rewarding narrow-collapse
    distributions (VAE) and uncapped distributions (WGAN-CNN).
    """
    from scipy.stats import wasserstein_distance

    real = np.asarray(real).ravel().astype(np.float64)
    fake = np.asarray(fake).ravel().astype(np.float64)
    if real.size == 0 or fake.size == 0:
        raise ValueError(
            "compute_histogram_density_emd: empty input "
            f"(real.size={real.size}, fake.size={fake.size})"
        )
    real_hist, edges = np.histogram(real, bins=n_bins, density=False)
    fake_hist, _ = np.histogram(fake, bins=edges, density=False)
    fake_in_range_mass = float(fake_hist.sum() / len(fake))
    real_norm = (
        real_hist / real_hist.sum()
        if real_hist.sum() > 0
        else real_hist.astype(np.float64)
    )
    fake_norm = (
        fake_hist / fake_hist.sum()
        if fake_hist.sum() > 0
        else fake_hist.astype(np.float64)
    )
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    emd = float(
        wasserstein_distance(bin_centers, bin_centers, real_norm, fake_norm)
    )
    return emd, fake_in_range_mass


def _real_references(repo: Path) -> dict:
    """Real OD-scale + log-return references.

    Plan 14-16 R3-HI-1 sister-fix: also returns ``norm_log_delta``, the
    standardized log-return reference, so the LR-scale histogram-density EMD
    compares standardized-vs-standardized (matching the already-standardized
    ``r_norm`` fake side). The raw ``real_log_delta`` is preserved for any
    other caller. ``mu``/``sigma`` are global standardization constants
    computed once from the full real series (verified cross-seed-identical
    in Plan 14-16 T1 Step A); read here from one representative
    ``inverse_kwargs.npz``.
    """
    d_real = load_and_preprocess(str(repo / DATA_CSV))
    real_windowed_OD = (
        rolling_window(d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy()
    )
    raw_log_delta = d_real["log_delta"].cpu().numpy().ravel()
    inv_path = (
        repo / "results" / "matched2000" / "runs"
        / "iqp_sel_55_repro" / "42" / "inverse_kwargs.npz"
    )
    inv = np.load(inv_path, allow_pickle=True)
    mu = float(inv["mu"])
    sigma = float(inv["sigma"])
    norm_log_delta = (raw_log_delta - mu) / sigma
    return {
        "real_OD_flat": real_windowed_OD.reshape(-1),
        "real_log_delta": raw_log_delta,        # PRESERVED (other callers)
        "norm_log_delta": norm_log_delta,       # NEW (standardized LR ref)
        "mu_logret": mu,                        # provenance disclosure
        "sigma_logret": sigma,                  # provenance disclosure
    }


def _fake_log_return_flat(repo: Path, model: str, seed: int) -> np.ndarray:
    """Reconstruct fake log-return-scale flattened array.

    Per the figure suite's transformed-space construction at lines 2384-2388
    (samples are stored on [-1, 1] and rescaled to the per-seed [r_min, r_max]
    log-return interval) — same idiom; no retraining; no resampling.
    """
    base = repo / "results" / "matched2000" / "runs" / model / str(seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model)  # Plan 14-21 T01
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)
    r_min = float(inv["r_min"])
    r_max = float(inv["r_max"])
    r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
    return r_norm.reshape(-1)


def _model_seed_rows(repo: Path, real_refs: dict) -> tuple[list[dict], bool]:
    """Build the per-(model, seed, scale) rows list + headline_present flag."""
    rows: list[dict] = []
    real_OD_flat = real_refs["real_OD_flat"]
    real_logret = real_refs["norm_log_delta"]  # Plan 14-16 R3-HI-1 sister-fix

    for model in MODEL_ORDER:
        for seed in SEEDS:
            base = repo / "results" / "matched2000" / "runs" / model / str(seed)
            if not (base / "samples.npy").exists():
                # Skip missing seeds quietly (the figure suite would loud-fail;
                # here we are aggregating and want the row absent rather than
                # an exception breaking the whole emit).
                continue
            od = reconstruct_od(repo, model, seed)
            fake_OD_flat = od.reshape(-1)
            fake_logret_flat = _fake_log_return_flat(repo, model, seed)

            emd_OD, fim_OD = compute_histogram_density_emd(
                real_OD_flat, fake_OD_flat, n_bins=50
            )
            emd_logret, fim_logret = compute_histogram_density_emd(
                real_logret, fake_logret_flat, n_bins=50
            )
            rows.append({
                "model_kind": model,
                "seed": int(seed),
                "scale": "OD",
                "value": emd_OD,
                "fake_in_range_mass": fim_OD,
            })
            rows.append({
                "model_kind": model,
                "seed": int(seed),
                "scale": "log_return",
                "value": emd_logret,
                "fake_in_range_mass": fim_logret,
            })

    # Optional headline (n=1) — present only if samples.npy is on disk.
    headline_present = False
    head_dir = repo / "results" / "matched2000" / "runs" / HEADLINE_MODEL_KIND
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
                emd_OD, fim_OD = compute_histogram_density_emd(
                    real_OD_flat, fake_OD_flat, n_bins=50
                )
                emd_logret, fim_logret = compute_histogram_density_emd(
                    real_logret, fake_logret_flat, n_bins=50
                )
                rows.append({
                    "model_kind": HEADLINE_MODEL_KIND,
                    "seed": int(cand_seed),
                    "scale": "OD",
                    "value": emd_OD,
                    "fake_in_range_mass": fim_OD,
                })
                rows.append({
                    "model_kind": HEADLINE_MODEL_KIND,
                    "seed": int(cand_seed),
                    "scale": "log_return",
                    "value": emd_logret,
                    "fake_in_range_mass": fim_logret,
                })
                headline_present = True
                break

    return rows, headline_present


def _aggregate(rows: list[dict]) -> list[dict]:
    """Per-(model_kind, scale) mean / std (ddof=1) / n aggregator."""
    groups: dict[tuple[str, str], list[float]] = {}
    fim_groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (r["model_kind"], r["scale"])
        groups.setdefault(key, []).append(float(r["value"]))
        fim_groups.setdefault(key, []).append(
            float(r.get("fake_in_range_mass", 1.0))
        )
    aggs: list[dict] = []
    for (model_kind, scale), vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = int(arr.size)
        mean = float(np.mean(arr))
        # ddof=1 sample standard deviation (matches 14-13 T3 convention; H1-2 fix)
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        fim_arr = np.asarray(fim_groups[(model_kind, scale)], dtype=np.float64)
        aggs.append({
            "model_kind": model_kind,
            "scale": scale,
            "mean": mean,
            "std": std,
            "n": n,
            "fake_in_range_mass_mean": float(np.mean(fim_arr)),
        })
    # Sort by MODEL_ORDER then scale for human readability.
    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    order_idx[HEADLINE_MODEL_KIND] = len(MODEL_ORDER)
    aggs.sort(key=lambda a: (order_idx.get(a["model_kind"], 999), a["scale"]))
    return aggs


def emit(repo: Path, out_path: Path) -> dict:
    real_refs = _real_references(repo)

    # Self-test: a sample distribution against itself has zero EMD and
    # fake_in_range_mass == 1.0 (every fake sample inside real's own range).
    self_emd, self_fim = compute_histogram_density_emd(
        real_refs["real_OD_flat"], real_refs["real_OD_flat"], n_bins=50
    )
    assert self_emd == 0.0, (
        f"self-EMD must be 0 on identical inputs; got {self_emd}"
    )
    assert self_fim == 1.0, (
        f"self fake_in_range_mass must be 1.0 on identical inputs; "
        f"got {self_fim}"
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
            "Phase 14 plan 14-15 — emit results/distribution_emd.json "
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
