"""Phase 12 SENS-03 driver — the multi-seed headline roll-up aggregator.

This is a **pure consumer / pure aggregator** (D-12-03): it reads the five
frozen Phase 10/11 headline JSONs, asserts their `data_hash` fields are
mutually equal (D-10-15 hard cross-artifact gate, expected
`91e447d4624e25b3`), groups every long-form `rows[]` element by
`(source_filename, model_kind, pipeline, metric_name, scale, injection_ratio)`,
and emits `revision/results/multiseed_summary.json` with `{mean, std, n,
seeds}` per cell plus a provenance header.

NO new training, NO new seeds, NO device, NO model, NO torch, NO pennylane,
NO `revision.core` model code (D-12-03 / D-10-13). Single invocation, stdlib
only — `statistics.fmean`/`statistics.stdev` (RESEARCH Open Q3: zero new
dependency; pandas/numpy are NOT required).

`revision/core/` is NEVER modified by this driver (D-10-13 invariant).

Pattern source: `revision/run_utility.py:38-58` (verbatim repo-root resolver +
`sys.path.insert`) and 12-RESEARCH.md Code Example 4 (groupby + cross-artifact
data_hash assertion). Anti-Pattern (forbidden): re-deriving `data_hash` from
`transform_ablation` configs — only mutual equality of the five frozen
`data_hash` fields is asserted; the existing `data_hash_verification`
quantum-equivalence blocks already encode by-construction equivalence.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (copied verbatim from run_utility.py:38-58 — the driver
# may run from a worktree / arbitrary cwd; results paths are repo-root anchored).
# This driver is a pure aggregator: it imports NO torch/pennylane/core, so the
# sys.path.insert exists only to keep the resolver shape identical to peers.
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    # Fallback: walk up from cwd (covers exotic invocations).
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (revision/core/preprocessing.py)")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "revision/results"

# The five frozen Phase 10/11 headline comparison JSONs (verified row counts:
# baseline_comparison 1710, tstr 144, predictive_discriminative 120,
# augmentation 180, fidelity_dualscale 3360 — 12-RESEARCH.md Code Example 4).
HEADLINE = [
    "baseline_comparison.json",
    "tstr.json",
    "predictive_discriminative.json",
    "augmentation.json",
    "fidelity_dualscale.json",
]

# Canonical training seed set re-emitted as mean ± std (n) per headline cell.
SEED_SET = [42, 43, 44, 45, 46]


def main() -> None:
    # Load all five frozen headline JSONs.
    docs = {f: json.loads((RESULTS / f).read_text()) for f in HEADLINE}

    # ── D-10-15 cross-artifact invariant (HARD gate, before any roll-up math) ──
    # Every headline JSON must agree on `data_hash`. We assert ONLY mutual
    # equality of the five frozen `data_hash` fields — we do NOT re-derive the
    # hash from transform_ablation configs (Anti-Pattern): the existing
    # `data_hash_verification` blocks already encode by-construction equivalence.
    hashes = {f: d["data_hash"] for f, d in docs.items()}
    # WR-01: explicit raise, NOT assert — `python -O` strips asserts, which
    # would silently disable the D-10-15 cross-artifact integrity gate.
    if len(set(hashes.values())) != 1:
        raise AssertionError(
            f"data_hash mismatch across headline artifacts: {hashes}"
        )
    canonical_hash = next(iter(hashes.values()))  # expect "91e447d4624e25b3"

    # ── Long-form groupby ─────────────────────────────────────────────────────
    # Concatenate every `rows[]` element across the five files; bucket by the
    # aggregation key. augmentation.json adds `injection_ratio`; for the other
    # four files `r.get("injection_ratio")` -> None so the key is well-defined.
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for f, d in docs.items():
        for r in d["rows"]:
            # WR-02: the 6-key groupby intentionally omits the SENS dimensions
            # (condition/shots/noise_model/noise_level). That is correct ONLY
            # while HEADLINE holds the five frozen Phase 10/11 files (all rows
            # have these unset). If a SENS file is ever folded into HEADLINE,
            # rows differing only by condition/shots/noise_* would be silently
            # averaged into one cell — a faithfulness-destroying collapse.
            # Fail loud instead of silently mis-aggregating.
            if any(
                r.get(k) is not None
                for k in ("condition", "shots", "noise_model", "noise_level")
            ):
                raise AssertionError(
                    f"{f} carries a SENS dimension "
                    f"(condition/shots/noise_*) but the roll-up groupby key "
                    f"omits it — refusing to silently average distinct "
                    f"sensitivity cells. Extend the groupby key before adding "
                    f"SENS files to HEADLINE."
                )
            key = (
                f,
                r["model_kind"],
                r["pipeline"],
                r["metric_name"],
                r["scale"],
                r.get("injection_ratio"),
            )
            buckets[key].append((r["seed"], r["value"]))

    # ── Per-cell mean ± std (n) ───────────────────────────────────────────────
    # Some frozen headline cells carry `value: null` for intentionally-undefined
    # metrics (e.g. fidelity_dualscale.json quantum/log_return cells have no
    # defined fidelity metric — 168 fully-null buckets, no mixed buckets). Null
    # values must be excluded from the mean/std math (statistics.fmean raises on
    # NoneType) while the cell is still emitted so no headline cell is silently
    # dropped (Success Criterion: roll-up covers every source combination). `n`
    # counts only the numeric values; `n_null` records the excluded count for
    # provenance. A fully-null cell -> mean/std null, n 0.
    def _is_num(x: object) -> bool:
        # IN-03: reject non-finite — a stray nan/inf would otherwise pass and
        # poison statistics.fmean (nan propagates through the whole cell).
        return (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(x)
        )

    rollup = []
    for (src, mk, pl, metric, scale, ratio), pairs in buckets.items():
        seeds = sorted({s for s, _ in pairs})
        num_vals = [v for _, v in pairs if _is_num(v)]
        n_null = len(pairs) - len(num_vals)
        if num_vals:
            cell_mean: float | None = statistics.fmean(num_vals)
            cell_std: float | None = (
                statistics.stdev(num_vals) if len(num_vals) > 1 else 0.0
            )
        else:
            cell_mean = None
            cell_std = None
        rollup.append(
            {
                "source": src,
                "model_kind": mk,
                "pipeline": pl,
                "metric_name": metric,
                "scale": scale,
                "injection_ratio": ratio,
                "mean": cell_mean,
                "std": cell_std,
                "n": len(num_vals),
                "n_null": n_null,
                "seeds": seeds,
            }
        )

    out = {
        "schema": "SENS-03 multi-seed roll-up: mean ± std per headline cell",
        "data_hash": canonical_hash,
        "consumed_artifacts": {f: docs[f]["data_hash"] for f in HEADLINE},
        "seed_set": SEED_SET,
        "rollup": rollup,
    }
    (RESULTS / "multiseed_summary.json").write_text(json.dumps(out, indent=2))
    print(
        f"multiseed_summary.json written: {len(rollup)} cells, "
        f"data_hash={canonical_hash}"
    )


if __name__ == "__main__":
    main()
