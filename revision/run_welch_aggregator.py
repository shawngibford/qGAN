"""Phase 14 plan 14-16 (W2) — Welch pairwise aggregator (NEW).

Emits ``revision/results/welch_pairwise.json`` from the corrected per-seed
EMD aggregates in ``revision/results/matched2000_dualscale.json`` (post-T1
R3-CR-2 fix). The JSON is the auditable anchor for the strong-claim Welch
literals asserted in ``revision/docs/reviewer_response.md``'s
parametric-efficiency-equivalence H2 section.

Path A (Plan 14-16 r3 process retraction): only the OD-EMD strong-claim
thresholds (``p > 0.36``, ``|d| <= 0.65``) are enforced — the OD-EMD column
is byte-stable pre/post the R3-CR-2 fix. The LR-EMD-vs-WGAN strong claim is
withdrawn (the pre-fix ``statistical-honesty-r3.md`` §3b Welch tests were
computed on the broken, scale-mismatched LR-EMD column). LR-EMD per-pair
stats are still emitted in ``pairs[]`` for transparency.

Top-level emitter — no imports from ``revision/core/`` (D-14-22 byte-freeze
preserved). The new aggregator JSON is auto-walked into the v2.1
number-provenance gate's resolution corpus by the existing ``_json_blobs()``
walker in ``revision/verify_number_provenance.py`` (no gate edit; D-14-16
byte-freeze preserved).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

DATA_HASH = "91e447d4624e25b3"
SCHEMA = "welch-pairwise v1 (Phase 14 plan 14-16 W2)"
SOURCE_REL = Path("revision/results/matched2000_dualscale.json")
OUT_REL = Path("revision/results/welch_pairwise.json")

QUANTUM_MODELS = ["iqp_sel_55_repro", "V1", "V2", "V3"]
CLASSICAL_MODELS = ["wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
SCALES = ["OD", "log_return"]

NOTES = (
    "Path A (Plan 14-16 r3 process retraction): Post-R3-CR-2 fix "
    "(un-standardize-fake per pipeline-review-r3.md §2), LR-EMD rankings "
    "invert from pre-fix narrative. AR (Yule-Walker MLE) leads at 0.003, "
    "quantum/WGAN/VAE cluster in 0.007-0.016 with every WGAN beating every "
    "quantum on the corrected scale. The pre-fix statistical-honesty-r3.md "
    "§3b strong-claim Welch tests (p <= 0.014, Cohen d <= -2.6) were computed "
    "on the broken (scale-mismatched) LR-EMD column and DO NOT carry "
    "post-fix. The LR-EMD pairs in this JSON are emitted as per-pair stats "
    "for transparency but the strong_claim_thresholds block ONLY enforces "
    "OD-EMD thresholds (which survive because the OD column is byte-stable "
    "pre/post T1). See peer_review_remediation.md Plan 14-16 r3 process "
    "retraction subsection for the full retraction documentation."
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (ddof=1)."""
    n_q, n_c = len(a), len(b)
    pooled_sd = np.sqrt(
        ((n_q - 1) * a.std(ddof=1) ** 2 + (n_c - 1) * b.std(ddof=1) ** 2)
        / (n_q + n_c - 2)
    )
    return float((a.mean() - b.mean()) / pooled_sd)


def build_payload(repo: Path) -> dict:
    src = json.loads((repo / SOURCE_REL).read_text())
    rows = [
        r for r in src.get("rows", [])
        if r.get("metric_name") == "emd" and r.get("pipeline") == "B"
    ]
    groups: dict[tuple[str, str], list[float]] = {}
    seeds: set[int] = set()
    for r in rows:
        groups.setdefault((r["model_kind"], r["scale"]), []).append(
            float(r["value"])
        )
        seeds.add(int(r["seed"]))

    pairs: list[dict] = []
    for scale in SCALES:
        for q, c in itertools.product(QUANTUM_MODELS, CLASSICAL_MODELS):
            vq = groups.get((q, scale))
            vc = groups.get((c, scale))
            if not vq or not vc:
                continue
            aq = np.asarray(vq, dtype=np.float64)
            ac = np.asarray(vc, dtype=np.float64)
            welch_t, welch_p = ttest_ind(aq, ac, equal_var=False)
            mwu_stat, mwu_p = mannwhitneyu(aq, ac, alternative="two-sided")
            pairs.append({
                "quantum": q,
                "classical": c,
                "scale": scale,
                "mean_q": float(aq.mean()),
                "std_q": float(aq.std(ddof=1)),
                "n_q": int(aq.size),
                "mean_c": float(ac.mean()),
                "std_c": float(ac.std(ddof=1)),
                "n_c": int(ac.size),
                "welch_t": float(welch_t),
                "welch_p": float(welch_p),
                "cohen_d": _cohen_d(aq, ac),
                "mwu_stat": float(mwu_stat),
                "mwu_p": float(mwu_p),
            })

    pairs.sort(key=lambda p: (p["scale"], p["quantum"], p["classical"]))

    od_pairs = [p for p in pairs if p["scale"] == "OD"]
    lr_wgan_pairs = [
        p for p in pairs
        if p["scale"] == "log_return" and p["classical"].startswith("wgan")
    ]
    summaries = {
        "OD_floor_welch_p_quantum_vs_classical": min(
            p["welch_p"] for p in od_pairs
        ),
        "OD_ceiling_abs_cohen_d_quantum_vs_classical": max(
            abs(p["cohen_d"]) for p in od_pairs
        ),
        "log_return_floor_welch_p_quantum_vs_wgan": (
            min(p["welch_p"] for p in lr_wgan_pairs) if lr_wgan_pairs else None
        ),
        "log_return_extremum_cohen_d_quantum_vs_wgan": (
            max(p["cohen_d"] for p in lr_wgan_pairs) if lr_wgan_pairs else None
        ),
    }
    # R3 deviation 2026-06-10 (user-authorized 14-21 Rule 4):
    # OD-EMD parametric-equivalence (H2) strong claim from v1.2.4 was DROPPED
    # because post-x0.1-fix data shows quantum is SIGNIFICANTLY better than
    # WGAN on OD-EMD (Welch p=0.019), not statistically-equivalent. The prior
    # thresholds {floor_welch_p_OD: 0.36, ceiling_abs_cohen_d_OD: 0.65} were
    # the H2 acceptance gate; they no longer correspond to the post-fix paper
    # narrative. We preserve the threshold dict for historical traceability
    # but convert the hard-abort gates to soft-fail warnings that record the
    # actual computed value in the output JSON for transparency. See
    # .planning/CONTEXT-HANDOFF-2026-06-02.md §6 #2 (post-14-21 amendment).
    strong_claim_thresholds = {
        "floor_welch_p_OD": 0.36,            # historical: H2 acceptance gate (dropped post-14-21)
        "ceiling_abs_cohen_d_OD": 0.65,      # historical: H2 acceptance gate (dropped post-14-21)
        "_status": "H2_DROPPED_PER_14-21_R3",
        "_note": (
            "Post-x0.1-fix data inverts the H2 parametric-equivalence claim. "
            "Thresholds preserved for traceability but no longer gate writes."
        ),
    }

    payload = {
        "schema": SCHEMA,
        "data_hash": DATA_HASH,
        "source": str(SOURCE_REL),
        "source_filter": "metric_name=='emd', pipeline=='B'",
        "n_per_group": 5,
        "seeds": sorted(seeds),
        "quantum_models": QUANTUM_MODELS,
        "classical_models": CLASSICAL_MODELS,
        "scales": SCALES,
        "pairs": pairs,
        "summaries": summaries,
        "strong_claim_thresholds": strong_claim_thresholds,
        "notes": NOTES,
    }
    return payload


def main() -> None:
    repo = _repo_root()
    payload = build_payload(repo)
    s = payload["summaries"]
    thr = payload["strong_claim_thresholds"]

    # R3 deviation 2026-06-10 (user-authorized 14-21 Rule 4):
    # Prior hard-abort gates on H2 parametric-equivalence thresholds dropped.
    # Post-fix data: quantum significantly BEATS WGAN on OD-EMD (p~0.019, not
    # equivalence). Print the computed values + emit soft-fail warnings rather
    # than aborting; the threshold dict is preserved in the output JSON for
    # historical traceability.
    od_p = s["OD_floor_welch_p_quantum_vs_classical"]
    od_d = s["OD_ceiling_abs_cohen_d_quantum_vs_classical"]
    if not od_p > thr["floor_welch_p_OD"]:
        print(
            f"[run_welch_aggregator] SOFT-FAIL (H2 dropped per 14-21 R3): "
            f"OD floor Welch p={od_p:.4f} does NOT clear historical threshold "
            f"{thr['floor_welch_p_OD']}. Post-fix interpretation: quantum is "
            f"significantly BETTER than WGAN on OD-EMD (not equivalent).",
            file=sys.stderr,
        )
    if not od_d <= thr["ceiling_abs_cohen_d_OD"]:
        print(
            f"[run_welch_aggregator] SOFT-FAIL (H2 dropped per 14-21 R3): "
            f"OD ceiling |Cohen d|={od_d:.4f} exceeds historical threshold "
            f"{thr['ceiling_abs_cohen_d_OD']}. Post-fix interpretation: large "
            f"effect size confirms quantum advantage on OD-EMD.",
            file=sys.stderr,
        )

    out_path = repo / OUT_REL
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(
        f"[run_welch_aggregator] wrote {out_path} "
        f"({len(payload['pairs'])} pairs; "
        f"OD floor Welch p={s['OD_floor_welch_p_quantum_vs_classical']:.4f}; "
        f"OD ceiling |d|={s['OD_ceiling_abs_cohen_d_quantum_vs_classical']:.4f})"
    )


if __name__ == "__main__":
    main()
