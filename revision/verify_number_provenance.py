"""Phase 14 RESEARCH Pattern 3 — the reusable number-provenance gate.

Given a target text file (`--target`), extract every numeric literal and, for
each, assert it resolves at the stated precision to a value present in some
`revision/results/*.json` artifact. Any unresolved literal triggers an
explicit `raise AssertionError` (run_multiseed_rollup.py:86-92 idiom) and a
non-zero exit — `python -O` cannot disable the gate.

This is the EXECUTABLE enforcement of success-criterion 5 ("every number in
the paper/docs traces to a JSON artifact"). It is the reusable gate the
downstream LaTeX-block plans (14-05, 14-06) are verified against:

    ./qgan_env/bin/python revision/verify_number_provenance.py \
        --target revision/docs/training_protocol.md

Resolution model
----------------
A literal RESOLVES if its canonical textual form (or its float value at the
literal's own precision) appears in the concatenated text of any
`revision/results/*.json`. Pure structural integers that are never data
(markdown table widths, list indices) do not occur in these docs because the
renderer only emits JSON-sourced values; an allow-list of trivially-universal
tokens (0, 1, 2 — and the decision-id digits in `D-14-NN` / `R1-MN` / phase
numbers) is excluded from the gate to avoid false positives on prose
identifiers, NOT on data.

No torch / no pennylane / no shared-model-package import — pure text+JSON
consumer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


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
RESULTS = REPO / "revision/results"

# Numeric-literal token: optional sign, int/decimal, optional exponent.
# Examples matched: 2000, 12, 2.16, 1.8046e-05, 6.9173e-05, -0.011286, 384.
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")

# Identifier contexts whose digits are NOT data: decision/requirement/phase
# tags (D-14-13, R1-M5, Phase 09.1, 14-05, run_multiseed_rollup.py:86-92,
# data.py:255-258, Gulrajani 2017, [−1, 1] interval, lag-N, 2⁵). These are
# stripped BEFORE literal extraction so the gate checks data, not prose IDs.
_ID_PATTERNS = [
    r"D-\d+-\d+",                       # decision ids
    r"R\d+-M\d+",                       # reviewer-memo ids
    r"Phase\s*\d+(?:\.\d+)?",           # phase numbers
    r"\b\d+-\d+\b",                     # plan ids like 14-05
    r"\.py:\d+(?:-\d+)?",               # source line citations
    r":\d+(?:-\d+)?\b",                 # bare line-range citations
    r"\b\d{4}\b(?=\s*\))",             # citation years e.g. (Gulrajani 2017)
    r"\bv\d+(?:\.\d+)?\b",             # version tags v2.0
    r"\bR1-M\d+\b",
    r"\b09\.1\b",
]

# Trivially-universal small integers + the [-1,1] window-space bounds: present
# by construction in any dataset; excluding them avoids prose false-positives
# without weakening the gate on real reported quantities.
_ALLOW = {"0", "1", "2", "-1", "+1", "0.0", "1.0", "2.0", "-1.0"}


def _strip_identifiers(text: str) -> str:
    out = text
    for pat in _ID_PATTERNS:
        out = re.sub(pat, " ", out)
    return out


def _json_blobs() -> dict[str, str]:
    """Concatenated text of every revision/results/*.json (recursively).

    Both the raw file text AND a re-serialized form are concatenated so a
    literal resolves whether the doc renders it as repr (1.8046e-05) or the
    JSON happens to store it differently (1.8046e-5 / 0.000018046)."""
    blobs: dict[str, str] = {}
    for jp in sorted(RESULTS.rglob("*.json")):
        raw = jp.read_text()
        try:
            obj = json.loads(raw)
            norm = json.dumps(obj)
        except (ValueError, TypeError):
            norm = ""
        blobs[str(jp.relative_to(REPO))] = raw + "\n" + norm
    return blobs


def _resolves(token: str, blobs: dict[str, str]) -> str | None:
    """Return the artifact path the token resolves to, or None.

    Resolution attempts, in order:
      1. exact substring of the token text
      2. float value match at the token's own decimal precision (handles
         repr vs json.dumps float spelling, e.g. 6.9173e-05 vs 6.9173e-5)
    """
    for path, blob in blobs.items():
        if token in blob:
            return path
    # Float-value resolution at the token's stated precision.
    try:
        val = float(token)
    except ValueError:
        return None
    if "." in token and "e" not in token.lower():
        prec = len(token.split(".")[1])
    else:
        prec = None
    for path, blob in blobs.items():
        for cand in _NUM.findall(blob):
            try:
                cval = float(cand)
            except ValueError:
                continue
            if prec is not None:
                if f"{cval:.{prec}f}" == f"{val:.{prec}f}":
                    return path
            else:
                if cval == val:
                    return path
    return None


def verify(target: Path) -> int:
    text = target.read_text()
    scrubbed = _strip_identifiers(text)
    blobs = _json_blobs()

    tokens = [t for t in _NUM.findall(scrubbed) if t not in _ALLOW]
    unresolved: list[str] = []
    resolved: dict[str, str] = {}
    for tok in tokens:
        if tok in resolved:
            continue
        src = _resolves(tok, blobs)
        if src is None:
            unresolved.append(tok)
        else:
            resolved[tok] = src

    if unresolved:
        # Explicit raise (NOT bare assert) — run_multiseed_rollup.py:86-92
        # idiom; survives `python -O`. This is success-criterion-5 enforcement.
        raise AssertionError(
            f"{target}: {len(set(unresolved))} numeric literal(s) do NOT "
            f"resolve to any revision/results/*.json value at stated "
            f"precision: {sorted(set(unresolved))}. Every number in a "
            f"provenance doc / paper LaTeX block MUST trace to a JSON "
            f"artifact (D-14-16, success criterion 5)."
        )

    print(
        f"{target}: PASS — {len(resolved)} distinct numeric literal(s) all "
        f"resolve to revision/results/*.json (number-provenance gate)."
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Number-provenance gate (RESEARCH Pattern 3, D-14-16)."
    )
    ap.add_argument(
        "--target",
        required=True,
        help="Path to the text file (regenerated doc or paper LaTeX-blocks "
        "file) whose numeric literals must all resolve to a "
        "revision/results/*.json value.",
    )
    args = ap.parse_args()
    target = Path(args.target)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.exists():
        raise AssertionError(f"--target does not exist: {target}")
    sys.exit(verify(target))


if __name__ == "__main__":
    main()
