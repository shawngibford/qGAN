"""Phase 14 RESEARCH Pattern 3 — the reusable number-provenance gate (v2).

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

v2 contract (Phase 14 plan 14-13, D-14-16 LIFTED for that plan only)
-------------------------------------------------------------------
v1 accepted substring matches (`if token in blob`) and float-precision
equality via formatting-string coincidence (`f"{cval:.{prec}f}" ==
f"{val:.{prec}f}"`). Both produced false-positive PASS verdicts (CR-5,
PROV-HIGH-1, PROV-CRIT-2 root cause). v2 fixes:

  (a) Exact-token boundary match:
        re.search(rf"(?<![\\d.]){re.escape(token)}(?![\\d])", blob)
      The lookbehind eliminates digit-after-decimal substring matches
      (e.g. token `0.6843` no longer matches inside `0.03006843578`);
      the lookahead eliminates prefix-substring matches.

  (b) Tighter float resolution as a real ε-neighborhood:
        abs(cval - val) <= 10**(-prec) / 2
      The true mathematical "matches to `prec` decimal places" rather than
      the formatting-string coincidence that can match e.g. `0.121` against
      `0.124` if both round to `0.12`.

  (c) Exclude render-only JSONs from the resolution corpus. Render-only
      companion JSONs (`{"render_only": true, ...}`) can still be GATED as
      documents (i.e. passed to `--target`) but are not used as a resolution
      SOURCE. This eliminates the failure mode where a render-only artifact
      becomes a tautological resolution path for its own literals.

  (d) Narrower `_ID_PATTERNS`. The year-strip pattern requires a closing
      paren lookahead and the 1900-2099 range (so it does not strip arbitrary
      4-digit data values); the file:line strip requires an explicit file
      extension prefix (so it does not strip stray `:<digits>` in prose).

  (e) New `--manifest` flag. When set, prints a resolution trace per PASSing
      literal: `<doc>:<literal> -> <json_path>#<key_path>`. The trace lets
      reviewers spot semantic mismatches (a literal that resolves to the
      WRONG JSON key by coincidence).

Schema string is bumped to identify the v2 contract. Gate is back in
byte-freeze under the v2 schema after Plan 14-13 Task 2 closes.

No torch / no pennylane / no shared-model-package import — pure text+JSON
consumer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_SCHEMA = "v2 (Phase 14 plan 14-13 — boundary-strict resolution + render-only exclusion)"


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
#
# v2 narrowing (Plan 14-13, PROV-MED-1 / PROV-MED-2):
#   - year strip requires 1900-2099 range AND a closing-paren lookahead, so
#     arbitrary 4-digit data values are not silently dropped (e.g. epoch
#     `1969` is preserved as data; `(Gulrajani 2017)` is still stripped)
#   - file:line strip requires an explicit file-extension prefix
#     `.py|.md|.json|.tex` so stray `:<digits>` in prose is preserved
_ID_PATTERNS = [
    r"D-\d+-\d+",                              # decision ids
    r"R\d+-M\d+",                              # reviewer-memo ids
    r"R\d+-m\d+",                              # lowercase reviewer-memo ids
    r"Phase\s*\d+(?:\.\d+)?",                  # phase numbers
    r"\b\d+-\d+\b",                            # plan ids like 14-05
    r"(?:\.py|\.md|\.json|\.tex):\d+(?:-\d+)?\b",  # file:line w/ extension (v2 narrower)
    r"\bmain:\d+(?:-\d+)?\b",                  # LaTeX manuscript line citations
    r"\bsupp:\d+(?:-\d+)?\b",                  # LaTeX supplementary-material line citations
    r"\bmain \(\d+\) copy\.tex:\d+(?:-\d+)?\b",  # quoted LaTeX manuscript line citations
    r"\bsupp_material\.tex:\d+(?:-\d+)?\b",    # LaTeX supplementary file:line citations
    r"\bline\s*~?\d+(?:-\d+)?\b",              # prose line citations 'line ~148', 'line 138'
    r"\bvolume\s*=\s*\{\d+\}",                 # BibTeX volume = {567}
    r"\[\d+\](?:-\[\d+\])?",                   # bracketed bibliography refs [21], [61], [21]-[23]
    r"\b(?:19|20)\d{2}\b(?=\s*\))",            # citation years e.g. (Gulrajani 2017) (v2 narrower)
    r"year\s*=\s*\{(?:19|20)\d{2}\}",          # BibTeX year fields year = {2017}
    r"arXiv:\d{4}\.\d{4,5}",                   # arXiv IDs (e.g. arXiv:1706.02633)
    r"arXiv preprint arXiv:\d{4}\.\d{4,5}",    # full prose form
    r"aic-\d{7}",                              # AIC manuscript IDs (e.g. aic-4719598)
    r"\bv\d+(?:\.\d+)?\b",                     # version tags v2.0
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


def _walk_json(obj, path=""):
    """Yield (key_path, value) for every leaf in a JSON-decoded object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}.{k}" if path else k
            yield from _walk_json(v, sub)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sub = f"{path}[{i}]"
            yield from _walk_json(v, sub)
    else:
        yield (path, obj)


def _json_corpus() -> dict[str, dict]:
    """Concatenated text + parsed object of every revision/results/*.json
    (recursively), EXCLUDING JSONs whose top-level dict carries
    `"render_only": true` (PROV-MED-3).

    Returns a mapping path -> {"raw": str, "norm": str, "obj": object}.
    Both the raw file text AND a re-serialized form are concatenated so a
    literal resolves whether the doc renders it as repr (1.8046e-05) or the
    JSON happens to store it differently (1.8046e-5 / 0.000018046)."""
    corpus: dict[str, dict] = {}
    for jp in sorted(RESULTS.rglob("*.json")):
        try:
            raw = jp.read_text()
        except OSError:
            continue
        try:
            obj = json.loads(raw)
            norm = json.dumps(obj)
        except (ValueError, TypeError):
            obj = None
            norm = ""
        if isinstance(obj, dict) and obj.get("render_only") is True:
            # render-only companions are not resolution sources (PROV-MED-3)
            continue
        corpus[str(jp.relative_to(REPO))] = {
            "raw": raw,
            "norm": norm,
            "blob": raw + "\n" + norm,
            "obj": obj,
        }
    return corpus


def _resolves(token: str, corpus: dict[str, dict]) -> tuple[str, str] | None:
    """Return (artifact_path, key_path) the token resolves to, or None.

    Resolution attempts, in order:
      1. Exact-token boundary match in concatenated text (v2 boundary-strict)
      2. Float ε-neighborhood match against any numeric leaf in any JSON
    """
    boundary_re = re.compile(rf"(?<![\d.]){re.escape(token)}(?![\d])")

    # Pass 1: boundary-strict text match.
    for path, entry in corpus.items():
        if boundary_re.search(entry["blob"]):
            # Try to find a specific key-path inside the parsed object.
            obj = entry.get("obj")
            if obj is not None:
                for kpath, leaf in _walk_json(obj):
                    s = str(leaf)
                    if boundary_re.search(s):
                        return (path, kpath or "<root>")
                    # Numeric leaves may serialize differently
                    try:
                        if isinstance(leaf, (int, float)):
                            if boundary_re.search(repr(leaf)) or boundary_re.search(json.dumps(leaf)):
                                return (path, kpath or "<root>")
                    except Exception:
                        pass
            return (path, "<text-match>")

    # Pass 2: float ε-neighborhood resolution.
    try:
        val = float(token)
    except ValueError:
        return None
    if "." in token and "e" not in token.lower():
        prec = len(token.split(".")[1])
        tol = 10 ** (-prec) / 2.0
    elif "e" in token.lower():
        # Exponent form: use mantissa precision for tolerance
        mantissa = token.lower().split("e")[0]
        if "." in mantissa:
            prec = len(mantissa.split(".")[1])
        else:
            prec = 0
        # Tolerance relative to the exponent magnitude
        try:
            exp = int(token.lower().split("e")[1])
            tol = 10 ** (exp - prec) / 2.0
        except (ValueError, IndexError):
            tol = abs(val) * 1e-9 if val != 0 else 1e-12
    else:
        prec = 0
        tol = 0.5  # integer ε-neighborhood

    for path, entry in corpus.items():
        obj = entry.get("obj")
        if obj is None:
            continue
        for kpath, leaf in _walk_json(obj):
            if not isinstance(leaf, (int, float)):
                continue
            try:
                cval = float(leaf)
            except (ValueError, TypeError):
                continue
            if abs(cval - val) <= tol:
                return (path, kpath or "<root>")
    return None


def verify(target: Path, manifest: bool = False) -> int:
    text = target.read_text()
    scrubbed = _strip_identifiers(text)
    corpus = _json_corpus()

    tokens = [t for t in _NUM.findall(scrubbed) if t not in _ALLOW]
    unresolved: list[str] = []
    resolved: dict[str, tuple[str, str]] = {}
    for tok in tokens:
        if tok in resolved:
            continue
        src = _resolves(tok, corpus)
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
            f"artifact (D-14-16 LIFTED → v2 schema under Plan 14-13)."
        )

    print(
        f"{target}: PASS — {len(resolved)} distinct numeric literal(s) all "
        f"resolve to revision/results/*.json (number-provenance gate, "
        f"schema={_SCHEMA!r})."
    )
    if manifest:
        print(f"--- resolution manifest ({len(resolved)} entries) ---")
        for tok in sorted(resolved):
            path, kpath = resolved[tok]
            print(f"{target}:{tok} -> {path}#{kpath}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Number-provenance gate (RESEARCH Pattern 3, D-14-16 LIFTED → v2 "
            "under Phase 14 plan 14-13)."
        )
    )
    ap.add_argument(
        "--target",
        required=True,
        help="Path to the text file (regenerated doc or paper LaTeX-blocks "
        "file) whose numeric literals must all resolve to a "
        "revision/results/*.json value.",
    )
    ap.add_argument(
        "--manifest",
        action="store_true",
        help="Print a resolution trace per PASSing literal: "
        "`<doc>:<literal> -> <json_path>#<key_path>`. New under v2 "
        "(Phase 14 plan 14-13).",
    )
    args = ap.parse_args()
    target = Path(args.target)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.exists():
        raise AssertionError(f"--target does not exist: {target}")
    sys.exit(verify(target, manifest=args.manifest))


if __name__ == "__main__":
    main()
