"""Phase 14 plan 14-11 Task 2 — framework version pinner (introspection-only).

Pure-introspection emitter that records the EXACT installed-package versions
of the running qgan_env interpreter at methods-doc emit time. The repo's
``revision/requirements.txt`` carries the ``>=`` constraint set; this JSON
carries the exact pin for paper reproducibility — re-emit on environment
change.

Zero network access (no installer calls, no out-of-process spawn, no HTTP
client). Pure ``importlib.metadata.version(...)`` introspection of the
running interpreter.

The recorded packages are the Phase-14 runtime dependencies actually used by
the pipeline (pennylane: quantum.py; torch: every model + training; numpy:
data + AR; scipy: revision/core/eval helpers; matplotlib: figure suite;
PyYAML: every ``run_*.py`` that reads ``config.yaml``). A
``PackageNotFoundError`` records ``None`` for that package rather than
crashing the emitter — but every listed package IS a Phase-14 runtime
dependency and IS expected to resolve.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (verbatim run_model_info.py:69-81 idiom).
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
RESULTS = REPO / "revision/results"

# Stable order: the JSON is diffable across emissions. Names are case-sensitive
# per PyPI metadata — "PyYAML" is the case-sensitive distribution name (the
# import is lowercase ``yaml`` but ``importlib.metadata.version`` resolves the
# distribution, not the import).
PACKAGES: tuple[str, ...] = (
    "pennylane",
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "PyYAML",
)


def _resolve_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def main() -> None:
    pkgs = _resolve_versions(PACKAGES)

    out = {
        "schema": "framework-versions v1 (Phase 14 plan 14-11)",
        "captured_at_iso": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "note": (
            "Recorded at methods-doc emit time from the running qgan_env "
            "interpreter; revision/requirements.txt carries the >= "
            "constraint set, this JSON carries the exact installed pin. "
            "Re-emit on environment change."
        ),
    }

    out_path = RESULTS / "framework_versions.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(
        f"framework_versions.json written: {len(pkgs)} packages "
        f"({sum(1 for v in pkgs.values() if v is not None)} resolved)"
    )
    print(f"  output: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
