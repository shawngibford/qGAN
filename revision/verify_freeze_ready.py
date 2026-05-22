"""Phase 14 D-14-22 step-7 — the pre-tag freeze-ready hard block.

This is the LAST gate before the repository is frozen at tag
``v2.0-revision`` and a Zenodo DOI is minted (RESEARCH Pattern 2,
reserve-DOI-first). It refuses to let the freeze proceed unless FIVE
invariants hold, each enforced with the explicit-``raise AssertionError``
idiom (run_multiseed_rollup.py:86-92) so ``python -O`` cannot disable it.

Plan 14-19 (Agent 6 HIGH-3) hardened this gate: it now certifies the
COMMITTED tag candidate, not the live working tree. Gate 0 asserts
``git status --porcelain`` is empty; gate (a) reads ``HEAD:.gitignore`` and
``git ls-files`` rather than the working file and NO LONGER self-heals
(self-healing would dirty the tree and contradict gate 0); gate (d)
asserts ``revision/docs/release.md`` (a 14-07 deliverable) exists.

  (0) clean-tree invariant (Agent 6 HIGH-3, Plan 14-19)
      ``git status --porcelain`` MUST be empty — a green freeze run over a
      dirty working tree is a false-green certification (T-14-30).

  (a) gitignore / archive-content invariant (RESEARCH Pitfall 4)
      Every committed ``revision/results/*.json`` provenance artifact MUST
      ship inside the frozen tag archive. ``.gitignore`` carries a broad
      ``results/`` pattern that could exclude the nested provenance JSON,
      which would publish a DOI'd archive WITHOUT its number backbone
      (success-criterion 5 broken, T-14-18). The committed ``.gitignore``
      MUST carry the ``!revision/results/`` negation and
      ``git check-ignore`` over the committed provenance JSON MUST be
      empty; if not, the gate FAILS hard (no self-heal).

  (b) number-provenance invariant (D-14-22, T-14-20)
      Every paper-blocks file
      (``paper_blocks_framing.md``, ``paper_blocks_refs_methods.md``,
      ``reviewer_response.md``) MUST pass
      ``revision/verify_number_provenance.py --target <file>``. The DOI
      may only mint over FINAL numbers — release is gated LAST behind
      every cited number passing the provenance gate.

  (c) tag-scope invariant (D-14-21, T-14-21)
      ``qgan_env/`` MUST NOT be tracked, ``data.csv`` MUST be tracked,
      and no LARGE training checkpoint (a ``*.pt`` / ``*.pth`` whose
      on-disk size exceeds ``LARGE_CKPT_BYTES``) may be tracked. D-14-21
      excludes only *large* checkpoints "referenced by hash, NOT
      committed"; small pre-existing ablation smoke artifacts that were
      committed by earlier waves are out of this plan's scope to delete
      (destructive-git prohibition) and are explicitly tolerated below
      the threshold.

  (d) release-note invariant (Agent 6 HIGH-3, Plan 14-19)
      ``revision/docs/release.md`` MUST exist. It is authored by plan
      14-07, so this assertion correctly fails until 14-07 runs — the
      intended ordering guard.

No torch / no pennylane / no shared-model-package import — pure
git+filesystem+subprocess consumer (same posture as
verify_number_provenance.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repo-root resolver (run_multiseed_rollup.py idiom): this file lives at
# <repo>/revision/verify_freeze_ready.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "revision" / "results"
GITIGNORE = REPO_ROOT / ".gitignore"

# A genuine training checkpoint for this project is hundreds of MB; the
# pre-existing committed ablation smoke checkpoints are ~1.9 MB each.
# 25 MB sits an order of magnitude above the smoke artifacts and far
# below any real training checkpoint, so it cleanly separates "large
# checkpoint that D-14-21 excludes" from "small artifact a prior wave
# already committed".
LARGE_CKPT_BYTES = 25 * 1024 * 1024

PAPER_BLOCKS = [
    REPO_ROOT / "revision" / "docs" / "paper_blocks_framing.md",
    REPO_ROOT / "revision" / "docs" / "paper_blocks_refs_methods.md",
    REPO_ROOT / "revision" / "docs" / "reviewer_response.md",
]


def _git(*args: str) -> str:
    """Run a git command at the repo root, return stripped stdout."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _check_ignored_json() -> list[str]:
    """Return the list of revision/results/*.json paths git would ignore.

    Plan 14-13 Task 5 (HI-6): rglob recursive walk — matches the verifier's
    own .rglob behaviour in revision/verify_number_provenance.py. The
    previous .glob walked only the top-level revision/results/*.json files
    and silently skipped nested subdir JSONs (e.g. sensitivity/*, matched2000/*),
    so freeze-ready could pass even when nested provenance was gitignored.

    Plan 14-19 (Agent 6 HIGH-3): the provenance set is the COMMITTED tree's
    tracked JSON (``git ls-files``), not the live working tree's rglob — the
    gate certifies the tag candidate, not whatever happens to be on disk.
    ``git check-ignore`` is still the ignore oracle (it consults the
    committed ``.gitignore`` once that file is itself committed).
    """
    tracked_json = [
        p
        for p in _git("ls-files", "revision/results").splitlines()
        if p.endswith(".json")
    ]
    if not tracked_json:
        raise AssertionError(
            "No revision/results/*.json tracked in the committed tree — the "
            "provenance backbone is missing; refusing to freeze (T-14-18)."
        )
    # git check-ignore exits 1 when NOTHING is ignored (that is the safe
    # case); the ignored paths, if any, are printed one per line.
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", *tracked_json],
        capture_output=True,
        text=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def gate_zero_clean_tree() -> None:
    """(0) The working tree MUST be clean — the gate certifies the COMMITTED
    tag candidate, not a dirty working tree (Agent 6 HIGH-3, Plan 14-19).

    A non-empty ``git status --porcelain`` means uncommitted edits or
    untracked files exist; a green freeze run over a dirty tree is a
    false-green certification (T-14-30). Enforced with the explicit
    ``raise AssertionError`` idiom so ``python -O`` cannot disable it.
    """
    porcelain = _git("status", "--porcelain")
    if porcelain:
        raise AssertionError(
            "Working tree is NOT clean — `git status --porcelain` is "
            "non-empty; the freeze gate certifies the COMMITTED tag "
            "candidate, not a dirty working tree (Agent 6 HIGH-3, "
            f"T-14-30):\n{porcelain}"
        )
    print("(0) clean-tree OK — git status --porcelain is empty.")


def gate_d_release_md() -> None:
    """(d) revision/docs/release.md MUST exist — it is a 14-07 must_have
    the prior gate did not check (Agent 6 HIGH-3, Plan 14-19).

    release.md is authored by plan 14-07, so this assertion correctly fails
    until 14-07 runs — that is the intended ordering guard: the freeze
    sequence cannot be certified complete before the release note exists.
    """
    release_md = REPO_ROOT / "revision" / "docs" / "release.md"
    if not release_md.is_file():
        raise AssertionError(
            f"release note absent: {release_md} — revision/docs/release.md "
            "is a 14-07 deliverable; the freeze cannot be certified until "
            "14-07 authors it (Plan 14-19 ordering guard)."
        )
    print(f"(d) release.md OK — {release_md.relative_to(REPO_ROOT)} exists.")


def gate_a_gitignore_archive() -> None:
    """(a) No provenance JSON may be gitignored out of the frozen tag.

    Plan 14-19 (Agent 6 HIGH-3): the self-heal block was REMOVED — this gate
    now validates the COMMITTED tree and MUST NOT mutate ``.gitignore`` or
    stage files (that would dirty the tree and contradict gate 0). The
    ``results/`` exclusion + both ``!revision/results/`` negations are
    committed atomically by plan 14-19 Task 2; if a provenance JSON is still
    ignored here, the gate FAILS hard rather than self-healing.
    """
    # The committed .gitignore is the ignore oracle — verify the negations
    # are in the committed tree, not just the working file.
    committed_gitignore = _git("show", "HEAD:.gitignore")
    if "!revision/results/" not in committed_gitignore:
        raise AssertionError(
            "The committed .gitignore (HEAD:.gitignore) is missing the "
            "'!revision/results/' negation — the provenance backbone would "
            "be ignored out of the frozen tag (RESEARCH Pitfall 4, T-14-18)."
        )

    ignored = _check_ignored_json()
    tracked = _git("ls-files", "revision/results").splitlines()
    if ignored:
        raise AssertionError(
            f"revision/results/*.json gitignored in the committed tree: "
            f"{ignored} — the DOI'd archive would ship without its number "
            f"backbone (RESEARCH Pitfall 4, T-14-18)."
        )
    if not tracked:
        raise AssertionError(
            "git ls-files revision/results is empty — the provenance "
            "backbone is not tracked; refusing to freeze (T-14-18)."
        )
    print(
        f"(a) gitignore/archive OK — git check-ignore empty; "
        f"{len(tracked)} tracked paths under revision/results."
    )


def gate_b_number_provenance() -> None:
    """(b) Every paper-blocks file must pass the number-provenance gate."""
    gate = REPO_ROOT / "revision" / "verify_number_provenance.py"
    if not gate.exists():
        raise AssertionError(
            f"number-provenance gate missing: {gate} — cannot certify the "
            f"cited numbers are final (D-14-22)."
        )
    failures: list[str] = []
    for blk in PAPER_BLOCKS:
        if not blk.exists():
            raise AssertionError(
                f"paper-blocks file absent: {blk} — release is hard-blocked "
                f"until every paper-blocks file passes the provenance gate "
                f"(D-14-22)."
            )
        proc = subprocess.run(
            [sys.executable, str(gate), "--target", str(blk)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failures.append(
                f"{blk.name}: exit {proc.returncode}\n"
                f"{proc.stdout}{proc.stderr}"
            )
        else:
            print(f"(b) provenance OK — {blk.name}: {proc.stdout.strip()}")
    if failures:
        raise AssertionError(
            "Number-provenance gate FAILED for "
            f"{len(failures)} paper-blocks file(s); the DOI may only mint "
            "over FINAL numbers (D-14-22, T-14-20):\n"
            + "\n".join(failures)
        )


def gate_c_tag_scope() -> None:
    """(c) qgan_env/ out, data.csv in, no LARGE checkpoint tracked."""
    ls_files = _git("ls-files").splitlines()

    qgan_env_tracked = [p for p in ls_files if p.startswith("qgan_env/")]
    if qgan_env_tracked:
        raise AssertionError(
            f"qgan_env/ is tracked ({len(qgan_env_tracked)} paths) — the "
            f"virtualenv must NOT ship in the frozen tag (D-14-21, T-14-21)."
        )

    if "data.csv" not in ls_files:
        raise AssertionError(
            "data.csv is NOT tracked — the frozen tag MUST carry the "
            "dataset so the DOI'd archive is self-contained (D-14-21)."
        )

    large: list[str] = []
    for p in ls_files:
        if p.endswith((".pt", ".pth")):
            fp = REPO_ROOT / p
            try:
                if fp.is_file() and fp.stat().st_size > LARGE_CKPT_BYTES:
                    large.append(f"{p} ({fp.stat().st_size} bytes)")
            except OSError:
                continue
    if large:
        raise AssertionError(
            "LARGE training checkpoint(s) tracked — D-14-21 requires large "
            f"checkpoints to be referenced by hash, NOT committed: {large}"
        )
    print(
        "(c) tag-scope OK — qgan_env/ not tracked, data.csv tracked, "
        f"no checkpoint exceeds {LARGE_CKPT_BYTES} bytes."
    )


def main() -> int:
    print("=== verify_freeze_ready.py — D-14-22 step-7 pre-tag hard block ===")
    gate_zero_clean_tree()
    gate_a_gitignore_archive()
    gate_b_number_provenance()
    gate_c_tag_scope()
    gate_d_release_md()
    print(
        "FREEZE-READY: all invariants hold — clean tree, gitignore/archive, "
        "number-provenance, tag-scope, release note. The tag may be cut."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
