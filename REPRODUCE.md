# REPRODUCE.md — Manuscript-revision reproducibility entry-point

> One-stop reviewer entry-point for reproducing the figures, numbers, and
> methods doc emitted by Phase 14 (manuscript revision). For the full
> per-script breakdown see `docs/methods_full.md §5.2`; for
> cross-plan navigation see `docs/completeness_sweep_manifest.md`.

This file lives at the repository root so reviewers landing on the
top-level directory listing find an oriented walkthrough rather than
having to spelunk inside `docs/` for the methods doc.

## 1 — Setup

```bash
git clone <repo URL>
cd qGAN
python -m venv qgan_env
source qgan_env/bin/activate
pip install -r requirements-pinned.txt
```

The pinned-requirements file captures the exact package versions used
to emit the manuscript-revision artifacts (matplotlib, numpy, pennylane,
PyYAML, scipy, statsmodels, torch). Re-run on environment change.

## 2 — Checkout the manuscript-revision tag

The repository will be frozen at the tag `v2.0-revision` when Plan 14-07
(the only outstanding Phase 14 plan) lands the Zenodo DOI deposit. Until
then, use the current `main` branch — every commit between the close of
Plan 14-13 and the eventual `v2.0-revision` tag is part of the revision
package.

```bash
git checkout v2.0-revision  # post-14-07; use `main` in the interim
```

## 3 — Regenerate the canonical artifacts (dependency order)

```bash
# Headline metric — single frozen-checkpoint evaluation on the audited
# dataset hash; outputs results/headline_canonical.json.
./qgan_env/bin/python scripts/run_canonical_headline.py

# Matched-budget dualscale sweep — aggregate mean/std across seeds 42-46
# at the matched 2000-epoch budget; outputs
# results/matched2000_dualscale.json#aggregates (the source of
# truth for the EMD OD-scale reconciliation table).
./qgan_env/bin/python scripts/run_matched2000_dualscale.py

# Methods doc + companion JSON — re-emit
# docs/methods_full.md from the live model_info.json,
# classical_architectures.json, circuit_diagrams.json,
# framework_versions.json, and the verbatim run_matched2000.py module
# docstring (lines 1-69).
./qgan_env/bin/python scripts/run_methods_full.py

# Figure suite — regenerate the figure PDFs/PNGs consumed by the
# manuscript.
./qgan_env/bin/python scripts/run_figure_suite.py

# Provenance gate v2.1 — verify every numeric literal in the methods doc
# resolves to a results/*.json artifact at the stated precision.
./qgan_env/bin/python scripts/verify_number_provenance.py \
    --target docs/methods_full.md
```

The gate's v2.1 schema (Phase 14 plan 14-14) adds a negative-sign-aware
lookbehind to the boundary-strict resolution regex, eliminating the
positive→negative sign-flip false positive that the v2 schema admitted.
Run with `--manifest` for a per-literal resolution trace.

## 4 — Further reading

- `docs/methods_full.md §5.2` — full per-script breakdown
  (emitter dependency order, input/output artifact map).
- `docs/completeness_sweep_manifest.md` — cross-plan artifact
  navigation (Plans 14-09 through 14-14).
- `docs/peer_review_remediation.md` — finding-to-commit index
  for the r1 + r2 peer-review passes (every reviewer finding mapped to
  the commit that resolved it).
- `docs/release.md` — frozen-tag commit SHA + Zenodo DOI
  (post-14-07 deposit).
- `docs/reviewer_response.md` — point-by-point response to
  reviewer R1 and R2 findings.
