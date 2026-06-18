# Phase 14: Paper Revision & Release Freeze - Research

**Researched:** 2026-05-19
**Domain:** Scientific manuscript revision (LaTeX, read-only) + reproducible-research release engineering (git tag + Zenodo DOI) + JSON-traceable number/figure provenance + lost-config recovery from a PyTorch checkpoint
**Confidence:** HIGH (repo-grounded; all file paths, the checkpoint tensor layout, the .tex section map, the reviewer plan, and the Zenodo DOI constraint were verified directly this session)

## Summary

Phase 14 is **not a typical code feature** — it is a documentation + release-engineering + recovery phase layered on the (intentionally expanded, per D-14-21/23) re-execution of Phases 10–13 at a matched 2000-epoch budget. The manuscript source (`main (4) copy.tex`, `supp_material.tex`) lives in the repo only as **read-only reference**; the canonical source is external (Overleaf). Phase 14 therefore produces a *revision package* (copy-paste LaTeX blocks keyed to `\label`/anchor sentences + a per-reviewer response document), never an edited `.tex`.

Three hard technical realities shape planning. (1) **The checkpoint is ground truth and recoverable**: `best_checkpoint.pt` was verified this session to contain `params_pqc` shape `(55,)`, `epoch=1969`, scalar `mu`/`sigma`, stored `emd`, and optimizer `param_groups` — the 55-param IQP:SEL circuit must be reconstructed deterministically from this layout (the current `revision/core` default is 75 params for qubits=5/layers=4). (2) **The Zenodo–GitHub DOI cannot be pre-reserved** — `[VERIFIED: Zenodo support FAQ]` GitHub-integration deposits do *not* support `prereserve_doi`; only **manual** Zenodo deposits can reserve a DOI before publishing. This directly governs how the DOI gets cited inside a manuscript minted from the tagged repo (the chicken-and-egg in the phase brief is real and has a specific resolution). (3) **No-hand-typed-numbers is enforceable**: every artifact already carries a `data_hash` and a long-form `rows[]+models[]` schema; the model-info table and docs must render *from* JSON, and a checkable acceptance gate can grep the LaTeX blocks for numeric literals not traceable to a `results/*.json` value.

**Primary recommendation:** Plan in the strict D-14-22 order (recover→equivalence-assert→tiered 2000ep regen→model_info+figures→reconciliation→LaTeX blocks+reviewer_response→release+DOI). Use the **manual Zenodo deposit** path (not GitHub auto-integration) so the version DOI can be reserved, written into the LaTeX blocks, and then published over the exact tagged tree. Build the number-provenance check as a script that the verifier runs, not a manual eyeball.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 55-param circuit reconstruction | `core/models/quantum.py` (config-selectable) | `best_checkpoint.pt` (ground-truth oracle) | D-14-01/02: checkpoint tensor layout drives a deterministic decomposition; added as NON-default circuit (core default stays byte-frozen) |
| Matched-budget re-execution | `run_*.py` + `*_sweep.sh` (`xargs -P2`) | `results/*/sweep_status.json` | D-14-08/12/14: established resumable sweep pattern; tier-gated |
| Number/figure provenance | `run_model_info.py` → `model_info.json` (NEW) | `core/eval` helpers | D-14-16: docs/table render FROM JSON; no hand-typed numbers |
| Figure suite | NEW `revision/` figure module | `run_introspect_figures.py` pattern (PNG+PDF+JSON) | D-14-17: per-model + cross-model + analysis; ≥ canonical set |
| Manuscript edits | Revision package (copy-paste LaTeX blocks) | `.tex` files READ-ONLY | D-14-18: source is Overleaf-external; in-repo `.tex` never edited |
| Reviewer traceability | `docs/reviewer_response.md` (NEW) | `QGAN_Review_Response_Plan.md.pdf` (comment IDs) | D-14-19: per-reviewer point-by-point rebuttal |
| Release freeze + DOI | git tag `v2.0-revision` + **manual** Zenodo deposit | `docs/release.md` (NEW) | D-14-21/22: DOI minted LAST, over final numbers |

## Standard Stack

This phase introduces **no new runtime libraries**. The stack is the existing repo toolchain plus external services for the freeze. Verification commands below use the correct ecosystem registry.

### Core (already in repo — no install)
| Tool | Role in Phase 14 | Notes |
|------|------------------|-------|
| PyTorch | Load `best_checkpoint.pt`; introspect `params_pqc (55,)`, `mu`/`sigma`, optimizer `param_groups` | `torch.load(..., weights_only=False)` — checkpoint stores optimizer state objects `[VERIFIED: loaded this session]` |
| PennyLane (`default.qubit`, `diff_method="backprop"`) | Re-execution backend | LOCKED by D-14-11 — NO `lightning.qubit` swap |
| git | Tag `v2.0-revision`; archaeology on `qgan_pennylane.ipynb` | Repo has only `v1.0` tag today `[VERIFIED: git tag]` |
| matplotlib | Figure suite (PNG + PDF @ dpi=150, `bbox_inches="tight"`) | Pattern in `run_introspect_figures.py` `[VERIFIED: read source]` |

### Supporting (external services for INFRA-03)
| Service | Purpose | When to Use |
|---------|---------|-------------|
| GitHub Releases | Anchor the tagged tree publicly so the DOI deposit references an immutable archive | After all numbers pass the strict gate (D-14-22 step 7) |
| Zenodo (manual deposit) | Mint a citable version DOI + concept DOI | Use the **manual upload + `prereserve_doi`** path, NOT GitHub auto-integration (see Pitfall 1) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual Zenodo deposit | Zenodo↔GitHub webhook auto-integration | `[VERIFIED: Zenodo support FAQ]` GitHub integration **cannot pre-reserve a DOI** → you would have to publish the release, get the DOI, *then* edit the manuscript, *then* the repo no longer matches the DOI'd archive unless you cut a second tag. Manual deposit supports `prereserve_doi`, breaking the chicken-and-egg cleanly. |
| `lightning.qubit` backend | — | Hard-rejected by D-14-11 (forces `adjoint`, reintroduces v1.1 broadcasting bugs, re-baselines Phases 8–13) |
| GitHub auto-DOI | DataCite Fabrica direct | Out of scope — Zenodo is the reviewer-named tool (R1-m4) |

**Installation:** None required. Verify the existing toolchain only:
```bash
python3 -c "import torch, pennylane, matplotlib; print(torch.__version__, pennylane.__version__)"
git --version
```

**Version verification:** No new packages are added in this phase, so the slopcheck/registry audit below is a formal NO-OP, recorded for completeness.

## Package Legitimacy Audit

> Phase 14 installs **zero external packages**. All tooling (torch, pennylane, matplotlib, git) is pre-existing and validated through Phases 8–13. The Zenodo/GitHub dependency is a *web service*, not an installed package.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| *(none — no install step in this phase)* | — | — | — | — | N/A | No-op |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*No `pip install` / `npm install` task should appear in the Phase 14 plan. If the planner finds itself adding a package, that is a scope signal to re-check against D-14-11 (frozen backend) and the no-new-dependencies posture.*

## Architecture Patterns

### System Architecture Diagram

```
                       ┌─────────────────────────────────────────────┐
                       │ best_checkpoint.pt  (GROUND TRUTH, D-14-02)  │
                       │ params_pqc(55,) · epoch 1969 · mu/sigma ·    │
                       │ emd · c_optimizer/g_optimizer param_groups   │
                       └───────────────┬─────────────────────────────┘
                                       │ (1) reverse-engineer decomposition
                                       ▼
   git history of            ┌──────────────────────────┐   config-equivalence
   qgan_pennylane.ipynb ────▶│ 55-param IQP:SEL circuit  │   ASSERT (D-14-07):
   (NUM_LAYERS=2/3/4 seen) ──▶│ added to quantum.py as a  │──▶ load ckpt → shape==55
   (corroborating only)      │ NON-default, locked, config│   structure match? else
                             │ -selectable variant        │   PHASE BLOCKS
                             └────────────┬───────────────┘
                                          │ (2) headline = frozen ckpt
                                          │     load + stored mu/sigma
                                          │     + fixed gen seed (D-14-05)
            ┌─────────────────────────────┼──────────────────────────────┐
            ▼                             ▼                              ▼
  T2 claim-bearing            T3 sensitivity/ansatz          reproducibility-only
  (headline, baseline_        (sensitivity grids,            clean 2000ep retrain
   comparison, tstr)           ansatz_comparison V1/V2/V3)   (NON-load-bearing,
            │                             │                  cross-check vs figs)
            └──────────────┬──────────────┘
                           │ all run at MATCHED 2000ep, seeds {42..46},
                           │ frozen Phase-09.1 data_hash
                           ▼
              ┌───────────────────────────────────────┐
              │ STRICT ACCEPT GATE (D-14-13)           │
              │ • device manifest assertion passed     │
              │ • data_hash matches across ALL artifacts│
              │ • seed set == {42..46}                 │
              │ • long-form JSON schema conforms       │
              │ • full 2000ep (no early-stop headline) │
              └───────────────┬───────────────────────┘
                              │ tier-by-tier acceptance
                              ▼
   run_model_info.py ──▶ results/model_info.json (data_hash)
                              │
            ┌─────────────────┼──────────────────────────┐
            ▼                 ▼                          ▼
   paper-ready model    figure suite          reconciliation note
   table (rendered      (PNG+PDF+JSON,         (1000ep→2000ep deltas)
   FROM JSON)           ≥ canonical 16-fig set)
            └─────────────────┬──────────────────────────┘
                              ▼
   PAPER-01..11 copy-paste LaTeX blocks  +  docs/reviewer_response.md
   (keyed to \label / anchor sentence;  each row → reviewer comment ID)
                              │  HARD-BLOCKED until every cited
                              │  number passes the gate (D-14-22)
                              ▼
   git tag v2.0-revision  ──▶  MANUAL Zenodo deposit (prereserve_doi)
                              ──▶ DOI written into LaTeX blocks
                              ──▶ publish deposit over tagged tree
                              ──▶ docs/release.md (tag SHA + DOI + steps)
```

### Recommended Artifact/Module Layout
```
revision/
├── core/models/quantum.py        # ADD: config-selectable 55-param IQP:SEL (default stays 75)
├── run_model_info.py             # NEW (D-14-16): emits model_info.json
├── run_<figure_suite>.py         # NEW (D-14-17): full per/cross-model figure module
├── run_<2000ep sweep>.py + .sh   # follows xargs -P2 resumable pattern
├── results/
│   ├── model_info.json           # NEW — long-form schema + data_hash
│   ├── <regenerated *.json>      # 1000ep→2000ep regen of existing 161 JSON artifacts
│   └── figures/                  # extend beyond current 13 introspection files
└── docs/
    ├── training_protocol.md      # REGENERATE from JSON (stop hand-maintaining)
    ├── dataset_stats.md          # REGENERATE from JSON
    ├── reviewer_response.md      # NEW (D-14-19): per-reviewer point-by-point
    └── release.md                # NEW (D-14-21): tag SHA + DOI + reproduce steps
```
`docs/release.md` and `docs/reviewer_response.md` **do not exist yet** `[VERIFIED: ls]` — both are new deliverables.

### Pattern 1: Checkpoint-Driven Config Reconstruction
**What:** Treat `best_checkpoint.pt`'s tensor layout as an oracle. The current circuit param formula in `quantum.py` is `num_qubits + num_layers*(num_qubits*3) + num_qubits*2` → `(5,4)`=75 `[VERIFIED: read quantum.py:77-114]`. A 55-param decomposition is **not uniquely determined by the formula** — solving this session showed `(qubits=5, layers=3)` candidate families (e.g. `enc=5 + 3*15 + fin=5 = 55`, or `enc=0 + 3*15 + fin=10 = 55`). Notebook git history shows `NUM_LAYERS` took values 2, 3, **and** 4 over time `[VERIFIED: git log -S]`, so history alone is ambiguous → the checkpoint disambiguates.
**When to use:** Step (1) of every plan, before any sweep (D-14-07 hard-assert).
**Example:**
```python
# Source: verified this session against best_checkpoint.pt
import torch
ck = torch.load("best_checkpoint.pt", map_location="cpu", weights_only=False)
assert ck["params_pqc"].shape == (55,), ck["params_pqc"].shape   # ground truth
assert ck["epoch"] == 1969
# scalar normalization stats — MUST be reused for byte-reproducible headline (D-14-05)
mu, sigma = ck["mu"].item(), ck["sigma"].item()
# c_optimizer/g_optimizer param_groups carry LR/betas breadcrumbs for model_info.json
```

### Pattern 2: Reserved-DOI-First Release (resolves the chicken-and-egg)
**What:** Use Zenodo's **manual** deposit flow which exposes `prereserve_doi` in both the web UI and REST API `[VERIFIED: Zenodo support FAQ]`. Sequence: cut tag `v2.0-revision` → create a Zenodo deposit & reserve the version DOI → write that DOI into the PAPER LaTeX blocks + `docs/release.md` → upload the exact tagged source archive → publish. The reserved DOI is stable once reserved, so the manuscript and the archived tree are mutually consistent.
**When to use:** D-14-22 steps 6–7, hard-blocked until all numbers pass the gate.
**Anti-pattern:** Using the GitHub↔Zenodo webhook integration — it mints the DOI only *on publish* and `[VERIFIED: Zenodo support FAQ]` "It is not possible to pre-reserve DOIs ... with the GitHub linkage."

### Pattern 3: Number-Provenance as a Grep-able Gate
**What:** Every numeric literal in a PAPER-* LaTeX block must be derivable from a value in some `results/*.json`. Make this a script: extract numeric tokens from the LaTeX blocks, and for each, assert it appears (at stated precision) in a JSON artifact. This converts success-criterion 5 from prose into a verifier check.
**When to use:** Acceptance check for the LaTeX-blocks deliverable; run by the verifier.

### Anti-Patterns to Avoid
- **Editing the in-repo `.tex`:** D-14-18 — `main (4) copy.tex` / `supp_material.tex` are read-only reference. Deliver copy-paste blocks keyed to `\label`/anchor sentences.
- **Conflating headline vs reproduction:** D-14-10 — the frozen-checkpoint headline and the clean-2000ep reproduction instance must be reported as *distinct* rows/numbers, never merged.
- **Fresh-computed normalization stats:** D-14-05 — recomputing `mu`/`sigma` breaks byte-for-byte figure reproduction. Always use the checkpoint's stored scalars.
- **Mutating the core default:** core is byte-frozen on the default path — add the 55-param circuit as a NON-default selectable config (like the ARCH-01 topology switch).
- **Mixed-budget tables:** D-14-09 — zero mixed-budget caveats; everything regenerates at 2000ep before paper integration.
- **GitHub-integration DOI:** see Pattern 2 anti-pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resumable parallel sweep | Custom job runner | Existing `run_*.py` + `*_sweep.sh` `xargs -P2` + `sweep_status.json` (D-14-12/14) | Validated M-series thermal cap; skip-already-done semantics already exist; `--parallel ≥3` hard-rejected |
| Figure rendering | New plotting framework | `run_introspect_figures.py` PNG+PDF+JSON pattern | Already enforces "every figure traceable to a reproducibility JSON" (success criterion 4) |
| Eval metrics | Re-implement EMD/ACF/DTW/moments | `revision.core.eval` helpers ONLY (D-10-20) | Provenance rule: `baseline_comparison.json` records `metric_helpers: "revision.core.eval ONLY"` |
| DOI minting / archiving | Tarball + DataCite by hand | Zenodo manual deposit (`prereserve_doi`) | Reviewer-named tool (R1-m4); concept DOI gives a stable "latest version" citation for free |
| Config-equivalence check | Ad-hoc shape print | Phase-8 parity-check harness model | D-14-07 hard-assert; parity harness already proven (`results/parity_check.json`) |

**Key insight:** Phase 14's risk is provenance and sequencing, not algorithm novelty. Every "build" should be a *renderer* (JSON→table/doc/figure) or a *gate* (assert numbers/hashes/seeds match), not new modeling code.

## Runtime State Inventory

> Phase 14 is a recovery + release-freeze phase, so this inventory applies.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data (ground-truth tensors) | `best_checkpoint.pt` — `params_pqc(55,)`, `epoch=1969`, scalar `mu`/`sigma`, stored `emd` (float64), `c_optimizer`/`g_optimizer` `param_groups`, `critic_state` (layers 0..11) `[VERIFIED: torch.load this session]` | Reconstruct 55-param config from it; freeze it as the canonical headline artifact (referenced by hash, NOT committed — see below) |
| Stale 1000ep artifacts | `results/` contains **161 JSON files** + `baseline_comparison.json` etc. produced at the *prior* (1000ep) budget; `tstr.json` even references `epochs: 50` for the soft-sensor `[VERIFIED: grep]`. Comparison runs were 1000ep vs the 2000ep headline checkpoint (the originating unfair-comparison bug). | Regenerate ALL at 2000ep behind the strict gate (D-14-09); archive (not delete) the 1000ep set first to avoid silent mixed-budget contamination (precedent: Phase-09.1 P03, STATE.md) |
| OS / build artifacts | `revision/__pycache__` (25 entries), `core/__pycache__` — stale bytecode after adding the new circuit | None blocking (Python recompiles); no egg-info/installed-package rename in scope |
| Secrets / env vars | None — no SOPS/.env touched. Zenodo needs a personal access token at *deposit time*; it is an operator credential, not a repo secret. | None in repo. Operator supplies Zenodo token interactively at release step |
| Build / freeze artifacts | `.gitignore` excludes `*.pt`, `*.pth`, `results/`, `qgan_env/` `[VERIFIED: grep .gitignore]`. D-14-21 says tag excludes `qgan_env/` and large checkpoints (referenced by hash), **includes** `data.csv` and `revision/results` JSON. Note `results/` IS gitignored today but `results/` is the artifact home — confirm the tag actually captures `results/*.json` (the gitignore `results/` pattern may or may not match the nested path; this must be verified, not assumed). | Planner must add an explicit step: verify `git check-ignore results/*.json` is empty before tagging, else the DOI'd archive ships without its provenance JSON |

**Canonical figures discrepancy (flag for planner):** Context/D-14-17 says "20 preserved canonical figures" but `Final Results from 2000 epochs - IQP:SEL circuit/` actually contains **16 `Figure_*.png`** (Figure_2..21 with gaps: 14, 16, 17, 18 absent) `[VERIFIED: ls | grep -c]`. The "match or exceed the canonical set" completeness bar should be stated as the *actual* 16-figure set, not 20. Resolve this before writing the figure-suite acceptance criterion.

## Common Pitfalls

### Pitfall 1: Zenodo GitHub-integration DOI cannot be pre-reserved
**What goes wrong:** Plan wires up the Zenodo↔GitHub webhook, publishes the release to get a DOI, then realizes the manuscript LaTeX blocks need that DOI *inside* them — but editing the paper after publish means the cited code state no longer matches the DOI'd archive without a second tag.
**Why it happens:** `[VERIFIED: Zenodo support FAQ]` "It is not possible to pre-reserve DOIs before using GitHub integration... However, you can manually upload your release to Zenodo, in which case it is possible to reserve DOIs beforehand" (`prereserve_doi` flag, UI + API).
**How to avoid:** Use the **manual deposit** path: reserve DOI → write into LaTeX + `release.md` → upload the `git archive` of tag `v2.0-revision` → publish. Concept DOI also gives a stable "all versions" citation if a v2.0.1 ever follows.
**Warning signs:** Any plan task that says "enable Zenodo GitHub integration" or "DOI obtained after release published."

### Pitfall 2: 55-param decomposition is non-unique from the formula
**What goes wrong:** Planner assumes `(qubits, layers)` is solvable from `55` alone and hard-codes a guess.
**Why it happens:** `q + L·3q + 2q = 55` has multiple integer families near q=5; notebook history shows `NUM_LAYERS` ∈ {2,3,4} `[VERIFIED: git log -S, solver this session]`. The *gate layout* (whether the IQP encoding contributes params, whether final RX/RY is per-qubit ×1 or ×2) changes the arithmetic.
**How to avoid:** D-14-02 — drive reconstruction from the checkpoint *plus* structural introspection of `quantum.py`'s `generator_circuit` (which already indexes `params_pqc[idx]` sequentially), then D-14-07 hard-assert that loading the checkpoint into the reconstructed circuit yields shape 55 *and* a structurally consistent forward pass.
**Warning signs:** A plan task that fixes `NUM_LAYERS=N` without a checkpoint-load assertion.

### Pitfall 3: Silent CPU/dtype fallback poisons the model-info table
**What goes wrong:** A run claims MPS-float32 but actually ran CPU-float64; the unified table reports a false device, undermining the honesty the reviewers demanded (R1-M5).
**Why it happens:** `training.py` auto-selects cuda→mps→cpu; PennyLane simulators are CPU-only (D-14-11) so the quantum row is *correctly* CPU, but classical rows can silently fall back.
**How to avoid:** D-14-12 — each run emits a device/dtype manifest and **hard-asserts** the actual backend; the strict gate (D-14-13) rejects any artifact whose manifest assertion failed.
**Warning signs:** Missing device manifest in a regenerated artifact; quantum row claiming MPS.

### Pitfall 4: `results/` may be gitignored out of the frozen tag
**What goes wrong:** Tag `v2.0-revision` is cut, Zenodo archive published, but `results/*.json` (the provenance backbone for success-criterion 5) is excluded because `.gitignore` line 62 is `results/`.
**Why it happens:** Broad gitignore pattern; D-14-21 explicitly *wants* the JSON included.
**How to avoid:** Add an explicit pre-tag verification task: `git check-ignore results/<file>.json` must return nothing; if it matches, add a `!results/` negation or force-add. Verify the `git archive` of the tag contains the JSON before the Zenodo upload.
**Warning signs:** `git status` not showing `results/*.json` as tracked; `git ls-files revision/results` empty.

### Pitfall 5: Hand-typed numbers leak into LaTeX blocks
**What goes wrong:** A DTW like `0.6843` (present in the current manuscript at line 266 `[VERIFIED: read main tex]`) is copied into a revision block without a JSON source — violating success-criterion 5.
**Why it happens:** The *existing* manuscript already contains hand-typed headline numbers; revision blocks naturally inherit them.
**How to avoid:** Pattern 3 grep-gate; every numeric literal in a PAPER block must resolve to a `results/*.json` value at stated precision. The reconciliation note (1000ep→2000ep deltas, D-14-13) is the authoritative source for any changed number.
**Warning signs:** A PAPER block citing a metric with no adjacent `# source: results/<file>.json#<path>` annotation.

## Code Examples

### Verify checkpoint is the 55-param oracle (config-equivalence assertion, D-14-07)
```python
# Source: verified against best_checkpoint.pt this session
import torch
ck = torch.load("best_checkpoint.pt", map_location="cpu", weights_only=False)
assert set(ck) >= {"epoch","emd","params_pqc","critic_state",
                   "c_optimizer","g_optimizer","mu","sigma"}
assert tuple(ck["params_pqc"].shape) == (55,)
assert ck["epoch"] == 1969
# headline must reuse THESE scalars (D-14-05), never freshly computed:
mu, sigma = float(ck["mu"]), float(ck["sigma"])
# optimizer breadcrumbs feed model_info.json:
g_lr = ck["g_optimizer"]["param_groups"][0]["lr"]
```

### Long-form artifact schema the model_info emitter must conform to
```python
# Source: results/baseline_comparison.json (read this session)
# Required top-level keys observed: schema, model_kinds, pipelines, seeds,
#   data_hash, data_hash_verification, metric_helpers, rows, models
# "schema": "long-form rows[] + models[] aggregate (D-10-16)"
# "data_hash": "91e447d4624e25b3"   <- must match across ALL artifacts (D-14-13)
# "metric_helpers": "revision.core.eval ONLY (D-10-20)"
```

### Tag + reserved-DOI release sequence (INFRA-03)
```bash
# 1. Freeze (after ALL numbers pass the strict gate — D-14-22)
git tag -a v2.0-revision -m "AIChE aic-4719598 major revision freeze"
git rev-parse v2.0-revision          # SHA -> docs/release.md
git check-ignore results/baseline_comparison.json  # MUST be empty

# 2. Manual Zenodo deposit (prereserve_doi) — NOT GitHub integration
#    Web UI: New upload -> "Reserve DOI" -> copy version DOI
#    Write DOI into PAPER LaTeX blocks + docs/release.md
git archive --format=tar.gz -o v2.0-revision.tar.gz v2.0-revision
#    Upload archive to the reserved deposit -> Publish
#    Concept DOI = stable "all versions" citation; version DOI = this exact freeze
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 1000ep comparison vs 2000ep headline | Matched 2000ep for ALL models | This phase (D-14-08) | Closes the unfair-comparison gap that motivated the expansion |
| 75-param default circuit as "the" circuit | Reconstructed 55-param IQP:SEL is the paper's quantum entrant everywhere (D-14-04) | This phase | Current `revision/core` default is NOT the paper circuit |
| Hand-maintained `training_protocol.md`/`dataset_stats.md` | Regenerated from `model_info.json` (D-14-16) | This phase | Docs stop drifting from artifacts |
| GitHub→Zenodo auto-DOI (common OSS pattern) | Manual Zenodo deposit with `prereserve_doi` | N/A — required by the cite-in-manuscript constraint | Only path that lets the DOI appear inside the frozen manuscript |

**Deprecated/outdated:**
- `lightning.qubit` acceleration: explicitly deferred (D-14-11; STATE deferred-ideas) — would re-baseline Phases 8–13.
- Closed-loop decision pipeline / first-principles Hybrid-GAN: Outlook-only (PROJECT.md out-of-scope; R2-3/R2-5a accept this).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `bib.bib` (referenced by `\bibliography{bib}` in `main (4) copy.tex` line 304) lives in Overleaf and is NOT in the repo | Manuscript reality | Low — PAPER-06/07 ref edits are still delivered as copy-paste `\bibitem`/`.bib` entry blocks regardless of where the `.bib` physically lives; just don't expect to grep it locally `[ASSUMED — no .bib found in repo root via glob]` |
| A2 | The "20 canonical figures" figure-completeness bar should actually be the verified **16**-figure set in `Final Results from 2000 epochs - IQP:SEL circuit/` | Runtime State Inventory | Medium — if the planner writes the acceptance criterion as "≥20 figures" it can never pass; needs user/planner confirmation of intent |
| A3 | The 55-param circuit is `qubits=5, layers=3` family (vs the 75-param `5,4` default) | Pitfall 2 | Medium — this is a *candidate*, not a verified decomposition. D-14-02/07 mandate checkpoint-driven reconstruction; the plan must not hard-code a layer count without the load-assert |
| A4 | `.gitignore` `results/` pattern (line 62) may exclude `results/` from the frozen tag | Pitfall 4 | High if true and unchecked — provenance JSON would be absent from the DOI'd archive; cheap to verify with `git check-ignore` |
| A5 | Zenodo `prereserve_doi` behavior is current as of 2026 | Pitfall 1 / Pattern 2 | Low-Medium — verified via Zenodo support FAQ this session, but Zenodo-RDM has open issues (#831, #47) tracking this; planner should re-confirm in the Zenodo UI at release time |

**If this table is non-empty:** discuss-phase / planner should confirm A2 and A4 before locking acceptance criteria; A1/A3/A5 are handled by existing decisions (D-14-02/07/18) but flagged for awareness.

## Open Questions (RESOLVED)

1. **Does the frozen tag actually capture `results/*.json`?**
   - What we know: `.gitignore` line 62 is `results/`; D-14-21 explicitly wants the JSON in the tag.
   - What's unclear: whether the pattern matches the nested `results/` path in this repo's git config.
   - Recommendation: First task in the release-freeze plan section runs `git check-ignore` + `git ls-files revision/results` and force-tracks if needed (Pitfall 4).
   - **RESOLVED:** Plan 14-07 Task 1 (`verify_freeze_ready.py`) runs `git check-ignore` on each `results/*.json` and raises (explicit-raise gate) before tagging — the pre-tag provenance check is a hard block, not a manual review.

2. **Where is `bib.bib`?**
   - What we know: `\bibliography{bib}` is referenced; no `*.bib` in repo root.
   - What's unclear: exact Overleaf location / current contents for PAPER-06 ref surgery.
   - Recommendation: Deliver PAPER-06/07 as self-contained `.bib`-entry + sentence-rewrite blocks keyed to the `\cite{...}` keys observed in `main (4) copy.tex` (e.g. `\cite{orlandi2024enhancing}`, `\cite{Dallaire_Demers_2018}`) so they apply regardless of `.bib` location.
   - **RESOLVED:** Plan 14-06 Task 1 delivers PAPER-06/07 as location-independent `.bib`-entry + sentence-rewrite blocks keyed to the observed `\cite{}` keys — no dependency on locating the Overleaf-external `.bib`.

3. **Canonical figure count: 16 vs 20.**
   - What we know: only 16 `Figure_*.png` exist on disk.
   - Recommendation: Planner sets the figure-suite completeness bar against the verified 16, and notes the discrepancy in the plan.
   - **RESOLVED:** Plan 14-04 sets the figure-suite completeness bar at the verified ≥16 (not 20) throughout its acceptance criteria.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python + torch + pennylane + matplotlib | Recovery, re-execution, figures | ✓ (used through Phases 8–13) | per repo env | — |
| git | Tag `v2.0-revision`, archaeology | ✓ | system | — |
| `best_checkpoint.pt` | 55-param recovery, headline | ✓ (6.0 MB, present) | epoch 1969 | NONE — phase blocks without it (it IS ground truth, D-14-02) |
| `Final Results from 2000 epochs - IQP:SEL circuit/` | Reproduction cross-check, figure bar | ✓ (16 PNGs) | — | NONE for cross-check; bar adjusts to 16 |
| Zenodo account + personal access token | INFRA-03 DOI mint | ✗ (operator-supplied at release time) | — | NONE — DOI is a hard reviewer ask (R1-m4); operator must authenticate interactively |
| Internet (GitHub Release + Zenodo) | Release freeze step only | assumed ✓ | — | Tag is created offline; DOI deposit needs network |

**Missing dependencies with no fallback:**
- Zenodo credentials at the *release* step (operator action, last task, after all gates pass) — not a blocker for the recovery/re-execution/paper-blocks work which is fully local.

**Missing dependencies with fallback:**
- None affecting the local 90% of the phase.

## Project Constraints (from CLAUDE.md / config)

- **No `./CLAUDE.md`** in working dir `[VERIFIED: ls]` — no project-specific override directives.
- **No `.claude/skills/` or `.agents/skills/`** `[VERIFIED: ls]` — no project skills to honor.
- **`.planning/config.json`:** `nyquist_validation: false` → the **Validation Architecture section is intentionally omitted** (per template skip rule). `commit_docs: true` → researcher commits this RESEARCH.md. `granularity: fine`, `parallelization: true`, interactive mode, quality profile.
- **Compute-heavy phase memory** (`project_quantum_sim_execution.md`): stall-watchdog 45 min / detect 10 min already set in config `executor`; subagent-permission settings apply to the 2000ep sweeps (D-14 discretion notes this).
- **`security_enforcement`:** not present in config and this is a docs/release/recovery phase with no auth/network-input surface (Zenodo auth is operator-interactive, no secrets in repo) → Security Domain section omitted as not applicable (no externally-reachable code path introduced).

## User Constraints (from CONTEXT.md)

### Locked Decisions (verbatim references — full text in 14-CONTEXT.md `<decisions>`)
- **D-14-01..07** Canonical recovery: 55-param IQP:SEL reverse-engineered from `best_checkpoint.pt` (ground truth, D-14-02); headline from frozen ckpt epoch 1969 NOT retrain (D-14-03); 55-param is the quantum entrant in EVERY cross-model comparison (D-14-04); reuse stored `mu`/`sigma` + fixed gen seed (D-14-05); pin native Phase-09.1 pipeline for headline (D-14-06); config-equivalence hard-assert before any sweep (D-14-07).
- **D-14-08..14** Matched-budget: ALL models at 2000ep (D-14-08); full regeneration, zero mixed-budget caveats (D-14-09); V1/V2/V3 also 2000ep, headline vs reproduction reported distinctly (D-14-10); backend = `default.qubit`+`backprop`, NO `lightning.qubit` (D-14-11); `xargs -P2`, device/dtype manifest hard-assert (D-14-12); strict accept gate (D-14-13); run-to-completion, tiered T1/T2/T3 (D-14-14).
- **D-14-15..17** One unified paper-ready table, every model a row (D-14-15); `run_model_info.py` → `model_info.json`, docs render FROM JSON (D-14-16); full figure suite, ≥ canonical set (D-14-17).
- **D-14-18..20** `.tex` is read-only Overleaf reference; deliver copy-paste LaTeX blocks keyed to `\label`/anchor (D-14-18); `docs/reviewer_response.md` per-reviewer (D-14-19); final tone deferred, but **PAPER-02 no-overclaiming is a LOCKED reviewer requirement regardless of result direction** (D-14-20).
- **D-14-21..23** Tag `v2.0-revision` scope (D-14-21); strict gated pipeline, release LAST, DOI mints only over final numbers (D-14-22); ROADMAP/REQUIREMENTS scope-reconciliation update flagged (D-14-23).

### Claude's Discretion
- `model_info.json` schema fields, table column ordering, markdown layout.
- Figure styling, subplot composition, file naming, port-vs-rewrite of legacy plot routines.
- Checkpoint-introspection / git-archaeology mechanics for the 55-param decomposition.
- Sweep driver structure (follow `run_*.py` + `*_sweep.sh` `xargs -P2` resumable pattern).
- Stall-watchdog / subagent-permission settings per compute-heavy phase memory.

### Deferred Ideas (OUT OF SCOPE)
- `lightning.qubit` backend acceleration (future perf milestone only).
- Closed-loop decision pipeline & first-principles Hybrid-GAN (Outlook-only, not implemented).
- New variance-collapse remediation / circuit-architecture re-attempt (v2.0 reports honestly).
- Hardware/QPU execution (simulator-only).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PAPER-01 | Reframe hypothesis in Section 1; soften quantum-necessity transition | Anchor: `main (4) copy.tex` §1 `\section{Introduction}` (line 61), §1.4 `\subsection{Quantum Generative Adversarial Networks}` (line 90), §2.4 (line 146–151). Reviewer plan R1-M5/R2-1/R2-2 give the exact reframed hypothesis text. Deliver as block keyed to anchor sentence at line 92/151. |
| PAPER-02 | Remove/soften overclaiming language | LOCKED (D-14-20). Targets verified in `main (4) copy.tex`: "industrial bioprocess engineering" (Concluding Remarks line 296), "high fidelity ... industrial bioprocesses" (line 266), "exponentially more compactly" (line 151). Block-replace each anchor sentence. |
| PAPER-03 | Circuit Design Rationale subsection | Reviewer plan R2-5b enumerates the 3 sub-points (why 5 qubits, ansatz expressibility/trainability, classical critic + quantum generator). New `\subsection` after §3.1 `\subsection{QWGAN-GP Architecture Overview}` (line 155). Numbers sourced from `ansatz_comparison.json` (2000ep, D-14-10). |
| PAPER-04 | Log-returns bioprocess justification | Supp already has finance-framed log-return rationale at `supp_material.tex` lines 358–365 ("highly favored in quantitative analysis"). Rewrite to growth-rate/bioprocess framing; block keyed to `supp_material.tex` §"Data Transformation Details" (line 352). |
| PAPER-05 | Move decision-tree + Hybrid-GAN to Outlook; caveat/remove Supp Table A2; fix 20L/300L | Hybrid-GAN material at `supp_material.tex` §"Hybrid-GAN Framework for Future Work" (line 142, `\label{fig:qgan_hybrid_appraoch}`); decision-tree `\label{fig:qgan_schemcatic}` (line 340); aspirational table `\label{tbl:various_approaches}` (line 226, Hybrid-GAN "Proposed" row line 242). Main-text future-work block at line 286. 20L/300L mismatch: `main (4) copy.tex` line 178 (`\label{fig:lucy}` text "300L configuration of the 20L version") + supp caption line 346. |
| PAPER-06 | Reference surgery | Reviewer plan R1-m1 table gives exact fixes for [27],[28],[39],[18],[19],[41],[55]–[57],[59]; keep anchors [21]–[23],[34]–[36],[61]. `.bib` is Overleaf-external (A1) — deliver `.bib`-entry + sentence-rewrite blocks keyed to `\cite{...}` keys in `main (4) copy.tex`. |
| PAPER-07 | Add Bernal et al. | "Perspectives of quantum computing for chemical engineering" — cite in §1.3/§2 transition (line 90/146). Reviewer R1-m6/R2-2. New `\bibitem`/`.bib` block + insertion sentence. |
| PAPER-08 | Dataset details in Methods | Source: regenerated `model_info.json` + `docs/dataset_stats.md` (778 raw → 777 log-returns → 384 windows; single-campaign LUCY). Insert into §3 Methods after Photobioreactor setup (line 176). MUST render from JSON (D-14-16, success-criterion 5). |
| PAPER-09 | Per-metric eval scale (transformed vs OD) | Source: `fidelity_dualscale.json` (dual-scale OD + log_return, 2000ep). Methods table block; every metric labeled. R1-m3. |
| PAPER-10 | Appendix A3 log-GAN vs Wasserstein discrepancy | A3 / Hybrid-GAN math at `supp_material.tex` lines 156–252 (`eq:balance`, `eq:constraint1/2`, `eq:constitutive`). R2-5a: relabel as "proposed extension," clarify the log-GAN vs Wasserstein eq. discrepancy. |
| PAPER-11 | Typos + notation unify | Reviewer plan R1-m7 gives the full checklist: Fig.6 "Laas"→"Lags"; "Figure A5).This"→". This"; "LUCY ©photobioreactor"→"LUCY® photobioreactor" (matches `supp_material.tex` line 346 `\textcopyright`); 300L/20L sentence (`main` line 178 — note literal `\label{fig:lucy}` mid-sentence is malformed); "Dry Biomass"→"dry biomass"; bio-manufacturing/biomanufacturing; Ref[39]"Approac"→"Approach"; Ref[51] caps; "QWGAN-GPs"→"QWGAN-GP" (Concluding Remarks); single return symbol (log δ vs ς); enlarge Figs 2–6. Each = one keyed block. |
| INFRA-03 | Tag `v2.0-revision` + Zenodo DOI cited in manuscript | Manual Zenodo deposit (`prereserve_doi`, Pattern 2 — GitHub integration cannot pre-reserve). Update Data Availability stmt `main (4) copy.tex` line 292 (currently just a GitHub URL, no DOI). `docs/release.md` (NEW) records tag SHA + DOI + reproduce steps. Hard-blocked until all numbers pass the gate (D-14-22). |

## Sources

### Primary (HIGH confidence — verified this session)
- `best_checkpoint.pt` — `torch.load` introspection: keys `epoch=1969, emd, params_pqc(55,), critic_state, c_optimizer, g_optimizer, mu, sigma`
- `main (4) copy.tex` — section/label/cite map (lines 58–304); `\bibliography{bib}`, `\bibliographystyle{ama}`, `natbib`
- `supp_material.tex` — Hybrid-GAN/A3/decision-tree/log-return/Table-A2 locations
- `QGAN_Review_Response_Plan.md.pdf` (pp.1–6) — full R1-M1..M5, R1-m1..m7, R2-1..6 itemized issues + proposed addressments
- `core/models/quantum.py` (param formula `q + L·3q + 2q`), `core/__init__.py` (75-param default), `run_introspect_figures.py` (PNG+PDF+JSON pattern)
- `results/baseline_comparison.json` (long-form schema + `data_hash` + `metric_helpers` D-10-20)
- `.gitignore`, `git tag` (only `v1.0`), `git log -S NUM_LAYERS` (history shows NUM_LAYERS ∈ {2,3,4}), `ls Final Results.../` (16 figures)

### Secondary (MEDIUM confidence — official source, verify at use)
- Zenodo support FAQ — GitHub integration cannot pre-reserve DOI; manual deposit supports `prereserve_doi`; concept vs version DOI semantics

### Tertiary (LOW confidence — flagged)
- Zenodo `prereserve_doi` currency in 2026 (open Zenodo-RDM issues #831/#47) — re-confirm in UI at release time (A5)

## Metadata

**Confidence breakdown:**
- Manuscript reality / requirement mapping: HIGH — every PAPER-* anchor located in the actual .tex
- Checkpoint recovery: HIGH for the oracle (verified tensor layout); MEDIUM for the *exact* decomposition (non-unique by formula — D-14-02/07 mitigate)
- Release/DOI workflow: MEDIUM — Zenodo behavior from official FAQ, but re-verify at execution
- Pitfalls: HIGH — derived from verified repo state + verified Zenodo docs

**Research date:** 2026-05-19
**Valid until:** ~2026-06-18 (30 days; stable repo, but re-verify Zenodo `prereserve_doi` UI at the release step)
