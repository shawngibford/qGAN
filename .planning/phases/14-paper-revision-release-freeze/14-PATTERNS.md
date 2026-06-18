# Phase 14: Paper Revision & Release Freeze - Pattern Map

**Mapped:** 2026-05-19
**Files analyzed:** 9 new/modified artifacts
**Analogs found:** 6 with code analogs / 9 total (3 are pure-prose/manual — explicitly no code analog)

> Phase 14 is a documentation + release-engineering + recovery phase. Per
> RESEARCH "Key insight": every "build" is a *renderer* (JSON→table/doc/figure)
> or a *gate* (assert numbers/hashes/seeds), not new modeling code. The analogs
> below are the established renderer / sweep / aggregator / config-selectable-core
> idioms already proven through Phases 8–13.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `core/models/quantum.py` (ADD 55-param IQP:SEL, non-default) | model/config | transform | same file — existing `topology` ARCH-01 config-selectable switch (`quantum.py:38-80`) | exact (in-file precedent) |
| `run_model_info.py` (NEW) | utility (JSON emitter) | transform / batch | `run_multiseed_rollup.py` (pure aggregator → long-form JSON + data_hash) | exact (role + data flow) |
| `run_<2000ep sweep>.py` + `.sh` (NEW) | driver + sweep harness | batch / event-driven (resumable) | `run_ansatz_sweep.sh` + `run_ansatz.py` (xargs -P2, sweep_status.json) | exact |
| `run_<figure_suite>.py` (NEW) | utility (figure renderer) | transform (JSON→PNG/PDF) | `run_introspect_figures.py` (render-only, PNG+PDF+JSON) | exact |
| Config-equivalence assertion (T1 gate; in recovery script) | test/gate | request-response (assert) | `run_multiseed_rollup.py:80-92` cross-artifact hard-gate idiom + Phase-8 `parity_check.json` harness | role-match |
| `docs/training_protocol.md` / `dataset_stats.md` (REGEN from JSON) | doc (rendered) | transform (JSON→md) | `results/baseline_comparison.md` rendered by `_build_baseline_notebook.py:550-593` | exact (md-from-JSON renderer) |
| `docs/reviewer_response.md` (NEW) | doc (structured prose) | n/a | `docs/training_protocol.md` (sourced-table doc structure) + 13-04-SUMMARY.md front-matter discipline | role-match (structure only) |
| `docs/release.md` (NEW) | doc (process record) | n/a | RESEARCH "Tag + reserved-DOI" code block; no in-repo doc analog | **no analog** (see below) |
| PAPER-01..11 copy-paste LaTeX blocks | manuscript revision package | n/a | none — `.tex` is read-only (D-14-18) | **no analog** (see below) |

## Pattern Assignments

### `core/models/quantum.py` — ADD 55-param IQP:SEL (model/config, transform)

**Analog:** the SAME file's existing ARCH-01 `topology` config-selectable switch. The 55-param circuit must be added the *identical* way (D-14-04, RESEARCH "add as NON-default selectable config"). Core default path stays byte-frozen.

**Config-selectable switch pattern** (`quantum.py:38-63`) — copy this shape for the new circuit selector:
```python
#: Allowed entangling-CNOT topologies. ``"range"`` is the v1.0/v1.1 default
#: (wrap-around range pattern, byte-identical to pre-Phase-13 code).
_TOPOLOGIES = ("range", "linear")
...
def __init__(self, num_qubits: int = 5, num_layers: int = 4,
             window_length: int = 10, diff_method: str = "backprop",
             topology: str = "range") -> None:
    super().__init__()
    if topology not in self._TOPOLOGIES:          # eager validate, fail at construction
        raise ValueError(f"Unknown topology {topology!r}; expected one of {self._TOPOLOGIES}")
    assert window_length == 2 * num_qubits, (...)  # v1.0 invariant preserved
```

**Param-count formula to re-derive against the checkpoint** (`quantum.py:77-80`):
```python
# IQP (num_qubits) + num_layers * (num_qubits * 3 Rot params) + final RX/RY (num_qubits * 2)
self.num_params = num_qubits + num_layers * (num_qubits * 3) + num_qubits * 2
# (5,4)=75 today. 55 is NON-unique by formula (RESEARCH Pitfall 2) — drive the
# decomposition from best_checkpoint.pt params_pqc.shape==(55,), NOT a hard-coded layer count.
```

**Config-equivalence hard-assert (D-14-07)** — use the `run_multiseed_rollup.py:87-91` *explicit-raise* idiom (NOT bare `assert`, which `python -O` strips and would silently disable the integrity gate):
```python
# Source: run_multiseed_rollup.py:86-92 (WR-01 explicit-raise gate)
if len(set(hashes.values())) != 1:
    raise AssertionError(f"data_hash mismatch across headline artifacts: {hashes}")
```
Applied here: load `best_checkpoint.pt` into the reconstructed config, then
`if tuple(ck["params_pqc"].shape) != (55,): raise AssertionError(...)` and a
structural forward-pass check. Failure BLOCKS the phase (D-14-07).

---

### `run_model_info.py` (NEW) — utility / JSON emitter (transform)

**Analog:** `run_multiseed_rollup.py` — the canonical "pure consumer / pure aggregator → long-form JSON + data_hash" driver. No torch/pennylane/core import; reads result JSONs, asserts cross-artifact `data_hash`, emits one JSON.

**Repo-root resolver + RESULTS anchor** (`run_multiseed_rollup.py:42-59`) — copy verbatim (drivers may run from a worktree; results paths are repo-root anchored):
```python
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (core/preprocessing.py)")

REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RESULTS = REPO / "revision/results"
```

**Cross-artifact data_hash gate** (`run_multiseed_rollup.py:85-92`) — every consumed artifact must agree on `data_hash` (D-14-13 strict gate). Reuse exactly.

**Long-form schema the emitter MUST conform to** — from `results/baseline_comparison.json` (read this session). Top-level keys observed: `schema`, `model_kinds`, `pipelines`, `seeds`, `data_hash`, `data_hash_verification`, `metric_helpers`, `models`, `rows`. Per-`models[]` shape (copy this record structure for each model row in `model_info.json`):
```json
{ "kind": "quantum", "parameter_count": 75, "family": "adversarial-quantum",
  "train_protocol_notes": "QuantumGenerator(num_qubits=5, num_layers=4) PQC = 5 + 4*15 + 10 = 75 params; ..." }
```
And the header literals to match: `"schema": "long-form rows[] + models[] aggregate (D-10-16)"`, `"data_hash": "91e447d4624e25b3"` (must match across ALL artifacts), `"metric_helpers": "revision.core.eval ONLY (D-10-20)"`. For Phase-14 the 55-param IQP:SEL row replaces the 75-param quantum row (D-14-04); add a `device`/`dtype` field from the device manifest (D-14-12).

**Output write idiom** (`run_multiseed_rollup.py:176-187`):
```python
out = {"schema": "...", "data_hash": canonical_hash, "consumed_artifacts": {...}, ...}
(RESULTS / "model_info.json").write_text(json.dumps(out, indent=2))
print(f"model_info.json written: {len(rows)} rows, data_hash={canonical_hash}")
```

---

### `run_<2000ep sweep>.py` + `.sh` (NEW) — driver + resumable sweep harness (batch)

**Analog:** `run_ansatz_sweep.sh` (+ `run_ansatz.py` per-run CLI). This is THE established `xargs -P2` resumable pattern (D-14-12/14, RESEARCH "Don't Hand-Roll"). Copy the whole skeleton; change only the matrix (VARIANTS/SEEDS/EPOCHS) and artifact-bundle definition.

**Thermal guardrail** (`run_ansatz_sweep.sh:147-157`) — `--parallel` must be 1 or 2; `>=3` hard-rejected with non-zero exit (LOCKED D-14-12, `--parallel ≥3` hard-rejected):
```bash
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]] || [[ "$PARALLEL" -gt 2 ]]; then
  echo "ERROR: --parallel must be 1 or 2 (got: '${PARALLEL}')." >&2
  exit 3
fi
```

**xargs -P2 dispatch, NEVER multiprocessing.Pool** (`run_ansatz_sweep.sh:394-409`) — Pitfall 5 LOCKED (Pool fork shares warm numpy RNG → corrupts reproduction):
```bash
< "$WORKLIST" xargs -P 2 -L 1 bash -c 'run_one "$0" "$1"'
```

**Resumable `sweep_status.json` (skip-already-done)** — `is_complete()` (`run_ansatz_sweep.sh:175-183`) checks all artifacts exist & non-empty; `run_one()` short-circuits complete pairs (`:294-300`); status written atomically via tmp-file + `os.rename` under `flock` advisory lock (`:264-281`). Status schema documented at `run_ansatz_sweep.sh:46-61`. Reuse all of this; add 2000ep + the device/dtype manifest hard-assert + `data_hash` check to the per-run accept logic (D-14-12/13).

**Venv-binary selection** (`run_ansatz_sweep.sh:98-108`) — invoke `./qgan_env/bin/python` directly (its activate script has a stale hardcoded path); copy verbatim.

---

### `run_<figure_suite>.py` (NEW) — figure renderer (transform JSON→PNG/PDF)

**Analog:** `run_introspect_figures.py` — render-only, every figure traceable to a reproducibility JSON (success criterion 4). D-14-17 explicitly names this as the pattern to follow; port the notebook's ~11 `savefig` routines into this shape for the full per-model suite.

**Headless matplotlib** (`run_introspect_figures.py:24-29`):
```python
import matplotlib
matplotlib.use("Agg")  # headless render before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
```

**Loud-fail on missing companion JSON — render-only, never a silent partial figure** (`run_introspect_figures.py:60-73`):
```python
def _load_json(figures_dir: Path, name: str) -> dict:
    path = figures_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"[run_..._figures] required companion JSON missing: {path}. "
            f"This renderer is render-only (no training/sampling); run the "
            f"generating step first.")
    return json.loads(path.read_text())
```

**PNG+PDF dual save @ dpi=150, bbox tight** (`run_introspect_figures.py:76-84`) — the success-criterion-4 contract:
```python
def _save(fig, figures_dir: Path, stem: str) -> list[Path]:
    written = []
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        written.append(out)
    plt.close(fig)
    return written
```

**CLI + repo-root + print-written-paths idiom** (`run_introspect_figures.py:51-57, 291-325`): `argparse --figures-dir` defaulting to `figures`, `_find_repo_root()` walking up to `core/preprocessing.py`, resolve relative dir against repo, `print` every written path. Mirror exactly. **Figure-completeness bar = the verified 16 `Figure_*.png`** in `Final Results from 2000 epochs - IQP:SEL circuit/`, NOT 20 (RESEARCH Runtime State + Open Q3 / Assumption A2).

---

### `docs/training_protocol.md` / `dataset_stats.md` (REGEN from JSON) — rendered doc (transform)

**Analog:** `results/baseline_comparison.md`, rendered FROM JSON by the markdown-render cell in `_build_baseline_notebook.py:550-593`. This is the exact "no hand-typed numbers, render markdown table from the long-form JSON" idiom (D-14-16, success criterion 5).

**JSON→markdown table render** (`_build_baseline_notebook.py:561-592`) — aggregate from `rows[]`, format cells, emit pipe-table lines:
```python
param_of = {m["kind"]: m["parameter_count"] for m in models}
lines = ["# BASE-03 — Classical Baselines Apples-to-Apples Comparison\n", ...,
         f"`data_hash` = `{expected_data_hash}` recomputed once and verified equal ..."]
for p in PIPELINES:
    lines.append(f"\n## Pipeline {p}\n")
    lines.append("| model | parameter_count | OD-EMD (mean±std) | ... |")
    lines.append("|---|---|---|---|---|---|---|")
    for mk in MODEL_KINDS:
        emd_m, emd_s = _agg(mk, p, "emd", "OD")          # mean/std straight off rows[]
        emd_c = f"{emd_m:.4f} ± {emd_s:.4f}" if emd_m is not None else "—"
        lines.append(f"| {mk} | {param_of.get(mk, '—')} | {emd_c} | ... |")
```
Note `_agg()` (`:553-559`) computes mean/std directly from `comparison["rows"]` — no number is ever typed. Phase-14 docs MUST be regenerated by an analogous renderer reading `model_info.json` (D-14-16: "the markdown docs stop being hand-maintained").

**Current hand-maintained doc structure to preserve** (`training_protocol.md:1-26`): a "Source of truth" callout + per-section pipe tables with a `| Constant | Value | Source |` schema and per-row file:line citations. The regenerated version keeps this layout but the Value/Source columns are driven by `model_info.json` provenance instead of hand-typed `core/__init__.py:NN` citations.

---

### `docs/reviewer_response.md` (NEW) — structured-prose doc (no data flow)

**Analog (structure only):** `docs/training_protocol.md` table discipline + the 13-04-SUMMARY.md front-matter `requires/provides/decisions/key-files` discipline. No exact content analog — this is a new AIChE point-by-point rebuttal artifact (D-14-19).

**Pattern to copy:** the sourced-row table convention from `training_protocol.md` (every claim carries a `| ... | Source |` provenance column). For `reviewer_response.md` the per-reviewer table is:
`| comment ID | verbatim concern | change made | manuscript location (§/table/fig) | supporting artifact (results/*.json or figure path) |`
Comment IDs come from `QGAN_Review_Response_Plan.md.pdf` (R1-M1..M5, R1-m1..m7, R2-1..6 — RESEARCH Sources). Every "supporting artifact" cell must point at a real `results/*.json` or `figures/*` path (success criterion 5, Pitfall 5).

---

### `docs/release.md` (NEW) — **NO CODE ANALOG** (process record)

No in-repo document plays this role (`docs/` holds only `training_protocol.md`, `dataset_stats.md`). Use the RESEARCH "Tag + reserved-DOI release sequence" block as the content spec, not a code analog. It must record: tag SHA (`git rev-parse v2.0-revision`), the reserved Zenodo version DOI + concept DOI, the `git check-ignore results/baseline_comparison.json` pre-tag verification result (must be empty — Pitfall 4), and copy-paste reproduce steps. Manual Zenodo deposit only (`prereserve_doi`) — NEVER the GitHub↔Zenodo webhook (RESEARCH Pattern 2 / Pitfall 1).

---

### PAPER-01..11 copy-paste LaTeX blocks — **NO CODE ANALOG** (manuscript revision package)

D-14-18: `main (4) copy.tex` and `paper/supp_material.tex` are READ-ONLY reference (Overleaf-canonical) — they are NEVER edited in-repo and there is no "edit the tex" analog by design. Deliverable is a set of copy-paste LaTeX blocks keyed to `\label`/anchor sentence + a one-line reviewer-comment rationale each. Anchors are fully enumerated in 14-RESEARCH.md "Phase Requirements" (PAPER-01..11 each cite exact `main (4) copy.tex` / `paper/supp_material.tex` line numbers and `\label`/`\cite` keys). The only enforceable code artifact here is the **number-provenance grep-gate** (RESEARCH Pattern 3): a verifier script that extracts numeric literals from the LaTeX blocks and asserts each resolves to a `results/*.json` value at stated precision — pattern-source for that gate is the explicit-raise cross-artifact gate in `run_multiseed_rollup.py:85-92`.

## Shared Patterns

### Repo-root resolution + repo-anchored results path
**Source:** `run_multiseed_rollup.py:42-59` (also `run_introspect_figures.py:51-57`, `run_ansatz_sweep.sh:98-108` venv variant)
**Apply to:** every new `run_*.py` and the new sweep `.sh`
Walk up to `core/preprocessing.py`; anchor `RESULTS = REPO / "revision/results"`. Drivers run from worktrees — never assume cwd.

### Explicit-raise integrity gate (NOT bare assert)
**Source:** `run_multiseed_rollup.py:86-92` (WR-01)
**Apply to:** config-equivalence assertion (D-14-07), the D-14-13 strict accept gate, the number-provenance grep-gate
`python -O` strips `assert`; integrity/data_hash/seed/shape gates must `raise AssertionError(...)` explicitly so they cannot be silently disabled.

### Cross-artifact data_hash equality (D-14-13 strict gate)
**Source:** `run_multiseed_rollup.py:85-92`; expected `91e447d4624e25b3`
**Apply to:** `run_model_info.py`, the 2000ep sweep accept logic, every regenerated artifact
Assert ONLY mutual equality of the frozen `data_hash` fields across all consumed artifacts; do NOT re-derive the hash (Anti-Pattern, run_multiseed_rollup.py:18-23). Also assert seed set == `{42,43,44,45,46}` and device-manifest assertion passed (D-14-12/13).

### Long-form `schema + rows[] + models[]` JSON contract
**Source:** `results/baseline_comparison.json` (schema header `"long-form rows[] + models[] aggregate (D-10-16)"`)
**Apply to:** `model_info.json` and every regenerated 2000ep artifact
Conform top-level keys (`schema, data_hash, metric_helpers, models, rows`); metric helpers `revision.core.eval ONLY (D-10-20)` — never re-implement EMD/ACF/DTW/moments.

### Render-only + no-hand-typed-numbers
**Source:** `run_introspect_figures.py` (figures), `_build_baseline_notebook.py:550-593` (markdown)
**Apply to:** figure suite, regenerated `training_protocol.md`/`dataset_stats.md`, the model-info table
Renderers read JSON and fail loudly on a missing companion; every number in a doc/figure/table/LaTeX block traces to a `results/*.json` value (success criteria 4 & 5, RESEARCH Pitfall 5).

### Resumable sweep harness (xargs -P2, never Pool)
**Source:** `run_ansatz_sweep.sh` end-to-end
**Apply to:** the new 2000ep sweep
`--parallel` 1|2 only (>=3 rejected, D-14-12); `xargs -P 2 -L 1 bash -c`; `sweep_status.json` skip-already-done with atomic tmp+rename under flock; venv-binary direct invocation; per-run device/dtype manifest hard-assert (D-14-12).

## No Analog Found

Planner should treat these via RESEARCH.md specs, not a codebase analog:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `docs/release.md` | process-record doc | n/a | No release/DOI doc exists in repo; spec = RESEARCH "Tag + reserved-DOI release sequence" + Pitfall 1/4. Manual Zenodo `prereserve_doi` only. |
| PAPER-01..11 LaTeX blocks | manuscript revision package | n/a | `.tex` is read-only by D-14-18 — there is intentionally no "edit the manuscript" code path. Anchors enumerated in 14-RESEARCH Phase Requirements. |
| Git tag + Zenodo deposition | manual operator process | n/a | Operator-interactive (Zenodo token at deposit time); INFRA-03 last step, hard-blocked behind the gate (D-14-22). Not a source file. |

## Metadata

**Analog search scope:** `run_*.py`, `revision/*sweep*.sh`, `_build_*notebook.py`, `core/models/quantum.py`, `docs/*.md`, `results/*.json` + `*.md`, `.planning/phases/13-*/` summaries
**Files scanned:** ~22 (6 read in full / targeted; structures of run_baselines, ansatz, sensitivity drivers confirmed by listing + grep)
**Key patterns identified:**
- Pure-aggregator JSON emitter idiom (`run_multiseed_rollup.py`) is the template for `run_model_info.py`
- Render-only PNG+PDF+JSON figure idiom (`run_introspect_figures.py`) is the D-14-17 template, bar = 16 figures (not 20)
- `xargs -P2` resumable `sweep_status.json` harness (`run_ansatz_sweep.sh`) is the 2000ep sweep template (Pool forbidden, --parallel ≥3 forbidden)
- Config-selectable in-`quantum.py` switch (existing `topology` ARCH-01) is the exact precedent for adding the non-default 55-param IQP:SEL
- JSON→markdown table render (`_build_baseline_notebook.py:550-593`) is the no-hand-typed-numbers doc-regeneration template
- `release.md`, the LaTeX blocks, and the Zenodo process have NO code analog — spec from RESEARCH, not invented

**Pattern extraction date:** 2026-05-19
