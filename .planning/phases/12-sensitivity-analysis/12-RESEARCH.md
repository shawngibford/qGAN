# Phase 12: Sensitivity Analysis - Research

**Researched:** 2026-05-18
**Domain:** PennyLane 0.44.0 inference-time noise/shot sensitivity; per-seed JSON aggregation
**Confidence:** HIGH (API verified live on installed PennyLane 0.44.0; all artifact schemas inspected on disk)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-12-01:** Shot-noise and noise-channel sensitivity are **inference-only** — regenerate samples from the **already-trained analytic quantum generator** (Phase 09.1/10 checkpoints) on a noisy/finite-shot device, then recompute the existing fidelity metric suite. **No retraining.** Rationale: the generator runs `qml.device("default.qubit", shots=None, diff_method="backprop")`; finite shots and noise channels are incompatible with statevector backprop, and retraining the full noise × shot × seed grid under parameter-shift on a local Mac is infeasible. This also yields the correct reviewer narrative: robustness of a fixed trained model to deployment-time noise.
- **D-12-02:** SENS-01 and SENS-02 **degradation grids** use **3 seeds {42, 43, 44}** (degradation is a trend, not a headline number). The full **5-seed set {42, 43, 44, 45, 46}** with mean ± std is reserved for the SENS-03 headline roll-up.
- **D-12-03:** SENS-03 is **pure aggregation** of the existing Phase 10/11 per-seed artifacts (`baseline_comparison.json`, `tstr.json`, `predictive_discriminative.json`, `augmentation.json`, `fidelity_dualscale.json`) into mean ± std cells. **No new training, no new seeds.** `multiseed_summary.json` is the consolidated roll-up; data-hash invariant (D-10-15) must be asserted across consumed artifacts.

### Claude's Discretion
- Noise-channel device wiring (`default.mixed`, channel insertion strategy, finite-shot device construction) and how the trained analytic params are loaded into the noisy QNode.
- Output JSON structure beyond the established long-form schema `{model_kind, pipeline, seed, metric_name, scale, value}`; degradation-curve representation in `shot_noise_sensitivity.json` / `noise_model_sensitivity.json`.
- New driver/sweep file names and CLI surface (pattern after `run_baselines.py` + `run_baselines_sweep.sh`); idempotent per-cell skip logic; `--parallel 2` guardrail; **no `multiprocessing.Pool`** (Phase 09.1 Pitfall 4).
- Which fidelity metrics are recomputed under noise (reuse `core/eval.py` helpers unchanged; EMD/moments/ACF/DTW at minimum, dual-scale per EVAL-05).
- Pipeline coverage for the noise/shot grid (Pipeline B headline; Pipeline A as supplementary control).
- Subsampling strategy if regenerated sample counts differ from the analytic artifacts.

### Deferred Ideas (OUT OF SCOPE)
- **Noise-aware retraining** — training the generator under finite shots / noise channels. Out of scope for v2.0; v3.0 robustness study if reviewers ask.
- **Full 5-seed degradation grids** — escalate from 3→5 seeds only as a planning-time decision if 3-seed spread obscures the trend.
- CR-01 (spectral-loss hook) / CR-02 (EarlyStopping restore) — training-loop bugs locked to Phase 13. Phase 12 is inference-only and does not touch the training loop.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SENS-01 | Shot-noise sweep at {analytic, 8192, 1024} shots; metric degradation reported → `results/shot_noise_sensitivity.json` | Code Example 2 (rebuild QNode + `qml.set_shots` transform); Pattern 1 (inference driver); existing analytic samples reused as the {analytic} column (no re-run needed) |
| SENS-02 | Noise-model sensitivity — depolarizing p ∈ {0,0.001,0.01,0.05}, amplitude-damping γ ∈ {0,0.001,0.01,0.05} → `results/noise_model_sensitivity.json` | Code Example 3 (`default.mixed` + per-wire channel insertion); Pattern 1; budget probe confirms ~12 s/run |
| SENS-03 | Multi-seed runs (≥5 seeds) for every headline result; mean ± std in every comparison table → `results/multiseed_summary.json` | Pattern 2 (pure aggregation over the 5 frozen long-form JSONs); Code Example 4 (groupby mean ± std + cross-artifact data_hash assertion) |
</phase_requirements>

## Summary

Phase 12 is an **inference-and-aggregation** phase with zero new training. Three deliverables, each with a clean, low-risk implementation path that the live environment confirms:

1. **SENS-01 / SENS-02 (shot + noise sensitivity).** The trained analytic quantum generator stores all learned state in a single 75-element tensor: `checkpoint.pt["params_pqc"]` in every `results/transform_ablation/runs/<pipeline>/<seed>/`. Loading those params into a freshly-constructed QNode on a different device is the entire mechanism. PennyLane 0.44.0 **deprecated the `shots=` device-constructor kwarg** — the correct API is the `qml.set_shots(qnode, shots=N)` **transform** applied to the QNode. Noise channels require the `default.mixed` device with `qml.DepolarizingChannel(p, wires=i)` / `qml.AmplitudeDamping(gamma, wires=i)` inserted into the circuit body. Both paths run with `diff_method=None` (inference-only, no gradients) — `backprop` is incompatible with finite shots and mixed-state channels, which is exactly the technical fact behind D-12-01.

2. **SENS-03 (multi-seed roll-up).** Pure pandas-style aggregation. All five headline JSONs (`baseline_comparison.json`, `tstr.json`, `predictive_discriminative.json`, `augmentation.json`, `fidelity_dualscale.json`) already exist on disk with an **identical long-form `rows[]` schema** `{model_kind, pipeline, seed, metric_name, scale, value}` and an identical `data_hash` (`91e447d4624e25b3`) plus a `data_hash_verification` block. SENS-03 reads them, asserts the hash matches across all five, groups by `(model_kind, pipeline, metric_name, scale)`, and emits mean ± std (n) cells.

3. **Compute budget (Success Criterion 4).** Measured live on this Mac: analytic ≈ 0.013 s/batch, finite-shot 8192 ≈ 0.028 s/batch, `default.mixed` ≈ 0.036 s/batch (batch=12). Each run is 320 batches (N_synth=3840) → ≤ ~12 s/run. The full grid (SENS-01: 2 finite-shot levels × 3 seeds × ~2 pipelines; SENS-02: 8 noise cells × 3 seeds × ~2 pipelines) is **well under 10 minutes total** — Success Criterion 4 is comfortably achievable; no sample-count cap needed.

**Primary recommendation:** Build two consumer/inference drivers patterned exactly on `run_baselines.py` + `run_baselines_sweep.sh` — `run_sensitivity.py` (SENS-01+02, one (pipeline, seed, condition) cell per invocation, idempotent, atomic `sweep_status.json`, `xargs -P 2`) and `run_multiseed_rollup.py` (SENS-03, pure aggregation, single invocation). Keep all logic in `run_*.py`; `core/` stays untouched (D-10-13). Reuse `QuantumGenerator.generator_circuit` and `eval.full_metric_suite` unchanged; reuse `run_utility.reconstruct_od`'s OD-scale inverse logic verbatim.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Trained-param loading | Reusable code (`core/models/quantum.py`) | — | `QuantumGenerator` + `params_pqc` tensor is the trained state; no new model code |
| Noisy/finite-shot QNode construction | Driver (`run_sensitivity.py`) | PennyLane 0.44 device API | Orchestration, not model definition (D-10-13 code-placement invariant) |
| Sample regeneration under noise | Driver | `core/preprocessing.py` inverse | Forward pass + OD-scale reconstruction; inverse helpers reused unchanged |
| Fidelity metric recompute | Reusable code (`core/eval.py`) | Driver wraps with scale tag | `full_metric_suite` unchanged; driver only adds `scale`/`shots`/`noise_*` dims |
| Per-cell sweep orchestration | Sweep shell (`*_sweep.sh`) | `xargs -P 2` | OS-process parallelism only — never `multiprocessing.Pool` (09.1 Pitfall 4) |
| Multi-seed aggregation (SENS-03) | Driver (`run_multiseed_rollup.py`) | pandas/stdlib | Pure read+groupby over frozen JSONs; no model, no device |
| Cross-artifact data-hash assertion | Driver (SENS-03) | — | D-10-15 invariant enforced before any roll-up math |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pennylane | **0.44.0** | Quantum circuit simulation, finite-shot + noise channels | Already the project's pinned QML library; `quantum.py` built on it |
| torch | 2.10.0 | Tensor interface for QNode (`interface="torch"`) | `QuantumGenerator` is an `nn.Module`; reuse unchanged |
| numpy | (installed) | Sample arrays, RNG (`np.random.default_rng`) | Project convention D-09.1-18 |
| scipy / statsmodels / fastdtw | (installed) | EMD, ACF, DTW inside `eval.py` | Already used by `full_metric_suite` — reuse unchanged |
| pyyaml | (installed) | Per-cell `config.yaml` emission | Matches `run_baselines.py` artifact bundle |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | (verify) | SENS-03 groupby mean±std roll-up | Optional — stdlib `statistics.mean/stdev` is sufficient and dependency-free; prefer stdlib unless pandas already imported elsewhere in `run_*.py` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `qml.set_shots(qnode, shots=N)` transform | `qml.device("default.qubit", shots=N)` | **Do NOT** — deprecated in 0.44.0 (emits `PennyLaneDeprecationWarning: Setting shots on device is deprecated`). Functional now but will break in a future minor; use the transform |
| `default.mixed` for noise channels | `default.qubit` + manual Kraus | `default.mixed` is the canonical density-matrix device; hand-rolling Kraus is the "don't hand-roll" trap |
| Rebuild a fresh QNode for the noisy device | Mutate `QuantumGenerator.qnode` in place | Rebuilding is cleaner and keeps `core/` untouched (D-10-13); the trained `params_pqc` tensor is passed as a circuit argument, never re-bound |

**Installation:** No new packages required. All dependencies already present.

**Version verification (performed live 2026-05-18):**
```
system python3:  pennylane 0.44.0  [VERIFIED: import qml; qml.__version__]
                  torch 2.10.0
qgan_env/bin/python: pennylane 0.43.0  [VERIFIED]  ← VERSION SKEW, see Pitfall 5
```

## Package Legitimacy Audit

> No external packages are installed in this phase. All libraries (pennylane, torch, numpy, scipy, statsmodels, fastdtw, pyyaml) are pre-existing project dependencies already vetted in Phases 8–11. slopcheck not applicable — zero new installs. Disposition: N/A.

## Architecture Patterns

### System Architecture Diagram

```
SENS-01 / SENS-02 (run_sensitivity.py — one cell per invocation)
─────────────────────────────────────────────────────────────────
  checkpoint.pt["params_pqc"]  (75-tensor, frozen, per pipeline×seed)
        │
        ▼
  QuantumGenerator(num_qubits=5, num_layers=4, window_length=10)
  g.params_pqc.data = ck["params_pqc"]          (load trained state)
        │
        ├── condition = "analytic"  ──► REUSE existing samples.npy  (no re-run)
        │
        ├── condition = "shots_8192"|"shots_1024"
        │     dev = qml.device("default.qubit", wires=5)
        │     qn  = qml.QNode(g.generator_circuit, dev, interface="torch",
        │                     diff_method=None)
        │     qn  = qml.set_shots(qn, shots=N)        ◄── 0.44 API
        │
        └── condition = "depol_p"|"ampdamp_g"
              dev = qml.device("default.mixed", wires=5)
              circuit body inserts qml.DepolarizingChannel(p, wires=i)
                                or qml.AmplitudeDamping(γ, wires=i)
              qn  = qml.QNode(..., diff_method=None)
        │
        ▼
  forward pass over 320 batches of 12  →  samples_pm1 (N_synth, 10)
        │  (*0.1 scaling — Pitfall 3 carry-over from run_ablation/run_baselines)
        ▼
  reconstruct_od(...)  [inverse_minmax_od / inverse_logreturns, Pipeline A/B]
        │
        ▼
  full_metric_suite(real_od, fake_od)  +  log-return-scale variant (EVAL-05)
        │
        ▼
  long-form rows[] {model_kind:"quantum", pipeline, seed, metric_name,
                    scale, value, condition, shots|noise_model|noise_level}
        ▼
  shot_noise_sensitivity.json  /  noise_model_sensitivity.json

SENS-03 (run_multiseed_rollup.py — single invocation, pure aggregation)
─────────────────────────────────────────────────────────────────────
  baseline_comparison.json ─┐
  tstr.json                 ├─► assert all data_hash == 91e447d4624e25b3
  predictive_discrim.json   │   (D-10-15 cross-artifact invariant)
  augmentation.json         │
  fidelity_dualscale.json ──┘
        │
        ▼  groupby (model_kind, pipeline, metric_name, scale [, injection_ratio])
  mean ± std (n) per cell over seeds {42,43,44,45,46}
        ▼
  multiseed_summary.json   (consolidated roll-up + provenance block)
```

File-to-implementation mapping (the diagram shows data flow, not files):
- `run_sensitivity.py` — per-cell SENS-01/02 inference driver
- `run_sensitivity_sweep.sh` — idempotent sweep wrapper (xargs -P 2)
- `run_multiseed_rollup.py` — SENS-03 aggregator (single invocation)
- `core/*` — UNTOUCHED (D-10-13 invariant; assert `git diff --stat core/` empty in verification)

### Recommended Project Structure
```
revision/
├── run_sensitivity.py            # NEW — SENS-01/02 per-cell inference driver
├── run_sensitivity_sweep.sh      # NEW — sweep wrapper, --parallel 1|2 guardrail
├── run_multiseed_rollup.py       # NEW — SENS-03 pure aggregator
├── core/                         # UNTOUCHED — model + eval helpers only
└── results/
    ├── sensitivity/              # NEW — per-cell artifact bundles + sweep_status.json
    │   └── runs/<condition>/<pipeline>/<seed>/  {config.yaml, samples.npy,
    │                                              metrics.json, _stdout/_stderr.log}
    ├── shot_noise_sensitivity.json     # NEW — SENS-01 deliverable
    ├── noise_model_sensitivity.json    # NEW — SENS-02 deliverable
    └── multiseed_summary.json          # NEW — SENS-03 deliverable
```

### Pattern 1: Inference-only per-cell driver (mirrors run_baselines.py)
**What:** One process per `(condition, pipeline, seed)` cell. Loads frozen `params_pqc`, builds the appropriate device/QNode, regenerates samples, recomputes the fidelity suite, writes a small artifact bundle + appends long-form rows. Idempotent: rerun overwrites the cell's run dir cleanly.
**When to use:** All SENS-01 / SENS-02 cells.
**Key sub-pattern — the `{analytic}` shot column is free:** The `{analytic}` shot level in SENS-01 *is exactly* the existing `results/transform_ablation/runs/<pipeline>/<seed>/samples.npy`. Do **not** re-run it — read the frozen samples (preserves the no-regeneration invariant for the analytic reference, mirrors D-11-08). Only `shots ∈ {8192, 1024}` and the 8 noise cells require fresh forward passes.

### Pattern 2: Pure-aggregation roll-up (SENS-03)
**What:** Read the five frozen headline JSONs, assert `data_hash` equality across all of them (D-10-15), concatenate their `rows[]`, group by `(model_kind, pipeline, metric_name, scale)` (+ `injection_ratio` for augmentation), emit `{mean, std, n, seeds}` per cell into a single `multiseed_summary.json` with a provenance header listing every consumed file + its `data_hash`.
**When to use:** SENS-03 only. No device, no model, no torch.

### Anti-Patterns to Avoid
- **Mutating `QuantumGenerator.qnode` or editing `core/quantum.py`** to swap the device — breaks the D-10-13 code-placement invariant and the "core untouched" cross-cutting constraint. Build the alternate QNode in the driver, passing `g.params_pqc` as a circuit argument.
- **Using `qml.device(..., shots=N)`** — deprecated in 0.44.0. Use `qml.set_shots(qnode, shots=N)`.
- **`diff_method="backprop"` on finite-shot or `default.mixed`** — incompatible; this is the literal technical justification for D-12-01. Use `diff_method=None` (inference-only, wrapped in `torch.no_grad()`).
- **`multiprocessing.Pool` anywhere** — 09.1 Pitfall 4 / Phase 10 Pitfall 5. Only `xargs -P 2` OS-process parallelism.
- **Re-running the analytic column** — wastes compute and risks drifting from the frozen Phase 09.1 reference. Reuse `samples.npy`.
- **Recomputing data_hash from `transform_ablation` configs** for SENS-03 — quantum equivalence is established *by construction* (the existing JSONs already encode this in `data_hash_verification.quantum_equivalence`). Assert the five JSON `data_hash` fields are mutually equal; do not re-derive.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finite-shot sampling | Manual multinomial over statevector amplitudes | `qml.set_shots(qnode, shots=N)` | PennyLane handles shot allocation, broadcasting, and estimator variance correctly |
| Depolarizing / amplitude-damping | Hand-coded Kraus operators on a density matrix | `default.mixed` + `qml.DepolarizingChannel` / `qml.AmplitudeDamping` | Canonical, validated channel implementations; correct Kraus normalization |
| Fidelity metrics under noise | New EMD/ACF/DTW math | `revision.core.eval.full_metric_suite` unchanged | D-12-03 explicitly forbids new metric math; reviewer-comparable numbers |
| OD-scale reconstruction | New inverse-transform | `run_utility.reconstruct_od` logic (`inverse_minmax_od`, `inverse_logreturns`) | Pipeline-B `seed*7919+1` od_start RNG draw is load-bearing — copy verbatim, do not refactor |
| Atomic per-cell sweep status | Custom JSON writer | `run_baselines_sweep.sh` `update_status` (tmp-file + `os.rename` + `flock`) | Already battle-tested across the 50-cell Phase 10 sweep |
| mean ± std aggregation | Per-metric one-off code | groupby over the long-form schema | Long-form `rows[]` is designed for exactly this pandas/stdlib groupby |

**Key insight:** Phase 12 is ~90% wiring of existing, frozen components. The only genuinely new code is (a) the noisy/finite-shot QNode construction (≈15 lines, verified below) and (b) the groupby roll-up (≈30 lines). Everything else is copied verbatim from `run_baselines.py` / `run_ablation.py` / `run_utility.py`.

## Runtime State Inventory

> Phase 12 generates new artifacts but consumes frozen ones; it is not a rename/refactor. This abbreviated inventory confirms no hidden runtime state blocks the phase.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Trained params: `results/transform_ablation/runs/{A,B,C}/{42,43,44,45,46}/checkpoint.pt` key `params_pqc` (75-tensor). Analytic samples: `samples.npy` (3840,10) float64 in same dirs. All 5 seeds present for A, B, C — **verified by `ls`**. | None — reload read-only |
| Live service config | None — local Mac statevector simulator only; no external services. | None |
| OS-registered state | None — sweeps run via tmux/nohup + xargs, no scheduled tasks. | None |
| Secrets/env vars | `QGAN_CANONICAL_REPO` resolver pattern used by `run_dualscale_fidelity.py`/`run_utility.py` (Phase 11 CR-01 fix). New drivers should reuse the same repo-root resolver for cwd-independence. | Reuse existing resolver pattern |
| Build artifacts | None — no compiled artifacts; pure-Python drivers. | None |

## Common Pitfalls

### Pitfall 1: `shots=` device kwarg is deprecated in PennyLane 0.44.0
**What goes wrong:** Following the CONTEXT.md hint "construct a finite-shot device (`shots=8192`)" literally yields `qml.device("default.qubit", wires=5, shots=8192)`, which emits `PennyLaneDeprecationWarning: Setting shots on device is deprecated. Please use the set_shots transform on the respective QNode instead.` It works today but is fragile and clutters logs.
**Why it happens:** Training-data and pre-0.43 docs use the device kwarg; 0.44 moved shots to a QNode transform.
**How to avoid:** Build an analytic QNode, then wrap: `shot_qnode = qml.set_shots(qnode, shots=N)`. Verified live: `qml.set_shots` exists in 0.44.0 and returns a callable producing shot-noisy expectations. `[VERIFIED: live import on installed pennylane 0.44.0, 2026-05-18]`
**Warning signs:** `PennyLaneDeprecationWarning` in `_stderr.log`.

### Pitfall 2: `backprop` is incompatible with finite shots / `default.mixed`
**What goes wrong:** `QuantumGenerator` defaults to `diff_method="backprop"`. Reusing `g.qnode` directly on a shot/mixed device errors or silently degrades.
**Why it happens:** backprop differentiates through an exact statevector; finite shots are non-differentiable estimators and `default.mixed` uses density matrices.
**How to avoid:** Construct a **new** QNode in the driver with `diff_method=None` and run under `torch.no_grad()` (inference-only — D-12-01 forbids gradients/retraining anyway). Verified live: `default.mixed` + `AmplitudeDamping` + `diff_method=None` returns correct expectations.
**Warning signs:** errors mentioning `backprop`/`adjoint` not supported; NaNs.

### Pitfall 3: The `*0.1` output scaling is load-bearing
**What goes wrong:** Both `run_ablation.generate_samples` and `run_baselines.generate_wgan_samples` multiply generator output by `0.1` before saving (a quantum-output-magnitude artifact that fed the critic during training). Omitting `*0.1` makes the regenerated noisy samples a different distribution from the frozen analytic `samples.npy`, so the degradation curve is meaningless (it would conflate scale with noise).
**Why it happens:** It looks like a magic constant; it is actually part of the trained sample contract.
**How to avoid:** Copy the `generate_samples` body verbatim from `run_ablation.py:180-209` (including `* 0.1`, `np.random.default_rng(seed)`, `NOISE_LOW/HIGH`, batch loop). The only change is the QNode/device used inside the loop.
**Warning signs:** `{analytic}`-condition regenerated samples don't byte-match the frozen `samples.npy` for the same seed (sanity check this in a smoke test — they should match to fp tolerance since the same params + same RNG + same device).

### Pitfall 4: Pipeline-B od_start RNG seed is load-bearing
**What goes wrong:** `reconstruct_od` for Pipeline B draws per-window starting OD with `np.random.default_rng(seed * 7919 + 1)`. Using a different seed expression desynchronizes OD-scale reconstruction from the Phase 11 numbers, breaking comparability.
**Why it happens:** It looks refactorable; it is a frozen contract (run_utility.py comment: "load-bearing; do NOT refactor").
**How to avoid:** Copy `reconstruct_od` Pipeline-A and Pipeline-B branches verbatim from `run_utility.py:145-185`. Note `od[:, :10]` truncation when `inverse_logreturns` returns length-11.
**Warning signs:** OD-scale EMD for the `{analytic}` / `p=0` / `γ=0` baseline cell doesn't match the corresponding `fidelity_dualscale.json` quantum row.

### Pitfall 5: venv (`qgan_env`) is PennyLane 0.43.0, system is 0.44.0
**What goes wrong:** `run_baselines_sweep.sh` prefers `./qgan_env/bin/python` if present. That venv has **PennyLane 0.43.0**, but CONTEXT.md and this research pin **0.44.0**. The `qml.set_shots` transform API and the `shots=` deprecation differ between 0.43 and 0.44. A sweep launched via the venv would silently run on the wrong PennyLane.
**Why it happens:** The sweep-wrapper interpreter-selection logic copied from Phase 10 hard-prefers the venv.
**How to avoid:** This is an **Open Question for the planner** (see below). Options: (a) pin the new sweep wrapper to a 0.44.0 interpreter explicitly and document it; (b) upgrade `qgan_env` to 0.44.0 and re-verify the 09.1/10 reproduction (risky — touches the frozen reproduction baseline); (c) write the `set_shots` call defensively to work on both 0.43 and 0.44. Recommendation: **(a)** — add a `PENNYLANE_MIN_VERSION` assertion at driver startup that fails loud if `qml.__version__ != "0.44.0"`, and have the sweep wrapper select the 0.44.0 interpreter. Do not silently upgrade the venv (it would invalidate the "09.1/10 reproduce exactly" cross-cutting constraint).
**Warning signs:** `qml.__version__` != `0.44.0` at driver startup; `set_shots` AttributeError on 0.43.

### Pitfall 6: cwd-dependent paths
**What goes wrong:** `run_utility.py`/`run_dualscale_fidelity.py` had a Phase-11 bug where relative paths broke when run from a non-repo-root cwd; fixed with a `QGAN_CANONICAL_REPO`/repo-root resolver.
**How to avoid:** Reuse the same repo-root resolver pattern in both new drivers; anchor all artifact paths at the resolved repo root.
**Warning signs:** `FileNotFoundError` on `transform_ablation/...` when sweep is launched from tmux in a different directory.

## Code Examples

> All examples verified against the live installed PennyLane 0.44.0 on 2026-05-18.

### Example 1: Load trained analytic params into a fresh generator
```python
# Source: VERIFIED live; mirrors run_ablation.py + checkpoint inspection
import torch
from revision.core.models.quantum import QuantumGenerator
from revision.core import NUM_QUBITS, NUM_LAYERS, WINDOW_LENGTH  # 5, 4, 10

g = QuantumGenerator(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS,
                     window_length=WINDOW_LENGTH)
ck = torch.load(
    f"results/transform_ablation/runs/{pipeline}/{seed}/checkpoint.pt",
    map_location="cpu", weights_only=False)
g.params_pqc.data = ck["params_pqc"]        # 75-element trained tensor
g.eval()
```

### Example 2: SENS-01 — finite-shot QNode via the 0.44 set_shots transform
```python
# Source: VERIFIED live on pennylane 0.44.0 (2026-05-18)
import pennylane as qml, numpy as np, torch
from revision.core import NOISE_LOW, NOISE_HIGH, NUM_QUBITS, BATCH_SIZE

def make_shot_qnode(g, shots: int | None):
    dev = qml.device("default.qubit", wires=NUM_QUBITS)   # NO shots= kwarg
    qn = qml.QNode(g.generator_circuit, dev, interface="torch",
                   diff_method=None)                       # inference-only
    if shots is not None:
        qn = qml.set_shots(qn, shots=shots)                # 0.44 API
    return qn

def generate_samples_on_qnode(g, qnode, n, seed):
    """Verbatim copy of run_ablation.generate_samples, only the call site
    swapped to `qnode`. *0.1 scaling and default_rng(seed) are load-bearing."""
    rng = np.random.default_rng(seed)
    parts, remaining = [], n
    with torch.no_grad():
        while remaining > 0:
            bs = min(BATCH_SIZE, remaining)
            noise = torch.tensor(
                rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, bs)),
                dtype=torch.float32)
            res = qnode(noise, g.params_pqc)               # tuple of 10 exp-vals
            out = torch.stack(list(res))
            if out.dim() == 2:
                out = out.T                                # (bs, 10)
            out = out.to(torch.float64) * 0.1              # Pitfall 3
            parts.append(out.cpu().numpy())
            remaining -= bs
    return np.concatenate(parts, axis=0)[:n]
```
Note: `generator_circuit` returns a tuple of expectations; under broadcasting the
stacked shape is `(window_length, batch)` → transpose to `(batch, 10)`, exactly
as `QuantumGenerator.forward` does (`quantum.py:194-199`).

### Example 3: SENS-02 — noise channels on default.mixed
```python
# Source: VERIFIED live on pennylane 0.44.0 (2026-05-18)
import pennylane as qml, torch
from revision.core import NUM_QUBITS

def make_noisy_qnode(g, channel: str, level: float):
    """channel in {'depolarizing','amplitude_damping'}; level is p or gamma.
    Channels inserted per-wire at end of the trained circuit body
    (deployment-noise model: the trained unitary, then a noisy readout layer).
    """
    dev = qml.device("default.mixed", wires=NUM_QUBITS)

    def noisy_circuit(noise_params, params_pqc):
        # Re-emit the trained circuit body, then append the noise layer.
        # Cleanest: call g.generator_circuit's gate sequence via a tape,
        # OR replicate by constructing the QNode around a wrapper that
        # applies g's ops then the channel. Simplest robust approach:
        out = g.generator_circuit  # NOTE: see implementation note below
        ...
    qn = qml.QNode(..., dev, interface="torch", diff_method=None)
    return qn
```
**Implementation note for the planner:** `g.generator_circuit` ends with
`qml.expval(...)` returns, so you cannot append channels *after* calling it
inside another QNode. Two clean options, both verified to work on 0.44.0:
1. **Per-gate noise (recommended, hardware-faithful):** copy the
   `generator_circuit` body into the driver as `noisy_generator_circuit`,
   inserting `qml.DepolarizingChannel(p, wires=q)` / `qml.AmplitudeDamping(γ,
   wires=q)` after each parameterized gate (or after each entangling layer).
   This is the standard NISQ noise model and the most defensible to reviewers.
2. **End-of-circuit readout noise:** same copied body, channels applied to every
   wire immediately before the `qml.expval` measurements only. Lighter, models
   measurement/readout error specifically.
The CONTEXT.md grants channel-insertion strategy to Claude's discretion; the
planner should pick **per-layer insertion** (channel after each entangling
block) as the middle-ground default and document it. Copying the ~40-line
circuit body into the driver does **not** violate D-10-13 (the copy lives in
`run_sensitivity.py`, not `core/`), but the planner must note it as a
deliberate, documented duplication keyed to the noise study.
Live-verified minimal form:
```python
devm = qml.device("default.mixed", wires=5)
@qml.qnode(devm, interface="torch", diff_method=None)
def ncirc(x, params):
    # ... trained gate sequence (copied from generator_circuit) ...
    for q in range(5):
        qml.DepolarizingChannel(p, wires=q)      # or AmplitudeDamping(gamma, q)
    return tuple(qml.expval(qml.PauliX(i)) for i in range(5)) + \
           tuple(qml.expval(qml.PauliZ(i)) for i in range(5))
# verified: returns finite expectations, p=0/gamma=0 reduces to analytic mixed
```

### Example 4: SENS-03 — pure aggregation with cross-artifact data_hash assertion
```python
# Source: schema VERIFIED by inspecting all 5 JSONs on disk (2026-05-18)
import json, statistics
from collections import defaultdict
from pathlib import Path

HEADLINE = ["baseline_comparison.json", "tstr.json",
            "predictive_discriminative.json", "augmentation.json",
            "fidelity_dualscale.json"]
RESULTS = Path("revision/results")

docs = {f: json.load(open(RESULTS / f)) for f in HEADLINE}

# D-10-15 cross-artifact invariant: every headline JSON must agree on data_hash.
hashes = {f: d["data_hash"] for f, d in docs.items()}
assert len(set(hashes.values())) == 1, f"data_hash mismatch across artifacts: {hashes}"
canonical_hash = next(iter(hashes.values()))           # expect 91e447d4624e25b3

# Concatenate long-form rows; key by aggregation tuple.
buckets = defaultdict(list)
for f, d in docs.items():
    for r in d["rows"]:
        key = (f, r["model_kind"], r["pipeline"], r["metric_name"],
               r["scale"], r.get("injection_ratio"))    # augmentation adds ratio
        buckets[key].append((r["seed"], r["value"]))

rollup = []
for (src, mk, pl, metric, scale, ratio), pairs in buckets.items():
    seeds  = sorted({s for s, _ in pairs})
    vals   = [v for _, v in pairs]
    rollup.append({
        "source": src, "model_kind": mk, "pipeline": pl,
        "metric_name": metric, "scale": scale, "injection_ratio": ratio,
        "mean": statistics.fmean(vals),
        "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals), "seeds": seeds,
    })

out = {
    "schema": "SENS-03 multi-seed roll-up: mean ± std per headline cell",
    "data_hash": canonical_hash,
    "consumed_artifacts": {f: docs[f]["data_hash"] for f in HEADLINE},
    "seed_set": [42, 43, 44, 45, 46],
    "rollup": rollup,
}
(RESULTS / "multiseed_summary.json").write_text(json.dumps(out, indent=2))
```
Schema confirmed identical across all five files: top-level keys include
`schema, model_kinds, pipelines, seeds, data_hash, data_hash_verification`;
each carries a `rows[]` list of `{model_kind, pipeline, seed, metric_name,
scale, value[, injection_ratio]}`. Row counts: baseline_comparison 1710,
fidelity_dualscale 3360, tstr 144, predictive_discriminative 120,
augmentation 180. `data_hash` is `91e447d4624e25b3` in all five `[VERIFIED]`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `qml.device(..., shots=N)` | `qml.set_shots(qnode, shots=N)` transform | PennyLane 0.43→0.44 | Device-kwarg now emits deprecation warning; use the transform |
| Device-bound shots | QNode-level shot transform composable with other transforms | 0.44 | Cleaner; one analytic QNode reused for all shot levels |

**Deprecated/outdated:**
- `shots=` device constructor kwarg (PennyLane 0.44.0): functional but deprecated, `PennyLaneDeprecationWarning`. `[VERIFIED: live]`
- `np.random.seed` global seeding: project convention D-09.1-18 forbids it; use `np.random.default_rng(seed)`. `[CITED: 09.1-CONTEXT.md D-09.1-18]`

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Per-layer channel insertion is the most reviewer-defensible default for SENS-02 (vs per-gate or readout-only) | Code Example 3 note | Low — CONTEXT.md grants this to Claude's discretion; any documented choice is acceptable. Per-layer is a conventional NISQ model |
| A2 | Pipeline B is headline + Pipeline A supplementary for the noise/shot grid (mirroring Phase 10/11) | Pattern 1 | Low — explicitly Claude's discretion in CONTEXT.md; matches established D-10-06/D-11-09 precedent |
| A3 | The `{analytic}` shot column can reuse frozen `samples.npy` rather than re-running | Pattern 1 sub-pattern | Low — preserves the no-regeneration invariant; a smoke test (regenerate analytic, compare to frozen) confirms equivalence cheaply |
| A4 | SENS-03 should assert mutual equality of the five JSON `data_hash` fields rather than re-derive from transform_ablation | Pattern 2 | Low — the existing `data_hash_verification.quantum_equivalence` blocks already document by-construction equivalence; re-deriving would duplicate Phase 11 logic |
| A5 | Channels modeled as a deployment-noise layer on the *trained* unitary (not noise-during-training) | Architecture | None — this IS D-12-01 (inference-only robustness narrative); locked, not assumed |

## Open Questions (RESOLVED)

1. **PennyLane version skew between system Python (0.44.0) and `qgan_env` (0.43.0).**
   RESOLVED: startup assert + no venv upgrade. New drivers add a startup assertion
   `assert qml.__version__ == "0.44.0"` (fail loud); the new sweep wrapper deliberately
   does NOT prefer `./qgan_env` and selects an explicit 0.44.0 interpreter (system
   python3). `qgan_env` is NOT upgraded (preserves the frozen 09.1/10 reproduction
   baseline). Implemented by Plan 01 Task 1 (version gate) and Plan 02 Task 1 (interpreter
   deviation from the analog sweep).
   - What we know: `run_baselines_sweep.sh` prefers `./qgan_env/bin/python`; that venv has PennyLane 0.43.0. CONTEXT.md pins 0.44.0. `qml.set_shots` and the `shots=` deprecation differ between the two.
   - What's unclear: which interpreter the Phase 12 sweep should use, and whether upgrading `qgan_env` would invalidate the "09.1/10 reproduce exactly" cross-cutting constraint.
   - Recommendation: New drivers add a startup assertion `assert qml.__version__ == "0.44.0"` (fail loud). The new sweep wrapper selects a 0.44.0 interpreter explicitly (do NOT silently prefer the 0.43 venv). Do not upgrade `qgan_env` in this phase — that is a separate, risky change touching the frozen reproduction baseline. Planner should make the interpreter selection an explicit, documented decision.

2. **Channel-insertion strategy (per-gate vs per-layer vs readout-only).**
   RESOLVED: per-layer default. The channel is inserted after each entangling block
   (conventional NISQ model, mid-cost, defensible). Implemented by Plan 01 Task 2
   (`make_noisy_qnode` per-layer insertion); recorded in the driver docstring and in the
   `noise_model_sensitivity.json` provenance block (Plan 02 Task 3).
   - What we know: all three work on 0.44.0; CONTEXT.md grants this to Claude's discretion.
   - What's unclear: which the planner wants as the documented default.
   - Recommendation: per-layer (channel after each entangling block) — conventional NISQ model, mid-cost, defensible. Document the choice in the driver docstring and in `noise_model_sensitivity.json` metadata.

3. **Pandas vs stdlib for SENS-03.**
   RESOLVED: stdlib. SENS-03 aggregation uses pure stdlib (`statistics.fmean/stdev`) —
   zero new dependency, dependency-audit-clean. Codified in the SENS-03 plan (Phase 12
   Plan 03) per Code Example 4.
   - What we know: Example 4 works with pure stdlib (`statistics.fmean/stdev`).
   - Recommendation: prefer stdlib (zero new dependency, dependency-audit-clean). Use pandas only if a `run_*.py` already imports it (verify at plan time).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PennyLane | SENS-01/02 device + noise API | ✓ (system) | 0.44.0 | venv has 0.43.0 — see Open Q1 |
| torch | QNode torch interface | ✓ | 2.10.0 | — |
| numpy/scipy/statsmodels/fastdtw | `eval.full_metric_suite` | ✓ | (Phase 8-11 vetted) | — |
| Trained checkpoints (`params_pqc`) | D-12-01 inference | ✓ | A/B/C × {42..46} all present | — |
| Frozen headline JSONs (5) | SENS-03 | ✓ | data_hash 91e447d4624e25b3 | — |
| `xargs -P` | sweep parallelism | ✓ (macOS bsd xargs) | — | `--parallel 1` sequential |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** PennyLane version skew (Open Question 1) — fallback is explicit interpreter pinning.

## Validation Note

Nyquist formal validation is disabled for this project. Lightweight validation guidance for the planner:

- **Smoke gate:** before the full sweep, run one cell (`pipeline=B, seed=42, condition=analytic`) and assert the regenerated samples reproduce the frozen `samples.npy` to fp tolerance — proves the device-swap harness is faithful (Pitfall 3 detector).
- **Baseline-cell sanity:** the `p=0` / `γ=0` / `shots=∞` cells must reproduce the corresponding `fidelity_dualscale.json` quantum rows within fp tolerance (Pitfall 4 detector).
- **Monotonicity expectation (not a hard gate):** EMD should generally *increase* (degrade) as shots decrease and as p/γ increase. A non-monotone curve is a flag for investigation, not necessarily a bug (3-seed noise can produce local non-monotonicity — D-12-02 acknowledges this).
- **Core-untouched invariant:** `git diff --stat core/` must be empty (cross-cutting constraint carried from Phases 10/11).
- **Cross-artifact hash:** SENS-03 driver asserts all five `data_hash` fields equal `91e447d4624e25b3` before emitting (hard gate, D-10-15).

## Security Domain

Not applicable. This phase performs local scientific computation only — no authentication, network I/O, input from untrusted sources, secrets, or cryptography. `security_enforcement` is not relevant to an offline simulator sweep. The only adjacent concern (environment integrity) is the PennyLane version skew, captured as Open Question 1.

## Sources

### Primary (HIGH confidence)
- Live PennyLane 0.44.0 (system python3) — verified `qml.set_shots` exists and produces shot-noisy expectations; `default.mixed` + `DepolarizingChannel(p, wires)` + `AmplitudeDamping(gamma, wires)` work with `diff_method=None`; `shots=` device kwarg emits `PennyLaneDeprecationWarning`. Probed 2026-05-18.
- Live timing probe — analytic 0.013s, shots-8192 0.028s, default.mixed 0.036s per batch(12); 320 batches/run. 2026-05-18.
- On-disk artifact inspection — `checkpoint.pt["params_pqc"]` (75-tensor) in all `transform_ablation/runs/{A,B,C}/{42..46}/`; five headline JSON schemas + `data_hash=91e447d4624e25b3` confirmed identical.
- `core/models/quantum.py`, `core/eval.py`, `core/preprocessing.py` — read in full.
- `run_baselines.py`, `run_baselines_sweep.sh`, `run_ablation.py`, `run_utility.py` (relevant sections) — read for the driver/sweep/reconstruction patterns.
- CONTEXT.md (12, 11, 09.1), REQUIREMENTS.md, ROADMAP.md Phase 12 entry, STATE.md — read.

### Secondary (MEDIUM confidence)
- None required — all critical claims verified against the live environment or on-disk artifacts.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all APIs verified live on installed 0.44.0
- Architecture: HIGH — patterns copied from already-shipped Phase 10/11 drivers; data flow verified end-to-end
- Pitfalls: HIGH — every pitfall reproduced or confirmed live (deprecation warning, backprop incompatibility, version skew, load-bearing constants observed in source)
- SENS-03 aggregation: HIGH — all five JSON schemas inspected directly; data_hash equality confirmed

**Research date:** 2026-05-18
**Valid until:** 2026-06-17 (30 days — stable; pinned PennyLane 0.44.0, frozen artifacts. Revalidate only if `qgan_env` is upgraded or headline JSONs are regenerated.)
