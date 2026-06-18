# Phase 13: Architecture & Introspection - Research

**Researched:** 2026-05-18
**Domain:** Quantum-circuit ansatz comparison + PQC training introspection (PennyLane statevector, WGAN-GP, reproducibility-JSON driver pattern)
**Confidence:** HIGH (the two highest-risk items — the entanglement-entropy/purity API and the callback instrumentation surface — were verified by executing live probes against the installed stack, not from training memory)

## Summary

Phase 13 is mostly an *integration + instrumentation* phase, not a new-technology phase. Every hard dependency already exists in the codebase: the dormant `callback(epoch, metrics)` hook in `train_wgan_gp` (`training.py:396-411`), the `QuantumGenerator.generator_circuit` QNode (`quantum.py:103-171`), the `full_metric_suite` evaluator (`eval.py:143`), the long-form `rows[]` schema (`baseline_comparison.json`, 1710 rows verified), and the idempotent `run_baselines.py` + `run_baselines_sweep.sh` driver/sweep template. The research effort therefore concentrated on the four open questions the CONTEXT.md explicitly deferred: the PennyLane entanglement-entropy/purity API, the V3 topology choice, the callback instrumentation gap, and the CR-01/CR-02 fix shapes.

**Critical correction to a CONTEXT.md assumption:** the canonical-refs section names "PennyLane 0.44.0", but the **installed and operative version is 0.43.0** [VERIFIED: `pennylane.__version__` == "0.43.0" in `./qgan_env`]. All API patterns below are verified against 0.43.0. The 0.43 measurement-process API (`qml.vn_entropy(wires=...)`, `qml.purity(wires=...)`, `qml.density_matrix(wires=...)`) works inside a QNode and **can coexist in the same return tuple as `qml.expval(...)`** — verified by execution. The offline `qml.math.vn_entropy(state, indices, ...)` helper has a *different, positional* signature in 0.43 (`indices` is positional, not a keyword, and on a statevector it requires the reduced-DM path), so the in-QNode measurement-process route is strictly simpler and is the recommended pattern.

**Primary recommendation:** Add a depth+topology spec to `QuantumGenerator` (keeping the range-based depth-4 default byte-identical), build a single new `run_ansatz.py` + `run_ansatz_sweep.sh` pair cloned structurally from `run_baselines.py`/`run_baselines_sweep.sh`, drive INTRO-* through a closure passed as the existing `callback=` kwarg that snapshots samples / param-stats / a dedicated read-only introspection QNode, fix CR-01 with a real Welch-free differentiable torch PSD term and CR-02 with `map_location` + dtype recast, and emit `ansatz_comparison.json` on the existing `rows[] + models[]` schema extended with `ansatz`/`depth`/`topology` fields.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Ansatz variant definition (V1/V2/V3) | `core/models/quantum.py` (model layer) | — | D-10-13 code-placement invariant: model defs live in `core/`, never in drivers |
| Ansatz selection API | `core/models/quantum.py` (model layer) | driver passes config | D-13-03; default must stay byte-unchanged |
| Multi-seed ansatz training sweep | `run_ansatz.py` + `*_sweep.sh` (orchestration) | `train_wgan_gp` UNCHANGED | D-10-13: all sweep orchestration in `run_*.py` |
| INTRO snapshot capture | `callback` closure in driver (orchestration) | `train_wgan_gp` hook (already built) | Hook is dormant-by-design; no training-loop surgery |
| Entanglement entropy / purity | dedicated read-only QNode on generator's `default.qubit` device (model layer helper) | callback invokes it | Statevector measurement belongs with the circuit, not the driver |
| Fidelity metrics for the table | `core/eval.py::full_metric_suite` UNCHANGED | aggregation notebook/script | D-10-20: no new metric math; reuse |
| Figure rendering | new `run_*.py` / a figures script (orchestration) | matplotlib | Notebook/script only orchestrates + plots + writes JSON |
| CR-01 / CR-02 fixes | `core/training.py` (training layer) | regression tests in `tests/` | Folded todos target this file specifically |

**Why this matters:** the single most likely misassignment here is putting the entanglement-entropy computation in the driver (operating on returned expvals) instead of in a QNode that has access to the statevector. Reduced-state entropy/purity is *not* recoverable from the 10 PauliX/PauliZ expectation values the generator returns — it requires either an in-QNode `qml.vn_entropy`/`qml.purity` measurement or a `qml.state()`/`qml.density_matrix()` return reduced offline. This must be a model-layer helper.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pennylane | **0.43.0** [VERIFIED: `pennylane.__version__` in `./qgan_env`] | Statevector sim + `vn_entropy`/`purity`/`density_matrix`/`state` measurement processes | Already the project's quantum backend; INTRO-03 measures are native built-ins |
| torch | 2.9.0 [VERIFIED: `torch.__version__`] | Autograd, `nn.Parameter`, optimizers, `torch.fft` for CR-01 differentiable PSD | Already the project ML backend; `diff_method="backprop"` requires the torch interface |
| numpy | 2.3.4 [VERIFIED] | Array glue, RNG (`np.random.default_rng(seed)`), JSON-safe casts | Established project dependency |
| scipy | 1.16.2 [VERIFIED] | `scipy.signal.welch` (current CR-01 implementation — to be *replaced* by torch.fft) | Already imported; CR-01 fix removes the welch round-trip from the grad path |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | (project-installed) | Training-progression / param-trajectory / entanglement-trajectory figures | Figure rendering only (Claude's discretion per D-13 discretion list) |
| pyyaml | (project-installed, used by `run_baselines.py`) | `config.yaml` per run-dir | Per-run frozen-config artifact (D-10-14 bundle) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| In-QNode `qml.vn_entropy(wires=[...])` | `qml.state()` return + offline `qml.math.reduce_statevector` → `qml.math.purity`/`vn_entropy` | Offline path works but the 0.43 `qml.math.*` signature is positional and brittle (`indices` positional; statevector must be reduced to a DM first — verified: `qml.math.vn_entropy(sv, indices=[0,1])` raises `ValueError: not enough values to unpack`). The in-QNode measurement is one line and verified to coexist with expvals. **Use in-QNode.** |
| New `run_ansatz.py` driver | Extend `run_baselines.py` with a `--quantum-ansatz` mode | Phase 13 trains quantum variants with epoch=1000/early-stop-off and an instrumented seed; bolting that onto the classical-baseline driver violates single-responsibility and the idempotent-bundle contract. **Use a new driver cloned from the template.** |
| `multiprocessing.Pool` for the sweep | `xargs -P 2` (OS processes) | LOCKED by D-10-24 / Phase 09.1 Pitfall 4 — fork-shared numpy RNG corrupts reproducibility. **Never `Pool`.** |

**Installation:** No new packages. All dependencies are already in `./qgan_env`. Verified:
```bash
./qgan_env/bin/python -c "import pennylane,torch,numpy,scipy; print(pennylane.__version__,torch.__version__)"
# -> 0.43.0 2.9.0
```

## Package Legitimacy Audit

> No external packages are installed by this phase — every dependency is already resident in `./qgan_env` and was used by Phases 8–12. Package legitimacy gate is **N/A** (no install step). slopcheck not run because there is nothing to install.

| Package | Registry | Disposition |
|---------|----------|-------------|
| pennylane / torch / numpy / scipy / matplotlib / pyyaml | (already installed, used by prior phases) | No install — N/A |

**Packages removed due to slopcheck [SLOP] verdict:** none (no installs).
**Packages flagged as suspicious [SUS]:** none (no installs).

## Architecture Patterns

### System Architecture Diagram

```
                       ┌─────────────────────────────────────────────────┐
                       │ core/models/quantum.py                  │
   ansatz spec ───────►│  QuantumGenerator(depth, topology)               │
 (depth, topology)     │   ├─ generator_circuit  (measurement QNode)      │
                       │   └─ introspection_qnode (vn_entropy + purity)   │◄─┐
                       └───────────────┬─────────────────────────────────┘  │
                                       │ generator instance                  │
                                       ▼                                     │
   ANSATZ COMPARISON PATH        ┌──────────────────────┐                    │
   (ARCH-01/02)                  │ train_wgan_gp(...)    │                    │
   run_ansatz.py ───(V2,V3 ×5)──►│  UNCHANGED happy path │                    │
   one (variant,seed)/proc       │  callback=None        │                    │
        │                        └──────────┬───────────┘                    │
        │ samples.npy + metrics.json         │ per-run 5-file bundle          │
        ▼                                    ▼  runs/<variant>/<seed>/        │
   eval.full_metric_suite ──► ansatz_comparison.json (rows[] + models[])      │
   (V1 reuses 09.1/10 final metrics — NO recompute, D-13-01)                  │
                                                                              │
   INTROSPECTION PATH           ┌──────────────────────┐                      │
   (INTRO-01/02/03)             │ train_wgan_gp(...)    │   callback fires on  │
   run_introspect.py ──(V1,s42)►│  callback=snapshot_cb │──►eval epochs ───────┘
   one instrumented run         └──────────────────────┘    {0,250,500,750,1000}
        │                                │
        │   snapshot_cb captures per snapshot-epoch:
        │     • generated-distribution samples  (INTRO-01)
        │     • params_pqc norm + angle histogram (INTRO-02)
        │     • introspection_qnode → vn_entropy + purity (INTRO-03)
        ▼
   training_progression.* + param_trajectory.* + entanglement_trajectory.*
   + companion *.json   (also: 3 classical WGAN variants, fresh instrumented,
                          Pipeline B seed 42 — INTRO-01 side-by-side panel)
```

### Recommended Project Structure

```
revision/
├── core/
│   ├── models/quantum.py       # + ansatz spec (depth/topology); + introspection_qnode helper
│   └── training.py             # + CR-01 fix (differentiable PSD); + CR-02 fix (map_location/dtype)
├── run_ansatz.py               # NEW — clone of run_baselines.py; one (variant,seed)/proc
├── run_ansatz_sweep.sh         # NEW — clone of run_baselines_sweep.sh; xargs -P 2
├── run_introspect.py           # NEW — single instrumented run + classical instrumented runs
├── run_introspect_figures.py   # NEW (or fold into run_introspect) — renders 4 figures + JSON
└── results/
    ├── ansatz_comparison.json          # rows[] + models[] + ansatz/depth/topology dims
    └── figures/
        ├── training_progression.{png,pdf}        + training_progression.json
        ├── param_trajectory.{png,pdf}            + param_trajectory.json
        └── entanglement_trajectory.{png,pdf}     + entanglement_trajectory.json
tests/
├── test_cr01_spectral_grad.py  # asserts non-zero grad into params_pqc when weight>0
└── test_cr02_es_restore.py     # asserts device/dtype consistency after restore (CPU + MPS)
```

### Pattern 1: Config-selectable ansatz, byte-unchanged default

**What:** Add an optional `topology` argument (and rely on the existing `num_layers` for depth) to `QuantumGenerator.__init__`. The entangling block branches on topology; the default value reproduces the existing range-based wrap-around CNOT pattern with identical gate order.

**When to use:** All three variants are constructed through this one API. V1 = `num_layers=4, topology="range"` (the existing default — produces 75 params and a byte-identical circuit). V2 = `num_layers=8, topology="range"` (135 params). V3 = `num_layers=4, topology="linear"` (75 params).

**Param-count formula** (verified against `count_params()`): `num_qubits + num_layers*(num_qubits*3) + num_qubits*2`. For `num_qubits=5`: V1/V3 = `5 + 4*15 + 10 = 75`; V2 = `5 + 8*15 + 10 = 135`. Topology does **not** change param count (CNOTs are non-parametric) — V1 vs V3 isolates topology at fixed 75 params, exactly as D-13-01 intends.

```python
# Source: pattern derived from quantum.py:137-156 (range block) — VERIFIED param math
# In generator_circuit Step 4, replace the fixed range block with a topology switch.
# Default MUST emit the identical gate sequence as today (byte-unchanged for Phases 8-12).
if self.num_qubits > 1:
    if self.topology == "range":                       # DEFAULT — existing behavior
        range_param = (layer % (self.num_qubits - 1)) + 1
        for qubit in range(self.num_qubits):
            target = (qubit + range_param) % self.num_qubits
            qml.CNOT(wires=[qubit, target])
    elif self.topology == "linear":                    # V3 — open nearest-neighbour chain
        for qubit in range(self.num_qubits - 1):       # i -> i+1, no wrap-around
            qml.CNOT(wires=[qubit, qubit + 1])
```

**Anti-pattern guard:** do NOT reorder the existing branch or change `idx` accounting — the Rot/RX/RY parameter consumption is identical across topologies (only the CNOT wiring differs), so `count_params()` and `params_pqc` indexing stay untouched. Add a regression test that `QuantumGenerator()` (no args) produces a circuit whose drawn tape equals the pre-change tape (or, minimally, that `count_params()==75` and a fixed-seed forward pass matches a saved reference vector).

### Pattern 2: INTRO instrumentation via the existing callback (no training-loop edits)

**What:** The hook already exists and fires correctly. `training.py:396-411` calls `callback(epoch, metrics_dict)` on every eval epoch (`epoch % eval_every == 0 or epoch+1 == num_epochs`), inside a `try/except` so a callback bug cannot kill training. The closure captures the `generator` object, so it can call any generator method to snapshot state.

**The gap (answering open question 3):** the hook passes only a *scalar metrics dict* (epoch, emd, losses, mean/std/kurtosis) — it does **not** pass the generator or samples. So the closure must close over the `generator` reference and *re-generate* a fresh evaluation batch + read `generator.params_pqc` + call a new introspection QNode itself. This is the only "gap": the hook is sufficient, but the snapshot logic lives entirely in the closure, not in the metrics dict. Snapshot epochs for N=1000, eval_every=10: `{0, 250, 500, 750, 1000}` are all eval epochs (0, 250, 500, 750 satisfy `epoch%10==0`; epoch index 999 satisfies `epoch+1==num_epochs` → record as "1000"). Use `eval_every=10` (the project default) so all five land.

```python
# Driver-side closure passed as callback= to train_wgan_gp. Read-only; no grad needed.
SNAP = {0, 250, 500, 750, 999}     # 999 == final epoch index for num_epochs=1000
snapshots = []
def snapshot_cb(epoch, metrics):
    label = 1000 if epoch == 999 else epoch
    if epoch not in SNAP:
        return
    with torch.no_grad():
        # INTRO-01: generated-distribution samples (same noise contract as train loop)
        noise = torch.tensor(np.random.default_rng(seed).uniform(
            NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, BATCH_SIZE)), dtype=torch.float32)
        gen = (generator(noise).to(torch.float64) * 0.1).cpu().numpy()
        # INTRO-02: param norms + angle histogram
        p = generator.params_pqc.detach().cpu().numpy()
        # INTRO-03: vn_entropy + purity from a dedicated read-only introspection QNode
        ent, pur = generator.introspect(noise[:, 0])     # single representative state
    snapshots.append({"epoch": label, "samples": gen.tolist(),
                       "param_norm": float(np.linalg.norm(p)),
                       "param_angles": p.tolist(),
                       "vn_entropy": float(ent), "purity": float(pur),
                       "emd": metrics["emd"], "std": metrics["std"]})
```

### Pattern 3: Entanglement entropy + purity — verified PennyLane 0.43 API

**What:** Add an `introspect(noise)` method to `QuantumGenerator` backed by a *second* QNode on the same `default.qubit` device that returns `qml.vn_entropy(wires=...)` and `qml.purity(wires=...)`. Verified by execution that these two measurement processes coexist in one return tuple **and** can sit alongside `qml.expval` if ever needed.

**Verified facts (all by live execution against pennylane 0.43.0):**
- `qml.vn_entropy(wires=[0,1])` and `qml.purity(wires=[0,1])` work in a `default.qubit` torch-interface backprop QNode. [VERIFIED: probe]
- They return **0-d scalar tensors** for unbatched noise; shape `(batch,)` for batched noise (so pass a single noise vector for a clean representative-state scalar). [VERIFIED: `e.shape == torch.Size([])`; batched probe gave shape `(8,)`]
- They run correctly **inside `torch.no_grad()`** (the callback fires in an eval/no-grad context — confirmed working). [VERIFIED: probe under `no_grad`]
- Purity of the 2-qubit reduced subsystem is bounded `[0.25, 1]` (1/d, d=4); entropy is bounded `[0, ln 4] ≈ [0, 1.386]`. Observed across a 16-sample noise ensemble: entropy mean ≈ 1.18 (std 0.024), purity mean ≈ 0.35 (std 0.013) — stable, so a **single representative noise vector** is adequate; a small noise-ensemble mean (e.g., 16 draws) is an optional robustness upgrade. [VERIFIED: ensemble probe]
- Offline route (`qml.math.vn_entropy(sv, indices=...)`) is **NOT** a drop-in: `indices` is positional and a raw statevector raises `ValueError: not enough values to unpack`. You must `qml.math.reduce_statevector(sv, [0,1])` to a 4×4 DM first, then `qml.math.purity(dm, indices=[0,1])`. The in-QNode measurement avoids all of this. **Use the in-QNode measurement.** [VERIFIED: probe]

```python
# Source: VERIFIED live against pennylane 0.43.0 (/tmp/_pl_probe*.py)
# Add to QuantumGenerator.__init__:
#   self._introspect_qnode = qml.QNode(self._introspect_circuit, self.dev,
#                                      interface="torch", diff_method="backprop")
def _introspect_circuit(self, noise_params, params_pqc):
    # EXACT same body as generator_circuit Steps 1-5 (Hadamard, IQP-RZ, encoding,
    # entangling layers per self.topology, final RX/RY) — but the RETURN is:
    return qml.vn_entropy(wires=[0, 1]), qml.purity(wires=[0, 1])
    # 2|3 balanced bipartition for 5 qubits: wires {0,1} | {2,3,4}.
    # Record this partition in the companion JSON metadata (D-13-09).

def introspect(self, noise_vec):                # noise_vec shape (num_qubits,)
    e, p = self._introspect_qnode(noise_vec, self.params_pqc)
    return float(e), float(p)
```

**Bipartition choice (D-13-09, Claude's discretion):** wires `{0,1}` vs `{2,3,4}` is the recommended 2|3 split — it is the natural "first two qubits" cut and must be recorded verbatim in `entanglement_trajectory.json` metadata. Any 2|3 split is defensible; consistency + recording is what matters.

### Pattern 4: Driver / sweep — clone the Phase 10 template exactly

**What:** `run_ansatz.py` mirrors `run_baselines.py`: one `(variant, seed)` per invocation, idempotent (`shutil.rmtree` + `mkdir` on rerun), writes a frozen `config.yaml` (with `data_hash` per D-10-15), `checkpoint.pt`, `samples.npy`, `metrics.json` into `runs/<variant>/<seed>/`. `run_ansatz_sweep.sh` mirrors `run_baselines_sweep.sh`: `is_complete()` artifact-bundle gate, atomic `sweep_status.json` (tmpfile + `os.rename` under `flock`), `xargs -P 2 -L 1` (NO `multiprocessing.Pool`), `--parallel` guard 1|2 with reject at ≥3, `--dry-run`.

**Sweep matrix:** only **V2 (5 seeds) + V3 (5 seeds) = 10 new quantum training runs** (V1 reuses existing 09.1/10 metrics — D-13-01, no recompute). Plus the introspection runs: **1 instrumented V1 quantum run (seed 42)** + **3 instrumented classical WGAN runs (`wgan_mlp/cnn/lstm`, Pipeline B, seed 42)**. The introspection runs are a separate small driver (`run_introspect.py`), not part of the ansatz sweep matrix, because they need `callback=` wired and a single seed.

**`is_complete()` bundle** for the ansatz sweep (per D-10-14 run-dir convention): `config.yaml`, `checkpoint.pt`, `samples.npy`, `metrics.json` all present and non-empty under `runs/<variant>/<seed>/`. (No `inverse_kwargs.npz` is needed here unless the ansatz table is reported on OD scale — see Open Question 2; if OD-scale is required, carry the Pipeline-B `inverse_kwargs.npz` exactly as `run_baselines.py` does.)

### Pattern 5: CR-01 — differentiable, device-resident spectral PSD loss

**What:** Replace the `scipy.signal.welch` + numpy + `mse*var/var.detach()` construction (`training.py:470-507`) with a pure-torch PSD computed via `torch.fft.rfft` on the device-resident generator output, MSE'd against a target PSD tensor moved to the generator's device + `compute_dtype`. The current code carries **zero** PSD-mismatch gradient (the MSE is a frozen Python float); the fix makes it a real torch term.

```python
# Source: standard differentiable periodogram (torch.fft); replaces welch round-trip
def _spectral_psd_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_flat = fake.reshape(-1)
    real_flat = real.reshape(-1).detach()          # target: no grad through real
    target_dev, target_dt = fake_flat.device, fake_flat.dtype
    real_flat = real_flat.to(device=target_dev, dtype=target_dt)   # CR-01 device fix
    eps = 1e-12
    psd_fake = (torch.fft.rfft(fake_flat).abs() ** 2)
    psd_real = (torch.fft.rfft(real_flat).abs() ** 2)
    return torch.mean((torch.log(psd_fake + eps) - torch.log(psd_real + eps)) ** 2)
```
**Regression test (CONTEXT-mandated):** with `spectral_loss_weight > 0`, after `generator_loss.backward()`, assert `generator.params_pqc.grad is not None and generator.params_pqc.grad.abs().sum() > 0`. **Default behavior unchanged:** `spectral_loss_weight=0.0` still skips the branch entirely (`training.py:356`), so Phases 8–12 reproduce byte-identically (D-13-06 keeps headline weight at 0.0).

### Pattern 6: CR-02 — device/dtype-consistent EarlyStopping restore

**What:** In `EarlyStopping._load_checkpoint` (`training.py:163-171`) pass `map_location` and recast restored tensors + push optimizer state to the active device so it matches `params_pqc` after the Phase-10 MPS device move.

```python
def _load_checkpoint(self, model):
    dev = model.params_pqc.device
    dt = model.params_pqc.dtype
    ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=dev)  # CR-02
    model.params_pqc.data = ckpt["params_pqc"].to(device=dev, dtype=dt)            # recast
    model.critic.load_state_dict(ckpt["critic_state"])
    model.c_optimizer.load_state_dict(ckpt["c_optimizer"])
    model.g_optimizer.load_state_dict(ckpt["g_optimizer"])
    for opt in (model.c_optimizer, model.g_optimizer):                             # opt state -> dev
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
    model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
```
**Regression test (CONTEXT-mandated):** early-stop + restore on CPU, and on MPS if available (skip-marker if not), asserting `params_pqc.device`/`.dtype` and every optimizer-state tensor match the live device/dtype. **Default unchanged:** Phase 13 headline runs pass `early_stopper=None` (D-13-05) so the path is never exercised by the sweeps — the fix lands here purely because the todo is `resolves_phase: 13` and the path is now reachable for future callers.

### Anti-Patterns to Avoid

- **Computing entanglement entropy from the returned expvals.** The 10 PauliX/PauliZ expectations do not determine the reduced density matrix. Must use a statevector-aware QNode (`qml.vn_entropy`/`qml.purity`). [VERIFIED necessity]
- **Editing the `train_wgan_gp` happy path to add instrumentation.** The callback hook exists for exactly this; touching the loop risks breaking the byte-unchanged guarantee Phases 8–12 depend on.
- **Re-running V1 quantum for the comparison table.** D-13-01 reuses existing 09.1/10 5-seed final metrics — recompute is forbidden and wasteful.
- **`multiprocessing.Pool` anywhere.** LOCKED (D-10-24, Phase 09.1 Pitfall 4). `xargs -P 2` only.
- **Changing param-count accounting when adding topology.** Topology touches only CNOT wiring; `idx`/`count_params()` must stay identical so V1/V3 are 75 params and V1 stays byte-unchanged.
- **Reordering or renaming the existing range-CNOT branch.** Keep it as the literal default branch so the drawn tape is unchanged.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reduced-state entanglement entropy | Manual partial trace + eigendecomposition of ρ | `qml.vn_entropy(wires=[0,1])` in-QNode | Native, differentiable, verified; partial trace by hand is error-prone and the wire-index bookkeeping is subtle |
| State purity Tr(ρ²) | Manual ρ² trace | `qml.purity(wires=[0,1])` in-QNode | Native, verified, returns scalar in known bounds |
| Idempotent resumable sweep + atomic status | New orchestration | Clone `run_baselines_sweep.sh` | flock + tmpfile+rename + is_complete bundle gate already battle-tested in Phases 10/12 |
| Per-run artifact bundle + data_hash | New config schema | Clone `run_baselines.py` bundle (config.yaml/checkpoint/samples/metrics + `data_hash`) | D-10-14/D-10-15 contract; Phase 14 reads this exact layout |
| Fidelity metrics for the ansatz table | New metric math | `eval.full_metric_suite` UNCHANGED | D-10-20; same metric set as `baseline_comparison.json` |
| Comparison-table schema | New JSON shape | Extend existing `rows[] + models[]` long-form schema | Phase 14 reads `baseline_comparison.json`-shaped JSON; just add `ansatz`/`depth`/`topology` columns |
| Differentiable PSD | Re-deriving Welch with autograd | `torch.fft.rfft(...).abs()**2` periodogram | One-liner, device-resident, fully differentiable; Welch's segmenting is unnecessary for a penalty term |

**Key insight:** Phase 13 has essentially no novel algorithmic work. Every capability maps onto an existing PennyLane built-in or an existing project artifact contract. The risk is integration discipline (byte-unchanged defaults, correct callback closure, correct API signature), not invention.

## Common Pitfalls

### Pitfall 1: Assuming PennyLane 0.44.0 API

**What goes wrong:** CONTEXT.md canonical-refs says "PennyLane 0.44.0"; planning against 0.44 docs could pick up an API that differs from the installed 0.43.0 (`qml.math` offline signature differs).
**Why it happens:** The CONTEXT.md was written from an assumption, not from `pip show`.
**How to avoid:** All patterns in this doc are verified against the *installed* 0.43.0. Use the in-QNode `qml.vn_entropy`/`qml.purity` measurement (stable across 0.43/0.44), not the offline `qml.math.*` helpers.
**Warning signs:** `ValueError: not enough values to unpack (expected 2, got 1)` from `qml.math.vn_entropy(statevector, ...)` — that's the 0.43 offline signature trap.

### Pitfall 2: Callback closure captures stale generator / wrong noise contract

**What goes wrong:** Snapshot samples don't match the training distribution, or grad accidentally flows.
**Why it happens:** The metrics dict passed to `callback` has no generator/samples; the closure must re-generate using the *exact* training noise contract (`np.random.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, BATCH_SIZE))`, float32, then `.to(float64)*0.1`).
**How to avoid:** Wrap snapshot in `torch.no_grad()`; reuse the exact noise + `*0.1` post-scaling from `training.py:304-313`.
**Warning signs:** Snapshot sample std wildly different from `metrics["std"]` at the same epoch.

### Pitfall 3: Breaking the byte-unchanged default when adding topology

**What goes wrong:** Phases 8–12 reproductions drift because the default circuit changed.
**Why it happens:** Reordering the entangling branch, changing `idx` math, or making `topology` non-defaulted.
**How to avoid:** `topology="range"` is the default and emits the *identical* gate sequence as today. Add a regression test pinning `count_params()==75` and a fixed-seed forward output to a saved reference.
**Warning signs:** A Phase 8–12 parity/regression test fails after the quantum.py edit.

### Pitfall 4: Snapshot epoch off-by-one

**What goes wrong:** "epoch 1000" never recorded because the loop's last index is 999.
**Why it happens:** `for epoch in range(num_epochs)` → final index is `num_epochs-1`; the hook fires it via `epoch+1 == num_epochs`.
**How to avoid:** Match `epoch in {0,250,500,750,999}` and relabel 999→1000 in the snapshot record. Use `eval_every=10` so 0/250/500/750 are all eval epochs.
**Warning signs:** `training_progression.json` has 4 snapshots instead of 5.

### Pitfall 5: `multiprocessing.Pool` reintroduced

**What goes wrong:** Fork-shared numpy RNG corrupts per-seed reproducibility (Phase 09.1 Pitfall 4 / D-10-24, LOCKED).
**How to avoid:** `xargs -P 2 -L 1 bash -c 'run_one ...'` exactly as `run_baselines_sweep.sh`. Reject `--parallel >= 3`.
**Warning signs:** `from multiprocessing import Pool` anywhere in the new driver/sweep.

### Pitfall 6: MPS float64 and entropy QNode device

**What goes wrong:** `train_wgan_gp` moves the generator onto MPS; an introspection QNode is a PennyLane `default.qubit` object (CPU/numpy-backed via torch interface) and is *not* moved by `.to(device)`. `qml.state()` returns complex128 (verified). On the introspection path this is fine because the QNode runs through PennyLane's own simulator, not an MPS tensor op — but reading `generator.params_pqc` (a float32 nn.Parameter possibly on MPS) and feeding it to the QNode must `.detach().cpu()` first if on MPS.
**How to avoid:** In `introspect()`, pass `self.params_pqc` through the QNode as-is for CPU runs; if the generator was moved to MPS by the trainer, the introspection QNode still executes on `default.qubit` (PennyLane backend) — verify the noise tensor and params are CPU-resident before the QNode call. (Sample generation in `run_baselines.py:194` already moves the generator back to CPU for sampling — mirror that for introspection.)
**Warning signs:** `Cannot convert a MPS Tensor to float64` or device-mismatch errors inside the introspection callback.

## Runtime State Inventory

> Phase 13 is **greenfield-additive**, not a rename/refactor. It adds new files and new optional code paths; it modifies `training.py` (CR-01/CR-02) and `quantum.py` (topology spec) but with byte-unchanged defaults. No stored data, service config, OS-registered state, or secrets reference any renamed string. The only "runtime state" concerns are reuse contracts:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data (reused artifacts) | Existing 09.1/10 V1 quantum 5-seed final metrics (reused as ansatz variant-1, D-13-01); Phase-10 classical run dirs (`results/baselines/runs/<model>/B/42/`) read for INTRO-01 instrumented re-runs | Read-only reuse; assert `data_hash` (D-10-15) before they enter the ansatz table / progression figure |
| Live service config | None | None — local-Mac statevector compute only, no external services (verified: PROJECT.md "Compute: Local Mac only") |
| OS-registered state | None | None — no Task Scheduler/launchd/pm2; sweeps run via tmux/nohup ad hoc (verified: `run_baselines_sweep.sh` invocation block) |
| Secrets/env vars | None | None — no secrets in this project; `PYTHON` resolution uses `./qgan_env/bin/python` (verified: sweep script lines 97-107) |
| Build artifacts | None new; `revision.core` is an importable package (no egg-info rename) | None — `quantum.py`/`training.py` edits are in-place; no package rename |

**Nothing found** in live-service / OS-registered / secrets / build-artifact categories — verified against PROJECT.md constraints and the existing sweep script.

## Code Examples

### Verified entanglement-entropy + purity extraction (PennyLane 0.43.0)

```python
# Source: VERIFIED live execution against pennylane 0.43.0 in ./qgan_env
import pennylane as qml, torch
dev = qml.device("default.qubit", wires=5, shots=None)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def introspect(noise, params):
    # ... identical Steps 1-5 of generator_circuit (Hadamard / IQP-RZ / encoding /
    #     topology-selected entangling layers / final RX,RY) ...
    return qml.vn_entropy(wires=[0, 1]), qml.purity(wires=[0, 1])   # 2|3 split

with torch.no_grad():                              # callback runs read-only
    e, p = introspect(noise_vec, params_pqc)       # noise_vec shape (5,)
# e, p are 0-d tensors; float(e) in [0, ln4≈1.386], float(p) in [0.25, 1]
```

### Verified coexistence with expvals (if a single fused QNode is preferred)

```python
# Source: VERIFIED — pennylane 0.43.0 allows expval + vn_entropy + purity in one return
return (qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0)), ...,
        qml.vn_entropy(wires=[0, 1]), qml.purity(wires=[0, 1]))
# Returned 12 values; entropy/purity are the last two. (A separate read-only
# QNode is still recommended to keep the production measurement QNode byte-unchanged.)
```

### ansatz_comparison.json schema (extends the verified long-form schema)

```json
// Source: VERIFIED shape of results/baseline_comparison.json (1710 rows)
{
  "schema": "long-form rows[] + models[] aggregate (D-10-16) + ansatz dim (Phase 13)",
  "model_kinds": ["quantum"],
  "ansatz_variants": [
    {"variant": "V1", "depth": 4, "topology": "range",  "parameter_count": 75,
     "source": "reused 09.1/10 5-seed final metrics (D-13-01, no recompute)"},
    {"variant": "V2", "depth": 8, "topology": "range",  "parameter_count": 135,
     "source": "Phase 13 new 5-seed runs"},
    {"variant": "V3", "depth": 4, "topology": "linear", "parameter_count": 75,
     "source": "Phase 13 new 5-seed runs"}
  ],
  "seeds": [42, 43, 44, 45, 46],
  "rows": [
    {"model_kind": "quantum", "variant": "V2", "depth": 8, "topology": "range",
     "pipeline": "B", "seed": 42, "metric_name": "emd", "scale": "OD", "value": 0.0}
  ]
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `qml.math.vn_entropy(state, indices=...)` keyword offline | In-QNode `qml.vn_entropy(wires=...)` measurement process | PennyLane ≥0.40 (return-type measurements stable) | Offline `qml.math` signature is positional + needs DM reduction in 0.43; in-QNode route is the stable, recommended path |
| Welch (`scipy.signal.welch`) PSD penalty with numpy round-trip | `torch.fft.rfft` periodogram (device-resident, differentiable) | CR-01 fix (this phase) | Removes the broken zero-gradient construction; spectral hook becomes correct if enabled |

**Deprecated/outdated:**
- The CONTEXT.md "PennyLane 0.44.0" reference — actual installed version is **0.43.0**. Plan against 0.43.0; the in-QNode measurement API is identical across both, so the V3/INTRO-03 patterns are version-robust.

## Project Constraints (from PROJECT.md / CONTEXT.md / Phase 10/12)

- **Main notebook untouched** — `qgan_pennylane.ipynb` stays as-is; all work in `revision/`.
- **Compute: Local Mac only** — statevector simulator; sweep sized accordingly (`--parallel ≤ 2`).
- **Results contract** — every artifact is structured JSON under `results/<name>.json`; figures under `results/figures/` with companion `*.json`.
- **`core/` byte-untouched except the two folded fixes** — `quantum.py` topology add and `training.py` CR-01/CR-02 must preserve byte-unchanged defaults (`spectral_loss_weight=0.0`, `callback=None`, `early_stopper=None`, default `topology="range"`).
- **No new variance-collapse remediation** — report honestly; explain dynamics, don't close the gap.
- **No `multiprocessing.Pool`** — `xargs -P 2` OS processes only (D-10-24, LOCKED).
- **`data_hash` (D-10-15)** asserted on any reused V1/classical artifact before it enters the ansatz table / progression figure.
- **Code-placement (D-10-13)** — model defs in `core/`, all sweep/figure orchestration in `run_*.py`.
- **Quantum-vs-quantum comparison** — param-count drift 75/135/75 is expected and correct; NOT matched-parameter.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The ansatz comparison table is reported on the same metric set + scale convention as `baseline_comparison.json` (EMD/moments/JSD, dual-scale per EVAL-05) | Code Examples / Don't Hand-Roll | Low — D-13 discretion explicitly says "same metric set as Phase 10"; if OD-scale required, carry Pipeline-B `inverse_kwargs.npz` |
| A2 | V3 = linear nearest-neighbour open chain is the most reviewer-defensible topology contrast (vs. circular NN) | Pattern 1 | Low — D-13-02 locks the axis and names linear as canonical; circular is the only alternative and is *not* strictly more defensible (open-chain is the textbook minimal-connectivity contrast to the wrap-around range pattern). **Recommend keeping linear.** |
| A3 | Single representative noise vector gives a stable entropy/purity reading (vs. requiring a noise ensemble) | Pattern 3 | Low — VERIFIED ensemble std is tiny (entropy std 0.024, purity std 0.013); single-vector is adequate, ensemble-of-16 is an optional robustness upgrade recorded in JSON |
| A4 | INTRO-01 classical instrumented runs use Pipeline B seed 42 at 1000 epochs (the headline pipeline) | Pattern 4 | Low — D-13-08 locks this explicitly |
| A5 | `eval_every=10` (project default) is used for the instrumented run so {0,250,500,750,1000} all land on eval epochs | Pattern 2 / Pitfall 4 | Low — verified arithmetic; any divisor of 250 works, 10 is the established default |

## Open Questions

1. **OD-scale vs log-return-scale for the ansatz table.**
   - What we know: D-13 discretion says "dual-scale per EVAL-05 convention; same metric set as Phase 10's `baseline_comparison.json`". `baseline_comparison.json` rows carry a `scale` field with `OD`.
   - What's unclear: whether the ansatz table needs OD-scale reconstruction (requiring Pipeline-B `inverse_kwargs.npz` in each run-dir) or log-return scale suffices for a quantum-vs-quantum depth/topology comparison.
   - Recommendation: emit both scales (mirror `baseline_comparison.json`'s `scale` dimension) so it slots into Phase 14 with no rework; carry `inverse_kwargs.npz` in the ansatz run-dir bundle exactly as `run_baselines.py` does for Pipeline B. Cheap insurance; planner can downscope to log-return-only if compute-tight.

2. **Whether V2 (depth-8) being a strong performer triggers the deferred Phase-10 matched-param re-run.**
   - What we know: Phase 13 deferred list flags this as a Phase-14-time decision; default assumption is V1 remains the published circuit.
   - What's unclear: nothing actionable in Phase 13 — this is explicitly out of scope and deferred.
   - Recommendation: Phase 13 produces the comparison and states the result honestly; the matched-param re-run decision is NOT a Phase 13 task. No action.

3. **Introspection driver granularity.**
   - What we know: ARCH sweep needs 10 runs (V2/V3 × 5 seeds); INTRO needs 1 quantum + 3 classical instrumented runs (seed 42).
   - What's unclear: one combined `run_introspect.py` vs. reusing `run_ansatz.py` with an `--instrument` flag.
   - Recommendation: separate `run_introspect.py` — the instrumented run has a different contract (callback wired, single seed, captures snapshots into a JSON, no sweep matrix). Keeps `run_ansatz.py` a clean clone of `run_baselines.py`. Planner's call; both are viable.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python (venv) | All | ✓ | `./qgan_env/bin/python` 3.11 | system python3 (sweep script auto-detects) |
| pennylane | INTRO-03, ansatz circuits | ✓ | 0.43.0 | — (no fallback; core dependency) |
| torch | training, CR-01 fft | ✓ | 2.9.0 | — |
| numpy | glue/RNG | ✓ | 2.3.4 | — |
| scipy | current welch (being removed by CR-01) | ✓ | 1.16.2 | torch.fft (the fix removes the scipy dependency from the grad path) |
| matplotlib | figures | ✓ (project-installed, used by prior phases' figure code) | — | — |
| MPS (Apple) | training accel (optional) | host-dependent | — | CPU float64 path (handled in `train_wgan_gp`) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** scipy welch (CR-01 replaces with torch.fft — this is the *intended* fix, not a degradation).

## Validation Architecture

> `.planning/config.json` not inspected for an explicit `workflow.nyquist_validation: false`; treating as enabled. Section included.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project convention — `tests/` dir, prior phases use it) |
| Config file | none detected at repo root — confirm/Wave 0 |
| Quick run command | `./qgan_env/bin/python -m pytest tests/test_cr01_spectral_grad.py tests/test_cr02_es_restore.py -x -q` |
| Full suite command | `./qgan_env/bin/python -m pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ARCH-01 | V1/V2/V3 selectable; default byte-unchanged (count_params==75, fixed-seed fwd matches ref) | unit | `pytest tests/test_ansatz_variants.py -x` | ❌ Wave 0 |
| ARCH-01 | V2 → 135 params, V3 → 75 params, V3 uses linear CNOT only | unit | `pytest tests/test_ansatz_variants.py -x` | ❌ Wave 0 |
| ARCH-02 | `ansatz_comparison.json` validates against extended rows[]+models[] schema | integration | `pytest tests/test_ansatz_json_schema.py -x` | ❌ Wave 0 |
| INTRO-01/02/03 | snapshot callback fires at {0,250,500,750,1000}; captures samples+param_norm+entropy+purity | integration | `pytest tests/test_introspect_callback.py -x` (short run, num_epochs small, SNAP rescaled) | ❌ Wave 0 |
| INTRO-03 | `introspect()` returns scalar entropy∈[0,ln4], purity∈[0.25,1] | unit | `pytest tests/test_entropy_purity.py -x` | ❌ Wave 0 |
| CR-01 | non-zero grad into params_pqc when spectral_loss_weight>0 | unit | `pytest tests/test_cr01_spectral_grad.py -x` | ❌ Wave 0 |
| CR-01 | spectral_loss_weight=0.0 path byte-unchanged (branch skipped) | unit | `pytest tests/test_cr01_spectral_grad.py -x` | ❌ Wave 0 |
| CR-02 | early-stop+restore device/dtype-consistent (CPU; MPS skip-if-unavailable) | unit | `pytest tests/test_cr02_es_restore.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** the relevant focused test file (e.g., `pytest tests/test_cr01_spectral_grad.py -x -q`)
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** full suite green before `/gsd:verify-work`; plus a `--dry-run` of `run_ansatz_sweep.sh` showing the 10-run matrix resolves correctly

### Wave 0 Gaps

- [ ] `tests/test_ansatz_variants.py` — covers ARCH-01 (selectability + byte-unchanged default + param counts)
- [ ] `tests/test_ansatz_json_schema.py` — covers ARCH-02 (extended long-form schema)
- [ ] `tests/test_introspect_callback.py` — covers INTRO-01/02/03 callback wiring (short run)
- [ ] `tests/test_entropy_purity.py` — covers INTRO-03 bounds + API
- [ ] `tests/test_cr01_spectral_grad.py` — covers CR-01 (grad present when on; skipped when off)
- [ ] `tests/test_cr02_es_restore.py` — covers CR-02 (device/dtype consistency, CPU + MPS-skip)
- [ ] Confirm pytest config / `tests/` conftest exists; add if absent

## Security Domain

> Local research/scientific code, no auth/network/user-input surface, no `security_enforcement` config detected. ASVS categories are **not applicable**: there is no authentication (V2), session (V3), access control (V4), or cryptography (V6). V5 input validation is limited to CLI argparse `choices=` guards (already the pattern in `run_baselines.py`). Threat surface is reproducibility integrity (covered by `data_hash` D-10-15 and idempotent bundles), not security. No security controls required for this phase.

## Sources

### Primary (HIGH confidence — verified by live execution)
- `./qgan_env` live probes (`/tmp/_pl_probe.py`, `_pl_probe2.py`, `_pl_probe3.py`) — pennylane 0.43.0 `vn_entropy`/`purity`/`density_matrix`/`state` in-QNode + offline `qml.math` signatures + no_grad behavior + batched/scalar shapes + entropy/purity bounds + noise-ensemble stability
- `core/models/quantum.py` (read) — generator_circuit structure, param-count formula, range-CNOT block
- `core/training.py` (read) — callback hook (396-411), spectral hook (356-360, 470-507), EarlyStopping restore (163-171), MPS/float64 device logic
- `run_baselines.py` + `run_baselines_sweep.sh` (read) — idempotent driver + atomic sweep template
- `results/baseline_comparison.json` (read) — verified `rows[] + models[]` long-form schema (1710 rows)
- `core/eval.py` (read) — `full_metric_suite` signature + keys
- `.planning/phases/13-architecture-introspection/13-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/PROJECT.md`, the two CR-01/CR-02 todos (read)

### Secondary (MEDIUM confidence)
- Phase 10/12 CONTEXT decisions referenced via 13-CONTEXT canonical-refs (not re-read in full; constraints quoted from CONTEXT.md summaries)

### Tertiary (LOW confidence)
- None — no unverified web claims; entire phase verified against installed stack and codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every version pinned by direct interpreter query; no new packages
- Architecture / API patterns: HIGH — entanglement API, callback hook, schema all verified by execution/read, not memory
- Pitfalls: HIGH — derived from code reading + the PennyLane-version correction caught by probing
- CR-01/CR-02 fix shapes: MEDIUM-HIGH — fix patterns are standard and the regression-test shapes are CONTEXT-mandated, but exact landed code is the planner/implementer's to finalize against the live `compute_dtype` logic

**Research date:** 2026-05-18
**Valid until:** ~2026-06-17 (30 days — stable; only risk is an environment package upgrade. If `pennylane` is bumped to 0.44+, re-verify the offline `qml.math` signature only — the in-QNode measurement API used here is version-robust.)
