# Phase 13: Architecture & Introspection - Pattern Map

**Mapped:** 2026-05-18
**Files analyzed:** 11 (3 modified core, 4 new drivers/sweeps, 1 JSON contract, 6 figure/JSON artifacts grouped, 6 test files)
**Analogs found:** 9 / 11 (figures = no in-repo analog; tests = no in-repo analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `revision/core/models/quantum.py` (MODIFIED — topology spec + `introspect()`) | model | transform | itself (`generator_circuit` lines 137–155) | exact (self-extension) |
| `revision/core/training.py` (MODIFIED — CR-01 `_spectral_psd_loss`) | training | transform | itself (lines 470–507) | exact (self-replace) |
| `revision/core/training.py` (MODIFIED — CR-02 `EarlyStopping._load_checkpoint`) | training | file-I/O | itself (lines 163–175) | exact (self-replace) |
| `revision/run_ansatz.py` (NEW) | driver | batch (request-response per proc) | `revision/run_baselines.py` | exact (clone) |
| `revision/run_ansatz_sweep.sh` (NEW) | driver/orchestration | batch | `revision/run_baselines_sweep.sh` | exact (clone) |
| `revision/run_introspect.py` (NEW) | driver | event-driven (callback) | `revision/run_baselines.py` (`_train_wgan`) + callback hook `training.py:396-411` | role-match |
| `revision/run_introspect_figures.py` (NEW) | utility | transform → file-I/O | `revision/run_dualscale_fidelity.py` (JSON-emit shape) | partial — no matplotlib analog in repo |
| `revision/results/ansatz_comparison.json` (NEW artifact) | config/contract | — | `revision/run_dualscale_fidelity.py` rows[] builder (lines 358–376, 400–434) | exact (schema clone) |
| `revision/results/figures/*.{png,pdf}` + companion `*.json` (NEW) | artifact | file-I/O | none (no figure-rendering code anywhere in repo) | **no analog** |
| `tests/test_cr01_spectral_grad.py`, `test_cr02_es_restore.py`, `test_ansatz_variants.py`, `test_ansatz_json_schema.py`, `test_introspect_callback.py`, `test_entropy_purity.py` (NEW) | test | — | none (`tests/` dir does not exist yet) | **no analog** |

---

## Pattern Assignments

### `revision/core/models/quantum.py` — topology spec + `introspect()` (model, transform)

**Analog:** itself — extend `QuantumGenerator` in place; default branch MUST be byte-identical (PROJECT constraint, Pitfall 3).

**Imports pattern** (lines 20–24) — keep exactly; only add nothing new (PennyLane already imported):
```python
from __future__ import annotations
import torch
import torch.nn as nn
import pennylane as qml
```

**`__init__` signature + param-count + QNode construction** (lines 38–78) — add `topology: str = "range"` keyword, store `self.topology`, leave `self.num_params` formula at line 59–61 **unchanged** (topology touches only CNOT wiring, not param count → V1/V3=75, V2=135). Add the second read-only QNode next to the existing one (mirror lines 73–78):
```python
self.qnode = qml.QNode(self.generator_circuit, self.dev,
                        interface="torch", diff_method=diff_method)
# NEW — read-only introspection QNode (same device, same interface)
self._introspect_qnode = qml.QNode(self._introspect_circuit, self.dev,
                                    interface="torch", diff_method="backprop")
```

**Core pattern — the topology switch** replaces the fixed range block at **lines 150–155**. The existing block (the DEFAULT, must stay literally first):
```python
# Range-based entangling CNOTs.   (lines 150-155 — DEFAULT branch, unchanged)
if self.num_qubits > 1:
    range_param = (layer % (self.num_qubits - 1)) + 1
    for qubit in range(self.num_qubits):
        target_qubit = (qubit + range_param) % self.num_qubits
        qml.CNOT(wires=[qubit, target_qubit])
```
Wrap with `if self.topology == "range":` (this exact body) `elif self.topology == "linear":` (open chain `for q in range(n-1): qml.CNOT([q, q+1])`). Do NOT touch the `idx` accounting at lines 122–163. RESEARCH Pattern 1 verified the param math.

**`introspect()` + `_introspect_circuit`** — clone Steps 1–5 of `generator_circuit` (lines 122–163) verbatim into `_introspect_circuit`; replace the Step-6 measurement loop (lines 165–171) with `return qml.vn_entropy(wires=[0,1]), qml.purity(wires=[0,1])`. `introspect(noise_vec)` calls `self._introspect_qnode(noise_vec, self.params_pqc)` and returns `float(e), float(p)`. RESEARCH Pattern 3 verified the 0.43.0 API and bounds.

---

### `revision/core/training.py` — CR-01 `_spectral_psd_loss` (training, transform)

**Analog:** itself — replace the body of `_spectral_psd_loss` at **lines 470–507** (the scipy.welch + numpy + `mse*var/var.detach()` proxy). The call site **lines 356–360** stays unchanged (the `if spectral_loss_weight > 0.0:` guard preserves byte-unchanged default — D-13-06).

**Pattern to copy** (RESEARCH Pattern 5 — differentiable torch periodogram, device-resident):
```python
def _spectral_psd_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_flat = fake.reshape(-1)
    real_flat = real.reshape(-1).detach()
    real_flat = real_flat.to(device=fake_flat.device, dtype=fake_flat.dtype)  # CR-01 device fix
    eps = 1e-12
    psd_fake = torch.fft.rfft(fake_flat).abs() ** 2
    psd_real = torch.fft.rfft(real_flat).abs() ** 2
    return torch.mean((torch.log(psd_fake + eps) - torch.log(psd_real + eps)) ** 2)
```
Drop the `from scipy.signal import welch` import inside the function. Keep the `real_log_returns_for_psd` helper (lines 464–467) unchanged.

---

### `revision/core/training.py` — CR-02 `EarlyStopping._load_checkpoint` (training, file-I/O)

**Analog:** itself — replace `_load_checkpoint` at **lines 163–175**. Current code does an un-mapped `torch.load` + raw `.data =` assignment:
```python
# CURRENT (lines 165-171) — device/dtype-unsafe:
checkpoint = torch.load(self.checkpoint_path, weights_only=False)
model.params_pqc.data = checkpoint["params_pqc"]
model.critic.load_state_dict(checkpoint["critic_state"])
model.c_optimizer.load_state_dict(checkpoint["c_optimizer"])
model.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
```
**Pattern to copy** (RESEARCH Pattern 6 — `map_location` + dtype recast + opt-state to device). Note `model` here is the `_ESAdapter` (lines 438–461) whose `params_pqc` property proxies `generator.params_pqc`; `.critic / .c_optimizer / .g_optimizer` are direct attributes:
```python
dev = model.params_pqc.device
dt = model.params_pqc.dtype
ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=dev)
model.params_pqc.data = ckpt["params_pqc"].to(device=dev, dtype=dt)
model.critic.load_state_dict(ckpt["critic_state"])
model.c_optimizer.load_state_dict(ckpt["c_optimizer"])
model.g_optimizer.load_state_dict(ckpt["g_optimizer"])
for opt in (model.c_optimizer, model.g_optimizer):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
```
Keep the trailing `print(...)` (lines 172–175). Default unchanged: Phase 13 headline runs pass `early_stopper=None` (D-13-05) so this path is unexercised by the sweep.

---

### `revision/run_ansatz.py` (driver, batch — one (variant, seed) per process)

**Analog:** `revision/run_baselines.py` — clone structurally; this is the cleanest match (idempotent per-run, 5-file bundle, `data_hash`, WGAN branch).

**Imports + constants pattern** (`run_baselines.py:50–91`) — copy verbatim; swap `revision.core.models.classical` import for `revision.core.models.quantum.QuantumGenerator`:
```python
import argparse, hashlib, json, shutil
from dataclasses import dataclass
from pathlib import Path
from revision.core import (BATCH_SIZE, EVAL_EVERY, LAMBDA, LR_CRITIC,
    LR_GENERATOR, N_CRITIC, NOISE_HIGH, NOISE_LOW, NUM_LAYERS, NUM_QUBITS,
    WINDOW_LENGTH)
from revision.core.data import load_and_preprocess, rolling_window
from revision.core.models.quantum import QuantumGenerator        # <- swapped
from revision.core.models.critic import Critic
from revision.core.training import train_wgan_gp
```

**Dataset bundle** — copy `DatasetBundle` (lines 101–114) + `build_dataset_for_pipeline` Pipeline-B branch (lines 140–158) verbatim. Phase 13 headline is Pipeline B; carry `inverse_kwargs.npz` exactly as lines 211–223 if OD-scale is emitted (RESEARCH Open Q1 — recommend emit both scales).

**WGAN training branch** — copy `_train_wgan` (lines 237–282) but construct `QuantumGenerator(num_layers=depth, topology=topology)` instead of `_WGAN_GENERATORS[model_kind]()`, and call `train_wgan_gp(..., num_epochs=1000)` with **no `early_stopper`** (D-13-05) and `spectral_loss_weight=0.0` (D-13-06 — the default). Sample generation: copy `generate_wgan_samples` (lines 177–208) **verbatim including the `.to("cpu")` + `*0.1`** (RESEARCH Pitfall 6).

**`data_hash` + idempotent run-dir + config.yaml + 5-file persist** — copy `_compute_data_hash` (lines 226–234), `main()` run-dir clean (lines 456–463), and the config/persist block (lines 490–522) verbatim; change run-dir to `runs/<variant>/<seed>/` and add `ansatz`/`depth`/`topology` to `extra_cfg`.

**Error handling pattern:** `run_baselines.py` has none beyond argparse `choices=` guards (lines 438–440) and the idempotent `shutil.rmtree` (lines 461–462). Mirror exactly — use `choices=["V1","V2","V3"]` or `--depth/--topology` argparse guards.

---

### `revision/run_ansatz_sweep.sh` (driver/orchestration, batch)

**Analog:** `revision/run_baselines_sweep.sh` — clone verbatim, change only the matrix and `is_complete` bundle.

**Copy verbatim:** `set -euo pipefail` (line 81); PYTHON resolution (lines 94–107); `--parallel` 1|2 guardrail with reject ≥3 (lines 146–156); `is_complete()` (lines 174–184); `iso_now` (lines 186–188); `update_status()` atomic tmpfile+`os.rename` under `flock -x 9` (lines 199–282); `run_one()` (lines 290–335); `xargs -P 2 -L 1 bash -c 'run_one "$0" "$1" "$2"'` dispatch (lines 400–419); final summary + non-zero-if-incomplete exit (lines 424–480).

**Change only:**
- Matrix constants (lines 86–91): `VARIANTS="V2 V3"`, `SEEDS="42 43 44 45 46"`, `EPOCHS=1000` → 10 runs, `total_count=10`. **V1 is NOT in the matrix** (D-13-01 reuses 09.1/10 final metrics — no recompute).
- `is_complete()` bundle (lines 179–183): drop the AR `.npz` special-case; check `config.yaml`, `checkpoint.pt`, `samples.npy`, `metrics.json` (+ `inverse_kwargs.npz` only if OD-scale emitted).
- The `run_one` python invocation (lines 313–319): `-m revision.run_ansatz --variant "$v" --seed "$s"`.
- 2-tuple key `(variant, seed)` instead of the 3-tuple `(model, pipeline, seed)` throughout `update_status`/`is_complete`.

**Anti-pattern guard:** never introduce `multiprocessing.Pool` (D-10-24 / Pitfall 5 — the sweep header lines 34–43 document why; preserve that comment).

---

### `revision/run_introspect.py` (driver, event-driven via callback)

**Analog:** `revision/run_baselines.py` `_train_wgan` (lines 237–282) for the train wiring + `training.py:396-411` callback hook for the instrumentation contract. Separate driver, not a `run_ansatz.py` flag (RESEARCH Open Q3 recommendation).

**Train wiring:** reuse `_train_wgan` shape but pass a `callback=` closure. Quantum run: `QuantumGenerator(num_layers=4, topology="range")` (V1, seed 42). Classical runs: `WGANMLPGenerator/WGANCNNGenerator/WGANLSTMGenerator` (Pipeline B, seed 42) — import from `revision.core.models.classical` (see `run_baselines.py:79–83`, `_WGAN_GENERATORS` lines 93–97).

**Callback closure pattern** — the hook (`training.py:396-411`) passes only a scalar metrics dict; the closure must close over `generator` and re-generate. Copy the exact training noise contract from `training.py:304-313` / `run_baselines.py:198-207` (`np.random.default_rng(seed).uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, BATCH_SIZE))`, float32, `.to(float64)*0.1`, generator on CPU). RESEARCH Pattern 2:
```python
SNAP = {0, 250, 500, 750, 999}      # 999 == final index for num_epochs=1000
snapshots = []
def snapshot_cb(epoch, metrics):
    if epoch not in SNAP: return
    label = 1000 if epoch == 999 else epoch
    with torch.no_grad():
        noise = torch.tensor(rng.uniform(NOISE_LOW, NOISE_HIGH,
                size=(NUM_QUBITS, BATCH_SIZE)), dtype=torch.float32)
        gen = (generator.to("cpu")(noise).to(torch.float64) * 0.1).cpu().numpy()
        p = generator.params_pqc.detach().cpu().numpy()
        ent, pur = generator.introspect(noise[:, 0])      # quantum only
    snapshots.append({"epoch": label, "samples": gen.tolist(),
        "param_norm": float(np.linalg.norm(p)), "param_angles": p.tolist(),
        "vn_entropy": float(ent), "purity": float(pur),
        "emd": metrics["emd"], "std": metrics["std"]})
```
Use `eval_every=10` (project default, `EVAL_EVERY`) so 0/250/500/750 all land on eval epochs (RESEARCH Pitfall 4). Classical variants have no `introspect()` — guard with `hasattr(generator, "introspect")`.

**Persist:** snapshots → JSON under `revision/results/figures/*.json` companion (the reproducibility contract, ROADMAP criterion 4). Record the bipartition `{0,1}|{2,3,4}` in metadata (D-13-09).

---

### `revision/run_introspect_figures.py` (utility, transform → file-I/O)

**Analog (JSON-emit half only):** `revision/run_dualscale_fidelity.py` `main()` (lines 306–376) — argparse + `out_path.write_text(json.dumps(obj, indent=2))` pattern.

**No analog for the matplotlib half** — there is zero figure-rendering code anywhere in `revision/` (`run_ablation.py:17` explicitly states "NO matplotlib import"). The planner should use RESEARCH §"Recommended Project Structure" + D-13 figure discretion (panel layout / format / styling free) and standard matplotlib idioms. Each figure MUST have a companion `*.json` (ROADMAP criterion 4) — reuse the `json.dumps(..., indent=2)` write pattern from `run_dualscale_fidelity.py:376`.

---

### `revision/results/ansatz_comparison.json` (config/contract artifact)

**Analog:** `revision/run_dualscale_fidelity.py` rows[] builder — lines 358–376 (envelope) + 400–434 (OD rows) + 471–509 (log_return rows).

**Envelope pattern** (lines 358–376):
```python
obj = {
    "schema": "long-form rows[] + models[] (D-10-16) + ansatz/depth/topology (Phase 13)",
    "model_kinds": ["quantum"],
    "ansatz_variants": [...],     # V1 reuse / V2 / V3 with depth/topology/param_count
    "seeds": [42, 43, 44, 45, 46],
    "rows": rows,
}
out_path.write_text(json.dumps(obj, indent=2))
```

**Row pattern** (lines 400–434) — one dict per `(model_kind, variant, depth, topology, pipeline, seed, metric_name, scale, value)`. Metrics come from `revision.core.eval.full_metric_suite` (`eval.py:143-163`) UNCHANGED (D-10-20) — same metric set as `baseline_comparison.json`. **V1 rows are read from the existing 09.1/10 quantum run metrics — NOT recomputed** (D-13-01); assert `data_hash` (D-10-15) via the `verify_data_hash` pattern (`run_dualscale_fidelity.py:241-266`) before V1/classical artifacts enter the table.

---

## Shared Patterns

### Idempotent per-run bundle + data_hash
**Source:** `revision/run_baselines.py:226-234` (`_compute_data_hash`), `:456-463` (clean run-dir), `:490-522` (config.yaml + 5-file persist)
**Apply to:** `run_ansatz.py`, `run_introspect.py`
```python
od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
data_hash = hashlib.sha256(od.tobytes()).hexdigest()[:16]   # D-10-15
if run_dir.exists(): shutil.rmtree(run_dir)                  # idempotent
run_dir.mkdir(parents=True, exist_ok=True)
```

### Training-noise contract (exact, load-bearing)
**Source:** `revision/core/training.py:304-313` and `revision/run_baselines.py:198-207`
**Apply to:** every sample/snapshot generation in `run_ansatz.py` and `run_introspect.py`
```python
generator = generator.to("cpu")               # MPS has no float64 (Pitfall 6)
noise = torch.tensor(rng.uniform(NOISE_LOW, NOISE_HIGH,
        size=(NUM_QUBITS, bs)), dtype=torch.float32)
out = generator(noise).to(torch.float64) * 0.1   # *0.1 is the quantum-output artifact
```

### Atomic resumable sweep status
**Source:** `revision/run_baselines_sweep.sh:199-282` (`update_status` — tmpfile + `os.rename` under `flock -x 9`), `:174-184` (`is_complete`)
**Apply to:** `run_ansatz_sweep.sh`
Never replace `xargs -P 2 -L 1` with `multiprocessing.Pool` (D-10-24 / Pitfall 5).

### Long-form metrics schema
**Source:** `revision/run_dualscale_fidelity.py:358-376` (envelope), `eval.py:143` (`full_metric_suite`)
**Apply to:** `ansatz_comparison.json` — extend `rows[]` with `ansatz`/`depth`/`topology`, do not replace; reuse `full_metric_suite` UNCHANGED (D-10-20).

### Byte-unchanged-default discipline
**Source:** `quantum.py:38-44` (defaulted kwargs), `training.py:189-208` (`spectral_loss_weight=0.0`, `callback=None`, `early_stopper=None` no-op defaults)
**Apply to:** all three `revision/core/` edits — `topology="range"` default, CR-01 guarded by `if spectral_loss_weight > 0.0`, CR-02 only on the `early_stopper is not None` path. Phases 8–12 must reproduce byte-identically.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `revision/results/figures/training_progression.{png,pdf}` | artifact | file-I/O | No matplotlib/figure-rendering code exists anywhere in `revision/` (`run_ablation.py:17`: "NO matplotlib import"). Planner uses RESEARCH structure + D-13 figure discretion + standard matplotlib; companion JSON reuses `run_dualscale_fidelity.py:376` write pattern. |
| `revision/results/figures/param_trajectory.{png,pdf}` | artifact | file-I/O | Same — no figure analog. |
| `revision/results/figures/entanglement_trajectory.{png,pdf}` | artifact | file-I/O | Same — no figure analog; bipartition metadata `{0,1}|{2,3,4}` recorded in companion JSON (D-13-09). |
| `tests/test_cr01_spectral_grad.py` | test | — | No `tests/` directory exists yet (verified). RESEARCH Validation Architecture mandates pytest; Wave 0 must create `tests/` (+ conftest if needed). Test shapes are CONTEXT-mandated (assert non-zero grad into `params_pqc` when `spectral_loss_weight>0`; skipped when `=0.0`). |
| `tests/test_cr02_es_restore.py` | test | — | Same — no test analog. CONTEXT-mandated: early-stop+restore device/dtype consistency on CPU and MPS (skip-marker if MPS unavailable). |
| `tests/test_ansatz_variants.py` | test | — | Same — covers ARCH-01: `count_params()==75` default byte-unchanged, V2=135, V3=75, V3 uses linear CNOT only. |
| `tests/test_ansatz_json_schema.py` | test | — | Same — covers ARCH-02 extended rows[]+models[] schema. |
| `tests/test_introspect_callback.py` | test | — | Same — covers INTRO-01/02/03 callback firing at {0,250,500,750,1000} on a short run. |
| `tests/test_entropy_purity.py` | test | — | Same — covers INTRO-03 bounds: entropy∈[0,ln4], purity∈[0.25,1]. |

---

## Metadata

**Analog search scope:** `revision/core/` (models, training, eval), `revision/run_*.py` (7 drivers), `revision/run_*sweep.sh` (3 sweeps), `tests/` (absent), `revision/results/figures/` (absent)
**Files scanned:** quantum.py, training.py, eval.py, run_baselines.py, run_baselines_sweep.sh, run_dualscale_fidelity.py (+ grep survey of run_ablation/run_multiseed_rollup/run_sensitivity)
**Key correction carried from RESEARCH:** PennyLane installed version is **0.43.0** (not 0.44.0 as CONTEXT canonical-refs states) — in-QNode `qml.vn_entropy`/`qml.purity` API is version-robust; the offline `qml.math.*` route is NOT a drop-in on 0.43.
**Pattern extraction date:** 2026-05-18
