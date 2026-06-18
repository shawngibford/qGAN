# Phase 12: Sensitivity Analysis - Pattern Map

**Mapped:** 2026-05-18
**Files analyzed:** 6 (3 source + 3 JSON artifacts)
**Analogs found:** 6 / 6 (all exact or strong role-match)

Phase 12 is ~90% wiring of frozen components. Every new file has a direct, shipped analog already in `revision/`. This map gives the planner the exact analog file + line-referenced excerpts to copy from.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `run_sensitivity.py` | driver (CLI, one cell/invocation) | transform + request-response | `run_baselines.py` (CLI shape) + `run_ablation.py:179-209` (`generate_samples`) + `run_utility.py:130-185` (`reconstruct_od`) | exact (composite) |
| `run_sensitivity_sweep.sh` | sweep orchestration | batch / event-driven | `run_baselines_sweep.sh` | exact |
| `run_multiseed_rollup.py` | aggregator (single invocation) | batch / transform | `run_utility.py` (consumer/aggregation shape, repo-resolver) + Code Example 4 (groupby) | role-match |
| `results/shot_noise_sensitivity.json` | output artifact | data emission | `results/baseline_comparison.json` (long-form `rows[]` schema) | exact (extend) |
| `results/noise_model_sensitivity.json` | output artifact | data emission | `results/baseline_comparison.json` | exact (extend) |
| `results/multiseed_summary.json` | output artifact | data emission | `results/baseline_comparison.json` + RESEARCH Code Example 4 | exact (extend) |
| `core/*` | UNTOUCHED | — | n/a (D-10-13 invariant — assert `git diff --stat core/` empty) | — |

---

## Pattern Assignments

### `run_sensitivity.py` (driver, SENS-01 + SENS-02 per-cell inference)

**Primary analog:** `run_baselines.py` (CLI/idempotency/config-emit skeleton)
**Secondary analogs:** `run_ablation.py:179-209` (`generate_samples` — *0.1 contract), `run_utility.py:38-58, 130-185` (repo-root resolver + `reconstruct_od`), `core/models/quantum.py:103-202` (circuit body to copy for the noisy QNode).

**Module docstring + imports pattern** — copy the structure of `run_baselines.py:1-91`. Notice: docstring states the invariant decisions (D-12-01/02/03) and Pitfalls inline; imports pull HPO constants from `revision.core` never as literals (`run_baselines.py:65-77`):
```python
from revision.core import (
    BATCH_SIZE, NOISE_HIGH, NOISE_LOW,
    NUM_LAYERS, NUM_QUBITS, WINDOW_LENGTH,
)
```

**Repo-root resolver (Pitfall 6)** — copy verbatim from `run_utility.py:38-58`:
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
```

**PennyLane version assertion (Pitfall 5)** — NEW code, ~3 lines, add at driver startup (no analog — RESEARCH Open Q1 recommendation (a)):
```python
import pennylane as qml
assert qml.__version__ == "0.44.0", (
    f"Phase 12 requires PennyLane 0.44.0 (set_shots transform / mixed-device "
    f"API); got {qml.__version__}. Do NOT run via ./qgan_env (0.43.0)."
)
```

**Trained-params load path** — RESEARCH Code Example 1, anchored at `REPO`. The trained state is a single 75-tensor; `default.qubit shots=None backprop` device is constructed by `QuantumGenerator.__init__` (`quantum.py:64,73-78`) but Phase 12 does NOT use `g.qnode` (Pitfall 2):
```python
from revision.core.models.quantum import QuantumGenerator
g = QuantumGenerator(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS,
                     window_length=WINDOW_LENGTH)
ck = torch.load(REPO / "results/transform_ablation/runs"
                / pipeline / str(seed) / "checkpoint.pt",
                map_location="cpu", weights_only=False)
g.params_pqc.data = ck["params_pqc"]   # 75-element trained tensor
g.eval()
```

**Generation contract (Pitfall 3 — the `*0.1` is load-bearing)** — copy `generate_samples` body **verbatim** from `run_ablation.py:179-208`; the ONLY change is the call site uses the alternate `qnode` instead of `generator(noise)`:
```python
# run_ablation.py:195-208 — copy verbatim, swap call site to qnode
rng = np.random.default_rng(seed)
out_parts: list[np.ndarray] = []
remaining = n
with torch.no_grad():
    while remaining > 0:
        bs = min(BATCH_SIZE, remaining)
        noise = torch.tensor(
            rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, bs)),
            dtype=torch.float32,
        )
        out = generator(noise).to(torch.float64) * 0.1   # *0.1 LOAD-BEARING
        out_parts.append(out.cpu().numpy())
        remaining -= bs
samples = np.concatenate(out_parts, axis=0)[:n]
```
The `(window_length, batch) → .T` transpose is in `quantum.py:194-199` (`forward`); replicate it when stacking the noisy QNode's tuple of 10 expvals.

**Noisy / finite-shot QNode construction** — NEW code (~15 lines, the only genuinely new logic for SENS-01/02), RESEARCH Code Examples 2 & 3. Build the alternate QNode in the driver — never mutate `g.qnode` or edit `core/` (Anti-Pattern, D-10-13):
```python
# SENS-01 finite-shot (RESEARCH Ex.2) — qml.set_shots transform, NOT shots= kwarg
def make_shot_qnode(g, shots: int | None):
    dev = qml.device("default.qubit", wires=NUM_QUBITS)   # NO shots= kwarg (Pitfall 1)
    qn = qml.QNode(g.generator_circuit, dev, interface="torch",
                   diff_method=None)                       # NOT backprop (Pitfall 2)
    if shots is not None:
        qn = qml.set_shots(qn, shots=shots)                # 0.44 API
    return qn
```
For SENS-02, copy the `generator_circuit` body (`quantum.py:122-171`) into the driver as `noisy_generator_circuit`, inserting `qml.DepolarizingChannel(p, wires=q)` / `qml.AmplitudeDamping(gamma, wires=q)` **after each entangling block** (per-layer, RESEARCH Assumption A1 default) on a `qml.device("default.mixed", wires=NUM_QUBITS)`. Document this copy as a deliberate noise-study duplication (does NOT violate D-10-13 — copy lives in `run_sensitivity.py`, not `core/`).

**OD reconstruction (Pitfall 4 — `seed*7919+1` load-bearing)** — copy `reconstruct_od` Pipeline-A and Pipeline-B branches **verbatim** from `run_utility.py:144-185`. Note the `od[:, :10]` truncation when `inverse_logreturns` returns length-11 (`run_utility.py:179-181`):
```python
# run_utility.py:166-181 — Pipeline B branch, verbatim
rng = np.random.default_rng(seed * 7919 + 1)   # load-bearing — do NOT refactor
od_start_per_window = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
...
od = od_full.cpu().numpy()
if od.shape[1] == 11:
    od = od[:, :10]
```

**Fidelity recompute** — reuse `revision.core.eval.full_metric_suite` UNCHANGED (`eval.py:143-163`); driver only wraps output with `scale`/`shots`/`noise_*` dims:
```python
from revision.core.eval import full_metric_suite   # returns flat dict:
# {emd, mean_real, mean_fake, std_real, std_fake, skew_*, kurt_*, jsd}
```
Dual-scale per EVAL-05: call once on OD-scale `(real_od, fake_od)` and once on the transformed/log-return scale; tag each row with `scale`.

**Idempotent per-cell run dir + 5-file bundle** — copy from `run_baselines.py:456-522`:
```python
run_dir = args.out_root / "runs" / args.condition / args.pipeline / str(args.seed)
if run_dir.exists():
    shutil.rmtree(run_dir)                  # idempotent — no stale partial bundle
run_dir.mkdir(parents=True, exist_ok=True)
...
(run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
np.save(run_dir / "samples.npy", samples)
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
```
**`{analytic}` sub-pattern (no regeneration, mirrors D-11-08):** for `condition=analytic` do NOT re-run — read frozen `transform_ablation/runs/<pipeline>/<seed>/samples.npy`. Only `shots ∈ {8192,1024}` and the 8 noise cells get fresh forward passes.

**CLI surface** — pattern after `run_baselines.py:430-454`: `--pipeline {A|B} --seed N --condition <cell> [--out-root results/sensitivity] [--csv-path ./data.csv]`, one `(pipeline, seed, condition)` cell per invocation.

---

### `run_sensitivity_sweep.sh` (sweep orchestration)

**Analog:** `run_baselines_sweep.sh` — copy near-verbatim; only the worklist (MODELS→CONDITIONS), `is_complete` artifact set, `total_count`, and the `python -m` target change.

**Idempotent per-cell skip (`is_complete`)** — `run_baselines_sweep.sh:174-184`:
```bash
is_complete() {
  local c="$1" p="$2" s="$3"
  local d="${OUT_ROOT}/runs/${c}/${p}/${s}"
  [[ -s "${d}/config.yaml" && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" ]]
}
```
The CLI driver overwrites the run dir cleanly, so partial dirs are safe to retry (`run_baselines_sweep.sh:20-24`).

**Atomic `sweep_status.json` (tmp-file + `os.rename` + `flock`)** — copy `update_status` verbatim from `run_baselines_sweep.sh:199-282`. Key load-bearing fragment:
```bash
update_status() {
  ...
  ( flock -x 9
    python3 - ... "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile
...
dirpath = os.path.dirname(status_file) or "."
fd, tmp = tempfile.mkstemp(prefix=".sweep_status.", suffix=".json", dir=dirpath)
with os.fdopen(fd, "w") as fh:
    json.dump(doc, fh, indent=2); fh.flush(); os.fsync(fh.fileno())
os.rename(tmp, status_file)            # POSIX-atomic rename
PY
  ) 9>"${LOCK_FILE}"
}
```
Status schema reference: `run_baselines_sweep.sh:45-60` — `{started_at, parallel, runs:[{...status, wall_seconds, return_code, skipped_already_done}], all_complete, completed_count, total_count}`.

**`--parallel 2` xargs guardrail (NEVER multiprocessing.Pool — 09.1 Pitfall 4)** — copy the guardrail `run_baselines_sweep.sh:146-156` and the xargs dispatch `run_baselines_sweep.sh:400-419`:
```bash
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]] || [[ "$PARALLEL" -gt 2 ]]; then
  echo "ERROR: --parallel must be 1 or 2 (got: '${PARALLEL}')." >&2
  exit 3
fi
...
# xargs -P 2 -L 1: at most 2 concurrent OS processes, one cell/invocation.
# NEVER replace with multiprocessing.Pool (Pitfall 5 / 09.1 Pitfall 4).
< "$WORKLIST" xargs -P 2 -L 1 bash -c 'run_one "$0" "$1" "$2"'
```

**Interpreter selection — DEVIATE from analog (Pitfall 5 / Open Q1).** The analog `run_baselines_sweep.sh:97-107` hard-prefers `./qgan_env/bin/python` (which is PennyLane 0.43.0). The Phase 12 sweep MUST NOT prefer the venv. Planner decision: select an explicit 0.44.0 interpreter and rely on the driver's `assert qml.__version__ == "0.44.0"` to fail loud. Document this deviation in the sweep header.

**Resilient `run_one` (failure does not abort sweep)** — copy `run_baselines_sweep.sh:290-335` (`set +e` around the python call, capture rc, mark `failed`, continue; per-cell `_stdout.log`/`_stderr.log`).

---

### `run_multiseed_rollup.py` (SENS-03 aggregator, single invocation)

**Analog:** `run_utility.py` (pure-consumer shape + repo-root resolver `:38-58`) + RESEARCH Code Example 4 (the groupby — ~30 lines, the only new logic).

**Repo-root resolver + no model/device/torch** — reuse the same `_find_repo_root()` block as `run_sensitivity.py`. This driver imports NO torch, NO pennylane, NO `core` model code (Pattern 2: "no device, no model, no torch").

**Cross-artifact data_hash assertion (D-10-15 — hard gate)** — RESEARCH Code Example 4, asserts mutual equality across the five frozen headline JSONs; do NOT re-derive from `transform_ablation` (Anti-Pattern):
```python
HEADLINE = ["baseline_comparison.json", "tstr.json",
            "predictive_discriminative.json", "augmentation.json",
            "fidelity_dualscale.json"]
docs = {f: json.load(open(RESULTS / f)) for f in HEADLINE}
hashes = {f: d["data_hash"] for f, d in docs.items()}
assert len(set(hashes.values())) == 1, f"data_hash mismatch: {hashes}"
canonical_hash = next(iter(hashes.values()))   # expect 91e447d4624e25b3
```

**Long-form groupby → mean ± std** — RESEARCH Code Example 4. Verified row schema (live): every headline JSON has `rows[]` of `{model_kind, pipeline, seed, metric_name, scale, value[, injection_ratio]}`; row counts baseline_comparison 1710, fidelity_dualscale 3360, tstr 144, predictive_discriminative 120, augmentation 180. Use stdlib `statistics.fmean/stdev` (RESEARCH Open Q3 — zero new dependency, audit-clean):
```python
from collections import defaultdict
import statistics
buckets = defaultdict(list)
for f, d in docs.items():
    for r in d["rows"]:
        key = (f, r["model_kind"], r["pipeline"], r["metric_name"],
               r["scale"], r.get("injection_ratio"))
        buckets[key].append((r["seed"], r["value"]))
rollup = [{
    "source": src, "model_kind": mk, "pipeline": pl, "metric_name": metric,
    "scale": scale, "injection_ratio": ratio,
    "mean": statistics.fmean(vals),
    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    "n": len(vals), "seeds": sorted({s for s, _ in pairs}),
} for (src, mk, pl, metric, scale, ratio), pairs in buckets.items()
  for vals in [[v for _, v in pairs]]]
```

---

### Output JSON artifacts (`shot_noise_sensitivity.json`, `noise_model_sensitivity.json`, `multiseed_summary.json`)

**Analog:** `results/baseline_comparison.json` — established long-form contract (verified live):
- Top-level keys: `schema, model_kinds, pipelines, seeds, data_hash, data_hash_verification, ...`
- `rows[]` element: `{model_kind, pipeline, seed, metric_name, scale, value}` (example row 0: `{'model_kind':'quantum','pipeline':'A','seed':42,'metric_name':'emd','scale':'OD','value':1.052...}`)
- `data_hash` = `91e447d4624e25b3` across all five headline JSONs.

**Extend, do not replace** (D-12 Claude's discretion + `code_context` line 89): SENS-01/02 add a `condition` + `shots` | `noise_model` + `noise_level` dimension to each row; keep `{model_kind:"quantum", pipeline, seed, metric_name, scale, value}` intact. SENS-03 emits `rollup[]` of `{source, model_kind, pipeline, metric_name, scale, injection_ratio, mean, std, n, seeds}` plus a provenance header `{schema, data_hash, consumed_artifacts:{<file>:<hash>}, seed_set:[42..46]}`.

---

## Shared Patterns

### Repo-root resolver (cwd-independence, Pitfall 6)
**Source:** `run_utility.py:38-58` (`_find_repo_root` + `sys.path.insert`)
**Apply to:** both new drivers (`run_sensitivity.py`, `run_multiseed_rollup.py`) — anchor every artifact path at `REPO`.

### Atomic status + flock + idempotent skip
**Source:** `run_baselines_sweep.sh:174-184` (`is_complete`), `:199-282` (`update_status`), `:290-335` (`run_one`)
**Apply to:** `run_sensitivity_sweep.sh` — copy verbatim, retarget worklist/artifact set.

### `--parallel 1|2` guardrail + xargs OS-process parallelism (NEVER Pool)
**Source:** `run_baselines_sweep.sh:146-156` (guardrail), `:400-419` (xargs `-P 2 -L 1`), `:34-43` (the "no multiprocessing.Pool" rationale, 09.1 Pitfall 4)
**Apply to:** `run_sensitivity_sweep.sh`.

### HPO constants from `revision.core`, never literals
**Source:** `run_baselines.py:65-77`
**Apply to:** `run_sensitivity.py` — import `BATCH_SIZE, NOISE_LOW/HIGH, NUM_QUBITS, NUM_LAYERS, WINDOW_LENGTH` from `revision.core`.

### `*0.1` generation contract + `default_rng(seed)`
**Source:** `run_ablation.py:195-208` (verbatim) + `core/models/quantum.py:194-199` (the `.T` transpose for batched expvals)
**Apply to:** `run_sensitivity.py` sample regeneration (Pitfall 3).

### Pipeline-B `seed*7919+1` od_start draw + `od[:, :10]` truncation
**Source:** `run_utility.py:144-185` (`reconstruct_od`, verbatim)
**Apply to:** `run_sensitivity.py` OD-scale reconstruction (Pitfall 4).

### `core/` untouched invariant
**Source:** D-10-13 (RESEARCH Validation Note) — `git diff --stat core/` must be empty.
**Apply to:** all of Phase 12 — alternate/noisy QNode and circuit-body copy live in `run_sensitivity.py`, never `core/`.

---

## No Analog Found

| Pattern | Why | Mitigation |
|---------|-----|------------|
| `qml.set_shots(qnode, shots=N)` finite-shot QNode | New PennyLane 0.44 API; project code (`quantum.py:64`) still uses the device-bound `shots=None` path | RESEARCH Code Example 2 (verified live) |
| `default.mixed` + `DepolarizingChannel`/`AmplitudeDamping` noisy circuit | No noise-channel code exists anywhere in repo (inference-only is new to Phase 12) | RESEARCH Code Example 3 + circuit body copied from `quantum.py:122-171` |
| Cross-artifact data_hash equality + long-form groupby | No prior aggregation-only driver | RESEARCH Code Example 4 (schema verified live) |
| PennyLane version assertion at startup | No precedent; analog sweep hard-prefers the wrong (0.43.0) venv | RESEARCH Open Q1 recommendation (a) — ~3 line assert |

These four are the only genuinely new code in Phase 12 (~50 lines total). Everything else is copied verbatim from the analogs above.

## Metadata

**Analog search scope:** `run_*.py`, `run_*.sh`, `core/`, `results/*.json`
**Files scanned:** `run_baselines.py`, `run_baselines_sweep.sh`, `run_ablation.py`, `run_utility.py`, `core/models/quantum.py`, `core/eval.py`, `core/preprocessing.py` (via RESEARCH), `baseline_comparison.json`
**Pattern extraction date:** 2026-05-18
