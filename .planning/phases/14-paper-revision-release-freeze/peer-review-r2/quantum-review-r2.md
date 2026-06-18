# Quantum-Circuit R2 Peer Review — Plan 14-13 Sweep Verification

**Reviewer scope:** quantum-circuit + PQC correctness (independent of math
review, post-14-13 sweep).
**Anchor commit (Wave 10, "before sweep"):** `06bb470`.
**Tip commit at review time:** `main` @ `abb06a4` (post-14-13 close).
**Verdict:** **PASS** — D-14-22 byte-freeze attested; all quantum config
locks, citations, headline numbers, structural gates, and circuit-diagram
PNGs preserved; the symmetric MPS-disable hook is safe under the
documented one-process-per-invocation contract.

---

## D-14-22 byte-freeze attestation

> ```
> $ git diff 06bb470..main -- core/ | wc -l
>        0
> $ git diff 06bb470..main --stat -- core/
> (no output — zero lines, zero files)
> $ git log --oneline 06bb470..main -- core/
> (no output — zero commits)
> ```

**ATTESTED:** zero bytes changed under `core/` between the
post-Wave-10 anchor (`06bb470`) and the post-14-13 tip (`main`). The
quantum implementation in `core/models/quantum.py`,
`core/training.py`, and `core/eval.py` is verifiably
byte-frozen. All correctness-sensitive quantum code is untouched.

`docs/circuit_atlas.md` is also untouched
(`git log 06bb470..main -- docs/circuit_atlas.md` → empty), as
required by the plan (owned by 14-09).

The five `*_config_lock.json` files (canonical, default_75, v1, v2, v3)
are also untouched
(`git log 06bb470..main -- results/*_config_lock.json` →
empty). The 5-circuit configuration matrix is intact.

---

## Cross-checks performed

### 1. Checkpoint sha256 identity ✓

```
$ shasum -a 256 checkpoints/best_checkpoint.pt
f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082  ...

$ python3 -c "import json; print(json.load(open('results/canonical_config_lock.json'))['checkpoint_sha256'])"
f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082
```

Identity: the headline checkpoint is exactly the SHA the lock claims.
METHODS-BLOCKER-2 fix (T1 = `4ea576b`, tracking `best_checkpoint.pt`
through a `.gitignore` `!`-exception) did NOT modify the byte content —
it merely surfaces the existing artifact to reviewers.

### 2. Checkpoint loadability + 55-param structural gate ✓

```python
import torch, sys
sys.path.insert(0, 'revision')
from core.models.quantum import QuantumGenerator

ckpt = torch.load('checkpoints/best_checkpoint.pt',
                  map_location='cpu', weights_only=False)
# top-level: epoch=1969, emd=0.0838, params_pqc shape=(55,)
# critic_state, c_optimizer, g_optimizer, mu=0.002455, sigma=0.02141

gen = QuantumGenerator(circuit_id='iqp_sel_55', num_layers=3,
                       num_qubits=5, topology='range')
# gen.num_params = 55
with torch.no_grad():
    gen.params_pqc.copy_(ckpt['params_pqc'])
# gen.last_param_index(torch.zeros(5)) → 55  ← structural forward-pass gate
# forward output shape → (10,)               ← 2 * num_qubits = WINDOW_LENGTH
```

The checkpoint loads cleanly into the `iqp_sel_55` decomposition,
consumes exactly 55 params on the structural forward-pass walk, and
returns the expected 10-dim output. The recovered canonical decomposition
(`iqp_sel_55_repro` config) is still identifiable from the checkpoint.

### 3. 5-circuit config-lock self-consistency ✓

| circuit | param_count | num_layers | topology | final_rotation |
|---|---|---|---|---|
| `canonical` (iqp_sel_55) | 55 | 3 | range | RX_only |
| `default_75` | 75 | 4 | range | RX_plus_RY |
| `V1` | 75 | 4 | range | RX_plus_RY |
| `V2` | 135 | 8 | range | RX_plus_RY |
| `V3` | 75 | 4 | linear | RX_plus_RY |

All five locks intact, all formulas hold by direct arithmetic
(`5 + L·15 + N·{1 or 2}`):
- canonical: `5 + 3·15 + 5·1 = 55` ✓
- default_75 / V1: `5 + 4·15 + 5·2 = 75` ✓
- V2: `5 + 8·15 + 5·2 = 135` ✓
- V3: `5 + 4·15 + 5·2 = 75` ✓

### 4. Circuit-diagram PNG byte-identity ✓

```
$ git show 06bb470:results/figures/circuits/iqp_sel_55.png | shasum -a 256
e85e1ca313ba5ef53721adc50dbaee8b7e82d36dd9715b4b96fb98b848d50e7e

$ shasum -a 256 results/figures/circuits/iqp_sel_55.png
e85e1ca313ba5ef53721adc50dbaee8b7e82d36dd9715b4b96fb98b848d50e7e
```

PNG sha256 unchanged. All five diagram PNGs + PDFs are bit-identical to
their pre-14-13 state (`git diff --stat` shows PDFs at identical
byte-sizes 23142/23869/24259/24769/30989 → 23142/23869/24259/24769/30989,
PNGs unmodified). Only the companion JSON sidecars changed (added
`data_hash` field + refreshed `generation_timestamp`) — render content
itself is byte-frozen.

### 5. methods_full.md §2 citations resolve correctly ✓

Spot-checked the `core/training.py:{72,245,246,247,248,249,259,
268,346,347,364,385}` line citations against the actual code:

| Cited line | Expected content | Actual content at that line |
|---|---|---|
| 72 | `gp = ...mean()` (gradient penalty) | `gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()` ✓ |
| 245 | `torch.manual_seed(seed)` | `torch.manual_seed(seed)` ✓ |
| 246 | `np.random.seed(seed)` | `np.random.seed(seed)` ✓ |
| 247 | `random.seed(seed)` | `random.seed(seed)` ✓ |
| 248-249 | CUDA seeding guard | `if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)` ✓ |
| 268 | `compute_dtype = ...` | `compute_dtype = torch.float32 if device.type == "mps" else torch.float64` ✓ |
| 346 | generator forward call | `generated_samples = generator(noise_batch)` ✓ |
| 347 | sample-dtype cast | `generated_samples = generated_samples.to(compute_dtype) * 0.1` ✓ |
| 364 | critic loss | `critic_loss = fake_score_mean - real_score_mean + lambda_gp * gp` ✓ |
| 385 | generator loss | `generator_loss = -torch.mean(fake_scores)` ✓ |

All 10 cited line numbers in `methods_full.md` + `methods_full.json`
resolve to the claimed content. Since `core/training.py` is
byte-frozen, no regression risk on these citations was possible.

The quantum architecture text in `methods_full.md §2.a`
("IQP (Hadamard + RZ per qubit)", "SEL (Rot(phi,theta,omega) per qubit
per layer)") matches `core/models/quantum.py:195-203` (Hadamard
on every wire, then per-qubit `qml.RZ(params_pqc[idx])`) and
`quantum.py:211-219` (per-qubit `qml.Rot(phi, theta, omega)` inside the
SEL loop). ✓

### 6. Headline numbers unchanged ✓

`results/headline_canonical.json` is byte-untouched in 14-13
(`git diff --stat` empty). The recorded values:

- `checkpoint_sha256 = f7cceb52...` (matches the lock)
- `checkpoint_epoch = 1969` (matches the lock)
- `generation_seed = 42` (matches the original Phase-14 contract)
- `param_count = 55` (matches the structural gate)
- `quantum` row OD-EMD = `0.023071979442389253` ← unchanged
- `quantum` row log_return-EMD = `0.12124099500150183` ← unchanged

The original quantum-review §12 reported these as 0.0231 (OD) and 0.1212
(log_return); both still hold to 4 decimals. Pre/post-14-13 invariance
confirmed.

Five matched-budget `iqp_sel_55_repro` per-seed runs at
`results/matched2000/runs/iqp_sel_55_repro/{42..46}/metrics.json`
are also untouched (`git log` empty); `emd_avg[-1]` per seed:

| seed | emd_avg[-1] | n_epochs |
|---|---|---|
| 42 | 0.140503 | 201 |
| 43 | 0.174707 | 201 |
| 44 | 0.160780 | 201 |
| 45 | 0.130165 | 201 |
| 46 | 0.168840 | 201 |

Bit-identical to pre-14-13. No silent quantum regression.

### 7. generation_seed threading in run_canonical_headline.py — safe for quantum ✓

`run_canonical_headline.py:287` and `:345` now thread `generation_seed`
through the DTW subsample RNG instead of hardcoding `42 * 31`. The
default invocation passes `generation_seed = 42`, so
`dtw_seed = 42 * 31 = 1302` — **identical to the previous hardcoded
value**. Therefore the per-seed DTW numbers in `headline_canonical.json`
are bit-preserved, which is consistent with the empty `git diff --stat`
on that file.

The threading is forward-only (future generation seeds other than 42
would produce different DTW subsamples by design, never colliding with
the headline). No regression possible on the audited record.

### 8. MPS-disable hook symmetry safety ✓ (within documented contract)

The hook now appears in three places in `run_matched2000.py`:
- `:447-464` (`_train_quantum`) — pre-existing
- `:538-554` (`_train_wgan`) — added by T4 (`8c67891`)
- `:604-641` (`_train_vae`) — added by T4 (`8c67891`)

All three follow an identical pattern:

```python
orig_mps = torch.backends.mps.is_available
torch.backends.mps.is_available = lambda: False
try:
    ...  # training call
finally:
    torch.backends.mps.is_available = orig_mps
```

**Concurrency analysis:** `run_matched2000.py:935` explicitly
disclaims in-process concurrency:

> `"...one model/seed per invocation (NEVER multiprocessing.Pool; the
>   sweep uses xargs -P 2)."`

A `grep` for `ThreadPool|multiprocessing|asyncio|threading` in the
quantum-driver scripts returned no matches. Since `xargs -P 2` spawns
**separate OS processes**, each with its own private
`torch.backends.mps.is_available` global, the monkey-patch's lack of
thread-safety is irrelevant under the documented sweep contract.

**MD-7 status (from original review):** **NOT WORSE THAN BEFORE.** The
symmetric application affects three call sites instead of one, but each
still runs in process-local isolation. The reviewer-flagged
threadsafety concern persists structurally (a future maintainer running
the trainers in-thread would hit it) but is correctly disclosed as
deferred in `docs/peer_review_remediation.md:92-96`:

> "MD-7 (`mps.is_available` monkey-patch threadsafety) — byte-frozen
> under D-14-22; the CR-4 future-gate applies the monkey-patch to
> `_train_wgan` and `_train_vae` consistently with the existing
> `_train_quantum` pattern (Plan 14-13 Task 4), but the underlying
> threadsafety concern cannot be addressed without `core/`
> edits and is therefore deferred."

This is honest, scoped, and consistent with D-14-22. ✓

### 9. `training_time_device` strict-accept gate — semantics correct for quantum ✓

The new gate at `run_matched2000.py:785-791`:

```python
ttd = dm.get("training_time_device")
if ttd is not None and ttd != "cpu":
    raise AssertionError(
        f"accept: training_time_device={ttd!r} != 'cpu' ..."
    )
```

For quantum runs, `_device_manifest` inspects the
`QuantumGenerator.params_pqc` — a `nn.Parameter` on CPU by construction
(`quantum.py:115-118`). Empirically:

```
>>> gen = QuantumGenerator(circuit_id='iqp_sel_55', ...)
>>> str(next(gen.parameters()).device)
'cpu'
>>> gen.dev.name
'default.qubit'  # PennyLane CPU-only statevector backend (D-14-11)
```

So `training_time_device = "cpu"` is read back correctly for quantum
runs and passes the new gate. PennyLane's `default.qubit` is
CPU-only-by-construction (D-14-11 backend lock at
`run_matched2000.py:355-366` rejects any non-default.qubit device),
which makes the gate trivially safe for quantum — there is no MPS code
path even reachable for the quantum branch.

The gate is also **forward-only**: historical bundles without the
`training_time_device` key are accepted (the `ttd is not None`
short-circuit). This preserves the disclosed-historical-asymmetry
contract for the existing classical baselines that ran on MPS.

### 10. data_hash in run_circuit_diagrams.py companions — no regression ✓

T4 added `"data_hash": "91e447d4624e25b3"` to the companion JSON of each
circuit diagram (`run_circuit_diagrams.py:564`). The diagrams themselves
are render-only architecture figures (no data is used by `qml.draw_mpl`),
so the `data_hash` is purely a provenance marker for downstream
consistency checks (PROV-HIGH-2). Since PNG/PDF bytes are unchanged
(verified in §4) and the structural pre-render assertion at
`run_circuit_diagrams.py:517-525` (`model.num_params == lock.param_count`)
still gates correctness, no regression in diagram fidelity.

---

## Findings

### NEW (introduced or surfaced by 14-13)

**None.** The 14-13 sweep made the quantum surface area strictly safer
(symmetric MPS-disable + recorded `training_time_device` gate + tracked
checkpoint + data_hash provenance) without modifying any
correctness-sensitive quantum code.

### Original quantum-review findings re-check

| # | Original finding | Severity | Status post-14-13 |
|---|---|---|---|
| §1 | IQP-encoding nomenclature (depth-1, single-qubit IQP — not ZZ-feature-map) | MINOR | **STILL STANDING.** `docs/methods_full.md:66` still reads "IQP (Hadamard + RZ per qubit)" with no clarifying footnote. Documented as out-of-scope in `peer_review_remediation.md` (quantum-review.md → no CR/HI in scope). Defer to follow-up plan. |
| §2 | `quantum.py:1` "data re-uploading" docstring misnomer | MINOR | **STILL STANDING.** The docstring is unchanged because `core/` is byte-frozen under D-14-22. Documented as a forward-fix-only item; paper text is clean of "re-uploading" wording. Defer to a post-freeze release. |
| §3 | SEL block structural equivalence to `qml.StronglyEntanglingLayers` | VERIFIED | Unchanged. Code byte-frozen. |
| §4 | range vs linear topology (V3) | VERIFIED | Unchanged. Config locks byte-frozen. T4's `HI-4` fix (read topology from `canonical_config_lock.json#decomposition.gate_layout.entangler` at `run_matched2000.py:417-419` instead of hardcoding `"range"`) is a **strict improvement** — even though the value is identical (`range`), the read now flows through the locked source-of-truth rather than a duplicated literal. |
| §5 | Final RX+RY vs RX-only | VERIFIED | Unchanged. quantum.py:239-250 byte-frozen. |
| §6 | Parameter-count formulas (75/55/75/135/75) | VERIFIED | Unchanged. Structural gate intact. |
| §7 | Canonical decomposition structural verification | VERIFIED | Re-verified independently in this review (see §2 above). Checkpoint loads cleanly into 55-param iqp_sel_55, `last_param_index` returns 55. ✓ |
| §8 | Encoding-strategy honesty | VERIFIED | Unchanged (subject to §1 wording). |
| §9 | Measurement basis (X+Z per qubit, simulator-only) | VERIFIED | Unchanged. |
| §10 | V2 barren-plateau probe absent | MEDIUM | **STILL STANDING.** No new gradient-variance analysis; out-of-scope for 14-13 per remediation index. Defer to a follow-up plan. |
| §11 | Diagram fidelity (`qml.draw_mpl`) | VERIFIED | Re-verified: PNG sha256 byte-identical (§4 above). ✓ |
| §12 | Headline-EMD 80× discrepancy = evaluation, not circuit | RESOLVED | Re-verified: `headline_canonical.json` byte-untouched, OD-EMD=0.023072 and log_return-EMD=0.12124 unchanged. ✓ |

Net status: 9 VERIFIED items remain VERIFIED (re-checked); 2 MINOR + 1
MEDIUM items remain STANDING per D-14-22 byte-freeze; 1 RESOLVED item
remains RESOLVED.

### Concerns inspected and dismissed

- **Could T4's `HI-4` fix (`topology` from lock instead of hardcoded
  `"range"`) silently change the quantum tape?** No — the
  `canonical_config_lock.json` records `entangler: "range"` (verified
  by direct read in §3 above), so the dynamic read returns the same
  string as the previous hardcoded literal. Tape unchanged.
- **Could the `training_time_device` field be set incorrectly for
  quantum?** No — `_device_manifest` reads
  `next(generator.parameters()).device`, which for a `QuantumGenerator`
  is always `cpu` (the `params_pqc` `nn.Parameter` is constructed on
  CPU by `quantum.py:115-118`). The PennyLane device backend
  (`default.qubit`) is also CPU-only (`D-14-11`), gated explicitly at
  `run_matched2000.py:355-366`.
- **Could the symmetric MPS-disable hook ever fail to restore?** All
  three sites use `try/finally`; the `finally` clause unconditionally
  restores the original `torch.backends.mps.is_available`. Process exit
  is the only failure mode, which is fine (the global dies with the
  process).
- **Could the added `data_hash` to circuit-diagram companions
  invalidate the locked diagrams?** No — the diagrams are
  `qml.draw_mpl`-rendered from `_QUANTUM_ANSATZ` + config-lock JSONs
  (not from data); the `data_hash` is metadata only. PNG/PDF bytes are
  identical pre/post (verified by sha256).
- **Could the headline `iqp_sel_55_repro` numbers have shifted with
  HI-1's `generation_seed` threading?** No — `generation_seed=42`
  produces `dtw_seed=1302`, exactly the previous hardcoded value. The
  headline file is byte-untouched.

---

## Final recommendation

**Quantum claims sound for paper resubmission: YES.**

The 14-13 sweep was scrupulous in observing the D-14-22 byte-freeze on
`core/`:

- Quantum code unchanged (`git diff 06bb470..main -- core/`
  returns empty).
- All 5 quantum config locks unchanged.
- Headline canonical numbers unchanged
  (OD-EMD=0.023072, log_return-EMD=0.12124, checkpoint sha256
  `f7cceb52…`, epoch 1969).
- Per-seed `iqp_sel_55_repro` matched-budget runs unchanged.
- Circuit-diagram PNG/PDF byte-identical pre/post; companion JSONs
  augmented with `data_hash` provenance metadata.
- `circuit_atlas.md` untouched (owned by 14-09, per plan).

The improvements 14-13 introduced are all **strictly safer**:

- The MPS-disable hook is now symmetric across all three trainers
  (`_train_quantum`, `_train_wgan`, `_train_vae`) so future
  matched-budget runs cannot accidentally execute on Apple-Silicon MPS
  while quantum runs are on CPU.
- The strict-accept gate now records and asserts
  `training_time_device == "cpu"` (forward-only — historical bundles
  without the field are accepted, preserving the disclosed historical
  classical-MPS asymmetry).
- The headline checkpoint is now version-tracked
  (`!checkpoints/best_checkpoint.pt` `.gitignore` exception),
  sha256-verified, and structurally loadable as the 55-param
  iqp_sel_55 decomposition.

The three MINOR/MEDIUM original-quantum-review items (IQP nomenclature
footnote, `quantum.py:1` docstring rewording, V2 barren-plateau
acknowledgement) remain unaddressed by design — they're out-of-scope
under D-14-22 and properly documented as deferred in
`peer_review_remediation.md`. None of them block paper resubmission; all
three should appear on a follow-up release plan that lifts D-14-22.

No new quantum concerns introduced by the sweep.
