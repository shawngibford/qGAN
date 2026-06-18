---
phase: 13-architecture-introspection
reviewed: 2026-05-19T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - core/models/quantum.py
  - core/training.py
  - run_ansatz.py
  - run_ansatz_comparison.py
  - run_ansatz_sweep.sh
  - run_introspect.py
  - run_introspect_figures.py
  - pytest.ini
  - tests/__init__.py
  - tests/conftest.py
  - tests/test_ansatz_json_schema.py
  - tests/test_ansatz_variants.py
  - tests/test_cr01_spectral_grad.py
  - tests/test_cr02_es_restore.py
  - tests/test_entropy_purity.py
  - tests/test_introspect_callback.py
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-05-19
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed the Phase-13 architecture-introspection deliverables: the topology-selectable
`QuantumGenerator`, the CR-01/CR-02 fixes in `training.py`, the ansatz sweep + comparison
drivers, the introspection training/figure drivers, and the new regression suite.

**Byte-unchanged invariant (core reproducibility) — verdict: PRESERVED for the forward
path, with one caveat.** I diffed `core/models/quantum.py` and
`core/training.py` against `b7c84d3`:

- `quantum.py`: the `range` CNOT block is wrapped verbatim inside
  `if self.topology == "range"` with `topology="range"` as the constructor default.
  Init RNG draw (`torch.randn(num_params) * 0.5`), param count, QNode construction order,
  and Steps 1–6 are unchanged on the default path. `test_default_forward_byte_unchanged`
  pins this with an `atol=1e-12` reference vector. The new `_introspect_qnode` is built in
  `__init__` *after* the `params_pqc` parameter is created, so it does not perturb the
  parameter RNG state. Default forward/training path is byte-identical.
- `training.py`: `_load_checkpoint` and `_spectral_psd_loss` were rewritten, but both sit
  behind runtime guards (`early_stopper is not None`, `spectral_loss_weight > 0.0`) that
  are off at default values, so the default training trace is unchanged. See WR-01 for the
  one residual reproducibility caveat (`_load_checkpoint` no longer reproduces the prior
  early-stopped numeric path bit-for-bit — acceptable per the documented rationale, flagged
  for the record).

The two BLOCKERs below are correctness/metadata defects in the Phase-13 driver layer, not
in `core/`.

## Critical Issues

### CR-01: `run_ansatz.py` silently ignores `--epochs`; trains 1000 epochs but records the requested count in config.yaml

**status:** fixed (commit f2671d6 — `num_epochs=int(epochs)` threaded; protocol-notes text now interpolates `{epochs}`; default stays 1000)
**File:** `run_ansatz.py:214-264`, `run_ansatz.py:344-356`
**Issue:**
`main()` parses `--epochs` (default 1000), passes it as `_train_wgan(args.variant, bundle,
args.epochs, args.seed)`, and writes `"epochs": int(args.epochs)` into `config.yaml`
(line 356). But `_train_wgan` receives `epochs` as a parameter and **never uses it** — the
`train_wgan_gp` call hardcodes `num_epochs=1000` (line 257). Consequences:

1. `run_ansatz_sweep.sh` forwards `--epochs "$EPOCHS"` (line 312-317) and supports
   a `--epochs M` override; that override is silently dead — every sweep run trains exactly
   1000 epochs regardless.
2. If anyone runs `--epochs 500` (or the sweep with `--epochs 2000`), `config.yaml` records
   `epochs: 500` while the checkpoint/samples/metrics reflect a 1000-epoch run. This is a
   provenance corruption: the frozen artifact bundle's recorded epoch count does not match
   the training that produced it, defeating the D-10-15/D-13-01 reproducibility intent and
   the sweep's resume/audit guarantees.

`_train_wgan` is also documented as "1000 epochs" in `extra_cfg["train_protocol_notes"]`
(line 292) — hardcoded text that will likewise lie if `--epochs` is changed.

**Fix:**
```python
def _train_wgan(variant, bundle, epochs, seed):
    ...
    metrics = train_wgan_gp(
        generator,
        critic,
        bundle.dataloader,
        num_epochs=int(epochs),   # was: num_epochs=1000
        n_critic=int(N_CRITIC),
        ...
    )
    ...
    extra_cfg = {
        ...
        "train_protocol_notes": (
            f"ARCH-01 ansatz variant {variant}: QuantumGenerator("
            f"num_layers={depth}, topology={topology!r}) trained via "
            f"train_wgan_gp UNCHANGED, {epochs} epochs, early-stop OFF ..."
        ),
    }
```
If the 1000-epoch budget is an intentional Phase-13 lock (D-13 decision), then the inverse
fix is required instead: remove the `--epochs` argument from both `run_ansatz.py` and
`run_ansatz_sweep.sh`, and stop writing a variable `epochs` into `config.yaml` — do not
expose a knob that is silently ignored.

### CR-02: `run_ansatz.py` writes `checkpoint.pt`, but `run_ansatz_comparison.py` requires `inverse_kwargs.npz` keys that the saved scalar encoding breaks at read time

**status:** fixed (commit 046dfad — added FileNotFoundError + KeyError schema guard in `reconstruct_dualscale` before scoring; on-disk format unchanged)
**File:** `run_ansatz.py:189-201` and `run_ansatz_comparison.py:147-160`
**Issue:**
`_save_inverse_kwargs` stores scalar entries via `np.asarray(v)` → 0-D arrays
(`r_min`, `r_max`, `mu`, `sigma`). `reconstruct_dualscale` reads them back with
`float(inv["r_min"])`. `float()` on a 0-D NumPy array works, but `od_starts` is stored as a
float64 ND array and later consumed by `np.random.default_rng(...).choice(od_starts_pool,
size=r_norm.shape[0], replace=True)`. `np.load(..., allow_pickle=True)` returns
`od_starts` as a 1-D array (OK), but for **V1** the bundle is read from
`transform_ablation/runs/B/<seed>` (line 133) which was written by a *different* driver
(`run_baselines._save_inverse_kwargs`) in a prior phase. There is no schema/version check
that the V1 frozen `inverse_kwargs.npz` exposes the same key set (`r_min/r_max/mu/sigma/
od_starts`) and dtype that `reconstruct_dualscale` assumes. If the 09.1/10 `run_baselines`
layout used different key names (e.g. a packed dict, or `od_start` singular), the V1 branch
raises a bare `KeyError` deep inside the aggregator with no actionable message, *after*
V2/V3 have already been re-scored — and the comparison JSON is never written, silently
failing the whole ARCH-02 deliverable.

This is a cross-module contract assumption with no guard. The aggregator hard-asserts the
V1 path "by construction" (`QUANTUM_EQUIVALENCE_NOTE`) but never validates the on-disk
schema it depends on.

**Fix:** Validate the V1 bundle schema before scoring, failing loudly and early:
```python
def reconstruct_dualscale(variant, seed, ansatz_root):
    base = _run_base(variant, seed, ansatz_root)
    spath, ipath = base / "samples.npy", base / "inverse_kwargs.npz"
    if not spath.is_file() or not ipath.is_file():
        raise FileNotFoundError(
            f"{variant} seed={seed}: frozen bundle missing under {base} "
            f"(expected samples.npy + inverse_kwargs.npz)"
        )
    inv = np.load(ipath, allow_pickle=True)
    required = {"r_min", "r_max", "mu", "sigma", "od_starts"}
    missing = required - set(inv.files)
    if missing:
        raise KeyError(
            f"{variant} seed={seed}: inverse_kwargs.npz missing keys "
            f"{sorted(missing)} (found {sorted(inv.files)}) — V1 reuse "
            f"contract (D-13-01) broken; aborting before partial re-score"
        )
    ...
```

## Warnings

### WR-01: `_load_checkpoint` rewrite changes the early-stopped numeric path vs. pre-Phase-13 (reproducibility caveat)

**status:** fixed (commit c1e779d — documentation-only docstring caveat added to `_load_checkpoint`; NO behavior change, pin test green)
**File:** `core/training.py:163-195`
**Issue:**
The pre-`b7c84d3` `_load_checkpoint` did `model.params_pqc.data = checkpoint["params_pqc"]`
(no device/dtype recast) and did not iterate optimizer state. The new version recasts
params to the live device+dtype and pushes every optimizer-state tensor onto the live
device. On the CPU/float64 path the *values* are numerically identical (a no-op `.to`), so
the default headline runs (early-stop OFF, D-13-05) are unaffected. However, any *prior*
phase that ran with `early_stopper` set and then resumed/early-stopped now follows a
different code path than it did pre-Phase-13. The change is correct and well-justified
(CR-02 docstring), and Phase-13 headline runs do not use early stopping — but the
"`core/` byte-behavior-unchanged for Phases 8-12" invariant is technically only
preserved because early stopping is off by default. This is acceptable, but should be
recorded explicitly in the phase decision log so a future early-stopped reproduction is not
mistakenly expected to match a pre-Phase-13 trace.
**Fix:** No code change required. Document the early-stop-path behavior delta in the
Phase-13 decision log / STATE so downstream reproductions of early-stopped runs are not
compared bit-for-bit against pre-Phase-13 outputs.

### WR-02: `_spectral_psd_loss` has no length-match guard between fake and real

**status:** fixed (commit 9872e00 — ValueError raised on `fake_flat.numel() != real_flat.numel()`; on the spectral_loss_weight>0 path, OFF by default; pin test green)
**File:** `core/training.py:511-520`
**Issue:**
`psd_fake = torch.fft.rfft(fake_flat)` and `psd_real = torch.fft.rfft(real_flat)` produce
tensors of length `len//2 + 1`. If `fake_flat.numel() != real_flat.numel()`, the final
`torch.log(psd_fake) - torch.log(psd_real)` either broadcasts incorrectly (silent wrong
loss) or raises a shape error mid-training. At the only in-repo call site
(`train_wgan_gp` line 377-379) the lengths match by construction
(`real_log_returns_for_psd` returns `batch_size` windows == `gen_out` size), so this is
latent, not live — but the function is public-ish (imported directly by
`tests/test_cr01_spectral_grad.py`) and the test only feeds equal-length tensors, so the
mismatch case is untested and unguarded.
**Fix:**
```python
fake_flat = fake.reshape(-1)
real_flat = real.reshape(-1).detach().to(device=fake_flat.device, dtype=fake_flat.dtype)
n = min(fake_flat.numel(), real_flat.numel())
fake_flat, real_flat = fake_flat[:n], real_flat[:n]
```
or assert `fake_flat.numel() == real_flat.numel()` with a clear message.

### WR-03: monkey-patching `torch.backends.mps.is_available` is global and non-restoring in `run_ansatz.py`

**status:** fixed (commit 43f807e — added restoring `_force_cpu_for_quantum` context manager mirroring run_introspect; wraps only the train_wgan_gp call)
**File:** `run_ansatz.py:243`
**Issue:**
`torch.backends.mps.is_available = lambda: False` permanently replaces a global torch
function for the lifetime of the process and is never restored. `run_introspect.py`
correctly uses a context manager (`_force_cpu_for_quantum`) that restores the original in a
`finally`. `run_ansatz.py` does it unconditionally and globally at module-execution time
inside `_train_wgan`. Because `run_ansatz.py` is one-process-per-invocation this is safe
*today*, but it is a fragile pattern: if `_train_wgan` is ever imported and called from a
longer-lived process (e.g. a future aggregator or a test), MPS is silently disabled for the
entire interpreter with no way to detect or undo it. The two drivers should use the same
restoring guard.
**Fix:** Reuse the `_force_cpu_for_quantum` context manager pattern from
`run_introspect.py:165-190` (or import it) and wrap only the `train_wgan_gp` call, instead
of mutating the global at function-body top level.

### WR-04: snapshot closure mutates the live generator's device mid-training (re-entrancy / exception risk)

**status:** fixed (commit 2717491 — `orig_device` captured before mutation; `.to("cpu")` moved inside the `try` so `finally` device-restore always runs)
**File:** `run_introspect.py:114-161`
**Issue:**
`cb` does `gen_model = generator.to("cpu")` (in-place for nn.Module), regenerates, then
restores with `generator.to(orig_device)` in a `finally`. The `finally` covers the normal
case, but the device capture itself (`next(generator.parameters()).device`) is *outside*
the `try`, and `train_wgan_gp` wraps the callback in its own `try/except` that prints and
swallows (`training.py:430-431`). If `generator.to("cpu")` succeeds but the subsequent
`next(generator.parameters())` ordering ever changed, or an exception is raised before the
`try` is entered, the generator is left on CPU and the *next* training epoch does a
cross-device matmul that the outer `except` swallows as a one-line warning — training
silently continues producing garbage rather than failing. The CPU-only quantum path
(guarded by `_force_cpu_for_quantum`) makes this benign in production, but the
classical-on-MPS targets (`wgan_mlp/cnn/lstm`) hit the real device round-trip every
snapshot epoch and are exposed.
**Fix:** Capture `orig_device` before any mutation and put the entire body (including the
`.to("cpu")`) inside the `try`, so the `finally` restore always runs even if generation
raises. Additionally, consider letting a snapshot-generation failure propagate (or set a
sentinel) rather than relying on `training.py`'s swallowing `except`, so a corrupted device
state surfaces instead of producing silent NaNs.

### WR-05: `run_introspect.py --assemble` indexes `inter["quantum"]` keys that only exist for the quantum target, with no presence check

**status:** fixed (commit 7ec0577 — added is_quantum/bipartition + per-snapshot key validation in `_assemble` with actionable re-run message)
**File:** `run_introspect.py:308`, `334`, `355`
**Issue:**
`_assemble` does `snap_epochs = inter["quantum"]["snapshot_epochs"]`, then
`inter["quantum"]["bipartition"]` (line 355) and `q = inter["quantum"]["snapshots"]` and
indexes `s["param_norm"]`, `s["vn_entropy"]`, `s["purity"]` for every snapshot. These keys
are only written when `hasattr(gen_model, "introspect")` is true
(`make_snapshot_cb` line 150). If the quantum intermediate was produced by a code path
where `introspect()` was unavailable (e.g. a future refactor, or a partially-written
intermediate from an interrupted run that still passed the `p.exists()` check at line 302),
`param_trajectory.json`/`entanglement_trajectory.json` assembly dies with a bare `KeyError`
naming an internal dict key, not an actionable "the quantum run is incomplete" message.
The existence check at line 302 only verifies the file exists, never that it is complete.
**Fix:** After loading `inter["quantum"]`, validate the quantum payload shape before
assembly:
```python
q_doc = inter["quantum"]
if not q_doc.get("is_quantum") or "bipartition" not in q_doc:
    raise ValueError(
        "quantum intermediate is missing introspection fields "
        "(bipartition/param_norm/vn_entropy) — re-run "
        "`--target quantum` to regenerate a complete intermediate"
    )
for s in q_doc["snapshots"]:
    missing = {"param_norm", "param_angles", "vn_entropy", "purity"} - set(s)
    if missing:
        raise ValueError(f"quantum snapshot epoch={s.get('epoch')} missing {missing}")
```

## Info

### IN-01: `_load_checkpoint` rebinds `checkpoint = ckpt` purely to keep the old print statement

**status:** deferred (Info, out of fix scope)

**File:** `core/training.py:191`
**Issue:** `checkpoint = ckpt` exists only so the trailing `print(f"... {checkpoint['epoch']
...}")` keeps its original variable name. This is dead-style aliasing that obscures intent.
**Fix:** Use `ckpt` directly in the print and delete the alias line.

### IN-02: `tests/__init__.py` is empty but present; `conftest.py` already handles path bootstrap

**status:** deferred (Info, out of fix scope)

**File:** `tests/__init__.py:1`
**Issue:** An empty `tests/__init__.py` turns `tests/` into a package. Combined with
`conftest.py` inserting the repo root on `sys.path`, this is redundant and can cause pytest
rootdir/import-mode ambiguity in some configurations. Not a defect given `pytest.ini`
`testpaths = tests`, but worth noting.
**Fix:** Optional — either remove `tests/__init__.py` or document why package-mode is
intended; no functional change needed.

### IN-03: `run_ansatz_comparison.py` defines `_bootstrap_repo_on_path` and `_find_repo_root` doing nearly the same walk-up

**status:** deferred (Info, out of fix scope)

**File:** `run_ansatz_comparison.py:50-58` and `115-121`
**Issue:** Two functions independently walk parents looking for
`core/preprocessing.py` — one for `sys.path` bootstrap, one for path resolution.
This duplicated traversal logic also appears verbatim in `run_introspect.py:193-199` and
`run_introspect_figures.py:51-57`. Code duplication across four driver modules.
**Fix:** Factor a single `repo_root()` helper into a shared module (e.g.
`core/_paths.py`) and import it; low priority since drivers are intentionally
self-contained.

### IN-04: `run_ansatz.py` `--epochs` default (1000) and hardcoded `_train_wgan` 1000 create a misleading "consistent" appearance

**status:** deferred (Info, out of fix scope — note: CR-01 fix removed the underlying hardcode)

**File:** `run_ansatz.py:313`, `257`
**Issue:** Because both the argparse default and the hardcoded `train_wgan_gp(num_epochs=
1000)` are 1000, the CR-01 bug is invisible in the default invocation and in every test —
the values only diverge when a non-default `--epochs` is supplied. This coincidental
agreement is why the defect is not caught by `test_ansatz_variants.py` / the schema test.
Flagged so the fix for CR-01 includes a regression test that runs with `--epochs != 1000`
(or asserts the `num_epochs` actually threaded into `train_wgan_gp`).
**Fix:** Add a test that constructs the driver with a small non-default epoch count and
asserts the resulting `metrics` length / `config.yaml` epochs matches the training actually
performed.

---

_Reviewed: 2026-05-19_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
