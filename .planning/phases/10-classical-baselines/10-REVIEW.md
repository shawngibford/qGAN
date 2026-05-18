---
phase: 10-classical-baselines
reviewed: 2026-05-17T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - revision/core/models/classical.py
  - revision/core/models/nonadversarial.py
  - revision/core/models/__init__.py
  - revision/core/training.py
  - revision/run_baselines.py
  - revision/run_baselines_sweep.sh
  - revision/_build_baseline_notebook.py
  - revision/tests/test_classical.py
  - revision/tests/test_nonadversarial.py
  - revision/tests/__init__.py
findings:
  critical: 2
  warning: 7
  info: 5
  total: 14
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-05-17
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 10 adds three matched-parameter classical WGAN-GP generators, two
non-adversarial baselines (VAE, AR), a per-run CLI driver, a resumable sweep
shell script, and a notebook generator. The matched-parameter generator design
(single flat `params_pqc` `nn.Parameter`, functional forward) is sound and the
parameter arithmetic (74/73/78) checks out against the test assertions. The AR
fit/sample lag bookkeeping is internally consistent.

The MPS device-move + `compute_dtype` change in `training.py` largely preserves
the CPU/CUDA float64 path, but introduces **two correctness defects that affect
the float64 (CPU/CUDA) path it claims to leave untouched**: (1) the spectral-loss
hook now mixes a CPU-derived target tensor with a device generator output and
produces a non-device-safe / non-differentiable scalar, and (2) the
`EarlyStopping` checkpoint restore path is now device/dtype-inconsistent with the
Adam optimizer state. Per-seed RNG reproducibility is preserved on CPU, and the
sweep correctly avoids `multiprocessing.Pool` (uses `xargs -P`), satisfying the
scientific-validity requirement. The remaining findings are robustness and
maintainability issues.

## Critical Issues

### CR-01: Spectral-loss hook is non-differentiable and device-unsafe; breaks the documented opt-in contract

**File:** `revision/core/training.py:356-360, 470-507`
**Issue:** When `spectral_loss_weight > 0` the generator phase computes
`_spectral_psd_loss(gen_out, real_log_returns_for_psd(...))`. Two defects:

1. `gen_out` lives on `device` (mps/cuda) after the Phase-10 device move, while
   `real_log_returns_for_psd` returns `torch.stack` of CPU tensors from
   `gan_data_list`. Inside `_spectral_psd_loss` the final expression
   `return mse * fake_flat.var() / (fake_flat.var().detach() + eps)` produces a
   tensor on `gen_out`'s device, but the function never validates device
   agreement and the surrounding code adds it to `generator_loss` (also on
   device) — this only works by accident and silently breaks if the real-target
   path is changed.
2. More seriously, the returned scalar is **mathematically constant** w.r.t.
   `params_pqc`: `mse` is a Python float computed from detached numpy arrays,
   and `mse * var / var.detach()` has gradient `mse * d(var)/d(params) / var.detach()`
   — the PSD mismatch itself contributes **zero gradient**. The spectral penalty
   therefore does not optimize spectral fidelity at all; it only nudges the
   generator's output variance by a scalar proportional to the (frozen) PSD MSE.
   The docstring claims this "re-implements the term so callers that want to opt
   back in can do so" — that contract is false; opting in trains the wrong
   objective.

**Fix:** Either (a) gate the hook off explicitly with a `NotImplementedError`
until a differentiable PSD is implemented (Phase 13), so callers cannot silently
opt into a broken objective, or (b) implement a differentiable torch FFT-based
log-PSD MSE and ensure the real target is moved to `gen_out.device`:
```python
def _spectral_psd_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real = real.to(fake.device).detach()
    # torch.fft-based PSD so gradient flows through `fake`
    f = torch.fft.rfft(fake.reshape(-1))
    r = torch.fft.rfft(real.reshape(-1))
    eps = 1e-12
    log_psd_fake = torch.log((f.abs() ** 2) + eps)
    log_psd_real = torch.log((r.abs() ** 2) + eps)
    return torch.mean((log_psd_fake - log_psd_real) ** 2)
```
At minimum, document loudly that the current implementation is a non-functional
placeholder and reject `spectral_loss_weight > 0` rather than silently training
the wrong loss.

### CR-02: EarlyStopping checkpoint restore is device/dtype-inconsistent with optimizer state after the MPS move

**File:** `revision/core/training.py:163-171, 244, 263`
**Issue:** After the Phase-10 change, `generator` is moved to `device` and
`g_opt = torch.optim.Adam([generator.params_pqc], ...)` binds Adam state
(`exp_avg`, `exp_avg_sq`) to the device tensor. `EarlyStopping._save_checkpoint`
saves `model.params_pqc.detach().clone()` (a device, float32-on-MPS tensor) and
`g_optimizer.state_dict()`. On restore, `_load_checkpoint` does
`model.params_pqc.data = checkpoint["params_pqc"]` then
`model.g_optimizer.load_state_dict(checkpoint["c_optimizer"-style state])` and
re-points `param_groups[0]["params"] = [model.params_pqc]`. The optimizer's
loaded `exp_avg`/`exp_avg_sq` tensors are restored to whatever device they were
saved on, but Adam's `step()` requires param and state tensors to share a device;
`torch.load` with `weights_only=False` and no `map_location` will restore to the
saved device, which on a different host or after the MPS/CPU split can mismatch
the live `params_pqc` device. This produces either a hard `RuntimeError`
("Expected all tensors to be on the same device") or, worse, a silent no-op
update if shapes line up but devices differ across backends. The pre-MPS path
was device-uniform (always CPU) so this defect is *introduced* by the Phase-10
change, contradicting the comment claiming "The CPU/CUDA path keeps float64 so
prior 09.1-style runs reproduce exactly."

**Fix:** Pin checkpoint I/O to the live device explicitly:
```python
def _load_checkpoint(self, model):
    dev = model.params_pqc.device
    checkpoint = torch.load(self.checkpoint_path, weights_only=False,
                            map_location=dev)
    model.params_pqc.data = checkpoint["params_pqc"].to(dev)
    model.critic.load_state_dict(checkpoint["critic_state"])
    model.c_optimizer.load_state_dict(checkpoint["c_optimizer"])
    model.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
```
and likewise `map_location` / `.to(dev)` for the critic state. Add a regression
test that runs `train_wgan_gp` with an `EarlyStopping` instance on the selected
device and asserts a post-restore optimizer `step()` succeeds.

## Warnings

### WR-01: WGAN-GP double-backward on MPS float32 is unverified and may silently degrade or fail

**File:** `revision/core/training.py:63-72, 234`
**Issue:** `compute_gradient_penalty` uses `torch.autograd.grad(..., create_graph=True)`
followed by `gp.backward()` (double-backward through the critic's Conv1d stack).
On the new MPS float32 path this exercises second-order autograd through MPS
Conv1d kernels, which historically have had incomplete/incorrect double-backward
support in several PyTorch releases. The code asserts (in a comment) that
"only the numeric precision ... differs, which is acceptable" but provides no
test that the MPS gradient penalty is numerically close to the CPU float64 one.
A silently wrong GP term invalidates the WGAN-GP baseline scientifically.

**Fix:** Add a test that builds the shared `Critic`, runs
`compute_gradient_penalty` on identical inputs on CPU(float64) and MPS(float32)
(skipped if MPS unavailable), and asserts the scalar agrees within a tolerance
(e.g. `rtol=1e-2`). If it diverges, force the GP/critic to run on CPU even when
MPS is selected for the generator.

### WR-02: `compute_gradient_penalty` ignores its `device` argument — dead parameter that invites misuse

**File:** `revision/core/training.py:36, 52-56`
**Issue:** The `device` parameter is accepted but never used; placement is driven
entirely by `real_samples.device`. The docstring rationalizes this as "API
symmetry," but a dead, silently-ignored argument is a latent bug: a caller
passing a different `device` (as `train_wgan_gp` does at line 326, passing the
loop `device`) will reasonably expect it to take effect and will not notice if
`real_samples` is accidentally on the wrong device.

**Fix:** Either drop the parameter entirely, or assert consistency:
`assert real_samples.device == device, (real_samples.device, device)`.

### WR-03: Optimizer rebinds to a stale parameter tensor if `params_pqc` is ever reassigned

**File:** `revision/core/training.py:263, 457-461`; `revision/core/models/classical.py:77`
**Issue:** `g_opt = torch.optim.Adam([generator.params_pqc], ...)` captures the
tensor object. The `_ESAdapter.params_pqc` setter reassigns
`self._generator.params_pqc = value`, which would orphan the optimizer (it still
points at the old tensor). `EarlyStopping._load_checkpoint` happens to use
`.data =` (in-place) so this is not hit today, but the setter exists "for safety"
and is actively wrong — using it would silently freeze generator training.

**Fix:** Remove the misleading setter, or make it raise:
```python
@params_pqc.setter
def params_pqc(self, value):
    raise AttributeError(
        "params_pqc must be updated in-place (.data =); reassignment "
        "would orphan the generator optimizer")
```

### WR-04: `train_wgan_gp` mutates global RNG state, coupling it to caller-side determinism

**File:** `revision/core/training.py:211-213`
**Issue:** The function calls `torch.manual_seed`, `np.random.seed`, and
`random.seed` as side effects on import-global RNGs. `run_baselines._train_wgan`
ALSO calls `torch.manual_seed(seed)` (line 246) before constructing the
generator, then `train_wgan_gp` reseeds again with the same `seed` at line 211 —
so generator weight init (in `__init__`, before `train_wgan_gp`) and the loop's
RNG draws are seeded by *different* `manual_seed` calls. This works for the
default `seed`, but is fragile: any future caller that constructs the generator
under one seed and calls `train_wgan_gp` with another will get a confusing
init/loop seed split, and the global `np.random.seed` clobbers any RNG the
caller set up. This is acceptable for the current single-process sweep but is a
reproducibility hazard worth documenting/guarding.

**Fix:** Document explicitly that `train_wgan_gp` owns and reseeds the global
RNGs, and that callers must construct the generator AFTER deciding the seed (or
seed identically). Consider moving generator construction inside `train_wgan_gp`
or accepting a pre-seeded generator and NOT reseeding numpy globally (use a
local `np.random.Generator`).

### WR-05: `_compute_data_hash` loads and preprocesses the CSV a second time — silent drift risk

**File:** `revision/run_baselines.py:226-234, 466-469`
**Issue:** `build_dataset_for_pipeline` calls `load_and_preprocess(str(csv_path))`
and `_compute_data_hash` calls it again independently. If `load_and_preprocess`
is non-deterministic in any way (e.g. row order, NaN handling, float rounding),
the `data_hash` written to `config.yaml` would not actually correspond to the
tensor used for training, defeating the D-10-15 provenance guarantee (the very
"Pitfall 4" the docstring cites).

**Fix:** Compute the hash from the *same* `raw["OD"]` tensor produced inside
`build_dataset_for_pipeline` and return it on the `DatasetBundle`, rather than
re-invoking `load_and_preprocess`.

### WR-06: Sweep skips a triple as "complete" on artifact presence alone — corrupt/short artifacts pass

**File:** `revision/run_baselines_sweep.sh:174-184, 295-301`
**Issue:** `is_complete` checks only that the five files exist and are non-empty
(`-s`). A run that crashed mid-`np.save` or wrote a truncated `samples.npy`
(non-empty but malformed) is treated as complete and permanently skipped on
resume, silently poisoning the Wave-4 comparison. The CLI driver is idempotent
on rerun, but only if the sweep actually reruns it — which it won't for a
non-empty-but-corrupt bundle.

**Fix:** Strengthen the completeness check: validate `samples.npy` loads and has
the expected `(n_synth, WINDOW_LENGTH)` shape and that `config.yaml` parses and
contains `data_hash`. A tiny Python `--validate` helper invoked by `is_complete`
is sufficient; on validation failure, treat the triple as incomplete and rerun.

### WR-07: VAE/AR run on CPU while WGAN runs on MPS — cross-family metric comparability not asserted

**File:** `revision/run_baselines.py:285-335, 380-393`
**Issue:** `_train_vae` and `_train_ar` never move to `device`; they always run
on CPU float32/float64. The WGAN branch trains on MPS float32 (when available).
The Wave-4 notebook compares all families "apples-to-apples," but the WGAN
samples were produced under a different precision/backend than VAE/AR. The
`generate_wgan_samples` step moves the generator back to CPU float64 for sample
generation (good), but the *trained weights* themselves were optimized in MPS
float32 — there is no test/assertion that this does not bias the comparison.

**Fix:** Document this asymmetry in the comparison artifact (it currently is not
mentioned in `train_protocol_notes`), and add the CPU-vs-MPS GP equivalence test
from WR-01 as the gate that justifies treating MPS-trained WGAN weights as
comparable to CPU-trained VAE/AR.

## Info

### IN-01: `elbo_hist` stores a loss, not an ELBO

**File:** `revision/run_baselines.py:300, 322, 327, 342-343`
**Issue:** `loss = recon + beta * kld` (a minimization objective) is appended to
`elbo_hist` and emitted in `metrics.json` under key `"elbo"`. The ELBO is the
*negative* of this (and `recon` here is MSE, not a log-likelihood). Downstream
consumers reading `metrics["elbo"]` as an ELBO will misinterpret it.

**Fix:** Rename the key to `"neg_elbo_proxy"` / `"train_loss"`, or document in
`train_protocol_notes` that `elbo` is `MSE + beta*KLD` (a loss, not the ELBO).

### IN-02: Redundant double `no_grad` on VAE sampling

**File:** `revision/core/models/nonadversarial.py:105`; `revision/run_baselines.py:334-335`
**Issue:** `VAEBaseline.sample` is decorated `@torch.no_grad()` and is also
called inside `with torch.no_grad():` in `_train_vae`. Harmless but redundant
and signals confusion about ownership of the no-grad context.

**Fix:** Keep the decorator (it is the model's contract) and drop the redundant
`with torch.no_grad()` wrapper at the call site, or vice versa.

### IN-03: Module-level grep sentinels are code smell

**File:** `revision/core/training.py:510-514`
**Issue:** `_NOISE_HIGH_LITERAL = 4 * math.pi` exists solely so a grep-based
verification step finds the literal `4 * math.pi` in the file. Production code
should not carry dead constants to satisfy a CI text search; this invites future
readers to "use" it and is a maintenance trap.

**Fix:** Move the verification to assert on `revision.core.NOISE_HIGH`'s value
instead of grepping source text, and delete the sentinel.

### IN-04: `args` shadows nothing but `else: raise` branch is provably unreachable

**File:** `revision/run_baselines.py:487-488`
**Issue:** The `else: raise ValueError(...)` after the `vae`/`ar`/`_WGAN_GENERATORS`
dispatch is unreachable because `argparse` `choices=_MODEL_CHOICES` already
constrains `args.model`. The inline comment acknowledges this. Dead defensive
code is acceptable but should be marked `# pragma: no cover` for honesty in
coverage reports (consistent with the callback `except` at training.py:410).

**Fix:** Add `# pragma: no cover` to the unreachable branch.

### IN-05: `eval` phase ACF/vol/lev metrics are hardcoded placeholders

**File:** `revision/core/training.py:390-392`
**Issue:** `acf_avg`, `vol_avg`, `lev_avg` are appended as constant `0.0` every
eval epoch. Anyone reading `metrics["acf_avg"]` from a baseline run will see a
flat zero trace and may mistake it for a real (degenerate) measurement rather
than an unimplemented placeholder. The comment explains this is intentional
(parity is checked on final state elsewhere), but the emitted metric is
misleading data.

**Fix:** Emit `None` (or omit the keys) instead of `0.0` so downstream consumers
can distinguish "not computed" from "computed and zero," or name the keys
`acf_avg_placeholder`.

---

_Reviewed: 2026-05-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
