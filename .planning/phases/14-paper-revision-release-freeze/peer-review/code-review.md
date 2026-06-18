---
reviewer: code-quality
scope: Phase 14 revision codebase
files_examined: 16
created: 2026-05-20T00:00:00Z
---

# Code Review — Phase 14 Peer-Review Pass

## Summary

The codebase shows strong discipline around the explicit-`raise AssertionError`
idiom (python -O safe), the device/dtype manifest pattern, and the
"read-from-JSON-never-hand-type" provenance ethos. However, **the audit
surfaces three categories of defect that block resubmission**: (1) a
non-deterministic seed in `run_figure_suite.render_time_series_comparison`
that breaks figure reproducibility across Python invocations, (2) a
silent metric-conflation in `render_cross_model_emd` where the bar chart
plots "best-training-window EMD" (log-return-space, single-batch) but
overlays the FROZEN-checkpoint headline reference line in OD-scale full-N_synth
EMD — two non-comparable quantities on one axis — and (3) `run_methods_full.py`
emits hardcoded line citations (`training.py:347` / `training.py:259-268`)
into the methods JSON instead of computing them through `_first_lineno`,
inviting the same drift the rest of the citation block is built to prevent.
A fourth class of defect — the strict-accept gate cannot detect a silent
device fallback for classical WGAN models because `_train_wgan` does not
guard `torch.backends.mps.is_available` the way `_train_quantum` does, while
the per-classical `_device_manifest(None)` unconditionally writes
`backend_assertion="PASSED"` — quietly mixes CPU-quantum and MPS-classical
runs into the same "matched-2000ep budget" cell on Apple Silicon. The single
most important finding is the cross-model EMD figure conflation (CRITICAL CR-2)
because it directly affects a manuscript headline figure.

## Findings by Severity

### CRITICAL (blocks publication)

#### CR-1: Non-deterministic figure-window selection seed in `render_time_series_comparison`
**File:** `run_figure_suite.py:382`
**Problem:** `rng = np.random.default_rng(model.__hash__() & 0xFFFF)`. Here
`model` is a Python `str` and Python 3 randomizes string hashes per
interpreter invocation (PYTHONHASHSEED defaults to "random"). Verified
empirically: `python3 -c "print(hash('iqp_sel_55_repro'))"` returns a
different value on consecutive Python invocations.
**Impact:** The "real" and "fake" window indices shown in the figure change
on every render, so the manuscript figure `timeseries_<model>.png` is
**not reproducible** from the same artifact bundle and the companion JSON's
`real_window_idx` / `fake_window_idx` are forensically meaningless across
invocations. Directly contradicts the render-only / reproducible-figure
contract (T-14-11).
**Fix:**
```python
rng = np.random.default_rng(
    int(hashlib.sha256(model.encode()).hexdigest()[:8], 16)
)
```
or even better, key off PRIMARY_SEED + model index so all panels in the
suite use comparable RNGs.

#### CR-2: Cross-model EMD figure silently merges two distinct EMD metrics
**File:** `run_figure_suite.py:620-654`
**Problem:** `render_cross_model_emd` builds bars from
`np.min(metrics["emd_avg"])` per model (line 630) — i.e., **best
training-window EMD over the eval trajectory**, computed by
`training.py:418-420` on a SINGLE batch of `real_log_returns` (the last
critic-phase batch, log-return scale, batch_size=12 samples). The
FROZEN-checkpoint headline reference horizontal line at line 653 is sourced
from `headline_canonical.json`'s `(metric_name="emd", scale="OD")` row —
which is `revision.core.eval.compute_emd` on the **OD-scale full N_synth =
10 * n_real_windows samples**. The two are different metrics on different
sample sets on different scales, but plotted on the **same y-axis** as if
comparable. The y-label literally says "best EMD over training (mean ± std
over 5 seeds)" — describing the bars — and the dashed reference line is
labelled "IQP:SEL 55p FROZEN headline (ckpt epoch 1969)" without disclosing
the metric mismatch.
**Impact:** Direct manuscript figure. A reviewer reading this will
conclude the frozen headline is on the same metric as the matched-budget
bars and either declare the headline trivially beats the reproduction (it
should — different metric!) or accuse the authors of cherry-picking. This
is exactly the headline-vs-reproduction conflation D-14-10 was written
to prevent.
**Fix:** Either (a) draw the reference line from `metrics["emd_avg"]` of
a hypothetical headline trajectory (does not exist for the frozen
checkpoint — it has no per-epoch `emd_avg`), so the comparison is by
construction impossible; or (b) replace the bars with OD-scale full-N_synth
EMD pulled from `matched2000_dualscale.json`'s `OD/emd` aggregates so
both sides are the same metric. Option (b) is consistent with the rest
of the figure suite (`render_matched2000_dualscale_sidebyside` already
sources from that JSON). Until the metric mismatch is reconciled, this
figure SHOULD NOT ship.

#### CR-3: Hardcoded line citations in methods_full.py contradict the docstring's anti-drift contract
**File:** `run_methods_full.py:466-468`
**Problem:**
```python
"dtype_samples": (
    "torch.float64 (sample-generation pipeline: "
    f"{cits['compute_dtype_split']} compute_dtype = torch.float64 on "
    "CPU/CUDA; core/training.py:347 generator output cast "
    ".to(compute_dtype) * 0.1; MPS path falls back to float32 "
    "because MPS lacks float64 — see training.py:259-268)"
),
```
Two of the four citations are programmatic (`cits['compute_dtype_split']`)
but `training.py:347` and `training.py:259-268` are **hardcoded text
literals** in the f-string. The whole point of `_citations` and
`_first_lineno` (lines 97-137) is to prevent drift — the explicit comment
at lines 132-134 says "training.py may have been refactored; regenerate
methods_full.json on every emitter run so citations cannot go stale".
The two hardcoded line numbers undermine that guarantee.
**Impact:** A future edit to `core/training.py` (e.g., adding
a docstring line above line 268) silently makes the methods JSON cite the
wrong code. The number-provenance gate would not catch this — it only
checks that the number resolves to SOME JSON, not that the citation
line points to the correct source.
**Fix:** Add two more grep targets:
```python
"generator_to_compute_dtype": "generated_samples = generated_samples.to(compute_dtype) * 0.1",
"mps_dtype_block_start": "compute_dtype = torch.float32 if device.type",
"mps_dtype_block_end": "generator = generator.to(device)",
```
and replace the f-string with `{cits['...']}` substitutions.

#### CR-4: Strict-accept gate cannot detect classical-WGAN silent MPS fallback
**File:** `run_matched2000.py:434-492` (`_train_wgan`),
`run_matched2000.py:255-314` (`_device_manifest`),
`run_matched2000.py:661-668` (gate check 4)
**Problem:** `_train_quantum` (line 372-389) wraps `train_wgan_gp` with a
`torch.backends.mps.is_available = lambda: False` patch so quantum
training runs CPU-only. `_train_wgan` does **not** apply the same patch
— so on Apple Silicon, classical WGAN-GP training runs on **MPS in
float32** (training.py:268 — `compute_dtype = torch.float32 if device.type
== "mps"`). The post-training `generate_wgan_samples` resets the generator
to CPU at line 214, so the samples ARE on CPU/float64. But the device
manifest `_device_manifest(None)` for classical models at line 484 takes
the `generator=None` branch — which **never touches** the actual
training-time device. It writes `backend_assertion = "PASSED"`
unconditionally (line 307). The strict-accept gate at line 663 only checks
`dm.get("backend_assertion") == "PASSED"` — it can never know the
classical run trained on MPS.
**Impact:** The manuscript's "matched 2000-epoch budget" claim quietly
mixes CPU-float64 quantum and MPS-float32 classical training on Apple
Silicon. Reproducibility on a CUDA / Linux box would produce different
classical numbers and the gate would still PASS. Reviewer A may not catch
this; reviewer B will, and the response involves disclosing that the
"matched" comparison was device-asymmetric.
**Fix:** Either (a) apply the same `torch.backends.mps.is_available =
lambda: False` patch in `_train_wgan` (and `_train_vae`) so every
matched-budget run is on CPU; or (b) record `training_time_device` (NOT
just `sample_generation_device`) into the device manifest by inspecting
the generator/critic AFTER training, and have the strict-accept gate
require equality across all models in the matched-budget sweep.

#### CR-5: `_resolves` substring match in verify_number_provenance.py admits trivial false positives
**File:** `verify_number_provenance.py:118-141`
**Problem:** The resolver first tries `if token in blob` (line 119) where
`blob` is the concatenated text of every `results/*.json`. A
literal like `"42"` resolves to **any** JSON containing the digits "42"
in any position — including in larger numbers (`"data_hash":
"91e447d4624e25b3"` contains 4, 2, 5 in many places; `"timestamp":
"2025-..."` contains 25; etc). Worse, the float resolution at
lines 122-141 uses `f"{cval:.{prec}f}" == f"{val:.{prec}f}"`: a manuscript
literal `0.0025` matches **any** JSON value that rounds to `0.0025` at
4-decimal precision — so a stale truncated `0.002493` in a stale JSON
silently resolves.
**Impact:** The gate is the EXECUTABLE enforcement of success-criterion 5
("every number traces to a JSON artifact") — but its resolution model
admits false positives broad enough that almost any 2-4 digit literal
resolves to some artifact, regardless of correctness. The gate provides
weaker guarantees than the docstring claims.
**Fix:** Tighten resolution: (a) require the match to be `token` flanked by
non-digit boundaries in the canonical JSON, not the raw blob (use a
regex `(?<![\d.])token(?![\d])`); (b) require float resolution to land
on a value within `10**-prec / 2` of the literal (not just same truncation);
(c) optionally require the matching key path to be recorded in a
`--manifest` output so the operator can audit which JSON key each
literal resolved to. Add a `--strict` flag that fails on multiple
ambiguous resolutions for the same token.

### HIGH (must address before resubmission)

#### HI-1: Hardcoded DTW RNG seed in `run_canonical_headline` ignores `--generation-seed`
**File:** `run_canonical_headline.py:280, 334`
**Problem:** Both `_od_scale_rows` and `_log_return_rows` use
`np.random.default_rng(42 * 31)` for DTW pair selection — the literal `42`
is hardcoded, not the `--generation-seed` CLI argument. The CLI default
IS 42, so today this matches, but the CLI accepts `--generation-seed N`
and uses it for sample generation (line 200) AND od_start draw
(line 234, `seed * 7919 + 1`). A user who runs `--generation-seed 43`
gets samples seeded with 43 but DTW pairs seeded with 42*31 — partially
seeded reproduction.
**Impact:** Provenance gap. The headline's `generation_seed` field is
recorded in the JSON but it does NOT fully parameterize the metric run.
**Fix:** Replace `42 * 31` with `generation_seed * 31` and thread the
seed through `_od_scale_rows` / `_log_return_rows`.

#### HI-2: `model_info.optimizer_betas` is hardcoded for every model, including VAE/AR
**File:** `run_model_info.py:158`
**Problem:** `"optimizer_betas": [0.0, 0.9]` is written for every model
row including VAE (Adam at lr=1e-3 with DEFAULT betas 0.9/0.999) and AR
(closed-form np.linalg.lstsq — no optimizer at all). The companion
`optimizer` string at line 155 correctly diverges per family
(`_optimizer_for`), but `optimizer_betas` does not. The methods_full.json
aggregator (`run_methods_full.py:430`) then reads
`repro.get("optimizer_betas", [0.0, 0.9])` from the quantum repro row,
so the canonical 3_training bucket also carries the wrong betas if
applied to non-WGAN entrants.
**Impact:** Provenance lie in the unified model registry. A reviewer
inspecting `model_info.models[]` for VAE will see `optimizer="Adam (single,
lr=1e-3) — VAE ELBO loop"` AND `optimizer_betas=[0.0, 0.9]` — the two
contradict.
**Fix:** Set `optimizer_betas = None` for non-WGAN-GP families, or
extract from a config field that was actually persisted at train time.

#### HI-3: Cross-artifact data_hash gate in run_model_info checks mutual equality but not the expected literal
**File:** `run_model_info.py:642-649`
**Problem:** The gate collects `data_hash` from headline + 9 sweep configs
and only checks `len(set(hashes.values())) != 1`. It does NOT compare
against the canonical literal `"91e447d4624e25b3"`. If every consumed
artifact agreed on a different hash (e.g., all referenced a corrupted
data.csv yielding `aaaaaaaaaaaaaaaa`), the gate silently passes and the
emitter publishes a `data_hash: "aaaa..."` model_info.json. `canonical_hash
= next(iter(hashes.values()))` (line 649) is the first hash in insertion
order — not a validated value.
**Impact:** Inconsistency with `run_matched2000_dualscale.verify_data_hash`
(line 233-240) and `run_matched2000.EXPECTED_DATA_HASH = "91e447d4624e25b3"`
(line 106), which both compare against the literal. The gate's strength
is asymmetric across emitters.
**Fix:** Add `EXPECTED_DATA_HASH = "91e447d4624e25b3"` and an explicit
`if next(iter(hashes.values())) != EXPECTED_DATA_HASH: raise
AssertionError(...)`.

#### HI-4: Hardcoded `topology = "range"` overrides the canonical lock JSON in iqp_sel_55_repro path
**File:** `run_matched2000.py:337-347`
**Problem:** For `_REPRO_MODEL`, the function reads the lock JSON
(`canonical_config_lock.json`), pulls `circuit_id` and `num_layers` from
it, but **hardcodes** `topology = "range"` at line 344 instead of
reading `lock["decomposition"]["gate_layout"]["entangler"]`. This is
correct today (the recovered canonical is range), but a future lock
JSON re-emit with a different topology would be silently overridden.
**Impact:** Single-source-of-truth violation. The lock JSON IS supposed
to be the source for ALL of (circuit_id, num_layers, topology); two
of three are read from it and one is hardcoded.
**Fix:**
```python
topology = str(decomp.get("gate_layout", {}).get("entangler", "range"))
```

#### HI-5: `model_kinds` field in matched2000_dualscale.json excludes the headline
**File:** `run_matched2000_dualscale.py:103-113, 595`
**Problem:** `MODEL_KINDS` is the 9-model list. The `rows[]` and
`aggregates[]` arrays include rows under
`model_kind="frozen_checkpoint_headline"`, but
`obj["model_kinds"] = MODEL_KINDS` (line 595) advertises only 9 kinds.
A downstream consumer that iterates `model_kinds` to drive rendering
would silently skip the headline.
**Impact:** Schema inconsistency. `run_figure_suite`'s
`DUALSCALE_MODEL_ORDER` happens to hardcode the same 9 (line 734) and
then queries the headline separately via `HEADLINE_KIND = "frozen_..."`,
so the inconsistency is invisible to the figure renderer — but the
JSON's self-description is wrong.
**Fix:**
```python
"model_kinds": [*MODEL_KINDS, HEADLINE_MODEL_KIND],
```
or add `"headline_model_kind": HEADLINE_MODEL_KIND` (already done at
line 600) and document that `model_kinds` excludes it.

#### HI-6: `verify_freeze_ready` gates only top-level `results/*.json` while `verify_number_provenance` walks rglob
**File:** `verify_freeze_ready.py:82, 116` vs
`verify_number_provenance.py:99`
**Problem:** Freeze gate uses `RESULTS_DIR.glob("*.json")` — one level
only. The number-provenance gate uses `RESULTS.rglob("*.json")` —
recursive. Paper literals may resolve to a NESTED JSON (e.g.,
`results/matched2000/runs/<m>/<s>/metrics.json`) that the
verifier sees, but the freeze gate doesn't audit / force-stage. If the
nested JSON is gitignored (default per the broad `results/` rule on
line 62 of `.gitignore`), the verifier passes pre-tag and the DOI'd
archive ships without those numbers' source artifacts.
**Impact:** Pitfall 4 (gitignore exclusion) is the exact reason
`verify_freeze_ready` exists — but its glob depth doesn't match the
verifier it gates.
**Fix:** Change `RESULTS_DIR.glob` to `RESULTS_DIR.rglob` at lines 82
and 116. Also fix the negation self-heal to write
`!results/**/*.json`.

#### HI-7: `_train_vae` does not seed numpy or random
**File:** `run_matched2000.py:503`
**Problem:** Only `torch.manual_seed(seed)` is set. `np.random.seed` and
`random.seed` are NOT set inside `_train_vae`. The DataLoader uses
`shuffle=False` so iteration order is deterministic, and VAE
`reparameterize` uses `torch.randn` (seeded). But the eventual
`vae.sample(n_synth, gen)` accepts a torch.Generator — correct. So
in current source, the lack of np/random seed is harmless. But the
docstring at run_matched2000.py:204 ("FIXED ``np.random.default_rng(seed)``")
implies numpy RNG is fixed across all training paths — which is not
guaranteed for VAE.
**Impact:** Reproducibility caveat. If a future edit to VAE training
introduces numpy random usage, results become irreproducible silently.
**Fix:** Add `np.random.seed(seed); random.seed(seed)` to `_train_vae`
top, mirroring `train_wgan_gp`'s seed block.

#### HI-8: `default=float` in json.dumps converts NaN/Inf to invalid JSON tokens
**File:** `run_matched2000.py:796`, `run_figure_suite.py:191`
**Problem:** `json.dumps(metrics, indent=2, default=float)` emits
`NaN` / `Infinity` / `-Infinity` for non-finite floats. These are NOT
valid JSON per RFC 8259 (Python's `json` accepts them due to a
backward-compatibility default, but many strict parsers — including
some JavaScript runtimes and YAML loaders — reject them).
**Impact:** If any metric becomes NaN (e.g., compute_emd on empty
samples, or a degenerate model collapse), the written JSON cannot
be re-loaded by strict downstream tooling. The strict-accept gate
doesn't validate finiteness.
**Fix:** Add `_finite_sanitize` (already exists in `run_sensitivity.py:649`
— reuse) that converts non-finite floats to None before dumping, with
an explicit raise if more than a threshold of metrics are non-finite.

#### HI-9: Lazy-evaluating `is_complete` re-runs the strict gate via subprocess on EVERY worklist entry
**File:** `run_matched2000_sweep.sh:194-209`
**Problem:** `is_complete` shells out to `${PYTHON} -m
${RUN_MODULE} --accept ...` for each (m, s) pair on every sweep
invocation — including during `--dry-run`, which calls `is_complete`
for all 45 pairs (line 362-368). Each Python boot is ~500ms-1s on
typical hardware; 45 * 1s = 45s of wasted wall time for a dry run,
and is_complete is also re-evaluated DURING dispatch (line 312) AND
AFTER training (line 344). For a typical sweep, that's 90+ subprocess
boots.
**Impact:** Performance, not correctness. Listed for completeness.
**Fix:** Cache `is_complete` results in a hash map, or fold the file
presence + JSON-readability fast path into bash before invoking python.
(Note: out-of-scope per review instructions but flagged because it
materially affects iteration time.)

### MEDIUM (should address)

#### MD-1: Stale `real_log_returns` reference in eval phase creates noisy per-epoch EMD trajectory
**File:** `core/training.py:418`
**Problem:** `real_log_returns` in the eval phase (line 418) is the LAST
critic-phase iteration's batch — a single batch of 12 real samples
that varies across epochs due to `torch.randint` re-sampling
(line 328). The recorded `emd_avg[t]` is therefore EMD(fake_batch,
real_batch_t) where real_batch_t is a random 12-sample draw at the
last critic iteration of epoch t. This injects sampling noise into
the trajectory unrelated to model quality.
**Impact:** The "best EMD over training" used in cross-model figures
is partly chasing favorable real-batch draws. Quantifies the
trajectory noise — does NOT bias the final number, but inflates
seed-to-seed variance. This file is byte-frozen so the fix is
out-of-scope, but worth disclosing in the manuscript.
**Fix:** (out-of-scope per byte-freeze, but for the v3.0 baseline:
fix to a held-out reference set computed once outside the loop).

#### MD-2: `compute_dtype` only exists for the WGAN-GP path; VAE training uses default torch dtype
**File:** `run_matched2000.py:511-531`
**Problem:** The VAE training loop has no `compute_dtype` switching,
unlike `train_wgan_gp` (training.py:268). It runs at whatever the
default torch dtype is — float32 on MPS, float32 on CPU. The
methods_full.py bucket 4 documents `dtype_samples` as float64 on
CPU/CUDA — but VAE samples are emitted via `vae.sample(...).to(
torch.float64)` (line 536), which casts UP from the natively-trained
float32. **Information lost** by upcast: the sample-generation precision
is float32 internally, then upcast to float64 for storage. The methods
JSON claim that samples are float64 on CPU is technically true after
the cast, but the underlying computation was float32.
**Impact:** Methodological precision disclosed in the paper claims
float64 sample generation; the underlying VAE pipeline is float32. A
reviewer pinning numerical reproducibility may catch this.
**Fix:** Disclose explicitly that VAE samples are upcast from float32
in `dtype_samples` documentation, OR force VAE training to compute_dtype
to match WGAN.

#### MD-3: `head_epoch != 1969` hardcoded assertion in figure renderer
**File:** `run_figure_suite.py:1117`
**Problem:** `if head_epoch != 1969: raise ValueError(...)`. The
canonical checkpoint epoch IS 1969 today, but if the recovery
provenance ever shifts (e.g., a different `best_checkpoint.pt` is
selected for the v3.0 baseline), this renderer breaks. The check
should reference the lock JSON, not a literal.
**Impact:** Coupling between figure renderer and current ckpt epoch.
**Fix:** Read `lock["checkpoint_epoch"]` and compare; if absent, fail
with a different message.

#### MD-4: `_slice_module_docstring` would silently truncate on triple-quoted strings inside the docstring
**File:** `run_methods_full.py:144-162`
**Problem:** The slicer finds the FIRST `"""` and the NEXT `"""`. If the
docstring contained an embedded triple-quoted code example (e.g.,
``"""\n>>> '''\nfoo\n'''\n"""``), the slice would terminate prematurely.
For the current `run_matched2000.py` docstring, there are no internal
triple-quotes (verified — only 2 occurrences total). Brittle but
functional.
**Impact:** Brittle to future docstring edits.
**Fix:** Use `ast.parse` + `ast.get_docstring` for robust docstring
extraction.

#### MD-5: `_check_ignored_json` and gate (a) silently mutate `.gitignore` + stage files as a "verify" step
**File:** `verify_freeze_ready.py:101-117`
**Problem:** Gate (a) appends to `.gitignore` and runs `git add -f`. A
"verify" script should NOT mutate the working tree — operators expect
verifiers to be observe-only. The self-heal pattern is documented but
the script name (`verify_freeze_ready.py`) implies pure observation.
**Impact:** Surprise side effects; in CI, this leaves uncommitted
.gitignore modifications. Operator may not notice and commit a
spurious change.
**Fix:** Rename to `prepare_freeze.py` OR add a `--check-only` flag that
disables remediation and exits non-zero with an actionable message.

#### MD-6: `verify_freeze_ready.gate_c` only checks `*.pt` / `*.pth` for size; misses large `.npz` / data files
**File:** `verify_freeze_ready.py:192-204`
**Problem:** The large-file check filters by extension (`.pt`, `.pth`).
A 500MB `.npz` or `.npy` would pass. D-14-21 says "large checkpoints
referenced by hash, not committed" — but a large dataset committed as
`.npz` is the same archive problem.
**Impact:** Incomplete enforcement of the tag-scope invariant.
**Fix:** Loop over `ls_files` and check size for any file > threshold,
regardless of extension. Whitelist `data.csv` and a few known-large
allowed paths.

#### MD-7: `_train_quantum` monkey-patches `torch.backends.mps.is_available` globally
**File:** `run_matched2000.py:372-389`
**Problem:** `torch.backends.mps.is_available = lambda: False` mutates a
module-level attribute. If anything inside `train_wgan_gp` queries this
in a thread that isn't waiting on the try/finally (e.g., a callback
spawned thread), it sees the patched value. In the xargs sweep this is
safe (separate processes), but the pattern is global mutation by
convention, not contract. The `# type: ignore[assignment]` comments
acknowledge the type-system violation.
**Impact:** Fragile pattern. A future addition of any concurrent
worker inside training would race.
**Fix:** Refactor `train_wgan_gp` to accept an explicit `device`
override argument so callers can force CPU without global patching.
(Out-of-scope per byte-freeze, but document the patch in a `THREADSAFETY.md`.)

#### MD-8: `framework_versions.json`, `classical_architectures.json` use deprecated `datetime.utcnow()`
**File:** `run_circuit_diagrams.py:564`,
`run_classical_arch_extract.py:388`,
`run_framework_versions.py:82`
**Problem:** `datetime.datetime.utcnow()` is deprecated in Python 3.12+
(PEP 685, DeprecationWarning since 3.12, scheduled for removal). On
modern interpreters this emits a warning that could pollute stdout in
CI.
**Impact:** Forward-compatibility hazard.
**Fix:** Use `datetime.datetime.now(datetime.timezone.utc)`.

#### MD-9: `_log_return_rows` in headline uses overlapping rolling-window stride for ACF references but non-overlapping for DTW
**File:** `run_canonical_headline.py:325, 333-339`
**Problem:** `acfs = np.stack([compute_acf(w, nlags=NLAGS) for w in win])`
operates on `win` (synth log_return windows, generated). The
**real-log-return ACF reference** in `run_figure_suite.render_acf_comparison`
line 308 uses `rolling_window(..., stride=2)` — overlapping. The
**real-log-return DTW reference** at line 336-339 uses non-overlapping
windows (`real_log_delta.reshape(n_real_w, win.shape[1])`). The
inconsistency is documented in the dualscale driver but not loudly
flagged.
**Impact:** ACF and DTW for the same data scale use different real-window
constructions. Statistically defensible but a reviewer may probe the
choice; the manuscript should disclose.
**Fix:** Document both windowing strategies in the methods JSON, with
rationale (DTW pairs avoid overlap; ACF benefits from more windows).

#### MD-10: `RESULTS.mkdir(parents=True, exist_ok=True)` is missing from several emitters
**File:** `run_model_info.py:765, 776, 779`,
`run_classical_arch_extract.py:482`,
`run_framework_versions.py:94`,
`run_methods_full.py:568`
**Problem:** These emitters write to `RESULTS / "*.json"` or `DOCS /
"*.md"` without first ensuring the directory exists. On a fresh
clone where neither `results/` nor `docs/` exists,
the writes raise `FileNotFoundError`. The peer drivers
(`run_canonical_headline.py:581`, `run_matched2000_dualscale.py:589`,
`run_recover_canonical.py:273`) DO `mkdir`.
**Impact:** Inconsistent robustness; fresh-clone or CI from scratch
fails.
**Fix:** Add `RESULTS.mkdir(parents=True, exist_ok=True)` and
`DOCS.mkdir(parents=True, exist_ok=True)` at the top of each `main()`.

### LOW / Style (nice to have)

#### LO-1: `quantum.py:87` uses bare `assert`
**File:** `core/models/quantum.py:87`
**Problem:** `assert window_length == 2 * num_qubits, ...` — under
`python -O`, this is stripped. The repo's stated discipline is explicit
`raise AssertionError`. File is byte-frozen so out-of-scope for change,
but worth flagging as a PRE-EXISTING WEAKNESS.

#### LO-2: methods_full.py docstring claims `import yaml` is the only non-stdlib top-level — but yaml is never imported
**File:** `run_methods_full.py:38-42`
**Problem:** Comment block lines 38-42 documents `import yaml` as the
non-stdlib allowance. The file does NOT actually `import yaml`. Comment
is misleading.
**Fix:** Remove the comment block or remove the yaml claim.

#### LO-3: `_resolves` is O(N×M) over JSON-blob text and unresolved tokens
**File:** `verify_number_provenance.py:118-141`
**Problem:** For each token, iterates every JSON blob's text and every
numeric in it. Out-of-scope per v1 review (performance), flagged
because of CR-5 — tightening resolution may exacerbate cost.

#### LO-4: `head_OD_mean` reference line drawn vertically in OD-distribution overlay panel
**File:** `run_figure_suite.py:1494-1495`
**Problem:** `axA.axvline(head_OD_mean, ...)` at the headline's OD-scale
**moment_mean**. The panel x-axis is OD value (distribution). The
vertical line marks the headline's MEAN OD — sensible — but the legend
elsewhere is "headline EMD reference line" which would mislead a
reader to expect the line marks an EMD value.
**Fix:** Add an in-figure caption noting the row-A reference line is
the headline OD mean, not EMD.

#### LO-5: `_strip_identifiers` does not handle citations without trailing paren
**File:** `verify_number_provenance.py:73`
**Problem:** `r"\b\d{4}\b(?=\s*\))"` only strips years followed by `)`.
Inline references like `Gulrajani et al. 2017,` would not strip 2017,
and the gate then requires 2017 to resolve to a JSON.
**Fix:** Extend pattern to match `r"\b\d{4}\b(?=[\s.,)])"` so years
followed by any reference punctuation are stripped.

#### LO-6: `top-level "dtype" in headline_canonical.json` is param dtype, not sample dtype
**File:** `run_canonical_headline.py:558`
**Problem:** `"dtype": actual_dtype` where `actual_dtype = "torch.float32"`
(the param dtype). The methods_full.py docstring explicitly states
`dtype_params` and `dtype_samples` are DISTINCT. The headline JSON only
emits one `dtype` field — ambiguous.
**Fix:** Rename top-level `dtype` to `dtype_params` and add a
`dtype_samples = "torch.float64"` field; update consumers
(`run_model_info.py:689`).

#### LO-7: `r_min` / `r_max` claim "pipeline-shape constants, not standardization stats" — technically false
**File:** `run_canonical_headline.py:492-498`
**Problem:** Comment claims `r_min`/`r_max` are pipeline-shape constants.
They are downstream of `forward_logreturns(od)`'s mu/sigma — they DO
depend on the standardization stats. They only happen to equal the
checkpoint's stored r_min/r_max because stored mu/sigma == fresh
mu/sigma (verified by the cross-check at lines 409-414).
**Fix:** Reword the comment to clarify that `r_min`/`r_max` are
DERIVED from the pipeline-B standardization, and that the cross-check
above guarantees consistency.

### INFORMATIONAL (positive findings + open questions)

#### IN-1: Excellent explicit-raise discipline throughout new code
The Phase-14 emitters consistently use `raise AssertionError(...)` instead
of bare `assert`, surviving `python -O`. The pattern is documented in
nearly every docstring and the citation (`run_multiseed_rollup.py:86-92`
idiom) is consistent. The only bare-assert in scope is the pre-existing
quantum.py:87 (LO-1).

#### IN-2: The xargs -P 2 + flock pattern is correctly implemented
The sweep harness's parallelism is process-level (not in-process Python),
flock is on an explicit FD (9), `update_status` uses tempfile + rename for
atomicity, the status JSON is rebuilt from scratch each write rather than
patched. This is robust against partial writes and concurrent workers.

#### IN-3: The classical architecture extractor's docstring-fallback path IS present and consistent
Per the user's question — for WGAN-MLP/CNN/LSTM (functional API, no
nn.Module submodules), the docstring-fallback layer trees ARE present at
`run_classical_arch_extract.py:118-215` (`_wgan_mlp_layers`,
`_wgan_cnn_layers`, `_wgan_lstm_layers`) and the slice indices in each
function are documented to match the source-of-truth classical.py
docstrings (lines 60-66, 104-109, 152-160). The drift gate
(`_drift_check`, line 303-328) cross-checks the walked totals against
`model_info.json`'s `parameter_count`. The fallback path is honest and
auditable.

#### IN-4: The headline-vs-reproduction conflation guard IS enforced — except in render_cross_model_emd (CR-2)
The conflation guard D-14-10 is correctly enforced in every other
emitter I examined: `run_matched2000._strict_accept` line 702 enforces
`source == "matched2000_reproduction"` for `iqp_sel_55_repro`;
`run_matched2000_dualscale._headline_rows` line 448 asserts
`source == HEADLINE_SOURCE`; `run_model_info._build_model_record`
emits separate rows for `iqp_sel_55_headline` and `iqp_sel_55_repro`;
the figure suite's dual-scale renderer (`render_matched2000_dualscale_sidebyside`)
overlays the headline as a distinct dashed line + diamond marker. The
**single defective render** is `render_cross_model_emd` (CR-2), where
the bar metric and the reference-line metric are silently different.

#### IN-5: The strict-accept gate enforces 6½ of 7 stated checks
The 7-check claim (D-14-13) decomposes to: (1) seed ∈ {42..46} ✓,
(2) data_hash == frozen ✓, (3) epochs == 2000 + no early-stop ✓,
(4) device-manifest backend_assertion == "PASSED" ✓ but **does NOT
verify the manifest's contents** (CR-4), (5) long-form schema fields
present ✓, (6) 5-file bundle non-empty ✓, (7) `iqp_sel_55_repro`
source == matched2000_reproduction ✓. The "½" deduction is for
check (4)'s shallow correctness: it confirms the field is "PASSED"
but the field is hardcoded "PASSED" for the classical path (CR-4).

#### IN-6: The cross-artifact data_hash gate IS enforced in run_methods_full + run_matched2000_dualscale but is WEAKER in run_model_info (HI-3)
The user explicitly asked whether the gate is "actually equivalent across
emitters". Answer: NO — `run_matched2000_dualscale.verify_data_hash`
and `run_methods_full._data_hash_gate` both compare against the literal
`91e447d4624e25b3`; `run_model_info` only checks mutual equality. The
asymmetry is HI-3.

#### IN-7: Open question — does `is_complete` strict-accept subprocess inherit env modifications?
The sweep harness exports `export -f is_complete` (line 353) and
calls `${PYTHON} -m ${RUN_MODULE} --accept ...` from inside the
function. If the parent shell has set any env vars (e.g.,
`PYTHONHASHSEED`), they propagate. Could not determine without running
the sweep whether the env is hygienic across workers. Recommend
documenting required env (PYTHONHASHSEED, OMP_NUM_THREADS) in the sweep
script header.

#### IN-8: Open question — `head_emd` from `checkpoint_emd` field
`render_training_convergence_all_models` plots `head_emd =
checkpoint_emd` at the headline's epoch 1969. The `checkpoint_emd`
field was saved by `EarlyStopping._save_checkpoint` (training.py:151-160)
as the EarlyStopping `best_emd` — which IS in the same metric space as
the matched-2000ep runs' `emd_avg` trajectories (both are
`compute_emd` on a single training-eval batch). This particular figure
IS comparing apples-to-apples (unlike CR-2). Worth confirming with a
sanity check that the checkpoint_emd was computed in the same way the
matched-2000ep runs compute their per-epoch eval EMD.

#### IN-9: The `iqp_sel_55` (55-param) bounds in quantum.py forward circuit are correct
Manual trace of the index walk for the iqp_sel_55 circuit (5q × 3L,
55 params) confirms idx reaches exactly 55 by the end of the
Step-5 final-RX loop. The `if idx + 2 < len(params_pqc)` guards are
correctly tight (idx+2 < 55 ⇒ idx ≤ 52 ⇒ params[52..54] valid). The
`if idx < len(params_pqc)` guards for the final RX-only branch are
correctly tight (idx < 55 ⇒ params[idx] valid). For the default_75
75-param circuit (5q × 4L), idx reaches exactly 75 by the final RY.
**No off-by-one bugs in the bounds checks.**

#### IN-10: The "best EMD over trajectory" vs "final-eval EMD" distinction is DOCUMENTED but the docs do not block the misuse
The user noted this confusion. The reconciliation_note.md generation
(`run_model_info.py:222-303`) cites `emd_avg[-1]` (final-eval) in its
"new basis" string for the 1000ep→2000ep delta. The cross-model EMD
figure (`render_cross_model_emd`) uses `np.min(emd_avg)` (best over
trajectory). **Both quantities are computed, both are exported in
different artifacts, and there is NO emitter-level enforcement that a
manuscript number stays consistent across docs.** A reviewer could
trivially produce a contradictory citation. Recommend either dropping
one of the two quantities or emitting both with explicit suffixes
(`emd_best_over_training`, `emd_final_eval`) so the verify gate can
catch a doc that cites one and renders the other.

## File-by-file notes

**`run_recover_canonical.py`** — Sound. The decomposition arithmetic
gate (line 226-234), the shape gate (line 220-224), and the equivalence
gate (line 286-389) all use the explicit-raise idiom. The repo-root +
gitignored-checkpoint resolver is robust (handles main checkout + git
worktree). Only nitpick: `_provenance` reads `g_pg["lr"]` etc. unguarded
— if `c_optimizer.state_dict()` is malformed, `KeyError` is raised
without a clear message.

**`run_canonical_headline.py`** — Mostly sound; the headline is
properly gated by checkpoint sha256 equality, mu/sigma equality, shape
equality, structural forward-pass equality, and explicit device/dtype
manifest. **HI-1** (hardcoded DTW seed), **LO-6** (param dtype recorded
as just `dtype`), and **LO-7** (misleading r_min/r_max comment) are the
real defects. The `_log_return_rows` DTW non-overlapping windowing is
mathematically equivalent to the matched2000_dualscale driver — good.

**`run_matched2000.py`** — The classical/quantum
asymmetric device handling is **CR-4**, the dominant finding. The
strict-accept gate is otherwise well-engineered: explicit-raise on every
check, bundle file presence check, schema completeness check. **HI-4**
(hardcoded topology) is a single-source-of-truth violation. The 7-check
gate (D-14-13) is honest about being 7 checks but **check #4 is shallow**
— the manifest's `backend_assertion` field is unconditionally PASSED for
the classical branch.

**`run_matched2000_sweep.sh`** — Strong. The flock/xargs/atomic
status pattern is correct (IN-2). The `is_complete` call inside
`is_complete` (line 207-208) effectively re-runs the python strict gate
for every check — performance concern (HI-9) not correctness. The
guardrail at line 169-177 correctly rejects `--parallel >= 3`. The
sweep_status.json is rebuilt from scratch per write (line 261-264), so
partial-writes are safe.

**`run_model_info.py`** — The cross-artifact data_hash gate
is WEAKER than peer emitters (**HI-3**). The hardcoded `optimizer_betas`
field for VAE/AR (**HI-2**) is a provenance lie. The `_dataset_block`
derivation (line 355-391) is honest — every count is derived from
data.csv + the locked window config, never hand-typed. The reconciliation
note is correctly built FROM JSON only (no recompute).

**`run_figure_suite.py`** — The biggest single source of
findings. CR-1 (non-deterministic seed), CR-2 (scale conflation), and
MD-3 (hardcoded epoch). The Plan 14-10 figures (lines 1064-2214) are
mostly defensive — loud-fail on missing input, distinct headline
marker, dynamic neg-r2 caption — but their scale of code (~1100 lines)
makes them hard to review in a single pass. The pattern of using
`_load_json` + `_require` with explicit FileNotFoundError is consistent
and good. The companion JSON-per-figure scheme is well-executed (every
plotted value is recorded, every source artifact is named).

**`run_matched2000_dualscale.py`** — Strong. Cross-artifact
data_hash gate is the strongest of any emitter (verifies recomputed
hash + every config + headline). The headline rows ARE emitted as a
distinct model_kind. **HI-5** (model_kinds list excludes headline) is
the only finding here. The DTW recipe is verbatim with
`run_dualscale_fidelity` so 1000ep/2000ep numbers reconcile.

**`run_circuit_diagrams.py`** — Sound. The param-count
consistency gate (build_config_locks) collects ALL mismatches into a
single explicit raise — operator sees every offending variant.
qml.draw_mpl under torch.no_grad is the right render-only pattern.
MD-8 (utcnow deprecation) is the only finding.

**`run_classical_arch_extract.py`** — Sound. The drift gate
against model_info.json IS implemented (line 303-328), with the
functional-API docstring fallback for WGAN-MLP/CNN/LSTM (IN-3). The
extractor never calls model fit / sample / checkpoint reload — pure
introspection. MD-8 (utcnow) is the only nit.

**`run_framework_versions.py`** — Trivial and correct. Pure
introspection over `importlib.metadata`. Handles
`PackageNotFoundError` by emitting None. MD-8 (utcnow) applies.

**`run_methods_full.py`** — **CR-3** (hardcoded line citations)
is the only critical defect. The text-only citation extraction
(`_first_lineno` + `_citations`) is otherwise the right pattern. The
dtype_params / dtype_samples split into TWO DISTINCT fields is good
(buckets.4_hardware_software). The LaTeX equation strings (lines
270-288) are NOT verified against any JSON — they would not pass the
number-provenance gate on numerals but rely on the gate's
identifier-strip patterns. LO-2 (misleading docstring about yaml) is
the only stylistic nit. MD-10 (no `RESULTS.mkdir`) applies.

**`verify_freeze_ready.py`** — **HI-6** (glob depth mismatch
with verifier) and **MD-5** (verify script mutates working tree) are
the meaningful findings. MD-6 (only *.pt/*.pth size-checked) is a
hardening gap. The git subprocess wrapper ignores returncode/stderr —
silent failure if git is unavailable or the repo is corrupt.

**`verify_number_provenance.py`** — **CR-5** (false-positive
resolution) is the dominant finding. The substring match + truncated-string
float comparison admit broad false positives that materially weaken the
"every number traces to a JSON" guarantee. The gate is the executable
enforcement of success-criterion 5, but its current resolution model is
too lax. LO-5 (citation-year strip pattern) and LO-3 (O(N×M) complexity)
are secondary.

**`core/models/quantum.py`** (byte-frozen, pre-existing only)
— LO-1 (bare assert). The bounds-check arithmetic for both circuit
variants is correct (IN-9). The `_introspect_circuit` mirrors
`generator_circuit` Steps 1-5 verbatim. The `par_light` hook is a
documented no-op.

**`core/models/classical.py`** (byte-frozen, pre-existing only)
— Sound. The `params_pqc` flat-tensor pattern with manual slicing into
F.linear / F.conv_transpose1d / functional LSTM is the right way to
satisfy the `optim.Adam([generator.params_pqc])` single-tensor contract
(training.py:297). Param counts (74/73/78) are consistent with the
docstring layouts and with the classical_architectures.json extractor.

**`core/training.py`** (byte-frozen, pre-existing only) —
MD-1 (stale `real_log_returns` reference in eval phase). The
`compute_dtype` split (line 268) is the canonical CPU-float64 /
MPS-float32 source of truth. The `_load_checkpoint` device/dtype
re-mapping in EarlyStopping (CR-02 fix at lines 162-209) IS correct;
the fix is documented and the prior-phase early-stopped reproducibility
delta is disclosed in the docstring. The `_spectral_psd_loss`
device-fix (CR-01 at lines 504-549) IS correct and gated by
`spectral_loss_weight > 0` (default OFF).

**`core/eval.py`** (byte-frozen, pre-existing only) — Sound.
The notebook-parity decisions (Fisher kurtosis, ddof=0 std, raw-samples
EMD) are documented and traced to notebook cell numbers. `compute_dtw`
uses fastdtw + euclidean. All functions are pure / stateless. No
defects within scope.

