---
reviewer: code-quality-r2
scope: Phase 14 plan 14-13 remediation sweep (commits 4ea576b…36eeabf)
files_audited: verify_number_provenance.py, run_methods_full.py,
  run_matched2000.py, run_matched2000_dualscale.py,
  run_canonical_headline.py, run_figure_suite.py,
  run_model_info.py, verify_freeze_ready.py,
  checkpoints/best_checkpoint.pt, requirements-pinned.txt,
  results/manuscript_apparatus_constants.json,
  results/reconciliation_deltas.json,
  results/total_adversarial_param_budget.json,
  docs/methods_full.md, docs/reconciliation_note.md,
  docs/paper_blocks_framing.md, core/ (delta check)
created: 2026-05-20T00:00:00Z
---

# Code Review R2 — Phase 14 Plan 14-13 Remediation Audit

## Summary verdict

**PASS-WITH-FINDINGS** — proceed to paper resubmission, but with three
findings ranked at HIGH that should be triaged and a fourth (the gate v2
sign-flip false-positive) that should be hot-fixed before the v2.0
Zenodo tag is cut. The 14-13 sweep is substantive and competent work
that closes the core of the original 28 findings. The two remediation
artifacts that were emitted to satisfy the gate (`reconciliation_deltas.json`,
`total_adversarial_param_budget.json`) are legitimate derived audit
artifacts, not back-fits. D-14-22 (`core/` byte-freeze) is
**literally** honored — `git diff 06bb470..main -- core/`
returns zero bytes. But the gate v2 rewrite still admits some false
positives, the `training_time_device` field captures the
post-sample-generation device (not training-time), and the f-string emit
in `run_methods_full.py` now cites the wrong line number for the
generator `*0.1` cast.

## Methodology

I read all 7 remediation commits via `git show <sha>`, then re-read the
final state of each modified file, then ran the gate against every
paper-facing doc (`methods_full.md`, `reconciliation_note.md`,
`paper_blocks_framing.md`), enabled `--manifest` to inspect which JSON
key each numeric literal resolved to, and ran `verify_freeze_ready.py`
end-to-end. I also ran a battery of regex edge-case tests against the
gate v2's `_resolves` boundary regex and `_ID_PATTERNS` strip patterns.
I verified the checkpoint sha256 against the lock JSON
(`f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082`
matches `canonical_config_lock.json#checkpoint_sha256` — correct).

## Findings by severity

### HIGH (should be addressed before v2.0 tag)

#### R2-HIGH-1: Gate v2 sign-flip false positive — positive token resolves to negative JSON value
**File:** `verify_number_provenance.py:193`
**Evidence:**
```python
boundary_re = re.compile(rf"(?<![\d.]){re.escape(token)}(?![\d])")
```
The lookbehind class `[\d.]` **does not include `-`**, so a positive
token like `0.0001` extracted from a doc matches the substring `0.0001`
inside the corpus value `-0.0001`.

Empirical test:
```
boundary_re = re.compile(r'(?<![\d.])0\.0001(?![\d])')
boundary_re.search('"x": -0.0001')  # → Match (FALSE POSITIVE)
```
The float ε-neighborhood pass-2 fallback DOES respect sign (|+0.0001 −
(−0.0001)| = 0.0002 > tol), but pass-1 (text-match) fires first and
returns the match. Result: the gate cannot distinguish positive from
negative values when the token is a substring of the negative form.

**Impact:** A doc that erroneously cites `+0.058710` (a sign-flipped
WGAN-CNN delta) would resolve to the JSON's `-0.058710` and the gate
would silently PASS. The reconciliation deltas in the new
`reconciliation_deltas.json` are all small negative numbers — exactly
the class where sign matters. Combined with R2-HIGH-2 below, the gate
cannot certify reconciliation_note.md's sign-direction claims.

**Recommended fix:** Tighten lookbehind to `(?<![\d.\-+])` AND treat the
leading sign as part of the token in `_NUM`. The current `_NUM` regex
already captures `-0.0001` as one token; the gate just doesn't enforce
sign at match time.

#### R2-HIGH-2: Gate v2 ε-neighborhood still admits broad false positives at low precision
**File:** `verify_number_provenance.py:215-237`
**Evidence:** With `--manifest`, methods_full.md's `-0.26` literal
(documenting the AR(p) ML-bias factor `(777-2)/777 - 1 ≈ -0.26%`)
resolves to:
```
docs/methods_full.md:-0.26 -> results/ansatz_comparison.json#rows[278].value
```
where `rows[278].value = -0.2570036284193482` — a `kurt_fake` value for
V3 quantum seed 45. With `prec=2`, the tolerance is `10^-2 / 2 = 0.005`;
|−0.26 − (−0.257)| = 0.003 ≤ 0.005, so it matches. The gate "resolves"
the AR-bias literal to an unrelated kurtosis value by ε-coincidence.

Reconciliation_note.md has the same pattern: `0.025740` (VAE old_1000ep)
resolves to `multiseed_summary.json#rollup[256].mean` and `0.027586`
(iqp_sel_55 old_1000ep) resolves to a per-sample WGAN-LSTM introspection
value `figures/_introspect_wgan_lstm.json#snapshots[3].samples[11][7]` —
both ARE the right value at 6-dp precision but the **wrong artifacts**.
The gate provides "some JSON has this value" guarantees, not
"the value resolves to its semantically correct source."

**Impact:** v2 is materially tighter than v1 (substring `0.6843` inside
`0.03006843578` is gone) but the gate is still not a
semantic-resolution layer. A reviewer reading the manifest will see
several literals pointing at semantically-unrelated artifacts. The gate
PASSES but the manifest doesn't certify the citations are correct.

**Recommended fix:** This is fundamental to the gate's text+epsilon
resolution model and cannot be cheaply fixed at v2. Disclose the
limitation in the gate's docstring. For a v3 of the gate, require
per-literal `--expected-source` annotations in the doc (e.g.,
`<!-- prov: methods_full.json#buckets.5_reproducibility.data_hash -->`)
so the gate can verify the key path the author intended.

#### R2-HIGH-3: `training_time_device` field captures post-sample-generation device, not training-time
**File:** `run_matched2000.py:331-337` (_device_manifest);
`run_matched2000.py:570` (_train_wgan call);
`run_matched2000.py:257-270` (generate_wgan_samples)
**Evidence:** `_train_wgan` calls `generate_wgan_samples(generator, ...)`
**before** `_device_manifest(generator)`. `generate_wgan_samples` at
line 270 explicitly does `generator = generator.to("cpu")`. The
`_device_manifest` then inspects `next(generator.parameters()).device`
which always returns `"cpu"` regardless of where training ran.

```python
def generate_wgan_samples(generator, n: int, seed: int):
    ...
    generator = generator.to("cpu")   # ← line 270; mutates generator in place
    ...
# _train_wgan, line 569-570:
samples = generate_wgan_samples(generator, n_synth, seed)
...
"device_manifest": _device_manifest(generator),   # generator is now on CPU
```
The "future-gate" promise — that `_strict_accept` will refuse runs that
trained on MPS — is broken. The MPS-disable hook (the
`torch.backends.mps.is_available = lambda: False` patch) IS the real
protection, but the post-training inspection that the gate keys on
cannot detect violation of that protection.

**Impact:** A future operator who accidentally removes the MPS-disable
hook in `_train_wgan` would have training run on MPS, but the
post-training device manifest would still read `cpu` (because
`generate_wgan_samples` moves the generator). `_strict_accept` would
PASS. The asymmetry the executor is trying to prevent (CR-4 from the v1
review) is not actually gated.

**Recommended fix:** Either (a) snapshot the training-time device
BEFORE `generate_wgan_samples` is called (e.g., capture
`training_device = next(generator.parameters()).device` immediately
after `train_wgan_gp(...)` returns); or (b) have `train_wgan_gp` itself
record the device it ran on into the returned metrics dict, and gate
on that. (b) is cleanest but requires either a `core/` change
(D-14-22 blocked) or a post-hoc wrapper that reads from a side-channel.
Documenting this as a "trust the hook, not the post-hoc inspection"
caveat is acceptable as long as the operator-facing comment on lines
325-329 of `run_matched2000.py` (which claims this reads back as `cpu`
"symmetrically across all training paths") is corrected — that claim
is true accidentally (because `generate_wgan_samples` moves to CPU),
not because of the MPS-disable hook.

#### R2-HIGH-4: `_train_vae` MPS-disable hook is purely cosmetic
**File:** `run_matched2000.py:625-660`
**Evidence:** `_train_vae` adds the `torch.backends.mps.is_available =
lambda: False` patch (lines 625-626) and the `try: ... finally: orig_mps`
restore. But `VAEBaseline.__init__` in
`core/models/nonadversarial.py` never queries
`torch.backends.mps.is_available()` — the VAE is constructed and trained
on whatever device the default torch context is (CPU by default), and
the model never moves devices. The MPS-disable hook protects nothing
for VAE.

```
$ grep -n 'mps\|device\|to(' core/models/nonadversarial.py
115:        device = self.dec_h.weight.device
116:        z = torch.randn(n, self.LATENT_DIM, generator=gen, device=device)
```
The only device-related code in the VAE module is reading the device
from existing parameters in `sample()` — which is CPU because the
parameters were never moved.

**Impact:** The patch in `_train_vae` is documentation-only (a no-op).
A future operator who adds device selection to VAE training code (e.g.,
`vae = vae.to(device)` where `device = torch.device("mps" if
torch.backends.mps.is_available() else "cpu")`) WOULD be protected by
the hook, but the gate at `_strict_accept` line 791 (the
`training_time_device` check from R2-HIGH-3) cannot certify this. The
hook is a defense-in-depth measure not connected to the gate.

**Recommended fix:** Document the hook as defensive-future-proofing
(not a current protection) in the `_train_vae` docstring. Alternatively,
add a real protection — e.g., explicitly `vae = vae.to("cpu")` at the
end of training so the device manifest snapshot is honest, or wire the
gate to check that the MPS-disable hook was actually applied during
training (via a sentinel field in the metrics dict).

### MEDIUM (should be addressed before resubmission, not blocking)

#### R2-MED-1: CR-3 fix cites the line PRECEDING the cast, not the cast itself
**File:** `run_methods_full.py:131`
**Evidence:** The `_citations` target pattern is:
```python
"generator_to_compute_dtype": "generated_samples = generator",
```
This matches `core/training.py:346` (`generated_samples =
generator(noise_batch)`). But the **cast** the citation describes
(`.to(compute_dtype) * 0.1`) is on **line 347**:
```
346:                generated_samples = generator(noise_batch)
347:                generated_samples = generated_samples.to(compute_dtype) * 0.1
```
The emitted methods_full.json text reads:
```
"... core/training.py:346 generator output cast .to(compute_dtype) * 0.1 ..."
```
Citation is one line off from the code it describes. The docstring
comment at `run_methods_full.py:126-128` acknowledges this is the
"immediately-PRECEDING line" of the cast chain — so the author was
aware. But the emitted text still incorrectly attributes the cast to
line 346.

**Impact:** Documentation drift. A reviewer who clicks into the cited
line sees a `generator(noise_batch)` call, not the cast. The whole
point of CR-3 was to prevent this kind of drift.

**Recommended fix:** Change the pattern to match line 347 directly:
```python
"generator_to_compute_dtype": "generated_samples = generated_samples.to(compute_dtype)",
```
This anchors the citation on the cast line itself.

#### R2-MED-2: `_finite_sanitize` silently stringifies numpy arrays
**File:** `run_matched2000.py:117-148`
**Evidence:** Lines 139-148:
```python
try:
    v = float(obj)
    ...
except (TypeError, ValueError):
    return str(obj)
```
If `obj` is a numpy array with shape > 0 (e.g., a per-epoch loss
trajectory), `float(obj)` raises `TypeError`, and the fallback returns
`str(obj)`. The JSON then contains the numpy array's repr as a string
field — silently lossy.

**Impact:** Currently triggered only if a metrics dict accidentally
contains an unflattened numpy array (none observed in the current
pipeline). Latent footgun.

**Recommended fix:** Add an `isinstance(obj, np.ndarray): return
[_finite_sanitize(v, _stats) for v in obj.tolist()]` branch BEFORE
the `try float()` fallback, OR raise loudly with a clear error.

#### R2-MED-3: `verify_freeze_ready.gate_b` checks only 3 of 9 paper-facing docs
**File:** `verify_freeze_ready.py:64-68`
**Evidence:**
```python
PAPER_BLOCKS = [
    REPO_ROOT / "revision" / "docs" / "paper_blocks_framing.md",
    REPO_ROOT / "revision" / "docs" / "paper_blocks_refs_methods.md",
    REPO_ROOT / "revision" / "docs" / "reviewer_response.md",
]
```
The gate v2 is being PASSED on 9 paper-facing docs per the 14-13
commit messages (paper_blocks_framing, paper_blocks_refs_methods,
reviewer_response, methods_full, circuit_atlas,
completeness_sweep_manifest, training_protocol, dataset_stats,
reconciliation_note). But the freeze-ready gate only enforces 3 of
them. methods_full.md, reconciliation_note.md, training_protocol.md,
circuit_atlas.md, dataset_stats.md, and completeness_sweep_manifest.md
could regress at any time without triggering the freeze gate.

**Impact:** Provenance enforcement is asymmetric across paper-facing
docs at freeze time. The v2 gate PASSES today on all 9, but the
freeze gate only enforces 3.

**Recommended fix:** Extend `PAPER_BLOCKS` to cover all 9 docs, or
extract a `PROVENANCE_GATED_DOCS` list that both 14-13 SUMMARY's
verify section and `verify_freeze_ready.py` import from.

#### R2-MED-4: `_ID_PATTERNS` `line ~?N` strip can swallow legitimate data
**File:** `verify_number_provenance.py:114`
**Evidence:** Pattern `\bline\s*~?\d+(?:-\d+)?\b` strips any `line N`
or `line ~N` occurrence — including legitimate data literals. Test:
```
re.sub(r'\bline\s*~?\d+(?:-\d+)?\b', ' ', 'see line 1969 of canonical_recovery')
# → 'see   of canonical_recovery'   ← 1969 gone, NOT gated
```
If a doc says "epoch 1969 (the checkpoint epoch)" — gated. If a doc
says "line 1969 of the trajectory" — NOT gated. The strip is
context-blind.

**Impact:** Latent over-strip. The pattern was added (per executor
deviation note #1) specifically to handle prose `line ~148` references,
but it admits any `line <NNN>` form including data-literal `line 1969`.

**Recommended fix:** Tighten the pattern to require a colon, prose
verb, or file path within ±20 chars of the match — or restrict to
4-digit-or-less numbers that are clearly source-line citations.

#### R2-MED-5: `_train_vae` integer-shape numpy/random/torch seeding doesn't cover Adam state
**File:** `run_matched2000.py:601-605`
**Evidence:** HI-7 fix adds `np.random.seed(seed)` and `_random.seed(seed)`
to `_train_vae`. But `torch.optim.Adam` initializes its first/second
moment state from zeros — no RNG dependence. The seeding fix is
defensive (covers future code that uses np/random in the VAE training
loop) but does NOT change current run output: VAE is already
deterministic via `torch.manual_seed(seed)` alone (VAE
reparameterization uses `torch.randn`, the data loader is
`shuffle=False`, Adam state is zero-initialized).

**Impact:** HI-7 fix is correctly precautionary but the v1 review's
claim that VAE training was "harmlessly under-seeded" is accurate.
The fix doesn't change any current artifact byte. This is fine — but
the 14-13 commit message implies the fix closes a determinism gap that
in practice was not open.

**Recommended fix:** None functionally; the fix is good practice.
Reframe the commit message to "defense in depth" rather than "fix" if
the SUMMARY.md is updated.

#### R2-MED-6: Apparatus-constants JSON is a back-fit to satisfy the gate
**File:** `results/manuscript_apparatus_constants.json`
**Evidence:** The JSON's own `note` field says:
> "Emitting them here lets the v2 provenance gate resolve them as
> legitimate manuscript-context literals rather than via substring
> coincidence in unrelated JSON values."

The values (20L, 300L, 880mm, 120, 6, 10) describe a photobioreactor
the manuscript references in BEFORE blocks. The artifact exists solely
to make the gate happy. That said: the values ARE real apparatus
constants quoted verbatim from the LaTeX source, not invented. The
JSON is honest about its provenance role.

**Impact:** Aesthetic concern, not a correctness bug. The pattern is:
"the gate found a literal it doesn't have a JSON for; emit a JSON
to make the gate pass." Acceptable when the literal is a legitimate
context constant (as here), unacceptable if it becomes a generic
escape hatch.

**Recommended fix:** None. Document the pattern as "manuscript-context
JSON" in the gate's docstring so future operators don't abuse it for
results values.

#### R2-MED-7: `optimizer_betas` in bucket_3 still reads from WGAN entry only
**File:** `run_methods_full.py:441`
**Evidence:**
```python
repro = mi_by_model.get("iqp_sel_55_repro", {})
bucket_3 = {
    "optimizer": "Adam",
    "optimizer_betas": repro.get("optimizer_betas", [0.0, 0.9]),
    ...
}
```
The HI-2 fix in `run_model_info.py` correctly sets `optimizer_betas =
None` for VAE/AR, but `methods_full.json#buckets.3_training.optimizer_betas`
unconditionally pulls from the WGAN repro entry. Since bucket_3 is
documented as the WGAN-GP training protocol, this is internally
consistent, but the methods_full.md table at line 259 reads
`| Optimizer betas | [0.0, 0.9] |` without family qualification — a
reader could interpret it as applying to all baselines.

**Impact:** Minor documentation precision gap. Doesn't affect the gate
or any artifact byte.

**Recommended fix:** Clarify the methods_full.md row to read
`Optimizer betas (WGAN-GP only) | [0.0, 0.9] | ...`.

### LOW

#### R2-LOW-1: Original review's MD-5 (verify mutates working tree) not addressed
The `verify_freeze_ready.gate_a_gitignore_archive()` still appends to
`.gitignore` and runs `git add -f`. The original review flagged this as
MD-5. The 14-13 sweep extended the negation pattern to recursive
(`!results/**/*.json`) but did NOT add a `--check-only` flag.
A CI run that exercises the verify will still leave dirty files.

#### R2-LOW-2: Original review's MD-6 (large `.npz` not size-checked) not addressed
`gate_c_tag_scope` filters by `.endswith((".pt", ".pth"))` only. The
14-13 sweep did not extend this to other large-data extensions
(`.npz`, `.npy`, `.csv` > threshold). Out of scope for 14-13 but worth
flagging that the original MD-6 is still open.

#### R2-LOW-3: Original review's MD-8 (deprecated `utcnow`) not addressed
`run_circuit_diagrams.py:564`, `run_classical_arch_extract.py:388`,
`run_framework_versions.py:82` still use `datetime.utcnow()`. The 14-13
sweep added `data_hash` to these emitters' outputs but did not address
the deprecation. Forward-compat hazard on Python 3.13+.

#### R2-LOW-4: Original review's MD-7 (global mutation of `torch.backends.mps.is_available`) widened
The MPS-disable hook is now applied in **3 places**
(`_train_quantum`, `_train_wgan`, `_train_vae`) rather than 1. Each
uses the try/finally idiom and is correct for single-process execution.
But the pattern of "globally patch a torch module attribute" is now
the canonical convention for matched-budget training. In a future xargs
sweep that runs multiple model trains in a single process (currently
not the case), the hooks would race. Document the convention in
THREADSAFETY.md.

#### R2-LOW-5: Bracketed-bibref strip can affect JSON-path-style citations
`_ID_PATTERNS` includes `\[\d+\](?:-\[\d+\])?` to strip `[21]` and
`[21]-[23]`. This also strips `aggregates[527]` array-index citations
in prose. Not a correctness bug (the index is a navigation aid, not a
data literal) but a context-blind strip.

### MINOR / stylistic

#### R2-MINOR-1: Gate v2 schema string includes plan id in user-visible PASS message
The `_SCHEMA` string is `"v2 (Phase 14 plan 14-13 — boundary-strict
resolution + render-only exclusion)"`. PASS messages print this verbatim.
After resubmission, the next plan that touches the gate will need to
update the schema string and re-stamp every PASS output. This is a
maintenance burden, not a bug.

#### R2-MINOR-2: `_finite_sanitize` is named `_dumps_finite` in some contexts
The wrapper `_dumps_finite` and the recursive sanitizer `_finite_sanitize`
have similar names — easy to miss the difference when scanning. Cosmetic.

#### R2-MINOR-3: 14-13 SUMMARY.md claims gate v2 PASS on 9 paper-facing docs
The SUMMARY claims gate PASSES on all 9. I verified 3 of them
end-to-end. The other 6 are claimed via the commit message; I did not
run all 9 to cross-check. (Spot-checked: methods_full.md PASS 64
literals, reconciliation_note.md PASS 33 literals, paper_blocks_framing.md
PASS 23 literals — all consistent with the commit messages.)

## Verification of original 28 findings

The original code-review.md catalogued 5 CRITICAL, 9 HIGH, 10 MEDIUM,
and 7 LOW/INFO. I spot-checked the resolution claims:

| Finding | Status | Notes |
|---|---|---|
| **CR-1** non-deterministic timeseries seed | **RESOLVED** | SHA-256 hex[:8] → uint32 seed. Deterministic across invocations. 32-bit entropy is sufficient for selecting `n_show ≤ 8` window indices from ~770 windows; collision risk is negligible. |
| **CR-2** cross_model_emd metric conflation | **RESOLVED** | Rebuilt on OD scale from `matched2000_dualscale.json#rows`, mean ± ddof=1 std, headline reference on same scale. Companion JSON caption explicit about scale and aggregation. |
| **CR-3** hardcoded `training.py:347`/`:259-268` | **PARTIALLY RESOLVED** | Now uses `_citations` for both. But the new target `generated_samples = generator` matches line 346 (the call), not line 347 (the cast). See R2-MED-1 — citation is 1 line off from the code it describes. |
| **CR-4** classical-WGAN silent MPS fallback | **PARTIALLY RESOLVED** | (a) MPS-disable hook IS applied in `_train_wgan` and `_train_vae` — future-gate intent achieved for forward training. (b) But the `training_time_device` field that's supposed to GATE the result captures post-sample-generation device (CPU), NOT training-time. See R2-HIGH-3. (c) Historical asymmetry is disclosed in methods_full.md §4.2 and reviewer_response.md — disclosure portion is sound. |
| **CR-5** substring-match false positives | **MOSTLY RESOLVED** | Boundary regex eliminates the `0.6843`-inside-`0.03006843578` class of false positive. But low-precision float ε-matching still admits coincidental resolutions (see R2-HIGH-2). Sign-flip false positive is new (R2-HIGH-1). |
| **HI-1** hardcoded DTW seed in headline | **RESOLVED** | `generation_seed` threaded through `_od_scale_rows` and `_log_return_rows`. The `* 31` offset is preserved. |
| **HI-2** `optimizer_betas` hardcoded for all models | **RESOLVED** | `betas = None` for non-adversarial families; downstream bucket_3 still reads WGAN entry which is OK because bucket_3 documents WGAN-GP convention. |
| **HI-3** data_hash gate mutual-equality only | **RESOLVED** | `EXPECTED_DATA_HASH = "91e447d4624e25b3"` + explicit-raise added to `run_model_info.py:740`. Gate now enforces equality to the literal. |
| **HI-4** hardcoded topology | **RESOLVED** | Reads `decomp.get("gate_layout", {}).get("entangler", "range")` with a `"range"` fallback default — safe. |
| **HI-5** `model_kinds` excludes headline | **RESOLVED** | `MODEL_KINDS + [HEADLINE_MODEL_KIND]` in the dualscale emitter. |
| **HI-6** freeze gate glob vs verifier rglob | **RESOLVED** | `verify_freeze_ready.py` now uses `RESULTS_DIR.rglob("*.json")` consistently. Negation written as `!results/**/*.json`. |
| **HI-7** `_train_vae` lacks np/random seed | **RESOLVED (defensive)** | Both seeds added. Functionally no current change (see R2-MED-5) but defense-in-depth is good. |
| **HI-8** `default=float` NaN/Inf serialization | **RESOLVED** | `_finite_sanitize` + `_dumps_finite` with 5% threshold. Minor latent footgun with numpy arrays (R2-MED-2). |
| **HI-9** subprocess re-spawn in sweep harness | **NOT ADDRESSED** | Original review flagged this as performance-only. The 14-13 sweep did not touch the sweep harness. Acceptable. |
| **MD-3** hardcoded `head_epoch != 1969` | **RESOLVED** | Reads from `canonical_config_lock.json#checkpoint_epoch`. |
| **MD-5** `verify` mutates working tree | **NOT ADDRESSED** | See R2-LOW-1. |
| **MD-6** large-file check only `.pt`/`.pth` | **NOT ADDRESSED** | See R2-LOW-2. |
| **MD-7** global mutation of `torch.backends.mps.is_available` | **WIDENED** | Now in 3 functions; pattern is the canonical convention. Single-process safe. |
| **MD-8** `datetime.utcnow()` deprecation | **NOT ADDRESSED** | See R2-LOW-3. |
| **MD-10** missing `RESULTS.mkdir` | **PARTIALLY ADDRESSED** | Spot-checking required; the 14-13 sweep did not enumerate this as a target. |
| **LO-1..LO-7** | **NOT ADDRESSED** | LO findings are out-of-scope by definition. |

**Resolution score:** 5 of 5 CRITICAL claimed resolved — 3 fully (CR-1,
CR-2, CR-5 substring), 2 partially (CR-3 line off; CR-4 gate doesn't
work). 8 of 9 HIGH claimed resolved — 8 actually resolved (HI-1..HI-8),
HI-9 deferred. The 14-13 sweep's claim of "27 of 28 findings closed" is
mostly accurate but the two partial resolutions (CR-3, CR-4) should be
counted as "addressed with reservations," not "closed."

## Positive observations

1. **D-14-22 byte-freeze is literally honored.** `git diff 06bb470..main
   -- core/` returns zero bytes. Every "documented not changed"
   commitment is upheld. The §3.x Metric conventions subsection in
   methods_full.md is the right kind of remediation — disclosure rather
   than core modification.

2. **The data_hash explicit-raise pattern is exemplary.** Adding
   `EXPECTED_DATA_HASH = "91e447d4624e25b3"` at module top and a
   loud-failing comparison is exactly the right idiom for dataset
   integrity. The fix is symmetric across `run_model_info.py`,
   `run_matched2000_dualscale.py`, and the new emitters
   (`run_circuit_diagrams.py`, `run_classical_arch_extract.py`,
   `run_framework_versions.py`).

3. **Gate v2 is meaningfully tighter than v1.** The
   `(?<![\d.])token(?![\d])` boundary regex eliminates the entire class
   of substring-coincidence false positives (the original PROV-CRIT-2
   `0.6843` case). Render-only JSON exclusion from the resolution corpus
   eliminates tautological resolution. The `--manifest` flag is a
   significant new operator-facing affordance.

4. **The two new audit JSONs are legitimate.**
   `reconciliation_deltas.json` carries `(old_1000ep, new_2000ep, delta)`
   tuples derived from `matched2000_dualscale.json#aggregates` — not
   invented values. `total_adversarial_param_budget.json` carries
   generator+critic totals derived from already-published lock JSONs
   and `classical_architectures.json`. Both are audit-trail artifacts,
   not result back-fits.

5. **The CR-2 fix is the most important and most clearly correct.**
   The `render_cross_model_emd` rebuild on the OD scale (vs. the
   previous min-over-trajectory log-return-scale conflation) cleanly
   removes the manuscript headline-vs-reproduction confusion. The
   companion JSON's explicit caption ("OD scale, final-eval mean ± std
   over 5 seeds 42-46; frozen headline reference line on the same
   scale") is operator-readable proof.

6. **The 14-13 SUMMARY.md is candid about its 6 Rule 1-3 deviations.**
   The executor honestly disclosed: 3 `_ID_PATTERNS` extensions for
   prose identifier classes the plan didn't enumerate, the apparatus
   constants JSON emit, the reconciliation deltas JSON emit, the total
   adversarial param budget JSON emit, and the latent_dim=4 correction
   in the §3.x.d VAE β derivation. All are sound auto-fixes that I
   would have applied under the same constraints; none paper over a
   deeper issue.

7. **The two-pass timeseries determinism verification is a real test,
   not a vacuous PASS.** The commit message claims "Verified by
   two-pass byte-identity: re-running the figure suite produces
   IDENTICAL real_window_idx / fake_window_idx values for all 9 models."
   The SHA-256 seeding is deterministic by construction, but the
   two-pass verification is the right empirical confirmation.

## Final recommendation

**READY FOR PAPER RESUBMISSION: YES** — with two pre-tag hot fixes
recommended.

The 14-13 sweep is competent, methodical, and substantive. The original
28 findings are addressed at a depth appropriate to the manuscript
deadline. D-14-22 byte-freeze is literally honored. Three of five
CRITICAL findings (CR-1, CR-2, CR-5) are cleanly resolved. The two
partial CR resolutions (CR-3 one-line-off citation, CR-4 gate doesn't
actually gate) are deficiencies in the FIX, not regressions to the
original bug — the underlying behavior (CR-3: programmatic citations
are now mostly programmatic; CR-4: MPS-disable hook is applied) is
materially improved.

**Recommended pre-tag hot fixes (≤30 min each):**
1. **R2-HIGH-1 (sign-flip)**: Add `-` to the boundary regex's
   lookbehind class. Verify the gate still passes on all 9 docs.
2. **R2-MED-1 (citation 1 line off)**: Change
   `"generator_to_compute_dtype"` target pattern to
   `"generated_samples = generated_samples.to(compute_dtype)"`.
   Regenerate `methods_full.json`.

**Recommended post-tag follow-ups (not blocking):**
- R2-HIGH-2 (ε-neighborhood false positives) requires architectural
  rework of the gate. Disclose as a gate-limitation note.
- R2-HIGH-3 (`training_time_device` captures post-sample device): fix
  in the next refactor; the disclosure in methods_full.md §4.2 is the
  current operational protection.
- R2-HIGH-4 (`_train_vae` cosmetic hook): clarify in docstring.

The peer-review-r2 verdict: the manuscript can ship, the resubmission
package is materially stronger than v1, and the residual findings are
either non-blocking or trivially hot-fixable within the resubmission
window.
