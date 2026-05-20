# 14-14-PLAN-CHECK — Goal-Backward Pre-Execution Verification

**Plan checked:** `.planning/phases/14-paper-revision-release-freeze/14-14-PLAN.md` (1,367 lines, 5 atomic tasks + SUMMARY)
**Approved design:** `~/.claude/plans/reviewer-1-1-r1-m1-merry-moonbeam.md` (148 lines)
**Source reports:** 5 r2 peer-review reports in `.planning/phases/14-paper-revision-release-freeze/peer-review-r2/`
**Date:** 2026-05-20
**Stance:** Adversarial (assume flawed until proven otherwise)

---

## Summary verdict

**PASS-WITH-FIXES** — 2 BLOCKERS in verify blocks (mechanically wrong assertions that will fail on a correctly-executed task) + 1 MEDIUM coverage gap (REPRO-LOW-1 partially addressed) + 4 LOW notes. None of the BLOCKERS touch the core science; all are off-by-one / count-mismatch errors in the plan's own verify shell snippets that will incorrectly fail when execution succeeds. Fixable in 5 minutes of plan revision.

**Recommended go/no-go for foreground execution:** **NO-GO until verify-block off-by-counts are patched.** The substantive plan content (gate v2.1 fix, training_time_device capture relocation, VAE β derivation correction, 12-item doc/JSON cleanup) is sound and matches the approved design. The blockers below are mechanical, not conceptual.

---

## BLOCKERS (must fix before execution)

### BLOCKER-1 — T4 verify block requires `count('1-69') >= 4` but only 3 sites exist (off-by-one)

**Plan lines:** 904 (verify block), 76 (read_first), 78 (action), 274–276 (interfaces), 715–725 (action Step B), 907 (acceptance criteria)
**Source reality:**

```
$ grep -nc "1-80" revision/run_methods_full.py
3            ← only 3 occurrences, not 4
$ grep -n "1-80" revision/run_methods_full.py
152:# Docstring slicer for run_matched2000.py:1-80 — captures the leading module
265:    # ── Verbatim docstring slice from run_matched2000.py:1-80 ─────────
524:            "revision/run_matched2000.py:1-80 (module docstring; preserved "
$ sed -n '560,575p' revision/run_methods_full.py     ← line 566 area
            "revision/run_matched2000.py": (
                "text-only docstring slice (no execution)"
            ),
```

Line 566 does NOT contain `1-80`. The plan asserts 4 sites at (152, 265, 524, **566**) repeatedly throughout the plan text, but the file only carries 3 sites at (152, 265, 524). The line-566 region is a `_citations` dict entry mentioning the file path without a line range.

**Concrete failure:** T4's verify block at line 904 contains:

```python
(rmf.count('1-69') >= 4) or errs.append(f'run_methods_full.py has only {rmf.count("1-69")}× `1-69` — expected ≥4 (lines 152, 265, 524, 566)')
```

After T4 correctly flips the three actual `1-80` sites to `1-69`, `rmf.count('1-69') == 3`, which is `< 4`, so this errs append fires and the verify exits non-zero. **A correctly executed task will fail its own verify block.**

**Source of truth conflict:** `code-review-r2.md` R2-MED-1 / docstring-1-80 finding cites three sites (not four). The 4th site (`566`) appears to be a phantom that propagated from the approved-design plan's prose into the verify block. Direct file verification confirms 3 sites.

**Severity:** BLOCKER — the verify block will reject every correct execution.

**Fix:** Change all five mentions (152, 265, 524, **566**) → (152, 265, 524) and change `rmf.count('1-69') >= 4` → `== 3`. Also adjust:
- Plan line 76 (interfaces / context): "all `1-80` line-range labels … at lines 152, 265, 524, 566" → drop 566
- Plan line 274–276 (interfaces section): same
- Plan line 715–725 (T4 action Step B): "four sites carrying the `1-80` line-range label" → "three sites"
- Plan line 907 (T4 acceptance criteria): "`run_methods_full.py:152, 265, 524, 566` all carry `1-69`" → three sites
- Plan line 1118 (completeness manifest table T4 row): "`1-80` → `1-69` at 4 sites" → "at 3 sites"
- Plan line 1232 (T-14-52 STRIDE entry): "off-by-11 at four sites" → "three sites"

---

### BLOCKER-2 — T4 verify block under-counts `methods_full.md` `1-80` sites (misses line 431)

**Plan lines:** 76, 274–276, 690, 722, 909 (acceptance criteria), 904 (verify block)
**Source reality:**

```
$ grep -n "1-80" revision/docs/methods_full.md
431:`revision/run_matched2000.py` (lines 1-80) — preserved character-for-character
455:Source: `revision/run_matched2000.py:1-80` (module docstring; preserved
536:  `revision/run_matched2000.py:1-80` (module docstring) by
```

Three sites in `methods_full.md`, but the plan claims only two: `methods_full.md:455, 536`. Line 431 is missed.

**Concrete failure:** T4's verify block at line 904 asserts `('1-80' in mf) and errs.append('methods_full.md still contains 1-80 mention (should be 1-69)')`. This is a correctness check — but if the executor only flips :455 and :536 per the plan's read_first / interfaces / action sections (which only mention those two lines), :431 will remain at `1-80`, causing the verify to FAIL. Conversely, if the executor flips all 3 sites including :431, the verify passes — but then the plan's acceptance criteria say "`methods_full.md:455, 536` mentions of `1-80` are flipped to `1-69`" implying only 2 sites should be touched. Ambiguous contract.

**Severity:** BLOCKER — under-specified scope produces ambiguous completion criteria.

**Fix:** Update the plan to enumerate all 3 sites in `methods_full.md` (lines 431, 455, 536) consistently across read_first (line 690), interfaces (line 76), action (line 722), and acceptance criteria (line 909). The verify block (line 904) checks for "any `1-80` in mf" which correctly enforces full-file scrubbing; only the narrative needs to list all 3 sites.

---

## HIGH (fix recommended; impacts review trust if shipped)

### HIGH-1 — T4 apparatus_dimensions JSON: field-name semantics contradict provenance-review-r2 §2.a

**Plan lines:** 80, 277–290 (interfaces), 695 (T4 action Step D introduction), 800–807 (action Step D template), 912 (acceptance criteria), 1233 (T-14-54 STRIDE)
**Source reality (LaTeX `main (4) copy.tex:176–180`):**

```latex
six vertical borosilicate glass tubes (6~cm OD, 120~cm length)
...
data logged at 10-minute intervals
...
The OD sensor operates at 880~nm
```

**Provenance-review-r2 §2.a (lines 74–77, 81, 247):** explicitly maps the four current fields:
- `depth_or_height_880` = **880 nm** (NIR wavelength, not mm)
- `ancillary_120` = **120 cm** (tube length, not mm)
- `ancillary_6` = **6 cm** (tube outer diameter, not mm)
- `ancillary_10` = **10 minutes** (data logging interval, not mm)

**The plan's proposed field names** (`height_mm`, `depth_mm`, `depth_alt_mm`, `ir_led_wavelength_nm`):
- `height_mm` — the LaTeX uses **cm**, and the dimension isn't height (it's tube length); **wrong unit + wrong semantic**
- `depth_mm` — wrong unit (cm) + wrong semantic (it's tube OD)
- `depth_alt_mm` — wrong unit (minute, not mm) + wrong semantic (it's data logging interval, NOT a dimension)
- `ir_led_wavelength_nm` — correct (the only one)

**Mitigating language in the plan:** lines 78, 790–793, and 686 instruct the executor to "verify by reading the LaTeX in T4 and matching the field semantics EXACTLY (do NOT guess — read the LaTeX)". The plan defers final naming to T4 execution time. The verify block (line 904) only checks for `apparatus_dimensions` and `ir_led_wavelength_nm` (one out of four fields), so the plan does not pre-commit to the wrong names mechanically.

**However:** the prompt classifies "Apparatus dimensions JSON restructure — split into per-unit subfields (`height_mm`, `depth_mm`, `depth_alt_mm`, `ir_led_wavelength_nm`, etc.)" as a user-locked decision. Per the verifier instructions, I should NOT flag this as a blocker because it is user-locked. But I am flagging it as HIGH because the literal field names in the plan are demonstrably wrong against the LaTeX, and an executor following the plan's defaults without re-reading the LaTeX would produce a JSON that lies about units and semantics — the *exact* concern that motivated the restructure.

**Severity:** HIGH (downgraded from BLOCKER because the user explicitly locked the suggested field-name template).

**Fix recommendation:** Replace the suggested field-name template in the plan's interfaces / action Step D / acceptance criteria with the LaTeX-correct names, e.g. `{tube_outer_diameter_cm: 6, tube_length_cm: 120, data_logging_interval_min: 10, ir_led_wavelength_nm: 880}`. Re-confirm with user before re-baselining if you want to preserve the user-locked decision verbatim.

---

## MEDIUM (should fix; correctness gap)

### MEDIUM-1 — REPRO-LOW-1 partially addressed: `ansatz_comparison.json` missing from data_hash sweep

**Plan lines:** 18 (Other findings list), 50 (must_haves truth #6), 65–68 (artifacts), 312–315 (interfaces), 437 (T1 action Step E), 461 (T1 acceptance criteria)
**Source:** `methods-reproducibility-review-r2.md` REPRO-LOW-1 (line 269) explicitly enumerates THREE JSONs missing `data_hash`:

> `noise_model_sensitivity.json` / `shot_noise_sensitivity.json` / `ansatz_comparison.json` lack `data_hash`

Direct file verification:

```
$ python3 -c "import json; d=json.load(open('revision/results/ansatz_comparison.json')); print('data_hash:', d.get('data_hash'))"
data_hash: None
```

The plan adds `data_hash: 91e447d4624e25b3` to two of three (noise_model_sensitivity, shot_noise_sensitivity) but omits `ansatz_comparison.json`. The §5 R2-follow-up table in T5's peer_review_remediation.md addendum maps `R2-prov-MED-1` to `<sha_T1>` for "Both files updated with data_hash" — only two files, not three. This will leave REPRO-LOW-1 partially open after 14-14 lands.

**Severity:** MEDIUM — a residual sensitivity JSON without `data_hash` breaks the corpus consistency claim made in the SUMMARY ("After Task 5, all paper-load-bearing sensitivity JSONs carry data_hash").

**Note:** The reviewer's text marks REPRO-LOW-1 as LOW because "the resolution chain back-resolves through audited artifacts that do carry the hash". So shipping without ansatz_comparison.json is defensible, but the plan should EITHER (a) explicitly OOS-mark `ansatz_comparison.json` in T1 + the R2 follow-up table, with rationale, OR (b) add it to T1's file list. Currently the plan silently omits it.

**Fix:** Add `revision/results/ansatz_comparison.json` to T1's files list, action Step E, and verify block, OR add an explicit OOS line in T5's `## R2 follow-up sweep` table with rationale (per reviewer's "back-resolves through audited artifacts" framing).

---

### MEDIUM-2 — T4 verify block over-greedy on the CR-3 citation regex check

**Plan lines:** 904 (verify block — the CR-3 portion)
**Source:** the verify block's CR-3 check:

```python
('generated_samples\\\\.to\\\\(compute_dtype' in rmf or 'generated_samples = generated_samples\\.to\\(compute_dtype' in rmf or 'generated_samples\\.to' in rmf) or errs.append('run_methods_full.py CR-3 citation pattern not re-pointed to dtype cast site (training.py:347)')
```

The third disjunct `'generated_samples\\.to' in rmf` matches even a substring of `"generated_samples = generator"` if it were ever paired with a different `.to(...)` call — but more importantly, the **current** state of `run_methods_full.py:131` is:

```python
"generator_to_compute_dtype": "generated_samples = generator",
```

After T4's fix, it should become e.g. `"generated_samples = generated_samples.to(compute_dtype)"`. The third disjunct would also pass if the file contained ANY `generated_samples.to(...)` substring — including in comments or docstrings unrelated to the citation. The check is not anchored to the `_citations` dict key.

Additionally, the first disjunct has 4-backslash escaping (`'generated_samples\\\\.to\\\\(compute_dtype'`) which is the regex source pattern, not a literal Python substring — that disjunct never matches in a plain `in` test (the file doesn't contain `\\\\` literally).

**Severity:** MEDIUM — false-positive PASS likely (the check is too lenient to actually catch a non-fix).

**Fix:** Replace the disjunct chain with a single anchored check on the citations dict, e.g.:
```python
m = re.search(r'"generator_to_compute_dtype":\s*"([^"]+)"', rmf)
(m and 'generated_samples.to(compute_dtype)' in m.group(1)) or errs.append('CR-3 _citations["generator_to_compute_dtype"] pattern not re-pointed to the dtype cast site')
```

---

## LOW (notes — not action items)

### LOW-1 — T2 verify block thresholds 4 occurrences of `training_time_device`; current file already has 1 (`_device_manifest` body)

**Plan lines:** 560 (verify block)
**Source reality:**

```
$ grep -c "training_time_device" revision/run_matched2000.py
4   (in the existing _device_manifest implementation from 14-13 T4)
```

The CURRENT `run_matched2000.py` already mentions `training_time_device` 4 times in `_device_manifest` (lines 326–335 region — see read above). After T2 adds 3 capture sites + extends the signature, the count rises to ≥7. The verify's `>= 4` threshold is too lenient — a degenerate T2 that only renames the existing field (without actually adding capture sites) would still pass. Not a blocker (executor should follow the action's explicit Step B/C/D), but the verify check is weak.

**Fix:** Tighten the threshold to `>= 6` (3 new capture sites × 2 mentions each: one assignment + one pass-through call) or count distinct line numbers.

### LOW-2 — Gate v2.1 regex scope is intentionally narrower than reviewer recommendation (USER-LOCKED, NOT FLAGGED)

The provenance-review-r2 R2-prov-HIGH-1 recommended `(?<![\d.\-+])` (both `-` and `+`). The plan implements `(?<![-\d.])` (only `-`). The user explicitly locked this scope (prompt: "one-character regex fix to lookbehind class" + "negative-sign-aware lookbehind"). Tested: positive token `0.0001` against `+0.0001` still false-matches under v2.1 (the `+` case is unaddressed). Acceptable per user lock; mentioning here only for completeness.

### LOW-3 — VAE β derivation: math is correct, but the derivation collapses the batch dimension

**Plan lines:** 66, 595–615 (action Step A), 665–667 (acceptance criteria)
**Source:** `revision/run_baselines.py:315-319` actually uses `F.mse_loss(x_hat, x)` (default `reduction='mean'`) which averages over `B × W × F`, and `torch.mean(1 + logvar - mu² - exp(logvar))` averages over `B × L`. The math reviewer's derivation at math-review-r2.md:122–124 correctly includes the batch factor: `loss = (1/(B×W)) Σ_b recon_sum_b + (β/(B×L)) Σ_b kld_sum_b`, leading to `W·B·loss = Σ_b (recon_sum_b + (W/L)·β·kld_sum_b)`, so β_eff = W/L·β = 2.5·1 = 2.5. The plan's derivation collapses to `sum_MSE/10 + sum_KLD/4` (no batch factor); the conclusion β_eff = 2.5 is correct because B cancels symmetrically, but the per-element-mean derivation as written is mathematically loose. Not a blocker; a careful reader would note the missing 1/B in both terms cancels.

**Note:** This matches what the prompt told me to verify — "β_eff = W/L = 10/4 = 2.5 (KL up-weighted)". The conclusion is sound. Flagging only because the prompt requested rigorous derivation cross-check.

### LOW-4 — T5 verify block: regex over-escaping of the `__main__` differential-test snippet

Plan line 1191's verify uses `--differential-test` flag invocation; the gate's actual `__main__` block (proposed at lines 978–999) parses `sys.argv` for `"--differential-test"`. The verify expects the gate to exit 0 with a printed PASS — works only if `argparse` (if used) or `sys.argv` parsing doesn't first reject the flag in the absence of `--target`. Executor should ensure the differential-test path is mutually exclusive with the `--target` path or short-circuits before any required-arg check. Not a blocker; executor can structure however.

---

## False positives skipped (locked decisions correctly NOT flagged)

Per the prompt's "Locked decisions to honor" list, the following were considered for flagging and correctly NOT flagged:

- **D-14-16 LIFTED for T1 only** — one-character regex fix; v2.1 schema bump; back to byte-freeze after T1. Reviewer recommended both `-` and `+` in lookbehind; plan scope is `-` only. User-locked, not flagged.
- **D-14-22 PRESERVED** — every task verify asserts `[ -z "$(git diff --stat revision/core/)" ]`. Verified across all 5 task verify blocks. Not flagged (correctly enforced).
- **D-14-13 PRESERVED** — T2 corrects only capture site; `_strict_accept` equality check semantics unchanged. Plan's action Step E (line 537–542) is explicit. Not flagged.
- **R2-code-HIGH-2 (ε-neighborhood broad coincidences) DOCUMENTED, NOT TIGHTENED** — T5 Step C appends `## Gate v2.1 known limitations` to peer_review_remediation.md with the `-0.26 → -0.2570036` concrete example. User-locked disclosure-not-tighten path. Not flagged.
- **Apparatus dimensions JSON restructure** — user-locked to "per-unit subfields", with execution-time LaTeX verification. The proposed field-name template is wrong vs LaTeX semantics, BUT the user locked the *approach* not the *literal names*. Flagged as HIGH-1 above (downgraded from BLOCKER) because executor discretion is preserved by the "verify at execution time" instruction.
- **REPRODUCE.md created at repo root in T4** — verified at plan line 808+, acceptance criteria line 915. Not flagged.

---

## Goal-backward trace (does T5's 12-point checklist actually establish what the phase needs?)

For each of T5's final 12 checks, traced back to the task that establishes it:

| # | Check | Established by |
|---|-------|----------------|
| 1 | Gate v2.1 PASSES on all 10 paper-facing docs with `--manifest` | T1 (gate fix) + T5 Step A (re-run) ✓ |
| 2 | Differential test: positive `0.0001` does NOT resolve to `-0.0001` | T5 Step B (`__main__` differential-test) ✓ |
| 3 | `training_time_device` reflects training device, not post-`.to(cpu)` | T2 (capture-before-`.to(cpu)`) ✓ |
| 4 | `methods_full.md §3.x.d` states β_eff = 2.5; §2.i carries not-param-matched caveat | T3 Steps A + B ✓ |
| 5 | `reconciliation_note.md` mentions wgan_cnn + Welch t-test | T3 Step C ✓ |
| 6 | `reviewer_response.md` R1-m4 says "Zenodo DOI pending under Plan 14-07" | T4 Step A ✓ |
| 7 | `_introspect_*.json` (all 4) carry `"render_only": true` | T1 Step D ✓ |
| 8 | `noise_model_sensitivity.json` + `shot_noise_sensitivity.json` carry `data_hash` | T1 Step E ✓ (but see MEDIUM-1: ansatz_comparison.json missed) |
| 9 | `requirements-pinned.txt` includes `statsmodels==0.14.5` | T4 Step E ✓ |
| 10 | `REPRODUCE.md` exists at repo root, links to methods_full.md §5.2 + completeness_sweep_manifest.md | T4 Step F ✓ |
| 11 | `peer_review_remediation.md` carries v2.1-limitations + r2-follow-up sections; every r2 finding mapped | T5 Steps C+D ✓ |
| 12 | `git diff` against pre-14-14 main shows zero `revision/core/` changes | every task verify asserts this ✓ |

All 12 checks trace to concrete establishing tasks. The only goal-backward gap is the `ansatz_comparison.json` data_hash (MEDIUM-1) which is not in the 12-check list but is in the reviewer's REPRO-LOW-1 — minor.

---

## Roadmap consistency (verified, no findings)

ROADMAP.md state matches plan claims:
- Line 48: `**Plans**: 14 plans` ✓
- Line 50: D-14-23 paragraph appended with the 14-14 sentence ✓
- Lines 77–78: Wave 12 block inserted with `14-14-PLAN.md` ✓
- Line 100: `12/14 | In Progress` ✓

The plan's "ROADMAP.md edits" section (lines 115–124) describes what's already committed. Consistent.

---

## Mechanics check (from prompt item 8)

The plan's resume-after-clear flow (line 136, 1355–1361) correctly:
- References `/gsd-execute-phase 14`
- Cites the foreground-executor requirement explicitly: "14-14 must be spawned **foreground** (background subagents auto-deny Bash on the first git safety call — `~/.claude/projects/-Users-shawngibford-dev-phd-qGAN/memory/project_quantum_sim_execution.md`)"
- Documents the stash-cycle protocol for pre-existing dirty working-tree files

No findings on mechanics.

---

## Final tally

| Severity | Count | Items |
|---|---|---|
| BLOCKER | 2 | Verify blocks 1-69 count + methods_full.md scope |
| HIGH | 1 | Apparatus field names contradict LaTeX (user-locked approach; flag suggests rename) |
| MEDIUM | 2 | ansatz_comparison.json gap; CR-3 verify too lenient |
| LOW | 4 | (notes) |

**Verdict:** PASS-WITH-FIXES. The 2 BLOCKERs are mechanical off-by-one errors in verify shell snippets that would falsely reject a correctly executed task. Fix the verify-block counts (BLOCKER-1: count==3 not >=4; BLOCKER-2: enumerate line 431) and the plan is execution-ready. Recommended fix-mode: a 3-edit patch on the planner's revision pass (≤5 minutes), no re-baseline needed.

