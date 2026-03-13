---
phase: 04-code-regression-fixes
verified: 2026-03-13T19:30:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 4: Code Regression Fixes Verification Report

**Phase Goal:** Training code produces correct results with trustworthy metrics -- noise range matches circuit design, evaluation reflects conditioned generation, and plotting cells are safe for re-execution
**Verified:** 2026-03-13T19:30:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Noise sampled from [0, 4pi] in all 3 training locations (critic, generator, eval) | VERIFIED | Cell 26 lines 308, 385, 449 all use `np.random.uniform(0, 4 * np.pi, ...)`. No `2 * np.pi` occurrences remain in cell 26. |
| 2 | Evaluation generates fake samples using real PAR_LIGHT values, not zeros | VERIFIED | Cell 26 eval block (lines 457-463) samples from `par_data_list` using `reshape(num_qubits, 2).mean(dim=1)` + remap -- same pattern as critic/generator training. `par_zeros` is completely absent. |
| 3 | Plotting cells can be re-executed without mu/sigma variable shadowing | VERIFIED | Cell 12 has zero standalone `mu =` or `sigma =` assignments. `norm.pdf` and `print` statements use inline `np.mean(log_delta_np)` and `np.std(log_delta_np)`. Cell 15 mu/sigma (normalization params) are preserved and unaffected. |
| 4 | 200-epoch validation run produces EMD within 2x of HPO baseline (0.001137) | VERIFIED | `results/phase4_validation.json` records EMD=0.001301, outcome=PASS. Threshold is 0.002274 (2x baseline). Ratio is 1.14x. All 200 epochs completed, no fallback used. |
| 5 | ACF loss code completely removed (not just zeroed) | VERIFIED | Cell 26: `self.lambda_acf` absent, `diff_acf_lag1` method absent, `acf_penalty` absent. Cell 28: `LAMBDA_ACF` absent. Cell 40: `lambda_acf` absent. `self.acf_avg` eval metric tracking is preserved. |
| 6 | HPO-tuned hyperparameters set and hyperparameter cell has no LAMBDA_ACF reference | VERIFIED | Cell 28: `N_CRITIC = 9`, `LAMBDA = 2.16`, `LR_CRITIC = 1.8046e-05`, `LR_GENERATOR = 6.9173e-05`. `LAMBDA_ACF` absent. `qGAN(...)` constructor call has no `lambda_acf` argument. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `qgan_pennylane.ipynb` | All regression fixes applied across cells 12, 26, 28, 29, 40, 41, 45 | VERIFIED | Valid JSON. All 6 cells fixed. `4 * np.pi` confirmed at cells 26 (x3), 29, 41, 45, 46. |
| `results/phase4_validation.json` | Complete validation results with config, metrics, outcome | VERIFIED | All required keys present: `phase`, `timestamp`, `git_hash`, `config`, `metrics`, `hpo_baseline`, `outcome`. EMD=0.001301, outcome=PASS. |
| `scripts/phase4_validation.py` | Standalone validation script for repeatable execution | VERIFIED | File exists at `scripts/phase4_validation.py`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Cell 26 eval block | `par_data_list` | `random_idx = torch.randint(0, len(par_data_list), ...)` + `reshape(num_qubits, 2).mean(dim=1).float()` + remap | VERIFIED | Lines 457-463 implement the full pattern. Matches critic training pattern (lines 298-314) and generator training pattern (lines 386-394). |
| Cell 28 qGAN() constructor | No `lambda_acf` | Constructor call: `qGAN(NUM_EPOCHS, BATCH_SIZE, WINDOW_LENGTH, N_CRITIC, LAMBDA, NUM_LAYERS, NUM_QUBITS, delta=delta)` | VERIFIED | No `lambda_acf` argument in cell 28. Cell 26 `__init__` signature also has no `lambda_acf` parameter. Consistent throughout. |
| Validation cell (41) | `results/phase4_validation.json` | `json.dump` at end of validation | VERIFIED | File exists with full structure. JSON `config.noise_range` = `[0, "4*pi"]` confirming corrected code was used. |
| `results/phase4_validation.json` | Phase 5-7 comparison reference | `hpo_baseline.best_emd` field | VERIFIED | `hpo_baseline`: `{"best_emd": 0.001137, "threshold_2x": 0.002274}`. Config includes all HPO params, noise range, git hash. Full PSD arrays (6 bins) stored for Phase 6. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| REG-01 | 04-01-PLAN.md | Training loop uses correct noise range [0, 4pi] in all 3 locations | SATISFIED | Cell 26 lines 308, 385, 449 -- all three training locations confirmed [0, 4pi]. Cell 29 and 45 also corrected (cosmetic and standalone gen). |
| REG-04 | 04-01-PLAN.md | Evaluation generation uses real PAR_LIGHT values instead of `torch.zeros` | SATISFIED | Cell 26 eval block lines 457-463 use `par_data_list` random sampling with reshape+mean+remap pattern. `par_zeros` absent from notebook. |
| REG-05 | 04-01-PLAN.md | mu/sigma variable shadowing eliminated in plotting cells | SATISFIED | Cell 12 has no standalone `mu =` or `sigma =` assignments. Uses `np.mean(log_delta_np)` and `np.std(log_delta_np)` inline. |

All 3 requirements from both plan frontmatter entries are satisfied. No orphaned requirements found (REG-02 and REG-03 are correctly assigned to Phase 5; SPEC-* to Phase 6; COND-* to Phase 7).

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `results/phase4_validation.json` config | `lambda_acf: 0` stored in JSON config | Info | Historical artifact -- the validation script stored `lambda_acf=0` in its config dict for documentation purposes even though it is not passed to qGAN constructor. The qGAN class itself has no `lambda_acf` parameter; this is a metadata field in the JSON only. No training impact. |

No blocker or warning anti-patterns found in modified cells. All 7 cells (12, 26, 28, 29, 40, 41, 45) scanned: zero TODO/FIXME/PLACEHOLDER/XXX/HACK occurrences. No empty returns or stub implementations.

### Human Verification Required

None -- all success criteria are verifiable programmatically:

- Noise range verified by string inspection of actual code
- PAR_LIGHT conditioning verified by absence of `par_zeros` and presence of `par_data_list` sampling
- mu/sigma shadowing verified by absence of standalone variable assignments
- EMD outcome verified from persisted JSON with numeric value 0.001301

The validation run was executed via standalone script (`scripts/phase4_validation.py`) rather than interactive notebook, eliminating display/plotting dependencies as a confound.

### Commit Verification

All three commits from SUMMARY files confirmed present in git history:

- `60b27e5` -- Task 1: Fix qGAN class (cell 26) noise range, par_zeros, ACF removal
- `d10cc05` -- Task 2: Fix supporting cells (mu/sigma, hyperparameters, circuit diagram, HPO retrain, standalone generation)
- `41f8540` -- Task 1 of plan 02: Create validation cell and execute 200-epoch training run

---

## Summary

Phase 4 fully achieves its goal. All six must-have truths are verified against the actual codebase:

1. The noise range fix is comprehensive -- three locations in cell 26 (critic training line 308, generator training line 385, evaluation line 449) plus cell 29 (circuit diagram) and cell 45 (standalone generation) all use `[0, 4pi]`. No `2 * np.pi` remains in any code cell.

2. The par_zeros eval bug is replaced with a real PAR_LIGHT conditioning pattern that is consistent with critic and generator training -- random index sampling from `par_data_list`, reshape to `(num_qubits, 2)`, mean across dim=1, remap from `[-1, 1]` to `[0, 1]`.

3. The mu/sigma shadowing in cell 12 is eliminated -- norm.pdf and print statements use inline `np.mean(log_delta_np)` and `np.std(log_delta_np)`. Cell 15 normalization parameters (`mu`, `sigma`) are unaffected.

4. ACF loss is completely removed at all sites: constructor parameter, instance attribute, `diff_acf_lag1` static method, penalty block, and combined loss line. The ACF evaluation metric (`self.acf_avg`) is preserved. Cells 28 and 40 have no `lambda_acf` references.

5. HPO-tuned hyperparameters are in place: `N_CRITIC=9`, `LAMBDA=2.16`, `LR_CRITIC=1.8046e-05`, `LR_GENERATOR=6.9173e-05`.

6. The 200-epoch validation run confirms the fixes produce a working model: EMD=0.001301 (1.14x HPO baseline of 0.001137, within the 2x threshold of 0.002274). Outcome is PASS. No fallback used. Comprehensive baseline metrics saved to `results/phase4_validation.json` for downstream Phase 5-7 comparison.

---

_Verified: 2026-03-13T19:30:00Z_
_Verifier: Claude (gsd-verifier)_
