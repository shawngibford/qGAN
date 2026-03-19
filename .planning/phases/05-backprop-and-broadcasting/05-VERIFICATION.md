---
phase: 05-backprop-and-broadcasting
verified: 2026-03-19T00:00:00Z
status: human_needed
score: 3/4 success criteria verified (SC4 waived by user)
gaps:
human_verification:
  - test: "Run equivalence test cell to confirm batched vs sequential output match"
    expected: "max_diff < 1e-6 between batched and sequential QNode calls on same inputs"
    why_human: "Validation cells were run and removed per SUMMARY 05-02. SUMMARY documents PASS but cell outputs were not retained. Cannot verify numerically from codebase alone — requires running the notebook."
---

# Phase 5: Backprop and Broadcasting Verification Report

**Phase Goal:** Training runs ~12x faster through batched quantum circuit execution, unblocking efficient iteration for spectral loss tuning and conditioning experiments
**Verified:** 2026-03-19
**Status:** human_needed (SC1 VERIFIED, SC2 VERIFIED, SC3 human-only, SC4 WAIVED by user)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| SC1 | QNode uses diff_method='backprop' on default.qubit with shots=None | VERIFIED | `diff_method='backprop'` present in Cell 26; `shots=None` on device; `PennyLane #4462` comment present; old `'parameter-shift'` and old comment text absent |
| SC2 | All three training loops (critic, generator, evaluation) use a single batched QNode call per step instead of per-sample Python loops | VERIFIED | All 4 per-sample loops eliminated: critic, generator, eval Cell 26, eval Cell 41. Batched replacements present with correct (num_qubits, N) shapes at every site. |
| SC3 | Batched QNode output matches sequential per-sample output within 1e-6 tolerance | HUMAN NEEDED | SUMMARY 05-02 documents equivalence test PASSED, but validation cells were removed post-run per plan. Cell outputs not retained. Cannot verify numerically from static codebase. |
| SC4 | Epoch wall-clock time is less than 30% of pre-broadcasting time | WAIVED | SC4 hard gate FAILED at 103.7% ratio. Root cause: circuit returns tuple of 10 separate `qml.expval()` calls, preventing PennyLane from vectorizing batch simulation. User accepted backprop-only gain and waived SC4 to proceed to Phase 6. |

**Score:** 2/3 programmatically verifiable criteria verified (SC4 waived; SC3 requires human)

---

## SC1 Verification: QNode diff_method='backprop'

**Cell 26 checks — all PASS:**

| Check | Result | Evidence |
|---|---|---|
| `diff_method='backprop'` present | PASS | Found in `define_generator_model` |
| `diff_method='parameter-shift'` absent | PASS | Not present anywhere in Cell 26 |
| `shots=None` on device | PASS | `qml.device("default.qubit", wires=self.num_qubits, shots=None)` |
| `PennyLane #4462` comment | PASS | Comment present at QNode definition |
| Old comment text absent | PASS | `'Explicit gradient method for better stability'` not found |

---

## SC2 Verification: Batched QNode Calls

### Per-sample loops eliminated

**Critic training loop (Cell 26):**

| Check | Result |
|---|---|
| `real_batch = []` absent | PASS |
| `fake_batch = []` absent | PASS |
| `for i in range(self.batch_size):` absent | PASS |
| `real_batch.append` absent | PASS |
| `fake_batch.append` absent | PASS |

**Generator training loop (Cell 26):**

| Check | Result |
|---|---|
| `input_circuits_batch = []` absent | PASS |
| `par_circuits_batch = []` absent | PASS |
| `for i in range(generator_inputs.shape[0]):` absent | PASS |
| `par_windows_batch` absent | PASS |

**Evaluation loop — Cell 26:**

| Check | Result |
|---|---|
| `for j, generator_input in enumerate(generator_inputs):` absent | PASS |
| `batch_generated = []` absent | PASS |

**Evaluation loop — Cell 41:**

| Check | Result |
|---|---|
| `fake_windows_list = []` absent | PASS |
| `fake_windows_list` (any reference) absent | PASS |
| `for _ in range(num_eval_samples):` absent | PASS |

### Batched replacements present

| Pattern | Location | Result |
|---|---|---|
| `rand_indices = torch.randint` | Cell 26 critic | PASS |
| `size=(self.num_qubits, self.batch_size)` noise shape | Cell 26 critic | PASS |
| `par_circuit_batch = par_compressed.T` | Cell 26 critic | PASS |
| `torch.no_grad()` wrapping critic fake generation | Cell 26 critic | PASS |
| `rand_indices_g = torch.randint` | Cell 26 generator | PASS |
| `noise_batch_g` | Cell 26 generator | PASS |
| `par_circuit_batch_g = par_compressed_g.T` | Cell 26 generator | PASS |
| `gradient flows through params_pqc` comment | Cell 26 generator | PASS |
| Generator section NOT wrapped in `torch.no_grad()` | Cell 26 generator | PASS |
| `size=(self.num_qubits, num_samples)` noise shape | Cell 26 eval | PASS |
| `par_circuit_batch_e` | Cell 26 eval | PASS |
| `size=(NUM_QUBITS, num_eval_samples)` noise shape | Cell 41 | PASS |
| `par_circuit_batch_41` | Cell 41 | PASS |
| `fake_windows.reshape` (replaces fake_windows_list) | Cell 41 | PASS |
| `torch.no_grad()` wrapping Cell 41 eval | Cell 41 | PASS |

**`torch.stack(list(results)).T` pattern count in Cell 26:** 3 (critic, generator, eval) — PASS
**`torch.stack(list(results_41)).T` pattern count in Cell 41:** 1 — PASS

**Shape comments at each call site:**
- `# noise: (num_qubits, batch_size) -- batched QNode call` — PASS
- `# noise: (num_qubits, num_samples) -- batched QNode call` — PASS
- `# noise: (NUM_QUBITS, num_eval_samples) -- batched QNode call` — PASS

**isinstance boilerplate removed from converted loops:** PASS
(Two remaining `isinstance` calls are in a separate stats/plotting helper method at the bottom of Cell 26 — not in any converted training section.)

---

## SC3 Verification: Equivalence Within 1e-6

SC3 cannot be verified programmatically from the static codebase. The equivalence test cell was:
1. Inserted into the notebook (commit `58d675c`)
2. Run by the user
3. Removed after the run (commit `a1f1825`)

SUMMARY 05-02 documents: "Equivalence test: batched output matches sequential within tolerance" — but cell output was not retained in the notebook. Verification requires re-running the equivalence check in a live kernel.

**Human verification item:** See Human Verification section below.

---

## SC4 Status: WAIVED by User

**SC4 result:** 103.7% — post-broadcasting epoch time was approximately equal to pre-broadcasting time (no speedup).

**Root cause (documented in SUMMARY 05-02):**
> The circuit returns a tuple of 10 separate `qml.expval()` calls (`return (*measurements,)` pattern). This forces PennyLane to evaluate each observable independently rather than vectorizing the batch simulation. The research phase verified speedup on single-return circuits; the actual notebook circuit has a structurally different return type.

**Secondary issue:** The pre-broadcasting baseline (`_sequential_epoch`) omitted the evaluation loop, making the comparison unfair.

**User decision:** Accept backprop-only gain. Broadcasting syntax retained for future PennyLane optimization. Circuit architecture (`return (*measurements,)`) preserved to avoid training quality regression risk.

**Backprop benefit retained:** `diff_method='backprop'` is correct and necessary for proper gradient flow (PennyLane issue #4462 — parameter-shift explicitly disallows gradients through broadcasted tapes). REG-02 is satisfied regardless of SC4.

**Validation cells removed:** No `ONE-TIME`, `SC4 Gate`, `Equivalence Test`, or `Same-Seed Mini-Run` content remains in any cell. — PASS

---

## Required Artifacts

| Artifact | Status | Details |
|---|---|---|
| `qgan_pennylane.ipynb` | VERIFIED | Exists; contains all required patterns; 71 cells; no validation cell residue |

**Key decisions recorded in SUMMARY 05-01:**
- `backprop` replaces `parameter-shift` due to PennyLane #4462 broadcasting gradient bugs
- Noise shape `(num_qubits, batch_size)` not `(batch_size, num_qubits)` per PennyLane broadcasting spec
- `torch.stack(list(results)).T` pattern adopted for all batched QNode output reshaping
- `fake_windows_list` eliminated from Cell 41 PSD section

---

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `qGAN.__init__` | `qml.device` call | `shots=None` kwarg | VERIFIED | `shots=None` present on device creation |
| `qGAN.define_generator_model` | `qml.QNode` | `diff_method='backprop'` | VERIFIED | Present; `parameter-shift` absent |
| Critic training loop | `self.generator` | Single batched call, noise shape `(num_qubits, batch_size)` | VERIFIED | `rand_indices`, `noise_batch`, `par_circuit_batch` present; per-sample loop absent |
| Generator training loop | `self.params_pqc` | Gradient flows through `torch.stack(list(results)).T` | VERIFIED | `gradient flows through params_pqc` comment present; loop NOT in `no_grad` |
| Evaluation loop | `self.generator` | Single batched call, noise shape `(num_qubits, num_samples)` | VERIFIED | `par_circuit_batch_e`, shape comments present |
| Cell 41 eval | `qgan_val.generator` | Single batched call, noise shape `(NUM_QUBITS, num_eval_samples)` | VERIFIED | `par_circuit_batch_41`, `fake_windows.reshape` present |

---

## Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|---|---|---|---|---|
| REG-02 | Phase 5 | QNode uses `diff_method='backprop'` instead of `parameter-shift` | SATISFIED | `diff_method='backprop'` in Cell 26; `parameter-shift` absent; `shots=None` on device |
| REG-03 | Phase 5 | Training loop uses batched/broadcasted QNode calls instead of per-sample Python loops (~12x speedup) | SATISFIED (partial) | All 4 per-sample loops eliminated and replaced with single batched calls. SC4 speedup failed (103.7%) and was waived; batched syntax is in place. |

**Traceability:** REQUIREMENTS.md marks REG-02 and REG-03 as Complete under Phase 5. No orphaned requirements found.

**Note on REG-03:** The requirement description says "~12x speedup" but the user waived this aspect after investigation revealed the multi-expval circuit prevents vectorization. The batched call syntax is in place as required. The requirement is marked Complete in REQUIREMENTS.md consistent with the user's decision.

---

## Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|---|---|---|---|
| `qgan_pennylane.ipynb` Cell 26 | `isinstance` at positions 21143, 21316 | Info | In stats/plotting helper method (`evaluate_and_compare`), not in converted training loops. Appropriate usage. |

No TODO/FIXME/PLACEHOLDER/HACK/XXX patterns found in Cell 26 or Cell 41.
No `return null`, `return {}`, `return []` stubs found.
No per-sample loop remnants in training sections.

---

## Human Verification Required

### 1. Equivalence Test (SC3)

**Test:** Open `qgan_pennylane.ipynb` in Jupyter. Run all prerequisite cells (through the cell that instantiates `qgan`). Then add and run this equivalence check:

```python
import torch, numpy as np
torch.manual_seed(42)
np.random.seed(42)
TEST_BATCH = 12
noise_seq = torch.tensor(np.random.uniform(0, 4 * np.pi, size=(NUM_QUBITS, TEST_BATCH)), dtype=torch.float32)
par_light_seq = torch.rand(NUM_QUBITS, TEST_BATCH, dtype=torch.float32)

with torch.no_grad():
    sequential_results = []
    for i in range(TEST_BATCH):
        out = qgan.generator(noise_seq[:, i], par_light_seq[:, i], qgan.params_pqc)
        sequential_results.append(torch.stack(list(out)))
    sequential_out = torch.stack(sequential_results)

with torch.no_grad():
    results = qgan.generator(noise_seq, par_light_seq, qgan.params_pqc)
    batched_out = torch.stack(list(results)).T

max_diff = (sequential_out - batched_out).abs().max().item()
print(f'max_diff={max_diff:.2e}  PASS={max_diff < 1e-6}')
```

**Expected:** `PASS=True` with `max_diff < 1e-6`
**Why human:** Validation cells were run and removed per the plan. The SUMMARY documents the test passed, but no cell output was retained in the notebook to verify programmatically.

---

## Commit Evidence

All 5 task commits verified present in git:

| Commit | Message |
|---|---|
| `fa811ee` | feat(05-01): switch QNode to backprop and add shots=None |
| `61996ea` | feat(05-01): convert critic and generator training loops to batched QNode calls |
| `51853f3` | feat(05-01): convert evaluation loops to batched QNode calls |
| `58d675c` | feat(05-02): add equivalence, reproducibility, and SC4 timing validation cells |
| `a1f1825` | feat(05-02): remove validation cells after SC4 investigation |

---

## Summary

**SC1 (backprop QNode):** Fully verified in codebase. `diff_method='backprop'`, `shots=None`, `PennyLane #4462` comment all present. Old `parameter-shift` and old comment text absent.

**SC2 (batched loops):** Fully verified in codebase. All 4 per-sample loops eliminated. Batched replacements with correct `(num_qubits, N)` shapes present at all 4 sites. Shape comments, gradient flow comment, and `torch.no_grad()` placement all correct.

**SC3 (equivalence tolerance):** Requires human verification. SUMMARY documents the test passed, but the cell and its outputs were removed per the plan. The implementation structure is consistent with the test passing (same patterns used in research-verified code).

**SC4 (timing speedup):** Waived by user after investigation. Root cause documented: multi-expval tuple return prevents PennyLane vectorization. Backprop-only gain accepted. Broadcasting syntax retained for future PennyLane improvements.

**Overall:** The codebase fully implements the structural changes required by REG-02 and REG-03. The primary performance goal (~12x speedup) was not achieved due to a PennyLane limitation, but the user explicitly accepted this outcome and the phase is marked complete per the user's decision.

---

_Verified: 2026-03-19_
_Verifier: Claude (gsd-verifier)_
