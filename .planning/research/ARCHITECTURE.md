# Architecture Research: qGAN Remediation Build Order

**Research Date:** 2026-02-26
**Research Type:** Project Research — Architecture dimension
**Question:** How should the fixes be organized within the notebook? What's the right order of changes to avoid breaking dependencies?

---

## Summary

The notebook has six component boundaries. The correct build order follows data flow direction. Five dependency tiers emerge, mapping to three remediation phases.

---

## Component Boundaries

### Component A: Imports (Cells 1-2)
**Issues:** `numpy` imported twice, `random` imported twice, inline imports inside `EarlyStopping` class.
**Fix complexity:** Trivial — deletion only.

### Component B: Data Loading & Preprocessing (Cells 3-18)
**Issues:**
- Hardcoded absolute path (`/Users/shawngibford/dev/qml/qGAN/data.csv`)
- `normalize()` discards `mu` and `sigma` (returns only normalized data)
- `delta` is an implicit global variable
**Fix complexity:** Low — path change, function signature change, scope delta.

### Component C: Hyperparameter Configuration (Cell 20)
**Issues:**
- `WINDOW_LENGTH = 10` should be `WINDOW_LENGTH = 2 * NUM_QUBITS`
- `n_critic = 1` should be 5
- `LAMBDA = 0.8` should be 10
- LR ratio backwards (generator > critic)
- Inconsistent naming (`n_critic` vs `LAMBDA`)
**Fix complexity:** Low — value changes + naming consistency.

### Component D: qGAN Class Definition (Cells 19-20)
Three sub-components:

**D1 — Quantum Circuit:**
- Redundant IQP RZ before noise encoding
- No data re-uploading
- Noise range `[0, 2pi]` should be `[0, 4pi]`
- `diff_method='parameter-shift'` should be `'backprop'`
- Single-qubit measurements only

**D2 — Critic Network:**
- `nn.Dropout(p=0.2)` violates WGAN-GP Lipschitz requirement

**D3 — Training Loop:**
- Evaluation runs every epoch (should be periodic)
- Memory leak: tensors with computation graphs in loss lists
- Three `* 0.1` magic scaling constants
- EMD computed on histogram bins
- Early stopping monitors critic loss instead of EMD
- `torch.load()` missing `weights_only=True`
- Checkpoint references `model.discriminator` not `model.critic`

### Component E: Training Execution & Post-Generation (Cells 20-29)
**Issues:**
- Debug artifact cells
- Post-generation uses hardcoded `delta=1` instead of computed delta
- Denormalization applied inconsistently (skipped in training, applied in generation)
- `eval()` call in debug cell
- `exit()` call

### Component F: Visualization (Cells 30-50)
**Issues:**
- Hardcoded histogram bins at 4 locations
- Duplicate generation and visualization cells
- Cell 50 contains just `d`
- Cell 49 data perturbation hack

---

## Data Flow

```
data.csv → B (preprocessing) → C (config) → D (model definition)
         → E (training execution) → F (visualization)
```

**Critical flow variables:**
- `delta`: B → D3 → E → F (global dependency)
- `WINDOW_LENGTH`: C → B (rolling window) → D1 (output dim) → D2 (input dim)
- `OUTPUT_SCALE` (0.1): D3 (3 sites) → E (1 site)

---

## Build Order (Dependency Tiers)

### Tier 1: No Dependencies (Safe to Do First)
- Remove duplicate imports (Component A)
- Fix data path to `./data.csv` (Component B)
- Remove Dropout from critic (Component D2)
- Change `diff_method='backprop'` (Component D1)
- Add `weights_only=True` to `torch.load` (Component D3)
- Remove `exit()` call (Component E)
- Remove `eval()` usage (Component E)
- Remove debug Cell 50 `d` (Component F)

### Tier 2: Depends on Tier 1
- Fix `normalize()` to return `(data, mu, sigma)` + update all call sites (Component B)
- Fix EMD to use raw samples + remove hardcoded bins (Component D3 + F)
- Fix memory leak: `.item()` on losses before appending (Component D3)
- Extract `OUTPUT_SCALE = 0.1` to config cell (Component C + D3)
- Fix checkpoint to save/load `model.critic` not `model.discriminator` (Component D3)
- Remove dead `compute_gradient_penalty` method (Component D)
- Remove unused `self.measurements` from `__init__` (Component D)

### Tier 3: Depends on Tier 2
- Restore `n_critic=5`, `LAMBDA=10`, fix LR ratio (Component C)
- Derive `WINDOW_LENGTH = 2 * NUM_QUBITS` (Component C)
- Remove redundant IQP RZ → add data re-uploading → update `count_params()` (Component D1)
- Expand noise to `[0, 4pi]` at all call sites (Component D1 + D3 + E)
- Add PauliX measurements alongside PauliZ (Component D1)
- Add evaluation frequency guard (every N epochs) (Component D3)
- Switch early stopping to monitor EMD (Component D3)
- Add `torch.no_grad()` around evaluation generation (Component D3)

### Tier 4: Depends on Tier 3
- Fix `delta` consistency in generation cells (Component E)
- Unify denormalization strategy (training eval vs standalone generation) (Component D3 + E)
- Replace `* 0.1` with `OUTPUT_SCALE` in generation cells (Component E)
- Remove debug function cell and duplicate generation cells (Component E)

### Tier 5: Final Cleanup
- Remove duplicate visualization cells (Component F)
- Remove Cell 49 data perturbation hack (Component F)
- Remove dead `train_qgan()` method (Component D)
- Consistent naming (`N_CRITIC` not `n_critic`) (Component C)

---

## Three Remediation Phases

### Phase 1: Foundation (Tiers 1 + 2)
**Goal:** Clean infrastructure, correct metrics, safe checkpoint loading.
**Risk:** Low — no architectural changes.
**Components touched:** A, B, C (partial), D (partial), E (partial), F (partial)

### Phase 2: WGAN-GP Correctness + Circuit Redesign (Tier 3)
**Goal:** Restore standard WGAN-GP training, redesign quantum circuit.
**Risk:** Medium — circuit changes invalidate existing checkpoints; training behavior changes significantly.
**Components touched:** C, D1, D2, D3

### Phase 3: Post-Processing Consistency + Cleanup (Tiers 4 + 5)
**Goal:** Unify scaling/denormalization pipeline, remove all debug artifacts.
**Risk:** Low — cleanup only.
**Components touched:** C, D, E, F

---

## Key Architectural Risks

1. **Circuit redesign changes parameter count** → invalidates `checkpoints_phase2c/` checkpoints → use new checkpoint directory
2. **`normalize()` signature change is breaking** → must update all call sites atomically
3. **Early stopping switch to EMD** requires EMD to be non-empty at check time → add guard for empty list
4. **`WINDOW_LENGTH = 2 * NUM_QUBITS`** cascades to DataLoader rolling window → verify measurement count stays consistent after adding multi-qubit measurements

---
*Research completed: 2026-02-26*
