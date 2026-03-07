# Phase 1: Foundation and Correctness Infrastructure - Research

**Researched:** 2026-02-26
**Domain:** PyTorch/PennyLane notebook correctness, checkpoint management, code safety
**Confidence:** HIGH

## Summary

Phase 1 addresses 14 requirements across bug fixes, performance improvements, and code quality issues in `qgan_pennylane.ipynb` (51 cells). The notebook contains a single `qGAN` class (cell 23) that defines a quantum GAN using PennyLane for the generator and PyTorch for the critic. The fixes are all infrastructure-level: no training behavior or model architecture changes.

The most architecturally significant change is the DataLoader restructuring (PERF-03), which replaces a broken pattern where DataLoader is immediately flattened to a list and then randomly sampled one element at a time. The checkpoint system (BUG-01) needs both key renaming (`discriminator` to `critic`) and proper parameter restoration that preserves gradient tracking. The remaining fixes are localized: removing `exit()`, fixing a hardcoded epoch number, scoping a global variable, wrapping evaluation in `torch.no_grad()`, and standardizing naming conventions.

**Primary recommendation:** Fix checkpoint save/load first (BUG-01 + QUAL-05), then DataLoader restructuring (PERF-03), then all remaining fixes in any order. The hyperparameter rename (QUAL-07) and variable rename (QUAL-10) touch many cells and should be done last to minimize merge conflicts with other fixes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Clean break: only support new `critic` key in save/load. No backward compatibility with old `discriminator` key — Phase 2 invalidates all checkpoints anyway
- Save weights + full training state: model weights, optimizer states, epoch number, and current losses
- Timestamped filenames: e.g., `checkpoint_2026-02-26_14-30.pt`
- Keep latest checkpoint only — overwrite on each save to prevent disk bloat
- Single hyperparameter config cell near the top of the notebook with all UPPER_CASE constants
- Replace unsafe code in place (don't delete cells) — preserves cell count and ordering for cross-referencing with notes
- Add markdown section header cells to organize the notebook into logical sections (Imports, Config, Data Loading, Model Definition, Training Loop, Evaluation)
- Replace `eval()` with explicit dictionary lookup mapping string names to values
- Use DataLoader for proper batching — remove the flatten-to-list hack where DataLoader is immediately iterated into `gan_data_list`
- Keep current `batch_size` hyperparameter value unchanged
- Sequential window order (shuffle=False) — preserve temporal locality within epochs
- Windows are already independent samples from rolling window preprocessing; DataLoader manages batch assembly
- Stage-based naming for data variables: `raw_data` (CSV load), `log_delta` (after log transform), `scaled_data` (after normalization), `windowed_data` (after rolling window)
- Flat UPPER_CASE for all hyperparameters: `N_CRITIC`, `LAMBDA`, `BATCH_SIZE`, `NUM_QUBITS`, `WINDOW_LENGTH`, `NUM_LAYERS`, `NUM_EPOCHS` — no domain prefixes
- Remove unused code AND stale comments that reference removed variables (clean slate)

### Claude's Discretion
- Data file path: Claude decides on `./data.csv` vs `./data/data.csv` based on current file location and project structure
- Exact markdown section header wording
- Ordering of hyperparameters within the config cell
- Specific dict structure for eval() replacement

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BUG-01 | Checkpoint saves and loads `model.critic` (not `model.discriminator`) | Checkpoint save uses `discriminator` key (line 1807); load references `model.discriminator` (line 1841). Both must change to `critic`. The `params_pqc` save uses `detach().clone()` which is correct for serialization, but load must use `nn.Parameter()` wrapper AND re-register with optimizer. See Architecture Pattern 1. |
| BUG-04 | Loss values stored as Python floats (`.item()`) not tensors retaining computation graphs | Some loss storage already uses `.item()` (lines 1357, 1364) but the `generator_loss` appended at line 383 may still be a tensor. All `*_loss_avg.append()` calls must use `.item()`. See Pitfall 3. |
| BUG-05 | Epoch condition uses `self.num_epochs` instead of hardcoded `3000` | Line 1350: `if epoch % 100 == 0 or epoch + 1 == 3000:` must change to `self.num_epochs`. Single line fix. |
| BUG-06 | `delta` variable scoped inside class as `self.delta` (no global dependency) | Cell 17 defines `delta = 1` as global. Cell 23 line 423 uses bare `delta` inside class method `_train_one_epoch`. Must pass as `self.delta` constructor parameter or accept as method argument. See Architecture Pattern 2. |
| BUG-07 | `exit()` call removed from notebook cells | Line 2829 (cell 29) contains `exit()`. Replace with descriptive comment or status message. Do NOT delete the cell (user decision: replace in place). |
| PERF-02 | All evaluation/inference forward passes wrapped in `torch.no_grad()` | Lines 401-435 in cell 23 perform evaluation forward passes (generator calls for metrics) without `torch.no_grad()`. Post-training cells (29, 32-35) have partial coverage. See Architecture Pattern 3. |
| PERF-03 | DataLoader used with proper batch sampling (not flattened to list) | Lines 1150-1155 and 1927-1931 flatten DataLoader to list. Lines 1180-1181 randomly index the list. Line 1672 creates DataLoader with batch_size=1. Must restructure to iterate DataLoader batches directly. See Architecture Pattern 4. |
| QUAL-01 | Data path changed to relative `./data.csv` | Line 189: hardcoded `/Users/shawngibford/dev/qml/qGAN/data.csv`. File exists at project root as `data.csv`. Change to `./data.csv`. Single line fix. |
| QUAL-02 | Duplicate imports removed (`numpy`, `random`) | Cell 2 lines 104/115: duplicate `import numpy as np`; lines 103/118: duplicate `import random`. Cell 26 line 1702: another `import numpy as np`. Remove duplicates keeping first occurrence. |
| QUAL-04 | `eval()` replaced with `globals().get()` or explicit logic | Line 3094 (cell 33): `var_value = eval(var_name)` for variable existence check. Replace with explicit dict or `globals().get()`. See Code Example 3. |
| QUAL-05 | `torch.load` uses `weights_only=True` | Line 1832: `torch.load(filepath)` without `weights_only=True`. Verified: PyTorch 2.10.0 supports `weights_only=True` and it works with optimizer state_dicts. Single parameter addition. |
| QUAL-07 | Hyperparameter naming consistent (all UPPER_CASE) | Cell 24 mixes: `n_critic` (lowercase), `LAMBDA`, `EPOCHS` (not `NUM_EPOCHS`), `LR_CRITIC`, `LR_GENERATOR`. Constructor uses `gp` parameter name for lambda. Must rename all to UPPER_CASE and update all references. See Architecture Pattern 5. |
| QUAL-09 | Unused `self.measurements` removed from `__init__` | Lines 945-949: `self.measurements` list populated with `qml.expval()` objects in `__init__` but never referenced again. Actual measurements are built locally in the circuit function (lines 1103-1109). Safe to remove. |
| QUAL-10 | Variable `data` not silently overwritten (use distinct names) | `data` assigned at line 189 (CSV), line 198 (dropna), line 360 (numpy conversion). Must use stage-based names per user decision: `raw_data`, etc. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | Neural network framework, optimizer, checkpointing | Already installed; all model code uses it |
| PennyLane | 0.44.0 | Quantum circuit definition and differentiation | Already installed; generator circuit uses it |
| pandas | (installed) | CSV data loading | Already used for `pd.read_csv()` |
| numpy | (installed) | Array operations, random noise generation | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.data | (part of PyTorch) | Dataset and DataLoader | PERF-03: proper batching |
| pathlib.Path | (stdlib) | File path handling | Checkpoint path construction |
| datetime | (stdlib) | Timestamp generation | Checkpoint filename timestamps |

### Alternatives Considered
None applicable. This phase uses only libraries already in the project. No new dependencies needed.

## Architecture Patterns

### Pattern 1: Checkpoint Save/Load with Gradient Preservation
**What:** Save model state including `params_pqc` as detached tensor, restore as `nn.Parameter`, and re-register with optimizer
**When to use:** Every checkpoint save and load operation

**Save (correct pattern):**
```python
from datetime import datetime

def save_checkpoint(model, epoch, losses, checkpoint_dir='checkpoints'):
    """Save checkpoint with timestamped filename, keeping only latest."""
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Remove previous checkpoint
    for old_ckpt in Path(checkpoint_dir).glob('checkpoint_*.pt'):
        old_ckpt.unlink()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filepath = Path(checkpoint_dir) / f'checkpoint_{timestamp}.pt'

    checkpoint = {
        'epoch': epoch,
        'params_pqc': model.params_pqc.detach().clone(),  # detach is correct for saving
        'critic_state': model.critic.state_dict(),         # 'critic' not 'discriminator'
        'c_optimizer': model.c_optimizer.state_dict(),
        'g_optimizer': model.g_optimizer.state_dict(),
        'critic_loss': losses.get('critic_loss'),
        'generator_loss': losses.get('generator_loss'),
    }
    torch.save(checkpoint, filepath)
```

**Load (critical: must re-register parameter with optimizer):**
```python
def load_checkpoint(model, checkpoint_dir='checkpoints'):
    """Load latest checkpoint and re-register params_pqc with optimizer."""
    ckpt_files = sorted(Path(checkpoint_dir).glob('checkpoint_*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    filepath = ckpt_files[-1]
    checkpoint = torch.load(filepath, weights_only=True)

    # Restore params_pqc as nn.Parameter (requires_grad=True by default)
    model.params_pqc = torch.nn.Parameter(checkpoint['params_pqc'])

    # Restore critic
    model.critic.load_state_dict(checkpoint['critic_state'])

    # CRITICAL: Re-create optimizers with new parameter references
    # The old optimizer holds references to the OLD params_pqc tensor
    model.c_optimizer.load_state_dict(checkpoint['c_optimizer'])

    # Re-register params_pqc with generator optimizer
    model.g_optimizer = torch.optim.Adam(
        [model.params_pqc],
        lr=model.g_optimizer.defaults['lr'],
        betas=model.g_optimizer.defaults['betas']
    )
    model.g_optimizer.load_state_dict(checkpoint['g_optimizer'])

    return checkpoint['epoch']
```

**Confidence:** HIGH -- verified with PyTorch 2.10.0 that `torch.load(..., weights_only=True)` works with optimizer state dicts.

### Pattern 2: Global Variable Scoping Fix (delta)
**What:** Move `delta = 1` from global scope into `qGAN.__init__` as `self.delta`
**When to use:** BUG-06 fix

```python
# In __init__:
self.delta = delta  # Accept as constructor parameter

# In _train_one_epoch (line 423 equivalent):
original_norm = lambert_w_transform(generated_data, self.delta)
```

The constructor signature gains a `delta` parameter. The instantiation in cell 24 passes it:
```python
qgan = qGAN(NUM_EPOCHS, BATCH_SIZE, WINDOW_LENGTH, N_CRITIC, LAMBDA, NUM_LAYERS, NUM_QUBITS, delta=1)
```

**Confidence:** HIGH -- straightforward Python scoping fix.

### Pattern 3: torch.no_grad() Wrapping for Evaluation
**What:** Wrap all non-training forward passes in `torch.no_grad()` context manager
**When to use:** PERF-02 -- evaluation metrics computation within training loop and all post-training generation/analysis

**In-training evaluation (lines 390-435 of cell 23):**
```python
# After generator training step, before metrics computation:
with torch.no_grad():
    num_samples = len(original_data) // self.window_length
    # ... noise generation ...
    for generator_input in generator_inputs:
        gen_out = self.generator(generator_input, self.params_pqc)
        # ...
    # ... metrics computation (EMD, stylized facts) ...
```

**Confidence:** HIGH -- standard PyTorch pattern. The `no_grad()` context only affects the forward pass; gradients accumulated during the training steps (critic and generator) earlier in the same method are not affected.

### Pattern 4: DataLoader Proper Batching
**What:** Replace flatten-to-list anti-pattern with proper DataLoader iteration
**When to use:** PERF-03

**Current broken pattern:**
```python
# Cell 25: Creates DataLoader with batch_size=1, then immediately flattens
gan_data = DataLoader(TensorDataset(data_tensor), batch_size=1, shuffle=True)

# In train():
gan_data_list = []
for batch in gan_data:
    for sample in batch:
        gan_data_list.append(sample)

# In _train_one_epoch():
random_idx = torch.randint(0, len(gan_data_list), (1,))
real_sample = gan_data_list[random_idx.item()]
```

**Correct pattern:**
```python
# Cell 25: Create DataLoader with actual batch_size
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # sequential per user decision

# In train(): iterate DataLoader per epoch
for epoch in range(self.num_epochs):
    for batch in dataloader:
        real_batch = batch[0]  # shape: (batch_size, window_length)
        self._train_one_epoch(real_batch, original_data, preprocessed_data, epoch)
```

**Key change in `_train_one_epoch`:** The method currently receives `gan_data_list` and picks random samples. With proper DataLoader, it receives a batch tensor directly. The critic training loop (n_critic iterations) should process the same batch or sub-batches from it.

**Confidence:** HIGH -- standard PyTorch DataLoader pattern. The main risk is ensuring the critic inner loop (which currently does n_critic random picks) adapts correctly to batch-based iteration.

### Pattern 5: Hyperparameter Naming Convention
**What:** Rename all hyperparameters to flat UPPER_CASE in a single config cell
**When to use:** QUAL-07

**Current naming (cell 24):**
```python
EPOCHS = 2000       # should be NUM_EPOCHS
n_critic = 1        # should be N_CRITIC
LAMBDA = 0.8        # already correct
LR_CRITIC = 3e-5    # already correct
LR_GENERATOR = 8e-5 # already correct
```

**Target naming:**
```python
# Config cell (all UPPER_CASE)
NUM_EPOCHS = 2000
BATCH_SIZE = 12
WINDOW_LENGTH = 10
NUM_QUBITS = 5
NUM_LAYERS = 2
N_CRITIC = 1
LAMBDA = 0.8
LR_CRITIC = 3e-5
LR_GENERATOR = 8e-5
```

**Constructor parameter rename:**
```python
# Old: qGAN(num_epochs, batch_size, window_length, n_critic, gp, num_layers, num_qubits)
# New: qGAN(num_epochs, batch_size, window_length, n_critic, lambda_gp, num_layers, num_qubits)
```
The `gp` parameter (gradient penalty lambda) should be renamed to `lambda_gp` or similar to be self-documenting. Inside the class, `self.gp` becomes `self.lambda_gp`.

**Ripple effects:** Every reference to `EPOCHS`, `n_critic` (lowercase), and `gp` must be updated. The constructor call, the class internals, and the print statements in cell 24 all need updating.

**Confidence:** HIGH -- mechanical rename with clear rules.

### Anti-Patterns to Avoid
- **Partial rename:** Renaming the config cell but missing references inside the class or in post-training cells. Must do a global search-and-replace for each renamed variable.
- **Deleting cells:** User explicitly chose "replace in place, don't delete cells" to preserve cell ordering for cross-referencing with notes.
- **Changing training behavior:** Phase 1 must NOT alter learning rates, loss functions, n_critic value, gradient penalty lambda, or any other training parameter. Only fix bugs and infrastructure.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batching data | Manual list + random indexing | `torch.utils.data.DataLoader` | Handles batching, shuffling, memory pinning, worker processes |
| Parameter serialization | Custom pickle/JSON | `torch.save` / `torch.load` with `weights_only=True` | Handles tensor serialization safely |
| Gradient control | Manual `tensor.requires_grad_(False)` on each tensor | `torch.no_grad()` context manager | Cleaner, applies to all tensors in scope, auto-restores state |
| Variable existence check | `eval(var_name)` | `globals().get(var_name)` or explicit dict | `eval()` is a security risk and can execute arbitrary code |

**Key insight:** Every "don't hand-roll" item in this phase is replacing a hand-rolled solution with its standard PyTorch/Python equivalent. The hand-rolled versions introduced bugs.

## Common Pitfalls

### Pitfall 1: Checkpoint Load Breaks Optimizer References
**What goes wrong:** After `model.params_pqc = nn.Parameter(checkpoint['params_pqc'])`, the optimizer still holds a reference to the OLD `params_pqc` tensor. Calling `optimizer.step()` updates the old tensor, not the new one. Training appears to proceed but the loaded parameters never change.
**Why it happens:** `nn.Parameter()` creates a new Python object. PyTorch optimizers store references to parameter objects, not names.
**How to avoid:** After loading `params_pqc`, recreate the generator optimizer with the new parameter reference, then load the optimizer state dict into the new optimizer.
**Warning signs:** After loading a checkpoint, verify `model.params_pqc.grad is not None` after a backward pass, and verify the optimizer's param_groups reference the same object as `model.params_pqc`.

### Pitfall 2: torch.no_grad() Placement Too Broad or Too Narrow
**What goes wrong:** Wrapping too much in `no_grad()` silently disables gradient computation for training steps. Wrapping too little leaves gradient accumulation during evaluation, wasting memory and potentially causing OOM.
**Why it happens:** The evaluation code (lines 390-471 in cell 23) is interleaved in the same method as training code.
**How to avoid:** Place `with torch.no_grad():` precisely around the evaluation block (noise generation through metrics computation), AFTER the optimizer step, BEFORE the next epoch.
**Warning signs:** If generator loss stops changing after adding no_grad, the scope is too broad.

### Pitfall 3: Loss Tensors Retaining Computation Graphs
**What goes wrong:** Appending a loss tensor (not `.item()`) to a list prevents the computation graph from being freed. Memory usage grows linearly with epochs. Eventually OOM.
**Why it happens:** `loss_avg.append(loss)` keeps the tensor alive, which keeps the entire computation graph alive.
**How to avoid:** Always use `.item()` when storing loss values for monitoring: `self.critic_loss_avg.append(critic_loss.item())`. Verify by checking `type(self.critic_loss_avg[0])` is `float`, not `torch.Tensor`.
**Warning signs:** Memory usage growing steadily during training. `torch.cuda.memory_summary()` showing increasing cached memory.

### Pitfall 4: DataLoader shuffle=False with Non-Random Critic Training
**What goes wrong:** With `shuffle=False`, the critic sees data in the same order every epoch. Combined with `n_critic > 1`, the critic may overfit to early batches.
**Why it happens:** User explicitly chose `shuffle=False` for temporal locality.
**How to avoid:** This is an accepted tradeoff per user decision. The user is aware. Do NOT override this to shuffle=True. Document it but respect the decision.
**Warning signs:** Not applicable -- this is a deliberate choice.

### Pitfall 5: Renaming Variables in Notebook Cells Without Verifying Output References
**What goes wrong:** Renaming `data` to `raw_data` in cell 4 but forgetting that cell 8 references the old name `data = OD.numpy()`. The notebook breaks when run top-to-bottom.
**Why it happens:** Jupyter notebooks have implicit dependencies between cells through shared global state.
**How to avoid:** After completing all renames, verify with a kernel restart + "Run All Cells" (or at minimum, trace each renamed variable through all downstream cells).
**Warning signs:** `NameError` when running cells in sequence.

### Pitfall 6: PennyLane qml.expval() Called Outside Quantum Context
**What goes wrong:** When removing `self.measurements` (QUAL-09), need to verify no other code references `self.measurements`. The list contains `qml.expval()` objects created in `__init__`, but these are PennyLane operations that should only exist inside a `@qml.qnode` decorated function.
**Why it happens:** The original code created measurement objects in `__init__` but never used them -- the circuit function builds its own measurements locally.
**How to avoid:** Search for ALL references to `self.measurements` before removing. Grep confirms only lines 943-949 reference it.
**Warning signs:** None -- removal is safe since the attribute is write-only.

## Code Examples

### Example 1: Safe Checkpoint Save with Timestamp
```python
from datetime import datetime
from pathlib import Path

def save_checkpoint(model, epoch, checkpoint_dir='checkpoints'):
    """Save training state. Keeps only the latest checkpoint."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Remove previous checkpoints
    for old in ckpt_dir.glob('checkpoint_*.pt'):
        old.unlink()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filepath = ckpt_dir / f'checkpoint_{timestamp}.pt'

    torch.save({
        'epoch': epoch,
        'params_pqc': model.params_pqc.detach().clone(),
        'critic_state': model.critic.state_dict(),
        'c_optimizer': model.c_optimizer.state_dict(),
        'g_optimizer': model.g_optimizer.state_dict(),
        'critic_loss': model.critic_loss_avg[-1] if model.critic_loss_avg else None,
        'generator_loss': model.generator_loss_avg[-1] if model.generator_loss_avg else None,
    }, filepath)
```
**Source:** PyTorch saving/loading tutorial + user decisions from CONTEXT.md

### Example 2: Safe Checkpoint Load with Optimizer Re-registration
```python
def load_checkpoint(model, checkpoint_dir='checkpoints'):
    """Load latest checkpoint. Re-registers params_pqc with optimizer."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob('checkpoint_*.pt'))
    if not ckpt_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None

    checkpoint = torch.load(ckpt_files[-1], weights_only=True)

    # Restore quantum parameters as nn.Parameter (requires_grad=True by default)
    model.params_pqc = torch.nn.Parameter(checkpoint['params_pqc'])

    # Restore critic network
    model.critic.load_state_dict(checkpoint['critic_state'])

    # Restore optimizer states
    model.c_optimizer.load_state_dict(checkpoint['c_optimizer'])

    # CRITICAL: Update generator optimizer to reference new params_pqc
    model.g_optimizer.param_groups[0]['params'] = [model.params_pqc]
    model.g_optimizer.load_state_dict(checkpoint['g_optimizer'])

    return checkpoint['epoch']
```
**Source:** PyTorch checkpoint tutorial; optimizer param_groups manipulation is documented in PyTorch Optimizer docs.

### Example 3: eval() Replacement with globals().get()
```python
# BEFORE (unsafe):
required_vars = ['qgan', 'OD_log_delta', 'transformed_norm_OD_log_delta', 'WINDOW_LENGTH', 'NUM_QUBITS']
for var_name in required_vars:
    try:
        var_value = eval(var_name)  # SECURITY RISK
        print(f"{var_name}: Found")
    except NameError:
        print(f"{var_name}: MISSING!")

# AFTER (safe, using explicit lookup):
required_vars = ['qgan', 'OD_log_delta', 'transformed_norm_OD_log_delta', 'WINDOW_LENGTH', 'NUM_QUBITS']
missing_vars = []
for var_name in required_vars:
    if var_name in globals():
        print(f"{var_name}: Found")
    else:
        print(f"{var_name}: MISSING!")
        missing_vars.append(var_name)
```
**Source:** Python standard library documentation for `globals()`.

### Example 4: DataLoader Proper Usage
```python
# Create dataset and dataloader (cell 25)
dataset = torch.utils.data.TensorDataset(data_tensor)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,       # sequential per user decision
    drop_last=True       # avoid partial batches that could cause shape mismatches
)

# In training loop
for epoch in range(self.num_epochs):
    for (real_batch,) in dataloader:
        # real_batch shape: (BATCH_SIZE, WINDOW_LENGTH)
        self._train_one_epoch(real_batch, original_data, preprocessed_data, epoch)
```
**Source:** PyTorch DataLoader documentation.

### Example 5: Variable Rename Map
```python
# Data pipeline stage names (QUAL-10):
# Old name          -> New name
# data (CSV)        -> raw_data
# data (dropna)     -> raw_data  (in-place operation, same variable is fine)
# data (OD.numpy()) -> REMOVE (cell 8 reassignment to numpy; use OD_log_delta_np or equivalent)
# OD_log_delta      -> log_delta  (user decision: stage-based naming)

# Hyperparameter names (QUAL-07):
# Old name       -> New name
# EPOCHS         -> NUM_EPOCHS
# n_critic       -> N_CRITIC
# gp (constructor param) -> lambda_gp
# self.gp        -> self.lambda_gp
# LAMBDA stays LAMBDA (already UPPER_CASE)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.load(f)` (default) | `torch.load(f, weights_only=True)` | PyTorch 2.0+ | Prevents arbitrary code execution during deserialization |
| `DataLoader` → flatten to list | Iterate `DataLoader` directly | Always was the correct pattern | Proper batching, memory efficiency, GPU pinning |
| `eval()` for variable lookup | `globals().get()` or explicit dict | Always was the correct pattern | Eliminates arbitrary code execution risk |

**Deprecated/outdated:**
- `torch.load()` without `weights_only` parameter: Emits FutureWarning in PyTorch 2.6+; will default to True in future versions. Current PyTorch 2.10.0 already has `weights_only=None` default which is transitional.

## Open Questions

1. **DataLoader inner loop structure for n_critic > 1**
   - What we know: The critic trains `n_critic` times per generator step. Currently each critic step picks one random sample. With DataLoader batching, each critic step should process a full batch.
   - What's unclear: Should all `n_critic` critic steps use the same batch, or should the DataLoader provide `n_critic` different batches? Standard WGAN-GP implementations typically use different batches for each critic step.
   - Recommendation: Use the same batch for all `n_critic` critic steps within one DataLoader iteration. This is simpler and the user chose sequential ordering, so different batches per critic step would require DataLoader restructuring. The current `n_critic=1` makes this moot anyway, and Phase 2 will set `n_critic=5` with potential further DataLoader tuning.

2. **Optimizer re-registration approach**
   - What we know: Two approaches: (a) manipulate `optimizer.param_groups[0]['params']` to point to new parameter, then load state dict; (b) create a brand new optimizer, then load state dict.
   - What's unclear: Approach (a) is more concise but less documented. Approach (b) is more explicit but requires knowing the optimizer hyperparameters.
   - Recommendation: Use approach (a) -- `model.g_optimizer.param_groups[0]['params'] = [model.params_pqc]` then `load_state_dict()`. This preserves the existing optimizer configuration and is a documented pattern in PyTorch forums. If issues arise, fall back to approach (b).

3. **Data file path decision (Claude's discretion)**
   - What we know: `data.csv` exists at project root `/Users/shawngibford/dev/phd/qGAN/data.csv`. The notebook is also at project root. Current path is `/Users/shawngibford/dev/qml/qGAN/data.csv` (a different, likely outdated path).
   - Recommendation: Use `./data.csv`. The file is in the same directory as the notebook. No subdirectory needed.

## Sources

### Primary (HIGH confidence)
- **Codebase inspection:** Direct grep/read of `qgan_pennylane.ipynb` -- all line numbers and patterns verified against actual code
- **PyTorch 2.10.0 runtime verification:** `torch.load(..., weights_only=True)` tested with optimizer state dicts -- works correctly
- **PyTorch 2.10.0 API inspection:** `torch.load` signature confirmed `weights_only` parameter with `None` default

### Secondary (MEDIUM confidence)
- **PyTorch checkpoint best practices:** Standard pattern of save/load state_dict + optimizer state_dict is well-documented in official tutorials
- **DataLoader usage:** Standard PyTorch pattern; verified against project's existing import of `Dataset, DataLoader`

### Tertiary (LOW confidence)
- None. All findings are from direct codebase inspection or verified runtime behavior.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries; all fixes use existing PyTorch/Python stdlib
- Architecture: HIGH -- patterns are standard PyTorch; verified against codebase structure
- Pitfalls: HIGH -- each pitfall identified from actual code inspection, not hypothetical

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain; PyTorch checkpoint API is mature)
