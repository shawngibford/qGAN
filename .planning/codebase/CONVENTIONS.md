# Coding Conventions

**Analysis Date:** 2026-02-26

## Overview

This codebase follows Python conventions primarily for scientific computing and deep learning research using PyTorch and PennyLane. The code is primarily organized in Jupyter notebooks with supporting Python scripts for utilities and standalone testing.

## Naming Patterns

**Files:**
- Notebook files: kebab-case with descriptive suffixes
  - Example: `qgan_pennylane.ipynb`, `qgan_pennylane_qutrit_phase2c.ipynb`
  - Archive files: same pattern, located in `archive/` directory
- Utility/standalone scripts: snake_case
  - Example: `early_stopping_code.py`, `load_checkpoint_phase2c.py`, `test_quantum_gradients.py`

**Functions:**
- snake_case for all function names
  - Example: `normalize()`, `inverse_lambert_w_transform()`, `analyze_gradients()`, `train_generator()`
  - Helper functions with underscore prefix for private usage
  - Example: `_save_checkpoint()`, `_train_one_epoch()`

**Variables:**
- snake_case for local and module-level variables
  - Example: `batch_size`, `loss`, `gradient_norms`, `real_data`, `fake_data`
  - Single-letter variables acceptable only in mathematical contexts (e.g., `x`, `y` for coordinates, `i` for loop indices)
  - Meaningful names preferred: `window_length` not `wl`, `num_epochs` not `ne`

**Classes:**
- PascalCase for all class names
  - Example: `EarlyStopping`, `QuantumGenerator`, `MLPGenerator`, `Critic`, `qGAN`
  - PyTorch modules inherit from `nn.Module`
  - Example: `class QuantumGenerator(nn.Module):`

**Constants:**
- UPPERCASE_WITH_UNDERSCORES for module-level constants
  - Example in `early_stopping_code.py`: Random seed at module level
  - Specific values like `seed = 42` used at module init

## Code Style

**Formatting:**
- No explicit formatter configured (black, pylint, flake8 not in dependencies)
- Follows Python conventions implicitly:
  - 4-space indentation (consistent across all files)
  - Maximum line length varies (some lines exceed 100 characters, not strictly enforced)
  - Blank lines between major sections (e.g., between function definitions and classes)

**Spacing:**
- Two blank lines before top-level class/function definitions
- One blank line before method definitions within classes
- Comments separated from code by at least one space

**Import Organization:**

**Order (observed in files like `archive/qgan_pennylane.py` and `archive/test_generator_comparison.py`):**
1. Standard library imports (math, random, time)
2. NumPy and pandas
3. Matplotlib and visualization (matplotlib, seaborn)
4. Scientific computing (scipy, statsmodels)
5. Data manipulation (fastdtw, dtaidistance)
6. PyTorch (torch, torch.nn, torch.optim)
7. PennyLane (pennylane)

**Pattern observed in `archive/test_generator_comparison.py`:**
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from torch.autograd import grad
import pandas as pd
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')
```

**Path Aliases:**
- No path aliases detected (no `~` or `@` imports observed)
- Absolute imports used for local modules
- Relative paths used in notebooks for file access

## Error Handling

**Patterns:**
- Try/except blocks used sparingly in production code
- Warnings suppressed at module level when needed
  - Example: `warnings.filterwarnings('ignore')` in `archive/test_generator_comparison.py` line 17
- Explicit checks for None/missing values before operations
  - Example in `early_stopping_code.py`: `if checkpoint['params_pqc'] is not None:`
  - Example: `if hasattr(model, 'state_dict'):` checks before attribute access
- Assertions used for tensor shape validation in quantum code
  - Example: Output shape validation in QuantumGenerator.forward()

**Assertions:**
- Used to verify critical assumptions
- Example in `test_quantum_gradients.py`: Checking gradient flow with `assert`-like patterns
- Conditional checks with meaningful error messages to stderr/print

## Logging

**Framework:** print() for standard logging (no logging module imports detected)

**Patterns:**
- Print statements used for progress updates and diagnostic information
- Example from `early_stopping_code.py`:
  ```python
  print(f'✓ Epoch {epoch}: Improved! Score: {score:.6f} (best so far)')
  print(f'  Epoch {epoch}: No improvement. Score: {score:.6f}')
  print(f'🛑 Early stopping triggered! No improvement for {self.patience} epochs.')
  ```
- Unicode emoji characters used for visual indicators in output
  - ✓ (checkmark), ⚠️ (warning), 🛑 (stop), ✅ (success), ❌ (error), 💾 (save)
  - 📊 (chart), 🔄 (refresh), ➡️ (arrow), 🔍 (search), 🧪 (test), 🚀 (rocket)
- Formatted strings with f-strings for dynamic output
- `.format()` style occasionally used (older code in archive)

**Numeric Formatting:**
- Decimal precision specified in f-strings: `.4f`, `.6f`, `.8f` for floats
- Scientific notation used for very small values: `1e-8`, `1e-10`

## Comments

**When to Comment:**
- Block comments above logical sections explain purpose
  - Example: `# ==================== TRAINING ====================`
  - Example: `# Create checkpoint directory`
- Inline comments explain non-obvious logic
  - Example: `# Store original parameters` before assignment
  - Example: `# Fix: Wrap params_pqc as Parameter` for workarounds
- Data manipulation steps have inline comments
  - Example: `# Pad or truncate output to match output_dim`

**Comment Style:**
- Single `#` with space after for inline comments
- Multiple `#` or `-` for section dividers
  - Example: `# ==================== VARIABLE NAME ====================` (22 equals signs for separator)
- Capitalized comment text (sentences with periods)
  - Example: `# Create checkpoint directory.` (some inconsistency - some lack periods)

**Docstrings:**
- Google-style docstrings used for classes and functions
- Example from `early_stopping_code.py`:
  ```python
  class EarlyStopping:
      """
      Early stopping to stop training when monitored metric doesn't improve.

      Args:
          patience (int): How many epochs to wait after last improvement. Default: 50
          min_delta (float): Minimum change to qualify as improvement. Default: 0.001
          mode (str): 'min' for loss (lower is better), 'max' for accuracy. Default: 'min'
          checkpoint_dir (str): Directory to save checkpoints. Default: 'checkpoints'
          save_every (int): Save checkpoint every N epochs regardless. Default: 50
      """
  ```
- Method docstrings include Args, Returns sections
- Example:
  ```python
  def __call__(self, epoch, score, model, model_name='model'):
      """
      Check if training should stop and save checkpoints.

      Args:
          epoch (int): Current epoch number
          score (float): Metric value to monitor (e.g., discriminator loss)
          model: The model object to save
          model_name (str): Prefix for checkpoint filename

      Returns:
          bool: True if training should stop, False otherwise
      """
  ```

**Docstring Standards:**
- Triple double-quotes `"""` for docstrings (not single quotes)
- Brief one-line summary followed by blank line, then detailed description
- Parameter documentation with type and description
- Return value documentation when applicable
- Example-heavy documentation in usage blocks at end of files

## Function Design

**Size:**
- Functions tend to be focused on single tasks
- Utility functions 5-30 lines (normalize, transform functions)
- Class methods range 10-50 lines
- Training functions may exceed 50 lines due to nested logic

**Parameters:**
- Defaults provided for non-critical parameters
  - Example: `def __init__(self, patience=50, min_delta=0.001, mode='min', ...)`
  - Example: `def train_generator(..., epochs=50, lr_g=1e-3, lr_c=1e-4):`
- **kwargs rarely used; explicit parameters preferred
- Position: required parameters first, then parameters with defaults

**Return Values:**
- Single return values for utility functions
- Tuples/dicts for multiple return values
  - Example: `def analyze_gradients(model, loss, model_name):` returns `(gradients, grad_norms)`
  - Example: Training returns dict with multiple metrics
- None implicitly returned when no return statement (side-effect functions like `_save_checkpoint()`)
- Type hints not used explicitly but implicit from docstrings

## Module Design

**Exports:**
- No `__all__` declarations observed
- Classes and functions intended for use are at module level
- Private functions prefixed with underscore: `_save_checkpoint()`
- Usage examples provided in docstring comments at end of files

**Notebook Structure:**
- Code cells grouped by functionality
- Markdown cells separate major sections with headers
- Common pattern: data loading → preprocessing → model definition → training → evaluation
- Each notebook is self-contained with complete imports and data loading

**Utility Scripts:**
- Standalone scripts in root or archive directories
- `early_stopping_code.py`: Designed for copy-paste into notebooks
- `test_quantum_gradients.py`, `test_generator_comparison.py`: Standalone test utilities with `if __name__ == "__main__":` guards

**Code Organization Pattern - Archives:**
- Older implementations preserved in `archive/` directory
- Consistent naming: `qgan_pennylane.py` (original), `qgan_pennylane_SEL.py` (variant), `qgan_pennylane_qutrit_phase2.ipynb` (phase variant)
- Phase tracking in file names: `phase2`, `phase2b`, `phase2c` for experimental iterations

---

*Convention analysis: 2026-02-26*
