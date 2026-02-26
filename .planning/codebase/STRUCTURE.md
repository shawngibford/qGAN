# Codebase Structure

**Analysis Date:** 2026-02-26

## Directory Layout

```
/Users/shawngibford/dev/phd/qGAN/
├── qgan_pennylane.ipynb                    # Main/latest notebook - full training pipeline
├── qgan_pennylane_qutrit_phase2c.ipynb     # Phase 2C - optimal 3-qutrit configuration
├── qgan_pennylane_qutrit_phase2b.ipynb     # Phase 2B - 4-qutrit variant
├── qgan_pennylane_qutrit_phase2.ipynb      # Phase 2 - original qutrit implementation
├── qgan_pennylane qutrit.ipynb             # Earlier qutrit exploration
├── data.csv                                 # Raw optical density time series data
├── fake.csv                                 # Generated synthetic samples (output)
├── real.csv                                 # Real data extracted for comparison
├── load_checkpoint_phase2c.py               # Utility script to load Phase 2C checkpoint
├── early_stopping_code.py                   # Early stopping implementation (reusable)
├── README.md                                # Project overview and setup instructions
├── LICENSE                                  # Licensing information
├── .planning/                               # GSD planning documents (this directory)
│   └── codebase/
├── archive/                                 # Previous versions and experimental code
│   ├── qgan_pennylane.py                   # Standalone Python version of main notebook
│   ├── qgan_pennylane_SEL.py               # Strongly entangled layer variant
│   ├── qwxlstmgan_pennylane.py             # XLSTM-integrated variant
│   ├── oldqgan.py                          # Original implementation
│   ├── test_quantum_gradients.py           # Gradient flow testing
│   ├── test_generator_comparison.py        # Generator variant comparison
│   ├── ALGORITHM.md                        # Algorithm specification
│   ├── plan.md                             # Development planning notes
│   ├── notes.txt                           # Implementation notes
│   ├── Quantum_Generative...pdf            # Reference paper
│   ├── qgan_cirq.ipynb                     # Google Cirq implementation attempt
│   └── xlstm-main copy/                    # XLSTM repository archive
├── checkpoints_phase2c/                    # Trained model checkpoints
│   ├── qgan_best.pt                        # Best model (lowest discriminator loss)
│   ├── qgan_epoch50.pt                     # Periodic checkpoints
│   ├── qgan_epoch100.pt
│   ├── qgan_epoch150.pt
│   ├── qgan_epoch200.pt
│   ├── qgan_epoch250.pt
│   ├── qgan_epoch300.pt
│   ├── qgan_epoch350.pt
│   ├── qgan_epoch400.pt
│   ├── qgan_epoch450.pt
│   └── qgan_epoch500.pt
├── Final Results from 2000 epochs - IQP:SEL circuit/  # Results visualization directory
│   ├── Figure_2.png through Figure_21.png  # Training/evaluation plots
├── pennylane_docs/                         # PennyLane documentation backup
│   ├── gradients copy/
│   ├── measurements copy/
│   ├── qnn copy/
│   ├── resource copy/
│   ├── tape copy/
│   └── templates copy/
├── qgan_env/                               # Python virtual environment
├── PHASE1_CHANGES_AT_A_GLANCE.md           # Phase 1 qutrit implementation summary
├── PHASE1_QUTRIT_IMPLEMENTATION_SUMMARY.md # Detailed Phase 1 documentation
├── PHASE2_IMPLEMENTATION_SUMMARY.md        # Phase 2 qutrit expansion notes
├── PHASE2_RESULTS.md                       # Phase 2 performance metrics
├── PHASE2_VARIANTS_COMPARISON.md           # Comparison of Phase 2 variants
├── PHASE2C_BREAKTHROUGH_RESULTS.md         # Phase 2C optimal results
├── PHASE2C_RECOVERY_INSTRUCTIONS.md        # Recovery procedure for Phase 2C
├── EARLY_STOPPING_GUIDE.md                 # Early stopping documentation
├── .gitignore                              # Git ignore patterns
├── .vscode/                                # VSCode settings
└── .claude/                                # Claude workspace configuration
```

## Directory Purposes

**Root Directory:**
- Purpose: Main project root containing active notebooks and data
- Contains: Jupyter notebooks (current development), training data, output CSV files
- Key files: `qgan_pennylane.ipynb` (main entry point), `data.csv` (raw input)

**archive/:**
- Purpose: Historical versions, experimental variants, and reference materials
- Contains: Legacy Python scripts, alternative implementations (Cirq, XLSTM), testing utilities
- Key files: `qgan_pennylane.py` (standalone Python version), `ALGORITHM.md` (algorithm spec)

**checkpoints_phase2c/:**
- Purpose: Persistent storage of trained model weights and optimizer states
- Contains: PyTorch .pt files with qGAN parameters, discriminator weights, optimizer states
- Key files: `qgan_best.pt` (best validation checkpoint), `qgan_epoch*.pt` (periodic saves)

**Final Results from 2000 epochs - IQP:SEL circuit/:**
- Purpose: Training visualization and performance plots
- Contains: PNG figures showing loss curves, distribution comparisons, statistical analyses
- Key files: Figure_* (numbered matplotlib outputs from training)

**pennylane_docs/:**
- Purpose: Local backup of PennyLane documentation for offline reference
- Contains: Markdown and HTML documentation for gradients, measurements, QNN, templates
- Key files: Various PennyLane API documentation

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents for future implementation phases
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md
- Used by: GSD planning and execution tools

## Key File Locations

**Entry Points:**
- `qgan_pennylane.ipynb`: Main Jupyter notebook - complete training pipeline from data loading through evaluation
- `qgan_pennylane_qutrit_phase2c.ipynb`: Latest optimized variant with best performance (Phase 2C)
- `load_checkpoint_phase2c.py`: Script to restore trained model for inference

**Configuration:**
- `data.csv`: Raw training data (optical density time series measurements)
- `requirements.txt`: Python dependencies (implicitly referenced, not present but documented in README)
- `qgan_env/`: Virtual environment containing installed packages

**Core Logic:**
- `qGAN` class: Defined in notebooks and `archive/qgan_pennylane.py`
  - Location: Cells in main notebook or Python file starting around line 382
  - Contains: `__init__()`, `define_critic_model()`, `define_generator_circuit()`, `train_qgan()`, `_train_one_epoch()`
- Preprocessing pipeline: Data transformation cells in notebooks
- Training loop: `_train_one_epoch()` method with critic and generator updates

**Utilities:**
- `early_stopping_code.py`: Reusable EarlyStopping class for training
- `load_checkpoint_phase2c.py`: Model restoration utilities

**Testing/Validation:**
- `archive/test_quantum_gradients.py`: Gradient flow validation
- `archive/test_generator_comparison.py`: Generator variant comparison
- Output CSV files: `real.csv`, `fake.csv` (generated samples for validation)

## Naming Conventions

**Files:**
- Notebooks: `qgan_pennylane[_modifier].ipynb` (e.g., `qgan_pennylane_qutrit_phase2c.ipynb`)
- Python scripts: Lowercase with underscores (e.g., `load_checkpoint_phase2c.py`, `early_stopping_code.py`)
- Data files: Lowercase descriptive names (e.g., `data.csv`, `real.csv`, `fake.csv`)
- Checkpoints: `qgan_[descriptor].pt` (e.g., `qgan_best.pt`, `qgan_epoch50.pt`)
- Documentation: UPPERCASE_WITH_UNDERSCORES.md (e.g., `PHASE2C_BREAKTHROUGH_RESULTS.md`)

**Directories:**
- Archive: `archive/` - lowercase, contains old/experimental code
- Checkpoints: `checkpoints_phase[number]/` - phase-specific naming
- Results: Descriptive full paths with spaces (e.g., `Final Results from 2000 epochs - IQP:SEL circuit/`)
- Phase docs: Inline in root with PREFIX_DESCRIPTOR.md pattern

**Python Classes/Methods:**
- Classes: PascalCase (e.g., `qGAN`, `EarlyStopping`)
- Methods: snake_case (e.g., `train_qgan()`, `define_critic_model()`, `encoding_layer()`)
- Parameters: snake_case (e.g., `num_qubits`, `batch_size`, `window_length`)

**Variables:**
- Tensors: Descriptive snake_case (e.g., `real_batch`, `generated_sample`, `critic_loss`)
- Numpy arrays: Append `_np` suffix (e.g., `OD_delta_np`, `OD_log_delta_np`)
- Indices: Short letters (e.g., `i`, `qubit`, `epoch`, `idx`)
- Hyperparameters: UPPERCASE (e.g., `WINDOW_LENGTH`, `EPOCHS`, `NUM_QUBITS`)

## Where to Add New Code

**New Feature Implementation:**
- Primary code: Add to new notebook cell in main `qgan_pennylane.ipynb` or create new `qgan_pennylane_[variant].ipynb`
- Quantum circuit changes: Modify `define_generator_circuit()` method or `encoding_layer()` method
- Discriminator changes: Modify `define_critic_model()` in qGAN class
- Tests: Add to `archive/test_*.py` files or create new test notebook

**New Module/Utility:**
- Reusable classes: Add to root-level Python script (e.g., `early_stopping_code.py` pattern)
- Complex functions: Either add to notebook or extract to `archive/` script for code reuse
- Reference materials: Place in `archive/` with descriptive naming

**Configuration & Hyperparameters:**
- Training parameters: Define as variables in notebook cells (e.g., `WINDOW_LENGTH = 10`)
- Model architecture: Hardcoded in class methods - no separate config file currently
- Data paths: Hardcoded in notebook cells (e.g., `data = pd.read_csv('data.csv')`)

**Data & Results:**
- Input data: Place in root directory with CSV format (e.g., `data.csv`)
- Generated/output data: Root directory (e.g., `real.csv`, `fake.csv`)
- Checkpoints: Organize in `checkpoints_phase[N]/` subdirectories
- Visualizations: Auto-saved to results directories named descriptively

## Special Directories

**checkpoints_phase2c/:**
- Purpose: Model persistence across training sessions
- Generated: Automatically during training via EarlyStopping class
- Committed: Yes (best models tracked in git)
- Naming: `qgan_best.pt` (best validation), `qgan_epoch*.pt` (periodic saves every 50 epochs)

**archive/:**
- Purpose: Historical record of implementation attempts and variants
- Generated: Manual - developer moves completed/deprecated code here
- Committed: Yes (full history preserved)
- Cleanup: Retains working variants (qgan_pennylane.py, test_*.py) for reference

**qgan_env/:**
- Purpose: Python virtual environment with dependencies
- Generated: Via `python -m venv qgan_env` and `pip install -r requirements.txt`
- Committed: No (listed in .gitignore)
- Activation: `source qgan_env/bin/activate` (macOS/Linux)

**pennylane_docs/:**
- Purpose: Offline documentation reference for development
- Generated: Manual download/backup of PennyLane API docs
- Committed: Yes (reference materials)
- Usage: Read-only reference during development

---

*Structure analysis: 2026-02-26*
