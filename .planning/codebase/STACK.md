# Technology Stack

**Analysis Date:** 2026-02-26

## Languages

**Primary:**
- Python 3.8+ (tested on 3.14.2) - Core implementation language for quantum GAN model training

**Secondary:**
- Jupyter Notebook - Interactive experimentation and model training environment

## Runtime

**Environment:**
- Python 3.8+ required (latest tested: 3.14.2)
- Virtual environment: `qgan_env/` (Python venv)

**Package Manager:**
- pip (Python package manager)
- Lockfile: `requirements.txt` (present, pinned versions)

## Frameworks

**Core:**
- PennyLane 0.32.0+ - Quantum computing framework (version 0.44.0 installed)
  - Purpose: Parameterized quantum circuits, IQP encoding, strongly entangled layers
  - Provides quantum generator backend for the GAN

- PyTorch 2.0.0+ - Deep learning framework (version 2.8.0 installed)
  - Purpose: Classical discriminator, optimization, tensor operations, Wasserstein loss with gradient penalty

- TorchVision 0.15.0+ - Computer vision utilities (version 0.23.0 installed)
  - Purpose: Image and data preprocessing utilities

**Testing:**
- pytest 7.0.0+ - Commented out in requirements.txt (optional, not currently configured)

**Build/Dev:**
- Jupyter - Interactive notebook environment for model development
- IPython/IPykernel - Interactive Python kernel (optional, commented in requirements.txt)

## Key Dependencies

**Critical:**
- pennylane 0.32.0+ (installed: 0.44.0) - Quantum circuit framework, parameter optimization
  - pennylane-lightning 0.44.0 - Fast quantum simulation backend
- torch 2.0.0+ (installed: 2.8.0) - Neural network training, Wasserstein GAN discriminator
- torchvision 0.15.0+ (installed: 0.23.0) - Data loading and preprocessing utilities

**Data Manipulation & Analysis:**
- numpy 1.24.0+ (installed: 1.26.4) - Numerical computing, array operations
- pandas 2.0.0+ (installed: 2.3.2) - Data loading, CSV handling, time series manipulation
- scipy 1.10.0+ (installed: 1.15.3) - Statistical functions, distance metrics
  - scipy.stats - Statistical tests (Kolmogorov-Smirnov, Wasserstein distance, entropy)
  - scipy.spatial.distance - Jensen-Shannon divergence, Euclidean distance
  - scipy.special - Lambert W transformation for data preprocessing

**Visualization:**
- matplotlib 3.7.0+ (installed via seaborn) - Plotting and data visualization
- seaborn 0.12.0+ - Statistical data visualization, plot styling

**Time Series Analysis:**
- fastdtw 0.3.4+ - Fast Dynamic Time Warping for sequence comparison
- dtaidistance 2.3.10+ - Distributed Time Alignment and Indexing distance computation
- statsmodels 0.14.0+ - Statistical modeling, ARIMA, time series analysis

**Development Utilities:**
- torchinfo 1.8.0 - Model architecture inspection
- torch-summary - Model summary generation

## Configuration

**Environment:**
- No `.env` file required - configuration is hardcoded in notebooks
- Virtual environment: `qgan_env/` directory (created via `python -m venv qgan_env`)
- Activation: `source qgan_env/bin/activate` (macOS/Linux) or `qgan_env\Scripts\activate` (Windows)

**Build:**
- No build configuration files present
- Direct execution via Jupyter notebook or Python scripts
- Checkpoints saved to `checkpoints_phase2c/` directory for model persistence

## Platform Requirements

**Development:**
- Python 3.8+ with pip
- macOS/Linux/Windows with shell access
- 8GB+ RAM recommended for quantum circuit simulation
- CPU or GPU (PyTorch supports both, CPU-only mode by default)

**Production:**
- Same as development - no separate deployment configuration
- Checkpoint loading via `load_checkpoint_phase2c.py` script
- CSV data files for training (e.g., `data.csv`, `real.csv`, `fake.csv`)

---

*Stack analysis: 2026-02-26*
