# External Integrations

**Analysis Date:** 2026-02-26

## APIs & External Services

**Scientific Libraries:**
- PennyLane Quantum Computing Framework
  - Purpose: Quantum circuit simulation, parameterized quantum gates, automatic differentiation through quantum circuits
  - SDK/Client: `pennylane` (0.32.0+)
  - Authentication: None required (local simulation)

**No external REST APIs detected** - All computation is local

## Data Storage

**Databases:**
- None configured - Project does not use databases

**File Storage:**
- Local filesystem only
  - Training data: `data.csv` - Time series input for model training
  - Generated samples output: `real.csv`, `fake.csv` - CSV format
  - Checkpoints: `checkpoints_phase2c/` directory - PyTorch `.pt` files
    - Connection: Direct file I/O via `pathlib.Path` and `torch.save()`/`torch.load()`
    - Client: PyTorch's checkpoint serialization

**Caching:**
- None - All intermediate computations are recalculated or stored in memory

## Authentication & Identity

**Auth Provider:**
- Not applicable - No external authentication required
- Implementation: N/A (local-only computation)

## Monitoring & Observability

**Error Tracking:**
- None configured - Standard Python exception handling only

**Logs:**
- Console output only - Print statements to stdout
- Training history plots saved to `checkpoints_phase2c/training_history.png`
- No centralized logging framework

## CI/CD & Deployment

**Hosting:**
- Not applicable - Research/academic code, no deployment pipeline
- Execution: Local Jupyter notebooks or Python scripts

**CI Pipeline:**
- None - No automated testing or deployment configured

## Environment Configuration

**Required env vars:**
- None - All configuration is hardcoded in notebooks or passed as function arguments

**Secrets location:**
- No secrets/credentials required
- `.env` file: Not present, not needed
- All access is to local files and local quantum simulator

## Webhooks & Callbacks

**Incoming:**
- None - Not applicable for notebook-based research code

**Outgoing:**
- None - No external API calls or event notifications

## Data Pipeline

**Input:**
- CSV files: `data.csv` (time series training data)
- Format: Single column univariate time series
- Loading: `pd.read_csv()` in notebooks

**Processing:**
- Lambert W transformation (via `scipy.special.lambertw`)
- Normalization and preprocessing
- Batching via `torch.utils.data.DataLoader`

**Output:**
- Generated samples: `real.csv`, `fake.csv` (synthetic data)
- Model checkpoints: `*.pt` files in `checkpoints_phase2c/`
- Evaluation metrics: Console output and visualization plots

## Checkpoint & Model Management

**Storage:**
- Location: `checkpoints_phase2c/` directory
- Format: PyTorch checkpoint files (`.pt`)
- Tracked metadata:
  - `epoch` - Training epoch number
  - `score` - Discriminator loss
  - `best_score`, `best_epoch` - Early stopping tracking
  - `model_state_dict` - Model parameters
  - `params_pqc` - Quantum circuit parameters
  - `discriminator_state` - Classical discriminator weights
  - `c_optimizer`, `g_optimizer` - Optimizer states
  - `timestamp` - Checkpoint creation time

**Loading:**
- Script: `load_checkpoint_phase2c.py` - Restores best model from checkpoint
- Method: `torch.load()` with parameter restoration logic

## Quantum Backend Configuration

**Simulator:**
- PennyLane Lightning (default simulator)
- Mode: CPU-only quantum simulation
- No hardware quantum device integration (local simulation only)

---

*Integration audit: 2026-02-26*
